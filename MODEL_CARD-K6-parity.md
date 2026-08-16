---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - exl3
  - exllamav3
  - trellis
  - mixed-precision
  - quantized
  - qwen3.8
  - gilded-gnosis
---

# At equal file bytes, this EXL3 recipe matches GGUF `Q6_K` — the 6-bit loss was a byte gap, not an engineering one

**The outcome, plainly, in the frame that was fixed before the number existed.** On shard 0 of
the v5 held-out suite — 512 contexts, 1,048,064 scored positions, 330 source clusters, one
shared BF16 head, one shared BF16 reference — this build measures **0.001634** mean KL
divergence, 95 % CI [0.001541, 0.001742]. GGUF `Q6_K` measures **0.002035** on the identical contexts and
**~0.001528** net of the measured cross-engine floor. The pre-registered prediction was
**0.001488**, with a registered interval of **[0.001175, 0.001601]** spanned by three
independent estimators, and the pre-registered acceptance question was whether this build
lands inside [0.001528, 0.002035] or beats it. It lands between the net and the measured `Q6_K` figures, so the answer to the pre-registered acceptance question is MATCHES: at equal file bytes the two are indistinguishable once the cross-engine floor's own slack is admitted, which is what the byte-gap decomposition predicted. The 6-bit-class loss recorded in `receipts/cross-engine-comparator.json` was a byte gap and not an engineering one, and this build reaches that result carrying about 2.3 GiB less transformer body than `Q6_K`.

**The axis, in the claim rather than in a footnote.** This is parity on **file** bytes, and it
is not parity on the weights that do the multiplying. Counting transformer body only — payload
minus embedding, output head, vision tower and MTP draft, which is the set of weights a
text-only GGUF's body covers — `Q6_K` carries **19.3599 GiB against this build's
17.054 GiB**, a deficit of **2.306 GiB, 13.5 % of ours**. Our bytes go
elsewhere: a BF16 embedding at 2.3682 GiB where theirs is Q6_K at 0.9713, a 0.8582 GiB BF16
vision tower that their text file does not contain at all (it ships separately as
`mmproj-BF16.gguf`), and a 0.2847 GiB MTP draft they have no equivalent of. In the other
direction, on everything you must download to serve the advertised capability, `Q6_K` plus its
`mmproj` is **22.18 GiB against hydrated's 20.13** — our artifact is 2.05 GiB smaller and
multimodal-complete. All of it is measured from the artifacts themselves in
[`receipts/cross-candidate-byte-accounting.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/cross-candidate-byte-accounting.json),
and it was recorded before this conversion ran. It cuts both ways, which is why it is up here.

## The question this build was made to answer

Across ten candidates on our own held-out suite, the one comparison we lost was the 6-bit
one: GGUF `Q6_K` measures **0.002035** mean KL against a BF16 reference on the same 512
contexts where our hydrated K5/K6 build measures **0.002700**, and net of the measured
cross-engine floor of 0.000507 the GGUF figure is about **0.001528**
([`receipts/cross-engine-comparator.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/cross-engine-comparator.json)).
That loss is roughly 1.77x net.

It is also a **byte** gap. `Q6_K` serializes to 21.31 GiB against hydrated's 20.10 GiB of
payload — **+1.183 GiB** over about 25.6 B quantized weights, i.e. **+0.397 bits per
weight**. Our own per-module error law says each further bit divides quantization error by
about **3.73x** ([docs/37](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docs/37-error-driven-allocation.md)),
and `3.73^0.397 = 1.69x`. A 1.69x-predicted advantage against a 1.77x observed one means
`Q6_K` sits on our own scaling curve, within the slack of the floor estimate. Nothing about
the format was beating us; it was spending more bytes.

**So the test is to spend the bytes.** This checkpoint is the hydrated recipe with exactly
one change — MLP `gate_proj` and `up_proj` promoted K5 → K6 — which buys back
**1.349 GiB** and lands at **21.453 GiB** of payload against
`Q6_K`'s 21.31 GiB. Equal bytes, same suite, same reference, same shared head.

## What was built

| role | hydrated (the incumbent) | **this build** |
|---|---|---|
| `full_attention` q/k/v/o, 16 L | K6, serialized and calibrated | unchanged |
| `linear_attention`, 48 L | K6, serialized and calibrated | unchanged |
| `mlp_gate_proj`, 64 L | K5 | **K6** |
| `mlp_up_proj`, 64 L | K5 | **K6** |
| `mlp_down_proj`, 64 L | K6 | unchanged |
| `lm_head` | K6, `mcg` | unchanged |
| MTP draft | `self_attn` K6, mlp K5/K5/K6, `eh_proj` K4 | mlp **K6/K6/K6**, rest unchanged |
| `embed_tokens` | BF16 | unchanged |
| vision tower | BF16 | unchanged |
| body modules moved | — | **128 of 400**, plus the draft's two |

The draft's `gate_proj` and `up_proj` move with the body because `EXL3_BITS_OVERRIDE` is
matched against every module key the allocator holds, and the draft's projections are among
them — the same behaviour the published hydrated build has, and it is priced into the byte
prediction rather than discovered afterwards
([`receipts/byte-law-recipe-audit.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/byte-law-recipe-audit.json)).

## Bytes

| | predicted before converting | measured |
|---|---:|---:|
| tensor payload | 23,035,310,148 B | 23,035,310,148 B |
| payload, GiB | 21.453 | 21.453 |
| whole tree on disk | — | 23,059,333,816 B |
| GGUF `Q6_K` file, for comparison | — | 21.313 GiB |
| GGUF `Q6_K` tensors only | — | 21.3025 GiB |
| — of which transformer body | — | 19.3599 GiB |
| this build's transformer body and draft | 18.196 GiB | 17.054 GiB |

The prediction is the published affine byte law — `bytes(role, K) = fixed(role) +
params(role)·K/8` — applied to the one width change, and it was committed before the
conversion ran. Promoting one bit across `gate_proj` and `up_proj` costs exactly
`params/8` per role, 713,031,680 B each, plus 22,282,240 B for the draft's two.

## Fidelity, measured

Shard 0 of the v5 held-out suite: **512 contexts, 1,048,064 scored positions, 330 source
clusters**, one shared BF16 `lm_head`, one shared BF16 reference capture, cluster-bootstrap
intervals over 10,000 resamples. Every comparison below is a **paired per-context**
difference, not a difference of two aggregates.

| candidate | mean KL | 95 % CI | p99.9 | max | top-1 | payload |
|---|---:|---:|---:|---:|---:|---:|
| **this build** | **0.001634** | [0.001541, 0.001742] | 0.068855 | 2.125406 | 98.248 % | 21.453 GiB |
| hydrated K5/K6 | 0.002700 | [0.002517, 0.002912] | 0.131263 | 3.734847 | 97.797 % | 20.104 GiB |
| GGUF `Q6_K` (llama.cpp) | 0.002035 | [0.001939, 0.002145] | — | — | 97.980 % | 21.313 GiB |
| GGUF `Q6_K`, net of the engine floor | ~0.001528 | — | — | — | — | 21.313 GiB |

**Against hydrated**, same engine and same reference capture, no cross-engine term:
this build is **0.001066 better than** hydrated, paired per context: -0.001066 [-0.001173, -0.000974] over 512 contexts and 330 source clusters, winning 511 of 512.

**Against `Q6_K`**: this build is **0.000401 better than** GGUF `Q6_K` as measured, paired per context: -0.000401 [-0.000425, -0.000378] over 512 contexts and 330 source clusters, winning 493 of 512. This pairing is across engines and must be read as such —
the GGUF candidate was captured in llama.cpp while the reference and this build were captured
in vLLM, so a GGUF number contains engine numerics on top of quantization error. That term is
itself measured, at 0.000507 mean
([`receipts/gguf-report-engine-floor.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gguf-report-engine-floor.json)),
which makes the GGUF figure an upper bound and the net figure a *naive* subtraction: KL is not
additive, so the net column is an estimate and not an identity.

## It was pre-registered

The prediction, the derivation, the exact width map, the exact command and the numeric
decision rule were committed and pushed **before** the conversion ran, in
[`receipts/preregistration-kld9-window.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/preregistration-kld9-window.json):

* registered primary **0.001488**, from the 3.73x-per-bit law charged at this build's own
  byte spend (+0.4526 bpw → 1.81x off hydrated's 0.002700);
* registered interval **[0.001175, 0.001601]**, spanned by three independent estimators — the
  same law at `Q6_K`'s +0.397 bpw surplus (0.001601, docs/29's headline figure), a role-share
  bound from the EDA ladder (0.001175, dividing gate+up's 77.1 % share of the
  sqrt-energy-weighted proxy error by the measured MLP K5→K6 rung ratio of 3.7390), and the
  sqrt-energy surrogate scaled by the two uniform role-group moves the published calibration
  fitted;
* the acceptance question, unchanged since registration: does this build land within
  [`Q6_K` net 0.001528, `Q6_K` measured 0.002035], or beat it?

Measured **0.001634**: the registered primary was 0.001488, so the measurement is 1.10x the prediction and outside the registered interval [0.001175, 0.001601]; it sits between `Q6_K`'s net-of-floor 0.001528 and its measured 0.002035. The body deficit against `Q6_K` at this file size is 2.306 GiB (13.5 % of ours), recorded before the conversion ran.

A uniform role-group promotion is the one prediction class the EDA surrogate calibration
found sign-correct; the between-role reallocation at a fixed budget is the class it failed,
and that failure is published too
([`malaiwah/Qwen3.8-27B-EXL3-EDA-research`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-EDA-research)).
Believing the hits requires publishing the misses.

## What this does not settle

* **Text-only, teacher-forced fidelity.** No generation quality, no long-context retrieval, no
  multimodal request is measured by this number.
* **One shard.** 512 of the suite's 5,120 contexts. The intervals are honest about that.
* **Serving cost is not fidelity.** 21.453 GiB of payload is larger than hydrated's
  20.10, and on a 32 GB card those bytes come out of the KV budget. If you are choosing a
  build for context length rather than for closeness to BF16, the context edition remains the
  right one.
* **The byte comparison is file-to-file, and the composition differs on four axes at once**:
  whole file against tensor payload, text-only against multimodal, non-uniform embedding and
  head widths per GGUF tier, and body against body. On the body axis `Q6_K` carries 2.306 GiB more than this build, whose body lands within 0.02 GiB of five-bit `UD-Q5_K_XL`'s.
  With the int8 embedding overlay on, the embedding half of that asymmetry would narrow to
  about 0.21 GiB resident — but the fidelity protocol runs no overlay, so the scored artifact
  is the BF16-embedding one.
* **The GGUF comparison stays cross-engine.** Measuring `Q6_K` inside vLLM would remove the
  floor term; nobody has done that.

## Reproducing it

```bash
# conversion (exllamav3 v1.4.2 at 5f3c537, worktree diff 578066cd...)
export EXL3_BITS_FIXED='{"^.*self_attn\\..*$": 6, "^.*linear_attn\\..*$": 6}'
export EXL3_BITS_OVERRIDE='{"^.*mlp\\.(gate|up|down)_proj$": 6}'
python convert.py -i Qwen3.8-27B -o qwen38-k6parity -w wd \
  -b 4 -hb 6 -mb 4 -vb 16 -cb mcg
python util/add_safetensors_index.py -m qwen38-k6parity --force
python util/add_quant_config.py -m qwen38-k6parity

# scoring, identical to every other candidate on this suite
python tools/fidelity.py capture --model qwen38-k6parity --suite shard-0000/suite \
  --out hidden-k6parity --quantization exl3 --quantization-config "$(cat qcfg.json)"
python tools/fidelity.py replay --reference hidden-bf16 --candidate hidden-k6parity \
  --head lm_head.safetensors --suite shard-0000/suite --out report-k6parity.json
python tools/fidelity.py paired --a report-hyd.json --b report-k6parity.json \
  --a-label hyd --b-label k6parity --bootstrap-samples 10000 --out paired.json
```

The calibration corpus is exllamav3's shipped default, unchanged — 250 rows of 2,048 tokens,
211 from text and 39 seeded random, whose exact token rows digest to
`2b30349958715e3d3ba069a21a57a83160fde96225fd6cae26a23b240921d201` in feed order.

## Provenance

| | |
|---|---|
| base model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| converter | `turboderp-org/exllamav3` @ `5f3c537` (v1.4.2), worktree diff `578066cd...` |
| conversion log | `parity/convert-k6parity.log`, 1,054 lines, every module's realised bpw and proxy error |
| receipt | [`receipts/k6-parity-kld.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/k6-parity-kld.json) |
| pre-registration | [`receipts/preregistration-kld9-window.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/preregistration-kld9-window.json) |
| digests | `SHA256SUMS` (payload), `DOCS-SHA256SUMS` (documentation), `build-receipt.json` |

Every number on this card is reproducible from the receipt, and the receipt's own
`content_sha256` covers it.
