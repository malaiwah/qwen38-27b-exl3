# 38. The KV-dtype lever on the physical RTX 5090: decision record

**Decision: keep `fp8`. It is confirmed as the right default for the context edition's
native-window profile, and the alternatives are now measured rather than assumed.**

[docs/29](29-plan-and-loose-ends.md) §F5 carried this as an open item: every
published **serving/capacity** profile used `fp8`, while the text-only fidelity
captures resolved KV to bfloat16, and no 4-bit serving arm had been measured for
capacity, retrieval, or KV-specific fidelity. A release-thread reader asked
whether quantized KV destroys long-context quality. The bounded answers follow.

Everything was measured on the user's own AIBoss host, one physical **GeForce RTX 5090**
(32,607 MiB, driver 610.57.04, `GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a`), on the promoted
four-module release image `localhost/vllm:gg-r34-patched-apc`
(manifest `sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218`) with **no
source bind mounts**. The three qualified patch modules inside that image were digest-verified
against `identity.patch_files_sha256` in
[`receipts/qualification-5090-context.json`](../receipts/qualification-5090-context.json) before
any GPU work, so the baseline arm here is comparable with the published qualification. Nothing
here was measured on the rental RTX PRO 6000 and **no number here is comparable to one**. Full
rows, all 43 launch argvs, every startup banner and every verbatim refusal:
[`receipts/kv-dtype-sweep-5090.json`](../receipts/kv-dtype-sweep-5090.json).

## 1. The baseline reproduces, which is what licenses the rest

The `fp8` arm was re-measured rather than cited. At the qualified profile it printed an engine
budget of **29.98 GiB**, usage of **18.19 / 1.78 / 0.27 / 0.45 GiB**, an available KV pool of
**9.28 GiB**, **265,122** KV tokens, **1.01x** concurrency at 262,144 and an attention block size
of **1600** tokens — every figure identical to
[`receipts/qualification-5090-context.json`](../receipts/qualification-5090-context.json) gate 1,
which was measured on the three-module image. Median decode came out at **107.36 tok/s** against
that receipt's 107.56. The promoted four-module image is capacity- and throughput-identical to
the qualified one at this profile.

## 2. What the flag list actually accepts

Seventeen dtypes, one startup attempt each at `--max-model-len 8192`, utilisation 0.85, otherwise
the qualified profile. **Eight start, nine refuse.**

| dtype | starts | what happens |
|---|---|---|
| `auto`, `bfloat16` | yes | FLASH_ATTN, block 800 |
| `fp8`, `fp8_e4m3`, `fp8_e5m2` | yes | FLASHINFER, block 1600. `fp8` and `fp8_e4m3` are the same thing |
| `int4_per_token_head` | yes | TRITON_ATTN, block 3104 |
| `int8_per_token_head`, `fp8_per_token_head` | yes | TRITON_ATTN, block 1584 |
| `float16` | **no** | `RuntimeError: query and key must have the same dtype` |
| `fp8_inc`, `fp8_ds_mla`, `nvfp4_ds_mla`, `nvfp4` | **no** | no attention backend accepts the dtype |
| `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc` | **no** | `ValueError: Unknown TurboQuant cache dtype: 'auto'` |

Two of those refusals are worth their own paragraph, because both retire a plan item.

**The whole TurboQuant family is unreachable, and it is a build defect.** All four presets load
the model successfully — 100 to 114 seconds each — and then fail during KV cache creation with
`Unknown TurboQuant cache dtype: 'auto'`. The TURBOQUANT backend *accepts* these dtypes at
selection time; the failure is downstream, in `vllm/v1/worker/gpu_model_runner.py`, where the
per-layer string is computed as
`"auto" if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE else self.cache_config.cache_dtype`.
The TurboQuant presets are not mapped into `KVQuantMode`, so the spec reports `NONE`, the literal
string `auto` is forwarded to the backend, and `TurboQuantConfig.from_cache_dtype('auto')` raises.
Two halves of the same build disagree about whether these dtypes are quantized. §F5 named
`turboquant_4bit_nc` as a sweep arm; **that arm cannot exist in this image**, and a `KVQuantMode`
mapping for the four presets is the whole fix. We did not patch it, so whether TurboQuant would
be fast or accurate here is unknown.

**`nvfp4` does not start, and the error names the missing piece precisely.** Verbatim:

```
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(
  head_size=256, dtype=torch.bfloat16, kv_cache_dtype=nvfp4, block_size=None,
  use_mla=False, has_sink=False, use_sparse=False, use_mm_prefix=False,
  use_per_head_quant_scales=False, attn_type=AttentionType.DECODER, ...).
Reasons: {FLASH_ATTN: [kv_cache_dtype not supported],
          FLASHINFER: [kv_cache_dtype not supported],
          TRITON_ATTN: [kv_cache_dtype not supported],
          FLEX_ATTENTION: [kv_cache_dtype not supported],
          TURBOQUANT: [kv_cache_dtype not supported]}.
```

Not `head_size` 256, not sm120, not the Gated DeltaNet hybrid, not the exl3 weight path: all five
candidate backends reject on the KV dtype alone, in backend selection, before a single KV byte is
sized. Replaying the build's own `CudaPlatform.get_valid_backends` off-GPU over a matrix of shapes
finds no valid backend for `nvfp4` or `nvfp4_ds_mla` anywhere, **including `use_mla=True` at
`head_size` 576**, where the MLA backends answer the same thing. That probe is validated: it
reproduces the observed backend choice for every dtype that actually started. So a follow-up build
task needs exactly one thing — an attention backend on this fork advertising `nvfp4` for a
non-MLA decoder. The owner reports that this fork serves GLM-5.2 with an nvfp4 KV cache; **that
precedent is the owner's and is unverified by us** — no GLM run, configuration or log was
inspected for this work.

## 3. Capacity, and a confounder that has to be stated first

Five arms at the qualified profile, `--max-model-len 262144` for four of them. **`bfloat16`
cannot hold the native window**: at utilisation 0.955 it refuses with
`(17.64 GiB KV cache is needed, which is larger than the available KV cache memory (9.67 GiB))`
and names its own ceiling, `the estimated maximum model length is 139200`. It therefore ran at
131,072, the largest power of two below that, where it allocated 138,519 tokens — consistent with
the 139,200 the engine predicted from the other side of the refusal.

| arm | KV pool | KV tokens | concurrency | block | CUDA-graph pool | decode tok/s | MTP accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`fp8`** (shipped) | 9.28 GiB | **265,122** | 1.01x | 1600 | 0.45 GiB | **107.36** | 0.5649 |
| `int8_per_token_head` | 9.67 | **272,453** | 1.04x | 1584 | 0.06 | 108.67 | 0.5590 |
| `fp8_per_token_head` | 9.67 | 272,453 | 1.04x | 1584 | 0.06 | 97.60 | 0.4717 |
| `int4_per_token_head` | 9.67 | **502,667** | **1.92x** | 3104 | 0.06 | 107.86 | 0.5745 |
| `bfloat16` @131,072 | 9.67 | 138,519 | 1.06x | 800 | 0.06 | 109.44 | 0.5684 |

**The KV dtype is the only flag changed, but it is not the only thing that changes.** The engine
derives the attention backend and the block size from it, and TRITON_ATTN is the *only* backend
that accepts the per-token-head schemes. So the 0.39 GiB of extra pool the four non-`fp8` arms
enjoy is **not a KV-byte effect**: those arms cost *more* per token than `fp8`, and their pool
advantage is entirely a CUDA-graph pool of 0.06 GiB against `fp8`'s 0.45. Read the table as
"the arm as the engine actually serves it", never as "this is what the bytes cost".

And the price of that backend is large:

| arm | 261,795-token prefill | vs `fp8` | 98,313-token prefill | vs `fp8` |
|---|---:|---:|---:|---:|
| `fp8` | **180.4 s** | 1.00x | **44.4 s** | 1.00x |
| `int4_per_token_head` | 501.0 s | 2.78x | 90.0 s | 2.03x |
| `int8_per_token_head` | 544.3 s | 3.02x | 95.3 s | 2.15x |
| `fp8_per_token_head` | 545.9 s | 3.03x | 95.1 s | 2.14x |
| `bfloat16` (130,728 tok) | 63.2 s | — | 43.5 s | 0.98x |

Decode is untouched — 107 to 109 tok/s everywhere except `fp8_per_token_head`, which loses 9.1 %
because its MTP acceptance falls to 0.4717. **Prefill is where the per-token-head family is
paid for**, and on a native-context profile prefill is the user-visible cost: a three-minute
prompt becomes a nine-minute prompt.

## 4. Retrieval is intact everywhere, and that is not the fidelity answer

Exact-match needle retrieval, deterministic prompts, `temperature 0`:

| arm | at the longest length that fits | at 98,313 tokens |
|---|---|---|
| `fp8` | **5/5** at 261,795 (depths 0.05/0.25/0.50/0.75/0.95) | **5/5** |
| `int4_per_token_head` | **5/5** at 261,795 (same five depths) | **5/5** |
| `int8_per_token_head` | **3/3** at 261,795 (0.05/0.50/0.95) | **5/5** |
| `fp8_per_token_head` | **3/3** at 261,795 (0.05/0.50/0.95) | **5/5** |
| `bfloat16` | **3/3** at 130,728 (0.05/0.50/0.95) | **5/5** |

Forty-four retrievals, forty-four exact. The long-length set was cut from five depths to three
for the last three arms because each of those prefills costs nine minutes; the three are a subset
of the five, so the long-length comparison across arms is on the shared depths.

**A retrieval pass is not a fidelity claim.** Every arm above, including 4-bit, finds a
ten-digit code at every depth. §5 shows those same arms are not producing the same text.

## 5. Fidelity: what a bfloat16 KV cache would have said

The reference is an arm, not an argument: `bfloat16` KV is the unquantized cache, so the question
"does quantized KV cost anything" is answered by grading each arm against it on identical bytes.
Four deterministic prompts at 98,266–98,344 tokens, byte-identical across arms and sha256-verified,
each continued greedily for 64 tokens with top-20 logprobs at every position. A position counts
only while both arms conditioned on an identical prefix.

| arm vs `bfloat16` KV | top-1 agreement | mean truncated KL | mean abs Δlogprob | identical 64-token continuations | paired positions |
|---|---:|---:|---:|---:|---:|
| `fp8_per_token_head` | **0.9884** | 0.001284 | 0.0241 | **2/4** | 173 |
| `int8_per_token_head` | 0.9725 | **0.000914** | **0.0170** | 1/4 | 109 |
| **`fp8`** (shipped) | 0.9560 | 0.001655 | 0.0262 | 0/4 | 91 |
| `int4_per_token_head` | 0.9429 | **0.005948** | 0.0512 | 0/4 | 70 |

Three things follow.

**The shipped default is not catastrophic on this probe.** Across four
98k-context prompts, `fp8` retains 95.60 % top-1 against bfloat16 KV with
0.001655 truncated top-20 KL. That rejects the claim that quantized KV destroys
these continuations; it is not a broad long-context quality result.

**4-bit costs 3.6× the shipped default on the same diagnostic while every needle
still passes**: 0.005948 versus 0.001655. Needle retrieval alone would have
missed the difference.

**The per-token-head 8-bit arms improve the diagnostic but not the whole serving
frontier.** `int8_per_token_head` is closer to bfloat16 and keeps decode near
baseline, but costs ~3× prefill. `fp8_per_token_head` is also closer in this
diagnostic, yet loses 9.1 % decode here through lower MTP acceptance and pays the
same prefill penalty.

**Resolution, stated plainly.** These are not v5 KLD and must never be
differenced against the body-fidelity ladder. They are truncated top-20
diagnostics over at most 173 common-prefix positions against a KV-cache
reference. Omitting reference-tail support and flooring missing candidate
logprobs does **not** produce a guaranteed lower or upper KL bound; individual
omitted terms need not share a sign. The floor was used on 1.10–4.36 % of
entries, and paired-position counts differ by arm.

Repeating prompt 0 returned byte-identical tokens and logprobs, establishing
determinism for that control prompt. The arms change KV dtype **and** attention
backend. Four prompts × 64 tokens provide a useful ranking on this diagnostic,
without a confidence interval or a general long-context fidelity claim.

## 6. The affine law, re-derived per dtype

`kv_needed(L) = a·L + M` was measured for each dtype exactly the way
[docs/34](34-vram-class-profiles.md) §10.2 measured it: two startup refusals at one capped
utilisation and two windows, reading the engine's own printed per-request requirement. Twenty
refusals, ten per MTP mode, no adaptive retry needed.

The exact law is not `a·L + M` but `a·⌈L/B⌉·B + M`, and using the engine's own block quantisation
turns the two printed requirements into an **integer** count of charged layers, which pins `a` to
a single value instead of a ±82 B/token interval. In all ten configurations that count lands
within 0.03 of an integer: **17 with MTP-3, 16 with MTP off**, for every dtype. The model has 16
full-attention layers; the seventeenth is the engine's own "Add 3 padding layers, may waste at most
6.25 % KV cache memory" allowance — exactly one sixteenth.

| dtype | MTP-3 `a` (B/token) | MTP-3 `M` | MTP-off `a` | MTP-off `M` | block, MTP-3 |
|---|---:|---:|---:|---:|---:|
| `bfloat16` | 69,632 | 0.62 GiB | 65,536 | 0.14 GiB | 800 |
| `fp8` | **34,816** | **0.63** | 32,768 | 0.14 | 1600 |
| `int8_per_token_head` | 35,360 | 0.63 | 33,280 | 0.14 | 1584 |
| `fp8_per_token_head` | 35,360 | 0.63 | 33,280 | 0.14 | 1584 |
| `int4_per_token_head` | 17,952 | 0.62 | 16,896 | 0.15 | 3104 |

`a` is the exact form; `M` is quoted from the published form for MTP-3 (`fp8`: 0.63) and rounded
from the exact form elsewhere — the receipt carries both to four decimals for every cell.

**The control is `fp8` itself.** Under MTP-3 the published-form arithmetic returns
`a = 34,816.0 B/token` and `M = 0.63 GiB`, digit for digit the docs/34 §10.2 figures, from an
independent set of refusals on a different image. Inverting the exact law reproduces the engine's
own binary-searched "estimated maximum model length" — 38,400 for `fp8`, 99,328 for
`int4_per_token_head`, both to the token, both a whole number of blocks.

Two structural facts fall out, and they matter more than the table.

**`M` is essentially dtype-independent**: 0.619–0.631 GiB under MTP-3 and 0.140–0.148 GiB without,
across a 3.9x range of per-token cost. The fixed per-request term does not shrink when the KV
cache does — which is what one expects if it is dominated by Gated DeltaNet recurrent state and
draft bookkeeping rather than by attention KV.

**4-bit is not half.** `int4_per_token_head` costs 17,952 B/token against `fp8`'s 34,816 — 51.6 %,
not 50 %. The last dimension halves as expected and then the dynamic per-token-head scales add
32 B per token per layer, two float32 values per KV head, giving 3.1 % back.

*One correction offered and not taken.* Under MTP off the exact form gives `a = 32,768 B/token`
exactly — 16 layers, no padding sixteenth — against the published **32,932**, which was fitted
with the unquantised divisor and carries ±82 B/token. docs/34 already described that figure as
"1.005 × the fp8 floor"; the exact form says the 0.5 % was rounding and it *is* the floor. It is
**not** applied here: it moves an `fp8` baseline four documents cite, it was measured on a 5090 at
utilisation 0.72 rather than on the 24 GiB proxy the published figure came from, and it changes no
published class length.

## 7. What this does to the 24 GB and 16 GB classes

Rule unchanged and in its **memory** form — largest multiple of 4,096 with
`a·L + M ≤ pool/1.15`. Pools are the `fp8`-derived figures from docs/34 and
`receipts/qualification-24gib-capped.json`, **held**, with only `a` and `M` varying by dtype.
Feeding the published `fp8` law through the same function returns 24,576 at 25.4 % headroom and
28,672 at 27.3 % — the two published figures, exactly — which is the control on the arithmetic.

| dtype | 24 GB, MTP-3 (pool 1.79 GiB) | 16 GB, MTP off (pool 1.298 GiB) |
|---|---:|---:|
| **`fp8`** — published | **24,576** (1.4269 GiB, 25.4 %) | **28,672** (1.0194 GiB, 27.3 %) |
| `int8_per_token_head` | 24,576 (1.4419, 24.1 %) | 28,672 (1.0738, 20.9 %) |
| `fp8_per_token_head` | 24,576 (1.4419, 24.1 %) | 28,672 (1.0738, 20.9 %) |
| `int4_per_token_head` | **53,248** (1.5556, 15.1 %) | **57,344** (1.0912, 18.9 %) |
| `bfloat16` | 12,288 (1.4178, 26.3 %) | 12,288 (0.9391, 38.2 %) |

**Exactly one arm would change the published class figures, and it roughly doubles both.**
`int4_per_token_head` takes 24 GB MTP-3 from 24,576 to **53,248** and 16 GB MTP-off from 28,672 to
**57,344**. The 24 GB row lands at 15.1 % headroom, on the edge of the rule. The two 8-bit
per-token-head schemes change **neither** figure — they cost marginally more per token than `fp8`,
so they land on the same lengths with thinner headroom. `bfloat16` halves both.

**The withdrawn 16 GB MTP-3 row stays withdrawn for every dtype.** Its pool is 0.553 GiB, the rule
allows 0.4809 GiB, and the smallest fixed term any dtype achieves is **0.6188 GiB**. A *free* KV
cache would not start it. That closes the question of whether a cheaper KV dtype rescues that row:
it does not, because the term that blocks it is not KV.

**Every window in this section is a prediction and none has been started.** No 24 GB or 16 GB proxy
ran in this sweep. Worse, the pool itself moved with dtype on the 5090 — 9.28 GiB for `fp8` against
9.67 for everything else — so these predictions inherit an `fp8`-shaped pool the other dtypes may
not have on a real board. A class-accurate answer re-measures the pool per dtype on a ballasted
proxy.

## 8. Verdict

**No arm beats `fp8` on the frontier we care about — native-or-beyond context on 32 GB with
retrieval intact.** `fp8` already sits on that frontier at 265,122 tokens with 5/5 needles exact
at 261,795. Two arms dominate it on capacity *and* fidelity at once — `int8_per_token_head` and
`fp8_per_token_head` both allocate 272,453 tokens and both sit closer to the bfloat16 reference —
and both pay **3.0x prefill**, 544 s against 180 s on the same 261,795-token prompt. That is what
it costs, and for a native-context serving profile it is not a trade worth 7,331 tokens and a
fidelity gain no retrieval or capability gate can see. `int4_per_token_head` is a real lever, but
it belongs to the small classes: 1.92x concurrency and roughly double the published class windows,
against 3.6x the distributional error and 2.78x prefill.

So `fp8` is confirmed as the right default, and — the part that was actually missing — the
alternatives are now **measured** rather than assumed.

**What would change the decision.** The per-token-head penalty is confounded with TRITON_ATTN
being the only backend that accepts those schemes. If a FLASHINFER build accepted per-token-head
scales and the 3.0x prefill went away, `int8_per_token_head` would be strictly better than `fp8`
on this card — more tokens, more fidelity, equal decode — and the default should move. That is one
build away, and it is the single most valuable follow-up this sweep produced.
