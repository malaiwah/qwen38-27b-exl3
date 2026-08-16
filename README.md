> **Status 2026-08-16: the collection is hardware-qualified, the comparator set is measured, and
> the method is now reproducible from published artifacts.** The current checkpoint is
> [`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6):
> mean KLD **0.003210** (95 % CI [0.002982, 0.003480]) body-only at **20.32 GiB** resident
> weights, versus official [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
> at **0.005294** — **39 % lower divergence at 71 % of its resident weight**, paired
> **-0.002084** [-0.002249, -0.001942] winning **5,105 of 5,120** held-out contexts. The best
> profile in the collection is the hydrated build at **0.002760**, paired **-0.002534** on
> **5,118/5,120**. The headline suite is v5: **5,120 contexts x 2,047 positions = 10,480,640
> scored positions** over **842 source clusters**, calibration- and v4-token-disjoint.
>
> Landed since: the context edition is **qualified on a physical RTX 5090** (all seven gates at
> `--gpu-memory-utilization 0.955` with `expandable_segments`, 265,122 KV tokens at 262,144 —
> and a second physical 5090 needed 0.956, so utilisation is a per-card measurement, not a
> constant). `unsloth`'s NVFP4 is now **on our suite** at 0.031059 over 10.48 M positions,
> losing 5,120 of 5,120 contexts to both the context edition and FP8. The external
> `llama-perplexity` protocol was **run for cross-citation** and reproduces our ordering.
> Capture and replay are **bit-reproducible**, and a fresh conversion of the same recipe — a
> *sibling*, 97.6 % of quantized modules differing in bytes — scores **indistinguishable** from
> the published checkpoint, so the recipe determines fidelity while the published bytes remain
> the artifact. The suite, the shared BF16 reference capture and every per-shard report are
> published as a [dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5),
> with archival mirrors of the third-party artifacts our numbers cite.
>
> **Two more artifacts published 2026-08-16, both from pre-registered conversions.**
> [`malaiwah/Qwen3.8-27B-EXL3-K6-parity`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K6-parity)
> spends the bytes GGUF `Q6_K` spends on the body - the hydrated recipe with `gate_proj`/`up_proj`
> promoted K5 -> K6 - and **matches it**: mean KLD **0.001634** [0.001541, 0.001742], between
> `Q6_K`'s net-of-floor 0.001528 and its measured 0.002035, beating our hydrated build on 511 of
> 512 contexts (-39.5 % for +1.348 GiB) while carrying **2.31 GiB less transformer body** than
> `Q6_K`. So the 6-bit-class loss was a byte gap, not an engineering one.
> [`malaiwah/Qwen3.8-27B-EXL3-S16-V-research`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-S16-V-research)
> is the opposite: the **rejected** sub-4-bit 16 GB candidate at **0.045374** [0.041959, 0.049351],
> 1.5x its pre-registered NO threshold, losing 512/512 contexts to K4 (4.39x) and to the context
> edition (13.31x). It is published *because* it failed - a NO nobody can audit is an assertion -
> and it is the empirical anchor for the K3 rung of the per-bit law
> ([docs/37](docs/37-error-driven-allocation.md) §3.3). Both payloads were predicted to the **byte**
> by the affine law before conversion, and both registered intervals are reported with their
> misses: S16-V's point estimate was 1.52x too high, K6-parity's interval fell 2.0 % short.
>
> The earlier headline — 0.007945 body-only on 136 v3 contexts / 278,392 positions
> ([docs/22](docs/22-results-iteration-2.md)) — stands exactly as measured and is superseded only
> as *the* headline; v5 absolute KLD is **not** comparable to v3 (the two suites differ by a
> measured 2.46-3.08x with identical ordering), so quote paired differences across suites, never
> means. Figures labelled K4 or v1 belong to iteration 1 and are superseded. One earlier control
> was **withdrawn**: the "CUDA-graph parity 0.000000" receipt captured a *prefill* forward, which
> `FULL_DECODE_ONLY` never captures, so it could not have measured decode; the decode probe that
> replaces it is [docs/27](docs/27-graph-decode-drift-control.md). Open items are tracked in
> [docs/29](docs/29-plan-and-loose-ends.md).

# Qwen3.8-27B EXL3 mixed-precision quants (`K4`, `EXL3-K5K6`)

Research materials and progress log for building a dense EXL3 quant of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) that is inspired by
NVIDIA's NVFP4 recipe for the previous generation, but keeps the attention projections
**BF16 on disk** for **runtime K6 encoding** by the Gilded Gnosis vLLM fork, and
serializes the MLP as EXL3 **K5 gate / K5 up / K6 down** with a K6 `mcg` head and a
quantized MTP draft. Iteration 1 serialized the whole MLP at **K4**.

Design goal: NVFP4-class VRAM footprint, lower KLD. Met on fidelity, memory, decode and
speculative decode; prefill is a measured structural deficit of a 4-bit-class trellis
format in this runtime ([docs/25](docs/25-goal-pareto-dominate-fp8.md),
[docs/26](docs/26-prefill-attribution.md)).

## Evidence at a glance

Held-out **v5** suite, `receipts/kld5-suite-manifest.json` (schema
`qwen38-distribution-fidelity/6`, `suite_token_sha256` `510541f6…09482b88`): **5,120 contexts
x 2,047 positions = 10,480,640 scored positions**, **842 source clusters**, 1,024 contexts per
stratum. 44 of 941 corpus documents were excluded whole *before* window selection for any
all-position 12-token overlap with exllamav3 calibration data, leaving 897 eligible, so
contamination hits are 0 by construction; all 160 v4 context token hashes were seeded as
exclusions and 0 were reachable. Every figure below is **body-only** — both operands scored
through one shared BF16 LM head — with a 95 % CI from a 10,000-resample bootstrap over the 842
source clusters.

| candidate | mean KLD | 95 % CI | top-1 | paired vs FP8 | contexts won | receipt |
|---|---|---|---|---|---|---|
| hydrated | **0.002760** | [0.002540, 0.003020] | 97.70 % | **-0.002534** | **5,118 / 5,120** | `receipts/kld5-10M-hyd.json` |
| online K5/K6 | **0.003210** | [0.002982, 0.003480] | 97.52 % | **-0.002084** | **5,105 / 5,120** | `receipts/kld5-10M-k5k6.json` |
| context | **0.003509** | [0.003220, 0.003852] | 97.44 % | **-0.001785** | **5,109 / 5,120** | `receipts/kld5-10M-ctx.json` |
| official FP8 | 0.005294 | [0.004927, 0.005728] | 96.79 % | — | — | `receipts/kld5-10M-fp8.json` |
| K4 (iteration 1) | 0.010604 | [0.009640, 0.011746] | 95.76 % | +0.005310 | 7 / 5,120 | `receipts/kld5-10M-k4.json` |

Paired intervals and win counts come from `receipts/kld5-10M-paired.json` (10,000 resamples,
seed 1); hydrated also beats the runtime overlay directly, **-0.000450** [-0.000469,
-0.000433] on **4,922/5,120**. Exact worst single position over the whole run: hydrated 8.258,
online 22.241, context 5.557, FP8 10.714, K4 14.283. The cumulative hydrated mean at the
1M / 2M / 5M / 10M ladder checkpoints is **0.002700 / 0.002759 / 0.002699 / 0.002760**
(`shards[]` of `receipts/kld5-10M-hyd.json`).

**The tail, not just the mean.** The ten-shard run predates the histogram, so the tail was
measured by re-running **shard 0** of the same v5 suite with the `qwen38-fidelity-report/2`
harness: **512 contexts x 2,047 positions = 1,048,064 scored positions**, the same contexts
every candidate saw. Quantiles are **bin-bounded** — 560 log-spaced bins from 1e-12 to 1e2
nats, one bin ~5.6 % wide, and every receipt carries `lower` / `upper` / `estimate` per
quantile — while the maxima and the exceedance counts are **exact**. The two columns worth
reading are **p99.9** and the **share of positions above 0.1 nats**.

| candidate | mean | p50 | p95 | p99 | **p99.9** | p99.99 | exact max | **above 0.1** | above 1.0 | receipt |
|---|---|---|---|---|---|---|---|---|---|---|
| hydrated | 0.002700 | 0.00109 | 0.0082 | 0.0276 | **0.1319** | 0.463 | 3.735 | **0.1534 %** | 0.00219 % | `receipts/kld5-1M-tail-hyd.json` |
| online K5/K6 | 0.003141 | 0.00128 | 0.0099 | 0.0321 | **0.1446** | 0.498 | 5.507 | **0.1820 %** | 0.00200 % | `receipts/kld5-1M-tail-k5k6.json` |
| context | 0.003409 | 0.00135 | 0.0107 | 0.0357 | **0.1642** | 0.587 | 3.749 | **0.2287 %** | 0.00305 % | `receipts/kld5-1M-tail-ctx.json` |
| official FP8 | 0.005197 | 0.00202 | 0.0167 | 0.0531 | **0.2438** | 0.812 | 5.296 | **0.3912 %** | 0.00592 % | `receipts/kld5-1M-tail-fp8.json` |
| K4 | 0.010345 | 0.00320 | 0.0332 | 0.1194 | **0.5555** | 1.870 | 7.565 | **1.2604 %** | 0.03807 % | `receipts/kld5-1M-tail-k4.json` |

The ordering at p50, p95, p99, p99.9 and p99.99 is the same as the ordering of the means, so
for these candidates the mean is not hiding a worse tail: every EXL3 K5/K6-class build has a
lighter tail than official FP8 at **every** measured quantile, and K4 is worse than FP8 at
every quantile. Receipts are schema `qwen38-kld-ladder-cumulative/2`, welded by
`tools/kld_aggregate.py` from `/2` replay reports.

**Against GGUF, measured, including the part that goes against us.** The standing critique is
that official FP8 is a throughput format whose quality is Q4-to-Q5 class, so beating it is a weak
claim, and that `Q8_0` and `Q6_K` are the honest bar. That is now measured rather than argued.
Three GGUFs from `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` were
captured under llama.cpp pinned at commit `ece963f41b0b02d7a0d61436ae365762c073a4c8` with
`tools/gguf_capture.cpp`, which reads the post-final-norm state — the same mathematical point our
vLLM hook takes, with bf16 rounding verified bit-identical to torch on 2,012,449 probe values —
and scored through **the same shared BF16 head** on **shard 0 of the v5 suite: the same 512
contexts and the same 1,048,064 positions every candidate above saw** (`tools/gguf_manifest.py`,
`tools/build_llamacpp.sh`; receipt `receipts/cross-engine-comparator.json`, per-candidate reports
`receipts/gguf-report-{q8_0,q6_k,q5_k_xl}.json`).

| candidate | engine | measured mean KLD | net of engine floor | top-1 | p99.9 | serialized |
|---|---|---:|---:|---:|---:|---:|
| GGUF `Q8_0` | llama.cpp | 0.001087 | ~0.000579 | 98.53 % | 0.0351 | 27.05 GiB |
| GGUF `Q6_K` | llama.cpp | 0.002035 | ~0.001528 | 97.98 % | 0.0794 | 21.31 GiB |
| hydrated EXL3 | vLLM | **0.002700** | n/a, same engine | 97.80 % | 0.1313 | 20.12 GiB payload |
| online K5/K6 EXL3 | vLLM | **0.003141** | n/a, same engine | 97.61 % | 0.1447 | — |
| context EXL3 | vLLM | **0.003409** | n/a, same engine | 97.55 % | 0.1632 | 19.27 GiB payload |
| GGUF `UD-Q5_K_XL` | llama.cpp | 0.004444 | ~0.003936 | 97.20 % | 0.2144 | 18.83 GiB |
| official FP8 | vLLM | 0.005197 | n/a, same engine | 96.92 % | 0.2440 | 28.51 GiB resident |
| K4 EXL3 | vLLM | 0.010345 | n/a, same engine | 95.91 % | 0.5576 | — |
| `unsloth/Qwen3.8-27B-NVFP4` | vLLM | 0.030115 | n/a, same engine | 93.16 % | 1.6228 | 22.57 GB checkpoint |

The **cross-engine floor** is measured the same way, not assumed: the unquantized BF16 GGUF
against the vLLM BF16 reference, identical tokens, identical shared head, the same 512 contexts,
reads **0.000507** mean, 99.07 % top-1, p99.9 0.0113
(`receipts/gguf-report-engine-floor.json`). Every GGUF row carries that term; no vLLM row does.
KL is not additive, so the net column is an estimate and not an identity — the measured GGUF value
is an upper bound, the net figure the naive lower one. The two `—` cells are the builds that ship
BF16 attention for the runtime to encode at load, so their disk bytes are not a like-for-like
payload; payload figures are `immutable_payload_bytes` from `receipts/collection-index.json`
(hydrated 21,610,916,123 B = 20.127 GiB, context 20,696,033,532 B = 19.275 GiB; the table truncates
both to two decimals) and are serialized bytes, never VRAM. The
FP8 figure is resident weights and is labelled as such.

The p99.9 column is each report's **exact** shard-0 p99.9 as the comparator receipt read them; the
tail table above quotes the **bin-bounded cumulative estimate** from the 560-bin histogram (bins
about 5.6 % wide), which is why hydrated reads 0.1319 there and 0.1313 here — each exact value lies
inside the bin its estimate names. The two differ by construction, not by measurement.

What that table says, in the order that matters:

- **At the 6-bit operating point GGUF `Q6_K` is genuinely better than our best build** — 0.001528
  net at 21.31 GiB against hydrated's 0.002700 at 20.12 GiB of payload. We lose that comparison
  on our own suite, and it is the first measurement here where an off-the-shelf artifact beats
  the recipe.
- **At the 5-bit operating point our context edition wins** — 0.003409 at 19.27 GiB against
  `UD-Q5_K_XL`'s 0.003936 net at 18.83 GiB, about 13 % better fidelity for **0.445 GiB** more
  payload.

**Correction, 2026-08-16 — the byte axis in the table above is not one axis, and every mixed
comparison flattered us.** A GGUF row is the **whole file** of a **text-only** artifact; our row is
**tensor payload** of a **multimodal** tree that also carries an MTP draft. Read from each artifact's
own tensor table, without downloading any payload
([`cross-candidate-byte-accounting.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/cross-candidate-byte-accounting.json)):

| candidate | file | tensor total | `token_embd` | `output` | transformer body | multimodal deployed |
|---|---:|---:|---:|---:|---:|---:|
| GGUF `Q8_0` | 27.05 | 27.04 | 1.258 (Q8_0) | 1.258 (Q8_0) | **24.526** | 27.92 (+ `mmproj-BF16` 0.867) |
| GGUF `Q6_K` | 21.31 | 21.30 | 0.971 (Q6_K) | 0.971 (Q6_K) | **19.360** | 22.18 (+ `mmproj-BF16` 0.867) |
| GGUF `UD-Q5_K_XL` | 18.83 | 18.82 | 0.814 (Q5_K) | 0.971 (Q6_K) | **17.034** | 19.70 (+ `mmproj-BF16` 0.867) |
| online K5/K6 | 28.50 | 28.47 | 2.368 (BF16) | 0.889 (K6) | **24.119** | 28.50, vision inside |
| K4 | 26.37 | 26.37 | 2.368 (BF16) | 0.593 (K4) | **21.463** | 26.37, vision inside |
| hydrated K5/K6 | 20.13 | 20.10 | 2.368 (BF16) | 0.889 (K6) | **15.726** | 20.13, vision inside |
| context edition | 19.27 | 19.25 | 2.368 (BF16) | 0.889 (K6) | **14.886** | 19.27, vision inside |

All figures GiB. The embedding and head widths are **not** uniform across GGUF tiers, and the vision
encoder is **absent from every GGUF text file** - it ships separately as `mmproj-BF16.gguf`, which no
earlier comparison of ours counted. What that does to the four published claims, two against us and
two for us:

1. **6 bits, `Q6_K` against hydrated: our sentence understated their byte spend roughly threefold.**
   "+1.186 GiB more file" is +1.198 on tensors and **+3.634 GiB of transformer body (23.1 % more than
   ours)**. Their fidelity win at 6 bits stands exactly as published - it is the *price* we
   mis-stated, in our own favour.
2. **6 bits, deployed: a multimodal `Q6_K` deployment is 22.18 GiB against our 20.13 GiB whole tree,
   so ours is 2.053 GiB smaller** and needs no second file.
3. **5 bits, `UD-Q5_K_XL` against the context edition: this claim was wrong against us.** We do not
   pay "0.445 GiB more" for the win - on transformer body **they** carry 2.148 GiB more (+14.4 %),
   and our deployed multimodal artifact is 0.422 GiB **smaller**.
4. **8 bits, `Q8_0` against online K5/K6: the bodies agree to 1.7 %** (24.526 against 24.119), the
   most format-comparable pair on the table, and **they win it cleanly**. That row is the reason the
   others are worth reading: this is not a table where every axis favours the author.

**Rule from here on, and it is printed rather than footnoted:** "at equal file bytes", "at equal
tensor bytes", "at equal transformer body" and "as a deployed multimodal artifact" are four different
claims, and whichever one a sentence means is written into the sentence. No fidelity number changes.
- **`Q8_0` is the fidelity leader**, 0.001087 at 27.05 GiB, and only about twice the engine floor,
  so its own number sits near the resolution limit of any cross-engine comparison: the net column is
  an estimate and not an identity, so **no ordering closer than a factor of two should be pressed
  against `Q8_0`**. Anyone who can spare 27 GiB of weights and does not need a long context on a
  32 GB card should use it.
- **Official FP8 is beaten by every GGUF point at or above 5 bits**, so our published "34-48 %
  below FP8" is true and a weaker achievement than it sounds.
- **K4 is the weakest of our own builds**, worse than every GGUF measured here and worse than
  official FP8. The one row it beats is `unsloth`'s NVFP4.
- **The other 4-bit-weight artifact is far behind.** `unsloth/Qwen3.8-27B-NVFP4` reads
  **0.030115** with **93.16 %** top-1 on this same shard, under the same engine so with no
  cross-engine term: 2.9x K4 at the same weight class, 8.8x the context edition, 27.7x `Q8_0`.
  It loses **512 of 512** contexts to both the context edition and official FP8 with zero ties,
  and over the full ten shards reads **0.031059** [0.027916, 0.034795] at 92.90 % top-1, losing
  **5,120 of 5,120** (`receipts/kld5-1M-nvfp4.json`, `receipts/kld5-10M-nvfp4.json`,
  `receipts/kld5-1M-paired-nvfp4.json`). The revision we measured, `9c73e2da`, no longer
  resolves upstream after a 2026-08-15 history squash; the weights at current HEAD are
  byte-identical and we keep the reviewed revision alive as
  [`malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da`](https://huggingface.co/malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da).

What it does **not** settle: this is text-only teacher-forced fidelity on one shard — one tenth of the
suite, though a close tenth: over all 10,480,640 positions the five vLLM means read 0.002760 / 0.003210 /
0.003509 / 0.005294 / 0.010604, **1.9-2.9 % above** these shard-0 values with the ordering unchanged
(`receipts/kld5-10M-*.json`), and the GGUFs have no ten-shard equivalent. It says
nothing about serving 262,144 tokens with vision and MTP on a 32 GB card, which is where these
artifacts actually differ, and llama.cpp KV-quant behaviour, prefill and decode speed are separate
axes that were not measured. A separate control bounds the protocol objection instead of arguing
it: restricting scoring to positions with at least 256 tokens of left context — the floor
`llama-perplexity` enforces — lowers every candidate's mean by 1.3-2.1 %, and second-half-only
scoring by 3.9-4.9 %, uniformly enough to change no ordering
(`receipts/scored-window-offset.json`), so the external protocol's scoring floor explains at most
about 5 % of any cross-protocol gap. Protocol identity and the remaining open comparators are in
[docs/35](docs/35-external-protocol-comparability.md) and
[docs/29](docs/29-plan-and-loose-ends.md) F2.

**Public capability, all six models on one pinned MMLU-Pro subset.** 70 questions, 14 official
categories x 5, official five-shot prefixes,
`TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, greedy, thinking at low effort,
5,120-token completion cap, every candidate scored **item-paired against the BF16 control**.
This is a first public, licence-compatible, item-paired benchmark, not a leaderboard claim.

| model | absolute | Wilson 95 % | BF16-pass retention | Wilson lower | regressions | improvements | completion-cap failures | receipt |
|---|---|---|---|---|---|---|---|---|
| BF16 `Qwen/Qwen3.8-27B` | 57/70 (81.4 %) | [70.8 %, 88.8 %] | reference | — | — | — | 4 | `receipts/public-capability-bf16.json` |
| context edition | **58/70** (82.9 %) | [72.4 %, 89.9 %] | 56/57 | **90.7 %** | 1 | 2 | 3 | `receipts/public-capability-ctx.json` |
| K4 | 57/70 (81.4 %) | [70.8 %, 88.8 %] | 55/57 | 88.1 % | 2 | 2 | 4 | `receipts/public-capability-k4.json` |
| hydrated | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 54/57 | 85.6 % | 3 | 2 | 4 | `receipts/public-capability-hyd.json` |
| official FP8 `Qwen/Qwen3.8-27B-FP8` | 56/70 (80.0 %) | [69.2 %, 87.7 %] | 55/57 | 88.1 % | 2 | 1 | 4 | `receipts/public-capability-fp8.json` |
| online K5/K6 | 55/70 (78.6 %) | [67.6 %, 86.6 %] | 54/57 | 85.6 % | 3 | 1 | 4 | `receipts/public-capability-k5k6.json` |

**The bar, and who clears it.** `receipts/public-capability-plan.json` pre-registered two
conditions before any candidate ran: BF16-pass retention with a Wilson 95 % lower bound **at or
above 0.90**, and **no category losing more than two BF16 passes**. All five candidates satisfy
the second — the worst per-category loss is two passes (hydrated and online K5/K6, philosophy).
Only the **context edition** satisfies the first, at **90.7 %**. K4 and **official FP8** read
**88.1 %**; hydrated and online K5/K6 read **85.6 %**. Four of five candidates therefore miss a
bar written down in advance, and that is published rather than restated as a pass.

**Why that is a power limit, not a verdict.** With 57 BF16 passes as the paired denominator,
**56/57 is the smallest count whose Wilson lower bound clears 0.90** — a single paired miss
already fails it — so a 70-item suite has too few items to certify this bar at all, and every
candidate interval overlaps every other. The matrix separates nothing at this size. It applies
equally to official FP8, which misses the same bar; none of this is evidence that any build is
broken. Exact-output agreement is **0/70** for every EXL3 candidate and **1/70** for official
FP8 (a single 113-token math answer) because long chains of thought differ token-wise, so the
pass/fail outcome is the only meaningful pairing. Harness `tools/public_capability.py`, sweep
runner `tools/run_public_capability.sh`, superseded 2,048-cap control
`receipts/public-capability-bf16-superseded-cap2048.json`. The honest next step is **more items
and more task types**, the plan's own P1 — HumanEval+/MBPP-style executable cases, IFEval-style
constraints, tool schemas and a larger MMLU-Pro draw — not another sweep of the same 70
questions; no capability claim graduates before that.

Three limits stated up front: v5 absolute KLD is **not** comparable to v3 (K4 reads 0.029679
there and 0.010604 here — only within-suite ordering and paired differences transfer); this run
has **no cumulative percentiles over all ten shards**, only per-shard p95/p99/p999 and the
exact global maximum, because its shard reports predate the `qwen38-fidelity-report/2` KLD
histogram — the tail table above is the shard-0 rerun, 1,048,064 of those 10,480,640
positions; and its hidden
states were deleted shard by shard to fit 135 GB of scratch, so it is reproducible from the
pinned corpus fetch log and suite manifest plus a GPU rather than from published captures.
See [PROGRESS.md](PROGRESS.md) for the full session record.

## File map

| Doc | Contents |
|---|---|
| [docs/01-nvfp4-composition.md](docs/01-nvfp4-composition.md) | Measured tensor-level composition of `nvidia/Qwen3.6-27B-NVFP4` and `unsloth/Qwen3.8-27B-NVFP4`, plus the BF16 parameter census |
| [docs/02-recipe-k4.md](docs/02-recipe-k4.md) | The proposed recipe, footprint arithmetic, and the headroom exchange rates |
| [docs/03-gg-runtime-contract.md](docs/03-gg-runtime-contract.md) | What the Gilded Gnosis EXL3 loader requires, and what it does *not* support |
| [docs/04-exllamav3-toolchain.md](docs/04-exllamav3-toolchain.md) | exllamav3 conversion flags, the missing per-module override, and the splice route |
| [docs/05-kld-protocol.md](docs/05-kld-protocol.md) | **Superseded** iteration-1 protocol: the single-window, full-vocabulary logits KLD inherited from the GG r34 runner. Kept for provenance; the current method is [docs/42](docs/42-kld-method.md) |
| [docs/06-baseline-validation.md](docs/06-baseline-validation.md) | Running the official GG image with no container runtime, and the two proven baselines |
| [docs/07-serving-recommendations.md](docs/07-serving-recommendations.md) | Serving-guide differences across upstream / NVIDIA / Unsloth cards, cross-checked against shipped configs |
| [docs/08-upstream-cards-digest.md](docs/08-upstream-cards-digest.md) | Per-card digest: declared recipe, benchmarks, harnesses, limitations |
| [docs/09-variant-publication.md](docs/09-variant-publication.md) | How iterative variants are published for independent re-measurement |
| [docs/14-fidelity-protocol-v2.md](docs/14-fidelity-protocol-v2.md) | Hidden-state replay protocol, supersedes the single-window KLD |
| [docs/18-results-fidelity-v3.md](docs/18-results-fidelity-v3.md) | Held-out re-measurement, the later offset-independent contamination correction, per-stratum means, and controls |
| [docs/22-results-iteration-2.md](docs/22-results-iteration-2.md) | Gate K5 / up K5 / down K6: corrected mean KLD **0.007945**, 38 % below official FP8 at 71 % of its resident weight |
| [docs/15-results-fidelity-v2.md](docs/15-results-fidelity-v2.md) | Superseded v2 run: 151,478 positions, 74/74 paired wins — measured on a corpus that overlapped our calibration data |
| [docs/16-head-attribution.md](docs/16-head-attribution.md) | Is `lm_head` the sensitive tensor? Measured: not at K6 |
| [docs/11-kld-external-comparison.md](docs/11-kld-external-comparison.md) | Published KLD data for this family; why "FP8 = 0.5" is wrong |
| [docs/13-upstream-contributions.md](docs/13-upstream-contributions.md) | Upstream issue + verified PR |
| [docs/19-cuda-graphs-patch.md](docs/19-cuda-graphs-patch.md) | The autotune-priming patch behind PR #314, deliverables and verification |
| [docs/20-context-extension-and-k3-gap.md](docs/20-context-extension-and-k3-gap.md) | How far context can be extended, and the distance to the reference protocol |
| [docs/12-iteration-2-plan.md](docs/12-iteration-2-plan.md) | Where to invest next, ranked |
| [docs/17-next-iteration-shopping-list.md](docs/17-next-iteration-shopping-list.md) | Iteration-2 shopping list, each item with its closing test |
| [docs/10-results-iteration-1.md](docs/10-results-iteration-1.md) | Iteration 1: build, serve, KLD, and the upstream defect |
| [docs/21-independent-review-response.md](docs/21-independent-review-response.md) | Independent review, per finding: fixed, fixed-in-v2, or still open |
| [docs/24-p0-results.md](docs/24-p0-results.md) | **P0 done**: prefill +113 %, and why fp32 replay was a negative result |
| [docs/23-next-attack-list.md](docs/23-next-attack-list.md) | **Ranked plan for iteration 3**, with evidence, cost and acceptance per item |
| [docs/25-goal-pareto-dominate-fp8.md](docs/25-goal-pareto-dominate-fp8.md) | the goal as a gate, and the iteration-3 verdict per axis |
| [docs/26-prefill-attribution.md](docs/26-prefill-attribution.md) | where prefill time goes: MLP 2.13-2.26x, attention overlay 1.05-1.11x, hgemm at cuBLAS parity |
| [docs/27-graph-decode-drift-control.md](docs/27-graph-decode-drift-control.md) | eager-vs-graph drift is ambient: BF16 drifts the same as EXL3 |
| [PROGRESS.md](PROGRESS.md) | Chronological work log |
| [docs/28-external-validation-and-corrections.md](docs/28-external-validation-and-corrections.md) | external RTX 5090 validation and the corrected capacity boundary |
| [docs/30-iteration-4-context-edition.md](docs/30-iteration-4-context-edition.md) | serialized-K5 context build, receipts and rejected vision quant |
| [docs/31-frozen-qualification.md](docs/31-frozen-qualification.md) | source-disjoint frozen v4 qualification |
| [docs/32-native-context-embedding-overlay.md](docs/32-native-context-embedding-overlay.md) | int8 input table, native MTP-3 plus 8.4 MP engine-budget proof (since superseded by the physical 5090 qualification at utilisation 0.955), and corrected draft accounting |
| [docs/33-evidence-volume-and-intervals.md](docs/33-evidence-volume-and-intervals.md) | the 10,480,640-position rerun, the measured tail, and why ten times the positions did not narrow the interval |
| [docs/29-plan-and-loose-ends.md](docs/29-plan-and-loose-ends.md) | ranked open work and acceptance gates, including the evidence-volume objection (F1) this session closed |
| [docs/35-external-protocol-comparability.md](docs/35-external-protocol-comparability.md) | why our KLD and Unsloth's are not interchangeable, the twelve deltas, and the two of them now measured: the scoring-window offset and the cross-engine floor |
| [docs/34-vram-class-profiles.md](docs/34-vram-class-profiles.md) | the 24 GB and 16 GB class verdicts, the measured affine KV law, and a capped-budget proxy qualification |
| [docs/36-performance-levers-5090.md](docs/36-performance-levers-5090.md) | 11 configurations on the physical 5090: what moved throughput, what is a no-op, and the GDN gate reachability rate |
| [docs/37-error-driven-allocation.md](docs/37-error-driven-allocation.md) | the error-driven allocation experiment that lost, the 3.73x-per-bit law it produced, and why the objective is not a surrogate for KLD |
| [docs/39-chat-template-audit.md](docs/39-chat-template-audit.md) | every chat template in this model family captured with revision and sha256, diffed by named construct, and the verdict that ours is byte-identical to official; the community "fixed" template measured losing tool-call arguments under `qwen3_coder`, and the loudest complaint traced to the 2.4T flagship's template rather than the 27B's |
| [docs/42-kld-method.md](docs/42-kld-method.md) | **the KLD method of record**: the metric, the shared BF16 head and why, the capture point, the suite and its contamination policy, the ladder, the bootstrap and the tail histogram, the harness-revision pin, how to re-derive a published number on a CPU in seconds, every floor that bounds an absolute value, and the claim boundary |

Tooling in [tools/](tools/) is what produced the evidence: an unprivileged OCI image
puller and a proot-based runner for it, the BF16 attention splice and checkpoint
finaliser, the fidelity harness (`fidelity.py`, `suite3.py`) that builds the suite and
replays captures, the decode-parity probe (`decode_parity.py`), the kernel
microbenchmarks (`prefill_micro*.py`, `gemm_cmp.py`) and the upstream patches as
standalone files. The v5 10.48 M-position run, the public-benchmark harness and the
collection index are these files:

| Tool | What it does | Receipts |
|---|---|---|
| [tools/fetch_corpus_v5.py](tools/fetch_corpus_v5.py) | fetches the five-stratum v5 corpus (941 documents / 70,348,971 bytes) and pins every document by URL and sha256 | `receipts/kld5-corpus-fetch-log.json` |
| [tools/suite3.py](tools/suite3.py) | builds the frozen suite: exact-advance non-overlapping windows, whole-document calibration-overlap pre-exclusion (44 of 941), prior-suite token exclusion (160 v4 hashes, 0 reachable) | `receipts/kld5-suite-manifest.json` (`qwen38-distribution-fidelity/6`) |
| [tools/kld_ladder.sh](tools/kld_ladder.sh) | walks the ladder one 512-context shard at a time — capture six models, replay five candidates, verify, delete the shard's ~64 GB of hidden states — because scratch is ~135 GB | ten per-shard reports, listed in every cumulative receipt |
| [tools/kld_aggregate.py](tools/kld_aggregate.py) | welds verified per-shard reports into cumulative means, cluster bootstraps, paired comparisons and (for `qwen38-fidelity-report/2` inputs) bin-bounded cumulative quantiles | `receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json`, `receipts/kld5-10M-paired.json` |
| [tools/fidelity.py](tools/fidelity.py) | capture/replay harness; now also counts every scored position into a fixed log-spaced `kld_tail` histogram, bumping reports to `qwen38-fidelity-report/2` | per-shard and per-candidate reports |
| [tools/public_capability.py](tools/public_capability.py) | paired MMLU-Pro run against a pinned dataset revision, official five-shot prefixes, greedy, with Wilson intervals and BF16-pass retention | `receipts/public-capability-{plan,suite-mmlupro-70,bf16,ctx,k4,hyd,fp8,k5k6}.json`, superseded 2,048-cap control kept as `receipts/public-capability-bf16-superseded-cap2048.json` |
| [tools/run_public_capability.sh](tools/run_public_capability.sh) | drives the six-model sweep one server at a time against the frozen BF16 reference, with the per-candidate environment each build needs (every EXL3 candidate requires `VLLM_EXL3_GRAPH_DECODE=1`) | the five candidate receipts above |
| [tools/collection_index.py](tools/collection_index.py) | one immutable row per published checkpoint, every field carrying its receipt path, sha256 and RFC 6901 pointer; disk bytes and `resident_weights` kept strictly distinct | `receipts/collection-index.json` (`qwen38-collection-index/1`) |
| [tools/gguf_capture.cpp](tools/gguf_capture.cpp) | llama.cpp-side capture of the post-final-norm state for a GGUF at pinned commit `ece963f4`, so a GGUF can be replayed through our shared BF16 head; bf16 rounding verified bit-identical to torch on 2,012,449 probe values | `receipts/gguf-report-{q8_0,q6_k,q5_k_xl,engine-floor}.json`, `receipts/cross-engine-comparator.json` |
| [tools/gguf_manifest.py](tools/gguf_manifest.py) | pins each capture to its GGUF blob digest and the llama.cpp identity that produced it | the four `receipts/gguf-report-*.json` |
| [tools/build_llamacpp.sh](tools/build_llamacpp.sh) | builds the capture binary against the pinned llama.cpp commit | `engine_identity` in every GGUF report |
| [tools/chat_template_matrix.py](tools/chat_template_matrix.py) | renders every candidate chat template over 34 fixed cases inside the pinned image, reconstructing transformers' Jinja environment exactly, and reports token counts and sha256 per rendered prompt; CPU only | `receipts/chat-template-audit.json`, `receipts/chat-templates/renders.json` |
| [tools/chat_template_roundtrip.py](tools/chat_template_roundtrip.py) | feeds each template's rendered tool call back through the image's own `vllm.parser.qwen3._qwen3_arg_converter` and reports whether `--tool-call-parser qwen3_coder` recovers the arguments losslessly | `tool_call_roundtrip` in the audit receipt |
| [tools/chat_template_prefix.py](tools/chat_template_prefix.py) | first-divergent-token between the generated token stream and the re-rendered history, per client reasoning-echo behaviour, i.e. the point past which no prefix-cache block is reusable | `prompt_growth_and_prefix_stability` in the audit receipt |
| [tools/fetch_chat_templates.sh](tools/fetch_chat_templates.sh), [tools/gguf_chat_template.py](tools/gguf_chat_template.py) | fetch every family template with its revision and digest, including off-default-branch refs and the one that exists only inside GGUF metadata (read over HTTP range requests, no tensor bytes) | `receipts/chat-templates/{SHA256SUMS,raw/}` |

## Status

- [x] Both upstream Qwen3.8-27B artifacts proven runnable under the official GG image
- [x] Iteration 1 (K4 MLP + BF16 attention splice) served under GG with `ONLINE_QUANT=exl3-b6`, 19.21 GB resident, vision verified, published as [`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4)
- [x] Held-out v3 fidelity suite with a contamination scan, plus the [dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3) that lets anyone recompute it without a GPU
- [x] Iteration 2 built, measured and published as [`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6): **20.32 GiB** resident, overlap-corrected mean KLD **0.007945** body-only / **0.008078** as served, top-1 96.86 %
- [x] CUDA graphs ([PR #314](https://github.com/local-inference-lab/vllm/pull/314)), row-count prefill dispatch ([PR #316](https://github.com/local-inference-lab/vllm/pull/316)), B12X prefill routing ([PR #318](https://github.com/local-inference-lab/vllm/pull/318)) and quantized-embedding construction ([PR #319](https://github.com/local-inference-lab/vllm/pull/319)) filed against the fork and exercised in the tested runtime
- [x] Prefill attributed and bounded: the MLP kernel is the whole story (2.13x / 2.26x), the online-K6 attention overlay is not the bottleneck (1.05x / 1.11x), `ext.hgemm` is already at cuBLAS parity — FP8 prefill parity needs a fused dequant-in-epilogue kernel, not tuning
- [x] Attention-overlay width is a runtime knob, measured on the same suite and real RTX 5090: K6 **0.007945** at 20.32 GiB and ~185k context; K5 **0.011801** at 19.82 GiB and **206,400 retrieval-verified**; K4 **0.026619** at 19.05 GiB
- [x] Serializing attention offline instead of encoding it at load — the "hydrated" build — measures **0.007172**, i.e. calibrated offline K6 beats the runtime overlay by 0.000773 on the overlap-corrected subset
- [x] Graph-vs-eager decode drift measured properly and traced to the build, not to EXL3 ([docs/27](docs/27-graph-decode-drift-control.md))
- [x] Frozen v4 suite: 160 source-disjoint contexts, 100 documents, zero token/document/content overlap; all five candidate capture sets and qualification receipts published (**2,708 files / 51.0 GB**)
- [x] Context edition plus per-row int8 input table starts native 262,144 with **MTP-3, decode graphs and an 8.4 MP image cap** under a 30.24 GiB engine budget; 266,612 KV tokens, exact retrieval at 261,794 text tokens and in a 236,824-token seven-megapixel request — that engine-budget run is now superseded by the physical-card qualification below
- [x] **Hard-limit qualification on a physical RTX 5090 closed** (`receipts/qualification-5090-context.json`, per-process server logs `receipts/qualification-5090-context-server-{B3,B4,C,D,E,F}.log`): the context edition serves native **262,144 with MTP-3, the full 8,388,608-pixel image ceiling and fp8 KV on one 32,607 MiB card** — but at **`--gpu-memory-utilization 0.955`** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, not the 0.97 the cards used to print. Measured at startup: engine budget **29.98 GiB**, free 30.9 of 31.4 GiB usable, usage 18.19 weight + 1.78 peak activation + 0.27 non-torch + 0.45 CUDAGraph = **20.69 GiB**, **available KV 9.28 GiB = 265,122 KV tokens**, 1.01x maximum concurrency at 262,144, attention block size forced to 1600 tokens so the attention page is at least the mamba page, 3 padding layers at at most 6.25 % KV waste, startup 55.7 s, model load 18.19 GiB in 3.99 s. **All seven gates PASS:** (1) startup allocates a native-length request inside the utilisation ceiling; (2) the 261,794-token needle retrieved exactly; (3) a combined 236,824-token plus 7,077,888-pixel request returns `1376346594 | red, blue` exactly; (4) the 30-case image suite at **24/30**; (5) three warmed 256-token concurrency-1 decode runs; (6) a second native-length request after release in the same process; (7) receipt identity complete
- [x] **0.97 published as a bounded negative, with its mechanism:** at 0.97 the combined text-plus-image request dies with `torch.OutOfMemoryError` inside `vllm/v1/attention/ops/vit_attn_wrappers.py` wanting 62.00 MiB with 26.50 MiB free, and `expandable_segments` does not rescue it because vLLM spends the freed bytes on more KV (272,570 → 280,017 tokens); lowering `max_pixels` to 4,194,304 at 0.97 is **strictly worse** (profiled peak activation falls to 1.35 GiB, KV grows to 291,933 tokens, OOM with 6.56 MiB free). The knob is utilisation, not the image ceiling — seven megapixels is **not** a hard 32 GB ceiling, since the identical request succeeds at 0.955 with the full 8,388,608-pixel ceiling
- [x] Task-retention smoke: BF16, all four EXL3 profiles and official FP8 each pass **40/40** deterministic paired tasks with zero regressions; a hardened rescore leaves every pass unchanged and corrects exact-final-answer agreement to 32/40–35/40 (`receipts/task-retention-v2-summary.json`, `receipts/task-retention-v2-strict-rescore.json`)
- [x] Evidence volume objection (F1) closed: fidelity re-measured on the held-out v5 suite at **10,480,640 scored positions** / **5,120 contexts** / **842 source clusters**, ten verified shards welded by `tools/kld_aggregate.py` — hydrated **0.002760**, online K5/K6 **0.003210**, context **0.003509**, official FP8 **0.005294**, K4 **0.010604**, all body-only through one shared BF16 head (`receipts/kld5-10M-*.json`)
- [x] Paired at 5,120 contexts: hydrated **-0.002534** (5,118 wins), online K5/K6 **-0.002084** (5,105), context **-0.001785** (5,109) against official FP8; hydrated beats the runtime overlay by **-0.000450** (4,922); K4 loses to FP8 on 5,113 of 5,120 (`receipts/kld5-10M-paired.json`)
- [x] Ladder stability shown, not assumed: cumulative hydrated mean **0.002700 / 0.002759 / 0.002699 / 0.002760** at the 1M / 2M / 5M / 10M checkpoints
- [x] Exact tail now aggregable for future runs: a fixed log-spaced KLD histogram in `tools/fidelity.py` (`qwen38-fidelity-report/2`) plus bin-bounded cumulative quantiles in `tools/kld_aggregate.py`; this run kept only per-shard percentiles and the exact global maximum, and says so in its receipts
- [x] First public-benchmark matrix, all six models: paired MMLU-Pro (70 questions, pinned `TIGER-Lab/MMLU-Pro@b189ec76`) — BF16 **57/70**, context **58/70** at **56/57** retention (Wilson lower **90.7 %**, the only candidate clearing the pre-registered 0.90 bar), K4 **57/70** and official FP8 **56/70** at **55/57** (88.1 %), hydrated **56/70** and online K5/K6 **55/70** at **54/57** (85.6 %); the four shortfalls are a 70-item power limit — 56/57 is the smallest count that can clear 0.90 — not a broken build, so the next step is item volume and task diversity
- [x] Collection index published: `receipts/collection-index.json` from `tools/collection_index.py`, four immutable rows, receipt-traced fields, one disclosed divergence left unfixed rather than rewriting a published receipt
- [x] Comparator breadth (F2) closed by measurement, including the unflattering half: GGUF `Q8_0` **0.001087**, `Q6_K` **0.002035** and `UD-Q5_K_XL` **0.004444** scored on shard 0 of the v5 suite through the same shared BF16 head, against a measured llama.cpp-versus-vLLM engine floor of **0.000507** — so at the 6-bit point `Q6_K` (0.001528 net, 21.31 GiB) **beats our best build** (0.002700, 20.12 GiB payload), at the 5-bit point our context edition (0.003409, 19.27 GiB) is ~13 % better than `UD-Q5_K_XL` (0.003936 net, 18.83 GiB), and `Q8_0` leads on fidelity at 27.05 GiB (`receipts/cross-engine-comparator.json`)
- [x] The largest cross-protocol objection bounded instead of argued: restricting scoring to their 256-token left-context floor lowers every candidate's mean by 1.3-2.1 % and second-half-only by 3.9-4.9 %, uniformly, changing no ordering (`receipts/scored-window-offset.json`)
- [x] **Their protocol run for cross-citation**: `llama-perplexity --kl-divergence` on wikitext-2 raw at ctx 512, 147,900 scored positions, base PPL 6.950230 ± 0.044933 — `Q8_0` **0.000926**, `Q6_K` **0.002286**, `UD-Q5_K_XL` **0.004426**, the **same ordering** as our suite at 1.60x / 1.50x / 1.12x our net-of-floor values, with their harness floor measured on our own hardware at 5.6e-5 to 8.0e-5 and tokenisation eliminated as an explanation (bit-identical 297,194-token streams). Also found: **PPL does not reproduce the KLD ordering** (`receipts/wikitext-kld-run-a.json`, [docs/35](docs/35-external-protocol-comparability.md))
- [x] **`unsloth/Qwen3.8-27B-NVFP4` measured on our suite**, the comparator readers ask for most: **0.031059** [0.027916, 0.034795] at 92.90 % top-1 over 10,480,640 positions, losing **5,120 of 5,120** contexts to both the context edition and official FP8; 2.9x K4 at the same 4-bit weight class (`receipts/kld5-10M-nvfp4.json`, `receipts/kld5-1M-paired-nvfp4.json`)
- [x] **Capture and replay are bit-reproducible**: 5,240,320 scored positions identical across independent runs, separate model loads and three harness generations, under eager single-sequence capture — which makes every published KLD number re-derivable rather than merely re-runnable (`receipts/capture-determinism.json`)
- [x] **The converter is not deterministic, and it does not matter for fidelity.** A fresh conversion of the published hydrated recipe returns 13 of 16 pinned files identical and three safetensors shards differing in tensor bytes only — 399 of 409 quantized modules, 97.6 %, at 41-92 % of bytes each — with identical headers, offsets, shard sizes and per-role byte totals. That *sibling* then scored **-3.755e-06** paired against the published checkpoint, 95 % [-2.854e-05, +2.062e-05] bracketing zero, 257 contexts to 255. So the recipe determines fidelity to within this protocol's resolution and the published bytes are the artifact (`receipts/converter-determinism.json`, `receipts/sibling-rebuild-fidelity.json`)
- [x] **Performance levers measured on the physical 5090**, 11 configurations: MTP depth is a concurrency decision — depth 3 wins at one stream (83.31 vs 74.28 per-request tok/s) and `num_speculative_tokens 1` wins at eight (**+30.67 %**, 313.28 → 409.35 tok/s aggregate, plus 10,911 more KV tokens). Closed as no-ops or losses: explicit `FLASHINFER` (already auto-selected on SM120), `TRITON_ATTN`, `custom_ops:["all"]`, and dynamic speculative decoding (refuses to start under `FULL_DECODE_ONLY`). Prefill did not move on any lever ([docs/36](docs/36-performance-levers-5090.md))
- [x] **Prefix caching shipped, scoped by measurement**: on the three 8,192-token recipes it is worth **11.6x** and **29.3x** warm TTFT at 32 k and 131 k prefixes (whole schedule 144.4 s → 84.0 s); at the context edition's native 262,144 it is **declined** because align-mode block rounding makes the window unservable three different ways, with a measured 256,000-token option priced at 6,144 tokens. The four-module image `gg-r34-patched-apc` is the release unit (`receipts/apc-poison-repro.json`, `receipts/qualification-5090-apc.json`)
- [x] **VRAM classes decided**: 24 GB is a GO as a serving profile over the published context edition at **24,576 tokens with MTP-3** or 45,056 with MTP off, validated by a capped-budget proxy with the physical-board gate left open; 16 GB is a NO-GO as a SKU and published as a design study. The per-token KV model was replaced by the measured affine law `a*L + M` ([docs/34](docs/34-vram-class-profiles.md))
- [x] **A user-reported defect investigated to a measured negative**: 266 requests across seven freshly started servers reproduced no corruption under the reporter's exact condition, which decomposed his four-variable control and left LMCache as the sole suspect; the two absent upstream fixes were requested at the fork (issue #392, PR #393) and a net-new admission livelock was filed upstream as [vllm-project/vllm#52520](https://github.com/vllm-project/vllm/issues/52520)
- [x] **Materials published for third-party replay**: the v5 suite, all ten shard views, the shard-0 BF16 reference capture and 79 per-shard reports as [`malaiwah/qwen38-27b-fidelity-suite-v5`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5), the losing error-driven candidate as a documented negative, and archival mirrors of the NVFP4 and GGUF revisions our published numbers cite — because one of them stopped resolving upstream mid-session
- [ ] Still open in F2: stock uniform-bitrate EXL3 controls, and the reader-suggested ik_llama (`IQ4_KS_KT`) and NVFP4-NInfer comparators — neither of which has any KLD number today ([docs/29](docs/29-plan-and-loose-ends.md) F2)
