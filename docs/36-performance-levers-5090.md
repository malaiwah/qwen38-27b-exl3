# 36. Performance levers on the physical RTX 5090: decision record

**Decision: change nothing on the qualified single-stream profile. Use MTP depth 1 instead of
depth 3 only when the deployment actually runs concurrent streams (measured at 8).**

Every number below was measured on the user's own AIBoss host, one physical **GeForce RTX 5090**
(32,607 MiB, driver 610.57.04, `GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a`), on the immutable
production image `localhost/vllm:gg-r34-patched`
(manifest `sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27`) with **no
source bind mounts** — the three patch modules are baked in and were re-verified by digest and
by import resolution inside the image before any GPU work. Nothing here was measured on the
rental RTX PRO 6000 and **no number here is comparable to it**. Full rows, commands and every
timed repeat: `receipts/perf-sweep-5090.json`.

## The profile these rows describe, and what it is not

The rows are an explicit **`--max-num-seqs 8` concurrency profile** at `--max-model-len 262144`,
fp8 KV, `FULL_DECODE_ONLY`, MTP depth 3, `VLLM_EXL3_EMBED_BITS=8`,
`VLLM_EXL3_PREFILL_RECONSTRUCT_M=128`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
They are **not** the rank-1 qualified numbers, which are single-sequence by design.

Two memory facts came out of building it, both load-bearing:

* At `--max-num-seqs 8` the 262,144-token profile **does not start at the qualified
  `--gpu-memory-utilization 0.955`**: the engine needs 9.13 GiB of KV and 0.955 leaves 9.07 GiB,
  so it refuses and suggests `max_model_len 259200`. The whole matrix therefore runs at **0.97**,
  which is legitimate for a text comparison but **cannot serve a large image** — so a 0.97 row is
  never a publishable serving profile. `--max-num-seqs 8` is not a drop-in for the card's
  single-sequence recipe.
* Widening the decode graph pool to every request count from 1 to 8 costs almost nothing:
  **8 captured decode shapes, 0.55 GiB** graph pool versus 0.46 GiB for the single shape of the
  qualification, and KV came out identical at **272,570 tokens**.

Sanity check against the qualification, which used a different prompt: our base step time at
concurrency 1 is **25.72 ms** against its **25.05 ms** implied — 2.6 % apart. The tok/s gap
(82.94 versus 107.56) is **entirely acceptance**: 2.14 accepted tokens per step here against
2.69 there, because our frozen prompts are literary prose and theirs was repetitive technical
prose. Acceptance is prompt-dependent; step time is not.

## What the levers actually did

Decision metric is **accepted tokens per step ÷ step time**, never acceptance rate. Aggregate
tok/s at temperature 0, 256 output tokens, three warmed repeats, identical frozen token-id
prompts for every row:

| configuration | C1 | C4 | C8 | acc. tok/step (C1) | step ms (C1) | step ms (C8) | KV tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| **base** (depth 3, auto backend, Inductor) | **82.94** | 263.12 | 313.28 | 2.14 | 25.72 | 49.52 | 272,570 |
| `custom_ops:["all"]` | 84.66 | 263.42 | 312.00 | 2.25 | 26.29 | 52.10 | 272,570 |
| `--attention-backend FLASHINFER` (explicit) | 82.27 | 249.34 | 312.89 | 2.15 | 25.93 | 50.35 | 272,570 |
| `--attention-backend TRITON_ATTN` | 87.52 | 263.77 | 323.46 | 2.29 | 25.88 | 52.24 | 272,570 |
| MTP depth 2 | 79.46 | **269.79** | 306.67 | 1.96 | 24.59 | 47.37 | 278,812 |
| **MTP depth 1** | 74.09 | 256.91 | **409.35** | 1.66 | 22.29 | **31.07** | **283,481** |

* **Speculative depth is the only real lever.** Depth 1 is **+30.7 % aggregate** and **+22.8 %
  per request** at concurrency 8 (decision metric 53.91 against 43.92 tok/s),
  because its step time collapses from 49.52 ms to 31.07 ms while accepted tokens per step only
  fall from 2.18 to 1.68. At concurrency 1 the trade reverses — depth 3 wins 82.94 to 74.09
  aggregate, 83.31 to 74.28 on the decision metric —
  and at concurrency 4 the three depths are within ±2.6 % of each other. Depth also buys back
  KV: depth 1 holds **10,911 more KV tokens** and was the only configuration whose needle ran at
  the full 261,794 tokens.
* **The attention backend is already FlashInfer.** The engine logs
  `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`
  with head_size 256 and fp8 KV on SM120, so passing `--attention-backend FLASHINFER` explicitly
  is a no-op (−0.8 % at C1, inside repeat spread) and the real A/B is the *other* direction.
  Forcing `TRITON_ATTN`, which the live K4 service does, changes **step time by under 1 %** at C1
  and worsens it by 5.5 % at C8; it looks like a +5.5 % win at temperature 0 and a **−7.3 % loss
  at temperature 0.6**, i.e. it moves which tokens the drafter proposes, not how fast a step
  runs. That is acceptance noise, not throughput, so neither direction is worth changing.
  `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1` cost nothing measurable and left KV unchanged.
* **`custom_ops:["all"]` is not a win here.** Step time is **2.2 % worse at C1 and 5.2 % worse at
  C8**; the small C1 tok/s gain is acceptance again, and at temperature 0.6 the row is 3.7-7.5 %
  *worse* everywhere. We run Inductor, which is already generating the fused elementwise kernels
  the vendor's launcher asks for by hand.
* **Prefill did not move at all.** 3,374 / 3,255 tok/s at 2,048 / 6,144 prompt tokens for base;
  no graph-decode row is more than 4.0 % away (`custom_ops` is the largest at −4.0 % / −3.8 %),
  and the FlashInfer arm is 1.7 % *slower*; the eager rows lose a further 8 %. The prefill
  deficit is structural and none of these levers touch it.
* **CUDA-graph decode remains the single biggest effect on this card.** Eager decode is
  **43.45 tok/s** at C1 against 82.94 with graphs — graphs are worth **1.91×**, which is why the
  next item matters so much.

## Both dynamic speculative-decoding knobs are closed on this build

`adaptive_speculative_tokens_window` and `num_speculative_tokens_per_batch_size` are real,
implemented, and **unusable with our graph-decode profile**. `VllmConfig`
`_maybe_override_dynamic_sd_cudagraph_mode` downgrades `cudagraph_mode` from `FULL_DECODE_ONLY`
to `PIECEWISE` whenever either knob is set on the MRV1 runner, `PIECEWISE` has a non-`NONE` mixed
mode, and `Exl3Config` refuses that outright, so the server **fails to start**:

> The EXL3 quantization backend requires eager execution: pass `--enforce-eager` … Graph decode
> was not permitted because `cudagraph_mode=PIECEWISE` also captures mixed prefill batches, whose
> token counts are not enumerable before capture.

Measured with `--enforce-eager`, the only form that runs, they are **losses even against an eager
baseline** (temperature 0, two repeats, utilisation 0.95 because eager turns the freed graph pool
into KV and OOMs at 0.97):

| eager configuration | C1 | C4 | C8 |
|---|---:|---:|---:|
| eager depth 3 (reference) | 43.45 | 154.21 | 273.82 |
| `adaptive_speculative_tokens_window=8` | 42.84 | 148.78 (−3.5 %) | 257.94 (−5.8 %) |
| depth 3 at batch 1-2, depth 1 at batch 3-8 | 43.46 | 131.62 (−14.6 %) | 240.90 (−12.0 %) |

So the per-concurrency depth schedule that our own depth measurements make so attractive cannot
be bought: taking it costs graph decode (−48 %), which is far more than the schedule can win.
The right way to get it is a static depth chosen for the expected concurrency. `VLLM_USE_V2_MODEL_RUNNER=1`
would keep full graphs, but MRV2 sizes graph shapes through a different path than the exl3
priming pass covers, which risks `exl3_gemm` autotuning *inside* graph capture — the exact fault
the priming pass exists to prevent — so it was not attempted on a card that had to be handed back.
`rejection_sample_method` was left at `standard`; `synthetic` fabricates acceptance and is never a
serving option.

## Closed avenues, not retried

`B12X_ATTN` needs block size exactly 64 or 128 and the hybrid mamba page forces
`Setting attention block size to 1600 tokens` here, so it cannot load. `FULL_AND_PIECEWISE` is
refused by the same exl3 guard as above, and our history records Xid 31 on graph-captured short
SM120 prefills. `VLLM_EXL3_PREFILL_FP8` is a silent no-op: the pinned extension exports neither
`reconstruct_fp8_slice` nor `reconstruct_had_slice`. `--max-num-batched-tokens 8192` was already
measured as noise ([`receipts/prefill-pp-chunk8k.json`](../receipts/prefill-pp-chunk8k.json)).
Prefix caching is off by default for this hybrid model
(`enable_prefix_caching=False` in every banner, measured hit rate 0.0) and enabling it needs the
`-apc` superset image, which is not the qualified artifact. SparkInfer and b12x are one component
under two names, so there is no second engine to switch on.

## Fidelity guard

Nothing was recommended without it. Both kept configurations score **24/30** on the deterministic
30-case image suite (digits 8/10, bars 9/10, grid 7/10 — identical to the rank-1 qualification)
and retrieve the needle exactly at the longest length that fits: **258,925 tokens** for base and
**261,794** for depth 1, at 1,472 and 1,450 tok/s including decode.

`custom_ops:["all"]`, either attention backend choice, and any non-`standard` rejection sampler
are **not bit-exact**: they change kernels, reduction order or the acceptance test, so published
KLD and capability numbers hold only for the configuration that produced them. This is one more
reason not to take a lever whose apparent gain is acceptance noise.

## Same window, same image: production-image serving gates

The runtime gates that `receipts/production-image.json` had recorded as `null` were closed in
this window with `docker/build-image.sh smoke` on all four card recipes (context, hydrated, k5k6,
k4), started from the release image with the source mounts deleted. That harness owns those rows;
this sweep contributed the GPU time and one defect fix — `docker/smoke_client.py` hardcoded the
model id `m` while the recipes serve `qwen38`/`qwen38-k4`, so vLLM answered **HTTP 404** and the
exact-match gate silently recorded `null`. The client now resolves the served id from `/models`.

Independently measured here: these servers bind loopback only. With `--network host` and
`--host 127.0.0.1`, `http://127.0.0.1:8231/health` answers and `http://10.15.0.151:8231/health`
does not, on every row (`exposure-*.json`). There is no API key; the endpoint must stay behind an
authenticated proxy.

## What to change

1. **Nothing, for the shipped single-stream profile.** No lever beat the qualified configuration
   on the decision metric at concurrency 1 by more than acceptance noise.
2. **If a deployment serves 8 concurrent streams, set `num_speculative_tokens` to 1**: +30.7 %
   aggregate throughput, +22.8 % per stream, +10,911 KV tokens, fidelity unchanged. Keep depth 3
   for interactive single-stream use. At concurrency 4 the choice does not matter.
3. **Do not chase dynamic depth** on this build; it costs CUDA-graph decode.
4. **Keep `--max-num-seqs 1` at utilisation 0.955** for any vision-capable deployment. The 8-seq
   profile only starts at 0.97, where a large image OOMs in the vision tower.

GPU handover: this window ended with the card handed to `ApcPoisonRepro` at 03:36 UTC with the
`qwen38-27b` unit stopped and the GPU idle. It owns the restore against
`receipts/aiboss-live-service-snapshot.json` (including the fifteen-field `podman inspect` diff,
which is a stronger check than health, since a drifted-env restart still reports healthy), and
`TwentyFourGigProxy` takes the card from it afterwards. Raw
server logs, probe JSONs and frozen prompts are kept on AIBoss under `/home/mbelleau/perfsweep`
(digests in the receipt); nothing was deleted.

## The Qwen GDN speculative-gate module (#51812) at eight streams

Receipt: `receipts/gdn-gate-concurrency.json`. This section is additive and does not revise any
measurement above; in particular it does not contradict item 2 of *What to change*.

Upstream vLLM PR #51812 is absent from the pinned image. The vendored
`qwen_gdn_linear_attn.py` gathers the speculative Q/K/V rows with `spec_token_indx` (line 1298)
but hands the recurrent update the ungathered gate tensors `a` and `b` (lines 1421-1424). Those
two agree **exactly** when `spec_token_indx == arange(T_spec)`, so the module can only change a
number on a forward pass where some non-speculative token sits at a lower batch index than a
speculative one. That is the whole question, and it is a reachability question rather than an
accuracy question.

**Why the batch order usually saves us.** Batch order comes from
`reorder_batch_to_split_decodes_and_prefills` (`utils.py:665`), whose `decode_threshold` is the
*minimum* over attention groups (`gpu_model_runner.py:7310-7328`). Both groups on this build
report `1 + num_spec`: the GDN builder at `gdn_attn.py:112`, and FlashInfer at
`flashinfer.py:855`, whose `supports_spec_as_decode` resolves true because the pinned
flashinfer 0.6.18+cu132 `fast_decode_plan` carries `q_len_per_req` (`flashinfer.py:2422`) and
trtllm/XQA decode is unavailable on SM120 (`vllm/utils/flashinfer.py:394` requires capability
family 100; this card reports 12.0). So the threshold is 4 at depth 3, a speculative decode's
four tokens are "below threshold", and it sorts into region 0 at the front while every prefill
and extend sorts behind it (`utils.py:691-709`). The permutation is then the identity and the
vendored file is bit-for-bit correct.

**Measured, with a counter rather than an argument.** A byte-identical copy of the image's own
`gdn_attn.py` plus a purely additive host-side counter (no device sync, so it cannot perturb what
it measures) recorded, per metadata build, whether the speculative tokens were the leading tokens
of the batch:

| arm | configuration | metadata builds | on the gather branch | miscomputed | rate |
|---|---|---|---|---|---|
| R1 | the published C8 recipe, 262144, budget 2048 | 3329 | 2112 (63 %) | **0** | below 0.90 per thousand builds (95 % UB, rule of three) |
| R3 | as R1 but `--max-num-batched-tokens 512` | 8065 | 6915 (86 %) | **0** | below 0.37 per thousand builds |
| R2 | the shipped 8192 prefix-caching recipe flags, at 8 streams | 5825 | 2211 (38 %) | **3** | 0.515 per thousand builds; 1.357 per thousand gather-branch builds |

The zero in R1 is not a missed window: the suspect gather branch executed 2112 times and the
permutation was the identity every time. R3 exists because a bare zero over 38 prefill
completions would have been consistent with the mechanism being real; it raised gather-branch
exposure 3.3x and still found nothing, bounding the token-budget mechanism below 0.332 per
thousand gather-branch builds over 9027 of them.

R2 is the one that fires, and it matters because prefix caching *ships* in the three
8192-token recipes. Three builds had composition `6s/1p/0d` with displacement 4 and first bad
index 20: tokens 0-19 speculative, then a four-token non-speculative request, then the sixth
speculative decode's four tokens behind it. Exactly one speculative decode was displaced and its
four gate rows were read from the wrong tokens.

**Verdict.** At eight streams the module changes **nothing measurable** under our published
concurrency recipe — below 0.90 miscomputed builds per thousand, N = 3329 — and it **does change
the computation** in the shipped prefix-caching regime, at 0.515 per thousand. What is *not*
resolved is whether those three passes changed an emitted token: upstream's own per-event effect
is 0.002755 mean absolute chosen-logprob error against this build's 0.0823 run-to-run floor
(`receipts/apc-poison-repro.json` arm Bn), roughly thirty times larger, so three events in 5825
builds are far below our resolution. The A/B/null comparison arms were deliberately **not** run:
they would have produced a null result that looked like evidence of no effect while carrying no
information. Recommendation is to keep the module as an optional read-only overlay and mount it
only where a recipe ships prefix caching **and** speculative decoding together.

That last conjunction matters and narrows the recommendation, a precision `CardFinalPass` supplied
while landing this on the cards: `gdn_attn.py:111` sets `use_spec_decode` from
`num_speculative_tokens > 0`, and with it false `spec_sequence_masks` is always `None`, so
`qwen_gdn_linear_attn.py:1293` never enters the speculative path at all and the module is a no-op
*by construction*. The published 8,192-token commands for K4 and hydrated carry no
`--speculative-config`, so the overlay buys them nothing until MTP is added. The one shipped
configuration this measurement actually indicts is **K5K6 Recipe B**, which ships mtp depth 3
alongside the cache at eight streams. It is likewise not needed for the native-window recipe,
where the rate is bounded below 0.90 per thousand and prefix caching is declined anyway.

One scope note on the arms themselves: all three served the context edition's weights. That is
immaterial to this result, because the counter reads only scheduler-side host arrays
(`spec_sequence_masks_cpu` and `query_start_loc_cpu`), and batch composition is a function of the
scheduler and the flags rather than of which quantisation is loaded.

**A free mitigation, for anyone who cannot patch.** The token-budget mechanism needs a drafted
decode clamped to exactly one token: `scheduler.py:518` clamps `num_new_tokens` to the remaining
budget, and at a clamp of exactly 1 `scheduler.py:637-642` computes zero scheduled speculative
tokens, so `gpu_model_runner.py:2240-2249` marks a still-region-0 decode non-speculative. Within
a step the budget walks down in steps of `1 + num_spec`, so only a remainder of exactly 1 breaks
it — a remainder of 2, 3 or 4 truncates the drafts but keeps the request speculative — and the
walk is bounded by the number of decodes behind the prefill. That puts the rate at roughly
`max_num_seqs / max_num_batched_tokens` per prefill completion:

$$\text{rate per prefill completion} \;\approx\; \frac{\texttt{max\_num\_seqs}}{\texttt{max\_num\_batched\_tokens}}$$

So **raising** `--max-num-batched-tokens` narrows the triggering window proportionally and
**lowering** it widens it; `--mamba-cache-mode align` narrows it further by rounding chunks to
whole 1,600-token mamba blocks (`scheduler.py:547-550`). This is **not** a recommendation to
change our published `--max-num-batched-tokens 2048`, which was measured for prefill throughput
and memory: trading a measured setting against a mechanism we could not observe in 9027
gather-branch builds would be backwards.

GPU handover: this window took the card from `ShipPrefixCaching` at 06:09 UTC by name, stopped the
`qwen38-27b` unit, and restored it at the end — systemd active, podman health `healthy`,
`/health` 200, `/v1/models` = `Qwen3.8-27B`, GPU back to 30,449 MiB used / 1,702 free, matching
`receipts/aiboss-live-service-snapshot.json`'s `gpu_state_before_stop` to the MiB. Raw server
logs, probe JSONs, the counter dumps and the frozen prompts are preserved under
`receipts/gdn-gate-raw/` with digests in the receipt; nothing was deleted.
