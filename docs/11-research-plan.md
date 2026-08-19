# Research plan: pushing every dimension past the 2026-08-18 frontier

Grounded in the roofline analysis in `receipts/peer-review-2026-08-18.md`
Section D. The headline finding that reframes everything:

> Prefill runs at **27% of its roofline** and decode at **24-31% of its
> roofline**, and in both cases the deficit is CPU/launch/small-kernel
> overhead — not quantized GEMM throughput. Kernel numerics are close to
> peak; the remaining 3-4x lives in the execution path.

Second finding, from per-layer attribution + the fp8dg result:

> Trellis W4A16 **decode** already beats FP4 W4A4 decode (83.5 vs 75 tok/s on
> prose). FP4's only value is prefill. If trellis prefill can be made fast,
> FP4 becomes unnecessary and KLD drops ~10x.

## 0. Prerequisite: instrumentation (2-3 days, blocks everything)

We cannot currently resolve the effects we are chasing. Most remaining levers
are 5-15%; our harness resolves maybe 5% and we have made 2-3% calls.

| item | what | why |
|---|---|---|
| P0.1 | `vllm bench serve`-based harness: concurrency sweep (1/2/4/8), TTFT/TPOT/ITL percentiles, **aggregate** throughput, prompt-length sweep (2k/8k/32k/128k/200k) | headline PP/TG are single-request and overhead-diluted; aggregate is the real serving figure and is entirely unmeasured |
| P0.2 | n>=3 boots per config, report mean±sd; log `clocks.sm,power.draw,temperature.gpu` during runs | separate signal from boot/thermal variance |
| P0.3 | nsys + torch-profiler trace of ONE prefill chunk and ONE decode step, layer/op attributed | directly localises the 130-210 ms non-GEMM prefill time and the 41 ms/step decode overhead. **Highest-information single action in the plan** |
| P0.4 | KLD harness: always emit ci95 + p95/p99/p999 + tail histogram; measure the replay-vs-live floor (audit gap G2); multi-shard means; `--score-from` sensitivity | we do not know our own resolution limit; mean-only reporting hid the p99=0.638 story |
| P0.5 | Profile registry (name -> env set -> expected numbers), boot-time log of the resolved profile, `tools/verify-profile.sh` regression suite (sanity, vision, long-ctx, MTP acceptance, PP/TG) diffed vs stored baseline | ~12 interacting env flags already caused one misattribution |
| P0.6 | Publish new captures/reports to the HF dataset with manifests | reproducibility parity with prior sessions |

## 1. The big swing: fused reconstruct -> UE8M0-FP8 kernel

**Prize: a config that beats today's flagship on all three axes at once,
using one weight copy.**

Today's all-trellis profile has TG 83.5/194 and KLD ~0.0047 but PP 1235-1591.
Trellis prefill is slow for a structural reason: `exl3_gemm` re-decodes the
weight for *every M-tile*, so an 8192-token chunk decodes each weight ~64
times. PR #316 (decode once + fp16 hgemm) reaches 5050. `reconstruct_fp8dg_nt`
+ DeepGEMM should beat that (FP8 math is 2x BF16), and we have it working.

The blocker is measured, not theoretical: all-layer *uncached* fp8dg gave
PP=1235. Cause hypothesis: the post-processing we bolted on in Python —
`requant_weight_ue8m0_inplace` + `transform_sf_into_required_layout` — runs 64
times per chunk over 5120x17408 tensors and dominates. The reconstruct kernel
(`reconstruct.cu:143-255`) *already* computes per-128x128 amax and emits e4m3 +
fp32 scales; it just emits the wrong scale format for SM120 DeepGEMM.

| step | action | cost | gate |
|---|---|---|---|
| 1.1 | **Stage-time the three phases separately** (reconstruct / requant / transform) on one MLP shape | 30 min | if post-processing is >60% of the time, proceed; else re-diagnose |
| 1.2 | Extend `reconstruct_fp8dg_nt` to emit **UE8M0 power-of-two scales directly in DeepGEMM's packed int32 layout** (new ext entry point, keep the old one) | 2-4 days | bit-compare vs the Python post-processed path |
| 1.3 | Wire all layers to fused-fp8dg prefill, trellis decode, **no cache** | 1 day | PP >= 5000 at 8k |
| 1.4 | KLD + tails + full matrix | 1 day | KLD <= 0.012 |

Expected end state: **PP ~6-9k, TG 83.5/194, KLD ~0.006-0.010** (attn-only
fp8dg measured +0.0009 on 16 layers; scaling to 64 layers additively gives
~+0.0036 over the 0.0047 all-trellis baseline — to be measured, not assumed),
one weight copy, context back toward native.

Important nuance for quality claims: KLD is measured on **prefill** hidden
states. In this hybrid, generation-time decode is trellis-**exact**, so the
prefill-measured KLD *overstates* the degradation a user experiences during
generation. Worth stating explicitly whenever this profile is published.

Also fold in here: **fast Walsh-Hadamard** instead of the dense `@ H` bmm. A
128-point Hadamard as a dense matmul costs 128 MACs/element; as a butterfly it
is 7 add/sub per element with no multiplies (~18x fewer ops). Plus fuse
(suh scale -> WHT -> per-token-group fp8 quant) into one kernel to remove 3
bf16 HBM round trips per layer per side.

## 2. Decode (TG): close the 3-4x

Ceiling 74.7 steps/s (= ~306 tok/s at essay-like acceptance); measured 18.3.
~41 ms/step of overhead across 7 sub-steps.

| # | lever | expected | cost | risk |
|---|---|---|---|---|
| 2.1 | **Capture the whole MTP draft loop in ONE CUDA graph** (6 autoregressive steps + sampling + acceptance), not one graph per step with Python between | up to **~2x TG** if overhead is per-sub-step as the arithmetic suggests | 1-2 wk (v2 `AutoRegressiveSpeculator`, static buffers) | med-high |
| 2.2 | **Draft vocab pruning**: draft head over top-32k tokens instead of 248,320 -> 0.12 GiB vs 0.91 per draft stream, saves ~4.7 GiB/step (~20% of step bytes). Verify keeps full vocab, so sampling stays exact | +10-18% TG | 3-5 d | low (needs acceptance-delta measurement) |
| 2.3 | **FP8 draft lm_head** via the fp8dg+banded machinery (0.45 vs 0.91 GiB, ~11% of step bytes). FP4 failed here because tile_m=64 wastes 63 rows at M=1; DeepGEMM behaves better at small M | +5-10% TG | 1-2 d | low |
| 2.4 | `num_spec` sweep (5/6/7/8) + adaptive-by-running-acceptance under the current stack | +0-8% | 1 d | none |
| 2.5 | 4-bit KV (below) — at 238k the KV read term dominates the step | large at long ctx | see 4.1 | high |

2.2 and 2.3 compound (both cut lm_head traffic); together ~25-30% of step
bytes.

## 3. Prefill (PP): attack overhead, not GEMMs

| # | lever | expected | cost | risk |
|---|---|---|---|---|
| 3.1 | **Concurrency + batching sweep** (max_num_batched_tokens 4096-16384, chunk 6144 known +8-9% at long ctx, max_num_seqs 1-8) | possibly large on aggregate; best value/cost in the plan | 1 d | none |
| 3.2 | **Prefill CUDA graphs / fewer op boundaries** — retry with memory freed by int6 embeds; or torch.compile on the prefill path only | 1.5-2.5x on the headline metric if launch-bound | 3-5 d | **Xid 31 hang history** — test at 8k, watch dmesg |
| 3.3 | Fused activation pre/post + WHT (shared with 1.x) | +5-12% | 2-3 d | low |
| 3.4 | Strided-C shard-cat (needs ext strided-store support; folded into an upstream ask) | +2-3% | med | low |
| 3.5 | CuTe JIT cubin reduction via compile-key M-bucketing (also frees 100-400 MB) | small PP, real memory | 1-2 d | low |

## 4. Memory / context

| # | lever | expected | cost | risk |
|---|---|---|---|---|
| 4.1 | **4-bit KV** (KIVI/KVQuant-style): 37.3 -> ~18.6 KB/token -> **~476k context** in the same 8.89 GiB | 2x context | high (needs 4-bit-KV attention kernel) | high |
| 4.2 | CPU KV offload / paging for cold blocks | 1M+ ctx with degraded cold TTFT | med | med |
| 4.3 | int6/FP4-banded lm_head (banded converter now exists): 0.91 -> 0.35-0.45 GiB | ~+12k tokens | 1 d | low |
| 4.4 | fp8dg cache size as an explicit PP<->context dial (cache hottest K layers only) | tunable | 1 d | low |
| 4.5 | **Vision at 8 MP** (currently engine-fatal OOM): tiled encode or vision-tower activation checkpointing | closes a real end-goal gap | med | med |

## 5. KLD / fidelity

| # | lever | expected | cost |
|---|---|---|---|
| 5.1 | Report tails + CIs everywhere; measure the replay floor | correctness of all future claims | P0.4 |
| 5.2 | **Held-out additivity test**: gate_up+down FP4 predicts 0.0416 | validates or kills the attribution model | 1 run |
| 5.3 | Measure `attrib-none` (all-trellis + int6 embeds) at lower util / fewer contexts | turns the inferred floor+embed term into a measurement | 1 run |
| 5.4 | **Layer-wise** (not group-wise) sensitivity via `VLLM_EXL3_FP4_LAYERS` regex per layer index; greedy/bisect ~10 runs instead of 64 | at equal memory, plausibly 0.0567 -> 0.02-0.03 | 1-2 d GPU |
| 5.5 | Capture post-rotation activation amax distributions; if outliers survive the Hadamard, add SmoothQuant-style per-channel scale migration folded into `suh` (zero runtime cost) | literature 20-40% W4A4 error reduction; may be subsumed by existing incoherence processing | 2-3 d |
| 5.6 | W4A8 (upstream #232 or DIY in CuTe DSL): activation error is the dominant term | KLD ~0.015-0.025 at ~FP8 speed | 1-2 wk DIY |
| 5.7 | Any published FP4 SKU should be quantized **direct from BF16** (0.0301) not converted from trellis (0.0567) | 1.9x better at the same bitrate | GPU day |

## 6. Upstream hygiene (standing directive)

- Fix `pre-run-check` on PR #436 and #437 (both FAILING; nothing merges red).
- Respond to the CodeRabbit auto-review on #436.
- Follow up issue #435 (no maintainer response).
- b12x #232/#233/#234 filed with measurements; offer to implement #232 (W4A8)
  ourselves if maintainers are cold — it is the highest-leverage kernel gap.
- New upstream candidate from this session: the **fused reconstruct ->
  UE8M0-FP8** entry point belongs in exllamav3, and the SM120 DeepGEMM
  UE8M0 requirement (raw fp32 scales -> NaN) is worth documenting for anyone
  wiring `reconstruct_fp8dg_nt`, which currently ships unused.

## 7. Phasing

- **Phase 0** (2-3 d): instrumentation. Gate: we can resolve 3% and we know
  where the time goes.
- **Phase 1** (1 wk): free/cheap wins — 3.1, 2.4, 5.2, 5.3, 1.1, 3.5, 4.3,
  banded-converter ulp proof, upstream CI fixes.
- **Phase 2** (2-4 wk): the big swings — fused fp8dg kernel (1.2-1.4), MTP
  draft-loop graph (2.1), draft vocab pruning (2.2), fused activation+WHT
  (3.3), layer-wise precision optimizer (5.4).
- **Phase 3** (research): prefill graphs (3.2), 4-bit KV (4.1), W4A8 (5.6),
  rotation refit (5.5), vision at 8 MP (4.5).

## 8. What would make us wrong

- P0.3 traces might show the overhead is *not* per-op launch cost but
  something unfixable in-process (e.g. driver-side submission), capping 2.1
  and 3.2.
- Xid 31 may be a hard hardware/driver constraint on graph-captured prefill.
- The fused-kernel prize depends on hypothesis 1.1; if reconstruct itself
  (not the post-processing) dominates, all-layer fp8dg stays dead and FP4
  prefill remains necessary.
- Unvalidated b12x tile paths (the 16-row FP4 tile) carry silent-NaN risk;
  any such work must be gated on bit-comparison, never on "it ran".
- 4-bit KV may cost more KLD at long context than the context is worth.
