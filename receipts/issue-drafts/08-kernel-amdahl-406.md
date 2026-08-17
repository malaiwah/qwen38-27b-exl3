## Correction from the filer: replacing this issue's "+3–15 %" end-to-end bracket with an Amdahl bound

The bare-metal confirmation promised in the body above has now run, and it **did not confirm the bracket** — it could not, and the reason is worth writing down because it changes how the claim in this issue should be read. Everything below is measured on the same card as the original filing (RTX PRO 6000 Blackwell SE) or on a 1×GPU slice of an 8× host.

### 1. The bare-metal A/B was inconclusive by construction

Real docker (no proot), single GPU, 5 repeats per cell, rungs C1–C8, three prompt shapes, factorial design byte-verified before launch:

| | valid cells | median Δ | mean Δ | range |
|---|--:|--:|--:|---|
| aggregate tok/s | 11 | **+0.98 %** | +1.41 % | −1.87 % … +8.94 % |
| per-request tok/s | 11 | **+1.44 %** | +2.24 % | −0.55 % … +8.94 % |

The number that governs all of those: **the median within-arm coefficient of variation is 2.84 %**, reaching 13.9–18.0 % in short-prompt cells. The noise floor is larger than the signal. Nine of eleven cells favour the gate (sign test **p = 0.065**) — weak evidence of a small positive effect and nothing more. The headline "+8.94 %" cell compares two overlapping samples whose baseline repeats alone span 70.5–93.6 tok/s.

**No magnitude is supportable from that run, and I am withdrawing the "+3–15 %" bracket rather than defending it.**

### 2. The gate's own per-call numbers already bound the end-to-end gain — below +3 %

The right move is not more repeats, it is arithmetic on the kernel deltas this issue already reports. Re-read at the served MTP verify width (m=4), from the same CUDA-event/CUDA-graph measurement on real hydrated K6 weights:

| shape | b12x µs | exl3_gemm µs | Δ/call | calls/step | Δ/step |
|---|--:|--:|--:|--:|--:|
| `lm_head` 5120×248320 | 710.07 | 652.06 | 58.01 | 4 | 232.04 µs |
| `attn.k_proj` 5120×1024 | 28.20 | 17.69 | 10.51 | 16 | 168.16 µs |
| `attn.v_proj` 5120×1024 | 28.07 | 17.71 | 10.36 | 16 | 165.76 µs |
| | | | | | **565.96 µs** |

Call counts are measured, not assumed: the 953.5 MB head runs once for the target and once per MTP depth (4×/step), and there are 16 full-attention layers with q/k/v serialized as separate trellises. Body pass only, so any draft-model attention k/v would only add saving — this is a conservative count. **0.566 ms/step independently reproduces the "~0.6 ms/step" already stated in this issue's body.**

Total b12x time in these three shapes is 3740.6 µs/step, so the effective speedup of the affected component is `3740.6 / (3740.6 − 565.96)` = **1.1783×**.

With component share `s` and speedup `k`, the post-speedup total is `(1−s) + s/k = 1 − s(1−1/k)`, so

```
gain = s(1 − 1/k) / (1 − s(1 − 1/k))
```

On a 26 ms measured step, `s_wall = 3740.6 / 26000 = 0.1439`, `1 − 1/1.1783 = 0.15132`, `x = 0.021771`:

```
bound = 0.021771 / 0.978229 = 0.02225  ->  +2.23 %
```

Equivalently `saved / (step − saved)` over the 23–28 ms step range gives **+2.06 % … +2.52 %**; on a decode-GPU-*busy* basis (the profiled window is 23 % GPU-idle) it is **+2.91 %**.

**So the gate's measured kernel deltas cap the end-to-end single-stream gain at roughly +2.2 %, which is below the +3 % floor of the bracket this issue published.** The **+15.4 %** I originally measured locally therefore cannot be kernel time. It is the removal of eager-Python dispatch for the 4 per-step `lm_head` calls, on a box where the container runs under **proot** (ptrace) and every launch ioctl is taxed. That saving is real *on that box* and is not portable, so it must not be quoted as a performance result.

### 3. The null result and the kernel win are the same fact at two scales

A ceiling of ~+2.2 % and an observation of +1.44 % ± 2.8 % are not in conflict — the observation's noise band spans the whole interval between zero and the ceiling. Power arithmetic, paired design (each cell its own control), `n = (z₁₋α/₂ + z₁₋β)²(σ/δ)²`, α = 0.05 two-sided, power 0.80, σ = 2.84 %, so `(1.959964 + 0.841621)² = 7.8489`:

| target effect δ | arithmetic | repeats/cell |
|---|---|--:|
| 2.23 % (this bound) | 7.8489 × (2.84/2.23)² = 12.7 | **13** |
| 1.44 % (observed median) | 7.8489 × (2.84/1.44)² = 30.5 | **31** |

**5 were run.** The A/B never had the resolution to see its own target. That is a design fault of mine, not evidence against the patch.

### 4. What this does *not* change

The per-call deltas in the original table stand — they were CUDA-event measured over CUDA graphs on real weights, and they are the whole basis of the report. **The gate still sends `lm_head` and the tiny k/v projections to a measurably slower kernel at decode row counts, and the one-clause fix still recovers 0.566 ms/step.** What changes is only the size of the claim attached to it: a bounded ~+2 %, stated as a ceiling, instead of an unbounded "+3–15 %".

### 5. Sibling finding, same method (issue #407)

Running the same treatment on the `in_proj_ba` transpose (#407) produced a result worth flagging here because it is the same class of bug: at the served m=4 shape the tiny-N tall-K linear is **3.58× (hot L2) to 4.58× (cold weight)** faster with the weight pre-transposed to K×N — 200 CUDA-event samples per arm, relative SEM 0.012 %. The mechanism, named from the profiler: the TN arm gets `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` **with no split-K reduction**, while the NN arm gets the same CUTLASS template's `nn_align8` variant *plus* a `cublasLt::splitKreduce_kernel`. cuBLASLt declines to split K for the TN layout at N=96/K=5120. And that cliff exists **only at m ∈ {2,3,4}** — at m=1 cuBLAS uses a GEMV and the current layout is *faster*, and from m=5 up the `nvjet_sm120` TMA kernel splits K for both layouts and the gap closes on its own.

### Standing method going forward

For any patch whose Amdahl ceiling sits below the harness's own coefficient of variation, **measure the kernel and publish the bound; do not spend ladder time trying to observe the effect.** Both numbers in this comment are bounds computed from measured kernel ratios and measured call counts. Neither was observed as a throughput delta and neither should be quoted as one.

---
*Receipts: `receipts/kernel-amdahl-bound.json` (the bound arithmetic and the MEASURED-vs-INFERENCE split), `receipts/kernel-gap-gemm-bandwidth.json` (the per-call deltas), `receipts/kernel-gap-bare-metal-ab.json` (the null result), `docs/47` F13 and `docs/46` §28 — all in the [qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3) research repo.*
