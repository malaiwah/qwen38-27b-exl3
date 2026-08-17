## Kernel-level follow-up from the filer: the speedup is confirmed and sharpened, the "+8.0 % end-to-end" is withdrawn and replaced by an Amdahl bound

Same card as the original filing (RTX PRO 6000 Blackwell SE). Three things changed: the speedup is now measured to a much finer resolution, the *shape of the problem* turned out to be narrower than described above, and the end-to-end claim in the body is retired.

### Method

`torch.cuda.Event` pairs around `CUDAGraph.replay()`. One replay contains **48 calls — one decode step's worth of GDN `in_proj_ba` — each against its own weight tensor** (48 × 983 KB = 47.2 MB, so no single weight stays hot for the whole measurement). **200 timed samples per arm per shape**, arms **interleaved sample-by-sample** so clock drift is common-mode. Warmup: 50 eager iterations of both arms, 3 pre-capture iterations on a side stream, then 20 untimed replays per arm; and because this card idles at 180 MHz SM / 405 MHz mem, 300 8192² bf16 matmuls run *before* any timing (2295 MHz at timing start, 2347 MHz at the end). Arms are the literal code paths: `F.linear(x, weight[96,5120])` versus `torch.mm(x, weight_kn[5120,96])`.

Resulting resolution: **relative SEM 0.012 %** (served arm, m=4) and **0.044 %** (patch arm).

### 1. The cliff exists only at m ∈ {2,3,4}

Per-call µs, mean of 200 samples × 48 calls, hot L2 and an L2-flushed cold bracket:

| m | served TN | patch NN | **k** | cold TN | cold NN | **k cold** |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | **4.72** | 6.30 | **0.75** ← *regression* | 4.33 | 6.18 | 0.70 |
| 2 | 20.08 | 5.59 | **3.60** | 30.63 | 6.63 | 4.62 |
| 3 | 20.08 | 5.58 | **3.60** | 30.60 | 6.64 | 4.61 |
| **4 (served)** | **20.08** | **5.61** | **3.58** | **30.59** | **6.68** | **4.58** |
| 5 | 2.99 | 2.71 | 1.10 | 4.05 | 3.61 | 1.12 |
| 6 | 3.03 | 2.62 | 1.16 | 4.06 | 3.51 | 1.16 |
| 8 | 2.66 | 2.63 | 1.01 | 4.06 | 3.67 | 1.10 |
| 16 | 2.88 | 2.82 | 1.02 | 4.08 | 4.01 | 1.02 |

At m=4: served mean **20.0826 µs** (median 20.084, stdev 0.0344), patch mean **5.6127 µs** (median 5.6147, stdev 0.0352). This **reproduces the earlier probe's 20.08 µs / 5.61 µs to three significant figures** with a different harness on a different day. The L2-flushed cold bracket returns **30.59 µs**, which brackets the **28 µs/call** in-model figure the 5.2 % share was computed from — so the share and the speedup are being read off the same regime.

So the framing in the body — "tiny-N tall-K is a bad TN problem" — is too broad. **The 4× penalty occupies a three-wide window in m, and nowhere else.** That window happens to be exactly the MTP verify width the served profile runs in (1 target + 3 draft tokens), which is why the op looked uniformly terrible in the trace.

### 2. The mechanism is a declined split-K, named from the profiler

| shape / arm | kernels selected | µs |
|---|---|--:|
| m=1, TN | `dot_kernel` + `reduce_1Block_kernel` (cuBLAS **GEMV**) | 3.66 + 1.35 |
| m=1, NN | same GEMV pair | 5.26 + 1.36 |
| m=2 / m=4, TN | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_`**`tn`**`_align8` — **and nothing else** | 20.3 / 20.1 |
| m=2 / m=4, NN | same family, `nn_align8` variant | 4.47 / 4.49 |
| m=2 / m=4, NN | **+ `cublasLt::splitKreduce_kernel`** | 1.21 / 1.21 |
| m=8, TN / NN | `nvjet_sm120_tst_mma_32x16x128_..._splitK_TNNN` / `_NNNN` | 2.24 / 2.33 |

**The fast arm has a split-K reduction kernel and the slow arm does not.** cuBLASLt declines to split K for the *TN* layout at N=96/K=5120, so one 16×16 tile walks all 5120 of K serially on effectively one SM's worth of work. Same CUTLASS template, one letter different in the layout tag, 4.5× the speed. This confirms the kernel attribution in the body independently, and it explains m≥5 for free: the modern `nvjet_sm120` TMA kernel splits K for *both* layouts, so the heuristic gap closes on its own.

### 3. The patch as committed regresses 33 % per call at m=1

`weight_kn` is registered for any `N≤128, K≥4096` CUDA weight and `apply` uses it unconditionally, with no row-count guard. At m=1 that is **6.30 µs instead of 4.72 µs** — because at m=1 cuBLAS runs a GEMV, and a GEMV wants the reduction dimension contiguous, which is the *current* layout. The served MTP profile is m=4 so nothing regresses today, but **the condition should be `m ≥ 2`** (or pick the layout per row count — both copies already exist, the memory is spent). Any deployment with speculative decoding disabled pays this today.

### 4. It is not free: the patch is the less accurate arm

Because the NN layout is the one cuBLASLt splits K for, the patch reduces **bf16** partials through `splitKreduce`. Relative error against an fp32 reference at m=4: **2.366e-03 for the patch vs 1.658e-03 for the served path**, max abs Δ of 1.0 on outputs of magnitude ≈72. Both are ~0.2 % and neither is alarming, but the patch is the worse of the two and the outputs are **not bit-identical**. A KLD spot-check is still owed before this ships.

### 5. The end-to-end claim: withdrawn, replaced with a bound

The body says **"+8.0 % single-stream decode alone (96.19 → 103.91 tok/s)"**. That number was measured inside **proot** (ptrace), which taxes every launch ioctl, and **it exceeds the Amdahl ceiling for its own kernel win** — so it cannot be the effect of this patch. Retracting it.

With component share `s` and speedup `k`, the post-speedup total is `(1−s) + s/k = 1 − s(1−1/k)`, so

```
gain = s(1 − 1/k) / (1 − s(1 − 1/k))
```

At `s = 0.052` (the 5.2 % share reported in this issue):

- **hot L2, k = 3.5781:** `1 − 1/k = 0.720527`; `x = 0.052 × 0.720527 = 0.037467`; `bound = 0.037467/0.962533 =` **+3.893 %**
- **cold weight, k = 4.5828:** `1 − 1/k = 0.781774`; `x = 0.040652`; `bound = 0.040652/0.959348 =` **+4.238 %**

| k | 1 − 1/k | s(1−1/k) | **bound** |
|--:|--:|--:|--:|
| 1.5 | 0.333333 | 0.017333 | **+1.764 %** |
| 2.0 | 0.500000 | 0.026000 | **+2.669 %** |
| 3.0 | 0.666667 | 0.034667 | **+3.591 %** |
| 4.4 | 0.772727 | 0.040182 | **+4.186 %** |

On a **wall-clock** basis the ceiling is lower still: Amdahl only accelerates GPU-busy time, and the profiled window is 23 % GPU-idle, so the component's share of the *step* is `0.052 × 0.77 = 0.040040`, giving a bound of **+3.231 %**.

Per-step saving as an independent route to the same place: `48 × (20.0826 − 5.6127) =` **694.6 µs** hot, `48 × (30.59 − 6.675) =` **1147.9 µs** cold — and the cold 1.15 ms/step matches the "~1.2 ms/step recovered" estimate derived independently from the 28 µs/call figure.

**Every percentage above is a ceiling. None was observed as a throughput delta.**

### 6. Addressing this issue's own falsifier

The body names the falsifier as "a bare-metal A/B showing the end-to-end gain < 1 %". That A/B ran: real docker, no proot, 1 GPU of an 8× host, 5 repeats/cell, 11 valid cells. Median **+1.44 %** per-request, mean +2.24 %, range −0.55 % … +8.94 % — against a **median within-arm CV of 2.84 %** (13.9–18.0 % in short-prompt cells). Sign test p = 0.065.

That is **not** a refutation, because the test could not resolve its own target. Paired power arithmetic, `n = (z₁₋α/₂ + z₁₋β)²(σ/δ)²`, α = 0.05 two-sided, power 0.80, σ = 2.84 %, `(1.959964 + 0.841621)² = 7.8489`:

| target effect δ | arithmetic | repeats/cell |
|---|---|--:|
| 4.24 % (the ceiling) | 7.8489 × (2.84/4.24)² = 3.5 | **4** |
| 1.50 % | 7.8489 × (2.84/1.50)² = 28.1 | **29** |
| 1.44 % (observed median) | 7.8489 × (2.84/1.44)² = 30.5 | **31** |

5 were run. A 4.58× kernel win, a ≤4.24 % ceiling, and an unresolvable +1.44 % median are **one result at three scales**. The falsifier as written cannot be evaluated by a harness whose noise floor is twice the quantity being measured — which is why the standing method is now: measure the kernel, publish the bound, and don't spend ladder time trying to observe an effect smaller than the harness CV.

### Provenance

The diff timed here is byte-identical to `git diff 3b35c04c6~1..3b35c04c6`, **sha256 `ef70bad524ad20eefe9b739de6234b81cb02efb3e3ed6e36ed9aa9ff38bed062`**. Note that the in-source comment on that commit still asserts "+8.0 % end-to-end single-stream decode"; that line needs the bound instead, and I'll fix it on the branch.

---
*Receipts: `receipts/kernel-amdahl-bound.json` (timings, arithmetic, MEASURED-vs-INFERENCE split), `receipts/kernel-amdahl-ba-raw.json`, `receipts/kernel-amdahl-kernel-names.json`, harness `tools/kernel-amdahl-ba.py`, write-up `docs/47` F13 — [qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3). Companion correction on #406.*
