# Reconstruct→fold→cuBLAS hgemm for the K5 matrices: +14.1% PP on the fidelity profile

**Date:** 2026-08-19
**Axis:** PP, on the profile that satisfies criteria 2-6
**Result:** PP **1628.5 ± 2.7 → 1857.7 ± 3.7** (+14.1%), TG-fox 210.1 → 210.5,
TG-essay 89.8 → 90.0, context 238,400 held, KLD 0.003407 → **0.003437**
(+0.9% point estimate, CIs overlapping), gate **9/9 exit 0**.

## Why there was anything left to do here

After B12X took the K6 matrices (`receipts/b12x-shared-scratch-2026-08-19.md`),
the **K5 `mlp.gate_proj`/`up_proj`** — the largest matrices in the model — were
still on the fused `exl3_gemm` trellis kernel, because `_b12x_trellis_k6_supported`
admits only 96-word payloads by default and routing K5 through B12X corrupts the
model for reasons unrelated to its GEMM
(`receipts/b12x-k5-parked-2026-08-19.md`).

`_exl3_gemm` already contains a faster route for M ≥ 128: `ext.reconstruct` to
FP16, Hadamard-fold, then `ext.hgemm` (cuBLAS). The fidelity profile had it
disabled (`VLLM_EXL3_PREFILL_RECONSTRUCT_M=0`) from before B12X existed, on the
grounds that its FP16 weight cache tried to hold ~21 GiB.

## Three failures, three distinct causes

1. **OOM allocating 2.37 GiB.** That is exactly `5120 × 248320 × 2` — the
   **lm_head**. The reconstruct route needs one full FP16 copy of the weight
   live, and for the head that is a single 2.37 GiB allocation. Reproduced at
   ctx 131,072 and util 0.93, so not a KV-headroom question.
2. **`ValueError: No available memory for the cache blocks`** once the head was
   excluded. The FP16 weight cache is populated **during vLLM's profiling
   forward**, so the profiler both reads an inflated peak and cannot count the
   cached bytes as free. A comment at that exact line claimed *"Skip caching
   during profiling (determine_available_memory)"* — **no such check existed**.
   The comment has been corrected to describe what the code actually does.
3. Fixed by two bounds, both opt-in so no existing profile changes:
   - `VLLM_EXL3_PREFILL_RECONSTRUCT_MAX_MB` (default **4096**, fidelity sets
     **512**) skips the route when the reconstructed weight would exceed the cap,
     keeping the lm_head on the fused kernel.
   - `VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE` (default **1**, fidelity sets **0**)
     stops cache population entirely, bounding the route to one 356 MB transient.

The throughput profile resolves to `recon=1 max=4096 cache=1` — byte-for-byte its
previous behaviour, verified by reading the effective values back out of the
launcher.

## Measured (n=3 boots, `receipts/bench-fidelity-recon-hgemm-2026-08-19.json`)

| metric | before | after | delta |
|---|---|---|---|
| PP, 2051-tok | 1628.5 ± 2.7 | **1857.7 ± 3.7** | **+14.1%** |
| TG-fox | 210.1 ± 0.3 [acc 1.000] | 210.5 ± 0.4 [acc 1.000] | +0.2% |
| TG-essay | 89.8 ± 0.1 [acc 0.281] | 90.0 ± 0.1 [acc 0.281] | +0.2% |
| max context | 238,400 | 238,400 | — |
| KV available | 9.30 GiB | 9.29 GiB | — |
| vision + MTP | pass | pass | — |

Every axis moved the right way or held; nothing regressed.

## Fidelity: measured, and reported honestly

`hgemm` accumulates differently from the fused trellis kernel, so bit-exactness
of the fold is **not** sufficient to claim fidelity neutrality — it was measured
(512 contexts, shard-0000, BF16 reference,
`/tmp/kld-data/reports/report-alltrellis-reconhgemm.json`):

| | mean | ci95 | p95 | p99 |
|---|---|---|---|---|
| B12X only | 0.003407 | [0.003167, 0.003673] | 0.010264 | 0.034823 |
| **+ recon→hgemm** | **0.003437** | [0.003196, 0.003706] | 0.010343 | **0.035204** |

The point estimate is **+0.88%** and the confidence intervals overlap almost
entirely, so this is *statistically indistinguishable* — but it is directionally
positive and it would be wrong to call it "identical". Criterion 3 (≤ 0.012) and
criterion 4 (p99 ≤ 0.15) both still pass with roughly 3.5× and 4× margin.

Trade summary: **+14.1% prefill for +0.9% KLD** on an axis with 3.5× of headroom.

## Does this rescue criterion 1?

No. 1857.7 is 3.8× short of 7000. Criterion 1 remains bounded by measurement:
the fastest trellis-exact arrangement of this model is 2457.7 tok/s
(`receipts/frontier-2026-08-19.md`), and the formats that clear 7000 cost
0.0638 KLD. What this does is cut the fidelity profile's prefill deficit for the
third time — 1080.6 → 1630.0 → **1857.7**, now a **4.1×** penalty versus the
throughput profile rather than the original 7×.

## Verification

- `tools/bench-profile.sh --boots 3` numbers above, acceptance reported with each
  TG figure.
- `tools/verify-profile.sh --baseline tools/baseline-fidelity.json` → **9/9
  PASS, exit 0**, including the 200k-token prompt and the vision fixture
  (`receipts/verify-fidelity-reconhgemm-2026-08-19.json`).
- KLD as tabulated; `tools/kld-run-recon-hgemm.sh` is the runner.
- `EXL3_PATCH_SHA256` re-pinned.
