# Goal Final Status — 2026-08-18

## Summary

| Lever | Code | Peer Review | Speed | KLD | Status |
|-------|------|-------------|-------|-----|--------|
| 1 (b12x gate) | ✅ pushed | ✅ PASS (0.93) | ❌ +1.0% (threshold +3%) | ⚠️ unmeasurable | FAIL speed |
| 2 (in_proj_ba transpose) | ✅ pushed | ✅ PASS | ✅ +4.4% (threshold +3%) | ⚠️ unmeasurable | PASS speed |
| 3 (fused kernel) | ✅ pushed | ✅ PASS (4 fixes) | ❌ infeasible (stop #3) | ✅ rel_diff < 0.4% | INFEASIBLE |
| Combined 1+2 | — | — | ✅ +5.4% | ⚠️ unmeasurable | PASS speed |

## Speed Results (local RTX 5090, C1 decode, 3×200 tokens)

- Baseline (both OFF): 142.5 tok/s
- Lever 2 only: 148.7 tok/s (+4.4%)
- Both levers: 150.2 tok/s (+5.4%)
- Lever 1 marginal: +1.0% (below +3% threshold)

## KLD Parity

Cannot be measured: the v5 suite, BF16 reference hidden states, and shared
lm-head weight are on the NFS vault (`/mnt/vault/`) which is down. The
fidelity tool (`tools/fidelity.py`) exists but has no data to run against.

For lever 3, per-GEMM parity was verified: rel_diff < 0.4% across 8 diverse
layer shapes (bits 4/5/6, K 5120-17408, N 1024-17408).

## Lever 3 (Fused Kernel) — Stop Condition #3

The fused dequant-in-epilogue GEMM kernel achieves:
- Numerical parity: ✅ (rel_diff < 0.4%)
- 2x speedup vs cooperative exl3_gemm at M≥128
- BUT: existing reconstruct+cuBLAS path is faster at M≥256
- The existing prefill strategy is already near-optimal
- A Marlin-scale persistent kernel would be needed (multi-month effort)

Report: `docs/48-lever3-fused-kernel-report.md`

## Artifacts Pushed

### malaiwah/vllm-voipmonitor (vLLM fork)
- `kernel-gap/b12x-gate-n-range` — lever 1 (b12x gate)
- `kernel-gap/tiny-n-mm-transpose` — lever 2 (in_proj_ba transpose)

### malaiwah/qwen38-27b-exl3 (research repo)
- `patches/exl3_gemm_prefill_v3.cu` — fused kernel outer + dispatch
- `patches/exl3_gemm_inner_prefill.cuh` — modified inner kernel
- `patches/test_benchmark.py` — multi-M parity + speed benchmark
- `patches/test_diverse.py` — diverse layer parity test
- `patches/test_vs_reconstruct.py` — v3 vs reconstruct benchmark
- `patches/test_ground_truth.py` — ground truth comparison
- `patches/build_full_ext.py` — build script
- `receipts/lever-ab-local-5090.json` — A/B test results
- `docs/48-lever3-fused-kernel-report.md` — lever 3 report

## What Would Complete the Goal

1. **Lever 1 speed**: Needs +3% at C1. Currently +1.0%. The b12x gate routes
   lm_head and k/v to exl3_gemm, but the per-call GPU savings are too small
   to observe end-to-end. Would need a different approach (e.g., fusing the
   Python dispatch, or a larger N-range gate).

2. **KLD parity**: Needs NFS vault to be back online to access the v5 suite,
   BF16 reference, and shared lm-head weight. Then run:
   ```
   python tools/fidelity.py replay --reference v5/reference/hidden-bf16 \
     --candidate <new_capture> --head v3/lm-head/weight.safetensors \
     --suite v5/suite/shard-0000 --out report.json
   ```

3. **Lever 3 speed**: Needs a Marlin-scale kernel (stop condition #3, infeasible
   with current toolchain).
