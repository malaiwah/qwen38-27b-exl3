# Goal Final Status — 2026-08-18

## Summary

| Lever | Code | Peer Review | Speed | Numerical check | Status |
|-------|------|-------------|-------|-----------------|--------|
| 1 (b12x gate) | pushed | reviewed | +1.0% (threshold +3%) | full KLD unavailable then | FAIL speed |
| 2 (in_proj_ba transpose) | pushed | reviewed | +4.4% (threshold +3%) | full KLD unavailable then | PASS speed only |
| 3 (fused kernel) | pushed | reviewed (4 fixes) | slower than reconstruct+cuBLAS | rel_diff <0.4%, not parity | INFEASIBLE for speed target |
| Combined 1+2 | — | — | +5.4% | full KLD unavailable then | PASS speed only |

## Speed Results (local RTX 5090, C1 decode, 3×200 tokens)

- Baseline (both OFF): 142.5 tok/s
- Lever 2 only: 148.7 tok/s (+4.4%)
- Both levers: 150.2 tok/s (+5.4%)
- Lever 1 marginal: +1.0% (below +3% threshold)

## Fidelity status at this checkpoint in the chronology

The v5 suite/reference/head were unavailable on the local host, so levers 1
and 2 had no KLD qualification at this point. Later receipts must be read
before shipping either path.

Lever 3 passed only a per-GEMM relative-difference tolerance (<0.4% across
eight shapes). That is not numerical parity or an end-to-end fidelity gate.

## Lever 3 stop condition

The prototype is ~2× faster than cooperative `exl3_gemm` at M≥128 but slower
than the actual reconstruct+cuBLAS path at M≥256. The result rejects this
prototype for the speed target; it does not prove the existing dispatch is
globally optimal.

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
