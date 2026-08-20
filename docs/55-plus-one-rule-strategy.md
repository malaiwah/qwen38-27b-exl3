
# +1 Rule Multi-Precision Strategy (User Directive 2026-08-18)

## Core Principle (unmeasured heuristic)
The proposed mapping gives tensor-core formats at least as many nominal bits
as the trellis width. It is not consistently a "+1" rule: K4→FP6 and K6→FP8
add two bits, while K5→FP6 adds one. Nominal bit count does not establish
equivalent error across formats.

| Trellis Bits | Candidate Precision | MMA Throughput | Status |
|---|---|---|---|
| K4 | FP6 (e2m3) | 2x (m16n8k32) | unmeasured |
| K5 | FP6 (e2m3) | 2x (m16n8k32) | unmeasured |
| K6 | FP8 (e4m3) | 2x (m16n8k32) | unmeasured |
| BF16 | BF16 | 1x | identity |

## Model Layer Assignment (from quantization_manifest.json)
- MLP gate_proj: K5 → FP6 (2x MMA)
- MLP up_proj: K5 → FP6 (2x MMA)  
- MLP down_proj: K6 → FP8 (2x MMA)
- Attention Q/K/V/O: K6 → FP8 (2x MMA)
- GDN in_proj/out_proj: K6 → FP8 (2x MMA)
- lm_head: K6 → FP8 (2x MMA, but skip if OOM)
- MTP draft: mixed K4/K5/K6 → FP6/FP6/FP8
- embed_tokens: BF16 → BF16 (no conversion)
- vision_tower: BF16 → BF16 (no conversion)

## Fidelity hypothesis
- Trellis-to-FP6/FP8 error percentages quoted in the original session note had
  no receipt and are retired.
- A wider floating-point format is a plausible lower-error candidate, not a
  guarantee. Each role mapping needs a paired KLD capture and tail check.
- The existing variable-width recipe motivates the experiment but does not
  validate cross-format equivalence.

## MMA Throughput Analysis
- FP6 (mxf8f6f4 m16n8k32): 2x FP16 throughput
- FP8 (mxfp8 m16n8k32): 2x FP16 throughput (same k=32)
- FP4 (mxf4nvf4 m16n8k64): 4x FP16 throughput
- With +1 rule: ALL layers get 2x MMA throughput (FP6 or FP8)
- Without +1 rule: K5→FP4 gives 4x but too much error, K6→FP6 gives 2x

## Theoretical with +1 rule
- All layers at 2x MMA: 4527 * 2 = 9054 tok/s theoretical
- At 80% efficiency: 7244 tok/s (close to 10k but not quite)
- At 100% efficiency: 9054 tok/s (still short of 10k)

## To reach 10k with +1 rule
Need to also optimize:
1. Eliminate activation quantization overhead (fused quantize+GEMM)
2. Fold only algebraically compatible Hadamard transforms into adjacent weights; transforms across nonlinearities remain runtime work
3. Eliminate kernel launch overhead (CUDA graphs for prefill)
4. Increase MMA utilization from 33% to 80%+ (no trellis dequant ALU)

## Configurable Tradeoffs
The multi-precision approach should be configurable:
- VLLM_EXL3_MP_MODE=aggressive: K5→FP4 (4x), K6→FP6 (2x) — max speed, some KLD risk
- VLLM_EXL3_MP_MODE=balanced: K5→FP6, K6→FP8 — candidate default only after KLD measurement
- VLLM_EXL3_MP_MODE=conservative: same formats on a smaller, explicitly named role set — lowest exposure
- VLLM_EXL3_MP_MODE=off: use trellis (baseline)
