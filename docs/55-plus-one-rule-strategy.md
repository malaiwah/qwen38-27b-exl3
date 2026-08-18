
# +1 Rule Multi-Precision Strategy (User Directive 2026-08-18)

## Core Principle
Trellis is a superior encoding (codebook + Hadamard transforms). To maintain KLD parity
when converting to tensor core formats, use the NEXT precision level UP:

| Trellis Bits | Target Precision | MMA Throughput | Rationale |
|---|---|---|---|
| K4 | FP6 (e2m3) | 2x (m16n8k32) | +2 bits compensates for trellis superiority |
| K5 | FP6 (e2m3) | 2x (m16n8k32) | +1 bit compensates, matches FP6 MMA |
| K6 | FP8 (e4m3) | 2x (m16n8k32) | +2 bits, FP8 MMA is 2x like FP6 |
| BF16 | BF16 | 1x | No conversion needed (embed/vision) |

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

## Why +1 and not same-bitrate?
- K6 trellis → FP6 (same bitrate): ~3% error from codebook→FP6 mapping
- K6 trellis → FP8 (+2 bits): ~1% error from codebook→FP8 mapping, BETTER fidelity
- K5 trellis → FP4 (-1 bit): ~8% error, TOO MUCH
- K5 trellis → FP6 (+1 bit): ~3% error, acceptable
- The +1 rule ensures the tensor core format has enough precision to compensate
  for the loss of trellis's codebook encoding

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
2. Eliminate Hadamard transforms (folded into weights at load time) ✓
3. Eliminate kernel launch overhead (CUDA graphs for prefill)
4. Increase MMA utilization from 33% to 80%+ (no trellis dequant ALU)

## Configurable Tradeoffs
The multi-precision approach should be configurable:
- VLLM_EXL3_MP_MODE=aggressive: K5→FP4 (4x), K6→FP6 (2x) — max speed, some KLD risk
- VLLM_EXL3_MP_MODE=balanced: K5→FP6 (2x), K6→FP8 (2x) — +1 rule, KLD safe (DEFAULT)
- VLLM_EXL3_MP_MODE=conservative: K5→FP6 (2x), K6→FP8 (2x), skip large layers — safest
- VLLM_EXL3_MP_MODE=off: use trellis (baseline)
