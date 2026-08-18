
# 10k PP Goal — Session Findings (2026-08-18)

## Current State
- Baseline: PP=1650 tok/s, TG=101 tok/s (vllm-exl3-pr314.py patch)
- Container running with baseline, healthy on port 8000
- 34 commits on malaiwah/qwen38-27b-exl3@main

## Key Findings

### Theoretical Analysis (CONFIRMED)
- FP16 MMA: 4527 tok/s theoretical, 1650 actual = 33% efficiency
- FP6 MMA (mxf8f6f4 m16n8k32): 9055 tok/s theoretical (2x), needs 99% eff for 10k
- FP4 MMA (mxf4nvf4 m16n8k64): 18131 tok/s theoretical (4x), needs 55% eff for 10k ← PATH TO 10k
- Prefill is COMPUTE-BOUND (not memory-bound) for M>41
- PP is flat at ~1700 from M=800 to M=12000 — bottleneck is per-layer overhead, not tile size

### b12x trellis K6 is already very well optimized
- 33% MMA utilization — 2/3 of time spent on trellis dequant ALU (codebook decode, bitstream extraction)
- Warp-spec kernel (our custom): 1022 tok/s — SLOWER than b12x trellis
- FP6 path (b12x dense_gemm): 1655 tok/s — same as trellis (activation quant overhead cancels 2x MMA gain)

### FP6 Path (WORKING but no speed improvement)
- exl3_fp6_conversion.py: Hadamard fold + quantize_dense_weight_to_fp6
- CPU reconstruction avoids GPU OOM for large layers
- 1044/1578 shards converted (MoE OOM'd — 4.74 GiB temporaries)
- Correctness: PASS, PP=1655, TG=105.8
- vllm-exl3-fp6.py: integrated into vLLM EXL3 patch

### FP4 Path (THE PATH TO 10k — not yet implemented)
- b12x dense_gemm supports float4_e2m1fn with mma_k=64 (4x throughput)
- b12x has quantization/nvfp4/ module for NVFP4 quantization
- FP4 weights: 0.5 bytes/weight (33% LESS than trellis K6's 0.75 bytes) — NO OOM expected
- FP4 scales: UE8M0 per 32 elements (MXFP4) or UE4M3 per 16 elements (NVFP4)
- Hadamard folding HELPS FP4 quantization (makes distribution more Gaussian)
- Risk: ~6-12% per-weight error vs K6's ~0%. KLD test essential.
- Fp4Conversion subagent was writing exl3_fp4_conversion.py — check agent://Fp4Conversion

### sm_120a Features
- compute_120a (not sm_120) unlocks: setmaxnreg, PDL, CLC, 256b loads, TMA
- TORCH_CUDA_ARCH_LIST="12.0a+PTX" for torch cpp_extension
- setmaxnreg: setmaxnreg.dec.sync.aligned.u32 %0 :: "n"(40) (CUTLASS syntax)
- PDL (griddepcontrol): caused TG regression when warp-spec shares stream with decode

### B12X JIT Fix
- Container has /usr/local/cuda-13.2/lib64 missing (read-only image)
- Fix: --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m + ln -sf in entrypoint
- Also: LIBRARY_PATH=/usr/local/cuda-13.2/targets/x86_64-linux/lib
- Also: TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions + persistent volume

## Next Steps (PRIORITY ORDER)
1. Check agent://Fp4Conversion for exl3_fp4_conversion.py
2. Integrate FP4 path into vllm-exl3-fp6.py (rename to vllm-exl3-fp4.py)
3. Test FP4: correctness, PP (target 10k), TG (target >=101), KLD
4. If KLD fails with FP4: try hybrid FP4/FP6 (FP4 for MLP, FP6 for attention)
5. If FP4 PP < 10k: optimize activation quantization (remove per-row scaling overhead)
6. If still < 10k: try warp-spec with trellis→FP6 dequant fused in producer warps


### MXFP4 vs NVFP4 (Explorer Recommendation)
- USE MXFP4: UE8M0 scales (power-of-2), sf_vec_size=32, 0.53 bytes/weight
- vLLM's B12xMxFp4LinearKernel already uses: blockscaled.mm(ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32)
- Simplest quantization: reuse b12x._lib.fp6._ue8m0_scale_from_block_max(block_max, FLOAT4_E2M1_MAX=6.0)
- FP4 total weight: 12.2 GB (saves 3.7 GB vs K6 — no MoE OOM)
- Weight read: 6.8ms (vs 8.9ms for K6)
- Runtime: 3 ops per GEMM (activation quantize, blockscaled.mm, output scale)

### Fp4Conversion Subagent
- Was writing exl3_fp4_conversion.py — check agent://Fp4Conversion for results
- If incomplete, write it manually using MXFP4 path (option b from explorer):
  1. Reuse hadamard_fold_weight from exl3_fp6_conversion.py
  2. Quantize to FP4: _ue8m0_scale_from_block_max + nearest_fp4 lookup
  3. Pack 2 codes per byte
  4. Call dense_gemm(ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32)
