
# 10k PP Goal — Session Findings (2026-08-18)

## Current State
- Baseline: PP=1650 tok/s, TG=101 tok/s (vllm-exl3-pr314.py patch)
- Container running with baseline, healthy on port 8000
- 34 commits on malaiwah/qwen38-27b-exl3@main

## Key Findings

### Theoretical analysis (roofline estimates, not confirmation)
- FP16 MMA-only bound: 4527 tok/s; observed 1650 is 36.4 % of that bound.
- FP6 MMA peak is 2× FP16 and FP4 peak 4×, but end-to-end speed does not
  inherit those multipliers because activation quantization, dequantization,
  launches and non-GEMM work remain.
- Flat PP over the tested M range localizes a fixed/per-layer cost; it does not
  by itself identify which component owns it.

### b12x trellis K6 baseline
- Observed throughput is ~36 % of the MMA-only bound.
- The original "two thirds is dequant ALU" split was inferred from the gap, not
  profiled, and is withdrawn.
- Warp-specialized prototype: 1022 tok/s; FP6 dense path: 1655 tok/s.

### FP6 Path (WORKING but no speed improvement)
- exl3_fp6_conversion.py: Hadamard fold + quantize_dense_weight_to_fp6
- CPU reconstruction avoids GPU OOM for large layers
- 1044/1578 tensors converted; large-tensor conversion temporaries OOMed.
  This is a dense model, so the earlier "MoE OOM" label was wrong.
- vllm-exl3-fp6.py: integrated into vLLM EXL3 patch

### FP4 Path (THE PATH TO 10k — not yet implemented)
- b12x dense_gemm supports float4_e2m1fn with mma_k=64 (4x throughput)
- b12x has quantization/nvfp4/ module for NVFP4 quantization
- FP4 weights: 0.5 bytes/weight (33% LESS than trellis K6's 0.75 bytes) — NO OOM expected
- FP4 scales: UE8M0 per 32 elements (MXFP4) or UE4M3 per 16 elements (NVFP4)
- Hadamard transforms redistribute outliers; they do not guarantee a Gaussian distribution.
- Cross-format error and KLD were unmeasured at this point; no "K6 ~0 %" baseline is valid.
- The completed implementation is preserved as `patches/exl3_fp4_conversion.py`.

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

## Next steps (historical)

1. Review and integrate `patches/exl3_fp4_conversion.py`.
2. Test FP4 correctness, PP, TG and paired KLD.
3. If KLD fails, test a pre-registered hybrid role map rather than tuning on the evaluation set.
4. Profile activation quantization before attributing any remaining PP gap.
5. Treat warp-specialized trellis→FP6 work as a separate, profiler-gated experiment.


### MXFP4 vs NVFP4 (Explorer Recommendation)
- USE MXFP4: UE8M0 scales (power-of-2), sf_vec_size=32, 0.53 bytes/weight
- vLLM's B12xMxFp4LinearKernel already uses: blockscaled.mm(ab_dtype="float4_e2m1fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32)
- Simplest quantization: reuse b12x._lib.fp6._ue8m0_scale_from_block_max(block_max, FLOAT4_E2M1_MAX=6.0)
- FP4 total weight: 12.2 GB (saves 3.7 GB vs K6 — no MoE OOM)
- Weight read: 6.8ms (vs 8.9ms for K6)
- Runtime: 3 ops per GEMM (activation quantize, blockscaled.mm, output scale)

### FP4 conversion artifact
- `patches/exl3_fp4_conversion.py` is the durable implementation.
- Its quantization and serving outcomes are recorded in the later 2026-08-18/19
  receipts and PROGRESS entries; the session-local agent URI is not durable
  evidence.
