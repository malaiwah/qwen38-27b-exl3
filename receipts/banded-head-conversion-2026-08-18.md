# Banded lm_head FP4 conversion + draft-head A/B (2026-08-18)

## Banded converter (LANDED, patches/exl3_fp4_conversion.py)
Unbanded conversion of the 5120x248320 head needs a 4.74 GiB fp32 fold temp
plus ~14 GiB quantizer transients -> always OOMed (why lm_head was skipped).
Banded path peaks at ~250 MB regardless of N:
- `ext.reconstruct_slice` decodes 128-aligned N-bands (reconstruct.cu:99-141:
  n_offset%128==0, band%128==0, n_offset in elements).
- Hadamard fold is exactly separable along 128-aligned N (Had_N block-diagonal
  128-point blocks, svh per-element) -> `hadamard_fold_weight_chunked` per band.
- Two-pass: pass 1 = global amax over folded bands; pass 2 = quantize each band
  against the single global scale (`_quantize_matrix_fp4_nvfp4` gained
  `global_scale_override`).
- Swizzled scale layout is row-block-major -> band storage is a contiguous
  slice at byte offset n0*cols_padded; bands write into the final buffer.
Result: `EXL3 FP4 draft head built (banded)... Δ0.67 GiB` (636MB packed +
79MB swizzled scales — exact design numbers).

Selftest (`VLLM_EXL3_FP4_BANDED_SELFTEST=1`, one-shot on first FP4 layer):
gs ref=6171.55 vs band=6175.01 (one fp16 ulp in amax — chunked vs full fold
fp32 reduction order), packed nibble mismatch 0.6974%, scale bytes 1.2% —
uniform rounding noise from the 1-ulp gs shift; a layout bug would show
50-90%. Layout verified correct.

Also fixed: draft-head build previously called `convert_all_shards_to_fp4`,
which stores `layer.fp4_weights` -> verify pass would have silently routed
through FP4 too. Banded variant does not store it; verify stays K6-exact.

## Draft-head A/B (NEGATIVE — default OFF)
8K ctx, FP8 KV, MTP=6, graphs; same prompts both sides:
| bench           | draft head ON | OFF   |
|-----------------|---------------|-------|
| TG(200, fox)    | 157.6         | 160.6 |
| TG(500, essay)  | 76.1          | 75.2  |
| acceptance      | 59.4%         | 58.2% |
No gain (within run-to-run noise ~±2%). Explanation consistent with earlier
B12X_N_RANGE finding: lm_head routes to exl3_gemm because the b12x FP4 GEMM
is not faster on 5120x248320 at M=1 (tile_m=64 -> 63 masked rows + act-quant
overhead) — the 0.28 GB/stream weight-traffic saving is eaten by kernel
inefficiency. A true win needs the 16-row FP4 GEMM tile (kernel work, see
decode-tile receipt). VLLM_EXL3_FP4_DRAFT_HEAD stays 0.

Note: when enabled, the head builds twice (target + MTP module) = 1.34 GiB;
dedupe via shared module would be required before any long-context use.
