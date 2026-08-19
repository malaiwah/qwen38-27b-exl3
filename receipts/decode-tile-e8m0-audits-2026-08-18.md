# FP4 decode tile + E8M0 audits (2026-08-18, read-only scouts)

## FP4 16-row decode tile: KERNEL WORK, not a config flip
- FP4 tile_m pinned to multiples of 64: can_implement rejects m%64!=0 for
  Float4E2M1FN (dense_gemm.py:4326-4333); default atom_shape (4,2,1) makes
  num_m_tiles=tile_m//64 (:793-803,887-888).
- The 2-warp (16,64)/(16,128) atom path exists but is MXFP8/FP6-only: the
  16-row SFA consumer (direct_sfa_prefix) is gated on b_tile_major, never set
  for FP4; SFA smem tile is always 128 rows (:717-718). Silent-NaN epilogue
  risk flagged at :4098-4105.
- Decode already bypasses the quantizer 128-pad via our Triton quant; the
  remaining decode waste is 63 masked GEMM rows.
- Experiment plan (needs GPU window): relax can_implement for (16,64), force
  FP4 plan_m==1 -> (16,64), JIT, bit-compare vs (64,64), bench M=1.

## E8M0/scale upcast audit: GEMM ALREADY OPTIMAL (item closed)
- NVFP4 GEMM: E4M3 block scales consumed by the block-scaled MMA in hardware
  (MmaMXF4NVF4Op m16n8k64, dense_gemm.py:866-870, 2222-2235); SF loaded once
  per pipeline stage in bulk (:2075,2271); alpha read once into a register
  (:1439) and fused into the epilogue (:2403,2622,2695). No separate scale
  kernel; no software upcast in the K-loop. E8M0 sites are MXFP8/MXFP4-only.
- Quantizer-side minor gaps (negligible, L1/smem-cached): per-block re-reads
  of global_scale (gmem) + its reciprocal (smem) (_kernel.py:413-414), and
  rcp(6.0) recomputed per block (:238). Not worth pursuing as a PP lever.
