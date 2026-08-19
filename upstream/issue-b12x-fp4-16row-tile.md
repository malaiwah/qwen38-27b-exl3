# Feature: 16-row (skinny-M) tile for the NVFP4 dense GEMM decode path

## Motivation
At M=1 decode the NVFP4 (W4A4) dense GEMM runs its minimum tile_m=64 — one
M-tile with 63 masked rows (epilogue `if m_coord < M`). The MXFP8/MXFP6 path
has a true 2-warp skinny tile ((16,64)/(16,128), atom_shape (1,2,1)) for
decode, but it is unavailable for FP4:

- `can_implement` rejects any FP4 tile with `mma_tiler_mn[0] % 64 != 0`
  (dense_gemm.py:4326-4333).
- The default FP4 atom_shape (4,2,1) makes `num_m_tiles = tile_m // 64`
  (dense_gemm.py:793-803, 887-888).
- The 16-row SFA consumer (`direct_sfa_prefix` / `direct_sfa_live16`,
  dense_gemm.py:778-793) is gated on `b_tile_major`, which is never set for
  FP4; the SFA smem tile is always 128 rows (`sfa_tile_shape_mk =
  (max(128, tile_m), tile_k)`, dense_gemm.py:717-718).

In a spec-decode serving setup (MTP), the lm_head and per-layer linears run
many M=1..7 GEMMs per step; the 64-row tile leaves ~7/8 of the MMA work
masked. We measured that an FP4 copy of a 5120x248320 lm_head is NOT faster
than a tuned W4A16 kernel at M=1 despite streaming 30% fewer weight bytes —
consistent with the tile inefficiency eating the bandwidth saving.

## Ask
Admit `(16,64)` / `(16,128)` FP4 tiles for small M (mirroring the MXFP8
skinny path): relax the `can_implement` gate, and wire the FP4 SFA smem
layout for the (1,2,1) atom (the epilogue `_legal` guard at
dense_gemm.py:4098-4105 flags the silent-NaN risk if the SFA path is wrong,
so this needs the 16-row SFA consumption path rather than just lifting the
gate). Happy to run bit-exactness (vs the (64,64) tile) and M=1..16 decode
benchmarks on SM120 / RTX 5090.
