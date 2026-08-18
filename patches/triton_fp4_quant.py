"""Triton FP4 (NVFP4) activation quantizer for small M (decode).

The b12x TMA quantizer requires M%128==0, so M=1 decode pays for 128 rows
(128x waste). This kernel quantizes only the real rows with no padding.

Output format matches b12x dense_gemm expectations:
  - packed: (M, K//2) uint8, two FP4 E2M1 nibbles per byte
  - scale_storage: flat uint8, swizzled E4M3FN block scales
  - global_scale: (1,) float32
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SF_VEC_SIZE = 16
_FP4_E2M1_MAX = 6.0
_NVFP4_GS_NUM = 448.0 * 6.0  # 2688.0


@triton.jit
def _fp4_quant_kernel_tiled(
    x_ptr,          # (M, K) bf16
    packed_ptr,     # (M, K//2) uint8
    scale_ptr,      # flat uint8 (E4M3FN as uint8)
    gs_ptr,         # (1,) float32 — global scale output
    inv_gs_ptr,     # (1,) float32 — 1/global_scale
    x_row_stride,
    x_col_stride,
    packed_row_stride,
    packed_col_stride,
    M: tl.constexpr,
    K: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    K_TILE: tl.constexpr,    # number of 16-element blocks per K tile
):
    """Quantize (M, K) BF16 → packed FP4 + swizzled E4M3FN scales + global scale.

    Uses 2D grid: (M, K // (K_TILE * 16)). Each program handles K_TILE blocks.
    """
    row = tl.program_id(0)
    k_tile_id = tl.program_id(1)
    k_start = k_tile_id * K_TILE * 16  # starting element index

    # Load K_TILE * 16 elements
    elem_offsets = k_start + tl.arange(0, K_TILE * 16)  # (K_TILE * 16,)
    x = tl.load(x_ptr + row * x_row_stride + elem_offsets * x_col_stride)
    x_f32 = x.to(tl.float32)

    # --- Per-tensor amax (for first k_tile only, row 0 writes gs) ---
    abs_x = tl.abs(x_f32)
    block_amax = tl.max(abs_x)

    # For global scale: we need the full-row amax. Since we're tiled,
    # we compute amax per-tile and use atomic max. But for simplicity,
    # we pass the global scale from the caller (computed outside).
    # Thread (0, 0) doesn't write gs — caller provides it.

    # --- Per-block quantization ---
    # Reshape to (K_TILE, 16)
    x_blocks = tl.reshape(x_f32, (K_TILE, 16))

    # Per-block max-abs
    block_max = tl.max(tl.abs(x_blocks), axis=1)  # (K_TILE,)

    # Global scale is passed via gs_ptr (pre-computed by caller)
    global_scale = tl.load(gs_ptr)
    inv_global_scale = tl.load(inv_gs_ptr)

    # Per-block E4M3FN scale
    scale_f32 = global_scale * block_max / 6.0  # (K_TILE,)
    scale_e4m3 = scale_f32.to(tl.float8e4nv)
    scale_f32_rounded = scale_e4m3.to(tl.float32)

    # Quantize
    inv_scale = 1.0 / scale_f32_rounded
    inv_scale = tl.where(scale_f32_rounded == 0, 0.0, inv_scale)
    output_scale = inv_scale * global_scale  # (K_TILE,) = gs / sf
    normalized = x_blocks * output_scale[:, None]  # (K_TILE, 16)
    normalized = tl.clamp(normalized, -6.0, 6.0)

    # Round to FP4 E2M1 values
    abs_norm = tl.abs(normalized)
    sign = tl.where(normalized >= 0, 1.0, -1.0)

    mag = tl.zeros_like(abs_norm)
    mag = tl.where(abs_norm < 0.25, 0.0, mag)
    mag = tl.where((abs_norm >= 0.25) & (abs_norm < 0.75), 0.5, mag)
    mag = tl.where((abs_norm >= 0.75) & (abs_norm < 1.25), 1.0, mag)
    mag = tl.where((abs_norm >= 1.25) & (abs_norm < 1.75), 1.5, mag)
    mag = tl.where((abs_norm >= 1.75) & (abs_norm < 2.5), 2.0, mag)
    mag = tl.where((abs_norm >= 2.5) & (abs_norm < 3.5), 3.0, mag)
    mag = tl.where((abs_norm >= 3.5) & (abs_norm < 5.0), 4.0, mag)
    mag = tl.where(abs_norm >= 5.0, 6.0, mag)

    fp4_values = mag * sign  # (K_TILE, 16)

    # Encode to nibbles
    abs_fp4 = tl.abs(fp4_values)
    nibble = tl.zeros_like(abs_fp4).to(tl.uint8)
    nibble = tl.where(abs_fp4 == 0.5, 1, nibble)
    nibble = tl.where(abs_fp4 == 1.0, 2, nibble)
    nibble = tl.where(abs_fp4 == 1.5, 3, nibble)
    nibble = tl.where(abs_fp4 == 2.0, 4, nibble)
    nibble = tl.where(abs_fp4 == 3.0, 5, nibble)
    nibble = tl.where(abs_fp4 == 4.0, 6, nibble)
    nibble = tl.where(abs_fp4 == 6.0, 7, nibble)
    sign_bit = tl.where(fp4_values < 0, 8, 0).to(tl.uint8)
    nibble = nibble | sign_bit  # (K_TILE, 16) uint8

    # Pack 2 nibbles per byte: reshape to (K_TILE, 8, 2) then combine pairs
    nibble_pairs = tl.reshape(nibble, (K_TILE, 8, 2))  # (K_TILE, 8, 2)
    # even = first element, odd = second element of each pair
    # packed = even | (odd << 4)
    packed_byte = nibble_pairs[:, :, 0] | (nibble_pairs[:, :, 1] << 4)  # (K_TILE, 8)
    packed_flat = tl.reshape(packed_byte, (K_TILE * 8,))

    packed_start = k_start // 2
    packed_offsets = packed_start + tl.arange(0, K_TILE * 8)
    tl.store(packed_ptr + row * packed_row_stride + packed_offsets * packed_col_stride, packed_flat)

    # --- Store swizzled scales ---
    # Row r, col c → flat_index = ((r%32)*4 + r//32) * SCALE_STRIDE + c
    # For small M (r=0..3), swizzled_row = (r%32)*4 + 0 = r*4
    swizzled_row = (row % 32) * 4 + (row // 32)
    k_block_start = k_start // 16
    scale_col_offsets = k_block_start + tl.arange(0, K_TILE)
    flat_indices = swizzled_row * SCALE_STRIDE + scale_col_offsets
    scale_u8 = scale_e4m3.to(tl.uint8, bitcast=True)
    tl.store(scale_ptr + flat_indices, scale_u8)


def triton_fp4_quant(
    x: torch.Tensor,
    scale_storage: torch.Tensor,
    packed_out: torch.Tensor,
    gs_out: torch.Tensor,
    inv_gs_out: torch.Tensor,
) -> None:
    """Quantize (M, K) BF16 → NVFP4 packed + swizzled scales + global scale.

    Args:
        x: (M, K) BF16 input
        scale_storage: pre-allocated flat uint8 buffer for swizzled E4M3FN scales
        packed_out: pre-allocated (M, K//2) uint8 buffer for packed FP4 codes
        gs_out: pre-allocated (1,) float32 for global scale (pre-computed)
        inv_gs_out: pre-allocated (1,) float32 for 1/global_scale (pre-computed)
    """
    M, K = x.shape
    assert K % 16 == 0, f"K must be multiple of 16, got K={K}"
    K_BLOCKS = K // 16
    SCALE_STRIDE = ((K_BLOCKS + 3) // 4) * 4

    # Compute global scale outside the kernel (needs full-row amax)
    amax = x.abs().max().clamp_min(1e-12)
    gs = (_NVFP4_GS_NUM / amax).reshape(1).to(torch.float32)
    gs_out.copy_(gs)
    inv_gs_out.copy_(1.0 / gs)

    # Tile K to avoid huge register arrays
    # K_TILE = number of 16-element blocks per K tile
    K_TILE = min(K_BLOCKS, 64)  # 64 blocks = 1024 elements per tile
    num_k_tiles = (K_BLOCKS + K_TILE - 1) // K_TILE

    grid = (M, num_k_tiles)
    _fp4_quant_kernel_tiled[grid](
        x, packed_out, scale_storage, gs_out, inv_gs_out,
        x.stride(0), x.stride(1),
        packed_out.stride(0), packed_out.stride(1),
        M=M, K=K, SCALE_STRIDE=SCALE_STRIDE, K_TILE=K_TILE,
    )
