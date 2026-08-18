"""Triton FP4 (NVFP4) activation quantizer for small M (decode).

The b12x TMA quantizer requires M%128==0, so M=1 decode pays for 128 rows
(~128x waste, ~30us per linear × 256 linears = ~7.7ms/step). This kernel
quantizes only the real rows in a single kernel launch with no padding.

Output format matches b12x dense_gemm expectations:
  - packed: (M, K//2) uint8, two FP4 E2M1 nibbles per byte
  - scale_storage: flat uint8, swizzled E4M3FN block scales
  - global_scale: (1,) float32

The swizzle maps row 0, col c → flat_index = (c//4)*512 + (c%4) in the
128×cols_padded scale buffer. For M>1, the mapping generalizes:
  row r, col c → ((r%32)*4 + r//32) * cols_padded + (c//4)*4 + (c%4)
But for decode (M≤16), we only write the first M rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SF_VEC_SIZE = 16
_FP4_E2M1_MAX = 6.0
_NVFP4_GS_NUM = 448.0 * 6.0  # 2688.0

# FP4 E2M1 magnitude thresholds (same as _fp4_quantize_values)
# Quantize to nearest of {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
_FP4_THRESHOLDS = torch.tensor([
    0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
], dtype=torch.float32)

_FP4_CODES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)


@triton.jit
def _fp4_quant_kernel(
    x_ptr,          # (M, K) bf16
    packed_ptr,     # (M, K//2) uint8
    scale_ptr,      # (rows_pad * cols_pad) uint8 (E4M3FN as uint8)
    gs_ptr,         # (1,) float32 — global scale output
    inv_gs_ptr,     # (1,) float32 — 1/global_scale (for alpha computation)
    x_row_stride,   # stride in elements
    x_col_stride,
    packed_row_stride,
    packed_col_stride,
    M: tl.constexpr,
    K: tl.constexpr,
    K_BLOCKS: tl.constexpr,    # K // 16
    SCALE_STRIDE: tl.constexpr,  # cols_padded = align_up(K//16, 4)
    ROWS_PAD: tl.constexpr,     # align_up(M, 128) = 128 for small M
):
    """Quantize (M, K) BF16 → packed FP4 + swizzled E4M3FN scales + global scale.

    Single program handles all K blocks for all M rows.
    """
    row = tl.program_id(0)  # which M row (0 to M-1)

    # --- Load entire row ---
    col_offsets = tl.arange(0, K)
    x = tl.load(x_ptr + row * x_row_stride + col_offsets * x_col_stride)
    x_f32 = x.to(tl.float32)

    # --- Compute per-tensor amax ---
    abs_x = tl.abs(x_f32)
    amax = tl.max(abs_x)
    amax = tl.maximum(amax, 1e-12)

    # --- Global scale ---
    global_scale = _NVFP4_GS_NUM / amax
    inv_global_scale = 1.0 / global_scale

    # Thread 0 writes global scale
    if row == 0:
        tl.store(gs_ptr, global_scale)
        tl.store(inv_gs_ptr, inv_global_scale)

    # --- Per-block quantization ---
    # Process K_BLOCKS blocks of 16 elements each
    block_offsets = tl.arange(0, K_BLOCKS)  # (K_BLOCKS,)
    # Load 16 elements per block: create (K_BLOCKS, 16) grid
    elem_offsets = tl.arange(0, 16)  # (16,)
    # Gather: x_block[b, e] = x_f32[b*16 + e]
    # Use 2D indexing
    block_idx = tl.arange(0, K_BLOCKS)[:, None]  # (K_BLOCKS, 1)
    elem_idx = tl.arange(0, 16)[None, :]  # (1, 16)
    full_idx = block_idx * 16 + elem_idx  # (K_BLOCKS, 16)
    x_blocks = tl.load(x_ptr + row * x_row_stride + full_idx * x_col_stride)
    x_blocks_f32 = x_blocks.to(tl.float32)

    # Per-block max-abs
    block_max = tl.max(tl.abs(x_blocks_f32), axis=1)  # (K_BLOCKS,)

    # Per-block E4M3FN scale: sf = e4m3(gs * block_max / 6.0)
    scale_f32 = global_scale * block_max / _FP4_E2M1_MAX  # (K_BLOCKS,)
    # Cast to E4M3FN and back to f32 (simulates E4M3FN rounding)
    scale_e4m3 = scale_f32.to(tl.float8e4nv)
    scale_f32 = scale_e4m3.to(tl.float32)

    # Quantize: q = round_fp4(x * gs / sf)
    inv_scale = 1.0 / scale_f32  # (K_BLOCKS,)
    # Avoid division by zero
    inv_scale = tl.where(scale_f32 == 0, 0.0, inv_scale)
    output_scale = inv_scale * global_scale  # (K_BLOCKS,) = gs / sf
    normalized = x_blocks_f32 * output_scale[:, None]  # (K_BLOCKS, 16)
    normalized = tl.clamp(normalized, -_FP4_E2M1_MAX, _FP4_E2M1_MAX)

    # Round to FP4 E2M1 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # Use threshold-based rounding
    abs_norm = tl.abs(normalized)
    sign = tl.where(normalized >= 0, 1.0, -1.0)

    # Quantize magnitudes to nearest FP4 value
    mag = tl.zeros_like(abs_norm)
    # 0.0 for [0, 0.25)
    # 0.5 for [0.25, 0.75)
    # 1.0 for [0.75, 1.25)
    # 1.5 for [1.25, 1.75)
    # 2.0 for [1.75, 2.5)
    # 3.0 for [2.5, 3.5)
    # 4.0 for [3.5, 5.0)
    # 6.0 for [5.0, inf)
    mag = tl.where(abs_norm < 0.25, 0.0, mag)
    mag = tl.where((abs_norm >= 0.25) & (abs_norm < 0.75), 0.5, mag)
    mag = tl.where((abs_norm >= 0.75) & (abs_norm < 1.25), 1.0, mag)
    mag = tl.where((abs_norm >= 1.25) & (abs_norm < 1.75), 1.5, mag)
    mag = tl.where((abs_norm >= 1.75) & (abs_norm < 2.5), 2.0, mag)
    mag = tl.where((abs_norm >= 2.5) & (abs_norm < 3.5), 3.0, mag)
    mag = tl.where((abs_norm >= 3.5) & (abs_norm < 5.0), 4.0, mag)
    mag = tl.where(abs_norm >= 5.0, 6.0, mag)

    fp4_values = mag * sign  # (K_BLOCKS, 16)

    # Encode to nibbles: 3 bits for magnitude code + 1 bit for sign
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
    nibble = nibble | sign_bit  # (K_BLOCKS, 16) uint8

    # Pack 2 nibbles per byte: (K_BLOCKS, 8)
    # Even elements are low nibble, odd are high nibble
    nibble_flat = tl.reshape(nibble, (K_BLOCKS * 16,))
    # Pack pairs
    even = nibble_flat[0::2]  # (K//2,) — first of pair
    odd = nibble_flat[1::2]   # (K//2,) — second of pair
    packed_byte = even | (odd << 4)  # (K//2,) uint8

    # Store packed codes
    packed_offsets = tl.arange(0, K // 2)
    tl.store(packed_ptr + row * packed_row_stride + packed_offsets * packed_col_stride, packed_byte)

    # --- Store swizzled scales ---
    # Row r, col c → flat_index = (r%32)*4*SCALE_STRIDE + (r//32)*SCALE_STRIDE + (c//4)*4 + (c%4)
    # Actually, the swizzle is:
    #   padded[r, c] → swizzled[(r%32)*4 + r//32, c]
    # And swizzled is stored row-major in flat buffer.
    # For M≤128 (ROWS_PAD=128), the swizzled row for original row r is:
    #   swizzled_row = (r % 32) * 4 + (r // 32)
    # For r=0: swizzled_row = 0
    # For r=1: swizzled_row = 4
    # For r=32: swizzled_row = 1
    # etc.

    # But the flat storage is (ROWS_PAD, SCALE_STRIDE) row-major after contiguous()
    # So flat_index = swizzled_row * SCALE_STRIDE + c

    # Wait, I need to re-derive this. The swizzle does:
    # 1. padded (batch, 128, cols_pad) with data at [0, r, c]
    # 2. reshape (batch, 1, 4, 32, cols_pad//4, 4)
    #    r → dim2 = r//32, dim3 = r%32
    #    c → dim4 = c//4, dim5 = c%4
    # 3. permute(0, 1, 4, 3, 2, 5) → (batch, 1, cols_pad//4, 32, 4, 4)
    # 4. contiguous + reshape(batch, 128, cols_pad)
    #    new_r = dim3_new * 4 + dim4_new = (r%32)*4 + (r//32)
    #    new_c = dim2_new * 4 + dim5_new = (c//4)*4 + (c%4) = c
    # So flat_index = new_r * cols_pad + new_c = ((r%32)*4 + r//32) * SCALE_STRIDE + c

    swizzled_row = (row % 32) * 4 + (row // 32)
    scale_offsets = tl.arange(0, K_BLOCKS)
    flat_indices = swizzled_row * SCALE_STRIDE + scale_offsets
    # Cast scale to uint8 for storage
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
        gs_out: pre-allocated (1,) float32 for global scale
        inv_gs_out: pre-allocated (1,) float32 for 1/global_scale

    All buffers must be pre-allocated and zeroed (for padding rows).
    """
    M, K = x.shape
    assert K % 16 == 0, f"K must be multiple of 16, got {K}"
    K_BLOCKS = K // 16
    SCALE_STRIDE = ((K_BLOCKS + 3) // 4) * 4  # align_up(K//16, 4)
    ROWS_PAD = ((M + 127) // 128) * 128

    grid = (M,)
    _fp4_quant_kernel[grid](
        x, packed_out, scale_storage, gs_out, inv_gs_out,
        x.stride(0), x.stride(1),
        packed_out.stride(0), packed_out.stride(1),
        M=M, K=K, K_BLOCKS=K_BLOCKS,
        SCALE_STRIDE=SCALE_STRIDE, ROWS_PAD=ROWS_PAD,
    )
