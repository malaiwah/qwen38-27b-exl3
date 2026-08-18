"""Triton FP4 (NVFP4) activation quantizer for small M (decode).

Eliminates 128× padding waste from b12x TMA quantizer (M%128==0 requirement).
Output format matches b12x dense_gemm: packed (M,K//2) uint8 + swizzled E4M3FN scales.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

_SF_VEC_SIZE = 16
_NVFP4_GS_NUM = 448.0 * 6.0  # 2688.0


@triton.jit
def _fp4_quant_kernel(
    x_ptr,          # (M, K) bf16
    packed_ptr,     # (M, K//2) uint8
    scale_ptr,      # flat uint8 (E4M3FN as uint8), 128 * SCALE_STRIDE elements
    gs_ptr,         # (1,) float32 — global scale (pre-computed)
    inv_gs_ptr,     # (1,) float32 — 1/global_scale (pre-computed)
    x_row_stride, x_col_stride,
    packed_row_stride, packed_col_stride,
    M: tl.constexpr, K: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 16-element blocks per program
):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    k_start = kb * BLOCK_K * 16

    # Load BLOCK_K * 16 elements
    offs = k_start + tl.arange(0, BLOCK_K * 16)
    x = tl.load(x_ptr + row * x_row_stride + offs * x_col_stride)
    x_f32 = x.to(tl.float32)

    global_scale = tl.load(gs_ptr)

    # Reshape to (BLOCK_K, 16) for per-block scale
    x_2d = tl.reshape(x_f32, (BLOCK_K, 16))
    block_max = tl.max(tl.abs(x_2d), axis=1)  # (BLOCK_K,)

    # Per-block E4M3FN scale
    scale_f32 = global_scale * block_max / 6.0
    scale_e4m3 = scale_f32.to(tl.float8e4nv)
    scale_rnd = scale_e4m3.to(tl.float32)

    # Quantize: normalized = x * gs / sf
    inv_sf = 1.0 / scale_rnd
    inv_sf = tl.where(scale_rnd == 0, 0.0, inv_sf)
    out_scale = inv_sf * global_scale  # (BLOCK_K,) = gs / sf
    norm = x_2d * out_scale[:, None]  # (BLOCK_K, 16)
    norm = tl.clamp(norm, -6.0, 6.0)

    # Round to FP4 E2M1: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
    anorm = tl.abs(norm)
    sgn = tl.where(norm >= 0, 1.0, -1.0)
    mag = tl.zeros_like(anorm)
    mag = tl.where(anorm < 0.25, 0.0, mag)
    mag = tl.where((anorm >= 0.25) & (anorm < 0.75), 0.5, mag)
    mag = tl.where((anorm >= 0.75) & (anorm < 1.25), 1.0, mag)
    mag = tl.where((anorm >= 1.25) & (anorm < 1.75), 1.5, mag)
    mag = tl.where((anorm >= 1.75) & (anorm < 2.5), 2.0, mag)
    mag = tl.where((anorm >= 2.5) & (anorm < 3.5), 3.0, mag)
    mag = tl.where((anorm >= 3.5) & (anorm < 5.0), 4.0, mag)
    mag = tl.where(anorm >= 5.0, 6.0, mag)
    fp4v = mag * sgn  # (BLOCK_K, 16)

    # Encode to nibbles: magnitude code (0-7) | sign bit (8)
    afp4 = tl.abs(fp4v)
    nib = tl.zeros_like(afp4).to(tl.uint8)
    nib = tl.where(afp4 == 0.5, 1, nib)
    nib = tl.where(afp4 == 1.0, 2, nib)
    nib = tl.where(afp4 == 1.5, 3, nib)
    nib = tl.where(afp4 == 2.0, 4, nib)
    nib = tl.where(afp4 == 3.0, 5, nib)
    nib = tl.where(afp4 == 4.0, 6, nib)
    nib = tl.where(afp4 == 6.0, 7, nib)
    sbit = tl.where(fp4v < 0, 8, 0).to(tl.uint8)
    nib = nib | sbit  # (BLOCK_K, 16) uint8

    # Pack 2 nibbles per byte using tl.split on reshaped (BLOCK_K*8, 2) tensor
    nib_pairs = tl.reshape(nib, (BLOCK_K * 8, 2))
    lo, hi = tl.split(nib_pairs)
    packed_byte = lo | (hi << 4)  # (BLOCK_K * 8,)

    # Store packed codes
    packed_start = k_start // 2
    packed_offs = packed_start + tl.arange(0, BLOCK_K * 8)
    tl.store(packed_ptr + row * packed_row_stride + packed_offs * packed_col_stride, packed_byte)

    # Store swizzled scales: row r → swizzled_row = (r%32)*4 + (r//32)
    sw_row = (row % 32) * 4 + (row // 32)
    kb_start = k_start // 16
    scale_offs = kb_start + tl.arange(0, BLOCK_K)
    flat_idx = sw_row * SCALE_STRIDE + scale_offs
    scale_u8 = scale_e4m3.to(tl.uint8, bitcast=True)
    tl.store(scale_ptr + flat_idx, scale_u8)


def triton_fp4_quant(
    x: torch.Tensor,
    scale_storage: torch.Tensor,  # flat 1D uint8, size 128 * SCALE_STRIDE
    packed_out: torch.Tensor,     # (M, K//2) uint8
    gs_out: torch.Tensor,         # (1,) float32
    inv_gs_out: torch.Tensor,     # (1,) float32
) -> None:
    """Quantize (M, K) BF16 → NVFP4 packed + swizzled scales + global scale."""
    M, K = x.shape
    assert K % 16 == 0
    K_BLOCKS = K // 16
    SCALE_STRIDE = ((K_BLOCKS + 3) // 4) * 4

    # Compute global scale outside the kernel
    amax = x.abs().max().clamp_min(1e-12)
    gs = (_NVFP4_GS_NUM / amax).reshape(1).to(torch.float32)
    gs_out.copy_(gs)
    inv_gs_out.copy_(1.0 / gs)

    BLOCK_K = min(K_BLOCKS, 64)
    num_k_tiles = (K_BLOCKS + BLOCK_K - 1) // BLOCK_K

    grid = (M, num_k_tiles)
    _fp4_quant_kernel[grid](
        x, packed_out, scale_storage, gs_out, inv_gs_out,
        x.stride(0), x.stride(1),
        packed_out.stride(0), packed_out.stride(1),
        M=M, K=K, SCALE_STRIDE=SCALE_STRIDE, BLOCK_K=BLOCK_K,
    )
