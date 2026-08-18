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
    x_ptr, packed_ptr, scale_ptr, gs_ptr, inv_gs_ptr,
    x_row_stride, x_col_stride,
    packed_row_stride, packed_col_stride,
    M: tl.constexpr, K: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 16-element blocks per program
):
    """Quantize (M, K) BF16 → packed FP4 + swizzled E4M3FN scales.

    Grid: (M, K // (BLOCK_K * 16)). Each program handles BLOCK_K consecutive 16-element blocks.
    """
    row = tl.program_id(0)
    kb = tl.program_id(1)  # which block of BLOCK_K 16-element chunks
    k_start = kb * BLOCK_K * 16

    # Load BLOCK_K * 16 elements
    offs = k_start + tl.arange(0, BLOCK_K * 16)
    x = tl.load(x_ptr + row * x_row_stride + offs * x_col_stride)
    x_f32 = x.to(tl.float32)

    # Global scale (pre-computed by caller, stored in gs_ptr)
    global_scale = tl.load(gs_ptr)
    inv_global_scale = tl.load(inv_gs_ptr)

    # Reshape to (BLOCK_K, 16) for per-block scale
    x_2d = tl.reshape(x_f32, (BLOCK_K, 16))
    block_max = tl.max(tl.abs(x_2d), axis=1)  # (BLOCK_K,)

    # Per-block E4M3FN scale
    scale_f32 = global_scale * block_max / 6.0
    scale_e4m3 = scale_f32.to(tl.float8e4nv)
    scale_rnd = scale_e4m3.to(tl.float32)

    # Quantize
    inv_sf = 1.0 / scale_rnd
    inv_sf = tl.where(scale_rnd == 0, 0.0, inv_sf)
    out_scale = inv_sf * global_scale  # (BLOCK_K,)
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
    # code = round(mag * 2) for mag in {0,0.5,1,1.5,2,3,4,6} → {0,1,2,3,4,5,6,7}
    # But 4*2=8→6, 6*2=12→7, so can't just multiply. Use comparisons.
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

    # Pack: 2 nibbles → 1 byte. Process as 1D then pair adjacent.
    # Use element-wise: byte[i] = nib[2*i] | (nib[2*i+1] << 4)
    # Reshape (BLOCK_K, 16) → (BLOCK_K, 8, 2), but 3D indexing not supported.
    # Instead: flatten to (BLOCK_K*16,), then use arange to pair.
    nib_flat = tl.reshape(nib, (BLOCK_K * 16,))
    # Create even/odd index patterns
    pair_idx = tl.arange(0, BLOCK_K * 8)  # (BLOCK_K*8,)
    even_idx = pair_idx * 2
    odd_idx = pair_idx * 2 + 1
    # Gather even and odd nibbles
    # Triton doesn't support arbitrary gather, but we can use tl.load on the
    # nibble array stored to a temporary... No, it's in registers.
    # Alternative: compute packed directly from 2D layout.
    # (BLOCK_K, 16) → each row has 8 pairs: (col0,col1), (col2,col3), ...
    # packed[r, p] = nib[r, 2p] | (nib[r, 2p+1] << 4)
    # Use arange over 8 pairs per row:
    pair_cols = tl.arange(0, BLOCK_K * 8)  # flattened (BLOCK_K*8,)
    # This is 1D, need to map to 2D. Use math:
    # row_in_block = pair_cols // 8, col_pair = pair_cols % 8
    # nib_idx = row_in_block * 16 + col_pair * 2  (even)
    #           row_in_block * 16 + col_pair * 2 + 1  (odd)
    row_in_blk = pair_cols // 8  # (BLOCK_K*8,)
    col_pair = pair_cols % 8     # (BLOCK_K*8,)
    even_2d = row_in_blk * 16 + col_pair * 2
    odd_2d = even_2d + 1
    # Load from reshaped 1D view
    nib_1d = tl.reshape(nib, (BLOCK_K * 16,))
    # Use tl.gather if available, else use indexing
    # Actually, in Triton we can't do arbitrary indexing into register tensors.
    # Let's use a different approach: store to SMEM and reload.
    # No — simplest: use 2D reshape to (BLOCK_K*8, 2) and reduce.
    # Try: reshape (BLOCK_K, 16) → (BLOCK_K*8, 2) then element-wise ops.
    nib_pairs = tl.reshape(nib, (BLOCK_K * 8, 2))  # (BLOCK_K*8, 2)
    # Get column 0 and 1 via masking
    col_arange = tl.arange(0, 2)[None, :]  # (1, 2)
    # This doesn't help. Let's use a different trick:
    # packed = (nib >> 0) & 0xF for even, (nib >> 4) for odd — no.
    # Actually the simplest: store nib to shared, reload with stride 2.
    # Or: just write the nibbles to a temporary buffer and read back packed.
    # That's too complex. Let me use the simplest approach that works:
    # Write all nibbles to a scratch buffer, then read packed.
    # Actually — Triton CAN do this if we treat the (BLOCK_K, 16) tensor
    # as having a known layout. Let me just compute packed bytes using
    # arithmetic on the 2D tensor.
    #
    # Key insight: in (BLOCK_K, 16), element [r, c] maps to byte [r, c//2].
    # packed[r, c//2] = nib[r, 2*(c//2)] | (nib[r, 2*(c//2)+1] << 4)
    #                   = nib[r, c_even] | (nib[r, c_even+1] << 4)
    # where c_even = 2 * (c // 2)
    #
    # But this is circular. Let me just use the reshape to (BLOCK_K, 8, 2)
    # and use tl.reduce or explicit indexing via broadcasting.
    #
    # Final approach: multiply by a mask matrix to extract even/odd.
    # No — let's just do it with a loop over 8 pairs per row using constexpr unroll.
    # Triton supports for loops with constexpr range.

    packed_byte = tl.zeros((BLOCK_K * 8,), dtype=tl.uint8)
    for p in tl.static_range(8):
        # nib has shape (BLOCK_K, 16)
        # even column = 2*p, odd column = 2*p+1
        # But we can't index columns in Triton 2D tensors either.
        # Use tl.load from the 1D view with computed offsets.
        pass

    # OK, let me try a completely different approach: write nibbles to
    # the output buffer directly, then pack in a second pass.
    # Actually, the simplest working approach in Triton:
    # Use tl.store to write nibbles to a scratch, then tl.load packed.
    # But we don't have a scratch buffer.

    # FINAL APPROACH: use the fact that Triton stores tensors in row-major
    # and we can use tl.reshape + arithmetic.
    # (BLOCK_K, 16) → (BLOCK_K, 8, 2) via reshape
    # Then: packed[r, p] = pairs[r, p, 0] | (pairs[r, p, 1] << 4)
    # Triton CAN do this if we use tl.view and element-wise ops on 3D.
    # The error was "unsupported tensor index" for nibble_pairs[:, :, 0].
    # Let's try using tl.split instead.
    # tl.split splits along the last dimension: returns tuple of (BLOCK_K, 8) tensors.
    nib_3d = tl.reshape(nib, (BLOCK_K, 8, 2))
    lo, hi = tl.split(nib_3d)  # each (BLOCK_K, 8)
    packed_byte = lo | (hi << 4)  # (BLOCK_K, 8)
    packed_flat = tl.reshape(packed_byte, (BLOCK_K * 8,))

    # Store packed codes
    packed_start = k_start // 2
    packed_offs = packed_start + tl.arange(0, BLOCK_K * 8)
    tl.store(packed_ptr + row * packed_row_stride + packed_offs * packed_col_stride, packed_flat)

    # Store swizzled scales: row r → swizzled_row = (r%32)*4 + (r//32)
    sw_row = (row % 32) * 4 + (row // 32)
    kb_start = k_start // 16
    scale_offs = kb_start + tl.arange(0, BLOCK_K)
    flat_idx = sw_row * SCALE_STRIDE + scale_offs
    scale_u8 = scale_e4m3.to(tl.uint8, bitcast=True)
    tl.store(scale_ptr + flat_idx, scale_u8)


def triton_fp4_quant(
    x: torch.Tensor,
    scale_storage: torch.Tensor,
    packed_out: torch.Tensor,
    gs_out: torch.Tensor,
    inv_gs_out: torch.Tensor,
) -> None:
    """Quantize (M, K) BF16 → NVFP4 packed + swizzled scales + global scale."""
    M, K = x.shape
    assert K % 16 == 0
    K_BLOCKS = K // 16
    SCALE_STRIDE = ((K_BLOCKS + 3) // 4) * 4

    # Compute global scale on host side (amax over full row)
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
