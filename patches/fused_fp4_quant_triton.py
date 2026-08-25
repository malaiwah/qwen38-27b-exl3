"""Triton-fused FP4 activation quantization — v3 with exact E4M3 scale."""
import torch
import triton
import triton.language as tl

FP4_E2M1_MAX = 6.0
NVFP4_GS_NUM = 448.0 * 6.0
SF_VEC_SIZE = 16


@triton.jit
def _fp4_amax_kernel(
    x_ptr, block_max_ptr,
    M, K: tl.constexpr, K_BLOCKS: tl.constexpr,
    x_stride_m, x_stride_k, bm_stride_m,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    k_start = pid_k * 16
    k_offs = k_start + tl.arange(0, 16)
    k_mask = k_offs < K
    x = tl.load(x_ptr + m_offs[:, None] * x_stride_m + k_offs[None, :] * x_stride_k,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
    block_max = tl.max(tl.abs(x), axis=1)
    tl.store(block_max_ptr + m_offs * bm_stride_m + pid_k, block_max, mask=m_mask)


@triton.jit
def _fp4_quant_kernel(
    x_ptr, nibble_ptr,
    sf_ptr,  # precomputed E4M3 block scales as float32
    inv_output_scale_ptr,  # precomputed output_scale = gs / sf
    M, K: tl.constexpr, K_BLOCKS: tl.constexpr,
    x_stride_m, x_stride_k,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    k_start = pid_k * 16
    k_offs = k_start + tl.arange(0, 16)
    k_mask = k_offs < K
    x = tl.load(x_ptr + m_offs[:, None] * x_stride_m + k_offs[None, :] * x_stride_k,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
    # Load precomputed output_scale for this block
    output_scale = tl.load(inv_output_scale_ptr + m_offs * K_BLOCKS + pid_k, mask=m_mask, other=0.0)
    normalised = x * output_scale[:, None]
    normalised = tl.clamp(normalised, -6.0, 6.0)
    abs_q = tl.abs(normalised)
    rounded = tl.where(abs_q <= 0.25, 0.0,
             tl.where(abs_q < 0.75, 0.5,
             tl.where(abs_q <= 1.25, 1.0,
             tl.where(abs_q < 1.75, 1.5,
             tl.where(abs_q <= 2.5, 2.0,
             tl.where(abs_q < 3.5, 3.0,
             tl.where(abs_q <= 5.0, 4.0, 6.0)))))))
    nibble = tl.where(rounded == 0.0, 0,
             tl.where(rounded == 0.5, 1,
             tl.where(rounded == 1.0, 2,
             tl.where(rounded == 1.5, 3,
             tl.where(rounded == 2.0, 4,
             tl.where(rounded == 3.0, 5,
             tl.where(rounded == 4.0, 6, 7)))))))
    # Only set sign bit for nonzero rounded values (matching _fp4_encode_nibbles behavior)
    is_negative = (normalised < 0) & (rounded > 0.0)
    nibble = tl.where(is_negative, nibble | 8, nibble)
    nibble_offsets = k_start + tl.arange(0, 16)
    tl.store(nibble_ptr + m_offs[:, None] * K + nibble_offsets[None, :],
             nibble, mask=m_mask[:, None] & (nibble_offsets[None, :] < K))


def fused_fp4_quant_triton(x: torch.Tensor):
    """Two-pass Triton-fused FP4 quantization with exact E4M3 scales."""
    M, K = x.shape
    assert K % 16 == 0
    K_BLOCKS = K // 16

    # Pass 1: per-block amax
    block_max = torch.empty(M, K_BLOCKS, dtype=torch.float32, device=x.device)
    BLOCK_M = min(16, M)
    grid1 = (triton.cdiv(M, BLOCK_M), K_BLOCKS)
    _fp4_amax_kernel[grid1](
        x, block_max, M, K, K_BLOCKS,
        x.stride(0), x.stride(1), block_max.stride(0), BLOCK_M=BLOCK_M,
    )

    # Compute global scale + E4M3 block scales + output_scale in Python
    tensor_amax = block_max.max().clamp_min(1e-12)
    global_scale = (NVFP4_GS_NUM / tensor_amax).reshape(1)

    # E4M3 block scale (exact hardware rounding)
    scale = (global_scale * (block_max / FP4_E2M1_MAX)).to(torch.float8_e4m3fn).to(torch.float32)
    # output_scale = gs / sf (with zero handling)
    output_scale = torch.where(
        (scale > 0) & (global_scale > 0),
        global_scale / scale,
        torch.zeros_like(scale)
    )

    # Pass 2: quantize with precomputed output_scale
    nibbles = torch.empty(M, K, dtype=torch.uint8, device=x.device)
    grid2 = (triton.cdiv(M, BLOCK_M), K_BLOCKS)
    _fp4_quant_kernel[grid2](
        x, nibbles, scale, output_scale,
        M, K, K_BLOCKS, x.stride(0), x.stride(1), BLOCK_M=BLOCK_M,
    )

    # Pack nibbles
    packed = nibbles.reshape(M, K // 2, 2)
    packed = (packed[:, :, 0] | (packed[:, :, 1] << 4)).contiguous()

    # Swizzle scales
    from fused_fp4_quant_v2 import _swizzle_block_scale
    scale_2d = scale.to(torch.float8_e4m3fn).unsqueeze(0)
    swizzled = _swizzle_block_scale(scale_2d)
    scale_storage = swizzled.view(torch.uint8).reshape(1, -1).contiguous()

    return packed, scale_storage, global_scale
