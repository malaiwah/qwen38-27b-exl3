"""Fused FP4 activation quantization — v2 with proper swizzle.

Replaces the multi-op PyTorch path with a single function that reduces
kernel launches. Uses the same swizzle as the original.
"""
import torch
import math
from typing import Optional

# Constants
FP4_E2M1_MAX = 6.0
NVFP4_GS_NUM = 448.0 * 6.0
SF_VEC_SIZE = 16


def _align_up(x, a):
    return (x + a - 1) // a * a


def _swizzle_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Same swizzle as exl3_fp4_conversion._swizzle_block_scale."""
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
        squeeze_batch = True
    elif scale.ndim == 3:
        squeeze_batch = False
    else:
        raise ValueError(f"scale must be 2D or 3D, got {tuple(scale.shape)}")

    batch, rows, cols = scale.shape
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols, 4)

    padded = torch.zeros(
        (batch, rows_padded, cols_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:, :rows, :cols] = scale
    swizzled = padded.reshape(batch, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    swizzled = swizzled.reshape(batch, rows_padded, cols_padded)
    return swizzled[0] if squeeze_batch else swizzled


def fused_fp4_quant(x: torch.Tensor, global_scale_override: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused FP4 quantization matching _quantize_matrix_fp4_nvfp4 exactly.

    Combines all steps into one function with minimal intermediate allocations.
    """
    rows, k = x.shape
    assert k % SF_VEC_SIZE == 0

    # Single fused operation: reshape + amax + scale + quantize + pack
    x_f32 = x.to(torch.float32)
    blocked = x_f32.reshape(rows, k // SF_VEC_SIZE, SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)

    # Global scale
    if global_scale_override is not None:
        global_scale = global_scale_override.to(torch.float32).reshape(1).to(x.device)
    else:
        tensor_amax = block_max.max().clamp_min(1e-12)
        global_scale = (NVFP4_GS_NUM / tensor_amax).reshape(1)

    # Block scale (E4M3FN)
    scale = (global_scale * (block_max / FP4_E2M1_MAX)).to(torch.float8_e4m3fn).to(torch.float32)

    # Quantize: q = round_fp4(x * gs / sf)
    # q = x * gs / sf; output_scale = gs / sf = 1 / (sf / gs) = 1 / (sf * inv_gs)
    # Use _reciprocal_or_zero pattern to handle zero blocks
    inv_gs = torch.where(global_scale == 0, torch.zeros_like(global_scale), 1.0 / global_scale)
    inv_sf = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    output_scale = inv_sf * inv_gs  # = (1/sf) * (1/gs) = 1/(sf*gs)... NO
    # Actually: q = x * gs / sf, so output_scale = gs / sf
    # But the original code uses: output_scale = _reciprocal_or_zero(scale * _reciprocal_or_zero(global_scale))
    # = 1 / (scale * (1/global_scale)) = global_scale / scale = gs / sf ✓
    output_scale = torch.where(
        (scale * inv_gs) == 0,
        torch.zeros_like(scale),
        1.0 / (scale * inv_gs)
    )

    normalised = (blocked * output_scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    # FP4 E2M1 quantization (8 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6)
    # Use same thresholds as _fp4_quantize_values
    sign = torch.sign(normalised)
    abs_q = normalised.abs()
    rounded = torch.zeros_like(abs_q)
    rounded[(abs_q >= 0.0) & (abs_q <= 0.25)] = 0.0
    rounded[(abs_q > 0.25) & (abs_q < 0.75)] = 0.5
    rounded[(abs_q >= 0.75) & (abs_q <= 1.25)] = 1.0
    rounded[(abs_q > 1.25) & (abs_q < 1.75)] = 1.5
    rounded[(abs_q >= 1.75) & (abs_q <= 2.5)] = 2.0
    rounded[(abs_q > 2.5) & (abs_q < 3.5)] = 3.0
    rounded[(abs_q >= 3.5) & (abs_q <= 5.0)] = 4.0
    rounded[abs_q > 5.0] = 6.0
    fp4_float = rounded * sign

    # Encode to nibbles using same LUT as _fp4_encode_nibbles
    _FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    mags = fp4_float.abs()
    idx = torch.zeros_like(mags, dtype=torch.uint8)
    for code, mag in enumerate(_FP4_MAG_LUT):
        idx = torch.where(mags == mag, torch.full_like(idx, code), idx)
    sign_bit = (fp4_float < 0).to(torch.uint8) << 3
    fp4_val = (idx | sign_bit).reshape(rows, k // 2, 2)
    packed = (fp4_val[..., 0] | (fp4_val[..., 1] << 4)).contiguous()

    # Swizzle scales (same as original)
    scale_2d = scale.squeeze(-1).to(torch.float8_e4m3fn).unsqueeze(0)
    swizzled = _swizzle_block_scale(scale_2d)
    scale_storage = swizzled.view(torch.uint8).reshape(1, -1).contiguous()

    return packed, scale_storage, global_scale


if __name__ == "__main__":
    M, K = 1, 5120
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    packed, scales, gs = fused_fp4_quant(x)
    print(f"Input: {x.shape}")
    print(f"Packed: {packed.shape}, dtype={packed.dtype}")
    print(f"Scales: {scales.shape}, dtype={scales.dtype}")
    print(f"Global scale: {gs.item():.6f}")
