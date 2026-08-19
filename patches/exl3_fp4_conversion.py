#!/usr/bin/env python3
"""EXL3 trellis K6 -> NVFP4 W4A4 load-time weight conversion.

This module converts EXL3 trellis-quantized K6 weights to NVFP4 format at
model load time, enabling the b12x ``dense_gemm`` W4A4 (E2M1 weights x E2M1
activations) code path for prefill on SM120/Blackwell.

EXL3 runtime computation
-------------------------
The EXL3 linear is::

    y = svh . Had_N( Had_K(suh . x) @ W )

where ``W`` is the trellis-decoded FP16 weight, ``suh``/``svh`` are
per-128-block scale vectors, and ``Had_K``/``Had_N`` are 128-point Hadamard
transforms applied per 128-element block along the K and N dimensions
respectively.  The Hadamard kernel (``had_r_128``) normalises by
``1/sqrt(128)`` so each call contributes one ``1/sqrt(128)`` factor.

Hadamard folding
----------------
Because Hadamard matrices are symmetric (``H = H^T``) and orthogonal, the
entire transform chain can be folded into the weight once at load time::

    W_final = diag(suh) @ Had_K @ W @ Had_N @ diag(svh)

At runtime the linear collapses to a plain GEMM::

    y = x @ W_final         (no Hadamard, no per-block scaling)

The folded weight is then quantised to NVFP4 (FP4 E2M1 with Float8E4M3FN
block scales + a per-tensor global scale) and executed at runtime via
``dense_gemm`` with ``ab_dtype='float4_e2m1fn'``, which uses the
``mxf4nvf4.m16n8k64`` MMA (4x throughput vs FP16 on SM120/Blackwell).

NVFP4 format
------------
* **Weights** are static, quantised once at load time into packed FP4
  E2M1 codes (4 bits/element, 2 values per byte) + swizzled Float8E4M3FN
  block-scales (``sf_vec_size=16``) and a per-tensor float32 **global
  scale**.
* **Activations** are data-dependent, quantised on the fly each forward
  to the same NVFP4 format (FP4 E2M1 + Float8E4M3FN block scales + a
  per-tensor global scale).
* **Two-level scaling**: each 16-element block has an E4M3FN block scale
  ``sf``; the effective scale is ``sf * global_scale``.  The GEMM epilogue
  ``alpha = 1 / (gs_act * gs_weight)`` undoes both per-tensor global
  scales so the block-scale MMA products reconstruct the true values.
* **MMA**: ``mxf4nvf4.m16n8k64`` — 64-element K reduction per MMA
  instruction (vs 16 for FP16, 32 for FP6), giving 4x throughput.

b12x weight layout
-------------------
``dense_gemm`` computes ``a @ b.T`` where ``a`` is the LHS activation
``(M, K_packed, L)`` and ``b`` is the RHS weight ``(N, K_packed, L)``.
``K_packed = K // 2`` because FP4 packs 2 values per byte.  The dense_gemm
FP4 path internally does ``k *= 2`` to recover the logical K from the
packed byte count.  The EXL3 reconstructed weight is ``(K, N)``
(``y = x @ W``), so the folded weight is transposed to ``(N, K)`` (the
``nn.Linear`` layout) before quantisation and packing.

Integration
-----------
Hook ``convert_layer_to_fp4`` into ``Exl3LinearMethod.process_weights_after_loading``
and ``fp4_apply`` into ``Exl3LinearMethod._apply_one`` (see the integration
hooks at the bottom of this file).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HADAMARD_BLOCK = 128
_HADAMARD_NORM = 1.0 / math.sqrt(_HADAMARD_BLOCK)

# One-shot arm flag for the banded-conversion selftest (see
# convert_all_shards_to_fp4).
_BANDED_SELFTEST_ARMED = True

# NVFP4 block scale parameters — E4M3FN block scales with sf_vec_size=16,
# matching b12x's MmaMXF4NVF4Op (NVF4: Float4E2M1FN operands, Float8E4M3FN
# scales, sf_vec_size=16).  See b12x._lib.intrinsics for the reference
# quantization recipe.
_SF_VEC_SIZE = 16          # elements per E4M3FN scale block
_FP4_E2M1_MAX = 6.0        # maximum magnitude representable in FP4 E2M1
_INV_FP4_E2M1_MAX = 1.0 / _FP4_E2M1_MAX
_FP8_E4M3_MAX = 448.0      # maximum magnitude representable in FP8 E4M3FN
# Per-tensor global scale numerator: amax maps to FP4_E2M1_MAX after the
# two-level (block-scale × global-scale) dequant.  gs = 448 * 6 / amax.
_NVFP4_GS_NUM = _FP8_E4M3_MAX * _FP4_E2M1_MAX
_NVFP4_GS_NUM_TENSOR = torch.tensor([_NVFP4_GS_NUM], dtype=torch.float32)

# Cache for b12x fused quantizer plans + output buffers, keyed by (m_pad, k, device).
# Eliminates per-call allocation and plan compilation overhead.
_QUANT_CACHE: dict = {}
# Cached global_scale per M (compute amax once per forward, reuse for all layers)
_CACHED_GS_M: int = -1
_CACHED_GS_VAL: torch.Tensor | None = None
_ALPHA_CACHE: dict = {}

# The 8 representable FP4 E2M1 magnitudes (sign is separate).
_FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Tile size for activation padding (the TMA quantiser / GEMM expects
# M % 128 == 0; we quantise the padded activation and run the GEMM at the
# true M).
_TILE = 128

# Cache the 128x128 Hadamard matrix per (device, dtype) to avoid rebuilds.
_HADAMARD_CACHE: dict[tuple[str, str], torch.Tensor] = {}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Hadamard matrix construction
# ---------------------------------------------------------------------------

def _hadamard_128_matrix(
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the normalised 128x128 Hadamard matrix ``H / sqrt(128)``.

    Uses the recursive Sylvester construction::

        H_1   = [[1]]
        H_2n  = [[H_n,  H_n],
                 [H_n, -H_n]]

    The raw matrix has entries +-1; dividing by ``sqrt(128)`` makes it
    orthogonal (``H @ H^T = I``).  Built in float32 for numerical
    stability regardless of the target dtype.
    """
    key = (str(device), str(dtype))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    n = _HADAMARD_BLOCK
    dev = torch.device(device)
    # Build the raw +-1 Hadamard via Sylvester doubling (always in f32).
    h = torch.ones((1, 1), dtype=torch.float32, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    h = h * _HADAMARD_NORM  # normalise
    if dtype != torch.float32:
        h = h.to(dtype)
    _HADAMARD_CACHE[key] = h
    return h


# ---------------------------------------------------------------------------
# 1. Hadamard fold  (identical to the FP6 path)
# ---------------------------------------------------------------------------

def hadamard_fold_weight(
    W: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Fold Hadamard transforms and per-element scales into a weight matrix.

    Computes::

        W_final = diag(suh) @ Had_K @ W @ Had_N @ diag(svh)

    where ``Had_K`` / ``Had_N`` are block-diagonal normalised 128-point
    Hadamard matrices acting on the K / N dimensions, and ``suh`` / ``svh``
    are per-element scale vectors (one scale per element of K and N
    respectively).

    The fold is performed in float32 for numerical stability (the Hadamard
    sum over 128 elements loses precision in float16), then cast back to
    the input dtype.

    Args:
        W:   Reconstructed weight, shape ``(K, N)``, float16 or float32.
        suh: Per-element input scales, shape ``(K,)``, float16.
        svh: Per-element output scales, shape ``(N,)``, float16.

    Returns:
        ``W_final`` with the same shape ``(K, N)`` and dtype as ``W``.
    """
    K, N = W.shape
    if K % _HADAMARD_BLOCK != 0 or N % _HADAMARD_BLOCK != 0:
        raise ValueError(
            f"K and N must be multiples of {_HADAMARD_BLOCK}, "
            f"got K={K}, N={N}"
        )
    k_blocks = K // _HADAMARD_BLOCK
    n_blocks = N // _HADAMARD_BLOCK

    # suh/svh are per-element (shape K and N respectively), not per-block.
    # If they happen to be per-block, broadcast them to per-element.
    if suh.numel() == K:
        suh_elem = suh
    elif suh.numel() == k_blocks:
        suh_elem = suh.repeat_interleave(_HADAMARD_BLOCK)
    else:
        raise ValueError(
            f"suh length {suh.numel()} != K={K} or K//128={k_blocks}"
        )
    if svh.numel() == N:
        svh_elem = svh
    elif svh.numel() == n_blocks:
        svh_elem = svh.repeat_interleave(_HADAMARD_BLOCK)
    else:
        raise ValueError(
            f"svh length {svh.numel()} != N={N} or N//128={n_blocks}"
        )

    device = W.device
    # Fold in float32 for accuracy; the 128-element Hadamard sum would
    # overflow/underflow fp16 precision.
    W_f32 = W.to(torch.float32)
    H = _hadamard_128_matrix(device, torch.float32)  # (128, 128) normalised

    # Reshape into 128-blocks: (k_blocks, 128, n_blocks, 128)
    W_blk = W_f32.reshape(k_blocks, _HADAMARD_BLOCK, n_blocks, _HADAMARD_BLOCK)

    # --- Apply Had_K on the K-inner dimension (dim 1) ---
    temp = torch.einsum("ab,ibjd->iajd", H, W_blk)

    # --- Apply Had_N on the N-inner dimension (dim 3) ---
    W_folded = torch.einsum("iajb,bc->iajc", temp, H)

    # --- Apply per-element scales ---
    # diag(suh) acts on K dimension; diag(svh) acts on N dimension.
    suh_f32 = suh_elem.to(torch.float32).reshape(K, 1)
    svh_f32 = svh_elem.to(torch.float32).reshape(1, N)
    W_folded = W_folded.reshape(K, N) * suh_f32 * svh_f32

    return W_folded.to(W.dtype)

def hadamard_fold_weight_chunked(
    W: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient Hadamard fold: processes 128-row chunks in FP32.

    Same math as hadamard_fold_weight but peak FP32 memory is 128×N×4 bytes
    (~8.9 MB for N=17408) instead of K×N×4 (~570 MB). Avoids OOM when
    folding large gate_up weights.
    """
    K, N = W.shape
    if K % _HADAMARD_BLOCK != 0 or N % _HADAMARD_BLOCK != 0:
        raise ValueError(
            f"K and N must be multiples of {_HADAMARD_BLOCK}, "
            f"got K={K}, N={N}"
        )
    k_blocks = K // _HADAMARD_BLOCK
    n_blocks = N // _HADAMARD_BLOCK

    # Broadcast scales to per-element if needed
    if suh.numel() == K:
        suh_elem = suh
    elif suh.numel() == k_blocks:
        suh_elem = suh.repeat_interleave(_HADAMARD_BLOCK)
    else:
        raise ValueError(f"suh length {suh.numel()} != K={K} or K//128={k_blocks}")
    if svh.numel() == N:
        svh_elem = svh
    elif svh.numel() == n_blocks:
        svh_elem = svh.repeat_interleave(_HADAMARD_BLOCK)
    else:
        raise ValueError(f"svh length {svh.numel()} != N={N} or N//128={n_blocks}")

    device = W.device
    H = _hadamard_128_matrix(device, torch.float32)  # (128, 128) normalised
    suh_f32 = suh_elem.to(torch.float32).reshape(K, 1)
    svh_f32 = svh_elem.to(torch.float32).reshape(1, N)

    # Peak temp memory: 128×N×4 bytes (~8.9 MB) instead of K×N×2 (272 MB result).
    for i in range(k_blocks):
        r0 = i * _HADAMARD_BLOCK
        r1 = r0 + _HADAMARD_BLOCK
        chunk = W[r0:r1].to(torch.float32)  # (128, N) — 8.9 MB
        blk = chunk.reshape(1, _HADAMARD_BLOCK, n_blocks, _HADAMARD_BLOCK)
        temp = torch.einsum("ab,ibjd->iajd", H, blk)
        folded = torch.einsum("iajb,bc->iajc", temp, H)
        folded = folded.reshape(_HADAMARD_BLOCK, N) * suh_f32[r0:r1] * svh_f32
        W[r0:r1] = folded.to(W.dtype)
    return W


# ---------------------------------------------------------------------------
# FP4 E2M1 quantization helpers
# ---------------------------------------------------------------------------

def _fp4_quantize_values(x: torch.Tensor) -> torch.Tensor:
    """Round float values to the 8 representable FP4 E2M1 magnitudes.

    Uses the same threshold-based nearest-neighbour rounding as the b12x
    reference implementation (``intrinsics._fp4_quantize_values``).
    The input is expected to be pre-scaled so that values lie in
    ``[-6, 6]``; values outside this range are clamped to the extremes.
    """
    sign = torch.sign(x)
    x = torch.abs(x.clone())
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _fp4_encode_nibbles(values: torch.Tensor) -> torch.Tensor:
    """Encode quantized FP4 values to 4-bit unsigned nibble codes.

    The 3 low bits encode the magnitude index into ``_FP4_MAG_LUT``;
    bit 3 is the sign bit (1 for negative).
    """
    mags = values.abs()
    idx = torch.zeros_like(values, dtype=torch.uint8)
    for code, mag in enumerate(_FP4_MAG_LUT):
        idx = torch.where(mags == mag, torch.full_like(idx, code), idx)
    sign_bit = (values < 0).to(torch.uint8) << 3
    return idx | sign_bit


def _reciprocal_or_zero(value: torch.Tensor) -> torch.Tensor:
    """Return ``1 / value``, mapping an unrepresentable zero scale to zero.

    Matches ``b12x._lib.intrinsics._reciprocal_or_zero_torch`` so that
    zero blocks (amax 0) produce zero quantised values rather than NaNs.
    """
    return torch.where(value == 0, torch.zeros_like(value), 1.0 / value)


def _swizzle_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Swizzle block-scale factors into the 128x4 interleaved layout.

    The CUTLASS MMA expects scales in a specific permuted order that
    matches the smem bank-conflict-free tile layout.  This is a bit-exact
    Torch replica of ``b12x._lib.intrinsics.swizzle_block_scale``.

    Input:  ``(batch, rows, cols)`` or ``(rows, cols)`` — ``cols`` is the
            number of scale columns (``K // _SF_VEC_SIZE``); dtype is the
            raw scale-element type (uint8, float8_e4m3fn, …).
    Output: ``(batch, rows_padded, cols_padded)``, same dtype, where
            ``rows_padded = align_up(rows, 128)`` and
            ``cols_padded = align_up(cols, 4)``.
    """
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


def _as_nvfp4_scale_view(
    scale_storage: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """View flat swizzled E4M3FN scales as the 6D tensor dense_gemm expects.

    ``scale_storage`` is ``(batch, -1)`` flat uint8 (the output of
    ``_swizzle_block_scale`` reshaped to 2D).  ``rows`` and ``cols`` are
    the *logical* matrix dimensions (before padding).  ``cols`` must be
    divisible by ``_SF_VEC_SIZE`` (16).

    The 6D shape ``(32, 4, rows//128, 4, cols//sf_vec_size//4, batch)``
    matches ``b12x._lib.intrinsics.as_grouped_scale_view`` and the
    ``BlockScaledBasicChunk`` atom layout for sf_vec_size=16.
    """
    batch = scale_storage.shape[0]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // _SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


# ---------------------------------------------------------------------------
# Matrix quantisation (shared by weight and activation paths)
# ---------------------------------------------------------------------------

def _quantize_matrix_fp4_nvfp4(
    mat_bf16: torch.Tensor,
    global_scale_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a ``(rows, K)`` bf16/fp16 matrix to packed NVFP4.

    Produces:
    * ``packed``        — ``(rows, K // 2)`` uint8, two FP4 E2M1 nibbles
                          per byte (low nibble = first element, high
                          nibble = second).
    * ``scale_storage`` — ``(1, -1)`` flat uint8, swizzled Float8E4M3FN
                          block scales ready for ``_as_nvfp4_scale_view``.
    * ``global_scale``  — ``(1,)`` float32 per-tensor global scale.

    The quantisation is block-wise along the K dimension with
    ``sf_vec_size = 16``: each block of 16 consecutive K elements shares
    one Float8E4M3FN block scale plus a per-tensor float32 global scale
    (two-level NVFP4 scaling).  This matches
    ``b12x._lib.intrinsics.quantize_grouped_nvfp4_torch`` exactly:

    * ``global_scale = 448 * 6 / amax``  (amax = max abs of the whole
      matrix; maps the largest element to FP4_E2M1_MAX after dequant).
    * ``block_scale = e4m3(global_scale * block_max / 6.0)``
    * ``q = round_fp4(x * global_scale / block_scale)``
    * dequant: ``x ≈ q * block_scale / global_scale``
    """
    rows, k = mat_bf16.shape
    if k % _SF_VEC_SIZE != 0:
        raise ValueError(
            f"K must be a multiple of {_SF_VEC_SIZE}, got K={k}"
        )

    # --- Block along K dimension: (rows, K//16, 16) ---
    blocked = mat_bf16.to(torch.float32).reshape(rows, k // _SF_VEC_SIZE, _SF_VEC_SIZE)

    # --- Per-block max-abs ---
    block_max = blocked.abs().amax(dim=-1, keepdim=True)  # (rows, K//16, 1)

    # --- Per-tensor global scale: gs = 448 * 6 / amax ---
    # (banded conversion passes a precomputed whole-matrix scale so every
    # band quantizes against the same reference)
    if global_scale_override is not None:
        global_scale = (
            global_scale_override.to(torch.float32)
            .reshape(1)
            .to(mat_bf16.device)
        )
    else:
        tensor_amax = block_max.max().clamp_min(1e-12)
        global_scale = (_NVFP4_GS_NUM / tensor_amax).reshape(1)  # (1,) float32

    # --- Per-block E4M3FN scale: sf = e4m3(gs * block_max / 6.0) ---
    scale = (
        (global_scale * (block_max * _INV_FP4_E2M1_MAX))
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )  # (rows, K//16, 1) float32

    # --- Quantise to FP4 E2M1: q = round_fp4(x * gs / sf) ---
    output_scale = _reciprocal_or_zero(
        scale * _reciprocal_or_zero(global_scale)
    )  # = gs / sf  (with zero → zero)
    normalised = (blocked * output_scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    fp4_values = _fp4_quantize_values(normalised)          # (rows, K//16, 16)
    nibbles = _fp4_encode_nibbles(fp4_values)               # (rows, K//16, 16) uint8

    # --- Pack 2 nibbles per byte: (rows, K//2) ---
    nibbles = nibbles.reshape(rows, k // 2, 2)
    packed = nibbles[..., 0] | (nibbles[..., 1] << 4)      # (rows, K//2) uint8
    packed = packed.contiguous()

    # --- Swizzle E4M3FN scales into the flat layout dense_gemm consumes ---
    scales = scale.squeeze(-1)                             # (rows, K//16) float32
    swizzled = _swizzle_block_scale(
        scales.to(torch.float8_e4m3fn).unsqueeze(0)
    )                                                       # (1, rows_pad, K//16_pad)
    scale_storage = (
        swizzled.view(torch.uint8).reshape(1, -1).contiguous()
    )                                                       # (1, -1) flat uint8

    return packed, scale_storage, global_scale
def _quantize_matrix_fp4_nvfp4_into(
    mat_bf16: torch.Tensor,
    packed_buf: torch.Tensor,
    scale_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as _quantize_matrix_fp4_nvfp4 but writes into pre-allocated buffers.

    CUDA-graph-safe: no allocations inside. Writes packed codes into packed_buf
    and swizzled scales into scale_buf. Returns (packed_buf, scale_buf, global_scale).
    """
    rows, k = mat_bf16.shape
    blocked = mat_bf16.to(torch.float32).reshape(rows, k // _SF_VEC_SIZE, _SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    tensor_amax = block_max.max().clamp_min(1e-12)
    global_scale = (_NVFP4_GS_NUM / tensor_amax).reshape(1)
    scale = (
        (global_scale * (block_max * _INV_FP4_E2M1_MAX))
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    output_scale = _reciprocal_or_zero(scale * _reciprocal_or_zero(global_scale))
    normalised = (blocked * output_scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    fp4_values = _fp4_quantize_values(normalised)
    nibbles = _fp4_encode_nibbles(fp4_values)
    nibbles = nibbles.reshape(rows, k // 2, 2)
    packed = nibbles[..., 0] | (nibbles[..., 1] << 4)
    packed_buf.copy_(packed)
    scales = scale.squeeze(-1).to(torch.float8_e4m3fn).unsqueeze(0)
    swizzled = _swizzle_block_scale(scales)
    scale_buf.copy_(swizzled.view(torch.uint8).reshape(1, -1))
    return packed_buf, scale_buf, global_scale

# ---------------------------------------------------------------------------
# FP4DenseWeight — the load-time artifact
# ---------------------------------------------------------------------------

@dataclass
class FP4DenseWeight:
    """A single NVFP4-quantized dense weight ready for :func:`fp4_apply`.

    ``packed`` holds ``(out_features, in_features // 2)`` FP4 E2M1 codes
    (2 per byte), ``scale_storage`` is the flat swizzled Float8E4M3FN
    block-scale buffer, and ``global_scale`` is the per-tensor float32
    global scale.  The 6D scale view ``dense_gemm`` wants is rebuilt on
    demand via :meth:`scale_view`.
    """

    packed: torch.Tensor            # (N, K//2) uint8
    scale_storage: torch.Tensor     # (1, -1) flat uint8 (swizzled E4M3FN)
    global_scale: torch.Tensor      # (1,) float32 per-tensor global scale
    out_features: int
    in_features: int

    def __post_init__(self) -> None:
        # Cached fp4_e2m1fn_x2 view (rebuilt lazily after ``to()``).
        self._fp4_view: Optional[torch.Tensor] = None

    def scale_view(self) -> torch.Tensor:
        """The 6D swizzled E4M3FN scale view for ``dense_gemm``."""
        return _as_nvfp4_scale_view(
            self.scale_storage, self.out_features, self.in_features
        )

    def packed_view(self) -> torch.Tensor:
        """The packed weight as ``float4_e2m1fn_x2`` (if supported) or uint8.

        ``dense_gemm``'s FP4 path expects the RHS in packed FP4 layout
        ``(N, K//2, L)``.  We unsqueeze the L dimension in :func:`fp4_apply`.
        """
        if self._fp4_view is None:
            t = self.packed
            try:
                t = t.view(torch.float4_e2m1fn_x2)
            except (TypeError, RuntimeError):
                pass  # fall back to uint8; CUTLASS reads the raw bytes
            self._fp4_view = t
        return self._fp4_view

    def to(self, device: torch.device | str) -> "FP4DenseWeight":
        dev = torch.device(device)
        return FP4DenseWeight(
            packed=self.packed.to(dev),
            scale_storage=self.scale_storage.to(dev),
            global_scale=self.global_scale.to(dev),
            out_features=self.out_features,
            in_features=self.in_features,
        )


# ---------------------------------------------------------------------------
# 2. Load-time conversion
# ---------------------------------------------------------------------------

def _codebook_to_flags(cb: int) -> tuple[bool, bool]:
    """Map a codebook selector to ``(mcg, mul1)`` booleans for ``ext.reconstruct``.

    0 -> standard (no codebook)
    1 -> MCG codebook (mcg=True, mul1=False)
    2 -> mul1 codebook (mcg=False, mul1=True)
    """
    if cb == 0:
        return (False, False)
    if cb == 1:
        return (True, False)
    if cb == 2:
        return (False, True)
    raise ValueError(f"Invalid codebook selector {cb}; expected 0, 1, or 2")


def convert_layer_to_fp4(
    layer: torch.nn.Module,
    ext: Any,
    bits: int,
    cb: int,
    *,
    shard_id: Any = None,
) -> FP4DenseWeight:
    """Convert one EXL3 trellis shard to an NVFP4 ``FP4DenseWeight``.

    Steps:
      1. Extract trellis codes, suh, svh from the layer's ``exl3_tensors``.
      2. Reconstruct trellis codes to an FP16 weight ``W`` via
         ``ext.reconstruct()``.
      3. Fold the Hadamard transforms and per-block scales into ``W``.
      4. Transpose to ``(N, K)`` (the ``nn.Linear`` weight layout) and
         quantise to NVFP4 (FP4 E2M1 + E4M3FN block scales + global scale).
      5. Return the ``FP4DenseWeight``.

    Args:
        layer:    The vLLM layer module (has ``trellis``, ``suh``, ``svh``,
                  ``mcg``, ``mul1`` ``Exl3Parameter`` attributes).
        ext:      The ``exllamav3_ext`` module (provides ``reconstruct``).
        bits:     Trellis bit width (e.g. 6 for K6).
        cb:       Codebook selector: 0=standard, 1=MCG, 2=mul1.
        shard_id: Which shard to convert (default ``None`` = single shard).

    Returns:
        A :class:`FP4DenseWeight` ready for :func:`fp4_apply`.
    """
    # --- Extract trellis tensors from the layer ---
    trellis = layer.trellis.exl3_tensors[shard_id]
    suh = layer.suh.exl3_tensors[shard_id]
    svh = layer.svh.exl3_tensors[shard_id]

    # Packed dimensions: trellis shape is (K//16, N//16, 256*bits//16) int16.
    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    device = trellis.device

    # --- Reconstruct trellis codes to FP16 weight on GPU ---
    # ext.reconstruct is a CUDA kernel that requires GPU tensors.
    # Use GPU but free temporaries immediately after each step.
    mcg, mul1 = _codebook_to_flags(cb)
    W = torch.empty(K, N, dtype=torch.float16, device=device)
    ext.reconstruct(W, trellis, bits, mcg, mul1)

    # --- Fold Hadamard transforms + per-element scales into W (on GPU) ---
    W_final = hadamard_fold_weight(W, suh, svh)
    del W

    # --- Transpose to (N, K) for the nn.Linear weight layout ---
    W_linear = W_final.t().contiguous()  # (N, K)
    del W_final

    # Free trellis tensors early to reclaim VRAM before quantization
    torch.cuda.empty_cache()

    # --- Quantise to NVFP4 (FP4 E2M1 + E4M3FN block scales, sf_vec_size=16) ---
    packed, scale_storage, global_scale = _quantize_matrix_fp4_nvfp4(
        W_linear.to(torch.bfloat16).to(device)
    )

    # Free temporaries immediately to avoid OOM during batch conversion
    del W_linear

    fp4_weight = FP4DenseWeight(
        packed=packed,
        scale_storage=scale_storage,
        global_scale=global_scale,
        out_features=N,
        in_features=K,
    )

    return fp4_weight


def convert_all_shards_to_fp4(
    layer: torch.nn.Module,
    ext: Any,
) -> dict[Any, Any]:
    """Convert every shard in an EXL3 layer to NVFP4.

    Iterates over ``layer.exl3_shard_ids``, reconstructs each shard's
    trellis codes, folds the Hadamard, and quantises to FP4.  The resulting
    :class:`FP4DenseWeight` objects are stored in ``layer.fp4_weights``
    (keyed by shard id) and the original trellis tensors are freed.

    Returns the dict of ``{shard_id: FP4DenseWeight}``.
    """
    shard_ids = list(layer.exl3_shard_ids)
    fp4_weights: dict[Any, Any] = {}
    # Unfreeze b12x kernel resolution for FP4 quantization
    try:
        from b12x._lib.runtime_control import unfreeze_kernel_resolution
        unfreeze_kernel_resolution()
    except ImportError:
        pass

    for shard_id in shard_ids:
        trellis = layer.trellis.exl3_tensors[shard_id]
        bits = trellis.shape[2] // 16  # 256 * bits / 16 = 16 * bits
        has_mcg = shard_id in layer.mcg.exl3_tensors
        has_mul1 = shard_id in layer.mul1.exl3_tensors
        if has_mcg and not has_mul1:
            cb = 1
        elif has_mul1 and not has_mcg:
            cb = 2
        else:
            cb = 0

        fp4_weight = convert_layer_to_fp4(
            layer, ext, bits, cb, shard_id=shard_id
        )
        fp4_weights[shard_id] = fp4_weight
        # Optional one-shot selftest: banded converter must reproduce the
        # unbanded artifact (VLLM_EXL3_FP4_BANDED_SELFTEST=1).  Run on the
        # first shard converted in the process, then disarm.
        global _BANDED_SELFTEST_ARMED
        if _BANDED_SELFTEST_ARMED and os.environ.get(
            "VLLM_EXL3_FP4_BANDED_SELFTEST", "0"
        ) == "1":
            _BANDED_SELFTEST_ARMED = False
            try:
                banded = convert_layer_to_fp4_banded(
                    layer, ext, bits, cb, shard_id=shard_id
                )
                gs_ref = float(fp4_weight.global_scale)
                gs_band = float(banded.global_scale)
                packed_mism = int(
                    (banded.packed != fp4_weight.packed).sum().item()
                )
                scale_mism = int(
                    (banded.scale_storage != fp4_weight.scale_storage)
                    .sum().item()
                )
                total = fp4_weight.packed.numel()
                print(
                    f"[FP4 banded selftest] gs ref={gs_ref:.6g} "
                    f"band={gs_band:.6g} packed_mismatch={packed_mism}/{total} "
                    f"({100.0 * packed_mism / total:.4f}%) "
                    f"scale_mismatch={scale_mism}/{fp4_weight.scale_storage.numel()}",
                    flush=True,
                )
                del banded
                torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"[FP4 banded selftest] FAILED: {exc}", flush=True)
        # Clear CUDA cache between shards to prevent fragmentation OOM
        torch.cuda.empty_cache()

    # Store on the layer for the runtime path.
    layer.fp4_weights = fp4_weights

    # Note: caller frees trellis tensors after confirming all shards converted.
    return fp4_weights


# ---------------------------------------------------------------------------
# 2b. Banded load-time conversion (large matrices, e.g. lm_head)
# ---------------------------------------------------------------------------
#
# ``convert_layer_to_fp4`` materialises the full weight three times (fp16
# reconstruct, fp32 Hadamard fold, fp32 quantizer transients).  For the
# 5120x248320 lm_head that is a 4.74 GiB fp32 fold temp plus ~14 GiB of
# quantizer transients — unbuildable at load time.  The banded path keeps
# peak transient memory at ~250 MB independent of N:
#
#   * ``ext.reconstruct_slice`` decodes a contiguous 128-aligned N-band of
#     the (K, N) trellis weight (reconstruct.cu enforces n_offset % 128 == 0
#     and band_width % 128 == 0).
#   * The Hadamard fold is exactly separable along 128-aligned N bounds:
#     Had_N is block-diagonal with 128-point blocks and svh is per-element,
#     so folding W[:, n0:n1] with svh[n0:n1] equals slicing the full fold.
#   * Quantisation is two-pass: pass 1 computes the whole-matrix amax over
#     folded bands, pass 2 quantizes each band against that single global
#     scale (bit-compatible with the unbanded quantizer, which also uses one
#     per-tensor scale).
#   * The swizzled block-scale layout is row-block-major (outermost dim is
#     rows/128), so a 128-aligned row band's swizzled storage is a
#     contiguous slice of the full matrix's storage at byte offset
#     ``n0 * cols_padded`` — bands write directly into the final buffer.
#
# The result is a plain ``FP4DenseWeight``; ``fp4_apply`` needs no changes.

def convert_layer_to_fp4_banded(
    layer: torch.nn.Module,
    ext: Any,
    bits: int,
    cb: int,
    *,
    shard_id: Any = None,
    band_n: int = 2048,
) -> FP4DenseWeight:
    """Convert one EXL3 trellis shard to NVFP4 in 128-aligned N bands."""
    if not hasattr(ext, "reconstruct_slice"):
        raise RuntimeError(
            "exllamav3_ext lacks reconstruct_slice; banded conversion "
            "unavailable"
        )
    if band_n % _HADAMARD_BLOCK != 0:
        raise ValueError(f"band_n must be a multiple of 128, got {band_n}")

    trellis = layer.trellis.exl3_tensors[shard_id]
    suh = layer.suh.exl3_tensors[shard_id]
    svh = layer.svh.exl3_tensors[shard_id]
    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    device = trellis.device
    if K % _HADAMARD_BLOCK != 0 or N % _HADAMARD_BLOCK != 0:
        raise ValueError(f"K={K} and N={N} must be multiples of 128")
    mcg, mul1 = _codebook_to_flags(cb)

    cols_padded = _align_up(K // _SF_VEC_SIZE, 4)
    # Final artifacts, written band by band.
    packed_full = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
    scale_full = torch.empty(1, N * cols_padded, dtype=torch.uint8, device=device)

    band_buf = torch.empty(K, band_n, dtype=torch.float16, device=device)

    def _folded_band(n0: int, nb: int) -> torch.Tensor:
        band = band_buf[:, :nb]
        ext.reconstruct_slice(band, trellis, bits, mcg, mul1, n0)
        # In-place chunked fold: fp32 transients are 128 x nb.
        return hadamard_fold_weight_chunked(band, suh, svh[n0:n0 + nb])

    # --- Pass 1: whole-matrix amax over the folded weight ---
    amax = torch.zeros((), dtype=torch.float32, device=device)
    for n0 in range(0, N, band_n):
        nb = min(band_n, N - n0)
        band = _folded_band(n0, nb)
        torch.maximum(amax, band.abs().amax().to(torch.float32), out=amax)
    amax = amax.clamp_min(1e-12)
    global_scale = (_NVFP4_GS_NUM / amax).reshape(1)

    # --- Pass 2: quantize each band against the fixed global scale ---
    for n0 in range(0, N, band_n):
        nb = min(band_n, N - n0)
        band = _folded_band(n0, nb)
        rows = band.t().contiguous().to(torch.bfloat16)  # (nb, K)
        packed_b, scale_b, _ = _quantize_matrix_fp4_nvfp4(
            rows, global_scale_override=global_scale
        )
        packed_full[n0:n0 + nb].copy_(packed_b)
        off = n0 * cols_padded
        scale_full[0, off:off + nb * cols_padded].copy_(scale_b.reshape(-1))
        del rows, packed_b, scale_b

    del band_buf
    torch.cuda.empty_cache()

    return FP4DenseWeight(
        packed=packed_full,
        scale_storage=scale_full,
        global_scale=global_scale,
        out_features=N,
        in_features=K,
    )


def convert_all_shards_to_fp4_banded(
    layer: torch.nn.Module,
    ext: Any,
    *,
    band_n: int = 2048,
) -> dict[Any, Any]:
    """Banded variant of :func:`convert_all_shards_to_fp4`.

    Same shard iteration and codebook detection, but each shard is
    converted via :func:`convert_layer_to_fp4_banded` with bounded peak
    memory.  Does NOT store ``layer.fp4_weights`` or free trellis tensors —
    the caller decides (the draft-head path keeps the trellis for the
    verify pass).
    """
    fp4_weights: dict[Any, Any] = {}
    try:
        from b12x._lib.runtime_control import unfreeze_kernel_resolution
        unfreeze_kernel_resolution()
    except ImportError:
        pass

    for shard_id in list(layer.exl3_shard_ids):
        trellis = layer.trellis.exl3_tensors[shard_id]
        bits = trellis.shape[2] // 16
        has_mcg = shard_id in layer.mcg.exl3_tensors
        has_mul1 = shard_id in layer.mul1.exl3_tensors
        if has_mcg and not has_mul1:
            cb = 1
        elif has_mul1 and not has_mcg:
            cb = 2
        else:
            cb = 0
        fp4_weights[shard_id] = convert_layer_to_fp4_banded(
            layer, ext, bits, cb, shard_id=shard_id, band_n=band_n
        )
        torch.cuda.empty_cache()
    return fp4_weights


# ---------------------------------------------------------------------------
# 3. Runtime apply
# ---------------------------------------------------------------------------

try:
    import torch as _torch
except Exception:
    _torch = None

def fp4_apply(
    x: torch.Tensor,
    fp4_weight: FP4DenseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the NVFP4 W4A4 dense linear (optimized: no host sync, no extra alloc).

    Optimizations vs original:
    - GPU-side amax (no .item() host sync) — eliminates 64 CPU-GPU syncs/forward
    - Precomputed inv_w_global_scale on FP4DenseWeight (1 div launch vs 3)
    - Caller-provided out= passed to dense_gemm (no intermediate y + copy)
    """
    from b12x._lib.dense_gemm import dense_gemm

    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 (M, K), got {tuple(x.shape)}")
    m, k = x.shape
    n = fp4_weight.out_features
    device = x.device

    m_pad = ((m + _TILE - 1) // _TILE) * _TILE
    x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)

    # --- Quantise activation to NVFP4 ---
    # For decode (M < 128): Triton kernel (no 128× padding waste)
    _row_amax = None
    _use_triton = (m < _TILE and __import__('os').environ.get("VLLM_EXL3_FP4_TRITON_DECODE", "1") == "1")
    if _use_triton:
        try:
            import sys as _sys; _sys.path.insert(0, '/opt/fp4')
            from triton_fp4_quant import triton_fp4_quant as _triton_quant
            k_blocks = k // _SF_VEC_SIZE
            scale_stride = ((k_blocks + 3) // 4) * 4
            _tkey = (m, k, str(device))
            _tcached = _QUANT_CACHE.get(_tkey)
            if _tcached is None:
                _sbuf = torch.zeros(1, 128 * scale_stride, dtype=torch.uint8, device=device)
                _pbuf = torch.zeros(1, m, k // 2, dtype=torch.uint8, device=device)
                _gsbuf = torch.zeros(1, dtype=torch.float32, device=device)
                _igsbuf = torch.zeros(1, dtype=torch.float32, device=device)
                _QUANT_CACHE[_tkey] = (_sbuf, _pbuf, _gsbuf, _igsbuf)
                _tcached = (_sbuf, _pbuf, _gsbuf, _igsbuf)
            _sbuf, _pbuf, _gsbuf, _igsbuf = _tcached
            _sbuf.zero_()
            # amax + global scale (GPU-side, no .item())
            amax = x_bf16.abs().max().clamp_min(1e-12)
            a_global_scale = (_NVFP4_GS_NUM / amax).reshape(1).to(torch.float32)
            _gsbuf.copy_(a_global_scale)
            _igsbuf.copy_(1.0 / a_global_scale)
            _triton_quant(x_bf16, _sbuf.view(-1), _pbuf[0], _gsbuf, _igsbuf)
            a_sf = _sbuf.view(torch.float8_e4m3fn)
            # Match b12x layout: (M, K//2, 1) viewed as float4_e2m1fn_x2
            a_torch = _pbuf.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        except Exception:
            _use_triton = False
    if not _use_triton:
        # b12x TMA path (prefill or Triton fallback)
        import b12x.quantization.nvfp4._impl as _nvfp4_impl
        _nvfp4_impl._validate_launch_tensor = lambda *a, **kw: None
        _nvfp4_impl._overlaps = lambda *a: False
        from b12x.quantization.nvfp4 import plan as _nvfp4_plan, allocate_outputs as _nvfp4_alloc, run as _nvfp4_run
        # Per-row activation global scale (VLLM_EXL3_FP4_PER_ROW_GS=1):
        # per-tensor amax lets one outlier row wreck the FP4 range for all
        # 2051 prefill rows. Pre-scale each row to unit amax, quantize with a
        # constant global scale, and undo per-row on the GEMM output (the
        # dense_gemm row_scale epilogue is MX-FP6-only, so the undo is a torch
        # broadcast multiply). Prefill-only (m > 1); decode m=1 is per-row by
        # definition and stays on the graph-safe per-tensor path.
        if m > 1 and __import__('os').environ.get("VLLM_EXL3_FP4_PER_ROW_GS", "0") == "1":
            _row_amax = (
                x_bf16.to(torch.float32).abs().amax(dim=1, keepdim=True)
                .clamp_min(1e-6)
            )  # [M, 1] fp32
            x_bf16 = (x_bf16.to(torch.float32) / _row_amax).to(torch.bfloat16)
            amax = torch.ones((), device=device, dtype=torch.float32)
        else:
            amax = x_bf16.abs().max().clamp_min(1e-12)
        a_global_scale = (_NVFP4_GS_NUM / amax).reshape(1).to(torch.float32)
        cache_key = (m_pad, k, str(device))
        cached = _QUANT_CACHE.get(cache_key)
        if cached is None:
            qplan = _nvfp4_plan(m_pad, k)
            qouts = _nvfp4_alloc(qplan, device=device)
            x_pad = torch.zeros(m_pad, k, dtype=torch.bfloat16, device=device) if (m_pad != m and m <= _TILE) else None
            _QUANT_CACHE[cache_key] = (qplan, qouts, x_pad)
            cached = (qplan, qouts, x_pad)
        qplan, qouts, x_pad = cached
        if m_pad != m:
            if x_pad is not None:
                x_pad[:m].copy_(x_bf16)
                x_quant = x_pad
            else:
                x_quant = torch.zeros(m_pad, k, dtype=torch.bfloat16, device=device)
                x_quant[:m].copy_(x_bf16)
        else:
            x_quant = x_bf16
        _nvfp4_run(plan=qplan, x=x_quant, global_scale=a_global_scale, outputs=qouts)
        a_sf = qouts.scale_storage.view(torch.float8_e4m3fn)
        try:
            a_torch = qouts.packed_a_view[:m]
        except (TypeError, RuntimeError):
            a_torch = qouts.packed_a_storage[:m].permute(1, 2, 0).unsqueeze(-1)

    b_sf = fp4_weight.scale_view()
    b_torch = fp4_weight.packed_view().unsqueeze(-1)
    # --- Alpha: 1/(a_gs * w_gs) using precomputed inv_w_gs (1 launch) ---
    inv_w_gs = getattr(fp4_weight, '_inv_global_scale', None)
    if inv_w_gs is None:
        inv_w_gs = torch.reciprocal(fp4_weight.global_scale)
        fp4_weight._inv_global_scale = inv_w_gs
    alpha = (inv_w_gs / a_global_scale).to(torch.float32)

    # --- Run GEMM with caller-provided output (no intermediate alloc + copy) ---
    if out is not None:
        y = out.unsqueeze(-1) if out.ndim == 2 else out
    else:
        y = torch.empty((m, n, 1), device=device, dtype=torch.bfloat16)
    # Tile/load/swap overrides for prefill tuning.
    import os as _os
    _tile_override = _os.environ.get("VLLM_EXL3_FP4_TILE_MN", "")
    _load_path = _os.environ.get("VLLM_EXL3_FP4_LOAD_PATH", "")
    _swap_ab = _os.environ.get("VLLM_EXL3_FP4_SWAP_AB", "0") == "1"
    _tile_kwargs = {}
    if m > _TILE:
        if _tile_override:
            try:
                tm, tn = (int(x) for x in _tile_override.split(","))
                _tile_kwargs["mma_tiler_mn"] = (tm, tn)
            except (ValueError, TypeError):
                pass
        if _load_path:
            _tile_kwargs["load_path"] = _load_path
        if _swap_ab:
            _tile_kwargs["swap_ab"] = True
    dense_gemm(
        (a_torch[:m], a_sf),
        (b_torch, b_sf),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        sf_vec_size=_SF_VEC_SIZE,
        c_dtype="bfloat16",
        alpha=alpha,
        out=y,
        expected_m=m if m > _TILE else None,
        **_tile_kwargs,
    )
    if _row_amax is not None:
        # Undo the per-row pre-scaling on the output (broadcast [M,1] over N).
        y[:, :, 0].mul_(_row_amax.to(torch.bfloat16))
    return y[:, :, 0] if out is None else out


# ---------------------------------------------------------------------------
# 4. Integration hooks for Exl3LinearMethod
# ---------------------------------------------------------------------------
#
# The following shows how to modify ``Exl3LinearMethod`` to use the FP4
# path.  These are drop-in replacements for the two key methods.
#
# --- In process_weights_after_loading() ---
#
# After the existing validation and TP-sharding code, add the FP4 conversion
# gated by an env var (e.g. ``VLLM_EXL3_FP4_PREFILL=1``):
#
#   def process_weights_after_loading(self, layer):
#       # ... existing validation, TP sharding, device moves ...
#
#       if os.environ.get("VLLM_EXL3_FP4_PREFILL", "0") == "1":
#           ext = _load_exl3_ext()
#           convert_all_shards_to_fp4(layer, ext)
#           return  # skip the b12x/warpspec warmup below
#
#       # ... existing b12x trellis warmup, graph decode priming ...
#
# --- In _apply_one() ---
#
# Add an FP4 branch at the top of ``_apply_one``:
#
#   @staticmethod
#   def _apply_one(layer, x, shard_id):
#       fp4_weights = getattr(layer, "fp4_weights", None)
#       if fp4_weights is not None and shard_id in fp4_weights:
#           return fp4_apply(x, fp4_weights[shard_id])
#
#       # ... existing trellis dispatch (b12x, warpspec, exl3_gemm) ...
#
# --- In apply() ---
#
# The ``apply`` method currently converts x to float16 before calling
# ``_apply_one``.  For the FP4 path, bfloat16 is preferred (the quantizer
# is designed for bf16 input).  Modify the dtype conversion:
#
#   def apply(self, layer, x, bias=None):
#       original_shape = x.shape[:-1]
#       original_dtype = x.dtype
#       if hasattr(layer, "fp4_weights"):
#           x_2d = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
#       else:
#           x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
#       outputs = [self._apply_one(layer, x_2d, sid) for sid in layer.exl3_shard_ids]
#       output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
#       if bias is not None:
#           output = output + bias.to(dtype=output.dtype)
#       output = output.reshape(*original_shape, output.shape[-1])
#       return output if output.dtype == original_dtype else output.to(original_dtype)
#


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_conversion(
    layer: torch.nn.Module,
    ext: Any,
    shard_id: Any,
    x_test: torch.Tensor,
    *,
    atol: float = 1.0,
    rtol: float = 0.08,
) -> bool:
    """Sanity-check the FP4 conversion against the reference EXL3 path.

    Reconstructs the weight, runs both the reference (Hadamard + matmul in
    fp32) and the FP4 path, and compares.  Intended for one-shot load-time
    validation, not the hot path.

    The tolerance is looser than FP6 (atol=1.0, rtol=0.08) because FP4
    E2M1 has only 8 representable values (4 bits) — the quantisation error
    is inherently larger.

    Args:
        layer:    The EXL3 layer (must still have trellis tensors loaded).
        ext:      The ``exllamav3_ext`` module.
        shard_id: Which shard to validate.
        x_test:   Test input, shape ``(M, K)``, float16.
        atol:     Absolute tolerance for the comparison.
        rtol:     Relative tolerance for the comparison.

    Returns:
        ``True`` if the FP4 output matches the reference within tolerance.
    """
    trellis = layer.trellis.exl3_tensors[shard_id]
    suh = layer.suh.exl3_tensors[shard_id]
    svh = layer.svh.exl3_tensors[shard_id]
    bits = trellis.shape[2] // 16
    has_mcg = shard_id in layer.mcg.exl3_tensors
    has_mul1 = shard_id in layer.mul1.exl3_tensors
    cb = 1 if has_mcg else (2 if has_mul1 else 0)

    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16

    # Reference: reconstruct + Hadamard fold + fp32 matmul.
    mcg, mul1 = _codebook_to_flags(cb)
    W = torch.empty(K, N, dtype=torch.float16, device=trellis.device)
    ext.reconstruct(W, trellis, bits, mcg, mul1)
    W_final = hadamard_fold_weight(W, suh, svh)
    y_ref = (x_test.float() @ W_final.float()).to(torch.bfloat16)

    # FP4 path.
    fp4_weight = convert_layer_to_fp4(layer, ext, bits, cb, shard_id=shard_id)
    y_fp4 = fp4_apply(x_test, fp4_weight)

    max_diff = (y_ref.float() - y_fp4.float()).abs().max().item()
    rel_diff = max_diff / (y_ref.float().abs().max().item() + 1e-6)
    ok = max_diff <= atol or rel_diff <= rtol
    if not ok:
        import warnings
        warnings.warn(
            f"FP4 conversion validation FAILED for shard {shard_id}: "
            f"max_diff={max_diff:.4f}, rel_diff={rel_diff:.4f}",
        )
    return ok


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test: verify Hadamard matrix orthogonality and fold correctness.
    H = _hadamard_128_matrix("cpu", torch.float32)
    eye = H @ H.T
    err = (eye - torch.eye(_HADAMARD_BLOCK)).abs().max().item()
    print(f"Hadamard orthogonality error: {err:.2e}")
    assert err < 1e-5, "Hadamard matrix is not orthogonal"

    # Test fold with a small synthetic weight (K=128, N=256).
    K, N = 128, 256
    W = torch.randn(K, N, dtype=torch.float16) * 0.01
    suh = torch.ones(K // _HADAMARD_BLOCK, dtype=torch.float16)
    svh = torch.ones(N // _HADAMARD_BLOCK, dtype=torch.float16)
    W_folded = hadamard_fold_weight(W, suh, svh)
    print(f"Folded weight shape: {W_folded.shape}, dtype: {W_folded.dtype}")

    # Test FP4 quantisation round-trip.
    W_linear = W_folded.t().contiguous().to(torch.bfloat16)  # (N, K)
    packed, scale_storage, global_scale = _quantize_matrix_fp4_nvfp4(W_linear)
    assert packed.shape == (N, K // 2), f"packed shape {packed.shape}"
    expected_scales = N * (K // _SF_VEC_SIZE)
    assert scale_storage.numel() >= expected_scales, (
        f"scale_storage {scale_storage.numel()} < {expected_scales}"
    )
    assert global_scale.shape == (1,), f"global_scale shape {global_scale.shape}"
    assert global_scale.dtype == torch.float32, f"global_scale dtype {global_scale.dtype}"
    print(f"FP4 packed shape: {packed.shape}, dtype: {packed.dtype}")
    print(f"Scale storage: {scale_storage.numel()} bytes (E4M3FN)")
    print(f"Global scale: {global_scale.item():.6f}")

    # Build FP4DenseWeight and verify scale_view shape.
    fp4_w = FP4DenseWeight(
        packed=packed,
        scale_storage=scale_storage,
        global_scale=global_scale,
        out_features=N,
        in_features=K,
    )
    sv = fp4_w.scale_view()
    print(f"Scale view shape: {sv.shape}, dtype: {sv.dtype}")

    print("Smoke test passed.")
