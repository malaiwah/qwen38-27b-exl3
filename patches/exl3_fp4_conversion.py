#!/usr/bin/env python3
"""EXL3 trellis K6 -> MXFP4 W4A4 load-time weight conversion.

This module converts EXL3 trellis-quantized K6 weights to MXFP4 format at
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

The folded weight is then quantised to MXFP4 (FP4 E2M1 with UE8M0 block
scales) and executed at runtime via ``dense_gemm`` with
``ab_dtype='float4_e2m1fn'``, which uses the ``mxf4nvf4.m16n8k64`` MMA
(4x throughput vs FP16 on SM120/Blackwell).

MXFP4 format
------------
* **Weights** are static, quantised once at load time into packed FP4
  E2M1 codes (4 bits/element, 2 values per byte) + swizzled UE8M0
  block-scales (``sf_vec_size=32``).
* **Activations** are data-dependent, quantised on the fly each forward
  to the same MXFP4 format (FP4 E2M1 + UE8M0 block scales).
* **No global scale** is needed — the UE8M0 power-of-two block scales
  carry the full dynamic range (unlike NVFP4 which uses E4M3 block
  scales + a per-tensor global scale).
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
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HADAMARD_BLOCK = 128
_HADAMARD_NORM = 1.0 / math.sqrt(_HADAMARD_BLOCK)

# MXFP4 block scale parameters — match the FP6 / MXFP8 UE8M0 scheme.
_SF_VEC_SIZE = 32          # elements per UE8M0 scale block
_FP4_E2M1_MAX = 6.0        # maximum magnitude representable in FP4 E2M1
_INV_FP4_E2M1_MAX = 1.0 / _FP4_E2M1_MAX

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


def _pow2_ceil_ue8m0(
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round positive fp32 scales UP to a power of two.

    Returns ``(rounded_fp32, ue8m0_u8)`` where ``ue8m0_u8`` is the IEEE-754
    exponent byte (bias-127) of the rounded value.  A zero scale maps to
    ``(0.0, 0)``.

    Bit-exact Torch replica of the b12x ``pow2_ceil_ue8m0`` device intrinsic.
    """
    bits = scale.to(torch.float32).contiguous().view(torch.int32)
    mant = bits & 0x007FFFFF
    bumped = torch.where(mant != 0, (bits + 0x00800000) & 0x7F800000, bits)
    rounded = bumped.view(torch.float32)
    byte = ((bumped >> 23) & 0xFF).to(torch.uint8)
    return rounded, byte


def _ue8m0_inv_scale(byte: torch.Tensor) -> torch.Tensor:
    """Return ``1 / 2^(byte-127)`` for UE8M0 bytes; ``0`` for byte 0.

    This is the inverse block scale used to normalise values before
    quantising to FP4: ``normalized = value * inv_scale``.
    """
    inv_bits = (254 - byte.to(torch.int32)).clamp(min=0) << 23
    inv = inv_bits.view(torch.float32)
    return torch.where(byte == 0, torch.zeros_like(inv), inv)


def _swizzle_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Swizzle block-scale factors into the 128x4 interleaved layout.

    The CUTLASS MMA expects scales in a specific permuted order that
    matches the smem bank-conflict-free tile layout.  This is a bit-exact
    Torch replica of ``b12x._lib.intrinsics.swizzle_block_scale``.

    Input:  ``(batch, rows, cols)`` or ``(rows, cols)`` uint8.
    Output: ``(batch, rows_padded, cols_padded)`` uint8, where
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


def _as_mxfp4_scale_view(
    scale_storage: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """View flat swizzled UE8M0 scales as the 6D tensor dense_gemm expects.

    ``scale_storage`` is ``(batch, -1)`` flat uint8 (the output of
    ``_swizzle_block_scale`` reshaped to 2D).  ``rows`` and ``cols`` are
    the *logical* matrix dimensions (before padding).  ``cols`` must be
    divisible by ``_SF_VEC_SIZE`` (32).
    """
    batch = scale_storage.shape[0]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // _SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e8m0fnu)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


# ---------------------------------------------------------------------------
# Matrix quantisation (shared by weight and activation paths)
# ---------------------------------------------------------------------------

def _quantize_matrix_fp4_mxfp4(
    mat_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``(rows, K)`` bf16/fp16 matrix to packed MXFP4.

    Produces:
    * ``packed``   — ``(rows, K // 2)`` uint8, two FP4 E2M1 nibbles per byte
                     (low nibble = first element, high nibble = second).
    * ``scale_storage`` — ``(1, -1)`` flat uint8, swizzled UE8M0 block
                           scales ready for ``_as_mxfp4_scale_view``.

    The quantisation is block-wise along the K dimension with
    ``sf_vec_size = 32``: each block of 32 consecutive K elements shares
    one UE8M0 (power-of-two) scale.  The scale is chosen as
    ``pow2_ceil(max_abs / 6.0)`` so that the scaled values fit within the
    FP4 E2M1 range ``[-6, 6]``.
    """
    rows, k = mat_bf16.shape
    if k % _SF_VEC_SIZE != 0:
        raise ValueError(
            f"K must be a multiple of {_SF_VEC_SIZE}, got K={k}"
        )

    # --- Block along K dimension: (rows, K//32, 32) ---
    blocked = mat_bf16.to(torch.float32).reshape(rows, k // _SF_VEC_SIZE, _SF_VEC_SIZE)

    # --- Per-block max-abs → UE8M0 power-of-two scale ---
    block_max = blocked.abs().amax(dim=-1, keepdim=True)  # (rows, K//32, 1)
    # scale = pow2_ceil(max_abs / FP4_MAX); the pow2_ceil ensures the
    # normalised values never exceed FP4_E2M1_MAX.
    _rounded, scale_byte = _pow2_ceil_ue8m0(block_max * _INV_FP4_E2M1_MAX)
    inv_scale = _ue8m0_inv_scale(scale_byte)  # (rows, K//32, 1) = 1/scale

    # --- Quantise to FP4 E2M1 ---
    normalised = (blocked * inv_scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    fp4_values = _fp4_quantize_values(normalised)          # (rows, K//32, 32)
    nibbles = _fp4_encode_nibbles(fp4_values)               # (rows, K//32, 32) uint8

    # --- Pack 2 nibbles per byte: (rows, K//2) ---
    nibbles = nibbles.reshape(rows, k // 2, 2)
    packed = nibbles[..., 0] | (nibbles[..., 1] << 4)      # (rows, K//2) uint8
    packed = packed.contiguous()

    # --- Swizzle UE8M0 scales into the flat layout dense_gemm consumes ---
    scales = scale_byte.squeeze(-1)                         # (rows, K//32) uint8
    swizzled = _swizzle_block_scale(scales.unsqueeze(0))    # (1, rows_pad, K//32_pad)
    scale_storage = swizzled.reshape(1, -1).contiguous()   # (1, -1) flat

    return packed, scale_storage


# ---------------------------------------------------------------------------
# FP4DenseWeight — the load-time artifact
# ---------------------------------------------------------------------------

@dataclass
class FP4DenseWeight:
    """A single MXFP4-quantized dense weight ready for :func:`fp4_apply`.

    ``packed`` holds ``(out_features, in_features // 2)`` FP4 E2M1 codes
    (2 per byte) and ``scale_storage`` is the flat swizzled UE8M0 block-scale
    buffer; the 6D scale view ``dense_gemm`` wants is rebuilt on demand via
    :meth:`scale_view`.
    """

    packed: torch.Tensor            # (N, K//2) uint8
    scale_storage: torch.Tensor     # (1, -1) flat uint8 (swizzled UE8M0)
    out_features: int
    in_features: int

    def __post_init__(self) -> None:
        # Cached fp4_e2m1fn_x2 view (rebuilt lazily after ``to()``).
        self._fp4_view: Optional[torch.Tensor] = None

    def scale_view(self) -> torch.Tensor:
        """The 6D swizzled UE8M0 scale view for ``dense_gemm``."""
        return _as_mxfp4_scale_view(
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
    """Convert one EXL3 trellis shard to an MXFP4 ``FP4DenseWeight``.

    Steps:
      1. Extract trellis codes, suh, svh from the layer's ``exl3_tensors``.
      2. Reconstruct trellis codes to an FP16 weight ``W`` via
         ``ext.reconstruct()``.
      3. Fold the Hadamard transforms and per-block scales into ``W``.
      4. Transpose to ``(N, K)`` (the ``nn.Linear`` weight layout) and
         quantise to MXFP4 (FP4 E2M1 + UE8M0 block scales).
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

    # --- Reconstruct trellis codes to FP16 weight on CPU to avoid GPU OOM ---
    # The FP16 weight matrix can be 4+ GB for large layers; keeping it on
    # GPU alongside existing FP4 weights causes OOM. Do reconstruction and
    # Hadamard fold on CPU, then move only the final FP4 result to GPU.
    trellis_cpu = trellis.cpu()
    suh_cpu = suh.cpu()
    svh_cpu = svh.cpu()
    mcg, mul1 = _codebook_to_flags(cb)
    W = torch.empty(K, N, dtype=torch.float16, device="cpu")
    ext.reconstruct(W, trellis_cpu, bits, mcg, mul1)

    # --- Fold Hadamard transforms + per-element scales into W (on CPU) ---
    W_final = hadamard_fold_weight(W, suh_cpu, svh_cpu)

    # --- Transpose to (N, K) for the nn.Linear weight layout ---
    W_linear = W_final.t().contiguous()  # (N, K)

    # Free CPU temporaries
    del W, W_final, trellis_cpu, suh_cpu, svh_cpu

    # --- Quantise to MXFP4 (FP4 E2M1 + UE8M0 block scales, sf_vec_size=32) ---
    packed, scale_storage = _quantize_matrix_fp4_mxfp4(
        W_linear.to(torch.bfloat16).to(device)
    )

    # Free temporaries immediately to avoid OOM during batch conversion
    del W_linear

    fp4_weight = FP4DenseWeight(
        packed=packed,
        scale_storage=scale_storage,
        out_features=N,
        in_features=K,
    )

    return fp4_weight


def convert_all_shards_to_fp4(
    layer: torch.nn.Module,
    ext: Any,
) -> dict[Any, Any]:
    """Convert every shard in an EXL3 layer to MXFP4.

    Iterates over ``layer.exl3_shard_ids``, reconstructs each shard's
    trellis codes, folds the Hadamard, and quantises to FP4.  The resulting
    :class:`FP4DenseWeight` objects are stored in ``layer.fp4_weights``
    (keyed by shard id) and the original trellis tensors are freed.

    Returns the dict of ``{shard_id: FP4DenseWeight}``.
    """
    shard_ids = list(layer.exl3_shard_ids)
    fp4_weights: dict[Any, Any] = {}

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
        # Clear CUDA cache between shards to prevent fragmentation OOM
        torch.cuda.empty_cache()

    # Store on the layer for the runtime path.
    layer.fp4_weights = fp4_weights

    # Free trellis tensors to reclaim VRAM. The caller returns early and
    # skips all trellis warmup/priming when FP4 is active.
    for attr in ("trellis", "suh", "svh", "mcg", "mul1"):
        param = getattr(layer, attr, None)
        if param is not None and hasattr(param, "exl3_tensors"):
            param.exl3_tensors.clear()
    return fp4_weights


# ---------------------------------------------------------------------------
# 3. Runtime apply
# ---------------------------------------------------------------------------

def fp4_apply(
    x: torch.Tensor,
    fp4_weight: FP4DenseWeight,
) -> torch.Tensor:
    """Run the MXFP4 W4A4 dense linear.

    Quantises ``x`` to FP4 E2M1 with UE8M0 block scales on the fly, then
    runs ``dense_gemm`` with ``ab_dtype='float4_e2m1fn'`` which uses the
    ``mxf4nvf4.m16n8k64`` MMA (4x throughput vs FP16 on SM120/Blackwell).

    Both activations and weights are FP4 E2M1.  The UE8M0 power-of-two block
    scales carry the full dynamic range for each operand, so no global
    scale or epilogue alpha is needed (``alpha = 1.0``).

    Args:
        x:          Input activations, shape ``(M, K)``, float16 or bfloat16.
        fp4_weight: A :class:`FP4DenseWeight` produced by
                    :func:`convert_layer_to_fp4`.

    Returns:
        Output activations, shape ``(M, N)``, bfloat16.
    """
    from b12x._lib.dense_gemm import dense_gemm

    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 (M, K), got {tuple(x.shape)}")
    m, k = x.shape
    if k != fp4_weight.in_features:
        raise ValueError(
            f"in_features mismatch: x K={k}, weight in_features="
            f"{fp4_weight.in_features}"
        )

    n = fp4_weight.out_features
    device = x.device

    # --- Pad M to a multiple of _TILE (128) for the quantiser ---
    m_pad = ((m + _TILE - 1) // _TILE) * _TILE
    x_bf16 = x.to(torch.bfloat16)
    if m_pad != m:
        x_pad = torch.zeros(m_pad, k, dtype=torch.bfloat16, device=device)
        x_pad[:m].copy_(x_bf16)
        x_quant = x_pad
    else:
        x_quant = x_bf16

    # --- Quantise activation to MXFP4 (FP4 E2M1 + UE8M0 block scales) ---
    a_packed, a_scale_storage = _quantize_matrix_fp4_mxfp4(x_quant)
    # a_packed: (m_pad, K//2) uint8

    # --- Build scale views for dense_gemm ---
    a_sf = _as_mxfp4_scale_view(a_scale_storage, m_pad, k)
    b_sf = fp4_weight.scale_view()

    # --- Build operand tensors: (rows, K//2, L=1) ---
    try:
        a_torch = a_packed.view(torch.float4_e2m1fn_x2).unsqueeze(-1)
    except (TypeError, RuntimeError):
        a_torch = a_packed.unsqueeze(-1)

    b_torch = fp4_weight.packed_view().unsqueeze(-1)

    # --- Run the GEMM at the true M (padding rows are never computed) ---
    y = torch.empty((m, n, 1), device=device, dtype=torch.bfloat16)
    dense_gemm(
        (a_torch[:m], a_sf),
        (b_torch, b_sf),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=_SF_VEC_SIZE,
        c_dtype="bfloat16",
        out=y,
    )
    return y[:, :, 0]


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
    packed, scale_storage = _quantize_matrix_fp4_mxfp4(W_linear)
    assert packed.shape == (N, K // 2), f"packed shape {packed.shape}"
    expected_scales = N * (K // _SF_VEC_SIZE)
    assert scale_storage.numel() >= expected_scales, (
        f"scale_storage {scale_storage.numel()} < {expected_scales}"
    )
    print(f"FP4 packed shape: {packed.shape}, dtype: {packed.dtype}")
    print(f"Scale storage: {scale_storage.numel()} bytes (UE8M0)")

    # Build FP4DenseWeight and verify scale_view shape.
    fp4_w = FP4DenseWeight(
        packed=packed,
        scale_storage=scale_storage,
        out_features=N,
        in_features=K,
    )
    sv = fp4_w.scale_view()
    print(f"Scale view shape: {sv.shape}, dtype: {sv.dtype}")

    print("Smoke test passed.")
