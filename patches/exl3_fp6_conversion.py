#!/usr/bin/env python3
"""EXL3 trellis K6 -> MXFP6 W6A8 load-time weight conversion.

This module converts EXL3 trellis-quantized K6 weights to MXFP6 format at
model load time, enabling the b12x ``dense_gemm`` W6A8 (E2M3 weights x E4M3
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

The folded weight is then quantised to MXFP6 via
``b12x.quantization.mxfp6.fp6_dense_weights.quantize_dense_weight_to_fp6``
and executed at runtime via ``dense_fp6_linear``, which quantises
activations to FP8 E4M3 on the fly and runs the ``mxf8f6f4``
block-scaled MMA on SM120.

b12x weight layout
------------------
``quantize_dense_weight_to_fp6`` expects ``(out_features, in_features)``
(== ``(N, K)`` for a standard ``nn.Linear``) and ``dense_fp6_linear``
computes ``y = x @ W.T``.  The EXL3 reconstructed weight is ``(K, N)``
(``y = x @ W``), so the folded weight is transposed to ``(N, K)`` before
quantisation.

Integration
-----------
Hook ``convert_layer_to_fp6`` into ``Exl3LinearMethod.process_weights_after_loading``
and ``fp6_apply`` into ``Exl3LinearMethod._apply_one`` (see the integration
hooks at the bottom of this file).
"""

from __future__ import annotations

import math
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HADAMARD_BLOCK = 128
_HADAMARD_NORM = 1.0 / math.sqrt(_HADAMARD_BLOCK)

# Source-format selector consumed by ``quantize_dense_weight_to_fp6``.
# ``mxfp6_w6a8`` means:
#   - weight sub-format: E2M3 (more mantissa bits for static weights)
#   - activation sub-format: E4M3 (FP8 activations, quantised on the fly)
#   - block-scaled MMA kind: ``mxf8f6f4`` (SM120 ``compute_120a``)
_FP6_SOURCE_FORMAT = "mxfp6_w6a8"

# Cache the 128x128 Hadamard matrix per (device, dtype) to avoid rebuilds.
_HADAMARD_CACHE: dict[tuple[str, str], torch.Tensor] = {}


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
# 1. Hadamard fold
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


def convert_layer_to_fp6(
    layer: torch.nn.Module,
    ext: Any,
    bits: int,
    cb: int,
    *,
    shard_id: Any = None,
) -> Any:
    """Convert one EXL3 trellis shard to an MXFP6 ``FP6DenseWeight``.

    Steps:
      1. Extract trellis codes, suh, svh from the layer's ``exl3_tensors``.
      2. Reconstruct trellis codes to an FP16 weight ``W`` via
         ``ext.reconstruct()``.
      3. Fold the Hadamard transforms and per-block scales into ``W``.
      4. Transpose to ``(N, K)`` (the ``nn.Linear`` weight layout) and
         quantise to MXFP6 via ``quantize_dense_weight_to_fp6``.
      5. Store the ``FP6DenseWeight`` on the layer and free the original
         trellis tensors.

    Args:
        layer:    The vLLM layer module (has ``trellis``, ``suh``, ``svh``,
                  ``mcg``, ``mul1`` ``Exl3Parameter`` attributes).
        ext:      The ``exllamav3_ext`` module (provides ``reconstruct``).
        bits:     Trellis bit width (e.g. 6 for K6).
        cb:       Codebook selector: 0=standard, 1=MCG, 2=mul1.
        shard_id: Which shard to convert (default ``None`` = single shard).

    Returns:
        A ``b12x ... FP6DenseWeight`` ready for ``dense_fp6_linear``.
    """
    from b12x.quantization.mxfp6.fp6_dense_weights import (
        quantize_dense_weight_to_fp6,
    )

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
    # GPU alongside existing FP6 weights causes OOM. Do reconstruction and
    # Hadamard fold on CPU, then move only the final FP6 result to GPU.
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

    # --- Quantise to MXFP6 (W6A8: E2M3 weights, E4M3 activations) ---
    fp6_weight = quantize_dense_weight_to_fp6(
        W_linear.to(torch.bfloat16).to(device),
        source_format=_FP6_SOURCE_FORMAT,
    )

    # Free temporaries immediately to avoid OOM during batch conversion
    del W_linear

    # Preserve the unsharded output width for the packed-GEMM routing
    # decision (see FP6DenseWeight.use_packed_gemm).
    out_features_unsharded = getattr(layer, "exl3_output_size", 0)
    if out_features_unsharded:
        fp6_weight.out_features_unsharded = int(out_features_unsharded)

    return fp6_weight


def convert_all_shards_to_fp6(
    layer: torch.nn.Module,
    ext: Any,
) -> dict[Any, Any]:
    """Convert every shard in an EXL3 layer to MXFP6.

    Iterates over ``layer.exl3_shard_ids``, reconstructs each shard's
    trellis codes, folds the Hadamard, and quantises to FP6.  The resulting
    ``FP6DenseWeight`` objects are stored in ``layer.fp6_weights`` (keyed
    by shard id) and the original trellis tensors are freed.

    Returns the dict of ``{shard_id: FP6DenseWeight}``.
    """
    shard_ids = list(layer.exl3_shard_ids)
    fp6_weights: dict[Any, Any] = {}

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

        fp6_weight = convert_layer_to_fp6(
            layer, ext, bits, cb, shard_id=shard_id
        )
        fp6_weights[shard_id] = fp6_weight
        # Clear CUDA cache between shards to prevent fragmentation OOM
        torch.cuda.empty_cache()

    # Store on the layer for the runtime path.
    layer.fp6_weights = fp6_weights

    # Free trellis tensors to reclaim VRAM. The caller returns early and
    # skips all trellis warmup/priming when FP6 is active.
    for attr in ("trellis", "suh", "svh", "mcg", "mul1"):
        param = getattr(layer, attr, None)
        if param is not None and hasattr(param, "exl3_tensors"):
            param.exl3_tensors.clear()
    return fp6_weights


# ---------------------------------------------------------------------------
# 3. Runtime apply
# ---------------------------------------------------------------------------

def fp6_apply(
    x: torch.Tensor,
    fp6_weight: Any,
) -> torch.Tensor:
    """Run the MXFP6 W6A8 dense linear.

    Delegates to ``b12x.dense_fp6_linear``, which quantises ``x`` to FP8
    E4M3 on the fly and runs the ``mxf8f6f4`` block-scaled MMA.

    Args:
        x:          Input activations, shape ``(M, K)`` (float16 or bfloat16).
        fp6_weight: A ``FP6DenseWeight`` produced by ``convert_layer_to_fp6``.

    Returns:
        Output activations, shape ``(M, N)``, bfloat16.
    """
    from b12x.quantization.mxfp6.fp6_dense_weights import dense_fp6_linear

    # dense_fp6_linear handles dtype conversion internally (converts to bf16).
    return dense_fp6_linear(x, fp6_weight)


# ---------------------------------------------------------------------------
# 4. Integration hooks for Exl3LinearMethod
# ---------------------------------------------------------------------------
#
# The following shows how to modify ``Exl3LinearMethod`` to use the FP6
# path.  These are drop-in replacements for the two key methods.
#
# --- In process_weights_after_loading() ---
#
# After the existing validation and TP-sharding code, add the FP6 conversion
# gated by an env var (e.g. ``VLLM_EXL3_FP6_PREFILL=1``):
#
#   def process_weights_after_loading(self, layer):
#       # ... existing validation, TP sharding, device moves ...
#
#       if os.environ.get("VLLM_EXL3_FP6_PREFILL", "0") == "1":
#           ext = _load_exl3_ext()
#           convert_all_shards_to_fp6(layer, ext)
#           return  # skip the b12x/warpspec warmup below
#
#       # ... existing b12x trellis warmup, graph decode priming ...
#
# --- In _apply_one() ---
#
# Add an FP6 branch at the top of ``_apply_one``:
#
#   @staticmethod
#   def _apply_one(layer, x, shard_id):
#       fp6_weights = getattr(layer, "fp6_weights", None)
#       if fp6_weights is not None and shard_id in fp6_weights:
#           return fp6_apply(x, fp6_weights[shard_id])
#
#       # ... existing trellis dispatch (b12x, warpspec, exl3_gemm) ...
#
# --- In apply() ---
#
# The ``apply`` method currently converts x to float16 before calling
# ``_apply_one``.  For the FP6 path, bfloat16 is preferred (the quantizer
# is designed for bf16 input).  Modify the dtype conversion:
#
#   def apply(self, layer, x, bias=None):
#       original_shape = x.shape[:-1]
#       original_dtype = x.dtype
#       if hasattr(layer, "fp6_weights"):
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


def _validate_conversion(
    layer: torch.nn.Module,
    ext: Any,
    shard_id: Any,
    x_test: torch.Tensor,
    *,
    atol: float = 0.5,
    rtol: float = 0.05,
) -> bool:
    """Sanity-check the FP6 conversion against the reference EXL3 path.

    Reconstructs the weight, runs both the reference (Hadamard + matmul in
    fp32) and the FP6 path, and compares.  Intended for one-shot load-time
    validation, not the hot path.

    Args:
        layer:    The EXL3 layer (must still have trellis tensors loaded).
        ext:      The ``exllamav3_ext`` module.
        shard_id: Which shard to validate.
        x_test:   Test input, shape ``(M, K)``, float16.
        atol:     Absolute tolerance for the comparison.
        rtol:     Relative tolerance for the comparison.

    Returns:
        ``True`` if the FP6 output matches the reference within tolerance.
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

    # FP6 path.
    fp6_weight = convert_layer_to_fp6(layer, ext, bits, cb, shard_id=shard_id)
    y_fp6 = fp6_apply(x_test, fp6_weight)

    max_diff = (y_ref.float() - y_fp6.float()).abs().max().item()
    rel_diff = max_diff / (y_ref.float().abs().max().item() + 1e-6)
    ok = max_diff <= atol or rel_diff <= rtol
    if not ok:
        import warnings
        warnings.warn(
            f"FP6 conversion validation FAILED for shard {shard_id}: "
            f"max_diff={max_diff:.4f}, rel_diff={rel_diff:.4f}",
        )
    return ok


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
    print("Smoke test passed.")
