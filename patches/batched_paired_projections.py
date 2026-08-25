"""Batched paired trellis projections with auto-detection.

Automatically detects when paired projections share the same suh (input Hadamard
scale) and batches them into a single GEMM. Works for:

1. GDN input: in_proj_qkv + in_proj_z
2. MLP gate/up: gate_proj + up_proj

When suh is shared, the two trellis weights are concatenated along N and run
as a single GEMM. When suh differs (the common case for EXL3 checkpoints),
the original two-GEMM path is used with zero overhead.

The batching is done at the PREPARED WEIGHT level: the caller provides two
prepared weights along with their original trellis/svh tensors, and this
module creates a batched prepared weight.

Env gates:
- VLLM_EXL3_BATCHED_PAIRED_PROJ=1  (master switch, default: 1)
- VLLM_EXL3_BATCHED_GDN_INPUT=1    (GDN in_proj_qkv + in_proj_z, default: 1)
- VLLM_EXL3_BATCHED_MLP_GATE_UP=1  (MLP gate_proj + up_proj, default: 1)
"""
from __future__ import annotations

import os
import torch
from dataclasses import replace
from typing import Optional

_MASTER = os.environ.get("VLLM_EXL3_BATCHED_PAIRED_PROJ", "1") == "1"
_GDN_ENABLED = _MASTER and os.environ.get("VLLM_EXL3_BATCHED_GDN_INPUT", "1") == "1"
_MLP_ENABLED = _MASTER and os.environ.get("VLLM_EXL3_BATCHED_MLP_GATE_UP", "1") == "1"

# Cache for batched prepared weights, keyed by (id(w1), id(w2))
_BATCH_CACHE: dict[tuple, object] = {}


def _suh_shared(w1, w2) -> bool:
    """Check if two prepared weights share the same suh (input Hadamard scale)."""
    s1 = getattr(w1, "suh", None)
    s2 = getattr(w2, "suh", None)
    if s1 is None or s2 is None:
        return False
    if s1.shape != s2.shape:
        return False
    return torch.equal(s1, s2)


def _get_batched_weight(w1, w2, tl_module):
    """Get or create a batched prepared weight from two individual ones.

    Uses the original trellis tensors stored on the prepared weights.
    The trellis is stored as a flat tensor on the prepared weight, but we can
    reconstruct the 3D shape from in_features, out_features, and trellis_bits.
    """
    cache_key = (id(w1), id(w2))
    if cache_key in _BATCH_CACHE:
        return _BATCH_CACHE[cache_key]

    # Reconstruct 3D trellis shape from prepared weight metadata
    # trellis shape: [K//16, N//16, 8*bits]
    K = w1.in_features
    N1 = w1.out_features
    N2 = w2.out_features
    bits = getattr(w1, "trellis_bits", 6)
    words_per_tile = 8 * bits

    t1_flat = w1.trellis
    t2_flat = w2.trellis

    # Calculate N//16 for each
    n1_blocks = N1 // 16
    n2_blocks = N2 // 16
    k_blocks = K // 16

    # Reshape to 3D
    expected_size_1 = k_blocks * n1_blocks * words_per_tile
    expected_size_2 = k_blocks * n2_blocks * words_per_tile

    if t1_flat.numel() != expected_size_1 or t2_flat.numel() != expected_size_2:
        return None

    t1_3d = t1_flat.reshape(k_blocks, n1_blocks, words_per_tile)
    t2_3d = t2_flat.reshape(k_blocks, n2_blocks, words_per_tile)

    # Concatenate along N (dim=1)
    batched_trellis = torch.cat([t1_3d, t2_3d], dim=1).contiguous()
    batched_svh = torch.cat([w1.svh, w2.svh], dim=0).contiguous()

    # Prepare batched weight
    try:
        w_batched = tl_module.prepare_weight(
            batched_trellis,
            w1.suh,  # shared suh
            batched_svh,
            codebook=getattr(w1, "trellis_codebook", "mcg"),
            params_dtype=getattr(w1, "params_dtype", torch.float16),
        )
    except Exception:
        return None

    _BATCH_CACHE[cache_key] = w_batched
    return w_batched


def try_batched_run(
    x: torch.Tensor,
    w1,
    w2,
    tl_module,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Try to run two trellis projections as one batched GEMM.

    Returns (out1, out2) if batching succeeded, None to fall back.
    Both w1 and w2 must be prepared trellis weights with the same K, bits,
    codebook, dtype, and shared suh.
    """
    if not _MASTER:
        return None

    # Validate inputs
    if x.ndim != 2 or x.shape[1] != w1.in_features:
        return None
    if w1.in_features != w2.in_features:
        return None

    # Check suh sharing
    if not _suh_shared(w1, w2):
        return None

    # Check trellis compatibility
    if (getattr(w1, "trellis_bits", 0) != getattr(w2, "trellis_bits", 0)
            or getattr(w1, "trellis_codebook", "") != getattr(w2, "trellis_codebook", "")
            or getattr(w1, "params_dtype", None) != getattr(w2, "params_dtype", None)
            or getattr(w1, "weight_layout", "") != getattr(w2, "weight_layout", "")):
        return None

    # Check N alignment (both must be multiples of 128 for trellis tiling)
    n1 = w1.out_features
    n2 = w2.out_features
    if n1 % 128 != 0 or n2 % 128 != 0:
        return None

    # Get or create batched weight
    w_batched = _get_batched_weight(w1, w2, tl_module)
    if w_batched is None:
        return None

    # Run single GEMM
    out_batched = tl_module.run(x, w_batched)

    # Split output
    return out_batched[:, :n1], out_batched[:, n1:]


def try_batched_gdn_input(x, w_qkv, w_z, tl_module):
    """Batch GDN in_proj_qkv + in_proj_z if enabled and suh shared."""
    if not _GDN_ENABLED:
        return None
    return try_batched_run(x, w_qkv, w_z, tl_module)


def try_batched_mlp_gate_up(x, w_gate, w_up, tl_module):
    """Batch MLP gate_proj + up_proj if enabled and suh shared."""
    if not _MLP_ENABLED:
        return None
    return try_batched_run(x, w_gate, w_up, tl_module)


def clear_batch_cache():
    """Clear the batched weight cache."""
    _BATCH_CACHE.clear()


__all__ = [
    "try_batched_run",
    "try_batched_gdn_input",
    "try_batched_mlp_gate_up",
    "clear_batch_cache",
]
