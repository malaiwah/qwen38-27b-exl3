"""Batched paired/N-way trellis projections with auto-detection.

Automatically detects when projections share the same suh (input Hadamard
scale) and batches them into a single GEMM. Handles:

1. GDN input: in_proj_qkv + in_proj_z (2-way)
2. MLP gate/up: gate_proj + up_proj (2-way)
3. Attention QKV: q_proj + k_proj + v_proj (3-way)
4. Any N-way group of projections sharing the same input and suh

When suh is shared, trellis weights are concatenated along N and run as
a single GEMM. When suh differs, two fallback options:
  a) Fall back to separate GEMMs (default, zero overhead)
  b) Pre-fold to BF16 at load time and batch (env-gated, costs VRAM)

Env gates:
- VLLM_EXL3_BATCHED_PAIRED_PROJ=1   (master switch, default: 1)
- VLLM_EXL3_BATCHED_GDN_INPUT=1     (GDN in_proj_qkv + in_proj_z)
- VLLM_EXL3_BATCHED_MLP_GATE_UP=1   (MLP gate_proj + up_proj)
- VLLM_EXL3_BATCHED_ATTN_QKV=1      (Attn q_proj + k_proj + v_proj)
- VLLM_EXL3_BATCHED_PREFOLD_BF16=0  (Pre-fold to BF16 when suh differs, default: 0)
- VLLM_EXL3_BATCHED_AUDIT=0         (Log suh-sharing audit at load time, default: 0)
"""
from __future__ import annotations

import os
import torch
import logging
from typing import Optional, Sequence

_MASTER = os.environ.get("VLLM_EXL3_BATCHED_PAIRED_PROJ", "1") == "1"
_GDN_ENABLED = _MASTER and os.environ.get("VLLM_EXL3_BATCHED_GDN_INPUT", "1") == "1"
_MLP_ENABLED = _MASTER and os.environ.get("VLLM_EXL3_BATCHED_MLP_GATE_UP", "1") == "1"
_QKV_ENABLED = _MASTER and os.environ.get("VLLM_EXL3_BATCHED_ATTN_QKV", "1") == "1"
_PREFOLD_BF16 = False  # Disabled — Hadamard rotation extraction not implemented
_AUDIT = os.environ.get("VLLM_EXL3_BATCHED_AUDIT", "0") == "1"

_log = logging.getLogger(__name__)

_BATCH_CACHE: dict[tuple, object] = {}
# _BF16_CACHE removed (pre-fold path disabled)


def _suh_shared(weights: Sequence) -> bool:
    """Check if ALL weights share the same suh."""
    if len(weights) < 2:
        return False
    s0 = getattr(weights[0], "suh", None)
    if s0 is None:
        return False
    for w in weights[1:]:
        s = getattr(w, "suh", None)
        if s is None or s.shape != s0.shape or not torch.equal(s, s0):
            return False
    return True


def _all_compatible(weights: Sequence) -> bool:
    """Check if all weights have compatible trellis params (bits, codebook, dtype, layout, K)."""
    if len(weights) < 2:
        return False
    w0 = weights[0]
    for w in weights[1:]:
        if (getattr(w, "trellis_bits", 0) != getattr(w0, "trellis_bits", 0)
                or getattr(w, "trellis_codebook", "") != getattr(w0, "trellis_codebook", "")
                or getattr(w, "params_dtype", None) != getattr(w0, "params_dtype", None)
                or getattr(w, "weight_layout", "") != getattr(w0, "weight_layout", "")
                or w.in_features != w0.in_features):
            return False
    # All N must be multiples of 128
    for w in weights:
        if w.out_features % 128 != 0:
            return False
    return True


def _get_batched_weight(weights: Sequence, tl_module):
    """Create a batched prepared weight from N individual ones."""
    cache_key = tuple(id(w) for w in weights)
    if cache_key in _BATCH_CACHE:
        return _BATCH_CACHE[cache_key]

    w0 = weights[0]
    K = w0.in_features
    bits = getattr(w0, "trellis_bits", 6)
    words_per_tile = 8 * bits
    k_blocks = K // 16

    # Reshape each trellis to 3D and concatenate along N
    trellis_3d_list = []
    for w in weights:
        n_blocks = w.out_features // 16
        expected = k_blocks * n_blocks * words_per_tile
        if w.trellis.numel() != expected:
            return None
        t_3d = w.trellis.reshape(k_blocks, n_blocks, words_per_tile)
        trellis_3d_list.append(t_3d)

    batched_trellis = torch.cat(trellis_3d_list, dim=1).contiguous()
    batched_svh = torch.cat([w.svh for w in weights], dim=0).contiguous()

    try:
        w_batched = tl_module.prepare_weight(
            batched_trellis,
            w0.suh,
            batched_svh,
            codebook=getattr(w0, "trellis_codebook", "mcg"),
            params_dtype=getattr(w0, "params_dtype", torch.float16),
        )
    except Exception:
        return None

    _BATCH_CACHE[cache_key] = w_batched
    return w_batched


def _get_bf16_weight(w, tl_module):
    """Disabled — see note in try_batched_run_n."""
    return None


    This is expensive (O(K*N) GEMM) but done once at load time.
    The BF16 weight is cached.
    """
    cache_key = id(w)
    if cache_key in _BF16_CACHE:
        return _BF16_CACHE[cache_key]

    K = w.in_features
    N = w.out_features

    # Dequantize by running trellis GEMM on chunks of identity matrix
    # Process K in chunks of 512 to avoid OOM
    chunk = 512
    bf16_weight = torch.empty(K, N, dtype=getattr(w, "params_dtype", torch.float16), device="cuda")
    for start in range(0, K, chunk):
        end = min(start + chunk, K)
        eye_chunk = torch.eye(end - start, K, dtype=bf16_weight.dtype, device="cuda")
        # Run trellis GEMM: (chunk, K) @ (K, N) → (chunk, N)
        out = tl_module.run(eye_chunk, w)
        bf16_weight[start:end] = out

    _BF16_CACHE[cache_key] = bf16_weight
    return bf16_weight


def try_batched_run_n(
    x: torch.Tensor,
    weights: Sequence,
    tl_module,
) -> Optional[list[torch.Tensor]]:
    """Try to run N trellis projections as one batched GEMM.

    Returns list of N output tensors if batching succeeded, None to fall back.
    All weights must have the same K, bits, codebook, dtype, layout, and shared suh.
    """
    if not _MASTER or len(weights) < 2:
        return None

    w0 = weights[0]
    if x.ndim != 2 or x.shape[1] != w0.in_features:
        return None
    if x.dtype != getattr(w0, "params_dtype", torch.float16):
        return None

    if not _all_compatible(weights):
        return None

    suh_ok = _suh_shared(weights)

    if suh_ok:
        # Batched trellis path
        w_batched = _get_batched_weight(weights, tl_module)
        if w_batched is None:
            return None
        out_batched = tl_module.run(x, w_batched)
        # Split output
        outputs = []
        offset = 0
        for w in weights:
            n = w.out_features
            outputs.append(out_batched[:, offset:offset + n])
            offset += n
        return outputs

    # NOTE: Pre-folded BF16 path removed. The trellis GEMM bakes Hadamard-128
    # rotations into both input and output, so running identity through the
    # trellis path does NOT yield a plain matmul weight. Extracting the raw
    # dequantized weight (without H128) requires modifying the trellis decode
    # kernel to skip rotations — left as future work.
    # When suh differs, the fallback is the original separate-GEMM path.

    return None


# Convenience wrappers

def try_batched_gdn_input(x, w_qkv, w_z, tl_module):
    """Batch GDN in_proj_qkv + in_proj_z."""
    if not _GDN_ENABLED:
        return None
    result = try_batched_run_n(x, [w_qkv, w_z], tl_module)
    if result is None:
        return None
    return result[0], result[1]


def try_batched_mlp_gate_up(x, w_gate, w_up, tl_module):
    """Batch MLP gate_proj + up_proj."""
    if not _MLP_ENABLED:
        return None
    result = try_batched_run_n(x, [w_gate, w_up], tl_module)
    if result is None:
        return None
    return result[0], result[1]


def try_batched_qkv(x, w_q, w_k, w_v, tl_module):
    """Batch attention q_proj + k_proj + v_proj (3-way)."""
    if not _QKV_ENABLED:
        return None
    result = try_batched_run_n(x, [w_q, w_k, w_v], tl_module)
    if result is None:
        return None
    return result[0], result[1], result[2]


def audit_suh_sharing(layer_idx: int, weights: dict[str, object], tl_module=None):
    """Log which projection groups could benefit from batching.

    Args:
        layer_idx: layer index for logging
        weights: dict mapping module name → prepared weight
        tl_module: the trellis_linear module (for BF16 dequant if needed)
    """
    if not _AUDIT:
        return

    # Group by K (same input dimension)
    by_k: dict[int, list[str]] = {}
    for name, w in weights.items():
        k = getattr(w, "in_features", 0)
        by_k.setdefault(k, []).append(name)

    for k, mods in by_k.items():
        if len(mods) < 2:
            continue
        ws = [weights[m] for m in mods]
        shared = _suh_shared(ws)
        compatible = _all_compatible(ws)
        if compatible:
            n_total = sum(w.out_features for w in ws)
            status = "SHARED suh → can batch" if shared else "different suh → needs pre-fold"
            _log.info(
                "Batched proj audit layer %d K=%d: %s → N_batched=%d (%s)",
                layer_idx, k, mods, n_total, status,
            )


def clear_caches():
    """Clear all caches."""
    _BATCH_CACHE.clear()
    # _BF16_CACHE removed


__all__ = [
    "try_batched_run_n",
    "try_batched_gdn_input",
    "try_batched_mlp_gate_up",
    "try_batched_qkv",
    "audit_suh_sharing",
    "clear_caches",
]
