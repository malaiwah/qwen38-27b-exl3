"""Pre-folded BF16 batched projections.

Extracts the full dequantized weight (including Hadamard rotations and suh/svh
scales) from each trellis projection at load time using M=1 identity extraction.
The resulting BF16 weight enables a plain matmul at decode time, which:
  1. Eliminates trellis decode overhead (~200us → ~10us per projection)
  2. Enables batching projections with DIFFERENT suh into one matmul
  3. Removes the input Hadamard rotation from the decode hot path

Extraction uses M=1 (one basis vector at a time) to match the decode
accumulation order exactly. Cost: ~4s per projection per layer at load time.

VRAM cost (BF16, 2 bytes/element):
  - QKV triple: ~2.3 GB (16 full-attn layers)
  - GDN pair: ~11 GB (48 GDN layers)
  - MLP gate+up: ~23 GB (64 layers)
  - Total: ~36 GB

On RTX 5090 (32 GB): enable only QKV (fits in ~2 GB)
On RTX 6000 (96 GB): enable everything

Env gates:
  VLLM_EXL3_PREFOLD_BF16=0           (master switch, default: 0)
  VLLM_EXL3_PREFOLD_QKV=0            (QKV triple, default: 0)
  VLLM_EXL3_PREFOLD_GDN=0            (GDN pair, default: 0)
  VLLM_EXL3_PREFOLD_MLP=0            (MLP gate+up, default: 0)
  VLLM_EXL3_PREFOLD_VRAM_BUDGET_MB=0 (max VRAM for pre-folded weights, 0=auto)
"""
from __future__ import annotations

import os
import torch
import logging
from typing import Optional, Sequence

_MASTER = os.environ.get("VLLM_EXL3_PREFOLD_BF16", "0") == "1"
_QKV = _MASTER and os.environ.get("VLLM_EXL3_PREFOLD_QKV", "0") == "1"
_GDN = _MASTER and os.environ.get("VLLM_EXL3_PREFOLD_GDN", "0") == "1"
_MLP = _MASTER and os.environ.get("VLLM_EXL3_PREFOLD_MLP", "0") == "1"
_VRAM_BUDGET_MB = int(os.environ.get("VLLM_EXL3_PREFOLD_VRAM_BUDGET_MB", "0"))

_log = logging.getLogger(__name__)

# Cache: id(prepared_weight) → BF16 weight tensor (K, N)
_W_FULL_CACHE: dict[int, torch.Tensor] = {}

# Cache: tuple of ids → concatenated BF16 weight (K, N_total)
_BATCH_CACHE: dict[tuple, torch.Tensor] = {}


def _extract_w_full(w, tl_module) -> Optional[torch.Tensor]:
    """Extract full BF16 weight from a prepared trellis weight using M=1 identity.

    Returns (K, N) BF16 tensor such that x @ W_full == trellis_path(x).
    """
    cache_key = id(w)
    if cache_key in _W_FULL_CACHE:
        return _W_FULL_CACHE[cache_key]

    K = w.in_features
    N = w.out_features
    dtype = getattr(w, "params_dtype", torch.bfloat16)

    W_full = torch.empty(K, N, dtype=dtype, device="cuda")
    for i in range(K):
        e_i = torch.zeros(1, K, dtype=dtype, device="cuda")
        e_i[0, i] = 1.0
        W_full[i] = tl_module.run(e_i, w)[0]

    _W_FULL_CACHE[cache_key] = W_full
    _log.info("Pre-folded BF16: extracted W_full (%d, %d) = %.1f MB",
              K, N, W_full.numel() * 2 / 1e6)
    return W_full


def _check_vram(weights: Sequence) -> bool:
    """Check if we have enough VRAM for the pre-folded weights."""
    if _VRAM_BUDGET_MB <= 0:
        return True  # No budget set, allow

    total_mb = sum(w.in_features * w.out_features * 2 for w in weights) / 1e6
    if total_mb > _VRAM_BUDGET_MB:
        _log.warning("Pre-folded BF16: needs %.1f MB but budget is %d MB, skipping",
                     total_mb, _VRAM_BUDGET_MB)
        return False
    return True


def try_prefolded_batched(
    x: torch.Tensor,
    weights: Sequence,
    tl_module,
) -> Optional[list[torch.Tensor]]:
    """Run N projections as one BF16 matmul using pre-folded weights.

    Returns list of N output tensors, or None to fall back.
    Works even when suh differs between projections (unlike trellis batching).
    """
    if not _MASTER or len(weights) < 2:
        return None

    w0 = weights[0]
    if x.ndim != 2 or x.shape[1] != w0.in_features:
        return None
    if x.dtype != getattr(w0, "params_dtype", torch.bfloat16):
        return None

    # Check K compatibility
    for w in weights:
        if w.in_features != w0.in_features:
            return None

    if not _check_vram(weights):
        return None

    # Get or create batched BF16 weight
    cache_key = tuple(id(w) for w in weights)
    if cache_key not in _BATCH_CACHE:
        # Extract each W_full and concatenate
        w_fulls = []
        for w in weights:
            wf = _extract_w_full(w, tl_module)
            if wf is None:
                return None
            w_fulls.append(wf)
        _BATCH_CACHE[cache_key] = torch.cat(w_fulls, dim=1).contiguous()

    W_batched = _BATCH_CACHE[cache_key]

    # Single matmul
    out_batched = torch.mm(x, W_batched)

    # Split output
    outputs = []
    offset = 0
    for w in weights:
        n = w.out_features
        outputs.append(out_batched[:, offset:offset + n])
        offset += n
    return outputs


def try_prefolded_qkv(x, w_q, w_k, w_v, tl_module):
    """Pre-folded BF16 batched QKV (3-way)."""
    if not _QKV:
        return None
    result = try_prefolded_batched(x, [w_q, w_k, w_v], tl_module)
    if result is None:
        return None
    return result[0], result[1], result[2]


def try_prefolded_gdn(x, w_qkv, w_z, tl_module):
    """Pre-folded BF16 batched GDN input (2-way)."""
    if not _GDN:
        return None
    result = try_prefolded_batched(x, [w_qkv, w_z], tl_module)
    if result is None:
        return None
    return result[0], result[1]


def try_prefolded_mlp(x, w_gate, w_up, tl_module):
    """Pre-folded BF16 batched MLP gate+up (2-way)."""
    if not _MLP:
        return None
    result = try_prefolded_batched(x, [w_gate, w_up], tl_module)
    if result is None:
        return None
    return result[0], result[1]


def clear_caches():
    """Clear all caches."""
    _W_FULL_CACHE.clear()
    _BATCH_CACHE.clear()


__all__ = [
    "try_prefolded_batched",
    "try_prefolded_qkv",
    "try_prefolded_gdn",
    "try_prefolded_mlp",
    "clear_caches",
]
