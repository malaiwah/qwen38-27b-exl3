"""Pre-folded BF16 batched projections with async extraction.

Extracts the full dequantized weight (including Hadamard rotations and suh/svh
scales) from each trellis projection at load time using M=1 identity extraction.
The resulting BF16 weight enables a plain matmul at decode time.

Key insight: extraction MUST use M=1 to match the decode accumulation order.
M>1 gives garbage (cosine ≈ 0) because the trellis kernel's split-K tiling
and persistent GEMM scheduler are M-dependent.

Speed: ~1.6s per projection (K=5120), ~5.5s for K=17408 (down_proj).
Uses async kernel pipelining: launches all M=1 kernels without per-call sync,
then synchronizes once at the end. Double-buffers output to avoid races.

VRAM costs (BF16, 2 bytes/element):
  - lm_head:        2.5 GB  (K=5120, N=248320)
  - QKV (×16 layers): 2.3 GB (K=5120, N=14336)
  - down_proj (×64):  11.4 GB (K=17408, N=5120)
  - GDN pair (×48):   11.0 GB
  - MLP gate+up (×64): 22.8 GB
  - ALL:              ~50 GB

Env gates:
  VLLM_EXL3_PREFOLD_BF16=0           (master switch)
  VLLM_EXL3_PREFOLD_QKV=0            (QKV triple)
  VLLM_EXL3_PREFOLD_GDN=0            (GDN pair)
  VLLM_EXL3_PREFOLD_MLP=0            (MLP gate+up)
  VLLM_EXL3_PREFOLD_DOWN=0           (down_proj)
  VLLM_EXL3_PREFOLD_LM_HEAD=0        (lm_head)
  VLLM_EXL3_PREFOLD_VRAM_BUDGET_MB=0 (max VRAM, 0=unlimited)
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
_DOWN = _MASTER and os.environ.get("VLLM_EXL3_PREFOLD_DOWN", "0") == "1"
_LM_HEAD = _MASTER and os.environ.get("VLLM_EXL3_PREFOLD_LM_HEAD", "0") == "1"
_VRAM_BUDGET_MB = int(os.environ.get("VLLM_EXL3_PREFOLD_VRAM_BUDGET_MB", "0"))

_log = logging.getLogger(__name__)

_W_FULL_CACHE: dict[int, torch.Tensor] = {}
_BATCH_CACHE: dict[tuple, torch.Tensor] = {}
_N_DOUBLE_BUFS = 8  # double-buffer depth for async extraction


def _extract_w_full(w, tl_module) -> Optional[torch.Tensor]:
    """Extract full BF16 weight from a prepared trellis weight using M=1 identity.

    Uses async kernel pipelining: launches all M=1 kernels without per-call
    sync, then synchronizes once. Double-buffers output to avoid write races.
    """
    cache_key = id(w)
    if cache_key in _W_FULL_CACHE:
        return _W_FULL_CACHE[cache_key]

    K = w.in_features
    N = w.out_features
    dtype = getattr(w, "params_dtype", torch.bfloat16)

    _log.info("Pre-folded BF16: extracting (%d, %d) = %.1f MB...", K, N, K * N * 2 / 1e6)

    # Pre-allocate identity matrix and output
    eyes = torch.eye(K, dtype=dtype, device="cuda")  # (K, K)
    W_full = torch.empty(K, N, dtype=dtype, device="cuda")

    # Double-buffer approach: cycle through N output buffers to avoid races
    out_bufs = [torch.empty(1, N, dtype=dtype, device="cuda") for _ in range(_N_DOUBLE_BUFS)]

    # Launch all K kernels without sync
    for i in range(K):
        buf = out_bufs[i % _N_DOUBLE_BUFS]
        # tl.run with output=buf writes directly into buf (no allocation)
        tl_module.run(eyes[i:i + 1], w, output=buf)
        # Async copy to W_full (won't race because buf is reused only after
        # _N_DOUBLE_BUFS more iterations, by which time the copy is done)
        W_full[i] = buf[0]

    # Single sync at the end
    torch.cuda.synchronize()

    _W_FULL_CACHE[cache_key] = W_full
    _log.info("Pre-folded BF16: extracted (%d, %d) = %.1f MB", K, N, W_full.numel() * 2 / 1e6)
    return W_full


def _check_vram(weights: Sequence) -> bool:
    """Check if we have enough VRAM budget for the pre-folded weights."""
    if _VRAM_BUDGET_MB <= 0:
        return True
    total_mb = sum(w.in_features * w.out_features * 2 for w in weights) / 1e6
    if total_mb > _VRAM_BUDGET_MB:
        _log.warning("Pre-folded BF16: needs %.1f MB but budget is %d MB, skipping",
                     total_mb, _VRAM_BUDGET_MB)
        return False
    return True


def _all_same_k(weights: Sequence) -> bool:
    if len(weights) < 2:
        return False
    k0 = weights[0].in_features
    return all(w.in_features == k0 for w in weights)


def try_prefolded_batched(
    x: torch.Tensor,
    weights: Sequence,
    tl_module,
) -> Optional[list[torch.Tensor]]:
    """Run N projections as one BF16 matmul using pre-folded weights.

    Works even when suh differs between projections.
    Returns list of N output tensors, or None to fall back.
    """
    if not _MASTER or len(weights) < 2:
        return None

    w0 = weights[0]
    if x.ndim != 2 or x.shape[1] != w0.in_features:
        return None
    if x.dtype != getattr(w0, "params_dtype", torch.bfloat16):
        return None
    if not _all_same_k(weights):
        return None
    if not _check_vram(weights):
        return None

    # Get or create batched BF16 weight
    cache_key = tuple(id(w) for w in weights)
    if cache_key not in _BATCH_CACHE:
        w_fulls = []
        for w in weights:
            wf = _extract_w_full(w, tl_module)
            if wf is None:
                return None
            w_fulls.append(wf)
        _BATCH_CACHE[cache_key] = torch.cat(w_fulls, dim=1).contiguous()

    W_batched = _BATCH_CACHE[cache_key]
    out_batched = torch.mm(x, W_batched)

    # Split output
    outputs = []
    offset = 0
    for w in weights:
        n = w.out_features
        outputs.append(out_batched[:, offset:offset + n])
        offset += n
    return outputs


def try_prefolded_single(
    x: torch.Tensor,
    w,
    tl_module,
) -> Optional[torch.Tensor]:
    """Run a single projection via pre-folded BF16 matmul.

    Used for lm_head, down_proj, o_proj, out_proj — projections that
    aren't part of a batchable group but still benefit from eliminating
    the 200us trellis floor tax.
    """
    if not _MASTER:
        return None
    if x.ndim != 2 or x.shape[1] != w.in_features:
        return None
    if x.dtype != getattr(w, "params_dtype", torch.bfloat16):
        return None

    wf = _extract_w_full(w, tl_module)
    if wf is None:
        return None
    return torch.mm(x, wf)


def try_prefolded_qkv(x, w_q, w_k, w_v, tl_module):
    if not _QKV:
        return None
    result = try_prefolded_batched(x, [w_q, w_k, w_v], tl_module)
    if result is None:
        return None
    return result[0], result[1], result[2]


def try_prefolded_gdn(x, w_qkv, w_z, tl_module):
    if not _GDN:
        return None
    result = try_prefolded_batched(x, [w_qkv, w_z], tl_module)
    if result is None:
        return None
    return result[0], result[1]


def try_prefolded_mlp(x, w_gate, w_up, tl_module):
    if not _MLP:
        return None
    result = try_prefolded_batched(x, [w_gate, w_up], tl_module)
    if result is None:
        return None
    return result[0], result[1]


def try_prefolded_down(x, w_down, tl_module):
    """Pre-folded BF16 for down_proj (single, not batched)."""
    if not _DOWN:
        return None
    return try_prefolded_single(x, w_down, tl_module)


def try_prefolded_lm_head(x, w_lm_head, tl_module):
    """Pre-folded BF16 for lm_head (single, not batched)."""
    if not _LM_HEAD:
        return None
    return try_prefolded_single(x, w_lm_head, tl_module)


def precompile_warmup(weights_by_shape: dict, tl_module):
    """Pre-compile all trellis shapes to eliminate first-request JIT latency.

    Args:
        weights_by_shape: dict mapping (K, N, bits) → prepared weight
        tl_module: the trellis_linear module
    """
    for shape, w in weights_by_shape.items():
        K, N, bits = shape
        _log.info("Pre-compile warmup: shape K=%d N=%d bits=%d", K, N, bits)
        dummy_x = torch.zeros(1, K, dtype=getattr(w, "params_dtype", torch.float16),
                              device="cuda")
        tl_module.run(dummy_x, w)
    torch.cuda.synchronize()
    _log.info("Pre-compile warmup complete: %d shapes compiled", len(weights_by_shape))


def clear_caches():
    _W_FULL_CACHE.clear()
    _BATCH_CACHE.clear()


__all__ = [
    "try_prefolded_batched",
    "try_prefolded_single",
    "try_prefolded_qkv",
    "try_prefolded_gdn",
    "try_prefolded_mlp",
    "try_prefolded_down",
    "try_prefolded_lm_head",
    "precompile_warmup",
    "clear_caches",
]
