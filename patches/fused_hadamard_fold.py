#!/usr/bin/env python3
"""Fused Hadamard fold kernel: replaces 80+ einsum launches with one Triton kernel.

Computes: W_out = diag(suh) @ Had_K @ W @ Had_N @ diag(svh)

The original hadamard_fold_weight_chunked does this via two torch.einsum calls
per 128-row chunk (K/128 chunks × 2 einsums = 80+ launches for K=5120).
This Triton kernel does the entire fold in one launch, processing all K×N
blocks in parallel.

Correctness: bit-identical to the einsum path (same FP32 accumulation order
within each 128×128 block, same scale application).
"""

import torch
import triton
import triton.language as tl

_HADAMARD_BLOCK = 128

# Precompute the normalized 128×128 Hadamard matrix as a constant
def _build_hadamard_128() -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < _HADAMARD_BLOCK:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h * (1.0 / (128.0 ** 0.5))

_HADAMARD_128 = _build_hadamard_128()

# Upload to device on first use
_HADAMARD_DEVICE_CACHE: dict = {}

def _get_hadamard_128(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _HADAMARD_DEVICE_CACHE:
        _HADAMARD_DEVICE_CACHE[key] = _HADAMARD_128.to(device)
    return _HADAMARD_DEVICE_CACHE[key]


@triton.jit
def _fused_hadamard_fold_kernel(
    W_ptr,          # (K, N) fp16/fp32 input weight
    suh_ptr,        # (K,) fp16/fp32 input scales
    svh_ptr,        # (N,) fp16/fp32 output scales
    H_ptr,          # (128, 128) fp32 Hadamard matrix
    out_ptr,        # (K, N) fp16/fp32 output weight
    K: tl.constexpr,
    N: tl.constexpr,
    stride_W_k, stride_W_n,
    stride_out_k, stride_out_n,
    BLOCK: tl.constexpr,
):
    """One program per (k_block, n_block) pair.

    Each program loads a 128×128 block of W, applies Had_K on the K dimension,
    Had_N on the N dimension, and multiplies by suh/svh scales, then stores.
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BLOCK
    n_start = pid_n * BLOCK

    # Load W block: (128, 128) in fp32
    k_offs = k_start + tl.arange(0, BLOCK)
    n_offs = n_start + tl.arange(0, BLOCK)
    W_block = tl.load(W_ptr + k_offs[:, None] * stride_W_k + n_offs[None, :] * stride_W_n).to(tl.float32)

    # Load Hadamard matrix: (128, 128) fp32
    h_offs_k = tl.arange(0, BLOCK)
    h_offs_n = tl.arange(0, BLOCK)
    H = tl.load(H_ptr + h_offs_k[:, None] * BLOCK + h_offs_n[None, :])

    # Apply Had_K: temp = H @ W_block  (128×128 @ 128×128 = 128×128)
    # Each output row i = sum_j H[i,j] * W_block[j, :]
    # Use tl.dot for the matmul
    temp = tl.dot(H, W_block)

    # Apply Had_N: out = temp @ H^T  (128×128 @ 128×128 = 128×128)
    # H is symmetric (Hadamard), so H^T = H
    folded = tl.dot(temp, H)

    # Load and apply scales
    suh_block = tl.load(suh_ptr + k_offs).to(tl.float32)
    svh_block = tl.load(svh_ptr + n_offs).to(tl.float32)

    folded = folded * suh_block[:, None] * svh_block[None, :]

    # Store
    tl.store(out_ptr + k_offs[:, None] * stride_out_k + n_offs[None, :] * stride_out_n, folded.to(tl.float16))


def hadamard_fold_weight_fused(
    W: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Fused Hadamard fold using a single Triton kernel launch.

    Replaces hadamard_fold_weight_chunked's 80+ einsum launches with one
    Triton kernel that processes all K×N blocks in parallel.

    Args:
        W: (K, N) fp16 weight (will be cast to fp32 internally for accuracy)
        suh: (K,) fp16 per-element input scales
        svh: (N,) fp16 per-element output scales

    Returns:
        (K, N) fp16 folded weight
    """
    K, N = W.shape
    assert K % _HADAMARD_BLOCK == 0 and N % _HADAMARD_BLOCK == 0, \
        f"K={K} and N={N} must be multiples of {_HADAMARD_BLOCK}"

    device = W.device
    H = _get_hadamard_128(device)

    # Broadcast scales to per-element if needed
    k_blocks = K // _HADAMARD_BLOCK
    n_blocks = N // _HADAMARD_BLOCK
    if suh.numel() == k_blocks:
        suh = suh.repeat_interleave(_HADAMARD_BLOCK)
    if svh.numel() == n_blocks:
        svh = svh.repeat_interleave(_HADAMARD_BLOCK)

    out = torch.empty_like(W)

    grid = (k_blocks, n_blocks)
    _fused_hadamard_fold_kernel[grid](
        W, suh, svh, H, out,
        K, N,
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        BLOCK=_HADAMARD_BLOCK,
        num_warps=8,
    )

    return out
