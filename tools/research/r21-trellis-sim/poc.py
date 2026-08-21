#!/usr/bin/env python3
"""
R21: Simplified Trellis / Viterbi Codebook Simulation

R16 found per-tile non-uniform codebooks are NOT worth it at matched bytes:
the per-tile codebook storage (2^K * 2 bytes/tile) is prohibitive (7.7-60%
overhead, exponential in K). But R16 explicitly noted: "EXL3 trellis coding
uses a SHARED codebook (O(1) storage). A shared-codebook experiment would be
needed to assess whether non-uniform quantization is worthwhile in the
trellis framework."

This POC bridges that gap. Key design decisions informed by external feedback:

- EXL3 ALREADY has Hadamard+signs+LDLQ incoherence processing. Our BiIP+
  Hadamard rotation approximates this. The novel question for codebooks is:
  does rotation (which homogenizes tile distributions) make a shared codebook
  more effective? If all tiles look the same after rotation, one codebook
  fits all → O(1) storage suffices.

- "B (BiP) changes the R-D curve, not just adds half a bit." For codebooks,
  preconditioning changes the codebook fit: post-rotation, the global weight
  distribution is more compact and uniform-like, so a shared codebook
  designed from the global distribution should approximate per-tile quality.

- Multi-precision alphabet (K4, K4+B, K5, K5+B, K6, K6+B): our multi-level
  shared codebook experiment directly implements this — multiple shared
  codebooks at different K values, with DP allocation assigning tiles.

Experiments:
1. SHARED CODEBOOK STRATEGIES: per-tile uniform (baseline), shared uniform,
   shared Lloyd-Max, shared k-means — with and without per-tile scale.
2. SIMPLIFIED VITERBI: DP over element sequence with transition penalty
   (trellis smoothness constraint). Compare to independent nearest-level.
3. SHARED CODEBOOK + ROTATION: Does BiIP+Hadamard make shared codebook
   more effective? Compare with/without rotation at matched bytes.
4. MULTI-LEVEL SHARED CODEBOOK + ALLOCATION: Multiple shared codebooks at
   different K values. DP allocation assigns each tile to a codebook.
   This is the multi-precision allocator with B as changing codebook fit.
5. MATCHED-BYTE DP: Per-tile uniform DP vs multi-level shared DP at
   identical total byte budgets.

Byte budget accounting (exact):
  Per-tile uniform:    payload + n_tiles*8 (min/max) + metadata
  Shared + tile scale: payload + n_tiles*8 (min/max) + 2^K*2 (codebook once) + metadata
  Shared (no scale):   payload + 2^K*2 (codebook once) + metadata
  Multi-level shared:  payload + n_tiles*8 (min/max) + sum(2^K_i*2) (codebooks) + metadata
"""

import json
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

TILE = 16
M_DIM = 128
N_DIM = 128
K_VALUES = [3, 4, 5, 6]
K_MIN = min(K_VALUES)
K_MAX = max(K_VALUES)
P_CAL = 512
N_TILES = (M_DIM // TILE) * (N_DIM // TILE)  # 64
EPS_PER_TILE = TILE * TILE  # 256
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r21-trellis-sim/receipts/research/r21-trellis-sim-results.json"

TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]

# Viterbi transition penalty sweep
ALPHA_VALUES = [0.0, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]

# ============================================================================
# Weight loading and slicing (from R16)
# ============================================================================

def load_real_weights():
    """Load real Qwen3.8-27B BF16 weights (correctly decoded)."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for name in TENSOR_NAMES:
        if name in data:
            tensors[name] = data[name].astype(np.float64)
    return tensors


def extract_slices(tensor, m=128, n=128, seed=42):
    """Extract 3 128x128 slices: first (top-left), mid (center), random."""
    M, N = tensor.shape
    slices = {}
    slices["first"] = tensor[:m, :n].copy()
    mid_r, mid_c = M // 2 - m // 2, N // 2 - n // 2
    slices["mid"] = tensor[mid_r:mid_r + m, mid_c:mid_c + n].copy()
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, M - m + 1)
    c0 = rng.integers(0, N - n + 1)
    slices["rand"] = tensor[r0:r0 + m, c0:c0 + n].copy()
    return slices


# ============================================================================
# Synthetic Hessian generation (from R3/R16)
# ============================================================================

def synthetic_hessians(W, n_samples=512, outlier_fraction=0.05,
                       outlier_scale=10.0, seed=42):
    """Generate synthetic activation Hessian H_X and output Hessian proxy H_G."""
    M, N = W.shape
    rng = np.random.default_rng(seed)
    act = rng.standard_normal((n_samples, N))
    n_out = int(n_samples * outlier_fraction)
    if n_out > 0:
        out_idx = rng.choice(n_samples, n_out, replace=False)
        act[out_idx] *= outlier_scale
    H_X = act.T @ act / n_samples + 0.01 * np.eye(N)
    Y = W @ act.T
    H_G = Y @ Y.T / n_samples + 0.01 * np.eye(M)
    return H_X, H_G


# ============================================================================
# BiIP + Hadamard rotation (from R3/R16)
# ============================================================================

def hadamard_matrix(n):
    assert (n & (n - 1)) == 0
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    return H @ np.diag(signs), signs


def biip_scaling(W, H_X, H_G):
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_G = np.diag(sg_diag)
    W_t = S_G @ W @ S_X
    sidecar = (d_in + d_out) * 4
    return S_G, S_X, W_t, sidecar


def apply_rotation(W, H_X, H_G, rng):
    d_out, d_in = W.shape
    sidecar = 0
    S_G, S_X, W_t, sc = biip_scaling(W, H_X, H_G)
    sidecar += sc
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = S_X_inv @ H_X @ S_X_inv
    H_G_t = S_G_inv @ H_G @ S_G_inv
    V, _ = signed_random_hadamard(d_in, rng)
    sidecar += d_in // 8 + 1
    W_t = W_t @ V.T
    H_X_t = V @ H_X_t @ V.T
    U, _ = signed_random_hadamard(d_out, rng)
    sidecar += d_out // 8 + 1
    W_t = U @ W_t
    H_G_t = U @ H_G_t @ U.T
    return W_t, H_X_t, H_G_t, U, V, S_G, S_X, sidecar


def inverse_rotation(W_q, U, V, S_G, S_X):
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ W_q @ V @ S_X_inv


# ============================================================================
# Evaluation metrics (from R16)
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """Primary metric: tr(H_G . E . H_X . E^T)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    """Raw weight MSE."""
    return float(np.mean(E ** 2))


def round_to_float16(levels):
    """Round codebook levels through float16 (the declared wire dtype).

    Codebooks are designed in float64 but stored as float16. This function
    simulates the serialization round-trip so evaluation reflects the
    actual stored precision.
    """
    return np.sort(levels.astype(np.float16).astype(np.float64))


def index_entropy(indices):
    """Shannon entropy of index sequence in bits."""
    vals, counts = np.unique(indices, return_counts=True)
    p = counts / len(indices)
    return float(-np.sum(p * np.log2(np.maximum(p, 1e-20))))


# ============================================================================
# Per-tile uniform quantizer (baseline, from R16)
# ============================================================================

def quantize_uniform_tile(w, k):
    """Per-tile uniform min/max quantizer. k bits -> 2^k levels."""
    if k <= 0:
        return np.zeros_like(w), 8.0, np.zeros_like(w, dtype=int)
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo), 8.0, np.zeros_like(w, dtype=int)
    step = (hi - lo) / (nl - 1)
    idx = np.clip(np.round((w - lo) / step), 0, nl - 1).astype(int)
    q = idx * step + lo
    return q, 8.0, idx


# ============================================================================
# Shared codebook design
# ============================================================================

def design_global_uniform(all_weights, k):
    """Uniform codebook from global min/max. Returns 2^k sorted levels."""
    nl = 2 ** k
    lo, hi = float(all_weights.min()), float(all_weights.max())
    if hi - lo < 1e-15:
        return np.full(nl, lo)
    return np.linspace(lo, hi, nl)


def design_global_lloyd_max(all_weights, k, max_iter=200, tol=1e-12):
    """Lloyd-Max on pooled weights. Returns 2^k sorted levels."""
    nl = 2 ** k
    w = all_weights.flatten()
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full(nl, lo)
    levels = np.linspace(lo, hi, nl)
    prev_mse = float('inf')
    for it in range(max_iter):
        sorted_lv = np.sort(levels)
        boundaries = 0.5 * (sorted_lv[:-1] + sorted_lv[1:])
        idx = np.clip(np.searchsorted(boundaries, w), 0, nl - 1)
        new_levels = np.zeros(nl)
        for i in range(nl):
            mask = (idx == i)
            new_levels[i] = np.mean(w[mask]) if np.any(mask) else sorted_lv[i]
        cur_mse = np.mean((w - new_levels[idx]) ** 2)
        levels = new_levels
        if abs(prev_mse - cur_mse) < tol * max(prev_mse, 1e-20):
            break
        prev_mse = cur_mse
    return np.sort(levels)


def design_global_kmeans(all_weights, k, max_iter=200, tol=1e-12, seed=42):
    """k-means++ on pooled weights. Returns 2^k sorted levels."""
    nl = 2 ** k
    w = all_weights.flatten()
    n = len(w)
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full(nl, lo)
    rng = np.random.default_rng(seed)
    centers = [w[rng.integers(n)]]
    for _ in range(1, nl):
        dists = np.min(np.stack([(w - c) ** 2 for c in centers]), axis=0)
        total = np.sum(dists)
        if total < 1e-15:
            centers.append(w[rng.integers(n)])
        else:
            probs = dists / total
            centers.append(w[rng.choice(n, p=probs)])
    levels = np.array(centers, dtype=np.float64)
    prev_mse = float('inf')
    for it in range(max_iter):
        sorted_lv = np.sort(levels)
        boundaries = 0.5 * (sorted_lv[:-1] + sorted_lv[1:])
        idx = np.clip(np.searchsorted(boundaries, w), 0, nl - 1)
        new_levels = np.zeros(nl)
        for i in range(nl):
            mask = (idx == i)
            new_levels[i] = np.mean(w[mask]) if np.any(mask) else sorted_lv[i]
        cur_mse = np.mean((w - new_levels[np.clip(idx, 0, nl - 1)]) ** 2)
        levels = new_levels
        if abs(prev_mse - cur_mse) < tol * max(prev_mse, 1e-20):
            break
        prev_mse = cur_mse
    return np.sort(levels)


def quantize_with_codebook(w, codebook):
    """Quantize w using given codebook (nearest level). Returns q, indices.

    Codebook levels are rounded through float16 (the declared wire dtype)
    before use, simulating the serialization round-trip.
    """
    sorted_lv = round_to_float16(codebook)
    nl = len(sorted_lv)
    boundaries = 0.5 * (sorted_lv[:-1] + sorted_lv[1:])
    idx = np.clip(np.searchsorted(boundaries, w.flatten()), 0, nl - 1).astype(int)
    q = sorted_lv[idx].reshape(w.shape)
    return q, idx.reshape(w.shape)


# ============================================================================
# Normalized codebook design (for per-tile scale approach)
# ============================================================================

def design_normalized_codebook(W, k, method="kmeans"):
    """Design codebook in normalized [0,1] space.

    Pool all tile-normalized weights, then design codebook.
    This is the key: after normalization, all tiles share the same [0,1] range,
    so one codebook can serve all tiles.
    """
    M, N = W.shape
    ntr, ntc = M // TILE, N // TILE
    pooled = []
    for ti in range(ntr):
        for tj in range(ntc):
            tw = W[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE]
            lo, hi = float(tw.min()), float(tw.max())
            if hi - lo < 1e-15:
                continue
            pooled.append(((tw - lo) / (hi - lo)).flatten())
    pooled = np.concatenate(pooled)

    if method == "uniform":
        return design_global_uniform(pooled, k)
    elif method == "lloyd_max":
        return design_global_lloyd_max(pooled, k)
    elif method == "kmeans":
        return design_global_kmeans(pooled, k)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Shared codebook quantization (with and without per-tile scale)
# ============================================================================

def quantize_shared_with_scale(W, codebook, k, tile=TILE):
    """Shared codebook + per-tile min/max scale.

    Each tile normalized to [0,1] using its own min/max, then quantized
    using the shared codebook (designed in normalized space).

    Sidecar: 8 bytes/tile (min,max) + 2^k * 2 bytes (codebook, once).
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    Wq = np.zeros_like(W)
    all_indices = []

    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            tw = W[r0:r0 + tile, c0:c0 + tile]
            lo, hi = float(tw.min()), float(tw.max())
            if hi - lo < 1e-15:
                Wq[r0:r0 + tile, c0:c0 + tile] = lo
                all_indices.append(np.zeros(tw.size, dtype=int))
                continue
            w_norm = (tw - lo) / (hi - lo)
            q_norm, idx = quantize_with_codebook(w_norm, codebook)
            Wq[r0:r0 + tile, c0:c0 + tile] = q_norm * (hi - lo) + lo
            all_indices.append(idx.flatten())

    sidecar = N_TILES * 8 + (2 ** k) * 2
    return Wq, sidecar, np.concatenate(all_indices) if all_indices else np.array([], dtype=int)


def quantize_shared_no_scale(W, codebook, k, tile=TILE):
    """Shared codebook without per-tile scale.

    All tiles quantized directly using the shared codebook (designed from
    global distribution). No per-tile sidecar.

    Sidecar: 2^k * 2 bytes (codebook, once).
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    Wq = np.zeros_like(W)
    all_indices = []

    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            tw = W[r0:r0 + tile, c0:c0 + tile]
            q, idx = quantize_with_codebook(tw, codebook)
            Wq[r0:r0 + tile, c0:c0 + tile] = q
            all_indices.append(idx.flatten())

    total_sidecar = (2 ** k) * 2
    return Wq, total_sidecar, np.concatenate(all_indices) if all_indices else np.array([], dtype=int)


def quantize_per_tile_uniform(W, k, tile=TILE):
    """Per-tile uniform (baseline). Sidecar: 8 bytes/tile."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    Wq = np.zeros_like(W)
    all_indices = []
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            tw = W[r0:r0 + tile, c0:c0 + tile]
            q, _, idx = quantize_uniform_tile(tw, k)
            Wq[r0:r0 + tile, c0:c0 + tile] = q
            all_indices.append(idx.flatten())
    total_sidecar = N_TILES * 8
    return Wq, total_sidecar, np.concatenate(all_indices) if all_indices else np.array([], dtype=int)


# ============================================================================
# Simplified Viterbi (trellis path search)
# ============================================================================

def viterbi_quantize_tile(w_flat, codebook, hw_flat, alpha=0.01):
    """Viterbi DP for a single tile (flattened).

    State: codebook level index (0..2^K-1)
    Cost at step t, level j: (w[t] - codebook[j])^2 * hw[t]
    Transition cost j->k: alpha * (codebook[j] - codebook[k])^2
      (using codebook-level differences, not index differences, so alpha
      is comparable across K values)

    This models a simplified trellis constraint: adjacent elements in the
    sequence are penalized for large codebook-level jumps. The Viterbi
    path finds the optimal sequence of indices that minimizes total
    quantization error + transition penalty.

    Returns: quantized values, indices.
    """
    n = len(w_flat)
    nl = len(codebook)
    sorted_lv = np.sort(codebook)

    # Per-element quantization cost: (w[t] - level[j])^2 * hw[t]
    cost = (w_flat[:, None] - sorted_lv[None, :]) ** 2 * hw_flat[:, None]

    if alpha == 0.0 or n == 1:
        idx = np.argmin(cost, axis=1)
        return sorted_lv[idx], idx

    # Transition cost: alpha * (level[j] - level[k])^2
    # Using codebook-level differences makes alpha comparable across K
    trans = alpha * (sorted_lv[:, None] - sorted_lv[None, :]) ** 2

    # DP forward
    dp = cost[0].copy()  # (nl,)
    backptr = np.zeros((n, nl), dtype=np.int32)

    for t in range(1, n):
        combined = dp[:, None] + trans  # (nl, nl): [prev_k, cur_j]
        best_k = np.argmin(combined, axis=0)
        dp = cost[t] + combined[best_k, np.arange(nl)]
        backptr[t] = best_k

    # Backtrack
    idx = np.zeros(n, dtype=np.int32)
    idx[-1] = np.argmin(dp)
    for t in range(n - 2, -1, -1):
        idx[t] = backptr[t + 1, idx[t + 1]]

    return sorted_lv[idx], idx


def quantize_shared_viterbi(W, codebook, k, H_G, H_X, alpha=0.01,
                            per_tile_scale=True, tile=TILE):
    """Shared codebook + Viterbi DP with Hessian weighting."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    Wq = np.zeros_like(W)
    all_indices = []

    diag_G = np.diag(H_G)
    diag_X = np.diag(H_X)
    hw = diag_G[:, None] * diag_X[None, :]

    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            tw = W[r0:r0 + tile, c0:c0 + tile]
            hw_tile = hw[r0:r0 + tile, c0:c0 + tile]

            if per_tile_scale:
                lo, hi = float(tw.min()), float(tw.max())
                if hi - lo < 1e-15:
                    Wq[r0:r0 + tile, c0:c0 + tile] = lo
                    all_indices.append(np.zeros(tw.size, dtype=int))
                    continue
                w_norm = (tw - lo) / (hi - lo)
                # Physical error is (hi-lo)^2 * e_norm^2, so the Hessian-weighted
                # emission must include the (hi-lo)^2 factor for correct scaling
                hw_scaled = hw_tile.flatten() * (hi - lo) ** 2
                q_norm, idx = viterbi_quantize_tile(
                    w_norm.flatten(), codebook, hw_scaled, alpha)
                Wq[r0:r0 + tile, c0:c0 + tile] = q_norm.reshape(tile, tile) * (hi - lo) + lo
            else:
                q, idx = viterbi_quantize_tile(
                    tw.flatten(), codebook, hw_tile.flatten(), alpha)
                Wq[r0:r0 + tile, c0:c0 + tile] = q.reshape(tile, tile)
            all_indices.append(idx.flatten())

    if per_tile_scale:
        total_sidecar = N_TILES * 8 + (2 ** k) * 2
    else:
        total_sidecar = (2 ** k) * 2
    return Wq, total_sidecar, np.concatenate(all_indices) if all_indices else np.array([], dtype=int)


# ============================================================================
# Multi-level shared codebook + allocation
# ============================================================================

def design_multilevel_codebooks(all_weights, k_values, normalized=True):
    """Design shared codebooks at multiple K values.

    If normalized=True, design in [0,1] normalized space (for per-tile scale).
    If normalized=False, design on raw global weights (for no-scale approach).
    """
    codebooks = {}
    if normalized:
        # Pool normalized tile weights
        M, N = all_weights.shape
        ntr, ntc = M // TILE, N // TILE
        pooled = []
        for ti in range(ntr):
            for tj in range(ntc):
                tw = all_weights[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE]
                lo, hi = float(tw.min()), float(tw.max())
                if hi - lo < 1e-15:
                    continue
                pooled.append(((tw - lo) / (hi - lo)).flatten())
        pooled = np.concatenate(pooled)
    else:
        pooled = all_weights.flatten()

    for k in k_values:
        codebooks[k] = design_global_kmeans(pooled, k, seed=42 + k)
    return codebooks


def measure_tile_distortion_shared(W_tile, codebook, k, H_G_sub, H_X_sub,
                                    per_tile_scale=True):
    """Measure HWE of a tile quantized with shared codebook at level K."""
    if per_tile_scale:
        lo, hi = float(W_tile.min()), float(W_tile.max())
        if hi - lo < 1e-15:
            return 0.0
        w_norm = (W_tile - lo) / (hi - lo)
        q_norm, _ = quantize_with_codebook(w_norm, codebook)
        q = q_norm * (hi - lo) + lo
    else:
        q, _ = quantize_with_codebook(W_tile, codebook)
    E = W_tile - q
    return max(float(np.trace(H_G_sub @ E @ H_X_sub @ E.T)), 0.0)


def alloc_dp_multilevel(W, H_X, H_G, codebooks, k_values, budget_bytes,
                        per_tile_scale=True, tile=TILE):
    """DP allocation: assign each tile to a shared codebook (K level).

    Per-tile byte cost: K * 256/8 (payload) + [8 if per_tile_scale else 0]
    Codebook cost: sum(2^K_i * 2) for all K_i (once)
    Metadata: n_tiles * ceil(log2(len(k_values))) bits
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc

    D_table = np.zeros((n_tiles, len(k_values)))
    cost_table = np.zeros((n_tiles, len(k_values)), dtype=int)
    sidecar_per_tile = 8 if per_tile_scale else 0

    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0 + tile, c0:c0 + tile]
            H_G_sub = H_G[r0:r0 + tile, r0:r0 + tile]
            H_X_sub = H_X[c0:c0 + tile, c0:c0 + tile]
            for ki, k in enumerate(k_values):
                D_table[t_idx, ki] = measure_tile_distortion_shared(
                    W_tile, codebooks[k], k, H_G_sub, H_X_sub, per_tile_scale)
                cost_table[t_idx, ki] = k * EPS_PER_TILE // 8 + sidecar_per_tile
            t_idx += 1

    codebook_bytes = sum((2 ** k) * 2 for k in k_values)
    meta_bits = n_tiles * math.ceil(math.log2(len(k_values)))
    metadata = (meta_bits + 7) // 8

    effective_budget = budget_bytes - codebook_bytes - metadata
    if effective_budget <= 0:
        return None, D_table, -1

    INF = float('inf')
    max_bytes = effective_budget
    dp = [INF] * (max_bytes + 1)
    dp[0] = 0.0
    choices = []

    for t in range(n_tiles):
        new_dp = [INF] * (max_bytes + 1)
        new_choice = [-1] * (max_bytes + 1)
        for j in range(max_bytes + 1):
            if dp[j] == INF:
                continue
            for ki in range(len(k_values)):
                cost = int(cost_table[t, ki])
                nj = j + cost
                if nj > max_bytes:
                    continue
                val = dp[j] + D_table[t, ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append(new_choice[:])

    best_j = 0
    best_d = INF
    for j in range(max_bytes + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    K_flat = np.zeros(n_tiles, dtype=int)
    j = best_j
    for t in range(n_tiles - 1, -1, -1):
        ki = choices[t][j]
        if ki < 0:
            for fallback_ki in range(len(k_values)):
                if int(cost_table[t, fallback_ki]) <= j:
                    ki = fallback_ki
                    break
        K_flat[t] = k_values[ki]
        j -= int(cost_table[t, ki])

    total_bytes = int(np.sum([int(cost_table[t, list(k_values).index(K_flat[t])])
                              for t in range(n_tiles)])) + codebook_bytes + metadata
    return K_flat, D_table, total_bytes


def quantize_multilevel_shared(W, codebooks, K_grid, per_tile_scale=True, tile=TILE):
    """Quantize W using multi-level shared codebooks with per-tile K allocation."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    Wq = np.zeros_like(W)
    all_indices = []

    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            tw = W[r0:r0 + tile, c0:c0 + tile]
            k = int(K_grid[ti, tj])
            codebook = codebooks[k]

            if per_tile_scale:
                lo, hi = float(tw.min()), float(tw.max())
                if hi - lo < 1e-15:
                    Wq[r0:r0 + tile, c0:c0 + tile] = lo
                    all_indices.append(np.zeros(tw.size, dtype=int))
                    continue
                w_norm = (tw - lo) / (hi - lo)
                q_norm, idx = quantize_with_codebook(w_norm, codebook)
                Wq[r0:r0 + tile, c0:c0 + tile] = q_norm * (hi - lo) + lo
            else:
                q, idx = quantize_with_codebook(tw, codebook)
                Wq[r0:r0 + tile, c0:c0 + tile] = q
            all_indices.append(idx.flatten())

    sidecar_per_tile = 8 if per_tile_scale else 0
    codebook_bytes = sum((2 ** k) * 2 for k in codebooks)
    total_sidecar = N_TILES * sidecar_per_tile + codebook_bytes
    return Wq, total_sidecar, np.concatenate(all_indices) if all_indices else np.array([], dtype=int)


# ============================================================================
# Per-tile uniform DP allocation (baseline, from R16)
# ============================================================================

def measure_tile_distortion_uniform(W_tile, k, H_G_sub, H_X_sub):
    q, _, _ = quantize_uniform_tile(W_tile, k)
    E = W_tile - q
    return max(float(np.trace(H_G_sub @ E @ H_X_sub @ E.T)), 0.0)


def alloc_dp_per_tile_uniform(W, H_X, H_G, budget_bytes, tile=TILE):
    """DP allocation for per-tile uniform quantizer at byte budget."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc

    D_table = np.zeros((n_tiles, len(K_VALUES)))
    cost_table = np.zeros((n_tiles, len(K_VALUES)), dtype=int)

    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0 + tile, c0:c0 + tile]
            H_G_sub = H_G[r0:r0 + tile, r0:r0 + tile]
            H_X_sub = H_X[c0:c0 + tile, c0:c0 + tile]
            for ki, k in enumerate(K_VALUES):
                D_table[t_idx, ki] = measure_tile_distortion_uniform(W_tile, k, H_G_sub, H_X_sub)
                cost_table[t_idx, ki] = k * EPS_PER_TILE // 8 + 8
            t_idx += 1

    # K-map metadata: n_tiles * ceil(log2(len(K_VALUES))) bits
    # 64 tiles * 2 bits = 128 bits = 16 bytes
    meta_bits = n_tiles * math.ceil(math.log2(len(K_VALUES)))
    metadata = (meta_bits + 7) // 8
    effective_budget = budget_bytes - metadata
    if effective_budget <= 0:
        return None, D_table, -1

    INF = float('inf')
    max_bytes = effective_budget
    dp = [INF] * (max_bytes + 1)
    dp[0] = 0.0
    choices = []

    for t in range(n_tiles):
        new_dp = [INF] * (max_bytes + 1)
        new_choice = [-1] * (max_bytes + 1)
        for j in range(max_bytes + 1):
            if dp[j] == INF:
                continue
            for ki in range(len(K_VALUES)):
                cost = int(cost_table[t, ki])
                nj = j + cost
                if nj > max_bytes:
                    continue
                val = dp[j] + D_table[t, ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append(new_choice[:])

    best_j = 0
    best_d = INF
    for j in range(max_bytes + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    K_flat = np.zeros(n_tiles, dtype=int)
    j = best_j
    for t in range(n_tiles - 1, -1, -1):
        ki = choices[t][j]
        if ki < 0:
            for fallback_ki in range(len(K_VALUES)):
                if int(cost_table[t, fallback_ki]) <= j:
                    ki = fallback_ki
                    break
        K_flat[t] = K_VALUES[ki]
        j -= int(cost_table[t, ki])

    total_bytes = int(np.sum([int(cost_table[t, list(K_VALUES).index(K_flat[t])])
                              for t in range(n_tiles)])) + metadata
    return K_flat, D_table, total_bytes


# ============================================================================
# Byte accounting
# ============================================================================

def _k_map_metadata(n_tiles=N_TILES, n_k_values=4):
    """K-map metadata: n_tiles * ceil(log2(n_k_values)) bits -> bytes."""
    meta_bits = n_tiles * math.ceil(math.log2(max(n_k_values, 2)))
    return (meta_bits + 7) // 8


def bytes_per_tile_uniform(K, n_tiles=N_TILES, eps=EPS_PER_TILE):
    payload = n_tiles * (K * eps // 8)
    sidecar = n_tiles * 8
    metadata = _k_map_metadata(n_tiles, 4)  # 16 bytes for 4 K values
    return payload + sidecar + metadata


def bytes_shared_with_scale(K, n_tiles=N_TILES, eps=EPS_PER_TILE):
    payload = n_tiles * (K * eps // 8)
    sidecar = n_tiles * 8
    codebook = (2 ** K) * 2
    metadata = _k_map_metadata(n_tiles, 4)  # 16 bytes
    return payload + sidecar + codebook + metadata


def bytes_shared_no_scale(K, n_tiles=N_TILES, eps=EPS_PER_TILE):
    payload = n_tiles * (K * eps // 8)
    codebook = (2 ** K) * 2
    metadata = _k_map_metadata(n_tiles, 4)  # 16 bytes
    return payload + codebook + metadata


# ============================================================================
# Noise floor
# ============================================================================

def compute_noise_floor(W, H_G, H_X):
    """Noise floor: HWE from float64 -> float16 rounding."""
    W_f16 = W.astype(np.float16).astype(np.float64)
    E = W - W_f16
    return hessian_weighted_error(E, H_G, H_X)


# ============================================================================
# Experiments
# ============================================================================

def run_strategy_comparison(W, H_X, H_G, K):
    """Compare all codebook strategies at a single K on a single slice."""
    results = {}
    nf = compute_noise_floor(W, H_G, H_X)

    # Baseline: per-tile uniform
    Wq, sidecar, indices = quantize_per_tile_uniform(W, K)
    E = W - Wq
    results["per_tile_uniform"] = {
        "hwe": hessian_weighted_error(E, H_G, H_X),
        "mse": weight_mse(E),
        "bytes": bytes_per_tile_uniform(K),
        "entropy": index_entropy(indices),
    }

    # Shared codebook + per-tile scale (normalized codebook)
    for method in ["uniform", "lloyd_max", "kmeans"]:
        cb = design_normalized_codebook(W, K, method)
        Wq, sidecar, indices = quantize_shared_with_scale(W, cb, K)
        E = W - Wq
        results[f"shared_{method}_scale"] = {
            "hwe": hessian_weighted_error(E, H_G, H_X),
            "mse": weight_mse(E),
            "bytes": bytes_shared_with_scale(K),
            "entropy": index_entropy(indices),
        }

    # Shared codebook without per-tile scale (global codebook on raw weights)
    all_w = W.flatten()
    for method in ["uniform", "lloyd_max", "kmeans"]:
        if method == "uniform":
            cb = design_global_uniform(all_w, K)
        elif method == "lloyd_max":
            cb = design_global_lloyd_max(all_w, K)
        else:
            cb = design_global_kmeans(all_w, K)
        Wq, sidecar, indices = quantize_shared_no_scale(W, cb, K)
        E = W - Wq
        results[f"shared_{method}_noscale"] = {
            "hwe": hessian_weighted_error(E, H_G, H_X),
            "mse": weight_mse(E),
            "bytes": bytes_shared_no_scale(K),
            "entropy": index_entropy(indices),
        }

    results["_noise_floor"] = nf
    return results


def run_viterbi_comparison(W, H_X, H_G, K):
    """Compare independent vs Viterbi quantization at various alpha."""
    results = {}
    nf = compute_noise_floor(W, H_G, H_X)
    results["_noise_floor"] = nf

    cb = design_normalized_codebook(W, K, "kmeans")

    # Independent (alpha=0)
    Wq, sidecar, indices = quantize_shared_with_scale(W, cb, K)
    E = W - Wq
    results["independent"] = {
        "hwe": hessian_weighted_error(E, H_G, H_X),
        "mse": weight_mse(E),
        "bytes": bytes_shared_with_scale(K),
        "entropy": index_entropy(indices),
    }

    # Viterbi at various alpha
    for alpha in ALPHA_VALUES:
        if alpha == 0.0:
            continue
        Wq, sidecar, indices = quantize_shared_viterbi(
            W, cb, K, H_G, H_X, alpha=alpha, per_tile_scale=True)
        E = W - Wq
        results[f"viterbi_a{alpha}"] = {
            "hwe": hessian_weighted_error(E, H_G, H_X),
            "mse": weight_mse(E),
            "bytes": bytes_shared_with_scale(K),
            "entropy": index_entropy(indices),
        }

    # Viterbi without per-tile scale (global codebook)
    cb_global = design_global_kmeans(W.flatten(), K)
    for alpha in [1e-8, 1e-7, 1e-6, 1e-4, 1e-2]:
        Wq, sidecar, indices = quantize_shared_viterbi(
            W, cb_global, K, H_G, H_X, alpha=alpha, per_tile_scale=False)
        E = W - Wq
        results[f"viterbi_noscale_a{alpha}"] = {
            "hwe": hessian_weighted_error(E, H_G, H_X),
            "mse": weight_mse(E),
            "bytes": bytes_shared_no_scale(K),
            "entropy": index_entropy(indices),
        }

    return results


def run_matched_byte_dp(W, H_X, H_G, K):
    """Matched-byte DP: per-tile uniform DP vs multi-level shared DP."""
    results = {}
    nf = compute_noise_floor(W, H_G, H_X)
    results["_noise_floor"] = nf
    budget = bytes_per_tile_uniform(K)
    results["_budget"] = budget

    ntr, ntc = M_DIM // TILE, N_DIM // TILE

    # Per-tile uniform DP
    K_flat_u, _, bytes_u = alloc_dp_per_tile_uniform(W, H_X, H_G, budget)
    if K_flat_u is None:
        results["per_tile_uniform_dp"] = {"error": "budget insufficient"}
        return results
    K_grid_u = K_flat_u.reshape(ntr, ntc)
    Wq_u = np.zeros_like(W)
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * TILE, tj * TILE
            tw = W[r0:r0+TILE, c0:c0+TILE]
            q, _, _ = quantize_uniform_tile(tw, int(K_grid_u[ti, tj]))
            Wq_u[r0:r0+TILE, c0:c0+TILE] = q
    E_u = W - Wq_u
    base_hwe = hessian_weighted_error(E_u, H_G, H_X)
    results["per_tile_uniform_dp"] = {
        "hwe": base_hwe,
        "mse": weight_mse(E_u),
        "bytes": bytes_u,
        "K_dist": {str(k): int(np.sum(K_flat_u == k)) for k in K_VALUES},
    }

    # Multi-level shared + scale + DP
    codebooks = design_multilevel_codebooks(W, K_VALUES, normalized=True)
    K_flat_m, _, bytes_m = alloc_dp_multilevel(
        W, H_X, H_G, codebooks, K_VALUES, budget, per_tile_scale=True)
    if K_flat_m is not None:
        K_grid_m = K_flat_m.reshape(ntr, ntc)
        Wq_m, _, _ = quantize_multilevel_shared(W, codebooks, K_grid_m, per_tile_scale=True)
        E_m = W - Wq_m
        hwe_m = hessian_weighted_error(E_m, H_G, H_X)
        results["multilevel_shared_dp"] = {
            "hwe": hwe_m,
            "mse": weight_mse(E_m),
            "bytes": bytes_m,
            "K_dist": {str(k): int(np.sum(K_flat_m == k)) for k in K_VALUES},
            "ratio_vs_uniform": hwe_m / max(base_hwe, 1e-20),
        }
    else:
        results["multilevel_shared_dp"] = {"error": "budget insufficient"}

    # Multi-level shared no-scale + DP
    codebooks_ns = design_multilevel_codebooks(W, K_VALUES, normalized=False)
    K_flat_n, _, bytes_n = alloc_dp_multilevel(
        W, H_X, H_G, codebooks_ns, K_VALUES, budget, per_tile_scale=False)
    if K_flat_n is not None:
        K_grid_n = K_flat_n.reshape(ntr, ntc)
        Wq_n, _, _ = quantize_multilevel_shared(W, codebooks_ns, K_grid_n, per_tile_scale=False)
        E_n = W - Wq_n
        hwe_n = hessian_weighted_error(E_n, H_G, H_X)
        results["multilevel_shared_noscale_dp"] = {
            "hwe": hwe_n,
            "mse": weight_mse(E_n),
            "bytes": bytes_n,
            "K_dist": {str(k): int(np.sum(K_flat_n == k)) for k in K_VALUES},
            "ratio_vs_uniform": hwe_n / max(base_hwe, 1e-20),
        }
    else:
        results["multilevel_shared_noscale_dp"] = {"error": "budget insufficient"}

    return results


def run_rotation_comparison(W, H_X, H_G, K):
    """Compare shared codebook with and without BiIP+Hadamard rotation.

    NOTE: EXL3 already has Hadamard+signs+LDLQ incoherence processing.
    Our BiIP+Hadamard approximates this. The question for codebooks is:
    does rotation (which homogenizes tile distributions) make a shared
    codebook more effective? If yes, shared codebook is viable post-rotation.
    """
    results = {}

    rng = np.random.default_rng(42)
    W_t, H_X_t, H_G_t, U, V, S_G, S_X, rot_sidecar = apply_rotation(W, H_X, H_G, rng)

    nf_unrot = compute_noise_floor(W, H_G, H_X)
    nf_rot = compute_noise_floor(W_t, H_G_t, H_X_t)

    # --- Unrotated ---
    Wq, _, _ = quantize_per_tile_uniform(W, K)
    E = W - Wq
    results["unrot_per_tile_uniform"] = {
        "hwe": hessian_weighted_error(E, H_G, H_X),
        "bytes": bytes_per_tile_uniform(K),
    }

    cb = design_normalized_codebook(W, K, "kmeans")
    Wq, _, _ = quantize_shared_with_scale(W, cb, K)
    E = W - Wq
    results["unrot_shared_kmeans_scale"] = {
        "hwe": hessian_weighted_error(E, H_G, H_X),
        "bytes": bytes_shared_with_scale(K),
    }

    cb_g = design_global_kmeans(W.flatten(), K)
    Wq, _, _ = quantize_shared_no_scale(W, cb_g, K)
    E = W - Wq
    results["unrot_shared_kmeans_noscale"] = {
        "hwe": hessian_weighted_error(E, H_G, H_X),
        "bytes": bytes_shared_no_scale(K),
    }

    # --- Rotated ---
    Wq, _, _ = quantize_per_tile_uniform(W_t, K)
    E = W_t - Wq
    results["rot_per_tile_uniform"] = {
        "hwe": hessian_weighted_error(E, H_G_t, H_X_t),
        "bytes": bytes_per_tile_uniform(K) + rot_sidecar,
    }

    cb = design_normalized_codebook(W_t, K, "kmeans")
    Wq, _, _ = quantize_shared_with_scale(W_t, cb, K)
    E = W_t - Wq
    results["rot_shared_kmeans_scale"] = {
        "hwe": hessian_weighted_error(E, H_G_t, H_X_t),
        "bytes": bytes_shared_with_scale(K) + rot_sidecar,
    }

    # KEY TEST: shared no-scale after rotation — rotation homogenizes tiles,
    # so a single global codebook should fit all tiles without per-tile scale
    cb_g = design_global_kmeans(W_t.flatten(), K)
    Wq, _, _ = quantize_shared_no_scale(W_t, cb_g, K)
    E = W_t - Wq
    results["rot_shared_kmeans_noscale"] = {
        "hwe": hessian_weighted_error(E, H_G_t, H_X_t),
        "bytes": bytes_shared_no_scale(K) + rot_sidecar,
    }

    # Tile homogeneity metrics: coefficient of variation of tile ranges
    def tile_range_cv(Wmat):
        ranges = []
        for ti in range(M_DIM // TILE):
            for tj in range(N_DIM // TILE):
                tw = Wmat[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE]
                lo, hi = float(tw.min()), float(tw.max())
                ranges.append(hi - lo)
        ranges = np.array(ranges)
        return float(np.std(ranges) / max(np.mean(ranges), 1e-15))

    results["_tile_range_cv_unrot"] = tile_range_cv(W)
    results["_tile_range_cv_rot"] = tile_range_cv(W_t)

    results["_noise_floor_unrot"] = nf_unrot
    results["_noise_floor_rot"] = nf_rot
    results["_rot_sidecar"] = rot_sidecar
    return results


def run_rotation_matched_byte_dp(W, H_X, H_G, K):
    """Matched-byte DP with rotation: does shared codebook + rotation beat
    per-tile uniform + rotation at matched bytes?

    This is the most important test: if rotation homogenizes tiles, then
    a shared codebook (O(1) storage) should match per-tile uniform at the
    same byte budget, because the per-tile min/max sidecar is no longer needed.
    """
    results = {}
    nf = compute_noise_floor(W, H_G, H_X)
    results["_noise_floor"] = nf

    rng = np.random.default_rng(42)
    W_t, H_X_t, H_G_t, U, V, S_G, S_X, rot_sidecar = apply_rotation(W, H_X, H_G, rng)

    budget = bytes_per_tile_uniform(K) + rot_sidecar
    results["_budget"] = budget
    ntr, ntc = M_DIM // TILE, N_DIM // TILE

    # Per-tile uniform DP (rotated)
    K_flat_u, _, bytes_u = alloc_dp_per_tile_uniform(W_t, H_X_t, H_G_t,
                                                      budget - rot_sidecar)
    if K_flat_u is not None:
        K_grid_u = K_flat_u.reshape(ntr, ntc)
        Wq_u = np.zeros_like(W_t)
        for ti in range(ntr):
            for tj in range(ntc):
                r0, c0 = ti * TILE, tj * TILE
                tw = W_t[r0:r0+TILE, c0:c0+TILE]
                q, _, _ = quantize_uniform_tile(tw, int(K_grid_u[ti, tj]))
                Wq_u[r0:r0+TILE, c0:c0+TILE] = q
        E_u = W_t - Wq_u
        base_hwe = hessian_weighted_error(E_u, H_G_t, H_X_t)
        results["rot_per_tile_uniform_dp"] = {
            "hwe": base_hwe,
            "bytes": bytes_u + rot_sidecar,
            "K_dist": {str(k): int(np.sum(K_flat_u == k)) for k in K_VALUES},
        }
    else:
        results["rot_per_tile_uniform_dp"] = {"error": "budget insufficient"}
        return results

    # Multi-level shared + scale + DP (rotated) — same quant budget as uniform
    quant_budget = budget - rot_sidecar
    codebooks = design_multilevel_codebooks(W_t, K_VALUES, normalized=True)
    K_flat_m, _, bytes_m = alloc_dp_multilevel(
        W_t, H_X_t, H_G_t, codebooks, K_VALUES, quant_budget, per_tile_scale=True)
    if K_flat_m is not None:
        K_grid_m = K_flat_m.reshape(ntr, ntc)
        Wq_m, _, _ = quantize_multilevel_shared(W_t, codebooks, K_grid_m, per_tile_scale=True)
        E_m = W_t - Wq_m
        hwe_m = hessian_weighted_error(E_m, H_G_t, H_X_t)
        results["rot_multilevel_shared_dp"] = {
            "hwe": hwe_m,
            "bytes": bytes_m + rot_sidecar,
            "K_dist": {str(k): int(np.sum(K_flat_m == k)) for k in K_VALUES},
            "ratio_vs_uniform": hwe_m / max(base_hwe, 1e-20),
        }
    else:
        results["rot_multilevel_shared_dp"] = {"error": "budget insufficient"}

    # Multi-level shared no-scale + DP (rotated) — KEY: no per-tile sidecar
    codebooks_ns = design_multilevel_codebooks(W_t, K_VALUES, normalized=False)
    K_flat_n, _, bytes_n = alloc_dp_multilevel(
        W_t, H_X_t, H_G_t, codebooks_ns, K_VALUES, quant_budget, per_tile_scale=False)
    if K_flat_n is not None:
        K_grid_n = K_flat_n.reshape(ntr, ntc)
        Wq_n, _, _ = quantize_multilevel_shared(W_t, codebooks_ns, K_grid_n, per_tile_scale=False)
        E_n = W_t - Wq_n
        hwe_n = hessian_weighted_error(E_n, H_G_t, H_X_t)
        results["rot_multilevel_shared_noscale_dp"] = {
            "hwe": hwe_n,
            "bytes": bytes_n + rot_sidecar,
            "K_dist": {str(k): int(np.sum(K_flat_n == k)) for k in K_VALUES},
            "ratio_vs_uniform": hwe_n / max(base_hwe, 1e-20),
        }
    else:
        results["rot_multilevel_shared_noscale_dp"] = {"error": "budget insufficient"}

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("R21: Simplified Trellis / Viterbi Codebook Simulation")
    print("=" * 80)

    tensors = load_real_weights()
    print(f"Loaded {len(tensors)} tensors: {list(tensors.keys())}")

    all_results = {
        "config": {
            "tile_size": TILE, "m_dim": M_DIM, "n_dim": N_DIM,
            "k_values": K_VALUES, "n_tiles": N_TILES,
            "elements_per_tile": EPS_PER_TILE,
            "tensor_names": TENSOR_NAMES,
            "alpha_values": ALPHA_VALUES,
        },
        "experiments": {},
    }

    for tname in TENSOR_NAMES:
        if tname not in tensors:
            continue
        tensor = tensors[tname]
        slices = extract_slices(tensor)
        all_results["experiments"][tname] = {}

        for sname, W in slices.items():
            print(f"\n--- {tname}/{sname} ({W.shape}) ---")
            H_X, H_G = synthetic_hessians(W, seed=42)
            slice_results = {}

            # Experiment 1: Strategy comparison at each K
            print("  [1] Strategy comparison...")
            strat_results = {}
            for K in K_VALUES:
                t0 = time.time()
                res = run_strategy_comparison(W, H_X, H_G, K)
                strat_results[str(K)] = res
                base_hwe = res["per_tile_uniform"]["hwe"]
                for sname2, vals in res.items():
                    if sname2.startswith("_"):
                        continue
                    ratio = vals["hwe"] / max(base_hwe, 1e-20)
                    print(f"    K={K} {sname2:30s} HWE={vals['hwe']:.6e} "
                          f"ratio={ratio:.4f} bytes={vals['bytes']} "
                          f"entropy={vals['entropy']:.3f}")
                print(f"    ({time.time()-t0:.1f}s)")
            slice_results["strategy_comparison"] = strat_results

            # Experiment 2: Viterbi comparison
            print("  [2] Viterbi comparison...")
            vit_results = {}
            for K in [4, 5, 6]:
                t0 = time.time()
                res = run_viterbi_comparison(W, H_X, H_G, K)
                vit_results[str(K)] = res
                ind_hwe = res["independent"]["hwe"]
                for sname2, vals in res.items():
                    if sname2.startswith("_"):
                        continue
                    if isinstance(vals, dict) and "hwe" in vals:
                        ratio = vals["hwe"] / max(ind_hwe, 1e-20)
                        print(f"    K={K} {sname2:30s} HWE={vals['hwe']:.6e} "
                              f"ratio={ratio:.4f} entropy={vals['entropy']:.3f}")
                print(f"    ({time.time()-t0:.1f}s)")
            slice_results["viterbi_comparison"] = vit_results

            # Experiment 3: Matched-byte DP (unrotated)
            print("  [3] Matched-byte DP (unrotated)...")
            dp_results = {}
            for K in [4, 5, 6]:
                t0 = time.time()
                res = run_matched_byte_dp(W, H_X, H_G, K)
                dp_results[str(K)] = res
                if "per_tile_uniform_dp" in res and "hwe" in res["per_tile_uniform_dp"]:
                    base_hwe = res["per_tile_uniform_dp"]["hwe"]
                    print(f"    K={K} per_tile_uniform_dp       HWE={base_hwe:.6e} "
                          f"bytes={res['per_tile_uniform_dp']['bytes']}")
                    for sname2 in ["multilevel_shared_dp", "multilevel_shared_noscale_dp"]:
                        if sname2 in res and "hwe" in res[sname2]:
                            r = res[sname2]
                            print(f"    K={K} {sname2:30s} HWE={r['hwe']:.6e} "
                                  f"ratio={r.get('ratio_vs_uniform', 0):.4f} "
                                  f"bytes={r['bytes']}")
                print(f"    ({time.time()-t0:.1f}s)")
            slice_results["matched_byte_dp"] = dp_results

            # Experiment 4: Rotation comparison (fixed K)
            print("  [4] Rotation comparison...")
            rot_results = {}
            for K in [4, 5, 6]:
                t0 = time.time()
                res = run_rotation_comparison(W, H_X, H_G, K)
                rot_results[str(K)] = res
                for sname2, vals in res.items():
                    if sname2.startswith("_"):
                        continue
                    if isinstance(vals, dict) and "hwe" in vals:
                        print(f"    K={K} {sname2:35s} HWE={vals['hwe']:.6e} "
                              f"bytes={vals['bytes']}")
                print(f"    ({time.time()-t0:.1f}s)")
            slice_results["rotation_comparison"] = rot_results

            # Experiment 5: Rotation + matched-byte DP
            print("  [5] Rotation + matched-byte DP...")
            rotdp_results = {}
            for K in [4, 5, 6]:
                t0 = time.time()
                res = run_rotation_matched_byte_dp(W, H_X, H_G, K)
                rotdp_results[str(K)] = res
                if "rot_per_tile_uniform_dp" in res and "hwe" in res["rot_per_tile_uniform_dp"]:
                    base_hwe = res["rot_per_tile_uniform_dp"]["hwe"]
                    print(f"    K={K} rot_per_tile_uniform_dp      HWE={base_hwe:.6e} "
                          f"bytes={res['rot_per_tile_uniform_dp']['bytes']}")
                    for sname2 in ["rot_multilevel_shared_dp", "rot_multilevel_shared_noscale_dp"]:
                        if sname2 in res and "hwe" in res[sname2]:
                            r = res[sname2]
                            print(f"    K={K} {sname2:30s} HWE={r['hwe']:.6e} "
                                  f"ratio={r.get('ratio_vs_uniform', 0):.4f} "
                                  f"bytes={r['bytes']}")
                print(f"    ({time.time()-t0:.1f}s)")
            slice_results["rotation_matched_byte_dp"] = rotdp_results

            all_results["experiments"][tname][sname] = slice_results

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
