#!/usr/bin/env python3
"""
R16-NonUniformCodebook: Non-uniform and learned codebooks for tile quantization.

All prior experiments (R1-R15) use uniform min/max quantization. EXL3 actually
uses trellis codebooks (Viterbi search over non-uniform levels). This PoC tests
whether non-uniform codebooks significantly improve quantization quality.

Quantizer strategies (all per-tile, 16×16 tiles):
  1. Uniform: min-max uniform quantization (baseline from R1/R3)
  2. Lloyd-Max: iterative optimal scalar quantizer (uniform init)
  3. K-means: k-means++ init + Lloyd iterations
  4. Hessian-weighted Lloyd-Max: centroids weighted by H_G[i,i]*H_X[j,j]
  5. Distribution-optimal: fit Laplacian, compute optimal quantizer

Experiments:
  A. Quantizer comparison at K=3,4,5,6 (matched levels, then matched bytes)
  B. With/without BiIP+Hadamard rotation
  C. Rate-distortion curves
  D. DP allocation with non-uniform quantizers
  E. Codebook sidecar accounting

Key hypothesis: non-uniform helps more on unrotated (heavy-tailed) weights,
gap narrows after rotation (Gaussian → uniform-optimal).
"""

import json
import numpy as np
import os
import sys
import time
import warnings
from scipy.stats import laplace, norm

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
ELEMENTS_PER_TILE = TILE * TILE  # 256
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r16-nonuniform-codebook/receipts/research/r16-nonuniform-codebook-results.json"

TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]

# ============================================================================
# Weight loading and slicing
# ============================================================================

def load_real_weights():
    """Load real Qwen3.8-27B BF16 weights (correctly decoded)."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in data.files:
        tensors[key] = data[key].astype(np.float64)
    return tensors


def extract_slices(tensor, m=128, n=128, seed=42):
    """Extract 3 128×128 slices: first (top-left), mid (center), random."""
    M, N = tensor.shape
    slices = []
    slices.append(("first", tensor[:m, :n].copy()))
    r0, c0 = M // 2 - m // 2, N // 2 - n // 2
    slices.append(("mid", tensor[r0:r0 + m, c0:c0 + n].copy()))
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, max(1, M - m))
    c0 = rng.integers(0, max(1, N - n))
    slices.append(("rand", tensor[r0:r0 + m, c0:c0 + n].copy()))
    return slices


# ============================================================================
# Synthetic Hessian generation (from R3)
# ============================================================================

def synthetic_hessians(W, n_samples=512, outlier_fraction=0.05, outlier_scale=10.0, seed=42):
    """Generate synthetic activation Hessian H_X and output Hessian proxy H_G."""
    rng = np.random.default_rng(seed)
    d_out, d_in = W.shape

    X = rng.standard_normal((d_in, n_samples))
    n_outliers = max(1, int(d_in * outlier_fraction))
    outlier_channels = rng.choice(d_in, n_outliers, replace=False)
    X[outlier_channels, :] *= outlier_scale

    H_X = (X @ X.T / n_samples).astype(np.float64)
    Y = W @ X
    H_G = (Y @ Y.T / n_samples).astype(np.float64)

    H_X *= d_in / np.trace(H_X)
    H_G *= d_out / np.trace(H_G)
    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)

    return H_X, H_G


# ============================================================================
# Quantizers
# ============================================================================

def quantize_uniform_tile(w, k):
    """Per-tile uniform min/max quantizer. k bits → 2^k levels."""
    if k <= 0:
        return np.zeros_like(w), 8.0  # 2 floats for min,max
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo), 8.0
    step = (hi - lo) / (nl - 1)
    q = np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo
    # Sidecar: 2 float32 (min, max) = 8 bytes
    return q, 8.0


def quantize_lloyd_max_tile(w, k, max_iter=100, tol=1e-10):
    """Lloyd-Max optimal scalar quantizer for a tile.

    Iteratively:
      1. Assign each element to nearest level
      2. Update levels to centroids of assigned elements
    Converges to local optimum (uniform initialization).

    Returns: quantized tile, sidecar_bytes, n_iters
    """
    if k <= 0:
        return np.zeros_like(w), 2.0, 0
    nl = 2 ** k
    w_flat = w.flatten()
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo), float(nl * 2), 0

    # Initialize levels uniformly
    levels = np.linspace(lo, hi, nl)
    prev_mse = float('inf')

    for it in range(max_iter):
        # Assign to nearest level
        # For efficiency, use searchsorted on sorted levels
        sorted_levels = np.sort(levels)
        # Boundaries are midpoints between consecutive levels
        boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
        # Assign each element
        idx = np.searchsorted(boundaries, w_flat)
        # Update levels to centroids
        new_levels = np.zeros(nl)
        for i in range(nl):
            mask = (idx == i)
            if np.any(mask):
                new_levels[i] = np.mean(w_flat[mask])
            else:
                new_levels[i] = sorted_levels[i]  # keep old level if empty

        # Check convergence
        cur_mse = np.mean((w_flat - new_levels[idx]) ** 2)
        levels = new_levels
        if abs(prev_mse - cur_mse) < tol * max(prev_mse, 1e-20):
            it += 1
            break
        prev_mse = cur_mse

    n_iters = it + 1
    # Final assignment
    sorted_levels = np.sort(levels)
    boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
    idx = np.searchsorted(boundaries, w_flat)
    q_flat = sorted_levels[np.clip(idx, 0, nl - 1)]
    q = q_flat.reshape(w.shape)

    # Sidecar: nl centroids as float16 = nl * 2 bytes
    sidecar = float(nl * 2)
    return q, sidecar, n_iters


def quantize_kmeans_tile(w, k, max_iter=100, tol=1e-10, seed=42):
    """K-means quantizer for a tile.

    Uses k-means++ initialization, then Lloyd iterations.
    Same optimization as Lloyd-Max but different initialization.

    Returns: quantized tile, sidecar_bytes, n_iters
    """
    if k <= 0:
        return np.zeros_like(w), 2.0, 0
    nl = 2 ** k
    w_flat = w.flatten()
    n = len(w_flat)
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo), float(nl * 2), 0

    rng = np.random.default_rng(seed + hash(w_flat[0]) % 2**31)

    # k-means++ initialization
    # First center: random choice
    centers = [w_flat[rng.integers(n)]]
    for _ in range(1, nl):
        # Compute squared distances to nearest center
        dists = np.min([(w_flat - c) ** 2 for c in centers], axis=0)
        total = np.sum(dists)
        if total < 1e-15:
            # All points identical, pick random
            centers.append(w_flat[rng.integers(n)])
        else:
            probs = dists / total
            idx = rng.choice(n, p=probs)
            centers.append(w_flat[idx])

    levels = np.array(centers, dtype=np.float64)
    prev_mse = float('inf')

    for it in range(max_iter):
        sorted_levels = np.sort(levels)
        boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
        idx = np.searchsorted(boundaries, w_flat)

        new_levels = np.zeros(nl)
        for i in range(nl):
            mask = (idx == i)
            if np.any(mask):
                new_levels[i] = np.mean(w_flat[mask])
            else:
                new_levels[i] = sorted_levels[i]

        cur_mse = np.mean((w_flat - new_levels[np.clip(idx, 0, nl - 1)]) ** 2)
        levels = new_levels
        if abs(prev_mse - cur_mse) < tol * max(prev_mse, 1e-20):
            it += 1
            break
        prev_mse = cur_mse

    n_iters = it + 1
    sorted_levels = np.sort(levels)
    boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
    idx = np.searchsorted(boundaries, w_flat)
    q_flat = sorted_levels[np.clip(idx, 0, nl - 1)]
    q = q_flat.reshape(w.shape)

    sidecar = float(nl * 2)
    return q, sidecar, n_iters


def quantize_hessian_weighted_lloyd_tile(w, k, hw, max_iter=100, tol=1e-10):
    """Hessian-weighted Lloyd-Max quantizer.

    Minimizes Σ h_ij * (w_ij - q(w_ij))² instead of Σ (w_ij - q(w_ij))².
    Centroids are weighted means: l_i = Σ_{j in cluster i} h_j * w_j / Σ_{j in cluster i} h_j
    Boundaries are weighted midpoints: b_i = (l_i * Σ_{left} + l_{i+1} * Σ_{right}) / ...

    Actually, for weighted Lloyd, the optimal boundary between two levels
    with weights is the weighted midpoint:
      b = (l_i + l_{i+1}) / 2  (same as unweighted! boundaries don't change)
    But centroids ARE weighted means.

    Returns: quantized tile, sidecar_bytes, n_iters
    """
    if k <= 0:
        return np.zeros_like(w), 2.0, 0
    nl = 2 ** k
    w_flat = w.flatten()
    hw_flat = hw.flatten()
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo), float(nl * 2), 0

    # Initialize levels uniformly (same as Lloyd-Max)
    levels = np.linspace(lo, hi, nl)
    prev_obj = float('inf')

    for it in range(max_iter):
        sorted_levels = np.sort(levels)
        boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
        idx = np.searchsorted(boundaries, w_flat)

        new_levels = np.zeros(nl)
        for i in range(nl):
            mask = (idx == i)
            if np.any(mask):
                w_cluster = w_flat[mask]
                h_cluster = hw_flat[mask]
                h_sum = np.sum(h_cluster)
                if h_sum > 1e-15:
                    new_levels[i] = np.sum(w_cluster * h_cluster) / h_sum
                else:
                    new_levels[i] = sorted_levels[i]
            else:
                new_levels[i] = sorted_levels[i]

        # Weighted MSE
        cur_obj = np.sum(hw_flat * (w_flat - new_levels[np.clip(idx, 0, nl - 1)]) ** 2)
        levels = new_levels
        if abs(prev_obj - cur_obj) < tol * max(abs(prev_obj), 1e-20):
            it += 1
            break
        prev_obj = cur_obj

    n_iters = it + 1
    sorted_levels = np.sort(levels)
    boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
    idx = np.searchsorted(boundaries, w_flat)
    q_flat = sorted_levels[np.clip(idx, 0, nl - 1)]
    q = q_flat.reshape(w.shape)

    sidecar = float(nl * 2)
    return q, sidecar, n_iters


def quantize_dist_optimal_tile(w, k, max_iter=50, tol=1e-10):
    """Distribution-optimal quantizer: fit Laplacian, compute optimal levels.

    Neural network weights are well-modeled by Laplacian distribution.
    We fit the Laplacian parameter, then run Lloyd-Max initialized from
    the analytic Laplacian-optimal levels (non-uniform spacing).

    For Laplacian f(x) = (1/2b) * exp(-|x|/b), the optimal quantizer has
    non-uniform levels that are denser near zero.

    Returns: quantized tile, sidecar_bytes, n_iters
    """
    if k <= 0:
        return np.zeros_like(w), 2.0, 0
    nl = 2 ** k
    w_flat = w.flatten()
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo), float(nl * 2), 0

    # Fit Laplacian: MLE of b is mean(|x - median(x)|)
    median = np.median(w_flat)
    b_hat = np.mean(np.abs(w_flat - median))
    if b_hat < 1e-15:
        b_hat = np.std(w_flat) + 1e-12

    # Initialize levels using Laplacian CDF quantiles (non-uniform spacing)
    # This gives levels denser near the median (where Laplacian peaks)
    probs = np.linspace(0, 1, nl + 1)[1:-1]  # interior quantile points
    # Laplacian inverse CDF
    levels = np.zeros(nl)
    for i in range(nl):
        if i == 0:
            levels[0] = lo
        elif i == nl - 1:
            levels[-1] = hi
        else:
            u = i / (nl - 1)
            # Laplacian quantile function
            if u < 0.5:
                levels[i] = median + b_hat * np.log(2 * u)
            else:
                levels[i] = median - b_hat * np.log(2 * (1 - u))

    # Run Lloyd-Max from this initialization (refine for actual distribution)
    prev_mse = float('inf')
    for it in range(max_iter):
        sorted_levels = np.sort(levels)
        boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
        idx = np.searchsorted(boundaries, w_flat)

        new_levels = np.zeros(nl)
        for i in range(nl):
            mask = (idx == i)
            if np.any(mask):
                new_levels[i] = np.mean(w_flat[mask])
            else:
                new_levels[i] = sorted_levels[i]

        cur_mse = np.mean((w_flat - new_levels[np.clip(idx, 0, nl - 1)]) ** 2)
        levels = new_levels
        if abs(prev_mse - cur_mse) < tol * max(prev_mse, 1e-20):
            it += 1
            break
        prev_mse = cur_mse

    n_iters = it + 1
    sorted_levels = np.sort(levels)
    boundaries = 0.5 * (sorted_levels[:-1] + sorted_levels[1:])
    idx = np.searchsorted(boundaries, w_flat)
    q_flat = sorted_levels[np.clip(idx, 0, nl - 1)]
    q = q_flat.reshape(w.shape)

    sidecar = float(nl * 2)
    return q, sidecar, n_iters


# ============================================================================
# Matrix-level quantization
# ============================================================================

QUANTIZER_NAMES = ["uniform", "lloyd_max", "kmeans", "hw_lloyd", "dist_optimal"]

def quantize_matrix(W, K_grid, quantizer, H_X=None, H_G=None, tile=TILE):
    """Quantize W using per-tile quantizer with given K_grid.

    quantizer: one of "uniform", "lloyd_max", "kmeans", "hw_lloyd", "dist_optimal"
    K_grid: (n_tiles_row, n_tiles_col) array of K values

    Returns: Wq, total_sidecar_bytes, convergence_info
    """
    M, N = W.shape
    Wq = np.zeros_like(W)
    ntr, ntc = M // tile, N // tile
    total_sidecar = 0.0
    total_iters = 0
    n_tiles_quantized = 0

    # Precompute per-element Hessian weights if needed
    hw = None
    if quantizer == "hw_lloyd":
        if H_X is None or H_G is None:
            raise ValueError("hw_lloyd requires H_X and H_G")
        diag_G = np.diag(H_G)  # (M,)
        diag_X = np.diag(H_X)  # (N,)
        hw = diag_G[:, None] * diag_X[None, :]  # (M, N)

    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            k = int(K_grid[ti, tj])
            tile_w = W[r0:r0+tile, c0:c0+tile]

            if quantizer == "uniform":
                q, sc = quantize_uniform_tile(tile_w, k)
                iters = 0
            elif quantizer == "lloyd_max":
                q, sc, iters = quantize_lloyd_max_tile(tile_w, k)
            elif quantizer == "kmeans":
                q, sc, iters = quantize_kmeans_tile(tile_w, k)
            elif quantizer == "hw_lloyd":
                hw_tile = hw[r0:r0+tile, c0:c0+tile]
                q, sc, iters = quantize_hessian_weighted_lloyd_tile(tile_w, k, hw_tile)
            elif quantizer == "dist_optimal":
                q, sc, iters = quantize_dist_optimal_tile(tile_w, k)
            else:
                raise ValueError(f"Unknown quantizer: {quantizer}")

            Wq[r0:r0+tile, c0:c0+tile] = q
            total_sidecar += sc
            total_iters += iters
            n_tiles_quantized += 1

    avg_iters = total_iters / max(n_tiles_quantized, 1)
    return Wq, total_sidecar, avg_iters


# ============================================================================
# Byte budget accounting
# ============================================================================

def compute_bytes_uniform(K_flat, n_tiles=N_TILES, eps=ELEMENTS_PER_TILE):
    """Bytes for uniform quantizer: payload + 2 float32 per tile + K metadata."""
    total_k = int(np.sum(K_flat))
    payload = total_k * eps // 8
    sidecar = n_tiles * 8  # 2 float32 (min, max)
    if len(set(K_flat.tolist())) == 1:
        metadata = 1
    else:
        metadata = (n_tiles * 3 + 7) // 8
    return payload + sidecar + metadata


def compute_bytes_nonuniform(K_flat, n_tiles=N_TILES, eps=ELEMENTS_PER_TILE):
    """Bytes for non-uniform quantizer: payload + 2^K float16 centroids per tile.

    Sidecar per tile: 2^K * 2 bytes (centroids as float16).
    This grows exponentially with K, which is the key overhead.
    """
    total_payload = 0
    total_sidecar = 0
    for k in K_flat:
        nl = 2 ** int(k)
        total_payload += int(k) * eps // 8
        total_sidecar += nl * 2  # float16 centroids

    if len(set(K_flat.tolist())) == 1:
        metadata = 1
    else:
        metadata = (n_tiles * 3 + 7) // 8
    return total_payload + total_sidecar + metadata


def budget_k_sum_for_avg_k(avg_k, n_tiles=N_TILES, eps=ELEMENTS_PER_TILE,
                           quantizer="uniform"):
    """Compute integer K-sum budget for mixed-K allocation.

    For the allocation experiment, we match the PAYLOAD bit budget (sum of K_t),
    which is the same regardless of quantizer type. Codebook sidecar differences
    are reported separately. This ensures a fair comparison of allocation
    effectiveness across quantizer types.
    """
    return avg_k * n_tiles


# ============================================================================
# BiIP + Hadamard rotation (from R3)
# ============================================================================

def hadamard_matrix(n):
    """Sylvester-type Hadamard matrix of size n (must be power of 2)."""
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    """Signed randomized Hadamard: H @ diag(±1)."""
    H = hadamard_matrix(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    return H @ np.diag(signs), signs


def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11)."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_G = np.diag(sg_diag)
    W_transformed = S_G @ W @ S_X
    sidecar_bytes = (d_in + d_out) * 4
    return S_G, S_X, W_transformed, sidecar_bytes


def apply_rotation(W, H_X, H_G, rng):
    """Apply BiIP + signed Hadamard on both sides.

    Returns: W_t, H_X_t, H_G_t, U, V, S_G, S_X, sidecar
    """
    d_out, d_in = W.shape
    sidecar = 0

    S_G, S_X, W_t, sc = biip_scaling(W, H_X, H_G)
    sidecar += sc
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = S_X_inv @ H_X @ S_X_inv
    H_G_t = S_G_inv @ H_G @ S_G_inv

    V, signs_v = signed_random_hadamard(d_in, rng)
    sidecar += d_in // 8 + 1
    W_t = W_t @ V.T
    H_X_t = V @ H_X_t @ V.T

    U, signs_u = signed_random_hadamard(d_out, rng)
    sidecar += d_out // 8 + 1
    W_t = U @ W_t
    H_G_t = U @ H_G_t @ U.T

    return W_t, H_X_t, H_G_t, U, V, S_G, S_X, sidecar


def inverse_rotation(W_q, U, V, S_G, S_X):
    """Inverse the rotation: W_hat = S_G^{-1} @ U^T @ W_q @ V @ S_X^{-1}."""
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ W_q @ V @ S_X_inv


# ============================================================================
# Evaluation metrics
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """Primary metric: tr(H_G · E · H_X · E^T)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    """Raw weight MSE."""
    return float(np.mean(E ** 2))


# ============================================================================
# DP allocation (adapted from R1 for non-uniform quantizers)
# ============================================================================

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub, quantizer):
    """Measure actual Hessian-weighted distortion of a tile at K=k."""
    if quantizer == "uniform":
        q, _ = quantize_uniform_tile(W_tile, k)
    elif quantizer == "lloyd_max":
        q, _, _ = quantize_lloyd_max_tile(W_tile, k)
    elif quantizer == "kmeans":
        q, _, _ = quantize_kmeans_tile(W_tile, k)
    elif quantizer == "hw_lloyd":
        hw = np.diag(H_G_sub)[:, None] * np.diag(H_X_sub)[None, :]
        q, _, _ = quantize_hessian_weighted_lloyd_tile(W_tile, k, hw)
    elif quantizer == "dist_optimal":
        q, _, _ = quantize_dist_optimal_tile(W_tile, k)
    else:
        raise ValueError(f"Unknown quantizer: {quantizer}")
    E = W_tile - q
    return max(float(np.trace(H_G_sub @ E @ H_X_sub @ E.T)), 0.0)


def alloc_dp(W, H_X, H_G, avg_k, quantizer, tile=TILE):
    """DP tile-local allocation: exact knapsack on additive surrogate.

    Uses the given quantizer to measure per-tile distortion at each K.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc
    budget_k = budget_k_sum_for_avg_k(avg_k, n_tiles, ELEMENTS_PER_TILE, quantizer)

    D_table = np.zeros((n_tiles, len(K_VALUES)))
    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0+tile, c0:c0+tile]
            H_G_sub = H_G[r0:r0+tile, r0:r0+tile]
            H_X_sub = H_X[c0:c0+tile, c0:c0+tile]
            for ki, k in enumerate(K_VALUES):
                D_table[t_idx, ki] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub, quantizer)
            t_idx += 1

    # Multiple-choice knapsack DP
    INF = float('inf')
    dp = [INF] * (budget_k + 1)
    dp[0] = 0.0
    choices = []

    for t in range(n_tiles):
        new_dp = [INF] * (budget_k + 1)
        new_choice = [-1] * (budget_k + 1)
        for j in range(budget_k + 1):
            if dp[j] == INF:
                continue
            for ki in range(len(K_VALUES)):
                k_val = K_VALUES[ki]
                nj = j + k_val
                if nj > budget_k:
                    continue
                val = dp[j] + D_table[t, ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append(new_choice[:])

    best_j = 0
    best_d = INF
    for j in range(budget_k + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    K_flat = np.zeros(n_tiles, dtype=int)
    j = best_j
    for t in range(n_tiles - 1, -1, -1):
        ki = choices[t][j]
        if ki < 0:
            for fallback_ki in range(len(K_VALUES)):
                if K_VALUES[fallback_ki] <= j:
                    ki = fallback_ki
                    break
        K_flat[t] = K_VALUES[ki]
        j -= K_VALUES[ki]

    assert int(np.sum(K_flat)) <= budget_k
    return K_flat, D_table


# ============================================================================
# Byte-budget DP: exact matched-byte comparison
# ============================================================================

def tile_byte_cost(k, quantizer, eps=ELEMENTS_PER_TILE):
    """Per-tile byte cost for a given K and quantizer type.
    Uniform: 32*K + 8 (payload + 2 float32 min/max)
    Non-uniform: 32*K + 2^(K+1) (payload + 2^K float16 centroids)
    """
    payload = k * eps // 8  # 32*K for 256-element tiles
    if quantizer == "uniform":
        sidecar = 8  # 2 float32
    else:
        sidecar = (2 ** k) * 2  # 2^K float16 centroids
    return payload + sidecar


def alloc_dp_byte_budget(W, H_X, H_G, budget_bytes, quantizer, tile=TILE):
    """Byte-budget DP: minimize total HWE subject to total bytes <= budget.

    Per-tile costs: tile_byte_cost(k, quantizer) for each K.
    This allows fair comparison of uniform vs non-uniform at identical byte budgets.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc

    # Precompute distortion and byte cost tables
    D_table = np.zeros((n_tiles, len(K_VALUES)))
    cost_table = np.zeros((n_tiles, len(K_VALUES)), dtype=int)
    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0+tile, c0:c0+tile]
            H_G_sub = H_G[r0:r0+tile, r0:r0+tile]
            H_X_sub = H_X[c0:c0+tile, c0:c0+tile]
            for ki, k in enumerate(K_VALUES):
                D_table[t_idx, ki] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub, quantizer)
                cost_table[t_idx, ki] = tile_byte_cost(k, quantizer)
            t_idx += 1
    # Check budget sufficiency: minimum cost = all tiles at K_MIN
    min_cost_per_tile = tile_byte_cost(K_MIN, quantizer)
    min_total = min_cost_per_tile * n_tiles + (n_tiles * 3 + 7) // 8  # mixed metadata
    if min_total > budget_bytes:
        # Budget too small for this quantizer at any valid K allocation
        return None, D_table, -1

    metadata = 1  # uniform K
    # Check if all-same-K is possible (cost < budget)
    for k_idx, k in enumerate(K_VALUES):
        if cost_table[0, k_idx] * n_tiles + metadata <= budget_bytes:
            break
    else:
        # Budget too small for any uniform K, use mixed
        metadata = (n_tiles * 3 + 7) // 8

    effective_budget = budget_bytes - metadata

    # DP in byte units
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

    # Find min distortion
    best_j = 0
    best_d = INF
    for j in range(max_bytes + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    # Backtrack
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

    total_bytes = int(np.sum([tile_byte_cost(int(k), quantizer) for k in K_flat])) + metadata
    return K_flat, D_table, total_bytes


def run_matched_byte_experiment(W, H_X, H_G, K, tile=TILE):
    """Compare uniform vs non-uniform at IDENTICAL total byte budget.

    For each quantizer, compute the byte-budget DP at the uniform-K budget.
    Then compare: does any non-uniform quantizer with DP allocation beat
    uniform with DP allocation at the same total bytes?
    """
    ntr, ntc = M_DIM // tile, N_DIM // tile
    n_tiles = ntr * ntc

    # Uniform K budget (with metadata)
    K_flat_u = np.full(n_tiles, K)
    budget_uniform = compute_bytes_uniform(K_flat_u)

    results = {}

    # Uniform DP at uniform-K budget
    K_flat_u_dp, _, bytes_u_dp = alloc_dp_byte_budget(W, H_X, H_G, budget_uniform, "uniform", tile)
    if K_flat_u_dp is None:
        return {"error": "budget insufficient for uniform"}
    K_grid_u_dp = K_flat_u_dp.reshape(ntr, ntc)
    Wq_u, _, _ = quantize_matrix(W, K_grid_u_dp, "uniform")
    E_u = W - Wq_u
    hwe_u = hessian_weighted_error(E_u, H_G, H_X)
    results["uniform_dp"] = {
        "hwe": hwe_u,
        "bytes": bytes_u_dp,
        "budget": budget_uniform,
        "K_dist": {str(k): int(np.sum(K_flat_u_dp == k)) for k in K_VALUES},
    }

    # Non-uniform quantizers at the SAME budget
    for qname in ["lloyd_max", "kmeans", "hw_lloyd"]:
        K_flat_n, _, bytes_n = alloc_dp_byte_budget(W, H_X, H_G, budget_uniform, qname, tile)
        if K_flat_n is None:
            results[qname + "_dp"] = {"error": "budget insufficient", "budget": budget_uniform}
            continue
        K_grid_n = K_flat_n.reshape(ntr, ntc)
        if qname == "hw_lloyd":
            Wq_n, _, _ = quantize_matrix(W, K_grid_n, qname, H_X, H_G)
        else:
            Wq_n, _, _ = quantize_matrix(W, K_grid_n, qname)
        E_n = W - Wq_n
        hwe_n = hessian_weighted_error(E_n, H_G, H_X)
        results[qname + "_dp"] = {
            "hwe": hwe_n,
            "bytes": bytes_n,
            "budget": budget_uniform,
            "K_dist": {str(k): int(np.sum(K_flat_n == k)) for k in K_VALUES},
            "ratio_vs_uniform": hwe_n / max(hwe_u, 1e-20),
        }

    return results


# ============================================================================
# Distribution analysis
# ============================================================================

def analyze_tile_distribution(w):
    """Analyze the statistical properties of a tile's weight distribution."""
    w_flat = w.flatten()
    return {
        "mean": float(np.mean(w_flat)),
        "std": float(np.std(w_flat)),
        "kurtosis": float(
            np.mean((w_flat - np.mean(w_flat)) ** 4) /
            (np.std(w_flat) ** 4 + 1e-20) - 3.0
        ),
        "laplace_b": float(np.mean(np.abs(w_flat - np.median(w_flat)))),
        "range": float(w_flat.max() - w_flat.min()),
        "n_unique": int(len(np.unique(w_flat))),
    }


# ============================================================================
# Experiments
# ============================================================================

def run_quantizer_comparison(W, H_X, H_G, K, rotated=False, U=None, V=None,
                             S_G=None, S_X=None, rot_sidecar=0):
    """Compare all quantizers at a single K value on a single slice.

    Returns dict of results.
    """
    results = {}
    K_grid = np.full((M_DIM // TILE, N_DIM // TILE), K)

    for qname in QUANTIZER_NAMES:
        # Quantize in the (possibly rotated) space
        if qname == "hw_lloyd":
            Wq, sidecar, avg_iters = quantize_matrix(W, K_grid, qname, H_X, H_G)
        else:
            Wq, sidecar, avg_iters = quantize_matrix(W, K_grid, qname)

        # If rotated, inverse to original space
        if rotated:
            Wq_orig = inverse_rotation(Wq, U, V, S_G, S_X)
        else:
            Wq_orig = Wq

        # Compute error in ORIGINAL space with ORIGINAL Hessians
        # (rotation invariance: tr(H_G E H_X E^T) is preserved)
        # But we need to use original H_G, H_X for the metric
        if rotated:
            # We need original Hessians — use the unrotated ones
            # Actually, for the HWE metric we should compute in original space
            # The rotation invariance says: tr(H_G E H_X E^T) = tr(H_G' E' H_X' E'^T)
            # where E' is error in rotated space. So we can compute in rotated space.
            E_rot = W - Wq  # error in rotated space
            # But we need rotated Hessians for the metric
            # Actually, apply_rotation already transformed H_X, H_G
            # So W, H_X, H_G here are already the rotated versions
            # We compute HWE in rotated space (equivalent by invariance)
            E = E_rot
            hwe = hessian_weighted_error(E, H_G, H_X)
            mse = weight_mse(E)
        else:
            E = W - Wq_orig
            hwe = hessian_weighted_error(E, H_G, H_X)
            mse = weight_mse(E)

        # Byte accounting
        K_flat = K_grid.flatten()
        if qname == "uniform":
            total_bytes = compute_bytes_uniform(K_flat)
        else:
            total_bytes = compute_bytes_nonuniform(K_flat)

        total_bytes_with_rot = total_bytes + rot_sidecar

        results[qname] = {
            "mse": mse,
            "hwe": hwe,
            "sidecar_bytes": sidecar,
            "total_bytes": total_bytes,
            "total_bytes_with_rot": total_bytes_with_rot,
            "avg_iters": avg_iters,
            "K": K,
        }

    return results


def run_rate_distortion(W, H_X, H_G, rotated=False, U=None, V=None,
                        S_G=None, S_X=None, rot_sidecar=0):
    """Run rate-distortion sweep across K=3,4,5,6 for all quantizers."""
    rd_curves = {}
    for K in K_VALUES:
        res = run_quantizer_comparison(W, H_X, H_G, K, rotated, U, V, S_G, S_X, rot_sidecar)
        rd_curves[K] = res
    return rd_curves


def run_allocation_experiment(W, H_X, H_G, K, quantizer, tile=TILE):
    """Run DP allocation with the given quantizer, compare to uniform-K."""
    ntr, ntc = M_DIM // tile, N_DIM // tile
    n_tiles = ntr * ntc

    # Uniform K baseline
    K_grid_uniform = np.full((ntr, ntc), K)
    Wq_u, sc_u, it_u = quantize_matrix(W, K_grid_uniform, quantizer,
                                       H_X if quantizer == "hw_lloyd" else None,
                                       H_G if quantizer == "hw_lloyd" else None)
    E_u = W - Wq_u
    hwe_u = hessian_weighted_error(E_u, H_G, H_X)
    mse_u = weight_mse(E_u)

    K_flat_u = K_grid_uniform.flatten()
    if quantizer == "uniform":
        bytes_u = compute_bytes_uniform(K_flat_u)
    else:
        bytes_u = compute_bytes_nonuniform(K_flat_u)

    # DP allocation
    K_flat_dp, _ = alloc_dp(W, H_X, H_G, K, quantizer, tile)
    K_grid_dp = K_flat_dp.reshape(ntr, ntc)
    Wq_d, sc_d, it_d = quantize_matrix(W, K_grid_dp, quantizer,
                                       H_X if quantizer == "hw_lloyd" else None,
                                       H_G if quantizer == "hw_lloyd" else None)
    E_d = W - Wq_d
    hwe_d = hessian_weighted_error(E_d, H_G, H_X)
    mse_d = weight_mse(E_d)

    if quantizer == "uniform":
        bytes_d = compute_bytes_uniform(K_flat_dp)
    else:
        bytes_d = compute_bytes_nonuniform(K_flat_dp)

    improvement = (hwe_u - hwe_d) / max(hwe_u, 1e-20) * 100.0

    return {
        "quantizer": quantizer,
        "K": K,
        "uniform_hwe": hwe_u,
        "uniform_mse": mse_u,
        "uniform_bytes": bytes_u,
        "dp_hwe": hwe_d,
        "dp_mse": mse_d,
        "dp_bytes": bytes_d,
        "dp_improvement_pct": improvement,
        "dp_K_dist": {str(k): int(np.sum(K_flat_dp == k)) for k in K_VALUES},
        "budget_k_sum": int(np.sum(K_flat_dp)),
    }


def run_convergence_analysis(W, K=5, tile=TILE):
    """Track Lloyd-Max convergence per tile."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    iters_lloyd = []
    iters_kmeans = []
    iters_dist = []

    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            w_tile = W[r0:r0+tile, c0:c0+tile]
            _, _, it_l = quantize_lloyd_max_tile(w_tile, K)
            _, _, it_k = quantize_kmeans_tile(w_tile, K)
            _, _, it_d = quantize_dist_optimal_tile(w_tile, K)
            iters_lloyd.append(it_l)
            iters_kmeans.append(it_k)
            iters_dist.append(it_d)

    return {
        "lloyd_max_iters": {
            "mean": float(np.mean(iters_lloyd)),
            "std": float(np.std(iters_lloyd)),
            "min": int(np.min(iters_lloyd)),
            "max": int(np.max(iters_lloyd)),
            "median": float(np.median(iters_lloyd)),
        },
        "kmeans_iters": {
            "mean": float(np.mean(iters_kmeans)),
            "std": float(np.std(iters_kmeans)),
            "min": int(np.min(iters_kmeans)),
            "max": int(np.max(iters_kmeans)),
            "median": float(np.median(iters_kmeans)),
        },
        "dist_optimal_iters": {
            "mean": float(np.mean(iters_dist)),
            "std": float(np.std(iters_dist)),
            "min": int(np.min(iters_dist)),
            "max": int(np.max(iters_dist)),
            "median": float(np.median(iters_dist)),
        },
    }


def run_kurtosis_analysis(W, tile=TILE):
    """Analyze tile kurtosis to understand distribution shapes."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    kurtoses = []
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            w_tile = W[r0:r0+tile, c0:c0+tile]
            stats = analyze_tile_distribution(w_tile)
            kurtoses.append(stats["kurtosis"])
    return {
        "mean": float(np.mean(kurtoses)),
        "std": float(np.std(kurtoses)),
        "min": float(np.min(kurtoses)),
        "max": float(np.max(kurtoses)),
        "median": float(np.median(kurtoses)),
        "n_tiles": len(kurtoses),
    }


# ============================================================================
# Main experiment
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("R16-NonUniformCodebook: Non-uniform and learned codebooks for tile quantization")
    print("=" * 80)

    # Load weights
    print("\nLoading real weights...")
    tensors = load_real_weights()
    print(f"  Loaded {len(tensors)} tensors")

    all_results = {
        "config": {
            "tile_size": TILE,
            "m_dim": M_DIM,
            "n_dim": N_DIM,
            "k_values": K_VALUES,
            "n_tiles": N_TILES,
            "elements_per_tile": ELEMENTS_PER_TILE,
            "tensor_names": TENSOR_NAMES,
            "quantizers": QUANTIZER_NAMES,
        },
        "experiments": {},
    }

    rng_seed = 42

    for tname in TENSOR_NAMES:
        if tname not in tensors:
            print(f"  Skipping {tname} (not found)")
            continue
        print(f"\n{'=' * 60}")
        print(f"Tensor: {tname} {tensors[tname].shape}")
        print(f"{'=' * 60}")

        tensor = tensors[tname]
        slices = extract_slices(tensor, M_DIM, N_DIM, seed=rng_seed)
        all_results["experiments"][tname] = {}

        for sname, W_orig in slices:
            print(f"\n  Slice: {sname} ({W_orig.shape})")
            # Generate Hessians from this slice
            H_X_orig, H_G_orig = synthetic_hessians(W_orig, seed=rng_seed)

            slice_data = {}

            # --- Kurtosis analysis ---
            kurt = run_kurtosis_analysis(W_orig)
            slice_data["kurtosis_unrotated"] = kurt
            print(f"    Kurtosis: mean={kurt['mean']:.2f}, median={kurt['median']:.2f}, "
                  f"max={kurt['max']:.2f}")

            # --- Unrotated rate-distortion ---
            print(f"    Running unrotated quantizer comparison...")
            rd_unrot = run_rate_distortion(W_orig, H_X_orig, H_G_orig, rotated=False)
            slice_data["rd_unrotated"] = rd_unrot

            for K in K_VALUES:
                res = rd_unrot[K]
                print(f"      K={K}: ", end="")
                for qn in QUANTIZER_NAMES:
                    print(f"{qn}:HWE={res[qn]['hwe']:.6e} ", end="")
                print()

            # --- Convergence analysis ---
            print(f"    Running convergence analysis...")
            conv = run_convergence_analysis(W_orig, K=5)
            slice_data["convergence"] = conv
            print(f"      Lloyd-Max: mean={conv['lloyd_max_iters']['mean']:.1f} iters, "
                  f"max={conv['lloyd_max_iters']['max']}")
            print(f"      K-means:    mean={conv['kmeans_iters']['mean']:.1f} iters, "
                  f"max={conv['kmeans_iters']['max']}")
            print(f"      Dist-opt:   mean={conv['dist_optimal_iters']['mean']:.1f} iters, "
                  f"max={conv['dist_optimal_iters']['max']}")

            # --- Rotated rate-distortion ---
            print(f"    Applying BiIP + Hadamard rotation...")
            rng = np.random.default_rng(rng_seed)
            W_rot, H_X_rot, H_G_rot, U, V, S_G, S_X, rot_sidecar = \
                apply_rotation(W_orig, H_X_orig, H_G_orig, rng)

            kurt_rot = run_kurtosis_analysis(W_rot)
            slice_data["kurtosis_rotated"] = kurt_rot
            print(f"    Rotated kurtosis: mean={kurt_rot['mean']:.2f}, "
                  f"median={kurt_rot['median']:.2f}, max={kurt_rot['max']:.2f}")

            print(f"    Running rotated quantizer comparison...")
            rd_rot = run_rate_distortion(W_rot, H_X_rot, H_G_rot, rotated=True,
                                          U=U, V=V, S_G=S_G, S_X=S_X,
                                          rot_sidecar=rot_sidecar)
            slice_data["rd_rotated"] = rd_rot

            for K in K_VALUES:
                res = rd_rot[K]
                print(f"      K={K}: ", end="")
                for qn in QUANTIZER_NAMES:
                    print(f"{qn}:HWE={res[qn]['hwe']:.6e} ", end="")
                print()

            # --- Allocation experiments ---
            print(f"    Running DP allocation experiments...")
            alloc_results = {}
            for qname in ["uniform", "lloyd_max", "dist_optimal"]:
                for K in K_VALUES:
                    key = f"{qname}_K{K}"
                    alloc = run_allocation_experiment(
                        W_orig, H_X_orig, H_G_orig, K, qname
                    )
                    alloc_results[key] = alloc
                    print(f"      {key}: DP improvement = {alloc['dp_improvement_pct']:.2f}%")
            slice_data["allocation"] = alloc_results

            # --- Non-uniform + rotation allocation ---
            print(f"    Running non-uniform + rotation allocation...")
            alloc_rot_results = {}
            for qname in ["uniform", "lloyd_max", "dist_optimal"]:
                for K in K_VALUES:
                    key = f"{qname}_K{K}"
                    alloc = run_allocation_experiment(
                        W_rot, H_X_rot, H_G_rot, K, qname
                    )
                    alloc_rot_results[key] = alloc
            slice_data["allocation_rotated"] = alloc_rot_results

            # --- Matched byte budget experiments ---
            print(f"    Running matched-byte-budget DP experiments...")
            matched_results = {}
            for K in K_VALUES:
                matched = run_matched_byte_experiment(W_orig, H_X_orig, H_G_orig, K)
                matched_results[f"K{K}"] = matched
                if "error" in matched:
                    print(f"      K={K}: budget insufficient, skipped")
                    continue
                u_hwe = matched["uniform_dp"]["hwe"]
                for qn in ["lloyd_max", "kmeans", "hw_lloyd"]:
                    qkey = qn + "_dp"
                    if qkey not in matched or "error" in matched.get(qkey, {}):
                        print(f"      K={K} {qn}: budget insufficient, skipped")
                        continue
                    n_hwe = matched[qkey]["hwe"]
                    ratio = n_hwe / max(u_hwe, 1e-20)
                    print(f"      K={K} {qn}_dp vs uniform_dp: ratio={ratio:.3f} "
                          f"({'non-uniform WINS' if ratio < 1.0 else 'uniform wins'})")
            slice_data["matched_byte"] = matched_results

            # --- Matched byte budget + rotation ---
            print(f"    Running matched-byte-budget DP (rotated)...")
            matched_rot_results = {}
            for K in K_VALUES:
                matched = run_matched_byte_experiment(W_rot, H_X_rot, H_G_rot, K)
                matched_rot_results[f"K{K}"] = matched
            slice_data["matched_byte_rotated"] = matched_rot_results


            all_results["experiments"][tname][sname] = slice_data

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    # Compute aggregate improvement of each quantizer over uniform
    print("\nAggregate improvement over uniform (HWE %):")
    for qname in ["lloyd_max", "kmeans", "hw_lloyd", "dist_optimal"]:
        for rot_label, rd_key in [("unrot", "rd_unrotated"), ("rot", "rd_rotated")]:
            improvements = []
            for tname in TENSOR_NAMES:
                if tname not in all_results["experiments"]:
                    continue
                for sname in all_results["experiments"][tname]:
                    sd = all_results["experiments"][tname][sname]
                    for K in K_VALUES:
                        u_hwe = sd[rd_key][K]["uniform"]["hwe"]
                        q_hwe = sd[rd_key][K][qname]["hwe"]
                        if u_hwe > 1e-20:
                            imp = (u_hwe - q_hwe) / u_hwe * 100.0
                            improvements.append(imp)
            if improvements:
                print(f"  {qname} ({rot_label}): mean={np.mean(improvements):.2f}%, "
                      f"median={np.median(improvements):.2f}%, "
                      f"min={np.min(improvements):.2f}%, max={np.max(improvements):.2f}%")

    # Codebook overhead analysis
    print("\nCodebook sidecar overhead (non-uniform vs uniform):")
    for K in K_VALUES:
        K_flat = np.full(N_TILES, K)
        b_uniform = compute_bytes_uniform(K_flat)
        b_nonuniform = compute_bytes_nonuniform(K_flat)
        overhead_pct = (b_nonuniform - b_uniform) / b_uniform * 100
        print(f"  K={K}: uniform={b_uniform} bytes, non-uniform={b_nonuniform} bytes, "
              f"overhead={overhead_pct:.1f}%")

    # Kurtosis change with rotation
    print("\nKurtosis change with rotation (macro mean):")
    for tname in TENSOR_NAMES:
        if tname not in all_results["experiments"]:
            continue
        kurt_unrot = []
        kurt_rot = []
        for sname in all_results["experiments"][tname]:
            sd = all_results["experiments"][tname][sname]
            kurt_unrot.append(sd["kurtosis_unrotated"]["mean"])
            kurt_rot.append(sd["kurtosis_rotated"]["mean"])
        print(f"  {tname}: unrot={np.mean(kurt_unrot):.2f} → rot={np.mean(kurt_rot):.2f}")

    # Matched-byte-budget summary
    print("\nMatched-byte-budget DP: non-uniform vs uniform at identical bytes:")
    for rot_label, mb_key in [("unrot", "matched_byte"), ("rot", "matched_byte_rotated")]:
        print(f"\n  {rot_label}:")
        for qname in ["lloyd_max", "kmeans", "hw_lloyd"]:
            for K in K_VALUES:
                ratios = []
                wins = 0
                total = 0
                for tname in TENSOR_NAMES:
                    if tname not in all_results["experiments"]:
                        continue
                    for sname in all_results["experiments"][tname]:
                        sd = all_results["experiments"][tname][sname]
                        if mb_key in sd and f"K{K}" in sd[mb_key]:
                            mb = sd[mb_key][f"K{K}"]
                            if "error" in mb or "uniform_dp" not in mb:
                                continue
                            u_hwe = mb["uniform_dp"]["hwe"]
                            qkey = qname + "_dp"
                            if qkey not in mb or "error" in mb.get(qkey, {}):
                                continue
                            n_hwe = mb[qkey]["hwe"]
                            ratio = n_hwe / max(u_hwe, 1e-20)
                            ratios.append(ratio)
                            if ratio < 1.0:
                                wins += 1
                            total += 1
                if ratios:
                    print(f"    {qname} K={K}: ratio mean={np.mean(ratios):.3f}, "
                          f"median={np.median(ratios):.3f}, "
                          f"win_rate={wins}/{total}, min={np.min(ratios):.3f}")


    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
