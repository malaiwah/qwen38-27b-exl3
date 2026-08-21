#!/usr/bin/env python3
"""
R14-NoiseShapeStack: Noise shaping within the full quantization stack.

Tests whether act-order GPTQ and noise shaping compose with rotation and scaling.
Wave 1 R7 showed act-order GPTQ gives +21-24% over RTN (standalone). This experiment
tests whether that benefit survives when composed with:

  1. BiIP + Hadamard rotation (R3: 57-82% OC-proxy error reduction)
  2. BiIP scaling alone (R3: dominant rotation factor)
  3. DP tile allocation (R1: +25.5% over uniform K)
  4. GPTAQ P-matrix correction (R2: α=1.0 paper-faithful)

Key questions:
  - Does act-order become a no-op after rotation uniformizes diag(H_X)?
  - Or does act-order still shape error even after rotation makes weights incoherent?
  - What is the interaction between noise shaping (error direction) and rotation (error magnitude)?
  - Can noise shaping and rotation be complementary?

Correct Cholesky: U = chol(inv(H+λI)).T, upper triangular, U^T U = H^{-1}.
Matched per-tile (16×16) quantizer for ALL arms.
4 real tensors, 3 slices, K=3,4,5,6.
"""

import numpy as np
import json
import warnings
import os
import time
from pathlib import Path

warnings.filterwarnings('ignore')

WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
OUTPUT_DIR = Path("/Users/mbelleau/Projects/qwen38-research-r14-noise-shape-stack/receipts/research")

K_VALUES = [3, 4, 5, 6]
TILE_SIZE = 16
SEEDS = [42, 123, 777]
N_SAMPLES_CALIB = 512
TENSOR_NAMES = ['L0_gate', 'L0_down', 'L55_gate', 'L55_down']
SLICE_SIZE = 128
SLICES = [(0, 0), (0, 128), (0, 256)]  # (row_offset, col_offset)

# ============================================================================
# Utilities
# ============================================================================

def load_real_weights():
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in TENSOR_NAMES:
        w = data[key].astype(np.float64)
        tensors[key] = w
    return tensors

def get_slice(W, slice_idx):
    r, c = SLICES[slice_idx]
    return W[r:r+SLICE_SIZE, c:c+SLICE_SIZE].copy()

def generate_calibration(n_in, n_samples, rng):
    """Returns X of shape (n_in, n_samples) — matching R9 convention."""
    cov_base = np.eye(n_in) * 0.5
    for i in range(n_in - 1):
        cov_base[i, i+1] = 0.1
        cov_base[i+1, i] = 0.1
    X = rng.multivariate_normal(np.zeros(n_in), cov_base, size=n_samples).T  # (n_in, n_samples)
    n_outliers = max(1, n_in // 20)
    outlier_idx = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_idx, :] *= 5.0
    return X  # (n_in, n_samples)

def compute_hessians(W, X):
    """X is (n_in, n_samples). Returns H_X (n_in×n_in), H_G (n_out×n_out)."""
    N = X.shape[1]
    H_X = np.nan_to_num(X @ X.T / N, nan=0.0, posinf=1e6, neginf=-1e6)
    Y = W @ X  # (n_out, n_samples)
    H_G = np.nan_to_num(Y @ Y.T / N, nan=0.0, posinf=1e6, neginf=-1e6)
    return H_X, H_G

def safe_eigh(H, epsilon=1e-10):
    H = np.nan_to_num((H + H.T) / 2, nan=0.0, posinf=1e6, neginf=-1e6)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvals, eigvecs

# ============================================================================
# Per-tile (16×16) uniform quantizer — MATCHED across all arms
# ============================================================================

def quantize_tiles(W, bits, tile=TILE_SIZE):
    """Per-tile uniform quantization. 
    bits: scalar (all tiles same K) or 2D array (n_tm, n_tn) for per-tile K."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    is_alloc = isinstance(bits, np.ndarray) and bits.ndim == 2
    for ti, i in enumerate(range(0, m, tile)):
        for tj, j in enumerate(range(0, n, tile)):
            block = W[i:i+tile, j:j+tile]
            k = int(bits[ti, tj]) if is_alloc else int(bits)
            nl = 2 ** k
            lo = float(block.min())
            hi = float(block.max())
            if hi - lo < 1e-15:
                Wq[i:i+tile, j:j+tile] = block
            else:
                step = (hi - lo) / (nl - 1)
                Wq[i:i+tile, j:j+tile] = np.clip(
                    np.round((block - lo) / step), 0, nl - 1) * step + lo
    return Wq

# ============================================================================
# Metrics
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """tr(H_G · E · H_X · E^T) where E = W - Wq."""
    val = np.trace(H_G @ E @ H_X @ E.T)
    return float(val) if np.isfinite(val) else 1e20

def raw_mse(E):
    return float(np.mean(E ** 2))

def spectral_analysis(E, H_X, H_G):
    """Project error onto H_X eigenvectors, compute H_G-weighted energy per direction."""
    eig_H_X, V_X = safe_eigh(H_X)
    V_X_desc = V_X[:, ::-1]
    eig_H_X_desc = eig_H_X[::-1]
    E_proj = E @ V_X_desc
    g_j = np.array([(E_proj[:, j] @ H_G @ E_proj[:, j]) for j in range(E_proj.shape[1])])
    return {'eig_H_X': eig_H_X_desc, 'error_energy_HG_weighted': g_j}

def anti_correlation(eig_H_X, error_energy):
    """Correlation between Hessian eigenvalues and error energy.
    Negative = anti-correlated (good: error pushed away from high-Hessian directions)."""
    h = eig_H_X / (eig_H_X.sum() + 1e-15)
    e = error_energy / (error_energy.sum() + 1e-15)
    if np.std(h) < 1e-15 or np.std(e) < 1e-15:
        return 0.0
    return float(np.corrcoef(h, e)[0, 1])

def diag_cv(H):
    """Coefficient of variation of diagonal entries — measures how uniform diag(H) is."""
    d = np.diag(H)
    return float(np.std(d) / (np.mean(np.abs(d)) + 1e-15))

# ============================================================================
# Cholesky (CORRECT convention from Wave 1 bug fix)
# ============================================================================

def inv_cholesky(H, damping=0.01):
    """Upper triangular U such that U^T @ U = inv(H + λI).
    Correct GPTQ convention: U[c, c:] is non-zero for row propagation.
    Construction: Hinv = inv(H+damp), U = cholesky(Hinv).T (upper)."""
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
    return U

# ============================================================================
# GPTAQ correction with per-tile quantizer and configurable column ordering
# ============================================================================

def gptaq_correction(W, X, Xt, bits, tile=TILE_SIZE, damping=0.01, alpha=0.0,
                     order=None, bits_per_tile=None):
    """GPTAQ correction with per-tile (16×16) quantizer, MATCHED with RTN/rotate arms.

    Processes column-tiles (blocks of `tile` columns). For each column-tile:
    1. Freeze one codebook per row-tile using current Ww (before any GPTQ update)
    2. Quantize all `tile` columns using frozen codebooks
    3. Apply GPTQ error propagation to remaining columns + GPTAQ P-matrix

    If order is provided, columns are processed in that order (act-order).
    Returns Q in the ORIGINAL column order.

    W: (m, n). X: (n_in, n_samples). Xt: (n_in, n_samples).
    """
    m, n = W.shape

    if order is not None:
        W_work = W[:, order].copy()
        X_work = X[order, :].copy()
        Xt_work = Xt[order, :].copy()
    else:
        W_work = W.copy()
        X_work = X.copy()
        Xt_work = Xt.copy()

    Ww = W_work.copy().astype(np.float64)
    W0 = W_work.copy()
    Q = np.zeros_like(Ww)
    H = X_work @ X_work.T
    L = inv_cholesky(H, damping)

    dX = Xt_work - X_work
    D = dX @ X_work.T
    P = alpha * (np.triu(D @ L.T, 1) @ L)

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    for ct in range(n_tn):
        c0 = ct * tile
        c1 = min(c0 + tile, n)
        B = c1 - c0

        # Freeze codebooks for this column-tile
        codebooks = []
        for ti in range(n_tm):
            r0 = ti * tile
            r1 = min(r0 + tile, m)
            if bits_per_tile is not None:
                k = int(bits_per_tile[ti, ct])
                k = max(3, min(7, k))
            else:
                k = bits
            tile_data = Ww[r0:r1, c0:c1]
            nl = 2 ** k
            lo = float(tile_data.min())
            hi = float(tile_data.max())
            step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
            codebooks.append((r0, r1, lo, step, k))

        def apply_frozen_codebook(col_data):
            q = np.zeros_like(col_data)
            for r0, r1, lo, step, k in codebooks:
                if step == 0.0:
                    q[r0:r1] = col_data[r0:r1]
                else:
                    nl = 2 ** k
                    q[r0:r1] = np.clip(
                        np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
            return q

        E_block = np.zeros((m, B))
        W_pre_block = np.zeros((m, B))
        for j in range(B):
            c = c0 + j
            w_pre = Ww[:, c].copy()
            W_pre_block[:, j] = w_pre

            Q[:, c] = apply_frozen_codebook(w_pre)
            e = w_pre - Q[:, c]
            E_block[:, j] = e / L[c, c]
            end = min(c0 + B, n)
            Ww[:, c:end] -= np.outer(E_block[:, j], L[c, c:end])
            Ww[:, c:end] += np.outer(w_pre, P[c, c:end])

        # Outer block propagation to remaining columns
        if c1 < n:
            Ww[:, c1:] -= E_block @ L[c0:c1, c1:]
            Ww[:, c1:] += W_pre_block @ P[c0:c1, c1:]

    # Unpermute back to original column order
    if order is not None:
        Q_orig = np.zeros_like(Q)
        Q_orig[:, order] = Q
        return Q_orig
    return Q

# ============================================================================
# RTN baseline (per-tile quantization, no correction)
# ============================================================================

def rtn_baseline(W, bits, tile=TILE_SIZE):
    return quantize_tiles(W, bits, tile)

# ============================================================================
# BiIP diagonal balancing (from R3)
# ============================================================================

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11).
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}, S_G similar.
    Returns: S_G, S_X, W_transformed."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)

    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_G = np.diag(sg_diag)

    W_transformed = S_G @ W @ S_X
    return S_G, S_X, W_transformed

# ============================================================================
# Hadamard rotation (from R3/R9)
# ============================================================================

def hadamard_matrix(n):
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
    return H / np.sqrt(n)

def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return np.diag(signs) @ H, signs

# ============================================================================
# Column orderings
# ============================================================================

def order_descending_diag_H(H_X):
    """Act-order: descending diag(H_X) — process high-sensitivity columns first."""
    return np.argsort(np.diag(H_X))[::-1]

def order_left_to_right(n):
    return np.arange(n)

def order_ascending_diag_H(H_X):
    """Reverse act-order: ascending diag(H_X)."""
    return np.argsort(np.diag(H_X))

# ============================================================================
# DP tile allocation (simplified from R9)
# ============================================================================

def compute_tile_distortions(W, H_G, H_X, tile, k_range):
    """Compute per-tile distortion for each K value."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    distortions = {}
    for k in k_range:
        dist = np.zeros((n_tm, n_tn))
        for ti in range(n_tm):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            for tj in range(n_tn):
                c0, c1 = tj * tile, min((tj + 1) * tile, n)
                block = W[r0:r1, c0:c1]
                Wq = quantize_tiles(block, k, tile)
                E = block - Wq
                # Hessian-weighted distortion for this tile
                H_G_t = H_G[r0:r1, r0:r1]
                H_X_t = H_X[c0:c1, c0:c1]
                dist[ti, tj] = np.trace(H_G_t @ E @ H_X_t @ E.T)
        distortions[k] = dist
    return distortions

def tile_k_dp_allocate(distortions, n_tm, n_tn, tile_elems, budget_bits, k_range):
    """DP allocation: minimize total distortion subject to bit budget."""
    k_list = sorted(k_range)
    n_tiles = n_tm * n_tn
    # Flatten distortions: tile index = ti * n_tn + tj
    D = np.zeros((n_tiles, len(k_list)))
    for ki, k in enumerate(k_list):
        dist = distortions[k]
        for ti in range(n_tm):
            for tj in range(n_tn):
                D[ti * n_tn + tj, ki] = dist[ti, tj]

    # DP: dp[b] = min distortion using first t tiles with b bits
    min_bits = k_list[0] * tile_elems
    max_bits = k_list[-1] * tile_elems
    total_budget = budget_bits

    # Use per-tile DP
    # dp[t][b] = min distortion for first t tiles using b bits
    INF = 1e18
    dp = np.full((n_tiles + 1, total_budget + 1), INF)
    dp[0, 0] = 0
    parent = np.zeros((n_tiles + 1, total_budget + 1), dtype=int)

    for t in range(n_tiles):
        for b in range(total_budget + 1):
            if dp[t, b] >= INF:
                continue
            for ki, k in enumerate(k_list):
                cost = k * tile_elems
                if b + cost <= total_budget:
                    new_dist = dp[t, b] + D[t, ki]
                    if new_dist < dp[t + 1, b + cost]:
                        dp[t + 1, b + cost] = new_dist
                        parent[t + 1, b + cost] = ki

    # Find best budget
    best_b = min(range(total_budget + 1), key=lambda b: dp[n_tiles, b])
    if dp[n_tiles, best_b] >= INF:
        # Fallback: uniform K
        avg_k = int(round(budget_bits / (n_tiles * tile_elems)))
        return np.full((n_tm, n_tn), max(k_list[0], min(k_list[-1], avg_k)))

    # Backtrack
    alloc = np.zeros(n_tiles, dtype=int)
    b = best_b
    for t in range(n_tiles, 0, -1):
        ki = parent[t, b]
        alloc[t - 1] = k_list[ki]
        b -= k_list[ki] * tile_elems

    K_alloc = alloc.reshape(n_tm, n_tn)
    return K_alloc

# ============================================================================
# Transform pipeline
# ============================================================================

def apply_transform(W, X, Xt, H_X, H_G, transform, rot_seed=42):
    """Apply transform pipeline. Returns transformed W, X, Xt, H_X, H_G, and inverse params.

    transform: 'none', 'scale' (BiIP only), 'rotate' (BiIP + Hadamard both sides)
    Uses a FIXED rotation seed so all arms share the same transform.
    X, Xt are (n_in, n_samples).
    """
    m, n = W.shape
    S_G = np.eye(m)
    S_X = np.eye(n)
    U = np.eye(m)
    V = np.eye(n)

    W_t = W.copy()
    X_t = X.copy()
    Xt_t = Xt.copy()
    H_X_t = H_X.copy()
    H_G_t = H_G.copy()

    if transform in ('scale', 'rotate'):
        # BiIP diagonal balancing
        S_G, S_X, W_t = biip_scaling(W, H_X, H_G)
        S_G_inv = np.linalg.inv(S_G)
        S_X_inv = np.linalg.inv(S_X)
        X_t = S_X_inv @ X_t       # (n_in, n_in) @ (n_in, n_samples) = (n_in, n_samples)
        Xt_t = S_X_inv @ Xt_t
        H_X_t = S_X_inv @ H_X_t @ S_X_inv
        H_G_t = S_G_inv @ H_G_t @ S_G_inv

    if transform == 'rotate':
        # Signed randomized Hadamard both sides (fixed seed for reproducibility)
        rng_rot = np.random.RandomState(rot_seed)
        V, _ = signed_random_hadamard(n, rng_rot)
        U, _ = signed_random_hadamard(m, rng_rot)
        W_t = U @ W_t @ V.T
        X_t = V @ X_t             # (n_in, n_in) @ (n_in, n_samples) = (n_in, n_samples)
        Xt_t = V @ Xt_t
        H_X_t = V @ H_X_t @ V.T
        H_G_t = U @ H_G_t @ U.T

    return W_t, X_t, Xt_t, H_X_t, H_G_t, U, V, S_G, S_X

def inverse_transform(Q_t, U, V, S_G, S_X):
    """Inverse: Q_orig = S_G^{-1} @ U^T @ Q_t @ V @ S_X^{-1}"""
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ Q_t @ V @ S_X_inv

# ============================================================================
# Experiment
# ============================================================================

def run_single_arm(W, X, Xt, H_X, H_G, K, transform, quant_method, alpha, rot_seed=42):
    """Run a single experimental arm.

    transform: 'none', 'scale', 'rotate'
    quant_method: 'rtn', 'gptq_lr', 'gptq_actorder', 'gptaq_lr', 'gptaq_actorder'
    alpha: GPTAQ alpha (0.0 for pure GPTQ, 1.0 for paper-faithful, 0.25 for R9 post-rotation)
    """
    m, n = W.shape

    # Apply transform (uses fixed rot_seed for reproducibility across arms)
    W_t, X_t, Xt_t, H_X_t, H_G_t, U, V, S_G, S_X = apply_transform(
        W, X, Xt, H_X, H_G, transform, rot_seed)

    # Determine column order
    if 'actorder' in quant_method:
        order = order_descending_diag_H(H_X_t)
    elif 'lr' in quant_method:
        order = order_left_to_right(n)
    else:
        order = None

    # Quantize
    if quant_method == 'rtn':
        Q_t = rtn_baseline(W_t, K)
    else:
        Q_t = gptaq_correction(W_t, X_t, Xt_t, K, TILE_SIZE,
                                damping=0.01, alpha=alpha, order=order)

    # Inverse transform
    Q_orig = inverse_transform(Q_t, U, V, S_G, S_X)

    # Compute error in ORIGINAL space
    E = W - Q_orig

    # Metrics
    hwe = hessian_weighted_error(E, H_G, H_X)
    mse = raw_mse(E)

    # Spectral analysis (in original space)
    spec = spectral_analysis(E, H_X, H_G)
    acorr = anti_correlation(spec['eig_H_X'], spec['error_energy_HG_weighted'])

    # Diagonal CV of H_X (in transformed space, to measure rotation effect)
    diag_cv_transformed = diag_cv(H_X_t)
    diag_cv_original = diag_cv(H_X)

    return {
        'hwe': hwe,
        'mse': mse,
        'anti_correlation': acorr,
        'diag_cv_original': diag_cv_original,
        'diag_cv_transformed': diag_cv_transformed,
        'eig_H_X': spec['eig_H_X'].tolist(),
        'error_energy': spec['error_energy_HG_weighted'].tolist(),
    }

def run_allocation_arm(W, X, Xt, H_X, H_G, avg_k, transform, quant_method, alpha, rot_seed=42):
    """Run arm with DP tile allocation (variable K per tile).

    Key fix: K_alloc is computed AFTER applying column order, so tile indices
    match between allocation and gptaq_correction processing.
    """
    m, n = W.shape

    W_t, X_t, Xt_t, H_X_t, H_G_t, U, V, S_G, S_X = apply_transform(
        W, X, Xt, H_X, H_G, transform, rot_seed)

    # Determine column order
    if 'actorder' in quant_method:
        order = order_descending_diag_H(H_X_t)
    else:
        order = order_left_to_right(n)

    # Apply column order BEFORE computing allocation so tile indices match
    W_t_perm = W_t[:, order]
    X_t_perm = X_t[order, :]
    Xt_t_perm = Xt_t[order, :]
    H_X_t_perm = H_X_t[np.ix_(order, order)]

    # Compute tile distortions on PERMUTED weights
    k_range = range(3, 7)
    n_tm = (m + TILE_SIZE - 1) // TILE_SIZE
    n_tn = (n + TILE_SIZE - 1) // TILE_SIZE
    dists = compute_tile_distortions(W_t_perm, H_G_t, H_X_t_perm, TILE_SIZE, k_range)
    budget = avg_k * m * n
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, TILE_SIZE * TILE_SIZE, budget, k_range)

    # Quantize with per-tile K (already permuted, so order=None)
    if quant_method == 'rtn':
        Q_t_perm = quantize_tiles(W_t_perm, K_alloc, TILE_SIZE)
    else:
        Q_t_perm = gptaq_correction(W_t_perm, X_t_perm, Xt_t_perm, avg_k, TILE_SIZE,
                                    damping=0.01, alpha=alpha, order=None,
                                    bits_per_tile=K_alloc)

    # Unpermute back to transformed space
    Q_t = np.zeros_like(Q_t_perm)
    Q_t[:, order] = Q_t_perm

    # Inverse transform to original space
    Q_orig = inverse_transform(Q_t, U, V, S_G, S_X)
    E = W - Q_orig

    hwe = hessian_weighted_error(E, H_G, H_X)
    mse = raw_mse(E)
    spec = spectral_analysis(E, H_X, H_G)
    acorr = anti_correlation(spec['eig_H_X'], spec['error_energy_HG_weighted'])

    return {
        'hwe': hwe,
        'mse': mse,
        'anti_correlation': acorr,
        'K_alloc': K_alloc.tolist(),
    }

def run_experiment():
    print("=" * 80)
    print("R14-NoiseShapeStack: Noise Shaping Within the Full Stack")
    print("=" * 80)

    print("\nLoading real Qwen3.8-27B weights...")
    all_weights = load_real_weights()
    print(f"  Loaded {len(all_weights)} tensors")

    all_results = {}
    t_start = time.time()

    # Define arms
    # (name, transform, quant_method, alpha)
    arms = [
        # --- No transform ---
        ('RTN',              'none',   'rtn',            0.0),
        ('GPTQ_LR',          'none',   'gptq_lr',        0.0),
        ('GPTQ_actorder',    'none',   'gptq_actorder',  0.0),
        ('GPTAQ_LR_a1',      'none',   'gptaq_lr',       1.0),
        ('GPTAQ_actorder_a1','none',   'gptaq_actorder', 1.0),
        # --- Scale (BiIP only) ---
        ('Scale+RTN',              'scale', 'rtn',            0.0),
        ('Scale+GPTQ_LR',          'scale', 'gptq_lr',        0.0),
        ('Scale+GPTQ_actorder',    'scale', 'gptq_actorder',  0.0),
        ('Scale+GPTAQ_actorder_a1','scale', 'gptaq_actorder', 1.0),
        # --- Rotate (BiIP + Hadamard) ---
        ('Rot+RTN',                'rotate', 'rtn',            0.0),
        ('Rot+GPTQ_LR',            'rotate', 'gptq_lr',        0.0),
        ('Rot+GPTQ_actorder',      'rotate', 'gptq_actorder',  0.0),
        ('Rot+GPTAQ_LR_a1',        'rotate', 'gptaq_lr',       1.0),
        ('Rot+GPTAQ_actorder_a1',  'rotate', 'gptaq_actorder', 1.0),
        ('Rot+GPTAQ_LR_a025',      'rotate', 'gptaq_lr',       0.25),
        ('Rot+GPTAQ_actorder_a025','rotate', 'gptaq_actorder', 0.25),
        # --- Reverse act-order (diagnostic) ---
        ('GPTQ_rev_actorder',      'none',   'gptq_rev',       0.0),
        ('Rot+GPTQ_rev_actorder',  'rotate', 'gptq_rev',       0.0),
    ]

    # Handle reverse act-order separately (not in standard quant_method set)
    # We'll use a special quant_method name

    alloc_arms = [
        # (name, transform, quant_method, alpha)
        ('Alloc+RTN',              'none',   'rtn',            0.0),
        ('Alloc+GPTQ_LR',          'none',   'gptq_lr',        0.0),
        ('Alloc+GPTQ_actorder',    'none',   'gptq_actorder',  0.0),
        ('Rot+Alloc+RTN',          'rotate', 'rtn',            0.0),
        ('Rot+Alloc+GPTQ_LR',      'rotate', 'gptq_lr',        0.0),
        ('Rot+Alloc+GPTQ_actorder','rotate', 'gptq_actorder',  0.0),
        ('Rot+Alloc+GPTAQ_actorder_a1', 'rotate', 'gptaq_actorder', 1.0),
    ]

    for tensor_name in TENSOR_NAMES:
        if tensor_name not in all_weights:
            continue
        W_full = all_weights[tensor_name]
        print(f"\n{'='*60}")
        print(f"Tensor: {tensor_name} (full shape {W_full.shape})")
        print(f"{'='*60}")

        for slice_idx in range(len(SLICES)):
            W = get_slice(W_full, slice_idx)
            print(f"\n  Slice {slice_idx}: shape {W.shape}, "
                  f"range [{W.min():.6f}, {W.max():.6f}]")

            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                X = generate_calibration(W.shape[1], N_SAMPLES_CALIB, rng)
                H_X, H_G = compute_hessians(W, X)
                Xt = X + rng.standard_normal(X.shape) * 0.01  # FP flow

                for K in K_VALUES:
                    for arm_name, transform, quant_method, alpha in arms:
                        try:
                            if quant_method == 'gptq_rev':
                                # Reverse act-order
                                W_t, X_t, Xt_t, H_X_t, H_G_t, U, V, S_G, S_X = \
                                    apply_transform(W, X, Xt, H_X, H_G, transform)
                                order = order_ascending_diag_H(H_X_t)
                                Q_t = gptaq_correction(W_t, X_t, Xt_t, K, TILE_SIZE,
                                                       damping=0.01, alpha=alpha, order=order)
                                Q_orig = inverse_transform(Q_t, U, V, S_G, S_X)
                                E = W - Q_orig
                                hwe = hessian_weighted_error(E, H_G, H_X)
                                mse = raw_mse(E)
                                spec = spectral_analysis(E, H_X, H_G)
                                acorr = anti_correlation(spec['eig_H_X'], spec['error_energy_HG_weighted'])
                                result = {
                                    'hwe': hwe, 'mse': mse,
                                    'anti_correlation': acorr,
                                    'diag_cv_original': diag_cv(H_X),
                                    'diag_cv_transformed': diag_cv(H_X_t),
                                }
                            else:
                                result = run_single_arm(
                                    W, X, Xt, H_X, H_G, K, transform, quant_method, alpha)
                        except Exception as e:
                            result = {'hwe': 1e20, 'mse': 1e20, 'error': str(e)}

                        key = f"{tensor_name}_s{slice_idx}_seed{seed}_K{K}_{arm_name}"
                        all_results[key] = {
                            'tensor': tensor_name, 'slice': slice_idx, 'seed': seed,
                            'K': K, 'arm': arm_name,
                            'transform': transform, 'quant_method': quant_method,
                            'alpha': alpha, **result,
                        }

                # Allocation arms (only at avg_k=5 for time)
                for arm_name, transform, quant_method, alpha in alloc_arms:
                    try:
                        result = run_allocation_arm(
                            W, X, Xt, H_X, H_G, 5, transform, quant_method, alpha)
                    except Exception as e:
                        result = {'hwe': 1e20, 'mse': 1e20, 'error': str(e)}

                    key = f"{tensor_name}_s{slice_idx}_seed{seed}_K5_{arm_name}"
                    all_results[key] = {
                        'tensor': tensor_name, 'slice': slice_idx, 'seed': seed,
                        'K': 5, 'arm': arm_name,
                        'transform': transform, 'quant_method': quant_method,
                        'alpha': alpha, **result,
                    }

    t_elapsed = time.time() - t_start
    print(f"\n\nTotal time: {t_elapsed:.1f}s")

    # Save raw results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'r14-noise-shape-stack-results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {OUTPUT_DIR / 'r14-noise-shape-stack-results.json'}")

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    # Helper: get mean HWE for a given (tensor, K, arm) across slices and seeds
    def get_mean_hwe(tensor, K, arm):
        vals = [v['hwe'] for k, v in all_results.items()
                if v['tensor'] == tensor and v['K'] == K and v['arm'] == arm
                and 'error' not in v]
        return np.mean(vals) if vals else float('inf')

    def get_mean_metric(tensor, K, arm, metric):
        vals = [v[metric] for k, v in all_results.items()
                if v['tensor'] == tensor and v['K'] == K and v['arm'] == arm
                and 'error' not in v and metric in v]
        return np.mean(vals) if vals else float('inf')

    # --- 1. Act-order vs LR: with and without rotation ---
    print("\n" + "=" * 80)
    print("1. ACT-ORDER vs LR: Does act-order compose with rotation?")
    print("=" * 80)
    print("\n  Improvement of act-order over LR (HWE reduction %):")
    print(f"  {'Tensor':<12} {'K':>3} {'No-transform':>14} {'Scale':>14} {'Rotate':>14}")
    print(f"  {'-'*60}")

    for tensor in TENSOR_NAMES:
        for K in K_VALUES:
            # GPTQ actorder vs GPTQ LR
            lr_none = get_mean_hwe(tensor, K, 'GPTQ_LR')
            ao_none = get_mean_hwe(tensor, K, 'GPTQ_actorder')
            lr_scale = get_mean_hwe(tensor, K, 'Scale+GPTQ_LR')
            ao_scale = get_mean_hwe(tensor, K, 'Scale+GPTQ_actorder')
            lr_rot = get_mean_hwe(tensor, K, 'Rot+GPTQ_LR')
            ao_rot = get_mean_hwe(tensor, K, 'Rot+GPTQ_actorder')

            imp_none = (1 - ao_none / max(lr_none, 1e-20)) * 100
            imp_scale = (1 - ao_scale / max(lr_scale, 1e-20)) * 100
            imp_rot = (1 - ao_rot / max(lr_rot, 1e-20)) * 100

            print(f"  {tensor:<12} {K:>3} {imp_none:>+13.1f}% {imp_scale:>+13.1f}% {imp_rot:>+13.1f}%")

    # --- 2. Act-order vs reverse act-order (direction matters?) ---
    print("\n" + "=" * 80)
    print("2. ACT-ORDER vs REVERSE: Does direction matter post-rotation?")
    print("=" * 80)
    print(f"\n  {'Tensor':<12} {'K':>3} {'AO-LR (none)':>14} {'Rev-LR (none)':>14} {'AO-LR (rot)':>14} {'Rev-LR (rot)':>14}")
    print(f"  {'-'*75}")

    for tensor in TENSOR_NAMES:
        for K in K_VALUES:
            ao_none = get_mean_hwe(tensor, K, 'GPTQ_actorder')
            lr_none = get_mean_hwe(tensor, K, 'GPTQ_LR')
            rev_none = get_mean_hwe(tensor, K, 'GPTQ_rev_actorder')
            ao_rot = get_mean_hwe(tensor, K, 'Rot+GPTQ_actorder')
            lr_rot = get_mean_hwe(tensor, K, 'Rot+GPTQ_LR')
            rev_rot = get_mean_hwe(tensor, K, 'Rot+GPTQ_rev_actorder')

            imp_ao_none = (1 - ao_none / max(lr_none, 1e-20)) * 100
            imp_rev_none = (1 - rev_none / max(lr_none, 1e-20)) * 100
            imp_ao_rot = (1 - ao_rot / max(lr_rot, 1e-20)) * 100
            imp_rev_rot = (1 - rev_rot / max(lr_rot, 1e-20)) * 100

            print(f"  {tensor:<12} {K:>3} {imp_ao_none:>+13.1f}% {imp_rev_none:>+13.1f}% {imp_ao_rot:>+13.1f}% {imp_rev_rot:>+13.1f}%")

    # --- 3. Error spectrum: anti-correlation with and without rotation ---
    print("\n" + "=" * 80)
    print("3. ERROR SPECTRUM: Anti-correlation (error energy vs Hessian eigenvalues)")
    print("=" * 80)
    print("\n  Negative = error pushed away from high-Hessian directions (good)")
    print(f"\n  {'Arm':<28} {'K=3':>8} {'K=4':>8} {'K=5':>8} {'K=6':>8}")
    print(f"  {'-'*55}")

    spectrum_arms = ['RTN', 'GPTQ_LR', 'GPTQ_actorder', 'GPTAQ_actorder_a1',
                     'Rot+RTN', 'Rot+GPTQ_LR', 'Rot+GPTQ_actorder', 'Rot+GPTAQ_actorder_a1']
    for arm in spectrum_arms:
        row = f"  {arm:<28}"
        for K in K_VALUES:
            vals = [get_mean_metric(t, K, arm, 'anti_correlation') for t in TENSOR_NAMES]
            row += f" {np.mean(vals):>+7.3f}"
        print(row)

    # --- 4. Diagonal CV: how rotation uniformizes diag(H_X) ---
    print("\n" + "=" * 80)
    print("4. DIAG(H_X) CV: How rotation uniformizes the Hessian diagonal")
    print("=" * 80)
    print(f"\n  {'Tensor':<12} {'CV original':>14} {'CV after BiIP':>14} {'CV after rotate':>14}")
    print(f"  {'-'*55}")

    for tensor in TENSOR_NAMES:
        cv_orig = get_mean_metric(tensor, 4, 'RTN', 'diag_cv_original')
        cv_scale = get_mean_metric(tensor, 4, 'Scale+RTN', 'diag_cv_transformed')
        cv_rot = get_mean_metric(tensor, 4, 'Rot+RTN', 'diag_cv_transformed')
        print(f"  {tensor:<12} {cv_orig:>14.4f} {cv_scale:>14.4f} {cv_rot:>14.4f}")

    # --- 5. Marginal contributions (at K=5, macro mean over tensors) ---
    print("\n" + "=" * 80)
    print("5. MARGINAL CONTRIBUTIONS (K=5, macro mean over tensors, HWE vs RTN)")
    print("=" * 80)

    rtn_hwe = np.mean([get_mean_hwe(t, 5, 'RTN') for t in TENSOR_NAMES])
    print(f"\n  RTN baseline: {rtn_hwe:.6e}")

    marginal_arms = [
        'GPTQ_LR', 'GPTQ_actorder', 'GPTAQ_actorder_a1',
        'Scale+RTN', 'Scale+GPTQ_actorder', 'Scale+GPTAQ_actorder_a1',
        'Rot+RTN', 'Rot+GPTQ_LR', 'Rot+GPTQ_actorder',
        'Rot+GPTAQ_LR_a1', 'Rot+GPTAQ_actorder_a1', 'Rot+GPTAQ_actorder_a025',
    ]
    print(f"\n  {'Arm':<28} {'HWE':>14} {'vs RTN':>10} {'vs prev':>10}")
    print(f"  {'-'*65}")

    prev_hwe = rtn_hwe
    for arm in marginal_arms:
        hwe = np.mean([get_mean_hwe(t, 5, arm) for t in TENSOR_NAMES])
        imp_rtn = (1 - hwe / max(rtn_hwe, 1e-20)) * 100
        imp_prev = (1 - hwe / max(prev_hwe, 1e-20)) * 100
        print(f"  {arm:<28} {hwe:>14.6e} {imp_rtn:>+9.1f}% {imp_prev:>+9.1f}%")
        prev_hwe = hwe

    # --- 6. Full stack: marginal contributions step by step ---
    print("\n" + "=" * 80)
    print("6. FULL STACK: Step-by-step marginal contribution (K=5)")
    print("=" * 80)
    print(f"\n  {'Step':<35} {'HWE':>14} {'vs RTN':>10} {'marginal':>10}")
    print(f"  {'-'*72}")

    stack_steps = [
        ('RTN (baseline)',                    'RTN'),
        ('+ GPTQ act-order',                   'GPTQ_actorder'),
        ('+ GPTAQ α=1.0',                     'GPTAQ_actorder_a1'),
        ('+ BiIP scaling',                     'Scale+GPTAQ_actorder_a1'),
        ('+ Hadamard rotation',                'Rot+GPTAQ_actorder_a1'),
        ('Full: Rot+GPTAQ α=0.25',             'Rot+GPTAQ_actorder_a025'),
    ]
    prev = None
    for label, arm in stack_steps:
        hwe = np.mean([get_mean_hwe(t, 5, arm) for t in TENSOR_NAMES])
        imp_rtn = (1 - hwe / max(rtn_hwe, 1e-20)) * 100
        if prev is not None:
            imp_marg = (1 - hwe / max(prev, 1e-20)) * 100
        else:
            imp_marg = 0.0
        print(f"  {label:<35} {hwe:>14.6e} {imp_rtn:>+9.1f}% {imp_marg:>+9.1f}%")
        prev = hwe

    # --- 7. Allocation interaction with act-order ---
    print("\n" + "=" * 80)
    print("7. ALLOCATION × ACT-ORDER (K=5 avg, macro mean)")
    print("=" * 80)
    print(f"\n  {'Arm':<35} {'HWE':>14} {'vs RTN':>10}")
    print(f"  {'-'*62}")

    alloc_arms_names = [
        'RTN', 'GPTQ_LR', 'GPTQ_actorder',
        'Alloc+RTN', 'Alloc+GPTQ_LR', 'Alloc+GPTQ_actorder',
        'Rot+RTN', 'Rot+GPTQ_LR', 'Rot+GPTQ_actorder',
        'Rot+Alloc+RTN', 'Rot+Alloc+GPTQ_LR', 'Rot+Alloc+GPTQ_actorder',
        'Rot+Alloc+GPTAQ_actorder_a1',
    ]
    for arm in alloc_arms_names:
        hwe = np.mean([get_mean_hwe(t, 5, arm) for t in TENSOR_NAMES])
        imp_rtn = (1 - hwe / max(rtn_hwe, 1e-20)) * 100
        print(f"  {arm:<35} {hwe:>14.6e} {imp_rtn:>+9.1f}%")

    # --- 8. Paired win rates: act-order vs LR ---
    print("\n" + "=" * 80)
    print("8. PAIRED WIN RATES: act-order vs LR (per tensor×slice×seed)")
    print("=" * 80)
    print(f"\n  {'Transform':<12} {'K':>3} {'AO wins':>10} {'LR wins':>10} {'Total':>8} {'AO win%':>10}")
    print(f"  {'-'*55}")

    for transform_label, ao_arm, lr_arm in [
        ('none',   'GPTQ_actorder',      'GPTQ_LR'),
        ('scale',  'Scale+GPTQ_actorder','Scale+GPTQ_LR'),
        ('rotate', 'Rot+GPTQ_actorder',  'Rot+GPTQ_LR'),
    ]:
        for K in K_VALUES:
            ao_wins = 0
            lr_wins = 0
            total = 0
            for tensor in TENSOR_NAMES:
                for slice_idx in range(len(SLICES)):
                    for seed in SEEDS:
                        ao_key = f"{tensor}_s{slice_idx}_seed{seed}_K{K}_{ao_arm}"
                        lr_key = f"{tensor}_s{slice_idx}_seed{seed}_K{K}_{lr_arm}"
                        ao_val = all_results.get(ao_key, {}).get('hwe', float('inf'))
                        lr_val = all_results.get(lr_key, {}).get('hwe', float('inf'))
                        if ao_val < lr_val:
                            ao_wins += 1
                        elif lr_val < ao_val:
                            lr_wins += 1
                        total += 1
            print(f"  {transform_label:<12} {K:>3} {ao_wins:>10} {lr_wins:>10} {total:>8} {ao_wins/max(total,1)*100:>9.0f}%")

    # --- 9. MSE vs HWE trade-off ---
    print("\n" + "=" * 80)
    print("9. MSE vs HWE TRADE-OFF (K=5, macro mean, ratios vs RTN)")
    print("=" * 80)
    print(f"\n  {'Arm':<28} {'HWE/RTN':>10} {'MSE/RTN':>10}")
    print(f"  {'-'*50}")

    rtn_mse = np.mean([get_mean_metric(t, 5, 'RTN', 'mse') for t in TENSOR_NAMES])
    tradeoff_arms = ['RTN', 'GPTQ_LR', 'GPTQ_actorder', 'GPTAQ_actorder_a1',
                     'Rot+RTN', 'Rot+GPTQ_LR', 'Rot+GPTQ_actorder',
                     'Rot+GPTAQ_actorder_a1', 'Rot+GPTAQ_actorder_a025']
    for arm in tradeoff_arms:
        hwe = np.mean([get_mean_hwe(t, 5, arm) for t in TENSOR_NAMES])
        mse = np.mean([get_mean_metric(t, 5, arm, 'mse') for t in TENSOR_NAMES])
        print(f"  {arm:<28} {hwe/max(rtn_hwe,1e-20):>10.3f} {mse/max(rtn_mse,1e-20):>10.3f}")

    # --- 10. GPTAQ α comparison post-rotation ---
    print("\n" + "=" * 80)
    print("10. GPTAQ α COMPARISON POST-ROTATION (K=5, macro mean)")
    print("=" * 80)
    print(f"\n  {'Arm':<35} {'HWE':>14} {'vs Rot+RTN':>12}")
    print(f"  {'-'*64}")

    rot_rtn_hwe = np.mean([get_mean_hwe(t, 5, 'Rot+RTN') for t in TENSOR_NAMES])
    alpha_arms = [
        ('Rot+GPTQ_LR (α=0)',          'Rot+GPTQ_LR'),
        ('Rot+GPTQ_actorder (α=0)',    'Rot+GPTQ_actorder'),
        ('Rot+GPTAQ_LR_a1 (α=1.0)',    'Rot+GPTAQ_LR_a1'),
        ('Rot+GPTAQ_actorder_a1 (α=1.0)','Rot+GPTAQ_actorder_a1'),
        ('Rot+GPTAQ_LR_a025 (α=0.25)',  'Rot+GPTAQ_LR_a025'),
        ('Rot+GPTAQ_actorder_a025 (α=0.25)','Rot+GPTAQ_actorder_a025'),
    ]
    for label, arm in alpha_arms:
        hwe = np.mean([get_mean_hwe(t, 5, arm) for t in TENSOR_NAMES])
        imp = (1 - hwe / max(rot_rtn_hwe, 1e-20)) * 100
        print(f"  {label:<35} {hwe:>14.6e} {imp:>+11.1f}%")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return all_results

if __name__ == '__main__':
    results = run_experiment()
