#!/usr/bin/env python3
"""
R25 — Multi-Seed Statistical Robustness Validation

Validates the top 5 findings from Waves 1-3 across 20 random calibration seeds.
Tests whether improvements are statistically robust, not artifacts of seed selection.

Methods tested (all with held-out evaluation):
  1. BiIP + Hadamard rotation (R3)
  2. DP tile allocation (R1)
  3. Full GPTQ post-rotation (R9/R11)
  4. Block-diagonal GPTQ (R17)
  5. Unified stack (R11)

Baseline: RTN (per-tile uniform quantization, no transforms)

Experiment matrix:
  20 seeds × 7 methods (6 + RTN) × 4 tensors × 3 K values = 1,680 evaluations

For each seed:
  - Generate DIFFERENT calibration (different outlier channels + Gaussian samples)
  - 80/20 train/test split
  - Fit transforms on train Hessians, evaluate on held-out test Hessians

Statistical analysis per (tensor, K) cell:
  - Mean ± std improvement % over RTN
  - 95% confidence interval (t-distribution, df=19)
  - Min / max improvement (worst/best seed)
  - Win rate (fraction of seeds where method beats RTN)
  - Paired t-test p-value vs RTN
  - Paired t-test between methods

Ranking stability:
  - For each seed, rank methods by held-out HWE
  - Check if ranking is stable across all 20 seeds
  - Identify any seed where a finding reverses (method worse than RTN)

Output: receipts/research/r25-multiseed-results.json
"""

import json
import os
import sys
import time
import warnings
import numpy as np
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "receipts", "research", "r25-multiseed-results.json"
)

# ─── Configuration ────────────────────────────────────────────────────────────
TILE         = 16
SLICE        = 128
K_VALUES     = [4, 5, 6]
N_SEEDS      = 20
N_CALIB      = 512
TRAIN_FRAC   = 0.80
DAMPING      = 0.01
BLOCK_SIZE   = 16
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
SEED_BASE    = 1000  # seed_base + i for i in range(20)

# ─── Quantizer: per-tile (16×16) uniform, MATCHED for all arms ─────────────────

def quantize_tile(w, k):
    """Per-tile uniform quantizer. k bits → 2^k levels."""
    nl = 2 ** k
    lo = float(w.min())
    hi = float(w.max())
    if hi - lo < 1e-15:
        return w.copy()
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_tiles(W, bits_or_alloc, tile=TILE):
    """Per-tile uniform quantization. bits_or_alloc: int or (n_tm, n_tn) array."""
    m, n = W.shape
    if np.isscalar(bits_or_alloc):
        K_alloc = np.full(((m + tile - 1) // tile, (n + tile - 1) // tile), bits_or_alloc)
    else:
        K_alloc = np.asarray(bits_or_alloc)
    Wq = np.zeros_like(W, dtype=np.float64)
    for ti in range(K_alloc.shape[0]):
        r0, r1 = ti * tile, min((ti + 1) * tile, m)
        for tj in range(K_alloc.shape[1]):
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            k = int(K_alloc[ti, tj])
            Wq[r0:r1, c0:c1] = quantize_tile(W[r0:r1, c0:c1], k)
    return Wq


# ─── Metrics ──────────────────────────────────────────────────────────────────

def hessian_weighted_error(E, H_G, H_X):
    """tr(H_G · E · H_X · E^T). E is (m, n), H_G (m,m), H_X (n,n)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))


# ─── Calibration generation (seeded, different per seed) ──────────────────────

def gen_calibration(n_in, n_samples, seed):
    """Synthetic activations: Gaussian + outlier channels.
    Different seed → different outlier channels + different Gaussian samples."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples)) * 0.1
    n_outliers = max(1, n_in // 20)
    outlier_rows = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_rows, :] *= 10.0
    return X


def compute_hessians(W, X):
    """H_X = X X^T / N, H_G = Y Y^T / N with Y = W X."""
    N = X.shape[1]
    H_X = (X @ X.T / N).astype(np.float64)
    Y = W @ X
    H_G = (Y @ Y.T / N).astype(np.float64)
    d_out, d_in = W.shape
    H_X *= d_in / max(np.trace(H_X), 1e-15)
    H_G *= d_out / max(np.trace(H_G), 1e-15)
    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)
    return H_G, H_X


# ─── Cholesky (correct convention: U = chol(inv(H+λI)).T) ─────────────────────

def inv_cholesky(H, damping):
    """Upper triangular U such that U^T U = inv(H + damping*I)."""
    d = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    try:
        U = np.linalg.cholesky(np.linalg.inv(H + lam * np.eye(d))).T
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(H + lam * np.eye(d))
        eigvals = np.maximum(eigvals, 1e-10)
        Hinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        U = np.linalg.cholesky(Hinv).T
    return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)


def block_diag_inv_cholesky(H, block_size, damping):
    """Block-diagonal upper triangular U."""
    d = H.shape[0]
    U = np.zeros((d, d))
    for i in range(0, d, block_size):
        j = min(i + block_size, d)
        block = H[i:j, i:j]
        U[i:j, i:j] = inv_cholesky(block, damping)
    return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)


# ─── Hadamard ─────────────────────────────────────────────────────────────────

def hadamard_matrix(n):
    """Sylvester-type normalized Hadamard (n must be power of 2)."""
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    """Signed randomized Hadamard: diag(±1) @ H."""
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return np.diag(signs) @ H, signs


# ─── BiIP diagonal balancing (R3 / KronQ Eq. 11) ──────────────────────────────

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11).
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}, S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}
    W' = S_G @ W @ S_X. Returns: S_G, S_X, W_transformed."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    S_G = np.diag(sg_diag)
    W_t = S_G @ W @ S_X
    return S_G, S_X, W_t


# ─── DP tile allocation (R1) ──────────────────────────────────────────────────

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    """Hessian-weighted distortion of one tile at K=k."""
    Wq = quantize_tile(W_tile, k)
    E = W_tile - Wq
    D = np.trace(H_G_sub @ E @ H_X_sub @ E.T)
    return max(D, 0.0)


def compute_tile_distortions(W, H_G, H_X, tile, k_range):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    dists = {}
    for ti in range(n_tm):
        r0, r1 = ti * tile, min((ti + 1) * tile, m)
        for tj in range(n_tn):
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            W_tile = W[r0:r1, c0:c1]
            H_G_sub = H_G[r0:r1, r0:r1]
            H_X_sub = H_X[c0:c1, c0:c1]
            for k in k_range:
                dists[(ti, tj, k)] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
    return dists


def tile_k_dp_allocate(distortions, n_tm, n_tn, tile_elements, budget_bits, k_range):
    """Multiple-choice knapsack DP for tile K allocation."""
    k_list = sorted(k_range)
    n_tiles = n_tm * n_tn
    n_k = len(k_list)
    INF = float('inf')

    D = np.full((n_tiles + 1, budget_bits + 1), INF)
    B = np.full((n_tiles + 1, budget_bits + 1), -1, dtype=int)
    D[0, 0] = 0.0

    for t in range(n_tiles):
        ti, tj = t // n_tn, t % n_tn
        for b in range(budget_bits + 1):
            if D[t, b] == INF:
                continue
            for ki, k in enumerate(k_list):
                kb = k * tile_elements
                if b + kb <= budget_bits:
                    nd = D[t, b] + distortions.get((ti, tj, k), INF)
                    if nd < D[t + 1, b + kb]:
                        D[t + 1, b + kb] = nd
                        B[t + 1, b + kb] = ki

    best_b = np.argmin(D[n_tiles])
    if D[n_tiles, best_b] == INF:
        best_b = budget_bits

    K_flat = np.zeros(n_tiles, dtype=int)
    b = best_b
    for t in range(n_tiles, 0, -1):
        ki = B[t, b]
        if ki < 0:
            ki = 0
        K_flat[t - 1] = k_list[ki]
        b -= k_list[ki] * tile_elements
    return K_flat.reshape(n_tm, n_tn)


def local_search_refine(W, K_in, H_G, H_X, tile=TILE, budget_k=None, max_iters=100):
    """Block-diagonal surrogate local search via single-bit TRANSFERS between tiles.
    Keeps total K-sum constant (budget-enforcing). Uses per-tile distortion
    (block-diagonal Hessian approximation), not full cross-tile HWE."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    K = np.asarray(K_in).copy()
    if K.ndim == 1:
        K = K.reshape(n_tm, n_tn)
    k_min, k_max = 3, 7

    # Precompute per-tile distortion at each K level
    tile_dists = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            W_tile = W[r0:r1, c0:c1]
            H_G_sub = H_G[r0:r1, r0:r1]
            H_X_sub = H_X[c0:c1, c0:c1]
            for k in range(k_min, k_max + 1):
                tile_dists[(ti, tj, k)] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)

    def total_hwe_fast(K_arr):
        return sum(tile_dists.get((ti, tj, int(K_arr[ti, tj])), 0.0)
                   for ti in range(n_tm) for tj in range(n_tn))

    current_hwe = total_hwe_fast(K)
    for _ in range(max_iters):
        best_swap = None
        best_delta = 1e-15
        for ti in range(n_tm):
            for tj in range(n_tn):
                cur_k = int(K[ti, tj])
                if cur_k > k_min:
                    donor_cost = tile_dists[(ti, tj, cur_k - 1)] - tile_dists[(ti, tj, cur_k)]
                    for ti2 in range(n_tm):
                        for tj2 in range(n_tn):
                            if ti == ti2 and tj == tj2:
                                continue
                            cur_k2 = int(K[ti2, tj2])
                            if cur_k2 < k_max:
                                recip_benefit = tile_dists[(ti2, tj2, cur_k2)] - tile_dists[(ti2, tj2, cur_k2 + 1)]
                                delta = recip_benefit - donor_cost
                                if delta > best_delta:
                                    best_delta = delta
                                    best_swap = ((ti, tj), (ti2, tj2))
        if best_swap is None:
            break
        (ti, tj), (ti2, tj2) = best_swap
        K[ti, tj] -= 1
        K[ti2, tj2] += 1
        current_hwe -= best_delta
    return K, current_hwe


def r1_dp_allocation(W, H_X_train, H_G_train, K, tile=TILE):
    """DP-refined tile allocation (fit on train Hessians). Returns Wq."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W, H_G_train, H_X_train, tile, k_range)
    budget = K * m * n
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
    K_alloc, _ = local_search_refine(W, K_alloc, H_G_train, H_X_train, tile, max_iters=50)
    return quantize_tiles(W, K_alloc, tile)


# ─── GPTQ variants ────────────────────────────────────────────────────────────

def _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile=None):
    """Pre-compute frozen tile codebooks from W."""
    tile_cb = {}
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    for ti in range(n_tm):
        r0, r1 = ti * tile, min((ti + 1) * tile, m)
        for tj in range(n_tn):
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            if bits_per_tile is not None:
                k = int(bits_per_tile[ti, tj])
            else:
                k = K
            tile_data = W[r0:r1, c0:c1]
            nl = 2 ** k
            lo = float(tile_data.min())
            hi = float(tile_data.max())
            step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
            tile_cb[(ti, tj)] = (r0, r1, lo, step, k)
    return tile_cb


def _quantize_col(col_data, col_idx, tile_cb, m, tile):
    """Quantize one column using frozen tile codebooks."""
    q = np.zeros_like(col_data)
    n_tm = (m + tile - 1) // tile
    for ti in range(n_tm):
        r0 = ti * tile
        r1 = min(r0 + tile, m)
        key = (ti, col_idx // tile)
        if key in tile_cb:
            _, _, lo, step, k = tile_cb[key]
            if step == 0.0:
                q[r0:r1] = col_data[r0:r1]
            else:
                nl = 2 ** k
                q[r0:r1] = np.clip(np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
    return q


def gptq_full(W, H_X, K, order=None, tile=TILE, damping=DAMPING, bits_per_tile=None):
    """Full GPTQ with correct Cholesky: U = chol(inv(H+λI)).T."""
    m, n = W.shape
    if order is None:
        order = np.arange(n)
    H_perm = H_X[np.ix_(order, order)]
    U = inv_cholesky(H_perm, damping)
    tile_cb = _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile)
    W_work = W.copy().astype(np.float64)
    Wq = np.zeros_like(W_work)
    for idx in range(n):
        q = order[idx]
        Wq[:, q] = _quantize_col(W_work[:, q], q, tile_cb, m, tile)
        e_q = W_work[:, q] - Wq[:, q]
        if idx < n - 1:
            remaining = order[idx + 1:]
            u_ii = U[idx, idx]
            if abs(u_ii) > 1e-15:
                update = np.outer(e_q, U[idx, idx + 1:] / u_ii)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)
    return Wq


def gptq_block_diag(W, H_X, K, order=None, tile=TILE, block_size=BLOCK_SIZE,
                    damping=DAMPING, bits_per_tile=None):
    """Block-diagonal GPTQ: full Cholesky within blocks, zero between blocks."""
    m, n = W.shape
    if order is None:
        order = np.arange(n)
    H_perm = H_X[np.ix_(order, order)]
    U = block_diag_inv_cholesky(H_perm, block_size, damping)
    tile_cb = _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile)
    W_work = W.copy().astype(np.float64)
    Wq = np.zeros_like(W_work)
    for idx in range(n):
        q = order[idx]
        Wq[:, q] = _quantize_col(W_work[:, q], q, tile_cb, m, tile)
        e_q = W_work[:, q] - Wq[:, q]
        if idx < n - 1:
            remaining = order[idx + 1:]
            u_ii = U[idx, idx]
            if abs(u_ii) > 1e-15:
                update = np.outer(e_q, U[idx, idx + 1:] / u_ii)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)
    return Wq


# ─── Method implementations ───────────────────────────────────────────────────

def method_rtn(W, H_X_train, H_G_train, K, rng):
    """RTN baseline: per-tile uniform quantization, no transforms."""
    return quantize_tiles(W, K, TILE)


def method_biip_hadamard(W, H_X_train, H_G_train, K, rng, U_rot=None, V_rot=None):
    """R3: BiIP + Hadamard rotation. Fit BiIP on train Hessians.
    Optional shared U_rot/V_rot for common-random-numbers ablation."""
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    if U_rot is None:
        U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    if V_rot is None:
        V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot.T @ W_s @ V_rot
    Wq_t = quantize_tiles(W_t, K, TILE)
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    W_hat = S_G_inv @ U_rot @ Wq_t @ V_rot.T @ S_X_inv
    return W_hat


def method_dp_allocation(W, H_X_train, H_G_train, K, rng):
    """R1: DP tile allocation (unrotated). Fit on train Hessians."""
    return r1_dp_allocation(W, H_X_train, H_G_train, K, TILE)


def method_rotation_full_gptq(W, H_X_train, H_G_train, K, rng, U_rot=None, V_rot=None):
    """R9/R11: BiIP + Hadamard rotation + Full GPTQ (α=0, pure GPTQ post-rotation).
    Uses U.T @ W @ V convention (consistent with BiIP_Hadamard and Unified).
    Optional shared U_rot/V_rot for common-random-numbers ablation."""
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    if U_rot is None:
        U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    if V_rot is None:
        V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot.T @ W_s @ V_rot
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = V_rot.T @ S_X_inv @ H_X_train @ S_X_inv @ V_rot
    H_G_t = U_rot.T @ S_G_inv @ H_G_train @ S_G_inv @ U_rot
    Wq_t = gptq_full(W_t, H_X_t, K, order=np.arange(W_t.shape[1]), tile=TILE)
    W_hat = S_G_inv @ U_rot @ Wq_t @ V_rot.T @ S_X_inv
    return W_hat


def method_rotation_blockdiag_gptq(W, H_X_train, H_G_train, K, rng, U_rot=None, V_rot=None):
    """R17: BiIP + Hadamard rotation + Block-diagonal GPTQ.
    Uses U.T @ W @ V convention. Optional shared U_rot/V_rot for CRN ablation."""
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    if U_rot is None:
        U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    if V_rot is None:
        V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot.T @ W_s @ V_rot
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = V_rot.T @ S_X_inv @ H_X_train @ S_X_inv @ V_rot
    H_G_t = U_rot.T @ S_G_inv @ H_G_train @ S_G_inv @ U_rot
    Wq_t = gptq_block_diag(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                           tile=TILE, block_size=BLOCK_SIZE)
    W_hat = S_G_inv @ U_rot @ Wq_t @ V_rot.T @ S_X_inv
    return W_hat


def method_rotation_dp_alloc(W, H_X_train, H_G_train, K, rng, U_rot=None, V_rot=None):
    """Rotation + DP allocation (no GPTQ). For proper bifurcation testing vs Rot_GPTQ.
    Uses U.T @ W @ V convention. Optional shared U_rot/V_rot for CRN ablation."""
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    if U_rot is None:
        U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    if V_rot is None:
        V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot.T @ W_s @ V_rot
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = V_rot.T @ S_X_inv @ H_X_train @ S_X_inv @ V_rot
    H_G_t = U_rot.T @ S_G_inv @ H_G_train @ S_G_inv @ U_rot
    Wq_t = r1_dp_allocation(W_t, H_X_t, H_G_t, K, TILE)
    W_hat = S_G_inv @ U_rot @ Wq_t @ V_rot.T @ S_X_inv
    return W_hat


def method_unified_stack(W, H_X_train, H_G_train, K, rng, U_rot=None, V_rot=None):
    """R11: Full unified stack: BiIP → Hadamard → p99 perm → DP alloc → GPTQ.
    Optional shared U_rot/V_rot for common-random-numbers ablation."""
    m, n = W.shape
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    if U_rot is None:
        U_rot, _ = signed_random_hadamard(m, rng)
    if V_rot is None:
        V_rot, _ = signed_random_hadamard(n, rng)
    W_sh = U_rot.T @ W_s @ V_rot
    p99_cols = np.percentile(np.abs(W_sh), 99, axis=0)
    perm_cols = np.argsort(p99_cols)
    p99_rows = np.percentile(np.abs(W_sh), 99, axis=1)
    perm_rows = np.argsort(p99_rows)
    W_shp = W_sh[perm_rows][:, perm_cols]
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = V_rot.T @ S_X_inv @ H_X_train @ S_X_inv @ V_rot
    H_G_t = U_rot.T @ S_G_inv @ H_G_train @ S_G_inv @ U_rot
    H_X_tp = H_X_t[perm_cols][:, perm_cols]
    H_G_tp = H_G_t[perm_rows][:, perm_rows]
    n_tm = (m + TILE - 1) // TILE
    n_tn = (n + TILE - 1) // TILE
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W_shp, H_G_tp, H_X_tp, TILE, k_range)
    budget = K * m * n
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, TILE * TILE, budget, k_range)
    K_alloc, _ = local_search_refine(W_shp, K_alloc, H_G_tp, H_X_tp, TILE, max_iters=50)
    Wq_shp = gptq_full(W_shp, H_X_tp, K, order=np.arange(n),
                       tile=TILE, bits_per_tile=K_alloc)
    inv_pr = np.argsort(perm_rows)
    inv_pc = np.argsort(perm_cols)
    Wq_sh = Wq_shp[inv_pr][:, inv_pc]
    Wq_s = U_rot @ Wq_sh @ V_rot.T
    W_hat = S_G_inv @ Wq_s @ S_X_inv
    return W_hat


# ─── Real weight loading ──────────────────────────────────────────────────────

def load_real_weights(slice_size=128):
    """Load real Qwen3.8-27B BF16 weights, extract 128×128 slices."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in data.keys():
        W = data[key].astype(np.float64)
        m, n = W.shape
        tensors[key] = W[:min(slice_size, m), :min(slice_size, n)]
    return tensors


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_held_out(W, W_hat, H_G_test, H_X_test):
    """Compute held-out HWE."""
    E = W - W_hat
    return hessian_weighted_error(E, H_G_test, H_X_test)


def eval_insample(W, W_hat, H_G_train, H_X_train):
    """Compute in-sample HWE."""
    E = W - W_hat
    return hessian_weighted_error(E, H_G_train, H_X_train)


def pct_improvement(baseline_err, method_err):
    if baseline_err > 0:
        return (baseline_err - method_err) / baseline_err * 100.0
    return 0.0


# ─── Statistical analysis ────────────────────────────────────────────────────

def compute_stats(values):
    """Compute mean, std, 95% CI, min, max for a list of values."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    # 95% CI using t-distribution
    if n > 1 and std > 0:
        ci_half = scipy_stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)
    else:
        ci_half = 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": n,
    }


def paired_ttest(a, b):
    """Paired t-test: returns (t_stat, p_value)."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    diff = a_arr - b_arr
    n = len(diff)
    if n < 2:
        return 0.0, 1.0
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-15:
        return 0.0, 1.0
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    p_value = 2 * scipy_stats.t.sf(np.abs(t_stat), df=n-1)
    return float(t_stat), float(p_value)


def win_rate(method_vals, baseline_vals):
    """Fraction of seeds where method < baseline (lower HWE = better)."""
    method_arr = np.array(method_vals)
    baseline_arr = np.array(baseline_vals)
    wins = np.sum(method_arr < baseline_arr)
    return float(wins) / len(method_arr)


# ─── Main experiment ──────────────────────────────────────────────────────────

METHODS = [
    ("RTN",              method_rtn,                 False),
    ("BiIP_Hadamard",    method_biip_hadamard,       True),
    ("DP_Alloc",         method_dp_allocation,       False),
    ("Rot_DP_Alloc",     method_rotation_dp_alloc,    True),
    ("Rot_FullGPTQ",     method_rotation_full_gptq,  True),
    ("Rot_BlockDiagGPTQ", method_rotation_blockdiag_gptq, True),
    ("UnifiedStack",     method_unified_stack,       True),
]
# Methods that use shared rotation (for common-random-numbers ablation)
ROTATION_METHODS = {m[0] for m in METHODS if m[2]}


def main():
    t_start = time.time()
    print("=" * 90)
    print("R25 — Multi-Seed Statistical Robustness Validation")
    print("=" * 90)
    print(f"  Seeds: {N_SEEDS} (base={SEED_BASE})")
    print(f"  Tensors: {TENSOR_NAMES}")
    print(f"  K values: {K_VALUES}")
    print(f"  Methods: {[m[0] for m in METHODS]}")
    print(f"  Calibration: {N_CALIB} samples, {int(N_CALIB*TRAIN_FRAC)} train / {N_CALIB - int(N_CALIB*TRAIN_FRAC)} test")
    print(f"  Total experiments: {N_SEEDS} × {len(METHODS)} × {len(TENSOR_NAMES)} × {len(K_VALUES)} = "
          f"{N_SEEDS * len(METHODS) * len(TENSOR_NAMES) * len(K_VALUES)}")

    # Load weights
    print("\nLoading real Qwen3.8-27B weights...")
    all_tensors = load_real_weights(SLICE)
    print(f"  Loaded: {list(all_tensors.keys())}")

    # Storage: results[seed][tensor][K][method] = {held_out_hwe, in_sample_hwe, improvement_pct}
    all_results = {}
    raw_hwe = {m[0]: {tname: {K: [] for K in K_VALUES} for tname in TENSOR_NAMES}
               for m in METHODS}
    raw_hwe_insample = {m[0]: {tname: {K: [] for K in K_VALUES} for tname in TENSOR_NAMES}
                        for m in METHODS}
    raw_improvement = {m[0]: {tname: {K: [] for K in K_VALUES} for tname in TENSOR_NAMES}
                       for m in METHODS if m[0] != "RTN"}

    for seed_idx in range(N_SEEDS):
        seed = SEED_BASE + seed_idx
        print(f"\n{'─' * 90}")
        print(f"Seed {seed_idx + 1}/{N_SEEDS} (seed={seed})")
        print(f"{'─' * 90}")

        seed_results = {}

        for tname in TENSOR_NAMES:
            W = all_tensors[tname]
            m, n = W.shape
            print(f"\n  Tensor: {tname} ({m}×{n})")

            # Generate calibration with this seed
            X_full = gen_calibration(n, N_CALIB, seed)
            n_train = int(N_CALIB * TRAIN_FRAC)
            rng_split = np.random.default_rng(seed)
            perm = rng_split.permutation(N_CALIB)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]
            X_train = X_full[:, train_idx]
            X_test = X_full[:, test_idx]

            H_G_train, H_X_train = compute_hessians(W, X_train)
            H_G_test, H_X_test = compute_hessians(W, X_test)

            tensor_results = {}

            for K in K_VALUES:
                print(f"    K={K}:", end=" ")
                rng_method = np.random.default_rng(seed * 100 + K)
                K_results = {}

                # Precompute shared rotations for common-random-numbers ablation
                rng_rot = np.random.default_rng(seed * 1000 + K * 10)
                U_shared, _ = signed_random_hadamard(m, rng_rot)
                V_shared, _ = signed_random_hadamard(n, rng_rot)

                method_hwes = {}
                method_hwes_insample = {}
                for method_name, method_fn, uses_rotation in METHODS:
                    try:
                        if uses_rotation:
                            W_hat = method_fn(W, H_X_train, H_G_train, K, rng_method,
                                             U_rot=U_shared, V_rot=V_shared)
                        else:
                            W_hat = method_fn(W, H_X_train, H_G_train, K, rng_method)
                        hwe_out = eval_held_out(W, W_hat, H_G_test, H_X_test)
                        hwe_in = eval_insample(W, W_hat, H_G_train, H_X_train)
                    except Exception as e:
                        hwe_out = float('inf')
                        hwe_in = float('inf')
                        print(f"\n      WARNING: {method_name} failed: {e}")

                    method_hwes[method_name] = hwe_out
                    method_hwes_insample[method_name] = hwe_in
                    raw_hwe[method_name][tname][K].append(hwe_out)
                    raw_hwe_insample[method_name][tname][K].append(hwe_in)

                rtn_hwe = method_hwes["RTN"]

                for method_name, _, _ in METHODS:
                    imp = pct_improvement(rtn_hwe, method_hwes[method_name])
                    K_results[method_name] = {
                        "held_out_hwe": method_hwes[method_name],
                        "in_sample_hwe": method_hwes_insample[method_name],
                        "improvement_pct": imp,
                    }
                    if method_name != "RTN":
                        raw_improvement[method_name][tname][K].append(imp)

                # Print summary for this K
                parts = []
                for method_name, _, _ in METHODS:
                    if method_name == "RTN":
                        parts.append(f"RTN={method_hwes[method_name]:.4e}")
                    else:
                        imp = K_results[method_name]["improvement_pct"]
                        parts.append(f"{method_name}:{imp:+.1f}%")
                print(" | ".join(parts))

                tensor_results[K] = K_results

            seed_results[tname] = tensor_results

        all_results[seed] = seed_results

    # ─── Statistical Analysis ──────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("STATISTICAL ANALYSIS")
    print(f"{'═' * 90}")

    stats_results = {}
    method_names = [m[0] for m in METHODS]
    non_rtn_methods = [m for m in method_names if m != "RTN"]

    for tname in TENSOR_NAMES:
        print(f"\n{'─' * 90}")
        print(f"Tensor: {tname}")
        print(f"{'─' * 90}")
        stats_results[tname] = {}

        for K in K_VALUES:
            print(f"\n  K={K}:")
            cell_stats = {}

            # Per-method statistics
            for method_name in method_names:
                hwe_vals = raw_hwe[method_name][tname][K]
                s = compute_stats(hwe_vals)
                cell_stats[method_name] = {"hwe_stats": s}
                print(f"    {method_name:20s}: HWE mean={s['mean']:.4e} ± {s['std']:.4e} "
                      f"95%CI=[{s['ci95_low']:.4e}, {s['ci95_high']:.4e}] "
                      f"min={s['min']:.4e} max={s['max']:.4e}")

            # Improvement statistics (vs RTN)
            print(f"\n    Improvement vs RTN:")
            rtn_vals = raw_hwe["RTN"][tname][K]
            for method_name in non_rtn_methods:
                imp_vals = raw_improvement[method_name][tname][K]
                s = compute_stats(imp_vals)
                wr = win_rate(raw_hwe[method_name][tname][K], rtn_vals)
                t_stat, p_val = paired_ttest(rtn_vals, raw_hwe[method_name][tname][K])

                cell_stats[method_name]["improvement_stats"] = s
                cell_stats[method_name]["win_rate"] = wr
                cell_stats[method_name]["pvalue_vs_rtn"] = p_val
                cell_stats[method_name]["tstat_vs_rtn"] = t_stat

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"      {method_name:20s}: {s['mean']:+.2f}% ± {s['std']:.2f}% "
                      f"95%CI=[{s['ci95_low']:+.2f}%, {s['ci95_high']:+.2f}%] "
                      f"min={s['min']:+.2f}% max={s['max']:+.2f}% "
                      f"win={wr:.0%} p={p_val:.4e} {sig}")

            # Pairwise comparisons between methods
            print(f"\n    Pairwise t-tests (method A vs method B):")
            pairwise = {}
            for i, m_a in enumerate(non_rtn_methods):
                for m_b in non_rtn_methods[i+1:]:
                    t_stat, p_val = paired_ttest(
                        raw_hwe[m_a][tname][K],
                        raw_hwe[m_b][tname][K]
                    )
                    pairwise[f"{m_a}_vs_{m_b}"] = {
                        "t_stat": t_stat, "p_value": p_val
                    }
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    a_mean = np.mean(raw_hwe[m_a][tname][K])
                    b_mean = np.mean(raw_hwe[m_b][tname][K])
                    better = m_a if a_mean < b_mean else m_b
                    print(f"      {m_a:20s} vs {m_b:20s}: p={p_val:.4e} {sig} → {better} better")

            cell_stats["pairwise"] = pairwise

            # Ranking stability
            print(f"\n    Ranking stability:")
            rankings = []
            for seed_idx in range(N_SEEDS):
                hwes = {m: raw_hwe[m][tname][K][seed_idx] for m in method_names}
                ranked = sorted(method_names, key=lambda x: hwes[x])
                rankings.append(tuple(ranked))

            # Count unique rankings
            from collections import Counter
            ranking_counts = Counter(rankings)
            most_common_rank = ranking_counts.most_common(1)[0]
            stability = most_common_rank[1] / N_SEEDS

            cell_stats["ranking_stability"] = {
                "most_common_ranking": list(most_common_rank[0]),
                "count": most_common_rank[1],
                "stability_fraction": stability,
                "n_unique_rankings": len(ranking_counts),
            }

            print(f"      Most common ranking: {' > '.join(most_common_rank[0])}")
            print(f"      Occurs {most_common_rank[1]}/{N_SEEDS} ({stability:.0%})")
            print(f"      Unique rankings: {len(ranking_counts)}")
            if len(ranking_counts) > 1:
                for rank, count in ranking_counts.most_common():
                    if count < most_common_rank[1]:
                        print(f"        Alt ranking ({count}x): {' > '.join(rank)}")

            # Reversal analysis: any seed where method is WORSE than RTN?
            print(f"\n    Reversal analysis (method worse than RTN):")
            reversals = {}
            for method_name in non_rtn_methods:
                rev_seeds = []
                for seed_idx in range(N_SEEDS):
                    if raw_hwe[method_name][tname][K][seed_idx] > raw_hwe["RTN"][tname][K][seed_idx]:
                        rev_seeds.append(SEED_BASE + seed_idx)
                reversals[method_name] = rev_seeds
                if rev_seeds:
                    print(f"      {method_name}: {len(rev_seeds)} reversals at seeds {rev_seeds}")
                else:
                    print(f"      {method_name}: NO reversals (always beats RTN)")

            cell_stats["reversals"] = reversals

            # Worst-case seed for each method
            print(f"\n    Worst-case seed (lowest improvement):")
            worst_seeds = {}
            for method_name in non_rtn_methods:
                imp_vals = raw_improvement[method_name][tname][K]
                worst_idx = int(np.argmin(imp_vals))
                worst_seeds[method_name] = {
                    "seed": SEED_BASE + worst_idx,
                    "improvement_pct": imp_vals[worst_idx],
                }
                print(f"      {method_name}: seed={SEED_BASE + worst_idx}, improvement={imp_vals[worst_idx]:+.2f}%")

            cell_stats["worst_case_seed"] = worst_seeds

            stats_results[tname][K] = cell_stats

    # ─── Stack bifurcation analysis ───────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("STACK BIFURCATION ANALYSIS (Path A vs Path B, both rotated)")
    print(f"{'═' * 90}")
    print("Path A: Rotation + DP allocation (no GPTQ) = Rot_DP_Alloc")
    print("Path B: Rotation + Full GPTQ (no DP allocation) = Rot_FullGPTQ")
    print("Both use same shared U/V (common random numbers)")
    print()

    bifurcation = {}
    for tname in TENSOR_NAMES:
        for K in K_VALUES:
            # Proper ablation: both rotated, same U/V, one does DP alloc, other does GPTQ
            rot_gptq_vals = raw_hwe["Rot_FullGPTQ"][tname][K]
            rot_dp_vals = raw_hwe["Rot_DP_Alloc"][tname][K]
            rot_gptq_wins = sum(1 for a, b in zip(rot_gptq_vals, rot_dp_vals) if a < b)

            blockdiag_vals = raw_hwe["Rot_BlockDiagGPTQ"][tname][K]
            unified_vals = raw_hwe["UnifiedStack"][tname][K]
            unified_wins = sum(1 for a, b in zip(unified_vals, blockdiag_vals) if a < b)

            key = f"{tname}_K{K}"
            bifurcation[key] = {
                "rot_gptq_beats_rot_dp": rot_gptq_wins,
                "unified_beats_blockdiag": unified_wins,
            }
            print(f"  {key}: Rot_GPTQ > Rot_DP_Alloc in {rot_gptq_wins}/20 seeds | "
                  f"Unified > BlockDiag in {unified_wins}/20 seeds")

    # ─── Overall summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("OVERALL SUMMARY (seed-clustered, df=19)")
    print(f"{'═' * 90}")
    print("Each seed contributes ONE equal-weight average over 12 tensor×K cells.")
    print("CI uses t-distribution (df=19). p-value: one-sided one-sample t-test (H0: mean improvement = 0).")
    print()

    overall = {}
    for method_name in non_rtn_methods:
        # Per-seed average improvement across all 12 cells
        seed_avgs = []
        for seed_idx in range(N_SEEDS):
            cell_imps = []
            for tname in TENSOR_NAMES:
                for K in K_VALUES:
                    cell_imps.append(raw_improvement[method_name][tname][K][seed_idx])
            seed_avgs.append(np.mean(cell_imps))

        s = compute_stats(seed_avgs)  # n=20, df=19

        # One-sample t-test: is mean improvement significantly > 0?
        # (seed-level percentage improvements, df=19)
        seed_arr = np.array(seed_avgs)
        n_seeds = len(seed_arr)
        mean_imp = np.mean(seed_arr)
        std_imp = np.std(seed_arr, ddof=1)
        if std_imp > 1e-15:
            t_stat = mean_imp / (std_imp / np.sqrt(n_seeds))
            p_val = float(scipy_stats.t.sf(np.abs(t_stat), df=n_seeds - 1))  # one-sided
        else:
            t_stat = 0.0
            p_val = 1.0

        # Win rate: fraction of (seed, tensor, K) where method < RTN
        all_wins = 0
        total = 0
        for tname in TENSOR_NAMES:
            for K in K_VALUES:
                for seed_idx in range(N_SEEDS):
                    total += 1
                    if raw_hwe[method_name][tname][K][seed_idx] < raw_hwe["RTN"][tname][K][seed_idx]:
                        all_wins += 1

        overall[method_name] = {
            "improvement_mean": s["mean"],
            "improvement_std": s["std"],
            "improvement_ci95": [s["ci95_low"], s["ci95_high"]],
            "improvement_min": s["min"],
            "improvement_max": s["max"],
            "win_rate": all_wins / total,
            "total_comparisons": total,
            "pvalue_vs_rtn": p_val,
            "tstat_vs_rtn": t_stat,
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {method_name:20s}: {s['mean']:+.2f}% ± {s['std']:.2f}% "
              f"95%CI=[{s['ci95_low']:+.2f}%, {s['ci95_high']:+.2f}%] "
              f"min={s['min']:+.2f}% max={s['max']:+.2f}% "
              f"win={all_wins}/{total} ({all_wins/total:.0%}) p={p_val:.4e} {sig}")

    # ─── Save results ─────────────────────────────────────────────────────────
    method_names = [m[0] for m in METHODS]
    output = {
        "experiment": "R25-MultiSeed",
        "config": {
            "n_seeds": N_SEEDS,
            "seed_base": SEED_BASE,
            "tensor_names": TENSOR_NAMES,
            "k_values": K_VALUES,
            "n_calib": N_CALIB,
            "train_frac": TRAIN_FRAC,
            "tile_size": TILE,
            "slice_size": SLICE,
            "damping": DAMPING,
            "block_size": BLOCK_SIZE,
            "methods": method_names,
            "shared_rotations": True,
        },
        "raw_hwe": {m: {t: {k: v for k, v in d.items()} for t, d in md.items()}
                    for m, md in raw_hwe.items()},
        "raw_improvement": {m: {t: {k: v for k, v in d.items()} for t, d in md.items()}
                            for m, md in raw_improvement.items()},
        "statistics": stats_results,
        "bifurcation": bifurcation,
        "overall": overall,
        "total_experiments": N_SEEDS * len(METHODS) * len(TENSOR_NAMES) * len(K_VALUES),
        "elapsed_seconds": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
