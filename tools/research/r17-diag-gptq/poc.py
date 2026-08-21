#!/usr/bin/env python3
"""
R17 — Diagonal-Covariance GPTQ That Generalizes
================================================

R15 found that full-covariance GPTQ overfits held-out (gap +57.5pp) while
diagonal-stat transforms (BiIP, Hadamard, DP allocation) generalize perfectly.
This script tests whether diagonal-only / block-diagonal / threshold GPTQ
generalizes better while still providing error correction.

GPTQ Variants
-------------
1. Full GPTQ:        W[:, rem] -= e_q * U[q, rem] / U[q, q]
                      where U = chol(inv(H+λI)).T (full Cholesky of inverse)
2. Diagonal GPTQ:    W[:, rem] -= e_q * H[q, rem] / (H[q, q] + λ)
                      Uses H (forward Hessian) row + diagonal preconditioner.
                      Equivalent to one Jacobi iteration (gradient step).
3. Block-diag GPTQ:  Block-diagonal U (16×16 blocks). Full Cholesky within
                      blocks, zero between blocks. Captures within-tile
                      correlations, ignores cross-tile.
4. Threshold GPTQ:   Full Cholesky but zero out small off-diagonal U terms
                      (|U[i,j]| < τ·|U[i,i]|). Sparse approximation.

Held-Out Protocol
-----------------
- 7 random 80/20 splits of calibration data
- Fit GPTQ on train Hessians, evaluate on both train and test Hessians
- Generalization gap = in_sample_improvement − held_out_improvement

Composition Tests
-----------------
- Each GPTQ variant tested standalone and composed with:
  (a) Rotation (BiIP + Hadamard)
  (b) Rotation + DP Allocation
- Rotation should make off-diagonal terms smaller → diagonal approx better

All arms use the SAME per-tile (16×16) uniform quantizer and matched byte budget.
Correct Cholesky: U = chol(inv(H+λI)).T  (upper triangular, U^T U = Hinv).
"""

import json
import os
import time
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "receipts", "research",
    "r17-diag-gptq-results.json",
)

# ─── Configuration ────────────────────────────────────────────────────────────
TILE       = 16
SLICE      = 128
K_VALUES   = [3, 4, 5, 6]
N_CALIB    = 512
N_SPLITS   = 5
TRAIN_FRAC = 0.80
DAMPING    = 0.01
BLOCK_SIZE = 16          # block-diagonal GPTQ block size
THRESHOLD  = 0.1         # threshold GPTQ: zero U[i,j] if |U[i,j]| < τ*|U[i,i]|
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
SLICE_NAMES = ["first"]

# ─── Quantizer: per-tile (16×16) uniform, MATCHED for all arms ─────────────────

def quantize_tile(w, k):
    """Per-tile uniform quantizer. k bits → 2^k levels."""
    nl = 2 ** k
    lo = float(w.min())
    hi = float(w.max())
    step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
    if step == 0.0:
        return w.copy()
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_tiles(W, bits_or_alloc, tile=TILE):
    """Per-tile uniform quantization. bits_or_alloc: int or (n_tm, n_tn) array."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    Wq = np.zeros_like(W, dtype=np.float64)
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            if isinstance(bits_or_alloc, np.ndarray):
                k = max(3, min(7, int(bits_or_alloc[ti, tj])))
            else:
                k = bits_or_alloc
            Wq[r0:r1, c0:c1] = quantize_tile(W[r0:r1, c0:c1], k)
    return Wq

# ─── Metrics ──────────────────────────────────────────────────────────────────

def hessian_weighted_error(E, H_G, H_X):
    """tr(H_G · E · H_X · E^T).  E is (m, n), H_G (m,m), H_X (n,n)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))

# ─── Calibration generation ───────────────────────────────────────────────────

def gen_calibration(n_in, n_samples, seed):
    """Synthetic activations: Gaussian + outlier channels (same recipe as R15).
    NO cross-channel correlation → off-diagonal Hessian = pure sampling noise.
    This is the worst case for GPTQ generalization."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples)) * 0.1
    n_outliers = max(1, n_in // 20)
    outlier_rows = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_rows, :] *= 10.0
    return X


def gen_calibration_correlated(n_in, n_samples, seed):
    """Synthetic activations WITH cross-channel correlation (R11 recipe).
    Off-diagonal Hessian has real structure → GPTQ should generalize better."""
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.5, 3.0, size=n_in)
    n_outliers = max(1, n_in // 20)
    outlier_idx = rng.choice(n_in, n_outliers, replace=False)
    scales[outlier_idx] *= rng.uniform(5.0, 15.0, size=n_outliers)
    corr = rng.standard_normal((n_in, n_in))
    corr = corr @ corr.T / n_in
    X = rng.standard_normal((n_in, n_samples))
    X = X * scales[:, None]
    X = corr @ X
    return X


def compute_hessians(W, X):
    """H_X = X X^T / N,  H_G = Y Y^T / N  with Y = W X.
    Normalize so mean diagonal = 1 (same as R15)."""
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

# ─── Cholesky (correct convention) ───────────────────────────────────────────

def inv_cholesky(H, damping):
    """Upper triangular U such that U^T U = inv(H + damping*I).
    Correct GPTQ convention: U[c, c:] is non-zero for error propagation.
    Falls back to eigendecomposition if Cholesky fails (non-PD Hinv)."""
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    Hinv = (Hinv + Hinv.T) / 2
    try:
        U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Hinv)
        eigvals = np.maximum(eigvals, 1e-12)
        L_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
        U = L_sqrt.T
    return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)


def block_diag_inv_cholesky(H, block_size, damping):
    """Block-diagonal upper triangular U.
    Full Cholesky within block_size×block_size blocks, zero between blocks.
    U^T U ≈ block-diagonal approximation of inv(H+λI).
    Uses GLOBAL λ (from full H diagonal mean) for all blocks — matched to Full GPTQ."""
    n = H.shape[0]
    U = np.zeros((n, n), dtype=np.float64)
    # Global λ from full H, not per-block
    global_lam = max(damping * np.mean(np.diag(H)), 1e-10)
    for i in range(0, n, block_size):
        j = min(i + block_size, n)
        block = H[i:j, i:j] + global_lam * np.eye(j - i)  # use global λ
        Hinv_block = np.linalg.inv(block)
        Hinv_block = (Hinv_block + Hinv_block.T) / 2
        try:
            U[i:j, i:j] = np.linalg.cholesky(Hinv_block).T
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(Hinv_block)
            eigvals = np.maximum(eigvals, 1e-12)
            L_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            U[i:j, i:j] = L_sqrt.T
    return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)


def threshold_inv_cholesky(H, damping, threshold):
    """Full Cholesky but threshold small off-diagonal terms to zero.
    U[i, j] = 0 if |U[i, j]| < threshold * |U[i, i]| for i != j.
    Creates a sparse Cholesky factor — between diagonal and full."""
    U = inv_cholesky(H, damping)
    n = U.shape[0]
    diag_abs = np.abs(np.diag(U))
    # Create mask: keep diagonal and large off-diagonal terms
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mask[i, i] = True  # always keep diagonal
        for j in range(i + 1, n):
            if abs(U[i, j]) >= threshold * diag_abs[i]:
                mask[i, j] = True
    U_thresh = U * mask
    return U_thresh

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

# ─── BiIP diagonal balancing ──────────────────────────────────────────────────

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11)."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    S_G = np.diag(sg_diag)
    W_transformed = S_G @ W @ S_X
    return S_G, S_X, W_transformed

# ─── DP tile allocation (from R1/R15) ──────────────────────────────────────────

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    """Hessian-weighted distortion of one tile at K=k."""
    Wq_tile = quantize_tile(W_tile, k)
    E_tile = W_tile - Wq_tile
    D = np.trace(H_G_sub @ E_tile @ H_X_sub @ E_tile.T)
    return max(D, 0.0)


def compute_tile_distortions(W, H_G, H_X, tile, k_range):
    """Compute distortion table D[ti, tj, k] for all tiles and K values."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    dists = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            W_tile = W[r0:r1, c0:c1]
            H_G_sub = H_G[r0:r1, r0:r1]
            H_X_sub = H_X[c0:c1, c0:c1]
            for k in k_range:
                dists[(ti, tj, k)] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
    return dists


def tile_k_dp_allocate(distortions, n_tm, n_tn, tile_elements, budget_bits, k_range):
    """Multiple-choice knapsack DP for tile K allocation."""
    K_list = sorted(k_range)
    n_K = len(K_list)
    # DP table: dp[i][b] = min total distortion using first i tiles with b bits
    n_tiles = n_tm * n_tn
    INF = float('inf')
    dp = np.full((n_tiles + 1, budget_bits + 1), INF)
    parent = np.full((n_tiles + 1, budget_bits + 1, 2), -1, dtype=int)  # (tile_idx, k_idx)
    dp[0, 0] = 0.0

    idx = 0
    for ti in range(n_tm):
        for tj in range(n_tn):
            for b in range(budget_bits + 1):
                if dp[idx, b] < INF:
                    for ki, k in enumerate(K_list):
                        bits = k * tile_elements
                        nb = b + bits
                        if nb <= budget_bits:
                            d = distorsions_get(distortions, ti, tj, k)
                            nd = dp[idx, b] + d
                            if nd < dp[idx + 1, nb]:
                                dp[idx + 1, nb] = nd
                                parent[idx + 1, nb] = (b, ki)
            idx += 1

    # Find minimum at exactly budget_bits (or closest)
    best_b = budget_bits
    for b in range(budget_bits, -1, -1):
        if dp[n_tiles, b] < INF:
            best_b = b
            break

    # Backtrack
    K_flat = np.zeros(n_tiles, dtype=int)
    b = best_b
    for i in range(n_tiles, 0, -1):
        prev_b, ki = parent[i, b]
        K_flat[i - 1] = K_list[ki]
        b = prev_b

    return K_flat.reshape(n_tm, n_tn)


def distorsions_get(distortions, ti, tj, k):
    key = (ti, tj, k)
    return distortions.get(key, float('inf'))


def local_search_refine(W, K_flat, H_G, H_X, tile=TILE, budget_k=None, max_iters=100):
    """Full-objective local search: try single-bit transfers between tiles."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    K = K_flat.copy()

    def total_hwe(K_arr):
        Wq = quantize_tiles(W, K_arr.reshape(n_tm, n_tn), tile)
        E = W - Wq
        return hessian_weighted_error(E, H_G, H_X)

    current_hwe = total_hwe(K)
    for _ in range(max_iters):
        improved = False
        for i in range(len(K)):
            for j in range(len(K)):
                if i == j:
                    continue
                if K[i] <= 3:
                    continue
                K_try = K.copy()
                K_try[i] -= 1
                K_try[j] += 1
                if K_try[j] > 7:
                    continue
                hwe_try = total_hwe(K_try)
                if hwe_try < current_hwe - 1e-15:
                    K = K_try
                    current_hwe = hwe_try
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return K.reshape(n_tm, n_tn), current_hwe


def r1_dp_allocation(W, H_X_train, H_G_train, K, tile=TILE):
    """DP-refined tile allocation (fit on train Hessians). Returns Wq."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W, H_G_train, H_X_train, tile, k_range)
    budget = K * m * n
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
    K_alloc, _ = local_search_refine(W, K_alloc.flatten(), H_G_train, H_X_train, tile, budget, 10)
    return quantize_tiles(W, K_alloc, tile)

# ═══════════════════════════════════════════════════════════════════════════════
# GPTQ VARIANTS — Core implementation
# ═══════════════════════════════════════════════════════════════════════════════

def _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile=None):
    """Pre-compute frozen tile codebooks from ORIGINAL W."""
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    tile_cb = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            k = max(3, min(7, int(bits_per_tile[ti, tj]))) if bits_per_tile is not None else K
            td = W[r0:r1, c0:c1]
            nl = 2 ** k
            lo = float(td.min())
            hi = float(td.max())
            step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
            tile_cb[(ti, tj)] = (lo, step, k)
    return tile_cb


def _quantize_col(col_data, col_idx, tile_cb, m, tile):
    """Quantize one column using frozen tile codebooks."""
    q = np.zeros_like(col_data)
    tj = col_idx // tile
    n_tm = (m + tile - 1) // tile
    for ti in range(n_tm):
        r0 = ti * tile
        r1 = min(r0 + tile, m)
        lo, step, k = tile_cb[(ti, tj)]
        if step == 0.0:
            q[r0:r1] = col_data[r0:r1]
        else:
            nl = 2 ** k
            q[r0:r1] = np.clip(np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
    return q


def gptq_full(W, H_X, K, order=None, tile=TILE, damping=DAMPING, alpha=1.0,
              bits_per_tile=None):
    """Full GPTQ with correct Cholesky: U = chol(inv(H+λI)).T.
    Update: W[:, rem] -= α * e_q * U[idx, rem_perm] / U[idx, idx]"""
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
                update = alpha * np.outer(e_q, U[idx, idx + 1:] / u_ii)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)
    return Wq


def gptq_forward_h(W, H_X, K, order=None, tile=TILE, damping=DAMPING,
                  bits_per_tile=None):
    """Forward-H diagonal GPTQ (Jacobi iteration): uses H row + 1/diag(H) step.
    Update: W[:,rem] -= e_q * H[idx,rem] / (H[idx,idx] + λ)
    This is one Jacobi step for H*delta = e. Uses forward Hessian, NOT H^{-1}.
    Direction has negative cosine similarity with true GPTQ direction."""
    m, n = W.shape
    if order is None:
        order = np.arange(n)
    H_perm = H_X[np.ix_(order, order)]
    n_h = H_perm.shape[0]
    lam = max(damping * np.mean(np.diag(H_perm)), 1e-10)
    diag_H = np.diag(H_perm) + lam  # damped diagonal

    tile_cb = _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile)

    W_work = W.copy().astype(np.float64)
    Wq = np.zeros_like(W_work)

    for idx in range(n):
        q = order[idx]
        Wq[:, q] = _quantize_col(W_work[:, q], q, tile_cb, m, tile)
        e_q = W_work[:, q] - Wq[:, q]
        if idx < n - 1:
            remaining = order[idx + 1:]
            h_qq = diag_H[idx]
            if abs(h_qq) > 1e-15:
                update = np.outer(e_q, H_perm[idx, idx + 1:] / h_qq)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)
    return Wq


def gptq_diag_hinv(W, H_X, K, order=None, tile=TILE, damping=DAMPING,
                  bits_per_tile=None):
    """True diagonal H^{-1} GPTQ: uses raw H^{-1} rows (no Cholesky, no Schur complement).
    Update: W[:,rem] -= e_q * Hinv[idx,rem] / Hinv[idx,idx]
    This is the 'no-Cholesky GPTQ' — uses full H^{-1} row but without sequential
    Schur complement updates. Should be close to full GPTQ for the first column
    but diverges for later columns (no Schur complement refinement)."""
    m, n = W.shape
    if order is None:
        order = np.arange(n)
    H_perm = H_X[np.ix_(order, order)]
    n_h = H_perm.shape[0]
    lam = max(damping * np.mean(np.diag(H_perm)), 1e-10)
    Hd = H_perm + lam * np.eye(n_h)
    Hinv = np.linalg.inv(Hd)
    Hinv = (Hinv + Hinv.T) / 2

    tile_cb = _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile)

    W_work = W.copy().astype(np.float64)
    Wq = np.zeros_like(W_work)

    for idx in range(n):
        q = order[idx]
        Wq[:, q] = _quantize_col(W_work[:, q], q, tile_cb, m, tile)
        e_q = W_work[:, q] - Wq[:, q]
        if idx < n - 1:
            remaining = order[idx + 1:]
            hinv_qq = Hinv[idx, idx]
            if abs(hinv_qq) > 1e-15:
                update = np.outer(e_q, Hinv[idx, idx + 1:] / hinv_qq)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)
    return Wq


def gptq_block_diag(W, H_X, K, order=None, tile=TILE, block_size=BLOCK_SIZE,
                    damping=DAMPING, bits_per_tile=None):
    """Block-diagonal GPTQ: full Cholesky within block_size×block_size blocks,
    zero between blocks. Captures within-tile correlations, ignores cross-tile.
    Update uses block-diagonal U."""
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


def gptq_threshold(W, H_X, K, order=None, tile=TILE, threshold=THRESHOLD,
                   damping=DAMPING, bits_per_tile=None):
    """Threshold GPTQ: full Cholesky but zero small off-diagonal U terms.
    U[i, j] = 0 if |U[i, j]| < threshold * |U[i, i]| for i != j.
    Sparse approximation — between diagonal and full."""
    m, n = W.shape
    if order is None:
        order = np.arange(n)
    H_perm = H_X[np.ix_(order, order)]
    U = threshold_inv_cholesky(H_perm, damping, threshold)

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

# ═══════════════════════════════════════════════════════════════════════════════
# ROTATION + GPTQ COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_gptq(W, H_X_train, H_G_train, K, rng, gptq_fn, use_allocation=False):
    """BiIP + Hadamard rotation + GPTQ variant + optional DP allocation.
    Fit on train Hessians. Returns W_hat in original space."""
    # 1. BiIP scaling (data-dependent: uses train Hessians)
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)

    # 2. Signed random Hadamard both sides
    U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    V_rot, _ = signed_random_hadamard(W.shape[1], rng)

    # 3. Transform: W' = U_rot @ S_G @ W @ S_X @ V_rot^T (R11 convention)
    W_t = U_rot @ W_s @ V_rot.T

    # Transform Hessians to rotated space
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = S_X_inv @ H_X_train @ S_X_inv
    H_X_t = V_rot @ H_X_t @ V_rot.T
    H_G_t = S_G_inv @ H_G_train @ S_G_inv
    H_G_t = U_rot @ H_G_t @ U_rot.T

    # 4. Optional DP allocation in rotated space
    bits_per_tile = None
    if use_allocation:
        m, n = W_t.shape
        n_tm = (m + TILE - 1) // TILE
        n_tn = (n + TILE - 1) // TILE
        k_range = range(max(3, K - 1), min(7, K + 2))
        dists = compute_tile_distortions(W_t, H_G_t, H_X_t, TILE, k_range)
        budget = K * m * n
        K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, TILE * TILE, budget, k_range)
        K_alloc, _ = local_search_refine(W_t, K_alloc.flatten(), H_G_t, H_X_t, TILE, budget, 10)
        bits_per_tile = K_alloc

    # 5. GPTQ in rotated space
    Wq_t = gptq_fn(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                   tile=TILE, bits_per_tile=bits_per_tile)

    # 6. Inverse: W = S_G^{-1} @ U^T @ Wq' @ V_rot @ S_X^{-1}
    W_hat = S_G_inv @ U_rot.T @ Wq_t @ V_rot @ S_X_inv
    return W_hat


def rotation_only(W, H_X_train, H_G_train, K, rng):
    """BiIP + Hadamard rotation only (no GPTQ)."""
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot @ W_s @ V_rot.T
    Wq_t = quantize_tiles(W_t, K, TILE)
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    W_hat = S_G_inv @ U_rot.T @ Wq_t @ V_rot @ S_X_inv
    return W_hat


def rotation_alloc_only(W, H_X_train, H_G_train, K, rng):
    """BiIP + Hadamard + DP allocation (no GPTQ)."""
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot @ W_s @ V_rot.T
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = S_X_inv @ H_X_train @ S_X_inv
    H_X_t = V_rot @ H_X_t @ V_rot.T
    H_G_t = S_G_inv @ H_G_train @ S_G_inv
    H_G_t = U_rot @ H_G_t @ U_rot.T
    Wq_t = r1_dp_allocation(W_t, H_X_t, H_G_t, K, TILE)
    W_hat = S_G_inv @ U_rot.T @ Wq_t @ V_rot @ S_X_inv
    return W_hat

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_weights(slice_size=128):
    """Load real Qwen3.8-27B BF16 weights, extract 128×128 slices."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in data.keys():
        W = data[key].astype(np.float64)
        m, n = W.shape
        tensors[key] = W[:min(slice_size, m), :min(slice_size, n)]
    return tensors


def load_real_weights_slices(slice_size=128):
    """Load real weights, extract 3 slices per tensor (first, middle, last)."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in data.keys():
        W = data[key].astype(np.float64)
        m, n = W.shape
        s = min(slice_size, m), min(slice_size, n)
        if n >= 3 * s[1]:
            mid_start = n // 2 - s[1] // 2
            last_start = n - s[1]
            tensors[f"{key}_first"] = W[:s[0], :s[1]]
            tensors[f"{key}_middle"] = W[:s[0], mid_start:mid_start + s[1]]
            tensors[f"{key}_last"] = W[:s[0], last_start:last_start + s[1]]
        elif n >= 2 * s[1]:
            last_start = n - s[1]
            tensors[f"{key}_first"] = W[:s[0], :s[1]]
            tensors[f"{key}_last"] = W[:s[0], last_start:last_start + s[1]]
        else:
            tensors[f"{key}_first"] = W[:s[0], :s[1]]
    return tensors


def run_single_evaluation(W, W_hat, H_G_train, H_X_train, H_G_test, H_X_test):
    """Compute in-sample and held-out HWE for a given W_hat."""
    E = W - W_hat
    return {
        "in_sample_hwe": hessian_weighted_error(E, H_G_train, H_X_train),
        "held_out_hwe": hessian_weighted_error(E, H_G_test, H_X_test),
        "weight_mse": weight_mse(E),
    }


def pct_improvement(baseline_err, method_err):
    if baseline_err > 0:
        return (baseline_err - method_err) / baseline_err * 100.0
    return 0.0


def main():
    t_start = time.time()
    print("=" * 90)
    print("R17 — Diagonal-Covariance GPTQ That Generalizes")
    print("=" * 90)
    print(f"  Tensors: {TENSOR_NAMES}")
    print(f"  Slices: {SLICE_NAMES}")
    print(f"  K values: {K_VALUES}")
    print(f"  Splits: {N_SPLITS} random 80/20 splits")
    print(f"  Calibration: {N_CALIB} samples, {int(N_CALIB*TRAIN_FRAC)} train / {N_CALIB - int(N_CALIB*TRAIN_FRAC)} test")
    print(f"  Quantizer: per-tile ({TILE}×{TILE}) uniform (matched for all arms)")
    print(f"  Cholesky: correct convention U = chol(inv(H+λI)).T")
    print(f"  Block size: {BLOCK_SIZE}, Threshold: {THRESHOLD}")
    print(f"  Damping: {DAMPING}")

    # Load weights — use 3 slices per tensor
    print("\nLoading real Qwen3.8-27B weights (3 slices per tensor)...")
    all_tensors = load_real_weights_slices(SLICE)
    print(f"  Available slices: {list(all_tensors.keys())}")

    # For simplicity, use the base tensor names and append slice
    # We'll test: L0_gate_first, L0_gate_middle, L0_gate_last, etc.
    test_keys = []
    for tname in TENSOR_NAMES:
        for sname in SLICE_NAMES:
            key = f"{tname}_{sname}"
            if key in all_tensors:
                test_keys.append(key)

    split_seeds = [42 + i * 100 for i in range(N_SPLITS)]

    # Define arms
    # Each arm: (name, function, is_gptq_variant)
    # GPTQ arms take H_X for fitting; non-GPTQ arms don't need it
    arms = [
        "RTN",
        "Full_GPTQ",
        "ForwardH_GPTQ",
        "DiagHinv_GPTQ",
        "BlockDiag_GPTQ",
        "Threshold_GPTQ",
        "Rotation_Only",
        "Rotation_Full_GPTQ",
        "Rotation_DiagHinv_GPTQ",
        "Rotation_BlockDiag_GPTQ",
        "Rotation_Threshold_GPTQ",
        "Rotation_Alloc_Only",
        "Rotation_Alloc_Full_GPTQ",
        "Rotation_Alloc_DiagHinv_GPTQ",
        "Rotation_Alloc_BlockDiag_GPTQ",
    ]

    results = {a: {k: {K: [] for K in K_VALUES} for k in test_keys} for a in arms}

    for split_idx, split_seed in enumerate(split_seeds):
        print(f"\n{'─'*90}")
        print(f"Split {split_idx+1}/{N_SPLITS} (seed={split_seed})")
        print(f"{'─'*90}")

        for tkey in test_keys:
            W = all_tensors[tkey]
            m, n = W.shape
            print(f"\n  Tensor: {tkey} ({m}×{n})")

            # Generate calibration and split
            X_full = gen_calibration(n, N_CALIB, split_seed)
            n_train = int(N_CALIB * TRAIN_FRAC)
            rng_split = np.random.default_rng(split_seed)
            perm = rng_split.permutation(N_CALIB)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

            X_train = X_full[:, train_idx]
            X_test = X_full[:, test_idx]

            H_G_train, H_X_train = compute_hessians(W, X_train)
            H_G_test, H_X_test = compute_hessians(W, X_test)

            for K in K_VALUES:
                print(f"    K={K}:", end=" ")
                rng_method = np.random.default_rng(split_seed + K)

                # RTN baseline
                Wq = quantize_tiles(W, K, TILE)
                res = run_single_evaluation(W, Wq, H_G_train, H_X_train, H_G_test, H_X_test)
                results["RTN"][tkey][K].append(res)
                rtn_in = res["in_sample_hwe"]
                rtn_out = res["held_out_hwe"]

                def report(arm_name, Wq_arm):
                    res_arm = run_single_evaluation(W, Wq_arm, H_G_train, H_X_train, H_G_test, H_X_test)
                    results[arm_name][tkey][K].append(res_arm)
                    in_imp = pct_improvement(rtn_in, res_arm["in_sample_hwe"])
                    out_imp = pct_improvement(rtn_out, res_arm["held_out_hwe"])
                    gap = in_imp - out_imp
                    return f"{arm_name}:{out_imp:+.1f}%(g{gap:+.1f})"

                parts = []

                # Full GPTQ
                Wq = gptq_full(W, H_X_train, K, order=np.arange(n))
                parts.append(report("Full_GPTQ", Wq))

                # Forward-H GPTQ (Jacobi — uses H, NOT H^{-1})
                Wq = gptq_forward_h(W, H_X_train, K, order=np.arange(n))
                parts.append(report("ForwardH_GPTQ", Wq))

                # Diagonal H^{-1} GPTQ (raw H^{-1} rows, no Schur complement)
                Wq = gptq_diag_hinv(W, H_X_train, K, order=np.arange(n))
                parts.append(report("DiagHinv_GPTQ", Wq))

                # Block-diagonal GPTQ
                Wq = gptq_block_diag(W, H_X_train, K, order=np.arange(n))
                parts.append(report("BlockDiag_GPTQ", Wq))

                # Threshold GPTQ
                Wq = gptq_threshold(W, H_X_train, K, order=np.arange(n))
                parts.append(report("Threshold_GPTQ", Wq))

                # Rotation only
                Wq = rotation_only(W, H_X_train, H_G_train, K, np.random.default_rng(split_seed + K + 1))
                parts.append(report("Rotation_Only", Wq))

                # Rotation + GPTQ variants
                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_full)
                parts.append(report("Rotation_Full_GPTQ", Wq))

                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_diag_hinv)
                parts.append(report("Rotation_DiagHinv_GPTQ", Wq))

                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_block_diag)
                parts.append(report("Rotation_BlockDiag_GPTQ", Wq))

                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_threshold)
                parts.append(report("Rotation_Threshold_GPTQ", Wq))

                # Rotation + Allocation (no GPTQ)
                Wq = rotation_alloc_only(W, H_X_train, H_G_train, K,
                                         np.random.default_rng(split_seed + K + 1))
                parts.append(report("Rotation_Alloc_Only", Wq))

                # Rotation + Allocation + GPTQ variants
                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_full,
                                   use_allocation=True)
                parts.append(report("Rotation_Alloc_Full_GPTQ", Wq))

                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_diag_hinv,
                                   use_allocation=True)
                parts.append(report("Rotation_Alloc_DiagHinv_GPTQ", Wq))

                Wq = rotation_gptq(W, H_X_train, H_G_train, K,
                                   np.random.default_rng(split_seed + K + 1), gptq_block_diag,
                                   use_allocation=True)
                parts.append(report("Rotation_Alloc_BlockDiag_GPTQ", Wq))

                print(" | ".join(parts))

    # ─── Aggregate and report ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("AGGREGATE RESULTS: Mean ± Std across splits and slices")
    print(f"{'='*90}")

    summary = {}
    for arm in arms:
        summary[arm] = {}
        for K in K_VALUES:
            in_imps, out_imps, gaps = [], [], []
            for tkey in test_keys:
                splits_data = results[arm][tkey][K]
                rtn_splits = results["RTN"][tkey][K]
                if not splits_data:
                    continue
                for r, s in zip(rtn_splits, splits_data):
                    in_imp = pct_improvement(r["in_sample_hwe"], s["in_sample_hwe"])
                    out_imp = pct_improvement(r["held_out_hwe"], s["held_out_hwe"])
                    in_imps.append(in_imp)
                    out_imps.append(out_imp)
                    gaps.append(in_imp - out_imp)
            if not in_imps:
                continue
            summary[arm][K] = {
                "in_sample_imp_mean": float(np.mean(in_imps)),
                "in_sample_imp_std": float(np.std(in_imps)),
                "held_out_imp_mean": float(np.mean(out_imps)),
                "held_out_imp_std": float(np.std(out_imps)),
                "gen_gap_mean": float(np.mean(gaps)),
                "gen_gap_std": float(np.std(gaps)),
                "n_samples": len(in_imps),
            }

    # Print table
    print(f"\n{'Arm':<32} {'K':>3} {'In-sample':>14} {'Held-out':>14} {'Gen Gap':>12} {'Overfit?':>8}")
    print(f"{'-'*86}")
    for arm in arms:
        for K in K_VALUES:
            s = summary[arm].get(K)
            if s is None:
                continue
            overfit = "YES" if s["gen_gap_mean"] > 2.0 else ("marginal" if s["gen_gap_mean"] > 0.5 else "no")
            print(f"{arm:<32} {K:>3} "
                  f"{s['in_sample_imp_mean']:>+9.1f}±{s['in_sample_imp_std']:>3.1f} "
                  f"{s['held_out_imp_mean']:>+9.1f}±{s['held_out_imp_std']:>3.1f} "
                  f"{s['gen_gap_mean']:>+8.2f}±{s['gen_gap_std']:>2.2f} "
                  f"{overfit:>8}")

    # ─── Per-tensor breakdown for key comparisons ──────────────────────────────
    print(f"\n{'='*90}")
    print("PER-TENSOR BREAKDOWN: K=5 (key comparison)")
    print(f"{'='*90}")
    key_arms = ["Full_GPTQ", "ForwardH_GPTQ", "DiagHinv_GPTQ", "BlockDiag_GPTQ", "Threshold_GPTQ",
                "Rotation_Full_GPTQ", "Rotation_DiagHinv_GPTQ",
                "Rotation_BlockDiag_GPTQ", "Rotation_Alloc_Full_GPTQ",
                "Rotation_Alloc_DiagHinv_GPTQ", "Rotation_Alloc_BlockDiag_GPTQ"]
    K_show = 5
    print(f"\n{'Arm':<32} {'Tensor':<20} {'In-sample':>10} {'Held-out':>10} {'Gen Gap':>8}")
    print(f"{'-'*82}")
    for arm in key_arms:
        for tname in TENSOR_NAMES:
            in_vals, out_vals, gap_vals = [], [], []
            for sname in SLICE_NAMES:
                tkey = f"{tname}_{sname}"
                splits_data = results[arm].get(tkey, {}).get(K_show, [])
                rtn_splits = results["RTN"].get(tkey, {}).get(K_show, [])
                if not splits_data or not rtn_splits:
                    continue
                for r, s in zip(rtn_splits, splits_data):
                    in_vals.append(pct_improvement(r["in_sample_hwe"], s["in_sample_hwe"]))
                    out_vals.append(pct_improvement(r["held_out_hwe"], s["held_out_hwe"]))
                    gap_vals.append(in_vals[-1] - out_vals[-1])
            if not in_vals:
                continue
            print(f"{arm:<32} {tname:<20} "
                  f"{np.mean(in_vals):>+8.1f} {np.mean(out_vals):>+8.1f} {np.mean(gap_vals):>+6.2f}")

    # ─── Paired comparison: diagonal vs full GPTQ ──────────────────────────────
    print(f"\n{'='*90}")
    print("PAIRED COMPARISONS (held-out HWE, per split/slice/K)")
    print(f"{'='*90}")

    def paired_test(name_a, name_b, label_a, label_b):
        a_wins, b_wins, total = 0, 0, 0
        for tkey in test_keys:
            for K in K_VALUES:
                a_data = results[name_a].get(tkey, {}).get(K, [])
                b_data = results[name_b].get(tkey, {}).get(K, [])
                rtn_data = results["RTN"].get(tkey, {}).get(K, [])
                for a, b, r in zip(a_data, b_data, rtn_data):
                    a_imp = pct_improvement(r["held_out_hwe"], a["held_out_hwe"])
                    b_imp = pct_improvement(r["held_out_hwe"], b["held_out_hwe"])
                    if a_imp > b_imp:
                        a_wins += 1
                    else:
                        b_wins += 1
                    total += 1
        print(f"  {label_a} wins: {a_wins}/{total} ({100*a_wins/total:.0f}%) | {label_b} wins: {b_wins}/{total} ({100*b_wins/total:.0f}%)")

    paired_test("ForwardH_GPTQ", "Full_GPTQ", "ForwardH", "Full")
    paired_test("DiagHinv_GPTQ", "Full_GPTQ", "DiagHinv", "Full")
    paired_test("BlockDiag_GPTQ", "Full_GPTQ", "BlockDiag", "Full")
    paired_test("Threshold_GPTQ", "Full_GPTQ", "Threshold", "Full")
    print()
    paired_test("Rotation_DiagHinv_GPTQ", "Rotation_Full_GPTQ", "Rot+DiagHinv", "Rot+Full")
    paired_test("Rotation_BlockDiag_GPTQ", "Rotation_Full_GPTQ", "Rot+BlockDiag", "Rot+Full")
    paired_test("Rotation_Threshold_GPTQ", "Rotation_Full_GPTQ", "Rot+Threshold", "Rot+Full")
    print()
    paired_test("Rotation_Only", "Rotation_Full_GPTQ", "Rot_Only", "Rot+Full")
    paired_test("Rotation_Only", "Rotation_BlockDiag_GPTQ", "Rot_Only", "Rot+BlockDiag")
    paired_test("Rotation_Alloc_Only", "Rotation_Alloc_Full_GPTQ", "Rot+Alloc_Only", "Rot+Alloc+Full")
    paired_test("Rotation_Alloc_Only", "Rotation_Alloc_BlockDiag_GPTQ", "Rot+Alloc_Only", "Rot+Alloc+BlockDiag")

    # ─── Ranking stability ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("RANKING STABILITY (by held-out improvement, macro across tensors/slices)")
    print(f"{'='*90}")
    eval_arms = [a for a in arms if a != "RTN"]
    for K in K_VALUES:
        print(f"\n  K={K}:")
        split_rankings = []
        for split_idx in range(N_SPLITS):
            comp_scores = {}
            for arm in eval_arms:
                vals = []
                for tkey in test_keys:
                    splits_data = results[arm].get(tkey, {}).get(K, [])
                    rtn_splits = results["RTN"].get(tkey, {}).get(K, [])
                    if split_idx < len(splits_data) and split_idx < len(rtn_splits):
                        imp = pct_improvement(rtn_splits[split_idx]["held_out_hwe"],
                                              splits_data[split_idx]["held_out_hwe"])
                        vals.append(imp)
                comp_scores[arm] = np.mean(vals) if vals else -999
            ranking = sorted(comp_scores.items(), key=lambda x: -x[1])
            split_rankings.append([r[0] for r in ranking])
            top5 = ranking[:5]
            print(f"    Split {split_idx+1}: " + " > ".join(f"{r[0]}({r[1]:+.1f}%)" for r in top5))

    # ─── Save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "config": {
            "n_splits": N_SPLITS,
            "train_frac": TRAIN_FRAC,
            "n_calib": N_CALIB,
            "k_values": K_VALUES,
            "tensor_names": TENSOR_NAMES,
            "slice_names": SLICE_NAMES,
            "test_keys": test_keys,
            "tile_size": TILE,
            "slice_size": SLICE,
            "block_size": BLOCK_SIZE,
            "threshold": THRESHOLD,
            "damping": DAMPING,
            "split_seeds": split_seeds,
            "calibration": "independent_channels (R15 recipe, worst case for GPTQ)",
        },
        "summary": summary,
        "raw_results": {a: {k: {K: [r for r in v] for K, v in d.items()} for k, d in d2.items()} for a, d2 in results.items()},
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
