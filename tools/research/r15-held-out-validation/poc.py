#!/usr/bin/env python3
"""
R15 — Held-Out Validation Framework for Wave 1 Quantization Components
=====================================================================

Most Wave 1 results use in-sample evaluation: the same calibration data is used
to (a) select transforms (BiIP scales, Hadamard signs, allocation, ordering)
and (b) score the Hessian-weighted error.  This script builds a proper
train/test split protocol and re-tests the five top components.

Protocol
--------
1.  Generate N calibration samples X (n_in x N).
2.  Random 80/20 split → X_train, X_test.
3.  H_X_train, H_G_train from X_train; H_X_test, H_G_test from X_test.
4.  For each method:
    a.  Fit transforms using *train* Hessians only.
    b.  Apply transforms + quantize + inverse → W_hat  (fixed once fitted).
    c.  In-sample  error = tr(H_G_train · E · H_X_train · E^T)
    d.  Held-out  error = tr(H_G_test  · E · H_X_test  · E^T)
5.  % improvement over RTN baseline computed with the *same* evaluation Hessian.
6.  Generalization gap = in_sample_improvement − held_out_improvement.
7.  Repeat for ≥5 random splits; report mean ± std.

Components re-tested
--------------------
  R3  BiIP + Hadamard rotation
  R1  DP-refined tile allocation
  R7  Act-order GPTQ
  R9  6-step alternating optimizer (full stack proxy)
  R4  Hadamard + p99 permutation

All arms use the SAME per-tile (16×16) uniform quantizer and matched byte budget.
Correct Cholesky: U = chol(inv(H+λI)).T  (upper triangular, U^T U = Hinv).
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "receipts", "research",
    "r15-held-out-validation-results.json",
)

# ─── Configuration ────────────────────────────────────────────────────────────
TILE       = 16
SLICE      = 128
K_VALUES   = [3, 4, 5, 6]
N_CALIB    = 512          # total calibration samples
N_SPLITS   = 7            # random train/test splits (≥5 required)
TRAIN_FRAC = 0.80
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]

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
    """Per-tile uniform quantization.  bits_or_alloc: int or (n_tm, n_tn) array."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            ti, tj = i // tile, j // tile
            k = int(bits_or_alloc[ti, tj]) if isinstance(bits_or_alloc, np.ndarray) else int(bits_or_alloc)
            r1, c1 = min(i + tile, m), min(j + tile, n)
            Wq[i:r1, j:c1] = quantize_tile(W[i:r1, j:c1], k)
    return Wq


# ─── Metrics ──────────────────────────────────────────────────────────────────

def hessian_weighted_error(E, H_G, H_X):
    """tr(H_G · E · H_X · E^T).  E is (m, n), H_G (m,m), H_X (n,n)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))


# ─── Calibration generation ───────────────────────────────────────────────────

def gen_calibration(n_in, n_samples, seed):
    """Synthetic activations: Gaussian + outlier channels (same recipe as Wave 1)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples)) * 0.1
    n_outliers = max(1, n_in // 20)
    outlier_rows = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_rows, :] *= 10.0
    return X


def compute_hessians(W, X):
    """H_X = X X^T / N,  H_G = Y Y^T / N  with Y = W X."""
    N = X.shape[1]
    H_X = (X @ X.T / N).astype(np.float64)
    Y = W @ X
    H_G = (Y @ Y.T / N).astype(np.float64)
    # Normalize so mean diagonal = 1 (relative structure only, prevents overflow)
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
    # Symmetrize for numerical stability
    Hinv = (Hinv + Hinv.T) / 2
    try:
        U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
    except np.linalg.LinAlgError:
        # Fallback: eigendecomposition-based Cholesky
        eigvals, eigvecs = np.linalg.eigh(Hinv)
        eigvals = np.maximum(eigvals, 1e-12)  # clamp to positive
        L_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
        U = L_sqrt.T  # upper triangular
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


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1: R3 — BiIP + Hadamard rotation
# ═══════════════════════════════════════════════════════════════════════════════

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11).
    S_X = diag(H_X_jj / ||W_:,j||^2)^{1/4}, S_G = diag(H_G_ii / ||W_i,:||^2)^{1/4}.
    W' = S_G @ W @ S_X."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    S_G = np.diag(sg_diag)
    W_transformed = S_G @ W @ S_X
    return S_G, S_X, W_transformed


def r3_biip_hadamard(W, H_X_train, H_G_train, K, rng):
    """Fit BiIP scales (from train Hessians) + signed Hadamard (random).
    Returns W_hat in original space."""
    # 1. BiIP scaling (data-dependent: uses H_X_train, H_G_train)
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)

    # 2. Signed random Hadamard both sides (random, NOT data-dependent)
    U, _ = signed_random_hadamard(W.shape[0], rng)
    V, _ = signed_random_hadamard(W.shape[1], rng)

    # 3. Transform: W' = U @ S_G @ W @ S_X @ V  (row=U*S_G, col=S_X*V)
    W_t = U @ W_s @ V

    # 4. Quantize per-tile
    Wq_t = quantize_tiles(W_t, K, TILE)

    # 5. Inverse: W_hat = S_G^{-1} @ U^T @ Wq_t @ V^T @ S_X^{-1}
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    W_hat = S_G_inv @ U.T @ Wq_t @ V.T @ S_X_inv
    return W_hat


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2: R1 — DP-refined tile allocation
# ═══════════════════════════════════════════════════════════════════════════════

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    """Actual Hessian-weighted distortion of one tile at K=k."""
    Wq = quantize_tile(W_tile, k)
    E = W_tile - Wq
    D = np.trace(H_G_sub @ E @ H_X_sub @ E.T)
    return max(D, 0.0)


def compute_tile_distortions(W, H_G, H_X, tile, k_range):
    """Compute distortion table D[ti, tj, k] for all tiles and K values."""
    m, n = W.shape
    n_tm, n_tn = (m + tile - 1) // tile, (n + tile - 1) // tile
    dists = np.zeros((n_tm, n_tn, len(k_range)))
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            W_tile = W[r0:r1, c0:c1]
            H_G_sub = H_G[r0:r1, r0:r1]
            H_X_sub = H_X[c0:c1, c0:c1]
            for ki, k in enumerate(k_range):
                dists[ti, tj, ki] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
    return dists


def tile_k_dp_allocate(distortions, n_tm, n_tn, tile_elements, budget_bits, k_range):
    """Multiple-choice knapsack DP for tile K allocation."""
    k_list = list(k_range)
    n_k = len(k_list)
    # dp[b] = min distortion for budget b
    INF = np.inf
    # Track allocation
    dp = np.full(budget_bits + 1, INF)
    dp_alloc = [[None] * (budget_bits + 1) for _ in range(n_tm * n_tn + 1)]
    dp[0] = 0.0
    dp_alloc[0][0] = []

    tile_idx = 0
    for ti in range(n_tm):
        for tj in range(n_tn):
            new_dp = np.full(budget_bits + 1, INF)
            new_alloc = [None] * (budget_bits + 1)
            for b in range(budget_bits + 1):
                if dp[b] == INF:
                    continue
                for ki, k in enumerate(k_list):
                    cost = k * tile_elements
                    nb = b + cost
                    if nb > budget_bits:
                        continue
                    val = dp[b] + distortions[ti, tj, ki]
                    if val < new_dp[nb]:
                        new_dp[nb] = val
                        new_alloc[nb] = (dp_alloc[tile_idx][b] or []) + [(ti, tj, k)]
            dp = new_dp
            dp_alloc[tile_idx + 1] = new_alloc
            tile_idx += 1

    # Find min distortion within budget
    best_b = np.argmin(dp)
    alloc_list = dp_alloc[n_tm * n_tn][best_b] or []

    K_alloc = np.full((n_tm, n_tn), k_list[0], dtype=int)
    for ti, tj, k in alloc_list:
        K_alloc[ti, tj] = k
    return K_alloc


def local_search_refine(W, K_flat, H_G, H_X, tile=TILE, budget_k=None, max_iters=100):
    """Full-objective local search: try single-bit transfers between tiles.
    O(n_tiles) per iteration: compute marginal costs, pick best donor/recipient pair."""
    m, n = W.shape
    n_tm, n_tn = (m + tile - 1) // tile, (n + tile - 1) // tile
    K = K_flat.copy()
    k_min, k_max = 3, 6

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
        return sum(tile_dists.get((ti, tj, K_arr[ti, tj]), 0.0)
                   for ti in range(n_tm) for tj in range(n_tn))

    current_hwe = total_hwe_fast(K)
    for _ in range(max_iters):
        # For each tile: cost of decreasing K (donor) and benefit of increasing K (recipient)
        best_swap = None
        best_delta = -1e-15
        for ti in range(n_tm):
            for tj in range(n_tn):
                cur_k = K[ti, tj]
                # Cost of donating a bit (K -> K-1)
                if cur_k > k_min:
                    donor_cost = tile_dists[(ti, tj, cur_k - 1)] - tile_dists[(ti, tj, cur_k)]
                    for ti2 in range(n_tm):
                        for tj2 in range(n_tn):
                            if ti == ti2 and tj == tj2:
                                continue
                            cur_k2 = K[ti2, tj2]
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
    """DP-refined tile allocation (fit on train Hessians).
    Returns W_hat."""
    m, n = W.shape
    n_tm, n_tn = (m + tile - 1) // tile, (n + tile - 1) // tile
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W, H_G_train, H_X_train, tile, k_range)
    budget = K * m * n  # total bits = avg_k * elements
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
    # Local search refinement
    K_alloc, _ = local_search_refine(W, K_alloc, H_G_train, H_X_train, tile, max_iters=200)
    Wq = quantize_tiles(W, K_alloc, tile)
    return Wq


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3: R7 — Act-order GPTQ (per-tile quantizer)
# ═══════════════════════════════════════════════════════════════════════════════

def gptq_per_tile_ordered(W, H_X, K, order, tile=TILE, alpha=1.0, damping=0.01,
                          bits_per_tile=None):
    """GPTQ with per-tile (16×16) quantizer and configurable column ordering.
    Uses correct Cholesky: U = chol(inv(H+λI)).T.
    order: permutation of columns (act-order = descending diag(H_X)).
    bits_per_tile: optional (n_tm, n_tn) array of per-tile K values.

    Codebooks are frozen per 16×16 tile from the ORIGINAL W (before any GPTQ
    update), ensuring exactly one codebook per physical tile — matching RTN.
    """
    m, n = W.shape
    H_perm = H_X[np.ix_(order, order)]
    U = inv_cholesky(H_perm, damping)

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    # Pre-compute frozen tile codebooks from ORIGINAL W
    tile_cb = {}  # (ti, tj) -> (lo, step, k)
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

    def quantize_col(col_data, col_idx):
        """Quantize one column using frozen tile codebooks."""
        q = np.zeros_like(col_data)
        tj = col_idx // tile
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

    W_work = W.copy().astype(np.float64)
    Wq = np.zeros_like(W_work)

    for idx in range(n):
        q = order[idx]
        Wq[:, q] = quantize_col(W_work[:, q], q)
        e_q = W_work[:, q] - Wq[:, q]

        if idx < n - 1:
            remaining = order[idx + 1:]
            u_ii = U[idx, idx]
            if abs(u_ii) > 1e-15:
                update = alpha * np.outer(e_q, U[idx, idx + 1:] / u_ii)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)

    return Wq


def r7_act_order_gptq(W, H_X_train, K):
    """Act-order GPTQ: descending diag(H_X) ordering, fit on train H_X.
    Returns W_hat."""
    order = np.argsort(np.diag(H_X_train))[::-1]  # descending diag(H_X)
    Wq = gptq_per_tile_ordered(W, H_X_train, K, order, tile=TILE, alpha=1.0, damping=0.01)
    return Wq


def r7_rtn_baseline(W, K):
    """Plain RTN (no GPTQ, no ordering) — same per-tile quantizer."""
    return quantize_tiles(W, K, TILE)


def r7_ltr_gptq(W, H_X_train, K):
    """Left-to-right GPTQ (no act-order) — control for ordering effect."""
    order = np.arange(W.shape[1])  # natural order
    Wq = gptq_per_tile_ordered(W, H_X_train, K, order, tile=TILE, alpha=1.0, damping=0.01)
    return Wq


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 4: R9 — 6-step alternating optimizer (full stack proxy)
# ═══════════════════════════════════════════════════════════════════════════════

def osborne_equilibrate(W, H_G, H_X, n_iters=10, clip_min=0.1, clip_max=10.0):
    """Osborne/Sinkhorn diagonal equilibration (log-domain, from R9 Wave 1).
    Equalizes H_G-weighted row norms and H_X-weighted column norms.
    Returns (d_G, d_X) diagonal scales of length m and n."""
    m, n = W.shape
    d_G = np.ones(m)
    d_X = np.ones(n)

    for _ in range(n_iters):
        # Row balance: equalize H_G-weighted row norms
        row_norms = np.sqrt(np.sum(W ** 2, axis=1))
        h_G_diag = np.sqrt(np.abs(np.diag(H_G)) + 1e-15)
        row_scale = np.clip(row_norms * h_G_diag, 1e-10, None)
        log_mean = np.mean(np.log(row_scale))
        d_G = np.clip(d_G * np.exp(-(0.5 * (np.log(row_scale) - log_mean))),
                      clip_min, clip_max)

        W_s = d_G[:, None] * W
        # Column balance: equalize H_X-weighted column norms
        col_norms = np.sqrt(np.sum(W_s ** 2, axis=0))
        h_X_diag = np.sqrt(np.abs(np.diag(H_X)) + 1e-15)
        col_scale = np.clip(col_norms * h_X_diag, 1e-10, None)
        log_mean_c = np.mean(np.log(col_scale))
        d_X = np.clip(np.exp(-(0.5 * (np.log(col_scale) - log_mean_c))),
                      clip_min, clip_max)

    return d_G, d_X


def balanced_partition(W, H_G, H_X, tile, axis="both"):
    """Sort-and-group partition for tile packing."""
    m, n = W.shape
    if axis in ("both", "rows"):
        row_keys = np.linalg.norm(W, axis=1)
        perm_rows = np.argsort(row_keys)
    else:
        perm_rows = np.arange(m)
    if axis in ("both", "cols"):
        col_keys = np.linalg.norm(W, axis=0)
        perm_cols = np.argsort(col_keys)
    else:
        perm_cols = np.arange(n)
    return perm_rows, perm_cols


def hadamard_orbit_sample(W, H_G, H_X, n_samples, rng, K, tile):
    """Sample N signed Hadamards, pick the one with lowest HWE after quantization."""
    m, n = W.shape
    best_obj = np.inf
    U_best = np.eye(m)
    V_best = np.eye(n)
    for _ in range(n_samples):
        U, _ = signed_random_hadamard(m, rng)
        V, _ = signed_random_hadamard(n, rng)
        W_t = U.T @ W @ V
        Wq = quantize_tiles(W_t, K, tile)
        E = W_t - Wq
        obj = hessian_weighted_error(U @ E @ V.T, H_G, H_X)
        if obj < best_obj:
            best_obj = obj
            U_best, V_best = U, V
    return U_best, V_best, best_obj


def givens_refine(W, H_G, H_X, U, V, n_iters, rng, K, tile):
    """Givens rotation refinement with golden-section search."""
    m, n = W.shape

    def eval_obj(U_cur, V_cur):
        W_t = U_cur.T @ W @ V_cur
        Wq = quantize_tiles(W_t, K, tile)
        E = W_t - Wq
        return hessian_weighted_error(U_cur @ E @ V_cur.T, H_G, H_X)

    best_obj = eval_obj(U, V)
    gr = (np.sqrt(5) - 1) / 2

    for _ in range(n_iters):
        # Pick a random pair in output dim
        i, j = rng.choice(m, 2, replace=False)
        def f_out(theta):
            G = np.eye(m)
            G[i, i] = np.cos(theta); G[i, j] = -np.sin(theta)
            G[j, i] = np.sin(theta); G[j, j] = np.cos(theta)
            return eval_obj(G @ U, V)
        # Golden section
        a, b = 0.0, np.pi / 2
        for _ in range(15):
            c = b - gr * (b - a); d = a + gr * (b - a)
            if f_out(c) < f_out(d): b = d
            else: a = c
        theta = (a + b) / 2
        G = np.eye(m)
        G[i, i] = np.cos(theta); G[i, j] = -np.sin(theta)
        G[j, i] = np.sin(theta); G[j, j] = np.cos(theta)
        obj = f_out(theta)
        if obj < best_obj - 1e-15:
            U = G @ U; best_obj = obj

        # Pick a random pair in input dim
        i, j = rng.choice(n, 2, replace=False)
        def f_in(theta):
            G = np.eye(n)
            G[i, i] = np.cos(theta); G[i, j] = -np.sin(theta)
            G[j, i] = np.sin(theta); G[j, j] = np.cos(theta)
            return eval_obj(U, V @ G)
        a, b = 0.0, np.pi / 2
        for _ in range(15):
            c = b - gr * (b - a); d = a + gr * (b - a)
            if f_in(c) < f_in(d): b = d
            else: a = c
        theta = (a + b) / 2
        G = np.eye(n)
        G[i, i] = np.cos(theta); G[i, j] = -np.sin(theta)
        G[j, i] = np.sin(theta); G[j, j] = np.cos(theta)
        obj = f_in(theta)
        if obj < best_obj - 1e-15:
            V = V @ G; best_obj = obj

    return U, V, best_obj
def gptq_correction_per_tile(W, X, K, tile, damping=0.01, alpha=0.0,
                              bits_per_tile=None):
    """GPTQ correction with per-tile (16×16) quantizer.
    Uses correct Cholesky: U = chol(inv(H+λI)).T.
    Codebooks frozen from original W (one per 16×16 tile, matching RTN).
    X is the calibration in transformed space (n, N).
    bits_per_tile: optional (n_tm, n_tn) array of per-tile K values.

    Note: alpha=0 (pure GPTQ, no GPTAQ P-matrix). The P-matrix requires
    separate Xt (FP-flow) which is not available in this framework.
    """
    m, n = W.shape
    H = X @ X.T
    L = inv_cholesky(H, damping)

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    # Pre-compute frozen tile codebooks from ORIGINAL W
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

    def quantize_col(col_data, col_idx):
        q = np.zeros_like(col_data)
        tj = col_idx // tile
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

    Ww = W.copy().astype(np.float64)
    Q = np.zeros_like(Ww)

    # Process columns in natural order (left-to-right) for the correction step
    for c in range(n):
        Q[:, c] = quantize_col(Ww[:, c], c)
        e = Ww[:, c] - Q[:, c]
        if c < n - 1:
            l_ii = L[c, c]
            if abs(l_ii) > 1e-15:
                Ww[:, c+1:] -= np.outer(e / l_ii, L[c, c+1:])

    return Q

@dataclass
class R9Config:
    tile_size: int = 16
    block_size: int = 16
    damping: float = 0.01
    n_hadamard_samples: int = 8
    n_givens_iters: int = 20
    n_osborne_iters: int = 10
    min_k: int = 3
    max_k: int = 7
    avg_k: float = 5.0
    gptaq_alpha: float = 0.25
    max_outer_iters: int = 5
    seed: int = 42
    scale_clip_min: float = 0.1
    scale_clip_max: float = 10.0
    tol: float = 1e-12


def r9_alternating_optimizer(W, X_train, H_G_train, H_X_train, K, cfg):
    """6-step alternating optimizer (equilibrate → partition → rotate → allocate → quantize → correct).
    Fit entirely on train data.  Returns W_hat in original space."""
    m, n = W.shape
    tile = cfg.tile_size
    rng = np.random.default_rng(cfg.seed)
    bits = K

    D_G = np.ones(m); D_X = np.ones(n)
    perm_rows = np.arange(m); perm_cols = np.arange(n)
    U_rot = np.eye(m); V_rot = np.eye(n)
    K_alloc = np.full((m // tile, n // tile), bits)
    use_correction = False

    def evaluate(dG, dX, U, V, K_cur, pr, pc, do_corr):
        W_s = dG[:, None] * W * dX[None, :]
        W_sp = W_s[pr][:, pc]
        W_spr = U.T @ W_sp @ V

        X_corr = None
        if do_corr:
            X_s = X_train / dX[:, None]
            X_corr = V.T @ X_s[pc]

        if do_corr and X_corr is not None:
            kpt = K_cur if isinstance(K_cur, np.ndarray) else None
            Wq_spr = gptq_correction_per_tile(W_spr, X_corr,
                                              bits if kpt is None else bits,
                                              tile, cfg.damping, 0.0,
                                              bits_per_tile=kpt)
        else:
            Wq_spr = quantize_tiles(W_spr, K_cur if isinstance(K_cur, np.ndarray) else bits, tile)

        E_spr = W_spr - Wq_spr
        E_sp = U @ E_spr @ V.T
        inv_pr = np.argsort(pr); inv_pc = np.argsort(pc)
        E_s = E_sp[inv_pr][:, inv_pc]
        Wq = (W_s - E_s) / dG[:, None] / dX[None, :]
        E = W - Wq
        herr = hessian_weighted_error(E, H_G_train, H_X_train)
        return Wq, herr

    Wq_best, herr_best = evaluate(D_G, D_X, U_rot, V_rot, K_alloc, perm_rows, perm_cols, False)

    for outer in range(cfg.max_outer_iters):
        improved = False

        # Step 1: Equilibrate
        dG, dX = osborne_equilibrate(W, H_G_train, H_X_train, cfg.n_osborne_iters,
                                      cfg.scale_clip_min, cfg.scale_clip_max)
        Wq, herr = evaluate(dG, dX, U_rot, V_rot, K_alloc, perm_rows, perm_cols, use_correction)
        if herr < herr_best - cfg.tol:
            D_G, D_X = dG, dX; herr_best = herr; Wq_best = Wq; improved = True

        # Step 2: Partition
        W_s = D_G[:, None] * W * D_X[None, :]
        pr, pc = balanced_partition(W_s, H_G_train, H_X_train, tile)
        Wq, herr = evaluate(D_G, D_X, U_rot, V_rot, K_alloc, pr, pc, use_correction)
        if herr < herr_best - cfg.tol:
            perm_rows, perm_cols = pr, pc; herr_best = herr; Wq_best = Wq; improved = True

        # Step 3: Rotate (Hadamard orbit + Givens)
        W_s = D_G[:, None] * W * D_X[None, :]
        W_sp = W_s[perm_rows][:, perm_cols]
        H_G_p = H_G_train[perm_rows][:, perm_rows]
        H_X_p = H_X_train[perm_cols][:, perm_cols]
        U_new, V_new, _ = hadamard_orbit_sample(W_sp, H_G_p, H_X_p, cfg.n_hadamard_samples, rng, bits, tile)
        U_new, V_new, _ = givens_refine(W_sp, H_G_p, H_X_p, U_new, V_new, cfg.n_givens_iters, rng, bits, tile)
        Wq, herr = evaluate(D_G, D_X, U_new, V_new, K_alloc, perm_rows, perm_cols, use_correction)
        if herr < herr_best - cfg.tol:
            U_rot, V_rot = U_new, V_new; herr_best = herr; Wq_best = Wq; improved = True

        # Step 4: Allocate (DP)
        W_s = D_G[:, None] * W * D_X[None, :]
        W_sp = W_s[perm_rows][:, perm_cols]
        W_spr = U_rot.T @ W_sp @ V_rot
        H_G_p = H_G_train[perm_rows][:, perm_rows]
        H_X_p = H_X_train[perm_cols][:, perm_cols]
        H_G_pr = U_rot.T @ H_G_p @ U_rot
        H_X_pr = V_rot.T @ H_X_p @ V_rot
        n_tm, n_tn = (m + tile - 1) // tile, (n + tile - 1) // tile
        k_range = range(cfg.min_k, cfg.max_k + 1)
        dists = compute_tile_distortions(W_spr, H_G_pr, H_X_pr, tile, k_range)
        budget = bits * m * n
        K_new = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
        Wq, herr = evaluate(D_G, D_X, U_rot, V_rot, K_new, perm_rows, perm_cols, use_correction)
        if herr < herr_best - cfg.tol:
            K_alloc = K_new; herr_best = herr; Wq_best = Wq; improved = True

        # Step 5: Correction (GPTAQ)
        Wq, herr = evaluate(D_G, D_X, U_rot, V_rot, K_alloc, perm_rows, perm_cols, True)
        if herr < herr_best - cfg.tol:
            use_correction = True; herr_best = herr; Wq_best = Wq; improved = True

        if not improved:
            break

    return Wq_best


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 5: R4 — Hadamard + p99 permutation
# ═══════════════════════════════════════════════════════════════════════════════

def r4_hadamard_p99_perm(W, H_X_train, H_G_train, K, rng):
    """Hadamard first, then p99-scale permutation (weight-based, NOT data-dependent).
    But Hadamard signs are random.  Permutation is from W statistics only.
    Returns W_hat."""
    m, n = W.shape

    # 1. Hadamard both sides (random signs)
    U, _ = signed_random_hadamard(m, rng)
    V, _ = signed_random_hadamard(n, rng)
    W_h = U.T @ W @ V

    # 2. p99-scale permutation on the Hadamard-transformed W
    # Sort columns by p99 of |W_h| column values
    p99 = np.percentile(np.abs(W_h), 99, axis=0)
    perm_cols = np.argsort(p99)
    # Sort rows by p99 of |W_h| row values
    p99_rows = np.percentile(np.abs(W_h), 99, axis=1)
    perm_rows = np.argsort(p99_rows)

    W_hp = W_h[perm_rows][:, perm_cols]

    # 3. Quantize per-tile
    Wq_hp = quantize_tiles(W_hp, K, TILE)

    # 4. Inverse permutation
    inv_pr = np.argsort(perm_rows)
    inv_pc = np.argsort(perm_cols)
    Wq_h = Wq_hp[inv_pr][:, inv_pc]

    # 5. Inverse Hadamard
    W_hat = U @ Wq_h @ V.T
    return W_hat


def r4_hadamard_only(W, H_X_train, H_G_train, K, rng):
    """Hadamard only (no permutation) — control for R4."""
    m, n = W.shape
    U, _ = signed_random_hadamard(m, rng)
    V, _ = signed_random_hadamard(n, rng)
    W_h = U.T @ W @ V
    Wq_h = quantize_tiles(W_h, K, TILE)
    W_hat = U @ Wq_h @ V.T
    return W_hat


# ═══════════════════════════════════════════════════════════════════════════════
# FULL STACK: R3+R4+R1+R7 (combined pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def full_stack(W, X_train, H_X_train, H_G_train, K, rng):
    """Combined pipeline: BiIP → Hadamard → p99 perm → DP alloc → act-order GPTQ.
    All transforms fit on train data.  Returns W_hat."""
    m, n = W.shape

    # 1. BiIP scaling
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)

    # 2. Hadamard
    U, _ = signed_random_hadamard(m, rng)
    V, _ = signed_random_hadamard(n, rng)
    W_sh = U.T @ W_s @ V

    # 3. p99 permutation
    p99_cols = np.percentile(np.abs(W_sh), 99, axis=0)
    perm_cols = np.argsort(p99_cols)
    p99_rows = np.percentile(np.abs(W_sh), 99, axis=1)
    perm_rows = np.argsort(p99_rows)
    W_shp = W_sh[perm_rows][:, perm_cols]

    # 4. DP allocation on transformed space
    # Transform Hessians
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = V.T @ S_X_inv @ H_X_train @ S_X_inv @ V
    H_G_t = U.T @ S_G_inv @ H_G_train @ S_G_inv @ U
    H_X_tp = H_X_t[perm_cols][:, perm_cols]
    H_G_tp = H_G_t[perm_rows][:, perm_rows]

    n_tm, n_tn = (m + TILE - 1) // TILE, (n + TILE - 1) // TILE
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W_shp, H_G_tp, H_X_tp, TILE, k_range)
    budget = K * m * n
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, TILE * TILE, budget, k_range)
    K_alloc, _ = local_search_refine(W_shp, K_alloc, H_G_tp, H_X_tp, TILE, max_iters=100)

    # 5. Act-order GPTQ on transformed space (with allocated K per tile)
    order = np.argsort(np.diag(H_X_tp))[::-1]
    Wq_shp = gptq_per_tile_ordered(W_shp, H_X_tp, K, order, tile=TILE,
                                    alpha=1.0, damping=0.01,
                                    bits_per_tile=K_alloc)

    # 6. Inverse all transforms
    inv_pr = np.argsort(perm_rows)
    inv_pc = np.argsort(perm_cols)
    Wq_sh = Wq_shp[inv_pr][:, inv_pc]
    Wq_s = U @ Wq_sh @ V.T
    W_hat = S_G_inv @ Wq_s @ S_X_inv
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


def run_single_evaluation(W, W_hat, H_G_train, H_X_train, H_G_test, H_X_test):
    """Compute in-sample and held-out HWE for a given W_hat."""
    E = W - W_hat
    return {
        "in_sample_hwe": hessian_weighted_error(E, H_G_train, H_X_train),
        "held_out_hwe": hessian_weighted_error(E, H_G_test, H_X_test),
        "weight_mse": weight_mse(E),
    }


def pct_improvement(baseline_err, method_err):
    """% improvement of method over baseline."""
    if baseline_err > 0:
        return (baseline_err - method_err) / baseline_err * 100.0
    return 0.0


def main():
    t_start = time.time()
    print("=" * 90)
    print("R15 — Held-Out Validation Framework for Wave 1 Quantization Components")
    print("=" * 90)
    print(f"  Tensors: {TENSOR_NAMES}")
    print(f"  K values: {K_VALUES}")
    print(f"  Splits: {N_SPLITS} random 80/20 splits")
    print(f"  Calibration: {N_CALIB} samples total, {int(N_CALIB*TRAIN_FRAC)} train / {N_CALIB - int(N_CALIB*TRAIN_FRAC)} test")
    print(f"  Quantizer: per-tile ({TILE}×{TILE}) uniform (matched for all arms)")
    print(f"  Cholesky: correct convention U = chol(inv(H+λI)).T")

    # Load weights
    print("\nLoading real Qwen3.8-27B weights...")
    all_tensors = load_real_weights(SLICE)
    print(f"  Available: {list(all_tensors.keys())}")

    # Split seeds: different random splits
    split_seeds = [42 + i * 100 for i in range(N_SPLITS)]

    # R9 config
    cfg9 = R9Config(avg_k=5.0, seed=42)

    # Results: results[component][tensor_name][K] = list of per-split dicts
    components = ["RTN", "R3_BiIP_Hadamard", "R1_DP_Allocation",
                  "R7_ActOrder_GPTQ", "R7_LTR_GPTQ",
                  "R9_Alternating_Optimizer",
                  "R4_Hadamard_p99_Perm", "R4_Hadamard_Only",
                  "Full_Stack"]
    results = {c: {t: {k: [] for k in K_VALUES} for t in TENSOR_NAMES} for c in components}

    for split_idx, split_seed in enumerate(split_seeds):
        print(f"\n{'─'*90}")
        print(f"Split {split_idx+1}/{N_SPLITS} (seed={split_seed})")
        print(f"{'─'*90}")

        for tensor_name in TENSOR_NAMES:
            if tensor_name not in all_tensors:
                continue
            W = all_tensors[tensor_name]
            m, n = W.shape
            print(f"\n  Tensor: {tensor_name} ({m}×{n})")

            # Generate calibration and split
            X_full = gen_calibration(n, N_CALIB, split_seed)
            n_train = int(N_CALIB * TRAIN_FRAC)
            # Shuffle indices for the split
            rng_split = np.random.default_rng(split_seed)
            perm = rng_split.permutation(N_CALIB)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

            X_train = X_full[:, train_idx]
            X_test = X_full[:, test_idx]

            # Compute train and test Hessians
            H_G_train, H_X_train = compute_hessians(W, X_train)
            H_G_test, H_X_test = compute_hessians(W, X_test)

            for K in K_VALUES:
                print(f"    K={K}:", end=" ")
                rng_method = np.random.default_rng(split_seed + K)

                # RTN baseline
                Wq_rtn = quantize_tiles(W, K, TILE)
                res_rtn = run_single_evaluation(W, Wq_rtn, H_G_train, H_X_train, H_G_test, H_X_test)
                results["RTN"][tensor_name][K].append(res_rtn)

                # R3: BiIP + Hadamard
                Wq_r3 = r3_biip_hadamard(W, H_X_train, H_G_train, K, np.random.default_rng(split_seed + K + 1))
                res_r3 = run_single_evaluation(W, Wq_r3, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R3_BiIP_Hadamard"][tensor_name][K].append(res_r3)

                # R1: DP allocation
                Wq_r1 = r1_dp_allocation(W, H_X_train, H_G_train, K, TILE)
                res_r1 = run_single_evaluation(W, Wq_r1, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R1_DP_Allocation"][tensor_name][K].append(res_r1)

                # R7: Act-order GPTQ
                Wq_r7 = r7_act_order_gptq(W, H_X_train, K)
                res_r7 = run_single_evaluation(W, Wq_r7, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R7_ActOrder_GPTQ"][tensor_name][K].append(res_r7)

                # R7 control: LTR GPTQ (natural order)
                Wq_r7c = r7_ltr_gptq(W, H_X_train, K)
                res_r7c = run_single_evaluation(W, Wq_r7c, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R7_LTR_GPTQ"][tensor_name][K].append(res_r7c)

                # R4: Hadamard + p99 permutation
                Wq_r4 = r4_hadamard_p99_perm(W, H_X_train, H_G_train, K, np.random.default_rng(split_seed + K + 2))
                res_r4 = run_single_evaluation(W, Wq_r4, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R4_Hadamard_p99_Perm"][tensor_name][K].append(res_r4)

                # R4 control: Hadamard only
                Wq_r4c = r4_hadamard_only(W, H_X_train, H_G_train, K, np.random.default_rng(split_seed + K + 2))
                res_r4c = run_single_evaluation(W, Wq_r4c, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R4_Hadamard_Only"][tensor_name][K].append(res_r4c)

                # R9: Alternating optimizer (only for K=5 to save time, and K=3,4,6)
                t0 = time.time()
                cfg_k = R9Config(avg_k=float(K), seed=split_seed + K, min_k=max(3, K-1), max_k=min(7, K+2))
                Wq_r9 = r9_alternating_optimizer(W, X_train, H_G_train, H_X_train, K, cfg_k)
                res_r9 = run_single_evaluation(W, Wq_r9, H_G_train, H_X_train, H_G_test, H_X_test)
                results["R9_Alternating_Optimizer"][tensor_name][K].append(res_r9)
                t_r9 = time.time() - t0

                # Full Stack
                t0 = time.time()
                Wq_fs = full_stack(W, X_train, H_X_train, H_G_train, K, np.random.default_rng(split_seed + K + 3))
                res_fs = run_single_evaluation(W, Wq_fs, H_G_train, H_X_train, H_G_test, H_X_test)
                results["Full_Stack"][tensor_name][K].append(res_fs)
                t_fs = time.time() - t0

                # Compute improvements
                rtn_in = res_rtn["in_sample_hwe"]
                rtn_out = res_rtn["held_out_hwe"]
                r3_in_imp = pct_improvement(rtn_in, res_r3["in_sample_hwe"])
                r3_out_imp = pct_improvement(rtn_out, res_r3["held_out_hwe"])
                r1_in_imp = pct_improvement(rtn_in, res_r1["in_sample_hwe"])
                r1_out_imp = pct_improvement(rtn_out, res_r1["held_out_hwe"])
                r7_in_imp = pct_improvement(rtn_in, res_r7["in_sample_hwe"])
                r7_out_imp = pct_improvement(rtn_out, res_r7["held_out_hwe"])
                r9_in_imp = pct_improvement(rtn_in, res_r9["in_sample_hwe"])
                r9_out_imp = pct_improvement(rtn_out, res_r9["held_out_hwe"])
                r4_in_imp = pct_improvement(rtn_in, res_r4["in_sample_hwe"])
                r4_out_imp = pct_improvement(rtn_out, res_r4["held_out_hwe"])
                fs_in_imp = pct_improvement(rtn_in, res_fs["in_sample_hwe"])
                fs_out_imp = pct_improvement(rtn_out, res_fs["held_out_hwe"])

                print(f"RTN in={rtn_in:.2e} out={rtn_out:.2e} | "
                      f"R3 in={r3_in_imp:+.1f}% out={r3_out_imp:+.1f}% | "
                      f"R1 in={r1_in_imp:+.1f}% out={r1_out_imp:+.1f}% | "
                      f"R7 in={r7_in_imp:+.1f}% out={r7_out_imp:+.1f}% | "
                      f"R9 in={r9_in_imp:+.1f}% out={r9_out_imp:+.1f}% ({t_r9:.1f}s) | "
                      f"R4 in={r4_in_imp:+.1f}% out={r4_out_imp:+.1f}% | "
                      f"FS in={fs_in_imp:+.1f}% out={fs_out_imp:+.1f}% ({t_fs:.1f}s)")

    # ─── Aggregate and report ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("AGGREGATE RESULTS: Mean ± Std across splits")
    print(f"{'='*90}")

    # For each component × tensor × K: compute mean and std of in-sample and held-out improvements
    summary = {}
    for comp in components:
        summary[comp] = {}
        for tname in TENSOR_NAMES:
            summary[comp][tname] = {}
            for K in K_VALUES:
                splits_data = results[comp][tname][K]
                if not splits_data:
                    continue
                rtn_splits = results["RTN"][tname][K]
                in_imps = [pct_improvement(r["in_sample_hwe"], s["in_sample_hwe"])
                           for r, s in zip(rtn_splits, splits_data)]
                out_imps = [pct_improvement(r["held_out_hwe"], s["held_out_hwe"])
                            for r, s in zip(rtn_splits, splits_data)]
                gaps = [i - o for i, o in zip(in_imps, out_imps)]

                summary[comp][tname][K] = {
                    "in_sample_imp_mean": float(np.mean(in_imps)),
                    "in_sample_imp_std": float(np.std(in_imps)),
                    "held_out_imp_mean": float(np.mean(out_imps)),
                    "held_out_imp_std": float(np.std(out_imps)),
                    "gen_gap_mean": float(np.mean(gaps)),
                    "gen_gap_std": float(np.std(gaps)),
                    "n_splits": len(splits_data),
                    "in_sample_hwe_mean": float(np.mean([s["in_sample_hwe"] for s in splits_data])),
                    "held_out_hwe_mean": float(np.mean([s["held_out_hwe"] for s in splits_data])),
                }

    # Print table
    print(f"\n{'Component':<28} {'Tensor':<10} {'K':>3} {'In-sample':>12} {'Held-out':>12} {'Gen Gap':>10} {'Overfit?':>8}")
    print(f"{'-'*85}")
    for comp in components:
        for tname in TENSOR_NAMES:
            for K in K_VALUES:
                s = summary[comp].get(tname, {}).get(K)
                if s is None:
                    continue
                overfit = "YES" if s["gen_gap_mean"] > 2.0 else ("marginal" if s["gen_gap_mean"] > 0.5 else "no")
                print(f"{comp:<28} {tname:<10} {K:>3} "
                      f"{s['in_sample_imp_mean']:>+8.1f}±{s['in_sample_imp_std']:>3.1f} "
                      f"{s['held_out_imp_mean']:>+8.1f}±{s['held_out_imp_std']:>3.1f} "
                      f"{s['gen_gap_mean']:>+7.2f}±{s['gen_gap_std']:>2.2f} "
                      f"{overfit:>8}")

    # Macro summary across tensors
    print(f"\n{'='*90}")
    print("MACRO SUMMARY: Mean across all tensors")
    print(f"{'='*90}")
    print(f"\n{'Component':<28} {'K':>3} {'In-sample':>14} {'Held-out':>14} {'Gen Gap':>12} {'Overfit?':>8}")
    print(f"{'-'*82}")
    macro = {}
    for comp in components:
        macro[comp] = {}
        for K in K_VALUES:
            in_vals, out_vals, gaps = [], [], []
            for tname in TENSOR_NAMES:
                s = summary[comp].get(tname, {}).get(K)
                if s is None:
                    continue
                in_vals.append(s["in_sample_imp_mean"])
                out_vals.append(s["held_out_imp_mean"])
                gaps.append(s["gen_gap_mean"])
            if not in_vals:
                continue
            macro[comp][K] = {
                "in_sample_mean": float(np.mean(in_vals)),
                "in_sample_std": float(np.std(in_vals)),
                "held_out_mean": float(np.mean(out_vals)),
                "held_out_std": float(np.std(out_vals)),
                "gen_gap_mean": float(np.mean(gaps)),
                "gen_gap_std": float(np.std(gaps)),
            }
            m = macro[comp][K]
            overfit = "YES" if m["gen_gap_mean"] > 2.0 else ("marginal" if m["gen_gap_mean"] > 0.5 else "no")
            print(f"{comp:<28} {K:>3} "
                  f"{m['in_sample_mean']:>+9.1f}±{m['in_sample_std']:>3.1f} "
                  f"{m['held_out_mean']:>+9.1f}±{m['held_out_std']:>3.1f} "
                  f"{m['gen_gap_mean']:>+8.2f}±{m['gen_gap_std']:>2.2f} "
                  f"{overfit:>8}")

    # Ranking stability
    print(f"\n{'='*90}")
    print("RANKING STABILITY: Component ranking per split (by held-out improvement, macro across tensors)")
    print(f"{'='*90}")
    eval_components = ["R3_BiIP_Hadamard", "R1_DP_Allocation", "R7_ActOrder_GPTQ",
                       "R9_Alternating_Optimizer", "R4_Hadamard_p99_Perm", "Full_Stack"]
    for K in K_VALUES:
        print(f"\n  K={K}:")
        split_rankings = []
        for split_idx in range(N_SPLITS):
            # Compute macro held-out improvement for each component in this split
            comp_scores = {}
            for comp in eval_components:
                vals = []
                for tname in TENSOR_NAMES:
                    splits_data = results[comp][tname][K]
                    rtn_splits = results["RTN"][tname][K]
                    if split_idx < len(splits_data):
                        imp = pct_improvement(rtn_splits[split_idx]["held_out_hwe"],
                                              splits_data[split_idx]["held_out_hwe"])
                        vals.append(imp)
                comp_scores[comp] = np.mean(vals) if vals else -999
            ranking = sorted(comp_scores.items(), key=lambda x: -x[1])
            split_rankings.append([r[0] for r in ranking])
            print(f"    Split {split_idx+1}: " + " > ".join(f"{r[0]}({r[1]:+.1f}%)" for r in ranking))

        # Check ranking stability: how often is each rank position the same?
        if split_rankings:
            from collections import Counter
            n_pos = len(eval_components)
            position_stability = []
            for pos in range(n_pos):
                names_at_pos = [r[pos] for r in split_rankings if pos < len(r)]
                most_common = Counter(names_at_pos).most_common(1)[0]
                stability = most_common[1] / len(names_at_pos)
                position_stability.append((pos + 1, most_common[0], stability))
            print(f"    Position stability: " + ", ".join(
                f"#{p}: {name} ({stab:.0%})" for p, name, stab in position_stability))

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "config": {
            "n_splits": N_SPLITS,
            "train_frac": TRAIN_FRAC,
            "n_calib": N_CALIB,
            "k_values": K_VALUES,
            "tensor_names": TENSOR_NAMES,
            "tile_size": TILE,
            "slice_size": SLICE,
            "split_seeds": split_seeds,
        },
        "summary": summary,
        "macro": macro,
        "raw_results": {c: {t: {k: [r for r in v] for k, v in d.items()} for t, d in d2.items()} for c, d2 in results.items()},
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
