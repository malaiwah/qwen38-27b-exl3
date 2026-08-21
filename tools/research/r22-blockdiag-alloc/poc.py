#!/usr/bin/env python3
"""
R22 — Block-Diagonal GPTQ + Allocation Composition (v3)

FIXES from v2 (per reviewer):
1. Use R17's EXACT allocator: tile_k_dp_allocate (bit-budget, k*tile_elements)
   + local_search_refine (full-HWE objective, K up to 7)
2. Remove false budget-bug claim
3. Fix realloc: correction factors use same bits_per_tile as first pass
4. Fix interlayer: compare RotBDGPTQ vs RotBDGPTQ (matched rotation), not vs NoGPTQ
5. Fix cross-flow: measure cross-BLOCK propagation directly (BD U is block-diagonal,
   so cross-block U entries are zero by construction; verify error flow respects this)

Key question: Does block-diag GPTQ compose with allocation where Full GPTQ cannot?
R17 found Rot+Alloc+BDGPTQ = 74.6% vs Rot+Alloc = 76.3% (−1.7pp, alloc wins 56%).
This v3 uses R17's exact allocator to verify whether this holds.
"""

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "receipts", "research", "r22-blockdiag-alloc-results.json"
)
RESULTS_PATH = os.path.normpath(RESULTS_PATH)

# ─── Configuration (matches R17 exactly) ──────────────────────────────────────
TILE       = 16
SLICE      = 128
K_VALUES   = [3, 4, 5, 6]
N_CALIB    = 512
N_SPLITS   = 5
TRAIN_FRAC = 0.80
DAMPING    = 0.01
BLOCK_SIZE = 16
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
SLICE_NAMES = ["first"]

# ─── Quantizer (same as R17) ──────────────────────────────────────────────────

def quantize_tile(w, k):
    nl = 2 ** k
    lo = float(w.min())
    hi = float(w.max())
    step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_tiles(W, bits_or_alloc, tile=TILE):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    Wq = np.zeros_like(W)
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            k = bits_or_alloc if isinstance(bits_or_alloc, int) else bits_or_alloc[ti, tj]
            Wq[r0:r1, c0:c1] = quantize_tile(W[r0:r1, c0:c1], k)
    return Wq

# ─── Metrics ──────────────────────────────────────────────────────────────────

def hessian_weighted_error(E, H_G, H_X):
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))

# ─── Calibration (same as R17) ───────────────────────────────────────────────

def gen_calibration(n_in, n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples)) * 0.1
    n_outliers = max(1, n_in // 20)
    outlier_rows = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_rows, :] *= 10.0
    return X


def compute_hessians(W, X):
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

# ─── Cholesky (same as R17) ──────────────────────────────────────────────────

def inv_cholesky(H, damping):
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    Hinv = (Hinv + Hinv.T) / 2
    try:
        U = np.linalg.cholesky(Hinv).T
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Hinv)
        eigvals = np.maximum(eigvals, 1e-12)
        L_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
        U = L_sqrt.T
    return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)


def block_diag_inv_cholesky(H, block_size, damping):
    n = H.shape[0]
    U = np.zeros((n, n), dtype=np.float64)
    global_lam = max(damping * np.mean(np.diag(H)), 1e-10)
    for i in range(0, n, block_size):
        j = min(i + block_size, n)
        block = H[i:j, i:j] + global_lam * np.eye(j - i)
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

# ─── Hadamard (same as R17) ──────────────────────────────────────────────────

def hadamard_matrix(n):
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return np.diag(signs) @ H, signs

# ─── BiIP (same as R17) ──────────────────────────────────────────────────────

def biip_scaling(W, H_X, H_G):
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    S_G = np.diag(sg_diag)
    W_transformed = S_G @ W @ S_X
    return S_G, S_X, W_transformed

# ─── DP tile allocation (EXACT R17 code) ─────────────────────────────────────

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    Wq_tile = quantize_tile(W_tile, k)
    E_tile = W_tile - Wq_tile
    D = np.trace(H_G_sub @ E_tile @ H_X_sub @ E_tile.T)
    return max(D, 0.0)


def compute_tile_distortions(W, H_G, H_X, tile, k_range):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    dists = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            W_tile = W[r0:r1, c0:c1]
            H_G_sub = H_G[r0:r1, r0:r1]
            H_X_sub = H_X[c0:c1, c0:c1]
            for k in k_range:
                dists[(ti, tj, k)] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
    return dists


def tile_k_dp_allocate(distortions, n_tm, n_tn, tile_elements, budget_bits, k_range):
    """Multiple-choice knapsack DP for tile K allocation.
    Charges bits = k * tile_elements per tile (R17 convention)."""
    K_list = sorted(k_range)
    n_K = len(K_list)
    n_tiles = n_tm * n_tn
    INF = float('inf')
    dp = np.full((n_tiles + 1, budget_bits + 1), INF)
    parent = np.full((n_tiles + 1, budget_bits + 1, 2), -1, dtype=int)
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
                            d = distortions.get((ti, tj, k), INF)
                            nd = dp[idx, b] + d
                            if nd < dp[idx + 1, nb]:
                                dp[idx + 1, nb] = nd
                                parent[idx + 1, nb] = (b, ki)
            idx += 1

    best_b = budget_bits
    for b in range(budget_bits, -1, -1):
        if dp[n_tiles, b] < INF:
            best_b = b
            break

    K_flat = np.zeros(n_tiles, dtype=int)
    b = best_b
    for i in range(n_tiles, 0, -1):
        prev_b, ki = parent[i, b]
        K_flat[i - 1] = K_list[ki]
        b = prev_b

    return K_flat.reshape(n_tm, n_tn)


def local_search_refine(W, K_flat, H_G, H_X, tile=TILE, budget_k=None, max_iters=100):
    """Full-objective local search: try single-bit transfers between tiles.
    EXACT R17 logic: full HWE (including cross-tile terms), K range [3, 7],
    break on first improvement. No additive filter — that prunes valid moves."""
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
    """DP-refined tile allocation (R17 exact). Returns Wq."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W, H_G_train, H_X_train, tile, k_range)
    budget = K * m * n  # bit-level budget (R17 convention)
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
    K_alloc, _ = local_search_refine(W, K_alloc.flatten(), H_G_train, H_X_train, tile, budget, 10)
    return quantize_tiles(W, K_alloc, tile)


def r1_dp_allocation_K(W, H_X_train, H_G_train, K, tile=TILE):
    """DP-refined tile allocation (R17 exact). Returns K_grid."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W, H_G_train, H_X_train, tile, k_range)
    budget = K * m * n
    K_alloc = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
    K_alloc, _ = local_search_refine(W, K_alloc.flatten(), H_G_train, H_X_train, tile, budget, 10)
    return K_alloc

# ─── GPTQ (same as R17) ──────────────────────────────────────────────────────

def _frozen_tile_codebooks(W, m, n, tile, K, bits_per_tile=None):
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


def gptq_block_diag(W, H_X, K, order=None, tile=TILE, block_size=BLOCK_SIZE,
                    damping=DAMPING, bits_per_tile=None):
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

# ─── Rotation helpers ──────────────────────────────────────────────────────────

def _rotate(W, H_X_train, H_G_train, rng):
    S_G, S_X, W_s = biip_scaling(W, H_X_train, H_G_train)
    U_rot, _ = signed_random_hadamard(W.shape[0], rng)
    V_rot, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U_rot @ W_s @ V_rot.T
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = V_rot @ (S_X_inv @ H_X_train @ S_X_inv) @ V_rot.T
    H_G_t = U_rot @ (S_G_inv @ H_G_train @ S_G_inv) @ U_rot.T
    return W_t, H_X_t, H_G_t, S_G_inv, U_rot, V_rot, S_X_inv


def _unrotate(Wq_t, S_G_inv, U_rot, V_rot, S_X_inv):
    return S_G_inv @ U_rot.T @ Wq_t @ V_rot @ S_X_inv

# ─── Arms ──────────────────────────────────────────────────────────────────────

def arm_rot_only(W, H_X_tr, H_G_tr, K, rng):
    W_t, _, _, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    return _unrotate(quantize_tiles(W_t, K, TILE), sg_inv, U, V, sx_inv)


def arm_rot_alloc(W, H_X_tr, H_G_tr, K, rng):
    W_t, H_X_t, H_G_t, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    return _unrotate(r1_dp_allocation(W_t, H_X_t, H_G_t, K, TILE), sg_inv, U, V, sx_inv)


def arm_rot_gptq(W, H_X_tr, H_G_tr, K, rng, gptq_fn):
    W_t, H_X_t, H_G_t, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    Wq_t = gptq_fn(W_t, H_X_t, K, order=np.arange(W_t.shape[1]), tile=TILE)
    return _unrotate(Wq_t, sg_inv, U, V, sx_inv)


def arm_rot_alloc_gptq(W, H_X_tr, H_G_tr, K, rng, gptq_fn):
    """R17 exact: rotate → DP allocate → GPTQ with bits_per_tile."""
    W_t, H_X_t, H_G_t, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    K_alloc = r1_dp_allocation_K(W_t, H_X_t, H_G_t, K, TILE)
    Wq_t = gptq_fn(W_t, H_X_t, K, order=np.arange(W_t.shape[1]), tile=TILE,
                   bits_per_tile=K_alloc)
    return _unrotate(Wq_t, sg_inv, U, V, sx_inv)


def _measure_per_tile_hwe(W, Wq, H_G, H_X, tile=TILE):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    E = W - Wq
    dists = np.zeros((n_tm, n_tn))
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            E_sub = E[r0:r1, c0:c1]
            dists[ti, tj] = float(np.trace(H_G[r0:r1, r0:r1] @ E_sub @ H_X[c0:c1, c0:c1] @ E_sub.T))
    return dists


def _realloc_corrected(W_t, Wq_pass1, H_G_t, H_X_t, K, K_pass1, tile=TILE):
    """Re-allocate using correction factors.

    FIX: measure expected distortion at SAME K_pass1 (not uniform K),
    so correction = D_actual(K_pass1) / D_expected(K_pass1) is meaningful.

    Then build corrected table: for each tile, D_corrected[ti,tj,k] =
    D_original[ti,tj,k] * c[ti,tj] where c captures how GPTQ changed
    this tile's relative sensitivity.
    """
    m, n = W_t.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    # Actual per-tile HWE from GPTQ pass 1 (with K_pass1)
    D_actual = _measure_per_tile_hwe(W_t, Wq_pass1, H_G_t, H_X_t, tile)

    # Expected: uniform quantization at the SAME K_pass1 grid
    Wq_expected = quantize_tiles(W_t, K_pass1, tile)
    D_expected = _measure_per_tile_hwe(W_t, Wq_expected, H_G_t, H_X_t, tile)

    # Correction factor
    c = np.ones((n_tm, n_tn))
    for ti in range(n_tm):
        for tj in range(n_tn):
            if D_expected[ti, tj] > 1e-15:
                c[ti, tj] = D_actual[ti, tj] / D_expected[ti, tj]

    # Build corrected distortion table
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W_t, H_G_t, H_X_t, tile, k_range)

    # Apply correction: scale each tile's distortion at every K by c[ti,tj]
    corrected_dists = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            for k in k_range:
                corrected_dists[(ti, tj, k)] = dists.get((ti, tj, k), float('inf')) * c[ti, tj]

    budget = K * m * n
    K_alloc = tile_k_dp_allocate(corrected_dists, n_tm, n_tn, tile * tile, budget, k_range)
    # NOTE: No local_search_refine here because it evaluates plain RTN HWE on W_t,
    # not the corrected/post-GPTQ objective. Refinement would erase the correction.
    return K_alloc


def arm_rot_bdgptq_realloc(W, H_X_tr, H_G_tr, K, rng, block_size=BLOCK_SIZE):
    """rotate → BD-GPTQ(uniform K) → correct → re-alloc → BD-GPTQ."""
    W_t, H_X_t, H_G_t, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    # Pass 1: BD-GPTQ with uniform K
    Wq_t_1 = gptq_block_diag(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                              tile=TILE, block_size=block_size)
    # Re-allocate using correction (expected at uniform K, matching pass 1)
    K_realloc = _realloc_corrected(W_t, Wq_t_1, H_G_t, H_X_t, K, K, TILE)
    # Pass 2: BD-GPTQ with re-allocated K
    Wq_t_2 = gptq_block_diag(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                              tile=TILE, block_size=block_size, bits_per_tile=K_realloc)
    return _unrotate(Wq_t_2, sg_inv, U, V, sx_inv)


def arm_rot_alloc_bdgptq_realloc(W, H_X_tr, H_G_tr, K, rng, block_size=BLOCK_SIZE):
    """rotate → alloc → BD-GPTQ → correct → re-alloc → BD-GPTQ."""
    W_t, H_X_t, H_G_t, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    # Pass 1: allocate then BD-GPTQ
    K_alloc_1 = r1_dp_allocation_K(W_t, H_X_t, H_G_t, K, TILE)
    Wq_t_1 = gptq_block_diag(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                              tile=TILE, block_size=block_size, bits_per_tile=K_alloc_1)
    # Re-allocate: expected at K_alloc_1 (matching pass 1), not uniform K
    K_realloc = _realloc_corrected(W_t, Wq_t_1, H_G_t, H_X_t, K, K_alloc_1, TILE)
    # Pass 2: BD-GPTQ with re-allocated K
    Wq_t_2 = gptq_block_diag(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                              tile=TILE, block_size=block_size, bits_per_tile=K_realloc)
    return _unrotate(Wq_t_2, sg_inv, U, V, sx_inv)


def arm_rot_alloc_fullgptq_realloc(W, H_X_tr, H_G_tr, K, rng):
    """rotate → alloc → Full-GPTQ → correct → re-alloc → Full-GPTQ (control)."""
    W_t, H_X_t, H_G_t, sg_inv, U, V, sx_inv = _rotate(W, H_X_tr, H_G_tr, rng)
    K_alloc_1 = r1_dp_allocation_K(W_t, H_X_t, H_G_t, K, TILE)
    Wq_t_1 = gptq_full(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                       tile=TILE, bits_per_tile=K_alloc_1)
    K_realloc = _realloc_corrected(W_t, Wq_t_1, H_G_t, H_X_t, K, K_alloc_1, TILE)
    Wq_t_2 = gptq_full(W_t, H_X_t, K, order=np.arange(W_t.shape[1]),
                       tile=TILE, bits_per_tile=K_realloc)
    return _unrotate(Wq_t_2, sg_inv, U, V, sx_inv)

# ─── Interference mechanism (fixed) ───────────────────────────────────────────

def analyze_interference(W_t, H_X_t, H_G_t, K, block_size=BLOCK_SIZE):
    m, n = W_t.shape
    n_tm = (m + TILE - 1) // TILE
    n_tn = (n + TILE - 1) // TILE

    Wq_uniform = quantize_tiles(W_t, K, TILE)
    uniform_dists = _measure_per_tile_hwe(W_t, Wq_uniform, H_G_t, H_X_t, TILE)

    Wq_full = gptq_full(W_t, H_X_t, K, order=np.arange(n), tile=TILE)
    full_dists = _measure_per_tile_hwe(W_t, Wq_full, H_G_t, H_X_t, TILE)

    Wq_bd = gptq_block_diag(W_t, H_X_t, K, order=np.arange(n), tile=TILE, block_size=block_size)
    bd_dists = _measure_per_tile_hwe(W_t, Wq_bd, H_G_t, H_X_t, TILE)

    # FIX: Cross-block error propagation.
    # BD-GPTQ U is block-diagonal → U[idx, idx+1:] is zero when idx and idx+1
    # are in different blocks. Measure the ACTUAL cross-block propagation by
    # checking which columns receive non-zero updates.
    H_perm = H_X_t[np.ix_(np.arange(n), np.arange(n))]
    U_full = inv_cholesky(H_perm, DAMPING)
    U_bd = block_diag_inv_cholesky(H_perm, block_size, DAMPING)

    # Count non-zero off-diagonal entries in each
    n_offdiag_full = int(np.sum(np.abs(U_full[np.triu_indices(n, k=1)]) > 1e-15))
    n_offdiag_bd = int(np.sum(np.abs(U_bd[np.triu_indices(n, k=1)]) > 1e-15))
    total_offdiag = n * (n - 1) // 2

    # Cross-block coupling mass: sum of |U[i,j] / U[i,i]| for cross-block entries.
    # This is NOT the actual update magnitude (which also depends on e_q),
    # but measures the coupling strength between blocks.
    cross_coupling_full = 0.0
    cross_coupling_bd = 0.0
    for idx in range(n - 1):
        block_start = (idx // block_size) * block_size
        block_end = min(block_start + block_size, n)
        cross_idx = np.arange(max(block_end, idx + 1), n)
        if len(cross_idx) > 0:
            u_ii_f = U_full[idx, idx]
            u_ii_b = U_bd[idx, idx]
            if abs(u_ii_f) > 1e-15:
                cross_coupling_full += np.sum(np.abs(U_full[idx, cross_idx] / u_ii_f))
            if abs(u_ii_b) > 1e-15:
                cross_coupling_bd += np.sum(np.abs(U_bd[idx, cross_idx] / u_ii_b))

    # Allocation stability
    k_range = range(max(3, K - 1), min(7, K + 2))
    dists = compute_tile_distortions(W_t, H_G_t, H_X_t, TILE, k_range)
    budget = K * m * n
    K_pre = tile_k_dp_allocate(dists, n_tm, n_tn, TILE * TILE, budget, k_range)
    K_pre, _ = local_search_refine(W_t, K_pre.flatten(), H_G_t, H_X_t, TILE, budget, 10)

    K_post_full = _realloc_corrected(W_t, Wq_full, H_G_t, H_X_t, K, K, TILE)
    K_post_bd = _realloc_corrected(W_t, Wq_bd, H_G_t, H_X_t, K, K, TILE)

    alloc_change_full = int(np.sum(K_pre != K_post_full))
    alloc_change_bd = int(np.sum(K_pre != K_post_bd))

    # Correlation: does GPTQ help sensitive tiles more?
    gptq_effect_full = uniform_dists - full_dists  # positive = GPTQ reduced distortion
    gptq_effect_bd = uniform_dists - bd_dists

    return {
        "uniform_dists_mean": float(np.mean(uniform_dists)),
        "full_dists_mean": float(np.mean(full_dists)),
        "bd_dists_mean": float(np.mean(bd_dists)),
        "n_offdiag_full": n_offdiag_full,
        "n_offdiag_bd": n_offdiag_bd,
        "total_offdiag": total_offdiag,
        "cross_coupling_full": float(cross_coupling_full),
        "cross_coupling_bd": float(cross_coupling_bd),
        "alloc_change_full": alloc_change_full,
        "alloc_change_bd": alloc_change_bd,
        "total_tiles": n_tm * n_tn,
        "dist_change_full": float(np.mean(np.abs(full_dists - uniform_dists))),
        "dist_change_bd": float(np.mean(np.abs(bd_dists - uniform_dists))),
        "corr_effect_sens_full": float(np.corrcoef(gptq_effect_full.flatten(), uniform_dists.flatten())[0, 1]),
        "corr_effect_sens_bd": float(np.corrcoef(gptq_effect_bd.flatten(), uniform_dists.flatten())[0, 1]),
        "K_pre_sum": int(np.sum(K_pre)),
        "K_post_full_sum": int(np.sum(K_post_full)),
        "K_post_bd_sum": int(np.sum(K_post_bd)),
    }

# ─── Inter-layer (fixed: matched comparison) ──────────────────────────────────

def interlayer_test(all_tensors, split_seed, K):
    """FIX: Compare Uniform_K vs InterLayer_DP at SAME GPTQ level.
    Not Uniform_NoGPTQ vs DP_GPTQ (which conflates rotation+GPTQ with allocation).
    """
    n_in = SLICE
    X_full = gen_calibration(n_in, N_CALIB, split_seed)
    n_train = int(N_CALIB * TRAIN_FRAC)
    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(N_CALIB)
    X_train = X_full[:, perm[:n_train]]
    X_test = X_full[:, perm[n_train:]]

    tensor_data = []
    for tname in TENSOR_NAMES:
        W = all_tensors[f"{tname}_first"]
        H_G_tr, H_X_tr = compute_hessians(W, X_train)
        H_G_te, H_X_te = compute_hessians(W, X_test)
        dists = {}
        for k in K_VALUES:
            Wq = quantize_tiles(W, k, TILE)
            dists[k] = hessian_weighted_error(W - Wq, H_G_tr, H_X_tr)
        tensor_data.append({"name": tname, "W": W, "H_G_tr": H_G_tr, "H_X_tr": H_X_tr,
                           "H_G_te": H_G_te, "H_X_te": H_X_te, "dist_curve": dists})

    # Inter-layer DP
    total_budget = len(TENSOR_NAMES) * K
    dp = {0: (0.0, [])}
    for td in tensor_data:
        dp_new = {}
        for b, (d, a) in dp.items():
            for k in K_VALUES:
                nb = b + k
                nd = d + td["dist_curve"][k]
                na = a + [k]
                if nb not in dp_new or nd < dp_new[nb][0]:
                    dp_new[nb] = (nd, na)
        dp = dp_new
    best_d, best_alloc = dp[total_budget]

    results = {}
    # For each GPTQ level, compare Uniform_K vs InterLayer_DP
    for gptq_label, gptq_fn_or_none in [("NoGPTQ", None), ("RotBDGPTQ", "bd"), ("RotFullGPTQ", "full")]:
        for alloc_label, alloc in [("Uniform_K", [K]*len(tensor_data)), ("InterLayer_DP", best_alloc)]:
            total = 0.0
            for i, td in enumerate(tensor_data):
                k_i = alloc[i]
                if gptq_fn_or_none is None:
                    Wq = quantize_tiles(td["W"], k_i, TILE)
                else:
                    rng_i = np.random.default_rng(split_seed + K + 100 + i)
                    gptq_fn = gptq_block_diag if gptq_fn_or_none == "bd" else gptq_full
                    Wq = arm_rot_gptq(td["W"], td["H_X_tr"], td["H_G_tr"], k_i, rng_i, gptq_fn)
                total += hessian_weighted_error(td["W"] - Wq, td["H_G_te"], td["H_X_te"])
            results[f"{alloc_label}_{gptq_label}"] = {"alloc": alloc, "total_hwe_test": total}

    return results

# ─── Main ──────────────────────────────────────────────────────────────────────

def load_real_weights_slices(slice_size=128, slices=["first"]):
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in data.keys():
        W = data[key].astype(np.float64)
        m, n = W.shape
        s = min(slice_size, m), min(slice_size, n)
        if "first" in slices:
            tensors[f"{key}_first"] = W[:s[0], :s[1]]
        if "middle" in slices and n >= 3 * s[1]:
            mid = n // 2 - s[1] // 2
            tensors[f"{key}_middle"] = W[:s[0], mid:mid + s[1]]
        if "last" in slices:
            tensors[f"{key}_last"] = W[:s[0], n - s[1]:n]
    return tensors


def run_eval(W, W_hat, H_G_tr, H_X_tr, H_G_te, H_X_te):
    E = W - W_hat
    return {"in_sample_hwe": hessian_weighted_error(E, H_G_tr, H_X_tr),
            "held_out_hwe": hessian_weighted_error(E, H_G_te, H_X_te),
            "weight_mse": weight_mse(E)}


def pct_imp(base, method):
    return (base - method) / base * 100.0 if base > 0 else 0.0


def main():
    t_start = time.time()
    print("=" * 95)
    print("R22 — Block-Diagonal GPTQ + Allocation Composition (v3)")
    print("=" * 95)
    print(f"  Tensors: {TENSOR_NAMES} | Slices: {SLICE_NAMES} | K: {K_VALUES}")
    print(f"  Splits: {N_SPLITS} × 80/20 | Block: {BLOCK_SIZE} | Damping: {DAMPING}")
    print(f"  v3 fixes: R17-exact allocator (full-HWE local search, K up to 7)")
    print(f"  v3 fixes: realloc uses matched K for correction")
    print(f"  v3 fixes: interlayer matched comparison (same GPTQ level)")
    print(f"  v3 fixes: cross-block propagation measured directly from U")

    print("\nLoading real Qwen3.8-27B weights...")
    all_tensors = load_real_weights_slices(SLICE, SLICE_NAMES)

    test_keys = []
    for tname in TENSOR_NAMES:
        for sname in SLICE_NAMES:
            key = f"{tname}_{sname}"
            if key in all_tensors:
                test_keys.append(key)

    split_seeds = [42 + i * 100 for i in range(N_SPLITS)]

    arms = [
        "RTN", "Rot_Only", "Rot_Alloc", "Rot_FullGPTQ", "Rot_BDGPTQ",
        "Rot_Alloc_FullGPTQ", "Rot_Alloc_BDGPTQ",
        "Rot_BDGPTQ_Realloc", "Rot_Alloc_BDGPTQ_Realloc", "Rot_Alloc_FullGPTQ_Realloc",
    ]

    results = {a: {k: {K: [] for K in K_VALUES} for k in test_keys} for a in arms}

    for split_idx, split_seed in enumerate(split_seeds):
        print(f"\n{'─'*95}")
        print(f"Split {split_idx+1}/{N_SPLITS} (seed={split_seed})")
        print(f"{'─'*95}")

        for tkey in test_keys:
            W = all_tensors[tkey]
            m, n = W.shape
            print(f"\n  {tkey} ({m}×{n})")

            X_full = gen_calibration(n, N_CALIB, split_seed)
            n_train = int(N_CALIB * TRAIN_FRAC)
            rng_split = np.random.default_rng(split_seed)
            perm = rng_split.permutation(N_CALIB)
            X_train = X_full[:, perm[:n_train]]
            X_test = X_full[:, perm[n_train:]]
            H_G_tr, H_X_tr = compute_hessians(W, X_train)
            H_G_te, H_X_te = compute_hessians(W, X_test)

            for K in K_VALUES:
                print(f"    K={K}:", end=" ")

                Wq = quantize_tiles(W, K, TILE)
                res = run_eval(W, Wq, H_G_tr, H_X_tr, H_G_te, H_X_te)
                results["RTN"][tkey][K].append(res)
                rtn_out = res["held_out_hwe"]

                def report(arm_name, Wq_arm):
                    r = run_eval(W, Wq_arm, H_G_tr, H_X_tr, H_G_te, H_X_te)
                    results[arm_name][tkey][K].append(r)
                    out = pct_imp(rtn_out, r["held_out_hwe"])
                    return f"{arm_name}:{out:+.1f}%"

                parts = []
                rng = np.random.default_rng(split_seed + K + 1)
                parts.append(report("Rot_Only", arm_rot_only(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1))))
                parts.append(report("Rot_Alloc", arm_rot_alloc(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1))))
                parts.append(report("Rot_FullGPTQ", arm_rot_gptq(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1), gptq_full)))
                parts.append(report("Rot_BDGPTQ", arm_rot_gptq(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1), gptq_block_diag)))
                parts.append(report("Rot_Alloc_FullGPTQ", arm_rot_alloc_gptq(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1), gptq_full)))
                parts.append(report("Rot_Alloc_BDGPTQ", arm_rot_alloc_gptq(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1), gptq_block_diag)))
                parts.append(report("Rot_BDGPTQ_Realloc", arm_rot_bdgptq_realloc(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1))))
                parts.append(report("Rot_Alloc_BDGPTQ_Realloc", arm_rot_alloc_bdgptq_realloc(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1))))
                parts.append(report("Rot_Alloc_FullGPTQ_Realloc", arm_rot_alloc_fullgptq_realloc(W, H_X_tr, H_G_tr, K, np.random.default_rng(split_seed + K + 1))))
                print(" | ".join(parts))

    # ─── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("AGGREGATE RESULTS: Mean ± Std across splits and tensors")
    print(f"{'='*95}")

    summary = {}
    for arm in arms:
        summary[arm] = {}
        for K in K_VALUES:
            in_imps, out_imps, gaps = [], [], []
            for tkey in test_keys:
                for r, s in zip(results["RTN"][tkey][K], results[arm][tkey][K]):
                    in_imps.append(pct_imp(r["in_sample_hwe"], s["in_sample_hwe"]))
                    out_imps.append(pct_imp(r["held_out_hwe"], s["held_out_hwe"]))
                    gaps.append(in_imps[-1] - out_imps[-1])
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

    # ─── Key comparisons ──────────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("KEY COMPARISONS")
    print(f"{'='*95}")

    def paired_test(name_a, name_b, label_a, label_b, K_filter=None):
        a_wins, b_wins, ties, total = 0, 0, 0, 0
        a_imps, b_imps = [], []
        for tkey in test_keys:
            for K in K_VALUES:
                if K_filter is not None and K != K_filter:
                    continue
                for a, b, r in zip(results[name_a].get(tkey, {}).get(K, []),
                                    results[name_b].get(tkey, {}).get(K, []),
                                    results["RTN"].get(tkey, {}).get(K, [])):
                    ai = pct_imp(r["held_out_hwe"], a["held_out_hwe"])
                    bi = pct_imp(r["held_out_hwe"], b["held_out_hwe"])
                    a_imps.append(ai)
                    b_imps.append(bi)
                    if ai > bi + 1e-12:
                        a_wins += 1
                    elif bi > ai + 1e-12:
                        b_wins += 1
                    else:
                        ties += 1
                    total += 1
        diff = np.mean(a_imps) - np.mean(b_imps) if a_imps else 0
        print(f"  {label_a:<35} vs {label_b:<35}  A:{a_wins} B:{b_wins} T:{ties} (n={total})  Δ={diff:+.2f}pp")

    print("\n  Q1: Does BD-GPTQ interfere with allocation? (R17 replication)")
    paired_test("Rot_Alloc_BDGPTQ", "Rot_Alloc", "Rot+Alloc+BDGPTQ", "Rot+Alloc (no GPTQ)")
    paired_test("Rot_Alloc_FullGPTQ", "Rot_Alloc", "Rot+Alloc+FullGPTQ", "Rot+Alloc (no GPTQ)")
    paired_test("Rot_Alloc_BDGPTQ", "Rot_Alloc_FullGPTQ", "BD-GPTQ+alloc", "Full-GPTQ+alloc")

    print("\n  Q2: Does BD-GPTQ+alloc beat BD-GPTQ alone? (synergy test)")
    paired_test("Rot_Alloc_BDGPTQ", "Rot_BDGPTQ", "Alloc+BDGPTQ", "BDGPTQ only")
    paired_test("Rot_Alloc_FullGPTQ", "Rot_FullGPTQ", "Alloc+FullGPTQ", "FullGPTQ only")

    print("\n  Q3: Does re-allocation help?")
    paired_test("Rot_BDGPTQ_Realloc", "Rot_BDGPTQ", "BDGPTQ+Realloc", "BDGPTQ only")
    paired_test("Rot_Alloc_BDGPTQ_Realloc", "Rot_Alloc_BDGPTQ", "Alloc+BDGPTQ+Realloc", "Alloc+BDGPTQ")
    paired_test("Rot_Alloc_FullGPTQ_Realloc", "Rot_Alloc_FullGPTQ", "Full+Realloc", "Alloc+Full no realloc")

    print("\n  Q4: Best triple stack vs best bifurcated?")
    paired_test("Rot_Alloc_BDGPTQ_Realloc", "Rot_Alloc", "Triple(realloc)", "Alloc only")
    paired_test("Rot_Alloc_BDGPTQ_Realloc", "Rot_BDGPTQ", "Triple(realloc)", "BD-GPTQ only")
    paired_test("Rot_Alloc_BDGPTQ", "Rot_BDGPTQ", "Alloc+BDGPTQ", "BD-GPTQ only")

    print("\n  Q5: Per-K interaction effects")
    for K in K_VALUES:
        paired_test("Rot_Alloc_BDGPTQ", "Rot_BDGPTQ", f"Alloc+BD K={K}", f"BD only K={K}", K_filter=K)
        paired_test("Rot_Alloc_BDGPTQ", "Rot_Alloc", f"Alloc+BD K={K}", f"Alloc only K={K}", K_filter=K)

    # ─── Interference mechanism ────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("INTERFERENCE MECHANISM ANALYSIS")
    print(f"{'='*95}")

    interference_results = []
    for tkey in ["L0_gate_first", "L55_gate_first", "L0_down_first", "L55_down_first"]:
        W = all_tensors[tkey]
        m, n = W.shape
        X = gen_calibration(n, N_CALIB, 42)
        H_G, H_X = compute_hessians(W, X)
        rng = np.random.default_rng(42 + 5)
        W_t, H_X_t, H_G_t, _, _, _, _ = _rotate(W, H_X, H_G, rng)
        analysis = analyze_interference(W_t, H_X_t, H_G_t, K=5, block_size=BLOCK_SIZE)
        analysis["tensor"] = tkey
        interference_results.append(analysis)
        print(f"\n  {tkey} (K=5, rotated, block_size={BLOCK_SIZE}):")
        print(f"    Mean dist: uniform={analysis['uniform_dists_mean']:.6f}  full={analysis['full_dists_mean']:.6f}  bd={analysis['bd_dists_mean']:.6f}")
        print(f"    Off-diag U entries: full={analysis['n_offdiag_full']}/{analysis['total_offdiag']}  bd={analysis['n_offdiag_bd']}/{analysis['total_offdiag']}")
        print(f"    Cross-block coupling mass: full={analysis['cross_coupling_full']:.4f}  bd={analysis['cross_coupling_bd']:.4f}")
        print(f"    Alloc change: full={analysis['alloc_change_full']}/{analysis['total_tiles']}  bd={analysis['alloc_change_bd']}/{analysis['total_tiles']}")
        print(f"    Corr(GPTQ help, sensitivity): full={analysis['corr_effect_sens_full']:+.3f}  bd={analysis['corr_effect_sens_bd']:+.3f}")

    # ─── Inter-layer (fixed) ────────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("INTER-LAYER ALLOCATION + BLOCK-DIAG GPTQ (matched comparison)")
    print(f"{'='*95}")

    interlayer_results = {}
    for si in range(N_SPLITS):
        ss = 42 + si * 100
        for K in [4, 5, 6]:
            res = interlayer_test(all_tensors, ss, K)
            interlayer_results[f"split{si}_K{K}"] = res

            # Matched comparisons: same GPTQ level, Uniform vs DP
            for gptq_label in ["NoGPTQ", "RotBDGPTQ", "RotFullGPTQ"]:
                uni = res[f"Uniform_K_{gptq_label}"]["total_hwe_test"]
                dp = res[f"InterLayer_DP_{gptq_label}"]["total_hwe_test"]
                imp = pct_imp(uni, dp)

            # Key: does DP allocation help WITH BD-GPTQ?
            uni_bd = res["Uniform_K_RotBDGPTQ"]["total_hwe_test"]
            dp_bd = res["InterLayer_DP_RotBDGPTQ"]["total_hwe_test"]
            dp_ng = res["InterLayer_DP_NoGPTQ"]["total_hwe_test"]
            dp_full = res["InterLayer_DP_RotFullGPTQ"]["total_hwe_test"]

            print(f"  Split{si} K={K}:")
            print(f"    NoGPTQ:  Uni={pct_imp(res['Uniform_K_NoGPTQ']['total_hwe_test'], res['InterLayer_DP_NoGPTQ']['total_hwe_test']):+.1f}%  "
                  f"BDGPTQ: Uni={pct_imp(uni_bd, dp_bd):+.1f}%  "
                  f"FullGPTQ: Uni={pct_imp(res['Uniform_K_RotFullGPTQ']['total_hwe_test'], dp_full):+.1f}%")
            print(f"    DP+BD vs DP: {pct_imp(dp_ng, dp_bd):+.1f}%  DP+Full vs DP: {pct_imp(dp_ng, dp_full):+.1f}%  alloc={res['InterLayer_DP_NoGPTQ']['alloc']}")

    # ─── Save ────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "config": {"n_splits": N_SPLITS, "train_frac": TRAIN_FRAC, "n_calib": N_CALIB,
                   "k_values": K_VALUES, "tensor_names": TENSOR_NAMES, "slice_names": SLICE_NAMES,
                   "test_keys": test_keys, "tile_size": TILE, "slice_size": SLICE,
                   "block_size": BLOCK_SIZE, "damping": DAMPING, "split_seeds": split_seeds,
                   "calibration": "independent_channels (R15 recipe)", "arms": arms,
                   "version": "v3",
                   "v3_fixes": "R17-exact allocator (full-HWE local search K up to 7), matched realloc K, matched interlayer comparison, direct cross-block U measurement"},
        "summary": summary,
        "interference_analysis": interference_results,
        "interlayer_results": interlayer_results,
        "raw_results": {a: {k: {K: [r for r in v] for K, v in d.items()} for k, d in d2.items()} for a, d2 in results.items()},
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
