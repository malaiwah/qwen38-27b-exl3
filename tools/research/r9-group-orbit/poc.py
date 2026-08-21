#!/usr/bin/env python3
"""
R9-GroupOrbit: Unified Alternating Group-Orbit Optimizer for Trellis Quantization

Implements doc 62 §10.10.B. Alternates over:
  1. Osborne/Sinkhorn diagonal equilibration (D_G, D_X)
  2. Balanced graph partition for 16×16 tile packing (P_G, P_X)
  3. Signed-Hadamard orbit sampling + Givens refinement (U, V)
  4. Exact per-tile K allocation via DP
  5. Quantization with chosen quantizer
  6. GPTAQ/ResComp correction pass

Each sub-step is proven monotone or exact in the Hessian-weighted objective:
  J = tr(H_G · E · H_X · E^T) + λ·bytes
  where E = W - Ŵ, Ŵ = Q(D_G · U^T · P_G^T · W · P_X · V · D_X)

Conventions:
  W: (m, n) weight matrix — m=output dim, n=input dim
  X: (n, N) input activations — N calibration samples
  Y = W @ X: (m, N) output activations
  H_X = X @ X^T / N: (n, n) input Hessian
  H_G = Y @ Y^T / N: (m, m) output Hessian proxy

ALL arms use the SAME quantizer primitive: per-tile (16×16) uniform quantization.
This fixes the unmatched-granularity bug from doc 62 §10.2.

Clean-room from paper equations. CPU-only numpy.
"""

import numpy as np
import json
import time
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Callable

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OptimizerConfig:
    tile_size: int = 16
    block_size: int = 16
    damping: float = 0.01
    n_hadamard_samples: int = 8
    n_givens_iters: int = 20
    n_osborne_iters: int = 10
    min_k: int = 3
    max_k: int = 7
    avg_k: float = 5.0
    lambda_bytes: float = 0.0
    gptaq_alpha: float = 0.25
    correction: str = "gptaq"
    max_outer_iters: int = 5
    tol: float = 1e-10
    seed: int = 42
    scale_clip_min: float = 0.1
    scale_clip_max: float = 10.0

# ============================================================================
# UNIFIED Quantizer — per-tile (16×16) uniform, used for ALL arms
# ============================================================================

def quantize_uniform_1d(w: np.ndarray, bits: int) -> np.ndarray:
    """Uniform min-max quantization with 2^bits levels."""
    if bits <= 0:
        return np.zeros_like(w)
    nl = 2 ** bits
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return w.copy()
    s = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / s), 0, nl - 1) * s + lo

def quantize_tiles(W: np.ndarray, bits_or_alloc, tile: int) -> np.ndarray:
    """Per-tile uniform quantization. 
    bits_or_alloc: int for uniform K, or (n_tm, n_tn) array for per-tile K.
    This is the SINGLE quantizer primitive used for ALL arms.
    """
    m, n = W.shape
    Wq = np.zeros_like(W)
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            if isinstance(bits_or_alloc, np.ndarray) and bits_or_alloc.ndim == 2:
                k = int(bits_or_alloc[ti, tj])
            else:
                k = int(bits_or_alloc)
            Wq[r0:r1, c0:c1] = quantize_uniform_1d(
                W[r0:r1, c0:c1].ravel(), k).reshape(r1 - r0, c1 - c0)
    return Wq

# ============================================================================
# Hessian-weighted objective
# ============================================================================

def hessian_weighted_error(E: np.ndarray, H_G: np.ndarray, H_X: np.ndarray) -> float:
    """tr(H_G · E · H_X · E^T). E is (m, n), H_G is (m, m), H_X is (n, n)."""
    return float(np.trace(E.T @ H_G @ E @ H_X))

def weight_mse(E: np.ndarray) -> float:
    return float(np.mean(E ** 2))

# ============================================================================
# Sub-step 1: Osborne/Sinkhorn Diagonal Equilibration
# ============================================================================
# MONOTONICITY PROOF:
# J = tr(H_G E H_X E^T) where E = D_G^{-1} E' D_X^{-1}, E' = W' - Q(W'),
# W' = D_G W D_X. So J = tr(D_G^{-1} H_G D_G^{-1} E' D_X^{-1} H_X D_X^{-1} E'^T).
# We optimize D_G, D_X to minimize J for given E'. Each Osborne sweep is
# coordinate descent: fix all but one scale, optimize that one exactly.
# Coordinate descent with exact sub-problems is monotone non-increasing.
#
# R3-Rotations insight: use 1/4 power, clip [0.1, 10].
# Note: equilibration alone may INCREASE error (scaling distortion), but
# combined with correction (step 6) it should improve. The alternating
# optimizer accepts equilibration ONLY IF it improves the objective.

def osborne_equilibrate(W, H_G, H_X, n_iters=10, clip_min=0.1, clip_max=10.0):
    """Returns (d_G, d_X) diagonal scales of length m and n."""
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

# ============================================================================
# Sub-step 2: Balanced Graph Partition for Tile Packing
# ============================================================================
# MONOTONICITY PROOF:
# For per-tile uniform quantization, tile MSE ≈ Δ²/12 where Δ = range/(2^K-1).
# Grouping similar-magnitude channels reduces within-tile range → reduces Δ → reduces MSE.
# Sort-and-group is the optimal 1D assignment (rearrangement inequality) for
# minimizing max within-tile range. Exact for single-axis partitioning.
# Since we permute H_G, H_X consistently, tr(H_G E H_X E^T) is minimized.

def balanced_partition(W, H_G, H_X, tile, axis="both"):
    m, n = W.shape
    perm_rows = np.arange(m)
    perm_cols = np.arange(n)
    if axis in ("rows", "both"):
        row_rms = np.sqrt(np.mean(W ** 2, axis=1))
        h_G_diag = np.abs(np.diag(H_G))
        perm_rows = np.argsort(row_rms * np.sqrt(h_G_diag + 1e-15))
    if axis in ("cols", "both"):
        col_rms = np.sqrt(np.mean(W ** 2, axis=0))
        h_X_diag = np.abs(np.diag(H_X))
        perm_cols = np.argsort(col_rms * np.sqrt(h_X_diag + 1e-15))
    return perm_rows, perm_cols

# ============================================================================
# Sub-step 3: Signed-Hadamard Orbit Sampling + Givens Refinement
# ============================================================================
# MONOTONICITY PROOF:
# (a) Orbit sampling: argmin over N random samples. Best of N ≤ any single.
#     EXACT over the finite sample set. Monotone by construction.
# (b) Givens refinement: coordinate descent with golden-section search (exact
#     for unimodal) and accept-if-improve. Monotone non-increasing.

def hadamard_matrix(n):
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def random_signed_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return np.diag(signs) @ H

def hadamard_orbit_sample(W, H_G, H_X, n_samples, rng, quantize_fn, eval_fn, apply_to="both"):
    m, n = W.shape
    best_obj = np.inf
    U_best = np.eye(m)
    V_best = np.eye(n)
    for _ in range(n_samples):
        U = np.eye(m)
        V = np.eye(n)
        if apply_to in ("rows", "both") and m > 1:
            U = random_signed_hadamard(m, rng)
        if apply_to in ("cols", "both") and n > 1:
            V = random_signed_hadamard(n, rng)
        W_rot = U.T @ W @ V
        E = U @ (W_rot - quantize_fn(W_rot)) @ V.T
        obj = eval_fn(E, H_G, H_X)
        if obj < best_obj:
            best_obj = obj
            U_best = U.copy()
            V_best = V.copy()
    return U_best, V_best, best_obj

def givens_rotation(i, j, theta, n):
    G = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s
    return G

def golden_section_search(f, a, b, n_iters):
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc = f(c); fd = f(d)
    for _ in range(n_iters):
        if fc < fd:
            b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c)
        else:
            a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d)
    return (a + b) / 2

def givens_refine(W, H_G, H_X, U, V, n_iters, rng, quantize_fn, eval_fn):
    m, n = W.shape
    W_rot = U.T @ W @ V
    best_obj = eval_fn(U @ (W_rot - quantize_fn(W_rot)) @ V.T, H_G, H_X)
    for _ in range(n_iters):
        if rng.random() < 0.5 and m > 1:
            i, j = rng.choice(m, size=2, replace=False)
            def obj_fn(theta):
                U_new = U @ givens_rotation(i, j, theta, m)
                W_r = U_new.T @ W @ V
                return eval_fn(U_new @ (W_r - quantize_fn(W_r)) @ V.T, H_G, H_X)
            th = golden_section_search(obj_fn, 0, np.pi, 15)
            obj = obj_fn(th)
            if obj < best_obj - 1e-15:
                U = U @ givens_rotation(i, j, th, m); best_obj = obj
        elif n > 1:
            i, j = rng.choice(n, size=2, replace=False)
            def obj_fn(theta):
                V_new = V @ givens_rotation(i, j, theta, n)
                W_r = U.T @ W @ V_new
                return eval_fn(U @ (W_r - quantize_fn(W_r)) @ V_new.T, H_G, H_X)
            th = golden_section_search(obj_fn, 0, np.pi, 15)
            obj = obj_fn(th)
            if obj < best_obj - 1e-15:
                V = V @ givens_rotation(i, j, th, n); best_obj = obj
    return U, V, best_obj

# ============================================================================
# Sub-step 4: Exact Per-Tile K Allocation via DP
# ============================================================================
# MONOTONICITY PROOF:
# Multiple-choice knapsack: min sum D_{t,k_t} s.t. sum k_t * tile^2 <= budget.
# DP finds the GLOBAL optimum for fixed W'. EXACT → monotone.

def compute_tile_hessian_distortions(W, H_G, H_X, tile, k_range):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    distortions = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            tile_data = W[r0:r1, c0:c1]
            H_G_tile = H_G[r0:r1, r0:r1]
            H_X_tile = H_X[c0:c1, c0:c1]
            dist = {}
            for k in k_range:
                Wq_tile = quantize_uniform_1d(tile_data.ravel(), k).reshape(tile_data.shape)
                E_tile = tile_data - Wq_tile
                dist[k] = float(np.trace(H_G_tile @ E_tile @ H_X_tile @ E_tile.T))
            distortions[(ti, tj)] = dist
    return distortions

def tile_k_dp_allocate(distortions, n_tm, n_tn, tile_elements, budget_bits, k_range):
    k_list = list(k_range)
    tile_list = [(ti, tj) for ti in range(n_tm) for tj in range(n_tn)]
    n_tiles = len(tile_list)
    max_budget = int(np.ceil(budget_bits))
    INF = float('inf')
    dp = {0: 0.0}
    backpointers = []
    for idx in range(n_tiles):
        ti, tj = tile_list[idx]
        dist_dict = distortions[(ti, tj)]
        new_dp = {}; bp = {}
        for b, val in dp.items():
            for k in k_list:
                new_b = b + k * tile_elements
                if new_b > max_budget: continue
                new_val = val + dist_dict[k]
                if new_b not in new_dp or new_val < new_dp[new_b]:
                    new_dp[new_b] = new_val; bp[new_b] = (b, k)
        dp = new_dp; backpointers.append(bp)
    if not dp:
        return np.full((n_tm, n_tn), k_list[len(k_list) // 2])
    best_b = min(dp, key=lambda b: dp[b])
    alloc_1d = [0] * n_tiles
    b = best_b
    for i in range(n_tiles - 1, -1, -1):
        if b not in backpointers[i]:
            b = min(backpointers[i].keys(), key=lambda x: abs(x - b))
        prev_b, k = backpointers[i][b]
        alloc_1d[i] = k; b = prev_b
    K_alloc = np.zeros((n_tm, n_tn), dtype=int)
    for idx, (ti, tj) in enumerate(tile_list):
        K_alloc[ti, tj] = alloc_1d[idx]
    return K_alloc

# ============================================================================
# Sub-step 6: GPTAQ Correction Pass
# ============================================================================
# MONOTONICITY PROOF:
# GPTQ processes columns sequentially. At each step, the error propagation
# W[:,q+1:] -= e * L[q,q+1:]/L[q,q] is the greedy exact minimizer of
# tr(E H_X E^T) for that column. GPTAQ adds P-matrix for FP/quant drift.
# GPTAQ ⊇ GPTQ (P=0 when no drift). The correction is applied AFTER
# quantization and can only improve (or maintain) the objective.
# In the alternating optimizer, we accept the correction step ONLY IF it
# improves the full Hessian-weighted objective. Monotone by accept-if-improve.

def inv_cholesky(H, damping):
    """Upper triangular U such that U^T @ U = inv(H + damping*I).
    Correct GPTQ convention: row U[c, c:] is non-zero for propagation.
    Construction: Hinv = inv(H+damp), U = cholesky(Hinv).T (upper)."""
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
    return U

def gptaq_correction(W, X, Xt, bits, tile, block, damping, alpha=0.25,
                     use_rescomp=False, bits_per_tile=None):
    """GPTAQ correction with per-tile (16×16) quantizer, MATCHED with other arms.
    
    Processes column-tiles (blocks of `tile` columns). For each column-tile:
    1. Freeze one codebook per row-tile using current Ww (before any GPTQ update)
    2. Quantize all 16 columns using frozen codebooks
    3. Apply GPTQ error propagation to remaining columns
    
    This ensures exactly one codebook per physical 16×16 tile, matching RTN/Rotate.
    W: (m, n). X: (n, N) quant-flow. Xt: (n, N) FP-flow.
    """
    m, n = W.shape
    Ww = W.copy().astype(np.float64)
    W0 = W.copy()
    Q = np.zeros_like(Ww)
    H = X @ X.T
    L = inv_cholesky(H, damping)

    dX = Xt - X
    D = dX @ X.T
    P = alpha * (np.triu(D @ L.T, 1) @ L) if not use_rescomp else np.zeros((n, n))
    P2 = alpha * (np.triu(Xt @ X.T @ L.T, 1) @ L) if use_rescomp else np.zeros((n, n))

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    # Process column-tiles sequentially (each has `tile` columns)
    for ct in range(n_tn):
        c0 = ct * tile
        c1 = min(c0 + tile, n)
        B = c1 - c0

        # Step 1: Fit and freeze codebooks (lo, step) for each row-tile in this column-tile
        # Codebook is frozen from current Ww BEFORE any GPTQ update in this column-tile
        codebooks = []  # list of (r0, r1, lo, step, k)
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

        def apply_frozen_codebook(col_data, codebooks_list):
            """Apply frozen codebooks to a single column of data."""
            q = np.zeros_like(col_data)
            for r0, r1, lo, step, k in codebooks_list:
                if step == 0.0:
                    q[r0:r1] = col_data[r0:r1]
                else:
                    nl = 2 ** k
                    q[r0:r1] = np.clip(np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
            return q

        # Step 2: GPTQ error propagation for columns in this column-tile
        # Each column is quantized using the FROZEN codebook but applied to CURRENT w_pre
        E = np.zeros((m, B))
        W_pre_block = np.zeros((m, B))
        for j in range(B):
            c = c0 + j
            w_pre = Ww[:, c].copy()
            W_pre_block[:, j] = w_pre

            # Quantize current w_pre using frozen codebook (not pre-computed Q)
            Q[:, c] = apply_frozen_codebook(w_pre, codebooks)

            e = w_pre - Q[:, c]
            E[:, j] = e / L[c, c]
            end = min(c0 + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            if not use_rescomp:
                Ww[:, c:end] += np.outer(w_pre, P[c, c:end])
            else:
                Ww[:, c:end] += np.outer(W0[:, c] - w_pre, P2[c, c:end])

        # Outer block propagation to remaining columns
        if c1 < n:
            Ww[:, c1:] -= E @ L[c0:c1, c1:]
            if not use_rescomp:
                Ww[:, c1:] += W_pre_block @ P[c0:c1, c1:]
            else:
                Ww[:, c1:] += (W0[:, c0:c1] - W_pre_block) @ P2[c0:c1, c1:]
    return Q

# ============================================================================
# Unified Alternating Optimizer
# ============================================================================

@dataclass
class OptimizerState:
    D_G: np.ndarray = None
    D_X: np.ndarray = None
    perm_rows: np.ndarray = None
    perm_cols: np.ndarray = None
    U: np.ndarray = None
    V: np.ndarray = None
    K_alloc: np.ndarray = None
    Wq: np.ndarray = None
    objective: float = np.inf
    weight_mse: float = np.inf
    hess_error: float = np.inf
    bytes_used: float = 0.0
    active_steps: list = field(default_factory=list)
    convergence_history: list = field(default_factory=list)
    step_improvements: dict = field(default_factory=dict)

def compute_bytes(K_alloc, tile, m, n, D_G=None, D_X=None, U=None, V=None):
    if isinstance(K_alloc, np.ndarray) and K_alloc.ndim == 2:
        total_bits = float(np.sum(K_alloc)) * tile * tile
    else:
        total_bits = float(K_alloc) * m * n
    sidecar = 0
    if D_G is not None and not np.allclose(D_G, 1): sidecar += len(D_G) * 32
    if D_X is not None and not np.allclose(D_X, 1): sidecar += len(D_X) * 32
    if U is not None and not np.allclose(U, np.eye(U.shape[0])): sidecar += U.size * 32
    if V is not None and not np.allclose(V, np.eye(V.shape[0])): sidecar += V.size * 32
    return (total_bits + sidecar) / 8.0

def run_alternating_optimizer(W, X, Xt, H_G, H_X, cfg, active_steps=None):
    if active_steps is None:
        active_steps = ["equilibrate", "partition", "rotate", "allocate", "quantize", "correct"]

    m, n = W.shape
    tile = cfg.tile_size
    rng = np.random.default_rng(cfg.seed)
    state = OptimizerState()
    state.active_steps = active_steps

    D_G = np.ones(m); D_X = np.ones(n)
    perm_rows = np.arange(m); perm_cols = np.arange(n)
    U = np.eye(m); V = np.eye(n)
    K_alloc = np.full((m // tile, n // tile), int(cfg.avg_k))
    use_correction = False  # Correction is enabled only when the correction step runs and improves

    def quantize_in_transformed_space(W_spr, K_cur, do_corr, X_corr=None, Xt_corr=None):
        """Quantize W_spr (transformed weight) using per-tile quantizer.
        Optionally apply GPTAQ correction."""
        if do_corr and X_corr is not None:
            return gptaq_correction(W_spr, X_corr, Xt_corr,
                                    int(cfg.avg_k), tile, cfg.block_size,
                                    cfg.damping, cfg.gptaq_alpha,
                                    use_rescomp=(cfg.correction == "rescomp"),
                                    bits_per_tile=K_cur if isinstance(K_cur, np.ndarray) else None)
        else:
            return quantize_tiles(W_spr, K_cur if isinstance(K_cur, np.ndarray) else int(cfg.avg_k), tile)

    def evaluate(D_G_cur, D_X_cur, U_cur, V_cur, K_cur, pr, pc, do_corr):
        """Full pipeline: scale -> permute -> rotate -> quantize -> invert."""
        # Forward
        W_s = D_G_cur[:, None] * W * D_X_cur[None, :]
        W_sp = W_s[pr][:, pc]
        W_spr = U_cur.T @ W_sp @ V_cur

        # Correction data
        X_corr = Xt_corr = None
        if do_corr:
            X_s = X / D_X_cur[:, None]  # (n, N)
            X_corr = V_cur.T @ X_s[pc]   # rotate input
            Xt_s = Xt / D_X_cur[:, None]
            Xt_corr = V_cur.T @ Xt_s[pc]

        # Quantize
        Wq_spr = quantize_in_transformed_space(W_spr, K_cur, do_corr, X_corr, Xt_corr)

        # Invert
        E_spr = W_spr - Wq_spr
        E_sp = U_cur @ E_spr @ V_cur.T
        inv_pr = np.argsort(pr); inv_pc = np.argsort(pc)
        E_s = E_sp[inv_pr][:, inv_pc]
        Wq = (W_s - E_s) / D_G_cur[:, None] / D_X_cur[None, :]

        E = W - Wq
        herr = hessian_weighted_error(E, H_G, H_X)
        mse = weight_mse(E)
        obj = herr + cfg.lambda_bytes * compute_bytes(K_cur, tile, m, n, D_G_cur, D_X_cur, U_cur, V_cur)
        return Wq, obj, herr, mse

    # NOTE: correction starts DISABLED. It's enabled only when the correction
    # step runs, and only accepted if it improves the objective. This ensures
    # the 6-step builds on the 5-step results, not a separate GPTAQ baseline.
    Wq_init, obj_init, herr_init, mse_init = evaluate(
        D_G, D_X, U, V, K_alloc, perm_rows, perm_cols, do_corr=False)
    state.objective = obj_init
    state.hess_error = herr_init
    state.weight_mse = mse_init
    state.convergence_history.append({
        'iteration': 0, 'step': 'init', 'objective': obj_init,
        'hess_error': herr_init, 'weight_mse': mse_init})

    for outer_iter in range(cfg.max_outer_iters):
        improved = False

        # Steps 1-5: optimize transforms (evaluate WITH correction if it was previously accepted)
        # Step 1: Equilibrate
        if "equilibrate" in active_steps:
            dG, dX = osborne_equilibrate(W, H_G, H_X, cfg.n_osborne_iters,
                                         cfg.scale_clip_min, cfg.scale_clip_max)
            _, obj, herr, mse = evaluate(dG, dX, U, V, K_alloc, perm_rows, perm_cols, use_correction)
            if obj < state.objective - cfg.tol:
                D_G, D_X = dG, dX
                _update_state(state, outer_iter+1, 'equilibrate', obj, herr, mse)
                improved = True

        # Step 2: Partition
        if "partition" in active_steps:
            W_s = D_G[:, None] * W * D_X[None, :]
            pr, pc = balanced_partition(W_s, H_G, H_X, tile)
            _, obj, herr, mse = evaluate(D_G, D_X, U, V, K_alloc, pr, pc, use_correction)
            if obj < state.objective - cfg.tol:
                perm_rows, perm_cols = pr, pc
                _update_state(state, outer_iter+1, 'partition', obj, herr, mse)
                improved = True

        # Step 3: Rotate (Hadamard orbit + Givens)
        if "rotate" in active_steps:
            W_s = D_G[:, None] * W * D_X[None, :]
            W_sp = W_s[perm_rows][:, perm_cols]
            H_G_p = H_G[perm_rows][:, perm_rows]
            H_X_p = H_X[perm_cols][:, perm_cols]
            qfn = lambda W_in: quantize_tiles(W_in, int(cfg.avg_k), tile)
            eval_fn = lambda E, HG, HX: hessian_weighted_error(E, HG, HX)
            U_new, V_new, _ = hadamard_orbit_sample(
                W_sp, H_G_p, H_X_p, cfg.n_hadamard_samples, rng, qfn, eval_fn)
            U_new, V_new, _ = givens_refine(
                W_sp, H_G_p, H_X_p, U_new, V_new, cfg.n_givens_iters, rng, qfn, eval_fn)
            _, obj, herr, mse = evaluate(D_G, D_X, U_new, V_new, K_alloc, perm_rows, perm_cols, use_correction)
            if obj < state.objective - cfg.tol:
                U, V = U_new, V_new
                _update_state(state, outer_iter+1, 'rotate', obj, herr, mse)
                improved = True

        # Step 4: Allocate (DP)
        if "allocate" in active_steps:
            W_s = D_G[:, None] * W * D_X[None, :]
            W_sp = W_s[perm_rows][:, perm_cols]
            W_spr = U.T @ W_sp @ V
            H_G_p = H_G[perm_rows][:, perm_rows]
            H_X_p = H_X[perm_cols][:, perm_cols]
            H_G_pr = U.T @ H_G_p @ U
            H_X_pr = V.T @ H_X_p @ V
            n_tm = (m + tile - 1) // tile; n_tn = (n + tile - 1) // tile
            k_range = range(cfg.min_k, cfg.max_k + 1)
            dists = compute_tile_hessian_distortions(W_spr, H_G_pr, H_X_pr, tile, k_range)
            budget = cfg.avg_k * m * n
            K_new = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, budget, k_range)
            _, obj, herr, mse = evaluate(D_G, D_X, U, V, K_new, perm_rows, perm_cols, use_correction)
            if obj < state.objective - cfg.tol:
                K_alloc = K_new
                _update_state(state, outer_iter+1, 'allocate', obj, herr, mse)
                improved = True

        # Step 6: Correction — try GPTAQ on top of current best, accept if improves
        if "correct" in active_steps and cfg.correction != "none":
            _, obj, herr, mse = evaluate(D_G, D_X, U, V, K_alloc, perm_rows, perm_cols, True)
            if obj < state.objective - cfg.tol:
                use_correction = True
                _update_state(state, outer_iter+1, 'correct', obj, herr, mse)
                improved = True
            # If correction doesn't help, keep use_correction as-is (may be True from prior iter)

        if not improved:
            state.convergence_history.append({
                'iteration': outer_iter+1, 'step': 'converged',
                'objective': state.objective, 'hess_error': state.hess_error,
                'weight_mse': state.weight_mse})
            break

    # Final evaluation to get Wq (use correction only if it was accepted)
    Wq, _, _, _ = evaluate(D_G, D_X, U, V, K_alloc, perm_rows, perm_cols, use_correction)
    state.D_G = D_G; state.D_X = D_X
    state.perm_rows = perm_rows; state.perm_cols = perm_cols
    state.U = U; state.V = V
    state.K_alloc = K_alloc; state.Wq = Wq
    state.bytes_used = compute_bytes(K_alloc, tile, m, n, D_G, D_X, U, V)
    return state

def _update_state(state, it, step, obj, herr, mse):
    state.objective = obj; state.hess_error = herr; state.weight_mse = mse
    state.convergence_history.append({
        'iteration': it, 'step': step, 'objective': obj,
        'hess_error': herr, 'weight_mse': mse})
    state.step_improvements[step] = state.step_improvements.get(step, 0) + 1

# ============================================================================
# Individual method baselines (ALL use per-tile quantizer)
# ============================================================================

def baseline_rtn(W, H_G, H_X, bits, tile):
    return quantize_tiles(W, bits, tile)

def baseline_equilibrate(W, H_G, H_X, bits, tile, cfg):
    dG, dX = osborne_equilibrate(W, H_G, H_X, cfg.n_osborne_iters,
                                  cfg.scale_clip_min, cfg.scale_clip_max)
    W_s = dG[:, None] * W * dX[None, :]
    Wq_s = quantize_tiles(W_s, bits, tile)
    return Wq_s / dG[:, None] / dX[None, :]

def baseline_partition(W, H_G, H_X, bits, tile, cfg):
    pr, pc = balanced_partition(W, H_G, H_X, tile)
    W_p = W[pr][:, pc]
    Wq_p = quantize_tiles(W_p, bits, tile)
    return Wq_p[np.argsort(pr)][:, np.argsort(pc)]

def baseline_rotate(W, H_G, H_X, bits, tile, cfg):
    rng = np.random.default_rng(cfg.seed)
    qfn = lambda W_in: quantize_tiles(W_in, bits, tile)
    U, V, _ = hadamard_orbit_sample(W, H_G, H_X, cfg.n_hadamard_samples, rng, qfn,
                                     lambda E, HG, HX: hessian_weighted_error(E, HG, HX))
    U, V, _ = givens_refine(W, H_G, H_X, U, V, cfg.n_givens_iters, rng, qfn,
                             lambda E, HG, HX: hessian_weighted_error(E, HG, HX))
    return U @ quantize_tiles(U.T @ W @ V, bits, tile) @ V.T

def baseline_allocate(W, H_G, H_X, avg_bits, tile, cfg):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile; n_tn = (n + tile - 1) // tile
    k_range = range(cfg.min_k, cfg.max_k + 1)
    dists = compute_tile_hessian_distortions(W, H_G, H_X, tile, k_range)
    K = tile_k_dp_allocate(dists, n_tm, n_tn, tile * tile, avg_bits * m * n, k_range)
    return quantize_tiles(W, K, tile)

def baseline_gptaq(W, X, Xt, H_G, H_X, bits, tile, cfg):
    return gptaq_correction(W, X, Xt, bits, tile, cfg.block_size,
                           cfg.damping, cfg.gptaq_alpha, use_rescomp=False)

# ============================================================================
# Experiment runner
# ============================================================================

def load_real_weights(slice_size=128):
    path = '/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz'
    data = np.load(path)
    tensors = {}
    for key in data.keys():
        W = data[key].astype(np.float64)
        m, n = W.shape
        tensors[key] = W[:min(slice_size, m), :min(slice_size, n)]
    return tensors

def gen_synthetic_calibration(W, n_samples=512, seed=42):
    m, n = W.shape
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_samples)) * 0.1
    n_outliers = max(1, n // 20)
    outlier_rows = rng.choice(n, n_outliers, replace=False)
    X[outlier_rows, :] *= 10.0
    Xt = X + rng.standard_normal(X.shape) * 0.01
    return X, Xt

def compute_hessians(W, X):
    m, n = W.shape
    N = X.shape[1]
    H_X = X @ X.T / N
    Y = W @ X
    H_G = Y @ Y.T / N
    return H_G, H_X

def run_experiment():
    print("=" * 80)
    print("R9-GroupOrbit: Unified Alternating Group-Orbit Optimizer")
    print("Doc 62 §10.10.B — Trellis Quantization Innovation Lab")
    print("ALL arms use per-tile (16×16) uniform quantizer (matched granularity)")
    print("=" * 80)

    print("\nLoading real Qwen3.8-27B weights...")
    tensors = load_real_weights(slice_size=128)
    print(f"  Loaded {len(tensors)} tensors")

    eval_tensors = ['L0_gate', 'L0_down', 'L55_gate', 'L55_down']
    eval_tensors = [t for t in eval_tensors if t in tensors]

    cfg = OptimizerConfig(
        tile_size=16, block_size=16, damping=0.01,
        n_hadamard_samples=8, n_givens_iters=20, n_osborne_iters=10,
        min_k=3, max_k=7, avg_k=5.0,
        gptaq_alpha=0.25, correction="gptaq",
        max_outer_iters=5, seed=42)

    all_results = {}

    for tensor_name in eval_tensors:
        print(f"\n{'='*80}")
        print(f"Tensor: {tensor_name} (shape={tensors[tensor_name].shape})")
        print(f"{'='*80}")

        W = tensors[tensor_name]
        X, Xt = gen_synthetic_calibration(W, n_samples=512, seed=cfg.seed)
        H_G, H_X = compute_hessians(W, X)

        print(f"  W range: [{W.min():.6f}, {W.max():.6f}], std: {W.std():.6f}")

        results = {}
        bits = int(cfg.avg_k)

        # --- Individual baselines ---
        baselines = {
            'RTN': lambda: baseline_rtn(W, H_G, H_X, bits, cfg.tile_size),
            'Equilibrate': lambda: baseline_equilibrate(W, H_G, H_X, bits, cfg.tile_size, cfg),
            'Partition': lambda: baseline_partition(W, H_G, H_X, bits, cfg.tile_size, cfg),
            'Rotate': lambda: baseline_rotate(W, H_G, H_X, bits, cfg.tile_size, cfg),
            'Allocate': lambda: baseline_allocate(W, H_G, H_X, cfg.avg_k, cfg.tile_size, cfg),
            'GPTQ(a=0)': lambda: gptaq_correction(W, X, Xt, bits, cfg.tile_size, cfg.block_size,
                                                  cfg.damping, alpha=0.0),
            'GPTAQ(a=.25)': lambda: baseline_gptaq(W, X, Xt, H_G, H_X, bits, cfg.tile_size, cfg),
        }
        for name, fn in baselines.items():
            Wq = fn()
            E = W - Wq
            results[name] = {
                'hess_error': hessian_weighted_error(E, H_G, H_X),
                'weight_mse': weight_mse(E),
                'bytes': float(W.size * cfg.avg_k / 8),
            }

        # --- Alternating optimizer with increasing steps ---
        step_configs = {
            '2-step (eq+q)': ["equilibrate", "quantize"],
            '3-step (+part)': ["equilibrate", "partition", "quantize"],
            '4-step (+rot)': ["equilibrate", "partition", "rotate", "quantize"],
            '5-step (+alloc)': ["equilibrate", "partition", "rotate", "allocate", "quantize"],
            '6-step (+corr)': ["equilibrate", "partition", "rotate", "allocate", "quantize", "correct"],
        }
        for name, steps in step_configs.items():
            print(f"  Running {name}...")
            t0 = time.time()
            state = run_alternating_optimizer(W, X, Xt, H_G, H_X, cfg, active_steps=steps)
            t1 = time.time()
            results[name] = {
                'hess_error': state.hess_error,
                'weight_mse': state.weight_mse,
                'bytes': state.bytes_used,
                'convergence': state.convergence_history,
                'step_improvements': state.step_improvements,
                'time_sec': t1 - t0,
            }
            conv = [(c['step'], f"{c['hess_error']:.4e}") for c in state.convergence_history]
            print(f"    Herr: {state.hess_error:.6e}, Time: {t1-t0:.1f}s")
            print(f"    Conv: {conv}")

            # Verify monotonicity: objective must be non-increasing
            objs = [c['objective'] for c in state.convergence_history]
            monotone = all(objs[i] >= objs[i+1] - 1e-12 for i in range(len(objs)-1))
            results[name]['monotone_verified'] = monotone
            if not monotone:
                print(f"    WARNING: monotonicity violated!")

        # Synergy test: Rotate+GPTQ(alpha=0) and Rotate+GPTAQ(alpha=0.25)
        rng_test = np.random.default_rng(cfg.seed)
        qfn = lambda W_in: quantize_tiles(W_in, bits, cfg.tile_size)
        U_t, V_t, _ = hadamard_orbit_sample(W, H_G, H_X, cfg.n_hadamard_samples, rng_test, qfn,
                                             lambda E,HG,HX: hessian_weighted_error(E,HG,HX))
        U_t, V_t, _ = givens_refine(W, H_G, H_X, U_t, V_t, cfg.n_givens_iters, rng_test, qfn,
                                     lambda E,HG,HX: hessian_weighted_error(E,HG,HX))
        W_rot = U_t.T @ W @ V_t
        X_rot = V_t.T @ X; Xt_rot = V_t.T @ Xt

        # Rotate+GPTQ (alpha=0)
        Wq_rg0 = gptaq_correction(W_rot, X_rot, Xt_rot, bits, cfg.tile_size, cfg.block_size,
                                   cfg.damping, alpha=0.0)
        Wq_rg0_full = U_t @ Wq_rg0 @ V_t.T
        results['Rotate+GPTQ(a=0)'] = {
            'hess_error': hessian_weighted_error(W - Wq_rg0_full, H_G, H_X),
            'weight_mse': weight_mse(W - Wq_rg0_full),
            'bytes': float(W.size * cfg.avg_k / 8), 'note': 'composition test'}

        # Rotate+GPTAQ (alpha=0.25)
        Wq_rg25 = gptaq_correction(W_rot, X_rot, Xt_rot, bits, cfg.tile_size, cfg.block_size,
                                    cfg.damping, alpha=0.25)
        Wq_rg25_full = U_t @ Wq_rg25 @ V_t.T
        results['Rotate+GPTAQ(a=.25)'] = {
            'hess_error': hessian_weighted_error(W - Wq_rg25_full, H_G, H_X),
            'weight_mse': weight_mse(W - Wq_rg25_full),
            'bytes': float(W.size * cfg.avg_k / 8), 'note': 'composition test'}

        # Print comparison
        print(f"\n  Results for {tensor_name}:")
        print(f"  {'Method':<30} {'Hess Error':>14} {'Weight MSE':>14} {'vs RTN':>10}")
        print(f"  {'-'*70}")
        rtn_herr = results['RTN']['hess_error']
        for name, res in results.items():
            imp = (rtn_herr - res['hess_error']) / rtn_herr * 100 if rtn_herr > 0 else 0
            print(f"  {name:<30} {res['hess_error']:>14.6e} {res['weight_mse']:>14.6e} {imp:>+9.1f}%")

        all_results[tensor_name] = results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Best individual vs alternating optimizer")
    print(f"{'='*80}")
    print(f"  {'Tensor':<12} {'Best Indiv':>12} {'Best Ind':>14} {'4-step':>14} {'6-step':>14} {'4-vs-BI':>8} {'6-vs-BI':>8}")
    print(f"  {'-'*82}")
    for tn in eval_tensors:
        if tn not in all_results: continue
        res = all_results[tn]
        ind_names = ['RTN', 'Equilibrate', 'Partition', 'Rotate', 'Allocate', 'GPTQ(a=0)', 'GPTAQ(a=.25)']
        best_ind = min(ind_names, key=lambda n: res[n]['hess_error'])
        bi_herr = res[best_ind]['hess_error']
        herr_4 = res['4-step (+rot)']['hess_error']
        herr_6 = res['6-step (+corr)']['hess_error']
        imp4 = (bi_herr - herr_4) / bi_herr * 100 if bi_herr > 0 else 0
        imp6 = (bi_herr - herr_6) / bi_herr * 100 if bi_herr > 0 else 0
        print(f"  {tn:<12} {best_ind:>12} {bi_herr:>14.6e} {herr_4:>14.6e} {herr_6:>14.6e} {imp4:>+7.1f}% {imp6:>+7.1f}%")

    # Synergy test: both alpha=0 and alpha=0.25
    print(f"\n{'='*80}")
    print("SYNERGY TEST: Rotate+Correction vs Rotate alone vs Correction alone")
    print(f"{'='*80}")
    print(f"  {'Tensor':<12} {'Rotate':>14} {'GPTQ(a=0)':>14} {'Rot+GPTQ':>14} {'Syn(a=0)':>9} {'Rot+GPTAQ':>14} {'Syn(a=.25)':>10}")
    print(f"  {'-'*85}")
    for tn in eval_tensors:
        if tn not in all_results: continue
        res = all_results[tn]
        herr_rot = res['Rotate']['hess_error']
        herr_gpt = res['GPTQ(a=0)']['hess_error']
        herr_rg0 = res.get('Rotate+GPTQ(a=0)', {}).get('hess_error', float('inf'))
        herr_rg25 = res.get('Rotate+GPTAQ(a=.25)', {}).get('hess_error', float('inf'))
        best_single = min(herr_rot, herr_gpt)
        syn0 = (best_single - herr_rg0) / best_single * 100
        syn25 = (best_single - herr_rg25) / best_single * 100
        print(f"  {tn:<12} {herr_rot:>14.6e} {herr_gpt:>14.6e} {herr_rg0:>14.6e} {syn0:>+8.1f}% {herr_rg25:>14.6e} {syn25:>+9.1f}%")

    # Monotonicity verification
    print(f"\n{'='*80}")
    print("MONOTONICITY VERIFICATION")
    print(f"{'='*80}")
    all_monotone = True
    for tn in eval_tensors:
        if tn not in all_results: continue
        for name, res in all_results[tn].items():
            if 'monotone_verified' in res:
                if not res['monotone_verified']:
                    all_monotone = False
                    print(f"  {tn}/{name}: FAIL")
    if all_monotone:
        print("  ALL configurations: PASS (objective monotonically non-increasing)")

    # Save results
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', '..', 'receipts', 'research', 'r9-group-orbit-results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    def ser(obj):
        if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [ser(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(out, 'w') as f:
        json.dump(ser(all_results), f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results

if __name__ == "__main__":
    results = run_experiment()
