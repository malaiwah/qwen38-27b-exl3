#!/usr/bin/env python3
"""
R11 — Unified Stack Factorial Experiment.

Builds and tests the full 9-step quantization stack as a unified pipeline with
matched byte budget. Combines:
  1. Scaling (lp_pinf from R8)
  2. BiIP diagonal balancing (from R3)
  3. Signed randomized Hadamard both sides (from R3)
  4. p99-scale permutation post-Hadamard (from R4)
  5. DP-refined tile allocation (from R1)
  6. Act-order GPTQ + GPTAQ correction (from R7, R2)
  7. Inverse transforms

Key requirements:
  - Matched per-tile 16×16 quantizer for ALL arms
  - EXACT byte budget including ALL sidecar costs
  - Hessian-weighted error tr(H_G E H_X E^T) as primary metric
  - HELD-OUT evaluation: separate calibration for scoring vs transform selection
  - Correct Cholesky: U = chol(inv(H+λI)).T (upper triangular, U^T U = H^{-1})
  - Block-size invariance test
  - GPTAQ-on ≠ GPTAQ-off sanity check
  - Accept-if-improve component selection
"""

import numpy as np
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

TILE = 16
SLICE_SIZE = 128
K_VALUES = [3, 4, 5, 6]
N_CALIB_SELECT = 512   # calibration for transform selection
N_CALIB_EVAL = 512     # held-out calibration for scoring
SEED = 42
DAMPING = 0.01
GPTAQ_ALPHA = 1.0      # paper-faithful (R2: 34/36 wins unrotated)
BLOCK_SIZE = 16         # GPTQ lazy block size
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = Path(__file__).parent.parent.parent.parent / "receipts" / "research" / "r11-unified-stack-results.json"

TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
SLICE_NAMES = ["first", "middle", "last"]

# ============================================================================
# Quantizer — per-tile 16×16 uniform, MATCHED for ALL arms
# ============================================================================

def quantize_tile(w: np.ndarray, k: int) -> np.ndarray:
    """Per-tile uniform quantizer. k bits → 2^k levels. Same for ALL arms."""
    nl = 2 ** k
    lo = float(w.min())
    hi = float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo)
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_matrix_uniform(W: np.ndarray, k: int, tile: int = TILE) -> np.ndarray:
    """Per-tile uniform quantization with uniform K."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            r1, c1 = min(i + tile, m), min(j + tile, n)
            Wq[i:r1, j:c1] = quantize_tile(W[i:r1, j:c1], k)
    return Wq


def quantize_matrix_alloc(W: np.ndarray, K_alloc: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Per-tile uniform quantization with per-tile K allocation."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            k = int(K_alloc[ti, tj])
            Wq[r0:r1, c0:c1] = quantize_tile(W[r0:r1, c0:c1], k)
    return Wq


# ============================================================================
# Cholesky — CORRECT convention: U = chol(inv(H+λI)).T, U^T U = H^{-1}
# ============================================================================

def inv_cholesky(H: np.ndarray, damping: float) -> np.ndarray:
    """Upper triangular U such that U^T @ U = inv(H + λI).
    Correct GPTQ convention: U = chol(inv(H+λI)).T (upper triangular).
    Damping is RELATIVE: λ = damping * mean(diag(H))."""
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
    return U


# ============================================================================
# Hadamard transform
# ============================================================================

def hadamard_matrix(n: int) -> np.ndarray:
    """Normalized Sylvester Hadamard matrix (n must be power of 2)."""
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
    return H / np.sqrt(n)


def signed_random_hadamard(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Signed randomized Hadamard: diag(±1) @ H_normalized."""
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return np.diag(signs) @ H, signs


# ============================================================================
# BiIP diagonal balancing (from R3)
# ============================================================================

def biip_scaling(W: np.ndarray, H_X: np.ndarray, H_G: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided diagonal balancing (KronQ Eq. 11).
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}, S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}
    W' = S_G @ W @ S_X
    Returns: S_G, S_X, W_transformed."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    S_G = np.diag(sg_diag)
    W_t = S_G @ W @ S_X
    return S_G, S_X, W_t


# ============================================================================
# Scaling — lp_pinf (from R8, best scaling method)
# ============================================================================

def lp_pinf_scales(W: np.ndarray, X: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Lp-norm p=∞ scaling: s_j = max|X_j|^alpha, normalized.
    Applied as W' = W * diag(s), X' = X / diag(s)."""
    a = np.max(np.abs(X), axis=1)  # (n,)
    r = np.power(np.clip(a, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    return s


# ============================================================================
# p99-scale permutation (from R4)
# ============================================================================

def perm_p99_scale(W: np.ndarray) -> np.ndarray:
    """Sort columns by p99 of |W| column values. Robust to outliers."""
    p99 = np.percentile(np.abs(W), 99, axis=0)
    return np.argsort(p99)


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    """Inverse permutation."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# ============================================================================
# Act-order (from R7)
# ============================================================================

def act_order_desc(H_X: np.ndarray) -> np.ndarray:
    """Descending diag(H_X) column order. Most important channels first."""
    return np.argsort(np.diag(H_X))[::-1]


# ============================================================================
# DP tile allocation (from R1)
# ============================================================================

def measure_tile_distortion(W_tile: np.ndarray, k: int, H_G_sub: np.ndarray, H_X_sub: np.ndarray) -> float:
    """Hessian-weighted distortion of a tile at K=k."""
    Wq = quantize_tile(W_tile, k)
    E = W_tile - Wq
    D = float(np.trace(H_G_sub @ E @ H_X_sub @ E.T))
    return max(D, 0.0)


def alloc_tile_local_dp(W: np.ndarray, H_X: np.ndarray, H_G: np.ndarray,
                        k_sum_budget: int, tile: int = TILE,
                        k_range=(2, 3, 4, 5, 6)) -> np.ndarray:
    """Tile-local DP: exact solver for additive tile-local surrogate.
    Multiple-choice knapsack: min sum D_{t,k} s.t. sum k_t <= budget."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    n_tiles = n_tm * n_tn

    # Build distortion table
    k_list = list(k_range)
    D_table = np.zeros((n_tiles, len(k_list)))
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            t_idx = ti * n_tn + tj
            for ki, k in enumerate(k_list):
                D_table[t_idx, ki] = measure_tile_distortion(
                    W[r0:r1, c0:c1], k,
                    H_G[r0:r1, r0:r1], H_X[c0:c1, c0:c1])

    # DP: minimize total distortion subject to K-sum budget
    # dp[j][s] = min distortion using first j tiles with total K-sum = s
    max_sum = k_sum_budget
    INF = 1e18
    dp = np.full((n_tiles + 1, max_sum + 1), INF)
    choice = np.zeros((n_tiles + 1, max_sum + 1), dtype=int)
    dp[0, 0] = 0.0

    for t in range(n_tiles):
        for s in range(max_sum + 1):
            if dp[t, s] >= INF:
                continue
            for ki, k in enumerate(k_list):
                new_s = s + k
                if new_s <= max_sum:
                    new_d = dp[t, s] + D_table[t, ki]
                    if new_d < dp[t + 1, new_s]:
                        dp[t + 1, new_s] = new_d
                        choice[t + 1, new_s] = ki

    # Find best feasible K-sum (closest to budget from below)
    best_s = max_sum
    while best_s >= 0 and dp[n_tiles, best_s] >= INF:
        best_s -= 1
    if best_s < 0:
        # Fallback: uniform
        avg_k = k_sum_budget // n_tiles
        return np.full((n_tm, n_tn), max(min(avg_k, max(k_range)), min(k_range)))

    # Backtrack
    K_flat = np.zeros(n_tiles, dtype=int)
    s = best_s
    for t in range(n_tiles, 0, -1):
        ki = choice[t, s]
        K_flat[t - 1] = k_list[ki]
        s -= k_list[ki]

    return K_flat.reshape(n_tm, n_tn)


def local_search_refine(W: np.ndarray, K_alloc: np.ndarray, H_G: np.ndarray, H_X: np.ndarray,
                        k_sum_budget: int, tile: int = TILE,
                        k_range=(2, 3, 4, 5, 6), max_iters: int = 500) -> np.ndarray:
    m, n = W.shape
    n_tm, n_tn = K_alloc.shape
    n_tiles = n_tm * n_tn
    K_flat = K_alloc.flatten().copy()

    # Compute current per-tile distortion
    def tile_dist(ti, tj, k):
        r0, c0 = ti * tile, tj * tile
        r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
        return measure_tile_distortion(W[r0:r1, c0:c1], k,
                                        H_G[r0:r1, r0:r1], H_X[c0:c1, c0:c1])

    dists = np.array([tile_dist(ti, tj, K_flat[ti * n_tn + tj])
                      for ti in range(n_tm) for tj in range(n_tn)])
    current_total = float(np.sum(dists))

    for _ in range(max_iters):
        improved = False
        # Try all pairs: move 1 bit from tile A to tile B
        for a in range(n_tiles):
            if K_flat[a] <= min(k_range):
                continue
            for b in range(n_tiles):
                if a == b or K_flat[b] >= max(k_range):
                    continue
                # Transfer 1 bit from a to b
                ka_new = K_flat[a] - 1
                kb_new = K_flat[b] + 1
                ti_a, tj_a = a // n_tn, a % n_tn
                ti_b, tj_b = b // n_tn, b % n_tn
                new_dist_a = tile_dist(ti_a, tj_a, ka_new)
                new_dist_b = tile_dist(ti_b, tj_b, kb_new)
                delta = (new_dist_a + new_dist_b) - (dists[a] + dists[b])
                if delta < -1e-15:
                    K_flat[a] = ka_new
                    K_flat[b] = kb_new
                    dists[a] = new_dist_a
                    dists[b] = new_dist_b
                    current_total += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return K_flat.reshape(n_tm, n_tn)


def alloc_uniform_matched(k_sum_budget: int, n_tm: int, n_tn: int,
                          k_range=(2, 3, 4, 5, 6)) -> np.ndarray:
    """Uniform allocation with K-sum matched to budget.
    Most tiles get K, some get K-1 to match budget exactly."""
    n_tiles = n_tm * n_tn
    avg_k = k_sum_budget / n_tiles
    k_base = int(np.floor(avg_k))
    k_base = max(min(k_range), min(k_base, max(k_range)))
    remainder = k_sum_budget - k_base * n_tiles
    K_flat = np.full(n_tiles, k_base, dtype=int)
    # Distribute remainder: give +1 to 'remainder' tiles
    for i in range(remainder):
        if K_flat[i] < max(k_range):
            K_flat[i] += 1
    return K_flat.reshape(n_tm, n_tn)


# ============================================================================
# GPTQ + GPTAQ correction (from R9/R2, correct Cholesky)
# ============================================================================

def gptq_gptaq_quantize(W: np.ndarray, X: np.ndarray, Xt: np.ndarray,
                        K_alloc: np.ndarray, tile: int = TILE,
                        damping: float = DAMPING,
                        alpha: float = GPTAQ_ALPHA, use_gptaq: bool = True) -> np.ndarray:
    """GPTQ with optional GPTAQ P-matrix correction.
    Uses MATCHED per-tile 16×16 quantizer: codebooks frozen per column-tile
    from CURRENT Ww (reflecting GPTQ updates from prior tiles).

    Processes column-tiles left-to-right. Within each column-tile, freezes
    codebooks, then processes columns sequentially with GPTQ error propagation
    and optional GPTAQ P-matrix correction.

    W: (m, n). X: (n, N) quant-flow. Xt: (n, N) FP-flow.
    K_alloc: (n_tm, n_tn) per-tile K values.
    """
    m, n = W.shape
    Ww = W.copy().astype(np.float64)
    Q = np.zeros_like(Ww)
    H = X @ X.T  # n×n input Hessian (in transformed space)
    L = inv_cholesky(H, damping)  # U^T U = inv(H+λI), upper triangular

    # P-matrix for GPTAQ
    if use_gptaq and alpha > 0:
        dX = Xt - X
        D = dX @ X.T
        P = alpha * (np.triu(D @ L.T, 1) @ L)
    else:
        P = np.zeros((n, n))

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    # Process column-tiles left-to-right (natural Cholesky order)
    tile_order = list(range(n_tn))

    processed_tiles = set()

    for tj in tile_order:
        c0 = tj * tile
        c1 = min(c0 + tile, n)
        B = c1 - c0

        # Freeze codebooks for THIS column-tile from CURRENT Ww
        # (reflects GPTQ updates from all previously processed tiles)
        codebooks = []
        for ti in range(n_tm):
            r0 = ti * tile
            r1 = min(r0 + tile, m)
            k = int(K_alloc[ti, tj])
            tile_data = Ww[r0:r1, c0:c1]
            nl = 2 ** k
            lo = float(tile_data.min())
            hi = float(tile_data.max())
            step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
            codebooks.append((r0, r1, lo, step, k))

        def apply_frozen_cb(col_data: np.ndarray, cbs: list) -> np.ndarray:
            q = np.zeros_like(col_data)
            for r0, r1, lo, step, k in cbs:
                if step == 0.0:
                    q[r0:r1] = col_data[r0:r1]
                else:
                    nl = 2 ** k
                    q[r0:r1] = np.clip(np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
            return q

        # GPTQ within this column-tile (columns processed left-to-right)
        E = np.zeros((m, B))
        W_pre_block = np.zeros((m, B))
        for j in range(B):
            c = c0 + j
            w_pre = Ww[:, c].copy()
            W_pre_block[:, j] = w_pre

            # Quantize using frozen codebook for this column-tile
            Q[:, c] = apply_frozen_cb(w_pre, codebooks)

            # GPTQ error propagation
            e = w_pre - Q[:, c]
            E[:, j] = e / L[c, c]
            # Propagate to remaining columns within this tile
            end = min(c0 + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            # GPTAQ correction within tile
            if use_gptaq and alpha > 0:
                Ww[:, c:end] += np.outer(w_pre, P[c, c:end])

        # Outer lazy block propagation to ALL unprocessed column-tiles
        unprocessed_cols = []
        for tj2 in tile_order:
            if tj2 not in processed_tiles and tj2 != tj:
                c0_2 = tj2 * tile
                c1_2 = min(c0_2 + tile, n)
                unprocessed_cols.extend(range(c0_2, c1_2))

        if unprocessed_cols:
            unprocessed_cols = np.array(unprocessed_cols)
            tile_cols = np.arange(c0, c1)
            L_block = L[np.ix_(tile_cols, unprocessed_cols)]
            Ww[:, unprocessed_cols] -= E @ L_block
            if use_gptaq and alpha > 0:
                P_block = P[np.ix_(tile_cols, unprocessed_cols)]
                Ww[:, unprocessed_cols] += W_pre_block @ P_block

        processed_tiles.add(tj)

    return Q


@dataclass
class StackConfig:
    """Configuration for the unified stack. Each component can be on/off."""
    use_scaling: bool = False       # Step 1: lp_pinf scaling
    use_biip: bool = False          # Step 2: BiIP diagonal balancing
    use_hadamard: bool = False      # Step 3: signed randomized Hadamard both sides
    use_permutation: bool = False   # Step 4: p99-scale permutation
    use_dp_alloc: bool = False      # Step 5: DP-refined tile allocation
    use_gptq: bool = False          # Step 6a: GPTQ error propagation
    use_gptaq: bool = False         # Step 6b: GPTAQ P-matrix correction
    gptaq_alpha: float = 1.0       # GPTAQ strength (0=GPTQ only, 0.25=modest, 1.0=paper-faithful)

    def name(self) -> str:
        parts = []
        if self.use_scaling: parts.append("scale")
        if self.use_biip: parts.append("biip")
        if self.use_hadamard: parts.append("had")
        if self.use_permutation: parts.append("perm")
        if self.use_dp_alloc: parts.append("dp")
        if self.use_gptq: parts.append("gptq")
        if self.use_gptaq and self.gptaq_alpha > 0: parts.append(f"gptaq_a{self.gptaq_alpha}")
        return "+".join(parts) if parts else "none"

@dataclass
class TransformState:
    """State of transforms applied (for inverse)."""
    s_scale: Optional[np.ndarray] = None     # lp_pinf scales (n,)
    S_G: Optional[np.ndarray] = None         # BiIP output scale matrix (m, m)
    S_X: Optional[np.ndarray] = None         # BiIP input scale matrix (n, n)
    U: Optional[np.ndarray] = None           # Output Hadamard (m, m)
    V: Optional[np.ndarray] = None           # Input Hadamard (n, n)
    perm_col: Optional[np.ndarray] = None   # Column permutation (n,)
    sidecar_bytes: int = 0


def apply_forward_transforms(W: np.ndarray, X: np.ndarray, Xt: np.ndarray,
                             H_X: np.ndarray, H_G: np.ndarray,
                             cfg: StackConfig, rng: np.random.Generator,
                             precomputed_rotations: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformState]:
    """Apply forward transforms. Returns (W', X', Xt', H_X', H_G', state)."""
    m, n = W.shape
    state = TransformState()
    W_t = W.copy().astype(np.float64)
    X_t = X.copy().astype(np.float64)
    Xt_t = Xt.copy().astype(np.float64)
    H_X_t = H_X.copy()
    H_G_t = H_G.copy()

    # Step 1: lp_pinf scaling
    if cfg.use_scaling:
        s = lp_pinf_scales(W_t, X_t, alpha=0.5)
        state.s_scale = s
        W_t = W_t * s[None, :]
        X_t = X_t / s[:, None]
        Xt_t = Xt_t / s[:, None]
        H_X_t = np.diag(1.0 / s) @ H_X_t @ np.diag(1.0 / s)
        # H_G unchanged (output dim not scaled)
        # Sidecar: n float16 values
        state.sidecar_bytes += n * 2

    # Step 2: BiIP diagonal balancing
    if cfg.use_biip:
        S_G, S_X, W_t = biip_scaling(W_t, H_X_t, H_G_t)
        state.S_G = S_G
        state.S_X = S_X
        S_G_inv = np.linalg.inv(S_G)
        S_X_inv = np.linalg.inv(S_X)
        H_X_t = S_X_inv @ H_X_t @ S_X_inv
        H_G_t = S_G_inv @ H_G_t @ S_G_inv
        # Transform calibration data: X' = S_X^{-1} @ X so GPTQ H = X' @ X'^T matches H_X_t
        X_t = S_X_inv @ X_t
        Xt_t = S_X_inv @ Xt_t
        # Sidecar: (m + n) float16 diagonal scales
        state.sidecar_bytes += (m + n) * 2
    # Step 3: Signed randomized Hadamard both sides
    if cfg.use_hadamard:
        if precomputed_rotations is not None:
            U, V = precomputed_rotations
        else:
            U, _ = signed_random_hadamard(m, rng)
            V, _ = signed_random_hadamard(n, rng)
        state.U = U
        state.V = V
        W_t = U @ W_t @ V.T
        H_X_t = V @ H_X_t @ V.T
        H_G_t = U @ H_G_t @ U.T
        # Transform calibration data: X' = V @ X so GPTQ H = X' @ X'^T matches H_X_t
        X_t = V @ X_t
        Xt_t = V @ Xt_t
        # Sidecar: (m + n) sign bits = (m + n) / 8 bytes
        state.sidecar_bytes += (m + n + 7) // 8

    # Step 4: p99-scale permutation (columns only)
    if cfg.use_permutation:
        perm = perm_p99_scale(W_t)
        state.perm_col = perm
        W_t = W_t[:, perm]
        H_X_t = H_X_t[np.ix_(perm, perm)]
        X_t = X_t[perm, :]
        Xt_t = Xt_t[perm, :]
        # Sidecar: n * ceil(log2(n)) bits
        bits_per_idx = int(np.ceil(np.log2(n)))
        state.sidecar_bytes += (n * bits_per_idx + 7) // 8

    return W_t, X_t, Xt_t, H_X_t, H_G_t, state


def apply_inverse_transforms(Q_t: np.ndarray, state: TransformState) -> np.ndarray:
    """Inverse transforms to get back to original space."""
    Q = Q_t.copy().astype(np.float64)

    # Undo permutation
    if state.perm_col is not None:
        inv_perm = inverse_perm(state.perm_col)
        Q = Q[:, inv_perm]

    # Undo Hadamard
    if state.U is not None and state.V is not None:
        Q = state.U.T @ Q @ state.V

    # Undo BiIP
    if state.S_G is not None and state.S_X is not None:
        S_G_inv = np.linalg.inv(state.S_G)
        S_X_inv = np.linalg.inv(state.S_X)
        Q = S_G_inv @ Q @ S_X_inv

    # Undo scaling
    if state.s_scale is not None:
        Q = Q / state.s_scale[None, :]

    return Q


# ============================================================================
# Byte budget accounting
# ============================================================================

def compute_total_bytes_v2(K_alloc: np.ndarray, state: TransformState,
                           codebook_bytes: int, k_meta_bytes: int, tile: int = TILE) -> dict:
    """Exact byte count: payload + codebook metadata + sidecar + K-metadata."""
    m_tiles, n_tiles = K_alloc.shape
    elements_per_tile = tile * tile

    # Payload: sum of K * elements per tile
    payload_bits = int(np.sum(K_alloc.flatten() * elements_per_tile))
    payload_bytes = payload_bits / 8.0

    # Sidecar (from TransformState)
    sidecar_bytes = state.sidecar_bytes

    total_bytes = payload_bytes + sidecar_bytes + codebook_bytes + k_meta_bytes

    return {
        "payload_bytes": payload_bytes,
        "sidecar_bytes": sidecar_bytes,
        "codebook_bytes": codebook_bytes,
        "k_meta_bytes": k_meta_bytes,
        "total_bytes": total_bytes,
        "bits_per_element": total_bytes * 8 / (m_tiles * tile * n_tiles * tile)
    }


def compute_total_bytes(K_alloc: np.ndarray, state: TransformState, tile: int = TILE) -> dict:
    """Legacy byte count (without codebook metadata)."""
    m_tiles, n_tiles = K_alloc.shape
    elements_per_tile = tile * tile
    payload_bits = int(np.sum(K_alloc.flatten() * elements_per_tile))
    payload_bytes = payload_bits / 8.0
    sidecar_bytes = state.sidecar_bytes
    k_meta_bytes = m_tiles * n_tiles
    total_bytes = payload_bytes + sidecar_bytes + k_meta_bytes
    return {
        "payload_bytes": payload_bytes,
        "sidecar_bytes": sidecar_bytes,
        "k_meta_bytes": k_meta_bytes,
        "total_bytes": total_bytes,
        "bits_per_element": total_bytes * 8 / (m_tiles * tile * n_tiles * tile)
    }


def k_sum_budget_for_target(target_bytes: float, sidecar_bytes: int, k_meta_bytes: int,
                            n_tiles: int, elements_per_tile: int,
                            codebook_bytes: int = 0) -> int:
    """Compute K-sum budget to match target total bytes.
    target_bytes is the TOTAL budget (payload + codebook + sidecar + k_meta).
    Returns the K-sum such that payload + codebook + sidecar + k_meta ≤ target."""
    available_payload_bytes = target_bytes - sidecar_bytes - k_meta_bytes - codebook_bytes
    available_payload_bits = int(available_payload_bytes * 8)
    k_sum = available_payload_bits // elements_per_tile
    return max(k_sum, n_tiles * 2)  # at least K=2 per tile (allows sidecar at low K)


# ============================================================================
# Metrics
# ============================================================================

def hessian_weighted_error(W: np.ndarray, Wq: np.ndarray, H_X: np.ndarray, H_G: np.ndarray) -> float:
    """Primary metric: tr(H_G @ E @ H_X @ E^T) where E = W - Wq."""
    E = W - Wq
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(W: np.ndarray, Wq: np.ndarray) -> float:
    """Raw weight MSE (secondary)."""
    return float(np.mean((W - Wq) ** 2))


# ============================================================================
# Calibration data generation
# ============================================================================

def gen_calibration_distribution(n_in: int, seed: int) -> dict:
    """Generate shared distribution parameters (scales, outliers, correlation).
    These are fixed across in-sample/held-out splits to simulate real model
    activations where the distribution is consistent across calibration batches."""
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.5, 3.0, size=n_in)
    n_outliers = max(1, n_in // 20)
    outlier_idx = rng.choice(n_in, n_outliers, replace=False)
    scales[outlier_idx] *= rng.uniform(5.0, 15.0, size=n_outliers)
    corr = rng.standard_normal((n_in, n_in))
    corr = corr @ corr.T / n_in
    return {"scales": scales, "outlier_idx": outlier_idx, "corr": corr}


def gen_calibration_from_dist(n_in: int, n_samples: int, dist: dict, seed: int) -> np.ndarray:
    """Generate calibration samples from a shared distribution.
    Different seeds → different samples, same distribution structure."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples))
    X = X * dist["scales"][:, None]
    X = dist["corr"] @ X
    return X


def gen_calibration_pair_from_dist(n_in: int, n_samples: int, dist: dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate X (quant-flow) and Xt (FP-flow) from shared distribution.
    Xt has small drift from X (simulating activation quantization)."""
    X = gen_calibration_from_dist(n_in, n_samples, dist, seed)
    rng = np.random.default_rng(seed + 1000)
    drift = rng.standard_normal(X.shape) * 0.01 * np.std(X)
    Xt = X + drift
    return X, Xt


def gen_calibration(n_in: int, n_samples: int, seed: int) -> np.ndarray:
    """Legacy: standalone calibration with per-channel scale variation + correlations."""
    dist = gen_calibration_distribution(n_in, seed)
    return gen_calibration_from_dist(n_in, n_samples, dist, seed)


def gen_calibration_pair(n_in: int, n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy: standalone calibration pair."""
    X = gen_calibration(n_in, n_samples, seed)
    rng = np.random.default_rng(seed + 1000)
    drift = rng.standard_normal(X.shape) * 0.01 * np.std(X)
    Xt = X + drift
    return X, Xt


def compute_hessians(W: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute input Hessian H_X = X X^T / N and output Hessian proxy H_G = Y^T Y / N."""
    N = X.shape[1]
    H_X = X @ X.T / N
    Y = W @ X  # (m, N)
    H_G = Y @ Y.T / N
    return H_X, H_G


# ============================================================================
# Real weight loading
# ============================================================================

def load_real_weights() -> Dict[str, np.ndarray]:
    """Load real Qwen3.8-27B BF16 weights."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for name in TENSOR_NAMES:
        if name in data.files:
            tensors[name] = data[name].astype(np.float64)
    return tensors


def extract_slice(tensor: np.ndarray, slice_name: str, size: int = SLICE_SIZE) -> np.ndarray:
    """Extract a 128×128 slice from a large tensor."""
    m, n = tensor.shape
    if slice_name == "first":
        return tensor[:size, :size].astype(np.float64)
    elif slice_name == "middle":
        r0, c0 = (m - size) // 2, (n - size) // 2
        return tensor[r0:r0+size, c0:c0+size].astype(np.float64)
    elif slice_name == "last":
        return tensor[-size:, -size:].astype(np.float64)
    else:
        raise ValueError(f"Unknown slice: {slice_name}")


# ============================================================================
# Run a single arm
# ============================================================================

def run_arm(W: np.ndarray, X_select: np.ndarray, Xt_select: np.ndarray,
            X_eval: np.ndarray, H_X_eval: np.ndarray, H_G_eval: np.ndarray,
            cfg: StackConfig, K: int, rng: np.random.Generator,
            precomputed_rotations: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
    """Run one arm of the experiment.

    X_select/Xt_select: calibration for transform selection and GPTQ Hessian.
    X_eval, H_X_eval, H_G_eval: held-out evaluation data (for scoring only).
    precomputed_rotations: (U, V) Hadamard matrices shared across arms for fair comparison.
    """
    m, n = W.shape
    n_tm = (m + TILE - 1) // TILE
    n_tn = (n + TILE - 1) // TILE
    n_tiles = n_tm * n_tn
    elements_per_tile = TILE * TILE

    # Compute Hessians for transform selection
    H_X_sel, H_G_sel = compute_hessians(W, X_select)

    # Apply forward transforms (using selection calibration)
    W_t, X_t, Xt_t, H_X_t, H_G_t, state = apply_forward_transforms(
        W, X_select, Xt_select, H_X_sel, H_G_sel, cfg, rng, precomputed_rotations)

    # Byte budget: target includes codebook metadata (common to all arms)
    # Codebook metadata: 2 float16 per tile (lo, step) = 4 bytes per tile
    codebook_bytes = n_tiles * 4
    # K-metadata: 1 byte per tile (only needed for mixed-K allocation arms)
    k_meta_bytes = n_tiles if cfg.use_dp_alloc else 0
    # Target: uniform-K baseline total = payload + codebook metadata
    target_bytes = m * n * K / 8.0 + codebook_bytes

    # Compute K allocation
    k_sum = k_sum_budget_for_target(target_bytes, state.sidecar_bytes, k_meta_bytes,
                                    n_tiles, elements_per_tile, codebook_bytes)

    if cfg.use_dp_alloc:
        # DP allocation + tile-local local search refinement
        K_alloc = alloc_tile_local_dp(W_t, H_X_t, H_G_t, k_sum, tile=TILE)
        K_alloc = local_search_refine(W_t, K_alloc, H_G_t, H_X_t, k_sum, tile=TILE, max_iters=200)
    else:
        # Uniform allocation matched to budget (accounting for sidecar)
        K_alloc = alloc_uniform_matched(k_sum, n_tm, n_tn)

    # Quantize
    if cfg.use_gptq or cfg.use_gptaq:
        Q_t = gptq_gptaq_quantize(
            W_t, X_t, Xt_t, K_alloc, tile=TILE,
            damping=DAMPING, alpha=cfg.gptaq_alpha if cfg.use_gptaq else 0.0,
            use_gptaq=cfg.use_gptaq)
    else:
        # RTN: simple per-tile quantization
        Q_t = quantize_matrix_alloc(W_t, K_alloc, tile=TILE)

    # Inverse transforms
    Q = apply_inverse_transforms(Q_t, state)

    # Evaluate with HELD-OUT Hessians (primary metric)
    hwe = hessian_weighted_error(W, Q, H_X_eval, H_G_eval)
    # Also compute in-sample HWE for overfitting diagnosis
    hwe_insample = hessian_weighted_error(W, Q, H_X_sel, H_G_sel)
    mse = weight_mse(W, Q)

    # Byte accounting
    bytes_info = compute_total_bytes_v2(K_alloc, state, codebook_bytes, k_meta_bytes, tile=TILE)

    return {
        "config": cfg.name(),
        "K": K,
        "hwe": hwe,
        "hwe_insample": hwe_insample,
        "overfitting_ratio": hwe_insample / hwe if hwe > 1e-15 else 1.0,
        "mse": mse,
        "bytes": bytes_info,
        "K_alloc": K_alloc.tolist(),
    }


# ============================================================================
# Sanity checks
# ============================================================================

def sanity_check_gptaq_not_noop():
    """Verify GPTAQ-on ≠ GPTAQ-off (unlike the Cholesky bug)."""
    rng = np.random.default_rng(42)
    m, n = 128, 128
    W = rng.standard_normal((m, n)) * 0.1
    X, Xt = gen_calibration_pair(n, 256, seed=42)
    K = 5

    K_alloc = np.full((8, 8), K)

    # GPTQ only (no GPTAQ)
    Q_gptq = gptq_gptaq_quantize(W, X, Xt, K_alloc, use_gptaq=False)
    # GPTQ + GPTAQ
    Q_gptaq = gptq_gptaq_quantize(W, X, Xt, K_alloc, use_gptaq=True)

    H_X, H_G = compute_hessians(W, X)
    hwe_gptq = hessian_weighted_error(W, Q_gptq, H_X, H_G)
    hwe_gptaq = hessian_weighted_error(W, Q_gptaq, H_X, H_G)

    diff = abs(hwe_gptq - hwe_gptaq)
    print(f"  GPTAQ-on ≠ GPTAQ-off: HWE(GPTQ)={hwe_gptq:.6e}, HWE(GPTAQ)={hwe_gptaq:.6e}, diff={diff:.6e}")
    assert diff > 1e-12, f"GPTAQ appears to be a no-op! diff={diff}"
    print("  PASS: GPTAQ correction is active (not a no-op)")
    return True


def sanity_check_reproducibility():
    """Verify deterministic results with same seed."""
    rng = np.random.default_rng(42)
    m, n = 128, 128
    W = rng.standard_normal((m, n)) * 0.1
    X, Xt = gen_calibration_pair(n, 256, seed=42)
    K = 5
    K_alloc = np.full((8, 8), K)

    Q1 = gptq_gptaq_quantize(W, X, Xt, K_alloc, use_gptaq=True)
    Q2 = gptq_gptaq_quantize(W, X, Xt, K_alloc, use_gptaq=True)

    assert np.allclose(Q1, Q2), "GPTQ is not deterministic!"
    print("  PASS: GPTQ results are deterministic (reproducible)")
    return True


def sanity_check_cholesky_convention():
    """Verify the Cholesky convention U^T U = inv(H+λI)."""
    rng = np.random.default_rng(42)
    n = 64
    H = rng.standard_normal((n, n))
    H = H @ H.T / n  # SPD
    U = inv_cholesky(H, 0.01)

    # Check upper triangular
    assert np.allclose(np.tril(U, -1), 0), "U is NOT upper triangular!"

    # Check U^T U = inv(H + λI)
    lam = max(0.01 * np.mean(np.diag(H)), 1e-10)
    Hinv = np.linalg.inv(H + lam * np.eye(n))
    assert np.allclose(U.T @ U, Hinv, atol=1e-10), "U^T U != inv(H+λI)!"

    print("  PASS: Cholesky convention correct (U^T U = inv(H+λI), upper triangular)")
    return True


# ============================================================================
# Accept-if-improve analysis
# ============================================================================

def run_accept_if_improve(W: np.ndarray, X_select: np.ndarray, Xt_select: np.ndarray,
                          X_eval: np.ndarray, H_X_eval: np.ndarray, H_G_eval: np.ndarray,
                          K: int, rng: np.random.Generator,
                          precomputed_rotations: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
    """Greedy forward selection: add each component if it improves HWE."""
    components = [
        ("biip", StackConfig(use_biip=True)),
        ("hadamard", StackConfig(use_biip=True, use_hadamard=True)),
        ("permutation", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True)),
        ("dp_alloc", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True)),
        ("gptq", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True, use_gptq=True)),
        ("gptaq_a1", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True, use_gptq=True, use_gptaq=True, gptaq_alpha=1.0)),
        ("scaling", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True, use_gptq=True, use_scaling=True)),
    ]

    results = {}
    prev_hwe = None
    accepted = []
    rejected = []

    for name, cfg in components:
        r = run_arm(W, X_select, Xt_select, X_eval, H_X_eval, H_G_eval, cfg, K, rng,
                    precomputed_rotations)
        results[name] = r
        if prev_hwe is None or r["hwe"] < prev_hwe:
            accepted.append(name)
            prev_hwe = r["hwe"]
            print(f"    ACCEPT {name}: HWE={r['hwe']:.6e} (total_bytes={r['bytes']['total_bytes']:.1f})")
        else:
            rejected.append(name)
            print(f"    REJECT {name}: HWE={r['hwe']:.6e} > prev={prev_hwe:.6e}")

    return {"accepted": accepted, "rejected": rejected, "results": results}


# ============================================================================
# Experiment configs
# ============================================================================

def get_configs() -> List[Tuple[str, StackConfig]]:
    """Define the factorial experiment configs.

    Key design decisions based on Wave 1 + R14 findings:
    - GPTAQ α=1.0 is paper-faithful for UNROTATED weights (R2: 34/36 wins)
    - GPTAQ α=0 (pure GPTQ) is best post-rotation (R14: α=1.0 harmful, α=0 best)
    - Act-order requires full column permutation + reordered Cholesky (not implemented)
    - So: all configs use left-to-right column processing (natural Cholesky order)
    """
    return [
        # Baselines
        ("none", StackConfig()),
        ("scaling_only", StackConfig(use_scaling=True)),
        ("biip_only", StackConfig(use_biip=True)),
        ("rotation_only", StackConfig(use_biip=True, use_hadamard=True)),
        ("rotation_perm", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True)),
        ("allocation_only", StackConfig(use_dp_alloc=True)),

        # Correction variants (unrotated)
        ("gptq_only", StackConfig(use_gptq=True)),
        ("gptaq_only", StackConfig(use_gptq=True, use_gptaq=True, gptaq_alpha=1.0)),
        ("scaling_gptq", StackConfig(use_scaling=True, use_gptq=True)),
        ("allocation_gptq", StackConfig(use_dp_alloc=True, use_gptq=True)),

        # Rotation + GPTQ (α=0 per R14)
        ("rotation_gptq", StackConfig(use_biip=True, use_hadamard=True, use_gptq=True)),
        ("rotation_perm_gptq", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_gptq=True)),
        ("rotation_allocation_gptq", StackConfig(use_biip=True, use_hadamard=True, use_dp_alloc=True, use_gptq=True)),

        # Rotation + GPTAQ α=1.0 (to show it's harmful post-rotation)
        ("rotation_gptaq_a1", StackConfig(use_biip=True, use_hadamard=True, use_gptq=True, use_gptaq=True, gptaq_alpha=1.0)),

        # Combinations without correction
        ("rotation_allocation", StackConfig(use_biip=True, use_hadamard=True, use_dp_alloc=True)),
        ("rotation_perm_allocation", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True)),

        # Full stacks
        ("full_stack_no_scaling", StackConfig(use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True, use_gptq=True)),
        ("full_stack_no_correction", StackConfig(use_scaling=True, use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True)),
        ("full_stack", StackConfig(use_scaling=True, use_biip=True, use_hadamard=True, use_permutation=True, use_dp_alloc=True, use_gptq=True)),
    ]


# ============================================================================
# Main experiment
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 100)
    print("R11 — Unified Stack Factorial Experiment")
    print("=" * 100)

    # ─── Sanity checks ───
    print("\n--- Sanity Checks ---")
    sanity_check_cholesky_convention()
    sanity_check_gptaq_not_noop()
    sanity_check_reproducibility()

    # ─── Load weights ───
    print("\n--- Loading real weights ---")
    tensors = load_real_weights()
    for name, t in tensors.items():
        print(f"  {name}: {t.shape}")

    # ─── Run factorial experiment ───
    configs = get_configs()
    print(f"\n--- Running {len(configs)} configs × {len(TENSOR_NAMES)} tensors × {len(SLICE_NAMES)} slices × {len(K_VALUES)} K values ---")

    all_results = {}
    noise_floor_results = {}
    budget_violations = []

    for tensor_name in TENSOR_NAMES:
        if tensor_name not in tensors:
            continue
        tensor = tensors[tensor_name]
        all_results[tensor_name] = {}
        noise_floor_results[tensor_name] = {}

        for slice_idx, slice_name in enumerate(SLICE_NAMES):
            W = extract_slice(tensor, slice_name, SLICE_SIZE)
            m, n = W.shape
            print(f"\n  {tensor_name} / {slice_name} ({m}×{n})")

            # Stable seed derivation (not hash() which is randomized per process)
            seed_base = SEED + slice_idx * 100 + TENSOR_NAMES.index(tensor_name) * 10000
            dist = gen_calibration_distribution(n, seed_base)
            X_select, Xt_select = gen_calibration_pair_from_dist(n, N_CALIB_SELECT, dist, seed=seed_base)
            X_eval, _ = gen_calibration_pair_from_dist(n, N_CALIB_EVAL, dist, seed=seed_base + 777)
            H_X_eval, H_G_eval = compute_hessians(W, X_eval)

            # Noise floor: HWE when Wq = W (no quantization error) → exactly 0
            nf = hessian_weighted_error(W, W, H_X_eval, H_G_eval)
            noise_floor_results[tensor_name][slice_name] = nf

            # Precompute Hadamard rotations ONCE per slice (shared across all arms)
            rot_rng = np.random.default_rng(seed_base + 4242)
            U_pre, _ = signed_random_hadamard(m, rot_rng)
            V_pre, _ = signed_random_hadamard(n, rot_rng)
            precomputed_rotations = (U_pre, V_pre)

            all_results[tensor_name][slice_name] = {}

            for K in K_VALUES:
                all_results[tensor_name][slice_name][K] = {}
                rng = np.random.default_rng(SEED)

                for cfg_name, cfg in configs:
                    t0 = time.time()
                    result = run_arm(W, X_select, Xt_select, X_eval, H_X_eval, H_G_eval,
                                     cfg, K, rng, precomputed_rotations)
                    elapsed = time.time() - t0

                    # Byte budget assertion
                    target_bytes = m * n * K / 8.0 + (m // TILE) * (n // TILE) * 4  # payload + codebook
                    if result["bytes"]["total_bytes"] > target_bytes + 0.5:
                        budget_violations.append((tensor_name, slice_name, K, cfg_name,
                                                  result["bytes"]["total_bytes"], target_bytes))

                    all_results[tensor_name][slice_name][K][cfg_name] = result
                    if elapsed > 1.0:
                        print(f"    K={K} {cfg_name}: HWE={result['hwe']:.6e} bytes={result['bytes']['total_bytes']:.1f} ({elapsed:.1f}s)")

    # ─── Accept-if-improve analysis ───
    print("\n--- Accept-if-Improve Analysis (K=5, L0_gate, first slice) ---")
    W_aif = extract_slice(tensors["L0_gate"], "first", SLICE_SIZE)
    m_aif, n_aif = W_aif.shape
    dist_aif = gen_calibration_distribution(n_aif, 42)
    X_sel_aif, Xt_sel_aif = gen_calibration_pair_from_dist(n_aif, N_CALIB_SELECT, dist_aif, seed=42)
    X_eval_aif, _ = gen_calibration_pair_from_dist(n_aif, N_CALIB_EVAL, dist_aif, seed=42 + 777)
    H_X_aif, H_G_aif = compute_hessians(W_aif, X_eval_aif)
    rot_rng_aif = np.random.default_rng(42 + 4242)
    U_aif, _ = signed_random_hadamard(m_aif, rot_rng_aif)
    V_aif, _ = signed_random_hadamard(n_aif, rot_rng_aif)
    aif_result = run_accept_if_improve(W_aif, X_sel_aif, Xt_sel_aif, X_eval_aif,
                                        H_X_aif, H_G_aif, K=5, rng=np.random.default_rng(SEED),
                                        precomputed_rotations=(U_aif, V_aif))

    # ─── Summary and analysis ───
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Aggregate across tensors and slices
    print("\n--- Mean HWE by config × K (averaged over tensors × slices) ---")
    print(f"{'Config':<30} {'K=3':>14} {'K=4':>14} {'K=5':>14} {'K=6':>14} {'bytes_K5':>12}")
    print("-" * 100)

    cfg_names = [c[0] for c in configs]
    for cfg_name in cfg_names:
        row = f"{cfg_name:<30}"
        bytes_k5 = None
        for K in K_VALUES:
            hwes = []
            for tname in TENSOR_NAMES:
                if tname not in all_results:
                    continue
                for sname in SLICE_NAMES:
                    if K in all_results.get(tname, {}).get(sname, {}):
                        r = all_results[tname][sname][K].get(cfg_name)
                        if r:
                            hwes.append(r["hwe"])
            if hwes:
                mean_hwe = np.mean(hwes)
                row += f" {mean_hwe:>14.6e}"
                if K == 5:
                    bytes_k5 = np.mean([all_results[t][s][K][cfg_name]["bytes"]["total_bytes"]
                                       for t in TENSOR_NAMES if t in all_results
                                       for s in SLICE_NAMES if s in all_results.get(t, {})
                                       and K in all_results.get(t, {}).get(s, {})
                                       and cfg_name in all_results[t][s][K]])
            else:
                row += f" {'N/A':>14}"
        if bytes_k5 is not None:
            row += f" {bytes_k5:>12.1f}"
        print(row)

    # Improvement over baseline
    print("\n--- % Improvement over 'none' baseline ---")
    print(f"{'Config':<30} {'K=3':>10} {'K=4':>10} {'K=5':>10} {'K=6':>10}")
    print("-" * 80)
    for cfg_name in cfg_names:
        if cfg_name == "none":
            continue
        row = f"{cfg_name:<30}"
        for K in K_VALUES:
            improvements = []
            for tname in TENSOR_NAMES:
                if tname not in all_results:
                    continue
                for sname in SLICE_NAMES:
                    base_r = all_results.get(tname, {}).get(sname, {}).get(K, {}).get("none")
                    cfg_r = all_results.get(tname, {}).get(sname, {}).get(K, {}).get(cfg_name)
                    if base_r and cfg_r and base_r["hwe"] > 1e-15:
                        imp = (1 - cfg_r["hwe"] / base_r["hwe"]) * 100
                        improvements.append(imp)
            if improvements:
                row += f" {np.mean(improvements):>+9.1f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Best config per K
    print("\n--- Best config per K ---")
    for K in K_VALUES:
        best_cfg = None
        best_hwe = float('inf')
        for cfg_name in cfg_names:
            hwes = []
            for tname in TENSOR_NAMES:
                if tname not in all_results:
                    continue
                for sname in SLICE_NAMES:
                    r = all_results.get(tname, {}).get(sname, {}).get(K, {}).get(cfg_name)
                    if r:
                        hwes.append(r["hwe"])
            if hwes:
                mean_hwe = np.mean(hwes)
                if mean_hwe < best_hwe:
                    best_hwe = mean_hwe
                    best_cfg = cfg_name
        print(f"  K={K}: best = {best_cfg} (mean HWE = {best_hwe:.6e})")

    # Full stack vs best individual
    print("\n--- Full stack vs best individual component ---")
    for K in K_VALUES:
        full_hwes = []
        individual_hwes = []
        for tname in TENSOR_NAMES:
            if tname not in all_results:
                continue
            for sname in SLICE_NAMES:
                full_r = all_results.get(tname, {}).get(sname, {}).get(K, {}).get("full_stack")
                if full_r:
                    full_hwes.append(full_r["hwe"])
                # Best individual (single-component configs only, including gptq_only)
                individual_cfgs = ["scaling_only", "biip_only", "rotation_only",
                                   "allocation_only", "gptq_only", "gptaq_only"]
                best_ind = float('inf')
                for ic in individual_cfgs:
                    r = all_results.get(tname, {}).get(sname, {}).get(K, {}).get(ic)
                    if r:
                        best_ind = min(best_ind, r["hwe"])
                if best_ind < float('inf'):
                    individual_hwes.append(best_ind)

        if full_hwes and individual_hwes:
            mean_full = np.mean(full_hwes)
            mean_ind = np.mean(individual_hwes)
            imp = (1 - mean_full / mean_ind) * 100
            print(f"  K={K}: full_stack={mean_full:.6e}, best_individual={mean_ind:.6e}, improvement={imp:+.1f}%")

    # Byte budget verification
    print("\n--- Byte budget verification (K=5, L0_gate, first) ---")
    codebook_bytes = (SLICE_SIZE // TILE) * (SLICE_SIZE // TILE) * 4
    base_bytes = SLICE_SIZE * SLICE_SIZE * 5 / 8.0 + codebook_bytes
    print(f"  Target (payload + codebook): {base_bytes:.1f} bytes")
    for cfg_name in cfg_names:
        r = all_results.get("L0_gate", {}).get("first", {}).get(5, {}).get(cfg_name)
        if r:
            b = r["bytes"]
            print(f"  {cfg_name:<30} payload={b['payload_bytes']:.1f} sidecar={b['sidecar_bytes']:.1f} "
                  f"codebook={b['codebook_bytes']:.1f} k_meta={b['k_meta_bytes']:.1f} total={b['total_bytes']:.1f} bpe={b['bits_per_element']:.4f}")
    if budget_violations:
        print(f"\n--- Budget Violations ({len(budget_violations)} arms over budget) ---")
        for tv in budget_violations[:10]:
            print(f"  {tv[3]} K={tv[2]} {tv[0]}/{tv[1]}: {tv[4]:.1f} > {tv[5]:.1f}")
        if len(budget_violations) > 10:
            print(f"  ... and {len(budget_violations) - 10} more")
    else:
        print("\n--- Budget: ALL ARMS WITHIN TARGET ---")
    # Overfitting analysis
    print("\n--- Overfitting Analysis (K=5, mean over tensors × slices) ---")
    print(f"{'Config':<30} {'In-sample':>14} {'Held-out':>14} {'Ratio':>10}")
    print("-" * 72)
    for cfg_name in cfg_names:
        insample = []
        heldout = []
        for tname in TENSOR_NAMES:
            if tname not in all_results:
                continue
            for sname in SLICE_NAMES:
                r = all_results.get(tname, {}).get(sname, {}).get(5, {}).get(cfg_name)
                if r:
                    insample.append(r["hwe_insample"])
                    heldout.append(r["hwe"])
        if insample and heldout:
            mi = np.mean(insample)
            mh = np.mean(heldout)
            ratio = mi / mh if mh > 1e-15 else float('inf')
            print(f"{cfg_name:<30} {mi:>14.6e} {mh:>14.6e} {ratio:>10.4f}")

    # Accept-if-improve results
    print("\n--- Accept-if-Improve Results (K=5, L0_gate) ---")
    print(f"  Accepted: {aif_result['accepted']}")
    print(f"  Rejected: {aif_result['rejected']}")

    # ─── Save results ───
    output = {
        "experiment": "r11-unified-stack",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "TILE": TILE,
            "SLICE_SIZE": SLICE_SIZE,
            "K_VALUES": K_VALUES,
            "N_CALIB_SELECT": N_CALIB_SELECT,
            "N_CALIB_EVAL": N_CALIB_EVAL,
            "DAMPING": DAMPING,
            "GPTAQ_ALPHA": GPTAQ_ALPHA,
            "BLOCK_SIZE": BLOCK_SIZE,
            "TENSOR_NAMES": TENSOR_NAMES,
            "SLICE_NAMES": SLICE_NAMES,
        },
        "sanity_checks": {
            "cholesky_convention": "PASS",
            "gptaq_not_noop": "PASS",
            "reproducibility": "PASS",
        },
        "budget_violations": budget_violations,
        "results": all_results,
        "noise_floor": noise_floor_results,
        "accept_if_improve": {
            "accepted": aif_result["accepted"],
            "rejected": aif_result["rejected"],
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
