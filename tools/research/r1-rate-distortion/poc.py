#!/usr/bin/env python3
"""
R1-RateDistortion: Exact trellis rate-distortion allocation.

Generalizes BAQ's closed-form bit allocation (arXiv:2506.05664) to the
trellis-tile level. Each 16×16 tile gets its own K value based on
Hessian-weighted sensitivity.

=== Mathematical Derivation: Tile-level BAQ ===

BAQ (original, per-element):
  Loss: L_{ij} = (w_{ij} - Q(w_{ij}))^2 / [H_F^{-1}]_{jj}
  High-res approx: L_{ij}(R) ≈ (w_range)^2 / (12 · [H_F^{-1}]_{jj} · 2^{2R})
  c_{ij} = (w_range)^2 / (12 · [H_F^{-1}]_{jj})
  Optimal: R*_{ij} = 0.5 · log2(c_{ij} / λ) + R_sum/(MN)   (Eq 5-6)

Tile-level generalization:
  For tile t (s×s = 16×16), all elements share one quantizer (range = tile range).
  Tile distortion (Hessian-weighted, one-sided):
    D_t(K) = (range_t)^2 / (12 · 2^{2K}) · Σ_{(i,j)∈t} 1/[H_X^{-1}]_{jj}
           = c_t · 2^{-2K}
  where c_t = (range_t)^2 / 12 · Σ_{(i,j)∈t} 1/[H_X^{-1}]_{jj}
            = (range_t)^2 · s / 12 · Σ_{j∈tile cols} 1/[H_X^{-1}]_{jj}

  Two-sided (with output Hessian H_G ≈ Y^T Y / P):
    c_t = (range_t)^2 / 12 · Σ_{(i,j)∈t} H_G[i,i] / [H_X^{-1}]_{jj}

  BAQ allocation: K_t* = 0.5 · log2(c_t / λ) + K_avg
  where λ = (Π_t c_t)^{1/T} (geometric mean over T tiles)

  Equal-loss principle: c_t · 2^{-2K_t} = c_{t'} · 2^{-2K_{t'}} for all t, t'.

Exact discrete DP (tile-local additive surrogate, multiple-choice knapsack):
  min Σ_t D_{t,k} z_{t,k}  s.t.  Σ_k z_{t,k} = 1 per tile, Σ C_{t,k} z_{t,k} ≤ B
  where D_{t,k} = tr(H_G_sub @ E_t @ H_X_sub @ E_t^T) (tile-local distortion)
  NOTE: This is an additive surrogate. The full objective tr(H_G @ E @ H_X @ E^T)
  also contains cross-tile terms. local_search_refine() closes this gap.

Strategies compared:
  1. Uniform K — all tiles same K
  2. Column-BAQ — BAQ per column, aggregated to tiles
  3. Tile-BAQ (one-sided) — BAQ per tile, input Hessian only (OBS inverse)
  4. Tile-BAQ (two-sided) — BAQ per tile, input + output Hessian (OBS inverse)
  5. Tile-BAQ (direct) — direct H_X·H_G sensitivity (no OBS assumption)
  6. Tile-local DP — additive surrogate knapsack (no cross-tile terms)
  7. DP-refined — tile-local DP + full-objective local search
  8. Iterative BAQ — tile-BAQ with sensitivity refinement (3 rounds)
  9. BAQ + weight magnitude — c_t augmented with |w_{ij}|^2
  10. Lagrangian frontier — sweep λ, trace R-D curve

All arms use the SAME per-tile (16×16) uniform quantizer.
Exact byte budget: payload + sidecar (min/max float16) + K-metadata (3 bits/tile).
Primary metric: tr(H_G · E · H_X · E^T) where E = W - Ŵ.
"""

import numpy as np
import json
import time
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

TILE = 16
M_DIM = 128
N_DIM = 128
K_VALUES = [3, 4, 5, 6, 7]
K_MIN = min(K_VALUES)
K_MAX = max(K_VALUES)
P_CAL = 512  # calibration samples
N_TILES = (M_DIM // TILE) * (N_DIM // TILE)  # 64
ELEMENTS_PER_TILE = TILE * TILE  # 256
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"

# ============================================================================
# Utilities
# ============================================================================

def load_real_weights():
    """Load real Qwen3.8-27B BF16 weights (correctly decoded)."""
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in data.files:
        tensors[key] = data[key].astype(np.float64)
    return tensors


def extract_slices(tensor, m=128, n=128, seed=42):
    """Extract multiple 128×128 slices from a large tensor."""
    M, N = tensor.shape
    slices = []
    # First (top-left)
    slices.append(("first", tensor[:m, :n].copy()))
    # Middle
    r0, c0 = M // 2 - m // 2, N // 2 - n // 2
    slices.append(("mid", tensor[r0:r0 + m, c0:c0 + n].copy()))
    # Random
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, max(1, M - m))
    c0 = rng.integers(0, max(1, N - n))
    slices.append(("rand", tensor[r0:r0 + m, c0:c0 + n].copy()))
    return slices


def gen_calibration(N, P, seed=42):
    """Generate synthetic calibration activations with realistic structure.
    Per-channel scale variation + outlier channels → Hessian variation."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    # Per-channel scale variation (log-uniform)
    scales = np.exp(rng.uniform(-2, 1, N))
    X *= scales[:, None]
    # Outlier channels (5% of channels get 5× boost)
    outlier = rng.random(N) < 0.05
    X[outlier] *= 5.0
    return X


# ============================================================================
# Quantizer (per-tile uniform, MATCHED for all arms)
# ============================================================================

def quantize_tile(w, k):
    """Per-tile uniform quantizer. k bits → 2^k levels. Same for ALL arms."""
    if k <= 0:
        return np.zeros_like(w)
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo)
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_matrix(W, K_grid, tile=TILE):
    """Quantize W using per-tile uniform quantizer.
    K_grid[ti, tj] gives K for tile at tile-position (ti, tj)."""
    M, N = W.shape
    Wq = np.zeros_like(W)
    n_tiles_row = M // tile
    n_tiles_col = N // tile
    for ti in range(n_tiles_row):
        for tj in range(n_tiles_col):
            k = int(K_grid[ti, tj])
            r0, c0 = ti * tile, tj * tile
            Wq[r0:r0+tile, c0:c0+tile] = quantize_tile(
                W[r0:r0+tile, c0:c0+tile], k)
    return Wq


# ============================================================================
# Exact byte budget accounting
# ============================================================================

def compute_bytes(K_flat, n_tiles=N_TILES, elements_per_tile=ELEMENTS_PER_TILE):
    """Exact byte count: payload + sidecar + K-metadata.
    - Payload: K bits per element, packed into bytes
    - Sidecar: 2 × float16 (min, max) per tile = 4 bytes/tile
    - K-metadata: 3 bits/tile for mixed-K, 1 byte for uniform K
    """
    total_k = int(np.sum(K_flat))
    # Payload: each tile has elements_per_tile elements at K bits each
    payload_bits = total_k * elements_per_tile
    payload_bytes = payload_bits // 8  # exact: 256 is divisible by 8

    # Sidecar: min/max per tile as float16
    sidecar_bytes = n_tiles * 4

    # K-metadata
    if len(set(K_flat.tolist() if hasattr(K_flat, 'tolist') else list(K_flat))) == 1:
        metadata_bytes = 1  # single K value for whole matrix
    else:
        metadata_bytes = (n_tiles * 3 + 7) // 8  # 3 bits per tile, packed

    return payload_bytes + sidecar_bytes + metadata_bytes


def budget_k_sum_for_avg_k(avg_k, n_tiles=N_TILES):
    """Compute the integer K-sum budget for mixed-K arms to match uniform-K bytes.
    Uniform-K at avg_k uses: 32*avg_k*n_tiles + 4*n_tiles + 1 bytes.
    Mixed-K must fit: 32*sum(K_t) + 4*n_tiles + ceil(n_tiles*3/8) ≤ B_uniform.
    So sum(K_t) ≤ (B_uniform - 4*n_tiles - metadata_mixed) / 32.
    """
    B_uniform = compute_bytes(np.full(n_tiles, avg_k), n_tiles)
    metadata_mixed = (n_tiles * 3 + 7) // 8
    budget = (B_uniform - 4 * n_tiles - metadata_mixed) // 32
    return budget


# ============================================================================
# Hessian computation
# ============================================================================

def compute_hessians(W, X):
    """Compute input and output Hessians.
    H_X = X @ X.T / P  (N×N, input Hessian)
    H_G = Y @ Y.T / P  (M×M, output Hessian, Y = W @ X)
    H_inv = (H_X + λI)^{-1}  (damped inverse for BAQ sensitivity)
    """
    M, N = W.shape
    P = X.shape[1]

    # Input Hessian
    H_X = X @ X.T / P  # (N, N)

    # Output Hessian proxy: H_G ≈ Y @ Y.T / P where Y = W @ X
    Y = W @ X  # (M, P)
    H_G = Y @ Y.T / P  # (M, M)

    # Damped inverse for BAQ sensitivity
    lam_damp = max(0.01 * np.mean(np.diag(H_X)), 1e-10)
    H_inv = np.linalg.inv(H_X + lam_damp * np.eye(N))

    return H_X, H_G, H_inv


# ============================================================================
# Tile sensitivity (BAQ c_t computation)
# ============================================================================

def tile_ranges(W, tile=TILE):
    """Compute per-tile range (max - min). Returns (n_tiles_row, n_tiles_col) array."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    ranges = np.zeros((ntr, ntc))
    for ti in range(ntr):
        for tj in range(ntc):
            t = W[ti*tile:(ti+1)*tile, tj*tile:(tj+1)*tile]
            ranges[ti, tj] = t.max() - t.min()
    return ranges


def tile_c_onesided(W, H_inv, tile=TILE):
    """One-sided tile sensitivity (input Hessian only, BAQ original).
    c_t = (range_t)^2 · s / 12 · Σ_{j∈tile cols} 1/[H_inv]_{jj}
    where s = tile size (16).
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    ranges = tile_ranges(W, tile)
    hinv_inv_diag = 1.0 / (np.diag(H_inv) + 1e-30)  # 1/[H^{-1}]_{jj} = sensitivity

    c = np.zeros((ntr, ntc))
    for ti in range(ntr):
        for tj in range(ntc):
            cols = range(tj * tile, (tj + 1) * tile)
            S_t = tile * np.sum(hinv_inv_diag[cols])  # s * sum over input channels
            c[ti, tj] = (ranges[ti, tj] ** 2) / 12.0 * S_t
    return c


def tile_c_twosided(W, H_inv, H_G, tile=TILE):
    """Two-sided tile sensitivity (input + output Hessian).
    c_t = (range_t)^2 / 12 · Σ_{(i,j)∈t} H_G[i,i] / [H_inv]_{jj}
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    ranges = tile_ranges(W, tile)
    hinv_inv_diag = 1.0 / (np.diag(H_inv) + 1e-30)
    H_G_diag = np.diag(H_G)

    c = np.zeros((ntr, ntc))
    for ti in range(ntr):
        for tj in range(ntc):
            rows = range(ti * tile, (ti + 1) * tile)
            cols = range(tj * tile, (tj + 1) * tile)
            S_t = 0.0
            for i in rows:
                for j in cols:
                    S_t += H_G_diag[i] * hinv_inv_diag[j]
            c[ti, tj] = (ranges[ti, tj] ** 2) / 12.0 * S_t
    return c


def tile_c_weightmag(W, H_inv, tile=TILE):
    """BAQ + weight magnitude: c_t augmented with |w_{ij}|^2.
    c_t = (range_t)^2 / 12 · Σ_{(i,j)∈t} |w_{ij}|^2 / [H_inv]_{jj}
    Allocates more bits to tiles with both high sensitivity AND large weights.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    ranges = tile_ranges(W, tile)
    hinv_inv_diag = 1.0 / (np.diag(H_inv) + 1e-30)

    c = np.zeros((ntr, ntc))
    for ti in range(ntr):
        for tj in range(ntc):
            rows = slice(ti * tile, (ti + 1) * tile)
            cols = slice(tj * tile, (tj + 1) * tile)
            t = W[rows, cols]
            S_t = 0.0
            for di in range(tile):
                for dj in range(tile):
                    j = tj * tile + dj
                    S_t += (t[di, dj] ** 2) * hinv_inv_diag[j]
            c[ti, tj] = (ranges[ti, tj] ** 2) / 12.0 * S_t
    return c

def tile_c_direct(W, H_X, H_G, tile=TILE):
    """Direct Hessian sensitivity (no OBS inverse-Hessian assumption).
    For independent zero-mean errors without GPTQ/OBS compensation:
    c_t = (range_t)^2 / 12 · Σ_{(i,j)∈t} H_G[i,i] · H_X[j,j]
    This uses H_X directly (not H_X^{-1}), matching the expected
    tr(H_G E H_X E^T) for independent per-element errors.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    ranges = tile_ranges(W, tile)
    H_G_diag = np.diag(H_G)
    H_X_diag = np.diag(H_X)

    c = np.zeros((ntr, ntc))
    for ti in range(ntr):
        for tj in range(ntc):
            rows = range(ti * tile, (ti + 1) * tile)
            cols = range(tj * tile, (tj + 1) * tile)
            S_t = 0.0
            for i in rows:
                for j in cols:
                    S_t += H_G_diag[i] * H_X_diag[j]
            c[ti, tj] = (ranges[ti, tj] ** 2) / 12.0 * S_t
    return c


# ============================================================================
# BAQ closed-form allocation with exact integer projection
# ============================================================================

def baq_allocate(c_flat, budget_k, k_min=K_MIN, k_max=K_MAX):
    """BAQ closed-form allocation with exact integer projection.
    c_flat: (n_tiles,) array of c_t values.
    budget_k: exact integer sum of K values (budget).
    Returns: (n_tiles,) integer K array with sum == budget_k.
    """
    n = len(c_flat)

    # Geometric mean (λ)
    log_c = np.log(np.clip(c_flat, 1e-30, None))
    lam = np.exp(np.mean(log_c))

    # Continuous optimal: K_t* = 0.5 * log2(c_t / λ) + budget_k / n
    avg_k = budget_k / n
    K_continuous = 0.5 * np.log2(c_flat / (lam + 1e-30)) + avg_k

    # Clip to [k_min, k_max]
    K_clipped = np.clip(K_continuous, k_min, k_max)

    # Integer projection: floor + distribute remaining by fractional part
    K_floor = np.floor(K_clipped).astype(int)
    K_floor = np.clip(K_floor, k_min, k_max)

    remaining = budget_k - int(np.sum(K_floor))

    if remaining > 0:
        # Give +1 to tiles with highest fractional part (closest to next integer)
        frac = K_clipped - K_floor
        eligible = np.where(K_floor < k_max)[0]
        order = eligible[np.argsort(-frac[eligible])]
        for i in range(min(remaining, len(order))):
            K_floor[order[i]] += 1
    elif remaining < 0:
        # Take -1 from tiles with lowest fractional part (closest to prev integer)
        frac = K_clipped - K_floor
        eligible = np.where(K_floor > k_min)[0]
        order = eligible[np.argsort(frac[eligible])]
        for i in range(min(-remaining, len(order))):
            K_floor[order[i]] -= 1

    # Verify exact budget
    assert int(np.sum(K_floor)) == budget_k, \
        f"Budget mismatch: {np.sum(K_floor)} != {budget_k}"

    return K_floor


def baq_allocate_lagrangian(c_flat, lam, k_min=K_MIN, k_max=K_MAX):
    """BAQ allocation for a given λ (Lagrangian mode, no fixed budget).
    K_t = clip(round(0.5 * log2(c_t / λ)), k_min, k_max)
    """
    K_continuous = 0.5 * np.log2(c_flat / (lam + 1e-30))
    K_int = np.clip(np.round(K_continuous).astype(int), k_min, k_max)
    return K_int


# ============================================================================
# Allocation strategies
# ============================================================================

def alloc_uniform(avg_k):
    """All tiles same K."""
    return np.full(N_TILES, avg_k, dtype=int)


def alloc_column_baq(W, H_inv, avg_k, tile=TILE):
    """Column-BAQ: BAQ per column, aggregated to tiles.
    Compute c_j per column (BAQ original), then aggregate to tiles using
    marginal-distortion-aware rounding: tiles get the K that minimizes
    the aggregate column sensitivity loss.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile

    # Per-column weight range
    w_range = W.max(axis=0) - W.min(axis=0)  # (N,)

    # Per-column c_j = (range_j)^2 / (12 * [H_inv]_{jj})
    hinv_diag = np.diag(H_inv)
    c_col = (w_range ** 2) / (12.0 * hinv_diag + 1e-30)

    # BAQ allocation per column: budget = avg_k * N so average column K ≈ avg_k
    # After aggregation (weighted mean of 16 columns per tile), tile sum ≈ avg_k * n_tiles
    budget_k = budget_k_sum_for_avg_k(avg_k)
    budget_col = avg_k * N
    K_col = baq_allocate(c_col, budget_col, K_MIN, K_MAX)

    # Aggregate to tiles: each tile's K = round(mean of its column K's)
    # Per-tile sensitivity uses actual tile weight range (row+col dependent)
    K_tile = np.zeros(ntr * ntc, dtype=int)
    tile_c = np.zeros(ntr * ntc)
    idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            cols = slice(tj * tile, (tj + 1) * tile)
            rows = slice(ti * tile, (ti + 1) * tile)
            # Per-tile sensitivity: range of the actual tile × column sensitivity sum
            tile_block = W[rows, cols]
            tile_range = tile_block.max() - tile_block.min()
            tile_c[idx] = (tile_range ** 2) / 12.0 * np.sum(c_col[cols])
            K_tile[idx] = int(np.round(np.mean(K_col[cols])))
            idx += 1

    # Fix budget iteratively using marginal sensitivity.
    # Marginal benefit of +1 bit: c_t * (2^{-2K} - 2^{-2(K+1)}) = c_t * 2^{-2K} * (1 - 1/4)
    # Marginal cost of -1 bit:  c_t * (2^{-2(K-1)} - 2^{-2K}) = c_t * 2^{-2K} * (4 - 1)
    # Add bits where marginal benefit is highest, remove where marginal cost is lowest.
    for _ in range(10000):  # safety bound
        current_sum = int(np.sum(K_tile))
        diff = budget_k - current_sum
        if diff == 0:
            break
        if diff > 0:
            # Add 1 bit to tile with highest marginal benefit
            eligible = np.where(K_tile < K_MAX)[0]
            if len(eligible) == 0:
                break
            marginal_benefit = tile_c[eligible] * (4.0 ** (-K_tile[eligible])) * 0.75
            best = eligible[np.argmax(marginal_benefit)]
            K_tile[best] += 1
        else:
            # Remove 1 bit from tile with lowest marginal cost
            eligible = np.where(K_tile > K_MIN)[0]
            if len(eligible) == 0:
                break
            marginal_cost = tile_c[eligible] * (4.0 ** (-K_tile[eligible])) * 3.0
            worst = eligible[np.argmin(marginal_cost)]
            K_tile[worst] -= 1

    K_tile = np.clip(K_tile, K_MIN, K_MAX)
    assert int(np.sum(K_tile)) == budget_k, \
        f"Column-BAQ budget mismatch: {np.sum(K_tile)} != {budget_k}"
    return K_tile


def alloc_tile_baq(c_flat, avg_k):
    """Tile-BAQ: BAQ formula per tile."""
    budget_k = budget_k_sum_for_avg_k(avg_k)
    return baq_allocate(c_flat, budget_k, K_MIN, K_MAX)


def alloc_iterative_baq(W, H_inv, H_G, avg_k, tile=TILE, n_rounds=3, twosided=True):
    """Iterative tile-BAQ: allocate → quantize → measure actual distortion → update c → reallocate.
    Round 1: BAQ closed-form from Hessian sensitivity.
    Rounds 2-3: Update c_t from measured distortion, reallocate.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    budget_k = budget_k_sum_for_avg_k(avg_k)

    # Initial c from BAQ formula
    if twosided:
        c = tile_c_twosided(W, H_inv, H_G, tile)
    else:
        c = tile_c_onesided(W, H_inv, tile)
    c_flat = c.flatten()

    K_best = None
    for rnd in range(n_rounds):
        # Allocate
        K_flat = baq_allocate(c_flat, budget_k, K_MIN, K_MAX)
        K_grid = K_flat.reshape(ntr, ntc)

        # Quantize
        Wq = quantize_matrix(W, K_grid, tile)
        E = W - Wq

        # Measure actual per-tile distortion
        H_G_diag = np.diag(H_G)
        hinv_inv_diag = 1.0 / (np.diag(H_inv) + 1e-30)

        for ti in range(ntr):
            for tj in range(ntc):
                r0, c0 = ti * tile, tj * tile
                E_t = E[r0:r0+tile, c0:c0+tile]
                # Actual Hessian-weighted distortion of this tile
                D_actual = np.sum(E_t ** 2 * H_G_diag[r0:r0+tile, None] *
                                  hinv_inv_diag[c0:c0+tile, None].T)
                # Implied c: D = c * 2^{-2K} → c = D * 2^{2K}
                k = K_flat[ti * ntc + tj]
                c_implied = D_actual * (2 ** (2 * k)) + 1e-30
                # Update c: blend old and new (exponential moving average)
                alpha = 0.5
                c_flat[ti * ntc + tj] = (1 - alpha) * c_flat[ti * ntc + tj] + alpha * c_implied

        K_best = K_flat.copy()

    return K_best


def alloc_perelement_baq(W, H_inv, H_G, avg_k, tile=TILE):
    """Per-element BAQ (two-sided) aggregated to tiles.
    Uses full diagonal of H^{-1} and H_G for per-element c_{ij},
    then aggregates to tiles.
    """
    c = tile_c_twosided(W, H_inv, H_G, tile)
    return alloc_tile_baq(c.flatten(), avg_k)


def alloc_baq_weightmag(W, H_inv, avg_k, tile=TILE):
    """BAQ + weight magnitude: c_t augmented with |w_{ij}|^2."""
    c = tile_c_weightmag(W, H_inv, tile)
    return alloc_tile_baq(c.flatten(), avg_k)


# ============================================================================
# Tile-local DP solver: multiple-choice knapsack (additive surrogate)
# ============================================================================

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    """Measure actual Hessian-weighted distortion of a tile at K=k.
    D = tr(H_G_sub @ E_t @ H_X_sub @ E_t^T)
    where E_t = W_tile - Q(W_tile).
    """
    Q_tile = quantize_tile(W_tile, k)
    E_t = W_tile - Q_tile
    # Exact: tr(H_G_sub @ E_t @ H_X_sub @ E_t.T)
    # = tr(E_t.T @ H_G_sub @ E_t @ H_X_sub)  (cyclic)
    D = np.trace(H_G_sub @ E_t @ H_X_sub @ E_t.T)
    return max(D, 0.0)  # numerical guard


def alloc_tile_local_dp(W, H_X, H_G, avg_k, tile=TILE, budget_k=None):
    """Tile-local DP: exact solver for the ADDITIVE tile-local surrogate.
    For each tile, measure actual distortion at each K. Solve min total
    tile-local distortion s.t. sum K_t <= budget_k, one K per tile.
    NOTE: This optimizes sum_t tr(H_G_sub @ E_t @ H_X_sub @ E_t^T), NOT the
    full coupled objective tr(H_G @ E @ H_X @ E^T). Cross-tile terms are
    omitted. Use local_search_refine() to close this gap.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc
    if budget_k is None:
        n_tiles_def = (M_DIM // TILE) * (N_DIM // TILE)
        eps_def = TILE * TILE
        B_uniform = compute_bytes(np.full(n_tiles_def, avg_k), n_tiles_def, eps_def)
        metadata_mixed = (n_tiles_def * 3 + 7) // 8
        budget_k = (B_uniform - 4 * n_tiles_def - metadata_mixed) // (eps_def // 8)

    # Precompute distortion table: D_table[t, ki] = distortion of tile t at K=K_VALUES[ki]
    D_table = np.zeros((n_tiles, len(K_VALUES)))
    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0+tile, c0:c0+tile]
            H_G_sub = H_G[r0:r0+tile, r0:r0+tile]
            H_X_sub = H_X[c0:c0+tile, c0:c0+tile]
            for ki, k in enumerate(K_VALUES):
                D_table[t_idx, ki] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
            t_idx += 1

    # Multiple-choice knapsack DP
    # dp[j] = min distortion with total K-sum = j
    INF = float('inf')
    dp = [INF] * (budget_k + 1)
    dp[0] = 0.0
    choices = []  # choices[t][j] = ki chosen for tile t at state j

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

    # Find min distortion (best j ≤ budget_k)
    best_j = 0
    best_d = INF
    for j in range(budget_k + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    # Backtrack
    K_flat = np.zeros(n_tiles, dtype=int)
    j = best_j
    for t in range(n_tiles - 1, -1, -1):
        ki = choices[t][j]
        if ki < 0:
            # Fallback: find any valid ki that fits remaining budget
            for fallback_ki in range(len(K_VALUES)):
                if K_VALUES[fallback_ki] <= j:
                    ki = fallback_ki
                    break
        K_flat[t] = K_VALUES[ki]
        j -= K_VALUES[ki]

    assert int(np.sum(K_flat)) <= budget_k, f"DP budget exceeded: {np.sum(K_flat)} > {budget_k}"

    return K_flat, D_table


def local_search_refine(W, K_flat, H_G, H_X, tile=TILE, budget_k=None,
                        max_iters=1000):
    """Full-objective local search: try single-bit transfers between tiles.
    For each pair (donor, receiver), if moving one bit from donor to receiver
    (donor K-1, receiver K+1) reduces full tr(H_G E H_X E^T), accept.
    Continues until no improving swap exists or max_iters reached.
    This closes the gap between tile-local DP and the full coupled objective.
    """
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc

    def full_hwe(K_arr):
        K_grid = K_arr.reshape(ntr, ntc)
        Wq = quantize_matrix(W, K_grid, tile)
        E = W - Wq
        return hessian_weighted_error(E, H_G, H_X)

    K = K_flat.copy()
    current_hwe = full_hwe(K)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try all donor→receiver pairs
        for donor in range(n_tiles):
            if K[donor] <= K_MIN:
                continue
            for receiver in range(n_tiles):
                if receiver == donor:
                    continue
                if K[receiver] >= K_MAX:
                    continue
                # Try transferring one bit
                K[donor] -= 1
                K[receiver] += 1
                new_hwe = full_hwe(K)
                if new_hwe < current_hwe - 1e-15:
                    current_hwe = new_hwe
                    improved = True
                    break  # accept, restart inner loop
                else:
                    K[donor] += 1
                    K[receiver] -= 1
            if improved:
                break

    converged = not improved  # True if no more improving swaps (not capped)
    # Verify K sum is preserved (bit transfers don't change total)
    assert int(np.sum(K)) == int(np.sum(K_flat)), "Local search changed K sum!"
    return K, current_hwe, converged


# ============================================================================
# Evaluation metrics
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """Primary metric: tr(H_G · E · H_X · E^T).
    H_G: (M, M) output Hessian.
    H_X: (N, N) input Hessian.
    E: (M, N) error matrix.
    """
    return float(np.trace(H_G @ E @ H_X @ E.T))


def hessian_weighted_error_diag(E, H_G_diag, H_X_diag):
    """Diagonal approximation: Σ_{ij} H_G[i,i] · E[i,j]^2 · H_X[j,j].
    This is the separable surrogate that tile_local_dp optimizes (plus
    within-tile off-diagonal Hessian terms). The full metric tr(H_G E H_X E^T)
    also contains cross-tile terms via off-diagonal Hessian blocks.
    """
    return float(np.sum(E ** 2 * H_G_diag[:, None] * H_X_diag[None, :]))

def weight_mse(E):
    """Raw weight MSE (secondary metric)."""
    return float(np.mean(E ** 2))


# ============================================================================
# Experiment runner
# ============================================================================

def run_all_strategies(W, X, avg_k, tile=TILE):
    """Run all allocation strategies and return results dict."""
    M, N = W.shape
    H_X, H_G, H_inv = compute_hessians(W, X)
    H_G_diag = np.diag(H_G)
    H_X_diag = np.diag(H_X)

    # No-quant baseline (noise floor)
    E_none = np.zeros_like(W)
    noise_floor_exact = hessian_weighted_error(E_none, H_G, H_X)
    noise_floor_diag = hessian_weighted_error_diag(E_none, H_G_diag, H_X_diag)

    # Precompute tile sensitivities
    c_onesided = tile_c_onesided(W, H_inv, tile).flatten()
    c_twosided = tile_c_twosided(W, H_inv, H_G, tile).flatten()
    c_weightmag = tile_c_weightmag(W, H_inv, tile).flatten()
    c_direct = tile_c_direct(W, H_X, H_G, tile).flatten()

    strategies = {}

    # 1. Uniform K
    K_uniform = alloc_uniform(avg_k)
    strategies["uniform"] = K_uniform

    # 2. Column-BAQ (fixed: sensitivity-weighted aggregation)
    K_colbaq = alloc_column_baq(W, H_inv, avg_k, tile)
    strategies["column_baq"] = K_colbaq

    # 3. Tile-BAQ (one-sided, OBS inverse-Hessian)
    K_tilebaq_1 = alloc_tile_baq(c_onesided, avg_k)
    strategies["tile_baq_1sided"] = K_tilebaq_1

    # 4. Tile-BAQ (two-sided, OBS inverse-Hessian)
    K_tilebaq_2 = alloc_tile_baq(c_twosided, avg_k)
    strategies["tile_baq_2sided"] = K_tilebaq_2

    # 5. Tile-BAQ (direct Hessian, no OBS assumption)
    K_tilebaq_direct = alloc_tile_baq(c_direct, avg_k)
    strategies["tile_baq_direct"] = K_tilebaq_direct

    # 6. Tile-local DP (additive surrogate, no cross-tile terms)
    K_dp, D_table = alloc_tile_local_dp(W, H_X, H_G, avg_k, tile)
    strategies["tile_local_dp"] = K_dp

    # 7. DP + full-objective local search (closes cross-tile gap)
    K_dp_refined, dp_refined_hwe, dp_converged = local_search_refine(W, K_dp, H_G, H_X, tile)
    assert dp_converged, "DP-refined local search did not converge (hit iteration cap)"
    strategies["dp_refined"] = K_dp_refined

    # 8. Iterative BAQ (3 rounds, two-sided)
    K_iter = alloc_iterative_baq(W, H_inv, H_G, avg_k, tile, n_rounds=3, twosided=True)
    strategies["iterative_baq"] = K_iter

    # 9. BAQ + weight magnitude
    K_wmag = alloc_baq_weightmag(W, H_inv, avg_k, tile)
    strategies["baq_weightmag"] = K_wmag

    # Evaluate all strategies
    results = {
        "noise_floor_exact": noise_floor_exact,
        "noise_floor_diag": noise_floor_diag,
        "strategies": {}
    }

    for name, K_flat in strategies.items():
        K_grid = K_flat.reshape(M // tile, N // tile)
        Wq = quantize_matrix(W, K_grid, tile)
        E = W - Wq

        exact_bytes = compute_bytes(K_flat)
        hwe_exact = hessian_weighted_error(E, H_G, H_X)
        hwe_diag = hessian_weighted_error_diag(E, H_G_diag, H_X_diag)
        wmse = weight_mse(E)

        results["strategies"][name] = {
            "K_assignment": K_flat.tolist(),
            "exact_bytes": exact_bytes,
            "hwe_exact": hwe_exact,
            "hwe_diag": hwe_diag,
            "weight_mse": wmse,
            "k_sum": int(np.sum(K_flat)),
            "k_mean": float(np.mean(K_flat)),
            "k_std": float(np.std(K_flat)),
        }

    # Persist DP-refined convergence info
    results["strategies"]["dp_refined"]["converged"] = True

    return results


# ============================================================================
# Block-size invariance test
# ============================================================================

def block_size_invariance_test(W, X, avg_k):
    """Test that DP-optimal gives consistent results across tile sizes.
    Finer tiles (smaller tile) should give equal or better distortion at same budget.
    Also test that uniform K is invariant to tile size (same K, different partition).
    """
    print("\n" + "=" * 80)
    print("BLOCK-SIZE INVARIANCE TEST")
    print("=" * 80)

    H_X, H_G, H_inv = compute_hessians(W, X)
    H_G_diag = np.diag(H_G)
    H_X_diag = np.diag(H_X)

    tile_sizes = [8, 16, 32]
    dp_results = {}
    uniform_results = {}

    for tile in tile_sizes:
        M, N = W.shape
        if M % tile != 0 or N % tile != 0:
            continue
        n_tiles = (M // tile) * (N // tile)
        eps = tile * tile

        # Compute budget for this tile size: match uniform-K byte budget
        B_uniform = compute_bytes(np.full(n_tiles, avg_k), n_tiles, eps)
        metadata_mixed = (n_tiles * 3 + 7) // 8
        budget_k = (B_uniform - 4 * n_tiles - metadata_mixed) // (eps // 8)

        # DP-optimal at this tile size (pass explicit budget)
        K_dp, _ = alloc_tile_local_dp(W, H_X, H_G, avg_k, tile, budget_k=budget_k)
        K_grid = K_dp.reshape(M // tile, N // tile)
        Wq = quantize_matrix(W, K_grid, tile)
        E = W - Wq
        hwe = hessian_weighted_error(E, H_G, H_X)
        bytes_dp = compute_bytes(K_dp, n_tiles, eps)

        dp_results[tile] = {
            "hwe": hwe,
            "bytes": bytes_dp,
            "n_tiles": n_tiles,
            "k_sum": int(np.sum(K_dp)),
            "budget_k": budget_k,
        }

        # Uniform K at this tile size
        K_uni = np.full(n_tiles, avg_k, dtype=int)
        Wq_uni = quantize_matrix(W, K_uni.reshape(M // tile, N // tile), tile)
        E_uni = W - Wq_uni
        hwe_uni = hessian_weighted_error(E_uni, H_G, H_X)
        bytes_uni = compute_bytes(K_uni, n_tiles, eps)

        uniform_results[tile] = {
            "hwe": hwe_uni,
            "bytes": bytes_uni,
            "n_tiles": n_tiles,
        }

        print(f"  tile={tile:2d} | DP: hwe={hwe:.6e}, bytes={bytes_dp}, "
              f"n_tiles={n_tiles}, k_sum={np.sum(K_dp)}, budget={budget_k}")
        print(f"           | Uni: hwe={hwe_uni:.6e}, bytes={bytes_uni}, "
              f"n_tiles={n_tiles}")

    # Check: uniform K with per-tile quantizer is NOT tile-size invariant
    # (smaller tiles → tighter ranges → less quantization error). This is expected.
    # What we verify: uniform K with finer tiles gives ≤ uniform K with coarser tiles.
    uni_hwes = [(t, uniform_results[t]["hwe"]) for t in tile_sizes if t in uniform_results]
    print(f"\n  Uniform K tile-size effect (finer ≤ coarser expected):")
    uni_monotone = True
    for i in range(len(uni_hwes) - 1):
        t1, h1 = uni_hwes[i]
        t2, h2 = uni_hwes[i + 1]
        if h1 <= h2 * 1.001:
            print(f"    tile={t1} ({h1:.6e}) ≤ tile={t2} ({h2:.6e}) ✓")
        else:
            print(f"    tile={t1} ({h1:.6e}) > tile={t2} ({h2:.6e}) ✗")
            uni_monotone = False
    if uni_monotone:
        print("  ✓ PASS: Uniform K improves with finer tiles (expected: tighter ranges)")
    else:
        print("  ✗ UNEXPECTED: Uniform K does not improve with finer tiles")

    # Check: DP with finer tiles should be ≤ DP with coarser tiles
    dp_hwes = [(t, dp_results[t]["hwe"]) for t in sorted(dp_results)]
    print(f"\n  DP monotonicity (finer ≤ coarser):")
    passed = True
    for i in range(len(dp_hwes) - 1):
        t1, h1 = dp_hwes[i]
        t2, h2 = dp_hwes[i + 1]
        # t1 < t2 (finer → coarser), so h1 should be ≤ h2
        if h1 <= h2 * 1.001:  # small tolerance
            print(f"    tile={t1} ({h1:.6e}) ≤ tile={t2} ({h2:.6e}) ✓")
        else:
            print(f"    tile={t1} ({h1:.6e}) > tile={t2} ({h2:.6e}) ✗")
            passed = False

    if passed:
        print("  ✓ PASS: DP monotonicity holds (finer tiles → equal or better)")
    else:
        print("  ✗ FAIL: DP monotonicity violated")

    # Check: exact byte budgets match
    print(f"\n  Byte budget matching:")
    for t in tile_sizes:
        if t in uniform_results and t in dp_results:
            b_uni = uniform_results[t]["bytes"]
            b_dp = dp_results[t]["bytes"]
            print(f"    tile={t}: uniform={b_uni}, dp={b_dp}, "
                  f"match={'✓' if b_dp <= b_uni else '✗'}")

    return dp_results, uniform_results


# ============================================================================
# Lagrangian frontier sweep
# ============================================================================

def lagrangian_frontier(W, X, tile=TILE, n_points=30):
    """Sweep λ to trace the full rate-distortion frontier.
    For each λ, compute K_t = clip(round(0.5 * log2(c_t / λ)), K_MIN, K_MAX),
    then measure actual bytes and Hessian-weighted error.
    """
    M, N = W.shape
    H_X, H_G, H_inv = compute_hessians(W, X)
    H_G_diag = np.diag(H_G)
    H_X_diag = np.diag(H_X)

    # Use two-sided sensitivity
    c = tile_c_twosided(W, H_inv, H_G, tile).flatten()

    # λ range: from very small (all K_MAX) to very large (all K_MIN)
    log_c = np.log(np.clip(c, 1e-30, None))
    # λ = c_t → K_t = 0, so λ should range around geometric mean
    gm = np.exp(np.mean(log_c))
    lam_min = gm * 2 ** (-2 * (K_MAX + 2))
    lam_max = gm * 2 ** (2 * (K_MAX + 2))

    lams = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_points))

    frontier = []
    for lam in lams:
        K_flat = baq_allocate_lagrangian(c, lam)
        K_grid = K_flat.reshape(M // tile, N // tile)
        Wq = quantize_matrix(W, K_grid, tile)
        E = W - Wq
        hwe = hessian_weighted_error(E, H_G, H_X)
        nbytes = compute_bytes(K_flat)
        frontier.append({
            "lambda": float(lam),
            "bytes": nbytes,
            "hwe": hwe,
            "k_mean": float(np.mean(K_flat)),
            "k_sum": int(np.sum(K_flat)),
        })

    return frontier


# ============================================================================
# Main experiment
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 80)
    print("R1-RateDistortion: Exact Trellis Rate-Distortion Allocation")
    print("=" * 80)
    print(f"  Matrix: {M_DIM}×{N_DIM}, Tile: {TILE}×{TILE}, "
          f"Tiles: {N_TILES}, K values: {K_VALUES}")
    print(f"  Calibration: {P_CAL} samples, synthetic (Gaussian + outliers)")

    # Load real weights
    print("\nLoading real weights...")
    weights = load_real_weights()
    print(f"  Available tensors: {list(weights.keys())}")

    # Select tensors and slices
    tensor_keys = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
    all_results = {}

    for key in tensor_keys:
        if key not in weights:
            continue
        tensor = weights[key]
        slices = extract_slices(tensor, M_DIM, N_DIM, seed=42)

        for slice_name, W in slices:
            if W.shape[0] < M_DIM or W.shape[1] < N_DIM:
                continue

            # Generate calibration data for this slice's input dimension
            X = gen_calibration(N_DIM, P_CAL, seed=42)

            print(f"\n{'=' * 80}")
            print(f"  Tensor: {key}, Slice: {slice_name}")
            print(f"  W shape: {W.shape}, W range: [{W.min():.4f}, {W.max():.4f}], "
                  f"std: {W.std():.6f}")
            print(f"{'=' * 80}")

            slice_results = {}
            for avg_k in [4, 5, 6]:
                print(f"\n  --- avg_k = {avg_k} ---")
                budget = budget_k_sum_for_avg_k(avg_k)
                B_uniform = compute_bytes(np.full(N_TILES, avg_k))
                print(f"  Budget: {budget} K-units, {B_uniform} bytes (uniform)")

                res = run_all_strategies(W, X, avg_k)

                # Print comparison table
                print(f"\n  {'Strategy':<20} {'HWE exact':>14} {'HWE diag':>14} "
                      f"{'Wt MSE':>12} {'Bytes':>8} {'K mean':>7} {'K std':>6}")
                print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*12} {'-'*8} {'-'*7} {'-'*6}")

                # Sort by HWE exact
                for name in sorted(res["strategies"].keys(),
                                   key=lambda n: res["strategies"][n]["hwe_exact"]):
                    s = res["strategies"][name]
                    print(f"  {name:<20} {s['hwe_exact']:>14.6e} "
                          f"{s['hwe_diag']:>14.6e} {s['weight_mse']:>12.4e} "
                          f"{s['exact_bytes']:>8} {s['k_mean']:>7.2f} "
                          f"{s['k_std']:>6.2f}")

                print(f"\n  Noise floor: {res['noise_floor_exact']:.6e} (exact), "
                      f"{res['noise_floor_diag']:.6e} (diag)")

                # Compute improvement over uniform
                uni_hwe = res["strategies"]["uniform"]["hwe_exact"]
                print(f"\n  Improvement over uniform K:")
                for name in sorted(res["strategies"].keys(),
                                   key=lambda n: res["strategies"][n]["hwe_exact"]):
                    if name == "uniform":
                        continue
                    hwe = res["strategies"][name]["hwe_exact"]
                    imp = (1 - hwe / uni_hwe) * 100 if uni_hwe > 0 else 0
                    print(f"    {name:<20} {imp:+.2f}%")

                slice_results[f"avg_k_{avg_k}"] = res

            all_results[f"{key}_{slice_name}"] = slice_results

    # Block-size invariance test (on first tensor, first slice)
    print("\n" + "=" * 80)
    print("Running block-size invariance test...")
    print("=" * 80)
    W_test = extract_slices(weights["L0_gate"], M_DIM, N_DIM, seed=42)[0][1]
    X_test = gen_calibration(N_DIM, P_CAL, seed=42)
    block_size_invariance_test(W_test, X_test, avg_k=5)

    # Lagrangian frontier (on first tensor, first slice)
    print("\n" + "=" * 80)
    print("Lagrangian frontier sweep (L0_gate, first slice)...")
    print("=" * 80)
    frontier = lagrangian_frontier(W_test, X_test, tile=TILE, n_points=25)
    print(f"\n  {'λ':>12} {'Bytes':>8} {'HWE':>14} {'K mean':>7}")
    print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*7}")
    for pt in frontier:
        print(f"  {pt['lambda']:>12.4e} {pt['bytes']:>8} {pt['hwe']:>14.6e} "
              f"{pt['k_mean']:>7.2f}")

    # Save results
    output = {
        "config": {
            "matrix_dim": [M_DIM, N_DIM],
            "tile_size": TILE,
            "n_tiles": N_TILES,
            "k_values": K_VALUES,
            "p_cal": P_CAL,
            "weights_path": WEIGHTS_PATH,
        },
        "results": all_results,
        "frontier": frontier,
        "elapsed_seconds": time.time() - t_start,
    }

    output_path = "/Users/mbelleau/Projects/qwen38-research-r1-rate-distortion/receipts/research/r1-rate-distortion-results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
