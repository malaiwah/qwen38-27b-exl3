#!/usr/bin/env python3
"""
R24 — Entropy-Constrained Allocation

R20 found that entropy coding saves 13-24% of quantization rate. This PoC
incorporates entropy into the allocation DP: instead of allocating K bits per
tile (uniform fixed rate), we allocate an *entropy budget* per tile. The actual
rate after entropy coding is H(quantized_indices | tile), not K bits.

Core idea:
  - Fixed-K DP: minimize total distortion s.t. sum(K_t) <= budget_k
    Rate per tile = K_t * elements_per_tile (fixed-rate, no entropy coding)
  - Entropy-constrained DP: minimize total distortion s.t.
    sum(H_t(K_t) * elements_per_tile) <= entropy_budget
    Rate per tile = entropy_t(K_t) * elements_per_tile (variable-rate)
  - At the same *payload bytes*, entropy-constrained can afford higher K for
    tiles with low-entropy (concentrated) index distributions.

External feedback adjustments:
  - B (BiP preconditioning) as an allocation action: {K, K+B} where B adds
    ~0.5 bpw sidecar but changes the R-D curve (reduces distortion, increases
    entropy rate because rotation makes indices more uniform).
  - Sidecar compression: test how much of the B benefit survives at reduced
    sidecar rates (0.5, 0.25, 0.125, 0.0625 bpw).
  - Entropy + B interaction: B changes entropy — the allocator must account
    for the entropy rate increase from rotation when budgeting.

Experiments:
  1. Entropy estimation per tile per K
  2. Fixed-K DP vs entropy-constrained DP (same byte budget)
  3. Entropy + rotation composition (BiIP+Hadamard before/after)
  4. R-D frontier comparison (fixed-K rate vs entropy rate)
  5. Arithmetic coding simulation (order-0 adaptive)
  6. Inter-layer entropy allocation (extend R19)
  7. B-as-action: multi-precision alphabet {K, K+B} in the entropy DP
  8. Sidecar compression sweep: B benefit at reduced sidecar rates

Configuration: 4 real tensors, 3 slices, K=3,4,5,6
"""

import json
import os
import time
import warnings

import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
WORKDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(WORKDIR, "results.json")
FINDINGS_PATH = os.path.join(WORKDIR, "findings.md")

# ─── Configuration ────────────────────────────────────────────────────────────
TILE         = 16
SLICE        = 128
K_VALUES     = [3, 4, 5, 6]
K_MIN        = min(K_VALUES)
K_MAX        = max(K_VALUES)
N_CALIB      = 512
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
RNG_SEED     = 42
N_TILES      = (SLICE // TILE) ** 2       # 64
ELEM_PER_T   = TILE * TILE                 # 256

# B (BiP) sidecar: 128 bits per tile (16 bytes) = 0.5 bpw for 256-element tiles
B_SIDECAR_BITS_PER_TILE = 128  # 16 bytes: S_G diagonal (64 float16) + S_X diagonal (64 float16)
                                # But for 16x16 tiles, we store per-tile min/max for BiP = 4 float16 = 8 bytes
                                # Actually BiP scales are per-channel, shared across tiles in a row/col.
                                # For PoC: use 0.5 bpw = 128 bits/tile as the sidecar cost.
B_SIDECAR_BPW = 0.5  # bits per weight for the BiP sidecar

# Sidecar compression levels to test (from external feedback)
SIDECAR_LEVELS = [0.5, 0.25, 0.125, 0.0625, 0.03]

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities (reused from R20/R1/R19)
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_weights():
    data = np.load(WEIGHTS_PATH)
    return {k: data[k].astype(np.float64) for k in data.files}


def extract_slices(tensor, m=128, n=128, seed=42):
    """Extract 3 non-overlapping 128×128 slices from a large tensor."""
    M, N = tensor.shape
    rng = np.random.default_rng(seed)
    max_r = max(M - m, 1)
    max_c = max(N - n, 1)
    slices = []
    for i in range(3):
        r = int(rng.integers(0, max_r))
        c = int(rng.integers(0, max_c))
        r = (r + i * (M // 4)) % max_r
        c = (c + i * (N // 4)) % max_c
        slices.append((f"s{i}", tensor[r:r+m, c:c+n].copy()))
    return slices


def gen_calibration(n_in, n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_in)) * 0.5
    for i in range(0, n_in, 64):
        X[:, i:i+64] += rng.standard_normal((n_samples, 1)) * 0.3
    return X


def compute_hessians(W, X):
    N = X.shape[1]
    H_X = X.T @ X / X.shape[0]
    Y = X @ W.T
    H_G = Y.T @ Y / Y.shape[0]
    return H_G, H_X


# ═══════════════════════════════════════════════════════════════════════════════
# Quantizer
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_tile(w, k):
    if k <= 0:
        return np.zeros_like(w)
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
    if step == 0.0:
        return w.copy()
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_tile_indices(w, k):
    if k <= 0:
        return np.zeros_like(w, dtype=int)
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
    if step == 0.0:
        return np.zeros_like(w, dtype=int)
    return np.clip(np.round((w - lo) / step), 0, nl - 1).astype(int)


def quantize_tiles(W, K, tile=TILE):
    m, n = W.shape
    Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            ti, tj = i // tile, j // tile
            k = int(K[ti, tj]) if isinstance(K, np.ndarray) else int(K)
            r1, c1 = min(i + tile, m), min(j + tile, n)
            Wq[i:r1, j:c1] = quantize_tile(W[i:r1, j:c1], k)
    return Wq


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def hessian_weighted_error(E, H_G, H_X):
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Entropy estimation
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_entropy(indices):
    """Shannon entropy of quantization indices in bits."""
    if len(indices) == 0:
        return 0.0
    counts = np.bincount(indices)
    probs = counts[counts > 0] / len(indices)
    return float(-np.sum(probs * np.log2(probs)))


def tile_entropy(W_tile, k):
    """Empirical entropy H(indices) for a single tile at K=k."""
    indices = quantize_tile_indices(W_tile, k)
    return empirical_entropy(indices.flatten())


def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    """Hessian-weighted distortion of a tile at K=k."""
    Q_tile = quantize_tile(W_tile, k)
    E_t = W_tile - Q_tile
    D = np.trace(H_G_sub @ E_t @ H_X_sub @ E_t.T)
    return max(D, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# BiIP + Hadamard rotation (from R20)
# ═══════════════════════════════════════════════════════════════════════════════

def hadamard_matrix(n):
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return np.diag(signs) @ H, signs


def biip_scaling(W, H_X, H_G):
    d_out, d_in = W.shape
    diag_X = np.maximum(np.diag(H_X), 1e-10)
    diag_G = np.maximum(np.diag(H_G), 1e-10)
    diag_W = np.maximum(np.diag(W.T @ W), 1e-10)
    S_X = np.diag(diag_X ** 0.25 / diag_W ** 0.25)
    S_G = np.diag(diag_G ** 0.25 / np.maximum(np.diag(W @ W.T), 1e-10) ** 0.25)
    return S_G, S_X, S_G @ W @ S_X


def apply_biip_hadamard(W, H_X, H_G, rng):
    S_G, S_X, W_s = biip_scaling(W, H_X, H_G)
    U, _ = signed_random_hadamard(W.shape[0], rng)
    V, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U @ W_s @ V
    return W_t, (S_G, S_X, U, V)


def inverse_biip_hadamard(Wq_t, S_G, S_X, U, V):
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ Wq_t @ V.T @ S_X_inv


# ═══════════════════════════════════════════════════════════════════════════════
# Byte budget accounting
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bytes_fixed_k(K_flat, n_tiles=N_TILES, elements_per_tile=ELEM_PER_T):
    """Byte count for fixed-K (no entropy coding)."""
    total_k = int(np.sum(K_flat))
    payload_bytes = total_k * elements_per_tile // 8
    sidecar_bytes = n_tiles * 4
    if len(set(K_flat.tolist() if hasattr(K_flat, 'tolist') else list(K_flat))) == 1:
        metadata_bytes = 1
    else:
        metadata_bytes = (n_tiles * 3 + 7) // 8
    return payload_bytes + sidecar_bytes + metadata_bytes


def compute_bytes_entropy(K_flat, entropy_per_tile, n_tiles=N_TILES,
                          elements_per_tile=ELEM_PER_T, include_model=True):
    """Byte count for entropy-coded payload.
    - Payload: H_t * elements_per_tile bits → bytes
    - Sidecar: 4 bytes/tile (min/max float16)
    - K-metadata: 3 bits/tile
    - Entropy model: 2^K * 2 bytes/tile (float16 probabilities) if include_model
    """
    total_entropy_bits = float(np.sum(entropy_per_tile * elements_per_tile))
    payload_bytes = int(np.ceil(total_entropy_bits / 8))
    sidecar_bytes = n_tiles * 4
    metadata_bytes = (n_tiles * 3 + 7) // 8
    model_bytes = int(np.sum([2 ** k * 2 for k in K_flat])) if include_model else 0
    return payload_bytes + sidecar_bytes + metadata_bytes + model_bytes


def compute_bytes_with_b(K_flat, entropy_per_tile, b_sidecar_bpw, n_tiles=N_TILES,
                         elements_per_tile=ELEM_PER_T, include_model=False):
    """Byte count for entropy-coded payload + B sidecar.
    B sidecar = b_sidecar_bpw * elements_per_tile bits per tile.
    """
    total_entropy_bits = float(np.sum(entropy_per_tile * elements_per_tile))
    payload_bytes = int(np.ceil(total_entropy_bits / 8))
    sidecar_bytes = n_tiles * 4  # min/max for quantizer
    metadata_bytes = (n_tiles * 3 + 7) // 8  # K per tile
    b_sidecar_bytes = int(np.ceil(b_sidecar_bpw * elements_per_tile * n_tiles / 8))
    model_bytes = int(np.sum([2 ** k * 2 for k in K_flat])) if include_model else 0
    return payload_bytes + sidecar_bytes + metadata_bytes + b_sidecar_bytes + model_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# Precompute tile tables: distortion and entropy at each K
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_tile_tables(W, H_X, H_G, tile=TILE, k_values=K_VALUES):
    """For each tile and each K, measure distortion D_t(K) and entropy H_t(K)."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc

    D_table = np.zeros((n_tiles, len(k_values)))
    H_table = np.zeros((n_tiles, len(k_values)))

    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0+tile, c0:c0+tile]
            H_G_sub = H_G[r0:r0+tile, r0:r0+tile]
            H_X_sub = H_X[c0:c0+tile, c0:c0+tile]
            for ki, k in enumerate(k_values):
                D_table[t_idx, ki] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
                H_table[t_idx, ki] = tile_entropy(W_tile, k)
            t_idx += 1

    return D_table, H_table


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed-K DP allocation (from R1, vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def alloc_fixed_k_dp(D_table, budget_k, k_values=K_VALUES):
    """Fixed-K DP: minimize total distortion s.t. sum(K_t) <= budget_k."""
    n_tiles = D_table.shape[0]
    INF = float('inf')
    dp = np.full(budget_k + 1, INF)
    dp[0] = 0.0
    choices = []

    for t in range(n_tiles):
        new_dp = np.full(budget_k + 1, INF)
        new_choice = np.full(budget_k + 1, -1, dtype=int)
        for ki, k_val in enumerate(k_values):
            # Vectorized: for all j where dp[j] < INF, try adding k_val
            valid = dp < INF
            if not np.any(valid):
                break
            src_j = np.where(valid)[0]
            dst_j = src_j + k_val
            mask = dst_j <= budget_k
            if not np.any(mask):
                continue
            src_j = src_j[mask]
            dst_j = dst_j[mask]
            vals = dp[src_j] + D_table[t, ki]
            improved = vals < new_dp[dst_j]
            if np.any(improved):
                idx = dst_j[improved]
                new_dp[idx] = vals[improved]
                new_choice[idx] = ki
        dp = new_dp
        choices.append(new_choice.copy())

    best_j = int(np.argmin(dp))
    best_d = float(dp[best_j])

    K_flat = np.zeros(n_tiles, dtype=int)
    j = best_j
    for t in range(n_tiles - 1, -1, -1):
        ki = int(choices[t][j])
        if ki < 0:
            for fki in range(len(k_values)):
                if k_values[fki] <= j:
                    ki = fki
                    break
        K_flat[t] = k_values[ki]
        j -= k_values[ki]

    return K_flat, best_d


# ═══════════════════════════════════════════════════════════════════════════════
# Entropy-constrained allocation via Lagrangian relaxation (FAST)
# ═══════════════════════════════════════════════════════════════════════════════

def entropy_lagrangian_alloc(D_table, H_table, lam, k_values=K_VALUES,
                             extra_cost_per_tile=None):
    """For Lagrange multiplier λ, pick K_t minimizing D_t(K) + λ * (H_t(K) * E + extra).

    If extra_cost_per_tile is provided, it's a [n_tiles, len(k_values)] array of
    additional per-tile costs (e.g., B sidecar bytes, model overhead).
    """
    n_tiles = D_table.shape[0]
    K_flat = np.zeros(n_tiles, dtype=int)

    for t in range(n_tiles):
        if extra_cost_per_tile is not None:
            costs = D_table[t, :] + lam * (H_table[t, :] * ELEM_PER_T + extra_cost_per_tile[t, :])
        else:
            costs = D_table[t, :] + lam * H_table[t, :] * ELEM_PER_T
        K_flat[t] = k_values[int(np.argmin(costs))]

    return K_flat


def fixed_k_lagrangian_alloc(D_table, lam, k_values=K_VALUES,
                             extra_cost_per_tile=None):
    """Fixed-K Lagrangian: pick K_t minimizing D_t(K) + λ * (K_t * E + extra)."""
    n_tiles = D_table.shape[0]
    K_flat = np.zeros(n_tiles, dtype=int)

    k_arr = np.array(k_values, dtype=float)
    for t in range(n_tiles):
        if extra_cost_per_tile is not None:
            costs = D_table[t, :] + lam * (k_arr * ELEM_PER_T + extra_cost_per_tile[t, :])
        else:
            costs = D_table[t, :] + lam * k_arr * ELEM_PER_T
        K_flat[t] = k_values[int(np.argmin(costs))]

    return K_flat


def find_lambda_for_budget(D_table, H_table, target_budget_bits, k_values=K_VALUES,
                           alloc_fn=None, cost_fn=None, lam_lo=1e-12, lam_hi=1e6,
                           n_iters=50):
    """Binary search for λ such that the Lagrangian allocation has total cost ≈ target.
    alloc_fn(D_table, H_table, lam) → K_flat
    cost_fn(K_flat) → total cost in bits
    """
    if alloc_fn is None:
        alloc_fn = lambda D, H, lam: entropy_lagrangian_alloc(D, H, lam, k_values)
    if cost_fn is None:
        def cost_fn(K_flat):
            return float(np.sum([H_table[t, k_values.index(K_flat[t])] * ELEM_PER_T
                                 for t in range(len(K_flat))]))

    for _ in range(n_iters):
        lam_mid = np.sqrt(lam_lo * lam_hi)
        K_flat = alloc_fn(D_table, H_table, lam_mid)
        cost = cost_fn(K_flat)
        if cost > target_budget_bits:
            lam_lo = lam_mid  # need higher λ to reduce cost
        else:
            lam_hi = lam_mid  # can afford lower λ for less distortion

    # Return the allocation with cost <= target (conservative)
    lam_final = lam_hi
    K_flat = alloc_fn(D_table, H_table, lam_final)
    return K_flat, lam_final


# ═══════════════════════════════════════════════════════════════════════════════
# Arithmetic coding simulation
# ═══════════════════════════════════════════════════════════════════════════════

def arithmetic_code_length(indices, n_symbols):
    """Simulate order-0 arithmetic coding length in bits."""
    n = len(indices)
    if n == 0:
        return 0.0
    h = empirical_entropy(indices)
    return h * n + 2.0  # ~2 bits overhead per block


def simulate_arithmetic_coding(W, K_grid, tile=TILE):
    """Simulate arithmetic coding of the full matrix and return total coded bits."""
    m, n = W.shape
    ntr, ntc = m // tile, n // tile
    total_bits = 0.0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            w_tile = W[r0:r0+tile, c0:c0+tile]
            k = int(K_grid[ti, tj])
            indices = quantize_tile_indices(w_tile, k)
            total_bits += arithmetic_code_length(indices.flatten(), 2 ** k)
    return total_bits


# ═══════════════════════════════════════════════════════════════════════════════
# Local search refinement
# ═══════════════════════════════════════════════════════════════════════════════

def local_search_refine(W, K_flat, H_G, H_X, budget_fn, tile=TILE, max_iters=300):
    """Full-objective local search. Swaps K between tiles if it improves HWE
    while staying within budget (budget_fn returns True if within budget)."""
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc

    def full_hwe(K_arr):
        K_grid = K_arr.reshape(ntr, ntc)
        Wq = quantize_tiles(W, K_grid, tile)
        E = W - Wq
        return hessian_weighted_error(E, H_G, H_X)

    K = K_flat.copy()
    current_hwe = full_hwe(K)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for donor in range(n_tiles):
            if K[donor] <= K_MIN:
                continue
            for receiver in range(n_tiles):
                if receiver == donor or K[receiver] >= K_MAX:
                    continue
                K[donor] -= 1
                K[receiver] += 1
                if not budget_fn(K):
                    K[donor] += 1
                    K[receiver] -= 1
                    continue
                new_hwe = full_hwe(K)
                if new_hwe < current_hwe - 1e-15:
                    current_hwe = new_hwe
                    improved = True
                    break
                else:
                    K[donor] += 1
                    K[receiver] -= 1
            if improved:
                break
    return K, current_hwe


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def eval_hwe(W, K_flat, H_G, H_X, tile=TILE):
    ntr, ntc = W.shape[0] // tile, W.shape[1] // tile
    K_grid = K_flat.reshape(ntr, ntc)
    Wq = quantize_tiles(W, K_grid, tile)
    E = W - Wq
    return hessian_weighted_error(E, H_G, H_X)


def eval_hwe_rotated(W, K_flat, W_rot, transforms, H_G, H_X, tile=TILE):
    """Evaluate HWE for allocation on rotated weights, measured in original space."""
    S_G, S_X, U, V = transforms
    ntr, ntc = W.shape[0] // tile, W.shape[1] // tile
    K_grid = K_flat.reshape(ntr, ntc)
    Wq_rot = quantize_tiles(W_rot, K_grid, tile)
    W_hat = inverse_biip_hadamard(Wq_rot, S_G, S_X, U, V)
    E = W - W_hat
    return hessian_weighted_error(E, H_G, H_X)


def eval_entropy(H_table, K_flat, k_values=K_VALUES):
    return float(np.sum([H_table[t, k_values.index(K_flat[t])] * ELEM_PER_T
                         for t in range(len(K_flat))]))


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Entropy estimation per tile per K
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_entropy_estimation(W, H_G, H_X, label=""):
    D_table, H_table = precompute_tile_tables(W, H_X, H_G)
    n_tiles = D_table.shape[0]
    results = {'label': label, 'n_tiles': n_tiles, 'per_K': {}}

    print(f"\n  [Exp1] Entropy estimation{' (' + label + ')' if label else ''}:")
    for ki, k in enumerate(K_VALUES):
        entropies = H_table[:, ki]
        mean_h = float(np.mean(entropies))
        std_h = float(np.std(entropies))
        savings = (k - mean_h) / k * 100
        results['per_K'][f'K{k}'] = {
            'fixed_rate': k, 'mean_entropy': mean_h, 'std_entropy': std_h,
            'min_entropy': float(np.min(entropies)), 'max_entropy': float(np.max(entropies)),
            'savings_pct': savings,
            'total_entropy_bits': float(np.sum(entropies * ELEM_PER_T)),
            'total_fixed_bits': k * n_tiles * ELEM_PER_T,
        }
        print(f"    K={k}: H_mean={mean_h:.3f} ± {std_h:.3f} bits, savings={savings:.1f}%")

    results['D_table'] = D_table.tolist()
    results['H_table'] = H_table.tolist()
    return results, D_table, H_table


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Fixed-K DP vs entropy-constrained DP
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_alloc_comparison(W, H_G, H_X, D_table, H_table, label=""):
    n_tiles = D_table.shape[0]
    results = {'label': label, 'budgets': []}

    print(f"\n  [Exp2] Allocation comparison{' (' + label + ')' if label else ''}:")

    for avg_k in [3, 4, 5, 6]:
        # ── Uniform-K baseline ──
        K_uniform = np.full(n_tiles, avg_k, dtype=int)
        bytes_uniform = compute_bytes_fixed_k(K_uniform, n_tiles)
        HWE_uniform = eval_hwe(W, K_uniform, H_G, H_X)
        ent_uniform = eval_entropy(H_table, K_uniform)

        # ── Fixed-K DP ──
        # Use pure payload budget: ΣK = n_tiles × avg_K (matches uniform payload exactly)
        # This is the fair comparison — both arms get the same payload bits.
        # Sidecar (4B/tile) and metadata are identical or negligible for both.
        budget_k = n_tiles * avg_k
        K_fixed_dp, _ = alloc_fixed_k_dp(D_table, budget_k)
        bytes_fixed_dp = compute_bytes_fixed_k(K_fixed_dp, n_tiles)
        HWE_fixed_dp = eval_hwe(W, K_fixed_dp, H_G, H_X)

        # ── Entropy-constrained DP (Lagrangian, no model overhead) ──
        # Target: same payload bytes as uniform-K
        # This is an empirical-entropy LOWER BOUND on achievable coded rate.
        # Real arithmetic coding would add model/framing overhead (see Exp5 caveat).
        uniform_payload_bytes = avg_k * n_tiles * ELEM_PER_T / 8
        target_entropy_bits = uniform_payload_bytes * 8

        K_ent, _ = find_lambda_for_budget(
            D_table, H_table, target_entropy_bits, K_VALUES,
            alloc_fn=lambda D, H, lam: entropy_lagrangian_alloc(D, H, lam, K_VALUES),
            cost_fn=lambda K: eval_entropy(H_table, K, K_VALUES))

        ent_rates = np.array([H_table[t, K_VALUES.index(K_ent[t])] for t in range(n_tiles)])
        bytes_ent = compute_bytes_entropy(K_ent, ent_rates, n_tiles, include_model=False)
        HWE_ent = eval_hwe(W, K_ent, H_G, H_X)

        # ── Entropy DP with per-tile model overhead ──
        # Model: 2^K × 2 bytes per tile (float16 probability table).
        # Include model cost IN the Lagrangian action (reviewer fix).
        # cost_t(K) = H_t(K) × 256 + 2^K × 2 × 8 bits (model table)
        model_cost_table = np.array([2 ** k * 2 * 8 for k in K_VALUES])  # bits per tile per K
        extra_cost_model = np.tile(model_cost_table[None, :], (n_tiles, 1))  # (n_tiles, |K|)

        # Total budget = uniform payload + sidecar + meta - sidecar - meta = uniform payload
        # But model overhead eats into payload budget
        target_with_model = (bytes_uniform - n_tiles * 4) * 8  # payload + meta budget

        K_ent_model, _ = find_lambda_for_budget(
            D_table, H_table, target_with_model, K_VALUES,
            alloc_fn=lambda D, H, lam: entropy_lagrangian_alloc(
                D, H, lam, K_VALUES, extra_cost_per_tile=extra_cost_model),
            cost_fn=lambda K: eval_entropy(H_table, K) + float(np.sum(
                [model_cost_table[K_VALUES.index(K[t])] for t in range(len(K))])))

        ent_rates_m = np.array([H_table[t, K_VALUES.index(K_ent_model[t])] for t in range(n_tiles)])
        bytes_ent_model = compute_bytes_entropy(K_ent_model, ent_rates_m, n_tiles, include_model=True)
        HWE_ent_model = eval_hwe(W, K_ent_model, H_G, H_X)

        # Improvements
        imp_fixed = (HWE_uniform - HWE_fixed_dp) / HWE_uniform * 100 if HWE_uniform > 1e-15 else 0
        imp_ent = (HWE_fixed_dp - HWE_ent) / HWE_fixed_dp * 100 if HWE_fixed_dp > 1e-15 else 0
        imp_ent_model = (HWE_fixed_dp - HWE_ent_model) / HWE_fixed_dp * 100 if HWE_fixed_dp > 1e-15 else 0

        entry = {
            'avg_k': avg_k,
            'uniform': {'bytes': bytes_uniform, 'hwe': HWE_uniform, 'total_entropy_bits': ent_uniform},
            'fixed_k_dp': {'bytes': bytes_fixed_dp, 'hwe': HWE_fixed_dp,
                           'improvement_vs_uniform_pct': imp_fixed, 'k_alloc': K_fixed_dp.tolist()},
            'entropy_dp': {'bytes': bytes_ent, 'hwe': HWE_ent,
                           'improvement_vs_fixed_pct': imp_ent, 'k_alloc': K_ent.tolist()},
            'entropy_dp_model': {'bytes': bytes_ent_model, 'hwe': HWE_ent_model,
                                 'improvement_vs_fixed_pct': imp_ent_model, 'k_alloc': K_ent_model.tolist()},
        }
        results['budgets'].append(entry)

        print(f"    avg_K={avg_k}: uniform HWE={HWE_uniform:.4e} ({bytes_uniform}B)")
        print(f"      fixed-K DP:  HWE={HWE_fixed_dp:.4e} ({bytes_fixed_dp}B), Δ={imp_fixed:+.1f}% vs uniform")
        print(f"      entropy DP:  HWE={HWE_ent:.4e} ({bytes_ent}B), Δ={imp_ent:+.1f}% vs fixed-K [no model]")
        print(f"      entropy DP+: HWE={HWE_ent_model:.4e} ({bytes_ent_model}B), Δ={imp_ent_model:+.1f}% vs fixed-K [w/ model]")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Entropy + rotation composition
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_rotation_composition(W, H_G, H_X, rng, label=""):
    n_tiles = N_TILES

    # ── Before rotation ──
    D_before, H_before = precompute_tile_tables(W, H_X, H_G)
    ent_before = {k: float(np.mean(H_before[:, ki])) for ki, k in enumerate(K_VALUES)}
    # ── After rotation ──
    W_rot, transforms = apply_biip_hadamard(W, H_X, H_G, rng)
    S_G, S_X, U, V = transforms
    # For W' = U @ S_G @ W @ S_X @ V, the inverse is W_hat = S_G⁻¹ @ Uᵀ @ Wq' @ Vᵀ @ S_X⁻¹
    # The HWE tr(H_G E H_X Eᵀ) transforms to:
    # tr(H_G' E' H_X' E'ᵀ) where H_G' = U S_G⁻¹ H_G S_G⁻¹ Uᵀ, H_X' = Vᵀ S_X⁻¹ H_X S_X⁻¹ V
    # (reviewer fix: use transformed Hessians for rotated distortion tables)
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_G_rot = U @ S_G_inv @ H_G @ S_G_inv @ U.T
    H_X_rot = V.T @ S_X_inv @ H_X @ S_X_inv @ V
    D_after, H_after = precompute_tile_tables(W_rot, H_X_rot, H_G_rot)
    ent_after = {k: float(np.mean(H_after[:, ki])) for ki, k in enumerate(K_VALUES)}

    results = {
        'label': label,
        'entropy_before_rotation': ent_before,
        'entropy_after_rotation': ent_after,
        'entropy_change': {},
    }

    print(f"\n  [Exp3] Rotation composition{' (' + label + ')' if label else ''}:")
    for k in K_VALUES:
        d = ent_after[k] - ent_before[k]
        pct = d / ent_before[k] * 100 if ent_before[k] > 1e-6 else 0
        results['entropy_change'][f'K{k}'] = float(d)
        results['entropy_change'][f'K{k}_pct'] = pct
        print(f"    K={k}: H_before={ent_before[k]:.3f}, H_after={ent_after[k]:.3f}, "
              f"Δ={d:+.3f} ({pct:+.1f}%)")

    # ── Allocation comparison after rotation ──
    results['rotation_alloc'] = {}
    for avg_k in [4, 5]:
        K_uniform = np.full(n_tiles, avg_k, dtype=int)
        bytes_uniform = compute_bytes_fixed_k(K_uniform, n_tiles)
        # Fixed-K DP before rotation (pure payload budget)
        budget_k = n_tiles * avg_k
        K_fixed_before, _ = alloc_fixed_k_dp(D_before, budget_k)
        HWE_fixed_before = eval_hwe(W, K_fixed_before, H_G, H_X)

        # Fixed-K DP after rotation
        K_fixed_after, _ = alloc_fixed_k_dp(D_after, budget_k)
        HWE_fixed_after = eval_hwe_rotated(W, K_fixed_after, W_rot, transforms, H_G, H_X)

        # Entropy DP before rotation
        target_bits = avg_k * n_tiles * ELEM_PER_T
        K_ent_before, _ = find_lambda_for_budget(
            D_before, H_before, target_bits, K_VALUES,
            alloc_fn=lambda D, H, lam: entropy_lagrangian_alloc(D, H, lam, K_VALUES),
            cost_fn=lambda K: eval_entropy(H_before, K, K_VALUES))
        HWE_ent_before = eval_hwe(W, K_ent_before, H_G, H_X)

        # Entropy DP after rotation
        K_ent_after, _ = find_lambda_for_budget(
            D_after, H_after, target_bits, K_VALUES,
            alloc_fn=lambda D, H, lam: entropy_lagrangian_alloc(D, H, lam, K_VALUES),
            cost_fn=lambda K: eval_entropy(H_after, K, K_VALUES))
        HWE_ent_after = eval_hwe_rotated(W, K_ent_after, W_rot, transforms, H_G, H_X)

        results['rotation_alloc'][f'avg_K{avg_k}'] = {
            'fixed_k_before_rot': {'hwe': HWE_fixed_before},
            'fixed_k_after_rot': {'hwe': HWE_fixed_after},
            'entropy_before_rot': {'hwe': HWE_ent_before},
            'entropy_after_rot': {'hwe': HWE_ent_after},
        }

        imp_rot_fixed = (HWE_fixed_before - HWE_fixed_after) / HWE_fixed_before * 100
        imp_ent_before = (HWE_fixed_before - HWE_ent_before) / HWE_fixed_before * 100
        imp_ent_after = (HWE_fixed_after - HWE_ent_after) / HWE_fixed_after * 100
        imp_combined = (HWE_fixed_before - HWE_ent_after) / HWE_fixed_before * 100

        print(f"    avg_K={avg_k} allocation:")
        print(f"      fixed-K  before rot: HWE={HWE_fixed_before:.4e}")
        print(f"      fixed-K  after rot:  HWE={HWE_fixed_after:.4e} ({imp_rot_fixed:+.1f}% from rotation)")
        print(f"      entropy  before rot: HWE={HWE_ent_before:.4e} ({imp_ent_before:+.1f}% vs fixed-K)")
        print(f"      entropy  after rot:  HWE={HWE_ent_after:.4e} ({imp_ent_after:+.1f}% vs fixed-K+rot)")
        print(f"      combined (ent+rot):  HWE={HWE_ent_after:.4e} ({imp_combined:+.1f}% vs fixed-K only)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: R-D frontier comparison
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_rd_frontier(W, H_G, H_X, D_table, H_table, label=""):
    n_tiles = D_table.shape[0]
    n_points = 50

    # λ range
    d_min = np.min(D_table[D_table > 0])
    d_max = np.max(D_table)
    lams = np.exp(np.linspace(np.log(d_max / (K_MAX * ELEM_PER_T) * 1e-3),
                              np.log(d_min / (K_MIN * ELEM_PER_T) * 1e3), n_points))

    # ── Fixed-K Lagrangian frontier ──
    frontier_fixed = []
    for lam in lams:
        K_flat = fixed_k_lagrangian_alloc(D_table, lam, K_VALUES)
        bytes_ = compute_bytes_fixed_k(K_flat, n_tiles)
        hwe = eval_hwe(W, K_flat, H_G, H_X)
        frontier_fixed.append({'lambda': float(lam), 'bytes': bytes_, 'hwe': hwe,
                               'k_mean': float(np.mean(K_flat))})

    # ── Entropy Lagrangian frontier ──
    frontier_entropy = []
    for lam in lams:
        K_flat = entropy_lagrangian_alloc(D_table, H_table, lam, K_VALUES)
        ent_rates = np.array([H_table[t, K_VALUES.index(K_flat[t])] for t in range(n_tiles)])
        bytes_nm = compute_bytes_entropy(K_flat, ent_rates, n_tiles, include_model=False)
        bytes_wm = compute_bytes_entropy(K_flat, ent_rates, n_tiles, include_model=True)
        hwe = eval_hwe(W, K_flat, H_G, H_X)
        frontier_entropy.append({'lambda': float(lam), 'bytes_nomodel': bytes_nm,
                                 'bytes_withmodel': bytes_wm, 'hwe': hwe,
                                 'k_mean': float(np.mean(K_flat))})

    # ── Pareto extraction ──
    def pareto(frontier, byte_key):
        sorted_pts = sorted(frontier, key=lambda p: p[byte_key])
        pareto_pts = []
        best_hwe = float('inf')
        for p in sorted_pts:
            if p['hwe'] < best_hwe:
                best_hwe = p['hwe']
                pareto_pts.append(p)
        return pareto_pts

    pareto_fixed = pareto(frontier_fixed, 'bytes')
    pareto_ent = pareto(frontier_entropy, 'bytes_nomodel')
    pareto_ent_m = pareto(frontier_entropy, 'bytes_withmodel')

    results = {
        'label': label,
        'frontier_fixed_k': frontier_fixed,
        'frontier_entropy': frontier_entropy,
        'pareto_fixed_k': pareto_fixed,
        'pareto_entropy_nomodel': pareto_ent,
        'pareto_entropy_model': pareto_ent_m,
    }

    print(f"\n  [Exp4] R-D frontier{' (' + label + ')' if label else ''}:")
    print(f"    Pareto points: fixed-K={len(pareto_fixed)}, entropy(nm)={len(pareto_ent)}, entropy(wm)={len(pareto_ent_m)}")

    # Compare at matched byte budgets
    if pareto_fixed and pareto_ent:
        all_bytes = sorted(set(p['bytes'] for p in pareto_fixed) |
                           set(p['bytes_nomodel'] for p in pareto_ent))
        print(f"    Matched budget comparison:")
        for b in all_bytes[:8]:
            hwe_f = min((p['hwe'] for p in pareto_fixed if p['bytes'] <= b), default=float('inf'))
            hwe_e = min((p['hwe'] for p in pareto_ent if p['bytes_nomodel'] <= b), default=float('inf'))
            if hwe_f < float('inf') and hwe_e < float('inf') and hwe_f > 1e-15:
                gain = (hwe_f - hwe_e) / hwe_f * 100
                print(f"      B={b}: fixed-K={hwe_f:.4e}, entropy={hwe_e:.4e}, gain={gain:+.1f}%")

    return results


# Experiment 5: Empirical entropy lower bound (NOT a real arithmetic coder)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_entropy_lower_bound(W, H_G, H_X, label=""):
    """Compute the empirical-entropy lower bound on coded rate.
    NOTE: This is NOT a real arithmetic coding implementation. It computes
    the Shannon entropy H(indices) as a theoretical lower bound on what
    any entropy coder could achieve. A real arithmetic/range coder would
    add model overhead (per-tile CDF tables, framing, termination bits).
    See R24 findings for the distinction between this bound and achievable rate."""

def experiment_entropy_lower_bound(W, H_G, H_X, label=""):
    n_tiles = N_TILES
    results = {'label': label, 'per_K': {}}

    print(f"\n  [Exp5] Empirical entropy lower bound{' (' + label + ')' if label else ''}:")
    for k in K_VALUES:
        K_uniform = np.full(n_tiles, k, dtype=int)
        K_grid = K_uniform.reshape(W.shape[0] // TILE, W.shape[1] // TILE)

        # Per-tile entropy
        ent_rates = np.zeros((W.shape[0] // TILE, W.shape[1] // TILE))
        for ti in range(W.shape[0] // TILE):
            for tj in range(W.shape[1] // TILE):
                r0, c0 = ti * TILE, tj * TILE
                ent_rates[ti, tj] = tile_entropy(W[r0:r0+TILE, c0:c0+TILE], k)
        theoretical_bits = float(np.sum(ent_rates * ELEM_PER_T))

        simulated_bits = simulate_arithmetic_coding(W, K_grid, TILE)
        fixed_bits = k * n_tiles * ELEM_PER_T
        cr_th = fixed_bits / theoretical_bits if theoretical_bits > 0 else 0
        cr_sim = fixed_bits / simulated_bits if simulated_bits > 0 else 0

        results['per_K'][f'K{k}'] = {
            'fixed_bits': fixed_bits,
            'theoretical_entropy_bits': theoretical_bits,
            'simulated_ac_bits': simulated_bits,
            'compression_ratio_theoretical': cr_th,
            'compression_ratio_simulated': cr_sim,
        }
        print(f"    K={k}: fixed={fixed_bits}b, H={theoretical_bits:.0f}b, "
              f"AC={simulated_bits:.0f}b, CR={cr_sim:.2f}x")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 6: Inter-layer entropy allocation (extend R19)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_interlayer_entropy(tensors):
    """Extend R19's inter-layer DP to use entropy instead of fixed K."""
    X = gen_calibration(SLICE, N_CALIB, RNG_SEED)

    items = []
    for tname in TENSOR_NAMES:
        if tname not in tensors:
            continue
        tensor = tensors[tname]
        W = tensor[:SLICE, :SLICE].copy()
        H_G, H_X = compute_hessians(W, X)
        D_table, H_table = precompute_tile_tables(W, H_X, H_G)

        item = {
            'id': tname,
            'D_table': D_table, 'H_table': H_table,
            'distortion': {k: float(np.sum(D_table[:, ki])) for ki, k in enumerate(K_VALUES)},
            'entropy': {k: float(np.sum(H_table[:, ki]) * ELEM_PER_T) for ki, k in enumerate(K_VALUES)},
            'n_tiles': D_table.shape[0],
        }
        items.append(item)

    results = {'items': [it['id'] for it in items], 'budgets': []}

    print(f"\n  [Exp6] Inter-layer entropy allocation ({len(items)} tensors):")

    for avg_k in [4, 5]:
        n_items = len(items)
        total_tiles = sum(it['n_tiles'] for it in items)

        # ── Fixed-K interlayer: brute-force (only 4 items × 4 K = 256 combos) ──
        import itertools
        k_budget = avg_k * total_tiles
        best_fixed = None
        best_d_fixed = float('inf')
        for combo in itertools.product(K_VALUES, repeat=n_items):
            k_sum = sum(combo[i] * items[i]['n_tiles'] for i in range(n_items))
            if k_sum > k_budget:
                continue
            d = sum(items[i]['distortion'][combo[i]] for i in range(n_items))
            if d < best_d_fixed:
                best_d_fixed = d
                best_fixed = {items[i]['id']: combo[i] for i in range(n_items)}

        if best_fixed is None:
            best_fixed = {it['id']: avg_k for it in items}
            best_d_fixed = sum(it['distortion'][avg_k] for it in items)

        total_ent_fixed = sum(it['entropy'][best_fixed[it['id']]] for it in items)

        # ── Entropy interlayer: brute-force ──
        # Budget: same total bytes as fixed-K uniform avg_k
        bytes_uniform = sum(compute_bytes_fixed_k(
            np.full(it['n_tiles'], avg_k), it['n_tiles']) for it in items)
        # Entropy budget = total_bytes - sidecar - meta (for each tensor)
        # Simplified: use same total payload budget as uniform
        ent_budget = avg_k * total_tiles * ELEM_PER_T  # in entropy bits

        best_ent = None
        best_d_ent = float('inf')
        for combo in itertools.product(K_VALUES, repeat=n_items):
            e_sum = sum(items[i]['entropy'][combo[i]] for i in range(n_items))
            if e_sum > ent_budget:
                continue
            d = sum(items[i]['distortion'][combo[i]] for i in range(n_items))
            if d < best_d_ent:
                best_d_ent = d
                best_ent = {items[i]['id']: combo[i] for i in range(n_items)}

        if best_ent is None:
            best_ent = {it['id']: avg_k for it in items}
            best_d_ent = sum(it['distortion'][avg_k] for it in items)

        total_ent_ent = sum(it['entropy'][best_ent[it['id']]] for it in items)

        entry = {
            'avg_k': avg_k,
            'fixed_k': {'k_assignment': best_fixed, 'total_distortion': best_d_fixed,
                        'total_entropy_bits': total_ent_fixed},
            'entropy': {'k_assignment': best_ent, 'total_distortion': best_d_ent,
                        'total_entropy_bits': total_ent_ent},
            'distortion_gain_pct': (best_d_fixed - best_d_ent) / best_d_fixed * 100 if best_d_fixed > 1e-15 else 0,
        }
        results['budgets'].append(entry)

        print(f"    avg_K={avg_k}:")
        print(f"      fixed-K: K={best_fixed}, D={best_d_fixed:.4e}, H={total_ent_fixed:.0f}b")
        print(f"      entropy: K={best_ent}, D={best_d_ent:.4e}, H={total_ent_ent:.0f}b")
        print(f"      gain: D {entry['distortion_gain_pct']:+.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
def experiment_b_as_action(W, H_G, H_X, rng, label=""):
    """Test B (BiP preconditioning) as a per-tensor allocation action.

    B is a full-matrix transform (BiIP+Hadamard) — it can't be applied per-tile.
    The allocator decides per-tensor: use K bits without B, or K bits with B
    (where B costs ~0.5 bpw sidecar but reduces distortion).

    Key comparison (from external feedback): K5+B at 5.5 bpw vs K6 at 6.0 bpw.
    If K5+B beats K6 at the same total bytes, that's a significant result.

    We compare 4 strategies at matched total bpw:
    1. Uniform K (no B) — baseline
    2. Uniform K+B — BiP rotation at same K, +0.5 bpw sidecar
    3. Entropy DP (no B) — entropy-constrained allocation
    4. Entropy DP+B — entropy-constrained allocation on rotated tensor
    """
    n_tiles = N_TILES

    # Precompute unrotated tables
    D_unrot, H_unrot = precompute_tile_tables(W, H_X, H_G)

    # Apply BiIP+Hadamard
    W_rot, transforms = apply_biip_hadamard(W, H_X, H_G, rng)
    S_G, S_X, U, V = transforms
    # Use transformed Hessians for rotated distortion tables (reviewer fix)
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_G_rot = U @ S_G_inv @ H_G @ S_G_inv @ U.T
    H_X_rot = V.T @ S_X_inv @ H_X @ S_X_inv @ V
    D_rot, H_rot = precompute_tile_tables(W_rot, H_X_rot, H_G_rot)

    results = {'label': label, 'budgets': []}

    print(f"\n  [Exp7] B-as-action (per-tensor){' (' + label + ')' if label else ''}:")

    for target_bpw in [4.0, 4.5, 5.0, 5.5, 6.0]:
        target_bits = target_bpw * n_tiles * ELEM_PER_T

        # ── Strategy 1: Uniform K (no B) ──
        k_uni = min(int(np.floor(target_bpw)), K_MAX)
        if k_uni < K_MIN:
            k_uni = K_MIN
        K_uniform = np.full(n_tiles, k_uni, dtype=int)
        HWE_uni = eval_hwe(W, K_uniform, H_G, H_X)
        bytes_uni = compute_bytes_fixed_k(K_uniform, n_tiles)

        # ── Strategy 2: Uniform K+B (BiP rotation) ──
        # With B, the effective rate is K + B_SIDECAR_BPW.
        # To fit in target_bpw: K = floor(target_bpw - B_SIDECAR_BPW)
        k_with_b = min(int(np.floor(target_bpw - B_SIDECAR_BPW)), K_MAX)
        if k_with_b < K_MIN:
            # Can't afford B at this rate
            HWE_uni_b = None
            bytes_uni_b = None
        else:
            K_uniform_b = np.full(n_tiles, k_with_b, dtype=int)
            Wq_rot = quantize_tiles(W_rot, K_uniform_b.reshape(W.shape[0]//TILE, W.shape[1]//TILE), TILE)
            W_hat_b = inverse_biip_hadamard(Wq_rot, S_G, S_X, U, V)
            HWE_uni_b = hessian_weighted_error(W - W_hat_b, H_G, H_X)
            # Bytes: payload (K bits/elem) + sidecar (4B/tile) + B sidecar (0.5bpw) + meta
            bytes_uni_b = compute_bytes_with_b(K_uniform_b,
                np.full(n_tiles, H_rot[:, K_VALUES.index(k_with_b)].mean()),
                B_SIDECAR_BPW, n_tiles)

        # ── Strategy 3: Entropy DP (no B) ──
        K_ent, _ = find_lambda_for_budget(
            D_unrot, H_unrot, target_bits, K_VALUES,
            alloc_fn=lambda D, H, lam: entropy_lagrangian_alloc(D, H, lam, K_VALUES),
            cost_fn=lambda K: eval_entropy(H_unrot, K, K_VALUES))
        HWE_ent = eval_hwe(W, K_ent, H_G, H_X)

        # ── Strategy 4: Entropy DP+B ──
        # Budget: target_bits - B sidecar bits
        b_sidecar_bits = B_SIDECAR_BPW * n_tiles * ELEM_PER_T
        ent_budget_with_b = target_bits - b_sidecar_bits
        if ent_budget_with_b < n_tiles * K_MIN * ELEM_PER_T:
            HWE_ent_b = None
        else:
            K_ent_b, _ = find_lambda_for_budget(
                D_rot, H_rot, ent_budget_with_b, K_VALUES,
                alloc_fn=lambda D, H, lam: entropy_lagrangian_alloc(D, H, lam, K_VALUES),
                cost_fn=lambda K: eval_entropy(H_rot, K, K_VALUES))
            # Evaluate in original space
            Wq_rot = quantize_tiles(W_rot, K_ent_b.reshape(W.shape[0]//TILE, W.shape[1]//TILE), TILE)
            W_hat_eb = inverse_biip_hadamard(Wq_rot, S_G, S_X, U, V)
            HWE_ent_b = hessian_weighted_error(W - W_hat_eb, H_G, H_X)

        entry = {
            'target_bpw': target_bpw,
            'uniform_k': {'k': k_uni, 'hwe': HWE_uni, 'bytes': bytes_uni},
            'uniform_k_b': {'k': k_with_b, 'hwe': HWE_uni_b, 'bytes': bytes_uni_b} if HWE_uni_b else None,
            'entropy_dp': {'hwe': HWE_ent, 'k_mean': float(np.mean(K_ent))},
            'entropy_dp_b': {'hwe': HWE_ent_b, 'k_mean': float(np.mean(K_ent_b))} if HWE_ent_b else None,
        }
        results['budgets'].append(entry)

        print(f"    target={target_bpw} bpw:")
        print(f"      uniform K{k_uni}:     HWE={HWE_uni:.4e}")
        if HWE_uni_b is not None:
            imp_b = (HWE_uni - HWE_uni_b) / HWE_uni * 100 if HWE_uni > 1e-15 else 0
            print(f"      uniform K{k_with_b}+B:  HWE={HWE_uni_b:.4e} ({imp_b:+.1f}% vs K{k_uni})")
        else:
            print(f"      uniform K+B:       N/A (can't afford B at {target_bpw} bpw)")
        print(f"      entropy DP:        HWE={HWE_ent:.4e}")
        if HWE_ent_b is not None:
            imp_eb = (HWE_ent - HWE_ent_b) / HWE_ent * 100 if HWE_ent > 1e-15 else 0
            print(f"      entropy DP+B:      HWE={HWE_ent_b:.4e} ({imp_eb:+.1f}% vs entropy)")
            imp_combined = (HWE_uni - HWE_ent_b) / HWE_uni * 100 if HWE_uni > 1e-15 else 0
            print(f"      combined (ent+B):  {imp_combined:+.1f}% vs uniform K{k_uni}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 8: Sidecar compression sweep
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_sidecar_sweep(W, H_G, H_X, rng, label=""):
    """Compare Hadamard-only vs full BiIP+Hadamard, with correct sidecar accounting.

    The BiIP sidecar consists of per-channel diagonal scales S_G (128 float16)
    and S_X (128 float16) = 512 bytes total for the 128x128 matrix.
    At 16384 weights, that's 512*8/16384 = 0.25 bpw (not 0.5 as initially stated).

    We test four arms:
    1. No rotation (baseline)
    2. Hadamard-only (no BiIP scales -- zero sidecar, just sign vectors)
    3. BiIP-8bit (quantized scales, 0.125 bpw sidecar)
    4. Full BiIP+Hadamard (0.25 bpw sidecar for per-channel float16 scales)
    """
    n_tiles = N_TILES
    n_weights = SLICE * SLICE  # 16384

    # Full BiIP+Hadamard
    W_rot, transforms = apply_biip_hadamard(W, H_X, H_G, rng)
    S_G, S_X, U, V = transforms
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)

    # Hadamard-only (no BiIP scaling)
    W_had = U @ W @ V

    # Compressed BiIP: quantize diagonals to 8-bit
    sg_8bit = _quantize_vector(np.diag(S_G), 8)
    sx_8bit = _quantize_vector(np.diag(S_X), 8)
    S_G_8 = np.diag(sg_8bit)
    S_X_8 = np.diag(sx_8bit)
    S_G_8_inv = np.linalg.inv(S_G_8)
    S_X_8_inv = np.linalg.inv(S_X_8)
    W_rot_8 = U @ S_G_8 @ W @ S_X_8 @ V

    sidecar_full_bpw = 512 * 8 / n_weights  # 0.25
    sidecar_8bit_bpw = 256 * 8 / n_weights  # 0.125

    results = {'label': label, 'arms': {}}
    print(f"\n  [Exp8] Sidecar comparison{' (' + label + ')' if label else ''}:")
    print(f"    Sidecar: Had=0 bpw, BiIP-8bit={sidecar_8bit_bpw:.4f} bpw, "
          f"BiIP-full={sidecar_full_bpw:.4f} bpw")

    arms = {
        'no_rotation': {'src': W, 'fn': lambda Wq: Wq, 'sc': 0.0},
        'hadamard_only': {'src': W_had, 'fn': lambda Wq: U.T @ Wq @ V, 'sc': 0.0},
        'biip_8bit': {'src': W_rot_8,
                      'fn': lambda Wq: S_G_8_inv @ U.T @ Wq @ V.T @ S_X_8_inv,
                      'sc': sidecar_8bit_bpw},
        'biip_full': {'src': W_rot,
                      'fn': lambda Wq: S_G_inv @ U.T @ Wq @ V.T @ S_X_inv,
                      'sc': sidecar_full_bpw},
    }

    D_norot_cache = {}
    for k in K_VALUES:
        Wq_nr = quantize_tiles(W, k, TILE)
        D_norot_cache[k] = hessian_weighted_error(W - Wq_nr, H_G, H_X)

    for arm_name, arm in arms.items():
        arm_res = {}
        for k in K_VALUES:
            Wq = quantize_tiles(arm['src'], k, TILE)
            W_hat = arm['fn'](Wq)
            D = hessian_weighted_error(W - W_hat, H_G, H_X)
            ben = (D_norot_cache[k] - D) / D_norot_cache[k] * 100 if D_norot_cache[k] > 1e-15 else 0
            arm_res[f'K{k}'] = {'hwe': float(D), 'sidecar_bpw': arm['sc'],
                                'benefit_vs_norot_pct': ben}
        results['arms'][arm_name] = arm_res
        k4 = arm_res.get('K4', {})
        print(f"    {arm_name}: K4 HWE={k4.get('hwe', 0):.4e}, "
              f"benefit={k4.get('benefit_vs_norot_pct', 0):+.1f}%")

    for k in K_VALUES:
        kk = f'K{k}'
        fb = results['arms']['biip_full'][kk]['benefit_vs_norot_pct']
        hb = results['arms']['hadamard_only'][kk]['benefit_vs_norot_pct']
        cb = results['arms']['biip_8bit'][kk]['benefit_vs_norot_pct']
        r8 = cb / fb * 100 if fb > 0.01 else 0
        rh = hb / fb * 100 if fb > 0.01 else 0
        results[f'retained_K{k}_8bit_pct'] = r8
        results[f'retained_K{k}_hadamard_pct'] = rh
        print(f"    K={k}: full={fb:+.1f}%, 8bit retained={r8:.0f}%, "
              f"Hadamard retained={rh:.0f}%")

    return results


def _quantize_vector(v, bits):
    """Quantize a vector to `bits` bits per element (uniform)."""
    if bits <= 0:
        return np.ones_like(v)
    nl = 2 ** bits
    lo, hi = float(v.min()), float(v.max())
    step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
    if step == 0.0:
        return v.copy()
    return np.clip(np.round((v - lo) / step), 0, nl - 1) * step + lo


def D_unrot_k(W, H_X, H_G, k):
    """Compute total distortion at uniform K without rotation."""
    Wq = quantize_tiles(W, k, TILE)
    E = W - Wq
    return hessian_weighted_error(E, H_G, H_X)


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    t_start = time.time()
    print("=" * 80)
    print("R24 — Entropy-Constrained Allocation")
    print("=" * 80)

    tensors = load_real_weights()
    rng = np.random.default_rng(RNG_SEED)

    all_results = {
        'config': {
            'tile': TILE, 'slice': SLICE, 'K_values': K_VALUES,
            'tensor_names': TENSOR_NAMES, 'n_tiles': N_TILES,
            'elem_per_tile': ELEM_PER_T, 'b_sidecar_bpw': B_SIDECAR_BPW,
            'sidecar_levels': SIDECAR_LEVELS,
        },
        'tensors': {},
    }

    for tname in TENSOR_NAMES:
        if tname not in tensors:
            continue
        print(f"\n{'═' * 70}")
        print(f"  Tensor: {tname} (shape {tensors[tname].shape})")
        print(f"{'═' * 70}")

        tensor_results = {'slices': {}}
        slices = extract_slices(tensors[tname], SLICE, SLICE, RNG_SEED)
        X = gen_calibration(SLICE, N_CALIB, RNG_SEED)

        for sname, W in slices:
            print(f"\n  ── Slice: {sname} ({W.shape}) ──")
            H_G, H_X = compute_hessians(W, X)
            lbl = f"{tname}/{sname}"

            # Exp1: Entropy estimation
            exp1, D_table, H_table = experiment_entropy_estimation(W, H_G, H_X, lbl)

            # Exp2: Fixed-K DP vs entropy DP
            exp2 = experiment_alloc_comparison(W, H_G, H_X, D_table, H_table, lbl)

            # Exp3: Entropy + rotation
            exp3 = experiment_rotation_composition(W, H_G, H_X, rng, lbl)

            # Exp4: R-D frontier
            exp4 = experiment_rd_frontier(W, H_G, H_X, D_table, H_table, lbl)

            # Exp5: Empirical entropy lower bound
            exp5 = experiment_entropy_lower_bound(W, H_G, H_X, lbl)

            # Exp7: B-as-action (only for first slice to save time)
            if sname == 's0':
                exp7 = experiment_b_as_action(W, H_G, H_X, rng, lbl)
            else:
                exp7 = None

            # Exp8: Sidecar sweep (only for first slice)
            if sname == 's0':
                exp8 = experiment_sidecar_sweep(W, H_G, H_X, rng, lbl)
            else:
                exp8 = None

            tensor_results['slices'][sname] = {
                'entropy_estimation': exp1,
                'alloc_comparison': exp2,
                'rotation_composition': exp3,
                'rd_frontier': exp4,
                'entropy_lower_bound': exp5,
                'sidecar_sweep': exp8,
            }

        all_results['tensors'][tname] = tensor_results

    # Exp6: Inter-layer entropy allocation
    print(f"\n{'═' * 70}")
    print(f"  Experiment 6: Inter-layer entropy allocation")
    print(f"{'═' * 70}")
    exp6 = experiment_interlayer_entropy(tensors)
    all_results['interlayer'] = exp6

    # ─── Summary ───
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    _print_summary(all_results)

    elapsed = time.time() - t_start
    all_results['elapsed_seconds'] = elapsed
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return all_results


def _print_summary(all_results):
    """Print aggregate summary."""
    print("\n  ── Entropy savings (avg across tensors/slices) ──")
    for k in K_VALUES:
        savings = []
        for tname in TENSOR_NAMES:
            if tname not in all_results['tensors']:
                continue
            for sname, sr in all_results['tensors'][tname]['slices'].items():
                ek = f'K{k}'
                if ek in sr['entropy_estimation']['per_K']:
                    savings.append(sr['entropy_estimation']['per_K'][ek]['savings_pct'])
        if savings:
            print(f"    K={k}: {np.mean(savings):.1f}% ± {np.std(savings):.1f}%")

    print("\n  ── Entropy DP improvement over fixed-K DP (no model, same payload) ──")
    for avg_k in [3, 4, 5, 6]:
        gains = []
        for tname in TENSOR_NAMES:
            if tname not in all_results['tensors']:
                continue
            for sname, sr in all_results['tensors'][tname]['slices'].items():
                for b in sr['alloc_comparison']['budgets']:
                    if b['avg_k'] == avg_k:
                        gains.append(b['entropy_dp']['improvement_vs_fixed_pct'])
        if gains:
            print(f"    avg_K={avg_k}: {np.mean(gains):+.1f}% ± {np.std(gains):.1f}%")

    print("\n  ── Rotation effect on entropy (avg K=4) ──")
    ent_changes = []
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        for sname, sr in all_results['tensors'][tname]['slices'].items():
            ec = sr['rotation_composition'].get('entropy_change', {})
            if 'K4_pct' in ec:
                ent_changes.append(ec['K4_pct'])
    if ent_changes:
        print(f"    ΔH after rotation: {np.mean(ent_changes):+.1f}% ± {np.std(ent_changes):.1f}%")

    print("\n  ── Inter-layer entropy allocation ──")
    for b in all_results.get('interlayer', {}).get('budgets', []):
        print(f"    avg_K={b['avg_k']}: distortion gain={b['distortion_gain_pct']:+.1f}%")

    print("\n  ── B-as-action: per-tensor B allocation ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        for sname, sr in all_results['tensors'][tname]['slices'].items():
            if sr.get('b_as_action'):
                for b in sr['b_as_action']['budgets']:
                    uk = b['uniform_k']
                    ukb = b.get('uniform_k_b')
                    eb = b.get('entropy_dp_b')
                    line = f"    {tname}/{sname} target={b['target_bpw']}bpw: "
                    line += f"K{uk['k']}={uk['hwe']:.4e}"
                    if ukb:
                        line += f" → K{ukb['k']}+B={ukb['hwe']:.4e}"
                    if eb:
                        line += f" → ent+B={eb['hwe']:.4e}"
                    print(line)
                break  # only first slice
        break  # only first tensor

    print("\n  ── Sidecar compression sweep ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        for sname, sr in all_results['tensors'][tname]['slices'].items():
            if sr.get('sidecar_sweep'):
                ss = sr['sidecar_sweep']
                for arm_name in ['hadamard_only', 'biip_8bit', 'biip_full']:
                    arm = ss.get('arms', {}).get(arm_name, {})
                    k4 = arm.get('K4', {})
                    print(f"    {tname}/{sname} {arm_name}: K4 benefit={k4.get('benefit_vs_norot_pct', 0):+.1f}%")
                print(f"    K4 retained: 8bit={ss.get('retained_K4_8bit_pct', 0):.0f}%, "
                      f"Hadamard={ss.get('retained_K4_hadamard_pct', 0):.0f}%")
                break
        break

if __name__ == "__main__":
    run_experiment()
