#!/usr/bin/env python3
"""
R28: Multi-precision allocator with BiP as an action.

Each tensor chooses from {K4, K4+B, K5, K5+B, K6, K6+B}.
The allocator is a multiple-choice knapsack minimizing total HWE at matched total bytes.
BiP (B) is not just +0.5 bits — it changes the rate-distortion curve.

Key questions:
  1. Does including B in the alphabet improve the global R-D frontier?
  2. Which tensors benefit most from B?
  3. How does optimal allocation change with sidecar cost?
  4. Does K5+B beat K6 at 5.5 bpw? (Expert's killer test)
  5. Does K4+B ≈ K5? (Expert's K4+B question)

Evidence:
  - Per-tensor (bytes, HWE) menu for all 6 alphabet points
  - Multiple-choice knapsack DP
  - Budget sweep from K4-uniform to K6-uniform
  - Inter-layer + BiP allocation
  - Sidecar rate sweep (0.5, 0.25, 0.125, 0.0625 bpw)
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
P_CAL = 512
N_TILES = (M_DIM // TILE) * (N_DIM // TILE)  # 64
ELEMENTS_PER_TILE = TILE * TILE  # 256
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
WAVE5_WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_wave5_weights.npz"

# The 9-option alphabet: (K, transform_type) where transform_type in {'none', 'bip', 'had'}
# B = BiIP scaling + Hadamard (expensive sidecar, may hurt per-tile quant)
# H = Hadamard-only (cheap sidecar, R26/R27 show this is the real benefit)
ALPHABET = [
    (4, 'none'),  # K4
    (4, 'bip'),   # K4+B
    (4, 'had'),   # K4+H
    (5, 'none'),  # K5
    (5, 'bip'),   # K5+B
    (5, 'had'),   # K5+H
    (6, 'none'),  # K6
    (6, 'bip'),   # K6+B
    (6, 'had'),   # K6+H
]
ALPHABET_LABELS = ["K4", "K4+B", "K4+H", "K5", "K5+B", "K5+H", "K6", "K6+B", "K6+H"]
K_VALUES = [4, 5, 6]

# Sidecar costs: exact bytes depend on tensor dimensions (M+N scales, M*N payload)
# At 128×128: BiP = 0.5156 bpw, Had = 0.0156 bpw (ASPECT RATIO ARTIFACT)
# At production 17408×5120: BiP ≈ 0.00834 bpw, Had ≈ 0.000253 bpw (62× lower)
# The allocator should use production-scale sidecar rates for realistic allocation.

# Slice-scale sidecar (128×128) — overprices sidecar relative to production
SLICE_BIP_SIDECAR_BPW = 0.515625
SLICE_HAD_SIDECAR_BPW = 0.015625

# Production-scale sidecar (17408×5120) — realistic rates
# BiP: ((17408+5120)*4 + ceil((17408+5120)/8)) / (17408*5120/8) = 90147 / 11141120 ≈ 0.00809 bpw
# Had: ceil((17408+5120)/8) / (17408*5120/8) = 2817 / 11141120 ≈ 0.000253 bpw
# Production-scale sidecar rates (computed from 17408×5120 tensor dimensions)
# BiP: ((17408+5120)*4 + ceil((22528)/8)) / (17408*5120/8) = 92928 / 11141120 ≈ 0.008341 bpw
# Had: ceil((22528)/8) / (17408*5120/8) = 2816 / 11141120 ≈ 0.000253 bpw
PROD_BIP_SIDECAR_BPW = 0.0083409926
PROD_HAD_SIDECAR_BPW = 0.0002527574

# Default: use production-scale sidecar rates (not slice-scale artifact)
DEFAULT_BIP_SIDECAR_BPW = PROD_BIP_SIDECAR_BPW
DEFAULT_HAD_SIDECAR_BPW = PROD_HAD_SIDECAR_BPW

# Model architecture
N_LAYERS = 56
HIDDEN_DIM = 5120
INTER_DIM = 17408
# Wave5: 9 depths × 5-6 roles, 8 blocks per tensor for screening
WAVE5_LAYERS = [0, 7, 14, 21, 28, 35, 42, 49, 55]
WAVE5_ROLES = ['gate', 'up', 'down', 'qkv', 'out', 'z']
WAVE5_N_BLOCKS = 8  # blocks per tensor for screening
ROLES = ['gate', 'down']
ROLE_SEED_MAP = {'gate': 0, 'down': 1, 'qkv': 2, 'out': 3, 'up': 4, 'z': 5}

RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r28-multi-precision/receipts/research/r28-multi-precision-results.json"


# ============================================================================
# Weight loading
# ============================================================================

def load_real_weights():
    """Load real Qwen3.8-27B BF16 weights (correctly decoded)."""
    data = np.load(WEIGHTS_PATH, allow_pickle=True)
    tensors = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        tensors[key] = arr
    return tensors


def extract_slice(tensor, m=M_DIM, n=N_DIM, seed=42):
    """Extract a representative 128×128 slice from a large tensor."""
    rng = np.random.RandomState(seed)
    M, N = tensor.shape
    r0 = rng.randint(0, max(M - m + 1, 1))
    c0 = rng.randint(0, max(N - n + 1, 1))
    return tensor[r0:r0 + m, c0:c0 + n].copy()


def gen_synthetic_weights(layer_idx, role, shape, real_stats=None, seed=42):
    """Generate synthetic weights with layer-appropriate statistics."""
    rng = np.random.RandomState(seed)
    M, N = shape
    if real_stats is not None:
        scale = real_stats.get('std', 0.02)
        mean = real_stats.get('mean', 0.0)
    else:
        # Layer-dependent scale: later layers tend to have smaller weights
        scale = 0.02 * (1.0 - 0.3 * layer_idx / 55.0)
        mean = 0.0
    W = rng.normal(mean, scale, (M, N))
    # Add some structure: correlated blocks
    for _ in range(5):
        bi = rng.randint(0, M)
        bj = rng.randint(0, N)
        si = rng.randint(4, 16)
        sj = rng.randint(4, 16)
        W[bi:bi+si, bj:bj+sj] *= rng.uniform(1.5, 3.0)
    return W


def measure_weight_stats(tensor):
    """Measure basic statistics of a weight tensor."""
    return {
        'mean': float(np.mean(tensor)),
        'std': float(np.std(tensor)),
        'min': float(np.min(tensor)),
        'max': float(np.max(tensor)),
        'norm': float(np.linalg.norm(tensor)),
    }


# ============================================================================
# Calibration and Hessian computation
# ============================================================================

def gen_calibration(N, P, seed=42):
    """Generate synthetic calibration activations with realistic structure."""
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((P, N)) * 0.1
    # Add outlier channels (5% of channels have 10x scale)
    outlier_ch = rng.choice(N, max(N // 20, 1), replace=False)
    X[:, outlier_ch] *= 10.0
    # Add correlation structure
    cov = np.eye(N) + 0.1 * rng.standard_normal((N, N))
    cov = cov @ cov.T / N + np.eye(N)
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(N))
    X = X @ L.T
    return X


def compute_hessians(W, X):
    """Compute input and output Hessians.
    H_X = X^T X / P (input covariance)
    H_G = (W X)^T (W X) / P (output covariance proxy)
    """
    P = X.shape[0]
    H_X = X.T @ X / P
    Y = X @ W.T
    H_G = Y.T @ Y / P
    return H_X, H_G


# ============================================================================
# Quantizer (per-tile uniform, MATCHED for all arms)
# ============================================================================

def quantize_tile(w, k):
    """Per-tile uniform quantizer. k bits → 2^k levels."""
    if k <= 0:
        return np.zeros_like(w)
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo)
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_matrix_uniform(W, k, tile=TILE):
    """Quantize W uniformly at K=k using per-tile quantizer."""
    M, N = W.shape
    Wq = np.zeros_like(W)
    ntr, ntc = M // tile, N // tile
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            Wq[r0:r0+tile, c0:c0+tile] = quantize_tile(
                W[r0:r0+tile, c0:c0+tile], k)
    return Wq


# ============================================================================
# BiP: BiIP diagonal balancing + Hadamard rotation
# ============================================================================

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11).
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}
    S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}
    W' = S_G @ W @ S_X
    """
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = (np.diag(H_X) / col_norms_sq) ** 0.25
    sx_diag = np.clip(sx_diag, 0.1, 10.0)
    S_X = np.diag(sx_diag)

    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = (np.diag(H_G) / row_norms_sq) ** 0.25
    sg_diag = np.clip(sg_diag, 0.1, 10.0)
    S_G = np.diag(sg_diag)

    W_transformed = S_G @ W @ S_X
    return S_G, S_X, W_transformed


def hadamard_matrix(n):
    """Sylvester-type Hadamard matrix of size n (must be power of 2)."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    """Signed randomized Hadamard: H @ diag(±1)."""
    H = hadamard_matrix(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    return H @ np.diag(signs), signs


def apply_bip_transform(W, H_X, H_G, seed=42):
    """Apply BiP (BiIP scaling + two-sided Hadamard rotation).
    Returns transformed W, Hessians, and inverse transform params.
    """
    rng = np.random.RandomState(seed)
    d_out, d_in = W.shape

    # Step 1: BiIP diagonal balancing
    S_G, S_X, W_bal = biip_scaling(W, H_X, H_G)
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_bal = S_X_inv @ H_X @ S_X_inv
    H_G_bal = S_G_inv @ H_G @ S_G_inv

    # Step 2: Two-sided Hadamard rotation
    U, _ = signed_random_hadamard(d_out, rng)
    V, _ = signed_random_hadamard(d_in, rng)
    W_rot = U @ W_bal @ V.T
    H_X_rot = V @ H_X_bal @ V.T
    H_G_rot = U @ H_G_bal @ U.T

    return W_rot, H_X_rot, H_G_rot, U, V, S_G, S_X



def apply_hadamard_only_transform(W, H_X, H_G, seed=42):
    """Apply Hadamard-only rotation (no BiIP diagonal scaling).
    R26/R27 finding: BiIP scaling hurts per-tile quant, Hadamard alone is the benefit.
    Returns transformed W, Hessians, and inverse transform params (U, V only).
    """
    rng = np.random.RandomState(seed)
    d_out, d_in = W.shape
    U, _ = signed_random_hadamard(d_out, rng)
    V, _ = signed_random_hadamard(d_in, rng)
    W_rot = U @ W @ V.T
    H_X_rot = V @ H_X @ V.T
    H_G_rot = U @ H_G @ U.T
    return W_rot, H_X_rot, H_G_rot, U, V


def inverse_hadamard_only_transform(W_quantized, U, V):
    """Inverse Hadamard-only transform.
    W_hat = U^T @ W_quantized @ V
    """
    return U.T @ W_quantized @ V

def inverse_bip_transform(W_quantized, U, V, S_G, S_X):
    """Inverse BiP transform to get back to original space.
    W_hat = S_G^{-1} @ U^T @ W_quantized @ V @ S_X^{-1}
    """
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ W_quantized @ V @ S_X_inv


# ============================================================================
# Byte budget accounting
# ============================================================================

def tensor_bytes_K(K, M, N, tile=TILE):
    """Bytes for uniform K quantization (no BiP).
    payload: M*N*K/8
    sidecar: n_tiles * 4 (2 float16 per tile: min, max)
    metadata: 1 byte (K value)
    """
    n_tiles = (M // tile) * (N // tile)
    payload = M * N * K / 8
    sidecar = n_tiles * 4
    metadata = 1
    return payload + sidecar + metadata


def bip_sidecar_bytes_exact(M, N):
    """Exact BiP sidecar cost: float32 scales + Hadamard sign bits.
    Scales: (M + N) * 4 bytes (float32 diagonal of S_G and S_X)
    Signs: (M + N + 7) // 8 bytes (1 bit per sign, packed)
    For 128x128: (128+128)*4 + ceil((128+128)/8) = 1024 + 32 = 1056 bytes = 0.5156 bpw
    """
    return (M + N) * 4 + (M + N + 7) // 8


def hadamard_sidecar_bytes_exact(M, N):
    """Exact Hadamard-only sidecar: just sign bits, no diagonal scales.
    Signs: (M + N + 7) // 8 bytes (1 bit per sign, packed)
    For 128x128: ceil((128+128)/8) = 32 bytes = 0.015625 bpw
    """
    return (M + N + 7) // 8


def bip_sidecar_bytes(M, N, bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW):
    """BiP sidecar cost in bytes for a tensor of shape (M, N).
    Uses the provided bpw rate (default: production-scale ~0.008 bpw).
    """
    elements = M * N
    return elements * bip_sidecar_bpw / 8


def hadamard_sidecar_bytes(M, N, had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Hadamard-only sidecar cost in bytes for a tensor of shape (M, N).
    Uses the provided bpw rate (default: production-scale ~0.00025 bpw).
    """
    elements = M * N
    return elements * had_sidecar_bpw / 8


# Exact default sidecar bytes for 128x128
EXACT_BIP_SIDECAR_BYTES = bip_sidecar_bytes_exact(M_DIM, N_DIM)  # 1056
EXACT_BIP_SIDECAR_BPW = EXACT_BIP_SIDECAR_BYTES * 8 / (M_DIM * N_DIM)  # ~0.5156
EXACT_HAD_SIDECAR_BYTES = hadamard_sidecar_bytes_exact(M_DIM, N_DIM)  # 32
EXACT_HAD_SIDECAR_BPW = EXACT_HAD_SIDECAR_BYTES * 8 / (M_DIM * N_DIM)  # ~0.0156



def tensor_bytes_KB(K, M, N, tile=TILE, bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW):
    """Bytes for K+BiP quantization."""
    base = tensor_bytes_K(K, M, N, tile)
    bip_sc = bip_sidecar_bytes(M, N, bip_sidecar_bpw)
    return base + bip_sc


def tensor_bytes_KH(K, M, N, tile=TILE, had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Bytes for K+Hadamard quantization."""
    base = tensor_bytes_K(K, M, N, tile)
    had_sc = hadamard_sidecar_bytes(M, N, had_sidecar_bpw)
    return base + had_sc


def option_bytes(K, transform_type, M, N, tile=TILE,
                 bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW,
                 had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Bytes for any alphabet option. transform_type in {'none', 'bip', 'had'}."""
    if transform_type == 'bip':
        return tensor_bytes_KB(K, M, N, tile, bip_sidecar_bpw)
    elif transform_type == 'had':
        return tensor_bytes_KH(K, M, N, tile, had_sidecar_bpw)
    else:
        return tensor_bytes_K(K, M, N, tile)


def option_bpw(K, transform_type, M, N, tile=TILE,
               bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW,
               had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Effective bpw for any alphabet option."""
    return option_bytes(K, transform_type, M, N, tile,
                        bip_sidecar_bpw, had_sidecar_bpw) * 8 / (M * N)


# ============================================================================
# Distortion measurement
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """Primary metric: tr(H_G · E · H_X · E^T)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def measure_option(W, H_X, H_G, K, transform_type, seed=42):
    """Measure (bytes, HWE, MSE) for one alphabet option.
    Error is measured in ORIGINAL basis after inverse transform.
    transform_type in {'none', 'bip', 'had'}
    """
    M, N = W.shape
    if transform_type == 'bip':
        W_t, H_X_t, H_G_t, U, V, S_G, S_X = apply_bip_transform(
            W, H_X, H_G, seed=seed)
        Wq_t = quantize_matrix_uniform(W_t, K)
        Wq = inverse_bip_transform(Wq_t, U, V, S_G, S_X)
    elif transform_type == 'had':
        W_t, H_X_t, H_G_t, U, V = apply_hadamard_only_transform(
            W, H_X, H_G, seed=seed)
        Wq_t = quantize_matrix_uniform(W_t, K)
        Wq = inverse_hadamard_only_transform(Wq_t, U, V)
    else:
        Wq = quantize_matrix_uniform(W, K)

    E = W - Wq
    hwe = max(hessian_weighted_error(E, H_G, H_X), 0.0)
    mse = float(np.mean(E ** 2))
    b = option_bytes(K, transform_type, M, N)
    return {'hwe': hwe, 'mse': mse, 'bytes': b}


def measure_full_menu(W, H_X, H_G, seed=42,
                      bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW,
                      had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Measure all 9 alphabet options for a tensor.
    Returns dict: {option_label: {hwe, mse, bytes, bpw, K, transform_type}}
    """
    M, N = W.shape
    menu = {}
    for (K, transform_type), label in zip(ALPHABET, ALPHABET_LABELS):
        result = measure_option(W, H_X, H_G, K, transform_type, seed=seed)
        result['bytes'] = option_bytes(K, transform_type, M, N,
                                        bip_sidecar_bpw=bip_sidecar_bpw,
                                        had_sidecar_bpw=had_sidecar_bpw)
        result['bpw'] = result['bytes'] * 8 / (M * N)
        result['K'] = K
        result['transform_type'] = transform_type
        menu[label] = result
    return menu


# ============================================================================
# Multiple-choice knapsack DP
# ============================================================================



def knapsack_dp_ksum(items, budget_bytes,
                     bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW,
                     had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Optimized DP for same-size items using scaled byte costs as budget unit.
    Works with any subset of the alphabet (K-only, K+BiP, K+Had, or full 9-option).
    """
    n_items = len(items)
    M = items[0]['M']
    N = items[0]['N']

    # Determine which labels are available across items
    available_labels = sorted(items[0]['menu'].keys())

    # Compute byte cost per option
    option_costs = {}
    for label in available_labels:
        K, transform_type = ALPHABET[ALPHABET_LABELS.index(label)]
        option_costs[label] = option_bytes(K, transform_type, M, N,
                                            bip_sidecar_bpw=bip_sidecar_bpw,
                                            had_sidecar_bpw=had_sidecar_bpw)

    # Use scaled integer costs. For budgets < 500K bytes, use 1-byte resolution
    # (exact). For larger, coarsen to keep DP array manageable.
    if budget_bytes < 500000:
        SCALE = 1.0
    else:
        costs = sorted(set(option_costs.values()))
        if len(costs) > 1:
            min_diff = min(np.diff(costs))
        else:
            min_diff = 1.0
        SCALE = max(1.0, min_diff)
    int_costs = {label: int(round(c / SCALE)) for label, c in option_costs.items()}
    budget_int = int(budget_bytes / SCALE)

    INF = float('inf')
    dp = np.full(budget_int + 1, INF)
    dp[0] = 0.0
    choices = []

    for i in range(n_items):
        new_dp = np.full(budget_int + 1, INF)
        new_choice = np.full(budget_int + 1, -1, dtype=int)
        menu = items[i]['menu']
        item_labels = sorted(menu.keys())
        dists = [menu[label]['hwe'] for label in item_labels]
        costs_i = [int_costs[label] for label in item_labels]

        for j in range(budget_int + 1):
            if dp[j] == INF:
                continue
            for ki in range(len(item_labels)):
                nj = j + costs_i[ki]
                if nj > budget_int:
                    continue
                val = dp[j] + dists[ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append((new_choice.copy(), item_labels))

    best_j = 0
    best_d = INF
    for j in range(budget_int + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    assignment = {}
    j = best_j
    for i in range(n_items - 1, -1, -1):
        new_choice, item_labels = choices[i]
        ki = new_choice[j]
        if ki < 0:
            ki = 0
        label = item_labels[ki]
        assignment[items[i]['id']] = label
        j -= int_costs[label]

    total_bytes = sum(option_costs[assignment[items[i]['id']]] for i in range(n_items))
    return assignment, best_d, total_bytes


# ============================================================================
# Budget computation helpers
# ============================================================================

def uniform_budget(items, K, transform_type='none',
                   bip_sidecar_bpw=DEFAULT_BIP_SIDECAR_BPW,
                   had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW):
    """Compute total bytes for uniform K with given transform across all items."""
    total = 0
    for item in items:
        total += option_bytes(K, transform_type, item['M'], item['N'],
                              bip_sidecar_bpw=bip_sidecar_bpw,
                              had_sidecar_bpw=had_sidecar_bpw)
    return total


def avg_bpw_budget(items, target_bpw):
    """Compute total bytes for a target average bpw."""
    total_elements = sum(item['M'] * item['N'] for item in items)
    return target_bpw * total_elements / 8


# ============================================================================
# Experiment 1: Per-tensor menu (all 9 alphabet options)
# ============================================================================

def experiment_per_tensor_menu(real_weights):
    """Measure all 9 alphabet options per tensor."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Per-tensor menu — all 9 alphabet options")
    print("=" * 80)

    tensors = {}
    menus = {}

    # Real weights: L0 and L55 gate + down
    for layer in [0, 55]:
        for role in ROLES:
            key = f"L{layer}_{role}"
            if key in real_weights:
                W = extract_slice(real_weights[key], seed=layer * 100 + ROLE_SEED_MAP[role])
            else:
                alt_keys = [f"L{layer}_{role}_proj", f"layer{layer}_{role}"]
                found = False
                for ak in alt_keys:
                    if ak in real_weights:
                        W = extract_slice(real_weights[ak], seed=layer * 100 + ROLE_SEED_MAP[role])
                        found = True
                        break
                if not found:
                    print(f"  WARNING: {key} not found in weights, using synthetic")
                    W = gen_synthetic_weights(layer, role, (M_DIM, N_DIM), seed=layer * 100 + ROLE_SEED_MAP[role])
            tensors[key] = W

    # Synthetic weights for intermediate layers
    for layer in [10, 20, 30, 40]:
        for role in ROLES:
            key = f"L{layer}_{role}"
            W = gen_synthetic_weights(layer, role, (M_DIM, N_DIM), seed=layer * 100 + ROLE_SEED_MAP[role])
            tensors[key] = W

    # Measure menus
    for key in sorted(tensors.keys()):
        W = tensors[key]
        X = gen_calibration(W.shape[1], P_CAL, seed=42)
        H_X, H_G = compute_hessians(W, X)
        menu = measure_full_menu(W, H_X, H_G, seed=42)
        menus[key] = menu

        print(f"\n  {key} (shape {W.shape}):")
        print(f"    {'Option':<8} {'bpw':>6} {'bytes':>8} {'HWE':>14} {'MSE':>14}")
        for label in ALPHABET_LABELS:
            m = menu[label]
            print(f"    {label:<8} {m['bpw']:6.3f} {m['bytes']:8.0f} {m['hwe']:14.6e} {m['mse']:14.6e}")

        # Transform benefit analysis: compare BiP vs Hadamard vs none
        for K in K_VALUES:
            k_label = f"K{K}"
            kb_label = f"K{K}+B"
            kh_label = f"K{K}+H"
            hwe_k = menu[k_label]['hwe']
            for t_label, t_name in [(kb_label, 'BiP'), (kh_label, 'Had')]:
                hwe_t = menu[t_label]['hwe']
                if hwe_k > 1e-15:
                    reduction = (1.0 - hwe_t / hwe_k) * 100
                else:
                    reduction = 0.0
                print(f"    {t_name} benefit at K{K}: {reduction:+.1f}% HWE reduction")

    return tensors, menus


# ============================================================================
# Experiment 2: Multiple-choice knapsack — three budget scenarios
# ============================================================================

def experiment_knapsack_scenarios(menus, tensors):
    """Run knapsack DP at three budget levels."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Multiple-choice knapsack — three budget scenarios")
    print("=" * 80)

    # Build items from menus
    items = []
    for key in sorted(menus.keys()):
        M, N = tensors[key].shape
        items.append({
            'id': key,
            'menu': menus[key],
            'M': M,
            'N': N,
        })

    n_items = len(items)
    results = {}

    # Scenario 1: K5-uniform budget (low)
    budget_k5 = uniform_budget(items, 5, 'none')
    print(f"\n  Scenario 1: K5-uniform budget = {budget_k5:.0f} bytes ({budget_k5*8/(n_items*M_DIM*N_DIM):.3f} bpw avg)")

    # K-only options (no B)
    items_konly = []
    for item in items:
        menu_konly = {k: v for k, v in item['menu'].items() if k in ('K4', 'K5', 'K6')}
        items_konly.append({**item, 'menu': menu_konly})

    assign_k5_konly, d_k5_konly, b_k5_konly = knapsack_dp_ksum(items_konly, budget_k5)
    assign_k5_full, d_k5_full, b_k5_full = knapsack_dp_ksum(items, budget_k5)

    print(f"    K-only DP:  D={d_k5_konly:.6e}  bytes={b_k5_konly:.0f}  alloc={assign_k5_konly}")
    print(f"    Full DP:    D={d_k5_full:.6e}  bytes={b_k5_full:.0f}  alloc={assign_k5_full}")
    improvement = (1.0 - d_k5_full / d_k5_konly) * 100 if d_k5_konly > 0 else 0
    print(f"    Transform improvement: {improvement:+.1f}%")

    results['k5_uniform'] = {
        'budget_bytes': budget_k5,
        'k_only': {'distortion': d_k5_konly, 'bytes': b_k5_konly, 'allocation': assign_k5_konly},
        'full': {'distortion': d_k5_full, 'bytes': b_k5_full, 'allocation': assign_k5_full},
        'improvement_pct': improvement,
    }

    # Scenario 2: 5.5 bpw average (expert's killer test)
    budget_55 = avg_bpw_budget(items, 5.5)
    print(f"\n  Scenario 2: 5.5 bpw average budget = {budget_55:.0f} bytes")

    assign_55_konly, d_55_konly, b_55_konly = knapsack_dp_ksum(items_konly, budget_55)
    assign_55_full, d_55_full, b_55_full = knapsack_dp_ksum(items, budget_55)

    # Also compute K5.5 mixed-K (without B) for comparison
    # This is the K-only DP at the same budget
    print(f"    K-only DP:  D={d_55_konly:.6e}  bytes={b_55_konly:.0f}  alloc={assign_55_konly}")
    print(f"    Full DP:    D={d_55_full:.6e}  bytes={b_55_full:.0f}  alloc={assign_55_full}")
    improvement_55 = (1.0 - d_55_full / d_55_konly) * 100 if d_55_konly > 0 else 0
    print(f"    Transform improvement: {improvement_55:+.1f}%")

    # Check: does mixed 9-option beat K6 at 5.5 bpw?
    k6_uniform_d = sum(menus[key]['K6']['hwe'] for key in sorted(menus.keys()))
    k6_uniform_b = uniform_budget(items, 6, 'none')
    print(f"    K6-uniform: D={k6_uniform_d:.6e}  bytes={k6_uniform_b:.0f}")
    if d_55_full < k6_uniform_d:
        print(f"    >>> Mixed 9-option BEATS K6-uniform at lower bytes! <<<")
    else:
        ratio = d_55_full / k6_uniform_d
        print(f"    Mixed 9-option / K6-uniform = {ratio:.3f}x")

    results['5.5bpw'] = {
        'budget_bytes': budget_55,
        'k_only': {'distortion': d_55_konly, 'bytes': b_55_konly, 'allocation': assign_55_konly},
        'full': {'distortion': d_55_full, 'bytes': b_55_full, 'allocation': assign_55_full},
        'k6_uniform': {'distortion': k6_uniform_d, 'bytes': k6_uniform_b},
        'improvement_pct': improvement_55,
        'mixed_beats_k6': d_55_full < k6_uniform_d,
    }

    # Scenario 3: K6-uniform budget (high) — does K5+B everywhere beat K6?
    budget_k6 = uniform_budget(items, 6, 'none')
    print(f"\n  Scenario 3: K6-uniform budget = {budget_k6:.0f} bytes ({budget_k6*8/(n_items*M_DIM*N_DIM):.3f} bpw avg)")

    assign_k6_konly, d_k6_konly, b_k6_konly = knapsack_dp_ksum(items_konly, budget_k6)
    assign_k6_full, d_k6_full, b_k6_full = knapsack_dp_ksum(items, budget_k6)

    print(f"    K-only DP:  D={d_k6_konly:.6e}  bytes={b_k6_konly:.0f}  alloc={assign_k6_konly}")
    print(f"    Full DP:    D={d_k6_full:.6e}  bytes={b_k6_full:.0f}  alloc={assign_k6_full}")
    improvement_k6 = (1.0 - d_k6_full / d_k6_konly) * 100 if d_k6_konly > 0 else 0
    print(f"    Transform improvement: {improvement_k6:+.1f}%")

    # K5+B everywhere vs K6 budget
    k5b_uniform_d = sum(menus[key]['K5+B']['hwe'] for key in sorted(menus.keys()))
    k5b_uniform_b = uniform_budget(items, 5, 'bip')
    # K5+H everywhere vs K6 budget (cheaper sidecar)
    k5h_uniform_d = sum(menus[key]['K5+H']['hwe'] for key in sorted(menus.keys()))
    k5h_uniform_b = uniform_budget(items, 5, 'had')
    print(f"    K5+B-uniform: D={k5b_uniform_d:.6e}  bytes={k5b_uniform_b:.0f}")
    print(f"    K5+H-uniform: D={k5h_uniform_d:.6e}  bytes={k5h_uniform_b:.0f}")
    if k5h_uniform_b <= budget_k6 and k5h_uniform_d < d_k6_konly:
        print(f"    >>> K5+H everywhere BEATS K6-uniform at FEWER bytes! <<<")
    else:
        print(f"    K5+H uniform / K6-uniform D-ratio: {k5h_uniform_d / d_k6_konly:.3f}x")

    results['k6_uniform'] = {
        'budget_bytes': budget_k6,
        'k_only': {'distortion': d_k6_konly, 'bytes': b_k6_konly, 'allocation': assign_k6_konly},
        'full': {'distortion': d_k6_full, 'bytes': b_k6_full, 'allocation': assign_k6_full},
        'k5b_uniform': {'distortion': k5b_uniform_d, 'bytes': k5b_uniform_b},
        'k5h_uniform': {'distortion': k5h_uniform_d, 'bytes': k5h_uniform_b},
        'improvement_pct': improvement_k6,
    }

    return results


# ============================================================================
# Experiment 3: Budget frontier sweep
# ============================================================================

def experiment_budget_frontier(menus, tensors):
    """Trace the full R-D frontier from K4-uniform to K6-uniform."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Budget frontier sweep (K4-uniform → K6-uniform)")
    print("=" * 80)

    items = []
    for key in sorted(menus.keys()):
        M, N = tensors[key].shape
        items.append({'id': key, 'menu': menus[key], 'M': M, 'N': N})

    items_konly = []
    for item in items:
        menu_konly = {k: v for k, v in item['menu'].items() if k in ('K4', 'K5', 'K6')}
        items_konly.append({**item, 'menu': menu_konly})

    n_items = len(items)
    total_elements = n_items * M_DIM * N_DIM

    # Budget range from K4-uniform to K6+H-uniform (Hadamard is cheapest transform)
    budget_min = uniform_budget(items, 4, 'none')
    budget_max = uniform_budget(items, 6, 'had')

    n_points = 25
    budgets = np.linspace(budget_min, budget_max, n_points)

    frontier_full = []
    frontier_konly = []
    uniform_points = {}

    # Uniform reference points
    for K in [4, 5, 6]:
        for t_type, t_suffix in [('none', ''), ('bip', '+B'), ('had', '+H')]:
            b = uniform_budget(items, K, t_type)
            d = sum(menus[key][f"K{K}{t_suffix}"]['hwe'] for key in sorted(menus.keys()))
            label = f"K{K}{t_suffix}"
            uniform_points[label] = {'bytes': b, 'hwe': d, 'bpw': b * 8 / total_elements}

    print(f"\n  Uniform reference points:")
    print(f"    {'Option':<8} {'bpw':>6} {'bytes':>8} {'HWE':>14}")
    for label in ALPHABET_LABELS:
        if label in uniform_points:
            u = uniform_points[label]
            print(f"    {label:<8} {u['bpw']:6.3f} {u['bytes']:8.0f} {u['hwe']:14.6e}")

    print(f"\n  Frontier sweep ({n_points} budget points):")
    print(f"    {'Budget':>8} {'bpw':>6} {'K-only D':>14} {'Full D':>14} {'Improv':>8} {'B-tensors':>10}")

    for budget in budgets:
        a_k, d_k, b_k = knapsack_dp_ksum(items_konly, budget)
        a_f, d_f, b_f = knapsack_dp_ksum(items, budget)
        bpw = b_f * 8 / total_elements
        improv = (1.0 - d_f / d_k) * 100 if d_k > 0 else 0
        n_b = sum(1 for v in a_f.values() if '+B' in v)
        frontier_full.append({'bytes': b_f, 'hwe': d_f, 'bpw': bpw, 'allocation': a_f, 'n_B': n_b})
        frontier_konly.append({'bytes': b_k, 'hwe': d_k, 'bpw': b_k * 8 / total_elements, 'allocation': a_k})
        print(f"    {budget:8.0f} {bpw:6.3f} {d_k:14.6e} {d_f:14.6e} {improv:+7.1f}% {n_b:10d}")

    return frontier_full, frontier_konly, uniform_points


# ============================================================================
# Experiment 4: Inter-layer + BiP allocation
# ============================================================================

def experiment_interlayer_bip(menus, tensors):
    """Combine inter-layer DP with BiP as an action.
    Each (layer, role) can choose from the full 6-option menu.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Inter-layer + BiP allocation")
    print("=" * 80)

    items = []
    for key in sorted(menus.keys()):
        M, N = tensors[key].shape
        items.append({'id': key, 'menu': menus[key], 'M': M, 'N': N})

    n_items = len(items)

    # Compare allocations at 5.5 bpw
    budget_55 = avg_bpw_budget(items, 5.5)

    # K-only allocation
    items_konly = []
    for item in items:
        menu_konly = {k: v for k, v in item['menu'].items() if k in ('K4', 'K5', 'K6')}
        items_konly.append({**item, 'menu': menu_konly})
    a_k, d_k, b_k = knapsack_dp_ksum(items_konly, budget_55)

    # Full 6-option allocation
    a_f, d_f, b_f = knapsack_dp_ksum(items, budget_55)

    print(f"\n  Budget: 5.5 bpw average = {budget_55:.0f} bytes")
    print(f"\n  K-only allocation (no BiP):")
    print(f"    {'Tensor':<16} {'Option':>8} {'HWE':>14}")
    for key in sorted(a_k.keys()):
        print(f"    {key:<16} {a_k[key]:>8} {menus[key][a_k[key]]['hwe']:14.6e}")
    print(f"    Total HWE: {d_k:.6e}  bytes: {b_k:.0f}")

    print(f"\n  Full 9-option allocation (with BiP+Had):")
    print(f"    {'Tensor':<16} {'Option':>8} {'HWE':>14} {'transform':>10}")
    for key in sorted(a_f.keys()):
        opt = a_f[key]
        t_type = menus[key][opt]['transform_type']
        print(f"    {key:<16} {opt:>8} {menus[key][opt]['hwe']:14.6e} {t_type:>10}")
    print(f"    Total HWE: {d_f:.6e}  bytes: {b_f:.0f}")

    improvement = (1.0 - d_f / d_k) * 100 if d_k > 0 else 0
    print(f"\n  Transform improvement at 5.5 bpw: {improvement:+.1f}%")

    # Which tensors get which transform?
    b_tensors = [k for k, v in a_f.items() if '+B' in v]
    h_tensors = [k for k, v in a_f.items() if '+H' in v]
    no_t_tensors = [k for k, v in a_f.items() if '+B' not in v and '+H' not in v]
    print(f"\n  Tensors getting BiP: {b_tensors}")
    print(f"  Tensors getting Had: {h_tensors}")
    print(f"  Tensors with no transform: {no_t_tensors}")

    return {
        'k_only': {'allocation': a_k, 'distortion': d_k, 'bytes': b_k},
        'full': {'allocation': a_f, 'distortion': d_f, 'bytes': b_f},
        'b_tensors': b_tensors,
        'h_tensors': h_tensors,
        'no_transform_tensors': no_t_tensors,
        'improvement_pct': improvement,
    }


# ============================================================================
# Experiment 5: Sensitivity to BiP vs Hadamard — which tensors benefit most?
# ============================================================================

def experiment_bip_sensitivity(menus):
    """Analyze which tensors benefit most from BiP vs Hadamard."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Transform sensitivity — BiP vs Hadamard")
    print("=" * 80)

    sensitivity = {}
    for key in sorted(menus.keys()):
        menu = menus[key]
        benefits = {}
        for K in K_VALUES:
            k_label = f"K{K}"
            kb_label = f"K{K}+B"
            kh_label = f"K{K}+H"
            hwe_k = menu[k_label]['hwe']
            hwe_kb = menu[kb_label]['hwe']
            hwe_kh = menu[kh_label]['hwe']
            red_bip = (1.0 - hwe_kb / hwe_k) * 100 if hwe_k > 1e-15 else 0.0
            red_had = (1.0 - hwe_kh / hwe_k) * 100 if hwe_k > 1e-15 else 0.0
            benefits[K] = {
                'hwe_k': hwe_k,
                'hwe_kb': hwe_kb,
                'hwe_kh': hwe_kh,
                'reduction_bip_pct': red_bip,
                'reduction_had_pct': red_had,
            }
        sensitivity[key] = benefits

    print(f"\n  {'Tensor':<16} {'K4 BiP':>7} {'K4 Had':>7} {'K5 BiP':>7} {'K5 Had':>7} {'K6 BiP':>7} {'K6 Had':>7}")
    for key in sorted(sensitivity.keys()):
        b = sensitivity[key]
        print(f"    {key:<16} {b[4]['reduction_bip_pct']:6.1f}% {b[4]['reduction_had_pct']:6.1f}% "
              f"{b[5]['reduction_bip_pct']:6.1f}% {b[5]['reduction_had_pct']:6.1f}% "
              f"{b[6]['reduction_bip_pct']:6.1f}% {b[6]['reduction_had_pct']:6.1f}%")

    # Rank by average Hadamard benefit
    ranked = sorted(sensitivity.keys(),
                    key=lambda k: np.mean([sensitivity[k][K]['reduction_had_pct'] for K in K_VALUES]),
                    reverse=True)
    print(f"\n  Ranking by average Hadamard benefit:")
    for i, key in enumerate(ranked):
        avg_had = np.mean([sensitivity[key][K]['reduction_had_pct'] for K in K_VALUES])
        avg_bip = np.mean([sensitivity[key][K]['reduction_bip_pct'] for K in K_VALUES])
        print(f"    {i+1}. {key}: Had={avg_had:.1f}% BiP={avg_bip:.1f}% avg reduction")

    # BiP vs Had head-to-head
    bip_wins = 0
    had_wins = 0
    for key in sensitivity:
        for K in K_VALUES:
            if sensitivity[key][K]['reduction_bip_pct'] > sensitivity[key][K]['reduction_had_pct']:
                bip_wins += 1
            else:
                had_wins += 1
    print(f"\n  BiP vs Had head-to-head: BiP wins {bip_wins}, Had wins {had_wins} (out of {len(sensitivity)*len(K_VALUES)})")

    return sensitivity, ranked


# ============================================================================
# Experiment 6: Sidecar rate sweep
# ============================================================================

def experiment_sidecar_sweep(tensors):
    """ORACLE COST-SENSITIVITY: How does optimal allocation change with BiP sidecar cost?

    NOTE: This is an oracle cost-sensitivity curve, NOT a realizable sidecar codec test.
    The BiP transform parameters (float64 scales, signs) are held fixed; only the
    numeric price charged to the DP changes. Lower price → larger feasible set →
    monotonically better DP optimum. To test realizable compressed sidecars
    (float16, int8, etc.), one would need to implement an encoder/decoder for
    each rate and remeasure HWE from the decoded parameters.

    Sweep: production rate (~0.008 bpw), slice rate (0.5156 bpw), and intermediate rates.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Sidecar rate sweep — ORACLE COST-SENSITIVITY (not realizable codec)")
    print("=" * 80)

    # Include both production-scale and slice-scale rates
    sidecar_rates = [PROD_BIP_SIDECAR_BPW, 0.0625, 0.125, 0.25, SLICE_BIP_SIDECAR_BPW]
    sidecar_labels = {
        PROD_BIP_SIDECAR_BPW: 'production (~0.008 bpw)',
        0.0625: '0.0625 bpw',
        0.125: '0.125 bpw',
        0.25: '0.25 bpw',
        SLICE_BIP_SIDECAR_BPW: 'slice 128×128 (0.5156 bpw)',
    }

    n_items = len(tensors)
    total_elements = n_items * M_DIM * N_DIM

    sweep_results = {}

    for sc_rate in sidecar_rates:
        label = sidecar_labels.get(sc_rate, f'{sc_rate:.4f} bpw')
        print(f"\n  Sidecar rate: {sc_rate:.6f} bpw ({label})")

        # Re-measure menus with this sidecar rate
        menus_sc = {}
        for key in sorted(tensors.keys()):
            W = tensors[key]
            X = gen_calibration(W.shape[1], P_CAL, seed=42)
            H_X, H_G = compute_hessians(W, X)
            menu = measure_full_menu(W, H_X, H_G, seed=42, bip_sidecar_bpw=sc_rate,
                                     had_sidecar_bpw=DEFAULT_HAD_SIDECAR_BPW)
            menus_sc[key] = menu

        # Build items
        items = []
        for key in sorted(menus_sc.keys()):
            M, N = tensors[key].shape
            items.append({'id': key, 'menu': menus_sc[key], 'M': M, 'N': N})

        # K-only items
        items_konly = []
        for item in items:
            menu_konly = {k: v for k, v in item['menu'].items() if k in ('K4', 'K5', 'K6')}
            items_konly.append({**item, 'menu': menu_konly})

        # Test at 5.5 bpw budget
        budget_55 = avg_bpw_budget(items, 5.5)

        a_k, d_k, b_k = knapsack_dp_ksum(items_konly, budget_55, bip_sidecar_bpw=sc_rate)
        a_f, d_f, b_f = knapsack_dp_ksum(items, budget_55, bip_sidecar_bpw=sc_rate)

        n_b = sum(1 for v in a_f.values() if '+B' in v)
        n_h = sum(1 for v in a_f.values() if '+H' in v)
        improv = (1.0 - d_f / d_k) * 100 if d_k > 0 else 0

        print(f"    K-only:  D={d_k:.6e}  bytes={b_k:.0f}")
        print(f"    Full:    D={d_f:.6e}  bytes={b_f:.0f}  BiP={n_b} Had={n_h} (of {n_items})")
        print(f"    Improvement: {improv:+.1f}%")
        print(f"    Allocation: {a_f}")

        sweep_results[sc_rate] = {
            'k_only': {'distortion': d_k, 'bytes': b_k, 'allocation': a_k},
            'full': {'distortion': d_f, 'bytes': b_f, 'allocation': a_f},
            'n_B': n_b,
            'n_H': n_h,
            'improvement_pct': improv,
        }

    # Summary
    print(f"\n  Sidecar sweep summary:")
    print(f"    {'Sidecar':>8} {'K-only D':>14} {'Full D':>14} {'Improv':>8} {'BiP':>5} {'Had':>5}")
    for sc in sidecar_rates:
        r = sweep_results[sc]
        print(f"    {sc:8.4f} {r['k_only']['distortion']:14.6e} {r['full']['distortion']:14.6e} {r['improvement_pct']:+7.1f}% {r['n_B']:5d} {r['n_H']:5d}")

    return sweep_results


# ============================================================================
# Experiment 7: K4+B vs K5 — does preconditioning help more at low K?
# ============================================================================

def experiment_k4b_vs_k5(menus):
    """Test if K4+B or K4+H ≈ K5 quality (expert's K4+B question, now with Hadamard)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: K4+B vs K4+H vs K5 — does preconditioning help more at low K?")
    print("=" * 80)

    print(f"\n  {'Tensor':<16} {'K4+B HWE':>14} {'K4+H HWE':>14} {'K5 HWE':>14} {'K4+B/K5':>8} {'K4+H/K5':>8}")
    for key in sorted(menus.keys()):
        m = menus[key]
        hwe_k4b = m['K4+B']['hwe']
        hwe_k4h = m['K4+H']['hwe']
        hwe_k5 = m['K5']['hwe']
        ratio_b = hwe_k4b / hwe_k5 if hwe_k5 > 0 else float('inf')
        ratio_h = hwe_k4h / hwe_k5 if hwe_k5 > 0 else float('inf')
        match = "≈K5" if ratio_h < 1.1 else ""
        print(f"    {key:<16} {hwe_k4b:14.6e} {hwe_k4h:14.6e} {hwe_k5:14.6e} {ratio_b:8.3f} {ratio_h:8.3f}  {match}")

    # Bytes comparison
    print(f"\n  Byte comparison (per 128×128 tensor):")
    for label in ['K4', 'K4+B', 'K4+H', 'K5']:
        m = menus[sorted(menus.keys())[0]][label]
        print(f"    {label}: {m['bytes']:.0f} bytes ({m['bpw']:.3f} bpw)")

    # Totals
    k4b_total_d = sum(menus[k]['K4+B']['hwe'] for k in sorted(menus.keys()))
    k4h_total_d = sum(menus[k]['K4+H']['hwe'] for k in sorted(menus.keys()))
    k5_total_d = sum(menus[k]['K5']['hwe'] for k in sorted(menus.keys()))
    k4b_total_b = sum(menus[k]['K4+B']['bytes'] for k in sorted(menus.keys()))
    k4h_total_b = sum(menus[k]['K4+H']['bytes'] for k in sorted(menus.keys()))
    k5_total_b = sum(menus[k]['K5']['bytes'] for k in sorted(menus.keys()))
    print(f"\n  K4+B total: D={k4b_total_d:.6e}  bytes={k4b_total_b:.0f}")
    print(f"  K4+H total: D={k4h_total_d:.6e}  bytes={k4h_total_b:.0f}")
    print(f"  K5  total: D={k5_total_d:.6e}  bytes={k5_total_b:.0f}")
    print(f"  K4+B/K5 ratio: {k4b_total_d/k5_total_d:.3f}x  K4+H/K5 ratio: {k4h_total_d/k5_total_d:.3f}x")

    return {
        'k4b_total': {'distortion': k4b_total_d, 'bytes': k4b_total_b},
        'k4h_total': {'distortion': k4h_total_d, 'bytes': k4h_total_b},
        'k5_total': {'distortion': k5_total_d, 'bytes': k5_total_b},
    }


# ============================================================================
# Experiment 8: Real-only analysis (4 real tensors, no synthetic)
# ============================================================================

REAL_TENSOR_KEYS = ['L0_gate', 'L0_down', 'L55_gate', 'L55_down']

def experiment_real_only(menus, tensors):
    """Run allocation using only the 4 real tensors (L0, L55 gate+down).
    This avoids contamination from synthetic weight statistics.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: Real-only analysis (4 real tensors, no synthetic)")
    print("=" * 80)

    real_menus = {k: menus[k] for k in REAL_TENSOR_KEYS if k in menus}
    real_tensors = {k: tensors[k] for k in REAL_TENSOR_KEYS if k in tensors}

    if len(real_menus) < 4:
        print(f"  WARNING: Only {len(real_menus)} real tensors found, need 4")
        return {}

    items = []
    for key in sorted(real_menus.keys()):
        M, N = real_tensors[key].shape
        items.append({'id': key, 'menu': real_menus[key], 'M': M, 'N': N})

    items_konly = []
    for item in items:
        menu_konly = {k: v for k, v in item['menu'].items() if k in ('K4', 'K5', 'K6')}
        items_konly.append({**item, 'menu': menu_konly})

    n_items = len(items)
    results = {}

    # Three budget scenarios on real-only
    for label, budget_fn in [
        ('K5-uniform', lambda: uniform_budget(items, 5)),
        ('5.5 bpw', lambda: avg_bpw_budget(items, 5.5)),
        ('K6-uniform', lambda: uniform_budget(items, 6)),
    ]:
        budget = budget_fn()
        a_k, d_k, b_k = knapsack_dp_ksum(items_konly, budget)
        a_f, d_f, b_f = knapsack_dp_ksum(items, budget)
        improv = (1.0 - d_f / d_k) * 100 if d_k > 0 else 0
        n_b = sum(1 for v in a_f.values() if '+B' in v)

        print(f"\n  {label} budget = {budget:.0f} bytes")
        print(f"    K-only:  D={d_k:.6e}  bytes={b_k:.0f}  alloc={a_k}")
        print(f"    Full:    D={d_f:.6e}  bytes={b_f:.0f}  alloc={a_f}")
        print(f"    BiP improvement: {improv:+.1f}%  B-tensors={n_b}/{n_items}")

        results[label] = {
            'budget_bytes': budget,
            'k_only': {'distortion': d_k, 'bytes': b_k, 'allocation': a_k},
            'full': {'distortion': d_f, 'bytes': b_f, 'allocation': a_f},
            'improvement_pct': improv,
            'n_B': n_b,
        }

    # K4+B vs K5 on real-only
    print(f"\n  K4+B vs K5 (real-only):")
    print(f"    {'Tensor':<16} {'K4+B HWE':>14} {'K5 HWE':>14} {'K4+B/K5':>8}")
    for key in sorted(real_menus.keys()):
        m = real_menus[key]
        ratio = m['K4+B']['hwe'] / m['K5']['hwe'] if m['K5']['hwe'] > 0 else float('inf')
        print(f"    {key:<16} {m['K4+B']['hwe']:14.6e} {m['K5']['hwe']:14.6e} {ratio:8.3f}")

    return results


# ============================================================================
# Experiment 9: Wave5 broad screening (9 depths × 5-6 roles, 8 blocks/tensor)
# ============================================================================

def extract_multiple_blocks(tensor, n_blocks, m=M_DIM, n=N_DIM, seed=42):
    """Extract n_blocks 128×128 blocks: diagonal + random + off-diagonal."""
    rng = np.random.RandomState(seed)
    M, N = tensor.shape
    blocks = []
    # First 3: diagonal blocks (first, middle, last)
    for frac in [0.0, 0.5, 1.0]:
        r0 = int(frac * max(M - m, 0))
        c0 = int(frac * max(N - n, 0))
        r0 = min(r0, M - m)
        c0 = min(c0, N - n)
        blocks.append(tensor[r0:r0+m, c0:c0+n].copy())
    # Remaining: seeded random blocks (including off-diagonal)
    for i in range(n_blocks - 3):
        r0 = rng.randint(0, max(M - m + 1, 1))
        c0 = rng.randint(0, max(N - n + 1, 1))
        blocks.append(tensor[r0:r0+m, c0:c0+n].copy())
    return blocks


def experiment_wave5_screening():
    """Wave5 broad screening: 9 depths × 5-6 roles, 8 blocks per tensor.
    Uses production-scale sidecar rates. Measures per-role, per-depth statistics.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 9: Wave5 broad screening (9 depths × 5-6 roles, 8 blocks/tensor)")
    print("=" * 80)

    weights = np.load(WAVE5_WEIGHTS_PATH, allow_pickle=True)
    available_keys = set(weights.files)
    print(f"  Loaded {len(available_keys)} tensors from Wave5 census")

    # Collect all blocks with metadata
    all_blocks = []  # list of (key, block_idx, W, H_X, H_G, menu)
    per_role_stats = {}
    per_depth_stats = {}
    per_tensor_menus = {}

    n_tensors = 0
    n_blocks_total = 0

    for layer in WAVE5_LAYERS:
        for role in WAVE5_ROLES:
            key = f"L{layer}_{role}"
            if key not in available_keys:
                continue
            n_tensors += 1
            W_full = weights[key]
            blocks = extract_multiple_blocks(W_full, WAVE5_N_BLOCKS,
                                               seed=layer * 100 + ROLE_SEED_MAP.get(role, 99))
            block_menus = []
            for bi, W in enumerate(blocks):
                X = gen_calibration(W.shape[1], P_CAL, seed=42 + bi)
                H_X, H_G = compute_hessians(W, X)
                menu = measure_full_menu(W, H_X, H_G, seed=42 + bi)
                block_menus.append(menu)
                n_blocks_total += 1

                # Aggregate per-role stats
                role_key = role
                if role_key not in per_role_stats:
                    per_role_stats[role_key] = {'bip_red': [], 'had_red': []}
                for K in K_VALUES:
                    k_lbl = f"K{K}"
                    bip_red = (1.0 - menu[f"K{K}+B"]['hwe'] / menu[k_lbl]['hwe']) * 100 if menu[k_lbl]['hwe'] > 1e-15 else 0.0
                    had_red = (1.0 - menu[f"K{K}+H"]['hwe'] / menu[k_lbl]['hwe']) * 100 if menu[k_lbl]['hwe'] > 1e-15 else 0.0
                    per_role_stats[role_key]['bip_red'].append(bip_red)
                    per_role_stats[role_key]['had_red'].append(had_red)

                # Aggregate per-depth stats
                depth_key = layer
                if depth_key not in per_depth_stats:
                    per_depth_stats[depth_key] = {'bip_red': [], 'had_red': []}
                for K in K_VALUES:
                    k_lbl = f"K{K}"
                    bip_red = (1.0 - menu[f"K{K}+B"]['hwe'] / menu[k_lbl]['hwe']) * 100 if menu[k_lbl]['hwe'] > 1e-15 else 0.0
                    had_red = (1.0 - menu[f"K{K}+H"]['hwe'] / menu[k_lbl]['hwe']) * 100 if menu[k_lbl]['hwe'] > 1e-15 else 0.0
                    per_depth_stats[depth_key]['bip_red'].append(bip_red)
                    per_depth_stats[depth_key]['had_red'].append(had_red)

            # Average menu across blocks for this tensor
            avg_menu = {}
            for label in ALPHABET_LABELS:
                avg_menu[label] = {
                    'hwe': np.mean([bm[label]['hwe'] for bm in block_menus]),
                    'mse': np.mean([bm[label]['mse'] for bm in block_menus]),
                    'bytes': block_menus[0][label]['bytes'],
                    'bpw': block_menus[0][label]['bpw'],
                    'hwe_std': np.std([bm[label]['hwe'] for bm in block_menus]),
                }
            per_tensor_menus[key] = avg_menu

    print(f"  Screened {n_tensors} tensors × {WAVE5_N_BLOCKS} blocks = {n_blocks_total} blocks total")

    # Per-role summary
    print(f"\n  Per-role BiP/Had benefit (averaged over K4/K5/K6, all depths, all blocks):")
    print(f"    {'Role':<8} {'BiP red%':>10} {'Had red%':>10} {'n_blocks':>10}")
    for role in sorted(per_role_stats.keys()):
        s = per_role_stats[role]
        print(f"    {role:<8} {np.mean(s['bip_red']):10.1f}% {np.mean(s['had_red']):10.1f}% {len(s['bip_red']):10d}")

    # Per-depth summary
    print(f"\n  Per-depth BiP/Had benefit (averaged over K4/K5/K6, all roles, all blocks):")
    print(f"    {'Layer':<8} {'BiP red%':>10} {'Had red%':>10} {'n_blocks':>10}")
    for layer in sorted(per_depth_stats.keys()):
        s = per_depth_stats[layer]
        print(f"    L{layer:<7} {np.mean(s['bip_red']):10.1f}% {np.mean(s['had_red']):10.1f}% {len(s['bip_red']):10d}")

    # Global summary
    all_bip = [v for s in per_role_stats.values() for v in s['bip_red']]
    all_had = [v for s in per_role_stats.values() for v in s['had_red']]
    print(f"\n  Global: BiP {np.mean(all_bip):.1f}% ± {np.std(all_bip):.1f}% | "
          f"Had {np.mean(all_had):.1f}% ± {np.std(all_had):.1f}% | n={len(all_bip)}")

    # BiP vs Had head-to-head
    bip_wins = sum(1 for b, h in zip(all_bip, all_had) if b > h)
    had_wins = sum(1 for b, h in zip(all_bip, all_had) if h > b)
    print(f"  BiP vs Had head-to-head: BiP wins {bip_wins}, Had wins {had_wins} (of {len(all_bip)})")

    return per_tensor_menus, per_role_stats, per_depth_stats


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 80)
    print("R28: Multi-precision allocator with BiP as an action")
    print("=" * 80)

    # Load real weights
    print("\nLoading real weights...")
    real_weights = load_real_weights()
    print(f"  Available keys: {sorted(real_weights.keys())}")

    # Experiment 1: Per-tensor menu
    tensors, menus = experiment_per_tensor_menu(real_weights)

    # Experiment 2: Three budget scenarios
    scenario_results = experiment_knapsack_scenarios(menus, tensors)

    # Experiment 3: Budget frontier
    frontier_full, frontier_konly, uniform_points = experiment_budget_frontier(menus, tensors)

    # Experiment 4: Inter-layer + BiP
    interlayer_results = experiment_interlayer_bip(menus, tensors)

    # Experiment 5: BiP sensitivity
    sensitivity, ranked = experiment_bip_sensitivity(menus)

    # Experiment 6: Sidecar rate sweep
    sweep_results = experiment_sidecar_sweep(tensors)

    # Experiment 7: K4+B vs K5
    k4b_results = experiment_k4b_vs_k5(menus)

    # Experiment 8: Real-only analysis
    real_only_results = experiment_real_only(menus, tensors)

    # Experiment 9: Wave5 broad screening
    wave5_menus, wave5_role_stats, wave5_depth_stats = experiment_wave5_screening()

    # Save results
    results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'R28: Multi-precision allocator with BiP as an action',
            'alphabet': ALPHABET_LABELS,
            'n_tensors': len(tensors),
            'tensor_keys': sorted(tensors.keys()),
            'tile_size': TILE,
            'slice_size': [M_DIM, N_DIM],
        },
        'per_tensor_menus': {
            key: {label: {k: v for k, v in m.items()} for label, m in menu.items()}
            for key, menu in menus.items()
        },
        'scenario_results': scenario_results,
        'frontier': {
            'full': [{'bytes': f['bytes'], 'hwe': f['hwe'], 'bpw': f['bpw'],
                       'n_B': f['n_B'], 'allocation': f['allocation']}
                      for f in frontier_full],
            'k_only': [{'bytes': f['bytes'], 'hwe': f['hwe'], 'bpw': f['bpw'],
                         'allocation': f['allocation']}
                        for f in frontier_konly],
            'uniform_points': uniform_points,
        },
        'interlayer_bip': interlayer_results,
        'bip_sensitivity': {
            key: {str(K): v for K, v in benefits.items()}
            for key, benefits in sensitivity.items()
        },
        'bip_ranking': ranked,
        'sidecar_sweep': {
            str(sc): {k: v for k, v in r.items()}
            for sc, r in sweep_results.items()
        },
        'k4b_vs_k5': k4b_results,
        'real_only': real_only_results,
        'wave5_screening': {
            'per_tensor_menus': {k: {l: v for l, v in m.items()} for k, m in wave5_menus.items()},
            'per_role_stats': {r: {'bip_red_mean': float(np.mean(s['bip_red'])),
                                    'bip_red_std': float(np.std(s['bip_red'])),
                                    'had_red_mean': float(np.mean(s['had_red'])),
                                    'had_red_std': float(np.std(s['had_red'])),
                                    'n_blocks': len(s['bip_red'])}
                               for r, s in wave5_role_stats.items()},
            'per_depth_stats': {str(d): {'bip_red_mean': float(np.mean(s['bip_red'])),
                                          'bip_red_std': float(np.std(s['bip_red'])),
                                          'had_red_mean': float(np.mean(s['had_red'])),
                                          'had_red_std': float(np.std(s['had_red'])),
                                          'n_blocks': len(s['bip_red'])}
                                 for d, s in wave5_depth_stats.items()},
        },
    }

    # Convert numpy types for JSON
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        if isinstance(o, (bool,)):
            return o
        return o

    results = convert(results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Q1: Does including transforms improve the frontier?
    print("\nQ1: Does including transforms (BiP/Had) improve the global R-D frontier?")
    for scenario, label in [('k5_uniform', 'K5-uniform budget'),
                            ('5.5bpw', '5.5 bpw budget'),
                            ('k6_uniform', 'K6-uniform budget')]:
        if scenario in scenario_results:
            r = scenario_results[scenario]
            print(f"  {label}: transforms improve HWE by {r['improvement_pct']:+.1f}%")

    # Q2: Which tensors benefit most from Hadamard?
    print(f"\nQ2: Which tensors benefit most from Hadamard?")
    for i, key in enumerate(ranked[:3]):
        avg_had = np.mean([sensitivity[key][K]['reduction_had_pct'] for K in K_VALUES])
        avg_bip = np.mean([sensitivity[key][K]['reduction_bip_pct'] for K in K_VALUES])
        print(f"  {i+1}. {key}: Had={avg_had:.1f}% BiP={avg_bip:.1f}% avg HWE reduction")

    # Q3: Sidecar sensitivity (oracle cost-sensitivity, not realizable codec)
    print(f"\nQ3: Sidecar rate sensitivity (ORACLE cost-sensitivity at 5.5 bpw):")
    for sc in [PROD_BIP_SIDECAR_BPW, 0.0625, 0.125, 0.25, SLICE_BIP_SIDECAR_BPW]:
        if str(sc) in sweep_results or sc in sweep_results:
            r = sweep_results.get(sc, sweep_results.get(str(sc), {}))
            if r:
                print(f"  {sc:.6f} bpw sidecar: BiP={r['n_B']} Had={r['n_H']} (of {len(tensors)}), improvement {r['improvement_pct']:+.1f}%")

    # Q4: Mixed 9-option at 5.5 bpw vs K6-uniform
    r55 = scenario_results['5.5bpw']
    print(f"\nQ4: Mixed 9-option allocation at 5.5 bpw vs K6-uniform:")
    print(f"  Mixed 9-opt D = {r55['full']['distortion']:.6e}  bytes={r55['full']['bytes']:.0f}")
    print(f"  K6-uniform D  = {r55['k6_uniform']['distortion']:.6e}  bytes={r55['k6_uniform']['bytes']:.0f}")
    ratio = r55['full']['distortion'] / r55['k6_uniform']['distortion']
    byte_ratio = r55['full']['bytes'] / r55['k6_uniform']['bytes']
    print(f"  HWE ratio: {ratio:.3f}x  byte ratio: {byte_ratio:.3f}x")
    if ratio < 1.0:
        print(f"  >>> YES: mixed 9-option beats K6-uniform at fewer bytes <<<")
    else:
        print(f"  Mixed 9-option has {ratio:.1%} of K6-uniform HWE at {byte_ratio:.1%} of bytes")

    # Q5: K4+B vs K5 and K4+H vs K5
    print(f"\nQ5: K4+B vs K5 and K4+H vs K5:")
    print(f"  K4+B total: D={k4b_results['k4b_total']['distortion']:.6e}  bytes={k4b_results['k4b_total']['bytes']:.0f}")
    print(f"  K5  total: D={k4b_results['k5_total']['distortion']:.6e}  bytes={k4b_results['k5_total']['bytes']:.0f}")
    ratio = k4b_results['k4b_total']['distortion'] / k4b_results['k5_total']['distortion']
    print(f"  K4+B/K5 ratio: {ratio:.3f}x")

    # Q6: Real-only results
    print(f"\nQ6: Real-only analysis (4 real tensors, no synthetic):")
    for label, r in real_only_results.items():
        print(f"  {label}: transform improvement {r['improvement_pct']:+.1f}%  B-tensors={r['n_B']}/4")

    # Caveats
    print(f"\n" + "=" * 80)
    print("IMPORTANT CAVEATS")
    print("=" * 80)
    print("  1. Baseline is naive 16x16 uniform quantization, NOT EXL3's existing incoherence.")
    print("     R26: BiP 68.9% vs naive but only 63.9% vs random Hadamard. Stock EXL3+GPTQ: 93.4%.")
    print("  2. R27: BiIP scaling HURTS per-tile quant (1/36 wins). Hadamard alone helps (20/36).")
    print("     BUT in original-basis HWE, BiP (scaling+Hadamard) dominates Hadamard-only.")
    print("     Discrepancy is FIT SCOPE (R28 slice-local scales vs R27 global scales), not metric.")
    print("  3. 8/12 tensors are SYNTHETIC. Real-only results (Q6) are more reliable.")
    print("  4. Sidecar sweep is ORACLE cost-sensitivity, not realizable compressed sidecar.")
    print("  5. Local HWE can INVERT end-to-end KLD (QSRT lesson). KLD harness must authorize.")
    print("  6. Sidecar rates: production 17408x5120 BiP=0.0083 bpw Had=0.00025 bpw.")
    print("     Slice 128x128 BiP=0.5156 bpw Had=0.0156 bpw (ASPECT RATIO ARTIFACT, 62x overpriced).")
    print("     Default uses production rates. At production scale, BiP sidecar is essentially free.")
    print("  7. Future: QSRT activation-boundary transform enables joint MLP gate/up/down rotation.")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    main()
