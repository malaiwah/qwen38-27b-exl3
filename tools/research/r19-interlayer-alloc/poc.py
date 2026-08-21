#!/usr/bin/env python3
"""
R19-InterlayerAlloc: Inter-layer optimal K allocation.

Given a TOTAL byte budget across ALL 56 layers, find the optimal K per layer.
This is the inter-layer allocation problem from KronQ (joint-trace sensitivity).

=== Mathematical Formulation ===

The trellis quantization error objective (from doc 62 §10.10.B):
  min  sum_l  tr(H_G_l · E_l · H_X_l · E_l^T)
  s.t. sum_l  bytes(K_l, size_l) <= B

where:
  - l indexes (layer, role) pairs
  - E_l = W_l - Q(W_l, K_l) is the quantization error
  - H_X_l is the input Hessian (activation covariance)
  - H_G_l is the output Hessian (gradient covariance proxy)
  - bytes(K_l, size_l) is the exact byte cost at K bits

This is a multiple-choice knapsack across layers:
  - Each (layer, role) tensor is an "item"
  - Each item has choices K=3,4,5,6,7 with cost=bytes and distortion=D_l(K)
  - DP finds the global optimum minimizing total distortion at total budget B

=== Components ===

1. Per-layer sensitivity measurement: D_l(K) for each layer × role at K=3..7
2. Inter-layer DP: multiple-choice knapsack across all layers
3. Layer sensitivity ranking: which layers are most sensitive?
4. Role-dependent allocation: gate, down, attention separately
5. Interaction with rotation: does rotation change inter-layer sensitivity?
6. Budget frontier: trace full R-D frontier across all layers
7. KronQ joint-trace: tr(H_G)×tr(H_X) as cheap sensitivity proxy vs measured

=== Constraints ===

- CPU-only numpy (no GPU)
- Real weights: L0 and L55 from npz (gate, down, qkv, out, z)
- Synthetic weights for L10, L20, L30, L40 with layer-appropriate statistics
- Per-tile (16×16) uniform quantizer (same as R1)
- Exact byte budget accounting (payload + sidecar + metadata)
- Primary metric: tr(H_G · E · H_X · E^T) (Hessian-weighted error, HWE)
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
M_DIM = 128  # slice dimension for analysis
N_DIM = 128
K_VALUES = [3, 4, 5, 6, 7]
K_MIN = min(K_VALUES)
K_MAX = max(K_VALUES)
P_CAL = 512  # calibration samples
N_TILES = (M_DIM // TILE) * (N_DIM // TILE)  # 64
ELEMENTS_PER_TILE = TILE * TILE  # 256
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"

# Model architecture
N_LAYERS = 56
HIDDEN_DIM = 5120
INTER_DIM = 17408
# Roles: gate_proj, down_proj, qkv (merged), out_proj
ROLES = {
    'gate': {'shape': (INTER_DIM, HIDDEN_DIM), 'desc': 'MLP gate/up'},
    'down': {'shape': (HIDDEN_DIM, INTER_DIM), 'desc': 'MLP down'},
    'qkv':  {'shape': (10240, HIDDEN_DIM), 'desc': 'Attention QKV merged'},
    'out':  {'shape': (HIDDEN_DIM, 6144), 'desc': 'Attention output proj'},
}
ROLE_SEED_MAP = {'gate': 0, 'down': 1, 'qkv': 2, 'out': 3}
def role_seed(role, layer):
    """Stable, deterministic seed for a (layer, role) pair. No hash() dependency."""
    return layer * 100 + ROLE_SEED_MAP.get(role, 99) * 37 + 17
# Sample layers for detailed analysis
SAMPLE_LAYERS = [0, 10, 20, 30, 40, 55]

# ============================================================================
# Weight loading and synthetic generation
# ============================================================================

def load_real_weights():
    """Load real Qwen3.8-27B BF16 weights."""
    tensors = {}
    data = np.load(WEIGHTS_PATH)
    for key in data.files:
        tensors[key] = data[key].astype(np.float64)
    return tensors

def extract_slice(tensor, m=M_DIM, n=N_DIM, seed=42):
    """Extract a representative 128×128 slice from a large tensor."""
    M, N = tensor.shape
    rng = np.random.default_rng(seed)
    # Center-weighted extraction: prefer center of tensor
    r0 = (M - m) // 2 + rng.integers(-M//8, M//8, endpoint=True)
    c0 = (N - n) // 2 + rng.integers(-N//8, N//8, endpoint=True)
    r0 = max(0, min(M - m, r0))
    c0 = max(0, min(N - n, c0))
    return tensor[r0:r0+m, c0:c0+n].copy()

def gen_synthetic_weights(layer_idx, role, shape, real_stats=None, seed=42):
    """Generate synthetic weights with layer-appropriate statistics.

    Late layers (higher index) tend to have larger weight magnitudes.
    We calibrate the magnitude scaling from real L0 and L55 data.
    """
    rng = np.random.default_rng(seed + role_seed(role, layer_idx))

    # Interpolate magnitude between L0 and L55
    # Real data: L0 gate std ~0.012, L55 gate std ~0.015 (approximate)
    # We'll measure from real data
    t = layer_idx / (N_LAYERS - 1)  # 0.0 for L0, 1.0 for L55

    if real_stats and f'L0_{role}' in real_stats and f'L55_{role}' in real_stats:
        std0 = real_stats[f'L0_{role}']['std']
        std55 = real_stats[f'L55_{role}']['std']
    else:
        std0, std55 = 0.01, 0.015

    # Linear interpolation in log space
    target_std = std0 * (std55 / std0) ** t

    m, n = min(shape[0], M_DIM), min(shape[1], N_DIM)

    # Generate weights: Gaussian base + sparse outliers
    W = rng.standard_normal((m, n)) * target_std
    # Add outliers (5% of elements get 3× boost)
    outlier_mask = rng.random((m, n)) < 0.05
    W[outlier_mask] *= 3.0
    # Add structured component (low-rank)
    U = rng.standard_normal((m, 4))
    V = rng.standard_normal((4, n))
    W += 0.3 * target_std * (U @ V)

    # Rescale to match target_std (outliers and low-rank inflate realized std)
    realized_std = np.std(W)
    if realized_std > 1e-15:
        W = W * (target_std / realized_std)

    return W

def measure_weight_stats(tensor):
    """Measure basic statistics of a weight tensor."""
    return {
        'mean': float(np.mean(tensor)),
        'std': float(np.std(tensor)),
        'min': float(np.min(tensor)),
        'max': float(np.max(tensor)),
        'abs_mean': float(np.mean(np.abs(tensor))),
        'abs_max': float(np.max(np.abs(tensor))),
    }

# ============================================================================
# Calibration and Hessian computation
# ============================================================================

def gen_calibration(N, P, seed=42):
    """Generate synthetic calibration activations with realistic structure."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    # Per-channel scale variation (log-uniform)
    scales = np.exp(rng.uniform(-2, 1, N))
    X *= scales[:, None]
    # Outlier channels (5% of channels get 5× boost)
    outlier = rng.random(N) < 0.05
    X[outlier] *= 5.0
    return X

def compute_hessians(W, X):
    """Compute input and output Hessians.
    H_X = X @ X.T / P  (N×N, input Hessian)
    H_G = Y @ Y.T / P  (M×M, output Hessian, Y = W @ X)
    """
    M, N = W.shape
    P = X.shape[1]
    H_X = X @ X.T / P
    Y = W @ X
    H_G = Y @ Y.T / P
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
# Byte budget accounting
# ============================================================================

def tensor_bytes(K, M, N, tile=TILE):
    """Exact byte count for a tensor at uniform K.
    payload: M*N * K / 8 bytes (K bits per element packed)
    sidecar: 2 float16 per tile (min, max) = 4 bytes per tile
    metadata: 3 bits per tile (K value, uniform → 0 if all same)
    For uniform K, metadata is negligible (1 value for whole tensor).
    """
    n_tiles = (M // tile) * (N // tile)
    payload = M * N * K / 8  # bits to bytes
    sidecar = n_tiles * 4  # 2 × float16 per tile
    # For uniform K, metadata = 1 byte (just the K value)
    metadata = 1
    return payload + sidecar + metadata

def slice_bytes(K, M=M_DIM, N=N_DIM, tile=TILE):
    """Bytes for a 128×128 slice at uniform K."""
    return tensor_bytes(K, M, N, tile)

# ============================================================================
# Distortion measurement
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """Primary metric: tr(H_G · E · H_X · E^T)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))

def measure_distortion_curve(W, H_X, H_G, k_values=K_VALUES):
    """Measure distortion D(K) at each K for a tensor slice.
    Returns dict: {K: (hwe, weight_mse, bytes)}
    """
    results = {}
    M, N = W.shape
    for k in k_values:
        Wq = quantize_matrix_uniform(W, k)
        E = W - Wq
        hwe = hessian_weighted_error(E, H_G, H_X)
        mse = float(np.mean(E ** 2))
        b = slice_bytes(k, M, N)
        results[k] = {
            'hwe': max(hwe, 0.0),
            'mse': mse,
            'bytes': b,
        }
    return results

# ============================================================================
# KronQ joint-trace sensitivity approximation
# ============================================================================

def kronq_joint_trace(H_G, H_X):
    """KronQ joint-trace sensitivity: tr(H_G) × tr(H_X).
    This is a cheap proxy that doesn't require quantizing at each K.
    Higher value → more sensitive layer → needs more bits.
    """
    return float(np.trace(H_G) * np.trace(H_X))

def kronq_normalized_sensitivity(H_G, H_X, W):
    """Normalized joint-trace: tr(H_G)×tr(H_X) / (M×N).
    Per-element sensitivity for cross-tensor comparison.
    """
    M, N = W.shape
    return kronq_joint_trace(H_G, H_X) / (M * N)

def hessian_trace_ratio(H_G, H_X):
    """Ratio tr(H_G)/tr(H_X) — asymmetry measure."""
    tg = float(np.trace(H_G))
    tx = float(np.trace(H_X))
    return tg / max(tx, 1e-15)

# ============================================================================
# BiIP + Hadamard rotation (from R3, for interaction test)
# ============================================================================

def biip_balance(W, H_X, H_G):
    """BiIP diagonal balancing: S_X = (diag(H_X)/diag(W^T W))^{1/4}.
    Uses the R3 convention: W_t = S_G @ W @ S_X.
    The Hessians transform as: H_X,t = S_X^{-1} H_X S_X^{-1}, H_G,t = S_G^{-1} H_G S_G^{-1}.
    This ensures tr(H_G,t E_t H_X,t E_t^T) = tr(H_G E H_X E^T) when E_t = S_G E S_X.
    Returns transformed W, transformed Hessians, and scale vectors.
    """
    M, N = W.shape
    diag_HX = np.diag(H_X).copy()
    diag_HG = np.diag(H_G).copy()
    diag_WTW = np.sum(W ** 2, axis=0)  # per-column (N,)
    diag_WWT = np.sum(W ** 2, axis=1)  # per-row (M,)

    # Avoid division by zero
    diag_WTW = np.maximum(diag_WTW, 1e-15)
    diag_WWT = np.maximum(diag_WWT, 1e-15)

    S_X = (diag_HX / diag_WTW) ** 0.25  # input scale (N,)
    S_G = (diag_HG / diag_WWT) ** 0.25  # output scale (M,)

    # R3 convention: W_t = S_G @ W @ S_X
    W_t = S_G[:, None] * W * S_X[None, :]

    # Transform Hessians: H_X,t = S_X^{-1} H_X S_X^{-1}, H_G,t = S_G^{-1} H_G S_G^{-1}
    H_X_t = H_X / S_X[None, :] / S_X[:, None]
    H_G_t = H_G / S_G[None, :] / S_G[:, None]

    return W_t, H_X_t, H_G_t, S_X, S_G

def signed_hadamard(n, seed=42):
    """Signed randomized Hadamard matrix of size n (n must be power of 2)."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    # Base Hadamard (Sylvester construction)
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    # Random sign flips (permutation of rows/cols)
    rng = np.random.default_rng(seed)
    signs_r = rng.choice([-1, 1], size=n)
    signs_c = rng.choice([-1, 1], size=n)
    H = np.diag(signs_r) @ H @ np.diag(signs_c)
    return H / np.sqrt(n)

def apply_rotation(W, H_X, H_G, seed=42):
    """Apply BiIP + Hadamard rotation to W.

    Correct R3 convention:
    1. BiIP balancing: W_t = S_G @ W @ S_X, H_X,t = S_X^{-1} H_X S_X^{-1}, H_G,t = S_G^{-1} H_G S_G^{-1}
    2. Hadamard rotation: W_rot = Q_G^T @ W_t @ Q_X, H_X_rot = Q_X^T @ H_X,t @ Q_X, H_G_rot = Q_G^T @ H_G,t @ Q_G

    The rotation invariance guarantees:
    tr(H_G E H_X E^T) = tr(H_G_rot E_rot H_X_rot E_rot^T)
    where E_rot = Q_G^T @ S_G @ E @ S_X @ Q_X
    """
    M, N = W.shape
    W_t, H_X_t, H_G_t, S_X, S_G = biip_balance(W, H_X, H_G)

    # Hadamard matrices
    Q_X = signed_hadamard(N, seed=seed)
    Q_G = signed_hadamard(M, seed=seed + 1)

    # Rotate both W and Hessians
    W_rot = Q_G.T @ W_t @ Q_X
    H_X_rot = Q_X.T @ H_X_t @ Q_X
    H_G_rot = Q_G.T @ H_G_t @ Q_G

    # Invariance test: verify tr(H_G E H_X E^T) ≈ tr(H_G_rot E_rot H_X_rot E_rot^T)
    # for a random error matrix (diagnostic only, not returned)
    # E_test = np.random.default_rng(99).standard_normal((M, N)) * 0.01
    # E_rot_test = Q_G.T @ (S_G[:, None] * E_test * S_X[None, :]) @ Q_X
    # hwe_orig = np.trace(H_G @ E_test @ H_X @ E_test.T)
    # hwe_rot = np.trace(H_G_rot @ E_rot_test @ H_X_rot @ E_rot_test.T)
    # assert abs(hwe_orig - hwe_rot) / max(abs(hwe_orig), 1e-15) < 1e-10, \
    #     f"Rotation invariance violated: {hwe_orig} vs {hwe_rot}"

    return W_rot, H_X_rot, H_G_rot

# ============================================================================
# Inter-layer DP: multiple-choice knapsack
# ============================================================================

def interlayer_dp(items, budget_bytes, k_values=K_VALUES):
    """Inter-layer DP: optimal K per (layer, role) minimizing total distortion.

    Multiple-choice knapsack using K-sum as budget unit.
    Since all slices have identical size, bytes = f(K_sum) exactly,
    so minimizing at fixed K-sum is equivalent to fixed byte budget.
    For heterogeneous sizes, we use a per-item cost = K * size_units.

    Returns: dict {item_id: K}, total_distortion, total_bytes
    """
    n_items = len(items)

    # Use K-sum as budget unit (all slices same size → exact)
    # If items have different sizes, use weighted K-sum
    # Check if all items have same byte costs
    ref_bytes = items[0]['bytes'][k_values[0]]
    all_same = all(
        abs(item['bytes'][k] - items[0]['bytes'][k]) < 1
        for item in items for k in k_values
    )

    if all_same:
        # Uniform-size DP: budget = K_sum
        # Compute target K-sum from budget_bytes
        # bytes_per_item at K = M*N*K/8 + n_tiles*4 + 1
        # total_bytes = sum over items = n_items * (M*N*K_i/8 + sidecar + meta)
        # K_sum = sum(K_i)
        # total_bytes = (M*N/8) * K_sum + n_items * sidecar + n_items * meta
        # So K_sum = (total_bytes - n_items * (sidecar + meta)) / (M*N/8)
        # But we don't know M,N here. Instead, compute from first item.
        b_at_k3 = items[0]['bytes'][k_values[0]]
        b_at_k7 = items[0]['bytes'][k_values[-1]]
        k_range = k_values[-1] - k_values[0]
        bytes_per_k = (b_at_k7 - b_at_k3) / k_range  # marginal bytes per K
        fixed_per_item = b_at_k3 - k_values[0] * bytes_per_k  # sidecar + meta
        # total_bytes = K_sum * bytes_per_k + n_items * fixed_per_item
        # K_sum_budget = (budget_bytes - n_items * fixed_per_item) / bytes_per_k
        k_sum_budget = int((budget_bytes - n_items * fixed_per_item) / bytes_per_k)
        k_sum_budget = max(n_items * min(k_values), min(k_sum_budget, n_items * max(k_values)))

        return _dp_ksum(items, k_sum_budget, k_values, bytes_per_k, fixed_per_item)
    else:
        # Heterogeneous: use weighted K-sum
        # Compute size units per item (proportional to element count)
        # cost[i][ki] = K * size_unit_i
        # For simplicity, use bytes directly with coarse quantization
        return _dp_coarse(items, budget_bytes, k_values)


def _dp_ksum(items, k_sum_budget, k_values, bytes_per_k, fixed_per_item):
    """DP using K-sum as exact budget unit. O(n_items * k_sum_budget * |K|)."""
    n_items = len(items)
    k_min, k_max = min(k_values), max(k_values)
    max_ksum = n_items * k_max
    k_sum_budget = min(k_sum_budget, max_ksum)

    INF = float('inf')
    dp = np.full(k_sum_budget + 1, INF)
    dp[0] = 0.0
    choices = []

    for i in range(n_items):
        new_dp = np.full(k_sum_budget + 1, INF)
        new_choice = np.full(k_sum_budget + 1, -1, dtype=int)
        dists_i = [items[i]['distortion'][k] for k in k_values]
        for j in range(k_sum_budget + 1):
            if dp[j] == INF:
                continue
            for ki, k_val in enumerate(k_values):
                nj = j + k_val
                if nj > k_sum_budget:
                    continue
                val = dp[j] + dists_i[ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append(new_choice.copy())

    # Find best
    best_j = 0
    best_d = INF
    for j in range(k_sum_budget + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    # Backtrack
    assignment = {}
    j = best_j
    for i in range(n_items - 1, -1, -1):
        ki = choices[i][j]
        if ki < 0:
            for fki in range(len(k_values)):
                if k_values[fki] <= j:
                    ki = fki
                    break
        assignment[items[i]['id']] = k_values[ki]
        j -= k_values[ki]

    total_bytes = sum(assignment[items[i]['id']] * bytes_per_k + fixed_per_item
                      for i in range(n_items))
    return assignment, best_d, total_bytes


def _dp_coarse(items, budget_bytes, k_values):
    """DP for heterogeneous item sizes using coarse byte quantization."""
    n_items = len(items)
    # Find GCD-like granularity: use K*elements as cost
    # All items have same element count (128*128), so fall back to K-sum
    # For truly heterogeneous, use 1-byte resolution but with numpy vectorization
    SCALE = 1  # 1 byte resolution
    budget_int = int(budget_bytes) + 1

    # If budget too large, use coarser scale
    if budget_int > 200000:
        SCALE = max(1, budget_int // 200000)
        budget_int = budget_int // SCALE + 1

    INF = float('inf')
    dp = np.full(budget_int + 1, INF)
    dp[0] = 0.0
    choices = []

    for i in range(n_items):
        new_dp = np.full(budget_int + 1, INF)
        new_choice = np.full(budget_int + 1, -1, dtype=int)
        costs_i = [int(items[i]['bytes'][k] / SCALE) for k in k_values]
        dists_i = [items[i]['distortion'][k] for k in k_values]
        for j in range(budget_int + 1):
            if dp[j] == INF:
                continue
            for ki in range(len(k_values)):
                nj = j + costs_i[ki]
                if nj > budget_int:
                    continue
                val = dp[j] + dists_i[ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append(new_choice.copy())

    best_j = 0
    best_d = INF
    for j in range(budget_int + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    assignment = {}
    j = best_j
    for i in range(n_items - 1, -1, -1):
        ki = choices[i][j]
        if ki < 0:
            for fki in range(len(k_values)):
                if int(items[i]['bytes'][k_values[fki]] / SCALE) <= j:
                    ki = fki
                    break
        assignment[items[i]['id']] = k_values[ki]
        j -= int(items[i]['bytes'][k_values[ki]] / SCALE)

    total_bytes = sum(items[i]['bytes'][assignment[items[i]['id']]] for i in range(n_items))
    return assignment, best_d, total_bytes

# ============================================================================
# Experiment 1: Per-layer distortion curves
# ============================================================================

def experiment_per_layer_curves(real_weights):
    """Measure D_l(K) for each sample layer × role."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Per-layer distortion curves D_l(K)")
    print("=" * 80)

    # Measure real weight stats for synthetic calibration
    real_stats = {}
    for key in real_weights:
        stats = measure_weight_stats(real_weights[key])
        layer = key.split('_')[0]  # L0 or L55
        role = key[len(layer)+1:]  # gate, down, qkv, out, z
        real_stats[f'{layer}_{role}'] = stats
        print(f"  Real {key}: std={stats['std']:.6f}, abs_max={stats['abs_max']:.6f}")

    # Build tensor registry
    tensor_registry = {}  # {id: (W_slice, source)}
    distortion_curves = {}  # {id: {K: {hwe, mse, bytes}}}
    kronq_sensitivities = {}  # {id: float}
    hessian_traces = {}  # {id: (tr_HG, tr_HX, ratio)}

    for layer in SAMPLE_LAYERS:
        for role in ROLES:
            tensor_id = f"L{layer}_{role}"

            # Get weight slice
            if layer == 0 and f'L0_{role}' in real_weights:
                W_slice = extract_slice(real_weights[f'L0_{role}'])
            elif layer == 55 and f'L55_{role}' in real_weights:
                W_slice = extract_slice(real_weights[f'L55_{role}'])
            else:
                # Synthetic
                W_slice = gen_synthetic_weights(layer, role,
                    ROLES[role]['shape'], real_stats, seed=layer)
                # Ensure 128×128
                if W_slice.shape != (M_DIM, N_DIM):
                    W_slice = W_slice[:M_DIM, :N_DIM]

            tensor_registry[tensor_id] = W_slice

            # Compute Hessians
            X = gen_calibration(N_DIM, P_CAL, seed=role_seed(role, layer))
            H_X, H_G = compute_hessians(W_slice, X)

            # Measure distortion curve
            curve = measure_distortion_curve(W_slice, H_X, H_G)
            distortion_curves[tensor_id] = curve

            # KronQ joint-trace sensitivity
            jt = kronq_joint_trace(H_G, H_X)
            jt_norm = kronq_normalized_sensitivity(H_G, H_X, W_slice)
            kronq_sensitivities[tensor_id] = jt_norm

            tr_g = float(np.trace(H_G))
            tr_x = float(np.trace(H_X))
            hessian_traces[tensor_id] = (tr_g, tr_x, tr_g / max(tr_x, 1e-15))

    # Print distortion curves
    print("\nPer-layer distortion curves (HWE):")
    print(f"{'Tensor':<15} {'K3':>12} {'K4':>12} {'K5':>12} {'K6':>12} {'K7':>12} {'KronQ_sens':>12}")
    print("-" * 90)
    for tid in sorted(distortion_curves.keys()):
        curve = distortion_curves[tid]
        hwe_vals = [f"{curve[k]['hwe']:.6e}" for k in K_VALUES]
        ks = f"{'K3':>12} {'K4':>12}"  # placeholder
        print(f"{tid:<15} " + " ".join(f"{v:>12}" for v in hwe_vals) + f" {kronq_sensitivities[tid]:>12.4e}")

    return distortion_curves, kronq_sensitivities, hessian_traces, tensor_registry, real_stats

# ============================================================================
# Experiment 2: Inter-layer DP allocation
# ============================================================================

def experiment_interlayer_dp(distortion_curves, tensor_registry):
    """Run inter-layer DP at various budget levels."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Inter-layer DP allocation")
    print("=" * 80)

    # Build items for DP
    items = []
    for tid in sorted(distortion_curves.keys()):
        curve = distortion_curves[tid]
        item = {
            'id': tid,
            'distortion': {k: curve[k]['hwe'] for k in K_VALUES},
            'bytes': {k: curve[k]['bytes'] for k in K_VALUES},
        }
        items.append(item)

    # Budget sweep: from K3-all to K7-all
    n_items = len(items)
    budget_sweep = []
    for avg_k in K_VALUES:
        uniform_bytes = sum(slice_bytes(avg_k) for _ in items)
        budget_sweep.append((f"K{avg_k}-uniform-budget", uniform_bytes))

    # Also add intermediate budgets
    for avg_k_float in np.arange(3.0, 7.1, 0.5):
        avg_k_int = int(np.floor(avg_k_float))
        frac = avg_k_float - avg_k_int
        # Approximate budget: blend between K_int and K_int+1
        b_low = sum(slice_bytes(avg_k_int) for _ in items)
        b_high = sum(slice_bytes(avg_k_int + 1) for _ in items) if avg_k_int < K_MAX else b_low
        budget = b_low * (1 - frac) + b_high * frac
        budget_sweep.append((f"avg-{avg_k_float:.1f}", budget))

    # Remove duplicates and sort by budget
    seen = set()
    unique_sweep = []
    for name, b in budget_sweep:
        b_int = int(b)
        if b_int not in seen:
            seen.add(b_int)
            unique_sweep.append((name, b_int))
    unique_sweep.sort(key=lambda x: x[1])

    print(f"\nBudget sweep ({len(unique_sweep)} budget levels):")
    print(f"{'Budget':>12} {'Name':<25} {'Total D':>12} {'Avg K':>6} {'K assignment':>40}")
    print("-" * 100)

    dp_results = []
    for name, budget in unique_sweep:
        assignment, total_d, total_b = interlayer_dp(items, budget)
        avg_k = np.mean(list(assignment.values()))
        k_str = ", ".join(f"{tid.split('_')[0]}:{v}" for tid, v in sorted(assignment.items()))
        print(f"{budget:>12} {name:<25} {total_d:>12.4e} {avg_k:>6.2f} {k_str}")

        dp_results.append({
            'budget_name': name,
            'budget_bytes': budget,
            'total_distortion': total_d,
            'total_bytes': total_b,
            'avg_k': avg_k,
            'assignment': assignment.copy(),
        })

    # Compare DP vs uniform at each K
    print("\nDP vs Uniform comparison:")
    print(f"{'Budget level':<25} {'DP distortion':>14} {'Uniform D':>14} {'Improvement':>12}")
    print("-" * 70)
    for avg_k in K_VALUES:
        budget = sum(slice_bytes(avg_k) for _ in items)
        # Uniform
        uniform_d = sum(distortion_curves[tid][avg_k]['hwe'] for tid in distortion_curves)
        # DP at same budget
        assignment, dp_d, _ = interlayer_dp(items, budget)
        improvement = (uniform_d - dp_d) / uniform_d * 100 if uniform_d > 0 else 0
        print(f"K{avg_k}-budget ({budget}B)  {dp_d:>14.4e} {uniform_d:>14.4e} {improvement:>11.1f}%")

    return dp_results

# ============================================================================
# Experiment 3: Layer sensitivity ranking
# ============================================================================

def experiment_sensitivity_ranking(distortion_curves, kronq_sensitivities):
    """Rank layers by sensitivity."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Layer sensitivity ranking")
    print("=" * 80)

    # Sensitivity = distortion at a reference K (use K5 as midpoint)
    ref_k = 5
    measured_sens = {tid: curve[ref_k]['hwe'] for tid, curve in distortion_curves.items()}

    # Rank by measured sensitivity
    ranked = sorted(measured_sens.items(), key=lambda x: -x[1])
    print(f"\nSensitivity ranking (by HWE at K={ref_k}):")
    print(f"{'Rank':<6} {'Tensor':<15} {'HWE@K5':>12} {'KronQ_sens':>12} {'Ratio':>8}")
    print("-" * 60)
    for rank, (tid, sens) in enumerate(ranked):
        ks = kronq_sensitivities[tid]
        ratio = sens / ks if ks > 0 else 0
        print(f"{rank+1:<6} {tid:<15} {sens:>12.4e} {ks:>12.4e} {ratio:>8.2f}")

    # Group by layer
    print("\nPer-layer aggregate sensitivity (sum across roles):")
    layer_sens = {}
    for tid, sens in measured_sens.items():
        layer = tid.split('_')[0]
        layer_sens.setdefault(layer, []).append(sens)
    for layer in sorted(layer_sens.keys(), key=lambda x: int(x[1:])):
        vals = layer_sens[layer]
        role_strs = [f"{t.split('_')[1]}:{measured_sens[t]:.2e}" for t in sorted(distortion_curves.keys()) if t.startswith(layer)]
        print(f"  {layer}: sum={sum(vals):.4e}, mean={np.mean(vals):.4e}, roles: {role_strs}")

    # Group by role
    print("\nPer-role aggregate sensitivity (sum across layers):")
    role_sens = {}
    for tid, sens in measured_sens.items():
        role = tid.split('_')[1]
        role_sens.setdefault(role, []).append(sens)
    for role in sorted(role_sens.keys()):
        vals = role_sens[role]
        print(f"  {role}: sum={sum(vals):.4e}, mean={np.mean(vals):.4e}, "
              f"std={np.std(vals):.4e}")

    # Correlation: measured sensitivity vs KronQ
    measured_vals = np.array([measured_sens[tid] for tid in kronq_sensitivities])
    kronq_vals = np.array([kronq_sensitivities[tid] for tid in kronq_sensitivities])
    correlation = np.corrcoef(np.log(measured_vals + 1e-30),
                              np.log(kronq_vals + 1e-30))[0, 1]
    print(f"\nCorrelation (log) measured vs KronQ: {correlation:.4f}")

    # Also at each K
    print("\nCorrelation measured vs KronQ at each K:")
    for k in K_VALUES:
        mv = np.array([distortion_curves[tid][k]['hwe'] for tid in kronq_sensitivities])
        kv = np.array([kronq_sensitivities[tid] for tid in kronq_sensitivities])
        corr = np.corrcoef(np.log(mv + 1e-30), np.log(kv + 1e-30))[0, 1]
        print(f"  K{k}: r={corr:.4f}")

    return ranked, measured_sens

# ============================================================================
# Experiment 4: Role-dependent allocation
# ============================================================================

def experiment_role_dependent(distortion_curves):
    """Test role-dependent allocation: separate budgets per role."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Role-dependent allocation")
    print("=" * 80)

    # Group items by role
    role_items = {}
    for tid in sorted(distortion_curves.keys()):
        role = tid.split('_')[1]
        curve = distortion_curves[tid]
        role_items.setdefault(role, []).append({
            'id': tid,
            'distortion': {k: curve[k]['hwe'] for k in K_VALUES},
            'bytes': {k: curve[k]['bytes'] for k in K_VALUES},
        })

    # Test 1: Global DP (all items together) vs role-partitioned DP
    # At K5 budget
    all_items = []
    for role in role_items:
        all_items.extend(role_items[role])

    budget_k5 = sum(slice_bytes(5) for _ in all_items)

    # Global DP
    global_assign, global_d, global_b = interlayer_dp(all_items, budget_k5)
    print(f"\nGlobal DP at K5 budget ({budget_k5} bytes):")
    print(f"  Total distortion: {global_d:.6e}")
    print(f"  Assignment: {global_assign}")

    # Role-partitioned DP: allocate budget proportional to role's share
    print(f"\nRole-partitioned DP (proportional budget):")
    role_assigns = {}
    role_d_total = 0
    role_b_total = 0
    for role in sorted(role_items.keys()):
        items_r = role_items[role]
        # Proportional budget: each role gets its share of K5 budget
        budget_r = sum(slice_bytes(5) for _ in items_r)
        assign_r, d_r, b_r = interlayer_dp(items_r, budget_r)
        role_assigns[role] = assign_r
        role_d_total += d_r
        role_b_total += b_r
        print(f"  {role}: budget={budget_r}, D={d_r:.6e}, assign={assign_r}")

    improvement = (global_d - role_d_total) / global_d * 100 if global_d > 0 else 0
    print(f"\n  Global D: {global_d:.6e}, Role-partitioned D: {role_d_total:.6e}")
    print(f"  Difference: {improvement:+.1f}% (positive = role-partitioned better)")

    # Test 2: Give more budget to more sensitive roles
    print(f"\nSensitivity-weighted budget allocation:")
    role_sens_sum = {}
    for role in role_items:
        role_sens_sum[role] = sum(item['distortion'][5] for item in role_items[role])

    total_sens = sum(role_sens_sum.values())
    weighted_assigns = {}
    weighted_d_total = 0
    weighted_b_total = 0
    for role in sorted(role_items.keys()):
        items_r = role_items[role]
        # Budget proportional to sensitivity
        budget_r = int(budget_k5 * role_sens_sum[role] / total_sens)
        assign_r, d_r, b_r = interlayer_dp(items_r, budget_r)
        weighted_assigns[role] = assign_r
        weighted_d_total += d_r
        weighted_b_total += b_r
        print(f"  {role}: sens={role_sens_sum[role]:.4e}, budget={budget_r}, "
              f"D={d_r:.6e}, avg_K={np.mean(list(assign_r.values())):.2f}")

    improvement_w = (global_d - weighted_d_total) / global_d * 100 if global_d > 0 else 0
    print(f"\n  Weighted D: {weighted_d_total:.6e} vs Global D: {global_d:.6e}")
    print(f"  Difference: {improvement_w:+.1f}%")

    return {
        'global_dp': {'distortion': global_d, 'assignment': global_assign},
        'role_partitioned': {'distortion': role_d_total, 'assignment': role_assigns},
        'sensitivity_weighted': {'distortion': weighted_d_total, 'assignment': weighted_assigns},
    }

# ============================================================================
# Experiment 5: Interaction with rotation
# ============================================================================

def experiment_rotation_interaction(tensor_registry, distortion_curves):
    """Test whether rotation changes inter-layer sensitivity distribution."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Interaction with rotation")
    print("=" * 80)

    rotated_curves = {}
    rotated_kronq = {}

    for tid in sorted(tensor_registry.keys()):
        W = tensor_registry[tid]
        layer = int(tid.split('_')[0][1:])
        role = tid.split('_')[1]
        X = gen_calibration(N_DIM, P_CAL, seed=role_seed(role, layer))
        H_X, H_G = compute_hessians(W, X)

        # Apply rotation
        W_rot, H_X_rot, H_G_rot = apply_rotation(W, H_X, H_G, seed=layer)


        # Measure rotated distortion curve
        curve = measure_distortion_curve(W_rot, H_X_rot, H_G_rot)
        rotated_curves[tid] = curve

        jt = kronq_normalized_sensitivity(H_G_rot, H_X_rot, W_rot)
        rotated_kronq[tid] = jt

    # Verify rotation invariance for first tensor
    first_tid = sorted(tensor_registry.keys())[0]
    W0 = tensor_registry[first_tid]
    layer0 = int(first_tid.split('_')[0][1:])
    role0 = first_tid.split('_')[1]
    X0 = gen_calibration(N_DIM, P_CAL, seed=role_seed(role0, layer0))
    H_X0, H_G0 = compute_hessians(W0, X0)
    W0_rot, H_X0_rot, H_G0_rot = apply_rotation(W0, H_X0, H_G0, seed=layer0)
    # Fixed random error
    rng_inv = np.random.default_rng(12345)
    E_fixed = rng_inv.standard_normal(W0.shape) * 0.001
    hwe_orig = hessian_weighted_error(E_fixed, H_G0, H_X0)
    # The error in rotated space: E_rot = Q_G^T S_G E S_X Q_X
    # We need to reconstruct this. Since apply_rotation returns W_rot and H_*_rot,
    # the invariance says tr(H_G E H_X E^T) = tr(H_G_rot E_rot H_X_rot E_rot^T)
    # where E_rot is the same E transformed. For the Hessian transform to be correct,
    # we need: if W_rot = Q_G^T S_G W S_X Q_X, then for E_rot = Q_G^T S_G E S_X Q_X,
    # tr(H_G E H_X E^T) = tr(H_G_rot E_rot H_X_rot E_rot^T).
    # Reconstruct S_X, S_G:
    W0_t, H_X0_t, H_G0_t, S_X0, S_G0 = biip_balance(W0, H_X0, H_G0)
    Q_X0 = signed_hadamard(N_DIM, seed=layer0)
    Q_G0 = signed_hadamard(M_DIM, seed=layer0 + 1)
    E_rot_fixed = Q_G0.T @ (S_G0[:, None] * E_fixed * S_X0[None, :]) @ Q_X0
    hwe_rot = hessian_weighted_error(E_rot_fixed, H_G0_rot, H_X0_rot)
    inv_ratio = abs(hwe_orig - hwe_rot) / max(abs(hwe_orig), 1e-15)
    print(f"\nRotation invariance test ({first_tid}):")
    print(f"  HWE original: {hwe_orig:.6e}")
    print(f"  HWE rotated:  {hwe_rot:.6e}")
    print(f"  Relative diff: {inv_ratio:.2e} (should be < 1e-10)")
    assert inv_ratio < 1e-8, f"Rotation invariance violated: {inv_ratio}"

    # Compare sensitivity distributions
    print("\nSensitivity comparison (HWE at K5):")
    print(f"{'Tensor':<15} {'Unrotated':>14} {'Rotated':>14} {'Change':>10} {'CV_unrot':>10} {'CV_rot':>10}")
    print("-" * 75)

    unrot_k5 = {tid: distortion_curves[tid][5]['hwe'] for tid in distortion_curves}
    rot_k5 = {tid: rotated_curves[tid][5]['hwe'] for tid in rotated_curves}

    for tid in sorted(distortion_curves.keys()):
        u = unrot_k5[tid]
        r = rot_k5[tid]
        change = (r - u) / u * 100 if u > 0 else 0
        print(f"{tid:<15} {u:>14.4e} {r:>14.4e} {change:>9.1f}%")

    # Coefficient of variation across tensors (homogenization measure)
    unrot_vals = np.array(list(unrot_k5.values()))
    rot_vals = np.array(list(rot_k5.values()))
    cv_unrot = np.std(unrot_vals) / np.mean(unrot_vals)
    cv_rot = np.std(rot_vals) / np.mean(rot_vals)
    print(f"\nCV across tensors: unrotated={cv_unrot:.4f}, rotated={cv_rot:.4f}")
    print(f"  Homogenization: {(cv_unrot - cv_rot) / cv_unrot * 100:.1f}% reduction")

    # Compare inter-layer DP: unrotated vs rotated
    all_items_unrot = []
    all_items_rot = []
    for tid in sorted(distortion_curves.keys()):
        all_items_unrot.append({
            'id': tid,
            'distortion': {k: distortion_curves[tid][k]['hwe'] for k in K_VALUES},
            'bytes': {k: distortion_curves[tid][k]['bytes'] for k in K_VALUES},
        })
        all_items_rot.append({
            'id': tid,
            'distortion': {k: rotated_curves[tid][k]['hwe'] for k in K_VALUES},
            'bytes': {k: rotated_curves[tid][k]['bytes'] for k in K_VALUES},
        })

    budget_k5 = sum(slice_bytes(5) for _ in all_items_unrot)

    assign_unrot, d_unrot, _ = interlayer_dp(all_items_unrot, budget_k5)
    assign_rot, d_rot, _ = interlayer_dp(all_items_rot, budget_k5)

    # Uniform baselines
    uniform_d_unrot = sum(distortion_curves[tid][5]['hwe'] for tid in distortion_curves)
    uniform_d_rot = sum(rotated_curves[tid][5]['hwe'] for tid in rotated_curves)

    print(f"\nInter-layer DP at K5 budget:")
    print(f"  Unrotated: uniform={uniform_d_unrot:.4e}, DP={d_unrot:.4e}, "
          f"improvement={(uniform_d_unrot - d_unrot) / uniform_d_unrot * 100:.1f}%")
    print(f"  Rotated:   uniform={uniform_d_rot:.4e}, DP={d_rot:.4e}, "
          f"improvement={(uniform_d_rot - d_rot) / uniform_d_rot * 100:.1f}%")

    print(f"\n  DP assignment (unrotated): {assign_unrot}")
    print(f"  DP assignment (rotated):   {assign_rot}")

    # Does rotation change the optimal allocation?
    assign_diff = sum(1 for tid in assign_unrot if assign_unrot[tid] != assign_rot[tid])
    print(f"\n  Assignment differences: {assign_diff}/{len(assign_unrot)} tensors differ")

    return {
        'unrotated': {'uniform_d': uniform_d_unrot, 'dp_d': d_unrot, 'assignment': assign_unrot},
        'rotated': {'uniform_d': uniform_d_rot, 'dp_d': d_rot, 'assignment': assign_rot},
        'cv_unrotated': cv_unrot,
        'cv_rotated': cv_rot,
        'assignment_diffs': assign_diff,
    }

# ============================================================================
# Experiment 6: Budget frontier
# ============================================================================

def experiment_budget_frontier(distortion_curves):
    """Trace the full rate-distortion frontier across all layers."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Budget frontier")
    print("=" * 80)

    all_items = []
    for tid in sorted(distortion_curves.keys()):
        curve = distortion_curves[tid]
        all_items.append({
            'id': tid,
            'distortion': {k: curve[k]['hwe'] for k in K_VALUES},
            'bytes': {k: curve[k]['bytes'] for k in K_VALUES},
        })

    # Sweep budgets from K3-all to K7-all in fine steps
    min_budget = sum(slice_bytes(3) for _ in all_items)
    max_budget = sum(slice_bytes(7) for _ in all_items)
    n_points = 40
    budgets = np.linspace(min_budget, max_budget, n_points)

    frontier = []
    print(f"\n{'Budget':>10} {'Distortion':>14} {'Avg K':>6} {'Assignment summary':>50}")
    print("-" * 85)

    for budget in budgets:
        budget_int = int(budget)
        assignment, total_d, total_b = interlayer_dp(all_items, budget_int)
        avg_k = np.mean(list(assignment.values()))
        # Summary: count per K
        k_counts = {k: sum(1 for v in assignment.values() if v == k) for k in K_VALUES}
        k_summary = " ".join(f"K{k}:{k_counts[k]}" for k in K_VALUES if k_counts[k] > 0)
        frontier.append({
            'budget': budget_int,
            'distortion': total_d,
            'avg_k': avg_k,
            'assignment': assignment.copy(),
            'k_counts': k_counts,
        })
        print(f"{budget_int:>10} {total_d:>14.4e} {avg_k:>6.2f} {k_summary:>50}")

    # Find transition points: where does K5K6 become optimal? When does K4K5 win?
    print("\nTransition analysis:")
    for i in range(1, len(frontier)):
        prev = frontier[i - 1]
        curr = frontier[i]
        # Detect when avg K crosses integer boundaries
        prev_k = prev['avg_k']
        curr_k = curr['avg_k']
        for threshold in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
            if prev_k < threshold <= curr_k or prev_k > threshold >= curr_k:
                print(f"  Budget {curr['budget']}: avg K crosses {threshold} "
                      f"({prev_k:.2f} → {curr_k:.2f})")

    # Find the budget where the K5K6 recipe (gate=K5, down=K6, attention=K6) is optimal
    print("\nK5K6 recipe comparison:")
    k5k6_assign = {}
    for tid in sorted(distortion_curves.keys()):
        role = tid.split('_')[1]
        if role == 'gate':
            k5k6_assign[tid] = 5
        else:
            k5k6_assign[tid] = 6
    k5k6_bytes = sum(slice_bytes(k5k6_assign[tid]) for tid in k5k6_assign)
    k5k6_d = sum(distortion_curves[tid][k5k6_assign[tid]]['hwe'] for tid in distortion_curves)

    # DP at same budget
    dp_assign, dp_d, dp_b = interlayer_dp(all_items, k5k6_bytes)
    improvement = (k5k6_d - dp_d) / k5k6_d * 100 if k5k6_d > 0 else 0
    print(f"  K5K6 recipe: budget={k5k6_bytes:.0f}, D={k5k6_d:.4e}")
    print(f"  DP at same budget: D={dp_d:.4e}")
    print(f"  DP improvement over K5K6: {improvement:.1f}%")
    print(f"  DP assignment: {dp_assign}")
    print(f"  K5K6 assignment: {k5k6_assign}")

    # K4K5 recipe (gate=K4, down=K5, attention=K5)
    k4k5_assign = {}
    for tid in sorted(distortion_curves.keys()):
        role = tid.split('_')[1]
        if role == 'gate':
            k4k5_assign[tid] = 4
        else:
            k4k5_assign[tid] = 5
    k4k5_bytes = sum(slice_bytes(k4k5_assign[tid]) for tid in k4k5_assign)
    k4k5_d = sum(distortion_curves[tid][k4k5_assign[tid]]['hwe'] for tid in distortion_curves)

    dp_assign_k4k5, dp_d_k4k5, _ = interlayer_dp(all_items, k4k5_bytes)
    improvement_k4k5 = (k4k5_d - dp_d_k4k5) / k4k5_d * 100 if k4k5_d > 0 else 0
    print(f"\n  K4K5 recipe: budget={k4k5_bytes:.0f}, D={k4k5_d:.4e}")
    print(f"  DP at same budget: D={dp_d_k4k5:.4e}")
    print(f"  DP improvement over K4K5: {improvement_k4k5:.1f}%")

    return frontier, k5k6_assign, k5k6_d, k4k5_assign, k4k5_d

# ============================================================================
# Experiment 7: KronQ joint-trace comparison
# ============================================================================

def experiment_kronq_comparison(distortion_curves, kronq_sensitivities, hessian_traces):
    """Compare KronQ joint-trace approximation to measured sensitivity."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: KronQ joint-trace vs measured sensitivity")
    print("=" * 80)

    # Measured sensitivity = HWE at K5 (or average across K)
    # KronQ sensitivity = tr(H_G) × tr(H_X) / (M×N)

    print("\nDetailed comparison:")
    print(f"{'Tensor':<15} {'HWE@K5':>12} {'KronQ':>12} {'tr(HG)':>12} {'tr(HX)':>12} "
          f"{'HG/HX':>8} {'Rank_meas':>10} {'Rank_KronQ':>10}")
    print("-" * 95)

    measured = {tid: curve[5]['hwe'] for tid, curve in distortion_curves.items()}
    measured_rank = {tid: i + 1 for i, (tid, _) in
                     enumerate(sorted(measured.items(), key=lambda x: -x[1]))}
    kronq_rank = {tid: i + 1 for i, (tid, _) in
                  enumerate(sorted(kronq_sensitivities.items(), key=lambda x: -x[1]))}

    for tid in sorted(distortion_curves.keys()):
        hwe = measured[tid]
        ks = kronq_sensitivities[tid]
        tg, tx, ratio = hessian_traces[tid]
        rm = measured_rank[tid]
        rk = kronq_rank[tid]
        print(f"{tid:<15} {hwe:>12.4e} {ks:>12.4e} {tg:>12.4e} {tx:>12.4e} "
              f"{ratio:>8.2f} {rm:>10} {rk:>10}")

    # Rank correlation (Spearman)
    measured_ranks = np.array([measured_rank[tid] for tid in sorted(distortion_curves.keys())])
    kronq_ranks = np.array([kronq_rank[tid] for tid in sorted(distortion_curves.keys())])
    n = len(measured_ranks)
    spearman = 1 - 6 * np.sum((measured_ranks - kronq_ranks) ** 2) / (n * (n ** 2 - 1))
    print(f"\nSpearman rank correlation: {spearman:.4f}")

    # Pearson correlation in log space
    mv = np.array([measured[tid] for tid in sorted(distortion_curves.keys())])
    kv = np.array([kronq_sensitivities[tid] for tid in sorted(distortion_curves.keys())])
    pearson_log = np.corrcoef(np.log(mv + 1e-30), np.log(kv + 1e-30))[0, 1]
    print(f"Pearson correlation (log): {pearson_log:.4f}")

    # Simpler baselines: tr(H_G) alone, tr(H_X) alone, weight MSE alone
    tr_hg = np.array([hessian_traces[tid][0] for tid in sorted(distortion_curves.keys())])
    tr_hx = np.array([hessian_traces[tid][1] for tid in sorted(distortion_curves.keys())])
    weight_mse_k5 = np.array([distortion_curves[tid][5]['mse'] for tid in sorted(distortion_curves.keys())])
    log_mv = np.log(mv + 1e-30)

    spearman_trhg = np.corrcoef(np.argsort(np.argsort(-tr_hg)), np.argsort(np.argsort(-mv)))[0, 1] if len(mv) > 2 else 0
    pearson_trhg = np.corrcoef(np.log(tr_hg + 1e-30), log_mv)[0, 1]
    spearman_trhx = np.corrcoef(np.argsort(np.argsort(-tr_hx)), np.argsort(np.argsort(-mv)))[0, 1] if len(mv) > 2 else 0
    pearson_trhx = np.corrcoef(np.log(tr_hx + 1e-30), log_mv)[0, 1]
    pearson_wmse = np.corrcoef(np.log(weight_mse_k5 + 1e-30), log_mv)[0, 1]

    # Also: quantization MSE × joint-trace (product of cheap proxy and quant error)
    mse_x_jt = weight_mse_k5 * kv
    pearson_mse_x_jt = np.corrcoef(np.log(mse_x_jt + 1e-30), log_mv)[0, 1]

    print(f"\nSimpler baseline correlations (log-Pearson with HWE@K5):")
    print(f"  tr(H_G) alone:         r={pearson_trhg:.4f}")
    print(f"  tr(H_X) alone:         r={pearson_trhx:.4f}")
    print(f"  Weight MSE@K5:         r={pearson_wmse:.4f}")
    print(f"  KronQ tr(H_G)*tr(H_X): r={pearson_log:.4f}")
    print(f"  MSE@K5 × KronQ:        r={pearson_mse_x_jt:.4f}")
    print(f"\n  Note: tr(H_G) alone may outperform joint-trace as predictor.")

    # KronQ as allocation predictor: allocate K proportional to sensitivity
    # Compare: KronQ-guided allocation vs measured-guided allocation vs DP
    print("\nKronQ-guided allocation vs DP:")
    all_items = []
    for tid in sorted(distortion_curves.keys()):
        all_items.append({
            'id': tid,
            'distortion': {k: distortion_curves[tid][k]['hwe'] for k in K_VALUES},
            'bytes': {k: distortion_curves[tid][k]['bytes'] for k in K_VALUES},
        })

    budget_k5 = sum(slice_bytes(5) for _ in all_items)

    # KronQ-guided: assign K based on KronQ sensitivity rank
    # Top 1/3 get K7, middle 1/3 get K5, bottom 1/3 get K3
    kronq_sorted = sorted(kronq_sensitivities.items(), key=lambda x: -x[1])
    n_items = len(kronq_sorted)
    kronq_assign = {}
    for i, (tid, _) in enumerate(kronq_sorted):
        if i < n_items // 3:
            kronq_assign[tid] = 7
        elif i < 2 * n_items // 3:
            kronq_assign[tid] = 5
        else:
            kronq_assign[tid] = 3

    # Adjust to fit budget: if over budget, reduce highest-K items
    kronq_bytes = sum(slice_bytes(kronq_assign[tid]) for tid in kronq_assign)
    while kronq_bytes > budget_k5:
        # Find highest-K item and reduce
        max_k_tid = max(kronq_assign, key=lambda t: kronq_assign[t])
        if kronq_assign[max_k_tid] > K_MIN:
            kronq_assign[max_k_tid] -= 1
            kronq_bytes = sum(slice_bytes(kronq_assign[tid]) for tid in kronq_assign)
        else:
            break

    kronq_d = sum(distortion_curves[tid][kronq_assign[tid]]['hwe'] for tid in distortion_curves)

    # DP
    dp_assign, dp_d, _ = interlayer_dp(all_items, budget_k5)

    # Uniform K5
    uniform_d = sum(distortion_curves[tid][5]['hwe'] for tid in distortion_curves)

    print(f"  Uniform K5: D={uniform_d:.4e}")
    print(f"  KronQ-guided: D={kronq_d:.4e}, budget={kronq_bytes:.0f}")
    print(f"  DP: D={dp_d:.4e}")
    print(f"  KronQ-guided vs uniform: {(uniform_d - kronq_d) / uniform_d * 100:+.1f}%")
    print(f"  KronQ-guided vs DP: {(dp_d - kronq_d) / dp_d * 100:+.1f}% (positive = KronQ better)")
    print(f"  DP vs uniform: {(uniform_d - dp_d) / uniform_d * 100:+.1f}%")

    return {
        'spearman': spearman,
        'pearson_log': pearson_log,
        'kronq_assignment': kronq_assign,
        'kronq_distortion': kronq_d,
        'dp_distortion': dp_d,
        'uniform_distortion': uniform_d,
    }

# ============================================================================
# Full model extrapolation (56 layers)
# ============================================================================

def experiment_full_model(distortion_curves, kronq_sensitivities, real_stats):
    """Extrapolate to full 56-layer model."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: Full 56-layer model extrapolation")
    print("=" * 80)

    # Build distortion curves for all 56 layers × 4 roles
    # For layers we don't have real data, use synthetic with interpolated stats
    all_curves = {}
    all_kronq = {}

    for layer in range(N_LAYERS):
        for role in ROLES:
            tid = f"L{layer}_{role}"

            if f"L{layer}_{role}" in distortion_curves:
                # Use measured data
                all_curves[tid] = distortion_curves[f"L{layer}_{role}"]
                all_kronq[tid] = kronq_sensitivities[f"L{layer}_{role}"]
            else:
                # Interpolate from sample layers
                # Find bracketing sample layers
                lo = max(l for l in SAMPLE_LAYERS if l <= layer)
                hi = min(l for l in SAMPLE_LAYERS if l >= layer)
                if lo == hi:
                    t = 0.0
                else:
                    t = (layer - lo) / (hi - lo)

                # Interpolate distortion curves
                lo_tid = f"L{lo}_{role}"
                hi_tid = f"L{hi}_{role}"
                if lo_tid in distortion_curves and hi_tid in distortion_curves:
                    interp_curve = {}
                    for k in K_VALUES:
                        d_lo = distortion_curves[lo_tid][k]['hwe']
                        d_hi = distortion_curves[hi_tid][k]['hwe']
                        # Log-linear interpolation (distortion scales geometrically)
                        if d_lo > 0 and d_hi > 0:
                            d_interp = d_lo * (d_hi / d_lo) ** t
                        else:
                            d_interp = d_lo * (1 - t) + d_hi * t
                        interp_curve[k] = {
                            'hwe': d_interp,
                            'mse': 0.0,
                            'bytes': slice_bytes(k),
                        }
                    all_curves[tid] = interp_curve
                    ks_lo = kronq_sensitivities.get(lo_tid, 0)
                    ks_hi = kronq_sensitivities.get(hi_tid, 0)
                    all_kronq[tid] = ks_lo * (ks_hi / ks_lo) ** t if ks_lo > 0 and ks_hi > 0 else (ks_lo + ks_hi) / 2

    # Full model DP at K5K6 budget
    all_items = []
    for tid in sorted(all_curves.keys()):
        all_items.append({
            'id': tid,
            'distortion': {k: all_curves[tid][k]['hwe'] for k in K_VALUES},
            'bytes': {k: all_curves[tid][k]['bytes'] for k in K_VALUES},
        })

    # K5K6 budget for full model
    k5k6_budget = sum(slice_bytes(5 if item['id'].split('_')[1] == 'gate' else 6)
                      for item in all_items)

    dp_assign, dp_d, dp_b = interlayer_dp(all_items, k5k6_budget)
    k5k6_d = sum(all_curves[tid][5 if tid.split('_')[1] == 'gate' else 6]['hwe']
                for tid in all_curves)
    uniform_k5_d = sum(all_curves[tid][5]['hwe'] for tid in all_curves)
    uniform_k6_d = sum(all_curves[tid][6]['hwe'] for tid in all_curves)

    print(f"\nFull model (56 layers × 4 roles = {len(all_items)} tensors):")
    print(f"  K5K6 budget: {k5k6_budget:.0f} bytes (per-slice)")
    print(f"  Uniform K5: D={uniform_k5_d:.4e}")
    print(f"  Uniform K6: D={uniform_k6_d:.4e}")
    print(f"  K5K6 recipe: D={k5k6_d:.4e}")
    print(f"  DP optimal: D={dp_d:.4e}")
    print(f"  DP vs K5K6: {(k5k6_d - dp_d) / k5k6_d * 100:+.1f}%")
    print(f"  DP vs uniform K5: {(uniform_k5_d - dp_d) / uniform_k5_d * 100:+.1f}%")

    # Per-layer K distribution
    print(f"\nDP K distribution (by role):")
    for role in sorted(ROLES.keys()):
        k_counts = {k: 0 for k in K_VALUES}
        for tid, k_val in dp_assign.items():
            if tid.split('_')[1] == role:
                k_counts[k_val] += 1
        print(f"  {role}: " + ", ".join(f"K{k}:{c}" for k, c in k_counts.items() if c > 0))

    # Per-layer average K
    print(f"\nDP average K per layer (first 10, last 10):")
    for layer in list(range(10)) + list(range(46, 56)):
        layer_items = {tid: dp_assign[tid] for tid in dp_assign if tid.startswith(f"L{layer}_")}
        avg_k = np.mean(list(layer_items.values()))
        print(f"  L{layer}: avg K={avg_k:.2f}, " +
              ", ".join(f"{tid.split('_')[1]}:{k}" for tid, k in sorted(layer_items.items())))

    return {
        'n_tensors': len(all_items),
        'k5k6_distortion': k5k6_d,
        'dp_distortion': dp_d,
        'uniform_k5_distortion': uniform_k5_d,
        'uniform_k6_distortion': uniform_k6_d,
        'dp_assignment': dp_assign,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    print("R19-InterlayerAlloc: Inter-layer optimal K allocation")
    print("=" * 80)
    print(f"Config: {len(SAMPLE_LAYERS)} sample layers × {len(ROLES)} roles = "
          f"{len(SAMPLE_LAYERS) * len(ROLES)} tensors")
    print(f"K values: {K_VALUES}")
    print(f"Tile size: {TILE}×{TILE}, Slice: {M_DIM}×{N_DIM}")

    # Load real weights
    real_weights = load_real_weights()
    print(f"\nLoaded {len(real_weights)} real weight tensors")

    # Run experiments
    distortion_curves, kronq_sensitivities, hessian_traces, tensor_registry, real_stats = \
        experiment_per_layer_curves(real_weights)

    dp_results = experiment_interlayer_dp(distortion_curves, tensor_registry)

    ranked, measured_sens = experiment_sensitivity_ranking(distortion_curves, kronq_sensitivities)

    role_results = experiment_role_dependent(distortion_curves)

    rotation_results = experiment_rotation_interaction(tensor_registry, distortion_curves)

    frontier, k5k6_assign, k5k6_d, k4k5_assign, k4k5_d = \
        experiment_budget_frontier(distortion_curves)

    kronq_results = experiment_kronq_comparison(distortion_curves, kronq_sensitivities,
                                                 hessian_traces)

    full_model_results = experiment_full_model(distortion_curves, kronq_sensitivities,
                                                 real_stats)

    # Save results
    results = {
        'config': {
            'sample_layers': SAMPLE_LAYERS,
            'roles': list(ROLES.keys()),
            'k_values': K_VALUES,
            'tile_size': TILE,
            'slice_dim': [M_DIM, N_DIM],
            'n_layers_model': N_LAYERS,
            'hidden_dim': HIDDEN_DIM,
            'inter_dim': INTER_DIM,
        },
        'distortion_curves': {
            tid: {str(k): v for k, v in curve.items()}
            for tid, curve in distortion_curves.items()
        },
        'kronq_sensitivities': kronq_sensitivities,
        'hessian_traces': {tid: list(v) for tid, v in hessian_traces.items()},
        'sensitivity_ranking': [(tid, float(s)) for tid, s in ranked],
        'role_allocation': {
            k: v if not isinstance(v, dict) else
               {k2: (v2 if not isinstance(v2, dict) else
                     {k3: v3 for k3, v3 in v2.items()})
                for k2, v2 in v.items()}
            for k, v in role_results.items()
        },
        'rotation_interaction': {
            'cv_unrotated': float(rotation_results['cv_unrotated']),
            'cv_rotated': float(rotation_results['cv_rotated']),
            'assignment_diffs': rotation_results['assignment_diffs'],
        },
        'kronq_comparison': {
            'spearman': float(kronq_results['spearman']),
            'pearson_log': float(kronq_results['pearson_log']),
        },
        'full_model': {
            'n_tensors': full_model_results['n_tensors'],
            'k5k6_distortion': float(full_model_results['k5k6_distortion']),
            'dp_distortion': float(full_model_results['dp_distortion']),
            'uniform_k5_distortion': float(full_model_results['uniform_k5_distortion']),
            'uniform_k6_distortion': float(full_model_results['uniform_k6_distortion']),
        },
        'elapsed_seconds': time.time() - t_start,
    }

    # Serialize assignments
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results = make_serializable(results)

    output_path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                               'receipts', 'research', 'r19-interlayer-alloc-results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")

    return results


if __name__ == "__main__":
    main()
