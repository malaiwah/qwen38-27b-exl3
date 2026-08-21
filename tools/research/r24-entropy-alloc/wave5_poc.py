#!/usr/bin/env python3
"""
R24 Wave 5 — Entropy-Constrained Allocation (Scaled)

Wave 5 scaling requirements:
- 9 depths: layers 0, 7, 14, 21, 28, 35, 42, 49, 55
- Roles: MLP gate, up, down; GDN qkv, z, out
- 8+ blocks per tensor (diagonal + random + off-diagonal)
- Separate calibration/validation/test splits
- Hadamard as rotation action (per R26/R27: BiIP scaling marginal over EXL3)
- Per-role, per-depth statistics
- Correct BF16 decode
- Full tensors where feasible
"""

import json
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_wave5_weights.npz"
WORKDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(WORKDIR, "wave5_results.json")
FINDINGS_PATH = os.path.join(WORKDIR, "wave5_findings.md")

# ─── Configuration ────────────────────────────────────────────────────────────
TILE       = 16
SLICE      = 128
K_VALUES   = [3, 4, 5, 6]
K_MIN      = min(K_VALUES)
K_MAX      = max(K_VALUES)
N_CALIB    = 512
RNG_SEED   = 42
ELEM_PER_T = TILE * TILE  # 256

# Wave 5: 9 depths, multiple roles
TARGET_LAYERS = [0, 7, 14, 21, 28, 35, 42, 49, 55]
ROLES = ['gate', 'up', 'down', 'qkv', 'z', 'out']

# Block sampling: 8 diagonal + 8 random + 4 off-diagonal = 20 blocks per tensor
N_DIAGONAL = 8
N_RANDOM = 8
N_OFFDIAG = 4
N_BLOCKS = N_DIAGONAL + N_RANDOM + N_OFFDIAG  # 20

# Calibration/validation/test split (by block index)
# First 8 blocks = calibration, next 8 = validation, last 4 = test
CALIB_BLOCKS = list(range(8))
VAL_BLOCKS = list(range(8, 16))
TEST_BLOCKS = list(range(16, 20))

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_wave5_weights():
    data = np.load(WEIGHTS_PATH)
    return {k: data[k].astype(np.float64) for k in data.files}


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


def extract_blocks(tensor, n_blocks=20, seed=42):
    """Extract 20 blocks from a full tensor: 8 diagonal, 8 random, 4 off-diagonal.
    Each block is 128x128. Returns list of (block_name, block_data, block_type)."""
    M, N = tensor.shape
    rng = np.random.default_rng(seed)
    blocks = []

    # Diagonal blocks (along main diagonal)
    n_diag = min(N_DIAGONAL, min(M, N) // SLICE)
    for i in range(n_diag):
        r = i * SLICE
        c = i * SLICE
        if r + SLICE <= M and c + SLICE <= N:
            blocks.append((f'diag{i}', tensor[r:r+SLICE, c:c+SLICE].copy(), 'diagonal'))

    # Random blocks (random positions, may be off-diagonal)
    n_rand = N_RANDOM
    for i in range(n_rand):
        r = int(rng.integers(0, max(M - SLICE, 1)))
        c = int(rng.integers(0, max(N - SLICE, 1)))
        blocks.append((f'rand{i}', tensor[r:r+SLICE, c:c+SLICE].copy(), 'random'))

    n_off = N_OFFDIAG
    for i in range(n_off):
        # Pick row block != col block
        r_block = int(rng.integers(0, max(M // SLICE, 1)))
        c_block = int(rng.integers(0, max(N // SLICE, 1)))
        while c_block == r_block and M // SLICE > 1:
            c_block = int(rng.integers(0, max(N // SLICE, 1)))
        r = r_block * SLICE
        c = c_block * SLICE
        if r + SLICE <= M and c + SLICE <= N:
            blocks.append((f'off{i}', tensor[r:r+SLICE, c:c+SLICE].copy(), 'offdiag'))

    return blocks


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
# Entropy
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_entropy(indices):
    if len(indices) == 0:
        return 0.0
    counts = np.bincount(indices)
    probs = counts[counts > 0] / len(indices)
    return float(-np.sum(probs * np.log2(probs)))


def tile_entropy(w_tile, k):
    indices = quantize_tile_indices(w_tile, k)
    return empirical_entropy(indices.flatten())


# ═══════════════════════════════════════════════════════════════════════════════
# Hadamard rotation (per R26/R27: use Hadamard, not BiIP)
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


def apply_hadamard(W, rng):
    U, _ = signed_random_hadamard(W.shape[0], rng)
    V, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U @ W @ V
    return W_t, (U, V)


def inverse_hadamard(Wq_t, U, V):
    return U.T @ Wq_t @ V.T


# ═══════════════════════════════════════════════════════════════════════════════
# Tile tables
# ═══════════════════════════════════════════════════════════════════════════════

def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    Q_tile = quantize_tile(W_tile, k)
    E_t = W_tile - Q_tile
    D = np.trace(H_G_sub @ E_t @ H_X_sub @ E_t.T)
    return max(D, 0.0)


def precompute_tile_tables(W, H_X, H_G, tile=TILE, k_values=K_VALUES):
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
# Fixed-K DP and entropy Lagrangian
# ═══════════════════════════════════════════════════════════════════════════════

def alloc_fixed_k_dp(D_table, budget_k, k_values=K_VALUES):
    n_tiles = D_table.shape[0]
    INF = float('inf')
    dp = np.full(budget_k + 1, INF)
    dp[0] = 0.0
    choices = []
    for t in range(n_tiles):
        new_dp = np.full(budget_k + 1, INF)
        new_choice = np.full(budget_k + 1, -1, dtype=int)
        for ki, k_val in enumerate(k_values):
            valid = dp < INF
            if not np.any(valid):
                break
            src_j = np.where(valid)[0]
            dst_j = src_j + k_val
            mask = dst_j <= budget_k
            if not np.any(mask):
                continue
            src_j, dst_j = src_j[mask], dst_j[mask]
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


def entropy_lagrangian_alloc(D_table, H_table, lam, k_values=K_VALUES):
    n_tiles = D_table.shape[0]
    K_flat = np.zeros(n_tiles, dtype=int)
    for t in range(n_tiles):
        costs = D_table[t, :] + lam * H_table[t, :] * ELEM_PER_T
        K_flat[t] = k_values[int(np.argmin(costs))]
    return K_flat


def fixed_k_lagrangian_alloc(D_table, lam, k_values=K_VALUES):
    n_tiles = D_table.shape[0]
    K_flat = np.zeros(n_tiles, dtype=int)
    k_arr = np.array(k_values, dtype=float)
    for t in range(n_tiles):
        costs = D_table[t, :] + lam * k_arr * ELEM_PER_T
        K_flat[t] = k_values[int(np.argmin(costs))]
    return K_flat


def find_lambda_for_budget(D_table, H_table, target_bits, k_values=K_VALUES,
                           n_iters=50):
    lam_lo, lam_hi = 1e-12, 1e6
    for _ in range(n_iters):
        lam_mid = np.sqrt(lam_lo * lam_hi)
        K_flat = entropy_lagrangian_alloc(D_table, H_table, lam_mid, k_values)
        cost = float(np.sum([H_table[t, k_values.index(K_flat[t])] * ELEM_PER_T
                             for t in range(len(K_flat))]))
        if cost > target_bits:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    K_flat = entropy_lagrangian_alloc(D_table, H_table, lam_hi, k_values)
    return K_flat, lam_hi


def eval_hwe(W, K_flat, H_G, H_X, tile=TILE):
    ntr, ntc = W.shape[0] // tile, W.shape[1] // tile
    K_grid = K_flat.reshape(ntr, ntc)
    Wq = quantize_tiles(W, K_grid, tile)
    E = W - Wq
    return hessian_weighted_error(E, H_G, H_X)


def eval_entropy(H_table, K_flat, k_values=K_VALUES):
    return float(np.sum([H_table[t, k_values.index(K_flat[t])] * ELEM_PER_T
                         for t in range(len(K_flat))]))


# ═══════════════════════════════════════════════════════════════════════════════
# Main Wave 5 experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_wave5():
    t_start = time.time()
    print("=" * 80)
    print("R24 Wave 5 — Entropy-Constrained Allocation (Scaled)")
    print("=" * 80)

    tensors = load_wave5_weights()
    rng = np.random.default_rng(RNG_SEED)

    all_results = {
        'config': {
            'tile': TILE, 'slice': SLICE, 'K_values': K_VALUES,
            'target_layers': TARGET_LAYERS, 'roles': ROLES,
            'n_blocks': N_BLOCKS, 'calib_blocks': CALIB_BLOCKS,
            'val_blocks': VAL_BLOCKS, 'test_blocks': TEST_BLOCKS,
        },
        'tensors': {},
    }

    # Per-role and per-depth aggregation
    role_stats = {role: {'entropy_dp_gains': [], 'hadamard_gains': [],
                         'entropy_savings': {k: [] for k in K_VALUES}}
                  for role in ROLES}
    depth_stats = {layer: {'entropy_dp_gains': [], 'hadamard_gains': []}
                   for layer in TARGET_LAYERS}

    for tname in sorted(tensors.keys()):
        tensor = tensors[tname]
        # Parse layer and role from name
        parts = tname.split('_')
        layer = int(parts[0][1:])
        role = '_'.join(parts[1:])

        if layer not in TARGET_LAYERS or role not in ROLES:
            continue

        print(f"\n  {tname} (shape {tensor.shape})")

        # Extract 20 blocks
        blocks = extract_blocks(tensor, N_BLOCKS, seed=RNG_SEED + layer * 100)
        if not blocks:
            print(f"    SKIP: no valid blocks")
            continue

        # Use calibration blocks for allocation, validation for reporting
        X = gen_calibration(SLICE, N_CALIB, RNG_SEED + layer)

        tensor_results = {'blocks': {}, 'layer': layer, 'role': role}

        for bname, W, btype in blocks:
            H_G, H_X = compute_hessians(W, X)
            n_tiles = (SLICE // TILE) ** 2

            # Precompute tables
            D_table, H_table = precompute_tile_tables(W, H_X, H_G)

            # Entropy estimation
            ent_savings = {}
            for ki, k in enumerate(K_VALUES):
                mean_h = float(np.mean(H_table[:, ki]))
                savings = (k - mean_h) / k * 100
                ent_savings[k] = savings

            # Fixed-K DP (pure payload budget)
            for avg_k in [4, 5]:
                budget_k = n_tiles * avg_k
                K_fixed, _ = alloc_fixed_k_dp(D_table, budget_k)
                HWE_fixed = eval_hwe(W, K_fixed, H_G, H_X)

                # Entropy DP (Lagrangian, no model overhead = lower bound)
                target_bits = avg_k * n_tiles * ELEM_PER_T
                K_ent, _ = find_lambda_for_budget(D_table, H_table, target_bits)
                HWE_ent = eval_hwe(W, K_ent, H_G, H_X)

                # Entropy DP improvement
                gain = (HWE_fixed - HWE_ent) / HWE_fixed * 100 if HWE_fixed > 1e-15 else 0

                # Hadamard rotation
                W_had, (U, V) = apply_hadamard(W, rng)
                # Use transformed Hessians
                H_G_had = U @ H_G @ U.T
                H_X_had = V.T @ H_X @ V
                D_had, H_had = precompute_tile_tables(W_had, H_X_had, H_G_had)

                K_had, _ = alloc_fixed_k_dp(D_had, budget_k)
                Wq_had = quantize_tiles(W_had, K_had.reshape(SLICE//TILE, SLICE//TILE), TILE)
                W_hat_had = inverse_hadamard(Wq_had, U, V)
                HWE_had = hessian_weighted_error(W - W_hat_had, H_G, H_X)

                had_gain = (HWE_fixed - HWE_had) / HWE_fixed * 100 if HWE_fixed > 1e-15 else 0

                bkey = f'{bname}_K{avg_k}'
                tensor_results['blocks'][bkey] = {
                    'block_type': btype,
                    'HWE_fixed_k': HWE_fixed,
                    'HWE_entropy': HWE_ent,
                    'HWE_hadamard': HWE_had,
                    'entropy_gain_pct': gain,
                    'hadamard_gain_pct': had_gain,
                    'k_mean_fixed': float(np.mean(K_fixed)),
                    'k_mean_entropy': float(np.mean(K_ent)),
                }

                # Only collect from validation blocks for promoted statistics
                block_idx = int(bname.replace('diag', '').replace('rand', '').replace('off', ''))
                is_calib = btype == 'diagonal' and block_idx < N_DIAGONAL
                is_val = not is_calib  # non-diagonal + later diagonal = validation/test

                if is_val:
                    role_stats[role]['entropy_dp_gains'].append(gain)
                    role_stats[role]['hadamard_gains'].append(had_gain)
                    depth_stats[layer]['entropy_dp_gains'].append(gain)
                    depth_stats[layer]['hadamard_gains'].append(had_gain)

            for k in K_VALUES:
                role_stats[role]['entropy_savings'][k].append(ent_savings[k])

        all_results['tensors'][tname] = tensor_results
        print(f"    {len(blocks)} blocks processed")

    # ─── Summary ───
    print(f"\n{'═' * 80}")
    print(f"  WAVE 5 SUMMARY")
    print(f"{'═' * 80}")

    # Per-role
    print("\n  ── Per-role statistics (validation blocks only) ──")
    for role in ROLES:
        ent_gains = role_stats[role]['entropy_dp_gains']
        had_gains = role_stats[role]['hadamard_gains']
        if ent_gains:
            print(f"    {role}: entropy DP gain={np.mean(ent_gains):+.1f}% ± {np.std(ent_gains):.1f}% "
                  f"(n={len(ent_gains)}), Hadamard gain={np.mean(had_gains):+.1f}% ± {np.std(had_gains):.1f}%")
        for k in K_VALUES:
            savings = role_stats[role]['entropy_savings'][k]
            if savings:
                pass  # print later in aggregate

    # Per-depth
    print("\n  ── Per-depth statistics (validation blocks only) ──")
    for layer in TARGET_LAYERS:
        ent_gains = depth_stats[layer]['entropy_dp_gains']
        had_gains = depth_stats[layer]['hadamard_gains']
        if ent_gains:
            print(f"    Layer {layer:2d}: entropy DP gain={np.mean(ent_gains):+.1f}% ± {np.std(ent_gains):.1f}% "
                  f"(n={len(ent_gains)}), Hadamard gain={np.mean(had_gains):+.1f}% ± {np.std(had_gains):.1f}%")

    # Entropy savings by K
    print("\n  ── Entropy savings by K (all blocks) ──")
    for k in K_VALUES:
        all_savings = []
        for role in ROLES:
            all_savings.extend(role_stats[role]['entropy_savings'][k])
        if all_savings:
            print(f"    K={k}: {np.mean(all_savings):.1f}% ± {np.std(all_savings):.1f}% (n={len(all_savings)})")

    # Macro
    all_ent = [g for role in ROLES for g in role_stats[role]['entropy_dp_gains']]
    all_had = [g for role in ROLES for g in role_stats[role]['hadamard_gains']]
    if all_ent:
        print(f"\n  MACRO: entropy DP gain={np.mean(all_ent):+.1f}% ± {np.std(all_ent):.1f}% (n={len(all_ent)})")
        print(f"  MACRO: Hadamard gain={np.mean(all_had):+.1f}% ± {np.std(all_had):.1f}% (n={len(all_had)})")

    all_results['summary'] = {
        'role_stats': {role: {
            'entropy_dp_gain_mean': float(np.mean(role_stats[role]['entropy_dp_gains'])) if role_stats[role]['entropy_dp_gains'] else 0,
            'entropy_dp_gain_std': float(np.std(role_stats[role]['entropy_dp_gains'])) if role_stats[role]['entropy_dp_gains'] else 0,
            'hadamard_gain_mean': float(np.mean(role_stats[role]['hadamard_gains'])) if role_stats[role]['hadamard_gains'] else 0,
            'hadamard_gain_std': float(np.std(role_stats[role]['hadamard_gains'])) if role_stats[role]['hadamard_gains'] else 0,
            'n_samples': len(role_stats[role]['entropy_dp_gains']),
        } for role in ROLES},
        'depth_stats': {str(layer): {
            'entropy_dp_gain_mean': float(np.mean(depth_stats[layer]['entropy_dp_gains'])) if depth_stats[layer]['entropy_dp_gains'] else 0,
            'hadamard_gain_mean': float(np.mean(depth_stats[layer]['hadamard_gains'])) if depth_stats[layer]['hadamard_gains'] else 0,
            'n_samples': len(depth_stats[layer]['entropy_dp_gains']),
        } for layer in TARGET_LAYERS},
        'macro': {
            'entropy_dp_gain': float(np.mean(all_ent)) if all_ent else 0,
            'entropy_dp_gain_std': float(np.std(all_ent)) if all_ent else 0,
            'hadamard_gain': float(np.mean(all_had)) if all_had else 0,
            'hadamard_gain_std': float(np.std(all_had)) if all_had else 0,
            'n_total': len(all_ent),
        }
    }

    elapsed = time.time() - t_start
    all_results['elapsed_seconds'] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {RESULTS_PATH}")

    return all_results


if __name__ == "__main__":
    run_wave5()
