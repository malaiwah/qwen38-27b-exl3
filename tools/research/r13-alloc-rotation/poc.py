#!/usr/bin/env python3
"""
R13-AllocRotation: Allocation + rotation composition.

Tests whether DP-refined tile allocation (R1) composes with BiIP+Hadamard
rotation (R3). The cross-review (doc 63 section 5 insight 6) identified this as
an open question: R1's DP allocation was tested on unrotated weights only
(+25.5%), and R9's surrogate allocation was rejected after rotation (rotation
homogenizes tile sensitivity). This experiment uses the FULL DP-refined
allocation (not surrogate) to test all 6 composition orders.

Arms:
1. neither:        uniform K, no rotation (baseline)
2. alloc_only:     DP-refined allocation, no rotation (R1 baseline)
3. rotate_only:    uniform K, with BiIP+Hadamard rotation (R3 baseline)
4. alloc_then_rot: DP allocation on unrotated, then rotate, quantize with SAME K_grid
5. rot_then_alloc: rotate, then DP allocation on rotated weights, quantize
6. alternating:    allocate, rotate, re-allocate (warm-start local search)

Primary metric: tr(H_G E H_X E^T) where E = W - W_hat, in ORIGINAL space.
H_G is output-covariance proxy (Y^T Y / N), not true gradient covariance.
"""

import json
import numpy as np
import os
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

TILE = 16
M_DIM = 128
N_DIM = 128
K_VALUES = [3, 4, 5, 6]
K_MIN = min(K_VALUES)
K_MAX = max(K_VALUES)
P_CAL = 512
N_TILES = (M_DIM // TILE) * (N_DIM // TILE)
ELEMENTS_PER_TILE = TILE * TILE
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r13-alloc-rotation/receipts/research/r13-alloc-rotation-results.json"


def quantize_tile(w, k):
    if k <= 0:
        return np.zeros_like(w)
    nl = 2 ** k
    lo, hi = w.min(), w.max()
    if hi - lo < 1e-12:
        return w.copy()
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_matrix(W, K_grid, tile=TILE):
    M, N = W.shape
    Wq = np.zeros_like(W)
    for ti in range(M // tile):
        for tj in range(N // tile):
            k = int(K_grid[ti, tj])
            r0, c0 = ti * tile, tj * tile
            Wq[r0:r0+tile, c0:c0+tile] = quantize_tile(W[r0:r0+tile, c0:c0+tile], k)
    return Wq


def compute_bytes(K_flat, n_tiles=N_TILES, elements_per_tile=ELEMENTS_PER_TILE,
                  rotation_sidecar=0):
    total_k = int(np.sum(K_flat))
    payload_bytes = (total_k * elements_per_tile) // 8
    sidecar_bytes = n_tiles * 4
    K_list = K_flat.tolist() if hasattr(K_flat, 'tolist') else list(K_flat)
    if len(set(K_list)) == 1:
        metadata_bytes = 1
    else:
        metadata_bytes = (n_tiles * 3 + 7) // 8
    return payload_bytes + sidecar_bytes + metadata_bytes + rotation_sidecar


def budget_k_sum_payload_matched(avg_k, n_tiles=N_TILES):
    """K-sum budget: exactly n_tiles * avg_k for ALL arms (true payload match).
    Mixed-K metadata is charged against the tile sidecar, not the K budget."""
    return n_tiles * avg_k


def synthetic_hessians(W, n_samples=512, seed=42):
    rng = np.random.default_rng(seed)
    d_out, d_in = W.shape
    X = rng.standard_normal((d_in, n_samples))
    n_outliers = max(1, int(d_in * 0.05))
    outlier_channels = rng.choice(d_in, n_outliers, replace=False)
    X[outlier_channels, :] *= 10.0
    H_X = (X @ X.T / n_samples).astype(np.float64)
    Y = W @ X
    H_G = (Y @ Y.T / n_samples).astype(np.float64)
    H_X *= d_in / np.trace(H_X)
    H_G *= d_out / np.trace(H_G)
    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)
    return H_X, H_G


def biip_scaling(W, H_X, H_G):
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_G = np.diag(sg_diag)
    W_transformed = S_G @ W @ S_X
    sidecar_bytes = (d_in + d_out) * 4
    return S_G, S_X, W_transformed, sidecar_bytes


def hadamard_matrix(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n)
    return H @ np.diag(signs), signs


def compute_rotation_sidecar(d_in, d_out):
    return (d_in + d_out) * 4 + (d_in + 7) // 8 + (d_out + 7) // 8


def apply_rotation(W, H_X, H_G, rng):
    d_out, d_in = W.shape
    sidecar = 0
    S_G, S_X, W_t, sc_bytes = biip_scaling(W, H_X, H_G)
    sidecar += sc_bytes
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = S_X_inv @ H_X @ S_X_inv
    H_G_t = S_G_inv @ H_G @ S_G_inv
    V, _ = signed_random_hadamard(d_in, rng)
    sidecar += (d_in + 7) // 8
    W_t = W_t @ V.T
    H_X_t = V @ H_X_t @ V.T
    U, _ = signed_random_hadamard(d_out, rng)
    sidecar += (d_out + 7) // 8
    W_t = U @ W_t
    H_G_t = U @ H_G_t @ U.T
    return W_t, H_X_t, H_G_t, U, V, S_G, S_X, sidecar


def inverse_rotation(W_q, U, V, S_G, S_X):
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ W_q @ V @ S_X_inv


def measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub):
    Q_tile = quantize_tile(W_tile, k)
    E_t = W_tile - Q_tile
    D = np.trace(H_G_sub @ E_t @ H_X_sub @ E_t.T)
    return max(D, 0.0)


def tile_sensitivities(W, H_X, H_G, k_ref=5, tile=TILE):
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    sens = np.zeros(ntr * ntc)
    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            sens[t_idx] = measure_tile_distortion(
                W[r0:r0+tile, c0:c0+tile], k_ref,
                H_G[r0:r0+tile, r0:r0+tile], H_X[c0:c0+tile, c0:c0+tile])
            t_idx += 1
    return sens


def alloc_tile_local_dp(W, H_X, H_G, budget_k, tile=TILE, k_values=None):
    if k_values is None:
        k_values = K_VALUES
    M, N = W.shape
    ntr, ntc = M // tile, N // tile
    n_tiles = ntr * ntc
    D_table = np.zeros((n_tiles, len(k_values)))
    t_idx = 0
    for ti in range(ntr):
        for tj in range(ntc):
            r0, c0 = ti * tile, tj * tile
            W_tile = W[r0:r0+tile, c0:c0+tile]
            H_G_sub = H_G[r0:r0+tile, r0:r0+tile]
            H_X_sub = H_X[c0:c0+tile, c0:c0+tile]
            for ki, k in enumerate(k_values):
                D_table[t_idx, ki] = measure_tile_distortion(W_tile, k, H_G_sub, H_X_sub)
            t_idx += 1

    INF = float('inf')
    dp = [INF] * (budget_k + 1)
    dp[0] = 0.0
    choices = []
    for t in range(n_tiles):
        new_dp = [INF] * (budget_k + 1)
        new_choice = [-1] * (budget_k + 1)
        for j in range(budget_k + 1):
            if dp[j] == INF:
                continue
            for ki in range(len(k_values)):
                nj = j + k_values[ki]
                if nj > budget_k:
                    continue
                val = dp[j] + D_table[t, ki]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    new_choice[nj] = ki
        dp = new_dp
        choices.append(new_choice[:])

    best_j = 0
    best_d = INF
    for j in range(budget_k + 1):
        if dp[j] < best_d:
            best_d = dp[j]
            best_j = j

    K_flat = np.zeros(n_tiles, dtype=int)
    j = best_j
    for t in range(n_tiles - 1, -1, -1):
        if j < 0 or j >= len(choices[t]) or choices[t][j] < 0:
            for fallback_ki in range(len(k_values)):
                if k_values[fallback_ki] <= max(j, 0):
                    ki = fallback_ki
                    break
            else:
                ki = 0
        else:
            ki = choices[t][j]
        K_flat[t] = k_values[ki]
        j -= k_values[ki]

    return K_flat, D_table


def local_search_refine(W, K_flat, H_G, H_X, tile=TILE, budget_k=None,
                        max_iters=1000, k_min=None, k_max=None):
    if k_min is None:
        k_min = K_MIN
    if k_max is None:
        k_max = K_MAX
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
        for donor in range(n_tiles):
            if K[donor] <= k_min:
                continue
            for receiver in range(n_tiles):
                if receiver == donor or K[receiver] >= k_max:
                    continue
                K[donor] -= 1
                K[receiver] += 1
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
    converged = not improved
    return K, current_hwe, converged


def dp_refined_allocate(W, H_X, H_G, budget_k, tile=TILE):
    K_dp, D_table = alloc_tile_local_dp(W, H_X, H_G, budget_k, tile)
    K_refined, hwe, converged = local_search_refine(W, K_dp, H_G, H_X, tile, budget_k)
    return K_refined, D_table, converged


def hessian_weighted_error(E, H_G, H_X):
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))


def arm_neither(W, H_X, H_G, K, tile=TILE):
    ntr, ntc = W.shape[0] // tile, W.shape[1] // tile
    K_flat = np.full(ntr * ntc, K, dtype=int)
    Wq = quantize_matrix(W, K_flat.reshape(ntr, ntc), tile)
    E = W - Wq
    total_bytes = compute_bytes(K_flat, rotation_sidecar=0)
    return {
        'K_flat': K_flat, 'hwe': hessian_weighted_error(E, H_G, H_X),
        'wmse': weight_mse(E), 'total_bytes': total_bytes,
        'effective_bits': total_bytes * 8 / (W.shape[0] * W.shape[1]),
        'k_std': float(np.std(K_flat)),
    }


def arm_alloc_only(W, H_X, H_G, K, tile=TILE):
    budget_k = budget_k_sum_payload_matched(K)
    K_flat, D_table, converged = dp_refined_allocate(W, H_X, H_G, budget_k, tile)
    Wq = quantize_matrix(W, K_flat.reshape(W.shape[0] // tile, W.shape[1] // tile), tile)
    E = W - Wq
    total_bytes = compute_bytes(K_flat, rotation_sidecar=0)
    return {
        'K_flat': K_flat, 'hwe': hessian_weighted_error(E, H_G, H_X),
        'wmse': weight_mse(E), 'total_bytes': total_bytes,
        'effective_bits': total_bytes * 8 / (W.shape[0] * W.shape[1]),
        'k_std': float(np.std(K_flat)), 'k_sum': int(np.sum(K_flat)),
        'converged': converged,
    }


def arm_rotate_only(W, H_X, H_G, K, rng, tile=TILE):
    d_out, d_in = W.shape
    rot_sidecar = compute_rotation_sidecar(d_in, d_out)
    W_t, H_X_t, H_G_t, U, V, S_G, S_X, sc = apply_rotation(W, H_X, H_G, rng)
    ntr, ntc = W.shape[0] // tile, W.shape[1] // tile
    K_flat = np.full(ntr * ntc, K, dtype=int)
    Wq_t = quantize_matrix(W_t, K_flat.reshape(ntr, ntc), tile)
    W_hat = inverse_rotation(Wq_t, U, V, S_G, S_X)
    E = W - W_hat
    total_bytes = compute_bytes(K_flat, rotation_sidecar=rot_sidecar)
    return {
        'K_flat': K_flat, 'hwe': hessian_weighted_error(E, H_G, H_X),
        'wmse': weight_mse(E), 'total_bytes': total_bytes,
        'effective_bits': total_bytes * 8 / (W.shape[0] * W.shape[1]),
        'k_std': float(np.std(K_flat)), 'rotation_sidecar': rot_sidecar,
    }


def arm_alloc_then_rot(W, H_X, H_G, K, rng, tile=TILE):
    d_out, d_in = W.shape
    rot_sidecar = compute_rotation_sidecar(d_in, d_out)
    budget_k = budget_k_sum_payload_matched(K)
    K_flat, D_table, converged = dp_refined_allocate(W, H_X, H_G, budget_k, tile)
    W_t, H_X_t, H_G_t, U, V, S_G, S_X, sc = apply_rotation(W, H_X, H_G, rng)
    Wq_t = quantize_matrix(W_t, K_flat.reshape(W.shape[0] // tile, W.shape[1] // tile), tile)
    W_hat = inverse_rotation(Wq_t, U, V, S_G, S_X)
    E = W - W_hat
    total_bytes = compute_bytes(K_flat, rotation_sidecar=rot_sidecar)
    return {
        'K_flat': K_flat, 'hwe': hessian_weighted_error(E, H_G, H_X),
        'wmse': weight_mse(E), 'total_bytes': total_bytes,
        'effective_bits': total_bytes * 8 / (W.shape[0] * W.shape[1]),
        'k_std': float(np.std(K_flat)), 'k_sum': int(np.sum(K_flat)),
        'converged': converged, 'rotation_sidecar': rot_sidecar,
    }


def arm_rot_then_alloc(W, H_X, H_G, K, rng, tile=TILE):
    d_out, d_in = W.shape
    rot_sidecar = compute_rotation_sidecar(d_in, d_out)
    W_t, H_X_t, H_G_t, U, V, S_G, S_X, sc = apply_rotation(W, H_X, H_G, rng)
    budget_k = budget_k_sum_payload_matched(K)
    K_flat, D_table, converged = dp_refined_allocate(W_t, H_X_t, H_G_t, budget_k, tile)
    Wq_t = quantize_matrix(W_t, K_flat.reshape(W.shape[0] // tile, W.shape[1] // tile), tile)
    W_hat = inverse_rotation(Wq_t, U, V, S_G, S_X)
    E = W - W_hat
    total_bytes = compute_bytes(K_flat, rotation_sidecar=rot_sidecar)
    return {
        'K_flat': K_flat, 'hwe': hessian_weighted_error(E, H_G, H_X),
        'wmse': weight_mse(E), 'total_bytes': total_bytes,
        'effective_bits': total_bytes * 8 / (W.shape[0] * W.shape[1]),
        'k_std': float(np.std(K_flat)), 'k_sum': int(np.sum(K_flat)),
        'converged': converged, 'rotation_sidecar': rot_sidecar,
    }


def arm_alternating(W, H_X, H_G, K, rng, tile=TILE):
    """Arm 6: allocate -> rotate -> re-allocate (warm-start local search).
    Uses the PURE warm-start path: unrotated allocation as initial point for
    local search on rotated weights. Reports warm-start result only (not best-of).
    Also tracks DP-on-rotated for comparison.
    """
    d_out, d_in = W.shape
    rot_sidecar = compute_rotation_sidecar(d_in, d_out)
    budget_k = budget_k_sum_payload_matched(K)
    K_unrot, _, conv1 = dp_refined_allocate(W, H_X, H_G, budget_k, tile)
    W_t, H_X_t, H_G_t, U, V, S_G, S_X, sc = apply_rotation(W, H_X, H_G, rng)
    K_dp_rot, _ = alloc_tile_local_dp(W_t, H_X_t, H_G_t, budget_k, tile)
    K_warm, hwe_warm, conv_warm = local_search_refine(W_t, K_unrot, H_G_t, H_X_t, tile, budget_k)
    K_dp_refined, hwe_dp, conv_dp = local_search_refine(W_t, K_dp_rot, H_G_t, H_X_t, tile, budget_k)
    # Use the PURE warm-start result (not best-of)
    K_flat = K_warm
    warm_start_won = hwe_warm <= hwe_dp
    k_diff = int(np.sum(K_warm != K_dp_refined))
    Wq_t = quantize_matrix(W_t, K_flat.reshape(W.shape[0] // tile, W.shape[1] // tile), tile)
    W_hat = inverse_rotation(Wq_t, U, V, S_G, S_X)
    E = W - W_hat
    total_bytes = compute_bytes(K_flat, rotation_sidecar=rot_sidecar)
    return {
        'K_flat': K_flat, 'hwe': hessian_weighted_error(E, H_G, H_X),
        'wmse': weight_mse(E), 'total_bytes': total_bytes,
        'effective_bits': total_bytes * 8 / (W.shape[0] * W.shape[1]),
        'k_std': float(np.std(K_flat)), 'k_sum': int(np.sum(K_flat)),
        'converged': conv_warm,
        'rotation_sidecar': rot_sidecar,
        'warm_start_won': warm_start_won, 'k_diff_warm_vs_dp': k_diff,
        'hwe_warm_start': float(hwe_warm), 'hwe_dp_refined': float(hwe_dp),
    }


def extract_slices(tensor, m=128, n=128, seed=42):
    M, N = tensor.shape
    slices = []
    slices.append(("first", tensor[:m, :n].astype(np.float64).copy()))
    r0, c0 = M // 2 - m // 2, N // 2 - n // 2
    slices.append(("mid", tensor[r0:r0+m, c0:c0+n].astype(np.float64).copy()))
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, max(1, M - m))
    c0 = rng.integers(0, max(1, N - n))
    slices.append(("rand", tensor[r0:r0+m, c0:c0+n].astype(np.float64).copy()))
    return slices


def run_experiment():
    print("=" * 80)
    print("R13-AllocRotation: Allocation + Rotation Composition")
    print("=" * 80)

    weights = np.load(WEIGHTS_PATH)
    tensor_keys = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
    all_results = {}
    t_start = time.time()

    for key in tensor_keys:
        if key not in weights:
            continue
        W_full = weights[key]
        print(f"\n{'-' * 60}")
        print(f"Tensor: {key} (shape {W_full.shape})")
        print(f"{'-' * 60}")

        slices = extract_slices(W_full, M_DIM, N_DIM)
        tensor_results = {}

        for slice_name, W in slices:
            if W.shape[0] < M_DIM or W.shape[1] < N_DIM:
                continue
            print(f"\n  Slice: {slice_name}, W shape: {W.shape}")
            H_X, H_G = synthetic_hessians(W, n_samples=P_CAL, seed=42)
            slice_results = {}

            for K in K_VALUES:
                print(f"\n  --- K={K} ---")
                arms = {}
                arms['neither'] = arm_neither(W, H_X, H_G, K)
                arms['alloc_only'] = arm_alloc_only(W, H_X, H_G, K)
                rng = np.random.default_rng(42)
                arms['rotate_only'] = arm_rotate_only(W, H_X, H_G, K, rng)
                rng = np.random.default_rng(42)
                arms['alloc_then_rot'] = arm_alloc_then_rot(W, H_X, H_G, K, rng)
                rng = np.random.default_rng(42)
                arms['rot_then_alloc'] = arm_rot_then_alloc(W, H_X, H_G, K, rng)
                rng = np.random.default_rng(42)
                arms['alternating'] = arm_alternating(W, H_X, H_G, K, rng)

                print(f"\n  {'Arm':<20} {'HWE':>14} {'Wt MSE':>12} {'Bytes':>8} {'bits/elem':>10} {'K std':>6}")
                print(f"  {'-'*20} {'-'*14} {'-'*12} {'-'*8} {'-'*10} {'-'*6}")
                for name in ['neither', 'alloc_only', 'rotate_only',
                             'alloc_then_rot', 'rot_then_alloc', 'alternating']:
                    r = arms[name]
                    print(f"  {name:<20} {r['hwe']:>14.6e} {r['wmse']:>12.4e} "
                          f"{r['total_bytes']:>8} {r['effective_bits']:>10.4f} "
                          f"{r['k_std']:>6.2f}")

                baseline_hwe = arms['neither']['hwe']
                print(f"\n  Improvement over neither:")
                for name in ['alloc_only', 'rotate_only', 'alloc_then_rot',
                             'rot_then_alloc', 'alternating']:
                    hwe = arms[name]['hwe']
                    imp = (1 - hwe / baseline_hwe) * 100 if baseline_hwe > 0 else 0
                    print(f"    {name:<20} {imp:+.2f}%")

                rot_hwe = arms['rotate_only']['hwe']
                rta_hwe = arms['rot_then_alloc']['hwe']
                atr_hwe = arms['alloc_then_rot']['hwe']
                alloc_hwe = arms['alloc_only']['hwe']
                if rot_hwe > 0:
                    print(f"\n  Marginal allocation ON TOP of rotation:")
                    print(f"    rot_then_alloc vs rotate_only: {(1 - rta_hwe / rot_hwe) * 100:+.2f}%")
                    print(f"    alloc_then_rot vs rotate_only: {(1 - atr_hwe / rot_hwe) * 100:+.2f}%")
                if alloc_hwe > 0:
                    print(f"\n  Marginal rotation ON TOP of allocation:")
                    print(f"    rot_then_alloc vs alloc_only: {(1 - rta_hwe / alloc_hwe) * 100:+.2f}%")
                    print(f"    alloc_then_rot vs alloc_only: {(1 - atr_hwe / alloc_hwe) * 100:+.2f}%")

                alt = arms['alternating']
                print(f"\n  Alternating: warm_start_won={alt['warm_start_won']}, "
                      f"k_diff={alt['k_diff_warm_vs_dp']}, "
                      f"hwe_warm={alt['hwe_warm_start']:.6e}, "
                      f"hwe_dp={alt['hwe_dp_refined']:.6e}")

                for name in arms:
                    arms[name]['K_flat'] = arms[name]['K_flat'].tolist()
                slice_results[f"K_{K}"] = arms

            # Tile sensitivity analysis at K_ref=5
            print(f"\n  --- Tile sensitivity analysis (K_ref=5) ---")
            rng_sens = np.random.default_rng(42)
            sens_before = tile_sensitivities(W, H_X, H_G, k_ref=5)
            W_t, H_X_t, H_G_t, _, _, _, _, _ = apply_rotation(W, H_X, H_G, rng_sens)
            sens_after = tile_sensitivities(W_t, H_X_t, H_G_t, k_ref=5)

            cv_before = np.std(sens_before) / (np.mean(sens_before) + 1e-15)
            cv_after = np.std(sens_after) / (np.mean(sens_after) + 1e-15)
            q_before = np.percentile(sens_before, [25, 50, 75])
            q_after = np.percentile(sens_after, [25, 50, 75])

            print(f"    Before: mean={np.mean(sens_before):.4e}, std={np.std(sens_before):.4e}, CV={cv_before:.4f}")
            print(f"    After:  mean={np.mean(sens_after):.4e}, std={np.std(sens_after):.4e}, CV={cv_after:.4f}")
            print(f"    CV ratio (after/before): {cv_after/cv_before:.4f}")
            print(f"    Before quartiles: {q_before}")
            print(f"    After quartiles:  {q_after}")
            print(f"    Before range: [{sens_before.min():.4e}, {sens_before.max():.4e}]")
            print(f"    After range:  [{sens_after.min():.4e}, {sens_after.max():.4e}]")
            print(f"    Max/min ratio: before={sens_before.max()/max(sens_before.min(),1e-15):.2f}x, "
                  f"after={sens_after.max()/max(sens_after.min(),1e-15):.2f}x")

            slice_results['sensitivity'] = {
                'before': {
                    'mean': float(np.mean(sens_before)), 'std': float(np.std(sens_before)),
                    'cv': float(cv_before), 'min': float(sens_before.min()),
                    'max': float(sens_before.max()),
                    'q25': float(q_before[0]), 'q50': float(q_before[1]), 'q75': float(q_before[2]),
                    'values': sens_before.tolist(),
                },
                'after': {
                    'mean': float(np.mean(sens_after)), 'std': float(np.std(sens_after)),
                    'cv': float(cv_after), 'min': float(sens_after.min()),
                    'max': float(sens_after.max()),
                    'q25': float(q_after[0]), 'q50': float(q_after[1]), 'q75': float(q_after[2]),
                    'values': sens_after.tolist(),
                },
                'cv_ratio': float(cv_after / cv_before),
                'std_ratio': float(np.std(sens_after) / max(np.std(sens_before), 1e-15)),
            }

            tensor_results[slice_name] = slice_results
        all_results[key] = tensor_results

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Mean HWE improvement over 'neither' across all tensors/slices/K")
    print("=" * 80)

    arm_names = ['alloc_only', 'rotate_only', 'alloc_then_rot', 'rot_then_alloc', 'alternating']
    arm_imps = {name: [] for name in arm_names}
    rta_over_rot = []
    atr_over_rot = []
    rta_over_alloc = []
    atr_over_alloc = []
    cv_ratios = []

    for key in tensor_keys:
        if key not in all_results:
            continue
        for sn in all_results[key]:
            sr = all_results[key][sn]
            for K in K_VALUES:
                kk = f"K_{K}"
                if kk not in sr:
                    continue
                baseline = sr[kk]['neither']['hwe']
                for name in arm_names:
                    hwe = sr[kk][name]['hwe']
                    arm_imps[name].append((1 - hwe / baseline) * 100 if baseline > 0 else 0)
                rot_hwe = sr[kk]['rotate_only']['hwe']
                rta_hwe = sr[kk]['rot_then_alloc']['hwe']
                atr_hwe = sr[kk]['alloc_then_rot']['hwe']
                alloc_hwe = sr[kk]['alloc_only']['hwe']
                if rot_hwe > 0:
                    rta_over_rot.append((1 - rta_hwe / rot_hwe) * 100)
                    atr_over_rot.append((1 - atr_hwe / rot_hwe) * 100)
                if alloc_hwe > 0:
                    rta_over_alloc.append((1 - rta_hwe / alloc_hwe) * 100)
                    atr_over_alloc.append((1 - atr_hwe / alloc_hwe) * 100)
            if 'sensitivity' in sr:
                cv_ratios.append(sr['sensitivity']['cv_ratio'])

    print(f"\n  {'Arm':<20} {'Mean imp%':>10} {'Median imp%':>12} {'Min%':>8} {'Max%':>8} {'N':>4}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*4}")
    for name in arm_names:
        imps = arm_imps[name]
        print(f"  {name:<20} {np.mean(imps):>10.2f} {np.median(imps):>12.2f} "
              f"{min(imps):>8.2f} {max(imps):>8.2f} {len(imps):>4}")

    print(f"\n  Marginal allocation ON TOP of rotation:")
    print(f"    rot_then_alloc vs rotate_only: mean {np.mean(rta_over_rot):+.2f}%, median {np.median(rta_over_rot):+.2f}%")
    print(f"    alloc_then_rot vs rotate_only: mean {np.mean(atr_over_rot):+.2f}%, median {np.median(atr_over_rot):+.2f}%")

    print(f"\n  Marginal rotation ON TOP of allocation:")
    print(f"    rot_then_alloc vs alloc_only: mean {np.mean(rta_over_alloc):+.2f}%, median {np.median(rta_over_alloc):+.2f}%")
    print(f"    alloc_then_rot vs alloc_only: mean {np.mean(atr_over_alloc):+.2f}%, median {np.median(atr_over_alloc):+.2f}%")

    print(f"\n  Tile sensitivity CV ratio (after/before rotation):")
    print(f"    mean {np.mean(cv_ratios):.4f}, median {np.median(cv_ratios):.4f}, "
          f"min {min(cv_ratios):.4f}, max {max(cv_ratios):.4f}")
    if np.mean(cv_ratios) < 1.0:
        print(f"    -> Rotation HOMOGENIZES tile sensitivity (CV reduced by {(1-np.mean(cv_ratios))*100:.1f}%)")
    else:
        print(f"    -> Rotation does NOT homogenize tile sensitivity (CV increased by {(np.mean(cv_ratios)-1)*100:.1f}%)")

    output = {
        "config": {"matrix_dim": [M_DIM, N_DIM], "tile_size": TILE, "n_tiles": N_TILES,
                   "k_values": K_VALUES, "p_cal": P_CAL, "weights_path": WEIGHTS_PATH,
                   "tensor_keys": tensor_keys, "budget_mode": "payload_matched"},
        "summary": {
            "arm_improvements": {name: {"mean": float(np.mean(imps)), "median": float(np.median(imps)),
                                        "min": float(min(imps)), "max": float(max(imps)), "n": len(imps)}
                                 for name, imps in arm_imps.items()},
            "marginal_alloc_on_rot": {
                "rot_then_alloc_vs_rotate_only": {"mean": float(np.mean(rta_over_rot)), "median": float(np.median(rta_over_rot))},
                "alloc_then_rot_vs_rotate_only": {"mean": float(np.mean(atr_over_rot)), "median": float(np.median(atr_over_rot))},
            },
            "marginal_rot_on_alloc": {
                "rot_then_alloc_vs_alloc_only": {"mean": float(np.mean(rta_over_alloc)), "median": float(np.median(rta_over_alloc))},
                "alloc_then_rot_vs_alloc_only": {"mean": float(np.mean(atr_over_alloc)), "median": float(np.median(atr_over_alloc))},
            },
            "sensitivity_cv_ratio": {"mean": float(np.mean(cv_ratios)), "median": float(np.median(cv_ratios)),
                                      "min": float(min(cv_ratios)), "max": float(max(cv_ratios))},
        },
        "results": all_results,
        "elapsed_seconds": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_experiment()
