#!/usr/bin/env python3
"""
R7-NoiseShaping v3: Hessian Noise Shaping — fully corrected after two review rounds.

Key fixes from v2 review:
1. GPTQ uses H^{-1}[q,R]/H^{-1}[q,q] computed via Cholesky (verified non-zero)
2. FIXED group membership: all arms use original contiguous 16-column groups
3. Code clipping: clip(round(...), 0, 2^K - 1) to guarantee K-bit representability
4. Norm-matched filter ablation: all filters normalized to equal upper-triangular Frobenius norm
5. Honest claims: macro-mean with paired win counts vs RTN, head-to-head vs other orders
6. Sanity check: GPTQ-on vs GPTQ-off MUST produce different results

Scope: uniform-group quantization (NOT EXL3 trellis). H_G = output covariance proxy
(NOT true gradient/Fisher Hessian). 4 fixed 128x128 slices, 3 calibration seeds.
"""

import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)

WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
OUTPUT_DIR = Path("/Users/mbelleau/Projects/qwen38-research-r7-noise-shaping/receipts/research")

K_VALUES = [3, 4, 5, 6]
GROUP_SIZE = 16
SEEDS = [42, 123, 777]
N_SAMPLES_CALIB = 512

# ─── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed):
    return np.random.RandomState(seed)

def load_real_weights():
    data = np.load(WEIGHTS_PATH)
    tensors = {}
    for key in ['L0_gate', 'L0_down', 'L55_gate', 'L55_down']:
        w = data[key].astype(np.float64)
        tensors[key] = w[:128, :128].copy()
    return tensors

def generate_calibration(n_in, n_samples, rng):
    cov_base = np.eye(n_in) * 0.5
    for i in range(n_in - 1):
        cov_base[i, i+1] = 0.1
        cov_base[i+1, i] = 0.1
    X = rng.multivariate_normal(np.zeros(n_in), cov_base, size=n_samples)
    n_outliers = max(1, n_in // 20)
    outlier_idx = rng.choice(n_in, n_outliers, replace=False)
    X[:, outlier_idx] *= 5.0
    return X

def compute_hessians(W, X):
    N = X.shape[0]
    H_X = np.nan_to_num(X.T @ X / N, nan=0.0, posinf=1e6, neginf=-1e6)
    Y = X @ W.T
    H_G = np.nan_to_num(Y.T @ Y / N, nan=0.0, posinf=1e6, neginf=-1e6)
    return H_X, H_G

def safe_eigh(H, epsilon=1e-10):
    H = np.nan_to_num((H + H.T) / 2, nan=0.0, posinf=1e6, neginf=-1e6)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvals, eigvecs

# ─── Quantizer with clipping ──────────────────────────────────────────────────

def quantize_with_scale(w_col, K, scale_min, scale_max):
    levels = 2 ** K
    if scale_max - scale_min < 1e-15:
        return w_col.copy()
    scale = (scale_max - scale_min) / (levels - 1)
    codes = np.clip(np.round((w_col - scale_min) / scale), 0, levels - 1)
    return codes * scale + scale_min

def compute_group_scales(W, group_size=GROUP_SIZE):
    n_cols = W.shape[1]
    scales = []
    for start in range(0, n_cols, group_size):
        end = min(start + group_size, n_cols)
        block = W[:, start:end]
        scales.append((block.min(), block.max()))
    return scales

def get_group_scale_for_col(col_idx, group_scales, group_size=GROUP_SIZE):
    return group_scales[col_idx // group_size]

# ─── Metrics ──────────────────────────────────────────────────────────────────

def hessian_weighted_error(E, H_G, H_X):
    val = np.trace(H_G @ E @ H_X @ E.T)
    return float(val) if np.isfinite(val) else 1e20

def raw_mse(E):
    return float(np.sum(E ** 2) / E.size)

def spectral_analysis(E, H_X, H_G):
    eig_H_X, V_X = safe_eigh(H_X)
    V_X_desc = V_X[:, ::-1]
    eig_H_X_desc = eig_H_X[::-1]
    E_proj = E @ V_X_desc
    g_j = np.array([(E_proj[:, j] @ H_G @ E_proj[:, j]) for j in range(E_proj.shape[1])])
    return {'eig_H_X': eig_H_X_desc, 'error_energy_HG_weighted': g_j}

def anti_correlation(eig_H_X, error_energy):
    h = eig_H_X / (eig_H_X.sum() + 1e-15)
    e = error_energy / (error_energy.sum() + 1e-15)
    if np.std(h) < 1e-15 or np.std(e) < 1e-15:
        return 0.0
    return float(np.corrcoef(h, e)[0, 1])

# ─── Column orderings ─────────────────────────────────────────────────────────

def order_left_to_right(n):
    return np.arange(n)

def order_right_to_left(n):
    return np.arange(n - 1, -1, -1)

def order_random(n, rng):
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm

def order_descending_diag_H(H_X):
    return np.argsort(np.diag(H_X))[::-1]

def order_ascending_diag_H(H_X):
    return np.argsort(np.diag(H_X))

# ─── GPTQ ─────────────────────────────────────────────────────────────────────

def compute_upper_cholesky_inv(H, damping_factor=0.01):
    """
    Compute U (upper triangular) such that U^T U = (H + lambda*I)^{-1}.
    Correct for GPTQ: update uses U[i, i+1:] / U[i, i].

    Construction (matching GPTAQ reference):
        Hd = H + lambda*I
        Hinv = inv(Hd)
        U = cholesky(Hinv).T   (upper triangular, U^T U = Hinv)
    """
    n = H.shape[0]
    damping = damping_factor * max(np.mean(np.diag(H)), 0.0) if np.mean(np.diag(H)) > 0 else 1e-6
    H_damped = H + damping * np.eye(n)
    try:
        H_inv = np.linalg.inv(H_damped)
        L_chol = np.linalg.cholesky(H_inv)  # L_chol @ L_chol.T = H_inv (lower)
        U = L_chol.T  # upper triangular, U^T U = H_inv
        return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)
    except np.linalg.LinAlgError:
        return np.nan_to_num(np.linalg.pinv(H_damped), nan=0.0, posinf=1e6, neginf=-1e6)

def gptq_ordered(W, H, K, order, group_scales, group_size=GROUP_SIZE, alpha=1.0):
    """
    Sequential GPTQ with configurable column ordering.
    Uses U[i, i+1:] / U[i, i] where U^T U = H^{-1} (upper Cholesky factor).
    Group scales precomputed from original W (fixed for all arms).
    Codes clipped to [0, 2^K - 1].
    """
    n_rows, n_cols = W.shape
    H_perm = H[np.ix_(order, order)]
    U = compute_upper_cholesky_inv(H_perm)

    W_work = W.copy()
    Wq = np.zeros_like(W)

    for idx in range(n_cols):
        q_orig = order[idx]
        gmin, gmax = get_group_scale_for_col(q_orig, group_scales, group_size)
        Wq[:, q_orig] = quantize_with_scale(W_work[:, q_orig], K, gmin, gmax)
        e_q = W_work[:, q_orig] - Wq[:, q_orig]

        if idx < n_cols - 1:
            remaining_orig = order[idx + 1:]
            u_ii = U[idx, idx]
            if abs(u_ii) > 1e-15:
                update = alpha * np.outer(e_q, U[idx, idx + 1:] / u_ii)
                W_work[:, remaining_orig] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)

    return Wq, W - Wq

# ─── GPTQ + Spectral Feedback (norm-matched) ─────────────────────────────────

def gptq_spectral_feedback(W, H, K, order, group_scales, group_size=GROUP_SIZE,
                           alpha=1.0, filter_type='inverse_eig', filter_strength=0.01):
    n_rows, n_cols = W.shape
    H_perm = H[np.ix_(order, order)]
    U = compute_upper_cholesky_inv(H_perm)

    eigvals, eigvecs = safe_eigh(H_perm)
    if filter_type == 'flat':
        F = np.ones((n_cols, n_cols))
    elif filter_type == 'inverse_sqrt':
        inv_sqrt = 1.0 / np.sqrt(eigvals)
        F = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T
    elif filter_type == 'inverse_eig':
        inv_eig = 1.0 / eigvals
        F = eigvecs @ np.diag(inv_eig) @ eigvecs.T
    else:
        F = np.eye(n_cols)

    # Norm-match: normalize upper-triangular part to unit Frobenius norm
    F_upper = np.triu(F, k=1)
    F_norm = np.linalg.norm(F_upper)
    if F_norm > 1e-15:
        F = F / F_norm
    F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)

    W_work = W.copy()
    Wq = np.zeros_like(W)

    for idx in range(n_cols):
        q_orig = order[idx]
        gmin, gmax = get_group_scale_for_col(q_orig, group_scales, group_size)
        Wq[:, q_orig] = quantize_with_scale(W_work[:, q_orig], K, gmin, gmax)
        e_q = W_work[:, q_orig] - Wq[:, q_orig]

        if idx < n_cols - 1:
            remaining_orig = order[idx + 1:]
            u_ii = U[idx, idx]
            update = np.zeros((n_rows, n_cols - idx - 1))
            if abs(u_ii) > 1e-15:
                update += alpha * np.outer(e_q, U[idx, idx + 1:] / u_ii)
            update += filter_strength * np.outer(e_q, F[idx, idx + 1:])
            W_work[:, remaining_orig] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)

    return Wq, W - Wq

# ─── RTN Baseline ─────────────────────────────────────────────────────────────

def rtn_baseline(W, K, group_scales, group_size=GROUP_SIZE):
    Wq = np.zeros_like(W)
    n_cols = W.shape[1]
    for start in range(0, n_cols, group_size):
        end = min(start + group_size, n_cols)
        gmin, gmax = group_scales[start // group_size]
        for col in range(start, end):
            Wq[:, col] = quantize_with_scale(W[:, col], K, gmin, gmax)
    return Wq, W - Wq

# ─── Non-budget-matched diagnostics ──────────────────────────────────────────

def hessian_basis_quantization(W, H_X, K, group_size=GROUP_SIZE):
    eigvals, V_X = safe_eigh(H_X)
    W_rot = W @ V_X
    Wq_rot = np.zeros_like(W_rot)
    n_cols = W.shape[1]
    for start in range(0, n_cols, group_size):
        end = min(start + group_size, n_cols)
        block = W_rot[:, start:end]
        levels = 2 ** K
        bmin, bmax = block.min(), block.max()
        if bmax - bmin < 1e-15:
            Wq_rot[:, start:end] = block.copy()
        else:
            scale = (bmax - bmin) / (levels - 1)
            codes = np.clip(np.round((block - bmin) / scale), 0, levels - 1)
            Wq_rot[:, start:end] = codes * scale + bmin
    Wq = Wq_rot @ V_X.T
    return Wq, W - Wq

def error_projection_diagnostic(W, H_X, K, group_scales, group_size=GROUP_SIZE, n_low=None):
    n_cols = W.shape[1]
    if n_low is None:
        n_low = n_cols // 2
    eigvals, eigvecs = safe_eigh(H_X)
    V_low = eigvecs[:, :n_low]
    P_low = V_low @ V_low.T
    Wq_initial = np.zeros_like(W)
    for start in range(0, n_cols, group_size):
        end = min(start + group_size, n_cols)
        gmin, gmax = group_scales[start // group_size]
        for col in range(start, end):
            Wq_initial[:, col] = quantize_with_scale(W[:, col], K, gmin, gmax)
    E_initial = W - Wq_initial
    E_shaped = E_initial @ P_low
    Wq = W - E_shaped
    return Wq, W - Wq

# ─── Sanity checks ────────────────────────────────────────────────────────────

def run_sanity_checks(tensors):
    """Verify GPTQ actually does something (not a no-op like R5's bug)."""
    print("\n" + "=" * 80)
    print("SANITY CHECK: GPTQ-on vs GPTQ-off (must differ)")
    print("=" * 80)

    W = tensors['L0_gate']
    rng = set_seed(42)
    X = generate_calibration(W.shape[1], N_SAMPLES_CALIB, rng)
    H_X, H_G = compute_hessians(W, X)
    group_scales = compute_group_scales(W)
    order = order_left_to_right(W.shape[1])
    K = 4

    Wq_rtn, E_rtn = rtn_baseline(W, K, group_scales)
    hwe_rtn = hessian_weighted_error(E_rtn, H_G, H_X)

    Wq_a0, E_a0 = gptq_ordered(W, H_X, K, order, group_scales, alpha=0.0)
    hwe_a0 = hessian_weighted_error(E_a0, H_G, H_X)

    Wq_a1, E_a1 = gptq_ordered(W, H_X, K, order, group_scales, alpha=1.0)
    hwe_a1 = hessian_weighted_error(E_a1, H_G, H_X)

    print(f"  RTN HWE:           {hwe_rtn:.6e}")
    print(f"  GPTQ alpha=0 HWE:  {hwe_a0:.6e}  (should = RTN)")
    print(f"  GPTQ alpha=1 HWE:  {hwe_a1:.6e}  (should differ)")
    print(f"  alpha=0 vs RTN:    max|diff| = {np.max(np.abs(Wq_rtn - Wq_a0)):.2e}  (should be ~0)")
    print(f"  alpha=1 vs RTN:    max|diff| = {np.max(np.abs(Wq_rtn - Wq_a1)):.2e}  (should be >0)")

    U = compute_upper_cholesky_inv(H_X)
    print(f"  U norm: {np.linalg.norm(U):.6e}  (should be >0)")

    if np.max(np.abs(Wq_rtn - Wq_a1)) < 1e-10:
        print("  FAIL: GPTQ is a no-op! Correction is broken!")
        return False
    print("  PASS: GPTQ correction is active.")
    return True

# ─── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(tensors):
    all_results = {}

    for tensor_name, W_orig in tensors.items():
        print(f"\n{'~'*60}")
        print(f"Tensor: {tensor_name} (shape {W_orig.shape})")
        print(f"  W range: [{W_orig.min():.6f}, {W_orig.max():.6f}], std: {W_orig.std():.6f}")
        print(f"{'~'*60}")

        for seed in SEEDS:
            rng = set_seed(seed)
            X = generate_calibration(W_orig.shape[1], N_SAMPLES_CALIB, rng)
            H_X, H_G = compute_hessians(W_orig, X)
            n_cols = W_orig.shape[1]
            group_scales = compute_group_scales(W_orig)

            order_lr = order_left_to_right(n_cols)
            order_rl = order_right_to_left(n_cols)
            order_rand = order_random(n_cols, rng)
            order_desc = order_descending_diag_H(H_X)
            order_asc = order_ascending_diag_H(H_X)

            fstrength = 0.01

            def make_arms(K_val):
                return {
                    'RTN': lambda: rtn_baseline(W_orig, K_val, group_scales),
                    'GPTQ_LR': lambda: gptq_ordered(W_orig, H_X, K_val, order_lr, group_scales),
                    'GPTQ_RL': lambda: gptq_ordered(W_orig, H_X, K_val, order_rl, group_scales),
                    'GPTQ_random': lambda: gptq_ordered(W_orig, H_X, K_val, order_rand, group_scales),
                    'GPTQ_actorder': lambda: gptq_ordered(W_orig, H_X, K_val, order_desc, group_scales),
                    'GPTQ_rev_actorder': lambda: gptq_ordered(W_orig, H_X, K_val, order_asc, group_scales),
                    'GPTQ_actorder_flat': lambda: gptq_spectral_feedback(W_orig, H_X, K_val, order_desc, group_scales, filter_type='flat', filter_strength=fstrength),
                    'GPTQ_actorder_invsqrt': lambda: gptq_spectral_feedback(W_orig, H_X, K_val, order_desc, group_scales, filter_type='inverse_sqrt', filter_strength=fstrength),
                    'GPTQ_actorder_inveig': lambda: gptq_spectral_feedback(W_orig, H_X, K_val, order_desc, group_scales, filter_type='inverse_eig', filter_strength=fstrength),
                    'GPTQ_LR_invsqrt': lambda: gptq_spectral_feedback(W_orig, H_X, K_val, order_lr, group_scales, filter_type='inverse_sqrt', filter_strength=fstrength),
                    'Hessian_basis*': lambda: hessian_basis_quantization(W_orig, H_X, K_val),
                    'Error_projection*': lambda: error_projection_diagnostic(W_orig, H_X, K_val, group_scales),
                }

            for K in K_VALUES:
                arms = make_arms(K)
                for arm_name, arm_fn in arms.items():
                    Wq, E = arm_fn()
                    hwe = hessian_weighted_error(E, H_G, H_X)
                    mse = raw_mse(E)
                    spec = spectral_analysis(E, H_X, H_G)
                    acorr = anti_correlation(spec['eig_H_X'], spec['error_energy_HG_weighted'])
                    key = f"{tensor_name}_K{K}_seed{seed}_{arm_name}"
                    all_results[key] = {
                        'tensor': tensor_name, 'K': K, 'seed': seed, 'strategy': arm_name,
                        'hessian_weighted_error': hwe, 'raw_mse': mse,
                        'anti_correlation': acorr, 'noise_floor': 0.0,
                    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'r7-noise-shaping-results-v3.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved.")

    tensor_names = list(tensors.keys())
    arm_names = list(arms.keys())

    # ─── HWE improvement ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HWE improvement vs RTN (macro mean of per-tensor seed-means)")
    print("=" * 80)

    for K in K_VALUES:
        print(f"\n  K={K}:")
        rtn_means = []
        for t in tensor_names:
            vals = [v['hessian_weighted_error'] for k, v in all_results.items()
                    if f'{t}_K{K}_' in k and k.endswith('_RTN')]
            rtn_means.append(np.mean(vals))

        for arm in arm_names:
            strat_means = []
            win_vs_rtn = 0
            total_vs_rtn = 0
            for ti, t in enumerate(tensor_names):
                vals = [v['hessian_weighted_error'] for k, v in all_results.items()
                        if f'{t}_K{K}_' in k and k.endswith(f'_{arm}')]
                if vals:
                    strat_means.append(np.mean(vals))
                for s in SEEDS:
                    rv = all_results.get(f'{t}_K{K}_seed{s}_RTN', {}).get('hessian_weighted_error')
                    sv = all_results.get(f'{t}_K{K}_seed{s}_{arm}', {}).get('hessian_weighted_error')
                    if rv is not None and sv is not None:
                        total_vs_rtn += 1
                        if sv < rv:
                            win_vs_rtn += 1

            if len(strat_means) == len(rtn_means):
                ratios = [s / max(r, 1e-20) for s, r in zip(strat_means, rtn_means)]
                improvement = (1 - np.mean(ratios)) * 100
                wr = win_vs_rtn / max(total_vs_rtn, 1) * 100
                marker = '*' if arm.endswith('*') else ' '
                print(f"    {marker}{arm:30s}: {improvement:+6.1f}%  (win {win_vs_rtn}/{total_vs_rtn} = {wr:.0f}% vs RTN)")

    # ─── Head-to-head ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD: act-order vs other orderings (paired wins)")
    print("=" * 80)

    for K in K_VALUES:
        print(f"\n  K={K}:")
        for other in ['GPTQ_LR', 'GPTQ_RL', 'GPTQ_random', 'GPTQ_rev_actorder']:
            wins = 0
            total = 0
            for t in tensor_names:
                for s in SEEDS:
                    av = all_results.get(f'{t}_K{K}_seed{s}_GPTQ_actorder', {}).get('hessian_weighted_error')
                    ov = all_results.get(f'{t}_K{K}_seed{s}_{other}', {}).get('hessian_weighted_error')
                    if av is not None and ov is not None:
                        total += 1
                        if av < ov:
                            wins += 1
            print(f"    act-order vs {other:20s}: win {wins}/{total} ({wins/max(total,1)*100:.0f}%)")

    # ─── Anti-correlation ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANTI-CORRELATION (H_G-weighted error energy vs H_X eigenvalues)")
    print("=" * 80)

    for K in K_VALUES:
        print(f"\n  K={K}:")
        for arm in arm_names:
            acs = [v['anti_correlation'] for k, v in all_results.items()
                   if f'_K{K}_' in k and k.endswith(f'_{arm}')]
            if acs:
                print(f"    {arm:30s}: {np.mean(acs):.4f} +/- {np.std(acs):.4f}")

    # ─── MSE vs HWE ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MSE vs HWE TRADE-OFF (geometric mean over tensors)")
    print("=" * 80)

    for K in K_VALUES:
        print(f"\n  K={K}:")
        print(f"    {'Strategy':30s} {'HWE/RTN':>8s} {'MSE/RTN':>8s}")
        rtn_hwe, rtn_mse = [], []
        for t in tensor_names:
            vh = [v['hessian_weighted_error'] for k, v in all_results.items()
                  if f'{t}_K{K}_' in k and k.endswith('_RTN')]
            vm = [v['raw_mse'] for k, v in all_results.items()
                  if f'{t}_K{K}_' in k and k.endswith('_RTN')]
            if vh: rtn_hwe.append(np.mean(vh))
            if vm: rtn_mse.append(np.mean(vm))
        rtn_hwe_gm = np.exp(np.mean(np.log(np.maximum(rtn_hwe, 1e-20))))
        rtn_mse_gm = np.exp(np.mean(np.log(np.maximum(rtn_mse, 1e-20))))
        print(f"    {'RTN (baseline)':30s} {'1.000':>8s} {'1.000':>8s}")
        for arm in arm_names:
            if arm == 'RTN': continue
            hwe_vals, mse_vals = [], []
            for t in tensor_names:
                vh = [v['hessian_weighted_error'] for k, v in all_results.items()
                      if f'{t}_K{K}_' in k and k.endswith(f'_{arm}')]
                vm = [v['raw_mse'] for k, v in all_results.items()
                      if f'{t}_K{K}_' in k and k.endswith(f'_{arm}')]
                if vh: hwe_vals.append(np.mean(vh))
                if vm: mse_vals.append(np.mean(vm))
            if hwe_vals:
                hwe_gm = np.exp(np.mean(np.log(np.maximum(hwe_vals, 1e-20))))
                mse_gm = np.exp(np.mean(np.log(np.maximum(mse_vals, 1e-20))))
                print(f"    {arm:30s} {hwe_gm/rtn_hwe_gm:8.3f} {mse_gm/rtn_mse_gm:8.3f}")

    # ─── Norm-matched filter ablation ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NORM-MATCHED FILTER ABLATION (act-order, alpha=1.0, K=4)")
    print("  All filters normalized to unit upper-triangular Frobenius norm")
    print("=" * 80)

    for tensor_name, W_orig in tensors.items():
        rng = set_seed(42)
        X = generate_calibration(W_orig.shape[1], N_SAMPLES_CALIB, rng)
        H_X, H_G = compute_hessians(W_orig, X)
        group_scales = compute_group_scales(W_orig)
        order = order_descending_diag_H(H_X)
        K = 4

        print(f"\n  {tensor_name}:")
        Wq, E = gptq_ordered(W_orig, H_X, K, order, group_scales, alpha=1.0)
        hwe = hessian_weighted_error(E, H_G, H_X)
        print(f"    no_filter           : HWE={hwe:.6e}")
        for fstrength in [0.005, 0.01, 0.02, 0.05]:
            for ftype in ['flat', 'inverse_sqrt', 'inverse_eig']:
                Wq, E = gptq_spectral_feedback(W_orig, H_X, K, order, group_scales,
                                               alpha=1.0, filter_type=ftype, filter_strength=fstrength)
                hwe = hessian_weighted_error(E, H_G, H_X)
                print(f"    {ftype:15s} fs={fstrength:.3f}: HWE={hwe:.6e}")

    # ─── Code range validation ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CODE RANGE VALIDATION")
    print("=" * 80)
    rng = set_seed(42)
    for tensor_name, W_orig in tensors.items():
        X = generate_calibration(W_orig.shape[1], N_SAMPLES_CALIB, rng)
        H_X, _ = compute_hessians(W_orig, X)
        group_scales = compute_group_scales(W_orig)
        order = order_descending_diag_H(H_X)
        for K in [3, 5]:
            Wq, E = gptq_ordered(W_orig, H_X, K, order, group_scales)
            ok = True
            for gi, (gmin, gmax) in enumerate(group_scales):
                if gmax - gmin < 1e-15: continue
                scale = (gmax - gmin) / (2**K - 1)
                codes = (Wq[:, gi * GROUP_SIZE:(gi + 1) * GROUP_SIZE] - gmin) / scale
                if codes.max() > 2**K - 1 + 1e-6 or codes.min() < -1e-6:
                    print(f"  FAIL: {tensor_name} K={K} group {gi}: codes [{codes.min():.2f}, {codes.max():.2f}]")
                    ok = False
                    break
            if ok:
                print(f"  PASS: {tensor_name} K={K}: all codes in [0, {2**K - 1}]")

    return all_results

# ─── Spectral viz ─────────────────────────────────────────────────────────────

def generate_spectral_viz(tensors):
    viz = {}
    for tensor_name, W_orig in tensors.items():
        rng = set_seed(42)
        X = generate_calibration(W_orig.shape[1], N_SAMPLES_CALIB, rng)
        H_X, H_G = compute_hessians(W_orig, X)
        group_scales = compute_group_scales(W_orig)
        order = order_descending_diag_H(H_X)
        eigvals_X, _ = safe_eigh(H_X)
        eig_H_X = eigvals_X[::-1]
        viz[tensor_name] = {'eig_H_X': eig_H_X.tolist(), 'spectra': {}}
        for K in [3, 5]:
            viz[tensor_name]['spectra'][f'K{K}'] = {}
            arms = {
                'RTN': lambda K=K: rtn_baseline(W_orig, K, group_scales),
                'GPTQ_LR': lambda K=K: gptq_ordered(W_orig, H_X, K, order_left_to_right(W_orig.shape[1]), group_scales),
                'GPTQ_actorder': lambda K=K: gptq_ordered(W_orig, H_X, K, order, group_scales),
                'Error_projection*': lambda K=K: error_projection_diagnostic(W_orig, H_X, K, group_scales),
            }
            for arm_name, arm_fn in arms.items():
                Wq, E = arm_fn()
                spec = spectral_analysis(E, H_X, H_G)
                viz[tensor_name]['spectra'][f'K{K}'][arm_name] = {
                    'error_energy_HG_weighted': spec['error_energy_HG_weighted'].tolist(),
                    'anti_correlation': anti_correlation(spec['eig_H_X'], spec['error_energy_HG_weighted']),
                }
    with open(OUTPUT_DIR / 'r7-spectral-viz-v3.json', 'w') as f:
        json.dump(viz, f, indent=2)
    print(f"\nSpectral viz saved.")
    return viz

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tensors = load_real_weights()

    if not run_sanity_checks(tensors):
        print("SANITY CHECK FAILED -- aborting.")
        exit(1)

    results = run_experiment(tensors)
    viz = generate_spectral_viz(tensors)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE (v3, fully corrected)")
    print("=" * 80)
    print(f"  Results: {OUTPUT_DIR / 'r7-noise-shaping-results-v3.json'}")
    print(f"  Spectral viz: {OUTPUT_DIR / 'r7-spectral-viz-v3.json'}")
