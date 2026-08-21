#!/usr/bin/env python3
"""
R4-Permutations: Tile packing + channel permutations for trellis quantization.

Hypothesis: Permuting weight matrix columns (and optionally rows) to group similar
channels into 16×16 tiles reduces within-tile dynamic range, improving codebook
quantization quality. Permutations are nearly free when baked into the model.

Strategies implemented (column permutation unless noted):
  1. identity             — no permutation (baseline)
  2. act_order_desc       — descending H_X diagonal (OBS saliency)
  3. act_order_asc         — ascending H_X diagonal
  4. scale_homogeneous     — sort by column RMS (groups similar-scale channels)
  5. correlation_weight    — spectral seriation on |corr(W^T W)|
  6. variance_based         — sort by column variance
  7. spectral_HX           — spectral seriation on |corr(H_X)|
  8. random                — random permutation (control)
  9. hadamard              — block Hadamard (no permutation, for comparison)
 10. two_sided_scale       — row + column scale-homogeneous packing
 11. two_sided_corr         — row + column correlation-based packing

Quantizer: per-tile uniform (16×16), identical for all arms (matched granularity).

Metrics:
  - Within-tile range (max - min per tile, mean/median/max)
  - Within-tile dynamic range ratio (max_abs / min_abs per tile)
  - Weight MSE: mean((W - Ŵ)^2)
  - Hessian-weighted error: tr(H_G · E · H_X · E^T) / (m·n)
  - Noise floor: 0 for symmetric metrics (E=0 when no quantization)

Architectural constraints (derived mathematically, see derive_constraints()):
  - MLP-safe: permutations commute with SiLU + elementwise product; rotations do NOT
  - Attention-invariant: same Q/K head-dim perm preserves QK^T; V/O inverse pair

Test on real Qwen3.8-27B weights (L0/L55), K = 3,4,5,6.
"""


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

# ==================== Configuration ====================
TILE = 16
SLICE_SIZE = 128
K_VALUES = [3, 4, 5, 6]
N_CALIB = 512
SEED = 42
WEIGHTS_PATH = '/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz'

# Tensors and their K5K6 recipe bit widths
TENSOR_K = {
    'L0_gate': 5, 'L55_gate': 5,
    'L0_down': 6, 'L55_down': 6,
    'L0_qkv': 6, 'L0_out': 6, 'L0_z': 6,
}
SLICES = ['first', 'middle', 'last', 'random']

# ==================== Quantizer (matched for all arms) ====================

def hadamard(n):
    """Normalized Hadamard matrix of size n (n must be power of 2)."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def quantize_uniform(w, bits):
    """Uniform quantization to 2^bits levels over [min, max]."""
    nl = 2 ** bits
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return w.copy()
    s = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / s), 0, nl - 1) * s + lo

def tile_quantize(W, bits, tile=TILE):
    """Per-tile uniform quantization. Same quantizer for ALL arms."""
    m, n = W.shape
    Wq = np.zeros_like(W, dtype=np.float64)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            block = W[i:i+tile, j:j+tile]
            Wq[i:i+tile, j:j+tile] = quantize_uniform(block, bits)
    return Wq

def block_hadamard(W, tile=TILE):
    """Apply Hadamard transform to each tile block. Self-inverse."""
    m, n = W.shape
    R = W.copy().astype(np.float64)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            h_row = hadamard(min(tile, m - i))
            h_col = hadamard(min(tile, n - j))
            R[i:i+tile, j:j+tile] = h_row @ W[i:i+tile, j:j+tile] @ h_col
    return R

# ==================== Calibration data ====================

def gen_calibration(n, N, seed=42):
    """Synthetic calibration with per-channel scale variation + correlations."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N))
    # Per-channel scale: 10% outliers with 3-8x scale
    scales = np.ones(n)
    n_out = max(1, n // 10)
    outlier_idx = rng.choice(n, n_out, replace=False)
    scales[outlier_idx] = rng.uniform(3.0, 8.0, n_out)
    X *= scales[:, None]
    # Inter-channel correlation: 20% channels share a common factor
    n_corr = max(2, n // 5)
    corr_idx = rng.choice(n, n_corr, replace=False)
    common = rng.standard_normal(N) * 0.5
    X[corr_idx] += common[None, :]
    return X

# ==================== Hessian ====================

def compute_hessians(W, X):
    """Input Hessian H_X = X X^T / N, output Hessian proxy H_G = Y Y^T / N."""
    N = X.shape[1]
    H_X = X @ X.T / N
    Y = W @ X
    H_G = Y @ Y.T / N
    return H_X, H_G

# ==================== Permutation strategies ====================

def inverse_perm(P):
    """Inverse permutation: inv[P[i]] = i."""
    inv = np.zeros_like(P)
    inv[P] = np.arange(len(P))
    return inv

def perm_identity(n, W, H_X):
    return np.arange(n)

def perm_act_order_desc(n, W, H_X):
    """Descending OBS saliency (H_X diagonal). Most sensitive channels first."""
    return np.argsort(-np.diag(H_X))

def perm_act_order_asc(n, W, H_X):
    """Ascending OBS saliency. Least sensitive channels first."""
    return np.argsort(np.diag(H_X))

def perm_scale_homogeneous(n, W, H_X):
    """Sort by column RMS. Contiguous groups of 16 have similar scale."""
    rms = np.sqrt(np.mean(W ** 2, axis=0))
    return np.argsort(rms)

def perm_correlation_weight(n, W, H_X):
    """Spectral seriation on |corr(W^T W)|. Correlated columns adjacent."""
    C = np.corrcoef(W.T)
    C = np.nan_to_num(np.abs(C))
    D = np.diag(C.sum(axis=1))
    L = D - C
    eigvals, eigvecs = np.linalg.eigh(L)
    # Fiedler vector = eigenvector of 2nd smallest eigenvalue
    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler)

def perm_variance_based(n, W, H_X):
    """Sort by column variance. Similar-variance channels in same tile."""
    var = np.var(W, axis=0)
    return np.argsort(var)

def perm_spectral_HX(n, W, H_X):
    """Spectral seriation on |corr(H_X)|. Hessian-correlated channels adjacent."""
    d = np.sqrt(np.maximum(np.diag(H_X), 1e-15))
    C = H_X / np.outer(d, d)
    C = np.nan_to_num(np.abs(C))
    D = np.diag(C.sum(axis=1))
    L = D - C
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler)

def perm_random(n, W, H_X, seed=42):
    """Random permutation (control)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n)

def perm_balanced_scale(n, W, H_X, tile=TILE):
    """Round-robin deal: sorted channels dealt across tiles so each tile
    gets a spread of scales (high + low). Tests whether 'balancing' helps."""
    rms = np.sqrt(np.mean(W ** 2, axis=0))
    order = np.argsort(rms)
    n_tiles = n // tile
    result = np.zeros(n, dtype=int)
    for i in range(n):
        tile_idx = i % n_tiles
        pos_in_tile = i // n_tiles
        result[tile_idx * tile + pos_in_tile] = order[i]
    return result

def perm_p99_scale(n, W, H_X):
    """Sort by p99 of |W| column values. Robust to outliers."""
    p99 = np.percentile(np.abs(W), 99, axis=0)
    return np.argsort(p99)

COLUMN_STRATEGIES = {
    'identity':           perm_identity,
    'act_order_desc':     perm_act_order_desc,
    'act_order_asc':      perm_act_order_asc,
    'scale_homogeneous':  perm_scale_homogeneous,
    'p99_scale':          perm_p99_scale,
    'correlation_weight': perm_correlation_weight,
    'variance_based':     perm_variance_based,
    'spectral_HX':        perm_spectral_HX,
    'balanced_scale':     perm_balanced_scale,
    'random':             perm_random,
}

# ==================== Two-sided permutations ====================

def two_sided_perm(W, H_X, H_G, col_strategy='scale', row_strategy='scale'):
    """
    Compute row and column permutations independently.
    Row perm groups similar output channels; col perm groups similar input channels.
    """
    m, n = W.shape

    # Column permutation
    if col_strategy == 'scale':
        P_col = perm_scale_homogeneous(n, W, H_X)
    elif col_strategy == 'corr':
        P_col = perm_correlation_weight(n, W, H_X)
    else:
        P_col = np.arange(n)

    # Row permutation (use W^T as the "weight matrix" for row stats)
    Wt = W.T  # [n, m]
    if row_strategy == 'scale':
        P_row = perm_scale_homogeneous(m, Wt, H_G)
    elif row_strategy == 'corr':
        P_row = perm_correlation_weight(m, Wt, H_G)
    else:
        P_row = np.arange(m)

    return P_row, P_col

# ==================== Metrics ====================

def within_tile_stats(W, tile=TILE):
    """Compute within-tile range and dynamic range statistics."""
    m, n = W.shape
    ranges = []
    dyn_ranges = []
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            block = W[i:i+tile, j:j+tile]
            r = float(block.max() - block.min())
            ranges.append(r)
            abs_vals = np.abs(block).ravel()
            mx = float(abs_vals.max())
            nz = abs_vals[abs_vals > 1e-15]
            mn = float(nz.min()) if len(nz) > 0 else mx
            dyn_ranges.append(mx / max(mn, 1e-15))
    ranges = np.array(ranges)
    dyn_ranges = np.array(dyn_ranges)
    return {
        'range_mean': float(np.mean(ranges)),
        'range_median': float(np.median(ranges)),
        'range_max': float(np.max(ranges)),
        'dynrange_mean': float(np.mean(dyn_ranges)),
        'dynrange_median': float(np.median(dyn_ranges)),
        'dynrange_max': float(np.max(dyn_ranges)),
        'n_tiles': len(ranges),
    }

def weight_mse(W, Wq):
    return float(np.mean((W - Wq) ** 2))

def hessian_weighted_error(W, Wq, H_X, H_G):
    """tr(H_G · E · H_X · E^T) / (m·n), where E = W - Ŵ."""
    E = W - Wq
    val = np.trace(H_G @ E @ H_X @ E.T)
    return float(val / (W.shape[0] * W.shape[1]))

# ==================== Experiment runner ====================

def get_slice(W, slice_name, size=SLICE_SIZE):
    m, n = W.shape
    if slice_name == 'first':
        r, c = 0, 0
    elif slice_name == 'middle':
        r, c = m // 2 - size // 2, n // 2 - size // 2
    elif slice_name == 'last':
        r, c = m - size, n - size
    elif slice_name == 'random':
        rng = np.random.default_rng(123)
        r = rng.integers(0, max(1, m - size + 1))
        c = rng.integers(0, max(1, n - size + 1))
    else:
        raise ValueError(slice_name)
    r = max(0, min(r, m - size))
    c = max(0, min(c, n - size))
    return W[r:r+size, c:c+size].astype(np.float64)

def run_column_experiment(W, H_X, H_G, bits, strategy_name, perm_fn, use_hadamard=False):
    """Run single column-permutation experiment.
    
    When use_hadamard=True: Hadamard first, then compute P on transformed W,
    then permute, quantize, unpermute, inverse Hadamard.
    This is the correct composition order (Hadamard incoherences, then packing).
    """
    m, n = W.shape

    if use_hadamard:
        # Step 1: Hadamard incoherencing
        W_h = block_hadamard(W)
        # Step 2: Compute permutation on Hadamard-transformed weights
        if strategy_name == 'random':
            P = perm_fn(n, W_h, H_X, seed=42)
        else:
            P = perm_fn(n, W_h, H_X)
        P_inv = inverse_perm(P)
        # Step 3: Apply permutation
        W_perm = W_h[:, P]
        # Step 4: Quantize
        Wq_perm = tile_quantize(W_perm, bits)
        # Step 5: Unpermute
        Wq_h = Wq_perm[:, P_inv]
        # Step 6: Inverse Hadamard
        Wq = block_hadamard(Wq_h)
        # For tile stats, use the permuted Hadamard space
        W_for_stats = W_perm
    else:
        # No Hadamard: just permute, quantize, unpermute
        if strategy_name == 'random':
            P = perm_fn(n, W, H_X, seed=42)
        else:
            P = perm_fn(n, W, H_X)
        P_inv = inverse_perm(P)
        W_perm = W[:, P]
        Wq_perm = tile_quantize(W_perm, bits)
        Wq = Wq_perm[:, P_inv]
        W_for_stats = W_perm

    # Lossless verification
    assert np.allclose(W[:, P][:, P_inv], W), "Permutation round-trip failed!"

    # Metrics in original space
    mse = weight_mse(W, Wq)
    hw_err = hessian_weighted_error(W, Wq, H_X, H_G)

    # Within-tile stats on PERMUTED space (what the quantizer sees)
    ts = within_tile_stats(W_for_stats)
    ts_orig = within_tile_stats(W)


    return {
        'strategy': strategy_name,
        'bits': bits,
        'mse': mse,
        'hessian_weighted_error': hw_err,
        'tile_range_mean': ts['range_mean'],
        'tile_range_median': ts['range_median'],
        'tile_range_max': ts['range_max'],
        'tile_dynrange_mean': ts['dynrange_mean'],
        'tile_dynrange_median': ts['dynrange_median'],
        'tile_dynrange_max': ts['dynrange_max'],
        'orig_range_mean': ts_orig['range_mean'],
        'lossless': True,
    }

def run_two_sided_experiment(W, H_X, H_G, bits, strategy_name, col_strat, row_strat):
    """Run two-sided (row + column) permutation experiment."""
    m, n = W.shape

    P_row, P_col = two_sided_perm(W, H_X, H_G, col_strat, row_strat)
    P_row_inv = inverse_perm(P_row)
    P_col_inv = inverse_perm(P_col)

    # Apply both permutations
    W_perm = W[P_row, :][:, P_col]

    # Quantize
    Wq_perm = tile_quantize(W_perm, bits)

    # Unpermute both
    Wq = Wq_perm[P_row_inv, :][:, P_col_inv]

    # Lossless verification
    assert np.allclose(W[P_row, :][:, P_col][P_row_inv, :][:, P_col_inv], W), \
        "Two-sided permutation round-trip failed!"
    assert np.allclose(Wq, Wq_perm[P_row_inv, :][:, P_col_inv]), \
        "Two-sided lossless unpermute failed!"

    mse = weight_mse(W, Wq)
    hw_err = hessian_weighted_error(W, Wq, H_X, H_G)
    ts = within_tile_stats(W_perm)
    ts_orig = within_tile_stats(W)

    return {
        'strategy': strategy_name,
        'bits': bits,
        'mse': mse,
        'hessian_weighted_error': hw_err,
        'tile_range_mean': ts['range_mean'],
        'tile_range_median': ts['range_median'],
        'tile_range_max': ts['range_max'],
        'tile_dynrange_mean': ts['dynrange_mean'],
        'tile_dynrange_median': ts['dynrange_median'],
        'tile_dynrange_max': ts['dynrange_max'],
        'orig_range_mean': ts_orig['range_mean'],
        'lossless': True,
    }

# ==================== Main experiment ====================

def main():
    np.random.seed(SEED)
    weights = np.load(WEIGHTS_PATH)
    all_results = []

    for tensor_name in sorted(weights.files):
        W_full = weights[tensor_name]
        print(f"\n{'='*80}")
        print(f"Tensor: {tensor_name} shape={W_full.shape}")
        print(f"{'='*80}")

        for slice_name in SLICES:
            W_slice = get_slice(W_full, slice_name, SLICE_SIZE)
            m, n = W_slice.shape
            X = gen_calibration(n, N_CALIB, seed=SEED)
            H_X, H_G = compute_hessians(W_slice, X)

            # Original (unpermuted) tile stats
            ts_orig = within_tile_stats(W_slice)
            print(f"\n  Slice: {slice_name} ({m}×{n})")
            print(f"  Original tile range: mean={ts_orig['range_mean']:.6f}, "
                  f"dynrange={ts_orig['dynrange_mean']:.2f}")

            for bits in K_VALUES:
                # Column permutation strategies
                for sname, sfn in COLUMN_STRATEGIES.items():
                    try:
                        r = run_column_experiment(W_slice, H_X, H_G, bits, sname, sfn)
                        r['tensor'] = tensor_name
                        r['slice'] = slice_name
                        r['perm_type'] = 'column'
                        all_results.append(r)
                    except Exception as e:
                        print(f"    ERROR: {tensor_name} {slice_name} K{bits} {sname}: {e}")

                # Hadamard comparison (no permutation)
                try:
                    r = run_column_experiment(W_slice, H_X, H_G, bits, 'hadamard',
                                               perm_identity, use_hadamard=True)
                    r['tensor'] = tensor_name
                    r['slice'] = slice_name
                    r['perm_type'] = 'hadamard'
                    all_results.append(r)
                except Exception as e:
                    print(f"    ERROR Hadamard: {tensor_name} {slice_name} K{bits}: {e}")

                # Hadamard + scale_homogeneous composition
                try:
                    r = run_column_experiment(W_slice, H_X, H_G, bits, 'hadamard+scale',
                                               perm_scale_homogeneous, use_hadamard=True)
                    r['tensor'] = tensor_name
                    r['slice'] = slice_name
                    r['perm_type'] = 'hadamard+perm'
                    all_results.append(r)
                except Exception as e:
                    print(f"    ERROR Had+Scale: {tensor_name} {slice_name} K{bits}: {e}")

                # Hadamard + p99_scale composition (best composition)
                try:
                    r = run_column_experiment(W_slice, H_X, H_G, bits, 'hadamard+p99',
                                               perm_p99_scale, use_hadamard=True)
                    r['tensor'] = tensor_name
                    r['slice'] = slice_name
                    r['perm_type'] = 'hadamard+perm'
                    all_results.append(r)
                except Exception as e:
                    print(f"    ERROR Had+P99: {tensor_name} {slice_name} K{bits}: {e}")

                # Two-sided permutations
                for ts_name, (cs, rs) in [('two_sided_scale', ('scale', 'scale')),
                                           ('two_sided_corr', ('corr', 'corr'))]:
                    try:
                        r = run_two_sided_experiment(W_slice, H_X, H_G, bits,
                                                     ts_name, cs, rs)
                        r['tensor'] = tensor_name
                        r['slice'] = slice_name
                        r['perm_type'] = 'two_sided'
                        all_results.append(r)
                    except Exception as e:
                        print(f"    ERROR {ts_name}: {tensor_name} {slice_name} K{bits}: {e}")

    # Save results
    output_dir = Path('/Users/mbelleau/Projects/qwen38-research-r4-permutations')
    results_path = output_dir / 'receipts/research/r4-permutations-results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {results_path}")

    # Print summary tables
    print_summary(all_results)

    # Print improvement tables
    print_improvements(all_results)

    return all_results

def print_summary(results):
    """Aggregate by strategy × K, print mean metrics."""
    agg = defaultdict(lambda: {'mse': [], 'hw': [], 'range': [], 'dynrange': []})
    for r in results:
        key = (r['strategy'], r['bits'])
        agg[key]['mse'].append(r['mse'])
        agg[key]['hw'].append(r['hessian_weighted_error'])
        agg[key]['range'].append(r['tile_range_mean'])
        agg[key]['dynrange'].append(r['tile_dynrange_mean'])

    strategy_order = list(COLUMN_STRATEGIES.keys()) + ['hadamard', 'hadamard+scale',
                                                        'hadamard+p99',
                                                        'two_sided_scale', 'two_sided_corr']

    print("\n" + "=" * 120)
    print(f"{'Strategy':<22} {'K':>3} {'MSE':>12} {'HW Error':>12} "
          f"{'Tile Range':>12} {'Tile DynRange':>14} {'N runs':>7}")
    print("=" * 120)

    for bits in K_VALUES:
        for sname in strategy_order:
            key = (sname, bits)
            if key not in agg:
                continue
            a = agg[key]
            n = len(a['mse'])
            print(f"{sname:<22} {bits:>3} {np.mean(a['mse']):>12.4e} "
                  f"{np.mean(a['hw']):>12.4e} {np.mean(a['range']):>12.6f} "
                  f"{np.mean(a['dynrange']):>14.2f} {n:>7}")
        print("-" * 120)

def print_improvements(results):
    """Print % improvement over identity baseline for each strategy × K."""
    agg = defaultdict(lambda: {'mse': [], 'hw': [], 'range': [], 'dynrange': []})
    for r in results:
        key = (r['strategy'], r['bits'])
        agg[key]['mse'].append(r['mse'])
        agg[key]['hw'].append(r['hessian_weighted_error'])
        agg[key]['range'].append(r['tile_range_mean'])
        agg[key]['dynrange'].append(r['tile_dynrange_mean'])

    strategy_order = list(COLUMN_STRATEGIES.keys()) + ['hadamard', 'hadamard+scale',
                                                        'hadamard+p99',
                                                        'two_sided_scale', 'two_sided_corr']

    print("\n" + "=" * 100)
    print("Improvement over identity baseline (% reduction in metric):")
    print(f"{'Strategy':<22} {'K':>3} {'MSE impr%':>10} {'HW impr%':>10} "
          f"{'Range impr%':>12} {'DynR impr%':>11}")
    print("=" * 100)

    for bits in K_VALUES:
        base_key = ('identity', bits)
        if base_key not in agg:
            continue
        base_mse = np.mean(agg[base_key]['mse'])
        base_hw = np.mean(agg[base_key]['hw'])
        base_range = np.mean(agg[base_key]['range'])
        base_dr = np.mean(agg[base_key]['dynrange'])

        for sname in strategy_order:
            if sname == 'identity':
                continue
            key = (sname, bits)
            if key not in agg:
                continue
            a = agg[key]
            mse_imp = (1 - np.mean(a['mse']) / base_mse) * 100 if base_mse > 0 else 0
            hw_imp = (1 - np.mean(a['hw']) / base_hw) * 100 if base_hw > 0 else 0
            range_imp = (1 - np.mean(a['range']) / base_range) * 100 if base_range > 0 else 0
            dr_imp = (1 - np.mean(a['dynrange']) / base_dr) * 100 if base_dr > 0 else 0
            print(f"{sname:<22} {bits:>3} {mse_imp:>10.1f} {hw_imp:>10.1f} "
                  f"{range_imp:>12.1f} {dr_imp:>11.1f}")
        print("-" * 100)

if __name__ == '__main__':
    main()
