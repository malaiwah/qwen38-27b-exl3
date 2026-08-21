#!/usr/bin/env python3
"""
R12-AlphaSweep: Systematic α sweep post-rotation (v3 — final, reviewer-revised).

The cross-review identified that α=1.0 (paper-faithful) was never tested
post-rotation. R2 showed α=1.0 beats α=0.25 unrotated (34/36) on the
ASYMMETRIC error ||Wq·X - W·X̃||². R9 only tested α=0 and α=0.25
post-rotation. This experiment tests the full α range post-rotation and
unrotated for direct comparison.

v2 fixes (reviewer-driven):
  - Stable seeds (no hash())
  - Structured drift with non-zero cross-covariance D
  - Separate calibration and evaluation sets
  - Reports both symmetric HWE and asymmetric GPTAQ error
  - K-dependent P-matrix analysis
  - Records ||D||, ||P||, Cholesky off-diagonal mass
  - Per-slice summary (no cross-slice grouping)
  - Zero-range tile fix
  - Scoped conclusions

Experiment matrix:
  - 4 real tensors (L0/L55 gate+down), 3 slices each
  - K ∈ {3, 4, 5, 6}
  - α ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
  - 2 conditions: unrotated, BiIP+Hadamard rotated
  - 2 orderings: natural, act-order (descending diag(H_X))
  - Metrics: HWE tr(H_G E H_X E^T), raw MSE, asymmetric error ||Wq·X - W·X̃||²
  - RTN baseline for sanity check
"""

import numpy as np
import json
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r12-alpha-sweep/receipts/research/r12-alpha-sweep-results.json"

ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
K_VALUES = [3, 4, 5, 6]
TILE = 16
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
TENSOR_SEEDS = {"L0_gate": 100, "L0_down": 200, "L55_gate": 300, "L55_down": 400}
N_SLICES = 3
SLICE_SIZE = 128
N_CALIB = 512       # calibration samples (for Hessian, P-matrix, act-order)
N_EVAL = 512        # evaluation samples (for HWE, asymmetric error)
DAMPING = 0.01

# ============================================================================
# Quantizer — per-tile (16×16) uniform, used for ALL arms
# ============================================================================

def quantize_uniform_1d(w, bits):
    nl = 2 ** bits
    lo = float(w.min())
    hi = float(w.max())
    if hi - lo < 1e-15:
        return np.full_like(w, lo)
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_tiles(W, bits, tile):
    """Per-tile uniform quantization. Returns quantized W."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            r1, c1 = min(i + tile, m), min(j + tile, n)
            Wq[i:r1, j:c1] = quantize_uniform_1d(W[i:r1, j:c1], bits)
    return Wq


# ============================================================================
# Metrics
# ============================================================================

def hessian_weighted_error(E, H_G, H_X):
    """tr(H_G · E · H_X · E^T). E is (m, n), H_G is (m, m), H_X is (n, n)."""
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))


def asymmetric_error(W, Wq, X_eval, Xt_eval):
    """||Wq·X - W·X̃||_F^2 / (m·k) — the GPTAQ asymmetric objective.
    Uses evaluation calibration (separate from calibration used for P-matrix)."""
    m = Wq.shape[0]
    k = X_eval.shape[1]
    return float(np.mean((Wq @ X_eval - W @ Xt_eval) ** 2))


# ============================================================================
# BiIP diagonal balancing (from R3)
# ============================================================================

def biip_scaling(W, H_X, H_G):
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_G = np.diag(sg_diag)
    W_t = S_G @ W @ S_X
    return S_G, S_X, W_t


# ============================================================================
# Hadamard transform (from R3)
# ============================================================================

def hadamard_matrix(n):
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n).astype(np.float64)
    return H @ np.diag(signs)


# ============================================================================
# Rotation pipeline (BiIP + Hadamard both sides, from R3)
# ============================================================================

def apply_rotation(W, H_X, H_G, rng):
    """Apply BiIP + signed randomized Hadamard on both sides.
    Returns (W_t, H_X_t, H_G_t, U, V, S_G, S_X)."""
    d_out, d_in = W.shape
    S_G, S_X, W_t = biip_scaling(W, H_X, H_G)
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    H_X_t = S_X_inv @ H_X @ S_X_inv
    H_G_t = S_G_inv @ H_G @ S_G_inv
    V = signed_random_hadamard(d_in, rng)
    W_t = W_t @ V.T
    H_X_t = V @ H_X_t @ V.T
    U = signed_random_hadamard(d_out, rng)
    W_t = U @ W_t
    H_G_t = U @ H_G_t @ U.T
    return W_t, H_X_t, H_G_t, U, V, S_G, S_X


def inverse_rotation(Q_t, U, V, S_G, S_X):
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ Q_t @ V @ S_X_inv


# ============================================================================
# Correct Cholesky (from R9, verified)
# ============================================================================

def inv_cholesky(H, damping):
    """Upper triangular U such that U^T @ U = inv(H + damping*I).
    Correct GPTQ convention: U = chol(inv(H+λI)).T (upper triangular)."""
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
    return U


# ============================================================================
# GPTAQ correction (from R9, per-tile quantizer, correct Cholesky)
# ============================================================================

def gptaq_correction(W, X, Xt, bits, tile, damping, alpha=0.25):
    """GPTAQ correction with per-tile (16×16) quantizer, MATCHED with RTN.
    α=0: standard GPTQ (no P-matrix correction)
    α>0: GPTAQ with P-matrix asymmetric correction
    """
    m, n = W.shape
    Ww = W.copy().astype(np.float64)
    Q = np.zeros_like(Ww)
    H = X @ X.T
    L = inv_cholesky(H, damping)

    dX = Xt - X
    D = dX @ X.T
    if alpha > 0:
        P = alpha * (np.triu(D @ L.T, 1) @ L)
    else:
        P = np.zeros((n, n))

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    for ct in range(n_tn):
        c0 = ct * tile
        c1 = min(c0 + tile, n)
        B = c1 - c0

        codebooks = []
        for ti in range(n_tm):
            r0 = ti * tile
            r1 = min(r0 + tile, m)
            tile_data = Ww[r0:r1, c0:c1]
            nl = 2 ** bits
            lo = float(tile_data.min())
            hi = float(tile_data.max())
            if hi - lo < 1e-15:
                step = 0.0
            else:
                step = (hi - lo) / (nl - 1)
            codebooks.append((r0, r1, lo, step, bits))

        def apply_frozen_codebook(col_data):
            q = np.zeros_like(col_data)
            for r0, r1, lo, step, k in codebooks:
                if step == 0.0:
                    q[r0:r1] = lo  # constant tile → return lo (matched with RTN)
                else:
                    nl = 2 ** k
                    q[r0:r1] = np.clip(np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
            return q

        E = np.zeros((m, B))
        W_pre_block = np.zeros((m, B))
        for j in range(B):
            c = c0 + j
            w_pre = Ww[:, c].copy()
            W_pre_block[:, j] = w_pre
            Q[:, c] = apply_frozen_codebook(w_pre)
            e = w_pre - Q[:, c]
            E[:, j] = e / L[c, c]
            end = min(c0 + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            if alpha > 0:
                Ww[:, c:end] += np.outer(w_pre, P[c, c:end])

        if c1 < n:
            Ww[:, c1:] -= E @ L[c0:c1, c1:]
            if alpha > 0:
                Ww[:, c1:] += W_pre_block @ P[c0:c1, c1:]

    return Q


# ============================================================================
# Act-order (from R7: descending diag(H_X) column order)
# ============================================================================

def act_order_perm(H_X):
    return np.argsort(-np.diag(H_X))


# ============================================================================
# Calibration and Hessian generation
# ============================================================================

def gen_calibration(W, seed):
    """Generate synthetic calibration with structured drift.

    Following R2's convention: Xt (FP-flow) is generated first, then
    X = Xt + drift. The drift is structured: larger on outlier channels,
    creating non-zero off-diagonal cross-covariance D = (Xt - X) @ X^T.

    Returns X_calib, Xt_calib (for Hessian/P-matrix) and X_eval, Xt_eval (for metrics).
    """
    m, n = W.shape
    rng = np.random.default_rng(seed)

    # Calibration set (for Hessian, P-matrix, act-order)
    Xt_calib = rng.standard_normal((n, N_CALIB)) * 0.5
    outlier_mask = rng.random(n) < 0.05
    Xt_calib[outlier_mask] *= 5.0

    # Structured drift: correlated with activation magnitude
    # Drift is larger on channels with larger activation variance
    drift_calib = rng.standard_normal((n, N_CALIB)) * 0.02
    # Scale drift by channel activation magnitude (creates non-zero off-diagonal D)
    channel_scale = np.sqrt(np.mean(Xt_calib ** 2, axis=1, keepdims=True))
    drift_calib *= (1.0 + channel_scale)  # more drift on high-activation channels
    X_calib = Xt_calib + drift_calib

    # Evaluation set (separate, for metrics only)
    rng_eval = np.random.default_rng(seed + 9999)
    Xt_eval = rng_eval.standard_normal((n, N_EVAL)) * 0.5
    Xt_eval[rng_eval.random(n) < 0.05] *= 5.0
    drift_eval = rng_eval.standard_normal((n, N_EVAL)) * 0.02
    channel_scale_eval = np.sqrt(np.mean(Xt_eval ** 2, axis=1, keepdims=True))
    drift_eval *= (1.0 + channel_scale_eval)
    X_eval = Xt_eval + drift_eval

    return X_calib, Xt_calib, X_eval, Xt_eval


def compute_hessians(W, X):
    """Compute H_G (output covariance proxy) and H_X (input Hessian)."""
    m, n = W.shape
    N = X.shape[1]
    H_X = X @ X.T / N
    Y = W @ X
    H_G = Y @ Y.T / N
    H_X *= n / np.trace(H_X)
    H_G *= m / np.trace(H_G)
    H_X += 1e-6 * np.eye(n)
    H_G += 1e-6 * np.eye(m)
    return H_G, H_X


# ============================================================================
# Real weight loading
# ============================================================================

def load_real_weight_slices():
    data = np.load(WEIGHTS_PATH)
    slices = {}
    for name in TENSOR_NAMES:
        W = data[name].astype(np.float64)
        m, n = W.shape
        tensor_slices = []
        for s in range(N_SLICES):
            r0 = (s * 37) % max(1, m - SLICE_SIZE)
            c0 = (s * 53) % max(1, n - SLICE_SIZE)
            tensor_slices.append(W[r0:r0 + SLICE_SIZE, c0:c0 + SLICE_SIZE].copy())
        slices[name] = tensor_slices
    return slices


# ============================================================================
# Transform calibration to rotated space
# ============================================================================

def transform_calibration(X, Xt, V, S_X):
    """Transform calibration to rotated input space.
    X_t = V @ S_X_inv @ X, giving X_t @ X_t^T / N = V @ S_X_inv @ (X @ X^T / N) @ S_X_inv @ V^T.
    Note: this gives the raw covariance transform; the normalized+regularized
    H_X_t from apply_rotation differs by the normalization factor and regularization."""
    S_X_inv = np.linalg.inv(S_X)
    X_t = V @ S_X_inv @ X
    Xt_t = V @ S_X_inv @ Xt
    return X_t, Xt_t


# ============================================================================
# Diagnostics: measure P-matrix and Cholesky structure
# ============================================================================

def compute_diagnostics(W, X, Xt, H_X, damping, rotated, rotation_data=None):
    """Compute ||D||, ||P|| (at α=1), Cholesky structure, and isolation of
    BiIP vs Hadamard effects on D and L.

    Key insight from reviewer: Hadamard (orthogonal) preserves ||D|| exactly;
    only BiIP scaling (non-orthogonal S_X^{-1}) changes ||D||. We measure
    both the combined effect and the BiIP-only effect."""
    n = H_X.shape[0]

    if rotated and rotation_data is not None:
        W_t, H_X_t, _, U, V, S_G, S_X = rotation_data
        X_t, Xt_t = transform_calibration(X, Xt, V, S_X)
        X_work, Xt_work = X_t, Xt_t
        H_work = X_work @ X_work.T

        # Isolate BiIP-only effect (without Hadamard)
        S_X_inv = np.linalg.inv(S_X)
        X_biip = S_X_inv @ X
        Xt_biip = S_X_inv @ Xt
        H_biip = X_biip @ X_biip.T
    else:
        X_work, Xt_work = X, Xt
        H_work = X_work @ X_work.T

    L = inv_cholesky(H_work, damping)

    dX = Xt_work - X_work
    D = dX @ X_work.T
    P = np.triu(D @ L.T, 1) @ L  # P at α=1

    # Absolute norms (not just ratios)
    chol_offdiag_abs = np.linalg.norm(np.triu(L, 1), 'fro')
    chol_total = np.linalg.norm(L, 'fro')
    chol_ratio = chol_offdiag_abs / (chol_total + 1e-15)
    triu_DL = np.triu(D @ L.T, 1)

    result = {
        "norm_D": float(np.linalg.norm(D, 'fro')),
        "norm_P_alpha1": float(np.linalg.norm(P, 'fro')),
        "norm_triu_DL": float(np.linalg.norm(triu_DL, 'fro')),
        "norm_L": float(chol_total),
        "norm_triu_L": float(chol_offdiag_abs),
        "chol_offdiag_ratio": float(chol_ratio),
        "norm_dX": float(np.linalg.norm(dX, 'fro')),
    }

    # BiIP-only diagnostics (for rotated case)
    if rotated and rotation_data is not None:
        L_biip = inv_cholesky(H_biip, damping)
        dX_biip = Xt_biip - X_biip
        D_biip = dX_biip @ X_biip.T
        result["norm_D_biip_only"] = float(np.linalg.norm(D_biip, 'fro'))
        result["norm_L_biip_only"] = float(np.linalg.norm(L_biip, 'fro'))
        result["norm_triu_L_biip_only"] = float(np.linalg.norm(np.triu(L_biip, 1), 'fro'))

    return result


# ============================================================================
# Experiment runner
# ============================================================================

def run_single(W, X_calib, Xt_calib, X_eval, Xt_eval,
               H_G_calib, H_X_calib, H_G_eval, H_X_eval,
               bits, alpha, tile, damping,
               rotated, act_order, rotation_data=None):
    """Run a single experiment arm. Returns (hwe, mse, asym_err).

    HWE uses evaluation Hessians (H_G_eval, H_X_eval) — not calibration.
    Asymmetric error uses original-coordinate X_eval, Xt_eval.
    Calibration data (X_calib, Xt_calib, H_G_calib, H_X_calib) is used for
    GPTQ Hessian, P-matrix, and act-order only.
    """
    if rotated and rotation_data is not None:
        W_t, H_X_t, _, U, V, S_G, S_X = rotation_data
        X_c, Xt_c = transform_calibration(X_calib, Xt_calib, V, S_X)
        W_work = W_t
        X_work, Xt_work = X_c, Xt_c
    else:
        W_work = W
        X_work, Xt_work = X_calib, Xt_calib

    if act_order:
        if rotated and rotation_data is not None:
            perm = act_order_perm(H_X_t)
        else:
            perm = act_order_perm(H_X_calib)
        W_work = W_work[:, perm]
        X_work = X_work[perm, :]
        Xt_work = Xt_work[perm, :]

    Q_work = gptaq_correction(W_work, X_work, Xt_work, bits, tile, damping, alpha)

    if act_order:
        Q_orig = np.empty_like(Q_work)
        Q_orig[:, perm] = Q_work
        Q_work = Q_orig

    if rotated and rotation_data is not None:
        W_hat = inverse_rotation(Q_work, U, V, S_G, S_X)
    else:
        W_hat = Q_work

    E = W - W_hat
    # HWE uses evaluation Hessians (held-out)
    hwe = hessian_weighted_error(E, H_G_eval, H_X_eval)
    mse = weight_mse(E)
    # Asymmetric error ALWAYS in original coordinates
    asym = asymmetric_error(W, W_hat, X_eval, Xt_eval)
    return hwe, mse, asym


def run_rtn(W, H_G_eval, H_X_eval, X_eval, Xt_eval, bits, tile,
            rotated, rotation_data=None):
    """RTN baseline. HWE uses evaluation Hessians. Asym in original coords."""
    if rotated and rotation_data is not None:
        W_t, _, _, U, V, S_G, S_X = rotation_data
        Q_t = quantize_tiles(W_t, bits, tile)
        W_hat = inverse_rotation(Q_t, U, V, S_G, S_X)
    else:
        W_hat = quantize_tiles(W, bits, tile)
    E = W - W_hat
    hwe = hessian_weighted_error(E, H_G_eval, H_X_eval)
    mse = weight_mse(E)
    asym = asymmetric_error(W, W_hat, X_eval, Xt_eval)
    return hwe, mse, asym


def run_experiment():
    print("=" * 80)
    print("R12-AlphaSweep v3: Systematic α Sweep Post-Rotation (final, reviewer-revised)")
    print("α ∈", ALPHA_GRID)
    print("K ∈", K_VALUES, "| Tile =", TILE, "| Tensors:", TENSOR_NAMES)
    print("Slices per tensor:", N_SLICES, "| Slice size:", SLICE_SIZE)
    print("Calib:", N_CALIB, "| Eval:", N_EVAL, "| Damping:", DAMPING)
    print("=" * 80)

    weights = load_real_weight_slices()
    all_results = []
    all_diagnostics = []

    for tname in TENSOR_NAMES:
        base_seed = TENSOR_SEEDS[tname]
        for si, W in enumerate(weights[tname]):
            seed = base_seed + si * 10
            X_calib, Xt_calib, X_eval, Xt_eval = gen_calibration(W, seed)
            H_G_calib, H_X_calib = compute_hessians(W, X_calib)
            H_G_eval, H_X_eval = compute_hessians(W, X_eval)

            rng = np.random.default_rng(seed)
            rotation_data = apply_rotation(W, H_X_calib, H_G_calib, rng)


            # Diagnostics
            for rotated in [False, True]:
                diag = compute_diagnostics(W, X_calib, Xt_calib, H_X_calib, DAMPING,
                                           rotated, rotation_data if rotated else None)
                diag["tensor"] = tname
                diag["slice"] = si
                diag["rotated"] = rotated
                all_diagnostics.append(diag)

            for K in K_VALUES:
                # RTN baselines (not crossed with act-order)
                for rotated in [False, True]:
                    hwe, mse, asym = run_rtn(W, H_G_eval, H_X_eval, X_eval, Xt_eval, K, TILE,
                                             rotated, rotation_data if rotated else None)
                    all_results.append({
                        "tensor": tname, "slice": si, "K": K,
                        "arm": "RTN", "alpha": None,
                        "rotated": rotated, "act_order": False,
                        "hwe": hwe, "mse": mse, "asym": asym,
                    })

                for rotated in [False, True]:
                    for act_order in [False, True]:
                        for alpha in ALPHA_GRID:
                            hwe, mse, asym = run_single(
                                W, X_calib, Xt_calib, X_eval, Xt_eval,
                                H_G_calib, H_X_calib, H_G_eval, H_X_eval,
                                K, alpha, TILE, DAMPING,
                                rotated, act_order,
                                rotation_data if rotated else None
                            )
                            all_results.append({
                                "tensor": tname, "slice": si, "K": K,
                                "arm": "GPTAQ" if alpha > 0 else "GPTQ",
                                "alpha": alpha,
                                "rotated": rotated, "act_order": act_order,
                                "hwe": hwe, "mse": mse, "asym": asym,
                            })

            print(f"  [{tname} slice {si}] Done (seed={seed})")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "config": {
            "alpha_grid": ALPHA_GRID,
            "k_values": K_VALUES,
            "tensor_names": TENSOR_NAMES,
            "tensor_seeds": TENSOR_SEEDS,
            "n_slices": N_SLICES,
            "slice_size": SLICE_SIZE,
            "tile_size": TILE,
            "n_calib": N_CALIB,
            "n_eval": N_EVAL,
            "damping": DAMPING,
        },
        "results": all_results,
        "diagnostics": all_diagnostics,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Total experiments: {len(all_results)} (1536 α-arms + 96 RTN)")
    print(f"Diagnostics: {len(all_diagnostics)}")

    print_summary(all_results, all_diagnostics)
    return all_results


def print_summary(results, diagnostics):
    """Print key summary tables (per-slice grouping, K-dependent)."""
    # Group by (tensor, slice, K, rotated, act_order)
    groups = defaultdict(list)
    for r in results:
        if r["alpha"] is not None:
            key = (r["tensor"], r["slice"], r["K"], r["rotated"], r["act_order"])
            groups[key].append(r)

    # Optimal α distribution
    print("\n" + "=" * 80)
    print("OPTIMAL α DISTRIBUTION (per-slice, HWE)")
    print("=" * 80)
    for rotated in [False, True]:
        for act_order in [False, True]:
            alphas = []
            for tname in TENSOR_NAMES:
                for si in range(N_SLICES):
                    for K in K_VALUES:
                        key = (tname, si, K, rotated, act_order)
                        if key in groups:
                            best = min(groups[key], key=lambda r: r["hwe"])
                            alphas.append(best["alpha"])
            rot_s = "rotated" if rotated else "unrotated"
            ao_s = "act-order" if act_order else "natural"
            counts = {a: alphas.count(a) for a in ALPHA_GRID}
            print(f"\n  {rot_s} {ao_s} (n={len(alphas)}):")
            print(f"    mean={np.mean(alphas):.3f}, median={np.median(alphas):.3f}")
            for a in ALPHA_GRID:
                c = counts.get(a, 0)
                print(f"    α={a:>4.2f}: {c:>3} ({100*c/len(alphas):>5.1f}%) {'#' * c}")

    # Win rate: α=1.0 vs α=0.25 (per-slice)
    print("\n" + "=" * 80)
    print("WIN RATE: α=1.0 vs α=0.25 (per-slice, HWE)")
    print("=" * 80)
    for rotated in [False, True]:
        for act_order in [False, True]:
            w10, w025, ties = 0, 0, 0
            for tname in TENSOR_NAMES:
                for si in range(N_SLICES):
                    for K in K_VALUES:
                        key = (tname, si, K, rotated, act_order)
                        if key not in groups: continue
                        runs = groups[key]
                        a10 = next((r for r in runs if r["alpha"] == 1.0), None)
                        a025 = next((r for r in runs if r["alpha"] == 0.25), None)
                        if a10 and a025:
                            if a10["hwe"] < a025["hwe"] * 0.999: w10 += 1
                            elif a025["hwe"] < a10["hwe"] * 0.999: w025 += 1
                            else: ties += 1
            rot_s = "rotated" if rotated else "unrotated"
            ao_s = "act-order" if act_order else "natural"
            total = w10 + w025 + ties
            print(f"  {rot_s:>9} {ao_s:>10}: α=1.0 wins {w10:>2}/{total}, α=0.25 wins {w025:>2}/{total}, ties {ties:>2}/{total}")

    # Win rate: α=1.0 vs α=0.25 (asymmetric error)
    print("\nWIN RATE: α=1.0 vs α=0.25 (per-slice, ASYMMETRIC error)")
    for rotated in [False, True]:
        for act_order in [False, True]:
            w10, w025, ties = 0, 0, 0
            for tname in TENSOR_NAMES:
                for si in range(N_SLICES):
                    for K in K_VALUES:
                        key = (tname, si, K, rotated, act_order)
                        if key not in groups: continue
                        runs = groups[key]
                        a10 = next((r for r in runs if r["alpha"] == 1.0), None)
                        a025 = next((r for r in runs if r["alpha"] == 0.25), None)
                        if a10 and a025:
                            if a10["asym"] < a025["asym"] * 0.999: w10 += 1
                            elif a025["asym"] < a10["asym"] * 0.999: w025 += 1
                            else: ties += 1
            rot_s = "rotated" if rotated else "unrotated"
            ao_s = "act-order" if act_order else "natural"
            total = w10 + w025 + ties
            print(f"  {rot_s:>9} {ao_s:>10}: α=1.0 wins {w10:>2}/{total}, α=0.25 wins {w025:>2}/{total}, ties {ties:>2}/{total}")

    # P-matrix value: K-dependent
    print("\n" + "=" * 80)
    print("P-MATRIX VALUE: best α>0 vs α=0 (per-slice, HWE, K-DEPENDENT)")
    print("=" * 80)
    for rotated in [False, True]:
        for act_order in [False, True]:
            for K in K_VALUES:
                improvements = []
                for tname in TENSOR_NAMES:
                    for si in range(N_SLICES):
                        key = (tname, si, K, rotated, act_order)
                        if key not in groups: continue
                        runs = groups[key]
                        a0 = next((r for r in runs if r["alpha"] == 0.0), None)
                        best_nz = min((r for r in runs if r["alpha"] > 0), key=lambda r: r["hwe"], default=None)
                        if a0 and best_nz:
                            improvements.append((1 - best_nz["hwe"] / a0["hwe"]) * 100)
                rot_s = "rot" if rotated else "unr"
                ao_s = "AO" if act_order else "nat"
                if improvements:
                    pos = sum(1 for x in improvements if x > 0)
                    print(f"  {rot_s} {ao_s} K={K}: mean {np.mean(improvements):>+6.2f}%, positive {pos}/{len(improvements)}")

    # Sanity checks
    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)
    gptq_vs_rtn = []
    for r in results:
        if r["alpha"] == 0.0 and not r["act_order"]:
            rtn = next((x for x in results if x["tensor"] == r["tensor"] and x["slice"] == r["slice"]
                        and x["K"] == r["K"] and x["rotated"] == r["rotated"] and x["arm"] == "RTN"), None)
            if rtn:
                gptq_vs_rtn.append((1 - r["hwe"] / rtn["hwe"]) * 100)
    print(f"  GPTQ (α=0) vs RTN: mean HWE improvement {np.mean(gptq_vs_rtn):>+6.2f}% (positive {sum(1 for x in gptq_vs_rtn if x > 0)}/{len(gptq_vs_rtn)})")

    # Diagnostics: P-matrix and Cholesky structure (absolute norms)
    print("\n" + "=" * 80)
    print("DIAGNOSTICS: ||D||, ||L||, ||triu(L,1)||, ||P(α=1)|| (absolute norms)")
    print("=" * 80)
    for rotated in [False, True]:
        diags = [d for d in diagnostics if d["rotated"] == rotated]
        rot_s = "rotated" if rotated else "unrotated"
        print(f"\n  {rot_s} (mean over {len(diags)} cells):")
        print(f"    ||D||_F:          {np.mean([d['norm_D'] for d in diags]):.4f}")
        print(f"    ||dX||_F:         {np.mean([d['norm_dX'] for d in diags]):.4f}")
        print(f"    ||L||_F:          {np.mean([d['norm_L'] for d in diags]):.4f}")
        print(f"    ||triu(L,1)||_F:  {np.mean([d['norm_triu_L'] for d in diags]):.4f}")
        print(f"    ||triu(DL^T,1)||: {np.mean([d['norm_triu_DL'] for d in diags]):.4f}")
        print(f"    ||P(α=1)||_F:     {np.mean([d['norm_P_alpha1'] for d in diags]):.4f}")
        print(f"    Chol offdiag ratio: {np.mean([d['chol_offdiag_ratio'] for d in diags]):.4f}")
        if rotated:
            biip_diags = [d for d in diags if 'norm_D_biip_only' in d]
            if biip_diags:
                print(f"    [BiIP-only] ||D||_F:      {np.mean([d['norm_D_biip_only'] for d in biip_diags]):.4f}")
                print(f"    [BiIP-only] ||L||_F:       {np.mean([d['norm_L_biip_only'] for d in biip_diags]):.4f}")
                print(f"    [BiIP-only] ||triu(L,1)||: {np.mean([d['norm_triu_L_biip_only'] for d in biip_diags]):.4f}")

    # Ratios
    unrot_diags = [d for d in diagnostics if not d["rotated"]]
    rot_diags = [d for d in diagnostics if d["rotated"]]
    ratios = {}
    for field in ["norm_D", "norm_P_alpha1", "norm_L", "norm_triu_L", "norm_triu_DL"]:
        r = []
        for du in unrot_diags:
            dr = next((d for d in rot_diags if d["tensor"] == du["tensor"] and d["slice"] == du["slice"]), None)
            if dr and du.get(field, 0) > 1e-15:
                r.append(dr[field] / du[field])
        if r:
            ratios[field] = np.mean(r)
    print(f"\n  Ratio rotated/unrotated (mean):")
    for k, v in ratios.items():
        print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    results = run_experiment()
