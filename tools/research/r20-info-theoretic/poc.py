#!/usr/bin/env python3
"""
R20 — Information-Theoretic Bounds and Distributional Quantization (v2)
=====================================================================

Compute the theoretical limits of quantization for actual Qwen3.8-27B weight
distributions.  How close are our methods to optimal?  This is a ceiling
analysis that reveals untapped potential.

Revision v2 (reviewer-driven):
- Entropy computed in BITS (not nats). KDE leave-one-out (exclude self-kernel).
- Gaussian R-D is a REFERENCE curve, not a lower bound for non-Gaussian sources.
- Uniform quantization is NOT claimed Gaussian-optimal at fixed rate; Lloyd-Max
  is the fixed-rate scalar optimum. Uniform is optimal only in the entropy-
  constrained scalar regime.
- GPTQ uses alpha=1.0 (error feedback ENABLED), not alpha=0.
- Transformed Hessians use correct inverse-scaling: H_X' = V^T S_X^{-1} H_X S_X^{-1} V,
  H_G' = U S_G^{-1} H_G S_G^{-1} U^T.
- Channel capacity replaced by empirical mutual information H(Q|tile) for
  deterministic quantization (exact, not AWGN heuristic).
- BiIP scaling separated from orthogonal Hadamard; standardized entropy
  (negentropy) used to isolate Gaussianization from variance scaling.
- Lloyd-Max uses multistart (uniform + quantile init), aggregates ALL tiles.
- Entropy-constrained quantization: entropy-code the unchanged K-level indices
  and measure coded length, rather than requantizing at lower K.
- GGD fitted by MLE (scipy.stats.gennorm), not method-of-moments.
- High-resolution approximation evaluated per-tile (16x16), not whole slice.

All quantizers use the SAME per-tile (16×16) uniform quantizer primitive.
Correct Cholesky: U = chol(inv(H+λI)).T.
"""

import json
import os
import time
import warnings
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from scipy.special import logsumexp, gammaln, gamma as gamma_func
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "receipts", "research",
    "r20-info-theoretic-results.json",
)
FINDINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "docs", "research",
    "r20-info-theoretic-findings.md",
)

# ─── Configuration ────────────────────────────────────────────────────────────
TILE       = 16
SLICE      = 128
K_VALUES   = [2, 3, 4, 5, 6, 7, 8]
N_CALIB    = 512
TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
RNG_SEED   = 42

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_weights():
    data = np.load(WEIGHTS_PATH)
    return {k: data[k].astype(np.float64) for k in data.files}


def extract_slices(tensor, m=128, n=128, seed=42):
    M, N = tensor.shape
    slices = []
    slices.append(("first", tensor[:m, :n].copy()))
    r0, c0 = M // 2 - m // 2, N // 2 - n // 2
    slices.append(("mid", tensor[r0:r0 + m, c0:c0 + n].copy()))
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, max(1, M - m))
    c0 = rng.integers(0, max(1, N - n))
    slices.append(("rand", tensor[r0:r0 + m, c0:c0 + n].copy()))
    return slices


def gen_calibration(n_in, n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples)) * 0.1
    n_outliers = max(1, n_in // 20)
    outlier_rows = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_rows, :] *= 10.0
    return X


def compute_hessians(W, X):
    N = X.shape[1]
    H_X = (X @ X.T / N).astype(np.float64)
    Y = W @ X
    H_G = (Y @ Y.T / N).astype(np.float64)
    d_out, d_in = W.shape
    H_X *= d_in / max(np.trace(H_X), 1e-15)
    H_G *= d_out / max(np.trace(H_G), 1e-15)
    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)
    return H_G, H_X


# ═══════════════════════════════════════════════════════════════════════════════
# Quantizer (per-tile uniform, MATCHED for all arms)
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


def quantize_tile_indices(w, k):
    """Return integer quantization indices (not reconstructed values)."""
    if k <= 0:
        return np.zeros_like(w, dtype=int)
    nl = 2 ** k
    lo, hi = float(w.min()), float(w.max())
    step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
    if step == 0.0:
        return np.zeros_like(w, dtype=int)
    return np.clip(np.round((w - lo) / step), 0, nl - 1).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def hessian_weighted_error(E, H_G, H_X):
    return float(np.trace(H_G @ E @ H_X @ E.T))


def weight_mse(E):
    return float(np.mean(E ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Hadamard + BiIP
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
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((np.diag(H_X) / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((np.diag(H_G) / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    S_G = np.diag(sg_diag)
    return S_G, S_X, S_G @ W @ S_X


# ═══════════════════════════════════════════════════════════════════════════════
# GPTQ (correct Cholesky, alpha=1.0 by default)
# ═══════════════════════════════════════════════════════════════════════════════

def inv_cholesky(H, damping):
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    Hinv = np.linalg.inv(Hd)
    Hinv = (Hinv + Hinv.T) / 2
    try:
        U = np.linalg.cholesky(Hinv).T
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Hinv)
        eigvals = np.maximum(eigvals, 1e-12)
        U = (eigvecs @ np.diag(np.sqrt(eigvals))).T
    return np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)


def gptq_quantize(W, H_X, K, tile=TILE, alpha=1.0, damping=0.01):
    """GPTQ with per-tile quantizer, correct Cholesky.
    alpha=1.0 enables full error feedback (paper-faithful)."""
    m, n = W.shape
    U = inv_cholesky(H_X, damping)
    order = np.arange(n)

    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile

    # Frozen tile codebooks from original W
    tile_cb = {}
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            td = W[r0:r1, c0:c1]
            nl = 2 ** K
            lo, hi = float(td.min()), float(td.max())
            step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
            tile_cb[(ti, tj)] = (lo, step, K)

    def quantize_col(col_data, col_idx):
        q = np.zeros_like(col_data)
        tj = col_idx // tile
        for ti in range(n_tm):
            r0 = ti * tile
            r1 = min(r0 + tile, m)
            lo, step, k = tile_cb[(ti, tj)]
            if step == 0.0:
                q[r0:r1] = col_data[r0:r1]
            else:
                nl = 2 ** k
                q[r0:r1] = np.clip(np.round((col_data[r0:r1] - lo) / step), 0, nl - 1) * step + lo
        return q

    W_work = W.copy().astype(np.float64)
    Wq = np.zeros_like(W_work)

    for idx in range(n):
        q = order[idx]
        Wq[:, q] = quantize_col(W_work[:, q], q)
        e_q = W_work[:, q] - Wq[:, q]
        if idx < n - 1:
            remaining = order[idx + 1:]
            u_ii = U[idx, idx]
            if abs(u_ii) > 1e-15:
                update = alpha * np.outer(e_q, U[idx, idx + 1:] / u_ii)
                W_work[:, remaining] -= np.nan_to_num(update, nan=0.0, posinf=1e6, neginf=-1e6)

    return Wq


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 1: Shannon Lower Bound (corrected: bits, leave-one-out)
# ═══════════════════════════════════════════════════════════════════════════════

def differential_entropy_bits(w_flat):
    """Empirical differential entropy h(W) in BITS using leave-one-out KDE.
    Excludes the self-kernel term (i=j) for proper LOO estimation.
    Returns h in bits (divided by ln(2)).
    """
    if len(w_flat) < 10:
        return 0.0
    std = np.std(w_flat)
    if std < 1e-15:
        return 0.0
    n = len(w_flat)
    bw = 1.06 * std * n ** (-1/5)
    bw = max(bw, 1e-10)

    # Leave-one-out: f_{-i}(x_i) = 1/((n-1)*bw) * sum_{j!=i} K((x_i - x_j)/bw)
    log_f = np.zeros(n)
    chunk = max(1, 2048 // max(1, n))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = (w_flat[start:end, None] - w_flat[None, :]) / bw
        log_k = -0.5 * diff ** 2 - 0.5 * np.log(2 * np.pi) - np.log(bw)
        # Set self-terms to -inf so they're excluded from logsumexp
        for i_local in range(end - start):
            i_global = start + i_local
            log_k[i_local, i_global] = -np.inf
        # logsumexp over j != i, then subtract log(n-1) for normalization
        log_f[start:end] = logsumexp(log_k, axis=1) - np.log(n - 1)

    h_nats = -np.mean(log_f)
    h_bits = h_nats / np.log(2)  # convert nats to bits
    return h_bits


def shannon_lower_bound_bits(w_flat, D):
    """Shannon lower bound on rate (bits/sample) for distortion D (MSE).
    R(D) >= h(W) - (1/2) * log2(2*pi*e*D)
    h(W) must be in bits.
    """
    h = differential_entropy_bits(w_flat)
    if D <= 0:
        return np.inf
    return h - 0.5 * np.log2(2 * np.pi * np.e * D)


def gaussian_rd_D(var, K):
    """Gaussian R-D reference: D = sigma^2 / 2^(2K).
    This is the achievable R-D for a Gaussian source under optimal coding
    (vector quantization + entropy coding). It is NOT a lower bound for
    non-Gaussian sources — Gaussian is the LEAST compressible at fixed variance.
    Used as a REFERENCE curve, not a bound."""
    return var / (2 ** (2 * K))


def high_rate_scalar_bound(var, K):
    """Panter-Dite high-rate asymptotic heuristic for fixed-rate SCALAR Gaussian
    quantization: D ~ (pi*sqrt(3)/2) * sigma^2 * 2^(-2K) ≈ 2.7207 * sigma^2 / 2^(2K).
    This is an ASYMPTOTIC reference, not a rigorous finite-K bound.
    The quantizer is tile-adaptive; this uses full-slice variance."""
    return (np.pi * np.sqrt(3) / 2) * var / (2 ** (2 * K))


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 2: Lloyd-Max Optimality (corrected: multistart, all tiles)
# ═══════════════════════════════════════════════════════════════════════════════

def lloyd_max_quantize(w_flat, k, n_iters=200, tol=1e-14, n_starts=3, seed=42):
    """Lloyd-Max optimal scalar quantizer with multistart (seeded, distinct starts).
    Start 0: uniform spacing. Start 1: quantile spacing. Start 2: k-means++.
    Returns best (quantized_values, mse)."""
    nl = 2 ** k
    if nl >= len(w_flat):
        return w_flat.copy(), 0.0

    rng = np.random.default_rng(seed)
    best_mse = np.inf
    best_wq = None
    lo, hi = float(w_flat.min()), float(w_flat.max())
    if hi - lo < 1e-15:
        return np.full_like(w_flat, lo), 0.0

    for start_idx in range(n_starts):
        if start_idx == 0:
            # Uniform initialization
            levels = np.linspace(lo, hi, nl)
        elif start_idx == 1:
            # Quantile initialization
            levels = np.quantile(w_flat, np.linspace(0, 1, nl))
        else:
            # k-means++ initialization (seeded)
            levels = np.zeros(nl)
            levels[0] = w_flat[rng.integers(len(w_flat))]
            for c in range(1, nl):
                dists = np.min(np.abs(w_flat[:, None] - levels[:c][None, :]), axis=1)
                probs = dists ** 2 / np.sum(dists ** 2)
                levels[c] = w_flat[rng.choice(len(w_flat), p=probs)]
            levels.sort()

        for _ in range(n_iters):
            dists = np.abs(w_flat[:, None] - levels[None, :])
            assign = np.argmin(dists, axis=1)
            new_levels = levels.copy()
            for i in range(nl):
                mask = assign == i
                if np.any(mask):
                    new_levels[i] = np.mean(w_flat[mask])
                else:
                    # Empty cell: reinitialize at nearest unassigned data point (seeded)
                    unassigned = np.ones(len(w_flat), dtype=bool)
                    unassigned[assign] = False
                    if np.any(unassigned):
                        new_levels[i] = w_flat[rng.choice(np.where(unassigned)[0])]
                    else:
                        new_levels[i] = w_flat[rng.integers(len(w_flat))]
            if np.max(np.abs(new_levels - levels)) < tol:
                levels = new_levels
                break
            levels = new_levels

        dists = np.abs(w_flat[:, None] - levels[None, :])
        assign = np.argmin(dists, axis=1)
        wq = levels[assign]
        mse = np.mean((w_flat - wq) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_wq = wq

    return best_wq, best_mse


def lloyd_max_matrix(W, K, tile=TILE):
    m, n = W.shape
    Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            r1, c1 = min(i + tile, m), min(j + tile, n)
            w_tile = W[i:r1, j:c1].flatten()
            wq_flat, _ = lloyd_max_quantize(w_tile, K, n_starts=3)
            Wq[i:r1, j:c1] = wq_flat.reshape(r1 - i, c1 - j)
    return Wq


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 3: Entropy-Constrained Quantization (corrected: entropy-code indices)
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_entropy(indices):
    """Shannon entropy of quantization indices in bits."""
    if len(indices) == 0:
        return 0.0
    counts = np.bincount(indices)
    probs = counts[counts > 0] / len(indices)
    return -np.sum(probs * np.log2(probs))


def tile_coded_rate(W, K, tile=TILE):
    """For each tile, compute the empirical entropy of the K-level indices.
    The coded rate per tile is H(indices). Total coded rate = mean over tiles.
    This is what entropy coding (Huffman/arithmetic) would achieve."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    rates = np.zeros((n_tm, n_tn))
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            w_tile = W[r0:r1, c0:c1].flatten()
            indices = quantize_tile_indices(w_tile, K)
            rates[ti, tj] = empirical_entropy(indices)
    return rates


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 4: Distribution Modeling (corrected: MLE via scipy.stats.gennorm)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_distribution_mle(w_flat):
    """Fit Gaussian, Laplacian, and Generalized Gaussian via MLE.
    Uses scipy.stats.gennorm.fit for GGD (proper joint MLE).
    Returns fits dict + AIC dict."""
    results = {}
    n = len(w_flat)

    # Gaussian MLE
    mu_g, sigma_g = np.mean(w_flat), np.std(w_flat, ddof=0)
    ll_gaussian = np.sum(sp_stats.norm.logpdf(w_flat, mu_g, max(sigma_g, 1e-15)))
    results['gaussian'] = {'ll': float(ll_gaussian), 'params': {'mu': float(mu_g), 'sigma': float(sigma_g)}}

    # Laplacian MLE
    mu_l = np.median(w_flat)
    b_l = np.mean(np.abs(w_flat - mu_l))
    ll_laplace = np.sum(sp_stats.laplace.logpdf(w_flat, mu_l, max(b_l, 1e-15)))
    results['laplacian'] = {'ll': float(ll_laplace), 'params': {'mu': float(mu_l), 'b': float(b_l)}}

    # Generalized Gaussian MLE via scipy.stats.gennorm
    try:
        beta_mle, loc_mle, scale_mle = sp_stats.gennorm.fit(w_flat)
        ll_ggd = np.sum(sp_stats.gennorm.logpdf(w_flat, beta_mle, loc=loc_mle, scale=scale_mle))
        results['generalized_gaussian'] = {
            'll': float(ll_ggd),
            'params': {'beta': float(beta_mle), 'loc': float(loc_mle), 'scale': float(scale_mle)}
        }
    except Exception:
        results['generalized_gaussian'] = {'ll': -np.inf, 'params': {}}

    # AIC = 2k - 2*ll (lower is better)
    aic = {}
    aic['gaussian'] = 2 * 2 - 2 * results['gaussian']['ll']
    aic['laplacian'] = 2 * 2 - 2 * results['laplacian']['ll']
    if np.isfinite(results['generalized_gaussian']['ll']):
        aic['generalized_gaussian'] = 2 * 3 - 2 * results['generalized_gaussian']['ll']

    return results, aic


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 5: Transform Coding Bound (corrected: separate BiIP vs Hadamard)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_hadamard_only(W, rng):
    """Apply only signed Hadamard rotation (orthogonal, preserves variance)."""
    U, _ = signed_random_hadamard(W.shape[0], rng)
    V, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U @ W @ V
    return W_t, (U, V)


def apply_biip_hadamard(W, H_X, H_G, rng):
    """Apply BiIP scaling + signed Hadamard rotation."""
    S_G, S_X, W_s = biip_scaling(W, H_X, H_G)
    U, _ = signed_random_hadamard(W.shape[0], rng)
    V, _ = signed_random_hadamard(W.shape[1], rng)
    W_t = U @ W_s @ V
    return W_t, (S_G, S_X, U, V)


def inverse_biip_hadamard(Wq_t, S_G, S_X, U, V):
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ Wq_t @ V.T @ S_X_inv


def inverse_hadamard_only(Wq_t, U, V):
    return U.T @ Wq_t @ V.T


def gaussianity_metrics(w_flat):
    n = len(w_flat)
    if n < 4:
        return {'kurtosis': 0.0, 'skewness': 0.0, 'jarque_bera': 0.0, 'anderson_darling': 0.0}
    kurt = float(sp_stats.kurtosis(w_flat, fisher=True))
    skew = float(sp_stats.skew(w_flat))
    jb = float(n / 6 * (skew ** 2 + kurt ** 2 / 4))
    try:
        ad_stat = float(sp_stats.anderson(w_flat, dist='norm').statistic)
    except Exception:
        ad_stat = np.inf
    return {'kurtosis': kurt, 'skewness': skew, 'jarque_bera': jb, 'anderson_darling': ad_stat}


def negentropy(w_flat):
    """Negentropy = h(Gaussian) - h(empirical), both in bits.
    h(Gaussian) = 0.5 * log2(2*pi*e*sigma^2).
    Theoretically nonneg (Gaussian has max entropy at fixed variance).
    NOTE: The KDE estimator has finite-sample bias. Negative estimated values
    indicate estimator bias, NOT that the distribution is more Gaussian than
    Gaussian. Use kurtosis as the primary Gaussianization metric."""
    var = np.var(w_flat)
    if var < 1e-15:
        return 0.0
    h_gauss = 0.5 * np.log2(2 * np.pi * np.e * var)
    h_emp = differential_entropy_bits(w_flat)
    return h_gauss - h_emp


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 6: Gap Analysis (corrected: proper references)
# ═══════════════════════════════════════════════════════════════════════════════

def rd_curve_uniform(W, H_G, H_X, tile=TILE):
    points = []
    for k in K_VALUES:
        Wq = quantize_tiles(W, k, tile)
        E = W - Wq
        points.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
    return points


def rd_curve_lloyd_max(W, H_G, H_X, tile=TILE):
    points = []
    for k in K_VALUES:
        Wq = lloyd_max_matrix(W, k, tile)
        E = W - Wq
        points.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
    return points


def rd_curve_gptq(W, H_G, H_X, tile=TILE, alpha=1.0):
    """GPTQ with error feedback ENABLED (alpha=1.0)."""
    points = []
    for k in K_VALUES:
        Wq = gptq_quantize(W, H_X, k, tile, alpha=alpha)
        E = W - Wq
        points.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
    return points


def rd_curve_hadamard(W, H_G, H_X, rng, tile=TILE):
    """Hadamard-only rotation (orthogonal, preserves variance)."""
    W_t, (U, V) = apply_hadamard_only(W, rng)
    points = []
    for k in K_VALUES:
        Wq_t = quantize_tiles(W_t, k, tile)
        W_hat = inverse_hadamard_only(Wq_t, U, V)
        E = W - W_hat
        points.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
    return points


def rd_curve_biip_hadamard(W, H_G, H_X, rng, tile=TILE):
    """BiIP + Hadamard rotation."""
    W_t, transforms = apply_biip_hadamard(W, H_X, H_G, rng)
    S_G, S_X, U, V = transforms
    points = []
    for k in K_VALUES:
        Wq_t = quantize_tiles(W_t, k, tile)
        W_hat = inverse_biip_hadamard(Wq_t, S_G, S_X, U, V)
        E = W - W_hat
        points.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
    return points


def rd_curve_full_stack(W, H_G, H_X, rng, tile=TILE, alpha=1.0):
    """Full stack: BiIP+Hadamard + GPTQ (alpha=1.0) with correct rotated Hessians.
    For transform W' = U @ S_G @ W @ S_X @ V, the inverse is
    W_hat = S_G^{-1} @ U^T @ Wq' @ V^T @ S_X^{-1}.
    Error E = W - W_hat. The HWE tr(H_G E H_X E^T) is computed in original space.
    For GPTQ in rotated space, H_X' = V^T S_X^{-1} H_X S_X^{-1} V."""
    W_t, transforms = apply_biip_hadamard(W, H_X, H_G, rng)
    S_G, S_X, U, V = transforms
    # Correct rotated Hessian for GPTQ: H_X' = V^T S_X^{-1} H_X S_X^{-1} V
    S_X_inv = np.linalg.inv(S_X)
    H_X_rot = V.T @ S_X_inv @ H_X @ S_X_inv @ V
    # Normalize to prevent overflow
    H_X_rot = H_X_rot / max(np.trace(H_X_rot), 1e-15) * H_X.shape[0]

    points = []
    for k in K_VALUES:
        Wq_t = gptq_quantize(W_t, H_X_rot, k, tile, alpha=alpha)
        W_hat = inverse_biip_hadamard(Wq_t, S_G, S_X, U, V)
        E = W - W_hat
        points.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
    return points


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 7: High-Resolution Approximation (per-tile, not whole slice)
# ═══════════════════════════════════════════════════════════════════════════════

def high_res_approx_per_tile(W, k, tile=TILE):
    """Compute actual vs predicted MSE per tile. Returns list of ratios."""
    m, n = W.shape
    ratios = []
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            r1, c1 = min(i + tile, m), min(j + tile, n)
            w_tile = W[i:r1, j:c1].flatten()
            nl = 2 ** k
            lo, hi = float(w_tile.min()), float(w_tile.max())
            step = (hi - lo) / (nl - 1) if hi - lo > 1e-15 else 0.0
            predicted = step ** 2 / 12.0
            wq = quantize_tile(W[i:r1, j:c1], k).flatten()
            actual = np.mean((w_tile - wq) ** 2)
            ratio = actual / predicted if predicted > 1e-15 else 1.0
            ratios.append(ratio)
    return np.array(ratios)


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS 8: Empirical Mutual Information (replaces AWGN capacity)
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_mutual_info_tile(W, K, tile=TILE):
    """For deterministic quantization, I(W; Q(W)) = H(Q(W)) since Q is a
    deterministic function of W. Compute per-tile H(Q) and aggregate."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    mis = np.zeros((n_tm, n_tn))
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            w_tile = W[r0:r1, c0:c1].flatten()
            indices = quantize_tile_indices(w_tile, K)
            mis[ti, tj] = empirical_entropy(indices)
    return mis


# ═══════════════════════════════════════════════════════════════════════════════
# Per-tile analysis (ALL tiles, not just 4)
# ═══════════════════════════════════════════════════════════════════════════════

def per_tile_analysis(W, H_G, H_X, tile=TILE):
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    tile_stats = []
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, c0 = ti * tile, tj * tile
            r1, c1 = min(r0 + tile, m), min(c0 + tile, n)
            w_tile = W[r0:r1, c0:c1]
            w_flat = w_tile.flatten()
            stat = {
                'ti': ti, 'tj': tj,
                'std': float(np.std(w_flat)),
                'kurtosis': float(sp_stats.kurtosis(w_flat, fisher=True)),
            }
            # Distribution fit (MLE)
            _, tile_aic = fit_distribution_mle(w_flat)
            if tile_aic:
                stat['best_dist'] = min(tile_aic, key=tile_aic.get)
            # GGD beta
            try:
                beta_mle, _, _ = sp_stats.gennorm.fit(w_flat)
                stat['ggd_beta'] = float(beta_mle)
            except Exception:
                stat['ggd_beta'] = 2.0
            # Uniform vs Lloyd-Max at K=3 and K=5
            for k in [3, 5]:
                wq_unif = quantize_tile(w_tile, k)
                mse_unif = np.mean((w_tile - wq_unif) ** 2)
                _, mse_lm = lloyd_max_quantize(w_flat, k, n_starts=3)
                stat[f'uniform_mse_k{k}'] = float(mse_unif)
                stat[f'lloyd_max_mse_k{k}'] = float(mse_lm)
                stat[f'lm_improvement_k{k}'] = float((mse_unif - mse_lm) / mse_unif * 100) if mse_unif > 1e-15 else 0.0
            tile_stats.append(stat)
    return tile_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    t_start = time.time()
    print("=" * 80)
    print("R20 — Information-Theoretic Bounds (v2, reviewer-corrected)")
    print("=" * 80)

    tensors = load_real_weights()
    rng = np.random.default_rng(RNG_SEED)

    all_results = {
        'config': {'tile': TILE, 'slice': SLICE, 'K_values': K_VALUES, 'tensor_names': TENSOR_NAMES},
        'tensors': {}
    }

    for tname in TENSOR_NAMES:
        if tname not in tensors:
            continue
        print(f"\n{'─' * 60}\n  Tensor: {tname} (shape {tensors[tname].shape})\n{'─' * 60}")
        tensor_results = {'slices': {}}
        slices = extract_slices(tensors[tname], SLICE, SLICE, RNG_SEED)
        X = gen_calibration(SLICE, N_CALIB, RNG_SEED)

        for sname, W in slices:
            print(f"\n  Slice: {sname} ({W.shape})")
            H_G, H_X = compute_hessians(W, X)
            w_flat = W.flatten()
            var_w = np.var(w_flat)
            slice_result = {}

            # ─── Axis 1: Shannon Lower Bound (corrected) ───
            print("  [1] Shannon lower bound (bits, LOO)...")
            h_bits = differential_entropy_bits(w_flat)
            # Entropy power: N = 2^(2h) / (2*pi*e) — should be <= variance
            entropy_power = 2 ** (2 * h_bits) / (2 * np.pi * np.e)
            # Shannon lower bound on distortion at rate K
            shannon_rd = []
            for k in K_VALUES:
                D_slb = max(2 ** (2 * (h_bits - k)) / (2 * np.pi * np.e), 1e-300)
                D_gauss_ref = gaussian_rd_D(var_w, k)
                D_scalar_bound = high_rate_scalar_bound(var_w, k)
                shannon_rd.append({
                    'K': k, 'D_shannon_lb': float(D_slb),
                    'D_gaussian_ref': float(D_gauss_ref),
                    'D_scalar_hr_bound': float(D_scalar_bound),
                })
            slice_result['shannon_lower_bound'] = {
                'h_bits': float(h_bits),
                'entropy_power': float(entropy_power),
                'variance': float(var_w),
                'entropy_power_le_variance': bool(entropy_power <= var_w),
                'entropy_power_ratio': float(entropy_power / var_w) if var_w > 1e-15 else 0.0,
                'estimator_note': 'KDE-based h is a biased point estimate. N(X) > Var(X) '
                    'indicates positive bias; SLB values are estimates, not rigorous bounds.',
                'rd_curve': shannon_rd,
            }
            ep_pass = entropy_power <= var_w
            print(f"      h(W) = {h_bits:.4f} bits, N(X) = {entropy_power:.6e}, var = {var_w:.6e}, "
                  f"{'PASS' if ep_pass else 'BIASED (N>Var)'}")

            # ─── Axis 2: Lloyd-Max vs Uniform (all tiles, multistart) ───
            print("  [2] Lloyd-Max vs uniform (all tiles, multistart)...")
            lm_rd = rd_curve_lloyd_max(W, H_G, H_X, TILE)
            unif_rd = rd_curve_uniform(W, H_G, H_X, TILE)
            slice_result['lloyd_max_vs_uniform'] = {'uniform_rd': unif_rd, 'lloyd_max_rd': lm_rd}
            for k_idx, k in enumerate(K_VALUES):
                u_mse = unif_rd[k_idx]['mse']
                lm_mse = lm_rd[k_idx]['mse']
                gap = (u_mse - lm_mse) / u_mse * 100 if u_mse > 1e-15 else 0.0
                print(f"      K={k}: uniform={u_mse:.4e}, LM={lm_mse:.4e}, gap={gap:.1f}%")

            # ─── Axis 3: Entropy-Coded Rate (corrected: code indices, don't requantize) ───
            print("  [3] Entropy-coded rate (Huffman/arithmetic on indices)...")
            ec_results = {}
            for k in [3, 4, 5, 6]:
                coded_rates = tile_coded_rate(W, k, TILE)
                mean_coded_rate = float(np.mean(coded_rates))
                ec_results[f'K{k}'] = {
                    'fixed_rate': k,
                    'mean_coded_rate': mean_coded_rate,
                    'rate_savings_pct': float((k - mean_coded_rate) / k * 100),
                    'coded_rates_per_tile': coded_rates.tolist(),
                }
                print(f"      K={k}: fixed={k} bits, coded={mean_coded_rate:.3f} bits, savings={((k-mean_coded_rate)/k*100):.1f}%")
            slice_result['entropy_coded_rate'] = ec_results

            # ─── Axis 4: Distribution Modeling (MLE) ───
            print("  [4] Distribution modeling (MLE via gennorm.fit)...")
            dist_results, aic = fit_distribution_mle(w_flat)
            best_dist = min(aic, key=aic.get)
            print(f"      Best fit: {best_dist} (AIC={aic[best_dist]:.1f})")
            for dname, dres in dist_results.items():
                print(f"      {dname}: ll={dres['ll']:.1f}, params={dres['params']}")

            # Per-tile distribution fits (ALL tiles)
            tile_dist_summary = {'gaussian': 0, 'laplacian': 0, 'generalized_gaussian': 0}
            tile_betas = []
            n_tm = (SLICE + TILE - 1) // TILE
            n_tn = (SLICE + TILE - 1) // TILE
            for ti in range(n_tm):
                for tj in range(n_tn):
                    r0, c0 = ti * TILE, tj * TILE
                    r1, c1 = min(r0 + TILE, SLICE), min(c0 + TILE, SLICE)
                    w_tf = W[r0:r1, c0:c1].flatten()
                    _, tile_aic = fit_distribution_mle(w_tf)
                    if tile_aic:
                        bd = min(tile_aic, key=tile_aic.get)
                        tile_dist_summary[bd] = tile_dist_summary.get(bd, 0) + 1
                    try:
                        beta_mle, _, _ = sp_stats.gennorm.fit(w_tf)
                        tile_betas.append(float(beta_mle))
                    except Exception:
                        pass
            slice_result['distribution_modeling'] = {
                'full_slice': {
                    'fits': {k: {'ll': float(v['ll']), 'params': v.get('params', {})} for k, v in dist_results.items()},
                    'aic': {k: float(v) for k, v in aic.items()},
                    'best_fit': best_dist,
                },
                'per_tile_best_fit_counts': tile_dist_summary,
                'per_tile_ggd_beta': {
                    'mean': float(np.mean(tile_betas)) if tile_betas else 2.0,
                    'std': float(np.std(tile_betas)) if tile_betas else 0.0,
                    'median': float(np.median(tile_betas)) if tile_betas else 2.0,
                    'min': float(np.min(tile_betas)) if tile_betas else 2.0,
                    'max': float(np.max(tile_betas)) if tile_betas else 2.0,
                },
            }
            print(f"      Per-tile: {tile_dist_summary}, GGD beta mean={np.mean(tile_betas):.3f}" if tile_betas else "")

            # ─── Axis 5: Transform Coding (separate BiIP vs Hadamard, negentropy) ───
            print("  [5] Transform coding (separate BiIP vs Hadamard)...")
            gauss_before = gaussianity_metrics(w_flat)
            negent_before = negentropy(w_flat)

            # Generate ONE U,V pair and reuse for both Hadamard-only and BiIP+Hadamard
            # to isolate the BiIP scaling effect on the SAME rotation
            U_fixed, _ = signed_random_hadamard(W.shape[0], rng)
            V_fixed, _ = signed_random_hadamard(W.shape[1], rng)

            # Hadamard only (orthogonal, preserves variance): W' = U @ W @ V
            W_had = U_fixed @ W @ V_fixed
            w_had_flat = W_had.flatten()
            gauss_had = gaussianity_metrics(w_had_flat)
            negent_had = negentropy(w_had_flat)

            # BiIP + Hadamard: W' = U @ S_G @ W @ S_X @ V (same U,V)
            S_G, S_X, W_s = biip_scaling(W, H_X, H_G)
            W_biip = U_fixed @ W_s @ V_fixed
            w_biip_flat = W_biip.flatten()
            gauss_biip = gaussianity_metrics(w_biip_flat)
            negent_biip = negentropy(w_biip_flat)

            slice_result['transform_coding'] = {
                'gaussianity_before': gauss_before,
                'gaussianity_hadamard': gauss_had,
                'gaussianity_biip_hadamard': gauss_biip,
                'negentropy_before': float(negent_before),
                'negentropy_hadamard': float(negent_had),
                'negentropy_biip_hadamard': float(negent_biip),
                'variance_before': float(var_w),
                'variance_hadamard': float(np.var(w_had_flat)),
                'variance_biip_hadamard': float(np.var(w_biip_flat)),
            }
            print(f"      Kurtosis: {gauss_before['kurtosis']:.3f} → Had:{gauss_had['kurtosis']:.3f} → BiIP+Had:{gauss_biip['kurtosis']:.3f}")
            print(f"      Negentropy: {negent_before:.4f} → Had:{negent_had:.4f} → BiIP+Had:{negent_biip:.4f}")
            print(f"      Variance: {var_w:.4e} → Had:{np.var(w_had_flat):.4e} → BiIP+Had:{np.var(w_biip_flat):.4e}")

            # ─── Axis 6: Gap Analysis (corrected references) ───
            print("  [6] Gap analysis (Gaussian ref + scalar asymptotic ref)...")
            # Generate ONE rotation pair and reuse for all rotation-based arms
            U_gap, _ = signed_random_hadamard(W.shape[0], rng)
            V_gap, _ = signed_random_hadamard(W.shape[1], rng)
            rd_uniform = rd_curve_uniform(W, H_G, H_X, TILE)
            rd_gptq = rd_curve_gptq(W, H_G, H_X, TILE, alpha=1.0)
            # Hadamard-only: reuse U_gap, V_gap
            W_had_gap = U_gap @ W @ V_gap
            rd_had = []
            for k in K_VALUES:
                Wq_t = quantize_tiles(W_had_gap, k, TILE)
                W_hat = U_gap.T @ Wq_t @ V_gap.T
                E = W - W_hat
                rd_had.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
            # BiIP+Hadamard: same U_gap, V_gap + BiIP scaling
            S_G_g, S_X_g, W_s_g = biip_scaling(W, H_X, H_G)
            W_biip_gap = U_gap @ W_s_g @ V_gap
            rd_biip = []
            for k in K_VALUES:
                Wq_t = quantize_tiles(W_biip_gap, k, TILE)
                S_G_inv = np.linalg.inv(S_G_g)
                S_X_inv = np.linalg.inv(S_X_g)
                W_hat = S_G_inv @ U_gap.T @ Wq_t @ V_gap.T @ S_X_inv
                E = W - W_hat
                rd_biip.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})
            # Full stack: BiIP+Hadamard+GPTQ with correct rotated Hessian
            S_X_inv_g = np.linalg.inv(S_X_g)
            H_X_rot_g = V_gap.T @ S_X_inv_g @ H_X @ S_X_inv_g @ V_gap
            H_X_rot_g = H_X_rot_g / max(np.trace(H_X_rot_g), 1e-15) * H_X.shape[0]
            rd_stack = []
            for k in K_VALUES:
                Wq_t = gptq_quantize(W_biip_gap, H_X_rot_g, k, TILE, alpha=1.0)
                S_G_inv = np.linalg.inv(S_G_g)
                S_X_inv = np.linalg.inv(S_X_g)
                W_hat = S_G_inv @ U_gap.T @ Wq_t @ V_gap.T @ S_X_inv
                E = W - W_hat
                rd_stack.append({'K': k, 'rate': k, 'mse': weight_mse(E), 'hwe': hessian_weighted_error(E, H_G, H_X)})

            gap_analysis = []
            for k_idx, k in enumerate(K_VALUES):
                D_gauss_ref = gaussian_rd_D(var_w, k)
                D_scalar_bound = high_rate_scalar_bound(var_w, k)
                D_slb = max(2 ** (2 * (h_bits - k)) / (2 * np.pi * np.e), 1e-300)

                def gap_pct(achieved, bound):
                    return float((achieved - bound) / bound * 100) if bound > 1e-300 else 0.0

                gap_analysis.append({
                    'K': k,
                    'D_uniform': float(rd_uniform[k_idx]['mse']),
                    'D_gptq': float(rd_gptq[k_idx]['mse']),
                    'D_hadamard': float(rd_had[k_idx]['mse']),
                    'D_biip_hadamard': float(rd_biip[k_idx]['mse']),
                    'D_full_stack': float(rd_stack[k_idx]['mse']),
                    'D_lloyd_max': float(lm_rd[k_idx]['mse']),
                    'D_shannon_lb': float(D_slb),
                    'D_gaussian_ref': float(D_gauss_ref),
                    'D_scalar_hr_bound': float(D_scalar_bound),
                    'gap_uniform_vs_gauss_ref_pct': gap_pct(rd_uniform[k_idx]['mse'], D_gauss_ref),
                    'gap_uniform_vs_scalar_bound_pct': gap_pct(rd_uniform[k_idx]['mse'], D_scalar_bound),
                    'gap_gptq_vs_gauss_ref_pct': gap_pct(rd_gptq[k_idx]['mse'], D_gauss_ref),
                    'gap_gptq_vs_scalar_bound_pct': gap_pct(rd_gptq[k_idx]['mse'], D_scalar_bound),
                    'gap_lm_vs_gauss_ref_pct': gap_pct(lm_rd[k_idx]['mse'], D_gauss_ref),
                    'gap_lm_vs_scalar_bound_pct': gap_pct(lm_rd[k_idx]['mse'], D_scalar_bound),
                    'gap_stack_vs_gauss_ref_pct': gap_pct(rd_stack[k_idx]['mse'], D_gauss_ref),
                    'gap_stack_vs_scalar_bound_pct': gap_pct(rd_stack[k_idx]['mse'], D_scalar_bound),
                    'HWE_uniform': float(rd_uniform[k_idx]['hwe']),
                    'HWE_gptq': float(rd_gptq[k_idx]['hwe']),
                    'HWE_hadamard': float(rd_had[k_idx]['hwe']),
                    'HWE_biip_hadamard': float(rd_biip[k_idx]['hwe']),
                    'HWE_full_stack': float(rd_stack[k_idx]['hwe']),
                })
            slice_result['gap_analysis'] = gap_analysis
            for ga in gap_analysis:
                if ga['K'] in [3, 5]:
                    print(f"      K={ga['K']}: D_unif={ga['D_uniform']:.4e}, "
                          f"D_gauss_ref={ga['D_gaussian_ref']:.4e}, "
                          f"D_scalar={ga['D_scalar_hr_bound']:.4e}, "
                          f"gap_unif/scalar={ga['gap_uniform_vs_scalar_bound_pct']:.1f}%")

            # ─── Axis 7: High-Resolution Approximation (per-tile) ───
            print("  [7] High-res approximation (per-tile)...")
            hr_results = {}
            for k in K_VALUES:
                ratios = high_res_approx_per_tile(W, k, TILE)
                hr_results[f'K{k}'] = {
                    'ratio_mean': float(np.mean(ratios)),
                    'ratio_std': float(np.std(ratios)),
                    'ratio_p5': float(np.percentile(ratios, 5)),
                    'ratio_p95': float(np.percentile(ratios, 95)),
                    'ratio_min': float(np.min(ratios)),
                    'ratio_max': float(np.max(ratios)),
                    'n_tiles': len(ratios),
                    'all_ratios': ratios.tolist(),
                }
                print(f"      K={k}: mean={np.mean(ratios):.3f}, p5={np.percentile(ratios,5):.3f}, p95={np.percentile(ratios,95):.3f}")
            slice_result['high_resolution_approx'] = hr_results

            # ─── Axis 8: Empirical Mutual Information ───
            print("  [8] Empirical mutual information I(W;Q) = H(Q)...")
            mi_results = {}
            for k in K_VALUES:
                mis = empirical_mutual_info_tile(W, k, TILE)
                mean_mi = float(np.mean(mis))
                mi_results[f'K{k}'] = {
                    'mean_mi_bits': mean_mi,
                    'fixed_rate': k,
                    'rate_utilization_pct': float(mean_mi / k * 100) if k > 0 else 0.0,
                }
                print(f"      K={k}: H(Q)={mean_mi:.3f} bits, utilization={mean_mi/k*100:.1f}%")
            slice_result['empirical_mutual_info'] = mi_results

            # ─── Per-tile analysis (ALL tiles) ───
            print("  [9] Per-tile analysis (all tiles)...")
            tile_stats = per_tile_analysis(W, H_G, H_X, TILE)
            slice_result['per_tile_analysis'] = tile_stats

            tensor_results['slices'][sname] = slice_result

        # Aggregate
        print(f"\n  Aggregating {tname}...")
        tensor_results['aggregate'] = aggregate_slices(tensor_results['slices'])
        all_results['tensors'][tname] = tensor_results

    # Cross-tensor summary
    print(f"\n{'=' * 80}\n  Cross-tensor summary\n{'=' * 80}")
    summary = cross_tensor_summary(all_results)
    all_results['summary'] = summary
    for line in summary:
        print(line)

    elapsed = time.time() - t_start
    all_results['elapsed_seconds'] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS_PATH}")
    return all_results


def aggregate_slices(slices):
    agg = {}
    snames = list(slices.keys())
    if not snames:
        return agg

    # Gap analysis
    agg['gap_analysis'] = []
    for k_idx, k in enumerate(K_VALUES):
        row = {'K': k}
        for field in ['D_uniform', 'D_gptq', 'D_hadamard', 'D_biip_hadamard', 'D_full_stack',
                       'D_lloyd_max', 'D_shannon_lb', 'D_gaussian_ref', 'D_scalar_hr_bound',
                       'gap_uniform_vs_gauss_ref_pct', 'gap_uniform_vs_scalar_bound_pct',
                       'gap_gptq_vs_gauss_ref_pct', 'gap_gptq_vs_scalar_bound_pct',
                       'gap_lm_vs_gauss_ref_pct', 'gap_lm_vs_scalar_bound_pct',
                       'gap_stack_vs_gauss_ref_pct', 'gap_stack_vs_scalar_bound_pct',
                       'HWE_uniform', 'HWE_gptq', 'HWE_hadamard', 'HWE_biip_hadamard', 'HWE_full_stack']:
            vals = []
            for sn in snames:
                ga = slices[sn].get('gap_analysis', [])
                if k_idx < len(ga):
                    vals.append(ga[k_idx].get(field, 0.0))
            row[f'{field}_mean'] = float(np.mean(vals)) if vals else 0.0
            row[f'{field}_std'] = float(np.std(vals)) if vals else 0.0
        agg['gap_analysis'].append(row)
    # High-res per-tile (pool ALL 192 ratios across slices)
    agg['high_resolution_approx'] = {}
    for k in K_VALUES:
        key = f'K{k}'
        all_ratios = []
        for sn in snames:
            hr = slices[sn].get('high_resolution_approx', {})
            if key in hr and 'all_ratios' in hr[key]:
                all_ratios.extend(hr[key]['all_ratios'])
        if all_ratios:
            all_ratios = np.array(all_ratios)
            agg['high_resolution_approx'][key] = {
                'ratio_mean': float(np.mean(all_ratios)),
                'ratio_p5': float(np.percentile(all_ratios, 5)),
                'ratio_p95': float(np.percentile(all_ratios, 95)),
                'n_tiles_pooled': len(all_ratios),
            }
        else:
            agg['high_resolution_approx'][key] = {'ratio_mean': 1.0, 'ratio_p5': 1.0, 'ratio_p95': 1.0, 'n_tiles_pooled': 0}

    # Lloyd-Max improvement (ALL tiles)
    agg['lloyd_max_improvement'] = {}
    for k in [3, 5]:
        improvements = []
        for sn in snames:
            for ts in slices[sn].get('per_tile_analysis', []):
                lm_key = f'lloyd_max_mse_k{k}'
                unif_key = f'uniform_mse_k{k}'
                if lm_key in ts and unif_key in ts and ts[unif_key] > 1e-15:
                    improvements.append((ts[unif_key] - ts[lm_key]) / ts[unif_key] * 100)
        agg['lloyd_max_improvement'][f'K{k}'] = {
            'mean_pct': float(np.mean(improvements)) if improvements else 0.0,
            'median_pct': float(np.median(improvements)) if improvements else 0.0,
            'max_pct': float(np.max(improvements)) if improvements else 0.0,
            'n_tiles': len(improvements),
        }

    # Distribution modeling (ALL tiles)
    dist_counts = {'gaussian': 0, 'laplacian': 0, 'generalized_gaussian': 0}
    all_betas = []
    for sn in snames:
        dm = slices[sn].get('distribution_modeling', {})
        for kk, v in dm.get('per_tile_best_fit_counts', {}).items():
            dist_counts[kk] = dist_counts.get(kk, 0) + v
        ggd = dm.get('per_tile_ggd_beta', {})
        if 'mean' in ggd:
            all_betas.append(ggd['mean'])
    agg['distribution_modeling'] = {
        'per_tile_best_fit_counts': dist_counts,
        'ggd_beta_mean': float(np.mean(all_betas)) if all_betas else 2.0,
    }

    # Empirical MI
    agg['empirical_mutual_info'] = {}
    for k in K_VALUES:
        key = f'K{k}'
        mis = []
        for sn in snames:
            mi = slices[sn].get('empirical_mutual_info', {})
            if key in mi:
                mis.append(mi[key]['mean_mi_bits'])
        agg['empirical_mutual_info'][key] = {
            'mean_mi_bits': float(np.mean(mis)) if mis else 0.0,
            'rate': k,
            'utilization_pct': float(np.mean(mis) / k * 100) if mis and k > 0 else 0.0,
        }

    # Transform coding
    tc_agg = {'kurt_before': [], 'kurt_had': [], 'kurt_biip': [],
              'negent_before': [], 'negent_had': [], 'negent_biip': [],
              'var_before': [], 'var_had': [], 'var_biip': []}
    for sn in snames:
        tc = slices[sn].get('transform_coding', {})
        if 'gaussianity_before' in tc:
            tc_agg['kurt_before'].append(tc['gaussianity_before']['kurtosis'])
            tc_agg['kurt_had'].append(tc['gaussianity_hadamard']['kurtosis'])
            tc_agg['kurt_biip'].append(tc['gaussianity_biip_hadamard']['kurtosis'])
        if 'negentropy_before' in tc:
            tc_agg['negent_before'].append(tc['negentropy_before'])
            tc_agg['negent_had'].append(tc['negentropy_hadamard'])
            tc_agg['negent_biip'].append(tc['negentropy_biip_hadamard'])
        if 'variance_before' in tc:
            tc_agg['var_before'].append(tc['variance_before'])
            tc_agg['var_had'].append(tc['variance_hadamard'])
            tc_agg['var_biip'].append(tc['variance_biip_hadamard'])
    agg['transform_coding'] = {
        'kurtosis_before': float(np.mean(tc_agg['kurt_before'])) if tc_agg['kurt_before'] else 0,
        'kurtosis_hadamard': float(np.mean(tc_agg['kurt_had'])) if tc_agg['kurt_had'] else 0,
        'kurtosis_biip_hadamard': float(np.mean(tc_agg['kurt_biip'])) if tc_agg['kurt_biip'] else 0,
        'negentropy_before': float(np.mean(tc_agg['negent_before'])) if tc_agg['negent_before'] else 0,
        'negentropy_hadamard': float(np.mean(tc_agg['negent_had'])) if tc_agg['negent_had'] else 0,
        'negentropy_biip_hadamard': float(np.mean(tc_agg['negent_biip'])) if tc_agg['negent_biip'] else 0,
        'variance_before': float(np.mean(tc_agg['var_before'])) if tc_agg['var_before'] else 0,
        'variance_hadamard': float(np.mean(tc_agg['var_had'])) if tc_agg['var_had'] else 0,
        'variance_biip_hadamard': float(np.mean(tc_agg['var_biip'])) if tc_agg['var_biip'] else 0,
    }

    # Entropy-coded rate
    agg['entropy_coded_rate'] = {}
    for k in [3, 4, 5, 6]:
        key = f'K{k}'
        rates, savings = [], []
        for sn in snames:
            ec = slices[sn].get('entropy_coded_rate', {})
            if key in ec:
                rates.append(ec[key]['mean_coded_rate'])
                savings.append(ec[key]['rate_savings_pct'])
        agg['entropy_coded_rate'][key] = {
            'mean_coded_rate': float(np.mean(rates)) if rates else 0,
            'mean_savings_pct': float(np.mean(savings)) if savings else 0,
        }

    return agg


def cross_tensor_summary(all_results):
    lines = []

    lines.append("\n── Gap to Scalar High-Rate Asymptotic Reference at K=5 (MSE) ──")
    lines.append("  (Panter-Dite asymptotic reference ≈ 2.72 × σ²/2^(2K), NOT a finite-K bound)")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        ga = agg.get('gap_analysis', [])
        if len(ga) >= 4:
            row = ga[3]
            lines.append(f"  {tname}: uniform={row['gap_uniform_vs_scalar_bound_pct_mean']:.1f}%, "
                         f"LM={row['gap_lm_vs_scalar_bound_pct_mean']:.1f}%, "
                         f"GPTQ={row['gap_gptq_vs_scalar_bound_pct_mean']:.1f}%, "
                         f"stack={row['gap_stack_vs_scalar_bound_pct_mean']:.1f}%")

    lines.append("\n── Gap to Gaussian R-D Reference at K=5 (MSE) ──")
    lines.append("  (D_gauss = σ²/2^(2K), achievable only with vector quantization + entropy coding)")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        ga = agg.get('gap_analysis', [])
        if len(ga) >= 4:
            row = ga[3]
            lines.append(f"  {tname}: uniform={row['gap_uniform_vs_gauss_ref_pct_mean']:.1f}%, "
                         f"LM={row.get('gap_lm_vs_gauss_ref_pct_mean', 0):.1f}%, "
                         f"stack={row['gap_stack_vs_gauss_ref_pct_mean']:.1f}%")

    lines.append("\n── Lloyd-Max vs Uniform (all tiles, multistart) ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        for k in [3, 5]:
            lm = agg.get('lloyd_max_improvement', {}).get(f'K{k}', {})
            lines.append(f"  {tname} K{k}: mean={lm.get('mean_pct', 0):.1f}%, "
                         f"median={lm.get('median_pct', 0):.1f}%, "
                         f"max={lm.get('max_pct', 0):.1f}% (n={lm.get('n_tiles', 0)})")

    lines.append("\n── High-Resolution Approx (per-tile ratio, p5–p95) ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        hr = agg.get('high_resolution_approx', {})
        for k in [3, 5, 7]:
            key = f'K{k}'
            if key in hr:
                lines.append(f"  {tname} K{k}: mean={hr[key]['ratio_mean']:.3f}, "
                             f"p5={hr[key]['ratio_p5']:.3f}, p95={hr[key]['ratio_p95']:.3f} (n={hr[key].get('n_tiles_pooled', 0)})")

    lines.append("\n── Distribution Modeling (MLE, per-tile) ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        dm = agg.get('distribution_modeling', {})
        lines.append(f"  {tname}: {dm.get('per_tile_best_fit_counts', {})}, "
                     f"GGD beta={dm.get('ggd_beta_mean', 2):.3f}")

    lines.append("\n── Transform Coding (kurtosis, negentropy) ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        tc = agg.get('transform_coding', {})
        lines.append(f"  {tname}: kurt {tc.get('kurtosis_before', 0):.3f} → "
                     f"Had:{tc.get('kurtosis_hadamard', 0):.3f} → "
                     f"BiIP+Had:{tc.get('kurtosis_biip_hadamard', 0):.3f}; "
                     f"negent {tc.get('negentropy_before', 0):.4f} → "
                     f"Had:{tc.get('negentropy_hadamard', 0):.4f} → "
                     f"BiIP+Had:{tc.get('negentropy_biip_hadamard', 0):.4f}")

    lines.append("\n── Empirical Mutual Information H(Q) vs rate ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        mi = agg.get('empirical_mutual_info', {})
        parts = []
        for k in [3, 5, 7]:
            key = f'K{k}'
            if key in mi:
                parts.append(f"K{k}={mi[key]['mean_mi_bits']:.2f} ({mi[key]['utilization_pct']:.0f}%)")
        lines.append(f"  {tname}: {', '.join(parts)}")

    lines.append("\n── Entropy-Coded Rate Savings ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        ec = agg.get('entropy_coded_rate', {})
        for k in [3, 5]:
            key = f'K{k}'
            if key in ec:
                lines.append(f"  {tname} K{k}: coded={ec[key]['mean_coded_rate']:.3f}, savings={ec[key]['mean_savings_pct']:.1f}%")

    lines.append("\n── HWE Improvement at K=5 (% vs RTN) ──")
    for tname in TENSOR_NAMES:
        if tname not in all_results['tensors']:
            continue
        agg = all_results['tensors'][tname].get('aggregate', {})
        ga = agg.get('gap_analysis', [])
        if len(ga) >= 4:
            row = ga[3]
            hwe_rtn = row['HWE_uniform_mean']
            for method, field in [('GPTQ', 'HWE_gptq'), ('Hadamard', 'HWE_hadamard'),
                                   ('BiIP+Had', 'HWE_biip_hadamard'), ('Full Stack', 'HWE_full_stack')]:
                hwe = row[f'{field}_mean']
                imp = (hwe_rtn - hwe) / hwe_rtn * 100 if hwe_rtn > 1e-15 else 0.0
                lines.append(f"  {tname} {method}: {imp:.1f}%")

    return lines


if __name__ == "__main__":
    run_experiment()
