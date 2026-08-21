#!/usr/bin/env python3
"""
R2-GPTAQ: GPTAQ/ResComp corrections + adaptive strength.

Clean-room implementation from paper equations (GPTQv2 arXiv:2504.02692,
ResComp arXiv:2604.07955). All arms use the SAME matched quantizer primitive:
per-column 16×1 segment uniform min-max quantization.

Key fixes from the v1/v2 harness:
  1. ResComp lazy-block propagation: cache every pre-quantization compensated
     column and use the saved CAE vectors in the outer-block update.
  2. P-matrix factor order: triu(D @ L^T, 1) @ L (not triu(D @ L, 1) @ L^T).
  3. GPTAQ outer-block term uses cached w_pre, not original W.
  4. All arms (including RTN baseline) use the same per-column quantizer.

Novel explorations:
  - Adaptive correction strength α (data-driven, grid search, per-layer)
  - Error-vector correction: (w_pre - Q) · P instead of w_pre · P
  - Blended coefficient: α·w_pre + (1-α)·Q
  - Iterative refinement: multi-round GPTAQ
  - Eigendecomposition P-matrix for numerical stability
"""

import numpy as np
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== Config ====================

@dataclass
class Config:
    m: int = 128            # output dim (rows of W)
    n: int = 128            # input dim (cols of W, also Hessian dim)
    k: int = 512            # calibration samples
    tile: int = 16          # quantizer segment size (rows per codebook)
    damping: float = 0.01   # Hessian regularization
    seeds: int = 3
    bits_list: tuple = (3, 4, 5, 6)
    block_sizes: tuple = (1, 16, 128)  # for invariance test
    alpha_grid: tuple = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0)


# ==================== Matched quantizer (ALL arms use this) ====================

def quantize_segment(w: np.ndarray, bits: int) -> np.ndarray:
    """Uniform min-max quantization of a 1D segment."""
    nl = 2 ** bits
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-12:
        return w.copy()
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_column(col: np.ndarray, bits: int, tile: int) -> np.ndarray:
    """Per-column quantization: split into segments of `tile` rows, quantize each."""
    m = len(col)
    out = np.zeros_like(col)
    for i in range(0, m, tile):
        out[i:i + tile] = quantize_segment(col[i:i + tile], bits)
    return out


def quantize_matrix(W: np.ndarray, bits: int, tile: int) -> np.ndarray:
    """Full matrix quantization using matched per-column-tile primitive."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    for j in range(n):
        Wq[:, j] = quantize_column(W[:, j], bits, tile)
    return Wq


# ==================== Hessian factorization ====================

def cholesky_factor(H: np.ndarray, damping: float) -> np.ndarray:
    """Upper-triangular factor U of inv(H + λI), so that U^T U = (H + λI)^{-1}.
    This is the standard GPTQ Cholesky reformulation.
    Returns U = R^T where R is the lower Cholesky of (H+λI)^{-1} (R R^T = (H+λI)^{-1})."""
    n = H.shape[0]
    Hi = np.linalg.inv(H + damping * np.eye(n))
    R = np.linalg.cholesky(Hi)   # lower-triangular R, R R^T = Hi
    return R.T                    # upper-triangular U = R^T, so U^T U = R R^T = Hi


def eigen_factor(H: np.ndarray, damping: float) -> np.ndarray:
    """Eigendecomposition-based factor of inv(H + λI), then Cholesky for triangularity.

    Instead of directly Cholesky-factoring (H+λI)^{-1}, we use eigendecomposition
    for better numerical stability on ill-conditioned Hessians:
      1. H + λI = Q Λ Q^T  (eigendecomposition, handles near-singular matrices)
      2. H^{-1} = Q Λ^{-1} Q^T  (inverse via eigenvalues, well-conditioned)
      3. R = chol(H^{-1})  (Cholesky of the well-conditioned inverse)
      4. U = R^T  (upper triangular, same as standard Cholesky path)

    The difference from direct Cholesky: step 2 uses eigenvalue clipping (min 1e-10)
    to handle near-singular directions, while direct inv() may amplify numerical
    errors from near-zero eigenvalues.
    """
    n = H.shape[0]
    evals, Q = np.linalg.eigh(H + damping * np.eye(n))
    evals = np.maximum(evals, 1e-10)
    Hinv = Q @ np.diag(1.0 / evals) @ Q.T
    # Ensure symmetric positive definite before Cholesky
    Hinv = (Hinv + Hinv.T) / 2
    R = np.linalg.cholesky(Hinv + 1e-12 * np.eye(n))
    return R.T


# ==================== P-matrix ====================

def compute_P(D: np.ndarray, U: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """P-matrix for GPTAQ/ResComp correction.

    From GPTQv2 Theorem 4.2 with Cholesky/neuron-decomposition reformulation.
    With H^{-1} = L L^T (lower Cholesky), the P-matrix is:
      P = α · triu(D · L, 1) · L^T

    The code returns U = L^T (upper triangular), so H^{-1} = U^T U.
    Substituting L = U^T:
      P = α · triu(D · U^T, 1) · U

    where D is the cross-covariance and U is upper-triangular with H^{-1} = U^T U.
    The strictly-upper mask (k=1) ensures corrections only propagate to future
    columns, matching the sequential column-processing order.
    """
    M = np.triu(D @ U.T, 1)   # strictly upper triangular
    return alpha * (M @ U)

def weight_mse(W: np.ndarray, Wq: np.ndarray) -> float:
    """||W - Wq||_F^2 / (m·n)"""
    return float(np.mean((W - Wq) ** 2))

def hessian_weighted_error(W: np.ndarray, Wq: np.ndarray, H: np.ndarray) -> float:
    """tr(E H E^T) / (m·n) where E = W - Wq, H = X X^T (unscaled).
    This equals ||(W - Wq) X||_F^2 / (m·n) when H = X X^T.
    Minimizing this is the GPTQ symmetric objective."""
    E = W - Wq
    return float(np.sum(E * (E @ H)) / (E.shape[0] * E.shape[1]))


def asymmetric_error(W: np.ndarray, Wq: np.ndarray, X: np.ndarray, Xt: np.ndarray) -> float:
    """||Wq·X - W·X̃||_F^2 / (m·k) — the GPTAQ asymmetric objective.
    Aligns quantized Quant-flow output with FP-flow reference.
    Division by m·k (np.mean over all elements of the m×k output matrix)."""
    return float(np.mean((Wq @ X - W @ Xt) ** 2))


def unmodified_weight_drift_error(W: np.ndarray, X: np.ndarray, Xt: np.ndarray) -> float:
    """||W·X - W·X̃||_F^2 / (m·k) when Wq = W (no quantization).
    This is the error from activation drift X̃ ≠ X with UNMODIFIED weights.
    NOT an irreducible floor — asymmetric calibration can deliberately change
    weights to partially compensate for this drift, achieving lower error."""
    return float(np.mean((W @ X - W @ Xt) ** 2))




# ==================== Adaptive alpha strategies ====================

def alpha_asymmetry_ratio(X: np.ndarray, Xt: np.ndarray) -> float:
    """α = ||ΔX · X^T||_F / ||X · X^T||_F
    Ratio of asymmetric cross-covariance to symmetric Hessian magnitude.
    Small when activation drift is small (early layers); large when drift is large."""
    dX = Xt - X
    num = np.linalg.norm(dX @ X.T, 'fro')
    den = np.linalg.norm(X @ X.T, 'fro')
    return float(num / (den + 1e-12))


def alpha_per_column(X: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Per-column adaptive α: diagonal ratio of cross-covariance drift to Hessian.
    α_j = |diag(ΔX · X^T)_j| / |diag(X · X^T)_j|
        = |sum_k(ΔX[j,k] · X[j,k])| / |sum_k(X[j,k] · X[j,k])|
    where ΔX = X̃ - X and j indexes input features (rows of X).
    This is the per-feature ratio of the diagonal of the cross-covariance
    to the diagonal of the Hessian."""
    dX = Xt - X
    # Diagonal of dX @ X^T per feature: sum over calibration samples of dX[j,k]*X[j,k]
    num = np.abs(np.sum(dX * X, axis=1))
    # Diagonal of X @ X^T per feature: sum over calibration samples of X[j,k]^2
    den = np.abs(np.sum(X * X, axis=1))
    return num / (den + 1e-12)


# ==================== Core quantization engine ====================

def quantize_core(
    W: np.ndarray,
    X: np.ndarray,
    Xt: np.ndarray,
    bits: int,
    tile: int,
    block: int,
    damping: float,
    *,
    use_gptaq: bool = False,
    use_rescomp: bool = False,
    alpha: float = 1.0,
    alpha_per_col: Optional[np.ndarray] = None,
    p_method: str = 'cholesky',
    correction_mode: str = 'weight',   # 'weight', 'error', 'blended'
    blend_coef: float = 1.0,
    n_iterations: int = 1,
) -> np.ndarray:
    """
    GPTQ lazy-batch quantization with optional GPTAQ / ResComp corrections.

    Parameters
    ----------
    use_gptaq : apply GPTAQ asymmetric correction (P-matrix from ΔX·X^T)
    use_rescomp : apply ResComp CAE correction (P2-matrix from X̃·X^T)
    alpha : scalar multiplier on P-matrix (0.25=reference, 1.0=paper-faithful, 0=off)
    alpha_per_col : if given, per-column alpha vector (overrides scalar)
    p_method : 'cholesky' or 'eigen' for H^{-1} factorization
    correction_mode :
        'weight'  — correction vector = w_pre (pre-quantization weight) [paper Eq 9/15]
        'error'   — correction vector = w_pre - Q[:,c] (quantization error, self-scaling)
        'blended' — correction vector = blend_coef·w_pre + (1-blend_coef)·Q[:,c]
    n_iterations : number of GPTAQ refinement passes (1=standard, 2+=iterative)
    """

    m, n = W.shape
    H = X @ X.T  # n×n input Hessian

    # Factor H^{-1}
    if p_method == 'eigen':
        U = eigen_factor(H, damping)
    else:
        U = cholesky_factor(H, damping)

    # Cross-covariance matrices
    dX = Xt - X  # ΔX

    # P-matrices (pre-computed once, block-invariant)
    if use_gptaq:
        if alpha_per_col is not None:
            # Per-column alpha: scale row c of P by alpha_per_col[c].
            # Column c correction uses P[c, c:], so row-level scaling gives
            # each column its own correction strength.
            P = compute_P(dX @ X.T, U, alpha=1.0)
            P = P * alpha_per_col[:, np.newaxis]
        else:
            P = compute_P(dX @ X.T, U, alpha=alpha)
    else:
        P = np.zeros((n, n))

    if use_rescomp:
        P2 = compute_P(Xt @ X.T, U, alpha=alpha if alpha_per_col is None else 1.0)
        if alpha_per_col is not None:
            P2 = P2 * alpha_per_col[:, np.newaxis]
    else:
        P2 = np.zeros((n, n))

    # Iterative refinement
    W_current = W.copy().astype(np.float64)
    Q_final = np.zeros_like(W)

    for iteration in range(n_iterations):
        Ww = W_current.copy()
        W0 = W_current.copy()  # original for this iteration
        Q = np.zeros_like(Ww)

        for i in range(0, n, block):
            B = min(block, n - i)
            E = np.zeros((m, B))
            # Cache pre-quantization compensated columns (for outer-block corrections)
            W_pre_cache = np.zeros((m, B))
            CAE_cache = np.zeros((m, B))
            Q_cache = np.zeros((m, B))

            for j in range(B):
                c = i + j

                # --- Save pre-quantization compensated weight ---
                w_pre = Ww[:, c].copy()
                W_pre_cache[:, j] = w_pre

                # --- Quantize current column ---
                Q[:, c] = quantize_column(Ww[:, c], bits, tile)
                Q_cache[:, j] = Q[:, c]

                # --- Standard GPTQ error and update ---
                e = w_pre - Q[:, c]
                E[:, j] = e / U[c, c]
                end = min(i + B, n)
                Ww[:, c:end] -= np.outer(E[:, j], U[c, c:end])

                # --- GPTAQ correction (inner block) ---
                if use_gptaq:
                    if correction_mode == 'weight':
                        corr = w_pre
                    elif correction_mode == 'error':
                        corr = w_pre - Q[:, c]
                    elif correction_mode == 'blended':
                        corr = blend_coef * w_pre + (1.0 - blend_coef) * Q[:, c]
                    else:
                        corr = w_pre
                    Ww[:, c:end] += np.outer(corr, P[c, c:end])

                # --- ResComp CAE correction (inner block) ---
                if use_rescomp:
                    cae = W0[:, c] - w_pre  # compensation-aware error
                    CAE_cache[:, j] = cae
                    Ww[:, c:end] += np.outer(cae, P2[c, c:end])

            # --- Outer lazy-block update ---
            if i + B < n:
                # GPTQ lazy
                Ww[:, i + B:] -= E @ U[i:i + B, i + B:]

                # GPTAQ outer (FIX: use cached w_pre, not original W)
                if use_gptaq:
                    if correction_mode == 'weight':
                        Ww[:, i + B:] += W_pre_cache @ P[i:i + B, i + B:]
                    elif correction_mode == 'error':
                        err_cache = W_pre_cache - Q_cache
                        Ww[:, i + B:] += err_cache @ P[i:i + B, i + B:]
                    elif correction_mode == 'blended':
                        blend_cache = blend_coef * W_pre_cache + (1.0 - blend_coef) * Q_cache
                        Ww[:, i + B:] += blend_cache @ P[i:i + B, i + B:]

                # ResComp outer (FIX: use cached CAE vectors, not W0-W which is zero)
                if use_rescomp:
                    Ww[:, i + B:] += CAE_cache @ P2[i:i + B, i + B:]

        Q_final = Q
        W_current = Ww  # for next iteration, quantize the compensated weights

    return Q_final


# ==================== Arm definitions ====================

def run_arm(name: str, W, X, Xt, bits, tile, block, damping) -> np.ndarray:
    """Dispatch to the appropriate quantization arm."""

    if name == 'rtn':
        return quantize_matrix(W, bits, tile)

    if name == 'gptq':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=False, use_rescomp=False)

    if name == 'gptaq_a0.25':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=0.25)

    if name == 'gptaq_a1.0':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=1.0)

    if name == 'gptaq_a0.5':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=0.5)

    if name == 'gptaq_adaptive':
        a = alpha_asymmetry_ratio(X, Xt)
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=a)

    if name == 'gptaq_per_col_adaptive':
        a_vec = alpha_per_column(X, Xt)
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha_per_col=a_vec)

    if name == 'gptaq_eigen':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=1.0, p_method='eigen')

    if name == 'rescomp_broken':
        """Original broken ResComp: outer term uses W0-W = 0."""
        m, n = W.shape
        H = X @ X.T
        U = cholesky_factor(H, damping)
        dX = Xt - X
        P2 = compute_P(Xt @ X.T, U, alpha=1.0)
        Ww = W.copy().astype(np.float64)
        W0 = W.copy()
        Q = np.zeros_like(Ww)
        for i in range(0, n, block):
            B = min(block, n - i)
            E = np.zeros((m, B))
            for j in range(B):
                c = i + j
                w_pre = Ww[:, c].copy()
                Q[:, c] = quantize_column(Ww[:, c], bits, tile)
                e = w_pre - Q[:, c]
                E[:, j] = e / U[c, c]
                end = min(i + B, n)
                Ww[:, c:end] -= np.outer(E[:, j], U[c, c:end])
                # ResComp inner (correct)
                cae = W0[:, c] - w_pre
                Ww[:, c:end] += np.outer(cae, P2[c, c:end])
            if i + B < n:
                Ww[:, i + B:] -= E @ U[i:i + B, i + B:]
                # BUG: W0 - W = 0 always
                Ww[:, i + B:] += (W0[:, i:i + B] - W[:, i:i + B]) @ P2[i:i + B, i + B:]
        return Q

    if name == 'rescomp_fixed':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_rescomp=True, alpha=1.0)

    if name == 'gptaq_rescomp':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, use_rescomp=True, alpha=1.0)

    if name == 'error_vec':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=1.0, correction_mode='error')

    if name == 'blended':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=1.0, correction_mode='blended', blend_coef=0.5)

    if name == 'repeated_requant_2':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=1.0, n_iterations=2)

    if name == 'repeated_requant_3':
        return quantize_core(W, X, Xt, bits, tile, block, damping,
                             use_gptaq=True, alpha=1.0, n_iterations=3)

    raise ValueError(f"Unknown arm: {name}")


# Grid search alpha (special handling — picks best alpha by asymmetric error)
def run_gptaq_grid(W, X, Xt, bits, tile, block, damping, alpha_grid):
    best_q, best_err, best_a = None, 1e18, 0.0
    for a in alpha_grid:
        Q = quantize_core(W, X, Xt, bits, tile, block, damping,
                          use_gptaq=True, alpha=a)
        err = asymmetric_error(W, Q, X, Xt)
        if err < best_err:
            best_err, best_q, best_a = err, Q, a
    return best_q, best_a


# ==================== Tensor generation ====================

def gen_tensors(cfg: Config, seed: int):
    """Generate synthetic W, X (quant-flow), Xt (FP-flow).
    X and Xt differ by a controlled drift, simulating inter-layer accumulation."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((cfg.m, cfg.n)) * 0.05
    Xt = rng.standard_normal((cfg.n, cfg.k)) * 0.5
    # Add outlier channels (realistic for LLM activations)
    outlier_mask = rng.random(cfg.n) < 0.05
    Xt[outlier_mask] *= 5.0
    # Quant-flow X = FP-flow Xt + noise (simulates quantization drift from prior layers)
    drift = rng.standard_normal((cfg.n, cfg.k)) * 0.02
    # Scale drift with depth (later layers have more drift)
    X = Xt + drift
    return W, X, Xt


def gen_tensors_large_drift(cfg: Config, seed: int):
    """Large-drift variant: simulates late-layer (L55) accumulation."""
    rng = np.random.default_rng(seed + 1000)
    W = rng.standard_normal((cfg.m, cfg.n)) * 0.05
    Xt = rng.standard_normal((cfg.n, cfg.k)) * 0.5
    Xt[rng.random(cfg.n) < 0.05] *= 5.0
    drift = rng.standard_normal((cfg.n, cfg.k)) * 0.15  # 7.5× larger drift
    X = Xt + drift
    return W, X, Xt


# ==================== Real weight loading ====================

def load_real_weights():
    """Load real Qwen3.8-27B weights and create 128×128 slices."""
    path = '/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz'
    data = np.load(path)
    slices = {}
    for key in data.files:
        W_full = data[key].astype(np.float64)
        # Take first 128×128 slice
        m, n = W_full.shape
        m_sl, n_sl = min(128, m), min(128, n)
        slices[key] = W_full[:m_sl, :n_sl]
    return slices


def gen_calibration_for_real(W_slice, cfg, seed):
    """Generate synthetic calibration data for real weight slices."""
    rng = np.random.default_rng(seed)
    n = W_slice.shape[1]
    Xt = rng.standard_normal((n, cfg.k)) * 0.5
    Xt[rng.random(n) < 0.05] *= 5.0
    X = Xt + rng.standard_normal((n, cfg.k)) * 0.02
    return X, Xt


# ==================== Experiment runner ====================

ARMS = [
    'rtn',
    'gptq',
    'gptaq_a0.25',
    'gptaq_a1.0',
    'gptaq_a0.5',
    'gptaq_adaptive',
    'gptaq_per_col_adaptive',
    'gptaq_eigen',
    'rescomp_broken',
    'rescomp_fixed',
    'gptaq_rescomp',
    'error_vec',
    'blended',
    'repeated_requant_2',
    'repeated_requant_3',
]

ARMS_GRID = ['gptaq_grid']


def run_experiment(cfg: Config, use_real: bool = False, large_drift: bool = False):
    """Run full experiment matrix."""
    all_results = []

    if use_real:
        real_slices = load_real_weights()
        tensor_keys = list(real_slices.keys())
    else:
        tensor_keys = None

    for seed in range(cfg.seeds):
        if use_real:
            for key in tensor_keys:
                W = real_slices[key]
                X, Xt = gen_calibration_for_real(W, cfg, seed)
                H = X @ X.T
                nf = unmodified_weight_drift_error(W, X, Xt)
                for bits in cfg.bits_list:
                    for arm in ARMS:
                        Q = run_arm(arm, W, X, Xt, bits, cfg.tile, 128, cfg.damping)
                        all_results.append({
                            'tensor': key, 'seed': seed, 'arm': arm, 'bits': bits,
                            'wmse': weight_mse(W, Q),
                            'hwe': hessian_weighted_error(W, Q, H),
                            'asym_err': asymmetric_error(W, Q, X, Xt),
                            'noise_floor': nf,
                        })
                    # Grid search arm (same bits value, no duplicate loop)
                    Q, best_a = run_gptaq_grid(W, X, Xt, bits, cfg.tile, 128,
                                                cfg.damping, cfg.alpha_grid)
                    all_results.append({
                        'tensor': key, 'seed': seed, 'arm': 'gptaq_grid', 'bits': bits,
                        'wmse': weight_mse(W, Q),
                        'hwe': hessian_weighted_error(W, Q, H),
                        'asym_err': asymmetric_error(W, Q, X, Xt),
                        'noise_floor': nf,
                        'best_alpha': best_a,
                    })
        else:
            gen = gen_tensors_large_drift if large_drift else gen_tensors
            W, X, Xt = gen(cfg, seed)
            H = X @ X.T
            nf = unmodified_weight_drift_error(W, X, Xt)
            for bits in cfg.bits_list:
                for arm in ARMS:
                    Q = run_arm(arm, W, X, Xt, bits, cfg.tile, 128, cfg.damping)
                    entry = {
                        'tensor': f'{"large_drift" if large_drift else "synthetic"}',
                        'seed': seed, 'arm': arm, 'bits': bits,
                        'wmse': weight_mse(W, Q),
                        'hwe': hessian_weighted_error(W, Q, H),
                        'asym_err': asymmetric_error(W, Q, X, Xt),
                        'noise_floor': nf,
                    }
                    # Serialize adaptive alpha values
                    if arm == 'gptaq_adaptive':
                        entry['chosen_alpha'] = alpha_asymmetry_ratio(X, Xt)
                    elif arm == 'gptaq_per_col_adaptive':
                        entry['chosen_alpha_mean'] = float(np.mean(alpha_per_column(X, Xt)))
                        entry['chosen_alpha_std'] = float(np.std(alpha_per_column(X, Xt)))
                    all_results.append(entry)
                # Grid search (in-sample discrete-grid minimum)
                Q, best_a = run_gptaq_grid(W, X, Xt, bits, cfg.tile, 128,
                                            cfg.damping, cfg.alpha_grid)
                all_results.append({
                    'tensor': f'{"large_drift" if large_drift else "synthetic"}',
                    'seed': seed, 'arm': 'gptaq_grid', 'bits': bits,
                    'wmse': weight_mse(W, Q),
                    'hwe': hessian_weighted_error(W, Q, H),
                    'asym_err': asymmetric_error(W, Q, X, Xt),
                    'noise_floor': nf,
                    'best_alpha': best_a,
                })

    return all_results


def run_block_size_comparison(cfg: Config):
    """Compare ResComp broken vs fixed at block=16 (where the bug manifests).
    At block=128=n there's no outer-block update, so broken=fixed.
    At block=16 there are 8 blocks, so the broken outer term (always 0) matters."""
    W, X, Xt = gen_tensors(cfg, seed=0)
    H = X @ X.T
    nf = unmodified_weight_drift_error(W, X, Xt)
    bits = 4
    results = {}
    for block in [16, 32, 64, 128]:
        row = {}
        for arm in ['gptq', 'gptaq_a1.0', 'rescomp_broken', 'rescomp_fixed', 'gptaq_rescomp']:
            Q = run_arm(arm, W, X, Xt, bits, cfg.tile, block, cfg.damping)
            row[arm] = {
                'asym_err': asymmetric_error(W, Q, X, Xt),
                'hwe': hessian_weighted_error(W, Q, H),
            }
        results[f'block={block}'] = row
    return results


def run_block_invariance_test(cfg: Config):
    """Test that results are invariant to block size.
    Standard GPTQ should be invariant to ~1e-16.
    ResComp broken should NOT be invariant.
    ResComp fixed SHOULD be invariant.
    GPTAQ should be invariant (with cached w_pre)."""
    W, X, Xt = gen_tensors(cfg, seed=0)
    results = {}
    bits = 4

    test_arms = ['gptq', 'gptaq_a1.0', 'rescomp_broken', 'rescomp_fixed', 'gptaq_rescomp']

    for arm in test_arms:
        qs = []
        for block in cfg.block_sizes:
            Q = run_arm(arm, W, X, Xt, bits, cfg.tile, block, cfg.damping)
            qs.append(Q)
        # Measure max deviation between block sizes
        max_dev = max(np.max(np.abs(qs[i] - qs[j]))
                      for i in range(len(qs)) for j in range(i + 1, len(qs)))
        results[arm] = {
            'max_block_deviation': float(max_dev),
            'block_invariant': max_dev < 1e-10,
        }

    return results


def aggregate_results(results):
    """Aggregate across seeds, compute means and stds."""
    agg = {}
    for r in results:
        key = (r['tensor'], r['arm'], r['bits'])
        if key not in agg:
            agg[key] = {m: [] for m in ['wmse', 'hwe', 'asym_err', 'noise_floor']}
            if 'best_alpha' in r:
                agg[key]['best_alpha'] = []
        for m in ['wmse', 'hwe', 'asym_err', 'noise_floor']:
            agg[key][m].append(r[m])
        if 'best_alpha' in r:
            agg[key]['best_alpha'].append(r['best_alpha'])

    out = []
    for (tensor, arm, bits), v in sorted(agg.items()):
        entry = {
            'tensor': tensor, 'arm': arm, 'bits': bits,
            'wmse_mean': float(np.mean(v['wmse'])),
            'hwe_mean': float(np.mean(v['hwe'])),
            'asym_err_mean': float(np.mean(v['asym_err'])),
            'noise_floor': float(np.mean(v['noise_floor'])),
            'asym_err_std': float(np.std(v['asym_err'])),
        }
        if 'best_alpha' in v and v['best_alpha']:
            entry['best_alpha_mean'] = float(np.mean(v['best_alpha']))
        out.append(entry)
    return out


def print_tables(agg, bits_list, tensor_name='synthetic'):
    """Print formatted results tables."""
    rows = [r for r in agg if r['tensor'] == tensor_name]

    for bits in bits_list:
        print(f"\n{'=' * 120}")
        print(f"  K{bits} ({bits}-bit) — {tensor_name}")
        print(f"{'=' * 120}")
        nf = next((r['noise_floor'] for r in rows if r['bits'] == bits), 0)
        print(f"  Unmodified-weight drift: {nf:.6e}  (NOT an irreducible floor — corrections can reduce this)")
        print(f"  {'Arm':<28} {'Asym Err':>12} {'vs RTN':>9} {'vs GPTQ':>9} {'HW Err':>12} {'Wt MSE':>12}")
        print(f"  {'-' * 28} {'-' * 12} {'-' * 9} {'-' * 9} {'-' * 12} {'-' * 12}")

        rtn = next((r['asym_err_mean'] for r in rows if r['bits'] == bits and r['arm'] == 'rtn'), 1e-10)
        gptq = next((r['asym_err_mean'] for r in rows if r['bits'] == bits and r['arm'] == 'gptq'), 1e-10)

        bit_rows = sorted([r for r in rows if r['bits'] == bits], key=lambda x: x['asym_err_mean'])
        for r in bit_rows:
            ae = r['asym_err_mean']
            vs_rtn = f"{(1 - ae / rtn) * 100:+.1f}%" if r['arm'] != 'rtn' else ""
            vs_gptq = f"{(1 - ae / gptq) * 100:+.1f}%" if r['arm'] != 'gptq' and r['arm'] != 'rtn' else ""
            extra = f"  α={r.get('best_alpha_mean', 0):.2f}" if 'best_alpha_mean' in r else ""
            print(f"  {r['arm']:<28} {ae:>12.4e} {vs_rtn:>9} {vs_gptq:>9} "
                  f"{r['hwe_mean']:>12.4e} {r['wmse_mean']:>12.4e}{extra}")

    # Best per K summary
    print(f"\n{'=' * 120}")
    print(f"  BEST ARM PER K — {tensor_name}")
    print(f"{'=' * 120}")
    print(f"  {'K':>4} {'Best Arm':<28} {'Asym Err':>12} {'vs RTN':>9} {'vs GPTQ':>9} {'Floor':>12}")
    print(f"  {'-' * 4} {'-' * 28} {'-' * 12} {'-' * 9} {'-' * 9} {'-' * 12}")
    for bits in bits_list:
        bit_rows = [r for r in rows if r['bits'] == bits]
        if not bit_rows:
            continue
        best = min(bit_rows, key=lambda x: x['asym_err_mean'])
        rtn = next((r['asym_err_mean'] for r in rows if r['bits'] == bits and r['arm'] == 'rtn'), 1e-10)
        gptq = next((r['asym_err_mean'] for r in rows if r['bits'] == bits and r['arm'] == 'gptq'), 1e-10)
        nf = best['noise_floor']
        print(f"  K{bits:<2} {best['arm']:<28} {best['asym_err_mean']:>12.4e} "
              f"{(1 - best['asym_err_mean'] / rtn) * 100:>+8.1f}% "
              f"{(1 - best['asym_err_mean'] / gptq) * 100:>+8.1f}% {nf:>12.4e}")


# ==================== Main ====================

def main():
    cfg = Config()
    t0 = time.time()

    # ---- Block invariance test ----
    print("=" * 120)
    print("  BLOCK-SIZE INVARIANCE TEST (block ∈ {1, 16, 128})")
    print("=" * 120)
    inv = run_block_invariance_test(cfg)
    for arm, info in inv.items():
        status = "✓ INVARIANT" if info['block_invariant'] else "✗ NOT INVARIANT"
        print(f"  {arm:<28} max_dev={info['max_block_deviation']:.2e}  {status}")

    # ---- Block-size comparison (shows ResComp broken vs fixed at different blocks) ----
    print("\n" + "=" * 120)
    print("  BLOCK-SIZE COMPARISON (K4, block ∈ {16, 32, 64, 128})")
    print("=" * 120)
    bsc = run_block_size_comparison(cfg)
    print(f"  {'Arm':<28}", end='')
    for block in [16, 32, 64, 128]:
        print(f" {'block=' + str(block):>14}", end='')
    print()
    print(f"  {'-' * 28}", end='')
    for _ in [16, 32, 64, 128]:
        print(f" {'-' * 14}", end='')
    print()
    for arm in ['gptq', 'gptaq_a1.0', 'rescomp_broken', 'rescomp_fixed', 'gptaq_rescomp']:
        print(f"  {arm:<28}", end='')
        for block in [16, 32, 64, 128]:
            v = bsc[f'block={block}'][arm]['asym_err']
            print(f" {v:>14.4e}", end='')
        print()
    # ---- Synthetic experiment (normal drift) ----
    print("\n" + "=" * 120)
    print("  SYNTHETIC EXPERIMENT (normal drift)")
    print("=" * 120)
    results_synth = run_experiment(cfg, use_real=False, large_drift=False)
    agg_synth = aggregate_results(results_synth)
    print_tables(agg_synth, cfg.bits_list, 'synthetic')

    # ---- Synthetic experiment (large drift — late layer simulation) ----
    print("\n" + "=" * 120)
    print("  SYNTHETIC EXPERIMENT (large drift — late layer)")
    print("=" * 120)
    results_large = run_experiment(cfg, use_real=False, large_drift=True)
    agg_large = aggregate_results(results_large)
    print_tables(agg_large, cfg.bits_list, 'large_drift')

    # ---- Real weight experiment ----
    print("\n" + "=" * 120)
    print("  REAL WEIGHT EXPERIMENT (Qwen3.8-27B, 128×128 slices)")
    print("=" * 120)
    results_real = run_experiment(cfg, use_real=True)
    agg_real = aggregate_results(results_real)

    # Print per-tensor best
    real_tensors = sorted(set(r['tensor'] for r in agg_real))
    for tensor in real_tensors:
        print_tables(agg_real, cfg.bits_list, tensor)

    # ---- Save results ----
    output = {
        'config': cfg.__dict__,
        'block_invariance': inv,
        'block_size_comparison': bsc,
        'synthetic': agg_synth,
        'large_drift': agg_large,
        'real_weights': agg_real,
        'elapsed_seconds': time.time() - t0,
    }
    out_path = '/Users/mbelleau/Projects/qwen38-research-r2-gptaq-corrections/receipts/research/r2-gptaq-corrections-results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
