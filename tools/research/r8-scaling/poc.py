#!/usr/bin/env python3
"""
R8-Scaling: AWQ/SmoothQuant alternatives and novel scaling strategies.

FIXED v2 — addresses reviewer blockers:
  1. inv_cholesky returns UPPER triangular (verified)
  2. ALL arms use matched per-tile 16×16 quantizer (no Hadamard mismatch)
  3. Per-tile quantizer does NOT commute with per-channel scaling (unlike per-column)
  4. Kurtosis/Hessian shape bugs fixed
  5. Added GPTQ-no-P arm to separate GPTQ error propagation from GPTAQ P correction
  6. No broad exception swallowing — errors crash visibly
  7. inv_cholesky validated: L is upper triangular, L^T @ L = inv(H+λI)

Strategies tested (14 total):
  1. No scaling (baseline)
  2. AWQ (mean|X|^alpha, geom-normalized)
  3. SmoothQuant (max|X|^alpha / max|W|^{1-alpha})
  4. Activation variance scaling: Var(X_j)^alpha
  5. Activation kurtosis scaling: kurt(X_j)^alpha
  6. Weight-activation product: (mean|X_j| * max|W_{:,j}|)^alpha  [NOVEL]
  7. Per-tile scaling: per 16x16 tile scales
  8. Hessian-based scaling: H_X,jj^{1/2}
  9. Lp-norm p=1: ||X_j||_1^alpha (= AWQ)
  10. Lp-norm p=2: ||X_j||_2^alpha (RMS)
  11. Lp-norm p=inf: ||X_j||_inf^alpha (max|X|)
  12. Outlier-aware: top-k% channels boosted
  13. Per-channel adaptive alpha: alpha_j = sigmoid(log(|X_j|/|W_j|))  [NOVEL]
  14. Paper-faithful AWQ: mean|X|^alpha, no normalization

Each strategy tested with:
  - No correction (raw tile quantization)
  - GPTQ error propagation (no P correction)
  - GPTAQ (GPTQ + P correction)
"""

import numpy as np
import json
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== Configuration ====================

@dataclass
class ScalingConfig:
    m: int = 128
    n: int = 128
    k: int = 512
    tile_size: int = 16
    block_size: int = 16
    damping: float = 0.01
    bits_list: tuple = (4, 5, 6)
    num_seeds: int = 3
    alpha_default: float = 0.5
    outlier_pct: float = 5.0
    outlier_boost: float = 4.0

# ==================== Utilities ====================

def softmax(x, axis=0):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# ==================== Quantizer (matched per-tile 16x16 for ALL arms) ====================

def quantize_uniform(w, bits):
    """Uniform quantization with adaptive min/max range."""
    nl = 2 ** bits
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-12:
        return np.full_like(w, lo)
    s = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / s), 0, nl - 1) * s + lo

def trellis_quantize(W, bits, tile):
    """Per-tile 16x16 uniform quantization. Each tile gets ONE range.
    This is the matched quantizer for ALL arms."""
    m, n = W.shape
    Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            Wq[i:i+tile, j:j+tile] = quantize_uniform(W[i:i+tile, j:j+tile], bits)
    return Wq

def quantize_col_on_grids(Ww, c, bits, grids, tile, m):
    """Quantize column c onto FIXED tile grids. Each output-row tile uses
    its own (lo, step) from grids, but all 16 columns in the block share
    the same grids — matching trellis_quantize's one-range-per-tile."""
    nl = 2 ** bits
    col_q = np.zeros(m)
    for r_idx, r in enumerate(range(0, m, tile)):
        r_end = min(r + tile, m)
        tlo, step = grids[r_idx]
        if step == 0.0:
            col_q[r:r_end] = tlo
        else:
            col_q[r:r_end] = np.clip(
                np.round((Ww[r:r_end, c] - tlo) / step), 0, nl - 1) * step + tlo
    return col_q

def cache_tile_grids(Ww, bits, tile, block_start, block_end):
    """Precompute (lo, step) for each output-row tile in columns [block_start, block_end).
    Grid is computed from Ww BEFORE any column is quantized, so all 16 columns
    in the block share the same grid — matching trellis_quantize's one-range-per-tile."""
    m = Ww.shape[0]
    grids = []
    nl = 2 ** bits
    for r in range(0, m, tile):
        r_end = min(r + tile, m)
        tile_block = Ww[r:r_end, block_start:block_end]
        tlo, thi = float(tile_block.min()), float(tile_block.max())
        if thi - tlo < 1e-12:
            step = 0.0
        else:
            step = (thi - tlo) / (nl - 1)
        grids.append((tlo, step))
    return grids

# ==================== Loss functions ====================

def kld_loss(y_fp, y_q):
    p = np.clip(softmax(y_fp, 0), 1e-12, 1)
    q = np.clip(softmax(y_q, 0), 1e-12, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=0)))

def wt_mse(W, Wq):
    return float(np.mean((W - Wq) ** 2))

def out_mse(W, Wq, X):
    return float(np.mean((W @ X - Wq @ X) ** 2))

def hessian_weighted_error(W, Wq, X):
    """tr(E @ H_X @ E^T) where H_X = X @ X^T / k (n×n), E = W - Wq (m×n)."""
    E = W - Wq
    H_X = X @ X.T / X.shape[1]
    return float(np.sum(E @ H_X * E))

def noise_floor(W, X, Xt):
    """No-quant baseline: compare W@Xt vs W@X (activation drift floor)."""
    return kld_loss(W @ Xt, W @ X)

# ==================== Cholesky (v3: correct GPTQ convention U^T@U=H^{-1}) ====================

def inv_cholesky(H, damping):
    """Upper triangular Cholesky factor U of inv(H+damping*I) such that
    U^T @ U = inv(H+damping*I).  GPTQ row update requires this convention.

    Damping is RELATIVE: lambda = damping * mean(diag(H)), per GPTQ reference."""
    lam = damping * np.mean(np.diag(H))
    damped = H + lam * np.eye(H.shape[0])
    Ainv = np.linalg.inv(damped)
    U = np.linalg.cholesky(Ainv).T  # upper triangular, U^T @ U = Ainv
    assert np.allclose(np.tril(U, -1), 0), "inv_cholesky: U is NOT upper triangular!"
    assert np.allclose(U.T @ U, Ainv, atol=1e-10), "inv_cholesky: U^T @ U != inv(H+λI)!"
    return U

def compute_P(dX_Xt, L, n, alpha=0.25):
    """P-matrix: 0.25 * triu(D @ L.T, 1) @ L where L = inv_cholesky (U^T@U=H^{-1})."""
    M = np.triu(dX_Xt @ L.T, 1)
    return alpha * (M @ L)

# ==================== GPTQ/GPTAQ (v3: fixed grid, cached w_pre, correct Cholesky, relative damping) ====================

def gptq_quantize(W, X, Xt, bits, tile, block, damping, use_p=True):
    """GPTQ with optional GPTAQ P-matrix correction.
    Uses MATCHED per-tile 16x16 quantizer: grids cached at block start,
    so all 16 columns in a block share the same tile grid.

    Fixes: (1) U^T@U=H^{-1} Cholesky convention, (2) cached tile grids,
    (3) cached w_pre for outer P update, (4) relative damping.
    """
    assert block == tile, f"gptq_quantize requires block==tile (got block={block}, tile={tile})"
    m, n = W.shape
    Ww = W.copy().astype(np.float64)
    Q = np.zeros_like(Ww)

    H = X @ X.T
    L = inv_cholesky(H, damping)  # U^T @ U = inv(H+λI)

    P = compute_P((Xt - X) @ X.T, L, n) if use_p else np.zeros((n, n))

    for i in range(0, n, block):
        B = min(block, n - i)
        E = np.zeros((m, B))
        W_pre_block = np.zeros((m, B))  # cache compensated w_pre for outer P update

        # Cache tile grids from CURRENT Ww (before any column is quantized)
        grids = cache_tile_grids(Ww, bits, tile, i, i + B)

        for j in range(B):
            c = i + j
            w_pre = Ww[:, c].copy()
            W_pre_block[:, j] = w_pre  # cache for outer update

            # Quantize column c onto the CACHED tile grid (not refitted)
            Q[:, c] = quantize_col_on_grids(Ww, c, bits, grids, tile, m)

            e = (w_pre - Q[:, c])
            E[:, j] = e / L[c, c]
            end = min(i + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            if use_p:
                Ww[:, c:end] += np.outer(w_pre, P[c, c:end])

        # Outer lazy block update — use CACHED w_pre, not original W
        if i + B < n:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
            if use_p:
                Ww[:, i+B:] += W_pre_block @ P[i:i+B, i+B:]

    return Q

# ==================== Scaling strategies ====================

def apply_scales(W, X, Xt, s):
    """Apply scaling: W' = W * diag(s), X' = X / diag(s), Xt' = Xt / diag(s)."""
    return W * s[None, :], X / s[:, None], Xt / s[:, None]

# --- Strategy 1: AWQ ---
def awq_scales(W, X, alpha=0.5):
    a = np.mean(np.abs(X), axis=1)  # (n,)
    r = np.power(np.clip(a, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    return s  # (n,)

# --- Strategy 2: SmoothQuant ---
def smoothquant_scales(W, X, alpha=0.5):
    a = np.max(np.abs(X), axis=1)  # (n,)
    w = np.max(np.abs(W), axis=0)  # (n,) max over rows per input channel
    s = np.power(np.clip(a, 1e-10, None), alpha) / np.power(np.clip(w, 1e-5, None), 1 - alpha)
    s = np.clip(s, 1e-5, None)
    return s  # (n,)

# --- Strategy 3: Variance ---
def variance_scales(W, X, alpha=0.5):
    v = np.var(X, axis=1)  # (n,)
    r = np.power(np.clip(v, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    return s  # (n,)

# --- Strategy 4: Kurtosis (FIXED: squeeze sigma to avoid broadcast bug) ---
def kurtosis_scales(W, X, alpha=0.5):
    """s_j = kurt(X_j)^alpha. Excess kurtosis, clipped to >= 0."""
    mu = np.mean(X, axis=1)  # (n,)
    sigma = np.std(X, axis=1)  # (n,) — NO keepdims to avoid broadcast bug
    kurt = np.mean((X - mu[:, None]) ** 4, axis=1) / np.power(np.clip(sigma, 1e-10, None), 4) - 3.0
    kurt = np.clip(kurt, 0, None)  # (n,)
    r = np.power(np.clip(kurt + 1.0, 1e-4, None), alpha)  # +1 so kurt=0 -> r=1
    if np.max(r) > np.min(r):
        s = r / np.sqrt(np.max(r) * np.min(r))
    else:
        s = np.ones_like(r)
    return s  # (n,)

# --- Strategy 5: Weight-activation product (NOVEL) ---
def product_scales(W, X, alpha=0.5):
    a_x = np.mean(np.abs(X), axis=1)  # (n,)
    a_w = np.max(np.abs(W), axis=0)   # (n,)
    product = a_x * a_w
    r = np.power(np.clip(product, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    return s  # (n,)

# --- Strategy 6: Per-tile scaling ---
def per_tile_scales(W, X, alpha=0.5, tile=16):
    m, n = W.shape
    n_tiles_row = (m + tile - 1) // tile
    n_tiles_col = (n + tile - 1) // tile
    a = np.mean(np.abs(X), axis=1)  # (n,)
    tile_scales = np.ones((n_tiles_row, n_tiles_col))
    for tj in range(n_tiles_col):
        j_start = tj * tile
        j_end = min(j_start + tile, n)
        tile_act = np.mean(a[j_start:j_end])
        tile_scales[:, tj] = np.power(np.clip(tile_act, 1e-10, None), alpha)
    return tile_scales

def apply_tile_scales(W, X, Xt, tile_scales, tile=16):
    m, n = W.shape
    W_p = W.copy().astype(np.float64)
    n_tiles_col = tile_scales.shape[1]
    for ti in range(tile_scales.shape[0]):
        for tj in range(n_tiles_col):
            i_s, j_s = ti * tile, tj * tile
            i_e, j_e = min(i_s + tile, m), min(j_s + tile, n)
            W_p[i_s:i_e, j_s:j_e] *= tile_scales[ti, tj]
    s_chan = np.ones(n)
    for tj in range(n_tiles_col):
        j_s, j_e = tj * tile, min(tj * tile + tile, n)
        s_chan[j_s:j_e] = tile_scales[0, tj]
    X_p = X / s_chan[:, None]
    Xt_p = Xt / s_chan[:, None]
    return W_p, X_p, Xt_p, s_chan

def invert_tile_scales(Wq_p, tile_scales, tile=16):
    m, n = Wq_p.shape
    Wq = Wq_p.copy().astype(np.float64)
    for ti in range(tile_scales.shape[0]):
        for tj in range(tile_scales.shape[1]):
            i_s, j_s = ti * tile, tj * tile
            i_e, j_e = min(i_s + tile, m), min(j_s + tile, n)
            Wq[i_s:i_e, j_s:j_e] /= tile_scales[ti, tj]
    return Wq

# --- Strategy 7: Hessian-based (FIXED: X @ X.T not X.T @ X) ---
def hessian_scales(W, X, alpha=0.5):
    """s_j = H_X,jj^{alpha/2}. H_X = X @ X^T / k is (n, n) input-channel Hessian."""
    H_X = X @ X.T / X.shape[1]  # (n, n) — FIXED: was X.T @ X giving (k, k)
    h_diag = np.diag(H_X)  # (n,)
    r = np.power(np.clip(h_diag, 1e-10, None), alpha / 2.0)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    return s  # (n,)

# --- Strategy 8: Lp-norm ---
def lpnorm_scales(W, X, alpha=0.5, p=2.0):
    if p >= 100:
        a = np.max(np.abs(X), axis=1)
    else:
        a = np.power(np.mean(np.power(np.abs(X), p), axis=1), 1.0 / p)
    r = np.power(np.clip(a, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    return s  # (n,)

# --- Strategy 9: Outlier-aware ---
def outlier_scales(W, X, alpha=0.5, pct=5.0, boost=4.0):
    a = np.mean(np.abs(X), axis=1)
    threshold = np.percentile(a, 100 - pct)
    is_outlier = a >= threshold
    r = np.power(np.clip(a, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    s = r / np.sqrt(np.max(r) * np.min(r))
    s[is_outlier] *= boost
    s = s / np.sqrt(np.max(s) * np.min(s))
    return s  # (n,)

# --- Strategy 10: Per-channel adaptive alpha (NOVEL) ---
def adaptive_alpha_scales(W, X):
    """alpha_j = sigmoid(log(|X_j|/|W_j|)). Per-channel SmoothQuant."""
    a_x = np.max(np.abs(X), axis=1)  # (n,)
    a_w = np.max(np.abs(W), axis=0)  # (n,)
    log_ratio = np.log(np.clip(a_x, 1e-10, None) / np.clip(a_w, 1e-10, None))
    alpha_j = 1.0 / (1.0 + np.exp(-log_ratio))  # sigmoid, (n,)
    s = np.power(np.clip(a_x, 1e-10, None), alpha_j) / \
        np.power(np.clip(a_w, 1e-5, None), 1.0 - alpha_j)
    s = np.clip(s, 1e-5, None)
    return s  # (n,)

# --- Strategy 11: Paper-faithful AWQ (no normalization) ---
def awq_no_norm_scales(W, X, alpha=0.5):
    a = np.mean(np.abs(X), axis=1)
    r = np.power(np.clip(a, 1e-10, None), alpha)
    r = np.clip(r, 1e-4, None)
    return r  # (n,) — no normalization

# ==================== Byte budget accounting ====================

def sidecar_bytes_channel(n, dtype_bytes=2):
    return n * dtype_bytes

def sidecar_bytes_tile(m, n, tile=16, dtype_bytes=2):
    n_tr = (m + tile - 1) // tile
    n_tc = (n + tile - 1) // tile
    return n_tr * n_tc * dtype_bytes

def weight_bytes(m, n, bits):
    return m * n * bits / 8

def total_bytes(m, n, bits, scale_key, tile=16):
    wb = weight_bytes(m, n, bits)
    if scale_key == "none":
        return wb
    elif scale_key == "per_tile":
        return wb + sidecar_bytes_tile(m, n, tile)
    else:
        return wb + sidecar_bytes_channel(n)

# ==================== Experiment pipeline ====================

def run_arm(W, X, Xt, bits, cfg, scale_fn=None, tile_scales=False, correction="none"):
    """Run one experiment arm.
    
    correction: "none" = raw tile quantization
                "gptq" = GPTQ error propagation (no P)
                "gptaq" = GPTQ + P correction
    """
    if scale_fn is None:
        # No scaling
        if correction == "none":
            return trellis_quantize(W, bits, cfg.tile_size)
        elif correction == "gptq":
            return gptq_quantize(W, X, Xt, bits, cfg.tile_size, cfg.block_size, cfg.damping, use_p=False)
        elif correction == "gptaq":
            return gptq_quantize(W, X, Xt, bits, cfg.tile_size, cfg.block_size, cfg.damping, use_p=True)
    
    if tile_scales:
        ts = scale_fn(W, X, alpha=cfg.alpha_default, tile=cfg.tile_size)
        W_p, X_p, Xt_p, s_chan = apply_tile_scales(W, X, Xt, ts, cfg.tile_size)
    else:
        s = scale_fn(W, X, alpha=cfg.alpha_default) if 'alpha' in scale_fn.__code__.co_varnames else scale_fn(W, X)
        W_p, X_p, Xt_p = apply_scales(W, X, Xt, s)
    
    if correction == "none":
        Wq_p = trellis_quantize(W_p, bits, cfg.tile_size)
    elif correction == "gptq":
        Wq_p = gptq_quantize(W_p, X_p, Xt_p, bits, cfg.tile_size, cfg.block_size, cfg.damping, use_p=False)
    elif correction == "gptaq":
        Wq_p = gptq_quantize(W_p, X_p, Xt_p, bits, cfg.tile_size, cfg.block_size, cfg.damping, use_p=True)
    
    if tile_scales:
        Wq = invert_tile_scales(Wq_p, ts, cfg.tile_size)
    else:
        Wq = Wq_p / s[None, :]
    
    return Wq

# ==================== Tensor generation ====================

def gen_synthetic_tensors(cfg, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((cfg.m, cfg.n)) * 0.05
    Xt = rng.standard_normal((cfg.n, cfg.k)) * 0.5
    outlier_mask = rng.random(cfg.n) < 0.05
    Xt[outlier_mask] *= 5.0
    X = Xt + rng.standard_normal((cfg.n, cfg.k)) * 0.02
    return W, X, Xt

def load_real_tensor(tensor_name, cfg):
    data = np.load('/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz')
    W_full = data[tensor_name].astype(np.float64)
    m_full, n_full = W_full.shape
    m_start = max(0, (m_full - cfg.m) // 2)
    n_start = max(0, (n_full - cfg.n) // 2)
    W = W_full[m_start:m_start + cfg.m, n_start:n_start + cfg.n]
    rng = np.random.default_rng(42)
    Xt = rng.standard_normal((cfg.n, cfg.k)) * np.std(W, axis=0, keepdims=True).T
    outlier_mask = rng.random(cfg.n) < 0.05
    Xt[outlier_mask] *= 5.0
    X = Xt + rng.standard_normal((cfg.n, cfg.k)) * 0.02
    return W, X, Xt

# ==================== Experiment matrix ====================

SCALING_STRATEGIES = [
    ("none",         None,                     False, "none"),
    ("awq",          awq_scales,              False, "per_channel"),
    ("smoothquant",  smoothquant_scales,      False, "per_channel"),
    ("variance",     variance_scales,         False, "per_channel"),
    ("kurtosis",     kurtosis_scales,         False, "per_channel"),
    ("product",      product_scales,          False, "per_channel"),
    ("per_tile",     per_tile_scales,         True,  "per_tile"),
    ("hessian",      hessian_scales,          False, "per_channel"),
    ("lp_p1",        lambda W, X, alpha=0.5: lpnorm_scales(W, X, alpha, p=1.0),    False, "per_channel"),
    ("lp_p2",        lambda W, X, alpha=0.5: lpnorm_scales(W, X, alpha, p=2.0),    False, "per_channel"),
    ("lp_pinf",      lambda W, X, alpha=0.5: lpnorm_scales(W, X, alpha, p=100.0),  False, "per_channel"),
    ("outlier",      outlier_scales,          False, "per_channel"),
    ("adaptive_a",   adaptive_alpha_scales,   False, "per_channel"),
    ("awq_no_norm",  awq_no_norm_scales,      False, "per_channel"),
]

CORRECTIONS = ["none", "gptq", "gptaq"]

def run_experiment(cfg, W, X, Xt, tensor_name="synthetic"):
    results = []
    nf = noise_floor(W, X, Xt)

    for bits in cfg.bits_list:
        for name, scale_fn, is_tile, scale_key in SCALING_STRATEGIES:
            for corr in CORRECTIONS:
                full_name = f"{name}+{corr}"
                t0 = time.time()
                Wq = run_arm(W, X, Xt, bits, cfg, scale_fn, is_tile, corr)
                dt = time.time() - t0

                kld = kld_loss(W @ Xt, Wq @ X)
                wmse_val = wt_mse(W, Wq)
                omse_val = out_mse(W, Wq, Xt)
                hwe = hessian_weighted_error(W, Wq, X)
                total_b = total_bytes(cfg.m, cfg.n, bits, scale_key, cfg.tile_size)

                results.append({
                    "tensor": tensor_name,
                    "method": full_name,
                    "base_strategy": name,
                    "correction": corr,
                    "bits": bits,
                    "kld": kld,
                    "wmse": wmse_val,
                    "omse": omse_val,
                    "hwe": hwe,
                    "noise_floor": nf,
                    "total_bytes": total_b,
                    "time": dt,
                })
    return results

def run_full(cfg):
    all_results = []
    for seed in range(cfg.num_seeds):
        W, X, Xt = gen_synthetic_tensors(cfg, seed)
        r = run_experiment(cfg, W, X, Xt, f"synthetic_s{seed}")
        all_results.extend(r)
        print(f"  Synthetic seed {seed} done ({len(r)} arms)")

    real_tensors = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
    for tname in real_tensors:
        W, X, Xt = load_real_tensor(tname, cfg)
        r = run_experiment(cfg, W, X, Xt, tname)
        all_results.extend(r)
        print(f"  Real {tname} done ({len(r)} arms)")
    return all_results

# ==================== Analysis ====================

def analyze_results(results, cfg):
    agg = {}
    for r in results:
        key = (r["tensor"], r["method"], r["bits"])
        if key not in agg:
            agg[key] = []
        agg[key].append(r)

    summary = {}
    for key, runs in sorted(agg.items()):
        tensor, method, bits = key
        k = f"{tensor}|{method}|K{bits}"
        summary[k] = {
            "kld_mean": float(np.mean([r["kld"] for r in runs])),
            "kld_std": float(np.std([r["kld"] for r in runs])),
            "wmse_mean": float(np.mean([r["wmse"] for r in runs])),
            "omse_mean": float(np.mean([r["omse"] for r in runs])),
            "hwe_mean": float(np.mean([r["hwe"] for r in runs])),
            "total_bytes": runs[0]["total_bytes"],
            "noise_floor": runs[0]["noise_floor"],
            "n_runs": len(runs),
        }
    return summary

def print_tables(summary, cfg):
    tensors = sorted(set(k.split("|")[0] for k in summary))

    for tensor in tensors:
        for bits in cfg.bits_list:
            print(f"\n{'='*130}")
            print(f"  {tensor} @ K{bits}")
            print(f"{'='*130}")

            arms = []
            for k, v in summary.items():
                if f"{tensor}|" in k and f"|K{bits}" in k:
                    method = k.split("|")[1]
                    arms.append((method, v))

            if not arms:
                continue

            # Baselines
            none_none = next((v["kld_mean"] for m, v in arms if m == "none+none"), 1e-10)
            none_gptq = next((v["kld_mean"] for m, v in arms if m == "none+gptq"), 1e-10)
            none_gptaq = next((v["kld_mean"] for m, v in arms if m == "none+gptaq"), 1e-10)
            nf = arms[0][1]["noise_floor"]

            print(f"  {'Method':<24} {'KLD':>12} {'vs none+none':>13} {'vs none+gptq':>13} {'vs none+gptaq':>14} "
                  f"{'Wt MSE':>12} {'HWE':>14} {'Bytes':>8} {'NF':>12}")
            print(f"  {'-'*24} {'-'*12} {'-'*13} {'-'*13} {'-'*14} {'-'*12} {'-'*14} {'-'*8} {'-'*12}")

            arms.sort(key=lambda x: x[1]["kld_mean"])
            for method, v in arms:
                imp_nn = f"{(1 - v['kld_mean'] / none_none) * 100:+.1f}%" if none_none > 0 else ""
                imp_gq = f"{(1 - v['kld_mean'] / none_gptq) * 100:+.1f}%" if none_gptq > 0 else ""
                imp_ga = f"{(1 - v['kld_mean'] / none_gptaq) * 100:+.1f}%" if none_gptaq > 0 else ""
                print(f"  {method:<24} {v['kld_mean']:>12.4e} {imp_nn:>13} {imp_gq:>13} {imp_ga:>14} "
                      f"{v['wmse_mean']:>12.4e} {v['hwe_mean']:>14.4e} {v['total_bytes']:>8.0f} {nf:>12.4e}")

    # Best per tensor/bits
    print(f"\n{'='*130}")
    print(f"  BEST STRATEGY PER TENSOR/BITS")
    print(f"{'='*130}")
    print(f"  {'Tensor':<16} {'K':>3} {'Best':<24} {'KLD':>12} {'vs none+none':>13} {'vs none+gptaq':>14}")
    print(f"  {'-'*16} {'-'*3} {'-'*24} {'-'*12} {'-'*13} {'-'*14}")

    for tensor in tensors:
        for bits in cfg.bits_list:
            best_name, best_kld = None, 1e10
            for k, v in summary.items():
                if f"{tensor}|" in k and f"|K{bits}" in k:
                    if v["kld_mean"] < best_kld:
                        best_kld = v["kld_mean"]
                        best_name = k.split("|")[1]
            nn = summary.get(f"{tensor}|none+none|K{bits}", {}).get("kld_mean", 1e-10)
            ga = summary.get(f"{tensor}|none+gptaq|K{bits}", {}).get("kld_mean", 1e-10)
            imp_nn = f"{(1 - best_kld / nn) * 100:+.1f}%" if nn > 0 else ""
            imp_ga = f"{(1 - best_kld / ga) * 100:+.1f}%" if ga > 0 else ""
            print(f"  {tensor:<16} K{bits:<2} {best_name:<24} {best_kld:>12.4e} {imp_nn:>13} {imp_ga:>14}")

    # Scaling impact averaged across tensors, per correction type
    print(f"\n{'='*130}")
    print(f"  SCALING IMPACT: averaged across all tensors, by correction type")
    print(f"{'='*130}")
    base_strats = [s[0] for s in SCALING_STRATEGIES]

    for bits in cfg.bits_list:
        print(f"\n  --- K{bits} ---")
        print(f"  {'Strategy':<24} {'none KLD':>14} {'gptq KLD':>14} {'gptaq KLD':>14} "
              f"{'gptq gain':>10} {'gptaq gain':>11} {'vs none+gptaq':>14}")
        print(f"  {'-'*24} {'-'*14} {'-'*14} {'-'*14} {'-'*10} {'-'*11} {'-'*14}")

        none_none_vals = []
        none_gptq_vals = []
        none_gptaq_vals = []
        for t in tensors:
            k1 = f"{t}|none+none|K{bits}"
            k2 = f"{t}|none+gptq|K{bits}"
            k3 = f"{t}|none+gptaq|K{bits}"
            if k1 in summary: none_none_vals.append(summary[k1]["kld_mean"])
            if k2 in summary: none_gptq_vals.append(summary[k2]["kld_mean"])
            if k3 in summary: none_gptaq_vals.append(summary[k3]["kld_mean"])

        for sname in base_strats:
            none_vals = []
            gptq_vals = []
            gptaq_vals = []
            for t in tensors:
                k1 = f"{t}|{sname}+none|K{bits}"
                k2 = f"{t}|{sname}+gptq|K{bits}"
                k3 = f"{t}|{sname}+gptaq|K{bits}"
                if k1 in summary: none_vals.append(summary[k1]["kld_mean"])
                if k2 in summary: gptq_vals.append(summary[k2]["kld_mean"])
                if k3 in summary: gptaq_vals.append(summary[k3]["kld_mean"])

            if not none_vals:
                continue

            avg_none = float(np.mean(none_vals))
            avg_gptq = float(np.mean(gptq_vals)) if gptq_vals else float('nan')
            avg_gptaq = float(np.mean(gptaq_vals)) if gptaq_vals else float('nan')
            gptq_gain = f"{(1 - avg_gptq / avg_none) * 100:+.1f}%" if avg_none > 0 and not np.isnan(avg_gptq) else ""
            gptaq_gain = f"{(1 - avg_gptaq / avg_none) * 100:+.1f}%" if avg_none > 0 and not np.isnan(avg_gptaq) else ""
            avg_nn = float(np.mean(none_none_vals)) if none_none_vals else 1e-10
            avg_ga = float(np.mean(none_gptaq_vals)) if none_gptaq_vals else 1e-10
            imp_ga = f"{(1 - avg_gptaq / avg_ga) * 100:+.1f}%" if avg_ga > 0 and not np.isnan(avg_gptaq) else ""
            print(f"  {sname:<24} {avg_none:>14.4e} {avg_gptq:>14.4e} {avg_gptaq:>14.4e} "
                  f"{gptq_gain:>10} {gptaq_gain:>11} {imp_ga:>14}")

# ==================== Main ====================

def main():
    cfg = ScalingConfig()
    print("=" * 130)
    print("R8-Scaling v3: matched per-tile quantizer, U^T@U=H^{-1} Cholesky, cached grids, cached w_pre, relative damping")
    print("=" * 130)
    print(f"  {cfg.m}x{cfg.n} matrices, {cfg.k} cal, tile={cfg.tile_size}")
    n_arms = len(SCALING_STRATEGIES) * len(CORRECTIONS) * len(cfg.bits_list) * (cfg.num_seeds + 4)
    print(f"  {len(SCALING_STRATEGIES)} strategies x {len(CORRECTIONS)} corrections x {len(cfg.bits_list)} K "
          f"x {cfg.num_seeds + 4} tensors = {n_arms} runs")

    # Validate inv_cholesky: U^T @ U = inv(H+λI)
    print("\n  Validating inv_cholesky...")
    H_test = np.random.default_rng(0).standard_normal((16, 16))
    H_test = H_test @ H_test.T + 0.01 * np.eye(16)
    L_test = inv_cholesky(H_test, 0.01)
    assert np.allclose(np.tril(L_test, -1), 0), "FAIL: L is not upper triangular"
    lam_test = 0.01 * np.mean(np.diag(H_test))
    assert np.allclose(L_test.T @ L_test, np.linalg.inv(H_test + lam_test * np.eye(16)), atol=1e-10), "FAIL: U^T @ U != inv(H+λI)"
    print("  inv_cholesky OK: upper triangular, U^T @ U = inv(H+λI)")

    # Sanity: none vs gptq vs gptaq must differ
    print("  Validating GPTQ/GPTAQ...")
    rng_test = np.random.default_rng(99)
    Wt = rng_test.standard_normal((128, 128)) * 0.05
    Xtt = rng_test.standard_normal((128, 512)) * 0.5
    Xtt[rng_test.random(128) < 0.05] *= 5.0
    Xt_test = Xtt + rng_test.standard_normal((128, 512)) * 0.02
    Wq_n = gptq_quantize(Wt, Xt_test, Xtt, 5, 16, 16, 0.01, use_p=False)
    Wq_g = gptq_quantize(Wt, Xt_test, Xtt, 5, 16, 16, 0.01, use_p=True)
    assert np.max(np.abs(Wq_n - Wq_g)) > 1e-6, "FAIL: GPTQ vs GPTAQ identical (P=0?)"
    print(f"  GPTQ vs GPTAQ differ: max|diff| = {np.max(np.abs(Wq_n - Wq_g)):.4e}")
    # Check GPTQ actually helps (not hurts) on unscaled synthetic
    Wq_raw = trellis_quantize(Wt, 5, 16)
    kld_raw = kld_loss(Wt @ Xtt, Wq_raw @ Xt_test)
    kld_gptq = kld_loss(Wt @ Xtt, Wq_n @ Xt_test)
    print(f"  Raw KLD={kld_raw:.4e}, GPTQ KLD={kld_gptq:.4e}, improvement={1-kld_gptq/kld_raw:+.1%}")

    print()
    t0 = time.time()
    results = run_full(cfg)
    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s ({len(results)} results, 0 errors)")

    summary = analyze_results(results, cfg)
    print_tables(summary, cfg)

    output = {
        "config": {
            "m": cfg.m, "n": cfg.n, "k": cfg.k, "tile_size": cfg.tile_size,
            "block_size": cfg.block_size, "damping": cfg.damping,
            "bits_list": list(cfg.bits_list), "num_seeds": cfg.num_seeds,
            "alpha_default": cfg.alpha_default,
            "outlier_pct": cfg.outlier_pct, "outlier_boost": cfg.outlier_boost,
            "cholesky_convention": "U^T@U = inv(H+λI), U = chol(inv(H+λI)).T",
        },
        "strategies": [s[0] for s in SCALING_STRATEGIES],
        "corrections": CORRECTIONS,
        "summary": summary,
        "raw": results,
    }
    outpath = "/Users/mbelleau/Projects/qwen38-research-r8-scaling/receipts/research/r8-scaling-results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")

if __name__ == "__main__":
    main()
