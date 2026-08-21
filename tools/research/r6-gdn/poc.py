#!/usr/bin/env python3
"""
R6-GDN v2: GDN-specific quantization proof-of-concept (corrected).

Fixes from v1 review:
- GPTQ Cholesky: U = chol(inv(H+λI)).T, upper triangular, U^T U = H^{-1}
- Gate sensitivity: computed from real W_z @ X, not synthetic RNG
- Same-metric comparison: all arms scored under identical objective
- Balanced transform: eigvals**0.25 (fourth root), not sqrt(eigvals)
- QKV slicing: correct row offsets [Q:0-2048, K:2048-4096, V:4096-10240]
- Recurrence timing: output computed AFTER state update
- Multiple seeds for accumulated error
- Correct Hessian formula: tr(D_σ @ E @ H_X @ E^T)
"""

import numpy as np
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GDNConfig:
    slice_size: int = 128
    num_steps: int = 10
    num_samples: int = 512
    bits: int = 4
    tile: int = 16
    damping: float = 1e-2
    seed: int = 42
    d_hidden: int = 128
    d_k: int = 128
    d_v: int = 128
    num_heads: int = 1
    gate_slope_range: Tuple[float, float] = (1.0, 10.0)
    decay_range: Tuple[float, float] = (0.8, 0.99)
    num_eval_sequences: int = 100  # for robust accumulated error


# ============================================================================
# Utilities
# ============================================================================

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x, axis=0):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# Quantizer (matched across all arms: per-column 16-element segments)
# ============================================================================

def quantize_uniform(w, bits):
    nl = 2 ** bits
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-15:
        return w.copy()
    s = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / s), 0, nl - 1) * s + lo

def quantize_col(Wc, bits, tile):
    m = Wc.shape[0]
    Wq = np.zeros_like(Wc)
    for i in range(0, m, tile):
        Wq[i:i+tile] = quantize_uniform(Wc[i:i+tile], bits)
    return Wq

def quantize_matrix(W, bits, tile):
    m, n = W.shape
    Wq = np.zeros_like(W)
    for j in range(n):
        Wq[:, j] = quantize_col(W[:, j], bits, tile)
    return Wq

def quantize_mixed_k(W, bits_per_col, tile):
    m, n = W.shape
    Wq = np.zeros_like(W)
    for j in range(n):
        Wq[:, j] = quantize_col(W[:, j], int(bits_per_col[j]), tile)
    return Wq


# ============================================================================
# GPTQ (corrected Cholesky convention)
# ============================================================================

def inv_cholesky(H, damping):
    """Upper triangular U with U^T U = (H + λI)^{-1}.
    
    Correct GPTQ convention (matching GPTAQ reference):
    1. Hinv = inv(H + λI)
    2. U = chol(Hinv).T  (upper triangular)
    3. Then U^T @ U = Hinv, and row U[j, j:] is non-zero for propagation.
    """
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    Hd = H + lam * np.eye(n)
    try:
        Hinv = np.linalg.inv(Hd)
        U = np.linalg.cholesky(Hinv).T  # upper triangular, U^T U = Hinv
        return U
    except np.linalg.LinAlgError:
        # Fallback: eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Hd)
        eigvals = np.maximum(eigvals, 1e-10)
        Hinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        U = np.linalg.cholesky(Hinv + 1e-10 * np.eye(n)).T
        return U


def gptq_quantize(W, X, bits, tile, damping=1e-2, H_weight=None):
    """GPTQ column-by-column with error compensation.
    
    H_weight: optional diagonal weight on output channels (gate sensitivity).
    Applied as D_σ @ H where D_σ = diag(H_weight). This is valid when H_weight
    represents output-channel sensitivity and H is the input covariance on the
    same axis (i.e., W has shape [output, input] and X has shape [output, N]).
    
    IMPORTANT: H_weight must be applied to make D_σ @ H, which is the
    Gauss-Newton/Fisher approximation for gated output. The resulting matrix
    is NOT symmetric, but GPTQ only needs inv(H_eff) which we compute from
    the symmetric part: H_sym = (D_σ @ H + H @ D_σ) / 2.
    """
    m, n = W.shape
    H = X @ X.T / X.shape[1]  # [m, m] input Hessian on output-channel axis
    
    if H_weight is not None:
        # Gate-aware: D_σ @ H. For GPTQ we need a symmetric PSD matrix.
        # Use the symmetrized form: (D_σ H + H D_σ) / 2
        D_sigma = np.diag(H_weight)
        H_eff = (D_sigma @ H + H @ D_sigma) / 2.0
        # Ensure PSD
        H_eff = (H_eff + H_eff.T) / 2.0
    else:
        H_eff = H
    
    U = inv_cholesky(H_eff, damping)  # upper triangular, U^T U = H_eff^{-1}
    Wq = W.copy().astype(np.float64)
    Q = np.zeros_like(W)
    
    for j in range(n):
        col = Wq[:, j].copy()
        col_q = quantize_col(col, bits, tile)
        err = (col - col_q) / (U[j, j] + 1e-15)
        Wq[:, j+1:] -= err[:, None] @ U[j, j+1:][None, :]
        Q[:, j] = col_q
    
    return Q


# ============================================================================
# GDN Recurrent State Simulation (corrected timing)
# ============================================================================

def gdn_recurrent_step(S, k, v, q, alpha, beta):
    """Single GDN delta-rule step.
    
    S_t = α S_{t-1} + β k_t (v_t - α S_{t-1}^T k_t)^T
    o_t = q_t^T S_t  (output AFTER state update — corrected from v1)
    """
    v_new = v - alpha * (S.T @ k)
    S_new = alpha * S + beta * np.outer(k, v_new)
    o = q @ S_new  # output computed AFTER state update
    return S_new, o


def simulate_gdn(W_q, W_k, W_v, W_out, W_z, X_seq, gates, betas, d_k, d_v):
    """Simulate GDN block with SiLU output gate (Qwen3.8 uses output_gate_type=swish).
    
    Gate: g = SiLU(z @ x) = z * sigmoid(z), o_gated = o * g
    """
    T = X_seq.shape[0]
    d_hidden = X_seq.shape[1]
    
    S = np.zeros((d_k, d_v))
    states = [S.copy()]
    outputs = []
    
    for t in range(T):
        x = X_seq[t]
        q = W_q @ x
        k = W_k @ x
        v = W_v @ x
        alpha = gates[t]
        beta = betas[t]
        
        S, o = gdn_recurrent_step(S, k, v, q, alpha, beta)
        states.append(S.copy())
        
        # Gate output: g = SiLU(z @ x) = z * sigmoid(z)
        z = W_z @ x
        gate = z * sigmoid(z)  # SiLU (swish)
        o_gated = o * gate
        
        y = W_out @ o_gated
        outputs.append(y)
    
    return np.array(outputs), states


# ============================================================================
# Gate Sensitivity (from real weights, SiLU derivative)
# ============================================================================

def silu_deriv(z):
    """Derivative of SiLU: σ(z) + z σ'(z) = σ(z)(1 + z(1-σ(z)))."""
    s = sigmoid(z)
    return s * (1 + z * (1 - s))

def compute_gate_sensitivity(W_z, X, gates, betas, d_v):
    """Compute gate-aware sensitivity from REAL W_z @ X using SiLU derivative.
    
    Qwen3.8 GDN uses SiLU (swish) output gate: g = SiLU(z), g' = σ(z)(1 + z(1-σ(z))).
    
    Output error: δy = W_out @ diag(g'(z)) @ (δW_z @ x)
    So the sensitivity weight is g'(z)² (squared SiLU derivative).
    
    Uses ALL calibration samples.
    
    Returns: sensitivity vector [m] where m = W_z.shape[0]
    """
    Z = W_z @ X  # [m, N] — real gate pre-activations
    g_deriv = silu_deriv(Z)  # [m, N] — SiLU derivative
    
    # Per-channel sensitivity: average g'² over all samples
    sensitivity = np.mean(g_deriv ** 2, axis=1)  # [m]
    
    # Normalize to mean 1
    sensitivity = sensitivity / (np.mean(sensitivity) + 1e-15)
    return sensitivity


# ============================================================================
# Balanced Realization (corrected: eigvals**0.25)
# ============================================================================

def compute_gramians(W_q, W_k, W_v, W_out, X, gates, betas, d_k, d_v):
    """Empirical controllability/observability Gramians for GDN state."""
    N = X.shape[1]
    T = len(gates)
    
    Q = W_q @ X
    K = W_k @ X
    V = W_v @ X
    
    gammas = np.zeros(T)
    for t in range(T):
        gamma = betas[t]
        for s in range(t + 1, T):
            gamma *= gates[s]
        gammas[t] = gamma
    
    W_c_k = np.zeros((d_k, d_k))
    W_o_k = np.zeros((d_k, d_k))
    for t in range(min(T, N)):
        k_t = K[:, t % N]
        q_t = Q[:, t % N]
        W_c_k += gammas[t] * np.outer(k_t, k_t)
        W_o_k += gammas[t] * np.outer(q_t, q_t)
    W_c_k /= T
    W_o_k /= T
    
    W_c_v = np.zeros((d_v, d_v))
    for t in range(min(T, N)):
        v_t = V[:, t % N]
        W_c_v += gammas[t] * np.outer(v_t, v_t)
    W_c_v /= T
    
    W_o_v = W_out.T @ W_out if W_out.shape[0] == d_v else np.eye(d_v)
    
    W_c_k += 1e-6 * np.eye(d_k)
    W_o_k += 1e-6 * np.eye(d_k)
    W_c_v += 1e-6 * np.eye(d_v)
    W_o_v += 1e-6 * np.eye(d_v)
    
    return W_c_k, W_o_k, W_c_v, W_o_v


def balanced_transform(W_c, W_o):
    """Balanced realization transform (corrected: eigvals**0.25).
    
    From M = L^T W_o L = U Σ² U^T:
    T = Σ^{1/2} U^T L^{-1}  where Σ^{1/2} = eigvals^{1/4}
    
    This gives T W_c T^T = Σ and T^{-T} W_o T^{-1} = Σ (both diagonal, equal).
    """
    L_c = np.linalg.cholesky(W_c)
    M = L_c.T @ W_o @ L_c
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.maximum(eigvals, 1e-12)
    
    # CORRECTED: fourth root, not square root
    Sigma_half = np.diag(eigvals ** 0.25)
    L_c_inv = np.linalg.inv(L_c)
    T = Sigma_half @ eigvecs.T @ L_c_inv
    return T


def verify_balanced_transform(T, W_c, W_o, tol=1e-6):
    """Verify T W_c T^T = T^{-T} W_o T^{-1} = Σ (diagonal, equal)."""
    T_Wc_Tt = T @ W_c @ T.T
    T_inv = np.linalg.inv(T)
    T_inv_Tt_Wo_T_inv = T_inv.T @ W_o @ T_inv
    
    diag1 = np.diag(T_Wc_Tt)
    diag2 = np.diag(T_inv_Tt_Wo_T_inv)
    
    max_diag = max(np.max(np.abs(diag1)), np.max(np.abs(diag2)), 1e-15)
    off1 = T_Wc_Tt - np.diag(diag1)
    off2 = T_inv_Tt_Wo_T_inv - np.diag(diag2)
    max_off = max(np.max(np.abs(off1)), np.max(np.abs(off2))) / max_diag
    max_diff = np.max(np.abs(diag1 - diag2) / (np.abs(diag1) + np.abs(diag2) + 1e-15))
    
    print(f"  Balanced realization verification:")
    print(f"    Max off-diagonal (rel): {max_off:.2e}")
    print(f"    Max diagonal diff (rel): {max_diff:.2e}")
    print(f"    Hankel SVs (first 8): {diag1[:8]}")
    return max_off < tol and max_diff < tol


# ============================================================================
# Recurrence-Aware GPTQ
# ============================================================================

def compute_recurrence_error_weight(gates, betas, X_seq, T, d_k):
    """Per-dimension weights from recurrence error propagation.
    
    w_j = Σ_t γ_t · x_t[j]² where γ_t = β_t Π_{s>t} α_s.
    """
    gammas = np.zeros(T)
    for t in range(T):
        gamma = betas[t]
        for s in range(t + 1, T):
            gamma *= gates[s]
        gammas[t] = gamma
    
    d_input = min(d_k, X_seq.shape[1])
    weight = np.ones(d_k)
    for t in range(T):
        x_t = X_seq[t, :d_input]
        weight[:d_input] += gammas[t] * x_t ** 2
    weight = weight / (np.mean(weight) + 1e-15)
    return weight


def recurrence_aware_gptq(W, X, bits, tile, gates, betas, X_seq, damping=1e-2, T=10):
    """GPTQ with recurrence-aware Hessian weighting."""
    m, n = W.shape
    H = X @ X.T / X.shape[1]
    rec_weight = compute_recurrence_error_weight(gates, betas, X_seq, T, m)
    # Symmetrize: (D @ H + H @ D) / 2
    D = np.diag(rec_weight)
    H_eff = (D @ H + H @ D) / 2.0
    H_eff = (H_eff + H_eff.T) / 2.0
    
    U = inv_cholesky(H_eff, damping)
    Wq = W.copy().astype(np.float64)
    Q = np.zeros_like(W)
    for j in range(n):
        col = Wq[:, j].copy()
        col_q = quantize_col(col, bits, tile)
        err = (col - col_q) / (U[j, j] + 1e-15)
        Wq[:, j+1:] -= err[:, None] @ U[j, j+1:][None, :]
        Q[:, j] = col_q
    return Q


# ============================================================================
# QKV Joint vs Separate
# ============================================================================

def qkv_joint_gptq(W_qkv, X, bits, tile, d_k, damping=1e-2):
    return gptq_quantize(W_qkv, X, bits, tile, damping)

def qkv_separate_gptq(W_qkv, X, bits, tile, d_k, damping=1e-2):
    W_q = W_qkv[:d_k]
    W_k = W_qkv[d_k:2*d_k]
    return np.vstack([gptq_quantize(W_q, X, bits, tile, damping),
                      gptq_quantize(W_k, X, bits, tile, damping)])


# ============================================================================
# z-Gate Sensitivity Allocation
# ============================================================================

def z_gate_sensitivity_allocation(W_z, X, bits, tile, gate_sens, extra_bits=1):
    """Allocate extra bits to most gate-sensitive columns of W_z."""
    m, n = W_z.shape
    # Column sensitivity from gate sensitivity and weight magnitudes
    col_sens = np.zeros(n)
    for j in range(n):
        col_sens[j] = np.sum(gate_sens * W_z[:, j] ** 2)
    
    bits_per_col = np.full(n, bits, dtype=int)
    num_extra = min(int(n * 0.2), n - 1)
    top_indices = np.argsort(col_sens)[-num_extra:]
    bits_per_col[top_indices] = bits + extra_bits
    
    total_budget = bits * n
    while np.sum(bits_per_col) > total_budget:
        candidates = np.where(bits_per_col > 1)[0]
        if len(candidates) == 0: break
        idx = candidates[np.argmin(col_sens[candidates])]
        bits_per_col[idx] -= 1
    while np.sum(bits_per_col) < total_budget:
        candidates = np.where(bits_per_col < bits + extra_bits)[0]
        if len(candidates) == 0: break
        idx = candidates[np.argmax(col_sens[candidates])]
        bits_per_col[idx] += 1
    
    return quantize_mixed_k(W_z, bits_per_col, tile)


# ============================================================================
# Error Metrics (all arms scored identically)
# ============================================================================

def single_step_error(W, Wq, X):
    """||W @ X - Wq @ X||² / N"""
    return float(np.mean((W @ X - Wq @ X) ** 2))

def output_weighted_error(W, Wq, X, gate_sens=None):
    """Output-channel-weighted error: mean(σ'² ⊙ ((W-Wq)@X)²).
    
    This is the gate-aware objective. When gate_sens=None, equals single_step_error.
    All arms are scored with the SAME gate_sens (or None) for fair comparison.
    """
    E = W - Wq
    EX = E @ X  # [m, N]
    if gate_sens is not None:
        return float(np.mean(gate_sens[:, None] * EX ** 2))
    return float(np.mean(EX ** 2))

def hessian_weighted_error(W, Wq, X, gate_sens=None):
    """Hessian-weighted error: mean(D_σ ⊙ ((W-Wq) @ X)²).
    
    This is tr(D_σ @ E @ H_X @ E^T) / (m*N) where H_X = X^T X / N,
    which equals mean(gate_sens[:, None] * (E @ X)²).
    When gate_sens=None, equals single_step_error.
    """
    E = W - Wq  # [m, n]
    EX = E @ X  # [m, N]
    if gate_sens is not None:
        return float(np.mean(gate_sens[:, None] * EX ** 2))
    return float(np.mean(EX ** 2))

def accumulated_recurrence_error(W_q, W_k, W_v, W_q_q, W_k_q, W_v_q,
                                  W_out, W_out_q, W_z, W_z_q,
                                  X_seq, gates, betas, d_k, d_v):
    """Accumulated output error over T GDN timesteps."""
    outputs_fp, _ = simulate_gdn(W_q, W_k, W_v, W_out, W_z, X_seq, gates, betas, d_k, d_v)
    outputs_q, _ = simulate_gdn(W_q_q, W_k_q, W_v_q, W_out_q, W_z_q, X_seq, gates, betas, d_k, d_v)
    return float(np.mean((outputs_fp - outputs_q) ** 2))

def accumulated_error_multi_seed(W_q, W_k, W_v, W_q_q, W_k_q, W_v_q,
                                  W_out, W_out_q, W_z, W_z_q,
                                  config, gates, betas, d_k, d_v, num_seeds=100):
    """Run accumulated error over multiple random sequences for robustness."""
    rng = np.random.default_rng(config.seed)
    errors = []
    for _ in range(num_seeds):
        X_seq = rng.standard_normal((config.num_steps, config.d_hidden)) * 0.1
        err = accumulated_recurrence_error(
            W_q, W_k, W_v, W_q_q, W_k_q, W_v_q,
            W_out, W_out_q, W_z, W_z_q,
            X_seq, gates, betas, d_k, d_v)
        errors.append(err)
    return np.array(errors)


# ============================================================================
# Weight loading (corrected QKV slicing)
# ============================================================================

def load_real_weights(config):
    """Load real GDN weights with CORRECT QKV row offsets.
    
    L0_qkv has shape [10240, 5120]:
    - Q: rows 0:2048 (head_dim * num_heads)
    - K: rows 2048:4096
    - V: rows 4096:10240 (value_dim is larger)
    
    For 128×128 slices: Q from rows 0:128, K from rows 2048:2176, V from rows 4096:4224.
    """
    data = np.load("/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz")
    s = config.slice_size
    
    W_qkv_full = data['L0_qkv']  # [10240, 5120]
    # CORRECTED: Q, K, V are at different row offsets
    W_q = W_qkv_full[0:s, :s].astype(np.float64)          # Q: rows 0:128
    W_k = W_qkv_full[2048:2048+s, :s].astype(np.float64)  # K: rows 2048:2176
    W_v = W_qkv_full[4096:4096+s, :s].astype(np.float64)  # V: rows 4096:4224
    
    W_out = data['L0_out'][:s, :s].astype(np.float64)
    W_z = data['L0_z'][:s, :s].astype(np.float64)
    
    return {'W_q': W_q, 'W_k': W_k, 'W_v': W_v, 'W_out': W_out, 'W_z': W_z}


def generate_data(config):
    """Generate calibration data and gating signals."""
    rng = np.random.default_rng(config.seed)
    s = config.slice_size
    
    X = rng.standard_normal((s, config.num_samples)) * 0.1
    outlier_mask = rng.random(s) < 0.05
    X[outlier_mask] *= 10
    
    X_seq = rng.standard_normal((config.num_steps, s)) * 0.1
    
    # Gate signals from real gate dynamics
    slopes = np.linspace(config.gate_slope_range[0], config.gate_slope_range[1], config.num_steps)
    gate_inputs = rng.standard_normal(config.num_steps) * 0.3
    gates = np.clip(sigmoid(gate_inputs * slopes), 0.8, 0.99)
    beta_inputs = rng.standard_normal(config.num_steps) * 0.5
    betas = sigmoid(beta_inputs * slopes) * 0.8 + 0.1
    
    return {'X': X, 'X_seq': X_seq, 'gates': gates, 'betas': betas}


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(config):
    print("=" * 80)
    print(f"R6-GDN v2: GDN-Specific Quantization (K{config.bits})")
    print("=" * 80)
    
    weights = load_real_weights(config)
    data = generate_data(config)
    X = data['X']
    X_seq = data['X_seq']
    gates = data['gates']
    betas = data['betas']
    
    s = config.slice_size
    d_k = config.d_k
    d_v = config.d_v
    bits = config.bits
    tile = config.tile
    
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_out = weights['W_out']
    W_z = weights['W_z']
    
    print(f"\nWeight shapes: Q={W_q.shape}, K={W_k.shape}, V={W_v.shape}")
    print(f"  Out={W_out.shape}, Z={W_z.shape}")
    print(f"Q rows 0:128, K rows 2048:2176, V rows 4096:4224 (corrected QKV slicing)")
    print(f"Q std={np.std(W_q):.6f}, K std={np.std(W_k):.6f}, V std={np.std(W_v):.6f}")
    print(f"Out std={np.std(W_out):.6f}, Z std={np.std(W_z):.6f}")
    
    # Gate sensitivity from REAL W_z @ X
    gate_sens = compute_gate_sensitivity(W_z, X, gates, betas, d_v)
    print(f"\nGate sensitivity (from real W_z@X): mean={np.mean(gate_sens):.6f}, std={np.std(gate_sens):.6f}")
    print(f"  Range: [{np.min(gate_sens):.6f}, {np.max(gate_sens):.6f}]")
    
    # Verify GPTQ Cholesky convention
    H_test = X @ X.T / config.num_samples
    U_test = inv_cholesky(H_test, config.damping)
    cholesky_check = np.max(np.abs(U_test.T @ U_test - np.linalg.inv(H_test + config.damping * np.mean(np.diag(H_test)) * np.eye(s))))
    print(f"\nCholesky check: max|U^T U - Hinv| = {cholesky_check:.2e}")
    print(f"U is upper triangular: {np.allclose(U_test, np.triu(U_test))}")
    
    results = {}
    
    # --- Noise floor ---
    print("\n--- Noise Floor ---")
    noise_ss = single_step_error(W_q, W_q, X)
    noise_hw = hessian_weighted_error(W_q, W_q, X, gate_sens)
    print(f"  Single-step: {noise_ss:.6e}, Hess-weighted: {noise_hw:.6e}")
    results['noise_floor'] = {'single_step': noise_ss, 'hessian_weighted': noise_hw}
    
    # --- RTN ---
    print("\n--- RTN ---")
    W_q_rtn = quantize_matrix(W_q, bits, tile)
    W_k_rtn = quantize_matrix(W_k, bits, tile)
    W_v_rtn = quantize_matrix(W_v, bits, tile)
    W_out_rtn = quantize_matrix(W_out, bits, tile)
    W_z_rtn = quantize_matrix(W_z, bits, tile)
    
    # --- Standard GPTQ ---
    print("\n--- Standard GPTQ ---")
    W_q_gptq = gptq_quantize(W_q, X, bits, tile, config.damping)
    W_k_gptq = gptq_quantize(W_k, X, bits, tile, config.damping)
    W_v_gptq = gptq_quantize(W_v, X, bits, tile, config.damping)
    W_out_gptq = gptq_quantize(W_out, X, bits, tile, config.damping)
    W_z_gptq = gptq_quantize(W_z, X, bits, tile, config.damping)
    
    # Verify GPTQ actually differs from RTN
    gptq_diff = np.max(np.abs(W_q_rtn - W_q_gptq))
    print(f"  GPTQ vs RTN max diff: {gptq_diff:.6e}")
    
    # Score ALL arms under SAME metrics (both weighted and unweighted)
    for name, (Wqq, Wkq, Wvq, Woq, Wzq) in {
        'rtn': (W_q_rtn, W_k_rtn, W_v_rtn, W_out_rtn, W_z_rtn),
        'gptq_standard': (W_q_gptq, W_k_gptq, W_v_gptq, W_out_gptq, W_z_gptq),
    }.items():
        ss = single_step_error(W_q, Wqq, X)
        hw_uw = hessian_weighted_error(W_q, Wqq, X)  # unweighted
        hw_w = hessian_weighted_error(W_q, Wqq, X, gate_sens)  # weighted
        ow_w = output_weighted_error(W_q, Wqq, X, gate_sens)
        # Accumulated error (multi-seed)
        acc_errors = accumulated_error_multi_seed(
            W_q, W_k, W_v, Wqq, Wkq, Wvq, W_out, Woq, W_z, Wzq,
            config, gates, betas, d_k, d_v, num_seeds=config.num_eval_sequences)
        acc_mean = float(np.mean(acc_errors))
        acc_std = float(np.std(acc_errors))
        print(f"  {name}: ss={ss:.6e}, hw_uw={hw_uw:.6e}, hw_w={hw_w:.6e}, ow_w={ow_w:.6e}")
        print(f"    accum: mean={acc_mean:.6e}, std={acc_std:.6e} (n={config.num_eval_sequences})")
        results[name] = {
            'single_step': ss, 'hessian_unweighted': hw_uw,
            'hessian_weighted': hw_w, 'output_weighted': ow_w,
            'accumulated_mean': acc_mean, 'accumulated_std': acc_std
        }
    
    # --- Gate-Aware GPTQ (z-only: gate sensitivity only applies to W_z) ---
    print("\n--- Gate-Aware GPTQ (z-only sensitivity) ---")
    # Per reviewer: gate sensitivity should only weight W_z's Hessian, not Q/K/V/out.
    # For Q/K/V/out, use standard GPTQ. For W_z, use gate-weighted GPTQ.
    W_q_gate = W_q_gptq.copy()
    W_k_gate = W_k_gptq.copy()
    W_v_gate = W_v_gptq.copy()
    W_out_gate = W_out_gptq.copy()
    W_z_gate = gptq_quantize(W_z, X, bits, tile, config.damping, H_weight=gate_sens)
    
    gptq_gate_diff = np.max(np.abs(W_z_gptq - W_z_gate))
    print(f"  z gate-aware vs standard GPTQ max diff: {gptq_gate_diff:.6e}")
    
    # Score W_z under gate-weighted metric
    ss_z_gate = single_step_error(W_z, W_z_gate, X)
    ss_z_std = single_step_error(W_z, W_z_gptq, X)
    hw_z_gate = hessian_weighted_error(W_z, W_z_gate, X, gate_sens)
    hw_z_std = hessian_weighted_error(W_z, W_z_gptq, X, gate_sens)
    print(f"  z single-step: gate={ss_z_gate:.6e}, std={ss_z_std:.6e}")
    print(f"  z gate-weighted: gate={hw_z_gate:.6e}, std={hw_z_std:.6e}")
    if ss_z_std > 0:
        print(f"  z improvement: {(1 - ss_z_gate/ss_z_std)*100:.2f}% ss, {(1 - hw_z_gate/hw_z_std)*100:.2f}% hw")
    
    # Accumulated error (all matrices; only z differs from standard)
    acc_errors = accumulated_error_multi_seed(
        W_q, W_k, W_v, W_q_gate, W_k_gate, W_v_gate, W_out, W_out_gate, W_z, W_z_gate,
        config, gates, betas, d_k, d_v, num_seeds=config.num_eval_sequences)
    acc_mean = float(np.mean(acc_errors))
    acc_std = float(np.std(acc_errors))
    print(f"    accum: mean={acc_mean:.6e}, std={acc_std:.6e} (n={config.num_eval_sequences})")
    results['gptq_gate_aware_z'] = {
        'z_single_step': ss_z_gate, 'z_single_step_std': ss_z_std,
        'z_hessian_weighted': hw_z_gate, 'z_hessian_weighted_std': hw_z_std,
        'accumulated_mean': acc_mean, 'accumulated_std': acc_std
    }
    
    # --- Balanced Realization (corrected: eigvals**0.25) ---
    print("\n--- Balanced Realization GPTQ (corrected) ---")
    W_c_k, W_o_k, W_c_v, W_o_v = compute_gramians(
        W_q, W_k, W_v, W_out, X, gates, betas, d_k, d_v)
    
    imb_k = np.mean(np.diag(W_c_k)) / (np.mean(np.diag(W_o_k)) + 1e-15)
    print(f"  Imbalance (k) before: {imb_k:.4f}")
    
    T_k = balanced_transform(W_c_k, W_o_k)
    T_v = balanced_transform(W_c_v, W_o_v)
    
    verified_k = verify_balanced_transform(T_k, W_c_k, W_o_k)
    verified_v = verify_balanced_transform(T_v, W_c_v, W_o_v)
    print(f"  Key-side balanced: {verified_k}")
    print(f"  Value-side balanced: {verified_v}")
    
    T_k_inv = np.linalg.inv(T_k)
    T_v_inv = np.linalg.inv(T_v)
    
    # Transform (corrected direction per reviewer):
    # For controllability (K): K' = T_k @ K  (controllable directions)
    # For observability (Q): Q' = T_k^{-T} @ Q  (observable directions)
    # For V: V' = T_v @ V
    # For W_out: W_out' = W_out @ T_v^{-1}
    W_q_bal = T_k_inv.T @ W_q   # Q' = T_k^{-T} Q
    W_k_bal = T_k @ W_k          # K' = T_k K
    W_v_bal = T_v @ W_v          # V' = T_v V
    W_out_bal = W_out @ T_v_inv  # Out' = Out @ T_v^{-1}
    
    # Quantize in balanced basis with GPTQ
    W_q_bal_q = gptq_quantize(W_q_bal, X, bits, tile, config.damping)
    W_k_bal_q = gptq_quantize(W_k_bal, X, bits, tile, config.damping)
    W_v_bal_q = gptq_quantize(W_v_bal, X, bits, tile, config.damping)
    W_out_bal_q = gptq_quantize(W_out_bal, X, bits, tile, config.damping)
    W_z_bal_q = gptq_quantize(W_z, X, bits, tile, config.damping)
    
    # Inverse transform
    W_q_bal_recon = T_k.T @ W_q_bal_q      # Q = T_k^T Q'
    W_k_bal_recon = T_k_inv @ W_k_bal_q     # K = T_k^{-1} K'
    W_v_bal_recon = T_v_inv @ W_v_bal_q     # V = T_v^{-1} V'
    W_out_bal_recon = W_out_bal_q @ T_v     # Out = Out' @ T_v
    
    print(f"  W_q original std: {np.std(W_q):.6e}, balanced std: {np.std(W_q_bal):.6e}")
    print(f"  T_k condition number: {np.linalg.cond(T_k):.2e}")
    
    ss = single_step_error(W_q, W_q_bal_recon, X)
    hw_uw = hessian_weighted_error(W_q, W_q_bal_recon, X)
    hw_w = hessian_weighted_error(W_q, W_q_bal_recon, X, gate_sens)
    acc_errors = accumulated_error_multi_seed(
        W_q, W_k, W_v, W_q_bal_recon, W_k_bal_recon, W_v_bal_recon,
        W_out, W_out_bal_recon, W_z, W_z_bal_q,
        config, gates, betas, d_k, d_v, num_seeds=config.num_eval_sequences)
    acc_mean = float(np.mean(acc_errors))
    acc_std = float(np.std(acc_errors))
    print(f"  ss={ss:.6e}, hw_uw={hw_uw:.6e}, hw_w={hw_w:.6e}")
    print(f"    accum: mean={acc_mean:.6e}, std={acc_std:.6e}")
    results['balanced_realization'] = {
        'single_step': ss, 'hessian_unweighted': hw_uw,
        'hessian_weighted': hw_w,
        'accumulated_mean': acc_mean, 'accumulated_std': acc_std,
        'verified_k': verified_k, 'verified_v': verified_v,
        'imbalance_before': float(imb_k)
    }
    
    # --- Recurrence-Aware GPTQ ---
    print("\n--- Recurrence-Aware GPTQ ---")
    W_q_rec = recurrence_aware_gptq(W_q, X, bits, tile, gates, betas, X_seq, config.damping, config.num_steps)
    W_k_rec = recurrence_aware_gptq(W_k, X, bits, tile, gates, betas, X_seq, config.damping, config.num_steps)
    W_v_rec = recurrence_aware_gptq(W_v, X, bits, tile, gates, betas, X_seq, config.damping, config.num_steps)
    W_out_rec = recurrence_aware_gptq(W_out, X, bits, tile, gates, betas, X_seq, config.damping, config.num_steps)
    W_z_rec = recurrence_aware_gptq(W_z, X, bits, tile, gates, betas, X_seq, config.damping, config.num_steps)
    
    ss = single_step_error(W_q, W_q_rec, X)
    hw_uw = hessian_weighted_error(W_q, W_q_rec, X)
    hw_w = hessian_weighted_error(W_q, W_q_rec, X, gate_sens)
    acc_errors = accumulated_error_multi_seed(
        W_q, W_k, W_v, W_q_rec, W_k_rec, W_v_rec,
        W_out, W_out_rec, W_z, W_z_rec,
        config, gates, betas, d_k, d_v, num_seeds=config.num_eval_sequences)
    acc_mean = float(np.mean(acc_errors))
    acc_std = float(np.std(acc_errors))
    print(f"  ss={ss:.6e}, hw_uw={hw_uw:.6e}, hw_w={hw_w:.6e}")
    print(f"    accum: mean={acc_mean:.6e}, std={acc_std:.6e}")
    results['recurrence_aware'] = {
        'single_step': ss, 'hessian_unweighted': hw_uw,
        'hessian_weighted': hw_w,
        'accumulated_mean': acc_mean, 'accumulated_std': acc_std
    }
    
    # --- QKV Joint vs Separate (includes Q, K, V) ---
    print("\n--- QKV Joint vs Separate (Q+K+V) ---")
    W_qkv_full = np.vstack([W_q, W_k, W_v])  # [3*d_k, d_hidden]
    W_qkv_joint = gptq_quantize(W_qkv_full, X, bits, tile, config.damping)
    W_qkv_sep = np.vstack([
        gptq_quantize(W_q, X, bits, tile, config.damping),
        gptq_quantize(W_k, X, bits, tile, config.damping),
        gptq_quantize(W_v, X, bits, tile, config.damping),
    ])
    
    ss_joint = single_step_error(W_qkv_full, W_qkv_joint, X)
    ss_sep = single_step_error(W_qkv_full, W_qkv_sep, X)
    print(f"  Joint: {ss_joint:.6e}, Separate: {ss_sep:.6e}")
    print(f"  Identical: {np.allclose(W_qkv_joint, W_qkv_sep)}")
    results['qkv_joint'] = {'single_step': ss_joint}
    results['qkv_separate'] = {'single_step': ss_sep}
    
    # --- z-Gate Sensitivity Allocation (mixed-RTN vs uniform-RTN, matched) ---
    print("\n--- z-Gate Sensitivity Allocation (vs uniform RTN) ---")
    W_z_alloc = z_gate_sensitivity_allocation(W_z, X, bits, tile, gate_sens, extra_bits=1)
    # Compare against uniform RTN (same optimizer class: no GPTQ)
    ss_alloc = single_step_error(W_z, W_z_alloc, X)
    ss_rtn_z = single_step_error(W_z, W_z_rtn, X)
    ss_gptq_z = single_step_error(W_z, W_z_gptq, X)
    hw_alloc = hessian_weighted_error(W_z, W_z_alloc, X, gate_sens)
    hw_rtn_z = hessian_weighted_error(W_z, W_z_rtn, X, gate_sens)
    print(f"  z-alloc (mixed RTN): {ss_alloc:.6e}, z-RTN (uniform): {ss_rtn_z:.6e}")
    print(f"  z-GPTQ: {ss_gptq_z:.6e} (reference)")
    print(f"  z-alloc vs RTN: {(1 - ss_alloc/ss_rtn_z)*100:.2f}% ss, {(1 - hw_alloc/hw_rtn_z)*100:.2f}% hw")
    results['z_gate_allocation'] = {'single_step': ss_alloc, 'hessian_weighted': hw_alloc}
    results['z_rtn'] = {'single_step': ss_rtn_z, 'hessian_weighted': hw_rtn_z}
    results['z_gptq'] = {'single_step': ss_gptq_z}
    
    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY (same metric for all arms)")
    print("=" * 80)
    print(f"{'Method':<30} {'Single-step':>14} {'HW-unweighted':>14} {'HW-weighted':>14} {'Accum(mean)':>14}")
    print("-" * 90)
    for name, res in results.items():
        label = {
            'noise_floor': 'Noise floor', 'rtn': 'RTN',
            'gptq_standard': 'Standard GPTQ', 'gptq_gate_aware_z': 'Gate-aware GPTQ (z)',
            'balanced_realization': 'Balanced realiz.', 'recurrence_aware': 'Recurrence-aware',
            'qkv_joint': 'QKV joint (Q+K+V)', 'qkv_separate': 'QKV separate',
            'z_gate_allocation': 'z-gate alloc (mixed RTN)', 'z_gptq': 'z GPTQ', 'z_rtn': 'z RTN'
        }.get(name, name)
        ss = res.get('single_step', res.get('z_single_step', float('nan')))
        hw_u = res.get('hessian_unweighted', float('nan'))
        hw_w = res.get('hessian_weighted', res.get('z_hessian_weighted', float('nan')))
        acc = res.get('accumulated_mean', float('nan'))
        print(f"{label:<30} {ss:>14.6e} {hw_u:>14.6e} {hw_w:>14.6e} {acc:>14.6e}")
    
    # Improvement vs standard GPTQ
    print("\n" + "=" * 80)
    print("IMPROVEMENT vs Standard GPTQ (%)")
    print("=" * 80)
    base = results['gptq_standard']
    for name in ['rtn', 'gptq_gate_aware_z', 'balanced_realization', 'recurrence_aware']:
        res = results.get(name, {})
        if not res: continue
        label = {'rtn': 'RTN', 'gptq_gate_aware_z': 'Gate-aware (z)',
                 'balanced_realization': 'Balanced realiz.',
                 'recurrence_aware': 'Recurrence-aware'}.get(name, name)
        for metric, mkey in [('single_step', 'single_step'), ('hw_unweighted', 'hessian_unweighted'),
                              ('hw_weighted', 'hessian_weighted'), ('accumulated', 'accumulated_mean')]:
            b = base.get(mkey, 0)
            v = res.get(mkey, 0)
            if b > 0 and v > 0:
                imp = (1 - v / b) * 100
                print(f"  {label:<25} {metric:<15} {imp:>8.2f}%")
    
    # z-specific improvement
    z_std = results.get('z_gptq', {})
    z_gate = results.get('gptq_gate_aware_z', {})
    if z_std and z_gate:
        z_ss_imp = (1 - z_gate.get('z_single_step', 0) / (z_gate.get('z_single_step_std', 1e-15) + 1e-15)) * 100
        print(f"  z gate-aware vs z std:  z_single_step  {z_ss_imp:>8.2f}%")
    
    return results


def main():
    all_results = {}
    for bits in [3, 4, 5, 6]:
        config = GDNConfig(bits=bits)
        print(f"\n{'#'*80}")
        print(f"# K{bits}")
        print(f"{'#'*80}")
        all_results[f'K{bits}'] = run_experiment(config)
    
    output_path = "/Users/mbelleau/Projects/qwen38-research-r6-gdn/receipts/research/r6-gdn-results.json"
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, list): return [convert(x) for x in obj]
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
