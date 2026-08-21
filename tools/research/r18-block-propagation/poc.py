#!/usr/bin/env python3
"""
R18-BlockPropagation: Block-level error propagation through MLP and attention.

All prior experiments measure per-tensor Hessian-weighted error. The real objective
is end-to-end model KLD. This PoC bridges the gap by propagating quantization error
through synthetic MLP and attention blocks.

Experiments:
1. MLP block error propagation: ||y_fp - y_quant||^2 vs sum of individual ||W - Wq||^2
2. Attention block error propagation (with GQA structure)
3. Error propagation with BiIP+Hadamard rotation: does the rotation benefit compound?
4. Error propagation with allocation: greedy allocation across all matrices in the block
5. Nonlinear error amplification: SiLU and softmax amplification factors
6. Cross-matrix error interaction: quantize one matrix at a time

Uses 128x128 slices of real Qwen3.8-27B L0/L55 MLP weights.
Attention uses synthetic weights (the L0_qkv/L0_out weights in the archive are GDN
projections, not standard softmax attention — see reviewer finding).

Reviewer fixes applied:
- GQA einsum: proper seq-to-seq attention (each token attends to all tokens)
- Attention weights: synthetic, statistics-matched, clearly labeled
- V/O rotation: correct orientation for (out,in) storage convention
- up_proj: distinct slice (not exact copy of gate)
- Amplification metric: labeled as batch-dependent, not intrinsic
- Nonlinearity metrics: renamed and clarified
- "DP" renamed to "greedy" (it's greedy marginal allocation, not dynamic programming)
"""

import numpy as np
import json
import time
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Configuration
# =============================================================================

WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
SLICE = 128          # matrix slice dimension
BATCH = 64           # synthetic input batch / sequence length
K_VALUES = [3, 4, 5, 6]
N_SEEDS = 5          # number of random input seeds for Monte Carlo
OUTLIER_FRACTION = 0.05
OUTLIER_SCALE = 10.0

# GQA: 3 Q heads per KV head, d_head = 128 (matches Qwen3.8-27B GDN q/k/v partition:
# Q=48*128=6144, K=16*128=2048, V=16*128=2048 in the real model. We use 3:1 ratio
# with synthetic weights for a proper softmax-attention block study.)
N_Q_PER_KV = 3
D_HEAD = SLICE  # 128

# =============================================================================
# Quantizers (per-column uniform, matching R3/R10)
# =============================================================================

def quantize_per_column(W, bits):
    """Per-column uniform quantization with 2^bits levels."""
    if bits == 0:
        return W.copy()
    m, n = W.shape
    levels = 2 ** bits
    Wq = np.zeros_like(W)
    for j in range(n):
        col = W[:, j]
        wmin, wmax = col.min(), col.max()
        rng_col = wmax - wmin
        if rng_col < 1e-12:
            Wq[:, j] = col
            continue
        scale = rng_col / (levels - 1)
        q = np.round((col - wmin) / scale)
        Wq[:, j] = np.clip(q, 0, levels - 1) * scale + wmin
    return Wq

def frob_sq(A):
    return float(np.sum(A ** 2))

# =============================================================================
# Nonlinearities
# =============================================================================

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -50, 50)))

def silu_deriv(x):
    x_c = np.clip(x, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-x_c))
    return sig + x_c * sig * (1.0 - sig)

def softmax(S):
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S)
    return A / A.sum(axis=-1, keepdims=True)

# =============================================================================
# Hadamard / BiIP rotation utilities (from R3)
# =============================================================================

def hadamard_matrix(n):
    assert (n & (n - 1)) == 0 and n > 0
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def signed_random_hadamard(n, rng):
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n)
    return H @ np.diag(signs)

def biip_scaling(W, H_X_diag, H_G_diag):
    """Two-sided diagonal balancing (KronQ Eq. 11)."""
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = np.clip((H_X_diag / col_norms_sq) ** 0.25, 0.1, 10.0)
    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = np.clip((H_G_diag / row_norms_sq) ** 0.25, 0.1, 10.0)
    S_G = np.diag(sg_diag)
    S_X = np.diag(sx_diag)
    W_t = S_G @ W @ S_X
    return S_G, S_X, W_t

# =============================================================================
# Synthetic input generation
# =============================================================================

def synthetic_input(d, batch, rng, outlier_fraction=OUTLIER_FRACTION, outlier_scale=OUTLIER_SCALE):
    """Gaussian input with outlier channels (realistic activation pattern)."""
    X = rng.standard_normal((batch, d))
    n_outliers = max(1, int(d * outlier_fraction))
    outlier_channels = rng.choice(d, n_outliers, replace=False)
    X[:, outlier_channels] *= outlier_scale
    return X

def input_hessian_diag(X):
    """Diagonal of H_X = X^T X / N."""
    return np.sum(X ** 2, axis=0) / X.shape[0]

def output_hessian_diag(W, X):
    """Diagonal of H_G proxy = Y^T Y / N where Y = X @ W^T."""
    Y = X @ W.T
    return np.sum(Y ** 2, axis=0) / Y.shape[0]

# =============================================================================
# MLP Block (128x128 slice)
# =============================================================================

class MLPBlockSlice:
    """MLP block with SiLU gating using 128x128 weight slices.

    Forward: y = (silu(x @ gate.T) * (x @ up.T)) @ down.T
    All matrices are [d, d] = [128, 128].
    Weight storage convention: W [d_out, d_in], forward uses X @ W.T.
    """
    def __init__(self, W_gate, W_up, W_down, X):
        self.W_gate = W_gate  # [d_inter, d_model]
        self.W_up = W_up      # [d_inter, d_model]
        self.W_down = W_down   # [d_model, d_inter]
        self.X = X             # [batch, d_model]
        self.Y_ref = self.forward(self.W_gate, self.W_up, self.W_down)
        self.gate_pre = X @ self.W_gate.T
        self.up_pre = X @ self.W_up.T
        self.h = silu(self.gate_pre) * self.up_pre

    def forward(self, W_gate, W_up, W_down, X=None):
        if X is None:
            X = self.X
        g = X @ W_gate.T
        u = X @ W_up.T
        h = silu(g) * u
        y = h @ W_down.T
        return y

    def block_error(self, W_gate, W_up, W_down, X=None):
        if X is None:
            X = self.X
        Y_hat = self.forward(W_gate, W_up, W_down, X)
        Y_ref = self.forward(self.W_gate, self.W_up, self.W_down, X)
        return frob_sq(Y_ref - Y_hat)

    def individual_errors(self, W_gate_q, W_up_q, W_down_q):
        return {
            'gate': frob_sq(self.W_gate - W_gate_q),
            'up': frob_sq(self.W_up - W_up_q),
            'down': frob_sq(self.W_down - W_down_q),
        }

# =============================================================================
# Attention Block — single-head and GQA
# =============================================================================

class AttentionBlockSlice:
    """Single-head softmax attention block using 128x128 weight slices.

    Forward: Q = x @ W_Q.T, K = x @ W_K.T, V = x @ W_V.T
    S = Q @ K^T / sqrt(d_head), A = softmax(S), y = (A @ V) @ W_O.T
    Weight storage: W [d_out, d_in], forward uses X @ W.T.
    Each of the BATCH input rows is a token in the sequence.
    """
    def __init__(self, W_Q, W_K, W_V, W_O, X, d_head=128):
        self.W_Q = W_Q  # [d_head, d_model]
        self.W_K = W_K
        self.W_V = W_V
        self.W_O = W_O  # [d_model, d_head]
        self.X = X
        self.d_head = d_head
        self.Y_ref = self.forward(self.W_Q, self.W_K, self.W_V, self.W_O)

    def forward(self, W_Q, W_K, W_V, W_O, X=None):
        if X is None:
            X = self.X
        Q = X @ W_Q.T  # [seq, d_head]
        K = X @ W_K.T  # [seq, d_head]
        V = X @ W_V.T  # [seq, d_head]
        S = Q @ K.T / np.sqrt(self.d_head)  # [seq, seq]
        A = softmax(S)
        attn = A @ V  # [seq, d_head]
        Y = attn @ W_O.T  # [seq, d_model]
        return Y

    def block_error(self, W_Q, W_K, W_V, W_O, X=None):
        if X is None:
            X = self.X
        Y_hat = self.forward(W_Q, W_K, W_V, W_O, X)
        Y_ref = self.forward(self.W_Q, self.W_K, self.W_V, self.W_O, X)
        return frob_sq(Y_ref - Y_hat)

    def individual_errors(self, W_Qq, W_Kq, W_Vq, W_Oq):
        return {
            'Q': frob_sq(self.W_Q - W_Qq),
            'K': frob_sq(self.W_K - W_Kq),
            'V': frob_sq(self.W_V - W_Vq),
            'O': frob_sq(self.W_O - W_Oq),
        }

class GQAAttentionBlockSlice:
    """GQA attention with N_Q_PER_KV Q heads, 1 KV head, d_head=128.

    Q: [N_Q_PER_KV * d_head, d_model] = [384, 128]
    K: [d_head, d_model] = [128, 128]
    V: [d_head, d_model] = [128, 128]
    O: [d_model, N_Q_PER_KV * d_head] = [128, 384]

    Each KV head serves N_Q_PER_KV Q heads. Each token in the sequence
    attends to ALL tokens (proper seq-to-seq attention).

    Weight storage: W [d_out, d_in], forward uses X @ W.T.
    """
    def __init__(self, W_Q, W_K, W_V, W_O, X, d_head=128, n_q_per_kv=3):
        self.W_Q = W_Q  # [n_q_per_kv * d_head, d_model]
        self.W_K = W_K  # [d_head, d_model]
        self.W_V = W_V  # [d_head, d_model]
        self.W_O = W_O  # [d_model, n_q_per_kv * d_head]
        self.X = X
        self.d_head = d_head
        self.n_q_per_kv = n_q_per_kv
        self.Y_ref = self.forward(self.W_Q, self.W_K, self.W_V, self.W_O)

    def forward(self, W_Q, W_K, W_V, W_O, X=None):
        if X is None:
            X = self.X
        seq = X.shape[0]
        Q = X @ W_Q.T  # [seq, n_q_per_kv * d_head]
        K = X @ W_K.T  # [seq, d_head]
        V = X @ W_V.T  # [seq, d_head]
        Q = Q.reshape(seq, self.n_q_per_kv, self.d_head)  # [seq, n_q, d_head]
        # Proper seq-to-seq attention: each token attends to all tokens
        # S[b,h,j] = Q[b,h,:] . K[j,:] / sqrt(d_head)
        S = np.einsum('bhd,jd->bhj', Q, K) / np.sqrt(self.d_head)  # [seq, n_q, seq]
        A = softmax(S)  # softmax over j (keys)
        # attn[b,h,d] = sum_j A[b,h,j] * V[j,d]
        attn = np.einsum('bhj,jd->bhd', A, V)  # [seq, n_q, d_head]
        attn_flat = attn.reshape(seq, self.n_q_per_kv * self.d_head)  # [seq, n_q*d_head]
        Y = attn_flat @ W_O.T  # [seq, d_model]
        return Y

    def block_error(self, W_Q, W_K, W_V, W_O, X=None):
        if X is None:
            X = self.X
        Y_hat = self.forward(W_Q, W_K, W_V, W_O, X)
        Y_ref = self.forward(self.W_Q, self.W_K, self.W_V, self.W_O, X)
        return frob_sq(Y_ref - Y_hat)

    def individual_errors(self, W_Qq, W_Kq, W_Vq, W_Oq):
        return {
            'Q': frob_sq(self.W_Q - W_Qq),
            'K': frob_sq(self.W_K - W_Kq),
            'V': frob_sq(self.W_V - W_Vq),
            'O': frob_sq(self.W_O - W_Oq),
        }

# =============================================================================
# Weight slicing
# =============================================================================

def load_weights():
    """Load real MLP weights and extract 128x128 slices.

    MLP weights (gate, down) are from real Qwen3.8-27B.
    up_proj uses a DISTINCT slice of gate_proj (columns 128:256) as a
    statistics-matched proxy — NOT an exact copy. Real up_proj weights
    are not in the archive.

    Attention weights are SYNTHETIC (statistics-matched to real weight
    magnitudes). The archive's L0_qkv/L0_out are GDN projections, not
    standard softmax attention.
    """
    data = np.load(WEIGHTS_PATH)
    weights = {}
    # MLP slices from L0 and L55
    for layer in ['L0', 'L55']:
        gate = data[f'{layer}_gate']  # [17408, 5120]
        down = data[f'{layer}_down']  # [5120, 17408]
        weights[f'{layer}_gate'] = gate[:SLICE, :SLICE].astype(np.float64)
        # up: distinct slice of gate (columns 128:256) — same stats, NOT identical
        up_slice = gate[:SLICE, SLICE:2*SLICE].astype(np.float64)
        # Ensure slice exists
        if up_slice.shape[1] < SLICE:
            up_slice = gate[SLICE:2*SLICE, :SLICE].astype(np.float64)
        weights[f'{layer}_up'] = up_slice
        weights[f'{layer}_down'] = down[:SLICE, :SLICE].astype(np.float64)

    # Synthetic attention weights — statistics-matched to real weight magnitudes
    # Real gate weights have std ~0.01-0.03. Use similar scale.
    rng = np.random.default_rng(12345)
    attn_scale = float(np.std(data['L0_gate'][:SLICE, :SLICE]))
    d_model = SLICE
    d_head = D_HEAD

    # Single-head attention: all [d_head, d_model] = [128, 128]
    weights['synth_WQ'] = rng.standard_normal((d_head, d_model)) * attn_scale
    weights['synth_WK'] = rng.standard_normal((d_head, d_model)) * attn_scale
    weights['synth_WV'] = rng.standard_normal((d_head, d_model)) * attn_scale
    weights['synth_WO'] = rng.standard_normal((d_model, d_head)) * attn_scale

    # GQA: Q [384, 128], K/V [128, 128], O [128, 384]
    gqa_q_rows = N_Q_PER_KV * D_HEAD
    weights['synth_GQA_Q'] = rng.standard_normal((gqa_q_rows, d_model)) * attn_scale
    weights['synth_GQA_K'] = rng.standard_normal((d_head, d_model)) * attn_scale
    weights['synth_GQA_V'] = rng.standard_normal((d_head, d_model)) * attn_scale
    weights['synth_GQA_O'] = rng.standard_normal((d_model, gqa_q_rows)) * attn_scale

    return weights

# =============================================================================
# Experiment 1: MLP block error propagation
# =============================================================================

def exp1_mlp_propagation(weights, rng):
    """Compare block error to sum of individual weight errors for MLP.

    NOTE: The amplification factor (block_error / sum_individual_weight_errors)
    is batch/dimension dependent — it compares output-space error (summed over
    tokens and features) with weight-space error (summed over parameters).
    It is NOT an intrinsic property. For an intrinsic additivity measure, see
    Exp 6 (all_quantized / sum_single_matrix_block_errors).
    """
    results = {}
    for layer in ['L0', 'L55']:
        layer_results = {}
        W_gate = weights[f'{layer}_gate']
        W_up = weights[f'{layer}_up']
        W_down = weights[f'{layer}_down']

        for K in K_VALUES:
            block_errs = []
            sum_indiv_errs = []
            indiv_errs_list = []
            for seed in range(N_SEEDS):
                srng = np.random.default_rng(seed * 100 + 42)
                X = synthetic_input(SLICE, BATCH, srng)
                mlp = MLPBlockSlice(W_gate, W_up, W_down, X)

                Wg_q = quantize_per_column(W_gate, K)
                Wu_q = quantize_per_column(W_up, K)
                Wd_q = quantize_per_column(W_down, K)

                be = mlp.block_error(Wg_q, Wu_q, Wd_q)
                ie = mlp.individual_errors(Wg_q, Wu_q, Wd_q)
                sie = sum(ie.values())

                block_errs.append(be)
                sum_indiv_errs.append(sie)
                indiv_errs_list.append(ie)

            block_err = np.mean(block_errs)
            sum_indiv = np.mean(sum_indiv_errs)
            avg_indiv = {}
            for key in indiv_errs_list[0]:
                avg_indiv[key] = np.mean([d[key] for d in indiv_errs_list])

            amp = block_err / (sum_indiv + 1e-30)

            layer_results[f'K{K}'] = {
                'block_error': float(block_err),
                'sum_individual_weight_errors': float(sum_indiv),
                'individual_weight_errors': {k: float(v) for k, v in avg_indiv.items()},
                'output_to_weight_error_ratio': float(amp),
                'output_to_weight_error_ratio_note': 'batch/dimension dependent, not intrinsic; see exp6 for additivity',
                'block_error_std': float(np.std(block_errs)),
                'batch_size': BATCH,
                'slice_dim': SLICE,
            }
        results[layer] = layer_results
    return results

# =============================================================================
# Experiment 2: Attention block error propagation
# =============================================================================

def exp2_attention_propagation(weights, rng):
    """Compare block error to sum of individual weight errors for attention.

    NOTE: Same batch/dimension dependency caveat as exp1.
    Attention weights are SYNTHETIC (statistics-matched).
    """
    results = {}

    for label, attn_cls, w_keys in [
        ('single_head_synth', AttentionBlockSlice,
         ['synth_WQ', 'synth_WK', 'synth_WV', 'synth_WO']),
        ('gqa_synth', GQAAttentionBlockSlice,
         ['synth_GQA_Q', 'synth_GQA_K', 'synth_GQA_V', 'synth_GQA_O']),
    ]:
        label_results = {}
        W_Q = weights[w_keys[0]]
        W_K = weights[w_keys[1]]
        W_V = weights[w_keys[2]]
        W_O = weights[w_keys[3]]

        for K in K_VALUES:
            block_errs = []
            sum_indiv_errs = []
            indiv_errs_list = []
            for seed in range(N_SEEDS):
                srng = np.random.default_rng(seed * 100 + 42)
                d_in = W_Q.shape[1]
                X = synthetic_input(d_in, BATCH, srng)

                if 'gqa' in label:
                    attn = attn_cls(W_Q, W_K, W_V, W_O, X, d_head=D_HEAD, n_q_per_kv=N_Q_PER_KV)
                else:
                    attn = attn_cls(W_Q, W_K, W_V, W_O, X, d_head=SLICE)

                WQ_q = quantize_per_column(W_Q, K)
                WK_q = quantize_per_column(W_K, K)
                WV_q = quantize_per_column(W_V, K)
                WO_q = quantize_per_column(W_O, K)

                be = attn.block_error(WQ_q, WK_q, WV_q, WO_q)
                ie = attn.individual_errors(WQ_q, WK_q, WV_q, WO_q)
                sie = sum(ie.values())

                block_errs.append(be)
                sum_indiv_errs.append(sie)
                indiv_errs_list.append(ie)

            block_err = np.mean(block_errs)
            sum_indiv = np.mean(sum_indiv_errs)
            avg_indiv = {}
            for key in indiv_errs_list[0]:
                avg_indiv[key] = np.mean([d[key] for d in indiv_errs_list])

            amp = block_err / (sum_indiv + 1e-30)

            label_results[f'K{K}'] = {
                'block_error': float(block_err),
                'sum_individual_weight_errors': float(sum_indiv),
                'individual_weight_errors': {k: float(v) for k, v in avg_indiv.items()},
                'output_to_weight_error_ratio': float(amp),
                'output_to_weight_error_ratio_note': 'batch/dimension dependent, not intrinsic',
                'block_error_std': float(np.std(block_errs)),
                'weight_source': 'synthetic',
            }
        results[label] = label_results
    return results

# =============================================================================
# Experiment 3: Error propagation with rotation
# =============================================================================

def exp3_rotation_propagation(weights, rng):
    """Apply rotation to each matrix, quantize, inverse transform.
    Does the rotation benefit compound through the block?

    Tests:
    - Hadamard-only (no BiIP): pure rotation, no scaling
    - BiIP+Hadamard (conservative): with diagonal balancing, tight clipping
    - For attention V/O: R10's shared R (free invariant) with CORRECT orientation

    V/O rotation for (out,in) storage where forward uses X @ W.T:
    - V operates on head dim: WV_rot = R.T @ W_V, recon: WV_hat = R @ WV_rot_q
    - O operates on head dim: WO_rot = W_O @ R, recon: WO_hat = WO_rot_q @ R.T
    - This preserves V @ O = (R @ R.T @ W_V) @ (W_O @ R @ R.T) = W_V @ W_O (approx)
    """
    results = {}

    HESSIAN_BATCH = 512

    def hadamard_only(W, rng):
        """Apply signed Hadamard on both sides."""
        d_out, d_in = W.shape
        U = signed_random_hadamard(d_out, rng) if (d_out & (d_out - 1)) == 0 and d_out > 0 else np.eye(d_out)
        V = signed_random_hadamard(d_in, rng) if (d_in & (d_in - 1)) == 0 and d_in > 0 else np.eye(d_in)
        W_rot = U @ W @ V
        return W_rot, U, V

    def inverse_hadamard_only(W_q, U, V):
        return U.T @ W_q @ V.T

    def biip_hadamard_conservative(W, H_X_diag, H_G_diag, rng):
        """BiIP with conservative clipping (0.5, 2.0) + Hadamard."""
        d_out, d_in = W.shape
        col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
        sx_diag = np.clip((H_X_diag / col_norms_sq) ** 0.25, 0.5, 2.0)
        row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
        sg_diag = np.clip((H_G_diag / row_norms_sq) ** 0.25, 0.5, 2.0)
        S_G = np.diag(sg_diag)
        S_X = np.diag(sx_diag)
        W_t = S_G @ W @ S_X
        U = signed_random_hadamard(d_out, rng) if (d_out & (d_out - 1)) == 0 and d_out > 0 else np.eye(d_out)
        V = signed_random_hadamard(d_in, rng) if (d_in & (d_in - 1)) == 0 and d_in > 0 else np.eye(d_in)
        W_rot = U @ W_t @ V
        return W_rot, S_G, S_X, U, V

    def inverse_biip_hadamard_conservative(W_q, S_G, S_X, U, V):
        S_G_inv = np.linalg.inv(S_G)
        S_X_inv = np.linalg.inv(S_X)
        return S_G_inv @ U.T @ W_q @ V.T @ S_X_inv

    # --- MLP rotation ---
    for layer in ['L0', 'L55']:
        layer_results = {}
        W_gate = weights[f'{layer}_gate']
        W_up = weights[f'{layer}_up']
        W_down = weights[f'{layer}_down']

        for K in K_VALUES:
            no_rot_errs = []
            hadamard_errs = []
            biip_had_errs = []
            no_rot_indiv = []
            hadamard_indiv = []
            biip_had_indiv = []

            for seed in range(N_SEEDS):
                srng = np.random.default_rng(seed * 100 + 42)
                X = synthetic_input(SLICE, BATCH, srng)
                mlp = MLPBlockSlice(W_gate, W_up, W_down, X)

                X_hess = synthetic_input(SLICE, HESSIAN_BATCH, srng)
                mlp_hess = MLPBlockSlice(W_gate, W_up, W_down, X_hess)
                H_X_diag = input_hessian_diag(X_hess)
                H_G_gate = output_hessian_diag(W_gate, X_hess)
                H_G_up = output_hessian_diag(W_up, X_hess)
                H_G_down = output_hessian_diag(W_down, mlp_hess.h)
                H_X_down = np.sum(mlp_hess.h ** 2, axis=0) / mlp_hess.h.shape[0]

                # No rotation
                Wg_q = quantize_per_column(W_gate, K)
                Wu_q = quantize_per_column(W_up, K)
                Wd_q = quantize_per_column(W_down, K)
                no_rot_errs.append(mlp.block_error(Wg_q, Wu_q, Wd_q))
                no_rot_indiv.append(mlp.individual_errors(Wg_q, Wu_q, Wd_q))

                # Hadamard-only
                Wg_r, Ug, Vg = hadamard_only(W_gate, srng)
                Wg_rq = quantize_per_column(Wg_r, K)
                Wg_rec = inverse_hadamard_only(Wg_rq, Ug, Vg)

                Wu_r, Uu, Vu = hadamard_only(W_up, srng)
                Wu_rq = quantize_per_column(Wu_r, K)
                Wu_rec = inverse_hadamard_only(Wu_rq, Uu, Vu)

                Wd_r, Ud, Vd = hadamard_only(W_down, srng)
                Wd_rq = quantize_per_column(Wd_r, K)
                Wd_rec = inverse_hadamard_only(Wd_rq, Ud, Vd)

                hadamard_errs.append(mlp.block_error(Wg_rec, Wu_rec, Wd_rec))
                hadamard_indiv.append(mlp.individual_errors(Wg_rec, Wu_rec, Wd_rec))

                # BiIP+Hadamard (conservative)
                Wg_r, Sg_g, Sx_g, Ug, Vg = biip_hadamard_conservative(W_gate, H_X_diag, H_G_gate, srng)
                Wg_rq = quantize_per_column(Wg_r, K)
                Wg_rec = inverse_biip_hadamard_conservative(Wg_rq, Sg_g, Sx_g, Ug, Vg)

                Wu_r, Sg_u, Sx_u, Uu, Vu = biip_hadamard_conservative(W_up, H_X_diag, H_G_up, srng)
                Wu_rq = quantize_per_column(Wu_r, K)
                Wu_rec = inverse_biip_hadamard_conservative(Wu_rq, Sg_u, Sx_u, Uu, Vu)

                Wd_r, Sg_d, Sx_d, Ud, Vd = biip_hadamard_conservative(W_down, H_X_down, H_G_down, srng)
                Wd_rq = quantize_per_column(Wd_r, K)
                Wd_rec = inverse_biip_hadamard_conservative(Wd_rq, Sg_d, Sx_d, Ud, Vd)

                biip_had_errs.append(mlp.block_error(Wg_rec, Wu_rec, Wd_rec))
                biip_had_indiv.append(mlp.individual_errors(Wg_rec, Wu_rec, Wd_rec))

            no_rot = np.mean(no_rot_errs)
            had = np.mean(hadamard_errs)
            biip = np.mean(biip_had_errs)

            avg_nr = {k: float(np.mean([d[k] for d in no_rot_indiv])) for k in no_rot_indiv[0]}
            avg_had = {k: float(np.mean([d[k] for d in hadamard_indiv])) for k in hadamard_indiv[0]}
            avg_biip = {k: float(np.mean([d[k] for d in biip_had_indiv])) for k in biip_had_indiv[0]}

            layer_results[f'K{K}'] = {
                'no_rotation_block_error': float(no_rot),
                'hadamard_only_block_error': float(had),
                'biip_hadamard_block_error': float(biip),
                'hadamard_improvement_pct': float((no_rot - had) / (no_rot + 1e-30) * 100),
                'biip_hadamard_improvement_pct': float((no_rot - biip) / (no_rot + 1e-30) * 100),
                'no_rotation_individual': avg_nr,
                'hadamard_individual': avg_had,
                'biip_hadamard_individual': avg_biip,
                'hadamard_block_error_std': float(np.std(hadamard_errs)),
                'biip_hadamard_block_error_std': float(np.std(biip_had_errs)),
            }
        results[layer] = layer_results

    # --- Attention V/O rotation (R10 free invariant, correct orientation) ---
    # Storage: W_V [d_head, d_model], W_O [d_model, d_head], forward uses X @ W.T
    # V/O shared R rotates the d_head axis (the shared V/O dimension):
    #   WV_rot = R.T @ W_V  (rotates output/head dim of V)
    #   WO_rot = W_O @ R    (rotates input/head dim of O)
    #   Recon: WV_hat = R @ WV_rot_q, WO_hat = WO_rot_q @ R.T
    # Preserves forward chain: WV_hat.T @ WO_hat.T = W_V.T @ W_O.T (via R @ R.T = I)
    # Equivalently preserves W_O @ W_V (the untransposed product).
    attn_results = {}
    W_Q = weights['synth_WQ']
    W_K = weights['synth_WK']
    W_V = weights['synth_WV']
    W_O = weights['synth_WO']

    for K in K_VALUES:
        no_rot_errs = []
        vo_rot_errs = []
        all_rot_errs = []

        for seed in range(N_SEEDS):
            srng = np.random.default_rng(seed * 100 + 42)
            X = synthetic_input(SLICE, BATCH, srng)
            attn = AttentionBlockSlice(W_Q, W_K, W_V, W_O, X, d_head=SLICE)

            # No rotation
            WQ_q = quantize_per_column(W_Q, K)
            WK_q = quantize_per_column(W_K, K)
            WV_q = quantize_per_column(W_V, K)
            WO_q = quantize_per_column(W_O, K)
            no_rot_errs.append(attn.block_error(WQ_q, WK_q, WV_q, WO_q))

            # V/O shared rotation (R10 free invariant, correct orientation)
            R = signed_random_hadamard(SLICE, srng)
            WV_rot = R.T @ W_V   # rotate head/output dim
            WO_rot = W_O @ R      # rotate head/input dim
            WV_rot_q = quantize_per_column(WV_rot, K)
            WO_rot_q = quantize_per_column(WO_rot, K)
            WV_recon = R @ WV_rot_q      # inverse: R @ (R.T @ W_V)_q
            WO_recon = WO_rot_q @ R.T     # inverse: (W_O @ R)_q @ R.T

            # Verify unquantized invariant
            # WV_recon_unquant = R @ (R.T @ W_V) = W_V
            # WO_recon_unquant = (W_O @ R) @ R.T = W_O

            vo_rot_errs.append(attn.block_error(WQ_q, WK_q, WV_recon, WO_recon))

            # Also test: rotate ALL four matrices independently with Hadamard
            WQ_r, UQ, VQ = hadamard_only(W_Q, srng)
            WQ_rq = quantize_per_column(WQ_r, K)
            WQ_rec = inverse_hadamard_only(WQ_rq, UQ, VQ)

            WK_r, UK, VK = hadamard_only(W_K, srng)
            WK_rq = quantize_per_column(WK_r, K)
            WK_rec = inverse_hadamard_only(WK_rq, UK, VK)

            all_rot_errs.append(attn.block_error(WQ_rec, WK_rec, WV_recon, WO_recon))

        no_rot = np.mean(no_rot_errs)
        vo_rot = np.mean(vo_rot_errs)
        all_rot = np.mean(all_rot_errs)

        attn_results[f'K{K}'] = {
            'no_rotation_block_error': float(no_rot),
            'vo_rotation_block_error': float(vo_rot),
            'all_hadamard_block_error': float(all_rot),
            'vo_improvement_pct': float((no_rot - vo_rot) / (no_rot + 1e-30) * 100),
            'all_hadamard_improvement_pct': float((no_rot - all_rot) / (no_rot + 1e-30) * 100),
            'vo_rotation_block_error_std': float(np.std(vo_rot_errs)),
            'all_hadamard_block_error_std': float(np.std(all_rot_errs)),
            'weight_source': 'synthetic',
        }
    results['attention_synth'] = attn_results

    return results

# =============================================================================
# Experiment 4: Joint block bit allocation (greedy marginal improvement)
# =============================================================================

def measure_block_error_mlp(mlp, W_gate, W_up, W_down, alloc, X=None):
    Wg_q = quantize_per_column(W_gate, alloc[0])
    Wu_q = quantize_per_column(W_up, alloc[1])
    Wd_q = quantize_per_column(W_down, alloc[2])
    return mlp.block_error(Wg_q, Wu_q, Wd_q, X)

def greedy_block_allocation_mlp(mlp, W_gate, W_up, W_down, total_budget_units, X=None):
    """Greedy marginal improvement allocation across 3 MLP matrices.

    NOTE: This is greedy, not dynamic programming. At each step, increment
    the matrix that reduces block error the most. For K5 with 3 matrices,
    exhaustive enumeration (5^3=125) confirmed greedy matches global optimum
    on all 5 trials.
    """
    matrices = [W_gate, W_up, W_down]
    n_mat = 3
    min_k = 2
    max_k = 6

    alloc = [min_k] * n_mat
    remaining = total_budget_units - sum(alloc)

    if remaining < 0:
        return alloc, measure_block_error_mlp(mlp, W_gate, W_up, W_down, alloc, X)

    while remaining > 0:
        best_err = float('inf')
        best_idx = -1
        for i in range(n_mat):
            if alloc[i] >= max_k:
                continue
            trial = alloc.copy()
            trial[i] += 1
            err = measure_block_error_mlp(mlp, W_gate, W_up, W_down, trial, X)
            if err < best_err:
                best_err = err
                best_idx = i
        if best_idx == -1:
            break
        alloc[best_idx] += 1
        remaining -= 1

    final_err = measure_block_error_mlp(mlp, W_gate, W_up, W_down, alloc, X)
    return alloc, final_err

def exp4_joint_allocation(weights, rng):
    """Greedy allocation across all matrices in the block vs uniform.
    Search on calibration X, evaluate on held-out X.
    """
    results = {}

    for layer in ['L0', 'L55']:
        layer_results = {}
        W_gate = weights[f'{layer}_gate']
        W_up = weights[f'{layer}_up']
        W_down = weights[f'{layer}_down']

        for K in K_VALUES:
            total_budget = 3 * K
            uniform_errs = []
            greedy_errs = []
            greedy_allocs = []
            uniform_alloc = [K, K, K]

            for seed in range(N_SEEDS):
                srng = np.random.default_rng(seed * 100 + 42)
                X = synthetic_input(SLICE, BATCH, srng)
                mlp = MLPBlockSlice(W_gate, W_up, W_down, X)

                ue = measure_block_error_mlp(mlp, W_gate, W_up, W_down, uniform_alloc)
                uniform_errs.append(ue)

                X_search = synthetic_input(SLICE, BATCH, np.random.default_rng(seed * 100 + 999))
                mlp_search = MLPBlockSlice(W_gate, W_up, W_down, X_search)
                alloc, _ = greedy_block_allocation_mlp(mlp_search, W_gate, W_up, W_down, total_budget, X_search)
                greedy_allocs.append(alloc)

                de = measure_block_error_mlp(mlp, W_gate, W_up, W_down, alloc)
                greedy_errs.append(de)

            uniform = np.mean(uniform_errs)
            greedy = np.mean(greedy_errs)
            improvement = (uniform - greedy) / (uniform + 1e-30) * 100

            alloc_counts = {}
            for a in greedy_allocs:
                key = str(tuple(a))
                alloc_counts[key] = alloc_counts.get(key, 0) + 1

            layer_results[f'K{K}'] = {
                'uniform_alloc': uniform_alloc,
                'uniform_block_error': float(uniform),
                'greedy_block_error': float(greedy),
                'improvement_pct': float(improvement),
                'greedy_alloc_frequencies': alloc_counts,
                'greedy_block_error_std': float(np.std(greedy_errs)),
                'held_out': True,
                'method': 'greedy_marginal (not DP)',
            }
        results[layer] = layer_results

    # Attention joint allocation
    attn_results = {}
    W_Q = weights['synth_WQ']
    W_K = weights['synth_WK']
    W_V = weights['synth_WV']
    W_O = weights['synth_WO']

    for K in K_VALUES:
        total_budget = 4 * K
        uniform_errs = []
        greedy_errs = []
        greedy_allocs = []
        uniform_alloc = [K, K, K, K]

        for seed in range(N_SEEDS):
            srng = np.random.default_rng(seed * 100 + 42)
            X = synthetic_input(SLICE, BATCH, srng)
            attn = AttentionBlockSlice(W_Q, W_K, W_V, W_O, X, d_head=SLICE)

            WQ_q = quantize_per_column(W_Q, K)
            WK_q = quantize_per_column(W_K, K)
            WV_q = quantize_per_column(W_V, K)
            WO_q = quantize_per_column(W_O, K)
            uniform_errs.append(attn.block_error(WQ_q, WK_q, WV_q, WO_q))

            X_search = synthetic_input(SLICE, BATCH, np.random.default_rng(seed * 100 + 999))
            attn_search = AttentionBlockSlice(W_Q, W_K, W_V, W_O, X_search, d_head=SLICE)
            mats = [W_Q, W_K, W_V, W_O]
            min_k, max_k = 2, 6
            alloc = [min_k] * 4
            remaining = total_budget - sum(alloc)
            while remaining > 0:
                best_err = float('inf')
                best_idx = -1
                for i in range(4):
                    if alloc[i] >= max_k:
                        continue
                    trial = alloc.copy()
                    trial[i] += 1
                    qs = [quantize_per_column(m, k) for m, k in zip(mats, trial)]
                    err = attn_search.block_error(*qs)
                    if err < best_err:
                        best_err = err
                        best_idx = i
                if best_idx == -1:
                    break
                alloc[best_idx] += 1
                remaining -= 1
            greedy_allocs.append(alloc)

            qs = [quantize_per_column(m, k) for m, k in zip(mats, alloc)]
            greedy_errs.append(attn.block_error(*qs))

        uniform = np.mean(uniform_errs)
        greedy = np.mean(greedy_errs)
        improvement = (uniform - greedy) / (uniform + 1e-30) * 100
        alloc_counts = {}
        for a in greedy_allocs:
            key = str(tuple(a))
            alloc_counts[key] = alloc_counts.get(key, 0) + 1

        attn_results[f'K{K}'] = {
            'uniform_alloc': uniform_alloc,
            'uniform_block_error': float(uniform),
            'greedy_block_error': float(greedy),
            'improvement_pct': float(improvement),
            'greedy_alloc_frequencies': alloc_counts,
            'greedy_block_error_std': float(np.std(greedy_errs)),
            'held_out': True,
            'method': 'greedy_marginal (not DP)',
            'weight_source': 'synthetic',
        }
    results['attention_synth'] = attn_results

    return results

# =============================================================================
# Experiment 5: Nonlinear error amplification
# =============================================================================

def exp5_nonlinear_amplification(weights, rng):
    """Measure how SiLU and softmax affect quantization error propagation.

    SiLU metric: finite_difference_to_jacobian_ratio
    = ||silu(g+dg)*u - silu(g)*u||^2 / ||silu'(g)*dg*u||^2
    A value near 1.0 means the first-order Taylor approximation is accurate
    (the perturbation is small enough for linearization). It does NOT mean
    SiLU has unit slope or operates in an identity regime.

    Softmax metric: empirical_squared_softmax_gain
    = ||softmax(S+dS) - softmax(S)||^2 / ||dS||^2
    This is the empirical squared local gain from score perturbation to
    probability perturbation. It depends on the attention regime (entropy,
    sequence length, masking).

    The SiLU_vs_linear_ratio compares block error with SiLU activation vs
    identity activation (g*u instead of silu(g)*u). This IS a meaningful
    comparison but changes the model.
    """
    results = {}

    # --- SiLU in MLP ---
    for layer in ['L0', 'L55']:
        layer_results = {}
        W_gate = weights[f'{layer}_gate']
        W_up = weights[f'{layer}_up']
        W_down = weights[f'{layer}_down']

        for K in K_VALUES:
            fd_to_jac_ratios = []
            silu_vs_linear_ratios = []
            silu_block_errs = []
            linear_block_errs = []
            mean_silu_derivs = []

            for seed in range(N_SEEDS):
                srng = np.random.default_rng(seed * 100 + 42)
                X = synthetic_input(SLICE, BATCH, srng)
                mlp = MLPBlockSlice(W_gate, W_up, W_down, X)

                Wg_q = quantize_per_column(W_gate, K)

                g = mlp.gate_pre
                g_q = X @ Wg_q.T
                dg = g - g_q

                # Finite difference: actual SiLU error
                h_err_silu = (silu(g) - silu(g_q)) * mlp.up_pre
                # Jacobian (first-order Taylor): silu'(g) * dg * u
                h_err_jac = silu_deriv(g) * dg * mlp.up_pre

                silu_err = frob_sq(h_err_silu)
                jac_err = frob_sq(h_err_jac)
                fd_to_jac_ratios.append(silu_err / (jac_err + 1e-30))
                mean_silu_derivs.append(float(np.mean(silu_deriv(g))))

                # Block error with SiLU (normal model)
                be_silu = mlp.block_error(Wg_q, W_up, W_down)

                # Block error with linear activation (g*u instead of silu(g)*u)
                def forward_linear(W_gate, W_up, W_down, X):
                    g = X @ W_gate.T
                    u = X @ W_up.T
                    h = g * u  # identity instead of SiLU
                    return h @ W_down.T

                Y_ref_lin = forward_linear(W_gate, W_up, W_down, X)
                Y_q_lin = forward_linear(Wg_q, W_up, W_down, X)
                be_linear = frob_sq(Y_ref_lin - Y_q_lin)

                silu_block_errs.append(be_silu)
                linear_block_errs.append(be_linear)
                silu_vs_linear_ratios.append(be_silu / (be_linear + 1e-30))

            layer_results[f'K{K}'] = {
                'fd_to_jacobian_ratio': float(np.mean(fd_to_jac_ratios)),
                'fd_to_jacobian_ratio_std': float(np.std(fd_to_jac_ratios)),
                'fd_to_jacobian_ratio_note': '~1.0 means first-order Taylor approximation is accurate',
                'mean_silu_derivative': float(np.mean(mean_silu_derivs)),
                'silu_block_error': float(np.mean(silu_block_errs)),
                'linear_block_error': float(np.mean(linear_block_errs)),
                'silu_vs_linear_ratio': float(np.mean(silu_vs_linear_ratios)),
                'silu_vs_linear_note': 'compares silu(g)*u model vs g*u model (changes the model)',
            }
        results[f'mlp_{layer}'] = layer_results

    # --- Softmax in attention ---
    W_Q = weights['synth_WQ']
    W_K = weights['synth_WK']
    W_V = weights['synth_WV']
    W_O = weights['synth_WO']

    attn_results = {}
    for K in K_VALUES:
        softmax_gains = []
        softmax_vs_linear_ratios = []
        softmax_block_errs = []
        linear_block_errs = []
        attn_entropies = []
        max_probs = []

        for seed in range(N_SEEDS):
            srng = np.random.default_rng(seed * 100 + 42)
            X = synthetic_input(SLICE, BATCH, srng)
            attn = AttentionBlockSlice(W_Q, W_K, W_V, W_O, X, d_head=SLICE)

            WQ_q = quantize_per_column(W_Q, K)

            Q = X @ W_Q.T
            Q_q = X @ WQ_q.T
            K_vals = X @ W_K.T
            V = X @ W_V.T

            S = Q @ K_vals.T / np.sqrt(SLICE)
            S_q = Q_q @ K_vals.T / np.sqrt(SLICE)

            A = softmax(S)
            A_q = softmax(S_q)

            dS = S - S_q
            softmax_err = frob_sq(A - A_q)
            score_err = frob_sq(dS)
            softmax_gains.append(softmax_err / (score_err + 1e-30))

            # Attention regime statistics
            attn_entropies.append(float(-np.sum(A * np.log(A + 1e-30)) / A.shape[0]))
            max_probs.append(float(np.mean(A.max(axis=-1))))

            # Block error with softmax (normal model)
            be_softmax = attn.block_error(WQ_q, W_K, W_V, W_O)

            # Block error with linear attention (raw scores, no softmax)
            def forward_linear_attn(W_Q, W_K, W_V, W_O, X):
                Q = X @ W_Q.T
                K = X @ W_K.T
                V = X @ W_V.T
                S = Q @ K.T / np.sqrt(SLICE)
                attn_out = S @ V  # no softmax
                return attn_out @ W_O.T

            Y_ref_lin = forward_linear_attn(W_Q, W_K, W_V, W_O, X)
            Y_q_lin = forward_linear_attn(WQ_q, W_K, W_V, W_O, X)
            be_linear = frob_sq(Y_ref_lin - Y_q_lin)

            softmax_block_errs.append(be_softmax)
            linear_block_errs.append(be_linear)
            softmax_vs_linear_ratios.append(be_softmax / (be_linear + 1e-30))

        attn_results[f'K{K}'] = {
            'empirical_squared_softmax_gain': float(np.mean(softmax_gains)),
            'empirical_squared_softmax_gain_std': float(np.std(softmax_gains)),
            'softmax_gain_note': '||dA||^2/||dS||^2, depends on attention regime',
            'mean_attention_entropy': float(np.mean(attn_entropies)),
            'mean_max_probability': float(np.mean(max_probs)),
            'sequence_length': BATCH,
            'softmax_block_error': float(np.mean(softmax_block_errs)),
            'linear_block_error': float(np.mean(linear_block_errs)),
            'softmax_vs_linear_ratio': float(np.mean(softmax_vs_linear_ratios)),
            'softmax_vs_linear_note': 'compares softmax attention vs raw-score attention (changes the model)',
            'weight_source': 'synthetic',
        }
    results['attention_synth'] = attn_results

    return results

# =============================================================================
# Experiment 6: Cross-matrix error interaction
# =============================================================================

def exp6_cross_matrix_interaction(weights, rng):
    """Quantize one matrix at a time, measure block error.
    Interaction = all_quantized - sum(single_matrix_quantized).
    Positive interaction = superadditive (matrices amplify each other's errors).
    Negative interaction = subadditive (matrices cancel each other's errors).

    This is an OUTPUT-SPACE additivity measure (unlike exp1 which compares
    output-space to weight-space), making it an intrinsic block property.
    """
    results = {}

    for layer in ['L0', 'L55']:
        layer_results = {}
        W_gate = weights[f'{layer}_gate']
        W_up = weights[f'{layer}_up']
        W_down = weights[f'{layer}_down']

        for K in K_VALUES:
            combos = {
                'gate_only': (True, False, False),
                'up_only': (False, True, False),
                'down_only': (False, False, True),
                'gate_up': (True, True, False),
                'gate_down': (True, False, True),
                'up_down': (False, True, True),
                'all': (True, True, True),
            }

            combo_results = {}
            for name, (qg, qu, qd) in combos.items():
                errs = []
                for seed in range(N_SEEDS):
                    srng = np.random.default_rng(seed * 100 + 42)
                    X = synthetic_input(SLICE, BATCH, srng)
                    mlp = MLPBlockSlice(W_gate, W_up, W_down, X)

                    Wg = quantize_per_column(W_gate, K) if qg else W_gate.copy()
                    Wu = quantize_per_column(W_up, K) if qu else W_up.copy()
                    Wd = quantize_per_column(W_down, K) if qd else W_down.copy()

                    errs.append(mlp.block_error(Wg, Wu, Wd))

                combo_results[name] = float(np.mean(errs))

            g = combo_results['gate_only']
            u = combo_results['up_only']
            d = combo_results['down_only']
            gu = combo_results['gate_up']
            gd = combo_results['gate_down']
            ud = combo_results['up_down']
            all3 = combo_results['all']

            interaction = all3 - (g + u + d)
            gu_interaction = gu - (g + u)
            gd_interaction = gd - (g + d)
            ud_interaction = ud - (u + d)

            # Output-space additivity ratio (intrinsic block property)
            additivity_ratio = all3 / (g + u + d + 1e-30)

            layer_results[f'K{K}'] = {
                'combos': combo_results,
                'sum_singles': float(g + u + d),
                'all_three': float(all3),
                'total_interaction': float(interaction),
                'interaction_pct': float(interaction / (all3 + 1e-30) * 100),
                'additivity_ratio': float(additivity_ratio),
                'additivity_ratio_note': '>1 = superadditive, <1 = subadditive, =1 = additive',
                'pair_interactions': {
                    'gate_up': float(gu_interaction),
                    'gate_down': float(gd_interaction),
                    'up_down': float(ud_interaction),
                },
                'dominant_matrix': max([('gate', g), ('up', u), ('down', d)], key=lambda x: x[1])[0],
            }
        results[layer] = layer_results

    # Attention cross-matrix interaction
    attn_results = {}
    W_Q = weights['synth_WQ']
    W_K = weights['synth_WK']
    W_V = weights['synth_WV']
    W_O = weights['synth_WO']

    for K in K_VALUES:
        combos = {
            'Q_only': (True, False, False, False),
            'K_only': (False, True, False, False),
            'V_only': (False, False, True, False),
            'O_only': (False, False, False, True),
            'QK': (True, True, False, False),
            'QV': (True, False, True, False),
            'QO': (True, False, False, True),
            'KV': (False, True, True, False),
            'VO': (False, False, True, True),
            'all': (True, True, True, True),
        }

        combo_results = {}
        for name, (qq, qk, qv, qo) in combos.items():
            errs = []
            for seed in range(N_SEEDS):
                srng = np.random.default_rng(seed * 100 + 42)
                X = synthetic_input(SLICE, BATCH, srng)
                attn = AttentionBlockSlice(W_Q, W_K, W_V, W_O, X, d_head=SLICE)

                WQ = quantize_per_column(W_Q, K) if qq else W_Q.copy()
                WK = quantize_per_column(W_K, K) if qk else W_K.copy()
                WV = quantize_per_column(W_V, K) if qv else W_V.copy()
                WO = quantize_per_column(W_O, K) if qo else W_O.copy()

                errs.append(attn.block_error(WQ, WK, WV, WO))

            combo_results[name] = float(np.mean(errs))

        q = combo_results['Q_only']
        k = combo_results['K_only']
        v = combo_results['V_only']
        o = combo_results['O_only']
        all4 = combo_results['all']
        sum_singles = q + k + v + o
        interaction = all4 - sum_singles
        additivity_ratio = all4 / (sum_singles + 1e-30)

        attn_results[f'K{K}'] = {
            'combos': combo_results,
            'sum_singles': float(sum_singles),
            'all_four': float(all4),
            'total_interaction': float(interaction),
            'interaction_pct': float(interaction / (all4 + 1e-30) * 100),
            'additivity_ratio': float(additivity_ratio),
            'additivity_ratio_note': '>1 = superadditive, <1 = subadditive, =1 = additive',
            'dominant_matrix': max([('Q', q), ('K', k), ('V', v), ('O', o)], key=lambda x: x[1])[0],
            'weight_source': 'synthetic',
        }
    results['attention_synth'] = attn_results

    return results

# =============================================================================
# Regression checks
# =============================================================================

def regression_checks(weights):
    """Verify key invariants before running experiments."""
    checks = []
    rng = np.random.default_rng(999)

    # Check 1: GQA attention — changing Q changes output
    W_Q = weights['synth_GQA_Q']
    W_K = weights['synth_GQA_K']
    W_V = weights['synth_GQA_V']
    W_O = weights['synth_GQA_O']
    X = synthetic_input(SLICE, BATCH, rng)
    gqa = GQAAttentionBlockSlice(W_Q, W_K, W_V, W_O, X, d_head=D_HEAD, n_q_per_kv=N_Q_PER_KV)
    Y1 = gqa.forward(W_Q, W_K, W_V, W_O)
    W_Q_zero = np.zeros_like(W_Q)
    Y2 = gqa.forward(W_Q_zero, W_K, W_V, W_O)
    q_sensitivity = frob_sq(Y1 - Y2)
    checks.append(('GQA Q sensitivity', q_sensitivity > 1e-10, f'err={q_sensitivity:.6e}'))

    # Check 2: GQA attention — score matrix has seq×seq shape
    Q = X @ W_Q.T
    K_vals = X @ W_K.T
    Q_r = Q.reshape(BATCH, N_Q_PER_KV, D_HEAD)
    S = np.einsum('bhd,jd->bhj', Q_r, K_vals) / np.sqrt(D_HEAD)
    checks.append(('GQA score shape', S.shape == (BATCH, N_Q_PER_KV, BATCH), f'shape={S.shape}'))

    # Check 3: V/O rotation invariant (unquantized, rectangular test)
    # Use rectangular matrices (d_model=5, d_head=3) to catch orientation bugs
    R_rect = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    WV_rect = rng.standard_normal((3, 5))  # [d_head, d_model]
    WO_rect = rng.standard_normal((5, 3))  # [d_model, d_head]
    WV_rect_rot = R_rect.T @ WV_rect
    WO_rect_rot = WO_rect @ R_rect
    WV_rect_rec = R_rect @ WV_rect_rot
    WO_rect_rec = WO_rect_rot @ R_rect.T
    vo_rect_err = frob_sq(WV_rect - WV_rect_rec) + frob_sq(WO_rect - WO_rect_rec)
    checks.append(('V/O rotation invariant (rectangular)', vo_rect_err < 1e-20, f'err={vo_rect_err:.2e}'))
    # Also verify forward chain preservation: WV.T @ WO.T preserved
    chain_orig = WV_rect.T @ WO_rect.T  # [d_model, d_model]
    chain_rot = WV_rect_rot.T @ WO_rect_rot.T
    chain_err = frob_sq(chain_orig - chain_rot)
    checks.append(('V/O forward chain preserved', chain_err < 1e-20, f'err={chain_err:.2e}'))

    # Check 4: Hadamard orthogonality
    H = hadamard_matrix(128)
    orth_err = frob_sq(H @ H.T - np.eye(128))
    checks.append(('Hadamard orthogonality', orth_err < 1e-20, f'err={orth_err:.2e}'))

    # Check 5: up != gate (distinct slices, both L0 and L55)
    for layer in ['L0', 'L55']:
        g = weights[f'{layer}_gate']
        u = weights[f'{layer}_up']
        checks.append((f'{layer} up != gate', not np.allclose(g, u),
                       f'gate_norm={frob_sq(g):.4f} up_norm={frob_sq(u):.4f}'))

    for name, passed, detail in checks:
        status = 'PASS' if passed else 'FAIL'
        print(f"  [{status}] {name}: {detail}")

    return all(p for _, p, _ in checks)

# =============================================================================
# Main
# =============================================================================

def run_all_experiments():
    print("=" * 80)
    print("R18-BlockPropagation: Block-level error propagation through MLP and attention")
    print("=" * 80)

    rng = np.random.default_rng(42)
    weights = load_weights()

    print("\n--- Weight slices loaded ---")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")

    # Regression checks
    print("\n--- Regression checks ---")
    if not regression_checks(weights):
        print("ERROR: Regression checks failed. Aborting.")
        return None

    all_results = {}

    # Exp 1: MLP block error propagation
    print("\n[1/6] MLP block error propagation...")
    t0 = time.time()
    all_results['exp1_mlp_propagation'] = exp1_mlp_propagation(weights, rng)
    print(f"  Done in {time.time()-t0:.1f}s")
    for layer in ['L0', 'L55']:
        r = all_results['exp1_mlp_propagation'][layer]['K5']
        print(f"  {layer} K5: block_err={r['block_error']:.6f}, sum_indiv={r['sum_individual_weight_errors']:.6f}, "
              f"ratio={r['output_to_weight_error_ratio']:.4f}")

    # Exp 2: Attention block error propagation
    print("\n[2/6] Attention block error propagation (synthetic weights)...")
    t0 = time.time()
    all_results['exp2_attention_propagation'] = exp2_attention_propagation(weights, rng)
    print(f"  Done in {time.time()-t0:.1f}s")
    for label in ['single_head_synth', 'gqa_synth']:
        r = all_results['exp2_attention_propagation'][label]['K5']
        print(f"  {label} K5: block_err={r['block_error']:.6f}, sum_indiv={r['sum_individual_weight_errors']:.6f}, "
              f"ratio={r['output_to_weight_error_ratio']:.4f}")

    # Exp 3: Rotation propagation
    print("\n[3/6] Error propagation with rotation...")
    t0 = time.time()
    all_results['exp3_rotation_propagation'] = exp3_rotation_propagation(weights, rng)
    print(f"  Done in {time.time()-t0:.1f}s")
    for layer in ['L0', 'L55']:
        r = all_results['exp3_rotation_propagation'][layer]['K5']
        print(f"  MLP {layer} K5: no_rot={r['no_rotation_block_error']:.6f}, "
              f"had={r['hadamard_only_block_error']:.6f} ({r['hadamard_improvement_pct']:+.1f}%), "
              f"biip_had={r['biip_hadamard_block_error']:.6f} ({r['biip_hadamard_improvement_pct']:+.1f}%)")
    r = all_results['exp3_rotation_propagation']['attention_synth']['K5']
    print(f"  Attn(synth) K5: no_rot={r['no_rotation_block_error']:.6f}, "
          f"vo_rot={r['vo_rotation_block_error']:.6f} ({r['vo_improvement_pct']:+.1f}%), "
          f"all_had={r['all_hadamard_block_error']:.6f} ({r['all_hadamard_improvement_pct']:+.1f}%)")

    # Exp 4: Joint allocation
    print("\n[4/6] Joint block bit allocation (greedy, not DP)...")
    t0 = time.time()
    all_results['exp4_joint_allocation'] = exp4_joint_allocation(weights, rng)
    print(f"  Done in {time.time()-t0:.1f}s")
    for layer in ['L0', 'L55']:
        r = all_results['exp4_joint_allocation'][layer]['K5']
        print(f"  MLP {layer} K5: uniform={r['uniform_block_error']:.6f}, greedy={r['greedy_block_error']:.6f}, "
              f"improvement={r['improvement_pct']:+.1f}%, allocs={r['greedy_alloc_frequencies']}")
    r = all_results['exp4_joint_allocation']['attention_synth']['K5']
    print(f"  Attn(synth) K5: uniform={r['uniform_block_error']:.6f}, greedy={r['greedy_block_error']:.6f}, "
          f"improvement={r['improvement_pct']:+.1f}%, allocs={r['greedy_alloc_frequencies']}")

    # Exp 5: Nonlinear amplification
    print("\n[5/6] Nonlinear error amplification (SiLU/softmax)...")
    t0 = time.time()
    all_results['exp5_nonlinear_amplification'] = exp5_nonlinear_amplification(weights, rng)
    print(f"  Done in {time.time()-t0:.1f}s")
    for layer in ['L0', 'L55']:
        r = all_results['exp5_nonlinear_amplification'][f'mlp_{layer}']['K5']
        print(f"  MLP {layer} K5: fd/jac={r['fd_to_jacobian_ratio']:.4f}, "
              f"silu/lin={r['silu_vs_linear_ratio']:.4f}, mean_silu'={r['mean_silu_derivative']:.4f}")
    r = all_results['exp5_nonlinear_amplification']['attention_synth']['K5']
    print(f"  Attn(synth) K5: softmax_gain={r['empirical_squared_softmax_gain']:.6f}, "
          f"sm/lin={r['softmax_vs_linear_ratio']:.4f}, entropy={r['mean_attention_entropy']:.3f}")

    # Exp 6: Cross-matrix interaction
    print("\n[6/6] Cross-matrix error interaction...")
    t0 = time.time()
    all_results['exp6_cross_matrix_interaction'] = exp6_cross_matrix_interaction(weights, rng)
    print(f"  Done in {time.time()-t0:.1f}s")
    for layer in ['L0', 'L55']:
        r = all_results['exp6_cross_matrix_interaction'][layer]['K5']
        print(f"  MLP {layer} K5: dominant={r['dominant_matrix']}, interaction={r['interaction_pct']:.1f}%, "
              f"additivity={r['additivity_ratio']:.4f}")
    r = all_results['exp6_cross_matrix_interaction']['attention_synth']['K5']
    print(f"  Attn(synth) K5: dominant={r['dominant_matrix']}, interaction={r['interaction_pct']:.1f}%, "
          f"additivity={r['additivity_ratio']:.4f}")

    return all_results

if __name__ == '__main__':
    results = run_all_experiments()

    if results is not None:
        results_path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                    'receipts', 'research', 'r18-block-propagation-results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
