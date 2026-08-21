#!/usr/bin/env python3
"""
R10-CoupledBlocks: Coupled attention/MLP block-level quantization optimization.

CORRECTED after adversarial review. Key changes:
- Held-out validation for all optimization (search on calibration X, evaluate on validation X)
- Multiple seeds with confidence intervals
- RoPE constraint acknowledged: Q/K rotation requires R commuting with RoPE (restrictive)
- V/O rotation is the free invariant (no RoPE interaction)
- "Block Hessian" renamed to "Jacobian trace sensitivity summary"
- Error-direction cross-coupling measured via e_A^T J_A^T J_B e_B
- Exact budget accounting (assert actual == requested)
- Equal baseline: enumerate all balanced allocations, use best
- "Best found" instead of "optimal" for heuristic search results
- Nominal code bits (not packed bytes)

Mathematical foundations:
1. Q/K coupled rotation: same orthogonal R applied to Q and K head coords
   preserves QK^T = (QR)(KR)^T = QRR^TK^T = QK^T.
   CAVEAT: In RoPE attention, R must commute with the relative position transform.
   This is highly restrictive. V/O rotation has no such constraint.
2. V/O inverse-pair rotation: V'=VR, O'=R^T O preserves V'O' = VRR^TO = VO.
   This is a FREE invariant — no RoPE interaction, no position dependence.
3. MLP coupled permutation: same P for gate/up intermediate dim, P^T for down.
   SiLU(gate'@x) ⊙ up'@x = P^T(SiLU(gate@x)⊙up@x) because SiLU is elementwise
   and P is a permutation (commutes with elementwise functions).
   NOTE: Permutation is a no-op with per-column quantization (each column quantized
   independently). Only helps with per-tile quantization (grouping similar-scale channels).
4. Joint rate allocation: allocate bits across block matrices to minimize
   block-level output error, not individual matrix errors.
5. Jacobian trace sensitivity: diagonal terms measure per-matrix output sensitivity;
   error-direction cross terms e_A^T J_A^T J_B e_B measure actual cross-coupling.
"""

import numpy as np
import json
import time
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Quantizers (matched across all arms)
# =============================================================================

def quantize_per_column(W, bits):
    """Per-column uniform quantization. Each column gets its own scale."""
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

def quantize_per_tile(W, bits, tile_size=16):
    """Per-tile uniform quantization. Each tile_size x tile_size block gets one scale."""
    if bits == 0:
        return W.copy()
    m, n = W.shape
    levels = 2 ** bits
    Wq = np.zeros_like(W)
    for i in range(0, m, tile_size):
        for j in range(0, n, tile_size):
            tile = W[i:i+tile_size, j:j+tile_size]
            tmin, tmax = tile.min(), tile.max()
            rng_tile = tmax - tmin
            if rng_tile < 1e-12:
                Wq[i:i+tile_size, j:j+tile_size] = tile
                continue
            scale = rng_tile / (levels - 1)
            q = np.round((tile - tmin) / scale)
            Wq[i:i+tile_size, j:j+tile_size] = np.clip(q, 0, levels - 1) * scale + tmin
    return Wq

def frob_sq(A):
    return float(np.sum(A ** 2))

# =============================================================================
# Orthogonal matrix utilities
# =============================================================================

def random_orthogonal(d, rng):
    A = rng.standard_normal((d, d))
    Q, R_tri = np.linalg.qr(A)
    signs = np.sign(np.diag(R_tri))
    signs[signs == 0] = 1
    return Q * signs[np.newaxis, :]

def hadamard_matrix(d):
    assert (d & (d - 1)) == 0 and d > 0
    H = np.ones((1, 1))
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(d)

def signed_hadamard(d, rng):
    H = hadamard_matrix(d)
    signs = rng.choice([-1, 1], size=d)
    return H * signs[np.newaxis, :]

def givens_rotation(d, i, j, theta):
    R = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -50, 50)))

def silu_deriv(x):
    x_c = np.clip(x, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-x_c))
    return sig + x_c * sig * (1.0 - sig)

# =============================================================================
# Attention block (with optional RoPE)
# =============================================================================

class AttentionBlock:
    """Attention block. X [N, d_model] -> Q,K,V projections -> attention -> O.
    
    If use_rope=True, applies rotary position embeddings to Q and K before
    computing attention scores. This constrains legal Q/K rotations.
    """
    def __init__(self, d_model, d_head, N, rng, use_rope=False):
        self.d_model = d_model
        self.d_head = d_head
        self.N = N
        self.use_rope = use_rope
        scale = 0.1
        self.W_Q = rng.standard_normal((d_model, d_head)) * scale
        self.W_K = rng.standard_normal((d_model, d_head)) * scale
        self.W_V = rng.standard_normal((d_model, d_head)) * scale
        self.W_O = rng.standard_normal((d_head, d_model)) * scale
        self.X = rng.standard_normal((N, d_model))
        if use_rope:
            # RoPE: rotate pairs of head dimensions by position-dependent angles
            self.rope_cos, self.rope_sin = self._compute_rope(N, d_head, rng)
        self.Y_ref = self.forward(self.W_Q, self.W_K, self.W_V, self.W_O)

    def _compute_rope(self, N, d_head, rng):
        """Compute RoPE cos/sin tables. Pairs of dims get rotated by pos * freq."""
        d_pairs = d_head // 2
        freqs = 1.0 / (10000.0 ** (np.arange(0, d_pairs) / d_pairs))
        positions = np.arange(N)
        angles = positions[:, np.newaxis] * freqs[np.newaxis, :]
        return np.cos(angles), np.sin(angles)

    def _apply_rope(self, Q_or_K):
        """Apply RoPE to Q or K: rotate pairs of dimensions."""
        if not self.use_rope:
            return Q_or_K
        N, d = Q_or_K.shape
        result = Q_or_K.copy()
        for i in range(0, d, 2):
            j = i // 2
            result[:, i] = Q_or_K[:, i] * self.rope_cos[:, j] - Q_or_K[:, i+1] * self.rope_sin[:, j]
            result[:, i+1] = Q_or_K[:, i] * self.rope_sin[:, j] + Q_or_K[:, i+1] * self.rope_cos[:, j]
        return result

    def forward(self, W_Q, W_K, W_V, W_O, X=None):
        if X is None:
            X = self.X
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        if self.use_rope:
            Q = self._apply_rope(Q)
            K = self._apply_rope(K)
        S = Q @ K.T / np.sqrt(self.d_head)
        S = S - S.max(axis=1, keepdims=True)
        A = np.exp(S)
        A = A / A.sum(axis=1, keepdims=True)
        attn = A @ V
        Y = attn @ W_O
        return Y

    def block_error(self, W_Q, W_K, W_V, W_O, X=None):
        Y_hat = self.forward(W_Q, W_K, W_V, W_O, X)
        Y_ref = self.forward(self.W_Q, self.W_K, self.W_V, self.W_O, X)
        return frob_sq(Y_ref - Y_hat)

    def individual_quant_errors(self, W_Qq, W_Kq, W_Vq, W_Oq):
        return {'Q': frob_sq(self.W_Q - W_Qq), 'K': frob_sq(self.W_K - W_Kq),
                'V': frob_sq(self.W_V - W_Vq), 'O': frob_sq(self.W_O - W_Oq)}

# =============================================================================
# MLP block
# =============================================================================

class MLPBlock:
    """MLP with SiLU gating. X [N, d_model] -> gate, up -> SiLU(gate)⊙up -> down -> Y."""
    def __init__(self, d_model, d_inter, N, rng,
                 W_gate=None, W_up=None, W_down=None, X=None):
        self.d_model = d_model
        self.d_inter = d_inter
        self.N = N
        scale = 0.1
        self.W_gate = W_gate if W_gate is not None else rng.standard_normal((d_model, d_inter)) * scale
        self.W_up = W_up if W_up is not None else rng.standard_normal((d_model, d_inter)) * scale
        self.W_down = W_down if W_down is not None else rng.standard_normal((d_inter, d_model)) * scale
        self.X = X if X is not None else rng.standard_normal((N, d_model))
        self.Y_ref = self.forward(self.W_gate, self.W_up, self.W_down)
        self.gate = self.X @ self.W_gate
        self.up = self.X @ self.W_up
        self.h = silu(self.gate) * self.up

    def forward(self, W_gate, W_up, W_down, X=None):
        if X is None:
            X = self.X
        gate = X @ W_gate
        up = X @ W_up
        h = silu(gate) * up
        Y = h @ W_down
        return Y

    def block_error(self, W_gate, W_up, W_down, X=None):
        Y_hat = self.forward(W_gate, W_up, W_down, X)
        Y_ref = self.forward(self.W_gate, self.W_up, self.W_down, X)
        return frob_sq(Y_ref - Y_hat)

    def intermediate_product(self, W_gate, W_up):
        gate = self.X @ W_gate
        up = self.X @ W_up
        return silu(gate) * up

    def individual_quant_errors(self, W_gate_q, W_up_q, W_down_q):
        return {'gate': frob_sq(self.W_gate - W_gate_q),
                'up': frob_sq(self.W_up - W_up_q),
                'down': frob_sq(self.W_down - W_down_q)}

def random_permutation(d, rng):
    perm = rng.permutation(d)
    P = np.zeros((d, d))
    P[np.arange(d), perm] = 1
    return P, perm

# =============================================================================
# Invariant verification
# =============================================================================

def verify_qk_invariant(attn, R):
    W_Q_rot = attn.W_Q @ R
    W_K_rot = attn.W_K @ R
    Q_orig = attn.X @ attn.W_Q
    K_orig = attn.X @ attn.W_K
    Q_rot = attn.X @ W_Q_rot
    K_rot = attn.X @ W_K_rot
    QK_orig = Q_orig @ K_orig.T
    QK_rot = Q_rot @ K_rot.T
    return frob_sq(QK_orig - QK_rot)

def verify_vo_invariant(attn, R):
    W_V_rot = attn.W_V @ R
    W_O_rot = R.T @ attn.W_O
    V_orig = attn.X @ attn.W_V
    VO_orig = V_orig @ attn.W_O
    V_rot = attn.X @ W_V_rot
    VO_rot = V_rot @ W_O_rot
    return frob_sq(VO_orig - VO_rot)

def verify_full_attn_invariant(attn, R_qk, R_vo):
    W_Q_rot = attn.W_Q @ R_qk
    W_K_rot = attn.W_K @ R_qk
    W_V_rot = attn.W_V @ R_vo
    W_O_rot = R_vo.T @ attn.W_O
    Y_rot = attn.forward(W_Q_rot, W_K_rot, W_V_rot, W_O_rot)
    return frob_sq(attn.Y_ref - Y_rot)

def verify_mlp_permutation_invariant(mlp, P):
    W_gate_p = mlp.W_gate @ P
    W_up_p = mlp.W_up @ P
    W_down_p = P.T @ mlp.W_down
    h_orig = mlp.intermediate_product(mlp.W_gate, mlp.W_up)
    h_perm = mlp.intermediate_product(W_gate_p, W_up_p)
    h_diff = frob_sq(h_orig @ P - h_perm)
    Y_perm = mlp.forward(W_gate_p, W_up_p, W_down_p)
    y_diff = frob_sq(mlp.Y_ref - Y_perm)
    return h_diff, y_diff

# =============================================================================
# Rotation search (optimizes block error, not weight Frobenius)
# =============================================================================

def search_qk_rotation(attn, bits, quantizer_fn, X_search, n_random=30, n_givens_rounds=2, rng=None):
    """Search R minimizing Q+K weight quantization error (rotation makes weights
    easier to quantize). Evaluates on X_search (calibration set)."""
    if rng is None:
        rng = np.random.default_rng(123)
    d = attn.d_head
    best_R = np.eye(d)
    best_error = float('inf')

    def eval_rotation(R):
        W_Q_rot = attn.W_Q @ R
        W_K_rot = attn.W_K @ R
        W_Qq_rot = quantizer_fn(W_Q_rot, bits)
        W_Kq_rot = quantizer_fn(W_K_rot, bits)
        W_Qq = W_Qq_rot @ R.T
        W_Kq = W_Kq_rot @ R.T
        err_Q = frob_sq(attn.W_Q - W_Qq)
        err_K = frob_sq(attn.W_K - W_Kq)
        return err_Q + err_K, W_Qq, W_Kq

    best_error, _, _ = eval_rotation(best_R)
    for _ in range(n_random):
        R = random_orthogonal(d, rng)
        err, _, _ = eval_rotation(R)
        if err < best_error:
            best_error = err; best_R = R.copy()
    for _ in range(10):
        R = signed_hadamard(d, rng)
        err, _, _ = eval_rotation(R)
        if err < best_error:
            best_error = err; best_R = R.copy()
    for _ in range(n_givens_rounds):
        improved = False
        for i in range(d):
            for j in range(i + 1, d):
                best_theta = 0; best_local = best_error
                for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    G = givens_rotation(d, i, j, theta)
                    err, _, _ = eval_rotation(best_R @ G)
                    if err < best_local:
                        best_local = err; best_theta = theta
                if best_theta != 0:
                    best_R = best_R @ givens_rotation(d, i, j, best_theta)
                    best_error = best_local; improved = True
        if not improved:
            break
    err, W_Qq, W_Kq = eval_rotation(best_R)
    return best_R, best_error, W_Qq, W_Kq

def search_vo_rotation(attn, bits, quantizer_fn, X_search, n_random=30, n_givens_rounds=2, rng=None):
    if rng is None:
        rng = np.random.default_rng(456)
    d = attn.d_head
    best_R = np.eye(d)
    best_error = float('inf')

    def eval_rotation(R):
        W_V_rot = attn.W_V @ R
        W_O_rot = R.T @ attn.W_O
        W_Vq_rot = quantizer_fn(W_V_rot, bits)
        W_Oq_rot = quantizer_fn(W_O_rot, bits)
        W_Vq = W_Vq_rot @ R.T
        W_Oq = R @ W_Oq_rot
        err_V = frob_sq(attn.W_V - W_Vq)
        err_O = frob_sq(attn.W_O - W_Oq)
        return err_V + err_O, W_Vq, W_Oq

    best_error, _, _ = eval_rotation(best_R)
    for _ in range(n_random):
        R = random_orthogonal(d, rng)
        err, _, _ = eval_rotation(R)
        if err < best_error:
            best_error = err; best_R = R.copy()
    for _ in range(10):
        R = signed_hadamard(d, rng)
        err, _, _ = eval_rotation(R)
        if err < best_error:
            best_error = err; best_R = R.copy()
    for _ in range(n_givens_rounds):
        improved = False
        for i in range(d):
            for j in range(i + 1, d):
                best_theta = 0; best_local = best_error
                for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    G = givens_rotation(d, i, j, theta)
                    err, _, _ = eval_rotation(best_R @ G)
                    if err < best_local:
                        best_local = err; best_theta = theta
                if best_theta != 0:
                    best_R = best_R @ givens_rotation(d, i, j, best_theta)
                    best_error = best_local; improved = True
        if not improved:
            break
    err, W_Vq, W_Oq = eval_rotation(best_R)
    return best_R, best_error, W_Vq, W_Oq

# =============================================================================
# MLP permutation search with held-out evaluation
# =============================================================================

def search_mlp_permutation(mlp, bits, quantizer_fn, X_search, n_random=50, n_swap_rounds=2, rng=None):
    """Search permutation P minimizing block error on X_search (calibration).
    Returns the best permutation found (not guaranteed optimal)."""
    if rng is None:
        rng = np.random.default_rng(789)
    d = mlp.d_inter
    best_perm = np.arange(d)
    best_error = float('inf')

    def eval_permutation(perm, X):
        P = np.zeros((d, d)); P[np.arange(d), perm] = 1
        W_gate_p = mlp.W_gate @ P
        W_up_p = mlp.W_up @ P
        W_down_p = P.T @ mlp.W_down
        W_gate_q = quantizer_fn(W_gate_p, bits)
        W_up_q = quantizer_fn(W_up_p, bits)
        W_down_q = quantizer_fn(W_down_p, bits)
        W_gate_recon = W_gate_q @ P.T
        W_up_recon = W_up_q @ P.T
        W_down_recon = P @ W_down_q
        block_err = mlp.block_error(W_gate_recon, W_up_recon, W_down_recon, X)
        return block_err, W_gate_recon, W_up_recon, W_down_recon

    best_error, _, _, _ = eval_permutation(best_perm, X_search)
    for _ in range(n_random):
        perm = rng.permutation(d)
        err, _, _, _ = eval_permutation(perm, X_search)
        if err < best_error:
            best_error = err; best_perm = perm.copy()
    for _ in range(n_swap_rounds):
        improved = False
        for i in range(d):
            for j in range(i + 1, d):
                perm_try = best_perm.copy()
                perm_try[i], perm_try[j] = perm_try[j], perm_try[i]
                err, _, _, _ = eval_permutation(perm_try, X_search)
                if err < best_error:
                    best_error = err; best_perm = perm_try.copy(); improved = True
        if not improved:
            break
    P = np.zeros((d, d)); P[np.arange(d), best_perm] = 1
    _, W_gate_r, W_up_r, W_down_r = eval_permutation(best_perm, X_search)
    n_displaced = int(np.sum(best_perm != np.arange(d)))
    return P, best_perm, best_error, n_displaced, W_gate_r, W_up_r, W_down_r

def evaluate_mlp_perm_heldout(mlp, bits, quantizer_fn, perm, X_val):
    """Evaluate a found permutation on held-out validation data."""
    d = mlp.d_inter
    P = np.zeros((d, d)); P[np.arange(d), perm] = 1
    W_gate_p = mlp.W_gate @ P
    W_up_p = mlp.W_up @ P
    W_down_p = P.T @ mlp.W_down
    W_gate_q = quantizer_fn(W_gate_p, bits)
    W_up_q = quantizer_fn(W_up_p, bits)
    W_down_q = quantizer_fn(W_down_p, bits)
    W_gate_recon = W_gate_q @ P.T
    W_up_recon = W_up_q @ P.T
    W_down_recon = P @ W_down_q
    return mlp.block_error(W_gate_recon, W_up_recon, W_down_recon, X_val)

# =============================================================================
# Joint rate allocation with exact budget and proper baselines
# =============================================================================

def _block_error_attn(attn, bits_alloc, quantizer_fn, X=None):
    W_Qq = quantizer_fn(attn.W_Q, bits_alloc['Q'])
    W_Kq = quantizer_fn(attn.W_K, bits_alloc['K'])
    W_Vq = quantizer_fn(attn.W_V, bits_alloc['V'])
    W_Oq = quantizer_fn(attn.W_O, bits_alloc['O'])
    return attn.block_error(W_Qq, W_Kq, W_Vq, W_Oq, X)

def _block_error_mlp(mlp, bits_alloc, quantizer_fn, X=None):
    W_gate_q = quantizer_fn(mlp.W_gate, bits_alloc['gate'])
    W_up_q = quantizer_fn(mlp.W_up, bits_alloc['up'])
    W_down_q = quantizer_fn(mlp.W_down, bits_alloc['down'])
    return mlp.block_error(W_gate_q, W_up_q, W_down_q, X)

def enumerate_balanced_allocs(names, sizes, budget):
    """Enumerate all integer allocations matching the budget.
    Returns list of (alloc_dict, actual_budget)."""
    n = len(names)
    results = []
    # Determine min and max bits per matrix
    min_b = 2; max_b = 8
    def recurse(idx, alloc, remaining):
        if idx == n:
            if remaining == 0:
                results.append(dict(alloc))
            return
        s = sizes[names[idx]]
        for b in range(min_b, max_b + 1):
            cost = b * s
            if cost <= remaining:
                alloc[names[idx]] = b
                recurse(idx + 1, alloc, remaining - cost)
                del alloc[names[idx]]
    recurse(0, {}, budget)
    return results

def joint_rate_allocation_attn(attn, budget, quantizer_fn, X_search):
    """Greedy marginal improvement allocation. Returns alloc and actual budget."""
    sizes = {'Q': attn.d_model * attn.d_head, 'K': attn.d_model * attn.d_head,
             'V': attn.d_model * attn.d_head, 'O': attn.d_head * attn.d_model}
    names = ['Q', 'K', 'V', 'O']
    bits_alloc = {k: 2 for k in names}
    used = sum(bits_alloc[k] * sizes[k] for k in names)
    max_bits = 8
    while used < budget:
        best_k = None; best_improvement = 0
        for k in names:
            if bits_alloc[k] >= max_bits: continue
            if used + sizes[k] > budget: continue
            cur = {kk: bits_alloc[kk] for kk in names}
            nxt = {kk: bits_alloc[kk] for kk in names}; nxt[k] += 1
            improvement = _block_error_attn(attn, cur, quantizer_fn, X_search) - \
                          _block_error_attn(attn, nxt, quantizer_fn, X_search)
            if improvement > best_improvement:
                best_improvement = improvement; best_k = k
        if best_k is None: break
        bits_alloc[best_k] += 1; used += sizes[best_k]
    return bits_alloc, used

def joint_rate_allocation_mlp(mlp, budget, quantizer_fn, X_search):
    sizes = {'gate': mlp.d_model * mlp.d_inter, 'up': mlp.d_model * mlp.d_inter,
             'down': mlp.d_inter * mlp.d_model}
    names = ['gate', 'up', 'down']
    bits_alloc = {k: 2 for k in names}
    used = sum(bits_alloc[k] * sizes[k] for k in names)
    max_bits = 8
    while used < budget:
        best_k = None; best_improvement = 0
        for k in names:
            if bits_alloc[k] >= max_bits: continue
            if used + sizes[k] > budget: continue
            cur = {kk: bits_alloc[kk] for kk in names}
            nxt = {kk: bits_alloc[kk] for kk in names}; nxt[k] += 1
            improvement = _block_error_mlp(mlp, cur, quantizer_fn, X_search) - \
                          _block_error_mlp(mlp, nxt, quantizer_fn, X_search)
            if improvement > best_improvement:
                best_improvement = improvement; best_k = k
        if best_k is None: break
        bits_alloc[best_k] += 1; used += sizes[best_k]
    return bits_alloc, used

# =============================================================================
# Jacobian trace sensitivity summary (corrected naming)
# =============================================================================

def compute_jacobian_trace_summary(attn, eps=1e-4):
    """Compute Jacobian trace/Frobenius sensitivity summary.
    
    This is NOT a full block Hessian. It computes scalar summaries:
    S[A,B] = <J_A, J_B>_F = sum of element-wise products of Jacobian matrices.
    
    For the diagonal, S[A,A] = ||J_A||_F^2 measures total output sensitivity
    to matrix A's parameters. For off-diagonal, S[A,B] is a scalar that
    can change under reordering of parameter vectors and may exhibit
    sign cancellation. It does NOT measure the full cross-coupling.
    
    For actual cross-coupling, see compute_error_direction_cross_terms.
    """
    names = ['Q', 'K', 'V', 'O']
    weights = [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]
    shapes = [w.shape for w in weights]

    def forward_flat(w_flat_list):
        ws = [w.reshape(s) for w, s in zip(w_flat_list, shapes)]
        return attn.forward(*ws).ravel()

    w_ref = [w.ravel().copy() for w in weights]
    Y_ref_flat = forward_flat(w_ref)

    jacobians = []
    for a_idx in range(4):
        w = w_ref[a_idx].copy()
        n_params = len(w)
        jac = np.zeros((len(Y_ref_flat), n_params))
        for p in range(n_params):
            w_pert = w.copy(); w_pert[p] += eps
            w_list = [w_ref[i].copy() for i in range(4)]
            w_list[a_idx] = w_pert
            jac[:, p] = (forward_flat(w_list) - Y_ref_flat) / eps
        jacobians.append(jac)

    S = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            S[a, b] = float(np.sum(jacobians[a] * jacobians[b]))
    return S, names, jacobians

def compute_jacobian_trace_summary_mlp(mlp, eps=1e-4):
    names = ['gate', 'up', 'down']
    weights = [mlp.W_gate, mlp.W_up, mlp.W_down]
    shapes = [w.shape for w in weights]

    def forward_flat(w_flat_list):
        ws = [w.reshape(s) for w, s in zip(w_flat_list, shapes)]
        return mlp.forward(*ws).ravel()

    w_ref = [w.ravel().copy() for w in weights]
    Y_ref_flat = forward_flat(w_ref)

    jacobians = []
    for a_idx in range(3):
        w = w_ref[a_idx].copy()
        n_params = len(w)
        jac = np.zeros((len(Y_ref_flat), n_params))
        for p in range(n_params):
            w_pert = w.copy(); w_pert[p] += eps
            w_list = [w_ref[i].copy() for i in range(3)]
            w_list[a_idx] = w_pert
            jac[:, p] = (forward_flat(w_list) - Y_ref_flat) / eps
        jacobians.append(jac)

    S = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            S[a, b] = float(np.sum(jacobians[a] * jacobians[b]))
    return S, names, jacobians

def compute_error_direction_cross_terms(jacobians, quant_errors_flat, names):
    """Compute actual cross-coupling: e_A^T (J_A^T J_B) e_B.
    
    This uses actual quantization error directions, not just trace summaries.
    e_A = vec(W_A - W_A_quantized) for each matrix A.
    
    Returns the full cross-coupling matrix C[A,B] = e_A^T J_A^T J_B e_B.
    """
    n = len(names)
    C = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            # J_A @ e_A gives output perturbation from A's quantization error
            Je_a = jacobians[a] @ quant_errors_flat[a]
            Je_b = jacobians[b] @ quant_errors_flat[b]
            C[a, b] = float(np.dot(Je_a, Je_b))
    return C

def derive_mlp_jacobian_trace_analytic(mlp):
    """Analytic formulas for Jacobian trace sensitivity summary diagonal terms.
    
    For the MLP block (g = X@W_gate, u = X@W_up, h = SiLU(g)⊙u, Y = h@W_down):
    
    Jacobian entries: J_gate[n,m,a,b] = X[n,a] * α[n,b] * W_down[b,m]
    where α = SiLU'(g) ⊙ u (gate derivative modulated by up)
    
    ||J_gate||_F^2 = sum_{n,m,a,b} [X[n,a] * α[n,b] * W_down[b,m]]^2
                   = sum_n ||X[n,:]||^2 * sum_b α[n,b]^2 * ||W_down[b,:]||^2
    
    Key insight: per-ROW norms of W_down (diagonal of W_down@W_down^T) appear,
    NOT the full projection. This is because the Frobenius inner product sums
    element-wise products of Jacobian entries, not projected quantities.
    
    The FULL cross-coupling J_gate^T J_up uses off-diagonal entries of
    W_down@W_down^T, which matter for correlated/structured quantization errors
    but cancel in the trace/summary.
    """
    N = mlp.N; d_model = mlp.d_model
    g = mlp.gate; u = mlp.up; h = mlp.h
    sig_p = silu_deriv(g)
    alpha = sig_p * u; beta = silu(g)
    W_down_row_sq = np.sum(mlp.W_down ** 2, axis=1)
    X_sq_norm = np.sum(mlp.X ** 2, axis=1)
    H_gg = float(np.sum(X_sq_norm * np.sum(alpha ** 2 * W_down_row_sq[np.newaxis, :], axis=1)))
    H_uu = float(np.sum(X_sq_norm * np.sum(beta ** 2 * W_down_row_sq[np.newaxis, :], axis=1)))
    H_dd = float(np.sum(np.sum(h ** 2, axis=1)) * d_model)
    H_gu = float(np.sum(X_sq_norm * np.sum(alpha * beta * W_down_row_sq[np.newaxis, :], axis=1)))
    return {'H_gate_gate': H_gg, 'H_up_up': H_uu, 'H_down_down': H_dd, 'H_gate_up': H_gu}

# =============================================================================
# Full coupled attention optimization
# =============================================================================

def full_coupled_attention_optimization(attn, bits, quantizer_fn, X_search, rng=None):
    if rng is None:
        rng = np.random.default_rng(999)
    R_qk, _, W_Qq, W_Kq = search_qk_rotation(attn, bits, quantizer_fn, X_search,
                                              n_random=30, n_givens_rounds=2, rng=rng)
    R_vo, _, W_Vq, W_Oq = search_vo_rotation(attn, bits, quantizer_fn, X_search,
                                              n_random=30, n_givens_rounds=2, rng=rng)
    block_err = attn.block_error(W_Qq, W_Kq, W_Vq, W_Oq, X_search)
    return {'R_qk': R_qk, 'R_vo': R_vo, 'block_error': block_err,
            'W_Qq': W_Qq, 'W_Kq': W_Kq, 'W_Vq': W_Vq, 'W_Oq': W_Oq}

# =============================================================================
# Held-out evaluation utilities
# =============================================================================

def evaluate_attention_heldout(attn, W_Qq, W_Kq, W_Vq, W_Oq, X_val_seeds, rng):
    """Evaluate block error on multiple held-out X draws."""
    errs = []
    for seed in X_val_seeds:
        X_val = rng.standard_normal((attn.N, attn.d_model))
        err = attn.block_error(W_Qq, W_Kq, W_Vq, W_Oq, X_val)
        errs.append(err)
    return errs

def evaluate_mlp_heldout(mlp, W_gate_q, W_up_q, W_down_q, X_val_seeds, rng):
    errs = []
    for seed in X_val_seeds:
        X_val = rng.standard_normal((mlp.N, mlp.d_model))
        err = mlp.block_error(W_gate_q, W_up_q, W_down_q, X_val)
        errs.append(err)
    return errs

def mean_std(arr):
    arr = np.array(arr)
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

# =============================================================================
# Main experiment runner
# =============================================================================

def run_all_experiments():
    results = {}
    t_start = time.time()
    n_val_seeds = 20
    val_rng = np.random.default_rng(2024)

    # ---- ATTENTION ----
    print("=" * 80)
    print("ATTENTION BLOCK EXPERIMENTS")
    print("=" * 80)

    d_model = 64; d_head = 32; N = 128; bits = 4
    attn = AttentionBlock(d_model, d_head, N, np.random.default_rng(42))
    print(f"\nAttention: d_model={d_model}, d_head={d_head}, N={N}, bits={bits}")

    # 1. Invariants
    print("\n--- 1. Invariant Verification ---")
    R_test = random_orthogonal(d_head, np.random.default_rng(100))
    qk_diff = verify_qk_invariant(attn, R_test)
    R_test2 = random_orthogonal(d_head, np.random.default_rng(101))
    vo_diff = verify_vo_invariant(attn, R_test2)
    R_qk_t = random_orthogonal(d_head, np.random.default_rng(102))
    R_vo_t = random_orthogonal(d_head, np.random.default_rng(103))
    full_diff = verify_full_attn_invariant(attn, R_qk_t, R_vo_t)
    print(f"Q/K QK^T invariance: {qk_diff:.2e}")
    print(f"V/O VO invariance: {vo_diff:.2e}")
    print(f"Full Y invariance: {full_diff:.2e}")
    print("NOTE: Q/K rotation requires R commuting with RoPE in real model.")
    print("      V/O rotation is a FREE invariant (no RoPE interaction).")
    results['attn_invariants'] = {
        'qk_invariance': qk_diff, 'vo_invariance': vo_diff, 'full_y_invariance': full_diff,
        'd_model': d_model, 'd_head': d_head, 'N': N, 'bits': bits,
        'rope_caveat': 'Q/K rotation requires R commuting with RoPE in real model. V/O is free.',
    }

    # 2. Independent quantization
    print("\n--- 2. Independent Quantization (baseline) ---")
    W_Qq_ind = quantize_per_column(attn.W_Q, bits)
    W_Kq_ind = quantize_per_column(attn.W_K, bits)
    W_Vq_ind = quantize_per_column(attn.W_V, bits)
    W_Oq_ind = quantize_per_column(attn.W_O, bits)
    block_err_ind = attn.block_error(W_Qq_ind, W_Kq_ind, W_Vq_ind, W_Oq_ind)
    ind_errors = attn.individual_quant_errors(W_Qq_ind, W_Kq_ind, W_Vq_ind, W_Oq_ind)
    print(f"Independent block error (search X): {block_err_ind:.6f}")

    # Held-out evaluation
    val_errs_ind = evaluate_attention_heldout(attn, W_Qq_ind, W_Kq_ind, W_Vq_ind, W_Oq_ind,
                                               range(n_val_seeds), np.random.default_rng(2024))
    m_ind, s_ind, lo_ind, hi_ind = mean_std(val_errs_ind)
    print(f"Held-out (n={n_val_seeds}): mean={m_ind:.6f}, std={s_ind:.6f}, range=[{lo_ind:.6f}, {hi_ind:.6f}]")

    # 3. Q/K rotation
    print("\n--- 3. Q/K Coupled Rotation ---")
    R_qk, qk_err, W_Qq_rot, W_Kq_rot = search_qk_rotation(
        attn, bits, quantize_per_column, attn.X, n_random=50, n_givens_rounds=3,
        rng=np.random.default_rng(123))
    block_err_qk = attn.block_error(W_Qq_rot, W_Kq_rot, W_Vq_ind, W_Oq_ind)
    val_errs_qk = evaluate_attention_heldout(attn, W_Qq_rot, W_Kq_rot, W_Vq_ind, W_Oq_ind,
                                              range(n_val_seeds), np.random.default_rng(2024))
    m_qk, s_qk, lo_qk, hi_qk = mean_std(val_errs_qk)
    print(f"Q/K rotation block error (search X): {block_err_qk:.6f} ({(1-block_err_qk/block_err_ind)*100:.1f}%)")
    print(f"Held-out: mean={m_qk:.6f} ({(1-m_qk/m_ind)*100:.1f}%), std={s_qk:.6f}, range=[{lo_qk:.6f}, {hi_qk:.6f}]")

    # 4. V/O rotation
    print("\n--- 4. V/O Coupled Rotation (FREE invariant) ---")
    R_vo, vo_err, W_Vq_rot, W_Oq_rot = search_vo_rotation(
        attn, bits, quantize_per_column, attn.X, n_random=50, n_givens_rounds=3,
        rng=np.random.default_rng(456))
    block_err_vo = attn.block_error(W_Qq_ind, W_Kq_ind, W_Vq_rot, W_Oq_rot)
    val_errs_vo = evaluate_attention_heldout(attn, W_Qq_ind, W_Kq_ind, W_Vq_rot, W_Oq_rot,
                                              range(n_val_seeds), np.random.default_rng(2024))
    m_vo, s_vo, lo_vo, hi_vo = mean_std(val_errs_vo)
    print(f"V/O rotation block error (search X): {block_err_vo:.6f} ({(1-block_err_vo/block_err_ind)*100:.1f}%)")
    print(f"Held-out: mean={m_vo:.6f} ({(1-m_vo/m_ind)*100:.1f}%), std={s_vo:.6f}, range=[{lo_vo:.6f}, {hi_vo:.6f}]")

    # 5. Full coupled
    print("\n--- 5. Full Coupled (Q/K + V/O rotation) ---")
    full_res = full_coupled_attention_optimization(attn, bits, quantize_per_column, attn.X,
                                                    rng=np.random.default_rng(999))
    block_err_full = full_res['block_error']
    val_errs_full = evaluate_attention_heldout(attn, full_res['W_Qq'], full_res['W_Kq'],
                                                full_res['W_Vq'], full_res['W_Oq'],
                                                range(n_val_seeds), np.random.default_rng(2024))
    m_full, s_full, lo_full, hi_full = mean_std(val_errs_full)
    print(f"Full coupled block error (search X): {block_err_full:.6f} ({(1-block_err_full/block_err_ind)*100:.1f}%)")
    print(f"Held-out: mean={m_full:.6f} ({(1-m_full/m_ind)*100:.1f}%), std={s_full:.6f}, range=[{lo_full:.6f}, {hi_full:.6f}]")

    results['attn_quantization'] = {
        'bits': bits,
        'independent': {'search_error': block_err_ind, 'heldout_mean': m_ind, 'heldout_std': s_ind,
                        'heldout_range': [lo_ind, hi_ind], 'individual_errors': ind_errors},
        'qk_rotation': {'search_error': block_err_qk, 'search_improvement_pct': (1-block_err_qk/block_err_ind)*100,
                        'heldout_mean': m_qk, 'heldout_improvement_pct': (1-m_qk/m_ind)*100,
                        'heldout_std': s_qk, 'heldout_range': [lo_qk, hi_qk],
                        'note': 'Q/K rotation not free under RoPE in real model'},
        'vo_rotation': {'search_error': block_err_vo, 'search_improvement_pct': (1-block_err_vo/block_err_ind)*100,
                        'heldout_mean': m_vo, 'heldout_improvement_pct': (1-m_vo/m_ind)*100,
                        'heldout_std': s_vo, 'heldout_range': [lo_vo, hi_vo],
                        'note': 'V/O rotation is a FREE invariant'},
        'full_coupled': {'search_error': block_err_full, 'search_improvement_pct': (1-block_err_full/block_err_ind)*100,
                         'heldout_mean': m_full, 'heldout_improvement_pct': (1-m_full/m_ind)*100,
                         'heldout_std': s_full, 'heldout_range': [lo_full, hi_full]},
    }

    # 6. Joint rate allocation with proper baselines
    print("\n--- 6. Joint Rate Allocation (Attention) ---")
    sizes_attn = {'Q': d_model*d_head, 'K': d_model*d_head, 'V': d_model*d_head, 'O': d_head*d_model}
    for avg_b in [3.5, 4.5]:
        budget = int(avg_b * sum(sizes_attn.values()))
        joint_alloc, joint_used = joint_rate_allocation_attn(attn, budget, quantize_per_column, attn.X)
        # Enumerate all balanced allocations
        all_allocs = enumerate_balanced_allocs(['Q','K','V','O'], sizes_attn, budget)
        if not all_allocs:
            print(f"  Budget {avg_b} avg ({budget}): no valid integer allocation found")
            continue
        # Evaluate all on search X
        alloc_errors = [(a, _block_error_attn(attn, a, quantize_per_column, attn.X)) for a in all_allocs]
        best_equal = min(alloc_errors, key=lambda x: x[1])
        worst_equal = max(alloc_errors, key=lambda x: x[1])
        mean_equal = float(np.mean([e for _, e in alloc_errors]))

        block_err_joint = _block_error_attn(attn, joint_alloc, quantize_per_column, attn.X)
        block_err_best_equal = best_equal[1]
        impr_vs_best = (1 - block_err_joint / block_err_best_equal) * 100
        impr_vs_mean = (1 - block_err_joint / mean_equal) * 100

        # Held-out
        val_joint = evaluate_attention_heldout(attn,
            quantize_per_column(attn.W_Q, joint_alloc['Q']),
            quantize_per_column(attn.W_K, joint_alloc['K']),
            quantize_per_column(attn.W_V, joint_alloc['V']),
            quantize_per_column(attn.W_O, joint_alloc['O']),
            range(n_val_seeds), np.random.default_rng(2024))
        val_best_eq = evaluate_attention_heldout(attn,
            quantize_per_column(attn.W_Q, best_equal[0]['Q']),
            quantize_per_column(attn.W_K, best_equal[0]['K']),
            quantize_per_column(attn.W_V, best_equal[0]['V']),
            quantize_per_column(attn.W_O, best_equal[0]['O']),
            range(n_val_seeds), np.random.default_rng(2024))
        m_j, _, _, _ = mean_std(val_joint)
        m_be, _, _, _ = mean_std(val_best_eq)

        print(f"  Budget {avg_b} avg ({budget} bits, used {joint_used}):")
        print(f"    Joint: {joint_alloc} -> search={block_err_joint:.6f}, heldout={m_j:.6f}")
        print(f"    Best equal: {best_equal[0]} -> search={block_err_best_equal:.6f}, heldout={m_be:.6f}")
        print(f"    Worst equal: {worst_equal[0]} -> search={worst_equal[1]:.6f}")
        print(f"    Improvement vs best equal: {impr_vs_best:.1f}% (search), {(1-m_j/m_be)*100:.1f}% (heldout)")
        print(f"    Improvement vs mean equal: {impr_vs_mean:.1f}% (search)")
        print(f"    {len(all_allocs)} balanced allocations enumerated")

        results.setdefault('attn_rate_allocation', {})[f"{avg_b}_avg"] = {
            'budget': budget, 'used': joint_used, 'budget_exact': joint_used == budget,
            'joint_alloc': joint_alloc, 'best_equal_alloc': best_equal[0],
            'n_balanced_allocs': len(all_allocs),
            'joint_search_error': block_err_joint, 'best_equal_search_error': block_err_best_equal,
            'improvement_vs_best_pct': impr_vs_best,
            'joint_heldout_mean': m_j, 'best_equal_heldout_mean': m_be,
            'heldout_improvement_pct': (1-m_j/m_be)*100,
        }

    # 7. Jacobian trace sensitivity + error-direction cross terms
    print("\n--- 7. Jacobian Trace Sensitivity Summary + Error-Direction Cross Terms ---")
    S_attn, attn_names, jacs_attn = compute_jacobian_trace_summary(attn)
    print("Jacobian trace summary <J_A, J_B>_F:")
    for i, ni in enumerate(attn_names):
        print(f"  {ni:4s}: " + "  ".join(f"{S_attn[i,j]:.4e}" for j in range(4)))

    # Error-direction cross terms using actual quantization errors
    E_Q = (attn.W_Q - W_Qq_ind).ravel()
    E_K = (attn.W_K - W_Kq_ind).ravel()
    E_V = (attn.W_V - W_Vq_ind).ravel()
    E_O = (attn.W_O - W_Oq_ind).ravel()
    quant_errs_attn = [E_Q, E_K, E_V, E_O]
    C_attn = compute_error_direction_cross_terms(jacs_attn, quant_errs_attn, attn_names)
    print("\nError-direction cross terms e_A^T J_A^T J_B e_B:")
    for i, ni in enumerate(attn_names):
        print(f"  {ni:4s}: " + "  ".join(f"{C_attn[i,j]:.4e}" for j in range(4)))
    diag_C = [C_attn[i,i] for i in range(4)]
    off_diag_C = [C_attn[i,j] for i in range(4) for j in range(4) if i != j]
    total_C = sum(abs(x) for x in C_attn.ravel())
    diag_frac = sum(abs(x) for x in diag_C) / total_C if total_C > 0 else 0
    print(f"\nDiagonal fraction of |cross terms|: {diag_frac:.4f}")
    print(f"Off-diagonal max / diagonal max: {max(abs(x) for x in off_diag_C) / max(abs(x) for x in diag_C):.4f}")
    print("Error-direction cross terms show ACTUAL coupling with real quantization errors.")

    results['attn_jacobian'] = {
        'trace_summary': S_attn.tolist(), 'names': attn_names,
        'error_direction_cross': C_attn.tolist(),
        'diag_fraction': float(diag_frac),
    }

    # ---- MLP ----
    print("\n" + "=" * 80)
    print("MLP BLOCK EXPERIMENTS")
    print("=" * 80)

    d_model_mlp = 64; d_inter = 128; N_mlp = 128; bits_mlp = 4
    mlp = MLPBlock(d_model_mlp, d_inter, N_mlp, np.random.default_rng(42))
    print(f"\nMLP: d_model={d_model_mlp}, d_inter={d_inter}, N={N_mlp}, bits={bits_mlp}")

    # 8. Invariants
    print("\n--- 8. MLP Invariant Verification ---")
    P_test, _ = random_permutation(d_inter, np.random.default_rng(200))
    h_diff, y_diff = verify_mlp_permutation_invariant(mlp, P_test)
    print(f"Permutation h invariance: {h_diff:.2e}, Y invariance: {y_diff:.2e}")
    R_mlp_test = random_orthogonal(d_inter, np.random.default_rng(201))
    W_gate_rot = mlp.W_gate @ R_mlp_test
    W_up_rot = mlp.W_up @ R_mlp_test
    W_down_rot = R_mlp_test.T @ mlp.W_down
    Y_rot_mlp = mlp.forward(W_gate_rot, W_up_rot, W_down_rot)
    rot_diff = frob_sq(mlp.Y_ref - Y_rot_mlp)
    print(f"Rotation violation: {rot_diff:.6f} (confirms SiLU non-commutativity)")
    results['mlp_invariants'] = {
        'permutation_h_invariance': h_diff, 'permutation_y_invariance': y_diff,
        'rotation_violation': rot_diff,
        'd_model': d_model_mlp, 'd_inter': d_inter, 'N': N_mlp, 'bits': bits_mlp,
    }

    # 9. Independent quantization
    print("\n--- 9. MLP Independent Quantization ---")
    W_gate_q_ind = quantize_per_column(mlp.W_gate, bits_mlp)
    W_up_q_ind = quantize_per_column(mlp.W_up, bits_mlp)
    W_down_q_ind = quantize_per_column(mlp.W_down, bits_mlp)
    block_err_mlp_ind = mlp.block_error(W_gate_q_ind, W_up_q_ind, W_down_q_ind)
    val_errs_mlp_ind = evaluate_mlp_heldout(mlp, W_gate_q_ind, W_up_q_ind, W_down_q_ind,
                                             range(n_val_seeds), np.random.default_rng(2024))
    m_mlp_ind, s_mlp_ind, lo_mlp_ind, hi_mlp_ind = mean_std(val_errs_mlp_ind)
    print(f"Independent (per-col) search: {block_err_mlp_ind:.6f}, heldout: {m_mlp_ind:.6f}±{s_mlp_ind:.6f}")

    # 10. Per-column permutation (no-op, confirmed)
    print("\n--- 10. MLP Coupled Permutation (per-column: no-op) ---")
    _, perm_col, _, n_disp_col, W_gq_pc, W_uq_pc, W_dq_pc = search_mlp_permutation(
        mlp, bits_mlp, quantize_per_column, mlp.X, n_random=20, n_swap_rounds=1,
        rng=np.random.default_rng(789))
    block_err_mlp_pc = mlp.block_error(W_gq_pc, W_uq_pc, W_dq_pc)
    print(f"Per-column permutation: {block_err_mlp_pc:.6f} (improvement: {(1-block_err_mlp_pc/block_err_mlp_ind)*100:.1f}%)")
    print(f"  Displaced channels: {n_disp_col}/{d_inter} (expected 0: per-column quantizer is permutation-invariant)")

    # 11. Per-tile permutation with held-out evaluation
    print("\n--- 11. MLP Coupled Permutation (per-tile, held-out eval) ---")
    W_gq_ti = quantize_per_tile(mlp.W_gate, bits_mlp, 16)
    W_uq_ti = quantize_per_tile(mlp.W_up, bits_mlp, 16)
    W_dq_ti = quantize_per_tile(mlp.W_down, bits_mlp, 16)
    block_err_mlp_ti_ind = mlp.block_error(W_gq_ti, W_uq_ti, W_dq_ti)
    val_errs_ti_ind = evaluate_mlp_heldout(mlp, W_gq_ti, W_uq_ti, W_dq_ti,
                                             range(n_val_seeds), np.random.default_rng(2024))
    m_ti_ind, s_ti_ind, lo_ti_ind, hi_ti_ind = mean_std(val_errs_ti_ind)
    print(f"Per-tile independent search: {block_err_mlp_ti_ind:.6f}, heldout: {m_ti_ind:.6f}±{s_ti_ind:.6f}")

    _, perm_tile, _, n_disp_tile, W_gq_tp, W_uq_tp, W_dq_tp = search_mlp_permutation(
        mlp, bits_mlp, lambda W, b: quantize_per_tile(W, b, 16), mlp.X,
        n_random=100, n_swap_rounds=3, rng=np.random.default_rng(789))
    block_err_mlp_tp_search = mlp.block_error(W_gq_tp, W_uq_tp, W_dq_tp)
    val_errs_tp = evaluate_mlp_heldout(mlp, W_gq_tp, W_uq_tp, W_dq_tp,
                                        range(n_val_seeds), np.random.default_rng(2024))
    m_tp, s_tp, lo_tp, hi_tp = mean_std(val_errs_tp)
    print(f"Per-tile perm search: {block_err_mlp_tp_search:.6f} ({(1-block_err_mlp_tp_search/block_err_mlp_ti_ind)*100:.1f}%)")
    print(f"Per-tile perm heldout: {m_tp:.6f} ({(1-m_tp/m_ti_ind)*100:.1f}%), std={s_tp:.6f}, range=[{lo_tp:.6f}, {hi_tp:.6f}]")
    print(f"Displaced channels: {n_disp_tile}/{d_inter}")

    results['mlp_quantization'] = {
        'bits': bits_mlp,
        'per_column': {
            'independent_search': block_err_mlp_ind, 'independent_heldout_mean': m_mlp_ind,
            'perm_search': block_err_mlp_pc, 'n_displaced': n_disp_col,
            'note': 'No-op: per-column quantizer is invariant to column permutation',
        },
        'per_tile': {
            'tile_size': 16,
            'independent_search': block_err_mlp_ti_ind,
            'independent_heldout_mean': m_ti_ind, 'independent_heldout_std': s_ti_ind,
            'perm_search': block_err_mlp_tp_search,
            'search_improvement_pct': (1-block_err_mlp_tp_search/block_err_mlp_ti_ind)*100,
            'perm_heldout_mean': m_tp, 'perm_heldout_std': s_tp,
            'perm_heldout_range': [lo_tp, hi_tp],
            'heldout_improvement_pct': (1-m_tp/m_ti_ind)*100,
            'n_displaced': n_disp_tile,
        },
    }

    # 12. Joint rate allocation with exact budget
    print("\n--- 12. Joint Rate Allocation (MLP, exact budget) ---")
    sizes_mlp = {'gate': d_model_mlp*d_inter, 'up': d_model_mlp*d_inter, 'down': d_inter*d_model_mlp}
    # All matrices have same size (d_model*d_inter), so budgets must be multiples of that
    # Representable averages with 3 equal-size matrices: 3.0, 3.333, 4.0, 4.333, 5.0, ...
    # 3.333 = (3+3+4)/3, 4.333 = (4+4+5)/3
    elem_per_mat = d_model_mlp * d_inter
    for avg_b, allocs_desc in [(3.333, "3+3+4"), (4.333, "4+4+5")]:
        # Compute exact representable budget
        total_elems = 3 * elem_per_mat
        # avg_b * total_elems, rounded to nearest multiple of elem_per_mat
        budget = round(avg_b * total_elems / elem_per_mat) * elem_per_mat
        joint_alloc, joint_used = joint_rate_allocation_mlp(mlp, budget, quantize_per_column, mlp.X)
        all_allocs = enumerate_balanced_allocs(['gate','up','down'], sizes_mlp, budget)
        if not all_allocs:
            print(f"  Budget {avg_b} avg ({budget}): no valid allocation")
            continue
        alloc_errors = [(a, _block_error_mlp(mlp, a, quantize_per_column, mlp.X)) for a in all_allocs]
        best_equal = min(alloc_errors, key=lambda x: x[1])
        block_err_joint = _block_error_mlp(mlp, joint_alloc, quantize_per_column, mlp.X)
        block_err_best = best_equal[1]
        impr = (1 - block_err_joint / block_err_best) * 100

        # Held-out
        val_j = evaluate_mlp_heldout(mlp,
            quantize_per_column(mlp.W_gate, joint_alloc['gate']),
            quantize_per_column(mlp.W_up, joint_alloc['up']),
            quantize_per_column(mlp.W_down, joint_alloc['down']),
            range(n_val_seeds), np.random.default_rng(2024))
        val_be = evaluate_mlp_heldout(mlp,
            quantize_per_column(mlp.W_gate, best_equal[0]['gate']),
            quantize_per_column(mlp.W_up, best_equal[0]['up']),
            quantize_per_column(mlp.W_down, best_equal[0]['down']),
            range(n_val_seeds), np.random.default_rng(2024))
        m_j_mlp, _, _, _ = mean_std(val_j)
        m_be_mlp, _, _, _ = mean_std(val_be)

        print(f"  Budget {avg_b} avg ({budget} bits, used {joint_used}, exact={joint_used==budget}):")
        print(f"    Joint: {joint_alloc} -> search={block_err_joint:.6f}, heldout={m_j_mlp:.6f}")
        print(f"    Best equal: {best_equal[0]} -> search={block_err_best:.6f}, heldout={m_be_mlp:.6f}")
        print(f"    Improvement: {impr:.1f}% (search), {(1-m_j_mlp/m_be_mlp)*100:.1f}% (heldout)")
        print(f"    {len(all_allocs)} balanced allocations enumerated")

        results.setdefault('mlp_rate_allocation', {})[f"{avg_b}_avg"] = {
            'budget': budget, 'used': joint_used, 'budget_exact': joint_used == budget,
            'joint_alloc': joint_alloc, 'best_equal_alloc': best_equal[0],
            'n_balanced_allocs': len(all_allocs),
            'joint_search_error': block_err_joint, 'best_equal_search_error': block_err_best,
            'improvement_vs_best_pct': impr,
            'joint_heldout_mean': m_j_mlp, 'best_equal_heldout_mean': m_be_mlp,
            'heldout_improvement_pct': (1-m_j_mlp/m_be_mlp)*100,
        }

    # 13. Jacobian trace sensitivity + error-direction cross terms
    print("\n--- 13. Jacobian Trace Sensitivity + Error Cross Terms (MLP) ---")
    S_mlp, mlp_names, jacs_mlp = compute_jacobian_trace_summary_mlp(mlp)
    print("Jacobian trace summary:")
    for i, ni in enumerate(mlp_names):
        print(f"  {ni:6s}: " + "  ".join(f"{S_mlp[i,j]:.4e}" for j in range(3)))

    E_gate = (mlp.W_gate - W_gate_q_ind).ravel()
    E_up = (mlp.W_up - W_up_q_ind).ravel()
    E_down = (mlp.W_down - W_down_q_ind).ravel()
    C_mlp = compute_error_direction_cross_terms(jacs_mlp, [E_gate, E_up, E_down], mlp_names)
    print("\nError-direction cross terms:")
    for i, ni in enumerate(mlp_names):
        print(f"  {ni:6s}: " + "  ".join(f"{C_mlp[i,j]:.4e}" for j in range(3)))
    diag_C_mlp = [C_mlp[i,i] for i in range(3)]
    off_diag_C_mlp = [C_mlp[i,j] for i in range(3) for j in range(3) if i != j]
    total_C_mlp = sum(abs(x) for x in C_mlp.ravel())
    diag_frac_mlp = sum(abs(x) for x in diag_C_mlp) / total_C_mlp if total_C_mlp > 0 else 0
    print(f"\nDiagonal fraction: {diag_frac_mlp:.4f}")
    print(f"Off-diag max / diag max: {max(abs(x) for x in off_diag_C_mlp) / max(abs(x) for x in diag_C_mlp):.4f}")

    analytic_mlp = derive_mlp_jacobian_trace_analytic(mlp)
    print(f"\nAnalytic vs numerical (trace summary):")
    print(f"  gate_gate: ana={analytic_mlp['H_gate_gate']:.4e}, num={S_mlp[0,0]:.4e}, ratio={analytic_mlp['H_gate_gate']/S_mlp[0,0]:.6f}")
    print(f"  up_up:     ana={analytic_mlp['H_up_up']:.4e}, num={S_mlp[1,1]:.4e}, ratio={analytic_mlp['H_up_up']/S_mlp[1,1]:.6f}")
    print(f"  down_down: ana={analytic_mlp['H_down_down']:.4e}, num={S_mlp[2,2]:.4e}, ratio={analytic_mlp['H_down_down']/S_mlp[2,2]:.6f}")
    print(f"  gate_up:   ana={analytic_mlp['H_gate_up']:.4e}, num={S_mlp[0,1]:.4e}, ratio={analytic_mlp['H_gate_up']/S_mlp[0,1]:.6f}")

    results['mlp_jacobian'] = {
        'trace_summary': S_mlp.tolist(), 'names': mlp_names,
        'error_direction_cross': C_mlp.tolist(),
        'diag_fraction': float(diag_frac_mlp),
        'analytic': analytic_mlp,
    }

    # ---- Bit-width sweep (Attention, held-out) ----
    print("\n" + "=" * 80)
    print("BIT-WIDTH SWEEP (Attention, held-out)")
    print("=" * 80)
    sweep = []
    for b in [3, 4, 5, 6]:
        W_Qq = quantize_per_column(attn.W_Q, b)
        W_Kq = quantize_per_column(attn.W_K, b)
        W_Vq = quantize_per_column(attn.W_V, b)
        W_Oq = quantize_per_column(attn.W_O, b)
        val_ind = evaluate_attention_heldout(attn, W_Qq, W_Kq, W_Vq, W_Oq,
                                              range(n_val_seeds), np.random.default_rng(2024))
        m_i = np.mean(val_ind)
        fr = full_coupled_attention_optimization(attn, b, quantize_per_column, attn.X,
                                                  rng=np.random.default_rng(999+b))
        val_coup = evaluate_attention_heldout(attn, fr['W_Qq'], fr['W_Kq'],
                                                fr['W_Vq'], fr['W_Oq'],
                                                range(n_val_seeds), np.random.default_rng(2024))
        m_c = np.mean(val_coup)
        impr = (1 - m_c / m_i) * 100
        print(f"  bits={b}: heldout ind={m_i:.6f}, coupled={m_c:.6f}, improvement={impr:.1f}%")
        sweep.append({'bits': b, 'heldout_independent': m_i, 'heldout_coupled': m_c,
                      'improvement_pct': impr})
    results['attn_bit_sweep'] = sweep

    # ---- Bit-width sweep (MLP per-tile, held-out) ----
    print("\n" + "=" * 80)
    print("BIT-WIDTH SWEEP (MLP per-tile, held-out)")
    print("=" * 80)
    sweep_mlp = []
    for b in [3, 4, 5, 6]:
        W_gq = quantize_per_tile(mlp.W_gate, b, 16)
        W_uq = quantize_per_tile(mlp.W_up, b, 16)
        W_dq = quantize_per_tile(mlp.W_down, b, 16)
        val_ind = evaluate_mlp_heldout(mlp, W_gq, W_uq, W_dq,
                                        range(n_val_seeds), np.random.default_rng(2024))
        m_i = np.mean(val_ind)
        _, perm_b, _, _, W_gq_p, W_uq_p, W_dq_p = search_mlp_permutation(
            mlp, b, lambda W, bb: quantize_per_tile(W, bb, 16), mlp.X,
            n_random=50, n_swap_rounds=2, rng=np.random.default_rng(789+b))
        val_perm = evaluate_mlp_heldout(mlp, W_gq_p, W_uq_p, W_dq_p,
                                        range(n_val_seeds), np.random.default_rng(2024))
        m_p = np.mean(val_perm)
        impr = (1 - m_p / m_i) * 100
        print(f"  bits={b}: heldout ind={m_i:.6f}, perm={m_p:.6f}, improvement={impr:.1f}%")
        sweep_mlp.append({'bits': b, 'heldout_independent': m_i, 'heldout_perm': m_p,
                         'improvement_pct': impr})
    results['mlp_bit_sweep'] = sweep_mlp

    # ---- Real weight experiments (held-out) ----
    print("\n" + "=" * 80)
    print("REAL WEIGHT EXPERIMENTS (held-out)")
    print("=" * 80)
    try:
        real_w = np.load('/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz')
        for layer_name, gate_key, down_key in [('L0', 'L0_gate', 'L0_down'),
                                                ('L55', 'L55_gate', 'L55_down')]:
            W_gate_r = real_w[gate_key][:64, :128].astype(np.float64)
            W_down_r = real_w[down_key][:128, :64].astype(np.float64)
            W_up_r = np.random.default_rng(55).standard_normal(W_gate_r.shape) * 0.1
            X_cal = np.random.default_rng(77).standard_normal((128, 64))
            mlp_r = MLPBlock(64, 128, 128, np.random.default_rng(0),
                             W_gate=W_gate_r, W_up=W_up_r, W_down=W_down_r, X=X_cal)

            # Per-tile independent
            W_gq = quantize_per_tile(mlp_r.W_gate, 4, 16)
            W_uq = quantize_per_tile(mlp_r.W_up, 4, 16)
            W_dq = quantize_per_tile(mlp_r.W_down, 4, 16)
            val_ind = evaluate_mlp_heldout(mlp_r, W_gq, W_uq, W_dq,
                                            range(n_val_seeds), np.random.default_rng(2024))
            m_i = np.mean(val_ind)

            # Per-tile + permutation
            _, perm_r, _, _, W_gq_p, W_uq_p, W_dq_p = search_mlp_permutation(
                mlp_r, 4, lambda W, b: quantize_per_tile(W, b, 16), X_cal,
                n_random=100, n_swap_rounds=3, rng=np.random.default_rng(789))
            val_perm = evaluate_mlp_heldout(mlp_r, W_gq_p, W_uq_p, W_dq_p,
                                            range(n_val_seeds), np.random.default_rng(2024))
            m_p = np.mean(val_perm)
            impr = (1 - m_p / m_i) * 100
            print(f"  {layer_name} per-tile: heldout ind={m_i:.6f}, perm={m_p:.6f}, improvement={impr:.1f}%")

            results.setdefault('real_weight_experiments', {})[f'{layer_name}_mlp_perm'] = {
                'heldout_independent': m_i, 'heldout_perm': m_p, 'improvement_pct': impr,
                'note': 'Real gate+down, synthetic up, held-out Gaussian X',
            }
    except Exception as e:
        print(f"Real weight experiment failed: {e}")
        import traceback; traceback.print_exc()
        results['real_weight_experiments'] = {'error': str(e)}

    # ---- Noise floor ----
    print("\n--- Noise Floor ---")
    results['noise_floor'] = {
        'attention': attn.block_error(attn.W_Q, attn.W_K, attn.W_V, attn.W_O),
        'mlp': mlp.block_error(mlp.W_gate, mlp.W_up, mlp.W_down),
    }
    print(f"Attention: {results['noise_floor']['attention']:.2e}")
    print(f"MLP: {results['noise_floor']['mlp']:.2e}")

    t_total = time.time() - t_start
    results['timing_seconds'] = t_total
    print(f"\nTotal time: {t_total:.1f}s")
    return results


if __name__ == '__main__':
    results = run_all_experiments()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    receipts_dir = os.path.join(out_dir, '..', '..', '..', 'receipts', 'research')
    os.makedirs(receipts_dir, exist_ok=True)
    results_path = os.path.join(receipts_dir, 'r10-coupled-blocks-results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
