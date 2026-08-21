#!/usr/bin/env python3
"""
R5-Subspace: ResQ + Subspace Quantization Proof-of-Concept (v2 -- REVIEWER-FIXED)
===============================================================================
Clean-room from paper equations (arXiv:2412.14363, Eq 3).

Fixes from v1 (reviewer-identified bugs):
1. FIXED inv_cholesky: returns upper-triangular (was lower, GPTQ propagation was zero)
2. FIXED GPTAQ: labeled as GPTQ (P-matrix needs activation quant, not available)
3. FIXED quantizer matching: ALL arms use per-tile (16x16) with Hadamard
4. FIXED block_hadamard: only applies to full tile blocks (no broken crop)
5. FIXED tile PCA byte counting: one U_tile per INPUT tile (not per output tile)
6. FIXED adaptive rank: enforces byte budget constraint
7. FIXED budget-matched comparison: includes relevant K' values
8. Added sanity check: GPTQ vs independent quantization must differ
"""

import numpy as np, json, warnings
from dataclasses import dataclass, field
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class Config:
    m: int = 128; n: int = 128; k: int = 512
    tile: int = 16; block: int = 16; damping: float = 1e-6
    bits_list: list = field(default_factory=lambda: [3, 4, 5, 6])
    seeds: list = field(default_factory=lambda: [42, 123, 777])
    high_bits: int = 8; proj_bytes_per_elem: int = 2

# ==================== Utilities ====================

def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def block_hadamard(W, tile):
    """Apply Hadamard to full tile-sized column blocks only. Partial blocks: identity."""
    m, n = W.shape
    r = W.copy().astype(np.float64)
    H = hadamard(tile)
    for i in range(0, n, tile):
        end = min(i + tile, n)
        if end - i == tile:
            r[:, i:end] = W[:, i:end] @ H
    return r

def inv_block_hadamard(W, tile):
    return block_hadamard(W, tile)

def softmax(x, axis=0):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# ==================== Quantizer (ALL arms use this matched quantizer) ====================

def quantize_uniform(w, bits):
    if bits >= 16:
        return w.astype(np.float16).astype(np.float64)  # FP16, not exact FP64
    nl = 2 ** bits
    mx = float(np.max(np.abs(w)))
    if mx < 1e-30:
        return w.copy()
    s = mx / (nl / 2 - 1)
    return np.clip(np.round(w / s), -nl // 2 + 1, nl // 2 - 1) * s

def trellis_quantize(W, bits, tile):
    """Per-tile (tile x tile) uniform quantization. MATCHED for all arms."""
    m, n = W.shape
    Wq = np.zeros_like(W, dtype=np.float64)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            ie, je = min(i + tile, m), min(j + tile, n)
            Wq[i:ie, j:je] = quantize_uniform(W[i:ie, j:je], bits)
    return Wq

def quantize_matched(W, bits, tile):
    """Hadamard + per-tile quantize + inverse Hadamard. MATCHED quantizer for ALL arms."""
    WH = block_hadamard(W, tile)
    WqH = trellis_quantize(WH, bits, tile)
    return inv_block_hadamard(WqH, tile)
def quantize_matched_col(Wc, bits, tile):
    """Per-tile quantization for a single column (16-element segments).
    When GPTQ operates on Hadamard-transformed weights, each column segment
    is quantized with one scale per 16 elements, matching the 16x16 tile scale
    that the baseline uses for the same Hadamard-transformed region."""
    m = Wc.shape[0]
    Wq = np.zeros_like(Wc, dtype=np.float64)
    for i in range(0, m, tile):
        ie = min(i + tile, m)
        Wq[i:ie, :] = quantize_uniform(Wc[i:ie, :], bits)
    return Wq
# ==================== Loss ====================

def wt_mse(W, Wq): return float(np.mean((W - Wq) ** 2))

def hessian_weighted_error(W, Wq, H_X):
    E = W - Wq
    return float(np.trace(E @ H_X @ E.T) / W.shape[0])

def output_mse(W, Wq, X):
    return float(np.mean((W @ X.T - Wq @ X.T) ** 2))

def proxy_kld(W, Wq, X):
    y_fp = W @ X.T; y_q = Wq @ X.T
    p = np.clip(softmax(y_fp, 0), 1e-12, 1)
    q = np.clip(softmax(y_q, 0), 1e-12, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=0)))

# ==================== Byte accounting ====================

def bytes_for_quantized(m, n, bits):
    return m * n * bits / 8.0

def bytes_for_projection(d, bytes_per_elem=2):
    return d * d * bytes_per_elem

def bytes_for_resq(m, n, r, high_bits, low_bits, bytes_per_elem=2):
    coeff = m * r * high_bits / 8.0 + m * (n - r) * low_bits / 8.0
    proj = bytes_for_projection(n, bytes_per_elem)
    return coeff + proj

# ==================== PCA subspace construction ====================

def activation_pca(X):
    H_X = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H_X)
    return eigvecs[:, np.argsort(eigvals)[::-1]]

def weight_pca(W):
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    return Vt.T

def joint_pca(X, W):
    """Joint PCA: generalized eigenvalue problem H_X u = lambda H_W u.
    Generalized eigenvectors are H_W-orthogonal, NOT Euclidean-orthogonal.
    QR-orthonormalize for ResQ (loses exact optimality but preserves approximate ordering)."""
    d = X.shape[1]
    H_X = X.T @ X / X.shape[0]
    H_W = W.T @ W / W.shape[0]
    from scipy.linalg import eigh as gen_eigh
    eigvals, eigvecs = gen_eigh(H_X, H_W + 1e-10 * np.eye(d))
    idx = np.argsort(eigvals)[::-1]
    U, _ = np.linalg.qr(eigvecs[:, idx])
    return U

# ==================== GPTQ with CORRECT upper-triangular Cholesky ====================

def inv_cholesky_upper(H, damping):
    """Upper-triangular Cholesky factor of inv(H + damping*I).
    Returns U (upper triangular) such that U.T @ U = inv(H + damping*I).
    CORRECT convention per GPTQv2 reference: chol(inv(H)).T gives U^T U = inv(H).
    v1 returned lower-triangular (zero propagation). v2 returned U U^T = inv(H) (wrong orientation).
    This v3 uses chol(inv(H+lam)).T for correct U.T U = inv(H)."""
    n = H.shape[0]
    lam = max(damping * np.mean(np.diag(H)), 1e-10)
    H_reg = H + lam * np.eye(n)
    H_inv = np.linalg.inv(H_reg)
    return np.linalg.cholesky(H_inv).T  # upper triangular, U.T @ U = inv(H)

def gptq_quantize(W_sub, X_sub, bits, tile, block, damping, gptaq=False):
    """GPTQ column processing with CORRECT Cholesky and MATCHED quantizer.
    FIX: Uses one shared scale per 16x16 tile (matching baseline trellis_quantize).
    Within each tile block, all 16 columns share the same scale computed from
    the full 16x16 tile region. This eliminates the quantizer granularity mismatch."""
    m, n_sub = W_sub.shape
    Ww = W_sub.copy().astype(np.float64)
    Q = np.zeros_like(Ww)
    
    H = X_sub.T @ X_sub
    L = inv_cholesky_upper(H, damping)
    
    for i in range(0, n_sub, block):
        B = min(block, n_sub - i)
        E = np.zeros((m, B))
        for j in range(B):
            c = i + j
            w_pre = Ww[:, c].copy()
            # Use shared tile scale: compute scale from the full tile column block
            # For each 16-row segment, use the max abs across ALL columns in this block
            # This matches trellis_quantize which uses one scale per 16x16 tile
            if j == 0:
                # Precompute scales for each row-segment in this column block
                tile_scales = []
                for ri in range(0, m, tile):
                    re = min(ri + tile, m)
                    tile_block = Ww[ri:re, i:i+B]
                    mx = float(np.max(np.abs(tile_block)))
                    if mx < 1e-30: mx = 1e-30
                    if bits >= 16:
                        tile_scales.append(None)  # FP16
                    else:
                        nl = 2 ** bits
                        tile_scales.append(mx / (nl / 2 - 1))
            
            # Quantize column c using the shared tile scale
            if bits >= 16:
                Q[:, c] = w_pre.astype(np.float16).astype(np.float64)
            else:
                for ri in range(0, m, tile):
                    re = min(ri + tile, m)
                    s = tile_scales[ri // tile]
                    nl = 2 ** bits
                    Q[ri:re, c] = np.clip(np.round(w_pre[ri:re] / s), -nl//2+1, nl//2-1) * s
            
            e = w_pre - Q[:, c]
            E[:, j] = e / L[c, c]
            end = min(i + B, n_sub)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
        if i + B < n_sub:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
    return Q
# ==================== ResQ variants (ALL use matched quantizer) ====================

def resq_activation(W, X, bits, high_bits=8, ratio=0.125, tile=16):
    """ResQ with activation PCA (correct paper algorithm, Eq 3)."""
    m, n = W.shape
    r = max(1, int(n * ratio))
    U = activation_pca(X)
    U_h, U_l = U[:, :r], U[:, r:]
    W_h, W_l = W @ U_h, W @ U_l
    Wq_h = quantize_matched(W_h, high_bits, tile)
    Wq_l = quantize_matched(W_l, bits, tile)
    Wq = Wq_h @ U_h.T + Wq_l @ U_l.T
    total_bytes = bytes_for_resq(m, n, r, high_bits, bits)
    return Wq, total_bytes, {'r': r, 'proj_bytes': bytes_for_projection(n),
                             'coeff_bytes': total_bytes - bytes_for_projection(n)}

def resq_weight(W, X, bits, high_bits=8, ratio=0.125, tile=16):
    m, n = W.shape
    r = max(1, int(n * ratio))
    U = weight_pca(W)
    U_h, U_l = U[:, :r], U[:, r:]
    W_h, W_l = W @ U_h, W @ U_l
    Wq_h = quantize_matched(W_h, high_bits, tile)
    Wq_l = quantize_matched(W_l, bits, tile)
    Wq = Wq_h @ U_h.T + Wq_l @ U_l.T
    total_bytes = bytes_for_resq(m, n, r, high_bits, bits)
    return Wq, total_bytes, {'r': r, 'proj_bytes': bytes_for_projection(n),
                             'coeff_bytes': total_bytes - bytes_for_projection(n)}

def resq_joint(W, X, bits, high_bits=8, ratio=0.125, tile=16):
    m, n = W.shape
    r = max(1, int(n * ratio))
    U = joint_pca(X, W)
    U_h, U_l = U[:, :r], U[:, r:]
    W_h, W_l = W @ U_h, W @ U_l
    Wq_h = quantize_matched(W_h, high_bits, tile)
    Wq_l = quantize_matched(W_l, bits, tile)
    Wq = Wq_h @ U_h.T + Wq_l @ U_l.T
    total_bytes = bytes_for_resq(m, n, r, high_bits, bits)
    return Wq, total_bytes, {'r': r, 'proj_bytes': bytes_for_projection(n),
                             'coeff_bytes': total_bytes - bytes_for_projection(n)}

def resq_tile_pca(W, X, bits, high_bits=8, ratio=0.125, tile=16):
    """Trellis-tile PCA: PCA within each tile x tile block.
    FIX: projection counted once per INPUT tile (was 8x overcounted)."""
    m, n = W.shape
    r_tile = max(1, int(tile * ratio))
    Wq = np.zeros_like(W, dtype=np.float64)
    total_proj_bytes = 0
    total_coeff_bytes = 0
    
    for j in range(0, n, tile):
        je = min(j + tile, n)
        tile_w = je - j
        if tile_w < tile:
            for i in range(0, m, tile):
                ie = min(i + tile, m)
                Wq[i:ie, j:je] = quantize_uniform(W[i:ie, j:je], bits)
                total_coeff_bytes += (ie - i) * tile_w * bits / 8.0
            continue
        
        X_tile = X[:, j:je]
        H_tile = X_tile.T @ X_tile / max(X_tile.shape[0], 1)
        eigvals, eigvecs = np.linalg.eigh(H_tile)
        U_tile = eigvecs[:, np.argsort(eigvals)[::-1]]
        r_t = min(r_tile, tile_w)
        U_h, U_l = U_tile[:, :r_t], U_tile[:, r_t:]
        
        for i in range(0, m, tile):
            ie = min(i + tile, m)
            tile_h = ie - i
            W_tile = W[i:ie, j:je].copy()
            Wq_h = quantize_uniform(W_tile @ U_h, high_bits)
            Wq_l = quantize_uniform(W_tile @ U_l, bits)
            Wq[i:ie, j:je] = Wq_h @ U_h.T + Wq_l @ U_l.T
            total_coeff_bytes += tile_h * r_t * high_bits / 8.0 + tile_h * (tile_w - r_t) * bits / 8.0
        
        total_proj_bytes += tile_w * tile_w * 2  # ONE projection per input tile
    
    total_bytes = total_coeff_bytes + total_proj_bytes
    return Wq, total_bytes, {'r_tile': r_tile, 'proj_bytes': total_proj_bytes,
                              'coeff_bytes': total_coeff_bytes}

def resq_gptq(W, X, bits, high_bits=8, ratio=0.125, tile=16, block=16, damping=1e-6):
    """ResQ + GPTQ per-subspace. 
    PCA projects into subspaces, then Hadamard is applied within each subspace
    to create non-diagonal Hessian (GPTQ needs off-diagonal structure to propagate).
    GPTQ uses matched Hadamard quantizer (same as baseline).
    Flow: PCA -> Hadamard -> GPTQ (matched quant) -> inv Hadamard -> inv PCA."""
    m, n = W.shape
    r = max(1, int(n * ratio))
    U = activation_pca(X)
    U_h, U_l = U[:, :r], U[:, r:]
    W_h, W_l = W @ U_h, W @ U_l
    X_h, X_l = X @ U_h, X @ U_l
    
    # Apply Hadamard within each subspace (creates non-diagonal Hessian for GPTQ)
    WH_h = block_hadamard(W_h, tile) if r >= tile else W_h
    WH_l = block_hadamard(W_l, tile) if (n - r) >= tile else W_l
    XH_h = block_hadamard(X_h, tile) if r >= tile else X_h
    XH_l = block_hadamard(X_l, tile) if (n - r) >= tile else X_l
    
    # GPTQ on Hadamard-transformed weights (Hessian is NOT diagonal now)
    WqH_h = gptq_quantize(WH_h, XH_h, high_bits, tile, block, damping)
    WqH_l = gptq_quantize(WH_l, XH_l, bits, tile, block, damping)
    
    # Inverse Hadamard
    Wq_h = inv_block_hadamard(WqH_h, tile) if r >= tile else WqH_h
    Wq_l = inv_block_hadamard(WqH_l, tile) if (n - r) >= tile else WqH_l
    
    Wq = Wq_h @ U_h.T + Wq_l @ U_l.T
    total_bytes = bytes_for_resq(m, n, r, high_bits, bits)
    return Wq, total_bytes, {'r': r, 'proj_bytes': bytes_for_projection(n),
                             'coeff_bytes': total_bytes - bytes_for_projection(n)}
def baq_allocate(W_sub, X_sub, avg_bits, tile, min_k=3, max_k=7):
    m, n_sub = W_sub.shape
    H = X_sub.T @ X_sub + 1e-8 * np.eye(n_sub)
    H_inv_diag = 1.0 / np.diag(H)
    w_range = W_sub.max(axis=0) - W_sub.min(axis=0)
    c = w_range ** 2 / (12.0 * H_inv_diag + 1e-30)
    c = np.clip(c, 1e-30, None)
    lam = np.exp(np.mean(np.log(c)))
    R_star = 0.5 * np.log2(c / lam) + avg_bits
    bits = np.clip(np.round(R_star), min_k, max_k).astype(int)
    target = avg_bits * n_sub
    while bits.sum() != target:
        if bits.sum() < target:
            cands = np.where(bits < max_k)[0]
            if len(cands) == 0: break
            bits[cands[np.argmax(c[cands])]] += 1
        else:
            cands = np.where(bits > min_k)[0]
            if len(cands) == 0: break
            bits[cands[np.argmin(c[cands])]] -= 1
    return bits

def gptq_quantize_baq(W_sub, X_sub, baq_bits, tile, block, damping):
    """GPTQ with per-column bit allocation (BAQ). Shared tile scale per 16x16 block."""
    m, n_sub = W_sub.shape
    Ww = W_sub.copy().astype(np.float64)
    Q = np.zeros_like(Ww)
    H = X_sub.T @ X_sub
    L = inv_cholesky_upper(H, damping)
    for i in range(0, n_sub, block):
        B = min(block, n_sub - i)
        E = np.zeros((m, B))
        # Precompute shared tile scales for this block (max bits in block for scale)
        max_bits_block = int(max(baq_bits[i:i+B]))
        tile_scales_baq = []
        for ri in range(0, m, tile):
            re = min(ri + tile, m)
            mx = float(np.max(np.abs(Ww[ri:re, i:i+B])))
            if mx < 1e-30: mx = 1e-30
            tile_scales_baq.append(mx)
        for j in range(B):
            c = i + j
            w_pre = Ww[:, c].copy()
            col_k = int(baq_bits[c])
            for ri in range(0, m, tile):
                re = min(ri + tile, m)
                mx = tile_scales_baq[ri // tile]
                if col_k >= 16:
                    Q[ri:re, c] = w_pre[ri:re].astype(np.float16).astype(np.float64)
                else:
                    nl = 2 ** col_k; s = mx / (nl / 2 - 1)
                    Q[ri:re, c] = np.clip(np.round(w_pre[ri:re] / s), -nl//2+1, nl//2-1) * s
            e = w_pre - Q[:, c]
            E[:, j] = e / L[c, c]
            end = min(i + B, n_sub)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
        if i + B < n_sub:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
    return Q

def resq_baq(W, X, bits, high_bits=8, ratio=0.125, tile=16, block=16, damping=1e-6):
    """ResQ + BAQ: PCA subspaces, Hadamard within low-prec subspace, GPTQ+BAQ."""
    m, n = W.shape
    r = max(1, int(n * ratio))
    U = activation_pca(X)
    U_h, U_l = U[:, :r], U[:, r:]
    W_h, W_l = W @ U_h, W @ U_l
    X_h, X_l = X @ U_h, X @ U_l
    
    # Hadamard within low-precision subspace (creates non-diagonal Hessian for GPTQ)
    WH_l = block_hadamard(W_l, tile) if (n - r) >= tile else W_l
    XH_l = block_hadamard(X_l, tile) if (n - r) >= tile else X_l
    
    Wq_h = quantize_matched(W_h, high_bits, tile)
    baq_bits = baq_allocate(WH_l, XH_l, bits, tile, max(3, bits-1), min(7, bits+1))
    WqH_l = gptq_quantize_baq(WH_l, XH_l, baq_bits, tile, block, damping)
    Wq_l = inv_block_hadamard(WqH_l, tile) if (n - r) >= tile else WqH_l
    
    Wq = Wq_h @ U_h.T + Wq_l @ U_l.T
    coeff_bytes = m * r * high_bits / 8.0 + sum(m * baq_bits[c] / 8.0 for c in range(W_l.shape[1]))
    proj_bytes = bytes_for_projection(n)
    return Wq, coeff_bytes + proj_bytes, {'r': r, 'proj_bytes': proj_bytes,
                                          'coeff_bytes': coeff_bytes, 'baq_bits': baq_bits.tolist()}
# ==================== Adaptive rank (FIXED: enforces byte budget) ====================

def adaptive_rank(W, X, bits, high_bits=8, tile=16, max_r=None):
    m, n = W.shape
    if max_r is None: max_r = n // 4
    default_r = max(1, n // 8)
    target_bytes = bytes_for_resq(m, n, default_r, high_bits, bits)
    H_X = X.T @ X / X.shape[0]
    best_r, best_err, best_Wq = 0, float('inf'), None
    results = []
    for r in range(1, min(max_r + 1, n)):
        total_bytes = bytes_for_resq(m, n, r, high_bits, bits)
        if total_bytes > target_bytes:  # FIX: enforce budget
            continue
        Wq, _, _ = resq_activation(W, X, bits, high_bits, r / n, tile)
        err = hessian_weighted_error(W, Wq, H_X)
        results.append({'r': r, 'error': err, 'bytes': total_bytes})
        if err < best_err:
            best_err, best_r, best_Wq = err, r, Wq
    if best_Wq is None:
        best_r = default_r
        best_Wq, _, _ = resq_activation(W, X, bits, high_bits, default_r / n, tile)
    total_bytes = bytes_for_resq(m, n, best_r, high_bits, bits)
    return best_Wq, total_bytes, {'r': best_r, 'all_r': results, 'target_bytes': target_bytes}

# ==================== Noise floor ====================

def noise_floor(W, X):
    W_f64 = W.astype(np.float64)
    W_f32 = W.astype(np.float32).astype(np.float64)
    H_X = X.T @ X / X.shape[0]
    return {
        'wt_mse_floor': wt_mse(W_f64, W_f32),
        'hess_err_floor': hessian_weighted_error(W_f64, W_f32, H_X),
        'output_mse_floor': output_mse(W_f64, W_f32, X),
        'proxy_kld_floor': proxy_kld(W_f64, W_f32, X),
    }

# ==================== Data ====================

def gen_tensors(cfg, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((cfg.m, cfg.n)) * 0.1
    X = rng.standard_normal((cfg.k, cfg.n)) * 0.5
    n_out = max(1, cfg.n // 20)
    idx = rng.choice(cfg.n, n_out, replace=False)
    X[:, idx] *= rng.uniform(5, 20, size=(cfg.k, n_out))
    return W.astype(np.float64), X.astype(np.float64)

def load_real_weights(tensor_name, slice_size=128):
    data = np.load('/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz')
    W_full = data[tensor_name].astype(np.float64)
    m, n = W_full.shape
    W = W_full[:min(slice_size, m), :min(slice_size, n)]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((512, W.shape[1])) * 0.5
    n_out = max(1, W.shape[1] // 20)
    idx = rng.choice(W.shape[1], n_out, replace=False)
    X[:, idx] *= rng.uniform(5, 20, size=(512, n_out))
    return W, X.astype(np.float64)

# ==================== Sanity check ====================

def sanity_check_gptq(cfg):
    """Verify GPTQ error propagation is non-zero (v1 had zero propagation).
    FIX: Compare propagation ON vs OFF with IDENTICAL quantizer (shared tile scale).
    Also add diagonal-Hessian negative control (propagation must be zero)."""
    W, X = gen_tensors(cfg, 42)
    H_X = X.T @ X / X.shape[0]
    
    # GPTQ with propagation (correct Cholesky)
    Wq_gptq = gptq_quantize(W, X, 4, cfg.tile, cfg.block, cfg.damping)
    
    # GPTQ with propagation DISABLED (diagonal Cholesky = identity/abs)
    H = X.T @ X
    L_diag = np.diag(np.diag(inv_cholesky_upper(H, cfg.damping)))  # diagonal only
    m, n = W.shape; Ww = W.copy(); Q_noprop = np.zeros_like(Ww)
    for i in range(0, n, cfg.block):
        B = min(cfg.block, n - i); E = np.zeros((m, B))
        for j in range(B):
            c = i + j; w_pre = Ww[:, c].copy()
            # Same shared tile scale as gptq_quantize
            if j == 0:
                tile_scales = []
                for ri in range(0, m, cfg.tile):
                    re = min(ri + cfg.tile, m)
                    mx = float(np.max(np.abs(Ww[ri:re, i:i+B])))
                    if mx < 1e-30: mx = 1e-30
                    nl = 2 ** 4; tile_scales.append(mx / (nl / 2 - 1))
            for ri in range(0, m, cfg.tile):
                re = min(ri + cfg.tile, m); s = tile_scales[ri // cfg.tile]; nl = 16
                Q_noprop[ri:re, c] = np.clip(np.round(w_pre[ri:re] / s), -nl//2+1, nl//2-1) * s
            e = w_pre - Q_noprop[:, c]; E[:, j] = e / L_diag[c, c]
            end = min(i + B, n); Ww[:, c:end] -= np.outer(E[:, j], L_diag[c, c:end])
        if i + B < n: Ww[:, i+B:] -= E @ L_diag[i:i+B, i+B:]
    
    diff = np.max(np.abs(Wq_gptq - Q_noprop))
    err_gptq = hessian_weighted_error(W, Wq_gptq, H_X)
    err_noprop = hessian_weighted_error(W, Q_noprop, H_X)
    print(f"Sanity check: GPTQ propagation ON vs OFF (matched quantizer)")
    print(f"  Max diff: {diff:.6e} (should be > 0)")
    print(f"  Propagation OFF hess_err: {err_noprop:.6e}")
    print(f"  Propagation ON hess_err: {err_gptq:.6e}")
    print(f"  GPTQ propagation improvement: {(1 - err_gptq/err_noprop)*100:.1f}%")
    assert diff > 1e-10, "FAIL: GPTQ propagation is zero!"
    assert err_gptq < err_noprop, "FAIL: GPTQ propagation makes things worse!"
    print(f"  PASS: GPTQ propagation is working and improving\n")
# ==================== Experiment runner ====================

def run_all_methods(W, X, bits, cfg):
    m, n = W.shape
    H_X = X.T @ X / X.shape[0]
    results = []
    
    Wq_base = quantize_matched(W, bits, cfg.tile)
    results.append({'method': 'baseline_K', 'wt_mse': wt_mse(W, Wq_base),
        'hess_err': hessian_weighted_error(W, Wq_base, H_X),
        'out_mse': output_mse(W, Wq_base, X), 'proxy_kld': proxy_kld(W, Wq_base, X),
        'bytes': bytes_for_quantized(m, n, bits), 'proj_bytes': 0})
    
    Wq_kp1 = quantize_matched(W, bits + 1, cfg.tile)
    results.append({'method': f'baseline_K{bits+1}', 'wt_mse': wt_mse(W, Wq_kp1),
        'hess_err': hessian_weighted_error(W, Wq_kp1, H_X),
        'out_mse': output_mse(W, Wq_kp1, X), 'proxy_kld': proxy_kld(W, Wq_kp1, X),
        'bytes': bytes_for_quantized(m, n, bits + 1), 'proj_bytes': 0})
    
    for ratio in [0.0625, 0.125, 0.25]:
        for name, func in [('resq_act_pca', resq_activation), ('resq_wt_pca', resq_weight),
                           ('resq_joint_pca', resq_joint), ('resq_tile_pca', resq_tile_pca)]:
            Wq, total_bytes, info = func(W, X, bits, cfg.high_bits, ratio, cfg.tile)
            results.append({'method': f'{name}_r{ratio}', 'wt_mse': wt_mse(W, Wq),
                'hess_err': hessian_weighted_error(W, Wq, H_X),
                'out_mse': output_mse(W, Wq, X), 'proxy_kld': proxy_kld(W, Wq, X),
                'bytes': total_bytes, 'proj_bytes': info['proj_bytes'],
                'r': info.get('r', info.get('r_tile'))})
    
    Wq, total_bytes, info = resq_gptq(W, X, bits, cfg.high_bits, 0.125, cfg.tile, cfg.block, cfg.damping)
    results.append({'method': 'resq_gptq_r0.125', 'wt_mse': wt_mse(W, Wq),
        'hess_err': hessian_weighted_error(W, Wq, H_X),
        'out_mse': output_mse(W, Wq, X), 'proxy_kld': proxy_kld(W, Wq, X),
        'bytes': total_bytes, 'proj_bytes': info['proj_bytes'], 'r': info['r']})
    
    Wq, total_bytes, info = resq_baq(W, X, bits, cfg.high_bits, 0.125, cfg.tile, cfg.block, cfg.damping)
    results.append({'method': 'resq_baq_r0.125', 'wt_mse': wt_mse(W, Wq),
        'hess_err': hessian_weighted_error(W, Wq, H_X),
        'out_mse': output_mse(W, Wq, X), 'proxy_kld': proxy_kld(W, Wq, X),
        'bytes': total_bytes, 'proj_bytes': info['proj_bytes'], 'r': info['r']})
    
    Wq, total_bytes, info = adaptive_rank(W, X, bits, cfg.high_bits, cfg.tile)
    results.append({'method': 'resq_adaptive_rank', 'wt_mse': wt_mse(W, Wq),
        'hess_err': hessian_weighted_error(W, Wq, H_X),
        'out_mse': output_mse(W, Wq, X), 'proxy_kld': proxy_kld(W, Wq, X),
        'bytes': total_bytes, 'proj_bytes': bytes_for_projection(n),
        'r': info['r'], 'all_r': info.get('all_r', [])})
    
    # Budget-matched: uniform K' with same bytes as ResQ (FIX: no restrictive filter)
    resq_bytes = bytes_for_resq(m, n, max(1, n//8), cfg.high_bits, bits)
    equiv_bits = resq_bytes * 8.0 / (m * n)
    for k_eq in sorted(set([int(np.floor(equiv_bits)), int(np.ceil(equiv_bits))])):
        if k_eq >= bits:  # FIX: only filter K' < K (irrelevant)
            Wq_eq = quantize_matched(W, k_eq, cfg.tile)
            results.append({'method': f'baseline_K{k_eq}_eq', 'wt_mse': wt_mse(W, Wq_eq),
                'hess_err': hessian_weighted_error(W, Wq_eq, H_X),
                'out_mse': output_mse(W, Wq_eq, X), 'proxy_kld': proxy_kld(W, Wq_eq, X),
                'bytes': bytes_for_quantized(m, n, k_eq), 'proj_bytes': 0})
    
    return results

# ==================== Main ====================

def main():
    cfg = Config()
    all_results = {}
    
    sanity_check_gptq(cfg)
    
    print("=" * 80)
    print("R5-Subspace v2: ResQ + Subspace Quantization (reviewer-fixed)")
    print("=" * 80)
    
    for seed in cfg.seeds:
        W, X = gen_tensors(cfg, seed)
        nf = noise_floor(W, X)
        print(f"\n--- Seed {seed} | W:{W.shape} X:{X.shape} ---")
        print(f"  Noise floor: hess_err={nf['hess_err_floor']:.2e}, kld={nf['proxy_kld_floor']:.2e}")
        for bits in cfg.bits_list:
            results = run_all_methods(W, X, bits, cfg)
            key = f"synthetic_seed{seed}_K{bits}"
            all_results[key] = {'results': results, 'noise_floor': nf,
                                'W_shape': list(W.shape), 'X_shape': list(X.shape)}
            print(f"\n  K={bits} ({len(results)} methods):")
            print(f"    {'Method':<28} {'Hess Err':>12} {'Bytes':>10} {'r':>4}")
            print(f"    {'-'*28} {'-'*12} {'-'*10} {'-'*4}")
            for r in sorted(results, key=lambda x: x['hess_err']):
                r_val = r.get('r', '')
                print(f"    {r['method']:<28} {r['hess_err']:12.4e} {r['bytes']:10.0f} {str(r_val):>4}")
    
    print("\n" + "=" * 80)
    print("Real weight experiment (Qwen3.8-27B)")
    print("=" * 80)
    
    real_tensors = {'L0_gate': 5, 'L0_down': 6, 'L55_gate': 5, 'L55_down': 6}
    for tname, recipe_k in real_tensors.items():
        try:
            W, X = load_real_weights(tname, 128)
            nf = noise_floor(W, X)
            results = run_all_methods(W, X, recipe_k, cfg)
            all_results[f"real_{tname}_K{recipe_k}"] = {'results': results, 'noise_floor': nf,
                                                         'W_shape': list(W.shape), 'recipe_k': recipe_k}
            print(f"\n--- {tname} (K{recipe_k}) | W:{W.shape} ---")
            print(f"  Noise floor: hess_err={nf['hess_err_floor']:.2e}")
            print(f"\n  K={recipe_k} ({len(results)} methods):")
            print(f"    {'Method':<28} {'Hess Err':>12} {'Bytes':>10} {'r':>4}")
            print(f"    {'-'*28} {'-'*12} {'-'*10} {'-'*4}")
            for r in sorted(results, key=lambda x: x['hess_err']):
                r_val = r.get('r', '')
                print(f"    {r['method']:<28} {r['hess_err']:12.4e} {r['bytes']:10.0f} {str(r_val):>4}")
        except Exception as e:
            print(f"\n  {tname}: ERROR - {e}")
            import traceback; traceback.print_exc()
    
    output_path = '/Users/mbelleau/Projects/qwen38-research-r5-subspace/receipts/research/r5-subspace-results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Hessian-weighted error by method (synthetic, 3 seeds)")
    print("=" * 80)
    for bits in cfg.bits_list:
        method_errs = {}
        for seed in cfg.seeds:
            key = f"synthetic_seed{seed}_K{bits}"
            if key not in all_results: continue
            for r in all_results[key]['results']:
                method_errs.setdefault(r['method'], []).append(r['hess_err'])
        if not method_errs: continue
        print(f"\n  K={bits}:")
        print(f"    {'Method':<28} {'Mean Hess Err':>14} {'Std':>10}")
        print(f"    {'-'*28} {'-'*14} {'-'*10}")
        for name in sorted(method_errs, key=lambda x: np.mean(method_errs[x])):
            errs = method_errs[name]
            print(f"    {name:<28} {np.mean(errs):14.4e} {np.std(errs):10.4e}")
    
    print("\n  Real weights (best per tensor):")
    for tname, recipe_k in real_tensors.items():
        key = f"real_{tname}_K{recipe_k}"
        if key not in all_results: continue
        results = all_results[key]['results']
        best = min(results, key=lambda x: x['hess_err'])
        nf = all_results[key]['noise_floor']
        print(f"    {tname:<15} K{recipe_k} {best['method']:<28} hess={best['hess_err']:.4e} "
              f"(floor={nf['hess_err_floor']:.2e})")

if __name__ == "__main__":
    main()
