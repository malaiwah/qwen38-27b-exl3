#!/usr/bin/env python3
"""
Historical real-weight proxy harness — INVALIDATED by docs/62 §10.

The L10/L20/L30/L40 slices used by the v2 receipt were BF16 bytes decoded as
IEEE FP16 and are corrupt. The harness also inherits the unmatched baseline,
proxy metrics, incomplete methods, rate mismatch, and correction bugs from
trellis_quant_full.py. Historical receipts are not decision-grade.
"""

import numpy as np, json, time, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")

# ==================== Utilities (fixed) ====================

def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def block_hadamard(W, tile):
    m, n = W.shape; r = W.copy().astype(np.float64)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            h = hadamard(min(tile, m-i))
            h2 = hadamard(min(tile, n-j))
            r[i:i+tile, j:j+tile] = h @ W[i:i+tile, j:j+tile] @ h2
    return r

def inv_cholesky(H, damping):
    n = H.shape[0]; lam = max(damping * np.mean(np.diag(H)), 1e-10)
    R = np.linalg.cholesky(H + lam * np.eye(n))
    return np.linalg.solve(R.T, np.eye(n))  # upper Cholesky of inv(H)

def softmax(x, axis=0):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def quantize_uniform(w, bits):
    nl = 2 ** bits; lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-12: return np.full_like(w, lo)
    s = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / s), 0, nl - 1) * s + lo

def trellis_quantize(W, bits, tile):
    m, n = W.shape; Wq = np.zeros_like(W)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            Wq[i:i+tile, j:j+tile] = quantize_uniform(W[i:i+tile, j:j+tile], bits)
    return Wq

def quantize_col(Wc, bits, tile):
    m = Wc.shape[0]; Wq = np.zeros_like(Wc)
    for i in range(0, m, tile): Wq[i:i+tile] = quantize_uniform(Wc[i:i+tile], bits)
    return Wq

def kld_loss(y_fp, y_q):
    p = np.clip(softmax(y_fp, 0), 1e-12, 1); q = np.clip(softmax(y_q, 0), 1e-12, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=0)))

def wt_mse(W, Wq): return float(np.mean((W - Wq) ** 2))
def out_mse(W, Wq, X): return float(np.mean((W @ X - Wq @ X) ** 2))

# ==================== AWQ (fixed: mean|X|, geom norm, clamp 1e-4) ====================

def awq_scales(W, X, alpha=0.5):
    a = np.mean(np.abs(X), axis=1)
    r = np.power(a, alpha)
    r = np.clip(r, 1e-4, None)
    return r / np.sqrt(np.max(r) * np.min(r))

def awq_scales_search(W, X, Xt, bits, tile, n_grid=20):
    a = np.mean(np.abs(X), axis=1)
    y_fp = W @ Xt
    best_kld = float('inf'); best_s = np.ones(X.shape[0])
    for i in range(n_grid):
        alpha = i / n_grid
        r = np.power(a, alpha)
        r = np.clip(r, 1e-4, None)
        s = r / np.sqrt(np.max(r) * np.min(r))
        W_p = W * s[None, :]
        Wq_p = trellis_quantize(W_p, bits, tile)
        Wq = Wq_p / s[None, :]
        kld = kld_loss(y_fp, Wq @ X)
        if kld < best_kld: best_kld = kld; best_s = s
    return best_s

# ==================== SmoothQuant (fixed: no mean norm, clamp 1e-5) ====================

def smoothquant_scales(W, X, alpha=0.5):
    a = np.max(np.abs(X), axis=1)
    w = np.max(np.abs(W), axis=0)
    s = np.power(a, alpha) / np.power(np.clip(w, 1e-5, None), 1 - alpha)
    return np.clip(s, 1e-5, None)

def apply_scales(W, X, Xt, s):
    return W * s[None, :], X / s[:, None], Xt / s[:, None]

# ==================== KronQ solver no-op (not KronQ BiIP/allocation) ====================

def kronq_weights(W, X, Xt):
    return np.ones(W.shape[1])

# ==================== Local Fisher-diagonal proxy (not GuidedQuant) ====================

def guidedquant_weights(W, Xt, num_groups=8):
    p = softmax(W @ Xt, 0)
    fisher = np.mean(p * (1 - p), axis=1)
    order = np.argsort(fisher)
    group_size = len(fisher) // num_groups
    weights = np.ones(len(fisher))
    for g in range(num_groups):
        idx = order[g*group_size:(g+1)*group_size]
        weights[idx] = np.mean(fisher[idx])
    return weights

# ==================== P-matrix (fixed: 0.25*triu(D@L.T,1)@L) ====================

def compute_P(dX_Xt, L, n, alpha=0.25):
    M = dX_Xt @ L.T
    M = np.triu(M, k=1)
    return alpha * (M @ L)

# ==================== BAQ (fixed: closed-form convex optimization) ====================

def baq_allocate(W, X, avg_bits, tile, min_k=3, max_k=7):
    M, N = W.shape
    H = X @ X.T
    lam_damp = max(0.01 * np.mean(np.diag(H)), 1e-10)
    Hinv = np.linalg.inv(H + lam_damp * np.eye(X.shape[0]))
    hinv_diag = np.diag(Hinv)
    w_range = W.max(axis=0) - W.min(axis=0)
    c = (w_range ** 2) / (12 * hinv_diag + 1e-20)
    log_c = np.log(np.clip(c, 1e-20, None))
    lam = np.exp(np.mean(log_c))
    R_star = 0.5 * np.log2(c / (lam + 1e-20)) + avg_bits
    bits = np.round(R_star).astype(int)
    bits = np.clip(bits, min_k, max_k)
    while np.mean(bits) > avg_bits + 0.1 and np.max(bits) > min_k:
        bits[np.argmax(bits)] -= 1
    while np.mean(bits) < avg_bits - 0.1 and np.min(bits) < max_k:
        bits[np.argmin(bits)] += 1
    return bits

# ==================== Partial ResQ-style PCA proxy (not rate-matched/full ResQ) ====================

def resq_quantize(W, X, bits, high_bits=8, ratio=0.125):
    m, n = W.shape
    r = max(1, int(n * ratio))
    cov = X @ X.T
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    U = eigvecs[:, idx]
    U_h = U[:, :r]; U_l = U[:, r:]
    W_h = W @ U_h; W_l = W @ U_l
    Wq_h = quantize_uniform(W_h, high_bits)
    Wq_l = quantize_uniform(W_l, bits)
    return Wq_h @ U_h.T + Wq_l @ U_l.T

# ==================== Core quantization (fixed) ====================

def fisher_diag(W, Xt):
    p = softmax(W @ Xt, 0)
    return np.mean(p * (1 - p), axis=1)

def quantize_core(W, X, Xt, bits, tile, block, damping,
                  gptaq=False, rescomp=False,
                  kronq_w=None, gq_w=None, baq_bits=None):
    m, n = W.shape
    Ww = W.copy().astype(np.float64); W0 = W.copy(); Q = np.zeros_like(Ww)
    H = X @ X.T; L = inv_cholesky(H, damping)
    sens = np.ones(m)
    if kronq_w is not None:
        sens *= 1.0 / (kronq_w + 1e-10); sens /= np.mean(sens)
    if gq_w is not None:
        sens *= 1.0 / (gq_w + 1e-10); sens /= np.mean(sens)
    P = compute_P((Xt - X) @ X.T, L, n) if gptaq else np.zeros((n, n))
    P2 = compute_P(Xt @ X.T, L, n) if rescomp else np.zeros((n, n))
    for i in range(0, n, block):
        B = min(block, n - i); E = np.zeros((m, B))
        for j in range(B):
            c = i + j
            w_pre = Ww[:, c].copy()
            if baq_bits is not None:
                col_k = int(baq_bits[c]) if c < len(baq_bits) else bits
                Q[:, c] = quantize_col(Ww[:, c:c+1], col_k, tile).ravel()
            else:
                Q[:, c] = quantize_col(Ww[:, c:c+1], bits, tile).ravel()
            e = (w_pre - Q[:, c]) * sens
            E[:, j] = e / L[c, c]; end = min(i + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            if gptaq:
                Ww[:, c:end] += np.outer(w_pre, P[c, c:end])
            if rescomp:
                Ww[:, c:end] += np.outer(W0[:, c] - w_pre, P2[c, c:end])
        if i + B < n:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
            if gptaq:
                Ww[:, i+B:] += (W[:, i:i+B]) @ P[i:i+B, i+B:]
            if rescomp:
                Ww[:, i+B:] += (W0[:, i:i+B] - W[:, i:i+B]) @ P2[i:i+B, i+B:]
    return Q

def baseline(W, bits, tile):
    return block_hadamard(trellis_quantize(block_hadamard(W, tile), bits, tile), tile)

# ==================== Unified pipeline (fixed) ====================

def run_method(W, X, Xt, bits, tile, block, damping, flags):
    n = W.shape[1]
    s = np.ones(n)
    if flags.get('awq'):
        if flags.get('awq_search'):
            s = s * awq_scales_search(W, X, Xt, bits, tile)
        else:
            s = s * awq_scales(W, X, 0.5)
    if flags.get('sq'):
        s = s * smoothquant_scales(W, X, 0.5)
    W_p, X_p, Xt_p = apply_scales(W, X, Xt, s) if not np.allclose(s, 1) else (W, X, Xt)
    
    if flags.get('resq'):
        Wq_p = resq_quantize(W_p, X_p, bits, high_bits=8, ratio=0.125)
    else:
        kronq_w = kronq_weights(W_p, X_p, Xt_p) if flags.get('kronq') else None
        gq_w = guidedquant_weights(W_p, Xt_p, 8) if flags.get('gq') else None
        baq_bits = baq_allocate(W_p, X_p, bits, tile) if flags.get('baq') else None
        has_correction = flags.get('gptaq') or flags.get('rescomp') or \
                         flags.get('kronq') or flags.get('gq') or flags.get('baq')
        if has_correction:
            Wq_p = quantize_core(W_p, X_p, Xt_p, bits, tile, block, damping,
                gptaq=flags.get('gptaq', False), rescomp=flags.get('rescomp', False),
                kronq_w=kronq_w, gq_w=gq_w, baq_bits=baq_bits)
        else:
            if baq_bits is not None:
                Wq_p = np.zeros_like(W_p)
                for j in range(n):
                    Wq_p[:, j] = quantize_col(W_p[:, j:j+1], int(baq_bits[j]), tile).ravel()
            else:
                Wq_p = trellis_quantize(W_p, bits, tile)
    
    Wq = Wq_p / s[None, :] if not np.allclose(s, 1) else Wq_p
    return Wq

# ==================== Real weight experiment ====================

@dataclass
class TensorSpec:
    name: str; bits: int; role: str; layer: int

def gen_calibration(n, k, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)) * 0.5
    outlier_mask = rng.random(n) < 0.05
    X[outlier_mask] *= 5.0
    Xtilde = X.copy()
    X = Xtilde + rng.standard_normal((n, k)) * 0.02
    return X, Xtilde

def validate_weight_slice(name, W):
    """Reject the known BF16-as-FP16 corruption before producing new receipts."""
    if not np.isfinite(W).all():
        raise ValueError(f"{name}: non-finite weight values")
    rms = float(np.sqrt(np.mean(np.square(W, dtype=np.float64))))
    if rms > 0.1:
        raise ValueError(
            f"{name}: RMS={rms:.4g} is implausible for this Qwen checkpoint; "
            "the v2 mid-layer file decoded BF16 bytes as IEEE FP16"
        )

def run_real_experiment():
    # Load existing weights (full tensors, subsample to 128x128)
    existing = np.load("/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz")
    # Load extra weights (already 128x128)
    extra = np.load("/Users/mbelleau/Projects/cleanroom/qwen38_extra_weights.npz")
    
    # K5K6 recipe: gate_proj=K5, down_proj/attention=K6
    specs = [
        # Original L0 tensors (subsample from full)
        TensorSpec("L0_gate",  5, "MLP gate (early)",  0),
        TensorSpec("L0_down",  6, "MLP down (early)",  0),
        TensorSpec("L0_qkv",   6, "GDN QKV (early)",   0),
        TensorSpec("L0_out",   6, "GDN out (early)",    0),
        TensorSpec("L0_z",     6, "GDN z (early)",      0),
        # New mid-layer tensors
        TensorSpec("L10_gate", 5, "MLP gate (early-mid)", 10),
        TensorSpec("L10_down", 6, "MLP down (early-mid)", 10),
        TensorSpec("L20_gate", 5, "MLP gate (mid)",      20),
        TensorSpec("L20_down", 6, "MLP down (mid)",      20),
        TensorSpec("L30_gate", 5, "MLP gate (mid-late)", 30),
        TensorSpec("L30_down", 6, "MLP down (mid-late)", 30),
        TensorSpec("L40_gate", 5, "MLP gate (late)",     40),
        TensorSpec("L40_down", 6, "MLP down (late)",     40),
        # Original L55 tensors
        TensorSpec("L55_gate", 5, "MLP gate (last)",    55),
        TensorSpec("L55_down", 6, "MLP down (last)",    55),
    ]
    
    tile = 16; block = 16; damping = 0.01
    k_cal = 512
    
    # Restructured method matrix (ResQ standalone, KronQ removed from combos)
    methods = [
        ("baseline",                {}),
        ("GPTAQ",                   {"gptaq": True}),
        ("AWQ",                     {"awq": True}),
        ("SQ",                      {"sq": True}),
        ("AWQ+SQ",                  {"awq": True, "sq": True}),
        ("GQ",                      {"gq": True}),
        ("ResQ",                    {"resq": True}),
        ("BAQ",                     {"baq": True}),
        ("GPTAQ+AWQ",               {"gptaq": True, "awq": True}),
        ("GPTAQ+SQ",                {"gptaq": True, "sq": True}),
        ("GPTAQ+AWQ+SQ",            {"gptaq": True, "awq": True, "sq": True}),
        ("GPTAQ+GQ",                {"gptaq": True, "gq": True}),
        ("GPTAQ+BAQ",               {"gptaq": True, "baq": True}),
        ("GPTAQ+AWQ+SQ+BAQ",        {"gptaq": True, "awq": True, "sq": True, "baq": True}),
        ("GPTAQ+AWQ+SQ+GQ+BAQ",     {"gptaq": True, "awq": True, "sq": True, "gq": True, "baq": True}),
        ("AWQ(search)",             {"awq": True, "awq_search": True}),
        ("GPTAQ+AWQ(search)+SQ+BAQ",{"gptaq": True, "awq": True, "awq_search": True, "sq": True, "baq": True}),
    ]
    
    all_results = []
    for spec in specs:
        # Get tensor: existing tensors need subsampling, extra are already 128x128
        if spec.name in existing:
            W = existing[spec.name].astype(np.float64)[:128, :128]
        elif spec.name in extra:
            W = extra[spec.name].astype(np.float64)
        else:
            print(f"  WARNING: {spec.name} not found, skipping")
            continue
        validate_weight_slice(spec.name, W)
        
        m, n = W.shape
        X, Xt = gen_calibration(n, k_cal, seed=42)
        
        print(f"\n{'='*95}")
        print(f"  {spec.name}: {spec.role}, W={W.shape}, K={spec.bits}")
        print(f"{'='*95}")
        print(f"  {'Method':<32} {'KLD':>12} {'vs base':>9} {'Wt MSE':>12} {'Out MSE':>12} {'Time':>7}")
        print(f"  {'-'*32} {'-'*12} {'-'*9} {'-'*12} {'-'*12} {'-'*7}")
        
        base_kld = None
        for name, flags in methods:
            t0 = time.time()
            if not flags:
                Wq = baseline(W, spec.bits, tile)
            else:
                Wq = run_method(W, X, Xt, spec.bits, tile, block, damping, flags)
            dt = time.time() - t0
            kld = kld_loss(W @ Xt, Wq @ X)
            wmse = wt_mse(W, Wq)
            omse = out_mse(W, Wq, Xt)
            if base_kld is None: base_kld = kld
            imp = f"{(1-kld/base_kld)*100:+.1f}%" if name != "baseline" else ""
            print(f"  {name:<32} {kld:>12.4e} {imp:>9} {wmse:>12.4e} {omse:>12.4e} {dt:>7.3f}")
            all_results.append({
                "tensor": spec.name, "role": spec.role, "layer": spec.layer,
                "bits": spec.bits, "method": name, "kld": kld, "wmse": wmse, "omse": omse, "time": dt
            })
    
    # Summary: best per tensor
    print(f"\n{'='*95}")
    print(f"  BEST METHOD PER TENSOR")
    print(f"{'='*95}")
    print(f"  {'Tensor':<12} {'Role':<22} {'K':>3} {'Best Method':<32} {'KLD':>12} {'vs base':>9}")
    print(f"  {'-'*12} {'-'*22} {'-'*3} {'-'*32} {'-'*12} {'-'*9}")
    for spec in specs:
        tensor_results = [r for r in all_results if r["tensor"] == spec.name]
        if not tensor_results: continue
        best = min(tensor_results, key=lambda r: r["kld"])
        base = next((r["kld"] for r in tensor_results if r["method"] == "baseline"), 1e-10)
        imp = f"{(1-best['kld']/base)*100:+.1f}%"
        print(f"  {spec.name:<12} {spec.role:<22} K{spec.bits:<2} {best['method']:<32} {best['kld']:>12.4e} {imp:>9}")
    
    # Summary: best per K (aggregate across tensors)
    print(f"\n{'='*95}")
    print(f"  BEST METHOD PER K (mean KLD across all tensors at that K)")
    print(f"{'='*95}")
    for bits in [5, 6]:
        tensor_names = [s.name for s in specs if s.bits == bits]
        if not tensor_names: continue
        method_klds = {}
        for name, _ in methods:
            klds = [r["kld"] for r in all_results if r["method"] == name and r["bits"] == bits]
            if klds: method_klds[name] = np.mean(klds)
        if not method_klds: continue
        best_name = min(method_klds, key=method_klds.get)
        base_mean = method_klds.get("baseline", 1e-10)
        gptaq_mean = method_klds.get("GPTAQ", 1e-10)
        print(f"  K{bits}: best={best_name:<32} mean_KLD={method_klds[best_name]:.4e} "
              f"vs base +{(1-method_klds[best_name]/base_mean)*100:.1f}% "
              f"vs GPTAQ +{(1-method_klds[best_name]/gptaq_mean)*100:.1f}%")
        for name in sorted(method_klds, key=method_klds.get)[:5]:
            print(f"    {name:<32} {method_klds[name]:.4e}")
    
    # Layer trend analysis
    print(f"\n{'='*95}")
    print(f"  LAYER TREND: GPTAQ+AWQ+SQ+BAQ vs GPTAQ (improvement by layer)")
    print(f"{'='*95}")
    print(f"  {'Tensor':<12} {'Layer':>5} {'K':>3} {'GPTAQ KLD':>12} {'Best KLD':>12} {'vs GPTAQ':>9}")
    print(f"  {'-'*12} {'-'*5} {'-'*3} {'-'*12} {'-'*12} {'-'*9}")
    for spec in specs:
        gptaq_r = next((r for r in all_results if r["tensor"] == spec.name and r["method"] == "GPTAQ"), None)
        best_r = min((r for r in all_results if r["tensor"] == spec.name), key=lambda r: r["kld"])
        if gptaq_r and best_r:
            imp = f"{(1-best_r['kld']/gptaq_r['kld'])*100:+.1f}%"
            print(f"  {spec.name:<12} L{spec.layer:<4} K{spec.bits:<2} {gptaq_r['kld']:>12.4e} {best_r['kld']:>12.4e} {imp:>9}")
    
    json.dump(all_results, open("/Users/mbelleau/Projects/cleanroom/real_weights_results.json", "w"),
              indent=2, default=str)
    print("\nSaved: real_weights_results.json")

if __name__ == "__main__": run_real_experiment()
