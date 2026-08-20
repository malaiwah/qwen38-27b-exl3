#!/usr/bin/env python3
"""
Real Qwen3.8-27B tensor experiment.

Uses actual BF16 weight matrices downloaded from Qwen/Qwen3.8-27B on HuggingFace:
  - Layer 0 MLP: gate_proj [17408, 5120], down_proj [5120, 17408]
  - Layer 0 GDN attention: in_proj_qkv [10240, 5120], out_proj [5120, 6144], in_proj_z [6144, 5120]
  - Layer 55 MLP: gate_proj [17408, 5120], down_proj [5120, 17408]

K5K6 recipe: gate_proj=K5, down_proj=K6, attention=K6

Calibration: synthetic activations with realistic statistics (outlier channels, 
correlations) matching the KLD suite's calibration data profile.
"""

import numpy as np, json, time, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")

# ==================================================================
# Import all algorithm implementations from the full experiment
# ==================================================================

def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)

def block_hadamard(W, tile):
    m, n = W.shape; r = W.copy().astype(np.float64)
    for i in range(0, m, tile):
        s = min(tile, m - i)
        if s > 1 and (s & (s-1)) == 0: r[i:i+s] = hadamard(s) @ r[i:i+s]
    for j in range(0, n, tile):
        s = min(tile, n - j)
        if s > 1 and (s & (s-1)) == 0: r[:, j:j+s] = r[:, j:j+s] @ hadamard(s)
    return r

def inv_cholesky(H, damping):
    n = H.shape[0]; lam = max(damping * np.mean(np.diag(H)), 1e-10)
    R = np.linalg.cholesky(H + lam * np.eye(n))
    return np.linalg.solve(R.T, np.eye(n))

def softmax(x, axis=0):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def quantize_uniform(w, bits):
    nl = 2 ** bits; lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-12: return w.copy()
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

def compute_P(dX_Xt, L, n):
    return (dX_Xt @ L) * np.triu(np.ones((n, n)), k=1) @ L.T

def fisher_diag(W, Xt):
    p = softmax(W @ Xt, 0)
    return np.mean(p * (1 - p), axis=1)

def awq_scales(W, X, alpha=0.5):
    x_max = np.max(np.abs(X), axis=1) + 1e-10
    w_max = np.max(np.abs(W), axis=0) + 1e-10
    s = (x_max / w_max) ** alpha
    return s / np.mean(s)

def smoothquant_scales(W, X, alpha=0.5):
    x_max = np.max(np.abs(X), axis=1) + 1e-10
    w_max = np.max(np.abs(W), axis=0) + 1e-10
    s = x_max ** alpha / w_max ** (1 - alpha)
    return s / np.mean(s)

def apply_scales(W, X, Xt, s):
    return W * s[None, :], X / s[:, None], Xt / s[:, None]

def kronq_weights(W, X, Xt):
    H_I = X @ X.T; tr_I = np.trace(H_I)
    p = softmax(W @ Xt, 0)
    H_O = (np.diag(p.sum(1)) - p @ p.T) / max(1, p.shape[1])
    tr_O = np.trace(H_O)
    raw = tr_O * np.diag(H_I)
    return raw / np.mean(raw)

def guidedquant_weights(W, Xt, num_groups=8):
    p = softmax(W @ Xt, 0)
    fisher = np.mean(p * (1 - p), axis=1) + 1e-10
    m = len(fisher); order = np.argsort(fisher)
    weights = np.ones(m); group_size = max(1, m // num_groups)
    for g in range(num_groups):
        idx = order[g * group_size:(g + 1) * group_size]
        if len(idx) > 0:
            gm = np.mean(fisher[idx]); weights[idx] = gm / np.mean(fisher)
    return weights

def baq_allocate(W, X, avg_bits, tile, min_k=3, max_k=7):
    H = X @ X.T; h_diag = np.diag(H)
    sens = h_diag / (np.mean(h_diag) + 1e-10)
    bits = np.round(avg_bits + np.clip(np.log2(sens + 1e-10), -1.5, 1.5))
    bits = np.clip(bits, min_k, max_k).astype(int)
    while np.mean(bits) > avg_bits + 0.1 and np.max(bits) > min_k:
        bits[np.argmax(bits)] -= 1
    while np.mean(bits) < avg_bits - 0.1 and np.min(bits) < max_k:
        bits[np.argmin(bits)] += 1
    return bits

def resq_correct(W, Wq, rank=4):
    R = W - Wq; U, S, Vt = np.linalg.svd(R, full_matrices=False)
    r = min(rank, len(S))
    return Wq + U[:, :r] @ np.diag(S[:r]) @ Vt[:r]

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
    P = compute_P((Xt - X) @ X.T, L, n) if (gptaq or rescomp) else np.zeros((n, n))
    P2 = compute_P(Xt @ X.T, L, n) if rescomp else np.zeros((n, n))
    for i in range(0, n, block):
        B = min(block, n - i); E = np.zeros((m, B))
        for j in range(B):
            c = i + j
            if baq_bits is not None:
                col_k = int(baq_bits[c]) if c < len(baq_bits) else bits
                Q[:, c] = quantize_col(Ww[:, c:c+1], col_k, tile).ravel()
            else:
                Q[:, c] = quantize_col(Ww[:, c:c+1], bits, tile).ravel()
            e = (Ww[:, c] - Q[:, c]) * sens
            E[:, j] = e / L[c, c]; end = min(i + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            if gptaq: Ww[:, c:end] += np.outer(Ww[:, c] * sens, P[c, c:end])
            if rescomp: Ww[:, c:end] += np.outer((W0[:, c] - Ww[:, c]) * sens, P2[c, c:end])
        if i + B < n:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
            if gptaq: Ww[:, i+B:] += (Ww[:, i:i+B] * sens[:, None]) @ P[i:i+B, i+B:]
            if rescomp: Ww[:, i+B:] += ((W0[:, i:i+B] - Ww[:, i:i+B]) * sens[:, None]) @ P2[i:i+B, i+B:]
    return Q

def baseline(W, bits, tile):
    return block_hadamard(trellis_quantize(block_hadamard(W, tile), bits, tile), tile)

def run_method(W, X, Xt, bits, tile, block, damping, flags):
    s = np.ones(W.shape[1])
    if flags.get('awq'): s = s * awq_scales(W, X)
    if flags.get('sq'): s = s * smoothquant_scales(W, X)
    Wp, Xp, Xtp = apply_scales(W, X, Xt, s) if not np.allclose(s, 1) else (W, X, Xt)
    kronq_w = kronq_weights(Wp, Xp, Xtp) if flags.get('kronq') else None
    gq_w = guidedquant_weights(Wp, Xtp) if flags.get('gq') else None
    baq_bits = baq_allocate(Wp, Xp, bits, tile) if flags.get('baq') else None
    has_corr = flags.get('gptaq') or flags.get('rescomp') or flags.get('kronq') or flags.get('gq') or flags.get('baq')
    if has_corr:
        Wqp = quantize_core(Wp, Xp, Xtp, bits, tile, block, damping,
                            gptaq=flags.get('gptaq', False), rescomp=flags.get('rescomp', False),
                            kronq_w=kronq_w, gq_w=gq_w, baq_bits=baq_bits)
    elif baq_bits is not None:
        m, n = Wp.shape; Wqp = np.zeros_like(Wp)
        for j in range(n): Wqp[:, j] = quantize_col(Wp[:, j:j+1], int(baq_bits[j]), tile).ravel()
    else:
        Wqp = trellis_quantize(Wp, bits, tile)
    Wq = Wqp / s[None, :] if not np.allclose(s, 1) else Wqp
    if flags.get('resq'): Wq = resq_correct(W, Wq, 4)
    return Wq

# ==================================================================
# Real weight experiment
# ==================================================================

@dataclass
class TensorSpec:
    name: str; bits: int; role: str; layer: int

def gen_calibration(n, k, seed=42):
    """Generate realistic calibration activations matching LLM hidden states."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)) * 0.5
    # Add outlier channels (5% of channels have 5x magnitude)
    outlier_mask = rng.random(n) < 0.05
    X[outlier_mask] *= 5.0
    # Quant-flow: FP + noise from previous layer quantization
    Xtilde = X.copy()
    X = Xtilde + rng.standard_normal((n, k)) * 0.02
    return X, Xtilde

def run_real_experiment():
    data = np.load("/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz")
    
    # K5K6 recipe: gate_proj=K5, down_proj=K6, attention=K6
    specs = [
        TensorSpec("L0_gate",  5, "MLP gate (early)",  0),
        TensorSpec("L0_down",  6, "MLP down (early)",  0),
        TensorSpec("L0_qkv",   6, "GDN QKV (early)",   0),
        TensorSpec("L0_out",   6, "GDN out (early)",    0),
        TensorSpec("L0_z",     6, "GDN z (early)",      0),
        TensorSpec("L55_gate", 5, "MLP gate (late)",   55),
        TensorSpec("L55_down", 6, "MLP down (late)",   55),
    ]
    
    tile = 16; block = 16; damping = 0.01
    k_cal = 512  # calibration samples
    
    methods = [
        ("baseline",           {}),
        ("GPTAQ",              {"gptaq": True}),
        ("AWQ",                {"awq": True}),
        ("SQ",                 {"sq": True}),
        ("KronQ",              {"kronq": True}),
        ("GQ",                 {"gq": True}),
        ("ResQ",               {"resq": True}),
        ("BAQ",                {"baq": True}),
        ("GPTAQ+AWQ",          {"gptaq": True, "awq": True}),
        ("GPTAQ+SQ",           {"gptaq": True, "sq": True}),
        ("GPTAQ+ResQ",         {"gptaq": True, "resq": True}),
        ("GPTAQ+BAQ",          {"gptaq": True, "baq": True}),
        ("GPTAQ+AWQ+SQ+ResQ",  {"gptaq": True, "awq": True, "sq": True, "resq": True}),
        ("GPTAQ+AWQ+SQ+GQ+ResQ", {"gptaq": True, "awq": True, "sq": True, "gq": True, "resq": True}),
        ("ALL",                {"gptaq": True, "awq": True, "sq": True, "kronq": True, "gq": True, "resq": True, "baq": True}),
    ]
    
    all_results = []
    for spec in specs:
        W = data[spec.name].astype(np.float64)
        m, n = W.shape
        # For large matrices, subsample to keep Cholesky tractable on CPU
        # Use first 128 output rows and first 128 input cols
        m_sub = min(128, m); n_sub = min(128, n)
        W_sub = W[:m_sub, :n_sub]
        
        X, Xt = gen_calibration(n_sub, k_cal, seed=42)
        
        print(f"\n{'='*95}")
        print(f"  {spec.name}: {spec.role}, W={W_sub.shape}, K={spec.bits}")
        print(f"{'='*95}")
        print(f"  {'Method':<28} {'KLD':>12} {'vs base':>9} {'Wt MSE':>12} {'Out MSE':>12} {'Time':>7}")
        print(f"  {'-'*28} {'-'*12} {'-'*9} {'-'*12} {'-'*12} {'-'*7}")
        
        base_kld = None
        for name, flags in methods:
            t0 = time.time()
            if not flags:
                Wq = baseline(W_sub, spec.bits, tile)
            else:
                Wq = run_method(W_sub, X, Xt, spec.bits, tile, block, damping, flags)
            dt = time.time() - t0
            kld = kld_loss(W_sub @ Xt, Wq @ X)
            wmse = wt_mse(W_sub, Wq)
            omse = out_mse(W_sub, Wq, Xt)
            if base_kld is None: base_kld = kld
            imp = f"{(1-kld/base_kld)*100:+.1f}%" if name != "baseline" else ""
            print(f"  {name:<28} {kld:>12.4e} {imp:>9} {wmse:>12.4e} {omse:>12.4e} {dt:>7.3f}")
            all_results.append({
                "tensor": spec.name, "role": spec.role, "bits": spec.bits,
                "method": name, "kld": kld, "wmse": wmse, "omse": omse, "time": dt
            })
    
    # Summary: best per tensor
    print(f"\n{'='*95}")
    print(f"  BEST METHOD PER TENSOR")
    print(f"{'='*95}")
    print(f"  {'Tensor':<12} {'Role':<22} {'K':>3} {'Best Method':<28} {'KLD':>12} {'vs base':>9}")
    print(f"  {'-'*12} {'-'*22} {'-'*3} {'-'*28} {'-'*12} {'-'*9}")
    for spec in specs:
        tensor_results = [r for r in all_results if r["tensor"] == spec.name]
        if not tensor_results: continue
        best = min(tensor_results, key=lambda r: r["kld"])
        base = next((r["kld"] for r in tensor_results if r["method"] == "baseline"), 1e-10)
        imp = f"{(1-best['kld']/base)*100:+.1f}%"
        print(f"  {spec.name:<12} {spec.role:<22} K{spec.bits:<2} {best['method']:<28} {best['kld']:>12.4e} {imp:>9}")
    
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
        print(f"  K{bits}: best={best_name:<28} mean_KLD={method_klds[best_name]:.4e} "
              f"vs base +{(1-method_klds[best_name]/base_mean)*100:.1f}% "
              f"vs GPTAQ +{(1-method_klds[best_name]/gptaq_mean)*100:.1f}%")
        # Show all methods ranked
        for name in sorted(method_klds, key=method_klds.get)[:5]:
            print(f"    {name:<28} {method_klds[name]:.4e}")
    
    json.dump(all_results, open("/Users/mbelleau/Projects/cleanroom/real_weights_results.json", "w"),
              indent=2, default=str)
    print("\nSaved: real_weights_results.json")

if __name__ == "__main__": run_real_experiment()
