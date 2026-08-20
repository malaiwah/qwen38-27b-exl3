#!/usr/bin/env python3
"""
Full trellis-tile quantization experiment: 8 algorithms, all combinations, best per K.

Algorithms:
  Preprocessing:  AWQ, SmoothQuant (SQ)
  Sensitivity:    KronQ, GuidedQuant (GQ)
  Correction:     GPTAQ, ResComp
  Postprocessing: ResQ (low-rank residual)
  Allocation:     BAQ (mixed-K per channel)

Pipeline: preprocess → sensitivity-weight → GPTAQ/ResComp correct → ResQ postprocess
"""

import numpy as np, json, time, warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from itertools import product

warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class Config:
    m: int = 128; n: int = 128; k: int = 512
    bits_list: tuple = (3, 4, 5, 6, 7)
    tile_size: int = 16; block_size: int = 16
    damping: float = 0.01; num_seeds: int = 3
    resq_rank: int = 4     # byte-equivalent to K+1 for 128x128
    gq_groups: int = 8     # GuidedQuant output channel groups
    awq_alpha: float = 0.5; sq_alpha: float = 0.5

# ==================== Utilities ====================

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

# ==================== Trellis quantizer ====================

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

def quantize_col_mixed_k(Wc, bits_vec, tile):
    """Quantize a column where each tile-row may use different bits."""
    m = Wc.shape[0]; Wq = np.zeros_like(Wc)
    for i in range(0, m, tile):
        b = int(bits_vec[i // tile])
        Wq[i:i+tile] = quantize_uniform(Wc[i:i+tile], b)
    return Wq

# ==================== Loss ====================

def kld_loss(y_fp, y_q):
    p = np.clip(softmax(y_fp, 0), 1e-12, 1); q = np.clip(softmax(y_q, 0), 1e-12, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=0)))

def wt_mse(W, Wq): return float(np.mean((W - Wq) ** 2))
def out_mse(W, Wq, X): return float(np.mean((W @ X - Wq @ X) ** 2))

# ==================== Preprocessing: AWQ + SmoothQuant ====================

def awq_scales(W, X, alpha=0.5):
    """AWQ: s_j = (max|X_j| / max|W_j|)^alpha"""
    x_max = np.max(np.abs(X), axis=1) + 1e-10
    w_max = np.max(np.abs(W), axis=0) + 1e-10
    s = (x_max / w_max) ** alpha
    return s / np.mean(s)  # normalize to mean 1

def smoothquant_scales(W, X, alpha=0.5):
    """SmoothQuant: s_j = max|X_j|^alpha / max|W_j|^(1-alpha)"""
    x_max = np.max(np.abs(X), axis=1) + 1e-10
    w_max = np.max(np.abs(W), axis=0) + 1e-10
    s = x_max ** alpha / w_max ** (1 - alpha)
    return s / np.mean(s)

def apply_scales(W, X, Xt, s):
    return W * s[None, :], X / s[:, None], Xt / s[:, None]

# ==================== Sensitivity: KronQ + GuidedQuant ====================

def kronq_weights(W, X, Xt):
    """Per-column KronQ sensitivity: tr(H_O) * H_I[j,j]"""
    H_I = X @ X.T
    tr_I = np.trace(H_I)
    p = softmax(W @ Xt, 0)
    H_O = (np.diag(p.sum(1)) - p @ p.T) / max(1, p.shape[1])
    tr_O = np.trace(H_O)
    raw = tr_O * np.diag(H_I)
    return raw / np.mean(raw)  # normalize to mean 1

def guidedquant_weights(W, Xt, num_groups=8):
    """Per-output-channel Fisher with group smoothing."""
    p = softmax(W @ Xt, 0)
    fisher = np.mean(p * (1 - p), axis=1) + 1e-10
    m = len(fisher)
    # Sort by Fisher, assign to groups, compute group-mean weight
    order = np.argsort(fisher)
    weights = np.ones(m)
    group_size = max(1, m // num_groups)
    for g in range(num_groups):
        idx = order[g * group_size:(g + 1) * group_size]
        if len(idx) > 0:
            gm = np.mean(fisher[idx])
            weights[idx] = gm / np.mean(fisher)
    return weights

# ==================== Correction helpers ====================

def compute_P(dX_Xt, L, n):
    return (dX_Xt @ L) * np.triu(np.ones((n, n)), k=1) @ L.T

def fisher_diag(W, Xt):
    p = softmax(W @ Xt, 0)
    return np.mean(p * (1 - p), axis=1)

# ==================== Core quantization ====================

def quantize_core(W, X, Xt, bits, tile, block, damping,
                  gptaq=False, rescomp=False,
                  kronq_w=None, gq_w=None,
                  baq_bits=None):
    """
    GPTQ lazy-batch with optional GPTAQ/ResComp corrections,
    KronQ/GuidedQuant sensitivity weighting, and BAQ mixed-K allocation.
    """
    m, n = W.shape
    Ww = W.copy().astype(np.float64); W0 = W.copy(); Q = np.zeros_like(Ww)

    H = X @ X.T; L = inv_cholesky(H, damping)

    # Sensitivity weights (multiply error before Cholesky update)
    sens = np.ones(m)
    if kronq_w is not None:
        sens *= 1.0 / (kronq_w + 1e-10)  # inverse: high sensitivity → less error tolerance
        sens /= np.mean(sens)
    if gq_w is not None:
        sens *= 1.0 / (gq_w + 1e-10)
        sens /= np.mean(sens)

    P = compute_P((Xt - X) @ X.T, L, n) if (gptaq or rescomp) else np.zeros((n, n))
    P2 = compute_P(Xt @ X.T, L, n) if rescomp else np.zeros((n, n))

    for i in range(0, n, block):
        B = min(block, n - i); E = np.zeros((m, B))
        for j in range(B):
            c = i + j
            if baq_bits is not None:
                # BAQ: per-tile-row mixed K
                tile_bits = baq_bits[c * tile // tile:(c * tile // tile + 1)]
                # Simplify: use column-level K
                col_k = int(baq_bits[c]) if c < len(baq_bits) else bits
                Q[:, c] = quantize_col(Ww[:, c:c+1], col_k, tile).ravel()
            else:
                Q[:, c] = quantize_col(Ww[:, c:c+1], bits, tile).ravel()

            e = (Ww[:, c] - Q[:, c]) * sens
            E[:, j] = e / L[c, c]; end = min(i + B, n)
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            if gptaq:
                Ww[:, c:end] += np.outer(Ww[:, c] * sens, P[c, c:end])
            if rescomp:
                Ww[:, c:end] += np.outer((W0[:, c] - Ww[:, c]) * sens, P2[c, c:end])
        if i + B < n:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
            if gptaq:
                Ww[:, i+B:] += (Ww[:, i:i+B] * sens[:, None]) @ P[i:i+B, i+B:]
            if rescomp:
                Ww[:, i+B:] += ((W0[:, i:i+B] - Ww[:, i:i+B]) * sens[:, None]) @ P2[i:i+B, i+B:]
    return Q

# ==================== Postprocessing: ResQ ====================

def resq_correct(W, Wq, rank=4):
    """Low-rank SVD residual correction. Byte cost: rank*(m+n)*2 vs K+1: m*n/8."""
    R = W - Wq
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    r = min(rank, len(S))
    return Wq + U[:, :r] @ np.diag(S[:r]) @ Vt[:r]

# ==================== BAQ: mixed-K allocation ====================

def baq_allocate(W, X, avg_bits, tile, min_k=3, max_k=7):
    """Allocate bits per column based on Hessian diagonal dispersion."""
    H = X @ X.T
    h_diag = np.diag(H)
    # Per-column sensitivity
    sens = h_diag / (np.mean(h_diag) + 1e-10)
    # Map to bits: log-scale around avg
    bits = np.round(avg_bits + np.clip(np.log2(sens + 1e-10), -1.5, 1.5))
    bits = np.clip(bits, min_k, max_k).astype(int)
    # Adjust to maintain average
    while np.mean(bits) > avg_bits + 0.1 and np.max(bits) > min_k:
        bits[np.argmax(bits)] -= 1
    while np.mean(bits) < avg_bits - 0.1 and np.min(bits) < max_k:
        bits[np.argmin(bits)] += 1
    return bits

# ==================== Baseline ====================

def baseline(W, bits, tile):
    return block_hadamard(trellis_quantize(block_hadamard(W, tile), bits, tile), tile)

# ==================== Unified pipeline ====================

def run_method(W, X, Xt, bits, cfg, flags):
    """
    flags: dict with keys: awq, sq, kronq, gq, gptaq, rescomp, resq, baq
    Returns: Wq (quantized weights in original space)
    """
    s = np.ones(cfg.n)

    # 1. Preprocessing
    if flags.get('awq'):
        s = s * awq_scales(W, X, cfg.awq_alpha)
    if flags.get('sq'):
        s = s * smoothquant_scales(W, X, cfg.sq_alpha)

    W_p, X_p, Xt_p = apply_scales(W, X, Xt, s) if not np.allclose(s, 1) else (W, X, Xt)

    # 2. Sensitivity weights
    kronq_w = kronq_weights(W_p, X_p, Xt_p) if flags.get('kronq') else None
    gq_w = guidedquant_weights(W_p, Xt_p, cfg.gq_groups) if flags.get('gq') else None

    # 3. BAQ mixed-K
    baq_bits = baq_allocate(W_p, X_p, bits, cfg.tile_size) if flags.get('baq') else None

    # 4. Core quantization
    has_correction = flags.get('gptaq') or flags.get('rescomp') or \
                     flags.get('kronq') or flags.get('gq') or flags.get('baq')
    if has_correction:
        Wq_p = quantize_core(
            W_p, X_p, Xt_p, bits, cfg.tile_size, cfg.block_size, cfg.damping,
            gptaq=flags.get('gptaq', False), rescomp=flags.get('rescomp', False),
            kronq_w=kronq_w, gq_w=gq_w, baq_bits=baq_bits)
    else:
        if baq_bits is not None:
            # BAQ without correction: mixed-K trellis
            m, n = W_p.shape
            Wq_p = np.zeros_like(W_p)
            for j in range(n):
                Wq_p[:, j] = quantize_col(W_p[:, j:j+1], int(baq_bits[j]), cfg.tile_size).ravel()
        else:
            Wq_p = trellis_quantize(W_p, bits, cfg.tile_size)

    # 5. Inverse scaling
    Wq = Wq_p / s[None, :] if not np.allclose(s, 1) else Wq_p

    # 6. ResQ postprocessing
    if flags.get('resq'):
        Wq = resq_correct(W, Wq, cfg.resq_rank)

    return Wq

# ==================== Sample tensors ====================

def gen_tensors(cfg, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((cfg.m, cfg.n)) * 0.05
    Xt = rng.standard_normal((cfg.n, cfg.k)) * 0.5
    Xt[rng.random(cfg.n) < 0.05] *= 5.0
    X = Xt + rng.standard_normal((cfg.n, cfg.k)) * 0.02
    return W, X, Xt

# ==================== Experiment matrix ====================

METHODS = [
    # (name, flags)
    ("baseline",           {}),
    ("GPTAQ",              {"gptaq": True}),
    ("AWQ",                {"awq": True}),
    ("SQ",                 {"sq": True}),
    ("AWQ+SQ",             {"awq": True, "sq": True}),
    ("KronQ",              {"kronq": True}),
    ("GQ",                 {"gq": True}),
    ("ResQ",               {"resq": True}),
    ("BAQ",                {"baq": True}),
    ("GPTAQ+AWQ",          {"gptaq": True, "awq": True}),
    ("GPTAQ+SQ",           {"gptaq": True, "sq": True}),
    ("GPTAQ+AWQ+SQ",       {"gptaq": True, "awq": True, "sq": True}),
    ("GPTAQ+KronQ",        {"gptaq": True, "kronq": True}),
    ("GPTAQ+GQ",           {"gptaq": True, "gq": True}),
    ("GPTAQ+ResQ",         {"gptaq": True, "resq": True}),
    ("GPTAQ+BAQ",          {"gptaq": True, "baq": True}),
    ("GPTAQ+AWQ+SQ+ResQ",  {"gptaq": True, "awq": True, "sq": True, "resq": True}),
    ("GPTAQ+AWQ+SQ+GQ+ResQ", {"gptaq": True, "awq": True, "sq": True, "gq": True, "resq": True}),
    ("GPTAQ+AWQ+SQ+KronQ+ResQ", {"gptaq": True, "awq": True, "sq": True, "kronq": True, "resq": True}),
    ("ALL",                {"gptaq": True, "awq": True, "sq": True, "kronq": True, "gq": True, "resq": True}),
]

def run(cfg):
    results = []
    for seed in range(cfg.num_seeds):
        W, X, Xt = gen_tensors(cfg, seed)
        for bits in cfg.bits_list:
            for name, flags in METHODS:
                t0 = time.time()
                if not flags:
                    Wq = baseline(W, bits, cfg.tile_size)
                else:
                    Wq = run_method(W, X, Xt, bits, cfg, flags)
                dt = time.time() - t0
                results.append({
                    "method": name, "bits": bits, "seed": seed,
                    "kld": kld_loss(W @ Xt, Wq @ X),
                    "wmse": wt_mse(W, Wq), "omse": out_mse(W, Wq, Xt),
                    "time": dt,
                })
    return results

def aggregate(results):
    agg = {}
    for r in results:
        k = (r["method"], r["bits"])
        if k not in agg: agg[k] = {m: [] for m in ["kld", "wmse", "omse", "time"]}
        for m in agg[k]: agg[k][m].append(r[m])
    out = {}
    for (meth, bits), v in sorted(agg.items()):
        out[f"{meth}@K{bits}"] = {f"{m}_mean": float(np.mean(v[m])) for m in v}
        out[f"{meth}@K{bits}"]["kld_std"] = float(np.std(v["kld"]))
    return out

def print_tables(s, bits_list):
    # Full table per K
    for bits in bits_list:
        print(f"\n{'='*100}")
        print(f"  K{bits} ({bits}-bit, {2**bits} levels/tile)")
        print(f"{'='*100}")
        print(f"  {'Method':<28} {'KLD':>12} {'vs base':>9} {'Wt MSE':>12} {'Out MSE':>12} {'Time':>7}")
        print(f"  {'-'*28} {'-'*12} {'-'*9} {'-'*12} {'-'*12} {'-'*7}")
        bk = s.get(f"baseline@K{bits}", {}).get("kld_mean", 1e-10)
        for name, _ in METHODS:
            k = f"{name}@K{bits}"
            if k not in s: continue
            d = s[k]
            imp = f"{(1-d['kld_mean']/bk)*100:+.1f}%" if name != "baseline" else ""
            print(f"  {name:<28} {d['kld_mean']:>12.4e} {imp:>9} "
                  f"{d['wmse_mean']:>12.4e} {d['omse_mean']:>12.4e} {d['time_mean']:>7.3f}")

    # Best per K summary
    print(f"\n{'='*100}")
    print(f"  BEST COMBINATION PER K")
    print(f"{'='*100}")
    print(f"  {'K':>4} {'Best Method':<28} {'KLD':>12} {'vs base':>9} {'vs GPTAQ':>9}")
    print(f"  {'-'*4} {'-'*28} {'-'*12} {'-'*9} {'-'*9}")
    for bits in bits_list:
        best_name, best_kld = None, 1e10
        for name, _ in METHODS:
            k = f"{name}@K{bits}"
            if k not in s: continue
            if s[k]["kld_mean"] < best_kld:
                best_kld = s[k]["kld_mean"]; best_name = name
        bk = s.get(f"baseline@K{bits}", {}).get("kld_mean", 1e-10)
        gk = s.get(f"GPTAQ@K{bits}", {}).get("kld_mean", 1e-10)
        imp_b = f"{(1-best_kld/bk)*100:+.1f}%"
        imp_g = f"{(1-best_kld/gk)*100:+.1f}%" if gk > 0 else ""
        print(f"  K{bits:<2} {best_name:<28} {best_kld:>12.4e} {imp_b:>9} {imp_g:>9}")

    # ResQ special: compare K+ResQ vs K+1 baseline
    print(f"\n{'='*100}")
    print(f"  ResQ: K+rank-4 residual vs K+1 baseline (same byte budget)")
    print(f"{'='*100}")
    print(f"  {'K':>4} {'K+ResQ KLD':>14} {'(K+1) base KLD':>16} {'Winner':>10}")
    print(f"  {'-'*4} {'-'*14} {'-'*16} {'-'*10}")
    for bits in cfg_bits if 'cfg_bits' in dir() else (3,4,5,6):
        kr = s.get(f"ResQ@K{bits}", {}).get("kld_mean")
        k1 = s.get(f"baseline@K{bits+1}", {}).get("kld_mean")
        if kr and k1:
            w = "ResQ" if kr < k1 else "K+1"
            print(f"  K{bits:<2} {kr:>14.4e} {k1:>16.4e} {w:>10}")
        kr2 = s.get(f"GPTAQ+ResQ@K{bits}", {}).get("kld_mean")
        g1 = s.get(f"GPTAQ@K{bits+1}", {}).get("kld_mean")
        if kr2 and g1:
            w = "GPTAQ+ResQ" if kr2 < g1 else "GPTAQ K+1"
            print(f"  K{bits}+G {kr2:>14.4e} {g1:>16.4e} {w:>10}")

def main():
    cfg = Config()
    global cfg_bits
    cfg_bits = cfg.bits_list
    print("Full trellis quantization: 8 algorithms, all combinations")
    print(f"  {cfg.m}x{cfg.n}, {cfg.k} cal, tile={cfg.tile_size}, seeds={cfg.num_seeds}")
    print(f"  ResQ rank={cfg.resq_rank} (byte-equiv K+1 for 128x128)")
    print(f"  {len(METHODS)} methods x {len(cfg.bits_list)} K x {cfg.num_seeds} seeds = {len(METHODS)*len(cfg.bits_list)*cfg.num_seeds} runs")
    t0 = time.time()
    results = run(cfg)
    print(f"Total: {time.time()-t0:.1f}s")
    s = aggregate(results)
    print_tables(s, cfg.bits_list)
    json.dump({"cfg": cfg.__dict__, "summary": s, "raw": results},
              open("trellis_quant_full_results.json", "w"), indent=2, default=str)
    print("\nSaved: trellis_quant_full_results.json")

if __name__ == "__main__": main()
