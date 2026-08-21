#!/usr/bin/env python3
"""
Historical v1 proxy harness — INVALIDATED by docs/62 §10.

This is not YAQA: it replaces YAQA's dense real-Fisher Kronecker factors and
three-term LDL rounding with a local softmax-diagonal multiplier. Its GPTAQ and
ResComp P-matrices also predate the reference-code audit. The baseline uses a
different quantizer/codebook granularity than corrected arms. Retained only to
reproduce the historical v1 receipt; do not use for algorithm conclusions.
"""

import numpy as np, json, time, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class Config:
    m: int = 128; n: int = 128; k: int = 512
    bits_list: tuple = (3, 4, 5, 6, 7)
    tile_size: int = 16; block_size: int = 16
    damping: float = 0.01; num_seeds: int = 3

# ---- Utils ----

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

# ---- Trellis quantizer ----

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

# ---- Loss ----

def kld_loss(y_fp, y_q):
    p = np.clip(softmax(y_fp, 0), 1e-12, 1); q = np.clip(softmax(y_q, 0), 1e-12, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=0)))

def wt_mse(W, Wq): return float(np.mean((W - Wq) ** 2))
def out_mse(W, Wq, X): return float(np.mean((W @ X - Wq @ X) ** 2))

# ---- Correction helpers ----

def compute_P(dX_Xt, L, n):
    return (dX_Xt @ L) * np.triu(np.ones((n, n)), k=1) @ L.T

def fisher_diag(W, Xt):
    """Diagonal of Fisher info matrix diag(p) - pp^T.
    diag_i = p_i * (1 - p_i) averaged over samples."""
    p = softmax(W @ Xt, 0)  # m x k
    return np.mean(p * (1 - p), axis=1)  # m-vector, always > 0

# ============================================================================
# Core: GPTQ + GPTAQ + YAQA + ResComp (unified column-by-column)
# ============================================================================

def quantize_combined(W, X, Xt, bits, tile, block, damping,
                      gptaq=False, yaqa=False, rescomp=False, sketch="A"):
    """
    GPTQ lazy-batch column processing with optional corrections:

    YAQA:   Pre-weights error by inverse Fisher diagonal (output sensitivity).
            This is the Kronecker approx with H_O = diag(fisher_diag).
            Well-conditioned: fisher_diag > 0 always.
    GPTAQ:  Adds P correction: W[:,j:] += W[:,j] * P[j,j:]
    ResComp: Adds CAE: W[:,j:] += (W0-Wq)[:,j] * P2[j,j:]
    """
    m, n = W.shape
    Ww = W.copy().astype(np.float64); W0 = W.copy(); Q = np.zeros_like(Ww)

    # Input Hessian + inverse Cholesky
    H = X @ X.T; L = inv_cholesky(H, damping)

    # YAQA: output Hessian preconditioner
    if yaqa and sketch == "A":
        # Sketch A: H_O from Fisher diagonal (3 power iterations)
        fd = fisher_diag(W, Xt)
        yaqa_scale = 1.0 / (fd + 1e-10)
        yaqa_scale /= np.mean(yaqa_scale)  # normalize to mean 1
    elif yaqa and sketch == "B":
        # Sketch B: H_O = I (identity, 1 round from identity — no Fisher)
        yaqa_scale = np.ones(m)
    else:
        yaqa_scale = np.ones(m)

    # GPTAQ P matrix
    P = compute_P((Xt - X) @ X.T, L, n) if (gptaq or rescomp) else np.zeros((n, n))
    # ResComp P2 matrix
    P2 = compute_P(Xt @ X.T, L, n) if rescomp else np.zeros((n, n))

    for i in range(0, n, block):
        B = min(block, n - i); E = np.zeros((m, B))
        for j in range(B):
            c = i + j
            Q[:, c] = quantize_col(Ww[:, c:c+1], bits, tile).ravel()
            # Error: YAQA pre-weights by inverse Fisher diagonal
            e = (Ww[:, c] - Q[:, c]) * yaqa_scale
            E[:, j] = e / L[c, c]; end = min(i + B, n)
            # GPTQ update
            Ww[:, c:end] -= np.outer(E[:, j], L[c, c:end])
            # GPTAQ correction (also pre-weighted by YAQA)
            if gptaq:
                Ww[:, c:end] += np.outer(Ww[:, c] * yaqa_scale, P[c, c:end])
            # ResComp CAE
            if rescomp:
                Ww[:, c:end] += np.outer((W0[:, c] - Ww[:, c]) * yaqa_scale, P2[c, c:end])
        if i + B < n:
            Ww[:, i+B:] -= E @ L[i:i+B, i+B:]
            if gptaq:
                Ww[:, i+B:] += (Ww[:, i:i+B] * yaqa_scale[:, None]) @ P[i:i+B, i+B:]
            if rescomp:
                Ww[:, i+B:] += ((W0[:, i:i+B] - Ww[:, i:i+B]) * yaqa_scale[:, None]) @ P2[i:i+B, i+B:]
    return Q

# ---- Baseline ----

def baseline(W, bits, tile):
    return block_hadamard(trellis_quantize(block_hadamard(W, tile), bits, tile), tile)

# ---- Sample tensors ----

def gen_tensors(cfg, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((cfg.m, cfg.n)) * 0.05
    Xt = rng.standard_normal((cfg.n, cfg.k)) * 0.5
    Xt[rng.random(cfg.n) < 0.05] *= 5.0
    X = Xt + rng.standard_normal((cfg.n, cfg.k)) * 0.02
    return W, X, Xt

# ---- Experiment ----

COMBOS = [
    ("baseline",          False, False, False),
    ("GPTAQ",             True,  False, False),
    ("YAQA",              False, True,  False),
    ("ResComp",           False, False, True),
    ("GPTAQ+YAQA",        True,  True,  False),
    ("GPTAQ+ResComp",     True,  False, True),
    ("YAQA+ResComp",      False, True,  True),
    ("GPTAQ+YAQA+ResComp", True, True,  True),
]
def run(cfg, sketch="A"):
    results = []
    for seed in range(cfg.num_seeds):
        W, X, Xt = gen_tensors(cfg, seed)
        for bits in cfg.bits_list:
            for name, g, y, r in COMBOS:
                t0 = time.time()
                if not g and not y and not r:
                    Wq = baseline(W, bits, cfg.tile_size)
                else:
                    Wq = quantize_combined(W, X, Xt, bits, cfg.tile_size,
                                           cfg.block_size, cfg.damping, g, y, r,
                                           sketch=sketch)
                dt = time.time() - t0
                results.append({"method": name, "bits": bits, "seed": seed,
                    "sketch": sketch,
                    "kld": kld_loss(W @ Xt, Wq @ X),
                    "wmse": wt_mse(W, Wq), "omse": out_mse(W, Wq, Xt), "t": dt})
    return results

def aggregate(results):
    agg = {}
    for r in results:
        sk = r.get("sketch", "A")
        k = (r["method"], r["bits"], sk)
        if k not in agg: agg[k] = {m: [] for m in ["kld", "wmse", "omse", "t"]}
        for m in agg[k]: agg[k][m].append(r[m])
    out = {}
    for (meth, bits, sk), v in sorted(agg.items()):
        out[f"{meth}@K{bits}@{sk}"] = {f"{m}_mean": float(np.mean(v[m])) for m in v}
        out[f"{meth}@K{bits}@{sk}"]["kld_std"] = float(np.std(v["kld"]))
    return out

def print_table(s, bits_list):
    for sk in ["A", "B"]:
        print(f"\n{'#'*95}")
        print(f"  YAQA Sketch {sk}" + (" (Fisher diagonal H_O)" if sk == "A" else " (identity H_O, no Fisher)"))
        print(f"{'#'*95}")
        for bits in bits_list:
            print(f"\n  K{bits} ({bits}-bit, {2**bits} levels/tile)")
            print(f"  {'Method':<22} {'KLD':>12} {'vs base':>10} {'Wt MSE':>12} {'Out MSE':>12} {'Time':>7}")
            print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*7}")
            bk = s.get(f"baseline@K{bits}@{sk}", s.get(f"baseline@K{bits}@A", {})).get("kld_mean", 1e-10)
            for meth, *_ in COMBOS:
                k = f"{meth}@K{bits}@{sk}"
                if k not in s: continue
                d = s[k]; imp = f"{(1-d['kld_mean']/bk)*100:+.1f}%" if meth != "baseline" else ""
                print(f"  {meth:<22} {d['kld_mean']:>12.4e} {imp:>10} "
                      f"{d['wmse_mean']:>12.4e} {d['omse_mean']:>12.4e} {d['t_mean']:>7.3f}")

    # Side-by-side comparison for YAQA-containing methods
    print(f"\n{'#'*95}")
    print(f"  Sketch A vs B comparison (YAQA-containing methods)")
    print(f"{'#'*95}")
    print(f"\n  {'Method':<22} {'Bits':>4} {'Sketch A KLD':>14} {'Sketch B KLD':>14} {'B vs A':>10}")
    print(f"  {'-'*22} {'-'*4} {'-'*14} {'-'*14} {'-'*10}")
    for bits in bits_list:
        for meth, g, y, r in COMBOS:
            if not y: continue  # only YAQA-containing methods
            ka = f"{meth}@K{bits}@A"; kb = f"{meth}@K{bits}@B"
            if ka not in s or kb not in s: continue
            va = s[ka]["kld_mean"]; vb = s[kb]["kld_mean"]
            diff = f"{(vb/va - 1)*100:+.1f}%" if va > 0 else ""
            print(f"  {meth:<22} K{bits:<2} {va:>14.4e} {vb:>14.4e} {diff:>10}")

def main():
    cfg = Config()
    print("WARNING: INVALIDATED historical proxy; see docs/62 §10")
    print(f"  {cfg.m}x{cfg.n}, {cfg.k} cal, tile={cfg.tile_size}, seeds={cfg.num_seeds}")
    t0 = time.time()
    results_A = run(cfg, sketch="A")
    results_B = run(cfg, sketch="B")
    results = results_A + results_B
    print(f"Total: {time.time()-t0:.1f}s, {len(results)} runs")
    s = aggregate(results); print_table(s, cfg.bits_list)
    json.dump({"cfg": cfg.__dict__, "summary": s, "raw": results},
              open("trellis_quant_results.json", "w"), indent=2, default=str)
    print("\nSaved: trellis_quant_results.json")

if __name__ == "__main__": main()
