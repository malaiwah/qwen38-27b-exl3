#!/usr/bin/env python3
"""
R27-EqualRateKLD v3: Equal-rate BiP+Had vs K5.5-alloc — 16×16 affine-uniform HWE proxy

Corrected per openai-reviewer v1 feedback:
  1. Hadamard orientation FIXED: H @ diag(signs), not diag(signs) @ H
  2. K5.5-alloc refined with full-HWE local search (not just tile-local surrogate)
  3. BiIP scales computed GLOBALLY from full tensor (production-representative sidecar)
  4. Byte accounting: K-map bytes, correct Hadamard sign bytes, scale endpoints
  5. Per-slice win counts (not just means)
  6. Proper lower convex envelope (not just K5-K6 chord test)
  7. Complete factorial: {none, BiP, Had, BiP+Had} × {K4, K5, K6}
  8. Fixed tautology claim: only S_X is tautological for per-column, not S_G

QSRT review insights incorporated:
  - Local HWE can INVERT end-to-end KLD (QSRT validation lesson).
    All findings are PROXY-only and must be validated with KLD harness on aiboss.
  - Legal MLP rotation via explicit activation-boundary transforms exists (QSRT).
    Our BiP+Had is a simpler version; QSRT's explicit inverse-transform approach
    may be more effective for gate/up/down tensors.
  - SQG loses at K5/K6; we use standard uniform quantizer (not SQG).

Key design:
  - Per-tile 16×16 uniform quantizer (matching EXL3 tile granularity)
  - BiIP scales computed from FULL tensor (global row/column norms + Hessians)
  - Sidecar charged at global rate: (d_out_full + d_in_full) * 4 / (d_out_full * d_in_full)
  - HWE measured in ORIGINAL space after inverse transform (expert test #3)
  - 4 real tensors, 3 slices, multiple RHT seeds
  - Complete {none, BiP, Had, BiP+Had} × {K4, K5, K6} factorial
"""

import numpy as np
import json
import time
import os
from pathlib import Path

# ─── Paths ───
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = Path(__file__).parent.parent.parent.parent / "receipts" / "research" / "r27-equal-rate-kld-results.json"

# ─── Configuration ───
TILE = 16
SLICE_SIZE = 128
N_SLICES = 3
SEED = 42
N_CALIB = 512
N_RHT_SEEDS = 3  # Multiple Hadamard draws per slice

TENSOR_NAMES = ["L0_gate", "L0_down", "L55_gate", "L55_down"]
SLICE_NAMES = ["first", "middle", "last"]

# ============================================================================
# Quantizer — per-tile 16×16 uniform
# ============================================================================

def quantize_tile(w, k):
    nl = 2 ** k
    lo, hi = w.min(), w.max()
    if hi - lo < 1e-12:
        return w.copy()
    step = (hi - lo) / (nl - 1)
    return np.clip(np.round((w - lo) / step), 0, nl - 1) * step + lo


def quantize_matrix_uniform(W, k, tile=TILE):
    m, n = W.shape
    Wq = np.zeros_like(W)
    for ti in range((m + tile - 1) // tile):
        for tj in range((n + tile - 1) // tile):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            Wq[r0:r1, c0:c1] = quantize_tile(W[r0:r1, c0:c1], k)
    return Wq


def quantize_matrix_alloc(W, K_alloc, tile=TILE):
    m, n = W.shape
    Wq = np.zeros_like(W)
    n_tm, n_tn = K_alloc.shape
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            Wq[r0:r1, c0:c1] = quantize_tile(W[r0:r1, c0:c1], K_alloc[ti, tj])
    return Wq


# ============================================================================
# BiIP diagonal balancing (KronQ Eq. 11) — GLOBAL scales
# ============================================================================

def compute_global_biip_scales(W_full, X_full, d_out_full, d_in_full):
    """Compute BiIP scales from the FULL tensor (production-representative).

    S_X = diag(H_X_jj / ||W[:,j]||^2)^{1/4}  (per input channel, global)
    S_G = diag(H_G_ii / ||W[i,:]||^2)^{1/4}  (per output channel, global)

    Returns sg_global (d_out_full,), sx_global (d_in_full,)
    """
    col_norms_sq = np.maximum(np.sum(W_full ** 2, axis=0), 1e-12)
    row_norms_sq = np.maximum(np.sum(W_full ** 2, axis=1), 1e-12)

    N = X_full.shape[1]
    H_X_diag = np.sum(X_full ** 2, axis=1) / N
    H_X_diag = np.maximum(H_X_diag, 1e-12)

    Y_full = W_full @ X_full
    H_G_diag = np.sum(Y_full ** 2, axis=1) / N
    H_G_diag = np.maximum(H_G_diag, 1e-12)

    H_X_diag *= d_in_full / np.sum(H_X_diag)
    H_G_diag *= d_out_full / np.sum(H_G_diag)

    sx_global = np.clip((H_X_diag / col_norms_sq) ** 0.25, 0.1, 10.0)
    sg_global = np.clip((H_G_diag / row_norms_sq) ** 0.25, 0.1, 10.0)

    return sg_global, sx_global


def compute_global_sidecar_bytes(d_out_full, d_in_full, scale_bits=32):
    """Sidecar bytes for global BiIP scales.
    In production: (d_out + d_in) * (scale_bits/8) bytes for the whole tensor.
    For int8 scales: also need 2 endpoints per vector (4 float32 = 16 bytes)."""
    base_bytes = (d_out_full + d_in_full) * (scale_bits // 8)
    if scale_bits < 32:
        # Need to store min/max log-range for dequantization: 2 float32 per vector
        base_bytes += 4 * 4  # 4 endpoints (sg_lo, sg_hi, sx_lo, sx_hi)
    return base_bytes


def biip_scales_quantized(sg, sx, scale_bits=32):
    """Quantize BiIP scales to lower precision for sidecar rate sweep."""
    if scale_bits >= 32:
        return sg, sx
    for diag in [sg, sx]:
        log_vals = np.log(np.maximum(diag, 1e-12))
        lo, hi = log_vals.min(), log_vals.max()
        if hi - lo < 1e-12:
            continue
        levels = 2 ** scale_bits
        step = (hi - lo) / (levels - 1)
        q_log = np.round((log_vals - lo) / step).clip(0, levels - 1)
        log_deq = q_log * step + lo
        diag[:] = np.exp(log_deq)
    return sg, sx


def inverse_biip(W_q, sg, sx):
    return (1.0 / sg)[:, None] * W_q * (1.0 / sx)[None, :]


# ============================================================================
# Hadamard transform — CORRECTED: H @ diag(signs)
# ============================================================================

def hadamard_matrix(n):
    H = np.array([[1]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    """CORRECTED RHT: H @ diag(signs).
    Signs applied BEFORE Hadamard mixing (QuIP#/KronQ convention).
    H @ diag(signs) = H * signs[None, :] (column scaling)."""
    H = hadamard_matrix(n)
    signs = rng.choice([-1, 1], size=n)
    return H * signs[None, :], signs


def hadamard_sign_bytes(n):
    """Exact sign storage: n bits, ceiling to bytes."""
    return (n + 7) // 8


# ============================================================================
# DP tile allocation + full-HWE local search refinement
# ============================================================================

def measure_tile_distortion_local(W_tile, k, H_G_sub, H_X_sub):
    if k == 0:
        return float(np.trace(H_G_sub @ W_tile @ H_X_sub @ W_tile.T))
    Wq = quantize_tile(W_tile, k)
    E = W_tile - Wq
    D = float(np.trace(H_G_sub @ E @ H_X_sub @ E.T))
    return max(D, 0.0)


def alloc_tile_dp(W, H_X, H_G, k_sum_budget, tile=TILE, k_range=(5, 6)):
    """Tile-local DP: exact for additive surrogate."""
    m, n = W.shape
    n_tm = (m + tile - 1) // tile
    n_tn = (n + tile - 1) // tile
    n_tiles = n_tm * n_tn

    D_table = np.zeros((n_tiles, len(k_range)))
    for ti in range(n_tm):
        for tj in range(n_tn):
            r0, r1 = ti * tile, min((ti + 1) * tile, m)
            c0, c1 = tj * tile, min((tj + 1) * tile, n)
            t = n_tn * ti + tj
            W_tile = W[r0:r1, c0:c1]
            H_G_sub = H_G[r0:r1, r0:r1]
            H_X_sub = H_X[c0:c1, c0:c1]
            for ki, k in enumerate(k_range):
                D_table[t, ki] = measure_tile_distortion_local(W_tile, k, H_G_sub, H_X_sub)

    k_min, k_max = min(k_range), max(k_range)
    max_ks = k_max * n_tiles
    INF = 1e18
    dp = np.full((n_tiles + 1, max_ks + 1), INF)
    choice = np.zeros((n_tiles + 1, max_ks + 1), dtype=int)
    dp[0, 0] = 0.0

    for t in range(n_tiles):
        for ks in range(max_ks + 1):
            if dp[t, ks] >= INF:
                continue
            for ki, k in enumerate(k_range):
                nks = ks + k
                if nks > max_ks:
                    continue
                val = dp[t, ks] + D_table[t, ki]
                if val < dp[t + 1, nks]:
                    dp[t + 1, nks] = val
                    choice[t + 1, nks] = k

    best_ks = k_sum_budget
    if dp[n_tiles, k_sum_budget] >= INF:
        for delta in range(1, max_ks):
            for ks in [k_sum_budget - delta, k_sum_budget + delta]:
                if 0 <= ks <= max_ks and dp[n_tiles, ks] < INF:
                    best_ks = ks
                    break
            if best_ks != k_sum_budget:
                break

    K_flat = np.zeros(n_tiles, dtype=int)
    ks = best_ks
    for t in range(n_tiles, 0, -1):
        k = choice[t, ks]
        K_flat[t - 1] = k
        ks -= k

    return K_flat.reshape(n_tm, n_tn), best_ks


def full_hwe(W, Wq, H_X, H_G):
    """Full Hessian-weighted error: tr(H_G @ E @ H_X @ E^T)."""
    E = W - Wq
    return float(np.trace(H_G @ E @ H_X @ E.T))


def local_search_refine(W, K_alloc, H_X, H_G, k_sum_target, tile=TILE,
                        k_range=(5, 6), max_iters=3000):
    """Full-HWE local search: try single-tile K swaps preserving K-sum.
    Uses the ACTUAL full HWE, not the tile-local surrogate."""
    m, n = W.shape
    n_tm, n_tn = K_alloc.shape
    n_tiles = n_tm * n_tn
    K_flat = K_alloc.flatten().copy()

    Wq = quantize_matrix_alloc(W, K_alloc, tile)
    current_hwe = full_hwe(W, Wq, H_X, H_G)

    k_min, k_max = min(k_range), max(k_range)
    rng = np.random.default_rng(SEED + 999)

    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        order = rng.permutation(n_tiles)
        for t_up in order:
            if K_flat[t_up] >= k_max:
                continue
            for t_down in order:
                if t_down == t_up:
                    continue
                if K_flat[t_down] <= k_min:
                    continue
                K_trial = K_flat.copy()
                K_trial[t_up] += 1
                K_trial[t_down] -= 1
                K_trial_alloc = K_trial.reshape(n_tm, n_tn)
                Wq_trial = quantize_matrix_alloc(W, K_trial_alloc, tile)
                trial_hwe = full_hwe(W, Wq_trial, H_X, H_G)
                if trial_hwe < current_hwe:
                    K_flat = K_trial
                    current_hwe = trial_hwe
                    improved = True
                    break
            if improved:
                break

    return K_flat.reshape(n_tm, n_tn), current_hwe


# ============================================================================
# Byte accounting — corrected
# ============================================================================

def compute_bytes_uniform(k, m, n, sidecar_bytes=0, tile=TILE):
    n_elements = m * n
    payload_bytes = n_elements * k / 8.0
    n_tiles = ((m + tile - 1) // tile) * ((n + tile - 1) // tile)
    meta_bytes = n_tiles * 2 * 4  # per-tile min/max (float32)
    total_bytes = payload_bytes + meta_bytes + sidecar_bytes
    return {
        "payload_bytes": payload_bytes, "meta_bytes": meta_bytes,
        "sidecar_bytes": sidecar_bytes, "k_map_bytes": 0,
        "total_bytes": total_bytes, "n_elements": n_elements,
        "bits_per_element": total_bytes * 8 / n_elements,
    }


def compute_bytes_alloc(K_alloc, m, n, sidecar_bytes=0, tile=TILE):
    n_elements = m * n
    ept = tile * tile
    n_tiles = K_alloc.size
    payload_bits = int(np.sum(K_alloc.flatten() * ept))
    payload_bytes = payload_bits / 8.0
    meta_bytes = n_tiles * 2 * 4
    # K-map: 1 bit per tile for K5/K6 binary choice
    k_map_bytes = (n_tiles + 7) // 8
    total_bytes = payload_bytes + meta_bytes + sidecar_bytes + k_map_bytes
    return {
        "payload_bytes": payload_bytes, "meta_bytes": meta_bytes,
        "sidecar_bytes": sidecar_bytes, "k_map_bytes": k_map_bytes,
        "total_bytes": total_bytes, "n_elements": n_elements,
        "bits_per_element": total_bytes * 8 / n_elements,
    }


# ============================================================================
# Metrics
# ============================================================================

def weight_mse(W, Wq):
    return float(np.mean((W - Wq) ** 2))


# ============================================================================
# Calibration
# ============================================================================

def gen_calibration(n_in, n_samples, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_in, n_samples))
    n_outliers = max(1, n_in // 20)
    outlier_ch = rng.choice(n_in, n_outliers, replace=False)
    X[outlier_ch, :] *= 10.0
    corr = rng.standard_normal((n_in, n_in))
    corr = corr @ corr.T / n_in
    X = corr @ X
    return X


def compute_hessians_slice(W, X):
    N = X.shape[1]
    d_out, d_in = W.shape
    H_X = X @ X.T / N
    Y = W @ X
    H_G = Y @ Y.T / N
    H_X *= d_in / np.trace(H_X)
    H_G *= d_out / np.trace(H_G)
    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)
    return H_X, H_G


# ============================================================================
# Weight loading
# ============================================================================

def load_real_weights():
    w = np.load(WEIGHTS_PATH)
    return {k: w[k] for k in w.files}


def extract_slice(tensor, slice_name, size=SLICE_SIZE):
    d_out, d_in = tensor.shape
    d = min(size, d_out, d_in)
    if slice_name == "first":
        return tensor[:d, :d].astype(np.float64), np.arange(d), np.arange(d)
    elif slice_name == "middle":
        off_out = (d_out - d) // 2
        off_in = (d_in - d) // 2
        return (tensor[off_out:off_out + d, off_in:off_in + d].astype(np.float64),
                np.arange(off_out, off_out + d), np.arange(off_in, off_in + d))
    elif slice_name == "last":
        return (tensor[d_out - d:, d_in - d:].astype(np.float64),
                np.arange(d_out - d, d_out), np.arange(d_in - d, d_in))
    raise ValueError(f"Unknown slice: {slice_name}")


# ============================================================================
# Run a single arm
# ============================================================================

def run_arm(W, H_X, H_G, arm_name, K, sidecar_bytes=0,
            use_bip=False, sg=None, sx=None,
            use_hadamard=False, U=None, V=None):
    m, n = W.shape

    W_t = W.copy()
    if use_bip:
        W_t = sg[:, None] * W_t * sx[None, :]

    if use_hadamard:
        W_t = U @ W_t @ V.T

    if isinstance(K, np.ndarray):
        W_q_t = quantize_matrix_alloc(W_t, K, TILE)
        bytes_info = compute_bytes_alloc(K, m, n, sidecar_bytes, TILE)
    else:
        W_q_t = quantize_matrix_uniform(W_t, K, TILE)
        bytes_info = compute_bytes_uniform(K, m, n, sidecar_bytes, TILE)

    if use_hadamard:
        W_q = U.T @ W_q_t @ V
    else:
        W_q = W_q_t

    if use_bip:
        W_q = inverse_biip(W_q, sg, sx)

    hwe = full_hwe(W, W_q, H_X, H_G)
    mse = weight_mse(W, W_q)

    return {
        "arm": arm_name, "hwe": hwe, "mse": mse,
        "payload_bytes": bytes_info["payload_bytes"],
        "meta_bytes": bytes_info.get("meta_bytes", 0),
        "sidecar_bytes": sidecar_bytes,
        "k_map_bytes": bytes_info.get("k_map_bytes", 0),
        "total_bytes": bytes_info["total_bytes"],
        "bits_per_element": bytes_info["bits_per_element"],
        "K": K if not isinstance(K, np.ndarray) else "alloc",
        "use_bip": use_bip, "use_hadamard": use_hadamard,
    }


# ============================================================================
# Lower convex envelope (proper Graham scan)
# ============================================================================

def lower_convex_envelope(points):
    """Compute lower convex envelope of (rate, distortion) points.
    1. Sort by rate ascending.
    2. Pareto-prune: discard points dominated (higher rate AND higher/equal distortion).
    3. Graham scan with correct sign: pop while cross <= 0 (removes concave points).
    Returns boolean mask: True if point is on the envelope."""
    n = len(points)

    # Sort by rate, then by distortion (lowest first at each rate)
    idx = sorted(range(n), key=lambda i: (points[i][0], points[i][1]))
    sorted_pts = [(points[i][0], points[i][1], i) for i in idx]

    # Pareto prune: keep only points where no other point has both
    # lower-or-equal rate AND lower-or-equal distortion (with at least one strict)
    pareto = []
    best_d = float('inf')
    for r, d, orig_i in sorted_pts:
        if d < best_d:
            pareto.append((r, d, orig_i))
            best_d = d
        # If d == best_d but r is higher, skip (dominated)
        # If d > best_d, skip (dominated by a lower-rate, lower-distortion point)

    # Graham scan for lower convex hull
    # For a convex decreasing curve, intermediate points lie BELOW the endpoint chord.
    # Pop while the last point makes a concave (left) turn: cross <= 0
    hull = []
    for r, d, orig_i in pareto:
        while len(hull) >= 2:
            r1, d1, _ = hull[-2]
            r2, d2, _ = hull[-1]
            # Cross product: (r2-r1)*(d-d1) - (d2-d1)*(r-r1)
            # If <= 0, the last hull point is concave → pop it
            if (r2 - r1) * (d - d1) - (d2 - d1) * (r - r1) <= 0:
                hull.pop()
            else:
                break
        hull.append((r, d, orig_i))

    on_env = [False] * n
    for _, _, orig_i in hull:
        on_env[orig_i] = True
    return on_env


# ============================================================================
# Main experiment
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 80)
    print("R27-EqualRateKLD v3: 16×16 affine-uniform HWE proxy")
    print("Hadamard FIXED: H@diag(signs). Global BiIP scales. Full-HWE alloc.")
    print("WARNING: Local HWE proxy — QSRT showed local metrics can invert KLD.")
    print("=" * 80)

    print("\nLoading real weights...")
    weights = load_real_weights()

    all_results = {}

    for tensor_name in TENSOR_NAMES:
        print(f"\n{'─' * 60}")
        print(f"Tensor: {tensor_name}")
        print(f"{'─' * 60}")

        W_full = weights[tensor_name]
        d_out_full, d_in_full = W_full.shape
        print(f"  Full shape: {W_full.shape}")

        # Global calibration and BiIP scales
        X_full = gen_calibration(d_in_full, N_CALIB, SEED)
        sg_global, sx_global = compute_global_biip_scales(
            W_full.astype(np.float64), X_full, d_out_full, d_in_full)

        # Global sidecar for full tensor
        sc_full_f32 = compute_global_sidecar_bytes(d_out_full, d_in_full, 32)
        sc_full_int16 = compute_global_sidecar_bytes(d_out_full, d_in_full, 16)
        sc_full_int8 = compute_global_sidecar_bytes(d_out_full, d_in_full, 8)
        sc_bpw_f32 = sc_full_f32 * 8 / (d_out_full * d_in_full)
        sc_bpw_int16 = sc_full_int16 * 8 / (d_out_full * d_in_full)
        sc_bpw_int8 = sc_full_int8 * 8 / (d_out_full * d_in_full)
        print(f"  Global BiIP sidecar: f32={sc_bpw_f32:.6f} bpw, int16={sc_bpw_int16:.6f} bpw, int8={sc_bpw_int8:.6f} bpw")

        # Pre-quantize global scales globally (not per-slice) for int8/int16 arms
        sg_global_int16, sx_global_int16 = biip_scales_quantized(
            sg_global.copy(), sx_global.copy(), 16)
        sg_global_int8, sx_global_int8 = biip_scales_quantized(
            sg_global.copy(), sx_global.copy(), 8)

        slice_results = []

        for slice_idx in range(N_SLICES):
            sn = SLICE_NAMES[slice_idx]
            W, row_idx, col_idx = extract_slice(W_full, sn)
            d = W.shape[0]

            if slice_idx == 0:
                print(f"  Slice shape: {W.shape}")
                print(f"  Weight stats: min={W.min():.6f}, max={W.max():.6f}, std={W.std():.6f}")

            X_slice = gen_calibration(d, N_CALIB, SEED)
            H_X, H_G = compute_hessians_slice(W, X_slice)

            W_noise = quantize_matrix_uniform(W, 16, TILE)
            noise_hwe = full_hwe(W, W_noise, H_X, H_G)

            # Extract global BiIP scales for this slice
            sg_slice = sg_global[row_idx].copy()
            sx_slice = sx_global[col_idx].copy()

            # Amortized sidecar for this slice's elements
            # Production: full tensor pays (d_out+d_in)*4 bytes.
            # Slice proportion: (d_out_full*d_in_full) / (d*d) times the full sidecar
            # But actually: per-element sidecar is constant = sc_bpw_f32
            # So for a d×d slice: sidecar_bytes = sc_bpw_f32 * d * d / 8
            sc_slice_f32 = sc_bpw_f32 * d * d / 8.0
            sc_slice_int16 = sc_bpw_int16 * d * d / 8.0
            sc_slice_int8 = sc_bpw_int8 * d * d / 8.0

            # Hadamard sign sidecar: amortized at full-tensor rate
            # Production: (d_out_full + d_in_full) bits for signs = (d_out+d_in)//8 bytes
            # Per-element: (d_out+d_in)/8 / (d_out*d_in) bytes
            # For 128×128 slice: this is ~62× smaller than slice-local (d+d)//8 bytes
            had_sign_bytes_full = (d_out_full + d_in_full + 7) // 8
            had_sign_bpw_full = had_sign_bytes_full * 8 / (d_out_full * d_in_full)
            had_sc_amortized = had_sign_bpw_full * d * d / 8.0

            # Pre-generate Hadamard matrices (same U/V for all K within a seed)
            rht_draws = []
            for rht_seed in range(N_RHT_SEEDS):
                rng_had = np.random.default_rng(SEED + slice_idx * 100 + rht_seed)
                U, signs_out = signed_random_hadamard(d, rng_had)
                V, signs_in = signed_random_hadamard(d, rng_had)
                # Use amortized sidecar (full-tensor rate, not slice-local)
                rht_draws.append((U, V, had_sc_amortized))

            # K5.5-alloc: DP + full-HWE local search
            n_tiles = (d // TILE) ** 2
            k_sum_55 = int(n_tiles * 5.5)
            K_alloc_55, actual_ks = alloc_tile_dp(
                W, H_X, H_G, k_sum_55, TILE, k_range=(5, 6))
            K_alloc_55_ref, alloc_hwe = local_search_refine(
                W, K_alloc_55, H_X, H_G, k_sum_55, TILE, k_range=(5, 6))

            arms = {}

            # Baseline uniform K (no preconditioning)
            for K in [3, 4, 5, 6, 7]:
                arms[f"K{K}"] = run_arm(W, H_X, H_G, f"K{K}", K=K)

            # Complete factorial: {none, BiP, Had, BiP+Had} × {K4, K5, K6}
            for K in [4, 5, 6]:
                # BiP only (global scales, f32 sidecar)
                arms[f"K{K}+BiP"] = run_arm(
                    W, H_X, H_G, f"K{K}+BiP", K=K,
                    use_bip=True, sg=sg_slice, sx=sx_slice,
                    sidecar_bytes=sc_slice_f32)

                # Hadamard only (average over RHT seeds)
                had_hwes = []
                had_mses = []
                for U, V, had_sc in rht_draws:
                    r = run_arm(W, H_X, H_G, f"K{K}+Had", K=K,
                                use_hadamard=True, U=U, V=V,
                                sidecar_bytes=had_sc)
                    had_hwes.append(r["hwe"])
                    had_mses.append(r["mse"])
                bpw_had = (d * d * K / 8.0 + n_tiles * 8 + rht_draws[0][2]) * 8 / (d * d)
                arms[f"K{K}+Had"] = {
                    "arm": f"K{K}+Had",
                    "hwe": float(np.mean(had_hwes)),
                    "hwe_std": float(np.std(had_hwes)),
                    "hwe_per_seed": [float(h) for h in had_hwes],
                    "mse": float(np.mean(had_mses)),
                    "payload_bytes": d * d * K / 8.0,
                    "meta_bytes": n_tiles * 8,
                    "sidecar_bytes": rht_draws[0][2],
                    "k_map_bytes": 0,
                    "total_bytes": d * d * K / 8.0 + n_tiles * 8 + rht_draws[0][2],
                    "bits_per_element": bpw_had,
                    "K": K, "use_bip": False, "use_hadamard": True,
                    "n_rht_seeds": N_RHT_SEEDS,
                }

                # BiP+Had (average over RHT seeds)
                bh_hwes = []
                bh_mses = []
                for U, V, had_sc in rht_draws:
                    total_sc = sc_slice_f32 + had_sc
                    r = run_arm(
                        W, H_X, H_G, f"K{K}+BiP+Had", K=K,
                        use_bip=True, sg=sg_slice, sx=sx_slice,
                        use_hadamard=True, U=U, V=V,
                        sidecar_bytes=total_sc)
                    bh_hwes.append(r["hwe"])
                    bh_mses.append(r["mse"])
                bpw_bh = (d * d * K / 8.0 + n_tiles * 8 + sc_slice_f32 + rht_draws[0][2]) * 8 / (d * d)
                arms[f"K{K}+BiP+Had"] = {
                    "arm": f"K{K}+BiP+Had",
                    "hwe": float(np.mean(bh_hwes)),
                    "hwe_std": float(np.std(bh_hwes)),
                    "hwe_per_seed": [float(h) for h in bh_hwes],
                    "mse": float(np.mean(bh_mses)),
                    "payload_bytes": d * d * K / 8.0,
                    "meta_bytes": n_tiles * 8,
                    "sidecar_bytes": sc_slice_f32 + rht_draws[0][2],
                    "k_map_bytes": 0,
                    "total_bytes": d * d * K / 8.0 + n_tiles * 8 + sc_slice_f32 + rht_draws[0][2],
                    "bits_per_element": bpw_bh,
                    "K": K, "use_bip": True, "use_hadamard": True,
                    "n_rht_seeds": N_RHT_SEEDS,
                }

            # K5.5-alloc (refined with full HWE)
            arms["K5.5-alloc"] = run_arm(
                W, H_X, H_G, "K5.5-alloc", K=K_alloc_55_ref)

            # Equal-rate Hadamard+alloc arm (rotate-then-allocate):
            # For each Had seed: apply Hadamard, recompute Hessians in transformed
            # space, run DP + full-HWE local search in transformed space, quantize,
            # inverse-transform, score HWE in original space.
            # This is the correct "rotate-then-allocate" comparison (not stale-map).
            had_alloc_hwes = []
            had_alloc_mses = []
            for U, V, had_sc in rht_draws:
                # Transform W and Hessians to rotated space
                W_rot = U @ W @ V.T
                H_X_rot = V @ H_X @ V.T
                H_G_rot = U @ H_G @ U.T
                # DP allocation in rotated space
                K_rot_dp, ks_rot = alloc_tile_dp(W_rot, H_X_rot, H_G_rot, k_sum_55, TILE, k_range=(5, 6))
                # Full-HWE local search in rotated space
                K_rot_ref, _ = local_search_refine(
                    W_rot, K_rot_dp, H_X_rot, H_G_rot, k_sum_55, TILE, k_range=(5, 6))
                # Quantize in rotated space
                Wq_rot = quantize_matrix_alloc(W_rot, K_rot_ref, TILE)
                # Inverse transform
                Wq_orig = U.T @ Wq_rot @ V
                # Score HWE in original space
                hwe_rot = full_hwe(W, Wq_orig, H_X, H_G)
                mse_rot = weight_mse(W, Wq_orig)
                had_alloc_hwes.append(hwe_rot)
                had_alloc_mses.append(mse_rot)
            # Bytes: same K-sum + amortized Had signs
            # Use K_rot_ref from last seed for byte computation (K-sum is same)
            ha_bytes = compute_bytes_alloc(K_rot_ref, d, d, rht_draws[0][2], TILE)
            arms["K5.5+Had-alloc"] = {
                "arm": "K5.5+Had-alloc",
                "hwe": float(np.mean(had_alloc_hwes)),
                "hwe_std": float(np.std(had_alloc_hwes)),
                "hwe_per_seed": [float(h) for h in had_alloc_hwes],
                "mse": float(np.mean(had_alloc_mses)),
                "payload_bytes": ha_bytes["payload_bytes"],
                "meta_bytes": ha_bytes["meta_bytes"],
                "sidecar_bytes": ha_bytes["sidecar_bytes"],
                "k_map_bytes": ha_bytes["k_map_bytes"],
                "total_bytes": ha_bytes["total_bytes"],
                "bits_per_element": ha_bytes["bits_per_element"],
                "K": "alloc+had", "use_bip": False, "use_hadamard": True,
                "n_rht_seeds": N_RHT_SEEDS,
                "note": "Rotate-then-allocate: DP+refine recomputed in transformed space per seed",
            }

            # Sidecar sweep: K5+BiP at int16 and int8 (globally quantized scales)
            for sbits, sname, sc_slice_q, sg_gq, sx_gq in [
                (16, "0.25", sc_slice_int16, sg_global_int16, sx_global_int16),
                (8, "0.125", sc_slice_int8, sg_global_int8, sx_global_int8)]:
                sg_q = sg_gq[row_idx].copy()
                sx_q = sx_gq[col_idx].copy()
                arms[f"K5+BiP({sname})"] = run_arm(
                    W, H_X, H_G, f"K5+BiP({sname})", K=5,
                    use_bip=True, sg=sg_q, sx=sx_q,
                    sidecar_bytes=sc_slice_q)

            # BiP+Had sidecar sweep (globally quantized scales)
            for sbits, sname, sc_slice_q, sg_gq, sx_gq in [
                (16, "0.25", sc_slice_int16, sg_global_int16, sx_global_int16),
                (8, "0.125", sc_slice_int8, sg_global_int8, sx_global_int8)]:
                sg_q = sg_gq[row_idx].copy()
                sx_q = sx_gq[col_idx].copy()
                bh_hwes = []
                bh_mses = []
                for U, V, had_sc in rht_draws:
                    total_sc = sc_slice_q + had_sc
                    r = run_arm(
                        W, H_X, H_G, f"K5+BiP+Had({sname})", K=5,
                        use_bip=True, sg=sg_q, sx=sx_q,
                        use_hadamard=True, U=U, V=V,
                        sidecar_bytes=total_sc)
                    bh_hwes.append(r["hwe"])
                    bh_mses.append(r["mse"])
                bpw = (d * d * 5 / 8.0 + n_tiles * 8 + sc_slice_q + rht_draws[0][2]) * 8 / (d * d)
                arms[f"K5+BiP+Had({sname})"] = {
                    "arm": f"K5+BiP+Had({sname})",
                    "hwe": float(np.mean(bh_hwes)),
                    "hwe_std": float(np.std(bh_hwes)),
                    "mse": float(np.mean(bh_mses)),
                    "payload_bytes": d * d * 5 / 8.0,
                    "meta_bytes": n_tiles * 8,
                    "sidecar_bytes": sc_slice_q + rht_draws[0][2],
                    "k_map_bytes": 0,
                    "total_bytes": d * d * 5 / 8.0 + n_tiles * 8 + sc_slice_q + rht_draws[0][2],
                    "bits_per_element": bpw,
                    "K": 5, "use_bip": True, "use_hadamard": True,
                    "n_rht_seeds": N_RHT_SEEDS,
                }

            slice_results.append({
                "slice_idx": slice_idx, "slice_name": sn,
                "row_idx": row_idx.tolist(), "col_idx": col_idx.tolist(),
                "noise_floor_hwe": float(noise_hwe),
                "arms": arms,
                "alloc_k_map": K_alloc_55_ref.tolist(),
                "alloc_actual_k_sum": int(np.sum(K_alloc_55_ref)),
                "alloc_hwe_before_refine": float(full_hwe(
                    W, quantize_matrix_alloc(W, K_alloc_55, TILE), H_X, H_G)),
                "alloc_hwe_after_refine": float(alloc_hwe),
            })

        # Print slice 0 table
        s0 = slice_results[0]
        print(f"\n  {'Arm':<22} {'HWE':>20} {'bpw':>8} {'Sidecar':>8}")
        print(f"  {'─'*22} {'─'*20} {'─'*8} {'─'*8}")
        arm_order = ["K3", "K4", "K4+BiP", "K4+Had", "K4+BiP+Had",
                     "K5", "K5+BiP", "K5+Had", "K5+BiP+Had",
                     "K5.5-alloc", "K6", "K6+BiP", "K6+Had", "K6+BiP+Had", "K7"]
        for name in arm_order:
            if name in s0["arms"]:
                r = s0["arms"][name]
                hwe_str = f"{r['hwe']:.4e}"
                if "hwe_std" in r:
                    hwe_str += f"±{r['hwe_std']:.1e}"
                print(f"  {name:<22} {hwe_str:>22} {r['bits_per_element']:.4f} {r['sidecar_bytes']:>6.0f}B")

        # Multi-slice statistics with win counts
        arm_names = list(slice_results[0]["arms"].keys())
        multi_slice = {}
        for an in arm_names:
            hwes = [sr["arms"][an]["hwe"] for sr in slice_results]
            mses = [sr["arms"][an].get("mse", 0) for sr in slice_results]
            bpws = [sr["arms"][an]["bits_per_element"] for sr in slice_results]
            multi_slice[an] = {
                "hwe_mean": float(np.mean(hwes)),
                "hwe_std": float(np.std(hwes)),
                "hwe_per_slice": [float(h) for h in hwes],
                "mse_mean": float(np.mean(mses)),
                "bpw_mean": float(np.mean(bpws)),
            }

        # Win counts
        win_counts = {}
        for K in [4, 5, 6]:
            base = f"K{K}"; bip = f"K{K}+BiP"; had = f"K{K}+Had"; biphad = f"K{K}+BiP+Had"
            if all(a in multi_slice for a in [base, bip, had, biphad]):
                win_counts[f"K{K}"] = {
                    "bip_beats_base": sum(1 for sr in slice_results if sr["arms"][bip]["hwe"] < sr["arms"][base]["hwe"]),
                    "had_beats_base": sum(1 for sr in slice_results if sr["arms"][had]["hwe"] < sr["arms"][base]["hwe"]),
                    "biphad_beats_base": sum(1 for sr in slice_results if sr["arms"][biphad]["hwe"] < sr["arms"][base]["hwe"]),
                    "n_slices": N_SLICES,
                }

        if "K5+BiP+Had" in multi_slice and "K5.5-alloc" in multi_slice:
            win_counts["K5_BiP_Had_vs_alloc"] = {
                "biphad_beats_alloc": sum(1 for sr in slice_results
                                          if sr["arms"]["K5+BiP+Had"]["hwe"] < sr["arms"]["K5.5-alloc"]["hwe"]),
                "n_slices": N_SLICES,
            }

        # Equal-rate: Had+alloc vs plain alloc
        if "K5.5+Had-alloc" in multi_slice and "K5.5-alloc" in multi_slice:
            win_counts["Had_alloc_vs_alloc"] = {
                "had_alloc_beats_alloc": sum(1 for sr in slice_results
                    if sr["arms"]["K5.5+Had-alloc"]["hwe"] < sr["arms"]["K5.5-alloc"]["hwe"]),
                "n_slices": N_SLICES,
            }

        # Full factorial totals across K4-K6
        total_bip = sum(win_counts.get(f"K{K}", {}).get("bip_beats_base", 0) for K in [4, 5, 6])
        total_had = sum(win_counts.get(f"K{K}", {}).get("had_beats_base", 0) for K in [4, 5, 6])
        total_biphad = sum(win_counts.get(f"K{K}", {}).get("biphad_beats_base", 0) for K in [4, 5, 6])
        win_counts["full_factorial_totals"] = {
            "bip_wins": total_bip, "had_wins": total_had, "biphad_wins": total_biphad,
            "total_comparisons": 3 * N_SLICES * 3,  # 3 K-values × 3 slices × 3 tensors
        }

        all_results[tensor_name] = {
            "slices": slice_results, "multi_slice": multi_slice,
            "win_counts": win_counts,
            "global_sidecar_bpw_f32": sc_bpw_f32,
            "global_sidecar_bpw_int8": sc_bpw_int8,
        }

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSES
    # ═══════════════════════════════════════════════════════════════════

    analysis = {}

    print(f"\n{'=' * 80}")
    print("ANALYSIS 1: Equal-rate comparison (K5+Had vs K5.5-alloc vs K5.5+Had-alloc vs K6)")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        wc = all_results[tn]["win_counts"]
        print(f"\n  {tn}:")
        for name in ["K5", "K5+Had", "K5+BiP+Had", "K5.5-alloc", "K5.5+Had-alloc", "K6"]:
            r = ms.get(name, {})
            if not r:
                continue
            hs = f"{r.get('hwe_mean', 0):.4e}"
            if "hwe_std" in r:
                hs += f"±{r.get('hwe_std', 0):.1e}"
            print(f"    {name:<18} HWE={hs:>22}  bpw={r.get('bpw_mean', 0):.4f}")

        if "K5_BiP_Had_vs_alloc" in wc:
            w = wc["K5_BiP_Had_vs_alloc"]
            print(f"    K5+BiP+Had beats alloc: {w['biphad_beats_alloc']}/{w['n_slices']} slices")
        if "Had_alloc_vs_alloc" in wc:
            w = wc["Had_alloc_vs_alloc"]
            print(f"    K5.5+Had-alloc beats alloc: {w['had_alloc_beats_alloc']}/{w['n_slices']} slices (TRUE equal-rate)")

        analysis.setdefault("equal_rate", {})[tn] = {
            n: ms.get(n, {}).get("hwe_mean", 0)
            for n in ["K5", "K5+Had", "K5+BiP+Had", "K5.5-alloc", "K5.5+Had-alloc", "K6"]
        }

    print(f"\n{'=' * 80}")
    print("ANALYSIS 2: Complete factorial {none, BiP, Had, BiP+Had} × {K4, K5, K6}")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        wc = all_results[tn]["win_counts"]
        print(f"\n  {tn}:")
        print(f"    {'K':>4} {'Base':>12} {'BiP':>12} {'Had':>12} {'B+H':>12} "
              f"{'BiP win':>8} {'Had win':>8} {'B+H win':>8}")
        print(f"    {'─'*4} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
        for K in [4, 5, 6]:
            vals = {n: ms.get(f"K{K}{n}", {}).get("hwe_mean", 0)
                    for n in ["", "+BiP", "+Had", "+BiP+Had"]}
            wcK = wc.get(f"K{K}", {})
            print(f"    K{K}   {vals['']:>12.4e} {vals['+BiP']:>12.4e} {vals['+Had']:>12.4e} {vals['+BiP+Had']:>12.4e} "
                  f"{wcK.get('bip_beats_base', 0):>4}/{wcK.get('n_slices', 0)} "
                  f"{wcK.get('had_beats_base', 0):>4}/{wcK.get('n_slices', 0)} "
                  f"{wcK.get('biphad_beats_base', 0):>4}/{wcK.get('n_slices', 0)}")

        analysis.setdefault("factorial", {})[tn] = {
            f"K{K}": {
                "base": ms.get(f"K{K}", {}).get("hwe_mean", 0),
                "bip": ms.get(f"K{K}+BiP", {}).get("hwe_mean", 0),
                "had": ms.get(f"K{K}+Had", {}).get("hwe_mean", 0),
                "biphad": ms.get(f"K{K}+BiP+Had", {}).get("hwe_mean", 0),
                "wins": wc.get(f"K{K}", {}),
            }
            for K in [4, 5, 6]
        }

    # Full factorial totals across all tensors
    print(f"\n  Full factorial totals (all tensors, K4-K6, {N_SLICES} slices each):")
    tot_bip = sum(all_results[tn]["win_counts"].get(f"K{K}", {}).get("bip_beats_base", 0)
                  for tn in TENSOR_NAMES for K in [4, 5, 6])
    tot_had = sum(all_results[tn]["win_counts"].get(f"K{K}", {}).get("had_beats_base", 0)
                  for tn in TENSOR_NAMES for K in [4, 5, 6])
    tot_biphad = sum(all_results[tn]["win_counts"].get(f"K{K}", {}).get("biphad_beats_base", 0)
                     for tn in TENSOR_NAMES for K in [4, 5, 6])
    total = len(TENSOR_NAMES) * 3 * N_SLICES
    print(f"    BiP wins:      {tot_bip}/{total}")
    print(f"    Had wins:      {tot_had}/{total}")
    print(f"    BiP+Had wins:  {tot_biphad}/{total}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS 3: Lower convex envelope (proper, all arms)")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        points = [(ms[an]["bpw_mean"], ms[an]["hwe_mean"], an) for an in ms]
        env_mask = lower_convex_envelope(points)

        print(f"\n  {tn}:")
        print(f"    {'Arm':<22} {'bpw':>8} {'HWE':>12} {'Envelope?':>10}")
        print(f"    {'─'*22} {'─'*8} {'─'*12} {'─'*10}")
        for (r, d, name), on_e in sorted(zip(points, env_mask), key=lambda x: x[0][0]):
            print(f"    {name:<22} {r:>8.4f} {d:>12.4e} {'★ YES' if on_e else '— no':>10}")

        analysis.setdefault("convex_envelope", {})[tn] = [
            {"arm": n, "bpw": r, "hwe": d, "on_envelope": e}
            for (r, d, n), e in zip(points, env_mask)
        ]

    print(f"\n{'=' * 80}")
    print("ANALYSIS 4: Chord test — D(K5+BiP+Had) vs interp(D(K5), D(K6))")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        k5 = ms["K5"]["hwe_mean"]; k5bh = ms["K5+BiP+Had"]["hwe_mean"]; k6 = ms["K6"]["hwe_mean"]
        bpw5 = ms["K5"]["bpw_mean"]; bpw5bh = ms["K5+BiP+Had"]["bpw_mean"]; bpw6 = ms["K6"]["bpw_mean"]
        frac = (bpw5bh - bpw5) / (bpw6 - bpw5) if bpw6 > bpw5 else 0.5
        interp_d = k5 * (1 - frac) + k6 * frac
        below = k5bh < interp_d

        print(f"\n  {tn}:")
        print(f"    D(K5)={k5:.4e}, D(K5+BiP+Had)={k5bh:.4e}, D(K6)={k6:.4e}")
        print(f"    Interp at bpw={bpw5bh:.4f} (frac={frac:.3f}): {interp_d:.4e}")
        print(f"    K5+BiP+Had {'BELOW' if below else 'ABOVE'} chord by {(1-k5bh/interp_d)*100:.1f}%")

        analysis.setdefault("chord_test", {})[tn] = {
            "D_K5": k5, "D_K5BiPHad": k5bh, "D_K6": k6,
            "interp_D": interp_d, "below_chord": below,
            "improvement_pct": float((1 - k5bh / interp_d) * 100) if interp_d > 0 else 0,
        }

    print(f"\n{'=' * 80}")
    print("ANALYSIS 5: K4+BiP+Had vs K5 (expert's explicit request)")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        k4 = ms["K4"]["hwe_mean"]; k4b = ms["K4+BiP"]["hwe_mean"]
        k4h = ms["K4+Had"]["hwe_mean"]; k4bh = ms["K4+BiP+Had"]["hwe_mean"]; k5 = ms["K5"]["hwe_mean"]

        print(f"\n  {tn}:")
        print(f"    D(K4)={k4:.4e}, D(K4+BiP)={k4b:.4e}, D(K4+Had)={k4h:.4e}, D(K4+BiP+Had)={k4bh:.4e}, D(K5)={k5:.4e}")
        for name, val in [("K4+BiP", k4b), ("K4+Had", k4h), ("K4+BiP+Had", k4bh)]:
            if k4 > 0 and k5 > 0:
                gap = (1 - val / k4) * 100
                ratio = val / k5
                print(f"      {name}: closes {gap:.1f}% of K4→K5 gap, ratio to K5 = {ratio:.3f}×")

        analysis.setdefault("k4_test", {})[tn] = {
            "D_K4": k4, "D_K4BiP": k4b, "D_K4Had": k4h, "D_K4BiPHad": k4bh, "D_K5": k5,
        }

    print(f"\n{'=' * 80}")
    print("ANALYSIS 6: Marginal value ΔD/ΔR")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        k5 = ms["K5"]["hwe_mean"]; k5bh = ms["K5+BiP+Had"]["hwe_mean"]; k6 = ms["K6"]["hwe_mean"]
        bpw5 = ms["K5"]["bpw_mean"]; bpw5bh = ms["K5+BiP+Had"]["bpw_mean"]; bpw6 = ms["K6"]["bpw_mean"]

        dd_k56 = (k5 - k6) / (bpw6 - bpw5) if bpw6 > bpw5 else 0
        dd_k5bh = (k5 - k5bh) / (bpw5bh - bpw5) if bpw5bh > bpw5 else 0
        ratio = dd_k5bh / dd_k56 if dd_k56 > 0 else 0

        print(f"\n  {tn}:")
        print(f"    K5→K6:         ΔD/ΔR = {dd_k56:.4e}/bpw ({bpw6-bpw5:.4f} bpw cost)")
        print(f"    K5→K5+BiP+Had: ΔD/ΔR = {dd_k5bh:.4e}/bpw ({bpw5bh-bpw5:.4f} bpw cost)")
        print(f"    Ratio = {ratio:.2f}×")

        analysis.setdefault("marginal_value", {})[tn] = {
            "dd_k56": dd_k56, "dd_k5biphad": dd_k5bh, "ratio": ratio,
        }

    print(f"\n{'=' * 80}")
    print("ANALYSIS 7: Sidecar rate sweep (BiP and BiP+Had)")
    print(f"{'=' * 80}")

    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        k5 = ms["K5"]["hwe_mean"]; k6 = ms["K6"]["hwe_mean"]

        print(f"\n  {tn}:")
        print(f"    {'Arm':<22} {'HWE':>12} {'bpw':>8} {'% gap closed':>12}")
        print(f"    {'─'*22} {'─'*12} {'─'*8} {'─'*12}")
        for an in ["K5+BiP", "K5+BiP(0.25)", "K5+BiP(0.125)",
                    "K5+BiP+Had", "K5+BiP+Had(0.25)", "K5+BiP+Had(0.125)"]:
            if an in ms:
                hwe = ms[an]["hwe_mean"]; bpw = ms[an]["bpw_mean"]
                gap = (1 - max(hwe - k6, 0) / max(k5 - k6, 1e-18)) * 100 if k5 > k6 else 0
                print(f"    {an:<22} {hwe:>12.4e} {bpw:>8.4f} {gap:>10.1f}%")

        analysis.setdefault("sidecar_sweep", {})[tn] = {"K5_hwe": k5, "K6_hwe": k6}

    # ─── Summary ───
    print(f"\n{'=' * 80}")
    print("SUMMARY: K5+BiP+Had vs K6 (mean over slices)")
    print(f"{'=' * 80}")
    print(f"  {'Tensor':<15} {'D(K5)':>12} {'D(K5+B+H)':>12} {'D(K5.5a)':>12} {'D(K6)':>12} {'B+H/K6':>8} {'Env?':>6}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*6}")
    for tn in TENSOR_NAMES:
        ms = all_results[tn]["multi_slice"]
        k5 = ms["K5"]["hwe_mean"]; k5bh = ms["K5+BiP+Had"]["hwe_mean"]
        k55 = ms["K5.5-alloc"]["hwe_mean"]; k6 = ms["K6"]["hwe_mean"]
        ce = analysis.get("convex_envelope", {}).get(tn, [])
        on_env = any(p["arm"] == "K5+BiP+Had" and p["on_envelope"] for p in ce)
        ratio = k5bh / k6 if k6 > 0 else 0
        print(f"  {tn:<15} {k5:>12.4e} {k5bh:>12.4e} {k55:>12.4e} {k6:>12.4e} {ratio:>8.3f} {'★' if on_env else '—':>6}")

    print(f"\n  WARNING: Local HWE proxy only. QSRT review showed local metrics")
    print(f"  can INVERT end-to-end KLD. All findings require KLD validation on aiboss.")

    # ─── Save ───
    output = {
        "config": {
            "tile_size": TILE, "slice_size": SLICE_SIZE, "n_slices": N_SLICES,
            "seed": SEED, "n_calib": N_CALIB, "n_rht_seeds": N_RHT_SEEDS,
            "tensors": TENSOR_NAMES, "slices": SLICE_NAMES,
        },
        "tensors": all_results,
        "analysis": analysis,
        "elapsed_seconds": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
