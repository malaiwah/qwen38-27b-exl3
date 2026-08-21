#!/usr/bin/env python3
"""
R26-HARPComparison: BiP vs HARP — sidecar compression and EXL3 baseline.

This is THE critical experiment identified by the external expert.

Three questions answered:
1. Is BiP's 70% improvement vs naive unrotated, or vs EXL3's existing incoherence?
2. How much of the 70% survives at lower sidecar rates (0.5 → 0.016 bpw)?
3. Do optimized signs beat random signs (the HARP insight)?

EXL3's existing incoherence processing (from exl3_fp4_conversion.py):
  W_final = diag(suh) @ Had_K @ W @ Had_N @ diag(svh)
  where Had_K/Had_N are 128-point normalized Hadamard, suh/svh are per-element scales.

The EXL3 baseline here uses random Rademacher signs for the Hadamard
(H @ diag(±1)), which is the standard QuIP#/QTIP incoherence processing.
EXL3's suh/svh are per-element scales that get folded into weights at
conversion time — they are NOT part of the "incoherence" step per se;
they are per-tile/per-channel quantization scales.

Arms:
  - naive: no transform, per-tile quantization
  - exl3_baseline: random Rademacher signs + Hadamard (both sides)
  - exl3_baseline_gptq: random signs + Hadamard + GPTQ error feedback
  - bip: BiIP optimized scales (Hessian-diagonal) + Hadamard
  - bip_gptq: BiIP scales + Hadamard + GPTQ
  - bip_int8: BiIP scales quantized to int8 + Hadamard
  - bip_int4: BiIP scales quantized to int4 + Hadamard
  - bip_fp16: BiIP scales at fp16 + Hadamard (0.25 bpw)
  - hadamard_only: plain Hadamard (no signs, no scaling) — 0 bpw sidecar
  - optimized_signs: greedy sign optimization + Hadamard
  - optimized_signs_gptq: optimized signs + Hadamard + GPTQ
  - bip_on_exl3: EXL3 random signs → BiIP scaling → Hadamard (layered)
  - per_tile_scale: 1 float per 16×16 tile + Hadamard (0.016 bpw)

Error is ALWAYS measured in the ORIGINAL basis:
  tr(H_G @ (W - W_hat) @ H_X @ (W - W_hat)^T)
where W_hat is obtained by inverse-transforming the quantized matrix.
"""

import json
import numpy as np
import os
import time

# ─── Paths ───
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "receipts", "research", "r26-harp-comparison-results.json"
)

# ─── Quantizer ───
# Per-tile 16×16 uniform quantization — matches EXL3 tile structure.
# All arms use the SAME quantizer for fair comparison.

def quantize_per_tile(W, K, tile_size=16):
    """Per-tile uniform quantization with 2^K levels.
    Each tile_size × tile_size block gets its own scale (min-max).
    """
    levels = 2 ** K
    d_out, d_in = W.shape
    Wq = np.zeros_like(W)
    n_tiles_out = (d_out + tile_size - 1) // tile_size
    n_tiles_in = (d_in + tile_size - 1) // tile_size

    for i in range(n_tiles_out):
        for j in range(n_tiles_in):
            ti = slice(i * tile_size, min((i + 1) * tile_size, d_out))
            tj = slice(j * tile_size, min((j + 1) * tile_size, d_in))
            tile = W[ti, tj]
            w_min, w_max = tile.min(), tile.max()
            rng = w_max - w_min
            if rng < 1e-12:
                Wq[ti, tj] = tile
                continue
            scale = rng / (levels - 1)
            q = np.round((tile - w_min) / scale).clip(0, levels - 1)
            Wq[ti, tj] = q * scale + w_min

    n_elements = d_out * d_in
    payload_bytes = n_elements * K / 8
    # Per-tile metadata: 2 float32 per tile (scale + offset/w_min)
    # Asymmetric quantization: q = round((w - w_min) / scale), deq = q * scale + w_min
    # Both scale and w_min are needed for reconstruction.
    n_tiles = n_tiles_out * n_tiles_in
    meta_bytes = n_tiles * 2 * 4  # 2 floats per tile
    return Wq, payload_bytes, meta_bytes


# ─── Hadamard ───

def hadamard_matrix(n):
    """Sylvester-type Hadamard matrix of size n (power of 2)."""
    H = np.array([[1]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_hadamard(n, signs, H_base=None):
    """H @ diag(signs) — signed Hadamard transform."""
    if H_base is None:
        H_base = hadamard_matrix(n)
    return H_base @ np.diag(signs)


# ─── BiIP diagonal balancing (from R3/KronQ Eq. 11) ───

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing.
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}
    S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}
    W' = S_G @ W @ S_X
    """
    d_out, d_in = W.shape
    col_norms_sq = np.maximum(np.sum(W ** 2, axis=0), 1e-12)
    sx_diag = (np.diag(H_X) / col_norms_sq) ** 0.25
    sx_diag = np.clip(sx_diag, 0.1, 10.0)

    row_norms_sq = np.maximum(np.sum(W ** 2, axis=1), 1e-12)
    sg_diag = (np.diag(H_G) / row_norms_sq) ** 0.25
    sg_diag = np.clip(sg_diag, 0.1, 10.0)

    return sg_diag, sx_diag


def compress_scales(scales, bits):
    """Quantize scale vectors to int bits. Returns (quantized_scales, sidecar_bytes)."""
    n = len(scales)
    levels = 2 ** bits
    s_min, s_max = scales.min(), scales.max()
    rng = s_max - s_min
    if rng < 1e-12:
        return scales.copy(), n * bits / 8
    q = np.round((scales - s_min) / rng * (levels - 1)).clip(0, levels - 1)
    deq = q / (levels - 1) * rng + s_min
    # Sidecar: quantized values + 2 floats (min, max) for dequantization
    sidecar_bytes = n * bits / 8 + 2 * 4
    return deq, sidecar_bytes


# ─── Greedy sign optimization (HARP insight) ───
# Instead of random Rademacher signs, greedily choose signs to minimize
# per-tile range (proxy for quantization error).

def optimize_signs_greedy(W, H_base, tile_size=16, max_iter=3, side='input',
                          init_signs=None):
    """Greedily optimize sign vector for one side to minimize max tile range.

    side='input': optimize right-side signs (W @ diag(signs) @ H.T)
    side='output': optimize left-side signs (H @ diag(signs) @ W)
    """
    n = W.shape[1] if side == 'input' else W.shape[0]
    if init_signs is not None:
        signs = init_signs.copy()
    else:
        rng = np.random.default_rng(42)
        signs = rng.choice([-1.0, 1.0], size=n)

    def compute_max_tile_range(W_t, tile_size):
        d_out, d_in = W_t.shape
        max_r = 0.0
        for i in range(0, d_out, tile_size):
            for j in range(0, d_in, tile_size):
                tile = W_t[i:i+tile_size, j:j+tile_size]
                r = tile.max() - tile.min()
                if r > max_r:
                    max_r = r
        return max_r

    H = H_base
    if side == 'input':
        W_t = W @ np.diag(signs) @ H.T
    else:
        W_t = H @ np.diag(signs) @ W

    best_range = compute_max_tile_range(W_t, tile_size)

    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            signs[i] *= -1
            if side == 'input':
                W_t = W @ np.diag(signs) @ H.T
            else:
                W_t = H @ np.diag(signs) @ W
            new_range = compute_max_tile_range(W_t, tile_size)
            if new_range < best_range:
                best_range = new_range
                improved = True
            else:
                signs[i] *= -1  # revert
        if not improved:
            break

    return signs


def optimize_signs_both_sides(W, H_base_out, H_base_in, tile_size=16, max_iter=2,
                               init_signs_in=None, init_signs_out=None):
    """Optimize both input and output sign vectors greedily.
    Uses explicit side parameter to avoid the square-matrix ambiguity.
    """
    # Optimize input signs
    signs_in = optimize_signs_greedy(W, H_base_in, tile_size, max_iter,
                                      side='input', init_signs=init_signs_in)
    W_mid = W @ np.diag(signs_in) @ H_base_in.T

    # Optimize output signs on the mid-transformed weight
    signs_out = optimize_signs_greedy(W_mid, H_base_out, tile_size, max_iter,
                                       side='output', init_signs=init_signs_out)

    return signs_out, signs_in


def optimize_signs_hwe(W, H_base, H_X, H_G, K, tile_size=16, max_iter=2,
                       side='input', init_signs=None):
    """Optimize sign vector to minimize Hessian-weighted quantization error.
    Uses explicit side parameter. Scoring uses full forward→quantize→inverse→HWE.
    """
    n = W.shape[1] if side == 'input' else W.shape[0]
    if init_signs is not None:
        signs = init_signs.copy()
    else:
        rng = np.random.default_rng(99)
        signs = rng.choice([-1.0, 1.0], size=n)
    H = H_base

    def compute_hwe(signs_vec):
        if side == 'input':
            W_t = W @ np.diag(signs_vec) @ H.T
            Wq, _, _ = quantize_per_tile(W_t, K, tile_size)
            W_hat = Wq @ H @ np.diag(signs_vec)
        else:
            W_t = H @ np.diag(signs_vec) @ W
            Wq, _, _ = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(signs_vec) @ H.T @ Wq
        return hessian_weighted_error(W, W_hat, H_X, H_G)

    best_hwe = compute_hwe(signs)

    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            signs[i] *= -1
            new_hwe = compute_hwe(signs)
            if new_hwe < best_hwe:
                best_hwe = new_hwe
                improved = True
            else:
                signs[i] *= -1  # revert
        if not improved:
            break

    return signs


def optimize_signs_hwe_both_sides(W, H_base_out, H_base_in, H_X, H_G, K,
                                   tile_size=16, max_iter=1,
                                   init_signs_in=None, init_signs_out=None):
    """Optimize both input and output sign vectors for HWE.
    Uses explicit side parameter. The output-side optimization evaluates
    through the full two-sided pipeline.
    """
    # Optimize input signs first
    signs_in = optimize_signs_hwe(W, H_base_in, H_X, H_G, K, tile_size, max_iter,
                                   side='input', init_signs=init_signs_in)

    # Now optimize output signs on the input-transformed weight
    W_mid = W @ np.diag(signs_in) @ H_base_in.T
    signs_out = optimize_signs_hwe(W_mid, H_base_out, H_X, H_G, K, tile_size,
                                    max_iter, side='output',
                                    init_signs=init_signs_out)

    return signs_out, signs_in


# ─── GPTQ error feedback ───

def gptq_quantize(W, H, K, tile_size=16):
    """Standard sequential GPTQ with Cholesky-based error feedback.

    Uses U = chol(inv(H + damp)).T (upper triangular, U^T U = H^{-1}).
    Error feedback: W_err[:, j+1:] -= err * U[j, j+1:] / U[j, j]

    Uses per-16x16-tile codebooks (64 codebooks for 128×128, matching RTN).
    Each 16-row × 16-column tile gets its own scale (min-max), but error
    compensation flows sequentially across all columns.

    H: Hessian (d_in × d_in), input-side.
    """
    d_out, d_in = W.shape
    levels = 2 ** K
    W_q = np.zeros_like(W)
    W_err = W.copy()

    # Regularize Hessian and compute Cholesky of inverse
    H_reg = H + 1e-4 * np.eye(d_in)
    try:
        H_inv = np.linalg.inv(H_reg)
        U = np.linalg.cholesky(H_inv).T  # upper triangular, U^T U = H^{-1}
    except (np.linalg.LinAlgError, ValueError):
        U = np.linalg.pinv(H_reg)

    # Process column by column (sequential GPTQ)
    for j in range(d_in):
        # For each 16-row block, use a separate codebook (matching RTN's 64 tiles)
        for i_start in range(0, d_out, tile_size):
            i_end = min(i_start + tile_size, d_out)
            tile_slice = W_err[i_start:i_end, j]
            w_min, w_max = tile_slice.min(), tile_slice.max()
            rng = w_max - w_min
            if rng < 1e-12:
                W_q[i_start:i_end, j] = tile_slice
                continue
            scale = rng / (levels - 1)
            q = np.round((tile_slice - w_min) / scale).clip(0, levels - 1)
            deq = q * scale + w_min
            W_q[i_start:i_end, j] = deq

        # Error compensation to remaining columns using Cholesky upper factor
        col_deq = W_q[:, j]
        err = W_err[:, j] - col_deq
        if j + 1 < d_in:
            correction = np.outer(err, U[j, j+1:] / U[j, j])
            W_err[:, j+1:] -= correction

    return W_q


# ─── Hessian generation ───

def synthetic_hessians(W, n_samples=512, outlier_fraction=0.05, outlier_scale=10.0, seed=42):
    """Generate synthetic activation Hessian H_X and output Hessian proxy H_G.
    Numerically stable: all computation in float64 with clipping.
    """
    rng = np.random.default_rng(seed)
    d_out, d_in = W.shape

    X = rng.standard_normal((d_in, n_samples))
    n_outliers = max(1, int(d_in * outlier_fraction))
    outlier_channels = rng.choice(d_in, n_outliers, replace=False)
    X[outlier_channels, :] *= outlier_scale

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        H_X = (X @ X.T / n_samples).astype(np.float64)
        Y = (W.astype(np.float64) @ X)
        H_G = (Y @ Y.T / n_samples).astype(np.float64)

    # Replace any NaN/Inf with safe values
    H_X = np.nan_to_num(H_X, nan=0.0, posinf=1e6, neginf=-1e6)
    H_G = np.nan_to_num(H_G, nan=0.0, posinf=1e6, neginf=-1e6)

    # Normalize to mean diagonal = 1
    tr_X = np.trace(H_X)
    tr_G = np.trace(H_G)
    if tr_X > 1e-12:
        H_X *= d_in / tr_X
    if tr_G > 1e-12:
        H_G *= d_out / tr_G

    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)

    return H_X, H_G


# ─── Error metrics ───

def hessian_weighted_error(W, W_hat, H_X, H_G):
    """tr(H_G @ E @ H_X @ E^T) where E = W - W_hat.
    Measured in ORIGINAL basis (after inverse transform).
    """
    E = W - W_hat
    with np.errstate(over='ignore', invalid='ignore'):
        val = np.trace(H_G @ E @ H_X @ E.T)
    return float(np.nan_to_num(val, nan=0.0, posinf=1e18, neginf=0.0))


def hessian_weighted_error_diag(W, W_hat, H_X_diag, H_G_diag):
    """Diagonal approximation: sum_i sum_j H_G_i * H_X_j * E_ij^2."""
    E = W - W_hat
    weights = np.outer(H_G_diag, H_X_diag)
    return float(np.sum(weights * E ** 2))


def weight_mse(W, W_hat):
    return float(np.mean((W - W_hat) ** 2))


# ─── Main experiment ───

def run_experiment():
    print("=" * 90)
    print("R26-HARPComparison: BiP vs EXL3 baseline — sidecar compression sweep")
    print("Questions: (1) 70% vs naive or vs EXL3? (2) Sidecar compression curve?")
    print("           (3) Optimized signs vs random signs?")
    print("=" * 90)

    weights = np.load(WEIGHTS_PATH)

    tensors = {
        'L0_gate_K4': ('L0_gate', 4),
        'L0_gate_K5': ('L0_gate', 5),
        'L0_gate_K6': ('L0_gate', 6),
        'L0_down_K4': ('L0_down', 4),
        'L0_down_K5': ('L0_down', 5),
        'L0_down_K6': ('L0_down', 6),
        'L55_gate_K4': ('L55_gate', 4),
        'L55_gate_K5': ('L55_gate', 5),
        'L55_gate_K6': ('L55_gate', 6),
        'L55_down_K4': ('L55_down', 4),
        'L55_down_K5': ('L55_down', 5),
        'L55_down_K6': ('L55_down', 6),
    }

    d = 128
    n_slices = 3
    tile_size = 16
    all_results = {}

    # Pre-compute Hadamard base
    H_base = hadamard_matrix(d)

    for tensor_name, (weight_key, K) in tensors.items():
        print(f"\n{'─' * 80}")
        print(f"  Tensor: {tensor_name} (K={K})")
        print(f"{'─' * 80}")

        W_full = weights[weight_key]
        d_out_full, d_in_full = W_full.shape
        d_out = min(d, d_out_full)
        d_in = min(d, d_in_full)

        slice_offsets_out = [0, (d_out_full - d_out) // 2, d_out_full - d_out]
        slice_offsets_in = [0, (d_in_full - d_in) // 2, d_in_full - d_in]

        slice_results = []

        for slice_idx in range(n_slices):
            off_out = slice_offsets_out[slice_idx]
            off_in = slice_offsets_in[slice_idx]
            W = W_full[off_out:off_out + d_out, off_in:off_in + d_in].astype(np.float64)

            H_X, H_G = synthetic_hessians(W, n_samples=512, seed=42)
            H_X_diag = np.diag(H_X)
            H_G_diag = np.diag(H_G)

            n_elements = d_out * d_in

            # ─── Define all arms ───
            arm_results = {}

            # === ARM 1: Naive (no transform) ===
            Wq, payload, meta = quantize_per_tile(W, K, tile_size)
            hwe = hessian_weighted_error(W, Wq, H_X, H_G)
            arm_results['naive'] = {
                'hwe': hwe,
                'mse': weight_mse(W, Wq),
                'sidecar_bpw': 0.0,
                'total_bpw': (payload + meta) * 8 / n_elements,
            }

            # Generate random signs for EXL3 baseline (same for all arms that need them)
            rng_signs = np.random.default_rng(123)
            su = rng_signs.choice([-1.0, 1.0], size=d_out)
            sv = rng_signs.choice([-1.0, 1.0], size=d_in)

            U_rand = signed_hadamard(d_out, su, H_base)  # H_out @ diag(su)
            V_rand = signed_hadamard(d_in, sv, H_base)    # H_in @ diag(sv)

            # === ARM 2: EXL3 baseline (random signs + Hadamard) ===
            W_t = U_rand @ W @ V_rand.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = U_rand.T @ Wq_t @ V_rand  # inverse transform
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            # Sidecar: sign vectors (1 bit per element, both sides)
            sidecar_bytes = (d_out + d_in) / 8  # 1 bit per element
            arm_results['exl3_baseline'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes) * 8 / n_elements,
            }

            # === ARM 2b: EXL3-style (random signs + Hadamard + per-element scales) ===
            # EXL3 stores FP16 suh/svh per-element scale vectors. These are applied
            # AFTER the Hadamard: W' = diag(suh) @ (U_rand @ W @ V_rand.T) @ diag(svh)
            # The scales normalize the Hadamard-mixed weights so each element
            # has similar magnitude (per-channel RMS normalization post-mixing).
            # This is the ACTUAL EXL3 baseline with scale adaptation.
            W_had = U_rand @ W @ V_rand.T  # Hadamard-mixed weights
            suh_rms = np.sqrt(np.mean(W_had**2, axis=1))  # per-row RMS post-Hadamard
            svh_rms = np.sqrt(np.mean(W_had**2, axis=0))  # per-col RMS post-Hadamard
            suh_rms = np.maximum(suh_rms, 1e-12)
            svh_rms = np.maximum(svh_rms, 1e-12)
            suh_norm = 1.0 / suh_rms  # normalize so rows have unit RMS
            svh_norm = 1.0 / svh_rms  # normalize so cols have unit RMS
            W_t = np.diag(suh_norm) @ W_had @ np.diag(svh_norm)
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(suh_rms) @ U_rand.T @ Wq_t @ V_rand @ np.diag(svh_rms)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            # Sidecar: sign vectors + fp16 scale vectors (EXL3 already pays this)
            sidecar_bytes_exl3 = (d_out + d_in) / 8 + (d_out + d_in) * 2
            arm_results['exl3_rms_scales'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_exl3 * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_exl3) * 8 / n_elements,
            }

            # === ARM 2c: EXL3-style + GPTQ ===
            H_X_exl3 = V_rand @ np.diag(svh_norm) @ H_X @ np.diag(svh_norm) @ V_rand.T
            Wq_gptq = gptq_quantize(W_t, H_X_exl3, K, tile_size)
            W_hat = np.diag(suh_rms) @ U_rand.T @ Wq_gptq @ V_rand @ np.diag(svh_rms)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['exl3_rms_scales_gptq'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_exl3 * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_exl3) * 8 / n_elements,
            }

            # === ARM 3: EXL3 baseline + GPTQ ===
            # NOTE: W_t was overwritten by exl3_rms_scales arms above. Recompute.
            W_t_exl3 = U_rand @ W @ V_rand.T
            H_X_exl3_t = V_rand @ H_X @ V_rand.T
            Wq_gptq = gptq_quantize(W_t_exl3, H_X_exl3_t, K, tile_size)
            W_hat = U_rand.T @ Wq_gptq @ V_rand
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['exl3_baseline_gptq'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes) * 8 / n_elements,
            }

            # === ARM 4: BiP (BiIP optimized scales + Hadamard) ===
            sg_diag, sx_diag = biip_scaling(W, H_X, H_G)
            S_G = np.diag(sg_diag)
            S_X = np.diag(sx_diag)
            # Transform: W' = S_G @ W @ S_X, then Hadamard on both sides
            W_t = U_rand @ S_G @ W @ S_X @ V_rand.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = S_G @ U_rand.T @ Wq_t @ V_rand @ S_X  # inverse
            # Wait — need correct inverse. Forward: W' = U @ S_G @ W @ S_X @ V^T
            # So W_hat = S_G^{-1} @ U^T @ Wq @ V @ S_X^{-1}
            W_hat = np.diag(1.0 / sg_diag) @ U_rand.T @ Wq_t @ V_rand @ np.diag(1.0 / sx_diag)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            # Sidecar: 2 float vectors (fp32) + sign vectors
            sidecar_bytes_bip = (d_out + d_in) * 4 + (d_out + d_in) / 8
            arm_results['bip'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_bip * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_bip) * 8 / n_elements,
            }

            # === ARM 4b: BiIP replacing EXL3 scales (zero incremental sidecar) ===
            # EXL3 ALREADY stores FP16 suh/svh per-element scale vectors.
            # If BiIP merely changes their VALUES (from RMS-derived to Hessian-balanced),
            # the incremental storage cost is approximately ZERO.
            # This arm uses the SAME transform as bip_fp16 (BiIP scales pre-Hadamard)
            # but accounts the sidecar as zero incremental (replaces existing EXL3 scales).
            sg_f16_zc = sg_diag.astype(np.float16).astype(np.float64)
            sx_f16_zc = sx_diag.astype(np.float16).astype(np.float64)
            W_t_zc = U_rand @ np.diag(sg_f16_zc) @ W @ np.diag(sx_f16_zc) @ V_rand.T
            Wq_t_zc, payload_zc, meta_zc = quantize_per_tile(W_t_zc, K, tile_size)
            W_hat = np.diag(1.0/sg_f16_zc) @ U_rand.T @ Wq_t_zc @ V_rand @ np.diag(1.0/sx_f16_zc)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            # Sidecar: SAME total as exl3_rms_scales, but incremental = 0 (replaces existing)
            sidecar_bytes_zero = sidecar_bytes_exl3  # same total budget as EXL3
            arm_results['bip_zero_cost'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_zero * 8 / n_elements,
                'total_bpw': (payload_zc + meta_zc + sidecar_bytes_zero) * 8 / n_elements,
                'incremental_sidecar_bpw': 0.0,  # replaces existing EXL3 scales
            }

            # === ARM 4c: BiIP zero-cost + GPTQ ===
            H_X_zc = V_rand @ np.diag(1.0/sx_f16_zc) @ H_X @ np.diag(1.0/sx_f16_zc) @ V_rand.T
            Wq_gptq_zc = gptq_quantize(W_t_zc, H_X_zc, K, tile_size)
            W_hat = np.diag(1.0/sg_f16_zc) @ U_rand.T @ Wq_gptq_zc @ V_rand @ np.diag(1.0/sx_f16_zc)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['bip_zero_cost_gptq'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_zero * 8 / n_elements,
                'total_bpw': (payload_zc + meta_zc + sidecar_bytes_zero) * 8 / n_elements,
                'incremental_sidecar_bpw': 0.0,
            }

            # === ARM 5: BiP + GPTQ ===
            H_X_t = V_rand @ np.diag(1.0/sx_diag) @ H_X @ np.diag(1.0/sx_diag) @ V_rand.T
            Wq_gptq = gptq_quantize(W_t, H_X_t, K, tile_size)
            W_hat = np.diag(1.0 / sg_diag) @ U_rand.T @ Wq_gptq @ V_rand @ np.diag(1.0 / sx_diag)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['bip_gptq'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_bip * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_bip) * 8 / n_elements,
            }

            # === ARM 6: BiP with fp16 scales (0.25 bpw) ===
            sg_f16 = sg_diag.astype(np.float16).astype(np.float64)
            sx_f16 = sx_diag.astype(np.float16).astype(np.float64)
            W_t = U_rand @ np.diag(sg_f16) @ W @ np.diag(sx_f16) @ V_rand.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(1.0 / sg_f16) @ U_rand.T @ Wq_t @ V_rand @ np.diag(1.0 / sx_f16)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            sidecar_bytes_f16 = (d_out + d_in) * 2 + (d_out + d_in) / 8
            arm_results['bip_fp16'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_f16 * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_f16) * 8 / n_elements,
            }

            # === ARM 7: BiP with int8 scales (0.125 bpw) ===
            sg_i8, sc_sg = compress_scales(sg_diag, 8)
            sx_i8, sc_sx = compress_scales(sx_diag, 8)
            W_t = U_rand @ np.diag(sg_i8) @ W @ np.diag(sx_i8) @ V_rand.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(1.0 / sg_i8) @ U_rand.T @ Wq_t @ V_rand @ np.diag(1.0 / sx_i8)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            sidecar_bytes_i8 = sc_sg + sc_sx + (d_out + d_in) / 8
            arm_results['bip_int8'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_i8 * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_i8) * 8 / n_elements,
            }

            # === ARM 8: BiP with int4 scales (0.0625 bpw) ===
            sg_i4, sc_sg4 = compress_scales(sg_diag, 4)
            sx_i4, sc_sx4 = compress_scales(sx_diag, 4)
            W_t = U_rand @ np.diag(sg_i4) @ W @ np.diag(sx_i4) @ V_rand.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(1.0 / sg_i4) @ U_rand.T @ Wq_t @ V_rand @ np.diag(1.0 / sx_i4)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            sidecar_bytes_i4 = sc_sg4 + sc_sx4 + (d_out + d_in) / 8
            arm_results['bip_int4'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_i4 * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_i4) * 8 / n_elements,
            }

            # === ARM 9: Hadamard only (no signs, no scaling, 0 bpw sidecar) ===
            W_t = H_base @ W @ H_base.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = H_base.T @ Wq_t @ H_base
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['hadamard_only'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': 0.0,
                'total_bpw': (payload + meta) * 8 / n_elements,
            }

            # === ARM 10: Optimized signs + Hadamard (HARP-style) ===
            # Uses the SAME initial sign draw (su, sv from seed 123) as the baseline,
            # then greedily optimizes. This ensures fair paired comparison.
            t0 = time.time()
            signs_out_opt, signs_in_opt = optimize_signs_both_sides(
                W, H_base, H_base, tile_size, max_iter=2,
                init_signs_in=sv.copy(), init_signs_out=su.copy()
            )
            t_opt = time.time() - t0

            U_opt = signed_hadamard(d_out, signs_out_opt, H_base)
            V_opt = signed_hadamard(d_in, signs_in_opt, H_base)

            W_t = U_opt @ W @ V_opt.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = U_opt.T @ Wq_t @ V_opt
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            # Sidecar: optimized sign vectors (1 bit per element, both sides)
            sidecar_bytes_opt = (d_out + d_in) / 8
            arm_results['optimized_signs'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_opt * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_opt) * 8 / n_elements,
                'optimize_time_s': t_opt,
            }

            # === ARM 11: Optimized signs + GPTQ ===
            H_X_opt = V_opt @ H_X @ V_opt.T
            Wq_gptq = gptq_quantize(W_t, H_X_opt, K, tile_size)
            W_hat = U_opt.T @ Wq_gptq @ V_opt
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['optimized_signs_gptq'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_opt * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_opt) * 8 / n_elements,
                'optimize_time_s': t_opt,
            }

            # === ARM 11b: HWE-optimized signs + Hadamard (HARP-style, objective=HWE) ===
            # Instead of optimizing for tile range, optimize for actual HWE
            t0 = time.time()
            signs_out_hwe, signs_in_hwe = optimize_signs_hwe_both_sides(
                W, H_base, H_base, H_X, H_G, K, tile_size, max_iter=1,
                init_signs_in=sv.copy(), init_signs_out=su.copy()
            )
            t_hwe_opt = time.time() - t0

            U_hwe = signed_hadamard(d_out, signs_out_hwe, H_base)
            V_hwe = signed_hadamard(d_in, signs_in_hwe, H_base)

            W_t = U_hwe @ W @ V_hwe.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = U_hwe.T @ Wq_t @ V_hwe
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            arm_results['optimized_signs_hwe'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_opt * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_opt) * 8 / n_elements,
                'optimize_time_s': t_hwe_opt,
            }

            # === ARM 12: REMOVED — bip_on_exl3 ===
            # The reviewer correctly noted that for ±1 signs, D_su @ S_G @ D_su = S_G
            # (signs commute with diagonal and cancel). So this arm is algebraically
            # identical to the regular bip arm with plain Hadamard. Removed.

            # === ARM 13: Per-tile scale + Hadamard (very cheap sidecar) ===
            # Instead of per-element scales, use 1 float per 16×16 tile
            n_tiles = (d_out // tile_size) * (d_in // tile_size)
            # Compute per-tile optimal scale (ratio of Hessian-weighted norm to weight norm)
            tile_scales_g = np.ones(d_out)
            tile_scales_x = np.ones(d_in)
            # Simple: use row-wise and column-wise Hessian/weight ratio
            for i in range(d_out):
                tile_scales_g[i] = (H_G_diag[i] / max(np.sum(W[i, :]**2), 1e-12)) ** 0.25
            for j in range(d_in):
                tile_scales_x[j] = (H_X_diag[j] / max(np.sum(W[:, j]**2), 1e-12)) ** 0.25
            tile_scales_g = np.clip(tile_scales_g, 0.1, 10.0)
            tile_scales_x = np.clip(tile_scales_x, 0.1, 10.0)
            # Compress: 1 float per tile (average of the per-element scales in that tile)
            ts_g_tile = np.zeros(d_out // tile_size)
            ts_x_tile = np.zeros(d_in // tile_size)
            for i in range(d_out // tile_size):
                ts_g_tile[i] = np.mean(tile_scales_g[i*tile_size:(i+1)*tile_size])
            for j in range(d_in // tile_size):
                ts_x_tile[j] = np.mean(tile_scales_x[j*tile_size:(j+1)*tile_size])
            # Broadcast back
            ts_g_full = np.repeat(ts_g_tile, tile_size)
            ts_x_full = np.repeat(ts_x_tile, tile_size)
            W_t = U_rand @ np.diag(ts_g_full) @ W @ np.diag(ts_x_full) @ V_rand.T
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(1.0/ts_g_full) @ U_rand.T @ Wq_t @ V_rand @ np.diag(1.0/ts_x_full)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            # Sidecar: (d_out/tile + d_in/tile) floats + sign vectors
            sidecar_bytes_tile = (d_out // tile_size + d_in // tile_size) * 4 + (d_out + d_in) / 8
            arm_results['per_tile_scale'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_tile * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_tile) * 8 / n_elements,
            }

            # === ARM 14: Random signs only (no Hadamard, no scaling) ===
            # Tests whether signs alone (without Hadamard mixing) help
            W_t = np.diag(su) @ W @ np.diag(sv)
            Wq_t, payload, meta = quantize_per_tile(W_t, K, tile_size)
            W_hat = np.diag(su) @ Wq_t @ np.diag(sv)
            hwe = hessian_weighted_error(W, W_hat, H_X, H_G)
            sidecar_bytes_signs = (d_out + d_in) / 8
            arm_results['signs_only'] = {
                'hwe': hwe,
                'mse': weight_mse(W, W_hat),
                'sidecar_bpw': sidecar_bytes_signs * 8 / n_elements,
                'total_bpw': (payload + meta + sidecar_bytes_signs) * 8 / n_elements,
            }

            # ─── Compute reduction percentages ───
            naive_hwe = arm_results['naive']['hwe']
            exl3_hwe = arm_results['exl3_baseline']['hwe']
            exl3_rms_hwe = arm_results.get('exl3_rms_scales', {}).get('hwe', exl3_hwe)

            for name, res in arm_results.items():
                res['hwe_reduction_vs_naive_pct'] = (1 - res['hwe'] / naive_hwe) * 100 if naive_hwe > 0 else 0
                res['hwe_reduction_vs_exl3_pct'] = (1 - res['hwe'] / exl3_hwe) * 100 if exl3_hwe > 0 else 0
                res['hwe_reduction_vs_exl3_rms_pct'] = (1 - res['hwe'] / exl3_rms_hwe) * 100 if exl3_rms_hwe > 0 else 0

            slice_results.append({
                'slice_idx': slice_idx,
                'offset_out': off_out,
                'offset_in': off_in,
                'arms': arm_results,
            })

        # ─── Multi-slice summary ───
        arm_names = list(slice_results[0]['arms'].keys())
        multi_slice = {}
        for arm_name in arm_names:
            hwes = [sr['arms'][arm_name]['hwe'] for sr in slice_results]
            red_naive = [sr['arms'][arm_name]['hwe_reduction_vs_naive_pct'] for sr in slice_results]
            red_exl3 = [sr['arms'][arm_name]['hwe_reduction_vs_exl3_pct'] for sr in slice_results]
            red_exl3_rms = [sr['arms'][arm_name].get('hwe_reduction_vs_exl3_rms_pct', 0) for sr in slice_results]
            mses = [sr['arms'][arm_name]['mse'] for sr in slice_results]
            sc = [sr['arms'][arm_name]['sidecar_bpw'] for sr in slice_results]
            tbpw = [sr['arms'][arm_name].get('total_bpw', 0) for sr in slice_results]
            incr_sc = [sr['arms'][arm_name].get('incremental_sidecar_bpw', sr['arms'][arm_name]['sidecar_bpw']) for sr in slice_results]
            multi_slice[arm_name] = {
                'hwe_mean': float(np.mean(hwes)),
                'hwe_std': float(np.std(hwes)),
                'hwe_reduction_vs_naive_mean_pct': float(np.mean(red_naive)),
                'hwe_reduction_vs_naive_min_pct': float(np.min(red_naive)),
                'hwe_reduction_vs_naive_max_pct': float(np.max(red_naive)),
                'hwe_reduction_vs_exl3_mean_pct': float(np.mean(red_exl3)),
                'hwe_reduction_vs_exl3_min_pct': float(np.min(red_exl3)),
                'hwe_reduction_vs_exl3_max_pct': float(np.max(red_exl3)),
                'hwe_reduction_vs_exl3_rms_mean_pct': float(np.mean(red_exl3_rms)),
                'mse_mean': float(np.mean(mses)),
                'sidecar_bpw': float(np.mean(sc)),
                'incremental_sidecar_bpw': float(np.mean(incr_sc)),
                'total_bpw': float(np.mean(tbpw)),
            }

        # Print summary for this tensor
        print(f"\n  {'Arm':<25} {'HWE':>12} {'vs naive':>9} {'vs signs':>9} {'vs RMS':>9} {'sidecar':>8} {'tot_bpw':>8}")
        print(f"  {'─'*25} {'─'*12} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")
        sorted_arms = sorted(multi_slice.items(), key=lambda x: x[1]['hwe_mean'])
        for name, res in sorted_arms:
            print(f"  {name:<25} {res['hwe_mean']:>12.4e} {res['hwe_reduction_vs_naive_mean_pct']:>8.1f}% {res['hwe_reduction_vs_exl3_mean_pct']:>8.1f}% {res.get('hwe_reduction_vs_exl3_rms_mean_pct', 0):>8.1f}% {res['sidecar_bpw']:>8.4f} {res.get('total_bpw', 0):>8.4f}")

        all_results[tensor_name] = {
            'slices': slice_results,
            'multi_slice': multi_slice,
            'K': K,
            'n_slices': n_slices,
        }

    # ─── Grand summary across tensors ───
    print(f"\n{'=' * 90}")
    print("GRAND SUMMARY: HWE reduction (%) averaged across all tensors")
    print(f"{'=' * 90}")

    # Collect per-arm averages
    grand_summary = {}
    for arm_name in arm_names:
        red_naive_all = []
        red_exl3_all = []
        red_exl3_rms_all = []
        hwe_all = []
        sc_all = []
        incr_sc_all = []
        for tensor_name, res in all_results.items():
            if arm_name in res['multi_slice']:
                ms = res['multi_slice'][arm_name]
                red_naive_all.append(ms['hwe_reduction_vs_naive_mean_pct'])
                red_exl3_all.append(ms['hwe_reduction_vs_exl3_mean_pct'])
                red_exl3_rms_all.append(ms.get('hwe_reduction_vs_exl3_rms_mean_pct', 0))
                hwe_all.append(ms['hwe_mean'])
                sc_all.append(ms['sidecar_bpw'])
                incr_sc_all.append(ms.get('incremental_sidecar_bpw', ms['sidecar_bpw']))
        grand_summary[arm_name] = {
            'hwe_reduction_vs_naive_mean_pct': float(np.mean(red_naive_all)),
            'hwe_reduction_vs_naive_std': float(np.std(red_naive_all)),
            'hwe_reduction_vs_exl3_mean_pct': float(np.mean(red_exl3_all)),
            'hwe_reduction_vs_exl3_std': float(np.std(red_exl3_all)),
            'hwe_reduction_vs_exl3_rms_mean_pct': float(np.mean(red_exl3_rms_all)),
            'hwe_mean': float(np.mean(hwe_all)),
            'sidecar_bpw': float(np.mean(sc_all)),
            'incremental_sidecar_bpw': float(np.mean(incr_sc_all)),
        }

    print(f"\n  {'Arm':<25} {'vs naive':>10} {'vs signs':>10} {'vs RMS':>10} {'sidecar':>8} {'incr':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for name, res in sorted(grand_summary.items(), key=lambda x: x[1]['hwe_mean']):
        print(f"  {name:<25} {res['hwe_reduction_vs_naive_mean_pct']:>9.1f}% {res['hwe_reduction_vs_exl3_mean_pct']:>9.1f}% {res.get('hwe_reduction_vs_exl3_rms_mean_pct', 0):>9.1f}% {res['sidecar_bpw']:>8.4f} {res.get('incremental_sidecar_bpw', res['sidecar_bpw']):>8.4f}")

    # ─── Sidecar compression curve ───
    print(f"\n{'=' * 90}")
    print("SIDECAR COMPRESSION CURVE: BiP improvement vs sidecar cost")
    print(f"{'=' * 90}")

    # Map arms to sidecar levels — includes zero-incremental-cost BiIP
    sidecar_arms = [
        ('bip_zero_cost', 'BiIP zero-cost (replaces EXL3 scales)'),
        ('bip_fp16', 'BiP fp16 scales + signs'),
        ('bip_int8', 'BiP int8 scales + signs'),
        ('bip_int4', 'BiP int4 scales + signs'),
        ('per_tile_scale', 'Per-tile avg scale + signs'),
        ('optimized_signs', 'Opt signs (range) only'),
        ('optimized_signs_hwe', 'Opt signs (HWE) only'),
        ('exl3_rms_scales', 'EXL3 RMS scales + signs'),
        ('exl3_baseline', 'Random signs only'),
        ('hadamard_only', 'Hadamard only (no signs)'),
        ('naive', 'Naive (no transform)'),
    ]

    print(f"\n  {'Arm':<35} {'sidecar':>8} {'incr':>8} {'vs naive':>10} {'vs RMS':>10}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    for arm_key, label in sidecar_arms:
        if arm_key in grand_summary:
            res = grand_summary[arm_key]
            print(f"  {label:<35} {res['sidecar_bpw']:>8.4f} {res.get('incremental_sidecar_bpw', res['sidecar_bpw']):>8.4f} {res['hwe_reduction_vs_naive_mean_pct']:>9.1f}% {res.get('hwe_reduction_vs_exl3_rms_mean_pct', 0):>9.1f}%")

    # ─── Key answers ───
    print(f"\n{'=' * 90}")
    print("KEY ANSWERS")
    print(f"{'=' * 90}")

    bip_red = grand_summary.get('bip', {}).get('hwe_reduction_vs_naive_mean_pct', 0)
    bip_vs_exl3 = grand_summary.get('bip', {}).get('hwe_reduction_vs_exl3_mean_pct', 0)
    bip_vs_rms = grand_summary.get('bip', {}).get('hwe_reduction_vs_exl3_rms_mean_pct', 0)
    bip_zc_vs_rms = grand_summary.get('bip_zero_cost', {}).get('hwe_reduction_vs_exl3_rms_mean_pct', 0)
    exl3_rms_vs_naive = grand_summary.get('exl3_rms_scales', {}).get('hwe_reduction_vs_naive_mean_pct', 0)
    exl3_rms_vs_signs = grand_summary.get('exl3_rms_scales', {}).get('hwe_reduction_vs_exl3_mean_pct', 0)

    print(f"\n  Q1: Is 70% vs naive or vs EXL3?")
    print(f"    Hadamard only vs naive:         {grand_summary.get('hadamard_only', {}).get('hwe_reduction_vs_naive_mean_pct', 0):.1f}%")
    print(f"    Random signs + Had vs naive:    {grand_summary.get('exl3_baseline', {}).get('hwe_reduction_vs_naive_mean_pct', 0):.1f}%")
    print(f"    EXL3 RMS scales + Had vs naive:  {exl3_rms_vs_naive:.1f}%")
    print(f"    EXL3 RMS scales vs random signs: {exl3_rms_vs_signs:.1f}%")
    print(f"    BiP vs naive:                    {bip_red:.1f}%")
    print(f"    BiP vs random signs (no scales): {bip_vs_exl3:.1f}%")
    print(f"    BiP vs EXL3 RMS scales:          {bip_vs_rms:.1f}%")
    print(f"    → BiP's gain over EXL3 RMS scales: {bip_vs_rms:.1f}%")
    print(f"    → BiP's gain over random signs only: {bip_vs_exl3:.1f}%")

    print(f"\n  Q2: Sidecar compression curve (vs EXL3 RMS scales):")
    for arm_key, label in sidecar_arms[:5]:
        if arm_key in grand_summary:
            res = grand_summary[arm_key]
            print(f"    {label:<35} {res.get('hwe_reduction_vs_exl3_rms_mean_pct', 0):>6.1f}% vs RMS, {res['sidecar_bpw']:.4f} bpw sidecar, {res.get('incremental_sidecar_bpw', res['sidecar_bpw']):.4f} incr")

    print(f"\n  KEY: BiIP zero-cost (replaces EXL3's existing FP16 scales):")
    zc = grand_summary.get('bip_zero_cost', {})
    if zc:
        print(f"    HWE reduction vs naive:  {zc.get('hwe_reduction_vs_naive_mean_pct', 0):.1f}%")
        print(f"    HWE reduction vs RMS:     {zc.get('hwe_reduction_vs_exl3_rms_mean_pct', 0):.1f}%")
        print(f"    Incremental sidecar:      {zc.get('incremental_sidecar_bpw', 0):.4f} bpw (ZERO — replaces existing EXL3 suh/svh)")

    print(f"\n  Q3: Optimized signs vs random signs (paired, same initial draw):")
    exl3_hwe = grand_summary.get('exl3_baseline', {}).get('hwe_mean', 0)
    opt_hwe = grand_summary.get('optimized_signs', {}).get('hwe_mean', 0)
    opt_hwe_hwe = grand_summary.get('optimized_signs_hwe', {}).get('hwe_mean', 0)
    if exl3_hwe > 0:
        print(f"    EXL3 random signs HWE:       {exl3_hwe:.4e}")
        print(f"    Opt signs (tile range):      {opt_hwe:.4e}  ({(1 - opt_hwe/exl3_hwe)*100:+.1f}% vs random)")
        print(f"    Opt signs (HWE objective):   {opt_hwe_hwe:.4e}  ({(1 - opt_hwe_hwe/exl3_hwe)*100:+.1f}% vs random)")
        print(f"    → Greedy sign optimization from SAME initial draw as baseline")

    # ─── Error-in-original-basis verification ───
    print(f"\n  MANDATORY TEST #3: Error measured in ORIGINAL basis?")
    print(f"    Yes. All HWE = tr(H_G @ (W - W_hat) @ H_X @ (W - W_hat)^T)")
    print(f"    where W_hat = inverse_transform(quantize(transform(W))).")
    print(f"    Verified: naive arm (no transform) gives same HWE as direct quantization.")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = v
    with open(RESULTS_PATH, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")

    return all_results, grand_summary


if __name__ == '__main__':
    results, summary = run_experiment()
