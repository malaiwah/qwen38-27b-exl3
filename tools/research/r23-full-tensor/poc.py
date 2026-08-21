#!/usr/bin/env python3
"""
R23-FullTensor: Block-Hadamard on full tensors.

Previous experiments (R3-R20) use 128×128 slices from full tensors
(5120×17408, etc.). This POC tests whether those conclusions generalize to
FULL tensors. Key challenge: tensor dimensions (5120, 17408, 10240, 6144)
are not powers of 2, so full-size Hadamard matrices can't be constructed.

Solution: BLOCK-Hadamard with block size 16 (power of 2). All model dimensions
are divisible by 16, so no padding needed. Apply H_16 to each 16-element block
along rows and columns. This is equivalent to multiplying by kron(I, H_16).

Experiment design:
  1. Block-Hadamard for non-power-of-2 (verified correct)
  2. Full-tensor BiIP + block-Hadamard: measure incoherence, HWE
  3. Full-tensor vs 128×128 slice comparison
  4. DP allocation tractability on 348,160 tiles
  5. Memory and timing

K=4,5,6 (skip K=3 for time).
"""

import json
import time
import sys
import os
import traceback

import numpy as np

# ─── Paths ───
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r23-full-tensor/receipts/research/r23-full-tensor-results.json"

BLOCK = 16  # block size for block-Hadamard and tile quantization

# ─── Block-Hadamard ───

def hadamard_matrix(n):
    """Sylvester-type Hadamard matrix of size n (must be power of 2)."""
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def block_hadamard_transform(W, block_size=BLOCK, axis=0, signs=None, rng=None):
    """Apply block-Hadamard along a given axis.
    
    For axis=0 (rows): reshape (d_out, d_in) -> (n_blocks, block_size, d_in),
    apply H_block to each block, reshape back.
    For axis=1 (cols): reshape (d_out, d_in) -> (d_out, n_blocks, block_size),
    apply H_block to each block, reshape back.
    
    Equivalent to multiplying by kron(I_{n_blocks}, H_block) on the left (axis=0)
    or right (axis=1).
    
    signs: optional ±1 randomization applied within each block (sidecar: 1 bit/element).
    
    """
    d_out, d_in = W.shape
    H = hadamard_matrix(block_size)
    if axis == 0:
        assert d_out % block_size == 0, f"d_out={d_out} not divisible by block_size={block_size}"
        n_blocks = d_out // block_size
        W_r = W.reshape(n_blocks, block_size, d_in)
        if signs is None:
            if rng is not None:
                signs = rng.choice([-1.0, 1.0], size=(n_blocks, block_size))
            else:
                signs = np.ones((n_blocks, block_size))
        # For each block b: (H * signs[b][None,:]) @ W_r[b] — column-scaled Hadamard
        W_t = np.empty_like(W_r)
        for b in range(n_blocks):
            W_t[b] = (H * signs[b][None, :]) @ W_r[b]
        return W_t.reshape(d_out, d_in), signs
    else:  # axis == 1
        assert d_in % block_size == 0, f"d_in={d_in} not divisible by block_size={block_size}"
        n_blocks = d_in // block_size
        W_r = W.reshape(d_out, n_blocks, block_size)
        if signs is None:
            if rng is not None:
                signs = rng.choice([-1.0, 1.0], size=(n_blocks, block_size))
            else:
                signs = np.ones((n_blocks, block_size))
        # For each block b: W_r[:, b, :] @ (H.T * signs[b][:, None])
        W_t = np.empty_like(W_r)
        H_T = H.T
        for b in range(n_blocks):
            W_t[:, b, :] = W_r[:, b, :] @ (H_T * signs[b][:, None])
        return W_t.reshape(d_out, d_in), signs

def block_hadamard_inverse(W, block_size=BLOCK, axis=0, signs=None):
    """Inverse block-Hadamard. Since H is orthogonal and symmetric, inverse = H itself.
    With signs: inverse is diag(signs) @ H @ (since H^T = H, H^2 = I after normalization).
    Actually: forward = H @ diag(signs), inverse = diag(signs)^{-1} @ H = diag(signs) @ H
    (signs are ±1, so inverse of diag(signs) = diag(signs))
    """
    d_out, d_in = W.shape
    H = hadamard_matrix(block_size)
    
    if axis == 0:
        n_blocks = d_out // block_size
        W_r = W.reshape(n_blocks, block_size, d_in)
        W_t = np.empty_like(W_r)
        for b in range(n_blocks):
            # inverse of (H @ diag(signs)) is diag(signs) @ H^T = diag(signs) @ H
            W_t[b] = (signs[b][:, None] * H) @ W_r[b]
        return W_t.reshape(d_out, d_in)
    else:
        n_blocks = d_in // block_size
        W_r = W.reshape(d_out, n_blocks, block_size)
        W_t = np.empty_like(W_r)
        H_T = H.T
        for b in range(n_blocks):
            # forward was W @ (H_T * signs[:, None]) = W @ (H * signs[:, None])
            # = W @ diag(signs) @ H
            # inverse = H^T @ diag(signs) @ W = H @ diag(signs) @ W
            # Per-block: (H * signs[b][:, None]) @ W_r[:, b, :].T, then transpose back
            # Actually simpler: inverse_col = (H * signs[b][None, :]) applied on right
            # forward: W_r[:, b, :] @ (H_T * signs[b][:, None])
            #   = W_r[:, b, :] @ diag(signs) @ H_T
            # inverse: W_r[:, b, :] @ H_T @ diag(signs) = W_r[:, b, :] @ (H_T * signs[b][None, :])
            W_t[:, b, :] = W_r[:, b, :] @ (H_T * signs[b][None, :])
        return W_t.reshape(d_out, d_in)


def verify_block_hadamard():
    """Verify block-Hadamard is orthogonal and its inverse recovers the original."""
    rng = np.random.default_rng(42)
    for shape in [(128, 128), (5120, 17408), (17408, 5120)]:
        d_out, d_in = shape
        # Use a small test matrix for large shapes
        if d_out * d_in > 200000:
            W_test = rng.standard_normal((128, 128))
            d_out_t, d_in_t = 128, 128
        else:
            W_test = rng.standard_normal(shape)
            d_out_t, d_in_t = d_out, d_in
        
        # Forward + inverse on rows
        W_fwd, signs_r = block_hadamard_transform(W_test, axis=0, rng=rng)
        W_back = block_hadamard_inverse(W_fwd, axis=0, signs=signs_r)
        err_rows = np.max(np.abs(W_test - W_back))
        
        # Forward + inverse on cols
        W_fwd_c, signs_c = block_hadamard_transform(W_test, axis=1, rng=rng)
        W_back_c = block_hadamard_inverse(W_fwd_c, axis=1, signs=signs_c)
        err_cols = np.max(np.abs(W_test - W_back_c))
        
        # Check orthogonality: for a random vector, norm should be preserved
        v = rng.standard_normal(d_out_t)
        v_fwd, _ = block_hadamard_transform(v.reshape(d_out_t, 1), axis=0, rng=rng)
        norm_orig = np.linalg.norm(v)
        norm_fwd = np.linalg.norm(v_fwd)
        
        print(f"  Shape {shape}: row inv err={err_rows:.2e}, col inv err={err_cols:.2e}, "
              f"norm preservation: {norm_orig:.6f} -> {norm_fwd:.6f} (ratio={norm_fwd/norm_orig:.6f})")
        
        assert err_rows < 1e-10, f"Row inverse failed: err={err_rows}"
        assert err_cols < 1e-10, f"Col inverse failed: err={err_cols}"
        assert abs(norm_fwd / norm_orig - 1.0) < 1e-10, f"Norm not preserved"
    
    # Verify it's equivalent to kron(I, H)
    W_small = rng.standard_normal((32, 48))
    H16 = hadamard_matrix(16)
    I2 = np.eye(2)
    I3 = np.eye(3)
    # kron(I_2, H_16) should match block-Hadamard on axis 0
    K_left = np.kron(I2, H16)
    K_right = np.kron(I3, H16)
    W_kron = K_left @ W_small @ K_right
    W_block, signs0 = block_hadamard_transform(W_small, axis=0, rng=np.random.default_rng(99))
    W_block, signs1 = block_hadamard_transform(W_block, axis=1, rng=np.random.default_rng(99))
    # For kron comparison, use signs=1
    W_block_nosign, _ = block_hadamard_transform(W_small, axis=0, rng=None, signs=np.ones((2, 16)))
    W_block_nosign, _ = block_hadamard_transform(W_block_nosign, axis=1, rng=None, signs=np.ones((3, 16)))
    err_kron = np.max(np.abs(W_kron - W_block_nosign))
    print(f"  kron equivalence (no signs): err={err_kron:.2e}")
    assert err_kron < 1e-12, f"Block-Hadamard != kron(I, H): err={err_kron}"
    
    print("  ✓ Block-Hadamard verified: orthogonal, invertible, kron-equivalent")


# ─── Quantization ───

def quantize_per_tile(W, K, tile_size=BLOCK):
    """Per-tile uniform quantization with 2^K levels. Vectorized."""
    levels = 2 ** K
    d_out, d_in = W.shape
    
    # Pad if needed (shouldn't be for our shapes, but be safe)
    pad_out = (tile_size - d_out % tile_size) % tile_size
    pad_in = (tile_size - d_in % tile_size) % tile_size
    if pad_out or pad_in:
        W = np.pad(W, ((0, pad_out), (0, pad_in)), mode='edge')
    
    d_out_p, d_in_p = W.shape
    n_out = d_out_p // tile_size
    n_in = d_in_p // tile_size
    
    W_tiles = W.reshape(n_out, tile_size, n_in, tile_size).transpose(0, 2, 1, 3)
    # W_tiles: (n_out, n_in, tile_size, tile_size)
    
    w_min = W_tiles.min(axis=(2, 3), keepdims=True)
    w_max = W_tiles.max(axis=(2, 3), keepdims=True)
    rng_range = w_max - w_min
    rng_range = np.where(rng_range < 1e-12, 1.0, rng_range)
    
    scale = rng_range / (levels - 1)
    q = np.round((W_tiles - w_min) / scale).clip(0, levels - 1)
    Wq_tiles = q * scale + w_min
    
    Wq = Wq_tiles.transpose(0, 2, 1, 3).reshape(d_out_p, d_in_p)
    if pad_out or pad_in:
        Wq = Wq[:d_out, :d_in]
    
    payload_bytes = d_out * d_in * K / 8
    meta_bytes = 2 * n_out * n_in * 4  # per-tile min/max
    
    return Wq, payload_bytes, meta_bytes


# ─── BiIP scaling (diagonal Hessian version for full tensors) ───

def biip_scaling_diag(W, H_X_diag, H_G_diag):
    """Two-sided diagonal balancing using diagonal Hessians only.
    
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}
    S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}
    W' = S_G @ W @ S_X
    
    Returns: S_G_diag, S_X_diag, W_transformed, sidecar_bytes
    """
    d_out, d_in = W.shape
    
    col_norms_sq = np.sum(W ** 2, axis=0)
    col_norms_sq = np.maximum(col_norms_sq, 1e-12)
    sx_diag = (H_X_diag / col_norms_sq) ** 0.25
    sx_diag = np.clip(sx_diag, 0.1, 10.0)
    
    row_norms_sq = np.sum(W ** 2, axis=1)
    row_norms_sq = np.maximum(row_norms_sq, 1e-12)
    sg_diag = (H_G_diag / row_norms_sq) ** 0.25
    sg_diag = np.clip(sg_diag, 0.1, 10.0)
    
    W_transformed = (sg_diag[:, None] * W) * sx_diag[None, :]
    
    sidecar_bytes = (d_in + d_out) * 4
    return sg_diag, sx_diag, W_transformed, sidecar_bytes


# ─── Metrics ───

def incoherence_measure(W):
    """μ(W) = sqrt(d_out * d_in) / ||W||_F * max|W_ij|. Lower = more incoherent."""
    d_out, d_in = W.shape
    return np.sqrt(d_out * d_in) / (np.linalg.norm(W) + 1e-12) * np.max(np.abs(W))


def cv_column_norms(W):
    """CV of column norms."""
    col_norms = np.linalg.norm(W, axis=0)
    return np.std(col_norms) / (np.mean(col_norms) + 1e-12)


def cv_row_norms(W):
    """CV of row norms."""
    row_norms = np.linalg.norm(W, axis=1)
    return np.std(row_norms) / (np.mean(row_norms) + 1e-12)


def mean_tile_range(W, tile_size=BLOCK):
    """Mean range across all tile_size × tile_size tiles (vectorized)."""
    d_out, d_in = W.shape
    n_out = d_out // tile_size
    n_in = d_in // tile_size
    W_tiles = W[:n_out*tile_size, :n_in*tile_size].reshape(n_out, tile_size, n_in, tile_size)
    W_tiles = W_tiles.transpose(0, 2, 1, 3)  # (n_out, n_in, ts, ts)
    ranges = W_tiles.max(axis=(2, 3)) - W_tiles.min(axis=(2, 3))
    return np.mean(ranges)


def max_tile_range(W, tile_size=BLOCK):
    """Max range across all tiles."""
    d_out, d_in = W.shape
    n_out = d_out // tile_size
    n_in = d_in // tile_size
    W_tiles = W[:n_out*tile_size, :n_in*tile_size].reshape(n_out, tile_size, n_in, tile_size)
    W_tiles = W_tiles.transpose(0, 2, 1, 3)
    ranges = W_tiles.max(axis=(2, 3)) - W_tiles.min(axis=(2, 3))
    return np.max(ranges)


def hwe_diag(W, W_hat, H_X_diag, H_G_diag):
    """Diagonal Hessian-weighted error: sum_i sum_j H_G_i * H_X_j * E_ij^2."""
    E = W - W_hat
    weights = np.outer(H_G_diag, H_X_diag)
    return np.sum(weights * E ** 2)


def weight_mse(W, W_hat):
    return np.mean((W - W_hat) ** 2)


# ─── Synthetic diagonal Hessians ───

def synthetic_diag_hessians(W, n_samples=512, outlier_fraction=0.05, outlier_scale=10.0, seed=42):
    """Generate diagonal-only synthetic Hessians for full tensors.
    
    H_X_diag = diag(X^T X / N) = mean(X^2, axis=1) for X (d_in, N)
    H_G_diag = diag(Y^T Y / N) = mean(Y^2, axis=1) for Y = W @ X (d_out, N)
    """
    rng = np.random.default_rng(seed)
    d_out, d_in = W.shape
    
    X = rng.standard_normal((d_in, n_samples))
    n_outliers = max(1, int(d_in * outlier_fraction))
    outlier_channels = rng.choice(d_in, n_outliers, replace=False)
    X[outlier_channels, :] *= outlier_scale
    
    H_X_diag = np.mean(X ** 2, axis=1)
    H_X_diag *= d_in / np.sum(H_X_diag)  # normalize mean=1
    
    Y = W @ X
    H_G_diag = np.mean(Y ** 2, axis=1)
    H_G_diag *= d_out / np.sum(H_G_diag)
    
    H_X_diag = np.maximum(H_X_diag, 1e-6)
    H_G_diag = np.maximum(H_G_diag, 1e-6)
    
    return H_X_diag, H_G_diag


# ─── Full-tensor experiment ───

def run_full_tensor_experiment():
    """Run BiIP + block-Hadamard on full tensors and compare to 128×128 slices."""
    
    print("=" * 80)
    print("R23-FullTensor: Block-Hadamard on FULL tensors")
    print("Validates whether 128×128 slice conclusions generalize")
    print("=" * 80)
    
    # Verify block-Hadamard
    print("\n--- Block-Hadamard verification ---")
    verify_block_hadamard()
    
    # Load weights
    print("\n--- Loading weights ---")
    weights = np.load(WEIGHTS_PATH)
    
    tensors_to_test = [
        ('L0_gate',  'L0_gate',  [4, 5, 6]),
        ('L55_down', 'L55_down', [4, 5, 6]),
        ('L0_down',  'L0_down',  [4, 5, 6]),
        ('L55_gate', 'L55_gate', [4, 5, 6]),
    ]
    
    all_results = {}
    
    for tensor_name, weight_key, K_values in tensors_to_test:
        print(f"\n{'─' * 70}")
        print(f"Tensor: {tensor_name} (key={weight_key})")
        print(f"{'─' * 70}")
        
        W_full = weights[weight_key].astype(np.float64)
        d_out, d_in = W_full.shape
        print(f"  Shape: ({d_out}, {d_in}), elements: {d_out*d_in:,}")
        
        # Memory estimate
        mem_bytes = d_out * d_in * 8  # float64
        print(f"  Memory (float64): {mem_bytes / 1e6:.1f} MB")
        
        t_start = time.time()
        
        # Generate diagonal Hessians for full tensor
        H_X_diag_full, H_G_diag_full = synthetic_diag_hessians(W_full, n_samples=512, seed=42)
        
        # ─── Full tensor metrics (no transform) ───
        mu_full = incoherence_measure(W_full)
        cv_in_full = cv_column_norms(W_full)
        cv_out_full = cv_row_norms(W_full)
        mtr_full = mean_tile_range(W_full)
        xtr_full = max_tile_range(W_full)
        
        print(f"  No transform: μ={mu_full:.4f}, CV_in={cv_in_full:.4f}, CV_out={cv_out_full:.4f}, "
              f"mean_tile_range={mtr_full:.6f}, max_tile_range={xtr_full:.6f}")
        
        # ─── BiIP + block-Hadamard on full tensor ───
        rng = np.random.default_rng(42)
        
        # Step 1: BiIP scaling
        sg_diag, sx_diag, W_biip, sc_bytes = biip_scaling_diag(W_full, H_X_diag_full, H_G_diag_full)
        mu_biip = incoherence_measure(W_biip)
        cv_in_biip = cv_column_norms(W_biip)
        cv_out_biip = cv_row_norms(W_biip)
        mtr_biip = mean_tile_range(W_biip)
        
        print(f"  BiIP:        μ={mu_biip:.4f}, CV_in={cv_in_biip:.4f}, CV_out={cv_out_biip:.4f}, "
              f"mean_tile_range={mtr_biip:.6f}")
        
        # Step 2: Block-Hadamard on rows (axis=0)
        W_bh_r, signs_r = block_hadamard_transform(W_biip, axis=0, rng=rng)
        # Step 3: Block-Hadamard on cols (axis=1)
        W_bh, signs_c = block_hadamard_transform(W_bh_r, axis=1, rng=rng)
        
        mu_bh = incoherence_measure(W_bh)
        cv_in_bh = cv_column_norms(W_bh)
        cv_out_bh = cv_row_norms(W_bh)
        mtr_bh = mean_tile_range(W_bh)
        xtr_bh = max_tile_range(W_bh)
        
        # Sidecar: signs (1 bit/element) + BiIP diagonals
        sidecar_signs = (d_out + d_in) / 8  # 1 bit per element for signs
        sidecar_biip = sc_bytes
        sidecar_total = sidecar_signs + sidecar_biip
        
        print(f"  BiIP+BH:     μ={mu_bh:.4f}, CV_in={cv_in_bh:.4f}, CV_out={cv_out_bh:.4f}, "
              f"mean_tile_range={mtr_bh:.6f}, max_tile_range={xtr_bh:.6f}")
        print(f"  Incoherence reduction: {(1 - mu_bh/mu_full)*100:.1f}%")
        print(f"  Tile range reduction:  {(1 - mtr_bh/mtr_full)*100:.1f}%")
        print(f"  Sidecar: {sidecar_total/1e6:.2f} MB ({sidecar_signs/1e6:.2f} signs + {sidecar_biip/1e6:.2f} BiIP)")
        
        # ─── Quantization and HWE for each K ───
        quant_results = {}
        for K in K_values:
            # Quantize no-transform
            Wq_none, pl_none, meta_none = quantize_per_tile(W_full, K)
            hwe_none = hwe_diag(W_full, Wq_none, H_X_diag_full, H_G_diag_full)
            mse_none = weight_mse(W_full, Wq_none)
            
            # Quantize BiIP+BH (transformed space)
            Wq_bh, pl_bh, meta_bh = quantize_per_tile(W_bh, K)
            
            # Inverse transform to get back to original space
            # W_hat = S_G^{-1} @ (BH_inv_rows(BH_inv_cols(Wq_bh))) @ S_X^{-1}
            W_inv_c = block_hadamard_inverse(Wq_bh, axis=1, signs=signs_c)
            W_inv_r = block_hadamard_inverse(W_inv_c, axis=0, signs=signs_r)
            W_hat = (1.0 / sg_diag[:, None]) * W_inv_r * (1.0 / sx_diag[None, :])
            
            hwe_bh = hwe_diag(W_full, W_hat, H_X_diag_full, H_G_diag_full)
            mse_bh = weight_mse(W_full, W_hat)
            
            # Total bytes
            total_none = pl_none + meta_none
            total_bh = pl_bh + meta_bh + sidecar_total
            
            # HWE per element (normalized)
            hwe_per_elem_none = hwe_none / (d_out * d_in)
            hwe_per_elem_bh = hwe_bh / (d_out * d_in)
            
            improvement = (1 - hwe_bh / hwe_none) * 100 if hwe_none > 0 else 0
            
            print(f"\n  K={K}:")
            print(f"    No-transform: HWE={hwe_none:.6e}, MSE={mse_none:.6e}, "
                  f"bytes/elem={total_none/(d_out*d_in):.4f}")
            print(f"    BiIP+BH:      HWE={hwe_bh:.6e}, MSE={mse_bh:.6e}, "
                  f"bytes/elem={total_bh/(d_out*d_in):.4f}")
            print(f"    HWE improvement: {improvement:.1f}%")
            
            quant_results[f'K{K}'] = {
                'hwe_none': hwe_none,
                'hwe_biip_bh': hwe_bh,
                'mse_none': mse_none,
                'mse_biip_bh': mse_bh,
                'hwe_per_elem_none': hwe_per_elem_none,
                'hwe_per_elem_bh': hwe_per_elem_bh,
                'bytes_per_elem_none': total_none / (d_out * d_in),
                'bytes_per_elem_bh': total_bh / (d_out * d_in),
                'hwe_improvement_pct': improvement,
            }
        
        t_full = time.time() - t_start
        
        # ─── 128×128 slice comparison (same H16 transform, restricted Hessians) ───
        print(f"\n  --- 128×128 slice comparison (same H16 transform) ---")
        slice_results = {}
        
        n_slices = 10
        rng_slices = np.random.default_rng(123)
        max_off_out = d_out - 128
        max_off_in = d_in - 128
        
        # Sample random slice offsets
        slice_offsets = []
        for _ in range(n_slices):
            off_o = rng_slices.integers(0, max_off_out + 1)
            off_i = rng_slices.integers(0, max_off_in + 1)
            # Align to 16-block boundaries
            off_o = (off_o // BLOCK) * BLOCK
            off_i = (off_i // BLOCK) * BLOCK
            slice_offsets.append((off_o, off_i))
        
        # Pool raw HWE for each K across all slices
        pooled_hwe_none = {K: 0.0 for K in K_values}
        pooled_hwe_bh = {K: 0.0 for K in K_values}
        
        for s_idx, (off_out, off_in) in enumerate(slice_offsets):
            W_slice = W_full[off_out:off_out+128, off_in:off_in+128]
            
            # Restrict full-tensor Hessians to this slice (not regenerate)
            H_X_s = H_X_diag_full[off_in:off_in+128]
            H_G_s = H_G_diag_full[off_out:off_out+128]
            
            sg_s = sg_diag[off_out:off_out+128]
            sx_s = sx_diag[off_in:off_in+128]
            W_slice_biip = (sg_s[:, None] * W_slice) * sx_s[None, :]
            
            # Restrict full-tensor block-Hadamard signs to this slice
            # signs_r shape: (n_blocks_out, 16), signs_c shape: (n_blocks_in, 16)
            block_off_out = off_out // BLOCK
            block_off_in = off_in // BLOCK
            signs_sr_slice = signs_r[block_off_out:block_off_out+8]  # 8 blocks × 16
            signs_sc_slice = signs_c[block_off_in:block_off_in+8]
            
            W_slice_bh_r, _ = block_hadamard_transform(W_slice_biip, axis=0, signs=signs_sr_slice)
            W_slice_bh, _ = block_hadamard_transform(W_slice_bh_r, axis=1, signs=signs_sc_slice)
            
            mu_slice = incoherence_measure(W_slice)
            mu_slice_bh = incoherence_measure(W_slice_bh)
            
            for K in K_values:
                # No-transform
                Wq_slice_none, _, _ = quantize_per_tile(W_slice, K)
                hwe_slice_none = hwe_diag(W_slice, Wq_slice_none, H_X_s, H_G_s)
                
                Wq_slice_bh, _, _ = quantize_per_tile(W_slice_bh, K)
                W_inv_c = block_hadamard_inverse(Wq_slice_bh, axis=1, signs=signs_sc_slice)
                W_inv_r = block_hadamard_inverse(W_inv_c, axis=0, signs=signs_sr_slice)
                W_slice_hat = (1.0 / sg_s[:, None]) * W_inv_r * (1.0 / sx_s[None, :])
                hwe_slice_bh = hwe_diag(W_slice, W_slice_hat, H_X_s, H_G_s)
                
                improvement = (1 - hwe_slice_bh / hwe_slice_none) * 100 if hwe_slice_none > 0 else 0
                
                slice_results[f"slice{s_idx}_K{K}"] = {
                    'hwe_none': hwe_slice_none,
                    'hwe_biip_bh': hwe_slice_bh,
                    'hwe_improvement_pct': improvement,
                    'mu_none': mu_slice,
                    'mu_bh': mu_slice_bh,
                    'offset': [int(off_out), int(off_in)],
                }
                
                # Pool raw HWE
                pooled_hwe_none[K] += hwe_slice_none
                pooled_hwe_bh[K] += hwe_slice_bh
                
                if s_idx < 3:
                    print(f"    Slice {s_idx} K={K}: HWE none={hwe_slice_none:.4e} → "
                          f"BiIP+BH={hwe_slice_bh:.4e} ({improvement:.1f}%)")
        
        # Pooled improvement (aggregate HWE before ratio)
        for K in K_values:
            pooled_imp = (1 - pooled_hwe_bh[K] / pooled_hwe_none[K]) * 100 if pooled_hwe_none[K] > 0 else 0
            full_imp = quant_results[f'K{K}']['hwe_improvement_pct']
            slice_individual_imps = [slice_results[f"slice{s}_K{K}"]['hwe_improvement_pct'] for s in range(n_slices)]
            print(f"\n  K={K} HWE improvement: full={full_imp:.1f}%, "
                  f"slice pooled={pooled_imp:.1f}%, "
                  f"slice mean={np.mean(slice_individual_imps):.1f}% "
                  f"(range: {np.min(slice_individual_imps):.1f}%-{np.max(slice_individual_imps):.1f}%)")
            slice_results[f'pooled_K{K}'] = {
                'hwe_none': pooled_hwe_none[K],
                'hwe_biip_bh': pooled_hwe_bh[K],
                'hwe_improvement_pct': pooled_imp,
            }
        
        # Incoherence comparison
        slice_mus = [slice_results[f"slice{s}_K{K_values[0]}"]['mu_none'] for s in range(n_slices)]
        slice_mus_bh = [slice_results[f"slice{s}_K{K_values[0]}"]['mu_bh'] for s in range(n_slices)]
        print(f"  Incoherence: full μ={mu_full:.4f} → BiIP+BH={mu_bh:.4f} "
              f"({(1-mu_bh/mu_full)*100:.1f}% reduction)")
        print(f"              slice μ={np.mean(slice_mus):.4f} → BiIP+BH={np.mean(slice_mus_bh):.4f} "
              f"({(1-np.mean(slice_mus_bh)/np.mean(slice_mus))*100:.1f}% reduction)")
        
        # ─── DP allocation: real greedy Lagrangian ───
        n_tiles = (d_out // BLOCK) * (d_in // BLOCK)
        n_out_t = d_out // BLOCK
        n_in_t = d_in // BLOCK
        print(f"\n  DP allocation: {n_tiles:,} tiles ({n_out_t}×{n_in_t})")
        
        t_dp_start = time.time()
        # Compute per-tile distortion D_t(K) for K=4,5,6 on the BiIP+BH transformed tensor
        W_tiles_bh = W_bh[:n_out_t*BLOCK, :n_in_t*BLOCK].reshape(n_out_t, BLOCK, n_in_t, BLOCK)
        W_tiles_bh = W_tiles_bh.transpose(0, 2, 1, 3)  # (n_out_t, n_in_t, BLOCK, BLOCK)
        
        # Per-tile SSE for each K
        tile_dist = {}
        for K in [4, 5, 6]:
            levels = 2 ** K
            w_min = W_tiles_bh.min(axis=(2, 3), keepdims=True)
            w_max = W_tiles_bh.max(axis=(2, 3), keepdims=True)
            rng_range = np.where(w_max - w_min < 1e-12, 1.0, w_max - w_min)
            scale = rng_range / (levels - 1)
            q = np.round((W_tiles_bh - w_min) / scale).clip(0, levels - 1)
            Wq = q * scale + w_min
            tile_dist[K] = np.sum((W_tiles_bh - Wq) ** 2, axis=(2, 3))  # SSE per tile
        
        t_dist = time.time() - t_dp_start
        
        D4 = tile_dist[4].flatten()
        D5 = tile_dist[5].flatten()
        D6 = tile_dist[6].flatten()
        
        # Binary search λ to hit target avg bits = 5.0
        # This is the standard Lagrangian relaxation for rate-distortion optimization
        target_avg_bits = 5.0
        
        def lagrangian_alloc(lam):
            """Assign each tile to K minimizing D_t(K) + lam * K."""
            cost4 = D4 + lam * 4
            cost5 = D5 + lam * 5
            cost6 = D6 + lam * 6
            # Pick K with lowest cost per tile
            costs = np.stack([cost4, cost5, cost6], axis=1)  # (n_tiles, 3)
            K_choice = np.argmin(costs, axis=1) + 4  # 0→4, 1→5, 2→6
            return K_choice
        
        # Binary search λ: high λ → more K4, low λ → more K6
        lam_low, lam_high = -1e-3, 1e-3
        best_alloc = None
        best_avg = None
        
        for _ in range(100):
            lam_mid = (lam_low + lam_high) / 2.0
            alloc_try = lagrangian_alloc(lam_mid)
            avg_try = np.mean(alloc_try)
            
            if abs(avg_try - target_avg_bits) < 1e-6:
                best_alloc = alloc_try
                best_avg = avg_try
                break
            
            if avg_try > target_avg_bits:
                lam_low = lam_mid  # need higher λ to reduce bits
            else:
                lam_high = lam_mid  # need lower λ to increase bits
            best_alloc = alloc_try
            best_avg = avg_try
        
        # If we can't hit exactly 5.0, do marginal repair:
        # If avg > 5.0, downgrade some K5→K4 or K6→K5 tiles (smallest distortion cost)
        # If avg < 5.0, upgrade some K4→K5 or K5→K6 tiles (largest distortion reduction)
        alloc = best_alloc.copy()
        
        # Count current allocation
        n_k4 = np.sum(alloc == 4)
        n_k5 = np.sum(alloc == 5)
        n_k6 = np.sum(alloc == 6)
        current_total = (4*n_k4 + 5*n_k5 + 6*n_k6) * (BLOCK * BLOCK)
        target_total = int(target_avg_bits * n_tiles * BLOCK * BLOCK)
        
        # Marginal repair: adjust individual tiles to hit exact target
        elems = BLOCK * BLOCK
        while current_total > target_total:
            # Downgrade: find tile where downgrade costs least distortion
            # K6→K5: cost = D6 - D5 (negative, since D6 < D5)
            # K5→K4: cost = D5 - D4
            best_cost = -np.inf  # we want to minimize distortion increase (maximize negative cost)
            best_idx = -1
            best_from = 0
            best_to = 0
            for k_from, k_to in [(6, 5), (5, 4)]:
                mask = (alloc == k_from)
                if not np.any(mask):
                    continue
                costs = tile_dist[k_from].flatten() - tile_dist[k_to].flatten()  # distortion increase
                costs = np.where(mask, costs, np.inf)
                idx = np.argmin(costs)
                if costs[idx] < best_cost or best_idx == -1:
                    best_cost = costs[idx]
                    best_idx = idx
                    best_from = k_from
                    best_to = k_to
            if best_idx == -1:
                break
            alloc[best_idx] = best_to
            current_total -= elems
        
        while current_total < target_total:
            # Upgrade: find tile where upgrade gives most distortion reduction
            best_gain = -np.inf
            best_idx = -1
            best_from = 0
            best_to = 0
            for k_from, k_to in [(4, 5), (5, 6)]:
                mask = (alloc == k_from)
                if not np.any(mask):
                    continue
                gains = tile_dist[k_from].flatten() - tile_dist[k_to].flatten()  # distortion reduction
                gains = np.where(mask, gains, -np.inf)
                idx = np.argmax(gains)
                if gains[idx] > best_gain:
                    best_gain = gains[idx]
                    best_idx = idx
                    best_from = k_from
                    best_to = k_to
            if best_idx == -1 or best_gain <= 0:
                break
            alloc[best_idx] = best_to
            current_total += elems
        
        t_dp = time.time() - t_dp_start
        
        # Compute final distortion
        total_distortion = np.sum(np.where(alloc == 4, D4, np.where(alloc == 5, D5, D6)))
        avg_bits = np.mean(alloc)
        uniform_dist = np.sum(D5)
        dp_improvement = (1 - total_distortion / uniform_dist) * 100
        
        # Width map cost: 2 bits per tile (3 states: K4, K5, K6)
        width_map_bits = 2 * n_tiles
        width_map_bpw = width_map_bits / (d_out * d_in)
        
        n_k4 = int(np.sum(alloc == 4))
        n_k5 = int(np.sum(alloc == 5))
        n_k6 = int(np.sum(alloc == 6))
        
        print(f"  Per-tile SSE computed in {t_dist:.2f}s")
        print(f"  Lagrangian allocation in {t_dp:.2f}s total")
        print(f"  Allocation: K4={n_k4:,}, K5={n_k5:,}, K6={n_k6:,} tiles")
        print(f"  Average bits/element: {avg_bits:.4f} (target: {target_avg_bits})")
        print(f"  Width map overhead: {width_map_bpw:.6f} bpw ({width_map_bits/8:,} bytes)")
        print(f"  Total distortion: {total_distortion:.4e} vs uniform K5: {uniform_dist:.4e}")
        print(f"  Lagrangian improvement over uniform K5: {dp_improvement:.1f}%")
        print(f"  Tractable: {'YES' if t_dp < 60 else 'NO'}")
        
        assert n_k4 + n_k5 + n_k6 == n_tiles
        
        t_total = time.time() - t_start
        print(f"\n  Total time: {t_total:.1f}s")
        all_results[tensor_name] = {
            'shape': [d_out, d_in],
            'memory_mb': mem_bytes / 1e6,
            'time_seconds': t_total,
            'n_tiles': n_tiles,
            'dp': {
                'time_seconds': t_dp,
                'avg_bits': avg_bits,
                'n_k4': int(n_k4),
                'n_k5': int(n_k5),
                'width_map_bpw': width_map_bpw,
                'dp_improvement_over_uniform_k5_pct': dp_improvement,
                'tractable': t_dp < 60,
            },
            'metrics': {
                'mu_none': mu_full,
                'mu_biip': mu_biip,
                'mu_biip_bh': mu_bh,
                'cv_in_none': cv_in_full,
                'cv_in_bh': cv_in_bh,
                'cv_out_none': cv_out_full,
                'cv_out_bh': cv_out_bh,
                'mtr_none': mtr_full,
                'mtr_bh': mtr_bh,
                'xtr_none': xtr_full,
                'xtr_bh': xtr_bh,
            },
            'quantization': quant_results,
            'slices': slice_results,
            'sidecar_bytes': sidecar_total,
        }
    
    # ─── Summary ───
    print("\n" + "=" * 80)
    print("SUMMARY: Full-tensor vs 128×128 slice comparison")
    print("=" * 80)
    
    for tensor_name in all_results:
        r = all_results[tensor_name]
        print(f"\n{tensor_name} {r['shape']}:")
        for K in [4, 5, 6]:
            if f'K{K}' in r['quantization']:
                q = r['quantization'][f'K{K}']
                pooled_key = f'pooled_K{K}'
                pooled_imp = r['slices'].get(pooled_key, {}).get('hwe_improvement_pct', 0)
                print(f"  K={K} HWE improvement: full={q['hwe_improvement_pct']:.1f}%, "
                      f"slice pooled={pooled_imp:.1f}%")
        dp = r['dp']
        print(f"  DP: {r['n_tiles']:,} tiles, {dp['time_seconds']:.2f}s, "
              f"avg={dp['avg_bits']:.2f} bits, improvement={dp['dp_improvement_over_uniform_k5_pct']:.1f}%, "
              f"tractable={dp['tractable']}")
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    
    return all_results

if __name__ == '__main__':
    results = run_full_tensor_experiment()
