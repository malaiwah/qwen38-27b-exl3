#!/usr/bin/env python3
"""
R3-Rotations: Two-sided incoherence + rotations for trellis quantization.

Clean-room implementation from KronQ (arXiv:2607.07964) and QuIP# equations.
Tests factorial decomposition of scaling, input rotation, and output rotation
on real Qwen3.8-27B weights with matched quantizer and exact byte budgets.

Key equations (from KronQ §4.2):
  S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}   (input-side balancing)
  S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}    (output-side balancing)
  W' = S_G @ W @ S_X                            (diagonal rescaling)
  W' = U @ W' @ V^T                             (orthogonal rotation)
  H_X' = S_X^{-1} H_X S_X^{-1}                  (transformed Hessians)
  H_G' = S_G^{-1} H_G S_G^{-1}
  H_X'' = V H_X' V^T,  H_G'' = U H_G' U^T

Hessian-weighted error objective:
  tr(H_G @ E @ H_X @ E^T)  where  E = W - W_hat

Rotation invariance (Theorem 1 of KronQ):
  tr(H_G E H_X E^T) = tr(H_G'' E' H_X'' E'^T)
  where E' = W' - W_hat' is the error in the transformed space.
  Rotations preserve the objective exactly; they make quantization easier
  by making weights more incoherent (i.i.d.-like), not by changing the objective.
"""

import json
import numpy as np
from itertools import product
import os

# ─── Paths ───
WEIGHTS_PATH = "/Users/mbelleau/Projects/cleanroom/qwen38_real_weights.npz"
RESULTS_PATH = "/Users/mbelleau/Projects/qwen38-research-r3-rotations/receipts/research/r3-rotations-results.json"

# ─── Quantizer ───
# Matched per-column uniform quantization for ALL arms.
# Each column gets its own scale and zero-point (same granularity as GPTQ).

def quantize_per_column(W, K):
    """Per-column uniform quantization with 2^K levels.
    
    W: (d_out, d_in) weight matrix
    K: bits per element (2^K quantization levels)
    Returns: quantized matrix (d_out, d_in), metadata bytes
    """
    levels = 2 ** K
    d_out, d_in = W.shape
    
    Wq = np.zeros_like(W)
    # Per-column scale (min-max per column)
    for j in range(d_in):
        col = W[:, j]
        w_min, w_max = col.min(), col.max()
        if w_max - w_min < 1e-12:
            Wq[:, j] = col
            continue
        scale = (w_max - w_min) / (levels - 1)
        q = np.round((col - w_min) / scale).clip(0, levels - 1)
        Wq[:, j] = q * scale + w_min
    
    # Metadata: 1 scale + 1 zero per column = 2 * d_in * 4 bytes (float32)
    # Plus per-column min/max = 2 * d_in * 4 bytes
    # Total metadata: 4 * d_in * 4 = 16 * d_in bytes
    # But for budget: the quantized values take d_out * d_in * K / 8 bytes
    payload_bytes = d_out * d_in * K / 8
    meta_bytes = 2 * d_in * 4  # per-column min and max as float32
    
    return Wq, payload_bytes, meta_bytes


def quantize_per_column_vectorized(W, K):
    """Vectorized per-column uniform quantization."""
    levels = 2 ** K
    d_out, d_in = W.shape
    
    w_min = W.min(axis=0, keepdims=True)
    w_max = W.max(axis=0, keepdims=True)
    rng = w_max - w_min
    rng = np.where(rng < 1e-12, 1.0, rng)
    
    scale = rng / (levels - 1)
    q = np.round((W - w_min) / scale).clip(0, levels - 1)
    Wq = q * scale + w_min
    
    payload_bytes = d_out * d_in * K / 8
    meta_bytes = 2 * d_in * 4  # per-column min/max as float32
    
    return Wq, payload_bytes, meta_bytes

def quantize_per_tile(W, K, tile_size=16):
    """Per-tile uniform quantization with 2^K levels.
    Each tile_size × tile_size block gets its own scale (min-max).
    This is the quantizer that interacts with rotations and permutations:
    rotations reduce within-tile dynamic range, permutations group similar scales.
    """
    levels = 2 ** K
    d_out, d_in = W.shape
    Wq = np.zeros_like(W)
    n_tiles_out = (d_out + tile_size - 1) // tile_size
    n_tiles_in = (d_in + tile_size - 1) // tile_size
    
    for ti in range(n_tiles_out):
        for tj in range(n_tiles_in):
            r0, r1 = ti * tile_size, min((ti + 1) * tile_size, d_out)
            c0, c1 = tj * tile_size, min((tj + 1) * tile_size, d_in)
            tile = W[r0:r1, c0:c1]
            w_min, w_max = tile.min(), tile.max()
            if w_max - w_min < 1e-12:
                Wq[r0:r1, c0:c1] = tile
                continue
            scale = (w_max - w_min) / (levels - 1)
            q = np.round((tile - w_min) / scale).clip(0, levels - 1)
            Wq[r0:r1, c0:c1] = q * scale + w_min
    
    payload_bytes = d_out * d_in * K / 8
    # Per-tile min/max: 2 floats per tile
    meta_bytes = 2 * n_tiles_out * n_tiles_in * 4
    
    return Wq, payload_bytes, meta_bytes


# ─── Hadamard transform ───

def hadamard_matrix(n):
    """Generate Sylvester-type Hadamard matrix of size n (must be power of 2)."""
    assert (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def signed_random_hadamard(n, rng):
    """Signed randomized Hadamard: H @ diag(±1).
    Sidecar: n bits for signs (or a seed if using deterministic PRNG).
    """
    H = hadamard_matrix(n)
    signs = rng.choice([-1.0, 1.0], size=n)
    return H @ np.diag(signs), signs


# ─── Butterfly / Kronecker rotation ───

def butterfly_rotation(n, rng):
    """Kronecker/butterfly orthogonal: U = U_1 ⊗ U_2 aligned to 16×16 tiles.
    For n = a*b, U = U_a ⊗ U_b where U_a, U_b are random orthogonal.
    Application cost: O(n log n) via butterfly.
    Sidecar: a^2 + b^2 floats (stored factors).
    """
    # Find factorization closest to sqrt(n)
    a = int(np.sqrt(n))
    while n % a != 0:
        a -= 1
    b = n // a
    
    # Random orthogonal via QR of random Gaussian
    X_a = rng.standard_normal((a, a))
    U_a, _ = np.linalg.qr(X_a)
    X_b = rng.standard_normal((b, b))
    U_b, _ = np.linalg.qr(X_b)
    
    U = np.kron(U_a, U_b)
    sidecar_bytes = (a * a + b * b) * 4  # float32 factors
    
    return U, sidecar_bytes


# ─── Block-Givens / Householder outlier annihilation ───

def block_givens_rotation(n, W, block_size=16, n_reflectors=3, rng=None):
    """Within blocks of size block_size, greedily choose orthogonal reflectors
    minimizing tile max/RMS. Fuse as butterflies.
    
    For each block, find the Householder reflector that reduces the max magnitude
    in the block, apply it, repeat for n_reflectors.
    Sidecar: n_reflectors * (block_size) floats per block.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    U = np.eye(n)
    sidecar_bytes = 0
    
    n_blocks = n // block_size
    for blk in range(n_blocks):
        start = blk * block_size
        end = start + block_size
        
        # Work on the rows of W in this block
        # We want to find reflectors that reduce max magnitude
        block_W = W[:, start:end].copy()  # (d_out, block_size) — but we rotate the block_size dim
        # Actually, for output-side rotation, we rotate rows of W
        # For input-side, we rotate columns
        # Here we're building a rotation on the n-dim space
        
        U_block = np.eye(block_size)
        
        for _ in range(n_reflectors):
            # Find the direction of the largest-magnitude element
            # Use Householder to zero out the largest outlier
            abs_block = np.abs(block_W)
            flat_idx = np.argmax(abs_block)
            r, c = np.unravel_index(flat_idx, block_W.shape)
            
            # Householder reflector on the block: reflect the row with the max
            # to spread its magnitude
            v = block_W[r, :].copy()
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            v = v / norm_v
            
            # Householder: H = I - 2 v v^T
            # This maps v to e_1 (first basis vector), spreading magnitude
            H_house = np.eye(block_size) - 2.0 * np.outer(v, v)
            U_block = U_block @ H_house
            block_W = block_W @ H_house
        
        U[start:end, start:end] = U_block
        sidecar_bytes += n_reflectors * block_size * 4  # float32 per reflector per block
    
    return U, sidecar_bytes


# ─── Joint generalized-eigen rotation ───

def generalized_eigen_rotation(H_X, WtW, n_dims):
    """Find directions important to both activations and weights.
    Generalized eigenvectors of (H_X, W^T W + εI).
    The rotation V diagonalizes both H_X and W^T W simultaneously (approx).
    """
    eps = 1e-4 * max(np.trace(WtW), np.trace(H_X)) / n_dims
    A = H_X.astype(np.float64)
    B = (WtW + eps * np.eye(n_dims)).astype(np.float64)
    
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(A, B)
        V = eigvecs
    except ImportError:
        # Fallback: Cholesky-based generalized eigen
        L = np.linalg.cholesky(B)
        L_inv = np.linalg.inv(L)
        M = L_inv @ A @ L_inv.T
        eigvals, w = np.linalg.eigh(M)
        V = L_inv.T @ w
    
    # Normalize columns to unit norm (guard against degenerate eigenvectors)
    norms = np.linalg.norm(V, axis=0)
    norms = np.maximum(norms, 1e-12)
    V = V / norms[np.newaxis, :]
    
    # Sidecar: full V matrix (n × n floats)
    sidecar_bytes = n_dims * n_dims * 4
    return V, sidecar_bytes


# ─── BiIP diagonal balancing ───

def biip_scaling(W, H_X, H_G):
    """Two-sided diagonal balancing (KronQ Eq. 11).
    S_X = diag(H_X_jj / ||W_{:,j}||^2)^{1/4}
    S_G = diag(H_G_ii / ||W_{i,:}||^2)^{1/4}
    W' = S_G @ W @ S_X
    
    Returns: S_G, S_X, W_transformed, sidecar_bytes
    """
    d_out, d_in = W.shape
    
    # Input-side: column norms
    col_norms_sq = np.sum(W ** 2, axis=0)  # ||W_{:,j}||^2
    col_norms_sq = np.maximum(col_norms_sq, 1e-12)
    sx_diag = (np.diag(H_X) / col_norms_sq) ** 0.25
    # Clip to prevent extreme scaling (allow up to 10x range)
    sx_diag = np.clip(sx_diag, 0.1, 10.0)
    S_X = np.diag(sx_diag)
    
    # Output-side: row norms
    row_norms_sq = np.sum(W ** 2, axis=1)  # ||W_{i,:}||^2
    row_norms_sq = np.maximum(row_norms_sq, 1e-12)
    sg_diag = (np.diag(H_G) / row_norms_sq) ** 0.25
    sg_diag = np.clip(sg_diag, 0.1, 10.0)
    S_G = np.diag(sg_diag)
    
    W_transformed = S_G @ W @ S_X
    
    # Sidecar: d_in + d_out floats for the diagonal scales
    sidecar_bytes = (d_in + d_out) * 4
    
    return S_G, S_X, W_transformed, sidecar_bytes


# ─── Transform pipeline ───

def apply_transform(W, H_X, H_G, config, rng):
    """Apply a transform pipeline based on config.
    
    config: dict with keys:
        'scaling': 'none' | 'biip'
        'input_rotation': 'none' | 'hadamard' | 'butterfly' | 'block_givens' | 'gen_eigen' | 'random_orth'
        'output_rotation': 'none' | 'hadamard' | 'butterfly' | 'block_givens' | 'random_orth'
        'hadamard_size': int (power of 2, default 128)
        'block_size': int (default 16)
    
    Returns: (W_transformed, H_X_transformed, H_G_transformed, U, V, S_G, S_X, sidecar_bytes)
    """
    d_out, d_in = W.shape
    sidecar = 0
    
    # Step 1: Diagonal balancing
    S_G = np.eye(d_out)
    S_X = np.eye(d_in)
    W_t = W.copy()
    H_X_t = H_X.copy()
    H_G_t = H_G.copy()
    
    if config.get('scaling', 'none') == 'biip':
        S_G, S_X, W_t, sc_bytes = biip_scaling(W, H_X, H_G)
        sidecar += sc_bytes
        # Transform Hessians
        S_G_inv = np.linalg.inv(S_G)
        S_X_inv = np.linalg.inv(S_X)
        H_X_t = S_X_inv @ H_X_t @ S_X_inv
        H_G_t = S_G_inv @ H_G_t @ S_G_inv
    
    # Step 2: Input rotation (V operates on columns/input dim)
    V = np.eye(d_in)
    inp_rot = config.get('input_rotation', 'none')
    hadamard_size = config.get('hadamard_size', d_in)
    
    if inp_rot == 'hadamard':
        V, signs = signed_random_hadamard(d_in, rng)
        sidecar += d_in // 8 + 1  # sign bits
    elif inp_rot == 'butterfly':
        V, sc = butterfly_rotation(d_in, rng)
        sidecar += sc
    elif inp_rot == 'block_givens':
        V, sc = block_givens_rotation(d_in, W_t, config.get('block_size', 16), rng=rng)
        sidecar += sc
    elif inp_rot == 'gen_eigen':
        WtW = W_t.T @ W_t
        V, sc = generalized_eigen_rotation(H_X_t, WtW, d_in)
        sidecar += sc
    elif inp_rot == 'random_orth':
        X = rng.standard_normal((d_in, d_in))
        V, _ = np.linalg.qr(X)
        sidecar += 4  # just a seed
    
    if inp_rot != 'none':
        W_t = W_t @ V.T
        H_X_t = V @ H_X_t @ V.T
    
    # Step 3: Output rotation (U operates on rows/output dim)
    U = np.eye(d_out)
    out_rot = config.get('output_rotation', 'none')
    
    if out_rot == 'hadamard':
        U, signs = signed_random_hadamard(d_out, rng)
        sidecar += d_out // 8 + 1
    elif out_rot == 'butterfly':
        U, sc = butterfly_rotation(d_out, rng)
        sidecar += sc
    elif out_rot == 'block_givens':
        # For output, transpose the logic
        U, sc = block_givens_rotation(d_out, W_t.T, config.get('block_size', 16), rng=rng)
        sidecar += sc
    elif out_rot == 'random_orth':
        X = rng.standard_normal((d_out, d_out))
        U, _ = np.linalg.qr(X)
        sidecar += 4
    
    if out_rot != 'none':
        W_t = U @ W_t
        H_G_t = U @ H_G_t @ U.T
    
    return W_t, H_X_t, H_G_t, U, V, S_G, S_X, sidecar


def inverse_transform(W_quantized, U, V, S_G, S_X):
    """Inverse the transform to get back to original space.
    W_hat = S_G^{-1} @ U^T @ W_quantized @ V @ S_X^{-1}
    """
    S_G_inv = np.linalg.inv(S_G)
    S_X_inv = np.linalg.inv(S_X)
    return S_G_inv @ U.T @ W_quantized @ V @ S_X_inv

def weight_mse(W, W_hat):
    """Raw weight MSE."""
    return np.mean((W - W_hat) ** 2)

def hessian_weighted_error(W, W_hat, H_X, H_G):
    """Output-covariance-weighted error: tr(H_G @ E @ H_X @ E^T) where E = W - W_hat.
    
    NOTE: H_G here is an output-covariance proxy (Y^T Y / N), NOT the true
    gradient covariance (Fisher) H_G = E[g g^T]. True H_G requires a backward
    pass. We use this proxy because CPU-only experiments cannot run backward.
    The proxy captures output channel correlation structure but is NOT the
    KronQ Hessian. All results should be interpreted as output-covariance-proxy
    weighted error, not true Hessian-weighted error.
    """
    E = W - W_hat
    return np.trace(H_G @ E @ H_X @ E.T)

def hessian_weighted_error_diag(W, W_hat, H_X_diag, H_G_diag):
    """Diagonal approximation: sum_i sum_j H_G_i * H_X_j * E_ij^2.
    This is the per-element weighted error using only diagonal Hessians.
    """
    E = W - W_hat
    weights = np.outer(H_G_diag, H_X_diag)  # (d_out, d_in)
    return np.sum(weights * E ** 2)


def max_tile_range(W, tile_size=16):
    """Max range across all tile_size × tile_size tiles."""
    d_out, d_in = W.shape
    max_range = 0
    for i in range(0, d_out, tile_size):
        for j in range(0, d_in, tile_size):
            tile = W[i:i+tile_size, j:j+tile_size]
            r = tile.max() - tile.min()
            max_range = max(max_range, r)
    return max_range


def mean_tile_range(W, tile_size=16):
    """Mean range across all tiles."""
    d_out, d_in = W.shape
    ranges = []
    for i in range(0, d_out, tile_size):
        for j in range(0, d_in, tile_size):
            tile = W[i:i+tile_size, j:j+tile_size]
            ranges.append(tile.max() - tile.min())
    return np.mean(ranges)


def incoherence_measure(W):
    """μ(W) = sqrt(d_out * d_in) / ||W||_F * max|W_ij|.
    Lower μ → more incoherent → better for uniform quantization.
    """
    d_out, d_in = W.shape
    return np.sqrt(d_out * d_in) / (np.linalg.norm(W) + 1e-12) * np.max(np.abs(W))


def cv_column_norms(W):
    """Coefficient of variation of column norms (CV_in in KronQ)."""
    col_norms = np.linalg.norm(W, axis=0)
    return np.std(col_norms) / (np.mean(col_norms) + 1e-12)


def cv_row_norms(W):
    """Coefficient of variation of row norms (CV_out in KronQ)."""
    row_norms = np.linalg.norm(W, axis=1)
    return np.std(row_norms) / (np.mean(row_norms) + 1e-12)


# ─── Synthetic Hessian generation ───

def synthetic_hessians(W, n_samples=512, outlier_fraction=0.05, outlier_scale=10.0, seed=42):
    """Generate synthetic activation Hessian H_X and output Hessian proxy H_G.
    
    H_X = X^T X / N where X has outlier channels (realistic activation pattern).
    H_G ≈ Y^T Y / N where Y = W @ X (output covariance proxy).
    """
    rng = np.random.default_rng(seed)
    d_out, d_in = W.shape
    
    # Synthetic activations: Gaussian + outliers on a few channels
    X = rng.standard_normal((d_in, n_samples))
    # Add outliers to ~5% of channels
    n_outliers = max(1, int(d_in * outlier_fraction))
    outlier_channels = rng.choice(d_in, n_outliers, replace=False)
    X[outlier_channels, :] *= outlier_scale
    
    H_X = (X @ X.T / n_samples).astype(np.float64)
    
    # Output covariance as H_G proxy
    Y = W @ X
    H_G = (Y @ Y.T / n_samples).astype(np.float64)
    
    # Normalize Hessians to have mean diagonal = 1 (relative scale only)
    # This prevents overflow in BiIP scaling while preserving relative structure
    H_X *= d_in / np.trace(H_X)
    H_G *= d_out / np.trace(H_G)
    
    # Regularize for numerical stability
    H_X += 1e-6 * np.eye(d_in)
    H_G += 1e-6 * np.eye(d_out)
    
    return H_X, H_G


# ─── Main experiment ───

def run_experiment():
    """Run the full factorial rotation experiment with multi-slice sampling."""
    
    print("=" * 80)
    print("R3-Rotations: Two-sided incoherence + rotations")
    print("Clean-room from KronQ (arXiv:2607.07964) and QuIP#")
    print("NOTE: H_G is an output-covariance proxy (Y^T Y / N), NOT true gradient covariance")
    print("=" * 80)
    
    # Load real weights
    print("\nLoading real weights...")
    weights = np.load(WEIGHTS_PATH)
    
    tensors = {
        'L0_gate_K5': ('L0_gate', 5),
        'L0_down_K6': ('L0_down', 6),
        'L55_gate_K5': ('L55_gate', 5),
        'L55_down_K6': ('L55_down', 6),
        'L0_qkv_K6': ('L0_qkv', 6),
        'L0_out_K6': ('L0_out', 6),
    }
    
    # Multi-slice sampling: first, middle, last 128×128 aligned slices
    n_slices = 3
    
    all_results = {}
    
    for tensor_name, (weight_key, K) in tensors.items():
        print(f"\n{'─' * 60}")
        print(f"Tensor: {tensor_name} (K={K})")
        print(f"{'─' * 60}")
        
        W_full = weights[weight_key]
        d_out_full, d_in_full = W_full.shape
        
        d = 128
        d_out = min(d, d_out_full)
        d_in = min(d, d_in_full)
        
        # Three deterministic slices: first, middle, last
        slice_offsets_out = [0, (d_out_full - d_out) // 2, d_out_full - d_out]
        slice_offsets_in = [0, (d_in_full - d_in) // 2, d_in_full - d_in]
        
        slice_results = []
        
        for slice_idx in range(n_slices):
            off_out = slice_offsets_out[slice_idx]
            off_in = slice_offsets_in[slice_idx]
            W = W_full[off_out:off_out+d_out, off_in:off_in+d_in].astype(np.float64)
            
            if slice_idx == 0:
                print(f"  Shape: {W.shape}, K={K}")
                print(f"  Weight stats (slice 0): min={W.min():.6f}, max={W.max():.6f}, std={W.std():.6f}")
            
            # Generate output-covariance proxy Hessians
            H_X, H_G = synthetic_hessians(W, n_samples=512, seed=42)
            H_X_diag = np.diag(H_X)
            H_G_diag = np.diag(H_G)
            
            # Noise floor
            W_noise, _, _ = quantize_per_column_vectorized(W, 16)
            noise_mse = weight_mse(W, W_noise)
            noise_hwerr = hessian_weighted_error(W, W_noise, H_X, H_G)
            
            # ─── Factorial design ───
            # Clean 2×3×3 factorial: 18 unique configs
            # Plus extra arms: butterfly, block_givens, gen_eigen
            configs = []
            for scaling in ['none', 'biip']:
                for inp_rot in ['none', 'hadamard', 'random_orth']:
                    for out_rot in ['none', 'hadamard', 'random_orth']:
                        configs.append({
                            'name': f"scale={scaling}|in={inp_rot}|out={out_rot}",
                            'scaling': scaling,
                            'input_rotation': inp_rot,
                            'output_rotation': out_rot,
                        })
            
            # Extra arms (not in clean factorial)
            for rot_type in ['butterfly', 'block_givens']:
                configs.append({
                    'name': f"scale=biip|in=hadamard|out={rot_type}",
                    'scaling': 'biip',
                    'input_rotation': 'hadamard',
                    'output_rotation': rot_type,
                })
                configs.append({
                    'name': f"scale=biip|in={rot_type}|out=hadamard",
                    'scaling': 'biip',
                    'input_rotation': rot_type,
                    'output_rotation': 'hadamard',
                })
            
            configs.append({
                'name': 'scale=biip|in=gen_eigen|out=hadamard',
                'scaling': 'biip',
                'input_rotation': 'gen_eigen',
                'output_rotation': 'hadamard',
            })
            
            # NOTE: No duplicate BiIP_full — scale=biip|in=hadamard|out=hadamard IS the BiIP recipe
            
            # Pre-generate rotation matrices with INDEPENDENT seeds per factor level
            # so the factorial is clean (input rotation doesn't affect output rotation RNG)
            rng_in = np.random.default_rng(100)  # seed for input rotations
            rng_out = np.random.default_rng(200)  # seed for output rotations
            
            # Pre-generate all rotation matrices
            input_rotations = {}
            output_rotations = {}
            
            for rot_name in ['hadamard', 'random_orth', 'butterfly', 'block_givens', 'gen_eigen']:
                if rot_name == 'hadamard':
                    V, signs = signed_random_hadamard(d_in, rng_in)
                    input_rotations[rot_name] = V
                elif rot_name == 'random_orth':
                    X = rng_in.standard_normal((d_in, d_in))
                    V, _ = np.linalg.qr(X)
                    input_rotations[rot_name] = V
                elif rot_name == 'butterfly':
                    V, _ = butterfly_rotation(d_in, rng_in)
                    input_rotations[rot_name] = V
                elif rot_name == 'block_givens':
                    V, _ = block_givens_rotation(d_in, W, block_size=16, rng=rng_in)
                    input_rotations[rot_name] = V
                elif rot_name == 'gen_eigen':
                    WtW = W.T @ W
                    V, _ = generalized_eigen_rotation(H_X, WtW, d_in)
                    input_rotations[rot_name] = V
            
            for rot_name in ['hadamard', 'random_orth', 'butterfly', 'block_givens']:
                if rot_name == 'hadamard':
                    U, signs = signed_random_hadamard(d_out, rng_out)
                    output_rotations[rot_name] = U
                elif rot_name == 'random_orth':
                    X = rng_out.standard_normal((d_out, d_out))
                    U, _ = np.linalg.qr(X)
                    output_rotations[rot_name] = U
                elif rot_name == 'butterfly':
                    U, _ = butterfly_rotation(d_out, rng_out)
                    output_rotations[rot_name] = U
                elif rot_name == 'block_givens':
                    U, _ = block_givens_rotation(d_out, W.T, block_size=16, rng=rng_out)
                    output_rotations[rot_name] = U
            
            tensor_results = {}
            
            for config in configs:
                name = config['name']
                
                # Apply transform using pre-generated rotations
                W_t = W.copy()
                H_X_t = H_X.copy()
                H_G_t = H_G.copy()
                sidecar = 0
                
                # Step 1: Diagonal balancing
                S_G = np.eye(d_out)
                S_X = np.eye(d_in)
                if config.get('scaling', 'none') == 'biip':
                    S_G, S_X, W_t, sc_bytes = biip_scaling(W, H_X, H_G)
                    sidecar += sc_bytes
                    S_G_inv = np.linalg.inv(S_G)
                    S_X_inv = np.linalg.inv(S_X)
                    H_X_t = S_X_inv @ H_X_t @ S_X_inv
                    H_G_t = S_G_inv @ H_G_t @ S_G_inv
                
                # Step 2: Input rotation (pre-generated)
                V = np.eye(d_in)
                inp_rot = config.get('input_rotation', 'none')
                if inp_rot in input_rotations:
                    V = input_rotations[inp_rot]
                    if inp_rot == 'hadamard':
                        sidecar += d_in // 8 + 1
                    elif inp_rot == 'random_orth':
                        sidecar += 4
                    elif inp_rot == 'butterfly':
                        a = int(np.sqrt(d_in))
                        while d_in % a != 0: a -= 1
                        sidecar += (a*a + (d_in//a)**2) * 4
                    elif inp_rot == 'block_givens':
                        sidecar += 3 * 16 * (d_in // 16) * 4
                    elif inp_rot == 'gen_eigen':
                        sidecar += d_in * d_in * 4
                    W_t = W_t @ V.T
                    H_X_t = V @ H_X_t @ V.T
                
                # Step 3: Output rotation (pre-generated)
                U = np.eye(d_out)
                out_rot = config.get('output_rotation', 'none')
                if out_rot in output_rotations:
                    U = output_rotations[out_rot]
                    if out_rot == 'hadamard':
                        sidecar += d_out // 8 + 1
                    elif out_rot == 'random_orth':
                        sidecar += 4
                    elif out_rot == 'butterfly':
                        a = int(np.sqrt(d_out))
                        while d_out % a != 0: a -= 1
                        sidecar += (a*a + (d_out//a)**2) * 4
                    elif out_rot == 'block_givens':
                        sidecar += 3 * 16 * (d_out // 16) * 4
                    W_t = U @ W_t
                    H_G_t = U @ H_G_t @ U.T
                
                # Quantize in transformed space — run BOTH quantizers
                # Per-column: same as GPTQ granularity (may have BiIP-scaling tautology)
                # Per-tile 16×16: interacts with rotations (reduces within-tile range)
                results_quant = {}
                for quant_name, quant_fn in [('per_col', quantize_per_column_vectorized),
                                              ('per_tile16', lambda w, k: quantize_per_tile(w, k, 16))]:
                    W_q_t, payload, meta = quant_fn(W_t, K)
                    total_bytes = payload + meta + sidecar
                    n_elements = d_out * d_in
                    effective_bits = total_bytes * 8 / n_elements
                    
                    W_hat = inverse_transform(W_q_t, U, V, S_G, S_X)
                    
                    mse = weight_mse(W, W_hat)
                    hw_err = hessian_weighted_error(W, W_hat, H_X, H_G)
                    hw_err_diag = hessian_weighted_error_diag(W, W_hat, H_X_diag, H_G_diag)
                    
                    results_quant[quant_name] = {
                        'mse': float(mse),
                        'hw_error': float(hw_err),
                        'hw_error_diag': float(hw_err_diag),
                        'payload_bytes': float(payload),
                        'meta_bytes': float(meta),
                        'total_bytes': float(total_bytes),
                        'effective_bits_per_elem': float(effective_bits),
                    }
                
                # Use per-column as primary for backward compat
                mse = results_quant['per_col']['mse']
                hw_err = results_quant['per_col']['hw_error']
                hw_err_diag = results_quant['per_col']['hw_error_diag']
                payload = results_quant['per_col']['payload_bytes']
                meta = results_quant['per_col']['meta_bytes']
                total_bytes = results_quant['per_col']['total_bytes']
                effective_bits = results_quant['per_col']['effective_bits_per_elem']
                
                mu_W = incoherence_measure(W_t)
                cv_in = cv_column_norms(W_t)
                cv_out = cv_row_norms(W_t)
                max_range = max_tile_range(W_t)
                mean_range = mean_tile_range(W_t)
                max_range_orig = max_tile_range(W)
                
                tensor_results[name] = {
                    'mse': float(mse),
                    'hw_error': float(hw_err),
                    'hw_error_diag': float(hw_err_diag),
                    'incoherence_mu': float(mu_W),
                    'cv_in': float(cv_in),
                    'cv_out': float(cv_out),
                    'max_tile_range': float(max_range),
                    'mean_tile_range': float(mean_range),
                    'max_tile_range_reduction_pct': float((1 - max_range / max_range_orig) * 100) if max_range_orig > 0 else 0,
                    'payload_bytes': float(payload),
                    'meta_bytes': float(meta),
                    'sidecar_bytes': float(sidecar),
                    'total_bytes': float(total_bytes),
                    'effective_bits_per_elem': float(effective_bits),
                    'K': K,
                    'config': {k: v for k, v in config.items() if k != 'name'},
                    'slice_idx': slice_idx,
                    'per_tile16': results_quant['per_tile16'],
                }
            
            # Compute baseline for comparison
            baseline = tensor_results.get('scale=none|in=none|out=none', {})
            baseline_mse = baseline.get('mse', 1)
            baseline_hw = baseline.get('hw_error', 1)
            baseline_hw_diag = baseline.get('hw_error_diag', 1)
            
            for name, res in tensor_results.items():
                res['mse_reduction_pct'] = float((1 - res['mse'] / baseline_mse) * 100) if baseline_mse > 0 else 0
                res['hw_error_reduction_pct'] = float((1 - res['hw_error'] / baseline_hw) * 100) if baseline_hw > 0 else 0
                res['hw_error_diag_reduction_pct'] = float((1 - res['hw_error_diag'] / baseline_hw_diag) * 100) if baseline_hw_diag > 0 else 0
            
            slice_results.append({
                'slice_idx': slice_idx,
                'offset_out': off_out,
                'offset_in': off_in,
                'noise_floor': {
                    'mse': float(noise_mse),
                    'hw_error': float(noise_hwerr),
                },
                'arms': tensor_results,
            })
        
        # Print results for slice 0
        s0 = slice_results[0]
        print(f"\n  {'Arm':<50} {'MSE':>10} {'OC_err':>10} {'OC_diag':>10} {'μ(W)':>6} {'CV_in':>6} {'CV_out':>6} {'Sidecar':>8} {'Eff_bits':>8}")
        print(f"  {'─'*50} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")
        
        sorted_results = sorted(s0['arms'].items(), key=lambda x: x[1]['hw_error'])
        for name, res in sorted_results[:15]:
            print(f"  {name:<50} {res['mse']:.2e} {res['hw_error']:.2e} {res['hw_error_diag']:.2e} "
                  f"{res['incoherence_mu']:.2f} {res['cv_in']:.3f} {res['cv_out']:.3f} "
                  f"{res['sidecar_bytes']:>6.0f}B {res['effective_bits_per_elem']:.3f}")
        
        if 'scale=none|in=none|out=none' in s0['arms']:
            res = s0['arms']['scale=none|in=none|out=none']
            print(f"\n  Baseline (no transform, slice 0):")
            print(f"    MSE={res['mse']:.2e}, OC_err={res['hw_error']:.2e}")
            print(f"    μ(W)={res['incoherence_mu']:.2f}, CV_in={res['cv_in']:.3f}, CV_out={res['cv_out']:.3f}")
        
        # Compute multi-slice statistics
        print(f"\n  Multi-slice summary (n={n_slices} slices):")
        arm_names = list(slice_results[0]['arms'].keys())
        multi_slice_stats = {}
        
        for arm_name in arm_names:
            hw_errs = [sr['arms'][arm_name]['hw_error'] for sr in slice_results]
            reductions = [sr['arms'][arm_name]['hw_error_reduction_pct'] for sr in slice_results]
            mses = [sr['arms'][arm_name]['mse'] for sr in slice_results]
            
            multi_slice_stats[arm_name] = {
                'hw_error_mean': float(np.mean(hw_errs)),
                'hw_error_std': float(np.std(hw_errs)),
                'hw_reduction_mean_pct': float(np.mean(reductions)),
                'hw_reduction_min_pct': float(np.min(reductions)),
                'hw_reduction_max_pct': float(np.max(reductions)),
                'mse_mean': float(np.mean(mses)),
                'config': slice_results[0]['arms'][arm_name].get('config', {}),
            }
            
            if arm_name == 'scale=biip|in=hadamard|out=hadamard':
                print(f"    BiIP+Had+Had: HW_red = {np.mean(reductions):.1f}% (range: {np.min(reductions):.1f}%-{np.max(reductions):.1f}%)")
            elif arm_name == 'scale=none|in=none|out=none':
                print(f"    Baseline:      HW_err = {np.mean(hw_errs):.2e} ± {np.std(hw_errs):.2e}")
        
        # Find best arm across slices
        best_arm = min(multi_slice_stats.keys(), 
                       key=lambda k: multi_slice_stats[k]['hw_error_mean'])
        best = multi_slice_stats[best_arm]
        print(f"    Best arm (mean HW err): {best_arm}")
        print(f"      HW_red = {best['hw_reduction_mean_pct']:.1f}% (range: {best['hw_reduction_min_pct']:.1f}%-{best['hw_reduction_max_pct']:.1f}%)")
        
        all_results[tensor_name] = {
            'slices': slice_results,
            'multi_slice_stats': multi_slice_stats,
            'best_arm': best_arm,
            'best_arm_stats': best,
            'n_slices': n_slices,
        }
    # ─── Summary across tensors ───
    print(f"\n{'=' * 80}")
    print("SUMMARY: Best arm per tensor (by mean HW error across slices)")
    print(f"{'=' * 80}")
    
    for tensor_name, results in all_results.items():
        best_name = results['best_arm']
        best = results['best_arm_stats']
        baseline_stats = results['multi_slice_stats'].get('scale=none|in=none|out=none', {})
        print(f"\n  {tensor_name}:")
        print(f"    Best: {best_name}")
        print(f"    HW_red = {best['hw_reduction_mean_pct']:.1f}% (range: {best['hw_reduction_min_pct']:.1f}%-{best['hw_reduction_max_pct']:.1f}%)")
        print(f"    Baseline HW_err = {baseline_stats.get('hw_error_mean', 0):.2e} ± {baseline_stats.get('hw_error_std', 0):.2e}")
    
    # ─── Factorial decomposition ───
    print(f"\n{'=' * 80}")
    print("FACTORIAL DECOMPOSITION: Effect of each factor (averaged across slices)")
    print(f"{'=' * 80}")
    
    factorial_factors = {'scaling': ['none', 'biip'],
                         'input_rotation': ['none', 'hadamard', 'random_orth'],
                         'output_rotation': ['none', 'hadamard', 'random_orth']}
    
    for tensor_name, results in all_results.items():
        # Use slice 0 arms for factorial analysis
        arms = {}
        for k, v in results['slices'][0]['arms'].items():
            cfg = v.get('config', {})
            if (cfg.get('scaling') in factorial_factors['scaling'] and
                cfg.get('input_rotation') in factorial_factors['input_rotation'] and
                cfg.get('output_rotation') in factorial_factors['output_rotation']):
                arms[k] = v
        
        print(f"\n  {tensor_name} ({len(arms)} clean factorial arms, slice 0):")
        
        all_hw = [res['hw_error'] for res in arms.values()]
        grand_mean = np.mean(all_hw)
        print(f"    Grand mean OC-proxy err = {grand_mean:.2e}")
        
        for factor in ['scaling', 'input_rotation', 'output_rotation']:
            levels = {}
            for name, res in arms.items():
                level = res['config'].get(factor, 'none')
                if level not in levels:
                    levels[level] = []
                levels[level].append(res['hw_error'])
            
            print(f"    Marginal effect of {factor}:")
            for level in factorial_factors[factor]:
                if level in levels:
                    vals = levels[level]
                    effect = np.mean(vals) - grand_mean
                    print(f"      {level:<20}: {effect:+.2e} (avg={np.mean(vals):.2e}, n={len(vals)})")
    
    # ─── Per-tile quantizer tautology check ───
    print(f"\n{'=' * 80}")
    print("PER-TILE vs PER-COLUMN QUANTIZER COMPARISON (tautology check)")
    print(f"{'=' * 80}")
    print("If BiIP scaling is tautological with per-column quant, per-tile should show")
    print("LESS scaling benefit. Hadamard rotation should benefit BOTH quantizers.")
    
    for tensor_name, results in all_results.items():
        print(f"\n  {tensor_name}:")
        # Compare per-col vs per-tile for key arms
        key_arms = ['scale=none|in=none|out=none', 
                    'scale=biip|in=none|out=none',
                    'scale=none|in=hadamard|out=none',
                    'scale=biip|in=hadamard|out=none',
                    'scale=biip|in=hadamard|out=hadamard']
        
        print(f"    {'Arm':<45} {'per_col HW':>12} {'per_tile16 HW':>14} {'tile/col ratio':>14}")
        print(f"    {'─'*45} {'─'*12} {'─'*14} {'─'*14}")
        
        for arm in key_arms:
            if arm in results['slices'][0]['arms']:
                res = results['slices'][0]['arms'][arm]
                pc_hw = res['hw_error']
                pt_hw = res.get('per_tile16', {}).get('hw_error', float('nan'))
                ratio = pt_hw / pc_hw if pc_hw > 0 else float('nan')
                print(f"    {arm:<45} {pc_hw:.2e} {pt_hw:.2e} {ratio:.2f}")
        
        # Compute BiIP scaling benefit for each quantizer
        arms_s0 = results['slices'][0]['arms']
        baseline_pc = arms_s0.get('scale=none|in=none|out=none', {}).get('hw_error', 1)
        biip_only_pc = arms_s0.get('scale=biip|in=none|out=none', {}).get('hw_error', 1)
        had_only_pc = arms_s0.get('scale=none|in=hadamard|out=none', {}).get('hw_error', 1)
        biip_had_pc = arms_s0.get('scale=biip|in=hadamard|out=hadamard', {}).get('hw_error', 1)
        
        baseline_pt = arms_s0.get('scale=none|in=none|out=none', {}).get('per_tile16', {}).get('hw_error', 1)
        biip_only_pt = arms_s0.get('scale=biip|in=none|out=none', {}).get('per_tile16', {}).get('hw_error', 1)
        had_only_pt = arms_s0.get('scale=none|in=hadamard|out=none', {}).get('per_tile16', {}).get('hw_error', 1)
        biip_had_pt = arms_s0.get('scale=biip|in=hadamard|out=hadamard', {}).get('per_tile16', {}).get('hw_error', 1)
        
        print(f"\n    Reduction vs baseline (slice 0):")
        print(f"      {'Component':<25} {'per_col':>10} {'per_tile16':>12}")
        print(f"      {'─'*25} {'─'*10} {'─'*12}")
        print(f"      {'BiIP scaling only':<25} {(1-biip_only_pc/baseline_pc)*100:>9.1f}% {(1-biip_only_pt/baseline_pt)*100:>11.1f}%")
        print(f"      {'Hadamard only':<25} {(1-had_only_pc/baseline_pc)*100:>9.1f}% {(1-had_only_pt/baseline_pt)*100:>11.1f}%")
        print(f"      {'BiIP+Hadamard (full)':<25} {(1-biip_had_pc/baseline_pc)*100:>9.1f}% {(1-biip_had_pt/baseline_pt)*100:>11.1f}%")
        
        if (1-biip_only_pc/baseline_pc)*100 > 5 and abs((1-biip_only_pt/baseline_pt)*100 - (1-biip_only_pc/baseline_pc)*100) < 2:
            print(f"      → BiIP scaling benefit is SIMILAR across quantizers (not a per-column tautology)")
        elif (1-biip_only_pc/baseline_pc)*100 > 5 and (1-biip_only_pt/baseline_pt)*100 < (1-biip_only_pc/baseline_pc)*100 - 5:
            print(f"      → BiIP scaling benefit is REDUCED with per-tile (partially tautological with per-column)")
        else:
            print(f"      → BiIP scaling benefit is SMALL or INCONCLUSIVE")
        
    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    
    return all_results


# ─── Mathematical proof (printed) ───

def print_proof():
    """Print the mathematical proof of rotation invariance."""
    proof = """
══════════════════════════════════════════════════════════════════════
THEOREM: Orthogonal rotations preserve the output-covariance-weighted
quantization error exactly (KronQ Theorem 1, clean-room proof).

NOTE: In this experiment, H_G is an output-covariance proxy (Y^T Y / N),
NOT the true gradient covariance (Fisher) H_G = E[g g^T]. The true H_G
requires a backward pass. The invariance proof holds for ANY symmetric
positive-definite H_G and H_X, including the true Fisher.

Setup:
  W ∈ R^{d_out × d_in}, quantization error E = W - Ŵ
  H_X = input covariance (d_in × d_in), H_G = output covariance (d_out × d_out)
  Objective: L = tr(H_G · E · H_X · E^T)

Transform:
  Diagonal: S_G (d_out × d_out), S_X (d_in × d_in)
  Orthogonal: U (d_out × d_out), V (d_in × d_in)
  W' = U · S_G · W · S_X · V^T
  Quantize: Ŵ' = Q(W')
  Reconstruct: Ŵ = S_G^{-1} · U^T · Ŵ' · V · S_X^{-1}
  Error: E = W - Ŵ = S_G^{-1} · U^T · (W' - Ŵ') · V · S_X^{-1}
          E = S_G^{-1} · U^T · E' · V · S_X^{-1}

Proof:
  L = tr(H_G · E · H_X · E^T)
    = tr(H_G · S_G^{-1} U^T E' V S_X^{-1} · H_X · S_X^{-1} V^T E'^T U S_G^{-1})

  By cyclic property of trace: tr(ABC) = tr(CAB)
    = tr(S_G^{-1} H_G S_G^{-1} · U^T E' V · S_X^{-1} H_X S_X^{-1} · V^T E'^T U)

  Define H_G' = S_G^{-1} H_G S_G^{-1}, H_X' = S_X^{-1} H_X S_X^{-1}:
    = tr(H_G' · U^T E' V · H_X' · V^T E'^T U)

  By cyclic property:
    = tr(U · H_G' · U^T · E' · V · H_X' · V^T · E'^T)

  Define H_G'' = U H_G' U^T, H_X'' = V H_X' V^T:
    = tr(H_G'' · E' · H_X'' · E'^T)

  This is exactly the output-covariance-weighted error in the transformed
  space with transformed covariances H_G'' and H_X''.

  ∴ L = tr(H_G · E · H_X · E^T) = tr(H_G'' · E' · H_X'' · E'^T)

  The objective is INVARIANT under the transform. The benefit comes from
  E' being smaller: rotations make W' more incoherent (i.i.d.-like),
  which reduces quantization error E' = W' - Q(W').

Sidecar cost:
  - Diagonal scales S_G, S_X: (d_out + d_in) × 4 bytes (float32)
  - Hadamard signs: (d_out + d_in) / 8 bytes (1 bit per dim)
  - Butterfly factors: O(sqrt(d)^2) = O(d) bytes
  - Random orthogonal: 4 bytes (seed only, regenerated at inference)
  - At d=128: BiIP scales = 1024B ≈ 0.5 bits/elem, Hadamard signs = 32B ≈ 0.002 bits/elem
══════════════════════════════════════════════════════════════════════
"""
    print(proof)


if __name__ == '__main__':
    print_proof()
    results = run_experiment()
