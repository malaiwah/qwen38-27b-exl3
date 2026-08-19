#!/usr/bin/env python3
"""
Pilot: test whether ParoQuant-style learned Givens rotations reduce EXL3 trellis
quantization error versus EXL3's existing fixed Hadamard transform.

Three arms are compared on a single weight matrix:

  1. Hadamard baseline -- EXL3's regularize() path: blockwise H_128 on both
     dimensions, per-channel RMS scaling, random +/-1 sign flips, and a golden-
     section global-scale search.  This replicates what EXL3 actually does at
     quantisation time (minus the Hessian-dependent LDL compensation, which is
     irrelevant for a standalone weight-reconstruction MSE measurement).

  2. Learned rotation -- K=8 sequential Givens rotation stages on both the input
     and output dimensions, plus learnable per-channel scaling, optimised with
     AdamW via a straight-through estimator (STE) to minimise trellis
     reconstruction MSE.

  3. Identity control -- no pre-transform at all, only the global-scale search.
     This proves the harness can detect a difference: if the Hadamard and
     learned-rotation arms do not both beat identity, the harness is broken.

Decision rule (printed explicitly at the end):
  GO        if learned rotation reduces MSE by >5%  versus Hadamard
  NO-GO     if reduction is <2%
  AMBIGUOUS if between 2% and 5%

What this pilot does NOT establish
----------------------------------
  * A single matrix at one layer (layer 0, mlp.gate_proj) is not evidence about
    the whole model.  Gate / up / down / q / k / v / o projections have different
    weight distributions; a 64-layer x 7-matrix study is needed before any
    model-wide conclusion.
  * Trellis reconstruction MSE is a proxy for, not a measurement of, KLD.
    MSE reduction does not guarantee KLD reduction: the trellis codebook,
    per-channel scales, and global scale interact non-linearly with the final
    output distribution.  A full v5 KLD measurement on a requantised model is
    the definitive test.
  * The STE optimisation uses a biased gradient estimate (the quantisation
    output is treated as constant w.r.t. the rotation parameters).  This is a
    standard approximation in quantisation-aware training but means the learned
    rotation is locally, not globally, optimal.

Container command line
----------------------
Run inside the project's container (the pinned image that has exllamav3 at
/opt/exllamav3-python/exllamav3 and the CUDA extension built):

  podman run --rm --device nvidia.com/gpu=all --ipc=host --network host \\
      -v /home/mbelleau/qwen38-27b-exl3/tools/rotation-pilot.py:/tmp/pilot.py:ro \\
      -v /home/mbelleau/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro \\
      docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34 \\
      python3 /tmp/pilot.py \\
          --layer 0 --matrix mlp.gate_proj \\
          --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \\
          --json-out /tmp/rotation-pilot.json

The --model-path must point to a directory containing the ORIGINAL (non-quantised)
Qwen3.8-27B safetensors files.  Download with:

  huggingface-cli download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

Requires: CUDA GPU, exllamav3 with exllamav3_ext built, safetensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Constants -- no heavy imports needed at module level.
# ---------------------------------------------------------------------------

HAD_BLOCK = 128          # EXL3 Hadamard block size (quantize.py:14)
CODEBOOK_SCALE = 1.24371088  # EXL3 trellis codebook scale (quantize.py:15)
TILE = 16                # EXL3 tile size (16x16)


# ---------------------------------------------------------------------------
# Argument parser -- works on a GPU-less shell with no torch installed.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pilot: test learned Givens rotations vs fixed Hadamard for "
            "EXL3 trellis quantization error reduction on a single weight "
            "matrix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--layer", type=int, default=0,
                   help="Model layer index (default: 0)")
    p.add_argument("--matrix", type=str, default="mlp.gate_proj",
                   help="Weight matrix name within the layer (default: mlp.gate_proj)")
    p.add_argument("--samples", type=int, default=128,
                   help="Tile samples per optimisation epoch (default: 128)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Optimisation epochs for learned rotation (default: 10)")
    p.add_argument("--k-rotations", type=int, default=8,
                   help="K: number of Givens rotation stages per dimension (default: 8)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate (default: 1e-3)")
    p.add_argument("--json-out", type=str, default=None,
                   help="Path to write JSON results file")
    p.add_argument("--exllama-path", type=str,
                   default="/opt/exllamav3-python/exllamav3",
                   help="Path to the exllamav3 package directory "
                        "(default: /opt/exllamav3-python/exllamav3)")
    p.add_argument("--model-path", type=str, default=None,
                   help="Directory containing original model safetensors "
                        "(default: auto-resolve from HF cache)")
    p.add_argument("--model-repo", type=str, default="Qwen/Qwen3.8-27B",
                   help="HuggingFace repo for the original model (default: Qwen/Qwen3.8-27B)")
    p.add_argument("--quant-bits", type=int, default=5,
                   help="Trellis quantization bitrate K (default: 5, matching gate_proj K5)")
    p.add_argument("--use-mcg", action="store_true", default=True,
                   help="Use mcg codebook multiplier (default: True, matching K5_mcg)")
    p.add_argument("--no-mcg", dest="use_mcg", action="store_false",
                   help="Disable mcg codebook multiplier")
    p.add_argument("--in-features", type=int, default=5120,
                   help="Input features of the weight matrix (default: 5120)")
    p.add_argument("--out-features", type=int, default=17408,
                   help="Output features of the weight matrix (default: 17408)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="CUDA device (default: cuda:0)")
    return p


# ---------------------------------------------------------------------------
# Lazy imports -- called only when the pilot actually runs.
# ---------------------------------------------------------------------------

def import_exllamav3(exllama_path: str):
    """Insert exllama_path's parent on sys.path and import exllamav3 modules.

    exllama_path is the path to the exllamav3 *package* directory (e.g.
    /opt/exllamav3-python/exllamav3).  We add its parent to sys.path so that
    ``import exllamav3`` resolves correctly.
    """
    parent = os.path.dirname(os.path.normpath(exllama_path))
    for p in (parent, os.path.normpath(exllama_path)):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # exllamav3/__init__ pulls modules/attn.py which imports flash_attn at
    # module scope. The pilot only uses quantize_tiles() and never dispatches
    # attention, and the serving image ships without flash_attn (vLLM uses its
    # own backends). Stub it so the package import succeeds; any actual USE of
    # the stub raises loudly instead of silently faking results.
    import types
    if "flash_attn" not in sys.modules:
        _stub = types.ModuleType("flash_attn")
        def _unavailable(*_a, **_k):
            raise RuntimeError(
                "flash_attn stubbed for the rotation pilot; attention "
                "dispatch must not be reached from quantize_tiles()"
            )
        _stub.flash_attn_func = _unavailable
        _stub.flash_attn_with_kvcache = _unavailable
        _stub.flash_attn_varlen_func = _unavailable
        sys.modules["flash_attn"] = _stub

    try:
        from exllamav3.ext import exllamav3_ext as ext  # noqa: F401
        from exllamav3.modules.quant.exl3_lib import quantize as qmod
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import exllamav3 from {exllama_path} "
            f"(parent added to sys.path: {parent}).  "
            f"Ensure the script runs inside the container or pass "
            f"--exllama-path pointing to the exllamav3 package directory.  "
            f"Original error: {exc}"
        ) from exc
    return qmod


def assert_cuda(device_str: str):
    """Verify CUDA is available.  Raises RuntimeError if not."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  This pilot requires a GPU for trellis "
            "quantization (exllamav3_ext.quantize_tiles is a CUDA kernel).  "
            "No CPU fallback exists -- the pilot cannot proceed without a GPU."
        )
    try:
        torch.cuda.set_device(device_str)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot use CUDA device {device_str}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _candidate_keys(layer: int, matrix: str) -> list[str]:
    """Build candidate safetensors keys for the weight tensor."""
    return [
        f"model.language_model.layers.{layer}.{matrix}.weight",
        f"model.layers.{layer}.{matrix}.weight",
        f"language_model.layers.{layer}.{matrix}.weight",
        f"layers.{layer}.{matrix}.weight",
    ]


def _resolve_model_path(model_path: str | None, model_repo: str) -> str:
    """Resolve model_path from the HF cache if not explicitly given."""
    if model_path is not None:
        return model_path
    cache_root = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface/hub"),
    )
    repo_dir = f"models--{model_repo.replace('/', '--')}"
    snapshots_dir = os.path.join(cache_root, repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(
            f"Model snapshots directory not found: {snapshots_dir}.  "
            f"Set --model-path to a directory containing the original model "
            f"safetensors files, or download the model with:  "
            f"huggingface-cli download {model_repo}"
        )
    revisions = sorted(os.listdir(snapshots_dir))
    if not revisions:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    return os.path.join(snapshots_dir, revisions[0])


def load_original_weight(
    model_path: str | None,
    model_repo: str,
    layer: int,
    matrix: str,
    in_features: int,
    out_features: int,
    device,
) -> "torch.Tensor":
    """Load the original (non-quantised) weight from safetensors.

    Returns a float32 tensor of shape (in_features, out_features) on *device*.
    HF stores Linear weights as (out_features, in_features); we transpose if
    necessary.
    """
    import torch
    from safetensors import safe_open

    resolved = _resolve_model_path(model_path, model_repo)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"Model directory not found: {resolved}.  "
            f"Pass --model-path pointing to a directory with .safetensors files."
        )

    st_files = sorted(
        os.path.join(resolved, f)
        for f in os.listdir(resolved)
        if f.endswith(".safetensors")
    )
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {resolved}")

    candidates = _candidate_keys(layer, matrix)
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            available = list(f.keys())
            for key in candidates:
                if key in available:
                    w = f.get_tensor(key)
                    # Ensure (in_features, out_features) orientation.
                    if w.shape == (out_features, in_features):
                        w = w.t().contiguous()
                    elif w.shape != (in_features, out_features):
                        raise ValueError(
                            f"Weight {key} has unexpected shape {tuple(w.shape)}; "
                            f"expected ({in_features}, {out_features}) or "
                            f"({out_features}, {in_features})."
                        )
                    return w.to(device=device, dtype=torch.float32)

    raise KeyError(
        f"Weight key not found in {resolved}.  Tried: {candidates}.  "
        f"First 20 available keys: {available[:20]}"
    )


# ---------------------------------------------------------------------------
# Tile operations
# ---------------------------------------------------------------------------

def extract_all_tiles(weight, perm):
    """Extract all 16x16 tiles in EXL3's tensor-core-permuted order.

    Mirrors the tile extraction in fallback_quant (quantize.py:559-563):
        rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
    but vectorised over all row-blocks.

    Returns (total_tiles, 256) float32 tensor.
    """
    import torch
    K, N = weight.shape
    tiles_k = K // TILE
    tiles_n = N // TILE
    # (K, N) -> (tiles_k, 16, tiles_n, 16) -> (tiles_k, tiles_n, 16, 16) -> (total, 256)
    tiles = weight.reshape(tiles_k, TILE, tiles_n, TILE)
    tiles = tiles.permute(0, 2, 1, 3).reshape(tiles_k * tiles_n, 256)
    tiles = tiles[:, perm]
    return tiles.contiguous()


def sample_tiles(weight, n_samples, perm, generator):
    """Sample n_samples random 16x16 tiles (differentiable).

    Returns (n_samples, 256) float32 tensor connected to *weight* in the
    autograd graph.
    """
    import torch
    K, N = weight.shape
    tiles_k = K // TILE
    tiles_n = N // TILE

    tk = torch.randint(tiles_k, (n_samples,), device=weight.device,
                       generator=generator)
    tn = torch.randint(tiles_n, (n_samples,), device=weight.device,
                       generator=generator)

    row_starts = tk * TILE        # (n_samples,)
    col_starts = tn * TILE        # (n_samples,)
    offsets = torch.arange(TILE, device=weight.device)
    row_idx = row_starts.unsqueeze(1) + offsets.unsqueeze(0)   # (n_samples, 16)
    col_idx = col_starts.unsqueeze(1) + offsets.unsqueeze(0)   # (n_samples, 16)

    # Advanced indexing -> (n_samples, 16, 16), differentiable.
    tiles = weight[
        row_idx.unsqueeze(2).expand(-1, TILE, TILE),
        col_idx.unsqueeze(1).expand(-1, TILE, TILE),
    ]
    tiles = tiles.reshape(n_samples, 256)
    tiles = tiles[:, perm]
    return tiles.contiguous()


# ---------------------------------------------------------------------------
# Givens rotation (differentiable, out-of-place via scatter)
# ---------------------------------------------------------------------------

def init_pair_indices(dim, k_rotations, group_size, device, seed):
    """Generate fixed random disjoint pair indices for K rotation stages.

    Returns (k_rotations, dim // group_size, group_size // 2, 2) int64 tensor
    of *local* indices (0 .. group_size-1 within each group).
    """
    import torch
    num_groups = dim // group_size
    pairs_per_group = group_size // 2
    gen = torch.Generator(device="cpu").manual_seed(seed)

    stages = []
    for _ in range(k_rotations):
        groups = []
        for _ in range(num_groups):
            perm = torch.randperm(group_size, generator=gen)
            groups.append(perm.reshape(pairs_per_group, 2))
        stages.append(torch.stack(groups))
    return torch.stack(stages).to(device).long()


def _apply_givens(w, angles, pair_indices, group_size, dim_axis):
    """Apply K Givens rotation stages along *dim_axis* of *w*.

    w: 2-D tensor.
    angles: (K, dim // 2) learnable.
    pair_indices: (K, dim // group_size, group_size // 2, 2) fixed local indices.

    Uses scatter (out-of-place) so the operation is fully differentiable.
    """
    import torch
    D = w.shape[dim_axis]
    other = w.shape[1 - dim_axis]
    num_groups = D // group_size
    pairs_per_group = group_size // 2
    K_stages = angles.shape[0]

    angles_r = angles.view(K_stages, num_groups, pairs_per_group)
    offsets = torch.arange(num_groups, device=w.device).view(
        1, num_groups, 1, 1
    ) * group_size
    global_pairs = pair_indices + offsets   # (K, num_groups, pairs_per_group, 2)

    result = w
    for k in range(K_stages):
        cos_t = torch.cos(angles_r[k]).reshape(-1)   # (num_groups * pairs_per_group,)
        sin_t = torch.sin(angles_r[k]).reshape(-1)
        i_idx = global_pairs[k, :, :, 0].reshape(-1)  # (num_groups * pairs_per_group,)
        j_idx = global_pairs[k, :, :, 1].reshape(-1)

        if dim_axis == 0:
            wi = result[i_idx]          # (P, other)
            wj = result[j_idx]
            cos_exp = cos_t.unsqueeze(1)
            sin_exp = sin_t.unsqueeze(1)
            new_wi = cos_exp * wi - sin_exp * wj
            new_wj = sin_exp * wi + cos_exp * wj
            i_exp = i_idx.unsqueeze(1).expand(-1, other)
            j_exp = j_idx.unsqueeze(1).expand(-1, other)
            result = result.scatter(0, i_exp, new_wi)
            result = result.scatter(0, j_exp, new_wj)
        else:
            wi = result[:, i_idx]       # (other, P)
            wj = result[:, j_idx]
            cos_exp = cos_t.unsqueeze(0)
            sin_exp = sin_t.unsqueeze(0)
            new_wi = cos_exp * wi - sin_exp * wj
            new_wj = sin_exp * wi + cos_exp * wj
            i_exp = i_idx.unsqueeze(0).expand(other, -1)
            j_exp = j_idx.unsqueeze(0).expand(other, -1)
            result = result.scatter(1, i_exp, new_wi)
            result = result.scatter(1, j_exp, new_wj)

    return result


def apply_givens_left(w, angles, pair_indices, group_size=128):
    """Apply K Givens rotations to dimension 0 (rows / input dimension)."""
    return _apply_givens(w, angles, pair_indices, group_size, dim_axis=0)


def apply_givens_right(w, angles, pair_indices, group_size=128):
    """Apply K Givens rotations to dimension 1 (columns / output dimension)."""
    return _apply_givens(w, angles, pair_indices, group_size, dim_axis=1)


# ---------------------------------------------------------------------------
# Global scale (g_scale) search -- golden section, matching EXL3's g_scale_gss
# ---------------------------------------------------------------------------

def find_optimal_gscale(weight, qmod, quant_args, perm, device):
    """Golden-section search for the global scale that minimises tile MSE.

    Mirrors g_scale_gss (quantize.py:704-781): samples tiles along a wrapped
    diagonal, then searches the scale range [0.1, 1.9].
    """
    import torch
    K, N = weight.shape
    tiles_k = K // TILE
    tiles_n = N // TILE
    width = 3

    tiles = []
    for i in range(max(tiles_k, tiles_n)):
        for w in range(width):
            r = (i % tiles_k) * TILE
            c = ((i + w) % tiles_n) * TILE
            tile = weight[r:r + TILE, c:c + TILE].reshape(256)
            tiles.append(tile)
    tiles = torch.stack(tiles)[:, perm].contiguous()

    def test_scale(scale):
        qw, _ = qmod.quantize_tiles(tiles * scale, quant_args)
        return ((qw / scale - tiles) ** 2).mean().item()

    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    a, b = 0.1, 1.9
    tol = 0.01
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = test_scale(x1)
    f2 = test_scale(x2)
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = test_scale(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = test_scale(x2)
    return (a + b) / 2


# ---------------------------------------------------------------------------
# Full MSE computation
# ---------------------------------------------------------------------------

def compute_full_mse(weight, qmod, quant_args, perm, g_scale=1.0):
    """Quantize ALL tiles and compute reconstruction MSE.

    MSE = mean over all tiles of ||Q(tile * g) / g - tile||^2
    (matching g_scale_gss's error definition, not the g_scale-adjusted MSE
    from fallback_quant which is g^2 times larger).
    """
    import torch
    tiles = extract_all_tiles(weight, perm)
    with torch.no_grad():
        qw, _ = qmod.quantize_tiles(tiles * g_scale, quant_args)
    mse = ((qw / g_scale - tiles) ** 2).mean().item()
    return mse


# ---------------------------------------------------------------------------
# Arm 1: Hadamard baseline (EXL3's regularize path)
# ---------------------------------------------------------------------------

def run_hadamard_baseline(weight, qmod, quant_args, perm, device, seed):
    """Replicate EXL3's regularize() without the Hessian-dependent parts.

    Steps (matching quantize.py:889-929):
      1. Random +/-1 sign vectors su, sv (seeded).
      2. Per-channel RMS scaling (output dim, then input dim).
      3. Blockwise Hadamard on both dimensions.
      4. Golden-section g_scale search.
      5. Full MSE.
    """
    import torch
    K, N = weight.shape
    w = weight.clone().to(torch.float32)

    torch.manual_seed(seed)
    sv = (torch.randn(N, device=device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
    su = (torch.randn(K, device=device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)

    # Output channel RMS (dim=0 -> one scale per column).
    out_rms = qmod.block_rms(w, dim=0, keepdim=True)
    mean_rms = out_rms.mean().item()
    if mean_rms > 1e-30:
        out_rms = out_rms / mean_rms
    else:
        out_rms = torch.ones_like(out_rms)
    sv = (sv * out_rms + 1e-10).float()
    w = w / sv

    # Right Hadamard (columns).
    qmod.blockwise_preapply_had_r_(w, HAD_BLOCK)

    # Input channel RMS (dim=1 -> one scale per row).
    in_rms = qmod.block_rms(w, dim=1, keepdim=True)
    in_rms[in_rms.abs() < 1e-30] = 0.1
    su = (su * in_rms / (-CODEBOOK_SCALE) + 1e-10).float()
    w = w / su

    # Left Hadamard (rows).
    qmod.blockwise_preapply_had_l_(w, HAD_BLOCK)

    # Global scale search + MSE.
    g_scale = find_optimal_gscale(w, qmod, quant_args, perm, device)
    mse = compute_full_mse(w, qmod, quant_args, perm, g_scale)
    return {"mse": mse, "g_scale": g_scale}


# ---------------------------------------------------------------------------
# Arm 2: Identity control (no transform)
# ---------------------------------------------------------------------------

def run_identity_control(weight, qmod, quant_args, perm, device):
    """No pre-transform; only g_scale search and MSE.

    If the Hadamard and learned-rotation arms do not both beat this, the
    harness is broken.
    """
    g_scale = find_optimal_gscale(weight, qmod, quant_args, perm, device)
    mse = compute_full_mse(weight, qmod, quant_args, perm, g_scale)
    return {"mse": mse, "g_scale": g_scale}


# ---------------------------------------------------------------------------
# Arm 3: Learned Givens rotations (STE + AdamW optimisation)
# ---------------------------------------------------------------------------

def run_learned_rotation(weight, qmod, quant_args, perm, device,
                          k_rotations, epochs, lr, n_samples, seed):
    """K=8 Givens rotations + per-channel scaling on both dimensions.

    Optimised with AdamW via straight-through estimator: the quantization
    output is detached, so the gradient of ||Q(W') - W'||^2 flows through
    W' to the rotation angles and channel scales.

    After optimisation, a g_scale search and full MSE are computed.
    """
    import torch
    K, N = weight.shape

    # Fixed pair indices (not learned).
    pairs_k = init_pair_indices(K, k_rotations, HAD_BLOCK, device, seed)
    pairs_n = init_pair_indices(N, k_rotations, HAD_BLOCK, device, seed + 1)

    # Learnable parameters.
    torch.manual_seed(seed)
    angles_k = torch.zeros(k_rotations, K // 2, device=device, requires_grad=True)
    angles_n = torch.zeros(k_rotations, N // 2, device=device, requires_grad=True)

    # Initialise channel scales from inverse normalised RMS (analogous to 1/su, 1/sv
    # in regularize, but without random signs -- the rotation handles incoherence).
    with torch.no_grad():
        out_rms = qmod.block_rms(weight, dim=0, keepdim=True)
        out_rms = out_rms / (out_rms.mean() + 1e-10)
        alpha_n = (1.0 / (out_rms + 1e-10)).squeeze(0).clone().requires_grad_(True)

        in_rms = qmod.block_rms(weight, dim=1, keepdim=True)
        in_rms = in_rms / (in_rms.mean() + 1e-10)
        alpha_k = (1.0 / (in_rms + 1e-10)).squeeze(1).clone().requires_grad_(True)

    params = [angles_k, angles_n, alpha_k, alpha_n]
    optimizer = torch.optim.AdamW(params, lr=lr)

    gen = torch.Generator(device=device).manual_seed(seed + 100)
    w_const = weight.detach()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward: W' = R_k @ diag(alpha_k) @ R_n @ diag(alpha_n) @ W
        w_t = w_const * alpha_n.unsqueeze(0)       # column scaling
        w_t = apply_givens_right(w_t, angles_n, pairs_n)
        w_t = w_t * alpha_k.unsqueeze(1)            # row scaling
        w_t = apply_givens_left(w_t, angles_k, pairs_k)

        # Sample tiles (differentiable).
        sampled = sample_tiles(w_t, n_samples, perm, gen)

        # Quantize (detached -- CUDA kernel, no autograd).
        with torch.no_grad():
            quant, _ = qmod.quantize_tiles(sampled, quant_args)

        # STE loss: gradient flows through *sampled* to angles/scales.
        loss = ((quant.detach() - sampled) ** 2).mean()
        loss.backward()
        optimizer.step()

        print(f"  epoch {epoch + 1:2d}/{epochs}  loss = {loss.item():.8f}")

    # Final evaluation with g_scale search.
    with torch.no_grad():
        w_t = w_const * alpha_n.unsqueeze(0)
        w_t = apply_givens_right(w_t, angles_n, pairs_n)
        w_t = w_t * alpha_k.unsqueeze(1)
        w_t = apply_givens_left(w_t, angles_k, pairs_k)
        w_final = w_t.detach()

    g_scale = find_optimal_gscale(w_final, qmod, quant_args, perm, device)
    mse = compute_full_mse(w_final, qmod, quant_args, perm, g_scale)
    return {"mse": mse, "g_scale": g_scale}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    # Validate dimensions.
    if args.in_features % HAD_BLOCK != 0 or args.out_features % HAD_BLOCK != 0:
        raise ValueError(
            f"Both in_features ({args.in_features}) and out_features "
            f"({args.out_features}) must be divisible by {HAD_BLOCK} "
            f"(the Hadamard block size)."
        )
    if args.in_features % TILE != 0 or args.out_features % TILE != 0:
        raise ValueError(
            f"Both dimensions must be divisible by {TILE} (tile size)."
        )

    # Heavy imports.
    import torch
    qmod = import_exllamav3(args.exllama_path)
    assert_cuda(args.device)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)

    # Build quant_args for quantize_tiles.
    quant_args = {"K": args.quant_bits}
    if args.use_mcg:
        quant_args["mcg"] = True

    # Permutation indices (device-specific, lru-cached in quantize.py).
    perm = qmod.tensor_core_perm(device)
    inv_perm = qmod.tensor_core_perm_i(device)

    print("=" * 72)
    print("Rotation Pilot: learned Givens vs fixed Hadamard for EXL3 trellis")
    print("=" * 72)
    print(f"  layer       : {args.layer}")
    print(f"  matrix      : {args.matrix}")
    print(f"  shape       : ({args.in_features}, {args.out_features})")
    print(f"  quant_bits  : K={args.quant_bits}  mcg={args.use_mcg}")
    print(f"  k_rotations : {args.k_rotations}")
    print(f"  epochs      : {args.epochs}")
    print(f"  samples     : {args.samples}")
    print(f"  lr          : {args.lr}")
    print(f"  device      : {device}")
    print()

    # Load weight.
    print("Loading original weight ...")
    weight = load_original_weight(
        args.model_path, args.model_repo, args.layer, args.matrix,
        args.in_features, args.out_features, device,
    )
    print(f"  loaded: shape={tuple(weight.shape)}, dtype={weight.dtype}")
    print()

    # Arm 1: Hadamard baseline.
    print("Arm 1: Hadamard baseline (EXL3 regularize path) ...")
    res_had = run_hadamard_baseline(weight, qmod, quant_args, perm, device, args.seed)
    print(f"  MSE = {res_had['mse']:.10f}  (g_scale = {res_had['g_scale']:.6f})")
    print()

    # Arm 2: Learned rotation.
    print(f"Arm 2: Learned Givens rotations (K={args.k_rotations}, "
          f"epochs={args.epochs}, samples={args.samples}) ...")
    res_rot = run_learned_rotation(
        weight, qmod, quant_args, perm, device,
        args.k_rotations, args.epochs, args.lr, args.samples, args.seed,
    )
    print(f"  MSE = {res_rot['mse']:.10f}  (g_scale = {res_rot['g_scale']:.6f})")
    print()

    # Arm 3: Identity control.
    print("Arm 3: Identity control (no transform) ...")
    res_id = run_identity_control(weight, qmod, quant_args, perm, device)
    print(f"  MSE = {res_id['mse']:.10f}  (g_scale = {res_id['g_scale']:.6f})")
    print()

    # Decision.
    mse_had = res_had["mse"]
    mse_rot = res_rot["mse"]
    mse_id = res_id["mse"]
    improvement = (mse_had - mse_rot) / mse_had if mse_had > 0 else 0.0
    improvement_pct = improvement * 100.0

    if improvement > 0.05:
        verdict = "GO"
    elif improvement < 0.02:
        verdict = "NO-GO"
    else:
        verdict = "AMBIGUOUS"

    had_vs_id = (mse_id - mse_had) / mse_id if mse_id > 0 else 0.0

    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  Hadamard baseline MSE : {mse_had:.10f}")
    print(f"  Learned rotation  MSE : {mse_rot:.10f}")
    print(f"  Identity control  MSE : {mse_id:.10f}")
    print(f"  Hadamard vs identity   : {had_vs_id * 100:.2f}% reduction "
          f"(harness sanity check)")
    print(f"  Learned vs Hadamard    : {improvement_pct:.2f}% reduction")
    print()
    print("DECISION RULE:")
    print(f"  GO       if learned rotation reduces MSE by >5%  vs Hadamard")
    print(f"  NO-GO    if reduction is <2%")
    print(f"  AMBIGUOUS if between 2% and 5%")
    print()
    print(f"  >>> VERDICT: {verdict}  "
          f"(improvement = {improvement_pct:.2f}%)")
    print("=" * 72)

    # JSON output.
    results = {
        "config": {
            "layer": args.layer,
            "matrix": args.matrix,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "quant_bits": args.quant_bits,
            "use_mcg": args.use_mcg,
            "k_rotations": args.k_rotations,
            "epochs": args.epochs,
            "samples": args.samples,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
        },
        "hadamard": res_had,
        "learned_rotation": res_rot,
        "identity": res_id,
        "improvement_pct": improvement_pct,
        "hadamard_vs_identity_pct": had_vs_id * 100,
        "verdict": verdict,
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
