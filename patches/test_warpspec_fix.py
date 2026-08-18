#!/usr/bin/env python3
"""Test the warpspec GEMM fix: M tiling + trellis flattening.

Verifies that:
1. The warpspec kernel produces correct output for M > 64 (M tiling)
2. The trellis tensor is correctly flattened
3. Output matches the baseline exl3_gemm for real weights

Run inside the container:
    python3 test_warpspec_fix.py
"""

import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")

import sys
import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

EXT_DIR = "/opt/exllamav3-python/exllamav3/exllamav3_ext"
sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext

MODEL_DIR = "/models/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"

# Build the warpspec kernel
extra_include_paths = [
    EXT_DIR,
    os.path.join(EXT_DIR, "quant"),
    os.path.join(EXT_DIR, "util"),
    os.path.dirname(os.path.abspath(__file__)),
]

extra_cuda_cflags = [
    "-O3", "--use_fast_math", "-std=c++17",
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
]

print("Compiling warp-specialized kernel...")
module = load(
    name="exl3_gemm_warpspec",
    sources=[os.path.join(os.path.dirname(__file__), "exl3_gemm_warpspec.cu")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=extra_cuda_cflags,
    extra_include_paths=extra_include_paths,
    verbose=True,
)

def load_layer_weights(layer_name):
    """Load trellis, suh, svh, mcg, mul1 for a given layer."""
    for shard in ["model-00001-of-00003.safetensors",
                  "model-00002-of-00003.safetensors",
                  "model-00003-of-00003.safetensors"]:
        path = os.path.join(MODEL_DIR, shard)
        if not os.path.exists(path):
            continue
        f = safe_open(path, framework="pt")
        keys = list(f.keys())
        trellis_key = f"{layer_name}.trellis"
        if trellis_key in keys:
            trellis = f.get_tensor(trellis_key).cuda()
            suh = f.get_tensor(f"{layer_name}.suh").cuda() if f"{layer_name}.suh" in keys else None
            svh = f.get_tensor(f"{layer_name}.svh").cuda() if f"{layer_name}.svh" in keys else None
            mcg = f.get_tensor(f"{layer_name}.mcg") if f"{layer_name}.mcg" in keys else None
            mul1 = f.get_tensor(f"{layer_name}.mul1") if f"{layer_name}.mul1" in keys else None
            return trellis, suh, svh, mcg, mul1
    return None


def warpspec_gemm_tiled(xh, trellis_flat, M, K, N, bits, cb, device):
    """Run warpspec GEMM with M tiling (mirrors the Python fix)."""
    _WS_TILE_M = 64
    tiles_n = N // 128
    post_scale = torch.empty(0, dtype=torch.float16, device=device)

    if M <= _WS_TILE_M:
        locks = torch.zeros(tiles_n, dtype=torch.int32, device=device)
        C_raw = module.warpspec_gemm(
            xh, trellis_flat, M, K, N, bits, cb, locks, post_scale
        )
    else:
        C_raw = torch.empty(M, N, dtype=torch.float16, device=device)
        for m_start in range(0, M, _WS_TILE_M):
            m_end = min(m_start + _WS_TILE_M, M)
            m_chunk = m_end - m_start
            locks = torch.zeros(tiles_n, dtype=torch.int32, device=device)
            C_chunk = module.warpspec_gemm(
                xh[m_start:m_end], trellis_flat,
                m_chunk, K, N, bits, cb, locks, post_scale,
            )
            C_raw[m_start:m_end] = C_chunk
    return C_raw


def test_layer(layer_name, M_values=(1, 64, 128, 256, 512)):
    """Test warpspec vs baseline for one layer."""
    result = load_layer_weights(layer_name)
    if result is None:
        print(f"  Layer not found: {layer_name}")
        return False

    trellis, suh, svh, mcg, mul1 = result
    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    bits = trellis.shape[2] // 16
    cb = 1 if (mcg is not None) else (2 if (mul1 is not None) else 0)
    mcg_bool = mcg is not None
    mul1_bool = mul1 is not None

    print(f"\n  Layer: {layer_name}")
    print(f"  Trellis: {trellis.shape} {trellis.dtype}")
    print(f"  K={K}, N={N}, bits={bits}, cb={cb}, mcg={mcg_bool}, mul1={mul1_bool}")

    # Skip if cb=2 (mul1) — warpspec kernel doesn't support it yet
    if cb == 2:
        print(f"  SKIP: cb=2 (mul1 codebook) not supported by warpspec kernel")
        return True

    # Skip if N is not a multiple of 128
    if N % 128 != 0:
        print(f"  SKIP: N={N} not a multiple of 128")
        return True

    # Flatten trellis
    trellis_flat = trellis.contiguous().reshape(-1)
    print(f"  Trellis flat: {trellis_flat.shape} {trellis_flat.dtype}")

    all_pass = True
    for M in M_values:
        torch.manual_seed(42)
        A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1

        # --- Baseline: existing exl3_gemm ---
        C_ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had_ref = torch.empty_like(A)
        ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)

        # --- Warpspec with M tiling ---
        xh = torch.empty_like(A)
        ext.had_r_128(A, xh, suh, None, 1.0)
        C_raw = warpspec_gemm_tiled(xh, trellis_flat, M, K, N, bits, cb, "cuda")
        C_new = torch.empty_like(C_raw)
        ext.had_r_128(C_raw, C_new, None, svh, 1.0)

        # Compare
        max_diff = (C_ref.float() - C_new.float()).abs().max().item()
        mean_diff = (C_ref.float() - C_new.float()).abs().mean().item()
        ref_norm = C_ref.float().abs().max().item()
        bit_exact = torch.equal(C_ref, C_new)

        status = "EXACT" if bit_exact else f"max_diff={max_diff:.6f}"
        rel = max_diff / ref_norm if ref_norm > 0 else float('inf')
        passed = bit_exact or max_diff < 1e-2 or rel < 1e-3

        if not passed:
            all_pass = False

        print(f"    M={M:5d}: {status}  mean_diff={mean_diff:.6f}  "
              f"ref_norm={ref_norm:.4f}  {'PASS' if passed else 'FAIL'}")

        if not passed:
            print(f"    C_ref[0,:4]: {C_ref[0, :4].tolist()}")
            print(f"    C_new[0,:4]: {C_new[0, :4].tolist()}")
            if M > 64:
                print(f"    C_ref[64,:4]: {C_ref[64, :4].tolist()}")
                print(f"    C_new[64,:4]: {C_new[64, :4].tolist()}")

    return all_pass


def main():
    # Test diverse layers
    test_layers = [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.out_proj",
    ]

    print("=" * 60)
    print("Warpspec GEMM Fix Verification")
    print("=" * 60)

    all_pass = True
    for layer_name in test_layers:
        try:
            if not test_layer(layer_name):
                all_pass = False
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
