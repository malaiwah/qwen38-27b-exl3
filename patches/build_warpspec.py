#!/usr/bin/env python3
"""Build and test the warp-specialized EXL3 GEMM kernel.

Compiles exl3_gemm_warpspec.cu with all exllamav3_ext sources via
torch.utils.cpp_extension.load() with TORCH_CUDA_ARCH_LIST=12.0+PTX.
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0+PTX"

import torch
import time
from torch.utils.cpp_extension import load

# Paths inside the container
EXT_DIR = "/opt/exllamav3-python/exllamav3/exllamav3_ext"

# Source files needed for compilation
sources = [
    os.path.join(os.path.dirname(__file__), "exl3_gemm_warpspec.cu"),
]

# Include paths
extra_include_paths = [
    EXT_DIR,
    os.path.join(EXT_DIR, "quant"),
    os.path.join(EXT_DIR, "util"),
    os.path.dirname(__file__),
]

# Extra CUDA flags
extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
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
print(f"  Sources: {sources}")
print(f"  Include dirs: {extra_include_paths}")
print(f"  CUDA flags: {extra_cuda_cflags}")

module = load(
    name="exl3_gemm_warpspec",
    sources=sources,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=extra_cuda_cflags,
    extra_include_paths=extra_include_paths,
    verbose=True,
)

print("\n=== Compilation successful! ===")
print("Module functions:", [x for x in dir(module) if not x.startswith('_')])

# Quick smoke test
if torch.cuda.is_available():
    print("\n=== Smoke test ===")
    device = "cuda"
    
    # Create test matrices
    M, K, N = 64, 512, 128
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    
    # Create a fake B tensor (packed trellis format)
    # For bits=6, K=512: K//16 = 32 K-blocks, N//16 = 8 N-blocks
    # B shape: (K//16 * blocks_n, 256/16 * bits) = (32*8, 96) uint16s
    # Actually B is more complex. For testing, create random uint16
    blocks_n = N // 16
    b_size = (K // 16) * blocks_n * 256 // 16 * 6  # uint16s
    B = torch.randint(0, 65536, (b_size,), dtype=torch.int16, device=device).view(torch.uint16)
    
    # Locks
    locks = torch.zeros(blocks_n, dtype=torch.int32, device=device)
    
    # Post scale (none for now)
    post_scale = torch.empty(0, dtype=torch.float16, device=device)
    
    print(f"  A: {A.shape} {A.dtype}")
    print(f"  B: {B.shape} {B.dtype}")
    print(f"  M={M}, K={K}, N={N}, bits=6, cb=1")
    
    try:
        C = module.warpspec_gemm(A, B, M, K, N, 6, 1, locks, post_scale)
        print(f"  C: {C.shape} {C.dtype}")
        print(f"  C stats: min={C.min().item():.4f}, max={C.max().item():.4f}, mean={C.mean().item():.4f}")
        print("  PASS: Kernel ran without crash")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No CUDA available — compilation only test")

print("\nDone.")
