#!/usr/bin/env python3
"""Build the exl3_gemm_prefill kernel as a standalone torch extension.

Compiles exl3_gemm_prefill_v3.cu + exl3_gemm_inner_prefill.cuh into a .so
that can be imported and called from Python.

Usage (inside the container with GPU free):
    python3 build_prefill_kernel.py

The resulting .so is saved to /opt/exllamav3/exllamav3_prefill_ext.so
"""

import os
import sys

# Source files are in /build (mounted volume)
BUILD_DIR = "/build"
EXT_SRC_DIR = "/opt/exllamav3-python/exllamav3/exllamav3_ext/quant"

# Set TORCH_CUDA_ARCH_LIST to match the existing extension
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0+PTX"

# Check environment
if not os.path.exists(os.path.join(EXT_SRC_DIR, "exl3_gemm_inner.cuh")):
    print(f"ERROR: exllamav3 ext sources not found at {EXT_SRC_DIR}")
    sys.exit(1)

from torch.utils.cpp_extension import load

sources = [
    os.path.join(BUILD_DIR, "exl3_gemm_v6.cu"),
]

# Include paths: ext source dir for headers, build dir for modified headers
include_dirs = [
    BUILD_DIR,                    # for exl3_gemm_inner_prefill.cuh
    EXT_SRC_DIR,                  # for exl3_kernel_map.cuh, hadamard_inner.cuh, exl3_dq.cuh, etc.
    os.path.dirname(EXT_SRC_DIR), # for util.h, util.cuh, ptx.cuh, etc.
    "/opt/venv/lib/python3.12/site-packages/torch/include",
    "/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include",
    "/usr/local/cuda/include",
]

# CUDA flags
cuda_flags = [
    "-lineinfo",
    "-O3",
    "--use_fast_math",
    "-Xcudafe", "--diag_suppress=177",
    "-Xcudafe", "--diag_suppress=20012",
]

cpp_flags = [
    "-O3",
    "-std=c++17",
    "-DWITH_CUDA",
]

os.makedirs("/tmp/exllamav3_prefill_build", exist_ok=True)

print(f"  Include dirs: {include_dirs}")
print(f"  CUDA flags: {cuda_flags}")

ext = load(
    name="exllamav3_prefill_ext",
    sources=sources,
    extra_include_paths=include_dirs,
    extra_cuda_cflags=cuda_flags,
    extra_cflags=cpp_flags,
    verbose=True,
    build_directory="/tmp/exllamav3_prefill_build",
)

print(f"\nBuild successful! Extension loaded: {ext}")
print(f"Available functions: {[f for f in dir(ext) if not f.startswith('_')]}")
