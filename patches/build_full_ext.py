#!/usr/bin/env python3
"""Build the prefill kernel as part of the full exllamav3_ext compilation.

Compiles ALL source files from exllamav3_ext (same as the existing extension)
PLUS our v6 kernel file, using the exact same build flags as ext.py.
"""

import os
import sys

# Set TORCH_CUDA_ARCH_LIST to match the existing extension
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0+PTX"

EXT_SRC_DIR = "/opt/exllamav3-python/exllamav3/exllamav3_ext"
QUANT_DIR = os.path.join(EXT_SRC_DIR, "quant")
BUILD_DIR = "/build"

os.makedirs("/tmp/exllamav3_prefill_build", exist_ok=True)
from torch.utils.cpp_extension import load
# Collect ALL source files from exllamav3_ext (same as ext.py)
# but exclude bindings.cpp to avoid duplicate PYBIND11_MODULE
sources = [
    os.path.abspath(os.path.join(root, file))
    for root, _, files in os.walk(EXT_SRC_DIR)
    for file in files
    if file.endswith(('.c', '.cpp', '.cu')) and file != 'bindings.cpp'
]

import shutil
shutil.copy(os.path.join(BUILD_DIR, "exl3_gemm_prefill_v3.cu"), os.path.join(QUANT_DIR, "exl3_gemm_prefill_v3.cu"))
shutil.copy(os.path.join(BUILD_DIR, "exl3_gemm_inner_prefill.cuh"), os.path.join(QUANT_DIR, "exl3_gemm_inner_prefill.cuh"))

# Add our v3 kernel from the quant dir
sources.append(os.path.join(QUANT_DIR, "exl3_gemm_prefill_v3.cu"))

print(f"Total source files: {len(sources)}")
for s in sources:
    print(f"  {s}")

# Include paths (same as ext.py)
include_dirs = []

# CUDA flags (same as ext.py)
cuda_flags = [
    "-lineinfo", "-O3", "--use_fast_math",
    "-Xcudafe", "--diag_suppress=177",
    "-Xcudafe", "--diag_suppress=20012",
]

cpp_flags = ["-Ofast"]

print(f"\nCUDA flags: {cuda_flags}")
print(f"CPP flags: {cpp_flags}")

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
