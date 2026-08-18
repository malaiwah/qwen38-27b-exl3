#!/usr/bin/env python3
"""Build the Marlin persistent kernel as part of full exllamav3_ext compilation."""

import os
import sys
import shutil

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

# Copy our files to the quant dir so relative includes work
shutil.copy(os.path.join(BUILD_DIR, "exl3_gemm_marlin.cu"), os.path.join(QUANT_DIR, "exl3_gemm_marlin.cu"))
shutil.copy(os.path.join(BUILD_DIR, "exl3_gemm_inner_prefill.cuh"), os.path.join(QUANT_DIR, "exl3_gemm_inner_prefill.cuh"))

# Add our Marlin kernel
sources.append(os.path.join(QUANT_DIR, "exl3_gemm_marlin.cu"))

print(f"Total source files: {len(sources)}")

cuda_flags = [
    "-lineinfo", "-O3", "--use_fast_math",
    "-Xcudafe", "--diag_suppress=177",
    "-Xcudafe", "--diag_suppress=20012",
]

cpp_flags = ["-Ofast"]

ext = load(
    name="exllamav3_prefill_ext",
    sources=sources,
    extra_include_paths=[],
    extra_cuda_cflags=cuda_flags,
    extra_cflags=cpp_flags,
    verbose=True,
    build_directory="/tmp/exllamav3_prefill_build",
)

print(f"\nBuild successful! Extension loaded: {ext}")
print(f"Available functions: {[f for f in dir(ext) if not f.startswith('_')]}")
