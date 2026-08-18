// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_v7.cu — Call the EXISTING exl3_gemm function directly
// This verifies the call path works correctly when using the existing function.

#pragma once

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "exl3_gemm.cuh"  // same directory

void exl3_gemm_v7_torch
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    at::Tensor& A_had,
    const c10::optional<at::Tensor>& svh,
    int bits,
    int cb,
    at::Tensor& locks,
    int num_sms
)
{
    // Call the EXISTING exl3_gemm function (defined in exl3_gemm.cu)
    // with force_shape_idx=2 (shape 2: M=16, K=32, N=128)
    bool mcg = (cb == 1);
    bool mul1 = (cb == 2);
    
    exl3_gemm(
        A, B, C, suh, A_had, svh,
        2,       // force_shape_idx = shape 2
        mcg,
        mul1,
        0        // force_num_sms (use default)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_gemm_prefill", &exl3_gemm_v7_torch, "EXL3 GEMM v7 (calls existing exl3_gemm)");
}
