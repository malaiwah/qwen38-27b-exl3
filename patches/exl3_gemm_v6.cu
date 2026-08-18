// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_v6.cu — Use EXACT existing exl3_gemm_kernel template, directly instantiated

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../compat.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_gemm_kernel.cuh"  // EXACT existing outer kernel template
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <torch/extension.h>

// Shape 2: TILESIZE_M=16, TILESIZE_K=32, TILESIZE_N=128, SH_STAGES=4, FRAG_STAGES=3
// Block dim for shape 2: EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16 = 256 * 32/16 = 512
#define V6_BLOCK_DIM 512
#define V6_SMEM_MAX (90 * 1024)

template<int bits, bool c_fp32, int cb>
void exl3_gemm_v6_launch
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int* locks,
    const half* suh,
    half* A_had,
    const half* svh,
    int num_sms
)
{
    // Directly instantiate the existing kernel template with shape 2 params
    auto kernel = exl3_gemm_kernel<
        bits, c_fp32, cb, 16, 32, 128, 4, 3>;

    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, V6_SMEM_MAX);

    void* kernelArgs[] = {
        (void*)&A, (void*)&B, (void*)&C, (void*)&size_m,
        (void*)&size_k, (void*)&size_n, (void*)&locks,
        (void*)&suh, (void*)&A_had, (void*)&svh
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)kernel,
        dim3(num_sms),
        dim3(V6_BLOCK_DIM),
        kernelArgs,
        V6_SMEM_MAX,
        at::cuda::getCurrentCUDAStream().stream()
    );
    if (err != cudaSuccess) {
        printf("ERROR: cudaLaunchCooperativeKernel failed: %s\n", cudaGetErrorString(err));
    }
}

void exl3_gemm_v6
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    int cb,
    bool c_fp32,
    const half* suh,
    half* A_had,
    const half* svh,
    int* locks,
    int num_sms
)
{
    #define DISPATCH(BITS_VAL, CB_VAL) \
        if (bits == BITS_VAL && cb == CB_VAL) { \
            if (c_fp32) \
                exl3_gemm_v6_launch<BITS_VAL, true, CB_VAL>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh, num_sms); \
            else \
                exl3_gemm_v6_launch<BITS_VAL, false, CB_VAL>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh, num_sms); \
            return; \
        }

    DISPATCH(4, 1)
    DISPATCH(5, 1)
    DISPATCH(6, 1)
    DISPATCH(4, 0)
    DISPATCH(5, 0)
    DISPATCH(6, 0)

    #undef DISPATCH
    TORCH_CHECK(false, "v6: unsupported bits=", bits, " cb=", cb);
}

void exl3_gemm_v6_torch
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
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16 / bits;
    bool c_fp32 = (C.scalar_type() == at::ScalarType::Float);

    const half* A_ptr = (const half*)A.data_ptr<at::Half>();
    const uint16_t* B_ptr = (const uint16_t*)B.data_ptr();
    void* C_ptr = C.data_ptr();
    const half* suh_ptr = suh ? (const half*)suh->data_ptr<at::Half>() : nullptr;
    half* A_had_ptr = (half*)A_had.data_ptr<at::Half>();
    const half* svh_ptr = svh ? (const half*)svh->data_ptr<at::Half>() : nullptr;
    int* locks_ptr = (int*)locks.data_ptr();

    exl3_gemm_v6(A_ptr, B_ptr, C_ptr, size_m, size_k, size_n, bits, cb, c_fp32,
                 suh_ptr, A_had_ptr, svh_ptr, locks_ptr, num_sms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_gemm_prefill", &exl3_gemm_v6_torch, "EXL3 GEMM v6 (exact existing kernel template)");
}
