// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_prefill_v5.cu — Test with ORIGINAL unmodified inner kernel
//
// Identical to v4 but includes the original exl3_gemm_inner.cuh and calls
// the original exl3_gemm_kernel_inner function. If this produces correct
// results, the issue is in the inner kernel modification.

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
#include "exl3_gemm_inner.cuh"  // ORIGINAL unmodified inner kernel
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define V5_TILE_M    16
#define V5_TILE_K    32
#define V5_TILE_N    128
#define V5_SH_STAGES 4
#define V5_FRAG_STAGES 3

template<int bits, bool c_fp32, int cb,
         int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
         int SH_STAGES, int FRAG_STAGES>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_v5_kernel
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ suh,
    half* __restrict__ A_had,
    const half* __restrict__ svh
)
{
    auto grid = cg::this_grid();

    // Input Hadamard
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f
            );

        grid.sync();
        A = A_had;
    }

    // M-serialization loop
    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        // Use ORIGINAL inner kernel
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, 16), size_k, size_n, locks, svh);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh)
            grid.sync();
    }
}

template<int bits, bool c_fp32, int cb>
void exl3_gemm_v5_launch
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
    constexpr int TILE_M = V5_TILE_M;
    constexpr int TILE_K = V5_TILE_K;
    constexpr int TILE_N = V5_TILE_N;
    constexpr int SH_STAGES = V5_SH_STAGES;
    constexpr int FRAG_STAGES = V5_FRAG_STAGES;

    const int block_dim = EXL3_GEMM_BASE_THREADS * TILE_K / 16;

    constexpr int TILEBLOCKS_K = TILE_K / 16;
    constexpr int TILEBLOCKS_N = TILE_N / 16;
    constexpr int sh_a_stage_size = TILE_M * TILE_K;
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    constexpr int frags_n_per_warp = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);
    constexpr int sh_c_size = std::max(
        4 * EXL3_GEMM_BASE_THREADS * frags_n_per_warp,
        TILE_N * TILE_M
    );
    constexpr int shmem_bytes = SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size;
    // Use SMEM_MAX (same as existing kernel) instead of exact shmem_bytes
    constexpr int launch_shmem = EXL3_SMEM_MAX_BYTES;

    auto kernel = exl3_gemm_v5_kernel<
        bits, c_fp32, cb, TILE_M, TILE_K, TILE_N, SH_STAGES, FRAG_STAGES>;

    cudaError_t err = cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, launch_shmem);
    if (err != cudaSuccess) {
        printf("ERROR: cudaFuncSetAttribute failed: %s (shmem=%d)\n", cudaGetErrorString(err), (int)launch_shmem);
    }

    void* kernelArgs[] = {
        (void*)&A, (void*)&B, (void*)&C, (void*)&size_m,
        (void*)&size_k, (void*)&size_n, (void*)&locks,
        (void*)&suh, (void*)&A_had, (void*)&svh
    };

    err = cudaLaunchCooperativeKernel(
        (void*)kernel,
        dim3(num_sms),
        dim3(block_dim),
        kernelArgs,
        launch_shmem,
        at::cuda::getCurrentCUDAStream().stream()
    );
    if (err != cudaSuccess) {
        printf("ERROR: cudaLaunchCooperativeKernel failed: %s (grid=%d, block=%d, shmem=%d)\n",
               cudaGetErrorString(err), num_sms, block_dim, (int)launch_shmem);
    }
}

void exl3_gemm_prefill_v5
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
                exl3_gemm_v5_launch<BITS_VAL, true, CB_VAL>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh, num_sms); \
            else \
                exl3_gemm_v5_launch<BITS_VAL, false, CB_VAL>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh, num_sms); \
            return; \
        }

    DISPATCH(4, 1)
    DISPATCH(5, 1)
    DISPATCH(6, 1)
    DISPATCH(4, 0)
    DISPATCH(5, 0)
    DISPATCH(6, 0)

    #undef DISPATCH

    TORCH_CHECK(false, "exl3_gemm_prefill_v5: unsupported bits=", bits, " cb=", cb);
}

#include <torch/extension.h>

void exl3_gemm_prefill_v5_torch
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

    exl3_gemm_prefill_v5(
        A_ptr, B_ptr, C_ptr, size_m, size_k, size_n, bits, cb, c_fp32,
        suh_ptr, A_had_ptr, svh_ptr, locks_ptr, num_sms
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_gemm_prefill", &exl3_gemm_prefill_v5_torch, "EXL3 GEMM v5 (original inner kernel)");
}
