// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_prefill_v3.cu — Fused dequant-in-epilogue GEMM for EXL3 trellis at prefill M.
//
// v3: True single-launch M-parallel grid tiling using the modified inner kernel
// (exl3_gemm_inner_prefill.cuh) that accepts num_slices and block_id as parameters.
//
// Two separate kernel launches:
//   1. Input Hadamard pre-pass: processes entire A matrix → A_had
//   2. GEMM kernel: M-parallel grid tiling, each block handles one (M-tile, N-slice)
//
// Grid layout for GEMM:
//   dim3 grid(grid_m * grid_n)  — total blocks
//   Each block: m_tile = blockIdx.x / grid_n
//               n_slice = blockIdx.x % grid_n
//   Inner kernel receives: num_slices = grid_n, block_id = n_slice

// (no #pragma once for .cu file)

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include "../util.h"
#include "../util.cuh"
#include "../compat.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner_prefill.cuh"  // Modified inner kernel (needs EXL3_GEMM_T_ARGS from kernel_map)
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
// Tile shape for prefill
#define PREFILL_TILE_M    16
#define PREFILL_TILE_K    32
#define PREFILL_TILE_N    128
#define PREFILL_SH_STAGES 4
#define PREFILL_FRAG_STAGES 3

// ============================================================================
// Kernel 1: Input Hadamard pre-pass
// ============================================================================
// Each warp processes 128 elements of A, applies Hadamard + suh scales → A_had
__global__ void exl3_input_hadamard_kernel
(
    const half* __restrict__  A,
    half* __restrict__ A_had,
    const half* __restrict__ suh,
    const int size_m,
    const int size_k
)
{
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

    for (; this_warp < total_warps; this_warp += warps_grid)
    {
        had_hf_r_128_inner<true, false>
        (
            A + this_warp * 128,
            A_had + this_warp * 128,
            suh + (this_warp * 128) % size_k,
            0.088388347648f  // 1/sqrt(128)
        );
    }
}

// ============================================================================
// Kernel 2: GEMM with M-parallel grid tiling
// ============================================================================
template<int bits, bool c_fp32, int cb,
         int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
         int SH_STAGES, int FRAG_STAGES>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_prefill_kernel_v3
(
    const half* __restrict__  A_had,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ svh,
    const int grid_n
)
{
    // Decode M-tile and N-slice from blockIdx.x
    const int m_tile = blockIdx.x / grid_n;
    const int n_slice = blockIdx.x % grid_n;
    const int m_offset = m_tile * TILESIZE_M;
    const int remaining_m = size_m - m_offset;
    const int active_m = MIN(remaining_m, TILESIZE_M);

    if (active_m <= 0) return;

    // Run the modified inner GEMM kernel.
    // A_had is already Hadamard-transformed by the pre-pass kernel.
    // The inner kernel does the GEMM + output Hadamard (svh).
    exl3_gemm_prefill_inner
    <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
    (
        A_had + m_offset * size_k,
        B,
        (char*)C + m_offset * size_n * (c_fp32 ? sizeof(float) : sizeof(half)),
        active_m,
        size_k,
        size_n,
        locks + m_tile * (size_n / 16),  // per-M-tile lock offset
        svh,
        grid_n,    // num_slices: K×N tiles split across grid_n blocks
        n_slice    // block_id: this block's K×N slice index
    );
}

// ============================================================================
// Host dispatch
// ============================================================================
template<int bits, bool c_fp32, int cb>
void exl3_gemm_prefill_v3_launch
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
    const half* svh
)
{
    constexpr int TILE_M = PREFILL_TILE_M;
    constexpr int TILE_K = PREFILL_TILE_K;
    constexpr int TILE_N = PREFILL_TILE_N;
    constexpr int SH_STAGES = PREFILL_SH_STAGES;
    constexpr int FRAG_STAGES = PREFILL_FRAG_STAGES;

    // Step 1: Launch input Hadamard pre-pass
    {
        int total_warps = size_m * size_k / 128;
        int warps_per_block = 128 / 32;  // 4 warps per block with 128 threads
        int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
        blocks = MIN(blocks, 65535);  // CUDA grid limit
        int threads = 128;
        exl3_input_hadamard_kernel<<<blocks, threads>>>(
            A, A_had, suh, size_m, size_k
        );
    }

    // Step 2: Launch GEMM kernel
    const int grid_m = (size_m + TILE_M - 1) / TILE_M;
    const int grid_n = (size_n + TILE_N - 1) / TILE_N;
    const int total_blocks = grid_m * grid_n;
    const int block_dim = EXL3_GEMM_BASE_THREADS * TILE_K / 16;

    // Shared memory for GEMM kernel
    constexpr int TILEBLOCKS_M = TILE_M / 16;
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

    static_assert(shmem_bytes <= EXL3_SMEM_MAX_BYTES,
                  "Prefill kernel shared memory exceeds device limit");

    auto kernel = exl3_gemm_prefill_kernel_v3<
        bits, c_fp32, cb, TILE_M, TILE_K, TILE_N, SH_STAGES, FRAG_STAGES>;

    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    kernel<<<total_blocks, block_dim, shmem_bytes>>>(
        A_had, B, C, size_m, size_k, size_n, locks, svh, grid_n
    );
}

// Public C API
void exl3_gemm_prefill
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
    int* locks
)
{
    #define DISPATCH(BITS_VAL, CB_VAL) \
        if (bits == BITS_VAL && cb == CB_VAL) { \
            if (c_fp32) \
                exl3_gemm_prefill_v3_launch<BITS_VAL, true, CB_VAL>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); \
            else \
                exl3_gemm_prefill_v3_launch<BITS_VAL, false, CB_VAL>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); \
            return; \
        }

    DISPATCH(4, 1)
    DISPATCH(5, 1)
    DISPATCH(6, 1)
    DISPATCH(4, 0)
    DISPATCH(5, 0)
    DISPATCH(6, 0)

    #undef DISPATCH

    TORCH_CHECK(false, "exl3_gemm_prefill: unsupported bits=", bits, " cb=", cb);
}

// ============================================================================
// Python binding
// ============================================================================
#include <torch/extension.h>

void exl3_gemm_prefill_torch
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
    int num_sms  // ignored, uses DevCtx
)
{
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;  // B shape is (K//16, N//16, 16*bits), so N = B.size(1)*16
    bool c_fp32 = (C.scalar_type() == at::ScalarType::Float);

    const half* A_ptr = (const half*)A.data_ptr<at::Half>();
    const uint16_t* B_ptr = (const uint16_t*)B.data_ptr();
    void* C_ptr = C.data_ptr();
    const half* suh_ptr = suh ? (const half*)suh->data_ptr<at::Half>() : nullptr;
    half* A_had_ptr = (half*)A_had.data_ptr<at::Half>();
    const half* svh_ptr = svh ? (const half*)svh->data_ptr<at::Half>() : nullptr;

    // Use DevCtx for locks (same as existing kernel)
    int device;
    cudaGetDevice(&device);
    int* locks_ptr = DevCtx::instance().get_locks(device);

    exl3_gemm_prefill(
        A_ptr, B_ptr, C_ptr, size_m, size_k, size_n, bits, cb, c_fp32,
        suh_ptr, A_had_ptr, svh_ptr, locks_ptr
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_gemm_prefill", &exl3_gemm_prefill_torch, "Fused EXL3 trellis GEMM with Hadamard for prefill");
}
