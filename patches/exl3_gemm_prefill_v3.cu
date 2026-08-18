// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_prefill_v3.cu — Fused dequant-in-epilogue GEMM for EXL3 trellis at prefill M.
//
// v3: True single-launch M-parallel grid tiling using the modified inner kernel
// (exl3_gemm_inner_prefill.cuh) that accepts num_slices and block_id as parameters.
//
// Grid layout:
//   dim3 grid(grid_m * grid_n)  — total blocks
//   Each block: m_tile = blockIdx.x / grid_n
//               n_slice = blockIdx.x % grid_n
//   Inner kernel receives: num_slices = grid_n, block_id = n_slice
//
// This gives each M-tile its own set of grid_n blocks that split K×N tiles,
// while M-tiles are parallel across blocks — no grid.sync(), no cooperative launch.

#pragma once

#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner_prefill.cuh"  // Modified inner kernel
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// Tile shape for prefill: TILESIZE_M=64 for 4 M-slabs per inner call.
#define PREFILL_TILE_M    64
#define PREFILL_TILE_K    32
#define PREFILL_TILE_N    128
#define PREFILL_SH_STAGES 4
#define PREFILL_FRAG_STAGES 3

template<int bits, bool c_fp32, int cb,
         int TILESIZE_M, int TILESIZE_K, int TILESIZE_N,
         int SH_STAGES, int FRAG_STAGES>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_prefill_kernel_v3
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
    const half* __restrict__ svh,
    const int grid_n  // Number of N-tile blocks per M-tile
)
{
    // Decode M-tile and N-slice from blockIdx.x
    const int m_tile = blockIdx.x / grid_n;
    const int n_slice = blockIdx.x % grid_n;
    const int m_offset = m_tile * TILESIZE_M;
    const int remaining_m = size_m - m_offset;
    const int active_m = MIN(remaining_m, TILESIZE_M);

    if (active_m <= 0) return;

    // Step 1: Apply input Hadamard (suh) to this block's M-slice.
    // Only the first n_slice==0 block per M-tile needs to do this,
    // but since blocks are independent we need each to write to its own
    // A_had region. To avoid races, we do the Hadamard in a separate
    // pre-pass kernel. For now, each block does its own Hadamard
    // (redundant across N-slices of the same M-tile, but correct).
    //
    // TODO: separate Hadamard pre-pass kernel for the M-slice.
    if (n_slice == 0)
    {
        int total_warps = active_m * size_k / 128;
        int warps_grid = blockDim.x / 32;
        int this_warp = threadIdx.x / 32;

        for (; this_warp < total_warps; this_warp += warps_grid)
        {
            had_hf_r_128_inner<true, false>(
                A + m_offset * size_k + this_warp * 128,
                A_had + m_offset * size_k + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );
        }
        __syncthreads();
    }

    // Step 2: Run the modified inner GEMM kernel.
    // Pass num_slices=grid_n and block_id=n_slice so the inner kernel
    // splits K×N tiles across the grid_n blocks of this M-tile.
    exl3_gemm_prefill_inner
    <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
    (
        A_had + m_offset * size_k,
        B,
        (char*)C + m_offset * size_n * (c_fp32 ? sizeof(float) : sizeof(half)),
        active_m,
        size_k,
        size_n,
        locks,
        svh,
        grid_n,    // num_slices: K×N tiles split across grid_n blocks
        n_slice    // block_id: this block's K×N slice index
    );
}

// Host dispatch
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

    const int grid_m = (size_m + TILE_M - 1) / TILE_M;
    const int grid_n = (size_n + TILE_N - 1) / TILE_N;
    const int total_blocks = grid_m * grid_n;
    const int block_dim = EXL3_GEMM_BASE_THREADS * TILE_K / 16;

    // Shared memory
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
        A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh, grid_n
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
    #define DISPATCH(B, CB) \
        if (bits == B && cb == CB) { \
            if (c_fp32) \
                exl3_gemm_prefill_v3_launch<B, true, CB>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); \
            else \
                exl3_gemm_prefill_v3_launch<B, false, CB>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); \
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
