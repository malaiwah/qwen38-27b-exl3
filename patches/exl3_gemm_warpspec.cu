// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_warpspec.cu — Warp-specialized W6A16 GEMM kernel for EXL3 trellis
//
// Design: Producer/consumer warp split to eliminate ALU/MMA contention.
//
// Block: 256 threads = 8 warps (TILE_K=16, TILEBLOCKS_K=1, no sub_k)
//   Warps 0-3 (Producer): load packed B from shared → dequant → write FP16 to shared
//   Warps 4-7 (Consumer): ldmatrix A + load FP16 B from shared → MMA
//
// The key insight: in the existing kernel, dequant (integer ALU) and MMA (tensor
// cores) run on the SAME warps, causing instruction-level contention. By splitting
// into producer/consumer warps, integer ALU and tensor cores run on independent
// warp schedulers, achieving true overlap.
//
// SM120 (RTX 5090): mma.sync m16n8k16 with fp16 accumulate (H_ACC=1).
// 101KB optin shared memory, 170 SMs.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ============================================================================
// PTX helpers (vendored from exllamav3/b12x)
// ============================================================================

template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

// Union for half2/uint32 reinterpretation
union half2_uint32 {
    uint32_t u32;
    half2 h2;
    __device__ half2_uint32(uint32_t u) : u32(u) {}
    __device__ half2_uint32(half2 h) : h2(h) {}
    __device__ half2 as_half2() const { return h2; }
};

union half_uint16 {
    uint16_t u16;
    half h;
    __device__ half_uint16(uint16_t u) : u16(u) {}
    __device__ half_uint16(half h) : h(h) {}
    __device__ half as_half() const { return h; }
};

// m16n8k16 MMA — fp32 accumulate
__device__ inline void ptx_mma_m16n8k16(const FragA& a, const FragB& b, FragC& c) {
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(&b);
    float* pc = reinterpret_cast<float*>(&c);
    const float* pd = reinterpret_cast<const float*>(&c);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(pc[0]), "=f"(pc[1]), "=f"(pc[2]), "=f"(pc[3])
        : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]),
          "r"(pb[0]), "r"(pb[1]),
          "f"(pd[0]), "f"(pd[1]), "f"(pd[2]), "f"(pd[3]));
}

// m16n8k16 MMA — fp16 accumulate (for SM120 H_ACC=1)
__device__ inline void ptx_mma_m16n8k16_h(const FragA& a, const FragB& b, FragC_h& c) {
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* pc = reinterpret_cast<uint32_t*>(&c);
    const uint32_t* pd = reinterpret_cast<const uint32_t*>(&c);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(pc[0]), "=r"(pc[1])
        : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]),
          "r"(pb[0]), "r"(pb[1]),
          "r"(pd[0]), "r"(pd[1]));
}

// ldmatrix.x4 — load 4 8×8 matrices for FragA
__device__ inline void ldsm4(FragA& frag, const void* smem) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag);
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}

// cp.async — 16-byte global to shared
__device__ inline void cp_async(void* smem, const void* glob) {
    uint32_t addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(glob));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Funnel shift
__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// Inter-block barrier (same as existing kernel)
__device__ inline void barrier_acquire(int* lock, int stage) {
    if (threadIdx.x == 0) {
        volatile int state = -1;
        do {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        } while (state != stage);
    }
    __syncthreads();
}

__device__ inline void barrier_release(int* lock, int val, bool reset) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) { *lock = 0; return; }
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(lock), "r"(val));
    }
}

// ============================================================================
// Codebook decode (vendored from exllamav3/b12x)
// ============================================================================

template <uint32_t w>
__device__ __forceinline__ uint32_t mul_const_u32(uint32_t x) { return x * w; }

template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
    if constexpr (cb == 1) {
        x0 = mul_const_u32<0xCBAC1FEDu>(x0);
        x1 = mul_const_u32<0xCBAC1FEDu>(x1);
        asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
        asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
        half2_uint32 xu0(x0), xu1(x1);
        half2 d0 = __lows2half2(xu0.as_half2(), xu1.as_half2());
        half2 d1 = __highs2half2(xu0.as_half2(), xu1.as_half2());
        return __hadd2(d0, d1);
    }
    if constexpr (cb == 0) {
        x0 *= 89226354u; x1 *= 89226354u;
        x0 += 64248484u; x1 += 64248484u;
        asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
        asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
        half2_uint32 xu0(x0), xu1(x1);
        half2 d0 = __lows2half2(xu0.as_half2(), xu1.as_half2());
        half2 d1 = __highs2half2(xu0.as_half2(), xu1.as_half2());
        return __hadd2(d0, d1);
    }
}

// ============================================================================
// Dequant (vendored from exl3_dq.cuh)
// ============================================================================

template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag) {
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w3 = fshift(b, a, s2) & 0xffff;
    uint32_t w2 = fshift(b, a, s2 + bits) & 0xffff;
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    frag[0] = decode_3inst_2<cb>(w0, w1);
    frag[1] = decode_3inst_2<cb>(w2, w3);
}

template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1) {
    if constexpr (bits == 6) {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    } else if constexpr (bits == 5) {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    } else if constexpr (bits == 4) {
        // W4 aligned path
        uint32_t i1 = idx >> 3;
        uint32_t i0 = (i1 + 31) & 31;
        uint32_t a = ptr[i0], b = ptr[i1], s;
        FSHF_IMM(s, b, a, 20);
        uint32_t w7 = b & 0xffff, w6, w5, w4, w3, w2, w1, w0;
        BFE16_IMM(w6, b, 4); BFE16_IMM(w5, b, 8); BFE16_IMM(w4, b, 12);
        BFE16_IMM(w3, b, 16);
        w2 = s & 0xffff;
        BFE16_IMM(w1, s, 4); BFE16_IMM(w0, s, 8);
        frag0[0] = decode_3inst_2<cb>(w0, w1);
        frag0[1] = decode_3inst_2<cb>(w2, w3);
        frag1[0] = decode_3inst_2<cb>(w4, w5);
        frag1[1] = decode_3inst_2<cb>(w6, w7);
    } else {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
}

// ============================================================================
// Hadamard helpers (simplified — for output transform)
// ============================================================================

// Hadamard transform for 128-element rows (same as existing kernel)
// This is a placeholder — the actual hadamard_inner.cuh is included from the
// container's ext sources during compilation.

// ============================================================================
// Constants
// ============================================================================

#define WS_BASE_THREADS 256
#define WS_NUM_WARPS 8
#define WS_PROD_WARPS 4
#define WS_CONS_WARPS 4

// Tile config: TILE_K=16 to avoid sub_k complexity, TILE_M=64 for Marlin scale
#define WS_TILE_M 64
#define WS_TILE_K 16
#define WS_TILE_N 128
#define WS_TILEBLOCKS_M (WS_TILE_M / 16)   // 4
#define WS_TILEBLOCKS_K (WS_TILE_K / 16)   // 1
#define WS_TILEBLOCKS_N (WS_TILE_N / 16)   // 8
#define WS_SH_STAGES 4
#define WS_FRAG_STAGES 3

// H_ACC for SM120
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
#define WS_H_ACC 1
#else
#define WS_H_ACC 0
#endif

// XOR-swizzle for A
#define WS_A_COLS (WS_TILE_K / 8)
#define WS_A_SWIZZLE_MASK (WS_A_COLS - 1)
#define WS_A_SWIZZLE_SHIFT ((WS_A_COLS <= 2) ? 2 : 1)

// Consumer warp config
#define WS_FRAGS_N_PER_CONS (2 * WS_TILEBLOCKS_N / WS_CONS_WARPS)  // 2*8/4 = 4

// Producer warp config
#define WS_N_BLOCKS_PER_PROD (WS_TILEBLOCKS_N / WS_PROD_WARPS)  // 8/4 = 2

// Shared memory sizes
#define WS_SH_A_STAGE (WS_TILE_M * WS_TILE_K)                    // 64*16 = 1024 halfs
#define WS_SH_B_STAGE (WS_TILEBLOCKS_K * WS_TILEBLOCKS_N * 256 / 16 * 6)  // 1*8*96 = 768 uint16s
// FP16 B buffer: per stage, per K-block, per N-block, 32 threads × 2 FragB × 2 half2
#define WS_SH_FP16_B_STAGE (WS_TILEBLOCKS_K * WS_TILEBLOCKS_N * 32 * 2 * 2 * 2)  // 1*8*32*2*2 = 1024 halfs
#define WS_SH_C_SIZE (4 * WS_CONS_WARPS * 32 * WS_FRAGS_N_PER_CONS)  // 4*128*4 = 2048 floats

// ============================================================================
// Warp-specialized inner kernel
// ============================================================================

template<int bits, bool c_fp32, int cb, bool shmem_out_had>
inline __device__
void warpspec_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    int block_id,
    int num_slices
)
{
    const int TILEBLOCKS_M = WS_TILEBLOCKS_M;
    const int TILEBLOCKS_K = WS_TILEBLOCKS_K;
    const int TILEBLOCKS_N = WS_TILEBLOCKS_N;

    // Thread info
    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane_id = t % 32;
    const bool is_producer = warp_id < WS_PROD_WARPS;
    const bool is_consumer = warp_id >= WS_PROD_WARPS;
    const int cons_warp = warp_id - WS_PROD_WARPS;  // 0-3 for consumers
    const int prod_warp = warp_id;                   // 0-3 for producers

    // Dimensions
    int tiles_k = size_k / WS_TILE_K;
    int tiles_n = size_n / WS_TILE_N;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    // Slice assignment
    int total_tiles = tiles_k * tiles_n;
    int slice_beg = total_tiles * block_id / num_slices;
    int slice_end = total_tiles * (block_id + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    // Shared memory layout
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*)(sh_a + WS_SH_STAGES * WS_SH_A_STAGE);
    half* sh_fp16_b = (half*)(sh_b + WS_SH_B_STAGE * WS_SH_STAGES);
    float* sh_c = (float*)(sh_fp16_b + WS_SH_FP16_B_STAGE * WS_FRAG_STAGES);

    // XOR-swizzle for A
    const int A_COLS = WS_A_COLS;
    const int A_SWIZZLE_MASK = WS_A_SWIZZLE_MASK;
    const int A_SWIZZLE_SHIFT = WS_A_SWIZZLE_SHIFT;

    // ---- Pipe 0: Global → Shared ----
    int slice0_k = (slice_beg % tiles_k);
    int slice0_n = (slice_beg / tiles_k);
    int slice0_iters = slice_len;

    int gl_a_stride_m = WS_TILE_M * size_k;
    const int gl_a_stride_k = WS_TILE_K;
    const half* gl_a_ptr = A + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % WS_SH_STAGES) * WS_SH_A_STAGE;

    const int load_a_iters = (WS_SH_A_STAGE / 8 + WS_BASE_THREADS - 1) / WS_BASE_THREADS;
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i) {
        int k = (i * WS_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * WS_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % WS_SH_STAGES) * WS_SH_B_STAGE;

    const int load_b_iters = (WS_SH_B_STAGE / 8 + WS_BASE_THREADS - 1) / WS_BASE_THREADS;
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i) {
        int n = (i * WS_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * WS_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * WS_BASE_THREADS + t < WS_SH_B_STAGE / 8;
    }

    auto advance0 = [&]() {
        slice0_k++;
        slice0_iters--;
        int stage = slice0_iters % WS_SH_STAGES;
        sh0_a_ptr = sh_a + stage * WS_SH_A_STAGE;
        sh0_b_ptr = sh_b + stage * WS_SH_B_STAGE;
        if (slice0_k >= tiles_k) {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        } else {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    auto async_load_gl = [&]() {
        if (slice0_iters) {
            {
                const int4* gl = (const int4*)gl_a_ptr;
                int4* sh = (int4*)sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
            }
            {
                const int4* gl = (const int4*)gl_b_ptr;
                int4* sh = (int4*)sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                    if (pred_b_gl[i]) cp_async(sh + WS_BASE_THREADS * i + t, gl + load_b_gl[i]);
            }
            advance0();
        }
        cp_async_fence();
    };

    auto wait_stage = [&]() {
        cp_async_wait<WS_SH_STAGES - 2>();
        __syncthreads();
    };

    // ---- Pipe 1: Shared → Registers (split producer/consumer) ----
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;
    half* sh1_a_ptr = sh_a + (slice1_iters % WS_SH_STAGES) * WS_SH_A_STAGE;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % WS_SH_STAGES) * WS_SH_B_STAGE;

    auto advance1 = [&]() {
        slice1_k++;
        slice1_iters--;
        int stage = slice1_iters % WS_SH_STAGES;
        sh1_a_ptr = sh_a + stage * WS_SH_A_STAGE;
        sh1_b_ptr = sh_b + stage * WS_SH_B_STAGE;
        if (slice1_k >= tiles_k) { slice1_k = 0; slice1_n++; }
    };

    // Register fragments — declared before lambdas that capture them
    FragA frag_a_reg[WS_FRAG_STAGES][WS_TILEBLOCKS_M];
    FragB frag_b_reg[WS_FRAG_STAGES][WS_FRAGS_N_PER_CONS];
    FragC frag_c[WS_TILEBLOCKS_M][WS_FRAGS_N_PER_CONS];
#if WS_H_ACC
    FragC_h frag_c_h[WS_TILEBLOCKS_M][WS_FRAGS_N_PER_CONS];
#endif
    // FP16 B buffer index
    // sh_fp16_b layout: [FRAG_STAGES][TILEBLOCKS_K][TILEBLOCKS_N][32 threads × 2 FragB × 2 half2]
    // = [FRAG_STAGES][TILEBLOCKS_K][TILEBLOCKS_N][32][4] halfs
    // Each FragB = 2 half2 = 4 halfs per thread per N-block
    #define FP16_B_OFFSET(stage, sk, sn) \
        ((stage) * WS_TILEBLOCKS_K * WS_TILEBLOCKS_N * 256 + \
         (sk) * WS_TILEBLOCKS_N * 256 + \
         (sn) * 256)

    // Producer: dequant B → FP16 shared
    auto producer_load_frags = [&](int buf) {
        if (!slice1_iters) return;
        // Each producer warp handles WS_N_BLOCKS_PER_PROD N-blocks
        #pragma unroll
        for (int nb = 0; nb < WS_N_BLOCKS_PER_PROD; ++nb) {
            int sub_n = prod_warp * WS_N_BLOCKS_PER_PROD + nb;
            const uint32_t* shb = (const uint32_t*)(
                sh1_b_ptr + sub_n * 256 / 16 * bits);
            FragB frag0, frag1;
            dq_dispatch<bits, cb>(shb, lane_id << 3, frag0, frag1);
            // Write to FP16 shared buffer
            half2* dst = (half2*)(sh_fp16_b + FP16_B_OFFSET(buf % WS_FRAG_STAGES, 0, sub_n));
            dst[lane_id * 2 + 0] = frag0[0];
            dst[lane_id * 2 + 1] = frag0[1];
            dst[lane_id * 2 + 64] = frag1[0];
            dst[lane_id * 2 + 65] = frag1[1];
        }
        // advance1() is called explicitly in the main loop after consumer loads
    };

    // Consumer: ldmatrix A + load FP16 B → registers
    auto consumer_load_frags = [&](int buf) {
        if (!slice1_iters) return;
        // Load A fragments via ldmatrix
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m) {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a_reg[buf][m], (int4*)sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }
        // Load FP16 B fragments from shared
        #pragma unroll
        for (int n2 = 0; n2 < WS_FRAGS_N_PER_CONS; n2 += 2) {
            int sub_n = cons_warp * (WS_FRAGS_N_PER_CONS / 2) + n2 / 2;
            // Map sub_n to the N-block index. Each N-block produces 2 N-fragments.
            // sub_n here is an N-fragment index. The N-block index is sub_n / 2
            // But actually, in the producer, each N-block produces frag0 and frag1
            // which map to 2 N-fragments. So N-fragment i comes from N-block i/2, frag i%2.
            int n_block = sub_n;  // Each producer N-block maps to one consumer N-fragment pair
            half2* src = (half2*)(sh_fp16_b + FP16_B_OFFSET(buf % WS_FRAG_STAGES, 0, n_block));
            frag_b_reg[buf][n2][0] = src[lane_id * 2 + 0];
            frag_b_reg[buf][n2][1] = src[lane_id * 2 + 1];
            frag_b_reg[buf][n2 + 1][0] = src[lane_id * 2 + 64];
            frag_b_reg[buf][n2 + 1][1] = src[lane_id * 2 + 65];
        }
        // advance1() is called explicitly in the main loop after consumer loads
    };


    auto clear_frag_c = [&]() {
        #pragma unroll
        for (int m = 0; m < WS_TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                frag_c[m][n] = {};
                #if WS_H_ACC
                frag_c_h[m][n] = {};
                #endif
            }
    };

    // Matmul — consumer only
    auto matmul = [&](int buf) {
        #pragma unroll
        for (int m = 0; m < WS_TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                #if WS_H_ACC
                ptx_mma_m16n8k16_h(frag_a_reg[buf][m], frag_b_reg[buf][n], frag_c_h[m][n]);
                #else
                ptx_mma_m16n8k16(frag_a_reg[buf][m], frag_b_reg[buf][n], frag_c[m][n]);
                #endif
            }
    };

    // ---- Pipe 2: Output ----
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = WS_TILE_N;
    int gl_c_stride_m = WS_TILE_M * size_n;
    half* gl_c_ptr_16 = ((half*)C) + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*)C) + slice2_n * gl_c_stride_n;

    auto advance2 = [&]() {
        slice2_k++;
        slice2_iters--;
        if (slice2_k >= tiles_k) {
            slice2_k = 0; slice2_k0 = 0; slice2_n++;
            if constexpr (c_fp32) gl_c_ptr_32 += gl_c_stride_n;
            else gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Reduction (consumer only, but all threads sync)
    auto reduce = [&]() {
        #if WS_H_ACC
        // Fold fp16 accumulators into fp32
        if (is_consumer) {
            #pragma unroll
            for (int m = 0; m < WS_TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                    float2 f0 = __half22float2(frag_c_h[m][n][0]);
                    float2 f1 = __half22float2(frag_c_h[m][n][1]);
                    frag_c[m][n][0] += f0.x; frag_c[m][n][1] += f0.y;
                    frag_c[m][n][2] += f1.x; frag_c[m][n][3] += f1.y;
                }
        }
        #endif

        // No threadblock reduction needed — TILEBLOCKS_K=1, single K-block per tile
        // Just inter-block reduction across slices

        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice2_n];

        barrier_acquire(lock, lock_i);
        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        // Read intermediate sum from global (not first block)
        if (is_consumer && !first) {
            int n0 = cons_warp * WS_FRAGS_N_PER_CONS;
            #pragma unroll
            for (int m = 0; m < WS_TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                    int r0 = m * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m) {
                        if constexpr (c_fp32) {
                            float* p = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][0] += *p++; frag_c[m][n][1] += *p++;
                        } else {
                            half2* p = (half2*)(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            float2 v = __half22float2(*p);
                            frag_c[m][n][0] += v.x; frag_c[m][n][1] += v.y;
                        }
                    }
                    if (r1 < size_m) {
                        if constexpr (c_fp32) {
                            float* p = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][2] += *p++; frag_c[m][n][3] += *p++;
                        } else {
                            half2* p = (half2*)(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            float2 v = __half22float2(*p);
                            frag_c[m][n][2] += v.x; frag_c[m][n][3] += v.y;
                        }
                    }
                }
        }

        // Write intermediate or final result
        if (is_consumer && !last) {
            int n0 = cons_warp * WS_FRAGS_N_PER_CONS;
            #pragma unroll
            for (int m = 0; m < WS_TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                    int r0 = m * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m) {
                        if constexpr (c_fp32) {
                            float* p = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *p++ = frag_c[m][n][0]; *p++ = frag_c[m][n][1];
                        } else {
                            half2* p = (half2*)(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            *p = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                        }
                    }
                    if (r1 < size_m) {
                        if constexpr (c_fp32) {
                            float* p = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *p++ = frag_c[m][n][2]; *p++ = frag_c[m][n][3];
                        } else {
                            half2* p = (half2*)(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            *p = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                        }
                    }
                }
        }

        // Last block: write final output directly to global (no sh_c round-trip)
        if (is_consumer && last) {
            int n0 = cons_warp * WS_FRAGS_N_PER_CONS;
            #pragma unroll
            for (int m = 0; m < WS_TILEBLOCKS_M; ++m) {
                int r0 = m * 16 + lane_id / 4;
                int r1 = r0 + 8;
                int c = (lane_id % 4) * 2;
                if (r0 < size_m) {
                    #pragma unroll
                    for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                        if constexpr (c_fp32) {
                            float* p = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *p++ = frag_c[m][n][0]; *p++ = frag_c[m][n][1];
                        } else {
                            half2* p = (half2*)(gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            *p = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                        }
                    }
                }
                if (r1 < size_m) {
                    #pragma unroll
                    for (int n = 0; n < WS_FRAGS_N_PER_CONS; ++n) {
                        if constexpr (c_fp32) {
                            float* p = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *p++ = frag_c[m][n][2]; *p++ = frag_c[m][n][3];
                        } else {
                            half2* p = (half2*)(gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            *p = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                        }
                    }
                }
            }
        }

        barrier_release(lock, lock_d, last);
        if (is_consumer) clear_frag_c();
    };

    // ---- Main loop ----
    // Pipeline: async_load → wait → [producer: dequant, consumer: ldmatrix A + load FP16 B] → sync → matmul

    // Prefetch
    #pragma unroll
    for (int i = 0; i < WS_SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    clear_frag_c();
    if constexpr (WS_FRAG_STAGES > 1) {
        // Phase 1: producer dequants B → sh_fp16_b[0]
        if (is_producer) producer_load_frags(0);
        __syncthreads();
        // Phase 2: consumer loads A (ldmatrix) + reads sh_fp16_b[0]
        if (is_consumer) consumer_load_frags(0);
        __syncthreads();
        advance1();
    }

    #define WS_FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        /* Phase 1: producer dequants next B while consumer matmuls current frags */ \
        if (is_producer) producer_load_frags(_load); \
        else matmul(_mul); \
        __syncthreads(); \
        /* Phase 2: consumer loads A + reads sh_fp16_b written by producer */ \
        if (is_consumer) consumer_load_frags(_load); \
        advance1(); \
        __syncthreads(); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break;

    if constexpr (WS_FRAG_STAGES == 3) {
        while (true) {
            WS_FSTAGE(1, 0);
            WS_FSTAGE(2, 1);
            WS_FSTAGE(0, 2);
        }
    } else if constexpr (WS_FRAG_STAGES == 2) {
        while (true) {
            WS_FSTAGE(1, 0);
            WS_FSTAGE(0, 1);
        }
    } else {
        while (true) {
            WS_FSTAGE(0, 0);
        }
    }
    #undef WS_FSTAGE
    #undef FP16_B_OFFSET
}

// ============================================================================
// Host launcher
// ============================================================================


// Forward declaration
template<int bits, bool c_fp32, int cb, bool shmem_out_had>
__global__ void warpspec_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ locks,
    const half* __restrict__ post_scale
);

template<int bits, bool c_fp32, int cb, bool shmem_out_had>
void launch_warpspec
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int* locks,
    const half* post_scale,
    cudaStream_t stream
)
{
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    int tiles_k = size_k / WS_TILE_K;
    int tiles_n = size_n / WS_TILE_N;
    int total_tiles = tiles_k * tiles_n;

    // Persistent grid: use min(total_tiles, num_sms) blocks
    int grid = total_tiles < num_sms ? total_tiles : num_sms;

    // Shared memory size
    int sh_a = WS_SH_STAGES * WS_SH_A_STAGE * sizeof(half);
    int sh_b = WS_SH_STAGES * WS_SH_B_STAGE * sizeof(uint16_t);
    int sh_fp16_b = WS_FRAG_STAGES * WS_SH_FP16_B_STAGE * sizeof(half);
    int sh_c = WS_SH_C_SIZE * sizeof(float);
    // For shmem_out_had, need TILE_N * TILE_M floats
    if (shmem_out_had) {
        int sh_c_had = WS_TILE_N * WS_TILE_M * sizeof(float);
        if (sh_c_had > sh_c) sh_c = sh_c_had;
    }
    int shmem = sh_a + sh_b + sh_fp16_b + sh_c;

    // Set optin shared memory
    cudaFuncSetAttribute(
        warpspec_kernel<bits, c_fp32, cb, shmem_out_had>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);

    warpspec_kernel<bits, c_fp32, cb, shmem_out_had>
        <<<grid, WS_BASE_THREADS, shmem, stream>>>(
            A, B, C, size_m, size_k, size_n, locks, post_scale);
}

// Kernel wrapper
template<int bits, bool c_fp32, int cb, bool shmem_out_had>
__global__ void warpspec_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ locks,
    const half* __restrict__ post_scale
)
{
    warpspec_inner<bits, c_fp32, cb, shmem_out_had>(
        A, B, C, size_m, size_k, size_n, locks, post_scale,
        blockIdx.x, gridDim.x);
}

// ============================================================================
// Python binding (via torch extension)
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor warpspec_gemm
(
    torch::Tensor A,       // [M, K] half
    torch::Tensor B,       // packed trellis [K//16 * blocks_n, ...] uint16
    int size_m,
    int size_k,
    int size_n,
    int bits,
    int cb,
    torch::Tensor locks,    // [blocks_n] int32
    torch::Tensor post_scale  // [N] half or None
)
{
    at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto opts_f = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    auto C = torch::empty({size_m, size_n}, opts_f);

    const half* a_ptr = (const half*)A.data_ptr();
    const uint16_t* b_ptr = (const uint16_t*)B.data_ptr();
    void* c_ptr = C.data_ptr();
    int* lock_ptr = locks.data_ptr<int>();
    const half* ps_ptr = post_scale.numel() > 0 ? (const half*)post_scale.data_ptr() : nullptr;

    // Dispatch based on bits and cb
    if (bits == 6 && cb == 1) {
        launch_warpspec<6, false, 1, false>(a_ptr, b_ptr, c_ptr, size_m, size_k, size_n, lock_ptr, ps_ptr, stream);
    } else if (bits == 5 && cb == 1) {
        launch_warpspec<5, false, 1, false>(a_ptr, b_ptr, c_ptr, size_m, size_k, size_n, lock_ptr, ps_ptr, stream);
    } else if (bits == 4 && cb == 1) {
        launch_warpspec<4, false, 1, false>(a_ptr, b_ptr, c_ptr, size_m, size_k, size_n, lock_ptr, ps_ptr, stream);
    } else {
        TORCH_CHECK(false, "Unsupported bits/cb combination: ", bits, "/", cb);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warpspec_gemm", &warpspec_gemm, "Warp-specialized EXL3 GEMM");
}
