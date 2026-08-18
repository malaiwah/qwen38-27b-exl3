// SPDX-License-Identifier: Apache-2.0
// Optimized W6 dequant for SM120
//
// Key optimizations vs original dq4:
// 1. Eliminate % 48 modulo by using bit manipulation (48 = 16 * 3, use i0 & 47 with
//    correction, or use a lookup table)
// 2. Use __funnelshift_r instead of 64-bit fshift (1 instruction vs 2-3)
// 3. Precompute all 4 shift amounts to reduce dependency chain

#pragma once

#include "codebook.cuh"

// Optimized fshift using funnelshift — 1 instruction on SM120
__device__ __forceinline__ uint32_t fshift_opt(uint32_t b, uint32_t a, int shift) {
    if (shift < 32) return __funnelshift_r(b, a, shift);
    return a >> (shift - 32);
}

// Precompute trellis buffer size mask for bits=6: 48 uint32s
// Use i0 % 48 = i0 - 48 * (i0 / 48) = i0 - 48 * (i0 * 0xAAAAAAAAAB >> 32) [reciprocal]
// Or: since 48 = 3 * 16, and i0 is always < 96 (2 words max), we can use:
//   idx = i0 < 48 ? i0 : i0 - 48;
// This is a single comparison + subtract (2 ops) vs 3 ops for reciprocal multiply

template <int bits, int cb>
__device__ __forceinline__ void dq4_opt(const uint32_t* ptr, int t_offset, FragB& frag) {
    constexpr int BUFSIZE = bits * 256 / 32;  // 48 for bits=6
    
    int b0 = (t_offset + 257) * bits - 16;
    int b2 = b0 + 3 * bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    
    // Eliminate modulo with conditional subtract (2 ops vs 3 for reciprocal multiply)
    uint32_t a = ptr[i0 < BUFSIZE ? i0 : i0 - BUFSIZE];
    uint32_t b = ptr[i2 < BUFSIZE ? i2 : i2 - BUFSIZE];
    
    // Use funnelshift for 1-instruction shifts
    uint32_t w3 = fshift_opt(b, a, s2) & 0xffff;
    uint32_t w2 = fshift_opt(b, a, s2 + bits) & 0xffff;
    uint32_t w1 = fshift_opt(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift_opt(b, a, s2 + bits * 3) & 0xffff;
    
    frag[0] = decode_3inst_2<cb>(w0, w1);
    frag[1] = decode_3inst_2<cb>(w2, w3);
}

// Dispatch wrapper
template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch_opt(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1) {
    if constexpr (bits == 6) {
        dq4_opt<bits, cb>(ptr, idx, frag0);
        dq4_opt<bits, cb>(ptr, idx + 4, frag1);
    } else if constexpr (bits == 5) {
        dq4_opt<bits, cb>(ptr, idx, frag0);
        dq4_opt<bits, cb>(ptr, idx + 4, frag1);
    } else {
        // Fallback to original for other bit widths
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
}
