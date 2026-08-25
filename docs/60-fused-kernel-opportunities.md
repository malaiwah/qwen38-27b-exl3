# Missed cooperative fused kernel opportunities

**Date:** 2026-08-25. Method: code-level analysis of b12x (PR #243 branch) and
the vLLM EXL3 multi-precision patch, grounded in exact file/line references.

## Background

PR [local-inference-lab/b12x#243](https://github.com/local-inference-lab/b12x/pull/243)
added a cooperative fused kernel (`b12x/gemm/trellis_linear/_k6_mcg_cute.py`)
that combines 5 operations into one `cudaLaunchCooperativeKernel` call:

1. Input H128 rotation: `suh` element-wise multiply + 128-point Hadamard
2. W4A16 GEMM: trellis decode + FP16/BF16 MMA
3. Split-K reduction: cooperative inter-CTA reduction with atomic locks
4. Output H128 rotation: 128-point Hadamard + `svh` element-wise multiply
5. MCG sign-vector multiply: folded into the trellis decode

**Eligibility gate** (`kernel.py:11213`): only `trellis_bits==6 AND
codebook=='mcg' AND dense-unpaired AND M≤16 AND sm_120`. Everything else
goes through the generic 3-kernel path: separate Hadamard → GEMM → Hadamard.

Our model has 128 K5 gate/up shards and ~464 K6 shards (attention + down).
The K6 shards already benefit from the fused kernel (+5.6% C1 decode). The
K5 shards do not.

## Opportunity inventory

### Tier 1: High impact, low complexity

| # | Opportunity | Complexity | Impact | Shards affected |
|---|---|---|---|---|
| O1 | K4/K5 MCG fused decode kernel | Low | High | 128 K5 gate/up + K4 late-layer MLP |
| O12 | K2/K3 MCG fused decode kernel | Low | Medium | K3 QSRT MoE atoms |
| O13 | Vision/MTP/online auto-benefit | Inherited | Medium | Vision + MTP draft (auto from O1-O3) |

### Tier 2: High impact, medium complexity

| # | Opportunity | Complexity | Impact | Shards affected |
|---|---|---|---|---|
| O4 | Prefill reconstruct+fold+GEMM fusion | Medium (partial) | High | All attention prefill (M≥128) |
| O5 | FP4 activation quant fused into GEMM | Medium | Medium | All FP4 layers (throughput profile) |
| O6 | FP6 activation quant fused into GEMM | Medium | Low-Medium | FP6 gate/up layers (balanced profile) |
| O9 | GDN input projection batched GEMM | Low-Medium | Low | 48 linear-attn layers |
| O10 | GDN fused RMSNormGated + out_proj | Medium | Low | 48+16 attention layers |

### Tier 3: Speculative, high complexity

| # | Opportunity | Complexity | Impact | Shards affected |
|---|---|---|---|---|
| O2 | SQG codebook fused decode | Medium | Low | SQG MoE (not in our checkpoint) |
| O3 | mul1 codebook fused kernel | Medium-High | Low | mul1 weights (currently ExLlamaV3 fallback) |
| O7 | Paired QSRT dense decode fused | High | Low | P24/P33 paired (not in our checkpoint) |
| O8 | FP8DG prefill fused rotation+quant+GEMM | High | Medium | FP8DG prefill (experimental) |
| O11 | W4A8 path fuse Hadamard into decode+MMA | High | Low | W4A8 paired QSRT |
| O14 | Prefill split-K cooperative for M>16 | High | Medium | All prefill GEMMs |

## Detailed opportunity descriptions

### O1: K4/K5 MCG fused decode kernel

**Location:** `kernel.py:11213` (`_use_k6_mcg_small` gate requires
`trellis_bits == 6`); generic path at `kernel.py:11460-11560`.

**Current path (3 separate kernels):**
1. `_run_trellis_dense_hadamard128(x_f16, rotated_f16, suh)` — input rotation
2. `_compile_w4a16_gemm_launch(...).compiled(...)` — W4A16 GEMM
3. `_run_trellis_dense_hadamard128(gemm_f16, output, svh)` — output rotation

**Fusion idea:** The `exl3_gemm_kernel` template is parameterized on `kBits`.
Instantiate for `kBits=4` and `kBits=5`. Extend `_use_k6_mcg_small` to admit
`bits ∈ {4, 5, 6}`. Add `_small_m.py` launch wrappers for each bit width.

**Expected benefit:** Our K5 `mlp.gate_proj`/`up_proj` are 6.64 GiB across 128
shards — the largest single category of trellis weights. At decode (M=1-7 with
MTP-6), each K5 shard does 3 kernel launches instead of 1. Eliminating 2
launches + 2 intermediate buffer writes per shard should give a similar ~5%
improvement on the K5 path, adding to the K6 improvement we already measured.

**Complexity:** Low — template instantiation + Python dispatch change. The
CUDA template already accepts `kBits` as a compile-time parameter.

### O4: Prefill reconstruct + Hadamard fold + GEMM fusion

**Location:** `vllm-exl3-multiprecision.py:1397-1411` (`_exl3_gemm`, prefill
path for M≥128).

**Current path (3 operations, 80+ kernel launches):**
1. `ext.reconstruct(weight, trellis, trellis_k, mcg, mul1)` — trellis→FP16 decode
2. `hadamard_fold_weight_chunked(weight, suh, svh)` — 80+ einsum launches
   (K/128 iterations × 2 einsums each, in fp32)
3. `ext.hgemm(x, weight, output)` — cuBLAS FP16 GEMM

The fold alone is 2880 bmm launches per request across 36 attention matrices.

**Fusion idea:**
- (a) Minimum: fuse reconstruct + Hadamard fold into one CUDA kernel
  (eliminates 80 einsum launches, keeps hgemm separate).
- (b) Full: cooperative kernel that streams weight tiles through
  reconstruct→fold→GEMM in one pass.

**Expected benefit:** Eliminates 80+ kernel launches per attention matrix per
prefill request. Estimated 20-30% of prefill GEMM time for attention layers.

**Complexity:** (a) Medium — single CUDA kernel for reconstruct+fold.
(b) High — streaming weight tiles through Hadamard within the GEMM.

### O5: FP4 activation quantization fused into GEMM

**Location:** `/opt/fp4/exl3_fp4_conversion.py:900-980` (`fp4_apply`),
`/opt/fp4/triton_fp4_quant.py`.

**Current path (3-4 separate kernels before GEMM):**
1. `amax = x_bf16.abs().max()` — reduction kernel
2. `a_global_scale = _NVFP4_GS_NUM / amax` — scale computation
3. `triton_fp4_quant(x_bf16, ...)` — quantization kernel (packed FP4 + scales)
4. `dense_gemm(...)` — NVFP4 GEMM

Our nsys profiler traces showed activation-quant support kernels (amax/abs/
copy) at significant cost in the throughput profile.

**Fusion idea:** Fuse activation quantization (amax + per-16-element block-scale
+ FP4 E2M1 rounding + nibble packing + scale swizzle) into the GEMM prologue.
The GEMM already loads activation tiles into shared memory; quantizing
on-the-fly during the A-matrix load avoids a separate pass.

**Expected benefit:** Both decode and prefill. For decode (M=1), the quant
kernel overhead is significant relative to the GEMM. Estimated 10-15% of FP4
GEMM time. The throughput profile (all-FP4) would benefit most.

### O9: GDN input projection batched GEMM

**Location:** `qwen_gdn_linear_attn.py:861-867` (`forward_cuda`).

**Current path:** Two separate GEMMs on the same input `hidden_states`:
1. `self.in_proj_qkvz(hidden_states)` — Q/K/V/Z projection
2. `self.in_proj_ba(hidden_states)` — B/A projection

**Fusion idea:** Concatenate weight matrices at load time, single GEMM, split
output. Standard vLLM QKVParallelLinear pattern.

**Blocker:** Each projection has its own EXL3 Hadamard vectors (suh/svh) and
codebook, so the trellis GEMM cannot trivially batch them. Would need a
custom batched trellis GEMM or fall back to FP4/FP6 where Hadamard is
pre-folded.

### O13: Vision/MTP/online-trellis auto-benefit

**Location:** `vllm-exl3-multiprecision.py:3020-3083` (online trellis),
`vllm-exl3-multiprecision.py:2499-2552` (vision trellis),
`qwen3_5_mtp_patch.py:57-98` (MTP draft).

All three use the same `_b12x_trellis_linear` → `run_trellis256_dense`
dispatch. When O1-O3 extend the fused kernel to other bit widths/codebooks,
these paths automatically benefit. No additional work needed.

The MTP lm_head at MTP=6 streams 7×/step (6.37 GB = 29% of step bytes), so a
fused kernel there has outsized impact on TG throughput.

### O14: Prefill split-K cooperative kernel for M > 16

**Location:** `kernel.py:11213` (gate requires `m ≤ 128`), cooperative kernel
at `_k6_mcg_cute.py` assumes `gridDim.y == 1`.

**Current path:** For M > 128, the generic path uses non-cooperative CuTe DSL
with route-block parallelism. The split-K reduction from the cooperative kernel
is not used.

**Fusion idea:** Cooperative split-K kernel for larger M with M-tile
parallelism + K-tile split-K.

**Blocker:** The cooperative kernel's `gridDim.y == 1` assumption (used for
H128 scale indexing) must be generalized.

## Priority recommendation

1. **O1 (K4/K5 fused)** — biggest bang for buck. 128 K5 shards, low complexity,
   directly extends the proven K6 kernel. Our EDA allocation uses K5 for all
   gate/up layers.
2. **O4 (prefill reconstruct+fold)** — 80+ einsum launches per attention matrix
   is pure overhead. Even the partial fusion (a) is worthwhile.
3. **O5 (FP4 quant+GEMM)** — the throughput profile is all-FP4; fusing the
   activation quant eliminates 3-4 kernels per GEMM across all 400+ shards.
4. **O12 (K2/K3 fused)** — same low-complexity template instantiation as O1,
   smaller impact but trivial to do alongside.
5. **O13 (auto-benefit)** — free once O1 is done.

## Cross-references

- K6 fused kernel PR: [b12x#243](https://github.com/local-inference-lab/b12x/pull/243)
- Integration PR: [qwen38-27b-exl3#1](https://github.com/malaiwah/qwen38-27b-exl3/pull/1)
- nsys profiler findings: `receipts/frontier-2026-08-19.md`
- B12X K5 cure: `receipts/b12x-k5-cured-2026-08-19.md`
- EDA allocation: `receipts/eda-resolve-2026-08-19.md`
- FP4 conversion: `patches/exl3_fp4_conversion.py`
- FP6 conversion: `patches/exl3_fp6_conversion.py`
- GDN patch: `patches/qwen_gdn_linear_attn_patch.py`
- MTP patch: `patches/qwen3_5_mtp_patch.py`
