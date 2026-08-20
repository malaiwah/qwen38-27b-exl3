# W4A8 Dense GEMM Kernel Design — e4m3 Activations × e2m1 Weights on SM120

> Additive extension answering upstream feature request **b12x#232**.
> Target: RTX 5090 (SM120, 170 SMs, no tcgen05/TMEM).
> Container b12x commit: `cd3ce19` (READ-ONLY reference).
> Host checkout: `c0a36ce` (known to differ; container is authoritative).

## 1. Background: existing W4A4 NVFP4 dense path

The production dense NVFP4 GEMM lives in `b12x/_lib/dense_gemm.py` (7797 lines).
Key facts mapped end-to-end:

| Aspect | W4A4 NVFP4 (existing) |
|---|---|
| A dtype | `cutlass.Float4E2M1FN` (4-bit, packed 2/B) |
| B dtype | `cutlass.Float4E2M1FN` (4-bit, packed 2/B) |
| Scale dtype | `cutlass.Float8E4M3FN` (E4M3, 8-bit) |
| sf_vec_size | 16 (one scale per 16 K-elements) |
| Global scale | per-tensor fp32, applied in epilogue via `alpha` / `w_gscale` |
| MMA op | `cute.nvgpu.warp.MmaMXF4NVF4Op` |
| MMA shape | m16n8k64 (K=64 per MMA instruction) |
| tile_k | sf_vec_size × 8 = 128 |
| mma_k | 64 (2 MMA per K-tile) |
| Tile pin | `can_implement` requires `tile_m % 64 == 0`, `tile_n % 16 == 0` |
| A quant | `b12x.quantization.nvfp4` BF16→FP4 TMA kernel, or `fused_quant_a` inline |
| B layout | (N, K/2) packed e2m1, K-major; SFB (N, K/16) E4M3 |
| Epilogue alpha | scalar fp32 = `global_scale` (applied as `C = alpha * (A·SFA) @ (B·SFB)`) |
| Smem budget | `_compute_stages` at `dense_gemm.py:4173`, capacity 100 KB/SM (`sm_120`) |

### B payload + scale layout

B arrives as `(N, K/2, 1)` uint8 (packed e2m1 nibbles, K-major) and SFB as
`(N, K/16)` `Float8E4M3FN` (one E4M3 scale per 16 K-elements per N-row).
The per-tensor `global_scale` (fp32) is folded into `alpha` and applied in the
epilogue. The effective weight is:

```
w[n,k] = global_scale × sfb[n, k//16] × e2m1_value(n,k)
```

### A quantisation (fused_quant_a)

`b12x.quantization.nvfp4.plan(m, k)` compiles a BF16→FP4 TMA tile kernel
(`_kernel.py:Bf16ToFp4TmaKernel`) that produces packed e2m1 values
`(M, K/2)` and E4M3 scales in the MMA swizzled layout.
`dense_gemm_fused_quant_a` (line 6913) is the inline alternative for M≤8:
each CTA quantises its A tile directly into smem, avoiding a separate launch.
Our issue **b12x#233** asked to extend `fused_quant_a` to the W4A8 A-quant;
for v1 we quantise A on the host (torch-side) and note the fused path as
future work.

### can_implement tile_m pin

`can_implement` (line 4431) gates `tile_m % 64 == 0` for Float4E2M1FN.
The MXFP8 path (Float8E4M3FN) allows `tile_m` in {16, 32, 64, 128}.
The W4A8 kernel inherits the MXFP8 constraints because it uses the same
`mxf8f6f4` instruction kind.

---

## 2. W4A8 variant: e4m3 × e2m1

### 2.1 MMA instruction

The SM120 `mxf8f6f4` QMMA kind supports mixed-precision operands. The exact
PTX instruction for e4m3(A) × e2m1(B) is:

```ptx
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
```

This is already wrapped in b12x as `mxfp8_mma_m16n8k32_f32_e2m1`
(`b12x/_lib/intrinsics.py:3769`) and is proven in production by the MoE W4A8
kernels (`moe/_shared/kernels/w4a8_phase1.py:510`, `w4a8_phase2.py:357`,
`dynamic.py:1803`).

**DSL enum names:**
- MMA op class: `cute.nvgpu.warp.MmaMXF8Op` (same as MXFP8 and MX-FP6; the
  `mxf8f6f4` kind is selected by the inline asm, not by the CuTe DSL op name).
- A container dtype: `cutlass.Float8E4M3FN` (e4m3 activations in byte containers)
- B container dtype: `cutlass.Float4E2M1FN` packed in bytes, spread to QMMA
  byte containers via `e2m1x8_to_qmma_e2m1x8` (intrinsics.py:4557)
- Scale dtype: `cutlass.Float8E8M0FNU` (UE8M0 — unsigned 8-bit exponent)
- MMA shape: **m16n8k32** (K=32 per instruction, half of W4A4's k64)
- `sf_vec_size`: 32 (one UE8M0 scale per 32 K-elements)

**Delta from W4A4:**
- `mma_k`: 64 → 32 (4 MMA per K128 tile instead of 2)
- A bytes per K-tile: `tile_m × tile_k / 2` → `tile_m × tile_k` (doubles)
- B bytes per K-tile: unchanged (`tile_n × tile_k / 2`, e2m1 packed)
- Scale format: E4M3/sf16 → UE8M0/sf32
- Global scale: folded into UE8M0 scales (no separate alpha for B; A has no
  global scale)

### 2.2 Operand and SF layouts

**A (e4m3 activations):**
- Global: `(M, K)` uint8 (one e4m3 byte per element), K-major (row-major)
- SFA: `(M, K/32)` uint8 (UE8M0), one scale per 32 K-elements per M-row
- Quantisation: per-token, per-32-element block. `amax` → `pow2_ceil_ue8m0` →
  UE8M0 exponent byte; values scaled by `2^(127-byte)` and cast to e4m3.
- v1: torch-side via `pow2_ceil_ue8m0_torch` (intrinsics.py:223) or b12x's
  `quantize_mxfp8_rows_cute` (CuTe DSL kernel, same numerics).
- Future: fuse into the GEMM kernel's A-load prologue (b12x#233).

**B (e2m1 weights — identical payload to NVFP4):**
- Global: `(N, K/2)` uint8 (packed e2m1 nibbles, K-major) — **unchanged from NVFP4**
- SFB: `(N, K/32)` uint8 (UE8M0) — **converted** from NVFP4's `(N, K/16)` E4M3
- The e2m1 packed payload is bit-identical to the NVFP4 checkpoint; only the
  scale plane changes.

**Smem layout (per CTA, per stage):**

| Region | Size (bytes) | Layout |
|---|---|---|
| A payload | `tile_m × tile_k` | `[row][vec×16]` with XOR swizzle `vec ^ (row&7)` |
| A scales (SFA) | `tile_m × 4` | `[row] → u32` (4 UE8M0 bytes packed, one per K32 block) |
| B payload | `tile_n × tile_k / 2` | `[n8×4+warp][lane] → 16 B` (4 u32, one per N8 group) |
| B scales (SFB) | `(tile_n/8) × 8 × 4` | `[n8][q] → u32` (8× replication for bank conflicts) |

The B smem layout and MMA emission follow the proven `w4a8_phase2` MoE kernel
(`moe/_shared/kernels/w4a8_phase2.py`), adapted for dense grid scheduling.

### 2.3 Scale format decision — UE8M0 lesson addressed

**The lesson (context constraint 1):** SM120 DeepGEMM fp8 required UE8M0
scale post-processing — raw fp32 scales produced NaN. The `mxf8f6f4` MMA
instruction **hard-requires** UE8M0 (power-of-2 only) scale factors. Passing
E4M3 or fp32 scales produces silent NaN or garbage.

**Decision:**
- **A scales: UE8M0, sf_vec_size=32.** The per-token e4m3 quantiser computes
  `amax` per 32-element block, rounds up to power-of-2 via
  `pow2_ceil_ue8m0`, and stores the UE8M0 exponent byte. This is exactly
  what `quantize_mxfp8_rows_cute` (b12x/_lib/quant/mxfp8_rows.py) already
  does. The torch replica `pow2_ceil_ue8m0_torch` (intrinsics.py:223) is
  bit-exact with the device intrinsic.

- **B scales: UE8M0, sf_vec_size=32.** The NVFP4 checkpoint stores E4M3
  scales with sf_vec_size=16 and a per-tensor global_scale. We convert:
  1. Effective fp32 scale = `global_scale × e4m3_scale` for each (N, K16) block
  2. Merge pairs of K16 blocks into K32 blocks: take `max` of the two
     effective scales
  3. Merge N8 groups: take `max` of the 8 N-row scales within each N8 group
     (required because the m16n8k32 MMA uses one SFB per N8×K32 block when
     `tid_b=0`)
  4. Round up to power-of-2 via `pow2_ceil_ue8m0_torch` → UE8M0 byte

  Steps 2–3 are lossy (coarsening K granularity 16→32 and N granularity
  1→8). The K-merge is inherent to switching from NVFP4 to MXFP4 scale
  granularity. The N8-merge is inherent to the `mxf8f6f4` MMA's N8 scale
  sharing when `tid_b=0` (the MoE W4A8 path uses the same convention).

  The residual (ratio between the original E4M3 scale and the coarsened
  UE8M0 scale) is **not** folded back into the e2m1 values for v1. This
  is the primary accuracy risk (see risk #3). A future refinement can use
  `e2m1x8_mul_residual_to_e4m3x8` (intrinsics.py:4643) to absorb the
  residual into the B operand, but this converts B from e2m1 to e4m3,
  losing the 4-bit B traffic advantage.

### 2.4 K-tiling delta and smem budget

W4A4: `tile_k = 128, mma_k = 64` → 2 MMA instructions per K-tile.
W4A8: `tile_k = 128, mma_k = 32` → 4 MMA instructions per K-tile.

A bytes per K-tile double (8-bit vs 4-bit): `tile_m × 128` vs `tile_m × 64`.
B bytes per K-tile unchanged: `tile_n × 64` (packed e2m1).

SM120 smem capacity: 100 KB = 102,400 B per SM.
Budget formula (from `_compute_stages`, dense_gemm.py:4212):

```
raw_stages = (smem_capacity - occupancy × 1024 - mbar(1024) - epi_bytes)
             / (ab_bytes_per_stage + sf_bytes_per_stage)
```

#### Tile candidate A: (128, 128, K=128), occupancy=1

| Region | Per-stage bytes |
|---|---|
| A (e4m3) | 128 × 128 = 16,384 |
| B (e2m1 packed) | 128 × 64 = 8,192 |
| SFA (UE8M0) | 128 × 4 = 512 |
| SFB (UE8M0) | 16 × 8 × 4 = 512 |
| **ab+sf per stage** | **25,600** |
| Epilogue (BF16, 1 stage) | 128 × 128 × 2 = 32,768 |
| mbar | 1,024 |

```
raw_stages = (102400 - 1024 - 1024 - 32768) / 25600 = 67584 / 25600 = 2.64
→ 2 stages (capped at 4, but smem limits to 2)
Total = 2 × 25600 + 32768 + 1024 = 84,992 B < 102,400 ✓
```

**Fits with 2 pipeline stages.** ~83% smem utilisation.

#### Tile candidate B: (64, 128, K=128), occupancy=1

| Region | Per-stage bytes |
|---|---|
| A (e4m3) | 64 × 128 = 8,192 |
| B (e2m1 packed) | 128 × 64 = 8,192 |
| SFA (UE8M0) | 64 × 4 = 256 |
| SFB (UE8M0) | 16 × 8 × 4 = 512 |
| **ab+sf per stage** | **17,152** |
| Epilogue (BF16, 1 stage) | 64 × 128 × 2 = 16,384 |

```
raw_stages = (102400 - 1024 - 1024 - 16384) / 17152 = 83968 / 17152 = 4.89
→ 4 stages (capped)
Total = 4 × 17152 + 16384 + 1024 = 86,016 B < 102,400 ✓
```

**Fits with 4 pipeline stages.** Better latency hiding for small-M decode.

#### Tile candidate C: (16, 128, K=128), occupancy=2

| Region | Per-stage bytes |
|---|---|
| A (e4m3) | 16 × 128 = 2,048 |
| B (e2m1 packed) | 128 × 64 = 8,192 |
| SFA (UE8M0) | 16 × 4 = 64 |
| SFB (UE8M0) | 16 × 8 × 4 = 512 |
| **ab+sf per stage** | **10,816** |
| Epilogue (BF16, no smem stage — m=1 direct store) | 0 |

```
raw_stages = (102400 - 2×1024 - 1024 - 0) / 10816 = 99328 / 10816 = 9.18
→ 3 stages (decode_stage3 cap for occupancy≥2, tile_m≤16)
Per-CTA = 3 × 10816 + 1024 = 33,472 B
Two CTAs = 2 × 33472 = 66,944 B < 102,400 ✓
```

**Fits with 2 resident CTAs, 3 stages each.** Candidate for M=1 decode;
performance was not established by the shared-memory calculation.

### 2.5 Tile sizes to try first

| Regime | Tile (M, N, K) | Stages | Occupancy | Notes |
|---|---|---|---|---|
| Large-M prefill | (128, 128, 128) | 2 | 1 | Matches W4A4 prefill tile |
| Small-M batch | (64, 128, 128) | 4 | 1 | 4× pipeline depth hides B latency |
| M=1 decode | (16, 128, 128) | 3 | 2 | Matches MXFP8 decode tile |

The v1 kernel implements (64, 128, 128) with 2 stages as the default,
matching the `w4a8_phase2` MoE kernel's proven configuration. Other tiles
are compile-time configurable.

### 2.6 Scratch sizing

The kernel reuses b12x's `packed_gemm_scratch_elements` contract
(`b12x/_lib/scratch.py`). The scratch is a single contiguous uint8 buffer.
For W4A8 dense, scratch is needed for:
- Split-K partials (if split_k > 1): `split_k × M × N × 4` bytes (fp32)
- A quantisation workspace: `M × K` bytes (e4m3) + `M × K/32` bytes (UE8M0)

The scratch cap follows the same `min(size_n × route_slots, sms × 4 ×
moe_block_size × 256)` formula as b12x#235. For dense (no routing), the
cap is `sms × 4 × tile_m × 256`. With 170 SMs, tile_m=64: ~42.5 MiB
(shape-independent), matching the W4A16 c_tmp contract.

v1 does not use split-K (single-kernel), so scratch is just the A quant
workspace, allocated per-call.

---

## 3. Fallback assessment

The assignment asks us to check whether the mixed `mxf8f6f4` e4m3×e2m1 MMA
is genuinely available in the CuTe DSL version shipped in the image.

**Finding: YES, it is available and proven.**

1. The inline asm wrapper `mxfp8_mma_m16n8k32_f32_e2m1` exists at
   `b12x/_lib/intrinsics.py:3769` and emits the correct PTX.
2. It is called in 5 production kernel files:
   - `moe/_shared/kernels/w4a8_phase1.py:510,531`
   - `moe/_shared/kernels/w4a8_phase2.py:357`
   - `moe/_shared/kernels/dynamic.py:1803,5204,5249,6534`
3. The container image ships `nvidia-cutlass-dsl` ≥ 4.6.0 (required by
   `b12x.gemm.blockscaled.is_supported`), which supports SM120 `sm_120a`.
4. The `mxf8f6f4` instruction kind is the SAME kind used by the MXFP8
   (e4m3×e4m3) and MX-FP6 (e4m3×e2m3) dense paths, which are already
   shipping in the container's `dense_gemm.py`.

**No fallback to e4m3×e4m3 (W8A8) is needed.** The primary path is the
true e4m3×e2m1 mixed-precision MMA, keeping B at 4-bit HBM traffic.

If a future DSL version breaks the inline asm, the fallback would be:
expand B's e2m1 to e4m3 (lossless via `e2m1x8_to_e4m3x8`, intrinsics.py:4594)
and use the existing MXFP8 `dense_gemm` with `ab_dtype="float8_e4m3fn"`.
This doubles B HBM traffic but requires no kernel changes.

---

## 4. Risk list

1. **SFB N8-merge accuracy loss.** The `mxf8f6f4` MMA with `tid_b=0` uses
   one UE8M0 scale per N8×K32 block. Merging 8 N-row scales into one (by
   max) coarsens the B quantisation. For weights with heterogeneous scale
   distributions across N, this may push `max_rel` above 5%.
   *Mitigation:* evaluate on real weights; if needed, use `tid_b` to
   provide per-N scales (requires changes to the MMA call and SFB layout).

2. **SFB K16→K32 merge accuracy loss.** NVFP4 has per-16-element scales;
   MXFP4/mxf8f6f4 requires per-32. Taking the max of two K16 scales
   overestimates by up to 2× for the smaller block.
   *Mitigation:* same as #1; the residual could be absorbed via
   `e2m1x8_mul_residual_to_e4m3x8` but this converts B to e4m3.

3. **B smem layout mismatch.** The simplified row-major B smem layout
   (vs w4a8_phase2's tile-major layout) may have bank conflicts that
   degrade performance. The MMA's B operand mapping assumes a specific
   nibble-to-byte-container spread that must match `e2m1x8_to_qmma_e2m1x8`'s
   expected input ordering.
   *Mitigation:* the MMA emission follows w4a8_phase2 exactly; only the
   global→smem staging changes. If the nibble ordering is wrong, the
   kernel will produce garbage (not NaN) — detectable by selftest.

4. **CuTe DSL compilation on first call.** The kernel JIT-compiles on
   first invocation. Compilation takes 30–120 s on SM120. If the inline
   asm constraints are wrong, the compile will fail with an LLVM error.
   *Mitigation:* the asm template is copied verbatim from the proven
   `mxfp8_mma_m16n8k32_f32_e2m1` intrinsic.

5. **cp.async alignment.** B is `(N, K/2)` uint8; `cp.async4` requires
   16-byte alignment on both source and destination. A packed row stride of
   `K/2` bytes is 16-byte aligned when `K/2 % 16 == 0`, i.e. `K % 32 == 0`.
   All target shapes (K=5120, 17408, 12288, 1024) satisfy this.
   *Mitigation:* validated for all selftest shapes.

6. **A quantisation cost.** v1 quantises A on the host side (torch), not
   fused into the kernel. For M=1 decode, this adds a small kernel launch
   overhead. The fused path (b12x#233) eliminates this but requires
   modifying `dense_gemm_fused_quant_a` to support the e4m3 format.
   *Mitigation:* use b12x's existing `quantize_mxfp8_rows_cute` which is
   already optimised for this; the overhead is < 5 µs per call.

7. **No split-K for small-M large-N.** v1 does not implement split-K.
   For M=1, N=17408, K=5120, the grid has 17408/128 = 136 CTAs on 170 SMs.
   This is CTA-starved (80% occupancy). Split-K would multiply CTAs by
   the split factor.
   *Mitigation:* add split-K in v2; for v1 the (16, 128) decode tile with
   occupancy=2 partially compensates.

8. **Global scale elimination.** NVFP4 uses a per-tensor global scale
   folded into alpha. W4A8 with UE8M0 scales has no global scale — the
   UE8M0 exponents are absolute. The B scale conversion must correctly
   absorb the global_scale into the UE8M0 scales (step 1 of the
   conversion). If the global_scale is very small or very large, the
   UE8M0 exponent range (byte 0–254, i.e. 2^-127 to 2^127) must cover it.
   *Mitigation:* fp32 global_scale is typically O(1); UE8M0 range is
   ~10^-38 to ~10^38. No overflow risk.

---

## 5. Implementation plan

### Module structure

```
/home/mbelleau/b12x-w4a8/
├── __init__.py          # Public API: prepare_b, quantize_a, run, mm
├── _kernel.py           # CuTe DSL kernel (lazy cutlass import)
├── _quant.py            # A quantisation + B scale conversion (torch)
└── selftest.py          # Correctness harness
```

### API surface (mirrors b12x.gemm.blockscaled)

```python
import b12x_w4a8_ext as w4a8

# One-shot (like b12x.gemm.blockscaled.mm)
out = w4a8.mm(a_bf16, b_packed, b_scales_e4m3, global_scale,
              c_dtype="bfloat16")

# Planned (like b12x.quantization.nvfp4.plan/run)
prepared_b = w4a8.prepare_b(b_packed, b_scales_e4m3, global_scale)
a_q, a_sf = w4a8.quantize_a(a_bf16)
out = w4a8.run(a_q, a_sf, prepared_b)
```

### Kernel architecture

The kernel (`_kernel.py:W4A8DenseKernel`) is a CuTe DSL `@cute.jit` class
based on the `w4a8_phase2` MoE kernel, adapted for dense GEMM:

- **Grid:** `(ceil(N/128), ceil(M/64), 1)` — N-tiles in x, M-tiles in y
- **Threads:** 128 (4 warps × 32)
- **Pipeline:** 2-stage cp.async double-buffer
- **MMA:** `mxfp8_mma_m16n8k32_f32_e2m1` with `e2m1x8_to_qmma_e2m1x8` B decode
- **Epilogue:** BF16 store, no alpha (UE8M0 scales are absolute)
- **K-iteration:** loop over `K/128` K-tiles, double-buffered

## Compile/verify plan (8 cycles max)

Budget: at most 8 compile/verify cycles on the serving container (RTX 5090,
SM120). Each cycle is one `python3 selftest.py` GPU run after a kernel edit;
JIT compilation takes 30–120 s on the first call for a given (N, K) shape and
is cached per shape thereafter (`@functools.cache` on `_get_compiled`).

### Most likely first-compile failure mode (cycle 1)

**B-fragment shared load unpacks a 2-tuple into four names.** `_kernel.py:257`
reads:

```python
w0, w1, w2, w3 = intr["ld_shared_v2_u32"](b_base + ...)
```
`ld_shared_v2_u32` returns `Tuple[Uint32, Uint32]` (a 2-tuple), so unpacking
into `w0, w1, w2, w3` raises `ValueError: not enough values to unpack
(expected 4, got 2)` while the CuTe DSL executes the kernel body to build
IR — i.e. the JIT *compile* aborts before any PTX is emitted. The proven
`w4a8_phase2.py:340` uses `ld_shared_v4_u32` (a 4-tuple) at the same site.
`_kernel.py` imports/registers only `ld_shared_v2_u32` (line 49/70), not
`ld_shared_v4_u32`.

*Fix (one cycle, mechanical):* import `ld_shared_v4_u32` from
`b12x._lib.intrinsics`, register it in `_intrinsics`, and replace the call at
line 257 with `intr["ld_shared_v4_u32"](...)`. No numerics change. This is
distinct from the four risk modes reserved for cycles 5–8 below.

### Cycle plan

| Cycle | Goal | Pass criterion |
|---|---|---|
| 1 | Kernel JIT-compiles **and** `(M,N,K)=(64,256,256)` correctness | compiles; `cos ≥ 0.999`, `max_rel ≤ 0.05` vs `compute_reference` |
| 2 | All four real shapes at `M=128` | each of the 4 `REAL_SHAPES` passes the cos/max_rel thresholds |
| 3 | `M ∈ {1, 2051, 3072}` on the real shapes | passes at every M (exercises the M-tiling / decode path) |
| 4 | Performance vs the **557 TFLOP/s** W4A4 reference at `M=2051` | measured effective TFLOP/s reported; target ≥ W4A4 parity class |
| 5 | Reserved — **SF layout mismatch** | see below |
| 6 | Reserved — **tile_m pin** | see below |
| 7 | Reserved — **smem overflow at doubled A bytes** | see below |
| 8 | Reserved — **epilogue alpha format** | see below |

Cycles 5–8 are reserved for the four most-likely *remaining* failure modes
surfaced by the diagnosis. Each is a correctness/performance defect, not a
compile crash; the cycle is spent only if the mode actually triggers.

### Reserved failure modes (cycles 5–8)

**Cycle 5 — SF (scale-factor) layout mismatch.** The SFB staging
(`_stage_ktile`, `_kernel.py:388`) loads one UE8M0 u32 per N8 group from the
*first* N-row of the group (`n_idx = n_start + n8*8`) and replicates it 8×.
`prepare_b` (`_quant.py`) does **not** N8-merge (it stores per-N-row UE8M0),
contradicting design §2.3 step 3 which requires taking the `max` of the 8
N-row scales per N8×K32 block. The `mxf8f6f4` MMA with `tid_b=0` consumes
one SFB per N8×K32 block, so feeding it the first row's scale (instead of the
N8-max) scales 7 of every 8 rows by the wrong factor. Symptom: output is
non-zero and non-NaN but `cos` well below 0.999. Fix: either N8-max-merge in
`prepare_b`, or load all 8 N-row scales in the kernel and pick per the MMA's
`tid_b` convention (matching `w4a8_phase2`'s prepared-B layout).

**Cycle 6 — tile_m pin.** The launch grid is `(n_tiles, 1, 1)` — N-tiles only;
M is a runtime arg guarded by `if row < m` / `if row_lo < m` in the epilogue.
There is **no M-tile loop**: each CTA writes at most `TILE_M=64` rows. For
`M ∈ {2051, 3072}` (cycle 3) only the first 64 rows are computed and the rest
are left as the uninitialised `out` buffer → garbage in rows 64+. The design
text (§5 "Kernel architecture") claims a grid of `ceil(M/64)` M-tiles, but the
code emits none. Fix: add an M-tile dimension to the grid (`(n_tiles,
ceil(M/TILE_M), 1)`) and offset the A/SFA loads and C stores by
`m_tile * TILE_M`, or loop M-tiles inside the CTA.

**Cycle 7 — smem overflow at doubled A bytes.** W4A8 doubles A traffic vs
W4A4 (e4m3 is 1 B/element vs FP4's 0.5 B). At the default tile (64,128,128)
with 2 stages the kernel uses 34,816 B (`SHARED_BYTES`, well under the 100 KB
SM120 budget). The risk appears only when raising `TILE_M` to 128 to address
cycle 6: `A_PAYLOAD_BYTES` doubles to 16,384 B/stage and, combined with the B
and SFB regions across 2 stages, can approach the per-CTA smem ceiling and
reduce occupancy. Fix: recompute `SHARED_BYTES` for the new tile and drop to
1 stage or lower `TILE_K` if it exceeds the budget; verify occupancy with the
`_compute_stages` formula from §2.4.

**Cycle 8 — epilogue alpha format.** W4A8 has **no scalar alpha**: the
per-tensor `global_scale` is absorbed into the UE8M0 SFB exponents in
`prepare_b` (step 1), and the MMA accumulates `SFA × SFB × A × B` directly.
If a fix for cycle 5 re-introduces an alpha (or double-applies
`global_scale`), the output is off by exactly the `global_scale` factor —
`cos` may stay high but `max_rel` is a constant ratio. Fix: confirm the
epilogue stores raw fp32 accumulators cast to BF16 with no multiplier, and
that `global_scale` appears exactly once (in `prepare_b`).

### Cycle economics

Cycle 1 is expected to fail on the `ld_shared_v2_u32` unpack bug (mechanical
fix). Cycles 2–4 then validate compile + correctness + perf in order. The four
reserved modes (5–8) are ranked by likelihood of surfacing during 2–4:
cycle 6 (tile_m pin) will certainly trigger at cycle 3's `M=2051`; cycle 5
(SF layout) will likely trigger at cycle 1's correctness check if the unpack
fix leaves the SFB layout untouched; cycles 7–8 are lower-probability and
only relevant after the tile is enlarged. The 8-cycle budget therefore covers
the unpack fix (1), the three validation gates (2–4), and one pass at each of
the four reserved modes (5–8).
