# 13. Learned Pairwise Rotations for EXL3 Trellis Quantisation — Feasibility

**Date:** 2026-08-19  
**Question:** Can ParoQuant-style learned pairwise rotations replace EXL3's fixed Hadamard as the pre-transform for trellis quantisation, and what would it cost to try?

---

## 1. What EXL3 Actually Does

All file:line citations refer to the READ-ONLY container overlay at  
`.../4fc65610.../diff/opt/exllamav3-python/exllamav3/`.

### 1.1 suh and svh: per-channel scale+sign vectors

`suh` and `svh` are **per-channel combined scale-and-sign vectors**, not pure Hadamard sign flips.

- **suh** (input-side): dtype `torch.half`, shape `(in_features,)` — one ±magnitude value per input channel.  
  `exl3.py:54`: `self.suh = suh if suh is not None else self.unpack_bf(su)`
- **svh** (output-side): dtype `torch.half`, shape `(out_features,)` — one ±magnitude value per output channel.  
  `exl3.py:55`: `self.svh = svh if svh is not None else self.unpack_bf(sv)`

Their construction during quantisation (`quantize.py:1016, 918-921, 1120-1121`):

1. `sv` initialised as **random ±1 signs** (`quantize.py:1016`):  
   `sv = (torch.randn(n, device=device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)`
2. `su` initialised from **Hessian-based sign flips** via `finalize_capture_H()` (`quantize.py:1010`).
3. In `regularize()`, both are multiplied by per-channel RMS scales (`quantize.py:918-921`):  
   `su = (su * in_channel_scales / (-codebook_scale) + 1e-10).float()`  
   `sv = (sv * out_channel_scales + 1e-10).float()`
4. Final packing (`quantize.py:1120-1121`):  
   `suh = su.flatten().contiguous().to(dtype=torch.half, copy=True)`  
   `svh = sv.flatten().contiguous().to(dtype=torch.half, copy=True)`

So suh/svh encode: **(Hadamard sign pattern) × (per-channel RMS scale) / (codebook scale)**. They are real-valued (not just ±1), which is why they're stored as `half` rather than packed bitfields like the legacy `su`/`sv` (`int16` bitfields, `exl3.py:37-38`).

### 1.2 Hadamard applied at BOTH quantisation time and runtime

**Quantisation time** — in `regularize()` (`quantize.py:915-922`):
```
weight /= sv                          # divide by output scales+signs
blockwise_preapply_had_r_(weight, had_n)  # right Hadamard: each 128-col block @ H_128/sqrt(128)
weight /= su                          # divide by input scales+signs
blockwise_preapply_had_l_(weight, had_k)  # left Hadamard: H_128/sqrt(128) @ each 128-row block
```
The trellis codes are then computed from this doubly-Hadamard-transformed, scale-normalised weight.

**Runtime** — in `reconstruct_hgemm()` (`exl3.py:151, 167`):
```
ext.had_r_128(x, xh, self.suh, None, 1.0)     # pre-scale input with suh, then H_128/sqrt(128)
y_ = xh @ w                                     # GEMM with reconstructed trellis weight
ext.had_r_128(y_, y_, None, self.svh, 1.0)      # H_128/sqrt(128), then post-scale output with svh
```

The Hadamard is also **fused into the exl3_gemm kernel** for the fast path (batch ≤ 144):
- `exl3_gemm_kernel.cuh:24-27`: applies `suh` + FWHT to input tiles before MMA
- `exl3_gemm_kernel.cuh:65-68, 211-235`: applies FWHT + `svh` to output tiles after MMA
- Scale `0.088388347648f` = 1/√128 (`hadamard.cu:112`, `exl3_gemm_kernel.cuh:26`)

### 1.3 Block size: 128

`quantize.py:14`: `had_k, had_n = 128, 128`

The Hadamard operates on blocks of 128 channels. Both input (k) and output (n) dimensions are divided into 128-element blocks, each transformed independently.

### 1.4 Orthogonal and involutory

The Hadamard matrix H₁₂₈ satisfies H · H = 128 · I, so (H/√128)² = I. It is **orthogonal** (H/√128 · (H/√128)ᵀ = I) and **involutory** (its own inverse).

This is critical: the same kernel `had_r_128` applies both the forward and inverse transform. At quantisation time, `preapply_had_l/r` applies H/√128 to the weight. At runtime, `had_r_128` applies H/√128 to the activations (which is the inverse since H/√128 is its own inverse).

**Implementation** (`hadamard_inner.cuh`): the 128-element FWHT runs in a **single warp** (32 threads). It decomposes into:
1. A 4-element Hadamard (butterfly add/subtract on 4 floats per thread)
2. A 32-element butterfly via `shuffle_had_f4x32` (7 stages of `__shfl_xor_sync` + sign-flip-via-XOR + add)

Total: 7 × 64 = 448 additions per 128-element vector. **No multiplications** — sign flips are implemented as XOR on the sign bit (`hadamard_inner.cuh:56-60`), and the 1/√128 scaling is a single multiply at the end.

### 1.5 Full algebraic identity

Let H_k = block-diagonal(H₁₂₈/√128) on the input dimension, H_n = block-diagonal(H₁₂₈/√128) on the output dimension.

**Stored trellis weight** (quantisation time):  
W' = H_k · diag(1/su) · W · diag(1/sv) · H_n

**Runtime**:  
y = (x · diag(suh)) · H_k · W' · H_n · diag(svh)  
= x · diag(suh/su) · W · diag(svh/sv)  
= x · W  (since suh = su, svh = sv)

The Hadamard's self-inverse property is what makes this work with a single transform kernel.

---

## 2. What ParoQuant Does

Sources: arXiv:2511.10645 (ICLR 2026), repo `github.com/z-lab/paroquant` (MIT), checkpoint `z-lab/Qwen3.8-27B-PARO`.

### 2.1 K=8 Givens rotations: form and storage

ParoQuant applies **K=8 sequential independent rotations** per 128-channel group, plus channel-wise scaling. The full transform (paper Eq. 8):

T(W) = (∏ₜ₌₁ᴷ R(Pₜ, Θₜ)) · diag(α) · W

where each R(Pₜ, Θₜ) is an independent rotation (paper Def. 2): a product of Givens rotations on **mutually disjoint channel pairs** within a 128-element group (up to 64 pairs per rotation).

**Parameter storage** (from `qlinear.py:55-73`):
| Parameter | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `pairs_grouped` | `[K, in_features//2]` | int16 | Pair indices: for each of K rotations, 64 pairs per 128-group |
| `angles_grouped` | `[K, in_features//2]` | float (or half) | Rotation angle θ for each pair |
| `channel_scales` | `[in_features]` | half | Per-channel scaling factor α |
| `mask` | `[K, in_features//2]` | bool | Dummy-pair mask (padding for incomplete groups) |
| `num_rotations` | scalar | — | K = 8 |

For a 5120-input matrix: `pairs_grouped` = [8, 2560] int16, `angles_grouped` = [8, 2560] float, `channel_scales` = [5120] half. Total per matrix: ~8 × 2560 × 2 + 8 × 2560 × 4 + 5120 × 2 ≈ 123 KB.

The published checkpoint (`z-lab/Qwen3.8-27B-PARO`) uses `group_size=128, n_bits=4, K=8` for all language-model linear layers.

### 2.2 Optimisation

**Objective** (paper Eq. 9, `qlinear.py:78-87`): layer-wise output-error minimisation  
L(Q) = ‖Q(D)(X') − D(X)‖  
where D is a decoder block, Q(D) is the quantised version, X is the original input, X' is the output of already-quantised preceding layers.

**Two-stage optimisation** (`qlinear.py:158-179`, `train.py`):
1. **Stage 1**: optimise rotation angles (θ) and channel scales (α) via gradient descent. The weight is pseudo-quantised (apply transform → quantise → inverse transform) and the output error is backpropagated to the angles and scales.
2. **Stage 2**: QAT-like fine-tuning of weights and quantisation parameters (scale s, zero-point z).

**Calibration data** (`train.py`, paper §5): 2048 training samples drawn evenly from WikiText2, C4, and RedPajama; 64 validation samples from Pile. The ablation (paper Table 5) shows 128 samples is "surprisingly strong" — C4 perplexity 7.30 with 128 samples vs 7.27 with 2048.

**Training budget**: 10 epochs per stage with AdamW, fixed hyperparameters. For a 27B model on a single GPU, this is ~1-2 hours per decoder layer (64 layers → ~64-128 hours total). The published checkpoint was trained on Qwen3.8-27B.

### 2.3 Runtime inverse

The inverse of a Givens rotation sequence is obtained by **reversing the order and negating the angles** (paper Eq. 5):

X^(m) = X · G(i₁,j₁,−θ₁) · G(i₂,j₂,−θ₂) · ⋯ · G(iₘ,jₘ,−θₘ)

In code (`qlinear.py:103-110`):
```python
flipped_pairs = torch.flip(self.pairs_grouped, dims=[0])
flipped_angles = torch.flip(self.angles_grouped, dims=[0])
weight = scaled_pairwise_rotation(weight, flipped_pairs, -flipped_angles, None, self.group_size)
```

**CUDA kernel** (`rotation.cu`, `rotation.cuh`): a single fused kernel applies all K=8 rotation stages:
- Grid: `(ceil(seq_len/CTA_M), num_groups)` where CTA_M=4, num_groups = in_features/128
- Block: 64 threads (= 128/2 pairs per group)
- Each thread loads one pair of elements into shared memory (128 × CTA_M halfs = 1 KB)
- Applies K=8 rotation stages sequentially, each stage: `__sincosf(θ)` → `fmaf(c, xi, s*xj)` for both elements
- `__syncthreads()` between stages
- Channel scaling is fused into the load step

Cost per 128-element vector: 8 stages × 64 pairs × (2 mul + 2 add) = 2048 flops. This uses **multiplications** (cos/sin), unlike the Hadamard's addition-only FWHT.

ParoQuant reports **<10% overhead** over AWQ (which has no transform) and **15-30% faster** than QTIP (Hadamard + trellis) on RTX A6000 (paper Table 3).

---

## 3. The Substitutability Question

### 3.1 Can a learned rotation be folded into the trellis codes at quantisation time?

**YES for the forward transform; NO for eliminating runtime cost.**

This is the definitive answer to the fold-vs-runtime question:

The trellis codes encode the **transformed** weight W' = T · W. At runtime, the model must compute y = x · W = (x · T⁻¹) · W'. This requires applying T⁻¹ to the activations x at runtime — there is no way around this for ANY invertible weight transform T, whether Hadamard or learned rotation.

EXL3 already pays this cost: `had_r_128` applies H/√128 (= H⁻¹) to the input and output at runtime. A learned rotation would require a different kernel (Givens rotations instead of FWHT), but the cost structure is identical: a small pre-GEMM transform on the input and a small post-GEMM transform on the output.

**Conclusion: a learned rotation does NOT force additional runtime cost beyond what EXL3 already pays for the Hadamard. The cost is comparable, not additive.**

### 3.2 Runtime cost on our shapes

For a typical Qwen3.8-27B matrix (e.g., gate_proj: 5120 × 17408):

| Transform | Input (5120) | Output (17408) | Total per token | Multiplications? |
|-----------|-------------|----------------|-----------------|------------------|
| Hadamard FWHT | 40 blocks × 448 adds = 17,920 | 136 blocks × 448 adds = 60,928 | **78,848 additions** | No (XOR sign flips) |
| ParoQuant K=8 | 40 blocks × 2048 flops = 81,920 | 136 blocks × 2048 flops = 278,528 | **360,448 flops** | Yes (fmaf) |

The rotation uses ~4.5× more transform flops, but both transform counts are
small beside the dense GEMM's ~178M nominal flops per token. That arithmetic is
not a runtime measurement: quantized-GEMM throughput, register pressure,
occupancy, memory traffic, and fusion feasibility can dominate at small M.
ParoQuant's <10% AWQ-relative result on RTX A6000 bounds its own implementation,
not a fused EXL3 kernel on SM120. A kernel benchmark is required before claiming
that runtime cost preserves the quality gain.

### 3.3 Would the gain be smaller for trellis than for INT4?

**Yes, likely smaller — but not zero.** Here is the reasoning:

**Why the gain is smaller for trellis:**

1. **Trellis quantisation already handles non-uniform distributions.** The EXL3 trellis codebook (`codebook.cuh`, `codebook_scale = 1.24371088`) is a lattice/trellis code that maps 256-element (16×16) tiles to K-bit codes. Unlike INT4 uniform scalar quantisation (16 equally-spaced levels), the trellis codebook has a non-uniform structure that adapts to the weight distribution. The Hadamard's main job for INT4 is to Gaussianise the distribution so uniform quantisation works well; for trellis, this is less critical because the codebook is already distribution-aware.

2. **EXL3 already stacks three adaptation mechanisms:** (a) Hadamard transform, (b) per-channel RMS scaling (su/svh), (c) trellis codebook. ParoQuant's learned rotation replaces only (a); the other two mechanisms remain. The marginal benefit of improving (a) when (b) and (c) already provide substantial adaptation is diminishing.

3. **Direct evidence from the paper.** ParoQuant (learned rotation + INT4 linear) **matches** QTIP (Hadamard + trellis vector quantisation) in accuracy across all tested models (paper Table 5.1). This means:
   - learned_rotation + INT4 ≈ Hadamard + trellis
   - Therefore: learned_rotation + trellis ≥ Hadamard + trellis (since learned ≥ fixed for pre-processing)
   - But the gap learned_rotation + trellis vs Hadamard + trellis is likely **smaller** than the gap learned_rotation + INT4 vs Hadamard + INT4, because trellis already captures much of the benefit that the learned rotation provides for INT4.

**Why the gain is not zero:**

1. **Per-layer adaptivity.** The Hadamard is fixed (or randomly sign-flipped via su/sv). A learned rotation is optimised per-layer using calibration data, adapting to each layer's specific weight distribution. For layers with atypical distributions (e.g., attention projections vs MLP layers), this per-layer adaptation could provide meaningful gains even with trellis.

2. **Outlier channel targeting.** ParoQuant's pair selection targets outlier channels (paper Fig. 2, §3). Even with trellis, outlier channels that fall in the same 128-group can dominate the quantisation error for that group. A learned rotation that specifically rotates outlier channels toward normal channels could reduce per-group error more effectively than a fixed Hadamard.

3. **The gap is empirical, not theoretical.** The trellis codebook is designed for approximately-Gaussian post-rotation weights, and a learned rotation could produce a more Gaussian (or more codebook-friendly) distribution than the fixed Hadamard. The magnitude of this effect is an empirical question that the pilot would answer.

**Bottom line:** The gain for trellis is likely in the range of 5-30% of the gain ParoQuant sees for INT4 (i.e., if ParoQuant improves INT4 perplexity by 0.1, the trellis improvement might be 0.005-0.03). Our trellis-attributable KLD is ~0.0027, so even a 10% reduction would be ~0.00027 — meaningful but modest.

---

## 4. Cost of the Experiment

### 4.1 Files a side-by-side build would touch

**Quantisation-time changes (Python, side-by-side build):**
- `modules/quant/exl3_lib/quantize.py` — replace `regularize()` to apply learned rotation instead of (or in addition to) Hadamard; add rotation parameter storage to `out_tensors`
- `modules/quant/exl3.py` — extend `LinearEXL3` to accept rotation parameters (pairs, angles, scales) alongside or instead of suh/svh
- New file: rotation optimisation script (adapt ParoQuant's `optim/train.py` and `optim/rotation.py` for trellis MSE objective)

**Runtime changes (CUDA, requires exllamav3 kernel modifications):**
- `exllamav3_ext/quant/hadamard.cu` + `hadamard_inner.cuh` — add a `givens_r_128` kernel (or reuse ParoQuant's `rotation.cu` kernel) to replace `had_r_128`
- `exllamav3_ext/quant/exl3_gemm_kernel.cuh` — modify the fused GEMM kernel to apply Givens rotations instead of FWHT in the pre/post-MMA phases (lines 14-27, 48-77, 152-235)
- `exllamav3_ext/quant/exl3_gemm.cu` — pass rotation parameters through the kernel dispatch
- `exllamav3_ext/bindings.cpp` — bind the new kernel
- `exllamav3_ext/libtorch/linear.cpp` — pass rotation parameters through the C++ linear layer

**Note:** A modified exllamav3 MUST be built side-by-side. Do not attempt to modify the container overlay in place.

### 4.2 Rotation search objective

Adapt from ParoQuant's INT4 output error to a **trellis-aware objective**:

**Primary objective (trellis MSE):**  
minimise ‖reconstruct(quantize_tiles(R · diag(α) · W)) − R · diag(α) · W‖²

This directly measures the trellis quantisation error on the transformed weight, using the existing `quantize_tiles()` function (`quantize.py:44-62`) and `ext.reconstruct()`.

**Secondary objective (output error, closer to ParoQuant):**  
minimise ‖X · reconstruct(quantize_tiles(R · diag(α) · W)) · R⁻¹ − X · W‖

This measures the end-to-end output error, which is what ultimately matters for KLD. It requires calibration activations X but is more expensive.

**Recommendation:** Start with the trellis MSE objective for the pilot (no activations needed), then switch to output error if the pilot shows promise.

### 4.3 Data volume

- **Weight-MSE pilot:** no corpus or activations; sample 128 weight tiles per epoch.
- **Output-error follow-up:** 2048 training samples from WikiText2 + C4 +
  RedPajama, with 64 disjoint validation samples from Pile (ParoQuant's setup).

### 4.4 Expected search wall-time

For the full Qwen3.8-27B model (~409 weight matrices across 64 layers):

| Phase | Per-layer time | Total (64 layers) | Notes |
|-------|---------------|-------------------|-------|
| Stage 1 (rotation + scale optimisation) | ~30-60 min | ~32-64 hours | 10 epochs, AdamW, single GPU |
| Stage 2 (weight fine-tuning) | ~30-60 min | ~32-64 hours | Optional; skip for pilot |
| **Total** | ~1-2 hours/layer | **~64-128 hours** | Single GPU; multi-GPU proportional speedup |

With 4 GPUs: ~16-32 hours. The quantisation itself (trellis encoding of the rotated weights) adds ~1-2 hours on top.

### 4.5 Smallest meaningful pilot

**One matrix, one metric, one comparison — no full requant needed.**

| Aspect | Detail |
|--------|--------|
| **Matrix** | Layer 0 `gate_proj`: shape (5120, 17408), the largest MLP matrix |
| **Transforms compared** | (A) Current EXL3: Hadamard H₁₂₈ + per-channel RMS scaling (the existing `regularize()` path) |
| | (B) Learned rotation: K=8 Givens rotations + per-channel scaling, optimised for trellis MSE |
| **Metric** | Per-tile trellis MSE: mean over all 16×16 tiles of ‖quantize_tile(W_tile) − W_tile‖² |
| **Optimisation** | 128 sampled weight tiles per epoch, 10 epochs, AdamW lr=1e-3, objective = trellis MSE on the transformed weight |
| **Implementation** | Pure Python using existing `quantize_tiles()` from `exl3_lib/quantize.py`. No CUDA kernel changes, additional model download, or serving. |
| **Decision criterion** | If learned rotation reduces trellis MSE by >5% vs Hadamard → proceed to full experiment. If <2% → NO-GO. |
| **Estimated time** | ~10 minutes on a single GPU (one matrix, 128 sampled weight tiles per epoch, 10 epochs) |
| **What it tells us** | Whether a learned rotation can reduce trellis quantisation error beyond what the fixed Hadamard + per-channel scaling already achieves — the core uncertainty. |

**Pilot script outline (Python, runs inside the container):**
```python
# 1. Load layer 0 gate_proj weight W (5120, 17408) from the FP16 checkpoint
# 2. Baseline: apply regularize(W) with Hadamard → quantize_tiles → compute MSE
# 3. Initialize K=8 random independent rotation pairs + channel scales
# 4. Optimise: for epoch in range(10):
#      W_rot = apply_rotation(W, pairs, angles, scales)
#      W_q = quantize_tiles(W_rot)  # using existing ext.quantize_tiles
#      loss = mse(W_q, W_rot)
#      loss.backward()  # gradients flow to angles, scales
#      optimizer.step()
# 5. Compare final MSE(Hadamard) vs MSE(learned rotation)
```

This pilot requires GPU access (for `quantize_tiles`) but no kernel changes, no model download, and no serving — it operates on a single weight matrix loaded from the existing checkpoint.

---

## 5. Verdict

### **GO** — with the pilot as the first step.

**Rationale:**

1. **The fold-vs-runtime question is resolved definitively.** A learned rotation CAN be folded into the trellis codes at quantisation time (same as Hadamard), and the runtime cost is comparable (both require a transform on activations/outputs, both can be fused into the GEMM kernel). The learned rotation is NOT involutory, so the inverse kernel differs from the forward (reverse order + negate angles vs. self-inverse FWHT), but this is a minor kernel difference, not a fundamental obstacle.

2. **The runtime cost does not erase gains.** The rotation overhead is <0.2% of GEMM compute. ParoQuant's own benchmarks confirm <10% total overhead on comparable hardware.

3. **The gain is uncertain but potentially meaningful.** The trellis codebook's distribution-awareness means the gain is likely smaller than for INT4, but per-layer adaptivity and outlier targeting could still provide a measurable reduction in trellis MSE. The pilot is the cheapest way to resolve this uncertainty.

4. **The pilot is concrete and cheap.** One matrix, 128 calibration samples, 10 epochs, ~10 minutes, pure Python with existing trellis quantisation functions. No kernel changes, no model download, no serving.

**Blocking considerations for full implementation (beyond the pilot):**

- **NEEDS-UPSTREAM for the runtime kernel.** Replacing the FWHT in `exl3_gemm_kernel.cuh` with Givens rotations is a non-trivial CUDA kernel modification in exllamav3. The quantisation-time changes can be done in a side-by-side Python build, but the fused GEMM kernel change requires modifying exllamav3's CUDA sources — this should be coordinated with the exllamav3 author (turboderp) or done as a fork.

- **The pilot must show >5% MSE reduction** to justify the implementation effort. Given that trellis already handles non-uniform distributions, a <2% reduction would indicate the Hadamard is already near-optimal for trellis and the learned rotation adds complexity without meaningful benefit.

- **If the pilot is positive, the next step** is a 4-layer pilot (all 7 matrices per layer × 4 layers) with the output-error objective and 512 calibration samples, to confirm the gain holds across matrix types (gate/up/down/q/k/v/o_proj) before committing to the full 64-layer requant.

---

## Appendix A: Source Citations

### EXL3 (exllamav3, READ-ONLY overlay)

| Claim | File | Lines |
|-------|------|-------|
| suh/svh are per-channel half tensors | `modules/quant/exl3.py` | 25-26, 41-42, 54-55 |
| suh/svh store combined sign+scale | `modules/quant/exl3_lib/quantize.py` | 918-921, 1120-1121 |
| Hadamard block size = 128 | `modules/quant/exl3_lib/quantize.py` | 14 |
| Hadamard applied at quantisation time | `modules/quant/exl3_lib/quantize.py` | 915-922 |
| Hadamard applied at runtime (reconstruct path) | `modules/quant/exl3.py` | 151, 167 |
| Hadamard fused in GEMM kernel | `exllamav3_ext/quant/exl3_gemm_kernel.cuh` | 14-27, 48-77, 152-235 |
| FWHT scale = 1/√128 | `exllamav3_ext/quant/hadamard.cu` | 112 |
| FWHT implemented via warp shuffles | `exllamav3_ext/quant/hadamard_inner.cuh` | 38-113 |
| Sign flips via XOR (no multiply) | `exllamav3_ext/quant/hadamard_inner.cuh` | 56-60 |
| su/sv legacy packed bitfields | `modules/quant/exl3.py` | 37-38, 131-142 |
| sv initialised as random ±1 | `modules/quant/exl3_lib/quantize.py` | 1016 |
| su initialised from Hessian | `modules/quant/exl3_lib/quantize.py` | 1010 |
| Trellis quantises 16×16 tiles | `modules/quant/exl3_lib/quantize.py` | 44-62 |
| Codebook scale = 1.24371088 | `modules/quant/exl3_lib/quantize.py` | 15 |

### ParoQuant (arXiv:2511.10645, github.com/z-lab/paroquant)

| Claim | Source | Location |
|-------|--------|----------|
| K=8 independent Givens rotations | Paper §4.1.3, Eq. 8 | — |
| Inverse = reverse order + negate angles | Paper Eq. 5 | — |
| Parameter shapes (pairs, angles, scales) | `paroquant/optim/qlinear.py` | 55-73 |
| Layer-wise output error objective | Paper Eq. 9 | — |
| 2048 calibration samples (128 sufficient) | Paper §5, Table 5 | — |
| <10% overhead vs AWQ | Paper Table 3 | — |
| Matches QTIP (Hadamard + trellis) accuracy | Paper Table 5.1 | — |
| CUDA kernel: shared memory, K=8 stages | `paroquant/kernels/cuda/rotation.cu` | 10-37 |
| CTA_M=4, GROUP_SIZE=128, block=64 threads | `paroquant/kernels/cuda/rotation.cu` | 74-80 |
| Channel scaling fused into load | `paroquant/kernels/cuda/rotation.cuh` | 21-40 |
| Group size 128, n_bits 4 | `paroquant/optim/qlinear.py` | 30-31 |
| Published checkpoint: z-lab/Qwen3.8-27B-PARO | HuggingFace | config |

---

## MEASURED OUTCOME (2026-08-19): NO-GO FOR THIS PILOT

The specified pilot ran (`tools/rotation-pilot.py`,
`receipts/rotation-pilot-2026-08-19.md`). In original weight space on layer 0
`mlp.gate_proj`, Hadamard MSE was 1.234e-07, identity 8.902e-05, and learned K=8
Givens 8.955e-05. This rejects the tested fixed-pair, 10-epoch,
128-weight-tile-per-epoch configuration and leaves the FWHT as the production
default.

The 725× one-matrix margin is not a universal proof against other layers,
pair-selection strategies, training budgets, or the activation/output-error
objective proposed above. The learned arm optimises transformed-space STE MSE
while also learning non-orthogonal channel scales; its final metric is inverted
to original space, but the training loss has no scale regularisation and is not
the final metric. That objective mismatch is an additional reason not to
generalise the result. The first run also required a measurement-domain fix
after failing its own sanity check; the receipt preserves both rounds.
