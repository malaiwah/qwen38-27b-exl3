# ParoQuant (z-lab/Qwen3.8-27B-PARO) vs Our EXL3 K5K6 — Decision-Grade Comparison

**Date:** 2026-08-19  
**Analyst:** ParoAnalyst (documentation/source investigation, no GPU work, no weight downloads)  
**Sources:** HF model card, HF file tree + safetensors header, arXiv 2511.10645 (v2, 14 Feb 2026), GitHub z-lab/paroquant, PyPI paroquant 0.1.16, vLLM plugin source code

---

## 1. What ParoQuant Is

**Acronym:** PARO = **Pairwise Rotation Quantization** (the package/model suffix is "ParoQuant").

**Method class:** Post-training quantization (PTQ) **pre-transform + weight-only INT4 linear quantization**. It is NOT a pruning scheme, NOT a KV method, NOT a fine-tune (though it includes a QAT-like weight fine-tuning stage during optimization). It is a **weight-only** method: activations stay in FP16/BF16 at inference.

**Core idea (from arXiv:2511.10645, §4):** Before quantizing weights to INT4, apply a learned **scaled pairwise rotation** — a series of K=8 **independent Givens rotations** (each rotating one disjoint channel pair within a 128-channel group) combined with **channel-wise scaling** — to suppress outlier channels and narrow the dynamic range within each quantization group. The inverse transform (inverse Givens rotations + inverse scaling) is applied to activations at runtime via a custom fused CUDA kernel, followed by a standard **AWQ-Marlin INT4 GEMM**. The rotation is "independent" (each channel appears in at most one pair per rotation step), enabling full GPU parallelism with no inter-thread synchronization within a block.

**Paper:** arXiv:2511.10645, "ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference", Yesheng Liang, Haisheng Chen, Zihan Zhang, Song Han, Zhijian Liu. **Accepted to ICLR 2026.** Authors are at UC San Diego, NVIDIA (Song Han), and MIT.

**Reference implementation:** https://github.com/z-lab/paroquant (330 stars, 33 forks, 13 issues, MIT license). PyPI package: `paroquant` v0.1.16.

**Who is z-lab:** A research lab associated with Song Han's group (UCSD/NVIDIA/MIT). The GitHub org `z-lab` hosts the ParoQuant code and model collection. The blog is at https://paroquant.z-lab.ai.

---

## 2. Format Facts From Actual Files

### 2.1 config.json (read verbatim)

Source: https://huggingface.co/z-lab/Qwen3.8-27B-PARO/resolve/main/config.json

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "layer_types": ["linear_attention", ..., "full_attention"] (48 linear + 16 full, every 4th),
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "tie_word_embeddings": false,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", ...},
    ...
  },
  "quantization_config": {
    "quant_method": "paroquant",
    "bits": 4,
    "group_size": 128,
    "krot": 8
  },
  "vision_config": { ... }  // VLM: depth 27, hidden 1152, patch 16
}
```

**This is a VLM** (Qwen3_5ForConditionalGeneration, image-text-to-text). The model card states: "The visual components in this checkpoint are stored in original precision, and only the language components are quantized to 4 bits." The `--language-model-only` flag skips VLM components.

### 2.2 Quantization Parameters (from config.json `quantization_config`)

| Parameter | Value |
|---|---|
| **quant_method** | `paroquant` |
| **bits** | 4 (INT4 weight-only) |
| **group_size** | 128 (per-group scales + zero-points along input/channel dim) |
| **krot** | 8 (number of independent Givens rotation steps) |
| **zero_point** | true (asymmetric quantization, AWQ-style — default from plugin `from_config`) |
| **Weight-only?** | **Yes.** Activations stay FP16/BF16. Paper §2.1: "we focus on weight-only PTQ." |
| **Per-channel vs per-group?** | **Per-group** (group_size=128 along input dimension). Channel-wise scaling is per-channel but is a pre-transform, not the quantization granularity. |
| **Which GEMM kernel?** | **AWQ-Marlin** (vLLM built-in). The `ParoQuantLinearMethod` inherits from `AWQMarlinLinearMethod`. |

### 2.3 Safetensors Tensor Inventory (from safetensors header, 414,824 bytes)

Source: https://huggingface.co/z-lab/Qwen3.8-27B-PARO/resolve/main/model.safetensors (single file, 18,773,962,320 bytes)

**Per quantized linear layer** (e.g., `model.language_model.layers.0.mlp.gate_proj`):
- `qweight` — I32, shape `[input_dim, output_dim/8]` — packed INT4 weights (8 values per I32)
- `qzeros` — I32, shape `[num_groups, output_dim/8]` — packed INT4 zero-points
- `scales` — F16, shape `[num_groups, output_dim]` — per-group quantization scales
- `theta` — F16, shape `[8, input_dim/2]` — Givens rotation angles (krot=8)
- `pairs` — I16, shape `[8, input_dim]` — channel pair indices for rotations
- `channel_scales` — F16, shape `[1, input_dim]` — channel-wise scaling factors

**Unquantized tensors** (stored as F16):
- `lm_head.weight` — F16, [248320, 5120] — **NOT quantized** (2.368 GiB)
- `model.language_model.embed_tokens.weight` — F16, [248320, 5120] — **NOT quantized** (2.368 GiB)
- `input_layernorm.weight`, `post_attention_layernorm.weight` — F16, [5120]
- `linear_attn.A_log`, `linear_attn.dt_bias` — F16, [48]
- `linear_attn.conv1d.weight` — F16, [10240, 1, 4]
- `linear_attn.in_proj_a.weight`, `in_proj_b.weight` — F16, [48, 5120] — **NOT quantized** (small projections)
- `linear_attn.norm.weight` — F16, [128]

**Modules excluded from quantization:** lm_head, embed_tokens, layernorms, Mamba/linear-attention internal parameters (A_log, dt_bias, conv1d, in_proj_a, in_proj_b, norm). All major linear layers (gate_proj, up_proj, down_proj, qkv, z, out_proj) ARE quantized.

### 2.4 File Size Breakdown

| Component | Tensors | Size (GiB) |
|---|---|---|
| Vision model (F16, unquantized) | 333 | 0.858 |
| Language model — quantized (I32 qweight/qzeros, I16 pairs) | — | 11.460 |
| Language model — full-precision (F16: weights + scales + theta + channel_scales) | — | 5.166 |
| **Language model total** | 2,851 | **16.626** |
| **Total file** | 3,184 | **17.485** |

### 2.5 Resident Weight Size vs Our 31.40 GiB Budget

**With `--language-model-only` (recommended, skips VLM):**

| Component | Size (GiB) |
|---|---|
| Language model weights (INT4 + F16 unquantized + rotation params) | ~16.6 |
| KV cache for 238,400 tokens (our stated need) | 8.89 |
| Activations + CUDA graphs + non-torch overhead | ~2.8 |
| **Total** | **~28.3** |
| **Remaining headroom** | **~3.1 GiB** |

**Verdict: YES, it fits in 31.40 GiB with ~3.1 GiB spare** when using `--language-model-only`.

If the VLM components were loaded too (full 17.485 GiB), total would be ~29.2 GiB — still fits with ~2.2 GiB spare, but tighter and unnecessary for text-only serving.

The rotation parameters (theta, pairs, channel_scales) add ~70 MB total across all 64 layers — negligible.

Marlin repacking does not change the total weight memory (still 4 bits per weight, different layout). Marlin pads N to 64-tile boundaries, but our dimensions (5120, 17408, 6144, etc.) are already multiples of 64.

---

## 3. Claimed Quality — With Protocol Forensics

### 3.1 What the MODEL CARD claims for Qwen3.8-27B-PARO

**The model card reports NO quality metrics for this specific checkpoint.** It contains only installation/usage instructions and a citation. No PPL, no KLD, no benchmark scores, no accuracy numbers. The card links to the paper (arXiv:2511.10645) and the blog (https://paroquant.z-lab.ai) for "more information."

**All quality claims must be sourced from the paper, which evaluates DIFFERENT models.**

### 3.2 What the PAPER reports (arXiv:2511.10645, §5)

The paper evaluates ParoQuant on: LLaMA-2-7B, LLaMA-3-8B/70B, LLaMA-3.1-8B-Instruct, DeepSeek-R1-distilled-LLaMA-3.1-8B, Qwen3-1.7B/4B/8B/14B. **No Qwen3.8-27B results are reported.**

#### Perplexity (Table in §5.1)

| Metric | Protocol | Qwen3-8B ParoQuant | Qwen3-8B FP16 | Qwen3-8B AWQ |
|---|---|---|---|---|
| WikiText2 PPL | context=8192 | 6.29 | 6.24 | 6.45 |
| C4 PPL | context=8192 | 7.04 | 6.97 | 7.14 |

#### Reasoning Accuracy (Table 1, §5.1)

| Metric | Protocol | Qwen3-8B ParoQuant | Qwen3-8B FP16 | Qwen3-8B AWQ |
|---|---|---|---|---|
| MMLU-Pro | 12k samples, zero-shot | 74.1 | 74.6 | 73.5 |
| GPQA Diamond | 198 samples | 57.7 | 60.3 | 60.2 |
| AIME-24 | 30 samples | 75.6 | 75.6 | 72.2 |
| AIME-25 | 30 samples | 63.3 | 72.2 | 61.1 |

#### Non-Reasoning Accuracy (Table 2, §5.1)

| Metric | Protocol | Qwen3-8B ParoQuant | Qwen3-8B FP16 |
|---|---|---|---|
| BoolQ / ARC-C / ARC-E / HellaSwag (avg) | zero-shot | 69.9 (avg) | 70.1 (avg) |

#### Decoding Throughput (Table 3, §5.2)

| Metric | Protocol | Qwen3-8B ParoQuant | Qwen3-8B AWQ |
|---|---|---|---|
| Decode tok/s | RTX A6000, batch=1, Transformers lib | 112 | 120 |

### 3.3 Protocol Incompatibility — Critical Flags

**EVERY number above is from the PAPER's protocol, on DIFFERENT models, and is NOT comparable to our v5 suite.**

| Dimension | Their Protocol | Our v5 Suite | Comparable? |
|---|---|---|---|
| **Model** | Qwen3-8B (paper) — NOT Qwen3.8-27B | Qwen3.8-27B (same base as the PARO checkpoint) | **NO** — different model size |
| **Metric** | PPL (WikiText2, C4) + accuracy (MMLU-Pro, GPQA, AIME, BoolQ, ARC, HellaSwag) | Mean KLD vs BF16 teacher | **NO** — entirely different metric |
| **Reference model** | FP16 of the same model | BF16 teacher (Qwen3.8-27B) | Different — PPL is absolute; KLD is relative to teacher |
| **Sample size** | PPL: full test set; MMLU-Pro: 12k; GPQA: 198; AIME: 30 | 5,120 contexts × 2,047 positions = 10,480,640 scored positions | **NO** — vastly different sample sizes |
| **Dataset** | WikiText2, C4, MMLU-Pro, GPQA, AIME-24/25, BoolQ, ARC, HellaSwag | Our internal context corpus | **NO** |
| **Harness** | lm-evaluation-harness, lighteval (custom vLLM fork) | Our v5 suite | **NO** |
| **Context length** | PPL: 8192 (Qwen3) | Up to 238,400 tokens | **NO** |
| **Throughput hardware** | RTX A6000 (SM86, 48GB) | RTX 5090 (SM120, 32GB) | **NO** — different GPU |

**To make any ParoQuant number comparable to our v5 suite, we would need to:**
1. Run our v5 KLD suite (5,120 contexts × 2,047 positions, BF16 teacher) on the Qwen3.8-27B-PARO checkpoint
2. Serve it on our RTX 5090 with our stack and measure prefill/decode throughput
3. No card-reported number can be directly compared — all are UNVERIFIABLE for our use case

**The paper's PPL and accuracy numbers for Qwen3-8B suggest ParoQuant is competitive with QTIP and better than AWQ on linear INT4.** But this is a statement about the METHOD on a SMALLER model, not about THIS checkpoint's KLD on our scale.

---

## 4. Runnability on Our Stack

### 4.1 Custom Kernel Requirements

**YES — two kernel components are required:**

1. **ParoQuant rotation kernel** (`paroquant/kernels/cuda/rotation.cu`): A custom CUDA kernel that applies the inverse Givens rotations + channel-wise scaling to activations. Uses shared memory, thread-level operations, no tensor cores, no SM-specific instructions. Compiled via PyTorch's `torch.utils.cpp_extension` (JIT). Supports FP16, BF16, FP32. Template parameters: `KROT=1/8`, `GROUP_SIZE=64/128`, `CTA_M=4`. **Architecture-agnostic — should compile and run on any CUDA GPU including SM120.**

2. **AWQ-Marlin INT4 GEMM** (vLLM built-in): The `ParoQuantLinearMethod` inherits from `AWQMarlinLinearMethod` and uses vLLM's standard `apply_awq_marlin_linear` for the matmul. The rotation is applied to activations first, then the Marlin kernel does the INT4×FP16 GEMM.

### 4.2 vLLM Support

| Aspect | Detail |
|---|---|
| **vLLM version** | `>=0.19.1, <0.20` (pyproject.toml). Card recommends `vllm==0.19.1` for CUDA 13.0. |
| **Integration mechanism** | vLLM general plugin via entry point: `paroquant.inference.backends.vllm:register`. Auto-registers `ParoQuantConfig` as quantization method `"paroquant"`. |
| **get_min_capability()** | Returns **75** (SM75/Turing+). Does NOT exclude SM120. |
| **CUDA version** | 12.9 or 13.0 supported (separate install paths). |
| **Docker image** | `ghcr.io/z-lab/paroquant:serve` (API server), `ghcr.io/z-lab/paroquant:chat` (interactive). |

### 4.3 SM120/Blackwell-Consumer Support

**Evidence that SM120 is supported (or at least not excluded):**

1. **`get_min_capability()` returns 75** — any SM ≥ 75 passes the gate, including SM120 (capability 120).
2. **The rotation kernel uses no SM-specific instructions** — only shared memory, thread-level multiply-add, and `__syncthreads()`. Architecture-agnostic.
3. **AWQ-Marlin in vLLM 0.19.1 with CUDA 13.0** — vLLM 0.19.x targets CUDA 13.x which includes Blackwell support. Marlin kernels in recent vLLM are compiled for SM80, SM89, SM90, and SM100/SM120.
4. **The card offers CUDA 13.0 install instructions** — CUDA 13.0 added SM120 (Blackwell) support.
5. **No architecture gate, no Hopper-only instructions, no tcgen05** — nothing in the code or config excludes SM120.

**Evidence AGAINST or unknown:**
- No explicit SM120 testing is documented in the paper, card, or repo.
- The paper's efficiency benchmarks use an RTX A6000 (SM86), not a Blackwell GPU.
- Marline kernel SM120 compilation in vLLM 0.19.1 is not explicitly confirmed in their docs.

**Verdict: LIKELY RUNNABLE on SM120.** The rotation kernel is trivially SM-agnostic. The Marlin kernel is the only risk — but vLLM 0.19.1 with CUDA 13.0 should include SM120 Marlin support. Would need empirical confirmation.

### 4.4 Compatibility With Our r34 Image

Our stack: vLLM "Gilded Gnosis" r34 fork (vLLM 4d006a4 + b12x cd3ce19 + FlashInfer 1ac6942), CUDA 13.2, PyTorch 2.12.

| Factor | Status |
|---|---|
| **vLLM version** | Our r34 fork is NOT vLLM 0.19.1. The plugin requires `vllm>=0.19.1,<0.20`. Our fork is based on a different commit (4d006a4). **LIKELY INCOMPATIBLE** — the plugin imports specific vLLM APIs (`AWQMarlinLinearMethod`, `scalar_types`, `PackedvLLMParameter`, `GroupQuantScaleParameter`, `get_safetensors_params_metadata`) that may have different signatures in our fork. |
| **CUDA version** | Our CUDA 13.2 vs their CUDA 13.0 — minor version difference, should be compatible for kernel compilation. |
| **PyTorch** | Our 2.12 vs their implied ≥2.8 — should be compatible. |
| **Plugin registration** | The plugin uses vLLM's `general_plugins` entry point mechanism. If our r34 fork supports this mechanism, the plugin would auto-register. Uncertain. |

**Verdict: Our r34 image CANNOT serve it as-is.** Three options:
- **(a)** Use the official `ghcr.io/z-lab/paroquant:serve` Docker image (cleanest, but different vLLM version — no b12x, no our EXL3 kernels)
- **(b)** Install `paroquant[vllm]` with `vllm==0.19.1` in a separate environment (requires CUDA 13.0 wheels)
- **(c)** Port the ParoQuant plugin to our r34 fork's API (moderate effort — the plugin is ~313 lines of Python, but depends on AWQ-Marlin internals)

---

## 5. Context Length

| Parameter | Value | Source |
|---|---|---|
| `max_position_embeddings` | 262,144 (256K) | config.json |
| `rope_theta` | 10,000,000 | config.json |
| `rope_type` | `"default"` (no scaling) | config.json |
| `mrope_interleaved` | true | config.json |
| `partial_rotary_factor` | 0.25 | config.json |

**Can it serve 238,400 tokens? YES** — 238,400 < 262,144. No rope scaling needed. Our KV cache requirement of 8.89 GiB at 238,400 tokens is within the memory budget (see §2.5).

Note: this model uses MRoPE (Multimodal RoPE) with interleaved sections [11, 11, 10], which is designed for VLM. For text-only serving, the rope sections would need to be handled correctly by the serving framework.

---

## 6. Orthogonality With EXL3 Trellis Weights

### 6.1 What ParoQuant Is (Transform vs Format)

ParoQuant is a **pre-quantization transform** (rotation + scaling) combined with **standard INT4 linear quantization** (AWQ-style). It is NOT a storage format like EXL3 trellis.

The pipeline is:
1. **Offline:** Apply learned rotation+scaling to BF16 weights → quantize transformed weights to INT4 (linear, group=128) → store qweight + scales + zeros + rotation params
2. **Runtime:** Apply inverse rotation+scaling to activations (custom CUDA kernel) → AWQ-Marlin INT4×FP16 GEMM

### 6.2 Is It Combinable With EXL3 Trellis?

**In principle: YES, the rotation+scaling transform is orthogonal to the quantization format.** The transform operates on the weight matrix before quantization and on activations before the GEMM. It does not prescribe the quantization method. You could:

1. Apply ParoQuant's learned pairwise rotations + channel-wise scaling to BF16 weights
2. Quantize the transformed weights using EXL3 trellis (K5/K6) instead of INT4 linear
3. At runtime: apply inverse rotation to activations → EXL3 trellis GEMM

**However, there are important caveats:**

| Factor | Assessment |
|---|---|
| **EXL3 already has a Hadamard pre-transform** | EXL3 trellis uses incoherence processing (Hadamard transform) similar to QTIP. ParoQuant's paper shows learned rotations outperform fixed Hadamard. **Replacing EXL3's Hadamard with ParoQuant's learned rotations could improve KLD.** |
| **Trellis vs linear quantization** | ParoQuant was designed for linear INT4. EXL3 uses trellis (vector quantization with codebooks). The outlier suppression benefit may be smaller for trellis, which already handles non-uniform distributions better than linear quant. |
| **Rotation optimization target** | ParoQuant optimizes rotation parameters to minimize linear-quantization-induced output error. For EXL3, the optimization would need to target trellis-quantization-induced KLD — a different objective function. The optimization code would need adaptation. |
| **Runtime overhead** | The inverse rotation adds ~10% overhead to each linear layer (per paper). This is on TOP of EXL3's dequantization cost. For our throughput profile, this may be acceptable (our throughput profile has 250K context headroom). |
| **Group size compatibility** | ParoQuant uses group_size=128. EXL3 trellis uses different group structures (trellis codes). The rotation operates per-128-channel group, which may or may not align with trellis group boundaries. |

### 6.3 The Most Valuable Finding

**ParoQuant's learned pairwise rotation could replace EXL3's fixed Hadamard transform as the pre-quantization incoherence step.** The paper demonstrates that learned rotations outperform fixed Hadamard transforms (they match QTIP, which uses Hadamard+trellis, while being 25% faster). If this advantage transfers to EXL3's trellis quantizer, it could reduce our KLD below 0.002700.

**This is option (c) in the recommendation.** The transform is orthogonal to the storage format and worth trialling as a pre-step.

---

## 7. Pros/Cons Table vs Our Three Profiles

Our three serving profiles (from context):
- **throughput:** prefill 9,639 | decode fox 187 / essay 94 | ctx 250K | weights 15.88 GiB | KLD 0.063759
- **balanced:** prefill 3,266 | decode fox 203 / essay 96 | ctx 199K | weights ~21.2 GiB | KLD 0.005672
- **fidelity:** prefill 1,966 | decode fox 208 / essay 93 | ctx 238K | weights 18.83 GiB | KLD 0.003437

| Dimension | Our EXL3 K5K6 (throughput) | Our EXL3 K5K6 (balanced) | Our EXL3 K5K6 (fidelity) | ParoQuant Qwen3.8-27B-PARO |
|---|---|---|---|---|
| **Resident weight size** | 15.88 GiB | ~21.2 GiB | 18.83 GiB | ~16.6 GiB (lang-only) / 17.5 GiB (full) |
| **Fits 31.40 GiB + 238K ctx?** | Yes (250K ctx) | Yes (199K ctx) | Yes (238K ctx) | **Yes** — 16.6 + 8.89 + 2.8 = 28.3 GiB, 3.1 GiB spare |
| **Quantization format** | EXL3 trellis K5/K6 (vector quant) | EXL3 trellis K5/K6 | EXL3 trellis K5/K6 | INT4 linear (AWQ-style) + learned rotation |
| **Bit-width** | Mixed K5 (5-bit) / K6 (6-bit) | Mixed K5/K6 | Mixed K5/K6 | Uniform 4-bit |
| **Weight-only?** | Yes (W4A16/W6A16) | Yes | Yes | **Yes** (W4A16) |
| **Fidelity (our KLD v5)** | 0.063759 (measured) | 0.005672 (measured) | 0.003437 (measured) | **UNKNOWN** — no KLD measured. Paper PPL on Qwen3-8B: 6.29 (vs 6.24 FP16). Card reports NO metrics for this checkpoint. **UNVERIFIABLE.** |
| **Fidelity (their protocol)** | N/A | N/A | N/A | PPL and accuracy only; no KLD. Different model (Qwen3-8B, not 27B). **PROTOCOL INCOMPARABLE.** |
| **Prefill throughput** | 9,639 tok/s (measured) | 3,266 tok/s | 1,966 tok/s | **UNKNOWN** — paper measures decode only (RTX A6000, batch=1). Paper says ~10% slower than AWQ. No prefill numbers. |
| **Decode throughput** | fox 187 / essay 94 tok/s | fox 203 / essay 96 | fox 208 / essay 93 | **UNKNOWN for 27B.** Paper: Qwen3-8B decode 112 tok/s on A6000 (vs AWQ 120, FP16 45). |
| **Context length** | 250K | 199K | 238K | 262K (max_position_embeddings) |
| **MTP support** | MTP-3/6 speculative decoding | Yes | Yes | Model has `mtp_num_hidden_layers=1` — MTP architecture present. Whether vLLM 0.19.1 serves MTP for this model is unconfirmed. |
| **Custom kernels** | EXL3 trellis kernels (b12x, exllamav3) | Same | Same | Custom rotation CUDA kernel + AWQ-Marlin (vLLM built-in) |
| **SM120 support** | Yes (our stack is SM120) | Yes | Yes | **Likely yes** — min_capability=75, no SM-specific instructions. Marlin SM120 support in vLLM 0.19.1 unconfirmed but probable. |
| **Serving on our r34 image** | As-is | As-is | As-is | **Not as-is** — requires vLLM 0.19.1; our r34 fork has different API. Need separate env or port. |
| **License** | Our checkpoint: (check) | Same | Same | Apache 2.0 (model), MIT (code) |
| **Maturity** | Our work (measured, documented) | Same | Same | Paper: ICLR 2026 (accepted). GitHub: 330★, 33 forks, 13 issues. PyPI: v0.1.16. HF downloads: 10. Likes: 2. Very new (Nov 2025 paper, recent HF upload). |
| **Maintenance signals** | Our team | Same | Same | Active development (main branch: "reproducibility not guaranteed, use legacy branch"). 13 open issues. Low download count (10). |
| **VLM?** | No (language model only) | No | No | **Yes** — Qwen3.5 VLM architecture. Visual components in F16. Use `--language-model-only` to skip. |

---

## 8. Recommendation

### **(c) Method is orthogonal and worth trialling as a pre-step to our own quantisation.**

**Rationale:**

1. **The transform is orthogonal to our format.** ParoQuant's learned pairwise Givens rotations + channel-wise scaling is a pre-quantization weight transform, not a storage format. It could in principle replace EXL3's fixed Hadamard pre-transform, potentially improving our KLD below 0.002700. The paper shows learned rotations outperform fixed Hadamard transforms (matching QTIP's accuracy while being 25% faster).

2. **The checkpoint itself is NOT directly comparable** — it uses INT4 linear quantization (AWQ-Marlin), not EXL3 trellis. Running our v5 KLD suite on it would measure a different quantization format (INT4 linear vs trellis), not an apples-to-apples comparison of the pre-transform. The card reports no metrics for this checkpoint, and the paper's metrics are on smaller models with a different protocol (PPL/accuracy, not KLD).

3. **Running it on our hardware is possible but requires a separate environment.** Our r34 fork cannot serve it as-is (different vLLM version). The official Docker image or a vLLM 0.19.1 environment would work, but that's a different serving stack from ours — throughput numbers would not be comparable.

4. **The highest-value experiment** would be: (i) take ParoQuant's optimization code (from `experiments/optimize/4bit.sh` and `paroquant/cli/optimize.py`), (ii) adapt it to optimize rotation parameters against EXL3 trellis quantization error instead of INT4 linear quantization error, (iii) apply the learned rotations to Qwen3.8-27B weights before our EXL3 K5K6 quantization, (iv) measure KLD with our v5 suite. This is option (c) — trialling the method as a pre-step.

### If Option (a) Were Desired (Add to Comparator Set)

To run our v5 KLD suite on the PARO checkpoint:
1. Set up a vLLM 0.19.1 environment with `paroquant[vllm]` (CUDA 13.0)
2. Serve: `vllm serve z-lab/Qwen3.8-27B-PARO --language-model-only`
3. Run our v5 fidelity suite (5,120 contexts × 2,047 positions, BF16 teacher)
4. Resource cost: one RTX 5090, ~16.6 GiB weights, KV for 238K tokens, ~2-3 hours GPU time for the KLD suite
5. **But this measures INT4-linear+rotation vs EXL3-trellis, a format comparison, not a transform comparison.**

### Blocking Reasons for Option (b) (Not Runnable)

None — it IS likely runnable on our hardware. But it is NOT runnable on our r34 image without a separate environment. The method is not blocked, but direct serving on our current stack is.

---

## Appendix: Source URLs

| Source | URL |
|---|---|
| Model card | https://huggingface.co/z-lab/Qwen3.8-27B-PARO |
| config.json | https://huggingface.co/z-lab/Qwen3.8-27B-PARO/resolve/main/config.json |
| generation_config.json | https://huggingface.co/z-lab/Qwen3.8-27B-PARO/resolve/main/generation_config.json |
| File tree | https://huggingface.co/api/models/z-lab/Qwen3.8-27B-PARO/tree/main |
| model.safetensors | https://huggingface.co/z-lab/Qwen3.8-27B-PARO/resolve/main/model.safetensors (18,773,962,320 bytes) |
| Paper (arXiv) | https://arxiv.org/abs/2511.10645 |
| Paper (HTML) | https://arxiv.org/html/2511.10645 |
| GitHub repo | https://github.com/z-lab/paroquant |
| vLLM plugin source | https://raw.githubusercontent.com/z-lab/paroquant/main/paroquant/inference/backends/vllm/plugin.py |
| Rotation CUDA kernel | https://raw.githubusercontent.com/z-lab/paroquant/main/paroquant/kernels/cuda/rotation.cu |
| pyproject.toml | https://raw.githubusercontent.com/z-lab/paroquant/main/pyproject.toml |
| PyPI | https://pypi.org/project/paroquant/ |
| Blog | https://paroquant.z-lab.ai |
| HF collection | https://huggingface.co/collections/z-lab/paroquant |
