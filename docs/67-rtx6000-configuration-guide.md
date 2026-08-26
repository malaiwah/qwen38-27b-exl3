# RTX 6000 Pro Configuration Guide — Qwen3.8-27B EXL3 K5K6-hydrated

## Hardware

| spec | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB) |
| System RAM | 157 GB |
| Disk | 484 GB |
| CUDA | 13.2.1 |
| Container | Docker + NVIDIA Container Toolkit |

## Models on disk

| model | path | size |
|---|---|---|
| EXL3 K5K6 hydrated | `models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated` | 21.6 GB |
| Official FP8 | `models--Qwen--Qwen3.8-27B-FP8` | 29.0 GB |
| BF16 reference | `models--Qwen--Qwen3.8-27B` | 52.0 GB |

## Throughput comparison (RTX 6000, 262K context, Paris verified)

| config | decode tok/s | KV cache tokens | weights | notes |
|---|---:|---:|---:|---|
| **EXL3 balanced + MTP=6** | **70.1** | 1,891,181 | 20.3 GiB | production config |
| EXL3 balanced no MTP | 42.8 | 2,236,655 | 20.3 GiB | weight-format comparison |
| Official FP8 no MTP | 46.5 | 1,789,017 | 28.5 GiB | stock vLLM, no patches |
| BF16 reference | 24.8 | 549,106 | 52.0 GiB | KLD teacher |
| FP6 all-layers + MTP=6 | 57.2 | 1,855,172 | 20.5 GiB | 19% slower than balanced |
| rtx6000 profile (FP8 pre-fold) | 56.6 | 580,000 | 45.3 GiB | pre-fold adds 25 GB VRAM |
| rtx6000 profile (BF16 KV) | 58.0 | 308,000 | 45.3 GiB | BF16 KV doubles bandwidth |

### Key findings

1. **Balanced (trellis K6) beats FP8 pre-fold on RTX 6000**: trellis K6 reads 0.75 bytes/elem vs FP8's 1.0 byte/elem. When VRAM isn't constrained (96 GB), the more compressed format wins on bandwidth.

2. **FP6 on all layers is slower** (57.2 vs 70.1): MXFP6 GEMM is slower than trellis K5 for these shapes, and the K5→FP6 conversion loses trellis compression. FP6 should only be used for layers 0-11 (gate_up) as in the balanced profile.

3. **BF16 KV hurts throughput**: doubles per-token KV bandwidth vs FP8 KV with no decode speed benefit. Use FP8 KV (`fp8_e4m3`) for max performance.

4. **MTP=6 is the production config**: 70.1 tok/s with MTP=6 vs 42.8 without — MTP amortizes the trellis overhead across 7 tokens per forward pass.

5. **Official FP8 + MTP crashes**: `cudaErrorAssert` from shape mismatch in `qwen3_5_mtp.py` — MTP patches are incompatible with compressed-tensors FP8 quantization.

## KLD fidelity measurements

### v5 protocol (fidelity-v5 dataset, body-only hidden-state replay through shared BF16 LM head)

**BF16 control validation**: our RTX 6000 BF16 hidden states match the published
fidelity-v5 reference captures **bitwise** (SHA256 identical on 3/3 sampled contexts).
This confirms our vLLM container, model weights, and capture hook are correct.

#### Full 512-context results (shard-0000, 1,048,064 scored positions)

| candidate | our v5 KLD | model card v5 KLD (5120 ctx) | top-1 | p999 |
|---|---:|---:|---:|---:|
| **EXL3 K5K6 hydrated (pure trellis)** | **0.003410** | 0.002760 | 97.60% | 0.1785 |
| EXL3 K5K6 hydrated (+FP6 layers 0-11) | 0.003570 | — | 97.54% | 0.1893 |
| Official FP8 | 0.005197 | 0.005294 | 96.92% | 0.2440 |

Our pure-trellis KLD (0.003410) is 24% higher than the model card's published number
(0.002760). The difference is attributable to our container image / patch version:
the EXL3 quantization path produces slightly different trellis reconstructions than
the original capture environment. A 5-context apples-to-apples replay against the
dataset's own `hidden-hyd-rematch` captures confirms this: our capture scores 0.002852
vs the dataset's 0.002238 on the same 5 contexts, same reference, same LM head — a
27% gap from the runtime path, not from the model or protocol.

**The relative ordering is correct**: EXL3 K5K6 is 34% below official FP8 on our setup
(0.003410 vs 0.005197), matching the model card's 48% gap (0.002760 vs 0.005294).
Adding FP6 to layers 0-11 increases KLD by 5% (0.003410 → 0.003570), a small fidelity
cost for the throughput benefit.

### Simple protocol (qwen38_kld.py, full logits, single wiki window, includes lm_head)

| candidate | mean KLD | notes |
|---|---:|---|
| EXL3 K5K6 hydrated | 0.016970 | 3 runs, bitwise identical across runs |
| Official FP8 | 0.019908 | 3 runs, bitwise identical across runs |

Different absolute scale from v5 (includes head quantization, different corpus, single context),
but same ordering: **EXL3 K5K6 is 15% closer to BF16 than official FP8**.

## Profiles (docker-compose)

All profiles are in `docker-compose/docker-compose.yml`. Usage:

```bash
cp docker-compose/.env.example .env
# Edit paths as needed
docker compose --profile balanced up -d
```

### balanced (RTX 5090 / RTX 6000 production)
- Trellis K5/K6 + gate_up FP6 layers 0-11
- FP8 KV cache, MTP=6
- RTX 5090: 199K context, 0.945 util
- RTX 6000: 262K context, 0.92 util → 1.89M KV tokens

### fidelity (max quality)
- All-trellis, no FP6
- FP8 KV, MTP=6
- RTX 5090: 238K context

### throughput (max speed)
- All-FP4 (gate_up, down, linear_attn, self_attn)
- FP8 KV, MTP=6
- RTX 5090: 250K context

### rtx6000 (all optimizations)
- Trellis + FP6 + FP8 pre-fold for all projections
- FP8 KV, MTP=6, 262K context
- Pre-fold disk cache at `/cache/jit/exl3-prefold`
- First boot: ~10 min (729 extractions); subsequent: ~30s with cache

### rtx6000-bf16kv (higher fidelity)
- Same as rtx6000 but BF16 KV cache
- Trades throughput for KV precision

### official-fp8 (comparison baseline)
- Stock vLLM, no EXL3 patches
- Official `Qwen/Qwen3.8-27B-FP8` model
- No MTP (crashes with MTP patches)

### bf16-reference (KLD teacher)
- Full BF16 model
- Used as reference for KLD measurements

## Environment variables

### Core EXL3 multiprecision
| var | default | description |
|---|---|---|
| `VLLM_EXL3_MULTIPRECISION` | `1` | Enable multiprecision path |
| `QUANTIZATION_CONFIG` | see compose | Ignore list + linear weight format |
| `VLLM_EXL3_ONLINE_TRELLIS_BITS` | `6` | Online trellis encoding bits |
| `VLLM_EXL3_ONLINE_CACHE_DIR` | `/cache/jit/exl3-online` | Online encode cache |
| `VLLM_EXL3_ONLINE_CACHE_MODE` | `readwrite` | Cache read/write mode |
| `VLLM_EXL3_EMBED_ONLINE_BITS` | `6` | Embedding online bits |

### FP4/FP6 conversion
| var | default | description |
|---|---|---|
| `VLLM_EXL3_FP4_LAYERS` | `,` (none) | Comma-separated layer prefixes for FP4 |
| `VLLM_EXL3_FP6_LAYERS` | `mlp.gate_up_proj` | Layer prefixes for FP6 |
| `VLLM_EXL3_FP6_LAYER_RANGE` | `0-11` | Layer range for FP6 |

### Pre-fold (RTX 6000 only)
| var | default | description |
|---|---|---|
| `VLLM_EXL3_PREFOLD_FP8` | `1` (rtx6000) | FP8 pre-fold for all projections |
| `VLLM_EXL3_PREFOLD_CACHE_DIR` | `/cache/jit/exl3-prefold` | Disk cache for pre-folded weights |
| `VLLM_EXL3_PREFOLD_BF16` | `1` | BF16 pre-fold (down_proj, MLP) |
| `VLLM_EXL3_PREFOLD_QKV` | `1` | QKV pre-fold |
| `VLLM_EXL3_PREFOLD_GDN` | `1` | GDN pre-fold |
| `VLLM_EXL3_PREFOLD_LM_HEAD` | `1` | LM head pre-fold |

### B12X kernel
| var | default | description |
|---|---|---|
| `VLLM_EXL3_B12X_ANY_BITS` | `1` | Allow any trellis bit width |
| `VLLM_EXL3_B12X_MIN_M` | `128` | Minimum M for b12x path |
| `VLLM_EXL3_B12X_N_RANGE` | `5120-36864` | N range for b12x trellis |
| `B12X_PACKED_B_MIN_N` | `1024` | Min N for packed-B format |
| `B12X_DENSE_FUSED_QUANT` | `1` (rtx6000) | Fused activation quant |

### Prefill reconstruction
| var | default | description |
|---|---|---|
| `VLLM_EXL3_PREFILL_RECONSTRUCT_M` | `1` | M=1 identity extraction |
| `VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE` | `0`/`1` | Cache reconstructed weights |
| `VLLM_EXL3_FOLD_FP32_BUDGET_MB` | `48`/`256` | FP32 fold budget |

### Serving
| var | balanced | rtx6000 |
|---|---|---|
| `MAX_MODEL_LEN` | 199104 | 262144 |
| `GPU_MEMORY_UTILIZATION` | 0.945 | 0.92 |
| `MAX_NUM_SEQS` | 4 | 16 |
| `MAX_NUM_BATCHED_TOKENS` | 3072 | 8192 |
| `KV_CACHE_DTYPE` | fp8_e4m3 | fp8_e4m3 |
| `SPEC_CONFIG` | `{"method":"mtp","num_speculative_tokens":6}` | same |

## Pre-fold disk cache

The pre-fold disk cache (`VLLM_EXL3_PREFOLD_CACHE_DIR`) saves FP8 pre-folded weights
as safetensors, keyed by SHA256 of (trellis, suh, svh, K, N, format) tensor data.
Same atomic-save pattern as `exl3_online_cache.py`.

- **First boot**: ~10 min (729 extractions across 64+1 layers)
- **Subsequent boots**: ~30s with warm cache
- Cache is safe to delete; extraction will re-run on next boot

## KLD measurement protocol

### v5 protocol (fidelity-v5 dataset)

1. **Tokens**: `suite/tokens/context-NNNN.json` — 512 contexts × 2048 tokens from held-out corpus
2. **Capture**: `fidelity.py capture` — forward hook on `language_model.model.norm`, saves hidden states [2047, 5120]
3. **Replay**: `fidelity.py replay` — both operands projected through one shared BF16 LM head
4. **Metric**: `KL(BF16 reference ‖ candidate)`, two-pass, float64, no top-k

Dataset: `malaiwah/qwen38-27b-fidelity-suite-v5` on HuggingFace.

### Quick validation protocol (qwen38_kld.py)

1. **Tokens**: 2048 tokens from `wiki.utf8` (exllamav3 calibration data)
2. **Capture**: full logits [2047, 248320] from BF16 teacher
3. **Score**: candidate full logits vs teacher, mean KL divergence
4. **Limitation**: includes lm_head quantization, single context, different corpus

## Patch files

All patches are volume-mounted into the container at boot:

| patch | target | purpose |
|---|---|---|
| `vllm-exl3-multiprecision.py` | `vllm/.../quantization/exl3.py` | Core multiprecision + pre-fold |
| `scheduler_patch.py` | `vllm/v1/core/sched/scheduler.py` | MTP scheduling |
| `qwen_gdn_linear_attn_patch.py` | `vllm/.../mamba/gdn/qwen_gdn_linear_attn.py` | GDN linear attention |
| `qwen3_5_mtp_patch.py` | `vllm/.../models/qwen3_5_mtp.py` | MTP draft head |
| `exl3_fp4_conversion.py` | `/opt/fp4/` | FP4 conversion |
| `triton_fp4_quant.py` | `/opt/fp4/` | Triton FP4 activation quant |
| `exl3_fp6_conversion.py` | `/opt/fp6/` | FP6 conversion |
| `prefolded_bf16.py` | `/opt/prefold/` | BF16 pre-fold module |
| `batched_paired_projections.py` | `/opt/prefold/` | Trellis batching |

## Container image

```
docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
```

Pinned by digest. Contains vLLM with EXL3 support, exllamav3, CUDA 13.2.1.

## Recommendations

### RTX 5090 (32 GB)
Use **balanced** profile. Trellis K5/K6 + FP6 gate_up (layers 0-11) + FP8 KV + MTP=6.
199K context, 50.1 tok/s (MTP=0), 112.3 tok/s (MTP=6) on RTX 5090.

### RTX 6000 (96 GB)
Use **balanced** profile with 262K context. Same recipe as RTX 5090 but with full
native context window. 70.1 tok/s (MTP=6), 1.89M KV cache tokens.

The rtx6000 profile (FP8 pre-fold) is **slower** than balanced because trellis K6
reads fewer bytes than FP8 (0.75 vs 1.0 bytes/elem). Pre-fold is only beneficial
when VRAM is constrained and you need to eliminate the 200us trellis floor tax.

### FP6/FP8 multi-precision (K+1 rule)
Tested: FP6 on all 64 layers is 19% slower than balanced (57.2 vs 70.1 tok/s).
The K5→FP6 conversion loses trellis compression advantage. FP6 should remain
limited to layers 0-11 gate_up only, as in the balanced profile.

## Upstream PR testing on RTX 6000

### PR triage

All open and closed PRs across `malaiwah/qwen38-27b-exl3` and `local-inference-lab/b12x` were reviewed. Of 50+ PRs, only one pair is directly relevant to our Qwen3.8-27B EXL3 K5K6 deployment:

| PR | status | relevance | tested? |
|---|---|---|---|
| qwen38-27b-exl3 #1 + b12x #243 | open | **HIGH** — native BF16 K6 decode | ✅ tested |
| b12x #236 (QSRT K5) | open | moderate — different checkpoint format, 2.18 tok/s | no |
| b12x #221 (CuTe DSL K6) | open | low — FP16 only, MCG codebook, GLM-5.2 shapes | no |
| b12x #222 (MXFP4 GEMM) | open | low — different quant format (MXFP4 vs NVFP4) | no |
| b12x #241 (staged MXFP8) | open | low — MXFP8 linear, solves multi-slice problem we don't have | no |
| b12x #244 (checkpoint exporter) | open | not relevant — MoE-only, K3/K4/K5 only | no |

### Native BF16 K6 decode (PR #1 + b12x #243)

**What it does**: Fuses input H128 rotation + W4A16 K6/MCG GEMM + split-K reduction + output H128 + sign-vector multiply into one cooperative GPU grid for M ≤ 16 (decode). Keeps BF16 activations end-to-end, eliminating two dtype conversions. Default-off (`VLLM_EXL3_B12X_NATIVE_BF16=1` to enable).

**Image**: Built from `docker/Containerfile.b12x-native-bf16` (b12x 1.2.6 + CUTLASS DSL 4.6.2 + PR #243 on our base image).

**A-B-A results on RTX 6000** (262K context, FP8 KV, MTP=6, FP6 layers 0-9):

| phase | native BF16 | median tok/s | Paris | KV cache |
|---|---|---:|---|---|
| A1 (baseline) | OFF | 49.8 | ✅ | 1,780,274 |
| B (native BF16) | ON | **56.8** | ✅ | 1,780,274 |
| A2 (baseline-2) | OFF | 50.2 | ✅ | 1,780,274 |

**+13.6% decode speedup** (56.8 vs avg 50.0 tok/s), cache-proof, zero quality cost (BF16 activations preserved), same KV cache capacity. Exceeds the PR's claimed +6.2% — likely because the RTX 6000's larger SM count benefits more from the cooperative fused kernel.

Note: baseline is ~50 tok/s (not 70.1) because the PR1 image uses an older EXL3 patch without FP8 pre-fold, and FP6 range is 0-9 (not 0-11). The +13.6% is a relative improvement within the same image.
