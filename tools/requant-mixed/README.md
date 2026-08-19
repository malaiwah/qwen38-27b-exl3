# llmcompressor Mixed-Precision Requant Recipes for Qwen3.8-27B

This directory contains runnable recipes for GPTQ mixed-precision
requantization of `Qwen/Qwen3.8-27B`, producing compressed-tensors
checkpoints loadable by stock vLLM via `--quantization compressed-tensors`.

## Pinned versions

| Package | Version | Verified against |
|---------|---------|------------------|
| llmcompressor | **0.13.0** | source at `/tmp/lc-venv/...` |
| compressed-tensors | **0.17.0** | source at `/tmp/lc-venv/...` |
| transformers | ≥ 5.8.0.dev0 | Qwen3.5 requires bleeding-edge or `trust_remote_code` |
| torch | ≥ 2.5 (CUDA 12.x) | FP8 E4M3 dtype support required |
| datasets | latest | For calibration data loading |
| accelerate | latest | For model offloading support |

## Venv setup

```bash
python3 -m venv /tmp/requant-venv
source /tmp/requant-venv/bin/activate
pip install llmcompressor==0.13.0 compressed-tensors==0.17.0
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
```

> **Note**: Qwen3.8-27B uses `Qwen3_5ForConditionalGeneration` architecture
> with `transformers_version: 5.8.0.dev0`. If your installed transformers
> does not have native Qwen3.5 support, the model ships remote code and
> `--no-trust-remote-code` must NOT be passed (trust_remote_code defaults
> to true in the driver).

## Recipes

### 1. `recipe-fp8attn-nvfp4mlp.yaml` — Fidelity winner

| Module group | Scheme | Format | Strategy | Group |
|---|---|---|---|---|
| `self_attn.{q,k,v,o}_proj` | FP8 W8A16 (weight-only) | `naive_quantized` | channel | — |
| `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}` | FP8 W8A16 (weight-only) | `naive_quantized` | channel | — |
| `mlp.{gate,up,down}_proj` | NVFP4 W4A16 | `nvfp4_pack_quantized` | tensor_group | 16 |

Model-level format: **mixed-precision** (two distinct per-module formats).

This reproduces the third-party result that measured KLD 0.002666 on an
independent harness — 3× better than EXL3 K5K6 trellis on the same harness.

### 2. `recipe-fp8attn-nvfp4w4a4mlp.yaml` — W4A4-MLP middle SKU

| Module group | Scheme | Format | Strategy | Group |
|---|---|---|---|---|
| `self_attn.{q,k,v,o}_proj` | FP8 W8A16 (weight-only) | `naive_quantized` | channel | — |
| `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}` | FP8 W8A16 (weight-only) | `naive_quantized` | channel | — |
| `mlp.{gate,up,down}_proj` | NVFP4 W4A4 | `nvfp4_pack_quantized` | tensor_group | 16 |

MLP activations are also quantized to FP4 with `dynamic: local` (per-group
scales computed at runtime, static global scale calibrated). This is the
preset `NVFP4` scheme from compressed-tensors.

### Ignored modules (kept unquantized)

- `lm_head` — kept BF16 in v1 to minimize error surface
- `embed_tokens` — embedding layer
- All `norm` / `layernorm` weights (RMSNorm)
- `linear_attn.conv1d`, `linear_attn.A_log`, `linear_attn.dt_bias`,
  `linear_attn.norm` — GDN conv/state/small modules
- `self_attn.q_norm`, `self_attn.k_norm` — small RMSNorm in attention
- `model.visual.*` — entire vision tower (27 blocks + merger + patch embed)
- `mtp.*` — MTP (multi-token prediction) head

## Module-name mapping (HF checkpoint)

Verified from `model.safetensors.index.json` in the local HF cache
(`~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2.../`).

The model is `Qwen3_5ForConditionalGeneration` — a multimodal model with
language model at `model.language_model.*` and vision tower at
`model.visual.*`.

| Role | HF checkpoint module name pattern | Layer count | Quantized? |
|---|---|---|---|
| Full attention Q/K/V/O | `model.language_model.layers.{N}.self_attn.{q,k,v,o}_proj` | 16 layers (every 4th: 3,7,11,...,63) | FP8 W8A16 |
| Linear-attn (GDN) in_proj_qkv | `model.language_model.layers.{N}.linear_attn.in_proj_qkv` | 48 layers (all non-attention) | FP8 W8A16 |
| Linear-attn (GDN) in_proj_z | `model.language_model.layers.{N}.linear_attn.in_proj_z` | 48 | FP8 W8A16 |
| Linear-attn (GDN) in_proj_a | `model.language_model.layers.{N}.linear_attn.in_proj_a` | 48 | FP8 W8A16 |
| Linear-attn (GDN) in_proj_b | `model.language_model.layers.{N}.linear_attn.in_proj_b` | 48 | FP8 W8A16 |
| Linear-attn (GDN) out_proj | `model.language_model.layers.{N}.linear_attn.out_proj` | 48 | FP8 W8A16 |
| MLP gate_proj | `model.language_model.layers.{N}.mlp.gate_proj` | 64 | NVFP4 |
| MLP up_proj | `model.language_model.layers.{N}.mlp.up_proj` | 64 | NVFP4 |
| MLP down_proj | `model.language_model.layers.{N}.mlp.down_proj` | 64 | NVFP4 |
| LM head | `lm_head` | 1 | Ignored (BF16) |
| Embeddings | `model.language_model.embed_tokens` | 1 | Ignored |
| Norms | `model.language_model.layers.{N}.{input_layernorm,post_attention_layernorm}`, `model.language_model.norm` | 65 | Ignored |
| GDN conv/state | `model.language_model.layers.{N}.linear_attn.{conv1d,A_log,dt_bias,norm}` | 48×4 | Ignored |
| Attn norms | `model.language_model.layers.{N}.self_attn.{q_norm,k_norm}` | 16×2 | Ignored |
| Vision tower | `model.visual.blocks.{0-26}.*`, `model.visual.merger.*`, `model.visual.patch_embed.*`, `model.visual.pos_embed` | 27 blocks | Ignored |
| MTP head | `mtp.*` | 1 layer | Ignored |

**Key note**: The HF checkpoint has **unfused** attention projections
(separate `q_proj`, `k_proj`, `v_proj` — not a fused `qkv_proj`). The GDN
modules use `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` as
separate weights (not a fused `in_proj_qkvz`). GPTQ operates on these
unfused HF checkpoint names.

## Running the requantization

```bash
source /tmp/requant-venv/bin/activate

# Fidelity winner (FP8 attn + NVFP4 W4A16 MLP)
python tools/requant-mixed/run-requant.py \
    --recipe recipe-fp8attn-nvfp4mlp.yaml \
    --output-dir /tmp/qwen38-27b-fp8attn-nvfp4mlp \
    --model Qwen/Qwen3.8-27B \
    --samples 512 \
    --seq-len 2048

# W4A4-MLP variant
python tools/requant-mixed/run-requant.py \
    --recipe recipe-fp8attn-nvfp4w4a4mlp.yaml \
    --output-dir /tmp/qwen38-27b-fp8attn-nvfp4w4a4mlp \
    --model Qwen/Qwen3.8-27B \
    --samples 512 \
    --seq-len 2048
```

### Calibration dataset

Default: `HuggingFaceH4/ultrachat_200k` split `train_sft`, 512 samples,
max sequence length 2048. This is a standard llmcompressor calibration
dataset. Alternative: `Open-Orca/OpenOrca` or `garage-bAInd/Open-Platypus`.

The `--samples` and `--seq-len` flags control the calibration budget.
More samples + longer sequences → better GPTQ Hessian estimates but
longer runtime. 512 × 2048 is a reasonable default for a 27B model.

## Memory plan (32 GiB GPU + host RAM)

The 27B model in BF16 is ~54 GB — it cannot fit entirely on a 32 GiB GPU.
The driver uses the **sequential pipeline** (`pipeline="sequential"`)
which processes one decoder layer at a time:

1. **Model loading**: Full model loads to CPU RAM (requires ≥ 64 GB host
   RAM). No `device_map="auto"` — the sequential pipeline handles GPU
   placement per-layer.

2. **Sequential calibration**: For each of the 64 decoder layers:
   - Layer weights moved to GPU (~0.8 GB per layer in BF16)
   - Calibration data (batch_size=1, seq_len=2048) forwarded through
   - GPTQ Hessian accumulated (offloaded to CPU via `offload_hessians: true`)
   - Weights quantized in-place
   - Layer offloaded back to CPU

3. **Peak VRAM**: ~2-4 GB (one layer's weights + activations + GPTQ
   working buffers). The 32 GiB GPU has ample headroom.

4. **Peak host RAM**: ~54 GB (full BF16 model) + ~10 GB (Hessians for
   all layers, offloaded). Total: ~64 GB minimum, 96 GB recommended.

### Expected runtime

| Component | Estimate |
|---|---|
| Model load (CPU) | 5-10 min (safetensors → CPU RAM) |
| GPTQ per layer | 2-5 min (512 samples × 2048 seq_len) |
| 64 layers total | 2-5 hours |
| Save (compressed) | 10-20 min |
| **Total wall time** | **3-6 hours** |

Runtime depends heavily on:
- CPU memory bandwidth (Hessian offload is bandwidth-bound)
- Number of calibration samples (linear scaling)
- Sequence length (quadratic for attention, linear for linear-attn)

## Serving the result

The output is a standard compressed-tensors checkpoint. vLLM loads it
via auto-detection from `config.json` (which will contain
`quantization_config.quant_method: "compressed-tensors"` and
`format: "mixed-precision"`).

### With stock vLLM

```bash
vllm serve /tmp/qwen38-27b-fp8attn-nvfp4mlp \
    --quantization compressed-tensors \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768
```

> vLLM should auto-detect the quantization from `config.json`, so
> `--quantization compressed-tensors` is often optional.

### With our containerized launcher

Our serving stack uses podman with the voipmonitor/vllm image. Mount the
checkpoint and serve:

```bash
podman run --rm --device nvidia.com/gpu=all --ipc=host --network host \
    -e HF_HOME=/root/.cache/huggingface \
    -v /tmp/qwen38-27b-fp8attn-nvfp4mlp:/model:ro \
    docker.io/voipmonitor/vllm:gilded-gnosis-v20-... \
    vllm serve /model --quantization compressed-tensors \
    --gpu-memory-utilization 0.90 --max-model-len 32768
```

No EXL3 patches or `VLLM_EXL3_*` environment variables are needed — this
is a compressed-tensors checkpoint, not an EXL3 checkpoint. The
`compressed_tensors` quantization method in vLLM handles FP8 weight-only
(naive_quantized) and NVFP4 (nvfp4_pack_quantized) natively.

## Measuring KLD with our harness

Our KLD harness lives at `tools/fidelity.py` with runner scripts following
the `tools/kld-run-*.sh` pattern. The harness uses `QCFG` env for
quantization config overrides and runs capture → replay → paired comparison.

### Serving for KLD measurement

```bash
# Serve the compressed-tensors checkpoint (no QCFG needed — quantization
# is baked into the checkpoint's config.json)
export MODEL_DIR=/tmp/qwen38-27b-fp8attn-nvfp4mlp

# The KLD harness captures hidden states and compares against a BF16 reference.
# Use the standard kld-run.sh pattern but point at the compressed-tensors model:
python tools/fidelity.py capture \
    --model "$MODEL_DIR" \
    --suite /tmp/kld-data/suite/shard-0000 \
    --out /tmp/kld-data/captures/shard-0000/hidden-requant \
    --quantization compressed-tensors \
    --gpu-memory-utilization 0.85

python tools/fidelity.py replay \
    --reference /tmp/kld-data/reference/hidden-bf16 \
    --candidate /tmp/kld-data/captures/shard-0000/hidden-requant \
    --head /tmp/kld-data/lm-head/weight.safetensors \
    --suite /tmp/kld-data/suite/shard-0000 \
    --out /tmp/kld-data/reports/report-requant.json
```

### KLD harness script

Create a `tools/kld-run-requant.sh` following the existing
`tools/kld-run-*.sh` pattern, adapting the `QCFG` env and model path:

```bash
#!/bin/bash
# KLD measurement for compressed-tensors requant checkpoint.
# Unlike EXL3 checkpoints, no QCFG override is needed — the quantization
# config is embedded in the checkpoint's config.json.
set -uo pipefail

MODEL_DIR="${1:?usage: kld-run-requant.sh /path/to/checkpoint}"
TAG="${2:-requant}"
```

Then run the standard capture → replay → paired pipeline as in the
existing `kld-run.sh` scripts.

## Scheme verification (source citations)

All scheme strings, modifier classes, and arguments are verified against
llmcompressor 0.13.0 and compressed-tensors 0.17.0 source:

| Element | Source file:line | Verification |
|---|---|---|
| `GPTQModifier` class | `llmcompressor/modifiers/gptq/base.py:40` | Inherits `QuantizationMixin`, supports `config_groups` |
| `config_groups` field | `llmcompressor/modifiers/quantization/quantization/mixin.py:103` | `dict[str, QuantizationScheme]` — multiple groups = mixed-precision |
| `resolve_quantization_config` | `mixin.py:344-383` | Builds `QuantizationConfig` from `config_groups` |
| FP8 weight-only (custom) | `quant_scheme.py` — no preset; defined inline | `QuantizationArgs(num_bits=8, type=float, strategy=channel)` |
| NVFP4A16 preset | `compressed_tensors/quantization/quant_scheme.py:145-157` | `weights: num_bits=4, type=float, strategy=tensor_group, group_size=16` |
| NVFP4 (W4A4) preset | `quant_scheme.py:159-185` | Same weights + `input_activations: dynamic=local` |
| `PRESET_SCHEMES` dict | `quant_scheme.py:408-428` | Lists all preset names including `NVFP4A16`, `NVFP4` |
| `CompressionFormat` enum | `compressed_tensors/config/base.py:16-28` | `naive_quantized`, `nvfp4_pack_quantized`, `mixed_precision` |
| Mixed-precision detection | `compressed_tensors/compressors/format.py:113` | `≥2` distinct formats → `mixed_precision` |
| NVFP4 compressor `can_compress` | `compressors/nvfp4/base.py:135-140` | FP4 + group_size=16 → `nvfp4_pack_quantized` |
| NaiveQuantized `can_compress` | `compressors/naive_quantized/base.py:118-120` | Fallback: any quantized scheme with weights |
| FloatQuantized `can_compress` | `naive_quantized/base.py:151-158` | Requires `input_activations is not None` + FLOAT type |
| `match_name` regex | `compressed_tensors/utils/match.py:432` | `re:` prefix → `re.match(target[3:], name)` |
| `sequential_targets` | `llmcompressor/args/dataset_arguments.py:170-178` | Layer pattern for sequential pipeline |
| `offload_hessians` | `llmcompressor/modifiers/gptq/base.py:100` | GPTQModifier field, offloads Hessian to CPU |
| `oneshot()` API | `llmcompressor/entrypoints/oneshot.py:309-398` | Python entrypoint with recipe, dataset, pipeline args |

## Limitations / things llmcompressor cannot express

1. **FP8 weight-only is not a preset scheme.** The `FP8` preset includes
   both weights and activations (W8A8). For weight-only FP8 (W8A16), we
   define a custom `QuantizationArgs` with no `input_activations`. This
   is fully supported but produces `naive_quantized` format (storing
   weights in `float8_e4m3fn` dtype) rather than a dedicated FP8 format.

2. **GPTQ with FP8 weights.** The GPTQ algorithm quantizes column-by-column
   using `fake_quantize`, which supports FLOAT type. However, FP8 (E4M3)
   has 256 levels and a non-uniform grid — GPTQ's error compensation is
   most impactful for low-bit (INT4) quantization. For FP8, the
   quantization error is already small and GPTQ provides marginal benefit.
   The recipe still uses GPTQModifier (as the third-party result specifies
   "GPTQ mixed quantization"), but a plain `QuantizationModifier` would
   also work for the FP8 groups.

3. **`device_map` is not a `oneshot()` argument.** The model loads to CPU
   by default. The sequential pipeline handles per-layer GPU placement.
   There is no way to pass `device_map="auto"` through the `oneshot()` API
   (it is not a field in `ModelArguments`). This is by design — the
   sequential pipeline is the intended memory-management mechanism.

4. **Qwen3.5 is a multimodal model.** `AutoModelForCausalLM.from_pretrained`
   is used internally, which may require `trust_remote_code=True` for the
   `Qwen3_5ForConditionalGeneration` architecture. The vision tower
   (`model.visual.*`) is ignored in all recipes.
