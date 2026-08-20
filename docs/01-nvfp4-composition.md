# Measured composition of the reference NVFP4 quants

Everything here was measured from the downloaded checkpoints' safetensors
headers and quantization sidecars, not from model-card prose. Byte totals are
decimal GB and are the sum of tensor data extents.

## Shared architecture

`Qwen/Qwen3.6-27B` and `Qwen/Qwen3.8-27B` are architecturally identical, so a
recipe transfers between them without change:

| key | value |
|---|---|
| `model_type` / architecture | `qwen3_5` / `Qwen3_5ForConditionalGeneration` |
| layers | 64 (`layer_types`: 48 `linear_attention`, 16 `full_attention`, `full_attention_interval=4`) |
| hidden / intermediate | 5120 / 17408 |
| heads | 24 q, 4 kv, `head_dim` 256 |
| vocab | 248320, `tie_word_embeddings=false` |
| vision tower | 27 blocks |
| MTP | 1 layer |
| BF16 size | 27.781 B params, 55.56 GB, 1199 tensors |

Every dimension that matters for EXL3 (5120, 6144, 10240, 12288, 17408, 1024,
248320) is a multiple of 128, so both the EXL3 kernel alignment rule and the
online-K6 eligibility rule (`input % 128 == 0 and output % 128 == 0`) are
satisfied without padding. The vision tower is the exception: its MLP width
4304 is **not** 128-aligned.

## BF16 parameter census (`Qwen/Qwen3.8-27B`)

| role | tensors | params | BF16 |
|---|---:|---:|---:|
| MLP `gate/up/down_proj` (64 L) | 192 | 17.1128 B | 34.23 GB |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (48 L) | 144 | 5.5365 B | 11.07 GB |
| `self_attn.{q,k,v,o}_proj` (16 L) | 64 | 1.6777 B | 3.36 GB |
| `lm_head` | 1 | 1.2714 B | 2.54 GB |
| `embed_tokens` | 1 | 1.2714 B | 2.54 GB |
| vision tower | 333 | 0.4607 B | 0.92 GB |
| MTP | 15 | 0.4247 B | 0.85 GB |
| norms/biases | 449 | 0.0262 B | 0.05 GB |

## `nvidia/Qwen3.6-27B-NVFP4` — 21.92 GB

`hf_quant_config.json`: modelopt 0.45.0, `quant_algo: MIXED_PRECISION`,
`kv_cache_quant_algo: FP8`. `config.json -> quantization_config.ignore = ["mtp*", "mtp.layers.0*"]`.

| role | scheme | bytes |
|---|---|---:|
| MLP, all 64 layers | `W4A16_NVFP4`, `group_size` 16 — U8 codes + F8_E4M3 `weight_scale` | 8.56 + 1.07 = **9.63 GB** (4.50 bpw) |
| `linear_attn` ×48, `self_attn` ×16 | FP8 E4M3, W8A8 | **7.22 GB** (8 bpw) |
| `lm_head` | NVFP4 | 0.72 GB |
| `embed_tokens` | BF16 | 2.54 GB |
| vision tower | BF16 | 0.92 GB |
| MTP | BF16 (ignored) | 0.85 GB |

Two config groups, verified: `group_0` = 208 targets at 8-bit float weights
*and* 8-bit float input activations; `group_1` = 193 targets at 4-bit float
weights, `group_size` 16, no activation entry (weight-only), and it includes
`lm_head`.

## `unsloth/Qwen3.8-27B-NVFP4` — 23.42 GB

compressed-tensors 0.17.2.a20260716, `format: mixed-precision`.
Target regexes, verbatim:

- `group_0` (W8A8 FP8, `strategy: channel`, dynamic activations):
  `re:.*self_attn\.(q|k|v|o)_proj$`, `re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$`,
  `re:.*lm_head`, `re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$`
- `group_1` (W4A4 NVFP4, `group_size` 16, `strategy: tensor_group`, `dynamic: local`):
  `re:.*mlp\.(gate|up|down)_proj$`
- `ignore`: every `model.visual.blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`

| role | scheme | bytes |
|---|---|---:|
| MLP layers 0-55 | NVFP4 W4A4 gs16 | 8.42 GB |
| MLP layers 56-63 | FP8 | 2.14 GB |
| attention, all 64 layers | FP8 W8A8 dynamic per-channel | 7.27 GB |
| `lm_head` | FP8 | 1.27 GB |
| `embed_tokens` | BF16 | 2.54 GB |
| vision tower | BF16 (explicit ignore) | 0.92 GB |
| MTP | BF16 | 0.85 GB |

## What the two vendors agree on

1. **Within the decoder blocks, 4-bit is applied only to the MLP.** NVIDIA
   separately applies NVFP4 to `lm_head`; attention, embeddings, the vision
   tower and the MTP module are never 4-bit in either recipe.
2. **The vision tower stays BF16** in both — Unsloth spells it out per block.
3. They diverge on `lm_head` (NVIDIA 4-bit, Unsloth 8-bit), on MLP activations
   (NVIDIA W4A16, Unsloth W4A4), and on tail protection (Unsloth keeps the last
   8 layers' MLP at 8-bit; NVIDIA does not).

Those two independent choices are the prior art our recipe copies: protect
attention, protect the head, protect the vision tower, spend the 4-bit budget
on the MLP stack.
