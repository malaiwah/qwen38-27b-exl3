# Iteration 1 results — `malaiwah/Qwen3.8-27B-K4`

Hardware for every number below: 1x RTX PRO 6000 Blackwell Server Edition
(SM120, 96 GB), driver 595.58.03, TP1, r34 image
`voipmonitor/vllm@sha256:820181fb...df20592b` run without a container runtime
(unprivileged OCI pull + proot; see [06](06-baseline-validation.md)).

## Build

| step | outcome |
|---|---|
| `convert.py -b 4 -hb 6 -mb 4 -vb 16 -cb mcg -d 0` | 64 blocks at ~29 s/block, ~33 min total, peak RSS 14.3 GB, well under 10 GB VRAM |
| crash at compile stage | `ModuleNotFoundError: marisa_trie` — installed with `--no-deps`; fixed and resumed with `convert.py -w wd-k4 -r`, 2m50s |
| splice | 216 modules returned to BF16 (144 `linear_attn` + 64 `self_attn` + 8 MTP), 864 EXL3 tensors dropped |
| metadata | `add_safetensors_index.py --force` + `add_quant_config.py`: `tensor_storage` = 192 MLP entries at K4, 1 `lm_head` at K6, 384 BF16 attention entries with `stored_tensors` only |
| checkpoint | 28.31 GB, 4 shards, 1778 tensors |

Observed nuance: `-cb mcg` is honoured for the decoder K4 tensors but the
`lm_head` and MTP payloads were written with `mul1`, and `config.json`
`quantization_config.codebook` reports `mul1` for the whole checkpoint. Harmless
for correctness (K4 and non-K6 shards route to `exl3_gemm` either way), but it
means the serialized K6 `lm_head` does not take the B12X native K6 path, which
requires `mcg`.

## Serving

| metric | value |
|---|---|
| resident weights | **17.89 GiB = 19.21 GB** (predicted 19.28 GB, 0.4 % error) |
| peak activation | 2.33 GiB |
| KV cache | 736,109 tokens at `--max-model-len 8192`, `--gpu-memory-utilization 0.85` |
| online K6 encode, cold | 208 attention projections, ~16 min, proxy error ~3.2e-4 each |
| CUDA graphs | none (`--enforce-eager` required by the loader) |
| text | `"three primary additive colors"` -> `Red, Green, Blue`, 1.6 s greedy |
| vision | 96x96 half-red/half-blue PNG -> `red, blue` |

## KLD, teacher-forced, full vocabulary

Protocol per [05](05-kld-protocol.md): one frozen 2048-token window
(exllamav3 `wiki.utf8`), 2047 scored positions, `KL(BF16 teacher || candidate)`,
3 repeats, `kv_cache_dtype=auto` pinned for every candidate.

| candidate | mean KLD | run SD | SD across positions | resident weights |
|---|---:|---:|---:|---:|
| `k4-online-k6` (this quant) | **0.034030** | 0.000000 | 0.4628 | 19.21 GB |

`run_sd = 0` means the three repeats were bit-identical, which is what the eager,
`max_num_seqs=1`, prefix-cache-disabled configuration is supposed to produce.

Remaining candidates in flight: `unsloth-nvfp4` (same-generation NVFP4 control)
and `k4-bf16-attn` (this checkpoint with the overlay switched off, isolating the
online-K6 contribution).

## Upstream defect found and reported

Serving initially crashed with
`ValueError: MXFP8 requires input_size_per_partition (4304) to be divisible by 32`.
Root cause: `Exl3OnlineLinearMethod.create_weights` installs
`Mxfp8OnlineLinearMethod` as an unconditional fallback for shards that are not
128-aligned, but MXFP8 rejects `K % 32 != 0`; the Qwen3.5/3.6/3.8 vision tower
(`linear_fc2`, K=4304) satisfies neither constraint. Filed as
[local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311)
with a patch that falls back to `UnquantizedLinearMethod` and warns.

Second, subtler finding in the same area: the overlay's `ignore` regexes are
matched against prefixes with no leading `model.`, so the natural-looking
`re:.*\.visual\..*` never matches the vision tower while `re:.*visual\..*`
does. Operators who write the first form get the crash above.
