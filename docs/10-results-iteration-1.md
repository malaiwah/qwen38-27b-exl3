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
| `self-bf16` (harness control: teacher vs its own logits) | **0.000000** | 0.000000 | 0.0000 | 55.56 GB |
| `k4-online-k6` (this quant) | **0.034030** | 0.000000 | 0.4628 | 19.21 GB |
| `k4-bf16-attn` (same checkpoint, overlay off) | 0.036775 | 0.000000 | 0.6210 | 24.63 GB |
| `unsloth/Qwen3.8-27B-NVFP4` | 0.091457 | 0.000000 | 0.8036 | 23.42 GB |

**The harness is validated**: scoring the BF16 teacher against its own captured
logits returns exactly 0.000000, so the densification path and the window
identity introduce no bias. Every candidate above shares that teacher file.

**2.69x closer to BF16 than the same-generation NVFP4 checkpoint**, with 4.2 GB
less resident weight.

**Online K6 attention beat BF16 attention** on the identical checkpoint
(0.034030 vs 0.036775). Not noise: run SD is 0 for both. The K6 encode applies
out-scales that partly cancel the K4 MLP's systematic shrinkage (`g_sc ~0.895`
per projection in the conversion log). Consequence for the next iteration:
attention is not where the remaining error lives, so it is not where bits should
go.

`run_sd = 0` means the three repeats were bit-identical, which is what the eager,
`max_num_seqs=1`, prefix-cache-disabled configuration is supposed to produce.

## Throughput, and what eager execution costs

Same GPU, `--max-num-seqs 8`, greedy, `ignore_eos`, 256 output tokens per request,
one warmup request discarded.

| candidate | mode | C1 tok/s | C4 tok/s | C8 tok/s | single-token latency |
|---|---|---:|---:|---:|---:|
| this quant | EXL3, **eager** (forced) | 28.77 | 103.47 | 215.84 | 56 ms |
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4 Cutlass, CUDA graphs | 49.09 | 171.78 | 371.06 | 39 ms |

| `Qwen/Qwen3.8-27B` BF16 | CUDA graphs | 27.47 | 101.04 | 208.31 | 46 ms |
| `Qwen/Qwen3.8-27B` BF16 | eager | 25.50 | 92.72 | 186.98 | 52 ms |

The BF16 pair separates the two confounds. **CUDA graphs are worth +7.7 % / +9.0 %
/ +11.4 %** at C1/C4/C8 on this architecture, so the loader's eager requirement
costs about 10 %. The remaining 1.7x gap to NVFP4 is kernel-bound: Cutlass NVFP4
GEMM versus `exl3_gemm`. Note that this quant at K4, eager, already **matches
BF16 with graphs** (28.77 vs 27.47 at C1) using 36 GB less weight.

Consequence for iteration 2: chasing CUDA graphs buys ~10 %; moving GEMMs onto the
B12X native Trellis kernel — which accepts **only exactly-K6 shards with the `mcg`
codebook** — is the larger perf lever, and it happens to align with the accuracy
lever (more K6). Our `lm_head` is K6 but was written with `mul1`, so it misses that
path today.

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
