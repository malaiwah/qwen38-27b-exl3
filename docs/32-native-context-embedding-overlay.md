# Native 262,144 context under a 32 GB engine budget, by narrowing the input table

[docs/30](30-iteration-4-context-edition.md) closed with native context out of reach:
8.18 GiB of KV needed against 7.55 available, gap 0.63 GiB, and the input
embedding table named as the only cheap lever large enough. This is that lever,
built and measured.

## Scope of the capacity claim

The final profile ran on an RTX PRO 6000 Blackwell with vLLM's memory budget capped
below a 32 GB card: **30.24 GiB**, versus 30.44 GiB from a 31.39 GiB RTX 5090 at
utilisation 0.97. It started at `--max-model-len 262144` with MTP-3, decode graphs,
multimodal profiling and an 8,388,608-pixel image ceiling; allocated **266,612 KV
tokens**; retrieved a planted code from 261,794 text tokens; and, in a separate combined
request, retrieved the code and read a 3,072 × 2,304 image from a 236,824-token prompt.

That proves the engine budget and served paths together. It does **not** reproduce a hard
32 GB physical limit: the host GPU still had memory outside vLLM's budget.

> **Superseded by measurement, 2026-08-16.** The physical RTX 5090 rerun this section asked for
> has been run and **passed**, so the native-context claim no longer rests on this
> engine-budget proof. Every number in this document stands exactly as measured; the hardware
> claim now comes from `receipts/qualification-5090-context.json` (schema
> `qwen38-qualification-5090-context/1`), measured on one physical RTX 5090 of 32,607 MiB,
> 31.4 GiB usable as vLLM sizes it. There the same window, MTP depth, KV dtype and
> 8,388,608-pixel ceiling serve native 262,144 at **`--gpu-memory-utilization 0.955`** with
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: engine budget **29.98 GiB**, available KV
> **9.28 GiB = 265,122 tokens**, 1.01x maximum concurrency at 262,144, all seven gates passing.
> **The utilisation 0.97 named just above does not survive on the physical card:** it does size
> the budget at 30.45 GiB and it serves text, but the combined 236,824-token plus
> 7,077,888-pixel request dies in the vision tower wanting 62.00 MiB with 26.50 MiB free, and
> lowering `max_pixels` to 4,194,304 instead of utilisation is strictly worse — KV grows to
> 291,933 tokens and it OOMs with 6.56 MiB free. Utilisation is the knob, not the image ceiling.

## What was done

The input embedding table is **248,320 x 5,120 in BF16 = 2.543 GB resident** —
second only to the MLP stack, larger than the entire quantized attention stack,
and pure lookup: one row per token, no accumulation or tensor-core matmul.

`VLLM_EXL3_EMBED_BITS=8` converts it after load to per-row symmetric int8. Each
row keeps its own scale. int8 rather than FP8 is deliberate: E4M3 has three
mantissa bits against int8's seven, and a gather cannot exploit FP8 tensor-core
throughput.

| | BF16 embeddings | int8 embeddings |
|---|---:|---:|
| embedding table | 2.543 GB | **1.272 GB** |
| resident weights, MTP off | 19.31 GiB | **18.13 GiB** |
| largest configured context tested | 229,376 | **262,144 (native)** |
| KV allocated at that length | 240,080 tokens | **279,007 tokens** |
| mean KLD, v3 full / overlap-corrected | 0.009673 / **0.009378** | 0.009738 / **0.009459** |
| multimodal, 30 synthetic cases | 24/30 | **24/30, identical** |

On the original 136-context receipt the cost is **+0.000065 mean KLD**, 95 % CI
[+0.0000046, +0.00013], 49/136 contexts. On the corrected 127-context subset the
point-estimate difference is **+0.000082**; both are below the replay-resolution caveat.

Generation, not allocation alone: a planted code was retrieved exactly at
depths 0.1, 0.5 and 0.9 from **227,334-token prompts**, 94.6-95.0 seconds each,
about 2,400 prompt tokens per total request wall second, including decode and HTTP overhead.
This is not an engine-timed prefill rate. The 30-case synthetic vision score was unchanged.

## The missing hook was in model construction

`VocabParallelEmbedding` already asks
`quant_config.get_quant_method(self, prefix)` and accepts methods implementing
`embedding()`. `qwen3_5.py` and `qwen3_5_mtp.py` constructed the layer without
passing `quant_config`, so the hook was unreachable. Passing the existing config
is sufficient; backends returning no embedding method keep the prior BF16 path.

## Correction: the MTP draft does not retain a second embedding table

The first write-up said the draft head kept a second resident table and that an
int4 draft table preserved acceptance. That was wrong.

Both MTP load logs say:

```text
Detected MTP model. Sharing target model embedding weights with the draft model.
```

vLLM materializes the draft embedding during load, then aliases it to the target
embedding. The int8/int4 comparison narrowed a temporary table that was replaced
before inference. Its 56.1 % versus 56.7 % acceptance difference was ordinary
run noise, **not an int4 measurement**. `VLLM_EXL3_MTP_EMBED_BITS` and the int4
method have therefore been removed. PR #319's body and model-card text were
corrected with a public comment recording the mistake.

The MTP model-file patch still matters for transient load memory and standalone
construction, but it does not buy steady-state KV capacity.

## MTP-3 recovered by bounding image profiling

The first native-MTP attempts kept the model's **16,777,216-pixel** image default. They
failed before serving: MTP-3 needed 9.13 GiB of KV against 8.59 GiB available. That was
not all immutable model cost; vLLM profiles activation memory for the largest permitted
image during startup.

The production profile makes that resource contract explicit:

```bash
VLLM_EXL3_EMBED_BITS=8 VLLM_EXL3_GRAPH_DECODE=1 \
vllm serve ... --max-model-len 262144 --max-num-seqs 1 \
  --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
```

| profile | budget | peak activation | available KV | KV tokens | result |
|---|---:|---:|---:|---:|---|
| default 16.8 MP, MTP-3 | 30.44 GiB | — | 8.59 GiB | estimated 244,800 | refuses native length |
| **8.4 MP, MTP-3** | **30.24 GiB** | **1.78 GiB** | **9.31 GiB** | **266,612** | **starts at 262,144** |

This is not a text-only loophole:

- 261,794 text prompt tokens, depth-0.5 needle: **exact**, 123.1 s, 2,127 tok/s including decode;
- the unchanged deterministic image suite: **24/30**, identical to the uncapped profile;
- a 3,072 × 2,304 image (7,077,888 pixels): `red, blue`, exact;
- one request combining 229,910 measured text tokens and that image: **236,824 prompt
  tokens**, exact `1376346594 | red, blue`, 106.5 s;
- warmed 256-token single-stream decode with MTP-3: **98.72 tok/s** in one timing run.

The image limit halves the published processor default but still admits a seven-megapixel
input. It is the resource exchange that makes native context, speculative decode and useful
multimodality coexist; it is not a claim that every 16.8 MP input fits. Full logs, requests,
fixture hash, module identities and the engine-budget caveat are in
`receipts/native-mtp-8mp-amendment.json`.

The physical RTX 5090 rerun is **closed**: run, all seven gates passed, published as
`receipts/qualification-5090-context.json` — at utilisation **0.955**, not the 0.97 this
document assumed. Compact KV grouping remains rejected: it reduced
padding from three layers to one but raised graph memory from 0.46 to 1.25 GiB and increased
the required KV allocation. More compression of the aliased draft embedding is not a route.
