# Iteration 2: gate K5 / up K5 / down K6

## Recipe and why

The v1 recipe put the whole MLP at K4 and left 2.71 GB of the NVFP4-equivalent budget
unspent. The head-attribution and per-tensor error data said the residual error lives in
the MLP, so iteration 2 spends the budget there:

| role | v1 | v2 | on-disk | resident |
|---|---|---|---:|---:|
| MLP `gate_proj` | EXL3 K4 | **EXL3 K5** | 3.568 GB | 3.568 GB |
| MLP `up_proj` | EXL3 K4 | **EXL3 K5** | 3.568 GB | 3.568 GB |
| MLP `down_proj` | EXL3 K4 | **EXL3 K6** | 4.281 GB | 4.281 GB |
| attention (208 proj.) | BF16 -> online K6 | same | 14.43 GB | 5.41 GB |
| `lm_head` | EXL3 K6, `mul1` | **EXL3 K6, `mcg`** | 0.954 GB | 0.954 GB |
| MTP draft | BF16 | **quantized** (attn K4, MLP K5/K6) | 0.257 GB | 0.257 GB |
| `embed_tokens` / vision / norms | BF16 | same | 3.465 GB | 3.465 GB |
| **total** | | | **30.60 GB** | **21.82 GB measured** |

Enabled by a 30-line patch to exllamav3's `create_q_strategy` adding an
`EXL3_BITS_OVERRIDE` regex->bpw map, since upstream exposes only global `--bits`.
Quantizing the MTP draft follows the owner's own GLM-5.2 result
([`malaiwah/GLM-5.2-EXL3-TR3-MTP78`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-TR3-MTP78)),
where a 3 bpw draft matched BF16 acceptance length (3.06 vs 3.054) at one fifth the size.

Conversion: 38m52s on one GPU, whole-model bpw 4.95 (v1: 4.02), per-block
`rfn 0.0066 / sqnr 45.1 dB` against v1's `0.0155 / 36.5 dB`.

## Fidelity — held-out v3 suite, 136 analysis contexts, 278,392 positions

| candidate | resident weight | mean KLD | bootstrap 95 % CI | median | p99.9 | JSD (bits) | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2 (this iteration)** | **21.82 GB** | **0.008157** | [0.00607, 0.01067] | 0.001529 | 0.475 | 0.002849 | 96.97 % |
| `Qwen/Qwen3.8-27B-FP8` | 30.61 GB | 0.013126 | [0.00981, 0.01709] | 0.002343 | 0.773 | 0.004528 | 96.22 % |
| v1 (all-K4 MLP) | 19.21 GB | 0.030736 | [0.02238, 0.04073] | 0.004218 | 1.758 | 0.010051 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 22.91 GB | 0.094978 | [0.06858, 0.12688] | 0.012911 | 4.509 | 0.028663 | 90.53 % |

Paired, same contexts:

| comparison | mean difference | bootstrap 95 % CI | wins |
|---|---:|---:|---|
| v2 - v1 | **-0.022579** | [-0.03009, -0.01631] | **136/136** v2 |
| v2 - FP8 | **-0.004969** | [-0.00643, -0.00371] | **136/136** v2 |

**v2 is 38 % lower mean KLD than official FP8 at 71 % of its resident weight**, and
73 % lower than v1. Both differences are unanimous across all 136 contexts with
bootstrap intervals excluding zero, and both are three orders of magnitude above the
measured runtime-repeat noise floor (0.000000). Per-stratum means: code 0.00863, encyclopedic 0.00491, literary 0.01500, multilingual 0.00278, scientific 0.00258.

## Performance — every number measured on the same box under the same flags

Median of 3 repeated runs (dispersion <1 %), `--max-num-seqs 8`, greedy, `ignore_eos`,
256 output tokens; prefill measured separately with **exact** token-count prompts built
from the frozen suite.

| configuration | resident weight | KV tokens @8k | TG C1 | TG C4 | TG C8 | PP 2k | PP 6k |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2 + CUDA graphs** | 20.32 GiB | 710,363 | 56.5 | 199.6 | 402.7 | 2,369 | 2,362 |
| **v2 + graphs + MTP-3** | 20.32 GiB | ~ | **113.8** | 206.8 | — | 2,292 | — |
| v1 + CUDA graphs | 17.89 GiB | 736,109 | 55.4 | 190.6 | 428.1 | — | — |
| `unsloth/…-NVFP4` | 21.34 GiB | 1,071,331 | 48.9 | 171.4 | 369.7 | **14,528** | 13,468 |
| `Qwen/…-FP8` | 28.51 GiB | 609,718 | 46.3 | 163.3 | 342.5 | 10,667 | 10,474 |

### Speculative decoding with the quantized draft works

`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, from the server's
own counters over 467 drafts / 1401 draft tokens:

| metric | value |
|---|---:|
| accepted tokens | 815 / 1401 = **58.2 %** |
| mean acceptance length | **2.745** tokens per step |
| per-position acceptance | 77.5 % / 57.2 % / 39.8 % |
| single-stream TG | 56.5 -> **113.8 tok/s (+101 %)** |
| C4 TG | 199.6 -> 206.8 (+3.6 %, batch already saturated) |

This required a metadata repair: exllamav3's `util/add_quant_config.py` walks the main
model only, so the quantized MTP module received **zero** `tensor_storage` entries and
the runtime built BF16 modules for it, failing with
`There is no module or parameter named 'fc.mcg'`.
[`tools/finalize_checkpoint.py`](../tools/finalize_checkpoint.py) scans the shards and
adds the missing entries (8 for this checkpoint).

### The remaining weakness is prefill, and the cause is now measured

We are 4.5-6x slower than NVFP4 at prefill. The vLLM EXL3 backend routes every
non-K6 shard through `exl3_gemm` at any row count, while exllamav3's own
`LinearEXL3.reconstruct_hgemm` switches to `reconstruct_had_slice` + `hgemm` above 1024
rows. Timed on this checkpoint's three real geometries (`tools/prefill_micro.py`,
reconstruct cost included at m>=1024):

| geometry | m=1 | m=32 | m=256 | m=1024 | m=2048 |
|---|---:|---:|---:|---:|---:|
| mlp.gate_proj | 0.33x | 0.60x | 3.91x | 3.85x | 4.57x |
| mlp.down_proj | 0.39x | 0.74x | 4.85x | 4.31x | 5.19x |
| lm_head | 0.38x | 0.71x | 5.49x | 5.07x | 6.07x |

Speedup is `exl3_gemm / (reconstruct + hgemm)`, so >1 favours reconstruction. The
crossover sits between m=32 and m=256: **reconstruction is 3.9-6.1x faster at prefill
shapes and 1.3-3x slower at decode shapes.** For `lm_head` at m=2048 it is 84.9 ms
versus 14.0 ms. A row-count-dependent dispatch in `Exl3LinearMethod._apply_one` should
therefore recover most of the prefill gap without touching decode; this is the next
upstream patch.

## Scorecard against the three axes

| axis | v1 | v2 | verdict |
|---|---|---|---|
| correctness (KLD) | 0.030736 | **0.008157** | 73 % better, and now 38 % better than official FP8 |
| memory | 19.21 GB | 21.82 GB | +2.61 GB, still **under** the 21.92 GB NVFP4-equivalent ceiling |
| TG | 55.4 / 190.6 / 428.1 | 56.5 / 199.6 / 402.7, **113.8 with MTP** | best single-stream of every candidate measured |
| PP | not measured | 2,369 | **worst** of every candidate; cause identified and quantified |
