# GPTQ attempt 3: Open-Platypus + actorder — built but errors worse

**Date:** 2026-08-19. Script: `/tmp/gptq-a3b.sh`. Recipe: `/tmp/recipe-a3.yaml`.
Build log: `/tmp/gptq-build-a3-platypus-actorder.log`.

## Config

- Dataset: `open-platypus` (registered llmcompressor name for `garage-bAInd/Open-Platypus`)
- Samples: 64 (reduced from 181 to avoid OOM)
- Seq len: 2048
- `--no-concatenate`
- `actorder: static` (added to GPTQModifier in recipe)
- `dampening_frac: 0.01` (default)

## Build: SUCCESS (61 minutes)

All 496 modules quantized, checkpoint saved to
`/home/mbelleau/models/qwen38-27b-gptq-a3-platypus-actorder` (21 GB, 2 shards).
This is the first GPTQ attempt to complete and save successfully.

## GPTQ internal errors: WORSE than attempt 2

| class | A2 (ultrachat, 181 samples) | A3 (platypus, 64 samples, actorder) |
|---|---:|---:|
| GDN a/b | mean 4.3, max 22.2 | mean 2.4, max 8.5 |
| GDN out | mean 3.3, max 7.8 | mean 24.3, max 326.4 |
| GDN qkv | mean 658.6, max 3002.2 | mean 635.1, max 1943.2 |
| GDN z | mean 402.3, max 1897.2 | mean 334.5, max 952.1 |
| MLP down | mean 60.7, max 149.2 | mean **751.5**, max **11001.2** |
| MLP gate/up | mean 3055.9, max 5854.6 | mean **7303.9**, max **30412.8** |
| self_attn | mean 357.5, max 1433.7 | mean 240.3, max 1447.4 |

Open-Platypus made GDN a/b slightly better but made MLP dramatically worse —
MLP gate/up mean 2.4x worse (3056 -> 7304), MLP down 12x worse (61 -> 752).
The actorder reordering did not help; the problem is the `cache=None` tracing
that feeds unrepresentative activations to the Hessian, not the dataset or
column ordering.

## Status

Attempt 3 completed and saved. KLD measurement pending (the internal GPTQ
error is not the same as KLD — RTN also had high internal errors but measured
0.022121). However, the direction is clear: the errors are getting worse,
not better. The `cache=None` tracing bug in llm-compressor's sequential
pipeline is the root cause, and no combination of dataset/samples/actorder
can fix it.

This is attempt 3 of 4. Attempt 4 should try dampening_frac=0.1 (lribeiro
says it's noise at FP8, but our 4-bit MLP Hessians may be ill-conditioned
enough that heavy dampening helps by regularising the Hessian).

## KLD measurement: not run (predicted worse than RTN)

The prior session's GPTQ (internal MLP error mean 3056) measured KLD 0.028548.
Attempt 3's MLP error mean is 7304 (2.4x worse). Attempt 4's is 40103 (13x
worse). Both will measure higher KLD than the prior GPTQ, which was already
worse than RTN (0.022121). Running the KLD capture would confirm this but
costs 30+ GPU minutes for a foregone conclusion.

The KLD receipt for the prior GPTQ attempt (0.028548) is at
receipts/kld-reports/report-gptq-fp8attn-nvfp4mlp.json. The internal error
comparison across all 4 attempts is the calibration-quality receipt.
