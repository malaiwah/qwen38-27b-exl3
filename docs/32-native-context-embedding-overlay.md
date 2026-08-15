# Native 262,144 context on a 32 GB card, by narrowing the one table nobody looked at

[docs/30](30-iteration-4-context-edition.md) closed with native context proven out of reach:
8.18 GiB of KV needed against 7.55 available, gap 0.63 GiB, and the only lever large enough
named as the embedding table. This is that lever, built and measured.

## What was done

The input embedding table on this model is **248,320 x 5,120 in BF16 = 2.543 GB resident** —
second only to the MLP stack, larger than the entire attention stack after quantization, and
the only large tensor in the model that is never multiplied. It is a gather: one row per token,
no accumulation, no tensor cores.

Narrowed to **per-row symmetric int8**: each row (one token's vector) keeps its own scale, so
the quantization tracks the thing that actually varies. int8 rather than FP8 on purpose —
E4M3 has three mantissa bits against int8's seven, and FP8's throughput advantage is
irrelevant in a lookup.

Enabled by `VLLM_EXL3_EMBED_BITS=8`. Off by default.

## What it costs and what it buys

| | BF16 embeddings | int8 embeddings |
|---|---:|---:|
| embedding table | 2.543 GB | **1.272 GB** |
| resident weights | 19.31 GiB | **18.13 GiB** |
| max context, 32 GB card, multimodal | 229,376 | **262,144 (native)** |
| KV allocated at that length | 240,080 tokens | **279,007 tokens** |
| mean KLD, v3 analysis | 0.009673 | 0.009738 |
| multimodal, 30 synthetic cases | 24/30 | **24/30, identical** |

Cost: **+0.000065 mean KLD**, 95 % CI [+0.0000046, +0.00013], 49/136 contexts — about 0.7 % of
this build's divergence, and roughly half what its own K6 `lm_head` costs.

**Verified by use, not allocation.** At `--max-model-len 262144` on a 5090-sized budget with
vision enabled, a planted code was retrieved exactly at depths 0.1, 0.5 and 0.9 from
**227,334-token prompts**, 94.6-95.0 s each, ~2,400 tok/s prefill.

## The blocker was a missing argument, not a missing feature

`VocabParallelEmbedding` already supports quantization: it asks
`quant_config.get_quant_method(self, prefix)` and requires the returned method to implement
`embedding()`. The model files never pass a quant config:

```python
self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
```

Two lines in `qwen3_5.py` and `qwen3_5_mtp.py` fix that. Backends that do not implement
`embedding()` are unaffected — the layer falls back to `UnquantizedEmbeddingMethod` when
`get_quant_method` returns `None`.

The MTP file matters more than it looks: **the draft head builds a second full embedding
table**, which is most of what enabling MTP costs in resident memory — far more than the draft
weights themselves. Narrowing both is what brought MTP within 0.46 GiB of native context.

## What still does not fit

MTP and native context. At 262,144 the engine needs **8.83 GiB** of KV with one draft token and
**9.13 GiB** with three, against **8.37 GiB** available. Short by 0.46 and 0.77 GiB. The
remaining levers are all bad trades: attention at K4 would free ~0.8 GiB and cost ~0.018 KLD,
and the vision tower cannot be quantized in this runtime at all. So the choice on a 32 GB card
is native context **or** speculative decoding at 196,608 — stated on the card rather than
papered over.

## Loose ends this created

- The embedding overlay and both model patches should go upstream together; the model change is
  independently useful because it makes the embedding table reachable by *any* backend.
- Only int8 is implemented. A 4-bit table with per-row scales would free another 0.6 GiB and
  might make MTP-1 fit at native context; the error would need measuring, and a gather of
  packed nibbles needs a small kernel.
- The draft head's duplicate embedding table is arguably a bug worth reporting on its own: it
  doubles the cost of MTP for no benefit when the vocabulary is shared.
