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

## int4 for the draft table: free, but it does not unlock MTP at native context

Four bits per element only works with fine-grained scales, and even then it is far coarser:
group-128 int4 carries **13.2x the relative error of per-row int8** (12.6 % against 0.95 %,
measured on the real distribution). Too much for the input path, where every token pays it —
but the MTP draft head is a different matter, because its output is *verified* before it is
accepted, so a coarser draft costs acceptance rate rather than correctness.

Measured with `VLLM_EXL3_MTP_EMBED_BITS=4` on the same server, draft depth 3:

| draft table | resident | TG C1 | TG C4 | drafted | accepted | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| int8 | 1.272 GB | 101.5 | 274.4 | 1,434 | 804 | 56.1 % |
| **int4 group-128** | **0.656 GB** | **104.0** | **344.5** | 1,428 | 809 | **56.7 %** |

**Acceptance is unchanged within noise and throughput is equal or better**, so the draft table
can be int4 for free — 0.62 GB of resident memory back.

**It still does not make MTP fit at native context**, and the reason is informative: available
KV stayed at 8.35-8.36 GiB whether the draft table was int8 or int4, against 8.83 GiB needed at
depth 1 and 9.13 at depth 3. Narrowing the draft's *weights* by 0.62 GB moved the KV budget by
nothing measurable. **MTP's cost at native context is not its weights** — it is the draft's own
KV plus its share of the profiled activation peak. So the choice on a 32 GB card remains native
context **or** speculative decoding at 196,608, and the lever that would change that is not the
embedding table.

## Loose ends this created

- The embedding overlay and both model patches should go upstream together; the model change is
  independently useful because it makes the embedding table reachable by *any* backend.
- int4 is implemented for the draft table and measured free; the main table stays int8
  because 12.6 % relative error on every input token is not a trade worth making. A 6-bit
  option would sit between them if anyone needs 0.3 GB more.
- The draft head's duplicate embedding table is arguably a bug worth reporting on its own: it
  doubles the cost of MTP for no benefit when the vocabulary is shared.
