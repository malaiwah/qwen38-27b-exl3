# Shopping list for iteration 2

Rewritten after the v3 held-out re-measurement and the CUDA-graph landing. Ordered by
expected value, each with the evidence that justifies it and the test that closes it.

## Status of the hard constraints

| constraint | state |
|---|---|
| resident weights <= 21.92 GB (NVFP4-equivalent) | **19.21 GB**, 2.71 GB unspent |
| minimise KLD | **0.030736** held-out; FP8 is 0.0176 better, NVFP4 is 3.09x worse |
| stay in EXL3 | yes |
| coherent + vision | verified under graphs: text, image, coherence all correct |
| **CUDA graphs** | **DONE** — +92 %/+84 %/+98 % decode, exact parity, [PR #314](https://github.com/local-inference-lab/vllm/pull/314) |

Ranking target for iteration 2: close the **0.0176** gap to FP8 while staying under
21.92 GB. That is a 57 % reduction in KLD from 2.71 GB of headroom.

## P0 — measurement precision, because it now gates everything

**P0.1 Replay qualification is 6.54e-04**, ~500x worse than the reference protocol's
1.23e-06. It is only 2 % of our candidate's KLD, so today's rankings hold, but it sets a
**resolution floor of ~1e-3**, and the iteration-1 head-attribution result (6.8e-05)
sits *below* that floor and must be re-derived.
*Fix:* store hidden states in fp32 (2x disk: 42 MB/context) and/or replay in fp32;
re-run `qualify` and target <= 1e-5. Cheap: one capture pass plus one replay.
*Acceptance:* `mean_kld_live_vs_replayed <= 1e-5`, then re-run the head ablation.

**P0.2 Add the four missing strata** — dialogue/instruction, worked
mathematics/reasoning, structured/JSON/tool-calls, news/legal/essays. The reference
suite has 10 strata; we have 5, and the missing ones are exactly where a thinking,
tool-calling model lives. Per-stratum spread within one candidate is already 7x
(literary 0.0574 vs scientific 0.0079), so stratum choice moves the headline
number materially.
*Acceptance:* >= 9 strata, >= 200 source clusters, >= 512 contexts.

**P0.3 Long-context tier.** Everything so far is 2048 tokens; the model supports
262,144. Chunk accumulation is already implemented (`--max-batched-tokens`,
`--chunk-accumulate`), and the architecture keeps KV cheap (only 16 of 64 layers cache).
Costs are computed in [20](20-context-extension-and-k3-gap.md): a 131k tier with 8
contexts is 43 GB for 4 candidates and yields more scored positions than the entire 2k
tier.
*Acceptance:* KLD reported at 2k / 32k / 131k, with the depth trend for each candidate.

**P0.4 Hygiene the reference protocol has and we do not:** MinHash near-duplicate pass
within the suite, benchmark-leakage scan against HumanEval / MMLU / GPQA-Diamond, and
live-logit sentinels so runtime variation stays separable from replay defects.

## P1 — recipe: spend the 2.71 GB on the MLP

**P1.1 Error-driven allocation.** Convert once more at `-b 5`, then
`util/measure.py -i k4 k5 -r bf16` and `util/optimize.py -m measurement.json -b <target>`,
which allocates by measured `dkld/dbits`. This replaces hand-picking with the same
metric we publish. ~35 min conversion + ~10 min measure.
*Budget:* all 64 `down_proj` K4->K5 is +0.713 GB, K4->K6 is +1.426 GB; both fit.
*Evidence:* `down_proj` carries the largest per-tensor proxy error in every layer
(2.5e-3 vs 1.1e-3 for `gate_proj`).
*Acceptance:* held-out mean KLD below 0.030736 with resident weights <= 21.92 GB, paired
CI excluding zero against the current checkpoint.

**P1.2 Calibration corpus.** Free in VRAM. The default 250x2048 generic mix contains no
chat-template-formatted text, no reasoning traces and no tool calls. Re-convert with a
domain-representative set — and measure only on held-out data, never on the calibration
text (the mistake v2 made).

**P1.3 `mcg` for every K6 tensor.** `-cb mcg` is honoured for decoder tensors but
`lm_head` and MTP were written `mul1`, and the B12X native K6 kernel requires `mcg`, so
the serialized K6 head misses the fast path. Perf only.

**P1.4 Do not** spend on attention (online K6 already beat BF16 attention) or on the
head (6.8e-05, pending P0.1 re-derivation).

## P2 — runtime

**P2.1 Land PR #314** (graphs) and #312 (BF16 fallback). Follow up with the
`vllm.utils.flashinfer has no attribute mm_mxfp8` finding: MXFP8 is non-functional in
the r34 build, so the documented "128-unaligned shards retain MXFP8" path cannot execute
and should probe-and-degrade to BF16.

**P2.2 Widen graph capture to `FULL_AND_PIECEWISE`** if prefill capture proves to use
the same size list — the patch deliberately refuses it today.

**P2.3 FP8 KV cache** to match both reference quants, then re-measure with KV dtype
pinned identically across teacher and candidates.

**P2.4 MTP speculative decoding.** The draft head is BF16 and untouched; vLLM enables it
with `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Untested
here, and the graph row plan already accounts for spec-decode row counts.

## P3 — publication

**P3.1** Variant branches on the model repo, each with its own receipt
(`quantization_config.json`, fidelity report, serve command, measured footprint).
**P3.2** Keep the dataset in step: every new candidate's hidden states appended, so
third parties can re-derive each published number.
**P3.3** Regenerate the chart from receipts only.

## Sequencing

1. P0.1 replay precision (gates the resolution of everything else).
2. P1.1 second conversion + error-driven allocation -> candidate B.
3. Measure candidate B on the v3 analysis partition; compare paired against the noise
   floor (0.000000) and against FP8.
4. P0.2 + P0.3 suite expansion, then re-measure the winner.
5. P2.3 FP8 KV, P1.3 `mcg`, re-measure throughput with graphs.
6. Freeze the recipe, open the qualification partition **once**, publish with receipts.
