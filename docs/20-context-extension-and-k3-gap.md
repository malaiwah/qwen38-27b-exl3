# How far we can extend, and how far we are from the reference protocol

> **Historical.** The chunk-accumulation and strata items were closed by the v5 suite
> ([42-kld-method.md](42-kld-method.md) / [33-evidence-volume-and-intervals.md](33-evidence-volume-and-intervals.md));
> the long-context fidelity tier is still open and tracked in [29-plan-and-loose-ends.md](29-plan-and-loose-ends.md).

## Context length: what the current measurement covers and what it can

Every number published so far is at **2048 tokens**. That is the reference protocol's
choice too, and it explicitly declares long context out of scope. For this model it
leaves 99.2 % of the supported window unmeasured (262,144 native).

Costs per context, measured or computed from measured rates:

| context length | hidden state per candidate | replay time per context | KV cache during capture (16 full-attn layers × 4 KV heads × 256 dim × 2 B × 2 for K+V) |
|---:|---:|---:|---:|
| 2,048 | 21 MB | 2.2 s | 0.13 GiB |
| 8,192 | 84 MB | 8.8 s | 0.5 GiB |
| 32,768 | 335 MB | 35 s | 2.0 GiB |
| 131,072 | 1.34 GB | 141 s | 8.0 GiB |
| 262,144 | 2.68 GB | 282 s | 16.0 GiB |

The architecture helps: only 16 of 64 layers keep a KV cache, the other 48 are linear
attention with constant state, so KV at 262k is 16 GiB rather than 64 GiB.

**Feasible today**, with the fix already landed: `fidelity.py capture` gained
`--max-batched-tokens` and `--chunk-accumulate`, which concatenate chunked-prefill
forwards in arrival order. Without it a context longer than one prefill chunk was
silently truncated to the largest single forward.

Practical tiers, sized against 500 GB of free scratch and one 96 GB GPU:

| tier | contexts | positions | disk for 4 candidates | wall clock |
|---|---:|---:|---:|---:|
| 2k (current) | 181 | 370,507 | 15 GB | ~20 min |
| 32k | 24 | 786,408 | 32 GB | ~40 min |
| 131k | 8 | 1,048,568 | 43 GB | ~70 min |
| 262k | 4 | 1,048,572 | 43 GB | ~90 min |

The BF16 reference is the binding constraint at 262k (55.6 GB weights + 16 GiB KV +
activations); the quantized candidates have ample headroom. A 131k tier with 8
contexts already yields **more scored positions than the entire 2k tier** and would
answer the question nobody has answered for this model: does 4-bit MLP degrade with
depth?

## Distance from the reference (Kimi-K3) protocol

| dimension | reference | ours (v3) | gap |
|---|---|---|---|
| contexts | 1,024 | 181 (136 analysis) | **5.7x fewer** |
| scored positions | 2,096,128 | 370,507 (278,392 analysis) | **7.5x fewer** |
| context length | 2,048 | 2,048 | same |
| strata | 10, including dialogue/instruction, worked mathematics, structured/tool-calls, news/legal/essays | 5: literary, code, scientific, encyclopedic, multilingual | **missing the 4 strata a thinking, tool-calling model is actually used for** |
| source clusters | 827 dataset-qualified families | 41 | **20x fewer**, so the bootstrap resamples fewer independent units |
| deduplication | exact token + MinHash shingle within the suite | exact token only | missing near-duplicate pass |
| benchmark-leakage scan | HumanEval, MMLU, GPQA-Diamond, 0 overlaps | none | missing |
| calibration contamination scan | not applicable (official checkpoint) | fixed-stride scan said 0; all-position 12-token audit later found 2/41 affected source documents | corrected subset preserves ranking |
| partitions | 768 analysis / 256 qualification, freeze-before-open | 136 / 45, declared but not yet exercised | must actually freeze in iteration 2 |
| sentinels | 64 contexts x 3 repeats, hidden **and** live logits | 32 contexts x 3 repeats, hidden only | live-logit sentinels missing |
| measurement noise floor | 0.0032 (TP16 nondeterminism) | **0.000000** (bit-deterministic TP1) | **ours is better** |
| replay qualification | 1.229e-06 mean, top-1 0.999954 | 6.54e-04 mean, top-1 0.98999 | **~500x worse — our biggest gap** |
| metrics | mean/median/p95/p99/p99.9/max KLD, JSD, top-1, micro+macro, depth buckets, per-stratum, worst-20, paired cluster bootstrap | same minus depth buckets and the micro/macro split | minor |
| capture mechanism | patched vLLM runner, TP16 assembly, rank-0 writes | forward hook via `collective_rpc`, TP1 | equivalent at TP1, no patch needed |
| published artifact | HF dataset with checksums, image digest, source commits | [HF dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3) with the same | parity |
| interpretation guardrails | no universal labels; KLD is not a capability eval | same, stated | parity |

### Ranked closing actions

1. **Replay precision** (biggest): store fp32 hidden states or replay in fp32, then
   re-run qualification. Until this is ~1e-5, differences below 1e-3 are noise — which
   currently includes the entire head-attribution result.
2. **Add the missing 4 strata**: dialogue/instruction, worked mathematics/reasoning,
   structured/JSON/tool-calls, news/legal/essays. This is a corpus-sourcing job, not a
   code job, and it is where a thinking model's behaviour actually lives.
3. **Scale to ~512 contexts and >=200 source clusters** — cheap: 20 minutes of GPU per
   candidate and 4 GB per candidate at 2k.
4. **Long-context tier** at 32k and 131k, now unblocked by chunk accumulation.
5. **MinHash near-duplicate pass** and a **benchmark-leakage scan** against HumanEval,
   MMLU and GPQA.
6. **Live-logit sentinels** so runtime variation and replay defects stay separable.
