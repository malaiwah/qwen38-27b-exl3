# Distribution-fidelity protocol v2 (hidden-state replay)

Adopted from
[`rtx6kpro:models/kimi-k3/distribution-fidelity-1024x2048.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/distribution-fidelity-1024x2048.md),
which supersedes the single-window `prompt_logprobs` protocol used in iteration 1.

## Why it is better

| iteration-1 protocol | v2 protocol |
|---|---|
| one 2048-token window, 2047 positions | many contexts x 2047 positions, stratified by content type |
| mean KLD only | mean, median, p95/p99/p99.9, max, Jensen-Shannon, top-1 agreement |
| no confidence interval | source-cluster bootstrap 95 % CI, per-stratum means |
| stores full-vocabulary logits (2.0 GB per context) | stores hidden states (21 MB per context), 96x smaller |
| head error and body error are entangled | one shared BF16 head replays both operands, so body and head can be attributed separately |

The storage difference is what makes many contexts affordable: full-vocabulary
fp32 logits for our vocab are `2047 x 248320 x 4 B = 2.03 GB` per context, versus
`2047 x 5120 x 2 B = 21 MB` of BF16 hidden state.

The reference protocol qualified this replay path on Kimi-K3 at mean
`KL(live logits || replayed logits) = 1.229e-6` with top-1 agreement 0.999954, so
replay error is orders of magnitude below the candidate differences being resolved.

## Our implementation

[`tools/fidelity.py`](../tools/fidelity.py), four subcommands:

- `suite` — build and freeze the context set: N contexts x 2048 tokens drawn from
  the six exllamav3 calibration corpora as content strata (encyclopedic, web, code,
  technical, multilingual, short-form), exact-token dedup, per-context and
  suite-level SHA-256, source-cluster labels for bootstrap.
- `capture` — in-process vLLM, `enforce_eager`, `max_num_seqs=1`,
  `max_num_batched_tokens=2048` so each context prefills in one chunk. A
  **forward hook on the final RMSNorm** captures `[2048, 5120]` and stores
  positions `0..2046` as BF16. No vLLM patch is required — the reference
  implementation needs a runner patch (`VLLM_KLD_HIDDEN_CAPTURE_DIR`, ~370 lines)
  because it captures inside a TP16 pipeline; at TP1 a hook is sufficient.
  Installed through `llm.collective_rpc` because the engine core runs in its own
  process (needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to ship the callable).
  Hook target on this model: `language_model.model.norm`.
- `replay` — exact two-pass full-vocabulary metrics against the reference
  captures, through one BF16 LM head: pass 1 accumulates `logsumexp` normalizers
  and argmax per position over vocabulary chunks; pass 2 accumulates
  `KL(reference || candidate)` and Jensen-Shannon (in bits) in float64. Never
  top-k. Emits per-context reports, strata means, bootstrap CI, tail quantiles,
  and the 20 worst contexts.
- `paired` — A/B on shared contexts: mean and median difference, cluster
  bootstrap CI of the difference, win counts, largest disagreements. Negative
  `difference_a_minus_b` favours A.

## Suite in use

| property | value |
|---|---|
| contexts | 74 |
| tokens per context | 2048, scored positions 2047 |
| total scored positions | 151,478 (74x the iteration-1 protocol) |
| strata | encyclopedic 16, code 16, technical 16, short-form 16, multilingual 9, web 1 |
| suite token SHA-256 | `2c943c3972e7d1ca68b8acf5c8ad6f0426c2d5d218c80a9be07166e0646a915a` |
| hidden width / vocab | 5120 / 248320 |

The web and multilingual strata came up short because those corpora do not split
into enough >=2048-token units under the current splitter; fixing the splitter is
the first improvement to make, and it does not invalidate existing captures since
contexts are content-hashed and additive.

## What it lets us answer that v1 could not

1. **Is `lm_head` the sensitive tensor?** Replay one candidate's hidden states
   through the BF16 head and through the quantized head; the difference is the
   head's contribution, with the body held byte-identical.
2. **Where does error concentrate?** Per-stratum means and p99.9 tails expose
   whether a recipe fails on code, multilingual, or reasoning text rather than
   uniformly.
3. **Is a difference real?** Bootstrap CIs over source clusters, plus paired
   per-context differences, instead of one number with no error bar.
