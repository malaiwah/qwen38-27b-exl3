# Long-context retrieval: 8/8 at 195,000 tokens — and a retraction

**Date:** 2026-08-19 (late)
**Profile:** `fidelity` (all-trellis, ANY_BITS, 238,400 ctx, fp8_e4m3 KV)
**Data:** `receipts/longctx-needles-fidelity-v2-2026-08-19.json`

## Result

| depth level | retrieved |
|---|---|
| 2,000 tokens (control) | **8/8** |
| 100,000 tokens | **8/8** |
| 195,000 tokens | **8/8** |

24/24 needles across all three levels, including the deepest position the served
context allows room to test.

## Retracting the earlier run

The first needle probe reported **5/8 at 100k and 4/8 at 195k** and was published in
`receipts/longctx-needles-fidelity-2026-08-19.json`. Those numbers were **wrong, and
the fault was mine, not the model's**.

The harness called raw `/v1/completions` with a small `max_tokens`. Qwen3.8 is a
reasoning model: it emits `<think>` before answering, so the token budget was consumed
by reasoning and the answer never appeared. The probe was measuring its own budget, not
retrieval.

The tell was in the run itself: **the 2k-token control also missed 1/8**. A control at
2k tokens cannot fail for context-length reasons — that was the harness announcing its
own bug, which is why the original result was recorded as "not interpretable" rather
than reported as a fidelity finding.

Fix: chat endpoint, `chat_template_kwargs={"enable_thinking": false}`, 48-token budget.
This is the *third* time this exact trap has bitten a harness in this project (vision
check, needle probe, and the essay sanity check) — it is now noted in the harness
library so the next probe inherits the fix.

The raw first run is kept in-repo for provenance rather than deleted.

## What this does and does not establish

**Does:** the trellis weights, the fp8 KV cache and the linear-attention path together
preserve exact retrieval at 195k tokens — a 95× extrapolation beyond the 2,051-token
prefill benchmark and 380× beyond the 512-token KLD capture. The KLD numbers are
measured at short context; this is independent evidence that nothing degrades
catastrophically at the context lengths the profile advertises.

**Does not:** needle retrieval is a *proxy*, and an easy one. 8/8 does not mean
long-context reasoning is unimpaired, only that a verbatim fact at a known depth is
recoverable. A harder probe (multi-hop, or needles requiring aggregation) could still
find degradation.
