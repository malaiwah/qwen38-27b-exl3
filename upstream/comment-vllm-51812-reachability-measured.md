# POSTED comment on vllm-project/vllm#51812 — "[Bugfix] Align Qwen GDN gates with speculative tokens"

<!--
Target: https://github.com/vllm-project/vllm/pull/51812 (merged 2026-08-11, 5af7c8dad798)
Status: POSTED 2026-08-16 -> https://github.com/vllm-project/vllm/pull/51812#issuecomment-5307331574 (approved by Main).
Value: the fix is already merged, so this is not a bug report. It is the missing datum: the batch
composition that violates the leading-speculative-token assumption occurs on real traffic in a
shipped configuration, at a measured rate, with the reorder in place. Useful to whoever backports,
and to the open issues in the same family (#49918, #51562, #47123).
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE; post everything after it verbatim, and nothing above it ===== -->
This landed before we could contribute to it, so this is not a report — it is the reachability datum
the PR did not need but a backporter might: **the batch composition your fix guards against occurs on
real traffic, in a shipped configuration, at 0.515 events per thousand GDN metadata builds.**

We instrumented `vllm/v1/attention/backends/gdn_attn.py` on a Qwen3.5-architecture hybrid to count,
per metadata build, which branch ran and — on the mixed branch — whether `spec_token_indx` was the
identity, i.e. whether the speculative tokens were actually the leading tokens of the batch. The
instrument reads only `spec_sequence_masks_cpu` and `query_start_loc_cpu`, both already on the host,
so it adds no sync and cannot perturb the schedule. The additivity of the instrument against the
vendored file was checked opcode-by-opcode and is reversible to the original sha256.

Three arms, one physical RTX 5090, MTP-3, 8 concurrent streams, adversarial-by-design traffic:

| arm | flags | metadata builds | on the gather branch | misordered | rate |
|---|---|---|---|---|---|
| R1 | prefix caching **off**, `--max-num-batched-tokens 2048` | 3,329 | 2,112 (63 %) | **0** | < 0.90 / 1000 (95 % upper bound, rule of three) |
| R3 | prefix caching **off**, `--max-num-batched-tokens 512` (amplifier) | 8,065 | 6,915 (86 %) | **0** | < 0.372 / 1000 |
| R2 | `--enable-prefix-caching --mamba-cache-mode align`, 8,192-token window | 5,825 | 2,211 | **3** | **0.515 / 1000 builds, 1.357 / 1000 gather-branch builds** |

Two readings, and the second is the interesting one:

1. **The batch reorder does most of the work.** The gather branch is not rare — it runs on 63–86 % of
   metadata builds at 8 streams — and in 9,027 gather-branch builds without prefix caching the
   speculative tokens were leading **every single time**. `reorder_batch_to_split_decodes_and_prefills`
   moves every prefill-shaped row behind the speculative decodes, including a brand-new short prompt
   (verified by executing the image's own reorder over seven hand-built compositions).
2. **What the reorder does not protect against is a non-speculative row sharing a low region with a
   speculative one.** All three events had the same composition: six speculative decodes and one
   non-speculative row, with the first misordered token at index 20 — tokens 0–19 were five
   speculative decodes of four tokens, token 20 began a four-token non-speculative row, and the sixth
   speculative decode's four gate rows sat behind it and were read from the wrong tokens
   (`max_displaced_spec_tokens = 4`).

The mechanism that supplies the pseudo-speculative row is the scheduler's speculative padding: a
request admitted needing exactly one new token is padded to `1 + num_spec_tokens` and
`[-1] * num_spec_tokens` is written into `scheduled_spec_decode_tokens`
(`vllm/v1/core/sched/scheduler.py:865-878` and `:1052` in our build), which makes the runner classify
it as speculative (`gpu_model_runner.py:2240-2249`) while the reorder leaves it in region 1 because
it is not done prefilling. **A full prefix-cache hit is exactly how `num_new_tokens == 1` arises**, so
this needs prefix caching on — which is why R1 and R3, with the cache off, are clean, and why we
would expect the rate to be workload-dependent rather than fixed. The non-speculative row ahead of it
was a chunked prefill whose final chunk was 2–4 tokens.

We also predicted, from source, a second mechanism that needs no prefix caching — a speculative
decode clamped by `--max-num-batched-tokens` to exactly one token, which silently reclassifies it as
non-speculative (`scheduler.py:511-518` then `:637-642`) — and gave it a 4x amplifier in R3. It
produced **zero** events in 9,027 gather-branch builds (95 % upper bound 0.332 per thousand), so on
this build it is bounded-rare rather than shown reachable. Recording that because it is the mechanism
that would matter on ordinary traffic, and our measurement does not support claiming it.

What we deliberately do **not** claim: that those three miscomputed forward passes changed any
emitted token. Your own measured per-event effect (mean absolute chosen-logprob error 0.002755) is
about 30x below our run-to-run floor on this build (0.08231278, two identically-flagged fresh
servers), so an A/B logprob comparison at 8 streams is pre-determined to report "below resolution"
whether the module matters or not. We therefore measured whether the defective path *executes*, and
report only that.

Relevant to the neighbouring open issues in this family, all of which turn on a row being classified
as a decode when it is not: #49918, #51562, #47123.

Method, counts, the pre-registered decision rule (fixed before the GPU window opened), the reorder
simulation and the raw per-worker dumps: `receipts/gdn-gate-concurrency.json` and
`receipts/gdn-gate-raw/` in <https://github.com/malaiwah/qwen38-27b-exl3>. Build is a downstream fork
(`0.11.2.dev280+…20260810.r34`, upstream integration tree `4d006a43`) that predates this PR;
`gdn_attn.py`, `flashinfer.py`, `backend.py` and `gpu_model_runner.py` are unmodified vendored
upstream code there, and this PR does not touch `gdn_attn.py`, so the batch composition we measured
is the same patched or unpatched.
