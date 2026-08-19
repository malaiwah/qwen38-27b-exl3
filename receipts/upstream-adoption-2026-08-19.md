# Upstream survey: what was adopted, what was ruled out, and why

**Date:** 2026-08-19 (evening)
**Trigger:** user request to survey open/commented issues and PRs across the stack for
anything worth adopting. Image is pinned at r34 (built 2026-08-10); anything merged or
opened after that date is absent from it by construction.

## ADOPTED into the deployment (bind-mounted patches)

### 1. upstream vllm-project#51812 — GDN gates misaligned with speculative tokens — **LIVE BUG, fixed**

`qwen_gdn_linear_attn.py` passes gate tensors `a`/`b` **un-sliced** to
`fused_sigmoid_gating_delta_rule_update` for the spec-token path whenever a batch mixes
spec and non-spec rows. With MTP active, spec tokens then run all 48 GDN layers with
wrong-row gates. Verified ABSENT from the r34 fork copy (all five fixed lines missing).

Reachability on our serving: **real under concurrency**. Pure spec-decode batches take
the safe branch (why single-stream benches never showed it); a batch mixing a prefill
with decoding requests takes the buggy one. Adopted as a new bind-mount
(`qwen_gdn_linear_attn_patch.py`, upstream hunks ported verbatim with provenance
comments). Post-adoption: throughput gate 9/9 PASS, C1/C4/C8 clean
(295.6 aggregate tok/s at C4), stress ROBUST.

### 2. upstream vllm-project#51113 via fork PR #393 pick-1 — mamba align-mode chunk split — **ported defensively**

Fixes silent wrong tokens (HTTP 200, no crash) when mamba `align`-mode prefill chunks
end mid-block past `last_cache_position`. Our `scheduler_patch.py` carried the unfixed
logic verbatim (`prefill_end`: 0 hits). **Unreachable today** — `mamba_cache_mode`
defaults to `"none"` and our launcher never sets it — but the guard flips the moment
anyone enables `align` for mamba prefix caching, which is attractive for exactly our
long-RAG workload. Both upstream hunks ported into `scheduler_patch.py` with comments.

## RULED OUT, with reasons (so nobody re-litigates from memory)

| item | ruling |
|---|---|
| fork #312 (BF16 overlay fallback) | Absent from our patch, **unreachable on the pinned checkpoint**: every attention shard encodes K6 successfully today, so the fallback never fires. Adopt only if the checkpoint changes. |
| fork #316/#314 | **Already carried** — our patch contains the reconstruct+hgemm dispatch and `_EXL3_GEMM_PRIMED_SIGNATURES` (~140 boots, no autotune-under-capture faults). |
| fork #397 (shared reconstruct arena) | **Queued for adoption** (todo). Explains our recon-vs-b12x prefill contradiction with #318's table; also would lift the 512 MB lm_head exclusion. |
| fork #398 (flashinfer wrapper keying) | **Not exposed** on this config: its signature is a deterministic boot-time death in `warmup_kernels` under V2+MTP, which we have never hit (~140 boots), and C1/C4/C8 + image stress run clean. Adopt on first symptom; it is one bind-mountable file. |
| b12x #182 (FC2 'ultra' tile) | **MoE-only in practice** (FC2 phase-2 tile upgrade) and fails LOUD (`ValueError` at plan time), never observed on our dense path. Not our path. |
| upstream #49171 (skip logits for unfinished prefills) | **Not needed on V2 runner**: `gpu/model_runner.py` gathers `hidden_states[input_batch.logits_indices]` — logits are computed only at sampled rows, so per-chunk lm_head cost during a 65-chunk 200k prefill is already ~zero. |
| LMCache #4600 | We do not serve with LMCache; offload tier not in the qualified profile. |
| upstream #52530 / #47272 | KV-pool admission/null-block hardening; our serving already fails oversized requests with a clean 400. Upstream-only value. |
| K4/trellis KV cache ("crazy idea") | **Not feasible for active KV**: trellis encode is offline-speed vs KV written per token, and no attention kernel reads trellis-packed KV — per-step dequant would cost reconstruct-scale work per layer. The 2-stage variant (freeze + batch-encode cold blocks) is a cache/offload-tier design, adjacent to LMCache #165, not an active-attention format. Out of scope on one card while `turboquant_*` exists untested. |

## Context that reframes the KV question

The fork ships sub-8-bit KV formats we had never exercised: `turboquant_k8v4 / 4bit_nc /
k3v4_nc / 3bit_nc` (dedicated Triton backend, `supports_spec_as_decode=False` — MTP
behaviour must be measured, not assumed) plus `int4_per_token_head` and bare `nvfp4`.
Measured rung 16→8-bit on the fidelity config: KLD 0.003437 → **0.003729 (+8.5%)** —
the first calibration point for how much 8→4-bit might cost. Test matrix registered.
