# Upstream issues and PRs: audit, CodeRabbit fixes, and maintainer responses

**Date:** 2026-08-20. All items retrieved via `gh api` as malaiwah.

## Summary

Audited 33 upstream items across 6 repos. Found 3 PRs with CodeRabbit reviews requiring fixes, 2 items with maintainer feedback requiring responses, and 1 PR (#3058) where prior fixes were already pushed but needed reply comments.

## Items audited

### local-inference-lab/vllm (18 items: 8 issues, 10 PRs)

| # | type | state | title (truncated) | comments | CR | action |
|---|---|---|---|---:|---|---|
| 311 | issue | open | EXL3 online overlay MXFP8 fallback raises | 0 | no | none |
| 312 | PR | open | keep online-overlay shards BF16 | 2 | skip | none |
| 313 | issue | open | GG v20 r34 truncates Qwen3.8 vision | 0 | no | none |
| 314 | PR | open | CUDA-graph decode for dense checkpoints | 6 | skip | none |
| 316 | PR | open | dispatch prefill-shaped GEMMs to reconstruct+hgemm | 2 | skip | none |
| 318 | PR | open | route K6/MCG shards away from B12X decode | 3 | skip | none |
| 392 | issue | open | Cherry-pick upstream #51113 (mamba align) | 1 | no | none |
| 393 | PR | open | Cherry-pick #51113 and #51812 into dev/gilded-gnosis | 3 | yes | none needed |
| 396 | issue | open | V2 model runner replay speculator captured data | 0 | no | none |
| 397 | PR | open | share one reconstruct-scratch arena | 1 | skip | none |
| 398 | PR | open | fix(flashinfer): key persistent decode wrappers | 1 | skip | none |
| 402 | issue | open | Hybrid scheduler silently corrupts output | 0 | no | none |
| 403 | PR | open | Gate hybrid+connector divergent local-hit path | 3 | skip | none |
| 406 | issue | open | b12x K6 gate routes lm_head to slower kernel | 1 | no | none |
| **436** | PR | open | online K-quant embedding table | 4 | **yes** | **FIXED** |
| **439** | PR | open | skip speculative decoding for single-token | 1 | **yes** | **FIXED** |
| 440 | issue | open | MTP draft loop runs eager on V1 runner | 0 | no | none |
| 442 | issue | open | DSpark: qwen3 external drafts misrouted | 0 | no | none |

### vllm-project/vllm (2 issues)

| # | state | title | comments | action |
|---|---|---|---:|---|
| 52871 | open | Forward-pass CUDA OOM kills EngineCore | 0 | none (no response) |
| **52872** | open | GDN/mamba-hybrid profiled peak activation | 1 | **REPLIED** to Tejas-Raj01 |

### vllm-project/llm-compressor (1 issue, 1 PR)

| # | type | state | title | comments | reviews | action |
|---|---|---|---|---:|---:|---|
| 3057 | issue | open | sequential pipeline cannot trace unwrapped forwards | 2 | — | none (maintainer acknowledged) |
| **3058** | PR | open | Resolve decorated forwards without functools.wraps | 3 | 2 | **REPLIED** (fixes already pushed in b9b6ff2) |

### local-inference-lab/b12x (4 issues)

| # | state | title | comments | action |
|---|---|---|---:|---|
| 232 | open | W4A8 dense GEMM for SM120 | 1 | none (no maintainer engagement) |
| 233 | open | extend fused_quant_a to NVFP4 dense GEMM | 1 | none |
| 234 | open | 16-row skinny-M tile for NVFP4 decode | 0 | none |
| 235 | open | W4A16 c_tmp sizing contract undocumented | 2 | none |

### local-inference-lab/rtx6kpro (1 issue)

| # | state | title | comments | action |
|---|---|---|---:|---|
| 79 | open | Transferable findings from Qwen3.8-27B EXL3 | 1 | none (no maintainer engagement) |

### LMCache/LMCache (2 issues, 1 PR)

| # | type | state | title | comments | action |
|---|---|---|---|---:|---|
| 4247 | issue | open | LMCacheMPConnector silently corrupts output | 16 | none (active thread, maintainer engaged, fix PR #4253 in progress) |
| 4492 | issue | open | MP-mode cross-restart corrupted output | 3 | none |
| 4600 | PR | open | fix(vllm/mp): mark failed retrieves as load errors | 0 | none (no reviews yet) |

## Actions taken

### CodeRabbit fixes (3 PRs)

**PR #436 (local-inference-lab/vllm)** — commit `3cd48af029` pushed to `feature/embed-online-kquant`:
1. `exl3_online_cache.py`: Added `.. note::` docstring to `resolve_model_identity` documenting that local checkpoints are not content-hashed (single-file >16MiB uses path/size/mtime_ns only, directory shards use (name,path,size,mtime_ns) tuples). Operators should clear `VLLM_EXL3_ONLINE_CACHE_DIR` or set `VLLM_EXL3_ONLINE_CACHE_MODE=off` after in-place weight replacements. `glob` left unchanged (not converted to `rglob`).
2. `deepseek_v2.py`: Renamed `_skip_disabled_mtp_weight` to `_skip_unloadable_mtp_weight`. Removed the `if nextn != 0` guard that blocked the MTP-enabled case. New logic: `upper_bound = hidden + nextn`; skip any layer with index >= upper_bound. Covers both MTP disabled (skip >= hidden) and MTP enabled (skip >= hidden+nextn, preventing KeyError on out-of-range layers).

**PR #439 (local-inference-lab/vllm)** — commit `c72c907427` pushed to `feature/spec-zero-depth-and-skip`:
1. `scheduler.py`: Added NOTE comment after `num_spec_tokens_to_schedule = 0` explaining that `allocate_slots` already reserved KV blocks for `self.num_lookahead_tokens` speculative slots (called earlier in the per-request loop), and this reservation cannot be retroactively shrunk because the aggregate skip condition is evaluated after all requests are scheduled. The reservation is harmless: single-token requests finish after one decode step so the extra blocks are released at the next scheduling iteration, and zeroing `num_spec_tokens_to_schedule` prevents the draft model from running.

**PR #3058 (vllm-project/llm-compressor)** — commit `b9b6ff2` already pushed (prior session):
- Both gemini-code-assist's bound-method `__closure__` bug and kylesayrs' helper-function/imports requests were addressed in a prior push. Posted reply comment confirming all fixes are in place and CI is green.

### Maintainer/community responses (2 items)

**vllm-project/vllm #52872**: Replied to Tejas-Raj01's volunteer offer, clarifying the scope (hybrid GDN model, 48/64 layers hold recurrent state, profiler under-predicts peak, mnbt also sizes CUDA-graph pool).

**vllm-project/llm-compressor #3058**: Posted issue comment confirming both gemini-code-assist's critical bug fix (bound method `__func__` extraction) and kylesayrs' refactoring requests (helper function + top-level imports) are addressed in commit b9b6ff2. CI is green. Ready for re-review.

## Items needing no action

- **b12x #232-235**: Zero maintainer engagement on any. Our measurements are posted; nothing more to do until maintainers respond.
- **rtx6kpro #79**: Zero maintainer engagement. Transferable findings posted.
- **LMCache #4247**: Active thread with maintainer engagement (thegoldenflow's fix PR #4253). Our 3 comments are substantive; nothing more to add.
- **LMCache #4492**: No maintainer response. Our 2 reproduction comments are posted.
- **LMCache PR #4600**: No reviews yet. DCO passes. Waiting for maintainer.
- **vllm-project/vllm #52871**: No comments or engagement. Bug report stands.
- **local-inference-lab/vllm issues #311, #313, #392, #396, #402, #406, #440, #442**: Bug reports, no maintainer engagement.
- **local-inference-lab/vllm PRs #312, #314, #316, #318, #393, #397, #398, #403**: CodeRabbit reviews were SKIPPED (auto-reviews disabled on non-default branches). No actionable findings.
