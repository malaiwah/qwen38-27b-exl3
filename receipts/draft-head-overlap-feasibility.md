# Draft-head cost reduction feasibility (MTP=6, v2 speculator) — scout verdict 2026-08-18

## Architectural correction (docs/47 F6 refs are STALE for served build)
Active GPU path is the **v2 speculator**: `gpu_model_runner.py:270` → `init_speculator` →
`MTPSpeculator` → `AutoRegressiveSpeculator` (`v1/worker/gpu/spec_decode/autoregressive/speculator.py`).
`llm_base_proposer.py` is the legacy/CPU path. Draft decode steps (5 of 6) are ALREADY captured
as FULL_DECODE_ONLY CUDA graphs (`init_cudagraph_manager:78-96`) — dispatch overhead is gone,
weight traffic is not. `use_local_argmax_reduction` is dead code in v2 (never calls get_top_tokens).

## Verdicts
| option | verdict | reason |
|---|---|---|
| Side-stream overlap | INFEASIBLE | strictly autoregressive: input_ids[i+1]=argmax(head[i](body[i])); head on critical path; bandwidth-bound anyway |
| Draft-only K4/FP4 head copy | **FEASIBLE-MODERATE (best)** | in-tree precedent `fp8_draft_head.py` + DSpark (`dspark.py:319-405`); verify head stays K6-exact; flips only cost rejected drafts |
| Fused GEMV+argmax | NO VALUE | logits write is 0.5MB; the 909MiB weight READ is the cost |
| Parallel/tree drafting | INFEASIBLE | needs mask-token-trained checkpoint; Qwen3_5MTP is plain AR |
| Draft CUDA graphs | ALREADY DONE | v2 captures _generate_draft as FULL graphs |

## Best option numbers (draft-only FP4 head)
- Draft+verify share one 909 MiB K6 lm_head (`load_eagle_model` shares; `has_own_lm_head=False`)
- FP4 copy ≈ 606 MiB → 6 draft streams save 1.78 GB/step (8.1% of 22 GB/step)
- TG ceiling +~8%, net +5-7% after acceptance drag; zero correctness risk (verify exact)

## Patch sites (for implementation)
1. `qwen3_5_mtp.py:217-248,290-295` — `maybe_init_draft_head()` + `compute_draft_logits()` (mirror dspark.py:359-405), env `VLLM_EXL3_FP4_DRAFT_HEAD`
2. `speculator.py:364-387` `sample_draft` — prefer `compute_draft_logits` when present
3. `eagle/utils.py:128-145` — call `maybe_init_draft_head()` eagerly BEFORE graph capture
4. New `exl3_draft_head.py` helper using existing FP4 conversion from our patch
