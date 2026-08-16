#!/usr/bin/env python3
"""Assemble receipts/gdn-gate-concurrency.json.

Two halves, deliberately separated:

  ANALYSIS   the source-level reachability condition, read out of the pinned image's own
             bytes with file and line numbers. It needs no GPU, it is checked into the
             repo before any server starts, and it is the substantive result: it says
             under exactly which batch compositions the vendored file miscomputes.

  measured   whatever the instrumented and comparison arms actually produced, folded in
             from $WORK/out. Regenerable from those raw files alone on any host.

Run with --work to fold in a GPU window's output; run without it to publish the
analysis half on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECEIPT = Path("receipts/gdn-gate-concurrency.json")

# --------------------------------------------------------------------------------------
# The pinned image's own bytes. Every digest below was taken with podman cp out of
# localhost/vllm:gg-r34-patched-apc and every line number refers to those exact files,
# not to upstream's tree, which differs.
# --------------------------------------------------------------------------------------
IMAGE_FILE_DIGESTS = {
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": "663dacd324b6b8224a4cb312b3e9c0bad4322c515e982a85f13c3450ffdb7d61",
    "vllm/v1/attention/backends/gdn_attn.py": "5cb3f14fbc3461256e985ea80f6329d75ebd721e67cb28155527389a3e726d45",
    "vllm/v1/attention/backends/flashinfer.py": "df16d41ef834a551ef948ad22a371eb9411574cb999bb4343b8627a34d44963d",
    "vllm/v1/attention/backends/utils.py": "3cc567dd0ce7a8bcfef16eb283f64b252cd993e3716ee240e7b0dd956d5990c8",
    "vllm/v1/attention/backend.py": "7560b2f6e0f71cb74cb609c1ab352660f9ebfafee4e756d1411bc1322a9d7353",
    "vllm/v1/worker/gpu_model_runner.py": "6e96f2810959aeb7003eeb1e4ce64f46d8e72bebbe38a27f33b6904d25e861ed",
    "vllm/v1/core/sched/scheduler.py": "b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf",
    "vllm/utils/flashinfer.py": "310b4670faf6c81cda01edd4bd30055e000d6cc2f2ce8b54a2ad2711d15317b7",
}

ANALYSIS = {
    "question": (
        "PR #51812 is absent from the pinned image. At --max-num-seqs 1 it is provably "
        "irrelevant because a single sequence never forms a mixed batch. Our published "
        "concurrent-serving recipe uses eight streams, which does form mixed batches. "
        "Does the module change anything there?"
    ),
    "the_defect_in_one_sentence": (
        "The vendored qwen_gdn_linear_attn.py gathers the speculative Q/K/V rows with "
        "spec_token_indx but hands the recurrent update the ungathered gate tensors a "
        "and b, so gate row i can belong to a different token than Q/K/V row i."
    ),
    "exact_condition_for_the_defect_to_change_any_number": {
        "statement": (
            "unpatched and patched are bit-for-bit identical unless "
            "spec_token_indx == arange(num_spec_decode_tokens) is FALSE, i.e. unless the "
            "batch places at least one non-speculative token at a lower token index than "
            "some speculative token."
        ),
        "why": [
            "qwen_gdn_linear_attn.py:1293 enters the speculative path only when spec_sequence_masks is not None.",
            "qwen_gdn_linear_attn.py:1294 takes a pure-speculative fast path when num_prefills == 0 and num_decodes == 0; there line 1295 sets mixed_qkv_spec = mixed_qkv with no gather at all, so a and b are already aligned and the module is a no-op by construction.",
            "qwen_gdn_linear_attn.py:1298 is the only other branch: mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx).",
            "qwen_gdn_linear_attn.py:1421-1424 then calls fused_sigmoid_gating_delta_rule_update with a=a and b=b, i.e. the first T_spec rows of the ungathered tensors.",
            "a.index_select(0, spec_token_indx) equals a[:T_spec] exactly when spec_token_indx is the identity permutation arange(T_spec). That is the whole difference the module makes.",
            "The identifiers a_spec and b_spec do not occur anywhere in the vendored file; the published module adds them at the same two places (8 changed lines, diff-identical to upstream).",
        ],
        "predicate_verified_by_brute_force": (
            "the counter's host-side predicate (all speculative tokens occupy the first "
            "T_spec positions) was checked against the image's own torch.argsort(stable) "
            "construction of spec_token_indx over 14409 batch shapes, 13148 of them "
            "non-identity: zero disagreements."
        ),
    },
    "which_batch_orders_can_occur_on_this_build": {
        "spec_token_indx_is_built_at": (
            "gdn_attn.py:278-286: spec_token_masks = repeat_interleave(spec_sequence_masks, "
            "query_lens); index = argsort(spec_token_masks, stable=True); "
            "spec_token_indx = index[num_non_spec_tokens:]. A stable argsort of a boolean "
            "key preserves batch order inside each class, so spec_token_indx is the "
            "ascending list of positions of the speculative tokens, and it is the identity "
            "exactly when they come first."
        ),
        "batch_order_is_set_by": [
            "gpu_model_runner.py:1120-1139 _may_reorder_batch calls reorder_batch_to_split_decodes_and_prefills with decode_threshold = self.reorder_batch_threshold.",
            "utils.py:665 reorder_batch_to_split_decodes_and_prefills, with the four regions defined at utils.py:691-709: has_context = num_computed > 0, is_below_threshold = num_scheduled <= threshold, done_prefilling = num_computed >= num_prompt_tokens; region 0 decode = has_context & below & done, region 1 short_extend = has_context & below & not done, region 2 long_extend = has_context & not below, region 3 pure prefill = not has_context; emitted in the order 0,1,2,3.",
            "gpu_model_runner.py:7310-7328 calculate_reorder_batch_threshold takes the MINIMUM threshold over all attention groups, so on a hybrid model the full-attention group can veto the GDN group's choice.",
        ],
        "the_threshold_on_this_build_is_1_plus_num_spec": {
            "gdn_group": "gdn_attn.py:85 declares 1 and gdn_attn.py:112 calls _init_reorder_batch_threshold(1, self.use_spec_decode).",
            "full_attention_group": "FLASHINFER (startup log: 'Using FLASHINFER attention backend out of potential backends: [FLASHINFER, TRITON_ATTN]'), flashinfer.py:855 calls _init_reorder_batch_threshold(1, supports_spec_as_decode=...).",
            "raise_rule": "backend.py:888-910: when supports_spec_as_decode is true the threshold is raised to max(1, 1 + num_speculative_tokens) for non-parallel drafting. backend.py:858 shows the base-class default is None, and the min-fold at gpu_model_runner.py:7328 treats None as no opinion, so a backend that never sets it cannot drag the threshold down.",
            "flashinfer_supports_spec_as_decode_is_TRUE_here": [
                "flashinfer.py:846-854 sets supports_spec_as_decode = (trtllm decode kernel is TRTLLM_GEN) or (trtllm decode is not used and flashinfer_supports_uniform_multi_token_decode()).",
                "flashinfer.py:829 sets use_trtllm_decode_attention = can_use_trtllm_attention(...), which at vllm/utils/flashinfer.py:411 requires supports_trtllm_attention, which at vllm/utils/flashinfer.py:394 requires current_platform.is_device_capability_family(100). The RTX 5090 is compute capability 12.0, so this is False: no trtllm-gen and no XQA decode on this card.",
                "so supports_spec_as_decode reduces to flashinfer_supports_uniform_multi_token_decode(), which at flashinfer.py:2422 is 'q_len_per_req' in inspect.signature(fast_decode_plan).parameters. Evaluated inside the image in a CPU-only container with no GPU device: flashinfer 0.6.18+cu132, q_len_per_req present, so TRUE.",
            ],
            "consequence": (
                "effective decode_threshold = 1 + num_speculative_tokens = 4 at mtp depth 3 "
                "(2 at depth 1). A speculative decode has query_len = 1 + drafts <= 4, so it "
                "is below threshold and, being done prefilling with context, lands in region "
                "0 at the FRONT. Every prefill and extend lands in region 1, 2 or 3, i.e. "
                "BEHIND it. With only speculative decodes in region 0 the speculative tokens "
                "are exactly the leading tokens of the batch, spec_token_indx is the "
                "identity, and the vendored code is bit-identical to the patched code."
            ),
        },
        "so_the_defect_needs_a_NON_speculative_request_ahead_of_a_speculative_one": {
            "which_requests_count_as_speculative": (
                "gpu_model_runner.py:2240-2249: num_decode_draft_tokens is -1 (non-speculative) "
                "unless the request appears in scheduler_output.scheduled_spec_decode_tokens "
                "AND num_scheduled_tokens equals draft_len + 1. gdn_attn.py then derives "
                "spec_sequence_masks from num_decode_draft_tokens >= 0."
            ),
            "candidate_mechanisms_found_in_the_image": [
                {
                    "mechanism": "a running request whose drafts were dropped, decoding one token beside other streams' speculative decodes",
                    "where": "scheduler.py:635-642 only schedules speculative tokens when request.spec_token_ids is non-empty and num_scheduled_spec_tokens > 0; scheduler.py:2131-2133 clears spec_token_ids for a prefill chunk",
                    "reachable_under_our_recipe": "no, on the evidence in the image: scheduler.py:1278 sets is_prefill_chunk = num_computed < num_tokens + num_output_placeholders, which is already False on the step that completes a prefill (placeholders are 0 without async scheduling), so drafts are assigned and the first decode step is already a speculative decode; and with --max-num-batched-tokens 2048 against 8x4 = 32 decode tokens the running loop is never squeezed into num_new_tokens == 1",
                    "reachable_elsewhere": "yes with --async-scheduling (non-zero num_output_placeholders), or for a generation that reaches --max-model-len, or under a token budget small enough to truncate a running request to a single token",
                },
                {
                    "mechanism": "the scheduler's speculative padding: a request admitted needing exactly one new token is padded to 1 + num_spec tokens and is then classified as a speculative decode by the runner, while the reorder leaves it behind the real decodes because it is not done prefilling (region 1, not region 0)",
                    "where": "scheduler.py:865-878 pad_spec_decode (requires num_spec_tokens > 0, dynamic_sd_lookup None, num_new_tokens == 1, running requests already scheduled and no prefill chunk scheduled) and scheduler.py:1052 which writes [-1] * num_spec_tokens into scheduled_spec_decode_tokens",
                    "how_to_reach_it": "prefix caching on, then re-send an identical prompt: the whole prompt hits the cache, admission has to recompute exactly one token, and the padding fires while the other streams decode",
                    "note": "this is the mechanism the adversarial arm R2 targets. It needs a non-speculative request ordered ahead of it as well, which region 1 can supply in the form of a chunked prefill whose final chunk is 2-4 tokens.",
                    "but_it_is_mechanism_only_for_us": (
                        "it requires prefix caching, and prefix caching is not shippable for "
                        "this model at the native window on this card. Measured independently by "
                        "agent ShipPrefixCaching on this same promoted digest with "
                        "--enable-prefix-caching --mamba-cache-mode align: at utilisation 0.955 "
                        "the engine refuses to start at 262144 because align rounds the request "
                        "up to 164 whole 1600-token blocks and it needs 9.29 GiB of KV against "
                        "9.28 available; at 0.9555 it starts with a pool of exactly 262144 and "
                        "deadlocks mid-prefill with reason=capacity; at 0.9585 with a pool of "
                        "265072 it livelocks, re-prefilling on a 30 second period at 960 tok/s "
                        "with zero output tokens while vllm:num_preemptions_total stays at 0. "
                        "The admission rule is not pool >= window but pool >= window rounded up "
                        "to whole mamba blocks plus MTP draft slots plus a decode block. So R2 "
                        "measures a configuration we do not ship and is reported as evidence "
                        "about the mechanism only, never as evidence about our recipes."
                    ),
                    "how_R2_avoids_that_wall": (
                        "R2 runs at a 32768 window instead of 262144. The longest frozen prompt "
                        "is 15400 tokens plus 220 of output, so every request clears 21 whole "
                        "mamba blocks with room to spare and no single prefill can approach the "
                        "pool, which makes the livelock impossible rather than merely unlikely. "
                        "The padding path depends on prefix-cache hits, not on window size."
                    ),
                    "explicitly_NOT_the_cause_of_that_livelock": (
                        "tempting and wrong, recorded so nobody assembles it later. This padding "
                        "path inflates a re-admitted request's cost from one token to 1 + "
                        "num_spec, which looks like it could explain an admission loop in a pool "
                        "at its ceiling. It cannot explain the livelock above: in that "
                        "instrumented run vllm:prefix_cache_queries_total was 261794 with "
                        "vllm:prefix_cache_hits_total exactly 0.0, so num_new_tokens was never 1 "
                        "and scheduler.py:868 never held. Each cycle there was a full cold "
                        "re-prefill, not a re-admitted cache hit. Checked and refuted by "
                        "ShipPrefixCaching against its own trace. The two findings share the "
                        "admission decision and nothing else."
                    ),
                },
                {
                    "mechanism": "token-budget clamping of a speculative decode down to exactly one token, which silently reclassifies it as non-speculative. This is the only mechanism found that needs no prefix caching, no async scheduling and no contrivance, so it is the one that could fire on ordinary mixed traffic.",
                    "where": [
                        "scheduler.py:511-515 computes num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens, which is 1 + drafts for a drafted decode.",
                        "scheduler.py:518 then clamps it: num_new_tokens = min(num_new_tokens, token_budget).",
                        "scheduler.py:637-642 recomputes num_scheduled_spec_tokens from the clamped count; at a clamp of exactly 1 it is 0, so the request is NOT entered into scheduled_spec_decode_tokens.",
                        "gpu_model_runner.py:2240-2249 therefore leaves num_decode_draft_tokens at -1, i.e. non-speculative, while the request is still a one-token decode with context that is done prefilling: region 0 by utils.py:703.",
                    ],
                    "why_the_slot_order_can_be_wrong": (
                        "the budget is consumed walking self.running (scheduler.py:480), whereas "
                        "the reorder and the GDN gather both work on input_batch slot order, and "
                        "reorder_batch_to_split_decodes_and_prefills only permutes slots and never "
                        "touches self.running. The two orders are independent, so the clamped "
                        "request can hold a lower slot than fully scheduled speculative decodes."
                    ),
                    "how_often": (
                        "the clamp has to land on exactly 1. A co-scheduled prefill chunk takes "
                        "min(remaining_prompt, token_budget), so it either exhausts the budget "
                        "completely (leaving nothing, and the later decodes are simply not "
                        "scheduled: scheduler.py:480 exits on token_budget > 0) or leaves a "
                        "remainder. A remainder of 2, 3 or 4 truncates the drafts but keeps "
                        "num_scheduled_tokens == draft_len + 1, so the request stays speculative "
                        "and nothing breaks. Only a remainder of exactly 1 breaks it, which at "
                        "--max-num-batched-tokens 2048 needs the prefill's remaining length to hit "
                        "one specific value. So: reachable on ordinary traffic, but rare per step "
                        "rather than constant. --mamba-cache-mode align makes it rarer still, "
                        "because scheduler.py:547-550 rounds chunks to whole mamba blocks and the "
                        "remainder is then structured rather than arbitrary."
                    ),
                    "verified_by": "tools/gdn_reorder_sim.py cases 8 and 9: the same clamped request miscomputes at a middle region-0 slot and is harmless at the highest region-0 slot",
                },
            ],
            "answered_and_refuted": [
                {
                    "hypothesis": "a request's FIRST decode step after its own prefill completes is a plain non-speculative decode, because no draft exists for it yet",
                    "refuted_because": "scheduler.py:1276-1280 sets is_prefill_chunk = num_computed_tokens < num_tokens + num_output_placeholders, evaluated after scheduling and before the forward. On the step that completes a prefill num_computed equals num_prompt_tokens and num_tokens is still num_prompt_tokens with no placeholders, so it is already False, and scheduler.py:2131-2135 therefore does NOT clear the drafts the drafter proposed on that step. The first decode step is already a speculative decode.",
                },
                {
                    "hypothesis": "the step after a full draft rejection is a plain non-speculative decode",
                    "refuted_because": "drafts are consumed and cleared every step at scheduler.py:650 and re-proposed every step regardless of how many were accepted; rejection does not suppress the next proposal. gpu_model_runner.py:4943-4952 returns a dense draft row for every request in the batch, so the drafter does not selectively skip requests.",
                },
                {
                    "hypothesis": "a very short brand-new prompt arriving mid-flight lands beside the speculative decodes",
                    "refuted_because": "region 0 requires has_context and a fresh request has num_computed_tokens == 0, so utils.py:700 puts it in region 3 at the very back. Checked by executing the image's own reorder: a brand-new 4-token prompt placed first is moved to the back and the permutation stays the identity.",
                },
                {
                    "hypothesis": "a preempted and resumed request does it",
                    "refuted_because": "scheduler.py:1245-1257 sets num_computed_tokens = 0 and clears spec_token_ids before returning the request to the waiting queue, so it comes back as a prefill in region 3.",
                },
            ],
            "not_a_mechanism": (
                "a short brand-new prompt on its own does not do it: region 0 requires "
                "has_context, and a fresh request has num_computed_tokens == 0, so "
                "utils.py:700 puts it in region 3 at the back of the batch."
            ),
            "does_the_rate_depend_on_max_num_batched_tokens": {
                "short_answer": "yes, and inversely: the predicted rate scales roughly as max_num_seqs / max_num_batched_tokens per prefill completion, so a LARGER budget is a free partial mitigation and a smaller one is a hazard.",
                "derivation": [
                    "Within one step the running loop (scheduler.py:480) spends the budget in self.running order. A drafted decode takes 1 + num_spec = 4. A prefilling request takes min(remaining_prompt, budget_left).",
                    "If a prefill takes the whole remaining budget, budget_left reaches 0 and the loop stops, so no decode is clamped and nothing breaks.",
                    "Otherwise it leaves R = budget_left - remaining_prompt, and each later drafted decode takes 4, so budget_left walks R, R-4, R-8, ...",
                    "A decode is clamped to exactly 1 only if that walk lands on 1, i.e. only if R = 1 (mod 4). Landing on 2, 3 or 4 truncates the drafts but leaves num_scheduled_tokens == draft_len + 1, so scheduler.py:637-642 still registers the request as speculative and nothing breaks.",
                    "The walk can only take as many steps as there are drafted decodes after the prefill in self.running order, at most max_num_seqs - 1 = 7, so additionally R <= 1 + 4 * 6 = 25.",
                    "R <= 25 forces remaining_prompt to land in a window of width about 4 * max_num_seqs immediately below the budget. As a fraction of the possible values of remaining_prompt (0 .. max_num_batched_tokens) that window is about max_num_seqs / max_num_batched_tokens, and only one residue class mod 4 inside it qualifies.",
                ],
                "predicted_rate_at_our_recipe": "order 8 / 2048, further divided by the mod-4 condition, i.e. a fraction of a percent of prefill completions, and then only when the clamped request also happens to hold a lower input_batch slot than a fully scheduled speculative decode",
                "why_this_is_a_prediction_and_not_a_result": "it assumes prefill remainder lengths are spread arbitrarily and that self.running order mixes prefills and decodes; both are plausible and neither is proven, so arm R3 tests the scaling by shrinking the budget 4x and checking whether the counter rises about 4x",
                "mitigation_for_an_operator_who_cannot_patch": "raise --max-num-batched-tokens. It does not close the hole (for any budget there is still a qualifying residue class) but it narrows the triggering window proportionally. --mamba-cache-mode align narrows it further because scheduler.py:547-550 rounds chunks to whole mamba blocks, which removes most of the arbitrary remainders.",
            },
        },
        "corroborated_by_executing_the_images_own_code_no_gpu": {
            "tool": "tools/gdn_reorder_sim.py",
            "what_it_does": (
                "imports reorder_batch_to_split_decodes_and_prefills out of the image, runs "
                "it over seven named batch compositions at decode_threshold 4, then applies "
                "gdn_attn.py:278-286 verbatim to the reordered batch and reports whether "
                "spec_token_indx is the identity. Nothing is reimplemented."
            ),
            "result": {
                "identity_preserved_so_vendored_is_bit_identical": [
                    "8 speculative decodes + a chunked prefill placed first: the reorder moves the prefill to the back (long_extend)",
                    "8 speculative decodes + a first-chunk prefill placed first: moved to the back (pure prefill)",
                    "8 speculative decodes + a brand-new 4-token prompt placed first: moved to the back, because a fresh request has no context",
                    "8 speculative decodes + a non-speculative plain decode placed LAST: already ordered behind them",
                    "8 speculative decodes + a padded cache-hit pseudo-speculative request ahead of a non-speculative tiny final chunk",
                ],
                "vendored_miscomputes": [
                    "8 speculative decodes + a non-speculative plain decode placed FIRST: both are region 0, the reorder does not move either, and the speculative tokens are no longer the leading tokens",
                    "8 speculative decodes + a non-speculative tiny final chunk ahead of a padded cache-hit pseudo-speculative request: both are region 1, same outcome",
                ],
                "reading": (
                    "the reorder is what protects the recipe: every prefill-shaped request, "
                    "including a very short brand-new prompt, is moved behind the speculative "
                    "decodes. What it does not protect against is a non-speculative request "
                    "that is already in the same low region, and there the slot order alone "
                    "decides. So the defect is reachable in principle on this build, and the "
                    "open question is purely whether our traffic ever produces such a request."
                ),
            },
        },
    },
    "why_a_logprob_comparison_cannot_settle_this_by_itself": {
        "upstream_effect_size": {
            "model": "Qwen3.5-2B",
            "mean_abs_chosen_logprob_error_before": 0.002755,
            "mean_abs_chosen_logprob_error_after": 0.000208,
            "max_abs_before": 0.020539,
            "max_abs_after": 0.001690,
            "greedy_token_ids": "identical before and after",
        },
        "our_resolution": {
            "run_to_run_floor_mean_abs_chosen_logprob_delta": 0.08231278,
            "source": "receipts/apc-poison-repro.json, arm Bn: two identically flagged fresh servers on this build",
            "identical_text_requests_floor": "20 of 38",
        },
        "ratio": "the floor is about 30x the effect upstream measured",
        "conclusion": (
            "an A/B/null logprob comparison at eight streams is pre-determined to report "
            "'below resolution' whether the module matters or not, so it is reported as a "
            "bound and never as the primary evidence. The primary evidence is the counter, "
            "which asks whether the defective code path executes at all."
        ),
    },
    "instrument": {
        "file": "tools/vllm-gdn-reach-instrument.py",
        "destination": "vllm/v1/attention/backends/gdn_attn.py",
        "replaces_vendored_sha256": "5cb3f14fbc3461256e985ea80f6329d75ebd721e67cb28155527389a3e726d45",
        "sha256": "eb61b24b2ab876bb24c433b2dbfd0ad2e6587014765e16d0928239f3c0c5889b",
        "what_it_does": (
            "counts, per GDN metadata build, which of the three branches ran and, on the "
            "mixed branch, whether the speculative tokens are the leading tokens of the "
            "batch. Reads only spec_sequence_masks_cpu and query_start_loc_cpu, which are "
            "already on the host, so there is no device synchronisation, no kernel launch "
            "and no extra device allocation: the instrument cannot perturb the schedule or "
            "the timing it observes."
        ),
        "why_one_instrumented_server_covers_both_arms": (
            "PR #51812 does not touch gdn_attn.py, so batch composition is identical "
            "patched and unpatched. Reachability measured once applies to both arms."
        ),
        "additivity_is_checked_not_asserted": (
            "the stage phase diffs the instrument against the vendored file inside the "
            "image, requires every opcode to be an insertion, and requires that deleting "
            "the insertions reproduces the vendored sha256 exactly."
        ),
    },
    "module_under_test": {
        "path": "tools/vllm-qwen-gdn-spec-gates.py",
        "sha256": "7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0",
        "destination": "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "replaces_vendored_sha256": "663dacd324b6b8224a4cb312b3e9c0bad4322c515e982a85f13c3450ffdb7d61",
        "upstream": "https://github.com/vllm-project/vllm/pull/51812 merged 2026-08-11T15:35:30Z as 5af7c8dad798bf899813f8f3c6b9eaf08a748e17",
    },
}

PRE_REGISTRATION = {
    "committed_before_any_server_started": True,
    "primary_measurement": "tools/vllm-gdn-reach-instrument.py mixed_misordered count",
    "decision_rule": {
        "fixed_by": "the parent agent, before the GPU window opened, so the measurement decides rather than a later argument",
        "if_mixed_misordered_is_zero_in_R1": "the cards say the #51812 module is provably unreachable under our published recipes, with the mechanism and the reason it cannot arise. The comparison arms are not run: there is nothing for them to detect and their resolution could not detect it anyway.",
        "if_mixed_misordered_is_nonzero_in_R1_at_any_rate": "the cards recommend mounting the overlay for concurrent serving. This is fixed in advance and is deliberately not a cost-benefit judgement: the module is free, py_compile-clean and diff-identical to upstream, and a rate above zero means some fraction of forward passes are silently miscomputed with no signal to the operator. A silent wrong computation with a free fix gets the fix.",
        "R2_is_mechanism_only": "R2 needs prefix caching. If prefix caching is not qualified for shipping then R2 measures a configuration we do not ship, and it is reported as evidence about the mechanism and explicitly not as evidence about our recipes.",
        "R3_is_the_scaling_test": "R3 shrinks --max-num-batched-tokens by 4x. If the derived scaling is right the counter should rise about 4x. A zero in R1 together with a nonzero in R3 means the mechanism is real and our budget merely makes it rare, which is a weaker claim than unreachable, and the receipt must say which of the two it earned.",
        "if_R1_and_R3_are_both_zero": "the derivation's own amplifier failed to produce an event, so the receipt reports the mechanism as unobserved on this build and states the resolution rather than claiming absence.",
        "reporting_requirement": "every count is also reported as a rate per thousand metadata builds, and every zero is accompanied by the smallest rate the run could have detected, one event over the builds observed, so a zero never claims more than the run supports.",
    },
    "floor_that_any_claimed_difference_must_beat": 0.08231278,
    "throughput_metric": "accepted tokens per step and step time reported separately, then aggregate tok/s, matching docs/36 and receipts/perf-sweep-5090.json",
    "baselines_at_eight_streams_from_receipts_perf_sweep_5090": {
        "accepted_tokens_per_step": 2.1753,
        "step_time_ms": 49.52,
        "aggregate_tok_s": 313.28,
        "caveat": "measured on localhost/vllm:gg-r34-patched, not on the promoted -apc superset; used as an order-of-magnitude anchor, not as this receipt's control",
    },
}


def load(out: Path, name: str):
    path = out / name
    return json.loads(path.read_text()) if path.exists() else None


def load_first_json(out: Path, name: str):
    """The container may prefix stdout with its own banner; take the first JSON value."""
    path = out / name
    if not path.exists():
        return None
    raw = path.read_text()
    start = raw.find("{")
    if start < 0:
        return None
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    return value


def facts(out: Path, tag: str) -> list[str]:
    path = out / f"startup-facts-{tag}.txt"
    return [l for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def text(out: Path, name: str) -> str | None:
    path = out / name
    return path.read_text() if path.exists() else None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default=None, help="GPU window working directory")
    parser.add_argument("--out", default=str(RECEIPT))
    args = parser.parse_args()

    receipt: dict = {
        "schema": "qwen38-gdn-gate-concurrency/1",
        "purpose": (
            "decide whether upstream vLLM PR #51812 (Qwen GDN speculative gate alignment) "
            "changes anything on the promoted image at eight concurrent streams, and say "
            "what the model cards should do about it"
        ),
        "image": {
            "tag": "localhost/vllm:gg-r34-patched-apc",
            "manifest_digest": "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218",
            "file_digests_read_for_the_analysis": IMAGE_FILE_DIGESTS,
        },
        "analysis": ANALYSIS,
        "pre_registration": PRE_REGISTRATION,
    }

    measured: dict = {"status": "not run"}
    if args.work:
        work = Path(args.work)
        out = work / "out"
        arms: dict[str, dict] = {}
        for arm in ("R1", "R2", "R3", "A", "An", "B"):
            command = load(out, f"command-{arm}.json")
            if command is None:
                continue
            arms[arm] = {
                "description": command["description"],
                "podman_argv": command["podman_argv"],
                "vllm_argv": command["vllm_argv"],
                "image_manifest_digest": command["image_manifest_digest"],
                "startup_facts": facts(out, arm),
                "module_digests_inside_the_running_container": text(
                    out, f"in-container-verify-{arm}.txt"
                ),
                "reachability": load(out, f"reach-{arm}.json"),
                "load": (load(out, f"arm-{arm}.json") or {}).get("summary")
                or (load(out, f"arm-{arm}.json") or {}).get("load"),
                "server_windows": load(out, f"windows-{arm}.json"),
            }
        measured = {
            "status": "run" if arms else "no arms found",
            "stage": load(out, "stage.json"),
            "instrument_additivity_check": load(out, "instrument-additive.json"),
            "reorder_simulation": load_first_json(out, "reorder-sim.json"),
            "module_verify_in_image": text(out, "module-verify-image.txt"),
            "arms": arms,
            "divergence": load(out, "verdict.json"),
            "raw_artifact_digests": {
                str(p.relative_to(work)): digest(p)
                for p in sorted(work.rglob("*"))
                if p.is_file() and p.name != "gdn-gate-concurrency.json"
            },
        }
    receipt["measured"] = measured

    body = json.dumps(
        {k: v for k, v in receipt.items() if k != "content_sha256"}, sort_keys=True
    ).encode()
    receipt["content_sha256"] = hashlib.sha256(body).hexdigest()
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": args.out,
                "bytes": Path(args.out).stat().st_size,
                "measured_status": measured["status"],
                "arms": sorted(measured.get("arms", {})),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
