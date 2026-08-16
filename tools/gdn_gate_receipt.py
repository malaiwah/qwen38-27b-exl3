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
                },
            ],
            "not_a_mechanism": (
                "a short brand-new prompt on its own does not do it: region 0 requires "
                "has_context, and a fresh request has num_computed_tokens == 0, so "
                "utils.py:700 puts it in region 3 at the back of the batch."
            ),
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
        "sha256": "e42c9694a7f538ecd3931f9cdb177108c1c3617fd2ece6af2176e06cdd5afc43",
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
        "if_mixed_misordered_is_zero_in_R1": "the module cannot change any number under our published eight-stream recipe; the comparison arms are not run, because there is nothing for them to detect and their resolution could not detect it anyway",
        "if_mixed_misordered_is_zero_in_R1_and_R2": "the module is a no-op on this build for both our recipe and adversarial mixed short-prompt traffic; recommend dropping it",
        "if_mixed_misordered_is_zero_in_R1_but_nonzero_in_R2": "unreachable under our published recipes, reachable under mixed short-prompt traffic; recommend keeping it as an optional overlay documented for that traffic class",
        "if_mixed_misordered_is_nonzero_in_R1": "run arms A, An and B and report the divergence, acceptance and throughput deltas against the null arm",
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
        for arm in ("R1", "R2", "A", "An", "B"):
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
