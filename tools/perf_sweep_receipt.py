#!/usr/bin/env python3
"""Assemble receipts/perf-sweep-5090.json from the raw phase artifacts.

Every published row carries the command that produced it (the podman and vLLM argv arrays
written by perf_sweep_5090.sh), the startup banner facts that changed, and every timed
repeat. Deltas are computed against the `base` row only; a row with no base counterpart is
published without a delta rather than compared to something else.

The winner is decided on accepted tokens per step divided by step time, which is the
per-request decode rate the engine can actually sustain, not on acceptance rate.

    perf_sweep_receipt.py --artifacts /tmp/perfsweep-out --out receipts/perf-sweep-5090.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ORDER = [
    "base",
    "customops",
    "flashinfer",
    "triton",
    "mtp1",
    "mtp2",
    "adaptive",
    "perbatch",
    "eagerbase",
    "adaptive_eager",
    "perbatch_eager",
    "combo",
]


def reference_for(tag: str) -> str | None:
    """Which row a row is compared against.

    The eager arms exist only because dynamic speculative decoding forces eager execution on
    this build; comparing them to the graph-decode base would price losing CUDA graphs as if
    it were the lever's effect, so they are compared to `eagerbase`.
    """
    if tag == "base":
        return None
    return "eagerbase" if tag.endswith("_eager") else "base"


FACT_KEYS = {
    "attention_backend": "attention backend out of potential backends",
    "vit_attention": "for MMEncoderAttention",
    "attention_block_size": "Setting attention block size",
    "kv_padding_layers": "padding layers",
    "available_kv_gib": "Available KV cache memory",
    "gpu_kv_cache_tokens": "GPU KV cache size",
    "max_concurrency": "Maximum concurrency for",
    "memory_breakdown": "Actual usage is",
    "graph_pool": "Graph capturing finished",
    "graph_shapes_captured": "Capturing CUDA graphs (decode, FULL)",
    "model_loading": "Model loading took",
    "engine_init": "init engine (profile, create kv cache, warmup model)",
    "prefix_caching": "enable_prefix_caching=",
    "num_spec_tokens": "num_spec_tokens=",
    "exl3_graph_decode": "EXL3 graph decode enabled by",
    "exl3_gemm_priming": "priming exl3_gemm for",
    "engine_version": "Initializing a V1 LLM engine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def facts(path: Path) -> dict:
    if not path.is_file():
        return {}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    out: dict = {"raw": lines}
    for key, needle in FACT_KEYS.items():
        match = next((line for line in lines if needle in line), None)
        if match is not None:
            out[key] = match
    return out


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def decode_rows(probe: dict) -> dict:
    """Key every decode point by temperature and concurrency for delta arithmetic."""
    rows = {}
    for point in probe.get("decode", []):
        key = f"T{point['temperature']:g}-C{point['concurrency']}"
        rows[key] = {
            "temperature": point["temperature"],
            "concurrency": point["concurrency"],
            "seed": point["seed"],
            "median_aggregate_tok_s": point["median_aggregate_tok_s"],
            "median_per_stream_tok_s": point["median_per_stream_tok_s"],
            "spread_aggregate_tok_s": [point["min_aggregate_tok_s"], point["max_aggregate_tok_s"]],
            "accepted_tokens_per_step_per_request": point["median_accepted_tokens_per_step"],
            "step_time_ms": point["median_step_time_ms"],
            "draft_acceptance_rate": point.get("median_draft_acceptance_rate"),
            "tokens_per_step_batch_wide": point.get("median_tokens_per_step_batch_wide"),
            "decision_metric_tok_s": (
                round(
                    point["median_accepted_tokens_per_step"]
                    / (point["median_step_time_ms"] / 1000.0),
                    2,
                )
                if point["median_accepted_tokens_per_step"] and point["median_step_time_ms"]
                else None
            ),
            "repeats": point["repeats"],
        }
    return rows


def prefill_rows(probe: dict) -> dict:
    return {
        f"P{point['measured_prompt_tokens']}": {
            "prompt_tokens": point["measured_prompt_tokens"],
            "median_prefill_tok_s": point["median_prefill_tok_s"],
            "spread_prefill_tok_s": [point["min_prefill_tok_s"], point["max_prefill_tok_s"]],
            "repeats": point["repeats"],
        }
        for point in probe.get("prefill", [])
    }


def pct(candidate, reference):
    if candidate is None or reference in (None, 0):
        return None
    return round((candidate - reference) / reference * 100.0, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--winner", required=True, help="tag of the recommended configuration")
    parser.add_argument("--winner-rationale", required=True)
    parser.add_argument("--unrun", nargs="*", default=[], help="tags the window did not reach")
    parser.add_argument(
        "--production-image-receipt",
        default=None,
        help="path to receipts/production-image.json, summarised and referenced by digest",
    )
    args = parser.parse_args()

    art = Path(args.artifacts)
    stage = load(art / "stage.json") or {}
    logs = load(art / "log-digests.json") or {}

    rows: dict[str, dict] = {}
    for tag in ORDER:
        probe = load(art / f"probe-{tag}.json")
        failure = load(art / f"failed-{tag}.json")
        if probe is None and failure is None:
            continue
        command = load(art / f"command-{tag}.json") or {}
        row = {
            "lever": command.get("lever") or (failure or {}).get("lever"),
            "command": {
                "podman_argv": command.get("podman_argv"),
                "vllm_argv": command.get("vllm_argv"),
                "source_bind_mounts": command.get("source_bind_mounts", []),
            },
            "startup_facts": facts(art / f"startup-facts-{tag}.txt"),
        }
        if probe is None:
            row.update(
                {
                    "started": False,
                    "outcome": failure.get("outcome"),
                    "refusal_evidence": failure.get("evidence"),
                    "log_tail": failure.get("log_tail"),
                }
            )
            rows[tag] = row
            continue
        row.update(
            {
                "started": True,
                "measured_utc": probe.get("utc"),
                "memory": load(art / f"memory-{tag}.json") or {},
                "exposure": load(art / f"exposure-{tag}.json") or {},
                "prefill": prefill_rows(probe),
                "decode": decode_rows(probe),
                "prefix_cache_hit_rate": probe.get("prefix_cache_hit_rate"),
                "final_accepted_per_pos": probe.get("final_accepted_per_pos"),
                "probe_artifact_sha256": sha256(art / f"probe-{tag}.json"),
            }
        )
        rows[tag] = row

    if "base" not in rows or not rows["base"].get("started"):
        raise SystemExit("no measured base row: every A/B needs the reference configuration")
    for tag, row in rows.items():
        reference_tag = reference_for(tag)
        if reference_tag is None or not row.get("started"):
            continue
        reference = rows.get(reference_tag)
        if reference is None or not reference.get("started"):
            row["delta_reference"] = f"{reference_tag} (not measured; no delta published)"
            continue
        row["delta_reference"] = reference_tag
        row["delta_vs_reference_pct"] = {
            "decode": {
                key: {
                    "aggregate_tok_s": pct(
                        value["median_aggregate_tok_s"],
                        reference["decode"].get(key, {}).get("median_aggregate_tok_s"),
                    ),
                    "decision_metric_tok_s": pct(
                        value["decision_metric_tok_s"],
                        reference["decode"].get(key, {}).get("decision_metric_tok_s"),
                    ),
                    "accepted_tokens_per_step": pct(
                        value["accepted_tokens_per_step_per_request"],
                        reference["decode"]
                        .get(key, {})
                        .get("accepted_tokens_per_step_per_request"),
                    ),
                    "step_time_ms": pct(
                        value["step_time_ms"],
                        reference["decode"].get(key, {}).get("step_time_ms"),
                    ),
                }
                for key, value in row["decode"].items()
            },
            "prefill": {
                key: pct(
                    value["median_prefill_tok_s"],
                    reference["prefill"].get(key, {}).get("median_prefill_tok_s"),
                )
                for key, value in row["prefill"].items()
            },
            "gpu_kv_cache_tokens_delta": (
                row["memory"].get("gpu_kv_cache_tokens", 0)
                - reference["memory"].get("gpu_kv_cache_tokens", 0)
            ),
        }

    # The four production-image serving gates are owned by receipts/production-image.json
    # (ImmutableImage's harness wrote it). Summarise and point at it by digest rather than
    # restating its numbers, so one fact has one owner.
    gates: dict = {}
    if args.production_image_receipt:
        path = Path(args.production_image_receipt)
        prod = load(path) or {}
        smoke = prod.get("smoke", {})
        gates = {
            "receipt": str(path),
            "receipt_sha256": sha256(path),
            "image_manifest_digest": prod.get("image", {}).get("manifest_digest"),
            "recipes_run": smoke.get("recipes_run"),
            "all_run_recipes_started": smoke.get("all_run_recipes_started"),
            "all_run_recipes_text_and_image_exact": smoke.get(
                "all_run_recipes_text_and_image_exact"
            ),
            "acceptance": prod.get("acceptance"),
            "ran_by": "PerfSweep5090, inside this GPU window, using "
            "docker/build-image.sh smoke context hydrated k5k6 k4",
            "harness_defect_found_and_fixed": "docker/smoke_client.py hardcoded model id 'm' "
            "while the recipes serve qwen38/qwen38-k4, so every request returned HTTP 404 and "
            "the exact-match gate recorded null; the client now resolves the served id from "
            "/models. Owned and tightened by ImmutableImage afterwards.",
        }
    fidelity = {
        path.stem.replace("fidelity-vision-", ""): load(path)
        for path in sorted(art.glob("fidelity-vision-*.json"))
    }
    needles = {
        path.stem.replace("fidelity-needle-", ""): load(path)
        for path in sorted(art.glob("fidelity-needle-*.json"))
    }

    receipt = {
        "schema": "qwen38-perf-sweep-5090/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "measure the open performance levers on the physical RTX 5090 and keep the "
        "configuration that wins on accepted tokens per step over step time",
        "measured_on": {
            "host": stage.get("host"),
            "gpu": stage.get("gpu"),
            "gpu_count": 1,
            "note": "every number in this receipt was measured on this one physical "
            "GeForce RTX 5090. No number here is comparable to the rental RTX PRO 6000 "
            "engine-budget receipts, and none was measured on it.",
        },
        "artifact_under_test": {
            "image": stage.get("image"),
            "image_manifest_digest": stage.get("image_manifest_digest"),
            "image_config_id": stage.get("image_config_id"),
            "patch_modules_verified_inside_image": stage.get(
                "patch_modules_verified_inside_image"
            ),
            "source_bind_mounts_used": stage.get("source_bind_mounts_used", []),
            "checkpoint": "malaiwah/Qwen3.8-27B-EXL3-K5K6-context",
            "podman_version": stage.get("podman_version"),
        },
        "profile": {
            "declared": "max_num_seqs=8 concurrency serving profile",
            "not_the_qualified_profile": "the rank-1 hardware qualification is single-sequence "
            "(--max-num-seqs 1, cudagraph_capture_sizes [4]); these rows widen the graph pool "
            "to every request count up to 8, which costs KV tokens. Do not quote them as the "
            "qualification's numbers.",
            "constant_across_every_row": [
                "--max-model-len 262144",
                "--kv-cache-dtype fp8",
                "--gpu-memory-utilization 0.97 for every graph-decode row; the three eager rows "
                "use 0.95 because cudagraph_mode NONE turns the freed graph pool into KV and "
                "OOMs at 0.97, so they are compared only against each other",
                "--max-num-batched-tokens 2048",
                "cudagraph_mode=FULL_DECODE_ONLY for every row except the three eager rows",
                "VLLM_EXL3_EMBED_BITS=8",
                "VLLM_EXL3_GRAPH_DECODE=1",
                "VLLM_EXL3_PREFILL_RECONSTRUCT_M=128",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "mm max_pixels 8388608",
            ],
            "jit_cache": "warm and shared with the sibling agents' runs, not cold; a row that "
            "compiled a new shape would show it as a first-repeat outlier, and every point "
            "discards one warm repeat before timing",
        },
        "protocol": {
            "prompts": "frozen token-id arrays, identical bytes for every configuration",
            "prompt_manifest": stage.get("prompt_manifest"),
            "decode": "256 output tokens, min_tokens=max_tokens, ignore_eos, one discarded "
            "warm repeat then three timed repeats per (temperature, concurrency)",
            "temperatures": "0 for the deterministic arm, 0.6 seeded for the "
            "acceptance-sensitive arm; acceptance moves with temperature so it is pinned",
            "prefill": "2,048 and 6,144 exact prompt tokens, max_tokens=1, three timed repeats",
            "step_count": "vllm:iteration_tokens_total_count delta (exact engine iterations)",
            "accepted_tokens_per_step": "(accepted draft tokens + drafts) / drafts, from "
            "vllm:spec_decode_* deltas around each timed repeat",
            "decision_metric": "accepted tokens per step / step time, reported alongside both "
            "factors so a depth change that buys acceptance and pays in latency is visible",
        },
        "rows": rows,
        "production_image_gates": gates
        or {
            "not_run_in_this_window": "no --production-image-receipt was passed",
        },
        "fidelity_guard": {
            "purpose": "a throughput win that breaks correctness is not a win",
            "image_suite": fidelity,
            "needle": needles,
            "reference": "the same 30-case generator scored 24/30 (digits 8, bars 9, grid 7) on "
            "the rank-1 qualification run of this checkpoint",
        },
        "bit_exactness": {
            "not_bit_exact": [
                "custom_ops:[\"all\"] replaces Inductor-generated elementwise kernels with the "
                "hand-written CUDA ops, so logits differ in the last bits",
                "FLASHINFER attention uses different reduction order and kernels than TRITON_ATTN",
                "any rejection_sample_method other than 'standard'",
            ],
            "consequence": "these levers are throughput/serving choices, not fidelity-neutral "
            "refactors; the published KLD and capability numbers were measured under the "
            "configuration they name, and a lever change needs the fidelity guard above",
        },
        "closed_avenues": {
            "B12X_ATTN": "requires block size exactly 64 or 128; this hybrid model's mamba page "
            "forces the attention block size to >=1600 (see startup facts), so the backend "
            "cannot load. Not run.",
            "cudagraph_mode=FULL_AND_PIECEWISE": "refused by the exl3 graph-decode guard for any "
            "mode whose mixed mode is not NONE, because mixed prefill token counts cannot be "
            "enumerated before capture; our own history also records Xid 31 on graph-captured "
            "short SM120 prefills. Not run.",
            "VLLM_EXL3_PREFILL_FP8": "silent no-op: the pinned exllamav3 extension exports "
            "neither reconstruct_fp8_slice nor reconstruct_had_slice. Not run.",
            "rejection_sample_method=synthetic": "accepts draft tokens against a calibrated "
            "synthetic rate instead of the target distribution, so it manufactures acceptance "
            "and changes outputs. Never a serving option; not run.",
            "--max-num-batched-tokens 8192": "already measured as noise on this geometry.",
            "--enable-prefix-caching": "needs the -apc superset image, which is explicitly not "
            "the hardware-qualified artifact; out of scope for this sweep.",
            "SparkInfer / b12x as a second engine": "the same component under two names in this "
            "image (identical commit label), so there is no second engine to enable.",
        },
        "winner": {
            "tag": args.winner,
            "rationale": args.winner_rationale,
            "decided_on": "accepted tokens per step divided by step time, not acceptance rate",
        },
        "unrun": args.unrun,
        "raw_artifact_sha256": logs,
        "harness_sha256": stage.get("harness_sha256"),
        "receipt_assembler_sha256": sha256(Path(__file__)),
    }

    out = Path(args.out)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps({"wrote": str(out), "rows": list(rows), "unrun": args.unrun}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
