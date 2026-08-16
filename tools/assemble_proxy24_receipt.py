#!/usr/bin/env python3
"""Assemble receipts/qualification-24gib-capped.json from the proxy24 run artifacts.

Input is a mirror of AIBoss:/home/mbelleau/proxy24/{out,logs}; output is the receipt
plus the prefixed evidence copies receipts/qualification-24gib-capped-*. Everything
numeric in the receipt comes from a parsed artifact, never from this script's opinion:
the only hand-written strings are the labelled prose blocks at the bottom.

    assemble_proxy24_receipt.py --raw <mirror dir> --repo <research repo>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

GIB = 1 << 30

# tag -> (arm, profile key, engine process role)
TAGS = {
    "a32e": ("budget", "mtp3_32768", "text_and_decode"),
    "a32m": ("budget", "mtp3_32768", "image"),
    "a45e": ("budget", "mtpoff_45056", "text_and_decode"),
    "a45m": ("budget", "mtpoff_45056", "image"),
    "b32e": ("board", "mtp3_32768", "text_and_decode"),
    "b32m": ("board", "mtp3_32768", "image"),
    "b45e": ("board", "mtpoff_45056", "text_and_decode"),
    "b45m": ("board", "mtpoff_45056", "image"),
    "p131k3": ("budget", "probe_window_131072_mtp3", "kv_law_probe"),
    "p131koff": ("budget", "probe_window_131072_mtpoff", "kv_law_probe"),
    "p262k3": ("budget", "probe_window_262144_mtp3", "kv_law_probe"),
    "p262koff": ("budget", "probe_window_262144_mtpoff", "kv_law_probe"),
    # kept deliberately: this is the run whose two --kv-cache-memory-bytes hints
    # established cudaMemGetInfo total to the byte, and whose printed "24.0 GiB"
    # budget was judged not to be evidence of anything at or under 24.0
    "preflight": ("budget", "discarded_calibration_util_0_7643", "discarded"),
}

FACT_PATTERNS = {
    "memory_line": re.compile(
        r"Free memory on device \((?P<free>[0-9.]+)/(?P<total>[0-9.]+) GiB\) on startup\. "
        r"Desired GPU memory utilization is \((?P<util>[0-9.]+), (?P<budget>[0-9.]+) GiB\)\. "
        r"Actual usage is (?P<weight>[0-9.]+) GiB for weight, (?P<activation>[0-9.]+) GiB "
        r"for peak activation, (?P<non_torch>[0-9.]+) GiB for non-torch memory, and "
        r"(?P<cudagraph>[0-9.]+) GiB for CUDAGraph memory"),
    "kv_advice": re.compile(
        r"`--kv-cache-memory-bytes=(?P<to_requested>[0-9]+)` \([0-9.]+ GiB\) to fit into "
        r"requested memory, or `--kv-cache-memory-bytes=(?P<to_gpu>[0-9]+)`"),
    "available_kv": re.compile(r"Available KV cache memory: (?P<gib>[0-9.]+) GiB"),
    "kv_tokens": re.compile(r"GPU KV cache size: (?P<tokens>[0-9,]+) tokens"),
    "concurrency": re.compile(
        r"Maximum concurrency for (?P<length>[0-9,]+) tokens per request: (?P<x>[0-9.]+)x"),
    "block_size": re.compile(r"Setting attention block size to (?P<n>[0-9]+) tokens"),
    "padding_layers": re.compile(
        r"Add (?P<layers>[0-9]+) padding layers, may waste at most (?P<pct>[0-9.]+)%"),
    "mamba_padding": re.compile(r"Padding mamba page size by (?P<pct>[0-9.]+)%"),
    "spec_tokens": re.compile(r"num_spec_tokens=(?P<n>[0-9]+)"),
    "prefix_caching": re.compile(r"enable_prefix_caching=(?P<v>[A-Za-z]+)"),
    "model_load": re.compile(
        r"Model loading took (?P<gib>[0-9.]+) GiB memory and (?P<sec>[0-9.]+) seconds"),
    "graph_capture": re.compile(
        r"Graph capturing finished in (?P<sec>[0-9]+) secs, took (?P<gib>[0-9.]+) GiB"),
    "init_engine": re.compile(
        r"init engine \(profile, create kv cache, warmup model\) took (?P<sec>[0-9.]+) s "
        r"\(compilation: (?P<comp>[0-9.]+) s\)"),
    "engine_version": re.compile(r"Initializing a V1 LLM engine \((?P<v>v[^)]+)\)"),
    "kv_needed": re.compile(r"(?P<gib>[0-9.]+) GiB KV cache is needed"),
}

# The startup refusal is the cheapest measurement in this receipt: it prints the KV a
# single request of the configured window needs, the pool that was on hand, and the
# window that pool would support. Three numbers, one 100-second run, no regression.
REFUSAL = re.compile(
    r"To serve at least one request with the model's max seq len \((?P<window>[0-9]+)\), "
    r"\((?P<needed_gib>[0-9.]+) GiB KV cache is needed, which is larger than the available "
    r"KV cache memory \((?P<available_gib>[0-9.]+) GiB\)\. Based on the available memory, "
    r"the estimated maximum model length is (?P<estimated_max_len>[0-9]+)")


def parse_refusal(lines: list[str]) -> dict:
    for line in lines:
        match = REFUSAL.search(line)
        if match:
            got = match.groupdict()
            return {
                "configured_window": int(got["window"]),
                "kv_gib_needed_for_one_request": float(got["needed_gib"]),
                "available_kv_gib": float(got["available_gib"]),
                "estimated_maximum_model_length": int(got["estimated_max_len"]),
                "raw": line.strip()[:400],
            }
    return {}

SPEC_COUNTERS = ("vllm:spec_decode_num_drafts_total",
                 "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() and path.stat().st_size else None


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def parse_facts(text: str) -> dict:
    """Pull the engine's own startup numbers out of the captured fact lines."""
    facts: dict = {"raw_lines": sorted(l for l in text.splitlines() if l.strip())}
    for name, pattern in FACT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            facts[name] = {k: v for k, v in match.groupdict().items()}
    return facts


def numeric_startup(facts: dict) -> dict:
    """The subset a reader wants as numbers, with the byte-exact budget where derivable."""
    out: dict = {}
    line = facts.get("memory_line")
    if line:
        out.update({
            "gpu_memory_utilization_requested": float(line["util"]),
            "engine_budget_gib_logged": float(line["budget"]),
            "free_memory_at_engine_init_gib_logged": float(line["free"]),
            "cuda_total_gib_logged": float(line["total"]),
            "weight_gib": float(line["weight"]),
            "peak_activation_gib": float(line["activation"]),
            "non_torch_gib": float(line["non_torch"]),
            "cudagraph_gib": float(line["cudagraph"]),
        })
        out["non_kv_sum_gib"] = round(
            out["weight_gib"] + out["peak_activation_gib"]
            + out["non_torch_gib"] + out["cudagraph_gib"], 3)
    advice = facts.get("kv_advice")
    if advice and line:
        to_requested = int(advice["to_requested"])
        to_gpu = int(advice["to_gpu"])
        # both advice figures share the same non-KV term and the same 150 MiB redundancy
        # buffer, so their difference is exactly free - requested, in bytes
        out["kv_cache_memory_bytes_to_requested_limit"] = to_requested
        out["kv_cache_memory_bytes_to_gpu_limit"] = to_gpu
        out["free_minus_requested_bytes"] = to_gpu - to_requested
        out["free_minus_requested_gib"] = round((to_gpu - to_requested) / GIB, 4)
        out["available_kv_bytes_derived"] = to_requested + 150 * (1 << 20)
        out["available_kv_gib_derived"] = round(out["available_kv_bytes_derived"] / GIB, 4)
    if "available_kv" in facts:
        out["available_kv_gib_logged"] = float(facts["available_kv"]["gib"])
    if "kv_tokens" in facts:
        out["gpu_kv_cache_tokens"] = int(facts["kv_tokens"]["tokens"].replace(",", ""))
    if "concurrency" in facts:
        out["maximum_concurrency"] = float(facts["concurrency"]["x"])
        out["concurrency_at_tokens"] = int(facts["concurrency"]["length"].replace(",", ""))
    for key, field, cast in (("block_size", "n", int), ("spec_tokens", "n", int),
                             ("mamba_padding", "pct", float)):
        if key in facts:
            out[f"attention_{key}" if key == "block_size" else key] = cast(facts[key][field])
    if "padding_layers" in facts:
        out["kv_padding_layers"] = int(facts["padding_layers"]["layers"])
        out["kv_padding_waste_pct"] = float(facts["padding_layers"]["pct"])
    if "prefix_caching" in facts:
        out["enable_prefix_caching"] = facts["prefix_caching"]["v"] == "True"
    if "graph_capture" in facts:
        out["cudagraph_capture_gib"] = float(facts["graph_capture"]["gib"])
    if "init_engine" in facts:
        out["init_engine_sec"] = float(facts["init_engine"]["sec"])
        out["compilation_sec"] = float(facts["init_engine"]["comp"])
    if "engine_version" in facts:
        out["vllm_version"] = facts["engine_version"]["v"].lstrip("v")
    if "kv_needed" in facts:
        out["kv_gib_needed_for_one_request"] = float(facts["kv_needed"]["gib"])
    if "gpu_kv_cache_tokens" in out and out.get("available_kv_bytes_derived"):
        out["bytes_per_kv_token_observed"] = round(
            out["available_kv_bytes_derived"] / out["gpu_kv_cache_tokens"], 1)
    return out


def parse_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    values: dict = {}
    for line in path.read_text().splitlines():
        for counter in SPEC_COUNTERS:
            if line.startswith(counter):
                try:
                    values[counter] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            match = re.search(r'position="(\d+)".*?\s([0-9.eE+-]+)$', line)
            if match:
                values.setdefault("per_position", {})[f"position={match.group(1)}"] = \
                    float(match.group(2))
    return values


def mtp_acceptance(before: dict, after: dict, depth: int) -> dict:
    if not before or not after:
        return {"available": False, "why": "metrics endpoint returned no spec_decode counters"}
    drafts = after[SPEC_COUNTERS[0]] - before[SPEC_COUNTERS[0]]
    draft_tokens = after[SPEC_COUNTERS[1]] - before[SPEC_COUNTERS[1]]
    accepted = after[SPEC_COUNTERS[2]] - before[SPEC_COUNTERS[2]]
    result = {
        "available": True,
        "source": "vLLM /metrics spec_decode counters, differenced across the gate",
        "counters_before": {k: before.get(k) for k in SPEC_COUNTERS},
        "counters_after": {k: after.get(k) for k in SPEC_COUNTERS},
        "drafts": drafts, "draft_tokens": draft_tokens, "accepted_tokens": accepted,
        "mtp_depth": depth,
        "draft_token_acceptance_rate": round(accepted / draft_tokens, 4) if draft_tokens else None,
        "mean_accepted_length": round(1 + accepted / drafts, 4) if drafts else None,
    }
    per_pos_before = before.get("per_position", {})
    per_pos_after = after.get("per_position", {})
    if per_pos_after:
        delta = {k: per_pos_after[k] - per_pos_before.get(k, 0.0) for k in sorted(per_pos_after)}
        result["accepted_per_position"] = delta
        if drafts:
            result["per_position_acceptance_rate"] = {
                k: round(v / drafts, 4) for k, v in delta.items()}
    return result


def bench_summary(path: Path) -> dict:
    """The last JSON object bench.py printed is its summary."""
    if not path.is_file():
        return {"available": False}
    payload = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "runs" in candidate:
                payload = candidate
    if not payload:
        return {"available": False, "why": "bench.py printed no summary object"}
    rates = sorted(r["per_stream_tok_s"] for r in payload["runs"])
    mid = rates[len(rates) // 2] if len(rates) % 2 else (rates[len(rates) // 2 - 1]
                                                         + rates[len(rates) // 2]) / 2
    return {
        "available": True,
        "protocol": "one warm-up request, then three timed runs, greedy, ignore_eos, "
                    "256 output tokens each, concurrency 1",
        "runs": payload["runs"],
        "n_runs": len(payload["runs"]),
        "output_tokens_per_request": payload.get("output_tokens_per_request"),
        "median_decode_tok_s": round(mid, 2),
        "min_decode_tok_s": rates[0],
        "max_decode_tok_s": rates[-1],
        "dispersion_range_tok_s": round(rates[-1] - rates[0], 3),
        "dispersion_relative_pct": round(100 * (rates[-1] - rates[0]) / mid, 3) if mid else None,
        "ttft_proxy_sec": payload.get("ttft_proxy_sec"),
    }


def phase_peaks(rows: list[dict]) -> dict:
    peaks: dict = {}
    for row in rows:
        phase = row.get("phase")
        used = row.get("memory_used_mib")
        if phase is None or used is None:
            continue
        current = peaks.get(phase)
        if current is None or used > current["memory_used_mib"]:
            peaks[phase] = {"memory_used_mib": used,
                            "memory_used_gib": round(used / 1024, 3),
                            "memory_free_mib": row.get("memory_free_mib"),
                            "utc": row.get("utc")}
    return peaks


def collect(raw: Path) -> dict:
    out_dir, log_dir = raw / "out", raw / "logs"
    runs: dict = {}
    for tag, (arm, profile, role) in TAGS.items():
        command = read_json(out_dir / f"command-{tag}.json")
        if command is None:
            continue
        facts_path = out_dir / f"startup-facts-{tag}.txt"
        fail_path = out_dir / f"startup-fail-{tag}.txt"
        gates = {row["gate"]: row for row in read_jsonl(out_dir / f"gates-{tag}.jsonl")}
        startup_ok = facts_path.is_file() and facts_path.stat().st_size > 0
        facts = parse_facts(facts_path.read_text()) if startup_ok else {}
        entry = {
            "tag": tag, "arm": arm, "profile": profile, "engine_process_role": role,
            "config": {k: command[k] for k in (
                "gpu_memory_utilization", "max_model_len", "mtp_depth", "max_pixels",
                "ballast_target_free_gib", "pytorch_cuda_alloc_conf", "model_revision",
                "prefix_caching_flag_passed")},
            "podman_argv": command["podman_argv"],
            "vllm_argv": command["vllm_argv"],
            "startup_succeeded": startup_ok,
            "startup_wall_sec": gates.get("gate1_startup_native_allocation", {}).get("wall_sec"),
            "gate_status": {g: {"rc": r["rc"], "wall_sec": r["wall_sec"]}
                            for g, r in gates.items()},
            "point_samples": {row["name"]: row
                              for row in read_jsonl(out_dir / f"point-samples-{tag}.jsonl")},
            "phase_peak_memory": phase_peaks(read_jsonl(log_dir / f"gpu-samples-{tag}.jsonl")),
        }
        if startup_ok:
            entry["startup_facts"] = facts
            entry["startup"] = numeric_startup(facts)
        elif fail_path.is_file():
            lines = [l for l in fail_path.read_text().splitlines() if l]
            entry["startup_failure_lines"] = lines
            entry["startup_refusal"] = parse_refusal(lines)
            entry["startup_partial_facts"] = numeric_startup(parse_facts("\n".join(lines)))
        ballast = read_json(out_dir / f"ballast-{tag}.json")
        if ballast:
            entry["ballast"] = ballast
        # nvidia-smi cannot attribute memory per process here, so the board arm's card peaks
        # include the ballast. Subtract its measured device footprint to get the engine's own
        # peak, then compare that with the CUDA total a physical 24 GiB board would report:
        # 24,576 MiB nominal minus the 458 MiB framebuffer reserve measured on this card.
        ballast_mib = round(ballast["device_used_after_ballast_gib"] * 1024) if ballast else 0
        entry["engine_only_peak_mib"] = {
            phase: {
                "card_used_mib": row["memory_used_mib"],
                "ballast_mib": ballast_mib,
                "engine_only_mib": row["memory_used_mib"] - ballast_mib,
                "margin_against_a_24gib_board_mib":
                    24118 - (row["memory_used_mib"] - ballast_mib),
            }
            for phase, row in entry["phase_peak_memory"].items()
            if phase not in ("before_load", "ballast", "after_release", "shutdown")
        }
        for key, filename in (("gate2_long_needle", f"g2-needle-{tag}.json"),
                              ("gate6_second_long_request", f"gate6-needle-{tag}.json"),
                              ("gate3_combined_text_and_image", f"gate3-longmm-{tag}.json"),
                              ("gate4_image_suite", f"gate4-vision-{tag}.json")):
            payload = read_json(out_dir / filename)
            if payload is not None:
                if key == "gate4_image_suite":
                    payload = {k: payload[k] for k in ("label", "seed", "per_task", "overall")
                               } | {"cases": payload.get("results", [])}
                entry[key] = payload
        bench = bench_summary(log_dir / f"g5-{tag}.out")
        if bench.get("available"):
            entry["gate5_decode"] = bench
            entry["gate5_mtp_acceptance"] = mtp_acceptance(
                parse_metrics(out_dir / f"metrics-before-g5-{tag}.txt"),
                parse_metrics(out_dir / f"metrics-after-g5-{tag}.txt"),
                command["mtp_depth"])
        runs[tag] = entry
    return runs


def copy_evidence(raw: Path, repo: Path, runs: dict) -> dict:
    prefix = "qualification-24gib-capped-"
    receipts = repo / "receipts"
    evidence: dict = {}
    for tag in runs:
        for source, name in (
            (raw / "logs" / f"server-{tag}.log", f"server-{tag}.log"),
            (raw / "logs" / f"gpu-samples-{tag}.jsonl", f"gpu-samples-{tag}.jsonl"),
            (raw / "out" / f"gates-{tag}.jsonl", f"gates-{tag}.jsonl"),
            (raw / "out" / f"command-{tag}.json", f"command-{tag}.json"),
            (raw / "out" / f"startup-facts-{tag}.txt", f"startup-facts-{tag}.txt"),
            (raw / "logs" / f"ballast-{tag}.log", f"ballast-{tag}.log"),
            (raw / "logs" / f"driver-{tag}.log", f"driver-{tag}.log"),
            (raw / "out" / f"startup-fail-{tag}.txt", f"startup-fail-{tag}.txt"),
        ):
            if source.is_file() and source.stat().st_size:
                target = receipts / (prefix + name)
                shutil.copyfile(source, target)
                evidence[target.name] = sha256(target)
    # the three chain logs are the only place the first, compute-mode-blocked board
    # attempt survives in full, since each rerun truncates its own per-tag artifacts
    for name in ("chain.log", "chain-board.log", "chain-probe2.log"):
        source = raw / "logs" / name
        if source.is_file() and source.stat().st_size:
            target = receipts / (prefix + name)
            shutil.copyfile(source, target)
            evidence[target.name] = sha256(target)
    return dict(sorted(evidence.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--repo", default="/home/mbelleau/research")
    parser.add_argument("--narrative", required=True,
                        help="JSON file holding the hand-written prose blocks")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    raw, repo = Path(args.raw), Path(args.repo)
    runs = collect(raw)
    if not runs:
        raise SystemExit(f"no run artifacts under {raw}")
    narrative = json.loads(Path(args.narrative).read_text())
    evidence = copy_evidence(raw, repo, runs)

    receipt = {
        "schema": "qwen38-qualification-24gib-capped/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_board_gate": "open",
    }
    receipt.update(narrative.get("head", {}))
    receipt["runs"] = runs
    receipt.update(narrative.get("body", {}))
    receipt["evidence_files"] = evidence

    target = Path(args.out) if args.out else repo / "receipts" / "qualification-24gib-capped.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"wrote {target} with {len(runs)} runs and {len(evidence)} evidence files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
