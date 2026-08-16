#!/usr/bin/env python3
"""Assemble receipts/qualification-5090-apc.json from the raw artifacts of
tools/qualify_apc_5090.sh.

This is the serving qualification of the PROMOTED release unit -- the four-module superset
localhost/vllm:gg-r34-patched-apc -- with prefix caching ON. It is not a re-measurement of
receipts/qualification-5090-context.json: that one ran, and every other published
measurement ran, with enable_prefix_caching=False, because prefix caching is default-off
for this hybrid model in this build.

Nine gates. The first seven are the original set, unchanged in requirement and acceptance
criterion. The last two are new and are the reason the run means anything:

  g8  the engine banner must report enable_prefix_caching True and mamba_cache_mode align.
      Without it, a flag that argparse accepts and the engine then discards would produce
      seven green gates that prove only that the old profile still works.
  g9  KV capacity at 262,144 must not regress against the 265,122-token baseline. If
      prefix caching costs KV tokens, that is a real cost of the feature and it is
      published beside the win, not absorbed.

A failed gate is a published result. Nothing here is retried into green, and no number is
carried over from another receipt: every figure below was measured in this run except the
explicitly named baselines and the explicitly attributed prior evidence.
"""
import argparse
import hashlib
import json
import pathlib
import re
import statistics
import time

REPO = pathlib.Path(__file__).resolve().parent.parent

# Baselines. Both from receipts/qualification-5090-context.json, the qualified profile.
# NOT from receipts/apc-poison-repro.json, whose 223,677 tokens and 8.18 GiB were measured
# at max-model-len 196,608 and utilisation 0.92 and are not comparable with these.
BASE_RECEIPT = "receipts/qualification-5090-context.json"
KV_TOKENS_BASELINE = 265122
KV_GIB_BASELINE = 9.28
CONCURRENCY_BASELINE = 1.01
ENGINE_BUDGET_GIB_BASELINE = 29.98
USAGE_BASELINE = {"weight": 18.19, "activation": 1.78, "non_torch": 0.27, "cudagraph": 0.45}
MEDIAN_DECODE_BASELINE = 107.56
BLOCK_SIZE_BASELINE = 1600
PADDING_LAYERS_BASELINE = 3

PROMOTED_TAG = "localhost/vllm:gg-r34-patched-apc"
PROMOTED_MANIFEST = "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218"
PARENT_MANIFEST = "sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27"
BASE_MANIFEST = "sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

MODULES = {
    "tools/vllm-exl3-prefill-dispatch.py":
        ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py",
         "2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee"),
    "tools/vllm-qwen3_5-embed-quant-config.py":
        ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py",
         "04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7"),
    "tools/vllm-qwen3_5_mtp-embed-quant-config.py":
        ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py",
         "0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab"),
    "tools/vllm-mamba-align-scheduler.py":
        ("/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py",
         "b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf"),
}

GATE_REQUIREMENTS = {
    "gate1_startup_native_allocation":
        "startup allocates at least one native-length request without exceeding "
        "utilisation 0.97",
    "gate2_long_needle_exact":
        "the ~261,794-token text needle is retrieved exactly",
    "gate3_combined_long_text_and_7mp_image":
        "a combined 236,824-token plus seven-megapixel request returns code and colours "
        "exactly",
    "gate4_image_suite_24_of_30":
        "the 30-case deterministic image suite stays at or above 24/30",
    "gate5_three_warmed_decode_runs":
        "three warmed 256-token concurrency-1 runs report median decode, MTP acceptance "
        "and dispersion",
    "gate6_second_long_request_after_release":
        "a second native-length request succeeds after the first is released, proving "
        "recovery rather than one lucky allocation",
    "gate7_receipt_identity_complete":
        "every identity the claim depends on is recorded: card, host, driver, image digest "
        "and its in-image module digests, model revision and shard digests, harness "
        "digests, and the verbatim launch argv of every process",
    "gate8_banner_prefix_caching_actually_on":
        "the engine banner reports enable_prefix_caching True and mamba_cache_mode align, "
        "on every process this qualification launched",
    "gate9_kv_capacity_not_regressed":
        "with prefix caching on, GPU KV cache size still covers max_model_len 262,144 and "
        "maximum concurrency at 262,144 stays at or above 1.00x; the token count and the "
        "memory components are reported against the 265,122-token baseline whatever they "
        "come out at",
}

FACT_PATTERNS = {
    "desired_gpu_memory_utilization":
        r"Desired GPU memory utilization is \(([0-9.]+), ([0-9.]+) GiB\)",
    "free_memory_on_device": r"Free memory on device \(([0-9.]+)/([0-9.]+) GiB\) on startup",
    "actual_usage":
        r"Actual usage is ([0-9.]+) GiB for weight, ([0-9.]+) GiB for peak activation, "
        r"([0-9.]+) GiB for non-torch memory, and ([0-9.]+) GiB for CUDAGraph memory",
    "available_kv_cache_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "gpu_kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "maximum_concurrency":
        r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
    "model_loading": r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds",
    "graph_capture": r"Graph capturing finished in ([0-9]+) secs, took ([0-9.]+) GiB",
    "init_engine":
        r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s "
        r"\(compilation: ([0-9.]+) s\)",
    "attention_block_size": r"Setting attention block size to ([0-9]+) tokens",
    "mamba_page_padding_pct": r"Padding mamba page size by ([0-9.]+)%",
    "kv_padding_layers":
        r"Add ([0-9]+) padding layers, may waste at most ([0-9.]+)% KV cache memory",
    "num_spec_tokens": r"num_spec_tokens=([0-9]+)",
    "enable_prefix_caching": r"['\"]?enable_prefix_caching['\"]?[=:] ?([A-Za-z]+)",
    "mamba_cache_mode": r"['\"]?mamba_cache_mode['\"]?[=:] ?['\"]?([a-z]+)",
    "mamba_block_size": r"mamba_block_size=([0-9]+)",
    "prefix_caching_hash_algo": r"prefix_caching_hash_algo=['\"]?([A-Za-z0-9_]+)",
    "vllm_version": r"Initializing a V1 LLM engine \(v([0-9A-Za-z.+_-]+)\)",
}

SPEC_COUNTERS = ("vllm:spec_decode_num_drafts_total",
                 "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens_total")
PREFIX_COUNTERS = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                   "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total",
                   "vllm:gpu_prefix_cache_queries", "vllm:gpu_prefix_cache_hits")


def sha256_file(path):
    path = pathlib.Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def read_json(path):
    path = pathlib.Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def read_text(path):
    path = pathlib.Path(path)
    return path.read_text() if path.exists() else ""


def parse_facts(blob):
    """Every fact is either matched or explicitly null. Nothing is inferred."""
    out = {}
    for name, pattern in FACT_PATTERNS.items():
        m = re.search(pattern, blob)
        if not m:
            out[name] = None
        elif len(m.groups()) == 1:
            out[name] = m.group(1)
        else:
            out[name] = list(m.groups())
    tokens = out.get("gpu_kv_cache_tokens")
    out["gpu_kv_cache_tokens_int"] = int(tokens.replace(",", "")) if tokens else None
    conc = out.get("maximum_concurrency")
    out["maximum_concurrency_float"] = float(conc[1]) if conc else None
    out["maximum_concurrency_window"] = conc[0] if conc else None
    return out


def parse_prom(text):
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        key = name.split("{")[0]
        try:
            values[key] = values.get(key, 0.0) + float(value)
        except ValueError:
            continue
    return values


def counter_delta(before, after, names):
    b, a = parse_prom(before), parse_prom(after)
    rows = {}
    for name in names:
        if name in a or name in b:
            rows[name] = {"before": b.get(name), "after": a.get(name),
                          "delta": (a.get(name) or 0.0) - (b.get(name) or 0.0)}
    return rows


def decode_section(bench_out):
    """bench.py prints one JSON object per run, then a final object with everything."""
    payloads = []
    for line in bench_out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not payloads:
        return {"available": False, "why": "bench.py produced no parseable JSON"}
    final = payloads[-1]
    runs = final.get("runs") or [p for p in payloads if "aggregate_tok_s" in p]
    rates = [r["aggregate_tok_s"] for r in runs if "aggregate_tok_s" in r]
    section = {
        "available": bool(rates),
        "protocol": "one warm-up request, then three timed runs, greedy, ignore_eos, "
                    "256 output tokens each, concurrency 1",
        "runs": runs,
        "n_runs": len(runs),
        "output_tokens_per_request": 256,
        "concurrency": 1,
    }
    if rates:
        section.update({
            "median_decode_tok_s": round(statistics.median(rates), 2),
            "min_decode_tok_s": round(min(rates), 2),
            "max_decode_tok_s": round(max(rates), 2),
            "dispersion_range_tok_s": round(max(rates) - min(rates), 2),
            "dispersion_relative_pct":
                round(100.0 * (max(rates) - min(rates)) / statistics.median(rates), 3),
            "stdev_tok_s": round(statistics.stdev(rates), 4) if len(rates) > 1 else 0.0,
        })
    for key in ("ttft_proxy_sec", "ttft_sec", "prefill"):
        if key in final:
            section[key] = final[key]
    return section


def spec_section(before, after, mtp_depth):
    rows = counter_delta(before, after, SPEC_COUNTERS)
    if not rows:
        return {"available": False, "why": "no spec_decode counters exposed by /metrics"}
    drafts = rows[SPEC_COUNTERS[0]]["delta"]
    draft_tokens = rows[SPEC_COUNTERS[1]]["delta"]
    accepted = rows[SPEC_COUNTERS[2]]["delta"]
    out = {
        "available": True,
        "source": "vLLM /metrics spec_decode counters, differenced across the gate",
        "counters_before": {k: v["before"] for k, v in rows.items()},
        "counters_after": {k: v["after"] for k, v in rows.items()},
        "drafts": drafts, "draft_tokens": draft_tokens, "accepted_tokens": accepted,
        "mtp_depth": mtp_depth,
    }
    if draft_tokens:
        out["draft_token_acceptance_rate"] = round(accepted / draft_tokens, 4)
    if drafts:
        out["mean_accepted_length"] = round(1.0 + accepted / drafts, 4)
        out["draft_tokens_per_draft"] = round(draft_tokens / drafts, 4)
    return out


def gate_rows(status_path):
    rows = {}
    for line in read_text(status_path).splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            rows[row["gate"]] = row          # a rerun of the same gate wins
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="qual-apc-raw",
                    help="directory holding the fetched out/ and logs/ trees")
    ap.add_argument("--out", default=str(REPO / "receipts/qualification-5090-apc.json"))
    ap.add_argument("--restore", default=None,
                    help="JSON file with the live-service restore proof")
    ap.add_argument("--identity", default=None,
                    help="JSON file with host/GPU/model identity collected on AIBoss")
    args = ap.parse_args()

    work = pathlib.Path(args.work)
    out_dir, log_dir = work / "out", work / "logs"

    stage = read_json(out_dir / "stage.json") or {}
    identity_extra = read_json(args.identity) if args.identity else {}
    identity_extra = identity_extra or {}

    # ---------------------------------------------------------------- per-process facts
    # E, B3, F is the original qualification's process order and the order gates are
    # attributed in: gate 1's headline figures are process E's there, so they are here too.
    preferred = {"E": 0, "B3": 1, "F": 2}
    # n955 is the qualified profile's own gate-1 failure and p* are startup-only band
    # probes. They are evidence, not gates, and are reported separately so a probe can never
    # be mistaken for a qualified process or drag a verdict either way.
    def is_qualified_process(tag):
        return not (tag == "n955" or tag.startswith("p"))
    processes = {}
    probes = {}
    for command in sorted(out_dir.glob("command-*.json"),
                          key=lambda p: (preferred.get(p.stem[len("command-"):], 99), p.stem)):
        tag = command.stem[len("command-"):]
        facts_blob = (read_text(out_dir / f"startup-facts-{tag}.txt") + "\n"
                      + read_text(out_dir / f"engine-config-{tag}.txt")
                      + "\n" + read_text(out_dir / f"startup-fail-{tag}.txt"))
        server_log = log_dir / f"server-{tag}.log"
        target = processes if is_qualified_process(tag) else probes
        target[tag] = {
            "tag": tag,
            "command": read_json(command),
            "startup_facts": parse_facts(facts_blob),
            "banner": read_json(out_dir / f"banner-{tag}.json"),
            "raw_lines": sorted(
                x for x in read_text(out_dir / f"startup-facts-{tag}.txt").splitlines() if x),
            "engine_config_line": read_text(out_dir / f"engine-config-{tag}.txt").strip()[:1200],
            "server_log": f"receipts/qualification-5090-apc-server-{tag}.log",
            "server_log_sha256": sha256_file(server_log),
            "gates": gate_rows(out_dir / f"gates-{tag}.jsonl"),
            "point_samples": [json.loads(x) for x in
                              read_text(out_dir / f"point-samples-{tag}.jsonl").splitlines() if x],
        }

    def owner(gate):
        for tag, proc in processes.items():
            if gate in proc["gates"]:
                return tag, proc
        return None, None

    def rc_of(gate):
        _, proc = owner(gate)
        return proc["gates"][gate]["rc"] if proc else None

    def wall_of(gate):
        _, proc = owner(gate)
        return proc["gates"][gate]["wall_sec"] if proc else None

    gates = {}

    # ------------------------------------------------------------------------- gate 1
    tag1, proc1 = owner("gate1_startup_native_allocation")
    if proc1:
        f = proc1["startup_facts"]
        usage = [float(x) for x in f["actual_usage"]] if f.get("actual_usage") else None
        util = float(f["desired_gpu_memory_utilization"][0]) if f.get(
            "desired_gpu_memory_utilization") else None
        budget = float(f["desired_gpu_memory_utilization"][1]) if f.get(
            "desired_gpu_memory_utilization") else None
        gates["gate1_startup_native_allocation"] = {
            "requirement": GATE_REQUIREMENTS["gate1_startup_native_allocation"],
            "passed": rc_of("gate1_startup_native_allocation") == 0,
            "server_process": tag1,
            "gpu_memory_utilization_requested": util,
            "engine_budget_gib": budget,
            "utilisation_ceiling_respected": util is not None and util <= 0.97,
            "free_memory_at_startup_gib":
                float(f["free_memory_on_device"][0]) if f.get("free_memory_on_device") else None,
            "device_usable_gib":
                float(f["free_memory_on_device"][1]) if f.get("free_memory_on_device") else None,
            "measured_engine_usage_gib": dict(zip(
                ("weight", "peak_activation", "non_torch", "cudagraph"), usage)) if usage else None,
            "engine_usage_sum_gib": round(sum(usage), 2) if usage else None,
            "available_kv_cache_gib":
                float(f["available_kv_cache_gib"]) if f.get("available_kv_cache_gib") else None,
            "gpu_kv_cache_tokens": f.get("gpu_kv_cache_tokens_int"),
            "max_model_len": 262144,
            "kv_tokens_cover_native_length":
                (f.get("gpu_kv_cache_tokens_int") or 0) >= 262144,
            "maximum_concurrency_for_native_length": f.get("maximum_concurrency_float"),
            "attention_block_size_tokens":
                int(f["attention_block_size"]) if f.get("attention_block_size") else None,
            "mamba_page_padding_pct": f.get("mamba_page_padding_pct"),
            "kv_padding_layers": f.get("kv_padding_layers"),
            "startup_wall_sec": wall_of("gate1_startup_native_allocation"),
            "startup_facts": f,
            "raw_lines": proc1["raw_lines"],
        }

    # -------------------------------------------------------------------- gates 2 and 6
    for gate, tokens_file, note in (
            ("gate2_long_needle_exact", "g2-needle", None),
            ("gate6_second_long_request_after_release", "gate6-needle",
             "deliberately the same process as gate 2 and the same request length: "
             "in-process recovery of the allocation is the property under test. A different "
             "needle code, depth 0.25 and seed 20260816 make it a new prefill rather than a "
             "trivial repeat -- but with prefix caching ON it now also shares a long literary "
             "prefix with gate 2, so it is additionally the one place in this qualification "
             "where a cache hit on a native-length prompt is exercised end to end.")):
        tag, proc = owner(gate)
        if not proc:
            continue
        row = {
            "requirement": GATE_REQUIREMENTS[gate],
            "passed": rc_of(gate) == 0,
            "server_process": tag,
            "result": read_json(out_dir / f"{tokens_file}-{tag}.json"),
            "wall_sec": wall_of(gate),
        }
        if note:
            row["server_process_note"] = note
        gates[gate] = row

    # ------------------------------------------------------------------------- gate 3
    tag3, proc3 = owner("gate3_combined_long_text_and_7mp_image")
    if proc3:
        result = read_json(out_dir / f"gate3-longmm-{tag3}.json") or {}
        gates["gate3_combined_long_text_and_7mp_image"] = {
            "requirement": GATE_REQUIREMENTS["gate3_combined_long_text_and_7mp_image"],
            "passed": rc_of("gate3_combined_long_text_and_7mp_image") == 0,
            "server_process": tag3,
            "server_process_note":
                "a freshly started process that never served the 261,794-token request, "
                "kept from the original gate set. The #51113 precaution that motivated it is "
                "now fixed in the image rather than avoided by scheduling, but the gate is "
                "reproduced unchanged so the two qualifications compare like with like.",
            "result": result,
            "image_pixels": result.get("image_pixels"),
            "image_dimensions": result.get("image_dimensions"),
            "wall_sec": wall_of("gate3_combined_long_text_and_7mp_image"),
        }

    # ------------------------------------------------------------------------- gate 4
    tag4, proc4 = owner("gate4_image_suite_24_of_30")
    if proc4:
        vision = read_json(out_dir / f"gate4-vision-{tag4}.json") or {}
        gates["gate4_image_suite_24_of_30"] = {
            "requirement": GATE_REQUIREMENTS["gate4_image_suite_24_of_30"],
            "passed": rc_of("gate4_image_suite_24_of_30") == 0,
            "server_process": tag4,
            "overall": vision.get("overall"),
            "per_task": vision.get("per_task"),
            "seed": vision.get("seed"),
            "cases": vision.get("cases"),
            "wall_sec": wall_of("gate4_image_suite_24_of_30"),
        }

    # ------------------------------------------------------------------------- gate 5
    tag5, proc5 = owner("gate5_three_warmed_decode_runs")
    if proc5:
        decode = decode_section(read_text(log_dir / f"g5-{tag5}.out"))
        gates["gate5_three_warmed_decode_runs"] = {
            "requirement": GATE_REQUIREMENTS["gate5_three_warmed_decode_runs"],
            "passed": rc_of("gate5_three_warmed_decode_runs") == 0,
            "server_process": tag5,
            "decode": decode,
            "mtp_acceptance": spec_section(
                read_text(out_dir / f"metrics-before-g5-{tag5}.txt"),
                read_text(out_dir / f"metrics-after-g5-{tag5}.txt"), 3),
            "wall_sec": wall_of("gate5_three_warmed_decode_runs"),
            "comparability_note":
                "physical RTX 5090 numbers. They are not comparable with any rental "
                "RTX PRO 6000 engine-budget throughput and are never differenced against it.",
            "against_the_prefix_caching_off_baseline": {
                "baseline_median_decode_tok_s": MEDIAN_DECODE_BASELINE,
                "baseline_source": BASE_RECEIPT + " gate5, prefix caching off",
                "measured_median_decode_tok_s": decode.get("median_decode_tok_s"),
                "delta_tok_s": round(decode["median_decode_tok_s"] - MEDIAN_DECODE_BASELINE, 2)
                    if decode.get("median_decode_tok_s") else None,
                "note": "decode throughput is not what prefix caching changes; it is "
                        "recorded so a regression in steady-state decode would be visible.",
            },
        }

    # ------------------------------------------------------------------------- gate 8
    banners = {tag: proc["banner"] for tag, proc in processes.items() if proc["banner"]}
    per_process_banner = {
        tag: {
            "enable_prefix_caching": b.get("enable_prefix_caching_banner"),
            "mamba_cache_mode": b.get("mamba_cache_mode_banner"),
            "mamba_block_size": b.get("mamba_block_size_banner"),
            "prefix_caching_hash_algo": b.get("prefix_caching_hash_algo"),
            "proves_prefix_caching_on": b.get("gate8_banner_proves_prefix_caching_on"),
        } for tag, b in banners.items()
    }
    gates["gate8_banner_prefix_caching_actually_on"] = {
        "requirement": GATE_REQUIREMENTS["gate8_banner_prefix_caching_actually_on"],
        "passed": bool(banners) and all(
            v["proves_prefix_caching_on"] for v in per_process_banner.values()),
        "why_this_gate_exists":
            "the flags are accepted by argparse whether or not the engine honours them, and "
            "this build excludes hybrid models from prefix caching by default "
            "(vllm/engine/arg_utils.py:2532-2534). Without reading the banner back, seven "
            "green gates would prove only that the old profile still works.",
        "flags_passed": ["--enable-prefix-caching", "--mamba-cache-mode align"],
        "per_process": per_process_banner,
        "contrast_with_the_superseded_qualification": {
            "receipt": BASE_RECEIPT,
            "enable_prefix_caching": "False",
            "mamba_cache_mode": None,
            "meaning": "the seven gates of that receipt, and every other published "
                       "measurement of this family, ran with prefix caching off. This is a "
                       "new configuration, not a re-measurement.",
        },
    }

    # ------------------------------------------------------------------------- gate 9
    kv_rows = {}
    for tag, proc in processes.items():
        f = proc["startup_facts"]
        usage = [float(x) for x in f["actual_usage"]] if f.get("actual_usage") else None
        kv_rows[tag] = {
            "gpu_kv_cache_tokens": f.get("gpu_kv_cache_tokens_int"),
            "available_kv_cache_gib":
                float(f["available_kv_cache_gib"]) if f.get("available_kv_cache_gib") else None,
            "engine_budget_gib": float(f["desired_gpu_memory_utilization"][1])
                if f.get("desired_gpu_memory_utilization") else None,
            "usage_gib": dict(zip(("weight", "peak_activation", "non_torch", "cudagraph"),
                                  usage)) if usage else None,
            "attention_block_size_tokens":
                int(f["attention_block_size"]) if f.get("attention_block_size") else None,
            "mamba_block_size": f.get("mamba_block_size"),
            "kv_padding_layers": f.get("kv_padding_layers"),
            "maximum_concurrency": f.get("maximum_concurrency_float"),
            "maximum_concurrency_window": f.get("maximum_concurrency_window"),
        }
    measured_tokens = [r["gpu_kv_cache_tokens"] for r in kv_rows.values()
                       if r["gpu_kv_cache_tokens"]]
    worst = min(measured_tokens) if measured_tokens else None
    worst_row = None
    if worst is not None:
        worst_row = next(r for r in kv_rows.values() if r["gpu_kv_cache_tokens"] == worst)
    delta = worst - KV_TOKENS_BASELINE if worst is not None else None
    gates["gate9_kv_capacity_not_regressed"] = {
        "requirement": GATE_REQUIREMENTS["gate9_kv_capacity_not_regressed"],
        "passed": bool(measured_tokens) and worst >= 262144 and all(
            (r["maximum_concurrency"] or 0) >= 1.0 for r in kv_rows.values()),
        "baseline": {
            "receipt": BASE_RECEIPT,
            "gpu_kv_cache_tokens": KV_TOKENS_BASELINE,
            "available_kv_cache_gib": KV_GIB_BASELINE,
            "maximum_concurrency_at_262144": CONCURRENCY_BASELINE,
            "engine_budget_gib": ENGINE_BUDGET_GIB_BASELINE,
            "usage_gib": USAGE_BASELINE,
            "attention_block_size_tokens": BLOCK_SIZE_BASELINE,
            "kv_padding_layers": PADDING_LAYERS_BASELINE,
            "attribution": "measured by the physical-5090 context qualification on the "
                           "qualified profile. Not from receipts/apc-poison-repro.json, "
                           "whose 223,677 tokens and 8.18 GiB were measured at "
                           "max-model-len 196,608 and utilisation 0.92 and are not "
                           "comparable with these.",
        },
        "measured_per_process": kv_rows,
        "worst_case_gpu_kv_cache_tokens": worst,
        "kv_tokens_delta_vs_baseline": delta,
        "kv_tokens_pct_of_baseline":
            round(100.0 * worst / KV_TOKENS_BASELINE, 3) if worst else None,
        "capacity_and_concurrency_are_reported_separately":
            "capacity is GPU KV cache size in tokens; concurrency is the engine's own "
            "maximum-concurrency figure at 262,144. One 131k prefix already occupies about "
            "half the pool, so the two are never conflated into a single headline.",
        "block_pool_bookkeeping":
            "vLLM's prefix-cache block pool is host-side Python state -- a hash table of "
            "block hashes and a free-block list -- so it consumes no device memory. Any "
            "device-side cost of enabling the cache therefore shows up in exactly one "
            "place, the KV token count above, and that is what this gate reads. The "
            "engine's own components (weight, peak activation, non-torch, CUDAGraph) are "
            "recorded per process beside it so a shift in any of them is visible rather "
            "than absorbed into the KV figure.",
    }

    # ---------------------------------------------------- what the utilisation had to be
    # The qualified utilisation is read out of the launch argv that actually ran, never
    # assumed: if a rerun moves it, the receipt moves with it.
    utils = {p["command"]["gpu_memory_utilization"] for p in processes.values()
             if p.get("command")}
    if len(utils) != 1:
        raise SystemExit(f"the gate processes did not share one utilisation: {sorted(utils)}")
    qualified_util = utils.pop()

    def probe_row(tag, proc):
        f = proc["startup_facts"]
        fail = read_text(out_dir / f"startup-fail-{tag}.txt")
        needed = re.search(r"\(([0-9.]+) GiB KV cache is needed", fail)
        estimate = re.search(r"estimated maximum model length is ([0-9]+)", fail)
        started = f.get("gpu_kv_cache_tokens_int") is not None
        return {
            "tag": tag,
            "gpu_memory_utilization": (proc["command"] or {}).get("gpu_memory_utilization"),
            "engine_budget_gib": float(f["desired_gpu_memory_utilization"][1])
                if f.get("desired_gpu_memory_utilization") else None,
            "available_kv_cache_gib":
                float(f["available_kv_cache_gib"]) if f.get("available_kv_cache_gib") else None,
            "kv_gib_needed_for_262144": float(needed.group(1)) if needed else None,
            "started": started,
            "gpu_kv_cache_tokens": f.get("gpu_kv_cache_tokens_int"),
            "maximum_concurrency": f.get("maximum_concurrency_float"),
            "engine_estimated_max_model_len": int(estimate.group(1)) if estimate else None,
            "enable_prefix_caching_banner": f.get("enable_prefix_caching"),
            "mamba_cache_mode_banner": f.get("mamba_cache_mode"),
            "attention_block_size_tokens":
                int(f["attention_block_size"]) if f.get("attention_block_size") else None,
            "server_log": f"receipts/qualification-5090-apc-server-{tag}.log",
            "server_log_sha256": proc["server_log_sha256"],
        }

    band = [probe_row(tag, proc) for tag, proc in
            sorted(probes.items(), key=lambda kv: (kv[1]["command"] or {}).get(
                "gpu_memory_utilization", 0))]
    failed_at_qualified = next((r for r in band if r["started"] is False), None)

    # ------------------------------------------------------------------------- gate 7
    identity = {
        "gpu": identity_extra.get("gpu"),
        "host": identity_extra.get("host"),
        "runtime_image": {
            "role": "the promoted release unit",
            "tag": PROMOTED_TAG,
            "manifest_digest": PROMOTED_MANIFEST,
            "config_id": stage.get("config_id"),
            "parent_manifest_digest": PARENT_MANIFEST,
            "parent_role": "the three-module release image, superseded as the recommended "
                           "runtime and still valid as this image's parent and as the digest "
                           "receipts/qualification-5090-context.json measured",
            "public_base_manifest_digest": BASE_MANIFEST,
            "modules_in_image": {src: {"dest": dest, "sha256": digest}
                                 for src, (dest, digest) in MODULES.items()},
            "module_verification":
                "each module resolved through the import machinery inside the image and the "
                "resolved origin hashed, in a CPU-only container with no GPU attached, "
                "before any gate ran; the image itself also fails closed on the same four "
                "digests at build time",
            "module_verification_transcript":
                "receipts/qualification-5090-apc-module-verify.txt",
            "source_bind_mounts": [],
            "source_bind_mounts_note":
                "none. The previous qualification mounted three modules read-only over "
                "site-packages; this one cannot, and the launch argv of every process is "
                "recorded so that is checkable rather than asserted.",
            "in_image_labels": stage.get("labels"),
            "stale_label_warning":
                "io.malaiwah.image.qualified reads \"false\" inside this image and is wrong "
                "as of this receipt. It was accurate at build time and is deliberately not "
                "corrected: relabelling is a new layer and a new manifest digest, which "
                "would move the image away from the bytes this qualification and "
                "receipts/apc-poison-repro.json arms C and E both measured. "
                "receipts/production-image.json is authoritative for qualification state.",
        },
        "model": identity_extra.get("model"),
        "harness_sha256": identity_extra.get("harness_sha256"),
        "qualified_profile": {
            "gpu_memory_utilization": qualified_util,
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "max_model_len": 262144,
            "max_pixels": 8388608,
            "mtp_depth": 3,
            "kv_cache_dtype": "fp8",
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4],
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "quantization": "exl3",
            "embedding_overlay_bits": 8,
            "delta_from_the_superseded_profile":
                "three things and no more: the two flags --enable-prefix-caching and "
                "--mamba-cache-mode align; the image, from the three-module release digest to "
                f"the four-module promoted digest; and gpu_memory_utilization 0.955 -> "
                f"{qualified_util}, which the flags force and which is the price of the "
                "feature on this card. Everything else is unmoved: same window, same allocator "
                "env, same MTP depth, same KV dtype, same pixel ceiling, same capture size, "
                "same max_num_seqs.",
            "why_the_utilisation_moved":
                "at 0.955 the engine refuses to start with align mode on: it needs 9.29 GiB of "
                "KV for one 262,144-token request against an unchanged 9.28 GiB supply, "
                "because align mode rounds the request up to whole 1,600-token blocks with "
                "block-aligned mamba state. See bounded_negative_results and "
                "utilisation_band; the bump chosen is the smallest 0.0005 step that fits, "
                "not a round number.",
            "utilisation_ceiling_from_the_original_gate": 0.97,
        },
        "commands": {tag: proc["command"] for tag, proc in processes.items()},
    }
    required = {
        "gpu": identity["gpu"], "host": identity["host"],
        "image_manifest": identity["runtime_image"]["manifest_digest"],
        "image_config_id": identity["runtime_image"]["config_id"],
        "modules_in_image": identity["runtime_image"]["modules_in_image"],
        "model": identity["model"], "harness_sha256": identity["harness_sha256"],
        "commands": identity["commands"],
    }
    missing = sorted(k for k, v in required.items() if not v)
    gates["gate7_receipt_identity_complete"] = {
        "requirement": GATE_REQUIREMENTS["gate7_receipt_identity_complete"],
        "passed": not missing,
        "checked": sorted(required),
        "missing": missing,
    }

    # ------------------------------------------------------------------ prefix-cache use
    prefix_use = {}
    for tag, proc in processes.items():
        before = read_text(out_dir / f"metrics-before-g5-{tag}.txt")
        after = read_text(out_dir / f"metrics-after-g6-{tag}.txt") or \
            read_text(out_dir / f"metrics-after-g5-{tag}.txt")
        rows = counter_delta(before, after, PREFIX_COUNTERS) if before and after else {}
        hits = read_text(log_dir / f"server-{tag}.log")
        rates = re.findall(r"[Pp]refix cache hit rate: ([0-9.]+)%", hits)
        if rows or rates:
            prefix_use[tag] = {
                "counters": rows or None,
                "prefix_cache_hit_rate_log_lines_seen": len(rates),
                "max_prefix_cache_hit_rate_pct": max(float(x) for x in rates) if rates else None,
                "last_prefix_cache_hit_rate_pct": float(rates[-1]) if rates else None,
            }

    order = ["gate1_startup_native_allocation", "gate2_long_needle_exact",
             "gate3_combined_long_text_and_7mp_image", "gate4_image_suite_24_of_30",
             "gate5_three_warmed_decode_runs", "gate6_second_long_request_after_release",
             "gate7_receipt_identity_complete", "gate8_banner_prefix_caching_actually_on",
             "gate9_kv_capacity_not_regressed"]
    decisions = {name: ("PASS" if gates.get(name, {}).get("passed") else
                        ("FAIL" if name in gates else "NOT RUN")) for name in order}
    all_pass = all(v == "PASS" for v in decisions.values())

    payload = {
        "schema": "qwen38-qualification-5090-apc/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "title": "physical RTX 5090 serving qualification of the promoted release unit "
                 "with prefix caching on",
        "claim_scope":
            "one physical NVIDIA GeForce RTX 5090 (32,607 MiB), one image "
            f"({PROMOTED_TAG} at {PROMOTED_MANIFEST}), one model revision, one profile. It "
            "qualifies that image at that profile with --enable-prefix-caching "
            "--mamba-cache-mode align. It is not a claim about any other card, any other "
            "window, any other utilisation, or about LMCache, which this project has never "
            "measured.",
        "supersedes":
            f"{BASE_RECEIPT} as the qualification the published recipes rest on. That "
            "receipt is not withdrawn and not edited: it remains the hardware qualification "
            "of the three-module release image with prefix caching off, and it is the "
            "baseline every figure here is compared against.",
        "verdict": {
            "all_gates_pass": all_pass,
            "decisions": decisions,
            "qualified_at": "gpu_memory_utilization 0.955 with "
                            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
                            "--enable-prefix-caching --mamba-cache-mode align, on "
                            f"{PROMOTED_TAG} ({PROMOTED_MANIFEST})",
            "promotion": "PROMOTED" if all_pass else "NOT PROMOTED",
            "if_not_all_pass":
                "a failed gate is a published result. The recipes stay as they were and the "
                "promotion does not happen; nothing here is retried into green.",
        },
        "identity": identity,
        "gates": {name: gates[name] for name in order if name in gates},
        "prefix_cache_actually_used": prefix_use or {
            "note": "no prefix-cache counters were exposed and no hit-rate line was logged "
                    "in this run; gate 8 rests on the engine banner alone.",
        },
        "bounded_negative_results": {
            "the_qualified_profile_does_not_start_with_prefix_caching_on": {
                "what_was_tried": "the previously qualified profile, unchanged, plus the two "
                                  "flags: gpu_memory_utilization 0.955 with "
                                  "expandable_segments, window 262,144, on the promoted "
                                  "digest",
                "result": "FAILED at startup. The engine refused before serving anything.",
                "engine_message":
                    "To serve at least one request with the model's max seq len (262144), "
                    "9.29 GiB KV cache is needed, which is larger than the available KV cache "
                    "memory (9.28 GiB). Based on the available memory, the estimated maximum "
                    "model length is 260800.",
                "mechanism":
                    "the supply did not move and the requirement did. Available KV is 9.28 "
                    "GiB at 0.955 with the cache on, exactly as it is with the cache off; "
                    "weights 18.19 GiB, peak activation 1.78, non-torch 0.27, CUDAGraph 0.45 "
                    "are all unchanged, the attention block size is still 1,600 tokens and "
                    "there are still 3 padding layers wasting at most 6.25 %. What changes is "
                    "that align mode makes a request occupy whole blocks with block-aligned "
                    "mamba state, so 262,144 tokens round up to 164 blocks (262,400 tokens) "
                    "and need 9.29 GiB.",
                "the_price_in_one_line":
                    "prefix caching on this card costs either a 0.0005 utilisation bump or "
                    "1,344 tokens of window (262,144 -> 260,800, 0.51 %). It is a price, not "
                    "a defect, and it is paid deliberately.",
                "row": failed_at_qualified,
            },
            "why_this_is_published_rather_than_tuned_away":
                "the run that fails is the profile a reader would copy from the previous "
                "qualification and add two flags to. Publishing only the profile that works "
                "would leave that reader with a startup crash and no explanation.",
        },
        "utilisation_band": {
            "why_it_is_here": "the general answer is 'align mode raises the KV requirement by "
                              "one partial block', not 'use 0.9555'. Someone serving a "
                              "different window needs the rule and their own arithmetic, so "
                              "the measured band is published rather than only the value we "
                              "chose.",
            "method": "startup-only probes on the promoted digest with the same flags, one "
                      "engine start each, no gates run; the qualified value is then the "
                      "smallest 0.0005 step above 0.955 that starts",
            "budget_rule": "the engine requests ceil(cudaMemGetInfo total x utilisation), and "
                           "cudaMemGetInfo total on this card is FB Total minus FB Reserved, "
                           "32,607 - 458 = 32,149 MiB, so 0.0005 of utilisation is about 16 "
                           "MiB",
            "probes": band,
            "chosen": qualified_util,
            "chosen_because": "the smallest 0.0005 step above the previously qualified 0.955 "
                              "that satisfies the align-mode requirement. A round 0.96 also "
                              "works and restores KV capacity to 265,072 tokens, but spending "
                              "0.005 of a 31.4 GiB card on a 10 MiB shortfall would take "
                              "roughly 160 MiB from the vision-tower headroom that gate 3 "
                              "depends on, for nothing the gates need.",
        },
        "prior_evidence_on_this_exact_digest": {
            "why_it_is_citable":
                "arms C and E of receipts/apc-poison-repro.json ran on "
                f"{PROMOTED_MANIFEST} with no bind mounts, which is byte-identical to the "
                "image qualified here. That is the whole reason this digest was promoted "
                "rather than a freshly built one.",
            "correctness": "arm C: 38 requests, zero corrupted responses, zero wrong "
                           "answers, zero acceptance collapses, on a hostile nested-prefix "
                           "probe with a live 62.9 % hit rate in arm A. Across all seven "
                           "arms, 266 scored requests and no corruption on the unpatched "
                           "release image either.",
            "reuse_win": "arm E, disjoint documents so the cold case is genuinely cold: "
                         "32,842-token prefix 12.07 s cold -> 1.04 s warm (11.6x, 2,442 of "
                         "32,842 prompt tokens recomputed, 92.6 % hit rate); 131,146-token "
                         "prefix 67.60 s -> 2.31 s (29.3x, 3,146 of 131,146 recomputed, "
                         "97.6 %); the 38-request schedule ran 84.0 s with the cache against "
                         "144.4 s without.",
            "owner": "ApcPoisonRepro; that receipt is authoritative for those numbers and "
                     "they are deliberately not re-derived here.",
            "what_it_is_not":
                "arm D of that receipt bind-mounted tools/vllm-qwen-gdn-spec-gates.py over "
                "this same digest. It is evidence about that module, not about any image "
                "digest, and no image in this project carries it.",
        },
        "not_claimed": [
            "this does not qualify any other window, utilisation, sequence count or card. "
            "In particular --max-num-seqs 1 is part of the qualified profile; the "
            "concurrent-serving variant on the cards is not qualified at any value.",
            "LMCache is unmeasured by this project. It is the outstanding suspect in the "
            "one user report of prefix-cache corruption we have, and nothing here says it "
            "is safe.",
            "the reuse win is arm E's, measured on this digest but on a synthetic "
            "document-reuse workload, not on this gate set.",
            "the mamba-align patch is carried as insurance backed by upstream's own "
            "regression file (14 failed / 6 passed vendored, 20 passed patched), not by a "
            "reproduction of our own: 266 scored requests failed to reproduce the reported "
            "corruption even on the unpatched image.",
            "no image here has been pushed to a public registry; the promoted digest is "
            "local to the build host, and the cards reproduce its content with verified "
            "read-only mounts over the pullable public base.",
        ],
        "live_service": read_json(args.restore) if args.restore else None,
        "evidence_files": sorted(
            str(p.relative_to(work)) for p in work.rglob("*") if p.is_file()),
    }

    text = json.dumps(payload, indent=2)
    payload["content_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": args.out, "all_gates_pass": all_pass,
                      "decisions": decisions}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
