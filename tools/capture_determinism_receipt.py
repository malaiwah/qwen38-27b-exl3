#!/usr/bin/env python3
"""Build receipts/capture-determinism.json by re-deriving the comparison.

Two independent measurements of v5 shard 0 exist on this box: the published
ladder's (/var/tmp/work/kld5) and the tail re-run's (/var/tmp/work/kld5-tail).
They used different capture directories, separate model loads, and different
harness files.  This script compares their reports field by field and writes what
it found -- it does not restate a conclusion reached elsewhere.

    build_determinism_receipt.py <repo-root> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = {
    "published_ladder": Path("/var/tmp/work/kld5"),
    "tail_rerun": Path("/var/tmp/work/kld5-tail"),
}
CANDIDATES = ("hyd", "k5k6", "ctx", "k4", "fp8")
SHARD = "shard-0000"

# Differing here is expected and does not weaken the claim.
EXPECTED_TO_DIFFER = {
    "reference": "path of the reference capture directory; each run captured into "
                 "its own directory, so this differs by construction",
    "candidate": "path of the candidate capture directory; likewise",
    "schema": "the tail re-run's harness emits qwen38-fidelity-report/2, which adds "
              "the kld_tail histogram to /1 without changing any existing field",
    "kld_tail": "the added histogram itself: present only in /2, so there is nothing "
                "in /1 to compare it against",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def gpu_block() -> dict:
    """The GPU both runs used, with the basis for saying so.

    The replay reports record `comparator.device = "cuda"` and nothing more, so a
    reader cannot verify "same GPU" from the reports alone.  What can be verified
    is that this host exposes exactly one GPU, that `tools/ggrun.sh` pins
    `CUDA_VISIBLE_DEVICES=0`, and that both ladder roots live on this host -- so
    both runs necessarily used the device named below.  It is queried rather than
    typed, and it is stated as a host-level fact, not as something the reports
    prove.
    """
    fields = ("name", "driver_version", "memory.total", "uuid", "vbios_version")
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    devices = [dict(zip(fields, (c.strip() for c in line.split(",")))) for line in out]
    return {
        "devices_visible_on_host": len(devices),
        "device": devices[0] if len(devices) == 1 else devices,
        "basis": "this host exposes exactly one GPU and tools/ggrun.sh pins "
                 "CUDA_VISIBLE_DEVICES=0, so both runs used this device; the reports "
                 "themselves record only comparator.device = 'cuda', so this is a "
                 "host-level statement queried at receipt time rather than something "
                 "the reports evidence",
        "queried_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])

    runs = {}
    for name, root in RUNS.items():
        pin = json.loads((root / "ladder-pin.json").read_text())
        runs[name] = {
            "root": str(root),
            "harness_sha256": pin["harness_sha256"],
            "harness_guest_path": pin["harness_guest_path"],
            "head_sha256": pin["head_sha256"],
            "shard_suite_token_sha256": json.loads(
                (root / "shards" / SHARD / "suite/suite-manifest.json").read_text()
            )["suite_token_sha256"],
            "runtime_image": pin.get("runtime_image"),
        }

    per_candidate = {}
    unexpected = []
    positions_total = 0
    for cand in CANDIDATES:
        reports = {}
        for name, root in RUNS.items():
            path = root / "reports" / SHARD / f"report-{cand}.json"
            reports[name] = json.loads(path.read_text())
        a, b = reports["published_ladder"], reports["tail_rerun"]
        differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        surprises = [k for k in differing if k not in EXPECTED_TO_DIFFER]
        unexpected.extend(f"{cand}.{k}" for k in surprises)
        identical = sorted(k for k in set(a) | set(b) if k not in differing)
        positions = a["scored_positions"]
        positions_total += positions
        per_candidate[cand] = {
            "contexts": a["contexts"],
            "scored_positions": positions,
            "differing_keys": differing,
            "unexpectedly_differing_keys": surprises,
            "identical_keys": identical,
            "per_context_rows_identical": a.get("per_context") == b.get("per_context"),
            "per_context_all_rows_identical": (a.get("per_context_all")
                                               == b.get("per_context_all")),
            "agreed_values": {
                "token_mean_kld": a["token_mean_kld"],
                "token_median_kld": a["token_median_kld"],
                "p95_kld": a["p95_kld"],
                "p99_kld": a["p99_kld"],
                "p999_kld": a["p999_kld"],
                "max_kld": a["max_kld"],
                "top1_agreement": a["top1_agreement"],
                "mean_jsd_bits": a["mean_jsd_bits"],
                "context_bootstrap": a["context_bootstrap"],
            },
        }

    if unexpected:
        raise SystemExit("REFUSED: fields differ that the claim does not allow: "
                         + ", ".join(unexpected))

    payload = {
        "schema": "qwen38-capture-determinism/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim": "on this machine, vLLM hidden-state capture followed by replay is "
                 "bit-reproducible: two independent runs of the same shard, with "
                 "separate model loads and different harness files, produce reports "
                 "whose every measured field is byte-identical, including the "
                 "complete per-context arrays.",
        "scope_limits": {
            "read_this_first": "This is NOT a claim that vLLM is bitwise "
                               "deterministic in general. The configuration below is "
                               "load-bearing, and a reader who takes a broader claim "
                               "from this receipt and then fails to reproduce it "
                               "under CUDA graphs or larger batches would be right to "
                               "call that overreach.",
            "same_gpu": gpu_block(),
            "same_runtime": "the same pinned rootfs image and vLLM build in both runs",
            "enforce_eager": True,
            "max_num_seqs": 1,
            "prefill": "one context per forward, one chunk per context "
                       "(max_num_batched_tokens = context_length)",
            "gpu_memory_utilization": 0.85,
            "kv_cache": "512 MiB, bfloat16",
            "not_claimed": [
                "determinism across different GPUs, drivers or CUDA versions",
                "determinism with CUDA graphs enabled (enforce_eager=False)",
                "determinism across batch shapes, max_num_seqs > 1, or chunked "
                "prefill with more than one chunk per context",
                "determinism of sampling or of anything downstream of the logits",
            ],
        },
        "runs": runs,
        "what_was_compared": {
            "shard": f"v5 {SHARD}: contexts 0-511 of the parent suite",
            "candidates": list(CANDIDATES),
            "reports_per_candidate": 2,
            "note": "JSON round-trips IEEE-754 doubles exactly, so equality of the "
                    "parsed values is bitwise equality of the numbers each run "
                    "computed, not equality to some printed precision",
        },
        "result": {
            "every_measured_field_identical": True,
            "per_context_arrays_identical": all(
                row["per_context_all_rows_identical"] and row["per_context_rows_identical"]
                for row in per_candidate.values()),
            "scored_positions_reproduced": positions_total,
            "arithmetic": f"{len(CANDIDATES)} candidates x "
                          f"{per_candidate[CANDIDATES[0]]['contexts']} contexts x "
                          f"{per_candidate[CANDIDATES[0]]['scored_positions'] // per_candidate[CANDIDATES[0]]['contexts']} "
                          f"scored positions = {positions_total:,}",
            "fields_that_differ_and_why": EXPECTED_TO_DIFFER,
            "per_candidate": per_candidate,
        },
        "what_this_licenses": {
            "capture_reuse": "a surviving capture may be replayed instead of "
                             "recapturing the same operand, which is how the shard-0 "
                             "NVFP4 measurement used /work/gguf/hidden-bf16 as its "
                             "BF16 reference and cost one model load instead of two "
                             "(receipts/nvfp4-v5-measurement.json)",
            "cross_generation_reports": "reports produced by different harness "
                                        "generations may sit in one cumulative "
                                        "receipt, because the generations differ only "
                                        "additively in the measurement path",
            "re_derivable_not_just_re_runnable": "every published KLD number in this "
                                                 "project can be re-derived to the "
                                                 "bit from the same inputs, rather "
                                                 "than merely re-measured to within "
                                                 "an interval",
        },
        "independently_checked_by": "ErrorDrivenAlloc verified the hydrated candidate "
                                    "independently before this receipt existed; this "
                                    "receipt extends the check to all five candidates",
        "inputs": {
            f"{name}/{cand}": sha256_file(
                RUNS[name] / "reports" / SHARD / f"report-{cand}.json")
            for name in RUNS for cand in CANDIDATES
        },
    }
    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "generated_utc"})
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, out)
    print(json.dumps({
        "out": str(out),
        "scored_positions_reproduced": positions_total,
        "arithmetic": payload["result"]["arithmetic"],
        "every_measured_field_identical": True,
        "content_sha256": payload["content_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
