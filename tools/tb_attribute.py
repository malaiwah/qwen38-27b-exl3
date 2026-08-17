#!/usr/bin/env python3
"""tb_attribute - split twice-failed Terminal-Bench tasks into three buckets.

Pass 1 scores the quantisation, pass 2 re-runs its failures to separate flaky
from persistent, pass 3 re-runs the twice-failed set on BF16 weights with every
other variable held. This tool reads all three job directories and assigns each
twice-failed task to exactly one bucket:

  quantization-suspect  BF16 resolved it, so the quantisation is implicated.
  capability            BF16 ran to a verdict and still failed, so it is not
                        the quantisation.
  inconclusive-timeout  Neither arm ever produced an answer. Folding this into
                        `capability` would exonerate the quantisation on tasks
                        where no arm finished, which the evidence cannot
                        support - so it is its own bucket and is never summed
                        into either of the others.
  harness-void          Every BF16 trial failed before the model could answer
                        (see tb_rows.pre_model_void). Not attributable at all.
  owed-rerun            The task reached the twice-failed set only because its
                        pass-2 attempt was itself a pre-model void, so it never
                        had a fair second attempt on the quantisation. Reported
                        separately: its bucket is provisional until re-run.

Usage:
  tb_attribute.py <pass1-job-dir> <pass2-job-dir> <pass3-job-dir> [--out FILE]
                  [--label TEXT]

Exit status is 0 even when buckets are empty; a missing job directory is an
error, because silently attributing against absent evidence is the failure mode
this tool exists to prevent.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tb_rows  # noqa: E402


def _by_task(job: pathlib.Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in tb_rows.load_rows(job):
        out.setdefault(r["task"], []).append(r)
    return out


def _resolved(trials: list[dict]) -> bool:
    return any(t.get("reward") == 1.0 for t in trials)


def _has_verdict(trials: list[dict]) -> bool:
    return any(t.get("reward") is not None for t in trials)


def _all_void(trials: list[dict]) -> bool:
    return bool(trials) and all(t.get("pre_model_void") for t in trials)


def _evidence(trials: list[dict]) -> dict:
    return {
        "trials": len(trials),
        "rewards": [t.get("reward") for t in trials],
        "exception_types": [t.get("exception_type") for t in trials],
        "pre_model_void": [t.get("pre_model_void") for t in trials if t.get("pre_model_void")],
        "agent_exec_s": [t.get("agent_exec_s") for t in trials],
    }


def attribute(p1: pathlib.Path, p2: pathlib.Path, p3: pathlib.Path,
              label: str | None) -> dict:
    a, b, c = _by_task(p1), _by_task(p2), _by_task(p3)

    # The twice-failed set is defined by the runs, not re-derived from a list
    # file: a task is twice-failed when it failed pass 1 and failed pass 2.
    twice = sorted(t for t in b if not _resolved(b[t]) and t in a and not _resolved(a[t]))

    buckets: dict[str, list[str]] = {
        "quantization-suspect": [],
        "capability": [],
        "inconclusive-timeout": [],
        "harness-void": [],
    }
    owed: list[str] = []
    rows: dict[str, dict] = {}

    for t in twice:
        bf16 = c.get(t, [])
        if not bf16:
            bucket = "inconclusive-timeout"
            why = "no BF16 trial found for this task in the pass-3 job directory"
        elif _all_void(bf16):
            bucket = "harness-void"
            why = "every BF16 trial failed before the model could answer"
        elif _resolved(bf16):
            bucket = "quantization-suspect"
            why = "BF16 resolved the task under otherwise identical conditions"
        elif _has_verdict(bf16):
            bucket = "capability"
            why = "BF16 reached a verifier verdict and still failed"
        else:
            bucket = "inconclusive-timeout"
            why = "neither arm produced a verdict; BF16 never finished either"
        buckets[bucket].append(t)

        # Provisional if the quantisation's own second attempt was voided.
        if _all_void(b.get(t, [])):
            owed.append(t)

        rows[t] = {
            "bucket": bucket,
            "why": why,
            "provisional_owed_rerun": t in owed,
            "pass1": _evidence(a.get(t, [])),
            "pass2": _evidence(b.get(t, [])),
            "pass3_bf16": _evidence(bf16),
        }

    attributable = len(buckets["quantization-suspect"]) + len(buckets["capability"])
    return {
        "schema": "qwen38-tb21-attribution/1",
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "job_dirs": {"pass1": str(p1), "pass2": str(p2), "pass3_bf16": str(p3)},
        "counts": {
            "pass1_tasks": len(a),
            "pass2_tasks": len(b),
            "pass3_tasks": len(c),
            "twice_failed": len(twice),
            **{k: len(v) for k, v in buckets.items()},
            "attributable": attributable,
            "owed_rerun": len(owed),
        },
        "buckets": buckets,
        "owed_rerun": sorted(owed),
        "reading_rules": [
            "The three buckets partition the twice-failed set and are never summed "
            "pairwise: quantization-suspect + capability is the attributable "
            "subset, and inconclusive-timeout is neither.",
            "A quantization-suspect verdict means BF16 resolved the task under "
            "otherwise identical conditions - same agent, sampling, eager mode, "
            "endpoint and concurrency. It is not a claim about which tensor.",
            "inconclusive-timeout is not evidence for the quantisation. It is the "
            "absence of evidence, recorded rather than resolved by assumption.",
            "owed_rerun tasks carry a provisional bucket: their pass-2 attempt was "
            "a pre-model void, so the quantisation never had a fair second try.",
        ],
        "per_task": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pass1")
    ap.add_argument("pass2")
    ap.add_argument("pass3")
    ap.add_argument("--out", help="write the receipt here (default: stdout only)")
    ap.add_argument("--label")
    args = ap.parse_args()

    dirs = [pathlib.Path(p).expanduser() for p in (args.pass1, args.pass2, args.pass3)]
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"not a job directory: {d}")

    rep = attribute(*dirs, args.label)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).expanduser().write_text(text + "\n")
        print(f"[attribute] -> {args.out}")
    c = rep["counts"]
    print(f"twice-failed {c['twice_failed']}: "
          f"quantization-suspect {c['quantization-suspect']}, "
          f"capability {c['capability']}, "
          f"inconclusive-timeout {c['inconclusive-timeout']}, "
          f"harness-void {c['harness-void']}"
          + (f", owed-rerun {c['owed_rerun']}" if c["owed_rerun"] else ""))
    for name in ("quantization-suspect", "capability"):
        if rep["buckets"][name]:
            print(f"  {name}: {', '.join(rep['buckets'][name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
