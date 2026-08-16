#!/usr/bin/env python3
"""Timeout-budget bin-packer for the Terminal-Bench 2.1 multi-GPU campaign.

WHY THIS EXISTS (docs/46 §5): task sharding IS the load balancer. There is no
router in front of vLLM; instead the 89 tasks are partitioned into K shards
(K = number of DP replicas), each pinned to one replica's api_base by
tb21_campaign.sh. Shards must finish together, so they are bin-packed by each
task's DECLARED agent timeout budget from the pinned inventory receipt
(receipts/terminal-bench-2.1-task-inventory.json) -- greedy makespan: sort
descending by budget, always assign to the currently lightest shard.

The declared timeout is a budget ceiling, not a runtime prediction; it is the
only per-task number that exists before a pass runs, and it is exactly the
quantity that bounds shard wall-clock at stock 1.0 timeouts.

DETERMINISM: ties in the sort are broken by task name; ties in shard load are
broken by shard index. Same inventory + same K => byte-identical output,
always. That matters because pass 1 job configs bake the shard membership in
via -i flags and harbor refuses to resume a job whose config changed.

Outputs:
  <outdir>/shard-<i>.txt      one task name per line, alphabetical
  <outdir>/manifest.json      per-shard budget sums + balance statement
                              (balance quality = max/min shard budget ratio)

Usage:
  tb21_shard.py --inventory receipts/terminal-bench-2.1-task-inventory.json \
                --shards 4 [--outdir shards/K4] [--only TASK ...]

--only (repeatable) restricts packing to a subset -- used by the mock proof
and usable for pass-2-style failure subsets; the manifest records the filter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def pack(tasks: dict[str, float], k: int) -> list[list[str]]:
    """Greedy makespan: descending budget, assign to lightest shard.

    Deterministic: sort ties by name, load ties by lowest shard index.
    """
    order = sorted(tasks, key=lambda t: (-tasks[t], t))
    loads = [0.0] * k
    shards: list[list[str]] = [[] for _ in range(k)]
    for t in order:
        i = min(range(k), key=lambda j: (loads[j], j))
        shards[i].append(t)
        loads[i] += tasks[t]
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", required=True,
                    help="terminal-bench-2.1-task-inventory.json receipt")
    ap.add_argument("--shards", type=int, required=True, metavar="K")
    ap.add_argument("--outdir", default=None,
                    help="default: shards/K<K> next to the CWD")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to this task (repeatable); unknown names are fatal")
    a = ap.parse_args()

    inv_path = pathlib.Path(a.inventory)
    raw = inv_path.read_bytes()
    inv = json.loads(raw)
    tasks_full = {
        name: float(spec["agent_timeout_sec"])
        for name, spec in inv["tasks"].items()
    }

    if a.only:
        unknown = sorted(set(a.only) - set(tasks_full))
        if unknown:
            print(f"error: --only names not in inventory: {unknown}", file=sys.stderr)
            return 2
        tasks = {t: tasks_full[t] for t in a.only}
    else:
        tasks = tasks_full

    k = a.shards
    if k < 1 or k > len(tasks):
        print(f"error: --shards must be in [1, {len(tasks)}], got {k}", file=sys.stderr)
        return 2

    shards = pack(tasks, k)
    outdir = pathlib.Path(a.outdir or f"shards/K{k}")
    outdir.mkdir(parents=True, exist_ok=True)

    per_shard = []
    for i, members in enumerate(shards):
        names = sorted(members)  # alphabetical in the file; budget drove packing
        (outdir / f"shard-{i}.txt").write_text("\n".join(names) + "\n")
        per_shard.append({
            "index": i,
            "file": f"shard-{i}.txt",
            "n_tasks": len(names),
            "budget_sum_sec": sum(tasks[t] for t in names),
            "max_single_task_sec": max(tasks[t] for t in names),
            "tasks": {t: tasks[t] for t in names},
        })

    sums = [s["budget_sum_sec"] for s in per_shard]
    ratio = max(sums) / min(sums)
    manifest = {
        "schema": "qwen38-tb21-shards/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": sys.argv,
        "inventory": str(inv_path),
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "inventory_tasks_tree_sha256": inv.get("tasks_tree_sha256"),
        "only_filter": sorted(a.only) or None,
        "algorithm": "greedy makespan: sort desc by agent_timeout_sec (ties by name), "
                     "assign to lightest shard (ties by lowest index); deterministic",
        "n_tasks": len(tasks),
        "n_shards": k,
        "total_budget_sec": sum(tasks.values()),
        "shards": per_shard,
        "balance": {
            "max_budget_sec": max(sums),
            "min_budget_sec": min(sums),
            "max_min_ratio": round(ratio, 4),
            "statement": (
                f"K={k}: heaviest shard carries {max(sums):.0f}s of declared budget, "
                f"lightest {min(sums):.0f}s; max/min ratio {ratio:.4f}. "
                "Budgets are ceilings, not predictions -- real skew is bounded by the "
                "max single task, which no packing can split."
            ),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[shard] {len(tasks)} tasks -> {k} shard(s) in {outdir}/")
    for s in per_shard:
        print(f"[shard]   shard-{s['index']}: {s['n_tasks']:3d} tasks, "
              f"{s['budget_sum_sec']:9.0f}s budget")
    print(f"[shard] balance max/min ratio: {ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
