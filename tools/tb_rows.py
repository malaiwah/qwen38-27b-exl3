#!/usr/bin/env python3
"""Per-task row extraction for the Terminal-Bench 2.1 three-pass protocol.

Reads a collected harbor job directory (one sub-directory per trial, each with
`result.json`) and emits per-task rows, an aggregate summary, or the failure
list that feeds the next pass.

Field names below are taken verbatim from a real harbor 0.21.0 `result.json`
(inspected on the oracle smoke trial); nothing here is guessed.

Usage:
    tb_rows.py rows     <job-dir>            # JSONL, one row per trial
    tb_rows.py summary  <job-dir> [label]    # aggregate JSON
    tb_rows.py failures <job-dir>            # task names with best reward < 1
    tb_rows.py gpu-assert <job-dir>          # exit 1 if any trial requested a GPU
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys


def _iso_delta(phase: dict | None) -> float | None:
    """Seconds between a phase's started_at/finished_at ISO-8601 stamps."""
    if not phase:
        return None
    a, b = phase.get("started_at"), phase.get("finished_at")
    if not a or not b:
        return None
    import datetime as _dt

    try:
        pa = _dt.datetime.fromisoformat(a.replace("Z", "+00:00"))
        pb = _dt.datetime.fromisoformat(b.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((pb - pa).total_seconds(), 3)


def _model_name(agent_info: dict) -> str | None:
    mi = agent_info.get("model_info")
    if isinstance(mi, dict):
        return mi.get("name")
    if isinstance(mi, str):
        return mi
    return None


def load_rows(job_dir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        rf = trial_dir / "result.json"
        if not rf.exists():
            # No result.json => the trial never completed. harbor deletes such
            # dirs on resume and re-runs the task; record it as incomplete so
            # the row set never silently loses a task.
            rows.append(
                {
                    "task": trial_dir.name.rsplit("__", 1)[0],
                    "trial_name": trial_dir.name,
                    "status": "incomplete",
                    "reward": None,
                    "resolved": False,
                    "exception_type": None,
                }
            )
            continue
        try:
            d = json.loads(rf.read_text())
        except (OSError, ValueError) as e:
            rows.append(
                {
                    "task": trial_dir.name.rsplit("__", 1)[0],
                    "trial_name": trial_dir.name,
                    "status": f"unreadable: {type(e).__name__}",
                    "reward": None,
                    "resolved": False,
                    "exception_type": None,
                }
            )
            continue

        agent_info = d.get("agent_info") or {}
        agent_result = d.get("agent_result") or {}
        verifier_result = d.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        reward = rewards.get("reward")
        exc = d.get("exception_info") or {}
        cfg = d.get("config") or {}
        env = cfg.get("environment") or {}

        task_name = d.get("task_name") or trial_dir.name.rsplit("__", 1)[0]
        try:
            reward_f = float(reward) if reward is not None else None
        except (TypeError, ValueError):
            reward_f = None

        rows.append(
            {
                "task": task_name.split("/")[-1],
                "task_name": task_name,
                "trial_name": d.get("trial_name") or trial_dir.name,
                "task_checksum": d.get("task_checksum"),
                "status": "errored" if exc else "completed",
                "reward": reward_f,
                "resolved": reward_f == 1.0,
                "exception_type": exc.get("exception_type"),
                "agent": agent_info.get("name"),
                "agent_version": agent_info.get("version"),
                "model": _model_name(agent_info),
                "n_input_tokens": agent_result.get("n_input_tokens"),
                "n_cache_tokens": agent_result.get("n_cache_tokens"),
                "n_output_tokens": agent_result.get("n_output_tokens"),
                "override_gpus": env.get("override_gpus"),
                "timeout_multiplier": cfg.get("timeout_multiplier"),
                "wall_s": _iso_delta(d),
                "env_setup_s": _iso_delta(d.get("environment_setup")),
                "agent_exec_s": _iso_delta(d.get("agent_execution")),
                "verifier_s": _iso_delta(d.get("verifier")),
                "started_at": d.get("started_at"),
                "finished_at": d.get("finished_at"),
            }
        )
    return rows


def best_by_task(rows: list[dict]) -> dict[str, float]:
    best: dict[str, float] = collections.defaultdict(lambda: -1.0)
    for r in rows:
        v = r.get("reward")
        best[r["task"]] = max(best[r["task"]], v if v is not None else -1.0)
    return dict(best)


def summarize(rows: list[dict], label: str | None) -> dict:
    best = best_by_task(rows)
    resolved = [t for t, v in best.items() if v == 1.0]
    unresolved = [t for t, v in best.items() if v != 1.0]
    attempted = [t for t, v in best.items() if v >= 0.0]
    tok_in = sum(r["n_input_tokens"] or 0 for r in rows)
    tok_out = sum(r["n_output_tokens"] or 0 for r in rows)
    tok_cache = sum(r["n_cache_tokens"] or 0 for r in rows)
    agent_s = sum(r.get("agent_exec_s") or 0 for r in rows)
    excs = collections.Counter(
        r["exception_type"] for r in rows if r.get("exception_type")
    )
    models = sorted({r["model"] for r in rows if r.get("model")})
    agents = sorted({
        f"{r['agent']}@{r['agent_version']}" for r in rows if r.get("agent")
    })
    return {
        "label": label,
        "n_tasks": len(best),
        "n_trials": len(rows),
        "n_attempted": len(attempted),
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
        "accuracy_resolved_over_tasks": (
            round(len(resolved) / len(best), 6) if best else None
        ),
        "agents": agents,
        "models": models,
        "tokens": {
            "input": tok_in,
            "output": tok_out,
            "cache_read": tok_cache,
        },
        "agent_execution_seconds_total": round(agent_s, 3),
        "output_tok_per_s_over_agent_exec": (
            round(tok_out / agent_s, 3) if agent_s and tok_out else None
        ),
        "exception_types": dict(excs),
        "resolved_tasks": sorted(resolved),
        "unresolved_tasks": sorted(unresolved),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    mode, jd = sys.argv[1], pathlib.Path(sys.argv[2])
    if not jd.is_dir():
        print(f"not a job directory: {jd}", file=sys.stderr)
        return 2
    rows = load_rows(jd)

    if mode == "rows":
        for r in rows:
            print(json.dumps(r, sort_keys=True))
    elif mode == "summary":
        label = sys.argv[3] if len(sys.argv) > 3 else jd.name
        print(json.dumps(summarize(rows, label), indent=2, sort_keys=True))
    elif mode == "failures":
        for t, v in sorted(best_by_task(rows).items()):
            if v != 1.0:
                print(t)
    elif mode == "gpu-assert":
        bad = [
            r for r in rows
            if r.get("override_gpus") not in (None, 0)
        ]
        if bad:
            for r in bad:
                print(
                    f"GPU REQUESTED: {r['trial_name']} override_gpus="
                    f"{r['override_gpus']}",
                    file=sys.stderr,
                )
            return 1
        print(
            f"no-gpu-assert OK: {len(rows)} trials, every "
            "config.environment.override_gpus is null/0"
        )
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
