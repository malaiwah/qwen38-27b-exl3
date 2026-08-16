#!/usr/bin/env python3
"""Reachability and effect of upstream vLLM PR #51812 (Qwen GDN speculative gate
alignment) under concurrent serving on the pinned Qwen3.8-27B runtime.

Why this tool exists at all
---------------------------
Upstream measured PR #51812's own effect on Qwen3.5-2B as a mean absolute
chosen-logprob error of 0.002755 falling to 0.000208, with *identical* greedy
token ids before and after. Our own run-to-run floor for the same instrument on
this build is 0.0823 (receipts/apc-poison-repro.json, arm Bn: two identically
flagged fresh servers). The expected effect is therefore about 30x below our
resolution, so an A/B/null logprob comparison is pre-determined to return "below
resolution" no matter what the truth is. It cannot answer the question.

What can answer it is a direct count of whether the defective code path is ever
entered. The vendored qwen_gdn_linear_attn.py gathers Q/K/V with
``spec_token_indx`` (vendored line 1298) but hands the recurrent update the
ungathered ``a``/``b`` (vendored lines 1423-1424). Those two agree exactly when
``spec_token_indx == arange(T_spec)``, i.e. when every speculative token
precedes every non-speculative token in batch order. So the module can only
change an answer on a forward pass where that is false.

``tools/vllm-gdn-reach-instrument.py`` is the image's own
``vllm/v1/attention/backends/gdn_attn.py`` plus a purely additive, host-side
counter for exactly that predicate. This file drives the load and aggregates
the counter.

Modes
-----
  adversarial   sustained mixed traffic designed to make the predicate false:
                eight long speculative-decode streams held in flight while
                short prompts and full-prefix-cache-hit repeats are injected.
                The repeats are the interesting ones: a request admitted needing
                exactly one new token is padded to 1+num_spec draft slots by
                the scheduler (scheduler.py:891-904) and is then *classified as a
                speculative decode* by the runner (gpu_model_runner.py:2298),
                while the batch reorder leaves it behind the real decodes
                because it is not done prefilling (utils.py:771-780).
  reach         aggregate the instrument's per-process dumps into one summary.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the qualified harness rather than growing a second convention for the
# same measurements: identical HTTP shape, identical greedy settings, identical
# metric scraping and identical divergence maths.
from apc_poison_probe import (  # noqa: E402
    completion,
    metric_delta,
    scrape_metrics,
)


# --------------------------------------------------------------------------------------
# adversarial traffic
# --------------------------------------------------------------------------------------
# Short-prompt lengths in tokens. 1 + num_spec is 4 on this recipe; the scheduler's
# padding path keys off num_new_tokens == 1, and the reorder threshold is 4, so the
# neighbourhood of 4 is where a non-speculative request can share region 0/1 with
# speculative decodes.
SHORT_LENS = (1, 2, 3, 4, 5, 6, 8, 12)
# Number of times each cache-hit repeat is re-sent. The first send populates the
# prefix cache; every later send is admitted needing exactly one new token, which is
# the padding trigger.
REPEAT_SENDS = 4


def _greedy_payload(model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "logprobs": 1,
    }


def _post(base: str, model: str, prompt_ids: list[int], max_tokens: int) -> dict:
    t0 = time.perf_counter()
    try:
        response = completion(base, _greedy_payload(model, prompt_ids, max_tokens))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return {"error": repr(exc), "wall_s": round(time.perf_counter() - t0, 4)}
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {
        "wall_s": round(time.perf_counter() - t0, 4),
        "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "response_tokens": logprobs.get("tokens"),
        "chosen_token_logprobs": [
            round(v, 6) if v is not None else None
            for v in (logprobs.get("token_logprobs") or [])
        ],
        "response_text": choice["text"],
    }


def adversarial(args: argparse.Namespace) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    if (
        args.expect_prompts_sha256
        and prompts["prompts_sha256"] != args.expect_prompts_sha256
    ):
        raise SystemExit("prompt file digest mismatch")
    specs = prompts["correctness_requests"]
    # The long streams: the longest frozen prompts, so they chunk-prefill for many
    # steps and then decode for many more, guaranteeing a long window in which the
    # engine is in steady speculative decode with room for injections.
    long_specs = sorted(specs, key=lambda s: -s["prompt_tokens"])[: args.streams]
    # Short prompts are prefixes of a frozen prompt, so they are still frozen token
    # ids from the same tokenizer and need no tokenizer at load time.
    donor = specs[0]["prompt_ids"]

    rng = random.Random(args.seed)
    rows: list[dict] = []
    lock = threading.Lock()
    stop = threading.Event()

    def record(kind: str, tag: str, result: dict) -> None:
        with lock:
            rows.append({"kind": kind, "tag": tag, **result})

    def long_stream(index: int) -> None:
        spec = long_specs[index % len(long_specs)]
        rounds = 0
        while not stop.is_set():
            result = _post(args.base, args.model, spec["prompt_ids"], args.max_tokens)
            record("long_stream", f"{spec['request_id']}#r{rounds}", result)
            rounds += 1
            if result.get("error"):
                time.sleep(1.0)

    def injector(index: int) -> None:
        # Wait until the long streams are past their prefills, so that injections
        # land while the engine is in steady speculative decode. That is the
        # composition the padding path needs (scheduler.py:895 requires running
        # requests scheduled and no prefill chunk scheduled in the same step).
        stop.wait(args.warmup_s)
        counter = 0
        while not stop.is_set():
            counter += 1
            mode = counter % 3
            if mode == 0:
                # bare short prompt: exercises region 3 -> region 0 transitions
                length = rng.choice(SHORT_LENS)
                ids = donor[:length]
                record(
                    "short_prompt",
                    f"len{length}#i{index}c{counter}",
                    _post(args.base, args.model, ids, args.inject_max_tokens),
                )
            elif mode == 1:
                # full-prefix-cache-hit repeat: send the same ids several times. From
                # the second send on, admission needs exactly one new token, which is
                # the scheduler's spec-padding trigger.
                length = rng.choice((37, 61, 113, 211))
                ids = donor[:length]
                for send in range(REPEAT_SENDS):
                    if stop.is_set():
                        break
                    record(
                        "cache_hit_repeat",
                        f"len{length}#i{index}c{counter}s{send}",
                        _post(args.base, args.model, ids, args.inject_max_tokens),
                    )
            else:
                # a prompt whose chunked prefill ends in a tiny final chunk: a
                # non-speculative short_extend, which shares region 1 with a padded
                # pseudo-speculative request.
                chunk = args.chunk_tokens
                length = chunk + rng.choice((1, 2, 3, 4))
                ids = donor[:length]
                record(
                    "tiny_final_chunk",
                    f"len{length}#i{index}c{counter}",
                    _post(args.base, args.model, ids, args.inject_max_tokens),
                )

    before = scrape_metrics(args.base)
    started = time.time()
    workers = args.streams + args.injectors
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(long_stream, i) for i in range(args.streams)]
        futures += [pool.submit(injector, i) for i in range(args.injectors)]
        time.sleep(args.duration_s)
        stop.set()
        for future in futures:
            future.result()
    after = scrape_metrics(args.base)

    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    errors = [r for r in rows if r.get("error")]
    out = {
        "schema": "qwen38-gdn-gate-adversarial/1",
        "arm": args.arm,
        "description": args.description,
        "base_url": args.base,
        "prompts_sha256": prompts["prompts_sha256"],
        "load": {
            "streams": args.streams,
            "injectors": args.injectors,
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "max_tokens_long": args.max_tokens,
            "max_tokens_injected": args.inject_max_tokens,
            "short_prompt_lengths": list(SHORT_LENS),
            "repeat_sends": REPEAT_SENDS,
            "chunk_tokens": args.chunk_tokens,
            "long_stream_prompt_tokens": [s["prompt_tokens"] for s in long_specs],
        },
        "summary": {
            "requests": len(rows),
            "by_kind": kinds,
            "errors": len(errors),
            "first_error": errors[0]["error"] if errors else None,
            "wall_s": round(time.time() - started, 2),
            "metric_delta": {
                k: round(v, 4) for k, v in metric_delta(before, after).items()
            },
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": args.out, **out["summary"]}, default=str))
    return 0


# --------------------------------------------------------------------------------------
# reach: aggregate the instrument's dumps
# --------------------------------------------------------------------------------------
SUM_KEYS = (
    "builds_total",
    "no_spec_branch",
    "pure_spec_branch",
    "mixed_branch",
    "mixed_identity",
    "mixed_misordered",
)


def reach(args: argparse.Namespace) -> int:
    files = sorted(Path(args.dir).glob(f"{args.prefix}*.json"))
    if not files:
        raise SystemExit(f"no instrument dumps matching {args.prefix}*.json in {args.dir}")
    dumps = [json.loads(p.read_text()) for p in files]
    total = {k: sum(d.get(k, 0) for d in dumps) for k in SUM_KEYS}
    compositions: dict[str, int] = {}
    misordered: dict[str, int] = {}
    for d in dumps:
        for key, count in (d.get("compositions") or {}).items():
            compositions[key] = compositions.get(key, 0) + count
        for key, count in (d.get("misordered_compositions") or {}).items():
            misordered[key] = misordered.get(key, 0) + count
    firsts = [
        d["first_misordered_token_index_min"]
        for d in dumps
        if d.get("first_misordered_token_index_min") is not None
    ]
    out = {
        "schema": "qwen38-gdn-gate-reach/1",
        "arm": args.arm,
        "description": args.description,
        "instrument_sha256": args.instrument_sha256,
        "dump_files": [p.name for p in files],
        "worker_pids": [d.get("pid") for d in dumps],
        "counts": total,
        "defective_path_entered": total["mixed_misordered"] > 0,
        # The only forward passes on which unpatched and patched can differ.
        "fraction_of_builds_miscomputed": (
            round(total["mixed_misordered"] / total["builds_total"], 8)
            if total["builds_total"]
            else None
        ),
        "fraction_of_mixed_branch_miscomputed": (
            round(total["mixed_misordered"] / total["mixed_branch"], 8)
            if total["mixed_branch"]
            else None
        ),
        "max_displaced_spec_tokens": max(
            (d.get("max_displaced_spec_tokens", 0) for d in dumps), default=0
        ),
        "first_misordered_token_index_min": min(firsts) if firsts else None,
        "compositions_seen": dict(sorted(compositions.items())),
        "misordered_compositions": dict(sorted(misordered.items())),
        "composition_key": "<num_spec_decodes>s/<num_prefills>p/<num_decodes>d",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": args.out,
                "builds_total": total["builds_total"],
                "mixed_branch": total["mixed_branch"],
                "mixed_misordered": total["mixed_misordered"],
                "defective_path_entered": out["defective_path_entered"],
            }
        )
    )
    return 0


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    a = sub.add_parser("adversarial", help="sustained mixed traffic")
    a.add_argument("--base", required=True)
    a.add_argument("--model", default="m")
    a.add_argument("--prompts", required=True)
    a.add_argument("--expect-prompts-sha256")
    a.add_argument("--arm", required=True)
    a.add_argument("--description", default="")
    a.add_argument("--streams", type=int, default=8)
    a.add_argument("--injectors", type=int, default=3)
    a.add_argument("--duration-s", type=float, default=300.0)
    a.add_argument("--warmup-s", type=float, default=45.0)
    a.add_argument("--max-tokens", type=int, default=192)
    a.add_argument("--inject-max-tokens", type=int, default=24)
    a.add_argument("--chunk-tokens", type=int, default=2048)
    a.add_argument("--seed", type=int, default=20260816)
    a.add_argument("--out", required=True)
    a.set_defaults(func=adversarial)

    r = sub.add_parser("reach", help="aggregate instrument dumps")
    r.add_argument("--dir", required=True)
    r.add_argument("--prefix", default="reach")
    r.add_argument("--arm", required=True)
    r.add_argument("--description", default="")
    r.add_argument("--instrument-sha256", default=None)
    r.add_argument("--out", required=True)
    r.set_defaults(func=reach)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
