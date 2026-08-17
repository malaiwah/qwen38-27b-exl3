#!/usr/bin/env python3
"""Client for the static-YaRN short-context penalty measurement.

One pass = one (server, probe-set) pairing.  Two phases:

  generate  greedy 8-token continuation per probe; writes the forced-continuation
            file that BOTH arms then replay.  Run once, on the reference arm.
  ladder    teacher-forced: for j in 0..7 submit prompt_ids + forced[:j] with
            max_tokens=1 and ask for the top-k logprob distribution at that
            position.  Every position in every arm is therefore conditioned on
            byte-identical context, so the only difference between arms is rope.

Token ids, not detokenized strings, key the distributions: the server's
`return_tokens_as_token_ids` extension makes every logprob key "token_id:<id>",
which removes the string-collision hazard of the plain OpenAI shape.

Usage:
  yarn_probe_run.py generate --base http://127.0.0.1:8231 --probes probes.json \
      --out gen-native.json
  yarn_probe_run.py ladder --base ... --probes probes.json --forced gen-native.json \
      --out raw-native-p1.json --arm NATIVE --pass-label p1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen38"
SEED = 314159
MAX_TOKENS = 8
TOPK_REQUEST = 20


def post(base: str, path: str, payload: dict, timeout: float = 3600.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(base: str, path: str, timeout: float = 60.0):
    req = urllib.request.Request(base.rstrip("/") + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode(errors="replace")


def parse_key(key: str) -> int:
    if not key.startswith("token_id:"):
        raise RuntimeError(f"expected token_id: keys, got {key!r}")
    return int(key.split(":", 1)[1])


def tokenize_probes(base: str, probes: list[dict]) -> list[dict]:
    """Verify every probe's token count against the live server's /tokenize."""
    out = []
    for p in probes:
        r = post(base, "/tokenize", {"model": MODEL, "prompt": p["text"], "add_special_tokens": False})
        ids = r["tokens"]
        if r["count"] != len(ids):
            raise RuntimeError(f"{p['id']}: /tokenize count {r['count']} != len {len(ids)}")
        if len(ids) != p["n_tokens_offline"]:
            raise RuntimeError(
                f"{p['id']}: /tokenize gave {len(ids)} tokens, offline builder said "
                f"{p['n_tokens_offline']}"
            )
        out.append(
            {
                "id": p["id"],
                "bucket": p["bucket"],
                "n_tokens_server_tokenize": len(ids),
                "ids_sha256": hashlib.sha256(
                    ",".join(map(str, ids)).encode()
                ).hexdigest(),
                "ids": ids,
            }
        )
    return out


def probe_completion(base: str, ids: list[int], max_tokens: int) -> dict:
    return post(
        base,
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": SEED,
            "logprobs": TOPK_REQUEST,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "stream": False,
        },
    )


def server_facts(base: str) -> dict:
    return {
        "health_status": "ok" if get(base, "/health") in ("", b"", None) else "ok",
        "models": get(base, "/v1/models"),
    }


def cmd_generate(args) -> None:
    probes = json.loads(Path(args.probes).read_text())["probes"]
    tok = tokenize_probes(args.base, probes)
    gen = []
    t0 = time.time()
    for i, t in enumerate(tok):
        r = probe_completion(args.base, t["ids"], MAX_TOKENS)
        ch = r["choices"][0]
        lp = ch["logprobs"]
        forced = [parse_key(k) for k in lp["tokens"]]
        gen.append(
            {
                "id": t["id"],
                "bucket": t["bucket"],
                "n_tokens_server_tokenize": t["n_tokens_server_tokenize"],
                "ids_sha256": t["ids_sha256"],
                "forced_continuation_ids": forced,
                "text": ch["text"],
                "finish_reason": ch["finish_reason"],
            }
        )
        print(f"[gen {i + 1}/{len(tok)}] {t['id']} -> {forced}", flush=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "schema": "yarn-forced-continuations-1",
                "seed": SEED,
                "max_tokens": MAX_TOKENS,
                "elapsed_s_proot_inflated": round(time.time() - t0, 2),
                "server": server_facts(args.base),
                "items": gen,
            },
            indent=1,
        )
    )
    print(f"wrote {args.out}")


def cmd_ladder(args) -> None:
    pdata = json.loads(Path(args.probes).read_text())
    probes = pdata["probes"]
    forced_doc = json.loads(Path(args.forced).read_text())
    forced_by_id = {it["id"]: it for it in forced_doc["items"]}

    tok = tokenize_probes(args.base, probes)
    rows = []
    short_positions = 0
    t0 = time.time()
    for i, t in enumerate(tok):
        f = forced_by_id[t["id"]]
        if f["ids_sha256"] != t["ids_sha256"]:
            raise RuntimeError(f"{t['id']}: prompt ids differ from the reference pass")
        forced = f["forced_continuation_ids"]
        positions = []
        for j in range(len(forced)):
            ctx = t["ids"] + forced[:j]
            r = probe_completion(args.base, ctx, 1)
            lp = r["choices"][0]["logprobs"]
            top = lp["top_logprobs"][0]
            dist = {parse_key(k): v for k, v in top.items()}
            if len(dist) < TOPK_REQUEST:
                short_positions += 1
            positions.append(
                {
                    "j": j,
                    "context_len": len(ctx),
                    "argmax_id": parse_key(lp["tokens"][0]),
                    "forced_next_id": forced[j],
                    "dist": {str(k): v for k, v in dist.items()},
                }
            )
        rows.append(
            {
                "id": t["id"],
                "bucket": t["bucket"],
                "n_tokens_server_tokenize": t["n_tokens_server_tokenize"],
                "ids_sha256": t["ids_sha256"],
                "forced_continuation_ids": forced,
                "positions": positions,
            }
        )
        print(f"[ladder {i + 1}/{len(tok)}] {t['id']} ok", flush=True)

    Path(args.out).write_text(
        json.dumps(
            {
                "schema": "yarn-ladder-raw-1",
                "arm": args.arm,
                "pass_label": args.pass_label,
                "seed": SEED,
                "topk_requested": TOPK_REQUEST,
                "positions_with_fewer_than_topk_entries": short_positions,
                "forced_continuations_from": str(args.forced),
                "probe_set_sha256": hashlib.sha256(
                    Path(args.probes).read_bytes()
                ).hexdigest(),
                "elapsed_s_proot_inflated": round(time.time() - t0, 2),
                "server": server_facts(args.base),
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"wrote {args.out} ({short_positions} short positions)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True)
    g.add_argument("--probes", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_generate)
    l = sub.add_parser("ladder")
    l.add_argument("--base", required=True)
    l.add_argument("--probes", required=True)
    l.add_argument("--forced", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--arm", required=True)
    l.add_argument("--pass-label", required=True)
    l.set_defaults(fn=cmd_ladder)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
