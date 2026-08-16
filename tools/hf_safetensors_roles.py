#!/usr/bin/env python3
"""Account a Hub checkpoint's tensor bytes by role, from safetensors headers over HTTP.

A safetensors shard puts its whole header -- every tensor's name, dtype, shape and byte
offsets -- in the first few hundred kB, so the exact per-role byte split of a 26 GiB
checkpoint costs a handful of range requests and no payload download.  This exists because
the two oldest published builds in this family predate `quantization_manifest.json`, and a
cross-candidate byte table that silently omits them, or estimates them, is worth less than
one that reads them.

Roles match `finalize_checkpoint.py`'s own classification, so the output is directly
comparable with a `quantization_manifest.json` from a newer build.

    hf_safetensors_roles.py --repo malaiwah/Qwen3.8-27B-K4 --out /tmp/k4-roles.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

GIB = 2 ** 30


def token() -> str | None:
    p = Path(os.environ.get("HF_HOME", ""), "token")
    if p.exists():
        t = p.read_text().strip().splitlines()[0]
        if t.startswith("hf_"):
            return t
    return None


def get(url: str, rng: str | None = None) -> bytes:
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", "bytes=" + rng)
    t = token()
    if t:
        req.add_header("Authorization", "Bearer " + t)
    return urllib.request.urlopen(req, timeout=300).read()


def role_of(name: str) -> str:
    if "visual." in name or "vision_tower" in name:
        return "vision_tower"
    if "embed_tokens" in name:
        return "embed_tokens"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head"
    if name.startswith("mtp") or ".mtp." in name:
        return "mtp_draft"
    if "mlp.gate_proj" in name:
        return "mlp_gate_proj"
    if "mlp.up_proj" in name:
        return "mlp_up_proj"
    if "mlp.down_proj" in name:
        return "mlp_down_proj"
    if "linear_attn" in name:
        return "linear_attention"
    if "self_attn" in name:
        return "full_attention"
    return "norms_and_small"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--head-bytes", type=int, default=4 << 20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    api = "https://huggingface.co/api/models/%s?blobs=true" % a.repo
    if a.revision != "main":
        api = ("https://huggingface.co/api/models/%s/revision/%s?blobs=true"
               % (a.repo, a.revision))
    meta = json.loads(get(api))
    shards = sorted(s["rfilename"] for s in meta["siblings"]
                    if s["rfilename"].endswith(".safetensors"))
    sizes = {s["rfilename"]: s.get("size") for s in meta["siblings"]}
    if not shards:
        raise SystemExit("no safetensors shards in %s" % a.repo)

    roles: dict = {}
    tensors = 0
    dtypes: dict = {}
    for shard in shards:
        url = "https://huggingface.co/%s/resolve/%s/%s" % (a.repo, a.revision, shard)
        buf = get(url, "0-%d" % (a.head_bytes - 1))
        (hlen,) = struct.unpack("<Q", buf[:8])
        if hlen + 8 > len(buf):
            buf = get(url, "0-%d" % (hlen + 8))
        header = json.loads(buf[8:8 + hlen])
        for name, info in header.items():
            if name == "__metadata__":
                continue
            lo, hi = info["data_offsets"]
            r = roles.setdefault(role_of(name), {"tensors": 0, "bytes": 0, "dtypes": {}})
            r["tensors"] += 1
            r["bytes"] += hi - lo
            r["dtypes"][info["dtype"]] = r["dtypes"].get(info["dtype"], 0) + 1
            dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + 1
            tensors += 1

    total = sum(v["bytes"] for v in roles.values())
    doc = {
        "schema": "hf-safetensors-role-accounting/1",
        "source": {"repo": a.repo, "revision": meta.get("sha", a.revision),
                   "shards": shards, "shard_bytes": sizes,
                   "read": "safetensors headers only, over HTTP range requests; no payload "
                           "bytes are downloaded"},
        "physical_tensors": tensors,
        "dtype_histogram": dtypes,
        "by_role": {k: dict(v, gib=round(v["bytes"] / GIB, 4))
                    for k, v in sorted(roles.items())},
        "tensor_bytes_total": total,
        "tensor_bytes_total_gib": round(total / GIB, 4),
        "shard_bytes_total": sum(sizes[s] for s in shards if sizes.get(s)),
        "note": "tensor_bytes_total is the sum of tensor data ranges; shard_bytes_total is the "
                "files on disk, which additionally carry each shard's own JSON header",
    }
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"repo": a.repo, "by_role": doc["by_role"],
                      "tensor_bytes_total_gib": doc["tensor_bytes_total_gib"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
