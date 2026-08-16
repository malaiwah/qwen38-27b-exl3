#!/usr/bin/env python3
"""Read a GGUF's tensor table over HTTP range requests and account its bytes by role.

Why over HTTP: a byte comparison against a GGUF needs to know how that GGUF spends its bytes,
and the whole file is 21 GiB of re-fetchable mirror.  The tensor table lives in the header, so
a range request over the first few tens of MB answers the question without downloading the
payload.  Everything printed is derived from the file's own table -- tensor names, shapes,
GGML types and the block sizes those types imply -- so it can be checked against upstream.

The role split exists because "equal file bytes" is not "equal transformer body bytes": a GGUF
quantizes `token_embd` while our EXL3 builds serialize the embedding at BF16, so at equal file
size the GGUF's body carries more bytes than ours.  That asymmetry has to be stated wherever
the two are compared at parity.

    gguf_tensor_table.py --repo unsloth/Qwen3.8-27B-GGUF --revision f1bfb12... \\
        --file Qwen3.8-27B-Q6_K.gguf --out /tmp/q6k-tensors.json
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

GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
             9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
             15: "Q8_K", 30: "BF16"}
# (elements per block, bytes per block) from ggml's own type traits.
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q6_K": (256, 210),
         "Q5_K": (256, 176), "Q4_K": (256, 144), "Q3_K": (256, 110), "Q2_K": (256, 84),
         "Q5_0": (32, 22), "Q4_0": (32, 18)}


class Reader:
    def __init__(self, buf: bytes) -> None:
        self.b, self.o = buf, 0

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v

    def string(self) -> str:
        n = self.u64(); v = self.b[self.o:self.o + n].decode("utf8", "replace")
        self.o += n; return v

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32(); n = self.u64()
            return [self.value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
               10: "<Q", 11: "<q", 12: "<d"}[t]
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += struct.calcsize(fmt)
        return v


def fetch_head(url: str, nbytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-%d" % (nbytes - 1)})
    home = os.environ.get("HF_HOME", "")
    for name in ("token",):
        p = Path(home, name)
        if p.exists():
            tok = p.read_text().strip().splitlines()[0]
            if tok.startswith("hf_"):
                req.add_header("Authorization", "Bearer " + tok)
            break
    return urllib.request.urlopen(req, timeout=300).read()


def parse(buf: bytes) -> tuple[dict, list]:
    r = Reader(buf)
    if r.b[:4] != b"GGUF":
        raise SystemExit("not a GGUF")
    r.o = 4
    version = r.u32(); n_tensors = r.u64(); n_kv = r.u64()
    kv = {}
    for _ in range(n_kv):
        k = r.string(); t = r.u32(); v = r.value(t)
        kv[k] = ("[%d items]" % len(v)) if isinstance(v, list) else v
    tensors = []
    for _ in range(n_tensors):
        name = r.string(); nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = GGML_TYPE.get(r.u32(), "?"); offset = r.u64()
        n = 1
        for d in dims:
            n *= d
        blk = BLOCK.get(ttype)
        tensors.append({"name": name, "dims": dims, "type": ttype, "elements": n,
                        "offset": offset,
                        "bytes": (n // blk[0] * blk[1]) if blk else None})
    return {"gguf_version": version, "tensor_count": n_tensors, "kv_count": n_kv,
            "kv": kv, "tensor_table_end_offset": r.o}, tensors


def role_of(name: str) -> str:
    if name.startswith("token_embd"):
        return "token_embd"
    if name.startswith("output."):
        return "output_head"
    return "body"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--bytes", type=int, default=64 << 20,
                    help="header slice to fetch; must cover the KV block and tensor table")
    ap.add_argument("--file-bytes", type=int, help="known total file size, for the metadata check")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    url = "https://huggingface.co/%s/resolve/%s/%s" % (a.repo, a.revision, a.file)
    head, tensors = parse(fetch_head(url, a.bytes))
    by_role: dict = {}
    for t in tensors:
        r = by_role.setdefault(role_of(t["name"]), {"tensors": 0, "bytes": 0, "types": {}})
        r["tensors"] += 1
        r["bytes"] += t["bytes"] or 0
        r["types"][t["type"]] = r["types"].get(t["type"], 0) + 1
    total = sum(v["bytes"] for v in by_role.values())
    doc = {
        "schema": "gguf-tensor-accounting/1",
        "source": {"repo": a.repo, "revision": a.revision, "file": a.file, "url": url,
                   "read": "HTTP range request over the first %d bytes; the payload is never "
                           "downloaded" % a.bytes},
        "header": {k: head[k] for k in ("gguf_version", "tensor_count", "kv_count",
                                        "tensor_table_end_offset")},
        "selected_kv": {k: v for k, v in head["kv"].items()
                        if any(s in k for s in ("file_type", "block_count", "embedding_length",
                                                "architecture", "name"))},
        "by_role": {k: dict(v, gib=round(v["bytes"] / GIB, 4)) for k, v in sorted(by_role.items())},
        "tensor_bytes_total": total,
        "tensor_bytes_total_gib": round(total / GIB, 4),
        "notable_tensors": {t["name"]: {"dims": t["dims"], "type": t["type"], "bytes": t["bytes"]}
                            for t in tensors if role_of(t["name"]) != "body"},
        "block_size_table_used": BLOCK,
    }
    if a.file_bytes:
        doc["file_bytes"] = a.file_bytes
        doc["file_bytes_gib"] = round(a.file_bytes / GIB, 4)
        doc["metadata_and_padding_bytes"] = a.file_bytes - total
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("by_role", "tensor_bytes_total_gib",
                                          "notable_tensors")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
