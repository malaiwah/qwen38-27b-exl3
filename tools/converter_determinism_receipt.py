#!/usr/bin/env python3
"""Is a fresh conversion of a published recipe byte-identical to the published checkpoint?

Compares two checkpoint trees at three depths, because "the digests differ" is not an
answer -- what a reader needs to know is *where* they differ:

  1. the pinned payload digests (the published `SHA256SUMS`, the 16 files that define the
     artifact) -- the headline yes/no;
  2. for a safetensors file that differs, its **header** against the rebuild's: tensor
     names, dtypes, shapes and data offsets.  A header-only difference means the same
     tensors were packed differently or the metadata moved, and the tensor bytes are
     intact;
  3. for a safetensors file whose header matches, every tensor's **bytes**, so a
     difference can be named per tensor rather than per 8 GB shard.  This is the
     distinction that decides whether a reconstruction claim survives.

Fail-closed: the published tree is re-verified against its own `SHA256SUMS` before
anything is compared, so a corrupted reference cannot be reported as a converter defect.

    converter_determinism_receipt.py --published DIR --rebuild DIR \\
        --sums receipts/hydrated-SHA256SUMS --out receipts/converter-determinism.json \\
        --toolchain toolchain.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

CHUNK = 8 << 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        out[name.strip()] = digest
    return out


def safetensors_header(p: Path) -> tuple[dict, int]:
    """`(header, data_start)`.  The header is the JSON blob safetensors puts up front."""
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def compare_headers(a: dict, b: dict) -> dict:
    a_meta, b_meta = a.get("__metadata__"), b.get("__metadata__")
    a_t = {k: v for k, v in a.items() if k != "__metadata__"}
    b_t = {k: v for k, v in b.items() if k != "__metadata__"}
    only_a = sorted(set(a_t) - set(b_t))
    only_b = sorted(set(b_t) - set(a_t))
    moved, retyped, reshaped = [], [], []
    for k in sorted(set(a_t) & set(b_t)):
        if a_t[k].get("dtype") != b_t[k].get("dtype"):
            retyped.append({"tensor": k, "published": a_t[k].get("dtype"),
                            "rebuild": b_t[k].get("dtype")})
        if a_t[k].get("shape") != b_t[k].get("shape"):
            reshaped.append({"tensor": k, "published": a_t[k].get("shape"),
                             "rebuild": b_t[k].get("shape")})
        if a_t[k].get("data_offsets") != b_t[k].get("data_offsets"):
            moved.append({"tensor": k, "published": a_t[k].get("data_offsets"),
                          "rebuild": b_t[k].get("data_offsets")})
    return {
        "identical": (a == b),
        "tensors_published": len(a_t),
        "tensors_rebuild": len(b_t),
        "metadata_identical": a_meta == b_meta,
        "metadata_published": a_meta,
        "metadata_rebuild": b_meta,
        "only_in_published": only_a,
        "only_in_rebuild": only_b,
        "dtype_changed": retyped,
        "shape_changed": reshaped,
        "offsets_changed_count": len(moved),
        "offsets_changed_sample": moved[:20],
    }


def compare_tensor_bytes(pa: Path, pb: Path, header: dict, base_a: int,
                         base_b: int) -> dict:
    """Per-tensor byte comparison, for two files whose headers already agree."""
    tensors = sorted(((k, v["data_offsets"]) for k, v in header.items()
                      if k != "__metadata__"), key=lambda kv: kv[1][0])
    differing, first_diff = [], None
    equal_bytes = diff_bytes = 0
    with pa.open("rb") as fa, pb.open("rb") as fb:
        for name, (beg, end) in tensors:
            n = end - beg
            fa.seek(base_a + beg)
            fb.seek(base_b + beg)
            left = n
            bad = 0
            local_first = None
            while left:
                take = min(CHUNK, left)
                ba, bb = fa.read(take), fb.read(take)
                if ba != bb:
                    for i, (x, y) in enumerate(zip(ba, bb)):
                        if x != y:
                            bad += 1
                            if local_first is None:
                                local_first = (n - left) + i
                    bad += abs(len(ba) - len(bb))
                left -= take
            if bad:
                differing.append({"tensor": name, "bytes": n, "differing_bytes": bad,
                                  "first_differing_offset_in_tensor": local_first})
                diff_bytes += n
                if first_diff is None:
                    first_diff = name
            else:
                equal_bytes += n
    return {
        "tensors_compared": len(tensors),
        "tensors_identical": len(tensors) - len(differing),
        "tensors_differing": len(differing),
        "bytes_identical": equal_bytes,
        "bytes_in_differing_tensors": diff_bytes,
        "first_differing_tensor": first_diff,
        "differing": differing[:200],
        "differing_truncated": len(differing) > 200,
    }


def json_diff(a, b, path="") -> list[str]:
    out = []
    if type(a) is not type(b):
        return [f"{path or '<root>'}: type {type(a).__name__} -> {type(b).__name__}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in rebuild")
            elif k not in b:
                out.append(f"{path}.{k}: only in published")
            else:
                out += json_diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += json_diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path or '<root>'}: {a!r} -> {b!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", required=True, type=Path)
    ap.add_argument("--rebuild", required=True, type=Path)
    ap.add_argument("--sums", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--toolchain", type=Path,
                    help="JSON describing both runs' toolchain; merged verbatim")
    args = ap.parse_args()

    pinned = read_sums(args.sums)

    # fail closed: the reference must still be the artifact its digests describe
    published_now = {}
    for name in sorted(pinned):
        p = args.published / name
        if not p.is_file():
            raise SystemExit(f"published tree is missing {name}")
        published_now[name] = sha256_file(p)
    drifted = [n for n in pinned if published_now[n] != pinned[n]]
    if drifted:
        raise SystemExit("published tree no longer matches its own SHA256SUMS: "
                         + ", ".join(drifted))

    rebuild_now = {}
    for name in sorted(pinned):
        p = args.rebuild / name
        rebuild_now[name] = sha256_file(p) if p.is_file() else None

    identical = [n for n in sorted(pinned) if rebuild_now[n] == pinned[n]]
    differing = [n for n in sorted(pinned) if rebuild_now[n] not in (pinned[n],)]

    per_file = {}
    tensor_bytes_reached = False
    for name in differing:
        a, b = args.published / name, args.rebuild / name
        entry = {"published_sha256": pinned[name], "rebuild_sha256": rebuild_now[name],
                 "published_bytes": a.stat().st_size,
                 "rebuild_bytes": b.stat().st_size if b.is_file() else None}
        if rebuild_now[name] is None:
            entry["kind"] = "absent-from-rebuild"
        elif name.endswith(".safetensors"):
            ha, base_a = safetensors_header(a)
            hb, base_b = safetensors_header(b)
            entry["header"] = compare_headers(ha, hb)
            if entry["header"]["identical"]:
                entry["kind"] = "tensor-bytes"
                entry["tensors"] = compare_tensor_bytes(a, b, ha, base_a, base_b)
                tensor_bytes_reached = True
            else:
                entry["kind"] = "safetensors-header"
                # the tensors both files hold in common, at their own offsets
                common = sorted((set(ha) & set(hb)) - {"__metadata__"})
                same = 0
                with a.open("rb") as fa, b.open("rb") as fb:
                    for t in common:
                        ba0, ba1 = ha[t]["data_offsets"]
                        bb0, bb1 = hb[t]["data_offsets"]
                        if ba1 - ba0 != bb1 - bb0:
                            continue
                        fa.seek(base_a + ba0)
                        fb.seek(base_b + bb0)
                        left, eq = ba1 - ba0, True
                        while left and eq:
                            take = min(CHUNK, left)
                            eq = fa.read(take) == fb.read(take)
                            left -= take
                        same += eq
                entry["common_tensors"] = len(common)
                entry["common_tensors_byte_identical"] = same
                tensor_bytes_reached = tensor_bytes_reached or same < len(common)
        elif name.endswith(".json"):
            entry["kind"] = "json-metadata"
            try:
                entry["diff"] = json_diff(json.loads(a.read_text()),
                                          json.loads(b.read_text()))[:200]
            except ValueError as exc:
                entry["diff"] = [f"unparseable: {exc}"]
        else:
            entry["kind"] = "other"
        per_file[name] = entry

    payload = {
        "schema": "qwen38-converter-determinism/1",
        "question": "does a fresh conversion of the published hydrated recipe reproduce the "
                    "published checkpoint byte for byte?",
        "verdict": {
            "byte_identical": not differing,
            "files_compared": len(pinned),
            "files_identical": len(identical),
            "files_differing": len(differing),
            "differs_only_in_metadata": bool(differing) and not tensor_bytes_reached,
            "reaches_tensor_bytes": tensor_bytes_reached,
        },
        "published": {"dir": str(args.published), "sums": str(args.sums),
                      "sha256": pinned},
        "rebuild": {"dir": str(args.rebuild), "sha256": rebuild_now},
        "identical_files": identical,
        "differing_files": per_file,
    }
    if args.toolchain:
        payload["toolchain"] = json.loads(args.toolchain.read_text())

    body = json.dumps({k: v for k, v in payload.items() if k != "content_sha256"},
                      sort_keys=True).encode()
    payload["content_sha256"] = hashlib.sha256(body).hexdigest()
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
