#!/usr/bin/env python3
"""Verify, repair and document an EXL3 checkpoint before publication.

Four jobs, all of them things that bit us or that an independent review flagged:

1. **Repair `tensor_storage`.** exllamav3's `util/add_quant_config.py` walks the main
   model only, so a quantized MTP module gets **no** `tensor_storage` entries even
   though its EXL3 payloads are in the shards. The runtime then builds BF16 modules
   for those prefixes and `load_weights` dies with
   `There is no module or parameter named 'fc.mcg'`, which makes speculative decoding
   impossible. This scans the shards and adds an entry for every module that actually
   has EXL3 payloads.

2. **Emit an honest manifest.** `config.json -> quantization_config` carries a single
   `bits`/`codebook`/`mtp_bits` triple that cannot describe a mixed checkpoint (ours
   has K4/K5/K6 Trellis, two codebooks, BF16 and FP16 passthrough). `quantization_manifest.json`
   records what is actually stored, per role, with counts and bytes.

3. **Emit a build receipt with an honest scope.** `build-receipt.json` + `SHA256SUMS` so the
   artifact can be audited without trusting the publisher: source revision, converter
   identity, command, per-file hashes, tensor inventory. The hash map is split in two:
   `immutable_payload` (shards, index, configs, tokenizer files -- everything that defines
   the artifact) and `mutable_documentation` (`README.md`, `.gitattributes`, `assets/**` and
   the receipt files themselves), because a card edit must not be able to falsify the build's
   checksums, and a receipt that mixes the two goes stale while still claiming to describe
   the build. `SHA256SUMS` covers the payload only, `DOCS-SHA256SUMS` the documentation. Both
   lists are produced by scanning the directory, never from a hardcoded list, so the receipt
   cannot name a file that is absent -- the receipt this replaced advertised a hash for a
   `crc32.txt` that was not in the repository.

4. **Verify against upstream.** One EXL3 module is several physical tensors standing in
   for one logical weight, so a shard count or a tensor count proves nothing about
   completeness. This rebuilds the logical tensor name set and requires it to equal the
   upstream release's `model.safetensors.index.json` key set exactly, and requires every
   logical shape to match upstream's shard headers where those shards are on disk.
   Any missing or extra name aborts the run.

`quantization_manifest.json` additions: `physical_tensor_count` (stored tensors -- what the
Hub counts), `logical_tensor_count`, and `logical_parameter_count` (sum of prod(shape) over
the dequantized logical tensors -- the real parameter count). `physical_tensors` is kept as
its pre-existing alias so the schema stays additive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from collections import defaultdict
from fnmatch import fnmatch
from math import prod
from pathlib import Path

EXL3_SUFFIX = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")

# ---- receipt scope, by rule rather than by hand.
# The receipt exists so a downloader can prove the artifact is the one that was built. The
# card, its charts and the checksum files are edited without rebuilding anything, so they
# cannot live in a hash map that claims to describe the build.
PAYLOAD_NAMES = frozenset({
    "model.safetensors.index.json", "config.json", "quantization_config.json",
    "quantization_manifest.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "vocab.json", "merges.txt",
    # the redistribution licence ships with the weights and is byte-stable, unlike the card
    "LICENSE",
})
PAYLOAD_GLOBS = ("model*.safetensors",)
DOC_NAMES = frozenset({"README.md", ".gitattributes", "crc32.txt"})
DOC_DIRS = frozenset({"assets"})
# a receipt cannot hash itself without a fixpoint, so it names itself instead
RECEIPT_NAMES = ("SHA256SUMS", "DOCS-SHA256SUMS", "build-receipt.json")
# absent from the checkpoint means the artifact is incomplete, not that a line may be dropped
REQUIRED_PAYLOAD = ("model.safetensors.index.json", "config.json", "quantization_config.json",
                    "quantization_manifest.json", "tokenizer.json", "tokenizer_config.json")
SCOPE_NOTE = (
    "SHA256SUMS and immutable_payload cover only the files that define the artifact; "
    "README.md, .gitattributes, assets/** and the receipt files themselves are documentation "
    "that is rewritten by card edits without rebuilding the weights, so hashing them here "
    "would make the receipt go stale while still claiming to describe the build -- they are "
    "listed under mutable_documentation and hashed in DOCS-SHA256SUMS instead."
)


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: temp file in the same directory, then os.replace.

    Every artifact this script emits is a VERIFICATION artifact - digest manifests and a build
    receipt. A bare write_text that is interrupted leaves a TRUNCATED digest file, which is
    strictly worse than no file at all, because a short SHA256SUMS still parses and silently
    verifies fewer files than it claims to. os.replace is atomic within a filesystem, so a
    reader sees either the previous content or the complete new content and never a partial one.

    Found by a peer review of AGENTS.md's claim that receipts are written atomically: the claim
    held for kld_aggregate.py and build-image.sh but was false here, for all four artifacts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def classify_scopes(root: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """Scan `root` and split what is really there into (immutable payload, mutable docs).

    Scanned, never listed: a hardcoded list is how the previous receipt came to advertise a
    hash for a `crc32.txt` that no longer existed and to carry a stale `README.md` digest. A
    file matching neither rule aborts the run rather than being guessed into a scope, because
    guessing wrong either hides a payload file from the checksums or nails documentation into
    a map that a card edit will invalidate.
    """
    payload: list[tuple[str, Path]] = []
    docs: list[tuple[str, Path]] = []
    unknown: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        name = rel.as_posix()
        if rel.parts[0] == ".git" or rel.name in RECEIPT_NAMES:
            continue  # vcs internals, and the receipts this run is about to rewrite
        if rel.parts[0] in DOC_DIRS or rel.name in DOC_NAMES:
            docs.append((name, p))
        elif len(rel.parts) == 1 and (rel.name in PAYLOAD_NAMES
                                      or any(fnmatch(rel.name, g) for g in PAYLOAD_GLOBS)):
            payload.append((name, p))
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(f"{root}: {len(unknown)} file(s) match no receipt scope rule "
                         f"({', '.join(unknown[:6])}); add them to PAYLOAD_NAMES, "
                         f"PAYLOAD_GLOBS, DOC_NAMES or DOC_DIRS instead of letting the "
                         f"receipt guess which scope they belong to")
    return payload, docs


def role_of(key: str) -> str:
    if ".visual." in key or key.startswith("visual."):
        return "vision_tower"
    if key.startswith("mtp"):
        return "mtp_draft"
    if key == "lm_head" or key.startswith("lm_head."):
        return "lm_head"
    if "embed_tokens" in key:
        return "embed_tokens"
    if "linear_attn" in key:
        return "linear_attention"
    if "self_attn" in key:
        return "full_attention"
    if ".mlp." in key:
        for p in ("gate_proj", "up_proj", "down_proj"):
            if key.endswith(p):
                return f"mlp_{p}"
        return "mlp_other"
    return "norms_and_small"


def verify_upstream(upstream: Path, logical: dict[str, list[int]]) -> dict:
    """Diff the reconstructed logical tensor set against the upstream release.

    Fail-closed: a missing name means the quantization silently dropped a weight, an
    extra name means it invented one, and either makes the artifact unpublishable.
    """
    index_path = upstream / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"upstream checkpoint has no {index_path}")
    up_index = json.loads(index_path.read_text())["weight_map"]
    up_keys = set(up_index)
    missing = sorted(up_keys - set(logical))
    extra = sorted(set(logical) - up_keys)
    if missing or extra:
        print(f"logical tensor set differs from upstream: {len(missing)} missing, "
              f"{len(extra)} extra", flush=True)
        for label, names in (("missing (upstream has it, we do not)", missing),
                             ("extra (we have it, upstream does not)", extra)):
            for name in names[:20]:
                print(f"  {label}: {name}", flush=True)
            if len(names) > 20:
                print(f"  ... and {len(names) - 20} more {label}", flush=True)
        raise SystemExit(1)

    up_shapes: dict[str, list[int]] = {}
    for shard in sorted(set(up_index.values())):
        p = upstream / shard
        if not p.is_file():
            continue
        for k, v in read_header(p).items():
            if k != "__metadata__":
                up_shapes[k] = v["shape"]
    bad = [(k, logical[k], up_shapes[k]) for k in sorted(up_shapes)
           if logical[k] != up_shapes[k]]
    if bad:
        print(f"{len(bad)} logical tensors disagree with upstream on shape", flush=True)
        for k, got, want in bad[:20]:
            print(f"  {k}: {got} != upstream {want}", flush=True)
        raise SystemExit(1)
    print(f"upstream: {len(up_keys)} logical tensor names match exactly, "
          f"{len(up_shapes)} shapes verified against shard headers", flush=True)
    return {"dir": str(upstream), "index_sha256": sha256_file(index_path),
            "logical_tensors": len(up_keys), "shapes_verified": len(up_shapes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--upstream", required=True,
                    help="unquantized upstream checkpoint to verify the tensor set against")
    ap.add_argument("--source-repo", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--source-revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    ap.add_argument("--converter", default="turboderp-org/exllamav3@5f3c537 (1.4.2)")
    ap.add_argument("--command", default="")
    ap.add_argument("--recipe", default="")
    ap.add_argument("--docs-dir", action="append", default=[], metavar="DIR",
                    help="extra directory holding card documentation (README.md, assets/) that "
                         "is published alongside the checkpoint; hashed into DOCS-SHA256SUMS, "
                         "never into SHA256SUMS. Repeatable. The checkpoint dir is always "
                         "scanned for documentation too")
    args = ap.parse_args()
    d = Path(args.model)

    # ---- inventory every physical tensor
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    headers = {}
    for s in shards:
        for k, v in read_header(d / s).items():
            if k != "__metadata__":
                headers[k] = v

    modules: dict[str, dict] = defaultdict(dict)
    plain: dict[str, str] = {}
    for key, meta in headers.items():
        prefix, _, suffix = key.rpartition(".")
        if suffix in EXL3_SUFFIX:
            modules[prefix][suffix] = meta
        else:
            plain[key] = meta["dtype"]

    # ---- reconstruct the logical tensor set and hold it against upstream
    logical: dict[str, list[int]] = {}
    for prefix, tensors in modules.items():
        suh, svh = tensors.get("suh"), tensors.get("svh")
        if suh is None or svh is None:
            raise SystemExit(f"{prefix}: EXL3 module has no suh/svh, cannot reconstruct "
                             f"its logical shape")
        name = f"{prefix}.weight"
        if name in headers:
            raise SystemExit(f"{name} is stored both as an EXL3 module and as a plain tensor")
        # exl3 keeps the input-side scales in suh (length K) and the output-side in
        # svh (length N), so the dequantized weight is (N, K) like the source.
        logical[name] = [svh["shape"][0], suh["shape"][0]]
    for key in plain:
        logical[key] = headers[key]["shape"]
    upstream_check = verify_upstream(Path(args.upstream), logical)
    logical_parameter_count = sum(prod(s) for s in logical.values())

    # ---- repair tensor_storage
    qc_path = d / "quantization_config.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {"tensor_storage": {}}
    ts = qc.setdefault("tensor_storage", {})
    added = []
    for prefix, tensors in sorted(modules.items()):
        entry = ts.get(prefix)
        stored = {f"{prefix}.{suf}": {"shape": m["shape"], "dtype": m["dtype"]}
                  for suf, m in sorted(tensors.items())}
        bits = tensors["trellis"]["shape"][-1] // 16 if "trellis" in tensors else None
        if entry is None or entry.get("quant_format") != "exl3":
            ts[prefix] = {"quant_format": "exl3", "bits_per_weight": bits,
                          "stored_tensors": stored}
            added.append(prefix)
    if added:
        write_atomic(qc_path, json.dumps(qc, indent=2) + "\n")
    print(f"tensor_storage: {len(ts)} entries, {len(added)} added "
          f"({', '.join(added[:3])}{'...' if len(added) > 3 else ''})", flush=True)

    # ---- honest manifest
    roles: dict[str, dict] = defaultdict(lambda: {"modules": 0, "bytes": 0,
                                                  "formats": defaultdict(int)})
    for prefix, tensors in modules.items():
        r = roles[role_of(prefix)]
        r["modules"] += 1
        bits = tensors["trellis"]["shape"][-1] // 16
        cb = "mcg" if "mcg" in tensors else ("mul1" if "mul1" in tensors else "none")
        r["formats"][f"exl3_K{bits}_{cb}"] += 1
        for m in tensors.values():
            r["bytes"] += m["data_offsets"][1] - m["data_offsets"][0]
    for key, dtype in plain.items():
        prefix = key.rpartition(".")[0] or key
        r = roles[role_of(prefix)]
        r["formats"][dtype] += 1
        r["bytes"] += headers[key]["data_offsets"][1] - headers[key]["data_offsets"][0]
    manifest = {
        "schema": "qwen38-quantization-manifest/1",
        "source": {"repo": args.source_repo, "revision": args.source_revision},
        "converter": args.converter,
        "recipe": args.recipe,
        "command": args.command,
        "physical_tensors": len(headers),
        "physical_tensor_count": len(headers),
        "logical_tensor_count": len(logical),
        "logical_parameter_count": logical_parameter_count,
        "exl3_modules": len(modules),
        "roles": {k: {"modules": v["modules"], "bytes": v["bytes"],
                      "gb": round(v["bytes"] / 1e9, 3),
                      "formats": dict(v["formats"])}
                  for k, v in sorted(roles.items())},
        "note": ("config.json -> quantization_config keeps a single bits/codebook/mtp_bits "
                 "triple for loader compatibility; it cannot describe a mixed checkpoint. "
                 "This file and quantization_config.json -> tensor_storage are authoritative. "
                 "physical_tensor_count counts stored tensors, and one EXL3 module is several "
                 "of them, so the Hub's tensor and parameter totals report packed storage "
                 "elements; logical_parameter_count is the real parameter count."),
    }
    write_atomic(d / "quantization_manifest.json", json.dumps(manifest, indent=2) + "\n")
    for role, v in manifest["roles"].items():
        print(f"  {role:20s} {v['gb']:7.3f} GB  {dict(v['formats'])}", flush=True)

    # ---- receipt + checksums, in two explicit scopes
    payload, docs = classify_scopes(d)
    for extra in args.docs_dir:  # the card and its charts usually live outside the checkpoint
        docs += classify_scopes(Path(extra))[1]
    absent = sorted({*shards, *REQUIRED_PAYLOAD} - {n for n, _ in payload})
    if absent:
        raise SystemExit(f"{d}: immutable payload is incomplete, {len(absent)} required "
                         f"file(s) are not on disk: {', '.join(absent)}")
    payload_sums = {n: sha256_file(p) for n, p in payload}
    payload_bytes = sum(p.stat().st_size for _, p in payload)
    write_atomic(d / "SHA256SUMS", "".join(f"{h}  {n}\n" for n, h in payload_sums.items()))

    doc_sums: dict[str, str] = {}
    doc_bytes = 0
    for name, p in sorted(docs):
        h = sha256_file(p)
        if name in doc_sums:
            if doc_sums[name] != h:
                raise SystemExit(f"two different files are published as {name}; pick one "
                                 f"before the documentation checksums can mean anything")
            continue
        doc_sums[name] = h
        doc_bytes += p.stat().st_size
    docs_path = d / "DOCS-SHA256SUMS"
    if doc_sums:
        write_atomic(docs_path, "".join(f"{h}  {n}\n" for n, h in doc_sums.items()))
    elif docs_path.exists():
        docs_path.unlink()  # no documentation on disk, so any previous list is now fiction

    receipt = {
        # /2: the single `files` map and `total_bytes` of /1 are replaced by the two scopes
        # below, so a consumer must not read either as the whole-repository inventory.
        "schema": "qwen38-build-receipt/2",
        "artifact": d.name,
        "source": manifest["source"],
        "converter": args.converter,
        "recipe": args.recipe,
        "command": args.command,
        "scope_note": SCOPE_NOTE,
        "immutable_payload_bytes": payload_bytes,
        "file_count": len(payload_sums),
        "immutable_payload": payload_sums,
        "mutable_documentation": {
            "bytes": doc_bytes,
            "file_count": len(doc_sums),
            "files": doc_sums,
            "self_excluded": list(RECEIPT_NAMES),
        },
        "tensor_inventory": {
            "physical_tensors": len(headers),
            "exl3_modules": len(modules),
            "logical_tensors_reconstructed": len(logical),
            "logical_parameter_count": logical_parameter_count,
        },
        "upstream_check": upstream_check,
        "host": subprocess.run(["uname", "-sr"], capture_output=True, text=True).stdout.strip(),
    }
    write_atomic(d / "build-receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(f"receipt: immutable payload {payload_bytes/1e9:.2f} GB over {len(payload_sums)} "
          f"files in SHA256SUMS, {len(logical)} logical tensors; documentation "
          f"{doc_bytes/1e6:.2f} MB over {len(doc_sums)} files in DOCS-SHA256SUMS, excluded "
          f"from the build hashes by design", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
