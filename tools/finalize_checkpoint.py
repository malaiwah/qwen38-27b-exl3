#!/usr/bin/env python3
"""Verify, repair and document an EXL3 checkpoint before publication.

Three jobs, all of them things that bit us or that an independent review flagged:

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

3. **Emit a build receipt.** `build-receipt.json` + `SHA256SUMS` so the artifact can be
   audited without trusting the publisher: source revision, converter identity,
   command, per-file hashes, tensor inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

EXL3_SUFFIX = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--source-repo", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--source-revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    ap.add_argument("--converter", default="turboderp-org/exllamav3@5f3c537 (1.4.2)")
    ap.add_argument("--command", default="")
    ap.add_argument("--recipe", default="")
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
        qc_path.write_text(json.dumps(qc, indent=2) + "\n")
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
        "exl3_modules": len(modules),
        "roles": {k: {"modules": v["modules"], "bytes": v["bytes"],
                      "gb": round(v["bytes"] / 1e9, 3),
                      "formats": dict(v["formats"])}
                  for k, v in sorted(roles.items())},
        "note": ("config.json -> quantization_config keeps a single bits/codebook/mtp_bits "
                 "triple for loader compatibility; it cannot describe a mixed checkpoint. "
                 "This file and quantization_config.json -> tensor_storage are authoritative."),
    }
    (d / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for role, v in manifest["roles"].items():
        print(f"  {role:20s} {v['gb']:7.3f} GB  {dict(v['formats'])}", flush=True)

    # ---- receipt + checksums
    files = sorted(p for p in d.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    sums = {p.name: sha256_file(p) for p in files}
    (d / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    receipt = {
        "schema": "qwen38-build-receipt/1",
        "artifact": d.name,
        "source": manifest["source"],
        "converter": args.converter,
        "recipe": args.recipe,
        "command": args.command,
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": sums,
        "tensor_inventory": {
            "physical_tensors": len(headers),
            "exl3_modules": len(modules),
            "logical_tensors_reconstructed": len(modules) + len(plain),
        },
        "host": subprocess.run(["uname", "-sr"], capture_output=True, text=True).stdout.strip(),
    }
    (d / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"receipt: {receipt['total_bytes']/1e9:.2f} GB over {len(files)} files, "
          f"{receipt['tensor_inventory']['logical_tensors_reconstructed']} logical tensors",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
