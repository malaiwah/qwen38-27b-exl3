#!/usr/bin/env python3
"""Merge BF16 MTP + vision weights back into a text-only requant checkpoint.

Why this is needed
------------------
`llmcompressor` loads via `AutoModelForCausalLM`, which for
`Qwen3_5ForConditionalGeneration` keeps only the language model.  The vision
tower (`model.visual.*`, 333 tensors) and the MTP draft head (`mtp.*`, 15
tensors) are therefore silently dropped from every requant artifact.  Without
them the checkpoint cannot satisfy two of this project's north-star criteria:
speculative decode (MTP acceptance) and the vision red/blue check.

The weights are recoverable from the BF16 base at full fidelity -- they were
never quantized, so re-attaching the originals is lossless, not an
approximation.  This is the same pattern the SPARK_Qwen3.5-122B deployment used
to restore MTP to an NVFP4 checkpoint.

What this does
--------------
1. Hardlinks the source checkpoint into the destination (no 21 GB copy; the
   safetensors shards are read-only at serve time).
2. Writes the extracted BF16 tensors as one additional shard.
3. Rewrites `model.safetensors.index.json`: adds the new tensors to
   `weight_map` and corrects `metadata.total_size`.
4. Extends `quantization_config.ignore` with regexes covering the re-attached
   modules, so vLLM's compressed-tensors path leaves them BF16 instead of
   looking for quantized parameters that do not exist.  This is the failure the
   SPARK deployment hit from the other direction: an ignore list that does not
   match the names vLLM actually resolves produces silently zero-loaded layers.
5. Verifies every tensor in the merged index resolves to a file that contains
   it, and refuses to leave a broken checkpoint behind.

Serving note: the merged checkpoint is multimodal again, so it must be served
WITHOUT `--language-model-only` for the vision check to work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Regexes appended to quantization_config.ignore for the re-attached BF16
# modules. Broad on purpose: vLLM fuses some names at load (e.g. GDN in_proj*),
# and an ignore entry that misses the resolved name yields zero-loaded weights.
IGNORE_PATTERNS = (
    r"re:.*\.visual\..*",
    r"re:.*visual\.blocks\..*",
    r"re:.*visual\.merger\..*",
    r"re:.*visual\.patch_embed\..*",
    r"re:^mtp\..*",
    r"re:.*\bmtp\b.*",
)


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="text-only requant checkpoint dir")
    ap.add_argument("--dst", required=True, help="destination dir for the merged checkpoint")
    ap.add_argument("--extras", required=True,
                    help="dir holding mtp_weights.safetensors and visual_weights.safetensors")
    ap.add_argument("--shard-name", default="model-extra-mtp-vision.safetensors",
                    help="filename for the added BF16 shard")
    ap.add_argument("--base-snapshot", default=None,
                    help="base model snapshot dir; multimodal PROCESSOR configs are "
                         "copied from it. llmcompressor saves via "
                         "AutoModelForCausalLM and therefore omits "
                         "preprocessor_config.json / video_preprocessor_config.json, "
                         "without which vLLM refuses to load the vision tower "
                         "(OSError: Can't load image processor).")
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst, extras = Path(args.src), Path(args.dst), Path(args.extras)
    for p in (src, extras):
        if not p.is_dir():
            raise SystemExit(f"not a directory: {p}")
    dst.mkdir(parents=True, exist_ok=True)

    # 1. hardlink everything from src
    modes = {}
    for f in sorted(src.iterdir()):
        if f.is_file():
            modes[f.name] = link_or_copy(f, dst / f.name)
    print(f"staged {len(modes)} files ({sum(1 for v in modes.values() if v=='hardlink')} hardlinked)")

    # 2. gather extras into one shard
    tensors = {}
    for name in ("mtp_weights.safetensors", "visual_weights.safetensors"):
        p = extras / name
        if not p.is_file():
            raise SystemExit(f"missing extras file: {p}")
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    added_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"extras: {len(tensors)} tensors, {added_bytes/1024**3:.3f} GiB")
    if not tensors:
        raise SystemExit("no extra tensors found")
    save_file(tensors, str(dst / args.shard_name), metadata={"format": "pt"})

    # 3. rewrite index
    idx_path = dst / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]
    before = len(wm)
    clash = [k for k in tensors if k in wm]
    if clash:
        raise SystemExit(f"refusing to overwrite {len(clash)} existing tensors, e.g. {clash[:3]}")
    for k in tensors:
        wm[k] = args.shard_name
    meta = idx.setdefault("metadata", {})
    meta["total_size"] = int(meta.get("total_size", 0)) + added_bytes
    idx_path.write_text(json.dumps(idx, indent=2) + "\n")
    print(f"index: {before} -> {len(wm)} tensors, total_size {meta['total_size']:,}")

    # 4. extend the quantization ignore list
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qc = cfg.get("quantization_config")
    if qc is None:
        raise SystemExit("config.json has no quantization_config (is --src the composite config?)")
    ig = list(qc.get("ignore", []))
    added = [p for p in IGNORE_PATTERNS if p not in ig]
    qc["ignore"] = ig + added
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"ignore: {len(ig)} -> {len(qc['ignore'])} (+{len(added)} patterns for BF16 extras)")

    # 4b. multimodal processor configs (absent from any text-only save)
    if args.base_snapshot:
        base = Path(args.base_snapshot)
        copied = []
        for name in ("preprocessor_config.json", "video_preprocessor_config.json",
                     "processor_config.json", "chat_template.jinja"):
            src_f, dst_f = base / name, dst / name
            if src_f.is_file() and not dst_f.is_file():
                shutil.copy2(src_f, dst_f)
                copied.append(name)
        print(f"processor configs copied: {copied or 'none needed'}")

    # 5. verify every tensor resolves
    by_file: dict[str, list[str]] = {}
    for k, f in wm.items():
        by_file.setdefault(f, []).append(k)
    missing_files = [f for f in by_file if not (dst / f).is_file()]
    if missing_files:
        raise SystemExit(f"index references missing files: {missing_files}")
    bad = []
    for f, keys in by_file.items():
        with safe_open(dst / f, framework="pt") as h:
            have = set(h.keys())
        absent = [k for k in keys if k not in have]
        if absent:
            bad.append((f, len(absent), absent[:2]))
    if bad:
        raise SystemExit(f"index/shard mismatch, refusing: {bad}")
    print(f"verified: {len(wm)} tensors across {len(by_file)} shards all resolve")
    print()
    print(f"merged checkpoint -> {dst}")
    print("serve WITHOUT --language-model-only so the vision tower loads.")


if __name__ == "__main__":
    main()
