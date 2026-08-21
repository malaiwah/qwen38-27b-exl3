#!/usr/bin/env python3
"""Immutable real-data contracts for Qwen3.8 Wave 5.

This tool deliberately separates manifest construction from method access.  It can
inventory safetensors without loading a model, freeze document-level token splits,
and capture exact vLLM module boundaries through worker RPC hooks.  Tensor payloads
stay outside the repository; manifests contain content hashes and durable paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = "qwen38-wave5-data/1"
SELECTED_DEPTHS = (0, 7, 14, 21, 28, 35, 42, 49, 55)
BLOCK_SIZE = 128
BLOCK16 = 16
SCREEN_SEED = 0x523239
SPLIT_SEED = 0x57354635
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I16": 2,
               "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8}
ROLE_SUFFIXES = {
    "gate": ("mlp.gate_proj.weight",),
    "up": ("mlp.up_proj.weight",),
    "down": ("mlp.down_proj.weight",),
    "gdn_qkvz": ("linear_attn.in_proj_qkvz.weight",),
    "gdn_qkv": ("linear_attn.in_proj_qkv.weight",),
    "gdn_z": ("linear_attn.in_proj_z.weight",),
    "gdn_out": ("linear_attn.out_proj.weight",),
    "gdn_conv": ("linear_attn.conv1d.weight", "linear_attn.conv1d.bias"),
    "gdn_a": ("linear_attn.in_proj_a.weight", "linear_attn.A_log"),
    "gdn_b": ("linear_attn.in_proj_b.weight", "linear_attn.dt_bias"),
    "full_q": ("self_attn.q_proj.weight",),
    "full_k": ("self_attn.k_proj.weight",),
    "full_v": ("self_attn.v_proj.weight",),
    "full_o": ("self_attn.o_proj.weight",),
    "full_qkv": ("self_attn.qkv_proj.weight",),
    "output_gate": ("self_attn.gate_proj.weight", "self_attn.output_gate.weight"),
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, offset: int = 0, length: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(offset)
        left = length
        while left is None or left:
            chunk = f.read(8 << 20 if left is None else min(8 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            if left is not None:
                left -= len(chunk)
    if length is not None and left:
        raise EOFError(f"short read hashing {path}: {left} bytes absent")
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def software_revision() -> dict[str, Any]:
    here = Path(__file__).resolve()
    root = next((p for p in here.parents if (p / ".git").exists()), None)
    revision = None
    dirty = None
    if root:
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                      check=True, text=True, capture_output=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(here)],
                                        cwd=root, check=True, text=True,
                                        capture_output=True).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "script": str(here), "script_sha256": sha256_file(here),
            "git_revision": revision, "script_dirty": dirty}


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size <= 1 or size > 100_000_000:
            raise ValueError(f"invalid safetensors header length {size}: {path}")
        header = json.loads(f.read(size))
    return 8 + size, header


def bf16_payload_to_float32(payload: bytes | memoryview | np.ndarray) -> np.ndarray:
    """Decode little-endian BF16 exactly; never reinterpret as IEEE FP16."""
    if isinstance(payload, np.ndarray):
        u16 = np.asarray(payload, dtype="<u2")
    else:
        u16 = np.frombuffer(payload, dtype="<u2")
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_decoder_self_test(shard: Path, name: str, record: dict[str, Any]) -> dict[str, Any]:
    data_start, _ = read_safetensors_header(shard)
    start, end = record["data_offsets"]
    n = min(4096, (end - start) // 2)
    with shard.open("rb") as f:
        f.seek(data_start + start)
        payload = f.read(n * 2)
    manual = bf16_payload_to_float32(payload).copy()
    wrong = np.frombuffer(payload, dtype="<f2").astype(np.float32)
    finite = np.isfinite(manual)
    if not finite.all():
        raise AssertionError(f"known tensor BF16 prefix is non-finite: {name}")
    try:
        import torch
        direct = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).float().numpy()
        bit_exact = bool(np.array_equal(manual.view(np.uint32), direct.view(np.uint32)))
    except ImportError:
        direct = None
        bit_exact = None
    wrong_equal = bool(np.array_equal(manual.view(np.uint32), wrong.view(np.uint32)))
    if wrong_equal:
        raise AssertionError("BF16-as-FP16 rejection sentinel unexpectedly did not differ")
    if bit_exact is False:
        raise AssertionError("manual BF16 decoder disagrees with torch.bfloat16")
    return {"known_tensor": name, "shard": str(shard), "values_checked": n,
            "manual_vs_torch_bit_exact": bit_exact,
            "bf16_as_fp16_rejected": True,
            "wrong_fp16_max_abs_error": float(np.max(np.abs(manual - wrong))),
            "decoded_prefix_sha256": sha256_bytes(manual.tobytes())}


def classify_role(name: str) -> str | None:
    for role, suffixes in ROLE_SUFFIXES.items():
        if any(name.endswith(s) for s in suffixes):
            return role
    return None


def layer_from_name(name: str) -> int | None:
    parts = name.split(".")
    for marker in ("layers", "h"):
        if marker in parts:
            i = parts.index(marker) + 1
            if i < len(parts) and parts[i].isdigit():
                return int(parts[i])
    return None


def block_coordinates(shape: list[int], seed: int) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    if len(shape) != 2:
        return [], []
    rows, cols = shape
    rb = max(1, math.ceil(rows / BLOCK_SIZE))
    cb = max(1, math.ceil(cols / BLOCK_SIZE))
    diag_last = min(rb, cb) - 1
    base = [
        (0, 0, "first_diagonal"),
        (diag_last // 2, diag_last // 2, "middle_diagonal"),
        (diag_last, diag_last, "last_diagonal"),
        (0, cb - 1, "first_row_last_col"),
        (rb - 1, 0, "last_row_first_col"),
    ]
    rng = random.Random(seed)
    base += [(rng.randrange(rb), rng.randrange(cb), f"seeded_random_{i}") for i in range(3)]
    seen: set[tuple[int, int]] = set()
    screen = []
    for r, c, kind in base:
        if (r, c) in seen:
            # Small matrices still get eight preregistered identifiers; repeated
            # coordinates are explicit rather than silently shrinking coverage.
            kind += "_coordinate_repeat"
        seen.add((r, c))
        screen.append({"row": r * BLOCK_SIZE, "col": c * BLOCK_SIZE,
                       "rows": min(BLOCK_SIZE, rows - r * BLOCK_SIZE),
                       "cols": min(BLOCK_SIZE, cols - c * BLOCK_SIZE), "kind": kind})
    promotion = list(screen)
    while len(promotion) < 20:
        r, c = rng.randrange(rb), rng.randrange(cb)
        promotion.append({"row": r * BLOCK_SIZE, "col": c * BLOCK_SIZE,
                          "rows": min(BLOCK_SIZE, rows - r * BLOCK_SIZE),
                          "cols": min(BLOCK_SIZE, cols - c * BLOCK_SIZE),
                          "kind": f"promotion_seeded_{len(promotion) - 8}"})
    return screen, promotion


def tensor_block_hash(shard: Path, entry: dict[str, Any], block: dict[str, Any]) -> str:
    # The file is row-major. Hash the exact raw BF16 rectangle with shape included,
    # so two equal byte strings at different geometries cannot alias semantically.
    data_start, _ = read_safetensors_header(shard)
    start, _ = entry["data_offsets"]
    rows, cols = entry["shape"]
    item = DTYPE_BYTES[entry["dtype"]]
    h = hashlib.sha256()
    h.update(canonical_bytes({k: block[k] for k in ("row", "col", "rows", "cols")}))
    with shard.open("rb") as f:
        for r in range(block["row"], block["row"] + block["rows"]):
            f.seek(data_start + start + (r * cols + block["col"]) * item)
            h.update(f.read(block["cols"] * item))
    return h.hexdigest()


def verify_topology(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    types = text["layer_types"]
    actual = {
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "layer_types_count": len(types),
        "linear_attention_layers": sum(t == "linear_attention" for t in types),
        "full_attention_layers": sum(t == "full_attention" for t in types),
        "hidden_size": int(text["hidden_size"]),
        "intermediate_size": int(text["intermediate_size"]),
        "full_attention": {"query_heads": int(text["num_attention_heads"]),
                           "kv_heads": int(text["num_key_value_heads"]),
                           "head_dim": int(text["head_dim"]),
                           "rotary_dim": int(text["head_dim"] * text["partial_rotary_factor"])},
        "gdn": {"key_heads": int(text["linear_num_key_heads"]),
                "value_heads": int(text["linear_num_value_heads"]),
                "key_head_dim": int(text["linear_key_head_dim"]),
                "value_head_dim": int(text["linear_value_head_dim"]),
                "conv_kernel": int(text["linear_conv_kernel_dim"])},
        "selected_depth_types": {str(d): types[d] for d in SELECTED_DEPTHS},
    }
    expected = (64, 48, 16, 5120, 17408, 24, 4, 256, 64)
    observed = (actual["num_hidden_layers"], actual["linear_attention_layers"],
                actual["full_attention_layers"], actual["hidden_size"],
                actual["intermediate_size"], actual["full_attention"]["query_heads"],
                actual["full_attention"]["kv_heads"], actual["full_attention"]["head_dim"],
                actual["full_attention"]["rotary_dim"])
    if observed != expected or len(types) != 64:
        raise AssertionError(f"official topology mismatch: {observed}")
    actual["official_config_verified"] = True
    return actual


def cmd_census(args: argparse.Namespace) -> int:
    model = Path(args.model).resolve()
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    topology = verify_topology(config)
    weight_map = index["weight_map"]
    records = []
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    known = None
    for name, filename in sorted(weight_map.items()):
        layer = layer_from_name(name)
        role = classify_role(name)
        if (not name.startswith("model.language_model.layers.") or
                layer not in SELECTED_DEPTHS or role is None):
            continue
        shard = (model / filename).resolve()
        if filename not in headers:
            headers[filename] = read_safetensors_header(shard)
        data_start, header = headers[filename]
        if name not in header:
            raise KeyError(f"index/header mismatch: {name} not in {filename}")
        ent = header[name]
        if ent["dtype"] != "BF16":
            raise AssertionError(f"source tensor is not BF16: {name}: {ent['dtype']}")
        start, end = ent["data_offsets"]
        expected = math.prod(ent["shape"]) * DTYPE_BYTES[ent["dtype"]]
        if end - start != expected:
            raise AssertionError(f"payload length mismatch: {name}")
        screen, promotion = block_coordinates(ent["shape"], SCREEN_SEED ^ int(layer) ^
                                               int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        for b in screen:
            b["sha256"] = tensor_block_hash(shard, ent, b)
        rec = {"layer": layer, "topology": topology["selected_depth_types"][str(layer)],
               "role": role, "tensor_name": name, "shard": filename,
               "shard_path": str(shard), "dtype": ent["dtype"], "shape": ent["shape"],
               "header_bytes": data_start, "data_offsets": [start, end],
               "absolute_offsets": [data_start + start, data_start + end],
               "payload_bytes": end - start,
               "sha256": sha256_file(shard, data_start + start, end - start),
               "screening_blocks": screen, "promotion_blocks": promotion,
               "screening_seed": SCREEN_SEED}
        records.append(rec)
        if known is None:
            known = (shard, name, ent)
    if not records or known is None:
        raise SystemExit("no selected Qwen3.8 weight tensors found")
    # Enforce MLP at all depths and topology-specific attention coverage.
    by = {(r["layer"], r["role"]) for r in records}
    for d in SELECTED_DEPTHS:
        for role in ("gate", "up", "down"):
            if (d, role) not in by:
                raise AssertionError(f"missing {role} tensor at layer {d}")
        if topology["selected_depth_types"][str(d)] == "full_attention":
            if not any((d, role) in by for role in ("full_q", "full_qkv")):
                raise AssertionError(f"missing full-attention Q/QKV at layer {d}")
        elif not any((d, role) in by for role in ("gdn_qkvz", "gdn_qkv")):
            raise AssertionError(f"missing GDN QKV/Z at layer {d}")
    result = {"schema": SCHEMA, "kind": "immutable-source-weight-census",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "model_path": str(model), "model_index_sha256": sha256_file(index_path),
              "official_config_path": str(config_path),
              "official_config_sha256": sha256_file(config_path), "topology": topology,
              "selected_depths": list(SELECTED_DEPTHS), "record_count": len(records),
              "records": records,
              "bf16_decoder_self_test": bf16_decoder_self_test(*known),
              "software": software_revision()}
    result["content_sha256"] = sha256_bytes(canonical_bytes(result))
    atomic_json(Path(args.out), result)
    print(json.dumps({"out": args.out, "records": len(records),
                      "content_sha256": result["content_sha256"],
                      "bf16_self_test": result["bf16_decoder_self_test"]}, indent=2))
    return 0


def _domain(stratum: str) -> str:
    return {"code": "code", "multilingual": "multilingual",
            "literary": "prose", "encyclopedic": "prose",
            "scientific": "prose"}.get(stratum, stratum)

def select_split_documents(manifest: dict[str, Any], split: str,
                           *, allow_untouched_test: bool = False) -> list[dict[str, Any]]:
    """Fail closed before returning any untouched-test locator."""
    allowed = {"calibration", "validation", "untouched_test"}
    if split not in allowed:
        raise ValueError(f"unknown split {split!r}")
    if split == "untouched_test" and not allow_untouched_test:
        raise PermissionError(
            "untouched_test is sealed; freeze actions and pass allow_untouched_test=True "
            "only from the one-shot evaluation gate")
    return [doc for doc in manifest["documents"] if doc["split"] == split]


def cmd_split(args: argparse.Namespace) -> int:
    """Freeze calibration vs v5 validation/test without opening test contents."""
    from urllib.request import urlopen

    suite = Path(args.suite).resolve()
    source_manifest_path = suite / "suite-manifest.json"
    shard0 = json.loads(source_manifest_path.read_text())
    revision = "7797fcce3ffed62b99871348887f4626dc9b2b3b"
    base = ("https://huggingface.co/datasets/malaiwah/"
            f"qwen38-27b-fidelity-suite-v5/resolve/{revision}")
    published = []
    for shard in range(10):
        url = f"{base}/suite/shard-{shard:04d}/suite-manifest.json"
        raw = urlopen(url, timeout=120).read()
        published.append((url, raw, json.loads(raw)))

    # Any source cluster used by the locally retained shard0 becomes validation.
    # Untouched test is exclusively contexts from the other shards whose complete
    # source document never appears in validation.
    validation_clusters = {x["source_cluster"] for x in shard0["context_index"]}
    document_meta = {name: meta for name, meta in shard0["documents"]}
    for _, _, manifest in published[1:]:
        document_meta.update({name: meta for name, meta in manifest["documents"]})

    calibration_sources = [
        ("c4.utf8", "2daf1ef16e02dbd337f86296acd4c3f2eb703800ad21fd1f48b40280c699cd41", "prose"),
        ("code.utf8", "12a80d96af7e16b3ecbf2e6899a9c70aee2f4e0003b581979fe6876944e2d891", "code"),
        ("multilingual.utf8", "68182e06506ead0d2676673fdee4bbe6564ccb06cee3b8373b7e4824297ea91e", "multilingual"),
        ("technical.utf8", "7cac2a0ddd7bc4db19e52e3b12ced3812514059f1f11bfc427b7849e0077a2ee", "prose"),
        ("tiny.utf8", "51449c33b579f8b12e776dcbfa58ef47bab69f6ab7f44e972b7d98c4cfcac7fd", "prose"),
        ("wiki.utf8", "d53fe0aeadf3355eb5d32b48bf8691575244c31caccae88ed46be8e1b40209c7", "prose"),
    ]
    expected_calibration_hashes = {name: digest for name, digest, _ in calibration_sources}
    contamination_evidence = []
    for shard, (url, raw, manifest) in enumerate(published):
        scan = manifest["contamination_scan"]
        observed = {Path(x["path"]).name: x["sha256"] for x in scan["calibration_sources"]}
        if observed != expected_calibration_hashes:
            raise AssertionError(f"v5 shard {shard} calibration-source hash set changed")
        if scan["contexts_with_any_hit"] != 0 or scan["total_hits"] != 0:
            raise AssertionError(f"v5 shard {shard} emitted calibration overlap")
        contamination_evidence.append({
            "shard": shard, "manifest_sha256": sha256_bytes(raw),
            "calibration_source_hashes_verified": True,
            "contexts_with_any_hit": scan["contexts_with_any_hit"],
            "total_hits": scan["total_hits"],
        })
    cal_root = Path(args.calibration_root).resolve()
    docs = [{"document_id": f"exl3-standard:{name}", "document_sha256": digest,
             "split": "calibration", "domain": domain, "source_stratum": domain,
             "path": str(cal_root / name), "access": "method_calibration"}
            for name, digest, domain in calibration_sources]
    for doc in docs:
        if not Path(doc["path"]).is_file() or sha256_file(Path(doc["path"])) != doc["document_sha256"]:
            raise AssertionError(f"calibration source missing or changed: {doc['path']}")

    validation_contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    published_shards = []
    for shard, (url, raw, manifest) in enumerate(published):
        published_shards.append({"shard": shard, "manifest_url": url,
                                 "manifest_sha256": sha256_bytes(raw),
                                 "suite_token_sha256": manifest["suite_token_sha256"],
                                 "contexts": manifest["contexts"]})
        for row in manifest["context_index"]:
            item = {"shard": shard, "index": int(row["index"]),
                    "token_url": f"{base}/suite/shard-{shard:04d}/{row['file']}",
                    "token_sha256": row["token_sha256"], "tokens": row["tokens"]}
            if shard == 0:
                validation_contexts[row["source_cluster"]].append(item)
            elif row["source_cluster"] not in validation_clusters:
                test_contexts[row["source_cluster"]].append(item)

    for split, groups in (("validation", validation_contexts),
                          ("untouched_test", test_contexts)):
        for cluster, contexts in sorted(groups.items()):
            meta = document_meta[cluster]
            docs.append({"document_id": cluster, "document_sha256": meta["sha256"],
                         "split": split, "domain": _domain(meta["stratum"]),
                         "source_stratum": meta["stratum"], "source_file": meta["file"],
                         "contexts": contexts,
                         "access": "evaluation_gate_only" if split == "untouched_test"
                         else "validation_replay"})

    # UltraChat contributes real dialogue without mixing train and published test
    # conversations. We use only the `gen` family, avoiding duplicate SFT views.
    dialogue_root = Path(args.dialogue).resolve()
    dialogue_plan = {
        "ultrachat_200k-train_gen-00002-of-00003.arrow": "validation",
        "ultrachat_200k-test_gen.arrow": "untouched_test",
    }
    if dialogue_root:
        for name, split in dialogue_plan.items():
            path = dialogue_root / name
            if not path.is_file():
                raise AssertionError(f"dialogue shard missing: {path}")
            docs.append({"document_id": f"ultrachat:{path.stem}",
                         "document_sha256": sha256_file(path), "split": split,
                         "domain": "dialogue", "source_stratum": "dialogue",
                         "path": str(path), "format": "arrow",
                         "access": "evaluation_gate_only" if split == "untouched_test"
                         else "method_calibration" if split == "calibration"
                         else "validation_replay"})

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_split[doc["split"]].append(doc)
    sets = {k: {d["document_sha256"] for d in v} for k, v in by_split.items()}
    for a, b in (("calibration", "validation"), ("calibration", "untouched_test"),
                 ("validation", "untouched_test")):
        if sets[a] & sets[b]:
            raise AssertionError(f"document hash overlap between {a} and {b}")
    if set(validation_contexts) & set(test_contexts):
        raise AssertionError("source cluster overlap between validation and untouched test")
    summary = {}
    for split, values in by_split.items():
        summary[split] = {"documents": len(values),
                          "contexts": sum(len(d.get("contexts", [])) for d in values),
                          "domains": dict(sorted(Counter(d["domain"] for d in values).items()))}
    split_contracts = {}
    for split in ("calibration", "validation", "untouched_test"):
        projection = {
            "schema": "qwen38-wave5-split-projection/1",
            "selector": {"field": "split", "op": "eq", "value": split},
            "document_sha256": sorted(d["document_sha256"] for d in by_split[split]),
            "context_token_sha256": sorted(
                c["token_sha256"] for d in by_split[split] for c in d.get("contexts", [])),
        }
        split_contracts[split] = {
            "selector": projection["selector"],
            "projection_sha256": sha256_bytes(canonical_bytes(projection)),
            "documents": len(projection["document_sha256"]),
            "contexts": len(projection["context_token_sha256"]),
        }
    leakage_audit = {
        "schema": "qwen38-wave5-leakage-audit/1",
        "document_hash_intersections": {
            "calibration_validation": [],
            "calibration_untouched_test": [],
            "validation_untouched_test": [],
        },
        "validation_test_source_cluster_intersection": [],
        "v5_calibration_contamination_evidence": contamination_evidence,
        "test_contents_opened_by_builder": False,
    }
    leakage_audit_sha256 = sha256_bytes(canonical_bytes(leakage_audit))
    result = {"schema": SCHEMA, "kind": "document-disjoint-three-way-split",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "seed": SPLIT_SEED, "seed_sha256": sha256_bytes(str(SPLIT_SEED).encode()),
              "policy": {"unit": "complete source document/conversation",
                         "calibration": "six pinned EXL3 standard calibration sources",
                         "validation": "published v5 shard0 source clusters plus held-out UltraChat train-gen",
                         "untouched_test": "v5 shards1-9 contexts only from clusters absent in shard0 plus UltraChat test-gen",
                         "activation_row_splitting": False,
                         "untouched_test_access": "fail-closed; method code MUST NOT open test paths",
                         "freeze": "all actions/hyperparameters/seeds before one evaluation-gate open"},
              "published_suite": {
                  "repo": "malaiwah/qwen38-27b-fidelity-suite-v5",
                  "revision": revision,
                  "authoritative_token_sha256":
                      "510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88",
                  "shards": published_shards,
              },
              "retained_validation_manifest": str(source_manifest_path),
              "retained_validation_manifest_sha256": sha256_file(source_manifest_path),
              "summary": summary, "split_contracts": split_contracts,
              "leakage_audit": leakage_audit,
              "leakage_audit_sha256": leakage_audit_sha256,
              "access_guard": {
                  "implementation": "tools.research.wave5.data_capture.select_split_documents",
                  "untouched_test_default": "deny",
                  "explicit_unlock_argument": "allow_untouched_test=True",
              },
              "documents": docs, "software": software_revision(),
              "disjoint_document_hashes_verified": True,
              "disjoint_source_clusters_verified": True,
              "test_contents_opened_by_builder": False}
    result["content_sha256"] = sha256_bytes(canonical_bytes(result))
    atomic_json(Path(args.out), result)
    print(json.dumps({"out": args.out, "summary": summary,
                      "content_sha256": result["content_sha256"]}, indent=2))
    return 0


# Functions below run inside a vLLM worker through collective_rpc. Keep imports local.
def _rpc_runtime_topology(self):
    model = self.model_runner.model
    modules = [(n, type(m).__module__ + "." + type(m).__qualname__)
               for n, m in model.named_modules()]
    return {"model_class": type(model).__module__ + "." + type(model).__qualname__,
            "modules": modules,
            "layer_ids": sorted({int(p) for n, _ in modules for p in n.split(".")
                                 if p.isdigit() and (f"layers.{p}." in n or f"layers.{p}" == n)})}


def _capture_match(name: str) -> bool:
    if not any(f"layers.{d}." in name for d in SELECTED_DEPTHS):
        return False
    endings = ("mlp.gate_up_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
               "linear_attn.in_proj_qkvz", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
               "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.conv1d",
               "linear_attn.out_proj", "self_attn.qkv_proj", "self_attn.q_proj",
               "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "self_attn.gate_proj", "self_attn.output_gate")
    return name.endswith(endings)


def _rpc_install_boundary_hooks(self, max_rows: int):
    import torch
    model = self.model_runner.model
    store = {"captures": {}, "max_rows": int(max_rows), "handles": []}

    def first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                got = first_tensor(item)
                if got is not None:
                    return got
        return None

    def make_hook(name):
        def hook(_module, inputs, output):
            x = first_tensor(inputs)
            y = first_tensor(output)
            rec = store["captures"].setdefault(name, {})
            for key, tensor in (("input", x), ("output", y)):
                if tensor is None or tensor.dim() < 2:
                    continue
                flat = tensor.detach().reshape(-1, tensor.shape[-1])
                take = min(store["max_rows"], flat.shape[0])
                rec[key] = flat[:take].to("cpu", torch.bfloat16, copy=True)
                rec[key + "_source_shape"] = list(tensor.shape)
            return output
        return hook

    matched = []
    for name, module in model.named_modules():
        if _capture_match(name):
            store["handles"].append(module.register_forward_hook(make_hook(name)))
            matched.append(name)
    if not matched:
        raise RuntimeError("no Wave5 module boundaries matched runtime symbols")
    self._wave5_boundary_store = store
    return matched


def _rpc_pop_boundaries(self):
    store = getattr(self, "_wave5_boundary_store", None)
    if store is None:
        return None
    captures = store["captures"]
    store["captures"] = {}
    return captures


def _runtime_llm(args: argparse.Namespace):
    from vllm import LLM
    kwargs = dict(model=args.model, trust_remote_code=True, tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  kv_cache_memory_bytes=256 * 1024 * 1024, dtype="bfloat16",
                  kv_cache_dtype=args.kv_cache_dtype, load_format="safetensors",
                  max_model_len=args.max_model_len, max_num_batched_tokens=args.max_model_len,
                  max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
                  enforce_eager=True)
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        kwargs["quantization"] = args.quantization
        kwargs["quantization_config"] = {
            "linear": {"weight": "mxfp8"},
            "ignore": ["re:.*visual\\..*", "re:.*in_proj_a$", "re:.*in_proj_b$",
                       "re:.*mtp\\..*", "lm_head"],
        }
    if args.cpu_offload_gb:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    return LLM(**kwargs)


def cmd_runtime_topology(args: argparse.Namespace) -> int:
    llm = _runtime_llm(args)
    reports = llm.collective_rpc(_rpc_runtime_topology)
    result = reports[0]
    if result["layer_ids"] != list(range(64)):
        raise AssertionError(f"runtime layer symbols are not exactly 0..63: {result['layer_ids']}")
    result.update({"schema": SCHEMA, "kind": "runtime-topology", "runtime_64_layers_verified": True,
                   "model": args.model, "software": software_revision()})
    result["content_sha256"] = sha256_bytes(canonical_bytes(result))
    atomic_json(Path(args.out), result)
    print(json.dumps({k: result[k] for k in ("model_class", "layer_ids", "content_sha256")}, indent=2))
    return 0


def _finite_summary(array: np.ndarray) -> dict[str, Any]:
    x = array.astype(np.float32, copy=False)
    return {"shape": list(x.shape), "finite": bool(np.isfinite(x).all()),
            "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "rms": float(np.sqrt(np.mean(x * x, dtype=np.float64)))}


def write_activation_bundle(captures: dict[str, Any], out: Path, identity: dict[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for ordinal, (name, rec) in enumerate(sorted(captures.items())):
        tensors = {k: v.contiguous() for k, v in rec.items() if torch.is_tensor(v)}
        if not tensors:
            continue
        filename = f"activation-{ordinal:03d}.safetensors"
        path = out / filename
        save_file(tensors, str(path))
        stats = {k: _finite_summary(v.float().numpy()) for k, v in tensors.items()}
        if not all(s["finite"] for s in stats.values()):
            raise AssertionError(f"non-finite activation at {name}")
        records.append({"module": name, "file": filename, "sha256": sha256_file(path),
                        "source_shapes": {k: v for k, v in rec.items() if not torch.is_tensor(v)},
                        "tensors": stats})
    manifest = {"schema": SCHEMA, "kind": "real-module-boundary-activations",
                "identity": identity, "records": records, "record_count": len(records),
                "software": software_revision()}
    manifest["content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(out / "capture-manifest.json", manifest)
    return manifest


def cmd_capture_vllm(args: argparse.Namespace) -> int:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    llm = _runtime_llm(args)
    topology = llm.collective_rpc(_rpc_runtime_topology)[0]
    if topology["layer_ids"] != list(range(64)):
        raise AssertionError("runtime topology did not expose exactly 64 transformer layers")
    hooked = llm.collective_rpc(_rpc_install_boundary_hooks, args=(args.max_rows,))[0]
    token_path = Path(args.tokens).absolute()
    ids = json.loads(token_path.read_text())[:args.max_model_len - 1]
    if len(ids) < 2:
        raise ValueError("capture document has fewer than two tokens")
    llm.collective_rpc(_rpc_pop_boundaries)
    llm.generate([TokensPrompt(prompt_token_ids=ids)],
                 sampling_params=SamplingParams(max_tokens=1, temperature=0, detokenize=False),
                 use_tqdm=False)
    captures = llm.collective_rpc(_rpc_pop_boundaries)[0]
    manifest = write_activation_bundle(captures, Path(args.out), {
        "model": str(Path(args.model).resolve()), "flow": args.flow,
        "quantization": args.quantization, "kv_cache_dtype": args.kv_cache_dtype,
        "token_path": str(token_path), "token_sha256": sha256_file(token_path),
        "tokens_presented": len(ids), "runtime_model_class": topology["model_class"],
        "runtime_64_layers_verified": True, "hooked_modules": hooked})
    print(json.dumps({"out": args.out, "records": manifest["record_count"],
                      "content_sha256": manifest["content_sha256"]}, indent=2))
    return 0


def _load_activation(path: Path, key: str) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key).float().numpy()


def cmd_stats(args: argparse.Namespace) -> int:
    capture = Path(args.capture).resolve()
    cap = json.loads((capture / "capture-manifest.json").read_text())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for rec in cap["records"]:
        path = capture / rec["file"]
        x = _load_activation(path, "input")
        y = _load_activation(path, "output")
        n = x.shape[0]
        if n == 0:
            continue
        x_diag = np.mean(x * x, axis=0, dtype=np.float64).astype(np.float32)
        y_center = y - y.mean(0, keepdims=True)
        y_cov_diag = np.mean(y_center * y_center, axis=0, dtype=np.float64).astype(np.float32)
        # Exact dense 16x16 Fisher/covariance blocks at preregistered first/middle/last
        # and seeded positions. Full 5120^2 H_X is intentionally not duplicated per
        # module; blocks plus diagonal are sufficient shared statistics, while raw X
        # is retained for any future full-H computation.
        starts = sorted({0, max(0, x.shape[1] // 2 - BLOCK16 // 2),
                         max(0, x.shape[1] - BLOCK16),
                         random.Random(SCREEN_SEED ^ len(records)).randrange(max(1, x.shape[1] - BLOCK16 + 1))})
        tensors = {"h_x_diag": x_diag, "output_cov_diag": y_cov_diag}
        block_meta = []
        for i, start in enumerate(starts):
            xb = x[:, start:start + BLOCK16]
            h = (xb.T @ xb / n).astype(np.float32)
            key = f"h_x_block16_{i}"
            tensors[key] = h
            block_meta.append({"key": key, "start": start, "shape": list(h.shape),
                               "sha256": sha256_bytes(h.tobytes())})
        stats_path = out / f"stats-{len(records):03d}.npz"
        np.savez(stats_path, **tensors)
        records.append({"module": rec["module"], "activation_file": rec["file"],
                        "activation_sha256": rec["sha256"], "sample_count": n,
                        "h_x_definition": "X^T X / N (uncentered second moment)",
                        "output_cov_definition": "diag((Y-mean Y)^T(Y-mean Y)/N)",
                        "stats_file": stats_path.name, "stats_sha256": sha256_file(stats_path),
                        "h_x_diag_sha256": sha256_bytes(x_diag.tobytes()),
                        "output_cov_diag_sha256": sha256_bytes(y_cov_diag.tobytes()),
                        "block16": block_meta, "finite": bool(all(np.isfinite(v).all() for v in tensors.values()))})
    manifest = {"schema": SCHEMA, "kind": "real-activation-curvature-statistics",
                "capture_manifest": str(capture / "capture-manifest.json"),
                "capture_manifest_sha256": sha256_file(capture / "capture-manifest.json"),
                "records": records, "record_count": len(records), "software": software_revision()}
    manifest["content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(out / "stats-manifest.json", manifest)
    print(json.dumps({"out": str(out), "records": len(records),
                      "content_sha256": manifest["content_sha256"]}, indent=2))
    return 0

def cmd_dense_hx(args: argparse.Namespace) -> int:
    """Materialize exact dense H_X for selected L0/L55 gate/down boundaries."""
    capture = Path(args.capture).resolve()
    cap = json.loads((capture / "capture-manifest.json").read_text())
    wanted = ("layers.0.mlp.gate_up_proj", "layers.0.mlp.down_proj",
              "layers.55.mlp.gate_up_proj", "layers.55.mlp.down_proj")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for rec in cap["records"]:
        if not rec["module"].endswith(wanted):
            continue
        x = _load_activation(capture / rec["file"], "input")
        h_x = (x.T @ x / x.shape[0]).astype(np.float32)
        if not np.isfinite(h_x).all() or not np.array_equal(h_x, h_x.T):
            raise AssertionError(f"invalid dense H_X at {rec['module']}")
        path = out / f"h-x-{len(records):02d}.npy"
        np.save(path, h_x, allow_pickle=False)
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        if loaded.shape != h_x.shape:
            raise AssertionError("dense H_X round-trip shape mismatch")
        records.append({"module": rec["module"], "sample_count": int(x.shape[0]),
                        "definition": "X^T X / N", "shape": list(h_x.shape),
                        "file": path.name, "sha256": sha256_file(path),
                        "finite": True, "symmetric_bit_exact": True})
    if len(records) != 4:
        raise AssertionError(f"expected four selected dense H_X matrices, got {len(records)}")
    manifest = {"schema": SCHEMA, "kind": "selected-dense-h-x",
                "capture_manifest_sha256": sha256_file(capture / "capture-manifest.json"),
                "records": records, "software": software_revision()}
    manifest["content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(out / "h-x-manifest.json", manifest)
    print(json.dumps({"records": len(records), "content_sha256": manifest["content_sha256"]}, indent=2))
    return 0


def fisher_roundtrip(x: np.ndarray, grad_output: np.ndarray, out: Path) -> dict[str, Any]:
    """Empirical Fisher diagonal for a Linear weight from a backward hook.

    Per sample gradient is outer(grad_output, input); E[g^2] therefore equals
    E[grad_output^2]^T @ input^2 without constructing per-sample full gradients.
    """
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError("input/grad_output sample mismatch")
    n = x.shape[0]
    fisher = (grad_output.astype(np.float64).T ** 2 @
              (x.astype(np.float64) ** 2) / n).astype(np.float32)
    np.save(out, fisher, allow_pickle=False)
    loaded = np.load(out, allow_pickle=False)
    if not np.array_equal(fisher.view(np.uint32), loaded.view(np.uint32)):
        raise AssertionError("Fisher diagonal round-trip is not bit-exact")
    return {"definition": "mean_s (d sampled-token NLL_s / d W)^2",
            "sample_count": n, "shape": list(fisher.shape), "finite": bool(np.isfinite(fisher).all()),
            "sha256": sha256_file(out), "roundtrip_bit_exact": True}


def cmd_fisher_roundtrip(args: argparse.Namespace) -> int:
    x = np.load(args.input, allow_pickle=False)
    g = np.load(args.grad_output, allow_pickle=False)
    result = fisher_roundtrip(x, g, Path(args.out))
    atomic_json(Path(args.manifest), {"schema": SCHEMA, "kind": "empirical-fisher-diagonal",
                                             "input": args.input, "input_sha256": sha256_file(Path(args.input)),
                                             "grad_output": args.grad_output,
                                             "grad_output_sha256": sha256_file(Path(args.grad_output)), **result})
    print(json.dumps(result, indent=2))
    return 0

def cmd_fisher_lm_head(args: argparse.Namespace) -> int:
    """Smoke the exact sampled-token NLL/backward-hook/Fisher round-trip."""
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open

    device = torch.device(args.device)
    with safe_open(args.hidden, framework="pt", device="cpu") as f:
        hidden_key = next(iter(f.keys()))
        hidden = f.get_tensor(hidden_key)[:args.samples].float()
    with safe_open(args.head, framework="pt", device="cpu") as f:
        head_key = next(iter(f.keys()))
        weight = f.get_tensor(head_key)
    ids = json.loads(Path(args.tokens).read_text())
    targets = torch.tensor(ids[1:1 + hidden.shape[0]], dtype=torch.long, device=device)
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                             device=device, dtype=torch.bfloat16)
    linear.weight.requires_grad_(False)
    with torch.no_grad():
        linear.weight.copy_(weight.to(device))
    captured: dict[str, torch.Tensor] = {}

    def backward_hook(_module, _grad_input, grad_output):
        captured["grad_output"] = grad_output[0].detach().float().cpu()

    handle = linear.register_full_backward_hook(backward_hook)
    x = hidden.to(device, torch.bfloat16).requires_grad_(True)
    logits = linear(x).float()
    losses = functional.cross_entropy(logits, targets, reduction="none")
    losses.sum().backward()
    handle.remove()
    grad = captured["grad_output"]
    # First/middle/last plus actual next-token rows; this is an affordable,
    # preregistered block of the full empirical Fisher diagonal.
    rows = sorted({0, weight.shape[0] // 2, weight.shape[0] - 1,
                   *targets.detach().cpu().tolist()})[:args.rows]
    x2 = hidden.double().square()
    g2 = grad[:, rows].double().square()
    fisher = (g2.T @ x2 / hidden.shape[0]).float().numpy()
    out = Path(args.out)
    np.savez(out, fisher_diag=fisher, output_rows=np.asarray(rows, dtype=np.int64))
    with np.load(out, allow_pickle=False) as loaded:
        roundtrip = np.array_equal(fisher.view(np.uint32),
                                   loaded["fisher_diag"].view(np.uint32))
    if not roundtrip or not np.isfinite(fisher).all():
        raise AssertionError("sampled-NLL Fisher round-trip failed")
    manifest = {
        "schema": SCHEMA, "kind": "empirical-fisher-diagonal-block",
        "definition": "mean_s (d full-vocabulary sampled-token NLL_s / d W)^2",
        "teacher_candidate_direction": "teacher token NLL; not called KLD",
        "parameters_frozen": True, "backward_hook": True,
        "hidden": args.hidden, "hidden_sha256": sha256_file(Path(args.hidden)),
        "head": args.head, "head_sha256": sha256_file(Path(args.head)),
        "tokens": args.tokens, "token_sha256": sha256_file(Path(args.tokens)),
        "sample_count": int(hidden.shape[0]), "vocab_size": int(weight.shape[0]),
        "hidden_size": int(weight.shape[1]), "selected_output_rows": rows,
        "fisher_shape": list(fisher.shape), "finite": True,
        "roundtrip_bit_exact": True, "artifact": str(out),
        "artifact_sha256": sha256_file(out), "software": software_revision(),
    }
    manifest["content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(Path(args.manifest), manifest)
    print(json.dumps({"loss_mean": float(losses.mean()), "fisher_shape": list(fisher.shape),
                      "artifact_sha256": manifest["artifact_sha256"],
                      "content_sha256": manifest["content_sha256"]}, indent=2))
    return 0



def cmd_fisher_transformers(args: argparse.Namespace) -> int:
    """Capture selected real-model empirical Fisher diagonals with backward hooks."""
    import torch
    from numpy.lib.format import open_memmap
    from transformers import AutoModelForImageTextToText

    all_cpu = args.gpu_memory == "0GiB"
    max_memory = {"cpu": args.cpu_memory} if all_cpu else {
        0: args.gpu_memory, "cpu": args.cpu_memory}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map={"": "cpu"} if all_cpu else "auto", max_memory=max_memory,
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    suffixes = (
        "layers.0.mlp.gate_proj", "layers.0.mlp.down_proj",
        "layers.55.mlp.gate_proj", "layers.55.mlp.down_proj",
        "layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.out_proj",
        "layers.55.self_attn.q_proj", "layers.55.self_attn.o_proj",
    )
    captures: dict[str, dict[str, torch.Tensor]] = {}
    handles = []

    def make_forward(name):
        def hook(_module, inputs, _output):
            tensor = next((x for x in inputs if torch.is_tensor(x)), None)
            if tensor is not None:
                captures.setdefault(name, {})["input"] = tensor.detach().reshape(
                    -1, tensor.shape[-1]).float().cpu()
        return hook

    def make_backward(name):
        def hook(_module, _grad_input, grad_output):
            tensor = next((x for x in grad_output if torch.is_tensor(x)), None)
            if tensor is not None:
                captures.setdefault(name, {})["grad_output"] = tensor.detach().reshape(
                    -1, tensor.shape[-1]).float().cpu()
        return hook

    matched = []
    for name, module in model.named_modules():
        if name.endswith(suffixes):
            handles.append(module.register_forward_hook(make_forward(name)))
            handles.append(module.register_full_backward_hook(make_backward(name)))
            matched.append(name)
    if len(matched) != len(suffixes):
        raise AssertionError(f"selected Fisher modules missing: matched={matched}")
    ids = torch.tensor(json.loads(Path(args.tokens).read_text())[:args.samples],
                       dtype=torch.long)
    embedding = model.get_input_embeddings()
    embedding_device = next(embedding.parameters()).device
    inputs_embeds = embedding(ids.to(embedding_device).unsqueeze(0)).detach().requires_grad_(True)
    labels = ids.unsqueeze(0).to(inputs_embeds.device)
    output = model(inputs_embeds=inputs_embeds, labels=labels, use_cache=False)
    output.loss.backward()
    for handle in handles:
        handle.remove()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for ordinal, name in enumerate(sorted(captures)):
        rec = captures[name]
        x = rec["input"].numpy()
        grad = rec["grad_output"].numpy()
        if x.shape[0] != grad.shape[0]:
            raise AssertionError(f"Fisher sample mismatch at {name}: {x.shape} vs {grad.shape}")
        token_positions = x.shape[0]
        path = out / f"fisher-{ordinal:02d}.npy"
        fisher = open_memmap(path, mode="w+", dtype=np.float32,
                             shape=(grad.shape[1], x.shape[1]))
        x64 = x.astype(np.float64)
        for start in range(0, grad.shape[1], 256):
            stop = min(start + 256, grad.shape[1])
            weight_grad = grad[:, start:stop].astype(np.float64).T @ x64
            fisher[start:stop] = (weight_grad ** 2).astype(np.float32)
        fisher.flush()
        del fisher
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        finite = bool(np.isfinite(loaded).all())
        block_starts = (0, max(0, x.shape[1] // 2 - 8), max(0, x.shape[1] - 16))
        dense_blocks = {}
        for block_start in block_starts:
            xb = x[:, block_start:block_start + 16].astype(np.float64)
            weight_grad_block = grad[:, 0].astype(np.float64) @ xb
            block = np.outer(weight_grad_block, weight_grad_block).astype(np.float32)
            dense_blocks[str(block_start)] = block
        block_path = out / f"fisher-block16-{ordinal:02d}.npz"
        np.savez(block_path, **dense_blocks)
        records.append({"module": name, "sample_count": 1,
                        "token_positions": token_positions,
                        "scored_token_count": max(0, token_positions - 1),
                        "input_shape": list(x.shape), "grad_output_shape": list(grad.shape),
                        "fisher_shape": list(loaded.shape), "finite": finite,
                        "fisher_file": path.name, "fisher_sha256": sha256_file(path),
                        "block16_coordinates": [
                            {"output_row": 0, "input_start": int(block_start),
                             "input_end_exclusive": int(block_start + 16)}
                            for block_start in block_starts
                        ],
                        "block16_file": block_path.name,
                        "block16_sha256": sha256_file(block_path)})
    manifest = {"schema": SCHEMA, "kind": "selected-real-model-empirical-fisher",
                "definition": "(d mean causal-token NLL for one real sequence / d W)^2",
                "fisher_sample_unit": "one token sequence",
                "cross_position_terms_included": True,
                "loss_reduction": "mean over shifted causal-token losses",
                "sequence_count": 1, "scored_token_count": max(0, args.samples - 1),
                "parameters_frozen": True, "backward_hooks": True,
                "model": args.model, "tokens": args.tokens,
                "token_sha256": sha256_file(Path(args.tokens)),
                "loss": float(output.loss.detach().cpu()), "records": records,
                "record_count": len(records), "software": software_revision()}
    manifest["content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(out / "fisher-manifest.json", manifest)
    print(json.dumps({"loss": manifest["loss"], "records": len(records),
                      "content_sha256": manifest["content_sha256"]}, indent=2))
    return 0

def cmd_repair_fisher_manifest(args: argparse.Namespace) -> int:
    """Rebind coordinate metadata without recomputing already-hashed tensors."""
    root = Path(args.fisher).resolve()
    path = root / "fisher-manifest.json"
    manifest = json.loads(path.read_text())
    for rec in manifest["records"]:
        input_width = int(rec["input_shape"][1])
        starts = (0, max(0, input_width // 2 - 8), max(0, input_width - 16))
        block_path = root / rec["block16_file"]
        with np.load(block_path, allow_pickle=False) as blocks:
            if set(blocks.files) != {str(x) for x in starts}:
                raise AssertionError(f"block coordinate mismatch: {rec['module']}")
        if sha256_file(root / rec["fisher_file"]) != rec["fisher_sha256"]:
            raise AssertionError(f"Fisher digest mismatch: {rec['module']}")
        if sha256_file(block_path) != rec["block16_sha256"]:
            raise AssertionError(f"block digest mismatch: {rec['module']}")
        rec["block16_coordinates"] = [
            {"output_row": 0, "input_start": int(start),
             "input_end_exclusive": int(start + 16)}
            for start in starts
        ]
    manifest["software"] = software_revision()
    manifest.pop("content_sha256", None)
    manifest["content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(path, manifest)
    print(json.dumps({"records": len(manifest["records"]),
                      "content_sha256": manifest["content_sha256"]}, indent=2))
    return 0

def cmd_self_test(args: argparse.Namespace) -> int:
    # Fixed patterns include normals, subnormals, signed zero, infinities, and NaN.
    bits = np.array([0x0000, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x7F7F, 0x7F80, 0xFF80, 0x7FC1], dtype="<u2")
    got = bf16_payload_to_float32(bits)
    assert got.view(np.uint32).tolist() == [int(v) << 16 for v in bits]
    assert not np.array_equal(got.view(np.uint32), bits.view(np.float16).astype(np.float32).view(np.uint32))
    screen, promotion = block_coordinates([5120, 17408], SCREEN_SEED)
    assert len(screen) == 8 and len(promotion) == 20
    print(json.dumps({"bf16_shift_decode": "pass", "bf16_as_fp16_rejected": True,
                      "screening_blocks": len(screen), "promotion_blocks": len(promotion)}))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census")
    c.add_argument("--model", required=True); c.add_argument("--out", required=True); c.set_defaults(func=cmd_census)
    s = sub.add_parser("split")
    s.add_argument("--suite", required=True); s.add_argument("--dialogue", required=True)
    s.add_argument("--calibration-root", required=True)
    s.add_argument("--out", required=True); s.set_defaults(func=cmd_split)
    for name, func in (("runtime-topology", cmd_runtime_topology), ("capture-vllm", cmd_capture_vllm)):
        r = sub.add_parser(name); r.add_argument("--model", required=True); r.add_argument("--out", required=True)
        r.add_argument("--quantization", default="auto"); r.add_argument("--kv-cache-dtype", default="auto")
        r.add_argument("--gpu-memory-utilization", type=float, default=0.90)
        r.add_argument("--cpu-offload-gb", type=float, default=0); r.add_argument("--max-model-len", type=int, default=128)
        if name == "capture-vllm":
            r.add_argument("--tokens", required=True); r.add_argument("--flow", required=True,
                choices=("bf16_reference", "running_quant")); r.add_argument("--max-rows", type=int, default=64)
        r.set_defaults(func=func)
    st = sub.add_parser("stats"); st.add_argument("--capture", required=True); st.add_argument("--out", required=True); st.set_defaults(func=cmd_stats)
    dh = sub.add_parser("dense-hx"); dh.add_argument("--capture", required=True)
    dh.add_argument("--out", required=True); dh.set_defaults(func=cmd_dense_hx)
    f = sub.add_parser("fisher-roundtrip"); f.add_argument("--input", required=True); f.add_argument("--grad-output", required=True)
    f.add_argument("--out", required=True); f.add_argument("--manifest", required=True); f.set_defaults(func=cmd_fisher_roundtrip)
    fh = sub.add_parser("fisher-lm-head")
    fh.add_argument("--hidden", required=True); fh.add_argument("--head", required=True)
    fh.add_argument("--tokens", required=True); fh.add_argument("--out", required=True)
    fh.add_argument("--manifest", required=True); fh.add_argument("--device", default="cuda")
    fh.add_argument("--samples", type=int, default=63); fh.add_argument("--rows", type=int, default=16)
    fh.set_defaults(func=cmd_fisher_lm_head)
    ft = sub.add_parser("fisher-transformers")
    ft.add_argument("--model", required=True); ft.add_argument("--tokens", required=True)
    ft.add_argument("--out", required=True); ft.add_argument("--samples", type=int, default=16)
    ft.add_argument("--gpu-memory", default="30GiB"); ft.add_argument("--cpu-memory", default="96GiB")
    ft.set_defaults(func=cmd_fisher_transformers)
    rf = sub.add_parser("repair-fisher-manifest")
    rf.add_argument("--fisher", required=True); rf.set_defaults(func=cmd_repair_fisher_manifest)
    t = sub.add_parser("self-test"); t.set_defaults(func=cmd_self_test)
    return p


def main() -> int:
    return int(parser().parse_args().func(parser().parse_args()))


if __name__ == "__main__":
    # Parse once (kept explicit so worker-side imports have no argument side effects).
    args = parser().parse_args()
    raise SystemExit(args.func(args))
