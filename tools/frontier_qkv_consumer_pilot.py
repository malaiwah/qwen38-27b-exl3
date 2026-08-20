#!/usr/bin/env python3
"""Validate actual producer metadata and payload keys through the QKV consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from vllm.model_executor.layers.quantization.exl3 import Exl3Config  # type: ignore[import-not-found]


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    producer = json.loads(args.producer_evidence.read_text(encoding="utf-8"))
    if producer.get("status") != "pass" or producer.get("payload_exclusive") is not True:
        raise SystemExit("producer evidence is not a passing exclusive payload")
    layer = producer["layer"]
    fused_keys = producer["fused_keys"]
    expected_keys = [f"{layer}.qkv_proj.{suffix}" for suffix in ("mcg", "suh", "svh", "trellis")]
    if fused_keys != expected_keys or producer.get("split_keys") != []:
        raise SystemExit("producer payload keys are not canonical fused-only keys")
    topology = {
        "schema": "exl3_qkv_topology/1",
        "layers": [{
            "layer": layer,
            "variant": "fused_uniform",
            "components": ["q_proj", "k_proj", "v_proj"],
            "output_splits": producer["output_splits"],
            "projection": {"name": "qkv_proj", "K": producer["K"], "codebook": producer["codebook"], "scale": "always"},
        }],
    }
    key = f"{layer}.qkv_proj"
    storage = {
        key: {
            "quant_format": "exl3",
            "bits_per_weight": producer["K"],
            "codebook": producer["codebook"],
            "scale": "always",
            "stored_tensors": {name: {} for name in fused_keys},
        }
    }
    config = Exl3Config(tensor_storage=storage, exl3_qkv_topology=topology)
    config.maybe_update_config(
        "unused",
        SimpleNamespace(layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]),
    )
    route = config.qkv_topology_for_prefix("language_model.model.layers.3.self_attn.qkv_proj")
    if route is None or route.get("variant") != "fused_uniform" or route.get("output_splits") != [12288, 1024, 1024]:
        raise SystemExit("consumer did not resolve the actual fused producer route")
    value = {
        "schema": "qwen38-frontier-qkv-consumer-evidence/1",
        "status": "pass",
        "producer_evidence_sha256": hashlib.sha256(canonical(producer)).hexdigest(),
        "consumer_commit": "5f167ce8dfadd91310142e5aadffc8101a14382c",
        "layer": layer,
        "runtime_alias": "language_model.model.layers.3.self_attn.qkv_proj",
        "variant": route["variant"],
        "output_splits": route["output_splits"],
        "payload_exclusive": True,
        "registry_rows": config.qkv_registry_rows,
    }
    payload = canonical(value) + b"\n"
    tmp = args.output.with_name(args.output.name + ".tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, args.output)


if __name__ == "__main__":
    main()
