#!/usr/bin/env python3
"""Quantize the BF16 embedding table to free VRAM for context.

The AtomicChat GGUF comparison found that the embedding table "is a lookup whose
error stays inside one token, so it can be cut hard" while the output head "decides
the next word directly and has to stay precise." Our checkpoint is exactly backwards:
lm_head IS trellis-quantized but embed_tokens is [248320, 5120] BF16 = 2.368 GiB,
absent from the 409-module trellis ladder entirely.

This script creates a modified checkpoint with the embedding table converted to
FP8 (E4M3), halving it from 2.37 GiB to 1.18 GiB. At the measured 40.7 KB/token
KV cost, that frees ~1.18 GiB = ~30,500 tokens of context.

The KLD cost must be measured: AtomicChat says it's small, but our suite scores
the full vocabulary distribution through a shared BF16 LM head, so an FP8
embedding could perturb the input to every layer. This script only creates the
checkpoint; the KLD measurement needs the model served (GPU).

Usage:
    python3 tools/quantize-embedding.py \
        --source /path/to/K5K6-hydrated \
        --output /path/to/K5K6-fp8-embed \
        --dtype fp8
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file


def quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP8 E4M3 quantization with per-row scale.

    Returns (quantized_weight [uint8], scales [fp32]) packed for vLLM's
    compressed-tensors loader. The scale is per-row (per-token-id) because
    embedding rows have very different magnitudes.
    """
    # Per-row max
    row_max = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    # FP8 E4M3 max value is 448.0
    fp8_max = 448.0
    scale = row_max / fp8_max
    # Quantize
    scaled = weight / scale
    # Round to nearest representable FP8 E4M3 value
    # (torch doesn't have native FP8 cast on CPU, so simulate)
    # Cast to float16 then to our simulated FP8
    quantized = scaled.to(torch.float32)
    # Simplest approach: use torch's native fp8 if available, else uint8 approximation
    if hasattr(torch, 'float8_e4m3fn'):
        q = quantized.to(torch.float8_e4m3fn).to(torch.uint8)
    else:
        # Fallback: scale to [-127, 127] as int8 (not true FP8 but same size)
        q = (quantized.clamp(-128, 127)).round().to(torch.int8).to(torch.uint8)
    return q, scale.to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to the source checkpoint")
    ap.add_argument("--output", required=True, help="Path to write the modified checkpoint")
    ap.add_argument("--dtype", choices=["fp8", "int8"], default="fp8",
                    help="Quantization dtype for the embedding table")
    a = ap.parse_args()

    src = Path(a.source)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    # Find the safetensors index
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        print(f"ERROR: no model.safetensors.index.json in {src}")
        return

    index = json.loads(idx_path.read_text())
    weight_map = index["weight_map"]

    # Find which shard has the embedding
    embed_key = "model.language_model.embed_tokens.weight"
    if embed_key not in weight_map:
        print(f"ERROR: {embed_key} not found in weight map")
        return

    embed_shard = weight_map[embed_key]
    print(f"Embedding table is in shard: {embed_shard}")

    # Copy all files except the shard containing the embedding
    for f in src.iterdir():
        if f.name == embed_shard or not f.is_file():
            continue
        dest = out / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    # Load the embedding from its shard
    shard_path = src / embed_shard
    print(f"Loading embedding from {shard_path}...")
    with safe_open(str(shard_path), framework="pt") as f:
        keys = list(f.keys())
        tensors = {}
        for k in keys:
            tensors[k] = f.get_tensor(k)

    embed = tensors[embed_key]
    print(f"Original embedding: shape={list(embed.shape)}, dtype={embed.dtype}, "
          f"size={embed.numel() * embed.element_size() / 1024**3:.3f} GiB")

    # Quantize
    if a.dtype == "fp8":
        q_embed, scales = quantize_fp8(embed)
        new_size = q_embed.numel() * q_embed.element_size() + scales.numel() * scales.element_size()
        print(f"Quantized embedding: shape={list(q_embed.shape)}, dtype={q_embed.dtype}, "
              f"scales={list(scales.shape)}, total={new_size / 1024**3:.3f} GiB")
        tensors[embed_key] = q_embed
        tensors[embed_key + ".scale"] = scales
    else:
        # INT8 per-row
        row_max = embed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = row_max / 127.0
        q_embed = (embed / scale).round().clamp(-128, 127).to(torch.int8)
        tensors[embed_key] = q_embed
        tensors[embed_key + ".scale"] = scale.to(torch.float32)

    # Save the modified shard
    out_shard = out / embed_shard
    save_file(tensors, str(out_shard), metadata={"format": "pt"})
    print(f"Saved modified shard to {out_shard}")

    # Update the index
    new_index = dict(index)
    if a.dtype == "fp8":
        new_index["weight_map"][embed_key + ".scale"] = embed_shard
    new_index["weight_map"][embed_key] = embed_shard
    # Update metadata if present
    if "metadata" in new_index:
        new_index["metadata"]["total_size"] = sum(
            t.numel() * t.element_size() for t in tensors.values()
        ) + sum(
            (out / s).stat().st_size for s in set(index["weight_map"].values()) if s != embed_shard
        )
    (out / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=1))

    freed = embed.numel() * embed.element_size() - new_size
    print(f"\nFreed: {freed / 1024**3:.3f} GiB")
    print(f"At 40.7 KB/token KV cost, that's ~{int(freed / 1024**3 * 1024**3 / 40700):,} tokens of context")
    print(f"\nNOTE: KLD cost not measured. Serve this checkpoint and run the fidelity")
    print(f"harness to measure the impact. The embedding feeds every layer, so the")
    print(f"KLD may be higher than AtomicChat's finding (they measure output-token")
    print(f"agreement, not full-vocabulary KL divergence).")


if __name__ == "__main__":
    main()
