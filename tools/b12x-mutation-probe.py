"""Does b12x prepare_weight mutate the trellis payload in place?

If it does, any other consumer of the same tensor (ext.reconstruct, the FP4/FP6
converters, the fp16 prefill cache) silently reads repacked bytes afterwards.
That would explain a model that produces plausible-shaped garbage while every
individual GEMM comparison passes.

Checks, for one K5 and one K6 matrix:
  1. byte checksum of trellis (+ suh/svh) before and after prepare_weight
  2. ext.reconstruct output before vs after prepare_weight (cos/max_rel)
"""
import importlib
import json
import os
import sys

import torch

sys.path.insert(0, "/opt/exllamav3")
sys.path.insert(0, "/opt/fp4")
from safetensors import safe_open  # noqa: E402

SNAP = os.environ["SNAP"]
DEV = "cuda:0"
ext = importlib.import_module("exllamav3_ext")
IDX = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]


def load(prefix):
    out = {}
    for suffix in ("trellis", "suh", "svh", "mcg"):
        key = f"{prefix}.{suffix}"
        if key not in IDX:
            continue
        with safe_open(f"{SNAP}/{IDX[key]}", framework="pt", device=DEV) as f:
            out[suffix] = f.get_tensor(key)
    return out


def csum(t):
    return int(t.detach().view(torch.uint8 if t.dtype == torch.int16 else t.dtype)
               .reshape(-1)[: 1 << 22].to(torch.float64).abs().sum().item() * 1000)


def recon(w):
    k16, n16 = w["trellis"].shape[0], w["trellis"].shape[1]
    W = torch.zeros((k16 * 16, n16 * 16), dtype=torch.float16, device=DEV)
    ext.reconstruct(W, w["trellis"], int(w["trellis"].shape[2]) // 16,
                    w.get("mcg") is not None, False)
    return W


for name, prefix in (
    ("K6 down_proj", "model.language_model.layers.3.mlp.down_proj"),
    ("K5 gate_proj", "model.language_model.layers.3.mlp.gate_proj"),
):
    w = load(prefix)
    bits = int(w["trellis"].shape[2]) // 16
    before_sums = {k: csum(v) for k, v in w.items()}
    W_before = recon(w).clone()

    from b12x.gemm.trellis_linear import api
    pw = api.prepare_weight(w["trellis"], w["suh"], w["svh"],
                            codebook="mcg", params_dtype=torch.float16)
    torch.cuda.synchronize()

    after_sums = {k: csum(v) for k, v in w.items()}
    changed = [k for k in before_sums if before_sums[k] != after_sums[k]]

    W_after = recon(w)
    cos = torch.nn.functional.cosine_similarity(
        W_before.float().flatten(), W_after.float().flatten(), dim=0).item()
    rel = ((W_after.float() - W_before.float()).abs().amax()
           / W_before.float().abs().amax().clamp_min(1e-6)).item()

    print(f"{name} (bits={bits}):")
    print(f"  tensors mutated by prepare_weight: {changed if changed else 'NONE'}")
    print(f"  reconstruct before vs after: cos={cos:.6f} max_rel={rel:.4e} "
          f"{'*** RECONSTRUCT CORRUPTED ***' if cos < 0.999 else 'identical'}")
    del pw
