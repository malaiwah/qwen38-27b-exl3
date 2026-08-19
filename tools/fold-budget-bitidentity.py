"""Is hadamard_fold_weight_chunked bit-identical across FP32 chunk budgets?

The fold operates on independent 128x128 blocks, so grouping should not change
any value - but PP tuning is about to change the default budget, so verify
instead of asserting.
"""
import importlib, json, os, sys
import torch
sys.path.insert(0, "/opt/exllamav3"); sys.path.insert(0, "/opt/fp4")
from safetensors import safe_open
ext = importlib.import_module("exllamav3_ext")
SNAP = os.environ["SNAP"]; DEV = "cuda:0"
IDX = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
pre = "model.language_model.layers.3.mlp.gate_proj"
w = {}
for sfx in ("trellis", "suh", "svh", "mcg"):
    k = f"{pre}.{sfx}"
    if k in IDX:
        with safe_open(f"{SNAP}/{IDX[k]}", framework="pt", device=DEV) as f:
            w[sfx] = f.get_tensor(k)
bits = int(w["trellis"].shape[2]) // 16
K = w["trellis"].shape[0] * 16; N = w["trellis"].shape[1] * 16

def fold_at(budget):
    os.environ["VLLM_EXL3_FOLD_FP32_BUDGET_MB"] = str(budget)
    for m in list(sys.modules):
        if "exl3_fp4_conversion" in m:
            del sys.modules[m]
    mod = importlib.import_module("exl3_fp4_conversion")
    W = torch.zeros((K, N), dtype=torch.float16, device=DEV)
    ext.reconstruct(W, w["trellis"], bits, w.get("mcg") is not None, False)
    return mod.hadamard_fold_weight_chunked(W, w["suh"], w["svh"]).clone()

ref = fold_at(96)
for b in (24, 32, 48, 64):
    got = fold_at(b)
    same = torch.equal(ref, got)
    maxdiff = (got.float() - ref.float()).abs().amax().item()
    print(f"  budget {b:>3} MB vs 96 MB: bit_identical={same}  max_abs_diff={maxdiff:.3e}")
