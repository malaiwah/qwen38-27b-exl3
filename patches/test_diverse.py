#!/usr/bin/env python3
"""Multi-layer parity test across diverse EXL3 shapes."""

import sys
import os
import torch
from safetensors import safe_open

sys.path.insert(0, "/tmp/exllamav3_prefill_build")
import exllamav3_prefill_ext

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext

MODEL_DIR = "/models/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"

def find_trellis_layers():
    """Find diverse trellis layers across all shards."""
    layers = []
    for shard in ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]:
        path = os.path.join(MODEL_DIR, shard)
        if not os.path.exists(path):
            continue
        f = safe_open(path, framework="pt")
        keys = list(f.keys())
        for k in keys:
            if k.endswith(".trellis"):
                base = k.replace(".trellis", "")
                trellis = f.get_tensor(k)
                K = trellis.size(0) * 16
                N = trellis.size(1) * 16
                bits = trellis.size(2) // 16
                # Diversify: pick different K/N/bits combos
                layers.append((base, shard, K, N, bits))
    return layers

def load_layer(base, shard):
    path = os.path.join(MODEL_DIR, shard)
    f = safe_open(path, framework="pt")
    keys = list(f.keys())
    trellis = f.get_tensor(f"{base}.trellis").cuda()
    suh = f.get_tensor(f"{base}.suh").cuda() if f"{base}.suh" in keys else None
    svh = f.get_tensor(f"{base}.svh").cuda() if f"{base}.svh" in keys else None
    mcg = f.get_tensor(f"{base}.mcg") if f"{base}.mcg" in keys else None
    mul1 = f.get_tensor(f"{base}.mul1") if f"{base}.mul1" in keys else None
    return trellis, suh, svh, mcg, mul1

def test_diverse_layers():
    layers = find_trellis_layers()
    print(f"Found {len(layers)} trellis layers total")
    
    # Pick diverse layers: different bits (4,5,6), different N, different K
    # Group by (bits, K, N) and pick one from each group
    seen = set()
    diverse = []
    for base, shard, K, N, bits in layers:
        key = (bits, K, N)
        if key not in seen:
            seen.add(key)
            diverse.append((base, shard, K, N, bits))
    
    print(f"Testing {len(diverse)} diverse shapes")
    
    M = 128  # Prefill size
    torch.manual_seed(42)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    
    all_pass = True
    print(f"\n{'Layer':>60} {'bits':>4} {'K':>6} {'N':>6} {'max_diff':>10} {'rel_diff':>10} {'PASS':>5}")
    
    for base, shard, K, N, bits in diverse[:15]:  # Test up to 15 shapes
        try:
            trellis, suh, svh, mcg, mul1 = load_layer(base, shard)
            cb = 1 if (mcg is not None) else 0
            mcg_bool = mcg is not None
            mul1_bool = mul1 is not None
            
            A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
            
            # Reference
            C_ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
            A_had_ref = torch.empty_like(A)
            ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)
            
            # New kernel
            C_new = torch.empty(M, N, dtype=torch.float16, device="cuda")
            A_had_new = torch.empty_like(A)
            locks = torch.zeros(N // 16, dtype=torch.int32, device="cuda")
            
            exllamav3_prefill_ext.exl3_gemm_prefill(
                A, trellis, C_new, suh, A_had_new, svh, bits, cb, locks, num_sms
            )
            
            max_diff = (C_ref.float() - C_new.float()).abs().max().item()
            ref_max = C_ref.float().abs().max().item()
            rel_diff = max_diff / max(ref_max, 1e-6)
            passed = rel_diff < 0.05
            
            if not passed:
                all_pass = False
            
            short_name = base.split(".")[-2] + "." + base.split(".")[-1]
            print(f"{short_name:>60} {bits:>4} {K:>6} {N:>6} {max_diff:>10.6f} {rel_diff:>10.6f} {'YES' if passed else 'NO':>5}")
            
        except Exception as e:
            print(f"{base:>60} ERROR: {e}")
            all_pass = False
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass

if __name__ == "__main__":
    success = test_diverse_layers()
    sys.exit(0 if success else 1)
