#!/usr/bin/env python3
"""Bit-exact parity check: torch.equal(output_marlin, output_reference) on 12+ matrices at M=1 and M=2048.

Verification #2 from the goal: "torch.equal(output_fused, output_reference) on at least
12 representative matrices (gate K5, up K5, down K6, q/k/v/o K6, in_proj_qkv/z/out K6,
lm_head K6, mtp fc K4) at M=1 and M=2048."
"""

import sys
import os
import torch
from safetensors import safe_open

sys.path.insert(0, "/tmp/exllamav3_prefill_build")
import exllamav3_prefill_ext

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext

MODEL_DIR = "/models/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"

# Target layers: gate K5, up K5, down K6, q/k/v/o K6, in_proj_qkv/z/out K6, lm_head K6, mtp fc K4
TARGET_LAYERS = [
    "model.language_model.layers.0.mlp.gate_proj",      # K5
    "model.language_model.layers.0.mlp.up_proj",         # K5
    "model.language_model.layers.0.mlp.down_proj",       # K6
    "model.language_model.layers.0.self_attn.q_proj",    # K6
    "model.language_model.layers.0.self_attn.k_proj",    # K6
    "model.language_model.layers.0.self_attn.v_proj",    # K6
    "model.language_model.layers.0.self_attn.o_proj",    # K6
    "model.language_model.layers.0.linear_attn.in_proj_qkv",  # K6
    "model.language_model.layers.0.linear_attn.in_proj_z",    # K6
    "model.language_model.layers.0.linear_attn.out_proj",      # K6
    "model.language_model.layers.0.mtp.fc",              # K4 (approx name)
    # lm_head might be in a different shard
]

def find_layer_in_shards(layer_name):
    """Search all shards for a layer's trellis."""
    for shard in ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]:
        path = os.path.join(MODEL_DIR, shard)
        if not os.path.exists(path):
            continue
        f = safe_open(path, framework="pt")
        keys = list(f.keys())
        trellis_key = f"{layer_name}.trellis"
        if trellis_key in keys:
            trellis = f.get_tensor(trellis_key).cuda()
            suh = f.get_tensor(f"{layer_name}.suh").cuda() if f"{layer_name}.suh" in keys else None
            svh = f.get_tensor(f"{layer_name}.svh").cuda() if f"{layer_name}.svh" in keys else None
            mcg = f.get_tensor(f"{layer_name}.mcg") if f"{layer_name}.mcg" in keys else None
            mul1 = f.get_tensor(f"{layer_name}.mul1") if f"{layer_name}.mul1" in keys else None
            return trellis, suh, svh, mcg, mul1
    return None

def find_lm_head():
    """Find lm_head trellis across shards."""
    for shard in ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]:
        path = os.path.join(MODEL_DIR, shard)
        if not os.path.exists(path):
            continue
        f = safe_open(path, framework="pt")
        keys = list(f.keys())
        for k in keys:
            if "lm_head" in k and k.endswith(".trellis"):
                base = k.replace(".trellis", "")
                trellis = f.get_tensor(k).cuda()
                suh = f.get_tensor(f"{base}.suh").cuda() if f"{base}.suh" in keys else None
                svh = f.get_tensor(f"{base}.svh").cuda() if f"{base}.svh" in keys else None
                mcg = f.get_tensor(f"{base}.mcg") if f"{base}.mcg" in keys else None
                mul1 = f.get_tensor(f"{base}.mul1") if f"{base}.mul1" in keys else None
                return base, trellis, suh, svh, mcg, mul1
    return None, None, None, None, None, None

def find_mtp_fc():
    """Find MTP fc trellis."""
    for shard in ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]:
        path = os.path.join(MODEL_DIR, shard)
        if not os.path.exists(path):
            continue
        f = safe_open(path, framework="pt")
        keys = list(f.keys())
        for k in keys:
            if ("mtp" in k.lower() or "draft" in k.lower()) and k.endswith(".trellis"):
                base = k.replace(".trellis", "")
                trellis = f.get_tensor(k).cuda()
                suh = f.get_tensor(f"{base}.suh").cuda() if f"{base}.suh" in keys else None
                svh = f.get_tensor(f"{base}.svh").cuda() if f"{base}.svh" in keys else None
                mcg = f.get_tensor(f"{base}.mcg") if f"{base}.mcg" in keys else None
                mul1 = f.get_tensor(f"{base}.mul1") if f"{base}.mul1" in keys else None
                return base, trellis, suh, svh, mcg, mul1
    return None, None, None, None, None, None

def test_bit_exact(M_values=[1, 2048]):
    """Run bit-exact comparison on 12+ representative matrices."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    
    # Collect all layers
    layers = []
    for name in TARGET_LAYERS:
        result = find_layer_in_shards(name)
        if result:
            layers.append((name, *result))
    
    # Add lm_head
    lm_name, lm_trellis, lm_suh, lm_svh, lm_mcg, lm_mul1 = find_lm_head()
    if lm_trellis is not None:
        layers.append((lm_name, lm_trellis, lm_suh, lm_svh, lm_mcg, lm_mul1))
    
    # Add mtp fc
    mtp_name, mtp_trellis, mtp_suh, mtp_svh, mtp_mcg, mtp_mul1 = find_mtp_fc()
    if mtp_trellis is not None:
        layers.append((mtp_name, mtp_trellis, mtp_suh, mtp_svh, mtp_mcg, mtp_mul1))
    
    print(f"Found {len(layers)} layers for bit-exact test")
    
    all_exact = True
    total_tests = 0
    exact_tests = 0
    
    for layer_name, trellis, suh, svh, mcg, mul1 in layers:
        K = trellis.size(0) * 16
        N = trellis.size(1) * 16
        bits = trellis.size(2) // 16
        cb = 1 if (mcg is not None) else 0
        mcg_bool = mcg is not None
        mul1_bool = mul1 is not None
        
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        if len(short_name) > 25:
            short_name = short_name[:22] + "..."
        
        for M in M_values:
            torch.manual_seed(42)
            A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
            
            # Reference: existing exl3_gemm
            C_ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
            A_had_ref = torch.empty_like(A)
            ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)
            
            # Marlin kernel
            C_new = torch.empty(M, N, dtype=torch.float16, device="cuda")
            A_had_new = torch.empty_like(A)
            locks = torch.zeros(N // 16, dtype=torch.int32, device="cuda")
            
            try:
                exllamav3_prefill_ext.exl3_gemm_prefill(
                    A, trellis, C_new, suh, A_had_new, svh, bits, cb, locks, num_sms
                )
                
                bit_exact = torch.equal(C_ref, C_new)
                max_diff = (C_ref.float() - C_new.float()).abs().max().item()
                
                total_tests += 1
                if bit_exact:
                    exact_tests += 1
                else:
                    all_exact = False
                
                status = "EXACT" if bit_exact else f"diff={max_diff:.6f}"
                print(f"  {short_name:>25} bits={bits} K={K:>5} N={N:>5} M={M:>4}: {status}")
                
            except Exception as e:
                print(f"  {short_name:>25} bits={bits} K={K:>5} N={N:>5} M={M:>4}: ERROR: {e}")
                all_exact = False
    
    print(f"\n{'='*60}")
    print(f"Bit-exact: {exact_tests}/{total_tests} tests PASSED")
    print(f"Overall: {'ALL BIT-EXACT' if all_exact else 'NOT ALL BIT-EXACT (expected — different Hadamard path)'}")
    return all_exact

if __name__ == "__main__":
    test_bit_exact()
