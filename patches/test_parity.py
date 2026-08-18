#!/usr/bin/env python3
"""Parity test for exl3_gemm_prefill kernel.

Loads real EXL3 trellis weights from the model, runs both the existing
exl3_gemm and the new exl3_gemm_prefill on the same input, and compares
outputs.
"""

import sys
import os
import torch
import numpy as np
from safetensors import safe_open

# Load the prefill extension from the build directory
sys.path.insert(0, "/tmp/exllamav3_prefill_build")
import exllamav3_prefill_ext

# Load the prebuilt exllamav3_ext directly (avoid heavy package deps)
sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext

MODEL_DIR = "/models/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"

def load_layer_weights(layer_name="model.language_model.layers.0.linear_attn.in_proj_qkv"):
    """Load trellis, suh, svh, mcg, mul1 for a given layer from safetensors."""
    # Search across all safetensors files
    for shard in ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]:
        path = os.path.join(MODEL_DIR, shard)
        if not os.path.exists(path):
            continue
        f = safe_open(path, framework="pt")
        keys = list(f.keys())
        trellis_key = f"{layer_name}.trellis"
        if trellis_key in keys:
            trellis = f.get_tensor(trellis_key).cuda()
            suh_key = f"{layer_name}.suh"
            svh_key = f"{layer_name}.svh"
            suh = f.get_tensor(suh_key).cuda() if suh_key in keys else None
            svh = f.get_tensor(svh_key).cuda() if svh_key in keys else None
            mcg_key = f"{layer_name}.mcg"
            mul1_key = f"{layer_name}.mul1"
            mcg = f.get_tensor(mcg_key) if mcg_key in keys else None
            mul1 = f.get_tensor(mul1_key) if mul1_key in keys else None
            return trellis, suh, svh, mcg, mul1
    raise RuntimeError(f"Layer {layer_name} not found in model files")

def test_parity(M_values=[64, 128, 256]):
    """Test parity between existing exl3_gemm and new exl3_gemm_prefill."""
    
    # Load real weights
    trellis, suh, svh, mcg, mul1 = load_layer_weights()
    
    K = trellis.size(0) * 16
    N = trellis.size(1) * 16
    bits = trellis.size(2) // 16
    cb = 1 if (mcg is not None) else (2 if (mul1 is not None) else 0)
    mcg_bool = mcg is not None
    mul1_bool = mul1 is not None
    
    print(f"Trellis: {trellis.shape} {trellis.dtype}")
    print(f"K={K}, N={N}, bits={bits}, cb={cb}, mcg={mcg_bool}, mul1={mul1_bool}")
    print(f"suh: {suh.shape if suh is not None else None}")
    print(f"svh: {svh.shape if svh is not None else None}")
    
    all_pass = True
    
    for M in M_values:
        print(f"\n=== M={M} ===")
        
        # Create random input
        A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
        
        # --- Reference: existing exl3_gemm ---
        C_ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had_ref = torch.empty_like(A)
        
        ext.exl3_gemm(
            A, trellis, C_ref, suh, A_had_ref, svh,
            2,        # force_shape_idx (shape 2: M=16, K=32, N=128)
            mcg_bool,
            mul1_bool,
            0         # force_num_sms
        )
        
        # --- New kernel ---
        C_new = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had_new = torch.empty_like(A)
        # Locks: v4 uses shared locks (cooperative, same as existing kernel)
        locks = torch.zeros(N // 16, dtype=torch.int32, device="cuda")
        
        try:
            # Get number of SMs
            num_sms = torch.cuda.get_device_properties(0).multi_processor_count
            exllamav3_prefill_ext.exl3_gemm_prefill(
                A, trellis, C_new, suh, A_had_new, svh,
                bits, cb, locks, num_sms
            )
            
            # Compare
            max_diff = (C_ref.float() - C_new.float()).abs().max().item()
            mean_diff = (C_ref.float() - C_new.float()).abs().mean().item()
            ref_norm = C_ref.float().abs().max().item()
            
            bit_exact = torch.equal(C_ref, C_new)
            
            print(f"  Ref max abs: {ref_norm:.4f}")
            print(f"  Max diff:    {max_diff:.6f}")
            print(f"  Mean diff:   {mean_diff:.6f}")
            print(f"  Bit-exact:   {bit_exact}")
            
            if max_diff < 1e-2 or (ref_norm > 0 and max_diff / ref_norm < 1e-3):
                print(f"  PASS")
            else:
                print(f"  FAIL (max diff {max_diff:.6f}, ref norm {ref_norm:.4f})")
                all_pass = False
                print(f"  C_ref[0,:8]: {C_ref[0, :8].tolist()}")
                print(f"  C_new[0,:8]: {C_new[0, :8].tolist()}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
    
    print(f"\n{'='*40}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass

if __name__ == "__main__":
    success = test_parity()
    sys.exit(0 if success else 1)
