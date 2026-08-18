#!/usr/bin/env python3
"""Multi-M parity + benchmark test for exl3_gemm_prefill kernel."""

import sys
import os
import torch
import time
from safetensors import safe_open

sys.path.insert(0, "/tmp/exllamav3_prefill_build")
import exllamav3_prefill_ext

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext

MODEL_DIR = "/models/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"

def load_layer_weights(layer_name="model.language_model.layers.0.linear_attn.in_proj_qkv"):
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
    raise RuntimeError(f"Layer not found")

def test_parity_and_benchmark(M_values=[1, 16, 64, 128, 256, 512, 1024, 2048]):
    trellis, suh, svh, mcg, mul1 = load_layer_weights()
    K = trellis.size(0) * 16
    N = trellis.size(1) * 16
    bits = trellis.size(2) // 16
    cb = 1 if (mcg is not None) else 0
    mcg_bool = mcg is not None
    mul1_bool = mul1 is not None
    
    print(f"K={K}, N={N}, bits={bits}, cb={cb}")
    
    torch.manual_seed(42)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    
    all_pass = True
    print(f"\n{'M':>6} {'ref_max':>10} {'max_diff':>10} {'rel_diff':>10} {'PASS':>5} {'ref_ms':>8} {'new_ms':>8} {'speedup':>8}")
    
    for M in M_values:
        A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
        
        # Reference
        C_ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had_ref = torch.empty_like(A)
        ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)
        
        # New kernel
        C_new = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had_new = torch.empty_like(A)
        locks = torch.zeros(N // 16, dtype=torch.int32, device="cuda")
        
        try:
            exllamav3_prefill_ext.exl3_gemm_prefill(
                A, trellis, C_new, suh, A_had_new, svh, bits, cb, locks, num_sms
            )
            
            max_diff = (C_ref.float() - C_new.float()).abs().max().item()
            ref_max = C_ref.float().abs().max().item()
            rel_diff = max_diff / max(ref_max, 1e-6)
            passed = rel_diff < 0.05  # 5% relative tolerance for fp16
            
            if not passed:
                all_pass = False
            
            # Benchmark (warmup + 10 iterations)
            torch.cuda.synchronize()
            for _ in range(3):
                ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)
            torch.cuda.synchronize()
            ref_ms = (time.perf_counter() - t0) / 10 * 1000
            
            torch.cuda.synchronize()
            for _ in range(3):
                exllamav3_prefill_ext.exl3_gemm_prefill(A, trellis, C_new, suh, A_had_new, svh, bits, cb, locks, num_sms)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                exllamav3_prefill_ext.exl3_gemm_prefill(A, trellis, C_new, suh, A_had_new, svh, bits, cb, locks, num_sms)
            torch.cuda.synchronize()
            new_ms = (time.perf_counter() - t0) / 10 * 1000
            
            speedup = ref_ms / new_ms if new_ms > 0 else 0
            
            print(f"{M:>6} {ref_max:>10.4f} {max_diff:>10.6f} {rel_diff:>10.6f} {'YES' if passed else 'NO':>5} {ref_ms:>8.2f} {new_ms:>8.2f} {speedup:>8.2f}x")
            
        except Exception as e:
            print(f"{M:>6} ERROR: {e}")
            all_pass = False
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass

if __name__ == "__main__":
    success = test_parity_and_benchmark()
    sys.exit(0 if success else 1)
