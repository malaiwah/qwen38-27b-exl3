#!/usr/bin/env python3
"""Benchmark v3 kernel vs reconstruct_hgemm path (the actual prefill path)."""

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

def reconstruct_hgemm(A, trellis, suh, svh, K_bits, mcg, mul1, out_dtype=torch.float16):
    """Simulate the reconstruct_hgemm path from exl3.py."""
    M, K = A.shape
    N = trellis.size(1) * 16
    
    # Input Hadamard
    xh = torch.empty_like(A)
    ext.had_r_128(A, xh, suh, None, 1.0)
    
    # Reconstruct weights
    w = torch.empty(K, N, dtype=torch.float16, device=A.device)
    ext.reconstruct(w, trellis, K_bits, mcg, mul1)
    
    # GEMM
    y = torch.empty(M, N, dtype=out_dtype, device=A.device)
    ext.hgemm(xh, w, y)
    
    # Output Hadamard
    y_out = torch.empty_like(y)
    ext.had_r_128(y, y_out, None, svh, 1.0)
    
    return y_out

def benchmark_vs_reconstruct(M_values=[128, 256, 512, 1024, 2048]):
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
    
    print(f"\n{'M':>6} {'recon_ms':>10} {'v3_ms':>10} {'exl3_ms':>10} {'v3_vs_recon':>12} {'v3_vs_exl3':>12} {'parity':>8}")
    
    for M in M_values:
        A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
        
        # reconstruct_hgemm benchmark
        for _ in range(3):
            reconstruct_hgemm(A, trellis, suh, svh, bits, mcg_bool, mul1_bool)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            C_recon = reconstruct_hgemm(A, trellis, suh, svh, bits, mcg_bool, mul1_bool)
        torch.cuda.synchronize()
        recon_ms = (time.perf_counter() - t0) / 10 * 1000
        
        # v3 kernel benchmark
        C_v3 = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had = torch.empty_like(A)
        locks = torch.zeros(N // 16, dtype=torch.int32, device="cuda")
        for _ in range(3):
            exllamav3_prefill_ext.exl3_gemm_prefill(A, trellis, C_v3, suh, A_had, svh, bits, cb, locks, num_sms)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            exllamav3_prefill_ext.exl3_gemm_prefill(A, trellis, C_v3, suh, A_had, svh, bits, cb, locks, num_sms)
        torch.cuda.synchronize()
        v3_ms = (time.perf_counter() - t0) / 10 * 1000
        
        # existing exl3_gemm benchmark
        C_exl3 = torch.empty(M, N, dtype=torch.float16, device="cuda")
        A_had2 = torch.empty_like(A)
        for _ in range(3):
            ext.exl3_gemm(A, trellis, C_exl3, suh, A_had2, svh, -1, mcg_bool, mul1_bool, 0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            ext.exl3_gemm(A, trellis, C_exl3, suh, A_had2, svh, -1, mcg_bool, mul1_bool, 0)
        torch.cuda.synchronize()
        exl3_ms = (time.perf_counter() - t0) / 10 * 1000
        
        # Parity check
        max_diff = (C_recon.float() - C_v3.float()).abs().max().item()
        ref_max = C_recon.float().abs().max().item()
        rel_diff = max_diff / max(ref_max, 1e-6)
        parity = "PASS" if rel_diff < 0.05 else "FAIL"
        
        v3_vs_recon = recon_ms / v3_ms
        v3_vs_exl3 = exl3_ms / v3_ms
        
        print(f"{M:>6} {recon_ms:>10.2f} {v3_ms:>10.2f} {exl3_ms:>10.2f} {v3_vs_recon:>11.2f}x {v3_vs_exl3:>11.2f}x {parity:>8}")

if __name__ == "__main__":
    benchmark_vs_reconstruct()
