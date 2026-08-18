#!/usr/bin/env python3
"""Ground truth test: compute EXL3 GEMM via reconstruct + matmul, compare with kernels."""

import sys
import os
import torch
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

def test_ground_truth(M=64):
    trellis, suh, svh, mcg, mul1 = load_layer_weights()
    K = trellis.size(0) * 16
    N = trellis.size(1) * 16
    bits = trellis.size(2) // 16
    cb = 1 if (mcg is not None) else 0
    mcg_bool = mcg is not None
    mul1_bool = mul1 is not None
    
    print(f"K={K}, N={N}, bits={bits}, cb={cb}")
    
    # Fixed input
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    
    # --- Ground truth: reconstruct + manual Hadamard + matmul ---
    # 1. Reconstruct weight matrix from trellis
    W = torch.empty(K, N, dtype=torch.float16, device="cuda")
    ext.reconstruct(W, trellis, bits, mcg_bool, mul1_bool)
    print(f"W: range=[{W.min():.4f}, {W.max():.4f}]")
    
    # 2. Input Hadamard: xh = had_r_128(x, suh)
    A_had_gt = torch.empty_like(A)
    ext.had_r_128(A, A_had_gt, suh, None, 1.0)
    print(f"A_had: range=[{A_had_gt.min():.4f}, {A_had_gt.max():.4f}]")
    
    # 3. GEMM: y = xh @ W (in fp32 for accuracy, then cast to fp16)
    Y_gt = (A_had_gt.float() @ W.float()).half()
    print(f"Y (before out Had): range=[{Y_gt.min():.4f}, {Y_gt.max():.4f}]")
    
    # 4. Output Hadamard: y = had_r_128(y, svh)
    Y_gt_h = torch.empty_like(Y_gt)
    ext.had_r_128(Y_gt, Y_gt_h, None, svh, 1.0)
    Y_gt = Y_gt_h.float()
    print(f"Y_gt (final): range=[{Y_gt.min():.4f}, {Y_gt.max():.4f}]")
    print(f"Y_gt[0,:8]: {Y_gt[0, :8].tolist()}")
    
    # --- Existing kernel ---
    C_ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
    A_had_ref = torch.empty_like(A)
    ext.exl3_gemm(A, trellis, C_ref, suh, A_had_ref, svh, -1, mcg_bool, mul1_bool, 0)
    print(f"\nExisting exl3_gemm (autotune):")
    print(f"  C_ref[0,:8]: {C_ref[0, :8].tolist()}")
    print(f"  Max diff vs GT: {(C_ref.float() - Y_gt).abs().max().item():.6f}")
    
    # --- v5 kernel ---
    C_new = torch.empty(M, N, dtype=torch.float16, device="cuda")
    A_had_new = torch.empty_like(A)
    locks = torch.zeros(N // 16, dtype=torch.int32, device="cuda")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    try:
        exllamav3_prefill_ext.exl3_gemm_prefill(
            A, trellis, C_new, suh, A_had_new, svh, bits, cb, locks, num_sms
        )
        print(f"\nv5 kernel:")
        print(f"  C_new[0,:8]: {C_new[0, :8].tolist()}")
        print(f"  Max diff vs GT: {(C_new.float() - Y_gt).abs().max().item():.6f}")
        print(f"  Max diff vs ref: {(C_new.float() - C_ref.float()).abs().max().item():.6f}")
        # Check for NaN/Inf
        nan_count = torch.isnan(C_new).sum().item()
        inf_count = torch.isinf(C_new).sum().item()
        print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
    except Exception as e:
        print(f"\nv5 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ground_truth(M=64)
