#!/usr/bin/env python3
"""Trellis K8 round-trip error on vision tower weights.

Loads all 2D vision weight tensors from safetensors, applies trellis K8
round-trip (quantize_exl3 + reconstruct + hadamard_fold_weight), and reports
per-tensor and aggregate round-trip error (proxy_error, MSE, max_err, cosine).

For weights where a dimension is not a multiple of 128 (MLP with 4304),
pads to the nearest 128 multiple before encoding, then crops back.

Usage (inside container):
    python3 vision_k8_roundtrip.py --model /path/to/checkpoint --bits 8
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch
from safetensors import safe_open


def trellis_roundtrip_2d(weight_bf16, bits, device):
    """Trellis K{bits} round-trip on a 2D weight [K, N]. Pads to 128-multiples if needed."""
    import exllamav3_ext as ext
    sys.path.insert(0, '/opt/fp4')
    from exl3_fp4_conversion import hadamard_fold_weight
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_online_quantizer
    quantize_exl3 = _load_exl3_online_quantizer()

    K, N = weight_bf16.shape
    pad_k = (128 - K % 128) % 128
    pad_n = (128 - N % 128) % 128

    if pad_k or pad_n:
        padded = torch.zeros(K + pad_k, N + pad_n, dtype=weight_bf16.dtype, device=device)
        padded[:K, :N] = weight_bf16
        source = padded.t().float().contiguous()  # [N+padded_n, K+padded_k]
    else:
        source = weight_bf16.t().float().contiguous()  # [N, K]

    H, V = source.shape  # trellis sees [K_trellis, N_trellis] = [N_orig, K_orig] after transpose

    # Meta Hessian for uncalibrated path
    H_meta = torch.empty(H, H, device='meta')
    H_data = {"H": H_meta, "L": None, "device": device, "count": 0, "finalized": False}

    quant_args = {"K": bits, "seed": 0, "devices": [str(source.device)],
                  "apply_out_scales": True, "mcg": True}
    _, proxy_err, tensors = quantize_exl3(
        source, H_data, quant_args, return_weight_q=False, verbose=False)

    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]

    # Reconstruct to fp16
    weight_fp16 = torch.empty(H, V, dtype=torch.float16, device=device)
    trellis_k = int(trellis.shape[2]) // 16
    ext.reconstruct(weight_fp16, trellis, trellis_k, True, False)

    # Hadamard fold
    weight_folded = hadamard_fold_weight(weight_fp16, suh, svh)

    # Transpose back and cast to BF16
    result = weight_folded.t().contiguous().to(torch.bfloat16)

    # Crop padding
    if pad_k or pad_n:
        result = result[:K, :N]

    del source, trellis, suh, svh, weight_fp16, weight_folded
    if pad_k or pad_n:
        del padded
    torch.cuda.empty_cache()

    return result, float(proxy_err)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    model_dir = Path(args.model)
    device = torch.device("cuda")

    # Collect all 2D vision weight tensors
    weights = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                if "visual" in k and k.endswith("weight"):
                    t = f.get_tensor(k)
                    if t.dim() == 2:
                        weights[k] = t.to(torch.bfloat16)

    print(f"Found {len(weights)} 2D vision weight tensors", flush=True)

    # Group by type for aggregate reporting
    results = []
    type_stats = {}  # pattern -> {count, total_numel, total_mse_sum, proxy_errors}

    for name in sorted(weights.keys()):
        w = weights[name]
        K, N = w.shape
        numel = K * N
        print(f"  {name} [{K},{N}] ({numel:,} params)...", flush=True, end=" ")

        w_gpu = w.to(device)
        t0 = time.time()
        result, proxy_err = trellis_roundtrip_2d(w_gpu, args.bits, device)
        elapsed = time.time() - t0

        # Measure error on CPU
        orig_f32 = w.float()
        recon_f32 = result.cpu().float()
        mse = ((orig_f32 - recon_f32) ** 2).mean().item()
        max_err = (orig_f32 - recon_f32).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            orig_f32.flatten().unsqueeze(0), recon_f32.flatten().unsqueeze(0)
        ).item()

        print(f"proxy={proxy_err:.6f} mse={mse:.2e} max={max_err:.6f} cos={cos:.8f} ({elapsed:.1f}s)", flush=True)

        # Determine type pattern
        parts = name.split(".")
        import re
        block_match = re.search(r'blocks\.(\d+)', name)
        if block_match:
            # Extract sub-module type
            remainder = name[name.index("blocks.") + len(f"blocks.{block_match.group(1)}."):]
            pattern = f"blocks.N.{remainder}"
        elif "merger" in name:
            pattern = f"merger.{parts[-2]}"
        else:
            pattern = name

        entry = {
            "name": name, "shape": [K, N], "numel": numel,
            "proxy_error": proxy_err, "mse": mse, "max_err": max_err,
            "cosine": cos, "elapsed_sec": elapsed,
            "pattern": pattern,
        }
        results.append(entry)

        if pattern not in type_stats:
            type_stats[pattern] = {"count": 0, "total_numel": 0, "proxy_errors": [], "mse_values": [], "cos_values": []}
        type_stats[pattern]["count"] += 1
        type_stats[pattern]["total_numel"] += numel
        type_stats[pattern]["proxy_errors"].append(proxy_err)
        type_stats[pattern]["mse_values"].append(mse)
        type_stats[pattern]["cos_values"].append(cos)

        del w_gpu, result, orig_f32, recon_f32
        torch.cuda.empty_cache()

    # Aggregate by type
    print(f"\n{'='*80}")
    print(f"Aggregate by weight type (trellis K{args.bits}):")
    print(f"{'pattern':45s} {'count':>5s} {'numel':>12s} {'avg_proxy':>10s} {'avg_mse':>12s} {'avg_cos':>10s}")
    total_numel = 0
    total_proxy = []
    total_mse = []
    total_cos = []
    for pattern in sorted(type_stats.keys()):
        s = type_stats[pattern]
        avg_proxy = sum(s["proxy_errors"]) / len(s["proxy_errors"])
        avg_mse = sum(s["mse_values"]) / len(s["mse_values"])
        avg_cos = sum(s["cos_values"]) / len(s["cos_values"])
        print(f"{pattern:45s} {s['count']:5d} {s['total_numel']:12,} {avg_proxy:10.6f} {avg_mse:12.2e} {avg_cos:10.8f}")
        total_numel += s["total_numel"]
        total_proxy.extend(s["proxy_errors"])
        total_mse.extend(s["mse_values"])
        total_cos.extend(s["cos_values"])

    overall_proxy = sum(total_proxy) / len(total_proxy)
    overall_mse = sum(total_mse) / len(total_mse)
    overall_cos = sum(total_cos) / len(total_cos)
    print(f"{'-'*80}")
    print(f"{'TOTAL':45s} {len(results):5d} {total_numel:12,} {overall_proxy:10.6f} {overall_mse:12.2e} {overall_cos:10.8f}")

    # Size estimate
    bf16_bytes = total_numel * 2
    k_bits = args.bits
    trellis_bytes = total_numel * k_bits / 8
    savings_bytes = bf16_bytes - trellis_bytes
    print(f"\nSize: BF16={bf16_bytes/1e9:.3f} GB → K{k_bits}={trellis_bytes/1e9:.3f} GB (saves {savings_bytes/1e9:.3f} GB)")

    output = {
        "bits": args.bits,
        "total_tensors": len(results),
        "total_params": total_numel,
        "overall_proxy_error": overall_proxy,
        "overall_mse": overall_mse,
        "overall_cosine": overall_cos,
        "bf16_gb": bf16_bytes / 1e9,
        "trellis_gb": trellis_bytes / 1e9,
        "savings_gb": savings_bytes / 1e9,
        "per_type": {p: {
            "count": s["count"],
            "total_numel": s["total_numel"],
            "avg_proxy_error": sum(s["proxy_errors"]) / len(s["proxy_errors"]),
            "avg_mse": sum(s["mse_values"]) / len(s["mse_values"]),
            "avg_cosine": sum(s["cos_values"]) / len(s["cos_values"]),
        } for p, s in sorted(type_stats.items())},
        "per_tensor": results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.output}", flush=True)
    else:
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
