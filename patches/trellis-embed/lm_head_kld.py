#!/usr/bin/env python3
"""Trellis K{bits} lm_head quantization KLD.

Projects the BF16 reference hidden states through both the original BF16 lm_head
and a trellis-quantized lm_head, computing full-vocabulary KL divergence at
every scored position. This measures the fidelity cost of quantizing the output
projection — the layer that directly produces logits.

No vLLM needed — pure GPU matmul + KL computation.

Usage (inside container):
    python3 lm_head_kld.py --model /path/to/checkpoint --bits 8 \
        --head /path/to/weight.safetensors \
        --reference /path/to/reference/hidden-bf16 \
        --suite /path/to/suite --output /path/to/report.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def trellis_roundtrip_head(head_bf16, bits, device):
    """Trellis K{bits} round-trip on lm_head [vocab, hidden]. Chunked along vocab."""
    import exllamav3_ext as ext
    sys.path.insert(0, '/opt/fp4')
    from exl3_fp4_conversion import hadamard_fold_weight
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_online_quantizer
    quantize_exl3 = _load_exl3_online_quantizer()

    V, H = head_bf16.shape
    print(f"  lm_head: [{V}, {H}], encoding to trellis K{bits}...", flush=True)

    orig_cpu = head_bf16.detach().cpu().float()

    CHUNK_V = 128 * 760  # 97280
    chunks = []
    proxy_errors = []

    for v_start in range(0, V, CHUNK_V):
        v_end = min(v_start + CHUNK_V, V)
        v_size = v_end - v_start
        print(f"  chunk [{v_start}:{v_end}] (V={v_size})...", flush=True)

        chunk_bf16 = head_bf16[v_start:v_end, :].to(device)
        source = chunk_bf16.detach().t().float().contiguous()  # [H, v_size]

        H_meta = torch.empty(H, H, device='meta')
        H_data = {"H": H_meta, "L": None, "device": device, "count": 0, "finalized": False}

        quant_args = {"K": bits, "seed": 0, "devices": [str(source.device)],
                      "apply_out_scales": True, "mcg": True}
        _, proxy_err, tensors = quantize_exl3(
            source, H_data, quant_args, return_weight_q=False, verbose=False)

        trellis = tensors["trellis"]
        suh = tensors["suh"]
        svh = tensors["svh"]

        weight_fp16 = torch.empty(H, v_size, dtype=torch.float16, device=device)
        trellis_k = int(trellis.shape[2]) // 16
        ext.reconstruct(weight_fp16, trellis, trellis_k, True, False)

        weight_folded = hadamard_fold_weight(weight_fp16, suh, svh)
        chunk_result = weight_folded.t().contiguous().to(torch.bfloat16).cpu()
        chunks.append(chunk_result)
        proxy_errors.append(float(proxy_err))

        del chunk_bf16, source, trellis, suh, svh, weight_fp16, weight_folded
        torch.cuda.empty_cache()

    result = torch.cat(chunks, dim=0).to(device)  # [V, H] BF16 on GPU
    del chunks
    avg_proxy = sum(proxy_errors) / len(proxy_errors)
    print(f"  avg proxy error: {avg_proxy:.6f} (over {len(proxy_errors)} chunks)", flush=True)

    # Round-trip error
    recon_cpu = result.cpu().float()
    mse = ((orig_cpu - recon_cpu) ** 2).mean().item()
    max_err = (orig_cpu - recon_cpu).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        orig_cpu.flatten().unsqueeze(0), recon_cpu.flatten().unsqueeze(0)
    ).item()
    print(f"  round-trip MSE: {mse:.8f}, max_err: {max_err:.6f}, cosine_sim: {cos:.8f}", flush=True)

    del orig_cpu, recon_cpu
    return result, avg_proxy, mse, max_err, cos


def compute_kld_context(hidden, head_bf16, head_quant, device, chunk_size=512):
    """Compute per-position KL divergence for one context.

    hidden: [seq_len, hidden_dim] BF16 on CPU
    head_bf16: [vocab, hidden] BF16 on GPU
    head_quant: [vocab, hidden] BF16 on GPU
    Returns: list of per-position KL values
    """
    seq_len, H = hidden.shape
    hidden_gpu = hidden.to(device, dtype=torch.float16)  # [seq_len, H]

    all_kld = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        h_chunk = hidden_gpu[start:end].float()  # [chunk, H]

        # Project: logits = h @ head.T → [chunk, vocab]
        logits_ref = h_chunk @ head_bf16.float().t()   # [chunk, vocab]
        logits_cand = h_chunk @ head_quant.float().t()  # [chunk, vocab]

        # KL(P || Q) where P=softmax(ref), Q=softmax(cand)
        log_p = torch.log_softmax(logits_ref, dim=-1)
        log_q = torch.log_softmax(logits_cand, dim=-1)
        p = log_p.exp()
        kld = (p * (log_p - log_q)).sum(dim=-1)  # [chunk]

        all_kld.append(kld.cpu())
        del logits_ref, logits_cand, log_p, log_q, p, kld, h_chunk

    del hidden_gpu
    torch.cuda.empty_cache()
    return torch.cat(all_kld)  # [seq_len]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model checkpoint dir (for quantize_exl3 import)")
    parser.add_argument("--bits", type=int, required=True, choices=[4, 6, 8])
    parser.add_argument("--head", required=True, help="Path to lm_head weight.safetensors")
    parser.add_argument("--reference", required=True, help="Path to reference hidden states dir")
    parser.add_argument("--suite", required=True, help="Path to suite dir")
    parser.add_argument("--output", required=True, help="Output report JSON path")
    args = parser.parse_args()

    device = torch.device("cuda")
    t0 = time.time()

    # --- Step 1: Load and trellis-quantize lm_head ---
    print(f"\n=== Trellis K{args.bits} lm_head quantization ===", flush=True)
    with safe_open(args.head, framework="pt") as f:
        for k in f.keys():
            head_bf16 = f.get_tensor(k)
            break
    print(f"  loaded lm_head: {list(head_bf16.shape)} {head_bf16.dtype}", flush=True)

    head_bf16_gpu = head_bf16.to(device)
    head_quant, proxy_err, mse, max_err, cos = trellis_roundtrip_head(
        head_bf16_gpu, args.bits, device)

    # --- Step 2: Load suite and reference hidden states ---
    suite_manifest = json.loads((Path(args.suite) / "suite-manifest.json").read_text())
    ctx_len = suite_manifest["context_length"]
    ref_dir = Path(args.reference)

    print(f"\n=== Computing KLD over {len(suite_manifest['context_index'])} contexts ===", flush=True)

    # --- Step 3: Compute KL divergence per context ---
    all_kld = []
    per_context_kld = []
    t1 = time.time()

    for i, ctx in enumerate(suite_manifest["context_index"]):
        index = ctx["index"]
        ref_path = ref_dir / f"hidden_{index:04d}.safetensors"
        with safe_open(str(ref_path), framework="pt") as f:
            hidden = f.get_tensor("hidden_states")  # [2047, 5120]

        kld = compute_kld_context(hidden, head_bf16_gpu, head_quant, device)
        all_kld.append(kld)
        ctx_mean = kld.mean().item()
        per_context_kld.append({"index": index, "mean_kld": ctx_mean, "positions": kld.shape[0]})

        del hidden
        if (i + 1) % 32 == 0:
            elapsed = time.time() - t1
            running_mean = torch.cat(all_kld).mean().item()
            print(f"  {i+1}/{len(suite_manifest['context_index'])} contexts, "
                  f"running mean {running_mean:.6f} ({elapsed:.0f}s)", flush=True)

    # --- Step 4: Aggregate ---
    all_kld_tensor = torch.cat(all_kld)
    total_positions = all_kld_tensor.shape[0]

    # Sort for percentiles
    sorted_kld = all_kld_tensor.sort().values
    mean_kld = all_kld_tensor.mean().item()
    p95 = sorted_kld[int(0.95 * total_positions)].item()
    p99 = sorted_kld[int(0.99 * total_positions)].item()
    p999 = sorted_kld[int(0.999 * total_positions)].item()
    max_kld = sorted_kld[-1].item()

    # Context macro mean
    context_means = [c["mean_kld"] for c in per_context_kld]
    context_macro_mean = sum(context_means) / len(context_means)

    # top1 agreement (greedy argmax match)
    # Recompute for a sample of contexts to check argmax agreement
    top1_matches = 0
    top1_total = 0
    for i, ctx in enumerate(suite_manifest["context_index"]):
        index = ctx["index"]
        ref_path = ref_dir / f"hidden_{index:04d}.safetensors"
        with safe_open(str(ref_path), framework="pt") as f:
            hidden = f.get_tensor("hidden_states").to(device, dtype=torch.float16)
        logits_ref = (hidden.float() @ head_bf16_gpu.float().t())
        logits_cand = (hidden.float() @ head_quant.float().t())
        top1_matches += (logits_ref.argmax(dim=-1) == logits_cand.argmax(dim=-1)).sum().item()
        top1_total += hidden.shape[0]
        del hidden, logits_ref, logits_cand
    top1_agreement = top1_matches / top1_total

    report = {
        "schema": "qwen38-fidelity-report/1",
        "title": f"Trellis K{args.bits} lm_head KLD",
        "bits": args.bits,
        "proxy_error": proxy_err,
        "round_trip_mse": mse,
        "round_trip_max_err": max_err,
        "round_trip_cosine": cos,
        "token_mean_kld": mean_kld,
        "context_macro_mean_kld": context_macro_mean,
        "p95_kld": p95,
        "p99_kld": p99,
        "p999_kld": p999,
        "max_kld": max_kld,
        "top1_agreement": top1_agreement,
        "scored_positions": total_positions,
        "contexts": len(per_context_kld),
        "per_context": [{"index": c["index"], "mean_kld": c["mean_kld"]} for c in per_context_kld],
        "elapsed_sec": time.time() - t0,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\n=== Results (Trellis K{args.bits} lm_head) ===")
    for k in ("token_mean_kld", "context_macro_mean_kld", "p95_kld", "p99_kld",
              "p999_kld", "max_kld", "top1_agreement", "scored_positions"):
        print(f"  {k}: {report[k]}")
    print(f"\nSaved to {args.output}", flush=True)
    print(f"Elapsed: {report['elapsed_sec']:.0f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
