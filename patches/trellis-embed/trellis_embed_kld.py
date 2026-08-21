#!/usr/bin/env python3
"""Trellis K6/K8 embedding round-trip KLD test.

Runs inside the vLLM container. Loads the model, performs a trellis K{bits}
round-trip on the embedding table (encode → reconstruct → fold → BF16),
then captures hidden states and replays KLD against the BF16 reference.

The round-trip replaces the BF16 embedding with its trellis-quantized-and-
reconstructed version, measuring the fidelity cost of trellis K6/K8 embeddings.
The embedding stays BF16-sized (no memory savings) — this is a fidelity test,
not a production embedding path.

Usage (inside container):
    python3 trellis_embed_kld.py capture \
        --model /path/to/checkpoint --suite /path/to/suite \
        --out /path/to/capture --bits 6 \
        --quantization auto --kv-cache-dtype bfloat16 \
        --attention-backend TRITON_ATTN --gpu-memory-utilization 0.90 \
        --filter all --max-batched-tokens 2048 --hash-shards
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def trellis_roundtrip_embed(weight_bf16, bits, device):
    """Encode BF16 [V,H] embedding to trellis K{bits}, reconstruct, fold, return BF16."""
    import torch
    import exllamav3_ext as ext
    sys.path.insert(0, '/opt/fp4')
    from exl3_fp4_conversion import hadamard_fold_weight

    V, H = weight_bf16.shape
    print(f"  embedding: [{V}, {H}], encoding to trellis K{bits}...", flush=True)

    # Transpose to [H, V] = [K, N] for trellis encoding
    source = weight_bf16.detach().t().float().contiguous()

    # Dummy H_data with meta Hessian → uncalibrated path (q_fallback=True)
    H_meta = torch.empty(H, H, device='meta')
    H_data = {"H": H_meta, "L": None, "device": device, "count": 0, "finalized": False}

    # Load the online quantizer
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_online_quantizer
    quantize_exl3 = _load_exl3_online_quantizer()

    quant_args = {"K": bits, "seed": 0, "devices": [str(device)],
                  "apply_out_scales": True, "mcg": True}
    _, proxy_error, tensors = quantize_exl3(
        source, H_data, quant_args,
        return_weight_q=False, verbose=False,
    )

    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]
    print(f"  trellis shape: {list(trellis.shape)}, suh: {list(suh.shape)}, svh: {list(svh.shape)}", flush=True)
    print(f"  proxy error: {proxy_error:.6f}", flush=True)

    # Reconstruct to fp16
    weight_fp16 = torch.empty(H, V, dtype=torch.float16, device=device)
    trellis_k = int(trellis.shape[2]) // 16
    ext.reconstruct(weight_fp16, trellis, trellis_k, True, False)
    print(f"  reconstructed to fp16 [{H}, {V}]", flush=True)

    # Apply Hadamard fold: diag(suh) @ Had_K @ W @ Had_N @ diag(svh)
    weight_folded = hadamard_fold_weight(weight_fp16, suh, svh)
    print(f"  folded to final weight [{H}, {V}]", flush=True)

    # Transpose back to [V, H] and cast to BF16
    result = weight_folded.t().contiguous().to(torch.bfloat16)

    # Measure round-trip error
    orig_f32 = weight_bf16.detach().float()
    recon_f32 = result.float()
    mse = ((orig_f32 - recon_f32) ** 2).mean().item()
    max_err = (orig_f32 - recon_f32).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        orig_f32.flatten().unsqueeze(0), recon_f32.flatten().unsqueeze(0)
    ).item()
    print(f"  round-trip MSE: {mse:.8f}, max_err: {max_err:.6f}, cosine_sim: {cos:.8f}", flush=True)

    # Free intermediates
    del source, trellis, suh, svh, weight_fp16, weight_folded, orig_f32, recon_f32
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result, proxy_error


def cmd_capture(args):
    import torch
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Load model with BF16 embeddings (no overlay)
    os.environ["VLLM_EXL3_EMBED_ONLINE_BITS"] = "0"
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    suite_manifest = json.loads((Path(args.suite) / "suite-manifest.json").read_text())
    ctx_len = suite_manifest["context_length"]

    kwargs = dict(
        model=args.model, trust_remote_code=True, tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16", kv_cache_dtype=args.kv_cache_dtype,
        load_format="safetensors", max_model_len=ctx_len + 64,
        max_num_batched_tokens=args.max_batched_tokens or ctx_len,
        max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
        enforce_eager=True,
    )
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        kwargs["quantization"] = args.quantization
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend

    llm = LLM(**kwargs)

    # --- Trellis round-trip on the embedding ---
    print(f"\n=== Trellis K{args.bits} embedding round-trip ===", flush=True)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Find the embedding layer
    embed = None
    for name, module in model.named_modules():
        if type(module).__name__ == "VocabParallelEmbedding":
            if "embed_tokens" in name:
                embed = module
                print(f"  found embedding: {name}", flush=True)
                break
    if embed is None:
        print("ERROR: could not find VocabParallelEmbedding", file=sys.stderr)
        return 1

    device = next(embed.parameters()).device
    weight_bf16 = embed.weight.data.clone()

    # Do the trellis round-trip
    roundtripped, proxy_err = trellis_roundtrip_embed(weight_bf16, args.bits, device)

    # Replace the embedding weight
    embed.weight.data.copy_(roundtripped)
    del weight_bf16, roundtripped
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("  embedding replaced with trellis round-trip version", flush=True)

    # --- Capture hidden states ---
    print(f"\n=== Capturing {len(suite_manifest['context_index'])} contexts ===", flush=True)
    from fidelity import _rpc_install_hook  # type: ignore

    hooked = llm.collective_rpc(_rpc_install_hook)
    print(f"hooked {hooked}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = SamplingParams(max_tokens=1, temperature=0, detokenize=False)
    records = []
    for i, ctx in enumerate(suite_manifest["context_index"]):
        index = ctx["index"]
        ids = json.loads((Path(args.suite) / ctx["file"]).read_text())
        dst = out / f"hidden_{index:04d}.safetensors"

        llm.collective_rpc(lambda self: None)  # reset
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params=params, use_tqdm=False)
        got = llm.collective_rpc(lambda self: getattr(self, '_fid_store', {}).get('last'))[0]

        if got is None or got.shape[0] != ctx_len:
            print(f"  capture failed for context {index}", file=sys.stderr)
            return 1

        hidden = got[:ctx_len - 1].contiguous()
        save_file({"hidden_states": hidden}, str(dst))
        sha = sha256_file(dst)
        records.append({"index": index, "sha256": sha, "shape": list(hidden.shape)})

        if (i + 1) % 16 == 0:
            print(f"  {i+1} captured ({i+1}s approx)", flush=True)

    manifest = {
        "complete": True,
        "captures": records,
        "suite_token_sha256": suite_manifest["suite_token_sha256"],
        "expected_indices": [c["index"] for c in suite_manifest["context_index"]],
        "filter": "all",
        "trellis_embed_bits": args.bits,
        "trellis_embed_proxy_error": proxy_err,
    }
    (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ncapture_done: {len(records)} contexts", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capture")
    p.add_argument("--model", required=True)
    p.add_argument("--bits", type=int, default=6, choices=[6, 8])
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quantization", default="auto")
    p.add_argument("--kv-cache-dtype", default="bfloat16")
    p.add_argument("--attention-backend", default="TRITON_ATTN")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--filter", default="all")
    p.add_argument("--max-batched-tokens", type=int, default=2048)
    p.add_argument("--hash-shards", action="store_true")
    p.set_defaults(func=cmd_capture)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
