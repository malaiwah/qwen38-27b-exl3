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
    """Encode BF16 [V,H] embedding to trellis K{bits} in chunks, reconstruct, fold, return BF16.

    Chunks along V (the trellis N dimension) to avoid the quantize_exl3 CPU-swap
    path (numel > 5e8 triggers weight_r.cpu() which breaks the device assertion
    in fallback_quant). Hadamard fold is block-diagonal (128×128), so chunking
    with multiples of 128 gives equivalent results.
    """
    import torch
    import exllamav3_ext as ext
    sys.path.insert(0, '/opt/fp4')
    from exl3_fp4_conversion import hadamard_fold_weight

    V, H = weight_bf16.shape
    print(f"  embedding: [{V}, {H}], encoding to trellis K{bits}...", flush=True)

    # Keep original on CPU for final MSE
    orig_cpu = weight_bf16.detach().cpu().float()

    # Chunk along V (trellis N dim). Each chunk [H, chunk_V] stays under 5e8 elements.
    # 5e8 / 5120 = 97656 → use 97280 (760 × 128) for safety margin.
    CHUNK_V = 128 * 760  # 97280
    chunks = []
    proxy_errors = []

    # Load the online quantizer once
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_online_quantizer
    quantize_exl3 = _load_exl3_online_quantizer()

    for v_start in range(0, V, CHUNK_V):
        v_end = min(v_start + CHUNK_V, V)
        v_size = v_end - v_start
        print(f"  chunk [{v_start}:{v_end}] (V={v_size})...", flush=True)

        # Move chunk to GPU as [H, v_size] float32
        chunk_bf16 = weight_bf16[v_start:v_end, :].to(device)  # [v_size, H]
        source = chunk_bf16.detach().t().float().contiguous()   # [H, v_size]

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
        weight_fp16 = torch.empty(H, v_size, dtype=torch.float16, device=device)
        trellis_k = int(trellis.shape[2]) // 16
        ext.reconstruct(weight_fp16, trellis, trellis_k, True, False)

        # Hadamard fold
        weight_folded = hadamard_fold_weight(weight_fp16, suh, svh)

        # Transpose back to [v_size, H] BF16 and move to CPU
        chunk_result = weight_folded.t().contiguous().to(torch.bfloat16).cpu()
        chunks.append(chunk_result)
        proxy_errors.append(float(proxy_err))

        del chunk_bf16, source, trellis, suh, svh, weight_fp16, weight_folded
        torch.cuda.empty_cache()

    # Concatenate chunks
    result = torch.cat(chunks, dim=0).to(device)  # [V, H] BF16 on GPU
    del chunks
    avg_proxy = sum(proxy_errors) / len(proxy_errors)
    print(f"  avg proxy error: {avg_proxy:.6f} (over {len(proxy_errors)} chunks)", flush=True)

    # Measure round-trip error (on CPU to avoid OOM)
    recon_cpu = result.cpu().float()
    mse = ((orig_cpu - recon_cpu) ** 2).mean().item()
    max_err = (orig_cpu - recon_cpu).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        orig_cpu.flatten().unsqueeze(0), recon_cpu.flatten().unsqueeze(0)
    ).item()
    print(f"  round-trip MSE: {mse:.8f}, max_err: {max_err:.6f}, cosine_sim: {cos:.8f}", flush=True)

    del orig_cpu, recon_cpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result, avg_proxy, mse, max_err, cos

def cmd_pre_encode(args):
    """Pre-encode embedding with trellis K{bits} and save to file. Runs in its own process."""
    import torch
    from safetensors.torch import load_file as load_safetensors

    model_dir = Path(args.model)
    embed_weight = None
    embed_key = None
    for sf in sorted(model_dir.glob("*.safetensors")):
        tensors = load_safetensors(str(sf), device="cpu")
        for name, t in tensors.items():
            if "embed_tokens" in name and "weight" in name:
                embed_weight = t.to(torch.bfloat16)
                embed_key = name
                break
        if embed_weight is not None:
            break
        del tensors

    if embed_weight is None:
        print("ERROR: could not find embed_tokens.weight in safetensors", file=sys.stderr)
        return 1

    print(f"  found: {embed_key} {embed_weight.shape}", flush=True)

    device = torch.device("cuda")
    weight_gpu = embed_weight.to(device)
    result, proxy_err, mse, max_err, cos = trellis_roundtrip_embed(
        weight_gpu, args.bits, device)

    torch.save({
        "weight": result.cpu(),
        "proxy_error": float(proxy_err),
        "mse": float(mse),
        "max_err": float(max_err),
        "cos": float(cos),
        "bits": args.bits,
    }, args.output)
    print(f"  saved to {args.output}", flush=True)
    return 0


def cmd_capture(args):
    import torch
    os.environ.pop("VLLM_EXL3_EMBED_ONLINE_BITS", None)  # disable int8/int6 overlay
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    suite_manifest = json.loads((Path(args.suite) / "suite-manifest.json").read_text())
    ctx_len = suite_manifest["context_length"]

    # Load pre-encoded embedding
    pre = torch.load(args.pre_encoded)
    rt_metrics = {"proxy_error": pre["proxy_error"], "mse": pre["mse"],
                  "max_err": pre["max_err"], "cos": pre["cos"]}
    bits = pre["bits"]
    embed_tmp = args.pre_encoded
    print(f"\n=== Trellis K{bits} embedding (pre-encoded) ===", flush=True)
    print(f"  proxy_error={rt_metrics['proxy_error']:.6f}, mse={rt_metrics['mse']:.8f}, "
          f"max_err={rt_metrics['max_err']:.6f}, cos={rt_metrics['cos']:.8f}", flush=True)

    # --- Step 2: Load model (GPU is now free of encoding transients) ---
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

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

    print(f"\n=== Injecting trellis K{bits} embedding ===", flush=True)

    def _inject_embed_rpc(self, embed_path):
        import torch as _torch
        model = self.model_runner.model
        for name, module in model.named_modules():
            if type(module).__name__ == "VocabParallelEmbedding" and "embed_tokens" in name:
                data = _torch.load(embed_path, map_location="cpu")
                weight = data["weight"] if isinstance(data, dict) else data
                # Copy in chunks to avoid full-size temporary GPU allocation
                chunk = 16384
                dev = module.weight.data.device
                for i in range(0, weight.shape[0], chunk):
                    end = min(i + chunk, weight.shape[0])
                    module.weight.data[i:end].copy_(weight[i:end].to(dev))
                del weight, data
                _torch.cuda.empty_cache()
                return {"injected": True, "shape": list(module.weight.data.shape)}
        return {"error": "no embedding found"}
    inject_result = llm.collective_rpc(_inject_embed_rpc, args=(embed_tmp,))[0]
    print(f"  injection: {inject_result}", flush=True)

    # --- Step 4: Capture hidden states ---
    # Inline hook functions (avoid fidelity module dependency in worker process)
    from safetensors.torch import save_file

    def _rpc_install_hook(self):
        """Runs inside the worker process: hook the final norm, stash captures."""
        import torch
        model = self.model_runner.model
        cands = [(n, m) for n, m in model.named_modules()
                 if n.endswith("language_model.norm") or n == "model.norm"
                 or n.endswith(".model.norm")]
        if not cands:
            cands = [(n, m) for n, m in model.named_modules()
                     if n.split(".")[-1] == "norm" and "layers" not in n and "visual" not in n]
        if len(cands) != 1:
            raise RuntimeError(f"final norm ambiguous: {[n for n, _ in cands]}")
        name, norm = cands[0]
        store: dict = {"last": None, "rows": 0, "parts": [], "accumulate": True, "fp32": False}
        def hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            if t.dim() != 2:
                return output
            dtype = torch.float32 if store.get("fp32") else torch.bfloat16
            cpu = t.detach().to("cpu", dtype, copy=True)
            if store["accumulate"]:
                store["parts"].append(cpu)
                store["rows"] = sum(p.shape[0] for p in store["parts"])
                store["last"] = torch.cat(store["parts"], dim=0) if len(store["parts"]) > 1 else cpu
            elif cpu.shape[0] > store["rows"]:
                store["rows"] = cpu.shape[0]
                store["last"] = cpu
            return output
        norm.register_forward_hook(hook)
        self._fid_store = store
        return name

    def _rpc_pop_capture(self):
        store = getattr(self, "_fid_store", None)
        if store is None:
            return None
        t = store["last"]
        store["last"] = None
        store["rows"] = 0
        store["parts"] = []
        return t

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

        llm.collective_rpc(_rpc_pop_capture)  # reset
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params=params, use_tqdm=False)
        got = llm.collective_rpc(_rpc_pop_capture)[0]

        if got is None or got.shape[0] != ctx_len:
            print(f"  capture failed for context {index}", file=sys.stderr)
            return 1

        hidden = got[:ctx_len - 1].contiguous()
        save_file({"hidden_states": hidden}, str(dst))
        sha = sha256_file(dst)
        records.append({"index": index, "sha256": sha, "shape": list(hidden.shape)})

        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(suite_manifest['context_index'])} captured", flush=True)

    manifest = {
        "complete": True,
        "captures": records,
        "contexts": len(records),
        "suite_token_sha256": suite_manifest["suite_token_sha256"],
        "expected_indices": [c["index"] for c in suite_manifest["context_index"]],
        "filter": "all",
        "trellis_embed_bits": bits,
        "trellis_embed_proxy_error": rt_metrics["proxy_error"],
        "trellis_embed_mse": rt_metrics["mse"],
        "trellis_embed_max_err": rt_metrics["max_err"],
        "trellis_embed_cos": rt_metrics["cos"],
    }
    (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ncapture_done: {len(records)} contexts", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_pre = sub.add_parser("pre-encode")
    p_pre.add_argument("--model", required=True)
    p_pre.add_argument("--bits", type=int, required=True, choices=[4, 6, 8])
    p_pre.add_argument("--output", required=True)
    p_pre.set_defaults(func=cmd_pre_encode)

    p = sub.add_parser("capture")
    p.add_argument("--model", required=True)
    p.add_argument("--pre-encoded", required=True, help="Path to pre-encoded embedding .pt file")
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
