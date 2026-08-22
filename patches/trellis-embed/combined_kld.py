#!/usr/bin/env python3
"""Combined trellis K{bits} KLD: embedding + lm_head quantized simultaneously.

Three subcommands run in separate containers:
  pre-encode-all: Encode both embedding and lm_head to trellis K{bits}.
  capture: Load vLLM, inject quantized embedding, capture hidden states to disk.
  kld: Load captured hidden states, project through quantized lm_head, compute KLD.
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


def trellis_roundtrip_2d_chunked(weight_bf16, bits, device):
    """Chunked trellis K{bits} round-trip on [V, H] weight. Returns BF16 result on GPU."""
    import torch
    import exllamav3_ext as ext
    sys.path.insert(0, '/opt/fp4')
    from exl3_fp4_conversion import hadamard_fold_weight
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_online_quantizer
    quantize_exl3 = _load_exl3_online_quantizer()

    V, H = weight_bf16.shape
    orig_cpu = weight_bf16.detach().cpu().float()
    CHUNK_V = 128 * 760
    chunks = []
    proxy_errors = []

    for v_start in range(0, V, CHUNK_V):
        v_end = min(v_start + CHUNK_V, V)
        v_size = v_end - v_start
        chunk_bf16 = weight_bf16[v_start:v_end, :].to(device)
        source = chunk_bf16.detach().t().float().contiguous()
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

    result = torch.cat(chunks, dim=0).to(device)
    del chunks
    avg_proxy = sum(proxy_errors) / len(proxy_errors)
    recon_cpu = result.cpu().float()
    mse = ((orig_cpu - recon_cpu) ** 2).mean().item()
    max_err = (orig_cpu - recon_cpu).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        orig_cpu.flatten().unsqueeze(0), recon_cpu.flatten().unsqueeze(0)
    ).item()
    del orig_cpu, recon_cpu
    return result, avg_proxy, mse, max_err, cos


def cmd_pre_encode_all(args):
    import torch
    from safetensors.torch import load_file as load_safetensors
    device = torch.device("cuda")

    # --- Encode embedding ---
    print(f"\n=== Pre-encoding embedding to trellis K{args.bits} ===", flush=True)
    model_dir = Path(args.model)
    embed_weight = None
    for sf in sorted(model_dir.glob("*.safetensors")):
        tensors = load_safetensors(str(sf), device="cpu")
        for name, t in tensors.items():
            if "embed_tokens" in name and "weight" in name:
                embed_weight = t.to(torch.bfloat16)
                break
        if embed_weight is not None:
            break
        del tensors
    if embed_weight is None:
        print("ERROR: embed_tokens.weight not found", file=sys.stderr)
        return 1
    print(f"  embedding: {list(embed_weight.shape)}", flush=True)
    embed_gpu = embed_weight.to(device)
    embed_result, e_proxy, e_mse, e_max, e_cos = trellis_roundtrip_2d_chunked(embed_gpu, args.bits, device)
    torch.save({"weight": embed_result.cpu(), "proxy_error": e_proxy, "mse": e_mse,
                "max_err": e_max, "cos": e_cos, "bits": args.bits}, args.embed_out)
    del embed_gpu, embed_result, embed_weight
    torch.cuda.empty_cache()
    print(f"  saved embedding to {args.embed_out}", flush=True)

    # --- Encode lm_head ---
    print(f"\n=== Pre-encoding lm_head to trellis K{args.bits} ===", flush=True)
    head_data = load_safetensors(args.head, device="cpu")
    head_key = list(head_data.keys())[0]
    head_weight = head_data[head_key].to(torch.bfloat16)
    del head_data
    print(f"  lm_head: {list(head_weight.shape)}", flush=True)
    head_gpu = head_weight.to(device)
    head_result, h_proxy, h_mse, h_max, h_cos = trellis_roundtrip_2d_chunked(head_gpu, args.bits, device)
    torch.save({"weight": head_result.cpu(), "proxy_error": h_proxy, "mse": h_mse,
                "max_err": h_max, "cos": h_cos, "bits": args.bits}, args.head_out)
    del head_gpu, head_result, head_weight
    torch.cuda.empty_cache()
    print(f"  saved lm_head to {args.head_out}", flush=True)
    print(f"\n  embedding: proxy={e_proxy:.6f} mse={e_mse:.2e} cos={e_cos:.8f}", flush=True)
    print(f"  lm_head:   proxy={h_proxy:.6f} mse={h_mse:.2e} cos={h_cos:.8f}", flush=True)
    return 0


def cmd_capture(args):
    """Load vLLM, inject quantized embedding, capture hidden states to disk."""
    import torch
    os.environ.pop("VLLM_EXL3_EMBED_ONLINE_BITS", None)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    suite_manifest = json.loads((Path(args.suite) / "suite-manifest.json").read_text())
    ctx_len = suite_manifest["context_length"]

    embed_pre = torch.load(args.pre_encoded_embed, map_location="cpu")
    bits = embed_pre["bits"]
    embed_tmp = args.pre_encoded_embed
    print(f"\n=== Trellis K{bits} embedding (pre-encoded) ===", flush=True)
    print(f"  proxy={embed_pre['proxy_error']:.6f} mse={embed_pre['mse']:.2e}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from safetensors.torch import save_file

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

    # Inject quantized embedding
    print(f"\n=== Injecting trellis K{bits} embedding ===", flush=True)
    def _inject_embed_rpc(self, embed_path):
        import torch as _torch
        model = self.model_runner.model
        for name, module in model.named_modules():
            if type(module).__name__ == "VocabParallelEmbedding" and "embed_tokens" in name:
                data = _torch.load(embed_path, map_location="cpu")
                weight = data["weight"] if isinstance(data, dict) else data
                module.weight.data.copy_(weight.to(module.weight.data.device))
                return {"injected": True, "shape": list(module.weight.data.shape)}
        return {"error": "no embedding found"}
    inject_result = llm.collective_rpc(_inject_embed_rpc, args=(embed_tmp,))[0]
    print(f"  injection: {inject_result}", flush=True)

    # Install hook
    def _rpc_install_hook(self):
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

        llm.collective_rpc(_rpc_pop_capture)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params=params, use_tqdm=False)
        got = llm.collective_rpc(_rpc_pop_capture)[0]
        if got is None or got.shape[0] != ctx_len:
            print(f"  capture failed for context {index}", file=sys.stderr)
            return 1
        hidden = got[:ctx_len - 1].contiguous()
        save_file({"hidden_states": hidden}, str(dst))
        sha = sha256_file(dst)
        records.append({"index": index, "sha256": sha, "shape": list(hidden.shape)})
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{len(suite_manifest['context_index'])} captured", flush=True)

    manifest = {
        "complete": True,
        "captures": records,
        "contexts": len(records),
        "suite_token_sha256": suite_manifest["suite_token_sha256"],
        "expected_indices": [c["index"] for c in suite_manifest["context_index"]],
        "filter": "all",
        "trellis_embed_bits": bits,
        "trellis_embed_proxy_error": embed_pre["proxy_error"],
    }
    (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ncapture_done: {len(records)} contexts", flush=True)
    return 0


def cmd_kld(args):
    """Load captured hidden states, project through quantized lm_head, compute KLD.
    No vLLM needed — pure GPU matmul."""
    import torch
    from safetensors.torch import load_file

    device = torch.device("cuda")
    suite_manifest = json.loads((Path(args.suite) / "suite-manifest.json").read_text())
    ctx_len = suite_manifest["context_length"]
    ref_dir = Path(args.reference)
    cap_dir = Path(args.captured)

    # Load heads
    head_pre = torch.load(args.pre_encoded_head, map_location="cpu")
    bits = head_pre["bits"]
    head_quant = head_pre["weight"].to(device)  # quantized lm_head on GPU

    head_data = load_file(args.head, device="cpu")
    head_key = list(head_data.keys())[0]
    head_bf16 = head_data[head_key].to(torch.bfloat16).to(device)  # BF16 lm_head on GPU
    del head_data

    print(f"\n=== Combined K{bits} KLD (quantized embed + quantized lm_head) ===", flush=True)
    print(f"  head proxy={head_pre['proxy_error']:.6f} mse={head_pre['mse']:.2e}", flush=True)

    all_kld = []
    all_top1_match = 0
    all_top1_total = 0
    per_context_kld = []
    t1 = time.time()

    for i, ctx in enumerate(suite_manifest["context_index"]):
        index = ctx["index"]

        # Load reference (BF16 embed → BF16 head) hidden states
        ref_data = load_file(str(ref_dir / f"hidden_{index:04d}.safetensors"), device="cpu")
        h_bf16 = ref_data["hidden_states"].to(device, dtype=torch.float16)
        del ref_data

        # Load candidate (quantized embed → model) hidden states
        cap_data = load_file(str(cap_dir / f"hidden_{index:04d}.safetensors"), device="cpu")
        h_quant = cap_data["hidden_states"].to(device, dtype=torch.float16)
        del cap_data

        # Project through heads and compute KLD
        chunk_size = 512
        for start in range(0, h_quant.shape[0], chunk_size):
            end = min(start + chunk_size, h_quant.shape[0])
            h_q = h_quant[start:end].float()
            h_r = h_bf16[start:end].float()

            logits_ref = h_r @ head_bf16.float().t()
            logits_cand = h_q @ head_quant.float().t()

            log_p = torch.log_softmax(logits_ref, dim=-1)
            log_q = torch.log_softmax(logits_cand, dim=-1)
            p = log_p.exp()
            kld = (p * (log_p - log_q)).sum(dim=-1)
            all_kld.append(kld.cpu())

            all_top1_match += (logits_ref.argmax(dim=-1) == logits_cand.argmax(dim=-1)).sum().item()
            all_top1_total += h_q.shape[0]

            del logits_ref, logits_cand, log_p, log_q, p, kld, h_q, h_r

        per_context_kld.append({"index": index, "mean_kld": 0})  # filled below
        del h_quant, h_bf16
        torch.cuda.empty_cache()

        if (i + 1) % 32 == 0:
            elapsed = time.time() - t1
            running_mean = torch.cat(all_kld).mean().item()
            print(f"  {i+1}/{len(suite_manifest['context_index'])} contexts, "
                  f"running mean {running_mean:.6f} ({elapsed:.0f}s)", flush=True)

    # Aggregate
    all_kld_tensor = torch.cat(all_kld)
    total_positions = all_kld_tensor.shape[0]
    sorted_kld = all_kld_tensor.sort().values
    mean_kld = all_kld_tensor.mean().item()
    p95 = sorted_kld[int(0.95 * total_positions)].item()
    p99 = sorted_kld[int(0.99 * total_positions)].item()
    p999 = sorted_kld[int(0.999 * total_positions)].item()
    max_kld = sorted_kld[-1].item()

    # Per-context means
    idx = 0
    for ctx in suite_manifest["context_index"]:
        ctx_len_actual = ctx_len - 1
        ctx_kld = all_kld_tensor[idx:idx + ctx_len_actual]
        per_context_kld[idx // ctx_len_actual] = {"index": ctx["index"], "mean_kld": ctx_kld.mean().item()}
        idx += ctx_len_actual

    context_means = [c["mean_kld"] for c in per_context_kld]
    context_macro_mean = sum(context_means) / len(context_means)

    report = {
        "schema": "qwen38-fidelity-report/1",
        "title": f"Combined trellis K{bits} (embedding + lm_head) KLD",
        "bits": bits,
        "components": "embed_tokens + lm_head (both trellis K{bits})",
        "embed_proxy_error": None,  # not available in kld subcommand
        "head_proxy_error": head_pre["proxy_error"],
        "token_mean_kld": mean_kld,
        "context_macro_mean_kld": context_macro_mean,
        "p95_kld": p95,
        "p99_kld": p99,
        "p999_kld": p999,
        "max_kld": max_kld,
        "top1_agreement": all_top1_match / all_top1_total if all_top1_total > 0 else None,
        "scored_positions": total_positions,
        "contexts": len(per_context_kld),
        "per_context": per_context_kld,
        "elapsed_sec": time.time() - t1,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\n=== Combined K{bits} KLD Results ===")
    for k in ("token_mean_kld", "context_macro_mean_kld", "p95_kld", "p99_kld",
              "p999_kld", "max_kld", "top1_agreement", "scored_positions"):
        print(f"  {k}: {report[k]}")
    print(f"\nSaved to {args.output}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("pre-encode-all")
    p_pre.add_argument("--model", required=True)
    p_pre.add_argument("--head", required=True)
    p_pre.add_argument("--bits", type=int, required=True, choices=[4, 6, 8])
    p_pre.add_argument("--embed-out", required=True)
    p_pre.add_argument("--head-out", required=True)
    p_pre.set_defaults(func=cmd_pre_encode_all)

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--model", required=True)
    p_cap.add_argument("--pre-encoded-embed", required=True)
    p_cap.add_argument("--suite", required=True)
    p_cap.add_argument("--out", required=True)
    p_cap.add_argument("--quantization", default="auto")
    p_cap.add_argument("--kv-cache-dtype", default="bfloat16")
    p_cap.add_argument("--attention-backend", default="TRITON_ATTN")
    p_cap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p_cap.add_argument("--max-batched-tokens", type=int, default=2048)
    p_cap.set_defaults(func=cmd_capture)

    p_kld = sub.add_parser("kld")
    p_kld.add_argument("--pre-encoded-head", required=True)
    p_kld.add_argument("--head", required=True, help="BF16 lm_head for reference projection")
    p_kld.add_argument("--reference", required=True, help="BF16 reference hidden states dir")
    p_kld.add_argument("--captured", required=True, help="Candidate hidden states dir (from capture step)")
    p_kld.add_argument("--suite", required=True)
    p_kld.add_argument("--output", required=True)
    p_kld.set_defaults(func=cmd_kld)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
