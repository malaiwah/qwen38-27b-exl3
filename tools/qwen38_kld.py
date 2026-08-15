#!/usr/bin/env python3
"""Teacher-forced full-vocabulary KLD for Qwen3.8-27B candidates, TP1.

Adapted from the published Gilded Gnosis harness
(`local-inference-lab/rtx6kpro@master:scripts/glm52_exl3_shared_h_kld.py` and
`models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/collect_prefill_return_logits_ref.py`).

`positional_kld` and the reporting statistics are kept byte-for-byte compatible
with that harness so numbers are comparable with the published GLM-5.2 tables:
full vocabulary, no top-k, KL(BF16 teacher || candidate) per position, mean of
per-run means and sample SD across run means.

Differences, all forced by this environment and recorded in the summary:
  * TP1 instead of TP4; no `B12X_MLA_SPARSE` / `moe_backend=b12x` (dense model).
  * No GLM `hf_overrides` (`use_index_cache`, `index_topk_pattern`).
  * `--quantization` is a flag, so BF16 / compressed-tensors / exl3 candidates
    all run through one code path.
  * This vLLM build has no `SamplingParams.return_prompt_logits`, so both the
    teacher capture and the candidates use the harness's documented fallback:
    `prompt_logprobs=-1, flat_logprobs=True`, densified to [npos, vocab].
  * The token window comes from exllamav3's bundled `wiki.utf8` (this image has
    no `datasets` package and no network at eval time). It is frozen to a JSON
    file on first use and asserted identical afterwards.

Usage:
    qwen38_kld.py tokens  --model DIR --out tokens.json [--context 2048]
    qwen38_kld.py capture --model DIR --tokens tokens.json --out REF_DIR [...]
    qwen38_kld.py score   --model DIR --tokens tokens.json --ref REF_DIR \\
                          --out RUN_DIR --label NAME [--quantization exl3] \\
                          [--quantization-config JSON] [--enforce-eager] [--repeats 3]
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

WIKI = "/work/exllamav3/exllamav3/conversion/standard_cal_data/wiki.utf8"


# ---------------------------------------------------------------- shared math
def dense_prompt_logprobs(prompt_logprobs: object, npos: int, vocab: int) -> torch.Tensor:
    """Densify vLLM prompt logprobs to [npos, vocab]; verbatim from the harness."""
    dense = torch.empty((npos, vocab), dtype=torch.float32)
    if hasattr(prompt_logprobs, "start_indices"):
        for pos in range(npos):
            start = prompt_logprobs.start_indices[pos + 1]
            end = prompt_logprobs.end_indices[pos + 1]
            ids = torch.as_tensor(prompt_logprobs.token_ids[start:end], dtype=torch.long)
            values = torch.as_tensor(prompt_logprobs.logprobs[start:end], dtype=torch.float32)
            row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
            valid = (ids >= 0) & (ids < vocab)
            row[ids[valid]] = values[valid]
            dense[pos] = row
        return dense

    for pos in range(npos):
        row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
        for token_id, logprob in prompt_logprobs[pos + 1].items():
            token_id = int(token_id)
            if 0 <= token_id < vocab:
                row[token_id] = float(logprob.logprob)
        dense[pos] = row
    return dense


def positional_kld(model_logits: torch.Tensor, reference_logits: torch.Tensor,
                   chunk_rows: int) -> torch.Tensor:
    """KL(reference || model) per position, full vocabulary; verbatim."""
    chunks = []
    for start in range(0, reference_logits.shape[0], chunk_rows):
        end = min(reference_logits.shape[0], start + chunk_rows)
        model_log_probs = F.log_softmax(model_logits[start:end].float(), dim=-1)
        reference_log_probs = F.log_softmax(reference_logits[start:end].float(), dim=-1)
        chunks.append(
            F.kl_div(model_log_probs, reference_log_probs,
                     reduction="none", log_target=True).sum(dim=-1)
        )
    return torch.cat(chunks).cpu()


def require_local_model(path: str) -> Path:
    root = Path(path).expanduser()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise SystemExit(
            "--model must be a reviewed local snapshot directory; download an immutable "
            "revision before evaluation"
        )
    return root


def build_llm(args, **overrides) -> LLM:
    require_local_model(args.model)
    kwargs = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Full prompt-logprob extraction needs several full-vocabulary
        # temporaries; do not let KV eat the spare VRAM.
        kv_cache_memory_bytes=512 * 1024 * 1024,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        load_format="safetensors",
        max_model_len=4096,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=-1,
        enforce_eager=True,
    )
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        kwargs["quantization"] = args.quantization
    if getattr(args, "quantization_config", None):
        kwargs["quantization_config"] = json.loads(args.quantization_config)
    kwargs.update(overrides)
    return LLM(**kwargs)


def prompt_logits(llm: LLM, token_ids: list[int], npos: int, vocab: int) -> torch.Tensor:
    supports = "return_prompt_logits" in inspect.signature(SamplingParams).parameters
    if supports:
        params = SamplingParams(prompt_logprobs=1, max_tokens=1,
                                return_prompt_logits=True, detokenize=False)
    else:
        params = SamplingParams(prompt_logprobs=-1, flat_logprobs=True,
                                max_tokens=1, detokenize=False)
    prompt: TokensPrompt = {"prompt_token_ids": token_ids}
    out = llm.generate([prompt], sampling_params=params)[0]
    if supports:
        if out.prompt_logits is None:
            raise RuntimeError("vLLM returned no prompt logits")
        logits = out.prompt_logits[:npos, :vocab].detach().cpu().float()
    else:
        if out.prompt_logprobs is None:
            raise RuntimeError("vLLM returned no prompt logprobs")
        logits = dense_prompt_logprobs(out.prompt_logprobs, npos, vocab)
    finite = torch.isfinite(logits)
    if not bool(finite.all()):
        raise RuntimeError(f"{int((~finite).sum())} non-finite logits; configuration invalid for KLD")
    return logits


# ------------------------------------------------------------------ commands
def cmd_tokens(args) -> int:
    require_local_model(args.model)
    from transformers import AutoTokenizer
    text = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
    text = "\n\n".join(p for p in text.split("\n") if p.strip())[: args.context * 8]
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=args.context)["input_ids"]
    if isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) != args.context:
        raise SystemExit(f"got {len(ids)} tokens, need {args.context}: corpus slice too short")
    Path(args.out).write_text(json.dumps(ids))
    print(json.dumps({"tokens": len(ids), "first16": ids[:16], "out": args.out}), flush=True)
    return 0


def cmd_capture(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    token_ids = json.loads(Path(args.tokens).read_text())
    llm = build_llm(args)
    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    npos = len(token_ids) - 1
    t0 = time.time()
    logits = prompt_logits(llm, token_ids, npos, vocab)
    save_file({"logits": logits}, str(out / "logits_0.safetensors"))
    manifest = {
        "model": args.model, "context_length": len(token_ids), "windows": 1,
        "token_first16": token_ids[:16], "shape": list(logits.shape),
        "kv_cache_dtype": args.kv_cache_dtype, "elapsed_sec": time.time() - t0,
        "corpus": args.corpus_note,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("capture_done " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0


def cmd_score(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref)
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    token_ids = json.loads(Path(args.tokens).read_text())
    if token_ids[:16] != manifest["token_first16"]:
        raise RuntimeError("tokenizer output does not match the BF16 reference window")
    with safe_open(str(ref_dir / "logits_0.safetensors"), framework="pt", device="cpu") as h:
        reference = h.get_tensor("logits")

    llm = build_llm(args)
    npos = min(len(token_ids) - 1, reference.shape[0])
    vocab = reference.shape[-1]

    results = []
    for run in range(1, args.repeats + 1):
        started = time.monotonic()
        logits = prompt_logits(llm, token_ids, npos, vocab)
        kld = positional_kld(logits, reference[:npos, :vocab], args.chunk_rows)
        if not bool(torch.isfinite(kld).all()):
            raise RuntimeError("non-finite KLD positions; refusing to publish this run")
        result = {"run": run, "mean_kld": float(kld.mean()),
                  "sd_across_positions": float(kld.std(unbiased=True)),
                  "positions": int(kld.numel()), "elapsed_sec": time.monotonic() - started}
        results.append(result)
        save_file({"kld": kld}, str(out / f"kld_positions_run{run}.safetensors"))
        print("kld_run " + json.dumps(result, sort_keys=True), flush=True)
        del logits, kld

    means = [r["mean_kld"] for r in results]
    summary = {
        "label": args.label, "model": args.model,
        "quantization": args.quantization,
        "quantization_config": json.loads(args.quantization_config) if args.quantization_config else None,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "reference_logits": str(ref_dir), "token_first16": token_ids[:16],
        "results": results, "mean_kld": statistics.fmean(means),
        "run_sd": statistics.stdev(means) if len(means) > 1 else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("kld_done " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokens")
    t.add_argument("--model", required=True)
    t.add_argument("--trust-remote-code", action="store_true",
                   help="execute model-repository Python (unsafe; off by default)")
    t.add_argument("--out", required=True)
    t.add_argument("--context", type=int, default=2048)
    t.add_argument("--corpus", default=WIKI)
    t.set_defaults(func=cmd_tokens)

    for name, fn in (("capture", cmd_capture), ("score", cmd_score)):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--trust-remote-code", action="store_true",
                       help="execute model-repository Python (unsafe; off by default)")
        s.add_argument("--tokens", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--quantization", default="auto")
        s.add_argument("--quantization-config", default=None)
        s.add_argument("--kv-cache-dtype", default="auto")
        s.add_argument("--tensor-parallel-size", type=int, default=1)
        s.add_argument("--gpu-memory-utilization", type=float, default=0.92)
        s.add_argument("--chunk-rows", type=int, default=32)
        if name == "capture":
            s.add_argument("--corpus-note", default="exllamav3 standard_cal_data/wiki.utf8, first 2048 tokens")
        else:
            s.add_argument("--ref", required=True)
            s.add_argument("--label", required=True)
            s.add_argument("--repeats", type=int, default=3)
        s.set_defaults(func=fn)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
