#!/usr/bin/env python3
"""Distribution-fidelity harness for Qwen3.8-27B candidates (hidden-state replay).

Adopts the protocol from
`local-inference-lab/rtx6kpro:models/kimi-k3/distribution-fidelity-1024x2048.md`:
capture the BF16 hidden state after the final RMSNorm and before the LM head, then
reconstruct complete full-vocabulary distributions offline through ONE shared BF16
LM head. Storage per candidate is `contexts x 2047 x 5120 x 2 B` (21 MB per
context) instead of `contexts x 2047 x 248320 x 4 B` (2 GB per context), which is
what makes many-context evaluation affordable.

Why this is better than scoring one window through `prompt_logprobs`:
  * many contexts, stratified by content type, so the mean has a confidence
    interval instead of being a single sample;
  * exact two-pass full-vocabulary metrics (normalizers first, then KL/JS), never
    top-k;
  * KL, Jensen-Shannon, top-1 agreement, tail quantiles and per-context reports
    from the same pass;
  * the LM head is factored out, so body quantization and head quantization can be
    attributed separately - replay the same hidden states through a different head
    with `--head`.

Capture needs no vLLM patch: a forward hook on the final norm module sees exactly
the tensor the reference protocol captures.

    fidelity.py suite    --model DIR --out SUITE_DIR [--contexts 128]
    fidelity.py capture  --model DIR --suite SUITE_DIR --out CAP_DIR [--quantization ...]
    fidelity.py replay   --reference REF_DIR --candidate CAP_DIR --head W.safetensors \\
                         --suite SUITE_DIR --out report.json
    fidelity.py paired   --a report_a.json --b report_b.json --out paired.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

CORPORA = {  # stratum -> exllamav3 calibration file
    "encyclopedic": "wiki.utf8",
    "web": "c4.utf8",
    "code": "code.utf8",
    "technical": "technical.utf8",
    "multilingual": "multilingual.utf8",
    "short_form": "tiny.utf8",
}
CAL_DIR = "/work/exllamav3/exllamav3/conversion/standard_cal_data"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ suite
def cmd_suite(args) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out = Path(args.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    ctx_len = args.context_length
    per_stratum = {k: args.contexts // len(CORPORA) for k in CORPORA}
    for k in list(per_stratum)[: args.contexts % len(CORPORA)]:
        per_stratum[k] += 1

    contexts, seen = [], set()
    for stratum, fname in CORPORA.items():
        want = per_stratum[stratum]
        raw = Path(CAL_DIR, fname).read_text(encoding="utf-8", errors="ignore")
        # Split into coherent units, then walk them until each context is full.
        units = [u.strip() for u in raw.split("\n\n") if len(u.strip()) > 400]
        buf, idx = [], 0
        while len(buf) < want and idx < len(units):
            piece, taken = [], 0
            while idx < len(units) and taken < ctx_len * 6:
                piece.append(units[idx])
                taken += len(units[idx])
                idx += 1
            text = "\n\n".join(piece)
            ids = tok(text, add_special_tokens=False, truncation=True,
                      max_length=ctx_len)["input_ids"]
            if isinstance(ids[0], list):
                ids = ids[0]
            if len(ids) < ctx_len:
                continue
            ids = ids[:ctx_len]
            digest = sha256_bytes(json.dumps(ids).encode())
            if digest in seen:
                continue
            seen.add(digest)
            buf.append((ids, digest))
        for ids, digest in buf:
            i = len(contexts)
            name = f"context-{i:04d}.json"
            (out / "tokens" / name).write_text(json.dumps(ids))
            contexts.append({"index": i, "stratum": stratum, "file": f"tokens/{name}",
                             "token_sha256": digest, "tokens": len(ids),
                             "source_cluster": f"{stratum}-{i // 4}"})
        print(f"{stratum}: {len(buf)} contexts", flush=True)

    manifest = {
        "schema": "qwen38-distribution-fidelity/1",
        "model": args.model, "context_length": ctx_len,
        "scored_positions_per_context": ctx_len - 1,
        "contexts": len(contexts), "total_scored_positions": len(contexts) * (ctx_len - 1),
        "hidden_size": 5120, "vocab_size": 248320,
        "tokenizer_sha256": sha256_file(Path(args.model, "tokenizer.json")),
        "strata": {k: sum(1 for c in contexts if c["stratum"] == k) for k in CORPORA},
        "context_index": contexts,
    }
    manifest["suite_token_sha256"] = sha256_bytes(
        "".join(c["token_sha256"] for c in contexts).encode())
    (out / "suite-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("contexts", "total_scored_positions", "suite_token_sha256")}), flush=True)
    return 0


# ---------------------------------------------------------------- capture
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
    store: dict = {"last": None, "rows": 0}

    def hook(_m, _i, output):
        t = output[0] if isinstance(output, tuple) else output
        if t.dim() == 2 and t.shape[0] > store["rows"]:
            store["rows"] = t.shape[0]
            store["last"] = t.detach().to("cpu", torch.bfloat16, copy=True)
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
    return t


def cmd_capture(args) -> int:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from safetensors.torch import save_file

    suite = Path(args.suite)
    manifest = json.loads((suite / "suite-manifest.json").read_text())
    ctx_len = manifest["context_length"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kwargs = dict(model=args.model, trust_remote_code=True, tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  kv_cache_memory_bytes=512 * 1024 * 1024, dtype="bfloat16",
                  kv_cache_dtype=args.kv_cache_dtype, load_format="safetensors",
                  max_model_len=ctx_len + 64, max_num_batched_tokens=ctx_len,
                  max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
                  enforce_eager=True)
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        kwargs["quantization"] = args.quantization
    if args.quantization_config:
        kwargs["quantization_config"] = json.loads(args.quantization_config)
    llm = LLM(**kwargs)

    hooked = llm.collective_rpc(_rpc_install_hook)
    print(f"hooked {hooked}", flush=True)

    params = SamplingParams(max_tokens=1, temperature=0, detokenize=False)
    done = 0
    t0 = time.time()
    records = []
    for ctx in manifest["context_index"]:
        dst = out / f"hidden_{ctx['index']:04d}.safetensors"
        if dst.exists():
            done += 1
            continue
        ids = json.loads((suite / ctx["file"]).read_text())
        if sha256_bytes(json.dumps(ids).encode()) != ctx["token_sha256"]:
            raise SystemExit(f"token hash drift on context {ctx['index']}")
        llm.collective_rpc(_rpc_pop_capture)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params=params,
                     use_tqdm=False)
        got = llm.collective_rpc(_rpc_pop_capture)[0]
        if got is None or got.shape[0] != ctx_len:
            raise SystemExit(f"capture failed for context {ctx['index']}: "
                             f"{None if got is None else tuple(got.shape)}")
        hidden = got[: ctx_len - 1].contiguous()
        save_file({"hidden_states": hidden}, str(dst))
        records.append({"index": ctx["index"], "sha256": sha256_file(dst),
                        "shape": list(hidden.shape)})
        done += 1
        if done % 16 == 0:
            print(f"{done}/{manifest['contexts']} ({time.time() - t0:.0f}s)", flush=True)
    meta = {"model": args.model, "quantization": args.quantization,
            "quantization_config": args.quantization_config,
            "kv_cache_dtype": args.kv_cache_dtype,
            "suite_token_sha256": manifest["suite_token_sha256"],
            "contexts": done, "captures": records, "elapsed_sec": time.time() - t0}
    (out / "capture-manifest.json").write_text(json.dumps(meta, indent=2))
    print("capture_done " + json.dumps({k: meta[k] for k in ("contexts", "elapsed_sec")}), flush=True)
    return 0


# ----------------------------------------------------------------- replay
@torch.inference_mode()
def normalizers_and_top1(hidden: torch.Tensor, head: torch.Tensor, vocab_chunk: int):
    rows = hidden.shape[0]
    log_z = torch.full((rows,), -math.inf, dtype=torch.float32, device=hidden.device)
    top_val = torch.full((rows,), -math.inf, dtype=torch.float32, device=hidden.device)
    top_id = torch.zeros((rows,), dtype=torch.int64, device=hidden.device)
    for start in range(0, head.shape[0], vocab_chunk):
        end = min(start + vocab_chunk, head.shape[0])
        logits = (hidden @ head[start:end].T).float()
        log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
        val, idx = logits.max(dim=-1)
        upd = val > top_val
        top_val = torch.where(upd, val, top_val)
        top_id = torch.where(upd, idx + start, top_id)
    return log_z, top_id


@torch.inference_mode()
def context_metrics(ref_h: torch.Tensor, cand_h: torch.Tensor, head: torch.Tensor,
                    vocab_chunk: int):
    ref_z, ref_top = normalizers_and_top1(ref_h, head, vocab_chunk)
    cand_z, cand_top = normalizers_and_top1(cand_h, head, vocab_chunk)
    rows = ref_h.shape[0]
    kl = torch.zeros(rows, dtype=torch.float64, device=ref_h.device)
    js = torch.zeros(rows, dtype=torch.float64, device=ref_h.device)
    ln2 = math.log(2.0)
    for start in range(0, head.shape[0], vocab_chunk):
        end = min(start + vocab_chunk, head.shape[0])
        w = head[start:end].T
        rl = (ref_h @ w).float() - ref_z[:, None]
        cl = (cand_h @ w).float() - cand_z[:, None]
        p, q = rl.exp(), cl.exp()
        kl += (p * (rl - cl)).sum(-1).double()
        m = 0.5 * (p + q)
        logm = m.clamp_min(1e-30).log()
        js += (0.5 * (p * (rl - logm)).sum(-1) + 0.5 * (q * (cl - logm)).sum(-1)).double()
    return kl.cpu(), (js / ln2).cpu(), (ref_top == cand_top).sum().item()


def bootstrap(values: list[float], clusters: list[str], samples: int, seed: int):
    import random
    by = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        pick = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [v for grp in pick for v in grp]
        means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return {"mean": statistics.fmean(values), "ci95_low": lo, "ci95_high": hi,
            "clusters": len(keys), "samples": samples}


def cmd_replay(args) -> int:
    from safetensors.torch import safe_open

    suite = json.loads(Path(args.suite, "suite-manifest.json").read_text())
    dev = torch.device(args.device)
    with safe_open(args.head, framework="pt", device="cpu") as f:
        key = "weight" if "weight" in f.keys() else f.keys()[0]
        head = f.get_tensor(key).to(dev, torch.bfloat16)
    vocab, hidden_size = head.shape
    per_ctx, strata, clusters, top1_hits, positions = [], {}, [], 0, 0
    all_kl = []

    for ctx in suite["context_index"]:
        i = ctx["index"]
        rp = Path(args.reference, f"hidden_{i:04d}.safetensors")
        cp = Path(args.candidate, f"hidden_{i:04d}.safetensors")
        if not (rp.exists() and cp.exists()):
            continue
        with safe_open(str(rp), framework="pt", device="cpu") as f:
            ref = f.get_tensor("hidden_states").to(dev, torch.bfloat16)
        with safe_open(str(cp), framework="pt", device="cpu") as f:
            cand = f.get_tensor("hidden_states").to(dev, torch.bfloat16)
        kl, js, hits = context_metrics(ref, cand, head, args.vocab_chunk)
        m = float(kl.mean())
        per_ctx.append({"index": i, "stratum": ctx["stratum"],
                        "source_cluster": ctx["source_cluster"],
                        "mean_kld": m, "median_kld": float(kl.median()),
                        "max_kld": float(kl.max()), "mean_jsd_bits": float(js.mean()),
                        "top1_agreement": hits / kl.numel()})
        strata.setdefault(ctx["stratum"], []).append(m)
        clusters.append(ctx["source_cluster"])
        top1_hits += hits
        positions += kl.numel()
        all_kl.append(kl)
        if len(per_ctx) % 16 == 0:
            print(f"{len(per_ctx)} contexts, running mean {statistics.fmean(x['mean_kld'] for x in per_ctx):.6f}", flush=True)

    if not per_ctx:
        raise SystemExit("no overlapping contexts between reference and candidate")
    kl_all = torch.cat(all_kl).double()
    q = lambda p: float(kl_all.quantile(p))
    means = [c["mean_kld"] for c in per_ctx]
    report = {
        "schema": "qwen38-fidelity-report/1",
        "reference": str(args.reference), "candidate": str(args.candidate),
        "head": str(args.head), "head_sha256": sha256_file(Path(args.head)),
        "suite_token_sha256": suite["suite_token_sha256"],
        "contexts": len(per_ctx), "scored_positions": positions,
        "vocab_size": vocab, "hidden_size": hidden_size,
        "token_mean_kld": float(kl_all.mean()),
        "token_median_kld": q(0.5), "p95_kld": q(0.95), "p99_kld": q(0.99),
        "p999_kld": q(0.999), "max_kld": float(kl_all.max()),
        "context_macro_mean_kld": statistics.fmean(means),
        "context_bootstrap": bootstrap(means, clusters, args.bootstrap_samples, args.bootstrap_seed),
        "mean_jsd_bits": statistics.fmean(c["mean_jsd_bits"] for c in per_ctx),
        "top1_agreement": top1_hits / positions,
        "strata": {k: {"contexts": len(v), "mean_kld": statistics.fmean(v)}
                   for k, v in sorted(strata.items())},
        "worst_contexts": sorted(per_ctx, key=lambda c: -c["mean_kld"])[:20],
        "per_context": per_ctx,
        "comparator": {"vocab_chunk": args.vocab_chunk, "device": args.device,
                       "accumulation": "float64", "two_pass": True},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("replay_done " + json.dumps({k: report[k] for k in (
        "contexts", "scored_positions", "token_mean_kld", "context_macro_mean_kld",
        "top1_agreement", "p999_kld")}), flush=True)
    return 0


def cmd_paired(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    ai = {c["index"]: c for c in a["per_context"]}
    bi = {c["index"]: c for c in b["per_context"]}
    shared = sorted(set(ai) & set(bi))
    diffs = [ai[i]["mean_kld"] - bi[i]["mean_kld"] for i in shared]
    clusters = [ai[i]["source_cluster"] for i in shared]
    wins_a = sum(1 for d in diffs if d < 0)
    out = {
        "schema": "qwen38-fidelity-paired/1",
        "a": {"report": str(args.a), "label": args.a_label, "mean": a["context_macro_mean_kld"]},
        "b": {"report": str(args.b), "label": args.b_label, "mean": b["context_macro_mean_kld"]},
        "contexts": len(shared),
        "difference_a_minus_b": statistics.fmean(diffs),
        "median_difference": statistics.median(diffs),
        "bootstrap_difference": bootstrap(diffs, clusters, args.bootstrap_samples, 1),
        "a_wins": wins_a, "b_wins": len(shared) - wins_a,
        "largest_a_advantage": sorted(zip(shared, diffs), key=lambda x: x[1])[:10],
        "largest_b_advantage": sorted(zip(shared, diffs), key=lambda x: -x[1])[:10],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("paired_done " + json.dumps({k: out[k] for k in (
        "contexts", "difference_a_minus_b", "a_wins", "b_wins")}), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suite")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--contexts", type=int, default=128)
    s.add_argument("--context-length", type=int, default=2048)
    s.set_defaults(func=cmd_suite)

    c = sub.add_parser("capture")
    c.add_argument("--model", required=True)
    c.add_argument("--suite", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--quantization", default="auto")
    c.add_argument("--quantization-config", default=None)
    c.add_argument("--kv-cache-dtype", default="auto")
    c.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    c.set_defaults(func=cmd_capture)

    r = sub.add_parser("replay")
    r.add_argument("--reference", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--vocab-chunk", type=int, default=24832)
    r.add_argument("--device", default="cuda")
    r.add_argument("--bootstrap-samples", type=int, default=10000)
    r.add_argument("--bootstrap-seed", type=int, default=1)
    r.set_defaults(func=cmd_replay)

    q = sub.add_parser("paired")
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.add_argument("--a-label", default="a")
    q.add_argument("--b-label", default="b")
    q.add_argument("--out", required=True)
    q.add_argument("--bootstrap-samples", type=int, default=10000)
    q.set_defaults(func=cmd_paired)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
