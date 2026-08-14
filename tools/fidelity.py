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


def select(index: list[dict], flt: str) -> list[dict]:
    """Filter the context index by partition or sentinel flag."""
    if flt in ("all", "", None):
        return index
    if flt == "sentinel":
        return [c for c in index if c.get("sentinel")]
    return [c for c in index if c.get("partition", "analysis") == flt]


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
    store: dict = {"last": None, "rows": 0, "parts": [], "accumulate": False, "fp32": False}

    def hook(_m, _i, output):
        t = output[0] if isinstance(output, tuple) else output
        if t.dim() != 2:
            return output
        dtype = torch.float32 if store.get("fp32") else torch.bfloat16
        cpu = t.detach().to("cpu", dtype, copy=True)
        if store["accumulate"]:
            # Contexts longer than max_num_batched_tokens arrive as several prefill
            # chunks; concatenating in arrival order reconstructs the full sequence.
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


def _rpc_set_fp32(self, on: bool):
    store = getattr(self, "_fid_store", None)
    if store is not None:
        store["fp32"] = bool(on)
    return bool(on)


def _rpc_set_accumulate(self, on: bool):
    store = getattr(self, "_fid_store", None)
    if store is not None:
        store["accumulate"] = bool(on)
        store["parts"] = []
    return bool(on)


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
                  max_model_len=ctx_len + 64,
                  max_num_batched_tokens=args.max_batched_tokens or ctx_len,
                  max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
                  enforce_eager=True)
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        kwargs["quantization"] = args.quantization
    if args.quantization_config:
        kwargs["quantization_config"] = json.loads(args.quantization_config)
    llm = LLM(**kwargs)

    hooked = llm.collective_rpc(_rpc_install_hook)
    print(f"hooked {hooked}", flush=True)
    if args.fp32:
        llm.collective_rpc(_rpc_set_fp32, args=(True,))
        print("capturing hidden states in float32", flush=True)
    if args.chunk_accumulate:
        llm.collective_rpc(_rpc_set_accumulate, args=(True,))
        print("chunk accumulation enabled for long contexts", flush=True)

    params = SamplingParams(max_tokens=1, temperature=0, detokenize=False)
    done = 0
    t0 = time.time()
    records = []
    for ctx in select(manifest["context_index"], args.filter):
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
            print(f"{done} captured ({time.time() - t0:.0f}s)", flush=True)
    meta = {"model": args.model, "quantization": args.quantization,
            "quantization_config": args.quantization_config,
            "kv_cache_dtype": args.kv_cache_dtype,
            "suite_token_sha256": manifest["suite_token_sha256"],
            "contexts": done, "captures": records, "elapsed_sec": time.time() - t0}
    (out / "capture-manifest.json").write_text(json.dumps(meta, indent=2))
    print("capture_done " + json.dumps({k: meta[k] for k in ("contexts", "elapsed_sec")}), flush=True)
    return 0


def dense_prompt_logprobs(prompt_logprobs: object, npos: int, vocab: int) -> torch.Tensor:
    """Densify vLLM prompt logprobs to [npos, vocab]; verbatim from the published harness."""
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
                    vocab_chunk: int, cand_head: torch.Tensor | None = None):
    """Reference uses `head`; the candidate uses `cand_head` when given.

    Passing a different candidate head is how head quantization is attributed:
    the body captures stay byte-identical and only the head changes.
    """
    ch = head if cand_head is None else cand_head
    ref_z, ref_top = normalizers_and_top1(ref_h, head, vocab_chunk)
    cand_z, cand_top = normalizers_and_top1(cand_h, ch, vocab_chunk)
    rows = ref_h.shape[0]
    kl = torch.zeros(rows, dtype=torch.float64, device=ref_h.device)
    js = torch.zeros(rows, dtype=torch.float64, device=ref_h.device)
    ln2 = math.log(2.0)
    for start in range(0, head.shape[0], vocab_chunk):
        end = min(start + vocab_chunk, head.shape[0])
        rl = (ref_h @ head[start:end].T).float() - ref_z[:, None]
        cl = (cand_h @ ch[start:end].T).float() - cand_z[:, None]
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
    def load_head(path: str) -> torch.Tensor:
        with safe_open(path, framework="pt", device="cpu") as f:
            key = "weight" if "weight" in f.keys() else f.keys()[0]
            return f.get_tensor(key).to(dev, torch.bfloat16)

    hdtype = torch.float32 if args.fp32 else torch.bfloat16
    head = load_head(args.head).to(hdtype)
    cand_head = load_head(args.candidate_head).to(hdtype) if args.candidate_head else None
    vocab, hidden_size = head.shape
    per_ctx, strata, clusters, top1_hits, positions = [], {}, [], 0, 0
    all_kl = []

    for ctx in select(suite["context_index"], args.filter):
        i = ctx["index"]
        rp = Path(args.reference, f"hidden_{i:04d}.safetensors")
        cp = Path(args.candidate, f"hidden_{i:04d}.safetensors")
        if not (rp.exists() and cp.exists()):
            continue
        with safe_open(str(rp), framework="pt", device="cpu") as f:
            ref = f.get_tensor("hidden_states").to(dev, hdtype)
        with safe_open(str(cp), framework="pt", device="cpu") as f:
            cand = f.get_tensor("hidden_states").to(dev, hdtype)
        kl, js, hits = context_metrics(ref, cand, head, args.vocab_chunk, cand_head)
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
        "candidate_head": str(args.candidate_head) if args.candidate_head else None,
        "candidate_head_sha256": sha256_file(Path(args.candidate_head)) if args.candidate_head else None,
        "suite_token_sha256": suite["suite_token_sha256"],
        "filter": args.filter,
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


@torch.inference_mode()
def qualification_metrics(live_logprobs: torch.Tensor, hidden: torch.Tensor,
                          head: torch.Tensor, vocab_chunk: int):
    """KL(live served distribution || distribution replayed from hidden states).

    The live operand is already a normalised log-probability matrix, so it must NOT
    go through the head; only the replayed operand does. Two passes, as elsewhere:
    normaliser and argmax first, then the divergence.
    """
    rows = hidden.shape[0]
    dev = hidden.device
    log_z = torch.full((rows,), -math.inf, dtype=torch.float32, device=dev)
    rep_top = torch.zeros((rows,), dtype=torch.int64, device=dev)
    rep_val = torch.full((rows,), -math.inf, dtype=torch.float32, device=dev)
    for start in range(0, head.shape[0], vocab_chunk):
        end = min(start + vocab_chunk, head.shape[0])
        logits = (hidden @ head[start:end].T).float()
        log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
        val, idx = logits.max(dim=-1)
        upd = val > rep_val
        rep_val = torch.where(upd, val, rep_val)
        rep_top = torch.where(upd, idx + start, rep_top)
    live_top = live_logprobs.argmax(dim=-1)
    kl = torch.zeros(rows, dtype=torch.float64, device=dev)
    for start in range(0, head.shape[0], vocab_chunk):
        end = min(start + vocab_chunk, head.shape[0])
        rep = (hidden @ head[start:end].T).float() - log_z[:, None]
        liv = live_logprobs[:, start:end]
        kl += (liv.exp() * (liv - rep)).sum(-1).double()
    return kl.cpu(), int((live_top == rep_top).sum().item())


def cmd_qualify(args) -> int:
    """Live-logit qualification: does hidden-state replay reproduce served logits?

    The reference protocol reports mean KL(live || replayed) = 1.2e-6 on Kimi-K3.
    Without this check, a replay defect would be indistinguishable from candidate
    error. Runs the candidate twice over the same contexts: once through the
    hidden-state capture already on disk, once through vLLM's own full-vocabulary
    prompt logprobs.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from safetensors.torch import safe_open

    suite = json.loads(Path(args.suite, "suite-manifest.json").read_text())
    ctx_len = suite["context_length"]
    chosen = select(suite["context_index"], args.filter)[: args.contexts]
    dev = torch.device(args.device)
    hdtype = torch.float32 if args.fp32 else torch.bfloat16
    with safe_open(args.head, framework="pt", device="cpu") as f:
        key = "weight" if "weight" in f.keys() else f.keys()[0]
        head = f.get_tensor(key).to(dev, hdtype)
    vocab = head.shape[0]

    kwargs = dict(model=args.model, trust_remote_code=True, tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  kv_cache_memory_bytes=512 * 1024 * 1024, dtype="bfloat16",
                  kv_cache_dtype=args.kv_cache_dtype, load_format="safetensors",
                  max_model_len=ctx_len + 64, max_num_batched_tokens=256,
                  max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
                  enforce_eager=True, max_logprobs=-1)
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        kwargs["quantization"] = args.quantization
    if args.quantization_config:
        kwargs["quantization_config"] = json.loads(args.quantization_config)
    llm = LLM(**kwargs)
    params = SamplingParams(prompt_logprobs=-1, flat_logprobs=True, max_tokens=1,
                            detokenize=False)

    rows = []
    for ctx in chosen:
        i = ctx["index"]
        hp = Path(args.hidden, f"hidden_{i:04d}.safetensors")
        if not hp.exists():
            continue
        ids = json.loads(Path(args.suite, ctx["file"]).read_text())
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                           sampling_params=params, use_tqdm=False)[0]
        npos = ctx_len - 1
        live = dense_prompt_logprobs(out.prompt_logprobs, npos, vocab).to(dev)
        with safe_open(str(hp), framework="pt", device="cpu") as f:
            hidden = f.get_tensor("hidden_states").to(dev, hdtype)
        kl, hits = qualification_metrics(live, hidden, head, args.vocab_chunk)
        rows.append({"index": i, "mean_kld": float(kl.mean()), "max_kld": float(kl.max()),
                     "p999_kld": float(kl.double().quantile(0.999)),
                     "top1_agreement": hits / kl.numel()})
        print("qualify " + json.dumps(rows[-1]), flush=True)
        del live, hidden, kl

    if not rows:
        raise SystemExit("no contexts qualified; check --hidden and --filter")
    report = {
        "schema": "qwen38-replay-qualification/1",
        "model": args.model, "hidden": str(args.hidden), "head": str(args.head),
        "contexts": len(rows),
        "mean_kld_live_vs_replayed": statistics.fmean(r["mean_kld"] for r in rows),
        "max_kld": max(r["max_kld"] for r in rows),
        "p999_kld": max(r["p999_kld"] for r in rows),
        "top1_agreement": statistics.fmean(r["top1_agreement"] for r in rows),
        "per_context": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("qualify_done " + json.dumps({k: report[k] for k in (
        "contexts", "mean_kld_live_vs_replayed", "max_kld", "top1_agreement")}), flush=True)
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
    c.add_argument("--filter", default="all",
                   help="all | analysis | qualification | sentinel")
    c.add_argument("--max-batched-tokens", type=int, default=0,
                   help="prefill chunk size; 0 means one chunk per context")
    c.add_argument("--fp32", action="store_true",
                   help="store hidden states in float32 (2x disk, removes bf16 rounding)")
    c.add_argument("--chunk-accumulate", action="store_true",
                   help="concatenate chunked-prefill forwards (needed above one chunk)")
    c.set_defaults(func=cmd_capture)

    r = sub.add_parser("replay")
    r.add_argument("--reference", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--head", required=True, help="LM head for the reference operand")
    r.add_argument("--candidate-head", default=None,
                   help="different LM head for the candidate operand; attributes head error")
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--vocab-chunk", type=int, default=24832)
    r.add_argument("--device", default="cuda")
    r.add_argument("--bootstrap-samples", type=int, default=10000)
    r.add_argument("--bootstrap-seed", type=int, default=1)
    r.add_argument("--filter", default="all",
                   help="all | analysis | qualification | sentinel")
    r.add_argument("--fp32", action="store_true",
                   help="replay in float32 (matches --fp32 captures)")
    r.set_defaults(func=cmd_replay)

    z = sub.add_parser("qualify")
    z.add_argument("--model", required=True)
    z.add_argument("--suite", required=True)
    z.add_argument("--hidden", required=True, help="capture dir for the same model")
    z.add_argument("--head", required=True)
    z.add_argument("--out", required=True)
    z.add_argument("--contexts", type=int, default=8)
    z.add_argument("--filter", default="analysis")
    z.add_argument("--quantization", default="auto")
    z.add_argument("--quantization-config", default=None)
    z.add_argument("--kv-cache-dtype", default="auto")
    z.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    z.add_argument("--vocab-chunk", type=int, default=24832)
    z.add_argument("--device", default="cuda")
    z.add_argument("--fp32", action="store_true",
                   help="float32 hidden states and head (precision experiment)")
    z.set_defaults(func=cmd_qualify)

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
