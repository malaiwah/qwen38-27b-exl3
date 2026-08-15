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

New keys, all additive; every pre-existing key keeps its name and meaning:
  * `per_context_all` (replay report) - one row per scored context, complete and
    untruncated, so the headline is recomputable from the report alone: `index`,
    `source_cluster`, `stratum`, `positions_scored`, `mean_kld`, `median_kld`,
    `top1_agreement`. `worst_contexts` remains the truncated worst-20 view and
    `per_context` remains the richer per-context array that `paired` consumes.
  * `candidate_identity` (replay report and capture manifest) - what was measured,
    by content instead of by local path: `model_path`, `model_revision` (git HEAD
    of `<model>/.git` when the checkpoint is a checkout, else the first line of
    `revision.txt`), `model_revision_source` (`git_head` | `revision_txt` |
    `none`), `index_sha256` (sha256 of `model.safetensors.index.json` - the digest
    that pins a locally built quant, which has no revision of any kind),
    `shard_sha256` (shard filename -> sha256, null unless `--hash-shards` is
    passed, since it reads every byte of the checkpoint), `quantization`,
    `quantization_config`, `kv_cache_dtype_requested`, `kv_cache_dtype_resolved`,
    `kv_cache_dtype_resolved_source`. `capture` writes it from the live engine;
    `replay` carries the manifest block forward and recomputes the digests when
    the checkpoint is still reachable.
  * `kv_cache_dtype_requested` / `kv_cache_dtype_resolved` - the requested string
    is normally `auto`, which identifies nothing. The resolved value is read back
    from the engine's cache config after construction; `auto` there means the
    model dtype, which is vLLM's own rule, and `kv_cache_dtype_resolved_source`
    says which of the two produced the answer. `unresolved` means the engine
    exposed no cache config in this process - never a guessed dtype. The legacy
    manifest key `kv_cache_dtype` keeps recording the requested string.
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

def require_local_model(path: str) -> Path:
    """Refuse mutable Hub IDs; callers must download and review a snapshot first."""
    root = Path(path).expanduser()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise SystemExit(
            "--model must be a local snapshot containing config.json; download an immutable "
            "revision before running the fidelity harness"
        )
    return root


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a JSON receipt atomically, so an interrupted run cannot bless partial work."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def canonical_sha256(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def git_head(repo: Path) -> str | None:
    """HEAD commit of a checkout, without shelling out to git.

    Hub snapshot directories are git clones, so their HEAD is the model revision.
    `.git` is a directory for a normal clone and a `gitdir:` pointer file for a
    worktree or submodule, and the branch ref may be loose or packed.
    """
    dot = repo / ".git"
    if dot.is_file():
        line = dot.read_text().strip()
        if not line.startswith("gitdir:"):
            return None
        gitdir = repo / line.split(":", 1)[1].strip()
    elif dot.is_dir():
        gitdir = dot
    else:
        return None
    head = gitdir / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text().strip()
    if not text.startswith("ref:"):
        return text or None  # detached HEAD holds the commit itself
    ref = text.split(":", 1)[1].strip()
    loose = gitdir / ref
    if loose.is_file():
        return loose.read_text().strip() or None
    packed = gitdir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line[:1] in ("#", "^"):
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def model_identity(model: str, hash_shards: bool) -> dict:
    """Content identity of a checkpoint directory.

    A path is not an identity: it can be rebuilt or moved under the same name.
    Revisions identify immutable Hub snapshots. An index digest identifies tensor
    layout but not shard payload bytes; locally built quants therefore require
    `shard_sha256` for a content identity.
    """
    p = Path(model)
    revision = git_head(p) if p.is_dir() else None
    source = "git_head"
    if revision is None:
        txt = p / "revision.txt"
        revision = txt.read_text().split("\n", 1)[0].strip() if txt.is_file() else ""
        source = "revision_txt" if revision else "none"
    index = p / "model.safetensors.index.json"
    config = p / "config.json"
    return {
        "model_path": str(p.resolve()) if p.exists() else str(p),
        "model_revision": revision or None,
        "model_revision_source": source,
        "index_sha256": sha256_file(index) if index.is_file() else None,
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "shard_sha256": {f.name: sha256_file(f) for f in sorted(p.glob("*.safetensors"))}
                        if hash_shards and p.is_dir() else None,
    }


def resolved_kv_cache_dtype(llm) -> tuple[str, str]:
    """What the engine stores KV in, and where that answer came from.

    An explicit `cache_dtype` (`fp8`, `fp8_e4m3`, ...) is the answer directly;
    `auto` means "use the model dtype", so the model dtype is reported and the
    source field says which one it was. vLLM keeps the cache config on the engine
    and on `vllm_config` and has moved it between the two, so both chains are
    tried; an engine that only holds a handle to a separate core process exposes
    neither, and then the dtype is `unresolved` rather than a guess.
    """
    engine = getattr(llm, "llm_engine", llm)
    config = getattr(engine, "vllm_config", None)
    cache = getattr(engine, "cache_config", None) or getattr(config, "cache_config", None)
    requested = getattr(cache, "cache_dtype", None)
    if isinstance(requested, str) and requested not in ("", "auto"):
        return requested, "cache_config.cache_dtype"
    model = getattr(engine, "model_config", None) or getattr(config, "model_config", None)
    dtype = getattr(model, "dtype", None)
    if requested == "auto" and dtype is not None:
        return str(dtype).removeprefix("torch."), "model_config.dtype"
    return "unresolved", "unavailable"


def capture_identity(capture: str, hash_shards: bool) -> dict:
    """`candidate_identity` for an existing capture directory.

    Engine-side facts can only come from the capture manifest, because only the
    capture process saw the engine. Digests are recomputed here when the
    checkpoint is still reachable, so a replay can be re-verified against the
    checkpoint it names, and are otherwise carried forward from the manifest.
    Captures written before the identity block existed keep null fields instead of
    invented ones.
    """
    path = Path(capture, "capture-manifest.json")
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    identity = dict(manifest.get("candidate_identity") or {})
    model = identity.get("model_path") or manifest.get("model")
    if model and Path(model).is_dir():
        fresh = model_identity(model, hash_shards)
        checked = ["model_revision", "index_sha256", "config_sha256"]
        if hash_shards:
            checked.append("shard_sha256")
        for key in checked:
            if key not in identity:
                raise SystemExit(
                    f"capture identity lacks {key}; cannot relabel stored hidden states"
                )
            if identity[key] != fresh[key]:
                raise SystemExit(
                    f"capture checkpoint identity changed for {key}; stored hidden "
                    "states do not describe the current model path"
                )
    identity.setdefault("model_path", model)
    identity.setdefault("quantization", manifest.get("quantization"))
    identity.setdefault("quantization_config", manifest.get("quantization_config"))
    identity.setdefault("trust_remote_code", manifest.get("trust_remote_code"))
    identity.setdefault("kv_cache_dtype_requested", manifest.get("kv_cache_dtype"))
    for key, unknown in (("model_revision", None), ("model_revision_source", "none"),
                         ("index_sha256", None), ("config_sha256", None),
                         ("shard_sha256", None), ("trust_remote_code", None),
                         ("kv_cache_dtype_resolved", "unresolved"),
                         ("kv_cache_dtype_resolved_source", "unavailable")):
        identity.setdefault(key, unknown)
    return identity


def capture_contract(
    identity: dict,
    suite_token_sha256: str,
    context_length: int,
    selected: list[dict],
    args,
) -> dict:
    """Facts that must be identical before capture files may be reused."""
    quant_config = (
        json.loads(args.quantization_config) if args.quantization_config else None
    )
    return {
        "suite_token_sha256": suite_token_sha256,
        "context_length": context_length,
        "filter": args.filter,
        "expected_indices": [int(c["index"]) for c in selected],
        "candidate_content": {
            "model_revision": identity.get("model_revision"),
            "index_sha256": identity.get("index_sha256"),
            "config_sha256": identity.get("config_sha256"),
            "shard_sha256": identity.get("shard_sha256"),
        },
        "runtime": {
            "quantization": args.quantization,
            "quantization_config": quant_config,
            "trust_remote_code": bool(args.trust_remote_code),
            "kv_cache_dtype_requested": args.kv_cache_dtype,
            "kv_cache_dtype_resolved": identity.get("kv_cache_dtype_resolved"),
            "fp32": bool(args.fp32),
            "chunk_accumulate": bool(args.chunk_accumulate),
            "max_batched_tokens": int(args.max_batched_tokens),
        },
    }


def validate_capture_files(
    capture: Path,
    suite_token_sha256: str,
    selected_indices: set[int],
) -> dict:
    """Fail closed on incomplete, stale, duplicated, or modified capture sets."""
    path = capture / "capture-manifest.json"
    if not path.is_file():
        raise SystemExit(f"missing capture manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True:
        raise SystemExit(f"incomplete capture manifest: {path}")
    if manifest.get("suite_token_sha256") != suite_token_sha256:
        raise SystemExit(f"suite identity mismatch in {path}")
    records = manifest.get("captures")
    if not isinstance(records, list):
        raise SystemExit(f"invalid captures array in {path}")
    by_index = {}
    for record in records:
        index = record.get("index")
        if not isinstance(index, int) or index in by_index:
            raise SystemExit(f"invalid or duplicate capture index {index!r} in {path}")
        by_index[index] = record
    if manifest.get("contexts") != len(by_index):
        raise SystemExit(
            f"incomplete capture manifest {path}: contexts={manifest.get('contexts')} "
            f"but records={len(by_index)}"
        )
    missing = sorted(selected_indices - set(by_index))
    if missing:
        raise SystemExit(f"capture manifest {path} is missing indices {missing[:16]}")
    for index in sorted(selected_indices):
        file = capture / f"hidden_{index:04d}.safetensors"
        if not file.is_file():
            raise SystemExit(f"capture file missing: {file}")
        expected = by_index[index].get("sha256")
        if not expected or sha256_file(file) != expected:
            raise SystemExit(f"capture digest mismatch: {file}")
    return manifest


# ------------------------------------------------------------------ suite
def cmd_suite(args) -> int:
    model_root = require_local_model(args.model)
    tokenizer_file = model_root / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise SystemExit("local model snapshot is missing tokenizer.json")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
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
        "model": args.model, "trust_remote_code": args.trust_remote_code,
        "context_length": ctx_len,
        "scored_positions_per_context": ctx_len - 1,
        "contexts": len(contexts), "total_scored_positions": len(contexts) * (ctx_len - 1),
        "tokenizer_sha256": sha256_file(tokenizer_file),
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
    require_local_model(args.model)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from safetensors import safe_open
    from safetensors.torch import save_file

    suite = Path(args.suite)
    suite_manifest = json.loads((suite / "suite-manifest.json").read_text())
    ctx_len = suite_manifest["context_length"]
    selected = select(suite_manifest["context_index"], args.filter)
    if not selected:
        raise SystemExit(f"capture filter {args.filter!r} selected no contexts")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "capture-manifest.json"

    kwargs = dict(model=args.model, trust_remote_code=args.trust_remote_code,
                  tensor_parallel_size=1,
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

    identity = model_identity(args.model, args.hash_shards)
    resolved, resolved_source = resolved_kv_cache_dtype(llm)
    identity.update({"quantization": args.quantization,
                     "quantization_config": args.quantization_config,
                     "trust_remote_code": args.trust_remote_code,
                     "kv_cache_dtype_requested": args.kv_cache_dtype,
                     "kv_cache_dtype_resolved": resolved,
                     "kv_cache_dtype_resolved_source": resolved_source})
    if not identity.get("model_revision") and not identity.get("shard_sha256"):
        raise SystemExit(
            "candidate payload is unpinned: use a revisioned checkout or leave "
            "--hash-shards enabled"
        )
    print("candidate_identity " + json.dumps({k: identity[k] for k in (
        "model_revision", "index_sha256", "kv_cache_dtype_resolved")}), flush=True)

    contract = capture_contract(
        identity, suite_manifest["suite_token_sha256"], ctx_len, selected, args
    )
    contract_digest = canonical_sha256(contract)
    expected_indices = set(contract["expected_indices"])
    existing_files = {
        int(path.stem.removeprefix("hidden_")): path
        for path in out.glob("hidden_*.safetensors")
        if path.stem.removeprefix("hidden_").isdigit()
    }
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if prior is None and existing_files:
        raise SystemExit(
            f"{out} has capture files but no manifest; refusing an unverifiable resume"
        )
    if prior is not None and prior.get("capture_contract_sha256") != contract_digest:
        raise SystemExit(
            f"capture contract changed in {out}; use a new output directory"
        )
    unexpected = sorted(set(existing_files) - expected_indices)
    if unexpected:
        raise SystemExit(f"{out} contains unselected capture indices {unexpected[:16]}")

    records_by_index = {}
    for record in (prior or {}).get("captures", []):
        index = record.get("index")
        if not isinstance(index, int) or index in records_by_index:
            raise SystemExit(f"invalid or duplicate capture record {index!r} in {out}")
        records_by_index[index] = record
    if set(records_by_index) != set(existing_files):
        raise SystemExit(
            f"capture files and manifest records disagree in {out}; refusing resume"
        )
    for index, path in existing_files.items():
        record = records_by_index[index]
        if sha256_file(path) != record.get("sha256"):
            raise SystemExit(f"existing capture digest mismatch: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as file:
            if "hidden_states" not in file.keys():
                raise SystemExit(f"hidden_states missing from {path}")
            shape = list(file.get_slice("hidden_states").get_shape())
        if shape != record.get("shape") or shape[0] != ctx_len - 1:
            raise SystemExit(f"existing capture shape mismatch: {path}: {shape}")

    hooked = llm.collective_rpc(_rpc_install_hook)
    print(f"hooked {hooked}", flush=True)
    if args.fp32:
        llm.collective_rpc(_rpc_set_fp32, args=(True,))
        print("capturing hidden states in float32", flush=True)
    if args.chunk_accumulate:
        llm.collective_rpc(_rpc_set_accumulate, args=(True,))
        print("chunk accumulation enabled for long contexts", flush=True)

    params = SamplingParams(max_tokens=1, temperature=0, detokenize=False)
    started = time.time()

    def write_manifest(complete: bool) -> None:
        records = [records_by_index[i] for i in sorted(records_by_index)]
        meta = {
            "schema": "qwen38-fidelity-capture/2",
            "model": args.model,
            "quantization": args.quantization,
            "quantization_config": args.quantization_config,
            "kv_cache_dtype": args.kv_cache_dtype,
            "candidate_identity": identity,
            "suite_token_sha256": suite_manifest["suite_token_sha256"],
            "filter": args.filter,
            "context_length": ctx_len,
            "expected_contexts": len(selected),
            "expected_indices": sorted(expected_indices),
            "capture_contract": contract,
            "capture_contract_sha256": contract_digest,
            "contexts": len(records),
            "captures": records,
            "complete": complete,
            "elapsed_sec": time.time() - started,
        }
        atomic_write_json(manifest_path, meta)

    write_manifest(complete=False)
    for ctx in selected:
        index = ctx["index"]
        dst = out / f"hidden_{index:04d}.safetensors"
        if index in records_by_index:
            continue
        ids = json.loads((suite / ctx["file"]).read_text())
        if sha256_bytes(json.dumps(ids).encode()) != ctx["token_sha256"]:
            raise SystemExit(f"token hash drift on context {index}")
        llm.collective_rpc(_rpc_pop_capture)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params=params,
                     use_tqdm=False)
        got = llm.collective_rpc(_rpc_pop_capture)[0]
        if got is None or got.shape[0] != ctx_len:
            raise SystemExit(f"capture failed for context {index}: "
                             f"{None if got is None else tuple(got.shape)}")
        hidden = got[: ctx_len - 1].contiguous()
        tmp = dst.with_name(dst.name + ".tmp")
        save_file({"hidden_states": hidden}, str(tmp))
        tmp.replace(dst)
        records_by_index[index] = {
            "index": index,
            "sha256": sha256_file(dst),
            "shape": list(hidden.shape),
        }
        write_manifest(complete=False)
        if len(records_by_index) % 16 == 0:
            print(
                f"{len(records_by_index)} captured ({time.time() - started:.0f}s)",
                flush=True,
            )
    if set(records_by_index) != expected_indices:
        raise SystemExit("capture ended without the exact selected context set")
    write_manifest(complete=True)
    print("capture_done " + json.dumps({
        "contexts": len(records_by_index), "elapsed_sec": time.time() - started
    }), flush=True)
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
    per_ctx, rows_all, strata, clusters = [], [], {}, []
    top1_hits, positions = 0, 0
    all_kl = []

    chosen = select(suite["context_index"], args.filter)
    if not chosen:
        raise SystemExit(f"replay filter {args.filter!r} selected no contexts")
    selected_indices = {int(ctx["index"]) for ctx in chosen}
    validate_capture_files(
        Path(args.reference), suite["suite_token_sha256"], selected_indices
    )
    validate_capture_files(
        Path(args.candidate), suite["suite_token_sha256"], selected_indices
    )

    for ctx in chosen:
        i = ctx["index"]
        rp = Path(args.reference, f"hidden_{i:04d}.safetensors")
        cp = Path(args.candidate, f"hidden_{i:04d}.safetensors")
        with safe_open(str(rp), framework="pt", device="cpu") as f:
            ref = f.get_tensor("hidden_states").to(dev, hdtype)
        with safe_open(str(cp), framework="pt", device="cpu") as f:
            cand = f.get_tensor("hidden_states").to(dev, hdtype)
        kl, js, hits = context_metrics(ref, cand, head, args.vocab_chunk, cand_head)
        npos = kl.numel()
        m, med, agree = float(kl.mean()), float(kl.median()), hits / npos
        per_ctx.append({"index": i, "stratum": ctx["stratum"],
                        "source_cluster": ctx["source_cluster"],
                        "mean_kld": m, "median_kld": med,
                        "max_kld": float(kl.max()), "mean_jsd_bits": float(js.mean()),
                        "top1_agreement": agree})
        rows_all.append({"index": i, "source_cluster": ctx["source_cluster"],
                         "stratum": ctx["stratum"], "positions_scored": npos,
                         "mean_kld": m, "median_kld": med, "top1_agreement": agree})
        strata.setdefault(ctx["stratum"], []).append(m)
        clusters.append(ctx["source_cluster"])
        top1_hits += hits
        positions += npos
        all_kl.append(kl)
        if len(per_ctx) % 16 == 0:
            print(f"{len(per_ctx)} contexts, running mean {statistics.fmean(x['mean_kld'] for x in per_ctx):.6f}", flush=True)
    kl_all = torch.cat(all_kl).double()
    q = lambda p: float(kl_all.quantile(p))
    means = [c["mean_kld"] for c in per_ctx]
    report = {
        "schema": "qwen38-fidelity-report/1",
        "reference": str(args.reference), "candidate": str(args.candidate),
        "reference_identity": capture_identity(args.reference, args.hash_shards),
        "candidate_identity": capture_identity(args.candidate, args.hash_shards),
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
        "per_context_all": rows_all,
        "per_context": per_ctx,
        "comparator": {"vocab_chunk": args.vocab_chunk, "device": args.device,
                       "accumulation": "float64", "two_pass": True},
    }
    atomic_write_json(Path(args.out), report)
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
    require_local_model(args.model)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from safetensors.torch import safe_open

    suite = json.loads(Path(args.suite, "suite-manifest.json").read_text())
    ctx_len = suite["context_length"]
    chosen = select(suite["context_index"], args.filter)[: args.contexts]
    if len(chosen) != args.contexts:
        raise SystemExit(
            f"qualification requested {args.contexts} contexts but selected {len(chosen)}"
        )
    hidden_manifest = validate_capture_files(
        Path(args.hidden), suite["suite_token_sha256"],
        {int(ctx["index"]) for ctx in chosen},
    )
    hidden_identity = capture_identity(args.hidden, hash_shards=False)
    live_identity = model_identity(args.model, hash_shards=True)
    for key in ("model_revision", "index_sha256", "config_sha256", "shard_sha256"):
        if hidden_identity.get(key) != live_identity.get(key):
            raise SystemExit(
                f"qualification model does not match hidden capture identity: {key}"
            )
    captured_runtime = (hidden_manifest.get("capture_contract") or {}).get("runtime") or {}
    requested_quant_config = (
        json.loads(args.quantization_config) if args.quantization_config else None
    )
    expected_runtime = {
        "quantization": args.quantization,
        "quantization_config": requested_quant_config,
        "trust_remote_code": bool(args.trust_remote_code),
        "kv_cache_dtype_requested": args.kv_cache_dtype,
        "fp32": bool(args.fp32),
    }
    for key, value in expected_runtime.items():
        if captured_runtime.get(key) != value:
            raise SystemExit(
                f"qualification runtime does not match hidden capture: {key}"
            )
    dev = torch.device(args.device)
    hdtype = torch.float32 if args.fp32 else torch.bfloat16
    with safe_open(args.head, framework="pt", device="cpu") as f:
        key = "weight" if "weight" in f.keys() else f.keys()[0]
        head = f.get_tensor(key).to(dev, hdtype)
    vocab = head.shape[0]

    kwargs = dict(model=args.model, trust_remote_code=args.trust_remote_code,
                  tensor_parallel_size=1,
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

    if not chosen:
        raise SystemExit("qualification selected no contexts")
    rows = []
    for ctx in chosen:
        i = ctx["index"]
        hp = Path(args.hidden, f"hidden_{i:04d}.safetensors")
        ids = json.loads(Path(args.suite, ctx["file"]).read_text())
        if sha256_bytes(json.dumps(ids).encode()) != ctx["token_sha256"]:
            raise SystemExit(f"token hash drift on context {i}")
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
        "schema": "qwen38-replay-qualification/2",
        "model": args.model,
        "candidate_identity": live_identity,
        "hidden": str(args.hidden),
        "capture_manifest_sha256": sha256_file(Path(args.hidden, "capture-manifest.json")),
        "head": str(args.head),
        "head_sha256": sha256_file(Path(args.head)),
        "suite_token_sha256": suite["suite_token_sha256"],
        "filter": args.filter,
        "runtime": expected_runtime,
        "contexts": len(rows),
        "mean_kld_live_vs_replayed": statistics.fmean(r["mean_kld"] for r in rows),
        "max_kld": max(r["max_kld"] for r in rows),
        "p999_kld": max(r["p999_kld"] for r in rows),
        "top1_agreement": statistics.fmean(r["top1_agreement"] for r in rows),
        "per_context": rows,
    }
    atomic_write_json(Path(args.out), report)
    print("qualify_done " + json.dumps({k: report[k] for k in (
        "contexts", "mean_kld_live_vs_replayed", "max_kld", "top1_agreement")}), flush=True)
    return 0


def cmd_paired(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    for field in ("suite_token_sha256", "filter", "head_sha256"):
        if a.get(field) != b.get(field):
            raise SystemExit(
                f"paired reports disagree on {field}: {a.get(field)!r} != {b.get(field)!r}"
            )
    ai = {c["index"]: c for c in a["per_context"]}
    bi = {c["index"]: c for c in b["per_context"]}
    if len(ai) != len(a["per_context"]) or len(bi) != len(b["per_context"]):
        raise SystemExit("paired report contains duplicate context indices")
    if set(ai) != set(bi):
        raise SystemExit(
            "paired reports must contain the same context set; "
            f"a_only={sorted(set(ai) - set(bi))[:16]}, "
            f"b_only={sorted(set(bi) - set(ai))[:16]}"
        )
    shared = sorted(ai)
    for index in shared:
        if ai[index].get("source_cluster") != bi[index].get("source_cluster"):
            raise SystemExit(f"source cluster mismatch at context {index}")
    if not shared:
        raise SystemExit("paired reports contain no contexts")
    diffs = [ai[i]["mean_kld"] - bi[i]["mean_kld"] for i in shared]
    clusters = [ai[i]["source_cluster"] for i in shared]
    wins_a = sum(1 for d in diffs if d < 0)
    out = {
        "schema": "qwen38-fidelity-paired/2",
        "suite_token_sha256": a["suite_token_sha256"],
        "filter": a["filter"],
        "head_sha256": a["head_sha256"],
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
    atomic_write_json(Path(args.out), out)
    print("paired_done " + json.dumps({k: out[k] for k in (
        "contexts", "difference_a_minus_b", "a_wins", "b_wins")}), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suite")
    s.add_argument("--model", required=True)
    s.add_argument("--trust-remote-code", action="store_true",
                   help="execute model-repository Python (unsafe; off by default)")
    s.add_argument("--out", required=True)
    s.add_argument("--contexts", type=int, default=128)
    s.add_argument("--context-length", type=int, default=2048)
    s.set_defaults(func=cmd_suite)

    c = sub.add_parser("capture")
    c.add_argument("--model", required=True)
    c.add_argument("--suite", required=True)
    c.add_argument("--trust-remote-code", action="store_true",
                   help="execute model-repository Python (unsafe; off by default)")
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
    c.add_argument("--hash-shards", action=argparse.BooleanOptionalAction, default=True,
                   help="sha256 every weight shard for content identity (default: true)")
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
    r.add_argument("--hash-shards", action=argparse.BooleanOptionalAction, default=True,
                   help="recheck every candidate weight shard (default: true)")
    r.set_defaults(func=cmd_replay)

    z = sub.add_parser("qualify")
    z.add_argument("--model", required=True)
    z.add_argument("--suite", required=True)
    z.add_argument("--hidden", required=True, help="capture dir for the same model")
    z.add_argument("--trust-remote-code", action="store_true",
                   help="execute model-repository Python (unsafe; off by default)")
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
