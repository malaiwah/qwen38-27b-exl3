#!/usr/bin/env python3
"""Turn a `gguf_capture` output directory into a capture manifest `fidelity.py` trusts.

`fidelity.py capture` writes `capture-manifest.json` itself, because it is the
process that saw the engine.  The GGUF path splits that in two: the C++ tool
writes the hidden states plus `capture-engine.json` (what the engine was and what
it consumed), and this script turns that into the `qwen38-fidelity-capture/2`
manifest - per-file sha256 and shape, the suite token digest, the expected index
set, an engine-identity block and `complete` - and then re-verifies the result by
calling `fidelity.validate_capture_files`, the exact function `replay` calls.  If
this script exits 0, replay cannot reject the directory for manifest reasons.

Everything is checked against a second source rather than copied:

  * every hidden file's safetensors header is parsed (dtype, shape, offsets) and
    its length is checked against the file size, so a truncated capture cannot be
    blessed;
  * each context's token digest recorded by the capture tool is compared with the
    suite manifest's own `token_sha256`, which is what proves the engine consumed
    the suite's tokens and not something adjacent;
  * the GGUF's sha256 and byte size are recomputed here and compared with the
    size the engine recorded (and with `--gguf-sha256` when a published digest is
    being asserted);
  * the capture binary's sha256 is compared with `llamacpp-provenance.json`, and
    the commit recorded in the capture must equal the commit that provenance was
    built from.

    gguf_manifest.py build  --capture CAP_DIR --suite SUITE_DIR \\
                            --provenance /var/tmp/work/llamacpp/llamacpp-provenance.json \\
                            [--gguf-revision REV] [--gguf-sha256 HEX] \\
                            [--expect-hidden 5120] [--filter all]
    gguf_manifest.py verify --capture CAP_DIR --suite SUITE_DIR [--filter all]
    gguf_manifest.py smoke-fixture --out SUITE_DIR --head HEAD.safetensors \\
                            --contexts 2 --context-length 128 --hidden 1024 --vocab 4096

`smoke-fixture` exists so the whole path can be proven without the 20-29 GB
comparators: it writes a synthetic suite (random token ids, real suite manifest
schema) and a random BF16 head of the smoke model's hidden width, which is what
makes a self-replay - candidate against itself, mean KLD exactly 0 - possible on
a model that is not 5120-wide.  It is a format fixture and nothing else: no
number measured through it says anything about a quant.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE_SCHEMA = "qwen38-fidelity-capture/2"
ENGINE_SCHEMA = "qwen38-gguf-capture-engine/1"
SUITE_SCHEMA_FAMILY = "qwen38-distribution-fidelity/"


def load_fidelity():
    """Import the harness itself, so validation here is validation there.

    `fidelity.py` only imports the standard library at module scope, so this
    costs nothing and needs no torch.
    """
    spec = importlib.util.spec_from_file_location("fidelity", HERE / "fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fid = load_fidelity()


def die(msg: str) -> "NoReturn":  # noqa: F821
    raise SystemExit(f"gguf_manifest: {msg}")


def safetensors_header(path: Path) -> tuple[str, list[int], int]:
    """`(dtype, shape, payload_bytes)` from the header, with the file size checked.

    Reading the header instead of the tensor keeps this dependency-free and still
    catches the failure that matters: a file whose declared payload does not fit
    in the bytes on disk.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            die(f"{path} is too short to be safetensors")
        header_len = struct.unpack("<Q", raw)[0]
        header = f.read(header_len)
    if len(header) != header_len:
        die(f"{path} header is truncated")
    meta = json.loads(header)
    meta.pop("__metadata__", None)
    if list(meta) != ["hidden_states"]:
        die(f"{path} holds {sorted(meta)}, expected exactly ['hidden_states']")
    entry = meta["hidden_states"]
    start, end = entry["data_offsets"]
    payload = end - start
    if start != 0:
        die(f"{path} does not start its tensor at offset 0")
    if 8 + header_len + payload != size:
        die(f"{path} declares {payload} payload bytes but the file holds "
            f"{size - 8 - header_len}")
    return entry["dtype"], list(entry["shape"]), payload


def gguf_identity(path: Path, expect_sha: str | None, engine: dict) -> dict:
    size = path.stat().st_size
    if size != engine["model_size_bytes"]:
        die(f"{path} is {size} B now but was {engine['model_size_bytes']} B during capture")
    digest = fid.sha256_file(path)
    if expect_sha and digest != expect_sha:
        die(f"{path} sha256 {digest} != --gguf-sha256 {expect_sha}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def check_binary(engine: dict, provenance: Path | None, binary: Path | None) -> dict:
    """Which binary produced these states, by content."""
    identity = {"llamacpp_commit": engine["llamacpp_commit"],
                "binary_path": str(binary) if binary else None,
                "binary_sha256": None,
                "provenance": str(provenance) if provenance else None,
                "provenance_verified": False}
    if provenance is not None:
        prov = json.loads(provenance.read_text())
        if prov.get("commit") != engine["llamacpp_commit"]:
            die(f"capture used commit {engine['llamacpp_commit']} but {provenance} "
                f"describes {prov.get('commit')}")
        artifact = (prov.get("artifacts") or {}).get("gguf_capture")
        if not artifact:
            die(f"{provenance} records no gguf_capture artifact")
        identity["binary_path"] = identity["binary_path"] or artifact["path"]
        identity["binary_sha256"] = artifact["sha256"]
        identity["llamacpp_source_digest"] = (prov.get("source_digest") or {}).get("value")
        identity["cuda_architectures"] = prov.get("cuda_architectures")
        identity["capture_tool_source_sha256"] = (
            prov.get("capture_tool_source") or {}).get("sha256")
        if binary is not None:
            live = fid.sha256_file(binary)
            if live != artifact["sha256"]:
                die(f"{binary} sha256 {live} != provenance {artifact['sha256']}; "
                    "rebuild or point --binary at the built tool")
        identity["provenance_verified"] = True
    elif binary is not None:
        identity["binary_sha256"] = fid.sha256_file(binary)
    return identity


def read_inputs(args) -> tuple[Path, dict, dict, list[dict], dict]:
    capture = Path(args.capture)
    engine_path = capture / "capture-engine.json"
    if not engine_path.is_file():
        die(f"missing {engine_path}; run gguf_capture first")
    engine = json.loads(engine_path.read_text())
    if engine.get("schema") != ENGINE_SCHEMA:
        die(f"{engine_path} schema {engine.get('schema')!r} != {ENGINE_SCHEMA!r}")

    suite_path = Path(args.suite, "suite-manifest.json")
    suite = json.loads(suite_path.read_text())
    if not str(suite.get("schema", "")).startswith(SUITE_SCHEMA_FAMILY):
        die(f"{suite_path} schema {suite.get('schema')!r} is not a {SUITE_SCHEMA_FAMILY}N suite")
    # Suites from v5 onward declare the width they were built for; when they do it
    # decides, not a command-line flag anyone can mistype.
    if suite.get("hidden_size") and engine["n_embd_out"] != suite["hidden_size"]:
        die(f"capture width {engine['n_embd_out']} != suite hidden_size {suite['hidden_size']}")
    if engine.get("suite_context_length") != suite["context_length"]:
        die(f"capture used context_length {engine.get('suite_context_length')}, suite "
            f"says {suite['context_length']}")
    if engine.get("suite_token_sha256") not in ("", None) and \
            engine["suite_token_sha256"] != suite["suite_token_sha256"]:
        die("capture recorded a different suite_token_sha256 than the suite manifest")

    selected = fid.select(suite["context_index"], args.filter)
    if not selected:
        die(f"filter {args.filter!r} selected no contexts")
    return capture, engine, suite, selected, {c["index"]: c for c in suite["context_index"]}


def cmd_build(args) -> int:
    capture, engine, suite, selected, by_suite_index = read_inputs(args)
    ctx_len = suite["context_length"]
    captured = {c["index"]: c for c in engine.get("captures", [])}
    if len(captured) != len(engine.get("captures", [])):
        die("capture-engine.json repeats a context index")

    selected_indices = [int(c["index"]) for c in selected]
    extra = sorted(set(captured) - set(selected_indices))
    if extra:
        die(f"{capture} holds unselected indices {extra[:16]}; filter or directory is wrong")
    missing = sorted(set(selected_indices) - set(captured))
    if missing and not args.allow_incomplete:
        die(f"{capture} is missing {len(missing)} contexts, first {missing[:16]}; "
            "capture them or pass --allow-incomplete")

    width = engine["n_embd_out"]
    if args.expect_hidden and width != args.expect_hidden:
        die(f"capture width {width} != --expect-hidden {args.expect_hidden}")

    records = []
    for index in sorted(captured):
        entry = captured[index]
        suite_ctx = by_suite_index.get(index)
        if suite_ctx is None:
            die(f"suite has no context {index}")
        if entry["tokens"] != suite_ctx.get("tokens", ctx_len) or entry["tokens"] != ctx_len:
            die(f"context {index}: capture consumed {entry['tokens']} tokens, suite "
                f"declares {suite_ctx.get('tokens')} with context_length {ctx_len}")
        if entry["token_sha256"] != suite_ctx["token_sha256"]:
            die(f"context {index}: capture token digest {entry['token_sha256']} != suite "
                f"{suite_ctx['token_sha256']}")
        if entry["token_file"] != suite_ctx["file"]:
            die(f"context {index}: capture read {entry['token_file']}, suite names "
                f"{suite_ctx['file']}")
        path = capture / f"hidden_{index:04d}.safetensors"
        if entry["file"] != path.name:
            die(f"context {index}: capture wrote {entry['file']}, expected {path.name}")
        if not path.is_file():
            die(f"missing capture file {path}")
        dtype, shape, _ = safetensors_header(path)
        if dtype != "BF16":
            die(f"{path} is {dtype}; the protocol replays bfloat16 hidden states")
        if shape != [ctx_len - 1, width]:
            die(f"{path} shape {shape} != [{ctx_len - 1}, {width}]")
        if shape != list(entry["shape"]):
            die(f"{path} shape {shape} != capture record {entry['shape']}")
        records.append({"index": index, "sha256": fid.sha256_file(path), "shape": shape})

    gguf = gguf_identity(Path(engine["model_path"]), args.gguf_sha256, engine)
    binary = check_binary(engine, Path(args.provenance) if args.provenance else None,
                          Path(args.binary) if args.binary else None)
    # `general.file_type` is the numeric ftype enum; `model_desc` is
    # "<arch> <size> <ftype name>", which is the readable label.  Marketing names
    # like `UD-Q5_K_XL` live in the repo filename, not in the file, so they can
    # only arrive through --quantization.
    quantization = args.quantization or engine["model_desc"]

    engine_identity = {
        "engine": "llama.cpp",
        "llamacpp_commit": binary["llamacpp_commit"],
        "llamacpp_source_digest": binary.get("llamacpp_source_digest"),
        "binary_path": binary["binary_path"],
        "binary_sha256": binary["binary_sha256"],
        "binary_provenance": binary["provenance"],
        "binary_provenance_verified": binary["provenance_verified"],
        "capture_tool_source_sha256": binary.get("capture_tool_source_sha256"),
        "cuda_architectures": binary.get("cuda_architectures"),
        "gguf_path": gguf["path"],
        "gguf_sha256": gguf["sha256"],
        "gguf_bytes": gguf["bytes"],
        "gguf_revision": args.gguf_revision,
        "quantization_label": quantization,
        "model_desc": engine["model_desc"],
        "architecture": engine["architecture"],
        "general_file_type": engine.get("general_file_type"),
        "n_embd": engine["n_embd"],
        "n_embd_out": engine["n_embd_out"],
        "n_ctx": engine["n_ctx"],
        "n_batch": engine["n_batch"],
        "n_ubatch": engine["n_ubatch"],
        "n_threads": engine["n_threads"],
        "n_gpu_layers": engine["n_gpu_layers"],
        "flash_attn": engine["flash_attn"],
        "devices": engine["devices"],
        "pooling_type": engine["pooling_type"],
        "attention_type": engine["attention_type"],
        "embeddings": engine["embeddings"],
        "hidden_dtype": engine["hidden_dtype"],
        "hidden_rounding": engine["hidden_rounding"],
        "dropped_position": engine["dropped_position"],
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    # `capture_identity` (replay) reads this block; a GGUF is one file, so its
    # content identity is that file's digest, carried in `shard_sha256` so the
    # existing shard-level plumbing keeps working.
    candidate_identity = {
        "model_path": gguf["path"],
        "model_revision": args.gguf_revision,
        "model_revision_source": "hub_revision" if args.gguf_revision else "none",
        "index_sha256": None,
        "config_sha256": None,
        "shard_sha256": {Path(gguf["path"]).name: gguf["sha256"]},
        "quantization": quantization,
        "quantization_config": None,
        "trust_remote_code": False,
        "kv_cache_dtype_requested": args.kv_cache_dtype,
        "kv_cache_dtype_resolved": args.kv_cache_dtype,
        "kv_cache_dtype_resolved_source": "llama_context_params.type_k",
        "engine_identity": engine_identity,
    }
    contract = {
        "suite_token_sha256": suite["suite_token_sha256"],
        "context_length": ctx_len,
        "filter": args.filter,
        "expected_indices": sorted(selected_indices),
        "candidate_content": {
            "model_revision": candidate_identity["model_revision"],
            "index_sha256": None,
            "config_sha256": None,
            "shard_sha256": candidate_identity["shard_sha256"],
        },
        "runtime": {
            "quantization": quantization,
            "quantization_config": None,
            "trust_remote_code": False,
            "kv_cache_dtype_requested": args.kv_cache_dtype,
            "kv_cache_dtype_resolved": args.kv_cache_dtype,
            "fp32": False,
            "chunk_accumulate": False,
            "max_batched_tokens": engine["n_batch"],
            "engine": "llama.cpp",
            "llamacpp_commit": engine_identity["llamacpp_commit"],
            "n_ubatch": engine["n_ubatch"],
            "n_gpu_layers": engine["n_gpu_layers"],
            "flash_attn": engine["flash_attn"],
            "devices": engine["devices"],
        },
    }
    complete = not missing
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "model": gguf["path"],
        "quantization": quantization,
        "quantization_config": None,
        "kv_cache_dtype": args.kv_cache_dtype,
        "candidate_identity": candidate_identity,
        "engine_identity": engine_identity,
        "suite_token_sha256": suite["suite_token_sha256"],
        "filter": args.filter,
        "context_length": ctx_len,
        "expected_contexts": len(selected_indices),
        "expected_indices": sorted(selected_indices),
        "capture_contract": contract,
        "capture_contract_sha256": fid.canonical_sha256(contract),
        "contexts": len(records),
        "captures": records,
        "complete": complete,
        "elapsed_sec": engine.get("elapsed_sec"),
        "capture_engine_json": "capture-engine.json",
    }
    fid.atomic_write_json(capture / "capture-manifest.json", manifest)

    if complete:
        verify(capture, suite, selected_indices)
    print("manifest_written " + json.dumps({
        "capture": str(capture),
        "contexts": len(records),
        "expected": len(selected_indices),
        "complete": complete,
        "hidden": [ctx_len - 1, width],
        "quantization": quantization,
        "gguf_sha256": gguf["sha256"],
        "llamacpp_commit": engine_identity["llamacpp_commit"],
        "validated": complete,
    }), flush=True)
    return 0 if complete else 1


def verify(capture: Path, suite: dict, selected_indices: list[int]) -> dict:
    """Re-verify with the harness's own validator, then re-check the shapes."""
    manifest = fid.validate_capture_files(
        capture, suite["suite_token_sha256"], set(selected_indices)
    )
    ctx_len = suite["context_length"]
    for record in manifest["captures"]:
        dtype, shape, _ = safetensors_header(
            capture / f"hidden_{record['index']:04d}.safetensors")
        if dtype != "BF16" or shape != list(record["shape"]) or shape[0] != ctx_len - 1:
            die(f"context {record['index']}: {dtype} {shape} contradicts the manifest")
    return manifest


def cmd_verify(args) -> int:
    capture, _engine, suite, selected, _by_index = read_inputs(args)
    manifest = verify(capture, suite, [int(c["index"]) for c in selected])
    identity = manifest.get("engine_identity") or {}
    print("manifest_verified " + json.dumps({
        "capture": str(capture),
        "contexts": manifest["contexts"],
        "complete": manifest["complete"],
        "quantization": manifest.get("quantization"),
        "gguf_sha256": identity.get("gguf_sha256"),
        "llamacpp_commit": identity.get("llamacpp_commit"),
    }), flush=True)
    return 0


# ------------------------------------------------------------ smoke fixture
def bf16_bytes(values) -> bytes:
    """Pack floats as bfloat16 by truncation - fine for synthesized fixtures."""
    out = bytearray()
    for v in values:
        out += struct.pack("<f", v)[2:]
    return bytes(out)


def write_safetensors_bf16(path: Path, name: str, rows: int, cols: int, payload: bytes) -> None:
    if len(payload) != rows * cols * 2:
        die("internal: fixture payload size mismatch")
    header = json.dumps(
        {name: {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [0, len(payload)]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(payload)
    tmp.replace(path)


def cmd_smoke_fixture(args) -> int:
    import random

    out = Path(args.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    contexts = []
    for i in range(args.contexts):
        ids = [rng.randrange(1, args.token_ceiling) for _ in range(args.context_length)]
        name = f"context-{i:04d}.json"
        (out / "tokens" / name).write_text(json.dumps(ids))
        contexts.append({
            "index": i,
            "stratum": "synthetic",
            "file": f"tokens/{name}",
            "token_sha256": fid.sha256_bytes(json.dumps(ids).encode()),
            "tokens": len(ids),
            "source_cluster": f"synthetic-{i}",
        })
    manifest = {
        "schema": SUITE_SCHEMA_FAMILY + "1",
        "model": "synthetic",
        "synthetic": True,
        "note": "format fixture for the GGUF capture path; token ids are random, "
                "so no fidelity number measured through it describes a model",
        "trust_remote_code": False,
        "context_length": args.context_length,
        "scored_positions_per_context": args.context_length - 1,
        "contexts": len(contexts),
        "total_scored_positions": len(contexts) * (args.context_length - 1),
        "tokenizer_sha256": None,
        "strata": {"synthetic": len(contexts)},
        "seed": args.seed,
        "context_index": contexts,
    }
    manifest["suite_token_sha256"] = fid.sha256_bytes(
        "".join(c["token_sha256"] for c in contexts).encode())
    fid.atomic_write_json(out / "suite-manifest.json", manifest)

    head = Path(args.head)
    head.parent.mkdir(parents=True, exist_ok=True)
    payload = bf16_bytes((rng.random() - 0.5) * 0.25
                         for _ in range(args.vocab * args.hidden))
    write_safetensors_bf16(head, "weight", args.vocab, args.hidden, payload)
    print("smoke_fixture " + json.dumps({
        "suite": str(out),
        "contexts": len(contexts),
        "context_length": args.context_length,
        "suite_token_sha256": manifest["suite_token_sha256"],
        "head": str(head),
        "head_shape": [args.vocab, args.hidden],
        "head_sha256": fid.sha256_file(head),
    }), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--capture", required=True, help="directory gguf_capture wrote")
    b.add_argument("--suite", required=True)
    b.add_argument("--provenance", default=None,
                   help="llamacpp-provenance.json from build_llamacpp.sh")
    b.add_argument("--binary", default=None, help="gguf_capture binary to re-digest")
    b.add_argument("--gguf-revision", default=None, help="Hub revision the GGUF came from")
    b.add_argument("--gguf-sha256", default=None, help="expected GGUF digest; fails closed")
    b.add_argument("--quantization", default=None,
                   help="quantization label; default is the GGUF's own file type")
    b.add_argument("--kv-cache-dtype", default="f16",
                   help="KV dtype the capture ran with (llama.cpp default f16)")
    b.add_argument("--expect-hidden", type=int, default=0)
    b.add_argument("--filter", default="all")
    b.add_argument("--allow-incomplete", action="store_true",
                   help="write complete=false for a partial capture (replay will reject it)")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify")
    v.add_argument("--capture", required=True)
    v.add_argument("--suite", required=True)
    v.add_argument("--filter", default="all")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("smoke-fixture")
    s.add_argument("--out", required=True, help="suite directory to create")
    s.add_argument("--head", required=True, help="random BF16 head to create")
    s.add_argument("--contexts", type=int, default=2)
    s.add_argument("--context-length", type=int, default=128)
    s.add_argument("--hidden", type=int, default=1024)
    s.add_argument("--vocab", type=int, default=4096)
    s.add_argument("--token-ceiling", type=int, default=100000,
                   help="exclusive upper bound on synthetic token ids")
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_smoke_fixture)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
