#!/usr/bin/env python3
"""KV-dtype sweep on the physical RTX 5090 (AIBoss).

One vLLM process per arm. Every launch records its verbatim argv, its full
server log, and the parsed startup banner. Refusals are recorded verbatim and
are results, not failures.

Stages:
  census    -- attempt every kv-cache-dtype the pinned build advertises, short
               window, and record start-or-refuse with the verbatim error.
  law       -- per measured arm, provoke two startup refusals at one capped
               utilisation and two windows so a and M can be re-derived from
               the engine's own printed per-request requirement.
  probe     -- native-window startup probe used only to size a reduced window
               for an arm that cannot hold 262,144.
  arm       -- full measurement of one arm: banner, capacity, needle retrieval
               at several depths at two lengths, paired greedy continuations
               with top-20 logprobs, three warmed decode runs, MTP acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

W = Path("/home/mbelleau/kvsweep")
OUT = W / "out"
LOGS = W / "logs"
PROMPTS = W / "prompts"
CORPUS = Path("/home/mbelleau/qual5090/corpus")
CACHE_JIT = "/home/mbelleau/qual5090/cache/jit"
CACHE_EXL3 = "/home/mbelleau/qual5090/cache/exl3-online"

UUID = "GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a"
IMAGE = "localhost/vllm:gg-r34-patched-apc"
IMAGE_DIGEST = "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218"
REPO_DIR = "/mnt/vault/llm/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context"
REV = "c45c273b0d6ef2859cb2d85b36dd52253c80d878"
MODEL = f"/models/ctx-repo/snapshots/{REV}"
PORT = 8241
API = f"http://127.0.0.1:{PORT}/v1"

QCFG = ('{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\\\..*",'
        '"re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$",'
        '"re:.*mtp\\\\..*","lm_head"]}')
SPEC = '{"method":"mtp","num_speculative_tokens":3}'
MM = '{"truncation":false,"max_pixels":8388608}'
COMPILE = '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'

ALL_DTYPES = [
    "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc",
    "fp8_ds_mla", "nvfp4_ds_mla", "turboquant_k8v4", "turboquant_4bit_nc",
    "turboquant_k3v4_nc", "turboquant_3bit_nc", "int4_per_token_head",
    "int8_per_token_head", "fp8_per_token_head", "nvfp4",
]

NEEDLE_DEPTHS = [float(x) for x in
                 os.environ.get("NEEDLE_DEPTHS", "0.05,0.25,0.50,0.75,0.95").split(",")]
# Long-length needles are the expensive probe (one full-window prefill each), so
# the long set is a subset of the common-length set: same depths, fewer of them.
NEEDLE_DEPTHS_LONG = [float(x) for x in
                      os.environ.get("NEEDLE_DEPTHS_LONG", "0.05,0.50,0.95").split(",") if x]

NEEDLE_SEED = 20260816
FIDELITY_PROMPTS = 4
FIDELITY_GEN = 64
FIDELITY_TOPK = 20


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with (LOGS / "driver.log").open("a") as fh:
        fh.write(line + "\n")


def sh(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def gpu_sample(name: str) -> dict:
    q = "memory.total,memory.used,memory.free,utilization.gpu,power.draw"
    raw = sh(["nvidia-smi", f"--id={UUID}", f"--query-gpu={q}",
              "--format=csv,noheader,nounits"]).stdout.strip()
    parts = [p.strip() for p in raw.split(",")]
    keys = ["memory_total_mib", "memory_used_mib", "memory_free_mib",
            "utilization_gpu_pct", "power_draw_w"]
    row = {"name": name, "utc": utc()}
    for k, v in zip(keys, parts):
        try:
            row[k] = int(v)
        except ValueError:
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v
    return row


# --------------------------------------------------------------------------
# launching


def build_argv(tag: str, dtype: str, util: float, maxlen: int,
               mtp: int = 3) -> tuple[list[str], list[str]]:
    podman = [
        "podman", "run", "--rm", "--name", f"kvsweep-{tag}",
        "--device", "nvidia.com/gpu=all", "--ipc=host", "--network", "host",
        "-v", f"{REPO_DIR}:/models/ctx-repo:ro",
        "-v", f"{CACHE_JIT}:/cache/jit:rw",
        "-v", f"{CACHE_EXL3}:/cache/exl3-online:rw",
        "-e", "VLLM_EXL3_EMBED_BITS=8",
        "-e", "VLLM_EXL3_GRAPH_DECODE=1",
        "-e", "VLLM_EXL3_PREFILL_RECONSTRUCT_M=128",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", "XDG_CACHE_HOME=/cache/jit",
        "-e", "CUDA_CACHE_PATH=/cache/jit",
        "-e", "TRITON_CACHE_DIR=/cache/jit/triton",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor",
        "--entrypoint", "/opt/venv/bin/vllm",
        IMAGE,
    ]
    vllm = [
        "serve", MODEL,
        "--served-model-name", "m",
        "--quantization", "exl3",
        "--quantization-config", QCFG,
        "--max-model-len", str(maxlen),
        "--gpu-memory-utilization", str(util),
        "--kv-cache-dtype", dtype,
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", "2048",
    ]
    if mtp:
        vllm += ["--speculative-config",
                 '{"method":"mtp","num_speculative_tokens":%d}' % mtp]
    vllm += [
        "--mm-processor-kwargs", MM,
        "--compilation-config", COMPILE,
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]
    return podman, vllm


def kill_container(tag: str) -> None:
    sh(["podman", "rm", "-f", f"kvsweep-{tag}"], timeout=120)


def wait_gpu_idle(limit_mib: int = 500, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = gpu_sample("wait_idle")
        if isinstance(s.get("memory_used_mib"), int) and s["memory_used_mib"] <= limit_mib:
            return s
        time.sleep(3)
    return gpu_sample("wait_idle_timeout")


def launch(tag: str, dtype: str, util: float, maxlen: int,
           startup_timeout: int = 1200, mtp: int = 3) -> dict:
    """Start one server. Returns a launch record; server stays up if ready."""
    kill_container(tag)
    wait_gpu_idle()
    podman, vllm = build_argv(tag, dtype, util, maxlen, mtp)
    logfile = LOGS / f"server-{tag}.log"
    logfile.write_text("")
    rec = {
        "tag": tag,
        "kv_cache_dtype": dtype,
        "gpu_memory_utilization": util,
        "max_model_len": maxlen,
        "mtp_depth": mtp,
        "image": IMAGE,
        "image_manifest_digest": IMAGE_DIGEST,
        "model_revision": REV,
        "podman_argv": podman,
        "vllm_argv": vllm,
        "server_log": str(logfile),
        "started_utc": utc(),
        "gpu_before_load": gpu_sample("before_load"),
    }
    log(f"launch tag={tag} dtype={dtype} util={util} maxlen={maxlen} mtp={mtp}")
    t0 = time.monotonic()
    with logfile.open("w") as fh:
        proc = subprocess.Popen(podman + vllm, stdout=fh, stderr=subprocess.STDOUT)
    ready = False
    exited = None
    while time.monotonic() - t0 < startup_timeout:
        time.sleep(2)
        text = logfile.read_text(errors="ignore")
        if "Application startup complete" in text:
            ready = True
            break
        rc = proc.poll()
        if rc is not None:
            exited = rc
            time.sleep(2)
            break
    rec["startup_wall_sec"] = round(time.monotonic() - t0, 3)
    text = logfile.read_text(errors="ignore")
    rec["ready"] = ready
    rec["process_exit_code"] = exited
    if ready and not health_ok():
        rec["ready"] = False
        rec["note"] = "log said startup complete but /health did not answer"
        ready = False
    rec["startup_facts"] = banner_facts(text)
    if not ready:
        rec["refusal"] = extract_error(text)
        proc.poll()
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
        kill_container(tag)
        wait_gpu_idle()
    else:
        rec["gpu_after_startup"] = gpu_sample("after_startup")
    rec["log_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return rec


def health_ok(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


BANNER_PATTERNS = {
    "desired_gpu_memory_utilization": r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)",
    "free_memory_on_device": r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup",
    "actual_usage": (r"Actual usage is ([\d.]+) GiB for weight, ([\d.]+) GiB for peak activation, "
                     r"([\d.]+) GiB for non-torch memory, and ([\d.]+) GiB for CUDAGraph memory"),
    "available_kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "gpu_kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "maximum_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "model_loading": r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds",
    "graph_capture": r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB",
    "init_engine": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s \(compilation: ([\d.]+) s\)",
    "attention_block_size": r"Setting attention block size to (\d+) tokens",
    "mamba_page_padding_pct": r"Padding mamba page size by ([\d.]+)%",
    "kv_padding_layers": r"Add (\d+) padding layers, may waste at most ([\d.]+)% KV cache memory",
    "num_spec_tokens": r"num_spec_tokens=(\d+)",
    "enable_prefix_caching": r"enable_prefix_caching=(\w+)",
    "kv_cache_dtype_note": r"(Using [\w.\- ]+ data type to store kv cache)",
    "vllm_version": r"Initializing a V1 LLM engine \(v([^)]+)\)",
}

RAW_LINE_MARKERS = [
    "Actual usage is", "Add ", "Available KV cache memory:",
    "Desired GPU memory utilization", "Free memory on device",
    "GPU KV cache size:", "Graph capturing finished",
    "Initializing a V1 LLM engine", "Maximum concurrency for",
    "Model loading took", "Padding mamba page size",
    "Setting attention block size", "data type to store kv cache",
    "init engine (profile", "num_spec_tokens=", "enable_prefix_caching=",
]


def banner_facts(text: str) -> dict:
    facts: dict = {}
    for key, pat in BANNER_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            facts[key] = None
        elif len(m.groups()) == 1:
            facts[key] = m.group(1)
        else:
            facts[key] = list(m.groups())
    raw = set()
    for line in text.splitlines():
        for marker in RAW_LINE_MARKERS:
            if marker in line:
                idx = line.find("] ")
                raw.add(line[idx + 2:].strip() if idx >= 0 else line.strip())
                break
    facts["raw_lines"] = sorted(raw)
    m = re.search(r"^.*Initializing a V1 LLM engine.*$", text, re.M)
    facts["engine_config_line"] = m.group(0)[:1400] if m else None
    return facts


def extract_error(text: str) -> dict:
    """Verbatim error extraction: last traceback, plus the KV refusal if present."""
    out: dict = {}
    m = re.search(
        r"To serve at least one request with the model's max seq len\s*"
        r"\((\d+)\), \(([\d.]+) GiB KV cache is needed, which is larger than the "
        r"available KV cache memory \(([\d.]+) GiB\)\.(?: Based on the available memory, "
        r"the estimated maximum model length is (\d+)\.)?",
        text.replace("\n", " "))
    if m:
        out["kind"] = "kv_capacity_refusal"
        out["max_model_len"] = int(m.group(1))
        out["kv_needed_gib"] = float(m.group(2))
        out["kv_available_gib"] = float(m.group(3))
        out["estimated_max_model_len"] = int(m.group(4)) if m.group(4) else None
    lines = text.splitlines()
    verbatim = []
    for i, line in enumerate(lines):
        if re.search(r"(ValueError|RuntimeError|NotImplementedError|AssertionError|"
                     r"KeyError|TypeError|argument --kv-cache-dtype|invalid choice|"
                     r"is not supported|Unsupported|Traceback \(most recent call last\))", line):
            verbatim.extend(lines[i:i + 12])
            break
    if not verbatim:
        verbatim = lines[-40:]
    seen = []
    for line in verbatim:
        s = line.rstrip()
        if s:
            seen.append(s)
    out["verbatim"] = seen[:40]
    out["verbatim_tail"] = [l.rstrip() for l in lines[-25:] if l.strip()]
    if "kind" not in out:
        out["kind"] = "startup_error"
    return out


# --------------------------------------------------------------------------
# http helpers


def post(path: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize_count(text: str) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/tokenize",
        data=json.dumps({"model": "m", "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read())
    n = res.get("count")
    if n is None:
        toks = res.get("tokens")
        n = len(toks) if isinstance(toks, list) else None
    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(f"tokenizer returned no count: {str(res)[:200]}")
    return n


def metrics() -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=60) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode"):
            parts = line.rsplit(" ", 1)
            name = parts[0].split("{")[0]
            try:
                out[name] = out.get(name, 0.0) + float(parts[1])
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------
# prompt construction (cached on disk so every arm sees identical bytes)


def corpus_text() -> str:
    files = sorted(CORPUS.rglob("*.txt"))
    if not files:
        raise SystemExit("no corpus")
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)


_CORPUS: str | None = None


def whole() -> str:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = corpus_text()
    return _CORPUS


def template(prompt: str) -> str:
    return ("<|im_start|>user\n" + prompt +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def converge(build, target: int, tol: int = 64, tries: int = 10) -> tuple[str, int]:
    chars = int(target * 3.6)
    text, n = "", 0
    for _ in range(tries):
        text = build(chars)
        n = tokenize_count(text)
        if abs(n - target) <= tol:
            return text, n
        chars = max(1, int(chars * target / n))
    raise RuntimeError(f"prompt sizing did not converge: target {target}, got {n}")


def needle_prompt(tokens: int, depth: float) -> dict:
    key = f"needle-{tokens}-{int(depth * 100):03d}"
    cache = PROMPTS / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import random
    rng = random.Random(NEEDLE_SEED + int(depth * 100))
    code = str(rng.randrange(10 ** 9, 10 ** 10))
    needle = ("\n\nIMPORTANT RECORD: the Gilded Gnosis maintenance access code is "
              f"{code}. Remember this exact number.\n\n")
    body = whole()

    def build(chars: int) -> str:
        filler = body
        while len(filler) < chars:
            filler = filler + "\n" + filler
        filler = filler[:chars]
        cut = int(len(filler) * depth)
        p = (filler[:cut] + needle + filler[cut:] +
             "\n\nQuestion: what is the Gilded Gnosis maintenance access code "
             "recorded above? Answer with the number only.\nAnswer:")
        return template(p)

    text, n = converge(build, tokens)
    rec = {"key": key, "target_tokens": tokens, "depth": depth, "code": code,
           "text": text, "measured_tokens": n,
           "sha256": hashlib.sha256(text.encode()).hexdigest()}
    cache.write_text(json.dumps(rec))
    return rec


def fidelity_prompt(tokens: int, index: int) -> dict:
    key = f"fid-{tokens}-{index}"
    cache = PROMPTS / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    body = whole()
    offset = (index * 977_003) % max(1, len(body) - 1)
    rotated = body[offset:] + "\n" + body[:offset]

    def build(chars: int) -> str:
        filler = rotated
        while len(filler) < chars:
            filler = filler + "\n" + filler
        filler = filler[:chars]
        p = (filler +
             "\n\nContinue the passage above in the same style, in prose, "
             "without commentary.\n")
        return template(p)

    text, n = converge(build, tokens)
    rec = {"key": key, "target_tokens": tokens, "index": index, "text": text,
           "measured_tokens": n,
           "sha256": hashlib.sha256(text.encode()).hexdigest()}
    cache.write_text(json.dumps(rec))
    return rec


# --------------------------------------------------------------------------
# measurements


def run_needles(tokens: int, depths: list[float], label: str) -> list[dict]:
    results = []
    for d in depths:
        p = needle_prompt(tokens, d)
        t0 = time.monotonic()
        res = post("/completions", {"model": "m", "prompt": p["text"],
                                    "max_tokens": 48, "temperature": 0,
                                    "stream": False})
        el = time.monotonic() - t0
        text = res["choices"][0]["text"].strip()
        usage = res["usage"]
        row = {
            "label": label,
            "requested_tokens": tokens,
            "prompt_tokens": usage["prompt_tokens"],
            "prompt_sha256": p["sha256"],
            "depth": d,
            "code": p["code"],
            "answer": text[:120],
            "retrieved_exact": text == p["code"],
            "completion_tokens": usage.get("completion_tokens"),
            "wall_sec": round(el, 3),
            "prefill_tok_s_including_decode": round(usage["prompt_tokens"] / el, 1),
        }
        log(f"needle {label} depth={d} tokens={row['prompt_tokens']} "
            f"exact={row['retrieved_exact']} {row['wall_sec']}s")
        results.append(row)
    return results


def run_fidelity(tokens: int, n_prompts: int) -> list[dict]:
    """Greedy continuations on identical prompts, with top-k logprobs.

    The last row repeats prompt 0 so within-arm run-to-run determinism is a
    measured control rather than an assumption.
    """
    rows = []
    for i in list(range(n_prompts)) + [0]:
        p = fidelity_prompt(tokens, i)
        t0 = time.monotonic()
        res = post("/completions", {
            "model": "m", "prompt": p["text"], "max_tokens": FIDELITY_GEN,
            "temperature": 0, "top_p": 1, "seed": 0, "ignore_eos": True,
            "logprobs": FIDELITY_TOPK, "stream": False})
        el = time.monotonic() - t0
        ch = res["choices"][0]
        lp = ch.get("logprobs") or {}
        rows.append({
            "index": i,
            "repeat": len(rows) >= n_prompts,
            "prompt_sha256": p["sha256"],
            "prompt_tokens": res["usage"]["prompt_tokens"],
            "completion_tokens": res["usage"].get("completion_tokens"),
            "text": ch["text"],
            "tokens": lp.get("tokens"),
            "token_logprobs": lp.get("token_logprobs"),
            "top_logprobs": lp.get("top_logprobs"),
            "wall_sec": round(el, 3),
        })
        log(f"fidelity prompt {i} tokens={rows[-1]['prompt_tokens']} {rows[-1]['wall_sec']}s")
    return rows


def run_decode() -> dict:
    body = {"model": "m",
            "prompt": "Write a detailed technical explanation of tensor parallelism. " * 4,
            "max_tokens": 256, "temperature": 0, "ignore_eos": True, "stream": False}
    post("/completions", dict(body, max_tokens=8))
    before = metrics()
    runs = []
    for _ in range(3):
        t0 = time.monotonic()
        r = post("/completions", body)
        el = time.monotonic() - t0
        n = r["usage"]["completion_tokens"]
        runs.append({"concurrency": 1, "elapsed_sec": round(el, 3),
                     "output_tokens": n, "aggregate_tok_s": round(n / el, 2),
                     "per_stream_tok_s": round(n / el, 2)})
    after = metrics()
    lat = []
    for _ in range(5):
        t0 = time.monotonic()
        post("/completions", dict(body, max_tokens=1))
        lat.append(time.monotonic() - t0)
    lat.sort()
    rates = sorted(r["aggregate_tok_s"] for r in runs)
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get("vllm:spec_decode_num_drafts_total", 0)
    dtok = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
    atok = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    return {
        "protocol": "one warm-up request, then three timed runs, greedy, "
                    "ignore_eos, 256 output tokens each, concurrency 1",
        "runs": runs,
        "median_decode_tok_s": rates[1],
        "min_decode_tok_s": rates[0],
        "max_decode_tok_s": rates[2],
        "dispersion_range_tok_s": round(rates[2] - rates[0], 3),
        "dispersion_relative_pct": round(100 * (rates[2] - rates[0]) / rates[1], 3),
        "ttft_proxy_sec": round(lat[2], 4),
        "mtp_acceptance": {
            "counters_before": before, "counters_after": after,
            "drafts": drafts, "draft_tokens": dtok, "accepted_tokens": atok,
            "draft_token_acceptance_rate": round(atok / dtok, 4) if dtok else None,
            "mean_accepted_length": round(1 + atok / drafts, 4) if drafts else None,
        },
    }


# --------------------------------------------------------------------------
# stages


def save(name: str, obj) -> None:
    p = OUT / f"{name}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)
    log(f"wrote {p}")


def stage_census(dtypes: list[str]) -> None:
    path = OUT / "census.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for d in dtypes:
        if d in done:
            continue
        rec = launch(f"census-{d}", d, 0.85, 8192, startup_timeout=900)
        if rec["ready"]:
            rec["probe_generation"] = post("/completions", {
                "model": "m", "prompt": "2+2=", "max_tokens": 8,
                "temperature": 0})["choices"][0]["text"]
            kill_container(f"census-{d}")
            wait_gpu_idle()
        done[d] = rec
        save("census", done)


def stage_law(arms: list[str], mtp: int = 3) -> None:
    name = "law" if mtp else "law-mtpoff"
    path = OUT / f"{name}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    cands = (0.72, 0.70, 0.68, 0.66) if mtp else (0.70, 0.68, 0.66, 0.64)
    for d in arms:
        if d in done and done[d].get("complete"):
            continue
        entry = done.get(d, {})
        util = None
        for cand in cands:
            tag = f"{name}-{d}-131072-{cand}"
            rec = launch(tag, d, cand, 131072, startup_timeout=900, mtp=mtp)
            if not rec["ready"] and rec.get("refusal", {}).get("kind") == "kv_capacity_refusal":
                entry["refusal_131072"] = rec
                util = cand
                break
            if rec["ready"]:
                kill_container(tag)
                wait_gpu_idle()
                entry.setdefault("started_instead_of_refusing", []).append(
                    {"util": cand, "startup_facts": rec["startup_facts"]})
                continue
            entry["hard_error_131072"] = rec
            break
        if util is None:
            entry["complete"] = False
            done[d] = entry
            save(name, done)
            continue
        rec2 = launch(f"{name}-{d}-262144-{util}", d, util, 262144,
                      startup_timeout=900, mtp=mtp)
        entry["refusal_262144"] = rec2
        entry["util"] = util
        entry["mtp_depth"] = mtp
        entry["complete"] = (
            rec2.get("refusal", {}).get("kind") == "kv_capacity_refusal")
        done[d] = entry
        save(name, done)


def stage_probe(dtype: str) -> None:
    rec = launch(f"probe-{dtype}", dtype, 0.955, 262144, startup_timeout=1200)
    if rec["ready"]:
        kill_container(f"probe-{dtype}")
        wait_gpu_idle()
    save(f"probe-{dtype}", rec)


def stage_arm(dtype: str, maxlen: int, needle_len: int, common_len: int) -> None:
    tag = f"arm-{dtype}"
    rec = launch(tag, dtype, 0.955, maxlen, startup_timeout=1200)
    refused_at = None
    if not rec["ready"]:
        r = rec.get("refusal") or {}
        est = r.get("estimated_max_model_len")
        if r.get("kind") == "kv_capacity_refusal" and est:
            refused_at = rec
            maxlen = (est // 4096) * 4096
            needle_len = maxlen - 344
            log(f"{dtype} refused at {r['max_model_len']}; falling back to {maxlen}")
            rec = launch(tag, dtype, 0.955, maxlen, startup_timeout=1200)
    result = {"arm": dtype, "launch": rec, "max_model_len": maxlen,
              "needle_len_long": needle_len, "common_len": common_len,
              "refused_at_native_window": refused_at}
    if not rec["ready"]:
        save(f"arm-{dtype}", result)
        return
    try:
        result["needles_long"] = run_needles(needle_len, NEEDLE_DEPTHS_LONG, "longest_fitting")
        save(f"arm-{dtype}", result)
        if common_len != needle_len:
            result["needles_common"] = run_needles(common_len, NEEDLE_DEPTHS, "common")
            save(f"arm-{dtype}", result)
        else:
            result["needles_common"] = result["needles_long"]
        result["fidelity"] = run_fidelity(common_len, FIDELITY_PROMPTS)
        save(f"arm-{dtype}", result)
        result["decode"] = run_decode()
        result["gpu_after_all"] = gpu_sample("after_all")
    except Exception as exc:  # keep whatever was measured
        result["error"] = repr(exc)
    finally:
        save(f"arm-{dtype}", result)
        kill_container(tag)
        wait_gpu_idle()
        result["gpu_after_release"] = gpu_sample("after_release")
        save(f"arm-{dtype}", result)


def main() -> int:
    for d in (OUT, LOGS, PROMPTS):
        d.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--dtypes", nargs="*", default=None)
    ap.add_argument("--dtype")
    ap.add_argument("--maxlen", type=int, default=262144)
    ap.add_argument("--needle-len", type=int, default=261800)
    ap.add_argument("--common-len", type=int, default=98304)
    ap.add_argument("--mtp", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "census":
        stage_census(a.dtypes or ALL_DTYPES)
    elif a.stage == "law":
        stage_law(a.dtypes or [], mtp=a.mtp)
    elif a.stage == "probe":
        stage_probe(a.dtype)
    elif a.stage == "arm":
        stage_arm(a.dtype, a.maxlen, a.needle_len, a.common_len)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
