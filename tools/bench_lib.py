#!/usr/bin/env python3
"""Shared benchmark library for the Qwen3.8-27B EXL3 Phase-0 measurement harness.

Both ``bench-profile.sh`` (multi-boot benchmark producing a JSON receipt) and
``verify-profile.sh`` (regression gate against a stored baseline) import this
module so that the prompt definitions, metric calculations, and server probes
are defined in exactly one place.  Comparability across runs depends on every
caller using the same prompts and the same measurement methodology; duplicating
these definitions would silently break A/B comparisons.

The module is pure host-side: it drives the already-running vLLM OpenAI server
over HTTP and reads ``nvidia-smi`` for telemetry.  It does NOT start or stop
the systemd service or podman containers — that orchestration belongs to the
shell scripts.  ``wait_healthy`` assumes the service launch has already been
initiated.

Dependencies: Python 3 stdlib + ``requests`` only (no PIL, no numpy).
"""
from __future__ import annotations

import base64
import os
import re
import statistics
import struct
import subprocess
import time
import zlib

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# These prompts are the canonical benchmark prompts for the Qwen3.8-27B EXL3
# stack.  They MUST be reused verbatim for comparability across boots and
# across profile/verify runs.  Changing a prompt invalidates every stored
# baseline.

# PP (prefill throughput) prompt: 2051 prompt_tokens at the default tokeniser.
# The three repeating sentences exercise different token patterns so the
# prefill path sees a realistic mixed-vocabulary context rather than a single
# repeated token id.
PP_PROMPT = (
    "The quick brown fox jumps over the lazy dog. " * 100
    + "In machine learning, gradient descent is an optimization algorithm. " * 50
    + "The mitochondria is the powerhouse of the cell. " * 50
)

# TG (decode / token-generation) prompts.
TG_FOX_PROMPT = "The quick brown fox jumps over the lazy dog. Test of decode."
TG_ESSAY_PROMPT = "Write a detailed essay about the history of computing."

# Warmup before PP timing: the fox sentence repeated by each multiplier.
# Gradually increasing prompt lengths prime the prefill path and KV cache so
# the first timed PP request is not paying cold-start JIT/compile cost.
WARMUP_MULTS = [50, 100, 200, 400]

# Server / model configuration — overridable by env for non-default ports.
BASE_URL = os.environ.get("BENCH_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B")

# Container and systemd names (fixed infrastructure, not overridable).
CONTAINER = "qwen38-27b"
SERVICE = "qwen38-27b.service"
LAUNCHER = "/home/mbelleau/run-qwen38-27b.sh"

# JSON receipt schema identifier.
SCHEMA = "qwen38-bench-profile/1"

# HTTP timeout for normal requests (seconds).
_HTTP_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BootDied(Exception):
    """Raised by ``wait_healthy`` when the container died during boot.

    The ``error_lines`` attribute holds the last distinct error lines from
    ``podman logs`` so the caller can record them in the receipt JSON.
    """

    def __init__(self, message: str, error_lines: list[str] | None = None):
        super().__init__(message)
        self.error_lines = error_lines or []


# ---------------------------------------------------------------------------
# Health / boot
# ---------------------------------------------------------------------------
def _container_status() -> str:
    """Return the podman container status string, or '' on inspection failure."""
    try:
        r = subprocess.run(
            ["podman", "inspect", "-f", "{{.State.Status}}", CONTAINER],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _boot_error_lines() -> list[str]:
    """Grep the last distinct Error/OutOfMemory/Traceback lines from podman logs."""
    try:
        r = subprocess.run(
            ["podman", "logs", CONTAINER],
            capture_output=True, text=True, timeout=30,
        )
        text = r.stdout + r.stderr
    except Exception:
        return []
    pat = re.compile(r"Error|OutOfMemory|Traceback", re.IGNORECASE)
    seen: list[str] = []
    for line in text.splitlines():
        if pat.search(line) and line not in seen:
            seen.append(line)
    return seen[-5:]


def wait_healthy(timeout_s: int = 600) -> float:
    """Poll ``/health`` until the server responds, or raise on boot death.

    Also polls ``podman inspect`` for the container status; if the container
    is not ``running`` the boot died and ``BootDied`` is raised with the last
    distinct error lines from ``podman logs``.

    Returns the wall-clock seconds elapsed from call start to first healthy
    response.  Raises ``TimeoutError`` if the server does not become healthy
    within ``timeout_s`` seconds (and the container is still running).
    """
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        # Check container liveness first — a dead container means we should
        # not keep waiting for /health.
        status = _container_status()
        if status and status != "running":
            raise BootDied(
                f"container {CONTAINER} status is '{status}', not 'running'",
                _boot_error_lines(),
            )
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return time.monotonic() - start
        except requests.RequestException:
            pass
        time.sleep(2)
    # Final container check before declaring timeout.
    status = _container_status()
    if status and status != "running":
        raise BootDied(
            f"container {CONTAINER} status is '{status}', not 'running'",
            _boot_error_lines(),
        )
    raise TimeoutError(f"server did not become healthy within {timeout_s}s")


# ---------------------------------------------------------------------------
# Speculative-decode counters (MTP acceptance)
# ---------------------------------------------------------------------------
_METRIC_DRAFT = "vllm:spec_decode_num_draft_tokens_total"
_METRIC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"


def spec_counters() -> tuple[int, int]:
    """Read cumulative MTP draft/accepted counters from ``/metrics``.

    Returns ``(draft_total, accepted_total)``.  Returns ``(0, 0)`` if the
    metrics endpoint is unreachable or the counters are absent.  Never raises.
    """
    try:
        r = requests.get(f"{BASE_URL}/metrics", timeout=15)
        if r.status_code != 200:
            return (0, 0)
        text = r.text
    except requests.RequestException:
        return (0, 0)

    def _extract(metric_name: str) -> int:
        # Prometheus text format: metric_name{labels} value\n
        # We match the metric name at line start, skip any labels, and parse
        # the trailing float as an int (counters are monotonically increasing
        # integers).
        pat = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.eE]+)", re.MULTILINE)
        m = pat.search(text)
        if m:
            try:
                return int(float(m.group(1)))
            except ValueError:
                return 0
        return 0

    return (_extract(_METRIC_DRAFT), _extract(_METRIC_ACCEPTED))


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def _post_completions(payload: dict, timeout: int = _HTTP_TIMEOUT) -> dict:
    """POST to /v1/completions and return the JSON response."""
    r = requests.post(
        f"{BASE_URL}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def warmup() -> None:
    """Warm the prefill path with increasing-length prompts (max_tokens=1).

    Best-effort: per-request errors are printed as warnings but do not raise,
    so a warmup failure does not abort the entire benchmark run.
    """
    for mult in WARMUP_MULTS:
        prompt = "The quick brown fox jumps over the lazy dog. " * mult
        try:
            _post_completions(
                {"model": MODEL, "prompt": prompt,
                 "max_tokens": 1, "temperature": 0},
                timeout=300,
            )
        except Exception as e:
            print(f"WARNING: warmup mult={mult} failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# PP (prefill throughput) measurement
# ---------------------------------------------------------------------------
def measure_pp(reps: int = 5) -> dict:
    """Measure prefill throughput (tok/s) over ``reps`` timed requests.

    Performs warmup first, then sends ``reps`` requests with the canonical
    2051-token PP prompt at ``max_tokens=1, temperature=0`` so the wall time
    is prefill-dominated.  Uses the median (not mean) for the headline figure
    to suppress single-rep outliers.

    Returns::

        {
            'tok_s_median': float,
            'tok_s_all':    list[float],
            'prompt_tokens': int,
            'wall_all':     list[float],
        }
    """
    warmup()
    tok_s_all: list[float] = []
    wall_all: list[float] = []
    prompt_tokens = 0
    for _ in range(reps):
        t0 = time.monotonic()
        resp = _post_completions(
            {"model": MODEL, "prompt": PP_PROMPT,
             "max_tokens": 1, "temperature": 0},
            timeout=300,
        )
        wall = time.monotonic() - t0
        wall_all.append(wall)
        prompt_tokens = resp.get("usage", {}).get("prompt_tokens", 0)
        tok_s_all.append(prompt_tokens / wall if wall > 0 else 0.0)
    return {
        "tok_s_median": statistics.median(tok_s_all),
        "tok_s_all": tok_s_all,
        "prompt_tokens": prompt_tokens,
        "wall_all": wall_all,
    }


# ---------------------------------------------------------------------------
# TG (decode throughput) measurement
# ---------------------------------------------------------------------------
def measure_tg(prompt: str, max_tokens: int, reps: int = 3) -> dict:
    """Measure decode throughput (tok/s) over ``reps`` timed requests.

    Samples the MTP spec-decode counters before and after the run so the
    acceptance rate can be differenced (the counters are cumulative).
    Uses ``ignore_eos=True`` so the output length is exactly ``max_tokens``
    and the measurement is decode-bound.

    Returns::

        {
            'tok_s_median':      float,
            'tok_s_all':         list[float],
            'completion_tokens': int,
            'acceptance':        float | None,
            'draft_delta':       int,
            'accepted_delta':    int,
        }

    ``acceptance`` is ``accepted_delta / draft_delta``, or ``None`` if
    ``draft_delta == 0`` (speculative decoding disabled or no drafts issued).
    """
    draft_before, accepted_before = spec_counters()

    tok_s_all: list[float] = []
    completion_tokens = 0
    for _ in range(reps):
        t0 = time.monotonic()
        resp = _post_completions(
            {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
             "temperature": 0, "ignore_eos": True},
            timeout=600,
        )
        wall = time.monotonic() - t0
        completion_tokens = resp.get("usage", {}).get("completion_tokens", 0)
        tok_s_all.append(completion_tokens / wall if wall > 0 else 0.0)

    draft_after, accepted_after = spec_counters()
    draft_delta = draft_after - draft_before
    accepted_delta = accepted_after - accepted_before
    acceptance = (accepted_delta / draft_delta) if draft_delta > 0 else None

    return {
        "tok_s_median": statistics.median(tok_s_all),
        "tok_s_all": tok_s_all,
        "completion_tokens": completion_tokens,
        "acceptance": acceptance,
        "draft_delta": draft_delta,
        "accepted_delta": accepted_delta,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity() -> str:
    """Quick sanity generation: 'The capital of France is', max_tokens=5.

    Returns the generated text (stripped).  A correct flagship model should
    produce 'Paris' (or ' Paris').
    """
    resp = _post_completions(
        {"model": MODEL, "prompt": "The capital of France is",
         "max_tokens": 5, "temperature": 0},
    )
    return resp["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Vision check (multimodal)
# ---------------------------------------------------------------------------
def _red_blue_png() -> bytes:
    """Encode a 256x128 PNG: left half pure red, right half pure blue.

    Uses only ``zlib`` and ``struct`` (no PIL dependency).  The image is
    RGB, 8 bits per channel.  Each row is 256 pixels = 768 bytes of raw
    data (prefixed by a filter-type byte per scanline).
    """
    width = 256
    height = 128

    # Build raw scanlines.  PNG scanlines are prefixed by a filter-type byte
    # (0 = None).  Left 128 pixels: red (R=255, G=0, B=0).  Right 128 pixels:
    # blue (R=0, G=0, B=255).
    # One scanline = filter byte (0=None) + 256 pixels of RGB data.
    # Left 128 pixels: red (R=255, G=0, B=0).  Right 128 pixels: blue
    # (R=0, G=0, B=255).  Both halves share a single filter byte per row.
    row = b"\x00" + b"\xff\x00\x00" * 128 + b"\x00\x00\xff" * 128
    raw_data = row * height

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        """Build a PNG chunk: length + type + data + CRC32."""
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    # PNG signature.
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), comp=0, filter=0,
    # interlace=0.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # IDAT: zlib-compressed raw image data.
    idat = zlib.compress(raw_data, 9)
    # IEND: empty.
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def vision_check() -> bool:
    """POST a red/blue image to the chat endpoint and check the colour answer.

    Builds a 256x128 PNG (left red, right blue) in-memory, base64-encodes it
    as a data URI, and asks the model to name the two colours.  Returns True
    iff the reply matches ``/red/i`` AND ``/blue/i``.  Returns False on any
    error (model does not support vision, HTTP error, etc.).
    """
    try:
        png_bytes = _red_blue_png()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                        {
                            "type": "text",
                            "text": "What are the two colors, left then right? Answer with two words.",
                        },
                    ],
                }
            ],
            # This is a reasoning model: with thinking enabled it spends the
            # whole budget on `reasoning` and returns content=None
            # (finish_reason=length). Disable thinking so the answer lands in
            # `content`, and still fall back to `reasoning` defensively.
            "max_tokens": 32,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        return bool(re.search(r"red", text, re.IGNORECASE)
                    and re.search(r"blue", text, re.IGNORECASE))
    except Exception as e:
        print(f"WARNING: vision_check failed: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
def max_model_len() -> int:
    """Read ``max_model_len`` from the ``/v1/models`` endpoint."""
    r = requests.get(f"{BASE_URL}/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json()["data"][0]
    # vLLM exposes max_model_len at the top level of the model object, or
    # nested under a quantization/config dict depending on version.
    mml = data.get("max_model_len")
    if mml is None:
        # Fallback: some vLLM versions nest it differently.
        for key in ("max_context_length", "context_length"):
            mml = data.get(key)
            if mml is not None:
                break
    return int(mml) if mml is not None else 0


# ---------------------------------------------------------------------------
# Long-context probe
# ---------------------------------------------------------------------------
def long_ctx_check(target_tokens: int = 200000) -> bool:
    """Check whether the server accepts a prompt exceeding ``target_tokens``.

    Builds a prompt by repeating a sentence enough times to exceed the
    target token count, then POSTs with ``max_tokens=1``.  Returns True iff
    the server returns HTTP 200.  Returns False on HTTP 400 (context too long)
    or any other error — never raises.
    """
    # Rough heuristic: ~1.3 tokens per word for English prose.  Over-build the
    # prompt so we comfortably exceed target_tokens even if the tokeniser is
    # efficient.  The sentence is 8 words ≈ 10 tokens.
    sentence = "The system processes each request with deterministic scheduling. "
    # ~10 tokens per sentence; over-build to exceed target with margin.
    # Calculate repetitions to exceed target_tokens with a 20% margin.
    reps = int(target_tokens / 10 * 1.2)
    prompt = sentence * reps
    try:
        r = requests.post(
            f"{BASE_URL}/v1/completions",
            json={"model": MODEL, "prompt": prompt,
                  "max_tokens": 1, "temperature": 0},
            timeout=300,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# GPU telemetry
# ---------------------------------------------------------------------------
def gpu_telemetry() -> dict:
    """Sample GPU clock, power, and temperature via ``nvidia-smi``.

    Returns::

        {'clocks_sm': float, 'power_draw': float, 'temperature_gpu': float}

    Returns all-zero values if ``nvidia-smi`` is unavailable or fails.
    Used to flag thermal/clock drift between boots.
    """
    result = {"clocks_sm": 0.0, "power_draw": 0.0, "temperature_gpu": 0.0}
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return result
        parts = r.stdout.strip().split(",")
        if len(parts) >= 3:
            result["clocks_sm"] = float(parts[0].strip())
            result["power_draw"] = float(parts[1].strip())
            result["temperature_gpu"] = float(parts[2].strip())
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def stats(values: list[float]) -> dict:
    """Return descriptive statistics for a list of values.

    Uses population standard deviation (``statistics.pstdev``).  When ``n < 2``
    the standard deviation is 0.0 (a single observation has no variance).

    Returns::

        {'mean': float, 'sd': float, 'n': int, 'min': float, 'max': float}
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "sd": 0.0, "n": 0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.pstdev(values) if n >= 2 else 0.0,
        "n": n,
        "min": min(values),
        "max": max(values),
    }
