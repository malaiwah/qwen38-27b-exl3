#!/usr/bin/env python3
"""Measure the sparse EXL3 route domain used by the frontier census.

The payloads in this probe are deterministic and structurally valid, but synthetic.
They qualify kernel routing and resource use; they do not establish model fidelity.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple


SCHEMA = "qwen38-frontier-runtime-routes/1"
LOG_SCHEMA = "qwen38-frontier-route-probe-log/1"
DEFAULT_RUNTIME_SHA = "b19029d2309b26c4942425e52b74a0e6dd5d141e"
GRAPH_MODE = "FULL_DECODE_ONLY"
SEED = 0x51A7E120
B12X_N_MIN = 5120
B12X_N_MAX = 36864
MAX_ROWS = 4096
COSINE_GATE = 0.999
MAX_RELATIVE_GATE = 0.05
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# (input features, output features, selected EXL3 bitrates)
CASE_DOMAIN: Tuple[Tuple[int, int, Tuple[int, ...]], ...] = (
    (5120, 17408, (4, 5, 6)),
    (17408, 5120, (5, 6)),
    (5120, 10240, (5, 6)),
    (5120, 6144, (5, 6)),
    (6144, 5120, (5, 6)),
    (5120, 12288, (5, 6)),
    (5120, 1024, (5, 6)),
    (10240, 5120, (4,)),
    (5120, 248320, (6,)),
)


class ProbeError(RuntimeError):
    """A route observation cannot be emitted safely."""


def fail(message: str) -> NoReturn:
    raise ProbeError(message)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the synthetic SM120 EXL3/B12X frontier route domain."
    )
    parser.add_argument("--out", required=True, type=Path, help="strict JSON route manifest")
    parser.add_argument("--log", required=True, type=Path, help="raw JSONL measurement log")
    parser.add_argument("--runtime-sha", default=DEFAULT_RUNTIME_SHA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-rows", type=int, default=4)
    parser.add_argument("--prefill-rows", type=int, default=256)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args(argv)

    args.runtime_sha = str(args.runtime_sha).lower()
    if SHA_RE.fullmatch(args.runtime_sha) is None:
        parser.error("--runtime-sha must be a full 40-character lowercase hexadecimal SHA")
    if not 1 <= args.decode_rows < 128:
        parser.error("--decode-rows must be in 1..127 for the decode route class")
    if not 128 <= args.prefill_rows <= MAX_ROWS:
        parser.error("--prefill-rows must be in 128..%d for the prefill route class" % MAX_ROWS)
    if not 1 <= args.reps <= 100:
        parser.error("--reps must be in 1..100")
    if args.out.resolve() == args.log.resolve():
        parser.error("--out and --log must name different files")
    return args


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    validate_finite(value, "root")
    if pretty:
        text = json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        text += "\n"
    else:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return text.encode("utf-8")


def validate_finite(value: Any, where: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            fail("%s contains a non-finite number" % where)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                fail("%s contains a non-string JSON key" % where)
            validate_finite(child, "%s.%s" % (where, key))
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite(child, "%s[%d]" % (where, index))
        return
    if value is None or isinstance(value, (str, int, bool)):
        return
    fail("%s contains non-JSON value of type %s" % (where, type(value).__name__))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


class AtomicJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".%s." % self.path.name, dir=str(self.path.parent)
        )
        self.temporary = Path(temporary)
        self.handle = os.fdopen(descriptor, "wb")
        self.digest = hashlib.sha256()
        self.records = 0
        self.committed = False

    def write(self, value: Dict[str, Any]) -> None:
        payload = strict_json_bytes(value) + b"\n"
        self.handle.write(payload)
        self.digest.update(payload)
        self.records += 1

    def commit(self) -> str:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(str(self.temporary), str(self.path))
        fsync_directory(self.path.parent)
        self.committed = True
        return self.digest.hexdigest()

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        if not self.committed:
            try:
                self.temporary.unlink()
            except FileNotFoundError:
                pass


def module_file_identity(module: Any, label: str) -> Dict[str, Any]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        fail("%s does not expose an import file for identity pinning" % label)
    path = Path(raw_path).resolve()
    if not path.is_file():
        fail("%s import file does not exist: %s" % (label, path))
    return {"module": label, "file": path.name, "sha256": sha256_file(path)}


def package_version(distribution_names: Iterable[str]) -> Optional[str]:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def import_exl3_extension() -> Any:
    search_roots = [os.environ.get("EXLLAMAV3_EXT_PATH"), "/work/exllamav3", "/opt/exllamav3"]
    for raw_root in search_roots:
        if raw_root and Path(raw_root).is_dir() and raw_root not in sys.path:
            sys.path.insert(0, raw_root)
    errors: List[str] = []
    try:
        package = importlib.import_module("exllamav3.ext")
        extension = getattr(package, "exllamav3_ext")
        if callable(getattr(extension, "exl3_gemm", None)):
            return extension
    except Exception as exc:
        errors.append("exllamav3.ext: %s: %s" % (type(exc).__name__, exc))
    try:
        extension = importlib.import_module("exllamav3_ext")
        if callable(getattr(extension, "exl3_gemm", None)):
            return extension
        errors.append("exllamav3_ext: missing callable exl3_gemm")
    except Exception as exc:
        errors.append("exllamav3_ext: %s: %s" % (type(exc).__name__, exc))
    fail("cannot import a real EXL3 extension (%s)" % "; ".join(errors))
    raise AssertionError("unreachable")


def import_b12x_api() -> Tuple[Any, Any]:
    try:
        module = importlib.import_module("b12x.gemm.trellis_linear")
    except Exception as exc:
        fail("cannot import b12x.gemm.trellis_linear: %s: %s" % (type(exc).__name__, exc))
    api = module
    if not callable(getattr(api, "prepare_weight", None)) or not callable(getattr(api, "run", None)):
        try:
            api = importlib.import_module("b12x.gemm.trellis_linear.api")
        except Exception as exc:
            fail("B12X trellis module has no usable API: %s: %s" % (type(exc).__name__, exc))
    if not callable(getattr(api, "prepare_weight", None)) or not callable(getattr(api, "run", None)):
        fail("B12X trellis API must expose prepare_weight() and run()")
    return module, api


def device_uuid(torch: Any, index: int, properties: Any) -> Optional[str]:
    value = getattr(properties, "uuid", None)
    if value:
        return str(value)
    try:
        raw_uuids = torch.cuda._raw_device_uuid_nvml()  # type: ignore[attr-defined]
        if raw_uuids and index < len(raw_uuids) and raw_uuids[index]:
            return str(raw_uuids[index])
    except Exception:
        pass
    return None


def runtime_identity(
    torch: Any,
    ext: Any,
    b12x_module: Any,
    b12x_api: Any,
    device: Any,
    runtime_sha: str,
    import_ms: float,
) -> Dict[str, Any]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = device_uuid(torch, index, properties)
    gpu: Dict[str, Any] = {
        "device_index": int(index),
        "name": str(properties.name),
        "sm": int(properties.major) * 10 + int(properties.minor),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
    }
    if uuid is not None:
        gpu["uuid"] = uuid

    b12x_identity = module_file_identity(b12x_module, "b12x.gemm.trellis_linear")
    if b12x_api is not b12x_module:
        b12x_identity["api"] = module_file_identity(b12x_api, "b12x.gemm.trellis_linear.api")
    version = package_version(("b12x", "B12X"))
    if version is not None:
        b12x_identity["version"] = version

    ext_identity = module_file_identity(ext, "exllamav3_ext")
    version = package_version(("exllamav3", "exllamav3-python"))
    if version is not None:
        ext_identity["version"] = version

    return {
        "runtime_sha": runtime_sha,
        "python": sys.version.split()[0],
        "torch": {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "file": module_file_identity(torch, "torch"),
        },
        "exllamav3_ext": ext_identity,
        "b12x": b12x_identity,
        "gpu": gpu,
        "module_import_ms": float(import_ms),
    }


def tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def c_tmp_elements(rows: int, columns: int, sms: int) -> int:
    # The frontier policy only selects B12X at M >= 128. Keep the generic
    # scheduler's padded-M scratch at the boundary too; the K6-only small-M
    # runtime shortcut that uses one element is not valid for K4/K5.
    padded_rows = max(((rows + 47) // 48) * 48, ((rows + 63) // 64) * 64)
    return max(1, min(columns * padded_rows, sms * 4 * 64 * 256))


def selected_route(rows: int, output_features: int) -> str:
    if rows >= 128 and B12X_N_MIN <= output_features <= B12X_N_MAX:
        return "b12x.gemm.trellis_linear"
    return "exllamav3_ext.exl3_gemm"


def setup_exl3(
    torch: Any,
    ext: Any,
    x: Any,
    trellis: Any,
    suh: Any,
    svh: Any,
    bits: int,
    output_features: int,
) -> Tuple[Callable[[], None], Any, List[Any]]:
    del bits
    output = torch.empty(
        (int(x.shape[0]), output_features), dtype=torch.float16, device=x.device
    )
    x_had = torch.empty_like(x)

    def run() -> None:
        ext.exl3_gemm(
            x,
            trellis,
            output,
            suh,
            x_had,
            svh,
            -1,
            True,
            False,
            0,
        )

    return run, output, [output, x_had]


def setup_b12x(
    torch: Any,
    api: Any,
    x: Any,
    trellis: Any,
    suh: Any,
    svh: Any,
    mcg: Any,
    output_features: int,
    sms: int,
) -> Tuple[Callable[[], None], Any, List[Any]]:
    try:
        weight = api.prepare_weight(
            trellis,
            suh,
            svh,
            codebook="mcg",
            params_dtype=torch.float16,
            mcg=mcg,
        )
    except Exception as exc:
        fail("B12X prepare_weight failed: %s: %s" % (type(exc).__name__, exc))
    output = torch.empty(
        (int(x.shape[0]), output_features), dtype=torch.float16, device=x.device
    )
    gemm_output = torch.empty_like(output)
    c_tmp = torch.empty(
        (c_tmp_elements(int(x.shape[0]), output_features, sms),),
        dtype=torch.float32,
        device=x.device,
    )
    rotated_f16 = torch.empty_like(x)

    def run() -> None:
        api.run(
            x,
            weight,
            output=output,
            gemm_output=gemm_output,
            c_tmp=c_tmp,
            rotated_f16=rotated_f16,
        )

    return run, output, [output, gemm_output, c_tmp, rotated_f16, weight]


def synchronized_call(
    torch: Any, run: Callable[[], None], device: Any
) -> Tuple[float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start_event.record()
    run()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    gpu_ms = float(start_event.elapsed_time(end_event))
    if not math.isfinite(wall_ms) or not math.isfinite(gpu_ms) or wall_ms <= 0.0 or gpu_ms <= 0.0:
        fail("kernel timing produced a non-positive or non-finite duration")
    return wall_ms, gpu_ms


def all_finite(torch: Any, tensor: Any) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def compare_outputs(torch: Any, reference: Any, candidate: Any) -> Dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        fail("route comparison produced different output shapes")
    if not all_finite(torch, reference) or not all_finite(torch, candidate):
        fail("route comparison produced NaN or infinity")
    reference32 = reference.float()
    candidate32 = candidate.float()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference32.flatten(), candidate32.flatten(), dim=0
        ).item()
    )
    denominator = reference32.abs().amax().clamp_min(1.0e-6)
    max_relative = float(((candidate32 - reference32).abs().amax() / denominator).item())
    if not math.isfinite(cosine) or not math.isfinite(max_relative):
        fail("route comparison metrics are non-finite")
    passed = cosine >= COSINE_GATE and max_relative <= MAX_RELATIVE_GATE
    result = {
        "reference_route": "exllamav3_ext.exl3_gemm",
        "candidate_route": "b12x.gemm.trellis_linear",
        "cosine": cosine,
        "max_relative": max_relative,
        "cosine_gate": COSINE_GATE,
        "max_relative_gate": MAX_RELATIVE_GATE,
        "passed": passed,
    }
    if not passed:
        fail(
            "B12X/EXL3 numerical mismatch: cosine=%.8f (minimum %.6f), "
            "max_relative=%.8g (maximum %.6g)"
            % (cosine, COSINE_GATE, max_relative, MAX_RELATIVE_GATE)
        )
    return result


def measure_row(
    torch: Any,
    ext: Any,
    b12x_api: Any,
    device: Any,
    sms: int,
    trellis: Any,
    suh: Any,
    svh: Any,
    mcg: Any,
    input_features: int,
    output_features: int,
    bits: int,
    row_class: str,
    rows: int,
    seed: int,
    reps: int,
    log: AtomicJsonl,
) -> Dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.empty((rows, input_features), dtype=torch.float16, device=device)
    x.normal_(mean=0.0, std=0.05, generator=generator)
    route = selected_route(rows, output_features)
    expected = (
        "b12x.gemm.trellis_linear"
        if rows >= 128 and B12X_N_MIN <= output_features <= B12X_N_MAX
        else "exllamav3_ext.exl3_gemm"
    )
    if route != expected:
        fail("route policy selected %s, expected %s" % (route, expected))

    torch.cuda.synchronize(device)
    baseline_bytes = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    setup_start = time.perf_counter()
    if route == "b12x.gemm.trellis_linear":
        run, output, owned = setup_b12x(
            torch, b12x_api, x, trellis, suh, svh, mcg, output_features, sms
        )
    elif route == "exllamav3_ext.exl3_gemm":
        run, output, owned = setup_exl3(
            torch, ext, x, trellis, suh, svh, bits, output_features
        )
    else:
        fail("unknown selected route %s" % route)
    torch.cuda.synchronize(device)
    startup_ms = (time.perf_counter() - setup_start) * 1000.0

    def call_with_context(stage: str, callback: Callable[[], None]) -> Tuple[float, float]:
        try:
            return synchronized_call(torch, callback, device)
        except Exception as exc:
            fail(
                "%s %s failed for M=%d, K=%d, input=%d, output=%d: %s: %s"
                % (
                    route,
                    stage,
                    rows,
                    bits,
                    input_features,
                    output_features,
                    type(exc).__name__,
                    exc,
                )
            )
        raise AssertionError("unreachable")

    first_wall_ms, first_gpu_ms = call_with_context("first call", run)

    gpu_samples_us: List[float] = []
    wall_samples_us: List[float] = []
    for repetition in range(reps):
        wall_ms, gpu_ms = call_with_context("warm repetition %d" % repetition, run)
        wall_samples_us.append(wall_ms * 1000.0)
        gpu_samples_us.append(gpu_ms * 1000.0)
    torch.cuda.synchronize(device)
    peak_bytes = max(0, int(torch.cuda.max_memory_allocated(device)) - baseline_bytes)
    latency_us = float(statistics.median(gpu_samples_us))
    if not all_finite(torch, output):
        fail("%s produced NaN or infinity" % route)

    comparison: Optional[Dict[str, Any]] = None
    if route == "b12x.gemm.trellis_linear":
        reference_run, reference, reference_owned = setup_exl3(
            torch, ext, x, trellis, suh, svh, bits, output_features
        )
        call_with_context("EXL3 comparison call", reference_run)
        comparison = compare_outputs(torch, reference, output)
        log.write(
            {
                "schema": LOG_SCHEMA,
                "event": "comparison",
                "input_features": input_features,
                "output_features": output_features,
                "K": bits,
                "row_class": row_class,
                "M": rows,
                "metrics": comparison,
            }
        )
        del reference_run, reference, reference_owned

    resources = {
        "latency_us": latency_us,
        "scratch_bytes": peak_bytes,
        # This is the directly observed synchronized first invocation. It is an
        # upper bound on JIT/autotune because it also contains one kernel run.
        "jit_ms": float(first_wall_ms),
        "startup_ms": float(startup_ms),
        "first_call_ms": float(first_wall_ms),
        "first_call_gpu_ms": float(first_gpu_ms),
    }
    row: Dict[str, Any] = {
        "K": bits,
        "codebook": "mcg",
        "codebook_markers": {"codebook": "mcg", "mcg": True, "mul1": False},
        "scale_mode": "calibrated_g_scale",
        "topology": "split_qkv",
        "shape": [output_features, input_features],
        "alignment": {"K": 128, "N": 128},
        "N": output_features,
        "M": rows,
        "row_class": row_class,
        "graph_mode": GRAPH_MODE,
        "route": route,
        "measurement_kind": "measured",
        "measured": True,
        "resources": resources,
        "fallback": {"observed": False, "route": None, "measured": True},
        "payload": {
            "kind": "synthetic_structural_only",
            "seed": seed,
            "trellis_shape": [input_features // 16, output_features // 16, bits * 16],
            "trellis_dtype": "int16",
            "suh_shape": [input_features],
            "svh_shape": [output_features],
            "scale_dtype": "float16",
            "mcg_shape": [],
            "mcg_dtype": "int32",
        },
    }
    if comparison is not None:
        row["comparison"] = comparison

    log.write(
        {
            "schema": LOG_SCHEMA,
            "event": "measurement",
            "input_features": input_features,
            "output_features": output_features,
            "K": bits,
            "row_class": row_class,
            "M": rows,
            "route": route,
            "startup_ms": startup_ms,
            "first_call_wall_ms": first_wall_ms,
            "first_call_gpu_ms": first_gpu_ms,
            "warm_gpu_us": gpu_samples_us,
            "warm_wall_us": wall_samples_us,
            "median_gpu_us": latency_us,
            "peak_scratch_bytes": peak_bytes,
            "explicit_buffer_bytes": sum(
                tensor_bytes(item) for item in owned if hasattr(item, "numel")
            ),
        }
    )
    del run, output, owned, x
    return row


def make_case_payloads(
    torch: Any,
    device: Any,
    input_features: int,
    output_features: int,
    bits: int,
    seed: int,
) -> Tuple[Any, Any, Any, Any]:
    if input_features % 128 or output_features % 128:
        fail("case dimensions must both be divisible by 128")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    trellis = torch.randint(
        -32768,
        32768,
        (input_features // 16, output_features // 16, bits * 16),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    suh = torch.empty((input_features,), dtype=torch.float16, device=device)
    svh = torch.empty((output_features,), dtype=torch.float16, device=device)
    suh.uniform_(0.75, 1.25, generator=generator)
    svh.uniform_(0.75, 1.25, generator=generator)
    mcg = torch.full((), -877912083, dtype=torch.int32, device=device)
    expected_shape = (input_features // 16, output_features // 16, bits * 16)
    if tuple(trellis.shape) != expected_shape or trellis.dtype != torch.int16:
        fail("generated trellis violates the EXL3 structural contract")
    if tuple(suh.shape) != (input_features,) or tuple(svh.shape) != (output_features,):
        fail("generated scale vectors violate the EXL3 structural contract")
    if tuple(mcg.shape) != () or mcg.dtype != torch.int32:
        fail("generated MCG marker must be scalar int32")
    return trellis, suh, svh, mcg


def expected_route_keys(decode_rows: int, prefill_rows: int) -> set:
    del decode_rows, prefill_rows
    keys = set()
    for input_features, output_features, bitrates in CASE_DOMAIN:
        for bits in bitrates:
            keys.add((output_features, input_features, bits, "decode"))
            keys.add((output_features, input_features, bits, "prefill"))
    return keys


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    import_start = time.perf_counter()
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        fail("cannot import torch: %s: %s" % (type(exc).__name__, exc))
    ext = import_exl3_extension()
    b12x_module, b12x_api = import_b12x_api()
    import_ms = (time.perf_counter() - import_start) * 1000.0

    if not bool(torch.cuda.is_available()):
        fail("CUDA is unavailable; refusing to emit modeled route rows")
    try:
        device = torch.device(args.device)
    except Exception as exc:
        fail("invalid CUDA device %r: %s" % (args.device, exc))
    if device.type != "cuda":
        fail("--device must select CUDA, got %s" % device)
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
        device = torch.device("cuda:%d" % index)
    if index < 0 or index >= torch.cuda.device_count():
        fail("CUDA device index %d is outside the visible device set" % index)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(index)
    sm = int(properties.major) * 10 + int(properties.minor)
    if sm != 120:
        fail("runtime-route contract requires SM120, found SM%d on %s" % (sm, properties.name))

    tool_path = Path(__file__).resolve()
    identity = runtime_identity(
        torch, ext, b12x_module, b12x_api, device, args.runtime_sha, import_ms
    )
    identity["tool"] = {
        "path": "tools/frontier_route_probe.py",
        "sha256": sha256_file(tool_path),
    }

    log = AtomicJsonl(args.log)
    routes: List[Dict[str, Any]] = []
    try:
        log.write(
            {
                "schema": LOG_SCHEMA,
                "event": "start",
                "runtime": identity,
                "protocol": {
                    "seed": SEED,
                    "decode_rows": args.decode_rows,
                    "prefill_rows": args.prefill_rows,
                    "lm_head_prefill_rows": 1,
                    "reps": args.reps,
                    "synthetic_model_fidelity_claim": False,
                },
            }
        )
        case_index = 0
        for input_features, output_features, bitrates in CASE_DOMAIN:
            for bits in bitrates:
                payload_seed = SEED + case_index * 1009
                trellis, suh, svh, mcg = make_case_payloads(
                    torch,
                    device,
                    input_features,
                    output_features,
                    bits,
                    payload_seed,
                )
                log.write(
                    {
                        "schema": LOG_SCHEMA,
                        "event": "case",
                        "case_index": case_index,
                        "input_features": input_features,
                        "output_features": output_features,
                        "K": bits,
                        "payload_seed": payload_seed,
                        "payload_bytes": sum(
                            tensor_bytes(item) for item in (trellis, suh, svh, mcg)
                        ),
                    }
                )
                decode_seed = payload_seed + 1
                routes.append(
                    measure_row(
                        torch,
                        ext,
                        b12x_api,
                        device,
                        int(properties.multi_processor_count),
                        trellis,
                        suh,
                        svh,
                        mcg,
                        input_features,
                        output_features,
                        bits,
                        "decode",
                        args.decode_rows,
                        decode_seed,
                        args.reps,
                        log,
                    )
                )
                prefill_rows = 1 if output_features == 248320 else args.prefill_rows
                routes.append(
                    measure_row(
                        torch,
                        ext,
                        b12x_api,
                        device,
                        int(properties.multi_processor_count),
                        trellis,
                        suh,
                        svh,
                        mcg,
                        input_features,
                        output_features,
                        bits,
                        "prefill",
                        prefill_rows,
                        payload_seed + 2,
                        args.reps,
                        log,
                    )
                )
                del trellis, suh, svh, mcg
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                log.write(
                    {
                        "schema": LOG_SCHEMA,
                        "event": "case_cleanup",
                        "case_index": case_index,
                        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    }
                )
                case_index += 1

        observed_keys = {
            (row["shape"][0], row["shape"][1], row["K"], row["row_class"])
            for row in routes
        }
        expected_keys = expected_route_keys(args.decode_rows, args.prefill_rows)
        if observed_keys != expected_keys or len(routes) != len(expected_keys):
            fail(
                "route domain mismatch: emitted %d rows/%d keys, expected %d"
                % (len(routes), len(observed_keys), len(expected_keys))
            )
        if any(row["fallback"]["observed"] for row in routes):
            fail("a measured row observed a fallback")
        log.write(
            {
                "schema": LOG_SCHEMA,
                "event": "complete",
                "route_rows": len(routes),
                "unique_cases": case_index,
            }
        )
        log_sha = log.commit()
    except BaseException:
        log.abort()
        raise

    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "runtime_sha": args.runtime_sha,
        "sm": sm,
        "graph_mode": GRAPH_MODE,
        "payload_provenance": {
            "kind": "synthetic_structural_only",
            "deterministic_seed": SEED,
            "model_fidelity_claim": False,
            "statement": (
                "Random EXL3 payloads qualify route structure, parity, latency, and memory only; "
                "they do not represent checkpoint values or model fidelity."
            ),
        },
        "route_policy": {
            "b12x_when": {
                "minimum_rows": 128,
                "minimum_output_features": B12X_N_MIN,
                "maximum_output_features": B12X_N_MAX,
            },
            "otherwise": "exllamav3_ext.exl3_gemm",
            "lm_head_prefill_rows": 1,
        },
        "measurement_protocol": {
            "decode_rows": args.decode_rows,
            "prefill_rows": args.prefill_rows,
            "repetitions": args.reps,
            "latency": "median CUDA-event time over synchronized post-first-call repetitions",
            "startup_ms": "synchronized route preparation and explicit-buffer allocation",
            "jit_ms": (
                "synchronized first-call wall time; measured upper bound including one execution "
                "and any JIT/autotune"
            ),
            "scratch_bytes": (
                "peak CUDA allocator bytes above immutable payload/input baseline, including route "
                "output and explicit working buffers"
            ),
            "comparison": {
                "scope": "every row selected for B12X",
                "reference": "exllamav3_ext.exl3_gemm",
                "cosine_minimum": COSINE_GATE,
                "max_relative_maximum": MAX_RELATIVE_GATE,
            },
        },
        "identity": identity,
        "raw_log": {
            "schema": LOG_SCHEMA,
            "path": args.log.name,
            "sha256": log_sha,
            "records": log.records,
        },
        "routes": routes,
    }
    atomic_write(args.out, strict_json_bytes(manifest, pretty=True))
    return {
        "schema": SCHEMA,
        "out": str(args.out.resolve()),
        "log": str(args.log.resolve()),
        "log_sha256": log_sha,
        "routes": len(routes),
        "sm": sm,
        "runtime_sha": args.runtime_sha,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        result = run_probe(args)
        print(strict_json_bytes(result).decode("utf-8"))
        return 0
    except ProbeError as exc:
        print("frontier_route_probe: error: %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("frontier_route_probe: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            "frontier_route_probe: unexpected %s: %s" % (type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
