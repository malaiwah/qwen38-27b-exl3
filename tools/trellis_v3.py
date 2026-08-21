#!/usr/bin/env python3
"""Run matched RTN/LDLQ controls with the actual EXL3 Viterbi quantizer."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from frontier_common import (
    atomic_write_json,
    canonical_sha256,
    load_strict_json,
    sha256_file,
)

PLAN_SCHEMA = "qwen38-trellis-v3-run-plan/1"
RESULT_SCHEMA = "qwen38-trellis-v3-result/1"


class HarnessError(ValueError):
    """A fail-closed v3 harness or evidence error."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise HarnessError(f"{label} must be an object with string keys")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise HarnessError(f"{label} must be a lowercase SHA256")
    return value


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _validate_ref(root: Path, value: object, label: str) -> tuple[Path, dict[str, Any]]:
    ref = _object(value, label)
    if set(ref) != {"path", "sha256", "canonical_sha256"}:
        raise HarnessError(f"{label} must contain path/file/canonical hashes")
    raw = ref["path"]
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise HarnessError(f"{label}.path must be a POSIX path")
    path = (root / raw).resolve(strict=True)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise HarnessError(f"{label}.path escapes its evidence root") from exc
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise HarnessError(f"{label}.path is not a nonempty regular file")
    if sha256_file(path) != _sha256(ref["sha256"], f"{label}.sha256"):
        raise HarnessError(f"{label} whole-file hash differs")
    parsed = _object(load_strict_json(path), f"{label} content")
    if canonical_sha256(parsed) != _sha256(
        ref["canonical_sha256"], f"{label}.canonical_sha256"
    ):
        raise HarnessError(f"{label} canonical hash differs")
    return path, parsed


def _validate_plan(path: Path) -> dict[str, Any]:
    plan = _object(load_strict_json(path), "run plan")
    expected = {
        "schema",
        "experiment_id",
        "evidence_root",
        "captures",
        "quantizer",
        "arms",
        "external_gates",
    }
    if set(plan) != expected or plan["schema"] != PLAN_SCHEMA:
        raise HarnessError("run plan schema or keys differ")
    if not isinstance(plan["experiment_id"], str) or not plan["experiment_id"]:
        raise HarnessError("experiment_id must be nonempty")
    root = Path(plan["evidence_root"]).resolve(strict=True)
    captures = _object(plan["captures"], "captures")
    if set(captures) != {"bf16", "quant"}:
        raise HarnessError("captures must contain bf16 and quant")
    _, bf16 = _validate_ref(root, captures["bf16"], "captures.bf16")
    _, quant = _validate_ref(root, captures["quant"], "captures.quant")
    if (
        bf16.get("schema") != "qwen38-trellis-v3-capture/1"
        or bf16.get("flow") != "bf16"
    ):
        raise HarnessError("BF16 capture identity differs")
    if (
        quant.get("schema") != "qwen38-trellis-v3-capture/1"
        or quant.get("flow") != "quant"
    ):
        raise HarnessError("quant capture identity differs")
    quantizer = _object(plan["quantizer"], "quantizer")
    if set(quantizer) != {"source", "source_sha256", "commit", "K", "codebook", "seed"}:
        raise HarnessError("quantizer keys differ")
    source = Path(quantizer["source"]).resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise HarnessError("quantizer source must be a real directory")
    quantizer_source = (
        source / "exllamav3/modules/quant/exl3_lib/quantize.py"
    ).resolve(strict=True)
    expected_source_sha = _sha256(quantizer["source_sha256"], "quantizer.source_sha256")
    if sha256_file(quantizer_source) != expected_source_sha:
        raise HarnessError("actual EXL3 quantizer source hash differs")
    if not isinstance(quantizer["K"], int) or quantizer["K"] not in range(3, 9):
        raise HarnessError("quantizer K must be 3..8")
    if quantizer["codebook"] not in {"mcg", "mul1"}:
        raise HarnessError("quantizer codebook must be mcg or mul1")
    if not isinstance(quantizer["seed"], int):
        raise HarnessError("quantizer seed must be an integer")
    if plan["arms"] != ["rtn", "ldlq_bf16_h", "ldlq_quant_h"]:
        raise HarnessError("required control arms or order differ")
    gates = _object(plan["external_gates"], "external_gates")
    if set(gates) != {"whole_block", "end_logit"}:
        raise HarnessError("external_gates must name whole_block and end_logit")
    for name, gate in gates.items():
        row = _object(gate, f"external_gates.{name}")
        if set(row) != {"required_for_method_claim", "status", "evidence"}:
            raise HarnessError(f"external_gates.{name} keys differ")
        if row["required_for_method_claim"] is not True:
            raise HarnessError(f"external_gates.{name} must be mandatory")
        if row["status"] not in {"pending", "pass"}:
            raise HarnessError(f"external_gates.{name}.status differs")
        if row["status"] == "pass" and not isinstance(row["evidence"], dict):
            raise HarnessError(f"external_gates.{name} pass lacks evidence")
    plan["_capture_values"] = {"bf16": bf16, "quant": quant}
    plan["_evidence_root"] = str(root)
    return plan


def _load_capture_tensors(root: Path, capture: dict[str, Any]) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = _object(capture.get("tensors"), "capture.tensors")
    path = (root / tensors["path"]).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HarnessError("capture tensor path escapes evidence root") from exc
    if path.is_symlink() or sha256_file(path) != tensors["sha256"]:
        raise HarnessError("capture tensor payload hash differs")
    loaded = load_file(path, device="cpu")
    if sorted(loaded) != tensors["keys"]:
        raise HarnessError("capture tensor key set differs")
    return loaded


def _hessian_distortion(error: Any, hessian: Any) -> float:
    numerator = (error * (hessian @ error)).sum(dtype=error.dtype)
    return float(numerator.item())


def _relative_mse(candidate: Any, reference: Any) -> float:
    numerator = (candidate - reference).double().square().sum()
    denominator = reference.double().square().sum().clamp_min(1.0e-30)
    return float((numerator / denominator).item())


def _payload_bytes(tensors: dict[str, Any]) -> int:
    return sum(value.numel() * value.element_size() for value in tensors.values())


def _quantize(
    quantize_exl3: Any,
    weight: Any,
    hessian: Any | None,
    hessian_count: int,
    *,
    K: int,
    codebook: str,
    seed: int,
    device: Any,
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    import torch

    if hessian is None:
        h_tensor = torch.empty((weight.shape[0], weight.shape[0]), device="meta")
        count = 0
    else:
        h_tensor = hessian.to(device=device, dtype=torch.float32).clone()
        count = hessian_count
    h_data = {
        "H": h_tensor,
        "first_key": "trellis-v3-control",
        "count": count,
        "finalized": False,
        "num_total": int(weight.shape[0]),
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }
    quant_args: dict[str, Any] = {
        "seed": seed,
        "K": K,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": True,
        "debug_dir": "/work/trellis-v3-debug",
    }
    quant_args[codebook] = True
    reconstructed, proxy_error, payload = quantize_exl3(
        weight.to(device=device, dtype=torch.float32),
        h_data,
        quant_args,
        True,
        progress_str=None,
        verbose=False,
    )
    if reconstructed is None:
        raise HarnessError("actual EXL3 quantizer returned no reconstructed weight")
    return (
        reconstructed.detach().float().cpu().contiguous(),
        float(proxy_error),
        {key: value.detach().cpu().contiguous() for key, value in payload.items()},
        quant_args,
    )


def _atomic_save_tensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(plan_path: Path, out_path: Path) -> dict[str, Any]:
    import torch

    if out_path.exists() or out_path.with_suffix(".safetensors").exists():
        raise HarnessError("refusing to overwrite v3 output")
    plan = _validate_plan(plan_path.resolve(strict=True))
    root = Path(plan.pop("_evidence_root"))
    capture_values = plan.pop("_capture_values")
    bf16_tensors = _load_capture_tensors(root, capture_values["bf16"])
    quant_tensors = _load_capture_tensors(root, capture_values["quant"])
    if capture_values["bf16"]["target"] != capture_values["quant"]["target"]:
        raise HarnessError("capture target identities differ")
    bf16_slices = capture_values["bf16"]["capture"]["slices"]
    quant_slices = capture_values["quant"]["capture"]["slices"]
    geometry_fields = ("id", "input_start", "output_start", "size")
    bf16_geometry = [
        {field: row[field] for field in geometry_fields} for row in bf16_slices
    ]
    quant_geometry = [
        {field: row[field] for field in geometry_fields} for row in quant_slices
    ]
    if bf16_geometry != quant_geometry:
        raise HarnessError("capture slice geometry differs")

    source = Path(plan["quantizer"]["source"])
    sys.path.insert(0, str(source))
    from exllamav3.modules.quant.exl3_lib.quantize import (  # pyright: ignore[reportMissingImports]
        quantize_exl3,
    )

    if not torch.cuda.is_available():
        raise HarnessError("actual EXL3 controls require CUDA")
    device = torch.device("cuda:0")
    K = plan["quantizer"]["K"]
    codebook = plan["quantizer"]["codebook"]
    seed = plan["quantizer"]["seed"]
    result_rows: list[dict[str, Any]] = []
    output_tensors: dict[str, Any] = {}
    deterministic_payloads = True
    bf16_hessian_count = int(capture_values["bf16"]["capture"]["hessian_count"])
    quant_hessian_count = int(capture_values["quant"]["capture"]["hessian_count"])
    if bf16_hessian_count <= 0 or quant_hessian_count <= 0:
        raise HarnessError("capture Hessian counts must be positive")
    for slice_index, slice_row in enumerate(bf16_slices):
        key = slice_row["id"].replace("-", "_")
        weight = bf16_tensors[f"weight.{key}"].float()
        if not torch.equal(weight, quant_tensors[f"weight.{key}"].float()):
            raise HarnessError(f"weight differs between flows for {key}")
        h_bf16 = bf16_tensors[f"hessian.{key}"].float()
        h_quant = quant_tensors[f"hessian.{key}"].float()
        x_bf16 = bf16_tensors[f"activations.{key}"].float()
        x_quant = quant_tensors[f"activations.{key}"].float()
        if x_bf16.shape != x_quant.shape:
            raise HarnessError(f"paired activation shape differs for {key}")
        reference_output = x_bf16 @ weight
        running_noquant_output = x_quant @ weight
        noquant_floor = _relative_mse(running_noquant_output, reference_output)
        arm_rows: dict[str, Any] = {}
        for arm_index, arm in enumerate(plan["arms"]):
            hessian = (
                None if arm == "rtn" else h_bf16 if arm == "ldlq_bf16_h" else h_quant
            )
            hessian_count = (
                0
                if arm == "rtn"
                else bf16_hessian_count
                if arm == "ldlq_bf16_h"
                else quant_hessian_count
            )
            reconstructed, proxy_error, payload, quant_args = _quantize(
                quantize_exl3,
                weight,
                hessian,
                hessian_count,
                K=K,
                codebook=codebook,
                seed=seed + slice_index * 16 + arm_index,
                device=device,
            )
            candidate_symmetric = x_bf16 @ reconstructed
            candidate_running = x_quant @ reconstructed
            error = weight - reconstructed
            bytes_total = _payload_bytes(payload)
            prefix = f"{key}.{arm}"
            output_tensors[f"{prefix}.weight_q"] = reconstructed
            for payload_name, payload_tensor in payload.items():
                if payload_tensor.dtype == torch.uint32:
                    payload_tensor = payload_tensor.view(torch.int32)
                output_tensors[f"{prefix}.{payload_name}"] = payload_tensor
            arm_rows[arm] = {
                "proxy_error": proxy_error,
                "q_fallback": bool(quant_args.get("q_fallback")),
                "g_scale": float(quant_args["g_scale"]),
                "payload_bytes": bytes_total,
                "effective_bpw_with_sidecars": bytes_total * 8 / weight.numel(),
                "weight_mse": float((error.double().square().mean()).item()),
                "bf16_hessian_distortion": _hessian_distortion(
                    error.double(), h_bf16.double()
                ),
                "quant_hessian_distortion": _hessian_distortion(
                    error.double(), h_quant.double()
                ),
                "symmetric_output_relative_mse": _relative_mse(
                    candidate_symmetric, reference_output
                ),
                "running_output_relative_mse": _relative_mse(
                    candidate_running, reference_output
                ),
                "excess_running_relative_mse_over_noquant_floor": max(
                    0.0,
                    _relative_mse(candidate_running, reference_output) - noquant_floor,
                ),
                "reconstructed_weight_sha256": _tensor_sha256(reconstructed),
                "payload_sha256": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(payload.items())
                },
            }
            if arm == "ldlq_quant_h":
                replica, _, replica_payload, _ = _quantize(
                    quantize_exl3,
                    weight,
                    hessian,
                    hessian_count,
                    K=K,
                    codebook=codebook,
                    seed=seed + slice_index * 16 + arm_index,
                    device=device,
                )
                same = torch.equal(replica, reconstructed) and all(
                    name in replica_payload
                    and torch.equal(replica_payload[name], value)
                    for name, value in payload.items()
                )
                deterministic_payloads &= same
                arm_rows[arm]["deterministic_replay"] = same
        payload_sizes = {row["payload_bytes"] for row in arm_rows.values()}
        if len(payload_sizes) != 1:
            raise HarnessError(f"control arms have unmatched exact bytes for {key}")
        result_rows.append(
            {
                "slice": slice_row,
                "noquant_running_relative_mse_floor": noquant_floor,
                "arms": arm_rows,
            }
        )
    if not deterministic_payloads:
        raise HarnessError("seeded actual EXL3 replay changed payload bytes")
    tensor_path = out_path.with_suffix(".safetensors")
    _atomic_save_tensors(tensor_path, output_tensors)
    gates = plan["external_gates"]
    method_claims_allowed = all(row["status"] == "pass" for row in gates.values())
    result = {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "experiment_id": plan["experiment_id"],
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
            "canonical_sha256": canonical_sha256(load_strict_json(plan_path)),
        },
        "quantizer": plan["quantizer"],
        "capture_identities": {
            name: {
                "capture_id": value["capture_id"],
                "flow": value["flow"],
                "tensor_sha256": value["tensors"]["sha256"],
                "activation_rows": value["capture"]["activation_sample_rows"],
                "hessian_count": value["capture"]["hessian_count"],
            }
            for name, value in capture_values.items()
        },
        "controls": {
            "actual_exl3": True,
            "same_quantizer_and_exact_bytes": True,
            "correct_bf16_decode": True,
            "paired_real_activations": True,
            "noquant_running_floor": True,
            "seeded_payload_replay": deterministic_payloads,
        },
        "slices": result_rows,
        "payload": {
            "path": tensor_path.name,
            "sha256": sha256_file(tensor_path),
            "bytes": tensor_path.stat().st_size,
            "keys": sorted(output_tensors),
        },
        "external_gates": gates,
        "method_claims_allowed": method_claims_allowed,
        "promotion_rule": (
            "whole-block and end-logit gates passed"
            if method_claims_allowed
            else "method ranking and promotion remain prohibited until whole-block and end-logit gates pass"
        ),
    }
    atomic_write_json(out_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args.plan, args.out)
    print(
        f"PASS: {len(result['slices'])} real slices, actual EXL3 controls, "
        f"method_claims_allowed={result['method_claims_allowed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
