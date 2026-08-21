#!/usr/bin/env python3
"""R32 zero-byte EXL3 scale/path co-refit.

The ``screen`` command is deliberately a broad MPS *surrogate* shortlist.  It
never imports or impersonates EXL3.  The ``encode`` command is the claim-bearing
path: it imports the pinned R30 adapter, invokes the stock CUDA encoder/Viterbi,
round-trips the actual FP16 ``suh``/``svh`` buffers, decodes in source basis and
records exact returned bytes.

Scale callbacks preserve the stock random signs, H128 transforms, K, codebook,
BlockLDLQ recurrence and serialized schema.  They rebuild the regularized matrix
from the stock five-tuple, so changing a decoder scale does not silently change
the unquantized target.  No callback runs at inference.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import sys
import shutil
import struct
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

DATA_MANIFEST_FILE_SHA256 = "68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37"
DATA_MANIFEST_CONTENT_SHA256 = "51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2"
SPLIT_MANIFEST_FILE_SHA256 = "a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc"
SPLIT_MANIFEST_CONTENT_SHA256 = "151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e"
# R30/R31 bind both identities: transport file for report lineage, content for semantics.
R30_SPLIT_MANIFEST_SHA256 = SPLIT_MANIFEST_FILE_SHA256
SPLIT_SELECTIONS = {
    "calibration": "490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259",
    "validation": "4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de",
    "untouched_test": "4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca",
}
SPLIT_AUDIT_SHA256 = "7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9"
FISHER_MANIFEST_FILE_SHA256 = "4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a"
FISHER_MANIFEST_CONTENT_SHA256 = "28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237"
DENSE_H_MANIFEST_CONTENT_SHA256 = "9f9d60127e61a1912385d6fdbfb9bb9e61e2929a0d326e9add3212de8932c69d"
R30_HARNESS_SHA256 = "d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d"
R30_SCHEMA_SHA256 = "275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8"
R30_EXTENSION_BINARY_SHA256 = "e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e"
R31_GATE_SHA256 = "f4fc059c03331905dca6ad7b0ad4ba0e6af515897e2fc90dfd82f1ce0e8e8482"
R31_CONTRACT_SHA256 = "e8e1d47694038bbec4aa6f4a4554c4b53e549d2082d87e07353e5d8d16a66783"
R31_PREREG_SHA256 = "75a81665c75761767a7c71d58f4d59c446a13d3d7b164c5a8b9da9070388a784"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
STOCK_SEED = 300030
SCALE_FAMILY_FIVE = (0.8, 0.9, 1.0, 1.1, 1.2)
SCALE_FAMILY_TWO = (2.0 / 3.0, 1.0)
ARM_CALLBACK_CONTRACTS = {
    "A0-S": ("fp16_stock_refit", None),
    "biip": ("biip", None),
    "finite-five": ("biip", "five"),
    "finite-two": ("biip", "two"),
    "path-frozen": ("frozen_scale", None),
    "scale-rerun": ("frozen_scale", None),
}
EPS = 1e-20


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: os.PathLike[str] | str, value: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical_json(value) + b"\n")
    os.replace(tmp, path)


def _load_json_checked(path: pathlib.Path, expected_file: str, expected_content: str) -> dict[str, Any]:
    if sha256_file(path) != expected_file:
        raise RuntimeError(f"frozen file hash mismatch: {path}")
    value = json.loads(path.read_text())
    if value.get("content_sha256") != expected_content:
        raise RuntimeError(f"frozen content hash mismatch: {path}")
    return value


def _npy_member_memmap(npz_path: pathlib.Path, key: str) -> np.memmap:
    """Memory-map an uncompressed .npy member without loading its full tensor."""
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{key} is compressed; zero-copy screening requires ZIP_STORED")
        with npz_path.open("rb") as handle:
            handle.seek(info.header_offset)
            local_header = handle.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", local_header)
        if fields[0] != 0x04034B50:
            raise RuntimeError(f"{key} has an invalid ZIP local header")
        data_start = info.header_offset + 30 + fields[-2] + fields[-1]
    with npz_path.open("rb") as handle:
        handle.seek(data_start)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        payload_offset = handle.tell()
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        order="F" if fortran else "C",
        offset=payload_offset,
    )


def _bf16_payload(block: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(block, dtype=np.float32)
    words = f32.view(np.uint32)
    if np.any(words & np.uint32(0xFFFF)):
        raise RuntimeError("screening census contains a value not exactly representable as BF16")
    return np.ascontiguousarray(words >> np.uint32(16), dtype="<u2").tobytes()


def _role_for_npz(key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    layer_text, short_role = key.split("_", 1)
    layer = int(layer_text[1:])
    aliases = {"qkv": "gdn_qkv", "z": "gdn_z", "out": "gdn_out"}
    role = aliases.get(short_role, short_role)
    matches = [r for r in records if r.get("layer") == layer and r.get("role") == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest record for {key}, found {len(matches)}")
    return matches[0]


def _geomean(x: Any) -> Any:
    import torch

    return torch.exp(torch.mean(torch.log(torch.clamp(x, min=1e-8))))


def _normalized_biip_scales(weight: Any) -> tuple[Any, Any]:
    """Weight-only MSE-screen BiIP factors; actual promotion uses real curvature."""
    import torch

    col = torch.clamp(torch.sum(weight.square(), dim=0), min=1e-12)
    row = torch.clamp(torch.sum(weight.square(), dim=1), min=1e-12)
    sx = torch.clamp(col.pow(-0.25), 0.1, 10.0)
    sg = torch.clamp(row.pow(-0.25), 0.1, 10.0)
    sx = sx / _geomean(sx)
    sg = sg / _geomean(sg)
    return sg, sx


def _round_family(values: Any, family: Iterable[float]) -> Any:
    import torch

    levels = torch.tensor(tuple(family), device=values.device, dtype=values.dtype)
    idx = torch.argmin(torch.abs(torch.log(values[..., None]) - torch.log(levels)), dim=-1)
    return levels[idx]


def _surrogate_quantize(weight: Any, sg: Any, sx: Any, K: int) -> tuple[Any, float, float]:
    """Fixed-grid affine surrogate used only to rank MPS shortlist directions."""
    import torch

    transformed = sg[:, None] * weight * sx[None, :]
    qmax = 2 ** (K - 1) - 1
    rms = torch.sqrt(torch.mean(transformed.square())).clamp(min=1e-8)
    best = None
    for multiplier in (0.8, 0.9, 1.0, 1.1, 1.2):
        step = (3.0 * rms * multiplier / qmax).clamp(min=1e-10)
        q = torch.clamp(torch.round(transformed / step), -qmax, qmax) * step
        decoded = q / sg[:, None] / sx[None, :]
        mse = torch.mean((decoded - weight).square())
        row = (float(mse.item()), float(multiplier), q)
        if best is None or row[0] < best[0]:
            best = row
    assert best is not None
    return best[2], best[0], best[1]


def _path_frozen_als(weight: Any, q: Any, iterations: int = 6, regularization: float = 0.05) -> tuple[Any, Any, Any]:
    """Positive two-sided rank-one scale fit with log shrinkage and FP16 in-loop."""
    import torch

    a = torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
    b = torch.ones(weight.shape[1], device=weight.device, dtype=weight.dtype)
    for _ in range(iterations):
        qb = q * b[None, :]
        a_fit = torch.sum(weight * qb, dim=1) / torch.clamp(torch.sum(qb.square(), dim=1), min=EPS)
        a_fit = torch.clamp(a_fit, 0.25, 4.0)
        a = torch.exp(torch.log(a_fit) / (1.0 + regularization))
        a = a.half().float()
        aq = q * a[:, None]
        b_fit = torch.sum(weight * aq, dim=0) / torch.clamp(torch.sum(aq.square(), dim=0), min=EPS)
        b_fit = torch.clamp(b_fit, 0.25, 4.0)
        b = torch.exp(torch.log(b_fit) / (1.0 + regularization))
        b = b.half().float()
        # Remove the exact a/b gauge freedom without changing their product.
        gauge = torch.sqrt(_geomean(a) / _geomean(b))
        a = (a / gauge).half().float()
        b = (b * gauge).half().float()
    decoded = a[:, None] * q * b[None, :]
    return decoded, a, b


def _screen_block(block: np.ndarray, K: int, device: str) -> dict[str, Any]:
    import torch

    weight = torch.from_numpy(np.array(block, dtype=np.float32, copy=True)).to(device)
    ones_g = torch.ones(weight.shape[0], device=device)
    ones_x = torch.ones(weight.shape[1], device=device)
    q0, mse0, g0 = _surrogate_quantize(weight, ones_g, ones_x, K)
    sg, sx = _normalized_biip_scales(weight)
    q_biip, mse_biip, gb = _surrogate_quantize(weight, sg, sx, K)
    sg5, sx5 = _round_family(sg, SCALE_FAMILY_FIVE), _round_family(sx, SCALE_FAMILY_FIVE)
    _, mse5, g5 = _surrogate_quantize(weight, sg5, sx5, K)
    sg2, sx2 = _round_family(sg, SCALE_FAMILY_TWO), _round_family(sx, SCALE_FAMILY_TWO)
    _, mse2, g2 = _surrogate_quantize(weight, sg2, sx2, K)
    fit, a, b = _path_frozen_als(weight, q0)
    mse_fit = float(torch.mean((fit - weight).square()).item())
    # Re-run the fixed-grid path after the fitted decoder scales are folded into
    # the encoder transform. This is a surrogate for scale-frozen Viterbi rerun.
    sg_fit = torch.reciprocal(torch.clamp(a, min=1e-6))
    sx_fit = torch.reciprocal(torch.clamp(b, min=1e-6))
    _, mse_rerun, gr = _surrogate_quantize(weight, sg_fit, sx_fit, K)
    errors = {
        "A0_surrogate": mse0,
        "biip_magnitude": mse_biip,
        "finite_five": mse5,
        "finite_two": mse2,
        "path_frozen_als": mse_fit,
        "scale_frozen_rerun": mse_rerun,
    }
    best = min(errors, key=errors.get)
    return {
        "metric": "fixed-grid source-basis MSE surrogate (not EXL3, not KLD)",
        "errors": errors,
        "ratios_to_A0": {name: value / max(mse0, EPS) for name, value in errors.items()},
        "best_arm": best,
        "global_scale_multiplier": {
            "A0_surrogate": g0,
            "biip_magnitude": gb,
            "finite_five": g5,
            "finite_two": g2,
            "scale_frozen_rerun": gr,
        },
        "fp16_path_fit": True,
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    repo = pathlib.Path(args.repo)
    manifest_path = repo / "receipts/wave5/data-manifest.json"
    manifest = _load_json_checked(manifest_path, DATA_MANIFEST_FILE_SHA256, DATA_MANIFEST_CONTENT_SHA256)
    npz_path = pathlib.Path(args.weights)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    with zipfile.ZipFile(npz_path) as archive:
        keys = sorted(pathlib.Path(name).stem for name in archive.namelist() if name.endswith(".npy"))
    rows: list[dict[str, Any]] = []
    start = time.time()
    for key in keys:
        record = _role_for_npz(key, manifest["records"])
        array = _npy_member_memmap(npz_path, key)
        if tuple(array.shape) != tuple(record["shape"]):
            raise RuntimeError(f"shape mismatch for {key}")
        blocks = record["screening_blocks"]
        if len(blocks) != 8:
            raise RuntimeError(f"{key} must have exactly eight preregistered screening blocks")
        for block_spec in blocks:
            r0, c0 = block_spec["row"], block_spec["col"]
            block = np.asarray(array[r0:r0 + block_spec["rows"], c0:c0 + block_spec["cols"]])
            coordinates = {name: block_spec[name] for name in ("row", "col", "rows", "cols")}
            observed = sha256_bytes(canonical_json(coordinates) + b"\n" + _bf16_payload(block))
            if observed != block_spec["sha256"]:
                raise RuntimeError(f"BF16 block hash mismatch for {key}/{block_spec['kind']}")
            outcome = _screen_block(block, args.K, device)
            rows.append({
                "tensor_key": key,
                "tensor_name": record["tensor_name"],
                "tensor_sha256": record["sha256"],
                "layer": record["layer"],
                "role": record["role"],
                "topology": record["topology"],
                "block": block_spec,
                **outcome,
            })
        del array
    by_tensor: list[dict[str, Any]] = []
    for key in keys:
        subset = [row for row in rows if row["tensor_key"] == key]
        arms = tuple(subset[0]["errors"])
        macro = {arm: float(np.mean([r["ratios_to_A0"][arm] for r in subset])) for arm in arms}
        by_tensor.append({
            "tensor_key": key,
            "layer": subset[0]["layer"],
            "role": subset[0]["role"],
            "topology": subset[0]["topology"],
            "block_count": len(subset),
            "macro_ratio_to_A0": macro,
            "best_candidate_arm": min((a for a in arms if a != "A0_surrogate"), key=macro.get),
            "candidate_block_wins": sum(r["best_arm"] != "A0_surrogate" for r in subset),
        })
    role_depth: list[dict[str, Any]] = []
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in by_tensor:
        cells[(row["role"], row["layer"])].append(row)
    for (role, layer), subset in sorted(cells.items()):
        candidates = tuple(a for a in subset[0]["macro_ratio_to_A0"] if a != "A0_surrogate")
        ratios = {a: float(np.mean([x["macro_ratio_to_A0"][a] for x in subset])) for a in candidates}
        role_depth.append({"role": role, "layer": layer, "tensor_count": len(subset), "ratios": ratios, "winner": min(ratios, key=ratios.get)})
    receipt = {
        "schema": "qwen38-wave5-r32-scale-refit/1",
        "runner_sha256": sha256_file(__file__),
        "stage": "broad_mps_surrogate_screen",
        "status": "success",
        "claim_scope": "shortlist only; no EXL3 or KLD claim",
        "untouched_test_opened": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start,
        "config": {
            "device": device,
            "K": args.K,
            "weights_path": str(npz_path),
            "weights_sha256": sha256_file(npz_path),
            "tensor_count": len(keys),
            "blocks_per_tensor": 8,
            "block_count": len(rows),
            "scale_family_five": list(SCALE_FAMILY_FIVE),
            "scale_family_two": list(SCALE_FAMILY_TWO),
            "correct_bf16_check": "float32 low 16 bits zero; recovered uint16 block hash equals R29",
        },
        "foundation": foundation_identity(repo),
        "arm_semantics": {
            "A0_surrogate": "identity transform plus matched five-point global fixed-grid search",
            "biip_magnitude": "weight-only two-sided fourth-root balancing; real curvature required before encode",
            "finite_five": "nearest per-channel ratio in {0.8,0.9,1,1.1,1.2}; value lives in scale slot",
            "finite_two": "nearest per-channel ratio in {2/3,1}; value lives in scale slot",
            "path_frozen_als": "six positive ALS scale steps on one fixed surrogate path, FP16 each step, log shrinkage 0.05",
            "scale_frozen_rerun": "fixed-grid rerun after accepted fitted scales",
        },
        "by_tensor": by_tensor,
        "by_role_depth": role_depth,
        "win_counts": dict(Counter(row["best_arm"] for row in rows)),
        "raw_rows": rows,
        "selection_rule": "shortlist candidate mechanisms only; actual R30 encode and validation KLD decide",
        "int8_int4_disposition": "not screened as deployable: stock decoder reads FP16 suh/svh and has no int8/int4 scale semantics",
    }
    atomic_json(args.output, receipt)
    return receipt


def _r30_module_dir(repo: pathlib.Path) -> pathlib.Path:
    candidates = [
        pathlib.Path(os.environ["WAVE5_R30_MODULE_DIR"]) if os.environ.get("WAVE5_R30_MODULE_DIR") else None,
        repo / "tools/research/wave5",
        repo.parent / "qwen38-research-r30-exl3-harness/tools/research/wave5",
        pathlib.Path("/work"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "exl3_action.py").is_file():
            if sha256_file(candidate / "exl3_action.py") == R30_HARNESS_SHA256:
                return candidate
    raise RuntimeError("final pinned R30 module directory not found; set WAVE5_R30_MODULE_DIR")


def foundation_identity(repo: pathlib.Path) -> dict[str, Any]:
    r30_dir = _r30_module_dir(repo)
    paths = {
        "data_manifest": repo / "receipts/wave5/data-manifest.json",
        "split_manifest": repo / "receipts/wave5/split-manifest.json",
        "exl3_action": r30_dir / "exl3_action.py",
        "exl3_action_schema": r30_dir / "exl3_action_schema.json",
        "fidelity_gate": repo / "tools/research/wave5/fidelity_gate.py",
        "fidelity_contract": repo / "tools/research/wave5/fidelity_contract.json",
        "fidelity_prereg": repo / "receipts/wave5/fidelity-prereg.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "data_manifest": DATA_MANIFEST_FILE_SHA256,
        "split_manifest": SPLIT_MANIFEST_FILE_SHA256,
        "exl3_action": R30_HARNESS_SHA256,
        "exl3_action_schema": R30_SCHEMA_SHA256,
        "fidelity_gate": R31_GATE_SHA256,
        "fidelity_contract": R31_CONTRACT_SHA256,
        "fidelity_prereg": R31_PREREG_SHA256,
    }
    mismatches = {name: {"observed": observed[name], "expected": expected[name]} for name in expected if observed[name] != expected[name]}
    if mismatches:
        raise RuntimeError(f"frozen foundation hash mismatch: {mismatches}")
    fisher_path_text = os.environ.get("WAVE5_FISHER_MANIFEST")
    fisher_identity: dict[str, Any] = {
        "observed": False,
        "approved_file_sha256": FISHER_MANIFEST_FILE_SHA256,
        "approved_content_sha256": FISHER_MANIFEST_CONTENT_SHA256,
    }
    if fisher_path_text:
        fisher_path = pathlib.Path(fisher_path_text)
        fisher_file_sha256 = sha256_file(fisher_path)
        fisher_content_sha256 = json.loads(fisher_path.read_text()).get("content_sha256")
        if (
            fisher_file_sha256 != FISHER_MANIFEST_FILE_SHA256
            or fisher_content_sha256 != FISHER_MANIFEST_CONTENT_SHA256
        ):
            raise RuntimeError("observed Fisher manifest does not match the approved R29 pins")
        fisher_identity = {
            "observed": True,
            "path": str(fisher_path),
            "file_sha256": fisher_file_sha256,
            "content_sha256": fisher_content_sha256,
        }
    return {
        "files_sha256": observed,
        "r30_module_dir": str(r30_dir),
        "data_manifest_content_sha256": DATA_MANIFEST_CONTENT_SHA256,
        "split_manifest_file_sha256": SPLIT_MANIFEST_FILE_SHA256,
        "split_manifest_content_sha256": SPLIT_MANIFEST_CONTENT_SHA256,
        "r30_extension_binary_sha256": R30_EXTENSION_BINARY_SHA256,
        "fisher_manifest": fisher_identity,
    }


def _load_scale_vector(path: str, expected_sha256: str, expected_length: int) -> Any:
    import torch

    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"scale target artifact hash mismatch: {path}")
    value = np.load(path, allow_pickle=False)
    if value.shape != (expected_length,) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise RuntimeError(f"invalid scale target vector: {path}")
    return torch.from_numpy(value.copy())


def _stock_original_from_five_tuple(weight_r: Any, su: Any, sv: Any) -> Any:
    """Invert stock regularize using the returned scales (which include g_scale)."""
    from exllamav3.modules.quant.exl3_lib import quantize as qlib

    original = qlib.preapply_had_l(weight_r.clone(), qlib.had_k)
    original.mul_(su)
    original = qlib.preapply_had_r(original, qlib.had_n)
    original.mul_(sv)
    return original


def _regularize_for_fp16_scales(original: Any, suh: Any, svh: Any) -> Any:
    """Apply the stock transform against exactly representable FP16 decoder scales."""
    from exllamav3.modules.quant.exl3_lib import quantize as qlib

    work = original / svh
    qlib.blockwise_preapply_had_r_(work, qlib.had_n)
    work = work / suh
    qlib.blockwise_preapply_had_l_(work, qlib.had_k)
    return work

def validate_arm_callback_parameters(arm: str, parameters: Mapping[str, Any]) -> None:
    if arm not in ARM_CALLBACK_CONTRACTS:
        raise RuntimeError(f"no callback contract for arm {arm}")
    expected = ARM_CALLBACK_CONTRACTS[arm]
    observed = (parameters.get("mode"), parameters.get("family"))
    if observed != expected:
        raise RuntimeError(f"{arm} requires mode/family {expected}, got {observed}")



def r32_scale_callback(state: tuple[Any, Any, float, Any, Any], action: Any) -> tuple[Any, Any, float, Any, Any]:
    """R30 stock-five-tuple callback; no closures and no inference sidecar."""
    import torch

    apply, weight_r, _stock_g_scale, su_stock, sv_stock = state
    params = action.callback.parameters
    mode = params["mode"]
    original = _stock_original_from_five_tuple(weight_r, su_stock, sv_stock)
    su_mag = su_stock.abs()
    sv_mag = sv_stock.abs()
    if mode in {"biip", "frozen_scale"}:
        input_target = _load_scale_vector(params["input_target_path"], params["input_target_sha256"], su_stock.numel()).to(su_stock.device).reshape_as(su_stock)
        output_target = _load_scale_vector(params["output_target_path"], params["output_target_sha256"], sv_stock.numel()).to(sv_stock.device).reshape_as(sv_stock)
        # Targets are dimensionless decoder-magnitude shapes. Preserve each
        # stock vector's geometric mean so this arm does not steal a free global
        # scale search degree of freedom.
        input_target = input_target / torch.exp(torch.mean(torch.log(input_target.clamp(min=1e-8))))
        output_target = output_target / torch.exp(torch.mean(torch.log(output_target.clamp(min=1e-8))))
        strength = float(params.get("strength", 1.0))
        su_mag = torch.exp((1.0 - strength) * torch.log(su_mag.clamp(min=1e-8)) + strength * (torch.log(input_target) + torch.mean(torch.log(su_mag.clamp(min=1e-8)))))
        sv_mag = torch.exp((1.0 - strength) * torch.log(sv_mag.clamp(min=1e-8)) + strength * (torch.log(output_target) + torch.mean(torch.log(sv_mag.clamp(min=1e-8)))))
    elif mode == "uniform_family":
        su_mag = su_mag * float(params["input_multiplier"])
        sv_mag = sv_mag * float(params["output_multiplier"])
    elif mode != "fp16_stock_refit":
        raise RuntimeError(f"unknown R32 scale mode: {mode}")
    family = params.get("family")
    if family not in (None, "five", "two"):
        raise RuntimeError(f"unknown selector-free scale family: {family}")
    if family:
        ratios_in = su_mag / su_stock.abs().clamp(min=1e-8)
        ratios_out = sv_mag / sv_stock.abs().clamp(min=1e-8)
        levels = SCALE_FAMILY_FIVE if family == "five" else SCALE_FAMILY_TWO
        level_t = torch.tensor(levels, dtype=su_stock.dtype, device=su_stock.device)
        ratios_in = level_t[torch.argmin(torch.abs(torch.log(ratios_in[..., None]) - torch.log(level_t)), dim=-1)]
        level_o = level_t.to(sv_stock.device)
        ratios_out = level_o[torch.argmin(torch.abs(torch.log(ratios_out[..., None]) - torch.log(level_o)), dim=-1)]
        su_mag = su_stock.abs() * ratios_in
        sv_mag = sv_stock.abs() * ratios_out
    su_sign = torch.where(su_stock < 0, -torch.ones_like(su_stock), torch.ones_like(su_stock))
    sv_sign = torch.where(sv_stock < 0, -torch.ones_like(sv_stock), torch.ones_like(sv_stock))
    suh = (su_sign * su_mag).half().float()
    svh = (sv_sign * sv_mag).half().float()
    if torch.any(suh == 0) or torch.any(svh == 0):
        raise RuntimeError("R32 candidate produced a zero FP16 scale")
    rebuilt = _regularize_for_fp16_scales(original, suh, svh)
    return bool(apply), rebuilt.contiguous(), 1.0, suh.contiguous(), svh.contiguous()


def make_scale_callbacks(parameters: Mapping[str, Any]) -> Any:
    from exl3_action import EncodeCallbacks

    return EncodeCallbacks(
        scale=r32_scale_callback,
        identifier="r32.zero-byte-scale-refit",
        version="1",
        parameters=dict(parameters),
    )


def _split_contract() -> dict[str, Any]:
    from exl3_action import SplitDisjointness

    selections = {
        name: {
            "selection_sha256": digest,
            "selector": {"field": "split", "op": "eq", "value": name},
        }
        for name, digest in SPLIT_SELECTIONS.items()
    }
    return {
        "split_manifest_sha256": R30_SPLIT_MANIFEST_SHA256,
        "split_manifest_content_sha256": SPLIT_MANIFEST_CONTENT_SHA256,
        "split_selections": selections,
        "split_disjointness": SplitDisjointness(
            artifact_sha256=SPLIT_AUDIT_SHA256,
            predicate_language="wave5.split-predicate/1",
            pairwise_overlap_counts={
                "calibration__validation": 0,
                "calibration__untouched_test": 0,
                "validation__untouched_test": 0,
            },
            source_document_overlap_count=0,
            domain_leakage_count=0,
            verified=True,
        ),
        "evidence": {"local_metrics": {}, "promoted_kld": None},
    }


def _load_matrix(path: str, key: str | None = None) -> Any:
    import torch

    suffix = pathlib.Path(path).suffix
    if suffix == ".npy":
        return torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32, copy=False)).contiguous()
    if key:
        from safetensors import safe_open
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        except Exception as exc:
            if suffix == ".safetensors":
                raise RuntimeError(f"failed to load safetensors matrix {path}:{key}") from exc
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if not key:
            raise RuntimeError("matrix checkpoint is a dict; provide its key")
        value = value[key]
    return value
def _verify_manifest_artifact(
    manifest_path: str,
    artifact_path: str,
    *,
    file_field: str,
    expected_content_sha256: str,
    expected_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_file_sha256 = sha256_file(manifest_path)
    if expected_manifest_file_sha256 and manifest_file_sha256 != expected_manifest_file_sha256:
        raise RuntimeError(f"manifest file hash mismatch: {manifest_path}")
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    if manifest.get("content_sha256") != expected_content_sha256:
        raise RuntimeError(f"manifest content hash mismatch: {manifest_path}")
    artifact_name = pathlib.Path(artifact_path).name
    matches = [row for row in manifest.get("records", []) if row.get(file_field) == artifact_name]
    if len(matches) != 1:
        raise RuntimeError(f"manifest does not uniquely bind {artifact_name}")
    artifact_sha256 = sha256_file(artifact_path)
    if matches[0].get("sha256" if file_field == "file" else "fisher_sha256") != artifact_sha256:
        raise RuntimeError(f"artifact hash differs from manifest: {artifact_path}")
    return {
        "path": manifest_path,
        "file_sha256": manifest_file_sha256,
        "content_sha256": expected_content_sha256,
        "record": matches[0],
        "artifact_sha256": artifact_sha256,
    }




def fit_actual_frozen_path(args: argparse.Namespace) -> dict[str, Any]:
    """Fit only the existing FP16 scale slots while the stock trellis stays fixed.

    This is an encode-time proposal generator, not a deployable sidecar.  Its
    accepted scale shapes are subsequently passed through ``encode`` so the
    pinned stock Viterbi produces the final complete action.
    """
    import torch
    from safetensors import safe_open

    repo = pathlib.Path(args.repo)
    foundation = foundation_identity(repo)
    sys.path.insert(0, foundation["r30_module_dir"])
    from exl3_action import EncodedPayload, StockEXL3, load_action
    from exllamav3.modules.quant.exl3_lib import quantize as qlib

    action = load_action(args.action)
    source = _load_matrix(args.source, args.tensor_key).float()
    with safe_open(args.payload, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    payload = EncodedPayload(
        tensors=tensors,
        action=action,
        source_shape=tuple(source.shape),
        encoder_shape=(source.shape[1], source.shape[0]),
        source_layout="out_in",
        proxy_error=float("nan"),
        encode_seconds=0.0,
    )
    stock = StockEXL3()
    device = torch.device(args.device)
    payload.tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    stock_recon = stock.decode(payload).float()
    stock_mse = float(torch.mean((stock_recon - source.to(device)).square()).item())

    unit_tensors = dict(payload.tensors)
    unit_tensors["suh"] = torch.ones_like(unit_tensors["suh"])
    unit_tensors["svh"] = torch.ones_like(unit_tensors["svh"])
    unit_payload = dataclasses.replace(payload, tensors=unit_tensors)
    unit_encoder = stock.decode(unit_payload).T.contiguous().float()
    # Undo only the right H128. q_left is the stock reconstructed trellis after
    # the left H128 and before input scales.
    q_left = qlib.preapply_had_r(unit_encoder, qlib.had_n)
    target = source.T.contiguous().to(device)
    su_stock = payload.tensors["suh"].float()
    sv_stock = payload.tensors["svh"].float()
    su_sign = torch.where(su_stock < 0, -torch.ones_like(su_stock), torch.ones_like(su_stock))
    sv_sign = torch.where(sv_stock < 0, -torch.ones_like(sv_stock), torch.ones_like(sv_stock))
    su0, sv0 = su_stock.abs().clamp(min=1e-8), sv_stock.abs().clamp(min=1e-8)
    su, sv = su0.clone(), sv0.clone()
    accepted: list[dict[str, Any]] = []
    best_mse = stock_mse
    best_su, best_sv = su.clone(), sv.clone()
    for iteration in range(1, args.iterations + 1):
        # With output magnitudes fixed, invert the right H128 and solve one
        # positive scalar least-squares problem per input coordinate.
        target_before_right = qlib.preapply_had_r(
            target / (sv_sign * sv)[None, :],
            qlib.had_n,
        )
        q_signed = q_left * su_sign[:, None]
        raw_su = torch.sum(q_signed * target_before_right, dim=1) / torch.clamp(
            torch.sum(q_signed.square(), dim=1), min=EPS
        )
        raw_su = torch.clamp(raw_su, su0 * args.min_ratio, su0 * args.max_ratio)
        su = torch.exp(
            (torch.log(raw_su.clamp(min=1e-8)) + args.log_regularization * torch.log(su0))
            / (1.0 + args.log_regularization)
        ).half().float()

        before_output_scale = qlib.preapply_had_r(q_signed * su[:, None], qlib.had_n)
        target_signed = target / sv_sign[None, :]
        raw_sv = torch.sum(before_output_scale * target_signed, dim=0) / torch.clamp(
            torch.sum(before_output_scale.square(), dim=0), min=EPS
        )
        raw_sv = torch.clamp(raw_sv, sv0 * args.min_ratio, sv0 * args.max_ratio)
        sv = torch.exp(
            (torch.log(raw_sv.clamp(min=1e-8)) + args.log_regularization * torch.log(sv0))
            / (1.0 + args.log_regularization)
        ).half().float()
        candidate_encoder = before_output_scale * (sv_sign * sv)[None, :]
        candidate_source = candidate_encoder.T.contiguous()
        mse = float(torch.mean((candidate_source - source.to(device)).square()).item())
        strict = mse < best_mse
        accepted.append({"iteration": iteration, "source_basis_mse": mse, "strict_accept": strict})
        if strict:
            best_mse, best_su, best_sv = mse, su.clone(), sv.clone()

    # Verify the pure-Torch reconstruction through the real LinearEXL3 decoder.
    fitted_tensors = dict(payload.tensors)
    fitted_tensors["suh"] = (su_sign * best_su).half()
    fitted_tensors["svh"] = (sv_sign * best_sv).half()
    fitted_payload = dataclasses.replace(payload, tensors=fitted_tensors)
    fitted_recon = stock.decode(fitted_payload).float()
    decoder_mse = float(torch.mean((fitted_recon - source.to(device)).square()).item())
    if not math.isclose(decoder_mse, best_mse, rel_tol=2e-4, abs_tol=1e-12):
        raise RuntimeError("path-fit reconstruction does not agree with actual LinearEXL3 decode")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The callback deliberately reattaches the fresh stock geometric mean.
    # Only these dimensionless shapes are calibration-fitted.
    input_target = (best_su / _geomean(best_su)).cpu().numpy().astype(np.float32)
    output_target = (best_sv / _geomean(best_sv)).cpu().numpy().astype(np.float32)
    input_path, output_path = out_dir / "input-target.npy", out_dir / "output-target.npy"
    np.save(input_path, input_target, allow_pickle=False)
    np.save(output_path, output_target, allow_pickle=False)
    callback_parameters = {
        "mode": "frozen_scale",
        "strength": 1.0,
        "input_target_path": str(input_path),
        "input_target_sha256": sha256_file(input_path),
        "output_target_path": str(output_path),
        "output_target_sha256": sha256_file(output_path),
        "family": args.family,
        "fit_source_payload_sha256": action.hashes.get("payload_sha256"),
        "fit_split": "calibration",
        "fit_iterations": args.iterations,
        "log_regularization": args.log_regularization,
    }
    params_path = out_dir / "callback-parameters.json"
    atomic_json(params_path, callback_parameters)
    receipt = {
        "schema": "qwen38-wave5-r32-path-frozen-fit/1",
        "runner_sha256": sha256_file(__file__),
        "status": "success",
        "claim_scope": "encode-time scale proposal; stock trellis frozen; not KLD",
        "untouched_test_opened": False,
        "source_action": {"path": args.action, "sha256": sha256_file(args.action)},
        "source_payload": {"path": args.payload, "sha256": sha256_file(args.payload)},
        "stock_source_basis_mse": stock_mse,
        "best_fitted_source_basis_mse": best_mse,
        "actual_decoder_source_basis_mse": decoder_mse,
        "strict_improvement": best_mse < stock_mse,
        "iterations": accepted,
        "fp16_in_loop": True,
        "trellis_unchanged": True,
        "callback_parameters": callback_parameters,
        "callback_parameters_file": {"path": str(params_path), "sha256": sha256_file(params_path)},
        "inference_sidecar_bytes": 0,
        "foundation": foundation,
    }
    atomic_json(out_dir / "path-fit-receipt.json", receipt)
    return receipt


def prepare_biip(args: argparse.Namespace) -> dict[str, Any]:
    """Build encode-only dimensionless BiIP decoder-magnitude targets."""
    import torch
    h_binding = _verify_manifest_artifact(
        args.h_manifest,
        args.h,
        file_field="file",
        expected_content_sha256=DENSE_H_MANIFEST_CONTENT_SHA256,
    )

    source = _load_matrix(args.source, args.tensor_key).float()
    H = _load_matrix(args.h, args.h_key).float()
    if source.ndim != 2 or H.shape != (source.shape[1], source.shape[1]):
        raise RuntimeError("source must be [out,in] and H must be [in,in]")
    hdiag = torch.diag(H).clamp(min=1e-12)
    col = source.square().sum(dim=0).clamp(min=1e-12)
    row = source.square().sum(dim=1).clamp(min=1e-12)
    sx = torch.clamp((hdiag / col).pow(0.25), 0.1, 10.0)
    if args.output_importance:
        if not args.fisher_manifest:
            raise RuntimeError("--fisher-manifest is required with --output-importance")
        fisher_binding = _verify_manifest_artifact(
            args.fisher_manifest,
            args.output_importance,
            file_field="fisher_file",
            expected_content_sha256=FISHER_MANIFEST_CONTENT_SHA256,
            expected_manifest_file_sha256=FISHER_MANIFEST_FILE_SHA256,
        )
        importance = _load_matrix(args.output_importance, args.output_importance_key).float()
        if tuple(importance.shape) == tuple(source.shape):
            out = importance.sum(dim=1)
            output_basis = "provided squared-weight-gradient Fisher, reduced by input-coordinate sum"
        elif importance.ndim == 1 and importance.numel() == source.shape[0]:
            out = importance
            output_basis = "provided output-covariance/Fisher diagonal"
        else:
            raise RuntimeError("output importance must be [out] or match source [out,in]")
        out = out.clamp(min=1e-12)
    else:
        out = torch.ones(source.shape[0])
        output_basis = "identity output metric"
        fisher_binding = None
    sg = torch.clamp((out / row).pow(0.25), 0.1, 10.0)
    # Decoder scales apply the inverse of W' = S_G W S_X.
    input_target = (1.0 / sx).numpy().astype(np.float32)
    output_target = (1.0 / sg).numpy().astype(np.float32)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path, output_path = out_dir / "input-target.npy", out_dir / "output-target.npy"
    np.save(input_path, input_target, allow_pickle=False)
    np.save(output_path, output_target, allow_pickle=False)
    receipt = {
        "schema": "qwen38-wave5-r32-scale-target/1",
        "runner_sha256": sha256_file(__file__),
        "source": {"path": args.source, "key": args.tensor_key, "sha256": sha256_file(args.source), "shape": list(source.shape)},
        "curvature": {"path": args.h, "key": args.h_key, "sha256": sha256_file(args.h), "shape": list(H.shape), "basis": args.h_basis},
        "curvature_manifest": h_binding,
        "fisher_manifest": fisher_binding,
        "output_importance_basis": output_basis,
        "input_target": {"path": str(input_path), "sha256": sha256_file(input_path), "length": len(input_target), "dtype": "float32"},
        "output_target": {"path": str(output_path), "sha256": sha256_file(output_path), "length": len(output_target), "dtype": "float32"},
        "formula": "decoder input=(diag(H_X)/diag(W^T W))^-1/4; decoder output=(diag(H_G)/diag(W W^T))^-1/4; clamp transform to [0.1,10] before inverse",
        "inference_sidecar_bytes": 0,
    }
    atomic_json(out_dir / "scale-target.json", receipt)
    return receipt




def run_encode(args: argparse.Namespace) -> dict[str, Any]:
    """Run one actual pinned-R30 stock or zero-byte callback action."""
    import torch

    repo = pathlib.Path(args.repo)
    foundation = foundation_identity(repo)
    sys.path.insert(0, foundation["r30_module_dir"])
    from exl3_action import (
        COORDINATE_CONVENTIONS,
        EncodeCallbacks,
        StockEXL3,
        Unit,
        make_curvature_capture,
        make_stock_action,
        payload_digest,
        sha256_bytes as r30_sha256_bytes,
        tensor_digest,
    )
    source = _load_matrix(args.source, args.tensor_key)
    H = _load_matrix(args.h, args.h_key).float().contiguous()
    source_hash = r30_sha256_bytes(source.detach().contiguous().view(torch.uint8).numpy().tobytes())
    data_manifest = _load_json_checked(
        repo / "receipts/wave5/data-manifest.json",
        DATA_MANIFEST_FILE_SHA256,
        DATA_MANIFEST_CONTENT_SHA256,
    )
    source_records = [row for row in data_manifest["records"] if row.get("tensor_name") == args.tensor_key]
    if len(source_records) != 1 or source_records[0].get("sha256") != source_hash:
        raise RuntimeError("source tensor is not the bound R29 census record")
    h_binding = _verify_manifest_artifact(
        args.h_manifest,
        args.h,
        file_field="file",
        expected_content_sha256=DENSE_H_MANIFEST_CONTENT_SHA256,
    )
    unit = Unit(
        unit_id=args.unit_id,
        granularity="tensor",
        topology=args.topology,
        role=args.role,
        tensor_keys=(args.tensor_key,),
        layer_index=args.layer,
    )
    curvature = make_curvature_capture(
        H,
        unit,
        capture_id=args.capture_id,
        observation_count=args.observation_count,
        normalization=args.h_normalization,
        basis=args.h_basis,
        coordinate_convention=COORDINATE_CONVENTIONS["out_in"],
    )
    if args.arm == "A0":
        if args.callback_parameters:
            raise RuntimeError("A0 cannot accept callback parameters")
        callbacks = None
        parameters = None
    elif args.arm == "strength-zero":
        if args.callback_parameters:
            raise RuntimeError("strength-zero cannot accept callback parameters")
        callbacks = EncodeCallbacks.identity()
        parameters = {"strength": 0.0}
    else:
        if not args.callback_parameters:
            raise RuntimeError(f"{args.arm} requires callback parameters")
        parameters = json.loads(pathlib.Path(args.callback_parameters).read_text())
        validate_arm_callback_parameters(args.arm, parameters)
        callbacks = make_scale_callbacks(parameters)
    action = make_stock_action(
        action_id=args.action_id,
        unit=unit,
        K=args.K,
        codebook=args.codebook,
        seed=args.seed,
        source_sha256=source_hash,
        source_revision=SOURCE_REVISION,
        source_layout="out_in",
        curvature=curvature,
        callbacks=callbacks,
        **_split_contract(),
    )
    stock = StockEXL3()
    payload = stock.encode(source, H, action, callbacks=callbacks, device=args.device, verbose=args.verbose)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / f"{args.action_id}.safetensors"
    manifest = payload.serialize(payload_path)
    recon = stock.decode(payload)
    recon_digest = tensor_digest(recon)
    error = recon.float() - source.to(recon.device, dtype=torch.float32)
    mse = float(torch.mean(error.square()).item())
    H_metric = H.to(recon.device)
    oc_hwe = float(torch.sum((error @ H_metric) * error).item() / source.shape[0])
    hashes = {
        "payload_sha256": payload_digest(payload.tensors),
        "source_basis_reconstruction_sha256": recon_digest["sha256"],
    }
    complete_action = dataclasses.replace(
        payload.action,
        serialized=dict(manifest),
        hashes=hashes,
        evidence={"local_metrics": {"mse": mse, "oc_hwe": oc_hwe}, "promoted_kld": None},
    )
    action_path = out_dir / f"{args.action_id}.action.json"
    action_path.write_bytes(canonical_json(complete_action.to_dict()) + b"\n")
    receipt = {
        "schema": "qwen38-wave5-r32-actual-encode/1",
        "runner_sha256": sha256_file(__file__),
        "stage": "actual_R30_full_tensor",
        "status": "success",
        "untouched_test_opened": False,
        "arm": args.arm,
        "callback_parameters": parameters,
        "action": complete_action.to_dict(),
        "action_file": {"path": str(action_path), "sha256": sha256_file(action_path)},
        "payload_file": {"path": str(payload_path), "sha256": sha256_file(payload_path)},
        "source_manifest_record": source_records[0],
        "curvature_manifest": h_binding,
        "source_basis_reconstruction": recon_digest,
        "local_metrics": {"mse": mse, "oc_hwe": oc_hwe, "stock_proxy_error": payload.proxy_error},
        "exact_bytes": manifest,
        "zero_incremental_bytes": complete_action.runtime.incremental_hot_bytes == 0 and complete_action.runtime.sidecar_bytes == 0,
        "decode_hot_route": complete_action.runtime.route_id,
        "decode_hot_ops": list(complete_action.runtime.decode_hot_ops),
        "encode_seconds": payload.encode_seconds,
        "foundation": foundation,
        "int8_int4_disposition": "defer/new-format: LinearEXL3 consumes FP16 suh/svh; no stock int8/int4 scale payload semantics",
    }
    atomic_json(out_dir / f"{args.action_id}.receipt.json", receipt)
    return receipt


def run_self_test(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(32)
    W = torch.randn(128, 128, dtype=torch.float32) / 20
    q, mse, _ = _surrogate_quantize(W, torch.ones(128), torch.ones(128), 5)
    fit, a, b = _path_frozen_als(W, q)
    fit_mse = float(torch.mean((fit - W).square()).item())
    if not (math.isfinite(fit_mse) and a.dtype == torch.float32 and torch.equal(a, a.half().float())):
        raise RuntimeError("FP16 path-fit self-test failed")
    if fit_mse > mse * 1.000001:
        raise RuntimeError("path-frozen ALS must not increase its fitted fixed-path MSE")
    family = _round_family(torch.tensor([0.66, 0.79, 0.96, 1.08, 1.22]), SCALE_FAMILY_FIVE)
    if family.tolist() != [0.800000011920929, 0.800000011920929, 1.0, 1.100000023841858, 1.2000000476837158]:
        raise RuntimeError("finite scale-family selector self-test failed")
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory(prefix="r32-patch-self-test-") as temp_text:
        temp = pathlib.Path(temp_text)
        base = temp / "base"
        base.mkdir()
        prefix = "model.language_model.layers.0.mlp.gate_proj"
        base_tensors = {
            f"{prefix}.suh": torch.ones(128, dtype=torch.float16),
            f"{prefix}.svh": torch.ones(128, dtype=torch.float16),
            f"{prefix}.trellis": torch.zeros((8, 8, 80), dtype=torch.int16),
            f"{prefix}.mcg": torch.tensor(1, dtype=torch.int32),
            "other": torch.arange(7, dtype=torch.float32),
        }
        shard = "model-00001-of-00001.safetensors"
        save_file(base_tensors, str(base / shard))
        (base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: shard for name in base_tensors},
        }))
        candidate_payload = temp / "payload.safetensors"
        save_file({
            "suh": torch.full((128,), 2, dtype=torch.float16),
            "svh": torch.full((128,), 3, dtype=torch.float16),
            "trellis": torch.ones((8, 8, 80), dtype=torch.int16),
            "mcg": torch.tensor(2, dtype=torch.int32),
        }, str(candidate_payload))
        patch_result = patch_checkpoint(argparse.Namespace(
            base=str(base),
            output=str(temp / "candidate"),
            payload=str(candidate_payload),
            tensor_prefix=prefix,
            marker="mcg",
            receipt=None,
        ))
        if not (
            patch_result["exact_checkpoint_bytes_unchanged"]
            and patch_result["non_target_hashes_unchanged"]
            and patch_result["non_target_tensor_count_verified"] == 1
        ):
            raise RuntimeError("same-byte checkpoint patch self-test failed")
    repo = pathlib.Path(args.repo)
    foundation = foundation_identity(repo)
    sys.path.insert(0, foundation["r30_module_dir"])
    from exl3_action import COORDINATE_CONVENTIONS, Unit, make_curvature_capture, make_stock_action
    unit = Unit(
        unit_id="self-test.r32",
        granularity="tensor",
        topology="mlp",
        role="gate",
        tensor_keys=("self-test.weight",),
        layer_index=0,
    )
    H_contract = torch.eye(128, dtype=torch.float32)
    curvature = make_curvature_capture(
        H_contract,
        unit,
        capture_id="self-test.r32.identity-H",
        observation_count=1,
        normalization="identity/no-sample-normalization",
        basis="synthetic-contract-only",
        coordinate_convention=COORDINATE_CONVENTIONS["out_in"],
    )
    callbacks = make_scale_callbacks({"mode": "fp16_stock_refit", "family": None})
    action = make_stock_action(
        action_id="self-test.r32.callback",
        unit=unit,
        K=5,
        codebook="mcg",
        seed=STOCK_SEED,
        source_sha256=sha256_bytes(b"r32-contract-source"),
        source_revision=SOURCE_REVISION,
        source_layout="out_in",
        curvature=curvature,
        callbacks=callbacks,
        **_split_contract(),
    )
    if action.runtime.incremental_hot_bytes != 0 or action.runtime.sidecar_bytes != 0:
        raise RuntimeError("R32 action contract added inference bytes")
    callback_invoked = False
    callback_roundtrip_max_abs = None
    extension_binary_observed = None
    if args.actual_callback:
        callback_device = torch.device(
            args.callback_device
            if args.callback_device != "auto"
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        original = (torch.randn(128, 128, device=callback_device) / 20).float()
        su_sign = torch.where(
            torch.arange(128, device=callback_device)[:, None] % 2 == 0,
            torch.ones((128, 1), device=callback_device),
            -torch.ones((128, 1), device=callback_device),
        )
        sv_sign = torch.where(
            torch.arange(128, device=callback_device)[None, :] % 2 == 0,
            torch.ones((1, 128), device=callback_device),
            -torch.ones((1, 128), device=callback_device),
        )
        su = (su_sign * torch.linspace(0.75, 1.25, 128, device=callback_device)[:, None]).half().float()
        sv = (sv_sign * torch.linspace(0.8, 1.2, 128, device=callback_device)[None, :]).half().float()
        weight_r = _regularize_for_fp16_scales(original.clone(), su, sv)
        changed = r32_scale_callback((True, weight_r, 1.0, su, sv), action)
        if not isinstance(changed, tuple) or len(changed) != 5:
            raise RuntimeError("actual scale callback did not return the stock five-tuple")
        reconstructed = _stock_original_from_five_tuple(changed[1], changed[3], changed[4])
        callback_roundtrip_max_abs = float(torch.max(torch.abs(reconstructed - original)).item())
        if callback_roundtrip_max_abs > 2e-5:
            raise RuntimeError("actual callback five-tuple round-trip exceeded tolerance")
        if not torch.equal(changed[3], changed[3].half().float()) or not torch.equal(changed[4], changed[4].half().float()):
            raise RuntimeError("actual callback scales did not round-trip through FP16")
        import exllamav3_ext as extension_module
        extension_path = pathlib.Path(extension_module.__file__).resolve()
        extension_sha256 = sha256_file(extension_path)
        if extension_sha256 != R30_EXTENSION_BINARY_SHA256:
            raise RuntimeError("actual callback loaded an unqualified EXL3 extension binary")
        extension_binary_observed = {
            "path": str(extension_path),
            "sha256": extension_sha256,
        }
        callback_invoked = True
    for arm, (mode, family_name) in ARM_CALLBACK_CONTRACTS.items():
        validate_arm_callback_parameters(arm, {"mode": mode, "family": family_name})
    for invalid_arm, invalid_parameters in (
        ("finite-five", {"mode": "biip", "family": "two"}),
        ("biip", {"mode": "frozen_scale", "family": None}),
    ):
        try:
            validate_arm_callback_parameters(invalid_arm, invalid_parameters)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("arm/callback mismatch was not rejected")
    result = {
        "schema": "qwen38-wave5-r32-self-test/1",
        "runner_sha256": sha256_file(__file__),
        "status": "pass",
        "fixed_path_mse_before": mse,
        "fixed_path_mse_after": fit_mse,
        "fp16_roundtrip": True,
        "finite_family": family.tolist(),
        "final_r30_contract_action_valid": True,
        "actual_callback_invoked": callback_invoked,
        "actual_callback_roundtrip_max_abs": callback_roundtrip_max_abs,
        "incremental_hot_bytes": action.runtime.incremental_hot_bytes,
        "inference_sidecar_bytes": action.runtime.sidecar_bytes,
        "foundation": foundation,
        "extension_binary_observed": extension_binary_observed,
        "untouched_test_opened": False,
    }
    if args.output:
        atomic_json(args.output, result)
    return result


def _hash_range(path: pathlib.Path, offset: int, length: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"short read hashing {path}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _safetensors_header(path: pathlib.Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    return 8 + header_len, header


def patch_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Hardlink-clone a checkpoint and replace exactly one same-size EXL3 payload."""
    import torch
    from safetensors import safe_open

    base, output = pathlib.Path(args.base), pathlib.Path(args.output)
    if output.exists():
        raise RuntimeError(f"output checkpoint already exists: {output}")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    payload_names = ("suh", "svh", "trellis", args.marker)
    target_keys = {name: f"{args.tensor_prefix}.{name}" for name in payload_names}
    shards = {index["weight_map"].get(key) for key in target_keys.values()}
    if None in shards or len(shards) != 1:
        raise RuntimeError("target EXL3 buffers are missing or span multiple shards")
    shard_name = next(iter(shards))
    shutil.copytree(base, output, copy_function=os.link)
    out_shard = output / shard_name
    temporary = out_shard.with_suffix(out_shard.suffix + ".copy")
    shutil.copy2(base / shard_name, temporary)
    os.replace(temporary, out_shard)

    with safe_open(args.payload, framework="pt", device="cpu") as handle:
        candidate = {name: handle.get_tensor(name).contiguous() for name in payload_names}
    data_start, header = _safetensors_header(out_shard)
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    non_target_before: dict[str, str] = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        digest = _hash_range(base / shard_name, data_start + start, end - start)
        if key in target_keys.values():
            before[key] = digest
        else:
            non_target_before[key] = digest
    with out_shard.open("r+b") as handle:
        for payload_name, target_key in target_keys.items():
            entry = header[target_key]
            tensor = candidate[payload_name]
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            start, end = entry["data_offsets"]
            if len(raw) != end - start:
                raise RuntimeError(f"same-byte replacement violated for {target_key}")
            expected_shape = list(tensor.shape)
            if entry["shape"] != expected_shape:
                raise RuntimeError(f"shape mismatch for {target_key}: {entry['shape']} != {expected_shape}")
            dtype_code = {
                torch.float16: "F16",
                torch.int16: "I16",
                torch.int32: "I32",
            }.get(tensor.dtype)
            if entry["dtype"] != dtype_code:
                raise RuntimeError(f"dtype mismatch for {target_key}: {entry['dtype']} != {dtype_code}")
            handle.seek(data_start + start)
            handle.write(raw)
    non_target_mismatches = []
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        digest = _hash_range(out_shard, data_start + start, end - start)
        if key in target_keys.values():
            after[key] = digest
        elif digest != non_target_before[key]:
            non_target_mismatches.append(key)
    if non_target_mismatches:
        raise RuntimeError(f"non-target tensors changed: {non_target_mismatches[:3]}")
    base_files = {path.relative_to(base).as_posix(): sha256_file(path) for path in base.rglob("*") if path.is_file()}
    output_files = {path.relative_to(output).as_posix(): sha256_file(path) for path in output.rglob("*") if path.is_file()}
    changed_files = sorted(name for name in base_files if base_files[name] != output_files.get(name))
    if changed_files != [shard_name]:
        raise RuntimeError(f"only the target shard may change, got {changed_files}")
    receipt = {
        "schema": "qwen38-wave5-r32-one-tensor-checkpoint/1",
        "runner_sha256": sha256_file(__file__),
        "status": "success",
        "base_checkpoint": {"path": str(base), "file_count": len(base_files), "bytes": sum((base / name).stat().st_size for name in base_files)},
        "candidate_checkpoint": {"path": str(output), "file_count": len(output_files), "bytes": sum((output / name).stat().st_size for name in output_files)},
        "target_prefix": args.tensor_prefix,
        "payload": {"path": args.payload, "sha256": sha256_file(args.payload)},
        "changed_file": shard_name,
        "changed_buffers_before": before,
        "changed_buffers_after": after,
        "non_target_tensor_count_verified": len(non_target_before),
        "non_target_hashes_unchanged": True,
        "exact_checkpoint_bytes_unchanged": sum((base / name).stat().st_size for name in base_files) == sum((output / name).stat().st_size for name in output_files),
        "untouched_test_opened": False,
    }
    receipt_path = pathlib.Path(args.receipt) if args.receipt else output.parent / f"{output.name}.replacement.json"
    atomic_json(receipt_path, receipt)
    return receipt




def parser() -> argparse.ArgumentParser:
    script_path = pathlib.Path(__file__).resolve()
    inferred_root = script_path.parents[3] if len(script_path.parents) > 3 else pathlib.Path.cwd()
    root = pathlib.Path(os.environ.get("WAVE5_REPO", inferred_root))
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen", help="broad MPS surrogate shortlist")
    screen.add_argument("--repo", default=str(root))
    screen.add_argument("--weights", default="/Users/mbelleau/Projects/cleanroom/qwen38_wave5_weights.npz")
    screen.add_argument("--device", default="mps")
    screen.add_argument("--K", type=int, default=5)
    screen.add_argument("--output", default=str(root / "receipts/wave5/r32-scale-refit.json"))
    screen.set_defaults(func=run_screen)

    prep = sub.add_parser("prepare-biip", help="create curvature-aware encode-only scale targets")
    prep.add_argument("--source", required=True)
    prep.add_argument("--tensor-key")
    prep.add_argument("--h", required=True)
    prep.add_argument("--h-key")
    prep.add_argument("--h-basis", required=True)
    prep.add_argument("--output-importance")
    prep.add_argument("--output-importance-key")
    prep.add_argument("--h-manifest", required=True)
    prep.add_argument("--fisher-manifest")
    prep.add_argument("--output-dir", required=True)
    prep.set_defaults(func=prepare_biip)

    patch = sub.add_parser("patch-checkpoint", help="same-size one-tensor EXL3 checkpoint replacement")
    patch.add_argument("--base", required=True)
    patch.add_argument("--output", required=True)
    patch.add_argument("--payload", required=True)
    patch.add_argument("--tensor-prefix", required=True)
    patch.add_argument("--marker", choices=("mcg", "mul1"), default="mcg")
    patch.add_argument("--receipt")
    patch.set_defaults(func=patch_checkpoint)

    fit = sub.add_parser("fit-path", help="fit FP16 scales with an actual stock trellis frozen")
    fit.add_argument("--repo", default=str(root))
    fit.add_argument("--stock-root", required=True)
    fit.add_argument("--source", required=True)
    fit.add_argument("--tensor-key", required=True)
    fit.add_argument("--action", required=True)
    fit.add_argument("--payload", required=True)
    fit.add_argument("--iterations", type=int, choices=(1, 2, 3), default=3)
    fit.add_argument("--log-regularization", type=float, default=0.05)
    fit.add_argument("--min-ratio", type=float, default=0.5)
    fit.add_argument("--max-ratio", type=float, default=2.0)
    fit.add_argument("--family", choices=("five", "two"))
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--output-dir", required=True)
    fit.set_defaults(func=fit_actual_frozen_path)

    encode = sub.add_parser("encode", help="actual pinned R30 full-tensor encode")
    encode.add_argument("--repo", default=str(root))
    encode.add_argument("--stock-root", required=True)
    encode.add_argument("--source", required=True)
    encode.add_argument("--tensor-key", required=True)
    encode.add_argument("--h", required=True)
    encode.add_argument("--h-key")
    encode.add_argument("--h-basis", required=True)
    encode.add_argument("--h-normalization", required=True)
    encode.add_argument("--observation-count", type=int, required=True)
    encode.add_argument("--capture-id", required=True)
    encode.add_argument("--unit-id", required=True)
    encode.add_argument("--topology", choices=("mlp", "gdn", "full_attention", "lm_head", "mtp"), required=True)
    encode.add_argument("--h-manifest", required=True)
    encode.add_argument("--role", required=True)
    encode.add_argument("--layer", type=int, required=True)
    encode.add_argument("--K", type=int, choices=(4, 5, 6), required=True)
    encode.add_argument("--codebook", choices=("mcg", "mul1", "3inst"), default="mcg")
    encode.add_argument("--seed", type=int, default=STOCK_SEED)
    encode.add_argument("--arm", choices=("A0", "strength-zero", "A0-S", "biip", "finite-five", "finite-two", "path-frozen", "scale-rerun"), required=True)
    encode.add_argument("--callback-parameters")
    encode.add_argument("--action-id", required=True)
    encode.add_argument("--device", default="cuda:0")
    encode.add_argument("--output-dir", required=True)
    encode.add_argument("--verbose", action="store_true")
    encode.set_defaults(func=run_encode)

    test = sub.add_parser("self-test")
    test.add_argument("--repo", default=str(root))
    test.add_argument("--callback-device", default="auto")
    test.add_argument("--output")
    test.add_argument("--actual-callback", action="store_true")
    test.set_defaults(func=run_self_test)
    return p


def main() -> None:
    args = parser().parse_args()
    result = args.func(args)
    print(json.dumps({k: result[k] for k in result if k in {"schema", "stage", "status", "elapsed_seconds", "win_counts"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
