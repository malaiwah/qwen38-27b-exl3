#!/usr/bin/env python3
"""R35: exact Schur-conditioned legal-path refinement for stock EXL3.

The runner starts from a fresh stock BlockLDLQ payload, derives a forward sweep
whose unquantized suffix has been eliminated exactly by a block Schur complement,
and asks the pinned stock Viterbi kernel for each replacement path.  A replacement
is committed only when the same dense-H quadratic used for the stock action
strictly decreases.  The packed tensor schema and decoder are never changed.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/work")
from exl3_action import (  # type: ignore  # mounted frozen R30 harness
    STOCK_COMMIT,
    STOCK_EXTENSION_BINARY_SHA256,
    STOCK_QUANTIZE_SHA256,
    StockEXL3,
    payload_digest,
    tensor_digest,
)

SCHEMA = "qwen38-wave5-r35-schur-refine/1"
TILE = 16
HADAMARD = 128
DATA_MANIFEST_FILE_SHA256 = "68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37"
DATA_MANIFEST_CONTENT_SHA256 = "51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2"
SPLIT_MANIFEST_FILE_SHA256 = "a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc"
SPLIT_MANIFEST_CONTENT_SHA256 = "151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e"
FISHER_MANIFEST_FILE_SHA256 = "4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a"
FISHER_MANIFEST_CONTENT_SHA256 = "28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237"
CALIBRATION_SELECTION_SHA256 = "490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259"
VALIDATION_SELECTION_SHA256 = "4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de"
UNTOUCHED_SELECTION_SHA256 = "4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca"
LEAKAGE_AUDIT_SHA256 = "7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9"
HARNESS_SHA256 = "717de7845c3f3812396f089e47c9e8c6a4d59175b12efcc120742ff0ea6cfd87"
ACTION_SCHEMA_SHA256 = "896f29d70c42de9544bac0719fe0a287c31dadbdf3ddbeafa74a209e40fa1478"


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def dense_objective(weight: torch.Tensor, quantized: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    error = quantized - weight
    return (error * (H @ error)).sum(dtype=torch.float64)


def reverse_block_indices(n: int, device: torch.device) -> torch.Tensor:
    if n % TILE:
        raise ValueError("input dimension must be block-16 aligned")
    return torch.arange(n, device=device).view(-1, TILE).flip(0).reshape(-1)


def explicit_forward_schur_target(
    H: torch.Tensor, weight: torch.Tensor, quantized: torch.Tensor, start: int
) -> torch.Tensor:
    """Reference formula used only by self-test: eliminate the continuous suffix."""
    stop = start + TILE
    prefix = slice(0, start)
    suffix = slice(stop, H.shape[0])
    Hbb = H[start:stop, start:stop]
    if stop < H.shape[0]:
        Hbs = H[start:stop, suffix]
        Hss = H[suffix, suffix]
        Hsb = H[suffix, start:stop]
        Gbb = Hbb - Hbs @ torch.linalg.solve(Hss, Hsb)
        if start:
            Hbp = H[start:stop, prefix]
            Hsp = H[suffix, prefix]
            Gbp = Hbp - Hbs @ torch.linalg.solve(Hss, Hsp)
        else:
            Gbp = H.new_zeros((TILE, 0))
    else:
        Gbb = Hbb
        Gbp = H[start:stop, prefix]
    ep = quantized[prefix] - weight[prefix]
    return weight[start:stop] - torch.linalg.solve(Gbb, Gbp @ ep)


def block_ldl_forward_target(
    weight_rev: torch.Tensor,
    quantized_rev: torch.Tensor,
    L_rev: torch.Tensor,
    rev_start: int,
) -> torch.Tensor:
    """Exact suffix-continuous target from stock block-LDL on reversed blocks."""
    stop = rev_start + TILE
    target = weight_rev[rev_start:stop]
    if stop < weight_rev.shape[0]:
        target = target + L_rev[stop:, rev_start:stop].T @ (
            weight_rev[stop:] - quantized_rev[stop:]
        )
    return target


@dataclasses.dataclass
class RefinementState:
    H: torch.Tensor | None = None
    L_stock: torch.Tensor | None = None
    control_encoded: torch.Tensor | None = None
    control_weight_q: torch.Tensor | None = None
    sweep_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    candidate_calls: int = 0
    control_calls: int = 0


class SchurRefiner:
    def __init__(self, adapter: StockEXL3, sweeps: int):
        if sweeps not in (1, 2):
            raise ValueError("sweeps must be one or two")
        self.adapter = adapter
        self.qlib = adapter.qlib
        self.sweeps = sweeps
        self.state = RefinementState()

    @contextlib.contextmanager
    def patch(self) -> Iterator[None]:
        qlib = self.qlib
        original_block_ldl = qlib.block_ldl
        original_ldlq = qlib.ldlq
        original_quantize = qlib.quantize_tiles_multigpu

        def block_ldl_capture(H: torch.Tensor, b: int, quant_args: dict[str, Any], verbose: bool, debug_info: Any = None):
            L, H_after = original_block_ldl(H, b, quant_args, verbose, debug_info)
            self.state.H = H_after.detach().clone()
            self.state.L_stock = L.detach().clone()
            return L, H_after

        def ldlq_refine(weight: torch.Tensor, L: torch.Tensor, quant_args: dict[str, Any], pb: Any = None):
            stock_q, stock_encoded = original_ldlq(weight, L, quant_args, pb)
            if self.state.H is None or self.state.L_stock is None:
                raise RuntimeError("dense H/stock L were not captured")
            device = L.device
            H = self.state.H.to(device=device, dtype=torch.float32)
            n, out_features = weight.shape
            idx = reverse_block_indices(n, device)
            H_rev = H.index_select(0, idx).index_select(1, idx).contiguous()
            L_rev, _ = original_block_ldl(H_rev, TILE, quant_args, False, {"key": "r35/reversed-schur"})
            diagonal = torch.arange(n, device=device)
            L_rev[diagonal, diagonal] = 0
            weight_work = weight.to(device)
            candidate_q = stock_q.to(device).clone()
            candidate_encoded = stock_encoded.to(device).clone()
            control_q = stock_q.to(device).clone()
            control_encoded = stock_encoded.to(device).clone()
            initial_objective = dense_objective(weight_work, candidate_q, H)
            control_initial = dense_objective(weight_work, control_q, H)
            perm = qlib.tensor_core_perm(device)
            inv_perm = qlib.tensor_core_perm_i(device)
            tiles_n = out_features // TILE

            for sweep in range(self.sweeps):
                accepted_tiles = 0
                strict_delta_sum = 0.0
                before = dense_objective(weight_work, candidate_q, H)
                # Forward in source order. In reversed coordinates, already-fixed prefix
                # occupies the factor suffix, while the original suffix is continuous.
                weight_rev = weight_work.index_select(0, idx)
                candidate_rev = candidate_q.index_select(0, idx)
                for start in range(0, n, TILE):
                    rev_start = n - start - TILE
                    target = block_ldl_forward_target(weight_rev, candidate_rev, L_rev, rev_start)
                    target_tiles = target.reshape(TILE, tiles_n, TILE).permute(1, 0, 2).reshape(tiles_n, 256)
                    target_tiles = target_tiles[:, perm].contiguous()
                    q_tiles, q_idx = original_quantize(target_tiles, quant_args)
                    self.state.candidate_calls += 1
                    q_block = q_tiles[:, inv_perm].reshape(tiles_n, TILE, TILE).permute(1, 0, 2).reshape(TILE, out_features)

                    current = candidate_q[start:start + TILE]
                    delta = q_block - current
                    residual_block = (H[start:start + TILE] @ (candidate_q - weight_work))
                    Hbb = H[start:start + TILE, start:start + TILE]
                    delta_tiles = delta.reshape(TILE, tiles_n, TILE).permute(1, 0, 2)
                    residual_tiles = residual_block.reshape(TILE, tiles_n, TILE).permute(1, 0, 2)
                    quad = (delta_tiles * (torch.einsum("ab,tbc->tac", Hbb, delta_tiles))).sum((1, 2), dtype=torch.float64)
                    linear = 2.0 * (delta_tiles * residual_tiles).sum((1, 2), dtype=torch.float64)
                    changes = linear + quad
                    accept = changes < 0.0
                    if accept.any():
                        mask = accept.view(1, tiles_n, 1)
                        merged = torch.where(mask, q_block.reshape(TILE, tiles_n, TILE), current.reshape(TILE, tiles_n, TILE))
                        candidate_q[start:start + TILE] = merged.reshape(TILE, out_features)
                        candidate_encoded[start // TILE] = torch.where(accept[:, None], q_idx, candidate_encoded[start // TILE])
                        accepted_tiles += int(accept.sum().item())
                        strict_delta_sum += float(changes[accept].sum().item())
                        candidate_rev = candidate_q.index_select(0, idx)

                # Matched-call stock control: rerun the pinned stock BlockLDLQ
                # itself. This preserves its buffer-chunk accumulation order,
                # which is observably relevant for full tensors, and spends the
                # same one actual Viterbi call per input block as the candidate.
                control_q, control_encoded = original_ldlq(weight, L, quant_args, None)
                control_q = control_q.to(device)
                control_encoded = control_encoded.to(device)
                self.state.control_calls += n // TILE

                after = dense_objective(weight_work, candidate_q, H)
                control_after = dense_objective(weight_work, control_q, H)
                self.state.sweep_rows.append({
                    "sweep": sweep + 1,
                    "objective_before": float(before.item()),
                    "objective_after": float(after.item()),
                    "objective_delta": float((after - before).item()),
                    "accepted_tiles": accepted_tiles,
                    "strict_accepted_delta_sum": strict_delta_sum,
                    "control_objective": float(control_after.item()),
                    "candidate_calls_cumulative": self.state.candidate_calls,
                    "control_calls_cumulative": self.state.control_calls,
                })

            final_objective = dense_objective(weight_work, candidate_q, H)
            if final_objective > initial_objective or not math.isfinite(float(final_objective.item())):
                raise RuntimeError("strict-accept refinement violated monotone dense-H objective")
            if not torch.equal(control_encoded, stock_encoded.to(device)):
                raise RuntimeError("matched stock rerun changed the stock legal paths")
            if float(control_initial.item()) != float(dense_objective(weight_work, control_q, H).item()):
                raise RuntimeError("matched stock rerun changed the stock objective")
            self.state.control_encoded = control_encoded
            self.state.control_weight_q = control_q
            return candidate_q.to(stock_q.device), candidate_encoded.to(stock_encoded.device)

        qlib.block_ldl = block_ldl_capture
        qlib.ldlq = ldlq_refine
        try:
            yield
        finally:
            qlib.block_ldl = original_block_ldl
            qlib.ldlq = original_ldlq


def source_tensor(path: pathlib.Path, key: str, row: int | None = None, col: int | None = None) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        value = f.get_tensor(key)
    if row is not None and col is not None:
        value = value[row:row + HADAMARD, col:col + HADAMARD]
    return value.to(torch.float32).contiguous()


def load_h(path: pathlib.Path, col: int | None = None) -> torch.Tensor:
    value = np.load(path, mmap_mode="r")
    if col is not None:
        value = np.asarray(value[col:col + HADAMARD, col:col + HADAMARD])
    else:
        value = np.asarray(value)
    return torch.from_numpy(value.astype(np.float32, copy=True)).contiguous()


def raw_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


def encode_one(
    adapter: StockEXL3,
    source: torch.Tensor,
    H: torch.Tensor,
    K: int,
    sweeps: int,
    seed: int,
    tensor_key: str,
    target_kind: str,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    encoder_weight = source.T.contiguous().to(device)
    H_gpu = H.to(device)

    def args() -> dict[str, Any]:
        return {
            "seed": seed,
            "K": K,
            "devices": [device],
            "device_ratios": None,
            "apply_out_scales": True,
            "sigma_reg": 0.025,
            "buf_size_k": 128,
            "mcg": True,
        }

    stock_h = adapter.h_data(H_gpu.clone(), key=tensor_key, device=device, count=63)
    started = time.perf_counter()
    stock_weight_q, stock_proxy, stock_tensors = adapter.qlib.quantize_exl3(
        encoder_weight.clone(), stock_h, args(), True, verbose=False
    )
    stock_seconds = time.perf_counter() - started

    refined_h = adapter.h_data(H_gpu.clone(), key=tensor_key, device=device, count=63)
    refiner = SchurRefiner(adapter, sweeps)
    started = time.perf_counter()
    with refiner.patch():
        refined_weight_q, refined_proxy, refined_tensors = adapter.qlib.quantize_exl3(
            encoder_weight.clone(), refined_h, args(), True, verbose=False
        )
    refined_seconds = time.perf_counter() - started

    control_tensors = dict(refined_tensors)
    control_tensors["trellis"] = adapter.qlib.pack_trellis(refiner.state.control_encoded, args())
    stock_payload_sha = payload_digest(stock_tensors)
    control_payload_sha = payload_digest(control_tensors)
    if stock_payload_sha != control_payload_sha:
        raise RuntimeError("matched-call control payload is not fresh-stock identical")
    if raw_bytes(stock_tensors) != raw_bytes(refined_tensors):
        raise RuntimeError("refinement changed serialized raw bytes")
    if {k: (v.dtype, tuple(v.shape)) for k, v in stock_tensors.items()} != {
        k: (v.dtype, tuple(v.shape)) for k, v in refined_tensors.items()
    }:
        raise RuntimeError("refinement changed stock payload schema")

    def decode_payload(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        linear = adapter.LinearEXL3(
            None,
            encoder_weight.shape[0],
            encoder_weight.shape[1],
            suh=tensors["suh"],
            svh=tensors["svh"],
            trellis=tensors["trellis"],
            mcg=tensors["mcg"],
            key=tensor_key,
        )
        value = linear.get_weight_tensor()
        if tuple(value.shape) != tuple(encoder_weight.shape) or not torch.isfinite(value).all().item():
            raise RuntimeError("stock decoder returned an invalid source-basis reconstruction")
        return value

    decoded = decode_payload(refined_tensors)
    stock_decoded = decode_payload(stock_tensors)
    decode_max_abs = float((decoded - refined_weight_q).abs().max().item())
    decoded_stock_objective = float(dense_objective(encoder_weight, stock_decoded, H_gpu).item())
    decoded_refined_objective = float(dense_objective(encoder_weight, decoded, H_gpu).item())

    stock_obj = refiner.state.sweep_rows[0]["objective_before"] if refiner.state.sweep_rows else float("nan")
    final_obj = refiner.state.sweep_rows[-1]["objective_after"] if refiner.state.sweep_rows else float("nan")
    return {
        "tensor_key": tensor_key,
        "target_kind": target_kind,
        "K": K,
        "seed": seed,
        "sweeps": sweeps,
        "source_shape": list(source.shape),
        "encoder_shape": list(encoder_weight.shape),
        "stock_proxy_error": float(stock_proxy),
        "refined_proxy_error": float(refined_proxy),
        "dense_h_objective_stock": stock_obj,
        "dense_h_objective_refined": final_obj,
        "dense_h_relative_change": (final_obj / stock_obj - 1.0) if stock_obj else 0.0,
        "decoded_dense_h_objective_stock": decoded_stock_objective,
        "decoded_dense_h_objective_refined": decoded_refined_objective,
        "decoded_dense_h_relative_change": (
            decoded_refined_objective / decoded_stock_objective - 1.0
            if decoded_stock_objective else 0.0
        ),
        "sweep_rows": refiner.state.sweep_rows,
        "legal_viterbi_calls": {
            "candidate_refinement": refiner.state.candidate_calls,
            "matched_stock_control": refiner.state.control_calls,
            "matched": refiner.state.candidate_calls == refiner.state.control_calls,
        },
        "payload": {
            "stock_sha256": stock_payload_sha,
            "matched_control_sha256": control_payload_sha,
            "refined_sha256": payload_digest(refined_tensors),
            "raw_bytes_stock": raw_bytes(stock_tensors),
            "raw_bytes_refined": raw_bytes(refined_tensors),
            "schema_equal": True,
            "bytes_equal": True,
            "buffers": {name: tensor_digest(value) for name, value in sorted(refined_tensors.items())},
        },
        "decoder": {
            "route": "codec-exact/all-trellis-stock-exl3",
            "implementation": "LinearEXL3.get_weight_tensor",
            "decode_max_abs_vs_encoder_reconstruction": decode_max_abs,
            "hot_path_change": "none",
            "cross_tile_state_continuity": False,
            "tile_stride": "self-contained fixed 16x16",
        },
        "timing_seconds": {"fresh_stock_encode": stock_seconds, "refined_encode": refined_seconds},
    }


def role_h_file(layer: int, role: str) -> str:
    table = {
        (0, "down"): "h-x-00.npy",
        (0, "gate"): "h-x-01.npy",
        (0, "up"): "h-x-01.npy",
        (55, "down"): "h-x-02.npy",
        (55, "gate"): "h-x-03.npy",
        (55, "up"): "h-x-03.npy",
    }
    return table[(layer, role)]


def container_path(host_path: str) -> pathlib.Path:
    path = pathlib.Path(host_path)
    if "/models--Qwen--Qwen3.8-27B/" in host_path:
        return pathlib.Path("/model-repo") / pathlib.Path(host_path).relative_to(
            "/home/mbelleau/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B"
        )
    return path


def run(args: argparse.Namespace) -> None:
    manifest_path = pathlib.Path(args.data_manifest)
    if sha256_file(manifest_path) != DATA_MANIFEST_FILE_SHA256:
        raise RuntimeError("R29 data manifest file pin mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["content_sha256"] != DATA_MANIFEST_CONTENT_SHA256:
        raise RuntimeError("R29 data manifest content pin mismatch")
    adapter = StockEXL3()
    records = [
        r for r in manifest["records"]
        if r["layer"] in args.layers and r["role"] in args.roles and len(r["shape"]) == 2
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        tensor_path = container_path(record["shard_path"])
        H_path = pathlib.Path(args.h_root) / role_h_file(record["layer"], record["role"])
        if args.mode == "screen":
            blocks = record["screening_blocks"]
            for block_index, block in enumerate(blocks):
                if block["rows"] != HADAMARD or block["cols"] != HADAMARD:
                    continue
                source = source_tensor(tensor_path, record["tensor_name"], block["row"], block["col"])
                H = load_h(H_path, block["col"])
                for K in args.K:
                    result = encode_one(adapter, source, H, K, args.sweeps, args.seed, record["tensor_name"], "source")
                    result.update({
                        "layer": record["layer"],
                        "role": record["role"],
                        "scope": "screen-128x128",
                        "block_index": block_index,
                        "block": block,
                        "source_tensor_manifest_sha256": record["sha256"],
                        "dense_h_file": H_path.name,
                        "dense_h_file_sha256": sha256_file(H_path),
                    })
                    rows.append(result)
        else:
            source = source_tensor(tensor_path, record["tensor_name"])
            H = load_h(H_path)
            for K in args.K:
                result = encode_one(adapter, source, H, K, args.sweeps, args.seed, record["tensor_name"], "source")
                result.update({
                    "layer": record["layer"],
                    "role": record["role"],
                    "scope": "full-tensor",
                    "source_tensor_manifest_sha256": record["sha256"],
                    "dense_h_file": H_path.name,
                    "dense_h_file_sha256": sha256_file(H_path),
                })
                rows.append(result)

    receipt = {
        "schema": SCHEMA,
        "status": "success",
        "stage": "calibration-local-objective",
        "claim_scope": "actual-stock EXL3 same-schema legal-path screen; not KLD",
        "code_sha256": sha256_file(__file__),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pins": {
            "r30_harness_sha256": HARNESS_SHA256,
            "r30_action_schema_sha256": ACTION_SCHEMA_SHA256,
            "stock_commit": STOCK_COMMIT,
            "stock_quantize_sha256": STOCK_QUANTIZE_SHA256,
            "stock_extension_binary_sha256": STOCK_EXTENSION_BINARY_SHA256,
            "data_manifest_file_sha256": DATA_MANIFEST_FILE_SHA256,
            "data_manifest_content_sha256": DATA_MANIFEST_CONTENT_SHA256,
            "split_manifest_file_sha256": SPLIT_MANIFEST_FILE_SHA256,
            "split_manifest_content_sha256": SPLIT_MANIFEST_CONTENT_SHA256,
            "fisher_manifest_file_sha256": FISHER_MANIFEST_FILE_SHA256,
            "fisher_manifest_content_sha256": FISHER_MANIFEST_CONTENT_SHA256,
            "calibration_selection_sha256": CALIBRATION_SELECTION_SHA256,
            "validation_selection_sha256": VALIDATION_SELECTION_SHA256,
            "untouched_selection_sha256": UNTOUCHED_SELECTION_SHA256,
            "leakage_audit_sha256": LEAKAGE_AUDIT_SHA256,
        },
        "method": {
            "target": "forward block Schur conditional; original suffix eliminated continuously",
            "factorization": "pinned stock block_ldl(b=16) on reversed block ordering",
            "path_generator": "pinned actual quantize_tiles_multigpu/Viterbi",
            "acceptance": "strict negative delta of same full damped dense-H quadratic",
            "sweeps": args.sweeps,
            "control": "fresh stock BlockLDLQ plus matched count of actual Viterbi reruns",
            "correction_inside_action": True,
            "cross_tile_state_continuity": False,
        },
        "config": {"mode": args.mode, "layers": args.layers, "roles": args.roles, "K": args.K, "seed": args.seed},
        "rows": rows,
    }
    receipt["summary"] = summarize(rows)
    receipt["content_sha256"] = canonical_sha256({k: v for k, v in receipt.items() if k != "content_sha256"})
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], sort_keys=True))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}
    improved = [r for r in rows if r["dense_h_relative_change"] < 0]
    changed = [r for r in rows if r["payload"]["stock_sha256"] != r["payload"]["refined_sha256"]]
    return {
        "row_count": len(rows),
        "improved_rows": len(improved),
        "changed_payload_rows": len(changed),
        "all_schema_equal": all(r["payload"]["schema_equal"] for r in rows),
        "all_bytes_equal": all(r["payload"]["bytes_equal"] for r in rows),
        "all_controls_stock_identical": all(r["payload"]["stock_sha256"] == r["payload"]["matched_control_sha256"] for r in rows),
        "all_viterbi_budgets_matched": all(r["legal_viterbi_calls"]["matched"] for r in rows),
        "best_relative_change": min(r["dense_h_relative_change"] for r in rows),
        "median_relative_change": float(np.median([r["dense_h_relative_change"] for r in rows])),
        "total_accepted_tiles": sum(sum(s["accepted_tiles"] for s in r["sweep_rows"]) for r in rows),
    }


def self_test() -> None:
    torch.manual_seed(35)
    n, out = 48, 7
    A = torch.randn(n, n, dtype=torch.float64)
    H = A @ A.T + 0.5 * torch.eye(n, dtype=torch.float64)
    W = torch.randn(n, out, dtype=torch.float64)
    Q = W + 0.1 * torch.randn_like(W)
    # A small exact block-LDL implementation is used only to verify the target identity.
    idx = reverse_block_indices(n, torch.device("cpu"))
    Hr = H[idx][:, idx]
    Lc = torch.linalg.cholesky(Hr)
    blocks = n // TILE
    DL = torch.diagonal(Lc.reshape(blocks, TILE, blocks, TILE), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    L = Lc.view(n, blocks, TILE)
    for i in range(blocks):
        L[:, i] = L[:, i] @ DL[i]
    L = L.reshape(n, n)
    for i in range(blocks):
        L[i * TILE:(i + 1) * TILE, i * TILE:(i + 1) * TILE] = torch.eye(TILE, dtype=L.dtype)
    Wr, Qr = W[idx], Q[idx]
    for start in (0, 16, 32):
        rev_start = n - start - TILE
        via_ldl = block_ldl_forward_target(Wr, Qr, L, rev_start)
        explicit = explicit_forward_schur_target(H, W, Q, start)
        mapped = via_ldl
        if not torch.allclose(mapped, explicit, atol=2e-10, rtol=2e-10):
            raise AssertionError(f"Schur target mismatch at {start}: {(mapped-explicit).abs().max().item()}")
    E = Q - W
    B = slice(16, 32)
    C = slice(0, 3)
    delta = torch.randn(TILE, 3, dtype=torch.float64) * 0.01
    before = dense_objective(W, Q, H)
    Q2 = Q.clone()
    Q2[B, C] += delta
    after = dense_objective(W, Q2, H)
    Rb = (H[B] @ E)[:, C]
    formula = 2 * (delta * Rb).sum() + (delta * (H[B, B] @ delta)).sum()
    if not torch.allclose(after - before, formula, atol=1e-10, rtol=1e-10):
        raise AssertionError("strict acceptance delta is not the full dense-H objective delta")
    print("r35_schur_refine self-test: pass")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run_p = sub.add_parser("run")
    run_p.add_argument("--data-manifest", required=True)
    run_p.add_argument("--h-root", required=True)
    run_p.add_argument("--output", required=True)
    run_p.add_argument("--mode", choices=("screen", "full"), default="screen")
    run_p.add_argument("--layers", type=lambda x: [int(v) for v in x.split(",")], default=[0, 55])
    run_p.add_argument("--roles", type=lambda x: x.split(","), default=["gate", "up", "down"])
    run_p.add_argument("--K", type=lambda x: [int(v) for v in x.split(",")], default=[4, 5, 6])
    run_p.add_argument("--sweeps", type=int, choices=(1, 2), default=2)
    run_p.add_argument("--seed", type=int, default=350035)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-test":
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
