#!/usr/bin/env python3
"""R33: real-covariance shrinkage in the pinned stock EXL3 BlockLDLQ path.

This runner is deliberately an R30 client.  It never implements a quantizer,
GPTQ correction, or Viterbi search.  Curvature is changed at R30's curvature
callback, and finite-band variants zero only strict-lower off-diagonal blocks of
the factor returned by stock ``block_ldl(H, b=16)``.  The harness then executes
the unmodified reverse recurrence and pinned stock Viterbi.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any

import numpy as np

import exl3_action as r30

SCHEMA = "qwen38-wave5-r33-curvature-shrink/1"
R29_DATA_FILE_SHA256 = "68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37"
R29_DATA_CONTENT_SHA256 = "51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2"
R29_SPLIT_FILE_SHA256 = "a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc"
R29_SPLIT_CONTENT_SHA256 = "151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e"
R29_FISHER_FILE_SHA256 = "4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a"
R29_FISHER_CONTENT_SHA256 = "28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237"
R30_HARNESS_SHA256 = "d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d"
R30_SCHEMA_SHA256 = "275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8"
R31_GATE_SHA256 = "f4fc059c03331905dca6ad7b0ad4ba0e6af515897e2fc90dfd82f1ce0e8e8482"
R31_CONTRACT_SHA256 = "e8e1d47694038bbec4aa6f4a4554c4b53e549d2082d87e07353e5d8d16a66783"
R31_PREREG_SHA256 = "75a81665c75761767a7c71d58f4d59c446a13d3d7b164c5a8b9da9070388a784"
RHOS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
BANDS = (16, 32, 64)
SAMPLE_COUNTS = (8, 16, 32, 63)
BLOCK = 16
SEED = 330033

_TRACE: dict[str, Any] = {}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: Any) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def _sample_matrix(t: Any, rows: int = 128, cols: int = 512) -> Any:
    if t.ndim < 2:
        return t.reshape(1, -1)
    x = t.reshape(t.shape[0], -1)
    # Frozen evenly spaced coordinates prevent a prefix-only diagnostic.
    torch = __import__("torch")
    ri = torch.linspace(0, x.shape[0] - 1, min(rows, x.shape[0]), device=x.device).long()
    ci = torch.linspace(0, x.shape[1] - 1, min(cols, x.shape[1]), device=x.device).long()
    return x.index_select(0, ri).index_select(1, ci).float()


def _gaussian_rate_bits(x: Any) -> float:
    var = float(x.float().var(unbiased=False).item())
    return 0.5 * math.log2(2.0 * math.pi * math.e * max(var, 1e-30))


def _adjacent_innovation(t: Any) -> dict[str, float]:
    x = _sample_matrix(t)
    if x.shape[1] < 2:
        return {"marginal_rate_bits": 0.0, "innovation_rate_bits": 0.0, "gap_bits": 0.0, "predictor": 0.0}
    a, b = x[:, :-1], x[:, 1:]
    beta = float((a * b).sum().item() / max(float((a * a).sum().item()), 1e-30))
    innovation = b - beta * a
    marginal = _gaussian_rate_bits(b)
    rate = _gaussian_rate_bits(innovation)
    return {"marginal_rate_bits": marginal, "innovation_rate_bits": rate,
            "gap_bits": marginal - rate, "predictor": beta}


def target_identity(weight: Any, action: Any) -> Any:
    global _TRACE
    _TRACE["source_adjacent_innovation"] = _adjacent_innovation(weight)
    return weight


def scale_trace(state: tuple[Any, Any, float, Any, Any], action: Any) -> tuple[Any, Any, float, Any, Any]:
    global _TRACE
    _TRACE["post_stock_transform_adjacent_innovation"] = _adjacent_innovation(state[1])
    return state


def curvature_variant(H: Any, action: Any) -> Any:
    params = action.callback.parameters
    mode = params["curvature_mode"]
    if mode in ("stock", "band"):
        return H
    rho = float(params["rho"])
    d = H.diagonal().clone()
    if mode == "rho":
        out = H.mul(rho)
        out.diagonal().copy_(d)
        return out
    if mode == "ledoit_wolf":
        # Analytic control targets mu*I; rho is off-diagonal retention (1-alpha).
        mu = d.mean()
        out = H.mul(rho)
        out.diagonal().copy_(rho * d + (1.0 - rho) * mu)
        return out
    raise ValueError(f"unknown curvature mode {mode}")


def recurrence_band(L: Any, action: Any) -> Any:
    params = action.callback.parameters
    band = params.get("band_features")
    if band is None:
        return L
    band = int(band)
    if band not in BANDS or L.shape[0] % BLOCK:
        raise ValueError("band must be 16/32/64 on a block-16 factor")
    out = L.clone()
    # L is the exact stock lower block-unit factor. Preserve every diagonal 16x16
    # block and the requested number of immediately preceding block couplings.
    for row in range(0, out.shape[0], BLOCK):
        cutoff = max(0, row - (band - BLOCK))
        if cutoff:
            out[row:row + BLOCK, :cutoff].zero_()
    return out


def legal_path_trace(tiles: Any, q: Any, idx: Any, action: Any) -> Any:
    global _TRACE
    t = _sample_matrix(tiles)
    q_s = _sample_matrix(q)
    residual = t - q_s
    marginal = _gaussian_rate_bits(t)
    residual_rate = _gaussian_rate_bits(residual)
    snr_bits = marginal - residual_rate
    _TRACE.setdefault("post_recurrence_samples", []).append({
        "target_rate_bits": marginal,
        "stock_residual_rate_bits": residual_rate,
        "sampled_gaussian_snr_equivalent_bits": snr_bits,
        "normalized_mse": float((residual.square().mean() / t.square().mean().clamp_min(1e-30)).item()),
        "scope": "one sampled legal-path callback: transformed stock BlockLDLQ target versus pinned Viterbi reconstruction",
        "definition": "sampled Gaussian target-rate minus reconstruction-error rate; not a remaining-headroom/QMM oracle",
    })
    return tiles


def callbacks(parameters: dict[str, Any]) -> r30.EncodeCallbacks:
    return r30.EncodeCallbacks(
        target=target_identity,
        scale=scale_trace,
        curvature=curvature_variant,
        recurrence=recurrence_band,
        legal_path=legal_path_trace,
        identifier="r33/stock-block-ldl-curvature",
        version="1",
        parameters=parameters,
    )


def diagonal_target_ledoit_wolf_rho(X: Any) -> dict[str, float]:
    """Analytic Ledoit-Wolf controls for uncentered ``H=X.T@X/n``.

    ``alpha/rho`` are the standard spherical-target control used by the LW arm.
    The separately reported diagonal-target coefficient matches the H_rho family.
    Both are evaluated without a p x p x n temporary.
    """
    x = X.double()
    n, p = x.shape
    H = x.T @ x / n
    h_sq = float(H.square().sum().item())
    diag_sq = float(H.diagonal().square().sum().item())
    offdiag_energy = h_sq - diag_sq
    row_sq = x.square()
    outer_norm_sum = float(row_sq.sum(1).square().sum().item())
    outer_off_norm_sum = float((row_sq.sum(1).square() - row_sq.square().sum(1)).sum().item())

    mu = float(H.trace().item() / p)
    spherical_delta = max(0.0, h_sq - 2.0 * mu * float(H.trace().item()) + p * mu * mu)
    spherical_beta = max(0.0, (outer_norm_sum - n * h_sq) / (n * n))
    alpha = min(1.0, spherical_beta / max(spherical_delta, 1e-300))

    diagonal_beta = max(0.0, (outer_off_norm_sum - n * offdiag_energy) / (n * n))
    diagonal_alpha = min(1.0, diagonal_beta / max(offdiag_energy, 1e-300))
    return {
        "alpha": alpha,
        "rho": 1.0 - alpha,
        "target": "mu_identity",
        "mu": mu,
        "spherical_delta": spherical_delta,
        "variance_numerator": spherical_beta,
        "diagonal_target_alpha": diagonal_alpha,
        "diagonal_target_rho": 1.0 - diagonal_alpha,
        "offdiag_energy": offdiag_energy,
        "diagonal_variance_numerator": diagonal_beta,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def verify_foundation(repo: Path, r29: Path) -> dict[str, str]:
    expected = {
        repo / "receipts/wave5/data-manifest.json": R29_DATA_FILE_SHA256,
        repo / "receipts/wave5/split-manifest.json": R29_SPLIT_FILE_SHA256,
        repo / "tools/research/wave5/exl3_action.py": R30_HARNESS_SHA256,
        repo / "tools/research/wave5/exl3_action_schema.json": R30_SCHEMA_SHA256,
        repo / "tools/research/wave5/fidelity_gate.py": R31_GATE_SHA256,
        repo / "tools/research/wave5/fidelity_contract.json": R31_CONTRACT_SHA256,
        repo / "receipts/wave5/fidelity-prereg.json": R31_PREREG_SHA256,
        r29 / "fisher-selected-bf16/fisher-manifest.json": R29_FISHER_FILE_SHA256,
    }
    observed = {str(p): sha256_file(p) for p in expected}
    mismatch = {str(p): {"expected": h, "observed": observed[str(p)]} for p, h in expected.items() if observed[str(p)] != h}
    if mismatch:
        raise RuntimeError(f"foundation hash mismatch: {json.dumps(mismatch, sort_keys=True)}")
    return observed


def find_records(repo: Path, r29: Path, layer: int, role: str) -> dict[str, Any]:
    dm = load_json(repo / "receipts/wave5/data-manifest.json")
    source = next(r for r in dm["records"] if r["layer"] == layer and r["role"] == role)
    cap = load_json(r29 / "activations-bf16-reference/capture-manifest.json")
    module = f"language_model.model.layers.{layer}.mlp.gate_up_proj"
    activation = next(r for r in cap["records"] if r["module"] == module)
    dense = load_json(r29 / "dense-hx-bf16/h-x-manifest.json")
    hx = next((r for r in dense["records"] if r["module"] == module), None)
    fisher_manifest = load_json(r29 / "fisher-selected-bf16/fisher-manifest.json")
    fmodule = f"model.language_model.layers.{layer}.mlp.{role}_proj"
    fisher = next((r for r in fisher_manifest["records"] if r["module"] == fmodule), None)
    stats_manifest = load_json(r29 / "stats-bf16-reference/stats-manifest.json")
    stats = next(r for r in stats_manifest["records"] if r["module"] == module)
    return {"source": source, "activation": activation, "hx": hx, "fisher": fisher, "stats": stats}

def load_real_inputs(r29: Path, records: dict[str, Any], count: int | None = None) -> Any:
    import torch
    from safetensors.torch import load_file
    act = load_file(str(r29 / "activations-bf16-reference" / records["activation"]["file"]))["input"]
    X = act.float()
    if count is not None:
        X = X[:count]
    return X.contiguous()


def load_source(records: dict[str, Any]) -> tuple[Any, str]:
    import torch
    from safetensors import safe_open
    p = records["source"]["shard_path"]
    key = records["source"]["tensor_name"]
    with safe_open(p, framework="pt", device="cpu") as f:
        raw = f.get_tensor(key)
    raw_sha = tensor_sha256(raw)
    if raw_sha != records["source"]["sha256"]:
        raise RuntimeError("R29 source BF16 payload hash mismatch")
    return raw.float().contiguous(), raw_sha


def npy_float(path: Path) -> np.ndarray:
    a = np.load(path, mmap_mode="r")
    # ml_dtypes bfloat16 converts safely through astype.
    return np.asarray(a).astype(np.float32, copy=False)


def output_and_fisher_diagnostics(r29: Path, records: dict[str, Any]) -> dict[str, Any]:
    stats_path = r29 / "stats-bf16-reference" / records["stats"]["stats_file"]
    with np.load(stats_path) as z:
        keys = list(z.keys())
        out_key = next(k for k in keys if "output" in k and "diag" in k)
        output_diag = np.asarray(z[out_key], dtype=np.float64)
    out_rows = records["source"]["shape"][0]
    # Fused gate/up output: gate is the first half, up the second.
    if output_diag.size == 2 * out_rows:
        output_diag = output_diag[:out_rows] if records["source"]["role"] == "gate" else output_diag[out_rows:]
    result: dict[str, Any] = {
        "output_covariance_diag_key": out_key,
        "output_covariance_diag_count": int(output_diag.size),
        "output_covariance_diag_mean": float(output_diag.mean()),
        "recurrence_input": False,
        "warning": "H_G/output diagonal is diagnostic only; stock recurrence always receives real input H_X",
    }
    if records["fisher"] is not None:
        f = npy_float(r29 / "fisher-selected-bf16" / records["fisher"]["fisher_file"])
        fisher_row = np.empty(f.shape[0], dtype=np.float64)
        fisher_sum = 0.0
        for lo in range(0, f.shape[0], 256):
            fc = np.asarray(f[lo:lo + 256], dtype=np.float64)
            fisher_row[lo:lo + fc.shape[0]] = fc.mean(axis=1)
            fisher_sum += float(fc.sum())
        result.update({
            "fisher_diag_mean": fisher_sum / f.size,
            "fisher_row_mean_vs_output_covariance_pearson": pearson(fisher_row, output_diag),
            "fisher_definition": "one-sequence squared weight gradient; diagnostic/scoring only",
        })
    return result


def arm_parameters(kind: str, value: float | int | None, lw_rho: float) -> dict[str, Any]:
    if kind == "stock":
        return {"curvature_mode": "stock", "rho": 1.0, "band_features": None, "post_hoc_correction": False}
    if kind == "rho":
        return {"curvature_mode": "rho", "rho": float(value), "band_features": None, "post_hoc_correction": False}
    if kind == "band":
        return {"curvature_mode": "band", "rho": 1.0, "band_features": int(value), "post_hoc_correction": False}
    if kind == "ledoit_wolf":
        return {"curvature_mode": "ledoit_wolf", "rho": float(lw_rho), "band_features": None,
                "target": "mu_identity", "post_hoc_correction": False}
    raise ValueError(kind)


def metrics(source: Any, recon: Any, X_score: Any, output_diag: np.ndarray | None, fisher: np.ndarray | None) -> dict[str, float]:
    import torch
    e = (source - recon.float()).float()
    mse = float(e.square().mean().item())
    # Exact real-activation HWE without materializing E H E'.
    pred = X_score.to(e.device) @ e.T
    hx_hwe = float(pred.square().mean().item())
    result = {"mse": mse, "real_input_hwe": hx_hwe}
    row_mse = e.square().mean(1).double().cpu().numpy()
    if output_diag is not None and output_diag.size == row_mse.size:
        result["output_covariance_diag_hwe"] = float(np.dot(row_mse, output_diag) / max(output_diag.sum(), 1e-300))
    if fisher is not None and fisher.shape == tuple(e.shape):
        ecpu = e.detach().cpu().numpy()
        weighted, fisher_sum = 0.0, 0.0
        for lo in range(0, fisher.shape[0], 256):
            ec = np.asarray(ecpu[lo:lo + 256], dtype=np.float64)
            fc = np.asarray(fisher[lo:lo + 256], dtype=np.float64)
            weighted += float(np.sum(ec * ec * fc))
            fisher_sum += float(np.sum(fc))
        result["fisher_hwe"] = weighted / max(fisher_sum, 1e-300)
    return result


def payload_manifest(payload: Any) -> dict[str, Any]:
    manifest = payload.buffer_manifest()
    return {
        "payload_sha256": r30.payload_digest(payload.tensors),
        "buffer_bytes": manifest["buffer_bytes"],
        "dense_byte_law": manifest["dense_byte_law"],
        "buffers": manifest["buffers"],
        "incremental_hot_bytes": 0,
        "sidecar_bytes": 0,
        "shared_checkpoint_overhead_bytes": 0,
    }


def action_base(repo: Path) -> dict[str, Any]:
    a0 = load_json(repo / "receipts/wave5/stock-control.json")["a0_full_tensor"]["action"]
    return a0


def make_action(repo: Path, unit: Any, source: Any, H: Any, count: int,
                params: dict[str, Any], action_id: str) -> tuple[Any, Any]:
    base = action_base(repo)
    cb = callbacks(params)
    curvature = r30.make_curvature_capture(
        H, unit, capture_id=f"r29/real-bf16-hx/n{count}", observation_count=count,
        normalization="sum(X^T X); pinned stock finalize_capture_H divides by observation_count",
        basis="source-model input-feature activation basis",
        coordinate_convention=r30.COORDINATE_CONVENTIONS["out_in"],
    )
    split_disjointness = r30.SplitDisjointness(
        artifact_sha256="7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9",
        predicate_language="wave5.split-predicate/1",
        pairwise_overlap_counts={
            "calibration__validation": 0,
            "calibration__untouched_test": 0,
            "validation__untouched_test": 0,
        },
        source_document_overlap_count=0,
        domain_leakage_count=0,
        verified=True,
    )
    split_selections = {
        "calibration": {
            "selection_sha256": "490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259",
            "selector": {"field": "split", "op": "eq", "value": "calibration"},
        },
        "validation": {
            "selection_sha256": "4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de",
            "selector": {"field": "split", "op": "eq", "value": "validation"},
        },
        "untouched_test": {
            "selection_sha256": "4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca",
            "selector": {"field": "split", "op": "eq", "value": "untouched_test"},
        },
    }
    action = r30.make_stock_action(
        action_id=action_id, unit=unit, K=5, codebook="mcg", seed=SEED,
        source_sha256=tensor_sha256(source), source_revision=base["source_revision"],
        source_layout="out_in", curvature=curvature, callbacks=cb,
        split_manifest_sha256=R29_SPLIT_FILE_SHA256,
        split_manifest_content_sha256=R29_SPLIT_CONTENT_SHA256,
        split_selections=split_selections, split_disjointness=split_disjointness,
        evidence={"local_metrics": {}, "promoted_kld": None},
    )
    return action, cb


def extract_output_diag(r29: Path, records: dict[str, Any]) -> np.ndarray | None:
    p = r29 / "stats-bf16-reference" / records["stats"]["stats_file"]
    with np.load(p) as z:
        key = next((k for k in z.keys() if "output" in k and "diag" in k), None)
        if key is None:
            return None
        d = np.asarray(z[key], dtype=np.float64)
    rows = records["source"]["shape"][0]
    if d.size == 2 * rows:
        return d[:rows] if records["source"]["role"] == "gate" else d[rows:]
    return d


def run_arm(adapter: Any, repo: Path, unit: Any, source: Any, H: Any, count: int,
            params: dict[str, Any], arm_id: str, X_score: Any,
            output_diag: np.ndarray | None, fisher: np.ndarray | None,
            payload_dir: Path | None) -> dict[str, Any]:
    global _TRACE
    _TRACE = {}
    action, cb = make_action(repo, unit, source, H, count, params, arm_id)
    payload = adapter.encode(source, H, action, callbacks=cb, device="cuda:0")
    recon = adapter.decode(payload)
    local = metrics(source.to(recon.device), recon, X_score, output_diag, fisher)
    pm = payload_manifest(payload)
    action = dataclasses.replace(
        action,
        evidence={"local_metrics": local, "promoted_kld": None},
        serialized={k: v for k, v in pm.items() if k != "payload_sha256"},
        hashes={"payload_sha256": pm["payload_sha256"]},
    )
    # Validate the complete action after binding scientific evidence and payload.
    action.validate()
    row = {
        "arm_id": arm_id,
        "parameters": params,
        "sample_count": count,
        "action": action.to_dict(),
        "action_identity_sha256": action.identity_sha256(),
        "payload": pm,
        "proxy_error": payload.proxy_error,
        "encode_seconds": payload.encode_seconds,
        "reconstruction_sha256": tensor_sha256(recon),
        "local_metrics": local,
        "innovation_oracle": dict(_TRACE),
        "stock_semantics": {
            "block_ldl_b": 16,
            "recurrence_executor": "pinned stock reverse LDLQ",
            "viterbi_executor": "pinned stock quantize_tiles_multigpu",
            "callback_returns_candidate_tiles_only": True,
            "post_hoc_correction": False,
        },
    }
    if payload_dir is not None:
        payload_dir.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(payload.tensors, payload_dir / f"{arm_id}.pt")
    return row


def broad_screen(adapter: Any, repo: Path, r29: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Actual-stock eight-block screen across all nine preregistered depths."""
    import torch
    results = []
    for layer in args.screen_layers:
        records = find_records(repo, r29, layer, args.role)
        source, _ = load_source(records)
        X = load_real_inputs(r29, records)
        output_diag = extract_output_diag(r29, records)
        fisher = None
        if records["fisher"] is not None:
            fisher = npy_float(r29 / "fisher-selected-bf16" / records["fisher"]["fisher_file"])
        layer_rows = []
        for block_spec in records["source"]["screening_blocks"][:args.screen_blocks]:
            r0, c0 = block_spec["row"], block_spec["col"]
            rows, cols = block_spec["rows"], block_spec["cols"]
            Wb = source[r0:r0 + rows, c0:c0 + cols].contiguous()
            if tensor_sha256(Wb.to(torch.bfloat16)) != block_spec["sha256"]:
                raise RuntimeError(f"R29 screening block hash mismatch at layer {layer} {block_spec['kind']}")
            Xb = X[:, c0:c0 + cols].contiguous()
            Hb = (Xb.T @ Xb).float().contiguous()
            lw = diagonal_target_ledoit_wolf_rho(Xb)
            odb = None if output_diag is None else output_diag[r0:r0 + rows]
            fb = None if fisher is None else fisher[r0:r0 + rows, c0:c0 + cols]
            unit = r30.Unit(
                unit_id=f"qwen38.layer{layer}.mlp.{args.role}_proj.screen.{block_spec['kind']}",
                granularity="shard", topology="mlp", role=f"{args.role}_proj",
                tensor_keys=(records["source"]["tensor_name"],), layer_index=layer,
                shard_id=f"rows-{r0}-{r0+rows}.cols-{c0}-{c0+cols}",
            )
            specs = [("stock", arm_parameters("stock", None, lw["rho"]))]
            specs += [(f"rho-{rho:g}", arm_parameters("rho", rho, lw["rho"])) for rho in RHOS if rho != 1.0]
            specs += [(f"band-{band}", arm_parameters("band", band, lw["rho"])) for band in BANDS]
            specs += [("ledoit-wolf", arm_parameters("ledoit_wolf", None, lw["rho"]))]
            for label, params in specs:
                row = run_arm(
                    adapter, repo, unit, Wb, Hb, Xb.shape[0], params,
                    f"r33-screen-l{layer}-{args.role}-{block_spec['kind']}-{label}",
                    Xb, odb, fb, None,
                )
                layer_rows.append({
                    "arm_id": row["arm_id"],
                    "block": block_spec,
                    "parameters": params,
                    "action_identity_sha256": row["action_identity_sha256"],
                    "payload_sha256": row["payload"]["payload_sha256"],
                    "buffer_bytes": row["payload"]["buffer_bytes"],
                    "local_metrics": row["local_metrics"],
                    "innovation_oracle": row["innovation_oracle"],
                    "proxy_error": row["proxy_error"],
                })
        results.append({
            "layer": layer,
            "role": args.role,
            "blocks": len(records["source"]["screening_blocks"][:args.screen_blocks]),
            "actual_stock_rows": layer_rows,
            "scope": "nonpromotable 128x128 source-basis block screen",
        })
        del source, X, fisher
        torch.cuda.empty_cache()
    return results


def experiment(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    repo, r29 = args.repo.resolve(), args.r29.resolve()
    foundation = verify_foundation(repo, r29)
    adapter = r30.StockEXL3()
    screen_rows = broad_screen(adapter, repo, r29, args) if args.broad_screen else []
    all_units = []
    for layer in args.layers:
        records = find_records(repo, r29, layer, args.role)
        source, source_bf16_sha = load_source(records)
        X = load_real_inputs(r29, records)
        lw = diagonal_target_ledoit_wolf_rho(X)
        # R30's pinned stock finalize_capture_H divides the supplied accumulator by
        # count. Feed the real X^T X sum, not the already normalized R29 covariance.
        H = (X.T @ X).float().contiguous()
        hx_file = r29 / "dense-hx-bf16" / records["hx"]["file"]
        hx_published = torch.from_numpy(npy_float(hx_file)).float()
        dense_crosscheck = {
            "published_sha256": records["hx"]["sha256"],
            "published_file_sha256": sha256_file(hx_file),
            "max_abs_vs_rebuilt": float((H / X.shape[0] - hx_published).abs().max().item()),
            "rebuilt_from_real_activation": True,
        }
        output_diag = extract_output_diag(r29, records)
        fisher = None
        if records["fisher"] is not None:
            fisher = npy_float(r29 / "fisher-selected-bf16" / records["fisher"]["fisher_file"])
        unit = r30.Unit(
            unit_id=f"qwen38.layer{layer}.mlp.{args.role}_proj", granularity="tensor",
            topology="mlp", role=f"{args.role}_proj",
            tensor_keys=(records["source"]["tensor_name"],), layer_index=layer,
        )
        arms: list[tuple[str, dict[str, Any], Any, int]] = []
        arms.append(("stock-full", arm_parameters("stock", None, lw["rho"]), H, X.shape[0]))
        for rho in RHOS:
            if rho != 1.0:  # rho=1 is exactly stock and is represented once.
                arms.append((f"rho-{rho:g}", arm_parameters("rho", rho, lw["rho"]), H, X.shape[0]))
        for band in BANDS:
            arms.append((f"band-{band}", arm_parameters("band", band, lw["rho"]), H, X.shape[0]))
        arms.append(("ledoit-wolf", arm_parameters("ledoit_wolf", None, lw["rho"]), H, X.shape[0]))
        rows = []
        for name, params, H_arm, n in arms:
            rows.append(run_arm(adapter, repo, unit, source, H_arm, n, params,
                                f"r33-l{layer}-{args.role}-{name}", X, output_diag, fisher, None))
            torch.cuda.empty_cache()
        # Calibration-size robustness: stock and the locally best rho are rerun.
        best_rho = min((r for r in rows if r["parameters"]["curvature_mode"] == "rho"),
                       key=lambda r: r["local_metrics"]["real_input_hwe"], default=rows[0])
        sample_rows = []
        for n in SAMPLE_COUNTS[:-1]:
            Xn = X[:n]
            Hn = (Xn.T @ Xn).float().contiguous()
            for label, params in (("stock", arm_parameters("stock", None, lw["rho"])),
                                  ("best-rho", dict(best_rho["parameters"]))):
                sample_rows.append(run_arm(adapter, repo, unit, source, Hn, n, params,
                    f"r33-l{layer}-{args.role}-n{n}-{label}", X, output_diag, fisher, None))
                torch.cuda.empty_cache()
        if args.payload_dir is not None:
            # Retain only stock and the locally shortlisted rho payload. All other
            # exact buffer hashes stay in the receipt, avoiding ~1 GiB/layer of
            # scientifically unnecessary duplicate payload files.
            for selected in (rows[0], best_rho):
                rerun = run_arm(
                    adapter, repo, unit, source, H, X.shape[0],
                    dict(selected["parameters"]), selected["arm_id"],
                    X, output_diag, fisher, args.payload_dir,
                )
                rerun["saved_payload_file"] = str(args.payload_dir / f'{selected["arm_id"]}.pt')
                rows[rows.index(selected)] = rerun
                torch.cuda.empty_cache()
        all_units.append({
            "unit": dataclasses.asdict(unit),
            "source_bf16_sha256": source_bf16_sha,
            "source_float32_sha256": tensor_sha256(source),
            "source_shape": list(source.shape),
            "real_activation_sha256": records["activation"]["sha256"],
            "dense_hx": dense_crosscheck,
            "ledoit_wolf": lw,
            "output_fisher_diagnostic": output_and_fisher_diagnostics(r29, records),
            "arms": rows,
            "sample_count_sweep": sample_rows,
        })
    receipt = {
        "schema": SCHEMA,
        "status": "measured-local-awaiting-validation-kld",
        "stage": "phase-b/calibration-and-validation-selection",
        "claim_scope": "actual stock EXL3 codec-exact local screen; no KLD claim without R31 replay",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_sha256": sha256_file(Path(__file__)),
        "foundation": {
            "verified_files": foundation,
            "r29_data_content_sha256": R29_DATA_CONTENT_SHA256,
            "r29_split_content_sha256": R29_SPLIT_CONTENT_SHA256,
            "r29_fisher_content_sha256": R29_FISHER_CONTENT_SHA256,
        },
        "encoder_runtime": adapter.source_identity,
        "fixed_contract": {"K": 5, "codebook": "mcg", "seed": SEED,
                           "scale_search": "stock_g_scale_gss", "bytes": "fixed per source shape/K",
                           "decoder_hot_path_changed": False, "post_hoc_correction": False},
        "broad_screen": screen_rows,
        "units": all_units,
        "selection": {
            "rule": "validation full-vocabulary KLD selects; local metrics only shortlist",
            "selected_action": None,
            "new_shrinkage_action": False,
            "reason": "R31 validation KLD not yet attached",
            "falsifier": "endpoint selection or absent interior validation KLD gain yields no action",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(receipt) + b"\n")
    return receipt


def parse_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, header


def patch_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Create a hardlinked checkpoint copy with one privately copied changed shard."""
    import torch
    source = args.checkpoint.resolve()
    candidate = args.candidate.resolve()
    if candidate.exists():
        raise RuntimeError("candidate path already exists")
    payload = torch.load(args.payload, map_location="cpu", weights_only=True)
    prefix = args.tensor_key.rsplit(".weight", 1)[0]
    index = load_json(source / "model.safetensors.index.json")["weight_map"]
    shard_to_updates: dict[str, dict[str, Any]] = {}
    for suffix, tensor in payload.items():
        key = f"{prefix}.{suffix}"
        if key not in index:
            raise RuntimeError(f"checkpoint lacks {key}")
        shard_to_updates.setdefault(index[key], {})[key] = tensor
    candidate.mkdir(parents=True)
    changed_shards = set(shard_to_updates)
    for src in source.rglob("*"):
        rel = src.relative_to(source)
        dst = candidate / rel
        if src.is_dir():
            dst.mkdir(exist_ok=True)
        elif src.name in changed_shards:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
    patched = []
    for shard_name, updates in shard_to_updates.items():
        p = candidate / shard_name
        n, header = parse_safetensors_header(p)
        data_start = 8 + n
        with p.open("r+b") as f:
            for key, tensor in updates.items():
                meta = header[key]
                lo, hi = meta["data_offsets"]
                raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                if len(raw) != hi - lo:
                    raise RuntimeError(f"same-byte replacement violated for {key}: {len(raw)} != {hi-lo}")
                f.seek(data_start + lo)
                f.write(raw)
                patched.append({
                    "key": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    "shard": shard_name, "offset": [data_start + lo, data_start + hi],
                })
    receipt = {
        "schema": "qwen38-wave5-one-tensor-replacement/1",
        "source_checkpoint": str(source), "candidate_checkpoint": str(candidate),
        "tensor_key": args.tensor_key, "payload_file_sha256": sha256_file(args.payload),
        "patched": patched,
        "candidate_shards": {p.name: sha256_file(p) for p in candidate.glob("*.safetensors")},
        "same_byte_replacement": True,
        "non_target_files_hardlinked": True,
        "changed_shards_private_copies": sorted(changed_shards),
    }
    (candidate / "r33-replacement-receipt.json").write_bytes(canonical_json(receipt) + b"\n")
    return receipt


def self_test() -> None:
    import torch
    X = torch.tensor([[1., 2., 3.], [2., -1., 0.], [-1., 1., 2.], [3., 0., -2.]])
    lw = diagonal_target_ledoit_wolf_rho(X)
    assert 0.0 <= lw["rho"] <= 1.0
    class C: pass
    c = C(); c.callback = C(); c.callback.parameters = {"curvature_mode": "rho", "rho": 0.0}
    H = X.T @ X / X.shape[0]
    H0 = curvature_variant(H, c)
    assert torch.equal(H0, torch.diag(H.diagonal()))
    c.callback.parameters = {"curvature_mode": "rho", "rho": 1.0}
    assert torch.equal(curvature_variant(H, c), H)
    L = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    c.callback.parameters = {"band_features": 16}
    b16 = recurrence_band(L, c)
    assert torch.equal(b16[16:32, :16], torch.zeros(16, 16))
    assert torch.equal(b16[16:32, 16:32], L[16:32, 16:32])
    c.callback.parameters = {"band_features": 32}
    b32 = recurrence_band(L, c)
    assert torch.equal(b32[32:48, :16], torch.zeros(16, 16))
    assert torch.equal(b32[32:48, 16:32], L[32:48, 16:32])
    assert target_identity(H, c) is H
    assert legal_path_trace(H, H.clone(), None, c) is H
    assert _TRACE["post_recurrence_samples"][-1]["normalized_mse"] == 0.0
    print("r33_curvature_shrink self-test: pass")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-test")
    r = sub.add_parser("run")
    r.add_argument("--repo", type=Path, required=True)
    r.add_argument("--broad-screen", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--screen-layers", type=int, nargs="+", default=[0, 7, 14, 21, 28, 35, 42, 49, 55])
    r.add_argument("--screen-blocks", type=int, default=8)
    r.add_argument("--r29", type=Path, default=Path("/tmp/qwen38-wave5-r29"))
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--payload-dir", type=Path)
    r.add_argument("--layers", type=int, nargs="+", default=[0, 55])
    r.add_argument("--role", choices=["gate", "up"], default="gate")
    c = sub.add_parser("patch-checkpoint")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--payload", type=Path, required=True)
    c.add_argument("--tensor-key", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "run":
        receipt = experiment(args)
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))
    else:
        print(json.dumps(patch_checkpoint(args), sort_keys=True))


if __name__ == "__main__":
    main()
