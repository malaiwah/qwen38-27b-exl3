#!/usr/bin/env python3
"""R34 candidate-conditioned down targets and legal Qwen boundary transforms.

The mainline changes only the stock EXL3 encoder target. Continuous fitted targets
are process-local and are never serialized by this program. The only persistent
candidate is the payload returned by the pinned R30 stock encoder.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np

DATA_MANIFEST_FILE_SHA256 = "68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37"
DATA_MANIFEST_CONTENT_SHA256 = "51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2"
SPLIT_MANIFEST_FILE_SHA256 = "a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc"
SPLIT_MANIFEST_CONTENT_SHA256 = "151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e"
FISHER_MANIFEST_FILE_SHA256 = "4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a"
FISHER_MANIFEST_CONTENT_SHA256 = "28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237"
R30_HARNESS_SHA256 = "d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d"
R30_SCHEMA_SHA256 = "275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8"
R30_EXTENSION_BINARY_SHA256 = "e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e"
R31_GATE_SHA256 = "f4fc0594e5f77e43e42e039f76f65364b25f2a73367d3342778761417dc65e76"
R31_CONTRACT_SHA256 = "e8e1d47f8fbef627219850542e66f9ff195e0295eb4aaf54eb850b67fbd14afc"
R31_PREREG_SHA256 = "75a81665c75761767a7c71d58f4d59c446a13d3d7b164c5a8b9da9070388a784"
SCHEMA = "qwen38-wave5-r34/1"
HADAMARD = 128
SEAM_BLOCK = 16


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return sha256_bytes(canonical_bytes(body))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def reconstructed_hidden(x: np.ndarray, gate_up_weight: np.ndarray) -> np.ndarray:
    """Real Qwen dense SwiGLU: first half gate, second half up."""
    if gate_up_weight.ndim != 2 or gate_up_weight.shape[0] % 2:
        raise ValueError("gate/up weight must be [2*intermediate, hidden]")
    if x.ndim != 2 or x.shape[1] != gate_up_weight.shape[1]:
        raise ValueError("activation and gate/up input dimensions disagree")
    half = gate_up_weight.shape[0] // 2
    projected = x.astype(np.float64) @ gate_up_weight.astype(np.float64).T
    return silu(projected[:, :half]) * projected[:, half:]


def block_output(hidden: np.ndarray, down_weight: np.ndarray) -> np.ndarray:
    if hidden.ndim != 2 or down_weight.ndim != 2 or hidden.shape[1] != down_weight.shape[1]:
        raise ValueError("down projection dimensions disagree")
    return hidden.astype(np.float64) @ down_weight.astype(np.float64).T


@dataclasses.dataclass(frozen=True)
class TargetFit:
    target: np.ndarray
    relative_ridge: float
    absolute_ridge: float
    rank: int
    residual_mse_before: float
    residual_mse_continuous: float
    target_sha256: str


def array_sha256(x: np.ndarray) -> str:
    """Match R30's digest of contiguous tensor storage bytes."""
    y = np.ascontiguousarray(x)
    return sha256_bytes(y.view(np.uint8).tobytes())


def fit_down_target(
    hidden: np.ndarray,
    teacher_output: np.ndarray,
    incumbent_down: np.ndarray,
    relative_ridge: float,
) -> TargetFit:
    """Fit W0 + argmin_D ||H D^T - (Y-H W0^T)||^2 + lambda ||D||^2.

    The dual N-by-N solve is exact for ridge and minimum-norm for lambda=0. This
    function must be called separately for every decoded upstream action and K.
    """
    if relative_ridge < 0 or not math.isfinite(relative_ridge):
        raise ValueError("relative ridge must be finite and nonnegative")
    h = np.asarray(hidden, dtype=np.float64)
    y = np.asarray(teacher_output, dtype=np.float64)
    w0 = np.asarray(incumbent_down, dtype=np.float64)
    if h.ndim != 2 or y.ndim != 2 or w0.ndim != 2:
        raise ValueError("target fit expects matrices")
    if h.shape[0] != y.shape[0] or h.shape[1] != w0.shape[1] or y.shape[1] != w0.shape[0]:
        raise ValueError("target fit matrix dimensions disagree")
    residual = y - h @ w0.T
    gram = h @ h.T
    scale = float(np.trace(gram) / max(1, gram.shape[0]))
    lam = relative_ridge * scale
    if lam == 0.0:
        dual = np.linalg.pinv(gram, rcond=1e-10) @ residual
    else:
        dual = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), residual)
    delta_t = h.T @ dual
    target = np.ascontiguousarray((w0.T + delta_t).T, dtype=np.float32)
    after = y - h @ target.astype(np.float64).T
    return TargetFit(
        target=target,
        relative_ridge=float(relative_ridge),
        absolute_ridge=float(lam),
        rank=int(np.linalg.matrix_rank(h)),
        residual_mse_before=float(np.mean(residual * residual)),
        residual_mse_continuous=float(np.mean(after * after)),
        target_sha256=array_sha256(np.ascontiguousarray(target.T)),
    )


def candidate_covariance(hidden: np.ndarray) -> np.ndarray:
    h = np.asarray(hidden, dtype=np.float32)
    if h.ndim != 2 or not np.isfinite(h).all():
        raise ValueError("candidate hidden must be a finite matrix")
    return np.ascontiguousarray((h.T @ h) / np.float32(h.shape[0]), dtype=np.float32)


def seam_reroll_target(
    continuous_target: np.ndarray,
    first_decode: np.ndarray,
    *,
    strength: float = 1.0,
    block: int = SEAM_BLOCK,
    width: int = 2,
) -> np.ndarray:
    """ICBQ-style second-pass target restricted to adjacent BlockLDLQ seams.

    Only the last/first ``width`` coordinates around every input block boundary
    receive the first-pass residual. Stock Viterbi still selects every legal path.
    """
    if not (0 <= strength <= 1) or block <= 0 or width <= 0 or width * 2 > block:
        raise ValueError("invalid seam reroll parameters")
    target = np.asarray(continuous_target, dtype=np.float32)
    decoded = np.asarray(first_decode, dtype=np.float32)
    if target.shape != decoded.shape or target.ndim != 2:
        raise ValueError("target/decode seam shapes disagree")
    mask = np.zeros(target.shape[1], dtype=np.float32)
    for seam in range(block, target.shape[1], block):
        mask[max(0, seam - width):min(target.shape[1], seam + width)] = 1.0
    return np.ascontiguousarray(target + np.float32(strength) * (target - decoded) * mask[None, :])


def factorial_plan() -> list[dict[str, Any]]:
    rows = []
    for target in ("source", "reconstructed-upstream"):
        for revisit in ("matched-control-second-encode", "icbq-seam-reroll"):
            rows.append({
                "target": target,
                "revisit": revisit,
                "stock_encoder_calls": 2,
                "legal_viterbi_budget": "identical stock scale-search and final Viterbi call graph per encode",
                "persistent_artifact": "second stock EXL3 payload only",
            })
    return rows


def fwht128(x: np.ndarray) -> np.ndarray:
    """Normalized block-H128 on the final dimension."""
    a = np.asarray(x, dtype=np.float64).copy()
    if a.shape[-1] % HADAMARD:
        raise ValueError("H128 requires a multiple-of-128 final dimension")
    blocks = a.reshape(*a.shape[:-1], -1, HADAMARD)
    h = 1
    while h < HADAMARD:
        old = blocks.copy()
        for start in range(0, HADAMARD, 2 * h):
            blocks[..., start:start + h] = old[..., start:start + h] + old[..., start + h:start + 2 * h]
            blocks[..., start + h:start + 2 * h] = old[..., start:start + h] - old[..., start + h:start + 2 * h]
        h *= 2
    blocks /= math.sqrt(HADAMARD)
    return blocks.reshape(a.shape)


def prove_swiglu_boundary_identity(gate: np.ndarray, up: np.ndarray) -> dict[str, Any]:
    """Prove U_A-only and U_B-only with U_R=I; both pay one inverse H128."""
    gate = np.asarray(gate, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    if gate.shape != up.shape:
        raise ValueError("gate/up shapes disagree")
    reference = silu(gate) * up
    recovered_a = fwht128(fwht128(gate))
    recovered_b = fwht128(fwht128(up))
    arms = {"U_A_only": silu(recovered_a) * up, "U_B_only": silu(gate) * recovered_b}
    return {
        name: {
            "u_r": "identity",
            "float64_max_abs_identity_error": float(np.max(np.abs(value - reference))),
            "runtime_boundary_ops": [f"inverse block-H128 on {'gate preactivation' if name == 'U_A_only' else 'up preactivation'}"],
            "zero_payload": True,
            "zero_runtime": False,
        }
        for name, value in arms.items()
    }


def prove_gated_attention_monomial(
    value: np.ndarray,
    gate: np.ndarray,
    o_weight: np.ndarray,
    permutation: np.ndarray,
    diagonal: np.ndarray,
) -> dict[str, Any]:
    """Exact gated-attention V/gate/O action with no gate sign/scale."""
    v = np.asarray(value, dtype=np.float64)
    g = np.asarray(gate, dtype=np.float64)
    o = np.asarray(o_weight, dtype=np.float64)
    p = np.asarray(permutation, dtype=np.int64)
    d = np.asarray(diagonal, dtype=np.float64)
    if v.shape != g.shape or v.ndim != 3 or p.shape != (v.shape[-1],) or d.shape != p.shape:
        raise ValueError("gated attention monomial shapes disagree")
    if sorted(p.tolist()) != list(range(len(p))) or np.any(d == 0):
        raise ValueError("monomial requires a permutation and nonzero diagonal")
    reference_z = v * (1.0 / (1.0 + np.exp(-np.clip(g, -40, 40))))
    reference = reference_z.reshape(v.shape[0], -1) @ o.T
    vp = v[..., p] * d
    gp = g[..., p]
    transformed_z = vp * (1.0 / (1.0 + np.exp(-np.clip(gp, -40, 40))))
    heads = v.shape[1]
    flat_p = np.concatenate([head * len(p) + p for head in range(heads)])
    flat_d = np.tile(d, heads)
    op = o[:, flat_p] / flat_d[None, :]
    transformed = transformed_z.reshape(v.shape[0], -1) @ op.T
    return {
        "float64_max_abs_identity_error": float(np.max(np.abs(reference - transformed))),
        "legal_family": "V monomial; gate coordinate permutation; inverse O monomial",
        "sigmoid_guard": "gate sign/scale forbidden",
        "runtime_hot_ops": [],
        "startup_fusion": "offline row/column rewrite only",
    }


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _partial_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, rotary_dim: int) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    half = rotary_dim // 2
    a = x[..., :half]
    b = x[..., half:rotary_dim]
    y[..., :half] = a * cos - b * sin
    y[..., half:rotary_dim] = b * cos + a * sin
    return y


def prove_qk_norm_rope_safe(
    q_pre: np.ndarray,
    k_pre: np.ndarray,
    q_norm_weight: np.ndarray,
    k_norm_weight: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    signs: np.ndarray,
    nonrot_permutation: np.ndarray,
    rotary_dim: int = 64,
) -> dict[str, Any]:
    """Prove shared paired rotary signs plus non-rotary coordinate permutations."""
    q = np.asarray(q_pre, dtype=np.float64)
    k = np.asarray(k_pre, dtype=np.float64)
    qw = np.asarray(q_norm_weight, dtype=np.float64)
    kw = np.asarray(k_norm_weight, dtype=np.float64)
    s = np.asarray(signs, dtype=np.float64)
    pnr = np.asarray(nonrot_permutation, dtype=np.int64)
    dim = q.shape[-1]
    if q.shape != k.shape or dim != len(qw) or dim != len(kw) or len(s) != dim:
        raise ValueError("Q/K norm/RoPE shapes disagree")
    half = rotary_dim // 2
    if not np.array_equal(s[:half], s[half:rotary_dim]):
        raise ValueError("rotary signs must repeat across paired RoPE halves")
    if sorted(pnr.tolist()) != list(range(dim - rotary_dim)):
        raise ValueError("invalid non-rotary permutation")
    perm = np.concatenate([np.arange(rotary_dim), rotary_dim + pnr])
    q_ref = _partial_rope(_rms_norm(q, qw), cos, sin, rotary_dim)
    k_ref = _partial_rope(_rms_norm(k, kw), cos, sin, rotary_dim)
    q_new = _partial_rope(_rms_norm(q[..., perm] * s, qw[perm]), cos, sin, rotary_dim)
    k_new = _partial_rope(_rms_norm(k[..., perm] * s, kw[perm]), cos, sin, rotary_dim)
    return {
        "q_float64_max_abs_identity_error": float(np.max(np.abs(q_new - q_ref[..., perm] * s))),
        "k_float64_max_abs_identity_error": float(np.max(np.abs(k_new - k_ref[..., perm] * s))),
        "legal_family": "shared paired rotary signs plus shared signed permutation on non-rotary coordinates; norm weights co-permuted",
        "forbidden": "dense rotations, rotary-pair permutations without a RoPE-cache/kernel contract, unmatched GQA head permutations",
        "runtime_hot_ops": [],
    }


GDN_REQUIRED_TRACE_FIELDS = {
    "q_pre_norm", "k_pre_norm", "v", "z", "conv_input", "conv_output",
    "state_before", "state_after", "rmsnorm_gated_input", "out_input", "out_output",
}


def audit_gdn_trace_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(manifest.get("trace_fields", []))
    missing = sorted(GDN_REQUIRED_TRACE_FIELDS - fields)
    if missing:
        return {"eligible": False, "reason": "topology-complete internal GDN state trace unavailable",
                "missing_trace_fields": missing, "simplified_recurrence_inference_forbidden": True}
    if manifest.get("actual_kernel_trace") is not True or manifest.get("topology") != "qwen3.8-gated-deltanet":
        return {"eligible": False, "reason": "trace does not prove the actual Qwen GDN kernel/topology",
                "missing_trace_fields": [], "simplified_recurrence_inference_forbidden": True}
    return {"eligible": True, "reason": "actual topology-complete state trace present",
            "missing_trace_fields": [], "simplified_recurrence_inference_forbidden": True}


def evaluate_candidate(hidden: np.ndarray, teacher_output: np.ndarray, decoded_down: np.ndarray) -> dict[str, float]:
    error = block_output(hidden, decoded_down) - np.asarray(teacher_output, dtype=np.float64)
    per_row = np.mean(error * error, axis=1)
    return {"block_output_mse": float(np.mean(per_row)),
            "block_output_p99_row_mse": float(np.quantile(per_row, 0.99)),
            "block_output_max_row_mse": float(np.max(per_row))}


def candidate_target_parameters(
    fit: TargetFit,
    *,
    fit_manifest_sha256: str,
    teacher_output_sha256: str,
    upstream_action_identity_sha256: str,
    recompute_scope: str,
) -> dict[str, Any]:
    """Exact additive R30 target object for make_stock_action(target_parameters=...)."""
    return {
        "basis": "candidate_conditioned",
        "target_tensor_sha256": fit.target_sha256,
        "fit_manifest_sha256": fit_manifest_sha256,
        "teacher_output_sha256": teacher_output_sha256,
        "upstream_action_identity_sha256": upstream_action_identity_sha256,
        "target_dtype": "float32",
        "target_shape": list(reversed(fit.target.shape)),
        "ridge_lambda": fit.absolute_ridge,
        "recompute_scope": recompute_scope,
    }


_ACTIVE_ENCODER_TARGET: Any | None = None


def r34_target_callback(weight: Any, action: Any) -> Any:
    """Closure-free R30 callback; target identity lives in the complete action."""
    if _ACTIVE_ENCODER_TARGET is None:
        raise RuntimeError("candidate target is not active")
    target = _ACTIVE_ENCODER_TARGET.to(device=weight.device, dtype=weight.dtype)
    if target.shape != weight.shape:
        raise RuntimeError("candidate target shape does not match encoder basis")
    return target


def _torch_raw_sha256(tensor: Any) -> str:
    import torch
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def _load_capture(path: pathlib.Path) -> tuple[Any, Any]:
    from safetensors.torch import load_file
    tensors = load_file(str(path))
    return tensors["input"].float().contiguous(), tensors["output"].float().contiguous()


def _split_contract(exl3: Any) -> tuple[dict[str, Any], Any]:
    selections = {
        "calibration": {"selection_sha256": "490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259",
                        "selector": {"field": "split", "op": "eq", "value": "calibration"}},
        "validation": {"selection_sha256": "4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de",
                       "selector": {"field": "split", "op": "eq", "value": "validation"}},
        "untouched_test": {"selection_sha256": "4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca",
                           "selector": {"field": "split", "op": "eq", "value": "untouched_test"}},
    }
    disjoint = exl3.SplitDisjointness(
        artifact_sha256="7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9",
        predicate_language="wave5.split-predicate/1",
        pairwise_overlap_counts={"calibration__validation": 0, "calibration__untouched_test": 0,
                                 "validation__untouched_test": 0},
        source_document_overlap_count=0, domain_leakage_count=0, verified=True,
    )
    return selections, disjoint


def run_actual_exl3_factorial(args: argparse.Namespace) -> dict[str, Any]:
    """Run one actual-stock gate/up action and the matched down-target factorial."""
    import sys
    import torch
    from safetensors import safe_open

    tool_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(tool_dir))
    import exl3_action as exl3
    if sha256_file(tool_dir / "exl3_action.py") != R30_HARNESS_SHA256:
        raise RuntimeError("R30 harness pin mismatch")
    if sha256_file(tool_dir / "exl3_action_schema.json") != R30_SCHEMA_SHA256:
        raise RuntimeError("R30 schema pin mismatch")

    def load_weight(path: pathlib.Path, key: str) -> Any:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).float().contiguous()

    gate = load_weight(args.gate_shard, args.gate_key)
    up = load_weight(args.up_shard, args.up_key)
    source_gate_up = torch.cat((gate, up), dim=0).contiguous()
    source_down = load_weight(args.down_shard, args.down_key)
    cal_x, _ = _load_capture(args.calibration_gate_up)
    _, cal_y = _load_capture(args.calibration_down)
    val_x, _ = _load_capture(args.validation_gate_up)
    _, val_y = _load_capture(args.validation_down)
    selections, disjoint = _split_contract(exl3)
    gate_unit = exl3.Unit(
        unit_id=f"layer{args.layer}.mlp.gate_up", granularity="module", topology="mlp",
        role="other_dense", tensor_keys=(args.gate_key, args.up_key), layer_index=args.layer,
        fused_group=f"layer{args.layer}.mlp", output_splits=(gate.shape[0], up.shape[0]),
    )
    down_unit = exl3.Unit(
        unit_id=f"layer{args.layer}.mlp.down", granularity="module", topology="mlp",
        role="down_proj", tensor_keys=(args.down_key,), layer_index=args.layer,
        fused_group=f"layer{args.layer}.mlp",
    )
    h_gate = (cal_x.T @ cal_x / cal_x.shape[0]).float().contiguous()
    gate_curvature = exl3.make_curvature_capture(
        h_gate, gate_unit, capture_id=f"r34-l{args.layer}-gateup-input",
        observation_count=cal_x.shape[0], normalization="X^T X/N",
        basis="real R29 BF16 reference input",
        coordinate_convention=exl3.COORDINATE_CONVENTIONS["out_in"],
    )
    common = {
        "split_manifest_sha256": SPLIT_MANIFEST_FILE_SHA256,
        "split_manifest_content_sha256": SPLIT_MANIFEST_CONTENT_SHA256,
        "split_selections": selections, "split_disjointness": disjoint,
        "evidence": {"local_metrics": {}, "promoted_kld": None},
        "source_revision": args.source_revision, "source_layout": "out_in",
        "route_id": "codec-exact/all-trellis-stock-exl3",
    }
    gate_action = exl3.make_stock_action(
        action_id=f"r34-l{args.layer}-gateup-k{args.gate_k}-stock",
        unit=gate_unit, K=args.gate_k, codebook=args.codebook, seed=args.seed,
        source_sha256=_torch_raw_sha256(source_gate_up), curvature=gate_curvature,
        callbacks=None, **common,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    codec = exl3.StockEXL3()
    gate_payload = codec.encode(source_gate_up, h_gate, gate_action, device=args.device)
    decoded_gate_up = codec.decode(gate_payload).float().cpu().contiguous()
    gate_manifest = gate_payload.serialize(args.output / "gate-up.safetensors")
    gate_payload_sha = exl3.payload_digest(gate_payload.tensors)
    identity_callbacks = exl3.EncodeCallbacks.identity()
    identity_action = exl3.make_stock_action(
        action_id=f"r34-l{args.layer}-gateup-k{args.gate_k}-strength-zero",
        unit=gate_unit, K=args.gate_k, codebook=args.codebook, seed=args.seed,
        source_sha256=_torch_raw_sha256(source_gate_up), curvature=gate_curvature,
        callbacks=identity_callbacks, **common,
    )
    identity_payload = codec.encode(
        source_gate_up, h_gate, identity_action,
        callbacks=identity_callbacks, device=args.device,
    )
    identity_decode = codec.decode(identity_payload).float().cpu().contiguous()
    operational_identity = {
        "extension_binary_sha256": codec.source_identity["extension_binary_sha256"],
        "a0_payload_sha256": gate_payload_sha,
        "strength_zero_payload_sha256": exl3.payload_digest(identity_payload.tensors),
        "byte_identical_payload": gate_payload_sha == exl3.payload_digest(identity_payload.tensors),
        "byte_identical_reconstruction": _torch_raw_sha256(decoded_gate_up) == _torch_raw_sha256(identity_decode),
        "finite_a0_decode": bool(torch.isfinite(decoded_gate_up).all().item()),
        "finite_strength_zero_decode": bool(torch.isfinite(identity_decode).all().item()),
    }
    if operational_identity != {
        **operational_identity,
        "byte_identical_payload": True,
        "byte_identical_reconstruction": True,
        "finite_a0_decode": True,
        "finite_strength_zero_decode": True,
    }:
        raise RuntimeError("operational A0/strength-zero identity gate failed")
    if operational_identity["extension_binary_sha256"] != R30_EXTENSION_BINARY_SHA256:
        raise RuntimeError("operational extension pin mismatch")
    gate_action = dataclasses.replace(
        gate_action, serialized=gate_manifest,
        hashes={"action_identity_sha256": gate_action.identity_sha256(),
                "payload_sha256": gate_payload_sha,
                "source_basis_reconstruction_sha256": _torch_raw_sha256(decoded_gate_up)},
    )
    (args.output / "gate-up.action.json").write_bytes(exl3.canonical_json(gate_action.to_dict()) + b"\n")

    h_cal = reconstructed_hidden(cal_x.numpy(), decoded_gate_up.numpy())
    h_val = reconstructed_hidden(val_x.numpy(), decoded_gate_up.numpy())
    h_down = torch.from_numpy(candidate_covariance(h_cal))
    down_curvature = exl3.make_curvature_capture(
        h_down, down_unit, capture_id=f"r34-l{args.layer}-down-{gate_action.identity_sha256()}",
        observation_count=cal_x.shape[0], normalization="H_q^T H_q/N",
        basis="candidate-specific decoded gate/up SwiGLU hidden",
        coordinate_convention=exl3.COORDINATE_CONVENTIONS["out_in"],
    )
    down_common = dict(common)
    down_common["source_sha256"] = _torch_raw_sha256(source_down)
    stock_action = exl3.make_stock_action(
        action_id=f"r34-l{args.layer}-down-k{args.down_k}-stock-target",
        unit=down_unit, K=args.down_k, codebook=args.codebook, seed=args.seed,
        curvature=down_curvature, callbacks=None, **down_common,
    )
    stock_payload = codec.encode(source_down, h_down, stock_action, device=args.device)
    incumbent = codec.decode(stock_payload).float().cpu().contiguous()
    teacher_sha = _torch_raw_sha256(cal_y)

    def encode_target(target_source: np.ndarray, label: str, scope: str,
                      fit_meta: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        global _ACTIVE_ENCODER_TARGET
        encoder_target = torch.from_numpy(np.ascontiguousarray(target_source.T)).float().contiguous()
        target_hash = _torch_raw_sha256(encoder_target)
        fit_sha = sha256_bytes(canonical_bytes(dict(fit_meta)))
        callback = exl3.EncodeCallbacks(
            target=r34_target_callback, identifier="r34/candidate-conditioned-down",
            version="1", parameters={"target_tensor_sha256": target_hash, "recompute_scope": scope},
        )
        params = {
            "basis": "candidate_conditioned", "target_tensor_sha256": target_hash,
            "fit_manifest_sha256": fit_sha, "teacher_output_sha256": teacher_sha,
            "upstream_action_identity_sha256": gate_action.identity_sha256(),
            "target_dtype": "float32", "target_shape": list(encoder_target.shape),
            "ridge_lambda": float(fit_meta["ridge_lambda"]), "recompute_scope": scope,
        }
        action = exl3.make_stock_action(
            action_id=label, unit=down_unit, K=args.down_k, codebook=args.codebook,
            seed=args.seed, curvature=down_curvature, callbacks=callback,
            target_parameters=params, **down_common,
        )
        _ACTIVE_ENCODER_TARGET = encoder_target
        try:
            payload = codec.encode(source_down, h_down, action, callbacks=callback, device=args.device)
        finally:
            _ACTIVE_ENCODER_TARGET = None
        return action, payload, codec.decode(payload).float().cpu().contiguous()

    fit_rows: dict[str, Any] = {}
    selected: TargetFit | None = None
    selected_mse = math.inf
    for ridge in (0.0, args.relative_ridge):
        fit = fit_down_target(h_cal, cal_y.numpy(), incumbent.numpy(), ridge)
        meta = {"ridge_lambda": fit.absolute_ridge, "relative_ridge": ridge,
                "target_tensor_sha256": fit.target_sha256,
                "upstream_action_identity_sha256": gate_action.identity_sha256()}
        action, payload, decoded = encode_target(
            fit.target, f"r34-l{args.layer}-ridge-select-{ridge:g}",
            "per-upstream-complete-action", meta,
        )
        val = evaluate_candidate(h_val, val_y.numpy(), decoded.numpy())
        fit_rows[f"{ridge:g}"] = {
            "action_identity_sha256": action.identity_sha256(),
            "payload_sha256": exl3.payload_digest(payload.tensors),
            "exact_buffer_bytes": payload.buffer_manifest()["buffer_bytes"], "validation": val,
        }
        if val["block_output_mse"] < selected_mse:
            selected_mse, selected = val["block_output_mse"], fit
    # Match the two ridge-selection encodes with two fresh source-target encodes.
    for _ in range(2):
        codec.encode(source_down, h_down, stock_action, device=args.device)
    if selected is None:
        raise AssertionError("ridge selector produced no fit")

    factorial_rows = []
    for target_name, target_value, ridge_lambda in (
        ("source", source_down.numpy(), 0.0),
        ("reconstructed-upstream", selected.target, selected.absolute_ridge),
    ):
        for revisit in ("matched-control-second-encode", "icbq-seam-reroll"):
            base_meta = {"ridge_lambda": ridge_lambda, "target_origin": target_name,
                         "upstream_action_identity_sha256": gate_action.identity_sha256()}
            if target_name == "source":
                first_action = stock_action
                first_payload = codec.encode(source_down, h_down, stock_action, device=args.device)
                first_decoded = codec.decode(first_payload).float().cpu().contiguous()
            else:
                first_action, first_payload, first_decoded = encode_target(
                    target_value, f"r34-l{args.layer}-{target_name}-{revisit}-pass1",
                    "per-upstream-complete-action", base_meta,
                )
            second_target = target_value
            second_scope = "per-upstream-complete-action"
            if revisit == "icbq-seam-reroll":
                second_target = seam_reroll_target(target_value, first_decoded.numpy(),
                                                   strength=args.seam_strength)
                second_scope = "matched-compute-seam-reroll"
            if target_name == "source" and revisit == "matched-control-second-encode":
                second_action = stock_action
                second_payload = codec.encode(source_down, h_down, stock_action, device=args.device)
                second_decoded = codec.decode(second_payload).float().cpu().contiguous()
            else:
                second_meta = dict(base_meta)
                second_meta.update({
                    "ridge_lambda": ridge_lambda,
                    "first_payload_sha256": exl3.payload_digest(first_payload.tensors),
                    "seam_strength": args.seam_strength if revisit == "icbq-seam-reroll" else 0.0,
                })
                second_action, second_payload, second_decoded = encode_target(
                    second_target, f"r34-l{args.layer}-{target_name}-{revisit}-pass2",
                    second_scope, second_meta,
                )
            stem = f"{target_name}-{revisit}"
            payload_path = args.output / f"{stem}.safetensors"
            serialized = second_payload.serialize(payload_path)
            payload_sha = exl3.payload_digest(second_payload.tensors)
            finalized = dataclasses.replace(
                second_action, serialized=serialized,
                hashes={"action_identity_sha256": second_action.identity_sha256(),
                        "payload_sha256": payload_sha,
                        "source_basis_reconstruction_sha256": _torch_raw_sha256(second_decoded)},
            )
            action_path = args.output / f"{stem}.action.json"
            action_path.write_bytes(exl3.canonical_json(finalized.to_dict()) + b"\n")
            factorial_rows.append({
                "target": target_name, "revisit": revisit, "ridge_lambda": ridge_lambda,
                "stock_encoder_calls": 2, "action_path": str(action_path),
                "action_identity_sha256": finalized.identity_sha256(),
                "payload_path": str(payload_path), "payload_sha256": payload_sha,
                "exact_buffer_bytes": serialized["buffer_bytes"],
                "standalone_safetensors_bytes": serialized["standalone_safetensors_bytes"],
                "calibration": evaluate_candidate(h_cal, cal_y.numpy(), second_decoded.numpy()),
                "validation": evaluate_candidate(h_val, val_y.numpy(), second_decoded.numpy()),
            })
    stock_validation = evaluate_candidate(h_val, val_y.numpy(), incumbent.numpy())
    best = min(factorial_rows, key=lambda row: row["validation"]["block_output_mse"])
    result = {
        "schema": SCHEMA, "status": "success",
        "foundation": {
            "r29_data_manifest_file_sha256": DATA_MANIFEST_FILE_SHA256,
            "r29_data_manifest_content_sha256": DATA_MANIFEST_CONTENT_SHA256,
            "r29_split_manifest_file_sha256": SPLIT_MANIFEST_FILE_SHA256,
            "r29_split_manifest_content_sha256": SPLIT_MANIFEST_CONTENT_SHA256,
            "r30_harness_sha256": R30_HARNESS_SHA256, "r30_schema_sha256": R30_SCHEMA_SHA256,
            "operational_identity": operational_identity,
        },
        "unit": {"layer": args.layer, "topology": "mlp", "gate_up_K": args.gate_k,
                 "down_K": args.down_k, "codebook": args.codebook},
        "upstream": {"action_identity_sha256": gate_action.identity_sha256(),
                     "payload_sha256": gate_payload_sha, "exact_buffer_bytes": gate_manifest["buffer_bytes"]},
        "candidate_conditioning": {
            "recomputed_per_upstream_action_and_K": True,
            "candidate_hidden_sha256": array_sha256(h_cal.astype(np.float32)),
            "candidate_covariance_sha256": _torch_raw_sha256(h_down),
            "teacher_output_sha256": teacher_sha,
        },
        "ridge_selection": {"legal_encoder_calls": 4, "rows": fit_rows,
                            "selected_relative_ridge": selected.relative_ridge,
                            "selected_absolute_ridge": selected.absolute_ridge},
        "factorial": factorial_rows, "stock_validation": stock_validation,
        "best_measured": best,
        "same_bytes_as_stock": all(row["exact_buffer_bytes"] ==
                                   stock_payload.buffer_manifest()["buffer_bytes"] for row in factorial_rows),
        "continuous_target_persisted": False, "untouched_test_opened": False,
        "one_tensor_kld": None,
    }
    result["content_sha256"] = content_hash(result)
    return result


def self_test() -> dict[str, Any]:
    rng = np.random.default_rng(3401)
    n, hidden, intermediate, out = 23, 128, 256, 128
    x = rng.normal(size=(n, hidden)).astype(np.float32)
    wgu_a = (rng.normal(size=(2 * intermediate, hidden)) / math.sqrt(hidden)).astype(np.float32)
    wgu_b = wgu_a.copy(); wgu_b[0, 0] += np.float32(0.125)
    wdown = (rng.normal(size=(out, intermediate)) / math.sqrt(intermediate)).astype(np.float32)
    teacher_hidden = reconstructed_hidden(x, wgu_a)
    teacher = block_output(teacher_hidden, wdown)
    incumbent = wdown + rng.normal(scale=0.02, size=wdown.shape).astype(np.float32)
    h_a = reconstructed_hidden(x, wgu_a); h_b = reconstructed_hidden(x, wgu_b)
    fit_a = fit_down_target(h_a, teacher, incumbent, 1e-4)
    fit_b = fit_down_target(h_b, teacher, incumbent, 1e-4)
    if fit_a.target_sha256 == fit_b.target_sha256:
        raise AssertionError("candidate-conditioned targets were not independently rebuilt")
    if fit_a.residual_mse_continuous >= fit_a.residual_mse_before:
        raise AssertionError("ridge fit did not reduce its calibration objective")
    reroll = seam_reroll_target(fit_a.target, incumbent, strength=0.5)
    if reroll.shape != incumbent.shape or not np.isfinite(reroll).all():
        raise AssertionError("seam target invalid")
    plan = factorial_plan()
    if len(plan) != 4 or {row["stock_encoder_calls"] for row in plan} != {2}:
        raise AssertionError("factorial budgets are not matched")
    gate = rng.normal(size=(7, intermediate)); up = rng.normal(size=(7, intermediate))
    swiglu = prove_swiglu_boundary_identity(gate, up)
    if max(row["float64_max_abs_identity_error"] for row in swiglu.values()) > 1e-12:
        raise AssertionError("SwiGLU boundary transform identity failed")
    heads, dim = 4, 128
    value = rng.normal(size=(7, heads, dim)); attn_gate = rng.normal(size=value.shape)
    o = rng.normal(size=(hidden, heads * dim)); p = rng.permutation(dim)
    d = rng.choice([-1.0, 1.0], size=dim) * rng.uniform(0.5, 1.5, size=dim)
    attn = prove_gated_attention_monomial(value, attn_gate, o, p, d)
    if attn["float64_max_abs_identity_error"] > 1e-10:
        raise AssertionError("gated attention monomial identity failed")
    q = rng.normal(size=(7, dim)); k = rng.normal(size=(7, dim))
    qw = rng.uniform(0.5, 1.5, size=dim); kw = rng.uniform(0.5, 1.5, size=dim)
    theta = rng.normal(size=(7, 32)); cos = np.cos(theta); sin = np.sin(theta)
    signs = rng.choice([-1.0, 1.0], size=dim); signs[32:64] = signs[:32]
    qk = prove_qk_norm_rope_safe(q, k, qw, kw, cos, sin, signs, rng.permutation(64))
    if max(qk["q_float64_max_abs_identity_error"], qk["k_float64_max_abs_identity_error"]) > 1e-10:
        raise AssertionError("Q/K norm-RoPE identity failed")
    gdn = audit_gdn_trace_manifest({"topology": "qwen3.8-gated-deltanet", "trace_fields": []})
    if gdn["eligible"]:
        raise AssertionError("incomplete GDN trace incorrectly accepted")
    fake_hash = "1" * 64
    params = candidate_target_parameters(fit_a, fit_manifest_sha256=fake_hash,
                                         teacher_output_sha256="2" * 64,
                                         upstream_action_identity_sha256="3" * 64,
                                         recompute_scope="per-upstream-complete-action")
    if set(params) != {"basis", "target_tensor_sha256", "fit_manifest_sha256", "teacher_output_sha256",
                       "upstream_action_identity_sha256", "target_dtype", "target_shape", "ridge_lambda",
                       "recompute_scope"}:
        raise AssertionError("candidate target schema drift")
    return {
        "schema": SCHEMA, "status": "pass", "seed": 3401,
        "foundation": {"r30_harness_sha256": R30_HARNESS_SHA256, "r30_schema_sha256": R30_SCHEMA_SHA256},
        "candidate_target_independence": {"action_a_target_sha256": fit_a.target_sha256,
                                          "action_b_target_sha256": fit_b.target_sha256, "distinct": True},
        "candidate_target_parameters": params,
        "factorial": plan, "swiglu": swiglu, "gated_attention": attn,
        "qk_norm_rope": qk, "gdn_guard": gdn,
    }


def _torch_fwht128(x: Any) -> Any:
    import torch
    if x.shape[-1] % HADAMARD:
        raise ValueError("H128 requires a multiple-of-128 final dimension")
    blocks = x.float().reshape(*x.shape[:-1], -1, HADAMARD)
    h = 1
    while h < HADAMARD:
        old = blocks.clone()
        for start in range(0, HADAMARD, 2 * h):
            blocks[..., start:start + h] = old[..., start:start + h] + old[..., start + h:start + 2 * h]
            blocks[..., start + h:start + 2 * h] = old[..., start:start + h] - old[..., start + h:start + 2 * h]
        h *= 2
    return blocks.reshape(x.shape) / math.sqrt(HADAMARD)


def screen_architecture_capture(
    gate_up_path: pathlib.Path,
    down_path: pathlib.Path,
    qkv_path: pathlib.Path,
    *,
    device_name: str,
    repetitions: int,
) -> dict[str, Any]:
    """Screen exact families against real R29 L55 BF16 module boundaries."""
    import torch
    from safetensors.torch import load_file

    gu = load_file(str(gate_up_path))
    down = load_file(str(down_path))
    qkv = load_file(str(qkv_path))
    projected = gu["output"]
    actual_hidden = down["input"]
    if tuple(projected.shape) != (63, 34816) or tuple(actual_hidden.shape) != (63, 17408):
        raise ValueError("capture is not the frozen real Qwen L55 MLP boundary")
    gate, up = projected.chunk(2, dim=-1)
    reference = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    split_error = (reference.float() - actual_hidden.float()).abs()

    swiglu_rows: dict[str, Any] = {}
    for name, operand in (("U_A_only", gate), ("U_B_only", up)):
        first = _torch_fwht128(operand).to(torch.bfloat16)
        recovered = _torch_fwht128(first).to(torch.bfloat16)
        candidate = (
            torch.nn.functional.silu(recovered.float()) * up.float()
            if name == "U_A_only"
            else torch.nn.functional.silu(gate.float()) * recovered.float()
        ).to(torch.bfloat16)
        delta = candidate.float() - reference.float()
        swiglu_rows[name] = {
            "u_r": "identity",
            "bf16_bit_exact": bool(torch.equal(candidate, reference)),
            "bf16_max_abs_identity_error": float(delta.abs().max()),
            "bf16_mse_identity_error": float((delta * delta).mean()),
            "runtime_boundary_ops": 1,
            "persistent_payload_bytes": 0,
            "consider_for_promotion": False,
            "stop_reason": "BF16 identity is not bit-exact and no fused production kernel is qualified",
        }

    device = torch.device(device_name)
    timed = gate.to(device)
    for _ in range(3):
        _torch_fwht128(timed)
    if device.type == "mps":
        torch.mps.synchronize()
    start = __import__("time").perf_counter()
    for _ in range(repetitions):
        _torch_fwht128(timed)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = __import__("time").perf_counter() - start

    fused = qkv["output"]
    if tuple(fused.shape) != (63, 14336):
        raise ValueError("capture is not actual gated full attention QKV")
    q_gate, k, v = torch.split(fused, [12288, 1024, 1024], dim=-1)
    q_heads = q_gate.reshape(63, 24, 512)
    q_pre, gate_pre = q_heads.chunk(2, dim=-1)
    if q_pre.shape[-1] != 256 or gate_pre.shape[-1] != 256:
        raise AssertionError("actual interleaved Q/gate head split failed")
    gdn_guard = audit_gdn_trace_manifest({
        "topology": "qwen3.8-gated-deltanet",
        "trace_fields": ["q_pre_norm", "k_pre_norm", "v", "z", "out_input", "out_output"],
        "actual_kernel_trace": False,
    })
    result = {
        "schema": SCHEMA,
        "claim_scope": "real-activation architecture screen; not KLD or runtime qualification",
        "inputs": {
            "gate_up": {"path": str(gate_up_path), "sha256": sha256_file(gate_up_path),
                        "shape": list(projected.shape), "dtype": str(projected.dtype)},
            "down": {"path": str(down_path), "sha256": sha256_file(down_path),
                     "shape": list(actual_hidden.shape), "dtype": str(actual_hidden.dtype)},
            "qkv": {"path": str(qkv_path), "sha256": sha256_file(qkv_path),
                    "shape": list(fused.shape), "dtype": str(fused.dtype)},
        },
        "swiglu_boundary_alignment": {
            "split": "first 17408 gate / second 17408 up",
            "bf16_recomputed_vs_captured_bit_exact": bool(torch.equal(reference, actual_hidden)),
            "bf16_max_abs_error": float(split_error.max()),
        },
        "swiglu_runtime_arms": swiglu_rows,
        "unfused_h128_screen": {
            "device": str(device),
            "shape": list(gate.shape),
            "repetitions": repetitions,
            "mean_seconds": elapsed / repetitions,
            "fused_production_cost_measured": False,
            "qualification": "not eligible until a fused RTX5090 kernel is measured",
        },
        "full_attention_topology": {
            "actual_fused_split": {"q_gate": list(q_gate.shape), "k": list(k.shape), "v": list(v.shape)},
            "q_gate_layout": "24 heads x (Q256 then output-gate256)",
            "gqa": "24 Q/output-gate heads, 4 K/V heads, head_dim 256",
            "zero_runtime_v_gate_o": "V monomial, gate permutation only, inverse O monomial",
            "zero_runtime_qk": "paired rotary signs plus shared signed permutation on non-rotary coordinates with norm weights co-permuted",
            "dense_rotation": "runtime arm; not screened without fused cost",
        },
        "gdn": gdn_guard,
    }
    result["content_sha256"] = content_hash(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test"); test.add_argument("--receipt", type=pathlib.Path)
    fit = sub.add_parser("fit-npz", help="fit/evaluate from process-local NPZ arrays; target is not saved")
    fit.add_argument("--input", type=pathlib.Path, required=True)
    fit.add_argument("--relative-ridge", type=float, default=1e-4)
    fit.add_argument("--summary", type=pathlib.Path, required=True)
    arch = sub.add_parser("screen-architecture")
    arch.add_argument("--gate-up", type=pathlib.Path, required=True)
    arch.add_argument("--down", type=pathlib.Path, required=True)
    arch.add_argument("--qkv", type=pathlib.Path, required=True)
    arch.add_argument("--device", default="cpu")
    arch.add_argument("--repetitions", type=int, default=10)
    arch.add_argument("--summary", type=pathlib.Path, required=True)
    run = sub.add_parser("run-exl3-factorial")
    run.add_argument("--gate-shard", type=pathlib.Path, required=True)
    run.add_argument("--up-shard", type=pathlib.Path, required=True)
    run.add_argument("--down-shard", type=pathlib.Path, required=True)
    run.add_argument("--gate-key", required=True)
    run.add_argument("--up-key", required=True)
    run.add_argument("--down-key", required=True)
    run.add_argument("--calibration-gate-up", type=pathlib.Path, required=True)
    run.add_argument("--calibration-down", type=pathlib.Path, required=True)
    run.add_argument("--validation-gate-up", type=pathlib.Path, required=True)
    run.add_argument("--validation-down", type=pathlib.Path, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--layer", type=int, default=55)
    run.add_argument("--gate-k", type=int, default=5)
    run.add_argument("--down-k", type=int, default=6)
    run.add_argument("--codebook", choices=("mcg", "mul1", "3inst"), default="mcg")
    run.add_argument("--seed", type=int, default=3401)
    run.add_argument("--relative-ridge", type=float, default=1e-4)
    run.add_argument("--seam-strength", type=float, default=1.0)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output", type=pathlib.Path, required=True)
    run.add_argument("--receipt", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "self-test":
        result = self_test()
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_bytes(canonical_bytes(result) + b"\n")
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "screen-architecture":
        result = screen_architecture_capture(
            args.gate_up, args.down, args.qkv,
            device_name=args.device, repetitions=args.repetitions,
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_bytes(canonical_bytes(result) + b"\n")
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "run-exl3-factorial":
        result = run_actual_exl3_factorial(args)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(canonical_bytes(result) + b"\n")
        print(json.dumps({"status": result["status"], "receipt": str(args.receipt)}, sort_keys=True))
        return
    arrays = np.load(args.input, allow_pickle=False)
    required = {"calibration_x", "calibration_teacher_y", "validation_x", "validation_teacher_y",
                "decoded_gate_up", "incumbent_down", "decoded_candidate_down"}
    missing = sorted(required - set(arrays.files))
    if missing:
        raise SystemExit(f"missing NPZ arrays: {missing}")
    h_cal = reconstructed_hidden(arrays["calibration_x"], arrays["decoded_gate_up"])
    h_val = reconstructed_hidden(arrays["validation_x"], arrays["decoded_gate_up"])
    fit = fit_down_target(h_cal, arrays["calibration_teacher_y"], arrays["incumbent_down"], args.relative_ridge)
    result = {
        "schema": SCHEMA, "input_sha256": sha256_file(args.input),
        "relative_ridge": args.relative_ridge, "absolute_ridge": fit.absolute_ridge,
        "rank": fit.rank, "target_sha256": fit.target_sha256, "continuous_target_persisted": False,
        "calibration": {"before_mse": fit.residual_mse_before,
                        "continuous_target_mse": fit.residual_mse_continuous},
        "validation": evaluate_candidate(h_val, arrays["validation_teacher_y"], arrays["decoded_candidate_down"]),
    }
    result["content_sha256"] = content_hash(result)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
