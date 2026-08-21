#!/usr/bin/env python3
"""R36 final-forward-KL guidance on the pinned stock EXL3 legal path.

The main arm changes only the encode-time Viterbi target.  Stock regularization,
FP16 scales, BlockLDLQ, the MCG graph, packing, and LinearEXL3 decoding remain
owned by :mod:`exl3_action`.  Conditional BCJR/Gumbel/QES helpers model the same
cyclic 16-bit state graph, but are gated off unless a hard projected payload wins
validation.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import random
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

K = 5
STATE_BITS = 16 - K
N_STATES = 1 << STATE_BITS
N_BRANCHES = 1 << K
TILE_VALUES = 256
MCG_MULT = np.uint64(0xCBAC1FED)
EXL3_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"
EXL3_TREE = "ffc0a1d31c25d4174b96adffef3727f12a7056c7"
HARNESS_SHA256 = "d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d"
SCHEMA_SHA256 = "275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8"
DATA_MANIFEST_FILE_SHA256 = "68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37"
DATA_MANIFEST_CONTENT_SHA256 = "51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2"
SPLIT_MANIFEST_FILE_SHA256 = "a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc"
SPLIT_MANIFEST_CONTENT_SHA256 = "151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e"
FISHER_MANIFEST_FILE_SHA256 = "4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a"
FISHER_MANIFEST_CONTENT_SHA256 = "28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237"
VALIDATION_SELECTION_SHA256 = "4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de"
CALIBRATION_SELECTION_SHA256 = "490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259"
TARGET_KEY = "model.language_model.layers.0.mlp.gate_proj.weight"
TARGET_SOURCE_SHA256 = "33f8e7677f361290d5f6a415c00cb56149a8fdbea20122bc13f0e1004b469565"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mcg_decode(indices: np.ndarray) -> np.ndarray:
    """Bit/FP16-exact CPU transcription of stock ``decode_3inst<1>``."""
    x = (indices.astype(np.uint64) * MCG_MULT) & np.uint64(0xFFFFFFFF)
    x = ((x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    low = (x & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (x >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return np.add(low, high, dtype=np.float16).astype(np.float32)


def predecessor_table() -> np.ndarray:
    """Actual K5 graph: 32 predecessors for each of 2048 output states."""
    out_state = np.arange(N_STATES, dtype=np.uint32)[None, :]
    branch = np.arange(N_BRANCHES, dtype=np.uint32)[:, None]
    encoded = (branch << np.uint32(STATE_BITS)) | out_state
    return (encoded >> np.uint32(K)).astype(np.uint16)


def encoded_table() -> np.ndarray:
    pred = predecessor_table().astype(np.uint32)
    out_state = np.arange(N_STATES, dtype=np.uint32)[None, :]
    return ((pred << np.uint32(K)) | out_state).astype(np.uint16)


def hard_viterbi(values: np.ndarray, target: np.ndarray, start_state: int) -> tuple[np.ndarray, float]:
    """Hard path on the actual state graph with fixed cyclic boundary state."""
    if values.shape != (N_BRANCHES, N_STATES) or target.shape != (TILE_VALUES,):
        raise ValueError("invalid actual-graph shapes")
    pred = predecessor_table().astype(np.int64)
    cost = np.full(N_STATES, np.inf, dtype=np.float64)
    cost[start_state] = 0.0
    back = np.empty((TILE_VALUES, N_STATES), dtype=np.uint16)
    for t, y in enumerate(target.astype(np.float64)):
        branch_cost = (values.astype(np.float64) - y) ** 2 + cost[pred]
        choice = np.argmin(branch_cost, axis=0)
        back[t] = pred[choice, np.arange(N_STATES)]
        cost = branch_cost[choice, np.arange(N_STATES)]
    end_cost = float(cost[start_state])
    path = np.empty(TILE_VALUES, dtype=np.uint16)
    state = start_state
    for t in range(TILE_VALUES - 1, -1, -1):
        prev = int(back[t, state])
        path[t] = np.uint16((prev << K) | state)
        state = prev
    if state != start_state:
        raise AssertionError("cyclic state closure failed")
    return path, end_cost


def bcjr_edge_marginals(
    values: np.ndarray, target: np.ndarray, start_state: int, temperature: float
) -> np.ndarray:
    """Low-temperature sum-product on the actual cyclic graph.

    This is conditional on one cyclic boundary state, matching the stock second
    Viterbi pass.  It is not a relaxed output: callers must hard-project by the
    pinned stock CUDA Viterbi before serialization.
    """
    if not (temperature > 0 and math.isfinite(temperature)):
        raise ValueError("temperature must be positive and finite")
    pred = predecessor_table().astype(np.int64)
    loga = np.full((TILE_VALUES + 1, N_STATES), -np.inf, dtype=np.float64)
    loga[0, start_state] = 0.0
    emissions = np.empty((TILE_VALUES, N_BRANCHES, N_STATES), dtype=np.float64)
    for t, y in enumerate(target.astype(np.float64)):
        emissions[t] = -((values.astype(np.float64) - y) ** 2) / temperature
        terms = loga[t, pred] + emissions[t]
        maximum = np.max(terms, axis=0)
        finite = np.isfinite(maximum)
        next_row = np.full(N_STATES, -np.inf, dtype=np.float64)
        next_row[finite] = maximum[finite] + np.log(
            np.exp(terms[:, finite] - maximum[finite]).sum(axis=0)
        )
        loga[t + 1] = next_row
    logb = np.full((TILE_VALUES + 1, N_STATES), -np.inf, dtype=np.float64)
    logb[-1, start_state] = 0.0
    # Accumulate outgoing transitions into predecessor states.
    for t in range(TILE_VALUES - 1, -1, -1):
        contrib = emissions[t] + logb[t + 1][None, :]
        row = np.full(N_STATES, -np.inf, dtype=np.float64)
        for branch in range(N_BRANCHES):
            np.logaddexp.at(row, pred[branch], contrib[branch])
        logb[t] = row
    logz = loga[-1, start_state]
    edge_logp = loga[:-1][:, pred] + emissions + logb[1:, None, :] - logz
    # Return [time, branch, out_state].
    return np.exp(edge_logp)


def legal_gumbel_target(target: np.ndarray, seed: int, scale: float) -> np.ndarray:
    """Fixed-seed target jitter; stock Viterbi remains the sole path projector."""
    rng = np.random.default_rng(seed)
    u = np.clip(rng.random(target.shape), 1e-7, 1 - 1e-7)
    return target + scale * (-np.log(-np.log(u))).astype(target.dtype)


def qes_directions(shape: tuple[int, ...], evaluations: int, seed: int) -> Iterable[np.ndarray]:
    """Antithetic zero-order directions with an exact disclosed evaluation cap."""
    if evaluations < 2 or evaluations % 2:
        raise ValueError("QES evaluation budget must be a positive antithetic pair count")
    rng = np.random.default_rng(seed)
    for _ in range(evaluations // 2):
        direction = rng.choice(np.array([-1.0, 1.0], np.float32), size=shape)
        yield direction
        yield -direction


def frozen_working_tiles(count: int = 20, seed: int = 0x523336) -> list[tuple[int, int]]:
    """Preregister 20 distinct (input tile, output tile) coordinates."""
    anchors = [(0, 0), (159, 544), (319, 1087), (0, 1087), (319, 0)]
    rng = random.Random(seed)
    seen = set(anchors)
    while len(anchors) < count:
        coordinate = (rng.randrange(320), rng.randrange(1088))
        if coordinate not in seen:
            seen.add(coordinate)
            anchors.append(coordinate)
    return anchors


def decoder_adjoint(gradient_source: Any, suh: Any, svh: Any, hadamard: Callable[[Any, int], Any]) -> Any:
    """Adjoint of the *serialized FP16-scale* LinearEXL3 decode map.

    Inputs use source ``[out,in]`` layout.  Output is stock working ``[in,out]``
    layout before tensor-core permutation.
    """
    g = gradient_source.T.contiguous().float()
    g = g * svh.float().reshape(1, -1)
    g = hadamard(g, 128)  # right H128 is self-adjoint
    g = g * suh.float().reshape(-1, 1)
    return hadamard(g.T.contiguous(), 128).T.contiguous()  # left H128


def decoder_forward(working: Any, suh: Any, svh: Any, hadamard: Callable[[Any, int], Any]) -> Any:
    q = hadamard(working.T.contiguous(), 128).T.contiguous()
    q = q * suh.float().reshape(-1, 1)
    q = hadamard(q, 128)
    q = q * svh.float().reshape(1, -1)
    return q.T.contiguous()


def inner_product_check(
    gradient_source: Any, working_delta: Any, suh: Any, svh: Any,
    hadamard: Callable[[Any, int], Any]
) -> dict[str, Any]:
    import torch
    source_delta = decoder_forward(working_delta, suh, svh, hadamard)
    working_gradient = decoder_adjoint(gradient_source, suh, svh, hadamard)
    source_ip = torch.sum(gradient_source.double() * source_delta.double()).item()
    working_ip = torch.sum(working_gradient.double() * working_delta.double()).item()
    abs_error = abs(source_ip - working_ip)
    tolerance = 2e-6 * max(1.0, abs(source_ip), abs(working_ip))
    return {
        "source_inner_product": source_ip,
        "working_inner_product": working_ip,
        "absolute_error": abs_error,
        "tolerance": tolerance,
        "pass": abs_error <= tolerance,
    }


def self_test() -> dict[str, Any]:
    import torch
    # Validate graph topology and hard/soft low-temperature consistency.
    pred = predecessor_table()
    enc = encoded_table()
    if pred.shape != (32, 2048) or enc.shape != pred.shape:
        raise AssertionError("actual K5 graph dimensions changed")
    if not np.array_equal((enc.astype(np.uint32) >> K).astype(np.uint16), pred):
        raise AssertionError("encoded predecessor relation changed")
    values = mcg_decode(enc)
    target = np.linspace(-2.0, 2.0, TILE_VALUES, dtype=np.float32)
    hard, hard_cost = hard_viterbi(values, target, start_state=17)
    if int(hard[-1] & (N_STATES - 1)) != 17:
        raise AssertionError("hard projection did not return to cyclic boundary")
    # Full BCJR is intentionally not part of the default self-test (large); test
    # its recurrence on a constant target through one hard projection instead.
    gumbel = legal_gumbel_target(target, 36, 0.01)
    g_path, _ = hard_viterbi(values, gumbel, start_state=17)
    if g_path.shape != hard.shape:
        raise AssertionError("legal Gumbel projection shape changed")
    directions = list(qes_directions((16,), 4, 36))
    if not np.array_equal(directions[0], -directions[1]):
        raise AssertionError("QES directions are not antithetic")

    # Synthetic orthonormal H2 tiled to exercise forward/adjoint ordering.
    def had2(x: torch.Tensor, _block: int) -> torch.Tensor:
        shape = x.shape
        y = x.reshape(-1, 2)
        a, b = y[:, 0].clone(), y[:, 1].clone()
        y[:, 0] = (a + b) / math.sqrt(2)
        y[:, 1] = (a - b) / math.sqrt(2)
        return y.reshape(shape)

    torch.manual_seed(36)
    grad = torch.randn(4, 6)
    delta = torch.randn(6, 4)
    adjoint = inner_product_check(grad, delta, torch.rand(6) + 0.5, torch.rand(4) + 0.5, had2)
    if not adjoint["pass"]:
        raise AssertionError(f"decoder adjoint failed: {adjoint}")
    return {
        "status": "pass",
        "graph": {"K": K, "states": N_STATES, "branches_per_state": N_BRANCHES,
                  "positions": TILE_VALUES, "cyclic": True, "codebook": "mcg",
                  "kernel_source_sha256": "85a9ab6295362212f3c6edc990cb6edb57c77a7b5473fe89b5109fdf57c28bfa"},
        "hard_path_cost": hard_cost,
        "hard_path_sha256": sha256_bytes(hard.tobytes()),
        "adjoint": adjoint,
        "working_tiles": frozen_working_tiles(),
    }


class LegalPathTargetCallback:
    """Stateful bound method; every constructor field is repeated in action params."""

    def __init__(self, mode: str, gradient_path: pathlib.Path | None, strength: float, seed: int):
        import torch
        self.mode = mode
        self.strength = strength
        self.seed = seed
        self.selected = set(frozen_working_tiles())
        self.gradient = None if gradient_path is None else torch.from_numpy(
            np.load(gradient_path, mmap_mode="r", allow_pickle=False)
        ).float()
        self.call_index = 0

    def __call__(self, tiles, _stock_values, _stock_indices, _action):
        import torch
        # Stock ldlq visits input 16-row tiles in a frozen reverse order.
        input_tile = 319 - self.call_index
        self.call_index += 1
        if self.mode == "zero":
            return tiles
        out = tiles.clone()
        perm = __import__(
            "exllamav3.modules.quant.exl3_lib.quantize", fromlist=["tensor_core_perm"]
        ).tensor_core_perm(tiles.device)
        for output_tile in range(1088):
            if (input_tile, output_tile) not in self.selected:
                continue
            if self.mode == "guided":
                if self.gradient is None or tuple(self.gradient.shape) != (5120, 17408):
                    raise ValueError("guided mode requires [5120,17408] working gradient")
                block = self.gradient[
                    input_tile * 16:(input_tile + 1) * 16,
                    output_tile * 16:(output_tile + 1) * 16,
                ].to(tiles.device)
                direction = block.reshape(-1)[perm]
            elif self.mode == "random":
                generator = torch.Generator(device="cpu").manual_seed(
                    self.seed ^ (input_tile << 12) ^ output_tile
                )
                direction = torch.randint(
                    0, 2, (256,), generator=generator, dtype=torch.int8
                ).float().mul_(2).sub_(1).to(tiles.device)
            else:
                raise ValueError(f"unsupported hard mode {self.mode}")
            rms = tiles[output_tile].square().mean().sqrt().clamp_min(1e-12)
            norm = direction.square().mean().sqrt().clamp_min(1e-12)
            out[output_tile] = tiles[output_tile] - self.strength * rms * direction / norm
        return out


def build_callbacks(mode: str, gradient_path: pathlib.Path | None, strength: float, seed: int):
    """Build an action-bound callback.  Scales are never changed here."""
    from exl3_action import EncodeCallbacks
    callback = LegalPathTargetCallback(mode, gradient_path, strength, seed)
    return EncodeCallbacks(
        identifier=("identity/strength-zero" if mode == "zero" else f"r36/{mode}/hard-stock-viterbi"),
        legal_path=callback.__call__,
        parameters={"mode": mode, "strength": strength, "seed": seed,
                    "gradient_path": None if gradient_path is None else str(gradient_path),
                    "selected_working_tiles": frozen_working_tiles(),
                    "scale_fit": "stock-unshifted-source", "hard_projection": "pinned-stock-cuda-viterbi"},
    )


def encode_action(args: argparse.Namespace) -> dict[str, Any]:
    """Run one complete actual-stock full-tensor action inside R30's container."""
    import torch
    from safetensors.torch import save_file
    from exl3_action import (SplitDisjointness, StockEXL3, Unit, load_safetensor,
                             make_curvature_capture, make_stock_action, payload_digest)

    source = load_safetensor(args.source, args.key)
    if tuple(source.shape) != (17408, 5120):
        raise ValueError(f"unexpected target shape {tuple(source.shape)}")
    source_sha = sha256_bytes(source.contiguous().view(torch.uint8).numpy().tobytes())
    if source_sha != TARGET_SOURCE_SHA256:
        raise ValueError(f"wrong source tensor {source_sha}")
    H = torch.from_numpy(np.load(args.curvature, mmap_mode="r", allow_pickle=False)).float().contiguous()
    callbacks = None if args.mode == "stock" else build_callbacks(
        args.mode, args.gradient, args.strength, args.seed
    )
    unit = Unit(
        unit_id="qwen38.layer0.mlp.gate_proj", granularity="tensor",
        topology="mlp", role="gate", tensor_keys=(args.key,), layer_index=0,
    )
    curvature = make_curvature_capture(
        H, unit, capture_id="r29/bf16-reference/l0-gate-up/XtX-over-63",
        observation_count=63, normalization="X^T X / N", basis="source-model input-feature activation basis",
        coordinate_convention="H rows/columns equal stored [out,in] source tensor input-feature axis before stock seeded sign/H128",
    )
    selections = {
        "calibration": {"selection_sha256": CALIBRATION_SELECTION_SHA256,
                        "selector": {"field": "split", "op": "eq", "value": "calibration"}},
        "validation": {"selection_sha256": VALIDATION_SELECTION_SHA256,
                       "selector": {"field": "split", "op": "eq", "value": "validation"}},
        "untouched_test": {"selection_sha256": "4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca",
                           "selector": {"field": "split", "op": "eq", "value": "untouched_test"}},
    }
    disjoint = SplitDisjointness(
        artifact_sha256="7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9",
        predicate_language="wave5.split-predicate/1",
        pairwise_overlap_counts={"calibration__validation": 0, "calibration__untouched_test": 0,
                                 "validation__untouched_test": 0},
        source_document_overlap_count=0, domain_leakage_count=0, verified=True,
    )
    action = make_stock_action(
        action_id=f"R36.{args.mode}.K5.mcg.s{args.strength:g}", unit=unit, K=5, codebook="mcg",
        seed=args.seed, source_sha256=source_sha, source_revision=SOURCE_REVISION, source_layout="out_in",
        curvature=curvature, callbacks=callbacks, split_manifest_sha256=SPLIT_MANIFEST_FILE_SHA256,
        split_manifest_content_sha256=SPLIT_MANIFEST_CONTENT_SHA256,
        split_selections=selections, split_disjointness=disjoint,
        evidence={"local_metrics": {}, "promoted_kld": None},
    )
    codec = StockEXL3()
    payload = codec.encode(source, H, action, callbacks=callbacks, device="cuda:0", verbose=False)
    decoded = codec.decode(payload).cpu()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    serialized = payload.serialize(out / "payload.safetensors")
    save_file({"weight": decoded.to(torch.bfloat16)}, str(out / "decoded.safetensors"))
    receipt = {
        "schema": "wave5/r36-encoded-action/1", "stage": "calibration-legal-encode",
        "mode": args.mode, "strength": args.strength, "seed": args.seed,
        "action": dataclasses.asdict(action), "action_identity_sha256": action.identity_sha256(),
        "payload": serialized, "payload_sha256": payload_digest(payload.tensors),
        "decoded_sha256": sha256_bytes(decoded.contiguous().view(torch.uint8).numpy().tobytes()),
        "source_sha256": source_sha, "finite": bool(torch.isfinite(decoded).all().item()),
        "complete_action_evaluations": 1,
        "legal_encoder_evaluations": 1 if args.mode == "stock" else 2,
        "viterbi_calls_per_tile": 1 if args.mode == "stock" else 2,
        "selected_working_tiles": [] if args.mode == "stock" else frozen_working_tiles(),
        "conditional_methods_run": False,
    }
    (out / "receipt.json").write_bytes(canonical_bytes(receipt))
    return receipt


def capture_final_kl_gradient(args: argparse.Namespace) -> dict[str, Any]:
    """Capture d KL(BF16||stock-K5-anchor) / d source-basis weight.

    This runs only on calibration tokens.  The teacher and candidate are the
    same BF16 model except that the candidate's L0 gate weight is the decoded
    actual-stock K5 anchor.
    """
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText

    tokens = json.loads(pathlib.Path(args.tokens).read_text())[:args.positions]
    if len(tokens) < 2:
        raise ValueError("gradient capture needs at least two calibration tokens")
    token_sha = sha256_file(pathlib.Path(args.tokens))
    if args.expected_token_sha256 and token_sha != args.expected_token_sha256:
        raise ValueError(f"calibration token identity mismatch: {token_sha}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = dict(model.named_modules())[args.module]
    embedding = model.get_input_embeddings()
    ids = torch.tensor(tokens, dtype=torch.long, device=next(embedding.parameters()).device).unsqueeze(0)
    with torch.no_grad():
        teacher_logits = model(input_ids=ids, use_cache=False).logits.detach().float().cpu()
    with safe_open(str(args.decoded), framework="pt", device="cpu") as handle:
        anchor = handle.get_tensor("weight")
    if tuple(anchor.shape) != tuple(module.weight.shape):
        raise ValueError(f"anchor/module shape mismatch: {anchor.shape} vs {module.weight.shape}")
    original = module.weight
    anchor_parameter = torch.nn.Parameter(anchor.to(device=original.device, dtype=torch.bfloat16), requires_grad=True)
    module.weight = anchor_parameter
    candidate_logits = model(input_ids=ids, use_cache=False).logits.float()
    teacher = teacher_logits.to(candidate_logits.device)
    teacher_logp = functional.log_softmax(teacher, dim=-1)
    teacher_p = teacher_logp.exp()
    candidate_logq = functional.log_softmax(candidate_logits, dim=-1)
    per_position = torch.sum(teacher_p * (teacher_logp - candidate_logq), dim=-1)
    loss = per_position.mean()
    loss.backward()
    gradient = anchor_parameter.grad.detach().float().cpu().numpy()
    if not np.isfinite(gradient).all():
        raise AssertionError("final-KL source gradient is non-finite")
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gradient_path = out / "source-gradient.npy"
    np.save(gradient_path, gradient, allow_pickle=False)
    result = {
        "schema": "wave5/r36-final-kl-gradient/1",
        "split": "calibration",
        "split_selection_sha256": CALIBRATION_SELECTION_SHA256,
        "token_sha256": token_sha,
        "positions": len(tokens),
        "vocab_size": int(candidate_logits.shape[-1]),
        "direction": "KL(BF16 reference || actual-stock-K5 one-tensor candidate)",
        "full_vocabulary": True,
        "loss_mean_nats": float(loss.detach().cpu()),
        "per_position_min": float(per_position.min().detach().cpu()),
        "per_position_max": float(per_position.max().detach().cpu()),
        "source_gradient_shape": list(gradient.shape),
        "source_gradient_l2": float(np.linalg.norm(gradient.astype(np.float64))),
        "source_gradient_sha256": sha256_file(gradient_path),
        "candidate_anchor": str(args.decoded),
        "candidate_anchor_sha256": sha256_file(pathlib.Path(args.decoded)),
        "relinearization_required_after_acceptance": True,
    }
    (out / "gradient-receipt.json").write_bytes(canonical_bytes(result))
    return result


def apply_decoder_adjoint(args: argparse.Namespace) -> dict[str, Any]:
    """Map a source gradient through the stock serialized-scale decoder adjoint."""
    import torch
    from safetensors import safe_open
    from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r

    with safe_open(str(args.payload), framework="pt", device="cpu") as handle:
        suh = handle.get_tensor("suh").float().to(args.device)
        svh = handle.get_tensor("svh").float().to(args.device)
    source_gradient = torch.from_numpy(
        np.load(args.source_gradient, mmap_mode="r", allow_pickle=False)
    ).float().to(args.device)
    if tuple(source_gradient.shape) != (17408, 5120):
        raise ValueError(f"wrong source gradient shape {source_gradient.shape}")
    g = source_gradient.T.contiguous()
    g.mul_(svh.reshape(1, -1))
    g = preapply_had_r(g, 128)
    g.mul_(suh.reshape(-1, 1))
    working_gradient = preapply_had_l(g, 128)

    def decoded_to_working(path: pathlib.Path) -> Any:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            source = handle.get_tensor("weight").float().to(args.device).T.contiguous()
        source.div_(svh.reshape(1, -1))
        source = preapply_had_r(source, 128)
        source.div_(suh.reshape(-1, 1))
        return preapply_had_l(source, 128)

    stock_working = decoded_to_working(args.stock_decoded)
    legal_working = decoded_to_working(args.legal_decoded)
    working_delta = legal_working - stock_working
    with safe_open(str(args.legal_decoded), framework="pt", device="cpu") as handle:
        legal_source = handle.get_tensor("weight").float().to(args.device)
    with safe_open(str(args.stock_decoded), framework="pt", device="cpu") as handle:
        stock_source = handle.get_tensor("weight").float().to(args.device)
    source_delta = legal_source - stock_source
    source_ip = torch.sum(source_gradient.double() * source_delta.double()).item()
    working_ip = torch.sum(working_gradient.double() * working_delta.double()).item()
    absolute_error = abs(source_ip - working_ip)
    tolerance = 5e-5 * max(1.0, abs(source_ip), abs(working_ip))
    block_rows = []
    for input_tile, output_tile in frozen_working_tiles():
        row = slice(input_tile * 16, (input_tile + 1) * 16)
        col = slice(output_tile * 16, (output_tile + 1) * 16)
        delta_block = working_delta[row, col]
        linear_term = torch.sum(
            working_gradient[row, col].double() * delta_block.double()
        ).item()
        block_rows.append({
            "input_tile": input_tile,
            "output_tile": output_tile,
            "linear_kl_term": linear_term,
            "changed_values": int(torch.count_nonzero(delta_block).item()),
        })
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gradient_path = out / "working-gradient.npy"
    np.save(gradient_path, working_gradient.float().cpu().numpy(), allow_pickle=False)
    result = {
        "schema": "wave5/r36-decoder-adjoint/1",
        "serialized_scale_dtype": "float16",
        "source_layout": "out_in",
        "working_layout": "in_out-before-tensor-core-permutation",
        "working_gradient_sha256": sha256_file(gradient_path),
        "working_gradient_shape": list(working_gradient.shape),
        "legal_displacement": {
            "stock_decoded_sha256": sha256_file(args.stock_decoded),
            "legal_decoded_sha256": sha256_file(args.legal_decoded),
            "source_inner_product": source_ip,
            "working_inner_product": working_ip,
            "absolute_error": absolute_error,
            "tolerance": tolerance,
            "pass": absolute_error <= tolerance,
        },
        "selected_block_signal": {
            "rows": block_rows,
            "negative": sum(row["linear_kl_term"] < 0 for row in block_rows),
            "positive": sum(row["linear_kl_term"] > 0 for row in block_rows),
            "zero": sum(row["linear_kl_term"] == 0 for row in block_rows),
            "stable_negative_gate": sum(row["linear_kl_term"] < 0 for row in block_rows) >= 16,
        },
    }
    if not result["legal_displacement"]["pass"]:
        raise AssertionError(f"decoder adjoint legal-displacement check failed: {result}")
    (out / "adjoint-receipt.json").write_bytes(canonical_bytes(result))
    return result


def materialize_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Hard-project one action into an otherwise hardlinked stock checkpoint."""
    import os
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate checkpoint {output}")
    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.name != args.shard:
            destination = output / path.name
            if path.is_dir():
                shutil.copytree(path, destination, copy_function=os.link)
            else:
                os.link(path, destination)
    shard_source = source / args.shard
    tensors: dict[str, Any] = {}
    with safe_open(str(shard_source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    replacements: dict[str, Any] = {}
    with safe_open(str(args.payload), framework="pt", device="cpu") as handle:
        for suffix in handle.keys():
            key = f"{args.prefix}.{suffix}"
            if key not in tensors:
                raise KeyError(f"checkpoint lacks replacement key {key}")
            value = handle.get_tensor(suffix)
            if value.shape != tensors[key].shape or value.dtype != tensors[key].dtype:
                raise ValueError(
                    f"replacement {key} shape/dtype {value.shape}/{value.dtype} "
                    f"!= {tensors[key].shape}/{tensors[key].dtype}"
                )
            replacements[key] = value
    if set(replacements) != {f"{args.prefix}.{name}" for name in ("suh", "svh", "trellis", "mcg")}:
        raise ValueError(f"incomplete stock payload replacement: {sorted(replacements)}")
    tensors.update(replacements)
    save_file(tensors, str(output / args.shard), metadata=metadata)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    for key in replacements:
        if index["weight_map"].get(key) != args.shard:
            raise AssertionError(f"index maps {key} to the wrong shard")
    components = []
    total_bytes = 0
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total_bytes += size
        components.append({"path": str(path.relative_to(output)), "bytes": size,
                           "sha256": sha256_file(path)})
    checkpoint_identity = sha256_bytes(canonical_bytes(components))
    result = {
        "schema": "wave5/r36-checkpoint-materialization/1",
        "source_checkpoint": str(source),
        "candidate_checkpoint": str(output),
        "replaced_prefix": args.prefix,
        "replaced_shard": args.shard,
        "payload_sha256": sha256_file(args.payload),
        "component_bytes": total_bytes,
        "components": components,
        "checkpoint_identity_sha256": checkpoint_identity,
        "unchanged_decoder": True,
        "incremental_action_bytes": 0,
        "hard_projection": "stock serialized buffers copied without reinterpretation",
    }
    (output / "r36-materialization.json").write_bytes(canonical_bytes(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    encode = sub.add_parser("encode-action")
    encode.add_argument("--source", required=True)
    encode.add_argument("--key", default=TARGET_KEY)
    encode.add_argument("--curvature", required=True, type=pathlib.Path)
    encode.add_argument("--gradient", type=pathlib.Path)
    encode.add_argument("--mode", choices=("stock", "zero", "guided", "random"), required=True)
    encode.add_argument("--strength", type=float, required=True)
    encode.add_argument("--seed", type=int, default=360036)
    encode.add_argument("--output", required=True)
    gradient = sub.add_parser("capture-gradient")
    gradient.add_argument("--model", required=True)
    gradient.add_argument("--tokens", required=True, type=pathlib.Path)
    gradient.add_argument("--expected-token-sha256")
    gradient.add_argument("--decoded", required=True, type=pathlib.Path)
    gradient.add_argument("--module", default="model.language_model.layers.0.mlp.gate_proj")
    gradient.add_argument("--positions", type=int, default=63)
    gradient.add_argument("--gpu-memory", default="26GiB")
    gradient.add_argument("--cpu-memory", default="160GiB")
    gradient.add_argument("--output", required=True)
    adjoint = sub.add_parser("adjoint-gradient")
    adjoint.add_argument("--payload", required=True, type=pathlib.Path)
    adjoint.add_argument("--source-gradient", required=True, type=pathlib.Path)
    adjoint.add_argument("--stock-decoded", required=True, type=pathlib.Path)
    adjoint.add_argument("--legal-decoded", required=True, type=pathlib.Path)
    adjoint.add_argument("--device", default="cuda:0")
    adjoint.add_argument("--output", required=True)
    materialize = sub.add_parser("materialize-checkpoint")
    materialize.add_argument("--source", required=True, type=pathlib.Path)
    materialize.add_argument("--payload", required=True, type=pathlib.Path)
    materialize.add_argument("--prefix", default="model.language_model.layers.0.mlp.gate_proj")
    materialize.add_argument("--shard", default="model-00001-of-00003.safetensors")
    materialize.add_argument("--output", required=True, type=pathlib.Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "self-test":
        result = self_test()
    elif args.command == "encode-action":
        result = encode_action(args)
    elif args.command == "capture-gradient":
        result = capture_final_kl_gradient(args)
    elif args.command == "adjoint-gradient":
        result = apply_decoder_adjoint(args)
    elif args.command == "materialize-checkpoint":
        result = materialize_checkpoint(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
