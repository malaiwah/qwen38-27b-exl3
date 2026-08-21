#!/usr/bin/env python3
"""Exact stock-EXL3 action harness for Wave 5.

The authoritative path imports the pinned ExLlamaV3 1.4.2 implementation and
uses its quantize_exl3, block_ldl(b=16), quantize_tiles/Viterbi, pack_trellis,
and LinearEXL3 reconstruction APIs.  This file deliberately contains no
clean-room quantizer or affine-uniform fallback.

The module is importable without torch/safetensors so action manifests can be
validated and allocated on a Mac.  Encoding requires the pinned CUDA extension
on aiboss.  Optional callbacks are encode-only: they can alter declared stages,
but their result is packed into the unchanged {suh, svh, trellis, marker}
schema and decoded by LinearEXL3.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import dataclasses
import hashlib
import importlib
import inspect
import importlib.util
import json
import math
import os
import pathlib
import sys
import threading
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

SCHEMA_ID = "wave5/exl3-action/1"
STOCK_REPO = "turboderp-org/exllamav3"
STOCK_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"
STOCK_VERSION = "1.4.2"
STOCK_EXTENSION_SOURCE_SHA256 = {
    "exllamav3_ext/bindings.cpp": "6e1ebdbd2cedacf7672a9de272bf70cb7ab0282088f6a2f55a4d55cef11dff95",
    "exllamav3_ext/quant/quantize.cu": "cee125a3e4bf8f12681380f52cf0ab9b0a586c7c12f167be57b073ba5557a73b",
    "exllamav3_ext/quant/quantize_tiles_kernel.cuh": "85a9ab6295362212f3c6edc990cb6edb57c77a7b5473fe89b5109fdf57c28bfa",
    "exllamav3_ext/quant/pack.cu": "27606eed6650acc31c6b6484aad1e89195da88823a5bd62ffb3e9911a9b47e60",
}
STOCK_PYTHON_DEPENDENCY_SHA256 = {
    "util/hadamard.py": "6884841b6137878874ee0b2942ec2f62cb6275a40ffc853146a73b2d92233cbb",
}
STOCK_QUANTIZE_SHA256 = "4cd368dab28e007d649e25b97c65fc73a56ef2a1482ca2b9298a53d4b0876dbf"
STOCK_TREE_SHA1 = "ffc0a1d31c25d4174b96adffef3727f12a7056c7"
STOCK_EXTENSION_BINARY_SHA256 = "79815da8b7d39559c2dea17cffb966fe7d78beba5b67c2f49f7f41832c40b2bf"
STOCK_DECODER_SHA256 = "c010bd18aaf5363632db25c0a4f7c4be0938011656f0446f933505a59b8d6cc0"
HADAMARD_BLOCK = 128
LDL_BLOCK = 16
TILE = 16
LEGAL_STOCK_K = (4, 5, 6)
PRODUCTION_ROUTE_SHA256 = "dcede1b494984b3ec29fae5187e8aa692557e4658a1601c7dc0fc337737cbaa8"
CODEBOOK_MARKERS = {"3inst": None, "mcg": "mcg", "mul1": "mul1"}
MARKER_U32 = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SPLIT_NAMES = ("calibration", "validation", "untouched_test")
SPLIT_SELECTOR_PREDICATES: dict[str, dict[str, str]] = {
    name: {"field": "split", "op": "eq", "value": name}
    for name in SPLIT_NAMES
}
CALLBACK_EXPECTED_INTERFACE: dict[str, str] = {
    "target": "(source_basis_weight, action) -> same-shape/dtype/device finite target",
    "scale": "(stock_five_tuple, action) -> identical structure/dtypes/devices",
    "curvature": "(transformed_damped_H, action) -> same-shape/dtype/device finite H",
    "recurrence": "(stock_block_ldl_L, action) -> same-shape/dtype/device finite L",
    "legal_path": "(tiles, stock_values, stock_indices, action) -> candidate target tiles only; harness reruns pinned Viterbi",
}
CALLBACK_STAGES = tuple(CALLBACK_EXPECTED_INTERFACE)
REQUIRED_ROUTING_ENV = (
    "PROFILE",
    "VLLM_EXL3_MULTIPRECISION",
    "VLLM_EXL3_FP4_LAYERS",
    "VLLM_EXL3_FP6_LAYERS",
    "VLLM_EXL3_FP4_MODULE",
    "VLLM_EXL3_FP6_MODULE",
    "VLLM_EXL3_FP4_LAYER_RANGE",
    "VLLM_EXL3_FP6_LAYER_RANGE",
    "VLLM_EXL3_FP4_DRAFT_HEAD",
    "VLLM_EXL3_B12X_ANY_BITS",
    "VLLM_EXL3_B12X_MIN_M",
    "VLLM_EXL3_B12X_N_RANGE",
    "VLLM_EXL3_B12X_LM_HEAD_MIN_M",
    "VLLM_EXL3_PREFILL_RECONSTRUCT_M",
    "VLLM_EXL3_SKIP_TRELLIS_PREP",
)
COORDINATE_CONVENTIONS = {
    "out_in": "H rows/columns equal stored [out,in] source tensor input-feature axis before stock seeded sign/H128",
    "in_out": "H rows/columns equal stored [in,out] source tensor input-feature axis before stock seeded sign/H128",
}
PRODUCTION_EFFECTIVE_ENV = {
    "PROFILE": "throughput",
    "VLLM_EXL3_MULTIPRECISION": "1",
    "VLLM_EXL3_FP4_LAYERS": "mlp.gate_up_proj,mlp.down_proj,linear_attn.,self_attn.",
    "VLLM_EXL3_FP6_LAYERS": "",
    "VLLM_EXL3_FP4_MODULE": "/opt/fp4/exl3_fp4_conversion.py",
    "VLLM_EXL3_FP6_MODULE": "/opt/fp6/exl3_fp6_conversion.py",
    "VLLM_EXL3_FP4_LAYER_RANGE": "",
    "VLLM_EXL3_FP6_LAYER_RANGE": "",
    "VLLM_EXL3_FP4_DRAFT_HEAD": "0",
    "VLLM_EXL3_B12X_ANY_BITS": "0",
    "VLLM_EXL3_B12X_MIN_M": "0",
    "VLLM_EXL3_B12X_N_RANGE": "5120-32768",
    "VLLM_EXL3_B12X_LM_HEAD_MIN_M": "",
    "VLLM_EXL3_PREFILL_RECONSTRUCT_M": "1",
    "VLLM_EXL3_SKIP_TRELLIS_PREP": "0",
}

ROUTE_REGISTRY: dict[str, dict[str, Any]] = {
    "codec-exact/all-trellis-stock-exl3": {
        "route_id": "codec-exact/all-trellis-stock-exl3",
        "codec_exact": True,
        "checkpoint_schema": ["suh", "svh", "trellis", "mcg|mul1|none"],
        "qualification_env": {"VLLM_EXL3_MULTIPRECISION": "0"},
        "decode": "LinearEXL3/BC_LinearEXL3 or vLLM ext.exl3_gemm; prefill may reconstruct the identical trellis payload",
        "implementation_sha256": PRODUCTION_ROUTE_SHA256,
        "fallbacks": {
            "b12x": "MCG only; K6 by default, K3-K6 only when VLLM_EXL3_B12X_ANY_BITS=1; shape/N-window guards apply",
            "extension": "K4/K5/K6 and qualified K7; MCG, MUL1, and 3inst use the stock EXL3 extension when B12X is ineligible",
        },
        "hot_payload_unchanged": True,
    },
    "production/throughput-fp4-fp6-materialized": {
        "route_id": "production/throughput-fp4-fp6-materialized",
        "codec_exact": False,
        "checkpoint_schema": ["suh", "svh", "trellis", "mcg|mul1|none"],
        "qualification_env": PRODUCTION_EFFECTIVE_ENV,
        "decode": "load-time EXL3 reconstruction followed by configured FP4/FP6 materialization; unmatched/lm_head/failure paths retain trellis",
        "implementation_sha256": PRODUCTION_ROUTE_SHA256,
        "fallbacks": {
            "fp4_fp6": "selected by VLLM_EXL3_FP4_LAYERS/VLLM_EXL3_FP6_LAYERS; successful conversion clears trellis tensors",
            "trellis": "unmatched layers, lm_head verify route, or conversion failures remain on exact trellis/B12X-or-extension",
            "codebook": "materialization reconstructs the declared marker; residual B12X trellis requires MCG, never MUL1",
        },
        "hot_payload_unchanged": False,
    },
}

TOPOLOGY_CONTRACT: dict[str, Any] = {
    "model": "Qwen3.8-27B",
    "layers": 64,
    "hidden_size": 5120,
    "mlp_intermediate_size": 17408,
    "linear_attention_layers": 48,
    "gated_full_attention_layers": 16,
    "full_attention": {"q_heads": 24, "kv_heads": 4, "head_dim": 256, "rotary_dim": 64},
    "legal_granularity": ["topology_group", "module", "tensor", "shard"],
    "nondeployable_without_new_format": ["per_tile_k", "variable_stride_stripe", "selector_sidecar"],
}

KLD_METHOD_CONTRACT: dict[str, Any] = {
    "protocol_id": "qwen38-kld-method-v5/body-only-shared-bf16-head",
    "method_of_record": "docs/42-kld-method.md",
    "suite_repo": "malaiwah/qwen38-27b-fidelity-suite-v5",
    "suite_revision": "7797fcce3ffed62b99871348887f4626dc9b2b3b",
    "suite_manifest_sha256": "c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019",
    "suite_token_sha256": "510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88",
    "shared_bf16_head_sha256": "25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff",
    "direction": "KL(BF16 reference || candidate)",
    "vocab_size": 248320,
    "vocab_chunk": 24832,
    "accumulation": "float64",
    "two_pass": True,
    "body_only": True,
    "scored_positions_per_context": 2047,
    "contexts": 5120,
    "source_clusters": 842,
    "bootstrap": {"unit": "source_cluster", "resamples": 10000, "seed": 1},
    "required_metrics": [
        "token_mean_kld", "context_macro_mean_kld", "p99", "cvar1pct",
        "top1_agreement", "full_vocab_ear", "worst_contexts",
    ],
    "local_metric_guard": "MSE/OC-HWE/Fisher-HWE/block-output-error are never called KLD",
    "served_head_axis": "candidate-own/served head contribution is separate from body-only shared-head KLD",
}


class ContractError(ValueError):
    """Raised before encoding when an action violates the immutable contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def validate_sha256(name: str, value: str, *, allow_empty_digest: bool = False) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{name} must be a lowercase sha256")
    if value == "0" * 64:
        raise ContractError(f"{name} cannot be the zero digest")
    if not allow_empty_digest and value == SHA256_EMPTY:
        raise ContractError(f"{name} cannot be the empty-content digest")


def _callback_executable_identity(
    callbacks: "EncodeCallbacks | None",
) -> tuple[str, dict[str, str]]:
    stage_identity: dict[str, Any] = {}
    module_files: dict[str, str] = {}
    for stage in CALLBACK_STAGES:
        fn = getattr(callbacks, stage, None)
        if fn is None:
            stage_identity[stage] = None
            continue
        if getattr(fn, "__closure__", None):
            raise ContractError(
                f"{stage} callback must not close over undeclared state; put parameters in the action"
            )
        try:
            source = inspect.getsource(fn).encode("utf-8")
            source_file = inspect.getsourcefile(fn)
            signature = str(inspect.signature(fn))
            defaults = {"positional": fn.__defaults__, "keyword": fn.__kwdefaults__}
            defaults_sha256 = sha256_bytes(canonical_json(defaults))
        except (OSError, TypeError, ValueError) as exc:
            raise ContractError(f"{stage} callback executable identity is unavailable") from exc
        if not source_file:
            raise ContractError(f"{stage} callback source module is unavailable")
        module_key = f"{fn.__module__}:{pathlib.Path(source_file).name}"
        module_files[module_key] = sha256_file(source_file)
        stage_identity[stage] = {
            "source_sha256": sha256_bytes(source),
            "signature": signature,
            "defaults_sha256": defaults_sha256,
            "module": module_key,
        }
    return sha256_bytes(canonical_json(stage_identity)), module_files


def stock_dense_buffer_bytes(numel: int, in_features: int, out_features: int, K: int, marker: str) -> int:
    """Raw stored tensor bytes, excluding a containing safetensors header."""
    if K not in range(1, 9):
        raise ContractError(f"K must be 1..8, got {K}")
    bits = numel * K
    if bits % 8:
        raise ContractError("trellis bit count is not byte aligned")
    marker_bytes = 0 if marker == "3inst" else 4
    return bits // 8 + 2 * (in_features + out_features) + marker_bytes


def _tensor_bytes(tensor: Any) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    return cpu.reshape(-1).view(__import__("torch").uint8).numpy().tobytes(order="C")


def tensor_digest(tensor: Any) -> dict[str, Any]:
    raw = _tensor_bytes(tensor)
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }

def payload_digest(tensors: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        raw = _tensor_bytes(tensor)
        descriptor = canonical_json({
            "name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "bytes": len(raw),
        })
        h.update(len(descriptor).to_bytes(8, "little"))
        h.update(descriptor)
        h.update(raw)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class Unit:
    unit_id: str
    granularity: Literal["topology_group", "module", "tensor", "shard"]
    topology: Literal["mlp", "gdn", "full_attention", "lm_head", "mtp"]
    role: str
    tensor_keys: tuple[str, ...]
    layer_index: int | None = None
    shard_id: str | None = None
    fused_group: str | None = None
    output_splits: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.unit_id or not self.tensor_keys:
            raise ContractError("unit_id and tensor_keys are required")
        if self.layer_index is not None and not 0 <= self.layer_index < 64:
            raise ContractError(f"invalid layer index {self.layer_index}")
        if self.granularity == "shard" and self.shard_id is None:
            raise ContractError("shard granularity requires shard_id")
        if self.output_splits and any(x <= 0 or x % HADAMARD_BLOCK for x in self.output_splits):
            raise ContractError("fused output_splits must be positive and H128-aligned")


@dataclasses.dataclass(frozen=True)
class Recipe:
    recipe_id: str
    kind: str
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    implementation: str = "stock"
    strength: float | None = None

    def validate(self) -> None:
        if not self.recipe_id or not self.kind:
            raise ContractError("recipe_id and kind are required")
        if self.strength is not None and not math.isfinite(self.strength):
            raise ContractError("recipe strength must be finite")


@dataclasses.dataclass(frozen=True)
class CurvatureCapture:
    capture_id: str
    h_tensor_sha256: str
    h_dtype: Literal["float32"]
    h_shape: tuple[int, int]
    capture_manifest_sha256: str
    observation_count: int
    normalization: str
    basis: str
    coordinate_convention: str
    tensor_boundary: str
    module_boundary: str

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": "wave5/curvature-capture/1",
            "capture_id": self.capture_id,
            "h_tensor_sha256": self.h_tensor_sha256,
            "h_dtype": self.h_dtype,
            "h_shape": list(self.h_shape),
            "observation_count": self.observation_count,
            "normalization": self.normalization,
            "basis": self.basis,
            "coordinate_convention": self.coordinate_convention,
            "tensor_boundary": self.tensor_boundary,
            "module_boundary": self.module_boundary,
        }

    def validate(self, unit: Unit) -> None:
        if not self.capture_id:
            raise ContractError("curvature capture_id is required")
        validate_sha256("curvature.h_tensor_sha256", self.h_tensor_sha256)
        validate_sha256("curvature.capture_manifest_sha256", self.capture_manifest_sha256)
        if self.h_dtype != "float32" or len(self.h_shape) != 2 or self.h_shape[0] != self.h_shape[1]:
            raise ContractError("curvature H must declare a square float32 tensor")
        if self.observation_count <= 0:
            raise ContractError("curvature observation_count must be positive")
        for name in ("normalization", "basis", "coordinate_convention"):
            if not getattr(self, name):
                raise ContractError(f"curvature {name} is required")
        if self.tensor_boundary not in unit.tensor_keys:
            raise ContractError("curvature tensor_boundary must name a tensor in the action unit")
        if self.module_boundary != unit.unit_id:
            raise ContractError("curvature module_boundary must equal action unit_id")
        expected_manifest = sha256_bytes(canonical_json(self.manifest_payload()))
        if self.capture_manifest_sha256 != expected_manifest:
            raise ContractError("curvature capture manifest hash is stale or mismatched")

    def verify_sha256(self, actual: str) -> None:
        validate_sha256("observed curvature tensor", actual)
        if actual != self.h_tensor_sha256:
            raise ContractError(
                f"curvature tensor hash {actual} does not match action {self.h_tensor_sha256}"
            )

    def verify_tensor(self, H: Any) -> None:
        dtype = str(H.dtype).removeprefix("torch.")
        if dtype != self.h_dtype or tuple(H.shape) != self.h_shape:
            raise ContractError("observed curvature dtype/shape does not match the action")
        self.verify_sha256(sha256_bytes(_tensor_bytes(H)))


@dataclasses.dataclass(frozen=True)
class CallbackContract:
    identifier: str
    version: str
    implementation_sha256: str
    module_files_sha256: Mapping[str, str]
    content_sha256: str
    parameters: Mapping[str, Any]
    expected_interface: Mapping[str, str]

    def validate(self) -> None:
        if not self.identifier or not self.version:
            raise ContractError("callback identifier and version are required")
        validate_sha256("callback.implementation_sha256", self.implementation_sha256)
        for name, value in self.module_files_sha256.items():
            validate_sha256(f"callback.module_files_sha256[{name}]", value)
        validate_sha256("callback.content_sha256", self.content_sha256)
        if dict(self.expected_interface) != CALLBACK_EXPECTED_INTERFACE:
            raise ContractError("callback expected_interface must equal the frozen harness interface")
        expected_content = sha256_bytes(canonical_json({
            "identifier": self.identifier,
            "version": self.version,
            "implementation_sha256": self.implementation_sha256,
            "module_files_sha256": self.module_files_sha256,
            "parameters": self.parameters,
            "expected_interface": self.expected_interface,
        }))
        if self.content_sha256 != expected_content:
            raise ContractError("callback content_sha256 does not bind module/code/parameters/interface")


@dataclasses.dataclass(frozen=True)
class SplitDisjointness:
    artifact_sha256: str
    predicate_language: Literal["wave5.split-predicate/1"]
    pairwise_overlap_counts: Mapping[str, int]
    source_document_overlap_count: int
    domain_leakage_count: int
    verified: bool

    def validate(self) -> None:
        validate_sha256("split_disjointness.artifact_sha256", self.artifact_sha256)
        if self.predicate_language != "wave5.split-predicate/1":
            raise ContractError("split predicate language is not frozen")
        expected_pairs = {
            "calibration__validation",
            "calibration__untouched_test",
            "validation__untouched_test",
        }
        if set(self.pairwise_overlap_counts) != expected_pairs:
            raise ContractError("split disjointness must report all three pairwise overlaps")
        if any(v != 0 for v in self.pairwise_overlap_counts.values()):
            raise ContractError("split selectors overlap")
        if self.source_document_overlap_count != 0 or self.domain_leakage_count != 0:
            raise ContractError("split audit reports source-document or domain leakage")
        if self.verified is not True:
            raise ContractError("split disjointness must be verified")


@dataclasses.dataclass(frozen=True)
class RuntimeContract:
    route_id: Literal["codec-exact/all-trellis-stock-exl3", "production/throughput-fp4-fp6-materialized"]
    decode_hot_ops: tuple[str, ...]
    startup_ops: tuple[str, ...] = ()
    graph_capturable: bool = True
    incremental_hot_bytes: int = 0
    sidecar_bytes: int = 0
    materialization_qualification: Mapping[str, Any] | None = None

    def validate(self, action_payload_sha256: str | None = None) -> None:
        if self.route_id not in ROUTE_REGISTRY:
            raise ContractError(f"unknown route_id {self.route_id}")
        if self.incremental_hot_bytes < 0 or self.sidecar_bytes < 0:
            raise ContractError("runtime byte counts cannot be negative")
        if self.route_id == "codec-exact/all-trellis-stock-exl3":
            if self.materialization_qualification is not None:
                raise ContractError("codec-exact route cannot carry a materialization qualification")
            return
        q = self.materialization_qualification
        if not isinstance(q, Mapping):
            raise ContractError("production materialization route requires a qualification")
        hash_names = {
            "route_implementation_sha256", "image_digest_sha256",
            "fp4_converter_source_sha256", "fp4_converter_binary_sha256",
            "fp6_converter_source_sha256", "fp6_converter_binary_sha256",
            "effective_environment_sha256", "source_payload_sha256",
            "materialized_tensor_sha256", "materialized_payload_sha256",
            "runtime_receipt_sha256",
        }
        path_names = {
            "route_implementation_path", "fp4_converter_source_path",
            "fp4_converter_binary_path", "fp6_converter_source_path",
            "fp6_converter_binary_path", "source_payload_path",
            "materialized_tensor_path", "materialized_payload_path",
            "runtime_receipt_path",
        }
        required = {"route_id", "effective_environment"} | hash_names | path_names
        if set(q) != required:
            raise ContractError("materialization qualification fields are missing or unknown")
        if q["route_id"] != self.route_id:
            raise ContractError("materialization qualification route_id mismatch")
        for name in hash_names:
            validate_sha256(f"materialization.{name}", q[name])
        if q["route_implementation_sha256"] != ROUTE_REGISTRY[self.route_id]["implementation_sha256"]:
            raise ContractError("materialization route implementation hash mismatch")
        if not action_payload_sha256 or q["source_payload_sha256"] != action_payload_sha256:
            raise ContractError("materialization source payload does not equal the action payload")
        artifact_pairs = {
            "route_implementation_path": "route_implementation_sha256",
            "fp4_converter_source_path": "fp4_converter_source_sha256",
            "fp4_converter_binary_path": "fp4_converter_binary_sha256",
            "fp6_converter_source_path": "fp6_converter_source_sha256",
            "fp6_converter_binary_path": "fp6_converter_binary_sha256",
            "source_payload_path": "source_payload_sha256",
            "materialized_tensor_path": "materialized_tensor_sha256",
            "materialized_payload_path": "materialized_payload_sha256",
            "runtime_receipt_path": "runtime_receipt_sha256",
        }
        for path_name, hash_name in artifact_pairs.items():
            path = pathlib.Path(q[path_name])
            if not path.is_file() or sha256_file(path) != q[hash_name]:
                raise ContractError(f"materialization observed artifact mismatch: {path_name}")
        observed_image = os.environ.get("WAVE5_CONTAINER_IMAGE_DIGEST", "")
        if observed_image.startswith("sha256:"):
            observed_image = observed_image.removeprefix("sha256:")
        if observed_image != q["image_digest_sha256"]:
            raise ContractError("running container image digest does not match qualification")
        env = q["effective_environment"]
        if not isinstance(env, Mapping) or set(env) != set(REQUIRED_ROUTING_ENV):
            raise ContractError("materialization effective environment is incomplete")
        if any(not isinstance(v, str) for v in env.values()):
            raise ContractError("materialization effective environment values must be strings")
        if env["PROFILE"] != "throughput" or env["VLLM_EXL3_MULTIPRECISION"] != "1":
            raise ContractError("materialization effective environment does not select the qualified production route")
        if q["effective_environment_sha256"] != sha256_bytes(canonical_json(dict(env))):
            raise ContractError("materialization effective environment hash is stale")
        actual = {name: os.environ.get(name, "") for name in REQUIRED_ROUTING_ENV}
        if actual != dict(env):
            raise ContractError("running materialization environment does not match the frozen qualification")


@dataclasses.dataclass(frozen=True)
class EXL3Action:
    action_id: str
    unit: Unit
    K: int
    codebook: Literal["mcg", "mul1", "3inst"]
    sign_scale_transform: Recipe
    target: Recipe
    curvature_correction: Recipe
    viterbi_refinement: Recipe
    curvature: CurvatureCapture
    callback: CallbackContract
    runtime: RuntimeContract
    seed: int
    split_manifest_sha256: str
    split_selections: Mapping[str, Mapping[str, Any]]
    split_disjointness: SplitDisjointness
    source_tensor_sha256: str
    source_revision: str
    source_layout: Literal["out_in", "in_out"]
    evidence: Mapping[str, Any]
    qualifications: Mapping[str, str] = dataclasses.field(default_factory=dict)
    encoder_repo: str = STOCK_REPO
    encoder_commit: str = STOCK_COMMIT
    encoder_tree_sha1: str = STOCK_TREE_SHA1
    encoder_version: str = STOCK_VERSION
    schema: str = SCHEMA_ID
    serialized: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    metric_contract: Mapping[str, Any] = dataclasses.field(default_factory=lambda: KLD_METHOD_CONTRACT)

    def validate(self, *, allow_qualified_k7: bool = False) -> None:
        if self.schema != SCHEMA_ID:
            raise ContractError(f"schema must be {SCHEMA_ID}")
        if not self.action_id or not self.source_revision:
            raise ContractError("action_id and source_revision are required")
        self.unit.validate()
        self.sign_scale_transform.validate()
        self.target.validate()
        self.curvature_correction.validate()
        self.viterbi_refinement.validate()
        self.curvature.validate(self.unit)
        if self.source_layout not in COORDINATE_CONVENTIONS:
            raise ContractError("source_layout must be out_in or in_out")
        if self.curvature.coordinate_convention != COORDINATE_CONVENTIONS[self.source_layout]:
            raise ContractError("curvature coordinate convention does not match source_layout")
        self.callback.validate()
        k7_qualification = self.qualifications.get("k7_artifact_sha256", "")
        if self.K not in LEGAL_STOCK_K and not (
            self.K == 7
            and (allow_qualified_k7 or (
                len(k7_qualification) == 64
                and all(c in "0123456789abcdef" for c in k7_qualification)
            ))
        ):
            raise ContractError("stock actions are K4/K5/K6; K7 requires k7_artifact_sha256")
        self.runtime.validate(self.hashes.get("payload_sha256"))
        if self.codebook not in CODEBOOK_MARKERS:
            raise ContractError(f"unknown codebook {self.codebook}")
        if (
            self.encoder_repo != STOCK_REPO
            or self.encoder_commit != STOCK_COMMIT
            or self.encoder_tree_sha1 != STOCK_TREE_SHA1
            or self.encoder_version != STOCK_VERSION
        ):
            raise ContractError("action does not name the pinned stock encoder repo/commit/tree/version")
        validate_sha256("split_manifest_sha256", self.split_manifest_sha256)
        if set(self.split_selections) != set(SPLIT_NAMES):
            raise ContractError("split selections must name calibration, validation, and untouched_test")
        selection_hashes: set[str] = set()
        selector_values: set[str] = set()
        for name in SPLIT_NAMES:
            selection = self.split_selections[name]
            if not isinstance(selection, Mapping) or set(selection) != {"selection_sha256", "selector"}:
                raise ContractError(f"{name} selection must contain selection_sha256 and selector")
            validate_sha256(f"{name}.selection_sha256", selection["selection_sha256"])
            if selection["selector"] != SPLIT_SELECTOR_PREDICATES[name]:
                raise ContractError(f"{name} selector is not the frozen predicate")
            selection_hashes.add(selection["selection_sha256"])
            selector_values.add(selection["selector"]["value"])
        if len(selection_hashes) != len(SPLIT_NAMES) or len(selector_values) != len(SPLIT_NAMES):
            raise ContractError("split selections/selectors must be pairwise distinct")
        validate_sha256("source_tensor_sha256", self.source_tensor_sha256)
        self.split_disjointness.validate()
        for name, value in self.qualifications.items():
            validate_sha256(f"qualification {name}", value)
        if self.metric_contract != KLD_METHOD_CONTRACT:
            raise ContractError("metric_contract must use the immutable Wave 5 v5 KLD method")
        self._validate_stock_recipe_semantics()
        self._validate_evidence()

    def _validate_stock_recipe_semantics(self) -> None:
        callback_enabled = self.callback.identifier != "none"
        expected_implementation = "stock-with-encode-callback" if callback_enabled else "stock"
        recipes = (
            self.sign_scale_transform,
            self.target,
            self.curvature_correction,
            self.viterbi_refinement,
        )
        if any(recipe.implementation != expected_implementation for recipe in recipes):
            raise ContractError("recipe implementation does not match callback presence")
        sign = self.sign_scale_transform.parameters
        if (
            sign.get("apply_out_scales") is not True
            or sign.get("sign_streams") != "torch.randn.sign/fixed-seed"
            or sign.get("hadamard") != HADAMARD_BLOCK
            or sign.get("scales_dtype") != "float16"
            or sign.get("global_scale_search") != "stock_g_scale_gss"
        ):
            raise ContractError("sign/scale recipe does not bind the pinned stock semantics")
        if self.target.parameters != {"basis": "source"}:
            raise ContractError("target recipe must bind the source-basis stock target")
        curvature = self.curvature_correction.parameters
        if curvature != {"sigma_reg": 0.025, "block_ldl": LDL_BLOCK, "buf_size_k": 128}:
            raise ContractError("curvature recipe does not bind pinned damping/block-LDL parameters")
        viterbi = self.viterbi_refinement.parameters
        if viterbi != {
            "tile": [TILE, TILE],
            "tensor_core_permutation": True,
            "callback_return": "candidate_tiles_only",
            "viterbi_executor": "harness/pinned-stock",
        }:
            raise ContractError("Viterbi recipe does not bind target-only callbacks and pinned execution")

    def _validate_evidence(self) -> None:
        if set(self.evidence) != {"local_metrics", "promoted_kld"}:
            raise ContractError("evidence must contain exactly local_metrics and promoted_kld")
        local = self.evidence["local_metrics"]
        if not isinstance(local, Mapping):
            raise ContractError("local_metrics must be an object")
        if any("kld" in str(key).lower() for key in local):
            raise ContractError("local metrics cannot populate or impersonate the promoted KLD slot")
        promoted = self.evidence["promoted_kld"]
        if promoted is None:
            return
        if not isinstance(promoted, Mapping):
            raise ContractError("promoted_kld must be null or a closed evidence object")
        required = {
            "protocol_id", "suite_manifest_sha256", "suite_token_sha256",
            "shared_bf16_head_sha256", "reference_model_sha256",
            "candidate_model_sha256", "reference_capture_sha256",
            "candidate_capture_sha256", "report_sha256", "candidate_payload_sha256",
            "direction", "full_vocabulary", "promoted_split", "metrics",
            "fail_closed_lineage",
        }
        if set(promoted) != required:
            raise ContractError("promoted KLD evidence fields are missing or unknown")
        exact = {
            "protocol_id": KLD_METHOD_CONTRACT["protocol_id"],
            "suite_manifest_sha256": KLD_METHOD_CONTRACT["suite_manifest_sha256"],
            "suite_token_sha256": KLD_METHOD_CONTRACT["suite_token_sha256"],
            "shared_bf16_head_sha256": KLD_METHOD_CONTRACT["shared_bf16_head_sha256"],
            "direction": KLD_METHOD_CONTRACT["direction"],
            "full_vocabulary": True,
        }
        if any(promoted[name] != value for name, value in exact.items()):
            raise ContractError("promoted KLD method identity or direction mismatch")
        for name in (
            "reference_model_sha256", "candidate_model_sha256",
            "reference_capture_sha256", "candidate_capture_sha256",
            "report_sha256", "candidate_payload_sha256",
        ):
            validate_sha256(f"promoted_kld.{name}", promoted[name])
        if promoted["candidate_payload_sha256"] != self.hashes.get("payload_sha256"):
            raise ContractError("promoted KLD lineage does not name this action payload")
        split = promoted["promoted_split"]
        if not isinstance(split, Mapping) or set(split) != {
            "name", "split_manifest_sha256", "selection_sha256", "selector"
        }:
            raise ContractError("promoted KLD split identity is incomplete")
        split_name = split["name"]
        if split_name not in ("validation", "untouched_test"):
            raise ContractError("promoted KLD must use validation or untouched-test data")
        expected = self.split_selections[split_name]
        if (
            split["split_manifest_sha256"] != self.split_manifest_sha256
            or split["selection_sha256"] != expected["selection_sha256"]
            or split["selector"] != expected["selector"]
        ):
            raise ContractError("promoted KLD split does not match the action split identity")
        metrics = promoted["metrics"]
        required_metrics = set(KLD_METHOD_CONTRACT["required_metrics"])
        if not isinstance(metrics, Mapping) or set(metrics) != required_metrics:
            raise ContractError("promoted KLD metrics are incomplete")
        numeric_metrics = required_metrics - {"worst_contexts"}
        if any(
            not isinstance(metrics[name], (int, float))
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            or metrics[name] < 0
            for name in numeric_metrics
        ):
            raise ContractError("promoted KLD numeric metrics must be finite and nonnegative")
        if not 0 <= metrics["top1_agreement"] <= 1:
            raise ContractError("promoted KLD top1_agreement must be in [0,1]")
        if not isinstance(metrics["worst_contexts"], list):
            raise ContractError("promoted KLD worst_contexts must be an array")
        lineage = promoted["fail_closed_lineage"]
        if lineage != {"all_hashes_verified": True, "no_missing_parents": True}:
            raise ContractError("promoted KLD lineage is not fail-closed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = dataclasses.asdict(self)
        value["unit"]["tensor_keys"] = list(self.unit.tensor_keys)
        value["unit"]["output_splits"] = list(self.unit.output_splits)
        value["curvature"]["h_shape"] = list(self.curvature.h_shape)
        value["runtime"]["decode_hot_ops"] = list(self.runtime.decode_hot_ops)
        value["runtime"]["startup_ops"] = list(self.runtime.startup_ops)
        return value

    def identity_sha256(self) -> str:
        value = self.to_dict()
        value.pop("serialized", None)
        value.pop("hashes", None)
        return sha256_bytes(canonical_json(value))


def callback_contract(callbacks: "EncodeCallbacks | None") -> CallbackContract:
    identifier = "none" if callbacks is None else callbacks.identifier
    version = "1" if callbacks is None else callbacks.version
    parameters: Mapping[str, Any] = {} if callbacks is None else copy.deepcopy(callbacks.parameters)
    implementation_sha256, module_files_sha256 = _callback_executable_identity(callbacks)
    content_sha256 = sha256_bytes(canonical_json({
        "identifier": identifier,
        "version": version,
        "implementation_sha256": implementation_sha256,
        "module_files_sha256": module_files_sha256,
        "parameters": parameters,
        "expected_interface": CALLBACK_EXPECTED_INTERFACE,
    }))
    return CallbackContract(
        identifier=identifier,
        version=version,
        implementation_sha256=implementation_sha256,
        module_files_sha256=module_files_sha256,
        content_sha256=content_sha256,
        parameters=parameters,
        expected_interface=CALLBACK_EXPECTED_INTERFACE,
    )


def verify_callback_contract(declared: CallbackContract, callbacks: "EncodeCallbacks | None") -> None:
    actual = callback_contract(callbacks)
    if dataclasses.asdict(declared) != dataclasses.asdict(actual):
        raise ContractError("invoked callback does not match the action callback identity/hash/parameters/interface")


def _identity_target(weight: Any, action: EXL3Action) -> Any:
    return weight


def _identity_scale(state: tuple[Any, Any, float, Any, Any], action: EXL3Action) -> tuple[Any, Any, float, Any, Any]:
    return state


def _identity_curvature(H: Any, action: EXL3Action) -> Any:
    return H


def _identity_recurrence(L: Any, action: EXL3Action) -> Any:
    return L


def _identity_legal_path(tiles: Any, q: Any, idx: Any, action: EXL3Action) -> Any:
    return tiles


@dataclasses.dataclass(frozen=True)
class EncodeCallbacks:
    """Encode-stage hooks whose exact source and parameters are action-bound."""

    target: Callable[[Any, EXL3Action], Any] | None = None
    scale: Callable[[tuple[Any, Any, float, Any, Any], EXL3Action], tuple[Any, Any, float, Any, Any]] | None = None
    curvature: Callable[[Any, EXL3Action], Any] | None = None
    recurrence: Callable[[Any, EXL3Action], Any] | None = None
    legal_path: Callable[[Any, Any, Any, EXL3Action], Any] | None = None
    identifier: str = "none"
    version: str = "1"
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def identity(cls) -> "EncodeCallbacks":
        return cls(
            target=_identity_target,
            scale=_identity_scale,
            curvature=_identity_curvature,
            recurrence=_identity_recurrence,
            legal_path=_identity_legal_path,
            identifier="identity/strength-zero",
            version="1",
            parameters={"strength": 0.0},
        )



@dataclasses.dataclass
class EncodedPayload:
    tensors: dict[str, Any]
    action: EXL3Action
    source_shape: tuple[int, int]
    encoder_shape: tuple[int, int]
    source_layout: Literal["out_in", "in_out"]
    proxy_error: float
    encode_seconds: float

    def buffer_manifest(self) -> dict[str, Any]:
        items = {name: tensor_digest(tensor) for name, tensor in sorted(self.tensors.items())}
        total = sum(x["bytes"] for x in items.values())
        expected = stock_dense_buffer_bytes(
            self.encoder_shape[0] * self.encoder_shape[1],
            self.encoder_shape[0],
            self.encoder_shape[1],
            self.action.K,
            self.action.codebook,
        )
        return {
            "buffers": items,
            "buffer_bytes": total,
            "dense_byte_law": {
                "formula": "numel*K/8 + 2*(in_features+out_features) + marker_bytes",
                "marker_bytes": 0 if self.action.codebook == "3inst" else 4,
                "expected_bytes": expected,
                "passes": total == expected,
            },
        }

    def serialize(self, path: os.PathLike[str] | str) -> dict[str, Any]:
        from safetensors.torch import save_file

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": SCHEMA_ID,
            "action_id": self.action.action_id,
            "action_identity_sha256": self.action.identity_sha256(),
            "source_layout": self.source_layout,
        }
        save_file({k: v.detach().contiguous().cpu() for k, v in self.tensors.items()}, str(path), metadata=metadata)
        manifest = self.buffer_manifest()
        file_bytes = path.stat().st_size
        manifest.update({
            "standalone_safetensors_bytes": file_bytes,
            "container_overhead_bytes": file_bytes - manifest["buffer_bytes"],
            "safetensors_sha256": sha256_file(path),
            "alignment": "safetensors v1: 8-byte header length and 8-byte-padded JSON header; tensor offsets are contiguous",
        })
        return manifest


def serialized_encode(fn: Callable[..., EncodedPayload]) -> Callable[..., EncodedPayload]:
    @functools.wraps(fn)
    def locked(self: "StockEXL3", *args: Any, **kwargs: Any) -> EncodedPayload:
        with self._patch_lock:
            return fn(self, *args, **kwargs)
    return locked


class StockEXL3:
    """Pinned source adapter. Only this class imports the CUDA implementation."""

    _patch_lock = threading.RLock()

    def __init__(self):
        self.torch = importlib.import_module("torch")
        self.qlib = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")
        self.decoder_module = importlib.import_module("exllamav3.modules.quant.exl3")
        self.LinearEXL3 = self.decoder_module.LinearEXL3
        self.source_identity = self._source_identity()
        if self.source_identity["commit"] != STOCK_COMMIT:
            raise ContractError("loaded EXL3 checkout HEAD is not the publication pin")
        if self.source_identity["tree_sha1"] != STOCK_TREE_SHA1:
            raise ContractError("loaded EXL3 checkout tree is not the publication pin")
        if self.source_identity["worktree_clean"] is not True:
            raise ContractError("loaded EXL3 checkout has tracked modifications")
        if self.source_identity["version"] != STOCK_VERSION:
            raise ContractError("loaded EXL3 package version is not the publication pin")
        if self.source_identity["quantize_py_sha256"] != STOCK_QUANTIZE_SHA256:
            raise ContractError("quantizer source is not the publication pin")
        if self.source_identity["decoder_py_sha256"] != STOCK_DECODER_SHA256:
            raise ContractError("LinearEXL3 source is not the publication pin")
        if not all(row["matches"] for row in self.source_identity["python_dependencies"].values()):
            raise ContractError("EXL3 Python transform dependency closure is not the publication pin")
        if not all(row["matches"] for row in self.source_identity["extension_sources"].values()):
            raise ContractError("extension source closure is not the publication pin")
        if self.source_identity["extension_binary_sha256"] != STOCK_EXTENSION_BINARY_SHA256:
            raise ContractError("compiled EXL3 extension binary is not the qualified stock build")

    @staticmethod
    def _git(repo_root: pathlib.Path, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ContractError("cannot observe loaded EXL3 git identity") from exc
        return result.stdout.strip()

    def _source_identity(self) -> dict[str, Any]:
        package_root = pathlib.Path(self.qlib.__file__).resolve().parents[3]
        repo_root = pathlib.Path(self._git(package_root, "rev-parse", "--show-toplevel"))
        extension_binary = pathlib.Path(self.qlib.ext.__file__).resolve()
        identity: dict[str, Any] = {
            "repo": STOCK_REPO,
            "git_root": str(repo_root),
            "commit": self._git(repo_root, "rev-parse", "HEAD"),
            "tree_sha1": self._git(repo_root, "rev-parse", "HEAD^{tree}"),
            "worktree_clean": self._git(repo_root, "status", "--porcelain", "--untracked-files=no") == "",
            "version": importlib.import_module("exllamav3.version").__version__,
            "quantize_py": str(pathlib.Path(self.qlib.__file__).resolve()),
            "quantize_py_sha256": sha256_file(self.qlib.__file__),
            "decoder_py": str(pathlib.Path(self.decoder_module.__file__).resolve()),
            "decoder_py_sha256": sha256_file(self.decoder_module.__file__),
            "extension_binary": str(extension_binary),
            "extension_binary_sha256": sha256_file(extension_binary),
            "extension_binary_expected_sha256": STOCK_EXTENSION_BINARY_SHA256,
            "python_dependencies": {},
            "extension_sources": {},
        }
        for relative, expected in STOCK_PYTHON_DEPENDENCY_SHA256.items():
            path = package_root / relative
            actual = sha256_file(path)
            identity["python_dependencies"][relative] = {
                "sha256": actual,
                "expected_sha256": expected,
                "matches": actual == expected,
            }
        for relative, expected in STOCK_EXTENSION_SOURCE_SHA256.items():
            path = package_root / relative
            actual = sha256_file(path)
            identity["extension_sources"][relative] = {
                "sha256": actual,
                "expected_sha256": expected,
                "matches": actual == expected,
            }
        return identity

    @staticmethod
    def h_data(H: Any, *, key: str, device: Any, count: int = 1) -> dict[str, Any]:
        torch = __import__("torch")
        if H.dtype != torch.float32 or H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ContractError("H must be square float32")
        return {
            "H": H,
            "first_key": key,
            "count": count,
            "finalized": False,
            "num_total": count * H.shape[0],
            "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
            "device": device,
        }

    def quant_args(self, action: EXL3Action, device: Any) -> dict[str, Any]:
        args: dict[str, Any] = {
            "seed": action.seed,
            "K": action.K,
            "devices": [device],
            "device_ratios": None,
            "apply_out_scales": action.sign_scale_transform.parameters.get("apply_out_scales", True),
            "sigma_reg": action.curvature_correction.parameters.get("sigma_reg", 0.025),
            "buf_size_k": action.curvature_correction.parameters.get("buf_size_k", 128),
        }
        marker = CODEBOOK_MARKERS[action.codebook]
        if marker:
            args[marker] = True
        return args

    def _validate_weight(self, weight: Any) -> None:
        torch = self.torch
        if weight.dtype != torch.float32 or weight.ndim != 2 or not weight.is_contiguous():
            raise ContractError("encoder weight must be contiguous float32 (in_features,out_features)")
        if weight.shape[0] % HADAMARD_BLOCK or weight.shape[1] % HADAMARD_BLOCK:
            raise ContractError("stock dense EXL3 requires both dimensions H128-aligned")
        if not torch.isfinite(weight).all().item():
            raise ContractError("encoder weight contains non-finite values")

    @contextlib.contextmanager
    def _hooks(self, action: EXL3Action, callbacks: EncodeCallbacks | None):
        if callbacks is None:
            yield
            return
        qlib = self.qlib
        originals = {
            "regularize": qlib.regularize,
            "block_ldl": qlib.block_ldl,
            "quantize_tiles_multigpu": qlib.quantize_tiles_multigpu,
        }

        def regularize_hook(*args: Any, **kwargs: Any):
            state = originals["regularize"](*args, **kwargs)
            if callbacks.scale is None:
                return state
            changed = callbacks.scale(state, action)
            if not isinstance(changed, tuple) or len(changed) != 5:
                raise ContractError("scale callback must return stock five-tuple")
            apply, weight_r, g_scale, su, sv = changed
            if not isinstance(apply, bool):
                raise ContractError("scale callback changed stock apply_out_scales type")
            tensors = ((weight_r, state[1]), (su, state[3]), (sv, state[4]))
            if any(
                changed_tensor.shape != original.shape
                or changed_tensor.dtype != original.dtype
                or changed_tensor.device != original.device
                for changed_tensor, original in tensors
            ):
                raise ContractError("scale callback changed stock buffer shape/dtype/device")
            if not all(self.torch.isfinite(x).all().item() for x in (weight_r, su, sv)):
                raise ContractError("scale callback produced non-finite tensors")
            if not isinstance(g_scale, (int, float)) or isinstance(g_scale, bool) or not math.isfinite(float(g_scale)):
                raise ContractError("scale callback produced an invalid global scale")
            return apply, weight_r.contiguous(), float(g_scale), su.contiguous(), sv.contiguous()

        def block_ldl_hook(H: Any, b: int, quant_args: dict[str, Any], verbose: bool, debug_info: Any = None):
            if b != LDL_BLOCK:
                raise ContractError(f"stock control requires block_ldl(b={LDL_BLOCK}), got {b}")
            if callbacks.curvature is not None:
                H2 = callbacks.curvature(H, action)
                if (
                    H2.shape != H.shape
                    or H2.dtype != H.dtype
                    or H2.device != H.device
                    or not self.torch.isfinite(H2).all().item()
                ):
                    raise ContractError("curvature callback violated H shape/dtype/device/finiteness")
                H = H2.contiguous()
            L, H_after = originals["block_ldl"](H, b, quant_args, verbose, debug_info)
            if callbacks.recurrence is not None:
                L2 = callbacks.recurrence(L, action)
                if (
                    L2.shape != L.shape
                    or L2.dtype != L.dtype
                    or L2.device != L.device
                    or not self.torch.isfinite(L2).all().item()
                ):
                    raise ContractError("recurrence callback violated L shape/dtype/device/finiteness")
                L = L2.contiguous()
            return L, H_after

        def tiles_hook(tiles: Any, quant_args: dict[str, Any]):
            q, idx = originals["quantize_tiles_multigpu"](tiles, quant_args)
            if callbacks.legal_path is None:
                return q, idx
            candidate_tiles = callbacks.legal_path(tiles, q, idx, action)
            if (
                candidate_tiles.shape != tiles.shape
                or candidate_tiles.dtype != tiles.dtype
                or candidate_tiles.device != tiles.device
                or not self.torch.isfinite(candidate_tiles).all().item()
            ):
                raise ContractError("legal-path callback must return finite candidate tiles with stock shape/dtype")
            return originals["quantize_tiles_multigpu"](candidate_tiles.contiguous(), quant_args)

        with self._patch_lock:
            qlib.regularize = regularize_hook
            qlib.block_ldl = block_ldl_hook
            qlib.quantize_tiles_multigpu = tiles_hook
            try:
                yield
            finally:
                for name, fn in originals.items():
                    setattr(qlib, name, fn)

    @serialized_encode
    def encode(
        self,
        source_weight: Any,
        H: Any,
        action: EXL3Action,
        *,
        callbacks: EncodeCallbacks | None = None,
        device: Any = None,
        verbose: bool = False,
    ) -> EncodedPayload:
        action.validate()
        verify_callback_contract(action.callback, callbacks)
        callbacks = None if callbacks is None else dataclasses.replace(
            callbacks, parameters=copy.deepcopy(callbacks.parameters)
        )
        torch = self.torch
        device = torch.device("cuda:0") if device is None else torch.device(device)
        source_shape = tuple(source_weight.shape)
        actual_source_sha256 = sha256_bytes(_tensor_bytes(source_weight))
        if actual_source_sha256 != action.source_tensor_sha256:
            raise ContractError(
                f"source tensor hash {actual_source_sha256} does not match action {action.source_tensor_sha256}"
            )
        source_layout = action.source_layout
        if source_layout == "out_in":
            weight = source_weight.T.contiguous().to(device=device, dtype=torch.float32)
        else:
            weight = source_weight.contiguous().to(device=device, dtype=torch.float32)
        self._validate_weight(weight)
        if H.shape != (weight.shape[0], weight.shape[0]):
            raise ContractError(f"H shape {tuple(H.shape)} does not match in_features {weight.shape[0]}")
        action.curvature.verify_tensor(H)
        if callbacks and callbacks.target:
            weight2 = callbacks.target(weight, action)
            if (
                weight2.shape != weight.shape
                or weight2.dtype != weight.dtype
                or weight2.device != weight.device
            ):
                raise ContractError("target callback changed weight shape/dtype/device")
            weight = weight2.contiguous()
            self._validate_weight(weight)
        H_work = H.detach().clone().contiguous().to(device=device, dtype=torch.float32)
        h_data = self.h_data(
            H_work,
            key=action.curvature.tensor_boundary,
            device=device,
            count=action.curvature.observation_count,
        )
        quant_args = self.quant_args(action, device)
        start = time.perf_counter()
        with self._hooks(action, callbacks):
            weight_q, proxy_error, tensors = self.qlib.quantize_exl3(
                weight,
                h_data,
                quant_args,
                True,
                progress_str=None,
                verbose=verbose,
                swap_to_device=None,
            )
        elapsed = time.perf_counter() - start
        expected = {"suh", "svh", "trellis"}
        marker = CODEBOOK_MARKERS[action.codebook]
        if marker:
            expected.add(marker)
        if set(tensors) != expected:
            raise ContractError(f"stock payload keys {sorted(tensors)} != {sorted(expected)}")
        if not torch.isfinite(weight_q).all().item():
            raise ContractError("stock returned non-finite source-basis reconstruction")
        return EncodedPayload(
            tensors=tensors,
            action=action,
            source_shape=source_shape,
            encoder_shape=tuple(weight.shape),
            source_layout=source_layout,
            proxy_error=float(proxy_error),
            encode_seconds=elapsed,
        )

    def decode(self, payload: EncodedPayload) -> Any:
        t = payload.tensors
        linear = self.LinearEXL3(
            None,
            payload.encoder_shape[0],
            payload.encoder_shape[1],
            suh=t["suh"],
            svh=t["svh"],
            trellis=t["trellis"],
            mcg=t.get("mcg"),
            mul1=t.get("mul1"),
            key=payload.action.unit.tensor_keys[0],
        )
        encoder_basis = linear.get_weight_tensor()
        source_basis = encoder_basis.T.contiguous() if payload.source_layout == "out_in" else encoder_basis
        if tuple(source_basis.shape) != payload.source_shape:
            raise ContractError("decoded tensor did not return to source basis")
        if not self.torch.isfinite(source_basis).all().item():
            raise ContractError("decoded source-basis tensor is non-finite")
        return source_basis


def load_action(path: os.PathLike[str] | str) -> EXL3Action:
    raw = json.loads(pathlib.Path(path).read_text())
    unit = Unit(**{
        **raw["unit"],
        "tensor_keys": tuple(raw["unit"]["tensor_keys"]),
        "output_splits": tuple(raw["unit"].get("output_splits", [])),
    })
    runtime = RuntimeContract(**{
        **raw["runtime"],
        "decode_hot_ops": tuple(raw["runtime"]["decode_hot_ops"]),
        "startup_ops": tuple(raw["runtime"].get("startup_ops", [])),
    })
    action = EXL3Action(
        **{
            **raw,
            "unit": unit,
            "sign_scale_transform": Recipe(**raw["sign_scale_transform"]),
            "target": Recipe(**raw["target"]),
            "curvature_correction": Recipe(**raw["curvature_correction"]),
            "viterbi_refinement": Recipe(**raw["viterbi_refinement"]),
            "curvature": CurvatureCapture(**{
                **raw["curvature"],
                "h_shape": tuple(raw["curvature"]["h_shape"]),
            }),
            "callback": CallbackContract(**raw["callback"]),
            "split_disjointness": SplitDisjointness(**raw["split_disjointness"]),
            "runtime": runtime,
        }
    )
    action.validate()
    return action


def make_stock_action(
    *,
    action_id: str,
    unit: Unit,
    K: int,
    codebook: str,
    seed: int,
    source_sha256: str,
    source_revision: str,
    source_layout: Literal["out_in", "in_out"],
    curvature: CurvatureCapture,
    callbacks: EncodeCallbacks | None,
    split_manifest_sha256: str,
    split_selections: Mapping[str, Mapping[str, Any]],
    split_disjointness: SplitDisjointness,
    evidence: Mapping[str, Any],
    route_id: str = "codec-exact/all-trellis-stock-exl3",
    materialization_qualification: Mapping[str, Any] | None = None,
) -> EXL3Action:
    declared_callback = callback_contract(callbacks)
    callback_enabled = declared_callback.identifier != "none"
    implementation = "stock-with-encode-callback" if callback_enabled else "stock"
    action = EXL3Action(
        action_id=action_id,
        unit=unit,
        K=K,
        codebook=codebook,  # type: ignore[arg-type]
        sign_scale_transform=Recipe(
            "stock.sign-scale-h128-gss",
            "stock_sign_scale_transform",
            {
                "apply_out_scales": True,
                "sign_streams": "torch.randn.sign/fixed-seed",
                "hadamard": HADAMARD_BLOCK,
                "scales_dtype": "float16",
                "global_scale_search": "stock_g_scale_gss",
            },
            implementation=implementation,
            strength=0.0 if declared_callback.identifier == "identity/strength-zero" else None,
        ),
        target=Recipe(
            "stock.source-target",
            "source_basis_target",
            {"basis": "source"},
            implementation=implementation,
        ),
        curvature_correction=Recipe(
            "stock.block-ldl16",
            "stock_curvature_recurrence",
            {"sigma_reg": 0.025, "block_ldl": LDL_BLOCK, "buf_size_k": 128},
            implementation=implementation,
        ),
        viterbi_refinement=Recipe(
            "stock.quantize-tiles-viterbi",
            "stock_legal_path",
            {
                "tile": [TILE, TILE],
                "tensor_core_permutation": True,
                "callback_return": "candidate_tiles_only",
                "viterbi_executor": "harness/pinned-stock",
            },
            implementation=implementation,
        ),
        curvature=curvature,
        callback=declared_callback,
        runtime=RuntimeContract(
            route_id=route_id,  # type: ignore[arg-type]
            decode_hot_ops=("H128 input", "stock trellis GEMM", "H128/output scale"),
            startup_ops=("load fixed-stride trellis",),
            materialization_qualification=materialization_qualification,
        ),
        seed=seed,
        split_manifest_sha256=split_manifest_sha256,
        split_selections=split_selections,
        split_disjointness=split_disjointness,
        source_tensor_sha256=source_sha256,
        source_revision=source_revision,
        source_layout=source_layout,
        evidence=evidence,
    )
    action.validate()
    return action


def load_safetensor(path: str, key: str) -> Any:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise ContractError(f"{key} not found in {path}")
        return f.get_tensor(key)

def make_curvature_capture(
    H: Any,
    unit: Unit,
    *,
    capture_id: str,
    observation_count: int,
    normalization: str,
    basis: str,
    coordinate_convention: str,
) -> CurvatureCapture:
    h_sha256 = sha256_bytes(_tensor_bytes(H))
    manifest = {
        "schema": "wave5/curvature-capture/1",
        "capture_id": capture_id,
        "h_tensor_sha256": h_sha256,
        "h_dtype": str(H.dtype).removeprefix("torch."),
        "h_shape": list(H.shape),
        "observation_count": observation_count,
        "normalization": normalization,
        "basis": basis,
        "coordinate_convention": coordinate_convention,
        "tensor_boundary": unit.tensor_keys[0],
        "module_boundary": unit.unit_id,
    }
    capture = CurvatureCapture(
        capture_id=capture_id,
        h_tensor_sha256=h_sha256,
        h_dtype=str(H.dtype).removeprefix("torch."),  # type: ignore[arg-type]
        h_shape=tuple(H.shape),
        capture_manifest_sha256=sha256_bytes(canonical_json(manifest)),
        observation_count=observation_count,
        normalization=normalization,
        basis=basis,
        coordinate_convention=coordinate_convention,
        tensor_boundary=unit.tensor_keys[0],
        module_boundary=unit.unit_id,
    )
    capture.validate(unit)
    return capture


def smoke_split_contract() -> dict[str, Any]:
    split_selections: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        selector = SPLIT_SELECTOR_PREDICATES[name]
        selection_body = {
            "schema": "wave5/split-selection/1",
            "scope": "codec-smoke/no-fidelity-claim",
            "name": name,
            "selector": selector,
        }
        split_selections[name] = {
            "selection_sha256": sha256_bytes(canonical_json(selection_body)),
            "selector": selector,
        }
    split_manifest_body = {
        "schema": "wave5/split-manifest/1",
        "scope": "codec-smoke/no-fidelity-claim",
        "selections": split_selections,
    }
    split_manifest_sha256 = sha256_bytes(canonical_json(split_manifest_body))
    audit_body = {
        "schema": "wave5/split-disjointness/1",
        "split_manifest_sha256": split_manifest_sha256,
        "split_selections": split_selections,
        "pairwise_overlap_counts": {
            "calibration__validation": 0,
            "calibration__untouched_test": 0,
            "validation__untouched_test": 0,
        },
        "source_document_overlap_count": 0,
        "domain_leakage_count": 0,
        "scope": "synthetic codec smoke partitions only",
    }
    return {
        "split_manifest_sha256": split_manifest_sha256,
        "split_selections": split_selections,
        "split_disjointness": SplitDisjointness(
            artifact_sha256=sha256_bytes(canonical_json(audit_body)),
            predicate_language="wave5.split-predicate/1",
            pairwise_overlap_counts=audit_body["pairwise_overlap_counts"],
            source_document_overlap_count=0,
            domain_leakage_count=0,
            verified=True,
        ),
        "evidence": {"local_metrics": {}, "promoted_kld": None},
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if not args.container_image:
        raise ContractError("container image is required")
    if not args.container_image_digest.startswith("sha256:"):
        raise ContractError("container image digest must use sha256:<digest>")
    validate_sha256("container image digest", args.container_image_digest.removeprefix("sha256:"))
    action_schema_sha256 = sha256_file(args.action_schema)
    torch = importlib.import_module("torch")
    torch.set_grad_enabled(False)
    stock = StockEXL3()
    source = load_safetensor(args.source, args.key)
    source_hash = sha256_bytes(_tensor_bytes(source))
    if source.ndim != 2:
        raise ContractError("representative source must be a dense 2D tensor")
    unit = Unit(
        unit_id="qwen38.layer3.full_attention.k_proj",
        granularity="tensor",
        topology="full_attention",
        role="k_proj",
        tensor_keys=(args.key,),
        layer_index=3,
    )
    in_features = source.shape[1]
    H = torch.eye(in_features, dtype=torch.float32, device=args.device)
    curvature = make_curvature_capture(
        H,
        unit,
        capture_id="codec-smoke/identity-H/full-tensor",
        observation_count=1,
        normalization="identity/no-sample-normalization",
        basis="source-model input-feature activation basis",
        coordinate_convention=COORDINATE_CONVENTIONS["out_in"],
    )
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_manifest_args = smoke_split_contract()
    identity_callbacks = EncodeCallbacks.identity()

    base = make_stock_action(
        action_id=f"A0.stock.K{args.K}.{args.codebook}",
        unit=unit,
        K=args.K,
        codebook=args.codebook,
        seed=args.seed,
        source_sha256=source_hash,
        source_revision=args.source_revision,
        source_layout="out_in",
        curvature=curvature,
        callbacks=None,
        **smoke_manifest_args,
    )
    direct = stock.encode(source, H, base, callbacks=None, device=args.device)
    direct_manifest = direct.serialize(out_dir / "a0-stock.safetensors")
    direct_recon = stock.decode(direct)

    identity_action = make_stock_action(
        action_id=f"A0.identity-callback.K{args.K}.{args.codebook}",
        unit=unit,
        K=args.K,
        codebook=args.codebook,
        seed=args.seed,
        source_sha256=source_hash,
        source_revision=args.source_revision,
        source_layout="out_in",
        curvature=curvature,
        callbacks=identity_callbacks,
        **smoke_manifest_args,
    )
    identity = stock.encode(
        source,
        H,
        identity_action,
        callbacks=identity_callbacks,
        device=args.device,
    )
    identity_manifest = identity.serialize(out_dir / "a0-identity.safetensors")
    identity_recon = stock.decode(identity)
    payload_keys = sorted(direct.tensors)
    identity_equal = all(torch.equal(direct.tensors[k], identity.tensors[k]) for k in payload_keys)
    direct_payload_sha = payload_digest(direct.tensors)
    identity_payload_sha = payload_digest(identity.tensors)
    direct_recon_sha = sha256_bytes(_tensor_bytes(direct_recon))
    identity_recon_sha = sha256_bytes(_tensor_bytes(identity_recon))
    base_final = dataclasses.replace(
        base,
        serialized=direct_manifest,
        hashes={
            "action_identity_sha256": base.identity_sha256(),
            "payload_sha256": direct_payload_sha,
            "source_basis_reconstruction_sha256": direct_recon_sha,
        },
    )
    identity_final = dataclasses.replace(
        identity_action,
        serialized=identity_manifest,
        hashes={
            "action_identity_sha256": identity_action.identity_sha256(),
            "payload_sha256": identity_payload_sha,
            "source_basis_reconstruction_sha256": identity_recon_sha,
        },
    )

    panel_source = source[:128, :128].contiguous()
    panel_hash = sha256_bytes(_tensor_bytes(panel_source))
    panel_unit = dataclasses.replace(unit, unit_id=unit.unit_id + ".panel128", tensor_keys=(args.key + "[0:128,0:128]",))
    panel_H = torch.eye(128, dtype=torch.float32, device=args.device)
    panel_curvature = make_curvature_capture(
        panel_H,
        panel_unit,
        capture_id="codec-smoke/identity-H/panel128",
        observation_count=1,
        normalization="identity/no-sample-normalization",
        basis="source-model input-feature activation basis",
        coordinate_convention=COORDINATE_CONVENTIONS["out_in"],
    )
    k_controls: dict[str, Any] = {}
    for K in LEGAL_STOCK_K:
        a = make_stock_action(
            action_id=f"stock-panel.K{K}.{args.codebook}", unit=panel_unit, K=K,
            codebook=args.codebook, seed=args.seed, source_sha256=panel_hash,
            source_revision=args.source_revision,
            source_layout="out_in",
            curvature=panel_curvature, callbacks=None,
            **smoke_manifest_args,
        )
        p = stock.encode(panel_source, panel_H, a, device=args.device)
        r = stock.decode(p)
        k_controls[f"K{K}"] = {
            "buffer_manifest": p.buffer_manifest(),
            "payload_sha256": payload_digest(p.tensors),
            "finite_reconstruction": bool(torch.isfinite(r).all().item()),
            "source_basis_shape": list(r.shape),
            "proxy_error": p.proxy_error,
        }

    codebook_controls: dict[str, Any] = {}
    for codebook in ("mcg", "mul1"):
        a = make_stock_action(
            action_id=f"codebook-panel.K{args.K}.{codebook}", unit=panel_unit, K=args.K,
            codebook=codebook, seed=args.seed, source_sha256=panel_hash,
            source_revision=args.source_revision,
            source_layout="out_in",
            curvature=panel_curvature, callbacks=None,
            **smoke_manifest_args,
        )
        p = stock.encode(panel_source, panel_H, a, device=args.device)
        r = stock.decode(p)
        codebook_controls[codebook] = {
            "buffer_manifest": p.buffer_manifest(),
            "trellis_sha256": tensor_digest(p.tensors["trellis"])["sha256"],
            "marker_sha256": tensor_digest(p.tensors[codebook])["sha256"],
            "finite_reconstruction": bool(torch.isfinite(r).all().item()),
            "proxy_error": p.proxy_error,
        }

    receipt: dict[str, Any] = {
        "schema": "wave5/stock-control/1",
        "status": "pass" if identity_equal and direct_manifest["dense_byte_law"]["passes"] else "fail",
        "encoder": stock.source_identity,
        "metric_contract": KLD_METHOD_CONTRACT,
        "extension": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(torch.device(args.device)),
            "device_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        },
        "execution_container": {
            "image": args.container_image,
            "image_digest": args.container_image_digest,
            "harness_sha256": sha256_file(__file__),
            "action_schema_sha256": action_schema_sha256,
        },
        "source": {
            "path": args.source,
            "key": args.key,
            "revision": args.source_revision,
            "dtype": str(source.dtype).removeprefix("torch."),
            "shape": list(source.shape),
            "sha256": source_hash,
            "bf16_decode": "safetensors torch.bfloat16 tensor converted numerically to float32; never reinterpreted as float16",
        },
        "curvature_control": {
            **dataclasses.asdict(curvature),
            "kind": "identity screening H_X",
            "shape": [in_features, in_features],
            "claim_scope": "codec/serialization smoke only; not fidelity evidence",
            "stock_pipeline": "sigma_reg then sign/H128 transform then stock block_ldl(b=16)",
        },
        "split_control": {
            "split_manifest_sha256": smoke_manifest_args["split_manifest_sha256"],
            "split_selections": smoke_manifest_args["split_selections"],
            "disjointness": dataclasses.asdict(smoke_manifest_args["split_disjointness"]),
            "claim_scope": "synthetic codec-smoke partitions; not promoted fidelity data",
        },
        "a0_full_tensor": {
            "action": base_final.to_dict(),
            "action_identity_sha256": base.identity_sha256(),
            "payload": direct_manifest,
            "encode_seconds": direct.encode_seconds,
            "proxy_error": direct.proxy_error,
            "decode_api": "LinearEXL3.get_weight_tensor",
            "finite_reconstruction": bool(torch.isfinite(direct_recon).all().item()),
            "source_basis_shape": list(direct_recon.shape),
            "reconstruction_sha256": direct_recon_sha,
        },
        "strength_zero_identity": {
            "action": identity_final.to_dict(),
            "byte_identical_buffers": identity_equal,
            "direct_payload_sha256": direct_payload_sha,
            "identity_payload_sha256": identity_payload_sha,
            "standalone_file_identical": direct_manifest["safetensors_sha256"] == identity_manifest["safetensors_sha256"],
            "standalone_file_difference_explanation": "files carry different action_id/action_identity metadata; tensor buffers are the identity criterion",
            "finite_reconstruction": bool(torch.isfinite(identity_recon).all().item()),
            "same_hot_schema": sorted(direct.tensors) == sorted(identity.tensors),
            "same_decoder_route": True,
        },
        "fresh_stock_panel": k_controls,
        "codebook_one_factor": {
            "qualified_K": args.K,
            "held_fixed": ["source panel", "H_X", "seed", "scales/sign/H128/LDL/Viterbi budgets", "K", "raw byte count"],
            "changed": "MCG vs MUL1 marker/codebook transition function",
            "controls": codebook_controls,
            "equal_buffer_bytes": codebook_controls["mcg"]["buffer_manifest"]["buffer_bytes"] == codebook_controls["mul1"]["buffer_manifest"]["buffer_bytes"],
            "different_trellis": codebook_controls["mcg"]["trellis_sha256"] != codebook_controls["mul1"]["trellis_sha256"],
        },
        "route_registry": ROUTE_REGISTRY,
        "topology_contract": TOPOLOGY_CONTRACT,
        "determinism": {
            "seed": args.seed,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "criterion": "canonical tensor buffers, not standalone container metadata",
        },
    }
    pathlib.Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def validate_schema(args: argparse.Namespace) -> None:
    schema = json.loads(pathlib.Path(args.schema).read_text())
    action = load_action(args.action)
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for validate-schema") from exc
    jsonschema.validate(action.to_dict(), schema)
    print(action.identity_sha256())


def self_test() -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for the contract self-test") from exc

    def expect_contract_error(fn: Callable[[], Any]) -> None:
        try:
            fn()
        except ContractError:
            return
        raise AssertionError("expected ContractError")

    def expect_schema_error(value: Mapping[str, Any]) -> None:
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError:
            return
        raise AssertionError("expected jsonschema.ValidationError")

    assert stock_dense_buffer_bytes(5120 * 1024, 5120, 1024, 5, "mcg") == 3_289_092
    assert stock_dense_buffer_bytes(128 * 128, 128, 128, 4, "3inst") == 8_704
    for route in ROUTE_REGISTRY.values():
        assert route["checkpoint_schema"] == ["suh", "svh", "trellis", "mcg|mul1|none"]

    unit = Unit(
        unit_id="self-test.module",
        granularity="tensor",
        topology="full_attention",
        role="k_proj",
        tensor_keys=("model.test.weight",),
        layer_index=0,
    )
    curvature_unhashed = CurvatureCapture(
        capture_id="self-test/capture",
        h_tensor_sha256="1" * 64,
        h_dtype="float32",
        h_shape=(128, 128),
        capture_manifest_sha256="2" * 64,
        observation_count=32,
        normalization="sum_xxt/observation_count",
        basis="source input-feature activation basis",
        coordinate_convention=COORDINATE_CONVENTIONS["out_in"],
        tensor_boundary=unit.tensor_keys[0],
        module_boundary=unit.unit_id,
    )
    curvature = dataclasses.replace(
        curvature_unhashed,
        capture_manifest_sha256=sha256_bytes(canonical_json(curvature_unhashed.manifest_payload())),
    )
    split_args = smoke_split_contract()
    action = make_stock_action(
        action_id="self-test.stock",
        unit=unit,
        K=5,
        codebook="mcg",
        seed=1,
        source_sha256="3" * 64,
        source_revision="self-test-revision",
        source_layout="out_in",
        curvature=curvature,
        callbacks=None,
        **split_args,
    )
    schema = json.loads(pathlib.Path(__file__).with_name("exl3_action_schema.json").read_text())
    jsonschema.validate(action.to_dict(), schema)

    for field, invalid in {
        "capture_id": "",
        "h_tensor_sha256": "0" * 64,
        "h_dtype": "float16",
        "h_shape": (128, 64),
        "capture_manifest_sha256": SHA256_EMPTY,
        "observation_count": 0,
        "normalization": "",
        "basis": "",
        "coordinate_convention": "",
        "tensor_boundary": "other.weight",
        "module_boundary": "other.module",
    }.items():
        expect_contract_error(lambda field=field, invalid=invalid: dataclasses.replace(
            action, curvature=dataclasses.replace(curvature, **{field: invalid})
        ).validate())
    for field, stale in {
        "normalization": "sum_xxt",
        "basis": "transformed basis",
        "coordinate_convention": COORDINATE_CONVENTIONS["in_out"],
    }.items():
        expect_contract_error(lambda field=field, stale=stale: dataclasses.replace(
            action, curvature=dataclasses.replace(curvature, **{field: stale})
        ).validate())
    expect_contract_error(lambda: dataclasses.replace(action, source_layout="in_out").validate())
    expect_contract_error(lambda: curvature.verify_sha256("4" * 64))
    for field, invalid in {
        "encoder_repo": "other/repo",
        "encoder_commit": "0" * 40,
        "encoder_tree_sha1": "0" * 40,
        "encoder_version": "not-stock",
    }.items():
        expect_contract_error(lambda field=field, invalid=invalid: dataclasses.replace(
            action, **{field: invalid}
        ).validate())

    identity_callbacks = EncodeCallbacks.identity()
    identity_contract = callback_contract(identity_callbacks)
    identity_contract.validate()
    for field, invalid in {
        "identifier": "",
        "version": "",
        "implementation_sha256": "0" * 64,
        "content_sha256": "5" * 64,
        "module_files_sha256": {"fake:callback.py": "0" * 64},
        "expected_interface": {},
    }.items():
        expect_contract_error(lambda field=field, invalid=invalid: dataclasses.replace(
            identity_contract, **{field: invalid}
        ).validate())
    expect_contract_error(lambda: verify_callback_contract(action.callback, identity_callbacks))
    identity_action = make_stock_action(
        action_id="self-test.identity",
        unit=unit,
        K=5,
        codebook="mcg",
        seed=1,
        source_sha256="3" * 64,
        source_revision="self-test-revision",
        source_layout="out_in",
        curvature=curvature,
        callbacks=identity_callbacks,
        **split_args,
    )
    verify_callback_contract(identity_action.callback, identity_callbacks)
    expect_contract_error(lambda: verify_callback_contract(identity_action.callback, None))
    expect_contract_error(lambda: dataclasses.replace(
        action,
        viterbi_refinement=dataclasses.replace(
            action.viterbi_refinement,
            parameters={"tile": [16, 16], "tensor_core_permutation": True},
        ),
    ).validate())

    class LockProbe:
        _patch_lock = threading.RLock()

        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0

        @serialized_encode
        def run(self) -> Any:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            time.sleep(0.01)
            self.active -= 1

    probe = LockProbe()
    threads = [threading.Thread(target=probe.run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert probe.maximum == 1

    expect_contract_error(lambda: dataclasses.replace(action, split_manifest_sha256="0" * 64).validate())
    duplicate_selection = copy.deepcopy(action.split_selections)
    duplicate_selection["validation"]["selection_sha256"] = duplicate_selection["calibration"]["selection_sha256"]
    expect_contract_error(lambda: dataclasses.replace(action, split_selections=duplicate_selection).validate())
    overlapping = copy.deepcopy(action.split_selections)
    overlapping["validation"]["selector"]["value"] = "calibration"
    expect_contract_error(lambda: dataclasses.replace(action, split_selections=overlapping).validate())
    expect_contract_error(lambda: dataclasses.replace(action, split_selections={}).validate())
    expect_contract_error(lambda: dataclasses.replace(
        action,
        split_disjointness=dataclasses.replace(
            action.split_disjointness,
            pairwise_overlap_counts={
                **action.split_disjointness.pairwise_overlap_counts,
                "calibration__validation": 1,
            },
        ),
    ).validate())
    expect_contract_error(lambda: dataclasses.replace(
        action,
        split_disjointness=dataclasses.replace(action.split_disjointness, domain_leakage_count=1),
    ).validate())

    expect_contract_error(lambda: dataclasses.replace(
        action, evidence={"local_metrics": {"context_macro_mean_kld": 0.1}, "promoted_kld": None}
    ).validate())
    payload_sha = "4" * 64
    promoted = {
        "protocol_id": KLD_METHOD_CONTRACT["protocol_id"],
        "suite_manifest_sha256": KLD_METHOD_CONTRACT["suite_manifest_sha256"],
        "suite_token_sha256": KLD_METHOD_CONTRACT["suite_token_sha256"],
        "shared_bf16_head_sha256": KLD_METHOD_CONTRACT["shared_bf16_head_sha256"],
        "reference_model_sha256": "5" * 64,
        "candidate_model_sha256": "6" * 64,
        "reference_capture_sha256": "7" * 64,
        "candidate_capture_sha256": "8" * 64,
        "report_sha256": "9" * 64,
        "candidate_payload_sha256": payload_sha,
        "direction": KLD_METHOD_CONTRACT["direction"],
        "full_vocabulary": True,
        "promoted_split": {
            "name": "validation",
            "split_manifest_sha256": action.split_manifest_sha256,
            "selection_sha256": action.split_selections["validation"]["selection_sha256"],
            "selector": action.split_selections["validation"]["selector"],
        },
        "metrics": {
            "token_mean_kld": 0.01,
            "context_macro_mean_kld": 0.01,
            "p99": 0.02,
            "cvar1pct": 0.03,
            "top1_agreement": 0.99,
            "full_vocab_ear": 0.001,
            "worst_contexts": [],
        },
        "fail_closed_lineage": {"all_hashes_verified": True, "no_missing_parents": True},
    }
    promoted_action = dataclasses.replace(
        action,
        hashes={"payload_sha256": payload_sha},
        evidence={"local_metrics": {}, "promoted_kld": promoted},
    )
    promoted_action.validate()
    jsonschema.validate(promoted_action.to_dict(), schema)
    for field in promoted:
        broken = copy.deepcopy(promoted)
        del broken[field]
        expect_contract_error(lambda broken=broken: dataclasses.replace(
            action,
            hashes={"payload_sha256": payload_sha},
            evidence={"local_metrics": {}, "promoted_kld": broken},
        ).validate())
    wrong_direction = copy.deepcopy(promoted)
    wrong_direction["direction"] = "KL(candidate || BF16 reference)"
    expect_contract_error(lambda: dataclasses.replace(
        action,
        hashes={"payload_sha256": payload_sha},
        evidence={"local_metrics": {}, "promoted_kld": wrong_direction},
    ).validate())
    wrong_split = copy.deepcopy(promoted)
    wrong_split["promoted_split"]["selection_sha256"] = action.split_selections["untouched_test"]["selection_sha256"]
    expect_contract_error(lambda: dataclasses.replace(
        action,
        hashes={"payload_sha256": payload_sha},
        evidence={"local_metrics": {}, "promoted_kld": wrong_split},
    ).validate())

    effective_env = dict(PRODUCTION_EFFECTIVE_ENV)
    image_sha = "a" * 64
    old_env = {name: os.environ.get(name) for name in (*REQUIRED_ROUTING_ENV, "WAVE5_CONTAINER_IMAGE_DIGEST")}
    production_dict: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="wave5-materialization-") as tmp:
        tmp_path = pathlib.Path(tmp)
        route_path = pathlib.Path(__file__).resolve().parents[3] / "patches/vllm-exl3-multiprecision.py"
        artifacts = {
            "fp4_converter_source": b"fp4 source",
            "fp4_converter_binary": b"fp4 binary",
            "fp6_converter_source": b"fp6 source",
            "fp6_converter_binary": b"fp6 binary",
            "source_payload": b"source trellis payload",
            "materialized_tensor": b"materialized tensor",
            "materialized_payload": b"materialized payload",
            "runtime_receipt": b"runtime receipt",
        }
        paths: dict[str, pathlib.Path] = {}
        for name, content in artifacts.items():
            path = tmp_path / name
            path.write_bytes(content)
            paths[name] = path
        source_payload_sha = sha256_file(paths["source_payload"])
        materialization = {
            "route_id": "production/throughput-fp4-fp6-materialized",
            "route_implementation_path": str(route_path),
            "route_implementation_sha256": PRODUCTION_ROUTE_SHA256,
            "image_digest_sha256": image_sha,
            "fp4_converter_source_path": str(paths["fp4_converter_source"]),
            "fp4_converter_source_sha256": sha256_file(paths["fp4_converter_source"]),
            "fp4_converter_binary_path": str(paths["fp4_converter_binary"]),
            "fp4_converter_binary_sha256": sha256_file(paths["fp4_converter_binary"]),
            "fp6_converter_source_path": str(paths["fp6_converter_source"]),
            "fp6_converter_source_sha256": sha256_file(paths["fp6_converter_source"]),
            "fp6_converter_binary_path": str(paths["fp6_converter_binary"]),
            "fp6_converter_binary_sha256": sha256_file(paths["fp6_converter_binary"]),
            "effective_environment": effective_env,
            "effective_environment_sha256": sha256_bytes(canonical_json(effective_env)),
            "source_payload_path": str(paths["source_payload"]),
            "source_payload_sha256": source_payload_sha,
            "materialized_tensor_path": str(paths["materialized_tensor"]),
            "materialized_tensor_sha256": sha256_file(paths["materialized_tensor"]),
            "materialized_payload_path": str(paths["materialized_payload"]),
            "materialized_payload_sha256": sha256_file(paths["materialized_payload"]),
            "runtime_receipt_path": str(paths["runtime_receipt"]),
            "runtime_receipt_sha256": sha256_file(paths["runtime_receipt"]),
        }
        production_runtime = RuntimeContract(
            route_id="production/throughput-fp4-fp6-materialized",
            decode_hot_ops=("materialized GEMM",),
            materialization_qualification=materialization,
        )
        for name, value in effective_env.items():
            os.environ[name] = value
        os.environ["WAVE5_CONTAINER_IMAGE_DIGEST"] = f"sha256:{image_sha}"
        production_action = dataclasses.replace(
            action,
            hashes={"payload_sha256": source_payload_sha},
            runtime=production_runtime,
        )
        production_action.validate()
        production_dict = production_action.to_dict()
        jsonschema.validate(production_dict, schema)
        for field in materialization:
            broken = dict(materialization)
            del broken[field]
            expect_contract_error(lambda broken=broken: dataclasses.replace(
                production_action,
                runtime=dataclasses.replace(production_runtime, materialization_qualification=broken),
            ).validate())
        mismatched_payload = dict(materialization)
        mismatched_payload["source_payload_sha256"] = sha256_bytes(b"other source payload")
        expect_contract_error(lambda: dataclasses.replace(
            production_action,
            runtime=dataclasses.replace(
                production_runtime, materialization_qualification=mismatched_payload
            ),
        ).validate())
        paths["fp4_converter_binary"].write_bytes(b"mutated binary")
        expect_contract_error(production_action.validate)
        paths["fp4_converter_binary"].write_bytes(artifacts["fp4_converter_binary"])
        stale = copy.deepcopy(materialization)
        stale["effective_environment"]["VLLM_EXL3_B12X_MIN_M"] = "128"
        expect_contract_error(lambda: dataclasses.replace(
            production_action,
            runtime=dataclasses.replace(production_runtime, materialization_qualification=stale),
        ).validate())
        os.environ["WAVE5_CONTAINER_IMAGE_DIGEST"] = f"sha256:{'b' * 64}"
        expect_contract_error(production_action.validate)
    for name, value in old_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    action_dict = action.to_dict()
    for field in schema["required"]:
        broken = copy.deepcopy(action_dict)
        del broken[field]
        expect_schema_error(broken)
    for section in ("curvature", "callback", "split_disjointness", "evidence", "runtime"):
        definition = {"split_disjointness": "splitDisjointness"}.get(section, section)
        for field in schema["$defs"][definition]["required"]:
            broken = copy.deepcopy(action_dict)
            del broken[section][field]
            expect_schema_error(broken)
    # production_dict was schema-validated while its observed artifacts/environment existed.
    for field in schema["$defs"]["materializationQualification"]["required"]:
        broken = copy.deepcopy(production_dict)
        del broken["runtime"]["materialization_qualification"][field]
        expect_schema_error(broken)

    print("exl3_action self-test: pass (curvature/callback/splits/KLD/materialization fail-closed)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    val = sub.add_parser("validate-schema")
    val.add_argument("--schema", required=True)
    val.add_argument("--action", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--source", required=True)
    smoke.add_argument("--key", required=True)
    smoke.add_argument("--source-revision", required=True)
    smoke.add_argument("--out-dir", required=True)
    smoke.add_argument("--receipt", required=True)
    smoke.add_argument("--container-image", required=True)
    smoke.add_argument("--container-image-digest", required=True)
    smoke.add_argument("--action-schema", required=True)
    smoke.add_argument("--K", type=int, default=5, choices=LEGAL_STOCK_K)
    smoke.add_argument("--codebook", default="mcg", choices=tuple(CODEBOOK_MARKERS))
    smoke.add_argument("--seed", type=int, default=300030)
    smoke.add_argument("--device", default="cuda:0")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "validate-schema":
        validate_schema(args)
    elif args.command == "smoke":
        receipt = run_smoke(args)
        print(json.dumps({"status": receipt["status"], "receipt": args.receipt}, sort_keys=True))

if __name__ == "__main__":
    main()
