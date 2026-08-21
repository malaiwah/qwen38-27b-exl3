#!/usr/bin/env python3
"""Prove clean packed-INT6 encoding matches the historical fidelity implementation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any

from frontier_common import atomic_write_json, sha256_file

SCHEMA = "qwen38-int6-embedding-compat/1"
DEFAULT_ROWS = ((0, 16), (124160, 124176), (248304, 248320))


class CompatibilityError(ValueError):
    """A fail-closed compatibility or provenance error."""


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _load_production_encoder(source: Path) -> tuple[Any, str]:
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    names = {"_validate_embedding_input", "_encode_int6_embedding"}
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    found = {
        node.name
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if found != names:
        raise CompatibilityError(
            f"production source lacks exact encoder functions: {sorted(names - found)}"
        )
    module = importlib.import_module("vllm.model_executor.layers.quantization.exl3")
    encoder = getattr(module, "_encode_int6_embedding", None)
    if not callable(encoder):
        raise CompatibilityError("imported runtime lacks _encode_int6_embedding")
    imported_path_text = inspect.getsourcefile(encoder)
    if imported_path_text is None:
        raise CompatibilityError("cannot resolve imported encoder source")
    imported_path = Path(imported_path_text).resolve(strict=True)
    if sha256_file(imported_path) != sha256_file(source):
        raise CompatibilityError(
            f"imported encoder source {imported_path} does not match {source}"
        )
    ast_identity = hashlib.sha256(
        "\\n".join(
            ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in nodes
        ).encode("utf-8")
    ).hexdigest()
    return encoder, ast_identity


def _historical_encode(weight: Any) -> tuple[Any, Any]:
    """Independent row/group implementation of the historical packed-INT6 format."""
    import torch

    if weight.ndim != 2 or weight.shape[1] % 4:
        raise CompatibilityError("historical oracle requires a rank-2, /4 width")
    rows, hidden = weight.shape
    q_weight = torch.empty((rows, hidden // 4 * 3), dtype=torch.uint8)
    scales = torch.empty(rows, dtype=torch.float16)
    for row_index in range(rows):
        row = weight[row_index].float()
        scale = row.abs().max().clamp(min=1.0e-8) / 31.0
        scales[row_index] = scale.half()
        unsigned = (
            (row / scale)
            .round()
            .clamp(-32, 31)
            .add(32)
            .to(torch.int64)
            .reshape(hidden // 4, 4)
        )
        packed = (
            unsigned[:, 0]
            | (unsigned[:, 1] << 6)
            | (unsigned[:, 2] << 12)
            | (unsigned[:, 3] << 18)
        )
        q_weight[row_index] = torch.stack(
            (
                (packed & 0xFF).to(torch.uint8),
                ((packed >> 8) & 0xFF).to(torch.uint8),
                ((packed >> 16) & 0xFF).to(torch.uint8),
            ),
            dim=-1,
        ).reshape(hidden // 4 * 3)
    return q_weight, scales


def _unpack(packed: Any, scales: Any) -> Any:
    import torch

    prefix = packed.shape[:-1]
    hidden = packed.shape[-1] // 3 * 4
    triples = packed.reshape(*prefix, hidden // 4, 3).to(torch.int64)
    values = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    unsigned = torch.stack(
        (
            values & 0x3F,
            (values >> 6) & 0x3F,
            (values >> 12) & 0x3F,
            (values >> 18) & 0x3F,
        ),
        dim=-1,
    ).reshape(*prefix, hidden)
    return (unsigned - 32).float() * scales.float().unsqueeze(-1)


def _load_rows(
    model: Path, tensor_name: str, ranges: tuple[tuple[int, int], ...]
) -> Any:
    import torch
    from safetensors import safe_open

    with safe_open(model, framework="pt", device="cpu") as handle:
        if tensor_name not in set(handle.keys()):
            raise CompatibilityError(f"tensor is absent from shard: {tensor_name}")
        view = handle.get_slice(tensor_name)
        shape = view.get_shape()
        if len(shape) != 2:
            raise CompatibilityError(f"embedding tensor must be rank 2, got {shape}")
        pieces = []
        previous_stop = -1
        for start, stop in ranges:
            if start < 0 or stop <= start or stop > shape[0] or start < previous_stop:
                raise CompatibilityError(
                    f"invalid or overlapping row range: {(start, stop)}"
                )
            pieces.append(view[start:stop, :])
            previous_stop = stop
    rows = torch.cat(pieces, dim=0)
    if rows.dtype != torch.bfloat16:
        raise CompatibilityError(f"embedding source must be BF16, got {rows.dtype}")
    return rows


def _validate_historical_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = (
        "VLLM_EXL3_EMBED_ONLINE_BITS",
        "self.bits == 6",
        "(u[..., 1] << 6)",
        "(u[..., 2] << 12)",
        "(u[..., 3] << 18)",
        "(val >> 18) & 0x3F",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise CompatibilityError(f"historical source contract changed: {missing}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    historical_source = args.historical_source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    if any(path.is_symlink() for path in (source, historical_source, model)):
        raise CompatibilityError("source and model inputs must not be symlinks")
    _validate_historical_source(historical_source)
    encoder, ast_identity = _load_production_encoder(source)
    rows = _load_rows(model, args.tensor, DEFAULT_ROWS)
    clean_q, clean_scales = encoder(rows, chunk_rows=7)
    historical_q, historical_scales = _historical_encode(rows)
    if not torch.equal(clean_q, historical_q):
        mismatch = int(torch.count_nonzero(clean_q != historical_q).item())
        raise CompatibilityError(f"packed INT6 bytes differ at {mismatch} positions")
    if not torch.equal(clean_scales, historical_scales):
        mismatch = int(torch.count_nonzero(clean_scales != historical_scales).item())
        raise CompatibilityError(f"stored FP16 scales differ at {mismatch} rows")
    clean_reconstruction = _unpack(clean_q, clean_scales)
    historical_reconstruction = _unpack(historical_q, historical_scales)
    if not torch.equal(clean_reconstruction, historical_reconstruction):
        raise CompatibilityError("dequantized rows differ despite matching payloads")
    expected_bytes = rows.shape[0] * rows.shape[1] * 3 // 4
    if clean_q.numel() != expected_bytes:
        raise CompatibilityError(
            f"packed byte count {clean_q.numel()} != expected {expected_bytes}"
        )
    known = torch.tensor([[-7.75, 0.0, 7.75, -0.25]], dtype=torch.float32)
    known_q, known_scale = encoder(known, chunk_rows=1)
    if not torch.equal(known_q, torch.tensor([[1, 248, 127]], dtype=torch.uint8)):
        raise CompatibilityError("known packing oracle failed")
    if not torch.equal(known_scale, torch.tensor([0.25], dtype=torch.float16)):
        raise CompatibilityError("known scale oracle failed")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "commit": args.source_commit,
            "encoder_ast_sha256": ast_identity,
        },
        "historical_source": {
            "path": str(historical_source),
            "sha256": sha256_file(historical_source),
        },
        "model": {
            "shard_path": str(model),
            "shard_sha256": sha256_file(model),
            "tensor": args.tensor,
            "dtype": "BF16",
            "row_ranges": [list(value) for value in DEFAULT_ROWS],
            "sample_shape": list(rows.shape),
            "sample_sha256": _tensor_sha256(rows),
        },
        "payload": {
            "q_dtype": "uint8",
            "q_shape": list(clean_q.shape),
            "q_sha256": _tensor_sha256(clean_q),
            "scale_dtype": "float16",
            "scale_shape": list(clean_scales.shape),
            "scale_sha256": _tensor_sha256(clean_scales),
            "packed_bytes": clean_q.numel(),
            "scale_bytes": clean_scales.numel() * clean_scales.element_size(),
            "historical_bytes_equal": True,
            "historical_scales_equal": True,
            "historical_reconstruction_equal": True,
        },
        "oracles": {
            "known_packed_bytes": [1, 248, 127],
            "known_scale": 0.25,
            "chunk_rows": 7,
            "arbitrary_row_ranges": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--historical-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--tensor",
        default="model.language_model.embed_tokens.weight",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    result = run(args)
    atomic_write_json(args.out, result)
    print(
        "PASS: clean packed INT6 is byte-identical to historical encoding "
        f"for {result['model']['sample_shape'][0]} real BF16 rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
