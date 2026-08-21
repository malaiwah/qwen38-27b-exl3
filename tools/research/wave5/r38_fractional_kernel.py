#!/usr/bin/env python3
"""R38 fixed-stride K5/K6 theory and two-stream container prototype.

This tool deliberately does not implement or simulate EXL3 quantization.  It packs
opaque files emitted by the pinned R30 actual-EXL3 harness and accounts for every
byte in a deterministic two-stream container.  Quality and SM120 commands are not
present: those experiments are conditional on an affirmative R37 residual-value
gate.

The prototype format partitions the *input* dimension into fixed equal halves at
an H128 boundary: K5 owns the prefix and K6 owns the suffix.  A decoder evaluates
both dense stock EXL3 streams on their corresponding input slices and sums their
output vectors.  The partition is derivable from the format ID; there is no
selector map.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MAGIC = b"EXL3FS1\0"
VERSION = 1
HEADER_BYTES = 256
ALIGNMENT = 256
H128_WIDTH = 128
FORMAT_ID = "exl3-fs1-input-h128-half-k5-prefix-k6-suffix"
# magic, version, header bytes, alignment, axis, K5, K6, flags,
# in, out, K5-in, K6-in, offsets/sizes, stream hashes, format id
_HEADER = struct.Struct("<8sHHHBBBBIIIIQQQQ32s32s64s")
_AXIS_INPUT = 0
_FLAG_BF16_PARTIAL_FP32_SUM = 1


class FormatError(ValueError):
    """The fixed-stride container or its declared semantics are invalid."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise FormatError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def _require_uint(name: str, value: int, bits: int = 32) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << bits):
        raise FormatError("%s must be an unsigned %d-bit integer" % (name, bits))


@dataclasses.dataclass(frozen=True)
class StripeSpec:
    input_features: int
    output_features: int
    k5_input_features: int

    def validate(self) -> "StripeSpec":
        for name, value in (
            ("input_features", self.input_features),
            ("output_features", self.output_features),
            ("k5_input_features", self.k5_input_features),
        ):
            _require_uint(name, value)
        if self.input_features == 0 or self.output_features == 0:
            raise FormatError("input and output dimensions must be nonzero")
        if self.input_features % (2 * H128_WIDTH):
            raise FormatError("input_features must contain an even number of H128 blocks")
        if self.output_features % H128_WIDTH:
            raise FormatError("output_features must be H128 aligned")
        if self.k5_input_features != self.input_features // 2:
            raise FormatError("FS1 fixes equal K5/K6 half stripes; the split is not selectable")
        return self

    @property
    def k6_input_features(self) -> int:
        return self.input_features - self.k5_input_features

    @property
    def k6_fraction(self) -> float:
        return self.k6_input_features / self.input_features

    @property
    def effective_payload_bpw(self) -> float:
        return 5.0 + self.k6_fraction

    def k_for_input_coordinate(self, input_coordinate: int) -> int:
        if not 0 <= input_coordinate < self.input_features:
            raise FormatError("input coordinate outside shard")
        return 5 if input_coordinate < self.k5_input_features else 6

    def descriptor(self) -> Dict[str, Any]:
        self.validate()
        return {
            "axis": "input",
            "format_id": FORMAT_ID,
            "h128_width": H128_WIDTH,
            "input_features": self.input_features,
            "k5": {"begin": 0, "end": self.k5_input_features},
            "k6": {"begin": self.k5_input_features, "end": self.input_features},
            "stripe_fraction": "1/2",
            "no_selector_map": True,
            "output_features": self.output_features,
            "runtime_semantics": "two stock EXL3 GEMMs return BF16 partial outputs; convert each to FP32, add K5 then K6, cast once to BF16; activation-slice materialization is additional counted work if zero-copy strided views are unsupported",
        }


@dataclasses.dataclass(frozen=True)
class ContainerInfo:
    path: Path
    spec: StripeSpec
    k5_offset: int
    k5_size: int
    k5_sha256: str
    k6_offset: int
    k6_size: int
    k6_sha256: str
    final_padding_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.k6_offset + self.k6_size + self.final_padding_bytes

    def byte_components(self) -> Dict[str, int]:
        return {
            "descriptor_header": HEADER_BYTES,
            "alignment_before_k5": self.k5_offset - HEADER_BYTES,
            "k5_actual_stream_file": self.k5_size,
            "alignment_between_streams": self.k6_offset - (self.k5_offset + self.k5_size),
            "k6_actual_stream_file": self.k6_size,
            "final_alignment": self.final_padding_bytes,
            "selector_map": 0,
            "total_actual_file": self.total_bytes,
        }

    def as_json(self) -> Dict[str, Any]:
        return {
            "schema": "qwen38-wave5-r38-fixed-stripe-container/1",
            "path": str(self.path),
            "format": self.spec.descriptor(),
            "streams": [
                {"K": 5, "offset": self.k5_offset, "size": self.k5_size, "sha256": self.k5_sha256},
                {"K": 6, "offset": self.k6_offset, "size": self.k6_size, "sha256": self.k6_sha256},
            ],
            "bytes": self.byte_components(),
            "file_sha256": _sha256_file(self.path),
        }


def _encoded_header(
    spec: StripeSpec,
    k5_offset: int,
    k5_size: int,
    k6_offset: int,
    k6_size: int,
    k5_digest: bytes,
    k6_digest: bytes,
) -> bytes:
    spec.validate()
    if len(k5_digest) != 32 or len(k6_digest) != 32:
        raise FormatError("stream SHA256 digests must be 32 bytes")
    format_bytes = FORMAT_ID.encode("ascii")
    if len(format_bytes) > 64:
        raise AssertionError("FORMAT_ID exceeds header field")
    packed = _HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        ALIGNMENT,
        _AXIS_INPUT,
        5,
        6,
        _FLAG_BF16_PARTIAL_FP32_SUM,
        spec.input_features,
        spec.output_features,
        spec.k5_input_features,
        spec.k6_input_features,
        k5_offset,
        k5_size,
        k6_offset,
        k6_size,
        k5_digest,
        k6_digest,
        format_bytes,
    )
    if len(packed) > HEADER_BYTES:
        raise AssertionError("header struct exceeds fixed header")
    return packed + bytes(HEADER_BYTES - len(packed))


def pack_streams(k5_path: Path, k6_path: Path, output_path: Path, spec: StripeSpec) -> ContainerInfo:
    """Pack two actual stream files without interpreting or rewriting them."""
    spec.validate()
    k5_path = k5_path.resolve()
    k6_path = k6_path.resolve()
    output_path = output_path.resolve()
    if output_path in (k5_path, k6_path):
        raise FormatError("output must not overwrite an input stream")
    k5 = k5_path.read_bytes()
    k6 = k6_path.read_bytes()
    if not k5 or not k6:
        raise FormatError("both actual stream files must be nonempty")
    k5_offset = _align_up(HEADER_BYTES)
    k6_offset = _align_up(k5_offset + len(k5))
    end = k6_offset + len(k6)
    final_end = _align_up(end)
    header = _encoded_header(
        spec,
        k5_offset,
        len(k5),
        k6_offset,
        len(k6),
        hashlib.sha256(k5).digest(),
        hashlib.sha256(k6).digest(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp-%d" % os.getpid())
    try:
        with tmp_path.open("xb") as handle:
            handle.write(header)
            handle.write(bytes(k5_offset - handle.tell()))
            handle.write(k5)
            handle.write(bytes(k6_offset - handle.tell()))
            handle.write(k6)
            handle.write(bytes(final_end - handle.tell()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(output_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return inspect_container(output_path, verify_streams=True)


def _decode_header(header: bytes) -> Tuple[StripeSpec, int, int, int, int, bytes, bytes]:
    if len(header) != HEADER_BYTES:
        raise FormatError("short fixed-stride header")
    values = _HEADER.unpack(header[: _HEADER.size])
    (
        magic,
        version,
        header_bytes,
        alignment,
        axis,
        low_k,
        high_k,
        flags,
        input_features,
        output_features,
        k5_input_features,
        k6_input_features,
        k5_offset,
        k5_size,
        k6_offset,
        k6_size,
        k5_digest,
        k6_digest,
        format_bytes,
    ) = values
    if magic != MAGIC or version != VERSION:
        raise FormatError("unknown magic/version")
    if header_bytes != HEADER_BYTES or alignment != ALIGNMENT:
        raise FormatError("header/alignment contract mismatch")
    if (axis, low_k, high_k, flags) != (_AXIS_INPUT, 5, 6, _FLAG_BF16_PARTIAL_FP32_SUM):
        raise FormatError("unsupported stripe or runtime semantics")
    expected_format = FORMAT_ID.encode("ascii").ljust(64, b"\0")
    if format_bytes != expected_format:
        raise FormatError("format ID field is not canonical")
    if any(header[_HEADER.size :]):
        raise FormatError("reserved header bytes must be zero")
    spec = StripeSpec(input_features, output_features, k5_input_features).validate()
    if k6_input_features != spec.k6_input_features:
        raise FormatError("inconsistent stripe widths")
    if k5_size == 0 or k6_size == 0:
        raise FormatError("both embedded stream files must be nonempty")
    return spec, k5_offset, k5_size, k6_offset, k6_size, k5_digest, k6_digest


def inspect_container(path: Path, verify_streams: bool = True) -> ContainerInfo:
    path = path.resolve()
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
        spec, k5_offset, k5_size, k6_offset, k6_size, k5_digest, k6_digest = _decode_header(header)
        if k5_offset != _align_up(HEADER_BYTES):
            raise FormatError("K5 stream offset is not canonical")
        if k6_offset != _align_up(k5_offset + k5_size):
            raise FormatError("K6 stream offset is not canonical")
        final_end = _align_up(k6_offset + k6_size)
        if file_size != final_end:
            raise FormatError("actual file size does not match canonical final alignment")
        handle.seek(HEADER_BYTES)
        if any(handle.read(k5_offset - HEADER_BYTES)):
            raise FormatError("nonzero padding before K5 stream")
        handle.seek(k5_offset)
        k5 = handle.read(k5_size)
        handle.seek(k5_offset + k5_size)
        if any(handle.read(k6_offset - (k5_offset + k5_size))):
            raise FormatError("nonzero inter-stream padding")
        handle.seek(k6_offset)
        k6 = handle.read(k6_size)
        final_padding = final_end - (k6_offset + k6_size)
        if any(handle.read(final_padding)):
            raise FormatError("nonzero final padding")
    if verify_streams:
        if hashlib.sha256(k5).digest() != k5_digest:
            raise FormatError("K5 stream digest mismatch")
        if hashlib.sha256(k6).digest() != k6_digest:
            raise FormatError("K6 stream digest mismatch")
    return ContainerInfo(
        path=path,
        spec=spec,
        k5_offset=k5_offset,
        k5_size=k5_size,
        k5_sha256=k5_digest.hex(),
        k6_offset=k6_offset,
        k6_size=k6_size,
        k6_sha256=k6_digest.hex(),
        final_padding_bytes=final_padding,
    )


def extract_streams(container_path: Path, k5_output: Path, k6_output: Path) -> Dict[str, str]:
    container_resolved = container_path.resolve()
    k5_resolved = k5_output.resolve()
    k6_resolved = k6_output.resolve()
    if len({container_resolved, k5_resolved, k6_resolved}) != 3:
        raise FormatError("container and both extraction outputs must be distinct resolved paths")
    info = inspect_container(container_resolved, verify_streams=True)
    with info.path.open("rb") as handle:
        handle.seek(info.k5_offset)
        k5 = handle.read(info.k5_size)
        handle.seek(info.k6_offset)
        k6 = handle.read(info.k6_size)
    k5_resolved.write_bytes(k5)
    k6_resolved.write_bytes(k6)
    return {"k5_sha256": _sha256_file(k5_resolved), "k6_sha256": _sha256_file(k6_resolved)}


def stock_raw_bytes(input_features: int, output_features: int, k: int, marker_bytes: int = 4) -> int:
    """R30 returned-buffer byte law for one dense H128-aligned MCG/MUL1 stream."""
    if k not in (5, 6):
        raise FormatError("R38 supports only K5/K6")
    for name, value in (("input_features", input_features), ("output_features", output_features)):
        _require_uint(name, value)
    if input_features == 0 or input_features % H128_WIDTH:
        raise FormatError("input_features must be nonzero and H128 aligned")
    if output_features == 0 or output_features % H128_WIDTH:
        raise FormatError("output_features must be nonzero and H128 aligned")
    numel = input_features * output_features
    if (numel * k) % 8:
        raise FormatError("trellis payload is not byte integral")
    return numel * k // 8 + 2 * (input_features + output_features) + marker_bytes


def two_stream_raw_bytes(spec: StripeSpec, marker_bytes_per_stream: int = 4) -> Dict[str, int]:
    """Exact returned-buffer law before the actual file/header/alignment layer."""
    spec.validate()
    k5 = stock_raw_bytes(spec.k5_input_features, spec.output_features, 5, marker_bytes_per_stream)
    k6 = stock_raw_bytes(spec.k6_input_features, spec.output_features, 6, marker_bytes_per_stream)
    total = k5 + k6
    weighted_payload = spec.output_features * (5 * spec.k5_input_features + 6 * spec.k6_input_features) // 8
    components = {
        "weighted_trellis_payload": weighted_payload,
        "input_scale_bytes_total": 2 * spec.input_features,
        "duplicated_output_scale_bytes": 4 * spec.output_features,
        "marker_bytes": 2 * marker_bytes_per_stream,
        "raw_total": total,
    }
    if sum(v for k, v in components.items() if k != "raw_total") != total:
        raise AssertionError("two-stream byte decomposition drifted")
    return components


def theory_report(spec: StripeSpec) -> Dict[str, Any]:
    spec.validate()
    k5 = stock_raw_bytes(spec.input_features, spec.output_features, 5)
    k6 = stock_raw_bytes(spec.input_features, spec.output_features, 6)
    two = two_stream_raw_bytes(spec)
    # At the same K6 fraction, an ideal whole-module mixture amortizes exactly one
    # output-scale vector and one marker per module.  This is an accounting control,
    # not a quality model and not an assertion that fractional module counts exist.
    module_mix_interpolated = k5 + round(spec.k6_fraction * (k6 - k5))
    overhead = two["raw_total"] - module_mix_interpolated
    numel = spec.input_features * spec.output_features
    return {
        "schema": "qwen38-wave5-r38-fixed-stripe-theory/1",
        "claim_scope": "exact returned-buffer byte accounting; no quality, KLD, or runtime claim",
        "spec": spec.descriptor(),
        "payload_bpw": spec.effective_payload_bpw,
        "controls": {"uniform_k5_raw_bytes": k5, "uniform_k6_raw_bytes": k6},
        "two_stream_raw": two,
        "whole_module_mix_interpolated_raw_bytes": module_mix_interpolated,
        "two_stream_overhead_vs_interpolated_module_mix": overhead,
        "overhead_effective_bpw": 8.0 * overhead / numel,
        "format_file_overhead_not_in_raw_law": {
            "fixed_descriptor_header_bytes": HEADER_BYTES,
            "stream_file_headers": "actual input file sizes, never estimated",
            "alignment_bytes": "actual pack result, never estimated",
        },
        "runtime_contract": {
            "initial": "at least two stock EXL3 GEMM launches plus one FP32 elementwise sum/cast launch; count activation-slice materialization if zero-copy strided views are unsupported; no fusion claim",
            "decode_hot_path": "BF16 K5/K6 partial outputs; convert both to FP32, add K5 then K6, cast once to BF16 at the module boundary",
            "fused_kernel_authorized": False,
        },
    }


def run_self_test() -> Dict[str, Any]:
    checks = []
    with tempfile.TemporaryDirectory(prefix="r38-fs1-") as tmp:
        root = Path(tmp)
        k5_path = root / "k5.safetensors"
        k6_path = root / "k6.safetensors"
        out = root / "fixed-stripe.bin"
        original_k5 = hashlib.shake_256(b"r38-k5").digest(701)
        original_k6 = hashlib.shake_256(b"r38-k6").digest(1003)
        k5_path.write_bytes(original_k5)
        k6_path.write_bytes(original_k6)
        spec = StripeSpec(5120, 1024, 2560)
        info = pack_streams(k5_path, k6_path, out, spec)
        prototype = info.as_json()
        prototype.pop("path")
        checks.append("pack_inspect")
        if info.total_bytes != out.stat().st_size or info.total_bytes % ALIGNMENT:
            raise AssertionError("actual size/alignment accounting failed")
        if sum(info.byte_components().values()) - info.total_bytes != info.total_bytes:
            # byte_components includes both the components and its reported total.
            raise AssertionError("component sum failed")
        checks.append("actual_byte_components")
        if (spec.k_for_input_coordinate(0), spec.k_for_input_coordinate(2559), spec.k_for_input_coordinate(2560), spec.k_for_input_coordinate(5119)) != (5, 5, 6, 6):
            raise AssertionError("fixed-stripe semantics failed")
        checks.append("deterministic_no_selector_semantics")
        out_k5 = root / "roundtrip-k5"
        out_k6 = root / "roundtrip-k6"
        hashes = extract_streams(out, out_k5, out_k6)
        if out_k5.read_bytes() != original_k5 or out_k6.read_bytes() != original_k6:
            raise AssertionError("stream round trip failed")
        if hashes != {"k5_sha256": _sha256_bytes(original_k5), "k6_sha256": _sha256_bytes(original_k6)}:
            raise AssertionError("round-trip hashes failed")
        checks.append("opaque_stream_roundtrip")
        alias_path = root / "alias-output"
        for alias_outputs in ((alias_path, alias_path), (out, alias_path)):
            try:
                extract_streams(out, alias_outputs[0], alias_outputs[1])
            except FormatError as exc:
                if "must be distinct resolved paths" not in str(exc):
                    raise
            else:
                raise AssertionError("aliased extraction paths were accepted")
        checks.append("extraction_aliases_rejected")
        broken = bytearray(out.read_bytes())
        broken[info.k6_offset] ^= 1
        broken_path = root / "broken.bin"
        broken_path.write_bytes(broken)
        try:
            inspect_container(broken_path, verify_streams=True)
        except FormatError as exc:
            if "K6 stream digest mismatch" not in str(exc):
                raise
        else:
            raise AssertionError("corrupt stream was accepted")
        checks.append("corruption_rejected")
        noncanonical = bytearray(out.read_bytes())
        format_offset = _HEADER.size - 64
        noncanonical[format_offset + len(FORMAT_ID) + 1] = 1
        noncanonical_path = root / "noncanonical-header.bin"
        noncanonical_path.write_bytes(noncanonical)
        try:
            inspect_container(noncanonical_path, verify_streams=True)
        except FormatError as exc:
            if "format ID field is not canonical" not in str(exc):
                raise
        else:
            raise AssertionError("noncanonical format field was accepted")
        checks.append("noncanonical_header_rejected")
        empty_path = root / "empty.safetensors"
        empty_path.write_bytes(b"")
        try:
            pack_streams(empty_path, k6_path, root / "empty-stream.bin", spec)
        except FormatError as exc:
            if "must be nonempty" not in str(exc):
                raise
        else:
            raise AssertionError("empty stream file was accepted")
        checks.append("empty_stream_rejected")
        try:
            StripeSpec(5000, 1024, 2500).validate()
        except FormatError as exc:
            if "even number of H128 blocks" not in str(exc):
                raise
        else:
            raise AssertionError("unaligned input width was accepted")
        checks.append("unaligned_input_rejected")
        try:
            StripeSpec(5120, 1024, 2432).validate()
        except FormatError as exc:
            if "split is not selectable" not in str(exc):
                raise
        else:
            raise AssertionError("selectable stripe fraction was accepted")
        checks.append("nonhalf_stripe_rejected")
        try:
            StripeSpec(5120, 1008, 2560).validate()
        except FormatError as exc:
            if "output_features must be H128 aligned" not in str(exc):
                raise
        else:
            raise AssertionError("unaligned output width was accepted")
        checks.append("unaligned_output_rejected")
        report = theory_report(spec)
        expected_overhead = 2 * spec.output_features + 4
        if report["two_stream_overhead_vs_interpolated_module_mix"] != expected_overhead:
            raise AssertionError("raw overhead law failed")
        checks.append("r30_raw_byte_law")
    return {
        "schema": "qwen38-wave5-r38-self-test/1",
        "status": "pass",
        "checks": checks,
        "format_id": FORMAT_ID,
        "prototype": prototype,
        "test_count": len(checks),
    }


def _write_json(path: Optional[Path], value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value)
    if path is None:
        sys.stdout.buffer.write(payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    self_test = sub.add_parser("self-test", help="exercise format, byte, corruption, and stripe invariants")
    self_test.add_argument("--output", type=Path)

    theory = sub.add_parser("theory", help="emit exact R30 raw-buffer byte accounting")
    theory.add_argument("--input-features", type=int, required=True)
    theory.add_argument("--output-features", type=int, required=True)
    theory.add_argument("--k5-input-features", type=int, required=True)
    theory.add_argument("--output", type=Path)

    pack = sub.add_parser("pack", help="pack two opaque actual-EXL3 stream files")
    pack.add_argument("--k5-stream", type=Path, required=True)
    pack.add_argument("--k6-stream", type=Path, required=True)
    pack.add_argument("--container", type=Path, required=True)
    pack.add_argument("--input-features", type=int, required=True)
    pack.add_argument("--output-features", type=int, required=True)
    pack.add_argument("--k5-input-features", type=int, required=True)
    pack.add_argument("--receipt", type=Path)

    inspect = sub.add_parser("inspect", help="validate and describe an existing container")
    inspect.add_argument("container", type=Path)
    inspect.add_argument("--output", type=Path)

    extract = sub.add_parser("extract", help="verify then extract both opaque streams")
    extract.add_argument("container", type=Path)
    extract.add_argument("--k5-output", type=Path, required=True)
    extract.add_argument("--k6-output", type=Path, required=True)
    extract.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "self-test":
        _write_json(args.output, run_self_test())
    elif args.command == "theory":
        spec = StripeSpec(args.input_features, args.output_features, args.k5_input_features)
        _write_json(args.output, theory_report(spec))
    elif args.command == "pack":
        spec = StripeSpec(args.input_features, args.output_features, args.k5_input_features)
        info = pack_streams(args.k5_stream, args.k6_stream, args.container, spec)
        _write_json(args.receipt, info.as_json())
    elif args.command == "inspect":
        _write_json(args.output, inspect_container(args.container, verify_streams=True).as_json())
    elif args.command == "extract":
        result = extract_streams(args.container, args.k5_output, args.k6_output)
        _write_json(args.output, {"schema": "qwen38-wave5-r38-extract/1", "status": "pass", **result})
    else:
        raise AssertionError("unreachable command")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormatError, FileNotFoundError, OSError) as exc:
        print("r38_fractional_kernel: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
