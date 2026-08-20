#!/usr/bin/env python3
"""Verify the pinned EXL3 GDN optimizer patch and its level-3 grouping oracle.

This checker is CPU-only and dependency-free. It does not import exllamav3 or
execute model code. When --source is supplied, the GDN module must be the exact
patched derivative of the pinned v1.4.2 source; the exact unpatched source is a
hard failure.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

UPSTREAM_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"
PATCHED_COMMIT = "a236a3e39fcb3343f369e5bb240587ad24730562"
SOURCE_RELATIVE_PATH = Path("exllamav3/modules/gated_delta_net.py")
TEST_RELATIVE_PATH = Path("tests/test_optimizer_targets.py")
BASE_GIT_BLOB = "f3705b67e806977e6608a8795cc4176d94206f53"
BASE_SHA256 = "b4b0d389dd2f34a61de8245ae3f69f597f4757dcfb3d7f59f0ea1b1c6de3213b"
PATCHED_GIT_BLOB = "36bda569fa6c4cab8e0b50b4c3eef6c9bc11eee2"
PATCHED_SHA256 = "062f9822b0c473963d06ebc7022976b9db16b8dd1436691511b8152801ee9216"
PATCHED_TEST_GIT_BLOB = "43e95f65bb468682345b99819fd3ae8ba564ee1b"
PATCHED_TEST_SHA256 = "e4c1a18c00ef1ea0c3af779dce9359aaa489292e99e083bd3f62b6ab388361b8"
PATCH_NAME = "exllamav3-1.4.2-gdn-optimizer-targets.patch"
PATCH_SHA256 = "bb69d839363d67ec6c5c8a3f727eddbaf0746340bdedd70914d0cc79800f5a2d"

EXPECTED_DIFF = """--- a/exllamav3/modules/gated_delta_net.py
+++ b/exllamav3/modules/gated_delta_net.py
@@ -523,13 +523,17 @@ class GatedDeltaNet(Module):
     @override
     def optimizer_targets(self):
         if self.qkvz_proj is not None:
-            return [[self.qkvz_proj.optimizer_targets()]]
+            return [[
+                self.qkvz_proj.optimizer_targets(),
+                self.o_proj.optimizer_targets(),
+            ]]
<SP>
         targets = []
         if self.qkv_proj is not None:
-            targets += self.qkv_proj.optimizer_targets()
+            targets.append(self.qkv_proj.optimizer_targets())
         if self.z_proj is not None:
-            targets += self.z_proj.optimizer_targets()
+            targets.append(self.z_proj.optimizer_targets())
+        targets.append(self.o_proj.optimizer_targets())
         return [targets]
""".replace("<SP>\n", " \n")


class VerificationError(RuntimeError):
    """A patch, source-identity, or grouping invariant failed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def verify_patch(path: Path) -> None:
    try:
        patch_bytes = path.read_bytes()
        patch = patch_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"cannot read patch {path}: {exc}") from exc

    if sha256_bytes(patch_bytes) != PATCH_SHA256:
        raise VerificationError("patch whole-file SHA256 does not match the reviewed artifact")
    headers = [
        line
        for line in patch.splitlines()
        if line.startswith("diff --git a/")
    ]
    expected_headers = [
        "diff --git a/exllamav3/modules/gated_delta_net.py b/exllamav3/modules/gated_delta_net.py",
        "diff --git a/tests/test_optimizer_targets.py b/tests/test_optimizer_targets.py",
    ]
    if headers != expected_headers:
        raise VerificationError(f"patch file set/order changed: {headers}")
    if EXPECTED_DIFF not in patch:
        raise VerificationError("patch does not contain the reviewed single-method GDN change")
    if "test_qwen38_optimizer_topology_has_expected_unique_groups_and_linears" not in patch:
        raise VerificationError("patch omits the checked-in architecture-level grouping proof")


def resolve_source(path: Path) -> Path:
    return path / SOURCE_RELATIVE_PATH if path.is_dir() else path


def verify_patched_source(path: Path) -> None:
    source_path = resolve_source(path)
    try:
        source = source_path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read source {source_path}: {exc}") from exc

    sha256 = sha256_bytes(source)
    blob = git_blob_sha1(source)
    if sha256 == BASE_SHA256 and blob == BASE_GIT_BLOB:
        raise VerificationError(
            f"{source_path} is the pinned but unpatched GDN source; apply {PATCH_NAME}"
        )
    if sha256 != PATCHED_SHA256 or blob != PATCHED_GIT_BLOB:
        raise VerificationError(
            f"{source_path} has unknown identity (sha256={sha256}, git_blob={blob}); "
            f"expected the patched derivative at {PATCHED_COMMIT}"
        )
    if path.is_dir():
        test_path = path / TEST_RELATIVE_PATH
        try:
            test_source = test_path.read_bytes()
        except OSError as exc:
            raise VerificationError(f"cannot read test proof {test_path}: {exc}") from exc
        if (
            sha256_bytes(test_source) != PATCHED_TEST_SHA256
            or git_blob_sha1(test_source) != PATCHED_TEST_GIT_BLOB
        ):
            raise VerificationError(
                f"{test_path} is not the architecture proof from {PATCHED_COMMIT}"
            )


def linear(key: str) -> list[str]:
    """Return the optimizer_targets() shape of an exllamav3 Linear."""
    return [key]


def flatten_measure_targets(node: Sequence[object], max_depth: int) -> list[list[str]]:
    """Mirror conversion/measure_model.py's target grouping traversal."""
    groups: list[list[str]] = []

    def flatten(current: Sequence[object], depth: int = 0) -> list[str]:
        flattened: list[str] = []
        for child in current:
            if isinstance(child, str):
                flattened.append(child)
            else:
                child_flat = flatten(child, depth + 1)  # type: ignore[arg-type]
                if depth >= max_depth:
                    flattened.extend(child_flat)
                else:
                    flattened.append(child_flat)  # type: ignore[arg-type]
        if depth == max_depth:
            groups.append(flattened)
        return flattened

    flatten(node)
    return groups


def qwen_body_oracle() -> tuple[list[list[str]], dict[str, int]]:
    all_groups: list[list[str]] = []
    expected_linears: set[str] = set()

    for layer_index in range(64):
        prefix = f"model.language_model.layers.{layer_index}"
        kv_pair: tuple[str, str] | None = None
        if (layer_index + 1) % 4:
            attn_prefix = f"{prefix}.linear_attn"
            qkv = f"{attn_prefix}.in_proj_qkv"
            z = f"{attn_prefix}.in_proj_z"
            output = f"{attn_prefix}.out_proj"
            attention_targets: list[object] = [[linear(qkv), linear(z), linear(output)]]
            expected_linears.update((qkv, z, output))
        else:
            attn_prefix = f"{prefix}.self_attn"
            q = f"{attn_prefix}.q_proj"
            k = f"{attn_prefix}.k_proj"
            v = f"{attn_prefix}.v_proj"
            output = f"{attn_prefix}.o_proj"
            attention_targets = [[linear(q), linear(k) + linear(v), linear(output)]]
            expected_linears.update((q, k, v, output))
            kv_pair = (k, v)

        mlp_prefix = f"{prefix}.mlp"
        gate = f"{mlp_prefix}.gate_proj"
        up = f"{mlp_prefix}.up_proj"
        down = f"{mlp_prefix}.down_proj"
        mlp_targets = [[linear(gate) + linear(up), linear(down)]]
        expected_linears.update((gate, up, down))

        block_targets = [attention_targets, mlp_targets]
        groups = flatten_measure_targets(block_targets, max_depth=3)
        all_groups.extend(groups)

        if kv_pair is not None:
            k, v = kv_pair
            kv_groups = [group for group in groups if k in group or v in group]
            if kv_groups != [[k, v]]:
                raise VerificationError(f"layer {layer_index}: k+v grouping was not retained")
        gate_up_groups = [group for group in groups if gate in group or up in group]
        if gate_up_groups != [[gate, up]]:
            raise VerificationError(f"layer {layer_index}: gate+up grouping was not retained")

    fused_groups = flatten_measure_targets(
        [[[["fused.linear_attn.in_proj_qkvz"], ["fused.linear_attn.out_proj"]]], []],
        max_depth=3,
    )
    if fused_groups != [
        ["fused.linear_attn.in_proj_qkvz"],
        ["fused.linear_attn.out_proj"],
    ]:
        raise VerificationError(f"fused GDN input/output grouping mismatch: {fused_groups}")

    counts = Counter(key for group in all_groups for key in group)
    if len(all_groups) != 320:
        raise VerificationError(f"level-3 oracle produced {len(all_groups)} groups, expected 320")
    if len(expected_linears) != 400:
        raise VerificationError(f"oracle fixture defines {len(expected_linears)} linears, expected 400")
    if set(counts) != expected_linears:
        missing = sorted(expected_linears - set(counts))
        extra = sorted(set(counts) - expected_linears)
        raise VerificationError(f"linear coverage mismatch: missing={missing}, extra={extra}")
    repeated = sorted(key for key, count in counts.items() if count != 1)
    if repeated:
        raise VerificationError(f"linears not covered exactly once: {repeated}")
    actual_counts = {
        "gdn": sum(any(".linear_attn." in key for key in group) for group in all_groups),
        "attention": sum(any(".self_attn." in key for key in group) for group in all_groups),
        "mlp": sum(any(".mlp." in key for key in group) for group in all_groups),
    }
    required_counts = {"gdn": 144, "attention": 48, "mlp": 128}
    if actual_counts != required_counts:
        raise VerificationError(f"group class counts mismatch: {actual_counts}")

    return all_groups, actual_counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify the pinned EXL3 GDN optimizer patch and Qwen level-3 grouping oracle."
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=repository / "patches" / PATCH_NAME,
        help="patch file to verify (default: repository patch)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "optional patched exllamav3 checkout root or gated_delta_net.py; "
            "the exact unpatched source fails"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        verify_patch(args.patch)
        if args.source is not None:
            verify_patched_source(args.source)
        groups, counts = qwen_body_oracle()
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    source_status = "patched source identity verified; " if args.source is not None else ""
    print(
        "PASS: "
        f"{source_status}patch identity verified; "
        f"{len(groups)} level-3 groups cover 400 body linears exactly once "
        f"(GDN {counts['gdn']}, full attention {counts['attention']}, MLP {counts['mlp']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
