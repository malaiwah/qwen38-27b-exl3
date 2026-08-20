#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

BASE_COMMIT = "fa033bd4e1b16d9d729ad94be2d87da5a13210ce"
PATCHED_COMMIT = "5f167ce8dfadd91310142e5aadffc8101a14382c"
PATCH_COMMITS = (
    "ad99c0aeae3531e0ac53bb9f0cc5679d6e373e90",
    "28f484b9c043f00690d536b76eaac3ee7d503297",
    "033f8ba37fe6a419bbc81750c9929d42f9de9fad",
    "b53ec52664dcc81cbffdbca0296e4b10c2a25a0f",
    PATCHED_COMMIT,
)
BRANCH = "publish-qkv-consumer"
PATCH_NAME = "vllm-fa033bd4-exl3-qkv-topology.patch"
PATCH_SHA256 = "7521396b8a6bab84f4f43387cf893504033fdb25d7856231ed239cb924a03f98"
EXPECTED_PATHS = {
    "tests/quantization/test_exl3_qkv_topology.py",
    "vllm/model_executor/layers/linear.py",
    "vllm/model_executor/layers/quantization/exl3.py",
    "vllm/model_executor/models/qwen3_next.py",
}
EXPECTED_BLOBS = {
    "tests/quantization/test_exl3_qkv_topology.py": "79b24444cb9919839042e2d5785c39e3ee11dc52",
    "vllm/model_executor/layers/linear.py": "b671c59768219624dbf651d872346c0fa40707d7",
    "vllm/model_executor/layers/quantization/exl3.py": "227be92bcddaf86e87bf2e7354c0ab7df03e34e4",
    "vllm/model_executor/models/qwen3_next.py": "5504222558e662976e60bfb80bc1eb6b76fe44fa",
}
EXPECTED_TEST_FUNCTIONS = {
    "test_checkpoint_mixes_explicit_split_and_fused_uniform_routes",
    "test_text_routes_ignore_visual_layer_index_collisions",
    "test_projection_accepts_supported_k_boundaries_and_codebooks",
    "test_topology_rejects_non_decoder_layer_identities",
    "test_topology_metadata_fails_closed",
    "test_declared_topology_rejects_route_fallback",
    "test_fused_uniform_rejects_tensor_parallel_runtime",
    "test_fused_uniform_rejects_runtime_output_split_mismatch",
    "test_split_rejects_runtime_output_split_mismatch",
    "test_split_tp_runtime_reconstructs_local_output_splits",
    "test_fused_uniform_rejects_packed_fallback",
    "test_split_view_fallback_rejects_packed_width_mismatch",
    "test_fused_uniform_issues_one_apply_and_returns_views",
    "test_split_route_retains_three_applies_and_cat",
    "test_qkv_views_honor_return_bias_false",
    "test_qkv_views_return_deferred_bias_when_requested",
    "test_metadata_objects_are_not_aliased_to_caller",
}
EXPECTED_TEST_CASES = 25


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise VerificationError(
            f"git {' '.join(args)} failed in {checkout}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_patch(repo_root: Path) -> Path:
    patch = repo_root / "patches" / PATCH_NAME
    require(patch.is_file(), f"missing patch: {patch}")
    payload = patch.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == PATCH_SHA256,
        "patch SHA256 mismatch",
    )
    text = payload.decode("utf-8")
    require(
        f"Base-Commit: {BASE_COMMIT}\n" in text,
        "patch base-commit header mismatch",
    )
    require(
        f"Source-Commit: {PATCHED_COMMIT}\n" in text,
        "patch source-commit header mismatch",
    )
    touched = set(
        re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.MULTILINE)
    )
    require(
        touched == EXPECTED_PATHS,
        f"unexpected patch paths: {sorted(touched ^ EXPECTED_PATHS)}",
    )
    for marker in (
        '"exl3_qkv_topology/1"',
        '"fused_uniform"',
        "apply_qkv_views",
        "qkv_route_counters",
        "qkv_warmup_counters",
        "one-launch QKV remains future research",
        "qkv_registry_rows",
        "fused_uniform EXL3 QKV currently requires TP=1",
        "test_checkpoint_mixes_explicit_split_and_fused_uniform_routes",
        "test_fused_uniform_issues_one_apply_and_returns_views",
        "test_split_route_retains_three_applies_and_cat",
    ):
        require(marker in text, f"patch is missing required marker: {marker}")
    return patch


def test_case_count(source: str) -> tuple[set[str], int]:
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    cases = 0
    for function in functions.values():
        multiplicity = 1
        for decorator in function.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
                and len(decorator.args) >= 2
            ):
                values_node = decorator.args[1]
                if isinstance(values_node, (ast.List, ast.Tuple, ast.Set)):
                    multiplicity *= len(values_node.elts)
                else:
                    multiplicity *= len(ast.literal_eval(values_node))
        cases += multiplicity
    return set(functions), cases


def verify_test_pins(checkout: Path) -> None:
    path = checkout / "tests" / "quantization" / "test_exl3_qkv_topology.py"
    functions, cases = test_case_count(path.read_text(encoding="utf-8"))
    require(
        functions == EXPECTED_TEST_FUNCTIONS,
        f"focused test function set mismatch: {sorted(functions ^ EXPECTED_TEST_FUNCTIONS)}",
    )
    require(
        cases == EXPECTED_TEST_CASES,
        f"focused test case count mismatch: {cases}",
    )


def verify_checkout(checkout: Path, patch: Path) -> None:
    checkout = checkout.resolve()
    require((checkout / ".git").exists(), f"not a git checkout: {checkout}")
    require(
        git(checkout, "rev-parse", "HEAD") == PATCHED_COMMIT,
        "patched checkout HEAD mismatch",
    )
    require(
        tuple(
            git(
                checkout,
                "rev-list",
                "--reverse",
                f"{BASE_COMMIT}..{PATCHED_COMMIT}",
            ).splitlines()
        )
        == PATCH_COMMITS,
        "patched checkout commit series mismatch",
    )
    require(
        git(checkout, "merge-base", BASE_COMMIT, PATCHED_COMMIT) == BASE_COMMIT,
        "pinned base is not the source commit's merge base",
    )
    require(
        git(checkout, "branch", "--show-current") == BRANCH,
        "implementation branch mismatch",
    )
    require(not git(checkout, "status", "--porcelain"), "patched checkout is dirty")
    changed = set(
        git(checkout, "diff", "--name-only", BASE_COMMIT, PATCHED_COMMIT).splitlines()
    )
    require(
        changed == EXPECTED_PATHS,
        f"patched checkout file set mismatch: {sorted(changed ^ EXPECTED_PATHS)}",
    )
    for path, expected_blob in EXPECTED_BLOBS.items():
        require(
            git(checkout, "rev-parse", f"{PATCHED_COMMIT}:{path}") == expected_blob,
            f"blob mismatch: {path}",
        )
    verify_test_pins(checkout)

    with tempfile.TemporaryDirectory(prefix="verify-vllm-qkv-") as temp:
        clone = Path(temp) / "vllm"
        clone_result = subprocess.run(
            ["git", "clone", "--quiet", "--shared", str(checkout), str(clone)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(
            clone_result.returncode == 0,
            f"temporary clone failed: {clone_result.stderr.strip()}",
        )
        git(clone, "checkout", "--quiet", BASE_COMMIT)
        apply_result = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=clone,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(
            apply_result.returncode == 0,
            f"patch does not apply to pinned base: {apply_result.stderr.strip()}",
        )
        git(clone, "apply", str(patch))
        for path, expected_blob in EXPECTED_BLOBS.items():
            require(
                git(clone, "hash-object", path) == expected_blob,
                f"applied patch blob mismatch: {path}",
            )
        status_lines = git(clone, "status", "--porcelain").splitlines()
        applied = {line.split(maxsplit=1)[1] for line in status_lines}
        require(
            applied == EXPECTED_PATHS,
            f"applied patch file set mismatch: {sorted(applied ^ EXPECTED_PATHS)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout",
        type=Path,
        default=Path("/tmp/vllm-qkv-consumer-publish"),
    )
    parser.add_argument("--skip-checkout", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    patch = verify_patch(repo_root)
    if not args.skip_checkout:
        verify_checkout(args.checkout, patch)
    print(
        json.dumps(
            {
                "base_commit": BASE_COMMIT,
                "patched_commit": PATCHED_COMMIT,
                "patch_commits": PATCH_COMMITS,
                "patch_sha256": PATCH_SHA256,
                "file_blobs": EXPECTED_BLOBS,
                "focused_test_cases": EXPECTED_TEST_CASES,
                "checkout_verified": not args.skip_checkout,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
