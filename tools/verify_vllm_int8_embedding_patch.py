#!/usr/bin/env python3
"""Verify the pinned, single-purpose vLLM int8 embedding patch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import tempfile
from pathlib import Path

BASE_COMMIT = "fa033bd4e1b16d9d729ad94be2d87da5a13210ce"
FEATURE_COMMIT = "df620ce052cbbff15c4d3a5b30a0a74c9ae554d2"
PATCHED_COMMIT = "342fd7e15f72a9f0f7b5dafded1910edac512b21"
PATCH_NAME = "0001-feat-exl3-add-opt-in-int8-embeddings.patch"
PATCH_SHA256 = "a961b40eda24c481314144f59555e5e32526ae709cd583778dd5bfbbdcd86138"
DEFAULT_CHECKOUT = Path("/tmp/vllm-clean-int8-embedding")
EVIDENCE = "isolated +0.0000650485 [0.0000045718,0.0001318019]"

CHANGED_FILES = (
    "tests/quantization/test_exl3_int8_embedding.py",
    "vllm/model_executor/layers/quantization/exl3.py",
    "vllm/model_executor/models/qwen3_5.py",
    "vllm/model_executor/models/qwen3_5_mtp.py",
    "vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/v1/worker/gpu/spec_decode/eagle/utils.py",
)
BASE_BLOBS = {
    "vllm/model_executor/layers/quantization/exl3.py": (
        "39932667b1da2017e9500f37b7587479b04acb41"
    ),
    "vllm/model_executor/models/qwen3_5.py": (
        "7206e0192bb59dc50d4036b9927dc5836cc8866d"
    ),
    "vllm/model_executor/models/qwen3_5_mtp.py": (
        "5c7122d985657c9dbada83f1557c004ec893ad4c"
    ),
    "vllm/v1/spec_decode/llm_base_proposer.py": (
        "327009bec2e03d2455abaec03dc256ed570865cc"
    ),
    "vllm/v1/worker/gpu/spec_decode/eagle/utils.py": (
        "6aee8a9ce8565632b5a4a46f2a5a9199eb98a8ca"
    ),
}
PATCHED_BLOBS = {
    "tests/quantization/test_exl3_int8_embedding.py": (
        "2b4a4ae058216f595f95b8256f417551c626ab54"
    ),
    "vllm/model_executor/layers/quantization/exl3.py": (
        "b97b98f0479420622f4f95290610e1a9abd2568e"
    ),
    "vllm/model_executor/models/qwen3_5.py": (
        "8177441b5a068137aeb56a981ad28071ba17c9dd"
    ),
    "vllm/model_executor/models/qwen3_5_mtp.py": (
        "485715f76e322775e1f4db50850aa6c926fd5a3f"
    ),
    "vllm/v1/spec_decode/llm_base_proposer.py": (
        "98b9478a88628f1cc4110d4107a376386acca397"
    ),
    "vllm/v1/worker/gpu/spec_decode/eagle/utils.py": (
        "40c42d1d453c1f5c8b24d87cabf0d052da410da9"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def git(checkout: Path, *args: str) -> str:
    return run("git", *args, cwd=checkout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_commit_and_blobs(checkout: Path) -> None:
    require(git(checkout, "rev-parse", "HEAD") == PATCHED_COMMIT, "wrong HEAD commit")
    require(
        git(checkout, "rev-parse", f"{PATCHED_COMMIT}^") == FEATURE_COMMIT,
        "review-fix commit is not based on the pinned feature commit",
    )
    require(
        git(checkout, "rev-parse", f"{FEATURE_COMMIT}^") == BASE_COMMIT,
        "feature commit is not based on the pinned base",
    )
    require(not git(checkout, "status", "--porcelain"), "checkout is not clean")
    changed = tuple(
        sorted(
            git(
                checkout,
                "diff",
                "--name-only",
                BASE_COMMIT,
                PATCHED_COMMIT,
            ).splitlines()
        )
    )
    require(changed == CHANGED_FILES, f"unexpected changed files: {changed!r}")

    for path, expected in BASE_BLOBS.items():
        actual = git(checkout, "rev-parse", f"{BASE_COMMIT}:{path}")
        require(actual == expected, f"base blob mismatch for {path}")
    for path, expected in PATCHED_BLOBS.items():
        actual = git(checkout, "rev-parse", f"{PATCHED_COMMIT}:{path}")
        require(actual == expected, f"patched blob mismatch for {path}")


def call_keywords(path: Path, class_name: str) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "VocabParallelEmbedding"
    )
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def verify_contract(tree: Path) -> None:
    exl3 = (
        tree / "vllm/model_executor/layers/quantization/exl3.py"
    ).read_text(encoding="utf-8")
    required_source = (
        'os.environ.get("VLLM_EXL3_EMBED_ONLINE_BITS")',
        "if bits != 8:",
        'layer.__class__.__name__ == "VocabParallelEmbedding"',
        "return Exl3Int8EmbeddingMethod()",
        "dtype=torch.int8",
        "dtype=torch.float16",
        "for start in range(0, rows, chunk_rows):",
        "row_max == 0",
        "del layer.weight",
        "torch.empty((0, hidden)",
        "layer.q_weight[input_]",
        "layer.embed_scale[input_]",
        "_require_untied_int8_embedding",
        "incompatible with tied",
    )
    for token in required_source:
        require(token in exl3, f"missing int8 contract token: {token}")
    require("_install_embed_online_hook" not in exl3, "import-time hook is forbidden")

    models = tree / "vllm/model_executor/models"
    for path, class_name in (
        (models / "qwen3_5.py", "Qwen3_5Model"),
        (models / "qwen3_5_mtp.py", "Qwen3_5MultiTokenPredictor"),
    ):
        keywords = call_keywords(path, class_name)
        require("quant_config" in keywords, f"missing quant_config in {class_name}")
        require("prefix" in keywords, f"missing prefix in {class_name}")
        prefix = keywords["prefix"]
        require(
            isinstance(prefix, ast.Call)
            and isinstance(prefix.func, ast.Name)
            and prefix.func.id == "maybe_prefix",
            f"non-native prefix wiring in {class_name}",
        )

    proposer = (
        tree / "vllm/v1/spec_decode/llm_base_proposer.py"
    ).read_text(encoding="utf-8")
    proposer_contract = (
        "elif draft_embed is not None and embeddings_equal(",
        "and embedding_quantization_compatible(",
        "self.model.model.embed_tokens = target_embed_tokens",
    )
    for token in proposer_contract:
        require(token in proposer, f"missing proposer sharing contract: {token}")

    eagle_utils = (
        tree / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py"
    ).read_text(encoding="utf-8")
    eagle_contract = (
        "def embedding_quantization_compatible(",
        "draft_q.shape[-1] == target_q.shape[-1]",
        "def embeddings_equal(",
        "_tensor_equal(draft_q, target_q)",
        'validate_embedding_compatibility=speculative_config.method == "mtp"',
    )
    for token in eagle_contract:
        require(token in eagle_utils, f"missing EAGLE sharing contract: {token}")

    tests_path = tree / "tests/quantization/test_exl3_int8_embedding.py"
    test_tree = ast.parse(tests_path.read_text(encoding="utf-8"))
    tests = {
        node.name
        for node in test_tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    expected_tests = {
        "test_int8_integer_round_trip_is_exact",
        "test_int8_float_round_trip_respects_rowwise_error_bound",
        "test_int8_encoder_handles_zero_rows_and_empty_tables",
        "test_int8_chunking_is_bit_identical",
        "test_quant_method_targets_exact_embedding_type_and_excludes_lm_head",
        "test_quant_method_is_inert_when_unset",
        "test_quantized_embedding_comparison_uses_rows_and_scales",
        "test_native_eagle_shares_only_equal_quantized_embeddings",
        "test_native_mtp_sharing_validates_width_and_quantization",
        "test_embedding_dequantizes_multidimensional_ids_and_zero_rows",
        "test_vocab_parallel_int8_embedding_masks_and_all_reduces",
        "test_tied_embeddings_are_rejected",
    }
    require(tests == expected_tests, f"focused test contract changed: {sorted(tests)!r}")


def verify_patch_application(checkout: Path, patch: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="vllm-int8-verify-") as tmp:
        clone = Path(tmp) / "vllm"
        run(
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--no-checkout",
            str(checkout),
            str(clone),
            cwd=Path(tmp),
        )
        git(clone, "checkout", "--quiet", "--detach", BASE_COMMIT)
        run("git", "apply", "--check", str(patch), cwd=clone)
        run("git", "apply", str(patch), cwd=clone)
        tracked = git(clone, "diff", "--name-only").splitlines()
        untracked = git(
            clone, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        changed = tuple(sorted((*tracked, *untracked)))
        require(changed == CHANGED_FILES, f"applied patch changed {changed!r}")
        for path, expected in PATCHED_BLOBS.items():
            actual = git(clone, "hash-object", path)
            require(actual == expected, f"applied blob mismatch for {path}")
        verify_contract(clone)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "patches" / PATCH_NAME,
    )
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    patch = args.patch.resolve()

    require(checkout.is_dir(), f"checkout not found: {checkout}")
    require(patch.is_file(), f"patch not found: {patch}")
    require(sha256(patch) == PATCH_SHA256, "patch SHA256 mismatch")
    patch_text = patch.read_text(encoding="utf-8")
    require(
        patch_text.startswith("diff --git a/tests/quantization/"),
        "aggregate patch header mismatch",
    )
    verify_commit_and_blobs(checkout)
    verify_patch_application(checkout, patch)

    print(f"verified base={BASE_COMMIT}")
    print(f"verified patched={PATCHED_COMMIT}")
    print(f"verified patch_sha256={PATCH_SHA256}")
    print(f"evidence: {EVIDENCE}")


if __name__ == "__main__":
    main()
