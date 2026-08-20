#!/usr/bin/env python3

import ast
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


BASE_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"
PATCHED_COMMIT = "dce9cfabde438512d946814645881fc7f9c127d0"
PATCH_COMMITS = (
    "d1dc0a3c8d224b880fc10886f39aa8d897640a08",
    "3777614a5af305d8e2db4bab87f940983e9c71e0",
    "0f92b6f4c80e24449d2cb3e36152b6e4d3e9ac3d",
    PATCHED_COMMIT,
)
PATCH_SHA256 = "06c9546676c211b0da20b9ed47eecfa4d1c31771a90ff3e4a6653ee30a9d1ac8"
SCHEMA = "exl3_qkv_topology/1"
COMPONENTS = ["q_proj", "k_proj", "v_proj"]
PATCH_NAME = "exllamav3-5f3c537-qkv-topology-poc.patch"
EXPECTED_PATHS = {
    "exllamav3/conversion/compile.py",
    "exllamav3/conversion/convert_model.py",
    "exllamav3/conversion/qkv_topology.py",
    "exllamav3/conversion/quant_config.py",
    "exllamav3/model/config.py",
    "exllamav3/modules/attn.py",
    "tests/test_qkv_topology.py",
}
EXPECTED_BLOBS = {
    "exllamav3/conversion/compile.py": "54bc7b3a6bb874b38715542ff779a3051137bf88",
    "exllamav3/conversion/convert_model.py": "3fa9e1236cccf4184b820c538073cd7c3b19a5e6",
    "exllamav3/conversion/qkv_topology.py": "6daf9ab574a4dcb5bec146ab25946d6758ba443b",
    "exllamav3/conversion/quant_config.py": "42cc866306325dbd49e20b59a17226a81465a3f7",
    "exllamav3/model/config.py": "dcb5aa4ee799f8c0329358b67cf2b3ae0bc716d1",
    "exllamav3/modules/attn.py": "ca27dbb00c35c5d1e3ec539a47987c79f434b873",
    "tests/test_qkv_topology.py": "11aaa955010797972d36bb157b0c05eb3144b093",
}
EXPECTED_TEST_FUNCTIONS = {
    "test_qwen35_interleaved_gate_uses_actual_doubled_q_projection_width",
    "test_fused_qkv_forward_deinterleaves_qwen_query_gate_payload",
    "test_fused_qkv_fast_deinterleave_receives_contiguous_qg",
    "test_topology_setup_is_opt_in_and_does_not_scan_existing_models_without_a_plan",
    "test_opted_in_split_metadata_rejects_incompatible_projection_domain",
    "test_target_topology_excludes_mtp_and_vision_side_models",
    "test_mixed_layer_map_is_sorted_complete_and_defaults_to_split",
    "test_attention_construction_rebuilds_mixed_logical_payloads_from_metadata",
    "test_bf16_concatenation_and_split_reconstruct_exact_bits",
    "test_fused_uniform_requires_one_common_K_codebook_and_scale",
    "test_plan_rejects_K_outside_compiled_qkv_domain",
    "test_plan_and_metadata_reject_codebooks_outside_compiled_qkv_domain",
    "test_metadata_rejects_K_outside_compiled_qkv_domain",
    "test_index_reconstruction_rejects_missing_or_duplicate_payloads",
    "test_plan_rejects_unknown_layers_and_duplicate_declarations",
}
EXPECTED_TEST_CASES = 23


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def git(checkout, *args):
    result = subprocess.run(
        ["git", *args],
        cwd = checkout,
        text = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    if result.returncode:
        raise VerificationError(
            f"git {' '.join(args)} failed in {checkout}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def validate_projection(projection, where):
    require(isinstance(projection, dict), f"{where}: projection must be an object")
    require(
        set(projection) == {"name", "K", "codebook", "scale"},
        f"{where}: projection fields changed",
    )
    require(
        isinstance(projection["K"], int)
        and not isinstance(projection["K"], bool)
        and 3 <= projection["K"] <= 8,
        f"{where}: K must be one integer from 3 through 8",
    )
    require(projection["codebook"] in {"mul1", "mcg"}, f"{where}: invalid codebook")


def validate_topology(topology, tensor_storage, tensor_keys):
    require(isinstance(topology, dict), "topology must be an object")
    require(set(topology) == {"schema", "layers"}, "topology root fields changed")
    require(topology["schema"] == SCHEMA, "topology schema mismatch")
    require(isinstance(topology["layers"], list), "topology layers must be a list")

    names = []
    variants = set()
    keys = set(tensor_keys)
    for row in topology["layers"]:
        require(isinstance(row, dict), "layer declaration must be an object")
        common = {"layer", "variant", "components", "output_splits"}
        layer = row.get("layer")
        require(isinstance(layer, str) and layer, "layer name is missing")
        require(layer not in names, f"duplicate layer declaration: {layer}")
        names.append(layer)
        require(row.get("components") == COMPONENTS, f"{layer}: component order changed")
        splits = row.get("output_splits")
        require(
            isinstance(splits, list)
            and len(splits) == 3
            and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in splits),
            f"{layer}: invalid output_splits",
        )

        variant = row.get("variant")
        variants.add(variant)
        if variant == "split":
            require(set(row) == common | {"projections"}, f"{layer}: invalid split fields")
            declarations = row["projections"]
            require(
                isinstance(declarations, list)
                and [p.get("name") for p in declarations] == COMPONENTS,
                f"{layer}: split projections changed order",
            )
            forbidden_prefixes = [layer + ".qkv_proj."]
        elif variant == "fused_uniform":
            require(set(row) == common | {"projection"}, f"{layer}: invalid fused fields")
            declarations = [row["projection"]]
            require(declarations[0].get("name") == "qkv_proj", f"{layer}: fused name changed")
            forbidden_prefixes = [layer + "." + name + "." for name in COMPONENTS]
        else:
            raise VerificationError(f"{layer}: invalid variant {variant!r}")

        for declaration in declarations:
            validate_projection(declaration, layer)
            logical_key = layer + "." + declaration["name"]
            require(logical_key in tensor_storage, f"missing tensor_storage entry: {logical_key}")
            storage = tensor_storage[logical_key]
            for field in ("bits_per_weight", "codebook", "scale"):
                require(storage.get(field) == declaration[field if field != "bits_per_weight" else "K"],
                        f"{logical_key}: tensor_storage {field} mismatch")
            if declaration["codebook"] == "mul1":
                require("mul1_multiplier" in storage, f"{logical_key}: missing mul1 marker")
            elif declaration["codebook"] == "mcg":
                require("mcg_multiplier" in storage, f"{logical_key}: missing mcg marker")
            require(logical_key + ".trellis" in keys, f"missing trellis payload: {logical_key}")

        duplicates = sorted(key for key in keys if any(key.startswith(prefix) for prefix in forbidden_prefixes))
        require(not duplicates, f"{layer}: duplicate split/fused payloads: {duplicates}")

    require(names == sorted(names), "layer declarations are not deterministic")
    return variants


def verify_schema_invariants():
    split_layer = "model.layers.0.self_attn"
    fused_layer = "model.layers.1.self_attn"
    split_projections = [
        {"name": "q_proj", "K": 6, "codebook": "mul1", "scale": "always"},
        {"name": "k_proj", "K": 5, "codebook": "mul1", "scale": "always"},
        {"name": "v_proj", "K": 4, "codebook": "mul1", "scale": "always"},
    ]
    fused_projection = {"name": "qkv_proj", "K": 6, "codebook": "mul1", "scale": "always"}
    topology = {
        "schema": SCHEMA,
        "layers": [
            {
                "layer": split_layer,
                "variant": "split",
                "components": COMPONENTS,
                "output_splits": [12288, 1024, 1024],
                "projections": split_projections,
            },
            {
                "layer": fused_layer,
                "variant": "fused_uniform",
                "components": COMPONENTS,
                "output_splits": [12288, 1024, 1024],
                "projection": fused_projection,
            },
        ],
    }
    declarations = [
        (split_layer, projection) for projection in split_projections
    ] + [(fused_layer, fused_projection)]
    tensor_storage = {
        layer + "." + projection["name"]: {
            "bits_per_weight": projection["K"],
            "codebook": projection["codebook"],
            "scale": projection["scale"],
            "mul1_multiplier": 1,
        }
        for layer, projection in declarations
    }
    tensor_keys = {
        logical_key + suffix
        for logical_key in tensor_storage
        for suffix in (".trellis", ".mul1")
    }
    variants = validate_topology(topology, tensor_storage, tensor_keys)
    require(variants == {"split", "fused_uniform"}, "mixed topology invariant was not exercised")


def verify_patch(repo_root):
    patch = repo_root / "patches" / PATCH_NAME
    require(patch.is_file(), f"missing patch: {patch}")
    payload = patch.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == PATCH_SHA256, "patch SHA256 mismatch")
    text = payload.decode("utf8")
    patch_commits = tuple(re.findall(r"^From ([0-9a-f]{40}) Mon Sep 17 00:00:00 2001$", text, re.MULTILINE))
    require(patch_commits == PATCH_COMMITS, "format-patch commit series mismatch")
    touched = set(re.findall(r"^diff --git a/(\S+) b/\S+$", text, re.MULTILINE))
    require(touched == EXPECTED_PATHS, f"unexpected patch paths: {sorted(touched ^ EXPECTED_PATHS)}")
    for marker in (
        'SCHEMA = "exl3_qkv_topology/1"',
        '"fused_uniform"',
        '"qkv_proj"',
        '"output_splits"',
        '"tensor_storage"',
        'validate_payload_index',
        'concatenate_qkv_bf16',
        'test_fused_qkv_forward_deinterleaves_qwen_query_gate_payload',
        'test_fused_qkv_fast_deinterleave_receives_contiguous_qg',
        'resolve_target_topology',
        'test_qwen35_interleaved_gate_uses_actual_doubled_q_projection_width',
        'test_topology_setup_is_opt_in_and_does_not_scan_existing_models_without_a_plan',
        'test_target_topology_excludes_mtp_and_vision_side_models',
        'test_plan_rejects_K_outside_compiled_qkv_domain',
        'test_plan_and_metadata_reject_codebooks_outside_compiled_qkv_domain',
        'test_metadata_rejects_K_outside_compiled_qkv_domain',
        'test_bf16_concatenation_and_split_reconstruct_exact_bits',
        'test_index_reconstruction_rejects_missing_or_duplicate_payloads',
    ):
        require(marker in text, f"patch is missing required marker: {marker}")
    return patch


def verify_test_pins(source):
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }
    require(set(functions) == EXPECTED_TEST_FUNCTIONS, "focused test function set mismatch")
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
                values = ast.literal_eval(decorator.args[1])
                multiplicity *= len(values)
        cases += multiplicity
    require(cases == EXPECTED_TEST_CASES, f"focused test case count mismatch: {cases}")


def verify_checkout(checkout, patch):
    checkout = checkout.resolve()
    require((checkout / ".git").exists(), f"not a git checkout: {checkout}")
    require(git(checkout, "rev-parse", PATCHED_COMMIT) == PATCHED_COMMIT, "patched commit object is unavailable")
    require(
        tuple(git(checkout, "rev-list", "--reverse", f"{BASE_COMMIT}..{PATCHED_COMMIT}").splitlines())
        == PATCH_COMMITS,
        "patched commit series mismatch",
    )
    require(
        git(checkout, "rev-parse", f"{PATCHED_COMMIT}~{len(PATCH_COMMITS)}") == BASE_COMMIT,
        "patched commit base mismatch",
    )
    changed = set(git(checkout, "diff", "--name-only", BASE_COMMIT, PATCHED_COMMIT).splitlines())
    require(changed == EXPECTED_PATHS, f"patched commit file set mismatch: {sorted(changed ^ EXPECTED_PATHS)}")
    for path, expected_blob in EXPECTED_BLOBS.items():
        require(
            git(checkout, "rev-parse", f"{PATCHED_COMMIT}:{path}") == expected_blob,
            f"blob mismatch: {path}",
        )
    verify_test_pins(git(checkout, "show", f"{PATCHED_COMMIT}:tests/test_qkv_topology.py"))

    with tempfile.TemporaryDirectory(prefix = "verify-exllamav3-qkv-") as temp:
        clone = Path(temp) / "exllamav3"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", str(checkout), str(clone)],
            check = True,
        )
        git(clone, "checkout", "--quiet", BASE_COMMIT)
        result = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd = clone,
            text = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        require(result.returncode == 0, f"patch does not apply to pinned base: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type = Path, default = Path("/tmp/exllamav3-qkv-poc"))
    parser.add_argument("--skip-checkout", action = "store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    patch = verify_patch(repo_root)
    verify_schema_invariants()
    if not args.skip_checkout:
        verify_checkout(args.checkout, patch)
    print(json.dumps({
        "schema": SCHEMA,
        "base_commit": BASE_COMMIT,
        "patched_commit": PATCHED_COMMIT,
        "patch_sha256": PATCH_SHA256,
        "patch_commits": PATCH_COMMITS,
        "file_blobs": EXPECTED_BLOBS,
        "focused_test_cases": EXPECTED_TEST_CASES,
        "checkout_verified": not args.skip_checkout,
    }, sort_keys = True))


if __name__ == "__main__":
    main()
