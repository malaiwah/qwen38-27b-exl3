#!/usr/bin/env python3
"""Metadata-only, fail-closed audit of the pinned Intel AutoRound checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn, cast

from frontier_common import atomic_write_json, canonical_sha256

SCHEMA = "qwen38-frontier-autoround-audit/1"
REPOSITORY = "Intel/Qwen3.8-27B-bpw2.8-AutoRound"
REVISION = "03a2e36af5fad7b8eb281ff27bfb081e6216a257"
V2_REVISION = "d10a7bedf41bace1b57c9902e4898a5ef47f157b"
V1_REVISION = "67565e1fe1bf838ec1dfefbf1b4e93f7dfcd7023"
INITIAL_UPLOAD = "8d7d1c2f65ec1f7d88bbe565f9fe20c17e95907b"
VLLM_HEAD = "b3da8bb8934667d42446614521bc234eaf24f192"
VLLM_PARENT = "5a4c8d99242e9e069b604d0e9b969e77f7dd501d"
HUMMING_COMMIT = "636ba85648c30ae2c2bfb335c9399593a67ecc1d"
HUMMING_WHEEL_SHA256 = "cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"
COMPRESSED_TENSORS_COMMIT = "f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0"
COMPRESSED_TENSORS_WHEEL_SHA256 = "4a1b89b508f7efb8ffb4eee8a6e69e0452d9b080cae130146025c64fbe9fa9aa"
SHA40 = re.compile(r"^[0-9a-f]{40}$")

# name -> (Git blob, byte size, LFS SHA256 or None, Xet reconstruction hash or None)
FILES: dict[str, tuple[str, int, str | None, str | None]] = {
    ".gitattributes": ("52373fe24473b1aa44333d318f578ae6bf04b49b", 1570, None, None),
    "README.md": ("08875279f21de33ece8d3bcff54973742abed41a", 5139, None, None),
    "chat_template.jinja": ("c0c686f9c38d70d179fb7b5f5aa7530bc913dda3", 8952, None, None),
    "config.json": ("8eb77bb65755d990a374ee55d1ef2e98910b50b7", 43371, None, None),
    "generation_config.json": ("0bc3addd19dc59c5c8899fc1fb887d50b592e7c3", 214, None, None),
    "model-00001-of-00005.safetensors": ("fba446b040334a6b2d93a446dfb22dd12271d864", 3203175128, "86429db46ffd8fc4237b40eb3113561a2a8709d6420f9bf7fc4b86311d0618ad", "a618421233b8cb22db138a6313e3bae056cf1668085e2e9faa665401ceef706d"),
    "model-00002-of-00005.safetensors": ("6adc64eec91ca887c6dfa4af3f105721bec6d567", 3199032200, "99c4e451dfdbdb914d32230da7cc5ed9f46149ad41971eab91875f683f3e2673", "04236f7ce1b8ccc0ea1236a33e69568e8d71c39bd599f8e7aadfcef56d467ab4"),
    "model-00003-of-00005.safetensors": ("d200a316928b1eb9c35b3b0f1d0ff0ee390d983f", 3047399664, "6ee41c7051d0e923e3882d42c0ccafa01cac57758f0b0fab2fce0eabab1c4962", "128eeb0ae51b2240415498719f0fdd62ac2f2c75a0d22b89c740b94bc6f89654"),
    "model-00004-of-00005.safetensors": ("8b3457f1507ee33dd0c01355d4034bea641a03ed", 2542807272, "55a14ee79d3e5a65a8731d89426f4df477e8bdc7daa7976d41254d8afb9432f0", "4acebe250b52ba20bc44921850de816a098c5de24486944aa5601a1114741f55"),
    "model-00005-of-00005.safetensors": ("d178738597c6a2dd538230062e08d4f47547eabd", 2542796896, "6866cf8adcccc4cc6a00e74bc025f1a774fb52103b70f2674d0288272951a733", "9638b8181ac01da87417839aee0e4e1fff3ce0bd772ae82b388e661351ead6a1"),
    "model.safetensors.index.json": ("f5e0efb8909a3a3d859957198f258e628523d3cb", 192181, None, None),
    "model_extra_tensors.safetensors": ("fbf595c943ff2d4c8000b9aa59812d79f6d7da42", 305575976, "850632551dfc245df79bd8950f19bf9b60205b65313793703c1967644d087011", "f78fa1f99d2cf2406f5ef8e08669e039a252d4a0e7b1fff09cfcc071f9c78e0a"),
    "preprocessor_config.json": ("8ed39680d90d989c35a3e308338a24875bafbc42", 443, None, None),
    "processor_config.json": ("33818c7f9e991ad735fd240209f4fa73e6c28c50", 1191, None, None),
    "quantization_config.json": ("98d2089656b704c25bc0462027eb5426e160bd49", 36790, None, None),
    "tokenizer.json": ("5520bfd2dd834ce386c1312c410fa71af56db5ad", 19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", "777bcaa63794fa47b8f53680be9d6d176f1fcbd7ba03cdc6c3bae2b3d76b323f"),
    "tokenizer_config.json": ("1d134cd298be1e3be25db393d93a1cefe80e3214", 1165, None, None),
}

HEADER_FACTS = {
    "model-00001-of-00005.safetensors": (74576, 3203100544, "ecd834677ed1d0334a1a4c839923e2db52f07c210b0afb060fb4aba5e226a6d0"),
    "model-00002-of-00005.safetensors": (78272, 3198953920, "ee914a3bdb64ee80de1e70da3e34f582e1fca8978df8ebcae8b9d3abc4b65629"),
    "model-00003-of-00005.safetensors": (99400, 3047300256, "b2a583a2457d0cd4b6b7be3eaf54000163d8cf82c9f40a7a9e427a67039f05b4"),
    "model-00004-of-00005.safetensors": (224, 2542807040, "f0e3b9326cf15023153dd20b3a36a20f0378366fce1dfe25257f004091c7d0c9"),
    "model-00005-of-00005.safetensors": (88, 2542796800, "6d00112c1c76c82d5a169dc9905dc3e8e2f4a65cad8bdf1f90a250cc9935640c"),
    "model_extra_tensors.safetensors": (3104, 305572864, "d6d972f7606652f3dc8c5981ed2d19a55a639c187aa0de232d9f17ead756f295"),
}

V1_CHANGED_LFS = {
    "model-00001-of-00005.safetensors": "49e93f726b87f8beef7ed8257d88c8d2d9d3666e3874059f88692a6dab789778",
    "model-00002-of-00005.safetensors": "28ec3b6808b3e2a5601fe9a1900cc8b5ad55841a0be1c8ef7983db0214db5efc",
    "model-00003-of-00005.safetensors": "7eacfbf5a02e16c577c6d04fe9d66f343432a093de894248c1236990a385050e",
}


class AuditError(ValueError):
    """A pinned fact or fail-closed network-policy violation."""


def fail(message: str, code: int = 2) -> NoReturn:
    print(f"frontier_autoround_audit.py: {message}", file=sys.stderr)
    raise SystemExit(code)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def strict_json(raw: bytes, where: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(AuditError(f"{where}: non-finite {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{where}: invalid JSON: {exc}") from exc


def object_at(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{where} must be an object")
    return value


def auth_headers() -> dict[str, str]:
    headers = {"Accept-Encoding": "identity", "User-Agent": "qwen38-frontier-autoround-audit/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def request_bytes(url: str, timeout: float, *, byte_range: tuple[int, int] | None = None, maximum: int) -> tuple[bytes, Any, int, str]:
    headers = auth_headers()
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            final_url = response.geturl()
            if byte_range is not None:
                start, end = byte_range
                require(status == 206, f"range request returned HTTP {status}: {url}")
                require(response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"), f"wrong Content-Range for {url}")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                require(int(declared) <= maximum, f"response exceeds metadata limit for {url}")
            raw = response.read(maximum + 1)
            require(len(raw) <= maximum, f"response exceeds metadata limit for {url}")
            return raw, response.headers, status, final_url
    except urllib.error.URLError as exc:
        raise AuditError(f"request failed for {url}: {exc}") from exc


def redirect_headers(url: str, timeout: float) -> Any:
    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(url, method="HEAD", headers=auth_headers()), timeout=timeout)
    except urllib.error.HTTPError as exc:
        require(exc.code in {302, 303, 307, 308}, f"expected immutable Hub redirect for {url}, got HTTP {exc.code}")
        return exc.headers
    except urllib.error.URLError as exc:
        raise AuditError(f"HEAD failed for {url}: {exc}") from exc
    raise AuditError(f"expected Hub redirect for {url}")


def resolve_url(hub: str, revision: str, name: str) -> str:
    return f"{hub}/{REPOSITORY}/resolve/{revision}/{urllib.parse.quote(name)}"


def api_url(hub: str, revision: str) -> str:
    return f"{hub}/api/models/{REPOSITORY}/revision/{revision}?blobs=true"


def get_json(url: str, timeout: float, maximum: int, where: str) -> Any:
    raw, _, _, _ = request_bytes(url, timeout, maximum=maximum)
    return strict_json(raw, where)


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324: Git object identity


def tensor_bytes(tensor: dict[str, Any]) -> int:
    offsets = tensor.get("data_offsets")
    require(isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(item, int) for item in offsets), "invalid safetensors data_offsets")
    offsets = cast(list[int], offsets)
    require(0 <= offsets[0] <= offsets[1], "non-monotonic safetensors data_offsets")
    return offsets[1] - offsets[0]


def safetensors_header(hub: str, revision: str, name: str, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_length, expected_data, expected_digest = HEADER_FACTS[name]
    url = resolve_url(hub, revision, name)
    prefix, _, _, _ = request_bytes(url, timeout, byte_range=(0, 7), maximum=8)
    require(len(prefix) == 8, f"short safetensors prefix for {name}")
    length = struct.unpack("<Q", prefix)[0]
    require(length == expected_length and length <= 1_048_576, f"unexpected safetensors header length for {name}: {length}")
    raw, _, _, _ = request_bytes(url, timeout, byte_range=(8, 7 + length), maximum=length)
    require(len(raw) == length, f"short safetensors header for {name}")
    require(hashlib.sha256(prefix + raw).hexdigest() == expected_digest, f"header digest mismatch for {name}")
    header = object_at(strict_json(raw, f"{name} header"), f"{name} header")
    tensors = {key: object_at(value, f"{name}:{key}") for key, value in header.items() if key != "__metadata__"}
    ranges = sorted((value["data_offsets"][0], value["data_offsets"][1], key) for key, value in tensors.items())
    cursor = 0
    for start, end, key in ranges:
        require(start == cursor and end >= start, f"non-contiguous tensor {name}:{key}")
        cursor = end
    require(cursor == expected_data, f"tensor-data total mismatch for {name}: {cursor}")
    require(8 + length + cursor == FILES[name][1], f"header/data/file size mismatch for {name}")
    return tensors, {"header_bytes": length, "prefix_and_header_sha256": expected_digest, "tensor_count": len(tensors), "tensor_data_bytes": cursor}


def validate_metadata(metadata: dict[str, Any], revision: str) -> dict[str, dict[str, Any]]:
    require(metadata.get("id") == REPOSITORY and metadata.get("sha") == revision, f"Hub did not resolve exact revision {revision}")
    siblings = metadata.get("siblings")
    require(isinstance(siblings, list), "Hub siblings must be an array")
    siblings = cast(list[Any], siblings)
    result: dict[str, dict[str, Any]] = {}
    for item in siblings:
        sibling = object_at(item, "Hub sibling")
        name = sibling.get("rfilename")
        require(isinstance(name, str) and name not in result, "invalid or duplicate Hub sibling")
        name = cast(str, name)
        result[name] = sibling
    return result


def validate_current_files(hub: str, timeout: float) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    metadata = object_at(get_json(api_url(hub, REVISION), timeout, 2_000_000, "Hub revision metadata"), "Hub revision metadata")
    siblings = validate_metadata(metadata, REVISION)
    require(set(siblings) == set(FILES), f"pinned file set changed: expected {sorted(FILES)}, got {sorted(siblings)}")
    small: dict[str, bytes] = {}
    lfs_rows: list[dict[str, Any]] = []
    for name, (blob, size, lfs_sha, xet_hash) in FILES.items():
        item = siblings[name]
        require(item.get("blobId") == blob and item.get("size") == size, f"Hub identity changed for {name}")
        url = resolve_url(hub, REVISION, name)
        headers = redirect_headers(url, timeout)
        require(headers.get("X-Repo-Commit") == REVISION, f"resolve revision mismatch for {name}")
        if lfs_sha is None:
            require(item.get("lfs") is None, f"unexpected LFS identity for {name}")
            raw, _, _, _ = request_bytes(url, timeout, maximum=size)
            require(len(raw) == size and git_blob_sha1(raw) == blob, f"small-file bytes changed for {name}")
            small[name] = raw
        else:
            lfs = object_at(item.get("lfs"), f"{name}.lfs")
            require(lfs.get("sha256") == lfs_sha and lfs.get("size") == size, f"LFS identity changed for {name}")
            require(headers.get("X-Linked-ETag", "").strip('"') == lfs_sha, f"linked LFS identity mismatch for {name}")
            require(headers.get("X-Linked-Size") == str(size), f"linked LFS size mismatch for {name}")
            require(headers.get("X-Xet-Hash") == xet_hash, f"Xet identity mismatch for {name}")
            lfs_rows.append({"path": name, "bytes": size, "git_blob": blob, "lfs_sha256": lfs_sha, "xet_hash": xet_hash})
    return small, lfs_rows


def validate_history(hub: str, timeout: float) -> dict[str, Any]:
    url = f"{hub}/api/models/{REPOSITORY}/commits/{REVISION}"
    history = get_json(url, timeout, 1_000_000, "pinned Hub history")
    require(isinstance(history, list), "Hub commit history must be an array")
    ids = [object_at(item, "history item").get("id") for item in history]
    require(bool(ids) and ids[0] == REVISION, "history is not anchored at the pinned revision")
    require(V2_REVISION in ids and V1_REVISION in ids and INITIAL_UPLOAD in ids, "required v1/v2/upload history is missing")
    require(ids.index(V2_REVISION) < ids.index(V1_REVISION) < ids.index(INITIAL_UPLOAD), "v1/v2 history order changed")
    v1 = object_at(get_json(api_url(hub, V1_REVISION), timeout, 2_000_000, "v1 metadata"), "v1 metadata")
    v2 = object_at(get_json(api_url(hub, V2_REVISION), timeout, 2_000_000, "v2 metadata"), "v2 metadata")
    v1_files = validate_metadata(v1, V1_REVISION)
    v2_files = validate_metadata(v2, V2_REVISION)
    for name, old_sha in V1_CHANGED_LFS.items():
        require(object_at(v1_files[name].get("lfs"), f"v1 {name}.lfs").get("sha256") == old_sha, f"v1 LFS identity changed for {name}")
        require(object_at(v2_files[name].get("lfs"), f"v2 {name}.lfs").get("sha256") == FILES[name][2], f"v2 LFS identity changed for {name}")
    for name in set(HEADER_FACTS) - set(V1_CHANGED_LFS):
        require(object_at(v1_files[name].get("lfs"), f"v1 {name}.lfs").get("sha256") == FILES[name][2], f"unchanged v1 identity changed for {name}")
    return {"initial_upload": INITIAL_UPLOAD, "v1": V1_REVISION, "v2_tensor_commit": V2_REVISION, "audited_tip": REVISION, "v1_to_v2_changed_payloads": sorted(V1_CHANGED_LFS)}


def shape(tensor: dict[str, Any], expected: list[int], dtype: str, where: str) -> None:
    require(tensor.get("shape") == expected and tensor.get("dtype") == dtype, f"unexpected tensor shape/dtype for {where}")


def audit_quantization(config: dict[str, Any], sidecar: dict[str, Any], index: dict[str, Any], main: dict[str, dict[str, Any]], extra: dict[str, dict[str, Any]]) -> dict[str, Any]:
    embedded = object_at(config.get("quantization_config"), "config.quantization_config")
    require(config.get("dtype") == "bfloat16" and config.get("tie_word_embeddings") is False, "top-level BF16/untied declaration changed")
    require(config.get("language_model_only") is False, "multimodal declaration changed")
    require(object_at(config.get("vision_config"), "vision_config").get("dtype") == "bfloat16", "vision dtype declaration changed")
    for doc, where in ((embedded, "embedded"), (sidecar, "sidecar")):
        require(doc.get("bits") == 2 and doc.get("group_size") == 64 and doc.get("sym") is True, f"{where} global W2/G64 symmetric config changed")
        require(doc.get("quant_method") == "auto-round" and doc.get("packing_format") == "auto_round:auto_gptq", f"{where} format changed")
        require(doc.get("autoround_version") == "0.15.0", f"{where} generator version changed")
    require(embedded.get("block_name_to_quantize") == ["model.language_model.layers", "mtp.layers"], "embedded block list changed")
    require(sidecar.get("block_name_to_quantize") == "model.language_model.layers", "sidecar block list changed")
    embedded_extra = object_at(embedded.get("extra_config"), "embedded.extra_config")
    sidecar_extra = object_at(sidecar.get("extra_config"), "sidecar.extra_config")
    require(embedded_extra.get("mtp.fc") == {"bits": 16, "data_type": "fp"}, "embedded MTP fc declaration changed")
    require(not any(key.startswith("mtp.") for key in sidecar_extra), "sidecar unexpectedly describes MTP")

    modules = sorted(key[:-8] for key in main if key.endswith(".qweight"))
    require(len(modules) == 400, "indexed language quantized-linear count changed")
    counts: Counter[tuple[int, int]] = Counter()
    weights: Counter[tuple[int, int]] = Counter()
    stored: Counter[tuple[int, int]] = Counter()
    for prefix in modules:
        override = object_at(embedded_extra.get(prefix, {}), f"override {prefix}")
        bits = override.get("bits", embedded["bits"])
        group = override.get("group_size", embedded["group_size"])
        require((bits, group) in {(2, 64), (3, 128)}, f"unexpected language scheme for {prefix}")
        qweight = main[prefix + ".qweight"]
        qzeros = main[prefix + ".qzeros"]
        scales = main[prefix + ".scales"]
        require(qweight.get("dtype") == "I32" and qzeros.get("dtype") == "I32" and scales.get("dtype") == "F16", f"packed dtypes changed for {prefix}")
        qshape = qweight.get("shape")
        require(isinstance(qshape, list) and len(qshape) == 2 and all(isinstance(item, int) for item in qshape), f"bad qweight shape for {prefix}")
        qshape = cast(list[int], qshape)
        require((qshape[0] * 32) % bits == 0, f"non-integral packed K for {prefix}")
        original = qshape[0] * 32 // bits * qshape[1]
        counts[(bits, group)] += 1
        weights[(bits, group)] += original
        stored[(bits, group)] += tensor_bytes(qweight) + tensor_bytes(qzeros) + tensor_bytes(scales)
    require(counts == Counter({(2, 64): 155, (3, 128): 245}), f"language scheme counts changed: {counts}")
    require(weights == Counter({(2, 64): 10134487040, (3, 128): 14192476160}), f"language weight counts changed: {weights}")
    require(sum(stored.values()) == 8475427840, "language packed byte total changed")
    actual_bpw = sum(stored.values()) * 8 / sum(weights.values())
    require(abs(actual_bpw - 2.7871716729525864) < 1e-15, "stored language bpw changed")

    passthrough = [key for key, value in main.items() if (key.endswith(".in_proj_a.weight") or key.endswith(".in_proj_b.weight")) and value.get("dtype") == "BF16"]
    require(len(passthrough) == 96, "BF16 in_proj_a/in_proj_b count changed")
    vision = {key: value for key, value in main.items() if key.startswith("model.visual.")}
    require(len(vision) == 333 and all(value.get("dtype") == "BF16" for value in vision.values()), "BF16 vision inventory changed")
    vision_parameters = sum(math.prod(value["shape"]) for value in vision.values())
    require(vision_parameters == 460730096, "vision parameter total changed")
    shape(main["model.language_model.embed_tokens.weight"], [248320, 5120], "BF16", "embed_tokens")
    shape(main["lm_head.weight"], [248320, 5120], "BF16", "lm_head")

    mtp_modules = sorted(key[:-8] for key in extra if key.endswith(".qweight"))
    require(len(mtp_modules) == 7, "MTP quantized-linear count changed")
    mtp_weights = 0
    mtp_stored = 0
    for prefix in mtp_modules:
        qweight = extra[prefix + ".qweight"]
        qzeros = extra[prefix + ".qzeros"]
        scales = extra[prefix + ".scales"]
        require(qweight.get("dtype") == "I32" and qzeros.get("dtype") == "I32" and scales.get("dtype") == "F16", f"MTP packed dtype changed for {prefix}")
        qshape = qweight.get("shape")
        require(isinstance(qshape, list) and len(qshape) == 2, f"bad MTP qweight shape for {prefix}")
        qshape = cast(list[int], qshape)
        k = qshape[0] * 8  # W4 packs eight values per I32 row.
        require(scales.get("shape") == [k // 64, qshape[1]], f"MTP is no longer W4/G64 at {prefix}")
        mtp_weights += k * qshape[1]
        mtp_stored += tensor_bytes(qweight) + tensor_bytes(qzeros) + tensor_bytes(scales)
    require(mtp_weights == 372244480 and mtp_stored == 200663040, "MTP stored weight/byte totals changed")
    require(mtp_stored * 8 / mtp_weights == 4.3125, "MTP stored bpw changed")
    shape(extra["mtp.fc.weight"], [5120, 10240], "BF16", "mtp.fc")

    metadata = object_at(index.get("metadata"), "index.metadata")
    weight_map = object_at(index.get("weight_map"), "index.weight_map")
    main_names = {f"model-{number:05d}-of-00005.safetensors" for number in range(1, 6)}
    require(metadata == {"format": "safetensors", "total_shards": 5, "total_parameters": 5283237360, "total_size": 14534958560}, "index metadata changed")
    require(set(weight_map.values()) == main_names | {"model_extra_tensors.safetensors"}, "index shard set changed")
    mtp_index = {key: value for key, value in weight_map.items() if key.startswith("mtp.")}
    require(len(mtp_index) == 29 and set(mtp_index.values()) == {"model_extra_tensors.safetensors"}, "MTP weight_map coverage changed")
    require(set(weight_map) == set(main) | set(extra), "weight_map/header tensor coverage changed")

    return {
        "language_linears": {
            "w2_g64": {"modules": counts[(2, 64)], "weights": weights[(2, 64)], "stored_bytes": stored[(2, 64)]},
            "w3_g128": {"modules": counts[(3, 128)], "weights": weights[(3, 128)], "stored_bytes": stored[(3, 128)]},
            "packed_weight_scale_zero_bytes": sum(stored.values()),
            "actual_stored_bpw": actual_bpw,
        },
        "bf16": {"language_in_proj_a_b_linears": 96, "vision_tensors": 333, "vision_parameters": vision_parameters, "embed_tokens_shape": [248320, 5120], "lm_head_shape": [248320, 5120], "embeddings_tied": False},
        "mtp": {
            "index_key_count": len(mtp_index),
            "index_file": "model_extra_tensors.safetensors",
            "index_metadata_incomplete": True,
            "index_metadata_omits_extra_shard_and_305572864_tensor_data_bytes": True,
            "linears": 7,
            "payload_scheme": "W4A16_G64_symmetric_auto_gptq",
            "weights": mtp_weights,
            "packed_weight_scale_zero_bytes": mtp_stored,
            "actual_stored_bpw": 4.3125,
            "fc": {"dtype": "BF16", "shape": [5120, 10240]},
            "embedded_declaration": "mtp.layers inherits global W2/G64; mtp.fc is BF16",
            "sidecar_declaration": "MTP omitted from block_name_to_quantize and extra_config",
            "published_runtime_trace": "sidecar makes the draft allocate unquantized .weight parameters while the payload supplies qweight/qzeros/scales; substituting embedded config allocates W2 (for q_proj [320,12288]) while payload is W4 ([640,12288])",
            "status": "blocked",
            "blocker": "native MTP needs a derived W4/G64 sidecar and therefore a new artifact identity plus runtime evidence",
        },
    }


def run_audit(hub: str, timeout: float) -> dict[str, Any]:
    require(hub == "https://huggingface.co", "only the canonical HTTPS Hub endpoint is allowed")
    small, lfs = validate_current_files(hub, timeout)
    history = validate_history(hub, timeout)
    config = object_at(strict_json(small["config.json"], "config.json"), "config.json")
    sidecar = object_at(strict_json(small["quantization_config.json"], "quantization_config.json"), "quantization_config.json")
    index = object_at(strict_json(small["model.safetensors.index.json"], "model.safetensors.index.json"), "model.safetensors.index.json")
    headers: dict[str, dict[str, Any]] = {}
    header_facts: list[dict[str, Any]] = []
    for name in HEADER_FACTS:
        tensors, facts = safetensors_header(hub, REVISION, name, timeout)
        headers[name] = tensors
        header_facts.append({"path": name, **facts})
    main = {key: value for name, values in headers.items() if name.startswith("model-") for key, value in values.items()}
    extra = headers["model_extra_tensors.safetensors"]
    quantization = audit_quantization(config, sidecar, index, main, extra)
    main_data = sum(HEADER_FACTS[f"model-{number:05d}-of-00005.safetensors"][1] for number in range(1, 6))
    main_files = sum(FILES[f"model-{number:05d}-of-00005.safetensors"][1] for number in range(1, 6))
    require(main_data == 14534958560 and main_files == 14535211160, "five-shard totals changed")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "artifact": {"repository": REPOSITORY, "revision": REVISION, "immutable_reference": f"{REPOSITORY}@{REVISION}", "files": len(FILES)},
        "network_policy": {"hub_only": True, "large_payload_downloaded": False, "allowed": ["pinned Hub revision metadata", "pinned Hub commit history", "small Git blobs", "LFS/Xet redirect headers", "safetensors byte ranges containing only prefix/header"]},
        "identities": {"lfs_xet": lfs, "small_git_blobs": [{"path": name, "bytes": FILES[name][1], "git_blob": FILES[name][0]} for name in FILES if FILES[name][2] is None]},
        "history": history,
        "safetensors_headers": sorted(header_facts, key=lambda item: item["path"]),
        "five_numbered_shards": {"file_bytes": main_files, "tensor_data_bytes": main_data, "index_metadata_total_size": object_at(index["metadata"], "index.metadata")["total_size"]},
        "quantization": quantization,
        "provenance": {
            "base_model_tag": "Qwen/Qwen3.8-27B",
            "exact_base_revision": None,
            "exact_generator_commit": None,
            "declared_generator_version": "0.15.0",
            "license_file_present": False,
            "resolved_license": None,
            "status": "unresolved",
            "reason": "the pinned tree and card do not establish an immutable base ancestry, an exact AutoRound source commit/calibration identity, or an artifact license",
        },
        "consumer": {
            "vllm_pr": 52890,
            "commit": VLLM_HEAD,
            "parent": VLLM_PARENT,
            "relationship": "one_commit_on_parent",
            "model_card_install_pr_is_incorrect": 52729,
            "torch": "2.13.0",
            "compressed_tensors": {"version": "0.17.0", "commit": COMPRESSED_TENSORS_COMMIT, "wheel_sha256": COMPRESSED_TENSORS_WHEEL_SHA256},
            "humming": {"version": "0.1.12", "commit": HUMMING_COMMIT, "x86_64_wheel_sha256": HUMMING_WHEEL_SHA256},
            "llm_compressor_required_for_serving": False,
        },
        "preliminary_disposition": "static_non_speculative_candidate",
        "runtime": {
            "runnable": False,
            "mtp": "blocked",
            "max_model_len_ceiling": 4096,
            "routes_eligible_for_smoke": ["text", "image", "video"],
            "required_before_runnable": ["actual AIBoss startup receipt", "cold JIT completion", "zero fallback evidence", "served route inventory", "BF16 vision load evidence", "text smoke", "image smoke", "video smoke", "measured peak and steady GPU memory"],
            "native_mtp_claim_allowed": False,
        },
    }
    receipt["canonical_sha256_without_digest_field"] = canonical_sha256(receipt)
    return receipt


def plan() -> dict[str, Any]:
    ranges = []
    for name, (header_bytes, _, _) in HEADER_FACTS.items():
        ranges.extend([{"path": name, "range": "bytes=0-7"}, {"path": name, "range": f"bytes=8-{7 + header_bytes}"}])
    return {
        "schema": "qwen38-frontier-autoround-audit-plan/1",
        "artifact": f"{REPOSITORY}@{REVISION}",
        "requests": {"revision_metadata": [REVISION, V2_REVISION, V1_REVISION], "commit_history_anchor": REVISION, "small_files": sorted(name for name, facts in FILES.items() if facts[2] is None), "lfs_redirect_headers": sorted(name for name, facts in FILES.items() if facts[2] is not None), "safetensors_ranges": ranges},
        "large_tensor_payload_downloads": 0,
        "maximum_range_header_bytes": sum(8 + facts[0] for facts in HEADER_FACTS.values()),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="perform the pinned metadata/range audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--hub", default="https://huggingface.co")
    audit.add_argument("--timeout", type=float, default=60.0)
    dry = subparsers.add_parser("plan", help="emit the exact no-download request plan")
    dry.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        value = plan() if args.command == "plan" else run_audit(args.hub.rstrip("/"), args.timeout)
        if args.output:
            atomic_write_json(args.output, value)
        else:
            sys.stdout.buffer.write(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
        return 0
    except (AuditError, OSError, ValueError) as exc:
        fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
