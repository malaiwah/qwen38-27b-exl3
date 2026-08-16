#!/usr/bin/env python3
"""One immutable index row per published checkpoint, built only from receipts.

The four releases (K4, K5K6, hydrated, context) are documented across a dozen
receipts, three model cards and a build log each.  A reader who wants to cite
the collection needs one file that answers, per release: which bytes, from which
base model, through which converter, at which precision per role, how large on
disk, how large resident, on which runtime, and how good -- with a path and a
digest behind every number.  This tool emits that file.

Every value is read *through* the pointer it records.  `Evidence.block()` takes
a receipt path and a field -> RFC 6901 pointer map, resolves each pointer, and
stamps the block with `{path, sha256, pointers}`; a field that cannot be
resolved is an error, never a silent null.  Nulls in the output are therefore
deliberate: they carry a `not_verified` note naming what the repo does not pin.

Three byte counts, never mixed (serialized bytes are not VRAM):

  * `whole_tree_bytes`      -- every file the release publishes, as the release
                               evidence counted it on disk;
  * `immutable_payload_bytes` -- the subset the build receipt treats as defining
                               the artifact (no README, no assets);
  * `tensor_payload_bytes`  -- the sum of the quantization manifest's per-role
                               tensor bytes, when that manifest is in the repo;
  * `resident_weights`      -- measured GPU allocation at load, reported by vLLM
                               in 2-decimal GiB, carried with the rounding
                               interval it implies.  This is the only figure of
                               the four that is memory rather than storage.

Fail-closed rules -- any of these aborts without writing:

  * a referenced receipt is missing, unreadable, not JSON, or carries an
    unexpected `schema`;
  * a recorded pointer does not resolve;
  * the index/shard digests in the release evidence disagree with the build
    receipt's immutable payload or with the published SHA256SUMS;
  * a receipt asserts the sha256 of another repo file and the file no longer
    hashes to it (amendment links, patch modules, amendment evidence files);
  * the headline fidelity numbers in the release evidence disagree with the
    fidelity report they came from (mean, interval, contexts, positions,
    top-1, suite digest), or the report's `candidate_identity.index_sha256`
    disagrees with the artifact;
  * a suite digest does not reproduce from the suite manifest's own ordered
    per-context token digests;
  * a curated per-role precision declaration is not a literal substring of the
    receipt field it cites;
  * the tensor payload exceeds the immutable payload, or the immutable payload
    exceeds the whole tree;
  * the quantization manifest and the build receipt disagree on base model,
    converter or recipe;
  * a number lands in a block that names no receipt, or a null lands in a block
    with no `not_verified` note -- the two properties the index exists to
    guarantee are enforced on the finished rows, not trusted.

`check` re-derives every row from the receipts under `--root` and compares row
digests, so it fails on three independent kinds of tampering: a value edited
inside the index (the self digest breaks), a value edited inside the index with
the self digest recomputed (the row no longer matches the receipts), and a
receipt edited underneath a valid index (same).  It never rewrites the index.

One divergence is expected and is recorded rather than rejected:
`native-mtp-8mp-amendment.json` pins the sha256 of the *pre-amendment*
`release-evidence-context.json`, which was then amended to link the amendment.
See `known_divergences`.

Usage:

    tools/collection_index.py generate --out receipts/collection-index.json
    tools/collection_index.py check receipts/collection-index.json
    tools/collection_index.py check receipts/collection-index.json --root /tmp/tree
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "qwen38-collection-index/1"
RELEASE_EVIDENCE_SCHEMA = "qwen38-release-evidence/1"
BUILD_RECEIPT_SCHEMA = "qwen38-build-receipt/2"
MANIFEST_SCHEMA = "qwen38-quantization-manifest/1"
REPORT_SCHEMA = "qwen38-fidelity-report/1"
CORRECTED_V3_SCHEMA = "qwen38-contamination-corrected-analysis/1"
CORRECTED_V4_SCHEMA = "qwen38-contamination-corrected-qualification/1"
SUITE_SCHEMA_PREFIX = "qwen38-distribution-fidelity/"

GIB = 1 << 30
# vLLM prints "Model loading took N.NN GiB", so a reported figure stands for a
# half-open 0.01 GiB band.  Bytes derived from it carry that band explicitly.
GIB_REPORT_HALF_STEP = 0.005

ROLE_ORDER = ("attention", "mlp_gate", "mlp_up", "mlp_down", "embed",
              "lm_head", "mtp", "vision", "norms_and_small")

# Quantization-manifest role keys behind each index role.  Attention is two
# manifest roles in this architecture (full attention layers and linear
# attention layers), and both are carried in `sub_roles`.
MANIFEST_ROLES = {
    "attention": ("full_attention", "linear_attention"),
    "mlp_gate": ("mlp_gate_proj",),
    "mlp_up": ("mlp_up_proj",),
    "mlp_down": ("mlp_down_proj",),
    "embed": ("embed_tokens",),
    "lm_head": ("lm_head",),
    "mtp": ("mtp_draft",),
    "vision": ("vision_tower",),
    "norms_and_small": ("norms_and_small",),
}

# Curated per-role precision, each entry `(value, verbatim quote)`.  The quote
# must appear literally in the cited receipt field or generation is rejected, so
# the mapping is human-written but never human-asserted.
DECLARED_PRECISION = {
    "k4": {
        "path": "receipts/release-evidence-k4.json",
        "pointer": "/artifact/recipe",
        "roles": {
            "attention": ("BF16 on disk, encoded to K6 at load",
                          "attention BF16 on disk encoded to K6 at load"),
            "mlp_gate": ("K4", "MLP all K4"),
            "mlp_up": ("K4", "MLP all K4"),
            "mlp_down": ("K4", "MLP all K4"),
            "embed": ("BF16", "BF16 MTP/vision/embeddings"),
            "lm_head": ("K6", "lm_head K6"),
            "mtp": ("BF16", "BF16 MTP/vision/embeddings"),
            "vision": ("BF16", "BF16 MTP/vision/embeddings"),
            "norms_and_small": (None, None),
        },
    },
    "k5k6": {
        "path": "receipts/k5k6-build-receipt.json",
        "pointer": "/recipe",
        "roles": {
            "attention": ("BF16 on disk, encoded online to K6",
                          "attention BF16 for online K6"),
            "mlp_gate": ("K5", "MLP gate/up K5"),
            "mlp_up": ("K5", "MLP gate/up K5"),
            "mlp_down": ("K6", "down K6"),
            "embed": ("BF16", "embed+vision BF16"),
            "lm_head": ("K6/mcg", "lm_head K6/mcg"),
            "mtp": ("quantized: attention K4, MLP K5/K6",
                    "MTP quantized (attn K4, MLP K5/K6)"),
            "vision": ("BF16", "embed+vision BF16"),
            "norms_and_small": (None, None),
        },
    },
    "hydrated": {
        "path": "receipts/hydrated-build-receipt.json",
        "pointer": "/recipe",
        "roles": {
            "attention": ("K6 serialized (calibrated)",
                          "attention K6 serialized (calibrated)"),
            "mlp_gate": ("K5", "MLP gate/up K5"),
            "mlp_up": ("K5", "MLP gate/up K5"),
            "mlp_down": ("K6", "down K6"),
            "embed": ("BF16", "BF16 embed+vision"),
            "lm_head": ("K6/mcg", "lm_head K6/mcg"),
            "mtp": ("quantized", "quantized MTP"),
            "vision": ("BF16", "BF16 embed+vision"),
            "norms_and_small": (None, None),
        },
    },
    "context": {
        "path": "receipts/context-build-receipt.json",
        "pointer": "/recipe",
        "roles": {
            "attention": ("K5 serialized (calibrated)",
                          "attention K5 serialized (calibrated)"),
            "mlp_gate": ("K5", "MLP gate/up K5"),
            "mlp_up": ("K5", "MLP gate/up K5"),
            "mlp_down": ("K6", "down K6"),
            "embed": ("BF16", "BF16 embeddings and vision"),
            "lm_head": ("K6/mcg", "lm_head K6/mcg"),
            "mtp": ("quantized", "quantized MTP"),
            "vision": ("BF16", "BF16 embeddings and vision"),
            "norms_and_small": (None, None),
        },
    },
}

NO_ROLE_DECLARATION = ("the published recipe does not name norms and other "
                       "small tensors")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_sha256(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace the index atomically, so an interrupted run cannot bless partial work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


class Rejected(SystemExit):
    """Every violation found, reported together rather than one per run.

    The same cross-check can be reached from several releases (one shared suite
    manifest, one shared corrected-metrics receipt), so identical messages are
    collapsed rather than repeated.
    """

    def __init__(self, errors: list[str]) -> None:
        unique = list(dict.fromkeys(errors))
        body = "\n".join(f"  - {e}" for e in unique)
        super().__init__(f"REJECTED: {len(unique)} problem(s):\n{body}")


def gib_bytes(gib: float) -> dict:
    """A 2-decimal GiB report as bytes, with the band that rounding leaves open."""
    lo = (gib - GIB_REPORT_HALF_STEP) * GIB
    hi = (gib + GIB_REPORT_HALF_STEP) * GIB
    return {
        "gib_reported": gib,
        "bytes_approx": round(gib * GIB),
        "bytes_interval": [int(lo), int(hi) + 1],
        "interval_note": ("the source reports 2-decimal GiB; bytes are derived and "
                          "span the +/-0.005 GiB the report leaves open"),
    }


class Evidence:
    """Repo-rooted receipt reader that hashes what it opens and records pointers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.errors: list[str] = []
        self._text: dict[str, bytes] = {}
        self._json: dict[str, object] = {}
        self._sha: dict[str, str] = {}
        self.touched: dict[str, str] = {}

    # -- raw access ---------------------------------------------------------
    def raw(self, rel: str) -> bytes | None:
        if rel in self._text:
            return self._text[rel]
        p = self.root / rel
        try:
            data = p.read_bytes()
        except FileNotFoundError:
            self.errors.append(f"{rel}: missing receipt")
            self._text[rel] = None
            self._sha[rel] = None
            return None
        except OSError as exc:
            self.errors.append(f"{rel}: unreadable ({exc})")
            self._text[rel] = None
            self._sha[rel] = None
            return None
        self._text[rel] = data
        self._sha[rel] = sha256_bytes(data)
        self.touched[rel] = self._sha[rel]
        return data

    def sha(self, rel: str) -> str | None:
        self.raw(rel)
        return self._sha.get(rel)

    def load(self, rel: str, schema: str | None = None,
             schema_prefix: str | None = None) -> object | None:
        if rel not in self._json:
            data = self.raw(rel)
            if data is None:
                self._json[rel] = None
            else:
                try:
                    self._json[rel] = json.loads(data)
                except ValueError as exc:
                    self.errors.append(f"{rel}: not JSON ({exc})")
                    self._json[rel] = None
        doc = self._json[rel]
        if doc is not None and (schema or schema_prefix):
            got = doc.get("schema") if isinstance(doc, dict) else None
            if schema is not None and got != schema:
                self.errors.append(f"{rel}: schema {got!r} != {schema!r}")
            if schema_prefix is not None and not str(got).startswith(schema_prefix):
                self.errors.append(f"{rel}: schema {got!r} is not {schema_prefix}*")
        return doc

    # -- pointer access ----------------------------------------------------
    def pointer(self, rel: str, ptr: str, *, required: bool = True):
        """RFC 6901 lookup.  A missing step is an error unless `required=False`."""
        node = self.load(rel)
        if node is None:
            return None
        if ptr in ("", "/"):
            return node
        for token in ptr.lstrip("/").split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list):
                try:
                    node = node[int(token)]
                    continue
                except (ValueError, IndexError):
                    node = None
            elif isinstance(node, dict) and token in node:
                node = node[token]
                continue
            else:
                node = None
            if node is None:
                break
        if node is None and required:
            self.errors.append(f"{rel}: pointer {ptr} does not resolve")
        return node

    def block(self, rel: str, pointers: dict[str, str], *,
              optional: tuple[str, ...] = ()) -> dict:
        """Fields resolved from one receipt, stamped with its path and digest."""
        out: dict[str, object] = {}
        used: dict[str, str] = {}
        for field, ptr in pointers.items():
            value = self.pointer(rel, ptr, required=field not in optional)
            if value is None and field in optional:
                continue
            out[field] = value
            used[field] = ptr
        out["source"] = {"path": rel, "sha256": self.sha(rel), "pointers": used}
        return out

    # -- assertions --------------------------------------------------------
    def expect(self, ok: bool, message: str) -> bool:
        if not ok:
            self.errors.append(message)
        return ok

    def expect_equal(self, a, b, message: str) -> bool:
        return self.expect(a == b, f"{message} ({json.dumps(a)} != {json.dumps(b)})")

    def expect_file_sha(self, rel: str, claimed: str, claimed_by: str) -> bool:
        got = self.sha(rel)
        return self.expect(
            got == claimed,
            f"{claimed_by} pins {rel} at {claimed} but the file hashes to {got}",
        )


def sha256sums(ev: Evidence, rel: str) -> dict[str, str] | None:
    data = ev.raw(rel)
    if data is None:
        return None
    out = {}
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        out[name.strip()] = digest.strip()
    return out


# ---------------------------------------------------------------------------
# release definitions: which receipts describe which release
# ---------------------------------------------------------------------------

RELEASES = [
    {
        "id": "k4",
        "evidence": "receipts/release-evidence-k4.json",
        "build_receipt": None,
        "manifest": None,
        "sha256sums": None,
        "report_v3": "receipts/v3-report-k4-analysis.json",
        "report_v4": None,
        "corrected_key": "k4",
        "model_card": "MODEL_CARD-K4.md",
    },
    {
        "id": "k5k6",
        "evidence": "receipts/release-evidence-k5k6.json",
        "build_receipt": "receipts/k5k6-build-receipt.json",
        "manifest": None,
        "sha256sums": "receipts/k5k6-SHA256SUMS",
        "report_v3": "receipts/v3-report-v2-analysis.json",
        "report_v4": "receipts/report-v4-k5k6-qual.json",
        "corrected_key": "k5k6",
        "model_card": "MODEL_CARD-K5K6.md",
    },
    {
        "id": "hydrated",
        "evidence": "receipts/release-evidence-hydrated.json",
        "build_receipt": "receipts/hydrated-build-receipt.json",
        "manifest": "receipts/hydrated-quantization-manifest.json",
        "sha256sums": "receipts/hydrated-SHA256SUMS",
        "report_v3": "receipts/v3-report-hyd.json",
        "report_v4": "receipts/report-v4-hyd-qual.json",
        "corrected_key": "hydrated",
        "model_card": "MODEL_CARD-K5K6-hydrated.md",
    },
    {
        "id": "context",
        "evidence": "receipts/release-evidence-context.json",
        "build_receipt": "receipts/context-build-receipt.json",
        "manifest": "receipts/context-quantization-manifest.json",
        "sha256sums": None,
        "report_v3": "receipts/v3-report-ctx.json",
        "report_v4": "receipts/report-v4-ctx-qual.json",
        "corrected_key": "context",
        "model_card": "MODEL_CARD-K5K6-context.md",
    },
]

CORRECTED_V3 = "receipts/analysis-v3-contamination-corrected.json"
CORRECTED_V4 = "receipts/qualification-v4-contamination-corrected.json"
SUITE_V4 = "receipts/v4-suite-manifest.json"
SUITE_KLD5 = "receipts/kld5-suite-manifest.json"
CONTEXT_AMENDMENT = "receipts/native-mtp-8mp-amendment.json"


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------

def artifact_block(ev: Evidence, spec: dict) -> dict:
    rel = spec["evidence"]
    art = ev.block(rel, {
        "hf_repo": "/artifact/hf_repo",
        "label": "/artifact/label",
        "recipe": "/artifact/recipe",
        "index_sha256": "/artifact/index_sha256",
        "shard_sha256": "/artifact/shard_sha256",
    })
    index_sha = art.get("index_sha256")
    shards = art.get("shard_sha256") or {}

    # One digest over the serialized identity of the checkpoint: the index plus
    # the shard map, canonically serialized.  Recipe is documented in
    # `conventions` so a third party can recompute it from the row alone.
    art["shard_map_sha256"] = canonical_sha256(
        {"index_sha256": index_sha, "shard_sha256": shards}
    )

    corroboration = []
    build = spec["build_receipt"]
    if build:
        payload = ev.pointer(build, "/immutable_payload")
        if isinstance(payload, dict):
            ev.expect_equal(payload.get("model.safetensors.index.json"), index_sha,
                            f"{build}: immutable payload index digest vs {rel}")
            for name, digest in sorted(shards.items()):
                ev.expect_equal(payload.get(name), digest,
                                f"{build}: immutable payload {name} vs {rel}")
            corroboration.append({"path": build, "sha256": ev.sha(build),
                                  "covers": "index + all shards", "agrees": True})
    sums_rel = spec["sha256sums"]
    if sums_rel:
        sums = sha256sums(ev, sums_rel) or {}
        ev.expect_equal(sums.get("model.safetensors.index.json"), index_sha,
                        f"{sums_rel}: index digest vs {rel}")
        for name, digest in sorted(shards.items()):
            ev.expect_equal(sums.get(name), digest, f"{sums_rel}: {name} vs {rel}")
        corroboration.append({"path": sums_rel, "sha256": ev.sha(sums_rel),
                              "covers": "index + all shards", "agrees": True})
    art["shard_map_corroboration"] = corroboration
    if not corroboration:
        art["shard_map_not_verified"] = [
            "no build receipt or SHA256SUMS for this release is published in the "
            "repo; the index and shard digests rest on the release evidence alone"
        ]

    art["hf_url"] = f"https://huggingface.co/{art['hf_repo']}"
    return art


def revision_block(ev: Evidence, spec: dict) -> dict:
    """No receipt pins a Hub revision for any artifact; say so with the proof."""
    out = {"revision": None, "not_verified": [
        "no receipt records a Hub revision for this artifact"
    ]}
    report = spec["report_v4"] or spec["report_v3"]
    src = ev.pointer(report, "/candidate_identity/model_revision_source", required=False)
    if src is not None:
        ev.expect_equal(src, "none",
                        f"{report}: candidate_identity.model_revision_source")
        out["evidence_of_absence"] = {
            "path": report, "sha256": ev.sha(report),
            "pointers": {"model_revision": "/candidate_identity/model_revision",
                         "model_revision_source":
                             "/candidate_identity/model_revision_source"},
            "model_revision": ev.pointer(report, "/candidate_identity/model_revision",
                                         required=False),
            "model_revision_source": src,
        }
    else:
        out["not_verified"].append(
            f"{report} predates candidate_identity, so not even the absence of a "
            "revision is recorded for the capture")
    return out


def bytes_block(ev: Evidence, spec: dict, manifest_roles: dict | None) -> dict:
    rel = spec["evidence"]
    out = {
        "note": ("serialized bytes are on-disk bytes and are never VRAM; resident "
                 "weight memory is reported separately under resident_weights"),
        "whole_tree_bytes": None,
        "immutable_payload_bytes": None,
        "tensor_payload_bytes": None,
    }
    disk = ev.pointer(rel, "/artifact/disk_bytes", required=False)
    if disk is not None:
        out["whole_tree_bytes"] = {
            "value": disk, "unit": "bytes",
            "covers": "every published file of the artifact as the release counted it",
            "source": {"path": rel, "sha256": ev.sha(rel),
                       "pointers": {"value": "/artifact/disk_bytes"}},
        }
    else:
        out["whole_tree_not_verified"] = [
            "the release evidence for this artifact records no disk byte count"]

    build = spec["build_receipt"]
    if build:
        payload_bytes = ev.pointer(build, "/immutable_payload_bytes")
        scope = ev.pointer(build, "/scope_note")
        out["immutable_payload_bytes"] = {
            "value": payload_bytes, "unit": "bytes",
            "covers": scope,
            "file_count": ev.pointer(build, "/file_count"),
            "source": {"path": build, "sha256": ev.sha(build),
                       "pointers": {"value": "/immutable_payload_bytes",
                                    "covers": "/scope_note",
                                    "file_count": "/file_count"}},
        }
        if disk is not None and payload_bytes is not None:
            ev.expect(payload_bytes <= disk,
                      f"{build}: immutable payload {payload_bytes} exceeds the "
                      f"whole tree {disk} in {rel}")
    else:
        out["immutable_payload_not_verified"] = [
            "no build receipt for this release is published in the repo"]

    if manifest_roles is not None:
        manifest = spec["manifest"]
        total = sum(int(r["bytes"]) for r in manifest_roles.values())
        out["tensor_payload_bytes"] = {
            "value": total, "unit": "bytes",
            "derivation": ("sum of the quantization manifest's per-role tensor bytes "
                           "over " + ", ".join(sorted(manifest_roles))),
            "source": {"path": manifest, "sha256": ev.sha(manifest),
                       "pointers": {role: f"/roles/{role}/bytes"
                                    for role in sorted(manifest_roles)}},
        }
        payload = out["immutable_payload_bytes"]
        if payload:
            ev.expect(total <= payload["value"],
                      f"{manifest}: tensor payload {total} exceeds the immutable "
                      f"payload {payload['value']}")
    else:
        out["tensor_payload_not_verified"] = [
            "no quantization manifest for this release is published in the repo, so "
            "tensor-payload bytes cannot be derived from per-role tensor sizes"]
    return out


def manifest_role_view(ev: Evidence, spec: dict) -> dict | None:
    """Per-manifest-role formats and bytes, after cross-checking the manifest."""
    manifest = spec["manifest"]
    if manifest is None:
        return None
    doc = ev.load(manifest, schema=MANIFEST_SCHEMA)
    if not isinstance(doc, dict):
        return None
    build = spec["build_receipt"]
    if build:
        for field in ("recipe", "converter", "command"):
            ev.expect_equal(doc.get(field), ev.pointer(build, "/" + field),
                            f"{manifest}: {field} vs {build}")
        for field in ("repo", "revision"):
            ev.expect_equal(doc.get("source", {}).get(field),
                            ev.pointer(build, f"/source/{field}"),
                            f"{manifest}: source.{field} vs {build}")
    roles = doc.get("roles")
    if not isinstance(roles, dict):
        ev.errors.append(f"{manifest}: no roles object")
        return None
    known = {name for names in MANIFEST_ROLES.values() for name in names}
    unknown = sorted(set(roles) - known)
    if unknown:
        ev.errors.append(f"{manifest}: unmapped manifest roles {unknown}")
    return roles


def precision_block(ev: Evidence, spec: dict, roles: dict | None) -> dict:
    rid = spec["id"]
    declared = DECLARED_PRECISION[rid]
    decl_path, decl_ptr = declared["path"], declared["pointer"]
    recipe = ev.pointer(decl_path, decl_ptr)
    recipe_text = recipe if isinstance(recipe, str) else ""

    out: dict[str, object] = {
        "declared_recipe": {
            "text": recipe,
            "source": {"path": decl_path, "sha256": ev.sha(decl_path),
                       "pointers": {"text": decl_ptr}},
        },
        "by_role": {},
    }
    for role in ROLE_ORDER:
        value, quote = declared["roles"][role]
        entry: dict[str, object] = {}
        if quote is None:
            entry["declared"] = None
            entry["not_verified"] = [NO_ROLE_DECLARATION]
        else:
            ev.expect(quote in recipe_text,
                      f"{decl_path}{decl_ptr}: declared {role} precision quote "
                      f"{quote!r} is not a literal substring of the recipe")
            entry["declared"] = value
            entry["declared_quote"] = quote
            entry["declared_source"] = {"path": decl_path, "sha256": ev.sha(decl_path),
                                        "pointer": decl_ptr}
        if roles is not None:
            names = MANIFEST_ROLES[role]
            present = [n for n in names if n in roles]
            for n in names:
                ev.expect(n in roles,
                          f"{spec['manifest']}: role {n} missing for index role {role}")
            formats: dict[str, int] = {}
            for n in present:
                for fmt, count in (roles[n].get("formats") or {}).items():
                    formats[fmt] = formats.get(fmt, 0) + int(count)
            entry["serialized_formats"] = dict(sorted(formats.items()))
            entry["serialized_bytes"] = sum(int(roles[n]["bytes"]) for n in present)
            entry["exl3_modules"] = sum(int(roles[n]["modules"]) for n in present)
            entry["measured_source"] = {
                "path": spec["manifest"], "sha256": ev.sha(spec["manifest"]),
                "pointers": {n: f"/roles/{n}" for n in present},
            }
            if len(present) > 1:
                entry["sub_roles"] = {
                    n: {"serialized_formats": dict(sorted(
                            (roles[n].get("formats") or {}).items())),
                        "serialized_bytes": int(roles[n]["bytes"]),
                        "exl3_modules": int(roles[n]["modules"])}
                    for n in present
                }
        else:
            entry["serialized_formats"] = None
            entry["serialized_bytes"] = None
            entry["exl3_modules"] = None
            entry.setdefault("not_verified", []).append(
                "no quantization manifest in the repo, so the stored formats and "
                "per-role byte sizes of this release are not machine-verifiable here")
        out["by_role"][role] = entry

    if rid == "k4":
        out["online_encoding"] = {
            "note": ("attention ships BF16 and is encoded to K6 at load; the online "
                     "width is a serving knob, not a property of the stored bytes"),
            "source": {"path": spec["evidence"], "sha256": ev.sha(spec["evidence"]),
                       "pointer": "/artifact/recipe"},
        }
    elif rid == "k5k6":
        out["online_encoding"] = ev.block(spec["evidence"],
                                         {"note": "/serving/note"})
    if rid == "context":
        overlay = ev.block(spec["evidence"], {
            "mean_kld": "/evaluation/embedding_overlay/mean_kld",
            "delta_vs_bf16_embedding":
                "/evaluation/embedding_overlay/delta_vs_bf16_embedding",
            "delta_ci95": "/evaluation/embedding_overlay/delta_ci95",
            "paired_contexts": "/evaluation/embedding_overlay/paired_contexts",
        })
        overlay["description"] = ("optional int8 input-embedding overlay applied at "
                                  "runtime; the stored embedding stays BF16")
        overlay["overlay_bits"] = ev.pointer(CONTEXT_AMENDMENT,
                                             "/runtime/embedding_overlay_bits")
        overlay["overlay_bits_source"] = {
            "path": CONTEXT_AMENDMENT, "sha256": ev.sha(CONTEXT_AMENDMENT),
            "pointer": "/runtime/embedding_overlay_bits"}
        overlay["corrected_delta"] = ev.block(CORRECTED_V3, {
            "contexts": "/runtime_overlays/context_int8_input_embedding/contexts",
            "mean_kld": "/runtime_overlays/context_int8_input_embedding/mean_kld",
            "delta_vs_bf16_embedding":
                "/runtime_overlays/context_int8_input_embedding/delta_vs_bf16_embedding",
            "delta_ci95_low":
                "/runtime_overlays/context_int8_input_embedding/delta_ci95/ci95_low",
            "delta_ci95_high":
                "/runtime_overlays/context_int8_input_embedding/delta_ci95/ci95_high",
        })
        out["runtime_overlays"] = {"embed": overlay}
    return out


def resident_block(ev: Evidence, spec: dict) -> dict:
    rel = spec["evidence"]
    rid = spec["id"]
    if rid != "context":
        gib = ev.pointer(rel, "/artifact/resident_weights_gib")
        measurement = gib_bytes(float(gib)) | {
            "configuration": "not stated by the receipt",
            "source": {"path": rel, "sha256": ev.sha(rel),
                       "pointers": {"gib_reported": "/artifact/resident_weights_gib"}},
            "log_corroboration": None,
            "not_verified": ["no vLLM startup log receipt is published in the repo "
                             "for this figure, and the receipt does not state the "
                             "MTP depth or embedding mode it was measured under"],
        }
        return {
            "note": ("measured GPU allocation for weights at load, as vLLM reports "
                     "it; not a serialized size"),
            "measurement": measurement,
            "additional_measurements": [],
        }

    # The context edition is the only release with a log-backed figure.
    log = "receipts/native-mtp-8mp-combined-server.log"
    claimed_log_sha = ev.pointer(CONTEXT_AMENDMENT, f"/evidence_files/{Path(log).name}")
    ev.expect_file_sha(log, claimed_log_sha, f"{CONTEXT_AMENDMENT}/evidence_files")
    gib = ev.pointer(CONTEXT_AMENDMENT, "/measured_memory/resident_weights_gib")
    measurement = gib_bytes(float(gib)) | {
        "configuration": ("int8 input-embedding overlay, MTP depth 3, fp8 KV, "
                          "max_model_len 262144, engine budget 30.24 GiB"),
        "source": {"path": CONTEXT_AMENDMENT, "sha256": ev.sha(CONTEXT_AMENDMENT),
                   "pointers": {
                       "gib_reported": "/measured_memory/resident_weights_gib",
                       "mtp_depth": "/runtime/mtp_depth",
                       "kv_cache_dtype": "/runtime/kv_cache_dtype",
                       "max_model_len": "/runtime/max_model_len",
                       "embedding_overlay_bits": "/runtime/embedding_overlay_bits",
                       "desired_engine_budget_gib": "/runtime/desired_engine_budget_gib",
                   }},
        "log_corroboration": {
            "path": log, "sha256": ev.sha(log),
            "line": ("gpu_model_runner.py:5415] Model loading took 18.41 GiB memory"),
            "pinned_by": {"path": CONTEXT_AMENDMENT,
                          "pointer": f"/evidence_files/{Path(log).name}"},
        },
    }
    variants = []
    for key, config in (
        ("bf16_embedding_mtp_off", "BF16 embeddings, MTP off"),
        ("bf16_embedding_mtp_3", "BF16 embeddings, MTP depth 3"),
        ("int8_embedding_mtp_off", "int8 embedding overlay, MTP off"),
        ("int8_embedding_mtp_1_or_3", "int8 embedding overlay, MTP depth 1 or 3"),
    ):
        ptr = f"/artifact/resident_weights_gib/{key}"
        variants.append(gib_bytes(float(ev.pointer(rel, ptr))) | {
            "configuration": config,
            "source": {"path": rel, "sha256": ev.sha(rel),
                       "pointers": {"gib_reported": ptr}},
        })
    return {
        "note": ("measured GPU allocation for weights at load, as vLLM reports it; "
                 "not a serialized size"),
        "measurement": measurement,
        "additional_measurements": variants,
        "measurement_selection": (
            "the amendment figure is primary because it is the only one corroborated "
            "by a published server log; it sits 0.03 GiB above the release evidence's "
            "int8/MTP figure, which was a separate run"),
    }


def runtime_block(ev: Evidence, spec: dict) -> dict:
    rel = spec["evidence"]
    rid = spec["id"]
    out = ev.block(rel, {
        "image_digest": "/toolchain/vllm_image",
        "gpu": "/hardware/gpu",
        "arch": "/hardware/arch",
        "tensor_parallel": "/hardware/tensor_parallel",
        "driver": "/hardware/driver",
    }, optional=("driver",))

    if rid == "k4":
        out["patch_modules"] = None
        out["patch_prs"] = None
        out["not_verified"] = [
            "the K4 release evidence names the runtime image but no patched-module "
            "digests or patch PRs, so the served code cannot be pinned from it",
            "the image digest was not authenticated against a rootfs manifest",
        ]
        return out

    if rid == "context":
        modules = ev.pointer(rel, "/toolchain/patched_modules_sha256")
        out["patch_modules"] = {
            "digests": modules,
            "source": {"path": rel, "sha256": ev.sha(rel),
                       "pointers": {"digests": "/toolchain/patched_modules_sha256"}},
        }
        current = ev.pointer(rel, "/toolchain/current_patch_files_sha256") or {}
        repo_files = []
        for name, claimed in sorted(current.items()):
            tool_rel = f"tools/{name}"
            ev.expect_file_sha(tool_rel, claimed,
                               f"{rel}/toolchain/current_patch_files_sha256")
            repo_files.append({"path": tool_rel, "sha256": ev.sha(tool_rel)})
        out["patch_sources_in_repo"] = repo_files
        out["patch_prs"] = ev.pointer(rel, "/toolchain/patches")
        out["image_identity_verified_during_capture"] = ev.pointer(
            CONTEXT_AMENDMENT, "/runtime/image_identity_verified_during_capture")
        out["image_identity_limit"] = ev.pointer(
            CONTEXT_AMENDMENT, "/runtime/image_identity_limit")
        out["image_identity_source"] = {
            "path": CONTEXT_AMENDMENT, "sha256": ev.sha(CONTEXT_AMENDMENT),
            "pointers": {
                "image_digest": "/runtime/image",
                "image_identity_verified_during_capture":
                    "/runtime/image_identity_verified_during_capture",
                "image_identity_limit": "/runtime/image_identity_limit"},
        }
        ev.expect_equal(ev.pointer(CONTEXT_AMENDMENT, "/runtime/image"),
                        out["image_digest"],
                        f"{CONTEXT_AMENDMENT}: runtime image vs {rel}")
        return out

    out["patch_modules"] = {
        "digests": {"vllm/model_executor/layers/quantization/exl3.py":
                    ev.pointer(rel, "/toolchain/vllm_patches/patched_module_sha256")},
        "note": ev.pointer(rel, "/toolchain/vllm_patches/note"),
        "source": {"path": rel, "sha256": ev.sha(rel),
                   "pointers": {"digest":
                                "/toolchain/vllm_patches/patched_module_sha256",
                                "note": "/toolchain/vllm_patches/note"}},
    }
    out["patch_prs"] = [ev.pointer(rel, "/toolchain/vllm_patches/graph_decode_pr"),
                        ev.pointer(rel, "/toolchain/vllm_patches/prefill_dispatch_pr")]
    out["not_verified"] = [
        "no published image digest contains these patches; the module must be copied "
        "over the pinned image at launch",
        "the image digest was not authenticated against a rootfs manifest",
    ]
    return out


def fidelity_block(ev: Evidence, spec: dict) -> dict:
    rel = spec["evidence"]
    rid = spec["id"]
    report = spec["report_v3"]
    ev.load(report, schema=REPORT_SCHEMA)

    headline = ev.block(report, {
        "suite_token_sha256": "/suite_token_sha256",
        "partition": "/filter",
        "contexts": "/contexts",
        "scored_positions": "/scored_positions",
        "mean_kld": "/token_mean_kld",
        "median_kld": "/token_median_kld",
        "context_macro_mean_kld": "/context_macro_mean_kld",
        "ci95_low": "/context_bootstrap/ci95_low",
        "ci95_high": "/context_bootstrap/ci95_high",
        "bootstrap_clusters": "/context_bootstrap/clusters",
        "bootstrap_samples": "/context_bootstrap/samples",
        "top1_agreement": "/top1_agreement",
        "reference_head_sha256": "/head_sha256",
    })
    headline["suite"] = ev.pointer(rel, "/evaluation/suite")
    headline["declared_by"] = {
        "path": rel, "sha256": ev.sha(rel),
        "pointers": {"suite": "/evaluation/suite",
                     "mean_kld": "/evaluation/mean_kld",
                     "ci95": "/evaluation/ci95",
                     "contexts": "/evaluation/contexts",
                     "top1_agreement": "/evaluation/top1_agreement",
                     "suite_token_sha256": "/evaluation/suite_token_sha256"},
    }
    # The release evidence and the report must agree exactly; the report is the
    # measurement, the release evidence is the claim.
    for field, ptr in (("mean_kld", "/evaluation/mean_kld"),
                       ("median_kld", "/evaluation/median_kld"),
                       ("contexts", "/evaluation/contexts"),
                       ("top1_agreement", "/evaluation/top1_agreement"),
                       ("suite_token_sha256", "/evaluation/suite_token_sha256"),
                       ("partition", "/evaluation/partition")):
        ev.expect_equal(headline.get(field), ev.pointer(rel, ptr),
                        f"{rel}{ptr} vs {report} {field}")
    ci = ev.pointer(rel, "/evaluation/ci95")
    ev.expect_equal([headline.get("ci95_low"), headline.get("ci95_high")], ci,
                    f"{rel}/evaluation/ci95 vs {report} context_bootstrap")
    positions = ev.pointer(rel, "/evaluation/scored_positions", required=False)
    if positions is not None:
        ev.expect_equal(headline.get("scored_positions"), positions,
                        f"{rel}/evaluation/scored_positions vs {report}")

    identity = {"report": report, "report_sha256": ev.sha(report)}
    report_index = ev.pointer(report, "/candidate_identity/index_sha256",
                              required=False)
    if report_index is None:
        identity["report_index_sha256"] = None
        identity["not_verified"] = [
            f"{report} predates candidate_identity, so this capture is not bound to "
            "the artifact's index digest by the report itself"]
    else:
        identity["report_index_sha256"] = report_index
        identity["agrees_with_artifact_index"] = True
        ev.expect_equal(report_index, ev.pointer(rel, "/artifact/index_sha256"),
                        f"{report}: candidate_identity.index_sha256 vs {rel}")

    suites = [{
        "suite": headline["suite"],
        "suite_token_sha256": headline["suite_token_sha256"],
        "partition": headline["partition"],
        "contexts": headline["contexts"],
        "manifest": None,
        "digest_reproduced_from_manifest": None,
        "not_verified": ["the v3 suite manifest is not published in the repo; the v3 "
                         "suite digest is carried by the reports that used it"],
    }]

    corrected_alternates = []
    v3 = ev.load(CORRECTED_V3, schema=CORRECTED_V3_SCHEMA)
    key = spec["corrected_key"]
    if isinstance(v3, dict):
        ev.expect_equal(v3.get("source_suite_token_sha256"),
                        headline["suite_token_sha256"],
                        f"{CORRECTED_V3}: source suite digest vs {report}")
        ev.expect(key in (v3.get("metrics") or {}),
                  f"{CORRECTED_V3}: no metrics for {key}")
    v3_block = ev.block(CORRECTED_V3, {
        "contexts": f"/metrics/{key}/contexts",
        "mean_kld": f"/metrics/{key}/mean_kld",
        "ci95_low": f"/metrics/{key}/ci95/ci95_low",
        "ci95_high": f"/metrics/{key}/ci95/ci95_high",
        "bootstrap_clusters": f"/metrics/{key}/ci95/clusters",
        "top1_agreement": f"/metrics/{key}/top1_agreement",
        "policy": "/policy",
        "contexts_before": "/analysis_contexts_before",
        "excluded_source_clusters": "/excluded_source_clusters",
    }) | {
        "basis": "v3 analysis partition, calibration-overlap-corrected subset",
        "suite": headline["suite"],
        "suite_token_sha256": headline["suite_token_sha256"],
        "source_disjoint": False,
    }

    report4 = spec["report_v4"]
    v4_block = None
    if report4:
        ev.load(report4, schema=REPORT_SCHEMA)
        v4 = ev.load(CORRECTED_V4, schema=CORRECTED_V4_SCHEMA)
        suite4_token = ev.pointer(report4, "/suite_token_sha256")
        if isinstance(v4, dict):
            ev.expect_equal(v4.get("source_suite_token_sha256"), suite4_token,
                            f"{CORRECTED_V4}: source suite digest vs {report4}")
            ev.expect(key in (v4.get("metrics") or {}),
                      f"{CORRECTED_V4}: no metrics for {key}")
        v4_block = ev.block(CORRECTED_V4, {
            "contexts": f"/metrics/{key}/contexts",
            "mean_kld": f"/metrics/{key}/mean_kld",
            "ci95_low": f"/metrics/{key}/ci95/ci95_low",
            "ci95_high": f"/metrics/{key}/ci95/ci95_high",
            "bootstrap_clusters": f"/metrics/{key}/ci95/clusters",
            "top1_agreement": f"/metrics/{key}/top1_agreement",
            "kld_reduction_vs_official_fp8": f"/metrics/{key}/kld_reduction_vs_fp8",
            "policy": "/policy",
            "contexts_before": "/qualification_contexts_before",
            "excluded_source_clusters": "/excluded_source_clusters",
        }) | {
            "basis": ("v4 qualification partition, source-disjoint from recipe "
                      "selection, calibration-overlap-corrected subset"),
            "suite": ev.pointer(rel, "/evaluation/qualification/suite", required=False)
                     or "qwen38-27b-fidelity-suite-v4",
            "suite_token_sha256": suite4_token,
            "source_disjoint": True,
        }
        # v4 uncorrected qualification, straight from the report.
        corrected_alternates.append(ev.block(report4, {
            "suite_token_sha256": "/suite_token_sha256",
            "partition": "/filter",
            "contexts": "/contexts",
            "scored_positions": "/scored_positions",
            "mean_kld": "/token_mean_kld",
            "ci95_low": "/context_bootstrap/ci95_low",
            "ci95_high": "/context_bootstrap/ci95_high",
            "top1_agreement": "/top1_agreement",
        }) | {"basis": "v4 qualification partition, full set, no overlap correction",
              "source_disjoint": True})
        ev.expect_equal(ev.pointer(report4, "/candidate_identity/index_sha256"),
                        ev.pointer(rel, "/artifact/index_sha256"),
                        f"{report4}: candidate_identity.index_sha256 vs {rel}")

        suite4 = ev.load(SUITE_V4, schema_prefix=SUITE_SCHEMA_PREFIX)
        reproduced = None
        if isinstance(suite4, dict):
            digests = [row.get("token_sha256") for row in suite4.get("context_index", [])]
            reproduced = sha256_bytes("".join(digests).encode())
            ev.expect_equal(reproduced, suite4.get("suite_token_sha256"),
                            f"{SUITE_V4}: suite_token_sha256 vs its ordered context "
                            "token digests")
            ev.expect_equal(suite4.get("suite_token_sha256"), suite4_token,
                            f"{SUITE_V4}: suite digest vs {report4}")
        suites.append({
            "suite": v4_block["suite"],
            "suite_token_sha256": suite4_token,
            "partition": ev.pointer(report4, "/filter"),
            "contexts": ev.pointer(report4, "/contexts"),
            "manifest": {"path": SUITE_V4, "sha256": ev.sha(SUITE_V4),
                         "contexts": ev.pointer(SUITE_V4, "/contexts"),
                         "partitions": ev.pointer(SUITE_V4, "/partitions")},
            "digest_reproduced_from_manifest": reproduced == suite4_token,
            "reproduction": ("sha256 of the concatenated per-context token_sha256 hex "
                            "strings in manifest order"),
        })

    primary = v4_block or v3_block
    if v4_block is not None:
        corrected_alternates.insert(0, v3_block)
    else:
        primary = v3_block | {"not_verified": [
            "no source-disjoint v4 qualification is published for this release, so "
            "the corrected figure comes from the development-set partition"]}

    out = {
        "protocol": ev.pointer(rel, "/evaluation/protocol", required=False)
                    or ("hidden-state replay, exact full-vocabulary two-pass "
                        "KL(BF16 || candidate)"),
        "protocol_source": {"path": report, "sha256": ev.sha(report),
                            "pointers": {"comparator": "/comparator",
                                         "reference": "/reference",
                                         "candidate": "/candidate"}},
        "comparator": ev.pointer(report, "/comparator"),
        "headline": headline,
        "corrected_mean_kld": primary,
        "corrected_alternates": corrected_alternates,
        "suite_digests": suites,
        "capture_identity": identity,
    }
    # Present only when the release evidence states one: an absent optional
    # string is not a gap to annotate, it is a sentence the receipt never wrote.
    caveat = ev.pointer(rel, "/evaluation/development_set_caveat", required=False)
    if caveat is not None:
        out["development_set_caveat"] = caveat
    return out


def amendment_block(ev: Evidence, spec: dict) -> dict:
    """Every amendment link, re-hashed against the file it claims."""
    rel = spec["evidence"]
    node = ev.pointer(rel, "/amendments")
    rows = []
    if isinstance(node, dict):
        items = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        scalars = {k: v for k, v in node.items() if not isinstance(v, dict)}
    elif isinstance(node, list):
        items = [(v.get("scope") or f"[{i}]", v) for i, v in enumerate(node)]
        scalars = {}
    else:
        ev.errors.append(f"{rel}: /amendments is neither object nor list")
        return {"links": [], "scalars": {}}
    for name, entry in items:
        path = entry.get("path")
        if path is None:
            scalars[name] = {k: v for k, v in entry.items()}
            continue
        claimed = entry.get("sha256")
        ev.expect_file_sha(path, claimed, f"{rel}/amendments ({name})")
        link = {"name": name, "path": path, "sha256": ev.sha(path),
                "verified": ev.sha(path) == claimed}
        if entry.get("scope") is not None:
            link["scope"] = entry["scope"]
        rows.append(link)
    return {"links": rows, "scalars": scalars,
            "source": {"path": rel, "sha256": ev.sha(rel),
                       "pointers": {"amendments": "/amendments"}}}


def status_block(ev: Evidence, spec: dict, rank_mean: float,
                 rank: int, best: bool, of: int) -> dict:
    rel = spec["evidence"]
    key = spec["corrected_key"]
    out = {
        "state": "published",
        "label": ev.pointer(rel, "/artifact/label"),
        "published_evidence": {"path": rel, "sha256": ev.sha(rel),
                               "generated_utc": ev.pointer(rel, "/generated_utc"),
                               "amended_utc": ev.pointer(rel, "/amended_utc",
                                                         required=False)},
        "fidelity_rank": {
            "rank": rank,
            "of": of,
            "mean_kld": rank_mean,
            "basis": ("ascending mean KLD on the one corrected basis every release in "
                      "this collection shares: the v3 analysis partition with "
                      "calibration-overlapping source documents excluded.  The "
                      "source-disjoint v4 figures in fidelity.corrected_mean_kld are "
                      "the stronger evidence but do not cover K4."),
            "source": {"path": CORRECTED_V3, "sha256": ev.sha(CORRECTED_V3),
                       "pointers": {"mean_kld": f"/metrics/{key}/mean_kld"}},
            "best_in_collection": best,
        },
        "not_verified": ev.pointer(rel, "/not_verified"),
    }
    note = ev.pointer(rel, "/evaluation/note", required=False)
    if note is not None:
        out["evaluation_note"] = note
    corrections = ev.pointer(rel, "/corrections", required=False)
    if corrections is not None:
        out["corrections"] = corrections
    if spec["id"] == "context":
        out["native_context"] = ev.block(rel, {
            "configured_max_model_len":
                "/memory/int8_embedding_overlay/configured_max_model_len",
            "native_262144_started":
                "/memory/int8_embedding_overlay/native_262144_started",
            "hard_limit_rtx5090_verified":
                "/memory/int8_embedding_overlay/hard_limit_rtx5090_verified",
        })
    if spec["id"] == "k4":
        out["native_context"] = ev.block(rel, {
            "native_262144_on_32gb": "/context/native_262144_on_32gb",
            "kv_tokens_measured": "/context/kv_tokens_measured",
            "measured_by": "/context/measured_by",
        })
    out["model_card"] = {"path": spec["model_card"], "sha256": ev.sha(spec["model_card"])}
    return out


def build_rows(ev: Evidence) -> list[dict]:
    rows = []
    for spec in RELEASES:
        rel = spec["evidence"]
        ev.load(rel, schema=RELEASE_EVIDENCE_SCHEMA)
        if spec["build_receipt"]:
            ev.load(spec["build_receipt"], schema=BUILD_RECEIPT_SCHEMA)
        roles = manifest_role_view(ev, spec)

        base = ev.block(rel, {
            "hf_repo": "/upstream/hf_repo",
            "revision": "/upstream/revision",
            "logical_tensors_verified": "/upstream/logical_tensors_verified",
        }, optional=("logical_tensors_verified",))
        build = spec["build_receipt"]
        if build:
            ev.expect_equal(ev.pointer(build, "/source/repo"), base["hf_repo"],
                            f"{build}: source.repo vs {rel}")
            ev.expect_equal(ev.pointer(build, "/source/revision"), base["revision"],
                            f"{build}: source.revision vs {rel}")
            base["index_sha256"] = ev.pointer(build, "/upstream_check/index_sha256")
            base["index_sha256_source"] = {
                "path": build, "sha256": ev.sha(build),
                "pointer": "/upstream_check/index_sha256"}
        else:
            base["index_sha256"] = None
            base["not_verified"] = [
                "no build receipt for this release is published in the repo, so the "
                "base model's index digest is not pinned for this artifact"]

        converter: dict = {}
        commit = ev.pointer(rel, "/toolchain/exllamav3_commit", required=False)
        if commit is not None:
            converter = {
                "repo": "turboderp-org/exllamav3",
                "commit": commit,
                "declared": ev.pointer(build, "/converter") if build else None,
                "source": {"path": rel, "sha256": ev.sha(rel),
                           "pointers": {"commit": "/toolchain/exllamav3_commit"}},
            }
            if build:
                short = str(converter["declared"] or "")
                ev.expect(commit.startswith(short.split("@")[-1].split(" ")[0]),
                          f"{build}: converter {short!r} is not a prefix of "
                          f"exllamav3_commit {commit} in {rel}")
                converter["declared_source"] = {"path": build, "sha256": ev.sha(build),
                                                "pointer": "/converter"}
        else:
            converter = {
                "repo": None, "commit": None, "declared": None,
                "not_verified": [
                    "the K4 release evidence records no exllamav3 converter commit, "
                    "and no K4 build receipt is published in the repo; "
                    f"{spec['model_card']} names exllamav3 1.4.2 @ 5f3c537 in prose "
                    "but no receipt pins it"],
            }
        converter["research_repo"] = ev.pointer(rel, "/toolchain/research_repo")
        converter["research_commit"] = ev.pointer(rel, "/toolchain/research_commit")
        converter["research_source"] = {
            "path": rel, "sha256": ev.sha(rel),
            "pointers": {"research_repo": "/toolchain/research_repo",
                         "research_commit": "/toolchain/research_commit"}}

        fidelity = fidelity_block(ev, spec)
        row = {
            "id": spec["id"],
            "artifact": artifact_block(ev, spec) | revision_block(ev, spec),
            "base_model": base,
            "converter": converter,
            "precision": precision_block(ev, spec, roles),
            "serialized_bytes": bytes_block(ev, spec, roles),
            "resident_weights": resident_block(ev, spec),
            "runtime": runtime_block(ev, spec),
            "fidelity": fidelity,
            "amendments": amendment_block(ev, spec),
        }
        rows.append(row)

    # Ranked on the one corrected basis all four releases share, so the ordering
    # compares like with like even though K4 has no v4 qualification.
    shared = {}
    for spec in RELEASES:
        shared[spec["id"]] = ev.pointer(
            CORRECTED_V3, f"/metrics/{spec['corrected_key']}/mean_kld")
    order = sorted(rows, key=lambda r: shared[r["id"]])
    for rank, row in enumerate(order, start=1):
        spec = next(s for s in RELEASES if s["id"] == row["id"])
        row["status"] = status_block(ev, spec, shared[row["id"]], rank,
                                     rank == 1, len(rows))
    return rows


def known_divergences(ev: Evidence) -> list[dict]:
    """Documented, structurally unavoidable mismatches -- recorded, not rejected."""
    claimed = ev.pointer(CONTEXT_AMENDMENT, "/base_release_evidence_sha256")
    actual = ev.sha("receipts/release-evidence-context.json")
    if claimed == actual:
        return []
    return [{
        "what": ("receipts/native-mtp-8mp-amendment.json pins the sha256 of the "
                 "release evidence it amends, and that file was then amended to "
                 "link the amendment"),
        "pinned_by": {"path": CONTEXT_AMENDMENT,
                      "pointer": "/base_release_evidence_sha256",
                      "sha256": ev.sha(CONTEXT_AMENDMENT)},
        "claimed_sha256": claimed,
        "current_sha256": actual,
        "path": "receipts/release-evidence-context.json",
        "why_not_fatal": ("the amendment (11:11:08Z) predates the amendment link added "
                          "to the release evidence (amended_utc 12:27:32Z); the "
                          "backreference cannot be satisfied without rewriting a "
                          "published receipt, which this repo never does"),
    }]


def unused_suites(ev: Evidence) -> list[dict]:
    """Suites that exist in the repo but that no row's numbers come from."""
    doc = ev.load(SUITE_KLD5, schema_prefix=SUITE_SCHEMA_PREFIX)
    if not isinstance(doc, dict):
        return []
    digests = [row.get("token_sha256") for row in doc.get("context_index", [])]
    reproduced = sha256_bytes("".join(digests).encode())
    ev.expect_equal(reproduced, doc.get("suite_token_sha256"),
                    f"{SUITE_KLD5}: suite_token_sha256 vs its ordered context token "
                    "digests")
    return [{
        "label": ("the 10.48M-position ladder suite; the manifest declares a schema, "
                  "not a suite name"),
        "manifest": {"path": SUITE_KLD5, "sha256": ev.sha(SUITE_KLD5),
                     "schema": doc.get("schema")},
        "suite_token_sha256": doc.get("suite_token_sha256"),
        "digest_reproduced_from_manifest": reproduced == doc.get("suite_token_sha256"),
        "contexts": doc.get("contexts"),
        "total_scored_positions": doc.get("total_scored_positions"),
        "used_by_rows": [],
        "note": ("no row's fidelity numbers come from this suite; ladder results will "
                 "be separate receipts"),
    }]


# ---------------------------------------------------------------------------
# non-release rows: the receipts that qualify or decide
# ---------------------------------------------------------------------------

QUALIFICATION_CONTEXT = "receipts/qualification-5090-context.json"
QUALIFICATION_24GIB = "receipts/qualification-24gib-capped.json"
VRAM_CLASS_VERDICT = "receipts/vram-class-verdict.json"

QUALIFICATION_CONTEXT_SCHEMA = "qwen38-qualification-5090-context/1"
QUALIFICATION_24GIB_SCHEMA = "qwen38-qualification-24gib-capped/1"
VRAM_CLASS_VERDICT_SCHEMA = "qwen38-vram-class-verdict/1"


def receipt_stamp(ev: Evidence, rel: str, schema: str) -> dict:
    """Path, digest and declared schema for a receipt indexed whole."""
    doc = ev.load(rel, schema=schema)
    return {"path": rel, "sha256": ev.sha(rel),
            "schema": doc.get("schema") if isinstance(doc, dict) else None,
            "schema_not_verified": None if isinstance(doc, dict) else
                                   f"{rel} did not load as an object"}


def self_digest_holds(ev: Evidence, rel: str) -> bool:
    """Recompute a receipt's own content_sha256 under the recipe it declares."""
    doc = ev.load(rel)
    if not isinstance(doc, dict) or "content_sha256" not in doc:
        return False
    excludes = set(doc.get("content_sha256_excludes") or ())
    got = canonical_sha256({k: v for k, v in doc.items() if k not in excludes})
    return ev.expect_equal(got, doc["content_sha256"],
                           f"{rel}: stored content_sha256 vs recomputed")


def affine_kv_needed(a_bytes: float, m_gib: float, window: int) -> float:
    """kv_needed(L) = a*L + M, the law receipts/qualification-24gib-capped.json measures."""
    return round((a_bytes * window + m_gib * GIB) / GIB, 4)


def qualifications_and_verdicts(ev: Evidence) -> list[dict]:
    """Receipts that qualify hardware or decide a class, rather than publish a checkpoint.

    These are deliberately not `releases` rows: they name no artifact, carry no
    shard map, and have no fidelity report, so `build_rows`' pointer set does not
    apply to them and a RELEASES entry would fail closed on the first missing
    field.  They are indexed anyway, because a reader asking "what has this
    project actually started, and what did it decide" should not have to already
    know the filenames.  Every number below is pulled through a JSON pointer and
    stamped with its file's digest, exactly as a release row is.
    """
    rows: list[dict] = []

    # -- the physical RTX 5090 context qualification ------------------------
    rel = QUALIFICATION_CONTEXT
    rows.append({
        "id": "qualification-5090-context",
        "kind": "hardware qualification",
        "label": ("seven-gate qualification of the published -context edition on a "
                  "physical RTX 5090, the measurement every VRAM-class arithmetic "
                  "in docs/34 is re-derived on"),
        "receipt": receipt_stamp(ev, rel, QUALIFICATION_CONTEXT_SCHEMA),
        "board": ev.block(rel, {
            "model": "/identity/gpu/model",
            "uuid": "/identity/gpu/uuid",
            "physical_card": "/identity/gpu/physical_card",
            "memory_total_mib": "/identity/gpu/memory_total_mib",
            "usable_gib_reported_by_vllm": "/identity/gpu/usable_gib_reported_by_vllm",
        }),
        "outcome": ev.block(rel, {
            "all_gates_pass": "/verdict/all_gates_pass",
            "decisions": "/verdict/decisions",
            "qualified_at": "/verdict/qualified_at",
            "not_qualified_at": "/verdict/not_qualified_at",
        }),
        "why_not_a_release_row": ("it qualifies a serving configuration of an "
                                  "already-published checkpoint; the checkpoint's own "
                                  "row is in `releases`"),
    })

    # -- the 24 GiB-class proxy qualification -------------------------------
    rel = QUALIFICATION_24GIB
    law = ev.block(rel, {
        "a_mtp3_bytes_per_token": "/kv_law_measured/law/mtp3/a_bytes_per_token",
        "m_mtp3_gib": "/kv_law_measured/law/mtp3/M_gib",
        "a_mtpoff_bytes_per_token": "/kv_law_measured/law/mtpoff/a_bytes_per_token",
        "m_mtpoff_gib": "/kv_law_measured/law/mtpoff/M_gib",
    })
    board_mtp3 = ev.block(rel, {
        "window": "/pre_registered_predictions/predictions/board~1mtp3_32768/window",
        "pool_gib": "/published_prediction_outcomes/prediction_1/board_arm_measured/pool_gib",
        "kv_tokens_allocated":
            "/published_prediction_outcomes/prediction_1/board_arm_measured/kv_tokens_allocated",
        "kv_needed_for_one_request_gib":
            "/published_prediction_outcomes/prediction_1/board_arm_measured/kv_needed_for_one_request_gib",
        "headroom_pct":
            "/published_prediction_outcomes/prediction_1/board_arm_measured/headroom_pct",
    }, optional=("window",))
    board_mtpoff = ev.block(rel, {
        "pool_gib": "/published_prediction_outcomes/prediction_2/board_arm_measured/pool_gib",
        "kv_tokens_allocated":
            "/published_prediction_outcomes/prediction_2/board_arm_measured/kv_tokens_allocated",
        "kv_needed_for_one_request_gib":
            "/published_prediction_outcomes/prediction_2/board_arm_measured/kv_needed_for_one_request_gib",
        "headroom_pct":
            "/published_prediction_outcomes/prediction_2/board_arm_measured/headroom_pct",
    })
    # fail closed if the receipt's own law stops reproducing its own requirements
    law_holds = (
        ev.expect_equal(
            affine_kv_needed(law["a_mtp3_bytes_per_token"], law["m_mtp3_gib"], 32768),
            board_mtp3["kv_needed_for_one_request_gib"],
            f"{rel}: a*32768 + M vs the MTP-3 requirement it reports")
        and ev.expect_equal(
            affine_kv_needed(law["a_mtpoff_bytes_per_token"], law["m_mtpoff_gib"], 45056),
            board_mtpoff["kv_needed_for_one_request_gib"],
            f"{rel}: a*45056 + M vs the MTP-off requirement it reports")
    )
    rows.append({
        "id": "qualification-24gib-capped",
        "kind": "proxy qualification",
        "label": ("the same seven gates at a capped 24 GiB-class engine budget, with a "
                  "second arm whose card is ballasted down to a 24 GiB board's free "
                  "memory; the source of the measured KV law"),
        "receipt": receipt_stamp(ev, rel, QUALIFICATION_24GIB_SCHEMA),
        "physical_board_gate": ev.block(rel, {
            "state": "/physical_board_gate",
            "claim_scope": "/claim_scope",
            "what_this_does_not_say": "/verdict/what_this_does_not_say",
        }),
        "gates": ev.block(rel, {"gates_1_to_7": "/verdict/gates_1_to_7"}),
        "measured_kv_law": law,
        "board_arm_mtp3": board_mtp3,
        "board_arm_mtp_off": board_mtpoff,
        "law_reproduces_its_own_reported_requirements": law_holds,
        "why_not_a_release_row": ("no artifact, no shard map and no fidelity report: it "
                                  "measures how much context an existing checkpoint can "
                                  "be served with, on emulated hardware"),
    })

    # -- the VRAM-class verdict --------------------------------------------
    rel = VRAM_CLASS_VERDICT
    rows.append({
        "id": "vram-class-verdict",
        "kind": "class decision",
        "label": ("go/no-go for the 24 GB and 16 GB classes, with the arithmetic and "
                  "the flip conditions"),
        "receipt": receipt_stamp(ev, rel, VRAM_CLASS_VERDICT_SCHEMA),
        "decisions": ev.block(rel, {
            "class_24_gib": "/class_24_gib/verdict",
            "class_16_gib": "/class_16_gib/verdict",
        }),
        "published_24_gib_windows": ev.block(rel, {
            "mtp_3_fp8": "/class_24_gib/context/mtp_3_fp8/published_recommendation",
            "mtp_3_requirement_gib": "/class_24_gib/context/mtp_3_fp8/requirement_gib",
            "mtp_off_fp8": "/class_24_gib/context/mtp_off_fp8/published_recommendation",
            "mtp_off_requirement_gib": "/class_24_gib/context/mtp_off_fp8/requirement_gib",
        }),
        "self_digest": {
            "content_sha256": ev.pointer(rel, "/content_sha256"),
            "recomputes": self_digest_holds(ev, rel),
            "source": {"path": rel, "sha256": ev.sha(rel),
                       "pointers": {"content_sha256": "/content_sha256",
                                    "content_sha256_excludes": "/content_sha256_excludes"}},
        },
        "amendments": ev.block(rel, {
            "count": "/amendments",
            "latest_id": "/amendments/0/id",
            "latest_amends": "/amendments/0/amends_content_sha256",
            "latest_by": "/amendments/0/by",
        }, optional=("count",)),
        "why_not_a_release_row": ("it decides classes, it does not publish a checkpoint; "
                                  "the 24 GB class is served by an artifact that already "
                                  "has a `releases` row"),
    })

    # Cross-receipt: the windows the verdict publishes must cost what the measured
    # law says they cost.  If either receipt moves without the other, this fails.
    v = ev.load(VRAM_CLASS_VERDICT)
    if isinstance(v, dict) and law_holds:
        ctx = v.get("class_24_gib", {}).get("context", {})
        for key, a_key, m_key in (("mtp_3_fp8", "a_mtp3_bytes_per_token", "m_mtp3_gib"),
                                  ("mtp_off_fp8", "a_mtpoff_bytes_per_token", "m_mtpoff_gib")):
            row = ctx.get(key, {})
            window = row.get("published_recommendation")
            claimed = row.get("requirement_gib")
            if not isinstance(window, int) or not isinstance(claimed, (int, float)):
                ev.errors.append(f"{VRAM_CLASS_VERDICT}: {key} has no published window "
                                 "and requirement to cross-check")
                continue
            derived = affine_kv_needed(law[a_key], law[m_key], window)
            ev.expect(abs(derived - claimed) <= 0.001,
                      f"{VRAM_CLASS_VERDICT}: {key} publishes {window} at "
                      f"{claimed} GiB, but the law measured in {QUALIFICATION_24GIB} "
                      f"makes that window cost {derived} GiB")
    return rows


def conventions() -> dict:
    return {
        "row_sources": ("every number in a row carries the receipt path, that file's "
                        "sha256 and the RFC 6901 pointer it was read through; a null "
                        "always carries a not_verified note in its block, and an "
                        "optional receipt string the receipt never wrote is absent "
                        "rather than null"),
        "byte_counts": {
            "whole_tree_bytes": "all published files of the artifact, on disk",
            "immutable_payload_bytes": ("the files the build receipt treats as "
                                        "defining the artifact"),
            "tensor_payload_bytes": ("sum of the quantization manifest's per-role "
                                     "tensor bytes"),
            "resident_weights": ("measured GPU allocation for weights at load; the "
                                 "only memory figure here, and never a disk size"),
        },
        "shard_map_sha256": ("sha256 of "
                             "json.dumps({\"index_sha256\": <index digest>, "
                             "\"shard_sha256\": <shard map>}, sort_keys=True, "
                             "separators=(\",\", \":\")).encode()"),
        "self_digest": ("content_sha256 is canonical_sha256 of this document with "
                        "generated_utc, generator, content_sha256 and "
                        "content_sha256_excludes removed; canonical means "
                        "json.dumps(..., sort_keys=True, separators=(\",\", \":\"))"),
        "row_digests": "canonical_sha256 of each row object, keyed by row id",
        "qualifications_and_verdicts": ("receipts that qualify hardware or decide a "
                                        "VRAM class rather than publish a checkpoint. "
                                        "They are kept out of `releases` because they "
                                        "name no artifact and carry no shard map or "
                                        "fidelity report, but they are held to the same "
                                        "sourcing rule, and two cross-receipt checks "
                                        "fail generation: the measured affine KV law "
                                        "must reproduce the per-request requirements "
                                        "its own receipt reports, and every context "
                                        "window the class verdict publishes must cost "
                                        "what that law says it costs"),
        "qualification_digests": "canonical_sha256 of each non-release row, keyed by id",
    }


DIGEST_EXCLUDES = ["generated_utc", "generator", "content_sha256",
                   "content_sha256_excludes"]


def self_digest(payload: dict) -> str:
    return canonical_sha256({k: v for k, v in payload.items()
                             if k not in DIGEST_EXCLUDES})


def generator_block(root: Path) -> dict:
    tool = Path(__file__).resolve()
    try:
        tool_rel = str(tool.relative_to(root.resolve()))
    except ValueError:
        tool_rel = str(tool)
    out = {"tool": tool_rel, "tool_sha256": sha256_bytes(tool.read_bytes())}
    for key, cmd in (("repo_commit", ["git", "rev-parse", "HEAD"]),
                     ("repo_status", ["git", "status", "--porcelain"])):
        try:
            res = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                                 timeout=30)
        except (OSError, subprocess.SubprocessError):
            out[key] = None
            continue
        if res.returncode != 0:
            out[key] = None
        elif key == "repo_commit":
            out[key] = res.stdout.strip()
        else:
            out["worktree_dirty"] = bool(res.stdout.strip())
    return out


SOURCE_KEYS = ("pinned_by", "report_sha256", "declared_by", "manifest")


def is_sourced(node: dict) -> bool:
    return any(k == "source" or k.endswith("_source") or k in SOURCE_KEYS
               for k in node)


def is_noted(node: dict) -> bool:
    return any(k == "not_verified" or k.endswith("_not_verified") for k in node)


def audit_row(node, path: str, sourced: bool, noted: bool, out: list[str]) -> None:
    """The two properties the index exists to guarantee, enforced not trusted.

    A number whose enclosing block carries no `source`, `*_source`,
    `declared_by`, `pinned_by`, `manifest` or `report_sha256` is a number a
    reader cannot check.  A null with no `not_verified` note in scope is a gap
    the row does not admit to.  Either one stops generation.
    """
    if isinstance(node, dict):
        sourced = sourced or is_sourced(node)
        noted = noted or is_noted(node)
        for key, value in node.items():
            audit_row(value, f"{path}/{key}", sourced, noted, out)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            audit_row(value, f"{path}[{i}]", sourced, noted, out)
    elif node is None:
        if not noted:
            out.append(f"{path}: null with no not_verified note in its block")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if not sourced:
            out.append(f"{path}: number {node} has no receipt in its block")


def assemble(ev: Evidence) -> dict:
    rows = build_rows(ev)
    for row in rows:
        audit_row(row, "/" + row["id"], False, False, ev.errors)
    if ev.errors:
        raise Rejected(ev.errors)
    non_release = qualifications_and_verdicts(ev)
    for row in non_release:
        audit_row(row, "/qualifications_and_verdicts/" + row["id"], False, False,
                  ev.errors)
    if ev.errors:
        raise Rejected(ev.errors)
    payload = {
        "schema": SCHEMA,
        "title": ("Qwen3.8-27B EXL3 collection: one immutable row per published "
                  "checkpoint, every field traced to a receipt"),
        "conventions": conventions(),
        "releases": rows,
        "row_digests": {row["id"]: canonical_sha256(row) for row in rows},
        "known_divergences": known_divergences(ev),
        "suites_not_used_by_any_row": unused_suites(ev),
        "qualifications_and_verdicts": non_release,
        "qualification_digests": {row["id"]: canonical_sha256(row)
                                  for row in non_release},
        "receipt_digests": dict(sorted(ev.touched.items())),
    }
    if ev.errors:
        raise Rejected(ev.errors)
    return payload


def cmd_generate(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    ev = Evidence(root)
    payload = assemble(ev)
    payload["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["generator"] = generator_block(root)
    payload["content_sha256"] = self_digest(payload)
    payload["content_sha256_excludes"] = DIGEST_EXCLUDES

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    atomic_write_json(out, payload)
    print(json.dumps({
        "out": str(out),
        "releases": [r["id"] for r in payload["releases"]],
        "receipts_hashed": len(payload["receipt_digests"]),
        "row_digests": payload["row_digests"],
        "known_divergences": len(payload["known_divergences"]),
        "content_sha256": payload["content_sha256"],
    }, indent=2), flush=True)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    path = Path(args.index)
    if not path.is_absolute():
        path = root / path
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"REJECTED: {path}: unreadable index ({exc})")

    errors: list[str] = []
    if stored.get("schema") != SCHEMA:
        errors.append(f"{path}: schema {stored.get('schema')!r} != {SCHEMA!r}")
    recomputed_digest = self_digest(stored)
    if stored.get("content_sha256") != recomputed_digest:
        errors.append(f"{path}: content_sha256 {stored.get('content_sha256')} does not "
                      f"match the document, which digests to {recomputed_digest}")
    if stored.get("content_sha256_excludes") != DIGEST_EXCLUDES:
        errors.append(f"{path}: content_sha256_excludes "
                      f"{stored.get('content_sha256_excludes')} != {DIGEST_EXCLUDES}")

    ev = Evidence(root)
    try:
        fresh = assemble(ev)
    except Rejected as exc:
        raise SystemExit(f"{exc}\n(receipts under {root} no longer satisfy the "
                         f"fail-closed rules that produced {path})")

    stored_rows = {r["id"]: r for r in stored.get("releases", [])}
    fresh_rows = {r["id"]: r for r in fresh["releases"]}
    for rid in sorted(set(stored_rows) | set(fresh_rows)):
        if rid not in stored_rows:
            errors.append(f"row {rid}: present in the receipts, absent from the index")
            continue
        if rid not in fresh_rows:
            errors.append(f"row {rid}: present in the index, no longer derivable")
            continue
        a = canonical_sha256(stored_rows[rid])
        b = canonical_sha256(fresh_rows[rid])
        if a != b:
            errors.append(f"row {rid}: recomputes to {b}, index stores {a}")
            for field in sorted(set(stored_rows[rid]) | set(fresh_rows[rid])):
                if stored_rows[rid].get(field) != fresh_rows[rid].get(field):
                    errors.append(f"row {rid}: field {field!r} differs from the receipts")
        if stored.get("row_digests", {}).get(rid) != a:
            errors.append(f"row {rid}: row_digests entry "
                          f"{stored.get('row_digests', {}).get(rid)} != {a}")
    for field in ("conventions", "known_divergences", "suites_not_used_by_any_row",
                  "qualifications_and_verdicts", "qualification_digests",
                  "receipt_digests"):
        if stored.get(field) != fresh.get(field):
            errors.append(f"{field}: differs from what the receipts now produce")

    if errors:
        raise Rejected(errors)
    print(json.dumps({
        "index": str(path),
        "root": str(root),
        "schema": stored["schema"],
        "generated_utc": stored.get("generated_utc"),
        "content_sha256": stored["content_sha256"],
        "releases_verified": sorted(fresh_rows),
        "receipts_rehashed": len(fresh["receipt_digests"]),
        "verdict": "index matches the receipts and its own self digest",
    }, indent=2), flush=True)
    return 0


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        prog="collection_index.py",
        description="Immutable, receipt-traced index of the published checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--root", default=str(default_root),
                   help="repository root the receipt paths are relative to "
                        f"(default: {default_root})")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build the index (written atomically)")
    g.add_argument("--out", default="receipts/collection-index.json",
                   help="index path, absolute or relative to --root")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("check", help="re-verify an index against the receipts")
    c.add_argument("index", help="index path, absolute or relative to --root")
    c.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
