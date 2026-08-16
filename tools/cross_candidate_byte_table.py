#!/usr/bin/env python3
"""One byte table for every candidate in the cross-engine comparator, on one axis at a time.

The comparator's `serialized` column mixes axes: GGUF rows are whole-**file** sizes for a
**text-only** artifact, our rows are tensor **payload** for a **multimodal** artifact that also
carries an MTP draft, and the two families quantize the token embedding and the output head
differently.  Four different things are being compared under one column heading, and every one
of the differences happens to flatter us.  This tool states each axis separately and names, per
row, what that row's number actually is.

Inputs, all read from the artifacts themselves rather than restated:
  * GGUF rows: `receipts/gguf-*-tensor-accounting.json` from `tools/gguf_tensor_table.py`
    (tensor table over an HTTP range request).
  * our newer rows: each build's own `quantization_manifest.json`.
  * our two oldest rows, which predate that file: `tools/hf_safetensors_roles.py` output,
    i.e. the safetensors headers read over range requests.
  * the K6-parity row: the pre-registered prediction, labelled as one.

    cross_candidate_byte_table.py --gguf q8_0=... q6_k=... --ours ... --out receipts/x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GIB = 2 ** 30


def canonical_sha256(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def g(n: int | None) -> float | None:
    return None if n is None else round(n / GIB, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--mmproj", metavar="LABEL=PATH")
    ap.add_argument("--manifest", action="append", default=[], metavar="LABEL=PATH:WHOLETREE")
    ap.add_argument("--roles", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--predicted", action="append", default=[], metavar="LABEL=PATH",
                    help="a predicted role table, e.g. the K6-parity row; labelled prediction")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows: dict = {}

    def add(label, **kw):
        rows[label] = kw

    for spec in a.gguf:
        label, _, path = spec.partition("=")
        d = json.loads(Path(path).read_text())
        br = d["by_role"]
        emb, out = d["notable_tensors"]["token_embd.weight"], d["notable_tensors"]["output.weight"]
        add(label,
            family="GGUF (llama.cpp)",
            modality="TEXT ONLY -- the vision encoder ships as a separate mmproj-*.gguf",
            has_mtp_draft=False,
            file_bytes=d["file_bytes"], file_gib=d["file_bytes_gib"],
            tensor_bytes=d["tensor_bytes_total"], tensor_gib=d["tensor_bytes_total_gib"],
            metadata_and_padding_bytes=d["metadata_and_padding_bytes"],
            embedding={"format": emb["type"], "bytes": emb["bytes"], "gib": g(emb["bytes"])},
            head={"format": out["type"], "bytes": out["bytes"], "gib": g(out["bytes"])},
            vision={"format": "absent from this file", "bytes": 0, "gib": 0.0},
            body_bytes=br["body"]["bytes"], body_gib=br["body"]["gib"],
            body_definition="tensor total minus token_embd and output; no vision, no MTP draft",
            body_types=br["body"]["types"],
            source=d["source"]["url"],
            axis_of_the_published_serialized_column="whole FILE bytes of a text-only artifact")

    if a.mmproj:
        label, _, path = a.mmproj.partition("=")
        d = json.loads(Path(path).read_text())
        add(label,
            family="GGUF (llama.cpp)",
            modality="VISION ENCODER ONLY -- required alongside a text GGUF for multimodal use",
            has_mtp_draft=False,
            file_bytes=d["file_bytes"], file_gib=d["file_bytes_gib"],
            tensor_bytes=d["tensor_bytes_total"], tensor_gib=d["tensor_bytes_total_gib"],
            metadata_and_padding_bytes=d["metadata_and_padding_bytes"],
            embedding=None, head=None,
            vision={"format": "BF16/F32 mix", "bytes": d["tensor_bytes_total"],
                    "gib": d["tensor_bytes_total_gib"]},
            body_bytes=0, body_gib=0.0,
            body_definition="not a transformer body",
            body_types=d["by_role"]["body"]["types"],
            source=d["source"]["url"],
            axis_of_the_published_serialized_column="not in the comparator at all, which is "
                                                   "itself the problem")

    def ours(label, roles, whole_tree, source, label_kind, formats):
        payload = sum(v["bytes"] for v in roles.values())
        emb = roles["embed_tokens"]["bytes"]
        head = roles["lm_head"]["bytes"]
        vis = roles["vision_tower"]["bytes"]
        draft = roles["mtp_draft"]["bytes"]
        add(label,
            family="EXL3 (vLLM)",
            modality="MULTIMODAL in one tree -- the vision tower is inside this payload",
            has_mtp_draft=True,
            file_bytes=whole_tree, file_gib=g(whole_tree),
            tensor_bytes=payload, tensor_gib=g(payload),
            metadata_and_padding_bytes=(whole_tree - payload) if whole_tree else None,
            embedding={"format": formats["embedding"], "bytes": emb, "gib": g(emb)},
            head={"format": formats["head"], "bytes": head, "gib": g(head)},
            vision={"format": "BF16", "bytes": vis, "gib": g(vis)},
            body_bytes=payload - emb - head - vis - draft,
            body_gib=g(payload - emb - head - vis - draft),
            body_definition="payload minus embedding, head, vision tower AND MTP draft, so it "
                            "is the same set of weights a text-only GGUF's body covers",
            body_plus_draft_bytes=payload - emb - head - vis,
            body_plus_draft_gib=g(payload - emb - head - vis),
            mtp_draft={"bytes": draft, "gib": g(draft)},
            body_types=formats.get("body", {}),
            source=source, evidence_label=label_kind,
            axis_of_the_published_serialized_column="tensor PAYLOAD of a multimodal artifact "
                                                    "that also carries an MTP draft")

    for spec in a.manifest:
        label, _, rest = spec.partition("=")
        path, _, whole = rest.partition(":")
        d = json.loads(Path(path).read_text())
        fm = {"embedding": "BF16",
              "head": "exl3 K6 mcg" if "K6" in json.dumps(d["roles"]["lm_head"]["formats"])
                      else json.dumps(d["roles"]["lm_head"]["formats"]),
              "body": {k: v["formats"] for k, v in d["roles"].items()
                       if k not in ("embed_tokens", "vision_tower", "lm_head", "mtp_draft")}}
        ours(label, d["roles"], int(whole) if whole else None, path, "measured", fm)

    for spec in a.roles:
        label, _, path = spec.partition("=")
        d = json.loads(Path(path).read_text())
        roles = {k: {"bytes": v["bytes"], "formats": v["dtypes"]}
                 for k, v in d["by_role"].items()}
        fm = {"embedding": "BF16", "head": "exl3 K6 mcg",
              "body": {k: v["dtypes"] for k, v in d["by_role"].items()
                       if k not in ("embed_tokens", "vision_tower", "lm_head", "mtp_draft")}}
        ours(label, roles, d.get("shard_bytes_total"),
             d["source"]["repo"] + " safetensors headers", "measured", fm)

    for spec in a.predicted:
        label, _, path = spec.partition("=")
        d = json.loads(Path(path).read_text())
        ours(label, d["roles"], d.get("whole_tree_bytes"), d["source"], "prediction",
             {"embedding": "BF16", "head": "exl3 K6 mcg", "body": d.get("body_formats", {})})

    # comparisons, one axis at a time
    def by(key):
        return {k: v[key] for k, v in rows.items() if v.get(key) is not None}
    comparisons = {
        "axis_1_transformer_body_bytes": {
            "what": "the weights both families actually quantize and multiply: no embedding, no "
                    "output head, no vision encoder, no MTP draft. This is the axis that "
                    "isolates format and allocation from packaging choices, and it is the "
                    "honest reply to 'your win is just more bytes'.",
            "ranking_gib": dict(sorted(by("body_gib").items(), key=lambda kv: -kv[1])),
        },
        "axis_2_whole_deployed_artifact": {
            "what": "everything you must download to serve the model's advertised capability. "
                    "For a GGUF that is the text file PLUS mmproj-BF16.gguf, because the text "
                    "file has no vision tensors at all; for ours it is the whole tree.",
            "multimodal_deployment_gib": {
                k: (round((v["file_bytes"] + mm) / GIB, 4) if v["family"].startswith("GGUF")
                    else v["file_gib"])
                for k, v in rows.items()
                if v.get("file_bytes") and "vision half" not in k
                for mm in [next((r["file_bytes"] for n, r in rows.items()
                                 if "vision half" in n), 0)]
            },
            "note": "a GGUF row here is text file + mmproj-BF16.gguf; ours is the whole tree, "
                    "vision tower included, MTP draft included",
        },
        "axis_3_tensor_payload_as_shipped": {
            "what": "tensor bytes in the file(s) as shipped, without normalising modality. This "
                    "is the axis closest to the comparator's current column, and it compares a "
                    "text-only artifact with a multimodal one.",
            "ranking_gib": dict(sorted(by("tensor_gib").items(), key=lambda kv: -kv[1])),
        },
    }
    mm = next((r["file_bytes"] for n, r in rows.items() if "vision half" in n), 0)

    def deployed(name):
        """Everything you must download to serve the advertised capability.

        A GGUF text file needs mmproj-BF16.gguf beside it. For a predicted row no whole-tree
        size exists yet, so the payload plus the measured 24.0 MB non-tensor allowance is used
        and stays labelled a prediction.
        """
        r = rows[name]
        if r["family"].startswith("GGUF"):
            return r["file_bytes"] + mm
        return r["file_bytes"] if r.get("file_bytes") else r["tensor_bytes"] + 24_000_000

    def pair(theirs, ours_name, published_sentence):
        t, o = rows[theirs], rows[ours_name]
        return {
            "published_claim": published_sentence,
            "axis_the_published_claim_uses": "GGUF whole FILE bytes beside our tensor PAYLOAD",
            "same_axis_tensor_bytes": {
                theirs: t["tensor_gib"], ours_name: o["tensor_gib"],
                "difference_gib_theirs_minus_ours": round(t["tensor_gib"] - o["tensor_gib"], 4)},
            "transformer_body_bytes": {
                theirs: t["body_gib"], ours_name: o["body_gib"],
                "difference_gib_theirs_minus_ours": round(t["body_gib"] - o["body_gib"], 4),
                "difference_pct_of_ours": round(100 * (t["body_gib"] - o["body_gib"])
                                                / o["body_gib"], 1)},
            "whole_deployed_multimodal_bytes": {
                theirs: round(deployed(theirs) / GIB, 4),
                ours_name: round(deployed(ours_name) / GIB, 4),
                "difference_gib_theirs_minus_ours": round((deployed(theirs)
                                                          - deployed(ours_name)) / GIB, 4),
                "note": "their column adds mmproj-BF16.gguf because their text file has no "
                        "vision tensors; ours already contains the vision tower"},
            "embedding_and_head_asymmetry": {
                "their_embedding": t["embedding"], "our_embedding": o["embedding"],
                "their_head": t["head"], "our_head": o["head"]},
        }

    published_claims = {
        "6-bit: Q6_K versus hydrated": pair(
            "GGUF Q6_K", "hydrated K5/K6",
            "cards and receipts/cross-engine-comparator.json: Q6_K gets about 43 % lower "
            "divergence for 1.186 GiB more file"),
        "5-bit: UD-Q5_K_XL versus the context edition": pair(
            "GGUF UD-Q5_K_XL", "context edition",
            "cards: the context edition is about 13 % better fidelity for 0.445 GiB more "
            "payload"),
        "8-bit: Q8_0 versus online K5/K6": pair(
            "GGUF Q8_0", "online K5/K6",
            "cards: Q8_0 leads everything at 27.05 GiB"),
        "the parity test: Q6_K versus the K6-parity prediction": pair(
            "GGUF Q6_K", "K6-parity [P]",
            "receipts/preregistration-kld9-window.json: the parity build lands at byte parity "
            "with Q6_K, 21.45 GiB against 21.31 GiB"),
    }

    payload = {
        "schema": "qwen38-cross-candidate-byte-accounting/1",
        "question": "on which axis is each candidate's published byte figure measured, and what "
                    "do the comparisons look like when a single axis is used throughout?",
        "why": "the comparator's `serialized` column carries GGUF whole-FILE sizes for TEXT-ONLY "
               "artifacts beside our tensor PAYLOAD for MULTIMODAL ones that also ship an MTP "
               "draft, and the two families quantize the token embedding and the output head "
               "differently. Four separate differences, and every one of them flatters us.",
        "rows": rows,
        "comparisons": comparisons,
        "published_claims_on_one_axis_at_a_time": published_claims,
        "reading_rules": [
            "name the axis in the claim, not in a footnote: 'at equal file bytes' and 'at equal "
            "transformer-body bytes' are different claims",
            "a GGUF text file is not a multimodal artifact; adding mmproj-BF16.gguf is what "
            "makes the capability comparable",
            "the embedding and head widths are NOT uniform across the GGUF tiers, so the "
            "asymmetry must be taken per row and never assumed",
        ],
        "errors": [],
    }
    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"})
    a.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(comparisons["axis_1_transformer_body_bytes"]["ranking_gib"], indent=1))
    print(json.dumps(comparisons["axis_3_tensor_payload_as_shipped"]["ranking_gib"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
