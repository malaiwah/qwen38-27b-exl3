#!/usr/bin/env python3
"""Assemble receipts/chat-template-audit.json from the measured artifacts.

Inputs (all produced by the other tools/chat_template_*.py steps):
  receipts/chat-templates/raw/*            fetched templates + repo API dumps
  $WORK/matrix.json                        rendered cases + token counts
  $WORK/roundtrip.json                     qwen3_coder arg-recovery results
  $WORK/prefix.json                        prompt growth + prefix-cache stability
  receipts/chat-templates/raw/issue-hunt.json  upstream issue survey

Host-side only; no GPU, no network.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_template_findings import (  # noqa: E402
    CONSTRUCTS, FROGGERIC_CLAIMS, RECOMMENDATION, UNSUBSTANTIATED, VERDICT,
)

RESEARCH = "/home/mbelleau/research"
RAW = f"{RESEARCH}/receipts/chat-templates/raw"
WORK = "/var/tmp/work/chat-template-audit"
OUT = f"{RESEARCH}/receipts/chat-template-audit.json"

SOURCES = [
    # (label, repo, revision, files)
    ("official-3.8", "Qwen/Qwen3.8-27B", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("official-3.8-FP8", "Qwen/Qwen3.8-27B-FP8", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("official-3.8-flagship", "Qwen/Qwen3.8-2.4T-A95B", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("RedHatAI-3.8-flagship", "RedHatAI/Qwen3.8-2.4T-A95B", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("RedHatAI-3.8-flagship-FP8", "RedHatAI/Qwen3.8-2.4T-A95B-FP8", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("official-3.6", "Qwen/Qwen3.6-27B", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("unsloth-3.8-NVFP4", "unsloth/Qwen3.8-27B-NVFP4", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("nvidia-3.6-NVFP4", "nvidia/Qwen3.6-27B-NVFP4", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("ours-hydrated", "malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("ours-K5K6", "malaiwah/Qwen3.8-27B-EXL3-K5K6", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("ours-K5K6-context", "malaiwah/Qwen3.8-27B-EXL3-K5K6-context", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
    ("ours-K4", "malaiwah/Qwen3.8-27B-K4", None,
     ["chat_template.jinja", "tokenizer_config.json"]),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def slug(repo):
    return repo.replace("/", "__")


def main():
    matrix = json.load(open(f"{WORK}/matrix.json"))
    roundtrip = json.load(open(f"{WORK}/roundtrip.json"))
    prefix = json.load(open(f"{WORK}/prefix.json"))
    issues = None
    ip = f"{RAW}/issue-hunt.json"
    if os.path.exists(ip):
        try:
            issues = json.load(open(ip))
        except Exception as e:  # noqa: BLE001
            issues = {"load_error": str(e)}

    catalog = []
    for label, repo, _rev, files in SOURCES:
        info_p = f"{RAW}/{slug(repo)}.info.json"
        rev = lm = None
        if os.path.exists(info_p):
            info = json.load(open(info_p))
            rev, lm = info.get("sha"), info.get("lastModified")
        ent = {
            "label": label,
            "repo": repo,
            "source_url": f"https://huggingface.co/{repo}",
            "revision": rev,
            "revision_last_modified": lm,
            "artifacts": {},
        }
        for f in files:
            p = f"{RAW}/{slug(repo)}.{f}"
            if not os.path.exists(p):
                ent["artifacts"][f] = {"present": False}
                continue
            rec = {
                "present": True,
                "resolve_url": f"https://huggingface.co/{repo}/resolve/{rev}/{f}",
                "sha256": sha256_file(p),
                "bytes": os.path.getsize(p),
                "local_copy": f"receipts/chat-templates/raw/{slug(repo)}.{f}",
            }
            if f == "tokenizer_config.json":
                j = json.load(open(p))
                ct = j.get("chat_template")
                rec["carries_chat_template_key"] = ct is not None
                if ct is not None:
                    rec["chat_template_key_sha256"] = sha256_str(ct)
                    rec["chat_template_key_bytes"] = len(ct.encode())
            ent["artifacts"][f] = rec
        # which artifact the engine actually uses
        tj = ent["artifacts"].get("chat_template.jinja", {})
        tc = ent["artifacts"].get("tokenizer_config.json", {})
        if tj.get("present") and tc.get("carries_chat_template_key"):
            ent["template_source_used_by_transformers_5_15"] = "chat_template.jinja"
            ent["dual_artifact"] = True
            ent["dual_artifact_identical"] = (
                tj["sha256"] == tc.get("chat_template_key_sha256")
            )
        elif tj.get("present"):
            ent["template_source_used_by_transformers_5_15"] = "chat_template.jinja"
            ent["dual_artifact"] = False
        else:
            ent["template_source_used_by_transformers_5_15"] = (
                "tokenizer_config.json:chat_template"
                if tc.get("carries_chat_template_key") else "none-at-repo-root"
            )
            ent["dual_artifact"] = False
        catalog.append(ent)

    # extras that do not follow the plain repo/file pattern
    extras = []

    def extra(label, path, **kw):
        d = {"label": label,
             "sha256": sha256_file(path),
             "bytes": os.path.getsize(path),
             "local_copy": path.replace(RESEARCH + "/", "")}
        d.update(kw)
        extras.append(d)

    extra("froggeric-v22",
          f"{RAW}/froggeric__Qwen-Fixed-Chat-Templates.chat_template.jinja",
          repo="froggeric/Qwen-Fixed-Chat-Templates",
          revision="9f14778c92c3b5ed3e0738085694c0d3452802dd",
          source_url="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/blob/main/chat_template.jinja",
          resolve_url="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/9f14778c92c3b5ed3e0738085694c0d3452802dd/chat_template.jinja",
          template_version_string="qwen3.8-froggeric-v22",
          artifact_kind="chat_template.jinja")
    extra("froggeric-v21",
          f"{RAW}/froggeric__Qwen-Fixed-Chat-Templates@main.v21_chat_template.jinja",
          repo="froggeric/Qwen-Fixed-Chat-Templates",
          revision="9f14778c92c3b5ed3e0738085694c0d3452802dd",
          resolve_url="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/9f14778c92c3b5ed3e0738085694c0d3452802dd/archive/v21_chat_template.jinja",
          artifact_kind="archived predecessor")
    extra("turboderp-exl3-5.00bpw",
          f"{RAW}/turboderp__Qwen3.8-27B-exl3@5.00bpw.chat_template.jinja",
          repo="turboderp/Qwen3.8-27B-exl3",
          revision="a35e75a73baee51da709329d19294245cbeeb5d8",
          branch="5.00bpw",
          note="main branch carries only README.md + .gitattributes; the "
               "quant payload and tokenizer live on per-bpw branches",
          resolve_url="https://huggingface.co/turboderp/Qwen3.8-27B-exl3/resolve/a35e75a73baee51da709329d19294245cbeeb5d8/chat_template.jinja",
          artifact_kind="chat_template.jinja")
    extra("RedHatAI-3.6-FP8",
          f"{RAW}/RedHatAI__Qwen3.6-27B-FP8.chat_template.jinja",
          repo="RedHatAI/Qwen3.6-27B-FP8",
          revision="57d986c3ab0397a811e5150190eff67f76fab6e0",
          note="only Red Hat quant of this family found; no RedHatAI Qwen3.8 "
               "repo exists as of the audit date",
          resolve_url="https://huggingface.co/RedHatAI/Qwen3.6-27B-FP8/resolve/57d986c3ab0397a811e5150190eff67f76fab6e0/chat_template.jinja",
          artifact_kind="chat_template.jinja")
    extra("unsloth-3.8-GGUF-Q4_K_M",
          f"{RAW}/unsloth__Qwen3.8-27B-GGUF__Q4_K_M.chat_template.jinja",
          repo="unsloth/Qwen3.8-27B-GGUF",
          revision="f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
          note="repo root carries no chat_template.jinja and no "
               "tokenizer_config.json; the template lives in the GGUF "
               "metadata key tokenizer.chat_template and was extracted with "
               "tools/gguf_chat_template.py over HTTP range requests "
               "(10945458 KV bytes read, no tensor data)",
          artifact_kind="GGUF KV tokenizer.chat_template",
          gguf_file="Qwen3.8-27B-Q4_K_M.gguf",
          gguf_arch="qwen35")
    extra("official-3.8-rev1-superseded",
          f"{RAW}/Qwen__Qwen3.8-27B@72a217afab.chat_template.jinja",
          repo="Qwen/Qwen3.8-27B",
          revision="72a217afab",
          note="first published 3.8 template (2026-08-13T08:23:30Z), "
               "superseded 1h53m later by 412f8b6bd7",
          resolve_url="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/72a217afab/chat_template.jinja",
          artifact_kind="chat_template.jinja (historical)")

    # in-image artifacts
    img = []
    for label, p, note in [
        ("vllm-rust-test-fixture-qwen3coder",
         "/var/tmp/gg-rootfs/opt/vllm/rust/src/chat/tests/templates/vllm_examples/"
         "tool_chat_template_qwen3coder.jinja",
         "Rust unit-test fixture only (loaded from tests/templates by "
         "src/renderer/hf/format.rs test helpers). Not installed on any "
         "runtime path; the pinned image ships no examples/ directory."),
        ("vllm-rust-test-fixture-qwen3",
         "/var/tmp/gg-rootfs/opt/vllm/rust/src/chat/tests/templates/qwen3.jinja",
         "Rust unit-test fixture (include_str! inside #[cfg(test)] mod tests)."),
        ("vllm-rust-test-fixture-qwen35",
         "/var/tmp/gg-rootfs/opt/vllm/rust/src/chat/tests/templates/qwen35.jinja",
         "Rust unit-test fixture (include_str! inside #[cfg(test)] mod tests)."),
    ]:
        if os.path.exists(p):
            img.append({"label": label, "path": p,
                        "sha256": sha256_file(p),
                        "bytes": os.path.getsize(p), "note": note})

    # collapse the render matrix: drop the full rendered text into a sidecar
    renders_path = f"{RESEARCH}/receipts/chat-templates/renders.json"
    with open(renders_path, "w") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=1)
    slim = {"generated_at": matrix["generated_at"],
            "jinja2_version": matrix["jinja2_version"],
            "tokenizer": matrix["tokenizer"],
            "templates": matrix["templates"], "results": {}}
    for c, row in matrix["results"].items():
        slim["results"][c] = {
            t: ({"error": r["error"]} if "error" in r
                else {"tokens": r["tokens"], "chars": r["chars"],
                      "sha256": r["sha256"]})
            for t, r in row.items()
        }

    receipt = {
        "receipt": "chat-template-audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "tools/chat_template_receipt.py",
        "audit_date": "2026-08-16",
        "scope": (
            "Side-by-side audit of every chat template that matters for the "
            "Qwen3.8-27B / Qwen3.6-27B family, and a verdict on whether the "
            "template shipped in our four published repos is wrong."
        ),
        "serving_context": {
            "image": "voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b",
            "rootfs": "/var/tmp/gg-rootfs (read-only)",
            "vllm_version": "0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34",
            "vllm_build_date": "2026-08-10",
            "transformers": "5.15.0",
            "jinja2": "3.1.6",
            "tokenizers": "0.22.2",
            "flags": ["--reasoning-parser qwen3", "--enable-auto-tool-choice",
                      "--tool-call-parser qwen3_coder"],
            "model_type": "qwen3_5",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "config_submodules_present": ["text_config", "vision_config"],
            "config_submodules_absent": ["audio_config"],
        },
        "how_measured": {
            "renderer": (
                "jinja2 3.1.6 ImmutableSandboxedEnvironment with "
                "trim_blocks=True, lstrip_blocks=True, the transformers "
                "tojson override, raise_exception global and the "
                "loopcontrols + AssistantTracker extensions -- i.e. an exact "
                "reconstruction of transformers/utils/chat_template_utils.py"
                "::_cached_compile_jinja_template as shipped in the pinned "
                "image."
            ),
            "executed_in": "the pinned image rootfs via tools/ggrun.sh (proot)",
            "gpu_used": False,
            "tokenizer": "/var/tmp/models/Qwen3.8-27B/tokenizer.json, "
                         "Tokenizer.encode(add_special_tokens=False)",
            "tools": ["tools/fetch_chat_templates.sh",
                      "tools/gguf_chat_template.py",
                      "tools/chat_template_matrix.py",
                      "tools/chat_template_roundtrip.py",
                      "tools/chat_template_prefix.py",
                      "tools/chat_template_receipt.py"],
        },
        "extracted_template_digests": "receipts/chat-templates/SHA256SUMS",
        "template_catalog": catalog,
        "template_catalog_extras": extras,
        "in_image_templates": img,
        "distinct_templates_by_digest": {},
        "engine_template_resolution": {
            "vllm_builtin_fallback_for_model_type_qwen3_5": None,
            "vllm_builtin_fallback_registry":
                "vllm/transformers_utils/chat_templates/registry.py -- "
                "_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK has no qwen* entry, so "
                "resolve_chat_template() falls through to the model repo's own "
                "template",
            "image_ships_examples_dir": False,
            "image_serve_scripts_pass_chat_template": False,
            "image_serve_script_parsers": {
                "serve-qwen35-397b-nvfp4.sh": "--reasoning-parser qwen3 "
                                              "--tool-call-parser qwen3_coder",
                "serve-qwen36-27b-nvfp4.sh": "--reasoning-parser qwen3 "
                                             "--tool-call-parser qwen3_xml",
            },
            "qwen3_coder_and_qwen3_xml_are_aliases": True,
            "qwen3_coder_target": "vllm/tool_parsers/qwen3_engine_tool_parser.py"
                                  "::Qwen3EngineToolParser -> "
                                  "vllm/parser/qwen3.py::Qwen3Parser, "
                                  "structural_tag_model='qwen_3_coder'",
            "reasoning_parser_qwen3_target":
                "vllm/reasoning/qwen3_engine_reasoning_parser.py"
                "::Qwen3ParserReasoningAdapter -> the same Qwen3Parser",
            "arg_converter_newline_semantics":
                "_trim_wrapping_newlines: exactly one leading and one trailing "
                "newline removed -- the correct inverse of the template's "
                "'<parameter=K>\\n' + value + '\\n</parameter>' markup, and NOT "
                "the strip() regression reported in vllm#48753",
            "template_artifact_precedence": {
                "transformers_version": "5.15.0",
                "winner": "chat_template.jinja",
                "source": "transformers/tokenization_utils_base.py:1783-1800 "
                          "('If independent chat template file(s) exist, they "
                          "take priority over template entries in the tokenizer "
                          "config')",
                "proved_empirically": True,
                "proof": "AutoTokenizer.from_pretrained on a directory holding "
                         "tokenizer.json plus a tokenizer_config.json whose "
                         "chat_template was 'SENTINEL_FROM_TOKENIZER_CONFIG' and "
                         "a chat_template.jinja holding "
                         "'SENTINEL_FROM_CHAT_TEMPLATE_JINJA' returned the "
                         ".jinja sentinel; deleting the .jinja returned the "
                         "tokenizer_config sentinel",
            },
            "tool_arguments_normalised_before_template": {
                "where": "vllm/entrypoints/chat_utils.py:1875-1915 "
                         "_postprocess_messages, called from both "
                         "parse_chat_messages and parse_chat_messages_async",
                "effect": "assistant function.arguments arriving as a JSON "
                          "string is json.loads()ed into a dict, and an empty or "
                          "absent value becomes {}, before any template runs",
                "residual_risk": "the json.loads call is unguarded, so a "
                                 "malformed arguments string in replayed history "
                                 "raises rather than 400ing cleanly "
                                 "(vllm#47761, fix PR vllm#48922 still open)",
            },
            "inbound_reasoning_field_normalisation":
                "an inbound assistant `reasoning_content` is renamed to "
                "`reasoning` by ChatCompletionRequest._normalize_messages_before "
                "(protocol.py:486-509) and then written back onto the "
                "conversation message under BOTH keys "
                "(chat_utils.py:1832-1836), so the template's "
                "`message.reasoning_content` does receive it",
            "developer_role_handling":
                "renderers/hf.py:722-726 converts developer->system and "
                "consolidates ONLY when a developer message is present and the "
                "template lacks developer support; two plain system messages are "
                "never consolidated and reach the template's raise",
        },
        "render_matrix": slim,
        "render_matrix_full_text": "receipts/chat-templates/renders.json",
        "tool_call_roundtrip": roundtrip,
        "prompt_growth_and_prefix_stability": prefix,
        "named_constructs": CONSTRUCTS,
        "froggeric_claim_adjudication": FROGGERIC_CLAIMS,
        "verdict": VERDICT,
        "recommendation": RECOMMENDATION,
        "claims_i_could_not_substantiate": UNSUBSTANTIATED,
        "upstream_issue_survey": issues,
    }

    # distinct digests
    dd = {}
    for e in catalog:
        a = e["artifacts"].get("chat_template.jinja")
        if a and a.get("present"):
            dd.setdefault(a["sha256"], []).append(e["label"])
    for e in extras:
        dd.setdefault(e["sha256"], []).append(e["label"])
    receipt["distinct_templates_by_digest"] = dd

    with open(OUT, "w") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes); "
          f"{len(dd)} distinct template digests")
    print(f"wrote {renders_path} ({os.path.getsize(renders_path)} bytes)")


if __name__ == "__main__":
    main()
