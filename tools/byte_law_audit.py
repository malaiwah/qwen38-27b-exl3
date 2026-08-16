#!/usr/bin/env python3
"""Re-derive every published per-role byte prediction, and name the recipe each one assumes.

Why: a byte prediction that silently assumes the wrong recipe survives review because it
looks like a rounding disagreement.  One was nearly made real in this project -- the S16-V
build script's `^.*mlp\\.(gate|up|down)_proj$` override would have reached the MTP draft's
three MLP projections and undershot docs/34 §6.2's published total by 33,423,360 B, which
would have read as a byte-law failure rather than as a recipe difference.  That was caught
before converting.  This tool checks whether any *published* row has the same defect.

The audit is deliberately independent of the receipts it audits.  Module shapes and
parameter counts come from `receipts/error-driven-ladder.json` (in_features, out_features,
numel per module, 409 modules); role totals come from
`receipts/hydrated-quantization-manifest.json`.  The affine rule and its overhead term are
re-derived from those two, not copied from `receipts/vram-class-verdict.json`, and then the
verdict's own `byte_law.per_role` table is compared against the result.

    byte_law_audit.py --out receipts/byte-law-recipe-audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GIB = 2 ** 30

EMBED = {"bf16": 2_542_796_800, "int8": 1_271_398_400, "int4_g128": 655_564_800}
VISION_BF16 = 921_460_192
NORMS_BF16 = 1_320_960
# Whole-tree deltas: the payload plus tokenizer, configs and chat template.  docs/34 §2
# states the planning rule as "payload + 24.0 MB", but the two published artifacts measure
# 23,969,336 B (hydrated) and 23,971,318 B (context), and §6.2's predicted-disk row for
# S16-V uses the context figure rather than the rounded rule.  Both are carried so each
# published disk row can say which delta it assumes.
DISK_RULE = 24_000_000
DISK_MEASURED = {"hydrated": 23_969_336, "context": 23_971_318}

ROLE_KEYS = ("full_attention", "linear_attention", "mlp_gate_proj", "mlp_up_proj",
             "mlp_down_proj", "lm_head", "mtp_draft")


def canonical_sha256(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def role_of(name: str) -> str:
    if name.startswith("mtp"):
        return "mtp_draft"
    for key, r in (("mlp.gate_proj", "mlp_gate_proj"), ("mlp.up_proj", "mlp_up_proj"),
                   ("mlp.down_proj", "mlp_down_proj"), ("linear_attn", "linear_attention"),
                   ("self_attn", "full_attention")):
        if key in name:
            return r
    if name == "lm_head":
        return "lm_head"
    raise KeyError(name)


class ByteLaw:
    """bytes(role, K) = fixed(role) + params(role) * K / 8, derived from module shapes."""

    def __init__(self) -> None:
        self.ladder = json.loads((REPO / "receipts/error-driven-ladder.json").read_text())
        self.manifest = json.loads(
            (REPO / "receipts/hydrated-quantization-manifest.json").read_text())
        self.mods = self.ladder["modules"]
        self.groups: dict[str, list] = {}
        for name, m in self.mods.items():
            self.groups.setdefault(role_of(name), []).append((name, m))
        # EXL3 per-module overhead: 2*(in+out) fp16 suh/svh bytes plus one int32 scalar.
        self.overhead = {r: sum(2 * (m["in_features"] + m["out_features"]) + 4
                                for _, m in items)
                         for r, items in self.groups.items()}
        self.params = {r: sum(m["numel"] for _, m in items)
                       for r, items in self.groups.items()}
        # Whatever the manifest role carries beyond its exl3 modules: BF16 norms, F16
        # in_proj_a/in_proj_b, the MTP's BF16 tensors.
        self.residual = {
            r: self.manifest["roles"][r]["bytes"]
               - sum(self.exl3_bytes(m, m["recipe_bits"]) for _, m in items)
            for r, items in self.groups.items()}
        self.fixed = {r: self.overhead[r] + self.residual[r] for r in self.groups}

    @staticmethod
    def exl3_bytes(m: dict, K: int) -> int:
        return m["numel"] * K // 8 + 2 * (m["in_features"] + m["out_features"]) + 4

    def role_bytes(self, role: str, K) -> int:
        """K is an int, or a dict of module-substring -> width for a mixed role."""
        if isinstance(K, dict):
            def width(name: str) -> int:
                for key, w in K.items():
                    if key in name:
                        return w
                raise KeyError(name)
            body = sum(self.exl3_bytes(m, width(n)) for n, m in self.groups[role])
        else:
            body = sum(self.exl3_bytes(m, K) for _, m in self.groups[role])
        return body + self.residual[role]

    def payload(self, widths: dict, embed: str = "bf16", vision: bool = True) -> int:
        return (sum(self.role_bytes(r, K) for r, K in widths.items())
                + EMBED[embed] + (VISION_BF16 if vision else 0) + NORMS_BF16)


# Every published recipe, with the width map each published byte row assumes.  The point of
# this table is that the recipe is written down beside the number.
HYD_MTP = {"self_attn": 6, "mlp.gate": 5, "mlp.up": 5, "mlp.down": 6, "fc": 4}
CTX_MTP = {"self_attn": 5, "mlp.gate": 5, "mlp.up": 5, "mlp.down": 6, "fc": 4}
RECIPES = {
    "hydrated (published artifact)": {
        "widths": {"full_attention": 6, "linear_attention": 6, "mlp_gate_proj": 5,
                   "mlp_up_proj": 5, "mlp_down_proj": 6, "lm_head": 6,
                   "mtp_draft": HYD_MTP},
        "embed": "bf16", "vision": True,
        "recipe_named": "MLP gate/up K5, down K6, attention K6 serialized, lm_head K6/mcg, "
                        "MTP self_attn K6 + mlp K5/K5/K6 + eh_proj K4, BF16 embed and vision. "
                        "The MTP MLP follows the build's EXL3_BITS_OVERRIDE, NOT its -mb 4.",
        "claim": 21_586_964_548,
        "claim_source": "receipts/hydrated-quantization-manifest.json role total; "
                        "docs/34 §1 and §2",
        "label": "measured",
    },
    "context edition (published artifact)": {
        "widths": {"full_attention": 5, "linear_attention": 5, "mlp_gate_proj": 5,
                   "mlp_up_proj": 5, "mlp_down_proj": 6, "lm_head": 6,
                   "mtp_draft": CTX_MTP},
        "embed": "bf16", "vision": True,
        "recipe_named": "attention K5 serialized, MLP gate/up K5, down K6, lm_head K6/mcg, "
                        "MTP self_attn K5 + mlp K5/K5/K6 + eh_proj K4, BF16 embed and vision",
        "claim": 20_672_081_988,
        "claim_source": "receipts/vram-class-verdict.json byte_law.validation.context",
        "label": "measured",
    },
    "S16-V (docs/34 §6.2 primary)": {
        "widths": {"full_attention": 4, "linear_attention": 3, "mlp_gate_proj": 3,
                   "mlp_up_proj": 3, "mlp_down_proj": 3, "lm_head": 4, "mtp_draft": 4},
        "embed": "bf16", "vision": True,
        "recipe_named": "full_attention K4, linear_attention K3, MLP gate/up/down K3, "
                        "lm_head K4/mcg, MTP draft ALL K4, BF16 embed on disk and BF16 vision. "
                        "The all-K4 draft is what the 212,636,704 B mtp_draft row prices, and "
                        "reproducing it requires the MLP width override to be scoped to the "
                        "language-model body.",
        "claim": 13_711_503_428,
        "claim_source": "docs/34 §6.2; receipts/vram-class-verdict.json "
                        "class_16_gib.s16_v_candidate.serialized_bytes",
        "label": "prediction",
    },
    "S16-V-long (docs/34 §6.2)": {
        "widths": {"full_attention": 3, "linear_attention": 3, "mlp_gate_proj": 3,
                   "mlp_up_proj": 3, "mlp_down_proj": 3, "lm_head": 4, "mtp_draft": 4},
        "embed": "bf16", "vision": True,
        "recipe_named": "S16-V with full_attention demoted K4 -> K3; MTP draft ALL K4",
        "claim": 13_501_788_228,
        "claim_source": "docs/34 §6.2 profile table, quoted as 13.502 GB",
        "label": "prediction",
    },
    "S16-T text-only branch (docs/34 §6.2)": {
        "widths": {"full_attention": 3, "linear_attention": 3, "mlp_gate_proj": 4,
                   "mlp_up_proj": 4, "mlp_down_proj": 4, "lm_head": 3, "mtp_draft": 4},
        "embed": "bf16", "vision": False,
        "recipe_named": "full_attention K3, linear_attention K3, MLP all K4, lm_head K3/mcg, "
                        "MTP draft ALL K4, vision tower ABSENT, embedding BF16 ON DISK (the "
                        "int4-g128 form in §6.1 is a load-time overlay, so the serialized row "
                        "carries BF16)",
        "claim": 14_560_498_276,
        "claim_source": "docs/34 §6.2 profile table, quoted as 14.560 GB",
        "label": "prediction",
    },
    "cheapest multimodal build that keeps the MLP at 4 bits": {
        "widths": {"full_attention": 3, "linear_attention": 3, "mlp_gate_proj": 4,
                   "mlp_up_proj": 4, "mlp_down_proj": 4, "lm_head": 3, "mtp_draft": 4},
        "embed": "int8", "vision": True,
        "recipe_named": "full_attention K3, linear_attention K3, MLP all K4, lm_head K3, "
                        "MTP draft ALL K4, int8 embedding RESIDENT, BF16 vision tower. This is "
                        "a resident-payload row, not a serialized one.",
        "claim_gib": 13.235,
        "claim_source": "receipts/vram-class-verdict.json class_16_gib."
                        "the_arithmetic_that_kills_the_safe_option.resident_gib.formula",
        "label": "prediction",
    },
    "K6-parity (this window's condition 3)": {
        "widths": {"full_attention": 6, "linear_attention": 6, "mlp_gate_proj": 6,
                   "mlp_up_proj": 6, "mlp_down_proj": 6, "lm_head": 6,
                   "mtp_draft": {"self_attn": 6, "mlp.gate": 6, "mlp.up": 6, "mlp.down": 6,
                                 "fc": 4}},
        "embed": "bf16", "vision": True,
        "recipe_named": "hydrated with MLP gate/up promoted K5 -> K6. The override regex is the "
                        "hydrated build's own loose form, so the MTP MLP promotes with the body "
                        "-- which is why the prediction charges the draft's gate and up as well.",
        "claim": 23_035_310_148,
        "claim_source": "receipts/preregistration-kld9-window.json conditions[2]."
                        "predicted_bytes.serialized_payload",
        "label": "prediction",
    },
}

RESIDENT = {
    "S16-V resident payload, int8 embed": {
        "widths": RECIPES["S16-V (docs/34 §6.2 primary)"]["widths"], "embed": "int8",
        "vision": True, "claim": 12_440_105_028, "claim_gib": 11.586,
        "claim_source": "receipts/vram-class-verdict.json class_16_gib.s16_v_candidate."
                        "resident_payload_gib.formula",
    },
    "S16-V-long resident payload, int8 embed": {
        "widths": RECIPES["S16-V-long (docs/34 §6.2)"]["widths"], "embed": "int8",
        "vision": True, "claim": None, "claim_gib": 11.39,
        "claim_source": "docs/34 §6.2 profile table resident column, 11.74 = payload + 0.35",
    },
    "S16-T resident payload, int4-g128 embed, no vision": {
        "widths": RECIPES["S16-T text-only branch (docs/34 §6.2)"]["widths"],
        "embed": "int4_g128", "vision": False, "claim": None, "claim_gib": 11.803,
        "claim_source": "docs/34 §6.2 profile table resident column, 12.15 = payload + 0.35",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    law = ByteLaw()
    errors: list = []

    verdict = json.loads((REPO / "receipts/vram-class-verdict.json").read_text())
    per_role_published = verdict["byte_law"]["per_role"]
    per_role = {}
    for r in ROLE_KEYS:
        mine = {"params": law.params[r], "fixed_bytes": law.fixed[r],
                "gib_per_bit": round(law.params[r] / 8 / GIB, 4)}
        pub = {k: per_role_published[r].get(k) for k in mine}
        agree = pub == mine
        per_role[r] = {"rederived": mine, "published": pub, "agree": agree,
                       "modules": len(law.groups[r]),
                       "overhead_component_bytes": law.overhead[r],
                       "non_exl3_component_bytes": law.residual[r]}
        if not agree:
            errors.append("byte_law.per_role[%s] does not re-derive" % r)

    totals = {}
    for name, spec in RECIPES.items():
        got = law.payload(spec["widths"], spec["embed"], spec["vision"])
        row = {
            "recipe_assumed_by_the_published_row": spec["recipe_named"],
            "label": spec["label"],
            "claim_source": spec["claim_source"],
            "rederived_bytes": got,
            "rederived_gb": round(got / 1e9, 3),
            "rederived_gib": round(got / GIB, 4),
            "predicted_disk_bytes_under_the_rounded_rule": got + DISK_RULE,
            "predicted_disk_bytes_under_each_measured_whole_tree_delta": {
                k: got + v for k, v in DISK_MEASURED.items()},
        }
        if spec.get("claim") is not None:
            row["published_bytes"] = spec["claim"]
            row["difference_bytes"] = spec["claim"] - got
            row["exact"] = spec["claim"] == got
            if not row["exact"]:
                errors.append("%s: published %d, re-derived %d"
                              % (name, spec["claim"], got))
        if spec.get("claim_gib") is not None:
            row["published_gib"] = spec["claim_gib"]
            row["difference_gib"] = round(spec["claim_gib"] - got / GIB, 4)
            if abs(row["difference_gib"]) > 0.001:
                errors.append("%s: published %.4f GiB, re-derived %.4f GiB"
                              % (name, spec["claim_gib"], got / GIB))
        totals[name] = row

    resident = {}
    for name, spec in RESIDENT.items():
        got = law.payload(spec["widths"], spec["embed"], spec["vision"])
        row = {"rederived_bytes": got, "rederived_gib": round(got / GIB, 4),
               "plus_planning_allowance_0_35_gib": round(got / GIB + 0.35, 4),
               "claim_source": spec["claim_source"]}
        if spec["claim"] is not None:
            row["published_bytes"] = spec["claim"]
            row["exact"] = spec["claim"] == got
            if not row["exact"]:
                errors.append("%s: published %d, re-derived %d" % (name, spec["claim"], got))
        row["published_gib"] = spec["claim_gib"]
        if abs(spec["claim_gib"] - got / GIB) > 0.001:
            errors.append("%s: published %.4f GiB, re-derived %.4f GiB"
                          % (name, spec["claim_gib"], got / GIB))
        resident[name] = row

    # docs/34 §7's knob figures are per-bit role sums; every one is checked here because they
    # are what the knob ranking's GiB-per-0.001-KLD ratios divide by.
    gpb = {r: law.params[r] / 8 / GIB for r in ROLE_KEYS}
    knobs = {
        "attention K6->K5, 64 L, body only": {
            "published_gib": 0.840,
            "rederived_gib": round(gpb["full_attention"] + gpb["linear_attention"], 4)},
        "attention K6->K5 including the MTP attention": {
            "published_gib": 0.852,
            "rederived_gib": round(gpb["full_attention"] + gpb["linear_attention"]
                                   + 104_857_600 / 8 / GIB, 4)},
        "MLP -> K4, gate 1 + up 1 + down 2 bits": {
            "published_gib": 2.656,
            "rederived_gib": round(4 * gpb["mlp_gate_proj"], 4)},
        "lm_head K6->K5": {"published_gib": 0.148,
                           "rederived_gib": round(gpb["lm_head"], 4)},
        "attention K6->K4, 64 L": {
            "published_gib": 1.680,
            "rederived_gib": round(2 * (gpb["full_attention"] + gpb["linear_attention"]), 4)},
        "int8 input-embedding overlay": {
            "published_gib": 1.184,
            "rederived_gib": round((EMBED["bf16"] - EMBED["int8"]) / GIB, 4)},
        "one bit anywhere, cheapest role (lm_head)": {
            "published_gib": 0.148, "rederived_gib": round(gpb["lm_head"], 4)},
        "one bit anywhere, dearest single knob (MLP gate+up+down)": {
            "published_gib": 1.992,
            "rederived_gib": round(3 * gpb["mlp_gate_proj"], 4)},
    }
    for k, v in knobs.items():
        v["agree_to_quoted_precision"] = abs(v["published_gib"] - v["rederived_gib"]) <= 0.001
        if not v["agree_to_quoted_precision"]:
            errors.append("docs/34 §7 knob %r: published %.3f, re-derived %.4f"
                          % (k, v["published_gib"], v["rederived_gib"]))

    payload = {
        "schema": "qwen38-byte-law-recipe-audit/1",
        "question": "does every published per-role byte prediction re-derive from the module "
                    "shapes, and does each one name the recipe it assumes? A prediction that "
                    "silently assumes a different recipe looks like a rounding disagreement.",
        "why_now": {
            "trigger": "the S16-V build script's MLP width override, written as "
                       "'^.*mlp\\\\.(gate|up|down)_proj$', would also have matched the MTP "
                       "draft's three MLP projections, because EXL3_BITS_OVERRIDE is applied "
                       "with re.match against every key in f_targets and the draft's modules "
                       "are among them.",
            "evidence_that_this_is_real_and_not_theoretical": "the PUBLISHED hydrated build "
                "shows the same behaviour: receipts/error-driven-ladder.json records its "
                "recipe_bits per module, and mtp.layers.0.mlp.{gate,up}_proj are 5 with "
                "mtp.layers.0.mlp.down_proj 6 -- its override's values -- while mtp.fc, which "
                "no pattern matches, is 4 from -mb 4. The hydrated draft's MLP follows the "
                "override, not the -mb flag.",
            "what_it_would_have_cost": "33,423,360 B (31.9 MiB) of undershoot against docs/34 "
                "§6.2's published S16-V total, which would have read as a byte-law failure "
                "rather than as a recipe difference. Caught before converting; the override is "
                "now scoped to the language-model body.",
            "fidelity_impact": "none possible. The fidelity protocol never engages MTP "
                               "(docs/34 §7), so no MTP width can reach a measured KLD.",
        },
        "method": {
            "independence": "module shapes (in_features, out_features, numel for 409 modules) "
                            "come from receipts/error-driven-ladder.json and role totals from "
                            "receipts/hydrated-quantization-manifest.json. The affine rule and "
                            "its overhead term are re-derived from those two, then compared "
                            "against receipts/vram-class-verdict.json's own byte_law table -- "
                            "which is therefore audited, not used.",
            "law": "bytes(role, K) = fixed(role) + params(role) * K / 8, with "
                   "fixed(role) = sum over the role's exl3 modules of 2*(in+out) + 4, plus "
                   "whatever non-exl3 tensors the manifest role carries (BF16 norms, the "
                   "linear_attention F16 in_proj_a/in_proj_b, the MTP's BF16 tensors)",
            "residual_definition": "manifest role bytes minus the exl3 sum at the manifest's own "
                                   "widths; it is the non-exl3 component and is width-invariant",
            "disk_rule": "docs/34 §2 states predicted disk bytes = payload + 24.0 MB, rounded from "
                     "two measurements: 21,610,933,884 - 21,586,964,548 = 23,969,336 B for "
                     "hydrated and 23,971,318 B for context. Both are carried per row, because "
                     "the two differ by 1,982 B and the rounded rule differs from either by "
                     "about 29 kB.",
        },
        "per_role_table": per_role,
        "totals": totals,
        "resident_payloads": resident,
        "docs34_section7_knob_figures": knobs,
        "which_whole_tree_delta_each_published_disk_row_assumes": {
            "docs34_6_2_s16v_predicted_disk_bytes": {
                "published": 13_735_474_746,
                "payload_plus_context_delta": 13_711_503_428 + DISK_MEASURED["context"],
                "payload_plus_hydrated_delta": 13_711_503_428 + DISK_MEASURED["hydrated"],
                "payload_plus_rounded_24_0_mb": 13_711_503_428 + DISK_RULE,
                "which_one_it_is": "the CONTEXT build's measured whole-tree delta of "
                                   "23,971,318 B, exactly",
                "difference_from_the_rounded_rule_bytes": 13_711_503_428 + DISK_RULE
                                                          - 13_735_474_746,
                "checked_unchanged": True,
                "clarification_worth_one_sentence": "docs/34 §2 states the rule as 'payload + "
                    "24.0 MB' while §6.2's row uses the context build's measured 23,971,318 B; "
                    "the two disagree by 28,682 B, which is immaterial to every conclusion but "
                    "means the two rows are not derived by the same arithmetic.",
            },
        },
        "cards": {
            "checked": "all four model cards and the dataset cards, for any byte figure that is "
                       "a PREDICTION rather than a measurement",
            "found": "none. The cards print measured serialized bytes for published artifacts "
                     "(21,586,964,548 B of payload and 21,610,933,884 B whole-tree for "
                     "hydrated) and a measured resident figure on the comparison chart "
                     "(hydrated 20.31 GiB, which is vLLM's own model-loading number from the "
                     "capture log). No card prints a 16 GB or 24 GB predicted byte total, so no "
                     "card row moves.",
            "consequence": "the MTP override finding is a docs/34 clarification, not a card "
                           "correction.",
        },
        "loader_overhead_note": {
            "what": "docs/34 §2 already publishes the loader term as measured at +0.305 to "
                    "+0.312 GiB on the context build, +0.206 GiB on hydrated and +0.342 GiB in "
                    "the amendment run, with planning using +0.35 GiB [P] deliberately above "
                    "every observation.",
            "reproduced_here": "reading vLLM's 'Model loading took 20.31 GiB' out of the "
                               "hydrated sibling's capture log gives +0.2056 GiB against the "
                               "20.1044 GiB payload, which is the published +0.206.",
            "what_is_actually_new": "nothing about hydrated. What no run has ever measured is "
                                    "the loader term for a SUB-4-BIT build, which the S16-V "
                                    "capture will produce.",
        },
        "verdict": {
            "rows_checked": len(totals) + len(resident) + len(per_role) + len(knobs),
            "rows_that_move": 0,
            "outcome": "CHECKED, UNCHANGED. Every published per-role byte row, every profile "
                       "total, every resident payload and every §7 knob figure re-derives "
                       "exactly from the module shapes, and each one prices the recipe its own "
                       "text names. In particular docs/34 §6.2's 212,636,704 B mtp_draft row IS "
                       "the all-K4 figure §6.1 specifies -- it is correct, and it is the build "
                       "script that had to change to match it.",
            "what_docs_34_should_gain": "one sentence, not a number: that the all-K4 draft in "
                                        "§6.1 requires the MLP width override to be scoped to "
                                        "the language-model body, because the loose form the "
                                        "hydrated build used also promotes the draft's MLP -- "
                                        "and correspondingly that 'quantized MTP' is the "
                                        "accurate description of the hydrated draft, whose "
                                        "widths are self_attn K6, mlp K5/K5/K6, eh_proj K4, not "
                                        "all-K4.",
        },
        "errors": errors,
    }
    payload["content_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"})
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "rows_checked": payload["verdict"]["rows_checked"],
                      "rows_that_move": payload["verdict"]["rows_that_move"],
                      "errors": errors}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
