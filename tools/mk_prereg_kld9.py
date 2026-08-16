#!/usr/bin/env python3
"""Emit receipts/preregistration-kld9-window.json.

Pre-registration for the three conversions of the kld9 window: the S16-V sub-4-bit
flip condition, the alt-calibration hydrated rebuild, and the K6-parity build.
Every prediction in the output is computed here from named published receipts, so
each number can be re-derived by re-running this script.  Written and committed
BEFORE any of the three conversion commands is executed.
"""
import json, math, os, subprocess, hashlib

R = "/home/mbelleau/research"
def j(p): return json.load(open(os.path.join(R, p)))

# ---------------------------------------------------------------- published anchors
shard0 = {}
for tag in ("hyd", "k5k6", "k4", "ctx", "fp8"):
    d = json.load(open("/var/tmp/work/kld5/reports/shard-0000/report-%s.json" % tag))
    shard0[tag] = d
q6k = j("receipts/gguf-report-q6_k.json")
floor = j("receipts/gguf-report-engine-floor.json")
comp = j("receipts/cross-engine-comparator.json")
sib = j("receipts/sibling-rebuild-fidelity.json")
manifest = j("receipts/hydrated-quantization-manifest.json")
ladder = j("receipts/error-driven-ladder.json")
surr = j("receipts/error-driven-surrogate-calibration.json")

HYD = shard0["hyd"]["token_mean_kld"]
K4 = shard0["k4"]["token_mean_kld"]
CTX = shard0["ctx"]["token_mean_kld"]
K5K6 = shard0["k5k6"]["token_mean_kld"]
Q6K = q6k["token_mean_kld"]
FLOOR = floor["token_mean_kld"]

# ---------------------------------------------------------------- byte predictions
roles = manifest["roles"]
HYD_PAYLOAD = sum(v["bytes"] for v in roles.values())
GATE_K5 = roles["mlp_gate_proj"]["bytes"]
DOWN_K6 = roles["mlp_down_proj"]["bytes"]
ONE_BIT_MLP_ROLE = DOWN_K6 - GATE_K5                      # 64 modules, K5 -> K6
MODS = ladder["modules"]
MTP_GU_NUMEL = MODS["mtp.layers.0.mlp.gate_proj"]["numel"] + MODS["mtp.layers.0.mlp.up_proj"]["numel"]
MTP_EXTRA = MTP_GU_NUMEL // 8                              # 1 bit/param
PAR_PAYLOAD = HYD_PAYLOAD + 2 * ONE_BIT_MLP_ROLE + MTP_EXTRA
DISK_RULE = 24_000_000
GIB = 2 ** 30
S16_PAYLOAD = 13_711_503_428                               # docs/34 §6.2 [P]
S16_DISK = 13_735_474_746

# quantized-weight bytes (payload minus the two BF16 roles) -- the base of the bpw law
def quantized_bytes(payload, embed, vision, norms):
    return payload - embed - vision - norms
HYD_QUANT = quantized_bytes(HYD_PAYLOAD, roles["embed_tokens"]["bytes"],
                            roles["vision_tower"]["bytes"], roles["norms_and_small"]["bytes"])
S16_QUANT = S16_PAYLOAD - 2_542_796_800 - 921_460_192 - 1_320_960
QUANT_PARAMS = 25.6e9   # docs/29 L190: ~25.6B quantized weights, the base of the +0.397 bpw figure
R_PER_BIT = 3.73        # docs/37 §3.1/3.2: per-module geometric constant 3.7294 +- 0.0536
R_K5K6 = 3.7390         # docs/37 §3.1: measured MLP K5->K6 rung ratio

def bpw(delta_bytes): return delta_bytes * 8 / QUANT_PARAMS

# ---------------------------------------------------------------- surrogate (sqrt_energy)
def role_of(n):
    for key, r in (("mlp.gate_proj", "mlp_gate"), ("mlp.up_proj", "mlp_up"),
                   ("mlp.down_proj", "mlp_down"), ("linear_attn", "linear_attn"),
                   ("self_attn", "full_attn")):
        if key in n:
            return r
    return "other"
body = [n for n in MODS if n.startswith("model.language_model.layers")]
def wsq(m): return math.sqrt(m["out_energy"])
def eps(m, b):
    return m["ladder"].get(str(b), m["ladder"].get(b))
def eps_x(m, b):
    e = eps(m, b)
    if e is not None:
        return e, False
    e4, e5 = eps(m, 4), eps(m, 5)
    return e4 * (e4 / e5), True       # only ever needed at K3 for numel < 52M modules

OBJ_HYD = sum(wsq(MODS[n]) * eps(MODS[n], MODS[n]["recipe_bits"]) for n in body)
OBJ_PAR = sum(wsq(MODS[n]) * eps(MODS[n], 6 if role_of(n) in ("mlp_gate", "mlp_up")
                                 else MODS[n]["recipe_bits"]) for n in body)
S16W = {"full_attn": 4, "linear_attn": 3, "mlp_gate": 3, "mlp_up": 3, "mlp_down": 3}
OBJ_S16, n_extrap = 0.0, 0
for n in body:
    e, x = eps_x(MODS[n], S16W[role_of(n)])
    OBJ_S16 += wsq(MODS[n]) * e
    n_extrap += x
# self-check: reproduce the published mlp_to_K4 sqrt_energy objective delta
K4W = {"full_attn": 6, "linear_attn": 6, "mlp_gate": 4, "mlp_up": 4, "mlp_down": 4}
OBJ_K4 = sum(wsq(MODS[n]) * eps(MODS[n], K4W[role_of(n)]) for n in body)
CHECK_PUB = surr["pairs"]["mlp_to_K4"]["objective_delta"]["sqrt_energy"]["delta"]
CHECK_MINE = OBJ_K4 - OBJ_HYD
SCALES = {k: v for k, v in surr["rules"]["sqrt_energy"]["implied_scale_per_pair"].items()}
UNIFORM = ("hydrated_to_context", "mlp_to_K4")   # the two uniform role-group pairs

# role-share bound: share of the sqrt-energy-weighted error carried by gate+up
SHARE_GU = sum(wsq(MODS[n]) * eps(MODS[n], MODS[n]["recipe_bits"])
               for n in body if role_of(n) in ("mlp_gate", "mlp_up")) / OBJ_HYD
ROLE_SHARE_FACTOR = (1 - SHARE_GU) + SHARE_GU / R_K5K6

# ---------------------------------------------------------------- corpora digests
def digest_dir(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".utf8"):
            b = open(os.path.join(d, f), "rb").read()
            out[f] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return out
DEF_CAL = "/var/tmp/work/exllamav3/exllamav3/conversion/standard_cal_data"
ALT_CAL = "/var/tmp/work/kld9/exllamav3-altcal/exllamav3/conversion/standard_cal_data"
def_files, alt_files = digest_dir(DEF_CAL), digest_dir(ALT_CAL)
raw_books = {}
for line in open("/var/tmp/work/kld9/altcal/raw-sha256.txt"):
    h, p = line.split()
    raw_books[os.path.basename(p)] = h
calrows = {}
for tag, p in (("default", "/var/tmp/work/kld9/calrows-default.json"),
               ("alt", "/var/tmp/work/kld9/calrows-altcal.json")):
    if os.path.exists(p):
        d = json.load(open(p))
        calrows[tag] = {k: d[k] for k in ("cal_rows", "cal_cols", "total_tokens",
                                          "token_rows_sha256_int64_feed_order",
                                          "tokenizer_json_sha256")}

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()
CODE_DIFF = sh("bash", "-c",
               "cd /var/tmp/work/exllamav3 && git diff | sha256sum | cut -d' ' -f1")
CODE_DIFF_ALT = sh("bash", "-c",
                   "cd /var/tmp/work/kld9/exllamav3-altcal && git diff -- . "
                   "':(exclude)exllamav3/conversion/standard_cal_data' | sha256sum | cut -d' ' -f1")
WT = sh("bash", "-c", "cd /var/tmp/work/exllamav3 && git rev-parse HEAD")

PROTOCOL = {
    "what": "shard 0 of the v5 suite, the identical published protocol",
    "suite_token_sha256": shard0["hyd"]["suite_token_sha256"],
    "contexts": shard0["hyd"]["contexts"],
    "scored_positions": shard0["hyd"]["scored_positions"],
    "reference_capture": "/work/gguf/hidden-bf16",
    "reference_capture_control": {
        "why": "the published shard-0 reports captured the BF16 reference at "
               "/work/kld5/hidden/shard-0000/hidden-bf16; this window reuses the surviving "
               "/work/gguf/hidden-bf16 capture instead",
        "control": "/var/tmp/work/kld6/reports/report-hyd-rematch.json re-scored the published "
                   "hydrated checkpoint against /work/gguf/hidden-bf16 and reproduced the "
                   "published mean BITWISE (0.002699883159684943), paired difference exactly 0.0 "
                   "on all 512 contexts (receipts/sibling-rebuild-fidelity.json -> "
                   "capture_noise_floor)",
        "consequence": "reports produced here are directly pairable against the published "
                       "shard-0 reports",
    },
    "head": "/work/kld2/lm_head.safetensors",
    "head_sha256": shard0["hyd"]["head_sha256"],
    "harness": {
        "tool": "tools/fidelity.py",
        "copy_used": "/work/kld9/tools/fidelity.py",
        "note": "byte-identical to the copy the sibling-rebuild run scored with",
    },
    "capture_flags": "--gpu-memory-utilization 0.85, VLLM_EXL3_PREFILL_RECONSTRUCT_M=128, "
                     "quantization exl3 with /work/kld4/qcfg.json",
    "aggregation": "token_mean_kld with a cluster bootstrap over 330 source clusters, "
                   "10,000 resamples; every comparison is a PAIRED per-context difference",
    "tail_row_required": "docs/34's flip item (1) asks for the tail row as well as the mean: "
                         "p95, p99, p99.9, max and top-1 agreement are reported for every "
                         "candidate, from the same report",
}

TOOLCHAIN = {
    "converter_worktree": WT,
    "converter_version": "exllamav3 v1.4.2",
    "worktree_diff_sha256": CODE_DIFF,
    "worktree_diff_sha256_expected": "578066cda60415e0376f261b7c907baea2baa8239076260013208d8e70c192f2",
    "alt_tree_code_only_diff_sha256": CODE_DIFF_ALT,
    "alt_tree_code_identity": CODE_DIFF_ALT == CODE_DIFF,
    "alt_tree_code_identity_note": "the alt-calibration tree's diff EXCLUDING "
        "exllamav3/conversion/standard_cal_data hashes identically to the published converter's "
        "full diff, so the two trees differ in calibration DATA only, not in code",
    "runner": "tools/ggrun.sh over the extracted image rootfs; marisa_trie supplied at "
              "/work/convertdeps/site for the convert and quantconfig steps",
    "finalize": "/work/kld3/finalize_checkpoint.py -m OUT --upstream /models/Qwen3.8-27B",
    "host": "rental RTX PRO 6000 Blackwell Server Edition, 97,887 MiB",
}

# ---------------------------------------------------------------- conditions
c1 = {
    "order": 1,
    "name": "S16-V sub-4-bit flip",
    "receipt_to_be_written": "receipts/sixteen-flip-kld.json",
    "why": "docs/34 §6 and receipts/vram-class-verdict.json rest the 16 GB NO-GO on an ABSENCE: "
           "no width below 4 bits has ever been measured for KLD in this family, at any role, on "
           "any suite. docs/34's flip list item (1) is exactly this measurement -- one conversion, "
           "one shard, no 16 GB card needed -- and calls it the blocking item.",
    "recipe": {
        "source": "docs/34 §6.1, the S16-V primary column",
        "full_attention_16L": "K4",
        "linear_attention_48L": "K3",
        "mlp_gate_up_down": "K3 / K3 / K3",
        "lm_head": "K4/mcg",
        "mtp_draft": "all K4",
        "embed_tokens": "BF16 on disk",
        "vision_tower": "BF16",
    },
    "exact_environment": {
        "EXL3_BITS_FIXED": '{"^.*self_attn\\\\..*$": 4, "^.*linear_attn\\\\..*$": 3}',
        "EXL3_BITS_OVERRIDE": '{"^.*mlp\\\\.(gate|up|down)_proj$": 3}',
    },
    "exact_command": "convert.py -i /models/Qwen3.8-27B -o /work/kld9/qwen38-s16v "
                     "-w /work/kld9/wd-s16v -b 3 -hb 4 -mb 4 -vb 16 -cb mcg",
    "build_script": "/var/tmp/work/kld9/build_s16v.sh",
    "predicted_bytes": {
        "serialized_payload": S16_PAYLOAD,
        "serialized_gib": round(S16_PAYLOAD / GIB, 3),
        "disk_bytes": S16_DISK,
        "source": "docs/34 §6.2, byte law summed over roles; disk = payload + 24.0 MB",
        "label": "prediction",
        "byte_law_accuracy_on_the_last_build": {
            "hydrated_predicted_disk": HYD_PAYLOAD + DISK_RULE,
            "hydrated_measured_immutable_payload_bytes": j("receipts/hydrated-build-receipt.json")["immutable_payload_bytes"],
            "error_bytes": HYD_PAYLOAD + DISK_RULE - j("receipts/hydrated-build-receipt.json")["immutable_payload_bytes"],
        },
    },
    "predicted_mean_kld": {
        "registered_range": [0.03, 0.10],
        "registered_range_source": "docs/34 §6.4 [P, extrapolation only]",
        "point_estimate_byte_law": {
            "value": None,   # filled below
            "derivation": None,
        },
        "point_estimate_surrogate": {
            "bracket": None,
            "derivation": None,
        },
    },
    "acceptance_question": "does sub-4-bit fidelity justify a 16 GB SKU?",
    "registered_decision_rule": {
        "YES": "measured shard-0 token_mean_kld <= K4's %.6f -- sub-4-bit costs no more than the "
               "worst ALREADY PUBLISHED candidate. Flip item (1) passes outright; the 16 GB "
               "no-go would then rest only on items (2)-(4) and the fidelity argument is "
               "withdrawn." % K4,
        "BORDERLINE": "measured mean in (%.6f, 0.030] -- worse than every published candidate but "
                      "below docs/34 §6.4's predicted range. Item (1) is answered with a number "
                      "that a card may print; the SKU verdict stays NO-GO on the arithmetic and "
                      "the remaining items, and the checkpoint is preserved as a research repo." % K4,
        "NO": "measured mean > 0.030, i.e. inside or above the predicted range. Sub-4-bit "
              "fidelity does not justify a 16 GB SKU; docs/34 §6 stands as a design study with "
              "item (1) now measured rather than absent.",
        "tail_side_condition": "reported alongside, not part of the rule: docs/34 calls K4's "
                               "p99.9 of 0.5555 (10M suite) already the worst of the five "
                               "published candidates. The shard-0 p99.9 and max of S16-V are "
                               "reported in every case.",
    },
    "pairings": ["published K4 (report-k4.json, mean %.6f)" % K4,
                 "published context edition (report-ctx.json, mean %.6f)" % CTX],
    "preservation_policy": "clear NO -> receipt records the recompute cost and the checkpoint is "
                           "deleted; BORDERLINE or YES -> uploaded as a research repo like the "
                           "error-driven one, and Main is told.",
}

c2 = {
    "order": 2,
    "name": "alt-calibration hydrated rebuild",
    "receipt_to_be_written": "receipts/calibration-corpus-kld.json",
    "why": "docs/37 settled ALLOCATION at a fixed byte budget and left CONTENT untested. This is "
           "the last zero-byte fidelity lever: identical widths, identical byte sizes, a "
           "deliberately different calibration corpus.",
    "recipe": "the published hydrated recipe, unchanged: MLP gate/up K5, down K6, attention K6 "
              "serialized (calibrated), lm_head K6/mcg, quantized MTP, BF16 embed+vision",
    "exact_environment": {
        "EXL3_BITS_FIXED": '{"^.*self_attn\\\\..*$": 6, "^.*linear_attn\\\\..*$": 6}',
        "EXL3_BITS_OVERRIDE": '{"^.*mlp\\\\.down_proj$": 6, "^.*mlp\\\\.(gate|up)_proj$": 5}',
    },
    "exact_command": "convert.py -i /models/Qwen3.8-27B -o /work/kld9/qwen38-hyd-altcal "
                     "-w /work/kld9/wd-altcal -b 4 -hb 6 -mb 4 -vb 16 -cb mcg, run from "
                     "/work/kld9/exllamav3-altcal",
    "build_script": "/var/tmp/work/kld9/build_hyd_altcal.sh",
    "the_one_change": {
        "what": "exllamav3/conversion/standard_cal_data -- the six .utf8 files the converter's own "
                "get_default_calibration reads -- replaced by same-byte-size samples drawn from a "
                "disjoint public source",
        "default_corpus": {
            "what": "exllamav3's shipped standard calibration data: c4, code, multilingual, "
                    "technical, tiny, wiki",
            "files": def_files,
        },
        "alt_corpus": {
            "what": "Project Gutenberg literary prose, 13 complete public-domain books, cut to "
                    "the SAME byte size as each default file and formatted for the same reader "
                    "(wiki.utf8 as <doc> articles, tiny.utf8 as <|endoftext|>-separated chunks, "
                    "c4/multilingual as paragraph-per-line, code/technical as raw sequential text)",
            "builder": "/var/tmp/work/kld9/altcal/build_altcal_corpus.py",
            "source_books_sha256": raw_books,
            "files": alt_files,
            "byte_sizes_identical_to_default": all(
                def_files[k]["bytes"] == alt_files[k]["bytes"] for k in def_files),
            "digests_all_differ": all(def_files[k]["sha256"] != alt_files[k]["sha256"]
                                      for k in def_files),
            "disjointness": "19th-century literary prose only. No web text, no source code, no "
                            "technical documentation, no encyclopaedia text; the multilingual "
                            "file is French/German/Spanish/Italian literary prose rather than the "
                            "shipped mix. Deliberately the worst realistic domain mismatch that "
                            "is still natural language.",
        },
        "calibration_rows_digested": calrows,
        "row_pipeline_unchanged": "250 rows x 2048 tokens (211 text rows + 39 seeded random rows), "
                                 "the same reader functions, the same weights",
    },
    "predicted_bytes": {
        "serialized_payload": HYD_PAYLOAD,
        "serialized_gib": round(HYD_PAYLOAD / GIB, 4),
        "expectation": "byte SIZES identical to the published hydrated build file for file, "
                       "tensor bytes different. EXL3 K5/K6 module sizes are fixed by width and "
                       "shape, so no size can move; only the packed values can.",
        "control": "the sibling rebuild with the DEFAULT corpus already differed in the tensor "
                   "bytes of 3 of 16 files at identical sizes "
                   "(/var/tmp/work/kld8/sibling-bytes.json, receipts/converter-determinism.json)",
    },
    "predicted_mean_kld": {
        "registered_point": None,   # filled below
        "registered_direction": "positive (alt-calibration worse than published hydrated)",
        "registered_interval": [1e-05, 2e-04],
        "derivation": None,
    },
    "acceptance_question": "does calibration CONTENT matter at these widths, at zero byte cost?",
    "registered_decision_rule": {
        "content_does_not_matter": "the paired 95% CI of (alt - published) contains zero AND "
            "|difference| <= 2.9e-05, the half-width of the same-corpus sibling-rebuild control "
            "(paired -3.755e-06 [-2.854e-05, +2.062e-05]). Then converter-draw noise bounds the "
            "whole content effect and the lever is closed at this width.",
        "content_matters": "the paired 95% CI excludes zero. The magnitude is then reported as a "
            "fraction of the smallest published margin this project draws conclusions from "
            "(hydrated vs error-driven, 0.000366) and of a one-bit move (hydrated vs context, "
            "attention K6->K5, 0.000710).",
        "either_way": "publishable. docs/37's framing applies: allocation solved, content tested.",
    },
    "pairings": ["published hydrated (report-hyd.json, mean %.6f)" % HYD],
    "preservation_policy": "the checkpoint is a same-bytes sibling of a published artifact and "
                           "carries no release value; recompute cost recorded, deleted after "
                           "scoring unless the effect is large enough to want the artifact.",
}

c3 = {
    "order": 3,
    "name": "K6-parity",
    "receipt_to_be_written": "receipts/k6-parity-kld.json",
    "why": "docs/29 L189-198: the Q6_K loss is a byte gap, not an engineering gap, and that is "
           "testable. Q6_K spends +1.183 GiB over hydrated's payload across ~25.6B quantized "
           "weights = +0.397 bpw; the 3.73x-per-bit constant predicts a 1.69x KLD advantage for "
           "that spend against an observed 1.77x net-of-floor, i.e. Q6_K sits ON our own scaling "
           "curve. The parity build asks whether our recipe matches or beats Q6_K at equal bytes.",
    "recipe": "the published hydrated recipe with exactly ONE change: MLP gate_proj and up_proj "
              "promoted K5 -> K6. down stays K6, attention K6 serialized, lm_head K6/mcg, MTP as "
              "hydrated, BF16 embed+vision.",
    "prediction_class": "uniform role-group promotion -- the class the EDA surrogate calibration "
                        "found sign-correct, unlike the between-role reallocation at a fixed "
                        "budget that it failed "
                        "(receipts/error-driven-surrogate-calibration.json)",
    "exact_environment": {
        "EXL3_BITS_FIXED": '{"^.*self_attn\\\\..*$": 6, "^.*linear_attn\\\\..*$": 6}',
        "EXL3_BITS_OVERRIDE": '{"^.*mlp\\\\.(gate|up|down)_proj$": 6}',
    },
    "exact_command": "convert.py -i /models/Qwen3.8-27B -o /work/kld9/qwen38-k6parity "
                     "-w /work/kld9/wd-k6parity -b 4 -hb 6 -mb 4 -vb 16 -cb mcg",
    "build_script": "/var/tmp/work/kld9/build_k6parity.sh",
    "predicted_bytes": {
        "serialized_payload": PAR_PAYLOAD,
        "serialized_gib": round(PAR_PAYLOAD / GIB, 3),
        "disk_bytes": PAR_PAYLOAD + DISK_RULE,
        "derivation": "hydrated payload %d + 2 x %d (gate and up, 64 modules each, K5->K6 costs "
                      "exactly the byte difference between the published K5 gate role and the "
                      "published K6 down role) + %d (the MTP layer's gate and up, which the "
                      "override regex also matches, at 1 bit per parameter) = %d"
                      % (HYD_PAYLOAD, ONE_BIT_MLP_ROLE, MTP_EXTRA, PAR_PAYLOAD),
        "byte_parity_target": {
            "gguf_q6_k_serialized_gib": 21.313,
            "ratio_parity_over_q6k": round((PAR_PAYLOAD / GIB) / 21.313, 4),
        },
        "label": "prediction",
    },
    "predicted_mean_kld": {
        "registered_headline": 0.0016,
        "registered_headline_source": "docs/29 L195, the figure already on the record",
        "estimators": None,   # filled below
        "registered_interval": None,
    },
    "acceptance_question": "does the parity build land WITHIN [Q6_K net %.6f, Q6_K measured "
                           "%.6f], or BEAT it (below the net figure)?" % (0.0015278671188742878, Q6K),
    "registered_decision_rule": {
        "BEATS_Q6K": "measured mean < %.6f (Q6_K's net-of-floor estimate). Our recipe is better "
                     "than GGUF Q6_K at equal bytes; the 6-bit-class loss in "
                     "receipts/cross-engine-comparator.json was a byte gap and is now closed. "
                     "Release candidate for a fifth repo." % 0.0015278671188742878,
        "MATCHES_Q6K": "measured mean in [%.6f, %.6f], i.e. between Q6_K net and Q6_K measured. "
                       "Indistinguishable from Q6_K at equal bytes once the cross-engine floor's "
                       "own slack is admitted; still a release candidate."
                       % (0.0015278671188742878, Q6K),
        "MISSES": "measured mean > %.6f (Q6_K's measured upper bound). Q6_K is better than our "
                  "recipe at equal bytes and the loss is NOT purely a byte gap; the byte-law "
                  "prediction is falsified in magnitude and docs/29's decomposition must be "
                  "rewritten." % Q6K,
        "owner_override_on_preservation": "the checkpoint is KEPT AND UPLOADED whatever the "
                                          "result -- owner priority, stated at scope-addition "
                                          "time.",
    },
    "pairings": [
        "published hydrated (report-hyd.json, mean %.6f) -- same engine, same reference capture, "
        "no cross-engine term" % HYD,
        "GGUF Q6_K (receipts/gguf-report-q6_k.json, mean %.6f) -- SAME 512 contexts and the same "
        "suite token, but the candidate was captured in llama.cpp, so this pairing carries the "
        "cross-engine term measured at %.6f mean (receipts/gguf-report-engine-floor.json). A "
        "GGUF number is an upper bound on its quantization error; the net figure is the naive "
        "subtraction and KL is not additive, so the net column is an estimate, not an identity."
        % (Q6K, FLOOR),
    ],
    "preservation_policy": "keep and upload the checkpoint in every branch of the decision rule.",
}

# ---- fill the computed predictions -------------------------------------------------
s16_bpw_deficit = bpw(HYD_QUANT - S16_QUANT)
s16_mult = R_PER_BIT ** s16_bpw_deficit
c1["predicted_mean_kld"]["point_estimate_byte_law"] = {
    "value": round(HYD * s16_mult, 4),
    "derivation": "hydrated quantized-weight bytes %d - S16-V quantized-weight bytes %d = %d "
                  "fewer bytes over ~%.1fB quantized weights = -%.3f bpw; the 3.73x-per-bit "
                  "constant gives %.1fx more KLD, so %.6f x %.1f = %.4f"
                  % (HYD_QUANT, S16_QUANT, HYD_QUANT - S16_QUANT, QUANT_PARAMS / 1e9,
                     s16_bpw_deficit, s16_mult, HYD, s16_mult, HYD * s16_mult),
    "caveats": "the uniform-bpw form assumes error is spread in proportion to bytes; S16-V moves "
               "five role groups at once, three of them by three bits, and the 3.73x constant is "
               "measured only over K3..K8 rungs of a proxy error, never against KLD below 4 bits. "
               "Extrapolation, not an estimate.",
}
s16_lo = HYD + (OBJ_S16 - OBJ_HYD) * SCALES["hydrated_to_context"]
s16_hi = HYD + (OBJ_S16 - OBJ_HYD) * SCALES["mlp_to_K4"]
c1["predicted_mean_kld"]["point_estimate_surrogate"] = {
    "bracket": [round(s16_lo, 4), round(s16_hi, 4)],
    "derivation": "sqrt_energy objective (w_m = sqrt(out_energy)) over the 400 laddered body "
                  "modules: hydrated %.4f, S16-V %.1f, delta %.1f. Scaled by the two implied "
                  "scales the published surrogate calibration fitted on its two UNIFORM "
                  "role-group pairs (%.6f from attention K6->K5, %.6f from MLP->K4) this gives "
                  "%.4f to %.4f."
                  % (OBJ_HYD, OBJ_S16, OBJ_S16 - OBJ_HYD, SCALES["hydrated_to_context"],
                     SCALES["mlp_to_K4"], s16_lo, s16_hi),
    "extrapolated_rungs": "%d of %d body modules have no measured K3 rung (numel below the "
                          "52M big-module threshold, so their ladder starts at K4); their K3 "
                          "proxy error is extrapolated by continuing the module's own K5->K4 "
                          "ratio one further bit." % (n_extrap, len(body)),
    "caveat": "the published calibration verdict is that NO rule is calibrated: sqrt_energy is "
              "merely the only sign-correct rule with a bounded scale spread (2.51x, worst "
              "out-of-sample ratio 2.47x). Treat this bracket as an order-of-magnitude check on "
              "docs/34 §6.4's range, which it corroborates.",
}
c1["predicted_mean_kld"]["registered_primary"] = round(HYD * s16_mult, 4)
c1["predicted_mean_kld"]["registered_falsification"] = (
    "the registered range is 0.03-0.10. A measured mean below 0.02 or above 0.20 falsifies both "
    "estimators and the design study's fidelity paragraph has to be rewritten.")

par_bpw = bpw(2 * ONE_BIT_MLP_ROLE + MTP_EXTRA)
par_byte_law = HYD / (R_PER_BIT ** par_bpw)
par_lo = HYD + (OBJ_PAR - OBJ_HYD) * SCALES["mlp_to_K4"]
par_hi = HYD + (OBJ_PAR - OBJ_HYD) * SCALES["hydrated_to_context"]
c3["predicted_mean_kld"]["estimators"] = {
    "byte_law_at_q6k_surplus": {
        "value": round(HYD / (R_PER_BIT ** 0.397), 6),
        "derivation": "hydrated %.6f / 3.73^0.397 = %.6f. This is docs/29's ~0.0016: it charges "
                      "the parity build Q6_K's byte surplus rather than its own."
                      % (HYD, HYD / (R_PER_BIT ** 0.397)),
    },
    "byte_law_at_own_spend": {
        "value": round(par_byte_law, 6),
        "derivation": "the parity build spends %d bytes = %.3f GiB over hydrated, which over "
                      "~%.1fB quantized weights is +%.4f bpw; 3.73^%.4f = %.3fx, so %.6f / %.3f "
                      "= %.6f"
                      % (2 * ONE_BIT_MLP_ROLE + MTP_EXTRA,
                         (2 * ONE_BIT_MLP_ROLE + MTP_EXTRA) / GIB, QUANT_PARAMS / 1e9,
                         par_bpw, par_bpw, R_PER_BIT ** par_bpw, HYD, R_PER_BIT ** par_bpw,
                         par_byte_law),
        "note": "the parity build buys MORE bytes than Q6_K's surplus (%.3f GiB against 1.183), "
                "so its own-spend prediction sits below docs/29's headline."
                % ((2 * ONE_BIT_MLP_ROLE + MTP_EXTRA) / GIB),
    },
    "role_share_bound": {
        "value": round(HYD * ROLE_SHARE_FACTOR, 6),
        "derivation": "gate+up carry %.1f%% of the sqrt-energy-weighted proxy error at hydrated "
                      "widths; promoting them one bit divides that share by the measured MLP "
                      "K5->K6 rung ratio %.4f (docs/37 §3.1), leaving a factor "
                      "(1-%.4f)+%.4f/%.4f = %.4f, so %.6f x %.4f = %.6f"
                      % (100 * SHARE_GU, R_K5K6, SHARE_GU, SHARE_GU, R_K5K6,
                         ROLE_SHARE_FACTOR, HYD, ROLE_SHARE_FACTOR, HYD * ROLE_SHARE_FACTOR),
        "caveat": "assumes the whole KLD scales with the weighted proxy error and that the "
                  "un-promoted roles contribute no floor of their own. This is the most "
                  "aggressive of the three and is registered as the lower end.",
    },
    "surrogate_scaled": {
        "bracket": [round(par_lo, 6), round(par_hi, 6)],
        "derivation": "sqrt_energy objective delta for gate/up K5->K6 is %.4f; scaled by the two "
                      "uniform role-group scales (%.6f, %.6f) that is %+.6f to %+.6f on the "
                      "hydrated mean." % (OBJ_PAR - OBJ_HYD, SCALES["mlp_to_K4"],
                                          SCALES["hydrated_to_context"],
                                          par_lo - HYD, par_hi - HYD),
    },
}
c3["predicted_mean_kld"]["registered_interval"] = [round(min(HYD * ROLE_SHARE_FACTOR, par_lo), 6),
                                                   round(HYD / (R_PER_BIT ** 0.397), 6)]
c3["predicted_mean_kld"]["registered_primary"] = round(par_byte_law, 6)
c3["predicted_mean_kld"]["registered_consequence"] = (
    "every estimator predicts the parity build BEATS Q6_K's net-of-floor %.6f or matches it. The "
    "sharp pre-registered claim is therefore BEATS_Q6K or MATCHES_Q6K; MISSES falsifies the byte "
    "decomposition." % 0.0015278671188742878)

c2["predicted_mean_kld"]["registered_point"] = 0.00275
c2["predicted_mean_kld"]["derivation"] = (
    "no law predicts this one -- there is no calibration-content term in any model this project "
    "has fitted, which is the reason the experiment exists. The registered prediction is "
    "reasoned: exllamav3 accumulates its Hessian over 512,000 calibration tokens, and the "
    "quantization error at K5/K6 is dominated by weight geometry rather than by which natural "
    "text drove the activations, so the effect should be SMALL; but the alt corpus is a "
    "deliberate domain mismatch against a suite that is not 19th-century literature, so it "
    "should be non-zero and POSITIVE. Registered: +1e-05 to +2e-04 on a hydrated base of "
    "%.6f, i.e. between the %.2e converter-draw noise floor and a quarter of the %.6f one-bit "
    "attention move. Registered point %.5f (hydrated + 5e-05)." % (
        HYD, 3.7550331881264944e-06, 0.0007095267975016164, 0.00275))

doc = {
    "schema": "qwen38-preregistration/1",
    "title": "Pre-registration: three conversions and three shard-0 scores (S16-V sub-4-bit flip, "
             "alt-calibration hydrated, K6-parity)",
    "why_this_file_exists": "all three conversions are prediction tests. This file records the "
        "exact recipe, the exact command, the predicted bytes, the predicted KLD with its "
        "derivation and the decision rule for each of them, and is committed and pushed BEFORE "
        "any of the three conversion commands is executed. The commit that adds this file is the "
        "registration; the conversion logs carry UTC timestamps that are strictly later.",
    "author": "SixteenFlipCalib2",
    "window": "rental RTX PRO 6000, handed over by RentalFidelityBatch2",
    "protocol": PROTOCOL,
    "toolchain": TOOLCHAIN,
    "published_anchors": {
        "shard_0_reports": {
            tag: {
                "report": "/var/tmp/work/kld5/reports/shard-0000/report-%s.json" % tag,
                "token_mean_kld": d["token_mean_kld"],
                "ci95": [d["context_bootstrap"]["ci95_low"], d["context_bootstrap"]["ci95_high"]],
                "p999_kld": d["p999_kld"],
                "max_kld": d["max_kld"],
                "top1_agreement": d["top1_agreement"],
            } for tag, d in shard0.items()
        },
        "gguf_q6_k": {
            "report": "receipts/gguf-report-q6_k.json",
            "token_mean_kld": Q6K,
            "ci95": [q6k["context_bootstrap"]["ci95_low"], q6k["context_bootstrap"]["ci95_high"]],
            "net_of_engine_floor_estimate": 0.0015278671188742878,
            "serialized_gib": 21.313,
            "engine": "llama.cpp, unsloth Qwen3.8-27B-Q6_K.gguf",
        },
        "cross_engine_floor": {
            "report": "receipts/gguf-report-engine-floor.json",
            "token_mean_kld": FLOOR,
        },
        "converter_draw_noise_floor": {
            "receipt": "receipts/sibling-rebuild-fidelity.json",
            "paired_difference": sib["verdict"]["paired_difference_published_minus_sibling"],
            "ci95": [sib["verdict"]["paired_ci95_low"], sib["verdict"]["paired_ci95_high"]],
            "meaning": "the same recipe rebuilt with the same corpus on the same toolchain is "
                       "indistinguishable from the published build. This is the control that "
                       "makes condition 2 interpretable.",
        },
        "capture_noise_floor": "exactly 0.0 on all 512 contexts (the published checkpoint "
                              "re-captured and re-replayed), receipts/sibling-rebuild-fidelity"
                              ".json -> capture_noise_floor",
        "smallest_margin_this_project_draws_conclusions_from": {
            "hydrated_vs_error_driven": 0.0003662964754934233,
            "hydrated_vs_context_one_bit_attention": 0.0007095267975016164,
        },
    },
    "surrogate_method": {
        "what": "second, independent estimator for conditions 1 and 3: exllamav3's own per-module "
                "Hessian-weighted relative proxy error, laddered at K3..K8 in "
                "receipts/error-driven-ladder.json, weighted by sqrt(out_energy) and scaled by "
                "the implied scale the published calibration fitted on a uniform role-group move",
        "self_check": {
            "what": "recomputing the published mlp_to_K4 sqrt_energy objective delta from the "
                    "ladder with this script's own code",
            "published": CHECK_PUB,
            "recomputed": CHECK_MINE,
            "identical": abs(CHECK_PUB - CHECK_MINE) < 1e-9,
        },
        "published_verdict_on_the_surrogate": surr["verdict"]["plain"][:400],
        "scales_used": {k: SCALES[k] for k in UNIFORM},
    },
    "conditions": [c1, c2, c3],
    "order_and_disk": {
        "order": "S16-V first (highest leverage: it is docs/34's blocking flip item), "
                 "alt-calibration second, K6-parity third",
        "why_that_order": "the flip condition is the one that changes a published verdict; the "
                          "parity build is the one whose checkpoint is kept whatever happens, so "
                          "it goes last where its bytes can stay on disk until upload.",
        "disk_discipline": "each conversion's working directory is deleted immediately after its "
                           "convert step (nothing downstream reads it), each candidate's hidden "
                           "capture is deleted after its replay, and df is checked between "
                           "conversions. /var/tmp/work/kld9/hidden/hidden-{ctx,fp8,hyd,k4,nvfp4} "
                           "belongs to RentalFidelityBatch2 and is not touched.",
        "if_the_window_cannot_fit_all_three": "the remainder is handed to ShortlistScore with "
                                              "this file as the plan.",
    },
}
out = os.path.join(R, "receipts/preregistration-kld9-window.json")
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
    f.write("\n")
print("wrote", out)
print(json.dumps({
    "s16v_registered_primary": c1["predicted_mean_kld"]["registered_primary"],
    "s16v_surrogate_bracket": c1["predicted_mean_kld"]["point_estimate_surrogate"]["bracket"],
    "altcal_registered": c2["predicted_mean_kld"]["registered_point"],
    "parity_registered_primary": c3["predicted_mean_kld"]["registered_primary"],
    "parity_registered_interval": c3["predicted_mean_kld"]["registered_interval"],
    "parity_predicted_payload_gib": c3["predicted_bytes"]["serialized_gib"],
    "surrogate_self_check": doc["surrogate_method"]["self_check"]["identical"],
    "alt_tree_code_identity": TOOLCHAIN["alt_tree_code_identity"],
}, indent=1))
