#!/usr/bin/env python3
"""Turn the raw sweep output into the receipt body.

Reads kvsweep/raw/*.json (rsynced from AIBoss) and writes
kvsweep/kv-dtype-sweep-5090.json.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
GIB = 1024 ** 3

# Class pools, held from the fp8 measurements they were derived with.
CLASS_POOLS = {
    "class_24_gib": {
        "budget_gib": 22.49,
        "mtp_3": {"pool_gib": 1.79, "basis": "measured board arm, receipts/qualification-24gib-capped.json"},
        "mtp_off": {"pool_gib": 2.60, "basis": "measured board arm, docs/34 s10.3"},
        "published_fp8": {"mtp_3": 24576, "mtp_off": 45056},
    },
    "class_16_gib": {
        "budget_gib": 14.99,
        "mtp_3": {"pool_gib": 0.553, "basis": "predicted S16-V body, docs/34 s6"},
        "mtp_off": {"pool_gib": 1.298, "basis": "predicted S16-V body, docs/34 s6"},
        "published_fp8": {"mtp_3": None, "mtp_off": 28672},
    },
}
PUBLISHED_FP8 = {"mtp_3": {"a": 34816.0, "M": 0.63}, "mtp_off": {"a": 32931.8, "M": 0.14}}


def load(name: str):
    p = RAW / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- affine law

# Per-layer, per-token KV bytes, read off the pinned build's own page-size code
# (vllm/v1/kv_cache_interface.py: AttentionSpec.real_page_size_bytes plus the
# per-token-head scale term in unpadded_page_size_bytes). This model has 16
# full-attention layers with 4 KV heads and head_size = head_size_v = 256; the
# other 48 layers are Gated DeltaNet.
NUM_KV_HEADS = 4
HEAD_SIZE = 256


def unit_bytes(dtype: str) -> int | None:
    """KV bytes charged per token per full-attention layer for this dtype."""
    per_head_scales = dtype.endswith("_per_token_head")
    if dtype == "int4_per_token_head":
        last_dim = HEAD_SIZE // 2 + HEAD_SIZE // 2
        elem = 1
    elif dtype in ("bfloat16", "float16", "auto"):
        last_dim = 2 * HEAD_SIZE
        elem = 2
    elif dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "int8_per_token_head",
                   "fp8_per_token_head"):
        last_dim = 2 * HEAD_SIZE
        elem = 1
    else:
        return None
    unit = NUM_KV_HEADS * last_dim * elem
    if per_head_scales:
        unit += 2 * NUM_KV_HEADS * 4  # float32 k/v scales, per token per layer
    return unit


def derive_law(entry: dict, dtype: str | None = None,
               block_size: int | None = None) -> dict:
    """a and M from two startup refusals at one utilisation and two windows.

    Two derivations are reported. The *published form* is the arithmetic docs/34
    s10.2 used, so this sweep's fp8 arm is directly comparable with the published
    34,816 / 0.63. The *exact form* uses the engine's own block quantisation --
    `needed(L) = ceil(L/B)*B*a + M` -- which turns the two printed requirements
    into an integer count of charged layers and therefore pins `a` to a single
    integer instead of a +/- 82 B/token interval.
    """
    r1 = entry.get("refusal_131072", {}).get("refusal") or {}
    r2 = entry.get("refusal_262144", {}).get("refusal") or {}
    if r1.get("kind") != "kv_capacity_refusal" or r2.get("kind") != "kv_capacity_refusal":
        return {"derived": False, "reason": "one or both windows did not refuse"}
    n1, n2 = r1["kv_needed_gib"], r2["kv_needed_gib"]
    a1, a2 = r1["kv_available_gib"], r2["kv_available_gib"]
    dL = 262144 - 131072
    a = (n2 - n1) * GIB / dL
    m = n2 - a * 262144 / GIB
    sig_a = 0.01 * GIB / dL          # each printed GiB carries +/- 0.005
    sig_m = 0.005 + 262144 * sig_a / GIB
    out = {
        "derived": True,
        "utilisation": entry.get("util"),
        "mtp_depth": entry.get("mtp_depth", 3),
        "windows": {
            "131072": {"kv_needed_gib": n1, "kv_available_gib": a1,
                       "estimated_max_model_len": r1.get("estimated_max_model_len")},
            "262144": {"kv_needed_gib": n2, "kv_available_gib": a2,
                       "estimated_max_model_len": r2.get("estimated_max_model_len")},
        },
        "published_form": {
            "a_bytes_per_token": round(a, 1),
            "M_gib": round(m, 4),
            "a_worst_case_pm_bytes_per_token": round(sig_a, 1),
            "M_worst_case_pm_gib": round(sig_m, 4),
            "method": ("a = (N(262144) - N(131072)) / 131072, "
                       "M = N(262144) - a*262144; identical arithmetic to docs/34 s10.2"),
        },
        "a_bytes_per_token": round(a, 1),
        "M_gib": round(m, 4),
    }
    unit = unit_bytes(dtype) if dtype else None
    if unit and block_size:
        nb1, nb2 = -(-131072 // block_size), -(-262144 // block_size)
        k_float = (n2 - n1) * GIB / ((nb2 - nb1) * block_size * unit)
        k = round(k_float)
        a_exact = k * unit
        m_exact = n2 - a_exact * nb2 * block_size / GIB
        exact = {
            "attention_block_size_tokens": block_size,
            "bytes_per_token_per_full_attention_layer": unit,
            "charged_full_attention_layers": k,
            "charged_layers_unrounded": round(k_float, 4),
            "charged_layers_rounding_slack": round(abs(k_float - k), 4),
            "a_bytes_per_token": a_exact,
            "M_gib": round(m_exact, 4),
            "attention_page_bytes_per_block_per_layer": block_size * unit,
            "law": "needed(L) = ceil(L/B)*B*a + M",
            "note": ("the model has 16 full-attention layers; the extra charged layer "
                     "is the engine's own 'Add 3 padding layers, may waste at most "
                     "6.25% KV cache memory' allowance, which is exactly one sixteenth"),
        }
        # Forward check against the engine's binary-searched maximum length,
        # which is quantised to whole blocks. Both inputs are printed to 2 dp,
        # so the honest statement is the slack, not a bare pass/fail.
        for w, r in (("131072", r1), ("262144", r2)):
            e = r.get("estimated_max_model_len")
            if not e:
                continue
            av = r["kv_available_gib"]
            page = block_size * a_exact / GIB
            blocks_float = (av - m_exact) / page
            blocks = int(blocks_float)
            e_blocks = e / block_size
            exact.setdefault("forward_check", {})[w] = {
                "engine_printed_estimated_max_model_len": e,
                "e_in_whole_blocks": e_blocks,
                "predicted_from_exact_law": blocks * block_size,
                "predicted_blocks_unrounded": round(blocks_float, 4),
                "matches": blocks * block_size == e,
                "slack_gib": round(av - (e_blocks * page + m_exact), 5),
                "consistent_within_printed_precision":
                    abs(blocks_float - e_blocks) * page <= 0.01,
            }
        out["exact_form"] = exact
    return out


def refusal_block_size(entry: dict) -> int | None:
    """Attention block size, read from either refusal run's own startup banner."""
    for key in ("refusal_262144", "refusal_131072"):
        facts = (entry.get(key) or {}).get("startup_facts") or {}
        blk = facts.get("attention_block_size")
        if blk:
            return int(blk)
    return None


def published_window(a: float, m_gib: float, pool_gib: float,
                     step: int = 4096, cap: int = 262144) -> dict:
    """Largest multiple of `step` whose requirement leaves >=15 % headroom.

    Memory form of the rule: a*L + M <= pool/1.15.
    """
    limit = pool_gib / 1.15
    best = 0
    L = step
    while L <= cap:
        need = a * L / GIB + m_gib
        if need <= limit:
            best = L
        else:
            break
        L += step
    l_max = (pool_gib - m_gib) * GIB / a if pool_gib > m_gib else 0.0
    need_best = a * best / GIB + m_gib if best else None
    return {
        "pool_gib": pool_gib,
        "memory_rule_limit_gib": round(limit, 4),
        "L_max_engine_capacity": int(l_max),
        "published_window": best if best else None,
        "requirement_gib_at_published": round(need_best, 4) if need_best else None,
        "headroom_pct_at_published": (round(100 * (pool_gib / need_best - 1), 1)
                                      if need_best else None),
        "withdrawn": best == 0,
        "withdrawn_reason": (
            f"fixed per-request term M={m_gib:.4f} GiB exceeds pool/1.15="
            f"{limit:.4f} GiB before a single token" if best == 0 and m_gib > limit
            else ("no multiple of %d fits the rule" % step if best == 0 else None)),
    }


# ---------------------------------------------------------------- fidelity

def top_dicts(row: dict) -> list[dict]:
    return row.get("top_logprobs") or []


def compare_pair(base_rows: list[dict], arm_rows: list[dict]) -> dict:
    """Paired greedy comparison, restricted to the identical-prefix region."""
    per_prompt = []
    tot_positions = tot_top1 = 0
    kl_sum = 0.0
    miss_num = miss_den = 0
    dlp_abs = []
    for b, c in zip(base_rows, arm_rows):
        if b.get("repeat") or c.get("repeat"):
            continue
        assert b["prompt_sha256"] == c["prompt_sha256"], "prompts differ"
        tb, tc = b["tokens"] or [], c["tokens"] or []
        n = min(len(tb), len(tc))
        div = n
        for i in range(n):
            if tb[i] != tc[i]:
                div = i
                break
        paired = min(div + 1, n)  # position div is still conditioned identically
        pb, pc = top_dicts(b), top_dicts(c)
        p_top1 = 0
        p_kl = 0.0
        p_miss = 0
        p_cnt = 0
        for i in range(min(paired, len(pb), len(pc))):
            da, db = pb[i], pc[i]
            if not da or not db:
                continue
            ka = max(da, key=da.get)
            kb = max(db, key=db.get)
            p_top1 += (ka == kb)
            if ka in db:
                dlp_abs.append(abs(db[ka] - da[ka]))
            floor = min(db.values())
            pa = {t: math.exp(v) for t, v in da.items()}
            za = sum(pa.values())
            qa = {t: math.exp(db.get(t, floor)) for t in da}
            zq = sum(qa.values())
            kl = 0.0
            for t in da:
                pi = pa[t] / za
                qi = qa[t] / zq
                if pi > 0 and qi > 0:
                    kl += pi * math.log(pi / qi)
            p_kl += kl
            p_miss += sum(1 for t in da if t not in db)
            miss_den += len(da)
            p_cnt += 1
        per_prompt.append({
            "index": b["index"],
            "prompt_sha256": b["prompt_sha256"],
            "prompt_tokens_base": b["prompt_tokens"],
            "prompt_tokens_arm": c["prompt_tokens"],
            "generated_tokens": n,
            "first_divergence_index": None if div == n else div,
            "identical_continuation": div == n,
            "paired_positions": p_cnt,
            "top1_agreement": round(p_top1 / p_cnt, 6) if p_cnt else None,
            "mean_truncated_kl_nats": round(p_kl / p_cnt, 8) if p_cnt else None,
            "text_base": b["text"][:200],
            "text_arm": c["text"][:200],
        })
        tot_positions += p_cnt
        tot_top1 += p_top1
        kl_sum += p_kl
        miss_num += p_miss
    return {
        "prompts": len(per_prompt),
        "paired_positions": tot_positions,
        "identical_continuations": sum(1 for r in per_prompt if r["identical_continuation"]),
        "top1_agreement": round(tot_top1 / tot_positions, 6) if tot_positions else None,
        "mean_truncated_kl_nats": round(kl_sum / tot_positions, 8) if tot_positions else None,
        "mean_abs_delta_logprob_on_base_argmax": (
            round(sum(dlp_abs) / len(dlp_abs), 6) if dlp_abs else None),
        "max_abs_delta_logprob_on_base_argmax": round(max(dlp_abs), 6) if dlp_abs else None,
        "support_miss_rate": round(miss_num / miss_den, 6) if miss_den else None,
        "per_prompt": per_prompt,
    }


def self_determinism(rows: list[dict]) -> dict:
    base = next((r for r in rows if r["index"] == 0 and not r.get("repeat")), None)
    rep = next((r for r in rows if r["index"] == 0 and r.get("repeat")), None)
    if not base or not rep:
        return {"available": False}
    same_tokens = base["tokens"] == rep["tokens"]
    dlp = None
    pb, pr = top_dicts(base), top_dicts(rep)
    if pb and pr:
        diffs = []
        for a, b in zip(pb, pr):
            if not a or not b:
                continue
            ka = max(a, key=a.get)
            if ka in b:
                diffs.append(abs(b[ka] - a[ka]))
        dlp = round(max(diffs), 8) if diffs else None
    return {"available": True, "identical_text": base["text"] == rep["text"],
            "identical_tokens": same_tokens,
            "max_abs_delta_logprob": dlp,
            "note": "same server process, same prompt, greedy; run-to-run control"}


# ---------------------------------------------------------------- assembly

ARM_ORDER = ["fp8", "int8_per_token_head", "fp8_per_token_head",
             "int4_per_token_head", "bfloat16"]


def banner_capacity(launch: dict) -> dict:
    f = launch.get("startup_facts") or {}

    def num(x):
        return float(x) if x is not None else None

    conc = f.get("maximum_concurrency") or [None, None]
    tokens = f.get("gpu_kv_cache_tokens")
    return {
        "ready": launch.get("ready"),
        "gpu_memory_utilization": launch.get("gpu_memory_utilization"),
        "max_model_len": launch.get("max_model_len"),
        "mtp_depth": launch.get("mtp_depth"),
        "engine_budget_gib": num((f.get("desired_gpu_memory_utilization") or [None, None])[1]),
        "free_memory_on_device_gib": f.get("free_memory_on_device"),
        "actual_usage_gib": f.get("actual_usage"),
        "available_kv_cache_gib": num(f.get("available_kv_cache_gib")),
        "gpu_kv_cache_tokens": int(tokens.replace(",", "")) if tokens else None,
        "maximum_concurrency_for_window": num(conc[1]),
        "attention_block_size_tokens": f.get("attention_block_size"),
        "kv_padding_layers": f.get("kv_padding_layers"),
        "mamba_page_padding_pct": f.get("mamba_page_padding_pct"),
        "kv_cache_dtype_note": f.get("kv_cache_dtype_note"),
        "enable_prefix_caching": f.get("enable_prefix_caching"),
        "num_spec_tokens": f.get("num_spec_tokens"),
        "startup_wall_sec": launch.get("startup_wall_sec"),
        "vllm_version": f.get("vllm_version"),
    }


def build() -> dict:
    census = load("census") or {}
    law3 = load("law") or {}
    law0 = load("law-mtpoff") or {}
    arms = {d: load(f"arm-{d}") for d in ARM_ORDER}
    arms = {k: v for k, v in arms.items() if v}

    census_out = {}
    for d, rec in census.items():
        r = rec.get("refusal") or {}
        census_out[d] = {
            "starts": rec["ready"],
            "probe_window": rec["max_model_len"],
            "gpu_memory_utilization": rec["gpu_memory_utilization"],
            "vllm_argv": rec["vllm_argv"],
            "podman_argv": rec["podman_argv"],
            "startup_wall_sec": rec["startup_wall_sec"],
            "kv_cache_dtype_note": (rec.get("startup_facts") or {}).get("kv_cache_dtype_note"),
            "attention_block_size_tokens": (rec.get("startup_facts") or {}).get("attention_block_size"),
            "gpu_kv_cache_tokens": (rec.get("startup_facts") or {}).get("gpu_kv_cache_tokens"),
            "probe_generation": rec.get("probe_generation"),
            "process_exit_code": rec.get("process_exit_code"),
            "verbatim_error": r.get("verbatim"),
            "verbatim_error_tail": r.get("verbatim_tail"),
            "server_log_sha256": rec.get("log_sha256"),
        }

    laws = {}
    for d in sorted(set(law3) | set(law0)):
        blk3 = refusal_block_size(law3.get(d, {}))
        blk0 = refusal_block_size(law0.get(d, {}))
        laws[d] = {
            "mtp_3": derive_law(law3.get(d, {}), d, blk3),
            "mtp_off": derive_law(law0.get(d, {}), d, blk0),
            "raw_refusals": {
                "mtp_3": {w: (law3.get(d, {}).get(f"refusal_{w}") or {}).get("refusal")
                          for w in ("131072", "262144")},
                "mtp_off": {w: (law0.get(d, {}).get(f"refusal_{w}") or {}).get("refusal")
                            for w in ("131072", "262144")},
            },
            "launches": {
                "mtp_3": {w: {k: (law3.get(d, {}).get(f"refusal_{w}") or {}).get(k)
                              for k in ("podman_argv", "vllm_argv", "startup_wall_sec",
                                        "log_sha256")}
                          for w in ("131072", "262144")},
                "mtp_off": {w: {k: (law0.get(d, {}).get(f"refusal_{w}") or {}).get(k)
                                for k in ("podman_argv", "vllm_argv", "startup_wall_sec",
                                          "log_sha256")}
                            for w in ("131072", "262144")},
            },
        }

    # class consequence
    classes = {}
    for cls, spec in CLASS_POOLS.items():
        rows = {}
        for mode in ("mtp_3", "mtp_off"):
            pool = spec[mode]["pool_gib"]
            per_dtype = {}
            for d, L in laws.items():
                law = L[mode]
                if not law.get("derived"):
                    continue
                pf = law["published_form"]
                cell = {"published_form": published_window(
                    pf["a_bytes_per_token"], pf["M_gib"], pool)}
                xf = law.get("exact_form")
                if xf:
                    cell["exact_form"] = published_window(
                        xf["a_bytes_per_token"], xf["M_gib"], pool)
                per_dtype[d] = cell
            pub = PUBLISHED_FP8[mode]
            per_dtype["_published_fp8_law_control"] = {
                "published_form": published_window(pub["a"], pub["M"], pool),
                "note": ("the law docs/34 s10.2 published; reproducing the published "
                         "class figure from it is the control on this arithmetic"),
            }
            rows[mode] = {
                "pool_gib": pool, "pool_basis": spec[mode]["basis"],
                "published_fp8_window": spec["published_fp8"][mode],
                "rule": "memory form: a*L + M <= pool/1.15, L a multiple of 4096",
                "headline_form": ("published_form -- the same arithmetic that produced "
                                  "the published figures, so a dtype comparison is like "
                                  "for like. exact_form is carried as a sensitivity row "
                                  "and is NOT proposed as a new published figure: it "
                                  "moves the fp8 baseline itself and was measured on the "
                                  "5090 at utilisation 0.72, not on the 24 GiB proxy."),
                "per_dtype": per_dtype,
            }
        classes[cls] = {"budget_gib": spec["budget_gib"], **rows}

    # fidelity
    base = arms.get("fp8", {}).get("fidelity")
    ref = arms.get("bfloat16", {}).get("fidelity")
    fidelity = {"vs_fp8": {}, "vs_bfloat16": {}, "self_determinism": {}}
    for d, a in arms.items():
        rows = a.get("fidelity")
        if not rows:
            continue
        fidelity["self_determinism"][d] = self_determinism(rows)
        if base and d != "fp8":
            fidelity["vs_fp8"][d] = compare_pair(base, rows)
        if ref and d != "bfloat16":
            fidelity["vs_bfloat16"][d] = compare_pair(ref, rows)
    if base and ref:
        fidelity["vs_bfloat16"]["fp8"] = compare_pair(ref, base)

    measured = {}
    for d, a in arms.items():
        measured[d] = {
            "launch": {k: a["launch"].get(k) for k in
                       ("tag", "kv_cache_dtype", "gpu_memory_utilization",
                        "max_model_len", "mtp_depth", "image",
                        "image_manifest_digest", "model_revision",
                        "podman_argv", "vllm_argv", "startup_wall_sec",
                        "ready", "log_sha256", "gpu_before_load",
                        "gpu_after_startup")},
            "startup_banner": a["launch"].get("startup_facts"),
            "capacity": banner_capacity(a["launch"]),
            "needles_longest_fitting": a.get("needles_long"),
            "needles_common_length": a.get("needles_common"),
            "decode": a.get("decode"),
            "gpu_after_release": a.get("gpu_after_release"),
            "error": a.get("error"),
        }

    # One prefill-throughput row per arm, from the needle runs, which are the
    # only long prompts in the sweep and are byte-identical across arms at the
    # common length.
    prefill = {}
    for d, a in arms.items():
        rows = {}
        for label, key in (("longest_fitting", "needles_long"),
                           ("common_98304", "needles_common")):
            got = a.get(key) or []
            if got:
                rows[label] = {
                    "prompt_tokens": got[0]["prompt_tokens"],
                    "median_wall_sec": sorted(r["wall_sec"] for r in got)[len(got) // 2],
                    "tok_s_including_48_token_decode":
                        sorted(r["prefill_tok_s_including_decode"] for r in got)[len(got) // 2],
                    "runs": len(got),
                }
        prefill[d] = rows


    # The one arm that cannot hold the native window: its refusal at the
    # qualified profile is the evidence for the reduced window it then ran at.
    native_refusals = {}
    probe = load("probe-bfloat16")
    if probe:
        native_refusals["bfloat16"] = {
            "attempted": {k: probe.get(k) for k in
                          ("kv_cache_dtype", "gpu_memory_utilization",
                           "max_model_len", "mtp_depth", "podman_argv",
                           "vllm_argv", "startup_wall_sec", "log_sha256")},
            "refusal": probe.get("refusal"),
            "startup_banner": probe.get("startup_facts"),
            "consequence": ("bfloat16 KV needs 17.64 GiB for one 262,144-token request "
                            "against 9.67 GiB available at the qualified utilisation, and "
                            "the engine's own binary-searched maximum is 139,200 tokens. "
                            "The arm therefore ran at --max-model-len 131072, the largest "
                            "power of two below that, and its measured allocation of "
                            "138,519 tokens at that window is consistent with the 139,200 "
                            "the engine predicted at 262,144."),
        }
    return {
        "census": census_out,
        "attention_backend_probe": load("backend-probe"),
        "kv_law_per_dtype": laws,
        "class_consequence": classes,
        "native_window_refusals": native_refusals,
        "measured_arms": measured,
        "prefill_throughput": prefill,
        "fidelity": fidelity,
    }


if __name__ == "__main__":
    out = build()
    p = Path(__file__).resolve().parent / "analysis.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p} ({p.stat().st_size} bytes)")
