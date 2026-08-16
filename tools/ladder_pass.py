#!/usr/bin/env python3
"""Measure exllamav3's own per-module proxy-error ladder in ONE conversion pass.

The converter allocates bits by a static priority order and prints
`proxy_err = tr(E^T H E) / tr(W^T H W)` per module *at the width it assigned*, so a
normal conversion yields one point per module instead of a curve. That is not enough
for error-driven allocation, and nothing on disk holds the curve.

This pass wraps `quantize_linears_single` / `quantize_linears_parallel`. For every
budgeted body Linear it

  1. copies the unquantized weight and computes the module's per-token calibration
     output energy  tr(W^T H W) / count  from the *raw* accumulated Hessian, before
     `finalize_capture_H` mutates H in place (H /= count, diagonal damping, sign
     flips, Hadamard rotations);
  2. runs the converter's unmodified quantization at the recipe width, so state
     propagation and the emitted checkpoint are exactly the hydrated build's, and
     records that width's proxy error by wrapping `print_quantized_linear`;
  3. re-quantizes the saved weight at every other candidate width against the same
     finalized Hessian, through the same batched entry point the converter uses
     (`quantize_exl3_batch`), and records the proxy error.

The Hessian is captured once and reused for the whole ladder, which is the only
reason this costs ~5x a conversion instead of 5 conversions. Nothing about the
emitted checkpoint changes: the extra calls take copies of the weight and their
packed tensors are discarded.

Output: LADDER_OUT json, rewritten after every module, keyed by module with
in/out features, numel, recipe width, output energy, and proxy_err per width.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/work/exllamav3")

import torch
from exllamav3.conversion import convert_model as cm
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

LADDER_OUT = os.environ.get("LADDER_OUT", "/work/kld6/ladder.json")
# Candidate widths. Large tensors get the K3 rung, where a bit is worth 700+ MB
# across a role and the byte lever is real; small ones get the K8 rung instead,
# where a bit costs almost nothing. Both sets are five rungs, so the pass costs
# the same either way.
BIG_NUMEL = 52_000_000
CAND_BIG = (3, 4, 5, 6, 7)
CAND_SMALL = (4, 5, 6, 7, 8)
# Not ladder targets: the head and the MTP draft are pinned by the recipe (-hb/-mb)
# and the vision tower is unquantized at -vb 16. Their recipe-width points are still
# recorded through the print hook.
SKIP_SUBSTR = ("lm_head", "mtp.", "visual")

records: dict[str, dict] = {}
_t_start = time.time()


def candidates(numel: int) -> tuple[int, ...]:
    return CAND_BIG if numel >= BIG_NUMEL else CAND_SMALL


def dump() -> None:
    tmp = LADDER_OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "schema": "qwen38-proxy-error-ladder/1",
                "metric": "proxy_err = tr(E^T H E) / tr(W^T H W), exllamav3's own per-module "
                          "Hessian-weighted relative quantization error",
                "out_energy": "tr(W^T H W) / count on the raw accumulated Hessian: the module's "
                              "mean per-calibration-row output energy, so out_energy * proxy_err "
                              "is the module's absolute output error energy",
                "candidate_widths": {"big": list(CAND_BIG), "small": list(CAND_SMALL),
                                     "big_numel_threshold": BIG_NUMEL},
                "propagation_recipe": "hydrated (attention K6, mlp gate/up K5, down K6, head K6/mcg, "
                                      "MTP as -mb 4 plus the fixed/override regexes, BF16 embed+vision)",
                "elapsed_sec": round(time.time() - _t_start, 1),
                "modules": records,
            },
            f,
            indent=1,
        )
    os.replace(tmp, LADDER_OUT)


def out_energy(linear, H_data) -> float | None:
    """Per-row calibration output energy, from the raw H, before any finalize call."""
    if H_data is None or H_data.get("finalized") or H_data["H"].is_meta:
        return None
    count = int(H_data["count"])
    if count == 0:
        return None
    dev = torch.device(H_data["device"])
    w = linear.inner.get_weight_tensor()
    W = w.to(dev, torch.float32)
    H = H_data["H"].to(dev, torch.float32)
    energy = float((W * (H @ W)).sum().item()) / count
    del W, H
    return energy


_orig_print = cm.print_quantized_linear


def print_hook(config, linear, quant_args, proxy_err, time_str=""):
    _orig_print(config, linear, quant_args, proxy_err, time_str)
    rec = records.setdefault(linear.key, {})
    rec.setdefault("ladder", {})[str(int(quant_args["K"]))] = proxy_err
    rec["recipe_bits"] = int(quant_args["K"])
    rec["recipe_proxy_err"] = proxy_err
    rec["q_fallback"] = bool(quant_args.get("q_fallback"))
    rec.setdefault("numel", linear.weights_numel())
    rec.setdefault("in_features", linear.in_features)
    rec.setdefault("out_features", linear.out_features)


cm.print_quantized_linear = print_hook


def wrap(orig):
    def fn(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state):
        todo = [
            l for l in linears
            if strategy.get(l.key, 16) != 16
            and capture_H is not None
            and l.qmap in capture_H
            and not any(s in l.key for s in SKIP_SUBSTR)
        ]

        # 1. weights and output energies, before the first finalize_capture_H mutates H
        saved = {}
        for l in todo:
            saved[l.key] = l.inner.get_weight_tensor().clone()
            rec = records.setdefault(l.key, {})
            rec["numel"] = l.weights_numel()
            rec["in_features"] = l.in_features
            rec["out_features"] = l.out_features
            rec["qmap"] = l.qmap
            rec["cal_rows"] = int(capture_H[l.qmap]["count"])
            if rec.get("out_energy") is None:
                rec["out_energy"] = out_energy(l, capture_H[l.qmap])

        # 2. the converter's own unmodified pass at the recipe widths
        orig(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state)

        # 3. the remaining rungs, same finalized Hessians, same batched entry point
        all_k = sorted({k for l in todo for k in candidates(records[l.key]["numel"])})
        for K in all_k:
            targets = [
                l for l in todo
                if K in candidates(records[l.key]["numel"]) and K != strategy[l.key]
            ]
            if not targets:
                continue
            ustrat = {l.key: K for l in targets}
            for group in cm.group_quant_linears(targets, ustrat, capture_H):
                qal = [
                    cm.make_quant_args(
                        args, idx, K,
                        *cm._tile_split_devices(l.weights_numel(), devices, device_ratios)
                    ) for l in group
                ]
                weights = [saved[l.key].float() for l in group]
                t0 = time.time()
                res = quantize_exl3_batch(weights, [capture_H[l.qmap] for l in group], qal, None)
                dt = (time.time() - t0) / len(group)
                for l, qa, (proxy_err, out_tensors) in zip(group, qal, res):
                    rec = records[l.key]
                    rec.setdefault("ladder", {})[str(K)] = proxy_err
                    rec.setdefault("ladder_sec", {})[str(K)] = round(dt, 2)
                    rec.setdefault("g_scale", {})[str(K)] = qa.get("g_scale")
                    if qa.get("q_fallback"):
                        rec["q_fallback"] = True
                    print(f" ~~ Ladder: {l.key:70}  K={K}  proxy_err: {proxy_err:10.8f}"
                          f"  g_sc: {qa.get('g_scale', float('nan')):.6f}  [{dt:5.2f} s]", flush=True)
                    del out_tensors
                del res, weights
        saved.clear()
        torch.cuda.empty_cache()
        dump()

    return fn


cm.quantize_linears_single = wrap(cm.quantize_linears_single)
cm.quantize_linears_parallel = wrap(cm.quantize_linears_parallel)


if __name__ == "__main__":
    _args = cm.parser.parse_args()
    _in_args, _job_state, _ok, _err = cm.prepare(_args)
    if not _ok:
        print(f" !! Error: {_err}")
        raise SystemExit(1)
    try:
        cm.main(_in_args, _job_state)
    finally:
        dump()
        print(f" ~~ Ladder written: {LADDER_OUT} ({len(records)} modules)", flush=True)
