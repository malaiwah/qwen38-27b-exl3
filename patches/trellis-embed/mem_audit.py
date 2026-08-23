#!/usr/bin/env python3
"""Enumerate every live tensor in a production-identical model load.

Purpose: reconcile the measured ~20.5 GiB resident model against the ~18.4 GiB
expected after the two online quantizations (embedding -> packed int6, 128-aligned
vision projections -> trellis K6). Finds tensors that are resident but should not be.
"""
from __future__ import annotations
import argparse, json, os, sys


def _rpc_enumerate(self):
    import torch
    from collections import defaultdict

    model = self.model_runner.model
    by_dtype = defaultdict(lambda: {"n": 0, "bytes": 0})
    by_group = defaultdict(lambda: {"n": 0, "bytes": 0})
    big, embed_detail, stray = [], [], []
    vision_by_dtype = defaultdict(lambda: {"n": 0, "bytes": 0})
    seen_storage = set()

    def group_of(name: str) -> str:
        if "visual" in name:
            return "vision"
        if "embed_tokens" in name:
            return "embed_tokens"
        if "lm_head" in name:
            return "lm_head"
        if name.startswith("mtp") or ".mtp." in name:
            return "mtp"
        return "language_model_body"

    def account(name, t):
        if t is None or not hasattr(t, "numel") or not hasattr(t, "data_ptr"):
            return
        try:
            ptr = t.data_ptr()
        except Exception:
            return
        nb = 0 if (ptr == 0 or t.numel() == 0) else t.numel() * t.element_size()
        key = (ptr, t.numel(), str(t.dtype))
        dup = key in seen_storage
        seen_storage.add(key)
        d = str(t.dtype).replace("torch.", "")
        g = group_of(name)
        by_dtype[d]["n"] += 1
        by_group[g]["n"] += 1
        if not dup:
            by_dtype[d]["bytes"] += nb
            by_group[g]["bytes"] += nb
        if "visual" in name:
            vision_by_dtype[d]["n"] += 1
            if not dup:
                vision_by_dtype[d]["bytes"] += nb
        if nb >= 64 * 1024 ** 2:
            big.append({"name": name, "shape": list(t.shape), "dtype": d,
                        "MiB": round(nb / 1024 ** 2, 1), "alias_of_earlier": dup})
        if "embed_tokens" in name:
            embed_detail.append({"name": name, "shape": list(t.shape), "dtype": d,
                                 "MiB": round(nb / 1024 ** 2, 2), "alias_of_earlier": dup})

    for n, p in model.named_parameters(recurse=True):
        account(n, getattr(p, "data", p))
    for n, b in model.named_buffers(recurse=True):
        account(n, b)

    SKIP = {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_state_dict_hooks"}
    for mod_name, mod in model.named_modules():
        for attr, val in list(vars(mod).items()):
            if attr in SKIP:
                continue
            fqbase = f"{mod_name}.{attr}" if mod_name else attr
            if hasattr(val, "numel") and hasattr(val, "data_ptr"):
                try:
                    nb = val.numel() * val.element_size() if val.data_ptr() else 0
                except Exception:
                    nb = 0
                if nb >= 16 * 1024 ** 2:
                    dup = (val.data_ptr(), val.numel(), str(val.dtype)) in seen_storage
                    stray.append({"name": fqbase, "shape": list(val.shape),
                                  "dtype": str(val.dtype).replace("torch.", ""),
                                  "MiB": round(nb / 1024 ** 2, 1), "alias_of_earlier": dup})
                    account(fqbase, val)
            elif isinstance(val, dict):
                for k, v in list(val.items()):
                    if hasattr(v, "numel") and hasattr(v, "data_ptr"):
                        try:
                            nb = v.numel() * v.element_size() if v.data_ptr() else 0
                        except Exception:
                            nb = 0
                        if nb >= 16 * 1024 ** 2:
                            fq = f"{fqbase}[{k}]"
                            dup = (v.data_ptr(), v.numel(), str(v.dtype)) in seen_storage
                            stray.append({"name": fq, "shape": list(v.shape),
                                          "dtype": str(v.dtype).replace("torch.", ""),
                                          "MiB": round(nb / 1024 ** 2, 1), "alias_of_earlier": dup})
                            account(fq, v)

    tot = sum(v["bytes"] for v in by_dtype.values())
    return {
        "model_tensor_total_GiB": round(tot / 1024 ** 3, 3),
        "torch_allocated_GiB": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
        "torch_reserved_GiB": round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
        "by_dtype": {k: {"n": v["n"], "GiB": round(v["bytes"] / 1024 ** 3, 3)}
                     for k, v in sorted(by_dtype.items(), key=lambda x: -x[1]["bytes"])},
        "by_group": {k: {"n": v["n"], "GiB": round(v["bytes"] / 1024 ** 3, 3)}
                     for k, v in sorted(by_group.items(), key=lambda x: -x[1]["bytes"])},
        "vision_by_dtype": {k: {"n": v["n"], "GiB": round(v["bytes"] / 1024 ** 3, 3)}
                            for k, v in sorted(vision_by_dtype.items(), key=lambda x: -x[1]["bytes"])},
        "embed_tensors": embed_detail,
        "tensors_ge_64MiB": sorted(big, key=lambda x: -x["MiB"])[:30],
        "stray_module_attrs_ge_16MiB": sorted(stray, key=lambda x: -x["MiB"])[:30],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    ap.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    a = ap.parse_args()

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    llm = LLM(
        model=a.model, trust_remote_code=True, tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_memory_utilization,
        dtype="bfloat16", kv_cache_dtype=a.kv_cache_dtype,
        load_format="safetensors", max_model_len=a.max_model_len,
        max_num_batched_tokens=3072, max_num_seqs=1,
        enable_prefix_caching=False, disable_log_stats=True,
        quantization="exl3", enforce_eager=True,
    )
    rep = llm.collective_rpc(_rpc_enumerate)[0]
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)

    print("=== resident model tensors ===")
    print(f"  sum of unique tensor storages : {rep['model_tensor_total_GiB']:.3f} GiB")
    print(f"  torch.cuda.memory_allocated   : {rep['torch_allocated_GiB']:.3f} GiB")
    print(f"  torch.cuda.memory_reserved    : {rep['torch_reserved_GiB']:.3f} GiB")
    print("\n=== by dtype ===")
    for k, v in rep["by_dtype"].items():
        print(f"  {k:12s} n={v['n']:5d}  {v['GiB']:8.3f} GiB")
    print("\n=== by subsystem ===")
    for k, v in rep["by_group"].items():
        print(f"  {k:22s} n={v['n']:5d}  {v['GiB']:8.3f} GiB")
    print("\n=== vision tower by dtype ===")
    for k, v in rep["vision_by_dtype"].items():
        print(f"  {k:12s} n={v['n']:5d}  {v['GiB']:8.3f} GiB")
    print("\n=== embed_tokens tensors ===")
    for e in rep["embed_tensors"]:
        print(f"  {e['name']:60s} {str(e['shape']):18s} {e['dtype']:9s} "
              f"{e['MiB']:9.2f} MiB alias={e['alias_of_earlier']}")
    print("\n=== tensors >= 64 MiB ===")
    for b in rep["tensors_ge_64MiB"]:
        print(f"  {b['name'][:66]:66s} {b['dtype']:9s} {b['MiB']:9.1f} MiB alias={b['alias_of_earlier']}")
    print("\n=== stray module attrs >= 16 MiB (not params/buffers) ===")
    if not rep["stray_module_attrs_ge_16MiB"]:
        print("  (none)")
    for s in rep["stray_module_attrs_ge_16MiB"]:
        print(f"  {s['name'][:66]:66s} {s['dtype']:9s} {s['MiB']:9.1f} MiB alias={s['alias_of_earlier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
