"""v2: traverse exl3_tensors dicts hanging off Parameter objects, plus mp_weights
and any expanded_weight materialization. v1 found only 10.3 GiB of 29.3 GiB
allocated because EXL3 payloads are not reachable via named_parameters/buffers.
"""
from __future__ import annotations
import argparse, json, os


def _rpc_enumerate_v2(self):
    import torch
    from collections import defaultdict

    model = self.model_runner.model
    seen = set()
    by_bucket = defaultdict(lambda: {"n": 0, "bytes": 0})
    big = []

    def acct(bucket, name, t):
        if t is None or not hasattr(t, "data_ptr") or not hasattr(t, "numel"):
            return 0
        try:
            ptr = t.data_ptr()
            nb = 0 if (ptr == 0 or t.numel() == 0) else t.numel() * t.element_size()
        except Exception:
            return 0
        key = (ptr, t.numel(), str(t.dtype))
        if key in seen:
            return 0
        seen.add(key)
        by_bucket[bucket]["n"] += 1
        by_bucket[bucket]["bytes"] += nb
        if nb >= 32 * 1024 ** 2:
            big.append({"bucket": bucket, "name": name,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype).replace("torch.", ""),
                        "MiB": round(nb / 1024 ** 2, 1)})
        return nb

    def walk_obj(bucket, name, obj, depth=0):
        """Recursively account tensors reachable from obj (bounded depth)."""
        if depth > 3 or obj is None:
            return
        if hasattr(obj, "data_ptr") and hasattr(obj, "numel"):
            acct(bucket, name, obj)
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                walk_obj(bucket, f"{name}[{k}]", v, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk_obj(bucket, f"{name}[{i}]", v, depth + 1)
            return

    # --- 1. plain params / buffers, splitting kv_cache out ---
    for n, p in model.named_parameters(recurse=True):
        acct("param_plain", n, getattr(p, "data", p))
    for n, b in model.named_buffers(recurse=True):
        acct("buffer_plain", n, b)

    # --- 2. EXL3 payloads: exl3_tensors on Parameter objects ---
    for mod_name, mod in model.named_modules():
        for pname, p in list(mod._parameters.items()) if hasattr(mod, "_parameters") else []:
            if p is None:
                continue
            ex = getattr(p, "exl3_tensors", None)
            if isinstance(ex, dict) and ex:
                walk_obj("exl3_tensors", f"{mod_name}.{pname}.exl3_tensors", ex)

    # --- 3. mp_weights (FP4/FP6 converted) + expanded_weight materialization ---
    for mod_name, mod in model.named_modules():
        mp = getattr(mod, "mp_weights", None)
        if mp:
            walk_obj("mp_weights", f"{mod_name}.mp_weights", mp)
            # expanded_weight is a cached property on the FP6 weight object
            try:
                for sid, w in (mp.items() if isinstance(mp, dict) else []):
                    for attr in ("_expanded_weight", "expanded_weight", "packed",
                                 "scale_storage", "global_scale", "weight"):
                        if attr == "expanded_weight":
                            # only read if already materialized, do NOT trigger it
                            v = getattr(w, "_expanded_weight", None)
                        else:
                            v = getattr(w, attr, None)
                        if v is not None:
                            walk_obj("mp_weights_inner",
                                     f"{mod_name}.mp_weights[{sid}].{attr}", v)
            except Exception:
                pass
        for attr in ("fp4_draft_weights", "exl3_online_trellis_weight",
                     "exl3_online_suh", "exl3_online_svh",
                     "q_weight", "embed_scale"):
            v = getattr(mod, attr, None)
            if v is not None:
                walk_obj("named_special", f"{mod_name}.{attr}", v)

    # --- 4. kv_cache attrs (separate bucket so they don't pollute weights) ---
    for mod_name, mod in model.named_modules():
        v = getattr(mod, "kv_cache", None)
        if v is not None:
            walk_obj("kv_cache", f"{mod_name}.kv_cache", v)

    # --- 5. module __dict__ residue not already counted ---
    SKIP = {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
            "_state_dict_hooks", "kv_cache", "mp_weights"}
    for mod_name, mod in model.named_modules():
        for attr, val in list(vars(mod).items()):
            if attr in SKIP:
                continue
            walk_obj("module_dict_residue", f"{mod_name}.{attr}", val)

    # --- 6. module-level caches in the exl3 patch ---
    patch_caches = {}
    try:
        import vllm.model_executor.layers.quantization.exl3 as E
        for cand in ("_b12x_trellis_linear",):
            fn = getattr(E, cand, None)
            bc = getattr(fn, "_buf_cache", None) if fn else None
            if isinstance(bc, dict):
                tot = 0
                for k, v in list(bc.items()):
                    tot += acct("b12x_buf_cache", f"{cand}._buf_cache[{k}]", v) or 0
                patch_caches[f"{cand}._buf_cache"] = {"entries": len(bc),
                                                      "GiB": round(tot / 1024 ** 3, 4)}
        for setname in ("_EXL3_ONLINE_WARMED_SIGNATURES",
                        "_EXL3_GEMM_PRIMED_SIGNATURES",
                        "_B12X_WARMED"):
            s = getattr(E, setname, None)
            if s is not None:
                patch_caches[setname] = {"len": len(s)}
        for dname in ("_FP6_CONVERSION_MODULE", "_FP4_CONVERSION_MODULE"):
            patch_caches[dname] = getattr(E, dname, None) is not None
    except Exception as exc:
        patch_caches["error"] = repr(exc)

    total = sum(v["bytes"] for v in by_bucket.values())
    alloc = torch.cuda.memory_allocated()
    return {
        "enumerated_GiB": round(total / 1024 ** 3, 3),
        "torch_allocated_GiB": round(alloc / 1024 ** 3, 3),
        "torch_reserved_GiB": round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
        "unaccounted_GiB": round((alloc - total) / 1024 ** 3, 3),
        "by_bucket": {k: {"n": v["n"], "GiB": round(v["bytes"] / 1024 ** 3, 3)}
                      for k, v in sorted(by_bucket.items(), key=lambda x: -x[1]["bytes"])},
        "patch_caches": patch_caches,
        "tensors_ge_32MiB_top40": sorted(big, key=lambda x: -x["MiB"])[:40],
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

    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=1,
              gpu_memory_utilization=a.gpu_memory_utilization,
              dtype="bfloat16", kv_cache_dtype=a.kv_cache_dtype,
              load_format="safetensors", max_model_len=a.max_model_len,
              max_num_batched_tokens=3072, max_num_seqs=1,
              enable_prefix_caching=False, disable_log_stats=True,
              quantization="exl3", enforce_eager=True)
    rep = llm.collective_rpc(_rpc_enumerate_v2)[0]
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)

    print("\n================ MEMORY RECONCILIATION ================")
    print(f"  enumerated           : {rep['enumerated_GiB']:8.3f} GiB")
    print(f"  torch allocated      : {rep['torch_allocated_GiB']:8.3f} GiB")
    print(f"  torch reserved       : {rep['torch_reserved_GiB']:8.3f} GiB")
    print(f"  UNACCOUNTED          : {rep['unaccounted_GiB']:8.3f} GiB")
    print("\n---- by bucket ----")
    for k, v in rep["by_bucket"].items():
        print(f"  {k:24s} n={v['n']:5d}  {v['GiB']:8.3f} GiB")
    print("\n---- patch-level caches ----")
    print(json.dumps(rep["patch_caches"], indent=2))
    print("\n---- largest tensors ----")
    for b in rep["tensors_ge_32MiB_top40"]:
        print(f"  [{b['bucket']:18s}] {b['name'][:56]:56s} {b['dtype']:9s} {b['MiB']:9.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
