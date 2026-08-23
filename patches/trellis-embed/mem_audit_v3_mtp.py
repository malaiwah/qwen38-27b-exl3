"""v3: does the MTP drafter SHARE the main embedding table, or own a duplicate?

GitHubPatchAudit claims qwen3_5_mtp_patch.py:83 unconditionally builds a second
full VocabParallelEmbedding(248320, 5120), and that remap_weight_names maps the
checkpoint's single real embedding onto it -- so the draft loads a duplicate that
then gets separately encoded to packed int6 (0.888 GiB wasted). Two "embed online
K6 conversion complete" log lines are the symptom.

v1/v2 could not see it: they walked only model_runner.model, while the drafter is
a separate object (model_runner.drafter / .model).
"""
from __future__ import annotations
import argparse, json, os


def _rpc(self):
    import torch

    out = {"probe": "mtp_embed_sharing"}

    runner = self.model_runner
    main = runner.model

    # Locate the drafter under any of the plausible attribute names.
    drafter = None
    drafter_attr = None
    for cand in ("drafter", "speculator", "proposer", "spec_decode_proposer",
                 "draft_model_runner", "_drafter"):
        d = getattr(runner, cand, None)
        if d is not None:
            drafter, drafter_attr = d, cand
            break
    out["drafter_attr"] = drafter_attr
    out["drafter_type"] = type(drafter).__name__ if drafter is not None else None

    def embed_modules(root, root_label):
        found = []
        if root is None:
            return found
        # root may be a runner-ish wrapper holding .model
        candidates = [(root_label, root)]
        inner = getattr(root, "model", None)
        if inner is not None and inner is not root:
            candidates.append((f"{root_label}.model", inner))
        for lbl, obj in candidates:
            if not hasattr(obj, "named_modules"):
                continue
            for n, m in obj.named_modules():
                if type(m).__name__ != "VocabParallelEmbedding":
                    continue
                w = getattr(m, "weight", None)
                q = getattr(m, "q_weight", None)
                s = getattr(m, "embed_scale", None)
                def info(t):
                    if t is None or not hasattr(t, "data_ptr"):
                        return None
                    try:
                        return {"ptr": t.data_ptr(), "shape": list(t.shape),
                                "dtype": str(t.dtype).replace("torch.", ""),
                                "MiB": round((t.numel() * t.element_size()) / 1024**2, 2)}
                    except Exception:
                        return None
                found.append({
                    "root": lbl,
                    "name": n or "<root>",
                    "module_id": id(m),
                    "quant_method": type(getattr(m, "quant_method", None)).__name__,
                    "weight": info(getattr(w, "data", w)),
                    "q_weight": info(q),
                    "embed_scale": info(s),
                })
            break  # only need the first traversable candidate per root
        return found

    main_embeds = embed_modules(main, "main")
    draft_embeds = embed_modules(drafter, "draft")

    out["main_embeds"] = main_embeds
    out["draft_embeds"] = draft_embeds

    # Distinct storages across BOTH trees
    ptrs = {}
    total_embed_bytes = 0
    for e in main_embeds + draft_embeds:
        for field in ("weight", "q_weight", "embed_scale"):
            v = e.get(field)
            if not v or not v["ptr"]:
                continue
            key = (v["ptr"], v["MiB"], v["dtype"])
            if key not in ptrs:
                ptrs[key] = f"{e['root']}/{e['name']}.{field}"
                total_embed_bytes += v["MiB"]
    out["distinct_embed_storages"] = [
        {"owner": v, "MiB": k[1], "dtype": k[2]} for k, v in ptrs.items()
    ]
    out["total_distinct_embed_MiB"] = round(total_embed_bytes, 2)
    out["n_vocab_parallel_embedding_modules"] = len(main_embeds) + len(draft_embeds)
    out["module_ids_shared"] = (
        len({e["module_id"] for e in main_embeds + draft_embeds}) <
        len(main_embeds + draft_embeds)
    )
    out["torch_allocated_GiB"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    ap.add_argument("--speculative-config", default=None)
    a = ap.parse_args()

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    kw = dict(model=a.model, trust_remote_code=True, tensor_parallel_size=1,
              gpu_memory_utilization=a.gpu_memory_utilization,
              dtype="bfloat16", kv_cache_dtype=a.kv_cache_dtype,
              load_format="safetensors", max_model_len=a.max_model_len,
              max_num_batched_tokens=3072, max_num_seqs=1,
              enable_prefix_caching=False, disable_log_stats=True,
              quantization="exl3", enforce_eager=True)
    if a.speculative_config:
        kw["speculative_config"] = json.loads(a.speculative_config)

    llm = LLM(**kw)
    rep = llm.collective_rpc(_rpc)[0]
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)

    print("\n============ MTP EMBEDDING SHARING PROBE ============")
    print(f"  drafter attr / type : {rep['drafter_attr']} / {rep['drafter_type']}")
    print(f"  VocabParallelEmbedding modules found : {rep['n_vocab_parallel_embedding_modules']}")
    print(f"  module objects shared between trees  : {rep['module_ids_shared']}")
    print(f"  distinct embedding storages          : {len(rep['distinct_embed_storages'])}")
    print(f"  total distinct embedding bytes       : {rep['total_distinct_embed_MiB']:.2f} MiB")
    print("\n  --- main tree ---")
    for e in rep["main_embeds"]:
        print(f"    {e['name'] or '<root>'}  qm={e['quant_method']}")
        for f_ in ("weight", "q_weight", "embed_scale"):
            v = e.get(f_)
            if v:
                print(f"        {f_:12s} {str(v['shape']):18s} {v['dtype']:9s} {v['MiB']:9.2f} MiB ptr={v['ptr']}")
    print("\n  --- draft tree ---")
    if not rep["draft_embeds"]:
        print("    (no VocabParallelEmbedding found under drafter)")
    for e in rep["draft_embeds"]:
        print(f"    {e['name'] or '<root>'}  qm={e['quant_method']}")
        for f_ in ("weight", "q_weight", "embed_scale"):
            v = e.get(f_)
            if v:
                print(f"        {f_:12s} {str(v['shape']):18s} {v['dtype']:9s} {v['MiB']:9.2f} MiB ptr={v['ptr']}")
    print("\n  --- distinct storages ---")
    for s in rep["distinct_embed_storages"]:
        print(f"    {s['owner']:52s} {s['dtype']:9s} {s['MiB']:9.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
