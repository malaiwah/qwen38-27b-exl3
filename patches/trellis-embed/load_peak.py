#!/usr/bin/env python3
"""Report load-time peak GPU allocation for a production-identical model load.

KV is capped small on purpose: vLLM otherwise sizes the KV pool to fill the
utilization budget, and that single allocation dominates max_memory_allocated,
masking the load-time transients this probe exists to measure.
"""
from __future__ import annotations
import argparse, json, os


def _rpc(self):
    import torch
    return {
        "allocated_GiB": torch.cuda.memory_allocated() / 1024**3,
        "max_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_GiB": torch.cuda.memory_reserved() / 1024**3,
        "max_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--kv-cache-mib", type=int, default=512)
    ap.add_argument("--no-spec", action="store_true",
                    help="load without the MTP drafter (control)")
    a = ap.parse_args()

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    kw = dict(
        model=a.model, trust_remote_code=True, tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_memory_utilization, dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3", load_format="safetensors",
        max_model_len=a.max_model_len, max_num_batched_tokens=3072,
        max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
        quantization="exl3", enforce_eager=True,
        kv_cache_memory_bytes=a.kv_cache_mib * 1024 * 1024,
    )
    if not a.no_spec:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 6}

    llm = LLM(**kw)
    r = llm.collective_rpc(_rpc)[0]
    r["label"] = a.label
    r["spec"] = not a.no_spec
    json.dump(r, open(a.out, "w"), indent=2)

    print(f"\n==== LOAD PEAK [{a.label}] spec={not a.no_spec} ====")
    for k in ("allocated_GiB", "max_allocated_GiB",
              "reserved_GiB", "max_reserved_GiB"):
        print(f"  {k:20s} {r[k]:8.3f} GiB")
    print(f"  {'transient_headroom':20s} "
          f"{r['max_allocated_GiB'] - r['allocated_GiB']:8.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
