#!/usr/bin/env python3
"""Why is dense EXL3 prefill slow, and how much is recoverable?

Measured: our checkpoint serves 2.4k prefill tok/s where NVFP4 serves 14.5k. The
vLLM EXL3 backend routes every non-K6 shard through `exl3_gemm` at any M, but
exllamav3's own `LinearEXL3.reconstruct_hgemm` switches to
`reconstruct_had_slice` + `hgemm` above 1024 rows precisely because the trellis
kernel is decode-shaped. This times both paths on the three real geometries of the
published checkpoint, at decode and prefill row counts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = Path("/work/Qwen3.8-27B-K5K6")
SHAPES = [("mlp.gate_proj", "model.language_model.layers.0.mlp.gate_proj"),
          ("mlp.down_proj", "model.language_model.layers.0.mlp.down_proj"),
          ("lm_head", "lm_head")]
ROWS = [1, 8, 32, 256, 1024, 2048]


def load(prefix: str):
    idx = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for suf in ("trellis", "suh", "svh", "mcg", "mul1"):
        key = f"{prefix}.{suf}"
        if key in idx:
            with safe_open(str(MODEL / idx[key]), framework="pt", device="cpu") as f:
                out[suf] = f.get_tensor(key).cuda()
    return out


def bench(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main() -> int:
    import exllamav3  # noqa: F401
    from exllamav3.ext import exllamav3_ext as ext

    print(f"{'geometry':16s} {'K':>6s} {'N':>7s} {'bits':>4s} {'m':>5s} "
          f"{'exl3_gemm ms':>13s} {'recon+hgemm ms':>15s} {'speedup':>8s}")
    results = []
    for label, prefix in SHAPES:
        t = load(prefix)
        trellis, suh, svh = t["trellis"], t["suh"], t["svh"]
        mcg = int(t["mcg"].item()) if "mcg" in t else 0
        mul1 = int(t["mul1"].item()) if "mul1" in t else 0
        K = trellis.shape[0] * 16
        N = trellis.shape[1] * 16
        bits = trellis.shape[2] // 16
        # reconstruct once per weight: this is the cost a real implementation would
        # amortise across the whole prefill chunk, so time it separately
        w = torch.empty((K, N), dtype=torch.float16, device="cuda")
        rec = bench(lambda: ext.reconstruct_had_slice(w, trellis, suh, svh, bits,
                                                      bool(mcg), bool(mul1), 0), iters=4)
        for m in ROWS:
            x = torch.zeros((m, K), dtype=torch.float16, device="cuda")
            y = torch.empty((m, N), dtype=torch.float16, device="cuda")
            xh = torch.empty_like(x)
            gemm = bench(lambda: ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1,
                                               mcg, mul1, 0))
            hg = bench(lambda: ext.hgemm(x, w, y))
            total = hg + (rec if m >= 1024 else 0.0)  # reconstruct amortised per chunk
            print(f"{label:16s} {K:6d} {N:7d} {bits:4d} {m:5d} "
                  f"{gemm*1e3:13.3f} {total*1e3:15.3f} {gemm/total:7.2f}x")
            results.append({"geometry": label, "K": K, "N": N, "bits": bits, "m": m,
                            "exl3_gemm_ms": gemm * 1e3, "recon_hgemm_ms": total * 1e3,
                            "reconstruct_ms": rec * 1e3, "speedup": gemm / total})
        del trellis, suh, svh, w
        torch.cuda.empty_cache()
    Path("/work/kld3/prefill-micro.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
