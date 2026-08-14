#!/usr/bin/env python3
"""Unfused reconstruct+hgemm vs exl3_gemm, using the extension vLLM actually loads.

The first microbenchmark used exllamav3 1.4.2's fused `reconstruct_had_slice`. The
image's baked extension (`VLLM_EXL3_EXT_PATH`) does **not** export it, so a patch that
must run in the published container has to use the unfused sequence:

    had_r_128(x, xh, suh, None, 1.0)      # input Hadamard + input scales
    reconstruct[_slice](w, trellis, K, mcg, mul1)
    hgemm(xh, w, y)
    had_r_128(y, y, None, svh, 1.0)       # output Hadamard + output scales

This measures that sequence and checks it against `exl3_gemm` numerically, so the
threshold in the patch is set from data rather than from the fused figures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext  # noqa: E402  the extension vLLM loads

MODEL = Path("/work/Qwen3.8-27B-K5K6")
SHAPES = [("mlp.gate_proj", "model.language_model.layers.0.mlp.gate_proj"),
          ("mlp.down_proj", "model.language_model.layers.0.mlp.down_proj"),
          ("lm_head", "lm_head")]
ROWS = [1, 8, 32, 64, 128, 256, 512, 1024, 2048]
SLICE_N = 32768


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
    results = []
    print(f"{'geometry':15s} {'K':>6s} {'N':>7s} {'m':>5s} {'exl3_gemm':>10s} "
          f"{'recon+hgemm':>12s} {'speedup':>8s} {'max|dy|':>9s}")
    for label, prefix in SHAPES:
        t = load(prefix)
        trellis, suh, svh = t["trellis"], t["suh"], t["svh"]
        mcg, mul1 = "mcg" in t, "mul1" in t
        K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
        bits = trellis.shape[2] // 16
        w_full = torch.empty((K, min(N, SLICE_N)), dtype=torch.float16, device="cuda")

        def recon_hgemm(x, y, xh):
            ext.had_r_128(x, xh, suh, None, 1.0)
            for n0 in range(0, N, SLICE_N):
                n1 = min(n0 + SLICE_N, N)
                w = w_full[:, : n1 - n0]
                if N <= SLICE_N:
                    ext.reconstruct(w, trellis, bits, mcg, mul1)
                else:
                    ext.reconstruct_slice(w, trellis, bits, mcg, mul1, n0)
                ext.hgemm(xh, w, y[:, n0:n1])
            ext.had_r_128(y, y, None, svh, 1.0)

        for m in ROWS:
            x = torch.randn((m, K), dtype=torch.float16, device="cuda") * 0.05
            y1 = torch.empty((m, N), dtype=torch.float16, device="cuda")
            y2 = torch.empty((m, N), dtype=torch.float16, device="cuda")
            xh = torch.empty_like(x)
            g = bench(lambda: ext.exl3_gemm(x, trellis, y1, suh, xh, svh, -1,
                                            mcg, mul1, 0))
            r = bench(lambda: recon_hgemm(x, y2, xh))
            dy = (y1.float() - y2.float()).abs().max().item()
            scale = y1.float().abs().max().item() or 1.0
            print(f"{label:15s} {K:6d} {N:7d} {m:5d} {g*1e3:9.3f}ms {r*1e3:11.3f}ms "
                  f"{g/r:7.2f}x {dy/scale:9.2e}")
            results.append({"geometry": label, "K": K, "N": N, "bits": bits, "m": m,
                            "exl3_gemm_ms": g * 1e3, "recon_hgemm_ms": r * 1e3,
                            "speedup": g / r, "rel_max_abs_diff": dy / scale})
            del x, y1, y2, xh
        del trellis, suh, svh, w_full
        torch.cuda.empty_cache()
    Path("/work/kld3/prefill-micro-unfused.json").write_text(json.dumps(results, indent=2))
    best = {}
    for r in results:
        if r["speedup"] > 1.0:
            best.setdefault(r["geometry"], r["m"])
    print("\ncrossover row count per geometry (first m where reconstruct wins):", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
