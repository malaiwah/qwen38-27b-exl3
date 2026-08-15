#!/usr/bin/env python3
"""Is B12X's native K6 kernel the right choice at prefill row counts?

The serialized EXL3 path routes any K6/MCG shard with 128-divisible dimensions to B12X's
`trellis_linear`, and that branch is taken *before* the reconstruct dispatch from PR #316 is
consulted. So on this checkpoint family `down_proj` (64 modules), `lm_head`, and - in the
serialized builds - all 208 attention projections never see the prefill path at all: they run
the native kernel at every row count, decode and prefill alike.

That is fine if B12X wins at large M. This measures it on the real shapes, against the two
alternatives the dispatch already knows how to build.

Run inside the container so b12x and the EXL3 extension are importable.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Use the JIT build, not the image's baked extension: only the former carries
# reconstruct_fp8_slice, and the whole point is to price the FP8 path.
sys.path.insert(0, "/work/exllamav3")
try:
    from exllamav3.ext import exllamav3_ext as ext  # noqa: E402
except Exception:
    sys.path.insert(0, "/opt/exllamav3")
    import exllamav3_ext as ext  # noqa: E402

E4M3_MAX = 448.0


def bench(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def load_b12x():
    """b12x.gemm.trellis_linear is a module: prepare_weight() then run() with scratch."""
    from b12x.gemm import trellis_linear
    return trellis_linear


def c_tmp_elements(rows: int, columns: int) -> int:
    # mirrors the runtime's scratch sizing for the native path
    return max(1, rows * columns)


def main() -> int:
    dev = "cuda"
    try:
        trellis_linear = load_b12x()
    except Exception as exc:
        print(json.dumps({"b12x_available": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 0
    print("b12x_available: True", flush=True)

    # real geometries from this family: down_proj K6, lm_head K6, attention in_proj_qkvz K6
    shapes = [("down_proj", 17408, 5120, 6), ("lm_head", 5120, 248320, 6),
              ("attn_in_proj_qkvz", 5120, 16384, 6)]
    rows = (1, 8, 128, 512, 2048)
    out = {"schema": "qwen38-b12x-vs-reconstruct/1", "b12x_available": True, "results": []}
    print(f"{'shape':20s} {'m':>5} {'b12x':>10} {'recon+hgemm':>12} {'recon+fp8':>10} "
          f"{'best':>12}")
    for name, k, n, bits in shapes:
        trellis = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                                dtype=torch.int16, device=dev)
        suh = torch.randn((k,), dtype=torch.float16, device=dev)
        svh = torch.randn((n,), dtype=torch.float16, device=dev)
        w16 = torch.empty((k, n), dtype=torch.float16, device=dev)
        ext.reconstruct(w16, trellis, bits, True, False)
        scale_b = (w16.abs().amax().float() / E4M3_MAX).clamp_(min=1e-9)
        inv_b = float(1.0 / scale_b.item())
        w8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=dev)
        for m in rows:
            x = (torch.randn((m, k), dtype=torch.float16, device=dev) * 0.05).contiguous()
            y = torch.empty((m, n), dtype=torch.float16, device=dev)
            scale_a = (x.abs().amax().float() / E4M3_MAX).clamp_(min=1e-9)
            pad = (m + 15) // 16 * 16
            x8 = torch.zeros((pad, k), dtype=torch.float8_e4m3fn, device=dev)
            x8[:m].copy_(x / scale_a)

            weight = trellis_linear.prepare_weight(
                trellis, suh, svh, codebook="mcg", params_dtype=torch.float16)
            b_out = torch.empty((m, n), dtype=torch.float16, device=dev)
            b_gemm = torch.empty_like(b_out)
            b_ctmp = torch.empty((c_tmp_elements(m, n),), dtype=torch.float32, device=dev)
            b_rot = torch.empty_like(x)

            def path_b12x():
                trellis_linear.run(x, weight, output=b_out, gemm_output=b_gemm,
                                   c_tmp=b_ctmp, rotated_f16=b_rot)

            def path_recon():
                x_had = torch.empty_like(x)
                ext.had_r_128(x, x_had, suh, None, 1.0)
                ext.reconstruct(w16, trellis, bits, True, False)
                ext.hgemm(x_had, w16, y)
                ext.had_r_128(y, y, None, svh, 1.0)

            def path_fp8():
                x_had = torch.empty_like(x)
                ext.had_r_128(x, x_had, suh, None, 1.0)
                ext.reconstruct_fp8_slice(w8, trellis, bits, True, False, 0, inv_b)
                torch._scaled_mm(x8, w8.t(), scale_a=scale_a, scale_b=scale_b,
                                 out_dtype=torch.float16)
                ext.had_r_128(y, y, None, svh, 1.0)

            try:
                a = bench(path_b12x)
            except Exception as exc:
                print(f"{name:20s} {m:5d} b12x failed: {type(exc).__name__}: {str(exc)[:60]}")
                continue
            b = bench(path_recon)
            c = bench(path_fp8) if hasattr(ext, "reconstruct_fp8_slice") else float("inf")
            best = min((a, "b12x"), (b, "recon+hgemm"), (c, "recon+fp8"))[1]
            print(f"{name:20s} {m:5d} {a*1e3:9.3f}ms {b*1e3:11.3f}ms {c*1e3:9.3f}ms "
                  f"{best:>12}")
            out["results"].append({"shape": name, "k": k, "n": n, "m": m,
                                   "b12x_ms": a * 1e3, "recon_hgemm_ms": b * 1e3,
                                   "recon_fp8_ms": c * 1e3, "best": best,
                                   "speedup_best_over_b12x": a / min(b, c)})
            del x, y, x8, b_out, b_gemm, b_ctmp, b_rot
        del trellis, w16, w8, suh, svh
        torch.cuda.empty_cache()
    Path("/work/kld3/b12x-vs-reconstruct.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
