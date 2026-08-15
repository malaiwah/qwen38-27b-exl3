#!/usr/bin/env python3
"""Is the prefill ceiling the trellis decode, or the GEMM's dtype?

Measured facts going in: reconstruct+hgemm at m=2048 costs 1.20 ms for a gate_proj
shape, of which the fp16 GEMM alone is 0.90 ms, and ext.hgemm is already at cuBLAS
parity. If an FP8 GEMM on the same shapes is materially faster than fp16, then the
remaining gap to official FP8 is the *output dtype* of our GEMM, not the trellis, and
the fix is to reconstruct into FP8 and use a scaled FP8 matmul for prefill.

Reports fp16 (torch.mm) against FP8 E4M3 (torch._scaled_mm) at the three shapes that
carry the MLP: gate/up, down, and the head.
"""
import time

import torch


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main() -> int:
    dev = "cuda"
    print(f"{'K':>6} {'N':>7} {'m':>5} {'fp16 mm':>10} {'fp8 scaled':>11} {'speedup':>8} "
          f"{'fp16 TFLOP/s':>13} {'fp8 TFLOP/s':>12}")
    for K, N in ((5120, 17408), (17408, 5120), (5120, 248320)):
        w16 = (torch.randn((K, N), dtype=torch.float16, device=dev) * 0.02)
        # FP8 operands: row-major activation, column-major weight, per-tensor scales
        w8 = w16.to(torch.float8_e4m3fn).t().contiguous().t()
        sw = torch.tensor(1.0, device=dev)
        for m in (256, 1024, 2048, 4096):
            x16 = torch.randn((m, K), dtype=torch.float16, device=dev) * 0.05
            y16 = torch.empty((m, N), dtype=torch.float16, device=dev)
            x8 = x16.to(torch.float8_e4m3fn)
            sx = torch.tensor(1.0, device=dev)
            flop = 2.0 * m * K * N
            a = bench(lambda: torch.mm(x16, w16, out=y16))
            try:
                b = bench(lambda: torch._scaled_mm(
                    x8, w8, scale_a=sx, scale_b=sw, out_dtype=torch.float16))
            except Exception as exc:  # unsupported layout/arch reports itself
                print(f"{K:6d} {N:7d} {m:5d} {a*1e3:9.3f}ms  fp8 unavailable: "
                      f"{type(exc).__name__}: {str(exc)[:80]}")
                continue
            print(f"{K:6d} {N:7d} {m:5d} {a*1e3:9.3f}ms {b*1e3:10.3f}ms {a/b:7.2f}x "
                  f"{flop/a/1e12:12.1f} {flop/b/1e12:11.1f}")
            del x16, y16, x8
        del w16, w8
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
