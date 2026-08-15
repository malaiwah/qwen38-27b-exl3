#!/usr/bin/env python3
"""Correctness and speed of reconstruct_fp8_slice against the fp16 reconstruct path.

Correctness: the FP8 kernel must decode the same trellis to the same values, transposed
and narrowed. Compared against reconstruct() -> transpose -> cast in torch, which is the
reference for what the kernel does in one pass.

Speed: the end-to-end prefill primitive, reconstruct + GEMM, in both dtypes.
"""
import sys
import time

import torch

sys.path.insert(0, "/work/exllamav3")
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

E4M3_MAX = 448.0


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
    torch.manual_seed(0)
    print("== correctness")
    for k, n, bits, mcg in ((5120, 17408, 5, True), (17408, 5120, 6, True),
                            (2048, 2048, 4, True)):
        trellis = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                                dtype=torch.int16, device=dev)
        w16 = torch.empty((k, n), dtype=torch.float16, device=dev)
        ext.reconstruct(w16, trellis, bits, mcg, False)
        scale = (w16.abs().amax().float() / E4M3_MAX).clamp_(min=1e-9)
        inv = float(1.0 / scale.item())
        # Reference must scale in fp32, like the kernel does: dividing in fp16 first
        # changes which ties round up, and one E4M3 ULP near 448 is 32.0.
        want = (w16.t().contiguous().float() * inv).to(torch.float8_e4m3fn)

        got = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=dev)
        ext.reconstruct_fp8_slice(got, trellis, bits, mcg, False, 0, inv)
        gb = got.view(torch.uint8).int()
        wb = want.view(torch.uint8).int()
        same = int((gb != wb).sum().item())
        # distance in E4M3 code space, sign bit excluded: 1 means an adjacent
        # representable value, i.e. a rounding tie rather than a decode error
        ulp = (gb & 0x7F) - (wb & 0x7F)
        sign_diff = int(((gb ^ wb) & 0x80 != 0).sum().item())
        maxulp = int(ulp.abs().max().item())
        rel = ((got.float() - want.float()).abs()
               / want.float().abs().clamp(min=1e-6)).max().item()
        print(f"  K={k:6d} N={n:6d} bits={bits}: differing bytes={same}/{gb.numel()} "
              f"max ULP={maxulp} sign flips={sign_diff} max rel={rel:.4f}")
        if maxulp > 1 or sign_diff:
            print("  MISMATCH: not a rounding tie")
            return 1
        del trellis, w16, want, got
        torch.cuda.empty_cache()

    print("\n== speed, reconstruct + GEMM at prefill row counts")
    print(f"{'K':>6} {'N':>7} {'bits':>4} {'m':>5} {'fp16 path':>11} {'fp8 path':>10} "
          f"{'speedup':>8}")
    for k, n, bits in ((5120, 17408, 5), (17408, 5120, 6)):
        trellis = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                                dtype=torch.int16, device=dev)
        w16 = torch.empty((k, n), dtype=torch.float16, device=dev)
        ext.reconstruct(w16, trellis, bits, True, False)
        scale_b = (w16.abs().amax().float() / E4M3_MAX).clamp_(min=1e-9)
        inv_b = float(1.0 / scale_b.item())
        w8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=dev)
        for m in (256, 1024, 2048, 4096):
            x = torch.randn((m, k), dtype=torch.float16, device=dev) * 0.05
            y = torch.empty((m, n), dtype=torch.float16, device=dev)
            scale_a = (x.abs().amax().float() / E4M3_MAX).clamp_(min=1e-9)
            x8 = (x / scale_a).to(torch.float8_e4m3fn)

            def fp16_path():
                ext.reconstruct(w16, trellis, bits, True, False)
                ext.hgemm(x, w16, y)

            def fp8_path():
                ext.reconstruct_fp8_slice(w8, trellis, bits, True, False, 0, inv_b)
                torch._scaled_mm(x8, w8.t(), scale_a=scale_a, scale_b=scale_b,
                                 out_dtype=torch.float16)

            a = bench(fp16_path)
            b = bench(fp8_path)
            print(f"{k:6d} {n:7d} {bits:4d} {m:5d} {a*1e3:10.3f}ms {b*1e3:9.3f}ms "
                  f"{a/b:7.2f}x")
            del x, y, x8
        del trellis, w16, w8
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
