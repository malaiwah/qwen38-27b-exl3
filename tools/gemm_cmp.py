#!/usr/bin/env python3
"""Is the custom `ext.hgemm` leaving performance on the table versus cuBLAS?

The reconstruct path from PR #316 spends its time in `ext.hgemm`. If cuBLAS
(`torch.mm`) is faster on the same fp16 shapes, the prefill patch can be improved for
free.
"""
import sys, time, torch
sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext


def bench(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


print(f"{'K':>6} {'N':>7} {'m':>5} {'ext.hgemm':>11} {'torch.mm':>11} {'ratio':>7}")
for K, N in ((5120, 17408), (17408, 5120), (5120, 32768)):
    w = torch.randn((K, N), dtype=torch.float16, device="cuda") * 0.02
    for m in (256, 1024, 2048, 4096):
        x = torch.randn((m, K), dtype=torch.float16, device="cuda") * 0.05
        y = torch.empty((m, N), dtype=torch.float16, device="cuda")
        a = bench(lambda: ext.hgemm(x, w, y))
        b = bench(lambda: torch.mm(x, w, out=y))
        print(f"{K:6d} {N:7d} {m:5d} {a*1e3:10.3f}ms {b*1e3:10.3f}ms {a/b:6.2f}x")
        del x, y
    del w
    torch.cuda.empty_cache()
