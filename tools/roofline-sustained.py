"""Steady-state roofline for the RTX 5090, with in-loop power/clock sampling.

Fixes two flaws in the first version:
  1. benchmarks were too short to raise clocks off idle (a 4096^3 fp16 GEMM
     measured 68 TFLOP/s at 195 MHz, versus 233 at 2842 MHz) -- each case now
     runs a sustained warmup until clocks stop rising;
  2. power/clocks were sampled after the timed loop, by which point the GPU had
     already dropped back to idle -- a background thread now samples during it.
"""
import os
import statistics
import subprocess
import threading
import time

import torch

DEV = "cuda:0"
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name}  SMs={p.multi_processor_count}  "
      f"VRAM={p.total_memory/1024**3:.2f} GiB")


def smi(fields):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout.strip()
    return [float(x) for x in out.split(",")]


LIMIT = smi("power.limit")[0]
print(f"enforced power limit: {LIMIT:.0f} W\n")


class Sampler(threading.Thread):
    """Poll power and SM clock while a benchmark is running."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.w, self.c = [], []

    def run(self):
        while not self.stop.is_set():
            try:
                w, c = smi("power.draw,clocks.sm")
                self.w.append(w)
                self.c.append(c)
            except Exception:
                pass
            time.sleep(0.05)

    def result(self):
        f = (max(self.w) if self.w else float("nan"),
             statistics.median(self.c) if self.c else float("nan"))
        return f


def measure(fn, flops_per_call=None, bytes_per_call=None, secs=4.0):
    """Sustained-load measurement: warm until clocks plateau, then time."""
    # sustained warmup so clocks and power reach steady state
    t_end = time.perf_counter() + 2.5
    while time.perf_counter() < t_end:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()

    s = Sampler(); s.start()
    n_iter, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        n_iter += 10
    dt = time.perf_counter() - t0
    s.stop.set(); s.join(timeout=1)
    watts, mhz = s.result()
    per = dt / n_iter
    rate = (flops_per_call / per / 1e12) if flops_per_call else (
        bytes_per_call / per / 1e9)
    return rate, watts, mhz


print("=== HBM bandwidth (sustained) ===")
best_bw = 0.0
for mb in (1024, 2048):
    n = mb * 1024 * 1024 // 2
    a = torch.empty(n, dtype=torch.float16, device=DEV).fill_(1.0)
    b = torch.empty(n, dtype=torch.float16, device=DEV).fill_(2.0)
    gbs, w, c = measure(lambda: b.copy_(a), bytes_per_call=2 * n * 2)
    best_bw = max(best_bw, gbs)
    print(f"  copy {mb:>5} MiB : {gbs:8.1f} GB/s  [{w:5.0f} W, {c:.0f} MHz]")
    del a, b
    torch.cuda.empty_cache()

print("\n=== dense GEMM (sustained) ===")
peaks = {}
for name, dt_ in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    best = 0.0
    for n in (8192, 12288):
        A = torch.randn((n, n), dtype=dt_, device=DEV)
        B = torch.randn((n, n), dtype=dt_, device=DEV)
        tf, w, c = measure(lambda: torch.matmul(A, B),
                           flops_per_call=2 * n ** 3)
        best = max(best, tf)
        print(f"  {name} {n}^3 : {tf:8.1f} TFLOP/s  [{w:5.0f} W, {c:.0f} MHz]")
        del A, B
        torch.cuda.empty_cache()
    peaks[name] = best

for n in (8192, 12288):
    try:
        A = torch.randn((n, n), device=DEV).to(torch.float8_e4m3fn)
        B = torch.randn((n, n), device=DEV).to(
            torch.float8_e4m3fn).t().contiguous().t()
        sc = torch.tensor(1.0, device=DEV)
        tf, w, c = measure(
            lambda: torch._scaled_mm(A, B, scale_a=sc, scale_b=sc,
                                     out_dtype=torch.bfloat16),
            flops_per_call=2 * n ** 3)
        peaks["fp8"] = max(peaks.get("fp8", 0.0), tf)
        print(f"  fp8  {n}^3 : {tf:8.1f} TFLOP/s  [{w:5.0f} W, {c:.0f} MHz]")
        del A, B
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  fp8 {n}^3: unavailable ({type(e).__name__})")

print("\n=== vs published dense specification ===")
SPEC = {"bandwidth": 1792.0, "fp16": 209.5, "bf16": 209.5, "fp8": 419.0}
print(f"  bandwidth : {best_bw:8.1f} vs {SPEC['bandwidth']:7.1f} GB/s  "
      f"= {100*best_bw/SPEC['bandwidth']:5.1f}%   (VRAM is +6000 offset via lactd)")
for k in ("fp16", "bf16", "fp8"):
    if k in peaks:
        print(f"  {k:<9} : {peaks[k]:8.1f} vs {SPEC[k]:7.1f} TFLOP/s = "
              f"{100*peaks[k]/SPEC[k]:5.1f}%")
print(f"\nRESULT_JSON {{\"bw\": {best_bw:.1f}, "
      + ", ".join(f'"{k}": {v:.1f}' for k, v in peaks.items()) + "}")
