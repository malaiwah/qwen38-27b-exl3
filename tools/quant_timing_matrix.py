#!/usr/bin/env python3
"""Quantization/dequantization timing matrix: all K bitwidths × all FP precisions, GPU + CPU.

Runs in a separate container alongside vLLM (GPU shared via default compute mode).
"""
import os, sys, time, json, statistics
import torch

RESULTS = []

def bench(label, fn, warmup=3, iters=20, device="cuda"):
    for _ in range(warmup):
        try: fn()
        except: pass
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    median = statistics.median(times)
    p95 = sorted(times)[min(int(len(times) * 0.95), len(times)-1)]
    return {"label": label, "median_ms": round(median, 4), "p95_ms": round(p95, 4),
            "min_ms": round(min(times), 4), "max_ms": round(max(times), 4),
            "iters": iters, "device": device}

def log(r):
    RESULTS.append(r)
    print(f"  {r['label']:65s} med={r['median_ms']:10.4f}ms  p95={r['p95_ms']:10.4f}ms  [{r['device']}]")

DEVICE = torch.device("cuda:0")
CPU = torch.device("cpu")

SHAPES = {
    "gate_up": (5120, 17408),
    "down":    (17408, 5120),
    "attn":    (5120, 6144),
    "gdn":     (5120, 5120),
}
ACT_M = [1, 8, 128, 512, 1024, 2048, 4096]

print("=" * 90)
print("QUANTIZATION/DEQUANTIZATION TIMING MATRIX — RTX 5090 (SM120)")
print("=" * 90)

# ─── 1. torch.to() precision conversions (GPU) ────────────────────────────
print("\n── Precision Conversions (torch.to) GPU ──")
for name, (K, N) in SHAPES.items():
    W_f32 = torch.randn(K, N, dtype=torch.float32, device=DEVICE) * 0.1
    for src_dt, src_name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
        W = W_f32.to(src_dt)
        for dst_dt, dst_name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16"),
                                  (torch.float8_e4m3fn, "FP8E4M3")]:
            if src_dt == dst_dt: continue
            def fn(W=W, dt=dst_dt):
                return W.to(dt)
            r = bench(f"to {src_name}→{dst_name} {name} ({K}x{N})", fn, warmup=3, iters=20)
            log(r)

# ─── 2. torch.to() precision conversions (CPU) ───────────────────────────
print("\n── Precision Conversions (torch.to) CPU ──")
for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
    W_f32 = torch.randn(K, N, dtype=torch.float32) * 0.1
    for src_dt, src_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        W = W_f32.to(src_dt)
        for dst_dt, dst_name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
            if src_dt == dst_dt: continue
            def fn(W=W, dt=dst_dt):
                return W.to(dt)
            r = bench(f"CPU to {src_name}→{dst_name} {name} ({K}x{N})", fn, warmup=1, iters=5, device="cpu")
            log(r)

# ─── 3. cuBLAS matmul (FP16/BF16/FP32 × same → same) ─────────────────────
print("\n── cuBLAS Matmul (A×W→C) GPU ──")
for name, (K, N) in SHAPES.items():
    for M in ACT_M:
        for dt, dt_name in [(torch.float16, "FP16"), (torch.bfloat16, "BF16"), (torch.float32, "FP32")]:
            A = torch.randn(M, K, dtype=dt, device=DEVICE) * 0.1
            W = torch.randn(K, N, dtype=dt, device=DEVICE) * 0.1
            def fn(A=A, W=W):
                return A @ W
            r = bench(f"{dt_name} matmul {name} M={M} ({M}x{K}x{N})", fn, warmup=3, iters=20)
            log(r)
            flops = 2 * M * K * N
            tflops = flops / (r["median_ms"] / 1000) / 1e12
            print(f"    → {tflops:.1f} TFLOPS")

# ─── 4. FP8 quantization (per-tensor, per-row) ──────────────────────────
print("\n── FP8 Quantization (BF16→E4M3) GPU ──")
for M in [1, 128, 1024, 2048, 4096]:
    for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1

        # Per-tensor
        def fn(x=x):
            scale = x.abs().max().clamp_min(1e-12) / 448.0
            return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        r = bench(f"fp8_quant per-tensor {name} M={M} ({M}x{K})", fn, warmup=3, iters=20)
        log(r)

        # Per-row
        def fn(x=x):
            scale = x.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12) / 448.0
            return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        r = bench(f"fp8_quant per-row {name} M={M} ({M}x{K})", fn, warmup=3, iters=20)
        log(r)

# ─── 5. FP8 dequantization (E4M3→BF16) ──────────────────────────────────
print("\n── FP8 Dequantization (E4M3→BF16) GPU ──")
for name, (K, N) in SHAPES.items():
    x_fp8 = torch.randn(K, N, device=DEVICE).clamp(-448, 448).to(torch.float8_e4m3fn)
    for dst_dt, dst_name in [(torch.bfloat16, "BF16"), (torch.float16, "FP16"), (torch.float32, "FP32")]:
        def fn(x=x_fp8, dt=dst_dt):
            return x.to(dt)
        r = bench(f"fp8_dequant E4M3→{dst_name} {name} ({K}x{N})", fn, warmup=3, iters=20)
        log(r)

# ─── 6. FP4 simulation (pack 2 values per byte) ──────────────────────────
print("\n── FP4 Simulated Quant/Dequant (E2M1) GPU ──")
# FP4 E2M1: values 0,1,2,3,4,6 (positive), sign bit
# 8 levels: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEVICE)

for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
    W = torch.randn(K, N, dtype=torch.float32, device=DEVICE) * 2.0

    # Quantize to FP4 (nearest level)
    def fn(W=W):
        sign = W.sign()
        mag = W.abs()
        # Find nearest FP4 level
        idx = (mag.unsqueeze(-1) - FP4_LEVELS.unsqueeze(0).unsqueeze(0)).abs().argmin(dim=-1)
        return sign * FP4_LEVELS[idx]
    r = bench(f"fp4_quant_sim {name} ({K}x{N})", fn, warmup=2, iters=10)
    log(r)

    # Dequantize from packed FP4 (2 per byte)
    packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device=DEVICE)
    def fn(p=packed):
        low = (p & 0x0F).to(torch.float16)
        high = ((p >> 4) & 0x0F).to(torch.float16)
        return torch.stack([low, high], dim=-1).reshape(K, N)
    r = bench(f"fp4_dequant_sim unpack {name} ({K}x{N})", fn, warmup=3, iters=20)
    log(r)

# ─── 7. Hadamard transform (128-point) ──────────────────────────────────
print("\n── Hadamard Transform (128-point) GPU ──")
# H128 matrix
H128 = torch.randn(128, 128, dtype=torch.float32, device=DEVICE)
# Make it orthogonal
H128 = torch.linalg.qr(H128)[0]

for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
    W = torch.randn(K, N, dtype=torch.float32, device=DEVICE) * 0.1
    # Reshape to blocks
    k_blk = K // 128
    n_blk = N // 128
    W_blk = W.reshape(k_blk, 128, n_blk, 128)

    # Apply H on K dimension
    def fn(W_blk=W_blk, H=H128):
        return torch.einsum("ab,ibjd->iajd", H, W_blk)
    r = bench(f"hadamard_K {name} ({K}x{N})", fn, warmup=2, iters=10)
    log(r)

    # Apply H on N dimension
    def fn(W_blk=W_blk, H=H128):
        return torch.einsum("iajb,bc->iajc", W_blk, H)
    r = bench(f"hadamard_N {name} ({K}x{N})", fn, warmup=2, iters=10)
    log(r)

    # Full fold (both dimensions + scales)
    suh = torch.ones(K, dtype=torch.float32, device=DEVICE)
    svh = torch.ones(N, dtype=torch.float32, device=DEVICE)
    def fn(W=W, H=H128, suh=suh, svh=svh):
        Wb = W.reshape(k_blk, 128, n_blk, 128)
        t = torch.einsum("ab,ibjd->iajd", H, Wb)
        f = torch.einsum("iajb,bc->iajc", t, H)
        return f.reshape(K, N) * suh.reshape(K,1) * svh.reshape(1,N)
    r = bench(f"hadamard_fold_full {name} ({K}x{N})", fn, warmup=2, iters=10)
    log(r)

# ─── 8. Scaling operations ────────────────────────────────────────────────
print("\n── Scaling Operations GPU ──")
for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
    W = torch.randn(K, N, dtype=torch.float16, device=DEVICE) * 0.1
    s = torch.ones(K, dtype=torch.float16, device=DEVICE)

    # Row scaling (diag(s) @ W)
    def fn(W=W, s=s):
        return W * s.reshape(-1, 1)
    r = bench(f"row_scale {name} ({K}x{N})", fn, warmup=3, iters=20)
    log(r)

    # Column scaling (W @ diag(s))
    s2 = torch.ones(N, dtype=torch.float16, device=DEVICE)
    def fn(W=W, s=s2):
        return W * s.reshape(1, -1)
    r = bench(f"col_scale {name} ({K}x{N})", fn, warmup=3, iters=20)
    log(r)

    # Both row + col scale
    def fn(W=W, sr=s, sc=s2):
        return W * sr.reshape(-1, 1) * sc.reshape(1, -1)
    r = bench(f"row+col_scale {name} ({K}x{N})", fn, warmup=3, iters=20)
    log(r)

# ─── 9. abs().max() (for amax calculation) ───────────────────────────────
print("\n── Amax Operations (for quant scale) GPU ──")
for M in [1, 128, 512, 2048, 4096]:
    for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1

        # Full tensor amax
        def fn(x=x):
            return x.abs().max()
        r = bench(f"amax full M={M} {name} ({M}x{K})", fn, warmup=3, iters=20)
        log(r)

        # Per-row amax
        def fn(x=x):
            return x.abs().max(dim=1).values
        r = bench(f"amax per-row M={M} {name} ({M}x{K})", fn, warmup=3, iters=20)
        log(r)

# ─── 10. CPU matmul for comparison ────────────────────────────────────────
print("\n── CPU Matmul for Comparison ──")
for name, (K, N) in [("gate_up", (5120, 17408)), ("down", (17408, 5120))]:
    for M in [1, 128]:
        for dt, dt_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
            A = torch.randn(M, K, dtype=dt) * 0.1
            W = torch.randn(K, N, dtype=dt) * 0.1
            def fn(A=A, W=W):
                return A @ W
            r = bench(f"CPU {dt_name} matmul {name} M={M} ({M}x{K}x{N})", fn, warmup=1, iters=3, device="cpu")
            log(r)
            flops = 2 * M * K * N
            tflops = flops / (r["median_ms"] / 1000) / 1e12
            print(f"    → {tflops:.1f} TFLOPS")

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Total benchmarks: {len(RESULTS)}")

with open("/tmp/quant_timing_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("Results saved to /tmp/quant_timing_results.json")

# Key comparisons
print("\n── Key Comparisons ──")
gpu_results = [r for r in RESULTS if r["device"] == "cuda"]
cpu_results = [r for r in RESULTS if r["device"] == "cpu"]

print("\nFastest GPU ops:")
for r in sorted(gpu_results, key=lambda x: x["median_ms"])[:10]:
    print(f"  {r['label']:65s} {r['median_ms']:10.4f}ms")

print("\nSlowest GPU ops:")
for r in sorted(gpu_results, key=lambda x: -x["median_ms"])[:10]:
    print(f"  {r['label']:65s} {r['median_ms']:10.4f}ms")

print("\nAll CPU ops:")
for r in sorted(cpu_results, key=lambda x: x["median_ms"]):
    print(f"  {r['label']:65s} {r['median_ms']:10.4f}ms")

# Matmul TFLOPS comparison
print("\n── Matmul TFLOPS by precision (gate_up, M=2048) ──")
for r in gpu_results:
    if "matmul gate_up M=2048" in r["label"]:
        K, N = 5120, 17408
        M = 2048
        flops = 2 * M * K * N
        tflops = flops / (r["median_ms"] / 1000) / 1e12
        print(f"  {r['label']:65s} {tflops:.1f} TFLOPS")

print("\n── Conversion speed by size ──")
for name in ["gate_up", "down"]:
    print(f"\n  {name}:")
    for r in gpu_results:
        if name in r["label"] and "→" in r["label"]:
            print(f"    {r['label']:65s} {r['median_ms']:10.4f}ms")
