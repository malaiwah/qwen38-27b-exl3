"""Measure this RTX 5090's achievable ceilings, then compare to served numbers.

Everything here is measured on the device; nothing is taken from a spec sheet.
"""
import json
import os
import time

import torch

DEV = "cuda:0"
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name}  SMs={p.multi_processor_count}  "
      f"VRAM={p.total_memory/1024**3:.2f} GiB")


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------- bandwidth
print("\n=== HBM bandwidth (achievable) ===")
best_bw = 0.0
for mb in (256, 512, 1024, 2048):
    n = mb * 1024 * 1024 // 2          # fp16 elements
    a = torch.empty(n, dtype=torch.float16, device=DEV)
    b = torch.empty(n, dtype=torch.float16, device=DEV)
    a.fill_(1.0); b.fill_(2.0)
    # copy_: reads a, writes b -> 2 * bytes moved
    t = timed(lambda: b.copy_(a))
    gbs = 2 * n * 2 / t / 1e9
    best_bw = max(best_bw, gbs)
    print(f"  copy  {mb:>5} MiB : {gbs:8.1f} GB/s")
    del a, b
    torch.cuda.empty_cache()
print(f"  -> achievable bandwidth: {best_bw:.1f} GB/s "
      f"({best_bw/1e3:.2f} TB/s)")

# ---------------------------------------------------------------- GEMM peak
print("\n=== dense GEMM throughput (achievable) ===")
peaks = {}
for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    best = 0.0
    for n in (4096, 8192):
        A = torch.randn((n, n), dtype=dt, device=DEV)
        B = torch.randn((n, n), dtype=dt, device=DEV)
        t = timed(lambda: torch.matmul(A, B), iters=10)
        tflops = 2 * n ** 3 / t / 1e12
        best = max(best, tflops)
        print(f"  {name} {n}^3 : {tflops:8.1f} TFLOP/s")
        del A, B
        torch.cuda.empty_cache()
    peaks[name] = best

# fp8 via _scaled_mm if the build exposes it
try:
    n = 8192
    A = torch.randn((n, n), device=DEV).to(torch.float8_e4m3fn)
    B = torch.randn((n, n), device=DEV).to(torch.float8_e4m3fn).t().contiguous().t()
    s = torch.tensor(1.0, device=DEV)
    t = timed(lambda: torch._scaled_mm(A, B, scale_a=s, scale_b=s,
                                       out_dtype=torch.bfloat16), iters=10)
    peaks["fp8"] = 2 * n ** 3 / t / 1e12
    print(f"  fp8  {n}^3 : {peaks['fp8']:8.1f} TFLOP/s")
    del A, B
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  fp8: unavailable ({type(e).__name__})")

# ---------------------------------------------------------------- model FLOPs
print("\n=== model matmul cost per token ===")
cfg = json.load(open(os.environ["SNAP"] + "/config.json"))
tc = cfg.get("text_config", cfg)
H = tc["hidden_size"]
L = tc["num_hidden_layers"]
I = tc.get("intermediate_size")
V = tc["vocab_size"]
n_heads = tc.get("num_attention_heads")
n_kv = tc.get("num_key_value_heads", n_heads)
head_dim = tc.get("head_dim") or (H // n_heads)
layer_types = tc.get("layer_types") or []
n_full = sum(1 for t in layer_types if "full" in t) or L
n_lin = L - n_full

mlp = 3 * H * I                      # gate, up, down
attn = H * (n_heads + 2 * n_kv) * head_dim + n_heads * head_dim * H
params_mlp = L * mlp
params_attn = n_full * attn
head = H * V
total = params_mlp + params_attn + head
print(f"  H={H} L={L} (full-attn {n_full}, linear-attn {n_lin}) I={I} V={V}")
print(f"  MLP        {params_mlp/1e9:6.2f} e9 params")
print(f"  full-attn  {params_attn/1e9:6.2f} e9")
print(f"  lm_head    {head/1e9:6.2f} e9")
print(f"  counted    {total/1e9:6.2f} e9  (excludes GDN projections / norms)")
flops_tok = 2 * total
print(f"  -> {flops_tok/1e9:.1f} GFLOP per token (2*params)")

# ---------------------------------------------------------------- ceilings
print("\n=== PREFILL ceiling vs measured ===")
MEAS = {
    "throughput (all-FP4)":  (7694.9, 15.88, "fp4"),
    "balanced (gate_up FP6)": (3266.3, 21.2, "fp6"),
    "fidelity (all-trellis)": (1965.4, 18.83, "trellis"),
}
# effective FP4 rate from the production bake-off: all matrices, M=2051, 164.7 ms
bakeoff_ms = 164.7
fp4_eff = flops_tok * 2051 / (bakeoff_ms / 1e3) / 1e12
print(f"  measured FP4 GEMM rate (bake-off, real shapes): {fp4_eff:.0f} TFLOP/s")
print(f"  measured fp16 dense peak                      : {peaks['fp16']:.0f} TFLOP/s")
for name, (pp, _gib, kind) in MEAS.items():
    rate = fp4_eff if kind == "fp4" else peaks["fp16"]
    ceil = rate * 1e12 / flops_tok
    print(f"  {name:<24} PP {pp:7.1f}  GEMM-bound ceiling "
          f"{ceil:8.0f}  -> {100*pp/ceil:5.1f}% of ceiling")

print("\n=== DECODE ceiling vs measured (weight-bandwidth bound) ===")
for name, (_pp, gib, _k) in MEAS.items():
    fwd_ceil = best_bw * 1e9 / (gib * 1024 ** 3)
    print(f"  {name:<24} weights {gib:5.2f} GiB -> "
          f"{fwd_ceil:6.1f} target forwards/s ceiling")
