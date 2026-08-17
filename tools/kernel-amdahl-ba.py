"""Kernel-level A/B for the in_proj_ba tiny-N tall-K linear (docs/47 F6/F10.2).

Measures the served TN path (F.linear on a [N=96, K=5120] weight) against the
patch path (torch.mm on a pre-transposed [K=5120, N=96] weight) at decode row
counts, using CUDA events around CUDA-graph replays so that host-side dispatch
(proot-inflated on this box) is excluded from the number.

Faithfulness choices:
  * one graph replay = 48 calls = one decode step's worth of GDN in_proj_ba,
    each against its OWN weight tensor (48 x 983 KB = 47 MB working set), so a
    single weight cannot sit hot in registers/L2 across the whole measurement.
  * arms are INTERLEAVED sample-by-sample so any clock/thermal drift hits both.
  * a second pass adds an L2-flushing read before every call and reports the
    call cost by DIFFERENCE against a flush-only graph, bracketing the cold case.
"""

import json
import statistics
import sys

import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16
K = 5120          # in_proj_ba input features
N = 96            # in_proj_ba output features
LAYERS = 48       # GDN layers -> calls per decode step
SAMPLES = 200     # timed graph replays per arm
WARM_EAGER = 50
WARM_REPLAY = 20

torch.manual_seed(0)


def clocks():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1),
        }
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}


def stats(per_call_us):
    n = len(per_call_us)
    mean = statistics.fmean(per_call_us)
    sd = statistics.stdev(per_call_us)
    return {
        "mean_us": round(mean, 4),
        "median_us": round(statistics.median(per_call_us), 4),
        "stdev_us": round(sd, 4),
        "sem_us": round(sd / (n ** 0.5), 5),
        "rel_sem_pct": round(100.0 * (sd / (n ** 0.5)) / mean, 4),
        "cv_pct": round(100.0 * sd / mean, 4),
        "min_us": round(min(per_call_us), 4),
        "max_us": round(max(per_call_us), 4),
        "samples": n,
        "calls_per_sample": LAYERS,
    }


def build(m, flush_mb=0):
    """Returns (fn_tn, fn_nn, fn_flushonly | None) closures over fresh tensors."""
    x = torch.randn(m, K, device=DEV, dtype=DT)
    # 48 distinct weights, exactly as the 48 GDN layers hold distinct weights.
    w_nk = [torch.randn(N, K, device=DEV, dtype=DT) for _ in range(LAYERS)]
    w_kn = [w.t().contiguous() for w in w_nk]

    flush = None
    if flush_mb:
        flush = torch.randn(flush_mb * 1024 * 1024 // 2, device=DEV, dtype=DT)
        sink = torch.empty(1, device=DEV, dtype=torch.float32)

    # tn(): the SERVED path -- UnquantizedLinearMethod.apply ->
    #       dispatch_unquantized_gemm() -> F.linear(x, weight[N,K], bias=None).
    # nn(): the PATCH path -- torch.mm(x, layer.weight_kn) with weight_kn = weight.t().contiguous().
    # Both allocate their output, exactly as both code paths do.
    def tn():
        for w in w_nk:
            if flush is not None:
                sink.copy_(flush.sum(dtype=torch.float32).view(1))
            F.linear(x, w)

    def nn():
        for w in w_kn:
            if flush is not None:
                sink.copy_(flush.sum(dtype=torch.float32).view(1))
            torch.mm(x, w)

    def flushonly():
        for _ in range(LAYERS):
            sink.copy_(flush.sum(dtype=torch.float32).view(1))

    return x, w_nk, w_kn, tn, nn, (flushonly if flush is not None else None)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def interleaved(graphs, samples):
    """graphs: dict name -> CUDAGraph. Returns dict name -> [per_call_us]."""
    ev = {k: (torch.cuda.Event(True), torch.cuda.Event(True)) for k in graphs}
    acc = {k: [] for k in graphs}
    for _ in range(WARM_REPLAY):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    for _ in range(samples):
        for k, g in graphs.items():
            a, b = ev[k]
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            acc[k].append(a.elapsed_time(b) * 1000.0 / LAYERS)
    return acc


def run_shape(m):
    x, w_nk, w_kn, tn, nn, _ = build(m)

    # correctness: the two arms must compute the same product
    ref = (x.float() @ w_nk[0].t().float())
    a = F.linear(x, w_nk[0]).float()
    b = torch.mm(x, w_kn[0]).float()
    relerr = {
        "tn_vs_fp32": round((a - ref).norm().item() / ref.norm().item(), 8),
        "nn_vs_fp32": round((b - ref).norm().item() / ref.norm().item(), 8),
        "tn_vs_nn_max_abs": round((a - b).abs().max().item(), 8),
    }

    # eager warmup (kernel selection / autotune caches)
    for _ in range(WARM_EAGER):
        tn()
        nn()
    torch.cuda.synchronize()

    res = interleaved({"tn_linear_served": capture(tn), "nn_mm_transposed": capture(nn)}, SAMPLES)
    out = {
        "m": m,
        "relerr": relerr,
        "arms": {k: stats(v) for k, v in res.items()},
    }
    t = out["arms"]["tn_linear_served"]
    p = out["arms"]["nn_mm_transposed"]
    out["speedup_k"] = {
        "mean_ratio": round(t["mean_us"] / p["mean_us"], 4),
        "median_ratio": round(t["median_us"] / p["median_us"], 4),
    }
    out["per_step_us"] = {
        "tn_served": round(t["mean_us"] * LAYERS, 2),
        "nn_transposed": round(p["mean_us"] * LAYERS, 2),
        "saved": round((t["mean_us"] - p["mean_us"]) * LAYERS, 2),
    }
    # effective bandwidth identity: weight bytes dominate at tiny m
    wbytes = N * K * 2
    out["effective_gbps"] = {
        "weight_bytes": wbytes,
        "tn_served": round(wbytes / (t["mean_us"] * 1e-6) / 1e9, 1),
        "nn_transposed": round(wbytes / (p["mean_us"] * 1e-6) / 1e9, 1),
    }
    return out


def run_flush(m, flush_mb=192):
    """Cold-weight bracket: cost by difference against a flush-only graph."""
    x, w_nk, w_kn, tn, nn, fo = build(m, flush_mb=flush_mb)
    for _ in range(10):
        tn(); nn(); fo()
    torch.cuda.synchronize()
    res = interleaved(
        {"tn_plus_flush": capture(tn), "nn_plus_flush": capture(nn), "flush_only": capture(fo)},
        60,
    )
    s = {k: stats(v) for k, v in res.items()}
    fl = s["flush_only"]["mean_us"]
    tn_c = s["tn_plus_flush"]["mean_us"] - fl
    nn_c = s["nn_plus_flush"]["mean_us"] - fl
    return {
        "m": m,
        "flush_mb": flush_mb,
        "raw": s,
        "derivation": (
            "cold_call = mean(op+flush) - mean(flush_only); "
            f"tn = {s['tn_plus_flush']['mean_us']} - {fl} = {round(tn_c, 3)} us; "
            f"nn = {s['nn_plus_flush']['mean_us']} - {fl} = {round(nn_c, 3)} us"
        ),
        "cold_tn_us": round(tn_c, 3),
        "cold_nn_us": round(nn_c, 3),
        "cold_speedup_k": round(tn_c / nn_c, 4) if nn_c > 0 else None,
    }


def main():
    out = {
        "schema": "qwen38-kernel-amdahl-ba/1",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": f"x[m,{K}] bf16 @ W[{N},{K}] bf16  (in_proj_ba)",
        "method": {
            "timer": "torch.cuda.Event(enable_timing=True) around CUDAGraph.replay()",
            "why_graph": "excludes host dispatch; this box runs under proot which inflates dispatch",
            "calls_per_replay": LAYERS,
            "samples_per_arm": SAMPLES,
            "warmup": f"{WARM_EAGER} eager iterations of BOTH arms, then graph capture "
                      f"(3 pre-capture iterations on a side stream), then {WARM_REPLAY} "
                      f"untimed replays of every arm",
            "interleaving": "one timed replay of each arm per sample round, so drift is common-mode",
            "distinct_weights": f"{LAYERS} independent weight tensors ({LAYERS * N * K * 2 / 1e6:.1f} MB)",
        },
        "clocks_before_any_gpu_work": clocks(),
        "shapes": {},
        "cold_brackets": {},
    }
    MS = tuple(int(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("1", "2", "4")))

    # Bring the card out of its 180 MHz idle state BEFORE any timing: the SM/mem
    # clocks ramp under load and a cold first arm would otherwise be penalised.
    ramp = torch.randn(8192, 8192, device=DEV, dtype=DT)
    for _ in range(300):
        ramp @ ramp
    torch.cuda.synchronize()
    del ramp
    torch.cuda.empty_cache()
    out["clocks_after_ramp_before_timing"] = clocks()

    for m in MS:
        out["shapes"][f"m{m}"] = run_shape(m)
        print(f"hot m={m} done", file=sys.stderr, flush=True)
    for m in MS:
        out["cold_brackets"][f"m{m}"] = run_flush(m)
        print(f"cold m={m} done", file=sys.stderr, flush=True)
    out["clocks_after"] = clocks()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
