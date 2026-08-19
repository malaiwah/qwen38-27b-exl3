# The card was capped at 400 W of 600 W — and it invalidated every prefill number

**Date:** 2026-08-19
**Trigger:** a spec comparison. Measured HBM bandwidth (1842 GB/s) came out *above*
the 1792 GB/s nameplate while measured fp16 (157 TFLOP/s at the time) came out at
75% of the 209.5 TFLOP/s dense spec. Two deviations in opposite directions is not
noise, it is a configuration problem.
**Outcome:** flagship prefill **7,694.9 → 9,638.9 tok/s (+25.3%)**. The trellis and
FP6 paths did not move. No conclusion about the north-star criteria changed — the
tradeoff got *wider*.

## Two independent sources, both persistent

1. **`/etc/lact/config.yaml`** — `power_cap: 400.0` under the GPU's PCI id, applied
   by the `lactd` daemon every `apply_settings_timer: 5` seconds.
2. **`~/.config/systemd/user/qwen38-27b.service`** — 
   `ExecStartPre=/usr/bin/sudo -n /usr/bin/nvidia-smi --id=GPU-506a575d... --power-limit=400`,
   re-applied on **every single service start**.

Removing only the first was not enough and produced a confusing intermediate state:
`nvidia-smi` showed 600 W right after the `lactd` restart, then silently reverted to
400 W the next time the unit started. The regression gate caught it — the throughput
profile measured 7,469.0 against a 9,157 threshold and **failed**, which is exactly
what that gate exists for. Credit for spotting the unit: the operator.

Both are now removed, each with an in-file comment recording what was taken out and
why, and backups at `/etc/lact/config.yaml.bak-2026-08-19` and
`~/.config/systemd/user/qwen38-27b.service.bak-2026-08-19`. Verified to survive a
`lactd` restart, a `daemon-reload`, and a full `systemctl --user start` cycle;
`compute-mode=EXCLUSIVE_PROCESS` was deliberately left in place.

## A methodology bug found on the way, which invalidated the first roofline

The first roofline (committed earlier the same day) is **wrong** and has been
corrected in place. Two flaws:

1. **Benchmarks too short to raise clocks.** An fp16 4096³ GEMM measured
   **68.4 TFLOP/s at 195 MHz** while 8192³ measured 232.8 at 2842 MHz — same kernel
   class, 3.4× apart, purely clock ramp. Fixed with a 2.5 s sustained warmup per case.
2. **Telemetry sampled after the timed loop**, by which point the GPU had dropped to
   idle (readings of 24 W were reported as if load power). Fixed with a background
   sampler polling during the measurement.

`tools/roofline-sustained.py` is the corrected harness.

## Measured, same harness, only the cap differing

| quantity | 400 W | 600 W | change | dense spec | 600 W vs spec |
|---|---|---|---|---|---|
| HBM bandwidth | 1840.3 GB/s | 1840.5 GB/s | **0.0%** | 1792 GB/s | **102.7%** |
| fp16 GEMM | 157.5 TFLOP/s | **214.5** | **+36.2%** | 209.5 | **102.4%** |
| bf16 GEMM | 164.0 | **217.6** | +32.7% | 209.5 | 103.9% |
| fp8 GEMM | 455.9 | **630.7** | +38.3% | 419 (2× fp16) | 150.5% |
| SM clock under GEMM | 1852 MHz | 2520 MHz | +36% | — | — |

**Bandwidth is not power-limited** — it drew only 411 W of the 600 W budget and did
not change. **Compute was throttled 36%.**

## Effect on the serving profiles (n=3 boots each)

| profile | PP @ 400 W | PP @ 600 W | change |
|---|---|---|---|
| `throughput` (all-FP4) | 7,694.9 ± 21.8 | **9,638.9 ± 18.3** | **+25.3%** |
| `balanced` (gate_up FP6) | 3,266.3 ± 13.9 | 3,250.6 ± 1.2 | −0.5% (tie) |
| `fidelity` (all-trellis) | 1,965.4 ± 2.1 | 1,965.9 ± 1.3 | +0.0% |
| all-FP6 (not shipped) | 4,648 | 4,742.4 | +2.0% |

Decode barely moved anywhere (fox 185.0 → 187.4 on `throughput`), consistent with
decode being latency-bound and bandwidth being unaffected by the cap.

**Only the FP4 path was power-limited.** Its GEMMs are dense tensor-core work that
converts watts into throughput; the trellis and FP6 paths are bound by something
else, which is why they were indifferent to a 50% larger power budget. The +25.3%
also matches the roofline decomposition: a 38% GEMM speedup applied to a 62%
GEMM / 38% other split predicts 9,285 tok/s, and 9,638.9 was measured.

## Does it change the conclusions?

**No — and it makes the central one stronger.** Criterion 1 (prefill ≥ 7000) is now
met with more room by `throughput` (9,639, i.e. 138% of target), but every
KLD-passing configuration gained nothing:

| | prefill @ 600 W | KLD | verdict |
|---|---|---|---|
| all-FP4 | **9,638.9** | 0.063759 | fails fidelity |
| all-FP6 | 4,742.4 | 0.010699 | passes fidelity, 32% short on speed, 99k ctx |
| all-trellis | 1,965.9 | 0.003437 | passes fidelity, 80% short on speed |

The gap between the fastest profile and the most faithful one **widened from 3.9× to
4.9×**. Removing the power cap bought throughput on the axis that was already
passing and nothing on the axis that was failing.

## Lesson

Benchmarks are only as trustworthy as the machine state behind them. A card can be
silently throttled by a vendor daemon *and* by a unit file, and neither shows up in
any application-level metric — only in a spec cross-check. The cheap tell was two
deviations in opposite directions: bandwidth above nameplate (a VRAM overclock, also
present here as `mem_clock_offsets: 6000`) and compute below it (the cap).
