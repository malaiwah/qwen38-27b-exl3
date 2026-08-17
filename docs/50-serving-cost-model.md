# Serving cost model — `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` for an oh-my-pi multi-subagent coding workload

**Denomination: GPU-hours per million *output* tokens (GPU-h/Mtok).** Multiply any cell by your
`$/GPU-hr` to get `$/Mtok`. §0–§10 carry no prices by design. **§11 adds the USD denomination** —
`$/Mtok` split input / output / cached input, plus the break-even utilisation — because that is the unit a
buyer compares against a commercial price list.

**Arithmetic identity used in every efficiency cell (no fitting, no modelling):**

```
GPU-h/Mtok  =  GPUs / (aggregate_output_tok_s × 3600) × 1e6
```

`aggregate_output_tok_s` is the ladder harness's `aggregate_tok_s`, which is **completion tokens ÷ cell
wall-clock**, verified against the raw repeat: `ab-ladder-A.json` 512x256 C1 repeat 0 has `wall_s 1.8843`,
`completion_tokens 256`, `aggregate_tok_s 135.863`, and 256/1.8843 = 135.859. **Prefill time is inside that
wall**, so prefill cost is already charged against the output-token denominator. This is the right
denominator for an agent workload where you pay for the box, not per token.

Medians are the harness's own `median_aggregate_tok_s` / `median_per_request_tok_s` across repeats (3 repeats
for the `tb21-ladder-*` arms, **5** for `ab-ladder-A/B`); recomputing medians from
`repeats[j].aggregate_tok_s` and `repeats[j].requests[k].per_request_tok_s` reproduced every published cell.

---

## 0. Receipt inventory and provenance

| receipt | tier / arm | host | base_url | repeats | rungs |
|---|---|---|---|---:|---|
| `receipts/ab-ladder-A.json` | **1 GPU, shipped baseline (arm A)** | `jl-vm-473501` (the DP8 host) | `127.0.0.1:8000` | 5 | 1–8 |
| `receipts/ab-ladder-B.json` | 1 GPU, b12x gate arm (arm B) | `jl-vm-473501` | `127.0.0.1:8000` | 5 | 1–8 |
| `receipts/tb21-ladder-8x-dp8.json` | 8 GPU, DP8 | `jl-vm-473501` | `10.0.0.5:8000` | 3 | 1–128 |
| `receipts/tb21-ladder-8x-tp2dp4.json` | 8 GPU, TP2×DP4 | `jl-vm-473501` | `10.0.0.5:8000` | 3 | 1–128 |
| `receipts/tb21-ladder-8x-tp4dp2.json` | 8 GPU, TP4×DP2 | `jl-vm-473501` | `127.0.0.1:8000` | 3 | 1–128 |
| `receipts/tb21-ladder-4x-{dp4,tp2dp2,tp4}.json` | 4 GPU | `127.0.0.1:8000` | `127.0.0.1:8000` | 3 | 1–64 |
| `receipts/tb21-ladder-2x-{dp2,tp2,tp2-p2p}.json` | 2 GPU | `10.0.0.4` | `10.0.0.4:8000` | 3 | 1–32 |
| `receipts/tb21-ladder-1x-hyd.json` | 1 GPU, older endpoint | `jl-vm-473319` (`tb21-1x-endpoint.json`) | `10.0.0.3:8000` | 3 | 1–64 |
| `receipts/tb21-8x-topology-ladder.json` | 8× topology aggregator | `jl-vm-473501` | — | — | — |
| `receipts/kernel-gap-bare-metal-ab.json` | arm A/B analysis, CVs, void cell | `jl-vm-473501` | — | 5 | 1–8 |
| `receipts/reasoning-effort-1x.json` | `reasoning_effort` probe | `jl-vm-473319` | `10.0.0.3:8000` | 2 | — |
| `receipts/apc-poison-repro.json` | prefix-cache economics (arm E) | 1× RTX 5090 | — | — | — |
| `receipts/tb21-8x-2xclock-arm.json` + `tb21-8x-p1-merged.json` | **real TB2.1 agentic workload on DP8 C32** | `jl-vm-473501` | — | 1 pass | — |

**The 1→8 ratio is legitimate here and only here.** `ab-ladder-A.json` ran a single GPU **on the DP8 host
`jl-vm-473501`**, bare docker, through the same `tb21_ladder.py` harness that produced the DP8 ladder. The
`10.0.0.3` (`jl-vm-473319`) 1× ladder is a **different host** and is carried below only as corroboration —
**never inside a ratio** (standing rule against cross-tier ratios).

**Two data-quality facts that constrain how hard you may lean on individual cells** (both from
`kernel-gap-bare-metal-ab.json`):

1. **Short-prompt cells are noisy.** Within-arm CV on the baseline repeats is **16.78 % at 512x256 C1,
   18.01 % at C2, 13.91 % at C4**, versus **0.76 % at 4kx1k C8, 1.78 % at 30kx2k C4, 0.49 % at 30kx2k C8**.
   So every 512x256 low-concurrency GPU-h/Mtok below carries a **±14–18 % error bar**; the **4kx1k C8 and
   30kx2k cells are the trustworthy ones**, and they are also the shapes that resemble a coding agent.
2. **`ab-ladder-B.json` 30kx2k C8 is VOID, not a measurement.** It reads `aggregate 0.0` with 24 refusals
   because the server was torn down mid-cell to reclaim the host. It is excluded from every number here
   (`invalidated_cell` in the analysis receipt; docs/46 §28).

**Every ladder number below is a strict prefix-cache-COLD floor.** The harness synthesises "seeded word
filler, unique per (shape,conc,repeat,slot) so prefix caching cannot inflate throughput", and the engine
counters prove it worked: the DP8 512x256 C128 cell records `prefix_cache_queries 240,956` and
**`prefix_cache_hits 0`** in `metric_deltas`. Real agent traffic with shared system prompts will do better
than these cells, never worse (§8).

---

## 1. Master efficiency table — GPU-h per Mtok of output, at knee and at peak

Knee rule (harness-fixed): *largest concurrency whose median per-request tok/s is ≥ 50 % of single-stream.*
Peak = the rung with the highest median aggregate **within that arm's measured rung range**.

**Why there is no TP8 row: TP8 is architecturally impossible for this model, not merely untested.** EXL3
shards trellis tensors only on 128-element boundaries, and `lm_head`'s output dim is the padded vocab
248320 = 128 × 1940 with 1940 = 2² × 5 × 97, so only TP ∈ {1, 2, 4} give 128-aligned slices. The engine
refuses at load with `ValueError: EXL3 TP output slice must be 128-aligned, got start=62080, size=31040`
(`exl3.py:2848 _slice_exl3_tensor`, via `_shard_tensors_for_tensor_parallel:2904`). **TP width is capped at
4 on any GPU count** — `tb21-8x-topology-ladder.json` → `tp8_refusal`, log at `receipts/tb21-tp8-refusal.log`.
Every wider host must therefore scale by adding data-parallel replicas, which is exactly what the knee law
(§3) rewards.

| arm | GPUs | DP | shape | knee C | agg tok/s | per-req tok/s | **GPU-h/Mtok @knee** | peak C | agg tok/s | per-req tok/s | **GPU-h/Mtok @peak** | receipt |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|---:|---|
| 1x arm A (baseline) | 1 | 1 | 512x256 | C4 | 330.7 | 107.8 | **0.840** | C8 | 464.0 | 62.6 | **0.599** | `ab-ladder-A.json` |
| 1x arm A (baseline) | 1 | 1 | 4kx1k | C8 | 393.0 | 49.9 | **0.707** | C8 | 393.0 | 49.9 | **0.707** | `ab-ladder-A.json` |
| 1x arm A (baseline) | 1 | 1 | 30kx2k | C4 | 166.9 | 43.8 | **1.665** | C8 | 188.0 | 23.8 | **1.478** | `ab-ladder-A.json` |
| 1x arm B (b12x gate) | 1 | 1 | 512x256 | C4 | 333.2 | 108.8 | **0.834** | C8 | 472.7 | 63.5 | **0.588** | `ab-ladder-B.json` |
| 1x arm B (b12x gate) | 1 | 1 | 4kx1k | C8 | 396.9 | 50.6 | **0.700** | C8 | 396.9 | 50.6 | **0.700** | `ab-ladder-B.json` |
| 1x arm B (b12x gate) | 1 | 1 | 30kx2k | C4 | 167.4 | 43.8 | **1.660** | C4 | 167.4 | 43.8 | **1.660** | `ab-ladder-B.json` |
| 2x DP2 | 2 | 2 | 512x256 | C8 | 616.4 | 108.8 | **0.901** | C32 | 1072.0 | 38.4 | **0.518** | `tb21-ladder-2x-dp2.json` |
| 2x DP2 | 2 | 2 | 4kx1k | C8 | 451.1 | 72.5 | **1.232** | C32 | 761.2 | 24.8 | **0.730** | `tb21-ladder-2x-dp2.json` |
| 2x TP2 (ForceP2P on) | 2 | 1 | 512x256 | C4 | 382.0 | 124.0 | **1.454** | C16 | 604.7 | 40.3 | **0.919** | `tb21-ladder-2x-tp2-p2p.json` |
| 2x TP2 (ForceP2P on) | 2 | 1 | 4kx1k | C4 | 322.7 | 83.1 | **1.722** | C16 | 496.0 | 31.5 | **1.120** | `tb21-ladder-2x-tp2-p2p.json` |
| 2x TP2 (P2P off) | 2 | 1 | 512x256 | C4 | 313.6 | 106.0 | **1.771** | C16 | 557.4 | 37.2 | **0.997** | `tb21-ladder-2x-tp2.json` |
| 2x TP2 (P2P off) | 2 | 1 | 4kx1k | C4 | 290.0 | 76.5 | **1.915** | C16 | 456.4 | 28.9 | **1.217** | `tb21-ladder-2x-tp2.json` |
| 4x DP4 | 4 | 4 | 512x256 | C16 | 1027.5 | 85.1 | **1.081** | C64 | 1958.3 | 38.3 | **0.567** | `tb21-ladder-4x-dp4.json` |
| 4x DP4 | 4 | 4 | 4kx1k | C16 | 858.9 | 72.6 | **1.294** | C64 | 1529.6 | 25.4 | **0.726** | `tb21-ladder-4x-dp4.json` |
| 4x TP2×DP2 | 4 | 2 | 512x256 | C8 | 695.2 | 120.8 | **1.598** | C64 | 1808.0 | 35.1 | **0.615** | `tb21-ladder-4x-tp2dp2.json` |
| 4x TP2×DP2 | 4 | 2 | 4kx1k | C8 | 612.8 | 82.8 | **1.813** | C64 | 946.0 | 19.1 | **1.174** | `tb21-ladder-4x-tp2dp2.json` |
| 4x TP4 | 4 | 1 | 512x256 | C4 | 369.6 | 113.9 | **3.006** | C64 | 1359.1 | 23.1 | **0.818** | `tb21-ladder-4x-tp4.json` |
| 4x TP4 | 4 | 1 | 4kx1k | C8 | 410.1 | 52.6 | **2.709** | C64 | 927.0 | 14.8 | **1.199** | `tb21-ladder-4x-tp4.json` |
| **8x DP8** | 8 | 8 | 512x256 | C32 | 1947.2 | 79.8 | **1.141** | C128 | 3552.6 | 34.5 | **0.626** | `tb21-ladder-8x-dp8.json` |
| **8x DP8** | 8 | 8 | 4kx1k | C32 | 1606.9 | 71.2 | **1.383** | C128 | 2782.2 | 25.4 | **0.799** | `tb21-ladder-8x-dp8.json` |
| 8x TP2×DP4 | 8 | 4 | 512x256 | C16 | 1074.6 | 98.3 | **2.068** | C128 | 2438.2 | 29.9 | **0.911** | `tb21-ladder-8x-tp2dp4.json` |
| 8x TP2×DP4 | 8 | 4 | 4kx1k | C16 | 915.5 | 78.6 | **2.427** | C128 | 1896.8 | 19.2 | **1.172** | `tb21-ladder-8x-tp2dp4.json` |
| 8x TP4×DP2 | 8 | 2 | 512x256 | C8 | 481.1 | 83.1 | **4.619** | C128 | 2205.6 | 24.7 | **1.008** | `tb21-ladder-8x-tp4dp2.json` |
| 8x TP4×DP2 | 8 | 2 | 4kx1k | C8 | 583.4 | 80.1 | **3.809** | C128 | 1874.1 | 15.3 | **1.186** | `tb21-ladder-8x-tp4dp2.json` |
| 8x TP4×DP2 | 8 | 2 | 30kx2k | C4 | 258.8 | 70.1 | **8.587** | C128 | 678.9 | 5.7 | **3.273** | `tb21-ladder-8x-tp4dp2.json` |
| *corroboration only (different host)* | | | | | | | | | | | | |
| 1x `jl-vm-473319` | 1 | 1 | 512x256 | C8 | 414.0 | 66.0 | *0.671* | C64 | 828.6 | 15.2 | *0.335* | `tb21-ladder-1x-hyd.json` |
| 1x `jl-vm-473319` | 1 | 1 | 4kx1k | C4 | 279.6 | 71.8 | *0.993* | C64 | 584.3 | 9.4 | *0.475* | `tb21-ladder-1x-hyd.json` |
| 1x `jl-vm-473319` | 1 | 1 | 30kx2k | C4 | 160.4 | 43.8 | *1.731* | C16 | 195.4 | 12.8 | *1.422* | `tb21-ladder-1x-hyd.json` |

Caveats attached to specific cells:
- **arm A / arm B 4kx1k knee is `knee_at_cap: true`** — the knee landed on the top measured rung (C8), so the
  true knee is ≥ C8 and the 0.707 / 0.700 figures are an **upper bound on cost** (the real knee, being at
  higher concurrency, would be cheaper still).
- arm A vs arm B reproduce each other to **0.7 % or better** on all three shapes (0.840/0.834, 0.707/0.700,
  1.665/1.660), which is the best available check on the single-GPU row.
- 512x256 cells at C1–C4 inherit the 13.9–18.0 % CV. Do not quote them to three digits in a decision.

### What the table says about cost, before any price is applied

1. **Cheapest output tokens come from ONE GPU, not eight.** At the knee, one GPU is **0.840** (512x256) /
   **0.707** (4kx1k) / **1.665** (30kx2k) GPU-h/Mtok. DP8 at its knee is **1.141 / 1.383** — i.e. **+35.9 %
   and +95.7 % more expensive per output token**. Multi-GPU is a *latency and concurrency* purchase, never a
   cost-per-token purchase.
2. **TP width is a pure cost multiplier at the knee.** On the same 8 GPUs: DP8 1.141 → TP2×DP4 2.068
   (+81 %) → TP4×DP2 4.619 (**4.05× DP8**). At peak the spread collapses to 1.61× (0.626 → 1.008) because
   over-knee batching hides the TP tax. Choose TP only to buy single-stream latency, and only at the prompt
   lengths where the topology ladder shows it wins (512x256 C1: TP2×DP4 153.2 > TP4×DP2 146.5 > DP8 135.2;
   4kx1k C1 inverts to TP4×DP2 109.3 > TP2×DP4 105.8 > DP8 90.1 — `tb21-8x-topology-ladder.json` findings
   T2/T3).
3. **Long prompts are the expensive axis, not concurrency.** 30kx2k on one GPU is 1.665 GPU-h/Mtok versus
   0.707 at 4kx1k — **2.36× the cost per output token for the same box**, because prefill dominates the wall.
   On TP4×DP2 the same shape costs 8.587. Every 30k-token subagent prompt you can shrink to 4k is a 2.4×
   cost cut on that call.

---

## 2. Same-host 1 GPU → DP8 scaling, and the per-GPU efficiency delta

Both sides are `jl-vm-473501` through the same `tb21_ladder.py`; `ab-ladder-A.json` (1 GPU, 5 repeats) vs
`tb21-ladder-8x-dp8.json` (8 GPUs, 3 repeats).

| shape | 1 GPU @knee | DP8 @knee | scaling factor | per-GPU efficiency | GPU-h/Mtok change |
|---|---|---|---:|---:|---:|
| 512x256 | C4, 330.7 tok/s | C32, 1947.2 tok/s | **5.889×** on 8 GPUs | **0.736** (**−26.4 %**) | 0.840 → 1.141 (**+35.9 %**) |
| 4kx1k | C8, 393.0 tok/s | C32, 1606.9 tok/s | **4.088×** on 8 GPUs | **0.511** (**−48.9 %**) | 0.707 → 1.383 (**+95.7 %**) |
| 30kx2k | C4, 166.9 tok/s | *not measured on DP8* | — | — | — |

**Not extrapolated:** 30kx2k was only ever run on the TP4×DP2 arm (`shape_coverage_note` in
`tb21-8x-topology-ladder.json`), so there is no 1→DP8 scaling number at 30kx2k and none is invented here.
The 4kx1k row is additionally an **upper bound on the scaling factor**: the 1-GPU knee was at-cap, so the
1-GPU denominator can only grow, which can only push 4.088× down. The honest statement is **per-GPU
efficiency at the knee falls by at least 26 % (512x256) and at least 49 % (4kx1k) going 1 → DP8.**

Matched-rung comparisons on the same two receipts, which isolate *why*:

| rung | 512x256: 1 GPU | 512x256: DP8 | agg ratio | per-req ratio | GPU-h/Mtok ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 136.0 tok/s | 135.2 tok/s | **0.994×** | 0.994× | **8.05× worse** (2.043 → 16.438) |
| C8 | 464.0 tok/s | 692.5 tok/s | 1.493× | **1.752×** | **5.36× worse** (0.599 → 3.209) |
| 4kx1k C1 | 92.0 | 90.1 | 0.980× | 0.980× | **8.16× worse** (3.020 → 24.654) |
| 4kx1k C8 | 393.0 | 678.8 | 1.727× | **1.869×** | **4.63× worse** (0.707 → 3.274) |

**At C1 DP8 is exactly 8× the price for 0.99× the speed** — one request lands on one replica, and the other
seven GPUs are billed idle. This is the single most expensive mistake available: renting 8 GPUs for a
low-concurrency agent loop. DP8 only starts earning its rent above C8, and only fully at C32 (§3).

**Peak is not comparable same-host and is not claimed.** Arm A's ladder stops at C8, so 3552.6/464.0 =
7.66× (per-GPU 0.957) compares DP8's C128 against a 1-GPU ladder truncated 4 rungs early. It is an **upper
bound only**. `[INFERENCE]` The 1× ladder on the other host kept climbing to 828.6 tok/s at C64, so a
same-host 1→8 peak ratio near **4.3×** (per-GPU ≈ 0.54) is the more likely truth — cross-host, therefore
inference, therefore not used anywhere else in this document.

---

## 3. Sizing for ~32 concurrent subagents

**The measured law:** knee concurrency = **4 × data-parallel degree**, verified independently at two shapes
across three 8-GPU arms — DP8→C32, TP2×DP4→C16, TP4×DP2→C8 (`tb21-8x-topology-ladder.json` finding T4) — and
consistent with DP2→C8 and DP4→C16 on the 2× and 4× receipts. TP width inside a replica does **not** raise
the knee.

**Therefore ~32 concurrent subagents at the per-request floor requires DP = 8**, and since each replica needs
its own GPU (21 GB of weights on a 96 GB card, so TP is never needed for capacity), **DP8 = 8 GPUs = the
whole `jl-vm-473501`-class host, TP1.** `[MEASURED law → arithmetic]`

### Every measured C32 cell, ranked by cost

| arm | GPUs | DP | shape | C32 agg tok/s | C32 per-req tok/s | % of single-stream | **GPU-h/Mtok** |
|---|---:|---:|---|---:|---:|---:|---:|
| 1x `jl-vm-473319` | 1 | 1 | 512x256 | 664.6 | 23.7 | 18.3 % | **0.418** |
| 2x DP2 | 2 | 2 | 512x256 | 1072.0 | 38.4 | 28.3 % | **0.518** |
| 4x DP4 | 4 | 4 | 512x256 | 1455.9 | 57.9 | 43.2 % | **0.763** |
| 4x TP2×DP2 | 4 | 2 | 512x256 | 1132.2 | 42.2 | 27.0 % | **0.981** |
| **8x DP8** | 8 | 8 | 512x256 | 1947.2 | **79.8** | **59.0 %** | **1.141** |
| 8x TP2×DP4 | 8 | 4 | 512x256 | 1622.4 | 72.4 | 47.3 % | **1.370** |
| 4x TP4 | 4 | 1 | 512x256 | 628.9 | 20.7 | 13.7 % | **1.767** |
| 8x TP4×DP2 | 8 | 2 | 512x256 | 1018.9 | 37.5 | 25.6 % | **2.181** |
| 1x `jl-vm-473319` | 1 | 1 | 4kx1k | 455.3 | 14.5 | 15.2 % | **0.610** |
| 2x DP2 | 2 | 2 | 4kx1k | 761.2 | 24.8 | 27.3 % | **0.730** |
| 4x DP4 | 4 | 4 | 4kx1k | 1156.1 | 44.4 | 48.6 % | **0.961** |
| **8x DP8** | 8 | 8 | 4kx1k | 1606.9 | **71.2** | **79.0 %** | **1.383** |
| 8x TP2×DP4 | 8 | 4 | 4kx1k | 1277.1 | 48.7 | 46.0 % | **1.740** |
| 8x TP4×DP2 | 8 | 2 | 4kx1k | 810.6 | 26.3 | 24.1 % | **2.741** |
| 1x `jl-vm-473319` | 1 | 1 | 30kx2k | 185.1 | 6.3 | 8.1 % | **1.501** |
| 8x TP4×DP2 | 8 | 2 | 30kx2k | 479.6 | 16.0 | 16.3 % | **4.633** |

Read that table as the actual decision. **DP8 is the only measured configuration that holds a subagent above
the 50 % floor at C32** (59.0 % at 512x256, 79.0 % at 4kx1k). DP4 at C32 delivers 43.2 % — visibly over its
own C16 knee, exactly as the law predicts. So the *quality* of a 32-subagent fan-out is a step function in DP
degree, and 8 is the step you need.

`[INFERENCE — cross-host ratio, flagged]` Against the 1-GPU C32 cell (different host, same GPU model), DP8
buys **3.37× the per-request speed (79.8 vs 23.7 tok/s) for 2.73× the GPU-h/Mtok (1.141 vs 0.418)**. That is
super-proportional and is the one regime where multi-GPU is a genuinely good buy. It is cross-host, so treat
the ratio as inference; both cells individually are measured.

### The invariant worth remembering: **GPUs per concurrent subagent at the floor = TP / 4**

Since knee = 4·DP and GPUs = TP·DP, GPUs/knee = TP/4, exactly. Checked against every arm:

| arm | TP | predicted GPU/subagent | measured knee | measured GPU/subagent |
|---|---:|---:|---|---:|
| 1x arm A (512x256), 2x DP2, 4x DP4, 8x DP8 | 1 | **0.25** | C4 / C8 / C16 / C32 | 0.25 / 0.25 / 0.25 / 0.25 |
| 2x TP2, 4x TP2×DP2, 8x TP2×DP4 | 2 | **0.50** | C4 / C8 / C16 | 0.50 / 0.50 / 0.50 |
| 4x TP4 (512x256), 8x TP4×DP2 (512x256) | 4 | **1.00** | C4 / C8 | 1.00 / 1.00 |

Holds exactly in all eight arms with DP ≥ 2. The two DP=1 arms land one rung high at 4kx1k (arm A C8 —
flagged `knee_at_cap`; 4x TP4 C8), which makes them cheaper than the law predicts, not more expensive.
**Consequence: DP topology does not change what a concurrent subagent costs in GPUs — 0.25 GPU each. TP
width multiplies that by the TP width, buying nothing at C32.**

### The measured cost of oversubscription (over-knee penalty)

DP8, `tb21-ladder-8x-dp8.json`, full rung walk with cost attached:

| C | 512x256 agg | 512x256 per-req | GPU-h/Mtok | 4kx1k agg | 4kx1k per-req | GPU-h/Mtok |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 135.2 | 135.2 | 16.438 | 90.1 | 90.2 | 24.654 |
| 8 | 692.5 | 109.7 | 3.209 | 678.8 | 93.4 | 3.274 |
| 16 | 1307.1 | 117.8 | 1.700 | 1239.4 | 86.4 | 1.793 |
| **32 (knee)** | **1947.2** | **79.8** | **1.141** | **1606.9** | **71.2** | **1.383** |
| 64 | 2835.5 | 67.0 | 0.784 | 2214.3 | 44.7 | 1.004 |
| **128** | 3552.6 | **34.5** | 0.626 | 2782.2 | **25.4** | 0.799 |

**C32 → C128 on DP8: per-request throughput collapses 79.8 → 34.5 tok/s (−56.8 %) and 71.2 → 25.4 (−64.3 %),
while aggregate rises only 1.82× / 1.73× for 4× the concurrency.** GPU-h/Mtok does improve (−45.2 % / −42.2 %),
so oversubscription is *cheaper per token and much worse per agent*. For an interactive agent loop that is
the wrong trade: at C128 every subagent decodes at a quarter of single-stream speed, wall-clock per task
roughly doubles-to-triples, and any per-task timeout starts firing (§4 shows timeouts were already 22/89 of
TB2.1's failures). **Queue above C32 rather than admitting above C32.** Zero refusals were observed C1–C128
in every arm, so the engine will happily accept the load that ruins your latency — the admission limit has
to come from you.

---

## 4. The reality tax: what a real agentic workload actually costs on this hardware

Same host, same image (`vllm:gg-r34-tb21-sr1 sha256:237a5025`), same topology, same concurrency (DP8, C32) as
the ladder — so this is a legitimate like-for-like against the DP8 rows.
Receipts: `tb21-8x-2xclock-arm.json` + `tb21-8x-p1-merged.json` (89 rows summed).

| quantity | value | source |
|---|---:|---|
| wall clock (8 shards in parallel) | 8454.1 s = 2.3484 h | `aggregate.wall_clock_s` |
| GPU-hours consumed | **18.787** | 8 × 2.3484, arithmetic |
| output tokens | 6,670,376 | Σ `rows[].n_output_tokens` |
| input tokens | 66,329,299 | Σ `rows[].n_input_tokens` |
| input : output ratio | **9.94 : 1** | arithmetic |
| **GPU-h/Mtok (output)** | **2.816** | arithmetic |
| GPU-h per task (89 tasks) | **0.2111** | arithmetic |
| GPU-h per *resolved* task (56) | **0.3355** | arithmetic |
| host output throughput | 789.0 tok/s (98.6 tok/s/GPU) | arithmetic |
| as a fraction of the DP8 512x256 knee | **40.5 %** | 789.0 / 1947.2 |
| straggler tax (slowest shard / mean shard) | 1.177 (8448.6 s vs 7177.5 s mean) | `per_shard[].wall_s` |

**A real coding-agent workload costs 2.47× what the microbenchmark says** (2.816 vs 1.141 GPU-h/Mtok at the
DP8 512x256 knee). Two measured reasons, both structural: the input:output ratio is **~10:1**, so most GPU
time is prefill on code context rather than decode; and agent loops idle on tool execution and verification,
so the host only sustains 40.5 % of its ladder throughput. **Use 2.8 GPU-h/Mtok, not 1.1, for budgeting
oh-my-pi at xhigh.** The 1.18× straggler tax also says: uneven shard sizing wastes ~15 % of a fan-out's
GPU-hours, so balance subagent work or over-shard.

---

## 5. The reasoning-effort lever — the largest single knob, quantified

`receipts/reasoning-effort-1x.json` (1 GPU, 2 repeats, 32,768-token budget so nothing is cap-truncated —
`truncated_by_cap: 0` in every arm; both repeats returned the correct Manacher implementation in every valid
arm).

| `reasoning_effort` | median completion tokens | median latency | ×tokens vs medium | ×wall vs medium |
|---|---:|---:|---:|---:|
| *(unset)* | 18,085 | 168.9 s | 5.263 | 6.447 |
| `xhigh` (documented default) | 18,085 | 168.8 s | **5.263** | **6.443** |
| `medium` | 3,436 | 26.2 s | 1.000 | 1.000 |
| `low` | 3,517 | 27.0 s | 1.024 | **1.031** |
| `none` | — | — | HTTP 400 ×2, unusable | — |

**The $/task multiplier for `xhigh` over `medium` is 5.26× if you are billed by token share and 6.44× if you
are billed by wall-clock occupancy.** On rented GPUs you are billed by occupancy, so **use 6.44×.**
Unset ≡ xhigh (byte-identical token and latency medians), so **the default is the expensive mode** and
silence costs 6.44×. **`low` is 2.4 % more tokens and 3.1 % more wall than `medium`** — there is no cheaper
setting below medium; medium is the floor.

### Where it dominates topology entirely

| decision | measured cost spread |
|---|---:|
| `reasoning_effort` xhigh → medium | **6.44×** (wall) / 5.26× (tokens) |
| worst → best 8-GPU topology at the knee (TP4×DP2 4.619 → DP8 1.141) | 4.05× |
| worst → best tier at C32, 512x256 (TP4×DP2 2.181 → 1 GPU 0.418) | 5.22× |
| 1 GPU → DP8 penalty at the knee, 512x256 | 1.36× |
| shape: 30kx2k → 4kx1k on one GPU | 2.36× |
| C32 → C128 over-knee discount on DP8 | 0.55× |

**`reasoning_effort` is the largest lever in this table — bigger than every hardware decision available,
including the worst-to-best tier spread at C32 (6.44× vs 5.22×) — and it is free.** Concretely: `medium` on
**one** GPU is cheaper per delivered task than
`xhigh` on **eight** — 6.44× effort saving against a 1.36× DP8 knee penalty is a **4.7× net win for the
single GPU**, and that is before the 8× idle-billing penalty at low concurrency (§2). **Set the effort knob
before you pick the box.** Any deliberation about DP8 vs TP2×DP4 while shipping unset/xhigh is optimising the
smaller term.

Applied to the measured TB2.1 arm: `[INFERENCE]` at 0.2111 GPU-h/task at xhigh, medium would land near
**0.0328 GPU-h/task** (wall-bound, ÷6.44) to **0.0401** (token-bound, ÷5.26) — an **81–84 % cost cut per
task**. This is inference for two reasons that must not be dropped: the 5.26×/6.44× ratios come from **one
prompt with 2 repeats on one GPU**, and TB2.1's own failure histogram (`AgentTimeoutError: 22`) means changing
effort changes *which tasks pass*, which nobody has measured. What is measured is that at xhigh, even with
the clock doubled, 22 of 89 tasks still died on the deadline — and medium's 6.44× wall reduction is a larger
budget intervention than the 2× clock arm that moved 31/89 → 56/89 in the first place.

---

## 6. Context-window cost — how context choice caps subagents per GPU

Hybrid attention is why this is cheap at all: **only 16 of 64 layers carry KV** (48 are linear/GDN with O(1)
state). At fp8 that is `2 × 4 kv-heads × 256 head_dim × 1 B × 16 layers` = **32 KiB/token**.
Engine-confirmed pool: **1,879,687 KV tokens per GPU (62.45 GiB)** `[MEASURED — engine startup banner,
parent-supplied; no committed receipt in this repo carries the string, so it is cited as an engine fact, not
a receipt line]`.

```
concurrent full-length sequences per GPU  =  1,879,687 / context_length
```

| context | GiB per sequence (32 KiB/tok) | **sequences per GPU** | sequences per 8-GPU DP8 host |
|---|---:|---:|---:|
| 32,768 | 1.00 | **57.4** | 458.9 |
| 131,072 | 4.00 | **14.3** | 114.7 |
| 262,144 (native) | 8.00 | **7.17** | 57.4 |
| 1,000,000 (static YaRN ×4) | 30.52 | **1.88** | 15.0 |

**Identity cross-checks.** The engine computes exactly this ratio itself: the 5090 qualification banner
reports `GPU KV cache size: 265,072 tokens` and `Maximum concurrency for 262,144 tokens per request: 1.01x`,
and 265,072/262,144 = 1.011 (`qualification-5090-apc.json` gate 1). The vLLM 1M recipe's "6.6M KV tokens at
1M context" on a 288 GB GB300 implies the same 32 KiB/token. **One reconciliation to state honestly:**
1,879,687 × 32 KiB = 57.36 GiB, which is 8.9 % below the quoted 62.45 GiB pool. The gap is the padding and
non-KV residents of the same pool — the engine logs `Add 3 padding layers, may waste at most 6.25 % KV cache
memory` plus mamba-page padding, and the 48 GDN layers' recurrent state lives in that allocation too. **The
token count is the binding number and is what the table uses**; the GiB figure is the allocation, not the
usable token capacity.

### What this means for subagent count

- **At 262k native, KV is not your constraint.** DP8 holds 57 full-length 262k sequences; you need 32. At
  C32 on DP8 each subagent could hold **469,921 tokens** of context before the pool binds
  (15,037,496 / 32). Throughput and per-request latency bind long before memory does.
- **Even one GPU holds 32 subagents** as long as average context stays under **58,740 tokens**
  (1,879,687/32) — comfortably above the 4k–30k coding-prompt band. So "how many subagents fit" is a
  *throughput* question on this model, not a memory question. That is the direct consequence of only 16 of 64
  layers carrying KV.
- **1M context and 32 subagents are mutually exclusive on an 8-GPU host.** 1.88 sequences per GPU means
  **15 concurrent 1M sequences on the entire host** — under half the 32 target — and a single 1M sequence
  occupies 53 % of one GPU's pool (30.52 of 57.36 GiB usable). Combined with Qwen's own warning that static
  YaRN "potentially impact[s] performance on shorter texts", **enable the 1M window on a separate endpoint
  only when a task needs it**; never as the fleet default for a fan-out workload.
- Cost consequence of the shape axis (§1): the 30kx2k row costs 2.36× the 4kx1k row per output token on the
  same single GPU. Context length costs you twice — pool residency *and* prefill wall.
- **Weight dtype moves this table too**, and by more than most people expect. See §7: BF16 weights leave
  only ~0.93 M KV tokens per GPU against our 1,879,687, so BF16 roughly **halves** every row above.

---

## 7. Quantisation: BF16 vs FP8 vs EXL3 K5/K6 at the same topology

This is the "why quantise at all" question, and it has **two independent cost channels** that must not be
conflated: decode bandwidth (§7.2) and KV capacity (§7.3). Both are driven by the same measured input.

### 7.1 The measured input: weight bytes on disk

| checkpoint | weights | × our bytes | source |
|---|---:|---:|---|
| `Qwen/Qwen3.8-27B` (BF16) | **51.77 GiB** | 2.5718× | MEASURED, on disk |
| `Qwen/Qwen3.8-27B-FP8` | **28.77 GiB** | 1.4292× | MEASURED, on disk |
| `malaiwah/…-EXL3-K5K6-hydrated` (ours) | **20.13 GiB** | 1.000× | MEASURED, on disk |

**Everything downstream of this table in §7 is `[INFERENCE]`.** There is **no measured BF16 or FP8 serving
throughput in this project.** The three receipts that carry `bf16` in their names —
`decode-parity-bf16-eager.json`, `decode-parity-bf16-graph.json`, `decode-parity-bf16-eager-vs-graph.json` —
are **greedy-determinism and logprob-parity probes** (32 prompts × 32 tokens, `exact_sequence_matches 24`,
`chosen_logprob_abs_delta_mean 0.0128`); they contain **no throughput field at all**, and a repo-wide search
found no BF16/FP8 tok/s measurement anywhere. So no BF16 or FP8 number in this section may be placed in the
same column as a measured EXL3 number without its `[INFERENCE]` marker, and none is.

### 7.2 Channel 1 — decode bandwidth `[INFERENCE, derived]`

Decode is bandwidth-bound, so the per-step **weight-byte numerator** is the cost. docs/47 (F6) decomposes
ours from the profiled decode, and the parts sum exactly:

| component | per-step bytes (ours) | scales with dtype? |
|---|---:|---|
| target-pass body (one verify GEMM serves all C requests) | 15.90 GB | **yes** — scaled by the 2.5718× / 1.4292× checkpoint ratio |
| `lm_head` 5120×248320, streamed **4× per step** (target verify + one per MTP draft depth) | 0.9535 × 4 = 3.81 GB | **yes, and exactly** — see below |
| draft-pass body, 3× per step | 0.27 × 3 = 0.81 GB | **yes** — same checkpoint ratio |
| **total** | **20.52 GB/step** (docs/47's 20.5) | |

**The head scales exactly, not by assumption.** 0.9535 GB over 5120 × 248320 elements is
**5.9997 bits/weight**, i.e. exactly K6. So BF16 `lm_head` = 5120 × 248320 × 2 B = **2.5428 GB** and FP8 =
**1.2714 GB**, computed from dimensions, not scaled. **What I scaled** by the checkpoint-wide byte ratio:
the body and the draft body. **What I held fixed:** the number of streams per step (4 head, 3 draft — an
MTP-schedule property, not a dtype property), the achievable bandwidth, and the accepted-tokens-per-step rate.

| dtype | body | head ×4 | draft ×3 | **numerator** | × ours | streaming floor @1494 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| EXL3 (ours) | 15.90 | 3.81 | 0.81 | **20.52 GB** | 1.000× | 13.74 ms/step |
| FP8 | 22.72 | 5.09 | 1.16 | **28.97 GB** | 1.411× | 19.39 ms |
| BF16 | 40.89 | 10.17 | 2.08 | **53.15 GB** | 2.589× | 35.57 ms |

Bandwidth denominator is docs/47's **measured** achievable ceiling, **1462–1525 GB/s** (`copy_` 1462.6,
read-only `sum` 1524.9), midpoint 1494 — *never* the 1.792 TB/s spec, which no kernel on this card reaches.

Anchoring on our own measured operating point (135.8 tok/s single-stream at ~2.4 accepted/step, docs/47):
step = 17.67 ms, of which 13.74 ms is the streaming floor → **77.7 % streaming efficiency** and **3.94 ms of
non-streaming overhead**, which reproduces docs/47's "~78 % at the midpoint pairing". Two bracketing models,
because the answer depends on what you assume about that 3.94 ms:

| model | assumption | FP8 | BF16 |
|---|---|---:|---:|
| A — everything scales | whole step scales with the numerator | 96.2 tok/s (**0.709×**) | 52.4 tok/s (**0.386×**) |
| B — fixed overhead | 3.94 ms of non-streaming work held constant | 102.9 tok/s (**0.758×**) | 60.7 tok/s (**0.447×**) |

**Decode bracket: FP8 is 1.32–1.41× slower than ours, BF16 is 2.24–2.59× slower.** Model B is the more
physical of the two and is the optimistic end for BF16.

### 7.3 Channel 2 — KV capacity, the one people forget `[INFERENCE, derived]`

At `--gpu-memory-utilization 0.92` on a 96 GB card the engine budget is **88.3 GiB**. Ours spends 20.13 on
weights and the engine confirms **62.45 GiB of KV**, so the non-weight non-KV residents (peak activation,
non-torch, CUDA-graph) come to **88.3 − 20.13 − 62.45 = 5.72 GiB**. Hold that overhead fixed — it is a
function of hidden size, batch and graph capture, not of weight dtype — and swap the weights:

| dtype | weights | KV left | KV tokens | × ours |
|---|---:|---:|---:|---:|
| EXL3 (ours) | 20.13 GiB | **62.45 GiB** *(engine-confirmed)* | **1,879,687** | 1.000× |
| FP8 | 28.77 | 53.81 | ~1,619,600 | 0.862× |
| BF16 | 51.77 | **30.81** | **~927,400** | **0.493×** |

**BF16 halves your concurrent-sequence capacity per GPU on top of being 2.2–2.6× slower.** Folded into the
§6 table:

| context | EXL3 seqs/GPU | FP8 seqs/GPU `[INF]` | BF16 seqs/GPU `[INF]` |
|---|---:|---:|---:|
| 32,768 | **57.4** | 49.4 | 28.3 |
| 131,072 | **14.3** | 12.4 | 7.08 |
| 262,144 (native) | **7.17** | 6.18 | **3.54** |
| 1,000,000 (static YaRN) | **1.88** | 1.62 | **0.93 — cannot hold even ONE** |

Two decision-grade consequences. **BF16 cannot serve a single 1M-token request on one 96 GB card at 0.92
utilisation** (0.93 sequences), so the 1M window becomes a TP2-minimum configuration for BF16 while it is a
single-card configuration for us. And at 262k, BF16's 3.54 sequences per GPU means a **DP8 host holds 28
full-length sequences, below the 32-subagent target**, where ours holds 57.

### 7.4 Bottom line — separable from the topology decision

Scale the measured DP8-knee aggregate by the §7.2 decode bracket. `tok/s per GPU` is the price-free core:
divide it by your `$/GPU-hr` to get **tok/s per dollar-per-hour**; and **tokens per dollar = 1e6 ÷
(GPU-h/Mtok × $/GPU-hr)**.

| dtype | shape | agg tok/s @DP8 knee C32 | tok/s per GPU | **GPU-h/Mtok** | × ours | tok per $ at $1/GPU-hr |
|---|---|---:|---:|---:|---:|---:|
| **EXL3 (MEASURED)** | 512x256 | **1947.2** | **243.4** | **1.141** | 1.00× | **876,400** |
| FP8 `[INFERENCE]` | 512x256 | 1381–1476 | 172.6–184.5 | 1.506–1.610 | 1.32–1.41× | 621,000–664,000 |
| BF16 `[INFERENCE]` | 512x256 | 752–870 | 94.0–108.8 | 2.553–2.957 | 2.24–2.59× | 338,000–392,000 |
| **EXL3 (MEASURED)** | 4kx1k | **1606.9** | **200.9** | **1.383** | 1.00× | **723,000** |
| FP8 `[INFERENCE]` | 4kx1k | 1139–1218 | 142.4–152.3 | 1.824–1.951 | 1.32–1.41× | 513,000–548,000 |
| BF16 `[INFERENCE]` | 4kx1k | 620–718 | 77.5–89.8 | 3.094–3.583 | 2.24–2.59× | 279,000–323,000 |

Anchored on the **real** agentic workload instead of the microbenchmark (§4's measured 2.816 GPU-h/Mtok at
xhigh on DP8 C32): FP8 `[INFERENCE]` **3.72–3.97** and BF16 `[INFERENCE]` **6.30–7.30** GPU-h/Mtok.

**The quantisation decision and the topology decision are separable and both are smaller than the effort
knob.** Ranked spreads, all from this document: `reasoning_effort` **6.44×** > BF16→EXL3 **2.24–2.59×**
(plus 2.03× the KV capacity) > worst-to-best 8-GPU topology at the knee **4.05×** at the pathological end but
**1.81×** for the realistic DP8-vs-TP2×DP4 choice > FP8→EXL3 **1.32–1.41×**. So: quantise (BF16 costs you
~2.4× per token *and* half your concurrency), prefer EXL3 over FP8 (~1.36× per token and +16 % KV), pick DP8
for a 32-way fan-out — and set `reasoning_effort: medium` before any of it.

**Two honesty limits on this section.** (1) The throughput scaling assumes decode-bound behaviour, so it
**overstates** the BF16 penalty at prefill-heavy shapes like 30kx2k, where FLOPs rather than weight bytes
dominate and EXL3 additionally pays dequantisation cost. (2) Capability is not in scope here and is not
traded away silently: `tb21-8x-2xclock-arm.json` withdrew all seven previously "quantization-suspect" TB2.1
tasks — they resolve **on the quantised model** once the clock is doubled — so **zero tasks in the suite
remain attributable to quantisation**. The cheap way to convert all of §7 from inference to measurement is one
`tb21_ladder.py` run per dtype at DP8; nothing else about the harness would change.

---

## 8. Prefix caching — what is measured, and what is not

**Measured (`receipts/apc-poison-repro.json`, arm E; owner-authoritative, cross-referenced by
`qualification-5090-apc.json` → `reuse_win`).** One physical RTX 5090, the K5K6-**context** edition,
`--max-num-seqs 1`, two documents with **disjoint** prefixes so the cold case is genuinely cold:

| prefix | cold TTFT | warm TTFT | speed-up | prompt tokens recomputed warm | warm hit rate |
|---|---:|---:|---:|---:|---:|
| 32,842 tok | 12.071 s | 1.044 s | **11.56×** | 2,442 of 32,842 | **92.56 %** |
| 131,146 tok | 67.597 s | 2.309 s | **29.27×** | 3,146 of 131,146 | **97.60 %** |

**Whole-schedule wall clock, 38 requests, one variable (APC on/off):**

| arm | condition | wall | speed-up |
|---|---|---:|---:|
| A | prefix caching ON, `max-num-seqs 1` | **84.03 s** | — |
| B | prefix caching OFF, `max-num-seqs 1` | 144.37 s | **1.718×** (−41.8 % wall) |
| A2 | prefix caching ON, `max-num-seqs 4` | **51.24 s** | — |
| B2 | prefix caching OFF, `max-num-seqs 4` | 111.85 s | **2.183×** (−54.2 % wall) |

Correctness alongside: 38 requests in arm C with a live 62.9 % hit rate in arm A, zero corrupted responses;
266 scored requests across seven arms with no corruption. Cost: on a 32 GiB card, APC costs either a 0.0005
utilisation bump or 1,344 tokens of window (262,144 → 260,800, 0.51 %) — irrelevant on a 96 GB card.

**The ladder measurements in §1–§3 contain ZERO prefix-cache benefit, by construction.** Prompt synthesis is
"unique per (shape,conc,repeat,slot) so prefix caching cannot inflate throughput", and the counters confirm
it: DP8 512x256 C128 shows `prefix_cache_queries 240,956` / **`prefix_cache_hits 0`**. Every GPU-h/Mtok in
this document is therefore a **cold-cache upper bound on cost**.

**Not measured — say so plainly.** No receipt quantifies APC benefit for *shared system prompts and tool
schemas across concurrent subagents* on this model at this scale. The TB2.1 arm, which is the only real
agentic workload here, reports `n_cache_tokens: 0` on all 89 rows (agent-side accounting; the harness does
not read server-side APC counters) and the word "prefix" appears nowhere in `tb21-8x-p1-merged.json` or
`tb21-8x-2xclock-arm.json`. Arm E is single-GPU, `max-num-seqs` 1 and 4, on a document-reuse pattern, on the
*context* edition — not 32 concurrent subagents on the *hydrated* edition.

`[INFERENCE]` The workload's shape argues APC is high-value: a 10:1 input:output ratio (§4) means prefill is
the dominant cost, and oh-my-pi subagents share a system prompt and tool-schema prefix. Arm E's mechanism
(92.6–97.6 % of a long prefix skipped) applies directly to that. The measured **1.72×–2.18× whole-schedule
wall reduction** is the best available anchor for the size of the effect, and A2/B2 shows it **grew** with
concurrency (1.718× at seqs 1 → 2.183× at seqs 4), which is the right direction for a 32-way fan-out. But
the number for our topology is unmeasured, and the honest way to close it is a DP8 ladder run with a **shared
prefix** instead of unique filler — one arm, and it would turn the largest remaining inference in this
document into a measurement. APC is already in the settled knob set (`--enable-prefix-caching`), so nothing
needs changing to collect it.

---

## 9. MEASURED vs INFERENCE — explicit split

### MEASURED (receipt-traceable, named above)
- Every aggregate and per-request tok/s cell in §1, §3 and the DP8 rung walk — 12 ladder receipts.
- Every GPU-h/Mtok figure — pure arithmetic on those cells via the stated identity, no fitting.
- Knee = 4 × DP at two shapes across DP8 / TP2×DP4 / TP4×DP2, corroborated by DP2 and DP4.
- Zero refusals C1–C128 in every arm (the engine will accept load that wrecks latency).
- 1 GPU → DP8 same-host scaling **5.889×** (512x256) and **4.088×** (4kx1k), per-GPU efficiency **0.736** and
  **0.511**; matched-rung C1 ratio **0.994×** and C8 ratio **1.493×**.
- Within-arm CV 13.9–18.0 % at 512x256 C1–C4 vs 0.49–2.87 % at the 4kx1k C8 / 30kx2k cells.
- `ab-ladder-B.json` 30kx2k C8 is void (experimenter tore the server down mid-cell).
- TP8 impossible: `EXL3 TP output slice must be 128-aligned, got start=62080, size=31040`; padded vocab
  248320 = 128 × 1940, 1940 = 2²×5×97, so TP ∈ {1,2,4} only.
- `reasoning_effort` medians: xhigh 18,085 tok / 168.8 s vs medium 3,436 tok / 26.2 s → **5.263× tokens,
  6.443× wall**; unset ≡ xhigh; `low` 1.024×/1.031× vs medium; `none` → HTTP 400.
- TB2.1 DP8 C32 real workload: 8454.1 s wall, 18.787 GPU-h, 6,670,376 output tokens, 66,329,299 input
  tokens → **2.816 GPU-h/Mtok, 0.2111 GPU-h/task, 40.5 % of ladder throughput, 1.177× straggler tax**.
- APC arm E: 11.56× / 29.27× TTFT speed-up, 92.6 % / 97.6 % hit rates, whole-schedule 84.03 s vs 144.37 s
  (1.718×) and 51.24 s vs 111.85 s (2.183×).
- Ladders are APC-cold: 240,956 prefix-cache queries, 0 hits, in the DP8 C128 cell.
- 32 KiB/token from 16-of-64 KV layers; 1,879,687 KV tokens/GPU (engine banner); engine's own
  `Maximum concurrency 1.01x` at 262,144 confirms the sequences = tokens/context identity.
- Weight bytes on disk: BF16 51.77 GiB, FP8 28.77 GiB, ours 20.13 GiB (2.5718× / 1.4292× / 1.000×).
- Our decode numerator 20.52 GB/step decomposed (body 15.90, `lm_head` 0.9535 × 4, draft 0.27 × 3) and the
  achievable bandwidth ceiling 1462–1525 GB/s — docs/47 F6/F2. `lm_head` at 5.9997 bits/weight is exact
  arithmetic on 5120 × 248320, not a scaling assumption.
- Zero TB2.1 tasks remain attributable to quantisation (all seven ex-"quantization-suspect" tasks resolve on
  the quantised model at 2× clock — `tb21-8x-2xclock-arm.json` → `attribution_update`).

### INFERENCE (labelled, never presented as measurement)
- Same-host 1→DP8 **peak** ratio. 7.66× is an upper bound (arm A's ladder stops at C8); ~4.3× is the likely
  value using the other host's C64 figure, which is cross-host.
- 1 GPU vs DP8 **at C32** (3.37× speed for 2.73× cost): both cells measured, but on different hosts.
- Medium-effort GPU-h/task for TB2.1 (**0.033–0.040**, an 81–84 % cut): the effort ratios come from one
  prompt / 2 repeats / 1 GPU, and changing effort changes the pass set, which is unmeasured.
- Size of the APC win for *shared subagent prefixes on DP8*. Anchored on arm E's 1.72×–2.18×, but arm E is
  1 GPU, `max-num-seqs` ≤ 4, a different model edition, and a document-reuse pattern.
- Any 2×/4× tier ratio. Those arms sit behind `10.0.0.4` / `127.0.0.1` with no host receipt, so their cells
  stand alone and no cross-tier ratio is drawn from them.
- Nothing is extrapolated to 30kx2k on DP8 or TP2×DP4 — that shape was never run there.
- **All of §7's BF16 and FP8 throughput, KV and cost figures.** There is no measured BF16/FP8 serving
  throughput in this project: the three `decode-parity-bf16-*.json` receipts are logprob/determinism probes
  with no throughput field, and a repo-wide search found nothing else. The decode bracket (BF16 0.386–0.447×,
  FP8 0.709–0.758×) is derived from measured bytes ÷ measured bandwidth under two stated overhead models; the
  KV rows are derived by swapping weights inside the measured 88.3 GiB budget while holding the 5.72 GiB
  activation/graph overhead fixed. The decode-bound assumption **overstates** the BF16 penalty at
  prefill-heavy shapes.

### Not measured at all, and not guessed
- 30kx2k on DP8 or TP2×DP4. 1× ladder rungs above C8 on `jl-vm-473501`. Arms C/D of the bare-metal A/B.
  DP8 behaviour at the 1M static-YaRN window. Any LMCache configuration (disabled in the settled knob set
  and never measured by this project).

---

## 10. Recommendation, in cost order

1. **Set `reasoning_effort: medium` before touching hardware.** 6.44× wall / 5.26× tokens, free, and larger
   than every topology decision below. Unset means xhigh — the expensive default. Do not use `low`; it is
   3.1 % *worse* than medium.
2. **If you can tolerate ~24 tok/s per subagent, one 96 GB GPU is the cheapest way to run 32 subagents**
   (0.418 GPU-h/Mtok at 512x256 C32; KV holds 32 sequences up to 58,740 tokens each). Budget 2.8 GPU-h/Mtok
   for real agent traffic, not 1.1.
3. **If subagents must stay near single-stream speed at 32-way fan-out, DP8 on 8 GPUs, TP1, is the only
   measured configuration that holds the floor** (79.8 tok/s = 59 % of single-stream at 512x256; 71.2 =
   79 % at 4kx1k). It costs 1.141–1.383 GPU-h/Mtok at the knee, +36 %/+96 % over one GPU, and 2.816 on the
   real TB2.1 workload. **Admit at most 32; queue the rest** — C128 halves per-agent speed for a 45 % token
   discount you do not want.
4. **Never use TP for capacity, and never TP4 for this workload.** 21 GB fits one card; TP4×DP2 at the knee
   costs 4.05× DP8 and 4x TP4 at C32 costs 4.2× a single GPU. TP2×DP4 is defensible only to buy short-prompt
   single-stream latency (153.2 vs 135.2 tok/s), at +81 % per token.
5. **Shorten prompts before buying GPUs.** 30k → 4k prompts is a 2.36× cost cut per output token on the same
   box; a 10:1 input:output ratio means prefill is where the money goes.
6. **Quantise — BF16 is the most expensive choice available after `reasoning_effort`.** `[INFERENCE]` BF16
   costs **2.24–2.59× per output token** *and* leaves **0.493×** the KV tokens per GPU (3.54 vs 7.17
   sequences at 262k; it cannot hold even one 1M sequence on a 96 GB card). FP8 costs 1.32–1.41× per token
   and 0.862× the KV. Ours is the cheapest on both channels simultaneously, and TB2.1 attribution says the
   capability price is zero. One `tb21_ladder.py` run per dtype at DP8 would turn all of §7 into measurement.
7. **Keep the 262k window as the default and put 1M behind a separate endpoint.** 1M costs 30.5 GiB of pool
   per sequence, caps an 8-GPU host at 15 concurrent sequences, and Qwen warn static YaRN hurts short-text
   quality.
8. **ForceP2P on every new multi-GPU rental** (35.5 → 51.9 GB/s, +46.5 %) — and note the measured price of
   forgetting: 2x TP2 with P2P off costs 1.771 vs 1.454 GPU-h/Mtok with it on, **+21.8 % per output token**.
9. **`jl-vm-473501` is busy serving production 1M traffic. It was not touched to produce this document, and
   no instance was created, resized, started or stopped.**

---

## 11. USD per million tokens — input, output, cached input, and the utilisation you must hold to win

Everything above is denominated in GPU-hours because that is what the receipts measure. This section applies
the measured prices from `docs/49-jarvislabs-pricing-and-inventory.md` (RTX PRO 6000 96 GB, `india-chennai-01`:
**on-demand $1.89**, **reserved-1y $1.19**, **spot $0.99 — priced but NOT PURCHASABLE**, see §11.5) and splits
the token stream into input, output and cached input, which §0–§10 deliberately did not do.

**Pricing facts carried in, all from docs/49:** billing granularity is **per minute** ("Instances are billed
per-minute"); multi-GPU is **strictly linear with no volume discount** (measured: instance 473501 accrues
15.293 $/h against a catalogue 15.260 = 8 × 1.89 + 1000 GB × 0.00014, ≤0.25 % agreement; FAQ verbatim *"we are
not able to offer any discounts"*); reserved-1y $1.19 is a **37 % discount, sales-negotiated, with no self-serve
endpoint**.

### 11.1 Lead with this: rentals bill OCCUPANCY, so an idle GPU is the most expensive way to buy a token

A rental charges for wall-clock possession, not for tokens. Every `$/Mtok` in this section therefore carries an
invisible divisor — the fraction of the rented hour the endpoint is actually producing tokens. Define it:

```
u  =  busy box-hours / rented box-hours          (occupancy)
$/Mtok(u)  =  $/Mtok(busy)  /  u
```

At `u = 0.10` every price below multiplies by **10×**. This term dominates the entire document: it is larger
than the 4.05× worst-to-best topology spread (§1) and comparable to the 6.443× reasoning-effort lever (§5).

**Break-even identity (no modelling, no fitted constants):**

```
u*  =  (GPUs × $/GPU-hr)  /  (Mtok_in_per_busy_hour × API_$/Mtok_in  +  Mtok_out_per_busy_hour × API_$/Mtok_out)
```

i.e. *self-hosting beats the API only if the box is busy more than `u*` of the hour you pay for.*

**Retrieved API list prices** (2026-08-17; these are published prices, not estimates):

| hosted offer | $/Mtok in | $/Mtok out | cached in | source |
|---|---:|---:|---:|---|
| **`Qwen3.8-27B` — our exact model, OpenRouter, 3 providers, 262,144 ctx** | **0.40** | **3.00** | not published | <https://openrouter.ai/qwen/qwen3.8-27b> (retrieved 2026-08-17) |
| `Qwen3-32B` open-weight commodity floor (cheapest of 7 providers, DeepInfra) | 0.080 | 0.280 | 0.145 | <https://pricepertoken.com/pricing-page/model/qwen-qwen3-32b> (page states "Last updated: August 17, 2026") |
| `Qwen3.8-Max` frontier, Alibaba Model Studio, International (Singapore) | 2.00 | 6.00 | **0.25** | <https://developer.puter.com/tutorials/qwen-api-pricing/> (Aug 2026); Model Studio implicit-cache rate corroborated by <https://windowsforum.com/windows-news.4/qwen3-8-max-costs-2-6-per-million-tokens-still-unproven-for-coding-agents.441542/> |

**No Qwen-operated hosted endpoint for `Qwen3.8-27B` was found** — Model Studio publishes per-token prices for
the Max/Plus/Flash tiers, not for the 27B open-weight checkpoint. The OpenRouter line is the closest thing to a
list price for *this* model and is what the tables below use; the Chinese-Mainland (Beijing) Model Studio
endpoint is reported 60–70 % cheaper than the Singapore rates quoted above, which would move the frontier row
but not the 27B row.

**Break-even against the two MEASURED workloads.** Both numerators are measured token counts, not assumptions.

*(a) Real agent traffic, 9.94:1* — `tb21-8x-2xclock-arm.json` + `tb21-8x-p1-merged.json` (§4): 66,329,299 input
+ 6,670,376 output tokens in 18.787 GPU-h = 2.3484 box-hours on DP8, i.e. **28.244 Mtok in + 2.8404 Mtok out
per busy box-hour** → API cost of that hour = 28.244×0.40 + 2.8404×3.00 = **$19.82/h**.

*(b) Real multimodal traffic, 29.8:1* — `receipts/multimodal-load-8x.json`: 26,183 tok/s prefill + 878 tok/s
decode aggregate = **94.259 Mtok in + 3.1608 Mtok out per box-hour** → API cost = **$47.19/h**.
(Identity check on the receipt: 94.259 / 3.1608 = **29.82**, reproducing its own published 29.8:1 ratio from its
two throughput fields — the receipt is internally consistent.)

| workload (measured) | tier | self-host $/box-h | API $/box-h @ 0.40/3.00 | API ÷ self | **break-even occupancy `u*`** |
|---|---|---:|---:|---:|---:|
| TB2.1 agent, 9.94:1, DP8 C32 | on-demand 1.89 | 15.12 | 19.82 | 1.31× | **76.3 %** |
| TB2.1 agent, 9.94:1, DP8 C32 | reserved-1y 1.19 | 9.52 | 19.82 | 2.08× | **48.0 %** |
| TB2.1 agent, 9.94:1, DP8 C32 | *spot 0.99 (unpurchasable)* | *7.92* | 19.82 | *2.50×* | *40.0 %* |
| live multimodal, 29.8:1, DP8 | on-demand 1.89 | 15.12 | 47.19 | 3.12× | **32.0 %** |
| live multimodal, 29.8:1, DP8 | reserved-1y 1.19 | 9.52 | 47.19 | 4.96× | **20.2 %** |
| ladder knee 4kx1k C32, DP8 (synthetic 4.14:1) | on-demand 1.89 | 15.12 | 26.94 | 1.78× | 56.1 % |
| ladder peak 4kx1k C128, DP8 (synthetic 4.14:1) | on-demand 1.89 | 15.12 | 46.64 | 3.08× | 32.4 % |
| ladder knee 4kx1k C8, **1 GPU** (synthetic 4.14:1) | on-demand 1.89 | 1.89 | 6.59 | 3.48× | **28.7 %** |
| ladder knee 4kx1k C8, **1 GPU** | reserved-1y 1.19 | 1.19 | 6.59 | 5.54× | 18.1 % |

**Read this as three findings, one of them uncomfortable:**

1. **On-demand DP8 against our own model's list price is a thin trade: you must keep the box busy 76 % of every
   rented hour.** The measured TB2.1 arm did exactly one thing for 2.35 h, so its own `u` was ~1.0 and it won
   1.31×. A DP8 box rented by the day for bursty subagent work will not hold 76 %. **Reserved-1y (48.0 %) or a
   single GPU (28.7 %) are the configurations that survive realistic idleness.**
2. **Prefill-heavy traffic wins much more easily** — 32.0 % on-demand — because the box's cheapest product
   (input tokens, §11.3) is exactly what the API charges most aggressively for at 29.8:1.
3. **Against commodity 32B-class pricing, self-hosting cannot win at any occupancy.** At $0.08/$0.28 the TB2.1
   hour is worth **$3.05** of API against $15.12 of rented DP8 → `u* = 495 %`, and $9.52 reserved → `u* = 312 %`.
   Even the multimodal hour ($8.43 of API) needs `u* = 180 %` on-demand / 113 % reserved. **Impossible.** If the
   decision is purely cost-per-token against a commodity provider, rent nothing. The reasons to self-host this
   checkpoint are the ones cost cannot express: data residency, a 262k–1M window under our control, `ForceP2P`
   and quantisation control, no per-request rate limit, and the fidelity work in docs/29/35.
   Against the frontier tier ($2.00/$6.00) the trade inverts hard: `u* = 20.6 %` on-demand, 12.9 % reserved.

**As an explicit function of an assumed API price**, holding the OpenRouter output:input ratio of 7.5:1
(3.00/0.40) and using the measured agent hour:

```
u*(P_in)  =  (GPUs × $/GPU-hr) / ( 28.244·P_in + 2.8404·7.5·P_in )  =  (GPUs × $/GPU-hr) / (49.548 · P_in)
```

so the **minimum API input price at which a 100 %-busy box breaks even** is
**$0.3052/Mtok on-demand** and **$0.1921/Mtok reserved-1y** for agent traffic
(15.12/49.548 and 9.52/49.548); for multimodal traffic (94.259 + 7.5×3.1608 = 117.965 Mtok-equivalent/h) it is
**$0.1282** and **$0.0807**. Substitute any provider's input price to get `u*` directly.

### 11.2 How input and output are separated — and why C1

**You cannot get here by dividing the GPU-h/Mtok figures in §1.** Those are denominated in *output* tokens with
prefill folded into the denominator (header identity). The split below is derived from the **per-request records**
in the ladder receipts, `tables[shape].cells[i].repeats[j].requests[k]`, which carry `ttft_s`, `duration_s`,
`prompt_tokens`, `completion_tokens` per request:

```
input (prefill) rate  =  prompt_tokens / ttft_s
output (decode) rate  =  completion_tokens / (duration_s − ttft_s)
$/Mtok                =  1e6 / (rate × 3600) × (GPUs × $/GPU-hr)
```

**C1 cells only, for the prefill rate.** At any concurrency, `ttft_s` measures *queue + prefill*, not prefill,
because the harness streams **one** request per repeat (slot 0) *inside* the concurrent batch (`ttft_note`:
"TTFT from one streamed request per repeat running INSIDE the cell's concurrent batch"). The contamination is
measurable and large — arm A 4kx1k `prompt_tokens/ttft_s` reads **4751.7** tok/s at C1, **2527.2** at C2,
**1321.5** at C4, **1326.7** at C8. Nothing slowed down by 3.6×; the queue grew. **C1 is the only rung where
TTFT is prefill.** The DP8 receipt proves the mechanism cleanly in the other direction: at 512x256 its slot-0
TTFT is flat at 0.1783 / 0.1808 / 0.1738 s across C1 / C2 / C4 (apparent rate 3333 / 3324 / 3425 tok/s) because
four requests land on four *distinct* DP replicas and never queue — then halves to 1658 tok/s at C8 when
replicas start doubling up.

Because prefill runs inside the request's own `ttft_s`, the derived input rate is a **floor**: TTFT also contains
HTTP/SSE transport ("first SSE body byte"), scheduler admission, and the first decode step. §11.3 quantifies
that excess at 64 ms and shows it only matters for short prompts.

### 11.3 The measured input rate, and its stability across shapes

Per-GPU prefill rate from every C1 cell in the four ladder receipts that carry one (medians over 5 repeats for
`ab-ladder-A/B`, 3 for the others):

| receipt | shape | prompt tok (actual) | ttft_p50 (s) | **input rate tok/s/GPU** | spread across repeats |
|---|---|---:|---:|---:|---|
| `ab-ladder-A.json` | 512x256 | 589 | 0.1885 | **3128.0** | 3028–3189 |
| `ab-ladder-A.json` | 4kx1k | 4238 | 0.8919 | **4751.7** | 4722–4775 |
| `ab-ladder-A.json` | 30kx2k | 31495 | 6.6436 | **4740.3** | 4733–4746 |
| `ab-ladder-B.json` | 512x256 | 589 | 0.1881 | 3151.5 | 3024–3192 |
| `ab-ladder-B.json` | 4kx1k | 4238 | 0.8880 | 4764.5 | 4749–4790 |
| `ab-ladder-B.json` | 30kx2k | 31495 | 6.6379 | 4743.1 | 4737–4750 |
| `tb21-ladder-8x-dp8.json` | 512x256 | 593 | 0.1783 | 3333.3 | 3247–3335 |
| `tb21-ladder-8x-dp8.json` | 4kx1k | 4242 | 0.8679 | **4887.7** | 4858–4905 |
| `tb21-ladder-1x-hyd.json` *(other host, corroboration)* | 4kx1k | 4242 | 0.8945 | *4742.3* | 4741–4750 |
| `tb21-ladder-1x-hyd.json` *(other host)* | 30kx2k | 31565 | 6.6587 | *4739.9* | 4731–4740 |

**Stability verdict: the input rate is stable to ±1.6 % for prompts ≥ 4k, and 34 % lower at 512 tokens.**

- For prompts ≥ 4k the full range across two arms, two hosts, one and eight GPUs is
  **4739.9 – 4887.7 tok/s/GPU** (±1.55 % about 4813.8). **Use 4740 tok/s/GPU** — the low end, and the value both
  large shapes on the DP8 host's single GPU agree on.
- **The 30k shape does not cost more per input token than the 4k shape**, contrary to what §1's shape row might
  suggest: 4740.3 vs 4751.7 tok/s on the same arm, a 0.24 % difference. §1's 2.36× penalty for 30kx2k is real
  but it lands **entirely on the output denominator** — the 30k prompt is 15.4 input tokens per output token
  instead of 4.1, so the same cheap input tokens are amortised over 3.7× fewer output tokens. Separating the
  legs is exactly what dissolves that apparent shape penalty. (Within the 262k window; §11.8 is where per-token
  prefill cost genuinely rises.)
- **Shown identity for the flatness, and for the 512 outlier.** Two points, one line: the 4k and 30k C1 cells on
  arm A give `(31495 − 4238)/(6.6436 − 0.8919) = 27257/5.7517 = ` **4739.0 tok/s** marginal, with intercept
  `0.8919 − 4238/4739.0 = −0.0024 s` — i.e. essentially zero fixed cost, so prefill is linear in tokens across
  a 7.4× prompt-length range. Feeding 589 tokens into that line predicts a TTFT of **0.1243 s** against a
  measured **0.1885 s**: **64 ms of the 512-shape TTFT is not prefill.** That is transport + admission + the
  first decode step, and it is why the 512-token row reads 3128 rather than ~4740. **The 512-token input price
  is therefore an upper bound contaminated by fixed overhead, and it inherits the 13.9–18.0 % CV that §0
  attaches to every 512x256 low-concurrency cell.**

### 11.4 The output rate — derived as the residual, so the split closes exactly against §1

Charging output with `completion_tokens/(duration_s − ttft_s)` from slot 0 alone does **not** close: slot 0 is
one request of C, it is the streamed one, and multiplying its rate by C over-states aggregate decode by up to
1.97× (DP8 512x256 C32: 32 × 120.10 = 3843 tok/s claimed against the cell's measured 1947.2 aggregate). So the
output leg is taken as the **exact residual of the measured cell wall after charging prefill at the C1 rate**:

```
wall            =  Σcompletion_tokens / aggregate_tok_s          (harness identity, verified in the header)
T_prefill       =  Σprompt_tokens / (min(C, DP) × input_rate_C1)
T_decode        =  wall − T_prefill
aggregate decode=  Σcompletion_tokens / T_decode
```

This is an identity, not a fit, and it has the right conservatism: **prefill is charged at its best measured
rate, so the input leg is a floor and the output leg absorbs every overlap, straggler and scheduling cost.**
By construction `output_GPU-h/Mtok + (prompt/completion) × input_GPU-h/Mtok` reproduces the committed §1 cell
**exactly** — the last two columns are the audit:

| cell | wall (s) | T_prefill | prefill share of wall | aggregate decode tok/s | **in GPU-h/Mtok** | **out GPU-h/Mtok** | sum | §1 cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GPU 512x256 C8 (peak) | 4.414 | 1.506 | 34.1 % | 704.4 | 0.08880 | 0.3943 | 0.599 | **0.599** |
| 1 GPU 4kx1k C8 (knee) | 20.845 | 7.135 | 34.2 % | 597.5 | 0.05846 | 0.4649 | 0.707 | **0.707** |
| 1 GPU 30kx2k C4 (knee) | 49.083 | 26.576 | 54.1 % | 364.0 | 0.05860 | 0.7632 | 1.664 | **1.665** |
| 1 GPU 30kx2k C8 (peak) | 87.149 | 53.153 | 61.0 % | 481.9 | 0.05860 | 0.5764 | 1.478 | **1.478** |
| DP8 512x256 C32 (knee) | 4.207 | 0.712 | 16.9 % | 2343.6 | 0.08333 | 0.9482 | 1.141 | **1.141** |
| DP8 512x256 C128 (peak) | 9.224 | 2.846 | 30.9 % | 5138.3 | 0.08333 | 0.4325 | 0.626 | **0.626** |
| DP8 4kx1k C32 (knee) | 20.392 | 3.472 | 17.0 % | 1936.6 | 0.05683 | 1.1475 | 1.383 | **1.383** |
| DP8 4kx1k C128 (peak) | 47.111 | 13.886 | 29.5 % | 3945.0 | 0.05683 | 0.5633 | 0.799 | **0.799** |

`min(C, DP) × input_rate_C1` assumes DP replicas prefill independently and linearly `[INFERENCE]`; it is
supported by the DP8 4kx1k C16 cell, whose slot-0 rate is **4840.7** tok/s with two requests per replica —
i.e. per-replica prefill is undamaged by spreading concurrency across replicas.

**Prefill is 17–61 % of the wall even on prefix-cache-COLD synthetic traffic** — 61 % at 30kx2k C8 on one GPU.
That is the structural reason the input leg cannot be ignored, and the reason §8's prefix caching is a
first-order cost lever rather than a latency nicety.

### 11.5 $/Mtok tables — input and output, three price points, 1 GPU and DP8

**INPUT — `$/Mtok` is GPU-count-invariant under data parallelism**, because DP8 has 8× the prefill rate and 8×
the bill: `1e6/(8R × 3600) × 8p ≡ 1e6/(R × 3600) × p`. The 1-GPU and DP8 rows are therefore identical *when
every replica is busy*; they diverge only through occupancy (§11.1), which is where DP8 is punished.

| input class | rate tok/s/GPU | GPU-h/Mtok | **$1.89 on-demand** | **$1.19 reserved-1y** | *$0.99 spot* |
|---|---:|---:|---:|---:|---:|
| prompts ≥ 4k, 1 GPU **or** DP8 saturated (planning number) | 4740.3 | 0.05860 | **$0.1108** | **$0.0697** | *$0.0580* |
| prompts ≥ 4k, best measured (DP8 C1 4kx1k) | 4887.7 | 0.05683 | $0.1074 | $0.0676 | *$0.0563* |
| 512-token prompts (overhead-contaminated upper bound) | 3128.0 | 0.08880 | $0.1678 | $0.1057 | *$0.0879* |
| **live production, 8× under real mixed load** (§11.6) | 3272.9 | 0.08484 | **$0.1604** | **$0.1010** | *$0.0840* |

**OUTPUT — this is where GPU count and concurrency matter enormously.**

| topology | cell | aggregate decode tok/s | GPU-h/Mtok | **$1.89** | **$1.19** | *$0.99* |
|---|---|---:|---:|---:|---:|---:|
| **1 GPU** | 512x256 C8 (peak) | 704.4 | 0.3943 | **$0.745** | $0.469 | *$0.390* |
| **1 GPU** | 4kx1k C8 (knee) | 597.5 | 0.4649 | **$0.879** | $0.553 | *$0.460* |
| **1 GPU** | 30kx2k C8 (peak) | 481.9 | 0.5764 | $1.089 | $0.686 | *$0.571* |
| **1 GPU** | 30kx2k C4 (knee) | 364.0 | 0.7632 | $1.442 | $0.908 | *$0.756* |
| **1 GPU** | single stream C1 4kx1k | 100.0 | 2.7778 | $5.250 | $3.306 | *$2.750* |
| **DP8** | 512x256 C128 (peak) | 5138.3 | 0.4325 | **$0.817** | $0.515 | *$0.428* |
| **DP8** | 4kx1k C128 (peak) | 3945.0 | 0.5633 | $1.065 | $0.670 | *$0.558* |
| **DP8** | 512x256 C32 (knee) | 2343.6 | 0.9482 | **$1.792** | $1.128 | *$0.939* |
| **DP8** | 4kx1k C32 (knee) | 1936.6 | 1.1475 | **$2.169** | $1.366 | *$1.136* |
| **DP8** | one request in flight (C1, 8 GPUs billed) | 97.6 | 22.766 | **$43.03** | $27.09 | *$22.54* |
| **DP8** | real TB2.1 agent workload, C32 (§4) | — | 2.816 | **$5.322** | $3.351 | *$2.788* |

**Output tokens cost 8–20× what input tokens cost on this hardware** ($0.879 vs $0.1108 at the 1-GPU knee;
$2.169 vs $0.1074 at the DP8 knee). Any pricing intuition imported from commercial API sheets — where the
output:input ratio is 7.5:1 on OpenRouter and 3:1 on Model Studio — **understates how output-skewed a
self-hosted box is**, which is the arithmetic reason prefill-heavy traffic self-hosts so much better (§11.1).

**The $43.03/Mtok row is the whole argument of §11.1 in one cell.** One request in flight on a DP8 box costs
**49× the same request's output on one GPU** ($43.03 vs $0.879), purely because seven replicas are billed and
idle. Buy DP8 for concurrency you actually have.

**Spot is marked but not usable.** $0.99/GPU-hr is real in the catalogue and on the public pricing page
("Spot · save up to 56 %"), but docs/49 §3.2 retrieved two blocking constraints from the SDK: spot is **GPU
containers only** (`"Spot instances are only supported for GPU containers."`, `instances.py:207`), so there is no
root VM and therefore no Docker, no driver control and **no `ForceP2P`** — which §10 item 8 measures at +46.5 %
interconnect bandwidth and +21.8 % per output token when forgotten; and `spot_num_free_devices` for the
RTX-PRO6000 container row is **0**. Every spot column here is italicised for that reason: it is a price, not an
option.

### 11.6 Cross-check against live production — ladder vs reality, and which to plan with

`receipts/multimodal-load-8x.json`, DP8 at 1M context under the owner's real interactive multimodal load
(727 s window, 92.9 % mean GPU utilisation, 5.6 % idle samples, zero requests waiting on any of the eight
engines): **26,183 tok/s prefill aggregate = 3272.9 tok/s/GPU**, **878 tok/s decode aggregate**.

| quantity | ladder (clean, synthetic) | live production | live ÷ ladder |
|---|---:|---:|---:|
| input rate, tok/s/GPU | 4740.3 | **3272.9** | **0.690** |
| **input $/Mtok @ $1.89** | $0.1108 | **$0.1604** | **1.448×** |
| output $/Mtok @ $1.89, DP8 | $2.169 (knee C32) | **$4.784** | 2.205× |
| all-token $/Mtok @ $1.89 | — | **$0.1552** | — |

**They disagree, by 1.45× on input and 2.21× on output, and that is expected rather than alarming.** Four
measured reasons, all named in the receipt or in §0: the live window is **1M static-YaRN** context, not 262k, and
§11.8 prices that; it is **multimodal**, with 48 `video_processing_qwen` events, and vision encoding is real GPU
work charged to no token; a **TB2.1 arm ran at C16 throughout** (`honest_limits[0]`: "absolute tok/s are
depressed by contention"); and the ladder cells are **prefix-cache-cold by construction** while carrying **zero**
of the real workload's request-mix cost.

**Which is the better planning number: the live one, for anything you are going to be billed for.**

- Use **$0.1604/Mtok input and $4.784/Mtok output** (on-demand DP8) to budget real traffic. It is a
  contention-and-mix-inclusive measurement on the production endpoint, and it is the same discipline §4 already
  established when it told you to budget 2.8 GPU-h/Mtok instead of the ladder's 1.1.
- Use the ladder's **$0.1108 / $0.879–$2.169** as the **engineering floor** — the number to compare topologies
  against, and the number that tells you how much of the gap is recoverable.
- The recoverable gap is **0.690** on the input leg: the live box ran prefill at 69.0 % of the clean rate. That
  is the input-side twin of §4's 40.5 % output-side "reality tax", and it is a *smaller* haircut, which is the
  useful news — **prefill degrades more gracefully under real load than decode does**.

**One honest limit on the split at 29.8:1.** With 92.9 % mean utilisation, 94.259 Mtok of input and only
3.1608 Mtok of output per box-hour, the input/output *allocation* of the $15.12 hourly bill is close to a
convention: charging the whole hour to input gives $0.1604/Mtok in and $0/Mtok out; charging it all to output
gives $4.784/Mtok out and $0/Mtok in. What is **invariant and safe to quote is the all-token
$0.1552/Mtok** (= 15.12 / 97.420 Mtok) and the **$15.12/box-hour**. The per-phase split above is the honest
decomposition *given the measured phase rates*; it is not a second independent measurement.

### 11.7 Cached input — the measured 48.0 %, with the unit caveat intact

**Mechanism, and why cached input is nearly free here.** A prefix-cache hit means the KV blocks for those
prompt tokens are already resident in the paged pool. The engine skips all prefill compute for them — no
attention, no MLP, no GDN recurrence — and pays only a block-table lookup plus the KV read that decode was
going to do anyway. There is no separate "cache write" cost on this stack either: the KV was written by the
prefill you already paid for, and APC's storage cost is measured at **1,344 tokens of window on a 32 GiB card
(0.51 %), irrelevant on 96 GB** (§8).

**MEASURED hit rate (`receipts/multimodal-load-8x.json`): 26.9 % on the text-agent window → 48.0 % under the
owner's real subagent traffic, +21.0 points**, on 32,370,301 queries / 15,529,600 hits. Mechanism per the
receipt: "omp subagents share system prompts and tool schemas".

> **UNIT CAVEAT, carried verbatim from the receipt:** *"vllm `prefix_cache_queries` counts BLOCK lookups, not
> tokens. 48.0 % is a block-level hit rate and is NOT the fraction of prompt tokens skipped. Do not convert it
> to a token discount without a token-level measurement."*

So the price below is built in two separable steps, only the second of which is inference.

**Step 1 — the price of one cached input token. MEASURED, token-level, from `apc-poison-repro.json` arm E**
(§8), which reports both wall and recomputed-token counts and therefore needs no unit conversion:

| prefix | cold TTFT | warm TTFT | warm ÷ cold | prompt tokens recomputed warm | recompute fraction |
|---|---:|---:|---:|---:|---:|
| 32,842 tok | 12.071 s | 1.044 s | **0.0865** | 2,442 / 32,842 | 0.0744 |
| 131,146 tok | 67.597 s | 2.309 s | **0.0342** | 3,146 / 131,146 | 0.0240 |

The wall ratio tracks the recompute fraction to within 1.2 points on both prefixes, which is the check that the
time really did go where the token accounting says. **Take the conservative (shorter-prefix) figure: a cache-hit
input token costs 0.0865× a cold one.**

| cached-input line item | $1.89 | $1.19 | *$0.99* |
|---|---:|---:|---:|
| cold input, prompts ≥ 4k (from §11.5) | $0.1108 | $0.0697 | *$0.0580* |
| **cached input, 0.0865× (32.8k-prefix anchor)** | **$0.0096** | **$0.0060** | *$0.0050* |
| cached input, 0.0342× (131k-prefix anchor) | $0.0038 | $0.0024 | *$0.0020* |
| lower bound (pure KV reuse, zero prefill compute) | $0 | $0 | $0 |

So **cached input on this stack prices at $0.004–$0.010/Mtok on-demand** — against OpenRouter's unpublished
cached rate, Model Studio's $0.25/Mtok implicit-cache rate, and the $0.145/Mtok commodity 32B cache-read rate.
**Cached input is the one token class where self-hosting wins by more than an order of magnitude, at 15–65×.**

**Step 2 — applying the 48.0 % fleet hit rate. `[INFERENCE]`, and here is the exact assumption.**
Converting a block-level rate to a token-level rate is only valid if every counted query covers a full
`block_size` window of prompt tokens and the queries partition the prompt — then `hits/queries` equals
`hit tokens/prompt tokens` because the same constant `block_size` cancels top and bottom. Reality deviates
through partial trailing blocks and through queries that are not prompt-prefix lookups, and the direction of
the error is **not measured**. `[INFERENCE]` treating 48.0 % as a token-level rate:

```
effective input factor  =  0.52  +  0.48 × 0.0865  =  0.5615
```

| effective input price, 48 % token-equivalent hit rate `[INFERENCE]` | $1.89 | $1.19 | *$0.99* |
|---|---:|---:|---:|
| prompts ≥ 4k, 0.5615 × cold | **$0.0622** | **$0.0392** | *$0.0326* |

**In production, cached input is very likely the DOMINANT input class**, and that is the single most important
consequence of this subsection. The measured hit rate nearly doubled (26.9 → 48.0 %) the moment real subagent
traffic replaced a single-agent campaign, for a structural reason that gets *stronger* with fan-out: 32
subagents share one system prompt and one tool-schema block. §8's arm E already shows the effect growing with
concurrency (1.718× whole-schedule speed-up at `max-num-seqs 1` → 2.183× at 4). **Every input price in §11.5
is therefore a cold-cache ceiling, and the 48 % row is the closer estimate for a fan-out workload** — while
remaining inference until someone runs a DP8 ladder arm with a shared prefix instead of unique filler, which
§8 already names as the cheap way to close this.

### 11.8 Long-context surcharge — input tokens are NOT flat-priced

`receipts/1m-context-effects.json` fits `prefill wall ~ n^k` on four measured needle rungs and finds **k rising
monotonically: 1.506 (262k→524k), 1.744 (524k→786k), 2.144 (786k→1M)**, crossing 2 at the top of the window.
Mechanism, from the receipt: 48 of 64 layers are linear attention, cost O(n); 16 of 64 are full attention, cost
O(n²) — the quadratic minority takes over as n grows. A single global exponent (k=1.584 fitted on the first
three rungs) **under-predicted the 1M wall by 15.5 %** (1043 s predicted vs 1205 s measured), which is the
receipt's own warning against flat or single-exponent pricing.

| window | doc tokens | wall (s) | effective prefill tok/s | **per-token cost, relative to 262k** | segment k into this rung | **input $/Mtok @ $1.89, one replica** | input $/Mtok, 8 GPUs billed for one request |
|---|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 259,303 | 125 | 2074.4 | **1.00** | — | **$0.2531** | $2.025 |
| 524,288 | 520,528 | 355 | 1466.3 | **1.42** | 1.506 | **$0.3580** | $2.864 |
| 786,432 | 781,653 | 720 | 1085.6 | **1.92** | 1.744 | **$0.4836** | $3.869 |
| 1,000,000 | 994,755 | 1205 | 825.5 | **2.53** | 2.144 | **$0.6360** | $5.088 |

Applied to the clean 262k-equivalent input price instead, the surcharge column gives
**$0.1108 → $0.1573 → $0.2127 → $0.2803 /Mtok**.

Three things to keep straight:
- **The relative column is the trustworthy one.** All four rungs were measured on the DP8 host *while a TB2.1
  arm ran at C16* (`honest_limits[0]`), so the absolute rates are contended: the 262k rung's $0.2531 is **2.28×**
  the clean 31k-prompt price of $0.1108, and that factor is contention *plus* window, not window alone. The
  ratios between rungs largely cancel the common-mode contention; the receipt itself states the monotone rise is
  robust while "the exact k values are not to three digits".
- **The two right-hand columns are the same measurement under two billing conventions**, and their 8× gap is
  §11.1 again: a single 1M request occupies one replica, so on a DP8 box you are billed 8 GPUs for it. A lone 1M
  prefill on a rented DP8 host costs **$5.088/Mtok input — 46× the saturated 4k input price**.
- **Compounding with §6:** 1M context also caps the host at **15 concurrent full-length sequences** (1.88 per
  GPU at 32 KiB/token from 1,879,687 KV tokens), so the configuration that makes each input token 2.53× more
  expensive is also the one that most restricts your ability to amortise the box. **Price 1M work at
  2.53× input and on a separate endpoint** (§10 item 7).

### 11.9 Blended $/Mtok at both measured traffic ratios

To compare against a single commercial per-token price you need the ratio. Both ratios below are **measured**:
**9.94:1** on the TB2.1 text-agent campaign (§4, 66,329,299 : 6,670,376) and **29.8:1** on the live mixed
multimodal DP8 endpoint (§11.6), with a text-agent window at **15.5:1** inside that same period — so the
9.94–29.8 span is the honest planning range, not a worst case.

`$/Mtok_output = r × input + output`; `blended $/Mtok(any token) = (r × input + output) / (r + 1)`.

| topology / tier | r | cold input | output | **$/Mtok of OUTPUT** | **blended $/Mtok, any token** | with 48 % cache `[INF]`: $/Mtok out | blended |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 GPU knee 4kx1k, **$1.89** | 9.94 | 0.1108 | 0.879 | **$1.980** | **$0.1809** | $1.497 | $0.1368 |
| 1 GPU knee 4kx1k, **$1.89** | 29.8 | 0.1108 | 0.879 | **$4.179** | **$0.1357** | $2.732 | $0.0887 |
| 1 GPU knee 4kx1k, **$1.19** | 9.94 | 0.0697 | 0.553 | $1.246 | $0.1139 | $0.942 | $0.0861 |
| 1 GPU knee 4kx1k, **$1.19** | 29.8 | 0.0697 | 0.553 | $2.631 | $0.0854 | $1.720 | $0.0558 |
| DP8 knee 4kx1k, **$1.89** | 9.94 | 0.1074 | 2.169 | **$3.236** | **$0.2958** | $2.768 | $0.2530 |
| DP8 knee 4kx1k, **$1.89** | 29.8 | 0.1074 | 2.169 | **$5.370** | **$0.1743** | $3.966 | $0.1288 |
| DP8 peak 4kx1k C128, **$1.89** | 9.94 | 0.1074 | 1.065 | $2.132 | $0.1949 | $1.664 | $0.1521 |
| DP8 peak 4kx1k C128, **$1.89** | 29.8 | 0.1074 | 1.065 | $4.265 | $0.1385 | $2.862 | $0.0929 |
| DP8 knee 4kx1k, **$1.19** | 9.94 | 0.0676 | 1.366 | $2.038 | $0.1863 | $1.743 | $0.1593 |
| DP8 knee 4kx1k, **$1.19** | 29.8 | 0.0676 | 1.366 | $3.381 | $0.1098 | $2.497 | $0.0811 |
| **live production DP8, $1.89** | 29.8 | 0.1604 | 4.784 | **$9.564** | **$0.3105** | — | — |
| **real TB2.1 agent DP8, $1.89** (§4) | 9.94 | — | — | **$5.322** | **$0.4865** | — | — |

Compare the single commercial prices for the same ratios: at 9.94:1 OpenRouter costs
`3.00 + 9.94×0.40 = $6.976` per Mtok of output ($0.6376 blended); at 29.8:1, `3.00 + 29.8×0.40 = $14.920`
($0.4844 blended).

**The two rows that matter:** on measured real agent traffic we pay **$5.322/Mtok output on-demand against
$6.976 of API** — a 1.31× win that evaporates below 76 % occupancy (§11.1). And **the ladder-derived DP8 knee
figure at the same 9.94:1 is $3.236**, so the input-inclusive reality tax is **1.64×** (5.322/3.236) — smaller
than §4's 2.47× because that comparison charged the microbenchmark nothing for the 9.94:1 prefill it never ran.
**§4's 2.47× is the right tax on an output-only denomination; 1.64× is the right one once input is priced.**

### 11.10 The 6.443× lever still dominates every cell above

Nothing in this section comes close to `reasoning_effort` (§5: xhigh = **5.263× output tokens and 6.443× wall**
versus medium; unset ≡ xhigh; `low` is 1.024×/1.031× and so never cheaper than medium; only xhigh/medium/low
exist). Ranked against the price levers introduced here:

| lever | multiplier on $/task |
|---|---:|
| **`reasoning_effort` xhigh → medium** | **6.443× (occupancy-billed, which is how rentals bill)** |
| occupancy 100 % → 15 % on the same box | 6.67× |
| worst → best 8-GPU topology at the knee (§5) | 4.05× |
| on-demand $1.89 → reserved-1y $1.19 | 1.588× |
| 262k → 1M window, per input token (§11.8) | 2.53× |
| cold → 48 %-cached input `[INF]` (§11.7) | 1.78× on the input leg only |
| $1.89 → $0.99 spot, if it were purchasable | 1.909× |

**And effort interacts with the buy-vs-rent decision, against self-hosting.** An API bills you 5.263× for
xhigh (tokens); a rental bills you 6.443× (wall). Running xhigh therefore **tilts the comparison 1.224× in the
API's favour** — you pay a 22 % premium for the same verbosity purely because you rent by the clock.
`[INFERENCE]`, carrying §5's two caveats forward (the ratios come from one prompt × 2 repeats × 1 GPU, and
changing effort changes which tasks pass): the measured TB2.1 arm at medium would cost **$5.51** of DP8
on-demand against **≤$30.33** of OpenRouter — **5.50× cheaper, break-even occupancy 18.2 %** instead of 76.3 %.
**Setting `reasoning_effort: medium` does more for the buy-vs-rent case than every pricing tier and topology
choice in this document combined.** The API-side figure is an upper bound because only its output leg was
divided by 5.263 while its input leg was held fixed; a shorter reasoning loop also shortens the transcript that
the next turn re-reads, so the real API saving is larger and the real `u*` lower still.

### 11.11 What is measured and what is inference in this section

**MEASURED (receipt-traceable):**
- Every C1 input rate in §11.3, from `requests[k].prompt_tokens / requests[k].ttft_s` in `ab-ladder-A.json`,
  `ab-ladder-B.json`, `tb21-ladder-8x-dp8.json`, `tb21-ladder-1x-hyd.json`. Range for prompts ≥ 4k:
  **4739.9–4887.7 tok/s/GPU**; 512-token prompts **3103–3333**.
- The queueing contamination that forces the C1 choice: arm A 4kx1k apparent rate 4751.7 → 2527.2 → 1321.5 →
  1326.7 at C1/C2/C4/C8; DP8 512x256 flat at 3333/3324/3425 across C1/C2/C4.
- Every wall, `aggregate_tok_s`, prompt and completion count feeding §11.4; the sums reproduce the committed §1
  cells to three digits.
- Live production: 26,183 tok/s prefill / 878 tok/s decode on 8 GPUs, 92.9 % mean utilisation, 5.6 % idle
  samples, zero waiting requests; 29.8:1 and 15.5:1 ratios; 48.0 % vs 26.9 % block-level prefix-cache hit rate
  on 32,370,301 queries.
- Arm E cache-hit economics: 12.071→1.044 s and 67.597→2.309 s, 2,442/32,842 and 3,146/131,146 tokens
  recomputed.
- TB2.1 real workload token counts and GPU-hours (66,329,299 / 6,670,376 / 18.787), giving every break-even
  number in §11.1 without an assumed constant.
- 1M-context prefill law: walls 125/355/720/1205 s at 259,303/520,528/781,653/994,755 tokens; k = 1.506 /
  1.744 / 2.144; relative per-token cost 1.00 / 1.42 / 1.92 / 2.53.
- Prices: $1.89 / $1.19 / $0.99 per GPU-hour, per-minute billing, linear multi-GPU (docs/49, catalogue +
  accrual-measured to ≤0.25 %).
- API list prices as retrieved 2026-08-17, with URLs in §11.1. These are *published prices*, i.e. measured
  facts about a price list — not measurements of anyone's cost.

**INFERENCE (labelled where used):**
- `min(C, DP) × input_rate_C1` as the saturated prefill capacity of a DP group (§11.4). Supported by the
  DP8 4kx1k C16 cell at 4840.7 tok/s/replica, not proven at C128.
- Treating the 48.0 % **block-level** hit rate as a token-level rate (§11.7). The block-size cancellation
  argument is given; the deviation from partial blocks and non-prefix queries is unmeasured and its sign is
  unknown. Every "48 %" price is labelled `[INFERENCE]` and the cold price is always shown beside it.
- Applying arm E's 0.0865× cache-hit factor to DP8: arm E is one RTX 5090, `max-num-seqs` ≤ 4, the *context*
  edition, on a document-reuse pattern (§8's standing caveat).
- The medium-effort break-even (18.2 %) and the 1.224× effort tilt, carrying §5's caveats.
- The input/output *allocation* of the live box-hour (§11.6). The all-token $0.1552/Mtok and the $15.12/box-hour
  are invariant; the split between legs follows from the measured phase rates and is a decomposition, not a
  second measurement.

**Not measured, and not guessed:** any token-level prefix-cache hit rate on DP8; any cached-input rate for
`Qwen3.8-27B` on OpenRouter (not published); 30kx2k on DP8, so no DP8 long-prompt output price appears above;
the pass-set consequences of dropping to medium effort.
