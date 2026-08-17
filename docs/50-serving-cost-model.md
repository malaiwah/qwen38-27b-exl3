# Serving cost model — `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` for an oh-my-pi multi-subagent coding workload

**Denomination: GPU-hours per million *output* tokens (GPU-h/Mtok).** Multiply any cell by your
`$/GPU-hr` to get `$/Mtok`. No prices appear in this document by design — the pricing slice owns those.

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
