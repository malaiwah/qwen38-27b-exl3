# 48. Overnight plan: the 8x TB2.1 speed run, the topology ladder, and the local-card work

**Written 2026-08-17 for the owner to confirm before sleeping.** Everything below is either already
verified (marked **[verified]**) or is a named step with its own gate. Nothing scored runs without the
five-check fidelity gate passing first, and no number enters a card or doc without its receipt.

---

## 1. Hosts and roles for the night

| host | role | state |
|---|---|---|
| **8x RTX PRO 6000** `151.185.34.106`, VPC `10.0.0.5` | TB2.1 speed run + the 8-GPU topology ladder | 224 vCPU, **1,259 GB RAM**, 915 GB free. **[verified]** driver 595.58.03 on all 8; ForceP2P override applied and `RegistryDwords` confirmed in `/proc/driver/nvidia/params`; topology PHB across all 8, single NUMA; weights staged and **sha256-verified 0 mismatches**; image streaming over VPC |
| **load driver** `151.185.34.98`, VPC `10.0.0.2`, no GPU | harness, campaign orchestration, receipts | **[verified]** VPC reach to `10.0.0.5` (ICMP + ssh); holds `~/research` clone and the sr1 image it built |
| **local 1x RTX PRO 6000** (this container) | vLLM-GG TP1 performance work + the LMCache defect | agent `KernelGap` working two tracks all night |
| **4x** `151.185.34.17` | — | **RELEASE.** All six receipts (DP4/TP4/TP2xDP2 gates+ladders) verified TRACKED in git; box holds nothing else unique |
| **2x** `151.185.34.111` | — | **RELEASE.** LMCache ladder closed; all 266 scored rows + 7 arms committed; GPUs 0 MiB, no containers |

## 2. Serving config, fixed across every 8x arm — CUDA graphs and APC both ON

Both knobs the owner asked about are **already in the shipping config** and will be identical in every
arm, so no arm-to-arm difference can come from them:

- **CUDA graph decode ON**: `VLLM_EXL3_GRAPH_DECODE=1` plus
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`. `FULL_DECODE_ONLY` is not a
  preference - it is **the only mode EXL3 permits** (`exl3.py:1755-1799` refuses any mode whose
  `mixed_mode() != NONE`, which is `PIECEWISE`, `FULL_AND_PIECEWISE` *and* bare `FULL`), because
  prefill row counts are not enumerable before capture. See docs/47 and docs/41.
- **APC ON**: `--enable-prefix-caching --mamba-cache-mode align`. `align` is mandatory with APC on this
  hybrid model; the four-module image carries the #51113 mamba-align scheduler fix that makes it safe.
- fp8 KV; MTP-3 with the **shipped depth schedule `[[1,4,3],[5,64,1]]`** (docs/46 §17); native
  **262,144** window; `--gpu-memory-utilization 0.92`; `--max-num-seqs 64`;
  `--max-num-batched-tokens 8192`.
- **`--kv-cache-memory-bytes` deliberately UNSET** - measured and declined (docs/46 §21: the engine's
  own suggested value will not boot, and 66 GiB boots, passes 4 of 5 gates, then kills the engine on
  the first full-window request).
- **LMCache OFF** - measured and declined (docs/46 §22: our own #403 gate patch takes corruption from
  7/38 to 37-38/38 because it exposes a connector that never restores GDN state).
- **No overlays beyond sr1's three audited patch layers.** Nothing from tonight's kernel work goes near
  the campaign image.

**One deliberate exception, stated so it is not mistaken for an oversight:** prefill chunk stays at
**8192**, not the 6144 that docs/47 measured as a prefill win. That measurement ran with **APC OFF**, and
with APC+align the 1600-token align clipping can cap the effective chunk - so 6144 is *unproven for the
campaign profile*. It gets measured as its own A/B tonight and is not used to serve the scored run.

## 3. The 8x topology ladder (runs first, decides the campaign topology)

Four arms, each **gated on all five checks including the 262,144-token needle** before any rung is kept,
each a single-variable change, same shapes/rungs/seed as every earlier tier so rows drop into the
existing tables:

| arm | flags | prediction from the 1x/2x/4x data |
|---|---|---|
| **DP8** | `--data-parallel-size 8` | best aggregate; DP4 reached 1,958 tok/s at C64, so ~3.4-3.9k if scaling holds at the 86-91 % of linear measured 2x->4x |
| **TP2xDP4** | `--tensor-parallel-size 2 --data-parallel-size 4` | best compromise: TP2 is the *latency* optimum (159.6 single-stream, better than TP4's 151.0) and DP4 multiplies it |
| **TP4xDP2** | `--tensor-parallel-size 4 --data-parallel-size 2` | expected to lose to TP2xDP4, because TP4 already measured worse single-stream than TP2 on PCIe |
| **TP8** | `--tensor-parallel-size 8` | expected worst throughput, best KV capacity (TP4 already gave 8.77 M KV tokens / 33.45x). Run to complete the curve and to bound the all-reduce tax at 7 peers |

Shapes `512x256` and `4kx1k`; rungs `1,2,4,8,16,32,64,128` (the extra rung because DP8 should still be
climbing where DP4 stopped). Readiness is **health + `/v1/models`**, never "Application startup
complete" - with DP>1 the API servers report ready long before the engines finish loading, which is what
made the first DP4 gate fail.

**Selection rule, fixed now so the result cannot be chosen after the fact:** the campaign runs the arm
with the best **per-request** throughput at the campaign's own per-replica concurrency, because
Terminal-Bench is timeout-bound (93 % of pass-1 failures were `AgentTimeoutError`, docs/45). Aggregate
throughput breaks ties. If two arms are within 3 % on both, the simpler one (pure DP) wins.

## 4. The TB2.1 speed run

- 89 tasks, `terminus-2`, stock per-task timeouts, the same pinned task tree
  (`sha256 c13961ac...`) and the same agent/sampling as the published 1x passes, so the wall clock is
  comparable to the 3.3 h pass-1 baseline.
- `tools/tb21_campaign.sh` with the shard-list load balancer: each task pinned to one replica by
  declared timeout budget, so per-replica concurrency is knowable in advance rather than emergent -
  which matters because **the depth schedule keys on requests-per-step per replica** (docs/46 §18/§20).
- `tools/tb21_metrics_poll.py` at a 5 s interval for the whole run, streamed JSONL, summary receipt at
  the end; the reset-aware summariser handles arm swaps.
- Gate before the run and gate after, so a mid-run degradation cannot pass unnoticed.
- Load generated from the **driver VM over the VPC**, which is the shape the campaign was designed for
  and is now **[verified]** reachable.

## 5. Local card, all night (agent `KernelGap`)

- **Track A - make TP1 faster.** P2.2 (delete the 18-26 GB/chunk shard-concatenation copies; `hgemm`
  already supports strided C) and P2.3 (double-buffer the reconstruct scratch so
  `reconstruct[i+1]` overlaps `hgemm[i]`). Same bar as tonight's accepted work: every arm gated on all
  five checks including a full-window needle with the engine alive afterwards, decode and prefill
  measured separately, honest estimate-vs-measured reporting.
- **Track B - the LMCache defect.** The open question is whether `LMCacheMPConnector` *can* store and
  restore the SSM/GDN state block alongside the KV it supplies, the way `nixl` does unconditionally in
  `_apply_prefix_caching`. If the connector API cannot express it, that is a finding worth filing; if it
  can, it is a patch nobody else has.
- Already banked tonight from this card: the **+15.4 %** b12x gate A/B, the **+8.0 %** `in_proj_ba`
  transpose fix, chunk-6144 requalification, and F11 (the served `exl3.py` has **no public ancestor**).

## 6. Also queued on the 8x

- **The bare-metal four-arm A/B** - baseline / b12x-gate / `in_proj_ba` / both - on bare docker with no
  proot, which is what separates the eliminated Python dispatch from the kernel delta and turns the
  +3-15 % bracket into a number. This is the gate on shipping either patch.
- **Charts for the cards** (§7).

## 7. Charts and plots for the HF cards

Built with the repo's existing chart conventions (`tools/make_knee_chart.py`, `make_kld_comparison_chart.py`):

1. **Topology ladder** - per-request and aggregate tok/s against concurrency, one series per arm across
   1x, TP2, DP2, DP4, TP4, TP2xDP2, and tonight's four 8-GPU arms. This is the chart that answers "what
   should an API host actually run".
2. **Scaling efficiency** - aggregate throughput against GPU count at matched *per-replica* load, with
   the linear reference drawn, so the 86-91 %-of-linear result is visible rather than asserted.
3. **TB2.1 speed run** - wall clock and resolved-count against tier, with the 1x 3.3 h pass-1 baseline
   as the left-hand anchor.
4. **Decode roofline** - measured achievable ceiling (1462-1525 GB/s) against the vendor spec and
   against where we actually run, which is the honest replacement for every "55-65 % of roofline"
   sentence we published.

Charts land in `assets/` light+dark SVG+PNG, cited from the cards with the receipt behind each number.

## 8. Failure and stop conditions while the owner sleeps

- **Any gate FAIL: do not score.** Investigate, record, and if unresolved fall back to the
  last-known-good topology rather than publishing an ungated run.
- **Needle FAIL specifically** is treated as the docs/46 §21 trap: the config is rejected outright, not
  tuned around.
- A subagent stuck on one step past an hour gets nudged, then cancelled with partial evidence landed.
- Disk below 50 GB on any host: stop staging, prune, report.
- **Nothing gets published to a card that has not passed its gate**, and no estimate is reported as a
  measurement. Every open question ends the night written down rather than guessed.

## 9. Owner decision 2026-08-17: 2x timeouts, and why that makes it a better experiment

The owner confirmed this run is **not an official submission**, and authorised **doubling the per-task
timeouts** (`--timeout-multiplier 2.0`). This is more than a convenience, and it must be reported
carefully.

**Why it is the right experiment to run tonight.** The single strongest finding of the 1x passes is that
this result is **timeout-bound, not quality-bound**: of 58 unresolved pass-1 tasks, **54 ended in
`AgentTimeoutError`** and only 2 ran to completion and answered wrong (docs/45), and the BF16 control
timed out on the same scale - 35 of 45 twice-failed tasks landed in `inconclusive-timeout` because
*neither arm ever finished*. A 2x-timeout arm converts that observation into a measurement: it asks
directly **how much of the 31/89 was the clock**. If the score rises materially, the published number is
a serving-budget artefact as we argued; if it barely moves, our timeout explanation was wrong and the
tasks are genuinely beyond the model. Either answer is worth more than another stock-timeout run.

**How it will be reported, so it cannot be confused with the published score.** The 2x-timeout run is a
**separate, explicitly labelled arm**. It does **not** revise 31/89, 44/89, or any attribution bucket -
those stay exactly as published, measured at stock timeouts, because a benchmark number is only
comparable against runs sharing its budget. Any card text will name the multiplier in the same sentence
as the number, and the receipt records `timeout_multiplier: 2.0` as a first-class field. Terminal-Bench's
own leaderboard rules are the reason this cannot be presented as a headline: **we are deliberately
outside them**, and saying so is the whole point.

**Second-order consequence worth stating up front:** doubling timeouts lengthens the tail, so the
campaign's wall clock will not be comparable to the 3.3 h pass-1 baseline either. The speed-run number
and the score number therefore come from different arms tonight: **wall clock from a stock-timeout arm,
score-vs-clock from the 2x arm.** Mixing them would produce a figure that describes no real
configuration.
