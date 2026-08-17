# 44. Handoff: state, running work, and the exact resume command per open item

**Refreshed 2026-08-17 during the overnight 8x campaign.** Written so that whoever picks this up - the
owner in the morning, or a fresh agent after a rental dies - can resume every open thread without
reading the transcript. Every claim here is checkable from a committed receipt.

---

## 1. Hosts, right now

| host | role | state |
|---|---|---|
| **8x RTX PRO 6000** `151.185.34.106` / VPC `10.0.0.5` | topology ladder + TB2.1 campaign | live. 224 vCPU, 1,259 GB RAM, 887 GB free. Kernel `6.8.0-137` after a planned reboot, driver **595.58.03**, ForceP2P override applied and **functionally verified** (P2P on, 51.85 GB/s). sr1 image loaded, weights sha256-verified (0 mismatches), harness + 4 arm scripts in `~/bin` |
| **load driver** `151.185.34.98` / VPC `10.0.0.2` | harness, receipts | live. Reaches the 8x at **ping, ssh, arbitrary TCP (1.39 ms connect) and the vLLM endpoint (`health=200`, `/v1/models`)** |
| **local 1x RTX PRO 6000** (this container) | kernel/perf + LMCache work | busy: agent `KernelGap` |
| 4x `151.185.34.17`, 2x `151.185.34.111` | — | **released** 2026-08-17 after verifying every receipt was committed |

**Standing action for any new multi-GPU rental:** write `/etc/modprobe.d/nvidia-p2p-override.conf` and
reload the driver **before the first measurement** - worth 7-22 % of decode (docs/46 §19). Both the vLLM
container and `nvidia-dcgm`'s `nv-hostengine` pin `nvidia_uvm`, so stop both or `modprobe -r` fails while
appearing to succeed.

## 2. What is measured and settled (do not redo)

- **Serving topology, 1x through 8x**, all gated PASS incl. the 262k needle: docs/46 §14-24. Headlines:
  DP wins throughput (**DP8 peak 3,553 tok/s at C128**, knee **C32** - a DP-N deployment holds its
  per-request floor to about **4N** concurrent requests); **TP2 is the TP latency optimum** and TP4 is
  *worse* than TP2 single-stream; **TP2xDP2 took the best single-stream of the 4-GPU arms** (156.7).
- **MTP depth schedule `[[1,4,3],[5,64,1]]`** shipped (docs/46 §17), and docs/47 F9 mechanises it: each
  depth level costs 1.22 GB/step, **78 % of it `lm_head`**.
- **The decode "roofline gap" was mostly arithmetic** (docs/47): numerator 20.5 GB/step not ~17.4,
  denominator 1462-1525 GB/s measured not 1792 spec -> decode runs at **~85 % of achievable**. The
  1.792 TB/s figure is **banned as headroom** in future claims.
- **LMCache: DO NOT ENABLE** (docs/46 §22, and docs/47 F12 for the corrected mechanism). Our own #403
  gate patch moves corruption 7/38 -> **37-38/38**; the real defect is **store-side** (null mamba block
  ids stored as boundary state under valid keys). PR #403 has been publicly re-labelled.
- **`--kv-cache-memory-bytes`: leave unset** (docs/46 §21) - the engine's own suggestion won't boot, and
  66 GiB boots, passes 4/5 gates, then kills the engine on the first full-window request.
- **`reasoning_effort` accepts only `xhigh`/`medium`/`low`** - `none` is rejected; cards corrected. xhigh
  costs **5.3x the tokens** of medium for the same answer.
- **TB2.1 published**: 31/89 pass 1 (both denominators disclosed), 44/89 best-of-2 as a **permanent**
  lower bound, and a three-way attribution where only **8 of 45** persistent failures are attributable.
- **Seven vLLM-GG issues filed: #406-#412.** One index page was mis-filed as #405 and closed within
  minutes; recorded in `receipts/vllm-gg-issues-filed.json`.

## 3. Open items and the exact resume command

| item | resume |
|---|---|
| **8x topology ladder** (DP8 done; TP2xDP4 running; TP4xDP2, TP8 to go) | on the 8x: `docker rm -f <prev>; setsid nohup ~/bin/serve-<arm>.sh vllm:gg-r34-tb21-sr1 > ~/logs/serve-<arm>.log 2>&1 &` then from the driver VM `cd ~/research && python3 tools/tb21_gate.py --base-url http://10.0.0.5:8000 --out ~/tb21/receipts/tb21-gate-8x-<arm>.json && python3 tools/tb21_ladder.py --base-url http://10.0.0.5:8000 --shapes 512x256,4kx1k --rungs 1,2,4,8,16,32,64,128 --out ~/tb21/receipts/tb21-ladder-8x-<arm>.json` |
| **TB2.1 campaign on the 8x** | pick the winning arm by the §3 rule in docs/48, serve it, gate it, then `tools/tb21_campaign.sh` from the driver VM with `TB_TIMEOUT_MULT=2.0`; poller: `tools/tb21_metrics_poll.py --base-url http://10.0.0.5:8000 --out receipts/tb21-metrics-8x.jsonl --interval 5` |
| **Bare-metal 4-arm patch A/B** (baseline / b12x-gate / in_proj_ba / both) | on the 8x, single GPU, bare docker: mount `receipts/kernel-gap-gate-ab.patch` and the ba patch as read-only overlays over `exl3.py` / `linear.py`, `--gpus '"device=0"'`, then measure single-stream greedy decode. Turns KernelGap's **+3-15 %** bracket into a number |
| **Four card charts** | `tools/make_knee_chart.py` conventions; sources are `receipts/tb21-ladder-{1x,2x,4x,8x}-*.json`. Tiers must show **measured only - 4x TB stays blank** (docs/46 §23-24) |
| **Alt-calibration experiment** (blocked by owner directive until the serving investigation closes) | packet at `/var/tmp/work/kld9` on this box: `tools/ggrun.sh bash /work/kld9/chain_kld9.sh altcal` then `tools/kld9_receipt.py --cond altcal`. Gates the docs/29 F2 attribution (6-bit deficit at equal body bytes: allocation shape or calibration content?) |

## 4. Running agents

- **`KernelGap`** (local card, all night). Track A adjudicated: **shipped-class** = b12x gate clause
  (+3-15 % bracket) and `in_proj_ba` transpose (+8.0 %) and chunk-6144 (+8-9 % PP at 197k+);
  **nulled** = reconstruct double-buffer (store-bandwidth-bound, overlap re-divides the same bytes);
  **marginal-archived** = shard-cat deletion (+0.8-1.9 %, below its own 2 % bar). Track B: LMCache
  store-side defect traced to `lmcache_mp_connector.py:372-374`, **~20-line fix identified**, live
  three-step repro in progress.
- Message any agent with `hub send`; transcripts at `history://<id>`.

## 5. Rules that cost something to learn

1. **A gate PASS is not fidelity when a KV connector is attached.** Five checks passed against a server
   simultaneously scoring 38/38 corrupted. Check 6 (`--check-connector-reuse`) exists now, off by
   default, **mandatory whenever `--kv-transfer-config` is present**.
2. **With DP>1, "Application startup complete" is not readiness** - it comes from the API servers before
   engines finish loading. Poll `/health` plus `/v1/models`.
3. **Foreground `docker run` over ssh dies with its session** (and with a daemon restart). Launch
   `setsid nohup ... &`.
4. **A cross-tier ratio is only evidence if both arms ran on the same host through the same harness
   path.** Ours did not; that is why no scaling law is published.
5. **Only removing bytes pays in a bandwidth-contended regime**, never re-scheduling them (docs/47 P2.3).
6. **We share one working tree with subagents** - use `git add <paths>`, never `git add -A`.
