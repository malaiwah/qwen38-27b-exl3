# 44. Handoff: state, running work, and the exact resume command per open item

**Refreshed 2026-08-17 ~13:40Z, mid production-serving window.** Written so whoever picks this up — the
owner, or a fresh agent after a rental dies — can resume every open thread without reading the transcript.
Every claim here is checkable from a committed receipt.

---

## 1. Hosts, right now

| host | role | state |
|---|---|---|
| `jl-vm-473501` (id 473501), 8× RTX PRO 6000, VPC `10.0.0.5`, ssh `ubuntu@151.185.34.106` | **production serving** | `qwen38-dp8-1m` container, DP8, **1,000,000-token window via static YaRN**, ForceP2P on. `/health` 200. Serving real omp traffic + a TB2.1 arm. |
| `jl-vm-473296` (id 473296), CPU-only, VPC `10.0.0.2`, ssh `ubuntu@151.185.34.98` | load driver | `omp` live in **tmux session `qwen38`** (attach: `tmux attach -t qwen38`). Runs the TB2.1 1M arm (8 harbor jobs) + 8 metrics pollers + a metrics puller. |
| `main-omp-session` (id 471041), 1× RTX PRO 6000 | this workstation | GPU **idle, 0 MiB**, torn down cleanly by agent YarnPenalty. **Not free — bills $1.904/hr.** |

**Billing.** Balance was topped to **$114.42** after nearly running dry (it hit $19.88 against $18.13/hr =
~66 min, and JarvisLabs' FAQ deletes data at zero balance). Burn: 473501 **$15.29/hr**, 471041 $1.90/hr,
473296 $0.94/hr. Multi-GPU is **strictly linear at $1.89/GPU-hr, no volume discount** —
see `docs/49-jarvislabs-pricing-and-inventory.md`.

## 2. Running work

| what | where | state / how to check |
|---|---|---|
| **TB2.1 1M arm** — measures task-level YaRN degradation | driver VM, 8 shards × n=2 (C16), `TB21_MAXLEN=1000000`, `TB21_TIMEOUT_MULT=2.0` | ~38/89 results, 15 resolved at 13:38Z. `find ~/tb21/jobs/tb21c-8x1m-hyd-p1-s*/ -name result.json \| wc -l` |
| **TokenPricing** subagent | local | appending USD/Mtok (input/output/cached) + break-even utilisation to `docs/50-serving-cost-model.md` |
| GPU telemetry + metrics archival | 8× → driver | `~/metrics/gpu-telemetry.csv` (5 s), rsync'd to driver `~/metrics/8x/`, 8 vLLM pollers at 5 s |

## 3. The serving answer, settled

**DP8 for a fleet, TP2×DP4 for a solo stream, and TP8 does not exist.**

| topology | GPUs | single-stream | peak agg | knee |
|---|---:|---:|---:|---:|
| 1× | 1 | 136.0 | 464 @C8 | C4 |
| TP4xDP2 | 8 | 146.5 | 2,206 @C128 | C8 |
| TP2xDP4 | 8 | **153.2** | 2,438 @C128 | C16 |
| **DP8** | 8 | 135.2 | **3,553 @C128** | **C32** |
| ~~TP8~~ | 8 | **refused at load** | — | — |

- **Knee = 4 × data-parallel degree**, proven on one host across three arms (C32/C16/C8). TP width *lowers*
  the knee, it does not raise it.
- **TP8 is architecturally impossible**: EXL3 shards trellis tensors on 128-element boundaries and `lm_head`
  is the padded vocab 248,320 = 128 × 1940 with 1940 = 2²×5×97. **Max TP width is 4 on any GPU count.**
- **Cheapest tokens are ONE GPU** (0.707–0.840 GPU-h/Mtok vs DP8's 1.141–1.383): price is linear, scaling is
  sub-linear (5.889× on 8 GPUs at 512×256). Multi-GPU is a **latency-and-concurrency purchase**.
- **Largest cost lever is not topology**: `reasoning_effort` xhigh vs medium is **6.44× $/task**.

## 4. The 1M verdict, and the standing recommendation

**1M works.** Needle retrieval PASSES at 262k / 524k / 786k / **994,755 tokens (99.5 % of a 1M window)**.
KV is not the constraint — only 16 of 64 layers carry KV, so fp8 KV is 32 KiB/token and the engine reports
**1,879,687 KV tokens (62.45 GiB) per GPU**; raising the window did not shrink the pool.

**But serving YaRN by default taxes every real request.** `docs/46 §31` decomposed it on a quiet card:

| arm | mean top-20 KLD | reading |
|---|---:|---|
| same-server replicate | **0.0e+00** | bit-identical at concurrency 1 |
| cross-boot replicate | 1.180e-03 | the resolution floor |
| **window only** (native rope @ 1M) | **1.248e-03** | **the window is fidelity-free** |
| **rope only** (matched 1M window) | **1.071e-02** | **the rope costs all of it** |
| native-262k vs 1M-YaRN | **1.057e-02** | ~9× floor; 19/48 prompts change their greedy output |

**RECOMMENDATION: run 262k native by default; add a separate 1M-YaRN endpoint only for requests that
exceed the native window.** Not yet actioned because the TB2.1 1M arm needs the current config to finish.

## 5. Open items and the exact resume command

1. **Run TB2.1 at 1M to measure YaRN degradation** — *running*. On completion:
   `cd ~/research && tools/tb21_campaign.sh merge /tmp/tb21-8x1m-p1.json ~/tb21/jobs/tb21c-8x1m-hyd-p1-s{0..7}`
   then compare against the 262k arm's **56/89** (`receipts/tb21-8x-2xclock-arm.json`). Note the confound:
   this arm is C16 where that one was C32, and *less* contention should help, so a lower score is real signal.
2. **Measure KLD on DP8 at 262k native during the config swap** — needs the swap window. Design is in
   `docs/29` ("two fidelity measurements"). Reuse the **committed** 48-prompt probe set
   (`receipts/yarn-short-context-raw/`); do not regenerate it or comparability with §31 is lost.
   `tools/yarn_probe_run.py` per condition, then `tools/yarn_penalty_analyze.py --work <dir>`.
3. **Quantify whether concurrency itself costs fidelity** — same window, step 3 of that plan: probe quiet,
   then under load, at fixed 262k-native rope. **Open and consequential**: if concurrency costs real KLD,
   every throughput number in this project has an unquoted fidelity price.
4. **Recommend the most cost-effective serving config** — TokenPricing agent is finishing it into `docs/50`.
5. **Refresh plan and push all evidence** — this document plus `docs/29`; final verification is
   `git status --porcelain` empty, both remotes at HEAD, `tools/publish_cards.py` `all_byte_identical: true`,
   and `tools/sync_card_assets.py` exit 0.

**Blocked (1):** *Bare-metal 4-arm A/B of gate and in_proj_ba patches.* Arms A and B ran
(`receipts/kernel-gap-bare-metal-ab.json`); C and D did not. Two reasons: no quiet bare-docker host (the 8×
is production, and 471041 has no bare docker — proot inflates dispatch, which is why bare metal was required),
and **the method is superseded** — §28 measured within-arm CV at 2.84 % median (up to 18 %) against an effect
Amdahl bounds at +4.19 %, so the ladder cannot resolve it. Re-scope to kernel-level timing + an Amdahl bound
before spending GPU time. Images `ab-c-transpose` / `ab-d-both` are built and byte-verified on the 8×.

## 6. Rules learned the hard way (do not re-derive these)

1. **`publish_cards.py` uploads only README.md.** Card byte-identity does *not* imply the figures exist on
   the Hub — six referenced assets were missing and rendered broken. Run `tools/sync_card_assets.py` after any
   figure change. (`receipts/card-asset-publication-defect.json`)
2. **A hash-equality check must assert it hashed something.** Our repeatability gate compared eight *empty
   strings* and reported PASS on every published run, because this model thinks by default and the 64-token
   probe cap sent everything to `reasoning_content`. (`receipts/gate-check3-vacuous.json`)
3. **`--tp-smoke` silently downgrades the gate needle from 262,144 to 32,768.** Never compare a tp-smoke gate
   against a full-needle gate.
4. **One instantaneous `nvidia-smi` sample cannot distinguish a load-balance defect from workload shape.** An
   idle GPU under the needle ladder looked exactly like DP imbalance; time-segmenting the telemetry showed the
   ladder issues one giant request at a time. (§32)
5. **Prefill is super-linear with a *rising* exponent** (k = 1.506 → 1.744 → 2.144). A single global exponent
   under-predicts the tail — mine missed the 1M point by 15.5 %. A 1M prompt costs **2.53× more per token**.
6. **ForceP2P ships OFF** on these hosts (35.5 → 51.9 GB/s, +46.5 %). Standing pre-measurement action on every
   new multi-GPU rental.
7. **LMCache stays disabled**: the dominant defect is fp8-KV transfer (bit-clean at bf16), and our qualified
   default *is* fp8 KV, so it could never have produced clean reuse.
8. `pkill -f` self-matches over ssh — use bracket patterns `"[h]arbor"`. Foreground `docker run` over ssh dies
   with the session — use `setsid nohup … &`. With DP>1, "Application startup complete" is **not** readiness —
   poll `/health` **and** `/v1/models`.
