# 44. Handoff: state, running work, and the exact resume command per open item

**Refreshed 2026-08-17 ~17:0xZ, after the production-serving window closed.** Written so whoever picks this up
— the owner, or a fresh agent after a rental dies — can resume every open thread without reading the
transcript. Every claim here is checkable from a committed receipt.

**Task list state: 100 of 102 done, 1 blocked with a written reason, 1 dropped by owner decision.**

---

## 1. Hosts, right now

| host | role | state |
|---|---|---|
| `jl-vm-473501` (id 473501), 8× RTX PRO 6000, VPC `10.0.0.5`, ssh `ubuntu@151.185.34.106` | **production serving** | container `qwen38-dp8`, **DP8 at 262,144 native** (migrated from 1M-YaRN on measured evidence, §34). `/health` 200, `max_model_len` 262144 verified. Idle. |
| `jl-vm-473296` (id 473296), CPU-only, VPC `10.0.0.2`, ssh `ubuntu@151.185.34.98` | load driver | `omp` live in **tmux `qwen38`** (`tmux attach -t qwen38`), repointed to 262k (`omp models` shows 262K). Metrics puller running. TB2.1 arms finished. |
| `main-omp-session` (id 471041), 1× RTX PRO 6000 | this workstation | GPU idle, 0 MiB. **Not free — bills $1.904/hr.** |

**Billing.** Balance was topped to $114.42 after nearly hitting zero (it reached $19.88 against $18.13/hr ≈
66 min, and JarvisLabs' FAQ **deletes data** at zero balance). Burn: 473501 $15.29/hr, 471041 $1.90/hr,
473296 $0.94/hr. **Multi-GPU is strictly linear at $1.89/GPU-hr, no volume discount** (`docs/49`).

## 2. Running work

**None.** No subagents, no campaigns, no probes. Everything dispatched this session has landed and is committed.

## 3. The serving answer, settled

**DP8 for a fleet, TP2×DP4 for a solo stream, TP8 does not exist, and 262k native for the window.**

| topology | GPUs | single-stream | peak agg | knee |
|---|---:|---:|---:|---:|
| 1× | 1 | 136.0 | 464 @C8 | C4 |
| TP4xDP2 | 8 | 146.5 | 2,206 @C128 | C8 |
| TP2xDP4 | 8 | **153.2** | 2,438 @C128 | C16 |
| **DP8** | 8 | 135.2 | **3,553 @C128** | **C32** |
| ~~TP8~~ | 8 | **refused at load** | — | — |

- **Knee = 4 × data-parallel degree**, proven on one host across three arms. TP width *lowers* the knee.
- **TP8 is architecturally impossible**: `lm_head` is padded vocab 248,320 = 128 × 1940, 1940 = 2²×5×97, so max
  TP width is **4 on any GPU count**.
- **Cheapest tokens are ONE GPU** (0.707–0.840 GPU-h/Mtok vs DP8's 1.141–1.383). Multi-GPU is a
  **latency-and-concurrency purchase**, not a cost-per-token one.
- **Output tokens cost 8–20× input** ($0.879 vs $0.111/Mtok at the 1-GPU knee). DP8 with one request in flight
  is **$43.03/Mtok** — eight GPUs billed for one stream.
- **Break-even occupancy** vs OpenRouter list for this model ($0.40/$3.00): **76.3 %** on-demand DP8, 48.0 %
  reserved-1y, **28.7 %** on a single GPU. Against commodity 32B pricing ($0.08/$0.28), self-hosting **cannot
  win at any occupancy** (`u* = 495 %`) — the reasons to self-host are the ones cost cannot express.
- **Largest single lever is not topology**: `reasoning_effort` xhigh vs medium is **6.44× $/task**.

## 4. Context: 262k native, and why

| arm | mean top-20 KLD | reading |
|---|---:|---|
| same-server replicate, concurrency 1 | **0.0e+00** | engine is deterministic alone |
| cross-boot replicate | 1.180e-03 | resolution floor |
| **1M window only** (native rope) | **1.248e-03** | **the window is FREE** |
| **static YaRN rope**, matched window | **1.071e-02** | **the rope costs all of it** |
| 262k vs 1M-YaRN, 1 GPU (§31) | 1.057e-02 | ~9× floor, 19/48 prompts change output |
| 262k vs 1M-YaRN, **DP8** (§34) | **1.0265e-02** | **replicates within 3 %** |
| **quiet vs LOADED, DP8** (§34) | **1.278e-03** | **1.08× floor — concurrency is fidelity-neutral** |

Retrieval at 1M works (needle at **994,755 tokens = 99.5 %** of window passes). Prefill is super-linear with a
**rising** exponent (k = 1.506 → 1.744 → 2.144), so a 1M prompt costs **2.53× more per token**.
Task-level corroboration, weaker: TB2.1 at 1M-YaRN scored **48/89** vs 262k's **56/89** (§33, sign-test
p = 0.096 — corroborates, does not prove).

## 5. Open items

**One blocked (1 of 102):** *Bare-metal 4-arm A/B of gate and `in_proj_ba` patches.* Arms A and B ran
(`receipts/kernel-gap-bare-metal-ab.json`); C and D did not. Two reasons: **no quiet bare-docker host** (471041
has only proot, which inflates dispatch — the reason bare metal was required), and **the method is superseded** —
§28 measured within-arm CV at 2.84 % median (up to 18 %) against an effect Amdahl bounds at +4.19 %, so the
ladder cannot resolve it. **Re-scope to kernel-level timing + an Amdahl bound before spending GPU time.** Images
`ab-c-transpose` / `ab-d-both` are built and byte-verified on the 8×; `bash /tmp/run_ab.sh` with `run_arm C` /
`run_arm D` uncommented.

**One dropped by owner decision:** re-run `build-pov-ray` to void a 404 — 44/89 stands as the permanent lower bound.

## 6. Rules learned the hard way (do not re-derive these)

1. **`publish_cards.py` uploads only README.md.** Card byte-identity does *not* imply the figures exist on the
   Hub — six referenced assets were missing and rendered broken. Run **`tools/sync_card_assets.py`** after any
   figure change. (`receipts/card-asset-publication-defect.json`)
2. **A hash-equality check must assert it hashed something.** Our repeatability gate compared eight *empty
   strings* and reported PASS on every published run, because this model thinks by default and the 64-token probe
   cap sent everything to `reasoning_content`. Now has a vacuity guard. (`receipts/gate-check3-vacuous.json`)
3. **An archival mirror without a provenance banner is indistinguishable from an unattributed re-upload.** Four
   of seven lacked one. **`tools/fix_mirror_provenance.py`** fixes and verifies.
4. **`--tp-smoke` silently downgrades the gate needle from 262,144 to 32,768.** Never compare across it.
5. **One instantaneous `nvidia-smi` sample cannot distinguish a load-balance defect from workload shape.** An
   idle GPU under the needle ladder looked exactly like DP imbalance; time-segmenting the telemetry showed the
   ladder issues one giant request at a time. (§32)
6. **A single global exponent under-predicts a rising one.** My registered 1M prefill prediction missed by 15.5 %
   because k is not constant.
7. **ForceP2P ships OFF** on these hosts (35.5 → 51.9 GB/s, +46.5 %). Standing pre-measurement action.
8. **LMCache stays disabled**: the dominant defect is fp8-KV transfer (bit-clean at bf16), and our qualified
   default *is* fp8 KV.
9. `pkill -f` self-matches over ssh — use `"[h]arbor"`. Foreground `docker run` over ssh dies with the session —
   use `setsid nohup … &`. With DP>1, "Application startup complete" is **not** readiness — poll `/health` **and**
   `/v1/models`.
10. **Verify a peer review, do not trust it — and do not dismiss it either.** A review generated by our own
    served model was accurate and *conservative*: two of its findings were understated, and one (`P5`, non-atomic
    receipt writes in `finalize_checkpoint.py`) was a real bug, not a wording gap.
