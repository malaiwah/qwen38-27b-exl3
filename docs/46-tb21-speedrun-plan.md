# 46. TB2.1 speed run on a multi-GPU Jarvis host: the execution plan

**Status: PLAN, written 2026-08-16 for an independent executor session. Nothing in this document has
been run on the target host. Every number labelled *measured* has a receipt in this repo; every number
without one is arithmetic or a target, labelled as such. The executor runs with a different model and
zero conversation context: this document is the entire brief.**

## 0. The objective, in one paragraph

Given a 4x (stretch: 8x) **RTX PRO 6000 Blackwell 96 GB** host from Jarvis AI plus a **separate load-driver
VM**, deliver the fastest honest **Terminal-Bench 2.1** run for two models — the original weights
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (BF16, ~52 GiB weights) and our flagship quant
[`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)
(20.31 GiB resident, mean KLD 0.002760 vs BF16) — and produce a card-referenceable ladder:
**quick 1x, 2x, 4x, and a full speed run** at the best measured configuration. The **native 262,144-token
window is exercised first**; the YaRN-extended **1M window is exercised after** if time allows. Parallelism
to prove out: **TP and DP. EP is not applicable** — this is a dense hybrid (48 GDN linear-attention + 16
full-attention layers); there are no experts to shard.

Two headline metrics per tier, never merged: **resolved / 89** and **wall-clock for the pass**, at stock
1.0 timeouts. Faster serving raises both (measured mechanism: timeouts dominate failures — see §2).

## 1. What is already established (do not re-derive)

All measured, receipts in this repo:

| fact | value | receipt |
|---|---|---|
| TB harness | harbor 0.21.0 + terminus-2 2.0.0, dataset `terminal-bench-2-1@6`, 89 tasks, task tree sha-pinned | `receipts/terminal-bench-2.1-pins.json`, `-task-inventory.json` |
| TB scores an agent+model **system** | framing fixed; stock `--timeout-multiplier 1.0` is the only comparable headline | `docs/45-terminal-bench.md` |
| Timeouts dominate at ~42 tok/s | at 25/89: 11 of 17 unresolved were `AgentTimeoutError`; turn-1 reasoning bursts up to **15,577 completion tokens** | pass-1 receipt (in flight), `history://TerminalBench2` |
| Single-stream decode, TB profile (96 GB card, **enforce-eager**, 32k window) | **40.8–42.5 tok/s**; C16 aggregate **516 tok/s**, per-request 32.3 | TB2 headroom block, pass-1 receipt |
| Eager vs CUDA-graph decode | forced eager **loses 48 % of decode** — the single biggest serving lever this plan turns on | `receipts/perf-sweep-5090.json`, cards §Serving |
| MTP-3 acceptance on TB traffic | 62.6 % in-interval; per-position 98.7 / 81.0 / 66.4 % | TB3 measurement, adopted in pass-1 receipt |
| Prefix caching on agent traffic | **51.7 % cumulative hit rate**; `prompt_tokens_details` is null on the response path, so client-side `cached_tokens` reads 0 regardless | pass-1 receipt |
| KV cost, fp8, this architecture | **~36.7 KiB/token** (9.28 GiB pool = 265,122 tokens at the 262k qualification); read the engine line per config, never assume | `receipts/qualification-24gib-capped.json` family |
| Native 262,144 on one 96 GB card | qualified with MTP-3 (capped 30.24 GiB profile: 98.7 tok/s, exact needle at 258,925 tokens) | `receipts/qualification-*`, context card |
| V2 runner + per-batch-size MTP depth schedule | **+38.0 % aggregate at C8** (416 vs 302 tok/s matched); needs fork fix commit `5723e072e` (PR [#398](https://github.com/local-inference-lab/vllm/pull/398)); costs headroom — OOMed at util 0.97 on 32 GB | `receipts/v2-fault-fix.json`, docs/41 W1 |
| Scratch-arena overlay | +6.7 % KV pool (+17,874 tokens at 262k), opt-in, byte-identical probe outputs | `receipts/scratch-arena.json`, PR #397 |
| int8 embed overlay | +1.8–2.1 % KV, **changes greedy output** — OFF for this exercise | `receipts/embed-overlay-8k.json` |
| LMCache | **PROHIBITED** — silent corruption reproduced 38/38 on restart-over-warm-L2; upstream LMCache#4247/#4492 | `receipts/lmcache-reuse-test.json` |
| Parser tolerance | 40 % of trials emit prose before the JSON object; terminus-2 tolerates it; a stricter parser would score lower — carry the count | pass-1 receipt |
| Cross-restart nondeterminism | greedy text differs across engine restarts (7/8), identical within a process (8/8) — all comparisons within-process or paired | `receipts/scratch-arena.json` |

Serving stack: the promoted release unit `localhost/vllm:gg-r34-patched-apc` (vLLM-GG fork
`0.11.2.dev280+gilded.gnosis.v20…r34`, carries the APC scheduler fix; manifest `sha256:16a936b8…`, also
running as the owner's service at digest `sha256:820181fb…`). Ship it with `podman save | zstd` from AIBoss
or the rental, verify the digest on arrival, never rebuild.

## 2. Why this is a *serving* problem: the timeout mechanism

At stock 1.0 timeouts the binding constraint is **per-request decode speed**. Measured: reasoning bursts of
~15k completion tokens at ~42 tok/s ≈ 6 min for one turn; task budgets are minutes-scale; 65 % of unresolved
trials ran out of clock, not ability. Every serving improvement below therefore converts directly into both
a faster wall clock *and* a higher resolved count. The three levers, in measured order:

1. **CUDA-graph decode ON** (`VLLM_EXL3_GRAPH_DECODE=1`, `FULL_DECODE_ONLY`): the TB pins deliberately used
   `--enforce-eager`; this plan deliberately does not, and states the deviation. Expected roughly +2x
   single-stream (eager loses 48 %).
2. **Concurrency shape**: DP replicas with task-sharded affinity (§5) rather than one big batch — per-request
   speed is what beats the timeout, so do not trade it for aggregate beyond the headroom rule.
3. **MTP-3 static** (qualified default). Optional labelled arm: V2 + depth schedule `[[1,2,3],[3,8,1]]`
   (+38 % C8 measured on a 32 GB card) — worth trying at 96 GB where its headroom cost is absorbed, gated in
   §7.

## 3. Hardware and placement

- **Serving host (Jarvis):** 4x (or 8x) RTX PRO 6000 Blackwell 96 GB. Record `nvidia-smi topo -m` and driver
  version in the receipt; pin one replica per GPU with `CUDA_VISIBLE_DEVICES`.
- **Load-driver VM (separate, same DC):** runs harbor + terminus-2 + all task containers, CPU-only.
  Spec: **32–64 vCPU, 128 GB RAM, 250 GB NVMe** (task images project to ~42 GB + workdirs + image cache),
  rootless podman ≥ 4.9, static docker CLI + compose on the podman socket, harbor 0.21.0, the pinned task
  tree. This mirrors the proven AIBoss↔rental split.
- **Network gate G-NET:** measure driver→host TCP RTT and TTFB through the exact serving path before
  anything else. Requirement: **RTT < 10 ms** (LAN/VPC). Our previous 581 ms was geography and cost nothing
  to attribution but did cost per-turn latency; on a speed run it would be ~0.6 s x thousands of turns. If a
  tunnel is unavoidable, use the proxy-owned-port pattern (`tools/tb_tunnel_proxy.py`) with
  `ControlMaster=no ControlPath=none` and self-unlinking sockets; otherwise plain VPC + `--api-key`.
- Throughput is always read from vLLM `/metrics` on the serving host loopback, **as deltas between named
  snapshots** (counters are contaminated by calibration by design; the delta discipline is mandatory).

## 4. Serving configurations

### 4.1 Base recipe (both models, all tiers)

From the qualified family recipes — deviations from the TB2 pins are deliberate and must be listed in the
receipt (`enforce-eager` -> graph decode; 32k -> native window):

```
vllm serve <model> --served-model-name qwen38
  --max-model-len 262144            # native window FIRST, per the owner
  --kv-cache-dtype fp8              # family default, both arms, stated
  --mamba-cache-mode align
  --enable-prefix-caching           # 51.7 % measured hit rate on TB traffic
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --max-num-seqs 16                 # per replica; revisit via headroom rule
  --max-num-batched-tokens 8192     # sweep {2048, 8192} in G1; align-safe (#51113 in image)
  --gpu-memory-utilization 0.92     # 96 GB card; raise only after G1 headroom read
  --api-key <key> --disable-log-requests
env: VLLM_EXL3_EXT_PATH=... VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128   # quant arm only
```

Quant arm adds `--quantization exl3 --quantization-config qcfg.json` exactly as the hydrated card's serving
section. BF16 arm uses the same flags minus the EXL3 env/flags. **No LMCache. No int8 embed overlay.**
Optional labelled overlays, each its own arm, never silently mixed: scratch arena (PR #397 file mount),
V2 depth schedule (PR #398 commit `5723e072e` overlay + `VLLM_USE_V2_MODEL_RUNNER=1`).

Capacity arithmetic at 36.7 KiB/token (verify per config from the engine's KV line):
one 96 GB card, quant: ~65–70 GiB KV ≈ **1.8–1.9 M KV tokens** — the full native window for ~7 concurrent
262k requests, or 16 agent sessions at ≤32k with room to spare. BF16 on one card: ~52 GiB weights leaves
~30–35 GiB ≈ 850k–1M KV tokens. TP pools KV across cards; DP multiplies pools.

### 4.2 Parallelism arms

| arm | what | gate | expectation to test |
|---|---|---|---|
| TP=1 DPk | k independent servers, one per GPU, task-sharded (§5) | none — proven pattern | best TB throughput/`$` for the quant; linear scaling |
| TP=2 / TP=4 | one server across 2/4 GPUs | **G0: EXL3-under-TP is UNVERIFIED** on this fork — smoke it before promising anything | lower per-request latency, one big KV pool; needed for 1M-class single requests (BF16) and long-window concurrency |
| vLLM native DP (`--data-parallel-size`) | engine-internal DP + LB | **G-DP: unverified on dev280 fork** | convenience only; replica-set DP is the fallback and the baseline |
| TP4xDP2 (8x host only) | two TP-4 replica sets | G0 | the 8x full-speed-run candidate vs DP8 |

**G0 (EXL3 TP smoke):** launch the quant at TP=2; require clean start, a generation proof, and the 8-frozen-
prompt set answered identically twice within the process. If EXL3 layers refuse TP (plausible: trellis
tensors are per-module and the fork has only ever been run TP=1 in this project), record the exact error —
that is a finding, not a failure — and the quant matrix collapses to **DP-only**, which the timeout mechanism
already favours. BF16 TP needs no gate (stock vLLM path).

**TP ceiling note:** the model has **4 KV heads**; TP>4 replicates KV heads and buys nothing for attention.
On the 8x host the sensible shapes are DP8, TP2xDP4, TP4xDP2 — never TP8.

### 4.3 The matrix, kept small

Synthetic saturation first (cheap), TB only where it decides a card line:

| tier | arms actually run with TB | chosen how |
|---|---|---|
| **1x quick** | quant DP1; BF16 DP1 | direct — the card's baseline pair |
| **2x quick** | quant DP2 vs (if G0 passes) TP2; BF16: better of TP2/DP2 by synthetic probe | one TB run per model at the synthetic winner |
| **4x quick** | quant DP4 (and TP4 only if G0 passed *and* synthetic says it beats DP4); BF16 TP4 vs DP4 by probe | same rule |
| **full speed run** | best 4x (or 8x) config per model, three-pass protocol (§6) | the headline |

Synthetic probe per candidate config (executor writes a ~100-line async OpenAI-client ladder; shape below):
concurrency 1,2,4,8,16,32,64 x {512-token prompt / 256 out, 4k prompt / 1k out, 30k prompt / 2k out},
3 repeats, medians; record aggregate tok/s, per-request tok/s, TTFT. Pick the knee by the standing headroom
rule: **cap at `max-num-seqs`, require per-request ≥ 50 % of single-stream, then maximise aggregate.**
Then read the real KV line and prefix-hit counters. One config change at a time; every probe run named and
snapshotted (counter deltas).

## 5. Load-driver shape

1. **Task sharding = the load balancer.** Do not put a router in front of vLLM. Partition the 89 tasks into
   K shards (K = number of replicas), **bin-packed by each task's declared timeout budget** from
   `receipts/terminal-bench-2.1-task-inventory.json` so shards finish together (greedy makespan). Run K
   parallel `harbor run` jobs, each pinned to its own replica's `api_base`. This gives perfect prefix-cache
   affinity (terminus-2 resends the whole transcript every turn — 51.7 % measured hit rate depends on
   landing on the same replica), zero new code, and per-shard resume.
2. **-n per shard:** from the headroom rule at that replica's synthetic knee (16 was correct on one 96 GB
   card at 32k eager; re-derive with graph decode on).
3. **Resume discipline:** `harbor job resume` is proven (finished job: 4/4 byte-identical in 3.5 s;
   SIGKILLed job: completed trials kept byte-identical, in-flight re-run). Keep per-shard job dirs; publish
   per-task artifacts to the dataset (`malaiwah/qwen38-27b-terminal-bench-2.1`) as they complete.
4. **disk_guard** as established: report free space per pass; below 15 GB remove only `docker.io/alexgshaw/*`;
   never prune under a live container; never a global prune.
5. Drive **BF16 and quant arms from the same driver VM, sequentially per tier** (never concurrently — they
   would contend for host GPUs and driver CPU and corrupt the wall-clock).

## 6. TB protocol per tier

- **Quick tier** = one full pass: 89 tasks x 1 trial, stock 1.0 timeouts, per-model. Report: resolved/89,
  wall-clock, `AgentTimeoutError` count vs real failures, parser-tolerance count (trials with "Extra text
  detected before JSON object" and how many of those resolved), per-turn token + wall-clock distribution,
  aggregate tok/s from metric deltas, prefix-hit rate.
- **Full speed run** = the owner's three-pass protocol at the best config: pass 1 full; pass 2 healing
  (retry every pass-1 failure, fresh containers); pass 3 attribution (every twice-failed task re-run on
  **BF16, same hardware tier, same agent, same harness**), with the **three-way split**: `quantization-suspect`
  (BF16 resolves it) / `capability` (BF16 completes and fails) / `inconclusive-timeout` (neither arm finished
  in budget). Plus the approved **diagnostic arm**: the inconclusive-timeout subset re-run once at
  `--timeout-multiplier 2.0`, labelled non-comparable, reported beside — never inside — the headline.
- **Fidelity guard before any scored pass** on a new config: generation proof; the 8 frozen decode prompts
  twice within the process (must repeat token-identically); needle retrieval at the full native window
  (qualification-style, exact match required). Cross-restart bit-exactness is not claimable on this stack —
  do not attempt it, cite `scratch-arena.json`.
- Stock 1.0 stays the headline everywhere. No server-side `max_tokens` clamp (comparability rule).

## 7. Optional labelled arms (run only after the card ladder is banked)

1. **V2 + per-batch-size depth schedule** (overlay `5723e072e`; `[[1,2,3],[3,8,1]]`): measured +38 % C8 on a
   32 GB card at 131k/0.95. At 96 GB the OOM constraint that forced the KV concession should not bind —
   verify headroom, run the synthetic ladder, and if it wins, one TB quick pass, labelled.
2. **Scratch arena** (PR #397 mount): +6.7 % KV. Only matters if KV becomes binding (262k concurrency, 1M).
3. **max-num-batched-tokens 2048 vs 8192** and **util 0.92 -> 0.95**: synthetic only, take the winner forward.

## 8. The 1M YaRN exercise (after TB deliverables)

Native first — that is banked by §4–6 (262,144 window serving TB and the needle gate). Then:

1. Read `config.json`: Qwen's published YaRN procedure for this family (rope_scaling `factor 4.0`,
   `original_max_position_embeddings: 262144` -> 1,048,576). **The cards mark Qwen's YaRN procedure as
   unverified on this runtime — treat this as a gated experiment, not a supported mode.**
2. Launch the quant, one card first (arithmetic says it fits: 20.31 weights + ~37 GiB KV for one 1M sequence
   + overheads < 96 GB): `--max-model-len 1048576 --rope-scaling '{"rope_type":"yarn","factor":4.0,
   "original_max_position_embeddings":262144}'` (exact key names verified against the fork's parser first).
   Gate: engine KV line must show ≥ 1.05 M tokens; if not, TP=4.
3. Exercise, in order: (a) needle probes at 512k and 1M (depth 0.25/0.5/0.75, exact-retrieval required —
   extend the qualification probe; our deepest prior evidence is 205k/262k, and 1M **retrieval quality is an
   open question, not a promise**); (b) TTFT for a 1M prefill (arithmetic at measured ~3.3k tok/s prefill:
   ~5 min — record actual); (c) 2–4 concurrent 512k sessions on TP=4 for the concurrency form.
   BF16 1M needs TP=4 (KV alone ~37 GiB + 52 GiB weights). Report serve-ability, TTFT, retrieval — no TB at
   1M (no TB task needs it; it would only burn the window).

## 9. Receipts and the card lines

One receipt per tier x model x config: `receipts/tb21-speedrun-<tier>-<model>-<config>.json` carrying: host
identity (GPUs, driver, topo), image digest, full argv + env, synthetic ladder table, chosen -n + rule
inputs, metric snapshot names + deltas, TB results (resolved/89, wall, timeout split, parser count),
fidelity-guard outputs, and every deviation from `terminal-bench-2.1-pins.json` named. Plus one summary
receipt `receipts/tb21-speedrun-ladder.json` feeding the card table:

```
| tier | config | resolved/89 | wall clock | agg tok/s | timeouts | notes |
| 1x RTX PRO 6000 | quant DP1 | [tbm] | [tbm] | [tbm] | [tbm] | |
| 1x RTX PRO 6000 | BF16 DP1  | [tbm] | [tbm] | [tbm] | [tbm] | |
| 2x ... 4x ... | winner cfg | [tbm] | ... | | | |
| full speed run | best cfg, 3-pass | [tbm] | [tbm] | | | + attribution split |
```

`[tbm]` = to be measured; the table shape and decision rules are committed **before** any run, in this
project's pre-registration habit. Card placement is Main's; the executor hands the filled table + one
paragraph per tier over hub/receipt, and does not edit cards.

## 10. Execution order (the executor's checklist)

1. G-NET (RTT/TTFB), host inventory receipt, image shipped + digest-verified, weights pulled
   (hydrated 21.6 GB; BF16 ~55.6 GB), driver VM provisioned, task tree + images staged (lazy pull + guard).
2. G0 EXL3-TP smoke (15 min, decides half the matrix). G-DP native-DP smoke (10 min, convenience only).
3. Synthetic ladders: 1x quant, 1x BF16, then only the 2x/4x arms the probes justify. Headroom rule fixed.
4. Fidelity guard per serving config that will host TB.
5. TB quick 1x (quant, then BF16). Publish incrementally. Then 2x, 4x winners.
6. Full speed run: three passes + diagnostic arm at the best config.
7. Optional arms (§7), then 1M (§8) if the window allows.
8. Ladder receipt + card table + hub handover. Leave the host clean: services stopped, receipts pushed to
   both remotes, nothing unpublished on scratch.

## 11. Risks, named

- **EXL3 under TP is the load-bearing unknown** (G0). If it fails, the quant still gets DP scaling — which
  the timeout mechanism favours anyway — and the finding itself is publishable.
- **Graph decode at 262k with 16 seqs on this fork at 96 GB** is a new combination (qualified separately,
  not jointly). The fidelity guard + needle gate covers it; if it misbehaves, fall back to the exact TB2
  pinned profile and say so.
- **YaRN 1M is unvalidated for quality** on this runtime; §8 measures retrieval before anything is claimed.
- **Two models on one host**: never concurrently; sequential arms only.
- **Comparability**: this plan deliberately deviates from the TB2 pins (graph decode, native window,
  batched-token size). Every deviation is named in the receipts; the stock-timeout headline keeps TB-to-TB
  comparability; serving-config differences are what the speed run *measures*.


## 12. Campaign decisions, 2026-08-16 (owner + Main)

**Allocation: one serving host at a time, resized per phase — never a fleet.**

| phase | hardware | purpose | why this is the cheap shape |
|---|---|---|---|
| A (now, already rented) | the existing 1x RTX PRO 6000 rental + the IN1 driver VM | finish the in-flight owner-protocol passes 2-3 (driven from AIBoss for pass-1 comparability); afterwards the rental doubles as the **1x tier + single-GPU knob lab** driven from the IN1 driver at 0.3 ms | both boxes are already paid; the 1x row needs nothing new |
| B (next rental) | **one 2x GPU VM in the Jarvis VPC** | the knob lab: G0 EXL3-under-TP gate, TP2-vs-DP2, graph-decode joint config, V2 depth-schedule arm, KV-dtype smokes (fp8 headline; BF16-KV and nvfp4-KV smoke arms), LMCache-fix GPU ladder if the PR lands | 2x is the smallest host that can answer every question that transfers to 4x/8x; every knob found here is a config, not a rebuild |
| C (short) | 4x for ~a day | one quick TB pass per model at the 2x-derived winner config — a **checkpoint against interpolation**, not a sweep | 1x->2x->4x scaling tells us whether DP scales linearly as expected; if it does, 8x needs no exploration |
| D (last minute, booked once B/C are banked) | 8x | synthetic probe (hours) to pick DP8 vs TP4xDP2, then the **full three-pass speed run** + BF16 arm + diagnostic | the expensive box only ever runs configs already proven |

Interpolation stance: 1x/2x/4x quick rows bound the curve; 8x is run to **measure the maximum**, not to
explore. If 2x->4x deviates from linear DP scaling by more than ~10 % on the synthetic ladder, the 4x TB
quick pass is mandatory before 8x; otherwise it may be skipped and the interpolation stated on the card.

**Measured driver-VM facts (receipt `tb21-driver-vm.json`):** `jl-vm-473296`, 32 vCPU / 125 GB / 968 GB
NVMe, Ubuntu 24.04.4 fully upgraded 2026-08-16 (kernel 6.8.0-137), docker 29.6.0 + compose v5.2.0 (newer
than the pinned 27.5.1/2.39.1 — named deviation), harbor 0.21.0 via uv, task tree copied from AIBoss and
sha-verified **byte-identical** (`c13961ac…`, 89 tasks), private VPC NIC `10.0.0.2/16`. RTTs: **rental <->
driver 0.3 ms** (same IN1 DC — the G-NET gate is already passed for phase A), driver -> ns1 jump ~250 ms,
driver -> AIBoss vLLM ~0.56-0.77 s TTFB (two geographic legs; used only for smoke). Adequacy: meets the
§3 spec; the only watch item is CPU load at -n 64 on the 8x tier, measured before use.

**The in-flight AIBoss run is NOT moved.** Pass 1 was scored with the AIBoss->rental path in its wall
clock; moving passes 2-3 to the IN1 driver would give the healing and BF16 arms a faster transport than
the arm they are compared against (and harbor refuses resume under a changed `api_base` anyway). The
owner-protocol result stays internally consistent; the speed-run campaign starts fresh from the IN1
driver with the sub-ms path.

**Flagship artifact: frozen — no re-quant.** The published hydrated bytes are the product: every number
(0.002760 KLD, K6-parity pairing, MMLU, TB pass 1) cites those bytes, conversions are measurably
nondeterministic (the sibling experiment: 97.6 % of modules differ, fidelity indistinguishable), so a
re-quant would sever every citation and buy nothing measurable. K6-parity exists as the 6-bit SKU.
Pre-flight is config-only: graph decode ON, native window, the §4 recipe.

**Campaign image `vllm:gg-r34-tb21-sr1` (build in progress, receipt `tb21-image-sr1.json`):** FROM the
promoted base **by digest**, plus audited patch layers, each sha-pinned with an in-image sentinel that
fails the build if absent: scratch arena (PR #397), the V2 FlashInfer decode-shape fix (PR #398 @
`5723e072e`), a port of the admission-livelock fix (vllm-project/vllm#52530 @ `479413adc`), and — **only
if the fix lands and passes gates — a patched LMCache** (see below). Canonical artifact: `docker save`
tar + sha256 on the driver; registry push once the owner names one (ghcr.io/malaiwah proposed; needs a
token). Rebuild command in the receipt; digests published on the cards with the results.

**LMCache is in the campaign only as an honestly-fixed component.** An agent is root-causing the MP-path
corruption we reported (LMCache#4247/#4492) and filing a real upstream PR. Inclusion gates, in order:
CPU tests fail-before/pass-after; the **4-arm corruption ladder re-run on the 2x lab must read
0-corruption on every arm** including restart-over-warm-L2 (previously 38/38); only then does the sr1
image enable it, as a labelled arm first. If the fix is not ready, the campaign ships without LMCache
and the cards say why. This is the battle-test the owner asked for, with the same fail-closed shape as
everything else here.

**KV dtype arms:** fp8 is the headline (family default, qualified). One smoke each of BF16-KV and
nvfp4-KV on the 2x lab for the comparison table — noting the prior sweep measured the nvfp4 KV variants
as refusals on this architecture (SM100/MLA-gated); a refusal is re-recorded, not retried into existence.
