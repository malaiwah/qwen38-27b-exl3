# 46. TB2.1 speed run on a multi-GPU Jarvis host: the execution plan

**Status: EXECUTED CAMPAIGN LOG.** Sections 0–12 preserve the pre-registered
plan; sections 13–35 append measured outcomes, corrections, and retractions from
2026-08-16/17. Read the later section for any item before acting on its original
plan entry.

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
  --max-num-seqs 64                 # per replica: a CEILING for the knee search, not a chosen value.
                                    # 16 was inherited from the TB pins, unmeasured. The ladder sweeps
                                    # concurrency up to this cap; if the knee lands AT the cap, raise the
                                    # cap and re-sweep - the server cap must never be the binding
                                    # constraint during the search. Mamba/GDN state is preallocated per
                                    # max-num-seqs on this hybrid, so the cap costs KV pool: record the
                                    # engine's KV line at each cap value tried. The measured knee (owner
                                    # rule: per-request >= 50 % of single-stream, then max aggregate)
                                    # becomes the BASELINE -n for every TB tier.
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
concurrency 1,2,4,8,16,32,64 (extend 96,128 if the knee is still rising at 64) x {512-token prompt /
256 out, 4k prompt / 1k out, 30k prompt / 2k out},
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
receipt `receipts/tb21-speedrun-ladder.json` feeding the card table (**SUPERSEDED — this single summary receipt was never created. It was replaced in practice by 15 per-tier `receipts/tb21-ladder-*.json` receipts plus the cross-arm `receipts/tb21-8x-topology-ladder.json`, which carry the same content at finer grain. Do not go looking for the file.**):

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

**Campaign image built and loaded 2026-08-16.** The image is `vllm:gg-r34-tb21-sr1` (manifest-list
digest `sha256:237a5025…`, identical on the build host and the 1x endpoint after save/load; receipt
`receipts/tb21-image-sr1.json`). It is the promoted apc release unit (base identity proven by the
runtime-invariant diff_id chain `886dbafc…`, plus an in-build byte gate on six load-bearing modules) with
three audited fail-closed patch layers: the PR #397 scratch-arena overlay (+6.7 % KV), the PR #398
FlashInfer decode-shape keying the V2 arm requires, and a hand-port of
[vllm-project/vllm#52530](https://github.com/vllm-project/vllm/pull/52530) (admission gate + pool-bound
length cap; the port's one adaptation is stated in the receipt). Each layer asserts its patch sha, the
post-patch file sha, a sentinel symbol, and recompiles under the image's own python, so a wrong base or
drifted patch cannot produce an image. LMCache is untouched and disabled by default - the pinned wheel's
698 RECORD hashes are verified intact in-build - and stays out until LMCacheFix's patchset passes CPU
gates and the 4-arm GPU ladder. Canonical artifact:
`ubuntu@151.185.34.98:~/images/gg-r34-tb21-sr1.tar.zst`, sha256 `1e5711f4…`; sr1 is already docker-loaded
and sentinel-verified on the endpoint (`vllm --help` OK under `--gpus all`). **The image is NOT
GPU-qualified: the §6 fidelity gate must pass on sr1 before any scored pass.** Registry (ghcr.io vs
docker.io, needs an owner token) is open; until then the driver tar is canonical. FROM the
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


## 13. Knob audit, 2026-08-17: what is at 11, what is not, and what is deliberately off

Asked directly whether every lever is maxed for the speed run. Audited against evidence, with each
item's state verified on the live 1x endpoint (`jl-vm-473319`, sr1 image, native 262,144 window)
rather than asserted from the plan.

### Verified ON and at 11

| knob | evidence it is actually active |
|---|---|
| CUDA-graph decode | `exl3.py:1868 EXL3 graph decode enabled by VLLM_EXL3_GRAPH_DECODE`, 48 decode graphs captured in 10 s. **129.5 tok/s single-stream** vs the 82.9 eager baseline. Required `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` - **not** `--enforce-eager`, which is what the EXL3 error message misleadingly suggests (`_graph_decode_refusal` permits graph decode only for a decode-only mode with capture sizes set). |
| Scratch arena (PR #397) | `exl3.py:818 EXL3 reconstruct scratch arena grew to 170.0 MiB` - exactly the predicted size, so the +6.7 % KV is already inside the measured 1,760,318-token pool. |
| MTP-3 speculation | 59.2 % acceptance measured live during the ladder (29,586 draft / 17,529 accepted over ~70 s). |
| fp8 KV | family default, qualified; the whole KV-dtype sweep is already published. |
| Prefix caching | on; 51.7 % hit rate measured on real agent traffic. |
| Reconstructed prefill (`VLLM_EXL3_PREFILL_RECONSTRUCT_M=128`) | on; this is PR #316's dispatch, which already collected the MLP prefill win. |
| Chunked prefill | `scheduler.py:252 Chunked prefill is enabled with max_num_batched_tokens=8192`. |
| Async scheduling | `vllm.py:1162 Asynchronous scheduling is enabled`. |
| `max_num_seqs` | raised 16 → 64 as a knee-search **ceiling**, so the search is not capped by an inherited value. |

### Known-unclaimed, ranked by expected value

1. **V2 runner + per-batch-size depth schedule — the largest known unclaimed win.** Measured
   **+38.0 % aggregate decode at C8** on a 32 GB card; the fix that unblocks it (PR #398) is already an
   audited layer in sr1. What stopped it there was a 32 GB constraint: V2 costs ~780-820 MiB more than
   MRV1, and at util 0.97 that left 58.56 MiB free and OOM'd the prefill reconstruct, forcing
   131,072/0.95 and −14.1 % KV. **On this card that cost is 0.8 % of 96 GB** (weights 20.78 GiB, KV
   62.38 GiB, free 94.43/94.97 at util 0.92), so it should not bind. Two arms are staged
   (`serve-hyd-v2base.sh` = V2 with static depth 3, isolating the runner; `serve-hyd-v2sched.sh` = V2 +
   schedule). **Unverified combination, not a broken one:** V2 + hybrid mamba + MTP-3 + EXL3 + fp8 KV +
   graph decode at the native window on 96 GB has never been run.
2. **The depth schedule must be re-derived, not copy-pasted.** The measured `[[1,2,3],[3,8,1]]` covers
   batch 1-8 only; the campaign sweeps to 64 and DP-K replicas each sit at their own knee, so batches
   9-64 would be unscheduled. The staged arm uses `[[1,2,3],[3,64,1]]` as a starting hypothesis - it
   needs measuring, not assuming.
3. **~7.2 GiB of KV is sitting unused.** The engine itself reports
   `--kv-cache-memory-bytes=74719122432` (69.59 GiB) "to fully utilize gpu memory" against the 62.38 GiB
   actually in use at util 0.92 - **+11.6 % KV tokens, free, no fidelity cost**. Honest scope: this does
   **not** help TB, where 1.76 M tokens already covers ~54 concurrent 32k sessions and KV is nowhere near
   binding. It helps the native-262k concurrency line (6.72× → ~7.5×) and the 1M exercise.
4. **`max_num_batched_tokens` 2048 vs 8192 and util 0.92 → 0.95** remain synthetic-sweep items, not yet
   measured on this card.
5. **Prefix-cache headroom.** The 51.7 % hit rate was measured on a 32 GB card; this one has 3× the KV
   pool, so the achievable rate on identical agent traffic is probably higher and is worth measuring
   rather than inheriting.

### Deliberately OFF, and why that is the correct setting

- **int8 embed overlay** (+1.8-2.1 % KV): **changes greedy output** (13/22 and 10/22 continuations
  diverge). Off for a fidelity-sensitive benchmark.
- **LMCache**: corruption reproduced 38/38 on restart-over-warm-L2; the real root cause turned out to be
  our own scheduler (see docs/29), whose fix is CPU-proven only. Stays out until the 4-arm ladder reads
  clean on the 2x lab.
- **`reasoning_effort`**: the upstream template accepts only
  `low|medium|xhigh`; `high` is rejected. Lower effort may shorten reasoning
  bursts, but timeout incidence does not isolate that mechanism. The owner kept
  the speed run at the default for comparability. A 4,096-token probe truncated
  mid-reasoning; a real characterization needs a valid effort value, an idle
  card, and at least a 32k output budget.
- **`VLLM_EXL3_PREFILL_FP8`** (+31 % prefill): costs **+0.0141 mean KLD**, 4.4× this artifact's entire
  quantization error. Never for a fidelity-carrying run.
- **`custom_ops:["all"]`** (2.2-5.2 % worse, not bit-exact), **`--attention-backend FLASHINFER`**
  (measured no-op: auto-selected already), **adaptive/per-batch speculative decoding** (downgrades
  `cudagraph_mode` to PIECEWISE, which EXL3 refuses), **PIECEWISE prefill** (never dispatched above
  `max_cudagraph_capture_size`≈512 while our prefill batches are 2,048-8,192, and prefill is
  kernel-bound at cuBLAS parity anyway), **`reconstruct_had_slice`** (predicted ~1.31× *cost*).

### Buildable, not yet built

- **b12x K5 support for the 208 attention projections.** Every one currently fails b12x's
  `trellis.shape[2] == 96` clause (bits must be 6) because our attention is serialized at **K5**, so 208
  matrices take a slower path. Widening `_b12x_trellis_k6_supported` beyond K6-only is docs/41's W2.
- **The prefill residual.** Our 3,374 tok/s PP2k against 5,050-5,250 measured for a dense EXL3 on a
  188-SM card: SM-count scaling explains only ~70 % of the gap, leaving hybrid geometry and head_dim 256
  as the source-verified candidates.

**Bottom line:** the fidelity, capacity and decode-graph knobs are maxed and verified; the single
significant unclaimed decode win is the V2 runner (staged, A/B pending), the KV headroom is free but
irrelevant to TB specifically, and the largest *TB-score* lever (reasoning effort) is deliberately left
at default to protect comparability.


## 14. MEASURED: the 1x knee ladder, and what it says about pass 1

Run 2026-08-17 on `jl-vm-473319` (1x RTX PRO 6000 96 GB, driver 595.58.03), sr1 image, native
262,144-token window, graph decode on, MTP-3, fp8 KV, prefix caching on, `max_num_seqs 64` as a search
ceiling. Seed 46, 3 repeats per cell, 7 rungs x 3 shapes, **zero refused or errored requests in all 21
cells**. Full per-cell data with verbatim `/metrics` snapshots:
[`tb21-ladder-1x-hyd.json`](../receipts/tb21-ladder-1x-hyd.json). Chart:
`assets/tb21-knee-1x-{light,dark}.svg`.

| shape | C1 | C2 | C4 | C8 | C16 | C32 | C64 | **knee** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **512-in/256-out** per-req tok/s | **129.57** | 87.76 | 104.25 | 65.97 | 36.95 | 23.73 | 15.16 | **C8** (50.9 %) |
| aggregate tok/s | 129.55 | 174.69 | 311.79 | 413.98 | 485.00 | 664.56 | **828.59** | |
| **4k-in/1k-out** per-req tok/s | **95.29** | 85.21 | 71.78 | 46.98 | 24.87 | 14.53 | 9.39 | **C4** (75.3 %) |
| aggregate tok/s | 95.28 | 168.87 | 279.63 | 344.70 | 381.55 | 455.31 | **584.32** | |
| **30k-in/2k-out** per-req tok/s | **77.29** | 60.59 | 43.79 | 23.38 | 12.78 | 6.29 | 3.38 | **C4** (56.7 %) |
| aggregate tok/s | 77.28 | 113.02 | 160.43 | 179.12 | **195.37** | 185.12 | 187.66 | |

### Four findings, in order of consequence

**1. The knee for TB's actual shape is C4 - and pass 1 ran at n=16.** A TB turn measured ~4,600 prompt
tokens with reasoning bursts to 15,577 completion tokens, i.e. squarely between the 4k and 30k shapes,
both of which knee at **C4**. Pass 1 ran `-n 16`, **four times past the knee**, where per-request
throughput is only **26 %** of single-stream. Quantified against the measured burst: 15,577 tokens at
C16's 24.87 tok/s is **10.4 minutes for one turn**; at C4's 71.78 tok/s it is **3.6 minutes**. Task
timeout budgets are minutes-scale. **That is very likely a major cause of pass 1's 54-of-58
`AgentTimeoutError` failures** - we were converting capability into clock by over-subscribing the card.
This is a correction to the campaign's baseline `-n`, not a criticism of the pass: `-n 16` was inherited
from the TB pins, never measured, which is exactly why the owner ordered a knee search before scaling.

**2. On long contexts, aggregate throughput has a hard ceiling, not a tradeoff.** The 30k shape peaks at
**C16 (195.37 tok/s)** and then *falls* - 185.12 at C32, 187.66 at C64. Past C16 on long prompts, extra
concurrency buys **nothing at all** while per-request collapses to 4.4 % of single-stream. The
"aggregate keeps climbing past the knee" tradeoff that holds on short prompts does **not** hold here.

**3. TTFT becomes the binding constraint before decode does.** 30k-in at C64: TTFT p50 **455.8 s**, p95
**575.0 s** - over seven minutes before the first token. 4k-in at C32/C64: 22.0 s / 52.2 s. For an agent
with minute-scale per-task budgets, high concurrency fails the task during *prefill*, before decode
speed matters at all. Any tier's `-n` must respect TTFT, not just per-request decode.

**4. MTP acceptance is healthy and concurrency-stable, and the server never refused a request.**
Acceptance ranges 0.54-0.94 (highest **0.9417** at C1 on the short shape, ~0.62 typical on longer
shapes) with no systematic collapse as concurrency rises. Zero refusals/errors across all 21 cells
including 64 concurrent 30k-token prompts - a robustness result worth stating for an API-hoster
audience.

### Consequences for the campaign

- **Baseline `-n` per replica is 4, not 16**, for any tier whose workload resembles TB. Short-prompt
  workloads may use C8.
- **DP-K scaling should multiply replicas at n=4**, not raise n per replica: 8 replicas x n=4 = 32 global
  concurrent trials, which is more useful than 2 replicas x n=16 because it keeps every request above the
  per-request floor.
- **A pass-1 re-run at n=4 is now the highest-value TB measurement available** - if the timeout rate
  drops sharply, the 31/89 headline was a serving artifact rather than a capability ceiling, and that
  matters for every number we publish about this model as an agent.
- The 50 %-of-single-stream rule is the owner's, fixed before the sweep; both it and
  max-aggregate-among-eligible are computed in the receipt so a reader can apply a different floor.


## 15. MEASURED: the V2-runner A/B at 96 GB, and a schedule crossover that does not transfer

Three serving arms, one variable each, same endpoint / image / window / seed / repeats, run 2026-08-17.
**All three gated PASS on all five fidelity checks including the 262,144-token needle**
(`tb21-gate-1x-{hyd,v2base,v2sched}.json`). Ladders:
`tb21-ladder-1x-{hyd,v2base,v2sched}.json`.

Arms: **MRV1** = the campaign baseline (§4.1). **V2base** = `VLLM_USE_V2_MODEL_RUNNER=1`, static depth 3
(isolates the runner). **V2sched** = V2 + `num_speculative_tokens_per_batch_size=[[1,2,3],[3,64,1]]`
(depth 3 at batch 1-2, depth 1 at batch 3+).

| shape / rung | MRV1 per-req / agg | V2base per-req / agg | V2sched per-req / agg |
|---|---:|---:|---:|
| **512-in** C1 | 129.57 / 129.55 | 134.87 / 134.84 | **134.94 / 134.91** |
| C4 | **104.25 / 311.79** | 108.00 / **329.50** | 74.84 / 282.58 |
| C8 | 65.97 / 413.98 | 66.98 / 431.69 | 62.47 / **451.84** |
| C16 | 36.95 / 485.00 | 37.20 / 497.07 | 36.90 / **555.79** |
| **4k-in** C1 | 95.29 / 95.28 | 89.73 / 89.72 | **98.33 / 98.32** |
| C4 | 71.78 / 279.63 | **74.16 / 282.60** | 62.14 / 245.51 |
| C8 | 46.98 / 344.70 | 46.67 / 348.25 | **49.70 / 389.67** |
| C16 | 24.87 / 381.55 | 24.67 / 380.71 | **28.23 / 446.63** |

### 1. The 32 GB OOM constraint does not transfer - V2 is free here

V2 started cleanly at the **native 262,144 window** with graph decode, **zero OOM**, and its KV pool came
out **1,776,428 tokens against MRV1's 1,760,318 - slightly *more***. On the 32 GB card V2 cost ~780-820
MiB and forced 131,072/0.95 with **−14.1 % KV**; on 96 GB that cost is 0.8 % of the card and the
concession disappears. **The blocker recorded in docs/41 W1 is card-specific and is now closed for this
tier.**

### 2. The V2 runner alone is neutral; the schedule is what pays

V2base vs MRV1 is within noise (512-in single-stream +4.1 %, 4k-in −5.8 %, mid-rungs ±3 %) - matching the
earlier 5090 finding that the runner alone is neutral-to-slightly-positive. **V2sched is where the win
is, and only above the crossover:** +**9.1 %** aggregate at 512-in C8, +**14.6 %** at C16; +**13.0 %** at
4k-in C8, +**17.1 %** at C16.

### 3. The schedule's crossover is wrong for this card, and fixing it should beat every arm

`[[1,2,3],[3,64,1]]` switches to depth 1 at batch **3**, and that is measurably too early here: at **C4
V2sched loses badly** - 512-in per-request 104.25 → 74.84 (**−28 %**), 4k-in aggregate 279.63 → 245.51
(**−12.2 %**) - because depth 1 gives up speculation before the batch is large enough for its lower
verification overhead to pay. Above C8 the same setting wins by 9-17 %.

**So the optimum is neither arm: keep depth 3 through the knee, switch to depth 1 above it.** Concretely,
a schedule with the crossover at ~5-8 rather than 3 (e.g. `[[1,4,3],[5,64,1]]`) should hold MRV1/V2base's
C4 numbers *and* capture V2sched's C8-C16 aggregate. That is an unrun, well-specified experiment, and it
is why the 5090's `[[1,2,3],[3,8,1]]` must not be copy-pasted: **the crossover is a per-card, per-window
property, not a constant.**

### 4. What this means for the campaign, given the knee

The knee (§14) puts TB at **n=4**, which sits *below* the crossover - so **for TB itself the depth
schedule is not the lever; depth 3 is already correct at C4** and V2base's small gain is the honest
expectation. The schedule matters for the **aggregate** story: a DP-K speed run whose replicas each run
at C8-C16 gains 9-17 % from it. Two different operating points, two different right answers, and the
receipts now separate them.

### 5. Methodological caveat on the knee metric

The knee is defined against single-stream, so a few percent of C1 noise moves the floor and can move the
reported knee a rung (512-in: C8 under MRV1, C4 under both V2 arms, driven by C1 shifting 129.57 → 134.9).
The **per-rung table is the durable artifact**; the single knee integer is a summary that inherits C1's
noise. Cite the table.


## 16. The continuous metrics timeline, rehearsed - and what it independently confirmed

Owner requirement: capture `/metrics` at a regular interval for post-analysis, published for
transparency. Built as `tools/tb21_metrics_poll.py` and **wired into `tb21_campaign.sh pass1`**, so every
pass records one timeline per replica automatically (soft teardown; poller failure can never fail a pass).

**Rehearsed live for 77 minutes across the entire ladder and all three V2 arms**: 929 samples at 5.01 s,
published as [`tb21-metrics-1x-hyd-rehearsal.jsonl`](../receipts/tb21-metrics-1x-hyd-rehearsal.jsonl) +
`.summary.json`.

### Two defects the rehearsal found in the tooling itself

1. **An unclean kill loses the summary but never the data.** The summary is written by the SIGTERM
   handler; the rehearsal's poller died with its SSH transport instead (exit 255), leaving the stream
   complete and the summary missing. The streaming design held exactly as intended, and the tool now has
   `--summarize-only` so a summary is always reconstructible from the stream rather than depending on
   process teardown. (The campaign path was never exposed to this: `tb21_campaign.sh` runs the poller
   locally on the driver and signals its PID directly.)
2. **A naive first-to-last delta is meaningless across a restart, and would have been published.**
   vLLM's counters are per-process, so each arm swap resets them; the first version of the summary
   reported **negative** token deltas (−7.2 M prompt tokens). The summary is now reset-aware: it detects
   counter resets, splits the stream into monotonic segments, reports per-segment deltas, and **refuses**
   the single delta when a reset exists, with the reason stated in the receipt.

### What the timeline then confirmed, from server-side counters rather than client timing

Three monotonic segments, one per arm, with the two V2 arms doing **identical work** (same ladder subset
under `ignore_eos`, hence identical 120,080 generation tokens - which is itself a check that the arms
were comparable):

| segment | arm | generation tokens | **MTP acceptance** |
|---|---|---:|---:|
| 0 (526 samples) | MRV1 baseline | 451,730 | 0.6480 |
| 1 (119 samples) | V2, static depth 3 | 120,080 | 0.6122 |
| 2 (136 samples) | V2 + depth schedule | 120,080 | **0.7854** |

**The depth schedule raises MTP acceptance from 0.61 to 0.79 (+28 % relative)** - independent, server-side
corroboration of §15's throughput result, and the mechanism behind it: at depth 1 the single drafted
token is far likelier to be accepted than the third token of a depth-3 draft. Two independent measurement
paths (client-side timing, server-side counters) agreeing on the same conclusion is the strongest form
this result could take.

**Prefix-cache hit rate is 0.0000 in every segment** - confirming the ladder's unique-seeded-prompt design
did what it was built to do: no cache reuse inflating any throughput number in §14 or §15.

## 17. MEASURED: the depth schedule re-derived for THIS card's knee, and the schedule we ship

§15 measured the 5090-derived schedule `[[1,2,3],[3,64,1]]` and found it **loses 12-28 % per-request
at C4** while gaining above C8 - and §14 had already measured this card's knees at **C4/C8**, i.e.
exactly where it loses. So the schedule was wrong for the machine, not wrong in principle. Three more
arms were measured to re-derive it, each a **single-variable** change off the same sr1 image, same
262,144-token window, same seed 46, same shapes and rungs, and each **gated PASS on all five checks
including the 262k needle** before any number was kept.

The mechanism is not a guess: the scheduler picks depth by
`dynamic_sd_lookup[len(num_scheduled_tokens)]` (`v1/core/sched/scheduler.py:1157-1160`), and
`num_scheduled_tokens` is keyed by request, so the ranges are **requests in flight per step** - which
is why a boundary written at 8 lands exactly on the C8 rung. Entries are inclusive
`(range_start, range_end, num_speculative_tokens)` (`config/speculative.py:181-186`).

| rung | MRV1 baseline | V2 static-3 | V2 `[[1,2,3],[3,64,1]]` (§15) | A `[[1,8,3],[9,64,1]]` | B `[[1,4,3],[5,8,2],[9,64,1]]` | **C `[[1,4,3],[5,64,1]]`** |
|---|---:|---:|---:|---:|---:|---:|
| 512x256 C4 per-req | 104.2 | 108.0 | **74.8** | 107.4 | 108.4 | **108.0** |
| 512x256 C8 agg | 414 | 432 | 452 | 414 | 404 | **463** |
| 512x256 C16 agg | 485 | 497 | 556 | 552 | 543 | **558** |
| 4kx1k C4 per-req | 71.8 | 74.2 | **62.1** | 74.6 | 78.1 | 74.1 |
| 4kx1k C8 agg | 345 | 348 | 390 | 348 | 348 | **392** |
| 4kx1k C16 agg | 382 | 381 | 447 | 447 | 446 | **447** |

**Ship `[[1,4,3],[5,64,1]]`.** Against the baseline it is **+11.8 % aggregate at C8** and **+15.1 % at
C16** on the short shape, **+13.6 %** and **+17.0 %** on the 4k shape, and it **gives up nothing** -
C4 per-request is +3.6 %. Against the schedule the campaign was about to ship it is **+44.4 %
per-request at C4** (108.0 against 74.8) while still winning C8 and C16 aggregate. Receipts:
[`tb21-ladder-1x-knee-{a,b,c}.json`](../receipts/) with
[`tb21-gate-1x-knee-{a,b,c}.json`](../receipts/).

**Two findings worth more than the tuning.**

**1. Draft depth is not monotone in throughput - the middle value is the worst.** Arm B put depth 2
across 5-8 expecting a compromise. At C8 it returned **57.9 per-req / 404 agg**, worse than depth 3
(67.0/414) *and* worse than depth 1 (62.5/452), on both metrics, on both shapes. Depth 2 pays draft
compute without earning enough acceptance to amortise it. Anyone tuning a schedule by interpolating
between two measured depths will land on the worst cell available.

**2. The knee label is relative, so a faster engine can appear to lose concurrency.** The knee is the
largest rung holding >=50 % of that arm's *own* single-stream. V2 raised single-stream ~4 % (129.6 ->
135.2), which raised the bar: at C8 the V2 arms deliver **more** absolute per-request throughput than
MRV1 (67.0 against 66.0) yet miss the floor by 0.3-0.4 points (49.6-49.7 % against 50.9 %), so their
knee **label** reads C4 where MRV1's reads C8. Nothing got worse. Any comparison of knee labels across
arms with different single-stream speeds is meaningless unless the ratio is printed beside it.

**One knob checked and correctly declined: acceptance-length adaptation.** `SpeculativeConfig` also
offers `adaptive_speculative_tokens_window`, which re-picks depth from observed accepted length
(`spec_decode/dynamic/acceptance_length.py:58-69`) as
`min(max_depth, max(1, floor(mean_accepted + 1.5)))`. It is supported with `mtp` and it **cannot help
this model**: our measured acceptance at depth 3 is 0.57-0.79, i.e. 1.7-2.4 accepted per draft, so the
rule evaluates to `floor(3.2..3.9) = 3` and pins at the maximum permanently. It is also **blind to
batch size** - the axis that produces the entire +11-17 % aggregate win above - so it optimises
acceptance where throughput is what binds. Declined by arithmetic on measured inputs rather than by
burning a GPU hour.

**Async scheduling needed no action: it is already on.** `vllm.py:1162 Asynchronous scheduling is
enabled` in both the API server and EngineCore on every arm above, because `EagleModelTypes` includes
`MTPModelTypes` (`config/speculative.py:66`, `39-55`), so vLLM's auto-enable path takes it rather than
the spec-decode disable path at `config/vllm.py:1124-1135`. Every number in this document was measured
with it active; there is no unclaimed gain here, and §13 records it under *Verified ON and at 11*.

## 18. Ready for the 2x host: the exact order, and one new question today's result created

Yes - the 1x tier is finished and the 2x lab is the next gate. Everything runs from the committed
tools; nothing needs writing first.

**Run order, each answering one question, cheapest-first.**

1. **G0: does EXL3 shard under TP>1 at all?** This has never been run anywhere in this project, and it
   is a hard gate rather than a measurement: rank-sliced EXL3 checkpoints are explicitly *exempt* from
   the piecewise capture path (`exl3.py:1802-1805`), which is a hint that sharding is a distinct code
   path, not a free flag. `--tensor-parallel-size 2` + the five-check fidelity gate, including the 262k
   needle. If it fails, TP is off the table for this checkpoint and every later tier is DP-only - and
   that is a publishable result on its own.
2. **TP2 against DP2 on the same two cards**, same shapes/rungs/seed as §14 and §17 so the rows drop
   straight into those tables. The prediction to falsify: **DP2 wins aggregate, TP2 wins latency.**
   TP2 halves per-GPU weight bytes (20.78 -> ~10.4 GiB), and since decode here is weight-streaming
   bound at 55-65 % of spec bandwidth, halving the bytes each step streams is the only lever that
   attacks **single-stream** speed - which is the lever that matters for a benchmark where **93 % of
   failures are timeouts**. DP2 instead buys two independent replicas and should scale aggregate
   ~linearly while leaving per-request speed unchanged.
3. **KV-dtype smokes** (fp8 vs BF16 vs nvfp4) on the 2x topology, to confirm the family default still
   holds when the KV pool is split across cards.
4. **LMCache 4-arm ladder**, the one arm that needs a GPU to close out the upstream MP-corruption work.

**PP is declined, with a reason rather than an omission.** Pipeline parallelism pays bubbles and
inter-stage latency to buy the ability to hold a model that does not fit. This one fits with room to
spare - 20.78 GiB of weights and **94.43 of 94.97 GiB free** at util 0.92 on a single card - so PP
would add latency to solve a problem we do not have. It is worth revisiting only if a future build
stops fitting on one card.

**The new question §17 created: the shipped schedule is topology-dependent, and DP changes which band
you land in.** The depth schedule keys on `len(num_scheduled_tokens)` - **requests in flight per
engine step, per replica** (`v1/core/sched/scheduler.py:1157-1160`). Under DP a router splits client
concurrency across replicas, so 16 concurrent clients against DP2 present **8 requests per replica**,
and against DP4 present **4** - which under `[[1,4,3],[5,64,1]]` moves a replica from the depth-1 band
into the depth-3 band without anyone changing a flag. That is not a bug, it is a coupling: **client
concurrency is not the quantity the schedule sees.** So the 2x lab must re-derive the boundary against
the *per-replica* batch distribution the load balancer actually produces, and the shipped schedule has
to be stated per topology (1x, DP2, DP4, DP8) rather than once. The shard-list load balancer in
`tb21_campaign.sh` pins each task to one replica by declared timeout budget, so the per-replica
distribution is knowable in advance rather than emergent - which is what makes re-deriving it cheap.

## 19. MEASURED: the Jarvis VM was serving TP with P2P switched off, and fixing it is worth 7-22 %

The 2x host arrived and G0 passed immediately - **EXL3 shards under `--tensor-parallel-size 2`** with
`Model loading took 10.61 GiB` on *each* rank against 20.78 GiB on one card, a **72.29 GiB** KV pool,
**4,080,073 KV tokens** (2.32x the single card's 1,760,318) and **15.56x** max concurrency at the native
window, with the five-check gate PASS including the 262k needle. That is the hard gate cleared: rank
sliced EXL3 is exempt from piecewise capture, so sharding was never guaranteed.

Then the first TP2 ladder said something suspicious: **+11 % single-stream over one card and zero
aggregate benefit** - a whole second GPU buying latency only. The startup log had already named the
cause: `SymmMemCommunicator: native P2P atomics are not supported between devices [0, 1]`.

**Measured, not assumed: P2P was entirely off.** `torch.cuda.can_device_access_peer(0,1)` returned
**False** both ways; a 64 MiB cross-GPU copy ran at **35.5 GB/s** and a 256 B copy took **12.48 us** -
squarely in the "P2P disabled ~14 us" band of
[`hardware/pcie-bandwidth.md`](https://github.com/local-inference-lab/rtx6kpro/blob/55323f94cd9d9ea98ccecef553791a63c3585816/hardware/pcie-bandwidth.md).

**What that doc gets us on a Jarvis VM, item by item.** Already correct before we touched anything:
`uvm_disable_hmm=Y`, **BAR1 131,072 MiB** per GPU (bigger than VRAM, so not the crippled-256 MB case),
`DmaRemapPeerMmio: 1`, ext4. Missing: `RegistryDwords` was **empty**. Not reachable from a guest:
`pcie_aspm=off`/`pcie_port_pm=off` (host root ports), `iommu=off` (passthrough needs the *host* IOMMU),
BIOS ReBAR/Above-4G/SR-IOV. Not applicable: `NCCL_P2P_LEVEL=SYS` - both GPUs report NUMA 0 with
identical affinity, so there is no cross-NUMA hop. And one honest divergence: the doc names the
override as critical where `nvidia-smi topo -m` shows **NODE**; ours shows **PHB**, which its own table
puts in the "often no" column. It was needed anyway - so **PHB is a new datapoint for that table.**

**The fix is one file plus a driver reload**, and the reload is the fiddly part: both the vLLM container
*and* `nvidia-dcgm`'s `nv-hostengine` pin `nvidia_uvm`, so `docker rm -f` the container and
`systemctl stop nvidia-dcgm` first, or `modprobe -r` fails with "Module nvidia_uvm is in use" while
appearing to succeed.

| | before | after |
|---|---:|---:|
| `can_device_access_peer` | **false** | **true** |
| 64 MiB P2P copy | 35.48 GB/s | **51.98 GB/s** (+46.5 %) |
| 256 B cross-GPU copy | 12.48 us | **8.34 us** (-33.2 %) |

51.98 GB/s is **~93 % of the doc's bare-metal same-NUMA band** (54-56 GB/s), inside a passthrough VM.
The 8.34 us is an end-to-end `copy_` including launch and sync, **not** the doc's 0.36-0.45 us P2P
write latency - different quantities, never to be differenced.

**Real decode, single variable, both arms gated PASS:**

| cell | TP2 no-P2P | TP2 +P2P | gain |
|---|---:|---:|---:|
| 512x256 single-stream | 149.6 | **159.6** | +6.7 % |
| 512x256 C4 per-request | 106.0 | **124.0** | **+17.0 %** |
| 512x256 C4 aggregate | 314 | **382** | **+21.7 %** |
| 512x256 C8 aggregate | 452 | **516** | +14.2 % |
| 512x256 C16 aggregate | 557 | **605** | +8.6 % |
| 4kx1k single-stream | 102.1 | **122.7** | **+20.2 %** |
| 4kx1k C8 aggregate | 389 | **438** | +12.6 % |
| 4kx1k C16 aggregate | 456 | **496** | +8.8 % |

**It also overturned a verdict I was one ladder away from publishing.** Pre-override, TP2 read as a bad
trade. Post-override TP2 beats the single card on **both** axes: **+18.4 %** single-stream and
**+11.4 %** C8 aggregate on 512x256, **+36.8 %** single-stream on 4kx1k. The earlier numbers were a
measurement of this VM's driver configuration wearing the costume of a fact about tensor parallelism.

**What it did not fix:** the SymmMem warning persists, because **P2P memory access and native P2P
atomics are separate capabilities** - the override buys the first, not the second, so vLLM's
symmetric-memory all-reduce stays unavailable and every gain above comes from the ordinary P2P data
path. A host with atomics may have more to give.

**Standing action for every future multi-GPU rental: write the override and reload the driver before the
first measurement.** Receipt: [`jarvis-p2p-override.json`](../receipts/jarvis-p2p-override.json).

## 20. MEASURED: TP2 against DP2, and the verdict splits exactly on the axis it was predicted to

Both arms on the same host with P2P enabled, same sr1 image, same shipping schedule
`[[1,4,3],[5,64,1]]`, same 262,144 window, same seed and shapes, **both gated PASS on five checks
including the 262k needle**. DP2 got an extra C32 rung because it was the arm that could still climb.

| cell | 1x (knee-c) | TP2 +P2P | DP2 |
|---|---:|---:|---:|
| 512x256 single-stream | 134.8 | **159.6** | 135.8 |
| 512x256 C4 per-req / agg | 108 / 330 | **124** / 382 | 121 / 372 |
| 512x256 C8 per-req / agg | 62 / 463 | 69 / 516 | **109** / **616** |
| 512x256 C16 per-req / agg | 37 / 558 | 40 / 605 | **68** / **850** |
| 512x256 C32 per-req / agg | - | - | **38 / 1072** |
| 4kx1k single-stream | 89.7 | **122.7** | 90.8 |
| 4kx1k C8 agg | 392 | 438 | **451** |
| 4kx1k C16 agg | 447 | 496 | **686** |
| 4kx1k C32 agg | - | - | **761** |
| knee label | C4 / C8 | C4 / C4 | **C8 / C8** |

**The prediction in §18 was right, and sharper than expected.** TP2 wins **single-stream** decisively -
**+17.5 %** over DP2 on the short shape (159.6 against 135.8) and **+35.1 %** on the 4k shape (122.7
against 90.8) - because halving the weight bytes each step streams is the only lever that touches
single-request decode on a weight-streaming-bound engine. DP2 wins **everything from C4 up**, and not
only on aggregate: at C8 it delivers **109 per-request against TP2's 69** and **616 aggregate against
516**, and at C16 **+40.5 % aggregate** (850 against 605).

**Why DP2 wins per-request too, which is the part worth understanding.** DP2 is not two engines racing
one workload; the router splits client concurrency, so **8 client requests are 4 per replica**. Each
replica therefore sits at its own C4 - fast, near the knee - while TP2 puts all 8 on one engine. This is
the coupling §18 predicted from the source: the depth schedule keys on
`len(num_scheduled_tokens)`, **requests per step per replica**, so under DP2 a client at C8 lands each
replica in the **depth-3** band rather than the depth-1 band. The schedule and the topology are not
independent knobs, and here they compound in DP's favour.

**Verdict for the campaign.** For a benchmark whose failures are **93 % timeouts**, per-request speed is
what buys score and aggregate is what buys wall clock, and DP2 is ahead on both at every concurrency the
campaign actually runs (`-n 16`). **Ship DP for the multi-GPU tiers.** TP is the right answer for
exactly one shape of problem - a single latency-critical stream, where it is worth +17.5 % to +35.1 % -
and it should be offered as that, not as the default. The 8x tier should therefore be **DP8**, with TP
kept as a documented single-stream option rather than a ladder rung.

## 21. MEASURED: do not use `--kv-cache-memory-bytes` here - the engine's own suggestion breaks long context

vLLM prints a suggestion when profiling leaves memory unused: *"Replace gpu_memory_utilization config
with `--kv-cache-memory-bytes 74719122432`"* (69.59 GiB) against the 62.38 GiB the profiler actually
took - the "+7.2 GiB of unclaimed KV" this plan listed as an open win. **Every value above the
profiler's own choice fails, and the last one fails in the worst possible way.**

| `--kv-cache-memory-bytes` | KV tokens | outcome |
|---|---:|---|
| unset (profiled, util 0.92) | 1,760,318 | **starts, all five gates PASS including the 262k needle** |
| **69.59 GiB** (the engine's own suggestion) | 1,963,883 (+11.6 %) | **will not start** - OOM, needed 4.09 GiB with 3.87 GiB free |
| 67.0 GiB | 1,890,658 (+7.4 %) | **will not start** - OOM mid graph capture, 192 MiB short of 45.56 MiB free |
| 66.0 GiB | 1,862,833 (+5.8 %) | **starts, then dies on first full-window request** |

**The 66 GiB arm is the dangerous one.** It starts cleanly, reports concurrency 6.72x -> 7.11x, and
passes liveness, generation, repeatability and MTP - **4 of 5 gates**. Then the 262,144-token needle
arrives and the engine is killed by `torch.OutOfMemoryError: Tried to allocate 94.00 MiB. GPU 0 has
94.97 GiB total of which 15.56 MiB is free`, surfacing as `EngineDeadError` and **HTTP 500**. A
configuration that boots, answers short prompts, and then dies on the long-context request is strictly
worse than one that refuses to boot - and it would have shipped if the gate did not include a
full-window needle. That single check is the reason this is a finding rather than an outage.

**Mechanism.** The suggested figure is derived from *initial free memory*, measured **before**
CUDA-graph capture, the reconstruct arena, and prefill activation for a full-window request allocate.
Claiming it starves exactly those. Capture alone is not even constant: **1.64 GiB** in the profiled
baseline against **1.98 GiB** in the 66 GiB run - which is why 67 GiB missed by 192 MiB, and why any
value near the edge cannot survive restart-to-restart variance on a stack already measured to shift by
tens of MiB between runs.

**Verdict: leave `--kv-cache-memory-bytes` unset on this configuration.** The profiler's 62.38 GiB is
the only value that survives a full-window request, so the "+7.2 GiB unclaimed" was never headroom - it
is working memory for capture, arena and long-prefill activation. The knob buys capacity, never decode
speed, and at TB shapes the pool is already 6.72x the window. Anyone who follows vLLM's own log
suggestion on a 96 GB card at a 262k window gets a server that boots and later dies under the exact
workload they enlarged it for. Todo closed as **measured and declined**, with the receipt
([`tb21-gate-1x-kvclaim66.json`](../receipts/tb21-gate-1x-kvclaim66.json)) recording the 4-of-5 pass
and the 500.

## 22. MEASURED: LMCache L1 CPU-DRAM on the 2x host - the baseline reproduces exactly, and our own gate patch makes it *worse*

Seven scored 38-request arms, one probe (`tools/apc_poison_probe.py` sha256 `a96168f9…`, byte-identical
to the 5090 run), one frozen prompt set (`prompts_sha256 9978404b…`), the pre-registered thresholds and
fail rule of [`lmcache-reuse-test.json`](../receipts/lmcache-reuse-test.json) unchanged. **Verdict: do
not enable LMCache, with or without the patch.** Receipt:
[`lmcache-l1-2x.json`](../receipts/lmcache-l1-2x.json), harness `tools/run_lmcache_test_2x.sh`, L1
configuration `tools/lmcache-l1-2x.env`.

| arm | LMCache | gate patch #403 | failing | corrupted | acceptance median |
|---|---|---|---:|---:|---:|
| L0 control | off | mounted | **0/38** | 0 | 0.619 |
| U1cold | on, fresh L2 | **absent** | **7/38** | 2 | 0.609 |
| U2warm | on, same server | **absent** | **7/38** | 1 | 0.602 |
| L1cold | on, fresh L2 | applied | **37/38** | 37 | **0.0** |
| L2warm | on, same server | applied | **38/38** | 38 | **0.0** |
| L3restart, **fresh** L2 | on, retained clean L2 | applied | **38/38** | 38 | **0.0** |
| L3restart, **poisoned** L2 | on, retained unpatched L2 | applied | **38/38** | 38 | **0.0** |

**The unpatched pair reproduces the 5090 baseline exactly: 7/38 and 7/38.** Different model edition
(hydrated, not context), different GPU, different driver, TP2 not TP1, docker not podman, 262,144 window
not 196,608 - and the same count, the same failing requests, and the same predictive rule. The
pre-registered rule (*a request fails iff a scored needle lies inside `[hit-1600, hit)`*) scores **7
predicted / 7 actual, zero false positives, zero false negatives in both arms**, and all 76 rows
reconcile as `hit_arm = hit_L0 + 1600` (or both-zero, or a warm full self-match) - the identical 76/76
reconciliation [`lmcache-fix.json`](../receipts/lmcache-fix.json) reported. On L0 the same rule flags 7
requests and **none** fails: the connector-absent control at identical reuse geometry, 7/7 correct.
Mean |chosen-logprob delta| vs L0 is **0.238/0.242** against the 5090's measured ~0.245. This is as
close to an independent replication as our records contain.

**Then the fix we filed made it catastrophically worse, for exactly the reason its own author flagged.**
`local-inference-lab/vllm#403` gates the hybrid+connector divergent-hit path, and its PR body predicted
0/38. Measured: **37/38 and 38/38**, U+FFFD replacement characters and out-of-script text on every
failing request, SpecDecoding acceptance median **0.0** - cosmicnag's symptom verbatim. The mechanism is
visible in one counter:

| arm | `external_prefix_cache_queries` delta | `external_prefix_cache_hits` delta | MP log `Retrieved` lines |
|---|---:|---:|---:|
| U1cold + U2warm (unpatched) | 88,760 | **0** | **0** |
| L1cold / L2warm (patched) | 110,780 / 97,980 | **59,200 / 60,800** | 158 |
| L3restart, either L2 | 113,980 | **76,800** | 76 |

Unpatched, **LMCache supplies zero tokens** - it is never asked to load, so the 7/38 is 100 % scheduler
side, confirming `lmcache-fix.json`'s correction of our upstream comments. The gate closes that path, so
vLLM finally *does* load from the connector (`ext > 0` for the first time in this investigation), and the
**LMCache MP retrieve path does not restore the lagging Mamba/GDN state either**. Divergence vs L0 jumps
from 0.24 to **4.36-4.50**, 18x, and the failure stops obeying the divergence window because there is no
longer a window - the state is gone. `lmcache-fix.json` listed *"whether LMCache retrieves restore
hybrid GDN state correctly when they DO supply tokens (ext > 0)"* as an open question. **Answered: they
do not.** #403 is necessary and insufficient; its 0/38 prediction is refuted and must be re-labelled
before anyone reads it as a green light.

**Both L3restart variants, as the ladder required.** Poisoned L2 (34 files, 2.71 GB, written by the
unpatched arms) stayed dirty - **confirmed**, 38/38. Fresh L2 (596 files, 47.5 GB, written by the
*patched* arms) was **not** clean - **refuted**, 38/38 - because a poison-free L2 is unreachable here:
every writer configuration is itself corrupt.

**The five-check gate is blind to this, so there is now a sixth check.** `tb21_gate.py` returned **PASS
on all five checks, 262k needle included**, against the same live server the probe had just scored 38/38
corrupted ([`tb21-gate-2x-lmcache-l1.json`](../receipts/tb21-gate-2x-lmcache-l1.json)). None of the five
reuses a cached prefix across a mamba block boundary, so none of them can see this defect class. **A gate
PASS must never be read as fidelity when a KV connector is attached.**

`--check-connector-reuse` (check 6, `connector_prefix_reuse`) is now implemented and **validated in both
directions on this host**. It issues request A at `4B + 500` tokens to publish cache blocks, then request
B at `4B + 900` over the same document, and scores two needles: N1 far from any boundary, and N2 placed
inside `[3B, 4B)` - the last block of B's hit, i.e. exactly the divergence window. Losing N2 while
keeping N1 is the hybrid+connector signature. It reports **`NOT_EXERCISED`, never PASS**, if B did not
reuse at least one block.

| server (same host, same image, same overlay) | checks 1,2,3,5 | check 6 | request B's answer |
|---|---|---|---|
| LMCache **off** ([receipt](../receipts/tb21-gate-2x-check6-lmcache-off.json)) | PASS | **PASS**, 4,800-token hit, N2 measured at depth 6,192 inside [4800, 6400) | `N1=143937167 N2=651805269` |
| LMCache **on**, gate patched ([receipt](../receipts/tb21-gate-2x-check6-lmcache-on.json)) | PASS | **FAIL** | `erein有意冲ityEngine匹awang联…` |

Request A, which reuses nothing, answered `143937167` correctly on **both** servers. Five checks green
and the sixth red on the same endpoint is the whole point. It defaults **off** so every previously
published gate receipt stays comparable, and is **mandatory by policy whenever `--kv-transfer-config` is
present**; the receipt now carries a `not_exercised_checks` list and a `verdict_scope` string saying so.

**What the feature would have bought, and why it does not matter.** L1 is CPU *pinned* DRAM
(`l1_memory_manager.py`: *"CPU pinned-DRAM L1 memory manager"*), configured at **160 GiB** of the host's
314 GB; it reached its full 171,798,691,840 bytes within ~13 s despite `--l1-use-lazy` defaulting True
(size it as an up-front reservation, not a ceiling), and peaked at 44.27 GiB / 27.7 % with no evictions.
On disjoint documents, warm vs cold as `/metrics` deltas between named snapshots: **32k, TTFT 6.3234 s ->
0.2387 s (26.5x, 0.9256 hit rate); 128k, 32.5141 s -> 0.6111 s (53.2x, 0.9760 hit rate)**. The prize is
real. The corruption detector then fired on **3 of the 4 warm variants**. A 53x prefill win that returns
U+FFFD is not a win.

**One more thing an operator needs to know before they try.** The shipping profile cannot run LMCache on
this model *at all*: with `--max-num-batched-tokens 8192` the connector refuses to initialise -
*"Mamba-hybrid models with LMCache require `block_size <= max_num_batched_tokens < 2 * block_size` … got
max_num_batched_tokens=8192, block_size=1600"*. At this model's 1600-token block the admissible band is
**[1600, 3199]**, so enabling LMCache costs a **2.6x cut in prefill chunk size** before correctness is
even discussed. We ran every arm at 2048, which is inside the band *and* the pre-registration's own
value, so the forced change removed a deviation instead of adding one. Note the irony: the code enforces
*"every block boundary gets a state snapshot"* for prefill chunking, and violates the same invariant on
the retrieve path.

**And that band collides head-on with our best measured prefill win, which settles the trade.**
[docs/47](47-kernel-gap-analysis.md) **F8** measures **+13-15 % on the linears** from moving the prefill
chunk `2048 -> 6144`, and **F5.1** adds **2.67x less redundant reconstruct** at the same chunk; F8 calls
`--max-num-batched-tokens 6144` *"the single largest prefill lever available"* (docs/47 recoverable table,
row 7, est. **+15-25 % PP**). **6144 is outside `[1600, 3199]`. LMCache and our largest measured prefill
lever are mutually exclusive on this model** - not a tuning tension, an admissibility one: the connector
refuses to initialise above 3199.

So the decision is not "corruption risk versus a 53x TTFT prize". It is **corruption risk, *plus* a 2.6x
prefill-chunk cut, *plus* forfeiting a measured +13-15 % (est. +15-25 % PP), in exchange for a 53x TTFT
prize that returns U+FFFD.** Every term except the prize is a cost, and the prize is unusable. There is no
version of this trade that clears.

**Disposition.** LMCache stays in the image and stays **disabled by default**, exactly as
[`tb21-image-sr1.json`](../receipts/tb21-image-sr1.json) `patch_layers[D]` has it. The remaining defect
is named: LMCache must store and restore the SSM/GDN state block alongside the attention KV it supplies
(what nixl does unconditionally in `_apply_prefix_caching`), or refuse hybrid models. Neither exists in
`0.5.2+glm52dcp.4`. A negative result, cleanly measured, on the box before it was released.

## 23. DETERMINATION: the §12 scaling rule fires, so no 4x TB number may be interpolated

§12 pre-registered the test: *if 2x-to-4x DP scaling deviates from linear by more than ~10 % on the
synthetic ladder, the 4x TB pass is mandatory; otherwise it may be skipped and the interpolation stated
on the card.* That test is now decidable, and it must be evaluated at **matched per-replica load** - DP2
at client concurrency C and DP4 at 2C put the same number of requests on each replica, which is the only
comparison in which "twice the GPUs" means "twice the work offered".

| requests per replica | 512x256 DP2 -> DP4 aggregate | of linear | 4kx1k DP2 -> DP4 aggregate | of linear |
|---:|---|---:|---|---:|
| 2 | 371.7 -> 654.5 (1.76x) | 88.1 % | 323.9 -> 611.7 (1.89x) | 94.4 % |
| 4 | 616.4 -> 1027.5 (1.67x) | **83.4 %** | 451.1 -> 858.9 (1.90x) | 95.2 % |
| 8 | 850.3 -> 1455.9 (1.71x) | 85.6 % | 685.5 -> 1156.1 (1.69x) | 84.3 % |
| 16 | 1072.0 -> 1958.3 (1.83x) | 91.3 % | 761.2 -> 1529.6 (2.01x) | 100.5 % |

**Verdict: the rule fires.** Deviation exceeds 10 % at **four of the eight matched points**, worst case
**16.6 %**. Scaling is not linear enough to interpolate through.

**Consequence, and it is a restriction rather than a task.** The 4x host was released to fund the 8x, so
the mandatory 4x TB pass will not be run. The rule existed to stop us publishing an interpolated tier
when scaling is nonlinear - so the honest outcome is that **the 4x TB row is simply absent, and no 4x TB
number may be inferred from the 1x and 8x rows.** The 4x *synthetic* ladder stands as measured
(DP4/TP4/TP2xDP2, all gated PASS, §20); only the task-suite number is unavailable. Any card or chart
showing tiers must leave 4x **blank rather than dashed-and-estimated**.

**Why it is sub-linear is worth naming, because it predicts DP8.** At matched per-replica load DP
replicas are nearly independent by construction - no shared weights, no shared KV, no collectives - so
sub-linearity can only come from **what the host shares**: memory bandwidth across the socket, the PCIe
root complex, and CPU-side scheduling for N engine processes plus N API servers. That the worst
deviation sits at *low* per-replica load (83.4 % at 2-4 requests each), where per-replica decode is most
bandwidth-hungry, is consistent with a shared-bandwidth ceiling rather than a collective cost - and it
agrees with docs/47's independent finding that the trellis GEMMs already run at 88-100 % of the
*measured* achievable bandwidth ceiling. The falsifiable prediction for the DP8 arm is therefore
**further sub-linear scaling, worst at low per-replica concurrency**; if DP8 instead scales *better* than
DP4 did, this explanation is wrong and must be replaced rather than patched.

## 24. MEASURED: DP8, and my §23 explanation is falsified by my own next measurement

DP8 gated **PASS on all five checks including the 262,144-token needle** (check 6, connector-reuse,
correctly reports SKIP - no KV connector attached), measured from the load driver **over the VPC**, which
is the campaign-shaped path and restores comparability with the 1x and 2x tiers.

| rung | 512x256 per-req / agg | 4kx1k per-req / agg |
|---|---|---|
| C1 | 135 / 135 | 90 / 90 |
| C4 | 135 / 370 | 93 / 339 |
| C16 | 118 / 1307 | 86 / 1239 |
| C32 | 80 / **1947** | 71 / 1607 |
| C64 | 67 / **2835** | 45 / 2214 |
| C128 | 35 / **3553** | 25 / 2782 |

**Peak aggregate 3,553 tok/s**, and the knee lands at **C32** on both shapes - against C16 for DP4 and
C8 for DP2. The knee therefore scales **linearly with replica count**, which is the clean expected
result and the single most useful planning number here: *a DP-N deployment holds its per-request floor to
about 4N concurrent requests.*

### The falsification

§23 explained DP2->DP4's sub-linearity as a **shared-host-bandwidth ceiling** and committed to a
falsifiable consequence: *"further sub-linear scaling, worst at low per-replica concurrency; if DP8
instead scales better than DP4 did, this explanation is wrong and must be replaced rather than patched."*

| requests per replica | DP2 -> DP4 | DP4 -> DP8 |
|---:|---:|---:|
| 2 | 88.1 % | **99.9 %** |
| 4 | 83.4 % | 94.8 % |
| 8 | 85.6 % | 97.4 % |
| 16 | 91.3 % | 90.7 % |

DP4->DP8 reaches **99.9-101.3 % of linear at low per-replica load** - the exact opposite of the predicted
pattern, and better everywhere except the 16-per-replica point. **The shared-bandwidth explanation is
withdrawn, not amended.**

### The replacement, and it is less satisfying but more honest

Both ratios compare **different physical hosts**, and Jarvis scales host resources with GPU count (2x:
56 vCPU / 314 GB; 4x: 112 / 629; 8x: 224 / 1,259), so per-GPU host resources are roughly *constant*
across the series - which removes the mechanism §23 proposed. Worse, the two ratios were not measured
through the same harness path: DP2 and DP8 were driven **over the VPC**, while DP4 had to be driven from
**localhost** because that host was VPC-isolated. So the "scaling deviation" is not a stable property of
the topology at all; it is dominated by **which box and which path** each arm happened to use, at a
resolution (5-17 %) coarser than the effect being claimed.

**What that means for the §12 rule: its premise was wrong, and its conclusion survives for a better
reason.** §12 assumed a scaling law could be read off two adjacent tiers. It cannot - not at this
resolution, across changing hosts. So the 4x TB tier still must be **left blank rather than
interpolated**, no longer because scaling is provably nonlinear, but because **we have no reliable
scaling law to interpolate with**. Cards and charts must show measured tiers only.

**The rule this project should carry forward:** a cross-tier ratio is only evidence if both arms ran on
the same host through the same harness path. Ours did not, and saying so is cheaper than publishing a
scaling law we cannot support.

## 25. CORRECTION to §22: the dominant LMCache defect is fp8-KV transfer, not GDN state restoration

§22 attributed the 37-38/38 corruption to the connector not restoring Mamba/GDN state, and a later source
walk moved the blame to a store-side null-block hole. **A live three-arm repro on a single card has now
superseded both.** The verdict is unchanged - **do not enable LMCache** - but the mechanism matters,
because it changes what a fix must touch and it changes what we tell upstream.

| arm (with #403 applied) | KV dtype | outcome |
|---|---|---|
| cold single-writer store -> APC-evict -> retrieve | **fp8** | **catastrophic corruption, garbage from token 1**, divergence 3.52 - the 2x host's 4.4 class - with **no poisoning precondition at all** |
| same sequence | **bf16** | **bit-clean**, divergence **0.0000** - the 48 GDN state pages round-trip exactly |
| one partial-APC-hit request interleaved between store and retrieve | bf16 | plausible-but-wrong text, divergence 0.71, **reproduced twice** |

**Three consequences, in order of importance.**

1. **Our serving profile runs fp8 KV, so LMCache could never have produced clean reuse on this stack -
   with or without #403.** That explains the 2x ladder's 38/38 far more directly than state restoration
   ever did. Candidate root causes are the fake-view byte arithmetic for 1-byte dtypes and unregistered
   fp8 scale surfaces; neither is fixed by anything we wrote.
2. **The GDN state machinery is now measured to work**, not assumed to. The bf16 arm's bitwise-clean
   round-trip is the direct refutation of §22's original claim, and it is a stronger form of evidence than
   the source reading that replaced it.
3. **A third, separate defect exists in the partial-retrieve path.** At bf16 an interleaved partial-APC-hit
   request still corrupts, and the store-side null clamp did **not** fix it - the server ledger proves the
   store-under-miss precondition never even fired, because a read TTL is not a since-write expiry. So the
   store-side hole remains source-true but was **not** what bit us; the clamp is archived as hygiene and
   labelled insufficient rather than shipped as a fix.

**Three operational facts nobody had written down**, each of which would have cost someone a day:
restarting the MP server **kills the live engine with no reconnect**; the connector **requires APC+align**
and refuses `mamba_cache_mode=none`, so the poisoning regime is mandatory rather than an edge case; and at
bf16 the align block is **800**, which forces `--max-num-batched-tokens` into **[800, 1599]** - a
different and tighter band than the 1600-block `[1600, 3199]` recorded in §22.

**Why this section exists rather than an edit in place.** Three mechanisms were proposed tonight and two
were withdrawn: retrieve-side state (measured false), store-side null blocks (source-true, did not fire),
fp8 transfer (measured, dominant). Overwriting §22 would hide that sequence, and the sequence is the
useful part - **each story was replaced because we kept testing it, not because we argued about it.**
Receipt: [`kernel-gap-lmcache-repro.json`](../receipts/kernel-gap-lmcache-repro.json), five arms with
server ledgers.


## §26 - The 8x campaign arm, and why we withdrew our own "quantization-suspect" bucket

**Headline: the overnight campaign resolved 56/89, and in doing so it destroyed the most
quant-unfavourable finding we had ever published about ourselves.** Receipt:
[`tb21-8x-2xclock-arm.json`](../receipts/tb21-8x-2xclock-arm.json), merged pass
[`tb21-8x-p1-merged.json`](../receipts/tb21-8x-p1-merged.json).

The arm: **single** pass, **doubled** agent timeout, 8x DP8, 8 shards x n=4 (C32, DP8's measured knee),
`vllm:gg-r34-tb21-sr1`, default reasoning effort. Wall clock **2h20.9m**. Exceptions: 22
`AgentTimeoutError`, 2 `RuntimeError`.

**First, the two RuntimeErrors are not a defect.** They are exactly `qemu-startup` and `qemu-alpine-ssh`,
the two tasks already classified `harness-void`. Nested-virtualisation qemu does not exist on a Jarvis VM,
so these two are **structurally unattemptable** here. After the Docker address-pool fix the RuntimeError
count is fully accounted for; nothing new broke.

### The correction

Pass 3 had re-run every twice-failed task on BF16 at stock clock, and 7 tasks resolved there but not on the
quant. We labelled those **`quantization-suspect`** - the honest reading at the time, and the only bucket
that pointed at our own weights.

**At 2x clock, all 7 of them resolve on the quantized model.** 7/7.

| bucket | was | resolved at 2x clock | remaining |
|---|---:|---:|---:|
| `quantization-suspect` | 7 | **7** | **0** |
| `inconclusive-timeout` | 35 | 10 | 25 |
| `capability` | 1 | 0 | 1 |
| `harness-void` | 2 | 0 | 2 |

The matched stock-clock result still supports the protocol label
`quantization-suspect`: BF16 passed where the quant did not under equal
conditions. The 2×-clock DP8 arm shows that label is **not stable to additional
resources**—all seven can pass—but it changes timeout and hardware/topology and
has no matched BF16 arm. It therefore cannot prove the failures were
fidelity-free or erase quantization as a contributor. The defensible update is
that zero tasks remain uniquely persistent across all tested quant conditions.

### What this arm is not

**56/89 is not a Terminal-Bench score**: timeout is non-stock. Published 31/89
single-pass and 44/89 best-of-two stock-clock results remain. The 31→56 pair
changes both timeout and tier, so the +25 cannot be assigned to either factor or
used as a quantization-fidelity attribution.

### Five regressions, reported not netted out

Against best-of-2 this arm **gained 17 and lost 5** for +12 net. The five - `db-wal-recovery`, `dna-insert`,
`filter-js-from-html`, `mcmc-sampling-stan`, `polyglot-c-py` - had each succeeded on *one of two* stock-clock
attempts and failed on this **single** attempt. That is ordinary best-of-1 versus best-of-2 variance, and it
is the reason the unresolved set decomposes as **33 = 25 inconclusive-still + 1 capability + 2 harness-void +
5 single-pass regressions** rather than the 28 a naive subtraction would predict. Any future arm that quotes
a gain against 44/89 owes the same decomposition.


## §27 - The topology ladder is complete, and one of its four arms does not exist

**Receipt: [`tb21-8x-topology-ladder.json`](../receipts/tb21-8x-topology-ladder.json).**
Three planned arms ran; the current checkpoint/runtime refused TP8.

### The current EXL3 checkpoint refuses TP8

```
ValueError: EXL3 TP output slice must be 128-aligned, got start=62080, size=31040
  exl3.py:2848 _slice_exl3_tensor  <- exl3.py:2904 _shard_tensors_for_tensor_parallel
```

EXL3 trellis tensors shard only on **128-element boundaries**. Reconstruct the tensor from the failing
slice: `31040 x 8 = 248320 = 128 x 1940`, and **`1940 = 2^2 x 5 x 97`**.

| TP | slice | tiles of 128 | |
|---:|---:|---:|---|
| 1 | 248320 | 1940.00 | aligned |
| 2 | 124160 | 970.00 | aligned |
| 4 | 62080 | 485.00 | aligned |
| **8** | **31040** | **242.50** | **refused** |

The tile count divides by four but not eight. **TP4 is the largest width this
checkpoint and loader can shard.** An eight-GPU deployment therefore needs at
least two data-parallel replicas unless a future checkpoint pads the head or a
runtime implements a different sharding strategy; no flag changes the current
bytes.

### The three real arms (512x256 / 4kx1k, all three arms ran both)

| | single-stream | peak agg @C128 | knee | DP degree |
|---|---:|---:|---:|---:|
| **DP8** | 135.2 | **3552.6** | **C32** | 8 |
| **TP2xDP4** | **153.2** | 2438.2 | C16 | 4 |
| **TP4xDP2** | 146.5 | 2205.6 | C8 | 2 |

**T4, the finding worth keeping: the knee tracks the data-parallel degree, not the GPU count.** DP8 knees at
C32, TP2xDP4 at C16, TP4xDP2 at C8 - **all exactly 4 x DP**, and the two shapes agree exactly. Our earlier
"knee ~ 4N" wording was ambiguous; the correct statement is **N = number of data-parallel replicas**. Adding
TP width *inside* a replica does not raise the knee, it lowers it proportionally, because it buys
single-stream speed with the very parallelism the knee is counting.

**T2/T3, the nuance that stops a wrong recommendation: the latency winner depends on prompt length.** At
512x256 the order is TP2xDP4 > TP4xDP2 > DP8, so **TP4xDP2 is strictly dominated** - worse single-stream
*and* worse peak than TP2xDP4, with no short-prompt reason to choose it. But at **4kx1k it inverts**:
TP4xDP2 109.3 > TP2xDP4 105.8 > DP8 90.1. Wider TP only pays once the prompt gives it enough work to split.

Zero refusals in every cell of every arm, C1 through C128.

**Serving recommendation:** DP8 for throughput; TP2xDP4 for latency on short prompts; TP4xDP2 only if
prompts run past ~4k; never TP8. Size concurrency to **4 x data-parallel degree**.

### A methodology trap we walked into and had to undo

The TP4xDP2 arm was first gated with `--tp-smoke`, which **silently downgrades the needle from 262,144 to
32,768 tokens**. That gate passed in 16 s and would have been quietly incomparable to the DP8 and TP2xDP4
gates, both of which used the full needle. We caught it on the wall-clock smell, re-gated at the full 262k
(PASS, 62 s, connector-reuse PASS too) and recorded the rule in the receipt: **never compare a `--tp-smoke`
gate against a full-needle gate.**

### Which tensor forbids TP8, measured

The traceback does not name the module, so we read the safetensors headers on the host rather than guess.
`config.vocab_size = 248320`; `lm_head.trellis` has shape **[320, 15520, 96]** and `15520 x 16 = 248320`;
`lm_head.svh` is [248320] and `embed_tokens.weight` is [248320, 5120]. Those are the only three tensors in
the checkpoint carrying that dimension. **The blocking tensor is `lm_head`, and the offending dimension is
simply the vocab size.**

That lands on something docs/47 already measured from the other direction: `lm_head` **streams 4x per decode
step** and is the largest single term in the 20.5 GB/step decode numerator. **The tensor that dominates
decode bandwidth is the same tensor that forbids TP8.** It is also exactly the tensor the b12x gate patch
reroutes (`N=248320`), so all three threads of this investigation converge on one weight.

**An arithmetic remedy exists, is not filed, and is not for this build.** Padding the head from 248,320 to
**248,832** rows (+512, **+0.21 %**) gives `248832 = 128 x 1944` with `1944 = 8 x 243`, making every TP in
{1,2,4,8} 128-aligned. This is **verified arithmetically only - not measured** - and it changes published
bytes, so it is a suggestion for a *future* build and never a retrofit of a shipped quant. We have not filed
it upstream; new upstream filings need owner approval.


## §28 - The bare-metal patch A/B, and the harness that could not see the answer

**Receipt: [`kernel-gap-bare-metal-ab.json`](../receipts/kernel-gap-bare-metal-ab.json).** This was meant to
turn KernelGap's predicted **+3-15%** bracket into a number. **It did not, and the reason is the interesting
part: the effect is smaller than our own measurement noise.**

Setup: real docker (no proot), single GPU on the 8x host, 5 repeats per cell, rungs C1-C8, three shapes.
Before running anything the factorial design was verified at the byte level - `exl3.py` changes only in the
gate arms, `linear.py` only in the transpose arms, both in the "both" arm - so no arm could silently be the
baseline.

| | valid cells | median delta | mean delta | range |
|---|---:|---:|---:|---|
| aggregate tok/s | 11 | **+0.98%** | +1.41% | -1.87% .. +8.94% |
| per-request tok/s | 11 | **+1.44%** | +2.24% | -0.55% .. +8.94% |

Now the number that governs all of those: **the median within-arm coefficient of variation is 2.84%, and in
the short-prompt cells it reaches 13.9-18.0%.** The noise floor is *larger than the signal*. The headline
"+8.94%" at 30kx2k C1 compares two overlapping samples whose baseline repeats alone span **70.5 to 93.6
tok/s**. Nine of eleven cells favour the gate (sign test **p = 0.065**), which is weak evidence of a small
positive effect and **nothing more**. **The +3-15% bracket is not confirmed, and no magnitude is supportable
from this run.**

One cell, `30kx2k/C8`, is void **and it is my fault**: I tore the server down to reclaim the host for
production 1M serving while that cell was in flight, which is why it shows 24 refusals and aggregate 0.0. It
stays in the receipt so the hole is visible. Arms **C** (transpose) and **D** (both) never ran for the same
reason; their images are built and byte-verified, so they remain runnable.

### Reconciling this with KernelGap's +8.0%

These are not contradictory results, they are **one result seen at two scales**, and Amdahl is the bridge.
KernelGap timed the *operation*; this timed the *whole server*. `in_proj_ba` is **5.2% of decode GPU time**
by KernelGap's own instrumentation, so:

| component share | 1.5x kernel | 2.0x | 3.0x | 4.4x |
|---|---:|---:|---:|---:|
| 5.2% (`in_proj_ba`) | +1.76% | +2.67% | +3.59% | **+4.19%** |
| ~8% (gate target) | +2.74% | +4.17% | +5.63% | **+6.59%** |

**Even a 4.4x kernel win on a 5.2% component caps the end-to-end gain at +4.19%.** So a genuine large kernel
speedup and a ~1% unresolvable server-level delta are the same fact. The mistake was mine in framing: I
expected a server-level ladder to confirm a kernel-level bracket it never had the resolution to see.

### What would actually resolve it

Resolving a 1.5% effect against a 2.84% CV needs on the order of **29 repeats per cell, not 5**. The cheaper
and better answer is to stop trying: **measure at the kernel level, where the signal is 8-15% and the noise
is far smaller, then use the Amdahl bound above to state the server-level consequence** rather than trying to
observe it. That is now the standing method for this class of patch, and issue #406 should carry the Amdahl
framing instead of an end-to-end promise.


## §29 - Our repeatability gate was passing by comparing nothing

**Receipt: [`gate-check3-vacuous.json`](../receipts/gate-check3-vacuous.json).** Found while reading a 1M
needle receipt closely enough to notice a hash I recognised.

Every `frozen_prompt_repeatability` row in every gate receipt we have published carries the same digest:

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

**That is the sha256 of the empty string.** The check hashed the OpenAI `content` field; Qwen3.8 thinks by
default at `xhigh` effort; the probe caps generation at 64 tokens; so the entire budget was consumed inside
`<think>` and routed to `reasoning_content`, leaving `content` empty. `completion_tokens` was 64 on every
row - **the model was generating the whole time, the text just never reached the field being hashed.** The
gate then compared eight empty strings, found them equal, and reported **PASS**. It has never once tested
determinism on this model.

### What this does and does not touch

**Unaffected: checks 1, 2, 4, 5.** Liveness, generation proof, needle retrieval and MTP sanity were all
substantive. The 262k and 786k needle passes are real and unchanged, and **no fidelity conclusion in this
project rests on check 3.**

**Invalidated: any claim that we verified determinism.** We did not. "Five-check gate PASS" must be read as
*four substantive checks passed and one contributed no evidence*, and the cards are being amended to say so
rather than left implying more than they proved.

### The fix, and what it immediately found

Identity is now `sha256(content + NUL + reasoning_content)`, and - more importantly - the check carries a
**vacuity guard**: it FAILS outright if every response returned zero content *and* zero reasoning, so it can
never again pass by comparing nothing.

Re-run on the live 1M endpoint it reports **substantive 8/8** (reasoning 261-310 chars) and **7 of 8
identical, with prompt 1 diverging** (299 vs 261 reasoning chars). The fixed check found real nondeterminism
in its first execution.

**And the limit on that result has since been closed, from an unexpected direction.** The divergence was
measured *under concurrent production load*, so the candidate causes were continuous-batching
nondeterminism (batch composition changes reduction order, which changes logits) versus engine
nondeterminism that would be present even at concurrency 1. §31's controls settle it: on a quiet card at
`--max-num-seqs 1`, a same-server replicate over **384 positions** is **exactly 0.0e+00 — bit-identical,
top-1 agreement 100 %**. **Engine nondeterminism at concurrency 1 does not exist on this card**, so the
7-of-8 seen under load is **batch composition**, which is a known vLLM property and **not** a
quantisation defect. That attribution is now made, on evidence, and the quiet-host re-run it was waiting
for is no longer needed.

**The transferable rule: a hash-equality check must assert that it hashed something.** An empty-vs-empty
comparison is the most confident-looking false pass available, and it survived every review we did because
the digest was constant and the status was green.


## §30 - What 1M context actually costs, measured

**Receipt: [`1m-context-effects.json`](../receipts/1m-context-effects.json).** The 8x host now serves this
checkpoint at **1,000,000 tokens** using Qwen's official static-YaRN recipe. Relative to the shipped
`config.json` that override changes exactly three things — `rope_type` from `default` to `yarn`, plus
`factor: 4.0` and `original_max_position_embeddings: 262144` — because `mrope_section`, `mrope_interleaved`,
`partial_rotary_factor` and `rope_theta: 10000000` are already what Qwen prescribe.

### Retrieval works, all the way up

| window | doc tokens | needle | wall |
|---:|---:|---|---:|
| 262,144 | 259,303 | **PASS** | 125 s |
| 524,288 | 520,528 | **PASS** | 355 s |
| 786,432 | 781,653 | **PASS** | 720 s |
| **1,000,000** | **994,755** | **PASS** | 1,205 s |

That top row is retrieval at **99.5 % of a one-million-token window**. (The 1M gate's *overall* verdict reads
FAIL, and that is not retrieval: one probe in check 3 timed out under production load, and check 3 was
independently found vacuous — §29.)

### KV is not the problem, and the reason is architectural

Only **16 of the 64 layers carry KV** — the other 48 are linear-attention GDN with O(1) state. At fp8 that is
`2 × 4 kv-heads × 256 head_dim × 1 B × 16 layers` = **32 KiB per token**. The engine reports **1,879,687 KV
tokens per GPU (62.45 GiB)**, and 1,879,687 × 32 KiB = 61.6 GiB, so the arithmetic closes. It also closes
against an outside source: the vLLM recipe page quotes 6.6M KV tokens at 1M context on a 288 GB GB300, and
6.6M × 32 KiB = 206 GiB, which fits that card.

So a full 1M sequence costs **32 GiB** of KV against a 62.45 GiB pool — **1.88 per GPU, ~15 host-wide**.
**Raising the window did not shrink the pool** (262k config: 1,776,428 tokens; 1M config: 1,879,687).

**The caveat that matters more than the numbers:** paged KV allocates per token *used*, not per
`--max-model-len`. Setting the window to 1M **reserves nothing**. Real agent traffic at 10-100k tokens is
untouched by the 1.88 figure, which binds only for sequences that actually fill the window.

### Prefill is super-linear, and the exponent is still climbing

Fit `wall ~ n^k` across the four rungs:

| segment | tokens × | time × | **k** |
|---|---:|---:|---:|
| 262k → 524k | 2.000 | 2.840 | **1.506** |
| 524k → 786k | 1.500 | 2.028 | **1.744** |
| 786k → 1M | 1.272 | 1.674 | **2.144** |

The segment exponents rise and cross 2. That is consistent with a hybrid of 48
linear-attention layers and 16 quadratic full-attention layers, but these
contended end-to-end wall times do not isolate that mechanism. A pre-registered
single-exponent prediction of 1,043 s missed the 1,205 s result by 15.5 %;
one global exponent is inadequate for this measured ladder.

**The consequence for pricing long context:**

| window | wall vs 262k | tokens vs 262k | **cost per token** |
|---:|---:|---:|---:|
| 262,144 | 1.00× | 1.00× | 1.00× |
| 1,000,000 | 9.64× | 3.81× | **2.53×** |

The 1M request consumed 2.53× more wall time per prompt token than the 262k
request in this contended needle run. The measurement includes decode, HTTP,
scheduling, and varying co-tenant load; it is not an engine-only prefill price.

**Limits, stated plainly.** Every wall-clock figure was measured while a TB2.1 arm ran at C16 on the same
endpoint, so absolute rates are depressed by contention, and the longer rungs span more wall clock and
therefore more varied contention. **The monotone rise is robust; the exact k values are not good to three
digits.** One needle position, one document generator — this is retrieval, not long-context reasoning.

## §31 - MEASURED: static YaRN costs at short context; the window-only effect is unresolved

**Receipt: [`yarn-short-context-penalty.json`](../receipts/yarn-short-context-penalty.json).** Probe-set
index: [`yarn-short-context-probes-index.json`](../receipts/yarn-short-context-probes-index.json).
Local RTX PRO 6000, idle and dedicated; the busy 8x rental hosts were not touched. **No timing number
appears in this section or its receipt as a result: this box runs vLLM under proot, which traces syscalls
and inflates dispatch, so it is fit to measure fidelity and unfit to measure speed.**

Qwen's own card warns that every open-source framework implements YaRN **statically** - the scaling is
applied at every length, not only past the native window - and advises enabling it only when long context is
needed, because it "may impact performance on shorter texts". Nobody publishes a number for that. We needed
one, because the production endpoint has to default to either 1M-YaRN or 262k-native. §30 measured what the
1M window *buys* and what prefill *costs* at the top of it; this section measures what enabling it costs the
other 99 % of traffic, the requests that never go near 262k.

### The arms, and the one variable

48 deterministic prompts, 12 each at **512 / 2,048 / 8,192 / 32,768 tokens**, cut from this repository's own
`docs/*.md`, cards and `receipts/*.json` in a recorded order, windows disjoint and allocated round-robin so
no length bucket is confounded with a region or genre of the corpus. Every prompt's token count was
re-verified by the **live server's `/tokenize`** in every arm and the sha256 of the id list asserted equal
across arms. Regenerating the probe set reproduced it **bit-exactly** (`sha256 374d3021...`).

Scoring is **teacher-forced**: the native arm generated an 8-token greedy continuation per prompt, then every
arm was replayed position by position (`prompt_ids + forced[:j]`, `max_tokens=1`, `temperature=0`, seed
314159, `logprobs=20` with `return_tokens_as_token_ids` so distributions are keyed by integer token id and
not by a detokenised string that can collide). 384 scored positions per arm, all conditioned on
byte-identical context, so a divergence at position 3 cannot be an artefact of the arms having walked
different paths at position 0. **Shared-support rule, stated once:** `S = P_top20 ∩ Q_top20`, both arms
renormalised over `S`, mass outside `S` **dropped, not smoothed** - the server never reports `q` for a token
outside `Q`'s top-20, and an epsilon filler would manufacture a large log-ratio out of a number nobody
measured. Mean shared support was 18.3 of 20 and the dropped P-mass averaged 3.7e-03; a `q_min` surrogate
variant moves the headline from 1.0566e-02 to 1.0717e-02, so the rule is not load-bearing.

Four arms, everything else pinned identical (exl3 + mxfp8, fp8 KV, `--mamba-cache-mode align`,
`--enable-prefix-caching`, `FULL_DECODE_ONLY`, util 0.92, `--max-num-seqs 1`,
`--max-num-batched-tokens 8192`, seed 314159):

| arm | rope | window | role |
|---|---|---:|---|
| **A** NATIVE | shipped `rope_type: default` | 262,144 | reference P, today's default |
| **B** NATIVE1M | shipped rope | 1,000,000 | window control |
| **C** YARN | Qwen's 1M recipe (`yarn`, factor 4.0, orig 262144) | 1,000,000 | candidate default |
| A' / A'' | arm A again, same server / cold reboot | 262,144 | resolution floor |

**The override provably took effect.** With `mrope_section` present, vLLM routes `rope_type: "yarn"` into
`MRotaryEmbedding` with `scaling_factor=4.0`, which swaps YaRN's `inv_freq` in *and* multiplies the whole
cos/sin cache by `mscale = 1.138629`. `mscale` applies at **every** position - that is exactly why static
YaRN cannot be free at short context. Reproduced from vLLM's own `get_rope`: 99.97 % of cache entries differ
over positions 0-32,775, and the max difference over positions 0-31 is **0.140625 = mscale-1 rounded to
bfloat16**. So a null result here would have meant "tolerated", never "ignored".

### The numbers

| comparison | mean top-20 KLD | p99 | top-1 agreement |
|---|---:|---:|---:|
| A vs A' - same server, warm cache | **0.0** (bit-identical) | 0.0 | 100.0 % |
| A vs A'' - **cold reboot, same launch line** | **1.180e-03** | 1.08e-02 | 97.9 % |
| A vs B - **window only**, native rope at 1M | 1.248e-03 | 9.84e-03 | 97.7 % |
| B vs C - **rope only**, matched 1M window | 1.071e-02 | 7.57e-02 | 93.8 % |
| **A vs C - the production decision** | **1.057e-02** | 7.23e-02 | **93.2 %** |

95 % CI on the headline, bootstrapping whole prompts (the 8 positions inside a prompt are not independent):
**[8.42e-03, 1.29e-02]**, 10,000 resamples, 48 clusters.

Two bounded findings follow. First, the same-server concurrency-1 replicate is
bit-identical over these 384 positions. That rules out same-process variation
for this probe set; it does not prove vLLM deterministic in general. The
production-load divergence is associated with batch composition/load because
that is the changed condition, but its magnitude still needs the dedicated
control in §34.

Second, the **window-only effect is unresolved**: native rope at a 1M configured
window measures 1.248e-03 against a 1.180e-03 cold-reboot control. The experiment
cannot distinguish that delta from its floor. The matched rope change is
resolved at 1.071e-02, so static YaRN—not merely raising `max_model_len`—is the
load-bearing measured difference. Compute cost remains separate.

### Per-bucket: the registered hypothesis is REFUTED

| prompt tokens | mean KLD | cold-reboot floor | **penalty / floor** | top-1 agreement |
|---:|---:|---:|---:|---:|
| 512 | **7.650e-03** | 8.49e-04 | 9.0x | 96.9 % |
| 2,048 | **7.298e-03** | 8.64e-04 | 8.4x | 94.8 % |
| 8,192 | **1.483e-02** | 1.69e-03 | 8.8x | 89.6 % |
| 32,768 | **1.249e-02** | 1.31e-03 | 9.5x | 91.7 % |

**We set out to test "static YaRN costs MORE at short lengths". It does not. The hypothesis is REFUTED.**
The two *short* buckets are the two *cheapest* measured - 7.65e-03 at 512 and 7.30e-03 at 2k against
1.48e-02 at 8k - so **the penalty is worst in the MIDDLE**, and the long half of the set is 1.83x worse than
the short half (1.37e-02 against 7.47e-03). It is not a near miss: the 8k bucket carries about **twice** the
2k bucket's mean KLD, and **top-1 agreement moves the same way** (96.9 % at 512 down to 89.6 % at 8k), so
both metrics refute it independently.

A separate observation explains the *shape* but does **not** rescue the hypothesis: the cold-reboot floor
rises with length in nearly the same proportion (8.49e-04, 8.64e-04, 1.69e-03, 1.31e-03), so dividing each
bucket by its own length-matched floor flattens the curve to **9.0x, 8.4x, 8.8x, 9.5x** - about 9x the floor
everywhere. Read that as *"static YaRN's cost tracks how hard the position already is to resolve"*, which is
also what the mechanism predicts (`mscale` is one scalar applied at every position, with no length
dependence). Do **not** read it as support for a short-length penalty.

**The production decision never depended on the hypothesis being true.** It depends on the penalty being
present and resolved at *every* length, which it is: the cheapest bucket measured is 2k at 7.30e-03, still
8.4x its own floor and 16x the adversarial calibration swap in absolute terms. On 12 prompts per bucket the
exact bucket *ordering* is suggestive at best; what is solid is the refutation and the fact that **no length
escapes the penalty**.

### Against our own yardsticks - and this is where we decline to overclaim

The published body-fidelity ladder uses full-vocabulary hidden-state replay; this
receipt uses a top-20 serving-API diagnostic with a 1.18e-03 cold-reboot floor.
Those estimators have different support, references, and systematics. Their
absolute values and floor-relative ratios cannot be used to rank YaRN against
weight quantization, calibration changes, or converter variance.

Within **this** diagnostic, the static-YaRN delta is 1.057e-02—about 9× its own
control—and is comfortably resolved. That supports "real and non-negligible on
this probe," not "larger/smaller than the quantization step."

The user-visible version: **19 of 48 prompts change their 8-token greedy continuation** under YaRN at
temperature 0, against **7 of 48 for a plain reboot** of the identical configuration. And a caveat that cuts
the other way: all 26 top-1 disagreements landed on positions where the *reference itself* was unconfident
(mean reference top-1 probability 0.236; **none above 0.9**), which argues for a smaller task impact than the
6.8 % disagreement rate suggests. Task accuracy is **not measured here**; a Terminal-Bench or needle arm
under YaRN is the honest next step.

### Production recommendation

**Default to 262,144 native. Do not make 1M-YaRN the default. If 1M is genuinely needed, run dual endpoints
on the same weights** - a native 262k endpoint carrying default traffic and a separately addressed 1M-YaRN
endpoint - **and route to YaRN only for requests that actually exceed the native window.**

The measured case, and nothing beyond it: the penalty is resolved at 9x its estimator's own floor; it is
present at **every** length including 512 tokens, so "most of our traffic is short" is not a defence for
enabling it globally; it is the **rope**, not the window, so there is no *fidelity* argument against a
long-window endpoint as such - only the compute argument in §30; and Qwen's own guidance says to enable YaRN only when long context is needed.
Flipping the default taxes 100 % of traffic to serve the fraction of requests above 262k, and the
dual-endpoint alternative costs routing, not fidelity. One probe set, one card, 48 prompts, 384 positions -
the direction is robust on three controls and a mechanism proof; the exact number is not a constant.


## §32 - Real multimodal load on the 8x at 1M, measured

**Receipt: [`multimodal-load-8x.json`](../receipts/multimodal-load-8x.json).** The owner drove text, image and
video traffic through `omp` at the DP8 1M endpoint while a TB2.1 arm ran at C16 underneath. Mixed traffic, so
this is the realistic case rather than a clean benchmark. Video is confirmed in the server log by **48
`video_processing_qwen`** events.

### All eight GPUs saturate, and they balance

| | mean util | cross-GPU spread | idle samples |
|---|---:|---:|---:|
| needle ladder, 12:50-13:20 | 63.6-78.2 % | up to **72.1 pts** | 20-35 % |
| **multimodal, 13:20-13:31** | **92.9 %** | **14.7 pts** | **5.6 %** |

Median utilisation is **97-100 % on every one of the eight cards**, power runs **382-575 W** against a 600 W
limit, SM clocks hold **2347-2407 MHz**, temperatures 37-67 °C, **zero uncorrected ECC**.

**And a mistake I nearly made.** A single `nvidia-smi` sample showed GPU 0 at 0 % while seven cards were at
97 %, and an earlier sample had shown GPU 7 idle instead. That looks exactly like a DP load-balancing defect,
and I was one step from writing it up as one. Segmenting the archived telemetry by time killed that story:
**the idleness belongs to the needle ladder, which issues one enormous request at a time and therefore leaves
seven of eight replicas idle by construction.** Under concurrent mixed load the spread collapses from 72
points to 14.7. **Workload shape explained it, not topology** - and a single instantaneous sample could never
have distinguished the two.

**Queueing: none.** `num_requests_running` per engine was `[1,3,1,2,2,3,2,2]` with
`num_requests_waiting` **0 on all eight** and zero capacity stalls, while saturated.

### Multimodal is a prefill workload

| | prefill tok/s | decode tok/s | input:output |
|---|---:|---:|---:|
| multimodal | **26,183** (3,273/GPU) | 878 (110/GPU) | **29.8 : 1** |
| text-agent window | 9,642 | 623 | 15.5 : 1 |
| TB2.1 campaign reference | — | — | 9.94 : 1 |

Prefill throughput is **2.72×** the text-agent window while decode rises only 1.41×, because images and video
expand into very large token counts. **At 29.8:1 input:output — three times the text-only campaign's 9.94:1 —
multimodal cost is almost entirely input-side, so any price sheet that leads with output tokens misprices it
badly.**

### The prefix-cache result is the one to keep

**Prefix cache hit rate rose from 26.9 % to 48.0 %, +21 points**, when the owner's subagent traffic arrived.
That is the **first measured evidence** for a benefit this project had only ever asserted: omp subagents share
system prompts and tool schemas, so their prefixes collide and the cache pays.

**Unit caveat, stated because it is easy to over-claim.** vLLM's `prefix_cache_queries` counts **block
lookups, not tokens**. 48.0 % is a block-level hit rate and is **not** the fraction of prompt tokens skipped;
converting it into a token discount needs a token-level measurement we have not made.

MTP acceptance also held up under multimodal traffic, **61.7 % against 58.9 %** on text — the shipped depth
schedule `[[1,4,3],[5,64,1]]` needs no change for vision work.

**Limits.** Mixed traffic throughout; the load was driven interactively so its composition is not
script-reproducible; counter deltas come from one poller against a DP8 endpoint whose `/metrics` aggregates
across engines, so rates are host-wide; and the engine exposes no modality-split prompt tokens, so 29.8:1
bundles text and vision tokens together.


## §33 - The task-level YaRN test: 48/89, and why that is corroboration rather than proof

**Receipt: [`tb21-1m-yarn-arm.json`](../receipts/tb21-1m-yarn-arm.json).** §31 measured the static-YaRN penalty
at the token level on a quiet card. This is the same question asked at the task level, on the production
endpoint, with a full Terminal-Bench 2.1 pass.

| arm | resolved | wall | concurrency | rope |
|---|---:|---:|---|---|
| 262k native | **56/89** | 2.35 h | C32 | native |
| 1M static YaRN | **48/89** | 3.60 h | C16 | YaRN factor 4.0 |

**Net −8, decomposed: 43 held, 13 lost, 5 gained.**

**The confound runs the wrong way for the sceptic, which is the useful part.** Concurrency differs — C16 here
against C32 there — and *less* contention should mean *fewer* contention-induced timeouts. So the confound
favours the 1M arm, and a −8 result is not explained by it. Wall clock, by contrast, is **not** comparable at
all: 1.53× at half the shard width is a function of concurrency, not rope, and no wall-clock claim is made.

### The honest statistics

With **18 discordant pairs**, a 13–5 split is a **two-sided sign test p = 0.096**. Restricting to non-timeout
failures — 8 lost against 2 gained — gives **p = 0.109**. **Neither reaches conventional significance.** Both
arms are single-pass, and the 262k arm alone showed 5 regressions against its own best-of-2 reference, so churn
of this magnitude is expected. **This arm corroborates §31; it does not independently prove anything.**

### One feature of the data is genuinely YaRN-shaped

**The total `AgentTimeoutError` count is identical at 22 in both arms.** The clock budget did not change; the
timeouts simply landed on different tasks (5 lost, 3 gained, net −2). The asymmetry lives almost entirely in
the **non-timeout** failures — **8 lost against 2 gained** — and non-timeout failures are where a *fidelity*
effect shows up, whereas a scheduling effect would move the timeout count. That is the most YaRN-consistent
structure in the result, and it is still only p = 0.11.

### What this does and does not change

**The recommendation is unchanged and still rests on §31**, whose 95 % CI of [8.42e-03, 1.29e-02] excludes zero
by a wide margin against a **bit-identical** same-server control — a far stronger instrument than 89 binary task
outcomes. Run **262k native by default**; add a separate 1M-YaRN endpoint only for requests that genuinely
exceed the native window.

**Nothing here revises a published number.** 31/89 (single pass, stock clock), 44/89 (best-of-2, stock clock,
the permanent lower bound) and 56/89 (single pass, 2× clock, 262k) all stand exactly as they were, and 48/89 is
likewise not a leaderboard figure — its agent timeout is non-stock.


## §34 - Closing both open fidelity questions on the production topology

Two questions were registered in docs/29 as needing the config-swap window. Both are now answered on the 8x
host, using the **identical 48-prompt probe set** as §31 (sha `374d3021…`) and §31's **own** `compare()` code,
so these are replications rather than similar-looking measurements.

### P1 — the YaRN penalty transfers to DP8, within 3 %

| | mean top-20 KLD | top-1 agreement |
|---|---:|---:|
| one GPU, concurrency 1 (§31) | 1.057e-02 | 93.23 % |
| **DP8, eight GPUs (this)** | **1.0265e-02** | **93.75 %** |

Ratio **0.971×**, and the DP8 value falls **inside §31's 95 % CI** of [8.42e-03, 1.29e-02]. The per-bucket
shape reproduces too, including the counter-intuitive part — worst in the **middle** of the length range
(8k: 1.314e-02 on DP8 against 1.483e-02 on one GPU), not at short lengths.

Rope is per-layer arithmetic, so topology-independence was the *expectation*; the point of measuring was that
this project does not ship expectations as results. §31 is now a two-machine, two-topology result.

### P2 — no concurrency cost resolved on this production-load probe

Same config (1M-YaRN), same probe set, **quiet versus real production load** (a TB2.1 arm at C16 plus
interactive traffic):

| | mean top-20 KLD | reading |
|---|---:|---|
| same-server replicate, concurrency 1 (§31) | 0.0e+00 | bit-identical |
| cross-boot replicate (resolution floor) | 1.180e-03 | the floor |
| **quiet vs LOADED on DP8** | **1.278e-03** | **1.08× the floor** |
| the static-YaRN rope change | 1.071e-02 | 8.4× larger |

Quiet-versus-loaded measures 1.278e-03, essentially the 1.180e-03 cross-boot
control, with 98.70 % top-1. No additional concurrency effect is resolved on
these 48 prompts/384 positions. This is a bounded negative result for this load
shape, not a blanket claim that every throughput or topology setting is
fidelity-neutral.

### The decision this supported

The production endpoint migrated to 262k native because static YaRN has a
resolved ~1.03e-02 penalty on the measured 512–32k traffic, while the
window-only comparison is unresolved at the reboot floor. Native 262k covered
the observed default workload; requests that genuinely exceed it should route
to a separately qualified YaRN endpoint.


## §35 - The rope penalty scales with the factor, and Qwen's advice is now quantified

**Receipt: [`yarn-512k-penalty.json`](../receipts/yarn-512k-penalty.json).** Qwen's card tells you to tune
`factor` to the window you actually need — *"if the typical context length for your application is 524,288, it
would be better to set factor as 2.0"* — but publishes no number for what that saves. Same host, same DP8
topology, same 48-prompt probe set (`374d3021…`), same `compare()` code, both arms quiet.

| arm | mean top-20 KLD | top-1 | vs cross-boot floor |
|---|---:|---:|---:|
| 262k native | — (reference) | — | — |
| **512k, YaRN factor 2.0** | **4.2315e-03** | **97.40 %** | **3.59×** |
| 1M, YaRN factor 4.0 (§34) | 1.0265e-02 | 93.75 % | 8.70× |

**Factor 2.0 costs 41 % of what factor 4.0 costs.** Halving the factor cuts the penalty by
roughly 59 %, and top-1 agreement recovers from 93.75 % to 97.40 %. **It is not free** — still
3.59× the floor — but the trade is far better, and the boundary condition holds: factor 1.0 *is*
native rope and costs zero.

Over the one octave measured the penalty goes as **factor^1.28**, slightly super-linear. **Two points one
octave apart do not establish a power law**, and that exponent is a description of the interval, not something to
extrapolate from.

The **worst-in-the-middle bucket shape reproduces** at factor 2.0 (8k most expensive at 6.26e-03, 512 cheapest
at 2.90e-03) — the same ordering as both factor-4.0 measurements, so whatever produces it is factor-independent.

**Practical consequence: set the factor to the window required, not automatically
to the maximum.** Factor 2.0 buys a 512k window at 41 % of factor 4.0's measured
top-20 diagnostic delta. This project did not measure request-length prevalence,
so it makes no claim about what fraction of traffic 512k covers.

### A config defect caught before it could corrupt the result

The first 512k serve script was derived from the 1M one with `sed`, and the pattern **did not match the
backslash-escaped argv** `\"factor\":4.0`. The endpoint booted at **factor 4.0 with a 512k window** — the
wrong experiment, which would have silently reproduced the factor-4.0 number and been reported as factor 2.0.
Caught by grepping the actual argv before any measurement, then fixed with a replacement that **asserts the
substitution landed exactly once and refuses to run otherwise**. The lesson is narrow and reusable: when a
config lives inside an escaped string, verify the *rendered* value, never the edit.
