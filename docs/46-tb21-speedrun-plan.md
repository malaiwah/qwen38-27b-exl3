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
- **`reasoning_effort`**: the chat template exposes `low|medium|high|xhigh`, and lowering it would cut
  the reasoning bursts that cause **93 % of our TB failures** - making it arguably the largest
  TB-score lever available. **Owner's ruling: the speed run runs at DEFAULT** so results stay comparable
  with passes 1-3; reasoning-effort behaviour is to be characterised separately for the card. A probe at
  4096 max_tokens was inconclusive (truncated mid-reasoning; `high` returned null, matching the known
  "upstream raises on high" note) - any real characterisation needs ≥32k output budget and an idle card.
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
