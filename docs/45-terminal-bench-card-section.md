# Card section for MODEL_CARD-K5K6-hydrated.md — Terminal-Bench 2.1

**Status: METHOD HALF FINAL, RESULTS PENDING PASS 1.** Prepared by `TerminalBench2` for Main to place
on the hydrated card (this agent does not edit cards). Committed *before* any result exists on
purpose: it is the same pre-registration argument applied to the write-up that the two-stage harness
validation applies to the plumbing — the method cannot be quietly reshaped to flatter a number that
has not been measured yet. The
results table and the attribution table are filled from
`receipts/terminal-bench-2.1-{hyd,healing,bf16-attribution}.json` the moment each pass lands; every
other sentence below is already backed by `receipts/terminal-bench-2.1-pins.json` and
`docs/45-terminal-bench.md`. Do not place the results rows until they carry measured numbers.

---

## Terminal-Bench 2.1

**This is a score for an agent + model system, not for a model.** Every number below was produced by
**Terminus-2 2.0.0** driving this checkpoint on a vLLM OpenAI-compatible endpoint. Terminal-Bench
measures whether that *system* made a container's verifier pass, so a failed task does not by itself
implicate the quantisation — it may be the agent's scaffold, the task, a timeout, or the weights.
Separating those is what the third pass exists for. The numbers are **not** comparable to a
leaderboard entry unless that entry used the same agent, harness version, sampling and timeouts;
ours are pinned so the comparison can be checked rather than assumed.

### What was run

| | |
|---|---|
| Harness | harbor 0.21.0 |
| Benchmark | `terminal-bench-2-1@6`, all **89** tasks, run from a **local tree pinned by sha256** (`c13961ac…a48d0ec2`) so a registry-side change cannot alter a pass mid-protocol |
| Agent | `terminus-2` **2.0.0**, with an explicit LiteLLM `model_info` (LiteLLM ships no metadata for `hosted_vllm/` names, so without it the agent miscomputes remaining context and its summariser misfires) |
| Sampling | the agent sends **no** temperature, so each checkpoint's `generation_config` applies — and those files are **byte-identical** between this checkpoint and BF16, which is what makes the attribution pass like-for-like |
| Timeouts | `task.toml` values at multiplier **1.0** (stock) |
| Concurrency | `-n` chosen from measured serving headroom on the 96 GB card, recorded in the pass receipt |
| Serving | vLLM `0.11.2.dev280+gilded.gnosis.v20…r34`, 32,768-token window, fp8 KV, `max_num_seqs 16`, MTP-3, prefix caching, `mamba_cache_mode align`, `--enforce-eager` |

Two deliberate deviations from this card's published 8,192-token recipe, both recorded: the window is
**32,768** with fp8 KV, because Terminus-2 summarises when free context runs low and an 8k window
would thrash on Terminal-Bench's long tool-output transcripts — measuring the summariser rather than
the model; and `--enforce-eager` is kept for the BF16 control arm too, where it is not required, so
that arm differs from this one **in the weights alone**.

### The three passes

| Pass | Tasks | Weights | Question |
|---|---|---|---|
| 1 | all 89 | this checkpoint | What does the shipped quantisation score? |
| 2 | pass-1 failures only | this checkpoint | Which failures were one-off, which persistent? |
| 3 | twice-failed only | BF16 `Qwen/Qwen3.8-27B` | Which persistent failures are the quantisation's fault? |

Pass 3 runs on the same card, same agent, same sampling, same eager mode, same endpoint, **and the same
`-n 16` concurrency** — only the weights change. Each twice-failed task then lands in exactly one of
**three** buckets, not two:

- **`quantization-suspect`** — BF16 resolves it, so the quantisation is implicated, with the BF16
  transcript published as evidence.
- **`capability`** — BF16 runs to completion and still fails it, so the quantisation is exonerated.
- **`inconclusive-timeout`** — **neither arm finished inside the stock budget.** This bucket exists
  because folding it into `capability` would exonerate the quantisation on tasks where *no arm ever
  produced an answer*, which the evidence cannot support. Given that timeouts dominate this run (below),
  this is not a refinement — it is load-bearing, and the three counts are reported separately and never
  summed.

The pass-2 filter matters: without it a single flaky failure would be promoted to a quantisation suspect
and then spend a BF16 run being disproved.

### Results

| Pass | Weights | Tasks | Resolved | Score | Pre-model voids | MTP acceptance | -n |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | K5K6-hydrated | 89 | **31** | **34.83 %** (31/87 = 35.63 % excl. voids) | 2 | 0.5707 | 16 |
| 2 (healing) | K5K6-hydrated | 58 | **13** | 22.41 % — cumulative best-of-2 **44/89 = 49.44 %** | 3 | — | 16 |
| 3 (attribution) | BF16 control | 45 twice-failed | **7** | — (attribution, not a score) | 2 | — | 16 |

Wall clock for pass 1 was **~3.3 h** (18:19Z → ~21:39Z), the tail bounded by the single 12,000-second
task (`build-pov-ray`), which was the only one outstanding from 88/89. Throughput and acceptance are
**deltas between two named `/metrics` snapshots**, both quoted verbatim in the receipt, because the
server's counters carry earlier calibration traffic: over pass 1, +3,330,706 generation tokens and
+3,683,358 draft tokens against +2,102,238 accepted, i.e. **0.5707 acceptance**. Server-side prompt
tokens (+26.9 M) exceed the agent's own reported input tokens (12.8 M) because the server counts every
scheduled prefill including speculative and retried work; both are reported rather than reconciled.

### The attribution, measured — and what it refuses to claim

BF16 ran the 45 twice-failed tasks under otherwise identical conditions and resolved **7** of them.
Filed into the three buckets
([`terminal-bench-2.1-attribution.json`](../receipts/terminal-bench-2.1-attribution.json)):

| bucket | count | what it means |
|---|---:|---|
| **`quantization-suspect`** | **7** | BF16 resolved it; the quantisation did not. Implicated, with the BF16 transcript published. |
| **`capability`** | **1** | BF16 ran to a verdict **without timing out** and still failed. The quantisation is exonerated. |
| **`inconclusive-timeout`** | **35** | **Neither arm ever finished.** Not evidence either way. |
| `harness-void` | 2 | The qemu pair; no arm could start a terminal. Not attributable. |
| *of which provisional* | 3 | `owed-rerun`: their pass-2 attempt was itself a pre-model void. |

**So only 8 of 45 persistent failures are attributable at all**, and the honest headline is the middle
column, not the first: **on 78 % of them this benchmark, at stock timeouts on this hardware, cannot
separate quantisation from capability, because BF16 runs out of clock too.** That is the same finding as
caveat 1 arriving from the other direction — and it is why `inconclusive-timeout` exists as a bucket
rather than being folded into `capability`, which is where a two-way split would have put all 35 and
thereby "exonerated" the quantisation on tasks no arm ever answered.

**One correction worth stating, because it changed the answer.** The first run of the classifier
returned `capability: 36, inconclusive-timeout: 0`, which looked like a clean exoneration. It was a bug
in our tool, not a result: Terminal-Bench **runs the verifier even after the agent times out**, so a
timed-out trial still carries `reward: 0.0`, and a rule that tested "is there a reward" counted 35
timeouts as graded verdicts. The corrected rule requires a verdict reached **without** an
`AgentTimeoutError`. The 7 suspects were unchanged by the fix; the 35 moved from a bucket that flattered
this build to one that says nothing about it.

The seven implicated tasks are `break-filter-js-from-html`, `configure-git-webserver`,
`feal-linear-cryptanalysis`, `headless-terminal`, `mteb-retrieve`, `overfull-hbox` and
`sqlite-db-truncate`; the single `capability` task is `query-optimize`. **A `quantization-suspect`
verdict is not a claim about which tensor** — it says BF16 finished a task this build did not, under the
same agent, sampling, eager mode, endpoint and concurrency.

### Four caveats that belong beside the score, not beneath it

**1. This is a timeout-dominated result.** Of the 58 unresolved tasks in pass 1, **54 ended in
`AgentTimeoutError`** and 2 in `RuntimeError` — leaving **only 2** that ran to completion and answered
wrong. **93 % of the failures ran out of clock, not out of ability.** The mechanism is measured: turns
spend up to **15,577 completion tokens** reasoning, and this system decodes at tens of tokens per second
under concurrency, against minute-scale per-task budgets. Read this score as *an agent+model system on
one card at stock timeouts*, which is what Terminal-Bench measures — not as a capability ceiling.

**2. The concurrency was subsequently measured to be too high, so 31/89 is a floor for this system.**
After this pass we ran a synthetic concurrency ladder on the same hardware and found the per-request
**knee at C4** for this workload's shape, where per-request throughput still holds ≥50 % of
single-stream. This pass ran **`-n 16`, four times past that knee**, at **26 %** of single-stream
per-request throughput — which turns the measured 15,577-token burst into **~10.4 minutes of decode per
turn instead of ~3.6**. `-n 16` was inherited from the harness pins and had never been measured. A
re-run at `-n 4` is the obvious next measurement and **we have not done it**, so this number stands as
published with its serving configuration named, not silently improved.

**3. The score depends on the agent's parser tolerating prose before JSON.** **52 of 89 trials (58 %)**
emit `Extra text detected before JSON object`; Terminus-2 tolerates it and dispatches the keystrokes
anyway. **A stricter parser would have scored this model lower.** Within that subset 19 of 52 resolved
(36.5 %) against 31 of 89 overall (34.8 %) — so emitting the warning does *not* predict failure; an
earlier small-sample reading suggesting it did was noise, and is recorded as such rather than dropped.

**4. Two of the 89 tasks never ran on this host, in either arm, and are counted as failures anyway.**
`qemu-startup` and `qemu-alpine-ssh` die at `RuntimeError: Failed to start tmux session` before a single
prompt is sent — in pass 1, in pass 2, and in the BF16 control alike. They are **environment-unsupported
here, not failed**, so the honest denominator is **87**; the table reports both, and the headline keeps
the conservative 89 rather than the flattering 87. Because both arms void identically, no task can move
between the `quantization-suspect` and `capability` buckets on their account. A third task,
`build-pov-ray`, lost its pass-2 attempt to a server-side `404 The model 'qwen38' does not exist` while
the pass overlapped a model swap — an infrastructure fault that was scored as a task failure because
`NotFoundError` was missing from the transport-retry allowlist. It has since been added; the trial is
flagged `pre_model_void` and **owed one re-run**, which is why the cumulative 44/89 is published as a
**lower bound**. All of this was found by auditing a `n_attempted=54` against `n_trials=58` mismatch in
the pass-2 publication, and is itemised in
[`terminal-bench-2.1-pre-model-voids.json`](../receipts/terminal-bench-2.1-pre-model-voids.json).

### Why the harness can be trusted before the model was ever loaded

The stack was validated in two stages **before any GPU time was spent**, which is the same argument
as pre-registration applied to plumbing:

1. **Infrastructure** — an oracle-agent trial scored **1.0** through the full path (rootless podman →
   docker CLI → compose → task container → verifier → `result.json`).
2. **The agent path** — because the oracle agent needs no model, it proves nothing about Terminus-2.
   So the agent was run against a minimal OpenAI-compatible **test double**, through the real
   endpoint tunnel, driving a real task container: 1 trial, **0 exceptions**, and every link
   exercised (LiteLLM `hosted_vllm/` naming, the explicit `model_info`, response parsing, the
   multi-turn loop, keystrokes reaching a root shell in the container, token accounting, asciinema
   and trajectory capture). **That run found a real defect** — LiteLLM's `model_info` was missing its
   cache-cost fields, which it warned about and which was fixed before the scored passes.

Checkpointing was likewise **proven, not asserted**: resuming a finished job returns in 3.5 s with
every `result.json` byte-identical, and a job `SIGKILL`ed mid-flight keeps its completed trials
untouched, deletes and re-runs only those in flight, and executes the never-started ones.

Task containers **never** touch a GPU: all 89 tasks pin `gpus = 0`, harbor's docker environment
contains no GPU code path at all, and a live task container was inspected showing
`devices=[] devreq=[]`. Co-tenant containers on that host that *do* hold `/dev/nvidia*` are listed by
name in the evidence rather than filtered out of it.

Per-task rows, agent transcripts, verifier output and the receipts are public at
<https://huggingface.co/datasets/malaiwah/qwen38-27b-terminal-bench-2.1>; the method is
`docs/45-terminal-bench.md`.

### One caveat carried from the serving image

The promoted image ships the un-merged **#51812** GDN spec-gate defect (upper bound 0.515/1000 builds
under adversarial traffic). The endpoint also crossed an `ssh -R` tunnel whose measured round-trip is
**581 ms per request** — a time-to-first-token cost, not a per-token one, paid identically by both
arms, so it cannot move a task between the two attribution buckets. Throughput is therefore read from
vLLM's own `/metrics` on the serving host's loopback rather than from agent-side wall clock.
