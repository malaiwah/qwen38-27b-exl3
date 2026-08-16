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

Pass 3 runs on the same card, same agent, same sampling, same eager mode, same endpoint — only the
weights change. Each twice-failed task therefore lands in exactly one bucket: **`capability`** (BF16
fails it too, so the quantisation is exonerated for that task) or **`quantization-suspect`** (BF16
passes it, so the quantisation is implicated, with the BF16 transcript published as the evidence).
The pass-2 filter matters: without it a single flaky failure would be promoted to a quantisation
suspect and then spend a BF16 run being disproved.

<!-- RESULTS — fill from the pass receipts; do not place until measured.
| Pass | Weights | Tasks attempted | Resolved | Score | MTP acceptance | output tok/s | -n |
|---|---|---|---|---|---|---|---|
| 1 | K5K6-hydrated | | | | | | |
| 2 | K5K6-hydrated | | | | | | |
| 3 | BF16 control | | | | | | |

| Twice-failed task | BF16 result | Classification |
|---|---|---|
-->

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
