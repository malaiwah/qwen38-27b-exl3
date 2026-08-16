---
license: apache-2.0
task_categories:
- text-generation
tags:
- terminal-bench
- agentic-evaluation
- quantization
- exl3
- vllm
pretty_name: Qwen3.8-27B EXL3 K5K6-hydrated on Terminal-Bench 2.1
---

# Qwen3.8-27B EXL3 K5K6-hydrated on Terminal-Bench 2.1

Per-task results, agent transcripts and verifier output for a three-pass Terminal-Bench 2.1
evaluation of `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`, with a BF16 control pass that attributes
each persistent failure either to capability or to the quantisation.

Method write-up: `docs/45-terminal-bench.md` in the project repo. Every knob is frozen in
`receipts/terminal-bench-2.1-pins.json`, published at the root of this dataset.

## Read this first: what these numbers are

Terminal-Bench scores an **agent + model system**, not a model. Everything here was produced by
**Terminus-2 2.0.0** driving a vLLM OpenAI-compatible endpoint. A failed task does not by itself
implicate the quantisation — it may be the agent scaffold, the task, a timeout, or the weights.
Separating those is what pass 3 is for.

These numbers are **not** comparable to a leaderboard entry unless that entry used the same agent,
harness version, sampling and timeouts. Ours are pinned so the comparison can be checked instead of
assumed.

## The three passes

| Pass | Tasks | Weights | Question |
|---|---|---|---|
| `passes/pass1-hydrated` | all 89 | `K5K6-hydrated` | What does the shipped quantisation score? |
| `passes/pass2-healing` | pass-1 failures only | `K5K6-hydrated` | Which failures were one-off vs persistent? |
| `passes/pass3-bf16-attribution` | twice-failed only | BF16 `Qwen/Qwen3.8-27B` | Which persistent failures are the quantisation's fault? |

Pass 3 runs on the same card, same agent, same sampling, same eager mode, same tunnel. Only the
weights change, so each twice-failed task lands in exactly one bucket:

- **`capability`** — BF16 fails too; the quantisation is exonerated for that task.
- **`quantization-suspect`** — BF16 passes; the quantisation is implicated, and the BF16 transcript
  in `passes/pass3-bf16-attribution/trials/` is the evidence.

The BF16 arm is served under the model name `qwen38-bf16` (the quantised arm is `qwen38`), so every
row says which weights answered it.

## Layout

```
receipts/…                      the frozen pins and the per-pass receipts
harness-validation/oracle-smoke/ oracle-agent trial through the full stack, before any GPU time
passes/<label>/
  results.jsonl                 one row per trial
  summary.json                  aggregate for the pass
  job.log                       harness log
  trials/<trial>/
    result.json  config.json  trial.log
    agent/…                     agent transcript
    verifier/…                  verifier stdout, CTRF report, reward
```

### `results.jsonl` fields

`task`, `task_name`, `trial_name`, `task_checksum`, `status`, `reward`, `resolved`,
`exception_type`, `agent`, `agent_version`, `model`, `n_input_tokens`, `n_cache_tokens`,
`n_output_tokens`, `override_gpus`, `timeout_multiplier`, `wall_s`, `env_setup_s`, `agent_exec_s`,
`verifier_s`, `started_at`, `finished_at`.

Field names come verbatim from harbor 0.21.0's `result.json`.

## Setup, in one screen

- **harness** — harbor 0.21.0; dataset `terminal-bench-2-1@6` run from a *local pinned tree*
  (`sha256 c13961ac…a48d0ec2`) so a registry change cannot alter a pass mid-protocol; 89 tasks
- **agent** — `terminus-2` 2.0.0 with an explicit `model_info` (LiteLLM ships no metadata for
  `hosted_vllm/` names, so without it the agent miscomputes remaining context)
- **serving** — vLLM `0.11.2.dev280+gilded.gnosis.v20…r34`, torch 2.12.0+cu132, on one
  RTX PRO 6000 Blackwell (96 GB): 32,768-token window, fp8 KV, `max_num_seqs 16`, MTP-3, prefix
  caching, `mamba_cache_mode align`, `--enforce-eager`
- **sampling** — the agent sends no temperature, so each checkpoint's `generation_config` applies;
  those files are **byte-identical** between the two checkpoints, which is what makes pass 3
  like-for-like
- **timeouts** — `task.toml` values at multiplier 1.0
- **topology** — the model serves on a rental box that cannot run containers at all (`unshare -U`
  and `-m` are seccomp-`EPERM`, no docker/podman, no systemd); the 89 task containers run on a
  second machine under rootless podman 4.9.3 via a static docker CLI + compose with `DOCKER_HOST`
  pointed at the podman socket; the endpoint crosses an `ssh -R` reverse tunnel whose round-trip
  is measured and recorded; the remote port is owned by our own proxy, not by sshd
- **CPU-only** — all 89 tasks pin `gpus = 0`, harbor's docker environment has no GPU code path at
  all, and TB containers plus `nvidia-smi` process lists are inspected during every pass

## Status

No `configs:` splits are declared in this card yet: declaring a split whose file does
not exist makes the dataset viewer error, so each pass's `results.jsonl` is registered as a split
when that pass actually lands.

Passes are published **as they complete**, not at the end. A pass absent from this dataset has not
been run yet.
