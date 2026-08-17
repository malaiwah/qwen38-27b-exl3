# Repository Guidelines

## Project Overview

This repository is a research and model-artifact companion for Qwen3.8-27B mixed-precision EXL3 checkpoints. It records conversion recipes, vLLM runtime patches, serving qualification, fidelity experiments, model cards, and receipt-backed evidence. It is **not** an installable Python package or a conventional application: model shards, the serving environment, and most GPU dependencies are external.

Use `README.md` for the current overview, `PROGRESS.md` for chronology, and the current receipts/model cards for published claims. Numbered docs often preserve superseded experiments; do not treat an older iteration as current without checking its status and receipt.

## Architecture & Data Flow

1. **Inputs:** a local, immutable Hugging Face model snapshot; calibration/held-out corpus data; and the pinned Gilded Gnosis runtime image.
2. **Checkpoint build:** exllamav3 conversion produces packed EXL3 tensors; `tools/splice_bf16_attn.py` restores selected BF16 attention tensors; index/config generation and `tools/finalize_checkpoint.py` validate logical tensor names/shapes and emit manifests and checksums. The published recipe keeps MLP projections in mixed EXL3 K5/K6 roles while attention, vision, embeddings, and norms remain BF16 where specified by the manifest.
3. **Serving:** `tools/ggrun.sh` runs the external vLLM image through static `proot`, binding host model/work/cache directories. Qwen EXL3 must use direct `vllm serve`; the image's family launchers do not accept Qwen. Patched modules implement Qwen/MTP construction and EXL3 dispatch; online K6 encoding and graph/prefill patches are explicit runtime variants, not package-local imports.
4. **Evaluation:** `tools/fetch_corpus_v5.py` and `tools/suite3.py` freeze a held-out suite; `tools/fidelity.py` captures final-RMSNorm hidden states, replays them through one shared BF16 LM head, and computes full-vocabulary KL/JS/top-1 metrics. `tools/kld_ladder.sh` runs bounded shards, while `tools/kld_aggregate.py` and paired/qualification tools produce validated JSON receipts.

The fidelity protocol is teacher-forced, text-only distribution comparison. It is not a generation, long-context, vision, or throughput result unless a separate receipt says so.

## Key Directories

- `tools/` — primary Python, Bash, and C++ CLIs: conversion/finalization, runtime patches, fidelity, serving probes, benchmark runners, and receipt builders.
- `docs/` — numbered design notes, runtime contracts, protocols, experiment reports, and open-work plans. `docs/42-kld-method.md` is the KLD method of record.
- `receipts/` — JSON manifests and measurement evidence. Treat schema, digest, runtime, suite, and model identities as authoritative; do not hand-edit published receipts.
- `docker/` — pinned runtime Dockerfiles, rootless build/verification workflow, and smoke client.
- `patches/` — standalone upstream patches used for provenance or image construction.
- `upstream/` — issue/PR, reproduction, and review artifacts; it is not a vendored vLLM source tree.
- `assets/` — generated figures and presentation assets.
- Root `MODEL_CARD*.md` and `DATASET_CARD*.md` — publication and replay instructions.

There is no `src/`, package manifest, root `Makefile`, or checked-in model payload directory. Do not create a parallel application structure for a tooling-only change.

## Development Commands

Start with the script's own help; most CLIs use explicit `argparse` subcommands.

### Host setup and pinned runtime

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt   # host-only dependencies

# Baseline through the extracted, pinned Gilded Gnosis image
GG_MODELS=/var/tmp/models tools/ggrun.sh vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen38-bf16 --max-model-len 8192 \
  --gpu-memory-utilization 0.92 --max-num-seqs 4 --port 8011
```

GPU work belongs inside the pinned image. `tools/pull_rootfs.py` plus `tools/ggrun.sh` is the documented no-container-runtime path; it uses `proot`, not a host Python installation of vLLM. The image is pinned by digest in `docs/06-baseline-validation.md` and `receipts/production-image.json`.

### Build and publish a checkpoint

Read `docs/04-exllamav3-toolchain.md` before converting. The normal sequence is conversion, BF16-attention splice, safetensors index/config generation, then finalization:

```bash
python convert.py -i <bf16-dir> -o <quant-dir> -w <work-dir> ...
python util/add_safetensors_index.py -m <quant-dir>
python util/add_quant_config.py -m <quant-dir>
python3 tools/finalize_checkpoint.py -m <quant-dir> --upstream <upstream-dir> \
  --source-repo <repo> --source-revision <revision> --converter <version>
```

The converter and `util/*` commands run in the external exllamav3 environment. Finalization is a publication gate: it must pass tensor/metadata checks and writes separate immutable payload and mutable-document checksums.

### Runtime image

```bash
docker/build-image.sh plan k4 context
VARIANT=release docker/build-image.sh build
VARIANT=release docker/build-image.sh verify
VARIANT=release docker/build-image.sh smoke k4 k5k6 hydrated context
VARIANT=release docker/build-image.sh receipt
```

These stages require rootless `podman`, `python3`, and `sha256sum`; `smoke` requires a usable GPU. `VARIANT=apc` adds the scheduler patch for prefix caching. `VARIANT=convert` is conversion-only and is not a serving-qualified image. `sbom` is available between `build` and `receipt`.

### Fidelity and benchmark workflows

```bash
python3 tools/fidelity.py --help
python3 tools/fidelity.py suite --model <local-snapshot> --out <suite> --contexts 128
python3 tools/fidelity.py capture --model <local-snapshot> --suite <suite> --out <capture>
python3 tools/fidelity.py replay --reference <ref> --candidate <capture> \
  --head <lm-head.safetensors> --suite <suite> --out <report.json>
python3 tools/fidelity.py paired --a <report-a.json> --b <report-b.json> --out <paired.json>

tools/kld_ladder.sh --dry-run 0-9   # plan and validate paths; no GPU work
tools/kld_ladder.sh 0                # one shard; expensive GPU workflow
python3 tools/kld_aggregate.py --help
```

A CPU-only aggregate can replay published reports when the suite and identities match. `tools/run_wikitext_kld.sh` has explicit `plan`, `fetch`, `preflight`, `tokencheck`, `run`, and `release` phases. Public capability, retention, vision, tool-call, decode-parity, and long-context checks are separate workflows; inspect their JSON output rather than treating them as unit tests.

## Code Conventions & Common Patterns

- Python CLIs use Python 3 type hints, `from __future__ import annotations`, `pathlib`, `argparse`, deterministic traversal/seeds, and composable subcommands. Most state is explicit files, manifests, environment variables, or server endpoints; there is no dependency-injection or application state framework.
- Bash tools normally use `set -euo pipefail`, explicit positional/options parsing, bounded phases, and artifact checks. Some vLLM processes report a teardown failure after writing valid output; the artifact and receipt are the success criterion, not exit status alone.
- Fail closed. Reject mutable Hub IDs, missing `config.json`, unknown files, unsupported tensor schemas, stale suite/head/runtime identities, digest mismatches, incomplete reports, and insufficient scratch space. Use `--dry-run` before GPU or destructive phases.
- JSON receipts are versioned and written atomically (`.tmp` then replace). Preserve command lines, source/runtime revisions, SHA256s, and schema names. Do not overwrite a receipt to make a result look current; create a new receipt or document supersession.
- Packed physical tensors are not the same as logical model tensors. Use `quantization_manifest.json`, build receipts, and finalizer checks when changing checkpoint layout.
- Runtime settings are intentionally explicit. For dense EXL3, the quantization config must exclude non-overlay modules with the established list:

  ```json
  {"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}
  ```

  Do not shorten or “correct” this list without rerunning the relevant startup and vision checks. Use `--quantization exl3`; graph decode additionally requires `VLLM_EXL3_GRAPH_DECODE=1` and the documented patched modules.
- Prefer new receipts and narrow tool changes over broad refactors. Keep current and superseded protocol paths clearly named; never mix reports from different suite versions, scoring windows, model revisions, engines, or shared heads.

## Important Files

- `README.md` — current project purpose, evidence table, tool map, status, and open work.
- `PROGRESS.md` — chronological experiment and operational record.
- `requirements.txt` — host-side dependencies only.
- `docs/03-gg-runtime-contract.md` — EXL3 metadata, overlay eligibility, ignore rules, and serving constraints.
- `docs/04-exllamav3-toolchain.md` — conversion and checkpoint assembly sequence.
- `docs/06-baseline-validation.md` — pinned image, rootfs/proot setup, and baseline serving commands.
- `docs/42-kld-method.md` — current fidelity metric, capture/replay protocol, and reproducibility commands.
- `tools/ggrun.sh` / `tools/pull_rootfs.py` — guest runtime and OCI rootfs preparation.
- `tools/splice_bf16_attn.py` / `tools/finalize_checkpoint.py` — checkpoint assembly and publication validation.
- `tools/fidelity.py` / `tools/suite3.py` — suite, hidden-state capture/replay, and scoring.
- `tools/kld_ladder.sh` / `tools/kld_aggregate.py` — sharded evaluation and cumulative receipt generation.
- `tools/vllm-exl3-prefill-dispatch.py`, `tools/vllm-qwen3_5-embed-quant-config.py`, `tools/vllm-qwen3_5_mtp-embed-quant-config.py` — source-mounted runtime patch modules; image destinations and SHA256s are recorded by `docker/build-image.sh`.
- `docker/build-image.sh` / `docker/smoke_client.py` — image stages and exact text+image smoke requests.
- `receipts/production-image.json`, `receipts/*-build-receipt.json`, and `receipts/*-quantization-manifest.json` — runtime and checkpoint provenance.

## Runtime/Tooling Preferences

- Use host Python only for lightweight fetch, manifest, plotting, and receipt tasks. `requirements.txt` contains only `huggingface_hub`, `hf_transfer`, and `matplotlib`; the pinned image supplies Python 3.12, CUDA 13.2, Torch 2.12, vLLM, exllamav3, and GPU extensions. Do not add serving dependencies to `requirements.txt`.
- `tools/ggrun.sh` defaults to `GG_ROOTFS=/var/tmp/gg-rootfs`, `GG_MODELS=/var/tmp/models`, `GG_WORK=/var/tmp/work`, and `GG_CACHE=/var/tmp/gg-cache`; guest paths are `/models`, `/work`, and `/cache`. It forces offline Hugging Face/Transformers operation and binds host NVIDIA driver libraries.
- Prefer pinned commits, image digests, local snapshots, and SHA256 verification. `tools/fidelity.py` intentionally refuses mutable Hub model IDs; download/review an immutable local revision first.
- Invoke `vllm serve` directly for Qwen EXL3. Stock upstream vLLM/SGLang/TensorRT/llama.cpp/stock exllamav3 compatibility is not assumed by this repository's published recipes.
- The release image and the source-mounted patched recipe are different evidence units. Verify the image digest and patch hashes before claiming a result. `receipts/production-image.json` is authoritative over stale in-image labels or older card text.
- Keep LMCache disabled for published workflows: the measured campaign found connector reuse corruption even with the scheduler patch. Do not enable it without a new qualification protocol and receipt. Do not use undocumented KV-memory overrides such as `--kv-cache-memory-bytes`.
- Serving endpoints should be loopback-bound and placed behind authenticated TLS when exposed beyond the host. Preserve the official chat template/tokenizer behavior; use the documented Qwen tool parser and client content/reasoning-history contract.

## Testing & QA

There is no checked-in pytest suite, CI workflow, coverage configuration, `Makefile`, or package test runner. QA is bespoke and receipt-backed.

- **Static/CPU checks:** use `--help`, manifest/index validation, `tools/fidelity.py paired`, and `tools/kld_aggregate.py aggregate` with published suite/head identities. `fidelity.py` lazily imports Torch, so help and pure-JSON paired work do not require Torch.
- **Runtime/image checks:** run `docker/build-image.sh verify` before smoke; use `docker/smoke_client.py` for exact text+image requests and inspect `podman diff`, identity, and readiness evidence.
- **Fidelity:** use `tools/kld_ladder.sh --dry-run <shards>` before a GPU run, then verify every shard before aggregation. Never reuse a report from another suite, model, head, runtime, or scoring window.
- **Protocol probes:** `tools/run_wikitext_kld.sh` validates the external KLD protocol; `tools/decode_parity.py`, `tools/vision_eval.py`, `tools/tool_calls_e2e.py`, `tools/longctx.py`, `tools/longmm.py`, and `tools/task_retention.py` cover separate serving axes. Many probes record failures in JSON while returning zero, so inspect the receipt and require the tool's documented gate flags.
- **Acceptance evidence:** check `identity_verified`, SHA256s, schema versions, gate fields, and exact pass/no-regression counts before reporting success. For GPU or long-context work, record GPU model, memory utilization, cache settings, image digest, and command line.
- **Historical paths:** `run_v3.sh`, `v4_qualify.sh`, old KLD docs, and hardcoded retention wrappers may be useful for provenance but are not general-purpose regression suites. Prefer the current README, docs/42, and current receipts.
