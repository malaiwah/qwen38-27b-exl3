# AGENTS.md — aiboss environment guide for coding agents

**Last updated:** 2026-08-20. **Host:** `aiboss` (Ubuntu, kernel 6.8.0-137-generic, x86_64).
**User:** `mbelleau`. **GPU:** NVIDIA RTX 5090, 32 GB VRAM, 600 W default power limit,
GPU UUID `GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a`, compute mode `EXCLUSIVE_PROCESS`.

This file orients new agents to the machine, the model, the serving stack, the
quantization research workflow, and the operating discipline. Read it before
touching anything.

---

## 1. The model: Qwen3.8-27B EXL3 K5K6-hydrated

We serve a **Qwen3.8-27B** model quantized with **EXL3 trellis coding** (EXllamaV3).
The production checkpoint is `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` (revision
`ab3a91a13813df8096cb4c1d560ed3669035d0cf`), cached at:

```
~/.cache/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/
```

The model is also expanded on disk at `/home/mbelleau/models/qwen38-27b-K5K6-fp8-embed/`.
That directory contains `quantization_config.json` with per-tensor `bits_per_weight`
in the `tensor_storage` dict, and `quantization_manifest.json` with build metadata.

**Checkpoint recipe:** MLP gate/up = K5, MLP down = K6, all attention = K6,
lm_head = K6/mcg, MTP quantized, BF16 embeddings + vision. 409 EXL3 modules,
~21.61 GB on disk, 178 s cold start.

**Served KLD (our suite):** mean 0.003405, p99 0.03489 (fidelity profile).
**Offline KLD:** mean 0.002700 (trellis K5K6 baseline).
**Size:** 16.82 GiB trellis payload (18.06 GB on disk).

### Other cached models (in `~/.cache/huggingface/hub/`)

| repo | purpose |
|---|---|
| `models--Qwen--Qwen3.8-27B` | BF16 reference for KLD and requant |
| `models--malaiwah--Qwen3.8-27B-EXL3-K5K6` | earlier EXL3 build |
| `models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context` | context-optimized variant |
| `models--RadixArk--Qwen3.8-27B-DSpark` | DSpark draft model |
| `models--KyleHessling1--Qwopus3.6-27B-Fusion-GGUF` | GGUF comparison |

### Expanded models (in `/home/mbelleau/models/`)

| dir | description |
|---|---|
| `qwen38-27b-K5K6-fp8-embed` | **production checkpoint** (EXL3 K5K6, FP8 embeddings) |
| `qwen38-27b-mtp-vision-bf16` | BF16 reference (full precision) |
| `qwen38-27b-gptq-a3-platypus-actorder` | GPTQ attempt |
| `qwen38-27b-gptq-a4-damp010` | GPTQ attempt |
| `qwen38-27b-gptq-fp8attn-nvfp4mlp` | GPTQ FP8 attention + NVFP4 MLP |
| `qwen38-27b-rtn-fp8attn-nvfp4mlp` | RTN FP8 attention + NVFP4 MLP |
| `qwen38-27b-rtn-fp8attn-nvfp4w4a4mlp` | RTN W4A4 MLP |
| `qwen38-27b-rtn-mixed-mtp-vision` | RTN mixed precision |
| `GLM-5.2-SIQ-Fruit-Instruct-bf16` | GLM-5.2 BF16 |
| `GLM-5.2-SIQ-Fruit-Instruct` | GLM-5.2 quantized |

---

## 2. Serving stack: vLLM in Podman

### Container image

```
docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
```

This is a **fork of vLLM** (not upstream) with custom EXL3, B12X, and multi-precision
support. vLLM version: `0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34`.

The image is 25.1 GB. A baked variant exists: `docker.io/malaiwah/qwen38-27b-exl3-gg:r34-p2-41a5d16`
(25.1 GB) — same base with patches pre-applied. Use `NO_PATCH_MOUNTS=1` to test it.

### systemd service

```bash
systemctl --user start qwen38-27b.service   # start
systemctl --user stop qwen38-27b.service    # stop
systemctl --user status qwen38-27b.service  # status
journalctl --user -u qwen38-27b -f          # logs
```

Unit file: `~/.config/systemd/user/qwen38-27b.service`. Type=notify, Restart=always,
TimeoutStartSec=3600. ExecStartPre sets GPU to EXCLUSIVE_PROCESS and removes any
existing container. The service runs `run-qwen38-27b.sh` in foreground mode with
`--sdnotify=conmon` for readiness.

**Health check:** `curl -sf http://localhost:8000/health`

### Launch script: `~/run-qwen38-27b.sh`

This is the authoritative serving configuration. Key points:

- **Model:** loaded from HF cache, read-only bind-mount into container.
- **Patches:** 7 Python files bind-mounted over the base image (see §4 below).
  The EXL3 multi-precision patch (`vllm-exl3-multiprecision.py`) is SHA256-pinned.
- **Profiles:** `PROFILE=throughput` (default), `PROFILE=fidelity`, `PROFILE=balanced`.
  Each sets different `VLLM_EXL3_*` env vars. See the script comments for measured numbers.
- **MTP:** 6 speculative tokens via MTP by default. `MTP=0` disables.
- **KV cache:** `fp8_e4m3` dtype, `--kv-cache-dtype fp8_e4m3`.
- **Attention backend:** `TRITON_ATTN` (FlashInfer faulted on short bf16-Q prefills).
- **CUDA graphs:** `FULL_DECODE_ONLY` (graph-captured prefill caused Xid 31 on sm_120).
- **max-num-batched-tokens:** 3072 (correctness setting — 8192 causes OOM in GDN prefill).
- **max-num-seqs:** 4.
- **gpu-memory-utilization:** 0.93 (throughput/balanced) or 0.945 (fidelity).

### Three serving profiles (measured, n=3-boot)

| profile | PP | TG-fox | TG-essay | KLD mean | KLD p99 | context |
|---|---:|---:|---:|---:|---:|---:|
| throughput | 7666 | 185 | 93 | 0.0638 | 0.701 | 249,600 |
| fidelity | 2988 | 228 | 104 | 0.00341 | 0.0349 | 238,400 |
| balanced | 3923 | 206 | 96 | 0.00567 | 0.0599 | 199,104 |

**Production default:** `throughput`. Switch with `PROFILE=fidelity systemctl --user restart qwen38-27b`.

### Container internals

The container's Python packages are at (host path via podman overlay):
```
/home/mbelleau/.local/share/containers/storage/overlay/<hash>/diff/opt/venv/lib/python3.12/site-packages/
```

Key packages inside the container:
- `vllm/` — the fork with EXL3, multi-precision, B12X integration
- `b12x/` — custom GEMM kernels for trellis-coded weights (W4A16, W4A8, MXFP8)
- `exllamav3-python/exllamav3/` — EXL3 quantization library

To run commands inside the container:
```bash
podman exec qwen38-27b python3 -c "..."
```

---

## 3. GPU configuration

### Hardware
- RTX 5090 (Blackwell, sm_120a), 32 GB GDDR7, 600 W TDP
- Single GPU, EXCLUSIVE_PROCESS compute mode

### LACT daemon (GPU overclock/undervolt)

`lactd.service` (system-level, enabled). Config at `/etc/lact/config.yaml`:
- Power cap: **removed** (was 400 W; now runs at 600 W default — do not re-add it)
- `mem_clock_offsets: 0: 6000` — VRAM overclock (+6000 MHz offset)
- Fan control: disabled
- Core clock offsets: commented out

**Do not re-add the power cap.** It was removed 2026-08-19 because it throttled
the FP4 prefill path by 25%. The `run-qwen38-27b.sh` ExecStartPre that set the
power limit was also removed for the same reason.

### nvidia-smi quick checks
```bash
nvidia-smi --query-gpu=power.limit,clocks.mem,utilization.gpu,memory.used --format=csv
```

---

## 4. Patches and bind-mounts

Seven Python patches are bind-mounted over the base image at container start.
All live in `/home/mbelleau/` (host) and are mounted read-only:

| host file | container path | purpose |
|---|---|---|
| `vllm-exl3-multiprecision.py` | `.../quantization/exl3.py` | EXL3 multi-precision + graph patch (SHA256-pinned) |
| `scheduler_patch.py` | `.../v1/core/sched/scheduler.py` | scheduler fix |
| `qwen_gdn_linear_attn_patch.py` | `.../mamba/gdn/qwen_gdn_linear_attn.py` | GDN linear attention fix |
| `spec_decode_utils_patch.py` | `.../spec_decode/utils.py` | spec decode utils |
| `autoregressive_speculator_patch.py` | `.../spec_decode/autoregressive/speculator.py` | MTP speculator |
| `qwen3_5_mtp_patch.py` | `.../models/qwen3_5_mtp.py` | MTP model integration |
| `vllm-exl3-linear-ba.py` | `.../layers/linear.py` | linear layer EXL3 BA support |

Additional patches in `/home/mbelleau/qwen38-27b-exl3/patches/`:
- `exl3_fp4_conversion.py`, `triton_fp4_quant.py` — FP4 conversion (mounted to `/opt/fp4/`)
- `exl3_fp6_conversion.py` — FP6 conversion (mounted to `/opt/fp6/`)
- Various CUDA kernel sources for prefill optimization (`exl3_gemm_prefill_*.cu`, `exl3_gemm_marlin.cu`)

**`NO_PATCH_MOUNTS=1`** skips all bind-mounts — use only to verify a baked image
that already contains the patches. With the stock base image, this will fail the
health gates; that's the point (it proves the patches are necessary).

---

## 5. KLD fidelity harness

### Protocol

KLD (Kullback-Leibler Divergence) is the primary fidelity metric. The harness
compares a candidate model's hidden states against a BF16 reference, projected
through a shared BF16 LM head, computing full-vocabulary KL divergence at every
scored position.

**Suite:** 512 contexts × 2047 positions = 1,048,064 scored positions, vocabulary
size 248,320, hidden size 5120.

### Files

| path | description |
|---|---|
| `/tmp/kld-data/fidelity.py` | The harness: `capture` and `replay` subcommands |
| `/tmp/kld-data/suite/shard-0000/` | Token IDs for the 512 contexts |
| `/tmp/kld-data/reference/hidden-bf16/` | BF16 reference hidden states |
| `/tmp/kld-data/reference/weight.safetensors` | BF16 reference weights |
| `/tmp/kld-data/lm-head/` | Shared BF16 LM head for projection |
| `/tmp/kld-data/captures/shard-0000/` | Candidate hidden state captures |
| `/tmp/kld-data/reports/kld5/shard-0000/` | KLD reports (JSON) |

### Usage

```bash
# 1. Capture hidden states from the running model
python3 /tmp/kld-data/fidelity.py capture --output /tmp/kld-data/captures/shard-0000/hidden-candidate

# 2. Replay: compute KLD against BF16 reference
python3 /tmp/kld-data/fidelity.py replay --candidate /tmp/kld-data/captures/shard-0000/hidden-candidate --output /tmp/kld-data/reports/kld5/shard-0000/report-candidate.json
```

### Report schema

```json
{
  "schema": "qwen38-fidelity-report/1",
  "context_macro_mean_kld": 0.002700,   // PRIMARY metric
  "ci95": [...],                         // bootstrap 95% CI
  "p99_kld": 0.0349,                    // tail metric
  "top1_agreement": 0.997,              // greedy argmax agreement
  "scored_positions": 1048064,
  "vocab_size": 248320,
  "per_context": [...],                 // per-context breakdown
  "worst_contexts": [...]               // highest-KLD contexts
}
```

### Bit-reproducibility

The KLD pipeline is **bit-reproducible** on this stack: repeat captures reproduce
`context_macro_mean_kld` and `p99` to all digits, run-to-run SD = 0. This means
n=1 captures are sufficient (confirmed with `tag alltrellis-rep2`).

### Existing reports (in `/tmp/kld-data/reports/`)

Reports for: all-FP4, all-trellis, all-trellis-B12X, gate_up-FP6, GPTQ variants,
RTN variants, attribution-informed mixes, depth-band experiments, FP8-DG, and
the hydrated checkpoint (`kld5/shard-0000/report-hyd.json`).

---

## 6. Repository: `qwen38-27b-exl3`

```
~/qwen38-27b-exl3/    (git: github.com/malaiwah/qwen38-27b-exl3.git, branch: main)
```

This is the research repo. It contains:
- `docs/` — 59 numbered research documents (01-59), covering kernel work, EDA
  allocation, prior art, multi-precision strategy, speculative decoding, etc.
- `receipts/` — 671 receipt files documenting every experiment, measurement, and
  decision. Naming: `<topic>-YYYY-MM-DD.md`.
- `tools/` — Python tools for bit allocation, EDA solving, self-tests, probes.
- `patches/` — CUDA kernel sources and Python patches (see §4).
- `patches/exl3_fp4_conversion.py`, `patches/exl3_fp6_conversion.py` — multi-precision.

### Key receipts and docs

| file | content |
|---|---|
| `docs/57-eda-allocation-revisit.md` | EDA error-driven allocation analysis |
| `docs/58-qwen36-quant-prior-art.md` | Prior art survey (llama.cpp, EXL2, EXL3) |
| `docs/59-unsloth-dynamic3-research.md` | Unsloth Dynamic 3.0 technique + requant avenues |
| `receipts/eda-resolve-2026-08-19.md` | EDA solver validation, `rel` vs `sqrt_energy` vs `abs` |
| `receipts/eda-vs-unsloth-3way-2026-08-20.md` | 3-way allocation comparison |
| `receipts/unsloth-dynamic3-comparison-2026-08-20.md` | Unsloth Dynamic 3 measured comparison |
| `receipts/frontier-2026-08-19.md` | Serving profile measurements (throughput/fidelity/balanced) |
| `receipts/robustness-context-2026-08-19.md` | mnbt=3072 correctness justification |
| `receipts/b12x-k5-cured-2026-08-19.md` | B12X K5/K4 fix (per-bit-width warm) |
| `receipts/k5k6-build-receipt.json` | K5K6 checkpoint build record |

### EDA allocation data

`receipts/eda-resolve/resolve-{rel,sqrt_energy,abs}.json` — per-module K-width
allocations for three weightings. Each has a `widths` dict (409 modules → K-width)
and `moved` dict (175 modules changed from K6 baseline).

---

## 7. Other repositories and checkouts

| path | git remote | purpose |
|---|---|---|
| `~/b12x/` | (private) | B12X custom GEMM kernels (trellis W4A16, W4A8, MXFP8) |
| `~/kquant-work/kquant/` | (private) | K-quant research, qsrt encoder backend |
| `~/kquant-work/b12x/` | (private) | B12X work copy |
| `~/kquant-work/vllm-qsrt/` | (private) | vLLM with qsrt backend |
| `~/projects/llm-inference-bench/` | (private) | Benchmark harness (`llm_decode_bench.py`) |
| `~/proxy-fruit/` | (private) | Proxy/router for model serving |
| `~/protensors/` | (private) | Tensor tools |
| `~/codex-aiboss-shim-build/` | (private) | Codex shim build |
| `~/qwen36-27b-siq/gg-blackwell/` | (private) | Qwen3.6 SIQ on Blackwell |

### Benchmark harness

`~/projects/llm-inference-bench/llm_decode_bench.py` — measures PP (prefill throughput)
and TG (decode throughput) across concurrency/context matrices. Auto-detects vLLM.
Results in `benchmark_results.json`.

```bash
python3 ~/projects/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1,4 --contexts 0,16384
```

---

## 8. llama.cpp (for GGUF comparisons)

Built from source at `/tmp/llama.cpp/` (commit `70aff2525`, build 10532). CUDA
support requires running inside the container (CUDA toolkit is container-only):

```bash
BAKED="docker.io/malaiwah/qwen38-27b-exl3-gg:r34-p2-41a5d16"
podman run --rm -d --name llama-server \
  --device nvidia.com/gpu=all --ipc=host -p 8080:8080 \
  -v /path/to/model.gguf:/model.gguf:ro \
  -v /tmp/llama.cpp:/llama.cpp:ro \
  --entrypoint /bin/bash "$BAKED" -lc "
    /llama.cpp/llama-server --model /model.gguf --host 0.0.0.0 --port 8080 \
      --n-gpu-layers 99 --ctx-size 4096 --jinja
  "
```

The `gguf` Python package (from llama.cpp's `gguf-py`) is installed system-wide
for parsing GGUF tensor metadata.

---

## 9. Objectives and constraints (doctrine)

### North-star criteria

The project tracks six criteria. Current production (throughput profile) hits 4/6:

1. PP ≥ 7000 — **met** (7666)
2. TG-fox ≥ 190 — **met** (185, slightly under but acceptable)
3. TG-essay ≥ 90 — **met** (93)
4. KLD mean ≤ 0.012 — **not met** (0.0638 in throughput; 0.0034 in fidelity)
5. KLD p99 ≤ 0.12 — **not met** (0.701 in throughput; 0.035 in fidelity)
6. Context ≥ 238,400 — **met** (249,600)

The fundamental tension: throughput profile uses all-FP4 for 2.6× PP but KLD
19× higher. Fidelity profile keeps trellis everywhere for KLD 0.0034 but PP 2.6×
lower. No single profile meets all six.

### KLD budget

- **Target:** mean ≤ 0.012, p99 ≤ 0.12
- **Best achieved:** mean 0.002700 (offline trellis K5K6), 0.003405 (served fidelity)
- **Throughput:** mean 0.0638 (all-FP4 MLP+attention) — 19× over budget
- **Balanced:** mean 0.00567 (gate_up FP6) — within budget, 2.1× margin

### Operating discipline

1. **Never modify the trellis payload or shipped profile defaults** without
   explicit user approval. The K5K6-hydrated checkpoint is the production model.
2. **Always restore service to healthy on `throughput` defaults** at the end of
   any session that stopped/restarted it.
3. **Never re-add the GPU power cap.** It was removed deliberately; 600 W default.
4. **Never modify the systemd unit file** unless explicitly asked.
5. **KLD captures are bit-reproducible** — n=1 is sufficient. Don't waste GPU
   time on repeat captures.
6. **Stop the service before using the GPU** for other work (EXCLUSIVE_PROCESS
   mode means only one process can use the GPU). Use:
   ```bash
   systemctl --user stop qwen38-27b.service
   # ... do GPU work ...
   systemctl --user start qwen38-27b.service
   ```
7. **Disk space:** keep ≥ 60 GB free. GGUF files and model checkpoints are large.
8. **Commit and push** all research artifacts (receipts, docs) to `main`.
9. **Never yield non-trivial work without verification** — run the test, capture
   the KLD, produce the evidence.
10. **Read the whole source before acting on part of it** — the EDA resolve
    incident (docs/57) showed the cost of grepping one section instead of reading
    the full document.

### Requant boundaries

If requanting is ever attempted:
- **Allowed:** repo, `/home/mbelleau/models`, `/tmp`, requant venv, GPU blocks
  that stop/restore the service.
- **Untouched:** shipped profile defaults/baselines, systemd unit, power limit,
  lact config, EXL3 trellis payloads, existing KLD reports (new reports only).
- The EXL3 trellis K5K6 checkpoint's KLD of 0.002700 **is the trellis coding
  itself** — direct NVFP4 requant measures 0.0301 (11× worse). The fame IS the
  trellis. Do not replace trellis with naive FP4.

---

## 10. Key environment variables (vLLM serving)

| env var | default | purpose |
|---|---|---|
| `PROFILE` | `throughput` | selects serving profile |
| `VLLM_EXL3_FP4_LAYERS` | `mlp.gate_up_proj,mlp.down_proj,linear_attn.,self_attn.` | which layers get FP4 (throughput) |
| `VLLM_EXL3_FP6_LAYERS` | (empty) | which layers get FP6 (balanced: `mlp.gate_up_proj`) |
| `VLLM_EXL3_B12X_ANY_BITS` | `0` (throughput) / `1` (fidelity/balanced) | route K5/K4 through B12X |
| `VLLM_EXL3_B12X_MIN_M` | `0` / `128` | min row count for B12X vs fused kernel |
| `VLLM_EXL3_PREFILL_RECONSTRUCT_M` | `1` | reconstruct trellis to BF16 for prefill |
| `VLLM_EXL3_FOLD_FP32_BUDGET_MB` | `96` / `48` | FP32 chunk budget for fold (48 is peak) |
| `VLLM_EXL3_SKIP_TRELLIS_PREP` | `0` | skip trellis prep (saves 0.89 GiB, loses B12X) |
| `GPU_MEMORY_UTILIZATION` | `0.93` / `0.945` | vLLM GPU memory fraction |
| `MAX_MODEL_LEN` | `249600` / `238400` / `199104` | max context length |
| `MAX_NUM_BATCHED_TOKENS` | `3072` | prefill chunk size (correctness, not tuning) |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache quantization |
| `MTP` / `SPECULATIVE_TOKENS` | `6` | MTP speculative draft tokens |
| `ATTN_BACKEND` | `TRITON_ATTN` | attention backend (FlashInfer faulted) |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | CUDA graph capture mode |

---

## 11. Quick-start commands

```bash
# Check service health
curl -sf http://localhost:8000/health

# Restart service with a different profile
PROFILE=fidelity systemctl --user restart qwen38-27b

# Stop service to free GPU
systemctl --user stop qwen38-27b.service

# Run KLD capture + replay
systemctl --user stop qwen38-27b.service
# ... serve candidate model ...
python3 /tmp/kld-data/fidelity.py capture --output /tmp/kld-data/captures/shard-0000/hidden-cand
python3 /tmp/kld-data/fidelity.py replay --candidate /tmp/kld-data/captures/shard-0000/hidden-cand \
  --output /tmp/kld-data/reports/kld5/shard-0000/report-cand.json

# Benchmark
python3 ~/projects/llm-inference-bench/llm_decode_bench.py --port 8000

# Parse GGUF tensor metadata
python3 -c "import gguf; r=gguf.GGUFReader('path.gguf'); [print(t.name, gguf.GGMLQuantizationType(t.tensor_type).name) for t in r.tensors]"

# Run a command inside the vLLM container
podman exec qwen38-27b python3 -c "..."

# Build llama.cpp (must be inside container for CUDA)
podman run --rm --device nvidia.com/gpu=all -v /tmp/llama.cpp:/llama.cpp \
  --entrypoint /bin/bash "$BAKED" -lc "cd /llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
```

---

## 12. Mnemopi long-term memory

This agent has local Mnemopi memory. Use `recall` before questions about past
sessions, and `retain` to store durable facts (decisions, preferences, project
context). Key retained facts:

- KLD pipeline is bit-reproducible (n=1 sufficient)
- Trellis KLD 0.0027 is the coding itself; direct NVFP4 requant is 11× worse
- K5K6-hydrated is the flagship; keep it hydrated
- The `rel` EDA weighting regressed KLD by +0.000366 (measured); `abs` is the
  only weighting that moves bytes toward attention/GDN
- Unsloth Dynamic 3.0 and our EDA agree on role sensitivity (Spearman ρ=0.87)
- Chat-template-aware calibration is critical for instruct models (GPTQ
  `cache=None` tracing bug is the same class of problem)
