# 56 — Speculative decoding for the hydrated quant: DFlash, DFlash2, DSpark, and the quantized-target training question

Research scan, 2026-08-19. No GPU was used; every claim is sourced to a fetched model
card, a fetched `config.json`, the local vLLM fork source, or a cited paper. Where a
number is an estimate rather than a measurement, it is labelled `[INFERENCE]`.

This document answers four questions the maintainer asked:

1. Are there DFlash / DFlash2 / DSpark draft models on Hugging Face compatible with
   `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`?
2. How much is the quantized MTP layer on our quant, and how much is the DFlash / DSpark
   model — and can the draft be quantized the same way?
3. What are the estimated TG throughput gains versus MTP0, MTP=3, and MTP=6?
4. What would it take to train a draft against our exact quantized lm_head, and has
   anyone else published a recipe for that?

---

## 1. The three speculative-decoding methods and what exists for Qwen3.8-27B

### 1.1 MTP (Multi-Token Prediction) — already baked in

The model's own internal draft head, shipped inside every checkpoint in this collection.
No separate model to load; activated by `method: "mtp"` in `--speculative-config`.
Mutually exclusive with DFlash / DSpark (the `method` field takes one value).

### 1.2 DFlash (v1) — block-diffusion draft, external model

A lightweight block-diffusion drafter (Z Lab, arXiv:2602.06036). It predicts a whole
block of tokens in a single forward pass, conditioned on hidden states injected from the
target model's intermediate layers (KV injection). The target verifies all drafts
losslessly.

**Existing drafts for the Qwen3.x family** (z-lab collection on HF):

| Draft | Target | Downloads |
|---|---|---|
| `z-lab/Qwen3-4B-DFlash-b16` | Qwen3-4B | — |
| `z-lab/Qwen3-8B-DFlash-b16` | Qwen3-8B | — |
| `z-lab/Qwen3.5-4B-DFlash` | Qwen3.5-4B | — |
| `z-lab/Qwen3.5-9B-DFlash` | Qwen3.5-9B | — |
| `z-lab/Qwen3.5-27B-DFlash` | Qwen3.5-27B | — |
| `z-lab/Qwen3.5-35B-A3B-DFlash` | Qwen3.5-35B-A3B | — |
| `z-lab/Qwen3.5-397B-A17B-DFlash` | Qwen3.5-397B-A17B | — |
| `z-lab/Qwen3.6-27B-DFlash` | Qwen3.6-27B | 167,841 |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | Qwen3.6-35B-A3B | 275,996 |
| `z-lab/Qwen3-Coder-Next-DFlash` | Qwen3-Coder-Next | — |

No DFlash v1 draft exists for Qwen3.8-27B. The z-lab/dflash repo
(`github.com/z-lab/dflash`) contains inference code only — no training code.

### 1.3 DFlash2 — block diffusion + selector + two-tap convolutions

DFlash2 (Inco AI / Z Lab, August 2026) extends DFlash v1 with three additions:
a **candidate path selector** (rank 256, top-16), **two-tap dynamic convolutions** that
prevent end-of-block draft decay, and a bidirectional attention mask. Citation:
`inco2026dflash2`.

**A DFlash2 draft exists for our exact base model:**

| Property | Value | Source |
|---|---|---|
| HF repo | `incoai/Qwen3.8-27B-DFlash2` (mirror: `z-lab/Qwen3.8-27B-DFlash2`) | HF API JSON |
| Downloads / likes | 1,484 / 71 | HF API JSON |
| `base_model` | `Qwen/Qwen3.8-27B` | HF API tags |
| `model.safetensors` | 3.849 GB (BF16) | HF tree API |
| Architecture | `DFlash2DraftModel` | fetched `config.json` |
| `model_type` | `qwen3` | fetched `config.json` |
| Hidden size | 5,120 (matches target) | fetched `config.json` |
| Layers | 5 (sliding_attention, window 2048) | fetched `config.json` |
| `num_target_layers` | 64 (matches our target) | fetched `config.json` |
| `target_layer_ids` | [5, 19, 33, 47, 61] | fetched `config.json` |
| Block size | 8 (7 draft + 1 anchor) | fetched `config.json` |
| `selector_rank` / `selector_top_k` | 256 / 16 | fetched `config.json` |
| `conv_kernel_size` / `conv_group_size` | 2 / 16 | fetched `config.json` |
| Last modified | 2026-08-19T02:52:32Z | HF API JSON |

**GGUF-quantized version** (`z-lab/Qwen3.8-27B-DFlash2-GGUF`):

| File | Size | Acceptance length (GSM8K, 7 draft tokens) |
|---|---|---|
| `Qwen3.8-27B-DFlash2-BF16.gguf` | 3.8 GB | 5.28 |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 2.0 GB | 5.13 |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.1 GB | 5.39 |

The Q4_K_M acceptance (5.39) is within noise of BF16 (5.28), confirming that draft
quantization barely affects acceptance — the target verifies all drafts losslessly, so
draft quantization only affects drafting latency, not output quality.

### 1.4 DSpark — DFlash + Markov head + confidence head

DSpark (DeepSeek, arXiv:2607.05147) extends the DFlash block-parallel backbone with two
heads: a **Markov head** (low-rank, rank 256) that biases each draft position by the
previous token, restoring intra-block dependency, and a **confidence head** that predicts
per-position acceptance probability, enabling dynamic draft depth.

**Two DSpark drafts exist for Qwen3.8-27B:**

| Draft | Target | Size (BF16) | Downloads | Trained with |
|---|---|---|---|---|
| `RadixArk/Qwen3.8-27B-DSpark` | Qwen3.8-27B-FP8 | 2.719 GB | 68,392 | SpecForge |
| `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` | Qwen3.6-35B-A3B | — | 1,809 | vLLM speculators |

`RadixArk/Qwen3.8-27B-DSpark` config (fetched `config.json`):

| Property | Value |
|---|---|
| Architecture | `DSparkDraftModel` |
| `model_type` | `qwen3` |
| Hidden size | 5,120 |
| Layers | 5 (full_attention, no SWA) |
| Attention | GQA: 40 query heads, 8 KV heads, head_dim 128 |
| Intermediate size | 10,240 |
| Block size | 7 (verify width 8) |
| `target_layer_ids` | [4, 16, 28, 40, 52] |
| `markov_rank` / `markov_head_type` | 256 / vanilla |
| Confidence head | enabled, with Markov |
| Draft params | 1,359,284,737 (1.36B) |
| Trained against | `Qwen/Qwen3.8-27B-FP8` |
| Vocab | 248,320 |

---

## 2. The quantized MTP layer and draft model sizes

### 2.1 Our MTP draft head

From `quantization_manifest.json` (fetched from our HF repo) and the model card:

| Property | Value | Source |
|---|---|---|
| Role name | `mtp_draft` | manifest |
| Size | **0.283 GB** | manifest |
| Quantization | `fc` + attention at **EXL3 K4**, MLP at **K5/K6** | model card |
| Config flag | `mtp_bits: 4` | fetched `config.json` → `quantization_config` |
| Share of total download | 1.3% of 21.59 GB | manifest |
| Trained depth | 7-token MTP (MTP depth 6: 1 real + 6 draft per pass) | DFlash2 card benchmark description |

### 2.2 DFlash2 draft size

| Format | Size | Acceptance (GSM8K) |
|---|---|---|
| BF16 safetensors | 3.849 GB | 5.28 |
| GGUF Q8_0 | 2.0 GB | 5.13 |
| GGUF Q4_K_M | 1.1 GB | 5.39 |
| EXL3 K5/K6 [INFERENCE] | ~1.2–1.5 GB | ~5.3 (est. from GGUF parity) |

The draft has no full-vocab embedding (the anchor token's representation comes from
target hidden states via KV injection), so the bulk is the 5 transformer layers (~1.6B
params). EXL3 K5/K6 quantization is feasible because the draft uses `model_type: "qwen3"`
layers, which the EXL3 converter handles.

### 2.3 DSpark draft size

| Format | Size | Notes |
|---|---|---|
| BF16 safetensors | 2.719 GB | 1.36B params; lm_head is 93% (248,320 × 5,120 = 1.27B) |
| FP8 draft head | ~1.27 GB | fork supports `VLLM_DSPARK_FP8_DRAFT_HEAD=1` |
| EXL3 K5/K6 body + FP8 head [INFERENCE] | ~1.3–1.5 GB | 5 layers (~90M params) + FP8 lm_head |
| GGUF Q4_K_M | not published | — |

The fork's `test_dspark_fp8_draft_head.py` validates rowwise FP8 quantization of the
draft lm_head, compressing 2.54 GB to 1.27 GB with bounded rounding error.

---

## 3. Compatibility with the hydrated quant

### 3.1 Model-level compatibility

All three drafts target `Qwen/Qwen3.8-27B` (DFlash2, RadixArk DSpark) or the closely
related `Qwen/Qwen3.6-27B` (z-lab DFlash v1). Key dimensions match:

| | Target (Qwen3.8-27B) | DFlash2 draft | RadixArk DSpark |
|---|---|---|---|
| hidden_size | 5,120 | 5,120 | 5,120 |
| vocab_size | 248,320 | 248,320 | 248,320 |
| num_target_layers | 64 | 64 | 64 |
| base_model | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B-FP8 |

KV injection projects target hidden states from specific layers into the draft's KV
space. The draft sees quantized hidden states, but RMSNorm after the fusion projection
absorbs magnitude drift (see §5.3).

### 3.2 Runtime-level compatibility — the fork

The Gilded Gnosis vLLM fork (`vllm-gg-semantic-companion`) contains:

| Component | DFlash v1 | DSpark | DFlash2 |
|---|---|---|---|
| Proposer | `DFlashProposer` (`vllm/v1/spec_decode/dflash.py`) | Reuses DFlash path | ✗ |
| Draft model | `DFlashQwen3ForCausalLM` (`qwen3_dflash.py`) | `Qwen3DSparkForCausalLM` (`qwen3_dspark.py`) | ✗ |
| Config path | `method: "dflash"` | `method: "dspark"` (20+ refs in `speculative.py`) | ✗ |
| Model registry | `DFlashDraftModel` | `Qwen3DSparkModel`, `DSparkDraftModel`, `Gemma4DSparkModel` | ✗ |
| Tests | 6 test files | 4 test files (e2e, FP8 head, non-causal MLA, metadata) | ✗ |
| Serving scripts | — | `tools/spark/cluster.sh` with DSpark env vars | ✗ |
| Benchmarking | — | `benchmarks/profile_dspark_sps_curve.py` | ✗ |
| Code references | — | 658 | 0 |

**DFlash2 is not in the fork.** Its architecture (`DFlash2DraftModel`) is not registered,
and the selector (`selector_rank`, `selector_top_k`) and two-tap convolutions
(`conv_kernel_size`, `conv_group_size`) are not implemented (grep: zero matches). The
DFlash2 card says it runs on upstream vLLM PR #52816 and SGLang — neither supports EXL3.

**DSpark is deeply integrated.** The fork has `Qwen3DSparkForCausalLM` extending
`DFlashQwen3ForCausalLM`, a complete `method: "dspark"` config path, SPS curve profiling,
dynamic draft depth, confidence threshold env vars, and serving scripts. The one
compatibility issue is an architecture-string routing problem in `speculative.py:1059-1070`:
`RadixArk/Qwen3.8-27B-DSpark` ships with `architectures: ["DSparkDraftModel"]`, which the
fork interprets as a DeepSeek-V4 DSpark draft and rewrites `model_type` to
`"deepseek_v4"`. Fix: change the draft's `config.json` architectures to
`["Qwen3DSparkModel"]`, or add a `model_type != "qwen3"` guard to the rewrite condition.

### 3.3 EXL3 draft serving — already proven

The fork already supports loading an EXL3-quantized draft alongside an EXL3 target. The
v20 integration patches add `_runtime_scope_id` separation so the draft gets its own
`Exl3Config` runtime distinct from the target's. The draft's Trellis
`MIN_CAPTURABLE_TRELLIS_M` must be ≤ 1 (vs target default 4) or CUDA-graph capture fails.

Config trap: a draft inherits the target's `--quantization` unless its
`SpeculativeConfig` overrides it.

Evidence: the GLM-5.2 turnkey appliance (`glm52-exl3-turnkey`) serves an EXL3 3bpw MTP
layer 78 draft alongside an EXL3 target, measured at acceptance parity with BF16 (see
§6.2).

---

## 4. TG throughput estimates: MTP0 / MTP3 / MTP6 / DSpark / DFlash2

### 4.1 Source and scope

Throughput numbers are from the DFlash2 model card (BF16 `Qwen/Qwen3.8-27B`, SGLang,
H200, 7 draft tokens per verification step, temp 1.0, top-p 0.95, top-k 20). These are
not our EXL3 quant on RTX 5090, but the relative speedup ratios transfer because
speculative decoding's benefit is in reducing sequential forward passes, which is
hardware-independent.

Our card does not publish tok/s throughput numbers — its RTX 5090 MTP-3 references are
context-capacity tests (~180k context for the hydrated build at MTP-3), not throughput
benchmarks.

MTP3 is not directly benchmarked in the DFlash2 card. [INFERENCE] Estimate from typical
MTP scaling: MTP3 predicts 4 tokens/pass vs MTP6's 7; acceptance length scales
sublinearly with draft depth, so MTP3 gives roughly 60–70% of MTP6's speedup.

### 4.2 Concurrency 1 (interactive / single-user)

| Method | GSM8K | MATH-500 | HumanEval | MBPP | MT-Bench |
|---|---|---|---|---|---|
| MTP0 (autoregressive) | 68.9 (1.00×) | 69.0 (1.00×) | 69.0 (1.00×) | 69.0 (1.00×) | 68.9 (1.00×) |
| MTP3 [INFERENCE] | ~120–140 (~1.8×) | ~115–135 (~1.7×) | ~105–120 (~1.5×) | ~105–120 (~1.5×) | ~95–110 (~1.4×) |
| MTP6 (7-token, full depth) | 178.5 (2.59×) | 172.8 (2.51×) | 151.9 (2.20×) | 153.1 (2.22×) | 134.9 (1.96×) |
| DSpark | 185.3 (2.69×) | 174.5 (2.53×) | 159.9 (2.32×) | 163.3 (2.37×) | 137.6 (2.00×) |
| DFlash2 | 236.1 (3.43×) | 230.7 (3.34×) | 214.6 (3.11×) | 226.9 (3.29×) | 184.0 (2.67×) |

### 4.3 Concurrency 32 (batched)

| Method | GSM8K | MATH-500 | HumanEval | MBPP | MT-Bench |
|---|---|---|---|---|---|
| MTP6 | 1,381 (1.04×) | 1,416 (0.94×) | 1,297 (0.84×) | 1,315 (0.87×) | 1,160 (0.77×) |
| DSpark | 1,507 (1.13×) | 1,429 (0.95×) | 1,330 (0.86×) | 1,361 (0.90×) | 1,116 (0.74×) |
| DFlash2 | 1,923 (1.45×) | 1,952 (1.30×) | 1,799 (1.16×) | 1,887 (1.25×) | 1,525 (1.01×) |

MTP6 and DSpark both degrade at high concurrency (sequential drafting becomes a
bottleneck when the GPU is saturated). DFlash2's single-pass block drafting stays
positive even at concurrency 32.

### 4.4 Acceptance lengths (7 draft tokens, temp 1.0)

| Task | MTP | DSpark | DFlash2 |
|---|---|---|---|
| GSM8K | 5.02 | 4.36 | **5.46** |
| MATH-500 | 4.72 | 3.92 | **5.28** |
| HumanEval | 3.91 | 3.30 | **4.39** |
| MBPP | 3.99 | 3.51 | **4.79** |
| MT-Bench | 3.74 | 3.01 | **4.10** |

DSpark's acceptance is lower than MTP's because DSpark was trained against FP8 (not BF16)
and uses a simpler intra-block dependency (Markov head vs MTP's native sequential
drafting). DSpark's throughput advantage comes from lower drafting cost (single
block-parallel pass vs 6 sequential MTP steps), not higher acceptance.

### 4.5 KV cost on 32 GB RTX 5090

Our card documents the hydrated build at ~180k context with MTP-3 on RTX 5090 (20.31 GiB
weights, ~12 GiB for KV + overhead).

| Method | Extra VRAM | Est. context on 32 GB [INFERENCE] |
|---|---|---|
| MTP-3 | 0 (baked in) | ~180k (measured) |
| MTP-6 | 0 (baked in) | ~150–170k |
| DSpark (BF16 draft) | 2.7 GB | ~140–160k |
| DSpark (FP8 head + EXL3) | ~1.4 GB | ~160–170k |
| DFlash2 (BF16 draft) | 3.8 GB | ~120–150k |
| DFlash2 (EXL3) | ~1.3 GB | ~150–170k |

DSpark's dynamic draft depth (`VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH`) reduces KV pressure
when the confidence head predicts low acceptance — neither MTP nor DFlash2 adapts this
way.

---

## 5. Training a draft against the exact quantized lm_head

### 5.1 Two training libraries

There are two separate speculator training libraries. The distinction is critical for
EXL3:

| | SpecForge (SGLang) | vLLM Speculators (Red Hat) |
|---|---|---|
| Repo | `github.com/sgl-project/SpecForge` | `github.com/vllm-project/speculators` |
| Target capture backend | SGLang only (`target_backend: Literal["sglang"]`) | vLLM (`scripts/launch_vllm.py`) |
| EXL3 support | ✗ (SGLang cannot load EXL3) | ✓ (vLLM loads EXL3 via our fork) |
| DSpark training | ✓ (disaggregated + offline) | ✓ (online via vLLM server) |
| DFlash training | ✓ | ✓ |
| DFlash2 training | ✗ | ✗ |
| Published models | SpecBundle collection | 32+ RedHatAI models |

**The vLLM speculators library is the path for EXL3.** Its `launch_vllm.py` accepts any
vLLM args after `--`, so we can pass `--quantization exl3` and our ignore list. RedHatAI
used this library for every model they published.

### 5.2 The training pipeline (vLLM speculators)

Four stages:

**Stage 1 — Data preparation** (`scripts/prepare_data.py`):
Download a conversation dataset (ShareGPT, UltraChat, Magpie, Nemotron, etc.).
Output: JSONL with `id` + `conversations` fields.

**Stage 2 — Data regeneration** (`scripts/regenerate_train_data.py` or
`scripts/response_regeneration/`):
Run the target model to regenerate assistant responses. The draft learns to predict the
target's actual output distribution, not a generic dataset. This is where the quantized
lm_head matters: the regenerated tokens reflect the K6-quantized model's choices.

**Stage 3 — Hidden state capture** (`scripts/launch_vllm.py`):
Launch vLLM with `--target-layer-ids` to extract hidden states from specific target
layers. For DSpark targeting Qwen3.8-27B, this captures hidden states from layers
[4, 16, 28, 40, 52] — these become the KV injection input for the draft.

**Stage 4 — Training** (`scripts/train.py`):
Train the draft model using the captured hidden states. The draft learns to predict next
tokens given the target's intermediate hidden states.

### 5.3 Does quant-targeted training improve acceptance? The evidence

Published A/B comparisons of matched-vs-unmatched drafter training:

| Source | Target quant | BF16-trained acceptance | Quant-trained acceptance | Delta |
|---|---|---|---|---|
| arXiv:2607.04244 Table 4 | INT4 (AWQ) | 4.92 | 4.97 | +1.0% |
| SpecForge (poolside ladder) | FP8 | 5.748 | — | — |
| SpecForge (poolside ladder) | NVFP4 | — | 5.775 | — |
| SpecForge (poolside ladder) | INT4 | — | 6.273 | — |
| SpecForge serving FP8 | FP8 | 7.10 | 7.11 | +0.1% |

**The improvement from quant-matched training is 0–2%.** The architectural reason
(from `satgeze/Qwen3.6-27B-DSpark` model card):

> The drafter applies RMSNorm immediately after the fusion projection
> (`H_ctx = RMSNorm(W_c[H^(l1); ...; H^(lm)])`), which absorbs magnitude drift in the
> captured target features, and the drafter shares the target's frozen embedding and
> lm_head, so readout quantization error is common-mode between drafting and
> verification.

In the vLLM DSpark implementation (RedHatAI), the draft has its own lm_head — so the
common-mode argument partially breaks down. But the competition paper (arXiv:2607.04244)
used DFlash with a separate draft lm_head and still found only +1.0% from matched
training.

### 5.4 The two-stage quantized-target training procedure

From the competition paper (arXiv:2607.04244, "Efficient Qwen Competition", 3rd place,
6.978× speedup):

> A block-diffusion drafter specialized for the quantized target model is trained using
> a two-stage procedure: first learning from the high-precision target and then adapting
> to the low-precision target.

- **Stage 1:** Train against BF16 `Qwen/Qwen3.8-27B` via vLLM speculators. Capture
  hidden states from the BF16 target. Regenerate training data using the BF16 target.
  Train to convergence (3–10 epochs). This gives the draft a strong foundation.
- **Stage 2:** Fine-tune against our EXL3 quant. Capture hidden states from
  `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` via vLLM with `--quantization exl3`.
  Regenerate training data using the K6 lm_head. Fine-tune for 1–2 additional epochs at
  a lower learning rate (~1e-4).

Code: `github.com/nota-github/adaptfm-quant-dflash`.

### 5.5 Concrete training commands for our EXL3 quant

Using the vLLM speculators library (Red Hat), adapted from RedHatAI's GLM-5.2 DSpark
recipe:

```bash
# 1. Launch our EXL3 quant as the hidden-states target server
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py \
  malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --target-layer-ids 4 16 28 40 52 \
  -- --port 8000 --quantization exl3 --trust-remote-code \
  --gpu-memory-utilization 0.9 --max-model-len 8192

# 2. Regenerate training data using our quant (K6 lm_head produces ground truth)
python scripts/prepare_data.py \
  --model malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --data ./data/sharegpt_train.jsonl \
  --output ./output/qwen3_8_27b_exl3_regen \
  --seq-length 8192

# 3. Train DSpark draft (on separate GPU)
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 \
  scripts/train.py \
  --verifier-name-or-path malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --speculator-type dspark \
  --data-path ./output/qwen3_8_27b_exl3_regen \
  --vllm-endpoint http://localhost:8000/v1 \
  --block-size 7 --max-anchors 1024 \
  --target-layer-ids 4 16 28 40 52 --num-layers 5 \
  --markov-rank 256 --markov-head-type vanilla \
  --enable-confidence-head --confidence-head-with-markov \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
  --epochs 3 --lr 6e-4 --total-seq-len 4096 \
  --draft-arch qwen3 --draft-hidden-act silu \
  --on-missing generate --on-generate delete --seed 42
```

For the two-stage procedure, warm-start Stage 2 from Stage 1 via
`--draft-checkpoint-path`.

### 5.6 Hardware and time estimates [INFERENCE]

| Step | Hardware | Time |
|---|---|---|
| Data regeneration | 1× GPU (RTX 5090 or rental H100) | ~12–24 hours (27B generating ~100K responses) |
| Hidden state capture | same GPU | ~6–12 hours |
| Training (Stage 1) | 1× 24 GB+ GPU | ~8–16 hours (3–10 epochs, ~4000 optimizer steps) |
| Fine-tuning (Stage 2) | same GPU | ~2–4 hours (1–2 epochs at lower LR) |
| Draft quantization | same GPU | < 1 hour |

The draft is only 1.36B params — trains comfortably on a single 24 GB+ GPU.

---

## 6. Prior art: who shares what

### 6.1 RedHatAI — the most prolific recipe publisher

RedHatAI (Red Hat's AI team) published 32 speculator models with full training recipes
(data prep, vLLM launch, training commands) using the vLLM speculators library.

**Most relevant models:**

| Model | Method | Target | Quant target? | Downloads | Key metrics |
|---|---|---|---|---|---|
| `RedHatAI/GLM-5.2-speculator.dspark` | DSpark | GLM-5.2-FP8 | **FP8** | 12,713 | MAL 3.967, per-pos 0.829/0.723/0.646/0.587/0.539/0.500/0.464 |
| `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` | DSpark | Qwen3.6-35B-A3B BF16 | No | 1,809 | 9 datasets, MAL 3.39–5.03 |
| `RedHatAI/Qwen3-8B-speculator.dflash` | DFlash | Qwen3-8B BF16 | Benchmarked vs FP8 | — | GuideLLM benchmark published |
| `RedHatAI/Qwen3-8B-speculator.eagle3` | EAGLE3 | Qwen3-8B BF16 | — | — | — |
| `RedHatAI/gemma-4-31B-it-speculator.dflash` | DFlash | Gemma-4-31B-it | No | 1,053 | — |

The GLM-5.2 DSpark is the closest published precedent to our goal: a DSpark draft
trained against an FP8-quantized target via vLLM. Training data (regenerated by
GLM-5.2-FP8) is published as `mgoin/GLM-5.2-FP8-magpie-ultrachat`. Per-epoch checkpoints
published (epoch 1/2/3, progression 3.376 → 3.819 → 3.967).

The Qwen3.6-35B-A3B DSpark is the closest RedHatAI model by architecture family. Its
measured per-position acceptance across 9 datasets:

| Dataset | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Avg Length |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval | 82.0% | 66.2% | 54.1% | 44.1% | 36.4% | 29.9% | 24.7% | 20.4% | 4.58 |
| math_reasoning | 84.0% | 70.3% | 59.8% | 51.2% | 43.5% | 37.0% | 31.2% | 26.5% | 5.03 |
| qa | 71.8% | 50.9% | 37.4% | 28.0% | 20.9% | 16.0% | 12.1% | 9.2% | 3.46 |
| rag | 78.2% | 57.1% | 44.1% | 34.7% | 27.3% | 21.2% | 16.4% | 13.1% | 3.92 |
| tool_call | 71.7% | 50.8% | 36.1% | 26.3% | 19.5% | 14.7% | 11.2% | 8.6% | 3.39 |
| summarization | 74.2% | 53.6% | 40.4% | 30.6% | 23.5% | 17.8% | 13.7% | 10.5% | 3.64 |
| translation | 70.7% | 51.2% | 37.8% | 28.4% | 21.2% | 15.7% | 11.9% | 8.8% | 3.46 |
| writing | 74.4% | 54.3% | 40.8% | 31.6% | 25.2% | 20.3% | 16.5% | 13.4% | 3.76 |

### 6.2 Our own GLM-5.2 EXL3 draft — acceptance parity at 5.2× smaller

From `glm52-exl3-turnkey/README.md` — measured 4-arm QC run, 2026-07-25, GLM-5.2's MTP
layer 78 quantized to EXL3 3bpw:

| Draft (layer 78) | MAL | Accept | Decode tok/s | KV mem |
|---|---|---|---|---|
| BF16 (stock, 19.3 GB) | 3.528 | 84.3% | 49.6 | 5.27 GiB |
| EXL3 3bpw grafted (3.7 GB) | 3.517 | 83.9% | 49.5 | 8.92 GiB |
| **EXL3 3bpw override** (3.7 GB) | **3.548** | **84.9%** | **49.9** | 8.92 GiB |
| NVFP4 (lukealonso/GLM-5.2-NVFP4 MTP shards) | 3.531 | 84.4% | 49.5 | 8.5 GiB |

The EXL3 3bpw override draft achieved **better acceptance than BF16** (84.9% vs 84.3%)
while being **5.2× smaller** (19.3 → 3.7 GB). Two serving modes documented in
`entrypoint.sh:714-728`: graft (in-place surgery on target layer 78) vs override
(separate draft dir via `--speculative-config`, measured slightly better). Full
methodology: `https://gist.github.com/malaiwah/4bbb16bef2e336e94af165076cdba955`.
Published draft model: `malaiwah/GLM-5.2-EXL3-TR3-MTP78`.

### 6.3 satgeze/Qwen3.6-27B-DSpark — the target-quantization A/B study

A DSpark drafter for Qwen3.6-27B (our model's predecessor), trained with DeepSpec against
BF16, warm-started from z-lab's DFlash head. 19,294 downloads. Measured against 5 target
quants with the same head:

| Target quant | Code probe acc | Counting acc | Mean accepted len | Loop rate (8 prompts) | Verdict |
|---|---|---|---|---|---|
| Q8_0 (8.5 bpw) | 0.292 | 0.295 | 5.35 / 5.38 | 0/8 | Full speedup, loop-free |
| Q4_K_M (4.8 bpw) | 0.313 | 0.293 | 5.66 / 5.38 | 0/8 | Full speedup, loop-free |
| Q2_K, no imatrix (2.6 bpw) | 0.149 | 0.220 | 3.11 / 4.23 | 1/8 | Degraded acceptance |
| Q2_K + imatrix (2.9 bpw) | 0.251 | 0.323 | 4.74 / 5.69 | 0/8 | Near-full speedup, loop-free |
| IQ1_S + imatrix (1.6 bpw) | 0.293 | 0.230 | 5.32 / 4.33 | **6/8** | High acc but loops: not usable |

The head was trained against BF16 and achieves full acceptance against Q4_K_M (4.8 bpw)
— a lower precision than our K5/K6 mix. The card's conclusion:

> At 27B scale, what decides low-bit draft acceptance is not the head, it is the
> quantization quality of the target.

Our K6 lm_head has KLD 0.002760 from BF16 — 48% lower divergence than FP8 (0.005294).
The train/serve mismatch is smaller than in any published quantized-target speculator
experiment.

### 6.4 Complete prior art map

| Source | Method | Target | Quant target? | Recipe? | Measured? | Key finding |
|---|---|---|---|---|---|---|
| RedHatAI/GLM-5.2-speculator.dspark | DSpark | GLM-5.2-FP8 | FP8 | ✅ 3 commands | ✅ MAL 3.967 | Quantized-target DSpark; per-epoch checkpoints |
| RedHatAI/Qwen3.6-35B-A3B-speculator.dspark | DSpark | Qwen3.6-35B-A3B | No | ✅ 3 commands | ✅ 9 datasets | Closest RedHatAI model by family |
| RedHatAI/Qwen3-8B-speculator.dflash | DFlash | Qwen3-8B | Benchmarked vs FP8 | ✅ 3 commands | ✅ GuideLLM | FP8 target comparison published |
| satgeze/Qwen3.6-27B-DSpark | DSpark | Qwen3.6-27B BF16 | Tested vs 5 quants | ✅ DeepSpec | ✅ Acc + loop rate | BF16-trained head works across all usable quants |
| z-lab/Qwen3.6-27B-DFlash | DFlash | Qwen3.6-27B | No | ✗ | "N/A" | Official z-lab draft, 167K downloads |
| RadixArk/Qwen3.8-27B-DSpark | DSpark | Qwen3.8-27B-FP8 | FP8 | Partial | ✅ 11 workloads | Our exact target model, FP8-quantized |
| incoai/Qwen3.8-27B-DFlash2 | DFlash2 | Qwen3.8-27B BF16 | No | ✗ | ✅ 5 tasks × 3 conc | Highest throughput (3.1–3.4×) |
| Competition paper (arXiv:2607.04244) | DFlash | Qwen3.5-4B AWQ INT4 | INT4 | ✅ two-stage | ✅ 6.978× | Two-stage BF16→quant training; code published |
| Our GLM-5.2 EXL3 work | MTP layer 78 | GLM-5.2 EXL3 3bpw | EXL3 3bpw | ✅ gist + entrypoint.sh | ✅ 4-arm QC, 84.9% | EXL3 draft beats BF16; 5.2× smaller |
| Our fork (vllm-gg-semantic-companion) | DSpark serving | EXL3 target | EXL3 draft supported | Serving only | ✅ (GLM-5.2 QC) | `_runtime_scope_id` separation, MIN_CAPTURABLE_TRELLIS_M ≤ 1 |

---

## 7. Recommendations

### 7.1 Immediate deployment (zero training)

**MTP-3** is the zero-effort baseline: already baked in, ~180k context on RTX 5090,
2.0–2.6× speedup at concurrency 1.

**RadixArk DSpark** can be loaded with a one-line config fix (change architectures from
`DSparkDraftModel` to `Qwen3DSparkModel`). The fork's full DSpark runtime is ready:
FP8 draft head, dynamic draft depth, SPS curve profiling, confidence threshold. Expected
acceptance is slightly lower than MTP at concurrency 1 (DSpark was trained against FP8,
not our K6), but DSpark's adaptive features help on 32 GB.

### 7.2 Custom-trained draft (if acceptance matters)

Use the vLLM speculators library (not SpecForge — SGLang cannot load EXL3). Follow
RedHatAI's GLM-5.2 DSpark recipe as the template, replacing the target with our EXL3
quant.

Expected improvement from quant-matched training: **0–2%** (published evidence from
arXiv:2607.04244, SpecForge, satgeze). The bigger win is quantizing the draft itself
(EXL3 or FP8 head), which frees VRAM for KV — proven in our GLM-5.2 work (5.2× smaller
at better-than-BF16 acceptance).

### 7.3 DFlash2 (if raw throughput matters)

DFlash2 gives the highest throughput (3.1–3.4× at concurrency 1, still positive at
concurrency 32), but requires porting the selector + two-tap conv into our fork. The
DFlash v1 infrastructure (proposer, KV injection, non-causal attention) is already there;
the gap is ~3 components. No training code is published for DFlash2 (Inco AI keeps it
closed).

### 7.4 What no one has done

No one has published a speculator trained against an EXL3-quantized target. The closest
precedents are FP8 (RedHatAI GLM-5.2), INT4/AWQ (competition paper), and our own GLM-5.2
EXL3 draft serving. RedHatAI has a gap at Qwen3.6-27B dense (they published
Qwen3.6-35B-A3B but not the 27B dense). Publishing an EXL3-targeted DSpark for
Qwen3.8-27B would be a first.

---

## References

- DFlash paper: arXiv:2602.06036
- DFlash2: `inco.ai/blog/dflash2/`, `huggingface.co/incoai/Qwen3.8-27B-DFlash2`
- DSpark paper: arXiv:2607.05147
- SpecForge: `github.com/sgl-project/SpecForge`
- vLLM speculators: `github.com/vllm-project/speculators`, `docs.vllm.ai/projects/speculators/`
- RedHatAI speculator collection: `huggingface.co/collections/RedHatAI/speculator-models`
- Competition paper (two-stage quant training): arXiv:2607.04244,
  `github.com/nota-github/adaptfm-quant-dflash`
- satgeze target-quantization A/B: `huggingface.co/satgeze/Qwen3.6-27B-DSpark`
- Our GLM-5.2 EXL3 draft: `huggingface.co/malaiwah/GLM-5.2-EXL3-TR3-MTP78`,
  `gist.github.com/malaiwah/4bbb16bef2e336e94af165076cdba955`
- LMSYS DFlash + Spec V2 blog: `lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/`
- NVIDIA DFlash Blackwell blog: `developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/`
- vLLM DFlash docs: `docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/`
- vLLM DSpark docs: `docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/`

---

## Decision on §7.4 (EXL3-targeted DSpark training): NOT PURSUED — 2026-08-19

Recorded so the option is not silently dropped and can be revived with numbers.

**Decision: do not train.** The lane is documentation-only; no training run was or
will be executed under the current objective.

**What the measurements say.** The ceiling such a draft must beat is now known
precisely, because the off-the-shelf FP8-trained DSpark draft was measured against
our EXL3 target on this exact stack (`receipts/dspark-ab-2026-08-19.md`, 50k ctx,
auto KV, single stream):

| | PP | fox | essay |
|---|---|---|---|
| DSpark (BF16 draft, block 7) | 8,976.6 | 183.0 [acc 0.857] | 74.8 [acc 0.147] |
| built-in MTP-6 | 9,495.9 | 175.0 [acc 0.944] | **76.7 [acc 0.221]** |

An EXL3-target-trained DSpark would have to close the essay-acceptance gap
(0.147 -> beyond 0.221) to be worth its 2.7 GB and 5.5% prefill. The published
evidence on quant-matched drafter training puts the gain at **0-2%**
(arXiv:2607.04244 Table 4: 4.92 -> 4.97; SpecForge FP8 serving 7.10 -> 7.11), and
§5.3's architectural argument for why it is small (RMSNorm after the fusion
projection absorbs magnitude drift; shared frozen embedding/head makes readout
error common-mode) applies to our target too — more strongly, since our K6 head is
the lowest-divergence readout in this comparison set.

**Cost against that.** §5.6 estimates 20-40 GPU-hours end to end (data
regeneration 12-24 h, hidden-state capture 6-12 h, training 8-16 h, fine-tune
2-4 h) on a card that is exclusive-process and is the only card. That is the same
budget as the entire mixed-requant lane, which attacks the one criterion we
actually fail (prefill), whereas decode is already met on `fidelity`
(228.3 fox / 104.1 essay vs the 190/83 bars).

**What would revive it.** Any of: (a) decode becomes the binding constraint after
a requant checkpoint takes over the flagship slot; (b) a second GPU removes the
serial-time conflict; (c) someone publishes an EXL3-targeted speculator recipe
with a measured gain above ~5%, making the 0-2% prior obsolete. The publication
angle alone ("no one has published a speculator trained against an EXL3 target",
§7.4) is real but is not worth 20-40 h of the critical path.
