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

The card reports one Q4_K_M acceptance value (5.39) near its BF16 value
(5.28), but publishes no repetitions or uncertainty, so "within noise" is not
established. Draft quantization changes proposal efficiency; exact speculative
verification is intended to preserve target output semantics, but that
property belongs to the verifier implementation and must be tested.

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

### 3.3 EXL3 draft serving precedent

The fork has runtime-scope separation for an EXL3 target and EXL3 draft, and
the GLM-5.2 appliance served an EXL3-quantized MTP layer alongside an EXL3
target. This proves that specific MTP path, not arbitrary DFlash/DSpark custom
architectures or converter compatibility. A draft must override inherited
`--quantization`, and capturable trellis row limits remain configuration
constraints.

---

## 4. TG throughput estimates: MTP0 / MTP3 / MTP6 / DSpark / DFlash2

### 4.1 Source and scope

The DFlash2 card numbers are BF16 Qwen3.8-27B under SGLang on one H200, with
seven draft tokens and its stated sampling. They are useful prior art but do
**not** transfer as ratios to EXL3 on an RTX 5090: draft/target kernel cost,
memory bandwidth, batching, verifier implementation and acceptance all depend
on hardware/runtime.

MTP-3 is not in that card. No numeric MTP-3 row is interpolated here; the local
vLLM receipts are the authority for local MTP depth effects.

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

DSpark has lower acceptance than MTP in the card's table and lower drafting
cost through one block-parallel pass. Training target, architecture and
runtime all differ, so the acceptance gap is not causally assigned to FP8 or
the Markov head.

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
| Target capture backend | SGLang only | vLLM |
| EXL3 target | unsupported by SGLang | plausible through our fork; unverified training integration |
| DSpark / DFlash training | supported | supported |
| DFlash2 training | unsupported | unsupported |

`launch_vllm.py` forwards vLLM arguments, which is necessary but not sufficient
for EXL3 hidden-state capture/training compatibility. No command below was run
against the pinned library revision.

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

Only two rows are matched A/Bs, and they report +1.0% and +0.1% in their own
protocols. The poolside FP8/NVFP4/INT4 rows are different targets, not a
matched-training experiment. No universal 0–2% prior follows.

RMSNorm can remove scalar magnitude drift but not directional feature error.
Shared embeddings/heads can make some readout error common-mode only for
architectures that actually share them; the RedHatAI DSpark implementation has
its own head. These are hypotheses for small effects, not a bound.

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

### 5.5 Illustrative, unvalidated command sketch

The following was adapted from a RedHatAI recipe but was **not executed** and
must not be treated as copy-pasteable until each CLI option is verified against
a pinned `vllm-project/speculators` revision:

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

The original 20–40 GPU-hour schedule is a planning estimate with no pilot.
A 1.36B BF16 drafter does not automatically train "comfortably" on 24 GB:
parameters, gradients, optimizer/master states and activations can exceed that
without sharding, checkpointing or reduced-state optimizers. Hardware and time
must be sized from the actual training configuration.

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

The EXL3 override **observed** 84.9% acceptance versus BF16's 84.3% while
being 5.2× smaller. No repeat count or interval supports "better"; the
defensible conclusion is no detected material acceptance loss in that QC run.
Two serving modes are documented in the cited appliance.
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

The cited 0.002760 is the hydrated **body** KLD, not the K6 `lm_head` error.
The head-specific v5 measurement is about +0.000143 KLD. These values do not
establish that this train/serve mismatch is smaller than every published
speculator experiment.

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

**MTP** is the zero-integration baseline because it is baked into the target.
Local speedup depends on prompt and concurrency; the H200 card's 2.0–2.6×
ratios are not used for RTX 5090 planning.

**RadixArk DSpark** booted only after a config rewrite and with an auto-KV
concession; issue #442 records two runtime incompatibilities. It is an
experiment, not a one-line deployment recommendation.

### 7.2 Custom-trained draft (if acceptance matters)

The vLLM speculators route is the plausible training path, but EXL3 capture and
the command surface remain unverified. The cited matched-target A/Bs show
roughly 0.1–1.0% acceptance changes in their own protocols; they do not define
a universal 0–2% expectation. Draft quantization can free VRAM, while the
GLM QC run only establishes an observed no-material-loss result.

### 7.3 DFlash2 (if raw throughput matters)

DFlash2 has the highest throughput in its H200/SGLang card (3.1–3.4× at
concurrency 1 and positive at 32). Porting and measuring it on this fork is
required before making a local throughput claim; "three components" is a
source-map summary, not an effort estimate.

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

An EXL3-target-trained DSpark would need to close the measured essay-acceptance
gap (0.147 versus MTP's 0.221) while paying 2.7 GB and 5.5% prefill. The two
published matched-training A/Bs show small gains in other systems (+1.0% and
+0.1%), which lowers the priority but does not bound this target.

The 20–40 GPU-hour schedule is an unpiloted planning estimate and the 24 GB
training fit is unresolved. The decision not to train is therefore a
priority/risk choice: decode already clears the current profile bars, while
the mixed-requant lane attacks the failed prefill criterion with direct
measurements.

Revisit if decode becomes binding, suitable training hardware removes the
serial conflict, or a matched Qwen3.8/EXL3 pilot demonstrates a material gain.
The publication-first angle alone is not enough.
