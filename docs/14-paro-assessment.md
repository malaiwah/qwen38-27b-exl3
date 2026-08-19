# ParoQuant: a pre-quantization transform, assessed against our EXL3 work

This is a decision record for [`z-lab/Qwen3.8-27B-PARO`](https://huggingface.co/z-lab/Qwen3.8-27B-PARO)
and the method behind it. The underlying investigation, with every file header read
verbatim and every URL, is
[`receipts/paro-comparison-2026-08-19.md`](../receipts/paro-comparison-2026-08-19.md);
this file is its citable summary. Every external claim below carries its link; every
internal number carries its receipt path. Our own context for the comparison is the
served `fidelity` profile at mean KLD **0.003405** over 512 contexts (`receipts/kld5-1M-tail-hyd.json`
shard-0 family) and the offline ten-shard hydrated mean **0.002700** over 10,480,640
positions (`receipts/kld5-10M-hyd.json`), both stated in the README status header and
serving table.

## What PARO is

PARO = **Pairwise Rotation Quantization** ([arXiv:2511.10645](https://arxiv.org/abs/2511.10645),
"ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference",
Liang, Chen, Zhang, Han, Liu; accepted to ICLR 2026; authors at UCSD, NVIDIA, MIT).
Reference implementation: [`github.com/z-lab/paroquant`](https://github.com/z-lab/paroquant)
(MIT licence, PyPI `paroquant` 0.1.16); blog at <https://paroquant.z-lab.ai>.

It is a **post-training, weight-only pre-quantization transform**, not a storage
format and not a fine-tune. Before quantizing weights to INT4 it applies a learned
**scaled pairwise rotation** — K=8 independent Givens rotations (each rotating one
disjoint channel pair within a 128-channel group) combined with channel-wise scaling
— to suppress outliers and narrow the per-group dynamic range (paper §4). At runtime
a custom fused CUDA kernel applies the inverse rotation + inverse scaling to
activations, then a standard **AWQ-Marlin INT4×FP16 GEMM** runs the matmul.
Activations stay FP16/BF16 throughout (paper §2.1: "weight-only PTQ"). The rotation
is "independent" (each channel in at most one pair per step), so the kernel is
embarrassingly parallel with no intra-block synchronisation beyond `__syncthreads`.

## The checkpoint vs the transform

[`z-lab/Qwen3.8-27B-PARO`](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) is one
*instance* of the method applied to our base model. It is a VLM
(`Qwen3_5ForConditionalGeneration`, image-text-to-text); `config.json` carries
`quantization_config` `{quant_method: paroquant, bits: 4, group_size: 128, krot: 8}`,
zero-point true (asymmetric, AWQ-style). It ships as a single 17.485 GiB
`model.safetensors` (18,773,962,320 bytes): ~11.46 GiB of packed INT4 language weights,
~5.17 GiB of unquantized F16 (embed, `lm_head`, layernorms, the small linear-attention
projections) and ~0.86 GiB of F16 vision tower, which `--language-model-only` skips.
The rotation parameters (`theta`, `pairs`, `channel_scales`) total ~70 MB across all
64 layers — negligible. The language model alone is ~16.6 GiB resident.

The distinction that matters: the **transform** (learned rotation + scaling) is
orthogonal to the **quantization format** that follows it. This checkpoint chose INT4
linear (AWQ-Marlin) as that format. The transform does not prescribe it.

## Why the checkpoint is not a comparator

Three independent reasons, any one of which is sufficient.

1. **A format difference would confound the transform's value.** Running our v5 KLD
   suite on the PARO checkpoint would measure INT4-linear-plus-rotation against our
   EXL3 trellis-plus-Hadamard. That is a comparison of two whole pipelines, not of the
   pre-transform; a PARO win or loss could not be attributed to the rotation because the
   storage format changes at the same time. The experiment we actually want (does a
   learned rotation beat a fixed Hadamard *in front of our trellis quantizer*) is
   different, and is described below.
2. **The card carries no quality metrics for this checkpoint.** No PPL, no KLD, no
   accuracy, no throughput — only install instructions and a citation. All quality
   claims must be sourced from the paper, which evaluates *different, smaller* models
   (LLaMA-2-7B, LLaMA-3-8B/70B, Qwen3-1.7B/4B/8B/14B). No Qwen3.8-27B result is
   reported anywhere.
3. **The paper's protocol is not comparable to ours.** It reports PPL (WikiText2, C4)
   and accuracy (MMLU-Pro, GPQA, AIME, BoolQ, ARC, HellaSwag) on Qwen3-8B, against an
   FP16 reference, on an RTX A6000 (SM86, 48 GB), with `lm-evaluation-harness` /
   `lighteval`. Our v5 suite is mean KLD against a BF16 teacher over 5,120 contexts ×
   2,047 positions on an RTX 5090 (SM120, 32 GB). Different model size, different
   metric, different reference, different sample size, different dataset, different
   hardware. None of their numbers transfers. The paper's Qwen3-8B PPL (6.29 vs 6.24
   FP16 on WikiText2) tells us the method is competitive with QTIP and better than AWQ
   on linear INT4 — a statement about the method on a smaller model, not about this
   checkpoint's KLD on our scale.

## Orthogonality: the transform could precede EXL3

The rotation+scaling operates on the weight matrix before quantization and on
activations before the GEMM; it does not prescribe the quantization method. In
principle one could: (i) apply ParoQuant's learned pairwise rotations + channel
scaling to BF16 weights, (ii) quantize the transformed weights with EXL3 trellis
(K5/K6) instead of INT4 linear, (iii) at runtime apply the inverse rotation to
activations then run the EXL3 trellis GEMM.

EXL3 already uses a **fixed Hadamard** pre-transform for incoherence processing (as
QTIP does). The paper's central empirical claim is that **learned rotations outperform
a fixed Hadamard** — ParoQuant matches QTIP's accuracy while being ~25 % faster
(paper §5). If that advantage transfers to our trellis quantizer, replacing EXL3's
Hadamard with a learned rotation could push our KLD floor below the published
**0.002700** (`receipts/kld5-10M-hyd.json`). That is the hypothesis worth testing, and
it is why the checkpoint itself is beside the point: the question is about the
transform, not the INT4-linear artefact.

## The honest counter-argument

The gain may be smaller for us than the paper implies, for three reasons that should
not be glossed over.

- **Trellis already handles non-Gaussian weights better than linear INT4.** Trellis is
  vector quantization with codebooks; its advantage over scalar INT4 is precisely on
  non-uniform, outlier-heavy distributions. The outlier suppression that rotation buys
  for linear quant may be partly redundant in front of a codebook quantizer, so the
  headroom could be narrower than the paper's linear-INT4 gains suggest.
- **The optimization objective would change.** ParoQuant tunes rotation parameters to
  minimise *linear-quantization-induced* output error. In front of EXL3 the objective
  must minimise *trellis-quantization-induced KLD* — a different loss, needing a
  different inner loop. The optimisation code (`paroquant/cli/optimize.py`,
  `experiments/optimize/4bit.sh`) would need adaptation, not just re-pointing.
- **Group and runtime alignment is open.** ParoQuant groups at 128 channels; EXL3
  trellis groups by code structure, and the two may not share boundaries. The inverse
  rotation also adds ~10 % per linear layer (paper) on top of EXL3's own dequant cost —
  acceptable against our throughput profile, which carries 250 K-context headroom, but
  not free.

These are open questions, not refutations. They are the substance of the follow-up
study and are tracked there.

## Runnability on our stack

The checkpoint is **likely runnable on SM120 but not on our r34 image as-is.** The
vLLM plugin requires `vllm >=0.19.1, <0.20` (`pyproject.toml`); our stack is the
Gilded Gnosis r34 fork (vLLM `4d006a4` + b12x `cd3ce19` + FlashInfer `1ac6942`,
CUDA 13.2), which is a different lineage and imports different internal APIs
(`AWQMarlinLinearMethod`, `scalar_types`, `PackedvLLMParameter`). The plugin
auto-registers via vLLM's `general_plugins` entry point and sets
`get_min_capability()` to 75, so SM120 passes the gate; the rotation kernel uses only
shared memory and thread-level multiply-add (no SM-specific instructions, no tensor
cores), and AWQ-Marlin in vLLM 0.19.1 with CUDA 13.0 should include SM120 support —
but none of that is empirically confirmed for Blackwell-consumer in their docs, and
their efficiency benchmarks use an A6000.

It fits the card: ~16.6 GiB language-only weights + 8.89 GiB KV for 238,400 tokens +
~2.8 GiB activations/graphs/overhead = ~28.3 GiB against our 31.40 GiB budget, ~3.1
GiB spare. Native context is 262,144 (`max_position_embeddings`), above our 238,400
need; no RoPE scaling required. Three serving routes exist, none on our current
image: the official `ghcr.io/z-lab/paroquant:serve` image (cleanest, but a different
vLLM with no b12x and no our EXL3 kernels), a separate `vllm==0.19.1` environment, or
porting the ~313-line plugin to our fork's API.

## Decision and follow-up

**Decision (c): the method is orthogonal and worth trialling as a pre-step to our own
quantisation; the checkpoint is not added to the comparator set.** The transform is
the valuable artefact, the INT4-linear checkpoint is not, and a direct v5 KLD run on
it would answer a format question rather than the transform question we care about.

The transform trial is the open item: adapt ParoQuant's rotation optimisation to
target trellis-quantization KLD, apply the learned rotations to Qwen3.8-27B weights
before our EXL3 K5K6 quantization, and measure with the v5 suite. The feasibility
study — whether the optimisation can be retargeted, how the group alignment and
runtime cost shake out, and whether the Hadamard-replacement hypothesis survives the
counter-arguments above — is in progress at
[`docs/13-learned-rotations-feasibility.md`](13-learned-rotations-feasibility.md).
The standing todo is: **trial learned rotations before EXL3 quant**.

## Sources

| Source | URL |
|---|---|
| Model card | <https://huggingface.co/z-lab/Qwen3.8-27B-PARO> |
| Paper (arXiv) | <https://arxiv.org/abs/2511.10645> |
| Code | <https://github.com/z-lab/paroquant> |
| PyPI | <https://pypi.org/project/paroquant/> |
| Blog | <https://paroquant.z-lab.ai> |
| HF collection | <https://huggingface.co/collections/z-lab/paroquant> |
| Full investigation | [`receipts/paro-comparison-2026-08-19.md`](../receipts/paro-comparison-2026-08-19.md) |
