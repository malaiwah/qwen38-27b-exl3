# Progress log

## 2026-08-14

**Artifacts staged.** Downloaded and byte-verified against the HF API at pinned
revisions (0 size mismatches, 0 missing across 121 files / 156.5 GB):
`Qwen/Qwen3.6-27B@6a9e13bd`, `nvidia/Qwen3.6-27B-NVFP4@0893e160`,
`Qwen/Qwen3.8-27B@1d4bf0f2`, `unsloth/Qwen3.8-27B-NVFP4@9c73e2da`. Stored on the
777 GB overlay at `/var/tmp/models` (the durable 100 GB volume cannot hold all
four); the overlay is container-local, so a rebuild means re-downloading.

**Reference recipes measured**, not read off cards — see
[docs/01-nvfp4-composition.md](docs/01-nvfp4-composition.md). Both vendors
4-bit only the MLP and keep attention at 8-bit; both keep the vision tower BF16.

**Recipe fixed** — [docs/02-recipe-k4.md](docs/02-recipe-k4.md). MLP K4,
attention BF16-on-disk for runtime K6, `lm_head` K6, MTP K6, embed/vision BF16.
18.75 GB VRAM vs NVIDIA's 21.92 GB, with the headroom exchange rates recorded
for a later `down_proj` promotion.

**Runtime contract established** — [docs/03-gg-runtime-contract.md](docs/03-gg-runtime-contract.md).
Four findings that change the plan: no Qwen `MODEL_FAMILY` (bypass the
launchers, call `vllm serve` directly), `--enforce-eager` is mandatory for a
dense EXL3 checkpoint, `--quantization exl3` must be explicit, and the
online-K6 `ignore` list must be written from scratch or the overlay will claim
the vision tower and silently degrade it to MXFP8.

**Toolchain gaps identified** — [docs/04-exllamav3-toolchain.md](docs/04-exllamav3-toolchain.md).
exllamav3 supports this architecture first-class but has no per-module bit
override and cannot emit BF16 for a decoder linear, so the plan is convert-then-splice.

**Baselines proven** — [docs/06-baseline-validation.md](docs/06-baseline-validation.md).
No container runtime is available (seccomp denies user/mount namespaces), so the
r34 image was pulled with an unprivileged OCI puller and is run through `proot`.
Both `unsloth/Qwen3.8-27B-NVFP4` and BF16 `Qwen/Qwen3.8-27B` serve and answer
requests under the official image, at full GPU speed.

**Card review landed** — [docs/07-serving-recommendations.md](docs/07-serving-recommendations.md)
and [docs/08-upstream-cards-digest.md](docs/08-upstream-cards-digest.md), plus
`https://recipes.vllm.ai/Qwen/Qwen3.8-27B`. Findings that changed our plan:
both NVFP4 vendors ship an undocumented FP8 KV-cache scheme (so KLD must pin one
KV dtype), both deliberately preserve the MTP head (so ours stays BF16 too), the
native context is 262144 with the 1M override nested under `text_config`, and
Unsloth's chat-template edits are partly undisclosed (we ship upstream's).

**exllamav3 1.4.2 installed inside the image rootfs** (`5f3c537`), because the
image's bundled copy is 0.0.43 and has no converter. The CUDA extension is being
JIT-built for `TORCH_CUDA_ARCH_LIST=12.0` against the image's torch 2.12.0+cu132.

**Iteration 1 built, served and measured** — [docs/10-results-iteration-1.md](docs/10-results-iteration-1.md).
K4 conversion took ~33 min on one GPU; the spliced checkpoint serves under the
official r34 image at **17.89 GiB resident weights** (predicted 19.28 GB, 0.4 %
error), answers text and image prompts correctly, and scores **mean KLD 0.034030
(run SD 0)** against the BF16 teacher. Published to
[`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4).

**Upstream defect filed** — [local-inference-lab/vllm#311](https://github.com/local-inference-lab/vllm/issues/311):
the online overlay's MXFP8 fallback raises for shapes divisible by neither 128
nor 32, which is exactly this architecture's vision tower. Patch pushed to a fork
branch, PR opened after behavioural verification on this box.

**v3 held-out re-measurement.** The v2 suite came from exllamav3's calibration
corpora — the text our own quant was tuned on. Rebuilt the suite from Gutenberg,
arXiv, Wikipedia (9 languages) and CPython with a shingle contamination scan (0 hits),
181 contexts / 370,507 positions, 41 real source clusters, analysis/qualification
partitions, 32 sentinels. Ours moved 0.026231 -> 0.030736, NVFP4 -> 0.094978,
FP8 -> 0.013126. Advantage over NVFP4 grew to 3.09x (136/136 contexts);
deficit to FP8 grew to 0.0176. Noise floor measured at exactly 0.000000.

**CUDA graphs landed.** Autotune-priming patch verified end to end: three priming
lines, decode-only capture, +92 %/+84 %/+98 % decode throughput, now faster than the
NVFP4 checkpoint. [PR #314](https://github.com/local-inference-lab/vllm/pull/314).
The receipt published with it also claimed "exact distribution parity against eager
(KLD 0.000000)"; that claim was **withdrawn the next day** — it captured a prefill
forward, which `FULL_DECODE_ONLY` never graphs. See the 2026-08-15 entry and
[docs/27](docs/27-graph-decode-drift-control.md).

**Everything needed to contest our numbers is published** as
[`malaiwah/qwen38-27b-fidelity-suite-v3`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3):
tokens, reference and candidate hidden states, the shared LM head, sentinel repeats,
reports and checksums — recomputable without a GPU.

**Weakest control identified:** replay qualification 6.54e-04 versus the reference
protocol's 1.23e-06, which sets a ~1e-3 resolution floor and invalidates the
iteration-1 head-attribution figure until fixed. Top of the iteration-2 list.

## 2026-08-15

**Iteration 2 built, measured and published** — [docs/22](docs/22-results-iteration-2.md).
Gate K5 / up K5 / down K6, `mcg` throughout including `lm_head`, quantized MTP draft,
attention still BF16-on-disk for the runtime's K6 overlay. Enabled by a 30-line
`EXL3_BITS_OVERRIDE` addition to exllamav3's allocator, since upstream exposes only a
global `--bits`. Conversion 38m52s, whole-model 4.95 bpw, per-block `rfn 0.0066 /
sqnr 45.1 dB`. 30.60 GB download serves at **20.32 GiB resident weights** and measures
**mean KLD 0.008157** body-only (**0.008284** as served, since the K6 head adds
0.000127) with top-1 96.97 % on the held-out analysis partition: **-0.004969 versus
official FP8** (95 % CI [-0.00643, -0.00371], 136/136 contexts) and -0.022579 versus the
K4 release. Published as
[`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6).
Speculative decoding with the quantized draft works: 58.2 % acceptance, 2.745 tokens per
step, single-stream 56.5 -> **113.8 tok/s**.

**Prefill dispatch landed** — [docs/24](docs/24-p0-results.md). Row-count dispatch
*inside* the registered `vllm::exl3_gemm` op (`exl3_gemm` below 128 rows, reconstruct +
`hgemm` at or above it) took prefill from 2,369/2,362 to **5,050/5,146 tok/s** at 2k/6k
with decode unchanged. A Python `if` *around* the two calls does not work: vLLM resolves
the branch once at trace time and the decode graphs then capture reconstruct+hgemm,
collapsing C1 decode to 22.6 tok/s. The patch is not bit-exact (fp16 summation order):
it costs **+0.43 %** of measured divergence, disclosed, and
`VLLM_EXL3_PREFILL_RECONSTRUCT_M=0` restores the previous path.
[PR #316](https://github.com/local-inference-lab/vllm/pull/316).

**Prefill attributed, then bounded** — [docs/26](docs/26-prefill-attribution.md). Four
configurations over the two candidate causes: the **MLP kernel is the whole story**
(2.13x with the overlay on, 2.26x with attention BF16) and the **online-K6 attention
overlay is not the bottleneck** (1.05x / 1.11x) while saving 6.7 GB. That refutes the
hypothesis iteration 2 closed with. `ext.hgemm` is already at cuBLAS parity
(0.92-1.06x), and even a perfect fused reconstruct lands at 7-8k tok/s end to end
against FP8's 10,667. The goal is therefore re-scoped on evidence
([docs/25](docs/25-goal-pareto-dominate-fp8.md)): dominate FP8 on fidelity, memory, decode and
speculative decode — done and measured — and publish prefill as a structural deficit of a
4-bit-class trellis format in this runtime.

**Graph-vs-eager parity retracted and re-measured** —
[docs/27](docs/27-graph-decode-drift-control.md). An outside qualification of PR #314 on an
RTX 5090 found graph decode internally deterministic but not bit-identical to eager. My
own "KLD 0.000000" receipt could not have contradicted that: `fidelity.py capture` takes
one prefill forward and `FULL_DECODE_ONLY` captures no prefill graph, so it compared two
runs of the same eager prefill. Withdrawn upstream. The replacement probe
(`tools/decode_parity.py`, 32 prompts x 32 greedy tokens) measures decode directly:
**24/32** exact sequences, mean |delta logprob| **0.0118** on the chosen token, each mode
internally deterministic 32/32. Unquantised BF16 on the same build drifts identically
(**24/32**, **0.0128**), so the drift is ambient to this build's graph decode path and is
not caused by the quantisation. Randomised autotune priming changed nothing (25/32, same
deltas to five digits), killing the zero-priming hypothesis.

**Attention-overlay width is a runtime knob, and it is measured.** Same suite, same
reference and comparator head, varying only `ONLINE_QUANT` bits: K6 **0.008157** at
20.32 GiB, K5 **0.012135** at 19.82 GiB, K4 **0.027530** at 19.05 GiB. Paired against K6,
K5 costs +0.003978 and K4 +0.019373, both 0/136. K5 is the interesting setting: still
below official FP8's 0.013126 while fitting native 262,144 context on a 32 GB card
(266,743 KV tokens verified against a 31.2 GiB budget).

**Offline attention beats the online overlay, slightly.** A "hydrated" build with the
same MLP recipe but attention **serialized at K6 by the converter** instead of encoded at
load measures **0.007406** — 0.000751 better than the published checkpoint (95 % CI
[-0.00098, -0.00057], 124/136 contexts) and 0.005719 better than FP8 (136/0). It costs
9 GB of download and nothing in VRAM, and removes the cold-start encode.

**fp32 replay was a negative result** — [docs/24](docs/24-p0-results.md). Storing hidden states
in fp32 moved the replay qualification only 6.54e-04 -> 6.25e-04 (-4.5 %), so operand
rounding is ~5 % of the floor and the rest is the implementation difference between vLLM's
logit path and our replay. Paired comparisons are unaffected (common-mode); only absolute
values below ~1e-3 stay unresolvable.

### Next

Ranked in [docs/23](docs/23-next-attack-list.md), with the re-scoped goal and its
per-axis verdict in [docs/25](docs/25-goal-pareto-dominate-fp8.md). The two open items
that are not kernel work: publish the K5/K6, attention-width and hydrated captures into
the fidelity dataset (it still carries iteration-1 captures only), and get task-level
retention evidence — neither the decode-drift probe nor the KLD suite says whether the
flipped near-ties cost anything downstream.
