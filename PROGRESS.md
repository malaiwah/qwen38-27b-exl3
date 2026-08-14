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

### Next

1. K4 conversion inside the image rootfs (its exllamav3 and prebuilt SM120
   `exl3_gemm` extension are the same ones the runtime loads).
2. Splice BF16 attention, regenerate index + `quantization_config.json`.
3. Serve with `--quantization exl3 --enforce-eager` and the Qwen `ignore` list.
4. KLD against the BF16 teacher, with the NVFP4 quant as control.
5. Publish `malaiwah/Qwen3.8-27B-K4`.
