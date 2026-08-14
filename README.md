# Qwen3.8-27B EXL3 mixed-precision quant (`malaiwah/Qwen3.8-27B-K4`)

Research materials and progress log for building a dense EXL3 quant of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) that is inspired by
NVIDIA's NVFP4 recipe for the previous generation, but keeps the
high-accuracy tensors **BF16 on disk** for **runtime K6 encoding** by the
Gilded Gnosis vLLM fork, and serializes everything else at **EXL3 K4**.

Design goal: NVFP4-class VRAM footprint, lower KLD.

| Doc | Contents |
|---|---|
| [docs/01-nvfp4-composition.md](docs/01-nvfp4-composition.md) | Measured tensor-level composition of `nvidia/Qwen3.6-27B-NVFP4` and `unsloth/Qwen3.8-27B-NVFP4`, plus the BF16 parameter census |
| [docs/02-recipe-k4.md](docs/02-recipe-k4.md) | The proposed recipe, footprint arithmetic, and the headroom exchange rates |
| [docs/03-gg-runtime-contract.md](docs/03-gg-runtime-contract.md) | What the Gilded Gnosis EXL3 loader requires, and what it does *not* support |
| [docs/04-exllamav3-toolchain.md](docs/04-exllamav3-toolchain.md) | exllamav3 conversion flags, the missing per-module override, and the splice route |
| [docs/05-kld-protocol.md](docs/05-kld-protocol.md) | Teacher-forced full-vocabulary KLD protocol to be reproduced |
| [docs/06-baseline-validation.md](docs/06-baseline-validation.md) | Running the official GG image with no container runtime, and the two proven baselines |
| [docs/07-serving-recommendations.md](docs/07-serving-recommendations.md) | Serving-guide differences across upstream / NVIDIA / Unsloth cards |
| [PROGRESS.md](PROGRESS.md) | Chronological work log |

Tooling in [tools/](tools/) is what produced the evidence: an unprivileged OCI
image puller, a proot-based runner for the image, and the BF16 attention splice.

## Status

- [x] Both upstream Qwen3.8-27B artifacts proven runnable under the official GG image
- [ ] K4 conversion of `Qwen/Qwen3.8-27B`
- [ ] BF16 attention splice + metadata regeneration
- [ ] Serve the mixed checkpoint under GG with `ONLINE_QUANT=exl3-b6`
- [ ] KLD vs BF16 teacher, against NVFP4 as control
- [ ] Publish to `malaiwah/Qwen3.8-27B-K4`
