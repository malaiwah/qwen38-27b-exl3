> **Status: facts under re-measurement (2026-08-14).** An independent review found
> that the first published comparison used evaluation prompts drawn from the
> quantizer's own calibration corpus. That measurement has been redone on a held-out
> corpus ([docs/18](docs/18-results-fidelity-v3.md)) and the numbers below are the
> held-out ones. A second iteration of the weights (gate K5 / up K5 / down K6) is in
> flight, so throughput, footprint and fidelity figures for the *current* published
> checkpoint are being re-measured and will be superseded. Items still open are
> tracked in [docs/21](docs/21-independent-review-response.md).

# Qwen3.8-27B EXL3 mixed-precision quants (`K4`, `EXL3-K5K6`)

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
| [docs/07-serving-recommendations.md](docs/07-serving-recommendations.md) | Serving-guide differences across upstream / NVIDIA / Unsloth cards, cross-checked against shipped configs |
| [docs/08-upstream-cards-digest.md](docs/08-upstream-cards-digest.md) | Per-card digest: declared recipe, benchmarks, harnesses, limitations |
| [docs/09-variant-publication.md](docs/09-variant-publication.md) | How iterative variants are published for independent re-measurement |
| [docs/14-fidelity-protocol-v2.md](docs/14-fidelity-protocol-v2.md) | Hidden-state replay protocol, supersedes the single-window KLD |
| [docs/15-results-fidelity-v2.md](docs/15-results-fidelity-v2.md) | **Headline results**: 151,478 positions, bootstrap CIs, 74/74 paired wins |
| [docs/11-kld-external-comparison.md](docs/11-kld-external-comparison.md) | Published KLD data for this family; why "FP8 = 0.5" is wrong |
| [docs/13-upstream-contributions.md](docs/13-upstream-contributions.md) | Upstream issue + verified PR |
| [docs/12-iteration-2-plan.md](docs/12-iteration-2-plan.md) | Where to invest next, ranked |
| [docs/10-results-iteration-1.md](docs/10-results-iteration-1.md) | Iteration 1: build, serve, KLD, and the upstream defect |
| [docs/23-next-attack-list.md](docs/23-next-attack-list.md) | **Ranked plan for iteration 3**, with evidence, cost and acceptance per item |
| [PROGRESS.md](PROGRESS.md) | Chronological work log |

Tooling in [tools/](tools/) is what produced the evidence: an unprivileged OCI
image puller, a proot-based runner for the image, and the BF16 attention splice.

## Status

- [x] Both upstream Qwen3.8-27B artifacts proven runnable under the official GG image
- [x] K4 conversion of `Qwen/Qwen3.8-27B`
- [x] BF16 attention splice + metadata regeneration
- [x] Serve the mixed checkpoint under GG with `ONLINE_QUANT=exl3-b6` (19.21 GB resident, vision verified)
- [x] KLD vs BF16 teacher, against NVFP4 and official FP8 controls (held-out, v3)
- [x] Publish to `malaiwah/Qwen3.8-27B-K4` + [fidelity dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3)
