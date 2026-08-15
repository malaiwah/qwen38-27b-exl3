> **Status: iteration 2 published (2026-08-15).** The current checkpoint is
> [`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6):
> mean KLD **0.008157** body-only, **0.008284** as served, at **20.32 GiB** resident
> weights on the held-out v3 suite — 38 % lower divergence than official FP8 at 71 % of
> its weight ([docs/22](docs/22-results-iteration-2.md)). Figures labelled K4 or v1 belong
> to iteration 1 and are superseded. One earlier control was **withdrawn**: the
> "CUDA-graph parity 0.000000" receipt captured a *prefill* forward, which
> `FULL_DECODE_ONLY` never captures, so it could not have measured decode; the decode
> probe that replaces it is [docs/27](docs/27-graph-decode-drift-control.md). Open items
> are tracked in [docs/21](docs/21-independent-review-response.md) and
> [docs/23](docs/23-next-attack-list.md).

# Qwen3.8-27B EXL3 mixed-precision quants (`K4`, `EXL3-K5K6`)

Research materials and progress log for building a dense EXL3 quant of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) that is inspired by
NVIDIA's NVFP4 recipe for the previous generation, but keeps the attention projections
**BF16 on disk** for **runtime K6 encoding** by the Gilded Gnosis vLLM fork, and
serializes the MLP as EXL3 **K5 gate / K5 up / K6 down** with a K6 `mcg` head and a
quantized MTP draft. Iteration 1 serialized the whole MLP at **K4**.

Design goal: NVFP4-class VRAM footprint, lower KLD. Met on fidelity, memory, decode and
speculative decode; prefill is a measured structural deficit of a 4-bit-class trellis
format in this runtime ([docs/25](docs/25-goal-pareto-dominate-fp8.md),
[docs/26](docs/26-prefill-attribution.md)).

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
| [docs/18-results-fidelity-v3.md](docs/18-results-fidelity-v3.md) | Held-out re-measurement: 181 contexts, contamination scan, per-stratum means, and the controls — including the one that was withdrawn |
| [docs/22-results-iteration-2.md](docs/22-results-iteration-2.md) | **Headline results**: gate K5 / up K5 / down K6, mean KLD 0.008157, 38 % below official FP8 at 71 % of its resident weight |
| [docs/15-results-fidelity-v2.md](docs/15-results-fidelity-v2.md) | Superseded v2 run: 151,478 positions, 74/74 paired wins — measured on a corpus that overlapped our calibration data |
| [docs/16-head-attribution.md](docs/16-head-attribution.md) | Is `lm_head` the sensitive tensor? Measured: not at K6 |
| [docs/11-kld-external-comparison.md](docs/11-kld-external-comparison.md) | Published KLD data for this family; why "FP8 = 0.5" is wrong |
| [docs/13-upstream-contributions.md](docs/13-upstream-contributions.md) | Upstream issue + verified PR |
| [docs/19-cuda-graphs-patch.md](docs/19-cuda-graphs-patch.md) | The autotune-priming patch behind PR #314, deliverables and verification |
| [docs/20-context-extension-and-k3-gap.md](docs/20-context-extension-and-k3-gap.md) | How far context can be extended, and the distance to the reference protocol |
| [docs/12-iteration-2-plan.md](docs/12-iteration-2-plan.md) | Where to invest next, ranked |
| [docs/17-next-iteration-shopping-list.md](docs/17-next-iteration-shopping-list.md) | Iteration-2 shopping list, each item with its closing test |
| [docs/10-results-iteration-1.md](docs/10-results-iteration-1.md) | Iteration 1: build, serve, KLD, and the upstream defect |
| [docs/21-independent-review-response.md](docs/21-independent-review-response.md) | Independent review, per finding: fixed, fixed-in-v2, or still open |
| [docs/24-p0-results.md](docs/24-p0-results.md) | **P0 done**: prefill +113 %, and why fp32 replay was a negative result |
| [docs/23-next-attack-list.md](docs/23-next-attack-list.md) | **Ranked plan for iteration 3**, with evidence, cost and acceptance per item |
| [docs/25-goal-pareto-dominate-fp8.md](docs/25-goal-pareto-dominate-fp8.md) | the goal as a gate, and the iteration-3 verdict per axis |
| [docs/26-prefill-attribution.md](docs/26-prefill-attribution.md) | where prefill time goes: MLP 2.13-2.26x, attention overlay 1.05-1.11x, hgemm at cuBLAS parity |
| [docs/27-graph-decode-drift-control.md](docs/27-graph-decode-drift-control.md) | eager-vs-graph drift is ambient: BF16 drifts the same as EXL3 |
| [PROGRESS.md](PROGRESS.md) | Chronological work log |

Tooling in [tools/](tools/) is what produced the evidence: an unprivileged OCI image
puller and a proot-based runner for it, the BF16 attention splice and checkpoint
finaliser, the fidelity harness (`fidelity.py`, `suite3.py`) that builds the suite and
replays captures, the decode-parity probe (`decode_parity.py`), the kernel
microbenchmarks (`prefill_micro*.py`, `gemm_cmp.py`) and the upstream patches as
standalone files.

## Status

- [x] Both upstream Qwen3.8-27B artifacts proven runnable under the official GG image
- [x] Iteration 1 (K4 MLP + BF16 attention splice) served under GG with `ONLINE_QUANT=exl3-b6`, 19.21 GB resident, vision verified, published as [`malaiwah/Qwen3.8-27B-K4`](https://huggingface.co/malaiwah/Qwen3.8-27B-K4)
- [x] Held-out v3 fidelity suite with a contamination scan, plus the [dataset](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3) that lets anyone recompute it without a GPU
- [x] Iteration 2 built, measured and published as [`malaiwah/Qwen3.8-27B-EXL3-K5K6`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6): **20.32 GiB** resident, mean KLD **0.008157** body-only / **0.008284** as served, top-1 96.97 %
- [x] CUDA graphs ([PR #314](https://github.com/local-inference-lab/vllm/pull/314)) and row-count prefill dispatch ([PR #316](https://github.com/local-inference-lab/vllm/pull/316)) landed as fork PRs: decode 56.6 / 199.6 / 404.6 tok/s, **113.8** with MTP-3, prefill 2,369 -> **5,050** tok/s at 2k
- [x] Prefill attributed and bounded: the MLP kernel is the whole story (2.13x / 2.26x), the online-K6 attention overlay is not the bottleneck (1.05x / 1.11x), `ext.hgemm` is already at cuBLAS parity — FP8 prefill parity needs a fused dequant-in-epilogue kernel, not tuning
- [x] Attention-overlay width is a runtime knob, measured on the same suite: K6 **0.008157** at 20.32 GiB, K5 **0.012135** at 19.82 GiB (still below FP8, and reaches native 262,144 context on a 32 GB card), K4 **0.027530** at 19.05 GiB
- [x] Serializing attention offline instead of encoding it at load — the "hydrated" build — measures **0.007406**, i.e. calibrated offline K6 beats the runtime overlay by 0.000751 (124/136 contexts)
- [x] Graph-vs-eager decode drift measured properly and traced to the build, not to EXL3 ([docs/27](docs/27-graph-decode-drift-control.md))
- [ ] K5/K6, attention-width and hydrated captures published into the fidelity dataset — the dataset currently carries iteration-1 captures only
- [ ] Task-level retention evidence: nothing here shows whether the graph-decode near-tie flips, or the quantization itself, cost downstream accuracy
