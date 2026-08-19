# Qwopus3.6-27B: what the Discord "best at 6bpw" claim actually is, and why it cannot be compared to our KLD

**Date:** 2026-08-19 (goal session). Prompted by a Discord relay: *"so far
Jackrong/Qwopus3.6-27B-Fusion at 6bpw was the best, wonder how 3.8 is against
it."* No GPU used; card/config inspection only.

## First: the reference in the message does not exist as stated

- `Jackrong/Qwopus3.6-27B-Fusion` -> **404**. It is not a Jackrong repo.
- The Fusion artifact is **`KyleHessling1/Qwopus3.6-27B-Fusion-GGUF`** — GGUF
  `Q4_K_M` (~16.5 GB), tagged `merge, task-vector, dare-ties, layer-weighted,
  experimental`, base `Qwen/Qwen3.6-27B`.
- The "**6bpw**" almost certainly refers to
  **`UnstableLlama/Qwopus3.6-27B-v2-exl3-6.00bpw`** — an EXL3 quant (our format)
  of the *reasoning parent*, not of the Fusion merge. UnstableLlama also ships
  2.50/2.90/3.08/4.15/8.00 bpw rungs.

"Qwopus" is Jackrong's Qwen x Claude-Opus-reasoning-distilled line (their
`Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` has 2,943 likes). So the thing
being praised is **a different model**, not a different quantization of ours.

## How it was done (the actually-interesting part)

The Fusion card states its merge math outright:

| | |
|---|---|
| anchor | `Qwen/Qwen3.6-27B` (both parents descend from it) |
| parent A (reasoning) | `Qwopus3.6-27B-v2` — merge base + embeddings/head/norms/MTP |
| parent B (code) | `Qwopus3.6-27B-Coder` — delta source |
| merge | `W(L) = V2 + alpha(L) * (Coder - V2)` |
| **alpha** | **linear ramp 0.12 (early) -> 0.48 (late)** |
| frozen from A | `embed_tokens`, `lm_head`, all `norm`, `mtp`/NextN, vision tensors |

Reported (Q4_K_M, thinking-on): HumanEval 94.5%, MBPP 87.9%, GSM8K 95.0%,
MATH-500 62-64.7%, IFEval strict 82.8%, SWE-bench Verified 7/15 on an astropy
slice — beating equal-weight DARE-TIES (5/15) and a DUS-38B frankenmerge (3/15)
on that slice. The card labels itself a research preview that "has **not** yet
been through a full rigorous evaluation".

## Why "how is 3.8 against it" has no answer today

There is **no shared axis**. Four independent mismatches:

1. **Different base.** Qwen3.6-27B vs our Qwen3.8-27B.
2. **Different bit budget.** 6.00 bpw (or GGUF Q4_K_M for Fusion) vs our 5.531
   average body bits.
3. **Different format family** for the Fusion artifact (GGUF/llama.cpp) vs EXL3.
4. **Fatally: different metric, and ours is inapplicable to theirs.** Our KLD
   measures divergence from *our own BF16 reference* — how faithfully the quant
   reproduces the base. A merge of two fine-tunes is *intentionally different*
   from its base; scoring `KLD(Fusion || Qwen3.6-27B-BF16)` would count the
   merge's entire purpose as error. Their "best" is a **task-quality** claim
   (HumanEval/MBPP/GSM8K/SWE-bench); we have never run task benchmarks on our
   checkpoint, and they publish no KLD.

So the two "best" claims are orthogonal: ours is *fidelity at low bits to a fixed
base*; theirs is *task quality of a deliberately modified model at higher bits*.
Neither number bounds the other.

## The transferable lever, and its honest price

Their result is an argument for an **orthogonal axis we have not touched: improve
the weights, not the encoding.** Nothing stops us quantizing a distilled/merged
27B with our EXL3 pipeline and serving it on our stack — our serving wins (the K5
cure, fold-48, recon->hgemm, 600 W, MTP-6) are base-agnostic.

The price is specific and should not be glossed: **it invalidates our quality
gate.** Every criterion we defend is KLD-to-base; for a merged base that metric
stops meaning "quality" and starts meaning "distance from a model we do not want
to reproduce". Adopting this lever means acquiring a task-benchmark harness
(HumanEval/MBPP/GSM8K/IFEval at minimum) that this project does not have, and
re-deriving what "fidelity" means for a checkpoint whose base is itself a merge.
That is a larger methodological change than any quantization work in the queue.

## One suggestive resonance with our depth question — flagged, not claimed

Their merge injects the code delta with **depth-increasing weight** (alpha 0.12
early -> 0.48 late): early layers are perturbed *least*, late layers most, and
embeddings/norms/head/MTP are frozen entirely. Independently, on the same day, our
allocator sweep (`receipts/eda-depth-weighting-2026-08-19.md`) measured that the
published depth-blind bit objectives do the **opposite** — they strip bytes from
the first 16 layers (`abs` -779 MB) and hand them to the last 16 (+786 MB).

Two unrelated practices, one direction: **treat early layers gently.** This is
suggestive only. Their alpha ramp is a merge coefficient, ours is a bit
allocation; there is no shared metric, and their card offers no ablation
justifying the ramp's shape. It does not calibrate our `depth_amp` — only the
matched-byte early-vs-late layer-range KLD experiment already queued can do that.
