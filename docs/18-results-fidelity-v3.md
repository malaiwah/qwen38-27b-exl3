# v3 results: held-out corpus, and what changed

The v2 numbers were measured on a suite built from exllamav3's bundled calibration
corpora — **the same text our K4 conversion was calibrated on**, while NVFP4 and FP8
were calibrated elsewhere. That is train-on-test and it flattered us. v3 re-measured
everything on separately sourced text.

**2026-08-15 correction:** the original fixed-stride character scan's “0 hits” result was
offset-sensitive. Scanning every normalized 12-token position found exact overlap in 2/41
source documents. Excluding all nine analysis contexts from those documents leaves 127:
K4 0.029679, FP8 0.012798, NVFP4 0.092727, and K5/K6 0.007945. The original 136-context
receipt below is retained for traceability; no ranking or conclusion changes.

Suite: 181 contexts (136 analysis / 45 qualification, 32 sentinels), 2048 tokens each,
5 strata, 41 real source clusters. Suite token SHA-256
`3f9d17f1b55f6487...fe735691`.

## Analysis partition — 136 contexts, 278,392 scored positions

| candidate | weights | mean KLD | bootstrap 95 % CI | median | p99.9 | JSD (bits) | top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | 30.9 GB | **0.013126** | [0.00981, 0.01709] | 0.002343 | 0.773 | 0.004528 | 96.22 % |
| **this quant** (K4 + online K6) | 19.2 GB | **0.030736** | [0.02238, 0.04073] | 0.004218 | 1.758 | 0.010051 | 94.50 % |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | **0.094978** | [0.06858, 0.12688] | 0.012911 | 4.509 | 0.028663 | 90.53 % |

Paired over the same contexts:

| comparison | mean difference | bootstrap 95 % CI | wins |
|---|---:|---:|---|
| ours - NVFP4 | **-0.064242** | [-0.08621, -0.04611] | **136/136** ours |
| ours - FP8 | **+0.017611** | [0.01256, 0.02368] | 136/136 FP8 |

## What moving to held-out text did to each candidate

| candidate | v2 (contaminated) | v3 (held out) | change |
|---|---:|---:|---:|
| ours | 0.026231 | 0.030736 | **+17 %** |
| NVFP4 | 0.073006 | 0.094978 | +30 % |
| FP8 | 0.019309 | 0.013126 | **-32 %** |

Our number got **worse** on text we were not calibrated on, FP8's got **better**, and
NVFP4's got worse by more than ours. Net effect on the two comparisons:

- advantage over NVFP4 grew from 2.78x to **3.09x**;
- deficit against FP8 grew from 0.0069 to **0.0176**, i.e. 2.5x larger.

The skepticism that prompted this re-measurement was justified: some of the v2
advantage was calibration overlap, not quantization quality.

## Per-stratum means

| stratum | contexts | ours | NVFP4 | FP8 |
|---|---:|---:|---:|---:|
| code | 36 | 0.03437 | 0.09458 | 0.01387 |
| encyclopedic | 13 | 0.01704 | 0.05798 | 0.00879 |
| literary | 41 | 0.05739 | 0.18945 | 0.02411 |
| multilingual | 7 | 0.00840 | 0.02269 | 0.00406 |
| scientific | 39 | 0.00794 | 0.02134 | 0.00396 |

`literary` is the hardest stratum for every candidate and `scientific`/`multilingual`
the easiest — a 7x spread within one candidate, which is why a single-corpus number
was never going to be trustworthy.

## Controls

| control | result | interpretation |
|---|---|---|
| runtime-repeat noise floor, captures 1-2 and 2-3 over 32 sentinels ([`receipts/v3-noise-r1-vs-r2.json`](../receipts/v3-noise-r1-vs-r2.json), [`receipts/v3-noise-r2-vs-r3.json`](../receipts/v3-noise-r2-vs-r3.json)) | **0.000000**, top-1 1.0 | this runtime is bit-deterministic across process restarts; every difference reported above is far outside noise. The reference protocol's TP16 runtime had a floor of 0.0032, which would have swamped our head-attribution result |
| harness self-check | 0.000000 | validates scoring arithmetic and identical-input handling; it cannot detect capture, densification, or window defects shared by both sides |
| ~~CUDA-graph parity~~ **withdrawn** | published as 0.000000, top-1 1.0 | not a parity measurement: `fidelity.py capture` takes one **prefill** forward, and `cudagraph_mode=FULL_DECODE_ONLY` captures no prefill graph, so this compared two runs of the same eager prefill. The narrow claim it supports is "enabling graph decode does not change prefill numerics". Replaced by a real decode probe — [27](27-graph-decode-drift-control.md) |
| **replay qualification** | mean `KL(live \|\| replayed)` = **6.54e-04**, top-1 98.999 %, max 0.0913 | **this is our weakest link**: ~500x worse than the reference protocol's 1.23e-06 |

### The retracted graph-parity control

The row above was published with [PR #314](https://github.com/local-inference-lab/vllm/pull/314)
as evidence that graph decode is distribution-exact. It could not have measured decode at
all. The replacement harness (`tools/decode_parity.py`, 32 prompts x 32 greedy tokens
through the OpenAI endpoint) measures the decode path directly: graph and eager agree on
**24/32** exact sequences with mean `|delta logprob|` **0.0118** on the chosen token, and
each mode is internally deterministic (32/32 self-repeat). Unquantised BF16 on the same
build drifts by the same amount (**24/32**, **0.0128**), so the drift belongs to this
build's graph decode path, not to the quantisation. Full result in
[27](27-graph-decode-drift-control.md).

### The replay-qualification caveat, in proportion

6.54e-04 is **2.1 %** of our own candidate's KLD and
**3.7 %** of the K4-vs-FP8 gap, so no ranking here depends on it.
It bounds absolute live-versus-replay agreement at roughly 1e-3. The isolated
head-only absolute KLD is below that level, but the paired body-only versus
end-to-end head comparison shares the identical replay path and is a separate
relative measurement; [24](24-p0-results.md) reran that comparison with fp32
captures.

Two candidate causes were proposed here: BF16 storage of the hidden states (the operand is
rounded before replay), and a different logit path on the live side (vLLM's own head kernel
and dtype versus our chunked bf16 matmul). Both were then measured — see
[24](24-p0-results.md). Storing fp32 hidden states moved the qualification only from
6.54e-04 to 6.25e-04 (−4.5 %), so operand rounding accounts for ~5 % of the floor and the
rest is the implementation difference between the two logit paths. Paired comparisons are
unaffected because both arms use the identical replay path; only *absolute* values below
~1e-3 remain unresolvable.
