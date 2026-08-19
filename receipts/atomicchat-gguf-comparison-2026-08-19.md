# Comparison: AtomicChat/Qwen3.8-27B-GGUF (imatrix GGUF ladder)

**Date:** 2026-08-19. Source: HF card + file manifest, read in full.
122K downloads, 124 likes — the mainstream 4-bit comparison for this model.

The most useful third-party artifact we have found: it is **4-bit class at our
exact size range**, measured **against real BF16 weights** (not against `Q8_0`),
and benchmarked on **RTX 5090**. Their calibration corpora are public
(`AtomicChat/calib-corpora`) and their raw logs are in a companion metrics repo.

## Their ladder

16 files, 8.5 GB (`AD-IQ1_M`) to 28.9 GB (`Q8_0`), plus a shared ~0.93 GB
`mmproj` vision projector. MTP head is inside every file, pinned `q5_k`.
`AD-` = "Atomic Dynamic" layout, named `AD-<ffn_down>-<ffn_up>`.

| File | Size | KL divergence | top-1 |
|---|---:|---:|---:|
| `Q8_0` | 28.9 GB | 0.00064 | 98.92% |
| `AD-Q6_K` | 25.0 GB | 0.00107 | 98.67% |
| `AD-Q6_K-Q5_K` | 23.1 GB | 0.00252 | 97.94% |
| `AD-Q5_K_M` | 20.2 GB | 0.00419 | 97.34% |
| `AD-Q5_K_M-Q4_K_M` | 18.6 GB | 0.00730 | 96.43% |
| `AD-Q4_K_M` | 17.1 GB | 0.01126 | 95.59% |
| `AD-IQ4_XS` | 16.5 GB | 0.01248 | 95.39% |

**No shared artifact exists between their measurements and ours**, so unlike the
lribeiro FP8 comparison there is *no anchor* and their KLD column cannot be
rescaled onto our axis. Our trellis payload is 16.82 GiB (18.06 GB) at
KLD 0.002700 on our v5 suite; their 18.6 GB file reads 0.00730 on theirs. Both
are ~4–5-bit-class at ~18 GB. **That is the honest extent of the comparison.**

## What they measured that we can check

### 1. Allocation beats bit-count — and they quantified it

Ten layouts at 16.8–18.6 GB, changing only *where* the bits go:

| layout | size | KLD |
|---|---:|---:|
| every layer the same | 16.8 GB | 0.01580 |
| more bits on `ffn_down` everywhere | 17.8 GB | 0.01189 |
| 16 layers lifted, first and last | 17.8 GB | 0.00981 |
| 24 layers lifted | 18.6 GB | 0.00743 |
| **24 lifted + attention gate + state output** | **18.6 GB** | **0.00730** |

2.2x KLD range at ~the same file size. Independent validation of the EDA premise.

### 2. Their depth finding CORROBORATES our depth calibration

They report the imatrix's "highest activation energy in the whole model sits on
**layers 52 to 62**, with a second peak on layer 0", and that lifting the first
4 + last 12 layers helped more than anything else; widening to 32 layers did not
help further.

Our measured banded-FP6 KLD (`receipts/eda-depth-calibration-2026-08-19.md`):
early L0-12 **0.003600**, mid L26-38 **0.003838**, late L51-63 **0.004395**,
early-vs-late CIs disjoint. **Monotone late-heavy — their energy peak at L52-62
is our most-sensitive band, measured by a completely different method.**

Note the irony recorded in `docs/58`: llama.cpp's *own* built-in heuristic
`use_more_bits` is U-shaped (first n/8, last n/8, every 3rd middle), and
exllamav3's `allocation.py` is U-shaped too. AtomicChat's *measurement* of this
model agrees with our measurement and not with the heuristic either toolchain
ships. Their "widening to 32 layers did not help" also matches our finding that
the `--max-width 6` cap is nearly free (93.9% agreement).

### 3. Their "single best trade" — REPLICATED on our ladder, then REFINED

They report: the attention gate and state output path are "5.5% of the weights
each", and giving both one extra step of precision cost 0.16 GB and removed
**11% of the remaining divergence** — "the single best trade we found".

Tested against our own 409-module error-driven ladder (widths 3–7, measured):

| our result | value | theirs |
|---|---:|---|
| weight share of `in_proj_z` + `out_proj` | **12.4%** | 11% |
| objective error removed by +1 step on both | **16.3%** | 11% |

Same phenomenon, same direction, slightly stronger in our data. **But our
per-class marginal returns localise it differently:**

| class | width | Δerr/GB for +1 step | share of err | share of weights |
|---|---:|---:|---:|---:|
| MLP gate/up | 5 | **0.0255** | 35.0% | 46.9% |
| **GDN state-out (`out_proj`)** | 6 | **0.0249** | **16.8%** | **6.2%** |
| self_attn | 6 | 0.0121 | 9.7% | 6.9% |
| MLP down | 6 | 0.0101 | 26.1% | 23.4% |
| GDN gate (`in_proj_z`) | 6 | 0.0084 | 5.7% | 6.2% |
| GDN qkv | 6 | 0.0059 | 6.7% | 10.3% |

**The win is `out_proj`, not the gate.** `out_proj` carries 16.8% of the
objective error on 6.2% of the weights — 2.7x over-represented. The gate is a
*poor* trade at 0.0084/GB, below self_attn and MLP down.

**Our K5K6 layout is already near-optimal by the equal-marginal-return test:**
the two best up-trades are MLP gate/up at 0.0255/GB and `out_proj` at
0.0249/GB — within 2.4% of each other, which is what an optimal allocation
looks like. Exhaustive budget-neutral search over class pairs finds exactly one
improvement: **+1 step on `out_proj`, paid by -1 step on 60% of GDN qkv, net
-0.000507** on a 0.038471 total — a **1.3%** objective-error reduction. That is
well inside the 31–40% of measured KLD our objective already fails to model, so
it does **not** justify a rebuild on its own.

### 4. A lever we do not have — and it is worth 1.78 GiB

They found: "the embedding table and the output head weigh the same, 4.7% each,
and behave nothing alike. The head decides the next word directly and has to
stay precise. The embedding table is a lookup whose error stays inside one
token, so it can be cut hard. Paying for the head out of the embedding table is
a net gain at every size we tested."

**Our checkpoint has this exactly backwards.** Verified in the hydrated
snapshot: `lm_head` **is** trellis-quantized (`lm_head.trellis`), while
`model.language_model.embed_tokens.weight` is **[248320, 5120] BF16 = 2.368
GiB**, untouched, and absent from the 409-module ladder entirely.

At the measured 40.7 KB/token KV cost, freeing that space buys context:

| action | frees | extra context |
|---|---:|---:|
| embed_tokens BF16 → FP8 | ~1.18 GiB | ~30,500 tok |
| embed_tokens BF16 → 4-bit | ~1.78 GiB | ~45,800 tok |

For the flagship (238,400 on 9.25 GiB KV) the 4-bit case would clear **native
262,144 with vision intact** — criterion 5 with margin. For the merged
MTP+vision checkpoint (capped at 49,152 by weight size) it is roughly +39%.
This is a new, unexplored lever on the critical path, and their measurement says
the fidelity cost is small. Registered as a todo; needs our own KLD measurement
before it is believed.

## Where they win, where we win

- **KV efficiency — the differentiator, quantified.** Their card states 256 KB
  of cache per token (2 GB at 8k, 8 GB at 32k) and their own guidance caps a
  32 GB card at "around 24k context". Ours is **40.7 KB/token** measured
  (9.25 GiB for 238,400) — **6.3x better** — because the 48 GDN layers hold
  recurrent state rather than KV. We serve 238,400–249,600 on 31.40 GiB.
- **Prefill, per GPU: parity to a win for us.** Their `AD-Q4_K_M` on **2x**
  RTX 5090 at 32k = 5,244 t/s (2,622/GPU). Our fidelity at 32,769 on **one**
  5090 = 2,538.6; our throughput = 5,935.0 on one GPU, beating their two.
  (Their 8k row reads 363 t/s prompt, which is anomalous against 5,244 at 32k
  and looks like a card typo; not used here.)
- **Decode: we win clearly.** Their generation is 77 t/s at 8k / 72 t/s at 32k
  with `--spec-type draft-mtp`. Ours: fox 228.8 at 2k, and 143.5 measured at
  131,073 on throughput. Different harnesses, but the gap is large.
- **They win on reach and honesty of presentation.** 16 files spanning 8.5–28.9
  GB, public calibration corpora, raw logs published, competitors re-measured
  in-house rather than quoting their numbers, and one case flagged where unsloth
  beats them. Also a genuinely useful warning: three publishers ship a file
  called `Q4_K_M` at 16.8/19.0/17.1 GB scoring 0.02094/0.01470/0.01126 — "a
  quant name tells you which recipe was requested, not what you are getting".
- **They win on portability.** llama.cpp, CPU offload, any GPU.

## Corroborations worth recording

- **`Q8_0` is not lossless** (0.00064, 98.92% top-1) and they measure against
  real BF16, warning that a `Q8_0` reference deflates every number. Our harness
  also references BF16 hidden states — same discipline.
- **MTP collects no calibration data**: "never executed during a normal forward
  pass, so the importance matrix has nothing to say about it at any corpus
  size"; they pin it to `q5_k`. This is the third independent project to hit MTP
  handling (lribeiro's `re:^mtp.*` ignore-list defect; our BF16 MTP re-merge),
  and it is the same class as our open suspicion that GDN calibration flows run
  with `cache=None` so `in_proj` Hessians see unrepresentative activations.
- **Superblock constraint**: their rows must divide by 256; ours must land on
  `n_words in (48,64,80,96)` for the b12x ANY_BITS fast path. Any `out_proj`
  lift to K7 would fall off that path for 48 matrices — a real cost their
  format does not have.
