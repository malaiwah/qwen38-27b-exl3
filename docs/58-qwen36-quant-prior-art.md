# 58 — Qwen 3.5/3.6 quantization prior art: how per-module bitrate attribution was decided, and what metric judged it

**Date:** 2026-08-19. **Scope:** read-only research into the prior art of
quantizing the Qwen 3.5/3.6 generation (hybrid Gated DeltaNet + periodic
full-attention, 64 layers), focused on two questions: (1) how did practitioners
decide per-module/per-layer bitrate attribution, and (2) what fidelity metric
judged it? No GPU used; every claim is from source code, model cards, or
community reports (labelled as such). Items not directly measured by us are
labelled `[INFERENCE]` or attributed to their source.

This document engages our own findings — the attribution physics (attention
error compounds via KV cache; MLP error is additive and nearly free), the
depth-weighting finding (depth-blind objectives strip early layers), and the
EDA solver's measured negative (the `rel` objective regressed because it moved
bits attention→MLP) — as falsification targets against the prior art.

---

## 1. llama.cpp / GGUF k-quant attribution heuristics

### 1.1 The `use_more_bits` layer-index heuristic

llama.cpp's `llama-quantize` tool selects per-tensor quantization types inside
`llama_model_quantize_internal` (`src/llama-quant.cpp`). The central
layer-dependent heuristic is a lambda at line 430–432:

```cpp
auto use_more_bits = [](int i_layer, int n_layers) -> bool {
    return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
};
```

**Exact behaviour and direction:** `use_more_bits` returns true for:
- the **first** `n_layers/8` layers (early layers),
- the **last** `n_layers/8` layers (late layers),
- and every 3rd layer in the middle band (`(i_layer - n_layers/8) % 3 == 2`).

This is a **U-shaped** depth heuristic: both ends of the stack get promoted to
higher bit widths, with a periodic sprinkling in between. For a 64-layer model,
layers 0–7 and 56–63 plus every 3rd middle layer (10, 13, 16, …) get more bits.

**How this tests our depth finding:** Our measured result
(`receipts/eda-depth-weighting-2026-08-19.md`) is that depth-blind objectives
strip bytes from the first 16 layers and hand them to the last 16 — the harmful
direction if error accumulates with depth. The `use_more_bits` heuristic
**partially agrees** with our finding: it protects the first 1/8 (layers 0–7),
which is the direction we want. But it **also** protects the last 1/8 (layers
56–63), which is the direction our depth-blind solver erroneously over-fed. If
our accumulation premise is correct (early layers matter more because their
error is amplified by more downstream layers), then the last-1/8 promotion is
wasted bytes — late-layer error has no downstream amplification. The heuristic
was not designed from a measured accumulation model; it is an empirical rule
from the llama.cpp community, and no published ablation justifies the U-shape
over a monotonic ramp.

### 1.2 Tensor-type promotion: attention over MLP

llama.cpp explicitly categorizes attention-v tensors as "more sensitive to
quantization" (line 153–157):

```cpp
// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}
```

For `Q4_K_M` and `Q5_K_M`, `attn_v` (and fused `attn_qkv`, `attn_kv_b`) tensors
are promoted to `Q6_K` when `use_more_bits(qs.i_attention_wv, qs.n_attention_wv)`
returns true (line 552–553):

```cpp
else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
        use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
```

For very low-bit ftypes (`IQ2_S`, `IQ2_M`, `IQ1_M`), `attn_v`-category tensors
are held at `Q4_K` or `IQ3_S` while the rest drops to 2-bit (line 506–510).

**FFN_DOWN** also gets early-layer special-casing. Across multiple ftypes, the
first 1/8 or 1/16 of `ffn_down` layers are promoted (line 591, 594, 597, 601,
610, 616, 620, 624). A comment at line 625 states the rationale:

```cpp
// Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
```

**Pattern summary:** llama.cpp's attribution favours **attention over MLP**
(matching our measured attention-dominance) and **early layers over late layers**
(matching our depth finding for the first 1/8, but not the last 1/8). The
pattern is: `attn_v` → Q6_K, `ffn_down` early → Q5_K/Q6_K, everything else →
the base ftype. There is no GDN/linear-attention-specific logic in llama.cpp's
quantization code — the tensor categories (`attn_v`, `attn_q`, `attn_k`, etc.)
are based on standard attention tensor names and do not include `ssm_*` or
`linear_attn.*` patterns. The Qwen3.5 architecture is handled in
`models/qwen35.cpp` but the quantization tensor selection in `llama-quant.cpp`
is architecture-agnostic except for MoE expert counting.

### 1.3 KLD computation in llama.cpp

`llama-perplexity --kl-divergence` computes KL divergence between the quantized
model's logits and a pre-saved BF16/FP16 reference. The reference logits are
saved as uint16-quantized log-probabilities (line 191–234 of
`tools/perplexity/perplexity.cpp`). The KLD per token is computed as:

```cpp
sum += p_base * (p_log_base - logits[i] + max_logit);
```

This is **KL(reference || quantized)** — `p_base` (the reference probability)
weights the log-ratio. The uncertainty on the mean KLD is calculated by assuming
the per-token KLD follows a Gaussian distribution
(`tools/perplexity/README.md` line 23). The default evaluation protocol is
Wikitext-2 at 512 context.

**Protocol caveats for community KLD numbers:** The `llama-perplexity` KLD is
computed over Wikitext-2 tokens at 512 context, which is a different protocol
from our 512-context v5 suite (different corpus, different positions, different
reference model). Cross-protocol KLD comparisons are invalid for absolute values;
only within-protocol orderings are evidence. Our project treats cross-protocol
KLD comparisons as invalid (`receipts/discord-leaderboard-2026-08-19.md`,
Caveats; `receipts/qwopus-comparison-2026-08-19.md`).

### 1.4 Community bpw↔KLD data for Qwen 3.5/3.6

**Unsloth** published the most extensive KLD table for Qwen3.5-27B (and the MoE
variants), with ~20 entries across quant levels and uploaders
(https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks). Protocol: wiki.test.raw,
512 context, `llama-perplexity --kl-divergence`. Selected rows (community report,
unknown protocol relative to our suite):

| Quant | Disk (GB) | PPL | KLD 99.9% | Mean KLD |
|---|---|---|---|---|
| Unsloth UD-Q4_K_XL | 19.17 | 6.5918 | 0.4097 | 0.0137 |
| Unsloth Q4_K_M | 18.49 | 6.6053 | 0.5478 | 0.0192 |
| AesSedai Q4_K_M | 20.62 | 6.5665 | 0.3171 | 0.0096 |
| Unsloth Q5_K_XL | 23.22 | 6.5489 | 0.236 | 0.0069 |
| Unsloth Q6_K_XL | 28.22 | 6.5392 | 0.1437 | 0.0041 |

These are **community reports of unknown protocol** — the absolute KLD values
are not comparable to our suite; only the ordering within their table is
evidence. The table shows the expected monotone KLD decrease with increasing
bpw, and the "Unsloth Dynamic" (UD) quants consistently beat standard K-quants
at similar sizes — which Unsloth attributes to their tensor-sensitivity-aware
attribution (see §3).

**localbench.substack.com** benchmarked 87 GGUF quants of Qwen3.6-27B against
BF16, using ~250,000 tokens across 6 categories, top-40 log-probability
distributions
(https://localbench.substack.com/p/qwen-3-6-27b-gguf-quality-benchmark).
Protocol: TextGen + patched llama.cpp, chat-template-rendered prompts. Again,
**community report, unknown protocol**. Key finding: "Long documents dominate
the quality loss" and "ubergarm's ik_llama.cpp quants earn both their frontier
spots."

**sokann/Qwen3.6-27B-GGUF-4.256bpw** published a detailed KLD table
(https://huggingface.co/sokann/Qwen3.6-27B-GGUF-4.256bpw) using
`llama-perplexity` on wiki.test.raw, 580 chunks:

| Quant | BPW | Mean KLD | 99.9% KLD | Max KLD |
|---|---|---|---|---|
| sokann 4.256bpw | 4.256 | 0.033452 | 2.907350 | 23.255 |
| bartowski Q3_K_M | 4.270 | 0.058818 | 3.986622 | 24.616 |
| unsloth UD-Q3_K_XL | 4.302 | 0.046348 | 3.614290 | 24.175 |
| bartowski IQ4_XS | 4.556 | 0.026270 | 2.385293 | 22.992 |
| unsloth IQ4_XS | 4.589 | 0.024728 | 2.201674 | 21.687 |

**No published EXL3 bpw↔KLD table for Qwen3.6-27B was found.** The
UnstableLlama EXL3 rungs (`Qwopus3.6-27B-v2-exl3-{2.50,2.90,3.08,4.15,6.00,8.00}bpw`
and `Qwen3.6-27B-exl3-{2.06,6.00,8.00}bpw`) carry only bpw labels and no
measured fidelity numbers on their model cards. Our project's own
`Qwen3.8-27B-EXL3-K6-parity` is the only EXL3 checkpoint with a published
bpw↔KLD pair (0.001634 at 6.0-class bpw, our suite).

---

## 2. exllamav2/v3 measurement-based allocation

### 2.1 EXL2: simulated annealing on relative Frobenius-norm accuracy

EXL2's conversion pipeline (`exllamav2/conversion/`) runs a measurement pass
(`measure.py`) that tests each module group (attention block or MLP block) at
multiple quantization options, then an optimizer (`optimize.py`) that allocates
bits per module to hit a target bpw.

**Measurement objective** (`measure.py`, `test_error` function): for each
quantization option, the module is run forward on calibration hidden states and
the output is compared to the BF16 reference output:

```python
rfn_sum += torch.linalg.norm(xtest - xref, 'fro') / torch.linalg.norm(xref, 'fro')
```

The "accuracy" is `1 - mean(relative Frobenius norm)` — i.e., `1 - ||Q(x) - x||_F / ||x||_F`.
This is a **per-module-group relative output error**, not a KLD. It is measured
on the module's output (the residual-stream contribution), not on logits.

**Optimizer** (`optimize.py`): uses `ext_c.sim_anneal` (a C extension) to
minimize a configurable norm of the error vector over module groups, subject to
a total-bit budget (`weight_budget = int(numel * target_bpw)`). The norm is a
parameter swept during annealing (`norm_interval = (1.5, 3.5)`, three annealing
stages). The final objective is `sum(log(err))` and `max(err)` (line 140–143).

**EXL2's allocation objective in one sentence:** minimize a configurable norm of
the per-module-group relative-Frobenius-norm output error, subject to a total
bit budget, via simulated annealing — **no layer-position term, no KLD, no
Hessian weighting of the objective** (the Hessian is used inside GPTQ
quantization via `AdaptiveGPTQ`, but the allocation optimizer itself is
error-norm-based, not Hessian-weighted).

### 2.2 EXL3: direct KLD optimization with a layer-position heuristic

EXL3 (`turboderp-org/exllamav3`) replaces EXL2's Frobenius-norm with **direct
KLD** as the allocation metric. The measurement pass
(`exllamav3/conversion/measure_model.py`) computes KLD between quantized and
reference model outputs:

```python
def kldiv(s, ref):
    ...
    kld += F.kl_div(torch.log(s_probs + 1e-10), ref_probs, reduction = "sum").item()
    return kld / bsz
```

This is **KL(reference || quantized)** — `F.kl_div(input, target)` computes
`target * (log(target) - input)`, so `ref_probs` weights the log-ratio. This is
the same direction as our suite and as llama.cpp's `--kl-divergence`.

The optimizer (`exllamav3/conversion/optimize_model.py`) uses a greedy
marginal-improvement algorithm over groups, with a power-law adjustment on
negative dkld:

```python
def adjust(dkld):
    if dkld > 0:
        return dkld
    return -((-dkld) ** 0.69)
```

The `0.69` exponent compresses negative KLD deltas (improvements) less than
linearly, which biases the greedy search toward improvements that reduce KLD
rather than just fitting more bits.

**Layer-position heuristic in EXL3's default strategy** (`exllamav3/conversion/allocation.py`,
`create_q_strategy` function): when no measurement file is provided, the
default bitrate assignment uses an explicit layer-position heuristic. The
comment at line 63 states:

```python
# Promotion order: group priority first, then distance to the nearer end of the forward pass.
# End layers contribute disproportionately to end-to-end error
```

The implementation computes `dist = min(layer, stack_max[stack] - layer)` —
layers closer to either end get promoted first. This is a **U-shaped heuristic**
like llama.cpp's `use_more_bits`: both ends of the stack get priority for bit
promotion. The rationale ("End layers contribute disproportionately") is stated
but not measured in the source; it is a design assumption, not an empirical
result.

**Comparison to our EDA solver:** Our EDA solver's `rel` objective
(`sum_m eps(m,K)`, unweighted relative proxy error) is **depth-blind** — it has
no layer-position term at all. EXL3's default strategy does have a layer term
(U-shaped), and EXL3's optimizer uses direct KLD rather than a proxy. Our
measured negative (the `rel` objective regressed because it moved bits
attention→MLP) is therefore a stronger result than "the objective was
depth-blind": even with a proxy that captures per-module error, the absence of
a compounding model (KV-cache propagation) caused a sign error in the
between-role allocation. EXL3's direct-KLD optimizer would not make this
specific sign error because it measures end-to-end KLD, but it requires a
measurement pass (GPU time) that our proxy-based solver avoids.

### 2.3 Published EXL2/EXL3 quality curves for Qwen 3.5/3.6 27B

**No published EXL2/EXL3 bpw↔quality curves with measured fidelity numbers were
found for Qwen 3.5/3.6 27B.** The UnstableLlama EXL3 rungs
(`UnstableLlama/Qwen3.6-27B-exl3-{2.06,6.00,8.00}bpw` and
`UnstableLlama/Qwopus3.6-27B-v2-exl3-{2.50,2.90,3.08,4.15,6.00,8.00}bpw`) carry
only bpw labels and conversion metadata on their model cards — no KLD, no
perplexity, no task-quality numbers. This is a gap in the public record: the
EXL3 format's Pareto curve for this architecture family has not been published
by anyone except our project.

---

## 3. Hybrid/GDN-specific precision practice

### 3.1 GDN/linear-attention at higher precision is standard practice in NVFP4

Qwen3.5/3.6 is a hybrid architecture: 48 Gated DeltaNet (linear attention)
layers + 16 periodic full-attention layers, 64 total. The linear-attention
block has fused projection tensors (`in_proj_qkvz`, `in_proj_ba`, `out_proj`)
that do not match the split-tensor names vLLM's weight loader expects. This
creates a **correctness trap**: if these fused names are not in the
`quantization_config.ignore` list, they get NVFP4-quantized, the weight loader
cannot find the packed weights, silently skips loading, and the model produces
garbage.

**vLLM bug #40252** (https://github.com/vllm-project/vllm/issues/40252) states
plainly: "Every community NVFP4 quant of a Qwen3-Next-family model ships a
quantization_config.ignore list that names the old split tensor names for the
linear-attention block."

Concrete verified examples:

| Checkpoint | Ignore pattern for linear_attn | Source |
|---|---|---|
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `re:.*linear_attn.*` (ALL linear_attn) | HF model card, creation script |
| `prithivMLmods/Qwen3.6-27B-NVFP4` | `re:.*linear_attn.*` (ALL linear_attn) | HF model card |
| `vrfai/Qwen3.6-27B-NVFP4` | `in_proj_qkvz`, `in_proj_ba` patterns | HF model card: "DeltaNet recurrent cores are untouched" |
| `Sehyo/Qwen3.5-122B-A10B-NVFP4` | Missing (bug) → model outputs only "!" | HF discussion #4 |
| `raydelossantos/Qwen3.6-27B-GPTQ-Int4` | `.*attn.*` (ALL attention, both types) | HF model card: Qwen's own recipe |

**Is leaving GDN at higher precision standard practice?** Yes, in the NVFP4/GPTQ
ecosystem. But the justification is **correctness** (the fused-tensor-name bug),
not measured fidelity. No NVFP4 or GPTQ checkpoint we found cites a KLD
measurement or an ablation justifying the GDN exclusion on quality grounds. The
practice is "if we don't exclude it, the model breaks," not "we measured that
GDN quantization hurts quality."

**NVIDIA's own `nvidia/Qwen3.6-27B-NVFP4`** card states: "Only the weights and
activations of the linear operators within transformer blocks are quantized."
The card does not publish the exact ignore list, but the evaluation table shows
near-lossless task-quality recovery (MMLU Pro 86.3 vs 86.1 FP8, GPQA 85.5 vs
86.0), suggesting that whatever is excluded does not materially affect quality
on their benchmark suite — which is task-accuracy, not KLD.

### 3.2 Unsloth's measured tensor-sensitivity finding

Unsloth published the closest thing to a measured justification for
attention-over-MLP attribution in this architecture family
(https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks):

> "Quantizing any attn_\* is especially sensitive for hybrid architectures, and
> so leaving them in higher precision works well."

> "For the worst items, ssm_out dramatically increases KLD and the disk space
> savings is minuscule."

> "For the best items to quantize, ffn_up_exps and ffn_gate_exps are generally
> ok to quantize to 3bit. ffn_down_exps is slightly more sensitive."

This is from their 150+ KLD benchmarks across 121 configs on Qwen3.5, using
`llama-perplexity --kl-divergence` (community protocol, unknown relative to our
suite). The direction — attention sensitive, MLP cheap — **matches our
attribution physics exactly**. However, Unsloth did not diagnose the **mechanism**
(KV-cache compounding); they report the empirical sensitivity ranking without
explaining why attention is sensitive. Our contribution is the measured
mechanism: attention error compounds through the KV cache across positions
(+46% under-prediction), while MLP error is per-position and additive (−1.9%,
−6.0% on held-out predictions).

### 3.3 ubergarm's and sokann's GGUF recipes: finer-grained GDN treatment

**ubergarm/Qwen3.5-27B-GGUF** (https://huggingface.co/ubergarm/Qwen3.5-27B-GGUF)
uses a role-based custom recipe with three precision tiers:
- Q6_0 for all attention (`attn_gate`, `attn_qkv`, `attn_output`, `attn_q`,
  `attn_k`, `attn_v`)
- Q8_0 for GDN state (`ssm_alpha`, `ssm_beta`)
- IQ5_KS for MLP (`ffn_down`, `ffn_gate`, `ffn_up`)

This is a three-tier attribution: GDN state > attention > MLP, matching the
direction of our finding (attention/GDN more sensitive than MLP). The Q8_0 for
`ssm_alpha`/`ssm_beta` specifically targets the GDN state-transition matrices,
which are tiny tensors where the byte cost of Q8_0 is negligible but the
precision benefit may be high.

**sokann/Qwen3.6-27B-GGUF-4.256bpw**
(https://huggingface.co/sokann/Qwen3.6-27B-GGUF-4.256bpw) uses an even simpler
version: Q8_0 for `ssm_alpha` and `ssm_beta` only, IQ4_XS for everything else
(including `ssm_out`, all attention, and all MLP). Published KLD: Mean 0.033452,
99.9% 2.907 (wiki.test.raw, 580 chunks, community protocol). The card claims
this beats bartowski Q3_K_M (0.058818) and unsloth UD-Q3_K_XL (0.046348) at
similar size — but the recipe gives no special treatment to attention at all,
only to the two smallest GDN tensors.

### 3.4 Qwen's own GPTQ-Int4 recipe: MLP-only quantization

**Qwen/Qwen3.5-27B-GPTQ-Int4** (the official Qwen GPTQ recipe, reused verbatim
by `raydelossantos/Qwen3.6-27B-GPTQ-Int4`) quantizes **only the MLP/FFN layers
to Int4** and keeps everything else BF16:

> **BF16 (not quantized):** `lm_head`, `embed_tokens`, `.*attn.*` (Gated
> DeltaNet + Gated Attention), `.*mtp.*`, `.*shared_expert.*`, `.*visual.*`.

This is the simplest possible version of the attention-over-MLP attribution: 4-bit
MLP, 16-bit everything else. It is the same direction as our K5/K6 recipe
(MLP at lower bits, attention at higher bits), as NVIDIA's NVFP4 (4-bit MLP,
8-bit attention), and as llama.cpp's `attn_v` promotion. No KLD is published for
this checkpoint.

### 3.5 AWQ recipes

**mattbucci/Qwen3.6-27B-AWQ** (AWQ 4-bit, AMD RDNA4) and
**Avesed/Qwen3.6-27B-INT4-W4A16** (AWQ W4A16) were found but their model cards
do not publish detailed ignore lists or measured fidelity numbers. The AWQ
approach (activation-aware scale search) is orthogonal to the
per-module-attribution question; it improves the quantization quality of each
module but does not by itself decide which modules to quantize.

---

## 4. Whether anyone found our two asymmetries

### 4.1 Attention-dominance / KV-cache compounding

**Partially found, by Unsloth — direction yes, mechanism no.** Unsloth's
150+ KLD benchmarks on Qwen3.5 found that "Quantizing any attn_\* is especially
sensitive for hybrid architectures" and published a Pareto plot showing
attention tensors dominate KLD when quantized. This matches our direction.

**Not found: the KV-cache compounding mechanism.** No source we found — paper,
model card, forum post, or source code — diagnoses **why** attention
quantization error is disproportionately harmful. Our measured mechanism
(attention error propagates through the KV cache and is re-read at every later
position, compounding over the scored window; `self_attn` under-predicted by
+46%; `receipts/glm52-transfer-2026-08-19.md`;
`receipts/selfattn-fp4-additivity-failure-2026-08-19.md`) appears to be novel.
The llama.cpp heuristic (`category_is_attn_v` comment: "more sensitive to
quantization") states the fact without the mechanism. EXL2 and EXL3's
optimizers capture the effect implicitly (measurement-based allocation would
naturally give attention more bits if it measurably hurts more) but do not
isolate or name the compounding mechanism.

**Not found: the MLP-is-free measurement.** Our finding that MLP quantization
error is nearly free and additive (−1.9%, −6.0% on held-out predictions;
quantizing all 64 MLP layers to NVFP4 costs the third-party harness nothing:
0.002666 vs 0.002670 attn-only, `receipts/discord-leaderboard-2026-08-19.md` §1)
is corroborated by Unsloth's finding that "ffn_up_exps and ffn_gate_exps are
generally ok to quantize to 3bit" — but Unsloth did not measure the additivity
property (that MLP error stacks linearly without compounding).

### 4.2 Early-layer sensitivity

**Partially found, by llama.cpp and EXL3 — heuristic yes, measurement no.**

- llama.cpp's `use_more_bits` protects the first `n_layers/8` layers (line 430).
  The comment "Guard against craziness in the first few ffn_down layers" (line
  625) is an empirical observation, not a measured depth model.
- EXL3's `allocation.py` states "End layers contribute disproportionately to
  end-to-end error" (line 63) and promotes both ends of the stack.

**Not found: a measured early-vs-late KLD contrast.** No source we found
published a controlled experiment isolating layer position as the only variable
(quantizing early layers vs late layers at the same bit width and measuring
KLD). Our queued experiment
(`receipts/eda-depth-weighting-2026-08-19.md`: `VLLM_EXL3_FP6_LAYER_RANGE=0-12`
vs `51-63`, 13 layers each, identical byte cost) would be the first such
measurement we are aware of.

**Not found: the depth-blind-objective-harms-early-layers finding.** Our
measured result that depth-blind bit objectives strip bytes from the first 16
layers and hand them to the last 16 (`abs` −779 MB from L00-15, +786 MB to
L48-63; `receipts/eda-depth-weighting-2026-08-19.md`) appears to be novel. The
prior art (llama.cpp, EXL3) anticipates the problem by building in a layer
heuristic, but no one measured what happens when you **remove** that heuristic
and let a depth-blind optimizer run free.

---

## 5. What is genuinely novel in our K5K6 + attribution approach vs prior art

Stated conservatively:

1. **The KV-cache compounding mechanism for attention-dominance.** Prior art
   (Unsloth, llama.cpp) found that attention is more sensitive to quantization.
   We measured **why**: attention error propagates through the KV cache and
   compounds across positions, while MLP error is per-position and additive.
   This is a mechanism, not just a ranking.

2. **The measured negative: a depth-blind, role-blind proxy objective regresses
   fidelity.** Prior art (EXL2, EXL3) uses measurement-based allocation that
   implicitly captures the attention/MLP asymmetry. We built a proxy-based
   solver (`rel` = `sum_m eps(m,K)`) that is blind to both role and depth,
   solved it exactly (DP over a byte grid), built the checkpoint, and measured
   the regression (+0.000366 KLD, hydrated wins 470/512). This is a falsification
   of the proxy-to-KLD monotonicity assumption, not just a negative result.

3. **The depth-direction measurement.** Prior art (llama.cpp's `use_more_bits`,
   EXL3's `allocation.py`) builds in a U-shaped layer heuristic. We measured
   that a depth-blind optimizer strips early layers specifically, and that the
   direction is harmful if error accumulates with depth. The U-shape's
   last-1/8 promotion is not justified by any measurement we found.

4. **The 3.73×-per-bit law and the closed-form allocation rule.** Our measured
   exponential error-per-bit law (`eps(m,K) = c_m * 3.73^(-K)`,
   `docs/57-eda-allocation-revisit.md` §1) and the closed-form allocation rule
   (sort by `log_3.73(c_m / numel_m)`, cut at budget) are not found in prior art.
   EXL2 uses simulated annealing; EXL3 uses greedy marginal KLD improvement;
   llama.cpp uses fixed heuristics. None derives a closed-form from an
   exponential error model.

5. **The byte law and budget-neutral solve.** Our affine byte model
   (`bytes(role,K) = fixed(role) + params(role)*K/8`, every module's cost an
   integer multiple of 655,360 B, budget on a 25,664-point grid) and the
   exact-DP solver over it are not found in prior art. EXL2's budget is a
   total-bit count; EXL3's is a target bpw. Neither models per-module byte
   cost as an affine function of K.

**What is NOT novel:**
- Role-based attribution (attention > MLP in bits): standard in llama.cpp,
  Unsloth, NVIDIA NVFP4, Qwen GPTQ-Int4, ubergarm.
- Measurement-based per-module allocation: EXL2 (Frobenius norm), EXL3 (KLD).
- Layer-position heuristics: llama.cpp (`use_more_bits`), EXL3
  (`allocation.py`).
- KLD as a fidelity metric: llama.cpp (`--kl-divergence`), EXL3 (optimizer),
  Unsloth (benchmarks), localbench (benchmarks).
- Leaving GDN/linear_attn at higher precision: standard in NVFP4 community,
  driven by the fused-tensor-name correctness bug.

---

## 6. Concrete inspirations worth trying, with cost and falsification

### 6.1 EXL3's direct-KLD optimizer vs our proxy-error solver

**Idea:** Replace our `rel` proxy objective with EXL3's direct-KLD measurement
pass. EXL3's `measure_model.py` computes `F.kl_div` between quantized and
reference model outputs per module group, and `optimize_model.py` does greedy
marginal improvement. This would avoid the proxy-to-KLD sign error we measured
(the `rel` objective predicted −0.000251, measured +0.000366).

**Cost:** Requires a GPU measurement pass. EXL3's measurement needs multiple
pre-quantized model variants and a streaming forward pass over calibration
data. [INFERENCE] Based on EXL2's measurement pass taking hours on large models,
this is likely 2–4 hours of serial GPU time for a 27B model, plus the
conversion time for the input quantized variants.

**Falsification:** If the direct-KLD optimizer still moves bits attention→MLP
(the wrong direction), the problem is not the metric but the
module-local-measurement paradigm: any per-module measurement that doesn't model
cross-module KV propagation will underweight attention. Our `sqrt_energy`
weighting candidate (which got the sign right on all four calibration pairs,
`docs/57` §4) would then be the cheaper path.

### 6.2 EXL3's U-shaped layer-position heuristic

**Idea:** Add EXL3's `dist = min(layer, stack_max - layer)` term to our
allocation solver's module weight, giving priority to both ends of the stack
for bit promotion.

**Cost:** No GPU. It is a one-parameter change to the solver's weight function.

**Falsification:** Our depth finding says early layers matter more, late layers
less. If our queued early-vs-late KLD experiment
(`VLLM_EXL3_FP6_LAYER_RANGE=0-12` vs `51-63`) shows KLD(early) > KLD(late)
significantly, the U-shape wastes bytes on the last 1/8. A monotonic ramp
(weight ∝ `(1+amp)^(n_layers-1-layer)`, our `--depth-amp` formulation) would
then be the better choice. Cost of that experiment: 2 boots + 2 captures, ~30–40
min GPU (`receipts/eda-depth-weighting-2026-08-19.md`).

### 6.3 Q8_0 for GDN state-transition tensors (ssm_alpha, ssm_beta)

**Idea:** ubergarm and sokann both use Q8_0 for `ssm_alpha` and `ssm_beta` —
the GDN state-transition matrices. These tensors are tiny (the byte cost of Q8_0
is negligible), but they control the recurrent state update, so quantization
error there may compound through the recurrence (analogous to KV-cache
compounding for attention).

**Cost:** No GPU for the solve (the byte cost is negligible). A conversion + KLD
capture would cost ~50 min conversion + ~6 min capture.

**Falsification:** If promoting `ssm_alpha`/`ssm_beta` to the highest available
width (K7 or K8 in EXL3, or BF16 passthrough) does not move KLD relative to
K5/K6, the GDN state-transition precision is not the bottleneck. Our attribution
physics would predict it matters (recurrent state error compounds like KV-cache
error), but this is [INFERENCE] — we have not measured GDN-state-specific error
propagation.

### 6.4 Qwen's own MLP-only-Int4 recipe as a baseline

**Idea:** Qwen's official `Qwen3.5-27B-GPTQ-Int4` quantizes only MLP to Int4 and
keeps all attention (both full and linear/GDN) at BF16. This is the extreme
version of our attribution direction. Running this recipe through our KLD suite
would give us a measured baseline for "attention entirely unquantized, MLP at
4-bit" — the upper bound on what attention preservation can buy.

**Cost:** Requires a GPTQ conversion (~30–60 min GPU) and a KLD capture (~6
min). No weight download if we use the existing `raydelossantos/Qwen3.6-27B-GPTQ-Int4`
checkpoint — but that is Qwen3.6, not Qwen3.8, so it would need our cross-model
KLD methodology or a fresh conversion of Qwen3.8.

**Falsification:** If the MLP-only-Int4 baseline has KLD indistinguishable from
our hydrated K5/K6 build, then our K6 attention serialization is already
capturing essentially all the attention-preservation benefit, and further
attention investment (K7, BF16) would not help. If the baseline is significantly
better, there is headroom in attention precision that our K6 does not exhaust.

### 6.5 Unsloth's imatrix calibration for sensitive tensors

**Idea:** Unsloth found that imatrix (importance matrix from calibration data)
"definitely helps weight the quantization process in the right way" and reduces
KLD on sensitive tensors, especially at lower bit widths. Our trellis
quantization uses exllamav3's calibration data but does not use an imatrix-style
per-channel importance weighting. [INFERENCE] If the trellis format supports
imatrix-style weighting, this could reduce KLD on the attention tensors that
dominate our error budget.

**Cost:** Requires a re-conversion with imatrix calibration (~50 min GPU).
Falsification: if imatrix does not change trellis quantization (the EXL3 format
may not support per-channel importance weighting the way GGUF does), it is
inapplicable to our format. **This needs a source-code check** of whether
exllamav3's `AdaptiveGPTQ` already incorporates Hessian-based weighting
equivalent to imatrix — it does use a Hessian (`adaptivegptq.py`, `add_batch`
computes `H += X^T X` from calibration activations), so the GPTQ step is already
Hessian-weighted. The imatrix in llama.cpp is a simpler per-column scale; EXL3's
GPTQ Hessian is richer. [INFERENCE] The imatrix benefit may already be captured
by EXL3's GPTQ, making this inspiration a no-op for our format.

---

## Summary answers to the four report-back questions

1. **llama.cpp `use_more_bits` layer-index heuristic — exact behaviour and
   direction:** U-shaped. Returns true for `i_layer < n_layers/8` (early),
   `i_layer >= 7*n_layers/8` (late), or `(i_layer - n_layers/8) % 3 == 2`
   (every 3rd middle layer). Both ends get more bits; the first-1/8 part aligns
   with our depth finding, the last-1/8 part does not. Source:
   `src/llama-quant.cpp:430-432`.

2. **EXL2's allocation objective in one sentence:** Minimize a configurable
   norm of per-module-group relative-Frobenius-norm output error
   (`1 - ||Q(x)-x||_F / ||x||_F`) subject to a total-bit budget, via simulated
   annealing — no layer-position term, no KLD. Source:
   `exllamav2/conversion/optimize.py`, `measure.py`.

3. **Is GDN-at-higher-precision standard practice?** Yes, in NVFP4/GPTQ
   checkpoints of Qwen3.5/3.6. But the justification is correctness (the
   fused-tensor-name loading bug, vLLM #40252), not measured fidelity. Unsloth's
   finding ("attn_\* especially sensitive for hybrid architectures") is the
   closest to a measured justification, but it covers attention broadly, not GDN
   specifically.

4. **Did anyone previously publish the attention-dominance or early-layer
   findings?** Attention-dominance: **partially** — Unsloth found the direction
   (attn sensitive, MLP cheap) empirically via 150+ KLD benchmarks, but did not
   diagnose the KV-cache compounding mechanism. Early-layer sensitivity:
   **partially** — llama.cpp and EXL3 have U-shaped heuristics that protect
   early layers, but no one measured the direction or magnitude of the depth
   effect, and no one published a controlled early-vs-late KLD experiment. The
   MLP-is-free/additive measurement and the depth-blind-objective-regresses
   measurement appear to be genuinely novel to our project.
