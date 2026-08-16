# VRAM-class profiles: a 24 GB mid-profile and a 16 GB profile, by arithmetic

This is the *design* half of [29](29-plan-and-loose-ends.md) F3; the **Verdict** section below
carries the decision it now supports, and F3 itself is down to three open items.
Nothing here is a claim that a profile fits a card: every profile below ends in an acceptance
test that must pass on the actual board class before the profile may be published, and every
number is either traced to a receipt or labelled **[P]** for prediction with its formula shown.

Terminology is the F4 convention: **serialized bytes** are what a checkpoint occupies on disk,
**resident weight bytes** are what the loader keeps on the device, **activation**, **CUDA-graph**
and **KV** are separate quantities, and none of them is called VRAM.

## Verdict

Both classes now have a decision, and it is published with its arithmetic in
[`receipts/vram-class-verdict.json`](../receipts/vram-class-verdict.json). That receipt re-derives
the card model of §3 on the constants the **physical** RTX 5090 qualification measured
(`receipts/qualification-5090-context.json`: all seven gates pass at
`--gpu-memory-utilization` **0.955**, engine budget 29.98 GiB of 31.4 GiB usable, 18.19 weight +
1.78 activation + 0.27 non-torch + 0.45 CUDAGraph = 20.69 GiB, KV **9.28 GiB → 265,122 tokens**),
not on the 0.97 utilisation §3 was written against. §10 then *measured* the KV law at a capped
24 GiB-class budget, which turns every bounded window below into a point value. Several lengths
are re-derived and §5.3's 40,960 is retired outright; each is flagged where it appears and listed
in the receipt's `supersedes` block.

| class | decision | artifact | context | fidelity |
|---|---|---|---|---|
| **24 GB** | **GO** | the published `-context` edition, **no conversion** | **24,576 [P]** MTP-3 + fp8 KV; **45,056 [P]** MTP off — predictions validated by the §10 capped-budget proxy, physical-board gate still **open** | **0.003409, measured**, carried unchanged (`receipts/kld5-1M-tail-ctx.json`) |
| **16 GB** | **NO-GO as a SKU — publish §6 as a design study** | none exists; every reachable point needs a sub-4-bit width | **28,672 [P]** MTP off; **MTP-3 impossible at any window** | **none, and none obtainable from anything published** |

**24 GB: go, because no new checkpoint is required.** The context edition's resident weights are
*measured* at **18.41 GiB** (`receipts/native-mtp-8mp-amendment.json`, as run; the physical 5090
logged 18.19 GiB for the same configuration and the verdict deliberately uses the larger figure).
The card model gives `24 × 0.98125 = 23.55` free at startup, `× 0.955 = 22.49` engine budget and
`− 2.50 = 19.99 GiB` for weights plus KV, so **KV available is 19.99 − 18.41 = 1.58 GiB [P]**.
§10's board arm then *measured* that pool at **1.79 GiB**, on a card ballasted down to a 24 GiB
board's free memory, and pinned both terms of the KV law — so the window is a point value now, not
a range: **24,576 [P]** with MTP-3. With MTP off the CUDA-graph pool vanishes along with the draft
weights, so overheads drop to `1.68 + 0.27 + 0.00 = 1.95 GiB` and resident to 18.158, giving
`22.49 − 1.95 − 18.158 = 2.38 GiB [P]` of KV and a published window of **45,056 [P]** — the length
§10 actually started and passed.
Fidelity is the already-published 0.003409 on shard 0 / 0.003509 over 10,480,640 positions,
unchanged, because it is the same weights — the 24 GB class carries **no new fidelity risk at
all**, which is the finding that makes it a go. What "go" licenses: publishing a 24 GB serving
profile of an existing artifact with its window labelled a bounded prediction, and running the
qualification. It does **not** license a fit claim; "predicted" and "allocated" are still
different sentences.

**§4's per-token KV model is retired, and two of the four lengths it produced move.** Reading the
pinned engine's own sizing code (`vllm/v1/core/kv_cache_utils.py`) shows that
`GPU KV cache size: N tokens` is `int(max_concurrency × max_model_len)` with
`max_concurrency = num_blocks / cdiv(max_memory_usage_per_request, memory_per_block)` — a
per-request quantity scaled by the window, **not** pool ÷ per-token cost. So the pool one request
needs is affine in the window, and §10.2 pins both of its terms from four startup refusals rather
than bounding them:

```
kv_needed(L) = a·L + M            reported tokens = int(L · pool / kv_needed(L))
MTP-3 :  a = 34,816 B/token (= 32,768 × 17/16 exactly)   M = 0.63 GiB    [measured]
MTP off: a = 32,932 B/token (= 1.005 × the fp8 floor)    M = 0.14 GiB    [measured]
```

This is no longer the one-parameter family an earlier revision of this section carried: two
windows per configuration determine `a` and `M` with no slack left, and inverting the law
reproduces the engine's own printed "estimated maximum model length" (§10.2). The lengths below
stay **[P]** anyway, because the *law* is measured but the *pool* is still predicted from §3's
card model — and the two resident-weight figures this project has measured put that pool anywhere
between 1.58 and 1.80 GiB.

| row | was | now | why |
|---|---|---|---|
| 24 GB, MTP-3 | 32,768 | **24,576** | 32,768 needs 1.6925 GiB. At the measured 1.79 GiB pool that clears 5.8 %, against the **≥15 %** rule every other published length here obeys; at the conservative 18.41 GiB resident the pool is 1.58 GiB and it does not fit at all. 24,576 needs 1.4269 GiB — 25.4 % headroom at the measured pool, and it still fits at the conservative one |
| 24 GB, MTP off | 45,056 | **45,056**, kept | the board arm ran this window and allocated 76,032 tokens, 1.69× concurrency, 70.8 % headroom. The engine supports 80,208 at that pool and the 15 % rule would allow 65,536, so 45,056 is deliberately conservative — but it is a length we have *started*, and a raise would be a prediction |
| 16 GB, MTP-3 | 8,192 | **withdrawn** | the measured fixed MTP-3 term, 0.63 GiB, exceeds the whole 0.55 GiB pool on its own, before a single token and before the 0.45 GiB of CUDA-graph pool MTP also costs |
| 16 GB, MTP off | 12,288 | **28,672** | the retired 0.20 GiB over-charge is replaced by a measured 0.14 GiB fixed term, and MTP off also frees the 0.45 GiB CUDA-graph pool and 0.10 GiB of draft activation. On the 16 GB owner's pool, `14.99 − 1.95 − 11.742 = 1.298 GiB`, `L_max` is 37,756 and the ≥15 % rule gives **28,672**, needing 1.0194 GiB for 27.3 % headroom. 32,768 needs 1.1450 GiB and clears only 13.4 %, so it misses the rule by one 4,096 step — the rule is on memory (`a·L + M ≤ pool/1.15`), not on tokens (`L ≤ L_max/1.15`), and the two diverge now that `M ≠ 0`. More than twice §6.2's figure, and entirely unstarted |

**§5.3's 40,960 is retired twice over.** At the qualified 0.955 utilisation it does not fit under
the old model (1.603 GiB against 1.580), and under the measured law it needs **1.9581 GiB** against
a pool of at most 1.80. The credit for spotting the window dependence — and for the observation
that neither reported token count is a multiple of the 1,600-token attention block, which is what
ruled out `num_blocks × block_size` — belongs to the 24 GiB proxy work, which then closed the open
parameter exactly the way `receipts/vram-class-verdict.json` → `open_risks_to_this_verdict` said it
would: any startup logging a second window pins `a` and `M`, and §10.2 logged four. Weights fit a
24 GB board throughout, so the **class** verdict never depended on any of this — only the
**window** did.

**16 GB: no-go as a SKU.** The byte law says the cheapest multimodal build that keeps the MLP at
4 bits is **13.58 GiB resident [P]** (payload 13.235 + 0.35 loader allowance) against a **12.70
GiB** budget — **0.88 GiB over before a single KV byte**, and 1.09 GiB over once the measured
0.955 utilisation and 2.50 GiB overheads are used instead (`13.58 − 12.49`), so today's result
*hardens* the conclusion. Every remaining path is therefore sub-4-bit, and **no width below 4 bits
has ever been measured for KLD in this family, at any role, on any suite.** The S16-V candidate is
entirely predicted: **11.94 GiB resident [P]** (= 11.586 payload + 0.35) and **28,672 tokens [P]
with MTP off** — more than twice §6.2's original figure, because the measured law charges MTP-off
only a 0.14 GiB fixed term and turning MTP off also frees the CUDA-graph pool. Its MTP-3 row is not
short, it is **gone**: the measured fixed per-request KV term is 0.63 GiB against a total pool of
0.553 GiB, short by 0.077 GiB, so speculative decode does not fit at this class at any window, and
S16-V-long's extra bit reaches only 764 tokens of it rather than rescuing it. Its
fidelity is **unknown**: the nearest measured neighbour below the published set is K4 at 0.010604
with a p99.9 of 0.5555, already the worst of the five candidates, and S16-V is one bit below it on
each MLP projection, three on the linear-attention stack, two on full attention and two on the
head. §6.4's 0.03-0.10 is a range with a shape, not an estimate. Two reader reports point the same
way and are not ours: a ~12 GB `IQ3_XXS` on a 16 GB RTX 5070 Ti gives "less than 5 tok/s" at 64K
or 128K (megathread `1voojjz`, `p3ui0np`), and `UD-Q4_K_XL` on a 24 GB 4090 "only leaves me with
about 18.4k context" (`p3vfwqh`) — self-reported configurations, no KV dtype or engine version,
never mixed into the arithmetic, but the only external evidence this model has at these classes
and both negative. So §6 stays published as a **design study**: budgets, bit allocation, the
error-driven allocator route and the acceptance gates are all useful to whoever attempts it. What
may not happen is a released 16 GB artifact, a 16 GB row in a model card, a 16 GB context length
quoted as a capability, or any fidelity number attached to a 16 GB profile.

**What flips 16 GB to go**, in order: (1) a measured KLD for one sub-4-bit width in this family on
shard 0 of the v5 suite with its tail row — one conversion and one shard, **no 16 GB card
needed**, and it is the blocking item; (2) a startup on a physical 16 GB board, since the
activation, non-torch and CUDAGraph figures here are carried from a 32 GB profile; (3) needle
retrieval and a combined text-plus-image request at the 4.2 MP cap on a K3 body; (4) the
non-termination check, weighted, because S16-V sits below the Q4 builds two readers describe
looping to context exhaustion.

## 1. What this design is built on

| fact | value | source |
|---|---|---|
| layers | 64 = 48 `linear_attention` + 16 `full_attention`, `full_attention_interval` 4 | [docs/01](01-nvfp4-composition.md) L15, [docs/07](07-serving-recommendations.md) §5.6 |
| attention geometry | `head_dim` 256, 24 Q heads, 4 KV heads, `attn_output_gate: true` | [docs/01](01-nvfp4-composition.md) L17, [docs/07](07-serving-recommendations.md) §5.6 |
| hidden / intermediate / vocab | 5120 / 17408 / 248320, `tie_word_embeddings: false` | [docs/01](01-nvfp4-composition.md) L16-18 |
| parameters | 27,781,427,952 logical | `receipts/hydrated-quantization-manifest.json` → `logical_parameter_count` |
| hydrated per-role bytes | 21,586,964,548 total | `receipts/hydrated-quantization-manifest.json` → `roles` |
| context per-role bytes | 20,672,081,988 total | `receipts/context-quantization-manifest.json` → `roles` |
| published disk bytes | hydrated 21,610,933,884; context 20,696,053,306 | `receipts/release-evidence-{hydrated,context}.json` → `artifact.disk_bytes` |
| resident weights, context build | BF16 embed / MTP off 19.31; BF16 / MTP-3 19.56; int8 / MTP off 18.13; int8 / MTP-3 18.38 GiB | `receipts/release-evidence-context.json` → `artifact.resident_weights_gib` |
| resident weights, as run | **18.41 GiB** (int8 embed, MTP-3) | `receipts/native-mtp-8mp-amendment.json` → `measured_memory.resident_weights_gib` |
| engine budget / activation / non-torch / graphs | 30.24 / 1.78 / 0.28 / 0.46 GiB | same receipt, `runtime.desired_engine_budget_gib`, `measured_memory` |
| KV pool as run | **9.31 GiB → 266,612 tokens**, fp8, MTP-3, `max_num_seqs` 1, 8,388,608-pixel cap | same receipt, `measured_memory.available_kv_gib`, `.gpu_kv_cache_tokens` |
| KV pool without MTP | 8.18 GiB → 266,743 tokens, attention-K5 body, `max_num_seqs` 4 | `receipts/context-capacity-5090-budget.json` → `runs["attn-k5"]` |
| 5090 free-at-startup | 31.39 GiB on a 32 GiB board; 0.97 of it = 30.45 GiB | `receipts/release-evidence-context.json` → `hardware.context_budget_note` |
| image cap exchange | 16.8 MP → 8.59 GiB KV; 8.4 MP → 9.31 GiB KV, same budget | [docs/32](32-native-context-embedding-overlay.md) table L97-100 |
| int8 input overlay | −1.272 GB embedding, +0.000065 mean KLD [+0.0000046, +0.00013] on v3 | [docs/32](32-native-context-embedding-overlay.md) L32-43 |
| headline fidelity, 10,480,640 positions | hydrated 0.002760, online K5/K6 0.003210, context 0.003509, official FP8 0.005294, K4 0.010604 | `receipts/kld5-10M-{hyd,k5k6,ctx,fp8,k4}.json` → `token_mean_kld`; [docs/33](33-evidence-volume-and-intervals.md) |
| accepted `--kv-cache-dtype` set in the pinned r34 build | `auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, nvfp4_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc, int4_per_token_head, int8_per_token_head, fp8_per_token_head, nvfp4` | [docs/29](29-plan-and-loose-ends.md) F5 L167-177 |
| vision tower is not a bit-width knob | `-vb 6` splits upstream's fused `visual.blocks.N.attn.qkv`; rejected by the topology check, on the do-not-reopen list | [docs/30](30-iteration-4-context-edition.md) L108-112, [docs/29](29-plan-and-loose-ends.md) L577 |

## 2. The byte law

Both published manifests are reproduced **exactly** by one affine rule per role:

```
bytes(role, K) = fixed(role) + params(role) * K / 8
```

`fixed` is EXL3's per-module overhead — `2*(in+out)` fp16 `suh`/`svh` bytes plus one int32
codebook scalar, the formula published in [docs/02](02-recipe-k4.md) — plus any BF16/F16
tensors the role carries. Worked check for `mlp_gate_proj`: 64 modules ×
(2·(5120+17408) + 4) = 2,883,840 B, and 2,883,840 + 5,704,253,440·5/8 = 3,568,042,240, the
manifest's K5 value to the byte. Same for `full_attention`: 16 × (81,936 + 1,024 B of BF16
`q_norm`/`k_norm`) = 1,327,360, and 1,327,360 + 1,677,721,600·6/8 = 1,259,618,560 = the
hydrated K6 value. The K5→K6 delta of every role is exactly `params/8`, which is what licenses
extrapolating the same rule to K3 and K4.

| role | params | `fixed` (B) | GiB per bit |
|---|---:|---:|---:|
| `full_attention`, 16 L, q/k/v/o (q is 12288-wide: 24×256 plus the output gate) | 1,677,721,600 | 1,327,360 | 0.1953 |
| `linear_attention`, 48 L, `in_proj_qkv`/`in_proj_z`/`out_proj` | 5,536,481,280 | 54,777,408 | 0.6445 |
| `mlp_gate_proj`, 64 L | 5,704,253,440 | 2,883,840 | 0.6641 |
| `mlp_up_proj`, 64 L | 5,704,253,440 | 2,883,840 | 0.6641 |
| `mlp_down_proj`, 64 L | 5,704,253,440 | 2,883,840 | 0.6641 |
| `lm_head` (248320 × 5120, untied) | 1,271,398,400 | 506,884 | 0.1480 |
| `mtp_draft`, 8 quantized modules (4 attention = 104,857,600 params, 3 MLP, 1 `eh_proj`) | 424,673,280 | 300,064 | 0.0494 |
| `embed_tokens` | 1,271,398,400 | — | BF16 2,542,796,800 B; int8 1,271,398,400; int4 g128 655,564,800 |
| `vision_tower` | 460,730,096 | — | BF16 921,460,192 B, fixed |
| `norms_and_small` | — | — | BF16 1,320,960 B, fixed |

int4-g128 embedding bytes are `248320·5120/2 + 248320·(5120/128)·2` = 635,699,200 + 19,865,600
= 655,564,800 **[P]**, the packing implied by the `int4_group_128` quantizer in
`receipts/embedding-quant-error.json`.

Whole-tree bytes are the tensor payload plus tokenizer, configs and chat template:
21,610,933,884 − 21,586,964,548 = 23,969,336 B for hydrated and 23,971,318 B for context, so
**predicted disk bytes = payload + 24.0 MB [P]**.

**Resident weights** are the same payload with the embedding replaced by its resident form,
plus a loader overhead measured at +0.305 to +0.312 GiB on the context build across all four of
its published embed/MTP combinations, +0.206 GiB on hydrated, and +0.342 GiB in the
amendment run. Planning uses **+0.35 GiB [P]**, above every observation:

```
resident_GiB = (payload_bytes / 2^30) + 0.35          payload = Σ role bytes, embed at its resident width
```

Check against the receipts: context payload with BF16 embed and MTP-3 is 19.252 GiB → 19.60
predicted against 19.56 measured; with the int8 overlay 18.068 → 18.42 predicted against 18.38
published and 18.41 as run. The model is therefore accurate to ±0.05 GiB on the one build where
all four combinations are measured, and conservative.

## 3. Card arithmetic

> **Superseded by the Verdict above.** This table is the derivation at utilisation **0.97** and
> 2.52 GiB of overheads. The physical qualification measured 0.955 and 2.50, which gives 19.99
> GiB for weights plus KV at 24 GiB and 12.49 at 16 GiB
> ([`receipts/vram-class-verdict.json`](../receipts/vram-class-verdict.json) →
> `card_budget_model`). It is kept here because the two-step model itself is unchanged and its
> free-at-startup fraction is now measured (31.4/32 = 0.98125) rather than quoted.

Only one board has ever been budget-proved here, and it was emulated: 30.24 GiB inside a 95.6
GiB RTX PRO 6000, chosen to sit under an RTX 5090's 0.97 × 31.39 = 30.45 GiB. The same two
steps give the smaller classes **[P]**:

```
free_at_startup = nominal_GiB * (31.39 / 32) = nominal_GiB * 0.98094
engine_budget   = free_at_startup * 0.97
```

| class | free at startup **[P]** | engine budget **[P]** | non-KV overheads (measured) | left for weights + KV **[P]** |
|---|---:|---:|---:|---:|
| 32 GiB | 31.39 (measured) | 30.45 | 2.52 | 27.93 |
| **24 GiB** | 23.54 | **22.84** | 2.52 | **20.32** |
| **16 GiB** | 15.70 | **15.22** | 2.52 | **12.70** |

Overheads are 1.78 activation + 0.28 non-torch + 0.46 CUDA-graph GiB, all from
`receipts/native-mtp-8mp-amendment.json` → `measured_memory`, at `max_num_seqs` 1,
`max_num_batched_tokens` 2048, `cudagraph_mode` `FULL_DECODE_ONLY`, `cudagraph_capture_sizes`
`[4]`, and an 8,388,608-pixel image ceiling. Carrying the 32 GB activation figure to a 16 GB
card is deliberately conservative: activation profiling scales with the permitted image and the
batched-token count, both of which a 16 GB profile lowers. The `max_num_seqs` 4 configuration in
`receipts/context-capacity-5090-budget.json` measured 2.05-2.11 GiB activation instead of 1.78,
so **concurrency is a memory decision, not a scheduling one**, and both profiles below are
single-stream.

The `0.98094` free-at-startup fraction and the `0.97` utilisation are the two softest numbers in
this document. A board that reserves more for the display, or a driver with a different
allocator, moves both. This is why every profile ends with a measured startup.

## 4. The KV model, and where the hybrid state went

KV exists only on the 16 `full_attention` layers. The 48 GDN layers keep a fixed-size recurrent
state per sequence instead of a per-token cache — that asymmetry is the whole reason 262,144
tokens is affordable on a 27 B model. The exact tensor cost at fp8 is:

```
per_token_floor = 2 (K and V) * 4 kv heads * 256 head_dim * 1 byte * 16 full_attention layers
                = 32,768 B/token = 32.0 KiB/token
```

### 4.1 What a pool actually costs: the measured affine law

The engine does not divide a pool by a per-token cost. `GPU KV cache size: N tokens` is
`int(max_concurrency × max_model_len)` with
`max_concurrency = num_blocks / cdiv(max_memory_usage_per_request, memory_per_block)`
(`vllm/v1/core/kv_cache_utils.py` in the pinned r34 image), so the quantity that scales is the pool
one **request** needs at the configured window — and that is affine in the window, not linear
through the origin. §10.2 measured both of its terms by provoking four startup refusals, which
print the requirement directly; two windows per configuration determine the pair with no slack:

```
kv_needed(L) = a·L + M          reported tokens = int(L · pool / kv_needed(L))
                                capacity        L_max = (pool − M) / a

            a (B/token)   as x the 32,768 B/token fp8 floor   M (GiB)
MTP-3       34,816        1.0625 = 17/16, exactly             0.63
MTP off     32,932        1.005                               0.14
```

`a` for MTP-3 lands exactly on the engine's own "Add 3 padding layers, may waste at most 6.25 % KV
cache memory" line. `a` for MTP off is 1.005× the tensor floor — the attention tensors and almost
nothing else, which independently confirms this section's original argument that the residual over
the floor is MTP-driven. Both are measured; both are `max_num_seqs` 1, fp8 KV, one engine version,
one hybrid allocator, one `block_size`, and neither is a physics constant.

Worked example, and the exact reason the old model failed: under MTP-3 one request at a 262,144
window needs `34,816 × 262,144 + 0.63 GiB = 9.13 GiB`, but at a 32,768 window it needs only
**1.6925 GiB** — so the *same* 3.27 GiB pool buys 265,122 tokens at the long window and 62,557 at
the short one. A fixed per-*pool* reserve cannot express that. A fixed per-**request** term can.

**Retired here:** `KV_bytes(T) = H + c_tok·T` with `H = 0.20 GiB`,
`c_tok(MTP-3) = 36,689.2 B/token` and `c_tok(no MTP) = 32,929.0 B/token`. Its per-token rate was
inadmissible — 36,765 B/token implied, above the 34,816 B/token the engine's own padding line
allows — and its fixed term was simultaneously three times too small for MTP-3 and 0.06 GiB too
large for MTP off. The 4-bit-KV rule `c_tok(fp8) × 16,384/32,768` survives only as `a/2` with `M`
held, and it is still **[P]**, never measured.

### 4.2 What MTP-3 costs, as its own line item

Budgeting a small board, this is the number that surprises people. Turning MTP-3 on costs
**1.19 GiB before a single KV token is stored**, in three separate places. All three are measured
in §10.3, by differencing the same configuration with and without the draft model:

| component | MTP-3 | MTP off | what MTP-3 costs |
|---|---:|---:|---:|
| draft weights, resident | 18.19 | 17.94 | **0.25 GiB** |
| CUDA-graph pool | 0.45 | **0.00** | **0.45 GiB** |
| fixed per-request KV, `M` | 0.63 | 0.14 | **0.49 GiB** |
| **total, before any token** | | | **1.19 GiB** |
| peak activation (the draft model's share) | 1.78 | 1.68 | 0.10 GiB |

…and then **5.7 %** on top of every KV token as well (34,816 against 32,932 B/token — equivalently
5.75 % of the fp8 floor). The CUDA-graph row is the one no arithmetic in this document had ever
charged to MTP: with MTP off this configuration captures no decode graph at all and that pool goes
to **zero**, so overheads are 1.95 GiB without MTP against 2.50 GiB with it.

On a 32 GB board 1.19 GiB is 13 % of a 9.28 GiB KV pool and nobody notices. On a 24 GiB board the
fixed KV term **alone** is 35 % of the measured 1.79 GiB pool, and the whole 1.19 GiB is two thirds
of it. That is the entire reason the same card serves **24,576** tokens with MTP-3 and **45,056**
without.

### 4.3 Where the hybrid state went

The question this section was written to answer: does the pool include the hybrid
linear-attention state? It includes whatever the engine put there, because the receipts report one
`available_kv_gib` and one `gpu_kv_cache_tokens` and itemize neither. The measured law now settles
it. The MTP-off fixed term is **0.14 GiB** — the same order as the ≤0.15 GiB the only external
sizing gives for the GDN pool at `max_num_seqs` 1 (48 GDN layers × 48 heads × 128 × 128 at
`mamba_ssm_dtype` plus a bf16 conv state, "153.9 MB at fp32, 78.4 MB at bf16" per slot, quoted from
SGLang in [docs/07](07-serving-recommendations.md) §5.6) — while the MTP-3 fixed term is
**0.63 GiB**. The 0.49 GiB difference is MTP-3's draft KV and page padding, and the remainder is
consistent with the recurrent state finally being visible in the pool once it is measured as a
per-request cost instead of inferred from a per-pool residual. What is *not* supportable any more
is the old reading of the 1.1736 GiB gap between 9.31 GiB and the tensor floor at 266,612 tokens as
a single mysterious residual: most of that gap is `a`'s 6.25 % padding, applied per token.

## 5. Profile M24 — the 24 GB mid-profile

The finding that shapes this profile: **at 24 GB no new checkpoint is required.** The published
`-context` edition, served with the int8 input overlay and MTP-3, has a *measured* resident
weight figure of 18.41 GiB, which leaves `22.49 − 2.50 − 18.41 = 1.58 GiB` for KV on a 24 GiB board
at the qualified 0.955 utilisation — and §10's board arm measured **1.79 GiB** on a card ballasted
to that class. So
M24 is a serving profile over an existing artifact, and its fidelity number is the already
published 0.003509 at 10,480,640 scored positions — no new conversion, no new fidelity risk.
M24-Q and M24-L are the two dials on either side of it.

### 5.1 Bit allocation

| role | M24-Q (= hydrated, published) | **M24 (= context, published)** | M24-L (new conversion) |
|---|---|---|---|
| `full_attention` q/k/v/o, 16 L | K6 serialized | **K5 serialized, calibrated** | K5 serialized |
| `linear_attention` `in_proj_qkv`/`z`/`out_proj`, 48 L | K6 serialized | **K5 serialized, calibrated** | K5 serialized |
| `linear_attn.in_proj_a`/`in_proj_b` | F16 (converter Gap 2) | **F16** | F16 |
| `mlp_gate_proj` / `mlp_up_proj` | K5 / K5 | **K5 / K5** | K5 / K5 |
| `mlp_down_proj` | K6 | **K6** | **K5** |
| `lm_head` | K6/mcg | **K6/mcg** | **K5/mcg** |
| `mtp_draft` | attn K6, gate/up K5, down K6, `eh_proj` K4 | **attn K5, gate/up K5, down K6, `eh_proj` K4** | **all K4** |
| `embed_tokens` | BF16 on disk → **int8 per-row at load** | BF16 on disk → **int8 per-row at load** | BF16 on disk → int8 at load |
| `vision_tower` | BF16, 8.4 MP cap | **BF16, 8.4 MP cap** | BF16, 8.4 MP cap |
| `norms_and_small` | BF16 | **BF16** | BF16 |

M24-Q and M24 bit maps are read from `receipts/hydrated-quantization-manifest.json` and
`receipts/context-quantization-manifest.json` → `roles[*].formats`. The MTP split is the
manifests' own: hydrated is `{K4:1, K5:2, K6:5}` and context `{K4:1, K5:6, K6:1}`, and the
13,107,200-byte difference between them is exactly 104,857,600 params × 1 bit / 8 — the four MTP
attention modules tracking the body's attention width.

### 5.2 Bytes

M24-L is the only new artifact, so only it needs the byte law:

| role | K | bytes | GB | GiB |
|---|---|---:|---:|---:|
| `full_attention` | 5 | 1,049,903,360 | 1.050 | 0.978 |
| `linear_attention` | 5 | 3,515,078,208 | 3.515 | 3.274 |
| `mlp_gate_proj` | 5 | 3,568,042,240 | 3.568 | 3.323 |
| `mlp_up_proj` | 5 | 3,568,042,240 | 3.568 | 3.323 |
| `mlp_down_proj` | 5 | 3,568,042,240 | 3.568 | 3.323 |
| `lm_head` | 5 | 795,130,884 | 0.795 | 0.741 |
| `mtp_draft` | 4 | 212,636,704 | 0.213 | 0.198 |
| `norms_and_small` | BF16 | 1,320,960 | 0.001 | 0.001 |
| `vision_tower` | BF16 | 921,460,192 | 0.921 | 0.858 |
| `embed_tokens` on disk | BF16 | 2,542,796,800 | 2.543 | 2.368 |
| **serialized total [P]** | | **19,742,453,828** | **19.742** | **18.387** |
| predicted disk bytes **[P]** | | 19,766,425,146 | 19.766 | 18.409 |
| resident payload, int8 embed **[P]** | | 18,471,055,428 | | 17.203 |
| **resident weights [P]** = 17.203 + 0.35 | | | | **17.55** |

| profile | serialized [P or receipt] | resident weights | basis |
|---|---:|---:|---|
| M24-Q | 21,586,964,548 B = 21.587 GB | **19.27 GiB [P]** = 20.104 − 1.184 + 0.35 | manifest; overlay never run on this checkpoint |
| **M24** | 20,672,081,988 B = 20.672 GB | **18.41 GiB** | `receipts/native-mtp-8mp-amendment.json` |
| M24-L | 19,742,453,828 B = 19.742 GB **[P]** | **17.55 GiB [P]** | byte law + 0.35 overhead |

The int8 overlay saves 2,542,796,800 − 1,271,398,400 = 1,271,398,400 B = 1.184 GiB of resident
weights and changes serialized bytes by zero: the table on disk stays BF16 and is narrowed after
load by `VLLM_EXL3_EMBED_BITS=8` ([docs/32](32-native-context-embedding-overlay.md)).

### 5.3 KV budget and context

> **Re-derived, twice over.** These lengths were computed at utilisation 0.97 with the
> `H + c_tok·T` model this document no longer uses. Both are gone: the budget below is the
> qualified **0.955**, and the requirement is the measured `a·L + M` of §4.1, whose fixed
> per-request term is **0.63 GiB** under MTP-3 and **0.14 GiB** without, against the 0.20 GiB `H`
> charged here. The M24 row becomes **24,576** with MTP-3 and **45,056** with MTP off, both
> started and gated on a ballasted 24 GiB-class proxy (§10); **40,960 is retired**, needing
> 1.9581 GiB against a pool of at most 1.80.

The publishing rule is unchanged — the largest multiple of 4,096 whose requirement leaves
**≥15 % headroom** against the pool — but the requirement is now `kv_needed(L) = a·L + M` from
§4.1 and the budget is the qualified `24 × 0.98125 × 0.955 = 22.49 GiB`. Note the rule is on
**memory**: `a·L + M ≤ pool/1.15`. A token-form rule (`L ≤ L_max/1.15`) is *not* equivalent and
over-publishes; the two coincide only when `M = 0`, which is exactly the assumption the
measurement retired. Overheads are the measured ones and differ by configuration: **2.50 GiB**
with MTP-3 (1.78 activation + 0.27 non-torch + 0.45 CUDA-graph) and **1.95 GiB** without
(1.68 + 0.27 + **0.00**); MTP-off rows also drop the draft weights (0.252 GiB for M24, 0.264 for
M24-Q, 0.198 for M24-L). The 15 % absorbs §3's two soft constants plus the ±0.05 GiB resident
model. All lengths **[P]** — the law is measured, these pools are not:

| profile | resident | KV avail **[P]** | `L_max` MTP-3 | **published, MTP-3** | resident, MTP off | KV avail **[P]** | `L_max` MTP off | **published, MTP off** | `L_max` 4-bit KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M24-Q (hydrated body) | 19.27 | 0.72 | 2,783 | **none — see below** | 19.006 | 1.53 | 45,459 | **36,864** | 5,566 |
| **M24 (context body)** | **18.41** | **1.58** | **29,306** | **24,576** † | **18.158** | **2.38** | **73,108** | **45,056** † | 58,612 |
| M24-L | 17.55 | 2.44 | 55,828 | **45,056** | 17.352 | 3.19 | 99,388 | **81,920** | 111,657 |

**†** M24's two published lengths are not derived from the predicted pools in this table. They are
the lengths §10 *started*, on measured pools of 1.79 and 2.60 GiB. See §5.3.1.

**M24-Q cannot run MTP-3 at this class at all.** Its 0.72 GiB pool is barely above the 0.63 GiB
fixed per-request term, leaving **2,783** tokens. The row is withdrawn rather than shortened —
the same failure mode §6 hits at 16 GB. The hydrated body at 24 GB is an MTP-off profile or
nothing, and that is a new consequence of the measured law: the retired model showed it at 20,480.

The 4-bit-KV column halves `a` and holds `M`. It is a raw maximum, not publishable at all until
that dtype has been measured (§7). Every figure here becomes a claim only after §8.

#### 5.3.1 M24, in three tiers

M24 is the only profile here with a measurement behind it, so it is published as a tier rather
than one number. All three tiers are computed from the **measured** board-arm pools of §10 —
1.79 GiB with MTP-3, 2.60 GiB without — not from the predicted pools above.

| | MTP-3 | MTP off | basis |
|---|---:|---:|---|
| **published — started and gated** | **24,576** | **45,056** | 7/7 gates on the ballasted 24 GiB-class proxy, `receipts/qualification-24gib-capped.json`. Physical-board gate still **open** |
| requirement at that length | 1.4269 GiB | 1.5219 GiB | `a·L + M`, §4.1 |
| headroom at the measured pool | 25.4 % | 70.8 % | against 1.79 / 2.60 GiB |
| **≥15 % envelope [P] — never started** | **24,576** | **65,536** | largest 4,096-multiple with `a·L + M ≤ pool/1.15`; exact bounds 28,574 and 69,150 |
| **engine ceiling [P] — never started** | **35,774** | **80,208** | `(pool − M)/a`: 1.00× concurrency, zero headroom |

For **MTP-3 the published length is the envelope**: 24,576 is simultaneously what we started and
the most our own headroom rule allows. That is why 32,768 had to go — it clears only 5.8 % — and
28,672 misses too, at 14.8 %.

**One caveat on 24,576, stated rather than buried.** That envelope is computed at the pool the
proxy measured, 1.79 GiB, which corresponds to the **18.19 GiB** resident figure this card loads
at. At the conservative **18.41 GiB** figure the pool is 1.58 GiB, the rule allows only 22,943, and
24,576 *fits* — 10.8 % — without clearing the margin. So: **a board that loads at 18.41 GiB should
serve 20,480 if it wants the full rule margin.** 24,576 is published because it is robust across
both of our measured resident figures where 28,672 (1.3 % at the conservative pool) and 32,768
(does not fit) are not, and because it is what the proxy started — not because every 24 GB board
will clear 15 % at it.

For **MTP off the published length sits well inside the envelope**: 49,152, 53,248, **57,344** and
61,440 all satisfy the ≥15 % rule at the measured pool, 57,344 with 36.9 % to spare. Every one of
them is arithmetically defensible and not one of them has been booted; 45,056 has. Raising the row
is a one-startup job, not a new artifact, and this document would rather under-publish a started
length than print an ungated one.

One live risk to all three tiers: none of these numbers was measured with `--enable-prefix-caching`,
and preliminary work on the physical 5090 indicates a pool merely *equal* to the window is not
sufficient once prefix caching is on. Cite `receipts/qualification-5090-apc.json` when it lands
rather than assuming these lengths transfer to that configuration.

Serving flags for M24 are the amendment's, with the length changed:

```bash
VLLM_EXL3_EMBED_BITS=8 VLLM_EXL3_GRAPH_DECODE=1 \
vllm serve <ctx-checkpoint> --max-model-len 24576 --max-num-seqs 1 \
  --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
```

M24-L is one conversion in the shape of `tools/build_ctx.sh`, which is where the per-module
width regexes live (exllamav3's own CLI exposes only a global width — Gap 1 in
[docs/04](04-exllamav3-toolchain.md)):

```bash
export EXL3_BITS_FIXED='{"^.*self_attn\\..*$": 5, "^.*linear_attn\\..*$": 5}'
export EXL3_BITS_OVERRIDE='{"^.*mlp\\.(gate|up|down)_proj$": 5}'
python convert.py -i /models/Qwen3.8-27B -o /work/qwen38-m24l -w /work/wd-m24l \
  -b 4 -hb 5 -mb 4 -vb 16 -cb mcg
```

`-vb 16` is not laziness: `-vb 6` splits the fused `visual.blocks.N.attn.qkv` and fails the
finalizer's topology check ([docs/30](30-iteration-4-context-edition.md) L108-112).

### 5.4 Supported-hardware tuple

Everything measured in this repository is **SM120, TP1** — but it is **not one driver tuple**, and
§10's numbers come from the first of these two. The physical RTX 5090 that qualified the context
edition and ran the 24 GiB proxy is on **driver 610.57.04, CUDA UMD 13.3**, host `aiboss`
(`receipts/qualification-5090-context.json` → `identity.host`); the rental RTX PRO 6000
measurements are on **595.58.03** (`receipts/release-evidence-context.json` → `hardware`).
The 24 GB class spans more than one
architecture — F3 makes that point specifically, that these boards "are not Blackwell-only, so
the supported-hardware tuple has to be stated, not assumed"
([docs/29](29-plan-and-loose-ends.md) F3). The budget arithmetic in §3 is
architecture-independent; the EXL3 kernels, the `FULL_DECODE_ONLY` graph path and the fp8 KV
path are not, and nothing here has run outside SM120. The card must print the exact board,
driver and arch it was measured on, must not generalise across architectures, and must not name
a board this project has not started.

## 6. Profile S16 — the 16 GB profile

16 GB leaves 12.70 GiB for weights plus KV. The hard result from the byte law:

**A multimodal 16 GB build cannot keep the MLP stack at 4 bits.** The cheapest such
configuration — `full_attention` K3, `linear_attention` K3, MLP all K4, `lm_head` K3, MTP K4,
int8 embedding, BF16 vision tower — has a resident payload of 13.235 GiB, 13.58 GiB with the
loader allowance, which is 0.88 GiB *over* budget before a single KV byte **[P]**. Either the
vision tower goes, or the fourth MLP bit goes. The MLP stack is 17.113 B params: one bit across
gate+up+down is 1.992 GiB, and it is the only lever large enough to matter at this size.

So S16 is a **sub-4-bit build**, at a width where this project has no fidelity measurement of
any kind. That is the headline of this section and it belongs on the card.

### 6.1 Bit allocation

| role | **S16-V (primary)** | S16-V-long | S16-T (text-only branch) |
|---|---|---|---|
| `full_attention`, 16 L | **K4** | K3 | K3 |
| `linear_attention`, 48 L | **K3** | K3 | K3 |
| `mlp_gate/up/down_proj` | **K3 / K3 / K3** | K3 / K3 / K3 | **K4 / K4 / K4** |
| `lm_head` | **K4/mcg** | K4/mcg | K3/mcg |
| `mtp_draft` | **all K4** | all K4 | all K4 |
| `embed_tokens` | BF16 disk → **int8 at load** | BF16 disk → int8 | BF16 disk → **int4 g128** |
| `vision_tower` | **BF16, 4.2 MP cap** | BF16, 4.2 MP cap | **absent** |

`full_attention` gets the extra bit over `linear_attention` because it is the cheapest place to
spend one: 0.195 GiB per bit against 0.644, and it is the part of the hybrid stack that carries
long-range retrieval. `lm_head` keeps K4 because the head is the one role with a measured
sensitivity anchor: serving a K6 head instead of the BF16 comparator head costs +0.000127
[+0.000105, +0.000148] on v3 (`receipts/v3-paired-head-asym-v2.json`), and the same order of
cost appears in the hydrated as-served comparison, +0.000125 [+0.000107, +0.000144]
(`receipts/v3-paired-hyd-head.json`).

### 6.2 Bytes — S16-V

| role | K | bytes | GB | GiB |
|---|---|---:|---:|---:|
| `full_attention` | 4 | 840,188,160 | 0.840 | 0.782 |
| `linear_attention` | 3 | 2,130,957,888 | 2.131 | 1.985 |
| `mlp_gate_proj` | 3 | 2,141,978,880 | 2.142 | 1.995 |
| `mlp_up_proj` | 3 | 2,141,978,880 | 2.142 | 1.995 |
| `mlp_down_proj` | 3 | 2,141,978,880 | 2.142 | 1.995 |
| `lm_head` | 4 | 636,206,084 | 0.636 | 0.593 |
| `mtp_draft` | 4 | 212,636,704 | 0.213 | 0.198 |
| `norms_and_small` | BF16 | 1,320,960 | 0.001 | 0.001 |
| `vision_tower` | BF16 | 921,460,192 | 0.921 | 0.858 |
| `embed_tokens` on disk | BF16 | 2,542,796,800 | 2.543 | 2.368 |
| **serialized total [P]** | | **13,711,503,428** | **13.712** | **12.770** |
| predicted disk bytes **[P]** | | 13,735,474,746 | 13.735 | 12.792 |
| resident payload, int8 embed **[P]** | | 12,440,105,028 | | 11.586 |
| **resident weights [P]** | | | | **11.94** |

| profile | serialized **[P]** | resident **[P]** | `T_max` MTP-3 | **published, MTP-3** | `T_max` MTP off | **published, MTP off** | `T_max` 4-bit KV, MTP-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S16-V** | 13.712 GB | 11.94 | 16,510 | **12,288** | 24,853 | **16,384** | 33,020 |
| S16-V-long | 13.502 GB | 11.74 | 22,363 | **16,384** | 31,374 | **24,576** | 44,727 |
| S16-T | 14.560 GB | 12.15 | 10,364 | 8,192 | 19,700 | 12,288 | 20,729 |

Read that table as the profile's real shape. **S16-V publishes 16,384 tokens with MTP-3 off**,
where the requirement is 0.70 GiB against 0.96 GiB available — 37 % headroom, the most
comfortable margin of any row here. With MTP-3 on, 16,384 needs 0.760 GiB against 0.764
available: a 0.5 % slack that the §3 constants can erase, so S16-V publishes 12,288 there.

> **Superseded by the Verdict above.** Under the measured KV law, S16-V publishes **28,672** with
> MTP off — more than twice the figure below, because the measured MTP-off fixed term is only
> 0.14 GiB and turning MTP off also frees the 0.45 GiB CUDA-graph pool and 0.10 GiB of draft
> activation — and it **cannot run MTP-3 at all**, its 0.63 GiB fixed term exceeding the whole
> 0.553 GiB pool by 0.077 GiB. S16-V-long publishes 36,864 without MTP and reaches only 764 tokens
> with it, so the one-bit trade below is a context trade, not a speculative-decode trade. Both
> lengths use the **memory** form of the ≥15 % rule (`a·L + M ≤ pool/1.15`); the token form
> `L ≤ L_max/1.15` over-publishes S16-V by a full 4,096 step. All predictions, and for a design
> study rather than a SKU.

**One `full_attention` bit is what buys MTP-3 at 16k.** S16-V-long spends 0.195 GiB by taking
`full_attention` from K4 to K3 and reaches 16,384 with MTP-3 at 27 % headroom, and 24,576
without. That is the sharpest trade in this document: one bit on 16 of 64 layers against
speculative decode at the advertised length. Which side is right depends on §8.2's retrieval
result at K3 attention, which is exactly why neither variant may be published first.

The route to 32,768 at this card size is the KV dtype, not more weight bits.

### 6.3 How to build it: budget, not width

Do not hand-pick uniform K3. exllamav3 allocates by a static priority order rather than by
measured error (Gap 1, [docs/04](04-exllamav3-toolchain.md)), but the error-driven path exists
post-hoc — `util/measure.py` over ≥2 quants plus the BF16 reference produces per-group
`dkld`/`dbits`, and `util/optimize.py` spends a **byte budget** greedily by that ratio, then
splices and recompiles. It can mix K3/K4/K5 but not BF16, which is exactly the S16 problem
shape. This is [docs/29](29-plan-and-loose-ends.md) rank 9, and S16 is the first profile that
genuinely needs it. Feed it these budgets, derived by inverting §2 and §4 **[P]**:

| context target, MTP-3 | KV needed | resident ceiling | payload ceiling | body budget (attention + MLP + head + MTP) | implied average |
|---|---:|---:|---:|---:|---:|
| 16,384, int8 embed + vision | 0.76 GiB | 11.94 | 11.59 | **10.255 GB** | 3.153 bpw |
| 32,768, int8 embed + vision | 1.32 GiB | 11.38 | 11.03 | **9.654 GB** | 2.968 bpw |
| 32,768, int8 embed, text-only | 1.32 GiB | 11.38 | 11.03 | **10.575 GB** | 3.251 bpw |
| 32,768, int4 embed + vision | 1.32 GiB | 11.38 | 11.03 | **10.270 GB** | 3.157 bpw |

Body params are 26,023,034,880, so `average bpw = body_bytes · 8 / 26,023,034,880`. Any
allocation meeting the budget is admissible; the optimizer decides where the bits go, and the
per-module proxy errors that `tools/build_ctx.sh` already tees are its input.

### 6.4 The honest 16 GB card line

Every reachable 16 GB point requires at least one knob for which no fidelity measurement exists:
sub-4-bit MLP, sub-4-bit attention, an int4 input overlay, or removal of the vision tower. The
nearest measured neighbour is the K4 build — MLP all-K4, attention online K6 — at 0.010604 on
the v5 suite, and its tail is already the worst of the five published candidates (p99.9 0.5555
against the context build's 0.1642, `receipts/kld5-1M-tail-{k4,ctx}.json`). Measured against
*that* neighbour, S16-V is one bit lower on each of gate/up/down, three lower on the
linear-attention stack (K6 online → K3), two lower on full attention (K6 online → K4) and two
lower on the head (K6 → K4) — every one of them in the direction that made K4 the worst
published candidate. The result could plausibly land anywhere in 0.03-0.10 **[P, extrapolation
only: the ladder has no point below 4 bits, so this is a range with a shape, not an estimate]**.
A card that prints a number here without §8.2 having passed is fabricating it.

## 7. Knobs, ranked

GiB of resident weights (or of KV budget where noted) released per +0.001 of mean KLD. Ratios are
written as `GiB / ΔKLD-in-units-of-0.001` so every one can be checked in one division. Absolute
KLD is suite-specific — the same K4 checkpoint reads 0.029679 on the corrected v3 subset and
0.010604 on v5 ([docs/33](33-evidence-volume-and-intervals.md)) — so v3 and v5 columns are not
interchangeable and are never mixed inside one ratio.

| knob | releases | ΔKLD, v3 suite | GiB / 0.001 (v3) | ΔKLD, v5 10M suite | GiB / 0.001 (v5) | receipt |
|---|---:|---:|---:|---:|---:|---|
| int8 input-embedding overlay | 1.184 GiB | +0.000065 [+0.0000046, +0.00013] | **18.2** = 1.184/0.065 | not rerun | — | [docs/32](32-native-context-embedding-overlay.md) L41-43 |
| image cap 16.8 → 8.4 MP | 0.72 GiB of **KV budget** | 0 by construction | ∞ | 0 | ∞ | [docs/32](32-native-context-embedding-overlay.md) L97-100 |
| MTP-3 off | 0.20-0.26 GiB weights **and** 11.4 % of every KV token | 0 (protocol never engages MTP) | ∞ | 0 | ∞ | §4; costs ~half of single-stream decode ([docs/23](23-next-attack-list.md) L89) |
| attention K6 → K5, 64 L | 0.840 GiB body, 0.852 including the MTP attention | +0.003978 [+0.003055, +0.005057] **online, uncalibrated** | 0.21 = 0.840/3.978 | +0.000750 **serialized, calibrated, unpaired** (0.003509 − 0.002760) | **1.14** = 0.852/0.750 | `receipts/v3-paired-attn-k5-vs-k6.json`; `receipts/kld5-10M-{hyd,ctx}.json` |
| `lm_head` K6 → K5 | 0.148 GiB | no K5 measurement; K6-vs-BF16-head anchor +0.000127 | ~0.6 **[P]** = 0.148/0.25, assuming K5 costs 2× the K6 anchor | none | — | `receipts/v3-paired-head-asym-v2.json` |
| MLP → K4 (gate 1 + up 1 + down 2 bits) | 2.656 GiB = 4 × 0.664 | none | — | +0.007394 **unpaired** (0.010604 − 0.003210) | **0.359** = 2.656/7.394 | `receipts/kld5-10M-{k4,k5k6}.json` |
| attention K6 → K4, 64 L | 1.680 GiB = 2 × (0.195 + 0.645) | +0.019373 [+0.014947, +0.024509] **online** | 0.087 = 1.680/19.373 | none | — | `receipts/v3-paired-attn-k4-vs-k6.json` |
| `mlp_down_proj` K6 → K5 alone | 0.664 GiB | none | — | none | — | **no measurement exists** |
| anything at K3 | 0.148-1.992 GiB per bit, by role (§2) | none | — | none | — | **no measurement exists at any width below 4 bits** |
| int4 input overlay, beyond int8 | 0.574 GiB | none | — | none | — | tensor-domain error 13.1× int8 (`receipts/embedding-quant-error.json`); the earlier int4 embedding claim was retracted ([docs/32](32-native-context-embedding-overlay.md) §"Correction") |
| CPU-resident embedding, beyond int8 | 1.184 GiB | zero weight error | ∞ | zero | ∞ | **not implemented**: the pinned patch adds only the quantized-embedding hook (`tools/vllm-qwen3_5-embed-quant-config.py`); per-token host gather latency unmeasured |
| vision tower bit width | 0.858 GiB if it worked | — | — | — | — | **unavailable**: `-vb 6` breaks fused-qkv topology; do-not-reopen ([docs/29](29-plan-and-loose-ends.md) L577) |
| fp8 → 4-bit KV | ~50 % of KV bytes | — | — | — | — | flags exist in the pinned r34 build; **never measured**, and the fidelity protocol runs `kv_cache_dtype_resolved: bfloat16` (`receipts/kld5-10M-*.json` → `reference_identity`), so it cannot see KV error at all |

Reading of the ranking:

1. **The int8 input overlay is two orders of magnitude better than any weight-width knob** and
   should be on by default in every profile below 32 GB. It is already the reason native context
   fits at 32 GB.
2. **Zero-KLD knobs come next**: image cap, MTP-3, single-stream. They cost capability and
   throughput, both of which a card can state precisely.
3. **Then attention width, and only when serialized and calibrated.** The v3 attention ablations
   were online, calibration-free encodings; on v5 the serialized-versus-online gap at equal
   width is a paired −0.000450 [−0.000469, −0.000433], 4,922/5,120 contexts
   (`receipts/kld5-10M-paired.json` → `hyd_vs_k5k6`). The two attention numbers therefore
   describe different mechanisms, and the cross-suite ratio between them is not a legitimate
   quantity — which is why there is no "attenuation factor" column here.
4. **MLP width is the worst measured buy per GiB** and the only lever large enough for 16 GB.
   That tension, not arithmetic, is what makes the 16 GB class hard.
5. **The 4-bit KV dtypes are the cheapest unexplored lever** for both classes, and the one whose
   measurement our current protocol cannot supply. It needs its own sweep: allocated tokens,
   needle retrieval at native length, and a KLD run with the KV dtype actually pinned.

## 8. Acceptance tests

No profile may be published before its test passes. Each test is a startup, a fidelity run, a
retrieval run and a throughput run, on the board class the card names, with receipts.

### 8.1 The smallest KLD suite that is meaningful

**Shard 0 of the v5 suite: 512 contexts × 2,047 = 1,048,064 scored positions, 330 source
clusters** (`receipts/kld5-1M-tail-ctx.json`). Not smaller, and no need for larger, because:

- the interval is governed by document diversity, not position count — relative 95 % width for
  the context build is 15.1 % at 1M and 18.0 % at 10.48M
  ([docs/33](33-evidence-volume-and-intervals.md), `receipts/kld5-ladder-convergence.json`), so
  the extra nine shards buy confirmation of the mean, not resolution;
- shard 0 is the **only** shard with published tail histograms for all five existing candidates
  (`receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json`), so a new profile gets p50/p95/p99/p99.9
  and exact exceedance counts against every published sibling without re-running any of them.

Protocol, unchanged: `tools/fidelity.py` captures hidden states at the final norm and replays
both operands through the one shared BF16 head (`head_sha256`
`25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff`), full-vocabulary
`KL(BF16 || candidate)`, float64 accumulation; `tools/kld_aggregate.py` writes the receipt;
per-context pairing against the published context build with a 10,000-resample bootstrap over
the 330 clusters. Report the mean, the interval, top-1, and the tail row. The replay resolution
floor is **0.000654** (`receipts/release-evidence-hydrated.json` →
`evaluation.controls.live_vs_replay_floor`): any delta below ~1e-3 must be reported as "at or
below the replay floor", not as a number.

### 8.2 Per-profile gates

| gate | M24 (context body) | M24-L | S16-V |
|---|---|---|---|
| new conversion needed | no | yes | yes |
| startup on the named board, `nvidia-smi` before and after | required | required | required |
| resident weights, activation, non-torch, graph, KV pool and allocated tokens read out of the engine log | required; **must** land ≥ the published length | required | required |
| KLD, §8.1 suite, paired vs context build | expected ≈ 0.003509, since it is the same checkpoint — a deviation is a harness bug | **must** be reported even if at the replay floor | **hard gate: no fidelity number may be quoted from any other build** |
| tail row p50/p95/p99/p99.9 + exceedance above 0.1 and 1.0 | required | required | **required; a p99.9 above the K4 build's 0.5555 blocks publication of a general-use claim** |
| needle retrieval, `tools/longctx.py`, depths 0.1 / 0.5 / 0.9 | at ≥0.98 × 24,576 with MTP-3 and ≥0.98 × 45,056 without, token-counted before submit | at 45,056 / 81,920 | at 28,672 with MTP off; **MTP-3 is withdrawn at this class** (Verdict, §5.3), so there is no MTP-3 needle run to make — run S16-V-long at its own published length instead, to settle §6.2's one-bit trade |
| combined long-text + image, `tools/longmm.py` | one request ≥0.9 × published length carrying a ≥7 MP image, exact code **and** colours | same | at the 4.2 MP cap; if it fails, the cap or the multimodal claim goes |
| vision suite, `tools/vision_eval.py`, 30 deterministic cases | ≥ the 24/30 measured at 8.4 MP | ≥24/30 | report; **do not** assume 24/30 transfers to a K3 body |
| throughput, `tools/bench.py`, warmed, ≥3 timed runs | single-stream with MTP-3 and off; sampler, temperature and reasoning effort printed | same | same |
| non-termination / repetition check | required | required | **required and weighted**: two independent Q4 reports describe looping to context exhaustion ([docs/29](29-plan-and-loose-ends.md) F5), which mean KLD cannot see |
| MTP acceptance at pinned temperature | report | report | report; a K3 draft may not accept |
| receipt | new `release-evidence-m24.json` with the same field set as `release-evidence-context.json`, plus a row in `receipts/collection-index.json` | same | same |

Rules that apply to all three, in the F4 spirit:

1. **No fit claim without a startup.** "Predicted 29,306 KV tokens" and "allocated 33,760 KV
   tokens" are different sentences — §10 produced exactly that pair — and only the second may
   appear on a card.
2. **The budget caveat travels.** The 32 GB result is an engine-budget proof on a 95.6 GiB card,
   not a physical hard-limit proof (`receipts/native-mtp-8mp-amendment.json` →
   `claim_scope`). A 24 GB or 16 GB profile emulated the same way inherits exactly the same
   caveat, and the physical rerun stays P0.
3. **The context length goes in the same table as the fidelity number**, per F3, together with
   the KV dtype, whether MTP is on, the image cap, and `max_num_seqs`. A context length without
   those five is not interpretable.
4. **Serialized bytes are never called VRAM**, and the KV figure is never folded into the weight
   figure.

## 9. What this document does not establish

- No 24 GB or 16 GB board has been started. Every context length here is arithmetic — except the
  two 24 GB-class lengths, which §10 now measures on a **proxy**, not on a board.
- The two card constants — 0.98094 free-at-startup and 0.97 utilisation — are transferred from
  one board's note and are the likeliest source of error in §3. §10 replaces the first with a
  measured mechanism: the engine multiplies utilisation by `cudaMemGetInfo` total, which is
  nvidia-smi's FB Total **minus** its FB Reserved, and never by the free figure.
- The GDN conv/SSM state is no longer carried as a modelling assumption: §4.1 *measures* the fixed
  per-request term at **0.63 GiB** with MTP-3 and **0.14 GiB** with MTP off, retiring the 0.20 GiB
  reserve this document used to charge. What is still unestablished is the **itemisation** — the
  engine reports one pool, so the split of `M` between recurrent state, draft KV and page padding
  is inferred rather than measured, and only the MTP-3-minus-MTP-off difference (0.49 GiB) is
  attributable with confidence.
- No sub-4-bit width of this architecture has ever been measured for fidelity, which makes S16 a
  design and not a product.
- The 4-bit KV dtypes are unmeasured, and the current fidelity protocol resolves the KV cache to
  bfloat16, so it cannot measure them even for the profiles that already ship fp8 KV.
- Activation, non-torch and CUDA-graph figures are carried unchanged from a 32 GB profile; a
  16 GB profile with a lower image cap and fewer batched tokens should measure its own.

## 10. Proxy qualification at a capped 24 GiB-class budget

> **This is a proxy, not a qualification, and it does not close §8's physical-board gate.** No
> 24 GB board was used. The receipt is
> [`receipts/qualification-24gib-capped.json`](../receipts/qualification-24gib-capped.json), whose
> `physical_board_gate` field says **open** on purpose. Nothing here licenses "fits on 24 GB" on a
> card, in a table, or in a sentence.

A real 24 GiB board is unavailable, so cap what the engine may take on the physical RTX 5090 the
Verdict's constants come from, and run that qualification's own gates 1-7 at the capped budget.
Two arms, because they answer different questions, and the difference between them turned out to be
the most useful number here.

**How the budget is set.** Read out of the pinned image rather than assumed
(`vllm/v1/worker/utils.py` `request_memory`): `requested = ceil(cudaMemGetInfo_total × utilisation)`,
and startup refuses if free < requested. On this card `cudaMemGetInfo` total is nvidia-smi's FB
Total 32,607 MiB minus its FB Reserved 458 MiB = **32,149 MiB = 31.3955 GiB**, confirmed to the
byte from the engine's own two `--kv-cache-memory-bytes` hints. The KV pool is
`requested − non_KV − CUDAGraph`, with **no** free-memory term — so utilisation sets the budget and
only a ballast can shrink the *card*.

| arm | utilisation | budget the engine printed | card | what it emulates |
|---|---:|---:|---|---|
| **budget** | 0.7635 | **23.97 GiB** | untouched 32 GiB | a 24 GiB board's budget only — 6.93 GiB of card stays free |
| **board** | 0.7164 | **22.49 GiB** | ballasted to 23.553 GiB free | a 24 GiB board's budget **and** card, at the qualified 0.955 |

The board arm's 22.49 GiB is exactly what
[`receipts/vram-class-verdict.json`](../receipts/vram-class-verdict.json) → `card_budget_model`
derives for a 24 GiB board at 0.955, and its engine reported **23.06 GiB** free at its own init —
that board's 23.5527 GiB CUDA total less the engine's measured 0.496 GiB context. Utilisation
0.7643 was tried first and rejected: it prints "24.0 GiB", which is 23.995 rounded to the engine's
two decimals and therefore evidence of nothing.

### 10.1 Both published lengths pass, and one of them passes on the edge

All seven gates pass in all four configurations — startup, needle at the profile's own length,
combined long-text-plus-7 MP image, the 30-case image suite, three warmed decode runs, a second
long request in the same process after release, and receipt completeness.

| profile | arm | KV pool | tokens allocated | concurrency | KV needed per request | KV headroom | decode |
|---|---|---:|---:|---:|---:|---:|---:|
| 32,768 MTP-3 | budget | 3.27 GiB | 62,557 | 1.91x | 1.6925 GiB | 93.2 % | 108.9 tok/s |
| 32,768 MTP-3 | **board** | **1.79 GiB** | **33,760** | **1.03x** | 1.6925 GiB | **5.8 %** | 98.96 tok/s |
| 45,056 MTP off | budget | 4.08 GiB | 119,680 | 2.66x | 1.5219 GiB | 168.1 % | 47.47 tok/s |
| 45,056 MTP off | **board** | **2.60 GiB** | **76,032** | **1.69x** | 1.5219 GiB | **70.8 %** | 47.76 tok/s |

**45,056 with MTP off is safe**, and the published length is conservative: 70.8 % headroom, the
engine supports 80,208 tokens at that pool, and it survives the conservative resident-weight figure
too (73,155 supported at 18.41 GiB resident). 65,536 is the largest 4,096-multiple that would still
clear §5.3's ≥15 % rule.

**32,768 with MTP-3 is a pass nobody should ship.** Three margins are thin:

1. **5.8 % KV headroom**, against the **≥15 %** rule §5.3 applies to every other published length.
   The largest 4,096-multiple clearing that rule at the measured pool is **24,576**.
2. It **fails outright under this project's other resident-weight figure.** At the 18.19 GiB this
   card measures, the pool is 1.802 GiB and the engine supports 36,135 tokens; at the **18.41 GiB**
   `receipts/native-mtp-8mp-amendment.json` measured as-run — the figure the Verdict deliberately
   preferred because it is larger — the pool is 1.582 GiB and the engine supports **29,350**, so
   32,768 does not fit. The published length sits *between* our own two measurements.
3. The combined text-plus-image gate passed with **68 MiB** of a 24 GiB board's 24,118 MiB unused.

**Superseding note, not a rewrite of history:** publish **24,576** with MTP-3 for this class rather
than 32,768; §5.3.1 carries the full tiering. 24,576 needs **1.4269 GiB**. Against the measured
1.79 GiB pool that is **25.4 %** headroom, which clears the 15 % rule and makes 24,576 the largest
4,096-multiple that does. Against the conservative 1.58 GiB pool it still fits outright, with
**10.8 %** headroom measured in memory — or 19.3 % if counted in tokens against the 29,306 the
engine supports there; the two conventions differ once `M ≠ 0` and this document uses the memory
one. 32,768 was correctly derived from the model available when it was written; §10.2 is why that
model was wrong, and the Verdict's own live-risk paragraph had already flagged it.

### 10.2 The KV law, measured instead of bounded

§4's `H + c_tok·T` is retired. The engine sizes capacity as
`int(max_concurrency × max_model_len)` with
`max_concurrency = num_blocks / cdiv(max_memory_usage_per_request, memory_per_block)`
(`vllm/v1/core/kv_cache_utils.py`), so the quantity that scales is the per-**request** cost at the
configured window. Four startup refusals print that cost directly, and two windows per
configuration pin it with no regression at all:

```
kv_needed(L) = a·L + M                 reported tokens = int(L · pool / kv_needed(L))
MTP-3 :  a = 34,816 B/token = 32,768 × 17/16 exactly,   M = 0.63 GiB
MTP off: a = 32,932 B/token = 1.005 × the fp8 floor,    M = 0.14 GiB
```

`a` for MTP-3 lands exactly on the engine's own "may waste at most 6.25 % KV cache memory" padding
line; `a` for MTP off matches §4's independently derived 32,929 B/token to 0.009 %. Inverting the
law reproduces the engine's own "estimated maximum model length" to within its printed precision
(81,600 tokens needs 3.2759 GiB against a pool printed as 3.27; 128,576 needs 4.0834 against 4.08).
All points `max_num_seqs` 1, fp8 KV, same image and revision as the qualification:

| config | window | KV needed per request | pool on hand | engine's own max length for that pool |
|---|---:|---:|---:|---:|
| MTP-3 | 131,072 | 4.88 GiB | 3.27 GiB | 81,600 |
| MTP-3 | 262,144 | 9.13 GiB | 3.27 GiB | 81,600 |
| MTP off | 131,072 | 4.16 GiB | 4.08 GiB | 128,576 |
| MTP off | 262,144 | 8.18 GiB | 4.08 GiB | 128,576 |

**MTP-3 is the whole difficulty at this class.** It costs 0.49 GiB of *fixed* KV per request plus
5.75 % on every token, on top of 0.25 GiB of draft weights **and** 0.45 GiB of CUDA-graph pool that
vanishes entirely when MTP is off — which no arithmetic here had charged to MTP. That fixed term
alone is 35 % of the 1.79 GiB pool a 24 GiB board has.

### 10.3 What a capped budget does not change, and the one thing it does

The four memory components are **invariant to the budget**, which is the direct test of whether
this proxy distorts them:

| budget | weight | activation | non-torch | CUDAGraph | sum | KV pool |
|---|---:|---:|---:|---:|---:|---:|
| 29.98 GiB — physical qualification, MTP-3 | 18.19 | 1.78 | **0.27** | **0.45** | 20.69 | 9.28 |
| 23.97 GiB — budget arm, MTP-3 | 18.19 | 1.78 | **0.27** | **0.45** | 20.69 | 3.27 |
| 22.49 GiB — board arm, MTP-3 | 18.19 | 1.78 | **0.27** | **0.45** | 20.69 | 1.79 |
| 23.97 / 22.49 GiB — both arms, MTP off | 17.94 | 1.68 | 0.27 | 0.00 | 19.89 | 4.08 / 2.60 |

Non-torch and CUDA-graph memory are identical to the full-budget qualification's 0.27 and 0.45 GiB
at every budget: **every byte the cap removes comes out of the KV pool and nothing else.**

What the cap *does* change is decode, indirectly and only with MTP: 98.96 tok/s in the board arm
against 108.9 in the budget arm, a 9.1 % loss, while MTP draft acceptance falls from 0.5556 to
0.4822 and mean accepted length from 2.6667 to 2.4465. Those ratios agree to a fifth of a percent
(0.9087 throughput, 0.9174 accepted length), so **the decode loss is the acceptance loss**, not a
bandwidth effect — and acceptance is monotone in pool size across three independent pools (0.5648
at 9.28 GiB, 0.5556 at 3.27, 0.4822 at 1.79). With MTP off the two arms decode at 47.47 and 47.76
tok/s, the smaller budget marginally faster, i.e. noise. Capping a budget costs nothing by itself;
shrinking the KV pool appears to cost speculative acceptance, and that deserves its own experiment.

### 10.4 Why the budget-only arm is not a proxy for a board, with a number

A combined text-plus-image request peaks **above** the engine's budget, because the vision tower's
transient is not part of the profiled budget — by 0.89 to 1.02 GiB in all four runs. That is what
makes the outside-budget slack decisive, and it is why F3-open-2 in `vram-class-verdict.json` was
right to say a budget carved out of a larger card cannot see this failure mode:

| arm | engine peak during the image gates | a 24 GiB board's CUDA total | margin |
|---|---:|---:|---:|
| budget | 25,614 MiB | 24,118 MiB | **−1,496 MiB** |
| board | 24,050 MiB | 24,118 MiB | **+68 MiB** |

**The budget arm passed the image gates on 1,496 MiB no 24 GiB board owns**, so a budget-only cap is
not a proxy for a board at all — now a measurement rather than an argument. The board arm passed by
68 MiB, 0.3 % of the board.

### 10.5 Residual risk, stated plainly

A physical 24 GiB board differs from this capped 32 GiB board in exactly two device properties,
both measured here and neither emulable: the driver's framebuffer reserve (**458 MiB** on this card)
and the CUDA context (**0.496 GiB**). Non-torch memory, the component most likely to move with them,
is 0.27 GiB here and 0.27 GiB in the full-budget qualification.

- A board reserving 68 MiB more than this one, or with a context 68 MiB larger, **fails** the
  combined text-plus-image gate at 32,768 with MTP-3 — a 0.3 % perturbation, and the most fragile
  number in the receipt.
- A 0.10 GiB rise in any non-KV component, or the 0.22 GiB gap between our own two resident-weight
  figures, takes 32,768 below 1.0x concurrency and the engine refuses to start.
- 45,056 with MTP off is fragile on neither count for KV (70.8 %) and carries the same order of
  vision margin (120 MiB), since the vision transient depends on `max_pixels`, not the window.

One methodological caveat: the board arm needs two CUDA contexts on one device — ballast plus engine
— and this card runs `Exclusive_Process`, so the first attempt died at ~25 s with
`cudaErrorDevicesUnavailable`. Compute mode was set to `DEFAULT` for that arm and back to
`EXCLUSIVE_PROCESS` afterwards. It gates context creation, not allocation or kernel behaviour, and
the two arms' weight, activation, non-torch and CUDA-graph figures are identical to the hundredth of
a GiB across the change.

**§8's physical-board gate remains open.** This section removes the arithmetic risk from two
published lengths and replaces it with a measured allocation, and it answers the "budget carved out
of a bigger card" objection by ballasting the card down as well. It cannot manufacture another
board's reserve or context, and at 68 MiB of margin that is not a rounding detail.

## 11. Card text for the 24 GB class — ready to lift

The four model cards are being landed centrally and this document does **not** edit them. The
block below is the 24 GB-class paragraph for the **context edition's** card, written to be copied
verbatim. It is short on purpose, labelled on purpose, and it never uses the word "fits". It
satisfies §8.2 rule 3: the window never appears without its KV dtype, its MTP setting, its image
cap and its `max_num_seqs`.

<!-- ===== BEGIN CARD BLOCK — 24 GB class, context edition — lift verbatim ===== -->

### 24 GB class

This checkpoint has a 24 GB-class serving profile. **No new conversion is required** — it is the
same weights, so every fidelity number above carries over unchanged.

| profile | context window | KV dtype | MTP | `max_num_seqs` | image cap | KV pool required |
|---|---:|---|---|---:|---:|---:|
| speculative | **24,576** | fp8 | 3 draft tokens | 1 | 8.4 MP | 1.43 GiB |
| non-speculative | **45,056** | fp8 | off | 1 | 8.4 MP | 1.52 GiB |

Both windows are **predictions**, not measurements on a 24 GB board. They were validated on a
**capped-budget proxy**: an RTX 5090 whose engine budget was restricted to 22.49 GiB *and* whose
card was ballasted down to 23.55 GiB free — a 24 GiB board's budget and its card together. All
seven qualification gates passed at both windows: startup, needle retrieval at the profile's own
length, a combined long-text plus 7 MP image request, the 30-case vision suite, warmed throughput,
a second long request in the same process, and receipt completeness. **The physical-board gate
remains open.** No 24 GB board has been started, and until one is, "predicted" and "allocated"
are different words.

**A budget cap on a big card is not a substitute for a small board.** The control arm, which
capped the engine's budget but left the card whole, peaked 1,496 MiB *above* the total memory a
24 GB board has — it passed the image gates on memory no such board owns, which is precisely why
the ballasted arm exists and why this profile is labelled a proxy.

Serving flags — the qualified 32 GB configuration with the window changed and nothing else:

```bash
VLLM_EXL3_EMBED_BITS=8 VLLM_EXL3_GRAPH_DECODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve <ctx-checkpoint> --gpu-memory-utilization 0.955 \
  --max-model-len 24576 --max-num-seqs 1 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
```

For the non-speculative profile, drop `--speculative-config` and set `--max-model-len 45056`.

Evidence: `receipts/qualification-24gib-capped.json` (proxy qualification, `physical_board_gate:
open`) and `receipts/vram-class-verdict.json` (class decision and arithmetic).

<!-- ===== END CARD BLOCK ===== -->

Three notes for whoever lands it, **not** part of the block:

1. **Do not raise 45,056 on this evidence.** It is the window that was started. The ≥15 % envelope
   reaches 65,536 and the engine ceiling is 80,208 (§5.3.1), but neither has been booted.
2. **Do not add a 24 GB row to the hydrated edition's card.** Under the measured law that body has
   a 0.72 GiB pool against a 0.63 GiB fixed MTP-3 term — 2,783 tokens — so its MTP-3 row is
   withdrawn, not shortened (§5.3).
3. **If the card names a board, it must name the one that was measured**, which is an RTX 5090 at
   SM120 / TP1 / driver 610.57.04 with CUDA UMD 13.3, capped — not any 24 GB product, and not
   the 595.58.03 the rental measurements used. §5.4 is the rule.
