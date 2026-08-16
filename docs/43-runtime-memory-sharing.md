# 43 — Runtime memory sharing: what "sharing H (suh/svh)" is, what it is worth here, and where the recoverable GiBs actually are

Receipt: `receipts/h-sharing-research.json`. Every byte figure below is derived from our own
manifests/safetensors headers (ranged HTTP reads of `malaiwah/Qwen3.8-27B-EXL3-K5K6-context`,
no weight-shard downloads) or quoted from a named receipt. Every mechanism claim carries a
rootfs `file:line` (pinned image `voipmonitor/vllm@sha256:820181fb…`, r34) or a fork URL with a
verbatim quote. Nothing here was run on a GPU **except lever R1's outcome**, added 2026-08-16 from
`receipts/scratch-arena.json` after the physical-5090 A/B; §3 keeps both the original
\[INFERENCE] figure and the measured one.

## 1. What "sharing H (suv/suh)" actually was

The term in circulation is **`shared_h_v1`**, a rank-sliced EXL3 **checkpoint artifact layout**
for MoE experts, built for GLM-5.2 on the vLLM-GG fork. (The owner's "suv" spelling matched
nothing: search `suv` returns 0 hits in both `local-inference-lab/vllm` and
`local-inference-lab/b12x`; the tensors are `suh`/`svh`.)

Primary sources:

- **vLLM-GG PR #225** — "[GG] EXL3: consolidate mixed execution, MXFP8 overlay, and shared
  rotations" (closed, not merged; superseded by #228):
  <https://github.com/local-inference-lab/vllm/pull/225>
  > "Gate/up SU and down SV are allocated as one physical `[1,H]` row and broadcast by
  > stride/pointer contract without expansion."
  > "For GLM-5.2, the shared-H layout removes 672.36 MiB/GPU of duplicated persistent
  > rotations across 75 MoE layers."
  > "The shared-H loader is explicit and backward compatible. A complete newly encoded
  > shared-H checkpoint still requires its own KLD and E2E release validation."
- **vLLM-GG PR #228** — "[GG] consolidate EXL3 runtime and prewarm mixed-Trellis routes"
  (open): <https://github.com/local-inference-lab/vllm/pull/228>
  > "shared-H checkpoint loading without expanding one rotation row per expert"
  > "Physical shared-H saving: 672.36 MiB/GPU at MTP0 and 681.33 MiB/GPU at MTP3."
- **b12x PR #117** — "fix(trellis): make mixed expert counts runtime-dynamic" (**merged** to
  master): <https://github.com/local-inference-lab/b12x/pull/117>
  > "Mixed Trellis K3/K4 expert counts and shared-H rotation flags are checkpoint [properties]"
  > "`206/50`, shared-H on both tiers, zero relative error, cosine 1.0."

**What it is:** a conversion-time constraint *plus* a loader/kernel change, not a runtime flag.
The checkpoint must declare `rotation_layout: "shared_h_v1"` with the exact
`shared_h_tensor_schema` `model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}`
(rootfs `vllm/model_executor/layers/quantization/exl3.py:351-365`, validation `:1590-1613`,
fail-closed name checks `:2124-2159`). The loader binds one `[1,H]` row per (layer, projection,
side) instead of one per expert; the b12x kernels index it with expert stride 0
(`b12x/moe/_shared/kernels/w4a16/prepare.py:1860-1862`: "(1, hidden_size) is a broadcast row
shared by all experts (kquant shared-su artifacts); kernels index it with expert stride 0";
`broadcast_suh`/`broadcast_svh` flags: `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py:74-76,
654-656, 1370-1372`, `kernel.py:5968-6522, 7757-7838`). **All of this code ships in our pinned
r34 rootfs** even though PR #228 is unmerged (the release branch integrates outside PR merges).

**Why it saved 672 MiB on GLM-5.2:** GLM-5.2 is a ~160-expert × 75-MoE-layer model; per-expert
`suh`/`svh` rows duplicate the same rotation across every expert of a projection. Sharing
collapses E rows to 1 per (layer, projection, side).

**Kimi K3 is a different mechanism:** its path is Fruit **QSRT `qsrt_atoms_v1`** ("`H=1024`,
`I=512`, 96 fixed atoms per expert", "physical atom rotation without reconstructing dense
expert weights" — b12x PR #129 <https://github.com/local-inference-lab/b12x/pull/129>, vLLM PR
#269 <https://github.com/local-inference-lab/vllm/pull/269>). Canonical-atom dedup, not
`shared_h_v1`. No Kimi shared-H PR exists (queries listed in the receipt).

## 2. What it is worth on OUR checkpoints: 13.69 MiB ceiling, 0 B today

From the safetensors headers of the qualified context edition (409 EXL3 modules, 818 vectors):

| field | count | total bytes | MiB |
|---|---:|---:|---:|
| `suh` (fp16, rank-1, len K) | 409 | 5,928,960 | 5.65 |
| `svh` (fp16, rank-1, len N) | 409 | 8,424,448 | 8.03 |
| **combined** | **818** | **14,353,408** | **13.69** |

Largest single groups: 130× `svh[17408]` (4.32 MiB), 278× `suh[5120]` (2.71 MiB); the lm_head
`svh[248320]` is 0.47 MiB. Full histogram in the receipt.

**Byte-identical dedup: zero.** All 818 vectors were ranged-read and SHA-256 hashed:
**0 duplicate groups**. A pure loader dedup saves 0 bytes today, and even converter-cooperating
sharing is capped at <13.69 MiB — noise against an 18.19 GiB weight footprint. The prior holds:
per-expert duplication is the entire GLM/Kimi saving, and a dense model has no experts.

Two byte-level facts sharpen this:

- The stored vectors are **not ±1 signs**: layer-10 `gate_proj.suh` has 260 unique magnitudes
  (~0.01), `up_proj.suh` 297. The converter folds per-input-channel RMS scales into `su`
  (`exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py:1207-1210`:
  `su = (su * in_channel_scales / (-codebook_scale) + 1e-10)`), and `sv` is generated per
  module (`quantize.py:1306`).
- The **sign pattern is already shared** where inputs coincide: gate.suh and up.suh of the same
  layer agree in sign at 5120/5120 positions (1.0000). The converter draws one random sign
  vector per Hessian capture (`quantize.py:890`: "Random sign flips for input channel, fixed
  for the first linear layer to quantize with this H") — so same-input modules already share H
  by construction; only the per-module scale magnitudes differ, which is why no two vectors are
  byte-identical and why "sharing" would have to discard per-module scaling to save anything.

**Loader side (dense path, our case):** `suh`/`svh` are registered per shard
(`exl3.py:2606-2610`), validated as fp16 rank-1 of length K/N (`:2774-2785`), TP-sliced in
place (`:2839-2919`; fused-QKV tuple shards already share one `suh` across q/k/v components at
`:2897`), and consumed by pointer in the kernels: `_exl3_gemm` (`:977-1007`),
`_b12x_trellis_linear_out` (`:1263-1305`), and the prefill reconstruct path's
`ext.had_r_128(..., suh/svh, ...)` (`:891, :925, :949, :960`). b12x `prepare_weight` is
**zero-copy** ("No trellis or rotation bytes are copied, permuted, stacked, or concatenated",
`b12x/moe/_shared/kernels/w4a16/prepare.py:2059-2060`), cached per-module on the trellis tensor
(`exl3.py:1181-1198`). CUDA-graph capture bakes the per-module device pointers into replayed
kernels and allocates op outputs from the graph pool; serialized dense EXL3 stays out of decode
graphs by default (`VLLM_EXL3_GRAPH_DECODE=0`, `exl3.py:1027-1036`) and when enabled is
pre-autotuned per capturable row count (`:1090-1169`). No copies of suh/svh are made anywhere.

**Verdict: dead end. Do not build.** Sharing needs converter cooperation to save anything, the
ceiling is 13.69 MiB, and it carries the fidelity risk in §4.

## 3. Where the recoverable GiBs actually are (dense path, from rootfs source)

Per-module allocations at serve time are: the weights themselves, the 4-byte `dummy_scale` per
prepared K6 module (`prepare.py:2123-2124`), and nothing else. The real overhead is **per
geometry**, and it is the prefill reconstruct scratch:

- `_EXL3_RECONSTRUCT_SCRATCH`: one persistent fp16 `(K, min(N, 32768))` buffer per distinct
  geometry, never freed (`exl3.py:766-767, 785-795`; the code's own comment prices "336 MB for
  a 5120x32768 chunk"). Reconstruct dispatch is **on by default** above 128 prefill rows
  (`:770-782`), for both the generic path (`:986-993`) and K6/MCG (`:1287-1294`).

Our checkpoint has 9 distinct geometries (census in the receipt). The buffers that allocate at
the qualified profile (every prefill-active geometry; the head's 5120×32768 chunk needs ≥128
sampled logit rows and does not trigger at max_num_seqs 1–8 with MTP-3):

| geometry (K×chunk) | MiB |
|---|---:|
| 5120×17408 (gate/up) | 170.0 |
| 17408×5120 (down) | 170.0 |
| 5120×12288 (q) | 120.0 |
| 5120×10240 (GDN in_proj) | 100.0 |
| 10240×5120 (mtp.fc) | 100.0 |
| 5120×6144 (GDN ba) | 60.0 |
| 6144×5120 (o/out) | 60.0 |
| 5120×1024 (k/v) | 10.0 |
| **sum (allocating)** | **790.0** |
| head 5120×32768 (only if ≥128 logit rows) | +320.0 |

**Lever R1 — one shared scratch arena. Built, measured on a GPU, and PR'd (2026-08-16).** Layers
execute serially per rank, so one byte arena of the largest live chunk (170 MiB) with per-geometry
typed views replaces 790 MiB of persistent buffers. This section originally priced that at
**~620 MiB recovered ≈ +18,700 tokens MTP-3** \[INFERENCE — arithmetic from source at the measured
law `receipts/qualification-5090-context.json`, MTP-3 a = 34,816 B/token; needs one 5090 run to
confirm it lands in the KV pool]. **That run happened, and the row is now measured:** a physical
RTX 5090 A/B at the qualified 262,144-token profile moved the engine-reported KV pool
**265,122 → 282,996 tokens — +17,874 (+6.7 %), 9.28 → 9.88 GiB, i.e. +0.60 GiB** — reproduced
identically across two server starts per arm, with the arena's growth log ending at exactly the
predicted 170 MiB (`receipts/scratch-arena.json`; overlay `tools/vllm-exl3-scratch-arena.py`, fork
PR <https://github.com/local-inference-lab/vllm/pull/397>). **Measured / predicted = 17,874 /
18,676 = 95.7 %**: the prediction used the marginal KV law alone, while the engine also charges
per-token block/page overheads. The \[INFERENCE] figure is kept above rather than overwritten —
it is this path's calibration datum for the next static prediction, and 95.7 % is how much of one
to expect. Quote the measured number, never the prediction.

The mechanism the prediction assumed also held: these buffers allocate during vLLM's profiling
prefill, so they sit inside the measured 1.78 GiB "peak activation", and shrinking them enlarges
the computed KV pool directly — which is what the A/B observed. The bytes are class-independent,
so the same delta applies on *every* class, including 24 GB where the published window is 24,576:
**42,450 raw token headroom**, supporting 40,960 at the next window step — arithmetic only, no
24 GB board booted ([docs/34](34-vram-class-profiles.md) §10.2). The **MTP-off** variant of this
row (+19,800 tokens at a = 32,932) was **not** in the A/B and remains \[INFERENCE]. The fork had
already blessed this exact pattern for the GLM MoE arenas — issue #203
(<https://github.com/local-inference-lab/vllm/issues/203>): "share one `max(total_nbytes)`
scratch tensor across the two owner keys (keep separate *plans*, share the *bytes*)" — and PR
#270 shipped the MoE variant, not this dense one (checked against its 1390-line diff: zero
occurrences of `_EXL3_RECONSTRUCT_SCRATCH`). Build cost, as forecast: a 2-hunk `exl3.py` patch, no
kernel change, byte-exact outputs. KLD risk: none, and no longer only by argument — the 30-case
deterministic vision suite returned byte-identical answers on both arms. One caveat travels with
that claim: it covers the deterministic probe set, because a baseline-vs-baseline control shows
this stack is not restart-deterministic on long greedy continuations (7 of 8 differ across
restarts of the *unpatched* image) with or without the patch.

- `_EXL3_FP8_SCRATCH` (`exl3.py:798, 814-821`): if `VLLM_EXL3_PREFILL_FP8=1` ever ships
  (B12xLeverPlan territory), it adds fp8 `(chunk, K)` buffers per geometry — **+555 MiB** across
  our geometries, plus 16.1 MiB of per-module fp32 output scales (`:824-858`). **Lever R2:**
  fold into the same arena before that flag ships; off by default today (`:803-811`).
- Everything else is already shared or negligible: mixed-Trellis buffers/R7 pools/ballasts are
  MoE-only (`:115-121, 560-617`); Hadamard is computed in-kernel (`had_r_128` — resolver at
  `b12x/moe/_shared/kernels/w4a16/kernel.py:11166-11180`; no resident H matrix; the TurboQuant
  backend's own cached H is 64 KB, `turboquant_attn.py:73-88`); the MCG codebook is a 4-byte
  sentinel per module (census: `mcg I32 []`); the b12x dense c_tmp is capped at
  192·4·64·256 fp32 = 12 MiB architecture-wide (`exl3.py:109-114`).

## 4. KLD risk of sharing H — mechanism, cost, ship policy

Mechanism, two distinct parts:

1. **Error decorrelation.** Per-module random sign vectors (`quantize.py:890-893, 1306`)
   randomize the sign pattern of each module's quantization residual; forcing one H across
   modules makes residual directions coherent, so per-layer errors can add instead of
   cancelling across the depth of the network. (Within one layer's same-input projections the
   signs are *already* shared by the converter — measured 1.0000 gate/up agreement — so the
   marginal correlation would come from sharing across layers and across the output side.)
2. **Scale flattening.** On our checkpoints the stored `suh` is sign⊙per-input-channel-scale
   (`quantize.py:1207-1210`; byte-verified, §2). Sharing one vector across modules discards
   per-module channel scaling outright — a direct change to the quantized function, not merely
   a statistical one.

The magnitude is **not guessed here**. Measuring it costs one conversion plus one shard-0 KLD
score (~1 h GPU; the shard-0 protocol already exists). PR #225's own release rule applies and
we adopt it verbatim: "A complete newly encoded shared-H checkpoint still requires its own KLD
and E2E release validation." If it costs fidelity it ships only as a **separate opt-in quant
variant**, never a silent change. Given the 13.69 MiB ceiling on this dense model, the
measurement is not worth the GPU hour — the lever only pays on a 300-expert MoE.

## 5. NVFP4 KV for Qwen 3.8 — the gates, named

Two unrelated nvfp4 KV paths exist in the r34 tree (`vllm/config/cache.py:27-37`):

**`nvfp4` (non-MLA, would be ours):** FlashInfer backend only.
- Layout: `head_size//2` fp4 data + `head_size//16` fp8 block scales
  (`vllm/utils/torch_utils.py:415-417`) → at head 256: **144 B/side/token vs fp8's 256 B**
  (0.5625×).
- Gate 1 — `flashinfer.py:447-452` (`supports_kv_cache_dtype`): requires
  `current_platform.is_device_capability_family(100)`; family = `capability//10` match
  (`vllm/platforms/interface.py:481-493`), so **10.x only. The RTX 5090 is 12.0 → refused.**
- Gate 2 — `flashinfer.py:786-796`: "`--kv-cache-dtype nvfp4 requires the SM100 trtllm-gen
  FlashInfer path.`" backed by `vllm/utils/flashinfer.py:373-394` ("SM100/SM103 has both
  prefill and decode TRTLLM kernels", everything else unsupported) and wrapper pinning
  `backend="trtllm-gen"` (`flashinfer.py:1084-1086, 1110-1112`).
- Gate 3 — `vllm/config/vllm.py:2326-2335`: nvfp4 KV explicitly rejected for MLA models.

**`nvfp4_ds_mla` (GLM-5.2's path):** `B12X_MLA_SPARSE` backend only
(`vllm/v1/attention/backends/mla/b12x_mla_sparse.py:722-726`), a 432-byte (368 with
`KV_FP8_ROPE=1`) compressed MLA latent record (`:802-806, 1399-1404`), explicitly GLM-gated
(`:1288-1301`: "the compact record is GLM nvfp4_ds_mla-only"). Qwen3.8 is not MLA →
structurally inapplicable.

**Cost for Qwen3.8 on RTX 5090: not config, not a fork patch — vendor kernel work.** The plain
nvfp4 path consumes closed trtllm-gen cubins that exist for SM100/SM103 only; the fork's own
SM120 attention stack (FLASHINFER-on-5090 per our startup logs) has no fp4-KV kernels. GDN
layers are unaffected either way (Mamba-style state, `MambaDType`, `cache.py:38`). The r34
`--kv-cache-dtype` registry also carries `turboquant_*` (own Triton backend,
`turboquant_attn.py`, head-256 explicitly supported) — but its slot layout stores **values at
fp16**: "For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612"
(`turboquant_attn.py:16`) → 612 B/token/head vs fp8's 512 B: a capacity *regression* against
our fp8 baseline, only a win against bf16. `KvDtypeSweep` is measuring the actual refusal/start
behavior and kv law per dtype on the card right now; capacity verdicts defer to
`receipts/kv-dtype-sweep-5090.json` (not landed at writing time — this doc states layout
arithmetic and gate locations only).

## 6. Everything else shareable at runtime

| item | bytes (ours) | finding | verdict |
|---|---:|---|---|
| MTP draft ↔ target lm_head | 0 to save | Draft carries **no** lm_head/embed tensors on disk (census: 39 mtp tensors, 257.8 MiB total); vLLM force-shares anyway: `llm_base_proposer.py:1649-1665` ("Always share it explicitly", `sh.head = target_language_model.lm_head`), and our 5090 log prints "Sharing target model lm_head weights with the draft model" (`receipts/apc-repro-raw/pilot/server-A.log:81`) | already shared |
| MTP draft ↔ target embed | 0 | same mechanism, `shared_weight_names=["embed_tokens"]`; prior retraction stands | already shared |
| int8 embed overlay double-residency | 0 | BF16 table explicitly released after narrowing: `exl3.py:2259-2264` (`layer.weight = empty(0)`, `empty_cache()`) | no double-residency |
| 18.41 vs 18.19 GiB gap | n/a | Manifest floor is **18.069 GiB** (19.252 GiB on-disk − 2.368 BF16 embed + 1.185 int8+scales). 5090 = floor +0.121; rental = floor +0.341. Both measurements sit *above* the same bytes; the 0.22 GiB is measurement-context overhead delta — the rental receipt itself says "image identity was not fully authenticated during capture" (`receipts/native-mtp-8mp-amendment.json`) and ran a different GPU (RTX PRO 6000). Not weights, not overlay double-residency (that would be +1.18 GiB) | explained; keep electing 18.41 for predictions |
| norms + small BF16 | 1.3 MiB | 129 tensors (`collection-index` roles) | nothing to win |
| codebook constants | 4 B/module | MCG is a scalar sentinel (`mcg I32 []`), decoded procedurally in-kernel | nothing |
| Hadamard matrices | 0 resident | in-kernel butterflies (`had_r_128`); TQ backend caches one 64 KB H | nothing |
| suh/svh dedup | 0 today, ≤13.69 MiB ever | §2: zero byte-identical vectors | dead end |
| online-overlay BF16 source | 0 | context edition is fully serialized; the only online overlay is the int8 embed (released, above). The K6 online cache path (`Exl3OnlineLinearMethod`) is not exercised by this build | n/a |
| vision tower (runtime-only) | up to ~439 MiB | 0.858 GiB BF16, 333 tensors. No duplicate exists → nothing to *share*. A runtime MXFP8 overlay is the in-fork precedent for BF16 linears (PR #225: "Only BF16 dense/shared-expert weights are eligible for the MXFP8 overlay") and would not touch conversion (do-not-reopen respected), but vision-path fidelity is unmeasured → separate opt-in variant + measurement first | candidate, gated on measurement |

## 7. Ranked levers and the 16 GB answer

| # | lever | 32 GB | 24 GB | 16 GB | KLD risk | build cost | verdict |
|---|---|---:|---:|---:|---|---|---|
| 1 | R1: single reconstruct-scratch arena (`exl3.py` dict→arena) | **measured +17,874 tok MTP-3 / +0.60 GiB KV pool** (265,122 → 282,996; predicted +620 MiB ≈ +18.7k = 95.7 %) | same measured bytes; window 24,576 → 42,450 raw headroom, supports 40,960 \[P] — arithmetic only | **+0.60 GiB** measured on 32 GB, class-independent bytes, but see below | none (byte-exact; deterministic vision suite byte-identical across arms) | 2-hunk `exl3.py` patch, as forecast; precedent issue #203/PR #270 (MoE variant only) | **built, measured on the 5090, PR'd — fork PR #397, `receipts/scratch-arena.json`** |
| 2 | KV dtype (fp8→?) | defer | defer | defer | measured elsewhere | n/a | **defer to `receipts/kv-dtype-sweep-5090.json`**; turboquant is −100 B/token vs fp8 by layout; nvfp4 hard-refused on SM120 |
| 3 | R2: fp8-prefill scratch arena (pre-req for `VLLM_EXL3_PREFILL_FP8=1`) | avoids +555 MiB | same | same | none | trivial once R1 exists | build with R1 if fp8 prefill ships |
| 4 | vision MXFP8 runtime overlay | +439 MiB | +439 MiB | +439 MiB | unmeasured (vision path) | moderate (extend overlay policy) | opt-in variant, measure first |
| 5 | embed int8→int4 overlay | +592 MiB | +592 MiB | +592 MiB | unmeasured | small | opt-in variant, measure first |
| 6 | shared-H (suh/svh) on this model | ≤13.69 MiB | ≤13.69 MiB | ≤13.69 MiB | real mechanism (§4), unmeasured | converter + loader | **do not build**; MoE-only lever |
| 7 | nvfp4 KV port to SM120 | — | — | — | n/a | vendor trtllm-gen cubins / upstream kernel work | out of reach for this project |

**The 16 GB question, answered explicitly.** No stack of runtime-sharing levers makes 16 GB
usable: resident weights alone are 18.19 GiB, and the *entire* runtime-recoverable inventory
above (R1 + vision overlay + int4 embed ≈ 1.6 GiB) lands at ~16.6 GiB resident — still above
the class's whole 12.49 GiB weights+KV budget (`receipts/vram-class-verdict.json`,
`card_budget_model.classes.16_gib`). The class flips only on serialized bytes: the S16-V
candidate (full-attn K4, GDN K3, MLP K3/K3/K3, head K4, 12.77 GiB serialized — all predictions)
plus int8 embed. What must be measured first, in order: **(1)** the flip condition already on
file — a sub-4-bit shard-0 KLD (one conversion + one shard-0 score ≈ 1 h GPU, protocol exists);
**(2)** `KvDtypeSweep`'s receipt for whatever KV dtype the 16 GB context budget would use;
**(3)** R1's pool gain, now **measured** at +17,874 tokens / +0.60 GiB on 32 GB
(`receipts/scratch-arena.json`) — at 16 GB those class-independent bytes are the difference between
a toy window and a usable one *after* (1) passes. Runtime sharing is margin, not the door.
