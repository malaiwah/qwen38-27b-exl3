# VRAM-class profiles: a 24 GB mid-profile and a 16 GB profile, by arithmetic

This closes the *design* half of [29](29-plan-and-loose-ends.md) F3. It does not close F3.
Nothing here is a claim that a profile fits a card: every profile below ends in an acceptance
test that must pass on the actual board class before the profile may be published, and every
number is either traced to a receipt or labelled **[P]** for prediction with its formula shown.

Terminology is the F4 convention: **serialized bytes** are what a checkpoint occupies on disk,
**resident weight bytes** are what the loader keeps on the device, **activation**, **CUDA-graph**
and **KV** are separate quantities, and none of them is called VRAM.

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

The measured datapoint is 9.31 GiB ÷ 266,612 tokens = **37,494.7 B/token**, which is 1.1442×
the floor; the gap is 9.31 GiB − 266,612 × 32,768 B = **1.1736 GiB** of the pool that is not
attention tensor bytes.

**Does that figure include the hybrid linear-attention state?** It includes whatever the engine
placed in that pool, because the receipt reports one `available_kv_gib` and one
`gpu_kv_cache_tokens` and itemizes neither. The evidence says the GDN state is *not* the
residual:

- the only sizing available for the GDN pool is external — 48 GDN layers × 48 heads × 128 × 128
  at `mamba_ssm_dtype` plus a bf16 conv state, "153.9 MB at fp32, 78.4 MB at bf16" per slot,
  quoted from SGLang in [docs/07](07-serving-recommendations.md) §5.6 — which at `max_num_seqs` 1
  is ≤0.15 GiB, at most 13 % of the 1.1736 GiB residual;
- the MTP-off run in `receipts/context-capacity-5090-budget.json` allocated 266,743 tokens from
  8.18 GiB = 32,929 B/token, only **1.0049×** the floor. A 0.15 GiB GDN pool would be 1.8 % of
  that 8.18 GiB and would have shown up. It did not, so vLLM is accounting the recurrent state
  somewhere other than `available_kv_cache`.

So the residual is dominated by MTP-3 — draft KV plus whatever hybrid page padding the
speculative path forces — and the GDN state is unaccounted in our receipts. Handling: carry it
as an **explicit 0.20 GiB reserve `H` outside the per-token rate**, and re-derive the per-token
rate net of that reserve so the model still reproduces the measured allocation exactly at
266,612 tokens. That makes the model conservative at every shorter context, which is precisely
where both new profiles live — folding a fixed cost into a per-token multiplier would
*under*-charge short contexts, the failure mode to avoid.

```
KV_bytes(T) = H + c_tok * T          H = 0.20 GiB
c_tok(MTP-3, fp8) = (9.31 GiB - H) / 266,612 = 36,689.2 B/token          [derived from the measured pool]
c_tok(no MTP, fp8) =  8.18 GiB      / 266,743 = 32,929.0 B/token          [derived, 1.0049x the floor]
c_tok(4-bit KV)    = c_tok(fp8) * 16,384 / 32,768                          [P] pure halving, never measured
```

The two rates treat `H` asymmetrically on purpose. The MTP-3 rate is derived net of the reserve,
because that pool has 1.1736 GiB of slack above the tensor floor to take it from. The MTP-off
rate is taken gross, because subtracting 0.20 GiB from 8.18 GiB over 266,743 tokens would put it
at 32,122 B/token — *below* the 32,768 floor, i.e. that run had no room for a separate reserve at
all. Planning still adds `H` on top in both cases, so the MTP-off predictions carry 0.20 GiB of
deliberate double-count.

MTP-3 therefore costs 11.4 % on top of every KV token in addition to its draft weights. Neither
per-token figure is a physics constant: both are one engine version, one hybrid allocator, one
`max_num_seqs`, one `block_size`.

## 5. Profile M24 — the 24 GB mid-profile

The finding that shapes this profile: **at 24 GB no new checkpoint is required.** The published
`-context` edition, served with the int8 input overlay and MTP-3, has a *measured* resident
weight figure of 18.41 GiB, which leaves 20.32 − 18.41 = 1.91 GiB for KV on a 24 GiB board. So
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

`KV_avail = 22.84 − resident − 2.52`, then `T_max = (KV_avail·2^30 − H) / c_tok`. A profile is
published at the largest multiple of 4,096 whose requirement `H + T·c_tok` leaves **≥15 %
headroom** against `KV_avail`. The shipped 32 GB profile only needed 262,144 < 266,612, 1.7 %,
because 266,612 was *allocated by the engine and read out of the log*; a prediction needs more,
and the 15 % absorbs the two soft constants in §3 plus the ±0.05 GiB resident model. Once a
profile has started, its published length may be raised to whatever the log actually allocated.
All lengths **[P]**:

| profile | resident | KV avail **[P]** | `T_max` fp8 + MTP-3 | **published, MTP-3** | `T_max` fp8, MTP off | **published, MTP off** | `T_max` 4-bit KV + MTP-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M24-Q (hydrated body) | 19.27 | 1.05 | 24,765 | **20,480** | 36,203 | **28,672** | 49,531 |
| **M24 (context body)** | **18.41** | **1.91** | 49,934 | **40,960** | 63,856 | **53,248** | 99,868 |
| M24-L | 17.55 | 2.77 | 75,102 | **61,440** | 90,139 | **73,728** | 150,205 |

MTP-off rows drop the draft weights too (0.252 GiB for M24, 0.264 for M24-Q, 0.198 for M24-L)
and use `c_tok` 32,929. The 4-bit-KV column is a raw maximum and is not publishable at all until
that dtype has been measured (§7). Every published figure here is still a prediction: it becomes
a claim only after §8.

Serving flags for M24 are the amendment's, with the length changed:

```bash
VLLM_EXL3_EMBED_BITS=8 VLLM_EXL3_GRAPH_DECODE=1 \
vllm serve <ctx-checkpoint> --max-model-len 40960 --max-num-seqs 1 \
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

Everything measured in this repository is **SM120, TP1, driver 595.58.03**
(`receipts/release-evidence-context.json` → `hardware`). The 24 GB class spans more than one
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
| needle retrieval, `tools/longctx.py`, depths 0.1 / 0.5 / 0.9 | at ≥0.98 × 40,960 with MTP-3 and ≥0.98 × 53,248 without, token-counted before submit | at 61,440 / 73,728 | at 16,384 with MTP off, and separately at 12,288 with MTP-3; run S16-V-long at 16,384 with MTP-3 to settle §6.2's one-bit trade |
| combined long-text + image, `tools/longmm.py` | one request ≥0.9 × published length carrying a ≥7 MP image, exact code **and** colours | same | at the 4.2 MP cap; if it fails, the cap or the multimodal claim goes |
| vision suite, `tools/vision_eval.py`, 30 deterministic cases | ≥ the 24/30 measured at 8.4 MP | ≥24/30 | report; **do not** assume 24/30 transfers to a K3 body |
| throughput, `tools/bench.py`, warmed, ≥3 timed runs | single-stream with MTP-3 and off; sampler, temperature and reasoning effort printed | same | same |
| non-termination / repetition check | required | required | **required and weighted**: two independent Q4 reports describe looping to context exhaustion ([docs/29](29-plan-and-loose-ends.md) F5), which mean KLD cannot see |
| MTP acceptance at pinned temperature | report | report | report; a K3 draft may not accept |
| receipt | new `release-evidence-m24.json` with the same field set as `release-evidence-context.json`, plus a row in `receipts/collection-index.json` | same | same |

Rules that apply to all three, in the F4 spirit:

1. **No fit claim without a startup.** "Predicted 49,934 KV tokens" and "allocated 49,934 KV
   tokens" are different sentences; only the second may appear on a card.
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

- No 24 GB or 16 GB board has been started. Every context length here is arithmetic.
- The two card constants — 0.98094 free-at-startup and 0.97 utilisation — are transferred from
  one board's note and are the likeliest source of error in §3.
- The GDN conv/SSM state is still unaccounted in our own receipts; §4 reserves 0.20 GiB for it on
  external sizing and reasoning, not on a measurement of ours.
- No sub-4-bit width of this architecture has ever been measured for fidelity, which makes S16 a
  design and not a product.
- The 4-bit KV dtypes are unmeasured, and the current fidelity protocol resolves the KV cache to
  bfloat16, so it cannot measure them even for the profiles that already ship fp8 KV.
- Activation, non-torch and CUDA-graph figures are carried unchanged from a 32 GB profile; a
  16 GB profile with a lower image cap and fewer batched tokens should measure its own.
