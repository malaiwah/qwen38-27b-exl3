# 40 — The Qwen3.8-27B quant landscape, scanned 2026-08-16

Full evidence: [`receipts/quant-landscape-scan.json`](../receipts/quant-landscape-scan.json)
(7.9 MB: 952 candidate repos with metadata, 464 with full detail, 55 hand-classified,
plus the digest forensics and the org-coverage sweep).

Read-only scan. No GPU was used, no weight file was downloaded. Byte counts come from the
API sibling listing; tensor names come from HTTP Range reads of safetensors JSON headers
(first 8 bytes for the header length, then the header) — metadata only.

---

## 1. The unsloth question, settled by digest

**No reconversion. The weights are byte-identical to our pin. What happened was a history
super-squash on 2026-08-15 plus two metadata edits.**

`unsloth/Qwen3.8-27B-NVFP4` now has exactly one commit on `main`:

| | |
|---|---|
| HEAD | `16b6615af3548b88e2d8e382457bc705b00479cf` |
| commit | `Super-squash branch 'main' using huggingface_hub`, 2026-08-15T07:04:09Z |
| commits on main | 1 |
| branches / tags | `main` only / none |
| `lastModified` | 2026-08-15T07:04:09Z — identical to the squash, so nothing has landed since |
| our pin | `9c73e2da…` → `404 Invalid rev id`; `paths-info` at that rev → `500` |

The commit list is useless for answering the question, so it was answered by content:

| file | bytes | our pin sha256 | HEAD sha256 | |
|---|---:|---|---|---|
| `model.safetensors` | 22,568,192,096 | `c473512c…afcc05` | `c473512c…afcc05` | **same object** |
| `model_mtp.safetensors` | 849,400,392 | `1d8268aa…3c9da9fe` | `1d8268aa…3c9da9fe` | **same object** |
| `tokenizer.json` | 19,989,325 | `f399b3cd…` | `06b95093…0e523` | replaced |

Source: `POST /api/models/unsloth/Qwen3.8-27B-NVFP4/paths-info/16b6615a…` with `expand=true`;
`lfs.oid` is the sha256.

Three independent corroborations, none of which requires trusting unsloth's history:

1. **Our own mirror.** `malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da` (rev `c6e3f1f0`,
   2026-08-16) carries exactly `c473512c…` and `1d8268aa…`. The pinned-era and HEAD blobs
   are provably the same objects under two revision ids.
2. **A third party carrying the pre-squash config.** `tlwu/Qwen3.8-27B-NVFP4-ONNX`
   (rev `b3574424`, 2026-08-15) declares `base_model: unsloth/Qwen3.8-27B-NVFP4` and its
   `config.json` still has `config_groups.group_1.weights.observer = "imatrix_mse"` with
   `actorder: "static"` — an independent copy of the metadata that HEAD dropped.
3. **Three bit-identical community re-uploads** — `FenomAI/`, `speakguru/` and
   `myhuggingbb/Qwen3.8-27B-NVFP4` all carry both digests unchanged. There is exactly one
   set of NVFP4 weights in circulation under this name.

### What actually changed in `config.json`

`quantization_config.config_groups.group_1.weights.observer`: our pin had `"imatrix_mse"`;
at HEAD **the key is absent entirely** (not `null` — the served JSON omits it).
`actorder` is still `"static"`. `group_0.weights.observer` (`memoryless_minmax`) and
`group_1.input_activations.observer` (`static_minmax`) are unchanged, as is the whole
mixed-precision structure: group_0 float-quantized W8A8 on `self_attn.{q,k,v,o}_proj`,
`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `lm_head` and layers 56–63 MLP; group_1
`nvfp4-pack-quantized` W4A4 tensor-group g16 on the remaining MLPs; 303 ignore entries
covering the vision tower and the MTP head; `max_position_embeddings` 262,144.

### Does our published number describe a superseded artifact?

**No.** Mean KLD `0.031059` [0.027916, 0.034795] with 92.90 % top-1 over 10,480,640 v5
positions, and the 21.34 GiB measured resident weight, were measured against blobs
`c473512c…` and `1d8268aa…`, which are what unsloth serves today. The one live hazard is
that a reader pulling HEAD gets the replacement `tokenizer.json` — which changes
tokenisation, not the model.

### Has unsloth shipped anything else for this model?

No. All 1,446 unsloth model repos were enumerated. There are exactly four Qwen3.8-27B
repos and no others: the bf16 mirror (`3ea932ce`), FP8 (`d51e38f6`), GGUF (`f1bfb127`)
and NVFP4 (`16b6615a`). **No AWQ, no GPTQ, no bnb / unsloth-dynamic-bits, no MLX, no
second NVFP4 variant.**

The GGUF repo is still at `f1bfb127`, our known revision. Its last *weight* change was
2026-08-14T14:58:56Z (`Upload Qwen3.8-27B-UD-Q8_K_XL.gguf`). The 18 commits after that are
thirteen `Add Unsloth style chat template - UD all fine` commits and five README edits. That
repo also super-squashed, at 2026-08-14T14:56:37Z. 24 quant types, 394.6 GiB, unchanged.

One absence is informative: for **Qwen3.6-27B** unsloth shipped `UD-MLX-NVFP4`,
`UD-MLX-MXFP4`, `UD-MLX-3bit/4bit/6bit`, `MLX-8bit` and an `MTP-GGUF`. None of those
exist for 3.8-27B. The Apple-silicon and MTP-GGUF niches unsloth normally occupies are
currently filled entirely by the community.

---

## 2. Shape of the landscape

952 public repos are derivatives of `Qwen/Qwen3.8-27B`. **556 of them were created on or
after 2026-08-15** against 396 before, i.e. more repos appeared in the two days after the
baseline recorded in `docs/29` than existed before it.

Format tally of the new cohort (a repo can carry several labels):

| gguf | mlx | mlx-oQ | compressed-tensors | nvfp4 | fp8 | int4 | awq | w4a16 | autoround | exl3 | gptq | mxfp4 | mxfp8 | openvino | w4a8 | onnx | mlc | coreml |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 242 | 131 | 57 | 51 | 34 | 27 | 25 | 17 | 16 | 12 | 11 | 11 | 7 | 6 | 3 | 2 | 2 | 2 | 1 |

Three structural facts distort any raw count. `Thireus` alone contributes 129 repos (126 of
them in the new cohort), one per quant type, because SPECIAL_SPLIT is a per-tensor shard
*library* rather than 129 opinions. 163 of the 952 are behavioural derivatives —
abliterated, heretic, uncensored, obliterated — rising to 202 if distills, merges and
persona finetunes are included; these are different *models*, not different quantisations
of ours, and KLD against the stock BF16 parent is meaningless for them. And 319 repo names
carry an Apple-silicon marker (`mlx`, `JANG`, `oQ*`, `MTPLX`, `coreml`), against 163 with
`GGUF` in the name.

Adoption is extremely concentrated and almost uncorrelated with evidence. The top NVFP4
builds by downloads are unsloth (276,269), RadixArk (96,972) and Inferact (51,350) — and
Inferact's card is 182 characters long.

---

## 3. New since our pin — what each one is *for*

Bytes are HEAD totals. "MTP" and "vision" are from the weight index or the safetensors
header, not from the card, wherever the two disagree.

| repo (rev, created) | GiB | objective | engine | MTP / vision | published fidelity evidence |
|---|---:|---|---|---|---|
| `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (`69274a0d`, 08-14, updated 08-15) | 19.20 | **262,144 ctx on one 32 GB RTX 5090, explicitly beating unsloth** | vLLM 0.27.x + FlashInfer SM120 | 15 mtp tensors (BF16, not enabled) / 333 visual | 20-item smoke, GPQA-D + AIME25 + MMLU-Pro: 45/60 both. No KLD. Card disclaims it. |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` (`63768c10`, 08-15) | 19.60 | AWQ INT4 on a STEM+agentic calibration corpus | transformers / vLLM / SGLang | 15 / 333 | **none** — 21,015 downloads, 30 likes, zero numbers |
| `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (`156be69f`, 08-15) | 18.46 | vendor MXFP4 W4A4 via Quark + AWQ | AMD Quark / vLLM | 15 / 333 (card mentions neither) | GSM8K 5-shot 94.996 % vs bf16 93.33 % (101.8 % recovery) |
| `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16` (`c3821914`, 08-15) | 18.55 | vendor INT4 via Qronos Hessian PTQ | AMD Quark / vLLM | 15 / 333 | GSM8K 94.62 % (101.4 % recovery) |
| `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` (`a7cc2681`, 08-15) | 18.19 | vendor INT4 via AWQ, control for Qronos | AMD Quark / vLLM | 15 / 333 | GSM8K table |
| `rdtand/…-PrismaAQUA-5.5bit-vllm` (`1e086823`, 08-14) | 22.00 | per-Linear error-priced allocation at 5.5 bpp on a stock runtime | vanilla vLLM | none / 333 | **KL 0.0338 over 4,088 positions**, 0.0187 confident-only, WT2 PPL 9.580 vs 9.361; every number hashed in `shipcard.json` |
| `rdtand/…-gridbook-12.1GiB-5080-vllm` (`bbbb4f53`, 08-16) | 12.09 | 16 GB Blackwell target, 19-rung codebook ladder | vLLM + out-of-tree GridBook 0.8.8 | **15 mtp present despite the card saying "removed"** / vision genuinely absent | bpp arithmetic published two ways |
| `kaitchup/Qwen3.8-27B-hum-*` ×6 (`5462dc4d` etc., 08-15/16) | 13.12 – 17.90 | deliberate memory ladder, ~2.8 → ~4.5 bits | vLLM v0.26+ | bf16 MTP / 333, every rung | **none — "currently under evaluation"** |
| `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` (`b67127bf`, 08-15) | 19.38 | mixed INT4, vision preserved | vLLM | 29 mtp-named / 333 | MMLU 83.07 vs 83.49 (99.50 % recovery); **allocation strategy withheld by design** |
| `chriswritescode/Qwen3.8-27B-MXFP8` (`05f53164`, 08-15) | 29.83 | accuracy-first MXFP8 | vLLM (Marlin dequant pre-Blackwell) | bf16 mtp / whole tower bf16 | **mean KLD 0.008471 [0.006421, 0.010920], 96.80 % top-1, 136/136 vs official FP8 0.013365 — measured on our v3 suite, unmodified** |
| `huginnfork/Qwen3.8-27B-NVFP4A16` (`6916a5bb`, created 08-05, updated 08-14) | 28.87 | MLP-only NVFP4; attention, SSM, vision, head, MTP all bf16 | vLLM + transformers on SM120 | 15 bf16, **84.0 % draft acceptance measured** / whole tower | KLD 0.1267, WT2 PPL 7.1497 vs 6.9416; protocol published (8 samples, max_seq 1024) |
| `HarimxChoi/WarpQuant-…-R16E4H4` (`8ecfd2b8`, 08-15) | 51.77 repo / 11.32 payload | Hadamard rotation + 3-bit group + block-GPTQ + Output-Fisher recovery | custom | n/s / tower + projector | WT2 PPL 7.4737 @ 3.6165 bpw vs **IQ3_S 7.1820 @ 3.6940** — it loses, and says so |
| `empero-ai/Qwen3.8-27B-Ridge-GGUF` (`486faa5f`, 08-15) | 12.60 | Gated-DeltaNet-aware 3.69 bpw GGUF, nothing stripped | stock llama.cpp / Ollama / LM Studio | `blk.64`/nextn kept inside the GGUF / separate BF16 mmproj | imatrix-probed mix described; no table. 64 likes. |
| `enginetown/Qwen3.8-27B-Calibrated` (`f24f31fe`, 08-15) | 39.59 | per-tensor-category KL floor search, then whole-model validation | llama.cpp | n/s | KL-divergence methodology across general/code/math/tool prompts; **requantized from unsloth Q8_0, not bf16** |
| `mixbits/…-NVFP4-MTP-VL-GGUF` (`0b9ab185`, 08-15) | 15.49 | make the official 1M static-YaRN window loadable in Ollama | llama.cpp / Ollama 0.32 | in-GGUF MTP / separate F16 mmproj | none — and it states it only rewrote GGUF KV, not tensors |
| `Jackrong/Qwen3.8-27B-MTP-GGUF` (`42245110`, 08-15) | 228.85 | MTP inside the GGUF, no separate drafter | llama.cpp | bundled / claimed | throughput only: 1.99× tok/s, 77.9 % draft acceptance, internal testing |
| `manjunathshiva/Qwen3.8-27B-tq3-mini-g64` (`5eee2054`, 08-15) | 11.57 | data-free 3-bit sized to a **real** 16 GB M4 mini | MLX / TurboQuant | n/s / multimodal, 4/4 vision battery | no KLD, but peak RSS 12.95–13.61 GiB measured on the actual machine, Opencode agentic loop passes, 3.7 tok/s |
| `Ostfralla/Qwen3.8-27B-NVFP4-NInfer` (`131c2c82`, 08-15) | 17.07 | NVFP4 for the NInfer engine | NInfer + bundled patch (upstream `ninfer#25` open) | none / none | HumanEval+ 152/164 and AIME25+26 55/60, paired greedy vs the NInfer author's build |
| `tlwu/Qwen3.8-27B-NVFP4-ONNX` (`b3574424`, 08-15) | 21.10 | ONNX Runtime GenAI export of unsloth NVFP4 with MTP packaged in | onnxruntime-genai CUDA | `mtp.onnx` shipped / **text-only, tower dropped** | none |
| `nm-testing/Qwen3.8-27B-NVFP4-FP8` (`8cb2319c`, 08-16) | 21.83 | Red Hat / Neural Magic staging | vLLM | ships **unsloth's exact `model_mtp` blob** / 333 | none — empty card |
| `CuTIsolation/Qwen3.8-27B-W4A8` (`6edc266b`, 08-15) | 16.69 | W4A8 with the conversion script in-repo | custom | **0 mtp — head dropped** / 333 | none |
| `dbrasdasilva/Qwen3.8-27B-Text-NVFP4-MTP` (`4421ddd3`, 08-15) | 18.31 | text-only NVFP4, tower amputated | vLLM | 15 / **0** | none |
| `pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB` (`2b550610`, 08-15) | 20.43 | two 16 GB Blackwell cards, TP=2 | vLLM | 15 / 333 | none |
| `lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF` (`e5d43136`, 08-15) | 36.20 | Strix Halo gfx1151 native tensor types | ciru-ai ROCmFPX fork only | none / mmproj | none |
| `jcbtc/…-CIRU-ActiveFPX-PromptForge` (`cd4ba547`, 08-15) | 49.78 | ROCm quality+speed, Q8 output projection | ROCm llama.cpp fork | native MTP / separate BF16 projector | none quantified |
| `michaelw9999/Qwen3.8-27B-MXFP8-GGUF` (`1de78a59`, 08-16) | 26.28 | MXFP8 faster than Q8_0 on Blackwell | **private `mxfp8_cuda` fork only** | none / none | none — "benchmarks soon" |
| `Chungulus/Qwen3.8-27B-JANG_6M` (`eb0427a3`, 08-15) | 22.27 | honest vanilla MLX conversion, base pinned to `1d4bf0f2` | JANG / MLX 2.5.46 | 15 / 333 | none; states calibration source is none |
| `Mungert/Qwen3.8-27B-GGUF` (`0206a350`, 08-15) | 415.28 | breadth + an external selection guide | llama.cpp | none / mmproj | none of its own |
| `logic65/Qwen3.8-p44w75-16.8B-unrepaired` (`26d706c6`, 08-15) | 56.18 | **not a quant** — zero-training depth+width prune 27B → 16.8B for 2×8 GB | GGUF | none / none | self-reported battery 25/39 un-repaired, 36/39 after QLoRA repair (`Qwen3.8-Whittle-16B`) |
| `FenomAI` / `speakguru` / `myhuggingbb` `/Qwen3.8-27B-NVFP4` (08-15) | 21.83 | **nothing** | — | — | **bit-identical re-uploads of unsloth — both digests match** |

Also new but not competing on our axes: 11 `FlagRelease` repos (Ascend, Kunlunxin, Hygon,
Iluvatar, Enflame, MThreads, Tsingmicro, Sunrise, Zhenwu, ARM, NVIDIA), the two
`OpenVINO` int4/int8 builds, `amd/…-text-cpu-onnx`, and 319 MLX / JANG / oQ / MTPLX /
CoreML repos on Apple unified memory.

### Re-uploads and merges that add nothing

Proven by sha256, not inferred:

- `FenomAI/`, `speakguru/`, `myhuggingbb/Qwen3.8-27B-NVFP4` — both weight digests equal
  `unsloth/Qwen3.8-27B-NVFP4`.
- `zherebetskyy/Qwen3.8-27B-nvfp4-mlx` = `mlx-community/Qwen3.8-27B-nvfp4`;
  `majentik/Qwen3.8-27B-MLX-MXFP4` = `mlx-community/Qwen3.8-27B-mxfp4`.
- Our own `malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da` is deliberately identical to
  unsloth — that identity *is* the evidence in §1.

Three shared blobs are worth knowing about because they show pipelines copying rather than
re-deriving:

- `1d8268aa…` (unsloth's 849,400,392-byte bf16 MTP head) is also shipped verbatim by
  `nm-testing/Qwen3.8-27B-NVFP4-FP8` and by `dbirks/Qwen3.8-27B-NVFP4-AutoRound` (renamed
  `model_extra_tensors.safetensors`).
- `9ce944d5…`, the trailing untouched bf16 shard, is byte-identical across
  `gittensor-model-hub`, `vroomfondel`, `PassingByPixels` and `kristianpaul` — four
  ModelOpt builds that differ *only* in their quantised shards.
- `90fa0e3e…` (849,400,424 B, 32 bytes larger than unsloth's) is shared by
  `kelnei/Qwen3.8-27B-NVFP4` and `sakamakismile/Qwen3.8-27B-MTP-NVFP4` — evidence of a
  second, independent MTP re-graft pipeline.

### Two card defects found while reading

- `rdtand/…-gridbook-12.1GiB` says "removed: MTP heads, visual tower". The vision tower is
  genuinely gone (0 visual tensors), but all 15 `mtp.*` tensors are in the file
  (`mtp.fc.weight`, `mtp.layers.0.*`, `mtp.norm.weight`, …). Half the claim is wrong.
- `huginnfork/Qwen3.8-27B-NVFP4A16` says "This build has lower divergence from the bf16
  parent than the official release on both measures" and then, in the same sentence,
  "142 % higher weight-only and 27 % higher as deployed". Its own table reads 0.1267
  against the official FP8's 0.0523. The percentages are right; the direction word is
  inverted. Anyone quoting that card's prose will quote it backwards.

---

## 4. Who could beat us, and what would have to be measured

Our axes: **native 262,144 context on a single 32 GB card, vision intact, MTP preserved,
low KLD.** Everything below is separated into *plausibly competitive but unmeasured*,
*measured and worse*, and *not a competitor because it targets a different constraint*.
Nothing here was measured by this scan.

### Plausibly competitive, unmeasured — ordered by what measuring would tell us

1. **`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`** (`69274a0d`, 19.20 GiB). The only
   repo that attacks our headline claim by name: 18.8 vs 22.7 GB resident, 275,941 vs
   77,184-token FP8 KV pool, 80.6 vs 42.4 tok/s decode, longest completed prompt 242,686.
   Vision and all 15 MTP tensors are in the file. 52 likes on 1,414 downloads.
   *Measure*: v5 KLD and top-1, paired. Then settle two things its card does not — whether
   the 275k pool is bought by FP8 KV with baked scales (in which case unsloth must be
   re-run with `--kv-cache-dtype fp8` for a like-for-like), and whether its MTP actually
   drafts, since the tensors are BF16 and the serve command never enables spec-dec.
   *Tells us*: whether uniform ModelOpt W4A4 at 19.2 GiB is genuinely on the frontier, or
   whether a 20-item smoke is hiding a gap a 10 M-position KLD would expose.

2. **`sakamakismile/Qwen3.8-27B-MTP-NVFP4`** (`6d98dc1f`, 19.18 GiB). Feature-identical to
   the artifact we benchmarked — NVFP4 W4A4 g16, MTP bf16 wired into the index, tower
   bf16, 262,144, stock vLLM — at 2.65 GiB less. 19,932 downloads, throughput table only.
   *Measure*: v5 KLD, top-1, MTP acceptance; diff its ignore list against unsloth's 303.
   *Tells us*: whether unsloth's extra 2.65 GiB (the group_0 FP8 attention / SSM / lm_head
   / last-8-MLP block) buys measurable fidelity or is dead weight. That is a decidable
   question about the recipe we published against.

3. **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** (`63768c10`, 19.60 GiB). 21,015 downloads and 30
   likes in under two days on a card whose only quant-specific content is a version string
   and a calibration link. Vision + MTP intact, in our envelope, calibrated on a STEM and
   agentic corpus rather than a generic one.
   *Measure*: v5 KLD **per stratum** (literary / code / encyclopedic / multilingual /
   scientific), not just pooled.
   *Tells us*: whether AWQ INT4 W4A16 competes with NVFP4 at the same bytes, and whether
   calibration domain shows up as a stratum-level effect. Also the clearest case of
   adoption running far ahead of evidence.

4. **`rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm`** (`1e086823`, 22.00 GiB). The closest
   methodological rival to our error-driven allocation work and the only third party
   besides chriswritescode publishing a hashed KL protocol. Their **KL 0.0338 over 4,088
   positions** sits close to our **0.031059 over 10,480,640** for unsloth NVFP4 — different
   corpora, so **those two numbers must never be quoted side by side.**
   *Measure*: v5 KLD on identical contexts; separately re-score their shipcard corpus
   through our harness to establish the cross-protocol offset.
   *Tells us*: whether per-Linear error pricing at 5.5 bpp beats K5/K6 at comparable bytes,
   and how much of any gap is protocol. Also a second calibration of how badly
   4,088-position protocols understate variance.

5. **`turboderp/Qwen3.8-27B-exl3` at 5.00bpw (18.56 GiB) and 6.00bpw (21.39 GiB)**.
   Our own engine, uniform bitrate, vision (333) and MTP (39 tensors at `mtp_bits=4`)
   preserved at *every* rung; exl3 v1.4.2, `head_bits` 6, `out_scales` always, codebook
   `mul1`, calibration 250 × 2048. Full ladder: 2.00 = 10.06, 2.50 = 11.48, 3.00 = 12.89,
   3.50 = 14.31, 4.00 = 15.73, 5.00 = 18.56, 6.00 = 21.39 GiB. Already on the `docs/29`
   attack list.
   *Tells us*: exactly what our K5/K6 allocation is worth versus uniform bitrate at the
   same bytes, same engine, same calibration family. 5.00bpw brackets our context edition
   (19.28 GiB); 6.00bpw brackets unsloth NVFP4 (21.83 GiB). The cleanest controlled
   comparison available anywhere in this landscape — and without it every allocation claim
   we make lacks its control arm.

6. **`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (18.46) and `…-Quark-Qronos-INT4-W4A16` (18.55)**.
   First-party AMD, created after our baseline, and — established here by reading the
   safetensors headers, since neither card says so — they keep both the vision tower (333)
   and the MTP head (15). Smaller than anything we ship. Their only evidence is GSM8K at
   >100 % recovery, which is the classic signature of a benchmark that cannot resolve the
   difference.
   *Tells us*: where Qronos (Hessian PTQ) and AWQ-MXFP4 land against NVFP4 at 4 bits, and
   gives us a vendor comparison point that is not NVIDIA's format.

7. **`kaitchup/Qwen3.8-27B-hum-*`**, six rungs from 13.12 to 17.90 GiB, vision and bf16 MTP
   at every rung, ~2.8 → ~4.5 bits, AutoRound quantised and llm-compressor packed. The
   cards say outright that evaluation is pending.
   *Measure*: the two ends — compact (13.12) and quality-MTP (17.90). Two points define
   the slope.
   *Tells us*: where AutoRound INT ladders sit against ours below 4 bits, the region where
   K4 (0.010604) is our only datapoint.

8. **`chriswritescode/Qwen3.8-27B-MXFP8`** (`05f53164`, 29.83 GiB). Not a 32 GB competitor
   — no KV room — but it is the only external artifact measured on **our** suite, and it
   beats official FP8 by 36.6 % on 136/136 contexts (0.008471 vs 0.013365) on v3 over
   278,392 positions, with a five-stratum breakdown.
   *Measure*: re-run on v5 so it lands on the master chart next to our 0.002760 / 0.003210
   / 0.003509 / 0.010604 and official FP8's 0.005294.
   *Tells us*: converts a v3 result to v5 — but the real value is that it is a second
   independent validation of our harness by someone who is not us.

9. **`empero-ai/Qwen3.8-27B-Ridge-GGUF`** (`486faa5f`, 12.60 GiB). 3.69 bpw with the MTP
   draft head inside the GGUF and a BF16 mmproj, built with explicit Gated-DeltaNet
   awareness (`ssm_alpha` / `ssm_beta` as first-class). 64 likes, the highest of any
   community GGUF with a stated objective. We already have the GGUF harness and a measured
   0.000507 cross-engine floor.
   *Tells us*: whether architecture-aware GGUF allocation beats unsloth's UD ladder at half
   the bytes — the GGUF-side analogue of our own thesis. Anchors: Q8_0 0.001087, Q6_K
   0.002035, UD-Q5_K_XL 0.004444.

### Measured and worse

- **`huginnfork/Qwen3.8-27B-NVFP4A16`** — by its own table, KLD 0.1267 against official
  FP8's 0.0523 on the same harness: 142 % higher weight-only, 27 % higher as deployed. At
  28.87 GiB it is also outside a practical 32 GB envelope. It remains the best MTP writeup
  in the landscape (84.0 % acceptance, 484/576, plus the vLLM failure mode where a
  mis-loaded bf16 head yields 0 % acceptance while logging clean) — worth reading, not
  worth measuring.
- **`HarimxChoi/WarpQuant-…-R16E4H4`** — WT2 PPL 7.4737 at 3.6165 bpw against IQ3_S 7.1820
  at 3.6940 bpw. It loses to a stock GGUF quant at essentially the same bitrate, and the
  card publishes that fact.

### Not competitors — different constraint

CPU / NPU reach (`OpenVINO` int4 + int8, `amd/…-text-cpu-onnx`); non-NVIDIA accelerators
(11 `FlagRelease` repos); Apple unified memory (319 MLX / JANG / oQ / MTPLX repos — of
which `manjunathshiva/…-tq3-mini-g64` at 11.57 GiB is the most rigorous, but measured in
peak RSS on an M4 mini, not KLD); AMD Strix Halo fork-only tensor types (`lmcoleman`,
`jcbtc`, `rcmorano`, `kingjones777`, `bg-digitalservices`, `hugobugo34`, `JackBinary`,
`julianmb`, `singulared`, `pugant`); the NInfer engine, whose NVFP4 identity is not even
registered upstream (`Neroued/ninfer#25`); ik_llama.cpp (`cHunter789`, `ubergarm`,
`Thireus`) which needs its own cross-engine floor established first; `tlwu`'s ONNX export,
which can only re-measure the exporter since it *is* the artifact we already measured; and
the ~124 abliterated / heretic / uncensored / distill derivatives, which are different
models.

Long-context watchlist, not measurement targets:
`mixbits/…-NVFP4-MTP-VL-GGUF` ships the official static-YaRN 1,048,576 recipe as a
loadable default by rewriting `qwen35.context_length` in the GGUF KV — honest about the
fact that it is metadata, not weights, and irrelevant on 32 GB since 1M FP16 KV is
~61–64 GiB. `AlexanderKyng/Qwen3.8-27B-48Gb` does full 262,144 F16 KV with MTP
self-speculation on dual 3090s — a 48 GB target, but a useful proof that F16 KV at 262k is
achievable at all.

---

## 5. Reputation: who has *not* shipped, and who cites us

### Credible quanters with nothing for this model

Checked by enumerating each org's repos, not by memory:

**`RedHatAI` / `neuralmagic` — nothing.** This is the loudest absence: `compressed-tensors`
is their format and unsloth's recipe is their reference recipe. But
`nm-testing/Qwen3.8-27B-NVFP4-FP8` appeared on 2026-08-16 with an empty card, the same
mixed NVFP4+FP8 group structure at `compressed-tensors` 0.18.1, and unsloth's exact
`model_mtp` blob. Read that as staging, not absence. `mgoin` (Red Hat / vLLM) has two
Qwen3.8 repos, both for the 2.4T-A95B, none for 27B.

**`nvidia` — nothing.** NVIDIA normally ships first-party ModelOpt NVFP4 for flagship
models. Here the ModelOpt work is entirely community (`RadixArk`, `gittensor-model-hub`,
`vroomfondel`, `PassingByPixels`, `kristianpaul`, `a2genesis`, `joshebbs`). `amd` shipped
four builds in the same window — three Quark (`AWQ-MXFP4`, `Qronos-INT4-W4A16`,
`AWQ-INT4-W4A16`) plus a CPU ONNX export. NVIDIA shipped zero.

**`Intel` — nothing**, despite AutoRound being their tool and 20 community AutoRound
builds existing (`kaitchup` ×6, `Vishva007` ×3, `dbirks` ×2, `Pilcothink`, `MIRALABS`,
`Avuja`, `biMEMO`, `MKRWW`, `Minachist`, `goldhub`, `devan-carlin`, `slopops`, …).

**`ModelCloud` / `QuantTrio` / `JunHowie` — nothing.** No GPTQModel-lineage build at all;
the GPTQ presence is `btbtyler09`, `SergiioB`, `cloudnathan5`, `YCWTG`, `Chungulus`,
`palmfuture`.

**The EXL crowd is almost entirely absent.** `LoneStriker`, `MikeRoz`, `ArtusDev`,
`bullerwins`, `async0x42`, `Downtown-Case`, `nytopop` — none. Only `turboderp` himself
shipped the reference ladder; the rest of the EXL3 corner is us, plus `Honkware` (4 repos,
three of them of *heretic-ara*), `Jon-Nielsen`, `Dampish`, `P4pps3n`, `darkbit1001`,
`WatchDG`, `akys`, `bombdefuser-124` — all uniform-bitrate single-rung builds, and most of
them of derivatives rather than the stock model. **We are still the only non-uniform EXL3
allocation for this model.**

Also nothing: `cognitivecomputations`, `anikifoss`, `ilintar`, `eaddario`, `noctrex`,
`Artefact2`, `tensorblock`, `second-state`.

Present as expected: `bartowski`, `mradermacher` (20 repos, mostly of derivatives),
`ubergarm`, `ggml-org`, `lmstudio-community`, `mlx-community` (15), `Mungert`, `Thireus`,
`kaitchup`, `nightmedia`, `inferencerlabs`, `DavidAU`, `gghfez`, `FlagRelease`, `OpenVINO`.

### Who cites our numbers or repos

Every one of the 952 model cards was fetched and scanned. **Exactly one external repo
cites us:**

**`chriswritescode/Qwen3.8-27B-MXFP8`** (rev `05f53164`, 2026-08-15). It uses
`malaiwah/qwen38-27b-fidelity-suite-v3` and `tools/fidelity.py` from
`malaiwah/qwen38-27b-exl3`, "used unmodified", and reports:

- its MXFP8 at mean KLD **0.008471** [0.006421, 0.010920], median 0.001795, top-1 96.80 %;
- `Qwen/Qwen3.8-27B-FP8` at 0.013365, losing 136/136 paired contexts (−0.004894,
  CI [−0.006547, −0.003522]);
- a replay control: re-scoring **our** published FP8 captures through their pipeline gives
  0.013201 against our published 0.013126, **+0.6 %**, with SHA-256-identical inputs;
- a capture-environment floor of **+0.000164**, CI [−0.000096, +0.000434] — includes zero.

That is an independent party reproducing our harness on different hardware (RTX 6000 Ada)
and getting our number back to within 0.6 %. It is the strongest external validation of
the v3 protocol we have, and it arrived without us asking.

One near-miss was rejected: `Honkware/Qwen3.8-27B-exl3-4.5bpw` matches a naive
`qwen38-27b-exl3` search only because of their own collection slug
(`Honkware/qwen38-27b-exl3-6a7f89ee…`). Not a citation. A loose full-text query against
HF's own search returned 79 apparent external hits; direct card scanning reduced that to one.

---

## 6. Sources not reached

**Hard failure, expected:** `unsloth/Qwen3.8-27B-NVFP4` at revision `9c73e2da…` returns
`404 Invalid rev id`, and `paths-info` at that rev returns `500`. No impact — the pin was
recovered from our archival mirror and cross-checked against `tlwu`'s derived config.

**Not attempted, by constraint:** no weight file was downloaded and no GPU work was run
(the 5090 is held by `KvDtypeSweep`). Every fidelity figure in this document is either
somebody else's published number or one of ours from a prior receipt. None was produced
here.

**Largest blind spot:** GGUF *internal* metadata. Reading per-tensor ggml types would mean
range-reading headers across 240+ GGUF repos. For GGUF repos, statements about which
tensors are quantised how are card claims, not verified facts — except where the card
publishes the mix itself (`empero-ai` Ridge, `AlexanderKyng`, `enginetown`).

**Partial coverage:** 464 of 952 candidates got full detail (metadata + card + `config.json`
+ weight index); the other 488 had their cards fetched and scanned but were not
structurally probed. They are, by construction, low-traction pre-cutoff repos with no focus
format. `Minachist/Qwen3.8-27B-INT8-AutoRound` has three non-`main` branches
(`main-gs128`, `linear-attn-bf16`, `linear-attn-bf16-gs128`) that were not inventoried;
`turboderp`'s seven bitrate branches were.

**Outside HF, not consulted:** GitHub issue trackers (`Neroued/ninfer#25` is cited from
Ostfralla's card, not verified upstream); `prismaquant.org`, `kaitchup.substack.com` and
`harimxchoi.github.io`, all linked from cards; Reddit and Discord threads, which
`RedditRefresh` and `CommunityRefresh` own.

**Caveat on counts:** Hugging Face reports a 30-day rolling download figure. Every count
here is as of 2026-08-16 and will drift.
