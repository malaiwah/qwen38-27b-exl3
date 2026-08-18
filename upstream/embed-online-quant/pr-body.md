# feat(exl3): online K-quant embedding table (`VLLM_EXL3_EMBED_ONLINE_BITS`)

## Motivation

On 32 GB consumer GPUs (e.g. RTX 5090, 31.4 GB usable) the unquantized token
embedding table is the single largest block of "easy" VRAM. For
**Qwen3.5‑27B‑EXL3** the table is `248320 × 5120` BF16 = **2.54 GB**
(2.37 GiB). With FP8 KV cache the run maxes out at ~212k native context — vLLM
reports `9.63 GiB KV needed, 8.04 GiB available`, i.e. a **~1.59 GiB shortfall**
for 256k. Freeing most of the embedding table closes that gap without touching
the checkpoint or the EXL3 linear/MoE paths.

This PR adds an **env‑gated, online (load‑time) quantizer for
`VocabParallelEmbedding`** that converts the BF16 table to a compact per‑row
format and frees the BF16 tensor. The checkpoint weights are never modified.

## Design

New `Exl3OnlineEmbeddingMethod(QuantizeMethodBase)` in
`vllm/model_executor/layers/quantization/exl3.py`, wired into
`Exl3Config.get_quant_method` for the embedding table only.

- `create_weights` mirrors `UnquantizedEmbeddingMethod`: a normal BF16
  `weight` Parameter with `input_dim=1, output_dim=0` and the stock vocab‑parallel
  `weight_loader`, so checkpoint loading is unchanged.
- `process_weights_after_loading` computes a per‑row symmetric scale, encodes
  the table, **deletes the BF16 `weight`**, registers compact buffers
  (`q_weight`, `embed_scale`, non‑persistent), and calls `torch.cuda.empty_cache()`.
  Logs GiB before/after (`EXL3 embed online K%d conversion complete …`).
- `embedding()` is a **CUDA‑graph‑safe gather + dequant**: it indexes the compact
  weight by token id (a pure gather — direct advanced indexing, not
  `F.embedding`, so integer/1‑D tensors are accepted uniformly), casts to bf16,
  and multiplies by the gathered per‑row scale. No host syncs, no `.item()`,
  steady‑state allocation limited to the gathered rows.
- `apply()` raises `NotImplementedError` (embeddings are never `.apply()`‑ed).
- `tie_weights()` raises `NotImplementedError`: online embed quant is incompatible
  with tied word embeddings; the EXL3 stack already unties `lm_head`.

### Formats

| `VLLM_EXL3_EMBED_ONLINE_BITS` | storage                            | footprint (248320×5120) | saves vs BF16 |
|-------------------------------|------------------------------------|-------------------------|---------------|
| unset / 0                     | BF16 (unchanged)                   | 2.54 GB                 | —             |
| 8                             | int8 `[V,H]` + fp16 scale `[V]`    | ~1.27 GB                | ~1.27 GB      |
| 6                             | packed int6 `[V,3H/4]` + fp16 scale | ~0.95 GB               | ~1.59 GB      |
| 3,4,5,7                       | N‑bit precision, int8 container    | ~1.27 GB                | ~1.27 GB      |

- **bits=8**: per‑row symmetric int8, `scale = amax(row)/127`, `q` clamped to
  `[-128,127]`, dequant `q*scale`.
- **bits=6**: per‑row symmetric int6, `scale = amax(row)/31`, `q` clamped to
  `[-32,31]`, stored unsigned `[0,63]` and **packed four elements to three bytes**
  (`4×6 = 24 bits = 3 uint8`); dequant unpacks the 24‑bit group and reverses.
  Requires `hidden % 4 == 0` (5120 ✓).
- Other widths quantize to the requested precision but stay in an int8
  container (no extra footprint reduction vs 8); a one‑time warning is logged.

### Gating / safety

- Env var `VLLM_EXL3_EMBED_ONLINE_BITS` (accepted: unset/0 = off, 3..8). Validated
  at first read; invalid values raise `ValueError` fail‑fast.
- Wired with an **exact `type(layer).__name__ == "VocabParallelEmbedding"`**
  check. `ParallelLMHead` *subclasses* `VocabParallelEmbedding`, so an
  `isinstance` check would wrongly quantize the LM head; the exact‑type check
  keeps `ParallelLMHead` on its ExL3 linear/head path.
- **Inert when unset**: the branch is guarded by `_embed_online_bits() is not None`
  (short‑circuited on type, so the env read doesn't even happen for non‑embedding
  layers). With the var unset, `get_quant_method` returns the same values as
  before and the caller falls back to `UnquantizedEmbeddingMethod`.
- Multimodal‑safe: vision tokens bypass the token table (image‑token replacement),
  so quantizing `embed_tokens` does not affect the vision path.

## KLD‑safety precedent

The qualified `gg-r34-patched` profile already ran **8‑bit embeddings**
(`VLLM_EXL3_EMBED_BITS=8`) with **KLD 0.002700 PASS**, establishing 8‑bit
embedding quantization as KLD‑safe for this model. bits=8 here matches that
precision; bits=6 trades a little accuracy for the extra ~0.32 GB.

## Why not EXL3 Trellis K6/K8 (preferred)?

Trellis would be smaller and KLD‑safer, but the shipped **exllamav3** extension
cannot back an embedding *lookup*:

- `bindings.cpp:96-97` exposes `reconstruct` and `reconstruct_slice`.
- `reconstruct.cuh:14-22` / `reconstruct.cu:99-141`: `reconstruct_slice(unpacked,
  packed, K, mcg, mul1, n_offset)` reconstructs a **contiguous, 128‑aligned band
  of the N dimension** for the **full K dimension** (`reconstruct.cu:118-121`:
  `unpacked.size(1) % 128 == 0`, `n_offset % 128 == 0`). There is **no
  arbitrary‑row (indexed) reconstruct**.
- An embedding lookup gathers **scattered** vocab rows (N = vocab). A Trellis‑backed
  gather would therefore either (a) reconstruct the whole table to fp16 up front
  (defeats the savings), or (b) launch one `reconstruct_slice` per 128‑row band
  touched by the batch — a **dynamic** band count, not CUDA‑graph‑capturable, and
  ~1.3 MB fp16 scratch per band.

So this PR ships the int8/int6 fallback. The Trellis row‑indexed reconstruct is
the upstream ask tracked in the linked issue.

## Verification

- `python3 -c "import ast; ast.parse(open('.../exl3.py').read())"` — OK.
- Numeric round‑trip (CPU, torch 2.13): int8 max‑err 0.0625 (rel 0.96%); int6
  max‑err 0.25 (rel 3.9%); **int6 pack/unpack integer round‑trip exact**; all 64
  6‑bit values survive packing; zero‑row handling correct.
- CUDA‑graph safety by construction (all ops are capturable tensor ops; no
  `.item()`/host sync). GPU live test deferred (running service must not be
  disturbed); integration test to follow in‑container.
- Inertness confirmed by reading the diff: with the env var unset the new branch
  is never taken.

## Risks / non‑goals

- **weight_loader**: only the BF16 `weight` is loader‑aware; the compact buffers
  are created after loading, so sharding (`output_dim=0`) is unaffected. TP>1 keeps
  per‑row granularity per partition (structurally sound; tested path is TP=1).
- **LoRA added vocab**: not supported (same envelope as the LM‑head added‑vocab
  check); `process_weights_after_loading` operates on the loaded partition shape.
- **Tied embeddings**: explicitly rejected (`tie_weights` raises); the EXL3 stack
  already unties `lm_head`.
- Does not change any linear / MoE / LM‑head path.

## Branch / files

- Fork head: `malaiwah/vllm-voipmonitor:feature/embed-online-kquant`
- Base: `local-inference-lab/vllm:dev/gilded-gnosis`
- Changed file: `vllm/model_executor/layers/quantization/exl3.py` (one file,
  +228/-1).
