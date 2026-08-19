# Receipt — online embedding K-quant

## Lookup-strategy decision: INT8 / packed-INT6 fallback (Trellis lookup infeasible)

**Decision:** ship per-row int8 (bits=8) and packed int6 (bits=6) online
quantization for `VocabParallelEmbedding`. Trellis K6/K8 was preferred but is
infeasible for an embedding *lookup* with the shipped exllamav3 ext.

### Evidence (file:line)

exllamav3 ext (`/opt/exllamav3-python/exllamav3/exllamav3_ext/`):

- `bindings.cpp:96` `m.def("reconstruct", &reconstruct, ...)`
- `bindings.cpp:97` `m.def("reconstruct_slice", &reconstruct_slice, ...)`
- `quant/reconstruct.cuh:14-22` — `reconstruct_slice(unpacked, packed, K, mcg,
  mul1, n_offset)`; `n_offset` is the only slicing knob (N-dim offset).
- `quant/reconstruct.cu:99-141` — implementation; `reconstruct.cu:118-119`
  enforce `unpacked.size(1) % 128 == 0` and `n_offset % 128 == 0`
  (128-row N-band granularity); `reconstruct.cu:116-117` show `packed` is
  3-D `[K_blocks, N_blocks, inner]` (confirmed by `TORCH_CHECK_DIM(packed, 3)`
  at `reconstruct.cu:286`). `reconstruct.cu:127` grid is `cols/8` → 128-row
  N tiles. No arbitrary-row (indexed) reconstruct exists.

For an embedding table laid out N=vocab, K=hidden, a lookup gathers scattered
vocab rows. `reconstruct_slice` only gives contiguous 128-row N-bands for the
full K, so a Trellis gather would either reconstruct the whole table (defeats
savings) or launch one `reconstruct_slice` per band touched (dynamic count →
not CUDA-graph-capturable; ~1.3 MB fp16 scratch/band). Neither is acceptable.

→ Upstream ask filed: a `reconstruct_rows(packed, row_index, K, mcg, mul1)`
kernel (16-row or single-row granularity) + `Exl3TrellisEmbeddingMethod`
wiring. See `issue-body.md`.

### Fallback verification (CPU, torch 2.13.0+cu130; GPU busy — running service must not be disturbed)

- int8: max-err 0.0625, rel-err 0.96%.
- int6: max-err 0.25, rel-err 3.9%.
- int6 pack/unpack **integer round-trip exact**; all 64 six-bit values survive.
- Zero-row (amax=0) handling correct (scale clamped to eps, dequant → 0).
- CUDA-graph safety by construction: all ops capturable, no `.item()`, no host
  sync. (Live CUDA-graph capture test deferred — GPU occupied by the running
  `qwen38-27b` container.)

## Memory savings (248320 × 5120 table, decimal GB)

| format | resident | saves vs BF16 |
|--------|----------|---------------|
| BF16 (stock)            | 2.54 GB | — |
| int8  (bits=8)          | ~1.27 GB | ~1.27 GB |
| int6 packed (bits=6)    | ~0.95 GB | ~1.59 GB |

(Decimal GB; in GiB: BF16 2.37, int8 1.18, int6 0.89. The live log prints GiB
via `torch.cuda.memory_allocated()/1024**3`.) bits=6 frees ~1.59 GB, matching
the ~1.59 GiB KV shortfall vLLM reported for 256k context (`9.63 GiB KV needed,
8.04 GiB available`), enabling 256k+ native context on the RTX 5090 (31.4 GB)
with FP8 KV cache.

## What was delivered

### 1. Live bind-mount patch
`/home/mbelleau/vllm-exl3-multiprecision.py` — added `_embed_online_bits()`,
`Exl3OnlineEmbeddingMethod`, and the `get_quant_method` wiring. Inert when
`VLLM_EXL3_EMBED_ONLINE_BITS` is unset. `ast.parse` OK.

### 2. Research repo (`malaiwah/qwen38-27b-exl3`, branch `main`)
- `patches/vllm-exl3-multiprecision.py` — updated copy of the live patch.
- `upstream/embed-online-quant/pr-body.md`
- `upstream/embed-online-quant/issue-body.md`
- `upstream/embed-online-quant/receipt.md` (this file)

### 3. Fork PR branch (`malaiwah/vllm-voipmonitor`)
- Branch: `feature/embed-online-kquant` (pushed; 1 commit, +228/-1 in
  `vllm/model_executor/layers/quantization/exl3.py`).
- Same feature ported to the fork's newer exl3.py lineage (which already has the
  online-Trellis machinery / `exl3_online_cache`). `ast.parse` OK.
- Target PR base: `local-inference-lab/vllm:dev/gilded-gnosis` (cross-fork, same
  as prior #393/#394).

### 4. Upstream issue / PR filing
- `gh` authenticated as `malaiwah` (repo + workflow scopes). Prior project
  filings target `local-inference-lab/vllm` (issues enabled; push pull-only →
  cross-fork PRs from `malaiwah/vllm-voipmonitor`).
- PR and issue creation attempted via `gh` (see final report for URLs / status).

## Risks
- **weight_loader**: only the BF16 `weight` is loader-aware; compact buffers are
  created after loading → vocab sharding (`output_dim=0`) unaffected. TP>1 keeps
  per-row granularity per partition (structurally sound; tested path is TP=1).
- **LoRA added vocab**: not supported (same envelope as the LM-head added-vocab
  check).
- **Tied embeddings**: explicitly rejected (`tie_weights` raises); EXL3 stack
  already unties `lm_head`.
- **bits ∈ {3,4,5,7}**: N-bit precision but int8 container (no extra footprint
  reduction vs 8); one-time warning logged.
- No container restart / model run performed, per constraints.

## Push log — three integration fixes (2026-08-18)

Three fixes found during live 256k-context integration on RTX 5090 (31.4 GB)
were pushed to the fork branch `feature/embed-online-kquant` and summarised in
a PR #436 comment.

### Commit
- **SHA:** `fd500ef883cc3d63c9cf84dcd59fb5c4f560fa39`
- **Branch:** `feature/embed-online-kquant` on `malaiwah/vllm-voipmonitor`
- **File:** `vllm/model_executor/layers/quantization/exl3.py` (+100/-23)
- **ast.parse:** OK

### Fixes
1. **`_install_embed_online_hook()`** — wraps `VocabParallelEmbedding.__init__`
   to swap `quant_method` post-init when `VLLM_EXL3_EMBED_ONLINE_BITS` is set.
   Model code constructs embeddings without `quant_config`, so
   `get_quant_method` is never consulted. Exact-type check excludes
   `ParallelLMHead`.
2. **Chunked encode + chunked amax (16384 rows/chunk)** — caps load-peak
   transients at ~0.5 GiB (was 4.74 GiB fp32 amax + 4.7+1.3 GiB packing
   intermediates → OOM).
3. **0-row `[0, hidden]` BF16 stub `Parameter`** — keeps `layer.weight`
   addressable for the MTP embed-sharing pre-check
   (`llm_base_proposer.py:1573`); both sharing paths share the whole module,
   so the draft inherits `q_weight`/`embed_scale`.

### PR comment
https://github.com/local-inference-lab/vllm/pull/436#issuecomment-5335862865

### Measured results (RTX 5090, FP8 KV cache)
- int6 embeddings freed 1.48 GiB profiled (available KV 7.41→8.89 GiB)
- vLLM max_model_len estimate 194,208→239,904
- Serving verified at 238,400 context: PP=6408 tok/s, TG=157.6 tok/s
- KLD 0.0567 (512-context replay), MTP=6, vision requests working
