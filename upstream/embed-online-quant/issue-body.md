# Feature request: row-indexed (gather) Trellis reconstruct in exllamav3 ext + EXL3 online embedding quantization

## Summary

EXL3 Trellis K6/K8 is the preferred format for online‑quantizing the token
embedding table (`VocabParallelEmbedding`) — it is smaller and KLD‑safer than
per‑row int8/int6. However, the shipped **exllamav3** extension cannot back an
embedding *lookup* because its reconstruct kernels only operate over
**contiguous, 128‑aligned bands** of the matrix N dimension, not arbitrary rows.
This issue asks for a **row‑indexed (gather) reconstruct kernel** in the ext,
plus the wiring in `vllm/.../quantization/exl3.py` to use it for embedding
gather + dequant.

## Why embeddings want Trellis

For a `248320 × 5120` BF16 embedding table (2.54 GB), Trellis K6 ≈ 0.95 GB and
K8 ≈ 1.27 GB, freeing up to ~1.59 GB of VRAM — the difference between ~212k and
256k+ native context on 32 GB GPUs with FP8 KV cache. 8‑bit embeddings are
already KLD‑safe for this model (KLD 0.002700 PASS in the qualified
`gg-r34-patched` profile), so sub‑byte Trellis is the natural next step.

## The gap (with file:line evidence)

The ext (`exllamav3_ext`) exposes only:

```
# exllamav3/exllamav3_ext/bindings.cpp
96:    m.def("reconstruct", &reconstruct, "reconstruct");
97:    m.def("reconstruct_slice", &reconstruct_slice, "reconstruct_slice");
98:    m.def("reconstruct_fp8dg_nt", &reconstruct_fp8dg_nt, "reconstruct_fp8dg_nt");
```

```cpp
// exllamav3/exllamav3_ext/quant/reconstruct.cuh
 5: void reconstruct(at::Tensor unpacked, at::Tensor packed, int K, bool mcg, bool mul1);
14: void reconstruct_slice(
15:     at::Tensor unpacked, at::Tensor packed, int K, bool mcg, bool mul1,
21:     int64_t n_offset
22: );
```

```cpp
// exllamav3/exllamav3_ext/quant/reconstruct.cu  (reconstruct_slice)
116:    int rows = packed.size(0);          // K blocks
117:    int packed_cols = packed.size(1);   // N blocks
118:    TORCH_CHECK(unpacked.size(1) % 128 == 0, "unpacked N dimension must be divisible by 128");
119:    TORCH_CHECK(n_offset % 128 == 0, "n_offset must be divisible by 128");
120:    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
121:    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "...");
```

`packed` is 3‑D `[K_blocks, N_blocks, inner]` (`reconstruct.cu:116-117`;
`reconstruct_fp8dg_nt` at `reconstruct.cu:286` `TORCH_CHECK_DIM(packed, 3)`).
`reconstruct_slice` reconstructs a **contiguous band of the N dimension** (the
vocab dimension, for an embedding table laid out as N=vocab, K=hidden), at
**128‑row granularity**, for the **full K dimension**.

An embedding lookup gathers **scattered** vocab rows by token id. With the current
API the only options are:

1. **Reconstruct the whole table** to fp16 up front → 2.54 GB fp16 resident,
   defeating the savings.
2. **One `reconstruct_slice` per 128‑row band touched by the batch** → a
   *dynamic* number of kernel launches (depends on which tokens appear), each
   writing a 128×hidden fp16 tile (~1.3 MB scratch at hidden=5120), then a
   secondary gather out of each band. The dynamic launch count is **not
   CUDA‑graph‑capturable**, and the per‑step cost is far higher than a single
   `F.embedding`‑style gather.

Neither is acceptable for a serving embedding path.

## Ask

1. **New ext kernel**, e.g.:

   ```cpp
   // reconstruct_rows: dequantize an arbitrary set of N (vocab) rows.
   void reconstruct_rows(
       at::Tensor unpacked,   // [num_rows, K] fp16 output
       at::Tensor packed,     // [K_blocks, N_blocks, inner] trellis
       at::Tensor row_index,  // [num_rows] int64, N-dim indices into the table
       int K, bool mcg, bool mul1
   );
   ```

   Semantics: for each `i`, `unpacked[i] = dequant_trellis(packed, row_index[i], K,
   mcg, mul1)`. The kernel already tiles N in 16‑row blocks and K in 16‑row blocks
   (`reconstruct_kernel`, `reconstruct.cu:14-85`); a row‑indexed variant would
   gather the relevant 16×16 tiles for each requested row. Output fp16 (matches
   existing `reconstruct`); bf16 cast is cheap in Python.

2. **Wiring in `vllm/model_executor/layers/quantization/exl3.py`**: an
   `Exl3TrellisEmbeddingMethod(QuantizeMethodBase)` whose
   `process_weights_after_loading` encodes the BF16 table to Trellis (reusing the
   existing `exl3_online_cache.load_or_quantize` / `quantize_exl3` encoder) and
   whose `embedding()` calls `exllamav3_ext.reconstruct_rows` on the gathered
   indices — CUDA‑graph‑safe, no host sync, no per‑band launch.

   The online‑cache encoder already produces the `{trellis, suh, svh}` payload and
   the `mcg`/`mul1` flags; `get_quant_method` already has the
   `VLLM_EXL3_ONLINE_TRELLIS_BITS` (3..8) gating precedent to mirror for an
   embedding‑specific `VLLM_EXL3_EMBED_ONLINE_BITS`.

## Interim fallback shipped

Until the ext lands `reconstruct_rows`, we ship a per‑row **int8/int6** fallback
(`Exl3OnlineEmbeddingMethod`, `VLLM_EXL3_EMBED_ONLINE_BITS`): bits=8 → int8
`[V,H]` + fp16 scale `[V]` (~1.27 GB); bits=6 → packed int6 (4 elems → 3 bytes)
`[V,3H/4]` + fp16 scale (~0.95 GB). `embedding()` is a graph‑safe gather + dequant
(direct index, cast bf16, multiply by gathered scale). This recovers most of the
VRAM now and can be upgraded to Trellis K6/K8 unchanged from the env‑var / config
surface once `reconstruct_rows` exists.

## Feasibility notes

- `reconstruct_slice` proves the per‑tile dequant (`dq_dispatch<K,cb>`,
  `reconstruct.cu:41`) is separable from the GEMM path; a row‑indexed wrapper is
  mechanically tractable.
- The 128‑row alignment constraint in `reconstruct_slice` (`reconstruct.cu:118-119`)
  is an artifact of the `cols/8` grid (`reconstruct.cu:127`), not a fundamental
  Trellis limit (the base tile is 16×16). A `reconstruct_rows` kernel with a 16‑row
  (or even single‑row) granularity would remove the lookup‑time alignment pain.

## Related

- Interim int8/int6 fallback PR: `malaiwah/vllm-voipmonitor:feature/embed-online-kquant`.
- Existing online‑Trellis machinery: `exl3_online_cache.load_or_quantize`,
  `VLLM_EXL3_ONLINE_TRELLIS_BITS` (3..8) in `exl3.py`.
