# 41. The b12x / SparkInfer lever surface: what exists, what is reachable, and what to build next

**Decision: the next thing to implement is not a b12x knob. It is `VLLM_USE_V2_MODEL_RUNNER=1` plus a
per-batch-size speculative depth schedule. Everything b12x-shaped below it is either already on, structurally
unreachable, or worth less than its build risk — and the one FP8 prefill lever the brief hoped for is a
measured negative that would cost 4.4× this artifact's entire quantization error.**

**Status 2026-08-16 — that rank-1 item is now measured, and the brief was wrong about one thing: it was not
configuration only.** The configuration could not start at all until a fork-local FlashInfer gate bug was
fixed ([#398](https://github.com/local-inference-lab/vllm/pull/398), closing
[#396](https://github.com/local-inference-lab/vllm/issues/396)). With the fix the schedule measures
**+38.0 % aggregate decode at C8** (416.29 against a matched in-window 301.63 tok/s) with C1 held —
better than the +31 % estimated below — **but only at 131,072 / 0.95, because at the published
262,144 / 0.97 profile the V2 runner OOMs in the EXL3 prefill reconstruct on the first 2,048-token
prefill.** So the lever is real, it costs 14.1 % of KV tokens today, and W1 below carries the full
measurement.

This is a build-plan document, not a benchmark. **Nothing in it was measured during this work**: the
physical RTX 5090 was held by a sibling agent for the whole window, so every number here is either a
re-quotation of an existing receipt (labelled **measured**), a statement by the kernel authors
(**vendor claim**), or my own arithmetic over those two (**[INFERENCE]**). Machine-readable form with
`file:line` for every claim: `receipts/b12x-lever-map.json`.

Read the rootfs as `/var/tmp/gg-rootfs`; `$SP` below is
`/var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages`.

---

## 0. Five things in the brief that turned out to be wrong

I was asked to interrogate the standing findings rather than restate them. Five did not survive.

| # | The brief said | What the image and the repos say |
|---|---|---|
| 1 | the mamba page forces attention block size **816/832/896** | the logged value is **1600** (1584 at MTP depth 1). **816 is arithmetically unreachable** — the splitter needs the storage block to be a multiple of 64 or 128, and 816 is neither. `docs/36` already said 1600; the brief's numbers are stale. |
| 2 | `VLLM_EXL3_PREFILL_FP8`'s **1.5–1.8× prefill** is a rebuilt extension away | 1.5–1.8× is an **isolated-GEMM microbenchmark**, ours, not a vendor claim. The end-to-end figure was already measured **by us** and published as **+31 % prefill at +0.0141 mean KLD, deliberately default-off**. And `reconstruct_fp8_slice` has never existed in any published revision of anything — it is our own uncommitted patch. |
| 3 | FP8 activations were measured as **+31 % prefill *cost*** | the sign is inverted. +31 % is the **gain**; +0.0141 mean KLD is the cost. |
| 4 | **48 linear-attention and 163 full-attention** layers | `config.json` `text_config.layer_types` is **48 linear_attention + 16 full_attention** over 64 layers, `vision_config.depth` 27, `mtp_num_hidden_layers` 1. There is no 163. The 208 figure that floats around is the count of *attention-side EXL3 projections*: $48\times3 + 16\times4 = 208$. |
| 5 | **`mul1`** may be silently costing us fallbacks on 208 projections | the served checkpoint has **409 `mcg` markers and zero `mul1`**. All 208 attention projections fail exactly one clause — `trellis.shape[2] == 96`, i.e. *bits must be 6* — because our attention is serialized at **K5**. `mul1` costs the *public K4* build one tensor (`lm_head`), which is also that build's only K6 matrix. |

A sixth correction is upstream's, aimed at a conclusion of ours rather than the brief's: our 3,374 tok/s
prefill is **not** an EXL3 floor. `local-inference-lab/vllm` PR #316 and #318 measure **5,050–5,250 PP2k**
for a dense Qwen3.8-27B EXL3 on a 188-SM RTX PRO 6000. Scaling by SM count alone (188/170) predicts
≈3,732 for us, so roughly 30 % of that gap is *not* explained by the smaller card. The source-verified
candidates for the residual are our hybrid geometry and head_dim 256 — see §5.

---

## 1. The surface, from the image's own bytes

`b12x 1.2.1` is 296 files exposing **20 logical ops** in six groups, declared statically at
`$SP/b12x/__init__.py:41-62`:

```
attention.{paged, dense_mla, sparse_mla, compressed_mla, nsa_indexer, varlen}
comm.pcie
gemm.{blockscaled, block_fp8_linear, bmm, mxfp8_linear, tensor_fp8_linear,
      mla_query_projection, trellis_linear, wo_projection}
moe.{fused_moe, ep_moe}   norm.mhc   quantization.{mxfp8, nvfp4}
```

It reads **158 distinct `B12X_*` environment variables** (enumerated by walking every `.py` for
`os.environ.get` / `os.getenv` / `_env_flag`, convention at `b12x/_lib/env.py:7-19`). vLLM adds
**19 `VLLM_*B12X*` variables** (`$SP/vllm/envs.py:61-94, 249`) and a `--linear-backend b12x` /
`--fused-moe-backend b12x` pair (`$SP/vllm/config/kernel.py:193, 214`). One vLLM `general_plugins`
entry point is installed and auto-loaded: `b12x_fp6` (`b12x-1.2.1.dist-info/entry_points.txt`).

Full classified inventory is in the receipt (`knob_inventory`). The count that matters:

| state | count | what it means for us |
|---|---|---|
| **(a) live and already on** | 4 | `gemm.trellis_linear` K6/MCG, the K6/MCG small-M CUDA kernel, `VLLM_EXL3_PREFILL_RECONSTRUCT_M`, `VLLM_EXL3_GRAPH_DECODE` |
| **(b) live and off by default** | 5 | `VLLM_EXL3_EXT_PATH`, the `b12x_fp6` plugin, `VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM` (unreachable), `freeze_kernel_resolution` (inert), the rank-sliced R7 arena knobs (inert) |
| **(c) present but declined for our shapes** | 13 | `B12X_ATTN`, the generic CuTe trellis scheduler, `run_w4a8`, every `--linear-backend b12x` kernel, MHC / WO / MoE / sparse-indexer / MLA / PCIe / absorb-BMM, and 150-odd of the 158 env knobs |
| **(d) absent from the build** | 3 | b12x's `mul1` codebook, the fused K6/MCG CuTe specialization (b12x PR #221), `gemm.bf16_gemv` (built but not wired) |

### 1.1 The only b12x code we actually execute

One path, and it is narrower than it looks.

> **CORRECTION 2026-08-17 (docs/47 F3): the 16.1 % figure below is a property of the `qwen38-ctx`
> checkpoint only — its attention/GDN projections are K5.** The **hydrated** K5/K6 tree
> (`Qwen3.8-27B-EXL3-K5K6-hydrated`) serialized those families at K6, so on the checkpoint we now
> ship, this same gate passes **261 of 409 matrices = 59.7 % of trellis bytes**. "b12x is a minor
> path" does not transfer to hydrated. Per-checkpoint census and measured consequences (b12x loses
> to exl3_gemm on lm_head and k/v at decode m): docs/47 §F3.

`Exl3LinearMethod._apply_one` (`$SP/vllm/model_executor/layers/quantization/exl3.py:2951-2990`) asks
`_b12x_trellis_k6_supported` (`:1202-1218`) and routes to `vllm::b12x_trellis_linear_out` or
`vllm::exl3_gemm`. The gate is:

```python
return bool(
    has_mcg and not has_mul1
    and trellis.dtype == torch.int16 and trellis.ndim == 3
    and int(trellis.shape[2]) == 96          # bits == 6
    and int(trellis.shape[0]) % 8 == 0        # K % 128 == 0
    and int(trellis.shape[1]) % 8 == 0)       # N % 128 == 0
```

I parsed the safetensors headers of the checkpoint the ship script actually serves,
`/var/tmp/work/qwen38-ctx`, and evaluated that predicate exactly:

| family | count | bits | K×N | b12x gate |
|---|--:|---|---|---|
| `mlp.down_proj` (+ MTP) | 65 | K6 | 17408×5120 | **pass** |
| `lm_head` | 1 | K6 | 5120×248320 | **pass** |
| `mlp.gate_proj` / `up_proj` | 130 | K5 | 5120×17408 | reject |
| `linear_attn.in_proj_qkv` | 48 | K5 | 5120×10240 | reject |
| `linear_attn.in_proj_z` | 48 | K5 | 5120×6144 | reject |
| `linear_attn.out_proj` | 48 | K5 | 6144×5120 | reject |
| `self_attn.{q,k,v,o}_proj` | 68 | K5 | 5120×12288 / 5120×1024 / 6144×5120 | reject |
| `mtp.fc` | 1 | K4 | 10240×5120 | reject |
| **total** | **409** | | | **66 pass (16.1 %)** |

409 `mcg` markers, **0** `mul1`. So b12x runs 66 of 409 matrices — and only at fewer than 128 rows.

### 1.2 At prefill, b12x contributes zero GEMM work

This is the single most important structural fact in the document, and it retires a whole class of
proposals. **Both** dispatch ops divert to exllamav3 `reconstruct`+`hgemm` above a row threshold:

* `vllm::b12x_trellis_linear_out` — `exl3.py:1287-1294`
* `vllm::exl3_gemm` — `exl3.py:985-993`
* threshold `VLLM_EXL3_PREFILL_RECONSTRUCT_M`, default 128, set explicitly to 128 in the live profile
  (`exl3.py:770-782`; `receipts/perf-sweep-5090.json` `profile.constant_across_every_row`)

At a 2,048-token prefill chunk **all 409 matrices** run
`had_r_128 → reconstruct[_slice] → hgemm → had_r_128`. b12x's only possible prefill participation is
`b12x_attn`, which does not load. **No b12x GEMM-side lever can move prefill.** Any prefill work must
target exllamav3's reconstruct/hgemm or the attention backend.

The dispatch deliberately lives *inside* the opaque custom op, and the reason is worth carrying forward
(`exl3.py:761-765`): a Python-level `rows >= N` branch is resolved once at trace time, the profile run
bakes in the prefill branch, and the decode graphs then capture reconstruct+hgemm — **measured
56.5 → 22.6 tok/s at C1**. Any new threshold added by any work item below inherits that constraint.

---

## 2. The main event: is the 64/128 attention page genuinely coupled to the mamba page?

**No. The decoupling is already implemented, already shipped in this image, and per-KV-cache-group. b12x
is the only attention backend in the tree that opts out of it, and it does so in three lines of the vLLM
adapter, not in a kernel.**

### 2.1 Where 64 and 128 come from on the b12x side

The kernel's atomic unit is a **64-row TMA tile**. 128 exists only because $128 = 2\times64$ and a vLLM
contract needed it. Maintainer Luke Alonso, b12x commit `edf8d3d` (2026-06-12,
*"MSA paged attention: page_size=128 (vLLM contract) + fp8 KV at 128"*), verbatim:

> Walk arithmetic stays in 64-token tiles; block id -> table entry via `entries_per_block`, table entry
> -> physical 64-row TMA tile via stride-derived `page_tiles_per_entry` (phantom tiles span the strided
> V-plane gap and are never addressed). page64 codegen byte-preserved.

`page_tiles_per_entry` is *stride-derived*, so 256 and 512 are arithmetically inside the same design —
they were simply never compiled. The constant is hard-coded in at least eight kernel sites
(`b12x/attention/paged/forward_paged.py:3131, 8902, 9946`; `forward_extend_generic.py:2978`;
`planner.py:1103, 1356, 1474, 1783, 1820`; `graph_replay.py:1370, 1473`; `_scratch.py:2139`) and
documented at `b12x/attention/nsa_indexer/MSA.md:29`. **Larger pages were never tried, requested or
rejected upstream**: `git log -S` and `--grep` over all 1,027 b12x commits, plus a body search over both
repos, return zero.

None of that matters, because with hybrid blocks the kernel is handed a 64 or 128 page regardless of the
storage block. **Relaxing the vLLM-side gate needs no kernel change.**

### 2.2 Where 1600 comes from on our side — reproduced exactly

The inflation is entirely `CudaPlatformBase._align_hybrid_block_size`
(`$SP/vllm/platforms/interface.py:764-934`). Two statements do the damage:

```python
# interface.py:898-901  — inflate the ATTENTION block until its page covers the mamba page
attn_block_size = kernel_block_alignment_size * cdiv(
    mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token)
# interface.py:914-925  — then pad the mamba page up to exactly equal it
attn_page_size = cache_config.block_size * attn_page_size_1_token
assert attn_page_size >= mamba_page_size
cache_config.mamba_page_size_padded = attn_page_size
```

with `kernel_block_alignment_size = max(min(backend.get_supported_kernel_block_sizes()),
cache_config.block_size)` at `:874-882`.

The inputs, all read from the image rather than assumed:

* **mamba page** — `MambaSpec.page_size_bytes` is just $\sum \mathrm{prod}(\text{shape})\times
  \text{sizeof}(\text{dtype})$ (`$SP/vllm/v1/kv_cache_interface.py:846-851`). The shapes come from
  `gated_delta_net_state_shape` (`mamba_utils.py:213-234`): conv
  $(10240,\,3+n_{\text{spec}})$ where $10240 = 128{\cdot}16{\cdot}2 + 128{\cdot}48$, and recurrent
  $(48,128,128)$. The dtypes come from `_mamba_state_dtype` (`:84-96`): conv follows the model dtype
  (bf16), recurrent follows `mamba_ssm_cache_dtype` (fp32 here). At MTP depth 3 that is
  $10240{\cdot}6{\cdot}2 + 48{\cdot}128{\cdot}128{\cdot}4 = 122{,}880 + 3{,}145{,}728 =
  \mathbf{3{,}268{,}608}$ B.
* **attention page per token** — $2 \times 4 \text{ kv heads} \times 256 \text{ head\_dim} \times 1
  \text{ B} = \mathbf{2048}$ B (`kv_cache_interface.py:217-231`). `fp8` maps to
  `KVQuantMode.FP8_PER_TENSOR` (`:62-74`), *not* a per-token-head mode, so no scale bytes are added.

Ratio $3{,}268{,}608 / 2048 = 1596.0$. Feed it through the formula:

| kernel alignment | attention block | mamba padding | matches a logged run? |
|---|--:|--:|---|
| 16 (`MultipleOf(16)`, FlashAttention/Triton) | **1600** | **0.25 %** | yes — `receipts/perf-sweep-5090.json`, MTP-3 |
| 16, at MTP depth 1 (ratio 1576.0) | **1584** | **0.51 %** | yes — same receipt |
| 16, at MTP depth 2 (ratio 1586.0) | **1600** | **0.88 %** | yes — same receipt |
| 64 | 1600 | 1.52 % | — |
| 128 | 1664 | 1.75 % | — |

All three logged block sizes and all three logged padding percentages fall out of the formula exactly.
So: **the mamba page size is a property of the GDN state layout and nothing else.** It is not a property
of `MambaSpec`, not of `mamba_cache_mode`, and not of any page allocator. What is coupled is the
*attention* block, and the coupling is a policy choice in one platform method.

### 2.3 The decoupling already exists, per cache group

`prepare_kernel_block_sizes` (`$SP/vllm/v1/worker/utils.py:346-387`) returns **one kernel block size per
KV cache group**:

```python
if isinstance(kv_cache_spec, AttentionSpec):
    selected_kernel_size = select_common_block_size(kv_manager_block_size, group_backends)
elif isinstance(kv_cache_spec, MambaSpec):
    kernel_block_sizes.append(kv_cache_spec.block_size)   # "no splitting"
```

`select_common_block_size` (`utils.py:277-343`) walks the int-format supported sizes descending and
returns the first that divides the storage block. `1600 % 128 = 64 \neq 0` but `1600 % 64 = 0`, so
**b12x would be handed a 64-token kernel page while the 48 GDN layers keep their 1600-token storage
geometry**. `BlockTable` then sets `use_hybrid_blocks` and `blocks_per_kv_block = 1600 // 64 = 25`
(`$SP/vllm/v1/worker/block_table.py:69-77`), and `get_kv_cache_shape` is called with the *kernel* block
size (`gpu_model_runner.py:7511-7570`). The base-class `supports_block_size` accepts any multiple and
says so in a comment (`$SP/vllm/v1/attention/backend.py:189-196`):

> With hybrid_blocks feature, the framework-level block size only needs to be a multiple of the kernel's
> requirement, even if the kernel requires a fixed block_size.

This is upstream `vllm-project/vllm#24486`, *"[Hybrid]: Decouple Kernel Block Size from KV Page Size"*,
**merged 2025-10-09**:

> This PR introduces a hybrid cache architecture that separates logical kernel block size from physical
> page size... This hybrid model decoupling enables independent development of high-performance
> operators **without being constrained by linear attention mechanisms like Mamba**.

Its motivating issue `#24280` (tomeras91, closed as completed) describes our situation verbatim —
*"672 tokens for nvidia/NVIDIA-Nemotron-Nano-9B-v2"* — and the reporter confirmed the fix works.

### 2.4 So what is actually blocking us: three lines in `b12x_attn.py`

| id | site | problem | fix | size |
|---|---|---|---|---|
| **G-A** | `b12x_attn.py:417-418` | `supports_block_size` overrides the base class with **exact** membership in `(64,128)`; consulted at `backend.py:343` during validation, so the backend is rejected before any cache exists | delete the override, or mirror `triton_attn.py:295`'s modulo test | 2 lines |
| **G-B** | `b12x_attn.py:690-696` | `B12XPagedAttentionImpl.__init__` validates `cache_config.block_size` — the **storage** block, 1600 — instead of the negotiated kernel page | drop it: the runtime already reads the page from the tensor at `_kv_page_size` (`:100-126`) and plans are pre-built for **both** 64 and 128 at `:707-715` | ~6 lines |
| **G-C** | `b12x_attn.py:85-97` | `_max_page_table_width` reserves `cdiv(L, kernel_page)`; under hybrid blocks the true width is `cdiv(L, storage) × (storage // kernel_page)` | compute from the storage block and the split ratio | ~4 lines |

G-B is not even a disagreement inside the file — `b12x_attn.py:103-107` already documents the split:
*"The KV manager can split the configured storage block into a smaller kernel page when another backend
shares its cache group."* The adapter's runtime path is correct; only its startup assertions are not.

**G-C is a third blocker nobody had named**, and it has an unanswered upstream symptom. At
$L = 262{,}144$, storage 1600, page 64: $\lceil 262144/1600 \rceil \times 25 = 164 \times 25 = 4100$
entries against 4096 reserved. That is exactly `local-inference-lab/b12x` issue **#29** (jeremylea,
opened 2026-07-11, **open, zero comments, no maintainer reply**):

> `RuntimeError: The size of tensor a (4096) must match the size of tensor b (4100) at non-singleton
> dimension 1` — *"This happens during startup without MTP, but only on first use with MTP."*

(The 4096-vs-4100 arithmetic is **[INFERENCE]** by me; upstream states only the error string.)
`mamba_cache_mode == "align"` adds `cdiv(max_num_batched_tokens, block_size)` at `:92-96` and
*accidentally* covers the shortfall; non-align mode does not. So a fix validated in align mode is not
validated in general.

Precedent for exactly this deletion: `vllm-project/vllm#36701`, *"[Core] Remove FlashAttention block size
restriction for hybrid models"* (tdoublep, **merged 2026-06-26**) — *"making the restriction
unnecessary"* — and that removal is already in our image (`flash_attn.py:80` returns plain
`[MultipleOf(16)]`). b12x is the **only** attention backend in the pinned tree narrowing this to
equality: `flashmla` [64], `cutlass_mla` [128], `sparse_mla` [256], `hpc_attn` [64], `b12x_mla_sparse`
[64] do not override at all; the three that do (`triton_attn:295`, `triton_mla:102`,
`rocm_aiter_unified_attn:49`) all use `% 16 == 0`.

**There is no free revision-bump win.** `b12x_attn.py` at `dev/gilded-gnosis` head `fa033bd4e` has md5
`4c54854746011046e6cc5a8dd2a475fe`, identical to the pinned copy; no commit has edited
`_B12X_SUPPORTED_PAGE_SIZES` since `b0d9820e3` introduced it on 2026-07-17.

One correction on the way out: the `TODO(tdoublep): this constraint can be relaxed fairly easily` at
`interface.py:886` is **not** about this coupling. It sits inside the `mamba_cache_mode == "all"` branch
and is scoped to mamba *chunk* alignment; its PR is `vllm-project/vllm#33194` (still open) and we run
`align`, so it would not touch us. Chasing it is a dead end.

### 2.5 …and what fixing it would actually buy

Honest answer: **not much**, which is why this lands at rank 4 and not rank 1.

* Attention representation is about a **tenth of prefill**: `docs/26` arms A→C and B→D measure 1.05×
  and 1.11×. **Measured.**
* head_dim 256 caps the prefill query tile at 64, never 128
  (`b12x/attention/paged/planner.py:698-704`, byte-faithful to the FlashInfer shipped in the same image
  at `flashinfer/data/include/flashinfer/utils.cuh:412`), and `_paged_is_invalid`
  (`traits.py:83-100`) independently forbids `cta_tile_q=128` at head_dim 256 because
  `num_mma_d_vo = 16` makes any `num_mma_q ≥ 2` exceed the 256-register budget. So we would pay twice
  the CTA count of a head_dim-128 model.
* **Every** SM120 tuned/analytic decode and verify family requires `head_dim_qk == head_dim_vo == 128`
  (`_forward.py:427-500`, `forward_paged.py:3155-3232`). Our 24 q heads / 4 kv heads / gqa 6 match those
  families *exactly* and miss only on head_dim. The one head_dim-256-exclusive TMA path
  (`forward_extend_generic.py:3008-3020`) additionally wants `page_size == 64` — which hybrid blocks
  would give us — but also `cta_tile_q == 16`, so it serves extend/verify, not a 2,048-row chunk.
* And `B12X_ATTN` advertises `AttentionCGSupport.UNIFORM_BATCH` (`b12x_attn.py:559`), which
  `resolve_cudagraph_mode_and_sizes` turns into `FULL_AND_PIECEWISE` — a mode EXL3 refuses. So this item
  **collides with rank 1** unless the mode is pinned to `FULL_DECODE_ONLY` explicitly.

**Verdict on the brief's hardest question:** it is a genuine knob, not an impossibility, and I am no
longer going to call it structurally impossible. It is ~12 lines of vLLM with no kernel change, on top of
capability that shipped upstream ten months ago. But its ceiling is a tenth of prefill against an
unpatched upstream page-table defect and a head_dim that declines every tuned path, so it is a
*third*-order lever wearing a first-order costume.

---

## 3. Costing the prefill rebuild — and why you should not do it

### 3.1 The rebuild is already done and sitting on this disk

| artifact | in image? | `reconstruct_fp8_slice` | `reconstruct_had_slice` | `reconstruct_fp8dg_nt` |
|---|---|---|---|---|
| `/opt/exllamav3/exllamav3_ext.cpython-312-x86_64-linux-gnu.so` (95.6 MB) | **yes** (manifest sha256 `829b9827…`) | no | no | **yes** |
| `/var/tmp/work/torch-ext/exllamav3_ext/exllamav3_ext.so` | **no** | **yes** | **yes** | no |

(`nm -D --defined-only | c++filt`, cross-checked with exact-line `strings`.) The second was built here
from `/var/tmp/work/exllamav3` with `-gencode=arch=compute_120,code=sm_120 -std=c++20 -O3
--use_fast_math`, `nvcc` from `/usr/local/cuda`, torch headers from the same venv. The loader override
`VLLM_EXL3_EXT_PATH` (`exl3.py:380-398`) prepends a directory to `sys.path` before importing
`exllamav3_ext`, so **`VLLM_EXL3_PREFILL_FP8` can be made live today with one environment variable and
no build.**

Qualification: the local build lacks `exl3_moe_fused`, `exl3_moe_fused_retile` and `exl3_moe_r7_fused`.
Those are the only names it misses out of the eleven `ext.*` attributes vLLM ever touches, they are all
routed-expert entry points, and they are required only at `exl3.py:5082-5091` and `:4270-4277` — both
MoE-only. Our checkpoint is dense (`text_config.num_experts` is null), so the local build is sufficient
**for this model and no other**.

**Can the pinned image build it?** Yes; the convert image is not needed. There are no preprocessor gates
of any kind — `grep '#if|#ifdef|#ifndef|__CUDA_ARCH__'` over `reconstruct.cu` and `bindings.cpp` returns
nothing, and `setup.py` globs every `.c/.cpp/.cu` unconditionally. **A symbol is present iff the
checked-out revision contains it.** The image pins
`brandonmmusic-max/exllamav3@704aefd` (branch `a1-retile-sm120`, `__version__ 0.0.43`, 2026-07-15);
`Dockerfile.vllm-b12x-cu132:45-47` and its assert at `:367-393` require the fork-only MoE exports, so a
naive repoint to upstream v1.4.x fails the build gate.

### 3.2 What "1.5–1.8×" applies to

`exl3.py:806-807` says *"An FP8 matmul is 1.5-1.8x faster than fp16 on these shapes (SM120,
measured)"*. Note the scope: **an FP8 matmul**, in an isolated microbenchmark of `torch.mm` fp16 against
`torch._scaled_mm` E4M3 over three MLP shapes. It is **ours**, not a vendor claim — zero hits for
"1.5-1.8x" in exllamav3, in b12x, or in any published repo. The end-to-end number exists and is
different by about 5×:

> **local-inference-lab/vllm PR #318** (open, 2026-08-15): *"Negative result recorded so nobody repeats
> it: emitting the reconstructed weight as FP8 E4M3 and using `torch._scaled_mm` gives **+31 % prefill**
> but costs **+0.0141 mean KLD** — worse than official FP8. Row-wise scaling does not help; **the loss is
> FP8 activations**. Left behind `VLLM_EXL3_PREFILL_FP8=1`, off by default."*

### 3.3 Same mechanism, and the trade is not close

It is the same mechanism as the "+31 % / +0.0141" result the brief half-remembered, with the sign
inverted. The image's own fidelity ledger, in comments the author wrote while measuring:

* `exl3.py:826-831` — per-tensor weight scaling cost **+0.0139 mean KLD**, *"worse than official FP8"*.
* `exl3.py:872-874` — the shipped path therefore uses **row-wise activation** scales plus
  **per-output-channel weight** scales. It still cost +0.0141.
* `exl3.py:912-916` — a landmine for anyone re-running it: *"out_dtype MUST be bfloat16 here. With
  row-wise scales and out_dtype=float16 this torch build returns silently wrong results — no error, ~6x
  relative error, and in serving it produced KLD 10.8 with 0.1 % top-1."*

Put +0.0141 next to what this artifact actually is. The published K5K6 figure is **mean KLD 0.00321**
over 10,480,640 scored positions (`receipts/kld5-10M-k5k6.json`). FP8 prefill costs **4.4× the entire
quantization error of the checkpoint**. That is not a trade-off to state and then take; it erases the
artifact's reason to exist. **Do not enable it. Do relabel the 1.5–1.8× number wherever it appears.**

### 3.4 The one FP8 door left, and why it is probably also shut

`reconstruct_fp8dg_nt` is a vendor kernel, **built into the shipped `.so`, with zero callers anywhere in
the image** (`reconstruct.cu:271`, `reconstruct.cuh:24`, `bindings.cpp:98`). It emits FP8 weights in
DeepGEMM NT layout with one FP32 scale per 128×128 tile, and `deep_gemm 2.5.0+a6b593d` is installed in
the same venv. Weights-only FP8 would leave activations alone, which is where PR #318 puts the loss.

**[INFERENCE]:** it does not escape. Every FP8 weight consumer on this stack is FP8×FP8 — DeepGEMM's NT
entry and `torch._scaled_mm` both require an FP8 A operand — so an FP8 weight operand *forces* FP8
activations. The only thing `reconstruct_fp8dg_nt` changes is weight-scale *granularity*, and granularity
is already ruled out: per-tensor cost +0.0139, per-output-channel cost +0.0141. Refining the weight
scaling did not help, so refining it further should not either. Ranked as a cheap **disproof** (W6), not
a lever.

---

## 4. What else is in b12x that we have never tried

| feature | site | qualifying rule | do our shapes qualify? |
|---|---|---|---|
| **`run_w4a8`** — direct E4M3 trellis decode into SM120 W4A8 MMA, no fp16 weight tile materialized | `b12x/gemm/trellis_linear/api.py:111-134`, `w4a8.py:43-45` | `_W4A8_CODEBOOKS == {'sqg_xor_cheb_t12'}`, `_PAIR_KINDS == {'P24','P33'}`; `prepare.py:2351-2352` | **No.** Our payload is unpaired MCG. Reaching it needs a re-quantized checkpoint in the QSRT P24/P33 sqg codebook — a converter project. Zero callers in vLLM. |
| **`b12x_fp6`** — MX-FP6 W6A6/W6A8 serving path, its own checkpoint format | `entry_points.txt`, `b12x/integration/vllm/plugin.py:53-54`, `b12x/quantization/mxfp6/*` | `B12X_FP6_MODEL_DIR` plus an MXFP6 checkpoint from `fp6_safetensors_export.py` | **No** without a new checkpoint. An entire alternative quantization we have never evaluated. |
| **generic CuTe trellis256 scheduler** with `_force_tile_config` / `_moe_block_size` | `w4a16/kernel.py:11045-11095, 11437-11478` | reached when `m > 128`; m-invariant tile (64,256) or (64,128) | **Never reached**: the prefill dispatch diverts every `m ≥ 128` call first. The vLLM adapter does not expose the tile override. |
| **`gemm.bf16_gemv`** — small-N BF16 GEMV | `b12x/gemm/bf16_gemv/api.py:19-25` | `1 ≤ m ≤ 8`, `K % 8 == 0`, bf16, 16-byte aligned W; routing `N ≤ 1024`, `K ≥ 1024` | **Not wired**: no reference outside the `b12x_fp6` plugin. Would help only at ≤8 rows — C1 with MTP-3 (4), not C4 (16) or C8 (32). Authors: *"loses by ~2x at m=16 — cap at 8."* |
| **fused K6/MCG CuTe specialization** (b12x PR #221) | absent from `gemm/trellis_linear/` in 1.2.1 | rtx6kpro#73: *"SM120/SM121, FP16, unpaired MCG K6, one expert, dimensions divisible by 128, and 1-16 input rows"* | **Would qualify** for the 66 K6 matrices at C1 (4 rows) and C4 (16), **not** C8 (32). See W3. |
| **b12x `mul1` codebook** | `w4a16/prepare.py:2004-2014`, `_lib/intrinsics.py:6148` | `prepare` raises outright; `kernel.py:167` allows only `{mcg, sqg_xor_cheb_t12}` | **Moot** (0 `mul1` tensors). Worth recording that `trellis_linear/__init__.py:32-36` *advertises* `w4a16/exl3_trellis_mul1_e4m3` and `w4a8/exl3_trellis_mul1_e4m3` recipes and both raise, and that `packed_decode_trellis_mul1_e4m3_to_e4m3x8` exists with zero callers. **The advertised recipe list is wrong.** |
| everything MLA / NSA / MoE / PCIe / MHC / WO-projection | `envs.py:61-94`, `models/deepseek_v4/*`, `models/minimax_m3/*` | architecture- or topology-gated | **No.** Qwen3.5 is dense GQA + GDN on one GPU. All six `VLLM_USE_B12X_*` toggles are off, which also makes `VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM` and `freeze_kernel_resolution` **dead config** for us (`compilation/b12x_capture.py:10-26`). |
| **158 `B12X_*` env knobs** | `b12x/_lib/env.py:7-19` | 33 paged-attention, 25 MHC-prefill-tiling, 15 PCIe, 12 NSA/MSA, 9 MLA, 8 `B12X_DENSE_*`, 8 compile/log, rest MoE + quant | only ~14 are on our live path, and all of them are cache, logging or codebook-staging controls — **none is a performance lever for a K6/MCG dense projection**. `B12X_DENSE_*` belongs to the FP4/FP8 dense GEMM (`dense_gemm.py:4430-4535` accepts no BF16/FP16 operand at all); `B12X_W4A16_SMALL_M_*` belongs to the MoE micro-kernel. |

---

## 5. Who actually owns the CUDA-graph refusal, and what it costs

**EXL3, unambiguously. Not b12x.** `Exl3Config._graph_decode_refusal`
(`exl3.py:1755-1799`) refuses any mode whose `mixed_mode()` is not `NONE` — which is `PIECEWISE`,
`FULL_AND_PIECEWISE` **and also bare `FULL`** (`config/compilation.py:56-70`) — and
`_require_enforce_eager` turns the refusal into a hard `ValueError` at `:1822-1828`. The stated cause
(`:1806-1811`) is exllamav3's, not b12x's:

> `exllamav3_ext`'s `exl3_gemm` autotunes with timing launches on the first call per (m-bucket, k, n, K)
> shape hash; under CUDA-graph capture those launches fault.

Two details that were invisible from the outside. First, **rank-sliced checkpoints are exempt**
(`:1802-1805`) — which is why upstream GLM-5.2 EXL3-TR3 runs `FULL_AND_PIECEWISE` with MTP-5 and we
cannot. Second, b12x's side of the house *expects* piecewise: `guard_b12x_kernel_resolution` is wired
into the **piecewise** capture site (`compilation/cuda_graph.py:333`), its `freeze_kernel_resolution`
is a fail-closed assertion rather than a capability limit (`b12x/_lib/runtime_control.py:17-64`),
`VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM` predates every EXL3 graph issue and is documented as a debugging
knob, and `B12X_ATTN` itself asks for `FULL_AND_PIECEWISE` via `UNIFORM_BATCH`. b12x#136's own validation
line reads *"FULL and PIECEWISE graph capture complete and coherent generation passes."*

**Would piecewise help prefill? It cannot even be dispatched.**
`v1/cudagraph_dispatcher.py:272-283` returns `CUDAGraphMode.NONE` whenever `num_tokens >
max_cudagraph_capture_size`, which defaults to `min(max_num_seqs*2, 512)`. Our prefill batches are 2,048
and 6,144 tokens. Every prefill step runs eager under either mode. And prefill is kernel-bound, not
launch-bound: `ext.hgemm` is at cuBLAS parity 0.92–1.06× (`docs/26`), so there is nothing for graph
capture to reclaim.

**Is the refusal fixable?** Yes, and PR #314's own evidence undercuts its own gate: the autotune hash
clamps with `MIN(roundup_pow2(MAX(size_m,2)),16)` (`exl3_gemm.cu:72`, call site `:241`), so **every
`m ≥ 16` — including 2,048 and 6,144 — hashes to the same bucket as `m = 16`**, and the priming pass
already covers 1…16. What remains genuinely unproven for mixed capture is everything *else*: the
`exl3_gemv_try_launch` small-m path (*"Not graph-capturable yet"*), `DevCtx::get_locks`'s lazy
`cudaMalloc`, and first-use allocation on the reconstruct arena. But **lifting it buys nothing
measurable**, per the paragraph above.

**What it does cost is one step removed, and here the brief was right for the wrong reason.** Dynamic
speculative decoding *is* supported at `FULL_DECODE_ONLY` in this image — the V2 draft-decode manager
explicitly rejects piecewise and upgrades to `FULL_DECODE_ONLY`
(`v1/worker/gpu/spec_decode/autoregressive/speculator.py:70-91`), and
`v1/worker/gpu/cudagraph_utils.py:215-276` captures one FULL uniform-decode graph per draft depth with a
`K=0` guard. That is upstream `vllm-project/vllm#45953` (merged 2026-07-04). **But our model does not
select the V2 runner by default**: `_is_default_v2_model_runner_model` returns `False` for
`is_hybrid` models (`config/vllm.py:640-666`), and `Qwen3_5ForConditionalGeneration` is declared
`IsHybrid` (`models/qwen3_5.py:411`). On V1, enabling dynamic depth makes vLLM **silently rewrite
`FULL_DECODE_ONLY` → `PIECEWISE`** (`config/vllm.py:862-887`) and EXL3 then raises. The escape is one
environment variable: `VLLM_USE_V2_MODEL_RUNNER` is honoured unconditionally (`:592-596`) and nothing in
`_get_v2_model_runner_unsupported_features` (`:2167-2261`) rejects a hybrid mamba model or MTP.

That is rank 1.

---

## 6. Ranked work items

Ordered by expected effect per unit of build risk. Full detail, including every `file:line`, in
`receipts/b12x-lever-map.json` → `ranked_work_items`.

### W1 — Per-batch-size speculative depth at `FULL_DECODE_ONLY` via the V2 model runner — **MEASURED 2026-08-16, and it required a code fix after all**
*Was "configuration only"; the configuration could not start until a fork-local kernel-side bug was fixed.*

**Result.** W1 is no longer closed-as-unavailable. The blocker was a FlashInfer gate admitting persistent
(CUDA-graph) decode wrappers only for `q_len == 1 + num_speculative_tokens`, so the V2 speculator's
draft-decode graph (`q_len == 1`) was captured *and* replayed on the **dynamic** wrapper, whose plan
rebinds `_paged_kv_indptr_buf` / `_paged_kv_last_page_len_buf` and reallocates `_qo_indptr_buf` on every
call - the captured graph replayed against freed plan buffers. Root cause proven with an instrumented
capture-versus-replay address log plus a control that changes nothing except pinning the capture-time
buffers alive (that alone makes the server ready). Fixed by keying wrappers on the shapes capture
actually planned: [local-inference-lab/vllm#398](https://github.com/local-inference-lab/vllm/pull/398),
closing [#396](https://github.com/local-inference-lab/vllm/issues/396). **Not upstream** - upstream's gate
has no `q_len` term and flattens spec-decode batches into single-token rows, and the three symbols the bug
lives in do not exist there, so there is nothing upstream to file.

**Measured, one server, T=0, aggregate decode:** the schedule `[[1,2,3],[3,8,1]]` gives **87.35 / 251.74 /
416.29 tok/s** at C1 / C4 / C8, i.e. **+38.01 % at C8 against an MRV1 baseline measured in the same window**
(82.78 / 255.75 / 301.63) with C1 held at +5.5 % - so the estimate below was not only confirmed but
exceeded, and it is +32.9 % even against the published 313.28 row. The mechanism is visible in the data:
2.25 accepted tokens per step per request at C1 (depth 3, 25.5 ms steps) and 1.69 at C8 (depth 1, 31.3 ms
steps), against the published `mtp1` C8 figure of 1.6754. The V2 runner *alone*, at static depth 3, is
neutral-to-positive: −0.04 % C1, +6.89 % C4, +3.73 % C8. Fidelity guard discharged: greedy T=0 output over
8 frozen prompts is token-for-token identical on two runs within each engine process in all four arms, and
acceptance does not fall (2.11-2.14 against a 2.14-2.16 baseline); cross-restart bit-exactness is not
claimable on this stack and is cited to `receipts/scratch-arena.json` rather than re-derived. Honesty
control: MRV1 on the published profile measured 1.6-5.3 % *hotter* today than its own receipt, which is
why every delta above is against the in-window matched baseline rather than the published numbers.

**The cost, which is the part that decides whether to ship it.** At `--gpu-memory-utilization 0.97` the V2
runner leaves 58.56 MiB free and the EXL3 prefill reconstruct OOMs on the first 2,048-row prefill
(`torch.OutOfMemoryError` in `_reconstruct_hgemm_into -> torch.empty_like(x)`), with the schedule **off and
on**, while 8x512-token prompts do not trigger it - which is why the engine looks healthy until a real
prefill arrives. 0.95 alone is refused (262,144 no longer meets the KV minimum), so the measured arms ran
**131,072 / 0.95 = 234,256 KV tokens against 272,570 published, −14.1 %**. Serving the published
262,144 / 0.97 profile under the V2 runner on a 32 GB card is **not demonstrated**, and that is the
shipping constraint recorded on the cards and in the PR body.
**Receipts:** [`v2-fault-fix.json`](../receipts/v2-fault-fix.json),
[`v2-runner-depth-schedule.json`](../receipts/v2-runner-depth-schedule.json).

#### Original plan entry, kept for the record
*Was: configuration only; no code.*

**Build:** set `VLLM_USE_V2_MODEL_RUNNER=1` and `num_speculative_tokens_per_batch_size`
(`config/speculative.py:181`) instead of a fixed `num_speculative_tokens=3`.
**Files:** `config/vllm.py:592-596, 640-666, 862-887, 2167-2261`; `config/speculative.py:181, 188,
1573-1578`; `v1/worker/gpu/spec_decode/autoregressive/speculator.py:70-91`;
`v1/worker/gpu/cudagraph_utils.py:215-276`.
**Estimated effect:** **+25 to +31 % aggregate decode at C8, with no loss at C1.**
**Basis — measured:** `receipts/perf-sweep-5090.json` — `num_speculative_tokens=1` at C8 gives
**409.35** tok/s against **313.28** for MTP-3 (+30.7 %), while MTP-3 is what wins at C1 (**82.94**). A
per-batch-size schedule takes both instead of choosing. Only the claim that the switch is overhead-free
is `[INFERENCE]`.
**Measurement:** re-run the decode rows at C1/C4/C8 in one server; require C1 ≥ 82.9 and C8 ≥ 400 tok/s;
confirm the startup log does **not** contain *"Overriding cudagraph_mode … to PIECEWISE"* and that the
EXL3 graph-decode line names `FULL_DECODE_ONLY`.
**What could break:** (1) forget the env var and you get a loud `ValueError`, not a silent regression.
(2) V2 + hybrid mamba + MTP + EXL3 + fp8 KV is a combination we have never run; V2 owns its own
KV/mamba plumbing, so re-verify context capacity and the GDN gate. (3) `local-inference-lab/vllm#298`
(open) warns a prompt chunk carrying `K+1` tokens per request can let a decode FULL graph replay over
prompt state, producing *"syntactically valid API responses containing repeated, raw-token, or
multilingual output"* — **run the fidelity guard, not just the throughput rows.**

### W2 — Widen `_b12x_trellis_k6_supported` from K6-only to every MCG bitrate b12x accepts
*One line of vLLM, gated behind a 5-minute microbenchmark.*

**Build:** `exl3.py:1215`, `int(trellis.shape[2]) == 96` → `in (32,48,64,80,96)`. That clause, and only
that clause, rejects all 208 attention projections and both MLP gate/up families — they are already
`mcg`, `int16`, `ndim 3` and 128-divisible on both axes. b12x's own dense path accepts bits 2–6
(`w4a16/kernel.py:166`, `prepare.py:2082-2087`).
**Estimated effect: unknown sign.** Possibly large at decode, possibly a regression.
**Basis:** the upside is **measured but only for K6** — `exl3.py:1275-1276`, *"measured on SM120 it beats
reconstruct+GEMM by ~5x at m=1-8"*. The catch is `[INFERENCE]` and structural: `_use_k6_mcg_small`
hard-requires `trellis_bits == 6` (`w4a16/kernel.py:11229`), so K5 shards admitted by a widened gate
would run the **generic** CuTe scheduler, not the tuned small-M CUDA kernel — and no measurement of the
generic scheduler at `m ∈ {1…32}` on a K5 shape exists anywhere.
**Measurement:** add a K5 arm and a small-m sweep to `tools/b12x_vs_reconstruct.py`:
`trellis_linear.run` vs `ext.exl3_gemm` at `m ∈ {1,4,8,16,32,128}` on 5120×17408, 5120×10240,
5120×6144, 6144×5120, 5120×12288. ~5 GPU-minutes, no serving run. Widen only if b12x wins at 4, 16
**and** 32 rows.
**What could break:** the 343 K5/K4 shards currently run the bit-faithful `ext.exl3_gemm`; moving them
changes the arithmetic and must be re-qualified against `receipts/decode-parity-*.json`, not merely
benchmarked. Also, the priming pass primes `exl3_gemm`; shards that move to b12x need b12x's own
warm-then-capture instead, or capture will resolve a new CuTe kernel inside capture.
**Upstream cover for the premise:** `local-inference-lab/vllm#297` (open, 2026-08-12) — *"The b12x PTX
dequant kernel is **parametric in `bits`** … **All restrictions were Python-level validation guards, not
compiled PTX limitations.**"* Widening this particular clause has **never been proposed upstream**.

### W3 — Bump the b12x wheel 1.2.1 → 1.2.4 for the fused K6/MCG CuTe specialization
*Vendor upgrade; image rebuild.*

**Estimated effect: at most ≈1.15× decode step at C1 and C4; nothing at C8; nothing at prefill.**
**Basis — vendor claim:** rtx6kpro#73 / b12x PR #221 report *"2.0x to 6.4x faster"* than the generic
separate-H128 scheduler across 15 projection shapes, E2E parity to +2.05 % prefill. The ceiling is mine
`[INFERENCE]`: the specialization covers unpaired MCG K6 at **1–16 rows only**, i.e. our 66 K6 matrices
at C1 (4) and C4 (16) but not C8 (32); those 66 carry **7.065 of 26.047 GMAC/token** of linear work
(27.1 %), so even a 2× kernel win bounds the whole-step improvement at 1.16× — and less, because decode
is weight-bandwidth bound rather than MAC bound.
**Measurement:** decode rows at C1/C4/C8 before and after, plus the microbench at `m ∈ {1,4,8,16}` on
17408×5120 K6.
**What could break:** a whole-image bump moves every other b12x kernel simultaneously, so any regression
is hard to attribute; and it does nothing for C8, the concurrency we actually serve.

### W4 — Make `B12X_ATTN` loadable on a hybrid model: relax G-A, G-B, G-C
*~12 lines of vLLM, no kernel change.* Detail in §2.4.

**Estimated effect: bounded above by ~10 % of prefill plus an unknown decode delta; realistically small.**
**Basis:** measured bound from `docs/26` (1.05×–1.11×); the rest `[INFERENCE]` from the head_dim-256
penalties in §2.5.
**Measurement:** after the three edits, start with `--attention-backend B12X_ATTN` and confirm the
startup log reports a **64**-token kernel page (because `1600 % 128 ≠ 0`), then run the existing prefill
and decode rows against the FLASHINFER baseline. Reaching 4,100 page-table entries without a crash
validates G-C independently and closes b12x#29.
**What could break:** b12x#29's page-table overflow is unpatched upstream and manifests *"only on first
use with MTP"*; align mode masks it, so an align-mode pass is not a general pass. And `UNIFORM_BATCH`
→ `FULL_AND_PIECEWISE` collides with W1 unless the mode is pinned.

### W5 — Wire exllamav3's fused `reconstruct_had_slice` into the vLLM prefill path
*~25 lines plus one env var; the extension is already built.*

**Estimated effect: predicted NEGATIVE — ≈1.31× the current cost on `gate_proj` at m=2048.**
**Basis — vendor claim vs our arithmetic.** Upstream
(`/var/tmp/work/exllamav3/exllamav3/modules/quant/exl3.py:171-184`) claims the standalone `had_r_128`
launches are *"~14% of long-chunk prefill GPU time"*, while conceding *"the fused kernel costs ~4x plain
reconstruct … breakeven is rows ~400-900 across shapes"*. Our measured inputs say that breakeven is far
higher on our geometries. At m=2048, `gate_proj` is **1.20 ms** total = **0.90** GEMM + **0.167**
reconstruct (`receipts/v3-prefill-micro.json`) + **0.133** for both Hadamard launches (`docs/26`,
by subtraction). Fused predicts $0.90 + 4\times0.167 = 1.57$ ms. Per element, Hadamard costs 2.88 ps and
reconstruct 1.88 ps — a ratio of only **1.53** — which puts breakeven at ≈**7,750 rows** for gate/up and
down, and ≈1,670 rows for `k_proj`/`v_proj`. Our `max_num_batched_tokens` is **2048**, so only the two
tiny projections would win. `[INFERENCE]`, and it **contradicts the naive reading of the vendor's 14 %**.
**Measurement (settles it in ~10 GPU-minutes, no serving run):** for 5120×17408 (K5), 17408×5120 (K6),
5120×248320 (K6) and 5120×1024 (K5), time `had_r_128 + reconstruct_slice + hgemm + had_r_128` against
`reconstruct_had_slice + hgemm` at `m ∈ {512, 1024, 2048, 4096, 8192}` using
`VLLM_EXL3_EXT_PATH=/work/torch-ext/exllamav3_ext`. This also *directly* measures the 0.133 ms Hadamard
figure that is currently a subtraction.
**What could break:** nothing in serving if the microbench says no — the item dies cheaply. If it says
yes for some shapes, the threshold must be per-geometry and must live **inside** the custom op, for the
same Dynamo reason documented at `exl3.py:761-765`. Fidelity: upstream's own tolerance is
`errs/scale < 2e-3` (`tests/test_reconstruct_had.py:67`), i.e. fp16 and not bit-exact, so the
decode-parity captures still apply.

### W6 — Disprove (or salvage) weights-only FP8 with the already-built `reconstruct_fp8dg_nt`
*One microbenchmark; an experiment, not a feature.* Reasoning in §3.4.
**Expected:** reproduces roughly the same +0.0141-class penalty. **Kill it** if mean KLD exceeds the
published 0.00321 by more than 2×. Value is closing the last FP8 door with evidence instead of inference.

### W7 — `VLLM_EXL3_PREFILL_FP8`: **do not do this**
Measured +31 % prefill for +0.0141 mean KLD = **4.4× the artifact's entire published error**. It can be
switched on today with one env var and it must not be. Action item is editorial: relabel "1.5–1.8×"
wherever it appears as an isolated-GEMM microbenchmark.

### W8 — Structural: the only remaining real prefill lever is a fused dequant-in-epilogue kernel
`ext.hgemm` is at cuBLAS parity (0.92–1.06×), so there is no free GEMM win; a *perfect* fused kernel
that made reconstruct free gives $64\times2.70 = 173$ ms, i.e. an 11.8k tok/s MLP-only ceiling and
7–8k end to end (`docs/26`). Maintainer statement, `local-inference-lab/vllm#316` comment
`5299700465` (2026-08-15): *"prefill parity with FP8 is not reachable for a 4-bit-class trellis format
by dispatch or tuning changes in this runtime. It needs dequant fused into the GEMM epilogue
(Marlin-style) — a new kernel."* Listed so the ranked list is honest about where the headroom is.

---

## 7. Measured / vendor-claimed / inferred — the separation, restated

**Measured (ours, with a receipt):** all decode and prefill throughputs; the 1.91× CUDA-graph win; the
+31 % / +0.0141 FP8 prefill result; mean KLD 0.00321; the 2,369 → 5,050 PP2k reconstruct win; the
per-shape ms in `b12x-vs-reconstruct.json` and `v3-prefill-micro.json`; the 1.20/0.90/0.167 ms
`gate_proj` decomposition; the 1600/1584 block sizes and their padding percentages.

**Vendor claim (cited, quoted verbatim, not verified by us):** the 64-row TMA tile rationale (b12x
`edf8d3d`); *"~14 % of long-chunk prefill GPU time"* and *"breakeven rows ~400-900"* for
`reconstruct_had_slice`; *"2.0x to 6.4x"* for b12x PR #221; *"Faster prefill (all models)"* in exllamav3
v1.4.0; the b12x README's SM120/SM121 targeting including the RTX 5090.

**[INFERENCE] (mine, labelled at every use):** the 4096-vs-4100 page-table arithmetic behind b12x#29;
the 7,750-row breakeven that predicts W5 loses; the ≈1.15× ceiling on W3; the reasoning that
`reconstruct_fp8dg_nt` cannot avoid FP8 activations; the claim that G-A/G-B/G-C are the *complete* set of
blockers for `B12X_ATTN`; the 27.1 % MAC share bound.

**Reproduced by executing the image's own formulas:** the 1600/1584/1600 block sizes and the
0.25 %/0.51 %/0.88 % paddings, and `select_common_block_size(1600, [b12x]) = 64`.

---

## 8. What I could not determine from the image without running it

1. **Whether the promoted serving image `localhost/vllm:gg-r34-patched-apc` (manifest
   `sha256:16a936b8…`) carries the same extension and source tree as the pinned base rootfs.** Neither
   podman, docker nor skopeo is installed on this host and I had no GPU window, so I could only read the
   base image `voipmonitor/vllm@sha256:820181fb…` through its flattened rootfs and 107,998-entry file
   manifest. Everything in §3.1 about *which* `.so` is baked in is a statement about the base image.
2. **Whether the generic CuTe trellis256 scheduler beats `ext.exl3_gemm` on a K5 shape at 4/16/32 rows.**
   This is the entire sign of W2 and no measurement of it exists anywhere. Five GPU-minutes.
3. **Whether the V2 model runner actually serves this hybrid VL model correctly.** Nothing in
   `_get_v2_model_runner_unsupported_features` rejects it, but "not rejected at config time" is not
   "known to work", and we have never run V2 on this checkpoint. This is the risk sitting under rank 1.
4. **The 0.133 ms two-launch Hadamard cost** underpinning the W5 refutation is a subtraction of two
   independently measured quantities, not a direct measurement.
5. **Device attribution for `receipts/b12x-vs-reconstruct.json`.** It carries no device field; its output
   path (`/work/kld3`) places it in the iteration-3 series, which `docs/26` documents as one RTX PRO 6000
   Blackwell. Treat its **ratios** as transferable and its absolute milliseconds as not ours.
6. **Nothing in this document was measured during this ticket.** The 5090 was held throughout.

---

## 9. Upstream provenance and the negatives

Four GitHub REST dossiers were built (full transcripts retained at `agent://B12xLeverPlan.UpBlockSize`,
`…UpPrefillFp8`, `…UpShapeGates`, `…UpGraphModes`). Coverage was by **exhaustive enumeration**, not
search: all **223** `local-inference-lab/b12x` issues+PRs (open and closed) and 522 comments; all **394**
`local-inference-lab/vllm` items and 847 comments; all **73** `local-inference-lab/rtx6kpro` items;
plus targeted upstream `vllm-project/vllm` and `turboderp-org/exllamav3` fetches. GitHub code and commit
search index only a repo's *default* branch, and this fork's default branch (`main`) is a stock vLLM
mirror with no `exl3.py` at all — so code-search zeros on that repo are weak evidence and were replaced
by contents-API fetches at explicit refs plus md5 comparison against the image.

The rename is confirmed mechanically: `GET /repos/local-inference-lab/SparkInfer` → `301` →
`/repositories/1183945688` → `local-inference-lab/b12x`, created 2026-03-17. b12x has **zero releases**;
its only tags are `1.2.4` and `pre-cute-rebase-20260807`, so there are no release notes to mine.
`si<hash>` identifiers appear only in the vLLM fork's image tags, never in b12x. One correction to our
own shorthand: **b12x is not a rename of exllamav3.** It is a separate CuTe-DSL kernel library by Luke
Alonso and contains none of the EXL3 reconstruct kernels; the two renames are independent lineages, which
matters when reasoning about pins.

**Negatives, so they are checkable:**

| question | result |
|---|---|
| anyone running b12x attention on a hybrid mamba model | exactly one report, **b12x#29**, open since 2026-07-11, **zero comments, no maintainer reply** |
| page size 256 or larger ever tried/requested/rejected | **zero** — `git log -S` and `--grep` over all 1,027 b12x commits, plus body search on both repos |
| `mul1` as a b12x topic | **zero** across 223 issues/PRs, 522 comments and all commit messages |
| head_dim 256 as a *performance* topic | **zero** in all four corpora; only correctness mentions (b12x#24) |
| RTX 5090 as a distinct tuning target | **zero** — every b12x qualification run is 188-SM RTX PRO 6000 or 48-SM GB10. 18 mentions of "5090" are all *"tested on one RTX 5090"* notes and **none** cautions against it |
| widening the K6-only clause | never proposed |
| `FULL_DECODE_ONLY` vs `FULL_AND_PIECEWISE` measured against each other | no artifact in either repo |
| anyone reporting `VLLM_EXL3_PREFILL_FP8` as a silent no-op | none. It is dead **by design**: the author left the flag off after measuring it, and the kernel that would make it live was never committed |

Also worth recording, because it changes how we should read our own image: the pinned rootfs is **not**
stock r34 for `exl3.py`. The pristine wheel source at
`/var/tmp/gg-rootfs/opt/vllm/vllm/model_executor/layers/quantization/exl3.py` is 198,665 bytes and
contains **no** `PREFILL_FP8`; the installed copy in site-packages is 227,678 bytes and does. Our image
is r34 + our own open PRs **#314** (graph decode) and **#316** (reconstruct prefill dispatch). One
consequence is load-bearing for how we describe the artifact: **the measured 1.91× CUDA-graph win exists
only because of PR #314** — on the shipped r34 image, dense EXL3 is `--enforce-eager` only.
