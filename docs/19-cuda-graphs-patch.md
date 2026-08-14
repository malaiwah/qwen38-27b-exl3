# EXL3 dense CUDA-graph decode (`VLLM_EXL3_GRAPH_DECODE`)

Deliverables in `/var/tmp/work/gh/`:

| file | what it is |
| --- | --- |
| `exl3-graphs.py` | full replacement for `vllm/model_executor/layers/quantization/exl3.py` (base: `exl3.py` in this directory, PR #280 head + the PR #312 fix). `diff -u exl3.py exl3-graphs.py` = **+271 / -6** lines, 8 hunks. |
| `envs-graph-decode.patch` | one-hunk unified diff registering `VLLM_EXL3_GRAPH_DECODE` in `vllm/envs.py` (`git apply -p1` from the repo root; generated against the r34 installed `envs.py`, context is the `VLLM_EXL3_*` block). |

Install into the running image:

```bash
cp /var/tmp/work/gh/exl3-graphs.py \
   /var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py
patch -d /var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages -p1 \
   < /var/tmp/work/gh/envs-graph-decode.patch      # a/vllm/envs.py -> vllm/envs.py
rm -f /var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__pycache__/exl3.*.pyc
```

---

## 1. Mechanism

Three parts: a **row plan**, a **priming pass**, and a **gate**. Nothing runs unless the
new env var is set, so the default behaviour of every existing configuration is byte-for-byte
what it is today.

### 1.1 Row plan — `_graph_decode_capture_rows()` (line 821)

The plan is the set of `m` values (rows of the 2-D activation reaching one shard) that a
CUDA graph can ever replay:

* `compilation_config.cudagraph_capture_sizes` — final at weight-load time; `VllmConfig.__post_init__`
  computes it (`config/vllm.py:_set_cudagraph_sizes`) long before the worker loads weights.
* Plus the superset that `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode()` can
  produce, because that pass runs *after* weight loading (from
  `gpu_model_runner._check_and_update_cudagraph_mode`, during KV-cache init). With
  `q = 1 + num_speculative_tokens > 1` it rounds every size up to a multiple of `q` (and to
  `max(q, tp)` when sequence parallelism binds captured sizes) and unconditionally adds
  `q * n` for `n = 1..32`. The plan adds all of those.
* Plus `size // q` for every plan entry — the per-request row count an MTP/draft layer sees.

Everything is clamped to `max_cudagraph_capture_size`. With `--max-num-seqs 8` and no spec
decode the plan is exactly `[1, 2, 4, 8, 16]`.

### 1.2 Priming pass — `_prime_exl3_gemm_rows()` (line 864) + `Exl3LinearMethod._prime_graph_decode_shapes()` (line 2286)

At the end of `Exl3LinearMethod.process_weights_after_loading` — i.e. eagerly, on the real
weights, before any capture stream exists — every shard that will route through `_exl3_gemm`
gets one zero-filled fp16 GEMM per plan row. Shards that satisfy
`_b12x_trellis_k6_supported` are skipped: they take the native B12X path, which is already
prepared and warmed two lines above by `_b12x_trellis_weight` / `_warm_b12x_trellis_device`.

Memoised in the module-global `_EXL3_GEMM_PRIMED_SIGNATURES` (line 107) keyed by
`(device_index, m, k, n, bits, codebook)`, mirroring
`_EXL3_ONLINE_WARMED_SIGNATURES`. One arena per shard geometry
(`torch.zeros((max(pending), k))`, then `narrow(0, 0, m)` — a leading-row view of a
contiguous tensor is contiguous) so priming 5–96 row counts costs one allocation, not one per row.

Why this is sufficient (verified in `exllamav3/exllamav3_ext/quant/exl3_gemm.cu`):

* `gemm_autotune_hash` (line 53) mixes `MIN(roundup_pow2(MAX(size_m,2)), 16)`, `size_k`,
  `size_n`, `K`, `c_fp32`, `device`, `cc`, `max_num_sms`, `cb` — **shape and codebook only, no
  weight pointers**. So the memo may legitimately skip a later shard with identical geometry,
  and the effective bucket set is tiny: `{2, 4, 8, 16}` (every `m >= 9` lands in bucket 16).
  Priming exact capture sizes is therefore a strict superset of bucket coverage — this
  retires the "m-bucketing means a warmup pass cannot reliably cover every bucket" objection
  in the old comment.
* `cb` is `1` for `mcg`, `2` for `mul1`, `0` otherwise — hence the codebook field in the memo
  key. The Qwen3.8-27B K4 checkpoint really does mix them (MLP = `mcg`, `lm_head` = `mul1`),
  so a shape-only key would have under-primed `lm_head`.
* First call per device also does `DevCtx::get_locks()` → `cudaMalloc` + `cudaMemset`
  (`exl3_devctx.cu:59`). `cudaMalloc` during capture is illegal, so this too must happen
  eagerly; priming forces it.

If any priming launch raises, the pass converts it into the enforce-eager `ValueError`
naming the shard, `m`, `K`, `N`, `bits` — startup fails **before** capture instead of
faulting inside it, which is what keeps the relaxation fail-closed even though the gate runs
earlier than the priming.

### 1.3 Gate — `Exl3Config._graph_decode_refusal()` (line 1500) / `_require_enforce_eager()` (line 1546)

`_require_enforce_eager` keeps its old shape: rank-sliced R7 returns immediately, the check
is memoised in `_eager_checked`, no vLLM config means no opinion, `enforce_eager=True`
returns. Only the "not eager" branch changed: it now asks `_graph_decode_refusal`, and
raises the original message with `Graph decode was not permitted because <reason>.`
appended. Refusal reasons, in order:

1. `VLLM_EXL3_GRAPH_DECODE is not 1, so the pre-capture exl3_gemm priming pass is disabled`
2. `compilation_config.cudagraph_mode is unset`
3. `cudagraph_mode=<mode> also captures mixed prefill batches …` — anything whose
   `mixed_mode() != NONE`, i.e. `PIECEWISE`, `FULL`, `FULL_AND_PIECEWISE`
4. `microbatched execution (DBO/ubatching) splits every captured size across ubatches …`
5. `cudagraph_mode=<mode> is decode-only but compilation_config.cudagraph_capture_sizes is empty …`

On success it stores the plan in the new public `Exl3Config.graph_decode_rows` (line 1119),
which is the single switch the priming pass reads. `cudagraph_mode=NONE` is granted with
`graph_decode_rows = None` (no capture happens, so nothing needs priming, and nothing is
primed).

`Exl3Config._require_eager_moe_experts()` (line 1585), called from `get_quant_method` right
before `Exl3MoEMethod` is constructed, re-raises when `graph_decode_rows` is set:
non-rank-sliced routed experts issue one `_exl3_gemm` per expert with a row count only the
router knows (`Exl3MoEMethod.apply` → `.nonzero()` → `_apply_expert`), so no plan can cover
them. This never fires for the K4 target (193 EXL3 matrices, zero `experts` tensors).

---

## 2. Env var semantics

| | |
| --- | --- |
| name | `VLLM_EXL3_GRAPH_DECODE` |
| default | **off** (unset) |
| enabled by | the literal string `"1"` — matching `VLLM_EXL3_PREFILL_TRELLIS`'s `== "1"` test |
| off values | unset, `""`, `0`, `true`, `yes`, `on`, anything but `1` |
| read at | `Exl3Config._require_enforce_eager` (once per config) via `_graph_decode_enabled()` (line 801) |
| effect | permits non-eager execution for **serialized dense** EXL3 checkpoints **and** turns on the pre-capture `exl3_gemm` priming pass. Never sufficient on its own: the cudagraph mode must be decode-only and the capture list non-empty. |
| no effect on | rank-sliced R7 checkpoints (return before the check), online-K6 overlay (untouched), eager runs |
| registered in | `vllm/envs.py` passthrough block (`envs-graph-decode.patch`), same convention as the other `VLLM_EXL3_*` knobs: registration exists so startup does not flag the variable as unknown; the default lives in `exl3.py` |

---

## 3. Why decode-only is the safe boundary

Priming is only sound if the set of reachable `m` values is enumerable **before** capture.

* A **uniform decode** graph is captured and replayed at token counts drawn from
  `cudagraph_capture_sizes` (`num_tokens = num_reqs * (1 + num_spec)`), and every dense linear
  in the model sees exactly that row count. Enumerable → primable.
* A mode that also captures **mixed prefill-decode** batches (`PIECEWISE`, `FULL`,
  `FULL_AND_PIECEWISE`) puts arbitrary scheduler-chosen prefill batches on the capture path.
  Even where the runner pads them to a capture size, that padding contract is not something
  this quantization backend can verify per release, and chunked-prefill token counts are not
  enumerable from config. Keep raising.
* `FULL_DECODE_ONLY` therefore captures exactly the window that carries the measured decode
  win (BF16 graphs 27.47/101.04/208.31 vs eager 25.50/92.72/186.98 tok/s at C1/C4/C8), and
  leaves prefill eager, where autotuning is legal and cheap.

---

## 4. Still unsafe / not covered

1. **Prefill / mixed capture** — refused, not primed (reason 3 above). Note for later: the
   piecewise mixed capture in the BF16 baseline used the *same* size list `[1,2,4,8,16]`, so
   the plan would already cover it; relaxing the gate to `FULL_AND_PIECEWISE` is a plausible
   follow-up but is deliberately **not** in this patch.
2. **Late mode downgrade** — `resolve_cudagraph_mode_and_sizes` can turn a requested
   `FULL_DECODE_ONLY` into `PIECEWISE` or `NONE` after weights are loaded
   (`config/compilation.py:1406-1450`) when the attention backend cannot do decode-only full
   graphs. `NONE` is harmless; a downgrade to `PIECEWISE` would capture mixed batches that
   the gate meant to exclude (still at capture sizes, so still primed — but outside the
   contract this patch verifies). Watch for
   `… is not supported with <backend> backend …; setting cudagraph_mode=PIECEWISE` in the log;
   the r34 BF16 run on this model showed decode FULL was supported, so this should not fire.
3. **Non-rank-sliced EXL3 MoE experts** — refused (`_require_eager_moe_experts`). Row counts
   are router-dependent, and the correctness path also calls `.nonzero()`, which is not
   capturable at all.
4. **DBO / ubatching** — refused, because a captured size is split across microbatches and
   the per-ubatch row counts are not the capture sizes.
5. **MTP / spec decode** — permitted, but the plan is a *superset* computed from
   `num_speculative_tokens`; it is not the post-alignment list, which does not exist yet at
   weight-load time. If a future vLLM adds a new spec-decode capture-size transform, the plan
   can miss a row count. The failure mode is a fault at capture, not silent wrong output.
6. **Rank-sliced R7 path** — untouched, including the pre-existing situation that it permits
   graphs without any `exl3_gemm` priming; `VLLM_EXL3_GRAPH_DECODE` does not prime it either.
7. **Online-K6 / B12X shards** — untouched. Assumption (see §7) that B12X selects kernels
   from shape without timing launches, which is what the already-captured R7 path relies on.
8. **Large eager batches** — `m` above `max_cudagraph_capture_size` runs outside graphs and
   autotunes on first use, as today. That is legal and unaffected.

---

## 5. Test plan

Baseline for comparison is the eager K4 run; the flags below are the BF16-graphs flags
(`--max-model-len 4096 --served-model-name m --max-num-seqs 8`) plus decode-only capture.

### 5.1 Positive run (the one that must work)

```bash
GG_ENV="VLLM_EXL3_GRAPH_DECODE=1" /var/tmp/work/ggrun.sh \
  vllm serve /work/Qwen3.8-27B-K4 \
    --host 127.0.0.1 --port 8022 \
    --max-model-len 4096 --served-model-name m --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  2>&1 | tee /var/tmp/work/k4-graphs.log
```

Note: there is **no** `--cudagraph-mode` CLI flag in this vLLM; the mode is only reachable
through `--compilation-config` / `-cc` JSON. `--enforce-eager` must **not** be passed.
`--cudagraph-capture-sizes` / `--max-cudagraph-capture-size` can be used to shrink or grow
the plan; the priming pass follows whatever list results.

Then run the existing decode benchmark (`/var/tmp/work/bench.py`) at C1/C4/C8 against the
eager K4 numbers.

### 5.2 Negative controls (each must abort at startup, before capture)

```bash
# a) env var missing -> today's behaviour
/var/tmp/work/ggrun.sh vllm serve /work/Qwen3.8-27B-K4 ... -cc '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# b) env var set but default (mixed-capturing) mode
GG_ENV="VLLM_EXL3_GRAPH_DECODE=1" /var/tmp/work/ggrun.sh vllm serve /work/Qwen3.8-27B-K4 ...
# c) unchanged legacy path: eager still works
GG_ENV="VLLM_EXL3_GRAPH_DECODE=1" /var/tmp/work/ggrun.sh vllm serve /work/Qwen3.8-27B-K4 ... --enforce-eager
```

Expected: (a) `… Graph decode was not permitted because VLLM_EXL3_GRAPH_DECODE is not 1, so
the pre-capture exl3_gemm priming pass is disabled.` (b) `… because
cudagraph_mode=FULL_AND_PIECEWISE also captures mixed prefill batches, whose token counts are
not enumerable before capture; select decode-only capture with --compilation-config
'{"cudagraph_mode": "FULL_DECODE_ONLY"}'.` (c) serves exactly as today, no priming lines.

### 5.3 Log lines that prove it

```bash
grep -aE "EXL3 graph decode enabled|EXL3 graph-decode priming|Capturing CUDA graphs|Graph capturing finished|setting cudagraph_mode|requires eager execution" /var/tmp/work/k4-graphs.log
```

Priming ran (logger `vllm.model_executor.layers.quantization.exl3`):

1. Once, during model construction:
   `EXL3 graph decode enabled by VLLM_EXL3_GRAPH_DECODE: cudagraph_mode=FULL_DECODE_ONLY captures decode only; priming exl3_gemm for 5 row counts (m=1..16) during weight loading.`
2. Once per distinct shard geometry, during weight loading — for this checkpoint **exactly
   three** lines (193 EXL3 matrices collapse to three geometries):
   * `EXL3 graph-decode priming: autotuned exl3_gemm for 5 capture row counts (m=1..16) at K=5120, N=17408, bits=4, codebook=1.`  (gate/up, 128 matrices)
   * `… at K=17408, N=5120, bits=4, codebook=1.`  (down_proj, 64 matrices)
   * `… at K=5120, N=248320, bits=6, codebook=2.`  (`lm_head`, `mul1` → not B12X-eligible)
   Counts scale with the capture list, so `--max-num-seqs 64` (plan `[1,2,4,8,16,24,…,128]`)
   raises `5` accordingly. More or fewer than three lines means the geometry/codebook split
   changed — worth a look, not a failure.

Graphs were captured (vLLM's own output):

3. A tqdm bar `Capturing CUDA graphs (decode, FULL): 100%|…| 4/4` — and **no**
   `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)` bar, which is what distinguishes
   this run from the BF16 baseline (`bf16-graphs.log` has both: 5 mixed + 4 decode).
4. `[gpu_model_runner.py:6971] Graph capturing finished in N secs, took X GiB`.
5. Absence of any `CUDAGraphMode.FULL_DECODE_ONLY is not supported …; setting
   cudagraph_mode=…` warning (see §4.2) and of `enforce_eager=True` in the
   `Initializing a V1 LLM engine … with config:` line.

A clean decode benchmark alone is not proof; items 1–4 together are.

---

## 6. Lines touched (`exl3-graphs.py` numbering)

| lines | change |
| --- | --- |
| 37 | import `CUDAGraphMode` alongside `get_current_vllm_config_or_none` (only modified pre-existing line besides the eager check) |
| 102-107 | new `_EXL3_GEMM_PRIMED_SIGNATURES` memo + comment |
| 800-937 | new `_graph_decode_enabled`, `_uniform_decode_query_len`, `_graph_decode_capture_rows`, `_prime_exl3_gemm_rows` (inserted after `_exl3_gemm_fake`, before `_b12x_trellis_weight`) |
| 1116-1119 | `Exl3Config.__init__`: `self.graph_decode_rows` |
| 1500-1543 | new `Exl3Config._graph_decode_refusal` |
| 1546-1583 | `_require_enforce_eager` rewritten (comment + the `enforce_eager` branch; the rank-sliced early return, memo and `vllm_config is None` guard are unchanged) |
| 1585-1601 | new `Exl3Config._require_eager_moe_experts` |
| 1621 | `get_quant_method`: `self._require_eager_moe_experts(prefix)` before `Exl3MoEMethod(...)` |
| 2285 / 2286-2321 | `Exl3LinearMethod.process_weights_after_loading` tail call / new `_prime_graph_decode_shapes` |

Removed lines: the old import line and the 5-line `if not vllm_config.model_config.enforce_eager: raise …`
body plus 3 comment lines. Nothing else in the 4883-line original is altered. No new imports
beyond `CUDAGraphMode` from the already-imported `vllm.config`.

## 7. Verification performed / assumptions

Performed without touching the GPU (the box is running measurements):

* `python3 -m py_compile exl3-graphs.py` and the patched `envs.py` — both clean; no line >88 chars.
* `diff -u exl3.py exl3-graphs.py` → +271/-6, 8 hunks, and the 6 removed lines are only those
  listed above.
* Behavioural harness: extracted the *actual source text* of `_graph_decode_enabled`,
  `_uniform_decode_query_len`, `_graph_decode_capture_rows`, `_prime_exl3_gemm_rows`,
  `_graph_decode_refusal`, `_require_enforce_eager`, `_require_eager_moe_experts` and
  `_prime_graph_decode_shapes` from the final file, bound them to stub config/tensor objects
  and the real `CUDAGraphMode` enum (execed from the installed `config/compilation.py`), then
  checked: eager → no-op; env unset → raises with reason 1; `PIECEWISE`/`FULL`/
  `FULL_AND_PIECEWISE` → raises with reason 3; ubatching → reason 4; empty capture list →
  reason 5; `NONE` → granted with no plan; `FULL_DECODE_ONLY` → granted with 51 rows for the
  stock list, 96 rows for `num_speculative_tokens=3` (verified: contains every `4n` for
  `n<=32`, every base size rounded up to 4, and the per-request counts, all `<=512`);
  priming dedupes across shards and across layers with identical geometry, re-primes for a
  different codebook, skips K6/`mcg` shards, reuses one arena, and converts a kernel error
  into the enforce-eager `ValueError`.
* Kernel-side claims read from `/var/tmp/work/exllamav3/exllamav3/exllamav3_ext/quant/`
  (`exl3_gemm.cu`, `exl3_devctx.cu`): autotune hash fields, the `{2,4,8,16}` m buckets, the
  `cb` codebook selector, the lazy `cudaMalloc` of the lock arena.
* Checkpoint facts from `/var/tmp/work/Qwen3.8-27B-K4/quantization_config.json`: 193 EXL3
  matrices, three geometries, no `experts` tensors, `lm_head` is K6+`mul1`.

Could not verify from source (state clearly):

1. **B12X dense-Trellis capture safety at arbitrary `m`.** The K6 path is left as-is on the
   assumption that it selects kernels from shape without timing launches (the R7 rank-sliced
   path is already captured today, and `_warm_b12x_trellis_device` warms only `m=1`). For the
   K4 target this is moot: every K4 shard has `trellis.shape[2] == 64`, and even the K6
   `lm_head` uses `mul1`, so `_b12x_trellis_k6_supported` is false everywhere and all 193
   matrices are primed through `_exl3_gemm`.
2. **Whether `compute_logits`/`lm_head` executes inside the decode graph.** I primed
   `lm_head` at the plan rows (plus the `size // q` per-request rows) so both answers are
   covered, but I did not confirm the runner's placement.
3. **Wall-clock cost of priming.** No GPU run was allowed. Expected to be small — 5 rows × 3
   geometries = 15 launches for the default `--max-num-seqs 8`, of which only the four
   distinct m-buckets per geometry pay a timing sweep — but this is an estimate, not a
   measurement.
4. **That capture never replays a row count outside `cudagraph_capture_sizes`** for
   `FULL_DECODE_ONLY`. Read from `gpu_model_runner`/`compilation.py` behaviour, not from an
   executed capture; the negative outcome would be a capture-time fault naming the shape, not
   silent corruption.


---

# Measured outcome (this box, 2026-08-14)

Patch installed into the r34 image, served with
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` and
`VLLM_EXL3_GRAPH_DECODE=1`, no `--enforce-eager`.

All five predicted proof points appeared:

1. `EXL3 graph decode enabled by VLLM_EXL3_GRAPH_DECODE: cudagraph_mode=FULL_DECODE_ONLY captures decode only; priming exl3_gemm for 5 row counts (m=1..16) during weight loading.`
2. Exactly **three** priming lines, one per shard geometry, as predicted from the
   checkpoint's 193 EXL3 matrices: `K=5120,N=17408,bits=4,codebook=1`,
   `K=17408,N=5120,bits=4,codebook=1`, `K=5120,N=248320,bits=6,codebook=2`.
3. `Capturing CUDA graphs (decode, FULL): 4/4` and **no** mixed prefill-decode bar.
4. `Graph capturing finished in 2 secs, took 0.06 GiB`.
5. `enforce_eager=False` in the engine config line, no mode-downgrade warning.

## Throughput

| configuration | C1 | C4 | C8 |
|---|---:|---:|---:|
| K4 eager (previous) | 28.77 | 103.47 | 215.84 |
| **K4 + CUDA graphs** | **55.39** | **190.59** | **428.12** |
| gain | **+92.5 %** | **+84.2 %** | **+98.4 %** |
| `unsloth/Qwen3.8-27B-NVFP4` (graphs, Cutlass FP4) | 49.09 | 171.78 | 371.06 |
| ours vs NVFP4 | **+12.8 %** | **+10.9 %** | **+15.4 %** |

The gain is ~9x larger than the +7.7/+9.0/+11.4 % that graphs buy the BF16 model,
because eager EXL3 pays per-call dispatch on 193 quantized matmuls where BF16 pays
cuBLAS once per fused linear. **With graphs this quant is both the smallest and the
fastest option measured**, and it is 3.1x closer to BF16 than NVFP4.

## Correctness under graphs

| check | result |
|---|---|
| text, greedy | `Red, Green, Blue` |
| vision, 96x96 half-red/half-blue PNG | `red, blue` |
| coherence, 2-sentence explanation | correct and fluent |
| **distribution parity vs eager** | **KLD 0.000000, top-1 1.000000** over 32 sentinel contexts |

Graphs change throughput, not numerics. The patch is ready to upstream.
