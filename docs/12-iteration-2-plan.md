# Iteration 2 plan

Constraints, restated as acceptance criteria:

1. Mean KLD **below 0.034030** on the frozen window (iteration 1's number).
2. Loss reduced *where it counts* — logit-space error first, since that is what
   KLD and top-1 agreement measure.
3. Stay in **EXL3**.
4. Resident weights **at or under 21.92 GB** (the `nvidia/Qwen3.6-27B-NVFP4`
   equivalent). Iteration 1 sits at 19.21 GB, so the budget is **+2.71 GB**.
5. Coherent, **performant**, vision intact.
6. **CUDA graphs are required.** Iteration 1 runs `--enforce-eager`; this is now a
   blocking defect, not a caveat.

## What iteration 1's measurements actually tell us

| observation | inference |
|---|---|
| `k4-online-k6` 0.034030 vs `k4-bf16-attn` 0.036775 | Attention representation is **not** the bottleneck. K6 attention is not merely free, it measured *better* than BF16 attention on the same checkpoint — the K6 out-scales partly cancel the K4 MLP's systematic shrinkage (`g_sc ~0.895`). Do not spend budget on attention. |
| `down_proj` proxy error 2.5e-3 vs `gate_proj` 1.1e-3, `in_proj_qkv` 1.0e-3 | The MLP stack, and `down_proj` in particular, owns the residual error. |
| online K6 proxy error 3.2e-4 | 6-bit Trellis is ~8x more accurate per tensor than 4-bit here. |
| `lm_head` is K6, and anecdotally the most quantization-sensitive tensor | Untested variable that sits directly on the logits the metric reads. |

## Ranked investments

### P0 — CUDA graphs (upstream code, not the checkpoint)

`Exl3Config._require_enforce_eager()` raises for any checkpoint without
`rank_sliced_metadata`. The stated reason: `exl3_gemm` autotunes with timing
launches per `(m-bucket, k, n, K)` and "m-bucketing means a warmup pass cannot
reliably cover every bucket".

That argument does not hold for **decode-only** capture, because vLLM captures a
known, finite set of batch sizes (`cudagraph_capture_sizes`), and the same file
already contains the pattern that solves it: `Exl3OnlineLinearMethod._warm_decode_shapes`
warms rows 1..6 per shard and memoises the signature in
`_EXL3_ONLINE_WARMED_SIGNATURES` precisely so that decode launches are
deterministic.

Proposed patch:

1. `Exl3LinearMethod.warm_autotune(layer, m_values)` — run one dummy `_exl3_gemm`
   per configured capture size per shard during `process_weights_after_loading`
   (or the profile run), memoised by `(device, m, k, n, K)`.
2. Relax `_require_enforce_eager` to allow non-eager when the warmup covered the
   capture set and the mode is decode-only (`FULL_DECODE_ONLY`), gated by an
   opt-in env (`VLLM_EXL3_GRAPH_DECODE=1`) so the default stays fail-closed.
3. Keep prefill piecewise/eager, where token counts are not enumerable.

Evidence to collect first: decode tok/s for this checkpoint (eager) against
`unsloth/Qwen3.8-27B-NVFP4` (graphs) at concurrency 1 and 4 on the same GPU, so
the PR quantifies what the guard costs.

### P1 — `lm_head` to BF16 (+1.589 GB, no re-conversion needed)

`lm_head` cannot be online-K6: `ParallelLMHead` is structurally excluded from the
overlay. It is currently serialized K6. Splicing the BF16 head over the existing
checkpoint costs one script run, no GPU, and lands at **20.80 GB — still 1.1 GB
under budget**. If the anecdote about head sensitivity holds, this is the cheapest
KLD win available and it is directly in the logit path the metric measures.

Ship as branch `v-head-bf16`; keep `v-head-k6` (iteration 1) for the paired
comparison. This is also a clean ablation: the only change is the head.

### P2 — error-driven MLP allocation (`down_proj` first)

Two routes to a mixed K4/K5/K6 MLP, both inside EXL3:

- **Upstream tooling, no patch:** convert a second time at `-b 5`, then
  `util/measure.py -i k4 k5 -r bf16` -> `util/optimize.py -m measurement.json -b <target>`.
  That allocator is genuinely error-driven (`dkld/dbits` greedy) and produces the
  optimal mix for a bit budget. Cost: ~35 min conversion + ~10 min measure +
  recompile.
- **Three-line converter patch:** teach `create_q_strategy` a glob->bpw override
  map, then set `down_proj` to K5/K6 directly.

Budget arithmetic: all 64 `down_proj` K4->K5 costs +0.713 GB, K4->K6 +1.426 GB.
With P1 taken (20.80 GB), `down_proj` at K5 fits at **21.51 GB** and K6 does not
(22.23 GB). So the combination to test is **BF16 head + K5 down_proj**, or drop
the head to K6 and take K6 `down_proj`.

### P3 — calibration, which costs nothing in VRAM

Iteration 1 used exllamav3's default calibration: 250 rows x 2048 tokens of a
generic mix (wiki 50, c4 20, code 20, random 20, multilingual 10, technical 10,
tiny 5). This model is a **thinking, multimodal, tool-calling** model; none of
that distribution is in the calibration set, and no chat-template-formatted text
is either. Replacing the corpus with template-formatted reasoning/tool traces is a
pure-upside experiment: identical bitrate, identical footprint, different Hessians.

Risk to control: over-fitting calibration to one domain can hurt general KLD, so
this must be measured on the same window, not assumed.

### P4 — reclaim VRAM to buy bits: FP8 KV cache

Both NVFP4 references ship FP8 KV schemes; we run KV at `auto` (BF16). At
`--kv-cache-dtype fp8` the KV cache halves, which does not change *weight*
footprint but does change the number that matters operationally (total resident).
If the comparison target is total VRAM rather than weights, this frees room for a
whole extra bit on `down_proj`. Must be applied to teacher and candidates alike
when measuring.

### P5 — `mcg` for the head, to unlock the native K6 kernel

`-cb mcg` was honoured for the decoder tensors but `lm_head`/MTP were written with
`mul1`, and `_b12x_trellis_k6_supported` requires `mcg`, so the serialized K6 head
falls back to `exl3_gemm`. If P1 (BF16 head) wins, this becomes moot; if the head
stays K6, forcing `mcg` moves it onto the fast path. Perf only, no accuracy change.

## Explicitly not worth budget

- **More bits on attention.** Measured: K6 attention already beats BF16 attention
  on this checkpoint.
- **Quantizing the vision tower.** 0.92 GB total, not 128-aligned, and both
  vendors leave it alone.
- **Quantizing the MTP head.** 0.85 GB, and a quantized draft head costs
  acceptance rate on the one component whose job is agreeing with the target.
- **W4A4-style activation quantization.** Not available in EXL3, and Unsloth's
  W4A4 checkpoint is the worst KLD we measured.

## Order of execution

1. Measure eager-vs-graph throughput (evidence for P0), then implement and PR the
   autotune-warmup patch.
2. Splice `v-head-bf16` (P1), measure KLD, publish the branch with its receipt.
3. Run the `-b 5` conversion and `measure`/`optimize` (P2), targeting 21.5 GB.
4. If P2 lands under budget with a KLD win, fold P3 (calibration) as the next
   free variable.
