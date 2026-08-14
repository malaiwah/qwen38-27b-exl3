# Upstream contributions

## local-inference-lab/vllm

| item | status |
|---|---|
| [Issue #311](https://github.com/local-inference-lab/vllm/issues/311) — EXL3 online overlay: MXFP8 fallback raises for shapes divisible by neither 128 nor 32 | filed with repro, root cause, and log evidence |
| [PR #312](https://github.com/local-inference-lab/vllm/pull/312) — keep those shards BF16 and warn | opened against `codex/gg-exl3-r7-k345-20260810` (the branch carrying PR #280's overlay), verified on this box |

The patch is a three-way branch in `Exl3OnlineLinearMethod.create_weights`:
Trellis if 128-aligned, MXFP8 if 32-divisible, otherwise `UnquantizedLinearMethod`
with a warning. `process_weights_after_loading` and `apply` already delegate to
`self.fallback`, so nothing else changes.

Verification performed before opening the PR:

- **Before:** `ValueError: MXFP8 requires input_size_per_partition (4304) to be divisible by 32`, engine core init fails.
- **After:** 27 `visual.blocks.N.mlp.linear_fc2` shards log
  `EXL3 online overlay keeps ... unquantized: K=4304 is neither 128-aligned for Trellis nor divisible by 32 for MXFP8`,
  startup completes, and a multimodal request through that path answers correctly
  (`red, blue` on the half-red/half-blue PNG).

### Reported but not patched

`Mxfp8OnlineLinearMethod` is non-functional in the r34 image: shards that *are*
32-divisible but not 128-aligned (`visual.blocks.N.attn.qkv` K=1152, fused GDN
`linear_attn.in_proj_ba` K=5120 N=96) fail with
`AttributeError: module 'vllm.utils.flashinfer' has no attribute 'mm_mxfp8'`.
So the documented MXFP8 retention path cannot execute in that build. Left out of
PR #312 deliberately (kernel availability, not shape logic); noted in the PR body
with an offer to follow up with probe-and-degrade.

### Next, and required by the project

`Exl3Config._require_enforce_eager()` blocks CUDA graphs for every dense
`tensor_storage` checkpoint. Measured cost on this model: ~10 % decode throughput.
The design for lifting it — prime `exl3_gemm`'s autotune cache over the configured
`cudagraph_capture_sizes` during warmup, mirroring the existing
`Exl3OnlineLinearMethod._warm_decode_shapes` pattern, then allow decode-only
capture behind an opt-in env — is in
[12-iteration-2-plan.md](12-iteration-2-plan.md) as P0.

## Operator findings worth documenting upstream

1. Overlay `ignore` regexes match prefixes **without** a leading `model.`:
   `re:.*visual\..*` matches, `re:.*\.visual\..*` does not. The second form is
   the intuitive one to write and silently fails.
2. Packed modules are expanded before ignore-matching, which is why
   `re:.*in_proj_a$` + `re:.*in_proj_b$` correctly excludes the fused
   `linear_attn.in_proj_ba`.
3. `-cb mcg` in exllamav3 is honoured for decoder tensors but `lm_head` and MTP
   payloads are written with `mul1`, and `config.json` then reports `codebook: mul1`
   for the whole checkpoint. Since the B12X native K6 kernel requires `mcg`, a
   serialized K6 head silently misses the fast path.
