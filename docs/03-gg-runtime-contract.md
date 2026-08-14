# Gilded Gnosis EXL3 loader: what a dense checkpoint must satisfy

Source of truth: `local-inference-lab/blackwell-llm-docker@76589eb` launchers,
`local-inference-lab/vllm` PR #280 @ `8e7be4d` (`exl3.py`), `b12x@195e26c`.
PR #280 is **not merged** into `dev/gilded-gnosis`; it exists in the r34 image as
an archived integration patch. Everything below therefore applies to the
published image, not to a source build of the branch.

## Blockers for a Qwen3.8 EXL3 checkpoint

1. **No `MODEL_FAMILY` accepts a Qwen model.** `serve-gilded-gnosis.sh` matches
   `glm52|glm5.2|glm`, `glm52-hybrid|nf3`, `glm52-exl3|exl3`, `ds4|ds4-flash|dspark`
   and `exit 2`s otherwise; `glm52-exl3` funnels into the GLM-5.2 server script
   with MLA/DCP/MTP and `nvfp4_ds_mla` KV. The launcher scripts are unusable.
   The engine itself is fine: `Qwen3_5ForConditionalGeneration` is registered in
   the fork's model registry. **Consequence: invoke `vllm serve` directly and
   skip the launchers.**
2. **Eager execution is mandatory.** `Exl3Config._require_enforce_eager()` raises
   unless `enforce_eager=True`, exempting only checkpoints with
   `rank_sliced_metadata` (the GLM-5.2 rank-sliced MoE format). A dense
   `tensor_storage` checkpoint must be served with `--enforce-eager`, i.e. no
   CUDA graphs. This is a throughput cost to state on the model card.
3. **Auto-detection will not fire.** `override_quantization_method` returns
   `exl3` only for `r7_routed_experts` or `hybrid_tr3_tail`. A dense checkpoint
   must be served with an explicit `--quantization exl3`.
4. **No qualified TP1 profile exists.** The code handles `tp_size == 1`
   (`_shard_tensors_for_tensor_parallel` early-returns) and the online cache key
   includes `tp_world_size`/`tp_rank`, but every published EXL3 receipt is TP4 and
   `glm52-exl3` hard-defaults `TP=4 DCP=4 MTP=3`. TP1 is unqualified, not
   unimplemented.

## Required checkpoint metadata

A root **`quantization_config.json`** sidecar with a non-empty `tensor_storage`
map — `Exl3Config.get_config_filenames()` returns exactly that filename, and an
empty map is a hard `ValueError`. Per entry:

```json
{"tensor_storage": {"model.language_model.layers.0.mlp.down_proj": {
  "quant_format": "exl3",
  "bits_per_weight": 4,
  "stored_tensors": {"....trellis": {...}, "....suh": {...}, "....svh": {...}, "....mcg": {...}}}}}
```

Validation, per tensor at load: `trellis` rank-3 int16 with
`1 <= shape[2]//16 <= 8`; `suh`/`svh` rank-1 **float16** with
`numel == trellis.shape[0]*16` and `shape[1]*16`; K and N both 128-aligned;
`mcg` and `mul1` must not both be present. exllamav3's
`util/add_quant_config.py` emits exactly this shape, and correctly writes
`stored_tensors`-only entries (no `quant_format`) for BF16 linears — which is how
a mixed checkpoint self-describes.

Prefix resolution tolerates VLM wrappers: `model.language_model.layers.N.mlp.down_proj`
resolves after stripping a leading/interior `model.`/`language_model.` segment.

## `ONLINE_QUANT=exl3-b6` eligibility, exactly

The overlay claims a module iff **all** of:

1. it is a `LinearBase` (so `ParallelLMHead` is structurally excluded — `lm_head`
   can never be online-K6),
2. its prefix is **absent** from `tensor_storage`,
3. it is not matched by the `ignore` list in `--quantization-config`,
4. the spec resolves to `weight: mxfp8` with no activation override,
5. `VLLM_EXL3_ONLINE_TRELLIS_BITS=6` then promotes MXFP8 to K6.

Shape gate: `input_size_per_partition % 128 == 0 and output % 128 == 0`, else the
module **silently becomes MXFP8**, not BF16 — an `info_once` log is the only
signal.

There is no notion of "attention" or "dense projection". On this VLM the overlay
would otherwise sweep in the vision tower and any leftover BF16 linear, so the
`ignore` list must be written from scratch:

```json
{"linear": {"weight": "mxfp8"},
 "ignore": ["re:.*\\.visual\\..*", "re:.*\\.in_proj_b$", "re:.*\\.in_proj_a$", "lm_head"]}
```

`in_proj_b`/`in_proj_a` are the GatedDeltaNet projections that exllamav3 builds
with `qmap = None`; they are never quantized and would otherwise be claimed by
the overlay. The vision tower must be excluded because its 4304-wide MLP is not
128-aligned and would degrade to MXFP8.

Gate+up merging is structural, not a checkpoint property: encoding runs in
`process_weights_after_loading` on the constructed module, so a
`MergedColumnParallelLinear` is always encoded as one merged matrix. The online
encoder is also required to take the RTN path — it raises if calibrated LDLQ was
used, since the Hessian handed to it is a meta-device zero tensor.

## Mixed bitrates and SM120

Per-tensor K is read per shard as `trellis.shape[2]//16`, any value 1-8, so
mixing K4 and K6 across dense linears is legal. The single uniformity rule:
a fused module's shards must all be EXL3 or all BF16
(`"Packed EXL3 projection ... mixes EXL3 and BF16 source shards"`), which is
satisfied here because `gate_proj`+`up_proj` are both K4 and `q/k/v_proj` are all
BF16.

Execution split: only exactly-K6-with-`mcg` shards reach the B12X native dense
kernel (`trellis.shape[2] == 96`); **all K4 shards go through
`exllamav3_ext.exl3_gemm`**, which has no SM gate. The 109,568-byte shared-memory
limit that bites on SM120 belongs to the three-tier K3/K4/K5 *routed-expert*
cooperative kernel, not to dense K4. The K6 small-M decode kernel is compiled
`sm_120`-only, i.e. SM120 is its only supported target.
