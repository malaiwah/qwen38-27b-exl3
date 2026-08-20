# exllamav3 conversion: capabilities and three gaps

Read at `turboderp-org/exllamav3@5f3c537` (version 1.4.2).

## The family is first-class

`exllamav3/architecture/qwen3_5.py` registers `Qwen3_5ForConditionalGeneration`
→ `Qwen3_5VLConfig` (text sub-config `text_config`, `Qwen3VLVisionModel`,
`Qwen3_5MTPModel`), and dispatches per layer on
`config.layer_types[idx] == "linear_attention"` to `GatedDeltaNet(key_qkv="in_proj_qkv",
key_z="in_proj_z", key_o="out_proj", ...)` else to `Attention(...)`. Tensor keys
are `model.language_model.layers.N.{linear_attn,self_attn,mlp}.*` plus a bare
`lm_head`.

## Flags that matter

| flag | default | effect here |
|---|---|---|
| `-b/--bits` | required | average bpw; integer floor is the base allocation |
| `-hb/--head_bits` | 6 | `lm_head`; accepts 1-8 or exactly 16 |
| `-mb/--mtp_bits` | 4 | MTP module; **quantizes MTP unless raised** |
| `-vb/--vision_bits` | 16 | 16 builds no vision model → tensors pass through as true BF16 |
| `-cb/--codebook` | `mul1` | use `mcg` to match the runtime's native dense path |
| `-cr`/`-cc` | 250 / 2048 | calibration rows / row length (512k tokens) |
| `-d/--devices` | `0` | single GPU is the default |
| `-hq` | off | **no-op on this model**: `select_hq_bits = 2 if use_moe else 0` |

Calibration corpus is bundled and mixed by weight: wiki 50, c4 20, code 20,
random tokens 20, multilingual 10, technical 10, tiny 5.

VRAM is bounded by design — one decoder block plus one float32 Hessian of
`in_features^2`. For this model the largest Hessian is `17408^2 * 4 = 1.21 GB`;
peak stays well under 10 GB, so one 96 GB GPU is ample. `[INFERENCE]` from those
shapes, not measured.

## Gap 1: no per-module bitrate override

`create_q_strategy` gives every module with `qbits_key == "bits"` the integer
floor of `--bits`, then spends the remaining budget by a **static
priority/position order**, not by measured error. `head_bits`, `mtp_bits` and
`vision_bits` are auxiliary targets outside that budget. `job_state["q_strategy"]`
is initialized and never read or written, so a persisted allocation cannot be
hand-edited either.

The error-driven mechanism exists only as a *post-hoc* pipeline over already
converted models: `util/measure.py` (>=2 quants + reference → per-group
`dkld`/`dbits`) then `util/optimize.py` (greedy `dkld/dbits` under a bit budget,
then a `VariantSafetensorsCollection` splice and recompile). It requires the
inputs to share an identical tensor key set, so it can mix K4 with K5/K6 but
cannot mix in BF16.

## Gap 2: stock level-3 measurement drops GDN groups

At this pinned source, `GatedDeltaNet.optimizer_targets()` does not put its
input and output projection lists at the same nesting depth as `Attention`.
For split Qwen GDN layers, stock `measure_model.py -l 3` sees no GDN groups:
`in_proj_qkv` and `in_proj_z` are one level too shallow and `out_proj` is absent.
The fused layout exposes its input but still omits its output. Those marginals
are not a valid whole-body allocator control.

Apply
[`patches/exllamav3-1.4.2-gdn-optimizer-targets.patch`](../patches/exllamav3-1.4.2-gdn-optimizer-targets.patch)
only to `turboderp-org/exllamav3@5f3c537`, then run
`python tools/verify_exllamav3_gdn_optimizer_targets.py --source <exllamav3-checkout>`
from this repository. The verifier rejects the exact unpatched source and
requires the level-3 oracle: 320 groups covering all 400 Qwen body linears
exactly once (144 GDN, 48 full-attention, 128 MLP), retaining k+v and gate+up
groups. Without that pass, `-l 3` is not a valid control.

## Gap 3: unquantized decoder linears come out FP16, not BF16

`Linear.load_fp16` uses `float2half=True` with `allow_bf16` defaulted False, and
`LinearFP16.get_tensors` re-emits whatever it holds. BF16 survives only on the
compile-time passthrough paths (`compile.py` extras, `Module.get_compile_tensors`),
which is exactly why `-vb 16` yields a genuinely BF16 vision tower. Also, any
bpw of 16 in the strategy trips the `all(b <= 8 ...)` guards and forces the whole
job onto the serial path.

`util/recompile.py -or spec.yaml` cannot paper over this: `VariantSafetensorsCollection.get_tensors`
enumerates keys only from `self.main`, so a BF16 donor's `.weight` is never
emitted over a quantized base.

## Consequence: convert then splice

**Where these commands run.** Steps 1 and 3 are the **external exllamav3 toolchain** at
`turboderp-org/exllamav3@5f3c537` (1.4.2) and run in that checkout's own environment — `convert.py`
and `util/*` are exllamav3's, not this repository's. Steps 2 and 5 are ours and run in this
repository's environment. Mixing the two environments is the single most common way to get a
checkpoint that converts but will not load.

1. `python convert.py -i <bf16> -o <k4> -w <wd> -b 4 -hb 6 -mb 6 -vb 16 -cb mcg -d 0`
2. [`tools/splice_bf16_attn.py`](../tools/splice_bf16_attn.py): drop
   `.trellis/.suh/.svh/.mcg` for every
   `layers.N.linear_attn.{in_proj_qkv,in_proj_z,out_proj,in_proj_b,in_proj_a}` and
   `layers.N.self_attn.{q,k,v,o}_proj`, and copy the source BF16 `.weight` in its
   original `(out, in)` orientation. `Linear`'s default `pad_to = 128` divides
   every relevant dimension, so no padding occurs and the raw tensor splices in
   cleanly.
3. `python util/add_safetensors_index.py -m <out>` then
   `python util/add_quant_config.py -m <out>`.
4. Inspect, but do not hand-repair, `config.json`: upstream utilities may leave
   `quantization_config.bits = 4.00` and may force
   `tied_word_embeddings = true` on this untied model. Step 5 owns the
   deterministic repair and records it in the build receipt.
5. [`tools/finalize_checkpoint.py`](../tools/finalize_checkpoint.py): the publication gate, and the
   step that must not be skipped. It validates logical tensor names and shapes against the recipe,
   repairs `quantization_config`, and emits `quantization_manifest.json`, `SHA256SUMS`,
   `DOCS-SHA256SUMS` and `build-receipt.json`. All four are written atomically
   (`write_atomic`, temp file then `os.replace`) precisely because a truncated `SHA256SUMS` still
   parses and would silently verify fewer files than it claims to.

The loader side needs no patching: `Linear.load()` probes `load_exl3` then
`load_fp16` per module, and `storage_size()` already has the non-EXL3 branch, so
a per-linear mix is a supported state.

## If patching is preferred over splicing

Three functions: `create_q_strategy` (accept a glob→bpw override map),
`Linear.load_fp16` (thread `allow_bf16`), and `quantize_linears_parallel` (add the
missing `strategy[key] == 16` skip branch).
