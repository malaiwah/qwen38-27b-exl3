# R30 — Exact stock EXL3 action/encoder harness

## Result

The corrective pass now has one fail-closed complete-action contract and one authoritative
encoder path. [`tools/research/wave5/exl3_action.py`](../../tools/research/wave5/exl3_action.py)
imports `turboderp-org/exllamav3` v1.4.2 at
`5f3c537ca9d89893d771256f5c43c93656553fbb`, checks the Python, CUDA-source and compiled
extension identities, and calls that checkout's actual CUDA extension. It does not contain a
uniform quantizer, a clean-room trellis simulator, or `chol(inv(H)).T`.

The frozen machine contract is
[`tools/research/wave5/exl3_action_schema.json`](../../tools/research/wave5/exl3_action_schema.json).
Every action now binds its curvature/capture tensor and manifest hashes, observation count,
normalization and coordinates; its callback identifier/version/source hashes/parameters/interface;
three distinct data projections, exact selectors and a zero-leakage audit; closed promotion KLD
lineage; and, when selected, the complete production materialization qualification. K, target,
correction, path refinement and runtime route remain inside the action and cannot be appended
after a rate curve was measured.

This is the post-fix implementation and evidence status. It is **not** a claim of Wave 5
foundation approval; approval remains gated on the independent reviewer verdict below.

## Pinned mechanisms, traced to source

The source pin matters. The served checkpoint was produced with this public v1.4.2 pin;
the stale aiboss `791c830` checkout and the image's older 0.0.43 runtime source are not the
publication encoder. R30 used a detached, clean `5f3c537` worktree and built the extension
from that tree.

|Mechanism|Pinned stock source/API|Harness boundary|
|---|---|---|
|MCG/MUL1 constants and H128|`exllamav3/modules/quant/exl3_lib/quantize.py:20-25`|Constants and marker tensors are never reimplemented; action declares `mcg`, `mul1`, or markerless `3inst`.|
|Actual Viterbi entry point|`quantize.py:69-96` calls `ext.quantize_tiles` with K and marker flags|`StockEXL3` imports and calls this module. There is no nearest-level or affine-uniform substitute.|
|CUDA Viterbi state graph|`exllamav3_ext/quant/quantize.cu:17-83` dispatches one of 24 `(K,codebook)` kernels; `quantize_tiles_kernel.cuh:1-304` performs forward costs, backpointers and two-pass closure|A legal-path callback may return only finite candidate target tiles. The harness itself invokes the pinned kernel and creates reconstructed values/indices; callback-supplied indices are impossible.|
|Sign stream and transformed curvature|`quantize.py:849-910` draws seeded input signs, applies sign/H128 to H, calls `block_ldl(H,16)`, and zeros the scalar diagonal; `quantize_exl3:1210-1408` draws output signs|The action fixes the seed. Curvature callbacks run after stock damping/sign/H128 and immediately before stock block-LDL.|
|Block-LDLQ, not inverse-Cholesky GPTQ|`quantize.py:402-476` factors H and constructs the block-unit lower factor; `:478-606` runs the reverse 16-row block recurrence|A recurrence callback may band/truncate the returned stock L, but stock `ldlq` remains the consumer. `b=16` is enforced.|
|FP16 `suh`/`svh`, H128 and stock scales|`quantize.py:1100-1208` applies output scales/signs, H128, input scales/signs and stock global scale search; `:1376-1388` casts `su`/`sv` to FP16|Scale callbacks start from the stock five-tuple and must preserve its shapes, dtypes and finiteness. Decode still reads FP16 `suh`/`svh`.|
|Stock scale search|`quantize.py:932-1052` samples wrapped-diagonal and RMS-extreme tiles and calls actual Viterbi during coarse/fine search|The A0 action uses it unchanged. Candidate recipes must name any changed magnitude/search policy.|
|Packing and alignment|`quantize.py:912-923` allocates `[in/16,out/16,16*K]` int16; `exllamav3_ext/quant/pack.cu:1-93` packs sixteen aligned spans per 16x16 tile|The harness serializes the returned tensors and records raw buffer bytes, standalone safetensors bytes/header overhead, tensor hashes and alignment.|
|Marker semantics|`quantize.py:1390-1400` stores one int32 marker with locked multiplier; `quantize.cu:65-70` selects `3inst`, MCG or MUL1|MCG and MUL1 are a one-factor control with equal raw bytes. Two markers at once are illegal.|
|Source-basis decode|`exllamav3/modules/quant/exl3.py:20-102` constructs `LinearEXL3`; `:147-236` reconstructs through the extension and folds both H128/sign-scale sides; `get_weight_tensor` returns the original basis|Every smoke decode instantiates `LinearEXL3` and calls `get_weight_tensor`; stored Transformers `[out,in]` orientation is restored explicitly.|
|Extension bindings|`exllamav3_ext/bindings.cpp:109-122` exports `quantize_tiles`, `pack_trellis`, `reconstruct`, H128 and EXL3 GEMM|The receipt hashes the built `.so` and the exact binding/quantize/Viterbi/pack source files.|

The enforced source hashes are:

- `quantize.py`: `4cd368dab28e007d649e25b97c65fc73a56ef2a1482ca2b9298a53d4b0876dbf`
- `exl3.py`: `c010bd18aaf5363632db25c0a4f7c4be0938011656f0446f933505a59b8d6cc0`
- `bindings.cpp`: `6e1ebdbd2cedacf7672a9de272bf70cb7ab0282088f6a2f55a4d55cef11dff95`
- `quantize.cu`: `cee125a3e4bf8f12681380f52cf0ab9b0a586c7c12f167be57b073ba5557a73b`
- `quantize_tiles_kernel.cuh`: `85a9ab6295362212f3c6edc990cb6edb57c77a7b5473fe89b5109fdf57c28bfa`
- `pack.cu`: `27606eed6650acc31c6b6484aad1e89195da88823a5bd62ffb3e9911a9b47e60`
- compiled extension `.so`: `79815da8b7d39559c2dea17cffb966fe7d78beba5b67c2f49f7f41832c40b2bf`
- `util/hadamard.py`: `6884841b6137878874ee0b2942ec2f62cb6275a40ffc853146a73b2d92233cbb`

The adapter no longer accepts a source-mismatch override. It observes the loaded worktree's
actual Git HEAD, tree and tracked-clean status plus `exllamav3.version`; the corrected receipt
records commit `5f3c537…`, tree `ffc0a1d…`, `worktree_clean=true` and version `1.4.2`.

## Complete action, curvature and callback contract

`EXL3Action` validates Qwen3.8's 64-layer topology and deployable granularity. Stock K4,
K5 and K6 are legal. K7 is rejected unless `qualifications.k7_artifact_sha256` names a
completed decoder/runtime qualification. A module/tensor/shard or fused topology group must
name every physical tensor key and, for fused outputs, every H128-aligned output split.
Per-tile K and selector sidecars are not deployable actions under the current format.

Every encode now fails before quantization unless the supplied H bytes, FP32 dtype and square
shape equal the action. The action binds and recomputes the capture-manifest SHA256 over H
identity, observation count, normalization, basis, coordinates and tensor/module boundaries;
stale nonempty metadata is rejected. `source_layout` is inside the action identity, and encode
derives orientation exclusively from it. The frozen coordinate convention must match that
layout. Observation count is passed into stock `h_data`.

Callbacks are stage-specific and encode-only:

1. **target** sees the contiguous FP32 encoder matrix before regularization;
2. **scale** starts from stock `(apply_out_scales, weight_r, g_scale, su, sv)`;
3. **curvature** sees stock's damped, sign-flipped, H128-transformed H immediately before
   `block_ldl`;
4. **recurrence** sees the stock block-LDL L immediately before `ldlq`;
5. **legal path** may return only finite candidate target tiles; the harness alone reruns
   pinned `quantize_tiles_multigpu` and creates reconstructed values and indices.

Each action binds callback identifier, version, parameters, exact expected interface,
per-stage source/signature/default hashes, every callback-module file SHA256 and a content
SHA256 over the complete identity. The harness recomputes it from the invoked callables
immediately before encoding. Closures are rejected; module hashes cover helpers/globals and
defaults are separately bound. The A0 no-hook contract is implementation
`f9f1b815…f189`, content `e4efe3df…5521`; strength-zero is implementation
`140b1763…579b`, module hash `88dd9ed6…436a`, content `f1f1e6ef…35e0`, with parameter
`strength=0.0`.

The same reentrant process lock now covers callback verification, target invocation, global
hook installation, actual `quantize_exl3` and cleanup for **every** encode, including A0.
All tensor outputs preserve shape, dtype and device and are finite; the stock scale tuple
also preserves its Boolean/scalar contract. All paths end in stock `pack_trellis` with
unchanged `{suh,svh,trellis,marker}` payload. No callback executes at inference.
Strength zero uses identity callbacks at all five boundaries and is compared by canonical
tensor-buffer and reconstruction digests; standalone safetensors headers intentionally differ
because the action identities differ.

## Exact bytes

For a dense H128-aligned module with `numel = in_features * out_features`, the raw buffers
obey

\[
B = \frac{\text{numel}\,K}{8} + 2(\text{in_features}+\text{out_features}) + B_{marker},
\]

where `B_marker=4` for MCG/MUL1 and zero for markerless 3inst. This is derived from the
actual returned tensors, not extrapolated slice bpw: trellis int16 shape
`[in/16,out/16,16*K]`, FP16 `suh[in]`, FP16 `svh[out]`, and optional scalar int32 marker.
The receipt separately records the real standalone safetensors size and header overhead.
Whole-checkpoint allocation must sum actual co-sharded file/container bytes rather than add a
standalone header per module.

## Route registry

The canonical route IDs, shared with R31, are:

- `codec-exact/all-trellis-stock-exl3`: qualify with
  `VLLM_EXL3_MULTIPRECISION=0`. Decode remains on the stock trellis payload. In the current
  vLLM patch, B12X is eligible only for MCG, K6 by default or K3-K6 with
  `VLLM_EXL3_B12X_ANY_BITS=1`, subject to shape/N guards; MUL1, K7 and other ineligible cases
  fall back to the EXL3 extension. Both B12X and extension rows are codec-exact routes, though
  floating execution agreement still belongs in runtime qualification.
- `production/throughput-fp4-fp6-materialized`: an action selecting this route is invalid
  without a closed, live materialization qualification. It binds file paths and hashes for
  the route, FP4/FP6 converter sources and binaries, source trellis payload, realized tensor
  and payload, and runtime receipt; validation rehashes every observed file. The source
  payload must equal the action's own `hashes.payload_sha256`, and the live container digest
  must equal the qualification. The effective environment binds all materialization and
  residual routing inputs, including FP4/FP6 module paths, layer patterns/ranges, draft-head,
  B12X bit/M/N/lm-head controls, prefill reconstruction and trellis prep. Validation compares
  the complete live environment, including explicit empty/default values, before a production
  action is accepted.

The deployed route implementation is `patches/vllm-exl3-multiprecision.py`, SHA256
`dcede1b494984b3ec29fae5187e8aa692557e4658a1601c7dc0fc337737cbaa8`. Binding this patch
alone is explicitly insufficient for a production-materialized promotion; the converter,
environment and realized-runtime qualification above are mandatory.

## Authoritative corrected RTX 5090 smoke

The post-schema-change run used Torch `2.12.0+cu132`/CUDA 13.2 on the aiboss RTX 5090.
The v1.4.2 extension was compiled from the clean pin for `sm_120a`; its `.so` SHA256 is
`79815da8b7d39559c2dea17cffb966fe7d78beba5b67c2f49f7f41832c40b2bf`. The execution
image was `docker.io/malaiwah/qwen38-27b-exl3-gg:r34-p2-41a5d16` at
`sha256:a9ac6a6da63b3ad5ca0fc2d3659c00727bfd30141e8e362e902042112655016b`.
The exact harness and schema hashes recorded by the run are respectively
`717de7845c3f3812396f089e47c9e8c6a4d59175b12efcc120742ff0ea6cfd87` and
`896f29d70c42de9544bac0719fe0a287c31dadbdf3ddbeafa74a209e40fa1478`.

The representative full tensor was BF16
`model.language_model.layers.3.self_attn.k_proj.weight`, source shape `[1024,5120]`, at
Qwen revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, SHA256
`e51b2206a8fbc35c980a3e4d2173c4a0050cd79944593e664a5b24aa21e91101`. Safetensors
loaded it as `torch.bfloat16` and the encoder performed a numeric FP32 conversion.

The screening H_X is `[5120,5120]` FP32 identity with SHA256
`1f82650ead249f12475b866a402ea4e8cbd4643acdf07431c25b4b6f0e277f3f`; its bound
capture manifest is `784703e26367e46295bd45a552f4f3ab9d6294f32c8c71de4384a8eb67d336bd`,
observation count 1 and normalization `identity/no-sample-normalization`. Its action scope
is exactly the k_proj tensor/module input-feature coordinates and it is only a codec smoke.

Fresh A0 action identity is
`f2c2b62071faf1834a306c5c8bf4b66c174073e94bb624be8df9d59d3cf5e7cc`; encode took
0.9031 s. `LinearEXL3.get_weight_tensor` returned a finite `[1024,5120]` source-basis
reconstruction with SHA256
`ad4a11ab1a1edf100df03b9b1d76a39274b3f9ace7a1f2de5ec8e1502c768394`. Exact buffers:

|buffer|dtype/shape|bytes|SHA256|
|---|---|---:|---|
|trellis|int16 `[320,64,80]`|3,276,800|`82a99fc0e3a4cbb386afb6f721b933735dfe58711eaf6c8f146554388528435e`|
|suh|FP16 `[5120]`|10,240|`fa4fb897c8f33cd01f11e1620e5c402175dbb842484419c0b48ffe9faa47d397`|
|svh|FP16 `[1024]`|2,048|`89e423e8f011ee8a492413c73b4e5276afe7a87f654bf2f0aac78c199f5e3211`|
|mcg|int32 scalar|4|`ade4fb124dda0f3537386cdd4a3cdcea3a223d386e506a4be89394bb33ee13fe`|

The raw total is **3,289,092 B**, exactly equal to the dense byte law. Regenerated A0
safetensors is 3,289,556 B with SHA256
`8578d8820e83b636ac3072aa7e09ae4676e6cdaae645994dd2edde751b4b3d1f`.

The regenerated strength-zero action identity is
`0b148f25c63b67bc268130eab25f31f899be4aa5666fccfb6cca72e11596bec8`.
Its canonical payload SHA256 is
`c934173131b84fdb599f40f75cc33718cfd97d03985f3037dfce3298071499ef`,
identical to fresh A0. Every tensor buffer and the source-basis reconstruction are
byte-identical, and both retain the same hot schema and codec-exact decoder route.
Its independently regenerated standalone file is 3,289,572 B, SHA256
`9dbd3342b00675d4a4af7fa36f542928f3c081c6173be0358b2dcd381c006bd3`.

Actual-stock panel controls on the real tensor's first 128x128 block all decoded finitely and
obeyed the byte law:

|K|raw bytes|trellis words/tile|stock proxy error (not KLD)|
|---:|---:|---:|---:|
|4|8,708|64|0.00456512|
|5|10,756|80|0.00119325|
|6|12,804|96|0.000318906|

At qualified K5, the MCG/MUL1 one-factor arms each used 10,756 raw bytes and finite stock
decode. Their trellis hashes differ (`3d706f…66f` vs `bdc58c…c0e`), as required for a real
codebook intervention rather than a marker-only relabel. All other declared inputs and
search budgets were held fixed.

The service was stopped for the GPU run, then started without a `PROFILE` override.
The default throughput service health endpoint succeeded after the recorded 50.3-second wait.
Full machine evidence is in
[`receipts/wave5/stock-control.json`](../../receipts/wave5/stock-control.json).

## Metric and split discipline

One `split_manifest_sha256` binds the combined source-disjoint split manifest. Each of
`calibration`, `validation`, and `untouched_test` then binds a pair of
`selection_sha256` plus a structured selector in the frozen language
`{field:"split",op:"eq",value:<split>}`. Selection hashes and selector values must both be
pairwise distinct. A bound disjointness artifact must report all three pairwise overlap
counts, source-document overlap and domain leakage as zero with `verified=true`; overlapping
predicates, duplicated selection hashes, or leakage fail action validation.

For real Phase B actions, R29's combined manifest content SHA256 is
`db63446e1ce174e340ae53c039632c7291d74a8bb9f263852673f4a649e3115f`
(the file SHA is transport integrity only). Its frozen selections are:

- calibration `490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259`;
- validation `4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de`;
- untouched test `4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca`;
- leakage audit `7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9`.

The codec smoke uses one synthetic combined manifest with three selector-disjoint synthetic
selections and explicitly labels them non-fidelity evidence; it does not use zero hashes.

The promotion KLD slot is closed. It requires exact suite-manifest, suite-token and shared
BF16-head hashes; reference/candidate model and capture hashes; report and candidate-payload
hashes; direction `KL(BF16 reference || candidate)`; `full_vocabulary=true`; a validation or
untouched-test identity matching this action; every required v5 metric; and fail-closed
complete/verified lineage. Local MSE, OC-HWE, Fisher-HWE, stock proxy error and block-output
error live only in `local_metrics`; any local key containing `kld` is rejected and cannot
satisfy `promoted_kld`.

## Evidence scope and approval gate

The corrected smoke uses a real full Qwen3.8 BF16 tensor but an identity screening H_X. It
proves the pinned actual-EXL3 encode/decode path, curvature and callback identity checks,
fresh action/container metadata, byte law, strength-zero identity and codebook control. It is
not a calibration-quality or KLD result. Phase B must use the R29 projections/audit above and
must separately qualify the materialized production route before a production promotion.

Targeted self-tests cover the declared curvature/callback/split/KLD and materialization fields,
including stale capture metadata, source layout, missing selectors, source-payload mismatch,
live artifact hashes and routing environment. Both regenerated A0 and strength-zero actions
validate against the frozen JSON Schema.

**Approval status: CHANGES_REQUIRED.** The second and final independent openai-reviewer round
verified the corrected stock receipt and identities above, but did not approve the Wave 5
foundation. Four remaining blockers are recorded rather than hidden:

1. production qualification conflates the canonical tensor-stream payload digest with the
   standalone safetensors file digest, so the actual emitted file cannot satisfy its current
   `source_payload_path` check;
2. residual-route identity still omits the FP8DG prefill/self-test gates that can change the
   serving path;
3. callback interface checks snapshot tensor metadata after invocation, so an in-place
   dtype/shape mutation can change both the returned tensor and the aliased comparison object;
4. callback module identity includes loader-dependent `fn.__module__`, so the CLI-generated
   strength-zero receipt does not replay under a normal imported module name.

The direct A0/identity actual-EXL3 codec result remains coherent: action IDs
`264b8e11…d8cf1` and `3ccadf0d…fe9d1`, shared payload `c9341731…9ef`, shared reconstruction
`ad4a11ab…8394`, and exact raw size 3,289,092 B. These facts do not override the gate:
**Wave 5 foundation approval is not claimed and must not open from this receipt.**
