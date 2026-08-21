# R29 — Real data, activation, and Fisher census

**Status:** Phase-A contract candidate, 2026-08-21. This work establishes data and curvature inputs; it does not optimize a quantization method.

## Frozen repository contracts

| Contract | Locator | File SHA256 | Canonical content SHA256 |
|---|---|---|---|
| Source-weight census | `receipts/wave5/data-manifest.json` | `68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37` | `51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2` |
| Three-way document split | `receipts/wave5/split-manifest.json` | `a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc` | `151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e` |
| Capture implementation | `tools/research/wave5/data_capture.py` | Re-hash the checked-out file; every generated artifact records the executing script hash | — |

The JSON contracts are compact single-line canonical JSON. Large tensors remain under `/tmp/qwen38-wave5-r29` or the immutable model/cache paths and are not committed.

## Topology and source weights

The official source config is SHA256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`. Both config fields and live vLLM module symbols prove exactly 64 language-model layers numbered 0–63: 48 `linear_attention` (GDN) and 16 `full_attention` blocks. Hidden size is 5120, MLP width 17408, full attention is 24 Q / 4 KV heads with head dimension 256 and rotary dimension 64. GDN is 16 key heads and 48 value heads, both dimension 128, with convolution width 4.

The selected depths are 0, 7, 14, 21, 28, 35, 42, 49, and 55. Layers 7, 35, and 55 are full-attention; the other six are GDN. The corrected census contains 87 language-model-only BF16 tensor records. MTP and vision tensors are deliberately excluded. It covers MLP gate/up/down; GDN QKV, Z, output, convolution, A and B parameters; and full-attention Q/K/V/O. Every record includes the source shard name and durable resolved path, tensor name, dtype, shape, safetensors-relative and absolute byte offsets, payload length, and raw-payload SHA256.

Each 2-D tensor has eight preregistered screening blocks: first/middle/last diagonal, two corner off-diagonals, and three seeded random row/column blocks. Each also carries a frozen 20-block promotion list. Screening blocks include raw rectangle hashes; promotion coordinates and the seed are frozen before method work.

## BF16 materialization gate

BF16 is decoded as little-endian `uint16`, widened to `uint32`, shifted left by 16, and viewed as IEEE float32. It is never interpreted as FP16. The known-tensor self-test uses `model.language_model.layers.0.linear_attn.A_log`:

- 48 values compared bit-for-bit with `torch.bfloat16 -> float32`;
- decoded-prefix SHA256 `402e0e023407e5c3d239aec23f8f1f7abfe84a922d8adfd913573dcb8ebc262f`;
- BF16-as-FP16 explicitly rejected; maximum wrong-decode absolute error 3.21484375.

The standalone sentinel also exercises normal values, signed zero, a subnormal, infinities, and NaN payloads.

## Source-disjoint documents

The final split is one combined immutable manifest with exact selectors `calibration`, `validation`, and `untouched_test`; no secondary manifests exist.

| Split | Complete documents/conversation shards | Pretokenized v5 contexts | Code | Dialogue | Multilingual | Prose |
|---|---:|---:|---:|---:|---:|---:|
| Calibration | 6 | 0 | 1 | 0 | 1 | 4 |
| Validation | 331 | 512 | 84 | 1 | 83 | 163 |
| Untouched test | 513 | 1,881 | 102 | 1 | 159 | 251 |

Canonical split projections are: calibration `490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259`, validation `4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de`, and untouched test `4eaaa72d93790195468c168ee03753fe2a0efa0d04334a5d5335fd083de88bca`. Their exact predicates are `{field: split, op: eq, value: <split>}`. Leakage-audit SHA256 is `7fb0bf2f932d93af78c2ca7f40a0cb41f577a2b7801ae80a4ff44bfcc07cc6d9`.

Calibration is exactly the six pinned EXL3 standard calibration files. Every local file is re-hashed before freeze, and every published v5 shard manifest is checked to contain the identical six-source hash set plus observed `contexts_with_any_hit=0` and `total_hits=0`; those ten observed records are bound into the leakage audit. Validation is v5 shard0 source clusters plus a disjoint UltraChat train-gen shard. Untouched test uses only contexts from v5 shards1–9 whose complete source cluster does not occur in shard0, plus UltraChat test-gen. Document hashes and source-cluster sets are pairwise disjoint. Test Arrow/token files are streamed only for byte identity, never deserialized or semantically inspected by the builder. `select_split_documents` mechanically denies untouched-test locators unless the one-shot evaluation gate passes `allow_untouched_test=True`.

The primary suite is `malaiwah/qwen38-27b-fidelity-suite-v5` at exact revision `7797fcce3ffed62b99871348887f4626dc9b2b3b`. Its authoritative whole-suite token digest `510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88` is embedded, as are all ten per-shard manifest and token digests. Authoritative token files are reused; no evaluation text is retokenized.

## Real activation captures

Durable capture root: `/tmp/qwen38-wave5-r29` on `aiboss`.

| Flow/artifact | Locator | Manifest file SHA256 | Internal content SHA256 |
|---|---|---|---|
| Correct BF16 reference module boundaries | `activations-bf16-reference/capture-manifest.json` | `7a3f59e5c1abc0123f7fa6ed8c35246a8d516bfdf54642bcf0d6fc43ed499eab` | `88fd48fa0116f1bca96924546dcc50f4ff27eaf1626132486767492cc1f9c6a6` |
| Shipped hydrated quant + throughput FP4 materialization | `activations-running-quant/capture-manifest.json` | `7ecd882964f37f30c1e32a807b1d1c8c0fc240d990f6bfed33cf02f111410e23` | `a48922c824e2508898f69aab5f3b4604dd94c0db1d6373abc11720799bde6f3b` |
| BF16 covariance statistics | `stats-bf16-reference/stats-manifest.json` | `7f7ecc6bcf72fde6f5241c520e4b1ed709d62a788c157ecbe18b2009a37351df` | `1aaf3bf417756aa4eb86395d36db880d2065ea0f51aa2ef5a7a82558a7dd0998` |
| Quant-flow covariance statistics | `stats-running-quant/stats-manifest.json` | `4a778dbb27871c944d74886140b1956c6089a86529ebb6df762c986481c718d0` | `e39801a7e2513988b73a2e612c5472260e11bd01183e70b44ef211a566ae9eb5` |
| Boundary cross-check | `activation-boundary-cross-check.json` | `a11b92f2d376ac0ed1869e54034bd69264ae9c252ff8e9739bf768b34bd274fe` | `9a9313ec45dbc34c4ef90f6ddf68d81b24029fdd4c4ae1d6eaae31c13c204a51` |

Both flows used the identical real calibration token file, SHA256 `f699bbe7260892cce2ebe7aaeee14e7c8f7a14bd66b1e03cb0bb14756bb5724c`, and 63 presented tokens. There are 36 matching live module records across all nine selected depths. Every value is finite. Module sets and input/output shapes agree exactly between flows. Captures include decoded gate/up fused outputs, actual down inputs and teacher outputs, GDN QKVZ and output-projection boundaries, and full-attention fused QKV/output boundaries. Shapes include `[63,5120] -> [63,34816]` for fused gate/up, `[63,17408] -> [63,5120]` for down, `[63,5120] -> [63,16384]` for GDN QKVZ, and `[63,5120] -> [63,14336]` for full-attention fused QKV.

The live GDN kernel fuses conv/state/a/b operations internally, so those tensors do not appear as separately hookable `nn.Module` outputs in the runtime. The capture retains the exact fused QKVZ input/output and GDN output input/output, which are the legal surrounding module boundaries. Phase-B code must not invent synthetic conv/state activations; any experiment requiring the internal recurrent state must add an explicit kernel trace rather than infer it from these files.

The quant-flow capture used the shipped hydrated checkpoint and production throughput FP4 layer selectors. `VLLM_EXL3_SKIP_TRELLIS_PREP=1` was used only in the isolated eager hook engine to avoid an unrelated duplicate B12X JIT link; this does not change the materialized FP4 tensors whose boundaries were captured. It is therefore activation data, not a runtime qualification result.

## Covariance and Fisher

For every captured module and both flows, the statistics bundle records sample count, real input second-moment diagonal `diag(X^T X/N)`, output covariance diagonal, and four exact dense 16×16 input-covariance blocks. Raw BF16 activation shards remain available, so alternative shrinkage can be rebuilt without model recapture.

Selected exact dense BF16 `H_X = X^T X/N` matrices are at `/tmp/qwen38-wave5-r29/dense-hx-bf16` (content SHA256 `9f9d60127e61a1912385d6fdbfb9bb9e61e2929a0d326e9add3212de8932c69d`):

- L0 down: 17408×17408, SHA256 `f98fdb5910c028d61686432a62d0f87d87d7c01e17d90d46480cc9d8c7653cf6`;
- L0 gate/up input: 5120×5120, SHA256 `545ded9f6cb7dd758d22492542d7d9690cacc29942a82651bafd5c88b7fb4239`;
- L55 down: 17408×17408, SHA256 `99c10bc31c808747fe951c3ace9a79effa95931ed03e3a44d19b58e7dfb1d84a`;
- L55 gate/up input: 5120×5120, SHA256 `9d6a959e6ebbd1931058027e6d10171dff7a14fccf2f9b02115e270c85302640`.

All use 63 real token samples, are finite, and round-trip with bit-exact symmetry.

The selected-module empirical Fisher is `/tmp/qwen38-wave5-r29/fisher-selected-bf16/fisher-manifest.json` (file SHA256 `4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a`, content SHA256 `28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237`). It freezes every BF16 model parameter and uses backward hooks on one real eight-position sequence. The statistic is the exact squared weight gradient of the model's mean shifted causal-token NLL for that sequence, `(d mean_NLL_sequence/dW)^2`; seven scored tokens and all causal cross-position terms are included. This is a one-sequence Fisher-compatible score, not a per-token empirical-Fisher average. Full finite diagonals and dense block-16 outer products cover L0 gate/down, L55 gate/down, L0 GDN QKV/output, and L55 full-attention Q/output. Every block records `output_row=0`, `input_start`, and `input_end_exclusive` alongside artifact hashes. The measured mean NLL is 3.345536470413208.

An independent end-to-end round-trip smoke is `/tmp/qwen38-wave5-r29/fisher-manifest.json` (content SHA256 `aa52840f18198d016d7db55f87dbbbc05883d63f5c691cc137b94b7d2791c88c`). It evaluates real next-token NLL against all 248,320 vocabulary entries through the shared BF16 head. Eight samples and 11 preregistered output rows produce an 11×5120 Fisher-diagonal block. The artifact SHA256 is `a3aa63d658f610d620f292c46dfff2f80472bba4e4cbca6b55ec289539c1d64b`; save/load is bit-exact and finite. Shared head SHA256 is `25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff`.

## KLD terminology boundary

Nothing in this R29 work is called KLD. Wave5 KLD is exclusively `KL(BF16 teacher || candidate)` over all 248,320 vocabulary entries, final post-norm rows 0–2046, replayed through the one shared BF16 head with fp32 vocabulary chunks and fp64 accumulation. Full-vocabulary EAR is `1-TV`. The independence unit is source cluster and the required interval is a 10,000-resample cluster bootstrap with seed 1. Shard percentiles are never averaged.

The shared-head result is body-only and does not measure candidate-head, decode-graph, generation, long-context, speculative, multimodal, or downstream behavior. The served-logit systematic near `6.54e-4`, cross-engine systematic, scored-window shifts, and unlike-suite differences must not be subtracted as additive floors or compared numerically. The retained shard0 BF16 reference must not be recaptured.

## Smoke and operating receipt

Targeted smoke evidence:

- BF16 shift decoder and FP16 rejection passed;
- official config and live runtime both exposed exactly layers 0–63;
- BF16 and quant activation module sets, token hash, and all 36 boundary shapes matched;
- every activation/covariance value checked finite;
- four selected dense `H_X` matrices round-tripped;
- one full-vocabulary sampled-NLL backward-hook Fisher block round-tripped bit-exactly.

The durable R29 root occupies 4.7 GB after dense `H_X` and full selected Fisher materialization; the initial capture bundle was 214 MB. The filesystem retained 79 GB free. Successful quant capture took about 118 s wall time, BF16 CPU-offloaded activation capture about 197 s, and the all-CPU frozen-model Fisher pass about 1,029 s. No checkpoint, shipped profile default, systemd unit, power setting, existing KLD report, or BF16 reference was modified. After every exclusive window the service was restarted on default `throughput`; the final `http://localhost:8000/health` check returned healthy after 51.2 s.
