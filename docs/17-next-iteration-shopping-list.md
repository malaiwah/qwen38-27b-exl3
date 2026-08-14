# Shopping list for iteration 2

Ordered by expected value per unit of effort, with the evidence that justifies each
item and the acceptance test that closes it. Everything here is derived from
measurements in this repo, not from intuition.

## Hard constraints carried forward

| constraint | current state |
|---|---|
| resident weights <= 21.92 GB (NVFP4-equivalent) | 19.21 GB, **2.71 GB unspent** |
| minimise KLD | 0.026231 body-only on the v2 suite; v3 held-out numbers in [15](15-results-fidelity-v2.md) / [18](18-results-fidelity-v3.md) |
| stay in EXL3 | yes |
| coherent, vision-capable | verified: text and image answers correct |
| **CUDA graphs required** | **blocked** by `_require_enforce_eager`; patch in flight |

## A. Runtime — CUDA graphs (non-negotiable)

**A1. Land the autotune-priming patch.** `Exl3Config._require_enforce_eager()` refuses
non-eager execution for any dense `tensor_storage` checkpoint because `exl3_gemm`
autotunes with timing launches. Decode capture sizes are a known finite set, and the
file already contains the pattern that solves it
(`Exl3OnlineLinearMethod._warm_decode_shapes` + `_EXL3_ONLINE_WARMED_SIGNATURES`).
Prime every serialized shard over `cudagraph_capture_sizes`, memoise, then permit
decode-only capture behind an opt-in env var.
*Evidence:* graphs are worth **+7.7 % / +9.0 % / +11.4 %** at C1/C4/C8 (BF16
graphs-vs-eager pair on this model).
*Acceptance:* server starts without `--enforce-eager`, logs FULL decode capture,
answers text + image, and decode throughput rises by ~10 % with identical KLD.

**A2. Chase the real perf gap, which is the kernel, not graphs.** NVFP4 serves at
49.09/171.78/371.06 tok/s versus our 28.77/103.47/215.84. Only exactly-K6 shards with
the `mcg` codebook reach the B12X native Trellis kernel; everything else goes through
`exl3_gemm`. Two sub-items:
- **A2a.** Re-emit `lm_head` (and any future K6 tensors) with `mcg` rather than `mul1`.
  exllamav3 honours `-cb mcg` for decoder tensors but wrote `mul1` for the head and MTP.
- **A2b.** Measure `exl3_gemm` versus `_b12x_trellis_linear` on our exact shapes to
  quantify what an all-K6 MLP would buy in throughput, so the accuracy/throughput
  tradeoff of K6 promotions is priced, not guessed.

**A3. FP8 KV cache.** Both NVFP4 vendors ship FP8 KV schemes; we run `auto` (BF16).
Halving KV frees VRAM for weights at fixed total budget and matches the references.
*Acceptance:* KLD re-measured with the same KV dtype across teacher and candidates.

## B. Recipe — spend the 2.71 GB where the error actually is

**B1. `down_proj` promotion.** Largest per-tensor proxy error in every layer
(2.5e-3 versus 1.1e-3 for `gate_proj`, 1.0e-3 for `in_proj_qkv`). K4 to K5 costs
+0.713 GB, K4 to K6 costs +1.426 GB; both fit.

**B2. Error-driven allocation instead of my hand-picking.** Upstream already ships
the machinery: convert a second time at `-b 5`, then `util/measure.py -i k4 k5 -r bf16`
followed by `util/optimize.py -m measurement.json -b <target>`, which greedily
allocates by measured `dkld/dbits`. Cost: one 35-minute conversion plus ~10 minutes of
measurement. This replaces guesswork with the same metric we publish.

**B3. Calibration corpus.** Iteration 1 used exllamav3's default 250x2048 generic mix.
This is a thinking, multimodal, tool-calling model and none of that distribution is in
the calibration set, nor is any chat-template-formatted text. Free in VRAM, so the only
cost is a re-conversion.
*Caveat to control:* over-fitting calibration to one domain can hurt general KLD, so
measure on the held-out v3 suite, never on the calibration text.

**B4. Do not spend on attention or the head.** Measured: online-K6 attention beat BF16
attention (0.034030 vs 0.036775 on v1), and the K6 head costs 6.78e-05, which is
0.26 % of total divergence ([16](16-head-attribution.md)). A K4 head is a different
question and is worth one cheap measurement if a variant ever needs the space.

## C. Measurement — precision and honesty

**C1. Held-out corpus (done, v3).** The v2 suite was drawn from exllamav3's own
calibration corpora — the text our quant was tuned on, while NVFP4 and FP8 were
calibrated elsewhere. That biased the comparison toward us. v3 uses Gutenberg, arXiv,
Wikipedia (7 languages) and CPython, with a 160-character shingle scan against every
calibration corpus reporting **0 contaminated contexts**.

**C2. Runtime-repeat noise floor (done, v3).** The reference protocol measures
0.0032 mean KLD between two captures of the *same* runtime, and warns that a candidate
difference of comparable size needs repeated capture. We now capture the K4 runtime
three times over 32 sentinel contexts and report the pairwise floor, so every reported
difference can be compared against it.

**C3. Replay qualification (done, v3).** Live full-vocabulary logits versus replayed
logits on the same contexts, the check that separates a replay defect from candidate
error. Reference value on Kimi-K3: 1.2e-6.

**C4. Analysis / qualification partition discipline (done, v3; enforce next).** 136
analysis contexts, 45 qualification. Recipe choices must be frozen on analysis before
qualification numbers are opened. Iteration 2 must not tune against qualification.

**C5. Still missing.**
- More balanced strata: encyclopedic 17 and multilingual 8 contexts are thin; fetch
  more long articles per language.
- No dialogue/instruction or structured/tool-call stratum, which is exactly where a
  thinking model is used.
- Nothing beyond 2048 tokens: no long-context fidelity, though the model claims 262k.
- No multimodal fidelity: the vision tower is BF16 and untested by KLD.
- No downstream task evaluation. KLD ranks candidates within one artifact; it does not
  substitute for coding, reasoning or tool-use benchmarks, and the reference protocol
  says so explicitly.

## D. Publication

**D1.** Publish each variant as a branch of `malaiwah/Qwen3.8-27B-K4` with its own
receipt: `quantization_config.json`, the fidelity report JSON, the serve command, and
the measured resident footprint. A variant without a receipt does not reach `main`.

**D2.** Keep the chart regenerated from receipts (`tools/make_charts.py`) so every
published number and every plotted point come from the same JSON.

**D3.** Upstream: [#311](https://github.com/local-inference-lab/vllm/issues/311) and
[#312](https://github.com/local-inference-lab/vllm/pull/312) are filed; the graph patch
(A1) and the MXFP8-kernel-absence finding
(`vllm.utils.flashinfer has no attribute mm_mxfp8`) are the next two.

## E. Sequencing

1. A1 graph patch — test, measure, PR (unblocks the hard constraint).
2. B2 second conversion at `-b 5` + measure/optimize, targeting 21.5 GB.
3. Re-measure the winner on the v3 analysis partition; compare against the noise floor.
4. A2a `mcg` head + A3 FP8 KV, re-measure throughput.
5. B3 calibration experiment, measured on held-out data only.
6. Freeze, open the qualification partition once, publish with receipts.
