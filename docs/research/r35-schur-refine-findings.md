# R35 — Schur-conditioned legal-path refinement

**Disposition:** local actual-EXL3 mechanism observed; **not promoted**. The lane has no validation full-vocabulary row and no completed reconstructed-upstream-target factorial, so R37 must exclude it.

## Question and frozen scope

R35 asked whether a valid stock BlockLDLQ payload can be improved offline without changing its serialized schema, byte count, decoder, hot path, or fixed-stride tile independence. The tested arm retains stock K, MCG codebook, signs, FP16 `suh`/`svh`, H128 transforms, global-scale search, damping, and `block_ldl(b=16)`. It changes only the legal trellis paths selected during encoding.

The execution used the pinned EXL3 v1.4.2 source at commit `5f3c537ca9d89893d771256f5c43c93656553fbb`, `quantize.py` SHA256 `4cd368dab28e007d649e25b97c65fc73a56ef2a1482ca2b9298a53d4b0876dbf`, and compiled extension SHA256 `79815da8b7d39559c2dea17cffb966fe7d78beba5b67c2f49f7f41832c40b2bf`. The experiment ran under the historical R30 execution harness/schema pins `717de784…fd87` / `896f29d7…1478`; the receipt also names the later additive contract pins `b77bae5c…4973` / `275644ed…7af8`. The stock encoder, state graph, packer, and `LinearEXL3` decoder are unchanged between those contract revisions.

All fitting used the R29 calibration capture. The untouched test was not opened. Local values below are dense-H objectives, **not KLD**.

## Exact conditional target

Let encoder-basis source and reconstruction be $W,Q\in\mathbb{R}^{d_{in}\times d_{out}}$, $E=Q-W$, and let the same dense transformed/damped curvature used by stock be $H$. The objective is

$$
J(Q)=\operatorname{tr}(E^T H E).
$$

At an original-order 16-row block $B$, partition earlier fixed blocks as $P$ and the not-yet-updated suffix as $S$. Eliminating $E_S$ continuously gives

$$
G = H_{P\cup B,P\cup B}
    - H_{P\cup B,S}H_{SS}^{-1}H_{S,P\cup B},
$$

and the exact conditional target is

$$
T_B=W_B-G_{BB}^{-1}G_{BP}E_P.
$$

The implementation does not form an inverse. It reverses the order of 16-row blocks, calls the pinned stock `block_ldl(b=16)` on the reversed $H$, and maps the unit-lower factor back to the forward sweep. In reversed coordinates the already-fixed original prefix is the factor suffix, so stock's block recurrence produces the same Schur target.

For every 16×16 output tile in $T_B$, R35 calls the actual pinned `quantize_tiles_multigpu` Viterbi entry point. It cannot supply indices. The returned legal path is accepted only when its exact change in the same full objective is strictly negative:

$$
\Delta J=2\langle\Delta,R_B\rangle+
          \operatorname{tr}(\Delta^T H_{BB}\Delta)<0,
\qquad R=H(Q-W).
$$

The whole objective is recomputed after each sweep as a monotonicity check. A one- and two-sweep stopping trace is retained.

## Matched stock control and a useful failed implementation

Candidate and control each spend one actual Viterbi call per input block per sweep. The corrected control reruns the pinned stock `ldlq` itself for every matched sweep. Its final unpacked legal paths and canonical payload digest must equal fresh stock exactly.

An initial full-tensor control reconstructed the stock algebra directly from $L$ and the suffix error. It changed paths on a full tensor because it failed to preserve stock's `buf_size_k=128` product-cache accumulation order. That attempt is invalid. It was replaced rather than rationalized away. The corrected K4/K5/K6 full runs all have matched call counts and stock-identical control payloads.

## Broad actual-stock screen

The screen covers layers 0 and 55, gate/up/down, each tensor's eight frozen 128×128 R29 blocks, and independently rebuilt K4/K5/K6 actions: 144 rows total. Every row used two candidate sweeps and an equal 16-call matched stock budget. Payload schema and raw bytes matched stock in all 144 rows; the matched control was stock-payload identical in all rows. There were 125 strict local wins, 125 changed payloads, and 397 accepted tiles. The median internal dense-H change was **−0.828%**; the best block was **−21.388%**.

|Tensor cell|K|Internal wins|Median internal $\Delta J/J$|Median decoded source-H change|
|---|---:|---:|---:|---:|
|L0 down|4|8/8|−3.220%|−3.273%|
|L0 down|5|7/8|−2.262%|−3.599%|
|L0 down|6|7/8|−2.649%|−3.960%|
|L0 gate|4|6/8|−0.784%|−0.694%|
|L0 gate|5|5/8|−0.499%|−0.515%|
|L0 gate|6|7/8|−0.369%|−0.603%|
|L0 up|4|6/8|−0.268%|−0.480%|
|L0 up|5|7/8|−0.288%|−0.367%|
|L0 up|6|6/8|−0.517%|−0.634%|
|L55 down|4|8/8|−2.509%|−3.034%|
|L55 down|5|7/8|−5.588%|−6.574%|
|L55 down|6|8/8|−3.675%|−4.005%|
|L55 gate|4|6/8|−0.750%|−0.757%|
|L55 gate|5|8/8|−0.508%|−1.240%|
|L55 gate|6|8/8|−1.206%|−1.983%|
|L55 up|4|7/8|−0.123%|−0.871%|
|L55 up|5|7/8|−0.743%|−1.115%|
|L55 up|6|7/8|−0.805%|−0.631%|

“Decoded source-H” reconstructs the serialized payload through stock `LinearEXL3`, including FP16 scale serialization, and scores it against the original calibration $H_X$. It remains a local calibration proxy.

## Full-tensor rebuild at every K

The promoted local mechanism was rebuilt independently at K4, K5, and K6 for the complete L0 gate tensor, stored source shape `[17408,5120]`. This is not a crop and no rate is extrapolated.

|K|Exact raw bytes|Sweep accepted tiles|Internal $\Delta J/J$|Decoded source-H change|Candidate/control Viterbi calls|
|---:|---:|---:|---:|---:|---:|
|4|44,609,540|151 + 5|−0.0179%|−0.0808%|640 / 640|
|5|55,750,660|269 + 13|−0.0598%|−0.2711%|640 / 640|
|6|66,891,780|399 + 12|−0.1215%|−0.4426%|640 / 640|

For all three K values:

- refined and stock buffers have the same names, dtypes, shapes, and raw byte count;
- `suh`, `svh`, MCG marker, packing, alignment, and decoder route are stock;
- matched-control payload SHA256 equals fresh stock;
- the refined payload SHA256 differs, proving a real legal-path change;
- `LinearEXL3.get_weight_tensor` returns a finite reconstruction;
- there is no sidecar, selector, cross-tile continuity, startup operation, or inference callback.

The second sweep accepted only 5, 13, and 12 additional tiles at K4/K5/K6. Most measurable local gain was exhausted by sweep one. Encode cost increased from 3.73/3.25/7.01 seconds for fresh stock to 16.37/15.89/33.58 seconds for the two-sweep refinement; decode cost and hot operations do not change.

## Factorial and validation disposition

The source-target arm was rebuilt independently for every measured K. R34 did not yet have a generated reconstructed-upstream target artifact during R35's exclusive GPU window, so the preregistered source-vs-reconstructed factorial could not be executed. No substitute or synthetic target was introduced.

More importantly, R35 has no validation block-output or full-vocabulary model-output row. It therefore has no KLD, EAR, p99, CVaR1%, or top-1 result. The local gain also shrank sharply from the 128×128 screen to the full tensor. This is exactly the setting in which surrogate over-optimization is plausible.

The scientifically correct disposition is:

1. retain the actual-stock local mechanism as measured evidence;
2. do not call it a fidelity improvement;
3. send R37 an explicit **no eligible action row** result;
4. close the lane for Wave 5 selection unless a separately registered future run supplies both the missing R34 target factorial and direct validation full-vocabulary evidence.

This is a no-promotion result, not proof that the method can never help.

## Traceability and operating evidence

Canonical receipt: `receipts/wave5/r35-schur-refine.json`. It contains the frozen data/split/Fisher pins, exact execution and current R30 contract pins, grouped screen rows, complete full-tensor K rows, legal-call counts, payload hashes and byte manifests, the invalid-control failure, and the no-promotion decision.

The RTX5090 window began with the throughput service active/healthy and 79 GB free. R35 stopped it before CUDA work. After the targeted runs, the service was restarted without a profile override; `qwen38-27b.service` was active and `/health` succeeded after 50.3 seconds. Disk remained at 79 GB free.
