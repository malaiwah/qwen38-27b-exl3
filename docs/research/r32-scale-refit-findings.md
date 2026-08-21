# R32 — Zero-byte EXL3 scale/path co-refit

**Stage:** Phase B calibration/validation research. The untouched test was not opened.

## Contract and method

R32 changes only the values already stored in stock EXL3's FP16 `suh` and `svh`
buffers. K, MCG codebook, random sign streams, H128 transforms, stock global-scale
search, damping, `block_ldl(b=16)`, actual Viterbi, fixed-stride trellis packing,
and `LinearEXL3` source-basis decode stay fixed. Candidate callbacks return the
R30 stock five-tuple; the harness alone invokes the pinned Viterbi and writes
`{suh, svh, trellis, mcg}`. There is no inference callback, selector map, sidecar,
new hot operation, or incremental byte.

The runner binds the following identities for every applicable row:

- R29 data manifest file/content: `68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37` / `51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2`;
- split manifest file/content: `a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc` / `151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e`;
- current R30 harness/schema/qualified extension: `d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d` / `275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8` / `e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e`;
- R31 gate/contract/prereg: `f4fc059c03331905dca6ad7b0ad4ba0e6af515897e2fc90dfd82f1ce0e8e8482` / `e8e1d47694038bbec4aa6f4a4554c4b53e549d2082d87e07353e5d8d16a66783` / `75a81665c75761767a7c71d58f4d59c446a13d3d7b164c5a8b9da9070388a784`.

The broad weight-only screen did not use Fisher and records it as unobserved.
The Fisher-conditioned staged target separately verifies the actual R29 manifest
file/content `4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a` /
`28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237`,
then binds its `fisher-05.npy` record and artifact hash. The real dense `H_X`
target likewise binds `h-x-03.npy` through the dense-H manifest record.

The callback first inverts stock regularization with stock's own H128 routines and
returned scales, then reapplies the identical transform against scales rounded to
FP16 *inside* the encoder loop. This prevents a scale edit from silently changing
the unquantized source target. BiIP target shapes use

\[
S_X=(\operatorname{diag}H_X/\operatorname{diag}(W^TW))^{1/4},\qquad
S_G=(\operatorname{diag}H_G/\operatorname{diag}(WW^T))^{1/4},
\]

with decoder magnitudes proportional to `1/S_X` and `1/S_G`; geometric means are
matched to fresh stock so the candidate does not steal a global-scale degree of
freedom. A path-frozen positive alternating least-squares fit retains stock signs,
uses log-scale regularization, rounds to FP16 after every step, and accepts only a
strict source-basis improvement. Its accepted shape is then passed through a new
stock Viterbi encode; fitted paths are never injected.

## Broad MPS shortlist

The screen covered 45 real Qwen3.8 tensors at all nine frozen depths and every
role present in the common census (MLP gate/up/down and GDN qkv/z/out), with all
eight preregistered blocks per tensor: **360 block cells** at K5. Every rectangle
was reconstructed as BF16 `uint16` payload and matched its R29 coordinate-bound
SHA256 before use. The screen used a plainly labeled fixed-grid source-basis MSE
surrogate; it is not EXL3 and not KLD.

Best-arm counts were BiIP magnitude 254/360, path-fit then scale-frozen rerun
78/360, finite five-level family 23/360, and finite `{2/3,1}` family 5/360. A
candidate beat the surrogate A0 in every block. BiIP was the depth/role macro
winner in 38/45 tensors; scale-frozen rerun won the other 7. Macro BiIP ratios to
surrogate A0 included L0/L55 gate `0.8642/0.8888`, L0/L55 down `0.8960/0.9235`,
L0 GDN-qkv `0.6323`, and L0 GDN-out `0.7128`. These values only selected the
actual-stock full-tensor arms; none supports an EXL3 claim.

The conclusion-bearing raw block rows, per-tensor macros, all 45 role/depth cells,
negative arms, data identities, and exact screen configuration are in
`receipts/wave5/r32-scale-refit.json`.

## Actual-stock full-tensor result

Main granted an exclusive RTX5090 window after the capture-launcher campaign
closed. R32 encoded the complete L55 gate tensor (`17408×5120`, K5/MCG, seed
300030) through the current d4df R30 harness and directly mounted qualified e2e
extension. Fresh A0 and R30 strength-zero produced the **same canonical payload
SHA256** `a7f1ca6ec59082a76089a62be9f59de8ed078b4d044fdae0217fc843cdbb13e0`
and the same source-basis reconstruction SHA256
`ff56c0d07a7929062f07e8e8d30fcd8d336a08775c3ee7880dc0f3dce7b4786e`.
Every arm returned exactly **55,750,660 bytes**, the stock
`{suh,svh,trellis,mcg}` schema, zero incremental hot/sidecar bytes, and the
codec-exact stock decoder route.

The table gives actual full-tensor local ratios to search-matched A0-S; lower is
better. These are source-basis MSE and calibration `H_X`-weighted OC-HWE, not KLD.

| Arm | MSE / A0-S | OC-HWE / A0-S | Result |
|---|---:|---:|---|
| A0 | 1.000055 | 1.015329 | fresh stock |
| A0-S | 1.000000 | 1.000000 | matched FP16-in-loop control |
| BiIP, identity output | 1.085412 | 1.813346 | negative |
| BiIP, selected Fisher | 5.500191 | 21.597850 | decisive negative |
| finite `{.8,.9,1,1.1,1.2}` | 1.039172 | **0.766876** | OC-HWE gain, MSE tradeoff |
| finite `{2/3,1}` | 1.036892 | **0.663799** | best OC-HWE, MSE tradeoff |
| path-fit → Viterbi round 1 | **0.999677** | **0.998217** | strictly accepted |
| path-fit → Viterbi round 2 | **0.999456** | **0.995635** | strictly accepted |
| path-fit → Viterbi round 3 | **0.999143** | **0.992238** | strictly accepted |

The three rounds refit the prior accepted stock trellis with positive ALS,
log-scale regularization and FP16 rounding, then let the harness rerun stock
Viterbi. Round 3 improved MSE by `0.0857%` and OC-HWE by `0.7762%` versus A0-S.
The finite two-level arm reduced OC-HWE by `33.62%` but worsened MSE by `3.69%`;
it is a metric-tradeoff shortlist, not a winner. Direct BiIP magnitudes were
falsified locally on this tensor despite the MPS proxy direction.

The approved R30 one-tensor capture launcher closed with zero rows after its
infrastructure retry cap. Main authorized local validation only and explicitly
forbade legacy captures from standing in for Wave-5 KLD. Therefore R32 has **no
eligible R37 row** and makes no KLD/p99 or promotion claim. The local scale/path
mechanism is positive but the lane remains validation-blocked rather than
promoted or scientifically closed by the preregistered KLD falsifier.

After the campaign R32 restarted the unchanged default throughput service and
`/health` succeeded. The untouched test was never opened.

## Scale serialization disposition

The current stock decoder reads FP16 `suh`/`svh` tensors. It has no int8 or int4
scale payload semantics, no dequantization parameters for such scales, and no
format ID selecting them. Int8/int4 scales are therefore a **new-format action**
with decoder work and metadata, not a zero-byte R32 action; they are deferred.
