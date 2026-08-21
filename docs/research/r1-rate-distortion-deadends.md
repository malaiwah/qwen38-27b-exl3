# R1-RateDistortion: Dead Ends

## 1. Column-BAQ is consistently harmful

**Approach:** Apply BAQ's closed-form formula per column (as in the original
paper), then aggregate column K values to tile K values by rounding the mean.

**Result:** Column-BAQ never beats uniform K in any of 36 configurations.
Mean improvement: -8.7% (worse than uniform). On L55_down_first at K=4, it
is 103.6% worse than uniform.

**Why it fails:** The aggregation step (averaging 16 column K values to one
tile K) destroys the allocation signal. Columns within a tile can have very
different sensitivities, but the tile quantizer uses one range for all 256
elements. Averaging column K values produces a tile K that is close to uniform
but with noise from the aggregation, making it worse than simply using uniform K.

**Lesson:** When the quantizer operates at tile level, the allocation must
also operate at tile level. Per-column allocation aggregated to tiles is
strictly worse than uniform.

## 2. Iterative BAQ does not improve over single-pass

**Approach:** After the initial tile-BAQ allocation, quantize, measure actual
per-tile distortion, update c_t (blending old and new with α=0.5), and
reallocate. Three rounds.

**Result:** Mean improvement +10.4% vs +10.5% for single-pass tile-BAQ. No
benefit from iteration.

**Why it fails:** The BAQ formula's sensitivity estimate (from Hessian
diagonal) is already a good predictor of actual distortion. The measured
distortion after quantization does not provide additional information that
improves the allocation. The exponential moving average of c_t converges to
approximately the same allocation as the initial estimate.

**Lesson:** The high-resolution approximation Δ²/12 is good enough for
allocation purposes, even at K=3-4 where the approximation is theoretically
questionable. The actual distortion measurement is only useful for the
DP-optimal approach, which uses it directly rather than as a refinement signal.

## 3. Weight magnitude augmentation is mixed

**Approach:** Augment c_t with |w_{ij}|² to allocate more bits to tiles with
both high sensitivity AND large weights.

**Result:** Mean improvement +8.8% vs +10.5% for plain tile-BAQ. Worse in
aggregate, though some individual cases benefit (L55_down_mid K=5: +34.2%
vs +23.2% for plain tile-BAQ).

**Why it's mixed:** The weight magnitude term can over-allocate bits to tiles
with large but insensitive weights. The BAQ formula already captures weight
range through (range_t)², so adding |w_{ij}|² double-counts the magnitude
information. The two-sided Hessian already captures output sensitivity through
H_G[i,i], making the weight magnitude redundant for tiles with large outputs.

**Lesson:** Don't augment the BAQ sensitivity with weight magnitude. The formula
already captures the relevant information through range and Hessian sensitivity.

## 4. Two-sided Hessian provides minimal benefit over one-sided

**Approach:** Use both input Hessian (H_X) and output Hessian (H_G) in the
tile sensitivity computation.

**Result:** One-sided mean +10.6%, two-sided mean +10.5%. Nearly identical.

**Why it doesn't help:** With our output Hessian proxy H_G ≈ Y^T Y / P where
Y = WX, the output sensitivity H_G[i,i] is proportional to the squared norm
of output row i. For well-mixed weight matrices, this is approximately uniform
across output channels, adding little discrimination between tiles. The input
Hessian H_X captures most of the sensitivity variation.

**Lesson:** The input Hessian is the dominant factor for tile-level allocation.
The output Hessian proxy is too uniform to add discrimination. A true gradient
Hessian (requiring backward passes) might provide more variation, but is not
available in our CPU-only setting.

## 5. BAQ closed-form is a good approximation but DP is better

**Approach:** Compare the BAQ closed-form allocation (continuous K → integer
projection) with the exact DP solution.

**Result:** DP-optimal achieves +14.2% mean vs +10.5% for tile-BAQ. DP wins
61% of cases. The gap is 3.7 percentage points.

**Why the gap exists:** The BAQ formula assumes the high-resolution approximation
D ≈ Δ²/12 and that distortion is separable across tiles. The DP measures actual
distortion, which captures:
- Non-uniform error distributions within tiles (the quantization grid doesn't
  perfectly match the weight distribution)
- Inter-tile interactions through the full Hessian (not just diagonal)
- The exact integer K constraint (BAQ's continuous solution is rounded)

**Lesson:** When per-tile distortion measurement is feasible (which it is for
128×128 matrices), DP-optimal is strictly better. For production-scale matrices
(17408×5120), the DP cost may be prohibitive, and tile-BAQ is a good
approximation.
