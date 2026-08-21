# R9-GroupOrbit: Dead Ends (v2, post-Cholesky fix)

## 1. Equilibration alone is catastrophic (CONFIRMED, not a dead end for composition)

Scaling without correction increases error by 400,000–765,000%. But this is NOT a dead end — it's a step that the alternating optimizer correctly rejects. Future work should test equilibration+rotation+GPTAQ composition.

## 2. Partition alone hurts initially (but helps after correction)

Scale-based partition hurts on its own, but the L0_down convergence shows it becomes useful AFTER rotation+correction changes the landscape. The alternating optimizer discovers this automatically.

## 3. Allocation DP doesn't help after rotation (CONFIRMED)

Rotation homogenizes tile sensitivity, making the DP unable to exploit heterogeneity. R1's allocation works best on unrotated weights. The alternating optimizer correctly rejects allocation after rotation.

## 4. RETRACTED: Rotation-GPTAQ antagonism

**This was caused by a Cholesky convention bug, not a real architectural incompatibility.** With the correct Cholesky factor (U^T U = H^{-1}), rotation+GPTAQ is SYNERGISTIC (+12.5% to +55.8% improvement). The i.i.d.-Gaussian error explanation was unsupported — it was an artifact of the buggy correction amplifying error rather than reducing it.

## 5. RETRACTED: "Two incompatible paradigms" architectural conclusion

The claim that rotation and correction are fundamentally incompatible was WRONG. The correct conclusion is: rotation and GPTAQ compose synergistically when both are correctly implemented. The full stack (rotation + GPTAQ + allocation) is the winner.

## 6. Greedy optimizer limitation (STILL VALID)

When GPTAQ alone would beat rotation (L55_gate), the greedy optimizer commits to rotation first. But with the corrected Cholesky, GPTAQ on top of rotation beats both, so this is less of an issue. The multi-iteration alternating optimizer does re-optimize after correction is accepted.
