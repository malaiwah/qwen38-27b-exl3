# R3-Rotations: Dead Ends

## 1. Gen-eigen rotation (joint generalized eigenvectors of (H_X, W^T W + εI))
**Status: Dead end.** The full eigenvector matrix V costs d²×4 bytes sidecar (32 bits/elem for d=128), which is prohibitively expensive. Additionally, numerically unstable for ill-conditioned W^T W — the Cholesky fallback produces NaN/Inf on some tensors. The joint optimization of activation and weight directions doesn't translate to better quantization because the eigenvectors concentrate energy in a few directions (the opposite of incoherence). Use Hadamard instead.

## 2. Block-Givens / Householder outlier annihilation
**Status: Weak.** Greedy Householder reflectors within 16-element blocks achieve local outlier suppression but don't create global incoherence. The μ(W) after block-Givens is much higher than after Hadamard (e.g., 8.81 vs 3.89 for L0_gate). The greedy approach creates local structure but doesn't spread outliers globally. Hadamard's guaranteed incoherence (μ=1) is superior. The block-Givens sidecar is also larger (1536B vs 32B for Hadamard signs).

## 3. Output-only rotation without BiIP scaling
**Status: Ineffective on most tensors.** Output Hadamard alone (scale=none|in=none|out=hadamard) provides minimal improvement over baseline. The BiIP scaling is necessary to equalize row norms before rotation can spread outliers effectively. On L0_down, output Hadamard without scaling is actually worse than baseline (9.34e-04 vs 9.37e-04 — essentially no change). This confirms KronQ's finding that BiIP scaling and rotation are complementary, not independent.

## 4. Butterfly rotation as a replacement for Hadamard
**Status: Comparable but more expensive.** Butterfly (U₁⊗U₂) achieves similar incoherence to Hadamard but with 2× the sidecar cost (2321B vs 1058B). It's the best arm on L0_down (64.6%) but the gain over Hadamard (57.1%) doesn't justify the extra bytes. Butterfly's advantage is logarithmic application cost, but for 128×128 matrices this is irrelevant. Could matter for full-size tensors (5120×17408) where the O(n log n) vs O(n²) difference is significant.

## 5. Per-column quantizer partially tautological for BiIP scaling
**Status: CONFIRMED by per-tile comparison.** With per-column quantization, BiIP scaling alone shows small or negative benefit on some tensors (L0_down: -50.2%). With per-tile 16×16 quantization, BiIP scaling shows much larger benefit (L0_down: +22.6%). The per-column result was partially tautological (scaling commutes with per-column quant). The Hadamard rotation benefit is NOT tautological — it helps both quantizers. R10-CoupledBlocks found per-column makes permutation a no-op; our per-tile results show rotations are MORE effective with per-tile quant (65-97% vs 55-87% per-column). The EXL3 trellis quantizer operates on tiles, so rotations should be even more effective in production.

## 6. Applying rotations to MLP gate/up jointly
**Status: Illegal.** R10-CoupledBlocks confirmed that rotations do NOT commute with SiLU+elementwise product. Applying a rotation to the intermediate dimension of gate_proj and up_proj would break the SiLU(gate(x)) * up(x) computation. Only permutations are legal for the MLP intermediate dimension. Rotations can be applied independently to each matrix's input and output dimensions, but not to the shared intermediate dimension.

## 7. Rotation-GPTAQ antagonism (RETRACTED)
**Status: RETRACTED — was a Cholesky convention bug.** Initial R9-GroupOrbit finding that rotation+GPTAQ degrades the objective by 476-5239% was caused by wrong Cholesky orientation (U U^T = H^{-1} instead of correct U^T U = H^{-1}). When fixed, rotation+GPTAQ is synergistic (+42-76% improvement). The correct stack is BiIP+Hadamard+GPTAQ+allocation. R3's PoC does not use GPTAQ, so our numbers are unaffected.

