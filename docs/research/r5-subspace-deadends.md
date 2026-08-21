# R5-Subspace: Dead Ends

## D1: Joint PCA (generalized eigenvalue) — broken by QR orthonormalization
Generalized eigenvectors are H_W-orthogonal, not Euclidean-orthogonal. QR orthonormalization destroys the importance ordering. Weakest PCA variant (7× worse than activation PCA). [Confirmed in v3]

## D2: 4-bit quantized projection — too much error
Storing U at 4-bit reduces overhead but introduces hess_err ~0.16-0.19, worse than baseline K. Projection must be at least 8-bit. [From v1, still valid]

## D3: Weight PCA worse than activation PCA
SVD of W identifies where W has energy, not where activation error is costly. 2-4× worse. The objective tr(H_X · E · E^T) is weighted by activation covariance. [Confirmed in v3]

## D4: Global ResQ impractical for weight-only PTQ
n²×2 projection overhead (32768 bytes at n=128) dominates. At matched bytes, uniform FP16 is 3600× better. [Confirmed in v3, with corrected tile PCA overhead]

## D5: Standalone ResQ (without GPTQ) is weak
Plain ResQ (PCA + uniform quantization in subspaces) is only 2-5× better than baseline K while using 3-5× more bytes. GPTQ within subspaces is what makes the decomposition worthwhile. [Confirmed in v3]

## D6: v1 Cholesky bug (lower-triangular) — zero GPTQ propagation
inv_cholesky returned lower-triangular matrix; GPTQ read upper rows (all zeros). v2 returned U@U.T=inv(H) (wrong orientation). v3 fixed: U.T@U=inv(H) via chol(inv(H)).T. GPTQ improvement verified at 59.6% over independent quantization. [Fixed in v3]

## D7: v1 quantizer mismatch — resq_gptaq used 16×1, baseline 16×16
The "5-15× improvement" in v1 was confounded by unmatched quantizer granularity (16× finer scales in ResQ arm). [Fixed in v2/v3]

## D8: v1 tile PCA 8× byte overcounting
Projection charged per output tile instead of per input tile. Corrected overhead: 1.22-1.56× K+1 (not 3-5×). [Fixed in v2/v3]

## D9: Adaptive rank non-monotone (not monotone as v1 claimed)
Error has 8-13 upward steps per case due to partial Hadamard blocks. Budget constraint now enforced; optimal r varies (r=12-16). [Fixed in v2/v3]
