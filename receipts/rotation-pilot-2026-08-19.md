# Learned-rotation pilot: NO-GO, by 725x

**Date:** 2026-08-19 (evening). Data: `receipts/rotation-pilot-2026-08-19.json`.
Question (docs/13): can ParoQuant-style learned K=8 Givens rotations replace EXL3's
fixed 128-point Hadamard pre-transform and push trellis KLD below 0.0027?

## Result (layer 0 mlp.gate_proj, 5120x17408, K5 trellis, MSE in ORIGINAL weight space)

| arm | reconstruction MSE | vs identity |
|---|---|---|
| **Hadamard (EXL3's own)** | **1.234e-07** | **-99.86%** |
| identity (no transform) | 8.902e-05 | — |
| learned K=8 Givens + scales | 8.955e-05 | +0.6% (worse) |

**Verdict: NO-GO.** The learned rotations do not even recover the identity baseline,
and sit 725x behind the fixed Hadamard they were meant to replace.

## Why this is physically sensible

Eight Givens stages touch each channel ~8 times; the dense 128-point FWHT mixes all
128 channels per block. The trellis codebook is TUNED for hadamard-rotated
(near-Gaussian) weights - the sparse rotation leaves the distribution nearly
unchanged, so tiles quantize as badly as raw weights. ParoQuant's wins are measured
against INT4 grids, not distribution-matched trellis codebooks; the transfer
RotationScout estimated at "5-30% of the INT4 gain" measures as ~0.

## The harness needed two rounds to be trustworthy - kept on the record

Round 1 reported "GO, 95.18%" and was INVALID by its own sanity line: each arm's MSE
was measured in its own scaled domain (non-orthogonal per-channel scalings change
the metric), making Hadamard read 20x WORSE than identity - impossible - while the
learned arm had collapsed to exact identity (angles zero-initialised, negligible
gradient). Round 2 (PilotFixer, commit 0a3f0e8) computes every arm in original
weight space via explicit inverse-transform chains, fails the run on sanity
violation instead of printing GO, and seeds angles randomly. The identity-invariant
diff is 0.0 and Hadamard's 99.86% reduction is exactly the behaviour EXL3 claims -
the instrument now reads correctly, and it reads NO-GO.

## Limits

One matrix, one layer, 10 epochs, 128 sampled tiles/epoch, no Hessian-LDL on either
arm (fair arm-to-arm). A 725x margin is not closable by training budget; more
matrices cannot flip the sign. Closes docs/13's open verdict: the FWHT stays.
