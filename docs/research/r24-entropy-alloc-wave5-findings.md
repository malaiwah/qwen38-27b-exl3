# R24 Wave 5 — Entropy-Constrained Allocation (Scaled)

## Methodology Disclaimer (Audit-Corrected)

**This is a broad in-sample oracle screen, NOT a calibration→validation generalization test.**

The Wave 5 code recomputes H/D/entropy per block, selects K on the same block, and scores HWE on that same block. Despite variable naming suggesting cal/val/test splits, no policy is fit on calibration blocks and applied to held-out blocks. The "validation" blocks are consumed in-sample. The ± values are raw SD over correlated block×K cells, not confidence intervals.

Claims of "generalizes" or "universal" are **retracted** pending:
1. A real arithmetic/range coder (not empirical-entropy oracle)
2. A genuine cal→val→untouched-test policy where allocation decisions are made on calibration blocks and evaluated on untouched test blocks
3. End-to-end KLD authorization (not proxy HWE)

## Coverage
- 9 depths: layers 0, 7, 14, 21, 28, 35, 42, 49, 55
- 6 roles: gate, up, down, qkv, z, out
- 45 tensors, 20 blocks each = 900 blocks total
- Block strata: 8 diagonal + 8 random + 4 off-diagonal per tensor (corrected)
- Correct BF16 decode from HuggingFace safetensors
- Hadamard rotation (not BiIP, per R26/R27)

## Macro Results (n=1080 validation block×K cells, in-sample oracle)

| Metric | Value |
|---|---|
| Entropy DP gain over fixed-K DP | **+58.3% ± 5.5%** (raw SD, not CI) |
| Hadamard gain over fixed-K DP | +6.5% ± 11.7% (raw SD, not CI) |
| Entropy savings K=3 | 22.5% ± 1.5% |
| Entropy savings K=4 | 15.9% ± 1.2% |
| Entropy savings K=5 | 13.0% ± 1.0% |
| Entropy savings K=6 | 12.1% ± 0.8% |

## Per-Role Statistics (in-sample oracle)

| Role | Entropy DP Gain | Hadamard Gain | n |
|---|---|---|---|
| gate | +57.4% ± 4.3% | +4.7% ± 10.0% | 216 |
| up | +56.3% ± 4.2% | +3.4% ± 10.4% | 216 |
| down | +57.7% ± 3.9% | +5.6% ± 10.2% | 216 |
| qkv | +60.7% ± 4.6% | +10.5% ± 11.2% | 144 |
| z | +58.8% ± 4.5% | +3.1% ± 11.7% | 144 |
| out | +60.5% ± 9.6% | +14.8% ± 13.5% | 144 |

Entropy DP gain is consistent across roles (+56.3-60.7%) within this oracle screen. Hadamard has higher variance and is role-dependent.

## Per-Depth Statistics (in-sample oracle)

| Layer | Entropy DP Gain | Hadamard Gain | n |
|---|---|---|---|
| 0 | +59.6% ± 6.0% | +8.2% ± 12.3% | 144 |
| 7 | +55.5% ± 3.8% | +0.7% ± 11.3% | 72 |
| 14 | +56.9% ± 7.5% | +6.9% ± 12.5% | 144 |
| 21 | +57.5% ± 4.8% | +4.9% ± 11.2% | 144 |
| 28 | +58.5% ± 6.1% | +7.4% ± 11.3% | 144 |
| 35 | +58.5% ± 3.9% | +6.2% ± 10.6% | 72 |
| 42 | +59.0% ± 4.4% | +6.7% ± 10.8% | 144 |
| 49 | +60.0% ± 4.6% | +8.4% ± 12.4% | 144 |
| 55 | +57.1% ± 3.9% | +5.9% ± 10.3% | 72 |

No depth-dependent variation observed within this oracle screen.

## Observations (Scoped to Oracle Screen)

1. **Entropy allocation shows consistent in-sample benefit**: +58.3% raw mean across 45 tensors, 9 depths, 6 roles. The mechanism (low-entropy tiles get higher K) produces consistent same-block improvement. Whether this generalizes to held-out blocks is untested.

2. **Hadamard is inconsistent**: +6.5% mean with 11.7% std — high variance, sometimes negative. Consistent with R26/R27.

3. **Role and depth show no trend in the oracle screen**: Entropy DP gain is flat across roles and depths. Hadamard varies by role.

4. **Entropy savings stable**: 12-22% savings across all tensors, consistent with R20.

## Limitations
- **In-sample oracle**: No cal→val→test generalization. Allocation is fit and scored on same block.
- **Raw SD, not CI**: ± values are raw SD over correlated block×K cells, not confidence intervals.
- **Empirical-entropy lower bound**: Not a real arithmetic/range coder. No model/framing bytes counted.
- **Proxy HWE**: Not end-to-end KLD. QSRT lesson: local metrics can invert model KLD.
- **Block-diagonal surrogate**: Cross-tile HWE terms omitted.
- **Per-tile model overhead not included**: Needs R21 shared codebook to be practical.
- **Synthetic calibration**: Not real held-out activations.
- **K=6 not tested**: 0% gain expected (all tiles at K_MAX).
