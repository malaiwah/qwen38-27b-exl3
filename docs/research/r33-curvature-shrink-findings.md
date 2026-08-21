# R33 — Real covariance shrinkage and stock BlockLDL variants

**Status:** scientific negative; no new shrinkage or banded action.

## Execution contract

The campaign used the operational R30 stock harness (`d4dfd35c…2605d`), schema (`275644ed…7af8`), clean EXL3 `5f3c537…`, and the directly mounted qualified extension (`e2e26e0d…801e`). A fresh A0 smoke passed before the campaign: finite source-basis reconstruction and byte-identical strength-zero buffers. Its receipt is `receipts/wave5/r33-a0-e2e.json` (SHA256 `36c5d0c501984251ae94672b858ed21683cfcd5d78bd3538db8b505aec1a7123`).

All full-tensor arms fixed K5, MCG, seed 330033, FP16 `suh`/`svh`, stock global scale search, stock H128/sign streams, pinned Viterbi, payload packing, and decoder route. Every L0 gate arm stored exactly 55,750,660 buffer bytes; every L55 gate arm stored the same count. There is no post-hoc correction or decode hot-path change.

R29 publishes $X^TX/N$, while stock `finalize_capture_H` divides its accumulator by `count`. The runner therefore supplied the real $X^TX$ sum and observation count, so stock consumed $X^TX/N$. Curvature changes ran at R30's curvature callback. Band variants changed only the stock factor: strict-lower off-diagonal 16×16 blocks outside the 16/32/64-feature band were zeroed before the unchanged reverse stock recurrence. R30 alone ran Viterbi.

## Full source-basis tensor result

Two promoted real-activation full tensors were measured: L0 gate and L55 gate, each using 63 real calibration tokens.

| Arm | L0 real-input HWE / stock | L55 real-input HWE / stock |
|---|---:|---:|
| stock full H / rho=1 | **1.000** | **1.000** |
| rho=0 | 112.407 | 34.250 |
| rho=.05 | 18.686 | 4.947 |
| rho=.10 | 10.544 | 3.168 |
| rho=.25 | 4.491 | 1.819 |
| rho=.50 | 2.186 | 1.313 |
| rho=.75 | 1.386 | 1.123 |
| band 16 | 112.407 | 34.250 |
| band 32 | 108.037 | 32.266 |
| band 64 | 100.875 | 29.403 |
| Ledoit–Wolf | 2.989 | 1.010 |

Stock/rho=1 is the endpoint winner on both full tensors. The best interior rho still regresses real-input HWE by 38.6% at L0 and 12.3% at L55. Every truncated recurrence is substantially worse. The analytic spherical-target Ledoit–Wolf retention was 0.4102 at L0 and 0.8912 at L55; neither beats stock.

The 8/16/32-token sweep is also unstable and much worse when scored on all 63 real calibration tokens. Rho=.75 never provides a robust advantage over the same-sample-count stock arm.

## Fisher and output-covariance diagnostics

Output covariance and Fisher were diagnostics only and never substituted for the input $H_X$ recurrence.

- L0 Fisher-row mean versus output-covariance-diagonal Pearson correlation: 0.0147.
- L55 correlation: 0.0390.
- Some shrunk arms reduce the one-sequence Fisher-diagonal weighted error slightly while severely worsening real-input HWE. This metric disagreement is exactly why the Fisher/output diagonals cannot redefine the stock input recurrence or promote an action without model-output validation.

## QMM / innovation-rate oracle

The runner recorded only the last legal-path callback's sampled Gaussian target-rate minus reconstruction-error rate: 4.8615 bits at L0 and 4.8594 bits at L55. This is an achieved SNR-equivalent bit diagnostic, not remaining coding headroom: it increases as reconstruction error falls. It is also not aggregated across callback invocations. Therefore the preregistered oracle-minus-stock QMM gap was **not measured**, the 0.1-bit early-stop threshold is not applied, and no QMM/headroom conclusion is drawn.

## Broad screen exception

The nine-depth block screen stopped before its first arm because the R29 preregistered block digest did not equal the reconstructed contiguous BF16 tensor digest. This was one invalid infrastructure attempt; no block row was retained. The full source tensor BF16 hashes were independently verified, and the full-tensor campaign completed. No digest check was weakened or bypassed.

## Validation boundary and decision

The authorized one-tensor capture launcher lane closed blocked before producing an R31 replay row, so no current direct full-vocabulary validation KLD exists. No legacy capture is substituted.

The local actual-stock result is negative: both promoted full tensors choose the rho=1 stock endpoint under real-input HWE, every interior rho worsens that metric, and bands are decisively worse. Because validation KLD is unavailable, this does **not** claim that the preregistered validation falsifier fired. The conservative promotion gate nevertheless requires:

- `selected_action = null`;
- `new_shrinkage_action = false`;
- no R33 row is eligible for R37;
- no QMM result is available to gate R38.

The default throughput service was restored without a profile override; systemd is active, `/health` passes, and 71 GB remained free.
