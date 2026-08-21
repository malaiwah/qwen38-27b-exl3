# R34 — Candidate-conditioned downstream targets and legal block transforms

**Status:** scientific negative at the Phase-B validation gate, 2026-08-21. No action is eligible for R37 allocation and the untouched test was not opened.

## Actual-stock experiment

The tensor mechanism screen used Qwen3.8 layer 55 with a fresh actual EXL3 MCG K5 fused gate/up action and K6 down actions. Calibration and validation each contain 63 real BF16 module-boundary rows from different complete documents. For every upstream action, the runner decoded gate/up, rebuilt

\[
H_q=\operatorname{SiLU}(XW_{g,q}^{T})\odot(XW_{u,q}^{T}),\qquad
C_q=H_q^T H_q/N,
\]

and fit the down target relative to the incumbent decoded down tensor with the dual ridge/minimum-norm solve. The continuous target was process-local and discarded. Every reported arm is a stock Viterbi/BlockLDLQ payload with the same 66,891,780 returned buffer bytes, no sidecar, and no new decode operation.

The operational encoder is the emergency-qualified clean EXL3 `5f3c537`/tree `ffc0a1d` sm_120a CUDA 13.2 build after the shared-cache relink:

- extension: `e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e`;
- harness: `d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d`;
- schema: `275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8`.

Before accepting results, fresh A0 and strength-zero full fused gate/up encodes produced the same payload SHA256 `5233c2349bf7754fddedb4fa59d444079e2311e10f138254d28f890082b83238`, byte-identical source-basis reconstruction, and finite decodes. Historical R30 evidence remains attributed to its older execution pins and is not relabeled.

## Validation result and stop decision

Ridge `1e-4` relative to mean Gram diagonal was marginally better than no ridge after stock encoding and was selected using validation only. Its absolute lambda was `0.34656941612257486`. The registered 2×2 factorial then used two stock encoder calls per arm; the ridge selector used two legal candidate calls plus two matched source-control calls.

| Target | Second pass | Calibration block MSE | Validation block MSE | vs source stock |
|---|---|---:|---:|---:|
| source | matched control | `1.51020e-5` | `6.04953e-4` | control |
| source | ICBQ seam reroll | `3.65177e-5` | `6.52140e-4` | **+7.80% worse** |
| reconstructed upstream | matched control | `9.18187e-7` | `7.44720e-4` | **+23.10% worse** |
| reconstructed upstream | ICBQ seam reroll | `2.18633e-5` | `7.92392e-4` | **+30.98% worse** |

The reconstructed target improved calibration by 93.9% yet reversed on the document-disjoint validation sequence. Seam reroll also hurt both target families. This is the preregistered falsifier: a local/capture fit does not support promotion when validation block output regresses.

One-tensor full-vocabulary KLD was therefore **not run**. Acceptance required improvement in validation block output before KLD; spending another checkpoint/KLD window after every arm failed that gate would violate the lean stop rule. There is no eligible action row to consume.

## Architecture screens

The real L55 fused full-attention capture verifies the actual layout: Q/output-gate `63×12288`, K `63×1024`, V `63×1024`, corresponding to 24 Q/output-gate heads, 4 KV heads, and head dimension 256. The legal zero-runtime families are narrowly:

- V monomial with gate coordinates permuted only and inverse O monomial; gate sign/scale is forbidden because it does not commute through sigmoid;
- shared paired signs on the 64 rotary coordinates plus a shared signed permutation on non-rotary coordinates, with Q/K norm weights co-permuted; arbitrary diagonal scaling, dense Q/K rotations, and unmatched GQA head permutations are not claimed legal.

For dense SwiGLU, mathematical U_A-only and U_B-only block-H128 identities with `U_R=I` passed in float64. On the real L55 BF16 boundary, however, H128→BF16→inverse-H128 was not bit-exact: U_A-only maximum error `0.125`, MSE `9.23e-7`; U_B-only maximum error `0.0625`, MSE `3.00e-7`. The unfused MPS screen measured 3.70 ms per H128 over `63×17408`, but this is not a fused RTX5090 production cost. Both runtime arms were stopped and are not candidates.

R29 exposes only surrounding fused GDN boundaries. It lacks `state_before`, `state_after`, conv input/output, and RMSNormGated internal traces. R34 therefore rejected GDN transform inference rather than substituting a simplified recurrence.

## Evidence

Canonical receipt: `receipts/wave5/r34-reconstructed-down.json`.

Raw remote evidence remains at `/tmp/qwen38-wave5-r34`:

- factorial file SHA256 `d511618a66086b51a63b7c8f22fe104ce3f66424ca2f49e09ebc40ea4a457993`, content SHA256 `4fa4e2f96b87f18015fb68e86eb20af1ec7e22db14869d182df6391d744256eb`;
- validation capture manifest file SHA256 `52c27c4cd00937c0672f52d8f0d117fb67dd99c65a60c81d7715016b70290d7a`, content SHA256 `d9f182f0f49b40746089a9a73a21d854892e74bf3ee2b182b5093a8b82ed9c3c`;
- action JSON and serialized payloads under `/tmp/qwen38-wave5-r34/results`. The executed action JSON used Python-accepted `gate_up`/`down` role labels that the frozen JSON Schema rejects. The current runner corrects these to `other_dense`/`down_proj`; the raw actions remain ineligible and are not sent to R37. This metadata defect does not change the decoded payloads or validation falsifier.
