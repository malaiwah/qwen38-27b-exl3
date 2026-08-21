# R36 — Final-KL legal-path guidance and conditional BCJR/GSQ/QES

**Status:** scientific negative at the preregistered hard-path gate. No action is eligible for R37. No validation or untouched-test KLD result is claimed.

## Question and decision

R36 asked whether an expensive encode-only final-forward-KL target shift could improve an actual stock EXL3 K5 payload without changing serialized buffers or the decoder. The lane stopped before validation because the first hard, stock-projected 20-block screen was not sign-stable: only 11 of 20 preregistered working-coordinate blocks had a negative final-KL linear term; 7 were positive and 2 did not change path.

The aggregate guided linear term was negative, but it is **not KLD** and does not override the block-stability gate. Therefore low-temperature BCJR, legal-transition Gumbel, QES, relinearization, checkpoint materialization, and validation KLD were not run.

Canonical raw result: `receipts/wave5/r36-finalkl-bcjr.json`.

## Frozen identities

- Source: Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Tensor: `model.language_model.layers.0.mlp.gate_proj.weight`, `[17408, 5120]`, BF16 source SHA256 `33f8e7677f361290d5f6a415c00cb56149a8fdbea20122bc13f0e1004b469565`.
- Action: full-tensor K5/MCG, stock H128 signs/scales, stock FP16 `suh`/`svh`, stock `block_ldl(b=16)`, actual CUDA Viterbi, and stock fixed-stride packing.
- EXL3: clean commit `5f3c537ca9d89893d771256f5c43c93656553fbb`, tree `ffc0a1d31c25d4174b96adffef3727f12a7056c7`.
- Operational extension: `e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e`, mounted directly read-only; no JIT/relink.
- R30 action harness/schema: `d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d` / `275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8`.
- R29 combined split manifest file/content: `a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc` / `151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e`.
- Calibration/validation selections: `490a15969bf7b62b585f24cce644dae48f8021f534b0b2bc7553a46a989ea259` / `4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de`.
- Untouched test was not accessed.

## Actual payload anchor and strength zero

Fresh stock and callback strength-zero actions were independently encoded under the current operational binary. Both decoded finitely and were identical:

| check | result |
|---|---:|
| stock payload SHA256 | `d21d385b80645fd53292cb5ac3c410107060873bd2937ecb97fd6d6a6769a27f` |
| strength-zero payload SHA256 | `d21d385b80645fd53292cb5ac3c410107060873bd2937ecb97fd6d6a6769a27f` |
| source-basis decoded SHA256, both | `ed3e55d604afb6094a17db76a8eaf6433e884c057b90ef2c9709c2ee5aff7e85` |
| raw action buffer bytes | 55,750,660 |
| incremental bytes / hot operations | 0 / 0 |

The payload contains only the normal stock `trellis`, `suh`, `svh`, and `mcg` buffers. The candidate callback returns shifted target tiles; the pinned harness alone runs Viterbi and produces indices.

## Final-forward-KL gradient and exact decoder adjoint

The gradient was captured on eight calibration positions through all 248,320 vocabulary logits in the registered forward direction:

\[
\nabla_W\,\mathrm{KL}(p_{\mathrm{BF16}}\,\|\,q_{\mathrm{stock\ K5\ anchor}}).
\]

The anchor calibration mean was 0.000539702596 nats. This number is a calibration diagnostic, not validation KLD.

The source-basis `[out,in]` gradient was mapped to pre-tensor-core-permutation working coordinates with the adjoint of the actual serialized-FP16-scale decoder:

1. transpose to encoder `[in,out]` layout;
2. multiply by serialized `svh`;
3. apply self-adjoint right H128;
4. multiply by serialized `suh`;
5. apply self-adjoint left H128.

Inner products agreed on both actual hard legal displacements:

| hard payload | source inner product | working inner product | absolute error |
|---|---:|---:|---:|
| guided | -1.530798316e-6 | -1.530824063e-6 | 2.575e-11 |
| random | -7.208258877e-8 | -7.210095487e-8 | 1.837e-11 |

## Equal-budget hard legal screen

Twenty working-coordinate 16×16 tiles were frozen before the hard run. Guided and random arms used the same tensor, K, MCG codebook, seed, strength 0.1, tile list, stock curvature, and unshifted-source stock scale fit. Each callback action consumed one complete encode and two actual Viterbi calls per tile: one stock anchor path and one hard projection of the callback target. Both serialized to the same 55,750,660-byte stock buffer layout.

| arm | negative blocks | positive blocks | unchanged blocks | aggregate linear term |
|---|---:|---:|---:|---:|
| final-KL guided | 11 | 7 | 2 | -1.530798316e-6 |
| equal-budget random legal control | see canonical raw rows | see canonical raw rows | 2 | -7.208258877e-8 |

The frozen stability threshold was at least 16/20 negative guided blocks. Observed 11/20 failed. The aggregate direction was about 21× the random control, but heterogeneous block signs make that tiny first-order result insufficient to spend a validation build or to call a model-fidelity improvement.

## Conditional actual-graph methods

The runner contains a source-bound implementation of the actual K5/MCG state graph for conditional follow-up:

- 2,048 states (`2^(16-K)`), 32 legal predecessors per output state, 256 positions;
- emitted 16-bit symbol `(predecessor << K) | output_state`;
- MCG marker/multiplier `0xCBAC1FED` and stock FP16 codebook decode;
- cyclic boundary closure matching the stock kernel's second pass;
- conditional low-temperature sum-product at positive temperature, with T≈0.3 intended;
- legal Gumbel target perturbations and antithetic QES directions;
- mandatory final hard projection by the pinned stock CUDA Viterbi.

These paths were self-tested for topology, cyclic closure, normalized BCJR marginals, hard projection, antithetic QES, and adjoint ordering. They were **not executed as research arms** because the hard gate failed. No soft objective or marginal is reported as an output action.

## Infrastructure failures and stopping discipline

Two GPU-placed final-KL gradient attempts failed with CUDA OOM before a metric was read (63 positions at 20 GiB placement, then 8 positions at 12 GiB). The valid run used 8 positions with CPU offload and 1 GiB GPU placement. This is recorded as infrastructure history, not additional scientific iteration.

After the hard gate failed, R36 stopped immediately and handed the already-stopped GPU/service state directly to the Main-queued R30 capture operator. R36 did not open the test, create a checkpoint candidate, run validation replay, or claim KLD/EAR/p99/top-1 improvement.

## Independent review

The required OpenAI reviewer returned **APPROVE** with no conclusion-changing blocker and explicitly confirmed the actual graph, stock hard projection, strength-zero identity, decoder adjoint, matched budgets, 20-block falsifier, absence of a validation-KLD claim, and complete action bytes. The only nonblocking caveat is that standalone safetensors container hashes may differ while the four operational buffer hashes/bytes and canonical payload digest agree.
