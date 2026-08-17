# Issue draft 01 — b12x K6 gate routes lm_head and k/v projections to a slower path at decode m

**Repo:** local-inference-lab/vllm · **Cites:** docs/47 F3, F7 (qwen38-27b-exl3 research repo)

## Summary
`_b12x_trellis_k6_supported` (`vllm/model_executor/layers/quantization/exl3.py:1202-1218` at
`4d006a4`) admits every K6/MCG matrix with K,N multiples of 128. On K6-heavy checkpoints
(hydrated Qwen3.8-27B: 261/409 matrices, 59.7 % of trellis bytes) this sends two shape classes to
the b12x small-M kernel where `ext.exl3_gemm` is measurably faster at decode row counts:

| shape | b12x µs (m=1, graph-replayed) | exl3_gemm µs | delta |
|---|--:|--:|--:|
| lm_head 5120×248320 | 705.1 | 647.8 | +57.3 |
| k/v_proj 5120×1024 | 27.8 | 17.5 | +10.3 |

Receipt: `receipts/kernel-gap-gemm-bandwidth.json` (RTX PRO 6000 Blackwell SE, real hydrated
weights, CUDA-events over 20-call graphs).

## End-to-end measurement — with its caveat stated up front
One-clause patch (reject `N<5120` and `N>32768` in the gate): single-stream greedy decode
**96.19 → 110.97 tok/s (+15.4 %)** median-of-3 on an RTX PRO 6000 (receipts/kernel-gap-gate-ab.json,
patch: `receipts/kernel-gap-gate-ab.patch`). **Honest bracket: +3–15 %.** The local measurement ran
inside proot (ptrace), which taxes eager launch dispatch; per-call GPU deltas explain only
~0.6 ms/step of the ~4.6 ms/step observed saving — the remainder is eliminated eager-Python dispatch
of the b12x path on the 4 per-step lm_head calls (they run outside the CUDA graph), and that
component is proot-inflated. A bare-docker confirmation on a rental 1×RTX6000 is queued; expect the
true gain between +3 % (pure kernel delta) and +15 % (this measurement).

## Proposed fix — PR branch ready
Shape-policy window behind `VLLM_EXL3_B12X_N_RANGE` (default `5120-32768`, `0` restores old
behaviour). **Branch: `malaiwah/vllm-voipmonitor` @ `kernel-gap/b12x-gate-n-range` (commit
704d94e93)**, based on `codex/gg-exl3-r7-k345-20260810` — chosen because its
`_b12x_trellis_k6_supported` body is byte-identical to the served r34 bytes (the served exl3.py,
5,536 lines sha256 2df9d0799fd3…, exists on no public branch — docs/47 F11). The normative diff
against the served file is `receipts/kernel-gap-gate-ab.patch`.

## What would falsify this
A bare-metal A/B (no proot) showing patched ≤ baseline single-stream at C1, or a KLD spot-check
showing the exl3_gemm route regresses fidelity beyond the checkpoint's published CI (it should not:
exl3_gemm is the fork's own bit-faithful reference path, exl3.py:976).
