# lm_head -> b12x small-M at decode: refuted, and the floor explains why

**Date:** 2026-08-19 (night). Follow-up to receipts/nsys-round2-2026-08-19.md, which
measured the fused-K6 lm_head at 14.8% of the MTP decode window (567 us/call) and
proposed routing it to B12X's cooperative small-M K6 kernel.

## The experiment

`VLLM_EXL3_B12X_LM_HEAD_MIN_M` added to the multiprecision patch (default-inert
None; 0 forces lm_head through B12X at every row count; per-layer cached flag, no
hot-path string scan). A/B on both profiles, same probe protocol arm-vs-arm:

| profile | arm | PP | fox | essay |
|---|---|---|---|---|
| fidelity | lm_head->B12X | 3015.3 | 218.6 | 88.7 |
| fidelity | control | 2979.9 | 216.4 | 88.8 |
| throughput | lm_head->B12X | 9707.4 | 164.4 | 82.0 |
| throughput | control | 9547.0 | 165.3 | 82.4 |

**No effect on either profile** (deltas within single-boot noise; acceptance
identical). The dial stays in the patch, documented and default-inert.

## Why the 14.8% share is irreducible at K6

One decode call streams the entire lm_head: 248,320 x 5,120 at 6 bpw = ~954 MB.
At the measured sustained 1,840.5 GB/s that is a ~519 us floor; the fused kernel's
567 us is **~92% of the weight-streaming bound**. The share is not kernel
inefficiency - it is arithmetic. The only real levers are (a) smaller lm_head
weights (FP8/FP4 head - a fidelity tradeoff against the K6 head that anchors our
0.0034 KLD), or (b) fewer lm_head calls per token (already minimized by MTP's
draft/verify structure).

Method note: the quick-probe protocol reads fox ~165 on throughput where the full
harness reads 187.4 - protocol offset, stated so the table is not misread as a
regression; the comparison is arm-vs-arm at identical protocol.
