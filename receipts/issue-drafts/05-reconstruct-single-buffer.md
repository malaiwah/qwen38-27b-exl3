# Issue draft 05 — reconstruct scratch is single-buffered: reconstruct[i+1] can never overlap hgemm[i]

**Repo:** local-inference-lab/vllm · **Cites:** docs/47 F5.2, F8, plan P2.3

## Summary
The prefill dequant scratch is ONE buffer per (device, K, N-chunk) shared by every same-geometry
matrix (`_EXL3_RECONSTRUCT_SCRATCH`, exl3.py:766-767, 785-795); each `_reconstruct_hgemm_into`
overwrites it (exl3.py:951-958), and reconstruct/hgemm issue on the same stream (reconstruct.cu:110,
hgemm.cu:101-104). The shared buffer is a WAR hazard that forecloses overlap even with a second
stream. Measured phase split (receipts/kernel-gap-prefill-phases.json): reconstruct is 12.9–13.2 %
of linear time at M=2048 chunks (4–5 % at M=6144) — all of it serialized against the tensor-core
pipe it could hide behind.

## Proposed fix
2-deep ring buffer + side stream + events. Extra memory: one more buffer per live geometry class
(~510 MB total at current shapes) — must be re-checked against the long-context fit before
defaulting on.

## What would falsify this
An nsys timeline showing reconstruct already overlaps hgemm (it cannot, by the WAR argument, but
that is the direct check), or an A/B where the ring buffer wins <1 % because cuBLAS already saturates
the memory subsystem alongside its MMA work at these shapes.
