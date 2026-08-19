# Evaluation: persistent cache for FP4 load-time conversion — NOT adopted

**Question** (peer-review item, GLM-inspired): the throughput profile converts ~15.9 GiB
of trellis weights to NVFP4 at every boot; GLM-5.2 caches its load-time K6 encodes
content-addressed. Should we do the same for FP4?

**Bound from existing harness data.** Boot wall (launcher-timed, n=3): throughput
**56.0 ± 0.0 s**, fidelity **47.5 ± 2.4 s**. The two profiles differ by the FP4
conversion plus ~0.4 GiB less weight I/O, so conversion costs **≤ 8.5 s of a 56 s boot
(≤ 15 %)** — an upper bound, since fidelity's boot carries b12x prep the throughput
profile also pays. (Next boot's `EXL3→FP4 conversion complete` log timestamps can
tighten this; the decision does not depend on the tightening.)

**Decision: not worth it.** Costs of a cache:
- **~15.9 GiB of disk** per (checkpoint, conversion-config) key;
- a **correctness surface**: the key must include checkpoint revision, GS mode,
  banded-head flag, layer ranges AND the conversion code itself (patch SHA) — a stale
  entry after a patch change would silently serve wrong weights, the exact class of
  silent-corruption bug this project has spent days hunting;
- validation/locking machinery (GLM needed atomic publish + 600 s lock bounds).

Benefit: ≤ 8.5 s per boot on a service whose `Restart=always` recovery is dominated by
weight load + CUDA-graph capture anyway.

GLM's trade is different in kind: their online **K6 encode** is expensive enough to
justify caching (BF16 shared experts encoded per boot across 4 GPUs); our FP4 conversion
is a GPU-side transform of already-loaded tensors. Same pattern, different arithmetic.

Revisit trigger: if boot time ever matters (e.g. autoscaling), measure the exact
conversion share first; if it exceeds ~30 % of boot, re-open with a key that includes
the patch SHA.
