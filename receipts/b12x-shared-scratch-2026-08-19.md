# B12X W4A16 for the trellis path: one shared scratch buffer, +51% prefill on the fidelity profile

**Date:** 2026-08-19
**Axis:** PP (prefill), on the profile that satisfies the KLD criteria
**Result:** fidelity-profile PP **1080.6 ± 2.2 → 1631.0** tok/s (+50.9%) at
unchanged context (238,400), unchanged vision, TG fox 207.6 → 210.0, essay
92.9 → 90.0, and (see below) unchanged fidelity.

## Why this was even tried

The `PROFILE=fidelity` config shipped with `VLLM_EXL3_SKIP_TRELLIS_PREP=1`,
which disables B12X entirely and sends every matrix through the fused
`vllm::exl3_gemm` trellis kernel. That was not a considered choice — it was
inherited from an earlier debugging session. The prefill torch-profile
(`receipts/prefill-profile-2026-08-19.md`) had already shown the two paths cost
very different CPU time per call:

| path | CPU per call |
|---|---|
| `vllm::exl3_gemm` (fused trellis) | 4.72 ms |
| `vllm::b12x_trellis_linear_out` (W4A16) | 0.30 ms |

Both are trellis-exact — B12X W4A16 consumes the *same* packed trellis payload,
so switching should be fidelity-neutral. That made "why is B12X off on the
fidelity profile?" a cheap question with a large possible upside.

## What happened first: engine OOM on the first real prefill

Turning it on booted, served a 1-token sanity request, and then died on the
first actual prefill:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 134.00 MiB.
GPU 0 has a total capacity of 31.40 GiB of which 117.56 MiB is free.
  exl3.py, apply(): output = outputs[0] if len(outputs) == 1 else torch.cat(...)
```

31.27 GiB of 31.40 in use. Reproduced at `--gpu-memory-utilization` 0.93, 0.955
and 0.967, and at contexts 131,072 and 238,400 — so it was not a headroom
setting and not a KV-cache sizing question.

## Root cause: our own buffer cache, multiplied by shape count

`_b12x_trellis_linear` cached its four working buffers under
`key = (m, k, n, bits, dtype, device)`. B12X's own sizing rule is
(`b12x/moe/_shared/kernels/w4a16/host.py:254`):

```python
elements = min(size_n * route_slots, sms * 4 * moe_block_size * 256)
```

For every matrix in this checkpoint the **right-hand cap binds**: with 170 SMs
and block 64 that is `170*4*64*256` = 11,141,120 fp32 elements = **42.5 MiB**,
regardless of shape. So the per-shape cache was paying 42.5 MiB of scratch
*per distinct shape*, plus two `m × n` tensors and one `m × k` tensor per shape.
With 409 matrices, several distinct `n` (5120 / 4096 / 17408 / 34816) and a
3072-row prefill chunk, that reaches tens of GiB. The 42.5 MiB figure also
explains the shape of the failure: the process was ~130 MiB short, i.e. it died
part-way through allocating the *next* few shapes, not on one huge request.

The insight that fixes it: **`c_tmp` is a transient GEMM accumulator**, not
state. Calls inside a forward pass are serialised on one stream, so a single
buffer is safe for the entire model, and because the cap is shape-independent,
one buffer at the cap satisfies every matrix.

## Fix

`vllm-exl3-multiprecision.py`:

- New `_b12x_trellis_c_tmp_shared(device)` — allocates **one** fp32 buffer per
  device at the shape-independent cap and returns it to every caller. Allocated
  once and **never grown**, so a pointer captured into a CUDA graph stays valid
  for the process lifetime.
- `_b12x_trellis_linear` keeps its per-shape cache **only for `m <= 128`**
  (`_B12X_TRELLIS_BUF_CACHE_MAX_ROWS`). Decode, including the MTP draft loop, is
  CUDA-graph captured and must stay allocation-free. Prefill chunks are much
  larger and are not graphed, so their `output` / `gemm_output` / `rotated_f16`
  are allocated per call and recycled by the caching allocator instead of being
  retained forever.

`EXL3_PATCH_SHA256` re-pinned to `89e2537290...`.

## Measured effect

Same profile, same checkpoint, only `SKIP_TRELLIS_PREP` and the fix differing:

| | b12x OFF (shipped) | b12x ON + shared scratch |
|---|---|---|
| PP, 2051-tok | 1080.6 ± 2.2 | **1631.0** (1615-1632 over 6 reps) |
| TG fox | 207.6 ± 0.2 | 210.0 (acc 1.000) |
| TG essay | 92.9 ± 0.1 | 90.0 |
| max context | 238,400 | 238,400 |
| vision + MTP | pass | pass |
| 200k-token prompt | 200 OK | **200 OK** |
| stress gate | ROBUST | **ROBUST** |
| KV cache available | 8.89 GiB @ util 0.93 | 9.30 GiB @ util 0.945 |

The fix is also worth **+0.83 GiB of KV** at equal utilisation (9.16 → 9.99 GiB
at util 0.967), because the retained per-shape buffers were counted against the
KV budget at profiling time.

Utilisation had to rise 0.93 → 0.945 because B12X's *load-time* prep costs
~0.89 GiB of persistent memory. That is a separate cost from the scratch bug and
is real: with prep on, `Available KV cache memory` drops 8.89 → 8.00 GiB at
identical utilisation. This partially rehabilitates ledger item L9 — B12X prep
does not copy the trellis payload (weights stay 18.83 GiB, as L9's retraction
established), but it is not free either.

## Verification

- PP/TG/vision/context: numbers above, `tools/bench_lib.py`.
- Criterion 5: `bench_lib.long_ctx_check(target_tokens=200000)` → `True`, and the
  service answered the "Paris" sanity prompt afterwards (no poisoning).
- Robustness: `tools/stress-gate.py` → **ROBUST** (mixed prefill / decode /
  vision / 12k interleaved, `alive=True` throughout).
- Fidelity: `tools/kld-run-alltrellis-b12x.sh` (identical to
  `kld-run-alltrellis.sh` except `SKIP_TRELLIS_PREP=0`), 512 contexts,
  shard-0000, BF16 reference — result recorded in the CORRECTNESS section below.
  This is the required kernel-correctness check: B12X W4A16 is a *different
  kernel* on the same packed payload, so "trellis-exact in principle" is not
  accepted without a measurement.

## Does this rescue criterion 1?

No. 1631 is 4.3× short of 7000, and the bake-off
(`tools/kernel-bakeoff.py`) bounds every decode-per-chunk trellis variant well
below that. What it does do is cut the fidelity profile's prefill deficit in
half at zero fidelity cost, which is explicit partial credit on the PP axis for
the profile that already satisfies criteria 2-6.

## CORRECTNESS (required kernel verification) — PASS

B12X W4A16 is a *different kernel* on the same packed trellis payload, so
"trellis-exact in principle" was not accepted. Verified at the strongest
available level — a full 512-context KLD run against the BF16 reference, not a
spot check — via `tools/kld-run-alltrellis-b12x.sh` (byte-identical to
`kld-run-alltrellis.sh` except `VLLM_EXL3_SKIP_TRELLIS_PREP=0`):

| | KLD mean | ci95 | p95 | p99 |
|---|---|---|---|---|
| b12x OFF (`report-alltrellis-int6emb.json`) | 0.003412 | [0.003172, 0.003680] | 0.010273 | 0.034886 |
| **b12x ON** (`report-alltrellis-b12x.json`) | **0.003407** | [0.003167, 0.003673] | 0.010264 | **0.034823** |

Delta **−0.16%** on the mean, with the two CIs almost entirely overlapping, and
p95/p99 marginally *better*. This is fidelity parity, as the shared-payload
argument predicted; criteria 3 (≤0.012) and 4 (p99 ≤0.15) both still pass.

## Phase-0 harness, n=3 boots

`receipts/bench-fidelity-b12x-2026-08-19.json`:

| metric | b12x OFF | b12x ON | delta |
|---|---|---|---|
| PP | 1080.6 ± 2.2 | **1630.0 ± 3.6** | **+50.8%** |
| TG fox | 207.6 ± 0.2 | 210.2 ± 0.2 | +1.3% |
| TG essay | 92.9 ± 0.1 | 89.8 ± 0.1 | −3.3% |
| MTP acceptance | 0.3041 | 0.281 | −7.6% |
| boot time | — | 47.5 ± 2.4 s | — |

The essay/acceptance dip is real but small and stays well clear of criterion 2a
(≥83): 89.8 passes with 8% margin. Reported rather than hidden.

## Regression gates — both profiles PASS with this patch

The throughput profile also runs with `SKIP_TRELLIS_PREP=0`, so this patch
changes it too and it was re-gated rather than assumed:

- `PROFILE=fidelity`: **8/8, exit 0** — PP 1596.4, essay 88.78, fox 208.3,
  ctx 238,400, acc 0.2814, sanity/vision/long_ctx PASS.
  `receipts/verify-fidelity-b12x-2026-08-19.json`
- `PROFILE=throughput`: **8/8, exit 0** — PP 7504.4, essay 93.13, fox 184.7,
  ctx 250,000, acc 0.298, sanity/vision/long_ctx PASS.
  `receipts/verify-throughput-b12x-2026-08-19.json`

## Process note

The first attempt at gating the fidelity profile produced PP 7449.5 at
ctx 250,000 — i.e. the *throughput* profile's numbers. Cause: `podman rm -f`
without `systemctl --user stop` first, so the unit's `Restart=always` relaunched
the container with default env. This is exactly ledger item **L8**, walked into
again; the tell is that the measured values match the other profile rather than
looking merely "off". Any A/B on this host must stop the unit first and assert
the effective config (`tools/boot-cfg.sh` does).
