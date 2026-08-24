# Native BF16 B12X decode

This integration opts Qwen3.8-27B into the fused K6/MCG dense-decode path from
[`local-inference-lab/b12x#243`](https://github.com/local-inference-lab/b12x/pull/243).
It requires exact b12x commit
`a850891255fb59cdfa49781fa94dc4f7f06629e9`; the dedicated container build
fetches the pull-request ref and fails if it does not resolve to that commit.

The feature is off by default. Eligible bias-free BF16 projections retain BF16
activations through the single cooperative B12X kernel for `M <= 16`. Load-time
preparation must bind and warm the exact launch, and the launch must accept the
runtime tensor. Other shards preserve the existing FP16 ExLlamaV3 boundary or
their existing multiprecision route. B12X owns kernel eligibility, grid choice,
workspace sizing, and launch policy; this integration supplies model metadata
and a finite graph-capture row set.

## Build and enable

Build the opt-in image from the repository root:

```bash
docker build \
  -f docker/Containerfile.b12x-native-bf16 \
  --build-arg REPO_COMMIT="$(git rev-parse HEAD)" \
  -t qwen38-27b-exl3:b12x-native-bf16 .
```

Run it with the normal serving configuration plus:

```yaml
environment:
  VLLM_EXL3_B12X_NATIVE_BF16: "1"
  VLLM_EXL3_B12X_REACHABILITY: "0"
```

The qualified Qwen routing policy remains:

```yaml
environment:
  VLLM_EXL3_B12X_MIN_M: "128"
  VLLM_EXL3_B12X_ANY_BITS: "1"
  VLLM_EXL3_B12X_N_RANGE: "5120-36864"
```

`VLLM_EXL3_B12X_REACHABILITY=1` is a diagnostic mode that emits one structured
record for each distinct load, warmup, or capture decision. Leave it disabled
for production. To roll back the native route, unset
`VLLM_EXL3_B12X_NATIVE_BF16` or set it to `0`; no checkpoint or default serving
profile changes are required.

## Contract and scope

- W4A16 retains its project meaning: BF16 activations with inline K6/MCG weight
  dequantization. No activation-scale math is added.
- The native route is currently for dense, unpaired, K6/MCG Trellis weights on
  the supported SM120/SM121 path. It is not a general speedup for every EXL3
  bit width, MoE layout, or large-prefill shape.
- Packed projections are decided per shard. A native Q/Z shard can coexist
  with narrow K/V fallbacks without changing the public BF16 output dtype.
- CUDA-graph rows are prepared eagerly from the enumerated serving capture
  sizes, bounded to rows 1 through 16. Runtime rejection after load-time
  qualification is fatal instead of silently entering a dtype-incompatible
  fallback.

## Qualification boundary

The original port was qualified on 2026-08-24 on a 98 GB RTX PRO 6000
Blackwell (`sm_120a`) with `FULL_DECODE_ONLY` CUDA graphs, MTP-3, and the real
Qwen3.8-27B hydrated checkpoint. It established exact BF16 projection oracles,
finite/nonzero and top-k checks, fixed caller-owned graph buffers, no replay
allocation, one cooperative kernel per captured projection, and end-to-end
A-B-B-A gains. The pull requests replace that port with the rebased b12x
implementation; exact-commit revalidation results are recorded below before
promotion.

### Exact PR-pair qualification

The rebased pair was qualified from Qwen runtime commit
`fcc12fa80d24672c76158c742239a74a819a3a56`, overlay SHA-256
`f293edf48d05f3b953b5603d7eede2b7bfffc926ab242980bf949678bd0c13eb`,
and b12x commit `a850891255fb59cdfa49781fa94dc4f7f06629e9`. The tested
image was
`sha256:59dd704a768dcb447cd524379a28ad099abfbd503fecdb098ad1127f9fa2a208`
with b12x 1.2.6, CUTLASS DSL 4.6.2, Torch 2.12.0+cu132, and
`CUTE_DSL_ARCH=sm_120a`.

The GPU was an NVIDIA RTX PRO 6000 Blackwell Workstation Edition
(`GPU-ca537fb1-a522-7429-e8c6-4795af22ab78`). Every measured decode sample
was P1 at a 16,365 MHz memory clock with throttle mask `0x0`. All serving arms
completed healthy with zero restarts and no OOM.

The exact hydrated checkpoint gate used
`model.language_model.layers.0.mlp.down_proj` (`K=17408`, `N=5120`) at rows
1, 4, 8, and 16. The 8.35 GB source shard matched its declared SHA-256. Every
row passed two BF16 oracle scenarios with finite, nonzero, fully overwritten
output; minimum cosine was 0.99999958, maximum relative L2 was 0.00039491,
maximum absolute error was 0.00390625, and top-8 set and order were exact. Each
captured native graph contained one cooperative K6/MCG kernel and no separate
rotation-like kernel. Caller-owned tensor addresses remained fixed and eight
replays per row had zero allocator delta. Source payload hashes were unchanged
after the run.

Reachability on the real MTP-3 service found 243 native BF16 plans (240
target/shared shards and three MTP/draft shards) and 1,962 native runtime
dispatches. Observed native rows were 1, 2, 3, 4, 8, 12, and 16, and every
native input was BF16.

The primary serving result is the fixed-trajectory, no-MTP A-B-A bracket below.
Each decode cell is the median of five 256-token samples after one warmup;
prefill is the median of three repetitions. Delta compares the candidate with
the mean of the two baseline medians, so a positive decode delta means faster.

| Workload | Baseline A1 | Native BF16 B1 | Baseline A2 | Candidate delta | A2 vs A1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 decode (tok/s) | 55.416 | 58.832 | 55.420 | +6.16% | +0.007% |
| C4 decode (aggregate tok/s) | 203.798 | 214.873 | 203.565 | +5.49% | -0.114% |
| Prefill (prompt tok/s) | 17,395.5 | 17,271.4 | 17,170.1 | -0.07% | -1.30% |

The MTP-3 bracket measured +5.85% at C1 and +1.96% at C4, with prefill at
+0.03%. Treat its C4 number as secondary evidence: MTP acceptance and generated
trajectories varied between samples, and that baseline bracket drifted 1.72%.

The exact commands, identities, route counts, raw serving samples, acceptance
values, derived ratios, GPU state, and checkpoint-gate summary are preserved in
[`receipts/b12x-native-bf16-pr243-2026-08-24.json`](../receipts/b12x-native-bf16-pr243-2026-08-24.json).
The repeatable serving harness is
[`tools/b12x_native_bf16_e2e.py`](../tools/b12x_native_bf16_e2e.py).
