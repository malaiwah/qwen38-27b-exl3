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

<!-- EXACT_PR_QUALIFICATION -->
