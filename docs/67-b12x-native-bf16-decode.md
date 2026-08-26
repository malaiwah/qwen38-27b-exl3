# Native BF16 B12X decode

This integration opts Qwen3.8-27B into the fused K6/MCG dense-decode path from
[local-inference-lab/b12x#243](https://github.com/local-inference-lab/b12x/pull/243).
The dedicated image pins pull-request head
<code>706ad0eb54014ed9156dc82a7e0acff691662a89</code> and fails its build if the
PR ref does not resolve to that commit.

The executable b12x source qualified on the GPU is its parent
<code>81595330adba568f6361b396088f91e18b8116f0</code>; <code>706ad0e</code> adds
only the committed raw qualification receipt. The executable Python and kernel
trees are therefore identical.

The feature is off by default. Eligible bias-free BF16 projections retain BF16
activations through one cooperative b12x kernel for <code>M <= 16</code>.
Load-time planning binds and warms the exact launch, retained lookup tensor,
caller-owned workspace, and capture buffers. Other shards preserve the
existing FP16 ExLlamaV3 boundary or multiprecision route. B12X owns kernel
eligibility, grid choice, workspace sizing, and launch policy; this integration
supplies model metadata and a finite graph-capture row set.

## Build and enable

Build the opt-in image from the repository root:

~~~bash
docker build \
  -f docker/Containerfile.b12x-native-bf16 \
  --build-arg REPO_COMMIT="$(git rev-parse HEAD)" \
  -t qwen38-27b-exl3:b12x-native-bf16 .
~~~

Run it with the normal serving configuration plus:

~~~yaml
environment:
  VLLM_EXL3_B12X_NATIVE_BF16: "1"
  VLLM_EXL3_B12X_REACHABILITY: "0"
~~~

The qualified Qwen routing policy remains:

~~~yaml
environment:
  VLLM_EXL3_B12X_MIN_M: "128"
  VLLM_EXL3_B12X_ANY_BITS: "1"
  VLLM_EXL3_B12X_N_RANGE: "5120-36864"
~~~

<code>VLLM_EXL3_B12X_REACHABILITY=1</code> is a diagnostic mode that emits one
structured record for each distinct load, warmup, capture, or runtime routing
decision. Leave it disabled for production. To roll back the native route,
unset <code>VLLM_EXL3_B12X_NATIVE_BF16</code> or set it to <code>0</code>; no
checkpoint or default serving profile changes are required.

## Contract and scope

- W4A16 retains its project meaning: BF16 activations with inline K6/MCG weight
  dequantization. No activation-scale math is added.
- The native route is for dense, unpaired K6/MCG Trellis weights on the
  supported SM120/SM121 path. It is not a general speedup for every EXL3 bit
  width, model shape, MoE layout, or large-prefill path.
- Packed projections are decided per shard. A native Q/Z shard can coexist
  with narrow K/V fallbacks without changing the public BF16 output dtype.
- CUDA-graph rows are prepared eagerly from the enumerated serving capture
  sizes and bounded to rows 1 through 16. An invalid static launch, workspace,
  or lookup contract is rejected during planning, before serving.
- The fused scratch namespace is <code>rotated_compute</code>. The route falls
  back to the generic implementation if another consumer needs an incompatible
  temporary-buffer contract.

## Exact qualification

Qualification used:

- b12x executable source
  <code>81595330adba568f6361b396088f91e18b8116f0</code> (tree
  <code>4e238da8d46f5091430741053a76b57881efb4a6</code>)
- b12x PR head <code>706ad0eb54014ed9156dc82a7e0acff691662a89</code>
  (receipt-only descendant, tree
  <code>5e4c5566062306b76a6860d6a40c0f743aac0c89</code>)
- Qwen build-context revision
  <code>67c338876acc904418d60c6b469d9a62ce62e225</code>
- Qwen runtime <code>0f6d68bcb4fc2b0c13fa6f7dd74a4cce617a6eeb</code>
- overlay SHA-256
  <code>49a80169aac1ca29a5272fe764253001875da34859bdf91f7d011a43a7aa5c6b</code>
- image ID
  <code>sha256:cb7666cb44c214a7ffa3fcbbaf53ce8179017dd18e6b749f9bac2c0c633f5ef3</code>
- b12x 1.2.6, CUTLASS DSL 4.6.2, Torch 2.12.0+cu132, CUDA 13.2, and
  <code>CUTE_DSL_ARCH=sm_120a</code>
- NVIDIA RTX PRO 6000 Blackwell Workstation Edition, physical UUID
  <code>GPU-ca537fb1-a522-7429-e8c6-4795af22ab78</code>, Default compute mode

Every measured decode sample was P1 at a 16,365 MHz memory clock with throttle
mask <code>0x0</code>. All service arms completed healthy with zero restarts
and no OOM.

### Checkpoint correctness and kernel timing

The exact hydrated checkpoint gate used
<code>model.language_model.layers.0.mlp.down_proj</code>
(<code>K=17408</code>, <code>N=5120</code>) at rows 1, 4, 8, and 16. The
8.35 GB source shard matched its declared SHA-256. Every row passed two BF16
oracle scenarios with finite, nonzero, fully overwritten output. Minimum cosine
was 0.99999958, maximum relative L2 was 0.00039491, maximum absolute error was
0.00390625, and the top-8 set and order were exact. Each captured fused graph
contained one cooperative K6/MCG kernel and no separate rotation kernel.
Caller-owned tensor addresses remained fixed, eight replays per row allocated
no memory, and source payload hashes were unchanged.

The benchmark cycled all six route orders equally. Each warm median contains
240 single-replay CUDA-event samples after 60 warmups; each cold median contains
12 samples. A ratio greater than one means the fused route has lower latency
than the served ExLlamaV3 route.

| Rows | Warm fused ms | Warm served ms | Warm served/fused | Cold fused ms | Cold served ms | Cold served/fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.032416 | 0.042592 | 1.313919 | 0.032416 | 0.041360 | 1.275913 |
| 4 | 0.032608 | 0.042560 | 1.305201 | 0.032416 | 0.041216 | 1.271471 |
| 8 | 0.033440 | 0.042656 | 1.275598 | 0.034432 | 0.042336 | 1.229554 |
| 16 | 0.034432 | 0.042656 | 1.238848 | 0.034432 | 0.042656 | 1.238848 |

The b12x PR commits the full 1.09 MB result, all raw samples, GPU snapshots,
allocator snapshots, package/toolchain identity, and the twelve exact CUDA
graph DOT files under
<code>validation/trellis_decode/evidence/qwen38_down_bf16_8159533/</code>. The
result SHA-256 is
<code>74372b9860ebf1854e17130aa923037063c45e3386ed38158288f90f78b4f1ff</code>.

### Real-service reachability

An MTP-3 service with <code>FULL_DECODE_ONLY</code> graphs produced 243 native
BF16 plans: 240 target/shared shards and three MTP/draft shards. It then
produced 2,205 native BF16 runtime dispatch records. Observed rows were 1, 2,
3, 4, 5, 8, 12, and 16; every native input was BF16, and observed launch grids
were 120, 160, and 188. The service remained healthy with zero restarts and no
OOM, and the exact eight-token Paris sanity request passed.

This reachability run proves that the target and draft model paths actually use
the qualified route. It is not presented as an MTP throughput comparison.

### End-to-end serving

The primary serving result is a fixed-trajectory, no-MTP A-B-A bracket. Each
decode cell is the median of five 256-token samples after one warmup. Each
prefill arm uses two untimed and three timed distinct 2,051-token token-ID
prompts with one output token. Every prompt has a unique first cache block, and
every request must record exactly 2,051 prefix-cache queries, zero hits, 2,051
locally computed prompt and KV tokens, and one completed prefill request. Delta
compares the candidate with the mean of the two baseline medians, so a positive
delta means the candidate is faster.

| Workload | Baseline A1 | Native BF16 B1 | Baseline A2 | Candidate delta |
| --- | ---: | ---: | ---: | ---: |
| C1 decode (tok/s) | 55.521 | 58.914 | 55.413 | +6.21% |
| C4 decode (aggregate tok/s) | 204.042 | 215.196 | 203.808 | +5.53% |
| Uncached 2,051-token prefill (prompt tok/s, including one decode step) | 5,092.6 | 5,073.2 | 5,052.4 | +0.014% |

The feature targets the small-row decode route and is not selected for this
prefill shape. The +0.014% prefill point estimate is therefore treated as
neutral, not as a positive claim. This table supersedes an earlier invalid
approximately 17k tok/s row: that harness warmed and timed the same prompt with
prefix caching enabled, so it divided the full logical prompt length by a
cached-request wall time rather than measuring uncached prefill.

The exact commands, identities, raw samples, derived ratios, service state, GPU
state, checkpoint-gate summary, and artifact hashes are preserved in
[the compact qualification receipt](../receipts/b12x-native-bf16-pr243-2026-08-24.json).
The three raw A-B-A JSON files and compressed MTP-3 service log are in
[the raw receipt directory](../receipts/b12x-pr243-hotpath-raw/). The
repeatable serving harness is
[tools/b12x_native_bf16_e2e.py](../tools/b12x_native_bf16_e2e.py).

After qualification, the production service was restored to its original image
<code>sha256:921a46dfed1e4846f55c7507f8e1e4ea0278148ff54c926dd3b2246d72a0ed91</code>
with MTP-3 enabled and native BF16/reachability disabled. It was healthy with
zero restarts, no OOM, and a passing Paris sanity request.
