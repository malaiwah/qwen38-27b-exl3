
## Serving profiles on one RTX 5090 — pick by workload

Three measured serving configurations ship with
[`run-qwen38-27b.sh`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/patches/run-qwen38-27b.sh),
selected with a single environment variable. All numbers below are measured on a
physical RTX 5090 (32 GB, 600 W) with the Gilded Gnosis vLLM fork, `n=3` boots per
profile, MTP acceptance reported beside every decode number, and KLD measured on
512 held-out contexts against the BF16 reference.

![Speed vs fidelity](assets/profiles-tradeoff.png)

![What each profile delivers](assets/profiles-throughput.png)

### The three profiles

| | `PROFILE=fidelity` | `PROFILE=balanced` | `PROFILE=throughput` |
|---|---|---|---|
| **weights** | all-trellis (K5/K6 as shipped) | trellis + `mlp.gate_up_proj` MXFP6 | all-FP4 (NVFP4) |
| **prefill** (2051-tok) | 2,987.7 ± 4.4 tok/s | 3,925.2 ± 13.1 tok/s | **9,638.9 ± 18.3 tok/s** |
| **decode**, short prompt | **228.3 ± 0.4** (acc 1.000) | 215.6 ± 0.2 (acc 1.000) | 187.4 ± 0.6 (acc 0.930) |
| **decode**, 500-tok generation | **104.1 ± 0.1** (acc 0.304) | 103.7 ± 0.1 (acc 0.324) | 94.3 ± 0.0 (acc 0.298) |
| **KLD** vs BF16 | **0.003405** [0.003166, 0.003672] | 0.005672 [0.005302, 0.006087] | 0.063759 |
| **KLD p99** | **0.034889** | 0.059908 | 0.7010 |
| **max context** | 238,400 | 199,104 | **249,600** |
| **weights resident** | 18.8 GiB | 21.2 GiB | **15.9 GiB** |
| vision + MTP | pass | pass | pass |
| **criteria met** | 5/6 (fails PP ≥ 7000) | 5/6 (fails ctx) | 4/6 (fails KLD) |
| **use it when** | fidelity is the product: RAG over long documents, agents, eval harnesses, anything where the quant must not change answers | mixed traffic: you want 1.3x the prefill of `fidelity` with strong decode (fox 215.6, essay 103.7 at MTP acc 0.324), and 199k context is enough | prompt-heavy batch work: reranking, classification, bulk summarisation, where 4-bit drift is acceptable |
| **the catch** | prefill is 3.2x slower than `throughput` | 39,296 fewer context tokens | 19x the divergence of `fidelity`; measurably worse draft agreement (0.930 vs 1.000) |

```bash
PROFILE=fidelity   ./run-qwen38-27b.sh   # most faithful, full context
PROFILE=balanced   ./run-qwen38-27b.sh   # strong decode, 1.3x prefill, 199k ctx
PROFILE=throughput ./run-qwen38-27b.sh   # fastest prefill  (default)
```

### Reproducible container image

A baked serving image is on Docker Hub at
`docker.io/malaiwah/qwen38-27b-exl3-gg:r34-p2-41a5d16` (also `:latest`), built
from [`docker/Containerfile`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/docker/Containerfile).
All 10 patches are baked in on top of the digest-pinned Gilded Gnosis r34 base
(`voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34`,
sha256 `820181fbb…df20592b`), so `podman run` plus a Hugging Face cache mount
is sufficient — the repo is not required at runtime. Gated **9/9 mount-free**
(`NO_PATCH_MOUNTS=1`): PP 2933.0, fox 229.1, essay 103.7, ctx 238,400, 200k
prompt OK, vision OK, MTP acceptance\_fox 1.000. See the repo README for a
copy-pasteable `podman run` example with the full flag set.

### How the fidelity compares to other quantisations

![Fidelity vs other quants](assets/fidelity-vs-quants.png)

On the same 512-context shard-0 protocol, `PROFILE=fidelity` measures
**0.003405** versus official `Qwen/Qwen3.8-27B-FP8` at **0.005197** and
Unsloth NVFP4 at **0.030115**. `PROFILE=balanced` measures 0.005672.
`PROFILE=throughput` trades fidelity for speed at 0.063759. These comparisons
use identical contexts and the shared BF16 head; do not substitute the
different 10.48M-position cumulative values into their ratios.

Long-context needle retrieval on the `fidelity` profile (fixed harness, single
needle per context): **8/8 at 2k, 8/8 at 100k, 8/8 at 195k — 24/24**. This is
an easy proxy — finding a planted needle does not establish unimpaired
long-context reasoning, only that the attention window and KV cache are intact
to 195k.

### Other combinations that were measured

Configurations that are reachable with the same launcher but are **not** shipped as
profiles, with the reason:

| configuration | prefill | decode (fox / essay) | KLD | max ctx | verdict |
|---|---|---|---|---|---|
| all-FP6 (`FP6_LAYERS` = all four groups) | 4,742 | 187.3 / 87.9 | 0.010699 | 99,000 | passes the 0.012 bar but costs 60 % of context; use only if ~99k is enough |
| `fidelity` + 13 gate_up layers in FP6 | 2,145 | 207.9 / 92.2 | ~0.0039 *(predicted)* | 238,400 | +9 % prefill at full context, but boots with only a **40 MiB** memory margin — documented, not shipped |
| `self_attn` in FP4 | 2,043 | 199.5 / 91.0 | 0.011534 [0.010944, **0.012203**] | 238,400 | rejected: the CI crosses the 0.012 bar, and p99 triples to 0.1159 |
| FP8-DeepGEMM prefill, cached | 7,062 | — | 0.05762 | 238,400 | superseded by all-FP4, which is faster and simpler |
| `B12X_ANY_BITS=1` (K5 through B12X) | 2,458 | — | — | 238,400 | **known broken**: +51 % prefill but the model emits garbage. B12X's bits=5 GEMM is correct in isolation (cos 1.000000); the fault is in the integration. Guarded with a warning |
| FP4 draft head | — | 157.6 / 76.1 | — | — | negative vs 160.6 / 75.2 with it off |
| per-row FP4 global scales | — | — | no measurable gain | — | disabled |

### Knob reference — measured effect of each lever

| knob | measured effect | note |
|---|---|---|
| GPU power limit 400 W → 600 W | FP4 prefill **+25.3 %** (7,695 → 9,639); trellis **+0.0 %**, FP6 **+2.0 %** | only the FP4 path is power-limited. Check `nvidia-smi -q -d POWER` before benchmarking — a vendor tool or systemd unit may be capping the card |
| `VLLM_USE_V2_MODEL_RUNNER=1` | decode **+32 %** (essay 70.9 → 93.5, fox 141.9 → 185.6) | the MTP draft loop only gets CUDA graphs in the V2 runner; on V1 it runs eager and silently costs a third of decode |
| `--max-num-batched-tokens 8192 → 3072` | **fixes an engine-fatal OOM** on any prompt over ~4k tokens, and *frees* 0.93 GiB of KV | correctness fix, not tuning |
| `VLLM_EXL3_SKIP_TRELLIS_PREP=0` (B12X for K6) | `fidelity` prefill **+50.8 %** (1,081 → 1,630) at unchanged KLD | verified fidelity-neutral over 512 contexts (0.003412 → 0.003407) |
| `VLLM_EXL3_PREFILL_RECONSTRUCT_M=1` + `MAX_MB=512` + `CACHE=0` | **+14.1 %** (1,630 → 1,858) | `MAX_MB` keeps the 2.37 GiB lm_head off the path; `CACHE=0` stops the FP16 cache being filled during vLLM's profiling pass, which otherwise leaves zero KV |
| `VLLM_EXL3_FOLD_FP32_BUDGET_MB=48` | **+5.9 %** (1,858 → 1,967), **bit-identical** output | measured optimum; 64 MB and above fall off a cliff (L2 residency). Bigger is *worse* |
| `VLLM_EXL3_B12X_MIN_M=128` | essay **+3.4 %**, MTP acceptance **+8.2 %** (0.281 → 0.304), prefill unchanged | B12X wins prefill, the fused kernel wins decode; route by row count |
| `VLLM_EXL3_FP6_LAYER_RANGE=lo-hi` | ~**0.029 GiB** of KV per converted layer | turns precision into a continuous memory dial instead of all-or-nothing |
| MTP depth 4 / 6 / 8 / 10 | flat on the tested short prompt: 185.3 / 185.0 / 185.2 / 184.8 | no general depth-independence claim |

### One measured relationship worth knowing

Weight fidelity and decode speed are **coupled through speculative decoding**.
Same runner, same draft depth, same scheduler — only the weight format differs:

| weights | MTP acceptance (short prompt) | decode |
|---|---|---|
| trellis (K5/K6) | **1.000** | **228.3** tok/s |
| trellis + `self_attn` FP4 | 0.967 | 199.5 tok/s |
| all-FP4 | 0.930 | 187.4 tok/s |

Lower-fidelity weight formats showed lower draft acceptance and decode in this
controlled profile matrix. That association is consistent with rejected drafts
costing verify slots; the three-format comparison does not isolate causality.
Quantising harder can therefore cost throughput as well as fidelity on this
stack.

### Chat-template and client contract

- The repo ships the official Qwen3.8-27B template in both
  `chat_template.jinja` and `tokenizer_config.json`; Transformers 5.15 gives the
  standalone file precedence. Override with `--chat-template`, not by editing
  only the tokenizer config.
- The template accepts `reasoning_effort` values `xhigh` (default), `medium`
  and `low`. Echo assistant `reasoning_content` into subsequent history turns
  to preserve the reusable prefix.
- Only a leading system message is accepted. Merge later system instructions
  into the first system message before rendering history.
- Tool-call argument values use an unescaped XML wire format. Reject literal
  `</parameter>` and `<parameter=` delimiters before replaying assistant
  history, and treat replayed calls and tool output as untrusted data.
