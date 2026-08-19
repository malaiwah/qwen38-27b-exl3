# What of the Qwen3.8-27B work transfers to GLM-5.2 R7 on Gilded Gnosis

**Date:** 2026-08-19
**Target:** `local-inference-lab/rtx6kpro` → `models/glm5.2_v20.md` (r34, GLM-5.2 R7
mixed-K3/K4/K5 EXL3, TP4/DCP1, 4× 96 GB Blackwell, MTP3, online K6).
**Why any of it transfers:** GLM-5.2 r34 and our Qwen3.8-27B work run the **same
image** — `voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34`,
digest `sha256:820181fb…`, i.e. identical vLLM `4d006a4`, b12x `cd3ce19`, FlashInfer
`1ac6942`. Same SM120 generation. So some of this is not analogy, it is the same code.

## Tier 1 — same code path, directly actionable

### 1. The B12X W4A16 scratch contract is on GLM's routed-expert hot path

Our [b12x #235](https://github.com/local-inference-lab/b12x/issues/235) documents that
`packed_gemm_scratch_elements` returns
`min(size_n × route_slots, sms × 4 × moe_block_size × 256)` and that the **right-hand
cap is shape-independent**. That function lives in
`b12x/moe/_shared/kernels/w4a16/host.py` and is consumed by `mixed_trellis.py`,
`kernel.py`, `ep_moe/_impl.py` and `fused_moe/_impl.py` — `mixed_trellis` being exactly
GLM's *"one mixed-Trellis launch plan for decode and prefill"*. `plan_w4a16_buffers`
sizes both `fc1_c_tmp_elements` and `fc2_c_tmp_elements` through it (host.py:314, 320).

On our dense path, caching that buffer per shape cost tens of GiB and OOM'd the engine,
because a *constant* was being multiplied by the number of shapes (ledger L13).

**Testable consequence for GLM.** `VLLM_EXL3_PREFILL_CAPACITY` defaults to
`max_batched_tokens` and narrows the arena (exl3.py:727, 4240); `route_slots` grows with
it. The r34 page documents that dropping capacity 2048 → 1024 "recovers persistent
workspace for KV cache" at a cost of **7-12 % prefill throughput**. But if

```
fc1_cols × route_slots(2048)  ≥  sms × 4 × block_size_m × 256
```

then `c_tmp` is already **at the cap at 2048 rows** and halving capacity returns
**zero** scratch bytes — the 7-12 % is being paid for savings that come only from the
other arena buffers. Worth checking before treating that dial as a memory lever:
print `plan_w4a16_buffers(...).fc1_c_tmp_elements` at capacity 2048 and 1024 and see
whether the number actually moves.

### 2. Speculative decoding does wasted work on ≤1-token requests

[local-inference-lab/vllm #439](https://github.com/local-inference-lab/vllm/pull/439)
skips speculation when every scheduled request needs at most one more token. GLM r34
runs MTP3 at `MAX_NUM_SEQS=8` with 65.44 % acceptance; in mixed traffic some scheduled
requests are on their last token, and each one currently pays three draft forwards for
nothing. One file, +13 lines, DCO-signed.

### 3. Route B12X by row count, not once for both phases

We measured that B12X W4A16 **wins prefill and loses decode** against the fused
`exl3_gemm` kernel on identical weights — and that the fused path also *drafts better*:
TG-essay 93.1 vs 90.0 and MTP acceptance **0.304 vs 0.281**, at unchanged prefill
(`receipts/m-dispatch-k6-2026-08-19.md`). We now dispatch on `m`
(`VLLM_EXL3_B12X_MIN_M=128`). The r34 page states GLM uses *one* mixed-Trellis plan for
both decode and prefill, so if the same asymmetry exists for the MoE kernels there is
free decode throughput available. Cheap to test: force the non-B12X path at decode row
counts only and compare acceptance.

## Tier 2 — general vLLM bugs we filed, which bite harder at GMU 0.98

### 4. A forward-pass OOM kills the whole EngineCore

[vllm-project/vllm #52871](https://github.com/vllm-project/vllm/issues/52871). GLM r34
runs at `GPU_MEMORY_UTILIZATION=0.98` and the page records that 0.97 exposes only
47,552 KV tokens versus 75,072 at 0.98 — i.e. the profile is deliberately operating on a
thin margin. On our card, a single activation spike past the profiled peak did not
degrade the request, it **terminated the engine** and required a restart. The higher the
utilisation, the more this matters.

### 5. `max_num_batched_tokens` also sizes the CUDA-graph pool

[vllm-project/vllm #52872](https://github.com/vllm-project/vllm/issues/52872). The
mamba/GDN half of that issue does **not** apply to GLM (MLA, not linear attention), but
the second half does: `mnbt` silently drives graph-pool size as well as activation peak.
GLM sets `MAX_BATCHED_TOKENS=2048` with a graph cap of 32 — so the pool is a real term in
that 0.98 budget, and lowering `mnbt` frees KV as well as activation. On our stack
8192 → 3072 *freed* 0.93 GiB.

## Tier 3 — measured relationships that transfer as predictions

### 6. Draft acceptance tracks weight fidelity — quantising harder costs throughput twice

Four independent measurements on identical runner/depth/scheduler, only weights differing:

| weights | MTP acceptance | decode |
|---|---|---|
| trellis (K5/K6) | **1.000** | 208.3 tok/s |
| trellis + `self_attn` FP4 | 0.967 | 199.5 |
| all-FP4 | 0.930 | 187.4 |

A less faithful target disagrees with its own draft head more often, and every rejected
token is a wasted verify slot. **Prediction for GLM:** its 65.44 % acceptance is partly a
*fidelity* number, not only a draft-quality number. Raising routed-expert bitrate (K3 →
K4/K5 on the most sensitive experts) should raise acceptance and therefore decode
throughput super-linearly — cheaper than tuning the draft head.

### 7. Additivity of quantisation error holds for MLP/experts, fails for attention

We validated an additive KLD model on held-out configurations: MLP groups came in at
−1.9 % and −6.0 % of prediction, but `self_attn` **under-predicted by +46 %**
(`receipts/selfattn-fp4-additivity-failure-2026-08-19.md`). Mechanism: attention error
propagates through the KV cache and is re-read at every later position, so it compounds
over the scored window instead of staying per-position.

**Direct relevance:** the r34 quality section states its paired test used **FP8 KV** while
the serving recipe uses **NVFP4 DS-MLA**, and correctly flags that as a limitation. Our
result says that gap is likely to be *larger than a per-position estimate suggests*,
because KV-format error is exactly the compounding kind. It also means per-expert error
budgets can be added, but any KV or attention change must be measured end-to-end.

### 8. Check the machine before trusting any benchmark

Our card was capped at **400 W of 600 W** by two independent persistent sources (a
`lactd` config *and* an `ExecStartPre` in the systemd unit). Dense-GEMM prefill was
throttled **25.3 %**; HBM bandwidth was unaffected. The tell was a spec cross-check
showing bandwidth *above* nameplate (a VRAM overclock) and compute *below* it
(`receipts/power-limit-2026-08-19.md`). Four cards in one chassis are a prime candidate
for power or thermal capping, and none of it shows up in an application metric.

Useful nuance for GLM specifically: in our A/B the **trellis and FP6 paths were
power-insensitive** (+0.0 % and +2.0 %) while only the dense-FP4 path gained. GLM R7 is
trellis end-to-end, so its prefill may legitimately be power-insensitive — but that is
a measurement, not an assumption.

### 9. Harness details worth copying

`tools/bench-profile.sh` / `tools/verify-profile.sh`: n≥3 boots with CIs, **MTP
acceptance recorded per prompt** (it is strongly prompt-dependent — 0.930 on a short
prompt vs 0.298 on a long generation for the same weights, so a single acceptance figure
beside a single throughput figure is ambiguous), a gated long-context probe, and a
boot-time metric. Two traps we hit and fixed: a long-context check that inflated the
prompt 20 % past target and so reported a false capability failure, and 369 of 374
selftest lines reporting `cos=0.000000` because vLLM's profiling run feeds all-zero
activations.

## What does **not** transfer

- All GDN / linear-attention work (the `chunk_o` prefill OOM, the 48-layer hybrid
  profiling gap). GLM-5.2 uses MLA.
- Our NVFP4/MXFP6 dense conversion and the `FP6_LAYER_RANGE` dial: R7 explicitly
  supports online K6 only, and its experts stay native trellis. It may apply to the
  separate GLM NVFP4 profile, which has its own qualification record.
- Every absolute number and the 2.87 GiB memory-wall arithmetic: single 31.4 GiB card
  versus 4× 96 GB with TP4.
- Our three serving profiles as such. The transferable object is the *method* — measure
  a small number of named, gated profiles and publish the tradeoff — not the profiles.

## Honest summary

Of the work: **three items are same-code actionable** (b12x #235 scratch contract on the
mixed-Trellis path, vllm #439 speculation skip, row-count dispatch), **two are filed
upstream bugs that GLM's 0.98-utilisation profile is more exposed to than ours** (vllm
#52871, #52872), and **four are transferable measured relationships** (acceptance↔fidelity
coupling, attention-error compounding vs additive expert error, machine-state
verification, harness design). The GDN, dense-FP4/FP6 and memory-budget work does not
transfer.

The single highest-value item is #1: it is a specific, cheap, falsifiable check against a
tradeoff their own page documents as costing 7-12 % prefill.
