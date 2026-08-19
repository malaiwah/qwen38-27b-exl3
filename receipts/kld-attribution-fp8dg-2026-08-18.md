# Per-layer KLD attribution + FP8-DG prefill path (2026-08-18)

## Per-layer KLD attribution (512-context v5 shard-0 suite, int6 embeds)
Single-group FP4 runs (everything else trellis):
- gate_up only:  0.028875
- down only:     0.017382
- GDN only:      0.019835
- flagship (all three): 0.056732 ; trellis floor (prior): 0.0027

Additive decomposition (system closes to 1e-5):
| component        | KLD    | share |
|------------------|--------|-------|
| gate_up FP4      | 0.0242 | 43%   |
| GDN in/out FP4   | 0.0152 | 27%   |
| down FP4         | 0.0127 | 22%   |
| int6 embeds      | 0.0020 | 3.5%  |
| trellis floor    | 0.0027 | 4.8%  |
(t+e = 0.00468 derived; attrib-none run OOMed at load — all-trellis weights
+ embed conversion exceed 31.4GB at util 0.85 — but the 4-equation system is
fully determined without it.)

## FP6 hybrid A/Bs (both REJECTED)
- gate_up->FP6 (+fp8dg): PP=5468 (<6000). FP6 streams 6 bits on the largest
  matrices.
- down->FP6 (+fp8dg): PP=6156 ok, TG=81 COLLAPSED (FP6 decode pathology on
  down_proj N=5120 shape with packed streaming).

## FP8-DG prefill path (LANDED, env-gated)
`ext.reconstruct_fp8dg_nt` (shipped-unused) decodes a trellis shard straight
to DeepGEMM NT blockwise FP8 (q_nt [N,K] e4m3 + [N/128,K/128] fp32 scales,
reconstruct.cu:266-322). Wiring (exl3.py):
- activations: y = ((x*suh)@Had_K) @ W_raw @ Had_N * svh — rotations moved to
  activations (two 128-block bmms), GEMM via deep_gemm.fp8_gemm_nt.
- SM120 REQUIRES UE8M0: raw fp32 scales -> NaN output. Fix (probe-verified):
  `deepgemm_post_process_fp8_weight_block(wq, ws, (128,128), use_e8m0=True)`
  + `per_token_group_quant_fp8(..., use_ue8m0=True)` + explicit
  `is_deep_gemm_e8m0_used=True` kwarg. cos=0.9995 vs trellis per layer.
- Envs: VLLM_EXL3_FP8DG_PREFILL_M (min M, 0=off), VLLM_EXL3_FP8DG_CACHE
  (cache post-processed fp8 weights), VLLM_EXL3_FP8DG_SELFTEST.

Measured (8K, FP8 KV, MTP6, graphs):
- attn-only, CACHED (~2GB): PP 6466 -> 7062 (+9.2%), KLD 0.05762 (+0.0009).
- attn-only, uncached: PP 5409 (-16%) — per-call reconstruct+UE8M0 requant
  (4.75B/elem traffic) swamps the GEMM gain. Cache is mandatory.
- ALL layers, uncached: PP 1235 (worse than native 1591) — dead on MLP sizes.
- Cache cost: ~2GB -> flagship context 238.4k -> ~185k if enabled.

## All-trellis profile rediscovered (with today's full stack)
All-trellis (no FP4/FP6), MTP6 + graphs + FP8 KV + int6 embeds:
- TG(fox bench) = 194.2 (acceptance 100% on repetitive text — exact-trellis
  draft aligns perfectly), TG(essay) = 83.5 vs flagship 75-76 (+10%).
- PP = 1235-1591 (trellis prefill still the bottleneck).
- KLD ~= 0.0047 est (0.0027 floor + 0.0020 int6 embeds).
Trellis W4A16 decode at M=1 BEATS the FP4 W4A4 decode path (64-row tile
waste + act-quant overhead) — FP4's value is prefill-only.
M-based dispatch (trellis decode + FP4 prefill copies) remains memory-
infeasible (26+GB weights, prior receipt).

## Serving frontier (8K numbers; all env-selectable from one checkpoint)
| profile | PP | TG(fox/essay) | KLD | max ctx @0.93 |
|---|---|---|---|---|
| flagship FP4 MLP+GDN (DEFAULT) | 6466 | 161 / 75 | 0.0567 | 238.4k |
| flagship + fp8dg-cached attn   | 7062 | ~161 / ~75 | 0.0576 | ~185k |
| all-FP6 W6A8 quality           | 4671 | 154 / ? | 0.0107 | ~? |
| all-trellis fidelity           | 1591 | 194 / 83.5 | ~0.0047 | ~205k |
