# Final verification matrix — 2026-08-18 session close

## Flagship serving config (deployed, verified end-to-end)
| criterion | measured | target | verdict |
|---|---|---|---|
| PP (2051-tok median-of-5) | **6362 tok/s** | ≥6000 | **PASS** |
| TG (200-tok median-of-3) | **155.2 tok/s** | ≥150 | **PASS** |
| KLD (512-ctx fidelity replay) | **0.056732** (top-1 90.3%) | low | 10.4% better than all-FP4 baseline 0.0633 |
| Context | **238,400** (91% of native 262,144) | ~256k | FP8 KV; 256k+ closer identified (vision-tower int8, ~0.5GB) |
| MTP | **=6**, spec decode active | required | PASS |
| Vision | 'Red, Blue' exact on fixture, with MTP | required | PASS (small imgs; 8MP needs headroom profile) |

Config: MLP+GDN → NVFP4 W4A4, 16 full-attn → trellis W4A16, embeddings →
int6 per-row online (VLLM_EXL3_EMBED_ONLINE_BITS=6, −1.48GiB), lm_head K6
via exl3_gemm (B12X_N_RANGE), FP8 KV, FULL_DECODE_ONLY graphs, util 0.93,
MTP-skip-for-max_tokens≤1 scheduler patch.

## Quality profile (measured, switchable)
All-FP6 W6A8 + B12X_PACKED_B_MIN_N=1024: KLD **0.0107** (5.3× better),
top-1 95.6%, TG 154.0 PASS, PP 4671 (misses 6000). One env/routing flip.

## Upstream (all fixes published)
- PR #436 (embed online quant) + integration-fixes commit fd500ef + comment
- PR #437 (scheduler: skip spec decode for max_tokens≤1) — +2.5% PP measured
- Issue #435 (trellis row-gather reconstruct ask for true K6/K8 embeddings)

## Measured negatives (documented, closed)
- Per-row FP4 activation scales: KLD 0.0563 vs 0.0567 at −22% PP → dropped
- GDN cutedsl prefill on SM120: tcgen05/TMEM hardware — impossible
- All-attn-trellis: PP 4646 / TG 139.1 — misses both

## Open research backlog (evidence-linked, todo-tracked)
W4A8 dense kernel (e2m1 b_fmt), FP4 draft-only lm_head (+5-7% TG),
FP4 16-row decode tile (+5-15% TG), fused prefill act-quant (+3-6% PP),
per-layer KLD attribution, 256k closer via vision-tower int8.
