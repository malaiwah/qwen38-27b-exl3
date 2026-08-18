# MTP=6 lm_head stream traffic quantification (mixed 16-attn-trellis config)

Basis: docs/47 F6/F9 head-streaming model + measured TG on this card (2026-08-18).

## Per-step weight traffic at MTP=6 (1 verify + 6 draft samplings)
| component | bytes/step | source |
|---|--:|---|
| FP4 body (MLP 9.6 + GDN 3.2 GB) | 12.8 GB | streamed once (verify m=7 in one pass) |
| Full-attn trellis (16 layers) | 1.0 GB | K6 codes + suh/svh |
| Draft body (K4 MTP layer × 6 passes) | ~1.8 GB | 0.3 GB × 6 |
| **lm_head K6 909.4 MiB × 7 streams** | **6.37 GB** | 1 verify + 6 draft argmax (each materializes full 248320-vocab logits, logits_processor.py) |
| **Total** | **~22.0 GB** | |

Head share: **29% of all bytes streamed per step.**

## Measured operating point (193.6k config, FP8 KV)
- TG=159.7 tok/s, acceptance ~3.87 tok/step → step ≈ 24.2 ms
- Effective streaming: 22.0 GB / 24.2 ms = 908 GB/s = **62% of the 1462 GB/s achievable ceiling** (docs/47 F2)

## Ceiling if 6 draft head streams eliminated (truncated draft vocab or side-stream overlap)
- Saves 5.46 GB/step (25%) → step ≈ 18.2 ms → **TG ≈ 213 (+33%)**
- This is the single largest remaining TG lever (matches docs/47 F10 item 4, scaled from MTP-3 to MTP-6).

## BF16-KV comparison (8K config)
- TG=196.1, same traffic model → step 19.7 ms → 1117 GB/s = 76% of achievable.
- FP8-KV acceptance drop costs ~0.4 accepted tok/step at MTP=6.
