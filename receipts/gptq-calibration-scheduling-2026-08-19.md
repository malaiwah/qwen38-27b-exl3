# GPTQ calibration cost scales with rows — and that is itself evidence

**Date:** 2026-08-19 (goal session).

## What happened

Attempt 1 (`--no-concatenate --samples 512 --seq-len 2048`) was **cancelled at
2 h 01 m**, ~22% through, and its GPU time redirected.

The baseline GPTQ run (concatenated, 42 sequences) took **45 m 07 s** for all 482
subgraphs. Unconcatenated 512 sequences is **~12x the calibration rows**, and GPTQ
per-module cost is dominated by Hessian accumulation, which is linear in rows.
Projected wall time therefore **~9 h**, consistent with the observed 2 h for ~22%.

## The scheduling call

Four calibration attempts at that cost is 24–36 h of exclusive GPU. Meanwhile five
queued items were starving, each cheaper and at least as valuable:

| queued item | cost | why it outranks a 9 h calibration run |
|---|---|---|
| depth 3-arm calibration (FP6 ranges 0-12 / 26-38 / 51-63) | ~1 h | answers a live question: is the depth term real, and is it monotone or U-shaped (llama.cpp/exllamav3 both use U) |
| W4A8 NVVM bisect | minutes/cycle, cap 10 | isolated to one construct; capped |
| MTP+vision merge + gate | ~30 min | unlocks two promotion-gate criteria (acceptance, vision) |

Sunk cost is not a reason to continue a run that is 22% done and blocking work
worth more per GPU-minute.

## The measurement hiding in the cancellation

This is not merely a scheduling note. **The 45-minute baseline was fast precisely
because its Hessians were starved.** Calibration cost scaling ~12x with rows
confirms the mechanism behind the starvation hypothesis — the baseline saw 42
sequences (~86k tokens) where a healthy GPTQ run wants hundreds — independently of
any KLD outcome. The hypothesis that starvation explains GPTQ-losing-to-RTN gains
support from the timing alone; what remains untested is whether fixing it is
*sufficient*.

## Revised plan

1. Cheap queue first (above), all measured and committed.
2. **One** overnight calibration attempt at a middle point:
   `--no-concatenate --samples 160` ≈ 3.8x baseline rows, ~2.5–3 h — enough to move
   the needle if row count is the binding constraint, cheap enough to leave GPU for
   the rest.
3. If 3.8x rows moves KLD materially toward RTN's 0.022121, row count is confirmed
   as the lever and a longer run is justified on evidence. If it does not, the
   remaining suspects (GDN `cache=None` calibration flows, actorder, dampening) are
   cheaper to test than more rows, and the attempt budget goes there.
