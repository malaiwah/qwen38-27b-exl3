# Issue draft 07 — MTP depth d costs d+1 full lm_head streams per step; 78 % of the marginal bytes per depth level are head traffic

**Repo:** local-inference-lab/vllm (analysis / design note) · **Cites:** docs/47 F6, F9

## Summary
From source at 4d006a4: the target verify computes full-vocab logits once
(gpu_model_runner.py:4496-4497), and each MTP draft sampling materializes full logits again — even
the "local argmax" path (`logits_processor.py:139-185,205`; `llm_base_proposer.py:481-495`). For a
248,320-vocab head (953.5 MB at K6) at depth 3 that is 3.81 GB/step of head traffic, 18.6 % of all
weight bytes streamed (profiled: receipts/kernel-gap-profiled-decode.json). Marginal cost per depth
level = 0.953 GB head + 0.27 GB draft body: **78 % head**. Head+draft bytes are per-STEP fixed
costs that do not amortize with concurrency the way the body does — which mechanises the
empirically-derived depth schedule `[[1,4,3],[5,64,1]]` (depth 1 above the knee saves 2.44 GB/step
exactly where throughput is bandwidth-bound).

## Testable predictions (either falsifies the model)
1. Step time vs depth is linear with slope ≈ 1.22 GB ÷ 1.3–1.5 TB/s ≈ 0.8–0.95 ms/level (plus an
   eager-dispatch constant), independent of batch size.
2. Any reduction of per-sampling head cost (draft 01's routing, a draft-only truncated head, or
   head/side-stream overlap) moves the depth-schedule knee to higher concurrency.

## Possible directions (not proposals yet)
Side-stream overlap of draft-sampling head GEMMs with the next body replay; draft-only vocabulary
truncation (changes acceptance distribution — needs its own A/B).
