# vLLM-GG issue drafts (docs/47 findings) — PREPARED, NOT FILED

Filing on `local-inference-lab/vllm` requires the owner's approval (Main holds that boundary).
Each draft below is copy-paste ready: title, body, evidence receipts, patch path where one exists,
and a falsification line. Numbering matches the IRC proposal of 2026-08-17 and stays stable so
drafts ↔ docs/47 sections ↔ eventual issue numbers remain traceable.

| draft | title (short) | docs/47 | patch ready |
|---|---|---|---|
| 01 | b12x K6 gate routes lm_head + k/v to a slower path at decode m | F3, F7 | yes |
| 02 | in_proj_ba unquantized 5120×96 GEMM at 33 GB/s (5.2 % of decode GPU time) | F6, P2.1 | probe first |
| 03 | VLLM_EXL3_PREFILL_FP8=1 silently no-ops without reconstruct_fp8_slice | F5.6, P1.3 | trivial |
| 04 | per-shard apply + torch.cat costs 18–26 GB/chunk at prefill | F5.3, P2.2 | design in plan |
| 05 | reconstruct scratch single-buffered: reconstruct[i+1] never overlaps hgemm[i] | F5.2, P2.3 | design in plan |
| 06 | prefill chunk 2048→6144: 2.67× less redundant reconstruct, +13–15 % measured on linears | F5.1, F8, P1.2 | no code |
| 07 | MTP depth costs d+1 full lm_head streams/step; 78 % of marginal depth bytes are head | F6, F9 | analysis |
