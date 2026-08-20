# Where prefill time actually goes, and what the ceiling is

Iteration 3, measured on one RTX PRO 6000 Blackwell (SM120, driver 595.58.03), GG r34
image, `malaiwah/Qwen3.8-27B-EXL3-K5K6` at `--max-model-len 8192`,
`--kv-cache-dtype fp8`, 2,048-token prefill, median of 3 runs.

## The matrix

Four configurations isolate the two candidate causes - the online-K6 attention overlay
and the MLP trellis kernel. "reconstruct" is the PR #316 dispatch
(`VLLM_EXL3_PREFILL_RECONSTRUCT_M=128`); "trellis" is the decode-shaped `exl3_gemm`
path that was used for every row count before it.

| # | attention | MLP kernel | PP 2k tok/s | PP 6k tok/s | weights GiB |
|---|---|---|---:|---:|---:|
| A | online K6 | trellis | 2,369 | 2,362 | 21.8 |
| B | online K6 | reconstruct | 5,050 | 5,146 | 21.8 |
| C | BF16 (no overlay) | trellis | 2,485 | 2,473 | 28.5 |
| D | BF16 (no overlay) | reconstruct | 5,618 | 5,728 | 28.5 |

Rows C and D are receipted in [`receipts/prefill-pp-mlp-trellis.json`](../receipts/prefill-pp-mlp-trellis.json)
and [`receipts/prefill-pp-overlay-off.json`](../receipts/prefill-pp-overlay-off.json).

Read the deltas along each axis:

- **MLP kernel: 2.13x (A->B) and 2.26x (C->D).** This is the whole story of prefill.
- **Attention representation: 1.05x (A->C) and 1.11x (B->D).** The online-K6 overlay is
  *not* the prefill bottleneck. It costs about a tenth of prefill and saves 6.7 GB.

That refutes the hypothesis I closed iteration 2 with ("the remaining 2-3x points at the
online-K6 overlay, 208 attention projections through a small-M decode kernel"). The
overlay is cheap; the MLP was expensive, and PR #316 already collected that win.

## Is the custom GEMM the next win? No.

The reconstruct path ends in `ext.hgemm`. If exllamav3's hand-written fp16 GEMM were
behind cuBLAS, replacing it would be free performance. Measured on the same device, same
shapes, fp16, `torch.mm` as the cuBLAS reference (`tools/gemm_cmp.py`):

| K | N | m | `ext.hgemm` | `torch.mm` | ratio |
|---:|---:|---:|---:|---:|---:|
| 5120 | 17408 | 256 | 0.154 ms | 0.158 ms | 0.98x |
| 5120 | 17408 | 2048 | 0.903 ms | 0.911 ms | 0.99x |
| 5120 | 17408 | 4096 | 1.760 ms | 1.702 ms | 1.03x |
| 17408 | 5120 | 1024 | 0.465 ms | 0.506 ms | 0.92x |
| 17408 | 5120 | 2048 | 0.968 ms | 0.919 ms | 1.05x |
| 5120 | 32768 | 2048 | 1.585 ms | 1.613 ms | 0.98x |

`ext.hgemm` is at cuBLAS parity across the range (0.92-1.06x). There is no free win
here; the GEMM itself is not the problem.

## The arithmetic ceiling for a trellis format

At m=2048 a `gate_proj` shape costs 1.20 ms end to end (measured, unfused path) of which
the GEMM is 0.90 ms; the remaining ~0.30 ms is Hadamard + reconstruct. Per layer that is
gate + up + down = 3.69 ms, so 64 layers = 236 ms, i.e. an MLP-only ceiling of
**8.7k tok/s**, and we measure 5.6k with attention, GDN, KV and sampling on top.

Even a *perfect* fused kernel that made reconstruct free would give 64 x 2.70 ms =
173 ms, or 11.8k tok/s MLP-only—roughly 7-8k end to end under the measured
non-MLP overhead. Official FP8 serves 10,667 tok/s and NVFP4 14,528 tok/s on
this box; neither pays the EXL3 reconstruct transform.

**Conclusion, bounded to the measured implementation:** tuning the existing
unfused reconstruct-plus-hgemm path cannot reach FP8 prefill parity. Closing the
gap requires a Marlin-style fused dequantization/GEMM kernel—a new kernel, not a
dispatch change. Two things remain worth doing and are cheap: amortise the reconstruct
over larger prefill chunks (config-only), and keep the reconstruct scratch resident so
repeated chunks skip the allocation.

The goal in [docs/25](25-goal-pareto-dominate-fp8.md) is therefore re-scoped, with the
evidence above as the justification: **dominate FP8 on fidelity, memory, decode and
speculative decode; document prefill as a measured, structural deficit** rather than
leave it open as if tuning would close it.
