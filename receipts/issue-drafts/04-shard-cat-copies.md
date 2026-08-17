# Issue draft 04 — per-shard apply + torch.cat moves 18–26 GB/chunk of pure copies at prefill; hgemm already supports strided C

**Repo:** local-inference-lab/vllm · **Cites:** docs/47 F5.3, plan P2.2

## Summary
`Exl3LinearMethod.apply` runs `_apply_one` per shard and concatenates (exl3.py:2710-2726). At a
2048-token prefill chunk the gate|up cat alone re-reads and re-writes 285 MB × 64 layers ≈ 18 GB;
q/k/v adds ~1.9 GB plus two N=1024 GEMMs at poor cuBLAS shapes; GDN qkvz/z ≈ 6 GB. Meanwhile the
extension's `hgemm` already accepts a strided-C output (ldc = `c.stride(-2)`, columns contiguous —
hgemm.cu:41-51,77): each shard's GEMM could write directly into its slice of the merged
destination, deleting the cat entirely.

## Proposed fix
Out-variant custom op (`_exl3_gemm_out`, same opacity pattern as `_b12x_trellis_linear_out`,
exl3.py:1258-1306) writing into caller-provided views. Sub-task: `had_r_128` needs a row-stride
argument for the svh transform on strided views (shard widths are 128-multiples so groups stay
aligned), or the svh hadamard keeps a contiguous temp (halves the win, still positive). Estimated
+5–8 % prefill.

## What would falsify this
A prefill profile showing the cat kernels are already overlapped/hidden behind GEMM tails (they are
not in the decode trace — `CatArrayBatchedCopy` is visible — but a prefill-window nsys would be the
direct evidence), or measured strided-C cuBLAS efficiency loss exceeding the copy savings.
