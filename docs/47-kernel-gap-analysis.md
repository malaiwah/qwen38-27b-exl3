# 47. Kernel gap analysis: where the missing 35–45 % of decode bandwidth and 30 % of prefill compute actually go

**Question.** Decode measures 129.5–135.8 tok/s single-stream (hydrated K5/K6, MTP-3, fp8 KV, graph
decode), which back-solves to ~55–65 % of the 1.792 TB/s spec memory bandwidth as effective
weight-streaming rate. Prefill measures 3.3–3.7k tok/s vs 5.0–5.2k for dense EXL3 on a 188-SM card.
docs/41 and docs/46 §14–21 already enumerate every lever that is ON and every lever measured-and-declined;
this document walks the pinned kernels themselves and names, with file:line or a local microbenchmark,
what is still on the table and what is structural.

Everything here is read from the **served bytes** — the extracted r34 image at `/var/tmp/gg-rootfs` —
not from upstream heads. `$SP` = `/var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages`;
`$EXT` = `/var/tmp/gg-rootfs/opt/exllamav3-python/exllamav3/exllamav3_ext` (full CUDA sources are
bundled in the image, so no clone can drift from what actually runs). Microbenchmarks ran on THIS
workstation's local RTX PRO 6000 Blackwell Server Edition (97,887 MiB, driver 595.58.03 — same SKU as
the 1x rental) through `/var/tmp/work/ggrun.sh`, with `nvidia-smi` verified to show zero compute
processes before each run. Receipts: `receipts/kernel-gap-*.json`.

---

## F1. The pins — every byte this analysis reads is identified

**Claim.** The full source of the served stack is pinned and locally present; no upstream-head guessing
is needed anywhere in this document.

| component | version / pin | where the bytes are | provenance |
|---|---|---|---|
| vLLM-GG fork | `0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34` — fork commit `4d006a4` | `$SP/vllm/` | `$SP/vllm/_version.py:21`; repo `github.com/local-inference-lab/vllm` |
| b12x | `1.2.1`, commit `cd3ce19` (encoded in the vLLM version string) | `$SP/b12x/` — CuTe-DSL kernels are Python source, fully readable | `b12x-1.2.1.dist-info/METADATA` |
| exllamav3 | `0.0.43` | Python: `/var/tmp/gg-rootfs/opt/exllamav3-python/exllamav3/`; **full CUDA sources bundled** at `$EXT/quant/*.cu[h]` (exl3_gemm, exl3_gemv, reconstruct, hadamard) plus `gdn.cu`, `hgemm.cu`; prebuilt ext `.so` sha256 `829b982709188f2f…` at `/var/tmp/gg-rootfs/opt/exllamav3/` | `exllamav3/version.py:1`; repo `github.com/turboderp-org/exllamav3` |
| flashinfer | `0.6.18+cu132`, commit `1ac6942` (version string) + `flashinfer_jit_cache 0.6.18+cu132` | `$SP/flashinfer/` | dist-info METADATA |
| torch | `2.12.0+cu132` | `$SP/torch/` | dist-info |
| CUDA | 13.2 (`nvidia_cuda_nvcc-13.2.78`) | wheel-shipped toolkit | dist-info |
| model | `Qwen3.8-27B-EXL3-K5K6-hydrated` (local copy, 20.2 GiB, 3 shards) | `/var/tmp/models/Qwen3.8-27B-EXL3-K5K6-hydrated/` | `quantization_manifest.json`, `release-evidence.json` in-tree |

Model geometry used throughout (from the hydrated tree's `config.json`, `text_config`): 64 layers =
**48 linear_attention (GDN) + 16 full_attention**, hidden 5120, MLP intermediate 17408, head_dim 256,
24 Q / 4 KV heads, GDN: 48 V-heads × 128, 16 K-heads × 128, conv kernel 4; vocab 248,320;
1 MTP hidden layer (served at MTP depth 3 by re-running the same draft layer).

The `/opt/vllm` tree inside the image is **not** byte-identical to `$SP/vllm` (`diff` on `exl3.py`
differs) — `$SP` is what Python imports, so every citation below is `$SP` or `$EXT`.

---
