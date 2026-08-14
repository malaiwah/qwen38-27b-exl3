# KLD protocol

The metric shape published for GG r34 ("2,047 positions over the full
vocabulary against one BF16 teacher, three repeats, mean KLD + run SD") is
produced by `local-inference-lab/rtx6kpro@master`:

- `scripts/glm52_exl3_shared_h_kld.py` — the candidate runner. In-process
  `vllm.LLM`, not an HTTP endpoint.
- `scripts/bench-glm52-exl3-shared-h-kld.sh` — docker driver, `MODE=checkpoint|runtime-fp8|runtime`.
- `models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/collect_prefill_return_logits_ref.py`
  — BF16 teacher capture.
- `benchmarks/data/glm52-kld-tokens-2048.json` — the frozen token window.
- `benchmarks/glm52-kld-evaluation.md` — methodology, and the interpretation
  scale this project uses: <0.01 near-lossless, 0.01-0.05 good, 0.05-0.1
  noticeable, >0.1 significant.

Caveat worth recording: the r34 receipt
(`blackwell-llm-docker@98224d13/validation/gilded-gnosis-v20-r34-remote-gpu.json`)
carries the 0.064467/0.065339 values under `evidence.teacher_forced_quality` with
an **empty artifacts map** for quality — only the perf JSONs are hashed. The
SHA-pinned r28 harness is the reusable artifact; r34 reused the protocol.

## Math

Both sides go through `F.log_softmax`, then
`F.kl_div(candidate_logprobs, reference_logprobs, reduction='none', log_target=True).sum(-1)`
per position, in row chunks — i.e. **KL(BF16 teacher || candidate)**, full
vocabulary, no top-k. Teacher forcing is a single prefill over one fixed
ground-truth token sequence; scored positions are prompt positions 1..2047.

Reported statistics: per run `mean_kld` over positions; headline `mean_kld` is
the mean of the run means and `run_sd` is the **sample SD across the three run
means**.

## Determinism controls

`enforce_eager=True`, `enable_prefix_caching=False`, `max_num_seqs=1`,
`max_num_batched_tokens=256`, `kv_cache_memory_bytes=512MiB`,
`gpu_memory_utilization=0.95`, `max_logprobs=-1`, `dtype=bfloat16`,
`load_format=safetensors`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`. No RNG seed:
nothing is sampled. The runner asserts `token_ids[:16] == manifest["token_first16"]`
to catch tokenizer drift, and refuses to emit results on any non-finite logit or
KLD.

## Adaptation for this effort

| change | reason |
|---|---|
| `--tensor-parallel-size 1` | one RTX PRO 6000 |
| drop `hf_overrides` (`use_index_cache`, `index_topk_pattern`), `B12X_MLA_SPARSE`, `moe_backend='b12x'` | GLM-5.2 MoE/MLA specifics; this model is a dense hybrid-attention VLM |
| make `quantization` a flag | must run `exl3`, `compressed-tensors` and unquantized BF16 through the same code path |
| regenerate the token window with the Qwen tokenizer and freeze it | vocab 248320, and the committed GLM window is tokenizer-specific |
| teacher and candidates share one KV dtype | the r34 receipt's own stated limitation was an FP8-KV teacher vs NVFP4-KV serving |

Reference logits are `2047 x 248320 x 4 B = 2.03 GiB` fp32 per window. Teacher
and candidates load sequentially, never together, so 96 GB is sufficient; keep
`kv_cache_memory_bytes` clamped because full-vocab prompt-logprob extraction
allocates several V-wide temporaries.

Candidates to score: BF16 teacher (control), `nvidia/Qwen3.6-27B-NVFP4` is a
different generation so the same-generation control is `unsloth/Qwen3.8-27B-NVFP4`,
and our mixed EXL3 checkpoint with `ONLINE_QUANT=exl3-b6`.
