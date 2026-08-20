# KLD protocol

> **Superseded.** This is the iteration-1 single-window logits protocol, superseded via
> [14-fidelity-protocol-v2.md](14-fidelity-protocol-v2.md) (v2) by
> [42-kld-method.md](42-kld-method.md), the method of record. Kept for provenance.

**The KLD scripts are not in the container image.** Searched the flattened r34
rootfs: no KLD harness, only vLLM's stock `benchmarks/` tree. That matches how
the published driver works — `bench-glm52-exl3-shared-h-kld.sh` mounts a
host-side `runner.py` into the image with
`--entrypoint /opt/venv/bin/python /runner.py`. The scripts live in the
`rtx6kpro` repo and are fetched from there.

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

The superseded runner used `enforce_eager=True`, prefix caching off,
`max_num_seqs=1`, `max_num_batched_tokens=256`, an internal
`kv_cache_memory_bytes=512MiB` clamp, `gpu_memory_utilization=0.95`,
`max_logprobs=-1`, bfloat16, safetensors, and spawn. The KV-memory override is
historical provenance, not a current serving or evaluation recommendation;
repository-head workflows leave it unset. No RNG seed was needed because
nothing was sampled. Token-prefix and finite-value assertions caught drift.

## Adaptation for this effort

| change | reason |
|---|---|
| `--tensor-parallel-size 1` | one RTX PRO 6000 |
| drop `hf_overrides` (`use_index_cache`, `index_topk_pattern`), `B12X_MLA_SPARSE`, `moe_backend='b12x'` | GLM-5.2 MoE/MLA specifics; this model is a dense hybrid-attention VLM |
| make `quantization` a flag | must run `exl3`, `compressed-tensors` and unquantized BF16 through the same code path |
| regenerate the token window with the Qwen tokenizer and freeze it | vocab 248320, and the committed GLM window is tokenizer-specific |
| teacher and candidates share one KV dtype | the r34 receipt's own stated limitation was an FP8-KV teacher vs NVFP4-KV serving |

Reference logits are `2047 x 248320 x 4 B = 2.03 GiB` fp32 per window.
Teacher and candidates loaded sequentially. Full-vocabulary extraction required
substantial temporary memory; current workflows rely on profiled allocation
rather than an undocumented KV-memory override.

Candidates to score: BF16 teacher (control), `nvidia/Qwen3.6-27B-NVFP4` is a
different generation so the same-generation control is `unsloth/Qwen3.8-27B-NVFP4`,
and our mixed EXL3 checkpoint with `ONLINE_QUANT=exl3-b6`.

## Our adapted harness

[`tools/qwen38_kld.py`](../tools/qwen38_kld.py) keeps `positional_kld` and the
reporting statistics byte-for-byte from the published runner, and adds three
subcommands: `tokens` (freeze the window), `capture` (BF16 teacher), `score`
(candidate, N repeats).

Environment-forced differences, all recorded in each `summary.json`:

| change | reason |
|---|---|
| TP1, no `B12X_MLA_SPARSE`, no `moe_backend=b12x`, no GLM `hf_overrides` | one GPU, dense hybrid-attention model |
| `--quantization` / `--quantization-config` as flags | one code path for BF16, `compressed-tensors` and `exl3` + online-K6 |
| `prompt_logprobs=-1, flat_logprobs=True` densified to `[npos, vocab]` | this build has no `SamplingParams.return_prompt_logits`; this is the harness's own documented fallback |
| window from exllamav3's bundled `wiki.utf8` (first 2048 tokens), frozen to JSON | the image has no `datasets` package; WikiText-2 is unavailable offline. Disclosed, deterministic, and the same window is asserted for every candidate |

The published GLM numbers are therefore *shape*-comparable, not
value-comparable: a different corpus window and a different model. Only our own
BF16-teacher-relative numbers are comparable to each other.

## Candidate set

| label | model | flags |
|---|---|---|
| `bf16-teacher` | `Qwen/Qwen3.8-27B` | reference; captured once |
| `unsloth-nvfp4` | `unsloth/Qwen3.8-27B-NVFP4` | `--quantization compressed-tensors` |
| `k4-online-k6` | ours | `--quantization exl3 --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":[...]}'` + `VLLM_EXL3_ONLINE_TRELLIS_BITS=6` |
| `k4-serialized` | ours, overlay off | `--quantization exl3` with no online config: attention stays BF16 at runtime, isolating the K6 overlay's contribution |

One KV dtype is pinned across all four.
