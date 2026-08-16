# Two protocols with one metric name

Unsloth and this project both publish a number called "mean KLD against BF16", both in the
direction `KL(BF16 ‖ candidate)`, both over the full vocabulary of the same 248,320-token
model. The numbers are still not interchangeable, because the two protocols differ in corpus,
scoring geometry, reference numerics, what is inside the measured path, and how uncertainty is
computed. Neither protocol is wrong. They measure different things, and this file says exactly
which things, so that both numbers can be published side by side without either being quoted
as a refutation of the other. Two of the differences it lists are no longer arguments: the
scoring window (D1) and the cross-engine term have both been measured on our own data, and this
file reports both. One run — theirs, on their corpus — is still open.

Every external number below carries a source tag defined in [Sources](#sources). Every
internal number carries a receipt path. Every byte size is serialized bytes on disk — never
VRAM, resident weights, activations or KV.

## Their protocol, as identified from primary artifacts

Unsloth's GGUF KLD pipeline is stock `llama.cpp` `llama-perplexity --kl-divergence`. The
protocol sentence is stated once in their docs, on the DeepSeek-V4 page rather than the
Qwen3.8 page: *"Reference = official weights. Perplexity and KL-divergence over wikitext-2 at
ctx 512 on 4x B200."* [U-DSV4] The Dynamic-2.0 page states the anti-overfitting rationale —
they deliberately do **not** evaluate on their own calibration data. [U-DYN2]

The verbatim commands come from their own published run logs for Qwen3.5-35B-A3B [U-LOG-B,
U-LOG-Q5]; `build: 8164 (b68d75165)`, 8x B200, GNU 13.3.0:

```
# reference capture
env LLAMA_SET_ROWS=1 ./llama.cpp/llama-perplexity --flash-attn on --fit off \
  --batch-size 16384 --ubatch-size 16384 --parallel 1 --mlock --no-mmap --device CUDA0 \
  --model .../BF16/<model>-BF16-00001-of-00002.gguf \
  --file .../wikitext-2-raw/wiki.test.raw --ctx-size 512 \
  --save-all-logits .../kld_logs/pipeline_base_logits.bin

# candidate scoring
env LLAMA_SET_ROWS=1 ./llama.cpp/llama-perplexity --flash-attn on --fit off \
  --batch-size 16384 --ubatch-size 16384 --parallel 1 --mlock --no-mmap --device CUDA4 \
  --model .../<model>-UD-Q5_K_XL.gguf \
  --file .../wikitext-2-raw/wiki.test.raw --ctx-size 512 \
  --kl-divergence --kl-divergence-base .../kld_logs/pipeline_base_logits.bin
```

The published doc tables are literally this output: docs row `Unsloth | Q5_K_XL | 23.22 |
6.5489 | 0.236 | 0.0069` [U-Q35] against log `Mean PPL(Q) 6.548881`, `99.9% KLD 0.236021`,
`Mean KLD 0.006949` [U-LOG-Q5]. Their "top-1" column is llama.cpp's `Same top p`; their
"RMS delta-p" is `RMS Δp`.

### Corpus identity

| property | value | source |
|---|---|---|
| identity | WikiText-2 **raw test** split, `wikitext-2-raw/wiki.test.raw` | [U-LOG-B] |
| pinnable archive | `https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip` | [HF-WT2] |
| archive bytes / sha256 | 4,721,645 / `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11` | [HF-WT2] |
| `wiki.test.raw` bytes / sha256 | **1,290,590** / `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` | [HF-WT2] |
| dataset of record | `Salesforce/wikitext` rev `b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-2-raw-v1` | [HF-WT2-DS] |
| framing | raw text through `--file`; no chat template, no system prompt; Qwen GGUF sets `add_bos=false` | [LC], [U-LOG-B] |

### Chunking arithmetic

`llama-perplexity` splits the token stream into `floor(n_tokens / n_ctx)` non-overlapping
chunks and scores only the second half of each chunk: `const int first = n_ctx/2;` and
`process_logits(..., n_ctx - 1 - first, ...)` [LC:541-544, LC:616-624].

| step | value | source |
|---|---:|---|
| tokens, Qwen3.8 tokenizer over `wiki.test.raw` | 297,193 | [INV] |
| chunks, `297193 // 512` | **580** | [INV]; matches their log's `calculating perplexity over 580 chunks, n_ctx=512` [U-LOG-B] |
| processed tokens, `580 x 512` | **296,960** | [INV]; matches their log's `296960 tokens` [U-LOG-B] |
| scored positions per chunk, `512 - 1 - 256` | 255 | [LC:541-544, LC:616-624] |
| scored positions, `580 x 255` | **147,900** | [LC], [INV] |

The geometry is confirmed independently at the byte level: their published
`pipeline_base_logits.bin` is 73,455,427,060 B, and the file layout
`20 + n_chunk*n_ctx*4 + n_chunk*255*nv*2` with `nv = 2*((248320+1)/2)+4 = 248,324`
[LC:1752] evaluates to exactly 73,455,427,060 — which pins `n_ctx=512`, `n_chunk=580`,
`n_vocab=248,320` and 255 scored positions per chunk with no reliance on their prose.

### The scoring floor

Because `first = n_ctx/2`, only positions 256-510 of each 512-token window are scored, and
**every scored position has at least 256 tokens of left context** [LC:541-544]. Positions
0-255 are prefill context and never enter the metric. Our suite scores positions 0-2046 of a
2,048-token window ([`receipts/kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json),
`scored_positions_per_context: 2047`), including the low-context positions at the start of
every window, which are the high-entropy ones.

### Reference numerics

The reference distribution is not stored exactly. The writer clamps
`min_logit = max(min_logit, max_logit - 16)` and encodes each log-probability as `uint16` over
that 16-nat span, `scale = (max_logit - min_logit)/65535` [LC:79-100]. The scoring loop then
runs over all `n_vocab` entries but is gated by `if (p_log_base > -16.f)`, so base-side terms
below -16 nats (p < 1.13e-7) are dropped [LC:224-240]:

```
KLD = Σ_{i : log p_base(i) > -16} p_base(i) · [log p_base(i) − log p_cand(i)]
```

Two consequences are visible in their own logs. `Minimum KLD: -0.000140` (Q8_K_XL
[U-LOG-Q8]) and `-0.000050` (Q5_K_XL [U-LOG-Q5]) are impossible for an exact KL, so the
estimator's numerical floor is at least 5e-5 nats/token. And `Mean PPL(base)` reconstructed
from the stored file is 6.532929 [U-LOG-Q5] against 6.5353 measured directly during capture
[U-LOG-B], a -0.037 % shift. At their Q8-class mean of 0.002562 [U-LOG-Q8] that floor is only
~20-50x below signal.

Ours replays float64 through one shared BF16 head over all 248,320 entries in two passes
(pass 1 accumulates `logsumexp` normalizers and argmax; pass 2 accumulates KL and
Jensen-Shannon), never top-k and never truncated
([`tools/fidelity.py`](../tools/fidelity.py) `cmd_replay`, described in
[14-fidelity-protocol-v2.md](14-fidelity-protocol-v2.md)). Our own floor is separately
measured, not assumed: replaying a model against its own live logits gives mean
`KL(live ‖ replayed)` = **6.54e-04** with top-1 98.999 % and max 0.0913
([`receipts/v3-qualification-bf16.json`](../receipts/v3-qualification-bf16.json)) — the
capture/replay path is the weak link on our side and is disclosed as such in
[18-results-fidelity-v3.md](18-results-fidelity-v3.md).

### What is inside the measured path

Their candidate runs end to end, so the candidate's **own output head** is in the loop; UD
quants typically keep `output.weight` at Q6_K/Q8_0 while stock quants do not, so the head is
both included and variable across their rows. Our protocol captures hidden states at the final
RMSNorm and replays **both** operands through one shared BF16 head
(`head_sha256: 25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff`, recorded in
every [`receipts/kld5-10M-*.json`](../receipts/kld5-10M-hyd.json)), so our number is body-only
by construction and head error is attributed separately
([16-head-attribution.md](16-head-attribution.md)).

### Uncertainty model

Their `±` is a per-token SEM assuming i.i.d. tokens: `Mean KLD: 0.006949 ± 0.000073`
[U-LOG-Q5]. Tokens inside a 512-token chunk are strongly correlated, so that interval is not
comparable to ours, which is a bootstrap over 842 source clusters with 10,000 resamples
(`context_bootstrap.clusters: 842`, `samples: 10000` in
[`receipts/kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json)).

## Our protocol, for the same table

| property | value | receipt |
|---|---|---|
| suite | schema `qwen38-distribution-fidelity/6`, `suite_token_sha256` `510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88` | [`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json) |
| contexts / window | 5,120 contexts of 2,048 tokens, exact-advance, non-overlapping | [`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json) |
| scored positions | 2,047 per context, **10,480,640** total | [`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json) |
| source clusters | 842, one document family per cluster | [`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json) |
| corpus | 941 documents, **70,348,971 bytes**, five strata (code 241 / encyclopedic 323 / literary 59 / multilingual 276 / scientific 42) | [`kld5-corpus-fetch-log.json`](../receipts/kld5-corpus-fetch-log.json) |
| contamination policy | whole-document exclusion on any exact NFKC-casefold 12-word shingle match against 859,426 calibration shingles, 10,772,868 corpus shingles scanned, **44 of 941 documents excluded before selection**, 897 eligible | [`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json) `document_scan`, `contamination_scan` |
| prior-suite disjointness | 0 of the v4 suite's 160 context token hashes reachable | [`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json) `prior_suite_exclusion` |
| metric | exact two-pass full-vocabulary `KL(BF16 ‖ candidate)` in float64, vocab 248,320, both operands through one shared BF16 head | [`tools/fidelity.py`](../tools/fidelity.py), [`kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json) |
| interval | cluster bootstrap, 842 clusters, 10,000 resamples | [`kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json) |

Current ladder on that suite, cumulative over all ten shards:

| candidate | mean KLD | bootstrap 95 % CI | top-1 | exact worst position | receipt |
|---|---:|---:|---:|---:|---|
| hydrated | 0.002760 | [0.002540, 0.003020] | 97.70 % | 8.258 | [`kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json) |
| online K5/K6 | 0.003210 | [0.002982, 0.003480] | 97.52 % | 22.241 | [`kld5-10M-k5k6.json`](../receipts/kld5-10M-k5k6.json) |
| context edition | 0.003509 | [0.003220, 0.003852] | 97.44 % | 5.557 | [`kld5-10M-ctx.json`](../receipts/kld5-10M-ctx.json) |
| official FP8 | 0.005294 | [0.004927, 0.005728] | 96.79 % | 10.714 | [`kld5-10M-fp8.json`](../receipts/kld5-10M-fp8.json) |
| K4 | 0.010604 | [0.009640, 0.011746] | 95.76 % | 14.283 | [`kld5-10M-k4.json`](../receipts/kld5-10M-k4.json) |

The GGUFs are no longer absent from our protocol, but they cannot join *that* table: they were
run on **shard 0** of the same suite, not on all ten shards, so their numbers belong in the
shard-0 table below and never in a column welded from 10,480,640 positions. Note also
that our FP8 row records `model_path: /models/Qwen3.8-27B-FP8` with
`model_revision_source: "none"` ([`kld5-10M-fp8.json`](../receipts/kld5-10M-fp8.json)); the
upstream repo and revision behind it must be named in any cross-published table, since
`unsloth/Qwen3.8-27B-FP8` and `Qwen/Qwen3.8-27B-FP8` are different artifacts.

## The deltas, ordered by expected effect

"Direction" is the expected sign of the difference **on their reported mean relative to
ours**, for the same checkpoint. "unknown" means the sign is not derivable without the
measurement.

| # | delta | theirs | ours | direction | note |
|---|---|---|---|---|---|
| **D1** | scoring floor on positions | positions 256-510 of each 512-token window only; every scored token has ≥256 tokens of left context [LC:541-544] | positions 0-2046 of a 2,048-token window; low-context positions are scored | **theirs lower** | largest structural difference; **now measured** on our own data with `--score-from` — worth 1.3-2.1 % at their absolute floor and 3.9-4.9 % at their proportional one ([below](#d1-resolved-the-scoring-window-moves-every-mean-by-at-most-about-5-percent)) |
| **D2** | corpus | WikiText-2 raw test only: English encyclopedic, 1,290,590 B [HF-WT2] | 941 documents, 70,348,971 B, five strata ([`kld5-corpus-fetch-log.json`](../receipts/kld5-corpus-fetch-log.json)) | **theirs lower** | their own NVFP4 rows span 4.7x across domains, 0.0124 → 0.05818 [U-Q38]; our own encyclopedic stratum sits below our all-strata mean, 0.004889 vs 0.005197 on the first shard ([`kld5-1M-tail-fp8.json`](../receipts/kld5-1M-tail-fp8.json)) |
| **D3** | reference numerics | BF16 GGUF in llama.cpp, logits stored `uint16` over a 16-nat span [LC:79-100]; observed `Minimum KLD: -0.000140` [U-LOG-Q8], so floor ≥5e-5 nats/token | float64 two-pass replay of BF16 hidden states ([`tools/fidelity.py`](../tools/fidelity.py)); own floor 6.54e-04 measured, not assumed ([`v3-qualification-bf16.json`](../receipts/v3-qualification-bf16.json)) | unknown, bounded | a noise floor, not a bias: it can push individual positions negative. Matters only near Q8-class means |
| **D4** | vocabulary coverage | all 248,320 entries **minus** base-side terms with `log p ≤ -16` [LC:224-240] | exact, all 248,320 entries, no truncation | theirs lower (small, unproved) | the dropped region carries positive divergence when a quantization under-represents rare tokens, which is the usual failure mode; the operator is not the one we call "full vocabulary" |
| **D5** | output head | candidate's own head is in the loop, and its precision varies by quant recipe | both operands through one shared BF16 head, body-only by construction | **theirs higher** | opposite sign to D1/D2; a head-inclusive number of ours exists separately ([16-head-attribution.md](16-head-attribution.md)) |
| **D6** | window length | 512 [U-LOG-B] | 2,048 ([`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json)) | unknown | interacts with D1; their windows never exercise long-range behaviour |
| **D7** | uncertainty model | per-token SEM assuming i.i.d. tokens, e.g. `± 0.000073` on 0.006949 [U-LOG-Q5]; 147,900 positions | source-cluster bootstrap, 842 clusters x 10,000 resamples; 10,480,640 positions ([`kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json)) | no effect on the point estimate | their interval is narrower than an honest one for correlated tokens; do not place the two intervals side by side without saying so |
| **D8** | framing | raw text, no chat template, no system prompt [U-LOG-B] | raw text, `add_special_tokens=False` | **none** | recorded so it is not mistaken for a difference; their own Dynamic-2.0 page argues text-only evaluation is inadequate for instruct models [U-DYN2], but that argument applies to both sides equally |
| **D9** | contamination auditability | eval set is deliberately not their calibration set [U-DYN2], but the calibration set is private and no imatrix blob ships in the GGUF repo [HF-GGUF], so overlap is unauditable | whole-document pre-exclusion on exact 12-word calibration overlap, 44 documents removed before any result was visible ([`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json)) | unknown; overlap would push theirs lower | asymmetry of evidence, not of intent — ours is auditable, theirs is not, and that must be stated when both numbers appear together |
| **D10** | engine, kernels, KV | llama.cpp CUDA, flash-attn on, f16 KV, batch/ubatch 16384, `n_seq=32` [U-LOG-B] | vLLM/EXL3, `enforce_eager`, `max_num_seqs=1`, one chunk per prefill, `kv_cache_dtype=auto` ([05-kld-protocol.md](05-kld-protocol.md)) | unknown, small at ctx 512 | second tokenizer implementation too: GGUF BPE (`tokenizer.ggml.model=gpt2`, `pre=qwen35`) vs HF `tokenizers`. Token-ID equality must be asserted per chunk, not assumed |
| **D11** | top-1 definition | `Same top p` = argmax(candidate logits) vs argmax of the **uint16-quantized** reference, no tie handling [LC:224-240] | exact argmax on both operands | theirs lower | near-ties flip under 16-bit reference storage; at 98-99 % agreement this is a real contributor |
| **D12** | the NVFP4 rows have no protocol at all | corpus labels only (zh, code, refgen, chat, ja/ko/ru/es); no tokens, ctx, reference precision, direction or engine [U-Q38]; NVFP4 cannot run under `llama-perplexity` | 278,392 positions, protocol above | not comparable in either direction | our 0.094978 ([`v3-report-nvfp4-analysis.json`](../receipts/v3-report-nvfp4-analysis.json)), 0.092727 contamination-corrected ([`analysis-v3-contamination-corrected.json`](../receipts/analysis-v3-contamination-corrected.json)) and their 0.0124-0.05818 [U-Q38] should both be published, never divided |

D1, D2, D4 and D11 push their number down relative to ours; D5 pushes it up; D3, D6, D9 and
D10 are unsigned. That is why the gap between their NVFP4 rows and ours cannot be attributed
to a single cause, and why D1 was the only one worth isolating first: it is the one that is both
large and exactly reproducible on our own data — which is what the next section does.

## Two of these deltas are now measured

D1 and the cross-engine term were the two items in this file that were arguments rather than
numbers. Both were run on **shard 0 of the v5 ladder — 512 contexts, 1,048,064 scored positions**,
the identical contexts and one shared BF16 head for every candidate, so both are measured on the
same data the rest of this project publishes.

### D1 resolved: the scoring window moves every mean by at most about 5 percent

`tools/fidelity.py replay --score-from N` was run at **N = 0 / 256 / 1024** over the five
existing candidates. It slices already-captured hidden states, so it cost no new capture and no
GPU. Receipt
[`receipts/scored-window-offset.json`](../receipts/scored-window-offset.json), schema
`qwen38-scored-position-window-offset/1`; the fifteen underlying reports are
`receipts/kld5-window-{hyd,k5k6,ctx,fp8,k4}-from{0,256,1024}.json`, schema
`qwen38-fidelity-report/3`.

| candidate | `--score-from 0`, 2,047 pos/ctx | `--score-from 256`, 1,791 | change | `--score-from 1024`, 1,023 | change |
|---|---:|---:|---:|---:|---:|
| hydrated | 0.002700 | 0.002660 | **-1.5 %** | 0.002580 | **-4.5 %** |
| online K5/K6 | 0.003141 | 0.003100 | **-1.3 %** | 0.003020 | **-3.9 %** |
| context edition | 0.003409 | 0.003342 | **-2.0 %** | 0.003243 | **-4.9 %** |
| official FP8 | 0.005197 | 0.005090 | **-2.1 %** | 0.004955 | **-4.7 %** |
| K4 | 0.010345 | 0.010154 | **-1.9 %** | 0.009876 | **-4.5 %** |

D1's predicted sign was right and its size is now bounded. Restricting our measurement to
**their absolute 256-token left-context floor** lowers every candidate's mean by **1.3-2.1 %**;
restricting it to **their proportional second-half rule** at our window length lowers it by
**3.9-4.9 %**. Top-1 agreement rises by 0.03 to 0.13 points over the same restriction. The effect
is nearly uniform across candidates, so **every ordering, every ratio between our own rows and
every paired conclusion is unchanged** — the windowed columns rank the five candidates exactly as
the full one does.

The consequence for cross-protocol reading is the point of the control: **the scoring floor
explains at most about 5 % of the distance between our numbers and externally published ones.**
Whatever remains is D2 (corpus composition), D3/D4/D11 (reference numerics, vocabulary gating,
top-1 definition) and D5 (head placement), which is exactly why a cross-protocol ratio must never
be computed in either direction. A windowed number of ours is also not a number on their axis: it
is our corpus, our window length, our reference numerics and our body-only head, with only the
position selection changed.

### The cross-engine term resolved: two engines disagree by 0.000507 on identical weights

Scoring a GGUF on our suite means capturing hidden states in llama.cpp and comparing them against
a reference captured in vLLM, so a GGUF number carries engine numerics on top of quantization
error. That term is now measured rather than waved at: the **unquantized BF16 GGUF**
(`unsloth/Qwen3.8-27B-GGUF@f1bfb127…`, two-part, 54,657,735,616 B) against the vLLM BF16
reference, identical token ids, the same shared BF16 head, the same 512 contexts.

| control | mean KLD | 95 % CI | top-1 | p99.9 | exact max | receipt |
|---|---:|---|---:|---:|---:|---|
| llama.cpp BF16 vs vLLM BF16, same weights | **0.000507** | [0.000492, 0.000523] | 99.07 % | 0.0113 | 1.519 | [`gguf-report-engine-floor.json`](../receipts/gguf-report-engine-floor.json) |

That is the floor of this comparison: two engines running the *same unquantized weights* disagree
by 0.000507 mean and 0.93 points of top-1. **Every GGUF row below contains that term; no vLLM row
— ours or official FP8's — does.** It is the asymmetry that had to be measured before any
cross-engine ranking could be published at all.

The three GGUF targets, on shard 0 of our suite, through the same shared BF16 head, with the naive
net-of-floor estimate beside each measured value
([`receipts/cross-engine-comparator.json`](../receipts/cross-engine-comparator.json); per-candidate
reports `receipts/gguf-report-{q8_0,q6_k,q5_k_xl}.json`):

| GGUF | measured mean KLD | net of the 0.000507 floor | top-1 | p99.9 | serialized bytes | GiB |
|---|---:|---:|---:|---:|---:|---:|
| `Q8_0` | 0.001087 | ~0.000579 | 98.53 % | 0.0351 | 29,047,086,048 | 27.052 |
| `Q6_K` | 0.002035 | ~0.001528 | 97.98 % | 0.0794 | 22,884,408,288 | 21.313 |
| `UD-Q5_K_XL` | 0.004444 | ~0.003936 | 97.20 % | 0.2144 | 20,218,178,624 | 18.830 |

**KL is not additive, so the net column is an estimate, not an identity.** The measured value is
an upper bound on each GGUF's quantization error and the net figure is the naive lower one; the
truth is between them, and `Q8_0`'s measured 0.001087 is only about twice the floor, so its own
number sits near the resolution limit of any cross-engine comparison: **no ordering closer than a
factor of two should be pressed against `Q8_0`**. Sizes above are serialized
disk bytes [HF-GGUF] — never VRAM, resident weights or KV.

Harness provenance, so the capture point is auditable rather than asserted:
[`tools/gguf_capture.cpp`](../tools/gguf_capture.cpp) reads the **post-final-norm** state
(`res->t_embd`, `src/models/qwen35.cpp`) — the same mathematical point our vLLM hook takes — with
llama.cpp pinned at commit `ece963f41b0b02d7a0d61436ae365762c073a4c8`, built by
[`tools/build_llamacpp.sh`](../tools/build_llamacpp.sh), manifests by
[`tools/gguf_manifest.py`](../tools/gguf_manifest.py), each manifest carrying the GGUF blob digest
and the llama.cpp identity, and bf16 rounding verified **bit-identical to torch on 2,012,449 probe
values**.

### What the two measurements change about the reading

Net of the floor, `Q6_K` reads ~0.001528 at 21.31 GiB against our hydrated build's 0.002700 at
20.12 GiB of payload, and `UD-Q5_K_XL` reads ~0.003936 at 18.83 GiB against our context edition's
0.003409 at 19.27 GiB. Our two payload figures are `immutable_payload_bytes` from
[`receipts/collection-index.json`](../receipts/collection-index.json) — hydrated 21,610,916,123 B =
20.127 GiB, context 20,696,033,532 B = 19.275 GiB, quoted above truncated to two decimals —
serialized disk bytes on the same axis as the GGUF sizes, and never
resident weights. **At the 6-bit operating point GGUF is genuinely better than our best
build; at the 5-bit operating point ours is about 13 % better than theirs.** The full eight-row
comparison, including official FP8 and K4, is in
[29-plan-and-loose-ends.md](29-plan-and-loose-ends.md) F2 and on every model card. The format
advantage at this bitrate is therefore real at 5 bits and negative at 6 bits on this suite — far
short of the "about a bit ahead" that turboderp's own chart shows on *his* protocol, which is a
difference between protocols and not a contradiction to settle by division. What none of this
settles: it is text-only teacher-forced fidelity, and it says nothing about serving 262,144 tokens
with vision and MTP on a 32 GB card, nor about llama.cpp KV-quant behaviour, prefill or decode
speed, which are separate axes.

## What they actually publish for this model family

### Qwen3.8-27B GGUFs: no KLD table exists

The entire published content of the Qwen3.8 "Quantization Analysis" section is one prose
figure — *"We ran top-1% and KLD for Qwen3.8 GGUFs, and we retain 82.5% accuracy (IQ2_XXS
9GB) whilst being 83.5% smaller (BF16 54.7GB)"* — one top-1-versus-size figure, and *"More
benchmarks coming soon!"* [U-Q38]. There is no per-quant KLD table, and no protocol metadata
for the figure. So the F2 comparison against `Q5_K_XL` / `Q6_K` / `Q8_0` could not be closed by
citation and was run instead — on our suite, in the section above
([29-plan-and-loose-ends.md](29-plan-and-loose-ends.md) F2). What citation still cannot give us,
and Run A below still would, is a number on *their* axis.

### The Qwen3.8 top-1 chart: digitized, therefore not a published number

The one figure, `q38_top1_vs_size.png` ("Qwen3.8 27B top-1% accuracy agreement", y = top-1
token agreement with BF16 %, x = GGUF size in GiB), was digitized locally by marker-blob
centroid after axis calibration on its gridlines [INV]. The recovered x-coordinates match the
real file sizes [HF-GGUF] to ≤0.009 GiB, which validates the digitization and the
point-to-artifact mapping:

| quant | actual size (GiB) [HF-GGUF] | x recovered [INV] | top-1 % read off the raster [INV] |
|---|---:|---:|---:|
| UD-IQ2_XXS | 8.391 | 8.386 | 82.39 |
| UD-IQ2_M | 9.611 | 9.606 | 85.49 |
| UD-Q2_K_XL | 9.943 | 9.947 | 86.15 |
| UD-IQ3_XXS | 11.095 | 11.104 | 90.31 |
| UD-Q3_K_XL | 12.518 | 12.517 | 92.43 |
| UD-Q4_K_XL | 16.692 | 16.698 | 96.10 |
| UD-Q5_K_XL | 18.830 | 18.829 | 97.15 |
| Q6_K | 21.313 | 21.318 | 97.86 |
| UD-Q6_K_XL | 24.144 | 24.148 | 98.50 |
| Q8_0 | 27.052 | 27.059 | 98.78 |
| UD-Q8_K_XL | 29.298 | 29.304 | 98.97 |

**These are read-off values from a published raster, not published numbers.** They carry
roughly ±0.05 pp of digitization uncertainty on top of whatever the underlying protocol was,
and that protocol is not stated for this figure at all. They may be cited as "read off
Unsloth's published chart", never as "Unsloth reports", and never used as the reference values
in a comparison table. Only 11 of the 22 uploaded quants appear in the figure; the stock
`Q3_K_S`, `Q3_K_M`, `IQ4_XS`, `IQ4_NL`, `Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`,
`Q5_K_M` builds are not charted.

### Their nearest published exact ladder: Qwen3.5-35B-A3B

Same harness, wikitext-2 at ctx 512, `KL(BF16 ‖ Q)`, from their "Full Benchmarks" table
[U-Q35] (values verbatim; sizes are their GB column):

| quantizer | quant | GB | PPL | KLD 99.9 % | mean KLD |
|---|---|---:|---:|---:|---:|
| Unsloth | Q4_K_XL | 19.17 | 6.5918 | 0.4097 | 0.0137 |
| Unsloth | Q5_K_XL | 23.22 | 6.5489 | 0.2360 | **0.0069** |
| Unsloth | Q6_K_S | 26.56 | 6.5456 | 0.2226 | 0.0065 |
| Unsloth | Q6_K_XL | 28.22 | 6.5392 | 0.1437 | **0.0041** |
| Unsloth | Q8_K_XL | 36.04 | 6.5352 | 0.1033 | **0.0026** |
| bartowski | Q5_K_M | 23.11 | 6.5828 | 0.3549 | 0.0106 |
| AesSedai | Q5_K_M | 24.45 | 6.5356 | 0.2100 | 0.0058 |

This is a different model (35B A3B MoE, not our 27B dense) and a different Dynamic version —
the Qwen3.8 GGUFs are labelled "Unsloth Dynamic V3.0 (preview)" while Qwen3.5/3.6 are Dynamic
2.0 [U-Q38, U-Q35] — so the ladder is a **scale anchor for their protocol**, not a prediction
for our artifacts. Two further version hazards: the Qwen3.5 quants were re-issued on
2026-03-05 with a new algorithm and new imatrix data (UD-Q5_K_XL moved 23.2 → 24.6 GB, max
KLD 5.536 → 3.210), so the table above and the `KLD_Logs` artifacts are both the pre-update
build [U-Q35, U-LOG-Q5]; and `--save-all-logits` file layout is build-coupled, with no version
check beyond the `_logits_` magic and an `n_ctx` comparison [LC].

For the same family they also publish a Qwen3.6-27B MLX table with a KLD distribution
(8-bit 0.0028, UD-6bit 0.0037, UD-4bit 0.0227, UD-NVFP4 0.0325, UD-MXFP4 0.0479, UD-3bit
0.0734) [U-Q36] — a different engine and quantizer again, useful only as a second scale
anchor.

### The NVFP4 rows for Qwen3.8-27B

The only exact Qwen3.8-27B KLD numbers Unsloth publishes, verbatim [U-Q38]:

| corpus | KLD mean | top-1 agreement |
|---|---:|---:|
| zh | 0.01628 | 93.55 % |
| code | 0.02600 | 96.68 % |
| refgen | 0.03993 | 94.46 % |
| chat | 0.05818 | 92.15 % |
| ja / ko / ru / es | 0.0124-0.0155 | 94-95 % |

Range **0.0124-0.05818**. No corpus identity, token count, context length, reference
precision, KL direction or engine accompanies them, and NVFP4 does not run under
`llama-perplexity`, so the protocol identified above does not apply to these rows. See D12.

## The EXL3 author's own charts, read off the images

`turboderp/Qwen3.8-27B-exl3` publishes three charts, and they matter more than the Unsloth
tables because they are the comparison the community keeps asking for: one BF16 reference, one
protocol, one machine, five families on the same axes — EXL3 at seven bitrates, the Unsloth
GGUF UD ladder, one GGUF-IQ point, Unsloth NVFP4 and official Qwen FP8. Subtitle on all three:
`openwebtext, 8 × 8192 tokens, formatted`. The x-axis is **quantized weight size, GiB,
excluding embeddings and including the output head** — decoder linear storage, not VRAM.

Values below are read off the published rasters. They are labelled on the plots, so they are
the authors' own printed numbers rather than digitisation guesses, but they are not a published
data file: treat them as chart labels, cite them as such, and never mix them with our numbers.

| family / point | size on his axis (GiB) | mean KLD | median KLD | mean ÷ median |
|---|---:|---:|---:|---:|
| EXL3 2.00 bpw | ~6.6 | 0.351 | 0.1268 | 2.8 |
| EXL3 2.50 bpw | ~8.1 | 0.299 | 0.0783 | 3.8 |
| EXL3 3.00 bpw | ~9.5 | 0.112 | 0.0273 | 4.1 |
| EXL3 4.00 bpw | ~12.4 | 0.052 | 0.0086 | 6.0 |
| EXL3 5.00 bpw | ~15.0 | **0.014** | 0.0022 | 6.4 |
| EXL3 6.00 bpw | ~17.8 | **0.007** | 0.0010 | 7.0 |
| GGUF UD-Q2_K_XL | ~9.6 | 0.237 | 0.0916 | 2.6 |
| GGUF UD-Q3_K_XL | ~12.0 | 0.089 | 0.0238 | 3.7 |
| GGUF UD-Q4_K_XL | ~16.0 | 0.039 | 0.0069 | 5.7 |
| GGUF UD-Q5_K_XL | ~18.0 | **0.019** | 0.0032 | 5.9 |
| GGUF UD-Q6_K_XL | ~23.1 | **0.009** | 0.0012 | 7.5 |
| GGUF-IQ IQ4_XS | ~13.9 | 0.055 | 0.0131 | 4.2 |
| NVFP4 (Unsloth) | ~17.9 | 0.041 | 0.0103 | 4.0 |
| FP8 (Qwen) | ~25.1 | **0.023** | 0.0035 | 6.6 |
| synthetic noise floor | — | 0.0052 | 0.0007 | 7.4 |

Five readings, in order of how much they change what we should say.

**1. At equal size, EXL3 is about a bit ahead of the GGUF UD ladder, on his measurement.**
5.00 bpw at ~15.0 GiB scores 0.014 against UD-Q5_K_XL's 0.019 at ~18.0 GiB — better *and*
3 GiB smaller. 6.00 bpw at ~17.8 GiB scores 0.007 against UD-Q6_K_XL's 0.009 at ~23.1 GiB —
better at 5.3 GiB less. The one place GGUF wins on the raw number is 4-bit (UD-Q4_K_XL 0.039
at 16.0 GiB versus EXL3 4.00's 0.052 at 12.4 GiB), and even there the EXL3 point is 3.6 GiB
smaller, so the size-matched comparison still favours EXL3. This is the format author's own
chart, so it is not independent of him — but it is the first same-protocol, same-reference
comparison of these families that exists for this model at all.

**2. Official FP8 is a Q4-to-Q5-class quality point that costs 25.1 GiB.** On his axis it
scores 0.023, between UD-Q4_K_XL (0.039) and UD-Q5_K_XL (0.019), while occupying more decoder
weight than any other point on the chart. The public framing that "FP8 benchmarks about the
same as a Q5/Q6" is therefore slightly generous to FP8 here, and our own result — three
builds 34-48 % below official FP8 at 18-20 GiB resident — is consistent in direction with his
ordering even though the absolute numbers are not comparable.

**3. Cross-protocol ratios are not stable, and this chart proves it with our own comparators.**
He measures Qwen FP8 at 0.023 and Unsloth NVFP4 at 0.041, a ratio of 1.8. We measure the same
two checkpoints at 0.012798 and 0.092727 on the corrected v3 suite, a ratio of 7.2. Same two
artifacts, same direction of KL, wildly different ratio. Nobody should compute
`ours ÷ theirs`, in either direction, and the earlier community objection that our NVFP4 number
disagrees with Unsloth's is best answered with this: three protocols, three different NVFP4
numbers (0.041 his, 0.0124-0.05818 theirs, 0.0927 ours), each internally consistent.

**4. His mean is a tail statistic and his caption says so.** Mean ÷ median runs 2.6-7.5 across
every family, rising with quality: the better the quant, the more its mean is carried by a thin
set of positions where the reference itself is undecided. His histogram chart shows the same
thing structurally — roughly log-normal distributions whose right tails, not their modes,
separate the good quants. That is the same conclusion our own tail work reached
(`docs/33`), and it is the reason we now publish p99/p99.9/p99.99 with exact exceedance
counts instead of a mean alone.

**5. His protocol runs out of resolution exactly where the interesting builds are.** The
synthetic floor — the BF16 reference perturbed at bf16-rounding scale, seeded — has mean
0.0052 and median 0.0007. His 6.00 bpw point is 0.007 mean, i.e. **1.35x the floor**, and on
the histogram the floor curve overlaps the 6.00 bpw and UD-Q6_K_XL curves. So the top of his
ladder is measured at the edge of what 65,536 positions and an FP16-cached reference can
resolve. Ours is a different regime: 10,480,640 positions, an exactly-deterministic
runtime-repeat floor of 0.000000 across three captures, and a replay qualification floor of
6.54e-04 — which is itself above his median floor, in the other direction. Neither floor
transfers; each bounds its own protocol.

### Where our builds sit on his x-axis

His axis is computable for our checkpoints from the published role manifest, so the *size*
comparison is exact even though the *quality* comparison is not. Summing attention, MLP and
`lm_head` bytes from [`receipts/hydrated-quantization-manifest.json`](../receipts/hydrated-quantization-manifest.json)
and excluding embeddings, vision, MTP and norms — his stated convention — the hydrated build is
**17,837,971,012 B = 16.61 GiB**, which lands between his 5.00 bpw (~15.0 GiB) and 6.00 bpw
(~17.8 GiB) points. That is the whole thesis of a role-asymmetric recipe in one number: K5
gate/up, K6 down, K6 attention and a K6 head buy 6-bit-class *placement* at 5-to-6-bit *cost*,
and against his GGUF ladder that same 16.61 GiB undercuts UD-Q6_K_XL by 6.5 GiB and official
FP8 by 8.5 GiB of decoder weight.

What we still cannot say is where our y-value would land on his chart, because we have never
run his protocol. Run A below is what would settle it; with the GGUF work now done on our own
axis, his qbench is the natural second harness to add, since it already contains every
comparator we care about.

## Pinned artifacts

Repo `unsloth/Qwen3.8-27B-GGUF` at revision **`f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`**;
every URL is `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/<path>`.
Sizes and digests from the repo tree at that revision [HF-GGUF].

| role | file | bytes | GiB | blob sha256 |
|---|---|---:|---:|---|
| 5-bit target (no plain `Q5_K_XL` exists; UD is the only XL 5-bit) | `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 20,218,178,624 | 18.830 | `176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab` |
| 6-bit target | `Qwen3.8-27B-Q6_K.gguf` | 22,884,408,288 | 21.313 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| 8-bit target | `Qwen3.8-27B-Q8_0.gguf` | 29,047,086,048 | 27.052 | `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348` |
| reference, part 1 of 2 | `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` | 49,986,159,616 | 46.553 | `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f` |
| reference, part 2 of 2 | `BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf` | 4,671,576,000 | 4.351 | `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02` |
| **reference total** | two-part BF16 control | **54,657,735,616** | 50.904 | matches their "BF16 54.7GB" prose [U-Q38] |
| 24 GB-class size competitor | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,923,394,624 | 16.692 | `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372` |
| 16 GB-class size competitor | `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13,441,059,904 | 12.518 | `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2` |

The three targets alone are 72,149,672,960 B = 67.19 GiB; adding the BF16 reference makes it
126,807,408,576 B = 118.10 GiB. The corpus adds 4,721,645 B [HF-WT2]. All of these are
serialized disk bytes; none of them is VRAM, resident weights or KV, and the size-class labels
above refer to the card a build is intended for, not to what these files occupy in memory.

## Execution plan: two runs and one control — B and C are done, A is open

### Run A — their protocol, exactly, for cross-citation (open, not run)

1. Fetch the corpus pinned at [HF-WT2] and verify `wiki.test.raw` against
   sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`, 1,290,590 B.
2. Fetch the three targets and the two-part BF16 reference at revision `f1bfb127…` and verify
   every blob digest in the table above.
3. Capture with `llama-perplexity --model BF16 --file wiki.test.raw --ctx-size 512
   --save-all-logits base.bin`; assert `580 chunks`, `296960 tokens` in the log and
   `sizeof(base.bin) == 20 + 580*512*4 + 580*255*248324*2` before scoring anything.
4. Score each target with `--kl-divergence --kl-divergence-base base.bin`, recording
   `Mean/Median/99.9%/Maximum/Minimum KLD`, `RMS Δp` and `Same top p` for each.
5. Assert cross-engine token identity: hash the per-chunk token ids read back from `base.bin`
   against our HF-tokenizer stream over the same file (D10), and publish the comparison.

Output: three rows on the same axis as their Qwen3.5 ladder [U-Q35] and as any third-party
`llama-perplexity` row (bartowski, AesSedai, ubergarm), plus their `Minimum KLD` values for
our own build, which measures their harness floor on our hardware rather than assuming it.

### Run B — our v5 suite, for placement on our axis (**done**, shard 0)

Score the same three GGUFs through `tools/fidelity.py` against the same BF16 teacher, shared
BF16 head, exact float64 two-pass, on the frozen v5 suite
([`receipts/kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json)) so they land in
the same table as hydrated / online K5-K6 / context / official FP8 / K4, with cluster-bootstrap
intervals and paired per-context differences. This requires the llama.cpp-side loader to expose
the final-norm hidden states for the same token ids; the shared head and the suite are
unchanged, so a GGUF row is directly paired against every existing row.

**Run, and reported above.** All three GGUFs plus the BF16 engine-floor control were captured
with [`tools/gguf_capture.cpp`](../tools/gguf_capture.cpp) at the pinned llama.cpp commit and
scored on **shard 0** of the frozen suite — 512 contexts, 1,048,064 positions, the same contexts
every candidate saw — receipt
[`receipts/cross-engine-comparator.json`](../receipts/cross-engine-comparator.json). Two
deviations from the plan as written, both stated in the receipt: it is one shard of ten rather than
the full 10,480,640-position suite, and the comparison is a shard-0 ranking rather than a paired
per-context bootstrap against the ten-shard rows, because those rows were welded from a different
position count. Extending the GGUFs to all ten shards is unrun.

### Control C — isolate D1 on our own data (**done**)

`tools/fidelity.py replay --score-from N` (added in this iteration; replay-time slicing of
already-captured hidden states, so no recapture and no new GPU capture, and one capture set
serves any `N`) restricts every statistic — token mean and median, percentiles, tail
histogram, per-context means, top-1, JSD, exact max, bootstrap — to positions with at least
`N` tokens of left context, and records the choice in a top-level `scored_position_window`
report block (schema `qwen38-scored-position-window/1`; fields `score_from`, `windowed`,
`policy`, `positions_per_context`, `positions_per_context_min`/`_max`,
`capture_positions_per_context`, `dropped_positions_per_context`, `dropped_positions_total`,
`first_scored_position_index`, `min_left_context_tokens`).

The window is declared in the schema string rather than left implicit: an unwindowed run
stays `qwen38-fidelity-report/2` with every pre-existing field unchanged, a windowed run is
`qwen38-fidelity-report/3`, and `tools/kld_aggregate.py` (now
`qwen38-kld-ladder-cumulative/3`, carrying the same block) rejects `/2`-with-window,
`/3`-without-window and any mixed-window set — each window aggregates into its own receipt.
Already-published receipts stay at `/2` and are untouched. One operational consequence:
`tools/kld_ladder.sh` verifies only `/1` and `/2` and requires every row to score the full
context, so it fails closed on a windowed report; the comparability run has to be driven
outside that script.

| setting | retained positions per context | total retained | reproduces |
|---|---:|---:|---|
| `--score-from 0` (default) | 2,047 | 10,480,640 | our published protocol, unchanged |
| `--score-from 256` | 1,791 | 9,169,920 | their **absolute** 256-token left-context floor [LC:541-544] |
| `--score-from 1024` | 1,023 | 5,237,760 | their **proportional** `first = n_ctx/2` rule at our window length |

This was run on the five existing candidates, and it converted D1 from a plausibility argument
into a number on identical data with identical numerics: **-1.3 % to -2.1 %** at their absolute
floor, **-3.9 % to -4.9 %** at their proportional one, uniform enough to change no ordering
([`receipts/scored-window-offset.json`](../receipts/scored-window-offset.json), reported in full
[above](#d1-resolved-the-scoring-window-moves-every-mean-by-at-most-about-5-percent)). Every other delta now has to
explain what remains, which is most of the distance.

## What each run can and cannot establish

**Run A can establish:** the three GGUFs' position in Unsloth's own published series, on their
corpus, their context floor, their reference numerics, their head-in-loop convention, directly
citable next to their Qwen3.5 rows and next to third-party `llama-perplexity` rows for other
quantizers. It also measures their harness's own floor for this model (their `Minimum KLD`
sign) instead of inferring it from someone else's log.

**Run A cannot establish:** anything about our candidates. It has 147,900 scored positions
[LC, INV] against our 10,480,640
([`kld5-suite-manifest.json`](../receipts/kld5-suite-manifest.json)), i.e. 1.41 % of the
evidence; it is English encyclopedic only; it includes each candidate's own
output head, so it cannot be compared to our body-only numbers; its interval is a per-token
SEM (D7); and it cannot produce a comparable number for NVFP4 or FP8 at all, since neither
runs under `llama-perplexity` (D12). It also cannot resolve differences below ~5e-5 nats/token,
which is where their own Q8-class rows live (D3).

**Run B established** (shard 0, receipt above): where the GGUFs sit against every candidate we
ship, on one corpus, one reference, one shared head, one uncertainty model, with contamination
excluded by whole-document pre-exclusion. This is the only run that answers the F2 question as
asked — same dataset, same size range, all candidates — with one deviation from the plan: it
ranks on one shard instead of pairing per context against the ten-shard rows.

**Run B cannot establish:** a number citable against Unsloth's table, because it is not their
protocol; nor anything about llama.cpp's own serving numerics beyond the prefill path used for
capture; nor the head-inclusive quality of a GGUF, unless the GGUF's own dequantized head is
replayed separately as the candidate head (D5).

**Control C established:** the size and sign of D1 alone, exactly, on our data — 1.3-2.1 % at
their absolute context floor, 3.9-4.9 % at their proportional one, no ordering changed.

**Control C cannot establish:** D2 through D11. Agreement between our windowed mean and their
published mean would be consistent with the remaining deltas cancelling, not evidence that they
are absent; disagreement bounds the residue that D2-D11 must account for.

Two of the three are done. Run A is what remains, and it is the only one of the three that yields
a number citable against Unsloth's own table; B and C have already produced our-axis placement for
the GGUFs plus a measured explanation for part of the distance between the two protocols. That is
the entire claim. Neither
protocol is being corrected here; the only defect worth naming is publishing a KLD without the
corpus, position count, context floor, reference precision and engine that produced it, and
that is the defect this document exists to avoid on our side.

## Sources

| tag | source |
|---|---|
| [U-Q38] | `unsloth.ai/docs` `models/qwen3.8` as published 2026-08-15 (full export `llms-full.txt`, 1,765,002 B): NVFP4 KLD/top-1 table, "Quantization Analysis" prose, `q38_top1_vs_size.png`, "Unsloth Dynamic V3.0 (preview)" labelling, "BF16 54.7GB" |
| [U-Q35] | `unsloth.ai/docs` `models/qwen3.5/gguf-benchmarks`: "Full Benchmarks" table (Qwen3.5-35B-A3B, wikitext-2 @ ctx 512) and the 2026-03-05 re-issue note |
| [U-Q36] | `unsloth.ai/docs` `models/qwen3.6`: Qwen3.6-27B MLX KLD distribution table |
| [U-DSV4] | `unsloth.ai/docs` `models/deepseek-v4`: *"Reference = official weights. Perplexity and KL-divergence over wikitext-2 at ctx 512 on 4x B200."* |
| [U-DYN2] | `unsloth.ai/docs` `basics/unsloth-dynamic-2.0-ggufs`: calibration/eval separation, `Calibration_v3`/`Calibration_v5` usage, instruct-model calibration argument |
| [U-LOG-B] | `huggingface.co/unsloth/Qwen3.5-35B-A3B-Experiments-GGUF` @ `eb6afb2fde8e3b4310c66e6e5d970215e936c9a3`, `KLD_Logs/unsloth/base/BF16_base_logits_ctx512.log`: verbatim capture command, `580 chunks, n_ctx=512, batch_size=16384, n_seq=32`, `296960 tokens`, `Final estimate: PPL = 6.5353 +/- 0.04158`, build `8164 (b68d75165)` |
| [U-LOG-Q5] | same repo/revision, `KLD_Logs/unsloth/compare/Qwen3.5-35B-A3B-UD-Q5_K_XL_ctx512.log`: `Mean PPL(Q) 6.548881`, `Mean PPL(base) 6.532929`, `Mean KLD 0.006949 ± 0.000073`, `99.9% KLD 0.236021`, `Median KLD 0.002997`, `Minimum KLD -0.000050`, `Maximum KLD 5.536102`, `Same top p 96.335 ± 0.049 %` |
| [U-LOG-Q8] | same repo/revision, `KLD_Logs/unsloth/compare/Qwen3.5-35B-A3B-UD-Q8_K_XL_ctx512.log`: `Mean KLD 0.002562 ± 0.000052`, `Minimum KLD -0.000140`, `Same top p 97.958 ± 0.037 %` |
| [LC] | `github.com/ggml-org/llama.cpp`, `tools/perplexity/perplexity.cpp` (master as fetched 2026-08-15, 82,240 B). Line references used above: 79-100 uint16 log-prob encoding and the 16-nat clamp; 224-240 the KLD sum, the `-16.f` gate and `Same top p`; 541-544 `const int first = n_ctx/2`; 616-624 `process_logits(..., n_ctx - 1 - first, ...)`; 1752 `nv = 2*((n_vocab + 1)/2) + 4` |
| [HF-GGUF] | `huggingface.co/unsloth/Qwen3.8-27B-GGUF` @ `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` (2026-08-15T05:48:44Z, apache-2.0, `base_model: Qwen/Qwen3.8-27B`); file sizes and blob sha256 from the repo tree at that revision |
| [HF-WT2] | `huggingface.co/datasets/ggml-org/ci` @ `927b3642933080f1b0e811e2f916e14c292992f9`, `wikitext-2-raw-v1.zip` (4,721,645 B, sha256 `ef7edb…5435a11`; member `wiki.test.raw` 1,290,590 B, sha256 `173c87…9e7dd08`) |
| [HF-WT2-DS] | `huggingface.co/datasets/Salesforce/wikitext` @ `b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-2-raw-v1`, licences cc-by-sa-3.0 + gfdl |
| [HF-NVFP4] | `huggingface.co/unsloth/Qwen3.8-27B-NVFP4` @ `16b6615af3548b88e2d8e382457bc705b00479cf`; `tokenizer.json` sha256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`, used for the corpus reproduction in [INV] |
| [INV] | our own offline reproduction, CPU only, no GPU and no repo receipt yet: `wiki.test.raw` tokenized with [HF-NVFP4]'s tokenizer at `add_special_tokens=False` gives 297,193 tokens → 580 chunks → 296,960 processed → 147,900 scored; and the digitization of `q38_top1_vs_size.png`. Recorded in the investigation transcript `agent://UnslothProtocol`. Run A above is what turns these into receipts |
