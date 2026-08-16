# What is left, ordered by evidence value

State on 2026-08-15 after four published builds, two independent reviews, one external
RTX 5090 test, a source-disjoint qualification, a contamination re-audit, and paired
downstream smoke testing. Priority is the claim each item can close, not implementation cost.

## Current measured frontier

KLD is `KL(BF16 || candidate)` through one shared BF16 head on the overlap-corrected
127-context v3 subset. Capacity profiles are deliberately not treated as interchangeable.

| profile | resident weights | corrected mean KLD | demonstrated/configured context | hardware |
|---|---:|---:|---:|---|
| hydrated K6 | 20.31 GiB | **0.007172** | 185,600, MTP-3 | physical RTX 5090 |
| online K6 | 20.32 GiB | 0.007945 | 185,600, MTP-3 | physical RTX 5090 |
| context + int8 input | **18.41 GiB** | 0.009459 | **262,144, MTP-3, 8.4 MP cap** | physical RTX 5090, utilisation 0.955 |
| K4 | 17.89 GiB | 0.029679 | **262,144, MTP-3** | physical RTX 5090 |
| official FP8 | 28.51 GiB | 0.012798 | not measured here | — |

The context profile now combines native context, speculative decode and multimodality, and it is
**hardware-qualified**: on one physical RTX 5090 it allocates **265,122 KV tokens** at 262,144
with MTP-3 and the full 8.4 MP ceiling, retrieves exactly from 261,794 text tokens, answers code
plus colours exactly from a 236,824-token prompt containing a 3,072 × 2,304 image, and decodes at
a median **107.56 tok/s** warmed at C1 — all seven gates passing at
`--gpu-memory-utilization 0.955` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(`receipts/qualification-5090-context.json`). The earlier 266,612-token, 98.72 tok/s result
remains valid as the engine-budget proof it was (`receipts/native-mtp-8mp-amendment.json`); the
two throughput figures come from different GPUs and are never differenced.

The frozen post-selection ranking also survives the offset-independent correction. After
excluding every context from four qualification source documents with an exact calibration
12-gram, hydrated / online / context score 0.003093 / 0.003455 / 0.003990 against FP8
0.005891, each winning 36/36 paired contexts. Absolute KLD is suite-specific.

## Priorities added from public feedback, 2026-08-15

Four independent readers attacked the same weak point: the fidelity claim is strong but
thin, and the comparator set is narrow. They now head the list: the physical RTX 5090
qualification that used to outrank them is closed (rank 1).

### F1 — evidence volume: 278,392 scored positions is not enough

"Do I understand this correctly that you evaluated only on 136 traces and 278k output
tokens?" The headline KLD rests on 136 analysis contexts. Bootstrap intervals are computed
over 39 source clusters, so the interval is honest, but the absolute volume invites the
objection that the estimate is fragile.

Built for this: `tools/fetch_corpus_v5.py` (941 documents, 70.3 MB of held-out public text)
and a v5 suite of **5,120 contexts x 2,047 = 10,480,640 scored positions** over 842 source
clusters, token-disjoint from v4 and with zero calibration 12-gram overlap by construction
(whole documents with any hit are excluded before selection). `tools/suite3.py` now advances
each document cursor to the exact end of the emitted context, so corpus capacity is
token-bound rather than window-bound.

`tools/kld_ladder.sh` walks it in 512-context shards because one shard of six models is
~64 GB of hidden states and `/var/tmp` holds ~135 GB: capture six models, replay the five
candidates, verify, delete, next shard. `tools/kld_aggregate.py` welds the per-shard reports
into cumulative receipts at the **1M / 2M / 5M / 10M** checkpoints. Escalate until the run
stops being reasonable on this hardware, and publish the point at which it stopped.

### F2 — comparator breadth: FP8 is a throughput format, not the quality ceiling

"A comparison to Q8 instead of FP8 would've looked differently... someone would need to run
a KLD/top-1 test on the same dataset for all of them in the same size/performance range."
That was correct. The GGUF half of it is now **measured**; the rest is still open.

**Done — the three GGUFs are in our suite.** `Q8_0`, `Q6_K` and `UD-Q5_K_XL` from
`unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` were captured under
llama.cpp pinned at commit `ece963f41b0b02d7a0d61436ae365762c073a4c8` through
[`tools/gguf_capture.cpp`](../tools/gguf_capture.cpp), which reads the post-final-norm state —
the same mathematical point the vLLM hook takes — with bf16 rounding verified bit-identical to
torch on 2,012,449 probe values; manifests from
[`tools/gguf_manifest.py`](../tools/gguf_manifest.py), build from
[`tools/build_llamacpp.sh`](../tools/build_llamacpp.sh), and every capture manifest carries the
GGUF digest and the llama.cpp identity. They were then scored against the same BF16 teacher,
through the same shared BF16 head, on **shard 0 of the v5 suite — the same 512 contexts and the
same 1,048,064 scored positions every other candidate saw**. Receipt
[`receipts/cross-engine-comparator.json`](../receipts/cross-engine-comparator.json),
per-candidate reports `receipts/gguf-report-{q8_0,q6_k,q5_k_xl}.json`.

| candidate | engine | measured mean KLD | net of engine floor | top-1 | p99.9 | serialized |
|---|---|---:|---:|---:|---:|---:|
| GGUF `Q8_0` | llama.cpp | 0.001087 | ~0.000579 | 98.53 % | 0.0351 | 27.05 GiB |
| GGUF `Q6_K` | llama.cpp | 0.002035 | ~0.001528 | 97.98 % | 0.0794 | 21.31 GiB |
| hydrated EXL3 | vLLM | **0.002700** | n/a, same engine | 97.80 % | 0.1313 | 20.12 GiB payload |
| online K5/K6 EXL3 | vLLM | **0.003141** | n/a, same engine | 97.61 % | 0.1447 | — |
| context EXL3 | vLLM | **0.003409** | n/a, same engine | 97.55 % | 0.1632 | 19.27 GiB payload |
| GGUF `UD-Q5_K_XL` | llama.cpp | 0.004444 | ~0.003936 | 97.20 % | 0.2144 | 18.83 GiB |
| official FP8 | vLLM | 0.005197 | n/a, same engine | 96.92 % | 0.2440 | 28.51 GiB resident |
| K4 EXL3 | vLLM | 0.010345 | n/a, same engine | 95.91 % | 0.5576 | — |

The **cross-engine floor** is measured the same way, not assumed: the unquantized BF16 GGUF
against the vLLM BF16 reference, identical tokens, identical shared head, the same 512 contexts,
reads **0.000507** mean, 99.07 % top-1, p99.9 0.0113
([`receipts/gguf-report-engine-floor.json`](../receipts/gguf-report-engine-floor.json)). Every
GGUF row above contains that term; no vLLM row does. KL is not additive, so the net column is an
estimate, not an identity — the measured GGUF value is an upper bound and the net figure is the
naive lower one. The two `—` cells are the two builds that ship BF16 attention for the runtime to
encode at load, so their disk bytes are not a like-for-like payload; the payload figures are
`immutable_payload_bytes` from
[`receipts/collection-index.json`](../receipts/collection-index.json) (hydrated 21,610,916,123 B =
20.127 GiB, context 20,696,033,532 B = 19.275 GiB; the table truncates both to two decimals) and are
serialized bytes, never VRAM. The FP8 figure is resident
weights, labelled as such, because that artifact has no payload row of ours.

The p99.9 column is each report's **exact** shard-0 value; the tail table in F5 below quotes the
bin-bounded cumulative estimate from the 560-bin histogram (bins about 5.6 % wide), which is why
hydrated reads 0.1319 there and 0.1313 here — the exact value lies inside the bin its estimate
names. The two differ by construction, not by measurement.

**What it found, including the half that is not flattering.**

- **At the 6-bit operating point we lose.** GGUF `Q6_K` is genuinely better than our best build:
  0.001528 net at 21.31 GiB against hydrated's 0.002700 at 20.12 GiB of payload. That is a
  1.186 GiB-larger file measuring better, on our own suite, and it is the first published
  measurement in this project where an off-the-shelf artifact beats the recipe.
- **At the 5-bit operating point we win.** The context edition reads 0.003409 at 19.27 GiB
  against `UD-Q5_K_XL`'s 0.003936 net at 18.83 GiB — about 13 % better fidelity for
  **0.445 GiB** more payload.
- **`Q8_0` is the fidelity leader**, 0.001087 at 27.05 GiB, and its measured value is only about
  twice the engine floor, so its own number sits near the resolution limit of any cross-engine
  comparison: the net column is an estimate and not an identity, so **no ordering closer than a
  factor of two should be pressed against `Q8_0`**. Anyone who can spare 27 GiB of weights and does
  not need a long context on a 32 GB card should use it.
- **Official FP8 is a weak bar.** Every GGUF point at or above 5 bits beats it. Our published
  "34-48 % below FP8" is true and a weaker achievement than it sounds.
- **K4 is the weakest point in the table**, 0.010345 mean and 0.5576 at p99.9 — worse than every
  GGUF measured here and worse than official FP8.
- So the format advantage at this bitrate is real at 5 bits, negative at 6 bits, and far short of
  a full bit. turboderp's own chart reads "about a bit ahead" on *his* protocol
  ([35](35-external-protocol-comparability.md)); that is a difference between protocols, not a
  contradiction to settle by dividing one by the other.

**Scope, stated with the result.** This is text-only teacher-forced fidelity on one shard, and it
says nothing about serving 262,144 tokens with vision and MTP on a 32 GB card, which is where
these artifacts actually differ. llama.cpp KV-quant behaviour, prefill and decode speed are
separate axes and none of them is measured here. Shard 0 is one tenth of the suite and close to it: over
all 10,480,640 positions the five vLLM means read 0.002760 / 0.003210 / 0.003509 / 0.005294 / 0.010604,
**1.9-2.9 % above** these shard-0 values, ordering unchanged
([`receipts/kld5-10M-hyd.json`](../receipts/kld5-10M-hyd.json) and siblings); the GGUFs have no ten-shard
equivalent and extending them is unrun.

**One protocol objection bounded at the same time.** The largest structural difference between our
protocol and `llama-perplexity`'s is which positions each scores: theirs floors every scored
position at 256 tokens of left context, ours scores from position 0. Re-scoring our own captures
under that restriction — no new capture, `tools/fidelity.py replay --score-from N` — lowers every
candidate's mean by **1.3-2.1 %** at a 256-token floor and **3.9-4.9 %** second-half-only,
uniformly enough to change no ordering
([`receipts/scored-window-offset.json`](../receipts/scored-window-offset.json), fifteen reports
`receipts/kld5-window-{hyd,k5k6,ctx,fp8,k4}-from{0,256,1024}.json`). So the external protocol's
scoring floor explains at most about 5 % of any cross-protocol gap, and none of the ordering above.

**Still open in F2:**

- **Their protocol, for cross-citation.** wikitext-2 raw test at ctx 512 under
  `llama-perplexity --kl-divergence`, so the three GGUFs also carry a number citable next to
  Unsloth's own published rows. Run A of
  [35-external-protocol-comparability.md](35-external-protocol-comparability.md); not run.
- **Stock EXL3 uniform-bitrate controls**, at least `turboderp/Qwen3.8-27B-exl3` 5.00bpw
  (`a35e75a7`) and 6.00bpw (`d32ba0bb`), to separate our role-aware allocation from EXL3 itself.
  Not run — and after the GGUF result this is the more interesting of the two, because it is the
  control that says whether the 6-bit loss is our allocation or the format.
- **An ik_llama comparator.** Reader-suggested: `cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF` under
  `ik_llama.cpp` (r/LocalLLaMA megathread `1voojjz`, comment `p3uk494`, "Try this with
  ik-llama"). A name and a link only — no KLD, no top-1, no protocol. It is an **unmeasured
  candidate, not a result**, and nothing about it may be published as a comparison until it is run
  through the harness above.
- **An NVFP4-NInfer comparator.** `Ostfralla/Qwen3.8-27B-NVFP4-NInfer` on the NInfer engine
  (`github.com/Neroued/ninfer`), announced as a single-RTX-5090 native-262,144 build in the same
  megathread, comment `p3xf3qf`. It carries no KLD either: its fidelity evidence is HumanEval+
  152/164 and AIME25+AIME26 55/60, identical to the int4 artifact it was benchmarked against, and
  its 1.56-1.98x is wall clock. Two traps for whoever writes it up: its "reconstruction error
  0.09471 → 0.09470" is a **weight-space** error, numerically adjacent to our NVFP4 KLD (0.094978
  v3, 0.092727 corrected) and therefore certain to be misread as confirmation of it — it is not;
  and its "16.8 GiB" is a weight payload, not measured resident weights, so it must not be placed
  on the same axis as our 20.31 / 20.32 / 18.41 / 28.51 / 17.89 GiB resident figures. It is
  Blackwell-only and needs a patch its author says is not upstream, so it is not reproducible from
  a released tree today.

Two corrections to what this section said earlier stand unchanged. **First, the NVFP4 range was
wrong here.** The same checkpoint reads 0.094978 on our v3 suite and 0.092727 after contamination
correction ([`receipts/v3-report-nvfp4-analysis.json`](../receipts/v3-report-nvfp4-analysis.json),
[`receipts/analysis-v3-contamination-corrected.json`](../receipts/analysis-v3-contamination-corrected.json)),
while Unsloth's own published per-corpus rows are **0.01628** zh, **0.02600** code, **0.03993**
refgen, **0.05818** chat and **0.0124-0.0155** ja/ko/ru/es — a range of **0.0124-0.05818**, not
the "0.016-0.068" written here before. Those rows carry no corpus identity, token count, context
length, reference precision, KL direction or engine, so they are not reproducible by us or by
anyone else. **Second, there is no upstream GGUF ladder to compare against.** Unsloth publishes
no per-quant KLD table for Qwen3.8-27B GGUFs at all — one prose figure ("82.5% accuracy (IQ2_XXS
9GB)"), one top-1-versus-size chart, and "More benchmarks coming soon!" — so the `Q5_K_XL` /
`Q6_K` / `Q8_0` comparison could not be closed by citation and had to be run by us, which is what
the receipt above is. Their nearest published exact ladder is Qwen3.5-35B-A3B on wikitext-2 at
ctx 512 (Q5_K_XL 0.0069, Q6_K_XL 0.0041, Q8_K_XL 0.0026), a scale anchor for their protocol and
not a prediction for these artifacts. Protocol identity, the ordered delta list, pinned artifacts
and the execution plan are in
[35-external-protocol-comparability.md](35-external-protocol-comparability.md).

Report confidence-conditioned buckets alongside the mean: a KLD below 0.01 is the threshold
readers already associate with "practically BF16", so the distribution shape matters as much
as the mean.

**First item-paired public benchmark now exists, official FP8 included.** The pinned MMLU-Pro
70-question run scores all six models on identical prompts against the BF16 control
(`receipts/public-capability-{bf16,ctx,k4,hyd,fp8,k5k6}.json`): BF16 **57/70**, context
**58/70**, K4 **57/70**, official FP8 **56/70**, hydrated **56/70**, online K5/K6 **55/70**. So
the comparator no longer exists only on the KLD axis. At 70 items every candidate interval
overlaps every other, so the run separates nothing; the bar it missed, and why more items are the
fix, is in the rank-4 section below.

### F3 — VRAM-class SKUs: decided. 24 GB go, 16 GB design study

**Decided, with arithmetic, in [`receipts/vram-class-verdict.json`](../receipts/vram-class-verdict.json)**;
the design it rests on is [34](34-vram-class-profiles.md), whose new Verdict section summarises
both classes. The reader asks that opened this item — a 24 GB mid-point and "make a version for
the 16GB of RAM or VRAM and you'll be a god" — are answered as follows.

**24 GB: go, and no new checkpoint is required.** The published `-context` edition's resident
weights are *measured* at 18.41 GiB (`receipts/native-mtp-8mp-amendment.json`, as run; the
physical 5090 logged 18.19 GiB for the same configuration, and the verdict uses the larger).
On the constants the physical qualification measured — 31.4 GiB usable of a 32 GB board,
`--gpu-memory-utilization` 0.955, 1.78 + 0.27 + 0.45 = 2.50 GiB of non-KV overheads
(`receipts/qualification-5090-context.json`) — a 24 GB board leaves
`24 × 0.98125 × 0.955 − 2.50 − 18.41 = 1.58 GiB` of KV **[P]**. At the per-token cost derived from
that receipt's own measured pool — its 9.28 GiB holding 265,122 tokens, so
`(9.28 − 0.20) GiB ÷ 265,122 = 36,773.9 B/token` — that is 40,293 tokens
**[P]**, published at **32,768** with MTP-3 and fp8 KV, or **45,056** with MTP off, under the
≥15 % headroom rule. Fidelity is the already-measured **0.003409** on shard 0
(`receipts/kld5-1M-tail-ctx.json`) and 0.003509 over 10,480,640 positions
(`receipts/kld5-10M-ctx.json`), carried unchanged because it is the same weights: the class adds
**no new fidelity risk**. So this is a serving profile over an existing artifact plus one
qualification —
not a conversion project. `--max-model-len 40960`, which [34](34-vram-class-profiles.md) §5.3
predicted at utilisation 0.97, is retired: at 0.955 it needs 1.603 GiB against 1.580 available.

**16 GB: no-go as a SKU; published as a design study.** The byte law puts the cheapest
multimodal build that keeps the MLP stack at 4 bits at 13.58 GiB resident **[P]** against a 12.70
GiB budget — **0.88 GiB over before a single KV byte**, and 1.09 GiB over at the measured 0.955
utilisation. Every remaining path is sub-4-bit, and **no width below 4 bits has ever been measured
for KLD in this family**. The S16-V candidate is all prediction: 11.94 GiB resident **[P]**,
16,384 tokens MTP-off **[P]** at [34](34-vram-class-profiles.md) §3's constants and 12,288 **[P]**
re-derived at 0.955, with fidelity unknown — the nearest measured neighbour is K4 at 0.010604
(`receipts/kld5-10M-k4.json`) with p99.9 0.5555 (`receipts/kld5-1M-tail-k4.json`), already the
worst published candidate, and S16-V is below it on every role. Two
reader reports run the same way and are not ours: a ~12 GB `IQ3_XXS` on a 16 GB RTX 5070 Ti gives
"less than 5 tok/s" at 64K-128K (megathread `1voojjz`, `p3ui0np`), and `UD-Q4_K_XL` on a 24 GB
4090 "only leaves me with about 18.4k context" (`p3vfwqh`). A build we cannot measure, on hardware
we do not have, at a width nobody has scored, is a design study — not a SKU. No 16 GB artifact, no
16 GB card row, no 16 GB context length as a capability, and no fidelity number for any 16 GB
profile.

**Genuinely open, and all that is open:**

- **Measure one sub-4-bit width's KLD before committing to a 16 GB SKU.** One conversion plus
  shard 0 of the v5 suite (512 contexts, 1,048,064 positions, the shared BF16 head), with the tail
  row and exceedance counts. **No 16 GB card is needed for this**, it is the condition that flips
  the no-go, and it replaces [34](34-vram-class-profiles.md) §6.4's "0.03-0.10" — a range with a
  shape, not an estimate — with a number.
- **Obtain or emulate a 24 GB card for a real qualification, not an engine-budget proof.** The
  24 GB go is a go to *qualify*: "predicted 40,293 KV tokens" and "allocated 40,293 KV tokens" are
  different sentences. Today's 5090 result showed exactly why an engine-budget proof on a larger
  card can mislead — the rental proof at 30.24 GiB inside a 95.6 GiB board said utilisation 0.97
  was fine, and the physical card then refused the combined 236,824-token plus 7 MP request at that
  utilisation, because vLLM spends every freed byte on KV and left the vision tower without the
  ~62 MiB of transient it needed, while lowering `max_pixels` made it strictly worse. A budget
  carved out of a larger card keeps real headroom behind it and cannot see that failure mode, so an
  emulated 24 GB qualification inherits the caveat verbatim and the physical run stays P0. The
  supported-hardware tuple still has to be stated rather than assumed: everything measured here is
  SM120, TP1, driver 595.58.03, and the 24 GB class spans more than one architecture.

### F4 — collection presentation

Adopted from the external ladder review: one immutable machine-readable collection index
(artifact commit, base commit, converter commit, whole-tree bytes, tensor-payload bytes,
measured resident bytes, per-role precision, runtime image and patch digests, suite digest),
the same compact ladder rendered in every card, and strict size terminology — serialized
bytes are never called VRAM.

### F5 — everything else the release megathread asked for

Harvested from 260 comments across the r/LocalLLaMA release megathread (`1voojjz`) and our
own KLD post (`1vp15wq`); the megathread and post are 403 to plain fetches, so the harvest of
record came from the Arctic Shift `comments/search` API, paginated. Items are grouped by what
they would change.

**Statistics.** "Avg KLD and top-1 are meaningless. What 99.9% tail looks like?" Publish
p99/p99.9/max per candidate, not only the mean. **Landed and measured.** `fidelity.py replay`
counts every scored position into a fixed log-spaced histogram — 560 bins from 1e-12 to 1e2
nats, plus explicit zero/underflow and overflow buckets — and stores it as `kld_tail` alongside
the shard's own exact p50/p95/p99/p999 and exact maximum, which bumps the report schema to
`qwen38-fidelity-report/2`. `tools/kld_aggregate.py` sums those counts across shards and
publishes cumulative p50/p95/p99/p999/p9999 as the bin interval that provably contains each one
(one bin, 5.6 % wide, wherever the tail is dense), plus the exact global maximum and exact
exceedance counts at 1e-4 … 1 nat. Counts recombine, percentiles do not — which is why the
histogram had to exist before any tail could be published at all.

**The tail is now on the record.** Shard 0 of the v5 suite was re-captured and re-replayed with
the `/2` harness — the same 512 contexts every candidate saw, **1,048,064 scored positions**
each — and welded into `receipts/kld5-1M-tail-{hyd,k5k6,ctx,fp8,k4}.json`
(`qwen38-kld-ladder-cumulative/2`):

| candidate | mean | p50 | p95 | p99 | p99.9 | p99.99 | exact max | above 0.1 | above 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| hydrated | 0.002700 | 0.00109 | 0.0082 | 0.0276 | **0.1319** | 0.463 | 3.735 | **0.1534 %** | 0.00219 % |
| online K5/K6 | 0.003141 | 0.00128 | 0.0099 | 0.0321 | **0.1446** | 0.498 | 5.507 | **0.1820 %** | 0.00200 % |
| context | 0.003409 | 0.00135 | 0.0107 | 0.0357 | **0.1642** | 0.587 | 3.749 | **0.2287 %** | 0.00305 % |
| official FP8 | 0.005197 | 0.00202 | 0.0167 | 0.0531 | **0.2438** | 0.812 | 5.296 | **0.3912 %** | 0.00592 % |
| K4 | 0.010345 | 0.00320 | 0.0332 | 0.1194 | **0.5555** | 1.870 | 7.565 | **1.2604 %** | 0.03807 % |

The ordering at every measured quantile matches the ordering of the means, so for these
candidates the mean is not hiding a worse tail: every EXL3 K5/K6-class build has a lighter tail
than official FP8 at every quantile, and K4 is worse than FP8 at every quantile.

**Remaining gap, precisely.** The 10.48 M-position ladder was captured with the pre-histogram
harness and emits `/1` reports, which carry no histogram; its tail is published per shard with
the exact global maximum via `kld_aggregate.py --allow-legacy-no-tail`, and the receipt says in
`not_aggregable` why there is no cumulative percentile. A mixed `/1` + `/2` shard set is
rejected rather than summed into a tail that would silently describe only part of the run. Its
hidden states are deleted, so it cannot be retrofitted. **Cumulative histograms over all ten
shards therefore require re-running the other nine shards with the `/2` harness — about 6 hours
of GPU time, and not yet done.** Until that runs, the published tail is one shard of ten
(1,048,064 of 10,480,640 positions) with bin-bounded quantiles, and the 10 M receipts stay the
authority for full-run means, bootstrap intervals and paired results.

**Long context.** Every scored position today comes from a 2,048-token window. Two readers
asked for 64k/128k behaviour, and one reports measured instruction-following collapse at Q3
beyond 40k. Long-window KLD is a separate suite, not a rescale of this one.

**Degeneration.** Two independent Q4 reports describe looping until the context is exhausted.
Mean KLD cannot see it; the downstream suite needs an explicit repetition/non-termination
check.

**Sampler and effort pinning.** Measured in-thread: MTP acceptance moves with temperature and
with reasoning effort, and some front-ends silently downgrade the effort they were asked for.
Every fidelity and throughput figure we publish must carry temperature, top-p/top-k and
reasoning effort. `high` is not a valid effort on this model; only `low`, `medium`, `xhigh`.

**Serving answers we owe.** An unanswered question on our own thread asks whether the
Gilded Gnosis vLLM EXL3 path supports q4 KV cache. Read out of the pinned r34 build, the
accepted `--kv-cache-dtype` set is `auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2,
fp8_inc, fp8_ds_mla, nvfp4_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc,
turboquant_3bit_nc, int4_per_token_head, int8_per_token_head, fp8_per_token_head, nvfp4`, so
sub-8-bit KV exists in several flavours — but every number we publish used `fp8`, and none of
the 4-bit schemes has been measured here for KV capacity, retrieval or KLD. The honest card
answer is the flag list plus that gap, and the fix is a KV-dtype sweep (fp8 versus
`int4_per_token_head` versus `turboquant_4bit_nc`) reporting allocated KV tokens, needle
retrieval at native length and KLD delta on the same checkpoint. That sweep is also the
cheapest lever for the 16 GB and 24 GB classes, where KV is the binding constraint.
A second reader reverse-engineered the `-context` spec sheet
(fp8 KV, native 262,144, roughly 200-215k with MTP, LMCache-compatible) and asked for
confirmation: publish it as a table.

**Throughput next to fidelity.** The 5090 audience compares against NVFP4 and 5.5-bit vLLM
formats on tok/s. Each card needs single-request and concurrent throughput at its native
context with MTP on and off, measured on the same machine as its fidelity number.

**MTP acceptance regression.** Five independent reporters see 3.8 acceptance at roughly
60-70 % where 3.6 was 80-90 %, across llama.cpp and vLLM. Reproduce on our checkpoints at
pinned temperature and publish acceptance per SKU against the in-thread BF16 reference
(MTP-3 optimal, MTP-4 regressing).

**Prefill, not just decode.** A 196k-token agentic turn spends ~75 s in prefill; prefix-cache
reuse dominates that workload. Our LMCache path is real and unpublished.

**Per-tensor bit maps.** Readers now inspect how a quant spends its bit budget before
downloading. Mixed precision is our differentiator and it is currently only prose.

**Non-CUDA tiers.** MLX/Apple and ROCm requests recur; EXL3 serves neither. One explicit
unsupported line per card prevents the question repeating.

**Ship the client config.** Three readers hand-rolled an opencode/pi model block for this
model. Ship a correct one with the valid effort variants, 262,144 context and image modality.



## Closed by the audit

- The fixed-stride contamination claim was wrong. Every normalized 12-token position (Unicode
  word or Han/Kana character) is now scanned; v3 excludes two affected documents (nine
  analysis contexts) and v4 excludes four affected qualification documents (six contexts).
  A sliding five-token receipt measures
  lightly edited overlap. No ranking changes.
- Capture resume now fails closed on missing or modified files; replay refuses an incomplete
  selected context set. The frozen corpus builder fails on language shortfalls.
- The deterministic task harness now executes code in a bounded subprocess with a strict
  builtin allowlist and validates every hidden case. BF16, all four EXL3 profiles and official
  FP8 each pass **40/40 with zero regressions**. A strict final-answer rescore corrects exact
  BF16 agreement to hydrated 35/40, online K5/K6 34/40, FP8 34/40, K4 33/40 and context
  32/40; the original diagnostic compared extracted test summaries for code tasks.
- The draft-embedding int4 claim was invalid and is retracted. vLLM aliases the draft input
  table to the target after load; the earlier comparison never exercised int4 at inference.
- Native MTP was not fundamentally blocked by model weight. Halving the image profiler ceiling
  from 16,777,216 to 8,388,608 pixels lowers peak activation enough to fit MTP-3 and native
  262,144 inside the same 30.24 GiB engine budget while accepting a tested 7.08 MP image.
- PRs #312, #314, #316, #318 and #319 are open as of this state. Cards no longer imply that
  their patches exist in the pinned r34 image.

Receipts: `analysis-v3-contamination-corrected.json`,
`qualification-v4-contamination-corrected.json`, `near-duplicate-v3.json`,
`near-duplicate-v4.json`, `task-retention-v2-summary.json`,
`task-retention-v2-strict-rescore.json`, and `native-mtp-8mp-amendment.json`.

## Queue A — finish on this RTX PRO 6000 rental

Use paid time only for work that the 31.39 GiB RTX 5090 cannot perform, chiefly live BF16
Qwen3.8-27B controls. Preserve each control as a reusable reference so candidate-side runs
can move to the free local card.

| rental rank | global rank | investigation | why it belongs here | closeout before moving on |
|---:|---:|---|---|---|
| R1 | prerequisite | freeze and publish this session | `/var/tmp` models, rootfs, logs and caches are ephemeral | Git commit plus pushed receipts/cards/tools; no evidence only on rental disk |
| R2 | 4 | public capability BF16 baseline | the 55.6 GB BF16 model cannot reside on a 32 GB card | content-hashed prompts, raw BF16 outputs and scorer receipt |
| R3 | 5 | real multimodal BF16 baseline | full BF16 target plus large image activations need the 96 GB card | OCR/chart/document/video BF16 control with original media hashes |
| R4 | 6 | BF16 context-quality control | establish the longest same-model loss/reasoning curve BF16 can actually serve | 32k/64k/maximum-fit token-counted reports; never extrapolate to 262k |
| R5 | 13 | BF16 safety/control side | candidate regressions need a frozen base-model response, not a policy assumption | raw paired-set BF16 outputs and item labels |
| R6 | 9 | conversion error measurements | source BF16 plus converter workspaces are already present and proven here | immutable converter log, `args.json`, per-module adjacent-width errors |
| R7 | 10 | large-shape kernel prototypes | the 96 GB card can retain reconstructed weights and broad shape sweeps without displacing evidence | reproducible microbench plus end-to-end prefill/KLD gate |

Run R2–R5 from one immutable prompt/media bundle and one BF16 server identity where possible.
Do **not** spend rental time repeating quant-only 5090 tests. Before the rental ends, copy
every accepted or negative result into `receipts/`, update the corresponding section here and
`PROGRESS.md`, commit, and push `main` to
`https://github.com/malaiwah/qwen38-27b-exl3`.

## Queue B — free local RTX 5090 32 GB handoff

The local session starts here after pulling the pushed `main`; it must not infer state from
this rental's `/var/tmp` paths.

| local rank | global rank | investigation | first action | push target |
|---:|---:|---|---|---|
| L1 — **done** | 1 | physical RTX 5090 qualification — **CLOSED 2026-08-16**, all seven gates pass at utilisation 0.955 | — | `receipts/qualification-5090-context.json` plus per-process server logs, with the corrected cards |
| L2 | 11 | image-cap/concurrency frontier | after L1, sweep 4.2/6.3/8.4/10.5 MP at sequence counts 1/2 | frontier receipt and card-supported ceiling |
| L3 | 7 | lifecycle/cache reliability | run cold/warm/corrupt/interrupted/restart cases against the L1 runtime | transition matrix and negative logs |
| L4 | 2 | immutable production image | image built and module-verified inside the image (`localhost/vllm:gg-r34-patched`); the L1-equivalent serving smoke is still to run | Dockerfile/source map, OCI digest, SBOM and simplified card commands |
| L5 | 8 | fair speculative matrix | run every quant artifact at MTP off/depth 1/depth 3, C1/C4/C8 | raw three-run matrix and revised throughput claims |
| L6 | 4–6, 13 | candidate sides of paired quality tests | consume the content-hashed rental BF16 bundles; run quant profiles with identical prompts/scorers | candidate reports plus summaries bound to each BF16 receipt hash |
| L7 | 3 | clean-room reproduction | use a fresh checkout/cache and only public model/dataset revisions | independent data-only and runtime reproduction receipt |
| L8 | 12 | 5090 portability boundary | repeat the production smoke on the local driver/CUDA stack | explicit supported tuple; leave TP2/TP4/other architectures open |

**Local start sequence.** Run
`git clone https://github.com/malaiwah/qwen38-27b-exl3 && cd qwen38-27b-exl3`,
then `git fetch origin main && git checkout -B main origin/main`. Record `git rev-parse HEAD`,
verify the
three patch hashes in `receipts/native-mtp-8mp-amendment.json`, and download
`malaiwah/Qwen3.8-27B-EXL3-K5K6-context`. Confirm `nvidia-smi` reports the physical card,
31.39 GiB-class usable memory and no competing process. Launch the content-verified command
in `MODEL_CARD-K5K6-context.md`; save GPU UUID/model/driver, full command and startup log
before sending requests. Run L1 in order: exact 261,794-token needle, 30-case image suite,
236,824-token combined seven-megapixel request, three warmed 256-token decode runs, then a
second long request after release. Do not tune after seeing a partial result; a failed gate
is published as a bounded negative result.

**L1 is closed.** It ran on 2026-08-16 and all seven gates passed, but at
`--gpu-memory-utilization 0.955` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, not at
0.97 (`receipts/qualification-5090-context.json`). This sequence stays as the reproduction
procedure, with 0.955 as the launch value.

All local work lands in this repository: tools under `tools/`, immutable JSON/log evidence
under `receipts/`, and narrative in this file and `PROGRESS.md`; then run
`git push github main` (and mirror with `git push origin main` when the private Gitea remote
is reachable). Upload card-only changes to `malaiwah/Qwen3.8-27B-EXL3-K5K6-context`;
dataset captures and reports go to `malaiwah/qwen38-27b-fidelity-suite-v3`. Never compare
throughput across the two GPUs;
cross-machine pairing is allowed only for deterministic capability outputs whose prompt,
scorer, runtime and BF16 receipt hashes match.

## Contract for every investigation

Before the first candidate run, write a small machine-readable plan containing:

1. question, primary metric, paired unit, acceptance threshold and explicit failure outcome;
2. checkpoint index/config/release-evidence hashes, tokenizer revision, runtime image digest,
   patch hashes, command, environment knobs, GPU UUID/model, driver, CUDA and Torch versions;
3. immutable input identities and partition policy, including every exclusion decided before
   results are visible;
4. BF16 or current-profile control, warm-up policy, repeat count and randomness controls;
5. output schema with raw per-item results, not only means.

Afterward, publish the plan unchanged beside atomic per-run reports, stdout/stderr, a summary
derived only from those reports, and a SHA-256 inventory. Failed and rejected routes stay in
the record. Any post-hoc correction gets a new amendment that names the superseded receipt;
old evidence is never rewritten. A result closes a row only when a fresh checkout can run the
documented verifier without private paths, mutable branches or unrecorded model state.

## P0 / rank 1 — physical RTX 5090 qualification — CLOSED 2026-08-16

**Closed and passed.** Receipt `receipts/qualification-5090-context.json` (schema
`qwen38-qualification-5090-context/1`), per-process server logs
`receipts/qualification-5090-context-server-{B3,B4,C,D,E,F}.log`. This was the highest-value
remaining test because it changes the context card from “budget-proven” to
“hardware-qualified”, and that is what it did.

One physical NVIDIA GeForce RTX 5090, `GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a`, 32,607 MiB
total and 32,149 MiB free with the card idle, which vLLM sizes as **31.4 GiB usable**; driver
610.57.04, CUDA user-mode driver 13.3; image
`voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b` plus
the three content-pinned patch modules bind-mounted read-only and sha256-verified before every
launch; vLLM `0.11.2.dev280+gilded.gnosis.v20.…r34`. No other process on the card, `nvidia-smi`
sampled before load, after startup, at peak prefill and after release.

**Qualified profile: `--gpu-memory-utilization 0.955` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** Everything else is the published recipe
unchanged: `--max-model-len 262144 --max-num-seqs 1 --kv-cache-dtype fp8
--max-num-batched-tokens 2048`, MTP depth 3, `max_pixels 8388608`, `VLLM_EXL3_EMBED_BITS=8`,
`VLLM_EXL3_GRAPH_DECODE=1`, `VLLM_EXL3_PREFILL_RECONSTRUCT_M=128`.

Measured at startup on the card: engine budget **29.98 GiB** at utilisation 0.955; free 30.9 of
31.4 GiB; usage 18.19 weight + 1.78 peak activation + 0.27 non-torch + 0.45 CUDAGraph =
**20.69 GiB**; **available KV 9.28 GiB = 265,122 KV tokens**; maximum concurrency at 262,144
tokens **1.01x**; attention block size forced to 1600 tokens so the attention page is at least
the mamba page; mamba page padded 0.25 %; 3 padding layers, at most 6.25 % KV waste; startup
**55.7 s**; model load 18.19 GiB in 3.99 s.

All seven acceptance gates PASS, in the order they were pre-registered:

1. startup allocates a native-length request without exceeding the utilisation ceiling;
2. the 261,794-token text needle is exact;
3. the combined 236,824-token plus 7,077,888-pixel request returns `1376346594 | red, blue`
   exactly;
4. the 30-case image suite holds **24/30** (digits 8/10, bars 9/10, grid 7/10, seed 20260815) —
   exactly on the threshold, a pass with no margin, per-case answers published;
5. three warmed 256-token C1 runs: median **107.56 tok/s**, dispersion 0.60 %, MTP acceptance
   **56.48 %** at mean accepted length 2.69 — physical-card numbers, never differenced against
   any rental figure;
6. a second native-length request succeeds after the first is released, in the same process, so
   this is recovery rather than one lucky allocation;
7. receipt identity complete: GPU UUID and model, driver, image digest, patch hashes, commands,
   the four memory points, KV allocation, outputs and wall times.

**The published 0.97 is a bounded negative, and its mechanism is the part worth keeping.** At
0.97 the engine starts and serves text (KV 272,570 tokens, 1.04x at native length), but the
combined text-plus-image request dies with `torch.OutOfMemoryError` inside
`vllm/v1/attention/ops/vit_attn_wrappers.py` wanting 62.00 MiB with 26.50 MiB free, and
`expandable_segments` does not save it: vLLM spends the freed bytes on more KV, 272,570 →
280,017 tokens. Lowering `max_pixels` to 4,194,304 at 0.97 is **strictly worse** — profiled peak
activation falls to 1.35 GiB, KV grows to 291,933 tokens, and it OOMs sooner with 6.56 MiB free.
So this plan's old failure advice — lower `max_pixels` before sacrificing anything — was wrong:
**the knob is utilisation**, because the engine spends every freed byte on KV. Seven megapixels
is not a hard 32 GB ceiling either: the identical request succeeds at 0.955 with the full
8,388,608-pixel ceiling. The remaining pixel-count and sequence-count frontier is now rank 11.

Prefix caching is default-off for this hybrid model (`enable_prefix_caching=False`, 0.0 % hit
rate on every scheduler line), so the absent upstream fix PR #51113 is latent here and no gate
depended on it; the explicit `--no-enable-prefix-caching` control arm is published with the
receipt.

## P0 / rank 2 — immutable production runtime

The published r34 image predates every required patch. Bind-mounting Python modules is
reproducible but not a production distribution.

**Status: built and verified, serving gates pending.** The immutable image exists —
`localhost/vllm:gg-r34-patched`, manifest
`sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27`, with all three patch
modules verified by digest *inside* the image against the published map, the import machinery
proven to resolve to them and no stale bytecode (`receipts/production-image.json`, schema
`qwen38-production-image/2`). Items 1 and 4 below are satisfied there, and item 2 only in its
no-runtime-package-install part. What is **not** done is the serving half: no recipe has served a
request from the image, so item 3 and the mount, privilege and endpoint gates in items 5 and 6
are recorded as `null` rather than passed, and item 7 cannot fire while the cards still document
the three-mount recipe. Those gates are owned by the current 5090 performance window, so this
rank stays open until that receipt lands.

Build one immutable image from the pinned r34 digest plus reviewed versions of:

- #312: BF16 fallback for unrepresentable online-overlay shapes;
- #314: EXL3 graph-decode autotune priming;
- #316 and #318: prefill reconstruction dispatch and B12X row-count routing;
- #319: pass the existing quant config to both Qwen3.5 input-embedding constructors.
- #392 / #393: cherry-picks of upstream vLLM #51113 (mamba `align` prefill-chunk splitting —
  wrong tokens, HTTP 200, no crash) and #51812 (Qwen GDN speculative gate ordering — logit
  drift), **requested upstream 2026-08-16**: issue
  <https://github.com/local-inference-lab/vllm/issues/392>, PR
  <https://github.com/local-inference-lab/vllm/pull/393>. Both were re-verified absent at
  `dev/gilded-gnosis` head `fa033bd4e` — the two target files there are byte-identical to the
  r34 vendored copies. Upstream's own CPU-only regression file gives 14 failed / 6 passed
  against that head and 20 passed against the PR tree, whose outputs are `cmp`-equal to
  `tools/vllm-mamba-align-scheduler.py` and `tools/vllm-qwen-gdn-spec-gates.py`. Artefacts in
  `upstream/`. Until #51113 lands, `--enable-prefix-caching` stays unsafe for this model; it is
  default-off for hybrids, which is why every published measurement is unaffected.

Acceptance:

1. source modules in the image match the published SHA-256 map and a generated SBOM;
2. no bind mounts, startup file copies, mutable branch fetches or runtime package installs;
3. all four model-card recipes start, one text and one image request pass, and the native
   context profile repeats its memory allocation;
4. image digest, composed Git trees, package version and CUDA/Torch/driver compatibility are
   recorded;
5. model weights mount read-only, the service runs without host-root privilege, and only the
   content-addressed cache is writable;
6. the public recipe either binds loopback or requires an API key behind TLS; it never exposes
   an unauthenticated generation endpoint on every host interface;
7. cards collapse the current “unmodified” and “patched” recipes to one command only after
   that digest exists.

Wheel archives alone are insufficient: the prior audit found they were not bit-reproducible.
The OCI digest and source-tree identities are the release unit.

## P0 / rank 3 — clean-room release reproduction

The current receipts are internally cross-checkable, but most were produced on one
workstation. A second operator should begin from empty caches and only public URLs.

Two rungs are required:

1. **data-only:** verify Git/Hugging Face revisions and every release-evidence hash, download
   the published hidden-state captures, apply the declared v3/v4 exclusion policies, and
   reproduce every corrected mean, paired win count and interval used in the cards;
2. **build/runtime:** from pinned BF16 shards, converter source and immutable runtime, rebuild
   one representative checkpoint, compare logical tensor names/shapes/dtypes/storage metadata,
   then run deterministic text, image, graph/eager and KLD smoke.

Acceptance: a fresh-machine transcript contains no `/var/tmp/work` or unpublished artifact,
the data-only values match at full stored precision, every fetched blob is content-identified,
and build differences are either byte-identical or explicitly reduced to a measured numerical
equivalence contract. The independent operator signs the receipt or publishes it from a
separate account. A copy of this workstation is not an independent reproduction.

## P1 / rank 4 — public capability retention

The 40-task suite is a good regression smoke test, not evidence of broad capability. Extend
the same paired BF16/candidate design to established, licence-compatible task sets:

- code: HumanEval+/MBPP-style executable cases;
- reasoning/knowledge: a fixed MMLU-Pro or equivalent subset with exact prompt templates;
- instruction following: IFEval-style verifiable constraints;
- tools: schema-constrained multi-turn calls;
- multimodal: OCR, ChartQA and document question answering with original image hashes.

Run BF16, hydrated, online K5/K6, context, K4 and official FP8 with identical tokenizer,
template, greedy settings and runtime where formats permit. Publish every prompt, raw output,
scorer version and model identity. Primary statistic is paired pass retention versus BF16 with
bootstrap or exact intervals; absolute scores are secondary. No claim graduates from smoke to
benchmark until the public items and scorer are independently rerunnable.

**Status, 2026-08-15: the first paired run exists, and one candidate of five clears its bar.**
The reasoning/knowledge line above is now measured: 70 MMLU-Pro questions, 14 official
categories x 5, pinned `TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, official
five-shot prefixes, greedy, thinking at low effort, 5,120-token completion cap, every candidate
item-paired against the BF16 control, harness `tools/public_capability.py`, sweep runner
`tools/run_public_capability.sh`. The pre-registered acceptance in
`receipts/public-capability-plan.json` — BF16-pass retention with a Wilson 95 % lower bound at or
above **0.90**, and no category losing more than two BF16 passes — resolves like this:

| candidate | absolute | BF16-pass retention | Wilson lower | pre-registered bar |
|---|---|---|---|---|
| context edition | 58/70 | 56/57 | **90.7 %** | **met** |
| K4 | 57/70 | 55/57 | 88.1 % | not met |
| official FP8 | 56/70 | 55/57 | 88.1 % | not met |
| hydrated | 56/70 | 54/57 | 85.6 % | not met |
| online K5/K6 | 55/70 | 54/57 | 85.6 % | not met |

The category condition is met by all five (worst per-category loss is two passes, hydrated and
online K5/K6 in philosophy); the retention bound is what fails. **The reason is item count, not
build quality.** With 57 BF16 passes as the paired denominator, 56/57 is the smallest count
whose Wilson lower bound clears 0.90 — a single paired miss fails — so a 70-item suite cannot
certify this bar for anything, and **official FP8 misses it on the same items under the same
scorer**. Exact-output agreement is 0/70 for every EXL3 candidate and 1/70 for official FP8, so
the pass/fail outcome is the only meaningful pairing. This is a first public,
licence-compatible, item-paired benchmark, not a leaderboard claim.

**Therefore the next step is item volume and task diversity, not another sweep of these 70
questions.** Re-running the same suite cannot move a bound fixed by its denominator. The work
that changes the answer is the list above: HumanEval+/MBPP-style executable cases, IFEval-style
verifiable constraints, schema-constrained tool calls, and a substantially larger MMLU-Pro draw.
Until that lands, the matrix is published as a measured shortfall against our own bar and no
capability claim graduates.

## P1 / rank 5 — real multimodal quality

The current 24/30 suite is deterministic and useful, but synthetic. Add OCR, chart,
document-layout and video subsets with redistribution-safe fixtures. Run BF16 first so the
result measures retention rather than base-model capability. The 8.4 MP ceiling must be
reported as part of every result; include resize dimensions and visual-token counts.

Acceptance: paired score and latency, no hidden resize/truncation, at least one near-cap image,
and a long-text-plus-image case on the same production profile.

## P1 / rank 6 — context quality beyond retrieval

Needle retrieval proves access, not reasoning over the window. At 32k, 128k, 236k and the
largest physical-5090 length:

- two-hop retrieval with evidence in separated regions;
- aggregation over many records;
- passkey distractor count sweeps;
- perplexity or next-token loss on held-out continuous text;
- one combined image-plus-text reasoning task, not only colour recognition.

Use the server-reported token count, record TTFT/inter-token latency and fail on silent
truncation. Compare the context build with BF16 at the same lengths that BF16 can support;
do not extrapolate a short-window KLD to 262k.

## P1 / rank 7 — lifecycle and cache reliability

The online-attention cache and reconstructed-prefill scratch paths have been measured during
successful runs, not under service faults. Test the immutable image with:

- empty, warm and read-only caches; one deliberately truncated or hash-mismatched cache entry;
- termination during online encoding followed by restart;
- ten sequential native-length requests and an intentionally rejected over-budget request,
  followed by a valid request on the same server;
- concurrency at every documented profile limit, client cancellation during prefill, and
  clean shutdown/restart without orphan GPU memory or processes;
- disk-full and unwritable-cache failures that must fail closed rather than silently encode
  into an unidentified location.

Record cold-start/encode time, cache bytes and content identities, resident/peak GPU memory,
file-descriptor/process counts, request status and post-release memory after every transition.
Acceptance: no corrupt cache reuse, no silent precision fallback, no unbounded host/GPU growth,
all expected 4xx/5xx failures are explicit, and the final valid request matches the first.

## P1 / rank 8 — fair speculative-decoding matrix

The 113.8 tok/s headline compared our MTP-enabled path with FP8/NVFP4 target-only runs. It is
real throughput but not a fair model-to-model speculative comparison.

Run, for every artifact whose preserved MTP head loads:

- MTP off, depth 1 and depth 3;
- concurrency 1 / 4 / 8;
- 256 fixed output tokens, identical prompts, three warmed repeats;
- short and long configured windows separately;
- resident memory, KV allocation, drafted/accepted tokens, acceptance by position, TTFT and
  aggregate/per-stream throughput.

The closing result is a matrix, not a single favourable ratio. Unsupported comparator paths
must be reported as such rather than silently run without MTP.

## P2 / rank 9 — error-driven mixed-precision allocation

This is the last untried lever likely to improve fidelity without spending more memory.
The role split was hand-designed; exllamav3 already emits per-module proxy errors during
conversion, but the original logs were lost.

For the next conversion:

1. tee immutable converter stdout and preserve `args.json`, codebook and per-module errors;
2. measure each eligible module at two or more adjacent bit widths;
3. solve the byte-constrained benefit curve rather than assigning one width per role;
4. build one candidate only if predicted gains exceed the ~1e-3 replay-resolution caveat;
5. select on v4 analysis, then run one untouched source-disjoint qualification.

Expected improvement is a hypothesis, not a promised 10–30 %. The candidate must beat the
current profile at equal resident bytes with a paired interval excluding zero.

## P2 / rank 10 — prefill kernels

FP8 activations are closed: +31 % prefill cost +0.0141 KLD and made the context build worse
than official FP8. Remaining credible work:

1. fuse `gate_proj` and `up_proj` reconstruction/GEMM because they share input and shape;
2. prototype a Marlin-shaped fused dequant-in-epilogue kernel in exllamav3;
3. retain B12X for decode and dispatch by row count.

Acceptance is end-to-end prefill gain at unchanged decode and a measured fidelity delta below
the replay floor. Kernel microbenchmarks alone do not close the item. NVFP4's 4-bit tensor-core
prefill remains an architectural ceiling, not a tuning target for Trellis.

## P2 / rank 11 — image-cap and concurrency frontier

Eight megapixels is the first safe native-MTP point, not necessarily the maximum. On the
physical 5090, sweep 4.2 / 6.3 / 8.4 / 10.5 MP and sequence counts 1 / 2 without changing
other knobs. Report startup activation, KV tokens, actual resized dimensions and combined
request success. Select the largest cap with at least 256 KV-token margin after the full
262,144 request; do not publish a zero-headroom maximum.

**Moved here from rank 1 when it closed.** The qualification answered one point — 8.4 MP at
utilisation 0.955 — and nothing about the others, so the frontier is this item's work. It must
start from the measured mechanism: at fixed utilisation, lowering `max_pixels` *enlarges* the KV
cache, so a lower cap can OOM sooner than a higher one. At 0.97 the 4.2 MP profile failed with
6.56 MiB free against 8.4 MP's 26.50 MiB, with KV at 291,933 against 280,017 tokens
(`receipts/qualification-5090-context.json`). Sweep the cap **and** utilisation together, never
the cap alone, and treat the 4.2 MP-fits-at-30.44-GiB datum in
`receipts/native-mtp-8mp-amendment.json` as a rental engine-budget observation rather than a
physical bound.

## P2 / rank 12 — portability and package quality

- Test TP2 and TP4; the input table is rank-sliced and the overlay must prove correct shard
  scales and no non-local payload.
- Test one SM100 and one pre-Blackwell CUDA target or declare the support boundary.
- Publish explicit mixed-precision metadata; the legacy `bits: 4.0` key cannot describe
  gate/up K5, down K6 and per-row int8 input.
- Upstream the side-model `AttributeError` diagnostic and report the
  `torch._scaled_mm(..., out_dtype=float16)` row-wise-scale corruption separately.

## P2 / rank 13 — quant-specific safety stability

Low mean KLD does not guarantee unchanged refusals, calibration or tool-use boundaries. Freeze
a redistribution-safe paired set spanning self-harm, cyber misuse, privacy, medical/legal
over-reliance, jailbreak pressure and benign near-neighbours. Run BF16 and every shipped
profile with identical templates and decoding, retaining raw outputs and per-item policy labels.

Primary outcomes are BF16-safe-to-unsafe regressions, benign over-refusals and exact refusal
agreement; report categories and paired examples, never only an aggregate “safety score”.
Any severe BF16-safe-to-unsafe regression blocks a broad intended-use claim until reviewed.
A clean small set supports only “no regression observed on this set”, not general safety.

## Prior art that constrains the plan

- NVIDIA's Qwen3.6 NVFP4 is the memory/performance reference, not a drop-in recipe for this
  architecture. Its compressed-tensors runtime and calibration are different.
- Unsloth's Qwen3.8 NVFP4 retains higher precision in the last eight MLP layers and still
  measures 0.092727 on this protocol; “keep the last layers high” is not sufficient by itself.
- Gilded Gnosis GLM-5.2 demonstrates online K6 and quantized MTP for an MoE model. Its BF16
  shared-expert exception does not map onto this dense Qwen topology.
- exllamav3 supplies LDLQ/Trellis/MCG and proxy errors, but its public CLI exposes a global
  width; the regex per-module strategy used here remains a local extension.
- The Kimi-K3 protocol motivated hidden-state replay. Our 6.54e-4 live/replay discrepancy is
  roughly 500× its reported reference floor, so sub-1e-3 absolute conclusions remain out of
  scope even though paired common-head comparisons are stable.

## Do not reopen without new evidence

- Quantizing the vision tower: the converter changes fused qkv tensor topology and the loader
  intentionally excludes vision.
- A separate low-bit draft embedding: it is aliased away after load.
- Compact KV grouping: measured total memory increased.
- BF16 `lm_head`: costs 1.589 GB for a delta at or below the replay floor.
- More online attention bits: K6 is already the measured best point and attention is not the
  prefill bottleneck.
- FP8 prefill activations: fidelity failure is measured under both tensor and row-wise scales.
