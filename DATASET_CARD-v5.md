---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- kl-divergence
- quantization
- fidelity
- qwen3
- evaluation
pretty_name: Qwen3.8-27B Fidelity Suite v5
size_categories:
- 1K<n<10K
---

# Qwen3.8-27B fidelity suite v5 — evaluation inputs, the BF16 reference, and every per-shard report

This dataset exists because we deleted expensive artifacts once and had to remake them. Every
tree here is *replayable input for a future candidate*, not a finished result — the finished
results live as receipts in the research repo. Publishing the inputs means the next candidate
costs one download instead of a fresh BF16 capture, and it means anyone can check our numbers
without our hardware.

Companion research repo: <https://github.com/malaiwah/qwen38-27b-exl3>
(harness: `tools/fidelity.py`; ladder driver: `tools/kld_ladder.sh`).

The metric throughout is `KLD = KL(BF16 || candidate)`, computed from final hidden states
pushed through **one shared BF16 LM head**, so no candidate is ever credited or penalised for
its own head. Absolute KLD is suite-specific and is never compared across suite versions.

## What is in here

```
suite/suite-manifest.json          v5 parent suite: 5,120 contexts x 2,048 tokens, 2,047
                                     scored positions each = 10,480,640 positions, over 842
                                     source clusters.  suite_token_sha256
                                     510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88
suite/tokens/*.json                5,120 token-id files, one per context.  AUTHORITATIVE:
                                     retokenising the source text does NOT reproduce the
                                     evaluation input.  Each context's token_sha256 is
                                     sha256 of json.dumps(ids).
suite/ladder-pin.json              what the whole ladder was pinned to: harness digest,
                                     shared-head digest, quantization config, shard size,
                                     runtime image digest.
suite/shard-00{00..09}/suite-manifest.json
                                   the ten 512-context shard views, byte-for-byte as captured
                                     and replayed against.  Each report under reports/kld5/
                                     cites its shard view's suite_token_sha256, so these are
                                     what make the per-shard reports independently checkable
                                     against the parent suite.  Shard 0's is
                                     caef8a4628d6c07c162100895096f890cdf9cafc8e4c48b3d66035d737ee7cf7
suite/shard-00NN/commands/*.sh     the exact capture and replay command lines the ladder ran
                                     for that shard (guest paths preserved, for reference).
corpus/corpus_fetch_log.json       every source document: stratum, stem, URL, bytes, chars,
                                     sha256, plus every skip and every failure.  Refetch the
                                     corpus from this; distrust a re-fetch that moves a digest.
reference/hidden-bf16/             THE most reusable artifact in this repo.  512 files,
                                     hidden_NNNN.safetensors, each [2047, 5120] bf16, plus
                                     capture-manifest.json.  Unquantized BF16
                                     Qwen/Qwen3.8-27B final hidden states over the shard-0
                                     contexts.  Every future candidate on shard 0 replays
                                     against this instead of re-capturing BF16.
reports/kld5/shard-00{00..09}/     50 per-shard reports: the full ten-shard ladder, five
                                     candidates (hyd, k5k6, ctx, fp8, k4).
reports/kld5-tail/shard-000{0,1}/  10 per-shard tail reports — percentiles and exact
                                     exceedance counts over 2,096,128 positions.
reports/kld5-win/shard-0000/       15 scored-window control reports: score_from 0 / 256 /
                                     1024 for all five candidates.
reports/gguf/report-*.json         4 cross-engine reports on shard 0: llama.cpp Q8_0, Q6_K,
                                     UD-Q5_K_XL, and the llama.cpp-vs-vLLM engine floor
                                     measured on the same unquantized BF16 weights.
captures/shard-0000/hidden-nvfp4/  512 captures + manifest.  unsloth/Qwen3.8-27B-NVFP4 final
                                     hidden states over the shard-0 contexts.
captures/shard-0000/hidden-eda/    512 captures + manifest.  The error-driven allocation
                                     research build (docs/37), the candidate that lost.
captures/shard-0000/hidden-hyd-rematch/
                                   512 captures + manifest.  The published hydrated
                                     checkpoint re-captured against the surviving BF16
                                     reference with the same harness, so the paired interval
                                     between it and hidden-eda uses one reference and one
                                     code version for both operands.
captures/shard-0000/error-driven-ladder.json
                                   the five-rung proxy-error ladder the error-driven build
                                     was solved from: 409 modules x up to five widths, with
                                     out_energy, numel, qmap and per-rung seconds.  Not a
                                     capture, but the input the two captures above exist to
                                     adjudicate.
```

The `captures/` prefix holds shard-0 *candidate* capture trees, each with its own
`capture-manifest.json`, added by `tools/preserve_artifacts.sh --deep`. A candidate capture
paired with the BF16 reference above is the one case where preserving a capture buys a reader
something they cannot cheaply recompute: the two together let anyone re-derive that
candidate's headline number by `replay` alone, with no GPU capture at all. The error-driven
pair is the worked example — replaying `hidden-eda` and `hidden-hyd-rematch` against
`reference/hidden-bf16` reproduces the paired **−0.00036630 [−0.00039779, −0.00033477]** that
closed that experiment as a negative, on a laptop.

### Sibling artifacts: two archival mirrors

Both are model repos, and both exist because a **pinned revision hash is not durable
provenance on its own**. Keep the distinction between them — one is a rescue, the other is
insurance, and calling insurance a rescue would be the same small overstatement we keep
finding in other people's claims.

* `malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da` @ `7a66267ebd34a01ba9a13e56aa2cea0b27bdacd4`
  — **recovery.** Mirrors `unsloth/Qwen3.8-27B-NVFP4` at revision
  `9c73e2daee1d0fd494ffbd1d8753f2174a953796`, the revision our published v3 NVFP4 comparison
  was measured against. That repo super-squashed its history on 2026-08-15 and the Hub now
  answers *Invalid rev id* for that revision. The weights are byte-identical to current
  upstream HEAD, but `tokenizer.json` and one metadata field did change, so the comparison
  can only be cited through the mirror.
* `malaiwah/Qwen3.8-27B-GGUF-archival-f1bfb127` @ `06992e2f16022347149d8545b1df04c68d46e6e7`
  — **precautionary.** The five `unsloth/Qwen3.8-27B-GGUF` files our cross-engine table cites
  (`Q8_0`, `Q6_K`, `UD-Q5_K_XL`, and the two-part BF16 set that produced the 0.000507 engine
  floor) at revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, which still resolves and is
  current HEAD. Nothing is lost yet; this is so nothing can be.

**A mirror is not a backup.** Hub storage is content-addressed, so our copy and upstream's
plausibly reference the same underlying chunks. What a mirror durably preserves is the
**citation** — a resolvable repo id, revision and digest table that survive an upstream squash
or delete. It does not establish independent byte-level redundancy, and nobody should later
assume we hold physical copies we do not hold; real redundancy is a separate decision with a
real bandwidth bill.

### What we preserve, and what we deliberately do not

Three tiers, not one rule:

1. **Mirror** any third-party artifact a published number of ours cites. Unconditional — the
   measured cost is 2.34 GB of wire transfer for 149.3 GB of content across the two mirrors
   above, about 1.6 %, because the bytes already exist on the Hub. At that price there is no
   trade-off to weigh.
2. **Preserve** a locally-produced artifact when recompute costs GPU-hours, when an input has
   ceased to exist, or when many future runs replay against it. That is this dataset.
3. **State the recompute cost** for everything else, and delete it freely.

Tier 3, itemised — what is *not* here and why:

* **The corpus text itself** (69 MB). Refetchable from `corpus/corpus_fetch_log.json`, which
  pins every file's sha256; and `suite/tokens/` is the authoritative evaluation input anyway.
* **The shared BF16 LM head** (2.54 GB). Already published once, in
  `malaiwah/qwen38-27b-fidelity-suite-v3` as `lm-head/weight.safetensors`, sha256
  `25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff` — verified equal to this
  suite's `ladder-pin.json` `head_sha256`, so it is the same head every report here used.
  Publishing it twice buys nothing.
* **Candidate hidden states for the five ladder candidates, shards 0-9.** One shard of six
  models is ~64 GB and `/var/tmp` holds ~135 GB, so the ladder captured, replayed, verified
  and deleted one shard before starting the next. Re-capturing one candidate on one shard is
  roughly 6 minutes of GPU, against ~190 GB of upload. This is the case where uploading is the
  wrong call, and the ten-shard NVFP4 set is the same call for the same reason.
* **Shards 1-9 of the BF16 reference.** Same ladder, same reasoning — about 6 minutes of GPU
  each. Shard 0 is preserved because it is the shared reference every future shard-0 candidate
  replays against; nothing replays against shards 1-9 until someone runs the full ladder again.
* **The GGUF weight files** (18.8-27.1 GiB each). Not redistributed here; they are mirrored as
  a third-party artifact under tier 1, above, and `reports/gguf/*.json` pins each one's sha256
  and size.

`receipts/preserved-artifacts.json` in the research repo states the exact recompute cost of
every artifact that was destroyed. That receipt is the reason this dataset exists.

## Verify what you downloaded

```bash
hf download malaiwah/qwen38-27b-fidelity-suite-v5 --repo-type dataset --local-dir v5

python - <<'PY'
import hashlib, json, pathlib
d = pathlib.Path("v5/reference/hidden-bf16")
m = json.loads((d / "capture-manifest.json").read_text())
assert m["complete"] is True, "partial capture"
for c in m["captures"]:
    p = d / f"hidden_{c['index']:04d}.safetensors"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(8 << 20)):
            h.update(b)
    assert h.hexdigest() == c["sha256"], f"digest drift: {p}"
print(f"{len(m['captures'])} reference captures verified against the capture manifest")
print("suite digest they were captured over:", m["suite_token_sha256"])
PY
```

That is the check that matters for the 10 GB tree. For the suite itself, the harness is the
authoritative verifier: every `capture`, `replay` and `paired` command below re-derives the
suite digest from the manifest's own ordered per-context token digests and aborts on drift, so
a command that completes has already proved the suite is intact.

## Replay: score your own checkpoint against the published BF16 reference

No BF16 capture, no 27B reference model, and no re-tokenisation required.

```bash
git clone https://github.com/malaiwah/qwen38-27b-exl3
hf download malaiwah/qwen38-27b-fidelity-suite-v5 --repo-type dataset --local-dir v5
hf download malaiwah/qwen38-27b-fidelity-suite-v3 --repo-type dataset \
  --include 'lm-head/*' --local-dir v3

# The shard-0 view is published with its exact captured bytes, which name their token files
# as `tokens/...` relative to the view.  Point that name at the parent token directory.
ln -s ../tokens v5/suite/shard-0000/tokens

# 1. capture YOUR candidate's final hidden states over the shard-0 contexts
python qwen38-27b-exl3/tools/fidelity.py capture \
  --model /path/to/your/checkpoint \
  --suite v5/suite/shard-0000 \
  --out hidden-mine

# 2. replay it against the published BF16 reference, through the shared head
python qwen38-27b-exl3/tools/fidelity.py replay \
  --reference v5/reference/hidden-bf16 \
  --candidate hidden-mine \
  --head v3/lm-head/weight.safetensors \
  --suite v5/suite/shard-0000 \
  --out report-mine.json

# 3. compare against a published candidate, paired per context
python qwen38-27b-exl3/tools/fidelity.py paired \
  --a report-mine.json --b v5/reports/kld5/shard-0000/report-hyd.json \
  --a-label mine --b-label hydrated-k5k6 --out paired-mine.json
```

`replay` is fail-closed: it rejects a candidate capture whose suite digest, context set,
tensor shape, head digest or scored-position window differs from the reference. A report it
accepts is comparable with the reports in this repo by construction.

**If your candidate is a rebuild of one of our recipes, it is a sibling and not our checkpoint.**
The EXL3 converter is nondeterministic: re-running the published hydrated recipe on the same box
with the same flags returns identical configs, index, quantization descriptors, safetensors
headers, per-role byte totals and widths, while 399 of the 409 quantized modules (97.6 %) differ
inside their `.trellis` payloads at 41-92 % of the bytes
([`receipts/converter-determinism.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/converter-determinism.json)).
Every published number here was measured on the published bytes, which is what a downloader
receives and what each repo's `SHA256SUMS` pins, so those numbers are unaffected — but a report
from your own rebuild is that sibling's number, not a reproduction of ours, and whether a sibling
lands inside our interval is an open experiment rather than a settled one.

To weld per-shard reports into a cumulative number the way the published receipts do, run
`tools/kld_aggregate.py` over `reports/kld5/shard-*/report-<name>.json`.

## What these inputs measured

**Shard 0 alone** — 512 contexts, 1,048,064 scored positions, the exact contexts
`reference/hidden-bf16` covers. These are the numbers a new candidate lands beside:

| candidate | mean KLD | net of the cross-engine floor |
|---|---:|---:|
| llama.cpp GGUF `Q8_0` | 0.001087 | ~0.000579 |
| llama.cpp GGUF `Q6_K` | 0.002035 | ~0.001528 |
| EXL3 K5/K6 hydrated | 0.002700 | — |
| EXL3 K5/K6 online | 0.003141 | — |
| EXL3 context + int8 input | 0.003409 | — |
| llama.cpp GGUF `UD-Q5_K_XL` | 0.004444 | ~0.003936 |
| official Qwen FP8 | 0.005197 | — |
| EXL3 K4 | 0.010345 | — |

**Cross-engine floor: 0.000507** (99.07 % top-1, p99.9 0.0113) — llama.cpp against vLLM on
the *same unquantized BF16 weights*, `reports/gguf/report-engine-floor.json`. Every llama.cpp
number above is measured against a vLLM BF16 reference and therefore carries this floor; the
"net of floor" column subtracts it and is an estimate, not a measurement. The EXL3 and FP8
numbers are vLLM-against-vLLM and carry no such floor.

**Full ten-shard ladder** — 10,480,640 scored positions over 842 clusters, bootstrap
intervals over clusters. This is what `reports/kld5/` aggregates to:

| candidate | mean KLD | 95 % interval | top-1 |
|---|---:|---|---:|
| EXL3 K5/K6 hydrated | **0.002760** | [0.002540, 0.003020] | 97.70 % |
| EXL3 K5/K6 online | 0.003210 | [0.002982, 0.003480] | 97.52 % |
| EXL3 context + int8 input | 0.003509 | [0.003220, 0.003852] | 97.44 % |
| official Qwen FP8 | 0.005294 | [0.004927, 0.005728] | 96.79 % |
| EXL3 K4 | 0.010604 | [0.009640, 0.011746] | 95.76 % |

`reports/kld5-win/` is the control for the obvious objection that scoring positions with
almost no left context inflates every mean. Requiring 256 tokens of left context lowers every
candidate's mean by 1.3-2.1 %, and second-half-only by 3.9-4.9 % — uniformly, so the ranking
does not depend on the scored window.

## Provenance

* Reference model: `Qwen/Qwen3.8-27B`, unquantized BF16, served by vLLM.
* Runtime image: `voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`.
* Capture hardware: 1x RTX PRO 6000 Blackwell Server Edition (rental box). Fidelity numbers
  are hardware-independent; throughput numbers elsewhere in the project are not, and are
  never differenced across cards.
* Harness digest, shared-head digest and quantization config: `suite/ladder-pin.json`.
* Corpus: held-out public text (Project Gutenberg, CPython `Lib/`, arXiv abstracts,
  Wikipedia), token-disjoint from the v4 suite, with any document containing an exact
  calibration 12-gram excluded *before* context selection rather than after.

## Licence

Apache-2.0 for the artifacts produced here — captures, reports, manifests, token id lists.
Source text is not redistributed; `corpus/corpus_fetch_log.json` names every upstream source,
and that text stays under its own licence.
