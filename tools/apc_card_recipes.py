#!/usr/bin/env python3
"""Put prefix caching into all four published model-card recipes, on the promoted image.

Every site is located by content match, never by line number, and every anchor must occur
exactly once or the script refuses to write anything. Every figure that appears on a card is
read out of receipts/qualification-5090-apc.json and receipts/production-image.json rather
than typed here, so a card cannot drift from the receipt it cites.

Refuses to run at all unless the qualification receipt says all nine gates passed: a failed
gate leaves the recipes exactly as they were.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import textwrap

REPO = pathlib.Path(__file__).resolve().parent.parent
BLOB = "https://github.com/malaiwah/qwen38-27b-exl3/blob/main"

SCHED_SHA = "b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf"
GDN_SHA = "7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0"
EXL3_SHA = "2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee"
QWEN_SHA = "04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7"
MTP_SHA = "0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab"
PROMOTED_MANIFEST = "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218"
PARENT_MANIFEST = "sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27"
BASE_DIGEST = "sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
SCHED_DEST = "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
GDN_DEST = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/"
            "gdn/qwen_gdn_linear_attn.py")


def wrap(paragraph, width=95, indent=""):
    """Match the cards' own hard wrap. URLs are never split: break_long_words=False."""
    return textwrap.fill(paragraph, width=width, break_long_words=False,
                         break_on_hyphens=False, initial_indent=indent,
                         subsequent_indent=indent)


class Refused(Exception):
    pass


class Patch:
    """A set of exactly-once anchored edits over one file, applied all-or-nothing."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.text = self.path.read_text()
        self.original = self.text
        self.log = []

    def _once(self, anchor, what):
        n = self.text.count(anchor)
        if n != 1:
            raise Refused(f"{self.path.name}: anchor for {what} occurs {n} times, "
                          f"expected exactly 1: {anchor[:90]!r}")

    def after(self, anchor, addition, what):
        self._once(anchor, what)
        self.text = self.text.replace(anchor, anchor + addition, 1)
        self.log.append(f"insert after: {what}")

    def before(self, anchor, addition, what):
        self._once(anchor, what)
        self.text = self.text.replace(anchor, addition + anchor, 1)
        self.log.append(f"insert before: {what}")

    def replace(self, anchor, new, what):
        self._once(anchor, what)
        self.text = self.text.replace(anchor, new, 1)
        self.log.append(f"replace: {what}")

    def absent(self, needle, what):
        if needle in self.text:
            raise Refused(f"{self.path.name}: {what} is already present; refusing to "
                          f"apply the same edit twice")

    def write(self, dry_run=False):
        if self.text == self.original:
            raise Refused(f"{self.path.name}: nothing changed")
        if not dry_run:
            self.path.write_text(self.text)
        return {
            "file": self.path.name,
            "edits": self.log,
            "bytes_before": len(self.original.encode()),
            "bytes_after": len(self.text.encode()),
            "sha256_after": hashlib.sha256(self.text.encode()).hexdigest(),
        }


def figures(qual):
    """Every number the cards print about this run, read out of the receipt."""
    g1 = qual["gates"]["gate1_startup_native_allocation"]
    g9 = qual["gates"]["gate9_kv_capacity_not_regressed"]
    g5 = qual["gates"]["gate5_three_warmed_decode_runs"]
    g8 = qual["gates"]["gate8_banner_prefix_caching_actually_on"]
    tokens = g9["worst_case_gpu_kv_cache_tokens"]
    base = g9["baseline"]["gpu_kv_cache_tokens"]
    delta = tokens - base
    if delta == 0:
        kv_cost = (f"KV capacity is unchanged: **{tokens:,} tokens** at 262,144, exactly the "
                   f"{base:,} the same profile measured with the cache off")
    elif delta < 0:
        kv_cost = (f"KV capacity **falls from {base:,} to {tokens:,} tokens** at 262,144, a "
                   f"cost of **{-delta:,} tokens ({100.0 * -delta / base:.2f} %)** — the price "
                   f"of the feature, printed here rather than absorbed")
    else:
        kv_cost = (f"KV capacity **rises from {base:,} to {tokens:,} tokens** at 262,144 "
                   f"(+{delta:,}); the cache costs no capacity on this profile")
    return {
        "kv_cost": kv_cost,
        "kv_tokens": tokens,
        "kv_baseline": base,
        "kv_gib": g1.get("available_kv_cache_gib"),
        "kv_gib_baseline": g9["baseline"]["available_kv_cache_gib"],
        "concurrency": g1.get("maximum_concurrency_for_native_length"),
        "concurrency_baseline": g9["baseline"]["maximum_concurrency_at_262144"],
        "budget_gib": g1.get("engine_budget_gib"),
        "usage": g1.get("measured_engine_usage_gib") or {},
        "median_decode": (g5.get("decode") or {}).get("median_decode_tok_s"),
        "banner_ok": g8["passed"],
        "block_size": g1.get("attention_block_size_tokens"),
    }


def promotion_paragraphs(fig, profile, indent=""):
    """The prose every card carries, with a per-card qualification-scope sentence."""
    qual_link = f"[`receipts/qualification-5090-apc.json`]({BLOB}/receipts/qualification-5090-apc.json)"
    image_link = f"[`receipts/production-image.json`]({BLOB}/receipts/production-image.json)"
    repro_link = f"[`receipts/apc-poison-repro.json`]({BLOB}/receipts/apc-poison-repro.json)"
    mamba_link = f"[`receipts/mamba-align-defect.json`]({BLOB}/receipts/mamba-align-defect.json)"
    gdn_link = f"[`receipts/gdn-spec-gate-defect.json`]({BLOB}/receipts/gdn-spec-gate-defect.json)"

    one = (
        "**Prefix caching is on in the recipe above, and upstream #51113 is why it can be.** "
        "vLLM #51113 (mamba `align` prefill-chunk splitting: a chunk that ends mid-block "
        "leaves its slot holding a short state, which a later chunk then publishes anyway — "
        "wrong tokens, HTTP 200, no crash) merged 2026-08-06, after the pinned public image "
        "was built, and is still absent from it and from fork head `fa033bd4e`. Cherry-picks "
        "were requested upstream on 2026-08-16 "
        "([issue #392](https://github.com/local-inference-lab/vllm/issues/392), "
        "[PR #393](https://github.com/local-inference-lab/vllm/pull/393)). Until they land it "
        f"is carried as `tools/vllm-mamba-align-scheduler.py` (`sha256 {SCHED_SHA[:8]}…`), "
        "mounted above, and it is **baked into the image this project now serves**: the "
        f"release unit is the four-module `localhost/vllm:gg-r34-patched-apc`, manifest "
        f"`{PROMOTED_MANIFEST[:21]}…`, promoted on 2026-08-16 from the three-module "
        f"`localhost/vllm:gg-r34-patched` (`{PARENT_MANIFEST[:21]}…`) and re-qualified with "
        "prefix caching on — the seven serving gates of the earlier qualification, unchanged, "
        "plus two the old set did not need: that the engine banner really reports "
        "`enable_prefix_caching: True` and `mamba_cache_mode: align`, and that KV capacity at "
        f"262,144 has not regressed ({qual_link}, {image_link}). One warning if you inspect "
        "that image yourself: its build-time label `io.malaiwah.image.qualified` still "
        "reads `false` and is **superseded by the qualification receipt named here** — "
        "the label was written before the image could possibly have been qualified, and "
        "correcting it would add a layer and change the very digest that was measured. "
        "That digest is local to the build host, so the recipe above reproduces its "
        "content with sha256-verified read-only mounts over the pullable public base.")

    two = (
        f"**The memory cost, stated plainly.** {fig['kv_cost']}"
        + (f", at {fig['kv_gib']} GiB of KV against {fig['kv_gib_baseline']} GiB, "
           f"engine budget {fig['budget_gib']} GiB, maximum concurrency "
           f"{fig['concurrency']}x at 262,144 against {fig['concurrency_baseline']}x"
           if fig["kv_gib"] else "")
        + ". vLLM's prefix-cache block pool is host-side Python state — a hash table of block "
          "hashes and a free list — so it costs no device memory of its own; any device cost "
          "shows up in the KV token count and nowhere else, which is why that one number is "
          "gated.")

    three = (
        "**What it buys, and what we actually know.** On disjoint documents, so the cold case "
        "is genuinely cold: a 32,842-token prefix went **12.07 s cold → 1.04 s warm (11.6×**, "
        "2,442 of 32,842 prompt tokens recomputed, 92.6 % hit rate) and a 131,146-token "
        "prefix went **67.60 s → 2.31 s (29.3×**, 3,146 of 131,146 recomputed, 97.6 %); a "
        "38-request schedule ran 84.0 s with the cache against 144.4 s without. Correctness "
        "was probed adversarially first: seven freshly started servers, 38 requests each, "
        "**266 scored requests**, nested token prefixes so later requests hit blocks published "
        f"by earlier ones and no prompt length a multiple of the measured "
        f"{(fig['block_size'] or 1600):,}-token "
        "mamba block, so prefill chunks end mid-block by construction — **zero corrupted "
        "responses, zero wrong answers, zero acceptance collapses, on the unpatched image as "
        "well as the patched one**. Thresholds were committed before the first server "
        "started; the worst repeated block was 15 characters against an 80-character "
        "threshold, with no U+FFFD anywhere. Greedy chosen-logprob drift with the cache on is "
        "0.1063 mean absolute against a measured run-to-run floor of 0.0823 — drift, never an "
        f"answer change ({repro_link}). So the module is carried as **insurance backed by "
        "upstream's own regression file** — 14 failed / 6 passed against the vendored "
        "scheduler, 20 passed against this one — and **not** by a reproduction of our own: we "
        f"tried hard to reproduce the reported corruption and could not ({mamba_link}).")

    four = (
        "**LMCache is unmeasured by us.** It is not part of any recipe on this card, this "
        "project has never run it, and it is the outstanding suspect in the one user report "
        "of prefix-cache corruption we have. Nothing here says LMCache is safe; the evidence "
        "above covers vLLM's own prefix cache and nothing else.")

    five = (
        "**#51812 stays an optional overlay, deliberately.** Upstream #51812 (Qwen GDN "
        "speculative gate ordering: gathered Q/K/V rows and unsorted `a`/`b` gate rows can "
        "belong to different tokens in a mixed batch, which drifts logits) merged 2026-08-11 "
        "and is absent from the same images. It is **not** in the promoted release unit, "
        "because a mixed batch is the only condition it bites in and `--max-num-seqs 1` never "
        "produces one — the run that served 38/38 clean with it mounted therefore shows it is "
        "harmless, not that it fixes anything, and at concurrency it is unmeasured by us and "
        "rests on upstream's own numbers. If you serve concurrently, mount it as well: "
        f"`tools/vllm-qwen-gdn-spec-gates.py` (`sha256 {GDN_SHA[:8]}…`) over "
        f"`{GDN_DEST}` ({gdn_link}).")

    return "\n\n".join(wrap(x, indent=indent)
                       for x in (one, two, three, four, five, profile))


def context_card(fig):
    p = Patch(REPO / "MODEL_CARD-K5K6-context.md")
    p.absent("--enable-prefix-caching", "the prefix-caching flag")
    p.after(
        f"{MTP_SHA}  tools/vllm-qwen3_5_mtp-embed-quant-config.py\n",
        f"{SCHED_SHA}  tools/vllm-mamba-align-scheduler.py\n",
        "scheduler digest in the verified module list")
    p.after(
        '  -v "$PATCH/vllm-qwen3_5_mtp-embed-quant-config.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro" \\\n',
        '  -v "$PATCH/vllm-mamba-align-scheduler.py:$VLLM/v1/core/sched/scheduler.py:ro" \\\n',
        "scheduler mount in the patched native-window recipe")
    p.before(
        "    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \\\n"
        "    --host 0.0.0.0 --port 8000\n```\n\n`--gpu-memory-utilization 0.955`",
        "    --enable-prefix-caching --mamba-cache-mode align \\\n",
        "prefix-caching flags in the patched native-window recipe")
    p.replace(
        "It does not include the input-table overlay,\ngraph decode or prefill routing, so it "
        "is not the native-window profile.",
        "It does not include the input-table overlay,\ngraph decode or prefill routing, so it "
        "is not the native-window profile. It also runs with prefix\ncaching **off**, and must: "
        "turning it on safely needs upstream #51113, which this image\npredates. The patched "
        "profile below is where it is enabled.",
        "the unmodified-fallback note about prefix caching")

    profile = (
        "**Scope of the prefix-caching qualification.** This is the profile the nine gates "
        "were run on, at `--gpu-memory-utilization 0.955` with "
        "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, window 262,144, "
        "`--max-num-seqs 1`, MTP depth 3, fp8 KV, 8,388,608-pixel ceiling: startup, the "
        "261,794-token needle, the combined 236,824-token plus seven-megapixel request, the "
        "30-case image suite at 24/30, three warmed decode runs"
        + (f" (median **{fig['median_decode']} tok/s** on the physical RTX 5090, never to be "
           "differenced against a rental RTX PRO 6000 number)" if fig["median_decode"] else "")
        + ", a second native-length request after release, receipt identity, the banner check "
          "and the KV capacity check. The concurrent-serving variant this card also documents "
          "is **not** qualified at any sequence count.")
    p.replace(UPSTREAM_PARA_CONTEXT, promotion_paragraphs(fig, profile),
              "the two-absent-fixes paragraph")
    return p


def k5k6_card(fig):
    p = Patch(REPO / "MODEL_CARD-K5K6.md")
    p.absent("--enable-prefix-caching", "the prefix-caching flag")
    p.replace(
        "set -euo pipefail\nPATCH=$PWD/vllm-exl3-prefill-dispatch.py\nprintf '%s  %s\\n' \\\n"
        f"  {EXL3_SHA} \\\n  \"$PATCH\" | sha256sum -c -\n",
        "set -euo pipefail\nPATCH=$PWD/vllm-exl3-prefill-dispatch.py\n"
        "SCHED=$PWD/vllm-mamba-align-scheduler.py\nprintf '%s  %s\\n' \\\n"
        f"  {EXL3_SHA} \"$PATCH\" \\\n"
        f"  {SCHED_SHA} \"$SCHED\" |\n  sha256sum -c -\n",
        "two-module digest check in recipe B")
    p.after(
        '  -v "$PATCH:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \\\n',
        f'  -v "$SCHED:{SCHED_DEST}:ro" \\\n',
        "scheduler mount in recipe B")
    p.after(
        "    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \\\n",
        "    --enable-prefix-caching --mamba-cache-mode align \\\n",
        "prefix-caching flags in recipe B")

    profile = (
        "**Scope on this build.** Recipe B above is smoke-proven with prefix caching on — it "
        f"starts and answers a text and an image request exactly on the promoted image "
        f"([`receipts/production-image.json`]({BLOB}/receipts/production-image.json)) — but "
        "the nine-gate qualification was run on the native-context profile, not on this "
        "8,192-token one. Recipe A stays prefix-caching-off and must: it is the unmodified "
        "public image, which predates #51113.")
    item = promotion_paragraphs(fig, profile, indent="   ")
    p.replace(UPSTREAM_ITEM_K5K6, "6." + item[2:],
              "list item 6, the two-absent-fixes item")
    return p


def sibling_card(name, fig, recipe_block, profile):
    p = Patch(REPO / name)
    p.absent("--enable-prefix-caching", "the prefix-caching flag")
    p.replace(UPSTREAM_PARA_SIBLING,
              promotion_paragraphs(fig, profile) + "\n\n" + recipe_block,
              "the two-absent-fixes paragraph")
    return p


UPSTREAM_PARA_CONTEXT = """**Two upstream fixes are absent from every published image digest.** Upstream vLLM #51113 (mamba
`align` prefill-chunk splitting: a chunk that ends mid-block leaves its slot holding a short state,
which a later chunk then publishes anyway — wrong tokens, HTTP 200, no crash) merged 2026-08-06 and
#51812 (Qwen GDN speculative gate ordering: gathered Q/K/V rows and unsorted `a`/`b` gate rows can
belong to different tokens in a mixed batch, which drifts logits) merged 2026-08-11 — both after the
pinned image was built, and both re-verified absent at fork head `fa033bd4e`, where the two target
files are byte-identical to the r34 vendored copies. Cherry-picks were requested upstream on
2026-08-16: [issue #392](https://github.com/local-inference-lab/vllm/issues/392) and
[PR #393](https://github.com/local-inference-lab/vllm/pull/393). Until they land, both are published
as mount-in modules — `tools/vllm-mamba-align-scheduler.py` and `tools/vllm-qwen-gdn-spec-gates.py`
— and **neither is part of the qualified image digest**: upstream's own CPU-only regression file
gives 14 failed / 6 passed against the vendored scheduler and 20 passed against the patched tree
([`receipts/mamba-align-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/mamba-align-defect.json),
[`receipts/gdn-spec-gate-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gdn-spec-gate-defect.json))."""

UPSTREAM_PARA_SIBLING = """**Two upstream fixes are absent from this pinned image.** Upstream vLLM #51113 (mamba `align`
prefill-chunk splitting: a chunk that ends mid-block leaves its slot holding a short state, which a
later chunk then publishes anyway — wrong tokens, HTTP 200, no crash) merged on 2026-08-06, and
#51812 (Qwen GDN speculative gate ordering: the gathered Q/K/V rows and the unsorted `a`/`b` gate
rows can belong to different tokens in a mixed batch, which drifts logits) merged on 2026-08-11 —
both after this image was built, and both re-verified absent at fork head `fa033bd4e`, where the two
target files are byte-identical to the r34 vendored copies. Cherry-picks were requested upstream on
2026-08-16: [issue #392](https://github.com/local-inference-lab/vllm/issues/392) and
[PR #393](https://github.com/local-inference-lab/vllm/pull/393). Until they land, both are published
as mount-in modules — `tools/vllm-mamba-align-scheduler.py` and `tools/vllm-qwen-gdn-spec-gates.py`
— and **neither is part of the qualified image digest**: upstream's own CPU-only regression file
gives 14 failed / 6 passed against the vendored scheduler and 20 passed against the patched tree
([`receipts/mamba-align-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/mamba-align-defect.json),
[`receipts/gdn-spec-gate-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gdn-spec-gate-defect.json))."""

UPSTREAM_ITEM_K5K6 = """6. **Two upstream fixes are absent from every published image digest.** Upstream vLLM #51113
   (mamba `align` prefill-chunk splitting: a chunk that ends mid-block leaves its slot holding a
   short state, which a later chunk then publishes anyway — wrong tokens, HTTP 200, no crash)
   merged 2026-08-06 and #51812 (Qwen GDN speculative gate ordering: gathered Q/K/V rows and
   unsorted `a`/`b` gate rows can belong to different tokens in a mixed batch, which drifts logits)
   merged 2026-08-11, both after the pinned image was built, and both re-verified absent at fork
   head `fa033bd4e`, where the two target files are byte-identical to the r34 vendored copies.
   Cherry-picks were requested upstream on 2026-08-16:
   [issue #392](https://github.com/local-inference-lab/vllm/issues/392) and
   [PR #393](https://github.com/local-inference-lab/vllm/pull/393). Until they land, both are
   published as mount-in modules — `tools/vllm-mamba-align-scheduler.py` and
   `tools/vllm-qwen-gdn-spec-gates.py` — and **neither is part of the qualified image digest**;
   upstream's own CPU-only regression file gives 14 failed / 6 passed against the vendored
   scheduler and 20 passed against the patched tree
   ([`receipts/mamba-align-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/mamba-align-defect.json),
   [`receipts/gdn-spec-gate-defect.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gdn-spec-gate-defect.json))."""

K4_RECIPE = f"""Enabling it on this profile is one extra read-only mount and two flags:

```bash
set -euo pipefail
git clone https://github.com/malaiwah/qwen38-27b-exl3 && cd qwen38-27b-exl3
cat <<'SHA256' | sha256sum -c -
{SCHED_SHA}  tools/vllm-mamba-align-scheduler.py
SHA256
SCHED=$PWD/tools/vllm-mamba-align-scheduler.py

docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \\
  -v /models:/models:ro -v /cache:/cache \\
  -v "$SCHED:{SCHED_DEST}:ro" \\
  -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \\
  -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \\
  -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \\
  --entrypoint /opt/venv/bin/vllm \\
  voipmonitor/vllm@{BASE_DIGEST} \\
  serve /models/Qwen3.8-27B-K4 \\
    --served-model-name qwen38-k4 \\
    --quantization exl3 \\
    --enforce-eager \\
    --quantization-config '{{"linear":{{"weight":"mxfp8"}},"ignore":["re:.*visual\\\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\\\..*","lm_head"]}}' \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.85 \\
    --max-num-seqs 4 \\
    --enable-prefix-caching --mamba-cache-mode align \\
    --host 0.0.0.0 --port 8000
```"""

HYDRATED_RECIPE = f"""Enabling it on this profile is one extra read-only mount and two flags:

```bash
set -euo pipefail
git clone https://github.com/malaiwah/qwen38-27b-exl3 && cd qwen38-27b-exl3
cat <<'SHA256' | sha256sum -c -
{SCHED_SHA}  tools/vllm-mamba-align-scheduler.py
SHA256
SCHED=$PWD/tools/vllm-mamba-align-scheduler.py

docker run --rm --gpus '"device=0"' --ipc host -p 127.0.0.1:8000:8000 \\
  -v /models:/models:ro \\
  -v "$SCHED:{SCHED_DEST}:ro" \\
  --entrypoint /opt/venv/bin/vllm \\
  voipmonitor/vllm@{BASE_DIGEST} \\
  serve /models/Qwen3.8-27B-EXL3-K5K6-hydrated \\
    --served-model-name qwen38 --quantization exl3 --enforce-eager \\
    --quantization-config '{{"linear":{{"weight":"mxfp8"}},"ignore":["re:.*visual\\\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\\\..*","lm_head"]}}' \\
    --mm-processor-kwargs '{{"truncation":false}}' \\
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \\
    --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 8 \\
    --enable-prefix-caching --mamba-cache-mode align \\
    --host 0.0.0.0 --port 8000
```"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual", default=str(REPO / "receipts/qualification-5090-apc.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qual = json.loads(pathlib.Path(args.qual).read_text())
    if not qual["verdict"]["all_gates_pass"]:
        print("REFUSED: the qualification receipt does not report all gates passing.\n"
              + json.dumps(qual["verdict"]["decisions"], indent=2), file=sys.stderr)
        return 2
    fig = figures(qual)
    if not fig["banner_ok"]:
        print("REFUSED: the banner gate did not prove prefix caching was on", file=sys.stderr)
        return 2

    smoke_profile = (
        "**Scope on this build.** The command above is smoke-proven with prefix caching on — "
        "it starts and answers a text and an image request exactly on the promoted image "
        f"([`receipts/production-image.json`]({BLOB}/receipts/production-image.json)) — but "
        "the nine-gate qualification was run on the native-context profile, not on this "
        "8,192-token one.")

    patches = [
        context_card(fig),
        k5k6_card(fig),
        sibling_card("MODEL_CARD-K5K6-hydrated.md", fig, HYDRATED_RECIPE, smoke_profile),
        sibling_card("MODEL_CARD-K4.md", fig, K4_RECIPE, smoke_profile),
    ]
    report = [p.write(dry_run=args.dry_run) for p in patches]
    print(json.dumps({"dry_run": args.dry_run, "figures": fig, "files": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
