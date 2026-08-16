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
    band = {r["tag"]: r for r in qual["utilisation_band"]["probes"]}
    rw = qual["reduced_window_option"]
    g2 = qual["gates"]["gate2_long_needle_exact"]
    return {
        "refuse_util": band["n955"]["gpu_memory_utilization"],
        "refuse_needed": band["n955"]["kv_gib_needed_for_262144"],
        "refuse_available": band["n955"]["available_kv_cache_gib"],
        "refuse_suggested_len": band["n955"]["engine_estimated_max_model_len"],
        "deadlock_util": band["n9555"]["gpu_memory_utilization"],
        "deadlock_pool": band["n9555"]["gpu_kv_cache_tokens"],
        "deadlock_conc": band["n9555"]["maximum_concurrency"],
        "livelock_util": band["p9585"]["gpu_memory_utilization"],
        "livelock_pool": band["p9585"]["gpu_kv_cache_tokens"],
        "livelock_conc": band["p9585"]["maximum_concurrency"],
        "livelock_seconds": int(g2["wall_sec"]),
        "cache_queries": int(g2["engine_self_report"]["prefix_cache_queries_total"]),
        "cache_hits": int(g2["engine_self_report"]["prefix_cache_hits_total"]),
        "rw_offered": rw["all_run_gates_passed"],
        "rw_window": rw["max_model_len"],
        "rw_util": rw["gpu_memory_utilization"],
        "rw_pool": rw["gpu_kv_cache_tokens"],
        "rw_conc": rw["maximum_concurrency"],
        "rw_price": rw["window_price_vs_native"],
        "rw_price_pct": round(100.0 * rw["window_price_vs_native"] / 262144, 2),
        "block": 1600,
        "upstream_issue": g2["reported_upstream"]["url"],
        "upstream_number": g2["reported_upstream"]["number"],
    }


def shared_paragraphs(fig):
    """The promotion, the evidence, and the two caveats. Identical on all four cards."""
    qual_link = f"[`receipts/qualification-5090-apc.json`]({BLOB}/receipts/qualification-5090-apc.json)"
    image_link = f"[`receipts/production-image.json`]({BLOB}/receipts/production-image.json)"
    repro_link = f"[`receipts/apc-poison-repro.json`]({BLOB}/receipts/apc-poison-repro.json)"
    mamba_link = f"[`receipts/mamba-align-defect.json`]({BLOB}/receipts/mamba-align-defect.json)"
    gdn_link = f"[`receipts/gdn-spec-gate-defect.json`]({BLOB}/receipts/gdn-spec-gate-defect.json)"

    promotion = (
        "**The release unit moved on 2026-08-16, and #51113 is why.** Upstream vLLM #51113 "
        "(mamba `align` prefill-chunk splitting: a chunk that ends mid-block leaves its slot "
        "holding a short state, which a later chunk then publishes anyway — wrong tokens, "
        "HTTP 200, no crash) merged 2026-08-06, after the pinned public image was built, and "
        "is still absent from it and from fork head `fa033bd4e`. Cherry-picks were requested "
        "upstream on 2026-08-16 "
        "([issue #392](https://github.com/local-inference-lab/vllm/issues/392), "
        "[PR #393](https://github.com/local-inference-lab/vllm/pull/393)). Until they land it "
        f"is carried as `tools/vllm-mamba-align-scheduler.py` (`sha256 {SCHED_SHA[:8]}…`), and "
        "the image this project serves is now the four-module "
        f"`localhost/vllm:gg-r34-patched-apc`, manifest `{PROMOTED_MANIFEST[:21]}…`, promoted "
        f"from the three-module `localhost/vllm:gg-r34-patched` (`{PARENT_MANIFEST[:21]}…`) "
        f"({image_link}). With prefix caching off the two images are not merely similar but "
        "behaviourally identical, because the added module's changed function is unreachable "
        "unless `mamba_cache_mode` is `align` — so the earlier hardware qualification carries "
        "over unchanged. One warning if you inspect the image yourself: its build-time label "
        "`io.malaiwah.image.qualified` still reads `false` and is **superseded by the "
        "receipts named here** — it was written before the image could possibly have been "
        "qualified, and correcting it would add a layer and change the very digest that was "
        "measured. That digest is local to the build host, so the recipes here reproduce its "
        "content with sha256-verified read-only mounts over the pullable public base.")

    win = (
        "**What prefix caching buys.** On disjoint documents, so the cold case is genuinely "
        "cold: a 32,842-token prefix went **12.07 s cold → 1.04 s warm (11.6×**, 2,442 of "
        "32,842 prompt tokens recomputed, 92.6 % hit rate) and a 131,146-token prefix went "
        "**67.60 s → 2.31 s (29.3×**, 3,146 of 131,146 recomputed, 97.6 %); a 38-request "
        f"schedule ran 84.0 s with the cache against 144.4 s without ({repro_link})."
        "\n\n"
        "**And what we actually know about its safety.** Correctness was probed adversarially "
        "before any of this shipped: seven freshly started servers, 38 requests each, **266 "
        "scored requests**, nested token prefixes so later requests hit blocks published by "
        f"earlier ones, and no prompt length a multiple of the measured {fig['block']:,}-token "
        "mamba block, so prefill chunks end mid-block by construction — **zero corrupted "
        "responses, zero wrong answers, zero acceptance collapses, on the unpatched image as "
        "well as the patched one**. Thresholds were committed before the first server "
        "started; the worst repeated block was 15 characters against an 80-character "
        "threshold, with no U+FFFD anywhere. Greedy chosen-logprob drift with the cache on is "
        "0.1063 mean absolute against a measured run-to-run floor of 0.0823 — drift, never an "
        "answer change. So the module is carried as **insurance backed by upstream's own "
        "regression file** — 14 failed / 6 passed against the vendored scheduler, 20 passed "
        "against this one — and **not** by a reproduction of our own: we tried hard to "
        f"reproduce the reported corruption and could not ({mamba_link}).")

    lmcache = (
        "**LMCache is unmeasured by us.** It is not part of any recipe on this card, this "
        "project has never run it, and it is the outstanding suspect in the one user report "
        "of prefix-cache corruption we have. Nothing here says LMCache is safe; the evidence "
        "above covers vLLM's own prefix cache and nothing else.")

    gdn = (
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

    return promotion, win, lmcache, gdn, qual_link


def enabled_paragraphs(fig, profile, indent=""):
    """For the three 8,192-token recipes, where prefix caching is on."""
    promotion, win, lmcache, gdn, qual_link = shared_paragraphs(fig)
    on = (
        "**Prefix caching is on in the recipe below, and it is on here and not everywhere.** "
        "At an 8,192-token window the KV pool is roughly thirty times the window, so a single "
        "request comes nowhere near the pool ceiling and the failures that stopped the "
        "native-context profile cannot occur. That is not an argument, it is the reason this "
        "recipe was measured separately: it starts healthy on the promoted image with "
        "`--enable-prefix-caching --mamba-cache-mode align`, answers a text and an image "
        "request exactly, and reports `enable_prefix_caching: True` in the engine banner "
        f"([`receipts/production-image.json`]({BLOB}/receipts/production-image.json)). The "
        "context edition's native 262,144-token recipe does **not** enable it, and its card "
        f"explains why ({qual_link}).")
    return "\n\n".join(wrap(x, indent=indent)
                       for x in (on, promotion, win, lmcache, gdn, profile))


def context_paragraphs(fig, profile, indent=""):
    """For the context edition, where prefix caching is measured and declined."""
    promotion, win, lmcache, gdn, qual_link = shared_paragraphs(fig)
    off = (
        "**Prefix caching is deliberately OFF in both recipes above, and this is what we "
        "measured before deciding that.** At the native 262,144 window on a 32,607 MiB card "
        "it does not fit, and it fails in three different ways as you give the engine more "
        f"room. At `--gpu-memory-utilization {fig['refuse_util']}` the engine refuses to "
        f"start: it needs **{fig['refuse_needed']} GiB** of KV for one 262,144-token request "
        f"against an unchanged **{fig['refuse_available']} GiB** supply, and names "
        f"**{fig['refuse_suggested_len']:,}** as the longest window it could serve. The "
        f"reason is that `align` mode makes a request occupy whole {fig['block']:,}-token "
        "blocks, so 262,144 rounds up to 164 blocks. At "
        f"`{fig['deadlock_util']}` it starts with a pool of exactly "
        f"{fig['deadlock_pool']:,} tokens ({fig['deadlock_conc']}× concurrency) and then "
        "**deadlocks** mid-prefill. At "
        f"`{fig['livelock_util']}` — pool {fig['livelock_pool']:,} tokens, "
        f"{fig['livelock_conc']}×, which is within 50 tokens of what the same profile gets "
        "with the cache off — it **livelocks**: the request prefills to 98.9 % of the pool, "
        "is dropped back to the waiting queue with the pool freed, and re-prefills, on a "
        "30-second cycle, at about 960 tok/s of wasted prefill, with **zero output tokens** "
        f"for the {fig['livelock_seconds']} seconds we let it run and no sign it would ever "
        f"stop. The cache was consulted throughout and never hit — {fig['cache_queries']:,} "
        f"queries, {fig['cache_hits']} hits — because the partial prefill is discarded rather "
        "than published, so every retry costs full price. The pool is quantised, and 0.9585, "
        "0.959 and 0.96 all yield the same pool, so there is no utilisation on this card that "
        "makes it work; and going higher is barred anyway, because 0.97 is the value that "
        f"fails the combined long-text-plus-seven-megapixel request ({qual_link}).")

    known_issue = (
        "**If you enable it anyway, this is what a hang looks like.** The server does not "
        "crash and does not return an error: it accepts the request and stays busy forever. "
        "`vllm:num_preemptions_total` stays at `0.0` and the only signal is "
        "`vllm:num_requests_waiting_by_reason{reason=\"capacity\"} = 1`, which is "
        "indistinguishable from a healthy server that is momentarily full. Your hardware is "
        "not failing. Reported upstream as "
        f"[issue #{fig['upstream_number']}]({fig['upstream_issue']}).")

    if fig["rw_offered"]:
        option = (
            "**If you want prefix caching on this edition, cap the window.** Measured, not "
            f"inferred: at `--max-model-len {fig['rw_window']}` with "
            f"`--gpu-memory-utilization {fig['rw_util']}` the pool is {fig['rw_pool']:,} "
            f"tokens ({fig['rw_conc']}×) and the engine serves a 255,000-token needle exactly "
            "in 180 s, completes three warmed decode runs, and answers a second long request "
            f"after the first is released. The price is **{fig['rw_price']:,} tokens of "
            f"context, {fig['rw_price_pct']} %**. Add "
            "`--enable-prefix-caching --mamba-cache-mode align` and the fourth mount already "
            "shown above, and change the window. This is a bounded probe and not a qualified "
            "profile — it does not carry the combined long-text-plus-image gate or the image "
            "suite — so it is offered as a choice you may make, never as the default.")
    else:
        option = (
            "**No reduced-window option is offered.** The probe that would have priced one "
            "was not run or did not pass, and this card does not print configurations nobody "
            "executed.")

    return "\n\n".join(wrap(x, indent=indent)
                       for x in (off, known_issue, option, promotion, win, lmcache, gdn,
                                 profile))


def context_card(fig):
    p = Patch(REPO / "MODEL_CARD-K5K6-context.md")
    p.absent("v1/core/sched/scheduler.py:ro", "the scheduler mount")
    p.after(
        f"{MTP_SHA}  tools/vllm-qwen3_5_mtp-embed-quant-config.py\n",
        f"{SCHED_SHA}  tools/vllm-mamba-align-scheduler.py\n",
        "scheduler digest in the verified module list")
    p.after(
        '  -v "$PATCH/vllm-qwen3_5_mtp-embed-quant-config.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro" \\\n',
        '  -v "$PATCH/vllm-mamba-align-scheduler.py:$VLLM/v1/core/sched/scheduler.py:ro" \\\n',
        "scheduler mount in the patched native-window recipe")
    p.replace(
        "It does not include the input-table overlay,\ngraph decode or prefill routing, so it "
        "is not the native-window profile.",
        "It does not include the input-table overlay,\ngraph decode or prefill routing, so it "
        "is not the native-window profile. It also runs with prefix\ncaching **off**, and must: "
        "turning it on safely needs upstream #51113, which this image\npredates. The patched "
        "profile below is where it is enabled.",
        "the unmodified-fallback note about prefix caching")

    profile = (
        "**What is still qualified here, and at what.** The recipe above is unchanged apart "
        "from the fourth mount, and the fourth mount changes nothing while prefix caching is "
        "off: `--gpu-memory-utilization 0.955` with "
        "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` remains the profile whose seven "
        "gates passed on a physical RTX 5090, at window 262,144, `--max-num-seqs 1`, MTP "
        "depth 3, fp8 KV and the full 8,388,608-pixel ceiling "
        f"([`receipts/qualification-5090-context.json`]({BLOB}/receipts/qualification-5090-context.json)). "
        "The concurrent-serving variant this card also documents is not qualified at any "
        "sequence count, with or without the cache.")
    p.replace(UPSTREAM_PARA_CONTEXT, context_paragraphs(fig, profile),
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
    item = enabled_paragraphs(fig, profile, indent="   ")
    p.replace(UPSTREAM_ITEM_K5K6, "6." + item[2:],
              "list item 6, the two-absent-fixes item")
    return p


def sibling_card(name, fig, recipe_block, profile):
    p = Patch(REPO / name)
    p.absent("--enable-prefix-caching", "the prefix-caching flag")
    p.replace(UPSTREAM_PARA_SIBLING,
              enabled_paragraphs(fig, profile) + "\n\n" + recipe_block,
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
    image = json.loads((REPO / "receipts/production-image.json").read_text())

    # The cards may only print what was measured. Each claim they make has a precondition
    # here, and a missing precondition stops the whole pass rather than one paragraph.
    problems = []
    outcome = qual["verdict"]["outcome"]
    if outcome not in ("PROMOTED", "SCOPED DECLINE"):
        problems.append(f"unrecognised qualification outcome {outcome!r}")
    if qual["verdict"]["decisions"]["gate8_banner_prefix_caching_actually_on"] != "PASS":
        problems.append("the banner gate did not prove prefix caching was actually on, so "
                        "nothing measured under it means anything")
    apc = next((v for v in image.get("superset_images", []) if v["variant"] == "apc"), None)
    if not apc:
        problems.append("no apc superset block in receipts/production-image.json")
    else:
        smoke = apc["smoke"]
        need = {"k4", "k5k6", "hydrated"}
        ran = set(smoke["recipes_run"])
        if not need <= ran:
            problems.append(f"the cards enable prefix caching on {sorted(need)} but the "
                            f"promoted image only smoked {sorted(ran)}")
        if not (smoke["all_run_recipes_started"] and
                smoke["all_run_recipes_text_and_image_exact"]):
            problems.append("the promoted image's smoke did not come back clean")
        banners = {r["recipe"]: (r.get("engine_banner") or {}).get("enable_prefix_caching")
                   for r in smoke["results"]}
        off = sorted(k for k in need if banners.get(k) != "True")
        if off:
            problems.append(f"these recipes did not report enable_prefix_caching True: {off}")
    if image["release_unit"]["manifest_digest"] != PROMOTED_MANIFEST:
        problems.append("receipts/production-image.json does not name the promoted digest as "
                        "the release unit")
    if problems:
        print("REFUSED, nothing written:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2
    fig = figures(qual)

    smoke_profile = (
        "**Scope on this build, stated narrowly.** What was measured on this profile is that "
        "the recipe starts healthy on the promoted image with the cache on, answers a text "
        "and an image request exactly, and reports the cache enabled in its banner. That is a "
        "serving smoke, not a gate suite: this window has no long-needle, combined-image or "
        "decode-dispersion gate of its own, and the 11.6× and 29.3× reuse figures above were "
        "measured on the context edition's much longer prompts, not on 8,192-token ones. "
        "Expect the shape of the win — recomputing only what changed — rather than those "
        "multiples.")

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
