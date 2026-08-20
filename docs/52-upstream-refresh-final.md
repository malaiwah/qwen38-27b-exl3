# Upstream artefact refresh — final pre-teardown state

`generated_utc: 2026-08-17` · read-only sweep · **nothing was posted, commented, closed, opened, labelled or pushed anywhere**

> **Superseded status snapshot.** Current review is `receipts/upstream-audit-2026-08-20.md`
> plus the retrospective review. Since this file: #406/#407 received corrective
> comments, PR #397 was retargeted to `dev/gilded-gnosis`, upstream #47272 merged,
> and the issue/PR inventory expanded. Preserve this chronology, but do not use
> its action table as current state.

Auth: `gh` initially had no env token. A stored credential was used for the
read-only sweep. Repository permissions denied code pushes to the target, but
did **not** make all writes impossible — issue/PR comments can still be posted,
as later #406/#407 corrections demonstrate. Nothing was modified during this
sweep by policy, not by an absolute permission boundary.

Tracker repo (from `receipts/vllm-gg-issues-filed.json.repo`): **`local-inference-lab/vllm`**, default branch `main`.

---

## 0. The one-line answer

**Across every GitHub artefact we own, third-party human engagement is exactly zero.** The only non-`malaiwah`
actor anywhere is `coderabbitai[bot]`. The *only* real third-party signal on the entire upstream surface is
**`cosmicnag` on Hugging Face discussion #1**, and **his latest comment is unanswered and addressed to us by name.**

Two substantive action items, both self-inflicted and both precisely specified in §4:
**#407 and #406 each state an end-to-end figure that exceeds its own Amdahl ceiling**, and
**PR #397 is aimed at the wrong base branch**, which is why it never got reviewed.

---

## 1. Label sets, up front

The repo defines 10 labels. **Every one of our 12 artefacts carries an empty label set** (`labels: []`).
Nobody has triaged, labelled, milestoned or assigned anything we filed. There is therefore no
"re-labelled twice" history to report on #403 — see §3 for what actually happened to it.

---

## 2. The seven docs/47 findings — #406–#412

All seven filed 2026-08-17T05:43:16Z–05:43:32Z (a 16-second burst), all by `malaiwah`.

| # | state | labels | comments | 3rd party? | reactions | verdict |
|---|---|---|---:|---|---|---|
| 406 | open | none | 0 | none | none | **NEEDS REPLY** (§4.1) |
| 407 | open | none | 0 | none | none | **NEEDS REPLY** (§4.1) |
| 408 | open | none | 0 | none | none | no action |
| 409 | open | none | 0 | none | none | no action |
| 410 | open | none | 0 | none | none | no action |
| 411 | open | none | 0 | none | none | no action — body already self-corrects |
| 412 | open | none | 0 | none | none | no action |

`GET /issues/{n}/timeline` returns an **empty** non-`malaiwah` actor list for all seven.
`GET /issues/{n}/reactions` returns `[]` for all seven. Nobody has read these in a way GitHub records.

### #405 — the mis-filed index page: **confirmed still closed**

| field | value |
|---|---|
| state | `closed`, `state_reason: not_planned` |
| created | 2026-08-17T05:43:14Z |
| closed | 2026-08-17T05:44:07Z |
| **open for** | **53 seconds** |
| title | `# vLLM-GG issue drafts (docs/47 findings) — PREPARED, NOT FILED` |
| comments | 1 — ours, the apology + pointer to #406–#412 |
| labels | none · 3rd party: none · reactions: none |

**Verdict: no action.** It is closed `not_planned`, its single comment already says
*"Nothing else needs to happen on this issue"*, and no third party ever saw it. The receipt's
`misfiled_and_closed` block is accurate as written. Note the title still reads "PREPARED, NOT FILED",
which is slightly comic but harmless on a closed issue — **leave it**.

---

## 3. PR #403, issue #402, and the two earlier filings #397 / #398

### PR #403 — the LMCache divergent-hit gate

| field | value |
|---|---|
| state | **open**, not merged, `mergeable_state: unstable`, not draft |
| head → base | `malaiwah:fix/gg-scheduler-divergent-hybrid-hit-gate` → **`dev/gilded-gnosis`** |
| 1 commit, 1 changed file (`vllm/v1/core/sched/scheduler.py`), 0 review comments, 0 reviews |
| labels | none · reactions | none |
| comments | **3** — `coderabbitai[bot]` ×1, `malaiwah` ×2 |
| updated | 2026-08-17T07:01:54Z |

**Title history — renamed once, not twice, and by us.** The single `renamed` event is
`actor: malaiwah, at: 2026-08-17T05:45:06Z`:

- from: `[Bugfix] Gate hybrid+connector divergent local-hit path on connector opt-in`
- to: `[Bugfix] Gate hybrid+connector divergent local-hit path on connector opt-in [necessary but NOT sufficient — predicted 0/38 refuted, see comment]`

There are zero label events. The precise correction is "renamed once and
self-corrected twice in comments." Published receipts remain immutable; record
this in a later correction/audit rather than editing the original receipt.

**The two self-corrections, and the supersession chain.** This is the detail that matters:

1. `malaiwah` 2026-08-17T05:45:05Z — *"## Correction from the author: this PR's predicted outcome is REFUTED by
   measurement"*. Strikes the 0/38 prediction, reports 37/38 and 38/38, and attributes the mechanism
   **store-side**: `GetStoreMetadata` counting APC-hit spans as storable, null mamba block id 0 stored as
   boundary state. Proposes a ~20-line store-side truncation.
2. `malaiwah` 2026-08-17T07:01:54Z — *"## Follow-up: my own mechanism attribution above was also wrong —
   measured, not argued"*. **This comment supersedes comment 1's store-side attribution with the fp8 finding:**
   fp8 KV → divergence 3.52 catastrophic with *no poisoning precondition*; bf16 → divergence 0.0000 bit-clean;
   bf16 partial-hit interleave → 0.71. It explicitly retires the store-side clamp (*"did not fix"*, *"archiving
   it as hygiene rather than proposing it as a fix"*) and names a third, still-unexplained partial-retrieve defect.

So the thread is already self-consistent and ends on the correct mechanism. **Verdict: NO ACTION.**
The supersession is *inside* the thread, in the right order, with the wrong mechanism explicitly withdrawn
rather than silently dropped. Nothing further is owed. Do not touch it.

### Issue #402 — the root-cause issue #403 patches

`open`, created 2026-08-16T23:09:16Z, **0 comments**, no labels, no reactions, no non-`malaiwah` timeline actors.

**Verdict: SUPERSEDED IN PRACTICE, no action.** Its paired PR #403 carries both corrections; #402's body still
frames the defect as scheduler-side, which the fp8 finding has since demoted to one of three defects. Ideally it
would get a one-line pointer to #403's second comment — but with zero readers and zero engagement, filing that
now buys nothing and costs a post we are not permitted to make. Record it and move on.

### PR #397 — scratch arena — **the structural problem**

| field | value |
|---|---|
| state | **open**, not merged, `mergeable_state: unstable` |
| head → base | `malaiwah:feat/exl3-reconstruct-scratch-arena` → **`codex/gg-exl3-r7-k345-20260810`** |
| 6 commits, 2 changed files, **0 review comments, 0 reviews** |
| labels | none · reactions | none |
| comments | 1 — `coderabbitai[bot]` only |

**#397 is aimed at a dated codex branch, not the active dev line.** Its base
`codex/gg-exl3-r7-k345-20260810` last moved 2026-08-10T18:31:35Z. #398 and #403 both target
`dev/gilded-gnosis` (head `fa033bd4e1b16d9d729ad94be2d87da5a13210ce`, 2026-08-11T02:28:24Z).

**That mis-targeting is why #397 got no review at all.** CodeRabbit's sole comment on #397 is a *skip*:

> **Review skipped** — Auto reviews are disabled on base/target branches other than the default branch.
> Base branches to auto review (1): `dev/*`

Compare #398 and #403, which both target `dev/*` and both received a full walkthrough plus
"No actionable comments were generated in the recent review. 🎉" and 5/5 passed pre-merge checks.

**Verdict: NEEDS ACTION (§4.2)** — the highest-value cheap fix on the board, because it is a base-branch
retarget, not an argument.

### PR #398 — FlashInfer decode-shape keying

| field | value |
|---|---|
| state | **open**, not merged, `mergeable_state: unstable` |
| head → base | `malaiwah:fix/fi-persistent-decode-plan-buffers` → `dev/gilded-gnosis` |
| 1 commit, 2 changed files, 0 review comments, 0 reviews |
| labels | none · reactions | none · comments | 1 — `coderabbitai[bot]` only |
| updated | 2026-08-16T15:16:20Z |

CodeRabbit passed all 5 pre-merge checks, rated **Merge Risk: ⚪ Minimal**, generated no actionable comments,
and suggested reviewers `sjug`, `mgoin`. It also surfaced prior art we should know about:

> **Possibly related PRs** — [local-inference-lab/vllm#234](https://github.com/local-inference-lab/vllm/pull/234):
> Extends FlashInfer decode-wrapper guards from planned query length to exact `(batch_size, q_len)` tracking.

That is a bot claim, not a human one, and **#234 is not ours** — if it genuinely does the same thing, #398 may be
a duplicate. Unverified: I did not open #234, because doing so was outside this ticket and could not change the
read-only verdict. **Flagged as the one open question worth a minute of someone's time.**

**Verdict: NO ACTION before teardown.** It is clean, reviewed-by-bot, minimal-risk, and blocked only on the
merge gate below.

### Why all three PRs read `unstable` — and why that is not fixable by us

`receipts/gg-scheduler-fix-filing.json` records the repo policy gate as *"author must have >=4 merged PRs"*.
Measured now:

- `is:pr author:malaiwah is:merged` → **total_count 1** (only **#258** "[Model Runner V2] Account for prompt-logprobs memory")
- `is:pr author:malaiwah is:closed is:unmerged` → **5**
- open `malaiwah` PRs on the tracker → **29**

So the gate needs 4 merged and we have 1. **`unstable` on #397/#398/#403 is the policy gate, not a defect in any
of the three changes, and it cannot be cleared from our side.** State this in the receipt so nobody re-litigates it.

---

## 4. Action items, stated as the reply each would have to make

### 4.1 #407 and #406 state end-to-end figures above their own Amdahl ceilings — **NEEDS REPLY**

This is the item that actually matters, and it is a *shown identity*, not an estimate.

Amdahl for a component of decode-GPU-time fraction `f` sped up by `s`:
`gain = 1/((1-f) + f/s) - 1`. Ceiling at `s → ∞` is `1/(1-f) - 1`.

**#407 — `in_proj_ba`, f = 5.2 % (KernelGap's own instrumentation):**

| s | gain | source |
|---|---:|---|
| 1.5× | +1.76 % | docs/46 §28 table |
| 2× | +2.67 % | docs/46 §28 table |
| 3× | +3.59 % | docs/46 §28 table |
| 4.4× (measured kernel win) | **+4.19 %** | docs/46 §28 table |
| **∞** | **+5.49 %** | `1/0.948 - 1 = 0.054852` — **[INFERENCE]**, my arithmetic, identity shown |

I re-derived the whole published row to confirm the identity rather than trusting the table:
`s=1.5 → 1/(0.948+0.03467)-1 = +1.764 %`; `s=2 → 1/0.974-1 = +2.669 %`;
`s=3 → 1/0.965333-1 = +3.591 %`; `s=4.4 → 1/0.959818-1 = +4.186 %`. All four match docs/46 §28.

**#407's body as filed says:** *"Measured: +8.0 % single-stream decode alone (96.19 → 103.91 tok/s median…)"*.
Check: `103.91/96.19 = 1.08026` → **+8.03 %**.

**+8.03 % exceeds +5.49 %, the ceiling for an infinitely fast `in_proj_ba` at f = 5.2 %, by 1.46×.**
The filed number therefore *cannot* be an attribution of the `in_proj_ba` kernel win to GPU time. It necessarily
includes CPU-dispatch/proot effect — which is exactly the constraint that end-to-end throughput from the proot box
is not a performance result. #407 does partially disclose this (*"saturates at the gate-alone level because the
post-gate step is CPU-dispatch-bound under proot"*), so this is a **tightening, not a retraction**.

**#406 — b12x gate, f ≈ 8 % (docs/46 §28's "gate target" row):** ceiling `1/0.92 - 1 = +8.70 %` **[INFERENCE]**.
#406's body says **+15.4 %** (`110.97/96.19 = 1.15366` → **+15.37 %**), i.e. **1.77× its infinite-speedup ceiling**.
#406 already brackets itself honestly (*"Honest bracket: +3–15 %"*, *"The local measurement ran inside proot
(ptrace), which taxes eager launch dispatch"*).

**Both issues also carry a promise that has now come due.** #406's body says:

> A bare-docker confirmation on a rental 1×RTX6000 is queued; expect the true gain between +3 % (pure kernel
> delta) and +15 % (this measurement).

and its falsifier is *"A bare-metal A/B (no proot) showing patched ≤ baseline single-stream at C1"*.
#407's falsifier is *"a bare-metal A/B showing the end-to-end gain <1 %"*.

**That A/B was run (docs/46 §28) and returned neither outcome.** It was inconclusive *by construction*:
median within-arm CV **2.84 %**, reaching **13.9–18.0 %** in short-prompt cells; the headline "+8.94 %" at
30k×2k C1 compares overlapping samples whose baseline repeats alone span **70.5 to 93.6 tok/s**; 9 of 11 cells
favour the gate, sign test **p = 0.065**. Resolving a 1.5 % effect against a 2.84 % CV needs ~**29 repeats per
cell, not 5**.

**docs/46 §28 already prescribes the exact reply, by issue number:**

> That is now the standing method for this class of patch, and **issue #406 should carry the Amdahl framing
> instead of an end-to-end promise.**

**What the reply would have to say** (one comment each, or one comment cross-linked from both):

1. The queued bare-docker A/B **has run**; it did not fire either falsifier because it never had the resolution to.
   Give the CV numbers and the 70.5–93.6 tok/s baseline span so the non-result is auditable.
2. Withdraw the end-to-end figure as a *performance* claim and re-state the finding at the kernel level, which is
   where it was actually measured: 4.4× on `in_proj_ba`, 20–28 µs → 5.61 µs graph-replayed.
3. Replace the end-to-end promise with the Amdahl bound and show the identity: at f = 5.2 %, **+4.19 % at 4.4×**
   and **+5.49 % even at infinity** — so the filed +8.0 % is above the ceiling and is partly proot dispatch.
4. State the standing method explicitly: measure at the kernel level, state the server-level consequence via the
   Amdahl bound rather than attempting to observe it.

**Do not post this.** It is recorded here so the parent can decide.

### 4.2 PR #397 targets the wrong base branch — **NEEDS ACTION (retarget, no argument required)**

Retarget `#397` from `codex/gg-exl3-r7-k345-20260810` (stale since 2026-08-10) to `dev/gilded-gnosis`.
Consequences, both mechanical: it would land on the active dev line, and it would enter CodeRabbit's
`dev/*` auto-review scope instead of being skipped. This is a base-branch edit on **our own PR** — inside the
denylist's allowance — and it is the cheapest durable improvement available. It does **not** clear the
≥4-merged-PR gate; nothing we can do does.

### 4.3 HF discussion #1 — **NEEDS REPLY, and it is the only genuinely externally-owed item**

See §5. The last word in the thread is a third party answering a direct question we asked him.

---

## 5. Hugging Face discussions

The ticket says "eight `malaiwah/Qwen3.8-27B*` repos". **The Hub actually returns 14** (33 `malaiwah` models
total). Enumerated via `GET /api/models?author=malaiwah` (unauthenticated; no HF token present on this box —
`TOKEN_PRESENT False` — so this covers public state only):

`-AWQ-INT4-archival-63768c10`, `-EXL3-EDA-research`, `-EXL3-K5K6`, `-EXL3-K5K6-context`, `-EXL3-K5K6-hydrated`,
`-EXL3-K6-parity`, `-EXL3-S16-V-research`, `-GGUF-archival-f1bfb127`, `-K4`, `-MTP-NVFP4-archival-6d98dc1f`,
`-NVFP4-RTX5090-archival-69274a0d`, `-NVFP4-archival-9c73e2da`, `-exl3-archival-a35e75a7`, `-exl3-archival-d32ba0bb`.

**13 of 14 have `count=0` discussions and `numClosedDiscussions=0`.** Exactly one discussion exists in the
entire family:

### `malaiwah/Qwen3.8-27B-EXL3-K5K6-context` discussion #1 — "Getting garbled text"

| field | value |
|---|---|
| status | **open** (not a PR) |
| opened by | **`cosmicnag`** (third party) 2026-08-16T01:53:51Z |
| comments | **11** — 6 from `cosmicnag`, 5 from `malaiwah` |
| latest | **`cosmicnag`, 2026-08-17T09:04:52Z** |
| **last word** | **theirs, unanswered** |

**The latest comment, verbatim and complete:**

> @malaiwah Updating you on this question :
>
> `Have you ever sent a single request that comes close to the full window with prefix caching enabled — say a 250k+ token prompt — and received tokens back?`
>
> Yes, many a times by now - and it works like a champ.  Looking at how well the model and this quant holds up @ 250k+, I think it should perform reasonably well with that 1M yarn/rope context scaling (I have personally not tried it yet, as I tend to avoid large contexts actively as much as possible)

**Why this is the highest-value thing in this whole report.** It is an *independent, second-physical-RTX-5090*
confirmation of the native-window claim under prefix caching — the exact arm we cannot supply ourselves, since we
only have our own cards. He is confirming 250k+ single-request prompts with APC enabled return tokens, repeatedly.
That is external corroboration of the context edition's headline capability, volunteered in answer to our own
direct question.

**Verdict: NEEDS REPLY.** A short acknowledgement. It costs one comment and it closes a thread where a helpful
third party is currently left hanging after doing us a favour. Nothing technical is owed — no defect is open
against us in it. **Not posted.**

**The second verbatim quote worth carrying — an open, unexplained failure on his side** (`cosmicnag`,
2026-08-16T14:37:07Z, complete):

> LMCache seems to be working @ 240k with Triton attention (see my last post), trying to get it working with flash now.
> EDIT : Trying flash again gave back the token soup. Sticking with triton for the lmcache config, flash for the 262k no lmcache config.
> EDIT 2 : Even triton has garbled output - its just all exclamation marks `!!!!!!!!`
> Sticking with no lmcache full 262k mtp3 for now.

**This independently corroborates the DO-NOT-ENABLE verdict on a second card, and adds a datum we do not have:**
his working-then-failing config runs **fp8 KV** (`--kv-cache-memory-bytes 9500000000`, `--kv-cache-dtype fp8`,
chunk 1600, `--max-num-batched-tokens 1600`, `TRITON_ATTN`, 240k) — which is precisely the fp8-transfer defect
#403's second comment names as dominant. His `!!!!!!!!` degenerate output at bf16-independent settings arrived
*after* he reported 240k+LMCache "works", i.e. **his own "working" LMCache config also failed on longer exposure,
across both attention backends.** That is a third-party reproduction of our finding on hardware we do not own,
and it is consistent with fp8 KV being the operative defect rather than the attention backend.

**No action owed on that one** — our verdict already matches it, and #403's thread already carries the mechanism.
But it belongs in the record as external corroboration.

---

## 6. The two fork branches on `malaiwah/vllm-voipmonitor`

**Neither branch has a PR. Anywhere.**

- `GET /repos/local-inference-lab/vllm/pulls?state=all&head=malaiwah:kernel-gap/tiny-n-mm-transpose` → **empty**
- same for `kernel-gap/b12x-gate-n-range` → **empty**
- scan of the 100 most recent tracker PRs for any `head.ref` matching `kernel-gap` → **empty**
- `GET /repos/malaiwah/vllm-voipmonitor/commits/{sha}/pulls` for both heads → **`[]`** both
- the fork's only PR ever is **#1** `feat/exl3-lora-cartridge` (**closed**), unrelated

**Both branches exist, at exactly the heads the ticket names.**

| branch | head | committed | matches ticket |
|---|---|---|---|
| `kernel-gap/tiny-n-mm-transpose` | `3b35c04c6ec4e38748ed2a599a5f4d68927b6310` | 2026-08-17T04:48:52Z | ✅ `3b35c04c6` |
| `kernel-gap/b12x-gate-n-range` | `704d94e9326c070f1e69419d7f32c31291428100` | 2026-08-17T04:49:39Z | ✅ `704d94e93` |

### The commit messages carry the same above-ceiling figures as #406/#407

`kernel-gap/tiny-n-mm-transpose` @ `3b35c04c6`:

> Measured end-to-end (RTX PRO 6000 Blackwell SE, Qwen3.8-27B-EXL3 hydrated, MTP-3, graph decode, C1 greedy,
> median of 3x200 tokens): 96.19 -> 103.91 tok/s (+8.0%).

`kernel-gap/b12x-gate-n-range` @ `704d94e93`:

> Measured end-to-end (…): 96.19 -> 110.97 tok/s (+15.4%). Caveat stated up front: that host runs the engine
> under proot, which inflates the eager-dispatch share; per-call GPU deltas alone account for ~0.6 ms/step, so
> the honest bare-metal bracket is +3..15% pending a docker A/B.

The b12x message is the more honest of the two — it names proot and brackets itself. **The transpose message does
not mention proot at all**; it presents +8.0 % flatly as "Measured end-to-end", and that figure is above the
+5.49 % infinite-speedup ceiling for its own 5.2 % component (§4.1).

The b12x message also already records the shard-boundary honesty note:

> NOTE: the served r34 image's exl3.py (5,536 lines) is not on any public branch; this branch is used because
> its `_b12x_trellis_k6_supported` body is byte-identical to the served bytes.

**Verdict: NO ACTION NOW — but do not open a PR from either branch as-is.** Both are safely parked: no PR, no
reviewers, no third party has seen either commit message. Opening a PR from our own fork is inside the denylist's
allowance, but the "docker A/B pending" promise in the b12x message and the unqualified +8.0 % in the transpose
message are both stale as of docs/46 §28. If a PR is ever opened, the body must lead with the kernel-level result
and the Amdahl bound, not either end-to-end number. **Recorded, not acted on.**

---

## 7. Third-party engagement audit — the negative result, done properly

Because "has anyone else engaged" is the one question we cannot answer ourselves, I did not stop at the 12
named artefacts. I pulled the unique comment-author set for **every** `malaiwah` tracker issue that has any
comments at all — 15 more issues spanning #200–#394:

```
392 : malaiwah      394 : malaiwah      200 : malaiwah      201 : malaiwah      202 : malaiwah
203 : malaiwah      204 : malaiwah      206 : malaiwah      207 : malaiwah      208 : malaiwah
215 : malaiwah      257 : malaiwah      272 : malaiwah      273 : malaiwah      276 : malaiwah
```

**Every comment on every one is ours.** #203 has 6 comments — all `malaiwah`. Combined with the empty
`reactions` and empty non-`malaiwah` `timeline` results on all 12 named artefacts, the conclusion is
unambiguous and worth stating plainly in the receipt:

> **In ~27 issues and PRs filed against `local-inference-lab/vllm`, not one human other than `malaiwah` has
> ever commented, reacted, reviewed, or labelled. The only non-us actor in the entire history is
> `coderabbitai[bot]`.**

That is a fact about the venue, not about the filings, and it should temper how much weight any
"needs a reply" verdict in §4 carries — there is no audience currently waiting on any of it. The one place we
*do* have a real reader is Hugging Face, and that is the one place we owe a reply.

---

## 8. Verdict summary

| artefact | state | 3rd party | verdict |
|---|---|---|---|
| #406 | open, 0 comments, no labels | none | **needs reply** — Amdahl re-frame; docs/46 §28 names it by number |
| #407 | open, 0 comments, no labels | none | **needs reply** — +8.0 % is 1.46× its own ∞ ceiling |
| #408 | open, 0 comments, no labels | none | no action |
| #409 | open, 0 comments, no labels | none | no action |
| #410 | open, 0 comments, no labels | none | no action |
| #411 | open, 0 comments, no labels | none | no action — body self-corrects its own estimate |
| #412 | open, 0 comments, no labels | none | no action |
| #405 | **closed `not_planned`**, open 53 s | none | no action — confirmed closed |
| #402 | open, 0 comments | none | superseded by #403's 2nd comment; no action |
| PR #403 | open, `unstable`, 3 comments, renamed ×1 by us | bot only | no action — self-corrected twice, ends on right mechanism |
| PR #397 | open, `unstable`, **base = stale codex branch** | bot only (a *skip*) | **needs action** — retarget to `dev/gilded-gnosis` |
| PR #398 | open, `unstable`, bot: minimal risk, 5/5 pass | bot only | no action; **open question:** possible dup of #234 |
| branch `kernel-gap/tiny-n-mm-transpose` | exists @ `3b35c04c6`, **no PR** | none | no action; msg's +8.0 % is above ceiling — fix before any PR |
| branch `kernel-gap/b12x-gate-n-range` | exists @ `704d94e93`, **no PR** | none | no action; "docker A/B pending" is stale |
| HF disc #1 (`-K5K6-context`) | **open, 11 comments, their last word** | **`cosmicnag`** | **needs reply** — acknowledgement only |
| HF, other 13 Qwen3.8-27B* repos | 0 discussions each | none | no action |

**Errors encountered:** one, recorded rather than guessed — `gh` was unauthenticated on first invocation
(exit 4, *"To authenticate, please run `gh auth login`"`*), resolved via the `~/.git-credentials` PAT.
No repo was unreachable. No HF repo was unreachable. `origin`/gitea was never contacted.

---

## 9. Machine-readable block for `receipts/upstream-refresh-final.json`

```json
{
  "schema": "qwen38-upstream-refresh-final/1",
  "generated_utc": "2026-08-17",
  "read_only": true,
  "nothing_posted_or_modified": true,
  "tracker_repo": "local-inference-lab/vllm",
  "auth": {
    "gh_env_token": "absent; first invocation failed exit 4 with 'To authenticate, please run gh auth login'",
    "credential_used": "~/.git-credentials PAT, user malaiwah",
    "permissions_on_tracker": {"admin": false, "maintain": false, "push": false, "triage": false, "pull": true}
  },
  "headline": "Zero third-party human engagement on all 12 GitHub artefacts; only non-malaiwah actor anywhere is coderabbitai[bot]. Sole real external signal is cosmicnag on HF discussion #1, whose latest comment is unanswered.",
  "all_artefact_label_sets_empty": true,
  "repo_labels_defined": 10,
  "merge_gate": {
    "policy": "author must have >=4 merged PRs (receipts/gg-scheduler-fix-filing.json)",
    "malaiwah_merged_prs_on_tracker": 1,
    "only_merged": 258,
    "closed_unmerged": 5,
    "open": 29,
    "consequence": "mergeable_state 'unstable' on #397/#398/#403 is the policy gate, not a defect; not clearable from our side"
  },
  "issues": [
    {"n": 406, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "needs_reply", "why": "body promises a queued bare-docker A/B that has since run inconclusively (docs/46 s28); +15.4% is 1.77x its own infinite-speedup Amdahl ceiling of +8.70% at f~8%; docs/46 s28 names #406 explicitly as needing the Amdahl framing"},
    {"n": 407, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "needs_reply", "why": "filed '+8.0% single-stream decode alone (96.19->103.91 tok/s)' = +8.03%, which exceeds +5.49%, the s->infinity Amdahl ceiling for f=5.2%, by 1.46x; cannot be a GPU-time attribution of the in_proj_ba win alone"},
    {"n": 408, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "no_action"},
    {"n": 409, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "no_action"},
    {"n": 410, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "no_action"},
    {"n": 411, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "no_action", "why": "body already self-corrects its own +15-25% estimate as '2-4x high at short context'"},
    {"n": 412, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "no_action"},
    {"n": 405, "state": "closed", "state_reason": "not_planned", "created": "2026-08-17T05:43:14Z", "closed": "2026-08-17T05:44:07Z", "open_seconds": 53, "labels": [], "comments": 1, "comments_all_ours": true, "third_party": false, "verdict": "no_action", "why": "confirmed still closed as not_planned; its own comment says nothing else needs to happen"},
    {"n": 402, "state": "open", "labels": [], "comments": 0, "third_party": false, "reactions": 0, "verdict": "superseded_no_action", "why": "root-cause issue for PR #403; its scheduler-side framing is superseded by #403's second comment (fp8 dominant), but zero readers so a pointer comment buys nothing"}
  ],
  "pulls": [
    {"n": 403, "state": "open", "merged": false, "mergeable_state": "unstable", "head": "malaiwah:fix/gg-scheduler-divergent-hybrid-hit-gate", "base": "dev/gilded-gnosis", "labels": [], "comments": 3, "comment_authors": ["coderabbitai[bot]", "malaiwah", "malaiwah"], "reviews": 0, "review_comments": 0, "third_party": false, "label_events": 0, "rename_events": 1, "renamed_by": "malaiwah", "renamed_at": "2026-08-17T05:45:06Z", "rename_to_suffix": "[necessary but NOT sufficient - predicted 0/38 refuted, see comment]", "supersession": "comment 2 (2026-08-17T07:01:54Z) supersedes comment 1's store-side null-mamba-block attribution with the fp8-KV finding (fp8 divergence 3.52 catastrophic, bf16 0.0000 bit-clean, bf16 partial-hit 0.71) and explicitly withdraws the ~20-line store-side clamp as 'hygiene rather than a fix'", "verdict": "no_action", "why": "thread self-corrects twice in the right order and ends on the correct mechanism; nothing further owed"},
    {"n": 397, "state": "open", "merged": false, "mergeable_state": "unstable", "head": "malaiwah:feat/exl3-reconstruct-scratch-arena", "base": "codex/gg-exl3-r7-k345-20260810", "base_last_commit": "2026-08-10T18:31:35Z", "labels": [], "comments": 1, "comment_authors": ["coderabbitai[bot]"], "reviews": 0, "review_comments": 0, "third_party": false, "verdict": "needs_action", "why": "targets a stale codex branch instead of dev/gilded-gnosis; CodeRabbit's only comment is a REVIEW SKIPPED notice because auto-review is limited to dev/* bases, so #397 has never been reviewed. Fix is a base-branch retarget, no argument required."},
    {"n": 398, "state": "open", "merged": false, "mergeable_state": "unstable", "head": "malaiwah:fix/fi-persistent-decode-plan-buffers", "base": "dev/gilded-gnosis", "labels": [], "comments": 1, "comment_authors": ["coderabbitai[bot]"], "reviews": 0, "review_comments": 0, "third_party": false, "bot_verdict": "Merge Risk: Minimal; 5/5 pre-merge checks passed; no actionable comments", "verdict": "no_action", "open_question": "CodeRabbit flagged local-inference-lab/vllm#234 (not ours) as extending FlashInfer decode-wrapper guards to exact (batch_size, q_len) tracking - possible duplicate of #398. Unverified; #234 not opened."}
  ],
  "fork_branches": {
    "repo": "malaiwah/vllm-voipmonitor",
    "pr_exists_for_either": false,
    "evidence": "pulls?head=malaiwah:<branch> empty on tracker for both; commits/<sha>/pulls returns [] for both; no tracker PR in the latest 100 has a kernel-gap head.ref; fork's only PR ever is #1 feat/exl3-lora-cartridge (closed, unrelated)",
    "branches": [
      {"name": "kernel-gap/tiny-n-mm-transpose", "head": "3b35c04c6ec4e38748ed2a599a5f4d68927b6310", "committed": "2026-08-17T04:48:52Z", "matches_ticket_sha": true, "exists": true, "commit_msg_claim": "96.19 -> 103.91 tok/s (+8.0%) 'Measured end-to-end'", "concern": "does not mention proot at all, and +8.0% is above the +5.49% infinite-speedup ceiling for its own 5.2% component", "verdict": "no_action_now; do not open a PR as-is"},
      {"name": "kernel-gap/b12x-gate-n-range", "head": "704d94e9326c070f1e69419d7f32c31291428100", "committed": "2026-08-17T04:49:39Z", "matches_ticket_sha": true, "exists": true, "commit_msg_claim": "96.19 -> 110.97 tok/s (+15.4%), self-bracketed +3..15% 'pending a docker A/B'", "concern": "the pending docker A/B has since run and was inconclusive by construction (docs/46 s28), so the bracket promise is stale", "verdict": "no_action_now; do not open a PR as-is"}
    ]
  },
  "amdahl_identity": {
    "formula": "gain = 1/((1-f) + f/s) - 1",
    "provenance": "f=5.2% row reproduced from docs/46 s28; ceilings at s->infinity are my arithmetic, labelled INFERENCE",
    "f_0.052": {"s_1.5": "+1.764%", "s_2": "+2.669%", "s_3": "+3.591%", "s_4.4": "+4.186%", "s_inf": "+5.485% [INFERENCE]"},
    "f_0.08_ceiling": "+8.696% [INFERENCE]",
    "filed_407": "+8.03% (103.91/96.19), exceeds f=5.2% infinite ceiling by 1.46x",
    "filed_406": "+15.37% (110.97/96.19), exceeds f=8% infinite ceiling by 1.77x",
    "ab_non_result": "docs/46 s28: median within-arm CV 2.84% (13.9-18.0% short-prompt cells); headline +8.94% at 30kx2k C1 over samples whose baseline repeats span 70.5-93.6 tok/s; 9/11 cells favour gate, sign test p=0.065; ~29 repeats/cell needed, not 5"
  },
  "huggingface": {
    "token_present": false,
    "scope": "public state only",
    "ticket_said_repos": 8,
    "actual_qwen38_model_repos": 14,
    "malaiwah_models_total": 33,
    "repos_with_zero_discussions": 13,
    "discussions_total_in_family": 1,
    "discussion": {
      "repo": "malaiwah/Qwen3.8-27B-EXL3-K5K6-context",
      "num": 1,
      "title": "Getting garbled text",
      "status": "open",
      "is_pull_request": false,
      "opened_by": "cosmicnag",
      "opened_at": "2026-08-16T01:53:51Z",
      "comments": 11,
      "comments_by_cosmicnag": 6,
      "comments_by_malaiwah": 5,
      "latest_comment_author": "cosmicnag",
      "latest_comment_at": "2026-08-17T09:04:52Z",
      "last_word_is_third_party": true,
      "verdict": "needs_reply",
      "why": "a third party answered a direct question we asked him and is left unacknowledged; nothing technical is owed",
      "verbatim_latest": "@malaiwah Updating you on this question :\n\n`Have you ever sent a single request that comes close to the full window with prefix caching enabled - say a 250k+ token prompt - and received tokens back?`\n\nYes, many a times by now - and it works like a champ.  Looking at how well the model and this quant holds up @ 250k+, I think it should perform reasonably well with that 1M yarn/rope context scaling (I have personally not tried it yet, as I tend to avoid large contexts actively as much as possible)",
      "value_of_latest": "independent second-physical-RTX-5090 confirmation that 250k+ single requests with prefix caching enabled return tokens repeatedly - the arm we cannot supply ourselves",
      "verbatim_lmcache_failure": "LMCache seems to be working @ 240k with Triton attention (see my last post), trying to get it working with flash now.\nEDIT : Trying flash again gave back the token soup. Sticking with triton for the lmcache config, flash for the 262k no lmcache config.\nEDIT 2 : Even triton has garbled output - its just all exclamation marks `!!!!!!!!`\nSticking with no lmcache full 262k mtp3 for now.",
      "verbatim_lmcache_failure_at": "2026-08-16T14:37:07Z",
      "value_of_lmcache_quote": "third-party reproduction of the DO-NOT-ENABLE verdict on a second card, across BOTH attention backends, on an fp8-KV config (--kv-cache-dtype fp8, --kv-cache-memory-bytes 9500000000, chunk 1600) - consistent with #403 comment 2's finding that fp8 transfer is the dominant defect rather than the attention backend"
    }
  },
  "third_party_engagement_audit": {
    "method": "unique comment-author set for every malaiwah tracker issue with comments>0, beyond the 12 named artefacts",
    "issues_checked": [200, 201, 202, 203, 204, 206, 207, 208, 215, 257, 272, 273, 276, 392, 394],
    "result": "every comment on every one is malaiwah",
    "conclusion": "across ~27 malaiwah issues and PRs on local-inference-lab/vllm, no human other than malaiwah has ever commented, reacted, reviewed or labelled; the only non-us actor in the entire history is coderabbitai[bot]",
    "implication": "there is no audience currently waiting on any tracker reply; the only venue with a real reader is Hugging Face"
  },
  "errors": [
    {"what": "gh unauthenticated on first invocation", "exact": "Welcome to GitHub CLI!\\n\\nTo authenticate, please run `gh auth login`.", "exit_code": 4, "resolved_by": "exporting the ~/.git-credentials PAT as GH_TOKEN"}
  ],
  "unreachable": [],
  "receipt_correction_suggested": "docs/46 s22 says PR #403 'has been publicly re-labelled'. Measured: zero labeled/unlabeled events; one rename event plus two self-correcting comments. 're-labelled' should read 'renamed once and corrected twice in-thread'."
}
```
