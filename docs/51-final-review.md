# 51. Final review: a deliberate gap hunt before teardown

**Written at the end of the program, with all three rentals about to be destroyed.** The point of this
document is to record that a search for gaps actually happened, what it found, and what was done — not to
assert that everything is fine.

## Method

Four mechanical checks over the 1,229 tracked files, plus a read of the open-item state:

1. every `receipts/*.json` path referenced by any doc or card, cross-checked against `git ls-files`;
2. every tracked receipt, checked for whether *anything* references it;
3. a grep for unresolved markers (`TODO`, `FIXME`, `TBD`, `pending`) across docs and cards;
4. the todo list's blocked and dropped items, checked against their written reasons.

## What it found, and what was done

### FIXED — a promised receipt that never existed (plan drift)

`docs/46 §3` specified a single summary receipt `receipts/tb21-speedrun-ladder.json` "feeding the card
table". **It was never created.** In practice it was replaced by **15 per-tier `tb21-ladder-*.json` receipts
plus the cross-arm `tb21-8x-topology-ladder.json`**, which carry the same content at finer grain. A reader
following the plan would have hunted for a file that does not exist. The line now says so explicitly.

### FIXED — a stale claim that our own later work had already overturned

`docs/29` still read: *"the rental figure is bracketed +3-15 % pending a bare-metal A/B."* By the time of
this review that sentence was wrong twice over — the A/B had run **and** the bracket had been withdrawn,
because the kernel-level re-scope showed **both endpoints exceed their own Amdahl ceiling** (transpose bounds
at +3.23 % wall-clock, gate at +2.23 %, against a published floor of +3 %). Corrected in place with a pointer
to `receipts/kernel-amdahl-bound.json`.

### NOT A DEFECT — one dead link that was my own bug

The check flagged `receipts/tb21-metrics-1x-hyd-rehearsal.json` as referenced-but-missing. It is **not
missing**: the reference is to the `.jsonl` file, which exists and is tracked. My regex matched `\.json` as a
prefix of `.jsonl`. Recorded because a check that produces false positives is worth knowing about — the next
person running it should match on a word boundary.

### ACCEPTED — 147 unreferenced top-level receipts

Of 840 tracked receipts, **271 are referenced nowhere; 147 of those are top-level** (the remaining 124 are
members of raw-evidence directories whose *parent* is cited, which is correct by design). The unreferenced
top-level ones are overwhelmingly **superseded intermediates** — `decode-parity-*`, `gguf-report-*`,
`card-consistency-audit*`, `converter-honesty-*` — kept deliberately under the preserve policy so that a
later reader can audit how a conclusion was reached rather than only its final form. **This is a considered
state, not an oversight**: deleting them would make the published numbers less auditable, and linking all 147
into prose would make the docs unreadable. Recorded so that the count is not mistaken for rot.

### ACCEPTED — remaining `pending` strings

The grep surfaced eight, all legitimate: six are ordinary prose (`spending`, `pending P0.1 re-derivation` in
a superseded shopping list, `torch.zeros((max(pending), k))` in a code quotation), and two are honest
statements that something remains unverified (`physical RTX 5090 validation remains pending`, and a
third-party card that "says outright that evaluation is pending"). None is a hidden gap.

## Open items at teardown

**One blocked, with its reason recorded in both the todo note and `docs/29`:** the bare-metal four-arm A/B
(arms C and D). It is now *partly resolved from a different direction* — the kernel-level measurement
(`receipts/kernel-amdahl-bound.json`) supplies the number the ladder could not, and it did so by measuring
the op instead of the server. What remains genuinely unrun is the end-to-end confirmation, which §28 showed
the ladder cannot deliver at achievable repeat counts.

**One left for the owner, deliberately not decided under time pressure:** whether PR #398 is redundant
against `local-inference-lab/vllm#234`. That needs reading someone else's diff and forming a judgement about
their work. Recorded in `receipts/upstream-refresh-final.json`.

**One dropped by owner decision:** re-running `build-pov-ray` to void a 404. 44/89 stands as the permanent
published lower bound.

## The honest summary

The two fixes above are small and both were **cases of our own later work invalidating our own earlier
prose** — a plan that promised a file we then organised differently, and a bracket our own kernel measurement
withdrew. That is the failure mode this project produced most often, and the reason the docs carry so many
explicit corrections rather than quiet edits: **the risk in a body of work this size is not being wrong once,
it is leaving the first version of a claim standing after the second version disproves it.**
