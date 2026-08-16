# POSTED comment on local-inference-lab/vllm#393 — cherry-pick PR: CI is gated, and one review nit answered

<!--
Target: https://github.com/local-inference-lab/vllm/pull/393 (our own open PR)
Status: POSTED 2026-08-16 -> https://github.com/local-inference-lab/vllm/pull/393#issuecomment-5307331684 (approved by Main).
The docstring half is purely factual and receipt-backed; it is bundled here rather than posted
alone because a bot asked it, not a maintainer, and one comment is politer than two.
Also applies to #312, #314, #316, #318, #319 — all six are `mergeable_state: unstable` for the same
reason. Consider one comment on the oldest, or a note to a maintainer, rather than six.
-->

<!-- ===== POSTABLE BODY BEGINS BELOW THIS LINE; post everything after it verbatim, and nothing above it ===== -->
Two housekeeping items, one of which needs a maintainer.

**1. `pre-commit` has never run on this PR, and cannot until someone adds a label.**
`pre-run-check` fails at its first step, "Check PR label and author merge count"
(run `31922350144`, job `95104176894`, 3 s). Reading
`.github/workflows/pre-commit.yml` on `dev/gilded-gnosis`, that step requires one of the `verified`,
`ready` or `ready-run-all-tests` labels, **or** an author with ≥4 merged PRs in this repository. I
have none merged here yet, and I have pull-only permissions, so I cannot label it myself. The
`pre-commit` job is `needs: pre-run-check`, so it is skipped rather than failing — meaning the red
check on this PR is a policy gate, not a lint or test failure, and no lint signal exists yet either
way. If a maintainer adds `ready` (or `verified`), CI will run and I will fix whatever it finds. The
same gate is currently red on #312, #314, #316, #318 and #319 for the same reason.

**2. On the CodeRabbit docstring finding in `tests/v1/core/test_mamba_align_chunk_split.py`.**
I would rather not act on it, and the reason is checkable: that file is **byte-identical to
upstream's**. It is `tests/v1/core/test_mamba_align_chunk_split.py` at vLLM merge commit
`c56f169d9ae46ca420617e2cf5f0c9135da0f651`,
`sha256 6b57360273223dbd208c7712b440a3fa267a61a039a8ec47c4fa476cf23e0b81`, and the copy on this PR
head hashes to the same value. Adding `Args:`/`Returns:` sections to its helpers would create a diff
against upstream in a file whose only job is to be upstream's regression test, which makes the next
rebase noisier and weakens the claim the PR rests on ("upstream's own test, unmodified, fails 14/20
against the branch and passes 20/20 against this tree"). If the repository wants Google-style
docstrings enforced on vendored upstream tests, I will do it — but I would want that to be a
deliberate policy decision rather than a side effect of this cherry-pick.

For completeness, all three files on this head are byte-identical to the audited artefacts:

| file | sha256 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | `b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0` |
| `tests/v1/core/test_mamba_align_chunk_split.py` | `6b57360273223dbd208c7712b440a3fa267a61a039a8ec47c4fa476cf23e0b81` (= upstream at `c56f169d`) |
