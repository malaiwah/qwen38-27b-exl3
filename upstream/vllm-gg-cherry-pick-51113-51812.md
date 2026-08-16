# Cherry-pick request: upstream vLLM #51113 and #51812 into `dev/gilded-gnosis`

**Filed 2026-08-16, amended the same day.** Issue
<https://github.com/local-inference-lab/vllm/issues/392> · PR
<https://github.com/local-inference-lab/vllm/pull/393>. The issue body below is verbatim the
current live body, re-fetched after the amendment; `upstream/filed.json` has the machine-readable
record.

**What the amendment changed.** The physical-RTX-5090 reproduction
(`receipts/apc-poison-repro.json`) came back **NOT REPRODUCED**. The user report is therefore no
longer offered as evidence that #51113 bites in practice, and the suggestion to gate
`--enable-prefix-caching` is withdrawn. The ask itself stands unchanged, on the source defect and
on upstream's own regression file failing 14 of 20 against the branch. The retraction is also
posted as a visible comment rather than only as a body edit.

---

## Issue body

**Title:** `[Bugfix] Cherry-pick upstream #51113 (mamba align prefill split) and #51812 (Qwen GDN spec gates) into dev/gilded-gnosis`

Two upstream bugfixes that this branch predates are both reachable from a shipped Qwen3.5 /
Qwen3.8 configuration. Ordered by severity. The first returns **wrong tokens with HTTP 200 and
no crash**; the second is a smaller logit-drift fix. PR with both cherry-picks is linked at the
bottom, one commit each so either can be taken alone.

### 1. #51113 — silent wrong output (severity: correctness, fails open)

Upstream [vllm-project/vllm#51113](https://github.com/vllm-project/vllm/pull/51113),
*"[Bugfix] Keep mamba align prefill chunks block-aligned past `last_cache_position`"*,
merged 2026-08-06T17:21:00Z, merge commit `c56f169d9ae46ca420617e2cf5f0c9135da0f651`,
milestone *v0.27.0 cherry picks*. Fixes upstream issue #43559.

**Still absent on `dev/gilded-gnosis`.** `vllm/v1/core/sched/scheduler.py` at branch head
`fa033bd4e1b16d9d729ad94be2d87da5a13210ce` is sha256
`1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce`, byte-identical to the copy
vendored in `voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`
(tag `…-cu132-20260810-r34`, built 2026-08-10, four days after the fix merged). In
`Scheduler._mamba_block_aligned_split`:

```python
400:        if end < last_cache_position:
401:            end = end // block_size * block_size
...
413:            next_block_boundary
414:            if start % block_size != 0 and next_block_boundary <= last_cache_position
415:            else 0,
```

`prefill_end` occurs **0 times** in the file. Upstream re-gates both rules on the true prefill
end instead of the cacheable-range bound.

**Why it corrupts output.** In `align` mode mamba slot *p* is defined to hold the state after
exactly `(p + 1) * block_size` tokens, and state is materialized only at prefill chunk ends.
`last_cache_position = num_tokens - num_tokens % block_size`, backed off one further block under
EAGLE/MTP and clamped at 0 — so for a prompt under two mamba blocks the guard never fires at
all. Meanwhile `KVCacheCoordinator` still publishes a hash one block *past* the aligned
boundary. A chunk ending mid-block in that window gets its slot hashed as a full-block state; a
later request hitting that hash restores a **truncated** state and generates from it. Exposed
window: `(P mod B) + B` tokens for prompt length `P`, mamba block size `B`.

**Reachability on Qwen3.5 + MTP + prefix caching** (paths inside the r34 image, all present on
this branch):

1. `model_executor/models/config.py:558-562` — with prefix caching on and no
   `supports_mamba_prefix_caching`, `mamba_cache_mode` is forced to `"align"`. Qwen3.5 never
   declares that attribute.
2. `config/speculative.py:1551-1555` — `use_eagle()` is true for method `mtp`, so
   `last_cache_position` takes the extra one-block back-off.
3. `v1/core/sched/scheduler.py:316-318` — `need_mamba_block_aligned_split` is enabled.
4. `v1/core/kv_cache_coordinator.py:800-806` — the eagle group caches one block past where the
   alignment guarantee stops, publishing exactly the unaligned window.

This branch also lacks the later partial mitigation: `MambaManager.find_longest_cache_hit`
accepts `drop_eagle_block` at `single_type_kv_cache_manager.py:1297` and never uses it, and
`kv_cache_coordinator.py:892` applies the eagle margin only when the spec is not a `MambaSpec`.

**Our test result (CPU only, no GPU, no model loaded).** Upstream's own regression file from the
fix commit, `tests/v1/core/test_mamba_align_chunk_split.py` (sha256
`6b57360273223dbd208c7712b440a3fa267a61a039a8ec47c4fa476cf23e0b81`), run **unmodified** against
the real `KVCacheManager` / `HybridKVCacheCoordinator` / `MambaManager` from this branch:

| scheduler module under test | result |
| --- | --- |
| branch head `fa033bd4e` (= vendored r34) | **14 failed, 6 passed** |
| same file with #51113 cherry-picked | **20 passed** |

Representative failures:

```
mamba slot 0 is hashed as state@1600 but holds state@900
chunk [2531, 3602) starts mid-block and runs past 3200; the slot holding state@2531
    gets hashed as state@3200
```

Full receipt: `receipts/mamba-align-defect.json`. Patched module
`tools/vllm-mamba-align-scheduler.py`, sha256
`b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf`; two-hunk diff
`patches/vllm-51113-mamba-align.patch`.

### 2. #51812 — logit drift under speculative decoding (severity: fidelity)

Upstream [vllm-project/vllm#51812](https://github.com/vllm-project/vllm/pull/51812),
*"[Bugfix] Align Qwen GDN gates with speculative tokens"*, merged 2026-08-11T15:35:30Z, merge
commit `5af7c8dad798bf899813f8f3c6b9eaf08a748e17`.

**Still absent on `dev/gilded-gnosis`.**
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` at branch head is sha256
`663dacd324b6b8224a4cb312b3e9c0bad4322c515e982a85f13c3450ffdb7d61` — again byte-identical to the
r34 vendored copy. Line 1298 gathers the QKV with `spec_token_indx`:

```python
1298:                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
```

while lines 1420-1427 pass the gates through **unsorted**:

```python
1420:            core_attn_out_spec, last_recurrent_state = (
1421:                fused_sigmoid_gating_delta_rule_update(
1422:                    A_log=self.A_log,
1423:                    a=a,
1424:                    b=b,
```

`a_spec` and `b_spec` occur **0 times** in the file. The kernel's first `T_spec` gate rows can
therefore belong to different tokens than the gathered Q/K/V rows; it bites only when a mixed
batch places non-speculative tokens before speculative ones.

Upstream's own measurement of the fix (Qwen3.5-2B BF16, TP1, eager): mean absolute chosen-logprob
error **0.002755 → 0.000208**, max **0.020539 → 0.001690**, greedy token ids unchanged in their
repro. This is sub-argmax drift — it degrades MTP acceptance but is not on its own a plausible
source of garbled text.

Receipt: `receipts/gdn-spec-gate-defect.json`. Patched module
`tools/vllm-qwen-gdn-spec-gates.py`, sha256
`7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0`, diff-identical to the
upstream two hunks, 8 changed lines, `py_compile` clean under the image's own Python 3.12.3.

### Why this matters downstream

A user running the r34 image reported it here:
<https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context/discussions/1>. With
`--enable-prefix-caching --mamba-cache-mode align` and MTP depth 3 they get intermittent
multilingual token soup after 1-2 turns — HTTP 200, no crash — and MTP acceptance walking
39.4 % → 0.0 % → 64.5 % → 20.4 % → 0.0 %. That symptom set matches #51113, not #51812.

Prefix caching is default-**off** for hybrid models
(`engine/arg_utils.py:2532-2534`: `default_prefix_caching = is_prefix_caching_supported and not
is_hybrid`), so the defect is latent for everyone who does not opt in.

**Amended 2026-08-16 — GPU reproduction: NOT reproduced.** Read this as a blast-radius bound, not
as an absence. 266 scored requests across seven freshly started servers on a physical RTX 5090, in
the reporter's exact condition (`--enable-prefix-caching --mamba-cache-mode align`, MTP depth 3,
`--max-model-len 196608`, `--gpu-memory-utilization 0.92`, `--max-num-seqs 1`) against the
**unpatched** release image, over deliberately mid-block shared prefixes: zero corrupted
responses, zero wrong planted answers, zero acceptance collapses (68 SpecDecoding windows, min
56.0 %, max 94.7 %). Receipt `receipts/apc-poison-repro.json`, content sha256
`e0d628144e8d5fb995f30aaa8b681a2e6b35f0d444a12d36d18d87a436eaa2f6`, pre-registered before the
first scored arm ran.

This does not weaken the cherry-pick case, and I am not withdrawing anything above. #51113 is
still absent from this branch, upstream's own regression file still fails 14 of 20 against the
branch's scheduler, and the patched arm was clean while prefix reuse delivered **11.6x and 29.3x
TTFT** speedups (12.07 s -> 1.04 s and 67.60 s -> 2.31 s) — i.e. the fix unlocks a large
performance lever safely. What the GPU run establishes is that the defect did not perturb that
one workload measurably: text-only, single-sequence, nested-document prefix, one card. It does
not show the defect is unreachable, and the source-level defect and the failing regression file
stand on their own.

Two corrections I owe you about the user report, in the interest of not overselling it:

- **It is not established that #51113 caused it.** Not shown to be, not shown not to be. The
  reporter's own "it's fixed" control removed LMCache, prefix caching and align mode all at once,
  so it is confounded. Our run decomposes it: prefix caching + align *without* LMCache stayed
  clean. That makes **LMCache 0.5.2 in `kv_both` disk mode** — a second, independent state-restore
  path layered on vLLM's prefix cache — the leading suspect rather than #51113. It is untested and
  is the named next experiment.
- At `--max-num-seqs 1` there is no mixed batch, which upstream says is the only condition where
  #51812 acts, so that arm did not exercise the GDN fix either way.

### Ask

Cherry-pick both upstream commits onto `dev/gilded-gnosis`, #51113 first. The case rests on the
source defect and on upstream's own regression file failing 14 of 20 against this branch — not on
the user report, which our GPU probe did not reproduce and which currently points more at LMCache
than at #51113.

I am **not** asking you to gate `--enable-prefix-caching`: measured on the patched image it is a
large win (11.6x / 29.3x TTFT) and it was clean on the unpatched image over the one workload we
probed. The honest statement is that its safety on the unpatched branch is unbounded rather than
disproven, and that #51113 is what makes it sound.

### How to verify

```bash
git clone --filter=blob:none -b dev/gilded-gnosis https://github.com/local-inference-lab/vllm
cd vllm

# 1. both fixes absent at branch head
grep -c prefill_end vllm/v1/core/sched/scheduler.py                                   # 0
grep -c a_spec vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py           # 0

# 2. reproduce the split defect with upstream's own CPU-only test (no GPU, no weights)
git remote add upstream https://github.com/vllm-project/vllm
git fetch upstream c56f169d9ae46ca420617e2cf5f0c9135da0f651
git checkout c56f169d9ae46ca420617e2cf5f0c9135da0f651 -- tests/v1/core/test_mamba_align_chunk_split.py
pytest tests/v1/core/test_mamba_align_chunk_split.py -q      # 14 failed, 6 passed

# 3. apply the fixes and re-run
git cherry-pick -x c56f169d9ae46ca420617e2cf5f0c9135da0f651  # conflicts; see note below
git fetch upstream 5af7c8dad798bf899813f8f3c6b9eaf08a748e17
git cherry-pick -x 5af7c8dad798bf899813f8f3c6b9eaf08a748e17  # clean
pytest tests/v1/core/test_mamba_align_chunk_split.py -q      # 20 passed
```

**Conflict note for #51113.** One hunk conflicts. This branch diverged from upstream in that
same block: it lacks upstream's `max_prefill_tokens` / `long_prefill_token_threshold` relaxation
inside the guarded branch and carries different comments. Resolution in the linked PR keeps this
branch's stricter body and applies only the behavioural change
(`last_cache_position` → `prefill_end`). #51812 cherry-picks clean.

Resulting file digests, both reproducible from the commands above:

| file | before | after |
| --- | --- | --- |
| `vllm/v1/core/sched/scheduler.py` | `1ea341f4cc28d282…` | `b431c1066dfee3ed…` |
| `…/gdn/qwen_gdn_linear_attn.py` | `663dacd324b6b822…` | `7cd3f5fe763b6210…` |

---

## Filing record

| item | value |
| --- | --- |
| issue | <https://github.com/local-inference-lab/vllm/issues/392> |
| PR | <https://github.com/local-inference-lab/vllm/pull/393> (2 commits, 3 files, +264 / -13) |
| PR head | `malaiwah/vllm-voipmonitor:fix/cherry-pick-51113-51812` |
| PR base | `local-inference-lab/vllm:dev/gilded-gnosis` @ `fa033bd4e1b16d9d729ad94be2d87da5a13210ce` |
| commit 1 | `c5e46a15e5993d536e72c51f0f96e5599a1447f3` — cherry-pick of `c56f169d9` (#51113), one hunk conflicted, resolved as described above |
| commit 2 | `50248cdb926137e69bc7858393335e0c47c2aeeb` — cherry-pick of `5af7c8dad` (#51812), clean |

**Access used.** `gh` CLI is not authenticated and neither `GITHUB_TOKEN` nor `GH_TOKEN` is set.
Filing went through the `credential.helper=store` entry for github.com in `~/.git-credentials`
(user `malaiwah`, classic PAT with `repo` scope) — the same credential the `github` remote in
this repository pushes with. We have **pull-only** permission on `local-inference-lab/vllm`, so
the PR is cross-fork from `malaiwah/vllm-voipmonitor`, exactly as #314 / #316 / #318 / #319 were.

**Verified by the filer, not taken from summary.** Both target files at branch head
`fa033bd4e` are byte-identical to the copies vendored in the r34 image (`diff` returned 0 lines
for `scheduler.py`; `qwen_gdn_linear_attn.py` sha256 matches to the digit), so neither fix has
landed upstream in the fork and the ask stands unchanged. Upstream's own regression file, run
unmodified on CPU with `CUDA_VISIBLE_DEVICES` empty, gives **14 failed / 6 passed** against the
branch head and **20 passed** against the PR tree; both changed files `py_compile` clean under
Python 3.12.3. The cherry-picked results are `cmp`-equal to the already-published modules
`tools/vllm-mamba-align-scheduler.py` and `tools/vllm-qwen-gdn-spec-gates.py`. No GPU was used.

### Amendment 2026-08-16

| item | value |
| --- | --- |
| trigger | `receipts/apc-poison-repro.json` landed, content sha256 `e0d628144e8d5fb995f30aaa8b681a2e6b35f0d444a12d36d18d87a436eaa2f6` |
| verdict | reproduction **NOT REPRODUCED**; fix **not exercised** |
| evidence | 266 scored requests, 7 fresh servers, physical RTX 5090, reporter's exact condition on the *unpatched* image: 0 corrupted, 0 wrong planted answers, 0 acceptance collapses (68 windows, 56.0–94.7 %) |
| retracted | the user report as proof that #51113 bites; the suggestion to gate `--enable-prefix-caching` |
| retained | #51113 and #51812 both still absent at `fa033bd4e`; regression file still 14/20 fail vs branch, 20/20 pass vs PR tree; patched arm not worse on any axis and unlocked 11.6x / 29.3x TTFT |
| new leading suspect for the user report | LMCache 0.5.2 `kv_both` disk reuse — the reporter's own control removed LMCache, prefix caching and align mode at once, so it is confounded; our run held prefix caching + align *without* LMCache and stayed clean. Untested, named next experiment. |
| caveat | `--max-num-seqs 1` means no mixed batch, the only condition upstream says #51812 acts under, so that arm did not exercise the GDN fix |
| issue comment | <https://github.com/local-inference-lab/vllm/issues/392#issuecomment-5305736361> |
| PR comment | <https://github.com/local-inference-lab/vllm/pull/393#issuecomment-5305736420> |
