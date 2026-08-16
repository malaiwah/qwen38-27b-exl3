#!/usr/bin/env python3
"""Run the pinned image's OWN batch reorder and its OWN spec_token_indx construction over
named batch compositions, and report whether the vendored GDN gate path miscomputes.

This is a CPU-only corroboration of the source-level argument in
receipts/gdn-gate-concurrency.json. It does not simulate or reimplement anything: it
imports vllm.v1.attention.backends.utils.reorder_batch_to_split_decodes_and_prefills from
the image, feeds it a batch, and then applies the exact three lines of
gdn_attn.py:278-286 to the reordered batch. The only stand-ins are the two objects the
reorder function reads fields off, and it reads only five of them.

Run inside the image, no GPU device needed:

  podman run --rm --network none -v .../gdn_reorder_sim.py:/tmp/sim.py:ro \\
      localhost/vllm:gg-r34-patched-apc python3 /tmp/sim.py
"""

from __future__ import annotations

import json

import numpy as np
import torch

from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_decodes_and_prefills,
)

# The recipe under test: --max-num-seqs 8, mtp depth 3, --max-num-batched-tokens 2048.
NUM_SPEC = 3
THRESHOLD = 1 + NUM_SPEC  # what both attention groups report on this build
CHUNK = 2048


class Req:
    """One request as the reorder function sees it."""

    def __init__(self, name, num_computed, num_prompt, num_scheduled, is_spec):
        self.name = name
        self.num_computed = num_computed
        self.num_prompt = num_prompt
        self.num_scheduled = num_scheduled
        self.is_spec = is_spec


class InputBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)

    @property
    def req_ids(self):
        return [r.name for r in self.reqs]

    @property
    def num_computed_tokens_cpu(self):
        return np.array([r.num_computed for r in self.reqs], dtype=np.int64)

    @property
    def num_prompt_tokens(self):
        return np.array([r.num_prompt for r in self.reqs], dtype=np.int64)

    def swap_states(self, i, j):
        self.reqs[i], self.reqs[j] = self.reqs[j], self.reqs[i]


class SchedulerOutput:
    def __init__(self, batch):
        self._batch = batch

    @property
    def num_scheduled_tokens(self):
        return {r.name: r.num_scheduled for r in self._batch.reqs}


def spec_token_indx(batch):
    """Exactly gdn_attn.py:278-286 applied to the batch as it stands."""
    mask = torch.tensor([r.is_spec for r in batch.reqs], dtype=torch.bool)
    query_lens = torch.tensor([r.num_scheduled for r in batch.reqs], dtype=torch.int64)
    spec_token_masks = torch.repeat_interleave(mask, query_lens)
    index = torch.argsort(spec_token_masks, stable=True)
    num_non_spec_tokens = int(query_lens[~mask].sum())
    return index[num_non_spec_tokens:]


def evaluate(name, reqs, note):
    batch = InputBatch(reqs)
    before = batch.req_ids[:]
    moved = reorder_batch_to_split_decodes_and_prefills(
        batch, SchedulerOutput(batch), decode_threshold=THRESHOLD
    )
    after = batch.req_ids[:]
    mask = [r.is_spec for r in batch.reqs]
    if not any(mask):
        return {
            "case": name,
            "note": note,
            "batch_before": before,
            "batch_after": after,
            "branch": "spec_sequence_masks is None: speculative path not entered",
            "vendored_miscomputes": False,
        }
    # gdn_attn.py:231-251, on the reordered batch
    lens = np.array([r.num_scheduled for r in batch.reqs])
    non_spec_lens = lens[~np.array(mask)]
    num_decodes = int((non_spec_lens == 1).sum())
    num_zero = int((non_spec_lens == 0).sum())
    num_prefills = len(non_spec_lens) - num_decodes - num_zero
    if num_decodes > 0:  # gdn_attn.py:247-251 reclassification
        num_prefills += num_decodes
        num_decodes = 0
    if num_prefills == 0 and num_decodes == 0:
        return {
            "case": name,
            "note": note,
            "batch_before": before,
            "batch_after": after,
            "reorder_moved_requests": bool(moved),
            "branch": "pure-speculative fast path (gdn_attn.py:253): no gather, no-op by construction",
            "vendored_miscomputes": False,
        }
    idx = spec_token_indx(batch)
    identity = bool(torch.equal(idx, torch.arange(idx.numel(), dtype=idx.dtype)))
    return {
        "case": name,
        "note": note,
        "batch_before": before,
        "batch_after": after,
        "reorder_moved_requests": bool(moved),
        "branch": "mixed branch (gdn_attn.py:277): gather happens",
        "spec_tokens": int(idx.numel()),
        "spec_token_indx_head": idx[:6].tolist(),
        "spec_token_indx_is_identity": identity,
        "vendored_miscomputes": not identity,
    }


def spec_decode(name):
    """A steady-state speculative decode: 4 scheduled tokens, generating."""
    return Req(name, num_computed=9000, num_prompt=8000, num_scheduled=4, is_spec=True)


CASES = []

# 1. Our published recipe in steady state, with the co-scheduled chunked prefill placed
#    FIRST in the persistent batch, i.e. the worst possible starting order.
CASES.append(
    (
        "recipe: 8 speculative decodes + 1 chunked prefill, prefill first before reorder",
        [Req("chunkprefill", 2048, 9000, CHUNK - 32, False)]
        + [spec_decode(f"s{i}") for i in range(8)],
        "long_extend (utils.py:701), so the reorder must move it behind the decodes",
    )
)

# 2. Same, with a first-chunk prefill (no context at all) placed first.
CASES.append(
    (
        "recipe: 8 speculative decodes + 1 first-chunk prefill, prefill first before reorder",
        [Req("newprefill", 0, 9000, CHUNK - 32, False)]
        + [spec_decode(f"s{i}") for i in range(8)],
        "pure prefill (utils.py:700), the case a short brand-new prompt also falls into",
    )
)

# 3. A brand-new SHORT prompt, which is the case one would naively expect to break it.
CASES.append(
    (
        "8 speculative decodes + 1 brand-new 4-token prompt",
        [Req("tiny", 0, 4, 4, False)] + [spec_decode(f"s{i}") for i in range(8)],
        "num_computed == 0 so has_context is false: region 3, back of the batch",
    )
)

# 4. The composition that DOES break it: a non-speculative plain decode sharing region 0.
CASES.append(
    (
        "8 speculative decodes + 1 non-speculative plain decode (drafts dropped)",
        [Req("nodraft", 9000, 8000, 1, False)] + [spec_decode(f"s{i}") for i in range(8)],
        "has_context, below threshold, done prefilling: region 0 alongside the speculative decodes, and nothing orders it behind them",
    )
)

# 5. Same as 4 but the non-speculative decode arrives last in the persistent batch.
CASES.append(
    (
        "8 speculative decodes + 1 non-speculative plain decode, decode last",
        [spec_decode(f"s{i}") for i in range(8)] + [Req("nodraft", 9000, 8000, 1, False)],
        "same regions, different slot order: shows the defect depends on slot order inside region 0",
    )
)

# 6. Region 1 collision: a padded pseudo-speculative request (scheduler.py:865-878, a full
#    prefix-cache hit needing one new token, padded to 4 and then classified speculative)
#    together with a non-speculative tiny final chunk.
CASES.append(
    (
        "8 speculative decodes + padded cache-hit pseudo-speculative + non-speculative tiny final chunk",
        [spec_decode(f"s{i}") for i in range(8)]
        + [Req("tinychunk", 2048, 2051, 3, False)]
        + [Req("padded", 210, 211, 4, True)],
        "both land in region 1 (has_context, below threshold, not done prefilling); if the non-speculative one sorts first, speculative tokens are no longer leading",
    )
)

# 7. Same as 6 with the padded request ahead of the tiny chunk.
CASES.append(
    (
        "8 speculative decodes + padded cache-hit pseudo-speculative ahead of the tiny final chunk",
        [spec_decode(f"s{i}") for i in range(8)]
        + [Req("padded", 210, 211, 4, True)]
        + [Req("tinychunk", 2048, 2051, 3, False)],
        "the same two requests in the other slot order",
    )
)

# 8. The ordinary-traffic mechanism, with no prefix caching and no contrivance. A running
#    request that HAS drafts can still be scheduled with exactly one token when the
#    step's token budget is down to 1 (scheduler.py:518, num_new_tokens = min(..., token
#    budget)). Then scheduler.py:637-642 computes num_scheduled_spec_tokens == 0, does
#    not enter it into scheduled_spec_decode_tokens, and gpu_model_runner.py:2248 marks
#    it non-speculative. The budget is consumed in self.running order (scheduler.py:480),
#    which is NOT input_batch slot order, and the reorder only permutes slots, so the
#    clamped request can hold a lower slot than fully scheduled speculative decodes.
CASES.append(
    (
        "budget-clamped decode: 4 speculative decodes, a clamped non-speculative decode at a middle slot, 3 more speculative decodes, plus the prefill that ate the budget",
        [spec_decode(f"s{i}") for i in range(4)]
        + [Req("clamped", 9000, 8000, 1, False)]
        + [spec_decode(f"s{i}") for i in range(4, 7)]
        + [Req("chunkprefill", 4096, 9000, 2047 - 28, False)],
        "the clamped request is region 0 like the speculative decodes, so the reorder leaves it where it is, in the middle of the leading tokens",
    )
)

# 9. Same mechanism, but the clamped request holds the highest slot among the region-0
#    requests, which is what happens whenever slot order happens to agree with
#    self.running order.
CASES.append(
    (
        "budget-clamped decode holding the highest region-0 slot",
        [spec_decode(f"s{i}") for i in range(7)]
        + [Req("clamped", 9000, 8000, 1, False)]
        + [Req("chunkprefill", 4096, 9000, 2047 - 28, False)],
        "identical regions, but the clamped request sorts last inside region 0, so the speculative tokens stay leading",
    )
)


def main() -> int:
    results = [evaluate(name, reqs, note) for name, reqs, note in CASES]
    print(json.dumps({"threshold_used": THRESHOLD, "cases": results}, indent=2))
    broken = [r["case"] for r in results if r["vendored_miscomputes"]]
    print(
        json.dumps(
            {
                "cases_evaluated": len(results),
                "cases_where_vendored_miscomputes": broken,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
