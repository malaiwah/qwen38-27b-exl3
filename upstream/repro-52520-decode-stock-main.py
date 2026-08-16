# Decode-case probe for the scoping question on vllm-project/vllm#52530.
# CPU only, no model weights, no GPU. Run from a vLLM source checkout:
#
#   EXTRA_PYTHONPATH=<checkout> py3 repro-52520-decode-stock-main.py
#
# Same harness and same config as the gate-zero prefill reproduction
# (repro-52520-stock-main.py): hybrid FullAttention + Mamba("align"), MTP
# depth 3, chunked prefill, block_size 16, max_model_len 1024, pool sized at
# exactly the number of blocks `get_kv_cache_configs` accepts.
#
# The prefill case put the whole sequence in the prompt. This one splits it:
# a prompt that fits under the servable ceiling plus a `max_tokens` that walks
# past it during decode, which is the case the PR's `_is_unservable` lets
# through (`num_tokens + 1`, not `num_tokens + max_tokens`).
import os
import sys

# The sweep at the bottom prices configs whose `max_model_len` exceeds
# opt-125m's 2048 position embeddings. Nothing here runs a model; only the
# block arithmetic is exercised.
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    _estimate_max_model_len_from_groups,
    _pool_bytes_per_block,
    get_kv_cache_configs,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager

sys.path.insert(0, "tests")
from tests.v1.core.utils import create_requests  # noqa: E402

BLOCK = 16
MAX_MODEL_LEN = 1024
NUM_SPEC = 3
CHUNK = 256
FILLER_TOKEN = 7  # not EOS_TOKEN_ID (== 50256 in tests/v1/core/utils.py)


def fa_spec():
    return FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def mamba_spec(num_spec=NUM_SPEC):
    return MambaSpec(
        block_size=BLOCK,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
        page_size_padded=fa_spec().page_size_bytes,
    )


def groups():
    return [
        KVCacheGroupSpec(["fa"], fa_spec()),
        KVCacheGroupSpec(["mamba"], mamba_spec()),
    ]


def make_vllm_config(num_blocks):
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=CHUNK,
            max_model_len=MAX_MODEL_LEN,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        speculative_config=(
            SpeculativeConfig(
                model="ngram",
                method="ngram",
                num_speculative_tokens=NUM_SPEC,
                prompt_lookup_max=NUM_SPEC,
                prompt_lookup_min=1,
            )
            if NUM_SPEC
            else None
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return vllm_config


def build(num_blocks):
    vllm_config = make_vllm_config(num_blocks)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups()
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK,
        hash_block_size=BLOCK,
        log_stats=True,
    )


def startup_accepts(num_blocks):
    vllm_config = make_vllm_config(num_blocks)
    page = fa_spec().page_size_bytes
    try:
        configs = get_kv_cache_configs(
            vllm_config, [{"fa": fa_spec(), "mamba": mamba_spec()}], [num_blocks * page]
        )
    except ValueError as e:
        return False, str(e).split(".")[0]
    return True, f"num_blocks={configs[0].num_blocks}"


def servable(num_blocks):
    """`max_servable_num_tokens` computed from stock-main helpers only.

    Identical arithmetic to the helper this PR adds: the same
    `_estimate_max_model_len_from_groups` the startup check reports as "the
    estimated maximum model length", evaluated against `num_blocks - 1` blocks
    because `BlockPool` keeps one as the null block.
    """
    vllm_config = make_vllm_config(num_blocks)
    g = groups()
    usable = max(num_blocks - 1, 0)
    return _estimate_max_model_len_from_groups(
        vllm_config, g, usable * _pool_bytes_per_block(vllm_config, g)
    )


def num_preempted_events(req):
    """What `IterationStats` feeds into `vllm:num_preemptions_total`.

    `metrics/stats.py` increments `num_preempted_reqs` once per
    `EngineCoreEventType.PREEMPTED` event on the request.
    """
    return sum(1 for e in req.events if e.type == EngineCoreEventType.PREEMPTED)


def run(num_blocks, prompt_len, max_tokens, steps, label, inject_spec=0):
    scheduler = build(num_blocks)
    pool = scheduler.kv_cache_manager.block_pool
    [req] = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=max_tokens,
        ignore_eos=True,
        block_size=BLOCK,
        req_ids=["victim"],
    )
    admitted_before = len(scheduler.waiting) + len(scheduler.running)
    scheduler.add_request(req)
    admitted = (len(scheduler.waiting) + len(scheduler.running)) > admitted_before
    print(f"  [{label}] pool={num_blocks} blocks free_at_start="
          f"{pool.get_num_free_blocks()} admitted_to_a_queue={admitted}")

    history = []
    first_wall_step = None
    prev_out = 0
    stalled_since = None
    for step in range(steps):
        if inject_spec and req.status.name == "RUNNING" and not req.spec_token_ids:
            # Stand in for the draft proposer: the scheduler reads
            # `request.spec_token_ids` at sched/scheduler.py:737 and asks the
            # allocator for `1 + len(spec_token_ids)` slots.
            req.spec_token_ids = [FILLER_TOKEN] * inject_spec
        out = scheduler.schedule()
        n = out.num_scheduled_tokens.get("victim", 0)
        # The model samples exactly when the whole sequence has been computed.
        ids = list(out.num_scheduled_tokens)
        sampled = []
        for rid in ids:
            r = scheduler.requests[rid]
            sampled.append(
                [FILLER_TOKEN] if r.num_computed_tokens >= r.num_tokens else []
            )
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        n_out = len(req.output_token_ids)
        row = (step, n, req.num_computed_tokens, req.num_tokens, n_out,
               pool.get_num_free_blocks(), str(req.status),
               req.num_preemptions, num_preempted_events(req))
        history.append(row)
        if req.num_preemptions and first_wall_step is None:
            first_wall_step = step
        if n_out == prev_out:
            stalled_since = step if stalled_since is None else stalled_since
        else:
            stalled_since = None
            prev_out = n_out
        if req.is_finished():
            break

    def show(rows):
        for (s, n, c, t, o, f, st, p, pe) in rows:
            print(f"    step {s:4d} sched={n:5d} computed={c:5d} seq_len={t:5d} "
                  f"out={o:3d} free={f:3d} {st:20s} preempt={p:4d} ev={pe:4d}")

    head = 24
    show(history[:head])
    if len(history) > head + 10:
        print(f"    ... {len(history) - head - 10} steps elided ...")
        show(history[-10:])
    elif len(history) > head:
        show(history[head:])

    print(f"  [{label}] VERDICT after {len(history)} steps: "
          f"finished={req.is_finished()} status={req.status} "
          f"output_tokens={len(req.output_token_ids)}/{max_tokens} "
          f"num_preemptions={req.num_preemptions} "
          f"preempted_events={num_preempted_events(req)} "
          f"first_wall_step={first_wall_step} "
          f"output_stalled_since_step={stalled_since}")
    return {
        "label": label,
        "num_blocks": num_blocks,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "steps_run": len(history),
        "finished": bool(req.is_finished()),
        "status": str(req.status),
        "output_tokens": len(req.output_token_ids),
        "num_preemptions": req.num_preemptions,
        "preempted_events": num_preempted_events(req),
        "first_wall_step": first_wall_step,
        "output_stalled_since_step": stalled_since,
        "final_seq_len": req.num_tokens,
        "history_head": history[:head],
        "history_tail": history[-10:],
    }


def gap_sweep():
    """How far below `max_model_len` the servable ceiling can sit.

    For each config, the smallest pool the startup check accepts, and the
    servable ceiling of that pool.
    """
    rows = []
    for block, mml, nspec in (
        (16, 1024, 3), (16, 1024, 0), (16, 1000, 3), (16, 1009, 3),
        (16, 4096, 3), (32, 1024, 3), (64, 1024, 3), (128, 1024, 3),
        (128, 8192, 3), (128, 8192, 16), (16, 8192, 16), (256, 8192, 8),
        (256, 8192, 0), (512, 32768, 8),
    ):
        global BLOCK, MAX_MODEL_LEN, NUM_SPEC
        BLOCK, MAX_MODEL_LEN, NUM_SPEC = block, mml, nspec
        need = cdiv(mml, block) + 2 + nspec
        lo = max(1, need - 4)
        accepted = [nb for nb in range(lo, need + 9)
                    if startup_accepts(nb)[0]]
        if not accepted:
            print(f"  [skip] block={block} mml={mml} nspec={nspec}: "
                  f"no pool in [{lo},{need + 8}] accepted "
                  f"({startup_accepts(need)[1]})")
            continue
        smallest = min(accepted)
        s = servable(smallest)
        # One block more than the smallest accepted pool: is the gap gone?
        s_next = servable(smallest + 1)
        rows.append((block, mml, nspec, need, smallest, s, mml - s,
                     s_next, mml - s_next))
    BLOCK, MAX_MODEL_LEN, NUM_SPEC = 16, 1024, 3
    print("\n=== gap sweep: smallest startup-accepted pool vs its servable ceiling ===")
    print("  block max_model_len nspec  need smallest_ok servable  gap_tok gap_blk "
          "| +1blk_servable +1blk_gap")
    for (b, m, ns, need, sm, s, gap, s1, gap1) in rows:
        print(f"  {b:5d} {m:13d} {ns:5d} {need:5d} {sm:11d} {s:8d} {gap:8d} "
              f"{gap / b:7.2f} | {s1:14d} {gap1:9d}")
    return rows


if __name__ == "__main__":
    import json

    import vllm

    print(f"vLLM {vllm.__version__}")
    need = cdiv(MAX_MODEL_LEN, BLOCK) + 2 + NUM_SPEC
    ok, detail = startup_accepts(need)
    ceiling = servable(need)
    print(f"config: block_size={BLOCK} max_model_len={MAX_MODEL_LEN} "
          f"num_speculative_blocks={NUM_SPEC} max_num_batched_tokens={CHUNK}")
    print(f"startup bound: cdiv({MAX_MODEL_LEN},{BLOCK})={cdiv(MAX_MODEL_LEN, BLOCK)} "
          f"+ mamba-align (2+{NUM_SPEC})={2 + NUM_SPEC} -> {need} blocks; "
          f"startup accepts={ok} ({detail})")
    print(f"max_servable_num_tokens({need} blocks) = {ceiling} "
          f"(gap to max_model_len = {MAX_MODEL_LEN - ceiling} tokens)")

    PROMPT = 1000
    MAX_TOKENS = MAX_MODEL_LEN - PROMPT  # 24: exactly the frontend budget
    print(f"\nrequest: prompt={PROMPT} max_tokens={MAX_TOKENS}")
    print(f"  frontend  : {PROMPT} + {MAX_TOKENS} = {PROMPT + MAX_TOKENS} "
          f"<= {MAX_MODEL_LEN} -> accepted")
    print(f"  _is_unservable: {PROMPT} + 1 = {PROMPT + 1} <= {ceiling} -> admitted")
    print(f"  wall          : reaches {PROMPT + MAX_TOKENS} > {ceiling} during decode")

    STEPS = 4000
    results = [
        run(need, PROMPT, MAX_TOKENS, STEPS, "decode-over-ceiling"),
        # Control A: identical request, one more block, so the pool covers
        # max_model_len. Proves the harness can finish a request.
        run(need + 1, PROMPT, MAX_TOKENS, STEPS, "control-pool+1"),
        # Control B: same undersized pool, a max_tokens that stays under the
        # ceiling. Proves the undersized pool is not fatal by itself.
        run(need, PROMPT, ceiling - PROMPT, STEPS, "control-under-ceiling"),
        # Variant: same undersized pool, a max_tokens that stays under the
        # ceiling, but a live draft proposer. Does the spec lookahead push the
        # allocator past the ceiling before the length cap can bite?
        run(need, PROMPT, ceiling - PROMPT, STEPS, "spec-lookahead-under-ceiling",
            inject_spec=NUM_SPEC),
    ]
    rows = gap_sweep()
    print("\n=== JSON ===")
    print(json.dumps({
        "vllm_version": vllm.__version__,
        "block_size": BLOCK,
        "max_model_len": MAX_MODEL_LEN,
        "num_speculative_blocks": NUM_SPEC,
        "max_num_batched_tokens": CHUNK,
        "pool_blocks": need,
        "max_servable_num_tokens": ceiling,
        "steps_budget": STEPS,
        "runs": results,
        "gap_sweep": [
            {"block_size": b, "max_model_len": m, "num_speculative_blocks": ns,
             "startup_min_blocks": need_, "smallest_accepted_blocks": sm,
             "max_servable_num_tokens": s, "gap_tokens": g,
             "gap_blocks": g / b,
             "max_servable_num_tokens_one_block_more": s1,
             "gap_tokens_one_block_more": g1}
            for (b, m, ns, need_, sm, s, g, s1, g1) in rows
        ],
    }, indent=1))
