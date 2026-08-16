# Gate-zero reproduction for vllm-project/vllm#52520 on UNMODIFIED upstream main.
# CPU only, no model weights, no GPU. Run from a vLLM source checkout:
#
#   python repro-52520-stock-main.py
#
# Hybrid full-attention + Mamba("align") model, MTP depth 3, chunked prefill.
# The pool is sized at exactly the number of blocks `get_kv_cache_configs`
# accepts for max_model_len; `BlockPool` then reserves one of them as the null
# block, so the pool is one block short of one max_model_len request. Admission
# does not notice, and the request is prefilled to 98.5 % and thrown away.
import sys

import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
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
        speculative_config=SpeculativeConfig(
            model="ngram",
            method="ngram",
            num_speculative_tokens=NUM_SPEC,
            prompt_lookup_max=NUM_SPEC,
            prompt_lookup_min=1,
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return vllm_config


def build(num_blocks):
    vllm_config = make_vllm_config(num_blocks)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["fa"], fa_spec()),
            KVCacheGroupSpec(["mamba"], mamba_spec()),
        ],
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
    """Does the startup check accept a pool of exactly `num_blocks` blocks?"""
    vllm_config = make_vllm_config(num_blocks)
    page = fa_spec().page_size_bytes
    try:
        configs = get_kv_cache_configs(
            vllm_config, [{"fa": fa_spec(), "mamba": mamba_spec()}], [num_blocks * page]
        )
    except ValueError as e:
        return False, str(e).split(".")[0]
    return True, f"num_blocks={configs[0].num_blocks}"


def trace(num_blocks, prompt_len, steps=24):
    scheduler = build(num_blocks)
    pool = scheduler.kv_cache_manager.block_pool
    [req] = create_requests(
        num_requests=1, num_tokens=prompt_len, max_tokens=1,
        block_size=BLOCK, req_ids=["victim"],
    )
    scheduler.add_request(req)
    print(f"  pool={num_blocks} blocks, free at start={pool.get_num_free_blocks()} "
          f"(one block is BlockPool's null block)")
    for step in range(steps):
        out = scheduler.schedule()
        n = out.num_scheduled_tokens.get("victim", 0)
        print(f"    step {step:3d} scheduled={n:5d} computed={req.num_computed_tokens:5d} "
              f"free={pool.get_num_free_blocks():3d} status={req.status} "
              f"preemptions={req.num_preemptions}")
        ids = list(out.num_scheduled_tokens)
        scheduler.update_from_output(out, ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[] for _ in ids],
            logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
        ))
    return req


if __name__ == "__main__":
    import vllm
    print(f"vLLM {vllm.__version__}")
    need = cdiv(MAX_MODEL_LEN, BLOCK) + 2 + NUM_SPEC
    print(f"max_model_len={MAX_MODEL_LEN} block_size={BLOCK} "
          f"num_speculative_blocks={NUM_SPEC} max_num_batched_tokens={CHUNK}")
    print(f"startup bound: full-attention cdiv({MAX_MODEL_LEN},{BLOCK})="
          f"{cdiv(MAX_MODEL_LEN, BLOCK)} + mamba-align (2+{NUM_SPEC})="
          f"{2 + NUM_SPEC}  ->  {need} blocks")
    for nb in (need, need + 1):
        ok, detail = startup_accepts(nb)
        print(f"\npool of {nb} blocks: startup check accepts={ok} ({detail})")
        req = trace(nb, MAX_MODEL_LEN - 1)
        print(f"  VERDICT prompt={MAX_MODEL_LEN - 1}: finished={req.is_finished()} "
              f"computed={req.num_computed_tokens} preemptions={req.num_preemptions} "
              f"status={req.status}")
