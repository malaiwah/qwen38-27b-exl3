#!/usr/bin/env python3
"""GPU-free probe: which attention backend, if any, accepts each KV dtype.

Calls the pinned build's own `CudaPlatform.get_valid_backends`, the same
function whose failure produces the "No valid attention backend found" error,
so a refusal can be attributed to the KV dtype, to head_size, or to the MLA
path rather than guessed at. No CUDA context is created: the device capability
is supplied explicitly as sm120.
"""
import json

import torch
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.selector import AttentionSelectorConfig

CAP = DeviceCapability(12, 0)
DTYPES = [
    "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc",
    "fp8_ds_mla", "nvfp4_ds_mla", "turboquant_k8v4", "turboquant_4bit_nc",
    "turboquant_k3v4_nc", "turboquant_3bit_nc", "int4_per_token_head",
    "int8_per_token_head", "fp8_per_token_head", "nvfp4",
]
# The engine passes use_per_head_quant_scales from the attention layer, not from
# the KV dtype: the verbatim startup errors show `use_per_head_quant_scales=False`
# even for int4_per_token_head. Probing with True contradicts the observed runs.
PER_HEAD: set[str] = set()

CASES = {
    "head256_no_mla_as_served": dict(head_size=256, use_mla=False),
    "head256_mla": dict(head_size=256, use_mla=True),
    "head128_no_mla": dict(head_size=128, use_mla=False),
    "head576_mla_deepseek_shape": dict(head_size=576, use_mla=True),
}


def probe(kv: str, head_size: int, use_mla: bool) -> dict:
    cfg = AttentionSelectorConfig(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv,
        block_size=None,
        use_mla=use_mla,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=kv in PER_HEAD,
    )
    try:
        valid, invalid = CudaPlatform.get_valid_backends(
            device_capability=CAP, attn_selector_config=cfg, num_heads=4)
    except Exception as exc:
        return {"raised": f"{type(exc).__name__}: {exc}"[:300]}
    return {
        "valid": [c.backend.name for c in valid],
        "invalid": {b.name: reasons for b, (_, reasons) in invalid.items()},
    }


def main() -> None:
    matrix = {kv: {name: probe(kv, **kw) for name, kw in CASES.items()}
              for kv in DTYPES}
    summary = {kv: {name: cell.get("valid", cell.get("raised"))
                    for name, cell in cells.items()}
               for kv, cells in matrix.items()}
    print(json.dumps({
        "method": ("vllm.platforms.cuda.CudaPlatform.get_valid_backends called "
                   "directly at sm120 with no CUDA context; this is the same "
                   "function whose empty result raises the startup refusal"),
        "device_capability": "sm120 (12, 0)",
        "num_heads": 4,
        "valid_backends": summary,
        "reasons": matrix,
    }, indent=1))


if __name__ == "__main__":
    main()
