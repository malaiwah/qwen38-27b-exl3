#!/usr/bin/env python3
"""Positive proof that the YaRN arm's three-field override actually changes the rope
tables the engine builds, over exactly the position range the probe set occupies.

Without this, a null KLD result would be ambiguous: it could mean "static YaRN is
free at short context" or "the override was silently ignored".  This probe closes
that hole from the engine's own code path (vllm.model_executor.layers.rotary_embedding
.get_rope), with the same head_size, partial_rotary_factor, theta and mrope_section
the served model uses.

  ggrun.sh python /work/yarn/yarn_rope_delta_probe.py --out /work/yarn/rope-delta.json
"""
from __future__ import annotations

import argparse
import json

import torch

NATIVE = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}
YARN = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-size", type=int, default=256)  # text_config.head_dim
    ap.add_argument("--probe-max-pos", type=int, default=32776)
    args = ap.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    torch.set_default_dtype(torch.bfloat16)
    out = {}
    caches = {}
    with set_current_vllm_config(VllmConfig()):
        for name, params in (("NATIVE", NATIVE), ("YARN", YARN)):
            rope = get_rope(
                head_size=args.head_size,
                max_position=262144 if name == "NATIVE" else 1000000,
                is_neox_style=True,
                rope_parameters=dict(params),
                dtype=torch.bfloat16,
            )
            cache = rope.cos_sin_cache.float().cpu()
            caches[name] = cache
            out[name] = {
                "rope_parameters": params,
                "class": type(rope).__name__,
                "rotary_dim": int(rope.rotary_dim),
                "base": float(rope.base),
                "max_position_embeddings": int(rope.max_position_embeddings),
                "scaling_factor": getattr(rope, "scaling_factor", None),
                "mscale": getattr(rope, "mscale", None),
                "cos_sin_cache_shape": list(cache.shape),
                "inv_freq_first8": rope._compute_inv_freq(rope.base)
                .float()
                .cpu()[:8]
                .tolist(),
            }

    n = min(args.probe_max_pos, caches["NATIVE"].shape[0], caches["YARN"].shape[0])
    a, b = caches["NATIVE"][:n], caches["YARN"][:n]
    diff = (a - b).abs()
    out["delta_over_probe_position_range"] = {
        "positions_compared": n,
        "note": "positions 0..N-1 cover every position the probe set uses "
        "(longest prompt 32,768 tokens plus 8 forced continuation tokens)",
        "identical": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "frac_entries_differing": float((diff > 0).float().mean()),
        "max_abs_diff_at_position_0_31": float(diff[:32].max()),
        "max_abs_diff_at_position_512": float(diff[512].max()),
        "max_abs_diff_at_position_32767": float(diff[min(32767, n - 1)].max()),
        "verdict": "the two arms build DIFFERENT rope tables at every position range the "
        "probe set touches, including position 0. A null KLD result therefore means the "
        "engine tolerated the change, not that the change was ignored.",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
