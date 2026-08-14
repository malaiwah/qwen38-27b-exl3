#!/usr/bin/env python3
"""Materialise a serialized EXL3 lm_head as a dense BF16 matrix.

Used to attribute head quantization error: replay the *same* candidate hidden
states through the source BF16 head and through this reconstruction. The
difference is the head's contribution with the transformer body held byte-identical.

Uses exllamav3's own `reconstruct_had_slice` kernel, which emits original-basis
weights with both Hadamards and the sign/scale vectors folded in — i.e. exactly
the matrix the serving path multiplies by, not an independent dequantiser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SLICE_N = 32768


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="EXL3 checkpoint dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--prefix", default="lm_head")
    args = ap.parse_args()

    import exllamav3  # noqa: F401  loads the compiled extension
    from exllamav3.ext import exllamav3_ext as ext

    idx = json.loads(Path(args.model, "model.safetensors.index.json").read_text())["weight_map"]
    want = {k: v for k, v in idx.items() if k.startswith(args.prefix + ".")}
    if not want:
        raise SystemExit(f"no tensors under {args.prefix}")
    t: dict[str, torch.Tensor] = {}
    for key, shard in want.items():
        with safe_open(str(Path(args.model, shard)), framework="pt", device="cpu") as f:
            t[key.split(".")[-1]] = f.get_tensor(key)
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in t.items()}, flush=True)

    trellis = t["trellis"].cuda()
    suh = t["suh"].cuda()
    svh = t["svh"].cuda()
    mcg = "mcg" in t
    mul1 = "mul1" in t
    K = trellis.shape[-1] // 16
    in_f = trellis.shape[0] * 16
    out_f = trellis.shape[1] * 16
    print(f"in={in_f} out={out_f} K={K} mcg={mcg} mul1={mul1}", flush=True)

    w_full = torch.empty((in_f, out_f), dtype=torch.float16, device="cuda")
    for n0 in range(0, out_f, SLICE_N):
        n1 = min(n0 + SLICE_N, out_f)
        w = w_full[:, n0:n1].contiguous()
        ext.reconstruct_had_slice(w, trellis, suh, svh[n0:], K, mcg, mul1, n0)
        w_full[:, n0:n1] = w
        print(f"  reconstructed [{n0}:{n1}]", flush=True)

    head = w_full.T.contiguous().to(torch.bfloat16)  # [out, in], HF lm_head.weight layout
    finite = torch.isfinite(head).all().item()
    print(f"head {tuple(head.shape)} {head.dtype} finite={finite} "
          f"absmax={head.abs().max().item():.5f}", flush=True)
    if not finite:
        raise SystemExit("reconstruction produced non-finite values")
    save_file({"weight": head.cpu()}, args.out)
    print("wrote", args.out, round(Path(args.out).stat().st_size / 1e9, 3), "GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
