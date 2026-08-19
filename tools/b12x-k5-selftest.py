"""Is B12X W4A16 correct for K5 (bits=5) trellis payloads?

Compares, on real checkpoint weights:
  reference = exllamav3 ext.reconstruct(trellis) -> W, then x @ W  (the path the
              fused exl3_gemm kernel agrees with, and what the served model used
              before B12X was enabled)
  candidate = b12x prepare_weight + trellis_linear.run

K6 matrix (down_proj) is the control: it must match, because that is the
configuration currently serving at KLD 0.003407.
K5 matrix (gate_proj) is the subject.

Also tests whether passing the checkpoint's own `mcg` tensor explicitly (rather
than only codebook="mcg", which is what our integration does) changes the K5
result - that distinguishes "our call is under-specified" from "the bits=5 dense
kernel is wrong".
"""
import os
import sys

import torch

sys.path.insert(0, "/opt/exllamav3")  # dir containing exllamav3_ext*.so
from safetensors import safe_open  # noqa: E402

SNAP = os.environ["SNAP"]
LAYER = 3
DEV = "cuda:0"
import os as _os
DTYPE = torch.bfloat16 if _os.environ.get("PROBE_DTYPE","fp16")=="bf16" else torch.float16


def load(prefix):
    out = {}
    import json
    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
    for suffix in ("trellis", "suh", "svh", "mcg"):
        key = f"{prefix}.{suffix}"
        if key not in idx:
            continue
        with safe_open(f"{SNAP}/{idx[key]}", framework="pt", device=DEV) as f:
            out[suffix] = f.get_tensor(key)
    return out


def bits_of(trellis):
    return int(trellis.shape[2]) // 16


def reference(w, x):
    """exllamav3 reconstruct + Hadamard fold: the bit-faithful reference path."""
    import importlib
    ext = importlib.import_module("exllamav3_ext")
    sys.path.insert(0, "/opt/fp4")
    from exl3_fp4_conversion import hadamard_fold_weight_chunked
    k16, n16 = w["trellis"].shape[0], w["trellis"].shape[1]
    K, N = k16 * 16, n16 * 16
    W = torch.zeros((K, N), dtype=torch.float16, device=DEV)  # ext requires kHalf
    ext.reconstruct(W, w["trellis"], bits_of(w["trellis"]),
                    w.get("mcg") is not None, False)
    W = hadamard_fold_weight_chunked(W, w["suh"].half(), w["svh"].half())
    return (x.float() @ W.float()).half()


def candidate(w, x, pass_mcg):
    from b12x.gemm.trellis_linear import api
    kw = {}
    if pass_mcg and w.get("mcg") is not None:
        kw["mcg"] = w["mcg"]
    pw = api.prepare_weight(
        w["trellis"], w["suh"], w["svh"],
        codebook="mcg", params_dtype=DTYPE, **kw,
    )
    m = x.shape[0]
    n = w["trellis"].shape[1] * 16
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    out = torch.empty((m, n), dtype=DTYPE, device=DEV)
    api.run(
        x, pw,
        output=out,
        gemm_output=torch.empty_like(out),
        c_tmp=torch.empty((sms * 4 * 64 * 256,), dtype=torch.float32, device=DEV),
        rotated_f16=torch.empty_like(x),
    )
    return out


def compare(tag, ref, got):
    ref32, got32 = ref.float(), got.float()
    cos = torch.nn.functional.cosine_similarity(
        ref32.flatten(), got32.flatten(), dim=0
    ).item()
    denom = ref32.abs().amax().clamp_min(1e-6)
    max_rel = ((got32 - ref32).abs().amax() / denom).item()
    bad = (cos < 0.999) or (max_rel > 0.05)
    print(f"  {tag:<34} cos={cos:.6f}  max_rel={max_rel:.4e}  "
          f"{'*** MISMATCH ***' if bad else 'ok'}")
    return not bad


torch.manual_seed(0)
base = f"model.language_model.layers.{LAYER}"
results = {}
for name, prefix in (("K6 control (down_proj)", f"{base}.mlp.down_proj"),
                     ("K5 subject (gate_proj)", f"{base}.mlp.gate_proj")):
    w = load(prefix)
    b = bits_of(w["trellis"])
    K = w["trellis"].shape[0] * 16
    x = torch.randn((256, K), dtype=DTYPE, device=DEV) * 0.05
    print(f"{name}: bits={b} K={K} N={w['trellis'].shape[1]*16} "
          f"mcg={'yes' if w.get('mcg') is not None else 'no'}")
    ref = reference(w, x)
    for pass_mcg in (False, True):
        try:
            got = candidate(w, x, pass_mcg)
            ok = compare(f"b12x (mcg tensor={'yes' if pass_mcg else 'no'})", ref, got)
            results[(b, pass_mcg)] = ok
        except Exception as e:
            print(f"  b12x (mcg tensor={'yes' if pass_mcg else 'no'}) RAISED: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            results[(b, pass_mcg)] = False
    print()

print("VERDICT")
print(f"  K6 without mcg tensor: {'PASS' if results.get((6, False)) else 'FAIL'}")
print(f"  K5 without mcg tensor: {'PASS' if results.get((5, False)) else 'FAIL'}")
print(f"  K5 with    mcg tensor: {'PASS' if results.get((5, True)) else 'FAIL'}")
