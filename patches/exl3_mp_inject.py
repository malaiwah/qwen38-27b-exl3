#!/usr/bin/env python3
"""Multi-precision injection: runs AFTER model load + AOT compilation.

This script is imported by vLLM's exl3.py (via the linear.py patch) AFTER
the model is loaded and AOT-compiled. It:

1. Iterates over all EXL3 linear layers in the model
2. Converts each layer's trellis weights to FP4 (MLP) or FP6 (attention/GDN)
   at load time (Hadamard folded into weights)
3. Monkey-patches each layer's forward method to use the multi-precision GEMM
   instead of the trellis path

The monkey-patch runs EAGERLY (not AOT-compiled), so the AOT cache stays
valid. The multi-precision GEMM (b12x dense_gemm) is fast enough that the
eager overhead is negligible for large M (prefill).

Enable via: VLLM_EXL3_MP_INJECT=1
"""

from __future__ import annotations

import os
import sys
import logging
import importlib.util
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_FP6_CONV = None
_FP4_CONV = None

# Layer routing: MLP → FP4 (4x MMA), attention/GDN → FP6 (2x MMA)
_FP4_PATTERNS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_FP6_PATTERNS = ("linear_attn.", "self_attn.")


def _load_module(name: str, path: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # Required for @dataclass
    spec.loader.exec_module(mod)
    return mod


def _get_precision(prefix: str) -> str:
    name = prefix.lower()
    for pat in _FP4_PATTERNS:
        if pat in name:
            return "fp4"
    for pat in _FP6_PATTERNS:
        if pat in name:
            return "fp6"
    return "fp6"  # Default: FP6 for unknown layers


def _load_fp6_conv():
    global _FP6_CONV
    if _FP6_CONV is None:
        _FP6_CONV = _load_module("exl3_fp6_conversion", "/opt/fp6/exl3_fp6_conversion.py")
        logger.info("FP6 conversion module loaded")
    return _FP6_CONV


def _load_fp4_conv():
    global _FP4_CONV
    if _FP4_CONV is None:
        _FP4_CONV = _load_module("exl3_fp4_conversion", "/opt/fp4/exl3_fp4_conversion.py")
        logger.info("FP4 conversion module loaded")
    return _FP4_CONV


def _get_exl3_ext():
    """Load the exllamav3 extension for trellis reconstruction."""
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_ext
    return _load_exl3_ext()


def _convert_layer(layer, ext, precision):
    """Convert one layer's trellis weights to FP4 or FP6."""
    prefix = getattr(layer, "prefix", layer.__class__.__name__)

    # Unfreeze b12x kernel resolution for CuTe DSL compilation
    try:
        from b12x._lib.runtime_control import unfreeze_kernel_resolution
        unfreeze_kernel_resolution()
    except ImportError:
        pass

    with torch.no_grad():
        if precision == "fp4":
            conv = _load_fp4_conv()
            weights = conv.convert_all_shards_to_fp4(layer, ext)
            layer._mp_weights = weights
            layer._mp_precision = "fp4"
            logger.info("EXL3→FP4 conversion complete for %s (%d shards)",
                        prefix, len(weights))
        elif precision == "fp6":
            conv = _load_fp6_conv()
            weights = conv.convert_all_shards_to_fp6(layer, ext)
            layer._mp_weights = weights
            layer._mp_precision = "fp6"
            logger.info("EXL3→FP6 conversion complete for %s (%d shards)",
                        prefix, len(weights))

    # Free trellis tensors to reclaim VRAM
    for attr in ("trellis", "suh", "svh", "mcg", "mul1"):
        param = getattr(layer, attr, None)
        if param is not None and hasattr(param, "exl3_tensors"):
            param.exl3_tensors.clear()
    torch.cuda.empty_cache()


def _patch_layer_forward(layer):
    """Monkey-patch a layer's forward to use multi-precision GEMM.

    This runs EAGERLY (not AOT-compiled), bypassing the trellis path.
    """
    mp_weights = getattr(layer, "_mp_weights", None)
    if mp_weights is None:
        return  # No conversion happened for this layer

    precision = getattr(layer, "_mp_precision", "fp6")
    original_forward = layer.forward

    def mp_forward(input, *args, **kwargs):
        original_shape = input.shape[:-1]
        original_dtype = input.dtype
        x_2d = input.reshape(-1, input.shape[-1]).to(torch.bfloat16).contiguous()

        outputs = []
        for shard_id in layer.exl3_shard_ids:
            if shard_id in mp_weights:
                w = mp_weights[shard_id]
                if precision == "fp4":
                    conv = _load_fp4_conv()
                    out = conv.fp4_apply(x_2d, w)
                else:
                    conv = _load_fp6_conv()
                    out = conv.fp6_apply(x_2d, w)
                outputs.append(out)
            else:
                # Fallback to original forward for this shard
                # (shouldn't happen if all shards converted)
                out = original_forward(input, *args, **kwargs)
                outputs.append(out)

        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        bias = getattr(layer, "bias", None)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    layer.forward = mp_forward
    logger.info("Patched forward for %s (%s)", getattr(layer, "prefix", ""), precision)


def inject_multiprecision(model):
    """Main entry point: convert and patch all EXL3 layers in the model.

    Called AFTER model load + AOT compilation, BEFORE inference.
    """
    if os.environ.get("VLLM_EXL3_MP_INJECT", "0") != "1":
        return

    logger.info("=== Multi-precision injection starting ===")
    ext = _get_exl3_ext()

    converted = 0
    failed = 0
    skipped = 0

    for name, module in model.named_modules():
        if not hasattr(module, "exl3_shard_ids"):
            continue

        prefix = getattr(module, "prefix", name)
        if "lm_head" in prefix.lower() or "embed" in prefix.lower():
            skipped += 1
            continue

        precision = _get_precision(prefix)
        try:
            _convert_layer(module, ext, precision)
            _patch_layer_forward(module)
            converted += 1
        except Exception as exc:
            logger.warning("Multi-precision conversion failed for %s: %s", prefix, exc)
            failed += 1

    logger.info("=== Multi-precision injection complete: %d converted, %d failed, %d skipped ===",
                converted, failed, skipped)
