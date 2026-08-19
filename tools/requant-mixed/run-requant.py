#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Driver script for GPTQ mixed-precision requantization of Qwen/Qwen3.8-27B
using llmcompressor 0.13.0 + compressed-tensors 0.17.0.

Produces a compressed-tensors checkpoint that stock vLLM can load via
--quantization compressed-tensors (or auto-detect from config.json).

Usage:
  python run-requant.py \
      --recipe recipe-fp8attn-nvfp4mlp.yaml \
      --output-dir /tmp/qwen38-27b-fp8attn-nvfp4mlp \
      --model Qwen/Qwen3.8-27B \
      --samples 512 \
      --seq-len 2048

Prerequisites (see README.md for full venv setup):
  pip install llmcompressor==0.13.0 compressed-tensors==0.17.0
  # plus transformers >= 5.8.0.dev0 (or trust_remote_code for Qwen3.5)
  # plus torch, datasets, accelerate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _fail(msg: str) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_prerequisites() -> None:
    """Fail loudly if llmcompressor or compressed-tensors are missing/wrong version."""
    try:
        import llmcompressor  # noqa: F401
    except ImportError:
        _fail(
            "llmcompressor is not installed. See README.md for venv setup: "
            "pip install llmcompressor==0.13.0"
        )
    try:
        import compressed_tensors  # noqa: F401
    except ImportError:
        _fail(
            "compressed-tensors is not installed. "
            "pip install compressed-tensors==0.17.0"
        )

    # Verify versions
    import llmcompressor as _lc
    _lc_ver = getattr(_lc, "__version__", "")
    if not _lc_ver.startswith("0.13"):
        print(
            f"WARNING: llmcompressor version is {_lc_ver}, "
            "recipe was verified against 0.13.0. Proceed at your own risk.",
            file=sys.stderr,
        )

    try:
        import importlib.metadata
        ct_ver = importlib.metadata.version("compressed-tensors")
        if not ct_ver.startswith("0.17"):
            print(
                f"WARNING: compressed-tensors version is {ct_ver}, "
                "recipe was verified against 0.17.0. Proceed at your own risk.",
                file=sys.stderr,
            )
    except Exception:
        pass  # best-effort version check

    # Check torch
    try:
        import torch  # noqa: F401
    except ImportError:
        _fail("torch is not installed. pip install torch")

    # Check transformers
    try:
        import transformers  # noqa: F401
    except ImportError:
        _fail("transformers is not installed. pip install transformers")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="GPTQ mixed-precision requantization via llmcompressor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--recipe",
        required=True,
        help="Path to recipe YAML file (recipe-fp8attn-nvfp4mlp.yaml or "
        "recipe-fp8attn-nvfp4w4a4mlp.yaml)",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the compressed-tensors checkpoint",
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3.8-27B",
        help="HF model id or local path (default: Qwen/Qwen3.8-27B)",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Max sequence length for calibration (default: 2048)",
    )
    ap.add_argument(
        "--dataset",
        default="ultrachat-200k",
        help="HF dataset name for calibration (default: "
        "HuggingFaceH4/ultrachat_200k)",
    )
    ap.add_argument(
        "--dataset-split",
        default="train_sft",
        help="Dataset split to use (default: train_sft)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size (default: 1, keep low for memory)",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="GPTQ block size (columns per pass, default: 128)",
    )
    ap.add_argument(
        "--dampening-frac",
        type=float,
        default=0.01,
        help="GPTQ Hessian dampening fraction (default: 0.01)",
    )
    ap.add_argument(
        "--precision",
        default="auto",
        help="Model loading precision (default: auto, uses config dtype)",
    )
    ap.add_argument(
        "--pipeline",
        default="sequential",
        choices=["sequential", "basic", "datafree", "independent"],
        help="Calibration pipeline (default: sequential; use datafree for "
        "RTN QuantizationModifier recipes - no calibration, no tracing)",
    )
    ap.add_argument(
        "--sequential-offload",
        default="cpu",
        help="Offload device for sequential pipeline (default: cpu)",
    )
    ap.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code (Qwen3.5 normally needs it)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    check_prerequisites()

    recipe_path = Path(args.recipe)
    if not recipe_path.is_absolute():
        # Resolve relative to this script's directory
        recipe_path = Path(__file__).resolve().parent / recipe_path
    if not recipe_path.exists():
        _fail(f"Recipe file not found: {recipe_path}")

    # Validate recipe parses as YAML
    try:
        import yaml

        with open(recipe_path) as f:
            yaml.safe_load(f)
    except Exception as e:
        _fail(f"Recipe YAML parse error: {e}")

    print(f"=== Recipe: {recipe_path}")
    print(f"=== Model:  {args.model}")
    print(f"=== Output: {args.output_dir}")
    print(f"=== Calibration: {args.dataset} / {args.dataset_split}")
    print(f"=== Samples: {args.samples}, Seq len: {args.seq_len}")
    print()

    # Import oneshot after prerequisite checks
    from llmcompressor import oneshot

    # Sequential targets: match each decoder layer in the language model.
    # The Qwen3.5 architecture has layers at model.language_model.layers.N
    # (64 layers total: 16 full_attention + 48 linear_attention/GDN).
    # The sequential pipeline processes one layer at a time, offloading
    # the rest to CPU, keeping VRAM usage bounded to ~1-2 layers on GPU.
    # Whole-decoder-layer subgraphs OOM on 32 GiB (512x2048 calibration
    # activations live across the full layer). Per-Linear granularity is the
    # pipeline's own recommendation for dense models and bounds live memory to
    # one projection at a time.
    sequential_targets = ["Linear"]

    # Qwen3.8-27B is multimodal (Qwen3_5ForConditionalGeneration); llmcompressor's
    # pre_process cannot auto-initialize its processor and aborts when a dataset is
    # supplied. Text-only calibration needs only the tokenizer, so build it
    # explicitly and hand it over.
    from transformers import AutoTokenizer
    processor = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=not args.no_trust_remote_code
    )

    oneshot(
        model=args.model,
        processor=processor,
        recipe=str(recipe_path),
        trust_remote_code_model=not args.no_trust_remote_code,
        precision=args.precision,
        save_compressed=True,

        # Dataset
        dataset=args.dataset,
        splits=f"{args.dataset_split}[:{args.samples}]",
        num_calibration_samples=args.samples,
        max_seq_length=args.seq_len,
        batch_size=args.batch_size,
        concatenate_data=True,
        pad_to_max_length=False,
        text_column="messages",

        # Sequential pipeline for memory-efficient GPTQ on large models.
        # The sequential pipeline loads one decoder layer to GPU at a time,
        # processes all calibration samples through it, then offloads to CPU.
        # This keeps peak VRAM at ~1 layer's weights + activations + Hessian
        # rather than the full 27B model.
        pipeline=args.pipeline,
        sequential_targets=sequential_targets,
        sequential_offload_device=args.sequential_offload,
        sequential_prefetch=False,

        output_dir=args.output_dir,
    )

    print()
    print(f"=== Done. Compressed checkpoint saved to: {args.output_dir}")
    print("=== To serve with vLLM:")
    print(f"    vllm serve {args.output_dir} --quantization compressed-tensors")
    print("=== (vLLM auto-detects compressed-tensors from config.json "
    "if quant_method is set)")


if __name__ == "__main__":
    main()
