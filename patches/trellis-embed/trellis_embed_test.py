#!/usr/bin/env python3
"""Trellis K6/K8 embedding quantization test for Qwen3.8-27B.

Encodes the BF16 embedding table to EXL3 trellis K6/K8 at load time,
reconstructs to fp16 via the stock reconstruct + hadamard_fold pipeline,
and measures KLD against the BF16 reference.

This is a fidelity test, not a production embedding path. The production
path requires a row-indexed reconstruct kernel (issue #435) to avoid
reconstructing the full 2.54 GB table at serving time.

Usage on aiboss (inside the vLLM container):
    python3 trellis_embed_test.py --model /path/to/checkpoint \
        --bits 6 --suite /path/to/suite --out /path/to/capture \
        --head /path/to/lm_head.safetensors --reference /path/to/bf16_ref
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--bits", type=int, default=6, choices=[6, 8],
                        help="trellis K width for embedding")
    parser.add_argument("--suite", required=True, help="suite directory with token files")
    parser.add_argument("--out", required=True, help="output capture directory")
    parser.add_argument("--head", required=True, help="shared BF16 LM head")
    parser.add_argument("--reference", required=True, help="BF16 reference hidden states")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Use the stock fidelity.py for capture and replay
    fidelity = Path(__file__).resolve().parents[1] / "tools" / "fidelity.py"

    # Step 1: Capture with trellis embeddings
    print(f"=== Trellis K{args.bits} embedding capture ===", flush=True)
    capture_dir = Path(args.out) / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)

    # We'll use fidelity.py capture but with a custom hook that encodes
    # the embedding to trellis and reconstructs it before the forward pass.
    # The simplest way: set VLLM_EXL3_EMBED_TRELLIS_BITS env var and mount
    # our test script as a sitecustomize hook.

    # Actually, for the KLD test, we can do it differently:
    # 1. Load the model normally (BF16 embeddings)
    # 2. Encode the embedding to trellis K{bits}, reconstruct to fp16
    # 3. Replace the embedding weight with the reconstructed fp16
    # 4. Run the capture
    # This measures the trellis round-trip error on embeddings.

    # Use fidelity.py with a custom pre-capture hook
    import subprocess
    cmd = [
        sys.executable, str(fidelity), "capture",
        "--model", args.model,
        "--suite", args.suite,
        "--out", str(capture_dir),
        "--quantization", "auto",
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--attention-backend", "TRITON_ATTN",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--filter", "all",
        "--max-batched-tokens", str(args.max_batched_tokens),
        "--hash-shards",
    ]
    env = os.environ.copy()
    env["VLLM_EXL3_EMBED_TRELLIS_BITS"] = str(args.bits)
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env["VLLM_EXL3_EMBED_ONLINE_BITS"] = "0"  # disable int8/int6 overlay
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"CAPTURE FAILED (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    # Step 2: Replay
    print(f"\n=== Trellis K{args.bits} embedding replay ===", flush=True)
    report_path = Path(args.out) / "report.json"
    cmd = [
        sys.executable, str(fidelity), "replay",
        "--reference", args.reference,
        "--candidate", str(capture_dir),
        "--head", args.head,
        "--suite", args.suite,
        "--out", str(report_path),
        "--filter", "all",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"REPLAY FAILED (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    # Print results
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"\n=== Trellis K{args.bits} Embedding KLD Results ===")
        for k in ("mean_kld", "token_mean_kld", "context_macro_mean_kld",
                   "p99_kld", "p999_kld", "max_kld", "top1_agreement"):
            if k in report:
                print(f"  {k}: {report[k]}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
