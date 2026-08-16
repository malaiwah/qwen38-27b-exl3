#!/usr/bin/env python3
"""Hold GPU memory so that CUDA-visible free memory equals a target figure.

Why this exists: vLLM sizes its budget as `ceil(cuda_total * gpu_memory_utilization)`
and then refuses to start if `cuda_free < requested`, but it never clamps the KV pool
to free memory -- the KV pool is `requested - non_kv - cudagraph`. So lowering
`--gpu-memory-utilization` alone emulates a smaller board's *budget* while leaving the
whole rest of a 32 GiB card available for transient allocations that live outside the
budget (the vision tower's, notably). That is the one thing the physical RTX 5090
qualification proved matters: at utilisation 0.97 the engine started and served text
and then could not find 62 MiB outside its budget for a 7 MP image.

This process removes that unearned slack. It allocates until `cudaMemGetInfo` reports
exactly the free figure a smaller board would report, so a following engine sees both
the smaller budget and the smaller card. It is a proxy for a physical board, never a
substitute: the CUDA context size and the driver's own framebuffer reserve are
properties of the real device and are not emulated here.

Run it inside the same container image as the engine so the CUDA user-mode driver and
torch build are identical.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch

GIB = 1 << 30
BLOCK = 512 * (1 << 20)  # allocate in 512 MiB steps so the driver can satisfy each one
TOLERANCE = 8 * (1 << 20)  # 8 MiB: below one allocator granule pair, not worth chasing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-free-gib", type=float, required=True,
                        help="CUDA-visible free bytes to leave, in GiB, as the engine "
                             "will see them before it creates its own context")
    parser.add_argument("--report", required=True)
    parser.add_argument("--hold-sec", type=int, default=7200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible to the ballast process")

    # Materialise the context before measuring: the context itself is part of what this
    # process costs the card, and the caller's target must be met net of it.
    torch.zeros(1, dtype=torch.uint8, device="cuda")
    free_before, total = torch.cuda.mem_get_info()
    want_free = int(args.target_free_gib * GIB)
    if want_free >= free_before:
        raise SystemExit(
            f"target free {args.target_free_gib:.3f} GiB is not below the free figure "
            f"already available ({free_before / GIB:.3f} GiB of {total / GIB:.3f} GiB "
            f"total): nothing to ballast, and the ballast may not release memory"
        )

    blocks: list[torch.Tensor] = []
    free_now = free_before
    guard = 0
    while free_now - want_free > TOLERANCE:
        step = min(BLOCK, free_now - want_free)
        while True:
            try:
                blocks.append(torch.empty(step, dtype=torch.uint8, device="cuda"))
                break
            except torch.cuda.OutOfMemoryError:
                step //= 2
                if step <= TOLERANCE:
                    raise SystemExit(
                        "the card refused even a small ballast block; another process "
                        "is taking memory concurrently, so this run would not be a "
                        "controlled measurement"
                    )
        free_now = torch.cuda.mem_get_info()[0]
        guard += 1
        if guard > 512:
            raise SystemExit("ballast did not converge on its target free figure")

    report = {
        "schema": "qwen38-proxy24-ballast/1",
        "cuda_total_bytes": total,
        "cuda_total_gib": round(total / GIB, 3),
        "cuda_free_before_ballast_bytes": free_before,
        "cuda_free_before_ballast_gib": round(free_before / GIB, 3),
        "target_free_gib": args.target_free_gib,
        "cuda_free_after_ballast_bytes": free_now,
        "cuda_free_after_ballast_gib": round(free_now / GIB, 3),
        "target_error_mib": round((free_now - want_free) / (1 << 20), 2),
        "ballast_blocks": len(blocks),
        "ballast_tensor_bytes": sum(b.numel() for b in blocks),
        "ballast_tensor_gib": round(sum(b.numel() for b in blocks) / GIB, 3),
        "device_used_after_ballast_gib": round((total - free_now) / GIB, 3),
        "context_and_overhead_gib": round((total - free_before) / GIB, 3),
        "note": "context_and_overhead_gib is what this process cost the card before any "
                "ballast tensor existed, i.e. its CUDA context plus torch's own "
                "allocations; a physical smaller board does not pay it, so it is "
                "reported rather than hidden",
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)

    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    deadline = time.monotonic() + args.hold_sec
    while not stop and time.monotonic() < deadline:
        time.sleep(0.5)
    print("ballast released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
