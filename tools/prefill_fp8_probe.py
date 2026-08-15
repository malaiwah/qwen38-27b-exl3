#!/usr/bin/env python3
"""What would an FP8 prefill path actually cost, end to end, per matrix?

The GEMM itself is 1.9x faster in FP8 (measured), but `torch._scaled_mm` requires the
weight operand to be column-major, and EXL3's `reconstruct` writes a contiguous (K, N)
fp16 buffer. So the honest comparison has to price the layout conversion:

  A  reconstruct + ext.hgemm                          - what ships today
  B  reconstruct + transposing cast to FP8 + _scaled_mm - implementable in Python now
  C  reconstruct + _scaled_mm with the FP8 weight already in place
       - the ceiling if the reconstruct kernel is changed to emit FP8 directly,
         which is the only way to avoid paying for a second pass over the weight

Weight scales are per-tensor and precomputed: the reconstructed matrix is the raw
trellis-decoded weight (suh/svh are applied to the activations and the output, not
folded in), so its dynamic range is a property of the codebook and can be measured once
per matrix at load instead of per chunk.
"""
import sys
import time

import torch

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext  # noqa: E402

E4M3_MAX = 448.0


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def fake_trellis(k: int, n: int, bits: int, device="cuda"):
    """A trellis tensor of the right shape and dtype for reconstruct to decode."""
    return torch.randint(
        -32768, 32767, (k // 16, n // 16, bits * 16), dtype=torch.int16, device=device
    )


def main() -> int:
    dev = "cuda"
    print(f"{'K':>6} {'N':>7} {'bits':>4} {'m':>5} "
          f"{'A recon+hgemm':>14} {'B +cast+fp8mm':>14} {'C fp8mm only':>13} "
          f"{'B/A':>6} {'C/A':>6}")
    for k, n, bits in ((5120, 17408, 5), (17408, 5120, 6), (5120, 17408, 6)):
        trellis = fake_trellis(k, n, bits, dev)
        w16 = torch.empty((k, n), dtype=torch.float16, device=dev)
        ext.reconstruct(w16, trellis, bits, True, False)
        # per-tensor weight scale, precomputed once (this is the load-time step)
        w_amax = w16.abs().amax().clamp_(min=1e-6)
        scale_b = (w_amax / E4M3_MAX).to(torch.float32).reshape(())
        w8_colmajor = (w16 / scale_b).to(torch.float8_e4m3fn).t().contiguous().t()

        for m in (256, 1024, 2048):
            x = torch.randn((m, k), dtype=torch.float16, device=dev) * 0.05
            y16 = torch.empty((m, n), dtype=torch.float16, device=dev)
            # Two valid scaling modes: TensorWise wants scalar scales on both
            # operands, RowWise wants (m,1) and (1,n). Mixing them is rejected.
            x_amax = x.abs().amax().clamp_(min=1e-6)
            scale_a = (x_amax / E4M3_MAX).to(torch.float32).reshape(())
            x8 = (x / scale_a).to(torch.float8_e4m3fn)

            def path_a():
                ext.reconstruct(w16, trellis, bits, True, False)
                ext.hgemm(x, w16, y16)

            def path_b():
                ext.reconstruct(w16, trellis, bits, True, False)
                w8 = (w16 / scale_b).to(torch.float8_e4m3fn).t().contiguous().t()
                torch._scaled_mm(x8, w8, scale_a=scale_a, scale_b=scale_b,
                                 out_dtype=torch.float16)

            def path_c():
                ext.reconstruct(w16, trellis, bits, True, False)
                torch._scaled_mm(x8, w8_colmajor, scale_a=scale_a, scale_b=scale_b,
                                 out_dtype=torch.float16)

            try:
                a = bench(path_a)
                b = bench(path_b)
                c = bench(path_c)
            except Exception as exc:
                print(f"{k:6d} {n:7d} {bits:4d} {m:5d} failed: "
                      f"{type(exc).__name__}: {str(exc)[:400]}")
                continue
            print(f"{k:6d} {n:7d} {bits:4d} {m:5d} "
                  f"{a*1e3:13.3f}ms {b*1e3:13.3f}ms {c*1e3:12.3f}ms "
                  f"{a/b:5.2f}x {a/c:5.2f}x")
            del x, y16, x8
        del trellis, w16, w8_colmajor
        torch.cuda.empty_cache()

    print("\nB/A > 1 means the Python-only FP8 path already wins; C/A is the ceiling "
          "available if reconstruct emits FP8 in column-major order itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
