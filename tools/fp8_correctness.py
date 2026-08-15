#!/usr/bin/env python3
"""Localise the row-wise FP8 defect: compare each stage against an fp16 reference.

Serving with row-wise scales produced KLD 10.8 and 0.1 % top-1, i.e. garbage, while the
per-tensor version merely lost fidelity. Something in the scaling contract is wrong, not the
kernel arithmetic, so this checks the pieces separately:

  A  kernel per-column store, dequantised by hand, against the fp16 reconstruct
  B  _scaled_mm with row-wise scales against an fp16 matmul of the same operands
  C  the full path as the serving code runs it
"""
import sys

import torch

# Imported only by ``main``; ``--variants`` is a standalone PyTorch reproducer.

E4M3_MAX = 448.0


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-9)).item()


def main() -> int:
    sys.path.insert(0, "/work/exllamav3")
    from exllamav3.ext import exllamav3_ext as ext
    dev = "cuda"
    torch.manual_seed(0)
    k, n, bits, m = 2048, 4096, 6, 256
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16),
                            dtype=torch.int16, device=dev)
    w16 = torch.empty((k, n), dtype=torch.float16, device=dev)
    ext.reconstruct(w16, trellis, bits, True, False)

    # A: per-column store round-trip
    col_amax = w16.abs().amax(dim=0).float().clamp_(min=1e-6)
    scale_b = (col_amax / E4M3_MAX)
    w8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=dev)
    ext.reconstruct_fp8_slice(w8, trellis, bits, True, False, 0, 1.0, scale_b.reciprocal())
    deq = w8.float() * scale_b.view(-1, 1)          # (n,k) rows scaled by their own scale
    print(f"A per-column store round-trip: max rel err {rel(deq.t(), w16):.5f}")

    # B: scaled_mm contract
    x = torch.randn((m, k), dtype=torch.float16, device=dev) * 0.05
    row_amax = x.abs().amax(dim=1, keepdim=True).float().clamp_(min=1e-6)
    scale_a = row_amax / E4M3_MAX
    x8 = (x / scale_a).to(torch.float8_e4m3fn)
    ref = (x.float() @ w16.float())
    got = torch._scaled_mm(x8, w8.t(), scale_a=scale_a, scale_b=scale_b.view(1, -1),
                           out_dtype=torch.float16)
    print(f"B rowwise _scaled_mm vs fp32 matmul: max rel err {rel(got, ref):.5f}")

    # B2: what per-tensor scaling costs, for contrast
    ts_a = (x.abs().amax().float() / E4M3_MAX).reshape(())
    ts_b = (w16.abs().amax().float() / E4M3_MAX).reshape(())
    x8t = (x / ts_a).to(torch.float8_e4m3fn)
    w8t = (w16 / ts_b).to(torch.float8_e4m3fn).t().contiguous().t()
    got_t = torch._scaled_mm(x8t, w8t, scale_a=ts_a, scale_b=ts_b, out_dtype=torch.float16)
    print(f"B2 per-tensor _scaled_mm vs fp32 matmul: max rel err {rel(got_t, ref):.5f}")

    # C: does the transposed view of the kernel output equal the reference weight layout?
    #    _scaled_mm needs mat2 column-major; w8.t() is (k,n) with stride (1,k)
    print(f"C w8.t() shape {tuple(w8.t().shape)} strides {w8.t().stride()} "
          f"expected (k,n)=({k},{n}) with stride(0)==1")
    return 0




def variants() -> int:
    """Which row-wise scale layout does this torch build actually implement?"""
    dev = "cuda"
    torch.manual_seed(0)
    k, n, m = 2048, 4096, 256
    w = (torch.randn((k, n), dtype=torch.float16, device=dev) * 0.02)
    x = (torch.randn((m, k), dtype=torch.float16, device=dev) * 0.05)
    ref = x.float() @ w.float()
    col = w.abs().amax(dim=0).float().clamp_(min=1e-6) / E4M3_MAX      # (n,)
    row = x.abs().amax(dim=1, keepdim=True).float().clamp_(min=1e-6) / E4M3_MAX  # (m,1)
    x8 = (x / row).to(torch.float8_e4m3fn)
    w8 = (w / col).to(torch.float8_e4m3fn).t().contiguous()            # (n,k) row-major
    for name, sa, sb, out_dt in (
        ("(m,1)+(1,n) fp16", row, col.view(1, -1), torch.float16),
        ("(m,1)+(n,1) fp16", row, col.view(-1, 1), torch.float16),
        ("(m,1)+(1,n) bf16", row, col.view(1, -1), torch.bfloat16),
        ("(m,)+(n,)   fp16", row.view(-1), col.view(-1), torch.float16),
    ):
        try:
            got = torch._scaled_mm(x8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=out_dt)
            print(f"  {name}: max rel err {rel(got, ref):.5f}")
        except Exception as exc:
            print(f"  {name}: {type(exc).__name__}: {str(exc)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(variants() if "--variants" in sys.argv else main())
