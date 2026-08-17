"""Which cuBLAS/CUTLASS kernel does each arm actually get? Confirms the F6
attribution ('cutlass_80_wmma...16x16') and shows why m in {2,3,4} is a cliff."""
import json
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

K, N = 5120, 96
DT = torch.bfloat16
res = {}
w_nk = torch.randn(N, K, device="cuda", dtype=DT)
w_kn = w_nk.t().contiguous()

for m in (1, 2, 4, 8):
    x = torch.randn(m, K, device="cuda", dtype=DT)
    for name, fn in (("tn_F.linear", lambda: F.linear(x, w_nk)),
                     ("nn_torch.mm", lambda: torch.mm(x, w_kn))):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
        ks = {}
        for e in p.key_averages():
            if e.device_time_total > 0 and e.key not in ("cudaLaunchKernel",):
                ks[e.key] = round(e.device_time_total / max(e.count, 1), 3)
        res[f"m{m}/{name}"] = ks
print(json.dumps(res, indent=1))
