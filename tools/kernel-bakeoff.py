# Kernel bake-off on REAL checkpoint shapes: which prefill/decode path is
# fastest per (shape, M)? Output = dispatch table + per-chunk projections.
import torch, sys, importlib, json, traceback
sys.path.insert(0, '/opt/exllamav3'); sys.path.insert(0, '/opt/fp4')
ext = importlib.import_module('exllamav3_ext')
import exl3_fp4_conversion as conv
from vllm.model_executor.layers.quantization.exl3 import (
    _exl3_gemm, _b12x_trellis_linear, _b12x_trellis_k6_supported)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block, per_token_group_quant_fp8)
from vllm.utils.deep_gemm import fp8_gemm_nt

def t_ms(fn, iters=5, warmup=2):
    try:
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        a,b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters): fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b)/iters
    except Exception as e:
        return float('nan')

# (name, K, N, bits, count) from /tmp/shape_inventory.json, language_model only
SHAPES = [
 ("mlp.gate+up", 5120, 17408, 5, 128),
 ("mlp.down",   17408,  5120, 6,  64),
 ("gdn.in_qkv",  5120, 10240, 6,  48),
 ("gdn.in_z",    5120,  6144, 6,  48),
 ("gdn.out",     6144,  5120, 6,  48),
 ("attn.q",      5120, 12288, 6,  16),
 ("attn.o",      6144,  5120, 6,  16),
 ("attn.kv",     5120,  1024, 6,  32),
]
MS = [int(x) for x in sys.argv[1].split(',')]
res = {}
for name,K,N,bits,cnt in SHAPES:
    trellis = torch.randint(-32768,32767,(K//16,N//16,16*bits),dtype=torch.int16,device='cuda')
    suh = (torch.randn(K, device='cuda', dtype=torch.float16)*0.1+1)
    svh = (torch.randn(N, device='cuda', dtype=torch.float16)*0.1+1)
    W = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)*0.02
    pk, sc, gs = conv._quantize_matrix_fp4_nvfp4(W)
    fp4w = conv.FP4DenseWeight(packed=pk, scale_storage=sc, global_scale=gs,
                               out_features=N, in_features=K)
    q_nt = torch.empty(N,K,dtype=torch.float8_e4m3fn,device='cuda')
    ws   = torch.empty(N//128,K//128,dtype=torch.float32,device='cuda')
    ext.reconstruct_fp8dg_nt(q_nt, ws, trellis, bits, False, False)
    q_c, ws_c = deepgemm_post_process_fp8_weight_block(q_nt.clone(), ws.clone(), (128,128), use_e8m0=True)
    b12x_ok = False
    try: b12x_ok = bool(_b12x_trellis_k6_supported(trellis, has_mcg=False, has_mul1=False))
    except Exception: pass
    t_recon = t_ms(lambda: ext.reconstruct_fp8dg_nt(q_nt, ws, trellis, bits, False, False))
    t_post  = t_ms(lambda: deepgemm_post_process_fp8_weight_block(q_nt.clone(), ws.clone(), (128,128), use_e8m0=True))
    for M in MS:
        xh = torch.randn(M,K,device='cuda',dtype=torch.float16)
        xb = xh.to(torch.bfloat16)
        out = torch.empty(M,N,dtype=torch.bfloat16,device='cuda')
        r = {}
        r['exl3_gemm'] = t_ms(lambda: _exl3_gemm(xh, trellis, suh, svh, False, False))
        r['b12x']      = t_ms(lambda: _b12x_trellis_linear(xh, trellis, suh, svh)) if b12x_ok else float('nan')
        r['fp4']       = t_ms(lambda: conv.fp4_apply(xb, fp4w))
        def _dg():
            xq,xs = per_token_group_quant_fp8(xb,128,column_major_scales=True,tma_aligned_scales=True,use_ue8m0=True)
            fp8_gemm_nt((xq,xs),(q_c,ws_c),out,is_deep_gemm_e8m0_used=True)
        r['fp8dg_cached'] = t_ms(_dg)
        r['fp8dg_fused_est'] = (r['fp8dg_cached'] + t_recon) if r['fp8dg_cached']==r['fp8dg_cached'] else float('nan')
        r['fp8dg_uncached']  = (r['fp8dg_cached'] + t_recon + t_post) if r['fp8dg_cached']==r['fp8dg_cached'] else float('nan')
        res[(name,M)] = r
        del xh, xb, out
    res[(name,'meta')] = {'recon':t_recon,'post':t_post,'cnt':cnt,'b12x':b12x_ok}
    del trellis,suh,svh,W,pk,sc,fp4w,q_nt,ws,q_c,ws_c
    torch.cuda.empty_cache()

PATHS = ['exl3_gemm','b12x','fp4','fp8dg_cached','fp8dg_fused_est','fp8dg_uncached']
for M in MS:
    print(f"\n===== M={M} (ms per call) =====")
    print(f"{'shape':12} {'cnt':>4} " + " ".join(f"{p:>16}" for p in PATHS) + "   best")
    for name,K,N,bits,cnt in SHAPES:
        r = res[(name,M)]
        cells = " ".join(f"{r[p]:16.3f}" for p in PATHS)
        best = min((v,p) for p,v in r.items() if v==v)[1]
        print(f"{name:12} {cnt:4} {cells}   {best}")
    print(f"{'-- chunk total (ms), weighted by count --':60}")
    for p in PATHS:
        tot = sum(res[(n,M)][p]*c for n,_,_,_,c in SHAPES if res[(n,M)][p]==res[(n,M)][p])
        miss = [n for n,_,_,_,_ in SHAPES if res[(n,M)][p]!=res[(n,M)][p]]
        pp = M/(tot/1000) if tot>0 else 0
        print(f"  {p:18} {tot:9.1f} ms   implied {pp:8.0f} tok/s" + (f"   [missing {miss}]" if miss else ""))
    # oracle: best path per shape
    tot = sum(min(v for v in res[(n,M)].values() if v==v)*c for n,_,_,_,c in SHAPES)
    print(f"  {'ORACLE(per-shape)':18} {tot:9.1f} ms   implied {M/(tot/1000):8.0f} tok/s")
print("\nrecon/post per shape (M-independent):")
for name,K,N,bits,cnt in SHAPES:
    m = res[(name,'meta')]
    print(f"  {name:12} recon={m['recon']:7.3f} post={m['post']:7.3f} cnt={m['cnt']:4} b12x={m['b12x']}")
