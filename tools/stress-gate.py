# Mixed stress gate: long prefills + vision + decode interleaved, which is
# what actually kills the engine (pure text ladders pass where mixed fails).
import sys, time, base64, requests
sys.path.insert(0,'/home/mbelleau/qwen38-27b-exl3/tools')
import bench_lib as bl
S="The quick brown fox jumps over the lazy dog. "
def alive():
    try: return requests.get(f"{bl.BASE_URL}/health",timeout=5).status_code==200
    except Exception: return False
def prompt(mult):
    try:
        a=time.time()
        r=requests.post(f"{bl.BASE_URL}/v1/completions",headers={"Content-Type":"application/json"},
          json={"model":bl.MODEL,"prompt":S*mult,"max_tokens":1,"temperature":0},timeout=900)
        b=time.time()
        if r.status_code!=200: return f"HTTP{r.status_code}", 0
        return "ok", r.json()["usage"]["prompt_tokens"]/(b-a)
    except Exception as e: return "EXC:"+type(e).__name__, 0
steps=[("prefill 2k",  lambda: prompt(205)),
       ("prefill 6k",  lambda: prompt(600)),
       ("prefill 12k", lambda: prompt(1200)),
       ("prefill 24k", lambda: prompt(2400)),
       ("vision",      lambda: ("ok" if bl.vision_check() else "FAIL", 0)),
       ("decode 500",  lambda: ("ok" if bl.measure_tg(bl.TG_ESSAY_PROMPT,500,reps=1)['tok_s_median']>0 else "FAIL", bl.measure_tg(bl.TG_ESSAY_PROMPT,200,reps=1)['tok_s_median'])),
       ("prefill 12k#2",lambda: prompt(1200)),
       ("vision #2",   lambda: ("ok" if bl.vision_check() else "FAIL", 0)),
       ("prefill 2k#2", lambda: prompt(205))]
ok=True
for label, fn in steps:
    try: st, rate = fn()
    except Exception as e: st, rate = "EXC:"+type(e).__name__, 0
    a=alive()
    print(f"    {label:15} -> {st:12} {rate:8.0f} tok/s  alive={a}")
    if st!="ok" or not a: ok=False; break
print("    STRESS VERDICT:", "ROBUST" if ok else "FRAGILE")
