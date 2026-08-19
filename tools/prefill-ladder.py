import requests, time, sys
u="http://localhost:8000/v1/completions"; h={"Content-Type":"application/json"}
S="The quick brown fox jumps over the lazy dog. "
def alive():
    try: return requests.get("http://localhost:8000/health",timeout=5).status_code==200
    except Exception: return False
def go(mult):
    try:
        a=time.time(); r=requests.post(u,headers=h,json={"model":"Qwen3.8-27B","prompt":S*mult,"max_tokens":1,"temperature":0},timeout=900); b=time.time()
        if r.status_code!=200: return (f"HTTP{r.status_code}",None,b-a)
        return ("ok",r.json()["usage"]["prompt_tokens"],b-a)
    except Exception as e: return ("EXC:"+type(e).__name__,None,0.0)
ok=True
for mult,label in [(205,"bench2k"),(600,"6k"),(1200,"12k"),(2400,"24k"),(205,"bench2k-again"),(1200,"12k-again")]:
    st,pt,dt=go(mult)
    print(f"    {label:14} ~{mult*10:6} tok -> {st:10} tok={str(pt):7} {dt*1000:8.1f} ms ({(pt/dt if pt else 0):7.0f} tok/s) alive={alive()}")
    if st!="ok": ok=False; break
print("    VERDICT:", "ROBUST" if ok else "FRAGILE")
