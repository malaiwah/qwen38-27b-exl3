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
def big_image(px_w, px_h, label):
    """A large synthetic image. The 8 MP OOM from early phases was never fixed;
    with vllm-project/vllm#52871 unpatched, a forward-pass OOM kills the whole
    EngineCore - so the gate must know whether a big image is (a) served,
    (b) refused cleanly, or (c) fatal. Only (c) is a FAIL: a clean 4xx while
    the engine stays alive is an acceptable, documented behaviour."""
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (px_w, px_h))
        # cheap non-constant content so vision encoding is not degenerate
        for x in range(0, px_w, max(1, px_w // 64)):
            for y in range(px_h):
                img.putpixel((x, y), (x % 256, y % 256, 128))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = requests.post(f"{bl.BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": bl.MODEL, "max_tokens": 8, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url":
                       f"data:image/png;base64,{b64}"}},
                      {"type": "text", "text": "Describe briefly."}]}]},
            timeout=600)
        if r.status_code == 200:
            return "ok", 0
        # clean refusal is acceptable IF the engine survived
        return (f"refused-{r.status_code}" if alive() else f"FATAL-{r.status_code}"), 0
    except Exception as e:
        return ("refused-" + type(e).__name__) if alive() else ("FATAL-" + type(e).__name__), 0
steps=[("prefill 2k",  lambda: prompt(205)),
       ("prefill 6k",  lambda: prompt(600)),
       ("prefill 12k", lambda: prompt(1200)),
       ("prefill 24k", lambda: prompt(2400)),
       ("vision",      lambda: ("ok" if bl.vision_check() else "FAIL", 0)),
       ("decode 500",  lambda: ("ok" if bl.measure_tg(bl.TG_ESSAY_PROMPT,500,reps=1)['tok_s_median']>0 else "FAIL", bl.measure_tg(bl.TG_ESSAY_PROMPT,200,reps=1)['tok_s_median'])),
       ("prefill 12k#2",lambda: prompt(1200)),
       ("vision #2",   lambda: ("ok" if bl.vision_check() else "FAIL", 0)),
       ("image 2MP",   lambda: big_image(1920, 1080, "2MP")),
       ("image 8MP",   lambda: big_image(3840, 2160, "8MP")),
       ("vision #3",   lambda: ("ok" if bl.vision_check() else "FAIL", 0)),
       ("prefill 2k#2", lambda: prompt(205))]
ok=True
for label, fn in steps:
    try: st, rate = fn()
    except Exception as e: st, rate = "EXC:"+type(e).__name__, 0
    a=alive()
    print(f"    {label:15} -> {st:12} {rate:8.0f} tok/s  alive={a}")
    if st.startswith("FATAL") or st in ("FAIL",) or st.startswith("EXC") or st.startswith("HTTP") or not a:
        ok = False; break
print("    STRESS VERDICT:", "ROBUST" if ok else "FRAGILE")
