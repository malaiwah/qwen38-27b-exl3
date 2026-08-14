#!/bin/bash
# Full KLD sweep: freeze the window, capture the BF16 teacher, score candidates.
# Each step is checked by its artifact, not by exit status: vLLM's engine
# sometimes aborts during interpreter teardown ("PyEval_SaveThread: the function
# must be called with the GIL held") *after* writing its results, which would
# otherwise kill the sweep with exit 134.
set -uo pipefail
K=/work/kld
OUT=/work/kld/runs
mkdir -p "$OUT"

need() { [ -f "$1" ] || { echo "MISSING $1"; return 1; }; }
step() { echo "=== $* ==="; }

if [ ! -f "$K/tokens.json" ]; then
  step tokens
  python "$K/qwen38_kld.py" tokens --model /models/Qwen3.8-27B --out "$K/tokens.json" --context 2048
fi

if [ ! -f "$OUT/ref-bf16/manifest.json" ]; then
  step "capture BF16 teacher"
  python "$K/qwen38_kld.py" capture --model /models/Qwen3.8-27B --tokens "$K/tokens.json" \
    --out "$OUT/ref-bf16" --kv-cache-dtype auto --gpu-memory-utilization 0.92
fi

step "score k4-online-k6"
VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
python "$K/qwen38_kld.py" score --model /work/Qwen3.8-27B-K4 --tokens "$K/tokens.json" \
  --ref "$OUT/ref-bf16" --out "$OUT/k4-online-k6" --label k4-online-k6 \
  --quantization exl3 \
  --quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}' \
  --kv-cache-dtype auto --gpu-memory-utilization 0.85 --repeats 3

step "score unsloth-nvfp4"
python "$K/qwen38_kld.py" score --model /models/Qwen3.8-27B-NVFP4 --tokens "$K/tokens.json" \
  --ref "$OUT/ref-bf16" --out "$OUT/unsloth-nvfp4" --label unsloth-nvfp4 \
  --quantization compressed-tensors --kv-cache-dtype auto \
  --gpu-memory-utilization 0.85 --repeats 3

step "score k4-no-overlay (attention stays BF16 at runtime)"
python "$K/qwen38_kld.py" score --model /work/Qwen3.8-27B-K4 --tokens "$K/tokens.json" \
  --ref "$OUT/ref-bf16" --out "$OUT/k4-bf16-attn" --label k4-bf16-attn \
  --quantization exl3 --kv-cache-dtype auto \
  --gpu-memory-utilization 0.85 --repeats 3

step done
for d in "$OUT"/*/summary.json; do
  python - "$d" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
print(f"{s['label']:16s} mean_kld={s['mean_kld']:.6f} run_sd={s['run_sd']:.6f}")
PY
done
