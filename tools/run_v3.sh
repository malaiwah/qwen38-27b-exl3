#!/bin/bash
# v3 measurement sweep on the held-out suite.
# Artifact-checked rather than exit-code-checked: vLLM sometimes aborts during
# interpreter teardown after writing its results.
set -uo pipefail
die() { echo "FAILED: $*" >&2; exit 1; }
F=/work/kld2/fidelity.py
S=/work/kld3/suite
H=/work/kld2/lm_head.safetensors
D=/work/kld3
K4_CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]}'
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext

cap() {  # name model [extra args...]
  local name=$1 model=$2; shift 2
  if [ -f "$D/hidden-$name/capture-manifest.json" ]; then echo "== $name captured already"; return; fi
  echo "== capture $name"
  python "$F" capture --model "$model" --suite "$S" --out "$D/hidden-$name" "$@" || die "capture $name"
}

cap bf16   /models/Qwen3.8-27B                --gpu-memory-utilization 0.9
cap k4     /work/Qwen3.8-27B-K4               --quantization exl3 --quantization-config "$K4_CFG" --gpu-memory-utilization 0.85
cap nvfp4  /models/Qwen3.8-27B-NVFP4          --quantization compressed-tensors --gpu-memory-utilization 0.85
cap fp8    /models/Qwen3.8-27B-FP8            --gpu-memory-utilization 0.9

# runtime-repeat sentinels: same runtime, same contexts, two more captures.
for r in r2 r3; do
  if [ ! -f "$D/hidden-k4-$r/capture-manifest.json" ]; then
    echo "== sentinel repeat $r"
    VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
    python "$F" capture --model /work/Qwen3.8-27B-K4 --suite "$S" --out "$D/hidden-k4-$r" \
      --quantization exl3 --quantization-config "$K4_CFG" \
      --gpu-memory-utilization 0.85 --filter sentinel
  fi
done

# analysis-partition metrics for every candidate
for c in k4 nvfp4 fp8; do
  echo "== replay $c (analysis)"
  python "$F" replay --reference "$D/hidden-bf16" --candidate "$D/hidden-$c" \
    --head "$H" --suite "$S" --filter analysis --out "$D/report-$c-analysis.json"
done

# paired comparisons
python "$F" paired --a "$D/report-k4-analysis.json" --b "$D/report-nvfp4-analysis.json" \
  --a-label k4 --b-label nvfp4 --out "$D/paired-k4-vs-nvfp4.json"
python "$F" paired --a "$D/report-k4-analysis.json" --b "$D/report-fp8-analysis.json" \
  --a-label k4 --b-label fp8 --out "$D/paired-k4-vs-fp8.json"

# noise floor: two captures of the SAME runtime over the sentinel contexts
echo "== sentinel noise floor"
python "$F" replay --reference "$D/hidden-k4" --candidate "$D/hidden-k4-r2" \
  --head "$H" --suite "$S" --filter sentinel --out "$D/noise-r1-vs-r2.json"
python "$F" replay --reference "$D/hidden-k4-r2" --candidate "$D/hidden-k4-r3" \
  --head "$H" --suite "$S" --filter sentinel --out "$D/noise-r2-vs-r3.json"

# replay qualification: live full-vocabulary logits vs replayed logits
echo "== replay qualification"
python "$F" qualify --model /models/Qwen3.8-27B --suite "$S" --hidden "$D/hidden-bf16" \
  --head "$H" --contexts 6 --out "$D/qualification-bf16.json"

echo "== summary"
for f in "$D"/report-*-analysis.json "$D"/noise-*.json; do
  python - "$f" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
b=r["context_bootstrap"]
print(f"{sys.argv[1].split('/')[-1]:28s} ctx={r['contexts']:3d} mean={r['token_mean_kld']:.6f} "
      f"CI=[{b['ci95_low']:.5f},{b['ci95_high']:.5f}] median={r['token_median_kld']:.6f} "
      f"top1={r['top1_agreement']*100:.2f}%")
PY
done
