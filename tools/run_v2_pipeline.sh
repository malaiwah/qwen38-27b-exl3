#!/bin/bash
# Post-conversion pipeline for the iteration-2 checkpoint (gate K5 / up K5 / down K6).
set -uo pipefail
die() { echo "FAILED: $*" >&2; exit 1; }
F=/work/kld2/fidelity.py; S=/work/kld3/suite; H=/work/kld2/lm_head.safetensors; D=/work/kld3
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext

echo "== splice BF16 attention (MTP stays quantized to hold the memory budget)"
# --force is deliberate here: the pipeline owns /work/Qwen3.8-27B-K5K6 and the splicer
# clears its shards, index and receipt first, so a rerun cannot leave a previous build behind.
SPLICE="python /work/splice_bf16_attn.py -q /work/qwen38-v2 -s /models/Qwen3.8-27B -o /work/Qwen3.8-27B-K5K6 --force"
$SPLICE || die splice
cd /work/exllamav3
python util/add_safetensors_index.py -m /work/Qwen3.8-27B-K5K6 --force || die index
python util/add_quant_config.py -m /work/Qwen3.8-27B-K5K6 || die quantconfig

echo "== finalize: repair tensor_storage, verify the tensor set against upstream, emit manifest + receipts"
python /work/kld3/finalize_checkpoint.py -m /work/Qwen3.8-27B-K5K6 --upstream /models/Qwen3.8-27B \
  --recipe "attn BF16->online K6, mlp gate/up K5 + down K6, lm_head K6, mtp quantized" \
  --command "$SPLICE" || die finalize

echo "== capture v2 candidate on the held-out analysis partition"
VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
python "$F" capture --model /work/Qwen3.8-27B-K5K6 --suite "$S" --out "$D/hidden-v2" \
  --quantization exl3 --quantization-config "$CFG" --gpu-memory-utilization 0.85 || die capture

echo "== replay + paired"
python "$F" replay --reference "$D/hidden-bf16" --candidate "$D/hidden-v2" --head "$H" \
  --suite "$S" --filter analysis --out "$D/report-v2-analysis.json" || die replay
python "$F" paired --a "$D/report-v2-analysis.json" --b "$D/report-k4-analysis.json" \
  --a-label v2-k5k6 --b-label v1-k4 --out "$D/paired-v2-vs-v1.json" || die paired1
python "$F" paired --a "$D/report-v2-analysis.json" --b "$D/report-fp8-analysis.json" \
  --a-label v2-k5k6 --b-label fp8 --out "$D/paired-v2-vs-fp8.json" || die paired2
python - <<'PY'
import json
r=json.load(open("/work/kld3/report-v2-analysis.json")); b=r["context_bootstrap"]
print(f"v2 mean={r['token_mean_kld']:.6f} CI=[{b['ci95_low']:.5f},{b['ci95_high']:.5f}] "
      f"median={r['token_median_kld']:.6f} top1={r['top1_agreement']*100:.2f}%")
for n in ("paired-v2-vs-v1","paired-v2-vs-fp8"):
    p=json.load(open(f"/work/kld3/{n}.json"))
    print(f"{n}: diff={p['difference_a_minus_b']:+.6f} CI=[{p['bootstrap_difference']['ci95_low']:+.5f},"
          f"{p['bootstrap_difference']['ci95_high']:+.5f}] wins={p['a_wins']}/{p['contexts']}")
PY
