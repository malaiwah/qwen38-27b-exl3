#!/bin/bash
# The frozen, source-disjoint qualification. Run ONCE; whatever it says gets published, and no
# recipe changes because of it. Future development uses the v4 analysis partition only.
#
# Nested quoting broke the first attempt (the quantization JSON lost its quotes crossing two
# shells, and the FP8 model rejects an explicit --quantization), so each capture is written to
# its own command file and executed as a file.
set -uo pipefail
W=/var/tmp/work
D=/work/kld4
printf '%s' '{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}' > $W/kld4/qcfg.json

mkcap() {  # name model envline extra...
  local NAME=$1 MODEL=$2 ENVS=$3; shift 3
  cat > $W/kld4/cap-$NAME.sh <<EOF
set -uo pipefail
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext $ENVS
exec python /work/kld2/fidelity.py capture --model $MODEL --suite $D/suite \\
  --out $D/hidden-$NAME --gpu-memory-utilization 0.85 $*
EOF
}

mkcap bf16 /models/Qwen3.8-27B ''
mkcap ctx  /work/qwen38-ctx 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' \
  '--quantization exl3 --quantization-config "$(cat /work/kld4/qcfg.json)"'
mkcap hyd  /work/qwen38-hyd 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' \
  '--quantization exl3 --quantization-config "$(cat /work/kld4/qcfg.json)"'
mkcap k5k6 /work/Qwen3.8-27B-K5K6 \
  'VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' \
  '--quantization exl3 --quantization-config "$(cat /work/kld4/qcfg.json)"'
mkcap fp8  /models/Qwen3.8-27B-FP8 ''

for NAME in bf16 ctx hyd k5k6 fp8; do
  if [ -d /var/tmp/work/kld4/hidden-$NAME ]; then echo "== $NAME already captured"; continue; fi
  echo "== capturing $NAME"
  $W/ggrun.sh bash /work/kld4/cap-$NAME.sh || echo "CAPTURE FAILED: $NAME"
done

cat > $W/kld4/replay.sh <<'EOF'
set -uo pipefail
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext
F=/work/kld2/fidelity.py; D=/work/kld4; S=$D/suite; H=/work/kld2/lm_head.safetensors
for c in ctx hyd k5k6 fp8; do
  python $F replay --reference $D/hidden-bf16 --candidate $D/hidden-$c --head $H \
    --suite $S --filter qualification --out $D/report-v4-$c-qual.json || exit 1
  python $F replay --reference $D/hidden-bf16 --candidate $D/hidden-$c --head $H \
    --suite $S --out $D/report-v4-$c-all.json || exit 1
done
for c in ctx hyd k5k6; do
  python $F paired --a $D/report-v4-$c-qual.json --b $D/report-v4-fp8-qual.json \
    --a-label $c --b-label official-fp8 --out $D/paired-v4-$c-vs-fp8.json || exit 1
done
python - <<'PY'
import json
D = "/work/kld4"
print(f"{'candidate':10s} {'partition':13s} {'mean':>9} {'CI95':>24} {'top1':>7}")
for c in ("ctx", "hyd", "k5k6", "fp8"):
    for part in ("qual", "all"):
        r = json.load(open(f"{D}/report-v4-{c}-{part}.json")); b = r["context_bootstrap"]
        print(f"{c:10s} {part:13s} {r['token_mean_kld']:9.6f} "
              f"[{b['ci95_low']:9.6f},{b['ci95_high']:9.6f}] {r['top1_agreement']*100:6.2f}%")
for c in ("ctx", "hyd", "k5k6"):
    q = json.load(open(f"{D}/paired-v4-{c}-vs-fp8.json"))
    print(f"{c} - FP8 (qualification): diff={q['difference_a_minus_b']:+.6f} "
          f"CI=[{q['bootstrap_difference']['ci95_low']:+.6f},"
          f"{q['bootstrap_difference']['ci95_high']:+.6f}] wins={q['a_wins']}/{q['contexts']}")
PY
EOF
$W/ggrun.sh bash /work/kld4/replay.sh || echo "REPLAY STAGE FAILED"
echo V4QUALDONE
