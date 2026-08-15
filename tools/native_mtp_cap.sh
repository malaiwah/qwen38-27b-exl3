#!/bin/bash
# Test whether an explicit image ceiling recovers enough activation memory for native
# 262,144-token context and MTP-3 under the same 30.24 GiB engine budget.
set -euo pipefail
W=${W:-/var/tmp/work}
R=${R:-/home/mbelleau/research}
PY=${PY:-/home/mbelleau/qwen38-27b/.venv/bin/python}
GGRUN=${GGRUN:-$R/tools/ggrun.sh}
ROOTFS=${GG_ROOTFS:-/var/tmp/gg-rootfs}
ROOT=$ROOTFS/opt/venv/lib/python3.12/site-packages/vllm
IMAGE='voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b'
ROOTFS_PROVENANCE=${ROOTFS_PROVENANCE:-$W/gg-rootfs-provenance.json}
MODEL=${MODEL:-/work/qwen38-ctx}
PORT=${PORT:-8220}
TAG=${TAG:-4mp}
MAX_PIXELS=${MAX_PIXELS:-4194304}
GPU_UTIL=${GPU_UTIL:-0.3184}
RUN_LONG=${RUN_LONG:-1}
RUN_VISION=${RUN_VISION:-1}
RUN_BIG=${RUN_BIG:-1}
RUN_BENCH=${RUN_BENCH:-1}
LONG_MM_IMAGE=${LONG_MM_IMAGE:-}
BIG_REQUEST=${BIG_REQUEST:-$W/vis_big.json}
BIG_EXPECTED=${BIG_EXPECTED:-red, blue}
LOG=$W/native-mtp-cap-$TAG-server.log
LONG=$W/kld3/native-mtp-cap-$TAG-needle.json
VISION=$W/kld3/native-mtp-cap-$TAG-vision.json
BIG=$W/kld3/native-mtp-cap-$TAG-large-image.json
BENCH=$W/kld3/native-mtp-cap-$TAG-bench.log
LONGMM=$W/kld3/native-mtp-cap-$TAG-longmm.json
PROVENANCE=$W/kld3/native-mtp-cap-$TAG-provenance.json
CFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
COMPILE_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}'
SERVER_PID=

[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid TAG" >&2; exit 2; }
[[ $PORT =~ ^[1-9][0-9]{0,4}$ ]] && (( PORT <= 65535 )) ||
  { echo "PORT must be in 1..65535" >&2; exit 2; }
[[ $MAX_PIXELS =~ ^[1-9][0-9]*$ ]] && (( MAX_PIXELS <= 100000000 )) ||
  { echo "MAX_PIXELS must be in 1..100000000" >&2; exit 2; }
[[ $GPU_UTIL =~ ^(0\.[0-9]+|1(\.0+)?)$ ]] && [[ ! $GPU_UTIL =~ ^0\.0+$ ]] ||
  { echo "GPU_UTIL must be in (0, 1]" >&2; exit 2; }
for flag in "$RUN_LONG" "$RUN_VISION" "$RUN_BIG" "$RUN_BENCH"; do
  [[ $flag == 0 || $flag == 1 ]] || { echo "RUN_* flags must be 0 or 1" >&2; exit 2; }
done
MM_CONFIG=$(printf '{"truncation":false,"max_pixels":%d}' "$MAX_PIXELS")
mkdir -p "$W/kld3"
mapfile -t ROOTFS_FACTS < <(
  "$PY" "$R/tools/pull_rootfs.py" verify "$ROOTFS_PROVENANCE" "$IMAGE" "$ROOTFS"
)
[[ ${#ROOTFS_FACTS[@]} == 2 ]] || {
  echo "verified extracted-rootfs provenance is required" >&2
  exit 1
}
ROOTFS_MANIFEST=${ROOTFS_FACTS[0]}
ROOTFS_MANIFEST_SHA=${ROOTFS_FACTS[1]}

EXL3_SHA=2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee
QWEN_SHA=04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7
MTP_SHA=0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab
printf '%s  %s\n' \
  "$EXL3_SHA" "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$QWEN_SHA" "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$MTP_SHA" "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" |
  sha256sum --quiet -c -
cp "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$ROOT/model_executor/layers/quantization/exl3.py"
cp "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5.py"
cp "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5_mtp.py"
find "$ROOT" -path '*/__pycache__/*.pyc' \
  \( -name 'exl3*.pyc' -o -name 'qwen3_5*.pyc' \) -delete
cmp "$R/tools/vllm-exl3-prefill-dispatch.py" \
  "$ROOT/model_executor/layers/quantization/exl3.py"
cmp "$R/tools/vllm-qwen3_5-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5.py"
cmp "$R/tools/vllm-qwen3_5_mtp-embed-quant-config.py" \
  "$ROOT/model_executor/models/qwen3_5_mtp.py"

stop_server() {
  if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap stop_server EXIT INT TERM

rm -f "$LOG" "$LONG" "$VISION" "$BIG" "$BENCH" "$LONGMM" "$PROVENANCE"
"$PY" -c 'import hashlib,json,os,pathlib,sys
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  while chunk:=f.read(8<<20): h.update(chunk)
 return h.hexdigest()
out,image,rootfs_provenance,manifest,manifest_sha,ggrun,exl3,qwen,mtp=sys.argv[1:]
payload={"schema":"qwen38-native-runtime-provenance/1","image":image,
 "rootfs_provenance":rootfs_provenance,
 "rootfs_provenance_sha256":sha(rootfs_provenance),
 "rootfs_file_manifest":manifest,"rootfs_file_manifest_sha256":manifest_sha,
 "ggrun_sha256":sha(ggrun),
 "installed_patch_sha256":{"exl3.py":exl3,"qwen3_5.py":qwen,"qwen3_5_mtp.py":mtp},
 "identity_verified":True}
tmp=out+".tmp"; pathlib.Path(tmp).write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,out)' \
  "$PROVENANCE" "$IMAGE" "$ROOTFS_PROVENANCE" "$ROOTFS_MANIFEST" \
  "$ROOTFS_MANIFEST_SHA" "$GGRUN" "$EXL3_SHA" "$QWEN_SHA" "$MTP_SHA"
GG_ENV='VLLM_EXL3_GRAPH_DECODE=1 VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 VLLM_EXL3_EMBED_BITS=8' \
  setsid "$GGRUN" vllm serve "$MODEL" --served-model-name m --quantization exl3 \
  --quantization-config "$CFG" --max-model-len 262144 --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --speculative-config "$SPEC_CONFIG" --mm-processor-kwargs "$MM_CONFIG" \
  --compilation-config "$COMPILE_CONFIG" --port "$PORT" --host 127.0.0.1 \
  >"$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 420); do
  if grep -qE "Uvicorn running on https?://127\\.0\\.0\\.1:$PORT" "$LOG" &&
     kill -0 "$SERVER_PID" 2>/dev/null &&
     curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" |
       "$PY" -c 'import json,sys; p=json.load(sys.stdin); assert any(x.get("id") == "m" for x in p["data"])'
  then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null ||
     grep -qE 'EngineCore failed|ValidationError|unrecognized arguments|ValueError|RuntimeError|Traceback' "$LOG"; then
    grep -E 'EngineCore failed|ValidationError|unrecognized arguments|ValueError|RuntimeError|Traceback|GiB KV cache is needed|Available KV cache memory' "$LOG" | tail -12
    exit 1
  fi
  sleep 5
done
grep -qE "Uvicorn running on https?://127\\.0\\.0\\.1:$PORT" "$LOG"
kill -0 "$SERVER_PID" 2>/dev/null
grep -qE 'Desired GPU memory utilization is .*30[.]24 GiB' "$LOG" || {
  echo "server did not enforce the required 30.24 GiB engine budget" >&2
  exit 1
}
grep -oE 'Desired GPU memory utilization is .{0,60}|GPU KV cache size: [0-9,]+ tokens|Actual usage is .{0,160}|\([0-9.]+ GiB KV cache is needed[^)]*\)' "$LOG" | tail -5

if [[ $RUN_LONG == 1 ]]; then
  "$PY" "$R/tools/longctx.py" --url "http://127.0.0.1:$PORT/v1" \
    --corpus "$W/kld3/corpus/literary" --tokens 261800 --depth 0.5 \
    --require-pass --out "$LONG"
fi
if [[ $RUN_VISION == 1 ]]; then
  "$PY" "$R/tools/vision_eval.py" --url "http://127.0.0.1:$PORT/v1" \
    --label "native-mtp-cap-$TAG" --cases 10 --out "$VISION"
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); o=r["overall"]; assert o["n"] == 30 and o["correct"] >= 24, o' "$VISION"
fi
if [[ $RUN_BIG == 1 ]]; then
  status=$(curl -sS -o "$BIG" -w '%{http_code}' -H 'Content-Type: application/json' \
    --data-binary @"$BIG_REQUEST" \
    "http://127.0.0.1:$PORT/v1/chat/completions")
  [[ $status == 200 ]]
  "$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); a=p["choices"][0]["message"]["content"].strip(); assert a == sys.argv[2], (a, sys.argv[2]); print(a)' "$BIG" "$BIG_EXPECTED"
fi
if [[ -n $LONG_MM_IMAGE ]]; then
  "$PY" "$R/tools/longmm.py" --url "http://127.0.0.1:$PORT/v1" \
    --corpus "$W/kld3/corpus/literary" --image "$LONG_MM_IMAGE" \
    --require-pass --out "$LONGMM"
fi
if [[ $RUN_BENCH == 1 ]]; then
  "$PY" "$R/tools/bench.py" --url "http://127.0.0.1:$PORT/v1/completions" \
    --tokens 256 --concurrency 1 --prefill-tokens --label "native-mtp-cap-$TAG" |
    tee "$BENCH"
fi
echo NATIVE_MTP_CAP_DONE
