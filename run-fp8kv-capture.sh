#!/bin/bash
set -uo pipefail
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
TOOL=$LAUNCHER/tools/research/wave5/candidate_capture.py
BASE=/home/mbelleau/.cache/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf
ACTION=/home/mbelleau/worktrees/r30-capture-smoke/final-evidence-20260821/action.json
PAYLOAD=/home/mbelleau/worktrees/r30-capture-smoke/final-evidence-20260821/changed-payload.safetensors
CAND=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-candidate
EVID=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence
CAP=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-capture
SUITE=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/suite
REF=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/reference/hidden-bf16
HEAD=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/lm-head/weight.safetensors
SPLIT=/home/mbelleau/worktrees/r30-capture-smoke/r29/split-manifest.json
DATA=/home/mbelleau/worktrees/r30-capture-smoke/r29/data-manifest.json
R31=/home/mbelleau/worktrees/r30-capture-smoke/r31-source
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
VLLM=/opt/venv/lib/python3.12/site-packages/vllm

rm -rf "$CAND" "$EVID" "$CAP" 2>/dev/null

systemctl --user stop qwen38-27b.service 2>/dev/null
echo "service stopped"

# Materialize checkpoint
python3 "$TOOL" materialize \
  --base-checkpoint "$BASE" \
  --action "$ACTION" \
  --payload "$PAYLOAD" \
  --candidate-checkpoint "$CAND" \
  --evidence-dir "$EVID"
if [ $? -ne 0 ]; then
  echo "MATERIALIZE FAILED"
  systemctl --user start qwen38-27b.service
  exit 1
fi
echo "materialize done"

# Build suite view
FIDELITY=$LAUNCHER/tools/fidelity.py
SUITE_VIEW=$EVID/capture-suite-view
mkdir -p "$CAP" "$SUITE_VIEW"

python3 -c "
import sys, os, json
sys.path.insert(0, '$LAUNCHER/tools/research/wave5')
from pathlib import Path
suite_dir = Path('$SUITE/shard-0000')
suite_manifest = json.load(open(suite_dir / 'suite-manifest.json'))
dst = Path('$SUITE_VIEW')
dst.mkdir(parents=True, exist_ok=True)
os.link(suite_dir / 'suite-manifest.json', dst / 'suite-manifest.json')
for ctx in suite_manifest['context_index']:
    rel = Path(ctx['file'])
    src = suite_dir / rel
    if not src.is_file():
        src = Path('$SUITE') / rel
    dst_file = dst / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst_file)
print(f'suite view ready: {len(suite_manifest[\"context_index\"])} contexts')
"

# Run fidelity.py capture with fp8_e4m3 KV cache
podman run --rm --name "wave5-fp8kv-capture-$$" \
  --device nvidia.com/gpu=all --ipc=host --network none \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_EXL3_MULTIPRECISION=0 \
  --env VLLM_EXL3_EMBED_ONLINE_BITS=6 \
  --env VLLM_USE_V2_MODEL_RUNNER=1 \
  --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  --env TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  --env PYTHONPATH=/opt/r30-ext \
  --env VLLM_EXL3_PRE_RECONSTRUCT=0 \
  --env VLLM_EXL3_B12X_ANY_BITS=1 \
  --env VLLM_EXL3_B12X_MIN_M=128 \
  --env VLLM_EXL3_PREFILL_RECONSTRUCT_CACHE=0 \
  --env LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  --volume "$LAUNCHER:$LAUNCHER:ro" \
  --volume "$CAND:$CAND:ro" \
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro" \
  --volume "$CAP:$CAP:rw" \
  --volume /home/mbelleau/.cache/jit:/cache/jit:rw \
  --volume /home/mbelleau/vllm-exl3-multiprecision.py:$VLLM/model_executor/layers/quantization/exl3.py:ro \
  --volume /home/mbelleau/scheduler_patch.py:$VLLM/v1/core/sched/scheduler.py:ro \
  --volume /home/mbelleau/qwen_gdn_linear_attn_patch.py:$VLLM/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro \
  --volume /home/mbelleau/spec_decode_utils_patch.py:$VLLM/v1/worker/gpu/spec_decode/utils.py:ro \
  --volume /home/mbelleau/autoregressive_speculator_patch.py:$VLLM/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro \
  --volume /home/mbelleau/qwen3_5_mtp_patch.py:$VLLM/model_executor/models/qwen3_5_mtp.py:ro \
  --volume /home/mbelleau/vllm-exl3-linear-ba.py:$VLLM/model_executor/layers/linear.py:ro \
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp4_conversion.py:/opt/fp4/exl3_fp4_conversion.py:ro \
  --volume /home/mbelleau/qwen38-27b-exl3/patches/triton_fp4_quant.py:/opt/fp4/triton_fp4_quant.py:ro \
  --volume /home/mbelleau/qwen38-27b-exl3/patches/exl3_fp6_conversion.py:/opt/fp6/exl3_fp6_conversion.py:ro \
  --volume /home/mbelleau/worktrees/r30-torch-ext/exllamav3_ext/exllamav3_ext.so:/opt/r30-ext/exllamav3_ext.so:ro \
  --entrypoint python3 "$IMAGE" "$FIDELITY" capture \
  --model "$CAND" --suite "$SUITE_VIEW" --out "$CAP" \
  --quantization auto --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.90 \
  --filter all --max-batched-tokens 2048 --hash-shards
CAPTURE_EXIT=$?
echo "capture exit: $CAPTURE_EXIT"

# Restore service
systemctl --user start qwen38-27b.service
echo "service restored"

exit $CAPTURE_EXIT
