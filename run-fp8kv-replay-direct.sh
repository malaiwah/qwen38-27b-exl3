#!/bin/bash
set -uo pipefail
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
FIDELITY=$LAUNCHER/tools/fidelity.py
CAP=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-capture
REF=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/reference/hidden-bf16
HEAD=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/lm-head/weight.safetensors
SUITE_VIEW=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence/capture-suite-view
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
REPORT=$CAP/fp8kv-replay-report.json

systemctl --user stop qwen38-27b.service
sleep 3
echo "service stopped"

podman run --rm --name "fp8kv-replay-$$" \
  --device nvidia.com/gpu=all --ipc=host --network none \
  --volume "$LAUNCHER:$LAUNCHER:ro" \
  --volume "$REF:$REF:ro" \
  --volume "$CAP:$CAP:rw" \
  --volume "$HEAD:$HEAD:ro" \
  --volume "$SUITE_VIEW:$SUITE_VIEW:ro" \
  --volume /home/mbelleau/.cache/jit:/cache/jit:rw \
  --entrypoint python3 "$IMAGE" "$FIDELITY" replay \
  --reference "$REF" --candidate "$CAP" \
  --head "$HEAD" --suite "$SUITE_VIEW" \
  --out "$REPORT" --filter all
REPLAY_EXIT=$?
echo "replay exit: $REPLAY_EXIT"

systemctl --user start qwen38-27b.service
echo "service restored"

if [ -f "$REPORT" ]; then
  echo "=== FP8 KV cache KLD ==="
  python3 -c "
import json
r = json.load(open('$REPORT'))
for k in ('mean_kld','token_mean_kld','context_macro_mean_kld','p50_kld','p95_kld','p99_kld','p999_kld','max_kld','top1_agreement'):
    if k in r: print(f'  {k}: {r[k]}')
"
fi

exit $REPLAY_EXIT
