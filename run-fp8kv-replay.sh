#!/bin/bash
set -uo pipefail
LAUNCHER=/home/mbelleau/worktrees/r30-capture-smoke/r30-launcher
TOOL=$LAUNCHER/tools/research/wave5/candidate_capture.py
EVID=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-evidence
CAP=/home/mbelleau/worktrees/r30-capture-smoke/fp8kv-capture
SUITE=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/suite
REF=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/reference/hidden-bf16
HEAD=/home/mbelleau/worktrees/r30-capture-smoke/canonical-mirror/lm-head/weight.safetensors
SPLIT=/home/mbelleau/worktrees/r30-capture-smoke/r29/split-manifest.json
DATA=/home/mbelleau/worktrees/r30-capture-smoke/r29/data-manifest.json
R31=/home/mbelleau/worktrees/r30-capture-smoke/r31-source
IMAGE="docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

systemctl --user stop qwen38-27b.service
sleep 3
echo "service stopped"

python3 "$TOOL" replay \
  --evidence-dir "$EVID" \
  --suite-root "$SUITE" \
  --reference-root "$REF" \
  --shared-head "$HEAD" \
  --r29-split-manifest "$SPLIT" \
  --r29-data-manifest "$DATA" \
  --r31-root "$R31" \
  --container-image "$IMAGE" \
  --capture-dir "$CAP" \
  --candidate-id fp8kv-strength-zero \
  2>&1 | tee /tmp/fp8kv-replay.log
REPLAY_EXIT=$?

systemctl --user start qwen38-27b.service
echo "service restored"

exit $REPLAY_EXIT
