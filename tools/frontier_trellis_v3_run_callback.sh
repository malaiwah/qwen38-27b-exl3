#!/bin/bash
# Run actual EXL3 RTN/LDLQ controls over the frozen matched-flow fixtures.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || {
  echo "frontier_trellis_v3_run_callback: transaction is not active" >&2
  exit 2
}
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE \
  FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || {
    echo "frontier_trellis_v3_run_callback: missing ${name}" >&2
    exit 2
  }
done

SOURCE=/home/mbelleau/final-frontier-g0/converter-source
EXT_CACHE_SOURCE=/home/mbelleau/final-frontier-g0/qkv-cache-a4
TOOL=/home/mbelleau/final-frontier-g02/tools/trellis_v3.py
COMMON=/home/mbelleau/final-frontier-g02/tools/frontier_common.py
PLAN=/home/mbelleau/final-frontier-g02/v3-run-plan.json
EVIDENCE=/home/mbelleau/final-frontier-g02/v3-evidence
for path in "$SOURCE" "$EXT_CACHE_SOURCE" "$EVIDENCE"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "missing immutable v3 control directory ${path}" >&2
    exit 2
  }
done
for path in "$TOOL" "$COMMON" "$PLAN"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || {
    echo "missing immutable v3 control input ${path}" >&2
    exit 2
  }
done
[[ ! -e "$FRONTIER_CAMPAIGN_WORK_DIR/run-result.json" && ! -e "$FRONTIER_CAMPAIGN_WORK_DIR/run-result.safetensors" ]] || {
  echo "v3 control output already exists" >&2
  exit 2
}
cp -a "$EXT_CACHE_SOURCE/." "$FRONTIER_CAMPAIGN_CACHE_DIR/"

podman run --rm --replace \
  --name "$FRONTIER_CAMPAIGN_CONTAINER" \
  --network none --ipc=host --device nvidia.com/gpu=all \
  --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$SOURCE:/src:ro" \
  -v "$TOOL:/opt/frontier/trellis_v3.py:ro" \
  -v "$COMMON:/opt/frontier/frontier_common.py:ro" \
  -v "$PLAN:/opt/frontier/run-plan.json:ro" \
  -v "$EVIDENCE:/evidence:ro" \
  -v "$FRONTIER_CAMPAIGN_CACHE_DIR:/cache:rw" \
  -v "$FRONTIER_CAMPAIGN_WORK_DIR:/work:rw" \
  --entrypoint /bin/bash \
  "$FRONTIER_CAMPAIGN_IMAGE" \
  -lc 'set -euo pipefail
    ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/
    export PYTHONPATH=/cache/exllamav3_ext:/opt/frontier:/src
    exec /opt/venv/bin/python /opt/frontier/trellis_v3.py \
      --plan /opt/frontier/run-plan.json --out /work/run-result.json' \
  >"$FRONTIER_CAMPAIGN_WORK_DIR/run.log" 2>&1
