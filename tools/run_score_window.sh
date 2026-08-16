#!/bin/bash
# Measure our own numbers on the external protocol's scoring geometry.
#
# llama-perplexity scores only the second half of each window (`first = n_ctx/2`
# in perplexity.cpp), so every position it scores carries at least n_ctx/2 tokens
# of left context. Our suite scores every position of a 2,048-token context,
# including the high-entropy opening ones. This script recaptures one shard and
# replays it three ways -- full context, second half only, and the external
# absolute 256-token left-context floor -- so the difference is a measured offset
# instead of an argument.
#
#   tools/run_score_window.sh              # shard 0, windows 0 / 1024 / 256
#   WINDOWS="0 1024" tools/run_score_window.sh 2
set -euo pipefail

R=/home/mbelleau/research
W=${GG_WORK:-/var/tmp/work}
GGRUN=$R/tools/ggrun.sh
SHARD=${1:-0}
TAG=$(printf 'shard-%04d' "$SHARD")
SDIR=$W/kld5/shards/$TAG
HDIR=$W/kld5/hidden/$TAG
ODIR=$W/kld5-win/reports/$TAG
HARNESS=$W/kld5-win/tools/fidelity.py
WINDOWS=${WINDOWS:-"0 1024 256"}
NAMES="bf16 k4 k5k6 hyd ctx fp8"
CANDIDATES="k4 k5k6 hyd ctx fp8"

[ -d "$SDIR/suite" ] || { echo "missing shard suite view: $SDIR/suite" >&2; exit 1; }
mkdir -p "$ODIR" "$(dirname "$HARNESS")"

# Captures come from the frozen per-shard scripts the published run used, so the
# hidden states are produced by the identical command line. Only the replay side
# uses the newer harness, because only replay gained the window.
cp "$R/tools/fidelity.py" "$HARNESS"
echo "replay harness $(sha256sum "$HARNESS" | cut -d' ' -f1)"

for name in $NAMES; do
  if [ -f "$HDIR/hidden-$name/capture-manifest.json" ] &&
     "${PYTHON:-python3}" - "$HDIR/hidden-$name/capture-manifest.json" <<'PY'
import json, sys
print("complete" if json.load(open(sys.argv[1])).get("complete") else "incomplete")
raise SystemExit(0 if json.load(open(sys.argv[1])).get("complete") else 1)
PY
  then
    echo "-- capture $name present"
    continue
  fi
  echo "-- capture $name"
  "$GGRUN" bash "/work/kld5/shards/$TAG/cap-$name.sh"
done

for window in $WINDOWS; do
  for cand in $CANDIDATES; do
    out=$ODIR/report-$cand-from$window.json
    [ -f "$out" ] && { echo "-- replay $cand window $window present"; continue; }
    echo "-- replay $cand window $window"
    "$GGRUN" env VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext \
      python /work/kld5-win/tools/fidelity.py replay \
        --reference "/work/kld5/hidden/$TAG/hidden-bf16" \
        --candidate "/work/kld5/hidden/$TAG/hidden-$cand" \
        --head /work/kld2/lm_head.safetensors \
        --suite "/work/kld5/shards/$TAG/suite" \
        --score-from "$window" \
        --out "/work/kld5-win/reports/$TAG/report-$cand-from$window.json"
  done
done

echo "-- releasing hidden states"
rm -rf "$HDIR"
echo SCORE_WINDOW_DONE
