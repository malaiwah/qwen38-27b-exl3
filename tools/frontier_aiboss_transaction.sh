#!/bin/bash
# Watchdog-bounded, fail-closed maintenance transaction for the AIBoss owner service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_SELF="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${BASH_SOURCE[0]}")"
SNAPSHOT_TOOL="${SCRIPT_DIR}/frontier_live_snapshot.py"
SCHEMA="qwen38-frontier-aiboss-transaction/1"
LOG_SCHEMA="qwen38-frontier-aiboss-transaction-log/1"

STATE_MACHINE=(
  "01 validate-explicit-inputs"
  "02 verify-snapshot"
  "03 verify-live-source-identities"
  "04 reject-live-writable-paths"
  "05 verify-campaign-artifacts-empty-and-distinct"
  "06 arm-external-restore-watchdog"
  "07 stop-owner-unit"
  "08 prove-owner-inactive"
  "09 prove-live-container-absent"
  "10 prove-gpu-compute-empty"
  "11 execute-single-bounded-callback"
  "12 remove-campaign-container-only"
  "13 restore-exact-owner-unit"
  "14 prove-active-running-healthy"
  "15 prove-model-revision-and-length"
  "16 prove-inspect-and-source-hash-equality"
  "17 prove-text-sentinel"
  "18 prove-vision-sentinel"
  "19 prove-second-request-and-post-health"
  "20 prove-gpu-power-clocks"
  "21 prove-throughput-sentinel"
  "22 disarm-watchdog"
  "23 publish-durable-receipt-and-log"
)

usage() {
  cat <<'EOF'
Usage:
  frontier_aiboss_transaction.sh [--dry-run] \
    --snapshot FILE --unit UNIT --unit-file FILE --launcher FILE \
    --live-container NAME --live-model-root DIR --api-base URL --gpu-uuid UUID \
    --campaign-container NAME --campaign-image NAME@sha256:DIGEST_OR_sha256:IMAGE_ID \
    --campaign-model-root DIR --campaign-model-revision REV --campaign-profile NAME \
    --campaign-cache-dir EMPTY_DIR --campaign-work-dir EMPTY_DIR \
    --callback EXECUTABLE [--callback-arg ARG ...] \
    --callback-timeout SEC --watchdog-timeout SEC \
    --vision-image FILE --text-prompt TEXT --text-expect TEXT \
    --vision-prompt TEXT --vision-expect TEXT \
    --second-prompt TEXT --second-expect TEXT \
    --throughput-prompt TEXT --throughput-tokens N --min-throughput-tps TPS \
    --gpu-memory-tolerance-mib MIB --receipt FILE --log FILE

--dry-run validates argument shape and prints the exact state machine. It does not
read or change systemd, Podman, GPU, endpoint, or output state.

The callback is invoked exactly once with FRONTIER_TRANSACTION_ACTIVE=1 and the
six explicit FRONTIER_CAMPAIGN_* identities in its process environment. PROFILE
is never placed in the systemd manager environment.
EOF
}

# The watchdog entry point is intentionally private. It can only remove the exact
# campaign container named by the parent, validate immutable unit/launcher hashes,
# and start the exact owner unit. It never prunes images, volumes, caches, or paths.
if [[ "${1:-}" == "--watchdog-restore" ]]; then
  shift
  [[ $# -eq 11 ]] || exit 125
  WD_SNAPSHOT="$1"; WD_UNIT="$2"; WD_UNIT_FILE="$3"; WD_LAUNCHER="$4"
  WD_CAMPAIGN_CONTAINER="$5"; WD_LIVE_CONTAINER="$6"; WD_API_BASE="$7"
  WD_LIVE_MODEL_ROOT="$8"; WD_LOG="$9"; WD_TRANSACTION_ID="${10}"; WD_SNAPSHOT_TOOL="${11}"
  python3 - "$WD_SNAPSHOT" "$WD_UNIT" "$WD_UNIT_FILE" "$WD_LAUNCHER" "$WD_SNAPSHOT_TOOL" <<'PY'
import hashlib, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[5]).resolve().parent))
from frontier_common import load_strict_json, sha256_file
from frontier_live_snapshot import validate_snapshot
snapshot, _ = validate_snapshot(load_strict_json(Path(sys.argv[1])))
service = snapshot.get("service")
if not isinstance(service, dict) or service.get("unit") != sys.argv[2]:
    raise SystemExit(125)
if sha256_file(Path(sys.argv[3])) != service.get("unit_file_sha256"):
    raise SystemExit(125)
if sha256_file(Path(sys.argv[4])) != service.get("launcher_sha256"):
    raise SystemExit(125)
unit_cat = subprocess.run(
    ["systemctl", "--user", "cat", sys.argv[2]], check=False, capture_output=True
)
if unit_cat.returncode != 0 or hashlib.sha256(unit_cat.stdout).hexdigest() != service.get("unit_cat_sha256"):
    raise SystemExit(125)
PY
  podman rm -f "$WD_CAMPAIGN_CONTAINER" >/dev/null 2>&1 || true
  systemctl --user start "$WD_UNIT"
  for _ in $(seq 1 720); do
    if systemctl --user is-active --quiet "$WD_UNIT" && \
       [[ "$(podman inspect -f '{{.State.Running}}' "$WD_LIVE_CONTAINER" 2>/dev/null || true)" == "true" ]] && \
       [[ "$(podman inspect -f '{{.State.Health.Status}}' "$WD_LIVE_CONTAINER" 2>/dev/null || true)" == "healthy" ]] && \
       curl --silent --show-error --fail --max-time 10 "${WD_API_BASE%/}/health" >/dev/null; then
      WD_VERIFY="/run/user/$(id -u)/${WD_TRANSACTION_ID}-watchdog-verify.json"
      [[ ! -e "$WD_VERIFY" && ! -L "$WD_VERIFY" ]] || exit 125
      "$WD_SNAPSHOT_TOOL" verify-live --input "$WD_SNAPSHOT" --output "$WD_VERIFY" \
        --unit "$WD_UNIT" --unit-file "$WD_UNIT_FILE" --launcher "$WD_LAUNCHER" \
        --container "$WD_LIVE_CONTAINER" --model-root "$WD_LIVE_MODEL_ROOT"
      rm -f "$WD_VERIFY"
      exit 0
    fi
    sleep 5
  done
  exit 125
fi

DRY_RUN=0
SNAPSHOT="" UNIT="" UNIT_FILE="" LAUNCHER="" LIVE_CONTAINER="" LIVE_MODEL_ROOT=""
API_BASE="" GPU_UUID=""
CAMPAIGN_CONTAINER="" CAMPAIGN_IMAGE="" CAMPAIGN_MODEL_ROOT="" CAMPAIGN_MODEL_REVISION=""
CAMPAIGN_PROFILE="" CAMPAIGN_CACHE_DIR="" CAMPAIGN_WORK_DIR="" CALLBACK=""
CALLBACK_TIMEOUT="" WATCHDOG_TIMEOUT="" VISION_IMAGE=""
TEXT_PROMPT="" TEXT_EXPECT="" VISION_PROMPT="" VISION_EXPECT=""
SECOND_PROMPT="" SECOND_EXPECT="" THROUGHPUT_PROMPT="" THROUGHPUT_TOKENS=""
MIN_THROUGHPUT_TPS="" GPU_MEMORY_TOLERANCE_MIB="" RECEIPT="" LOG=""
declare -a CALLBACK_ARGS=()

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "frontier_aiboss_transaction: missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --snapshot|--unit|--unit-file|--launcher|--live-container|--live-model-root|--api-base|--gpu-uuid|\
    --campaign-container|--campaign-image|--campaign-model-root|--campaign-model-revision|--campaign-profile|\
    --campaign-cache-dir|--campaign-work-dir|--callback|--callback-timeout|--watchdog-timeout|--vision-image|\
    --text-prompt|--text-expect|--vision-prompt|--vision-expect|--second-prompt|--second-expect|\
    --throughput-prompt|--throughput-tokens|--min-throughput-tps|--gpu-memory-tolerance-mib|--receipt|--log)
      need_value "$@"
      key="${1#--}"; key="${key//-/_}"; printf -v "${key^^}" '%s' "$2"; shift 2 ;;
    --callback-arg) need_value "$@"; CALLBACK_ARGS+=("$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "frontier_aiboss_transaction: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

required=(SNAPSHOT UNIT UNIT_FILE LAUNCHER LIVE_CONTAINER LIVE_MODEL_ROOT API_BASE GPU_UUID
  CAMPAIGN_CONTAINER CAMPAIGN_IMAGE CAMPAIGN_MODEL_ROOT CAMPAIGN_MODEL_REVISION CAMPAIGN_PROFILE
  CAMPAIGN_CACHE_DIR CAMPAIGN_WORK_DIR CALLBACK CALLBACK_TIMEOUT WATCHDOG_TIMEOUT VISION_IMAGE
  TEXT_PROMPT TEXT_EXPECT VISION_PROMPT VISION_EXPECT SECOND_PROMPT SECOND_EXPECT THROUGHPUT_PROMPT
  THROUGHPUT_TOKENS MIN_THROUGHPUT_TPS GPU_MEMORY_TOLERANCE_MIB RECEIPT LOG)
for name in "${required[@]}"; do
  [[ -n "${!name}" ]] || { echo "frontier_aiboss_transaction: --${name,,} is required" | tr '_' '-' >&2; exit 2; }
done
[[ "$UNIT" =~ ^[A-Za-z0-9_.@:-]+\.service$ ]] || { echo "invalid --unit" >&2; exit 2; }
[[ "$LIVE_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid --live-container" >&2; exit 2; }
[[ "$CAMPAIGN_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid --campaign-container" >&2; exit 2; }
[[ "$CAMPAIGN_CONTAINER" != "$LIVE_CONTAINER" ]] || { echo "campaign container must differ from live container" >&2; exit 2; }
[[ "$CAMPAIGN_IMAGE" =~ ^([^[:space:]]+@)?sha256:[0-9a-f]{64}$ ]] || { echo "campaign image must be a registry digest reference or immutable local image ID" >&2; exit 2; }
[[ "$CAMPAIGN_MODEL_REVISION" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] || { echo "invalid campaign model revision" >&2; exit 2; }
[[ "$CAMPAIGN_PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid campaign profile" >&2; exit 2; }
for value in CALLBACK_TIMEOUT WATCHDOG_TIMEOUT THROUGHPUT_TOKENS GPU_MEMORY_TOLERANCE_MIB; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "$value must be a positive integer" >&2; exit 2; }
done
python3 - "$MIN_THROUGHPUT_TPS" <<'PY'
import math, sys
try: value = float(sys.argv[1])
except ValueError: raise SystemExit(2)
if not math.isfinite(value) or value <= 0: raise SystemExit(2)
PY
[[ "$WATCHDOG_TIMEOUT" -ge $((CALLBACK_TIMEOUT + 900)) ]] || { echo "watchdog timeout must be at least callback timeout + 900 seconds" >&2; exit 2; }

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' "${STATE_MACHINE[@]}"
  exit 0
fi

# All path and runtime checks below are real-mode only.
[[ -x "$SNAPSHOT_TOOL" && -f "$SNAPSHOT_TOOL" ]] || { echo "snapshot tool is missing or not executable" >&2; exit 2; }
for path in "$SNAPSHOT" "$UNIT_FILE" "$LAUNCHER" "$VISION_IMAGE" "$CALLBACK"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || { echo "required nonempty regular non-symlink file is missing: $path" >&2; exit 2; }
done
[[ -x "$CALLBACK" ]] || { echo "callback is not executable" >&2; exit 2; }
for path in "$LIVE_MODEL_ROOT" "$CAMPAIGN_MODEL_ROOT" "$CAMPAIGN_CACHE_DIR" "$CAMPAIGN_WORK_DIR"; do
  [[ -d "$path" && ! -L "$path" ]] || { echo "required real directory is missing: $path" >&2; exit 2; }
done
python3 - "$CAMPAIGN_MODEL_ROOT" <<'PY'
import os, sys
with os.scandir(sys.argv[1]) as entries:
    if next(entries, None) is None:
        raise SystemExit(2)
PY
for path in "$CAMPAIGN_CACHE_DIR" "$CAMPAIGN_WORK_DIR"; do
  if ! python3 - "$path" <<'PY'
import os, sys
with os.scandir(sys.argv[1]) as entries:
    if next(entries, None) is not None:
        raise SystemExit(2)
PY
  then
    echo "campaign cache/work directory is nonempty: $path" >&2
    exit 2
  fi
done
for path in "$RECEIPT" "$LOG"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "output already exists: $path" >&2; exit 2; }
  [[ -d "$(dirname "$path")" ]] || { echo "output parent is missing: $(dirname "$path")" >&2; exit 2; }
done
[[ "$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$RECEIPT")" != "$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$LOG")" ]] || { echo "receipt and log paths alias" >&2; exit 2; }

TRANSACTION_ID="frontier-$(date -u +%Y%m%dT%H%M%SZ)-$$"
TMP_DIR="$(mktemp -d "$(dirname "$LOG")/.frontier-transaction.XXXXXX")"
VERIFY_JSON="$TMP_DIR/snapshot-verify.json"
PRELIVE_JSON="$TMP_DIR/pre-live-verify.json"
POSTLIVE_JSON="$TMP_DIR/post-live-verify.json"
WATCHDOG_UNIT="${TRANSACTION_ID}.timer"
WATCHDOG_SERVICE="${TRANSACTION_ID}.service"
CALLBACK_RC=125
RESTORE_PROVEN=0
WATCHDOG_ARMED=0
FINALIZED=0
CURRENT_STATE="01 validate-explicit-inputs"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Every update is a same-directory atomic, fsync-backed strict JSON replacement.
python3 - "$LOG" "$TRANSACTION_ID" "$STARTED_UTC" "$LOG_SCHEMA" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
from frontier_common import atomic_write_json, canonical_sha256
path, txid, utc, schema = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
body = {"schema": schema, "transaction_id": txid, "started_utc": utc, "events": []}
atomic_write_json(path, {**body, "integrity": {"canonical_sha256": canonical_sha256(body)}})
PY

log_event() {
  local state="$1" status="$2" detail="$3"
  python3 - "$LOG" "$state" "$status" "$detail" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[6])
from frontier_common import atomic_write_json, canonical_sha256, load_strict_json
path = Path(sys.argv[1]); doc = load_strict_json(path)
body = {k: v for k, v in doc.items() if k != "integrity"}
events = body.get("events")
if not isinstance(events, list): raise SystemExit(125)
events.append({"state": sys.argv[2], "status": sys.argv[3], "detail": sys.argv[4], "utc": sys.argv[5]})
atomic_write_json(path, {**body, "integrity": {"canonical_sha256": canonical_sha256(body)}})
PY
}

json_get() {
  python3 - "$SNAPSHOT" "$1" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from frontier_common import load_strict_json
value = load_strict_json(Path(sys.argv[1]))
for part in sys.argv[2].split('.'):
    if not isinstance(value, dict) or part not in value: raise SystemExit(125)
    value = value[part]
if isinstance(value, bool): print("true" if value else "false")
elif isinstance(value, (str, int, float)): print(value)
else: raise SystemExit(125)
PY
}

publish_receipt() {
  local status="$1" reason="$2"
  [[ "$FINALIZED" -eq 0 ]] || return 0
  FINALIZED=1
  local snapshot_digest callback_sha log_digest finished
  snapshot_digest="$(python3 - "$SNAPSHOT" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from frontier_common import sha256_file
print(sha256_file(Path(sys.argv[1])))
PY
)"
  callback_sha="$(python3 - "$CALLBACK" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from frontier_common import sha256_file
print(sha256_file(Path(sys.argv[1])))
PY
)"
  log_digest="$(python3 - "$LOG" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from frontier_common import load_strict_json
print(load_strict_json(Path(sys.argv[1]))["integrity"]["canonical_sha256"])
PY
)"
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$RECEIPT" "$SCHEMA" "$TRANSACTION_ID" "$STARTED_UTC" "$finished" "$status" "$reason" \
    "$snapshot_digest" "$callback_sha" "$log_digest" "$CAMPAIGN_CONTAINER" "$CAMPAIGN_IMAGE" \
    "$CAMPAIGN_MODEL_REVISION" "$CAMPAIGN_PROFILE" "$CALLBACK_RC" "$RESTORE_PROVEN" "$SCRIPT_DIR" \
    "$CAMPAIGN_MODEL_ROOT" "$CAMPAIGN_CACHE_DIR" "$CAMPAIGN_WORK_DIR" <<'PY'
import hashlib, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[17])
from frontier_common import atomic_write_json, canonical_sha256
binding = lambda p: hashlib.sha256(os.fsencode(os.path.realpath(p))).hexdigest()
body = {
 "schema": sys.argv[2], "transaction_id": sys.argv[3], "started_utc": sys.argv[4], "finished_utc": sys.argv[5],
 "status": sys.argv[6], "reason": sys.argv[7], "snapshot_file_sha256": sys.argv[8],
 "callback": {"sha256": sys.argv[9], "exit_code": int(sys.argv[15])},
 "campaign": {"container": sys.argv[11], "image": sys.argv[12], "model_revision": sys.argv[13],
              "profile": sys.argv[14], "model_binding_sha256": binding(sys.argv[18]),
              "cache_binding_sha256": binding(sys.argv[19]), "work_binding_sha256": binding(sys.argv[20])},
 "restore_proven": sys.argv[16] == "1", "log_canonical_sha256": sys.argv[10],
}
atomic_write_json(Path(sys.argv[1]), {**body, "integrity": {"canonical_sha256": canonical_sha256(body)}})
PY
}

remove_campaign_container() {
  podman rm -f "$CAMPAIGN_CONTAINER" >/dev/null 2>&1 || true
  ! podman container exists "$CAMPAIGN_CONTAINER"
}

wait_restore_runtime() {
  local deadline=$((SECONDS + 3600))
  while (( SECONDS < deadline )); do
    if systemctl --user is-active --quiet "$UNIT" && \
       [[ "$(podman inspect -f '{{.State.Running}}' "$LIVE_CONTAINER" 2>/dev/null || true)" == "true" ]] && \
       [[ "$(podman inspect -f '{{.State.Health.Status}}' "$LIVE_CONTAINER" 2>/dev/null || true)" == "healthy" ]] && \
       curl --silent --show-error --fail --max-time 10 "${API_BASE%/}/health" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

restore_owner() {
  remove_campaign_container || true
  systemctl --user start "$UNIT" || return 1
  wait_restore_runtime || return 1
  return 0
}

on_exit() {
  local original_rc=$?
  trap - EXIT INT TERM HUP
  if [[ "$RESTORE_PROVEN" -ne 1 ]]; then
    restore_owner || true
  fi
  if [[ "$WATCHDOG_ARMED" -eq 1 ]]; then
    systemctl --user stop "$WATCHDOG_UNIT" "$WATCHDOG_SERVICE" >/dev/null 2>&1 || true
    systemctl --user reset-failed "$WATCHDOG_UNIT" "$WATCHDOG_SERVICE" >/dev/null 2>&1 || true
  fi
  if [[ "$FINALIZED" -eq 0 ]]; then
    log_event "$CURRENT_STATE" "fail" "transaction exited before full restore proof (rc=${original_rc})" || true
    publish_receipt "fail" "transaction exited before full restore proof at ${CURRENT_STATE}" || true
  fi
  rm -rf "$TMP_DIR"
  exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

CURRENT_STATE="01 validate-explicit-inputs"
# Campaign model identity must be evidenced by either its snapshot-directory name
# or a config _commit_hash; no inferred or guessed revision is accepted.
python3 - "$CAMPAIGN_MODEL_ROOT" "$CAMPAIGN_MODEL_REVISION" <<'PY'
import json, os, sys
from pathlib import Path
root, expected = Path(sys.argv[1]).resolve(), sys.argv[2]
proved = root.name == expected
config = root / "config.json"
if config.is_file():
    with config.open(encoding="utf-8") as f: doc = json.load(f)
    if isinstance(doc, dict) and doc.get("_commit_hash") == expected: proved = True
if not proved: raise SystemExit("campaign model revision is not evidenced by path or config _commit_hash")
PY
podman image exists "$CAMPAIGN_IMAGE" || { echo "campaign image is not local" >&2; exit 2; }
! podman container exists "$CAMPAIGN_CONTAINER" || { echo "campaign container already exists" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "explicit immutable inputs accepted"

CURRENT_STATE="02 verify-snapshot"
"$SNAPSHOT_TOOL" verify --input "$SNAPSHOT" --output "$VERIFY_JSON"
[[ "$(json_get service.unit)" == "$UNIT" ]] || { echo "snapshot unit mismatch" >&2; exit 2; }
[[ "$(json_get container.name)" == "$LIVE_CONTAINER" ]] || { echo "snapshot container mismatch" >&2; exit 2; }
[[ "$(json_get gpu.uuid)" == "$GPU_UUID" ]] || { echo "snapshot GPU mismatch" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "strict snapshot and canonical digest verified"

CURRENT_STATE="03 verify-live-source-identities"
systemctl --user is-active --quiet "$UNIT" || { echo "owner unit is not active" >&2; exit 2; }
"$SNAPSHOT_TOOL" verify-live --input "$SNAPSHOT" --output "$PRELIVE_JSON" --unit "$UNIT" \
  --unit-file "$UNIT_FILE" --launcher "$LAUNCHER" --container "$LIVE_CONTAINER" --model-root "$LIVE_MODEL_ROOT"
log_event "$CURRENT_STATE" pass "unit, launcher, overlays, model artifacts, and inspect identity match"

CURRENT_STATE="04 reject-live-writable-paths"
python3 - "$LIVE_CONTAINER" "$CAMPAIGN_MODEL_ROOT" "$CAMPAIGN_CACHE_DIR" "$CAMPAIGN_WORK_DIR" <<'PY'
import json, os, subprocess, sys
campaign = [os.path.realpath(path) for path in sys.argv[2:]]
result = subprocess.run(["podman", "inspect", sys.argv[1]], check=False, capture_output=True)
if result.returncode != 0:
    raise SystemExit(2)
doc = json.loads(result.stdout)
if not isinstance(doc, list) or len(doc) != 1: raise SystemExit(2)
live = []
for mount in doc[0].get("Mounts", []):
    if mount.get("RW") is True:
        source = mount.get("Source")
        if not isinstance(source, str) or not source: raise SystemExit(2)
        live.append(os.path.realpath(source))
for candidate in campaign:
    for source in live:
        try: overlap = os.path.commonpath((candidate, source)) in (candidate, source)
        except ValueError: overlap = False
        if overlap: raise SystemExit("campaign path overlaps live writable mount")
PY
log_event "$CURRENT_STATE" pass "campaign model/cache/work do not equal, contain, or descend from live writable mounts"

CURRENT_STATE="05 verify-campaign-artifacts-empty-and-distinct"
python3 - "$CAMPAIGN_MODEL_ROOT" "$CAMPAIGN_CACHE_DIR" "$CAMPAIGN_WORK_DIR" "$LIVE_MODEL_ROOT" <<'PY'
import os, sys
paths = [os.path.realpath(path) for path in sys.argv[1:]]
if len(set(paths)) != len(paths): raise SystemExit("campaign paths alias each other or the live model")
for left_index, left in enumerate(paths[:3]):
    for right in paths[left_index + 1:3]:
        if os.path.commonpath((left, right)) in (left, right): raise SystemExit("campaign paths overlap")
PY
log_event "$CURRENT_STATE" pass "campaign image/model/profile/container and empty cache/work are distinct"

CURRENT_STATE="06 arm-external-restore-watchdog"
systemd-run --user --unit="$TRANSACTION_ID" --on-active="${WATCHDOG_TIMEOUT}s" --collect \
  /bin/bash "$SCRIPT_SELF" --watchdog-restore "$SNAPSHOT" "$UNIT" "$UNIT_FILE" "$LAUNCHER" \
  "$CAMPAIGN_CONTAINER" "$LIVE_CONTAINER" "$API_BASE" "$LIVE_MODEL_ROOT" "$LOG" \
  "$TRANSACTION_ID" "$SNAPSHOT_TOOL" >/dev/null
WATCHDOG_ARMED=1
log_event "$CURRENT_STATE" pass "independent user-systemd restore watchdog armed"

CURRENT_STATE="07 stop-owner-unit"
systemctl --user stop "$UNIT"
log_event "$CURRENT_STATE" pass "exact owner unit stop completed"

CURRENT_STATE="08 prove-owner-inactive"
OWNER_INACTIVE=0
for _ in {1..120}; do
  if [[ "$(systemctl --user is-active "$UNIT" 2>/dev/null || true)" == "inactive" ]]; then
    OWNER_INACTIVE=1
    break
  fi
  sleep 1
done
[[ "$OWNER_INACTIVE" -eq 1 ]] || { echo "owner unit did not become inactive within 120 seconds" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "owner unit is inactive"

CURRENT_STATE="09 prove-live-container-absent"
LIVE_ABSENT=0
for _ in {1..120}; do
  if ! podman container exists "$LIVE_CONTAINER"; then
    LIVE_ABSENT=1
    break
  fi
  sleep 1
done
[[ "$LIVE_ABSENT" -eq 1 ]] || { echo "live container remains present after 120 seconds" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "live container is absent"

CURRENT_STATE="10 prove-gpu-compute-empty"
GPU_EMPTY=0
for _ in {1..120}; do
  if [[ -z "$(nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; then
    GPU_EMPTY=1
    break
  fi
  sleep 1
done
[[ "$GPU_EMPTY" -eq 1 ]] || { echo "GPU still has compute applications after 120 seconds" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "GPU compute-app list is empty"

CURRENT_STATE="11 execute-single-bounded-callback"
set +e
timeout --signal=TERM --kill-after=30 "$CALLBACK_TIMEOUT" env \
  FRONTIER_TRANSACTION_ACTIVE=1 \
  FRONTIER_CAMPAIGN_CONTAINER="$CAMPAIGN_CONTAINER" \
  FRONTIER_CAMPAIGN_IMAGE="$CAMPAIGN_IMAGE" \
  FRONTIER_CAMPAIGN_MODEL_ROOT="$CAMPAIGN_MODEL_ROOT" \
  FRONTIER_CAMPAIGN_MODEL_REVISION="$CAMPAIGN_MODEL_REVISION" \
  FRONTIER_CAMPAIGN_PROFILE="$CAMPAIGN_PROFILE" \
  FRONTIER_CAMPAIGN_CACHE_DIR="$CAMPAIGN_CACHE_DIR" \
  FRONTIER_CAMPAIGN_WORK_DIR="$CAMPAIGN_WORK_DIR" \
  "$CALLBACK" "${CALLBACK_ARGS[@]}"
CALLBACK_RC=$?
set -e
if [[ "$CALLBACK_RC" -eq 0 ]]; then
  log_event "$CURRENT_STATE" pass "single callback exited zero"
else
  log_event "$CURRENT_STATE" fail "single callback exited ${CALLBACK_RC}; restore proof still required"
fi

CURRENT_STATE="12 remove-campaign-container-only"
remove_campaign_container || { echo "campaign container could not be removed" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "removed exact campaign container; no image, volume, cache, work, or global prune performed"

CURRENT_STATE="13 restore-exact-owner-unit"
systemctl --user start "$UNIT"
log_event "$CURRENT_STATE" pass "exact snapshot owner unit start requested without manager environment injection"

CURRENT_STATE="14 prove-active-running-healthy"
wait_restore_runtime || { echo "restored service did not become active/running/healthy" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "unit active, container running and healthy, /health HTTP 200"

CURRENT_STATE="15 prove-model-revision-and-length"
EXPECTED_REVISION="$(json_get model.revision)"
EXPECTED_MAX_LEN="$(json_get model.max_model_len)"
SERVED_MODEL="$(json_get model.served.id)"
python3 - "$API_BASE" "$EXPECTED_REVISION" "$EXPECTED_MAX_LEN" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip('/') + '/v1/models', timeout=30) as response:
    if response.status != 200: raise SystemExit(2)
    doc = json.load(response)
entries = doc.get('data') if isinstance(doc, dict) else None
matches = 0
for item in entries if isinstance(entries, list) else []:
    revision = item.get('revision') or item.get('model_revision') or item.get('root')
    length = item.get('max_model_len') or item.get('max_model_length')
    matches += isinstance(revision, str) and sys.argv[2] in revision and length == int(sys.argv[3])
if matches != 1: raise SystemExit(2)
PY
log_event "$CURRENT_STATE" pass "served model revision and max-model-len match snapshot"

CURRENT_STATE="16 prove-inspect-and-source-hash-equality"
"$SNAPSHOT_TOOL" verify-live --input "$SNAPSHOT" --output "$POSTLIVE_JSON" --unit "$UNIT" \
  --unit-file "$UNIT_FILE" --launcher "$LAUNCHER" --container "$LIVE_CONTAINER" --model-root "$LIVE_MODEL_ROOT"
log_event "$CURRENT_STATE" pass "restored inspect, unit, launcher, overlays, and model hashes equal snapshot"

sentinel_request() {
  local kind="$1" prompt="$2" expect="$3" output="$4"
  python3 - "$kind" "$API_BASE" "$SERVED_MODEL" "$prompt" "$expect" "$VISION_IMAGE" "$output" "$SCRIPT_DIR" <<'PY'
import base64, json, mimetypes, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, sys.argv[8])
from frontier_common import atomic_write_json, canonical_sha256
kind, base, model, prompt, expected, image_path, output = sys.argv[1:8]
if kind == 'vision':
    mime = mimetypes.guess_type(image_path)[0]
    if not mime or not mime.startswith('image/'): raise SystemExit(2)
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    content = [{'type':'image_url','image_url':{'url':f'data:{mime};base64,{encoded}'}},{'type':'text','text':prompt}]
else:
    content = prompt
payload = {
    'model': model,
    'messages': [{'role':'user','content':content}],
    'temperature': 0,
    'max_tokens': 64,
    'chat_template_kwargs': {'enable_thinking': False},
}
data = json.dumps(payload, separators=(',',':')).encode()
request = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions', data=data, method='POST', headers={'Content-Type':'application/json'})
t0 = time.monotonic()
with urllib.request.urlopen(request, timeout=300) as response:
    if response.status != 200: raise SystemExit(2)
    doc = json.load(response)
elapsed = time.monotonic() - t0
try: text = doc['choices'][0]['message']['content']
except (KeyError, IndexError, TypeError): raise SystemExit(2)
if not isinstance(text, str) or expected not in text: raise SystemExit(2)
body = {'kind':kind,'status':'pass','response_text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),'elapsed_seconds':elapsed}
atomic_write_json(Path(output), {**body,'integrity':{'canonical_sha256':canonical_sha256(body)}})
PY
}

proof_digest() {
  python3 - "$1" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from frontier_common import load_strict_json
doc = load_strict_json(Path(sys.argv[1]))
print(doc["integrity"]["canonical_sha256"])
PY
}

CURRENT_STATE="17 prove-text-sentinel"
sentinel_request text "$TEXT_PROMPT" "$TEXT_EXPECT" "$TMP_DIR/text.json"
log_event "$CURRENT_STATE" pass "deterministic text sentinel matched expected content; proof_sha256=$(proof_digest "$TMP_DIR/text.json")"

CURRENT_STATE="18 prove-vision-sentinel"
sentinel_request vision "$VISION_PROMPT" "$VISION_EXPECT" "$TMP_DIR/vision.json"
log_event "$CURRENT_STATE" pass "vision sentinel matched expected content; proof_sha256=$(proof_digest "$TMP_DIR/vision.json")"

CURRENT_STATE="19 prove-second-request-and-post-health"
sentinel_request text "$SECOND_PROMPT" "$SECOND_EXPECT" "$TMP_DIR/second.json"
curl --silent --show-error --fail --max-time 30 "${API_BASE%/}/health" >/dev/null
[[ "$(podman inspect -f '{{.State.Health.Status}}' "$LIVE_CONTAINER")" == "healthy" ]] || { echo "container unhealthy after second request" >&2; exit 2; }
log_event "$CURRENT_STATE" pass "independent second request passed and service remained healthy; proof_sha256=$(proof_digest "$TMP_DIR/second.json")"

CURRENT_STATE="20 prove-gpu-power-clocks"
SNAP_POWER="$(json_get gpu.power_limit_w)"
SNAP_COMPUTE="$(json_get gpu.compute_mode)"
SNAP_PERSIST="$(json_get gpu.persistence_mode)"
SNAP_MEM_USED="$(json_get gpu.memory_used_mib)"
GPU_ROW="$(nvidia-smi --id="$GPU_UUID" --query-gpu=uuid,memory.used,compute_mode,persistence_mode,power.limit,clocks.current.memory --format=csv,noheader,nounits)"
python3 - "$GPU_ROW" "$GPU_UUID" "$SNAP_MEM_USED" "$GPU_MEMORY_TOLERANCE_MIB" "$SNAP_COMPUTE" "$SNAP_PERSIST" "$SNAP_POWER" <<'PY'
import math, sys
parts=[part.strip() for part in sys.argv[1].split(',')]
if len(parts)!=6 or parts[0]!=sys.argv[2]: raise SystemExit(2)
used=int(float(parts[1])); expected_used=int(float(sys.argv[3])); tolerance=int(sys.argv[4])
if abs(used-expected_used)>tolerance: raise SystemExit(2)
if parts[2]!=sys.argv[5] or parts[3]!=sys.argv[6]: raise SystemExit(2)
if not math.isclose(float(parts[4]),float(sys.argv[7]),rel_tol=0,abs_tol=0.5): raise SystemExit(2)
if not math.isfinite(float(parts[5])) or float(parts[5]) <= 0: raise SystemExit(2)
PY
GPU_ROW_SHA256="$(printf '%s' "$GPU_ROW" | sha256sum | cut -d' ' -f1)"
log_event "$CURRENT_STATE" pass "GPU UUID, normal memory, compute/persistence modes, and power limit match snapshot; dynamic current memory clock recorded but not equality-gated; row_sha256=${GPU_ROW_SHA256}"

CURRENT_STATE="21 prove-throughput-sentinel"
THROUGHPUT_TPS="$(python3 - "$API_BASE" "$SERVED_MODEL" "$THROUGHPUT_PROMPT" "$THROUGHPUT_TOKENS" "$MIN_THROUGHPUT_TPS" <<'PY'
import json, math, sys, time, urllib.request
payload={'model':sys.argv[2],'prompt':sys.argv[3],'temperature':0,'max_tokens':int(sys.argv[4]),'ignore_eos':True}
request=urllib.request.Request(sys.argv[1].rstrip('/')+'/v1/completions',data=json.dumps(payload).encode(),method='POST',headers={'Content-Type':'application/json'})
t0=time.monotonic()
with urllib.request.urlopen(request,timeout=600) as response:
    if response.status!=200: raise SystemExit(2)
    doc=json.load(response)
elapsed=time.monotonic()-t0
usage=doc.get('usage') if isinstance(doc,dict) else None
tokens=usage.get('completion_tokens') if isinstance(usage,dict) else None
if not isinstance(tokens,int) or tokens<=0 or elapsed<=0: raise SystemExit(2)
tps=tokens/elapsed
if not math.isfinite(tps) or tps<float(sys.argv[5]): raise SystemExit(2)
print(f"{tps:.6f}")
PY
)"
log_event "$CURRENT_STATE" pass "measured completion throughput ${THROUGHPUT_TPS} tok/s met minimum ${MIN_THROUGHPUT_TPS} tok/s"

CURRENT_STATE="22 disarm-watchdog"
systemctl --user stop "$WATCHDOG_UNIT" "$WATCHDOG_SERVICE" >/dev/null 2>&1 || true
systemctl --user reset-failed "$WATCHDOG_UNIT" "$WATCHDOG_SERVICE" >/dev/null 2>&1 || true
WATCHDOG_ARMED=0
RESTORE_PROVEN=1
log_event "$CURRENT_STATE" pass "watchdog disarmed only after complete restore proof"

CURRENT_STATE="23 publish-durable-receipt-and-log"
if [[ "$CALLBACK_RC" -eq 0 ]]; then
  log_event "$CURRENT_STATE" pass "transaction and restore both passed"
  publish_receipt pass "callback passed and full restore proof passed"
  exit 0
fi
log_event "$CURRENT_STATE" fail "restore passed but callback failed with exit ${CALLBACK_RC}"
publish_receipt fail "callback failed with exit ${CALLBACK_RC}; full restore nevertheless passed"
exit "$CALLBACK_RC"
