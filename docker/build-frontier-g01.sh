#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LOCK="$REPO_ROOT/docker/frontier-g01-sources.json"
RECEIPT_ROOT="$REPO_ROOT/receipts/frontier-g01"
WORK_DIR="$RECEIPT_ROOT/build-work"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/qwen38-frontier-g01"
RESOLVED_LOCK="$RECEIPT_ROOT/runtime-resolved-lock.json"
SOURCE_RECEIPT="$RECEIPT_ROOT/runtime-source-receipt.json"
SOURCE_SBOM="$RECEIPT_ROOT/runtime-source-sbom.json"
VERIFICATION_RECEIPT="$RECEIPT_ROOT/runtime-source-verification.json"
IMAGE_RECEIPT="$RECEIPT_ROOT/runtime-image-receipt.json"
IMAGE_REFERENCE="localhost/qwen38/frontier-g01:clean"

usage() {
  cat >&2 <<'EOF'
usage: docker/build-frontier-g01.sh [options]
  --lock PATH
  --work-dir PATH
  --cache-dir PATH
  --resolved-lock PATH
  --source-receipt PATH
  --sbom PATH
  --verification-receipt PATH
  --image-receipt PATH
  --image-reference localhost/qwen38/frontier-g01:TAG
EOF
  exit 2
}

while (( $# )); do
  case "$1" in
    --lock) [[ $# -ge 2 ]] || usage; LOCK=$2; shift 2 ;;
    --work-dir) [[ $# -ge 2 ]] || usage; WORK_DIR=$2; shift 2 ;;
    --cache-dir) [[ $# -ge 2 ]] || usage; CACHE_DIR=$2; shift 2 ;;
    --resolved-lock) [[ $# -ge 2 ]] || usage; RESOLVED_LOCK=$2; shift 2 ;;
    --source-receipt) [[ $# -ge 2 ]] || usage; SOURCE_RECEIPT=$2; shift 2 ;;
    --sbom) [[ $# -ge 2 ]] || usage; SOURCE_SBOM=$2; shift 2 ;;
    --verification-receipt) [[ $# -ge 2 ]] || usage; VERIFICATION_RECEIPT=$2; shift 2 ;;
    --image-receipt) [[ $# -ge 2 ]] || usage; IMAGE_RECEIPT=$2; shift 2 ;;
    --image-reference) [[ $# -ge 2 ]] || usage; IMAGE_REFERENCE=$2; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

case "$IMAGE_REFERENCE" in
  localhost/qwen38/frontier-g01:*) ;;
  *)
    printf 'build-frontier-g01.sh: image reference must stay in localhost/qwen38/frontier-g01:*\n' >&2
    exit 2
    ;;
esac

for binary in git podman python3 install mkdir; do
  command -v "$binary" >/dev/null 2>&1 || {
    printf 'build-frontier-g01.sh: missing required binary: %s\n' "$binary" >&2
    exit 127
  }
done

[[ -f "$LOCK" ]] || {
  printf 'build-frontier-g01.sh: source lock is not a regular file: %s\n' "$LOCK" >&2
  exit 2
}
[[ ! -e "$WORK_DIR" ]] || {
  printf 'build-frontier-g01.sh: work directory must not already exist: %s\n' "$WORK_DIR" >&2
  exit 2
}
for output in "$RESOLVED_LOCK" "$SOURCE_RECEIPT" "$SOURCE_SBOM" "$VERIFICATION_RECEIPT" "$IMAGE_RECEIPT"; do
  [[ ! -e "$output" ]] || {
    printf 'build-frontier-g01.sh: refusing nonempty/existing output artifact: %s\n' "$output" >&2
    exit 2
  }
done
if podman image exists "$IMAGE_REFERENCE"; then
  printf 'build-frontier-g01.sh: refusing to replace image reference: %s\n' "$IMAGE_REFERENCE" >&2
  exit 2
fi

mkdir -p -- "$CACHE_DIR/sources" "$CACHE_DIR/build"
mkdir -p -- "$(dirname -- "$RESOLVED_LOCK")" "$(dirname -- "$SOURCE_RECEIPT")" \
  "$(dirname -- "$SOURCE_SBOM")" "$(dirname -- "$VERIFICATION_RECEIPT")" \
  "$(dirname -- "$IMAGE_RECEIPT")"
mkdir -m 0700 -- "$WORK_DIR"
CONTEXT="$WORK_DIR/context"
mkdir -m 0700 -- "$CONTEXT"
CANDIDATE_REFERENCE=""
completed=0
cleanup() {
  if [[ $completed -ne 1 && -n "$CANDIDATE_REFERENCE" ]]; then
    podman image rm --force "$CANDIDATE_REFERENCE" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

python3 "$REPO_ROOT/tools/frontier_runtime_source.py" validate-lock \
  --lock "$LOCK" \
  --report "$WORK_DIR/lock-validation.json"

python3 "$REPO_ROOT/tools/frontier_runtime_source.py" assemble \
  --lock "$LOCK" \
  --repo-root "$REPO_ROOT" \
  --cache "$CACHE_DIR/sources" \
  --workspace "$CONTEXT/sources" \
  --resolved-lock "$RESOLVED_LOCK" \
  --source-receipt "$SOURCE_RECEIPT" \
  --sbom "$SOURCE_SBOM"

python3 "$REPO_ROOT/tools/frontier_runtime_source.py" base-reference \
  --lock "$LOCK" \
  --output "$WORK_DIR/base-reference.json"
BASE_REFERENCE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["base_reference"])' "$WORK_DIR/base-reference.json")
RESOLVED_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["resolved_sha256"])' "$RESOLVED_LOCK")
[[ "$BASE_REFERENCE" == *@sha256:* ]] || {
  printf 'build-frontier-g01.sh: validated lock returned a non-digest base reference\n' >&2
  exit 2
}
[[ "$RESOLVED_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'build-frontier-g01.sh: invalid resolved source digest\n' >&2
  exit 2
}

install -m 0644 "$REPO_ROOT/docker/Containerfile.frontier-g01" "$CONTEXT/Containerfile"
install -m 0644 "$REPO_ROOT/tools/frontier_common.py" "$CONTEXT/frontier_common.py"
install -m 0755 "$REPO_ROOT/tools/frontier_runtime_source.py" "$CONTEXT/frontier_runtime_source.py"
install -m 0644 "$RESOLVED_LOCK" "$CONTEXT/frontier-resolved.json"
install -m 0644 "$SOURCE_RECEIPT" "$CONTEXT/frontier-source-receipt.json"
install -m 0644 "$SOURCE_SBOM" "$CONTEXT/frontier-source-sbom.json"

CANDIDATE_REFERENCE="localhost/qwen38/frontier-g01-candidate:${RESOLVED_SHA256:0:16}"
if podman image exists "$CANDIDATE_REFERENCE"; then
  printf 'build-frontier-g01.sh: refusing to replace candidate image: %s\n' "$CANDIDATE_REFERENCE" >&2
  exit 2
fi

podman build \
  --pull=never \
  --network=none \
  --volume "$CACHE_DIR/build:/var/cache/frontier-g01:rw" \
  --build-arg "FRONTIER_BASE_IMAGE=$BASE_REFERENCE" \
  --build-arg "FRONTIER_RESOLVED_SHA256=$RESOLVED_SHA256" \
  --tag "$CANDIDATE_REFERENCE" \
  "$CONTEXT"

mkdir -m 0700 -- "$WORK_DIR/verification"
podman run --rm --network=none \
  --volume "$WORK_DIR/verification:/verification:rw" \
  --entrypoint python \
  "$CANDIDATE_REFERENCE" \
  /opt/frontier/tools/frontier_runtime_source.py verify-runtime \
  --resolved-lock /opt/frontier/frontier-resolved.json \
  --workspace /usr/local/src/frontier \
  --report /verification/runtime-source-verification.json
install -m 0644 "$WORK_DIR/verification/runtime-source-verification.json" "$VERIFICATION_RECEIPT"

podman tag "$CANDIDATE_REFERENCE" "$IMAGE_REFERENCE"
IMAGE_DIGEST=$(podman image inspect --format '{{.Id}}' "$IMAGE_REFERENCE")
if [[ "$IMAGE_DIGEST" =~ ^[0-9a-f]{64}$ ]]; then
  IMAGE_DIGEST="sha256:${IMAGE_DIGEST}"
fi
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  printf 'build-frontier-g01.sh: podman returned an invalid image digest: %s\n' "$IMAGE_DIGEST" >&2
  exit 2
}
python3 "$REPO_ROOT/tools/frontier_runtime_source.py" finalize-image \
  --source-receipt "$SOURCE_RECEIPT" \
  --verification-report "$VERIFICATION_RECEIPT" \
  --sbom "$SOURCE_SBOM" \
  --image-reference "$IMAGE_REFERENCE" \
  --image-digest "$IMAGE_DIGEST" \
  --output "$IMAGE_RECEIPT"

podman image rm "$CANDIDATE_REFERENCE" >/dev/null
CANDIDATE_REFERENCE=""
completed=1
printf '%s\n' "$IMAGE_DIGEST"
