#!/usr/bin/env bash
# Fail-closed build/receipt path for the pinned AutoRound G0/G1 comparator.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINERFILE=$REPO_ROOT/docker/Containerfile.frontier-autoround
VLLM_REPOSITORY=https://github.com/vllm-project/vllm.git
VLLM_COMMIT=b3da8bb8934667d42446614521bc234eaf24f192
VLLM_PARENT=5a4c8d99242e9e069b604d0e9b969e77f7dd501d
HUMMING_REPOSITORY=https://github.com/inclusionAI/humming.git
HUMMING_COMMIT=636ba85648c30ae2c2bfb335c9399593a67ecc1d
HUMMING_FILENAME=humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl
HUMMING_SHA256=cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331
HUMMING_URL=https://files.pythonhosted.org/packages/ec/a2/f96912ee7d68f56e61c15555657ffa057b8afbaa21241c63342b5969bb74/$HUMMING_FILENAME
COMPRESSED_REPOSITORY=https://github.com/vllm-project/compressed-tensors.git
COMPRESSED_COMMIT=f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0
COMPRESSED_FILENAME=compressed_tensors-0.17.0-py3-none-any.whl
COMPRESSED_SHA256=4a1b89b508f7efb8ffb4eee8a6e69e0452d9b080cae130146025c64fbe9fa9aa
COMPRESSED_URL=https://files.pythonhosted.org/packages/35/63/6edf0415b072fff0bf8b546074dea3f0f9b148e49b601ac98bdc60a76c68/$COMPRESSED_FILENAME
MODEL=Intel/Qwen3.8-27B-bpw2.8-AutoRound
MODEL_REVISION=03a2e36af5fad7b8eb281ff27bfb081e6216a257
export PYTHONOPTIMIZE=0

COMMAND=${1:-plan}
if (($#)); then shift; fi
BASE_IMAGE=${FRONTIER_BASE_IMAGE:-}
WHEEL_LOCK=${FRONTIER_WHEEL_LOCK:-}
WHEELHOUSE=${FRONTIER_WHEELHOUSE:-}
IMAGE=${FRONTIER_IMAGE:-localhost/qwen38-frontier-autoround:g1}
OUTPUT_DIR=${FRONTIER_OUTPUT_DIR:-$PWD/frontier-autoround-output}
AIBOSS_URL=${AIBOSS_URL:-http://127.0.0.1:8000}
ROUTE=
MEDIA_URL=
EXPECTED_TEXT=
SMOKE_IMAGE_URL=
SMOKE_VIDEO_URL=
SMOKE_TEXT_ANSWER=
SMOKE_IMAGE_ANSWER=
SMOKE_VIDEO_ANSWER=
IMAGE_DIGEST=
STARTUP_EVIDENCE=
MEMORY_EVIDENCE=
TEXT_EVIDENCE=
IMAGE_EVIDENCE=
VIDEO_EVIDENCE=
OUTPUT=

usage() {
    cat <<'EOF'
Usage:
  docker/build-frontier-autoround.sh plan --base-image REPO@sha256:... --wheel-lock FILE --wheelhouse DIR [--output FILE]
  docker/build-frontier-autoround.sh build --base-image REPO@sha256:... --wheel-lock FILE --wheelhouse DIR [--image TAG] [--output-dir DIR]
  docker/build-frontier-autoround.sh verify [--image TAG] [--output FILE]
  docker/build-frontier-autoround.sh smoke-plan --image-digest sha256:... --smoke-image-url URL --smoke-video-url URL --smoke-text-answer TEXT --smoke-image-answer TEXT --smoke-video-answer TEXT [--aiboss-url URL] [--output FILE]
  docker/build-frontier-autoround.sh smoke-route --route text|image|video --image-digest sha256:... --expected-text TEXT [--media-url URL] [--aiboss-url URL] --output FILE
  docker/build-frontier-autoround.sh runtime-receipt --image-digest sha256:... --startup-evidence FILE --memory-evidence FILE --text-evidence FILE --image-evidence FILE --video-evidence FILE --output FILE

The wheel lock schema is qwen38-frontier-autoround-wheel-lock/1. It binds base_image
and one immutable SHA256 per wheel, with scope build, runtime, or both. The runtime
lock must contain the exact Humming 0.1.12 and compressed-tensors 0.17.0 wheels.
EOF
}

die() { printf 'build-frontier-autoround.sh: %s\n' "$*" >&2; exit 2; }
need_value() { (($# >= 2)) || die "$1 requires a value"; }

while (($#)); do
    case "$1" in
        --base-image) need_value "$@"; BASE_IMAGE=$2; shift 2 ;;
        --wheel-lock) need_value "$@"; WHEEL_LOCK=$2; shift 2 ;;
        --wheelhouse) need_value "$@"; WHEELHOUSE=$2; shift 2 ;;
        --image) need_value "$@"; IMAGE=$2; shift 2 ;;
        --output-dir) need_value "$@"; OUTPUT_DIR=$2; shift 2 ;;
        --aiboss-url) need_value "$@"; AIBOSS_URL=$2; shift 2 ;;
        --route) need_value "$@"; ROUTE=$2; shift 2 ;;
        --media-url) need_value "$@"; MEDIA_URL=$2; shift 2 ;;
        --expected-text) need_value "$@"; EXPECTED_TEXT=$2; shift 2 ;;
        --smoke-image-url) need_value "$@"; SMOKE_IMAGE_URL=$2; shift 2 ;;
        --smoke-video-url) need_value "$@"; SMOKE_VIDEO_URL=$2; shift 2 ;;
        --smoke-text-answer) need_value "$@"; SMOKE_TEXT_ANSWER=$2; shift 2 ;;
        --smoke-image-answer) need_value "$@"; SMOKE_IMAGE_ANSWER=$2; shift 2 ;;
        --smoke-video-answer) need_value "$@"; SMOKE_VIDEO_ANSWER=$2; shift 2 ;;
        --image-digest) need_value "$@"; IMAGE_DIGEST=$2; shift 2 ;;
        --startup-evidence) need_value "$@"; STARTUP_EVIDENCE=$2; shift 2 ;;
        --memory-evidence) need_value "$@"; MEMORY_EVIDENCE=$2; shift 2 ;;
        --text-evidence) need_value "$@"; TEXT_EVIDENCE=$2; shift 2 ;;
        --image-evidence) need_value "$@"; IMAGE_EVIDENCE=$2; shift 2 ;;
        --video-evidence) need_value "$@"; VIDEO_EVIDENCE=$2; shift 2 ;;
        --output) need_value "$@"; OUTPUT=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

require_binary() { command -v "$1" >/dev/null 2>&1 || die "missing required binary: $1"; }
require_digest() { [[ $1 =~ ^sha256:[0-9a-f]{64}$ ]] || die "$2 must be sha256:<64 lowercase hex>"; }
require_base() {
    [[ $BASE_IMAGE =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] || die "--base-image must be an immutable repository@sha256 digest (tags and mutable refs are forbidden)"
}
require_build_inputs() {
    require_base
    [[ -f $WHEEL_LOCK ]] || die "--wheel-lock must name a regular file"
    [[ -d $WHEELHOUSE ]] || die "--wheelhouse must name a directory"
    [[ -f $CONTAINERFILE ]] || die "missing $CONTAINERFILE"
}

validate_wheel_lock() {
    BASE_IMAGE="$BASE_IMAGE" WHEEL_LOCK="$WHEEL_LOCK" WHEELHOUSE="$WHEELHOUSE" \
    HUMMING_FILENAME="$HUMMING_FILENAME" HUMMING_SHA256="$HUMMING_SHA256" \
    COMPRESSED_FILENAME="$COMPRESSED_FILENAME" COMPRESSED_SHA256="$COMPRESSED_SHA256" \
    PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
import re
from pathlib import Path
from frontier_common import load_strict_json, sha256_file

lock = load_strict_json(Path(os.environ['WHEEL_LOCK']))
assert isinstance(lock, dict) and set(lock) == {'schema', 'base_image', 'python', 'torch', 'wheels'}
assert lock['schema'] == 'qwen38-frontier-autoround-wheel-lock/1'
assert lock['base_image'] == os.environ['BASE_IMAGE']
assert lock['python'] == '3.12'
assert lock['torch'] == '2.13.0'
assert isinstance(lock['wheels'], list) and lock['wheels']
seen_files = set()
seen_distributions = set()
runtime_distributions = set()
expected = {
    os.environ['HUMMING_FILENAME']: ('humming-kernels', os.environ['HUMMING_SHA256']),
    os.environ['COMPRESSED_FILENAME']: ('compressed-tensors', os.environ['COMPRESSED_SHA256']),
}
for entry in lock['wheels']:
    assert isinstance(entry, dict) and set(entry) == {'distribution', 'filename', 'sha256', 'scope'}
    filename = entry['filename']
    distribution = entry['distribution'].lower().replace('_', '-')
    assert isinstance(filename, str) and filename == Path(filename).name and filename.endswith('.whl')
    assert re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])
    assert entry['scope'] in {'build', 'runtime', 'both'}
    assert filename not in seen_files and distribution not in seen_distributions
    assert distribution != 'llm-compressor'
    seen_files.add(filename)
    seen_distributions.add(distribution)
    if entry['scope'] in {'runtime', 'both'}:
        runtime_distributions.add(distribution)
    path = Path(os.environ['WHEELHOUSE']) / filename
    if path.exists():
        assert path.is_file() and sha256_file(path) == entry['sha256']
for filename, (distribution, digest) in expected.items():
    matches = [item for item in lock['wheels'] if item['filename'] == filename]
    assert len(matches) == 1
    assert matches[0]['distribution'].lower().replace('_', '-') == distribution
    assert matches[0]['sha256'] == digest and matches[0]['scope'] in {'runtime', 'both'}
assert {'nvidia-cuda-runtime', 'nvidia-cuda-cccl', 'nvidia-cuda-nvcc', 'nvidia-cuda-nvrtc'} <= runtime_distributions
PY
}

write_plan() {
    local destination=${OUTPUT:-/dev/stdout}
    BASE_IMAGE="$BASE_IMAGE" WHEEL_LOCK="$WHEEL_LOCK" WHEELHOUSE="$WHEELHOUSE" IMAGE="$IMAGE" \
    OUTPUT_DIR="$OUTPUT_DIR" CONTAINERFILE="$CONTAINERFILE" VLLM_REPOSITORY="$VLLM_REPOSITORY" \
    VLLM_COMMIT="$VLLM_COMMIT" VLLM_PARENT="$VLLM_PARENT" HUMMING_REPOSITORY="$HUMMING_REPOSITORY" \
    HUMMING_COMMIT="$HUMMING_COMMIT" HUMMING_URL="$HUMMING_URL" HUMMING_SHA256="$HUMMING_SHA256" \
    COMPRESSED_REPOSITORY="$COMPRESSED_REPOSITORY" COMPRESSED_COMMIT="$COMPRESSED_COMMIT" \
    COMPRESSED_URL="$COMPRESSED_URL" COMPRESSED_SHA256="$COMPRESSED_SHA256" DESTINATION="$destination" \
    PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
from pathlib import Path
from frontier_common import atomic_write_json, load_strict_json, sha256_file

lock = load_strict_json(Path(os.environ['WHEEL_LOCK']))
value = {
    'schema': 'qwen38-frontier-autoround-build-plan/1',
    'executes': False,
    'base_image': os.environ['BASE_IMAGE'],
    'image_tag': os.environ['IMAGE'],
    'source_fetches': [
        {'repository': os.environ['VLLM_REPOSITORY'], 'commit': os.environ['VLLM_COMMIT'], 'required_parent': os.environ['VLLM_PARENT'], 'depth': 2},
        {'repository': os.environ['HUMMING_REPOSITORY'], 'commit': os.environ['HUMMING_COMMIT'], 'depth': 1},
        {'repository': os.environ['COMPRESSED_REPOSITORY'], 'commit': os.environ['COMPRESSED_COMMIT'], 'depth': 1},
    ],
    'pinned_wheel_fetches': [
        {'url': os.environ['HUMMING_URL'], 'sha256': os.environ['HUMMING_SHA256']},
        {'url': os.environ['COMPRESSED_URL'], 'sha256': os.environ['COMPRESSED_SHA256']},
    ],
    'wheel_lock': {'path': str(Path(os.environ['WHEEL_LOCK']).resolve()), 'sha256': sha256_file(Path(os.environ['WHEEL_LOCK'])), 'entries': len(lock['wheels'])},
    'wheelhouse': str(Path(os.environ['WHEELHOUSE']).resolve()),
    'containerfile': {'path': os.environ['CONTAINERFILE'], 'sha256': sha256_file(Path(os.environ['CONTAINERFILE']))},
    'build_argv': ['podman', 'build', '--pull=never', '--network=none', '-f', 'Containerfile.frontier-autoround', '-t', os.environ['IMAGE'], '--build-arg', f"FRONTIER_BASE_IMAGE={os.environ['BASE_IMAGE']}", '<verified-temporary-context>'],
    'outputs': ['source-lock.json', 'source-sbom.json', 'installed-bytes.json', 'image-receipt.json'],
    'runtime_policy': {'model_revision_fixed': '03a2e36af5fad7b8eb281ff27bfb081e6216a257', 'max_model_len': 4096, 'mtp': 'forbidden', 'llm_compressor_serving_dependency': False},
}
if os.environ['DESTINATION'] == '/dev/stdout':
    import sys
    sys.stdout.buffer.write(__import__('frontier_common').canonical_bytes(value) + b'\n')
else:
    atomic_write_json(Path(os.environ['DESTINATION']), value)
PY
}

fetch_git() {
    local repository=$1 commit=$2 destination=$3 depth=$4
    git init --quiet "$destination"
    git -C "$destination" remote add origin "$repository"
    git -C "$destination" fetch --quiet --no-tags --depth="$depth" origin "$commit"
    git -C "$destination" checkout --quiet --detach "$commit"
    [[ $(git -C "$destination" rev-parse HEAD) == "$commit" ]] || die "fetched commit mismatch for $repository"
}

stage_build() {
    require_binary python3; require_binary git; require_binary curl; require_binary podman; require_binary sha256sum
    require_build_inputs
    validate_wheel_lock
    podman image exists "$BASE_IMAGE" || die "pinned base image is not present locally; pull it explicitly by digest before building"
    mkdir -p "$OUTPUT_DIR"
    local context
    context=$(mktemp -d "${TMPDIR:-/tmp}/frontier-autoround.XXXXXX")
    trap 'rm -rf -- "$context"' EXIT
    mkdir -p "$context/wheelhouse"
    fetch_git "$VLLM_REPOSITORY" "$VLLM_COMMIT" "$context/vllm" 2
    [[ $(git -C "$context/vllm" rev-parse HEAD^) == "$VLLM_PARENT" ]] || die "vLLM head has the wrong parent"
    [[ $(git -C "$context/vllm" rev-list --count "$VLLM_PARENT..$VLLM_COMMIT") == 1 ]] || die "vLLM consumer is not exactly one commit"
    fetch_git "$HUMMING_REPOSITORY" "$HUMMING_COMMIT" "$context/humming-source" 1
    fetch_git "$COMPRESSED_REPOSITORY" "$COMPRESSED_COMMIT" "$context/compressed-tensors-source" 1

    cp -- "$WHEELHOUSE"/*.whl "$context/wheelhouse/" 2>/dev/null || true
    if [[ ! -f $context/wheelhouse/$HUMMING_FILENAME ]]; then
        curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 "$HUMMING_URL" -o "$context/wheelhouse/$HUMMING_FILENAME"
    fi
    if [[ ! -f $context/wheelhouse/$COMPRESSED_FILENAME ]]; then
        curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 "$COMPRESSED_URL" -o "$context/wheelhouse/$COMPRESSED_FILENAME"
    fi
    cp -- "$WHEEL_LOCK" "$context/wheel-lock.json"
    WHEELHOUSE="$context/wheelhouse" validate_wheel_lock

    local vllm_tree humming_tree compressed_tree wheel_lock_sha
    vllm_tree=$(git -C "$context/vllm" rev-parse HEAD^{tree})
    humming_tree=$(git -C "$context/humming-source" rev-parse HEAD^{tree})
    compressed_tree=$(git -C "$context/compressed-tensors-source" rev-parse HEAD^{tree})
    wheel_lock_sha=$(sha256sum "$context/wheel-lock.json" | cut -d' ' -f1)
    BASE_IMAGE="$BASE_IMAGE" VLLM_TREE="$vllm_tree" HUMMING_TREE="$humming_tree" COMPRESSED_TREE="$compressed_tree" \
    WHEEL_LOCK_SHA="$wheel_lock_sha" CONTEXT="$context" OUTPUT_DIR="$OUTPUT_DIR" \
    PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
from pathlib import Path
from frontier_common import atomic_write_json, load_strict_json, sha256_file

context = Path(os.environ['CONTEXT'])
lock = {
    'schema': 'qwen38-frontier-autoround-source-lock/1',
    'base_image': os.environ['BASE_IMAGE'],
    'vllm': {'repository': 'https://github.com/vllm-project/vllm.git', 'commit': 'b3da8bb8934667d42446614521bc234eaf24f192', 'parent': '5a4c8d99242e9e069b604d0e9b969e77f7dd501d', 'relationship': 'one_commit_on_parent'},
    'sources': [
        {'name': 'vllm', 'repository': 'https://github.com/vllm-project/vllm.git', 'commit': 'b3da8bb8934667d42446614521bc234eaf24f192', 'tree': os.environ['VLLM_TREE']},
        {'name': 'humming', 'repository': 'https://github.com/inclusionAI/humming.git', 'commit': '636ba85648c30ae2c2bfb335c9399593a67ecc1d', 'tree': os.environ['HUMMING_TREE']},
        {'name': 'compressed-tensors', 'repository': 'https://github.com/vllm-project/compressed-tensors.git', 'commit': 'f3b707b7d37515fa7d61c7f65d76fa6867c0b3e0', 'tree': os.environ['COMPRESSED_TREE']},
    ],
    'wheel_lock_sha256': os.environ['WHEEL_LOCK_SHA'],
    'model': {'repository': 'Intel/Qwen3.8-27B-bpw2.8-AutoRound', 'revision': '03a2e36af5fad7b8eb281ff27bfb081e6216a257'},
}
atomic_write_json(context / 'source-lock.json', lock)
wheel_lock = load_strict_json(context / 'wheel-lock.json')
sbom = {
    'schema': 'qwen38-frontier-autoround-source-sbom/1',
    'base_image': os.environ['BASE_IMAGE'],
    'sources': lock['sources'],
    'wheels': [{**item, 'bytes': (context / 'wheelhouse' / item['filename']).stat().st_size} for item in wheel_lock['wheels']],
    'serving_exclusions': ['llm-compressor', 'MTP/speculative decoding'],
}
atomic_write_json(context / 'source-sbom.json', sbom)
atomic_write_json(Path(os.environ['OUTPUT_DIR']) / 'source-lock.json', lock)
atomic_write_json(Path(os.environ['OUTPUT_DIR']) / 'source-sbom.json', sbom)
PY
    cp -- "$REPO_ROOT/tools/frontier_common.py" "$context/frontier_common.py"
    cp -- "$CONTAINERFILE" "$context/Containerfile.frontier-autoround"
    rm -rf -- "$context/humming-source" "$context/compressed-tensors-source"
    local source_lock_sha
    source_lock_sha=$(sha256sum "$context/source-lock.json" | cut -d' ' -f1)
    podman build --pull=never --network=none \
        -f "$context/Containerfile.frontier-autoround" -t "$IMAGE" \
        --build-arg "FRONTIER_BASE_IMAGE=$BASE_IMAGE" \
        --build-arg "FRONTIER_SOURCE_LOCK_SHA256=$source_lock_sha" \
        "$context"

    local digest config_id
    digest=$(podman image inspect "$IMAGE" --format '{{.Digest}}')
    config_id=$(podman image inspect "$IMAGE" --format '{{.Id}}')
    require_digest "$digest" "built image manifest digest"
    require_digest "$config_id" "built image config digest"
    podman run --rm --network none --entrypoint cat "$IMAGE" /opt/frontier/installed-bytes.json > "$OUTPUT_DIR/installed-bytes.json"
    podman run --rm --network none --entrypoint cat "$IMAGE" /opt/frontier/vllm-wheel.sha256 > "$OUTPUT_DIR/vllm-wheel.sha256"
    IMAGE="$IMAGE" DIGEST="$digest" CONFIG_ID="$config_id" BASE_IMAGE="$BASE_IMAGE" SOURCE_LOCK_SHA="$source_lock_sha" \
    OUTPUT_DIR="$OUTPUT_DIR" PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
import re
from pathlib import Path
from frontier_common import atomic_write_json, load_strict_json, sha256_file

out = Path(os.environ['OUTPUT_DIR'])
installed = load_strict_json(out / 'installed-bytes.json')
source_sbom = load_strict_json(out / 'source-sbom.json')
wheel_line = (out / 'vllm-wheel.sha256').read_text(encoding='utf-8').strip()
match = re.fullmatch(r'([0-9a-f]{64})\\s+(.+\\.whl)', wheel_line)
assert match
image_sbom = {
    'schema': 'qwen38-frontier-autoround-image-sbom/1',
    'image': {'manifest_digest': os.environ['DIGEST'], 'config_digest': os.environ['CONFIG_ID']},
    'source_sbom': source_sbom,
    'built_vllm_wheel': {'sha256': match.group(1), 'filename': Path(match.group(2)).name},
    'installed_packages': installed['packages'],
}
atomic_write_json(out / 'image-sbom.json', image_sbom)
receipt = {
    'schema': 'qwen38-frontier-autoround-image-receipt/1',
    'status': 'built_not_runtime_qualified',
    'image': {'tag': os.environ['IMAGE'], 'manifest_digest': os.environ['DIGEST'], 'config_digest': os.environ['CONFIG_ID']},
    'base_image': os.environ['BASE_IMAGE'],
    'source_lock_sha256': os.environ['SOURCE_LOCK_SHA'],
    'installed_bytes_sha256': sha256_file(out / 'installed-bytes.json'),
    'image_sbom_sha256': sha256_file(out / 'image-sbom.json'),
    'vllm_wheel_sha256': match.group(1),
    'runtime': {'runnable': False, 'max_model_len': 4096, 'mtp': 'forbidden'},
}
atomic_write_json(out / 'image-receipt.json', receipt)
PY
    printf '%s\n' "$digest"
}

stage_verify() {
    require_binary python3; require_binary podman
    podman image exists "$IMAGE" || die "image is not present: $IMAGE"
    local destination=${OUTPUT:-/dev/stdout}
    local temporary
    temporary=$(mktemp "${TMPDIR:-/tmp}/frontier-installed.XXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    podman run --rm --network none --entrypoint python "$IMAGE" - <<'PY' > "$temporary"
import hashlib
import importlib.metadata
import json
from pathlib import Path

wheel_lock = json.loads(Path('/opt/frontier/wheel-lock.json').read_text(encoding='utf-8'))
names = sorted({item['distribution'] for item in wheel_lock['wheels'] if item['scope'] in {'runtime', 'both'}} | {'vllm'})
packages = []
for name in names:
    distribution = importlib.metadata.distribution(name)
    files = []
    for item in sorted(distribution.files or [], key=str):
        path = Path(distribution.locate_file(item))
        if path.is_file():
            files.append({'path': str(item), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert files, name
    packages.append({'distribution': name, 'version': distribution.version, 'files': files})
print(json.dumps({'schema': 'qwen38-frontier-autoround-installed-bytes/1', 'packages': packages}, sort_keys=True, separators=(',', ':')))
PY
    local baked digest
    baked=$(podman run --rm --network none --entrypoint cat "$IMAGE" /opt/frontier/installed-bytes.json)
    cmp -s "$temporary" <(printf '%s\n' "$baked") || die "installed bytes differ from the in-image manifest"
    digest=$(podman image inspect "$IMAGE" --format '{{.Digest}}')
    require_digest "$digest" "image manifest digest"
    IMAGE="$IMAGE" DIGEST="$digest" DESTINATION="$destination" TEMPORARY="$temporary" \
    PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
from pathlib import Path
from frontier_common import atomic_write_json, canonical_bytes, load_strict_json, sha256_file

value = {'schema': 'qwen38-frontier-autoround-image-verification/1', 'status': 'pass', 'image': os.environ['IMAGE'], 'manifest_digest': os.environ['DIGEST'], 'installed_bytes_sha256': sha256_file(Path(os.environ['TEMPORARY'])), 'runtime_qualified': False}
if os.environ['DESTINATION'] == '/dev/stdout':
    import sys; sys.stdout.buffer.write(canonical_bytes(value) + b'\n')
else:
    atomic_write_json(Path(os.environ['DESTINATION']), value)
PY
}

stage_smoke_plan() {
    require_digest "$IMAGE_DIGEST" "--image-digest"
    [[ -n $SMOKE_IMAGE_URL && -n $SMOKE_VIDEO_URL ]] || die "smoke-plan requires exact --smoke-image-url and --smoke-video-url values"
    [[ -n $SMOKE_TEXT_ANSWER && -n $SMOKE_IMAGE_ANSWER && -n $SMOKE_VIDEO_ANSWER ]] || die "smoke-plan requires all three exact --smoke-*-answer values"
    local destination=${OUTPUT:-/dev/stdout}
    IMAGE="$IMAGE" IMAGE_DIGEST="$IMAGE_DIGEST" AIBOSS_URL="$AIBOSS_URL" DESTINATION="$destination" \
    SMOKE_IMAGE_URL="$SMOKE_IMAGE_URL" SMOKE_VIDEO_URL="$SMOKE_VIDEO_URL" \
    SMOKE_TEXT_ANSWER="$SMOKE_TEXT_ANSWER" SMOKE_IMAGE_ANSWER="$SMOKE_IMAGE_ANSWER" SMOKE_VIDEO_ANSWER="$SMOKE_VIDEO_ANSWER" \
    PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
from pathlib import Path
from frontier_common import atomic_write_json, canonical_bytes

base = os.environ['AIBOSS_URL'].rstrip('/')
def request(content):
    return {'model': 'qwen38-frontier-autoround', 'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 64}
value = {
    'schema': 'qwen38-frontier-autoround-smoke-plan/1',
    'executes': False,
    'image': {'tag': os.environ['IMAGE'], 'manifest_digest': os.environ['IMAGE_DIGEST']},
    'container_argv_for_aiboss': ['podman', 'run', '--rm', '--gpus', 'all', '--network', 'host', os.environ['IMAGE'] + '@' + os.environ['IMAGE_DIGEST'], '--enforce-eager'],
    'startup_owner': 'external_AIBoss',
    'fixed_by_entrypoint': {'model': 'Intel/Qwen3.8-27B-bpw2.8-AutoRound', 'revision': '03a2e36af5fad7b8eb281ff27bfb081e6216a257', 'served_model_name': 'qwen38-frontier-autoround', 'max_model_len': 4096, 'mtp': False},
    'aiboss_route': base + '/v1/chat/completions',
    'route_hooks': {
        'text': {'request': request([{'type': 'text', 'text': f\"Return exactly: {os.environ['SMOKE_TEXT_ANSWER']}\"}]), 'expected_answer': os.environ['SMOKE_TEXT_ANSWER']},
        'image': {'request': request([{'type': 'text', 'text': f\"Inspect the image and return exactly: {os.environ['SMOKE_IMAGE_ANSWER']}\"}, {'type': 'image_url', 'image_url': {'url': os.environ['SMOKE_IMAGE_URL']}}]), 'expected_answer': os.environ['SMOKE_IMAGE_ANSWER']},
        'video': {'request': request([{'type': 'text', 'text': f\"Inspect the video and return exactly: {os.environ['SMOKE_VIDEO_ANSWER']}\"}, {'type': 'video_url', 'video_url': {'url': os.environ['SMOKE_VIDEO_URL']}}]), 'expected_answer': os.environ['SMOKE_VIDEO_ANSWER']},
    },
    'runtime_receipt_requires': ['AIBoss startup', 'cold JIT completion', 'fallback_count=0', 'served routes', 'BF16 vision loaded', 'text response', 'image response', 'video response', 'measured peak and steady GPU memory'],
}
if os.environ['DESTINATION'] == '/dev/stdout':
    import sys; sys.stdout.buffer.write(canonical_bytes(value) + b'\n')
else:
    atomic_write_json(Path(os.environ['DESTINATION']), value)
PY
}

stage_smoke_route() {
    [[ $ROUTE == text || $ROUTE == image || $ROUTE == video ]] || die "--route must be text, image, or video"
    require_digest "$IMAGE_DIGEST" "--image-digest"
    [[ -n $EXPECTED_TEXT ]] || die "--expected-text is required"
    [[ $ROUTE == text || -n $MEDIA_URL ]] || die "--media-url is required for image/video"
    [[ -n $OUTPUT ]] || die "--output is required"
    ROUTE="$ROUTE" MEDIA_URL="$MEDIA_URL" EXPECTED_TEXT="$EXPECTED_TEXT" IMAGE_DIGEST="$IMAGE_DIGEST" \
    AIBOSS_URL="$AIBOSS_URL" OUTPUT="$OUTPUT" PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import json
import os
import urllib.request
from pathlib import Path
from frontier_common import atomic_write_json, canonical_sha256

route = os.environ['ROUTE']
expected = os.environ['EXPECTED_TEXT']
instruction = f'Return exactly: {expected}' if route == 'text' else f'Inspect the {route} and return exactly: {expected}'
content = [{'type': 'text', 'text': instruction}]
if route != 'text':
    media_type = route + '_url'
    content.append({'type': media_type, media_type: {'url': os.environ['MEDIA_URL']}})
payload = {'model': 'qwen38-frontier-autoround', 'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 64}
request = urllib.request.Request(os.environ['AIBOSS_URL'].rstrip('/') + '/v1/chat/completions', data=json.dumps(payload, separators=(',', ':')).encode(), headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(request, timeout=300) as response:
    raw = response.read(8 << 20)
    status = response.status
body = json.loads(raw)
answer = body['choices'][0]['message']['content']
passed = status == 200 and answer == os.environ['EXPECTED_TEXT']
receipt = {
    'schema': 'qwen38-frontier-autoround-route-smoke/1',
    'route': route,
    'status': 'pass' if passed else 'fail',
    'passed': passed,
    'image_digest': os.environ['IMAGE_DIGEST'],
    'model_revision': '03a2e36af5fad7b8eb281ff27bfb081e6216a257',
    'request_sha256': canonical_sha256(payload),
    'http_status': status,
    'expected_text': os.environ['EXPECTED_TEXT'],
    'answer': answer,
    'response_sha256': __import__('hashlib').sha256(raw).hexdigest(),
    'evidence_scope': 'route_only_not_runtime_qualification',
}
atomic_write_json(Path(os.environ['OUTPUT']), receipt)
if not passed:
    raise SystemExit(3)
PY
}

stage_runtime_receipt() {
    require_digest "$IMAGE_DIGEST" "--image-digest"
    [[ -f $STARTUP_EVIDENCE && -f $MEMORY_EVIDENCE && -f $TEXT_EVIDENCE && -f $IMAGE_EVIDENCE && -f $VIDEO_EVIDENCE ]] || die "all five evidence files are required"
    [[ -n $OUTPUT ]] || die "--output is required"
    IMAGE_DIGEST="$IMAGE_DIGEST" STARTUP_EVIDENCE="$STARTUP_EVIDENCE" MEMORY_EVIDENCE="$MEMORY_EVIDENCE" \
    TEXT_EVIDENCE="$TEXT_EVIDENCE" IMAGE_EVIDENCE="$IMAGE_EVIDENCE" VIDEO_EVIDENCE="$VIDEO_EVIDENCE" OUTPUT="$OUTPUT" \
    PYTHONPATH="$REPO_ROOT/tools" python3 - <<'PY'
import os
import re
from pathlib import Path
from frontier_common import atomic_write_json, load_strict_json, sha256_file

image = os.environ['IMAGE_DIGEST']
def load(name): return load_strict_json(Path(os.environ[name]))
startup = load('STARTUP_EVIDENCE')
memory = load('MEMORY_EVIDENCE')
routes = {route: load(route.upper() + '_EVIDENCE') for route in ('text', 'image', 'video')}
assert isinstance(startup, dict) and startup.get('schema') == 'qwen38-frontier-aiboss-startup-evidence/1'
assert startup.get('image_digest') == image and startup.get('model_revision') == '03a2e36af5fad7b8eb281ff27bfb081e6216a257'
startup_gates = ['aiboss_started', 'cold_jit_completed', 'routes_observed', 'bf16_vision_loaded']
assert all(startup.get(gate) is True for gate in startup_gates)
assert startup.get('jit_cache_cold_before_start') is True
assert startup.get('fallback_count') == 0
assert startup.get('mtp_enabled') is False and startup.get('max_model_len') == 4096
assert startup.get('vision_dtype') == 'bfloat16'
assert '/v1/chat/completions' in startup.get('route_inventory', [])
assert re.fullmatch(r'[0-9a-f]{64}', startup.get('observed_log_sha256', ''))
assert isinstance(memory, dict) and memory.get('schema') == 'qwen38-frontier-memory-evidence/1'
assert memory.get('image_digest') == image and memory.get('measured') is True
assert isinstance(memory.get('sample_count'), int) and memory['sample_count'] >= 2
assert isinstance(memory.get('peak_gpu_bytes'), int) and memory['peak_gpu_bytes'] > 0
assert isinstance(memory.get('steady_gpu_bytes'), int) and memory['steady_gpu_bytes'] > 0
assert re.fullmatch(r'[0-9a-f]{64}', memory.get('measurement_log_sha256', ''))
for route, receipt in routes.items():
    assert isinstance(receipt, dict) and receipt.get('schema') == 'qwen38-frontier-autoround-route-smoke/1'
    assert receipt.get('route') == route and receipt.get('passed') is True and receipt.get('status') == 'pass'
    assert receipt.get('image_digest') == image and receipt.get('model_revision') == '03a2e36af5fad7b8eb281ff27bfb081e6216a257'
value = {
    'schema': 'qwen38-frontier-autoround-runtime-receipt/1',
    'status': 'runnable_non_speculative',
    'runnable': True,
    'image_digest': image,
    'model_revision': '03a2e36af5fad7b8eb281ff27bfb081e6216a257',
    'max_model_len': 4096,
    'mtp': {'enabled': False, 'native_claim_allowed': False},
    'gates': {'aiboss_startup': True, 'cold_jit': True, 'zero_fallback': True, 'routes': True, 'bf16_vision': True, 'text': True, 'image': True, 'video': True, 'memory': True},
    'evidence': {key.lower(): {'path': os.environ[key], 'sha256': sha256_file(Path(os.environ[key]))} for key in ('STARTUP_EVIDENCE', 'MEMORY_EVIDENCE', 'TEXT_EVIDENCE', 'IMAGE_EVIDENCE', 'VIDEO_EVIDENCE')},
}
atomic_write_json(Path(os.environ['OUTPUT']), value)
PY
}

case "$COMMAND" in
    plan)
        require_binary python3; require_build_inputs; validate_wheel_lock; write_plan ;;
    build)
        stage_build ;;
    verify)
        stage_verify ;;
    smoke-plan)
        require_binary python3; stage_smoke_plan ;;
    smoke-route)
        require_binary python3; stage_smoke_route ;;
    runtime-receipt)
        require_binary python3; stage_runtime_receipt ;;
    help|-h|--help)
        usage ;;
    *)
        usage >&2; die "unknown command: $COMMAND" ;;
esac
