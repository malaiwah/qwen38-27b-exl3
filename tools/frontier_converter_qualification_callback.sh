#!/bin/bash
# Execute deterministic A/A and resume controls for the pinned EXL3 converter.
set -euo pipefail

[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || {
  echo "frontier_converter_qualification_callback: transaction is not active" >&2
  exit 2
}
for name in FRONTIER_CAMPAIGN_CONTAINER FRONTIER_CAMPAIGN_IMAGE \
  FRONTIER_CAMPAIGN_CACHE_DIR FRONTIER_CAMPAIGN_WORK_DIR; do
  [[ -n "${!name:-}" ]] || { echo "missing ${name}" >&2; exit 2; }
done

SOURCE=/home/mbelleau/final-frontier-g0/converter-source
ENTRY=/home/mbelleau/final-frontier-g0/tools/frontier_converter_deterministic.py
BF16_REPO=/home/mbelleau/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B
EXT_CACHE_SOURCE=/home/mbelleau/final-frontier-g0/cache/converter-base-ext
RESULT="${FRONTIER_CAMPAIGN_WORK_DIR}/converter-controls.json"
for path in "$SOURCE" "$BF16_REPO" "$EXT_CACHE_SOURCE"; do
  [[ -d "$path" && ! -L "$path" ]] || { echo "missing immutable converter input ${path}" >&2; exit 2; }
done
[[ -f "$ENTRY" && ! -L "$ENTRY" && -s "$ENTRY" ]] || { echo "missing deterministic converter entry" >&2; exit 2; }
[[ ! -e "$RESULT" ]] || { echo "converter result already exists" >&2; exit 2; }
cp -a "$EXT_CACHE_SOURCE/." "${FRONTIER_CAMPAIGN_CACHE_DIR}/"

run_converter() {
  local label=$1
  shift
  podman run --rm --replace \
    --name "${FRONTIER_CAMPAIGN_CONTAINER}" \
    --network none --ipc=host --device nvidia.com/gpu=all \
    --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
    -v "${SOURCE}:/src:ro" \
    -v "${ENTRY}:/opt/frontier/frontier_converter_deterministic.py:ro" \
    -v "${BF16_REPO}:/models/bf16-repo:ro" \
    -v "${FRONTIER_CAMPAIGN_CACHE_DIR}:/cache:rw" \
    -v "${FRONTIER_CAMPAIGN_WORK_DIR}:/work:rw" \
    -e CUDA_HOME=/usr/local/cuda-13.2 \
    -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e TORCH_EXTENSIONS_DIR=/cache \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /bin/bash \
    "${FRONTIER_CAMPAIGN_IMAGE}" \
    -lc "set -euo pipefail; ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/; export PYTHONPATH=/src; exec /opt/venv/bin/python /opt/frontier/frontier_converter_deterministic.py -i /models/bf16-repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 -w /work/${label} -o /work/${label}-out -b 6 -hb 6 -mb 4 -vb 16 -cr 2 -cc 128 -cpi 0 -cb mcg -d 0 --max_module 2 $*" \
    >"${FRONTIER_CAMPAIGN_WORK_DIR}/${label}.log" 2>&1
}

run_converter a
run_converter b
cp -a "${FRONTIER_CAMPAIGN_WORK_DIR}/a" "${FRONTIER_CAMPAIGN_WORK_DIR}/a-initial"
cp -a "${FRONTIER_CAMPAIGN_WORK_DIR}/a.log" "${FRONTIER_CAMPAIGN_WORK_DIR}/a-initial.log"
run_converter a -r

python3 - "$FRONTIER_CAMPAIGN_WORK_DIR" "$RESULT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tensor_map(directory):
    rows = {}
    for path in sorted(directory.rglob("*"), key=lambda value: value.relative_to(directory).as_posix().encode("utf-8")):
        if path.is_symlink():
            raise SystemExit("converter work contains a symlink")
        if path.is_file() and path.suffix in {".safetensors", ".pt", ".bin"}:
            rows[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    if not rows:
        raise SystemExit("converter control produced no tensor checkpoint files")
    return rows

def log_record(label):
    path = root / (label + ".log")
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    forbidden = [token for token in ("traceback", "cuda out of memory", "non-finite", " nan") if token in lowered]
    if forbidden:
        raise SystemExit("converter log contains forbidden markers: " + ",".join(forbidden))
    return {"path": label + ".log", "bytes": path.stat().st_size, "sha256": sha(path)}

a_initial = tensor_map(root / "a-initial")
b = tensor_map(root / "b")
resumed = tensor_map(root / "a")
if a_initial != b:
    raise SystemExit("independent A/A converter tensor checkpoints differ")
if a_initial != resumed:
    raise SystemExit("resume-path tensor checkpoints differ from the completed control")
value = {
    "schema": "qwen38-frontier-converter-controls/1",
    "status": "pass",
    "source": {"repo": "malaiwah/exllamav3", "commit": "a71fbd8f841fd8772f4a411e43686f15fb16f166", "tree": "2b3a373faa27e8cfc885c0ced715321df0ed6830"},
    "bf16_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "parameters": {"bits": 6, "head_bits": 6, "mtp_bits": 4, "vision_bits": 16, "codebook": "mcg", "cal_rows": 2, "cal_cols": 128, "max_module": 2},
    "tensor_checkpoint_files": len(a),
    "a_a_byte_identical": True,
    "resume_byte_identical": True,
    "tensor_map_sha256": hashlib.sha256(json.dumps(a_initial, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    "logs": [log_record(label) for label in ("a-initial", "b", "a")],
}
payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
tmp = out.with_name(out.name + ".tmp")
with tmp.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, out)
fd = os.open(str(out.parent), os.O_RDONLY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
