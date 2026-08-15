#!/bin/bash
# Walk the KLD ladder one shard at a time: capture six models over 512 contexts,
# replay the five candidates against the BF16 reference, verify every report,
# then delete that shard's hidden states before touching the next shard.
#
# Why shards exist: one capture of 512 x 2047 positions x 5120 hidden in bf16 is
# 10.7 GB per model, so a shard of all six models is ~64 GB of scratch and
# /var/tmp holds ~135 GB.  Nothing but one shard of hidden states may be live at
# a time; the reports are the durable artifact and they are three orders of
# magnitude smaller.  `tools/kld_aggregate.py` welds per-shard reports into the
# cumulative receipt at the 1M / 2M / 5M / 10M scored-position checkpoints.
#
# Resume semantics, all fail-closed:
#   * a shard whose five reports exist and verify is skipped without capturing;
#   * an incomplete capture is resumed by the harness itself (identical capture
#     contract) and a capture whose contract or digests moved aborts the run;
#   * a report that describes a different suite, shard, head, or model identity
#     aborts the run instead of being silently reused or overwritten;
#   * candidate/reference identity, head digest, comparator and runtime are
#     pinned on the first verified shard and compared on every later shard.
#
# The per-candidate runtime is the published one (see /var/tmp/work/kld4/cap-*.sh):
# BF16 and the official FP8 checkpoint take no --quantization at all, and every
# exl3 candidate takes `--quantization exl3` with /work/kld4/qcfg.json plus
# VLLM_EXL3_PREFILL_RECONSTRUCT_M=128; K5K6 additionally builds its 6-bit head
# online (VLLM_EXL3_ONLINE_TRELLIS_BITS=6).
#
#   tools/kld_ladder.sh 0            # first shard: 1,048,064 scored positions
#   tools/kld_ladder.sh 1-4          # up to the 5M checkpoint
#   tools/kld_ladder.sh --dry-run 0-9  # build+verify shard suites, print commands
#
# Environment (defaults are the published layout):
#   KLD_ROOT=/var/tmp/work/kld5   KLD_SUITE=$KLD_ROOT/suite   SHARD_SIZE=512
#   KLD_HEAD=/var/tmp/work/kld2/lm_head.safetensors
#   KLD_QCFG=/var/tmp/work/kld4/qcfg.json   KLD_GPU_UTIL=0.85
#   KLD_HARNESS=<host path to fidelity.py>  GG_WORK=/var/tmp/work
set -euo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
WORK=${GG_WORK:-/var/tmp/work}
MODELS_DIR=${GG_MODELS:-/var/tmp/models}
ROOT=${KLD_ROOT:-$WORK/kld5}
SUITE=${KLD_SUITE:-$ROOT/suite}
SHARD_SIZE=${SHARD_SIZE:-512}
HEAD=${KLD_HEAD:-$WORK/kld2/lm_head.safetensors}
QCFG=${KLD_QCFG:-$WORK/kld4/qcfg.json}
GPU_UTIL=${KLD_GPU_UTIL:-0.85}
GGRUN=${GGRUN:-$HERE/ggrun.sh}
PROVENANCE=${GG_PROVENANCE:-$WORK/gg-rootfs-provenance.json}
PY=${PYTHON:-python3}
NAMES="bf16 k4 k5k6 hyd ctx fp8"
CANDIDATES="k4 k5k6 hyd ctx fp8"
DRY=0

die() { echo "FATAL: $*" >&2; exit 1; }

usage() {
  cat >&2 <<'USAGE'
usage: kld_ladder.sh [--dry-run] <shards>

  <shards>   REQUIRED explicit shard list: "0", "1-4", "0,2,5".  There is no
             default: the ladder never guesses how much GPU time to spend.
  --dry-run  build and verify the shard suite views, write the capture/replay
             command files, check free space, print what would run.  No GPU
             work and no deletion.
USAGE
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) usage ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) break ;;
  esac
done
[ $# -eq 1 ] || usage
SHARD_SPEC=$1

# ------------------------------------------------------------------ paths
# Everything the harness touches must live under GG_WORK or GG_MODELS, because
# the pinned rootfs only sees those two trees (as /work and /models).
guest() {
  case "$1" in
    "$WORK"/*) printf '/work/%s' "${1#"$WORK"/}" ;;
    "$MODELS_DIR"/*) printf '/models/%s' "${1#"$MODELS_DIR"/}" ;;
    *) die "$1 is outside GG_WORK ($WORK) and GG_MODELS ($MODELS_DIR); the rootfs cannot see it" ;;
  esac
}

host_model_of() {
  case "$1" in
    bf16) printf '%s' "$MODELS_DIR/Qwen3.8-27B" ;;
    fp8)  printf '%s' "$MODELS_DIR/Qwen3.8-27B-FP8" ;;
    k4)   printf '%s' "$WORK/Qwen3.8-27B-K4" ;;
    k5k6) printf '%s' "$WORK/Qwen3.8-27B-K5K6" ;;
    hyd)  printf '%s' "$WORK/qwen38-hyd" ;;
    ctx)  printf '%s' "$WORK/qwen38-ctx" ;;
    *) die "unknown model name: $1" ;;
  esac
}

env_of() {
  case "$1" in
    bf16|fp8) printf '%s' '' ;;
    hyd|ctx) printf '%s' 'VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' ;;
    k4|k5k6) printf '%s' 'VLLM_EXL3_ONLINE_TRELLIS_BITS=6 VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online VLLM_EXL3_PREFILL_RECONSTRUCT_M=128' ;;
    *) die "unknown model name: $1" ;;
  esac
}

# `$(cat ...)` must reach the generated command file unexpanded: the JSON lost its
# quotes crossing two shells the first time this was tried, so the quantization
# config is read by the guest shell from the file the guest can see.
QCFG_GUEST=$(guest "$QCFG")
EXL3_ARGS='--quantization exl3 --quantization-config "$(cat '"$QCFG_GUEST"')"'
quant_args_of() {
  case "$1" in
    bf16|fp8) printf '%s' '' ;;
    k4|k5k6|hyd|ctx) printf '%s' "$EXL3_ARGS" ;;
    *) die "unknown model name: $1" ;;
  esac
}

# ------------------------------------------------------------------ harness
[ -x "$GGRUN" ] || die "runtime launcher not executable: $GGRUN"
[ -f "$SUITE/suite-manifest.json" ] || die "no suite manifest at $SUITE/suite-manifest.json"
[ -f "$HEAD" ] || die "missing LM head: $HEAD"
[ -f "$QCFG" ] || die "missing quantization config: $QCFG"

REPO_HARNESS=$HERE/fidelity.py
[ -f "$REPO_HARNESS" ] || die "missing $REPO_HARNESS"
REPO_HARNESS_SHA=$(sha256sum "$REPO_HARNESS" | cut -d' ' -f1)
if [ -n "${KLD_HARNESS:-}" ]; then
  HARNESS=$KLD_HARNESS
  [ -f "$HARNESS" ] || die "KLD_HARNESS does not exist: $HARNESS"
elif [ -f "$WORK/kld2/fidelity.py" ] &&
     [ "$(sha256sum "$WORK/kld2/fidelity.py" | cut -d' ' -f1)" = "$REPO_HARNESS_SHA" ]; then
  HARNESS=$WORK/kld2/fidelity.py
else
  # The published measurement CLI lives at /work/kld2/fidelity.py, but the ladder
  # needs the hardened capture contract (resume validation, per-file digests,
  # `complete` flag).  Stage the repository copy next to the ladder rather than
  # overwriting a path other runs share.
  mkdir -p "$ROOT"
  cp -f "$REPO_HARNESS" "$ROOT/fidelity.py"
  HARNESS=$ROOT/fidelity.py
  echo "note: $WORK/kld2/fidelity.py is not the repository harness; staged $HARNESS instead"
fi
for feature in 'def validate_capture_files' 'def capture_contract' 'def atomic_write_json'; do
  grep -q "$feature" "$HARNESS" || die "$HARNESS lacks '$feature'; the ladder needs the fail-closed harness"
done
HARNESS_SHA=$(sha256sum "$HARNESS" | cut -d' ' -f1)
HARNESS_GUEST=$(guest "$HARNESS")
HEAD_GUEST=$(guest "$HEAD")
HEAD_SHA=$(sha256sum "$HEAD" | cut -d' ' -f1)
REF_MODEL_HOST=$(host_model_of bf16)
REF_MODEL_GUEST=$(guest "$REF_MODEL_HOST")
for name in $NAMES; do
  m=$(host_model_of "$name")
  [ -f "$m/config.json" ] || die "candidate $name has no snapshot at $m"
done

mkdir -p "$ROOT/shards" "$ROOT/reports" "$ROOT/hidden" "$ROOT/identity"

# ------------------------------------------------------------------ run pin
# One immutable statement of what this ladder measures with.  A later shard that
# would be measured with a different harness, head, quantization config or
# runtime image aborts instead of quietly joining the same receipt.
pin_run() {
  "$PY" - "$ROOT/ladder-pin.json" "$SUITE/suite-manifest.json" "$PROVENANCE" \
        "$HARNESS_SHA" "$HEAD_SHA" "$QCFG" "$SHARD_SIZE" "$HARNESS_GUEST" <<'PY'
import hashlib, json, os, sys
pin_path, suite_manifest, provenance, harness_sha, head_sha, qcfg, shard_size, harness_guest = sys.argv[1:9]

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

suite = json.loads(open(suite_manifest).read())
image = None
if os.path.isfile(provenance):
    prov = json.loads(open(provenance).read())
    image = {"image": prov.get("image"),
             "file_manifest_sha256": prov.get("file_manifest_sha256")}
pin = {
    "schema": "qwen38-kld-ladder-pin/1",
    "harness_sha256": harness_sha,
    "harness_guest_path": harness_guest,
    "head_sha256": head_sha,
    "quantization_config_sha256": sha_file(qcfg),
    "quantization_config": json.loads(open(qcfg).read()),
    "shard_size": int(shard_size),
    "suite": {
        "schema": suite.get("schema"),
        "suite_token_sha256": suite.get("suite_token_sha256"),
        "contexts": suite.get("contexts"),
        "context_length": suite.get("context_length"),
        "scored_positions_per_context": suite.get("scored_positions_per_context"),
        "manifest_sha256": sha_file(suite_manifest),
    },
    "runtime_image": image,
}
if os.path.isfile(pin_path):
    prior = json.loads(open(pin_path).read())
    if prior != pin:
        drift = [k for k in set(prior) | set(pin) if prior.get(k) != pin.get(k)]
        print(f"ladder pin changed since the first shard: {sorted(drift)}", file=sys.stderr)
        print(f"  stored: {json.dumps({k: prior.get(k) for k in sorted(drift)})}", file=sys.stderr)
        print(f"  now:    {json.dumps({k: pin.get(k) for k in sorted(drift)})}", file=sys.stderr)
        sys.exit(1)
else:
    tmp = pin_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pin, f, indent=2)
    os.replace(tmp, pin_path)
print("pin_ok " + json.dumps({"suite_token_sha256": pin["suite"]["suite_token_sha256"],
                              "contexts": pin["suite"]["contexts"],
                              "harness_sha256": harness_sha[:16]}))
PY
}
pin_run || die "run pin rejected the current configuration"

# ------------------------------------------------------------------ shard list
SHARDS=$("$PY" - "$SHARD_SPEC" "$SUITE/suite-manifest.json" "$SHARD_SIZE" <<'PY'
import json, sys
spec, suite_manifest, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
if size <= 0:
    raise SystemExit("SHARD_SIZE must be positive")
total = len(json.loads(open(suite_manifest).read())["context_index"])
shards = (total + size - 1) // size
out = []
for piece in spec.replace(" ", "").split(","):
    if not piece:
        continue
    if "-" in piece:
        lo, hi = piece.split("-", 1)
        if not lo.isdigit() or not hi.isdigit() or int(hi) < int(lo):
            raise SystemExit(f"bad shard range {piece!r}")
        out.extend(range(int(lo), int(hi) + 1))
    elif piece.isdigit():
        out.append(int(piece))
    else:
        raise SystemExit(f"bad shard index {piece!r}")
seen, ordered = set(), []
for s in out:
    if s in seen:
        raise SystemExit(f"shard {s} listed twice")
    if s >= shards:
        raise SystemExit(f"shard {s} does not exist: {total} contexts / {size} = {shards} shards")
    seen.add(s)
    ordered.append(s)
if not ordered:
    raise SystemExit("empty shard list")
print(" ".join(str(s) for s in ordered))
PY
) || die "bad shard list: $SHARD_SPEC"
echo "== ladder shards: $SHARDS (shard size $SHARD_SIZE, suite $SUITE)"

# ------------------------------------------------------------------ helpers
# Materialise shard S as a self-consistent sub-suite: the parent's context rows
# for that shard only, its own suite_token_sha256 (so capture and replay agree on
# what was measured), a `shard` block naming the parent, and a relative symlink to
# the parent token files so the tokens themselves are never copied or rewritten.
build_shard_suite() {  # shard_index out_dir
  "$PY" - "$SUITE/suite-manifest.json" "$2" "$1" "$SHARD_SIZE" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
parent_manifest, out_dir, shard, size = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

manifest = json.loads(open(parent_manifest).read())
index = manifest["context_index"]
whole = sha_bytes("".join(r["token_sha256"] for r in index).encode())
if manifest.get("suite_token_sha256") != whole:
    raise SystemExit(f"{parent_manifest}: suite_token_sha256 does not match its context hashes")
rows = index[shard * size:(shard + 1) * size]
if not rows:
    raise SystemExit(f"shard {shard} is empty for {len(index)} contexts")
positions_per = manifest["scored_positions_per_context"]
digest = sha_bytes("".join(r["token_sha256"] for r in rows).encode())
view = dict(manifest)
view["context_index"] = rows
view["contexts"] = len(rows)
view["total_scored_positions"] = len(rows) * positions_per
view["strata"] = {}
for r in rows:
    view["strata"][r["stratum"]] = view["strata"].get(r["stratum"], 0) + 1
view["strata"] = dict(sorted(view["strata"].items()))
view["partitions"] = {}
for r in rows:
    key = r.get("partition", "analysis")
    view["partitions"][key] = view["partitions"].get(key, 0) + 1
view["partitions"] = dict(sorted(view["partitions"].items()))
view["source_clusters"] = len({r["source_cluster"] for r in rows})
view["suite_token_sha256"] = digest
view["shard"] = {
    "index": shard,
    "size": size,
    "contexts": len(rows),
    "partial": len(rows) != size,
    "context_indices": [int(r["index"]) for r in rows],
    "parent_suite": str(Path(parent_manifest).parent),
    "parent_suite_token_sha256": whole,
    "parent_contexts": len(index),
    "parent_manifest_sha256": sha_file(parent_manifest),
}
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / "suite-manifest.json"
if path.is_file():
    prior = json.loads(path.read_text())
    if prior.get("suite_token_sha256") != digest or prior.get("shard", {}).get(
            "context_indices") != view["shard"]["context_indices"]:
        raise SystemExit(
            f"{path} describes a different shard than {parent_manifest} shard {shard}; "
            "the parent suite moved under an existing ladder -- use a new KLD_ROOT"
        )
tmp = path.with_name(path.name + ".tmp")
tmp.write_text(json.dumps(view, indent=2))
os.replace(tmp, path)
# tokens are read as `<suite>/tokens/context-XXXX.json`; a relative symlink
# resolves identically on the host and inside the rootfs.
link = out_dir / "tokens"
target = os.path.relpath(Path(parent_manifest).parent / "tokens", out_dir)
if link.is_symlink():
    if os.readlink(link) != target:
        raise SystemExit(f"{link} points at {os.readlink(link)!r}, expected {target!r}")
elif link.exists():
    raise SystemExit(f"{link} exists and is not the expected symlink to {target}")
else:
    link.symlink_to(target)
first = rows[0]
if not (out_dir / first["file"]).is_file():
    raise SystemExit(f"token file unreachable through the shard view: {out_dir / first['file']}")
print("shard_suite " + json.dumps({
    "shard": shard, "contexts": len(rows),
    "context_index_min": int(rows[0]["index"]), "context_index_max": int(rows[-1]["index"]),
    "scored_positions": len(rows) * positions_per, "suite_token_sha256": digest,
    "partial": view["shard"]["partial"]}))
PY
}

require_space() {  # dir bytes_needed
  "$PY" - "$1" "$2" <<'PY'
import os, sys
path, need = sys.argv[1], int(sys.argv[2])
st = os.statvfs(path)
free = st.f_bavail * st.f_frsize
if free < need:
    raise SystemExit(
        f"{path}: {free / 2**30:.1f} GiB free but this shard needs "
        f"{need / 2**30:.1f} GiB of hidden states"
    )
print(f"space_ok {free / 2**30:.1f} GiB free, {need / 2**30:.1f} GiB required")
PY
}

# exit 0 = usable, 3 = absent or incomplete (capture again), 1 = abort
verify_capture() {  # capture_dir shard_suite_dir
  "$PY" - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
capture, suite = Path(sys.argv[1]), Path(sys.argv[2])
view = json.loads((suite / "suite-manifest.json").read_text())
want = {int(i) for i in view["shard"]["context_indices"]}
rows = view["scored_positions_per_context"]
hidden = view.get("hidden_size")
path = capture / "capture-manifest.json"
if not path.is_file():
    print(f"capture absent: {path}")
    sys.exit(3)
manifest = json.loads(path.read_text())
if manifest.get("suite_token_sha256") != view["suite_token_sha256"]:
    print(f"{path}: suite identity does not match the shard view", file=sys.stderr)
    sys.exit(1)
records = {int(r["index"]): r for r in manifest.get("captures", []) if "index" in r}
if len(records) != len(manifest.get("captures", [])):
    print(f"{path}: duplicate capture records", file=sys.stderr)
    sys.exit(1)
stray = sorted(set(records) - want)
if stray:
    print(f"{path}: records outside the shard: {stray[:8]}", file=sys.stderr)
    sys.exit(1)
for index, record in records.items():
    shape = record.get("shape")
    expected = [rows, hidden] if hidden else None
    ok = shape == expected if expected is not None else (bool(shape) and shape[0] == rows)
    if not ok:
        print(f"{path}: context {index} has shape {shape}, expected {expected or [rows, '*']}",
              file=sys.stderr)
        sys.exit(1)
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        print(f"{path}: context {index} has no digest", file=sys.stderr)
        sys.exit(1)
    file = capture / f"hidden_{index:04d}.safetensors"
    if not file.is_file() or file.stat().st_size < rows * (hidden or 1) * 2:
        print(f"{path}: capture file missing or short: {file}", file=sys.stderr)
        sys.exit(1)
missing = sorted(want - set(records))
if manifest.get("complete") is not True or missing:
    print(f"capture incomplete: {len(records)}/{len(want)} contexts in {capture}")
    sys.exit(3)
print(f"capture ok: {len(records)} contexts in {capture}")
PY
}

# exit 0 = verified, 3 = absent (replay again), 1 = abort
verify_report() {  # report_path shard_suite_dir candidate_guest_model identity_pin
  "$PY" - "$1" "$2" "$3" "$4" "$REF_MODEL_GUEST" "$HEAD_GUEST" "$HEAD_SHA" <<'PY'
import json, math, os, sys
from pathlib import Path
report_path, suite, cand_model, pin_path, ref_model, head, head_sha = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
    sys.argv[5], sys.argv[6], sys.argv[7])
view = json.loads((suite / "suite-manifest.json").read_text())
want = {int(i) for i in view["shard"]["context_indices"]}
rows_per = view["scored_positions_per_context"]
if not report_path.is_file():
    print(f"report absent: {report_path}")
    sys.exit(3)
try:
    report = json.loads(report_path.read_text())
except ValueError as exc:
    print(f"{report_path}: unreadable report ({exc})", file=sys.stderr)
    sys.exit(1)

bad = []
if report.get("schema") != "qwen38-fidelity-report/1":
    bad.append(f"schema {report.get('schema')!r}")
if report.get("suite_token_sha256") != view["suite_token_sha256"]:
    bad.append("suite_token_sha256 does not match the shard view")
if report.get("filter") not in ("all", ""):
    bad.append(f"filter {report.get('filter')!r} is not the whole shard")
if report.get("contexts") != len(want):
    bad.append(f"contexts {report.get('contexts')!r} != {len(want)}")
if report.get("scored_positions") != len(want) * rows_per:
    bad.append(f"scored_positions {report.get('scored_positions')!r} != {len(want) * rows_per}")
rows = report.get("per_context_all") or []
got = {int(r["index"]) for r in rows}
if got != want:
    bad.append(f"per-context rows cover {len(got)} contexts, not this shard's {len(want)}")
if any(int(r["positions_scored"]) != rows_per for r in rows):
    bad.append("a per-context row does not score the full context")
if (report.get("reference_identity") or {}).get("model_path") != ref_model:
    bad.append(f"reference is {(report.get('reference_identity') or {}).get('model_path')!r}, "
               f"expected {ref_model!r}")
if (report.get("candidate_identity") or {}).get("model_path") != cand_model:
    bad.append(f"candidate is {(report.get('candidate_identity') or {}).get('model_path')!r}, "
               f"expected {cand_model!r}")
if report.get("head") != head or report.get("head_sha256") != head_sha:
    bad.append("head path or digest is not the pinned BF16 head")
if report.get("candidate_head") is not None:
    bad.append("candidate_head must be null: the head is shared, only the body differs")
mean = report.get("token_mean_kld")
top1 = report.get("top1_agreement")
if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < 0:
    bad.append(f"token_mean_kld {mean!r}")
if not isinstance(top1, (int, float)) or not 0.0 <= top1 <= 1.0:
    bad.append(f"top1_agreement {top1!r}")
if rows and isinstance(mean, (int, float)):
    positions = sum(int(r["positions_scored"]) for r in rows)
    weighted = math.fsum(float(r["mean_kld"]) * int(r["positions_scored"]) for r in rows) / positions
    scale = max(abs(weighted), abs(mean))
    if scale and abs(weighted - mean) / scale > 1e-6:
        bad.append(f"token_mean_kld {mean!r} disagrees with its rows ({weighted!r})")
if bad:
    print(f"{report_path}:", file=sys.stderr)
    for item in bad:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)

identity = {
    "reference_identity": report.get("reference_identity"),
    "candidate_identity": report.get("candidate_identity"),
    "head": report.get("head"),
    "head_sha256": report.get("head_sha256"),
    "candidate_head": report.get("candidate_head"),
    "filter": report.get("filter"),
    "vocab_size": report.get("vocab_size"),
    "hidden_size": report.get("hidden_size"),
    "comparator": report.get("comparator"),
}
if pin_path.is_file():
    prior = json.loads(pin_path.read_text())
    if prior != identity:
        drift = sorted(k for k in set(prior) | set(identity) if prior.get(k) != identity.get(k))
        print(f"{report_path}: identity drifted from {pin_path}: {drift}", file=sys.stderr)
        print(f"  stored: {json.dumps({k: prior.get(k) for k in drift})}", file=sys.stderr)
        print(f"  now:    {json.dumps({k: identity.get(k) for k in drift})}", file=sys.stderr)
        sys.exit(1)
else:
    tmp = pin_path.with_name(pin_path.name + ".tmp")
    tmp.write_text(json.dumps(identity, indent=2))
    os.replace(tmp, pin_path)
print("report ok " + json.dumps({
    "contexts": report["contexts"], "scored_positions": report["scored_positions"],
    "token_mean_kld": report["token_mean_kld"], "top1_agreement": report["top1_agreement"]}))
PY
}

status_of() {  # runs a verifier, echoes its stdout, returns 0/3, aborts on 1
  local rc=0
  "$@" || rc=$?
  case $rc in
    0) return 0 ;;
    3) return 3 ;;
    *) die "verification aborted (see above)" ;;
  esac
}

# ------------------------------------------------------------------ the ladder
TOTAL_POSITIONS=0
for S in $SHARDS; do
  TAG=$(printf 'shard-%04d' "$S")
  SDIR=$ROOT/shards/$TAG
  RDIR=$ROOT/reports/$TAG
  HDIR=$ROOT/hidden/$TAG
  mkdir -p "$SDIR" "$RDIR"
  SUITE_GUEST=$(guest "$SDIR/suite")
  RDIR_GUEST=$(guest "$RDIR")
  HDIR_GUEST=$(guest "$HDIR")

  echo "== $TAG"
  build_shard_suite "$S" "$SDIR/suite" || die "$TAG: shard suite view rejected"

  SHARD_CONTEXTS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["contexts"])' \
                   "$SDIR/suite/suite-manifest.json")
  SHARD_POSITIONS=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["contexts"]*m["scored_positions_per_context"])' \
                    "$SDIR/suite/suite-manifest.json")
  SHARD_BYTES=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["contexts"]*m["scored_positions_per_context"]*m["hidden_size"]*2)' \
                "$SDIR/suite/suite-manifest.json")

  # write the command files first: they are the auditable record of the runtime
  for name in $NAMES; do
    MODEL_GUEST=$(guest "$(host_model_of "$name")")
    ENVS=$(env_of "$name")
    QARGS=$(quant_args_of "$name")
    cat > "$SDIR/cap-$name.sh" <<EOF
set -uo pipefail
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext $ENVS
exec python $HARNESS_GUEST capture --model $MODEL_GUEST --suite $SUITE_GUEST \\
  --out $HDIR_GUEST/hidden-$name --gpu-memory-utilization $GPU_UTIL $QARGS
EOF
  done
  for cand in $CANDIDATES; do
    cat > "$SDIR/replay-$cand.sh" <<EOF
set -uo pipefail
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TORCH_EXTENSIONS_DIR=/work/torch-ext
exec python $HARNESS_GUEST replay --reference $HDIR_GUEST/hidden-bf16 \\
  --candidate $HDIR_GUEST/hidden-$cand --head $HEAD_GUEST --suite $SUITE_GUEST \\
  --out $RDIR_GUEST/report-$cand.json
EOF
  done

  # already finished?  every report must verify, or the shard is redone.
  READY=1
  for cand in $CANDIDATES; do
    if ! status_of verify_report "$RDIR/report-$cand.json" "$SDIR/suite" \
                   "$(guest "$(host_model_of "$cand")")" "$ROOT/identity/$cand.json"; then
      READY=0
    fi
  done
  if [ "$READY" = 1 ]; then
    echo "-- $TAG already verified: $SHARD_CONTEXTS contexts, $SHARD_POSITIONS scored positions"
    rm -rf "$HDIR"
    TOTAL_POSITIONS=$((TOTAL_POSITIONS + SHARD_POSITIONS))
    continue
  fi

  require_space "$ROOT" "$((SHARD_BYTES * 6 + SHARD_BYTES / 2))" \
    || die "$TAG: not enough scratch for six captures of this shard"

  if [ "$DRY" = 1 ]; then
    echo "-- dry run: would capture $NAMES then replay $CANDIDATES"
    for name in $NAMES; do echo "   $GGRUN bash $(guest "$SDIR")/cap-$name.sh"; done
    for cand in $CANDIDATES; do echo "   $GGRUN bash $(guest "$SDIR")/replay-$cand.sh"; done
    echo "   rm -rf $HDIR"
    TOTAL_POSITIONS=$((TOTAL_POSITIONS + SHARD_POSITIONS))
    continue
  fi

  mkdir -p "$HDIR"
  for name in $NAMES; do
    if status_of verify_capture "$HDIR/hidden-$name" "$SDIR/suite"; then
      echo "-- capture $name: reusing verified hidden states"
      continue
    fi
    echo "-- capture $name ($SHARD_CONTEXTS contexts)"
    "$GGRUN" bash "$(guest "$SDIR")/cap-$name.sh" || die "$TAG: capture $name failed"
    status_of verify_capture "$HDIR/hidden-$name" "$SDIR/suite" \
      || die "$TAG: capture $name did not produce the whole shard"
  done

  for cand in $CANDIDATES; do
    if status_of verify_report "$RDIR/report-$cand.json" "$SDIR/suite" \
                 "$(guest "$(host_model_of "$cand")")" "$ROOT/identity/$cand.json"; then
      echo "-- replay $cand: report already verified"
      continue
    fi
    echo "-- replay $cand vs bf16"
    "$GGRUN" bash "$(guest "$SDIR")/replay-$cand.sh" || die "$TAG: replay $cand failed"
    status_of verify_report "$RDIR/report-$cand.json" "$SDIR/suite" \
              "$(guest "$(host_model_of "$cand")")" "$ROOT/identity/$cand.json" \
      || die "$TAG: replay $cand produced no verifiable report"
  done

  # the reports are verified, so the 64 GB of hidden states are now disposable
  echo "-- $TAG verified; releasing hidden states"
  rm -rf "$HDIR"
  TOTAL_POSITIONS=$((TOTAL_POSITIONS + SHARD_POSITIONS))
done

echo "== ladder shards $SHARDS complete: $TOTAL_POSITIONS scored positions in this run"
echo "== cumulative receipt over EVERY verified shard under $ROOT/reports (add --shards to"
echo "   aggregate a deliberate subset; the glob below must be a gap-free run from shard 0):"
for cand in $CANDIDATES; do
  printf '   %s aggregate --suite %s --shard-size %s --candidate %s \\\n' \
    "$HERE/kld_aggregate.py" "$SUITE" "$SHARD_SIZE" "$cand"
  printf '     --out receipts/kld5-%s.json %s/reports/shard-*/report-%s.json\n' \
    "$cand" "$ROOT" "$cand"
done
