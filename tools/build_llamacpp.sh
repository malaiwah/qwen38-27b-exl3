#!/bin/bash
# Build a pinned llama.cpp plus `gguf_capture` so GGUF quants can be measured
# under the same hidden-state protocol as everything else in this repo.
#
# Why a second engine at all: the pinned vLLM build cannot read GGUF (no
# `gguf.py`, `gguf` absent from `QuantizationMethods`, no `gguf` load format),
# and installing a plugin would change the image every other number here was
# measured on.  llama.cpp reads GGUF and exposes the exact tensor the protocol
# needs, so it becomes the capture engine for GGUF candidates only - the replay
# side, the LM head and the suite stay bit-identical.
#
# Nothing here is allowed to float.  The commit is pinned, the working tree must
# be clean, and the content of that tree is re-hashed on every run:
# `git ls-files -s` prints mode, blob sha and path for every tracked file, so its
# digest changes if any byte of the source changes, independent of how the clone
# was packed.  A mismatch aborts before a single object is compiled.
#
# The host has gcc-11 and make but no cmake; the pinned rootfs at /var/tmp/gg-rootfs
# has cmake, gcc-13 and the CUDA 13.2 toolkit, so configure/compile run inside it
# through tools/ggrun.sh.  CUDA is enabled at compile time; this script never runs
# CUDA work.
#
#   tools/build_llamacpp.sh              # build, verify, print digests
#   CUDA_ARCH=120 JOBS=28 tools/build_llamacpp.sh
set -euo pipefail
die() { echo "FAILED: $*" >&2; exit 1; }

REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
PIN_COMMIT=ece963f41b0b02d7a0d61436ae365762c073a4c8            # 2026-08-15, has LLM_ARCH_QWEN35
PIN_TREE=f59cbdf04f233655507cc98ee9f704b71bfd1403              # git tree of PIN_COMMIT
PIN_SRC_DIGEST=a4324243d219e65c7b783f3856bd1f29835ff595566025baf46823a4369f8ab3  # sha256 of `git ls-files -s`

WORK="${GG_WORK:-/var/tmp/work}"
ROOT="${LLAMACPP_ROOT:-$WORK/llamacpp}"
CUDA_ARCH="${CUDA_ARCH:-120}"          # RTX PRO 6000 Blackwell = sm_120
JOBS="${JOBS:-$(nproc)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ggrun binds $GG_WORK to /work, so the build tree has to live under it.
case "$ROOT" in
  "$WORK"/*) GUEST="/work/${ROOT#"$WORK"/}" ;;
  *) die "LLAMACPP_ROOT ($ROOT) must live under GG_WORK ($WORK) to be visible in the rootfs" ;;
esac
SRC="$ROOT/src"
BUILD="$ROOT/build"
mkdir -p "$ROOT"

# ------------------------------------------------------------------ fetch
if [ ! -d "$SRC/.git" ]; then
  echo "== fetching $PIN_COMMIT"
  mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote add origin "$REPO"
  # Fetching the commit itself, not a branch: a branch would move under us.
  git -C "$SRC" fetch -q --depth 1 origin "$PIN_COMMIT"
  git -C "$SRC" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
fi

# --------------------------------------------------------------- verify
origin="$(git -C "$SRC" remote get-url origin)"
[ "$origin" = "$REPO" ] || die "checkout points at $origin, not $REPO"
head="$(git -C "$SRC" rev-parse HEAD)"
[ "$head" = "$PIN_COMMIT" ] || die "checkout is at $head, not the pinned $PIN_COMMIT"
tree="$(git -C "$SRC" rev-parse "HEAD^{tree}")"
[ "$tree" = "$PIN_TREE" ] || die "tree $tree != pinned $PIN_TREE"
dirty="$(git -C "$SRC" status --porcelain)"
[ -z "$dirty" ] || die "working tree is dirty; a patched engine is not this pin:
$dirty"
src_digest="$(git -C "$SRC" ls-files -s | sha256sum | cut -d' ' -f1)"
[ "$src_digest" = "$PIN_SRC_DIGEST" ] || \
  die "source digest $src_digest != pinned $PIN_SRC_DIGEST"
echo "== verified $PIN_COMMIT tree=$tree src_digest=$src_digest (clean)"

# Our capture tool is not part of the pinned tree - it is copied next to it, so
# the tree stays clean and the pin keeps meaning what it says.
cp "$HERE/gguf_capture.cpp" "$ROOT/gguf_capture.cpp"
tool_digest="$(sha256sum "$ROOT/gguf_capture.cpp" | cut -d' ' -f1)"

# ---------------------------------------------------------------- build
ggrun() {
  GG_ENV="PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin CC=gcc-13 CXX=g++-13 CUDACXX=/usr/local/cuda/bin/nvcc" \
  GG_WORK="$WORK" bash "$HERE/ggrun.sh" "$@"
}

CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
  -DGGML_NATIVE=ON
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_SERVER=OFF
)
echo "== configuring (cuda arch $CUDA_ARCH)"
ggrun cmake -S "$GUEST/src" -B "$GUEST/build" "${CMAKE_FLAGS[@]}" >"$ROOT/cmake-configure.log" 2>&1 \
  || { tail -30 "$ROOT/cmake-configure.log"; die "cmake configure (see $ROOT/cmake-configure.log)"; }
echo "== building libllama with $JOBS jobs"
ggrun cmake --build "$GUEST/build" -j "$JOBS" --target llama >"$ROOT/cmake-build.log" 2>&1 \
  || { tail -40 "$ROOT/cmake-build.log"; die "cmake build (see $ROOT/cmake-build.log)"; }

echo "== compiling gguf_capture"
ggrun g++-13 -O3 -std=c++17 -Wall -Wextra \
  -DLLAMACPP_COMMIT="\"$PIN_COMMIT\"" \
  -I "$GUEST/src/include" -I "$GUEST/src/ggml/include" -I "$GUEST/src/vendor" \
  "$GUEST/gguf_capture.cpp" -o "$GUEST/build/bin/gguf_capture" \
  -L "$GUEST/build/bin" -lllama -Wl,-rpath,'$ORIGIN' \
  || die "compiling gguf_capture"

# -------------------------------------------------------------- provenance
BIN="$BUILD/bin/gguf_capture"
[ -x "$BIN" ] || die "gguf_capture was not produced"
mapfile -t artifacts < <(printf '%s\n' "$BIN" "$BUILD"/bin/libllama.so "$BUILD"/bin/libggml*.so)

cmake_ver="$(ggrun cmake --version | head -1)"
gcc_ver="$(ggrun g++-13 --version | head -1)"
nvcc_ver="$(ggrun nvcc --version | tail -2 | head -1)"

python3 - "$ROOT/llamacpp-provenance.json" \
  --flags "${CMAKE_FLAGS[@]}" --artifacts "${artifacts[@]}" <<PROV
import hashlib, json, os, platform, sys, time

out = sys.argv[1]
argv = sys.argv[2:]
flags = argv[argv.index("--flags") + 1:argv.index("--artifacts")]
artifacts = argv[argv.index("--artifacts") + 1:]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

rootfs_prov = "$WORK/gg-rootfs-provenance.json"
payload = {
    "schema": "qwen38-llamacpp-build/1",
    "purpose": "capture engine for GGUF candidates in the hidden-state fidelity protocol",
    "repo": "$REPO",
    "commit": "$PIN_COMMIT",
    "tree": "$tree",
    "source_digest": {"algorithm": "sha256", "over": "git ls-files -s", "value": "$src_digest"},
    "fetch": {"mode": "git fetch --depth 1 <commit>", "detached": True},
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "builder": {
        "rootfs": os.environ.get("GG_ROOTFS", "/var/tmp/gg-rootfs"),
        "rootfs_provenance_sha256": sha256(rootfs_prov) if os.path.isfile(rootfs_prov) else None,
        "cmake": """$cmake_ver""".strip(),
        "cxx": """$gcc_ver""".strip(),
        "nvcc": """$nvcc_ver""".strip(),
        "host_cpu": platform.processor() or platform.machine(),
        "jobs": int("$JOBS"),
    },
    "cmake_flags": flags,
    "cuda_architectures": "$CUDA_ARCH",
    "capture_tool_source": {"path": "tools/gguf_capture.cpp", "sha256": "$tool_digest"},
    "artifacts": {os.path.basename(p): {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}
                  for p in artifacts},
}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(payload, f, indent=2)
os.replace(tmp, out)
print("provenance " + out)
for name, meta in payload["artifacts"].items():
    print(f"{meta['sha256']}  {meta['path']}")
PROV

echo "== run it as:"
echo "   $BIN --model <FILE.gguf> --suite <SUITE_DIR> --out <CAP_DIR> --indices 0-511 --ngl 999 --expect-hidden 5120"
echo LLAMACPPBUILDDONE
