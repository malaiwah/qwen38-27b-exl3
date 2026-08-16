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
# Two build trees, never one.  `capture` is the published tree that produced the
# GGUF hidden-state captures; its artifact digests are quoted in receipts, so it is
# never reconfigured or relinked again.  `tools` is a second tree at the same pin
# with `LLAMA_BUILD_TOOLS=ON`, which is the only way to get `llama-perplexity` -
# the external protocol's harness (docs/35-external-protocol-comparability.md).
# It writes its own block into the same provenance JSON and touches nothing in
# `capture`.
#
#   tools/build_llamacpp.sh              # capture engine: libllama + gguf_capture
#   tools/build_llamacpp.sh tools        # + llama-perplexity, llama-tokenize
#   CUDA_ARCH=120 JOBS=28 tools/build_llamacpp.sh
set -euo pipefail
die() { echo "FAILED: $*" >&2; exit 1; }

MODE="${1:-capture}"
case "$MODE" in
  capture|tools) ;;
  *) die "usage: ${BASH_SOURCE[0]} [capture|tools]" ;;
esac

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
BUILD="$ROOT/build"                    # published capture tree - never relinked
TBUILD="$ROOT/build-tools"             # llama-perplexity tree, same pin, tools on
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
if [ "$MODE" = capture ]; then
  cp "$HERE/gguf_capture.cpp" "$ROOT/gguf_capture.cpp"
  tool_digest="$(sha256sum "$ROOT/gguf_capture.cpp" | cut -d' ' -f1)"
fi

# ---------------------------------------------------------------- build
ggrun() {
  GG_ENV="PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin CC=gcc-13 CXX=g++-13 CUDACXX=/usr/local/cuda/bin/nvcc" \
  GG_WORK="$WORK" bash "$HERE/ggrun.sh" "$@"
}

# ------------------------------------------------- tools tree (Run A harness)
# `llama-perplexity` is the external protocol's entire harness
# (docs/35-external-protocol-comparability.md).  The capture tree was configured
# LLAMA_BUILD_TOOLS=OFF, so the binary does not exist there and cannot be made to
# exist without relinking artifacts whose digests are already published.  Second
# tree, same pin, same compiler and same flags except the tools switch.
if [ "$MODE" = tools ]; then
  TOOL_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
    -DGGML_NATIVE=ON
    -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_SERVER=OFF
  )
  echo "== configuring tools tree (cuda arch $CUDA_ARCH)"
  ggrun cmake -S "$GUEST/src" -B "$GUEST/build-tools" "${TOOL_FLAGS[@]}" \
    >"$ROOT/cmake-configure-tools.log" 2>&1 \
    || { tail -30 "$ROOT/cmake-configure-tools.log"; die "cmake configure (see $ROOT/cmake-configure-tools.log)"; }

  echo "== building llama-perplexity + llama-tokenize with $JOBS jobs"
  ggrun cmake --build "$GUEST/build-tools" -j "$JOBS" \
    --target llama-perplexity llama-tokenize >"$ROOT/cmake-build-tools.log" 2>&1 \
    || { tail -40 "$ROOT/cmake-build-tools.log"; die "cmake build (see $ROOT/cmake-build-tools.log)"; }

  PPL="$TBUILD/bin/llama-perplexity"
  TOK="$TBUILD/bin/llama-tokenize"
  [ -x "$PPL" ] || die "llama-perplexity was not produced"
  [ -x "$TOK" ] || die "llama-tokenize was not produced"

  # Existence check and inventory in one go: `--help` loads the binary and prints
  # the KV-cache types *this* build accepts.  Nothing is inferred and no model is
  # touched.  The advertised list is not the whole truth - `--cache-type-k` will
  # accept nine type names while the CUDA FlashAttention path only has kernels for
  # the K/V pairs actually compiled, which is four unless GGML_CUDA_FA_ALL_QUANTS
  # is ON (ggml/src/ggml-cuda/CMakeLists.txt).  Both lists go in the provenance so
  # a later run cannot silently pick a KV type that has no kernel behind it.
  echo "== llama-perplexity --help (binary existence check, no model)"
  ggrun "$GUEST/build-tools/bin/llama-perplexity" --help >"$ROOT/llama-perplexity-help.txt" 2>&1 \
    || { tail -20 "$ROOT/llama-perplexity-help.txt"; die "llama-perplexity --help"; }

  FA_DIR="$TBUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances"
  mapfile -t fa_pairs < <(
    for o in "$FA_DIR"/fattn-vec-instance-*.cu.o; do
      [ -e "$o" ] || continue
      b="${o##*/}"; b="${b#fattn-vec-instance-}"; echo "${b%.cu.o}"
    done | sort)
  [ "${#fa_pairs[@]}" -gt 0 ] || die "no FlashAttention vec instances found under $FA_DIR"

  mapfile -t tool_artifacts < <(printf '%s\n' "$PPL" "$TOK" "$TBUILD"/bin/*.so)

  cmake_ver="$(ggrun cmake --version | head -1)"
  gcc_ver="$(ggrun g++-13 --version | head -1)"
  nvcc_ver="$(ggrun nvcc --version | tail -2 | head -1)"

  python3 - "$ROOT/llamacpp-provenance.json" "$ROOT/llama-perplexity-help.txt" \
    --flags "${TOOL_FLAGS[@]}" --fa-pairs "${fa_pairs[@]}" --artifacts "${tool_artifacts[@]}" <<PROV
import hashlib, json, os, re, sys, time

out, help_path = sys.argv[1], sys.argv[2]
argv = sys.argv[3:]
flags     = argv[argv.index("--flags") + 1:argv.index("--fa-pairs")]
fa_pairs  = argv[argv.index("--fa-pairs") + 1:argv.index("--artifacts")]
artifacts = argv[argv.index("--artifacts") + 1:]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

if not os.path.isfile(out):
    raise SystemExit("FAILED: %s does not exist; run the capture build first" % out)
prov = json.load(open(out))

# The published capture artifacts must still hash to what the receipts quote.
# This build is only allowed to add a tree, never to disturb that one.
for name, meta in prov["artifacts"].items():
    p = meta["path"]
    if not os.path.isfile(p):
        raise SystemExit("FAILED: published capture artifact %s is gone" % p)
    if sha256(p) != meta["sha256"]:
        raise SystemExit("FAILED: published capture artifact %s changed" % p)

help_text = open(help_path, encoding="utf-8", errors="replace").read()
m = re.search(r"--cache-type-k[^\n]*\n(?:[^\n]*\n)??\s*allowed values:\s*([^\n]+)", help_text)
if not m:
    raise SystemExit("FAILED: could not read the KV type list out of --help")
kv_types = [t.strip() for t in m.group(1).split(",") if t.strip()]

prov["schema"] = "qwen38-llamacpp-build/2"
prov["external_protocol_tools"] = {
    "purpose": "Run A of docs/35-external-protocol-comparability.md: llama-perplexity"
               " --kl-divergence on wikitext-2 raw test at ctx 512",
    "commit": prov["commit"],
    "tree": prov["tree"],
    "source_digest": prov["source_digest"],
    "build_dir": "$TBUILD",
    "separate_tree_because": "the capture build is configured LLAMA_BUILD_TOOLS=OFF and its"
                             " artifact digests are already published; adding tools there would"
                             " relink them",
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "builder": {
        "rootfs": os.environ.get("GG_ROOTFS", "/var/tmp/gg-rootfs"),
        "cmake": """$cmake_ver""".strip(),
        "cxx": """$gcc_ver""".strip(),
        "nvcc": """$nvcc_ver""".strip(),
        "jobs": int("$JOBS"),
    },
    "cmake_flags": flags,
    "cuda_architectures": "$CUDA_ARCH",
    "targets": ["llama-perplexity", "llama-tokenize"],
    "kv_cache": {
        "advertised_types": kv_types,
        "advertised_source": "llama-perplexity --help, --cache-type-k allowed values",
        "ggml_cuda_fa_all_quants": any(f.endswith("GGML_CUDA_FA_ALL_QUANTS=ON") for f in flags),
        "flash_attn_kv_pairs_compiled": fa_pairs,
        "flash_attn_pairs_source": "compiled fattn-vec-instance-*.cu.o objects in the build tree",
        "note": "the CLI accepts every advertised type, but with GGML_CUDA_FA_ALL_QUANTS OFF"
                " (upstream default) only the compiled K/V pairs have a CUDA FlashAttention"
                " kernel. Run A uses the default f16/f16 cache, which is in both lists.",
    },
    "artifacts": {os.path.basename(p): {"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}
                  for p in artifacts},
}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(prov, f, indent=2)
os.replace(tmp, out)
print("provenance " + out)
print("kv types advertised: " + ", ".join(kv_types))
print("flash-attn kv pairs: " + ", ".join(fa_pairs))
for name, meta in prov["external_protocol_tools"]["artifacts"].items():
    print(f"{meta['sha256']}  {meta['path']}")
PROV

  echo "== run it as:"
  echo "   tools/run_wikitext_kld.sh plan"
  echo LLAMACPPTOOLSDONE
  exit 0
fi

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

# A capture rebuild rewrites this file wholesale; the tools tree it knows nothing
# about must survive that, or its digests would silently vanish.
if os.path.isfile(out):
    prev = json.load(open(out))
    if "external_protocol_tools" in prev:
        payload["external_protocol_tools"] = prev["external_protocol_tools"]
        payload["schema"] = "qwen38-llamacpp-build/2"
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
