#!/bin/bash
# Run A of docs/35-external-protocol-comparability.md: the external GGUF protocol,
# executed exactly as its authors execute it, so our rows can be cross-cited
# against theirs and against any third-party `llama-perplexity` row.
#
# Their protocol is stock llama.cpp: capture full log-probabilities from the
# unquantized BF16 GGUF with `llama-perplexity --save-all-logits`, then score each
# quant against that file with `--kl-divergence`, on WikiText-2 raw test at
# `--ctx-size 512` [U-LOG-B, U-LOG-Q5].  Nothing here is our protocol: no v5 suite,
# no shared BF16 head, no float64 replay.  That comparison is Run B and is already
# published in receipts/cross-engine-comparator.json.
#
# Everything that can be checked without a GPU is checked before the GPU is asked
# for anything:
#
#   * every downloaded byte is size- and sha256-verified against the pinned
#     revisions (the corpus zip and its member, three quants, both BF16 parts);
#   * the corpus is tokenised with `llama-tokenize` - vocabulary only, CPU only,
#     no weights and no inference - and the chunk arithmetic that the whole run
#     depends on must come out at exactly 580 chunks / 296,960 processed tokens /
#     147,900 scored positions, or this script refuses to continue;
#   * every command-line flag it is about to emit must appear in the built
#     binary's own `--help`, because a flag that silently does not exist is a
#     protocol difference nobody would notice in the output.
#
# After the reference capture the base logits file must be exactly
# 73,455,427,060 B.  That number is `20 + 580*512*4 + 580*255*248324*2`: an 8-byte
# `_logits_` magic plus n_ctx, n_vocab and n_chunk; then every processed token id;
# then 255 scored positions per chunk at `2*((n_vocab+1)/2)+4` uint16 log-probs
# each [LC:461-527, LC:1752].  Chunk count, window geometry, token count and
# vocabulary all appear in that one integer, so byte identity is the cheapest
# proof the protocol matched - and it costs nothing to check.
#
#   tools/run_wikitext_kld.sh              # print the plan, touch nothing
#   tools/run_wikitext_kld.sh fetch        # download + verify (network, no GPU)
#   tools/run_wikitext_kld.sh preflight    # tokenise + assert (CPU, no GPU)
#   KLD_GPU_OK=1 tools/run_wikitext_kld.sh run    # the GPU run.  Refuses without
#     that variable, refuses if the pre-flight arithmetic is off, and refuses if
#     another job already holds the card.  Resumable: a base logits file already
#     at the right size is not recaptured.
set -euo pipefail
die() { echo "FAILED: $*" >&2; exit 1; }
say() { echo "== $*"; }

PHASE="${1:-plan}"
case "$PHASE" in plan|fetch|preflight|tokencheck|run|release) ;; *) die "usage: ${BASH_SOURCE[0]} [plan|fetch|preflight|tokencheck|run|release]" ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"
RECEIPTS="${KLD_RECEIPTS:-$REPO_DIR/receipts}"   # override only to rehearse off to the side
WORK="${GG_WORK:-/var/tmp/work}"
ROOT="${WIKITEXT_KLD_ROOT:-$WORK/wikitext-kld}"
LLAMACPP_ROOT="${LLAMACPP_ROOT:-$WORK/llamacpp}"

# ggrun binds $GG_WORK to /work; every path the binary sees has to be under it.
guest_path() {
  case "$1" in
    "$WORK"/*) echo "/work/${1#"$WORK"/}" ;;
    *) die "$1 must live under GG_WORK ($WORK) to be visible in the pinned rootfs" ;;
  esac
}
G_ROOT="$(guest_path "$ROOT")"
G_BIN="$(guest_path "$LLAMACPP_ROOT")/build-tools/bin"
BIN="$LLAMACPP_ROOT/build-tools/bin"
PROVENANCE="$LLAMACPP_ROOT/llamacpp-provenance.json"

CORPUS="$ROOT/corpus"
MODELS="$ROOT/models"
LOGS="$ROOT/logs"
LOGITS="$ROOT/logits/pipeline_base_logits.bin"
[ "$PHASE" = plan ] || mkdir -p "$RECEIPTS"

# ------------------------------------------------------------ pinned artifacts
# docs/35-external-protocol-comparability.md, "Pinned artifacts" [HF-WT2, HF-GGUF].
WT2_URL=https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip
WT2_ZIP_BYTES=4721645
WT2_ZIP_SHA=ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11
WT2_MEMBER=wikitext-2-raw/wiki.test.raw
WT2_BYTES=1290590
WT2_SHA=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08

GGUF_REPO=unsloth/Qwen3.8-27B-GGUF
GGUF_REV=f1bfb127c64f7072bdd2cad55f258b9c8b2910fe
# tag | repo path | bytes | sha256
ARTIFACTS=(
  "bf16-1|BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf|49986159616|b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f"
  "bf16-2|BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf|4671576000|92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02"
  "q8_0|Qwen3.8-27B-Q8_0.gguf|29047086048|a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348"
  "q6_k|Qwen3.8-27B-Q6_K.gguf|22884408288|562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727"
  "q5_k_xl|Qwen3.8-27B-UD-Q5_K_XL.gguf|20218178624|176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab"
)
REFERENCE=bf16-1                        # llama.cpp opens part 1 and finds part 2
CANDIDATES=(q8_0 q6_k q5_k_xl)

field() { echo "$1" | cut -d'|' -f"$2"; }
spec_for() {
  local tag=$1 spec
  for spec in "${ARTIFACTS[@]}"; do
    [ "$(field "$spec" 1)" = "$tag" ] && { echo "$spec"; return; }
  done
  die "no such artifact: $tag"
}
rel_for() { field "$(spec_for "$1")" 2; }

# ------------------------------------------------------- protocol arithmetic
# Fixed by the corpus, the tokenizer and n_ctx; asserted, never assumed.
N_CTX=512
N_VOCAB=248320                          # Qwen3.8 vocabulary
NV=$(( 2 * ((N_VOCAB + 1) / 2) + 4 ))   # 248324, the uint16 stride [LC:1752]
N_CHUNK=580                             # floor(n_tokens / n_ctx)      [LC:493]
N_PROCESSED=$(( N_CHUNK * N_CTX ))      # 296,960 token ids in the file
FIRST=$(( N_CTX / 2 ))                  # scoring starts here          [LC:541]
PER_CHUNK=$(( N_CTX - 1 - FIRST ))      # 255 scored positions per chunk
N_SCORED=$(( N_CHUNK * PER_CHUNK ))     # 147,900
BASE_LOGITS_BYTES=$(( 20 + N_PROCESSED * 4 + N_SCORED * NV * 2 ))   # 73,455,427,060
REF_TOKENS=297193                       # our own offline count [INV]; band-checked

PPL_TXT="$CORPUS/wiki.test.raw.perplexity-input"
TOKENS_JSON="$ROOT/wiki.test.raw.token-ids.json"

# ------------------------------------------------------------ command sequence
# Built once and used for both printing and running, so the printed plan can
# never drift from what actually executes.  Verbatim from [U-LOG-B] except the
# device (one card here, not eight) and the paths.
PPL_COMMON=(
  --flash-attn on --fit off
  --batch-size 16384 --ubatch-size 16384 --parallel 1
  --mlock --no-mmap
  --device "${KLD_DEVICE:-CUDA0}"
  --ctx-size "$N_CTX"
  --file "$G_ROOT/corpus/wiki.test.raw.perplexity-input"
)
G_LOGITS="$G_ROOT/logits/pipeline_base_logits.bin"

argv_base() {
  printf '%s\0' "$G_BIN/llama-perplexity" "${PPL_COMMON[@]}" \
    --model "$G_ROOT/models/$(rel_for "$REFERENCE")" \
    --save-all-logits "$G_LOGITS"
}
argv_candidate() {
  printf '%s\0' "$G_BIN/llama-perplexity" "${PPL_COMMON[@]}" \
    --model "$G_ROOT/models/$(rel_for "$1")" \
    --kl-divergence --kl-divergence-base "$G_LOGITS"
}
argv_tokenize() {
  printf '%s\0' "$G_BIN/llama-tokenize" \
    --model "$G_ROOT/models/$(rel_for "$REFERENCE")" \
    --file "$G_ROOT/corpus/wiki.test.raw.perplexity-input" \
    --ids --show-count --verbose
}

# LLAMA_SET_ROWS=1 is in their published command line and no longer read at this
# commit; it is kept so the two command lines are literally comparable.
GPU_ENV="GG_ENV=LLAMA_SET_ROWS=1"
ggrun_ppl() { GG_ENV="LLAMA_SET_ROWS=1" GG_WORK="$WORK" bash "$HERE/ggrun.sh" "$@"; }
# Tokenising needs the vocabulary only; -1 removes every device so the claim
# "no GPU was touched" is enforced rather than asserted.
CPU_ENV="CUDA_VISIBLE_DEVICES=-1"
ggrun_cpu() { CUDA_VISIBLE_DEVICES=-1 GG_WORK="$WORK" bash "$HERE/ggrun.sh" "$@"; }

print_cmd() {   # env-prefix argv... -- renders exactly what the phase invokes
  local env=$1; shift
  printf '  GG_WORK=%q %s %s/ggrun.sh' "$WORK" "$env" "$HERE"
  printf ' %q' "$@"
  printf '\n'
}

# ------------------------------------------------------------------ verifying
gib() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1073741824}'; }
gb()  { awk -v b="$1" 'BEGIN{printf "%.2f", b/1e9}'; }

stamp_of() { echo "$1.verified"; }

verify_file() {   # path bytes sha256 -- hashes, then leaves a stamp
  local path=$1 want_bytes=$2 want_sha=$3 have_bytes have_sha
  [ -f "$path" ] || die "missing $path"
  have_bytes=$(stat -c %s "$path")
  [ "$have_bytes" = "$want_bytes" ] || die "$path is $have_bytes B, pinned at $want_bytes B"
  have_sha=$(sha256sum "$path" | cut -d' ' -f1)
  [ "$have_sha" = "$want_sha" ] || die "$path sha256 $have_sha != pinned $want_sha"
  printf '%s %s %s\n' "$want_sha" "$want_bytes" "$(stat -c %Y "$path")" >"$(stamp_of "$path")"
  echo "  ok  $want_sha  $(printf '%14d' "$want_bytes") B  ${path##*/}"
}

require_verified() {   # path bytes sha256 -- cheap re-check against the stamp
  local path=$1 want_bytes=$2 want_sha=$3 stamp sha bytes mtime
  stamp="$(stamp_of "$path")"
  [ -f "$path" ] || die "missing $path -- run: ${BASH_SOURCE[0]} fetch"
  [ -f "$stamp" ] || die "$path was never verified -- run: ${BASH_SOURCE[0]} fetch"
  read -r sha bytes mtime <"$stamp"
  [ "$sha" = "$want_sha" ] && [ "$bytes" = "$want_bytes" ] \
    || die "$path stamp does not match the pinned digest -- run: ${BASH_SOURCE[0]} fetch"
  [ "$(stat -c %s "$path")" = "$want_bytes" ] && [ "$(stat -c %Y "$path")" = "$mtime" ] \
    || die "$path changed since it was verified -- run: ${BASH_SOURCE[0]} fetch"
}

download() {   # url path bytes -- resumable, skips a file already at full size
  local url=$1 path=$2 want_bytes=$3
  mkdir -p "$(dirname "$path")"
  if [ -f "$path" ] && [ "$(stat -c %s "$path")" = "$want_bytes" ]; then
    echo "  have  ${path##*/}"
    return
  fi
  echo "  get   ${path##*/}  ($(gib "$want_bytes") GiB)"
  curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors -C - -o "$path" "$url" \
    || die "download failed: $url"
}

# --------------------------------------------------------------- disk footprint
declare -A FOOTPRINT_BYTES FOOTPRINT_WHAT
footprint() {
  local models=0 spec
  for spec in "${ARTIFACTS[@]}"; do models=$(( models + $(field "$spec" 3) )); done
  FOOTPRINT_WHAT[models]="five GGUFs (three targets + two-part BF16 reference)"
  FOOTPRINT_BYTES[models]=$models
  FOOTPRINT_WHAT[logits]="base logits (llama-perplexity --save-all-logits)"
  FOOTPRINT_BYTES[logits]=$BASE_LOGITS_BYTES
  FOOTPRINT_WHAT[corpus]="wikitext-2-raw-v1.zip + wiki.test.raw + perplexity input"
  FOOTPRINT_BYTES[corpus]=$(( WT2_ZIP_BYTES + WT2_BYTES + WT2_BYTES ))
  FOOTPRINT_WHAT[misc]="token-id dump, four run logs, receipt"
  FOOTPRINT_BYTES[misc]=$(( 16 * 1024 * 1024 ))
  HIGH_WATER=0
  local k
  for k in models logits corpus misc; do HIGH_WATER=$(( HIGH_WATER + FOOTPRINT_BYTES[$k] )); done
}

# ==================================================================== phases
phase_plan() {
  footprint
  local avail; avail=$(df -B1 --output=avail "$WORK" | tail -1)
  cat <<EOF
Run A - external-protocol cross-citation, docs/35-external-protocol-comparability.md
  engine    llama.cpp @ ece963f41b0b02d7a0d61436ae365762c073a4c8, tools tree
  corpus    WikiText-2 raw test, $WT2_BYTES B, sha256 ${WT2_SHA:0:16}...
  models    $GGUF_REPO @ $GGUF_REV
  geometry  n_ctx=$N_CTX, $N_CHUNK chunks, $N_PROCESSED processed, $N_SCORED scored, vocab $N_VOCAB

Command sequence (nothing below is executed by this phase):

  # 0. one-time build - compilation only, no model is ever loaded
  $HERE/build_llamacpp.sh tools

  # 1. download and digest-verify the corpus and all five GGUFs
  ${BASH_SOURCE[0]} fetch

  # 2. CPU-only pre-flight: tokenise, assert the chunk arithmetic, check every flag
  ${BASH_SOURCE[0]} preflight
EOF
  local -a a
  mapfile -d '' a < <(argv_tokenize)
  echo "  #    which runs, vocabulary-only and with CUDA_VISIBLE_DEVICES=-1:"
  print_cmd "$CPU_ENV" "${a[@]}"
  echo
  echo "  # 3. reference capture (GPU): BF16 GGUF -> base logits"
  mapfile -d '' a < <(argv_base)
  print_cmd "$GPU_ENV" "${a[@]}"
  echo
  echo "  #    then the byte identity that proves the protocol matched:"
  echo "  test \$(stat -c %s $LOGITS) -eq $BASE_LOGITS_BYTES"
  echo
  echo "  # 4. score each target against those logits (GPU)"
  local tag
  for tag in "${CANDIDATES[@]}"; do
    mapfile -d '' a < <(argv_candidate "$tag")
    print_cmd "$GPU_ENV" "${a[@]}"
  done
  cat <<EOF

  # 5. all of 3 and 4, gated on 2, with the receipt written:
  KLD_GPU_OK=1 ${BASH_SOURCE[0]} run

  # 6. once the receipt exists, hand the disk back (each scope is separate, and
  #    the blobs need KLD_MIRROR=<repo>@<commit> naming an archival mirror):
  KLD_RELEASE_OK=1 RELEASE_SCOPE=logits ${BASH_SOURCE[0]} release
  KLD_RELEASE_OK=1 RELEASE_SCOPE=blobs KLD_MIRROR=<repo>@<commit> ${BASH_SOURCE[0]} release

Disk high-water mark on $WORK (serialized bytes on disk - not VRAM, not KV):
EOF
  local k
  for k in models logits corpus misc; do
    printf '  %14d B  %8s GiB  %s\n' \
      "${FOOTPRINT_BYTES[$k]}" "$(gib "${FOOTPRINT_BYTES[$k]}")" "${FOOTPRINT_WHAT[$k]}"
  done
  printf '  %14d B  %8s GiB  %s GB  TOTAL high-water (nothing is deleted mid-run)\n' \
    "$HIGH_WATER" "$(gib "$HIGH_WATER")" "$(gb "$HIGH_WATER")"
  printf '  %14d B  %8s GiB  free now on %s\n' "$avail" "$(gib "$avail")" "$WORK"
  if [ "$avail" -lt "$HIGH_WATER" ]; then
    printf '  SHORT by %s GiB\n' "$(gib $(( HIGH_WATER - avail )))"
  else
    printf '  headroom %s GiB\n' "$(gib $(( avail - HIGH_WATER )))"
  fi
}

phase_fetch() {
  mkdir -p "$CORPUS" "$MODELS" "$LOGS" "$(dirname "$LOGITS")"
  say "corpus"
  download "$WT2_URL" "$CORPUS/wikitext-2-raw-v1.zip" "$WT2_ZIP_BYTES"
  verify_file "$CORPUS/wikitext-2-raw-v1.zip" "$WT2_ZIP_BYTES" "$WT2_ZIP_SHA"
  # Only the member the protocol reads, so the tree stays exactly as pinned.
  unzip -o -q "$CORPUS/wikitext-2-raw-v1.zip" "$WT2_MEMBER" -d "$CORPUS"
  verify_file "$CORPUS/$WT2_MEMBER" "$WT2_BYTES" "$WT2_SHA"

  # `llama-perplexity -f` drops one trailing newline before tokenising
  # [LC:common/arg.cpp:1791-1800].  `llama-tokenize -f` does not.  Feeding both
  # the same pre-stripped bytes is the only way the pre-flight count is the count
  # the GPU run will see.
  python3 - "$CORPUS/$WT2_MEMBER" "$PPL_TXT" <<'PY'
import sys
raw = open(sys.argv[1], "rb").read()
open(sys.argv[2], "wb").write(raw[:-1] if raw.endswith(b"\n") else raw)
PY
  echo "  made  ${PPL_TXT##*/}  $(stat -c %s "$PPL_TXT") B  (one trailing newline dropped, as -f does)"

  say "models  $GGUF_REPO @ $GGUF_REV"
  local spec tag rel bytes sha pids=() jobs="${FETCH_JOBS:-3}"
  for spec in "${ARTIFACTS[@]}"; do
    rel=$(field "$spec" 2); bytes=$(field "$spec" 3)
    while [ "${#pids[@]}" -ge "$jobs" ]; do wait -n; pids=($(jobs -pr)); done
    download "https://huggingface.co/$GGUF_REPO/resolve/$GGUF_REV/$rel" "$MODELS/$rel" "$bytes" &
    pids+=($!)
  done
  wait
  for spec in "${ARTIFACTS[@]}"; do
    rel=$(field "$spec" 2); bytes=$(field "$spec" 3); sha=$(field "$spec" 4)
    verify_file "$MODELS/$rel" "$bytes" "$sha"
  done

  local out="$RECEIPTS/wikitext-kld-fetch-verify.json"
  python3 - "$out" "$CORPUS" "$MODELS" "$PPL_TXT" --artifacts "${ARTIFACTS[@]}" <<PROV
import json, os, sys, time
out, corpus, models, ppl_txt = sys.argv[1:5]
specs = sys.argv[sys.argv.index("--artifacts") + 1:]
rows = []
for s in specs:
    tag, rel, nbytes, sha = s.split("|")
    p = os.path.join(models, rel)
    rows.append({"tag": tag, "repo_path": rel, "path": p,
                 "bytes": int(nbytes), "sha256": sha,
                 "bytes_on_disk": os.path.getsize(p), "verified": True})
payload = {
    "schema": "qwen38-external-protocol-fetch/1",
    "purpose": "digest-verified inputs for Run A of docs/35-external-protocol-comparability.md",
    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "corpus": {
        "source": "$WT2_URL",
        "archive": {"path": os.path.join(corpus, "wikitext-2-raw-v1.zip"),
                    "bytes": $WT2_ZIP_BYTES, "sha256": "$WT2_ZIP_SHA", "verified": True},
        "member": {"name": "$WT2_MEMBER",
                   "path": os.path.join(corpus, "$WT2_MEMBER"),
                   "bytes": $WT2_BYTES, "sha256": "$WT2_SHA", "verified": True},
        "perplexity_input": {
            "path": ppl_txt, "bytes": os.path.getsize(ppl_txt),
            "derivation": "wiki.test.raw with a single trailing newline removed, which is what"
                          " llama-perplexity -f tokenises (common/arg.cpp:1791-1800)"},
    },
    "models": {"repo": "$GGUF_REPO", "revision": "$GGUF_REV", "files": rows},
    "retention": {
        "policy": "preserve before delete",
        "this_pipeline_deletes_nothing": True,
        "bytes_on_disk": sum(r["bytes"] for r in rows) + $WT2_ZIP_BYTES + $WT2_BYTES + os.path.getsize(ppl_txt),
        "recovery": "re-obtainable, not recomputable: every file above is a pinned public blob and"
                    " the digests here make a re-fetch verifiable. Cost of losing them is one"
                    " 126.8 GB download, about 17 minutes at the rate measured here.",
    },
    "note": "every size here is serialized bytes on disk; none of it is VRAM, resident weights or KV",
}
tmp = out + ".tmp"
json.dump(payload, open(tmp, "w"), indent=2)
os.replace(tmp, out)
print("receipt " + out)
PROV
  echo FETCHDONE
}

require_inputs() {
  local spec rel bytes sha
  require_verified "$CORPUS/wikitext-2-raw-v1.zip" "$WT2_ZIP_BYTES" "$WT2_ZIP_SHA"
  require_verified "$CORPUS/$WT2_MEMBER" "$WT2_BYTES" "$WT2_SHA"
  [ -f "$PPL_TXT" ] || die "missing $PPL_TXT -- run: ${BASH_SOURCE[0]} fetch"
  for spec in "${ARTIFACTS[@]}"; do
    rel=$(field "$spec" 2); bytes=$(field "$spec" 3); sha=$(field "$spec" 4)
    require_verified "$MODELS/$rel" "$bytes" "$sha"
  done
}

phase_preflight() {
  say "binaries"
  [ -x "$BIN/llama-perplexity" ] || die "no $BIN/llama-perplexity -- run: $HERE/build_llamacpp.sh tools"
  [ -x "$BIN/llama-tokenize" ]   || die "no $BIN/llama-tokenize -- run: $HERE/build_llamacpp.sh tools"
  [ -f "$PROVENANCE" ] || die "no $PROVENANCE -- run: $HERE/build_llamacpp.sh tools"
  local help="$LLAMACPP_ROOT/llama-perplexity-help.txt"
  [ -f "$help" ] || die "no $help -- run: $HERE/build_llamacpp.sh tools"

  # A flag that does not exist in this build is a silent protocol difference.
  local -a a; local flag missing=()
  mapfile -d '' a < <(argv_base)
  for flag in "${a[@]}"; do
    case "$flag" in
      --*) grep -q -- "$flag" "$help" || missing+=("$flag") ;;
    esac
  done
  mapfile -d '' a < <(argv_candidate "${CANDIDATES[0]}")
  for flag in "${a[@]}"; do
    case "$flag" in
      --*) grep -q -- "$flag" "$help" || missing+=("$flag") ;;
    esac
  done
  [ "${#missing[@]}" -eq 0 ] || die "flags absent from this build's --help: ${missing[*]}"
  echo "  ok  every emitted flag is present in llama-perplexity --help"

  say "inputs"
  require_inputs
  echo "  ok  corpus and five GGUFs match their pinned digests"

  say "tokenising (vocabulary only, CUDA_VISIBLE_DEVICES=-1, no weights, no inference)"
  local -a t; mapfile -d '' t < <(argv_tokenize)
  local tok_log="$LOGS/tokenize.log"
  mkdir -p "$LOGS" "$(dirname "$TOKENS_JSON")"
  ggrun_cpu "${t[@]}" >"$TOKENS_JSON.raw" 2>"$tok_log" || { tail -20 "$tok_log"; die "llama-tokenize"; }

  local n_tokens vocab
  n_tokens=$(sed -n 's/^Total number of tokens: \([0-9]\+\)$/\1/p' "$TOKENS_JSON.raw" | tail -1)
  [ -n "$n_tokens" ] || die "llama-tokenize printed no token count"
  grep -o '^\[.*\]$' "$TOKENS_JSON.raw" >"$TOKENS_JSON" || die "llama-tokenize printed no id list"
  rm -f "$TOKENS_JSON.raw"
  vocab=$(sed -n 's/.*n_vocab *= *\([0-9]\+\).*/\1/p' "$tok_log" | head -1)
  [ -n "$vocab" ] || die "could not read n_vocab out of the loader log"

  local chunks processed scored expect_bytes
  chunks=$(( n_tokens / N_CTX ))
  processed=$(( chunks * N_CTX ))
  scored=$(( chunks * PER_CHUNK ))
  expect_bytes=$(( 20 + processed * 4 + scored * (2 * ((vocab + 1) / 2) + 4) * 2 ))

  printf '  tokens     %d  (reference %d [INV])\n' "$n_tokens" "$REF_TOKENS"
  printf '  vocab      %d\n' "$vocab"
  printf '  chunks     %d  expected %d\n' "$chunks" "$N_CHUNK"
  printf '  processed  %d  expected %d\n' "$processed" "$N_PROCESSED"
  printf '  scored     %d  expected %d\n' "$scored" "$N_SCORED"
  printf '  base bytes %d  expected %d\n' "$expect_bytes" "$BASE_LOGITS_BYTES"

  # This is the refusal the whole script exists to make.
  [ "$vocab"     = "$N_VOCAB" ]     || die "vocabulary is $vocab, protocol pinned at $N_VOCAB"
  [ "$chunks"    = "$N_CHUNK" ]     || die "chunk arithmetic: $chunks chunks, expected $N_CHUNK -- refusing to spend GPU time"
  [ "$processed" = "$N_PROCESSED" ] || die "chunk arithmetic: $processed processed tokens, expected $N_PROCESSED -- refusing to spend GPU time"
  [ "$scored"    = "$N_SCORED" ]    || die "chunk arithmetic: $scored scored positions, expected $N_SCORED -- refusing to spend GPU time"
  [ "$expect_bytes" = "$BASE_LOGITS_BYTES" ] || die "base-logits size would be $expect_bytes, protocol pinned at $BASE_LOGITS_BYTES"
  [ "$n_tokens" -ge "$N_PROCESSED" ] && [ "$n_tokens" -lt "$(( N_PROCESSED + N_CTX ))" ] \
    || die "token count $n_tokens is outside the band that yields $N_CHUNK chunks"
  echo "  ok  580 chunks / 296,960 processed / 147,900 scored -- protocol arithmetic matches"

  local out="$RECEIPTS/wikitext-kld-preflight.json"
  python3 - "$out" "$PROVENANCE" "$TOKENS_JSON" "$tok_log" <<PROV
import hashlib, json, os, sys, time
out, prov_path, tokens_path, tok_log = sys.argv[1:5]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

prov = json.load(open(prov_path))
tools = prov.get("external_protocol_tools")
if tools is None:
    raise SystemExit("FAILED: %s has no external_protocol_tools block" % prov_path)

payload = {
    "schema": "qwen38-external-protocol-preflight/1",
    "purpose": "CPU-only gate for Run A: chunk arithmetic, vocabulary, build inventory",
    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "gpu_touched": False,
    "how_gpu_was_excluded": "llama-tokenize runs vocab_only under CUDA_VISIBLE_DEVICES=-1",
    "build": {
        "commit": tools["commit"],
        "tree": tools["tree"],
        "source_digest": tools["source_digest"],
        "cmake_flags": tools["cmake_flags"],
        "cuda_architectures": tools["cuda_architectures"],
        "kv_cache": tools["kv_cache"],
        "artifacts": tools["artifacts"],
    },
    "corpus": {
        "perplexity_input_bytes": $(stat -c %s "$PPL_TXT"),
        "token_ids": {"path": tokens_path, "bytes": os.path.getsize(tokens_path),
                      "sha256": sha256(tokens_path),
                      "use": "step 5 of Run A, cross-engine token identity (D10)"},
        "tokenizer_log": {"path": tok_log, "sha256": sha256(tok_log)},
    },
    "arithmetic": {
        "n_tokens": $n_tokens,
        "n_tokens_reference_INV": $REF_TOKENS,
        "n_ctx": $N_CTX,
        "n_vocab": $vocab,
        "nv_uint16_stride": 2 * (($vocab + 1) // 2) + 4,
        "n_chunk": $chunks,
        "processed_tokens": $processed,
        "first_scored_index": $FIRST,
        "scored_per_chunk": $PER_CHUNK,
        "scored_positions": $scored,
        "base_logits_bytes": $expect_bytes,
        "base_logits_formula": "20 + n_chunk*n_ctx*4 + n_chunk*(n_ctx-1-n_ctx/2)*nv*2",
        "all_match_protocol": True,
    },
}
tmp = out + ".tmp"
json.dump(payload, open(tmp, "w"), indent=2)
os.replace(tmp, out)
print("receipt " + out)
PROV
  echo PREFLIGHTDONE
}

# Step 5 of Run A: cross-engine token identity (D10).  Two tokenizers, one file.
# llama.cpp's GGUF BPE tokenised the corpus for the scoring runs; the HF
# HF tokenizer our whole protocol is built on - the base model's own, digest
# 0997f410… as recorded in every receipt that names one - is what the rest of
# this repo tokenised with.  If those two disagree, every cross-citation is
# comparing different text and no divergence number means anything.  CPU only.
phase_tokencheck() {
  local hf_tok="${KLD_HF_TOKENIZER:-/var/tmp/models/Qwen3.8-27B/tokenizer.json}"
  [ -f "$hf_tok" ] || die "no HF tokenizer at $hf_tok (set KLD_HF_TOKENIZER)"
  [ -f "$TOKENS_JSON" ] || die "no $TOKENS_JSON -- run: ${BASH_SOURCE[0]} preflight"
  local g_tok; g_tok="$(guest_path "$PPL_TXT")"

  say "token identity: llama.cpp GGUF BPE vs the HF tokenizer, same bytes"
  ggrun_cpu python3 - "/models/${hf_tok#/var/tmp/models/}" "$g_tok" \
    "$(guest_path "$TOKENS_JSON")" "$(guest_path "$ROOT")/token-identity.json" <<'PY'
import hashlib, json, struct, sys, time
from tokenizers import Tokenizer

hf_path, text_path, llama_ids_path, out = sys.argv[1:5]

raw = open(text_path, "rb").read()
hf = Tokenizer.from_file(hf_path)
# tokenizer.json ships a 2,048-token truncation policy for training use; leaving
# it on would silently compare 2,048 tokens against 297,194 and call it a
# mismatch.
hf.no_truncation()
hf.no_padding()
hf_ids = hf.encode(raw.decode("utf-8"), add_special_tokens=False).ids
llama_ids = json.load(open(llama_ids_path))

def digest(ids):
    return hashlib.sha256(struct.pack("<%di" % len(ids), *ids)).hexdigest()

n = min(len(hf_ids), len(llama_ids))
first_diff = next((i for i in range(n) if hf_ids[i] != llama_ids[i]), None)
scored = 580 * 512
payload = {
    "schema": "qwen38-external-protocol-token-identity/1",
    "purpose": "Run A step 5 / D10: do the two tokenizers produce the same stream?",
    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "input": {"path": text_path, "bytes": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest()},
    "llama_cpp": {"source": "llama-tokenize on the BF16 GGUF's own vocabulary; the same"
                            " common_tokenize call llama-perplexity makes on the same bytes",
                  "n_tokens": len(llama_ids), "sha256_int32le": digest(llama_ids)},
    "huggingface": {"source": hf_path, "add_special_tokens": False,
                    "tokenizer_sha256": hashlib.sha256(open(hf_path, "rb").read()).hexdigest(),
                    "truncation": "disabled explicitly; tokenizer.json ships a 2,048 policy",
                    "n_tokens": len(hf_ids), "sha256_int32le": digest(hf_ids)},
    "identical": hf_ids == llama_ids,
    "first_divergence_index": first_diff,
    "processed_prefix": {
        "n_tokens": scored,
        "identical": hf_ids[:scored] == llama_ids[:scored],
        "sha256_int32le": digest(llama_ids[:scored]),
        "why": "llama-perplexity stores and scores exactly this prefix, 580 chunks of 512",
    },
    "caveat": "compared against the llama-tokenize stream, not against the token block inside"
              " the base logits file, which was released after the run",
}
json.dump(payload, open(out, "w"), indent=2)
print("identical=%s  llama=%d  hf=%d  prefix_identical=%s" %
      (payload["identical"], len(llama_ids), len(hf_ids),
       payload["processed_prefix"]["identical"]))
PY
  cp "$ROOT/token-identity.json" "$RECEIPTS/wikitext-kld-token-identity.json"
  echo "receipt $RECEIPTS/wikitext-kld-token-identity.json"
  echo TOKENCHECKDONE
}

phase_run() {
  phase_preflight

  # Two separate gates, because they fail for different reasons.  The card is
  # shared and its queue is arbitrated elsewhere, so spending it is opt-in and
  # never a side effect of a resume: KLD_GPU_OK=1 is the operator saying "this
  # run owns the GPU now".  KLD_FORCE=1 is narrower - it only waives the
  # occupancy check, for the case where the memory in use is known to be ours.
  if [ "${KLD_GPU_OK:-0}" != 1 ]; then
    die "this phase can launch a ~12 minute GPU job (one capture + three scorings).
       Set KLD_GPU_OK=1 when the card is actually yours.  Everything that can be
       checked without a GPU has already passed above."
  fi
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-0}" -gt 1024 ] && [ "${KLD_FORCE:-0}" != 1 ]; then
    die "GPU already holds ${used} MiB - another job owns this card. KLD_FORCE=1 to override."
  fi

  mkdir -p "$LOGS" "$(dirname "$LOGITS")"
  local -a a
  if [ -f "$LOGITS" ] && [ "$(stat -c %s "$LOGITS")" = "$BASE_LOGITS_BYTES" ]; then
    say "base logits already present and the right size"
  else
    say "reference capture: BF16 -> $BASE_LOGITS_BYTES B of logits"
    mapfile -d '' a < <(argv_base)
    ggrun_ppl "${a[@]}" 2>&1 | tee "$LOGS/base.log"
  fi

  # The post-run assertion. Chunk count, window, token count and vocabulary all
  # land in this one integer.
  local got; got=$(stat -c %s "$LOGITS")
  [ "$got" = "$BASE_LOGITS_BYTES" ] \
    || die "base logits are $got B, protocol requires exactly $BASE_LOGITS_BYTES B for vocab $N_VOCAB"
  if [ -f "$LOGS/base.log" ]; then
    grep -q "calculating perplexity over $N_CHUNK chunks, n_ctx=$N_CTX" "$LOGS/base.log" \
      || die "the capture log does not say '$N_CHUNK chunks, n_ctx=$N_CTX'"
  fi
  say "base logits verified: $got B"

  local tag
  for tag in "${CANDIDATES[@]}"; do
    if [ -s "$LOGS/$tag.log" ] && grep -q "^Same top p:" "$LOGS/$tag.log"; then
      say "$tag already scored"
      continue
    fi
    say "scoring $tag"
    mapfile -d '' a < <(argv_candidate "$tag")
    ggrun_ppl "${a[@]}" 2>&1 | tee "$LOGS/$tag.log"
    grep -q "^Same top p:" "$LOGS/$tag.log" || die "$tag produced no KL divergence summary"
  done

  local out="$RECEIPTS/wikitext-kld-run-a.json"
  python3 - "$out" "$RECEIPTS/wikitext-kld-preflight.json" "$RECEIPTS/wikitext-kld-fetch-verify.json" \
    "$LOGITS" "$LOGS" --candidates "${CANDIDATES[@]}" <<PROV
import hashlib, json, os, re, sys, time
out, pre_path, fetch_path, logits_path, logs = sys.argv[1:6]
candidates = sys.argv[sys.argv.index("--candidates") + 1:]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

LABELS = {
    "mean_kld": r"^Mean\s+KLD:\s*(-?[\d.]+)\s*±\s*([\d.]+)",
    "max_kld": r"^Maximum KLD:\s*(-?[\d.]+)",
    "kld_99_9": r"^99\.9%\s+KLD:\s*(-?[\d.]+)",
    "median_kld": r"^Median\s+KLD:\s*(-?[\d.]+)",
    "min_kld": r"^Minimum KLD:\s*(-?[\d.]+)",
    "rms_dp_pct": r"^RMS Δp\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "same_top_pct": r"^Same top p:\s*([\d.]+)\s*±\s*([\d.]+)",
    "mean_ppl_q": r"^Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "mean_ppl_base": r"^Mean PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
}

def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    row = {}
    for key, pat in LABELS.items():
        m = re.search(pat, text, re.M)
        if not m:
            raise SystemExit("FAILED: %s has no %s line" % (path, key))
        row[key] = float(m.group(1))
        if m.lastindex and m.lastindex > 1:
            row[key + "_sem"] = float(m.group(2))
    return row

pre = json.load(open(pre_path))
rows = {}
for tag in candidates:
    log = os.path.join(logs, tag + ".log")
    rows[tag] = parse(log)
    rows[tag]["log"] = {"path": log, "sha256": sha256(log)}

payload = {
    "schema": "qwen38-external-protocol-run-a/1",
    "question": "what do our three GGUF targets measure under the external protocol,"
                " run exactly as its authors run it?",
    "protocol": {
        "harness": "llama.cpp llama-perplexity --kl-divergence",
        "source": "docs/35-external-protocol-comparability.md, Run A; commands verbatim from [U-LOG-B]",
        "corpus": "WikiText-2 raw test",
        "n_ctx": pre["arithmetic"]["n_ctx"],
        "direction": "KL(BF16 ‖ candidate), reference = unquantized BF16 GGUF in the same engine",
        "not_our_protocol": "no v5 suite, no shared BF16 head, no float64 replay; the candidate's"
                            " own output head is inside the measured path. Run B in"
                            " receipts/cross-engine-comparator.json is the comparable one.",
    },
    "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "build": pre["build"],
    "kv_cache_used": {
        "cache_type_k": "f16", "cache_type_v": "f16",
        "why": "upstream default; no -ctk/-ctv is passed, so the protocol is unchanged",
        "enabled_types": pre["build"]["kv_cache"],
    },
    "arithmetic": pre["arithmetic"],
    "base_logits": {
        "path": logits_path,
        "bytes": os.path.getsize(logits_path),
        "bytes_required": pre["arithmetic"]["base_logits_bytes"],
        "byte_identity_holds": os.path.getsize(logits_path) == pre["arithmetic"]["base_logits_bytes"],
        "why_this_is_the_proof": "chunk count, scoring window, processed-token count and vocabulary"
                                 " all appear in this single integer",
        "capture_log": ({"path": os.path.join(logs, "base.log"),
                         "sha256": sha256(os.path.join(logs, "base.log"))}
                        if os.path.isfile(os.path.join(logs, "base.log")) else None),
    },
    "inputs": {
        "fetch_receipt": fetch_path,
        "preflight_receipt": pre_path,
    },
    "results": rows,
    "retention": {
        "policy": "preserve before delete",
        "this_pipeline_deletes_nothing": True,
        "base_logits_recompute_cost": "one full BF16 forward pass over 296,960 tokens of"
                                      " wikitext-2 raw test on a single card; the inputs are"
                                      " pinned public blobs, this file is not - if it is ever"
                                      " removed it has to be recomputed, not re-downloaded.",
        "inputs_recovery": "see the retention block of the fetch receipt",
    },
    "note": "every byte size in the inputs is serialized bytes on disk; none of it is VRAM or KV",
}
tmp = out + ".tmp"
json.dump(payload, open(tmp, "w"), indent=2)
os.replace(tmp, out)
print("receipt " + out)
for tag, row in rows.items():
    print("%-8s mean KLD %.6f ± %.6f   same top %.3f %%" %
          (tag, row["mean_kld"], row["mean_kld_sem"], row["same_top_pct"]))
PROV
  echo RUNADONE
}

# Deleting is not the default anywhere in this project - preserve before delete.
# Two artifact classes here, and they are exceptions for two different reasons,
# so they are released separately and never in one gesture:
#
#   logits  the 68.41 GiB base capture is published nowhere, but it recomputes
#           from the GGUFs in a single BF16 pass measured at 2 min 54 s on this
#           card, and its correctness is re-checkable by the byte identity this
#           run asserted.  Cheaper to remake than to store.
#   blobs   the five GGUFs are public, but "public" is not the same as durable:
#           a Hub repo can be squashed and take the cited revision with it, which
#           is exactly what happened to unsloth/Qwen3.8-27B-NVFP4 on 2026-08-15.
#           So these are released only against a named archival mirror, passed in
#           KLD_MIRROR as <repo>@<commit>.  No mirror, no delete.
#
# Nothing is released before the receipt that cites it exists and is complete;
# nothing is released at all unless KLD_RELEASE_OK=1 says so out loud; and each
# scope writes its own receipt rather than amending one.
phase_release() {
  local scope="${RELEASE_SCOPE:-all}"
  case "$scope" in logits|blobs|all) ;; *) die "RELEASE_SCOPE must be logits, blobs or all" ;; esac

  local receipt="$RECEIPTS/wikitext-kld-run-a.json"
  [ -f "$receipt" ] || die "no $receipt -- nothing is released before the run it feeds"

  local -a doomed=() ; local spec
  [ "$scope" = blobs ] || doomed+=("$LOGITS")
  if [ "$scope" != logits ]; then
    [ -n "${KLD_MIRROR:-}" ] || die "releasing the GGUFs needs a durable source.
       Pass KLD_MIRROR=<repo>@<commit> naming the archival mirror that keeps the
       citation resolvable after an upstream squash.  A pinned upstream revision
       is not one: unsloth/Qwen3.8-27B-NVFP4 lost its cited revision that way."
    for spec in "${ARTIFACTS[@]}"; do doomed+=("$MODELS/$(field "$spec" 2)"); done
  fi

  local total=0 p
  for p in "${doomed[@]}"; do
    [ -f "$p" ] && total=$(( total + $(stat -c %s "$p") ))
  done

  if [ "${KLD_RELEASE_OK:-0}" != 1 ]; then
    say "print-only, scope=$scope. KLD_RELEASE_OK=1 to actually free $(gib "$total") GiB"
    for p in "${doomed[@]}"; do [ -f "$p" ] && echo "  rm -f $p"; done
    return
  fi

  local out="$RECEIPTS/wikitext-kld-release-$scope.json"
  python3 - "$out" "$receipt" "$RECEIPTS/wikitext-kld-fetch-verify.json" "$LOGITS" "$scope" \
    --candidates "${CANDIDATES[@]}" --artifacts "${ARTIFACTS[@]}" <<PROV
import json, os, sys, time
out, run_receipt, fetch_receipt, logits_path, scope = sys.argv[1:6]
argv = sys.argv[6:]
candidates = argv[argv.index("--candidates") + 1:argv.index("--artifacts")]
specs = argv[argv.index("--artifacts") + 1:]
mirror = os.environ.get("KLD_MIRROR", "")

run = json.load(open(run_receipt))
if not run["base_logits"]["byte_identity_holds"]:
    raise SystemExit("FAILED: the run receipt does not assert byte identity; nothing is released")
missing = [t for t in candidates if t not in run["results"]]
if missing:
    raise SystemExit("FAILED: %s were never scored; nothing is released" % ", ".join(missing))

freed = []
if scope != "blobs":
    freed.append({
        "what": os.path.basename(logits_path),
        "bytes": run["base_logits"]["bytes"],
        "sha256": None,
        "recovery": "recompute",
        "source": None,
        "cost": "one llama-perplexity --save-all-logits pass over the two-part BF16 GGUF,"
                " measured at 2 min 54 s on an RTX PRO 6000. The remake is self-checking:"
                " it is correct only if it is again exactly %d B."
                % run["base_logits"]["bytes"],
    })
if scope != "logits":
    repo, _, commit = mirror.partition("@")
    for s in specs:
        tag, rel, nbytes, sha = s.split("|")
        freed.append({
            "what": rel, "bytes": int(nbytes), "sha256": sha,
            "recovery": "re-fetch",
            "source": {"repo": repo, "commit": commit,
                       "url": "https://huggingface.co/%s/resolve/%s/%s" % (repo, commit, rel),
                       "mirrors": "$GGUF_REPO@$GGUF_REV"},
            "cost": "download only; the digest above makes the re-fetch verifiable",
        })

payload = {
    "schema": "qwen38-external-protocol-release/1",
    "purpose": "records what Run A deleted, and exactly what it costs to get each piece back",
    "scope": scope,
    "released_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "policy": "preserve before delete; these are the exception because every line below names"
              " either a durable source with a digest or a measured recompute cost",
    "measurements_survive_in": [run_receipt, fetch_receipt],
    "freed_bytes": sum(f["bytes"] for f in freed),
    "freed": freed,
    "note": "serialized bytes on disk; none of this was VRAM or KV",
}
if scope != "logits":
    payload["durable_source"] = {
        "mirror": mirror,
        "what_it_guarantees": "the citation: a repo id, revision and digest table that stay"
                              " resolvable if the upstream repo is squashed or deleted, as"
                              " unsloth/Qwen3.8-27B-NVFP4 was on 2026-08-15",
        "what_it_does_not_guarantee": "independent byte-level redundancy. Hugging Face Xet storage"
                                      " is content-addressed, so the mirror and upstream plausibly"
                                      " reference the same chunks - the 126.8 GB mirror moved"
                                      " 2.13 GB on the wire. Say re-fetchable, not backed up.",
        "upstream_status_at_release": "$GGUF_REPO@$GGUF_REV still resolves; this mirror is"
                                      " insurance, not rescue",
    }
tmp = out + ".tmp"
json.dump(payload, open(tmp, "w"), indent=2)
os.replace(tmp, out)
print("receipt " + out)
PROV

  for p in "${doomed[@]}"; do rm -f "$p" "$p.verified"; done
  say "freed $(gib "$total") GiB, scope=$scope; every measurement survives in $receipt"
  echo RELEASEDONE
}

case "$PHASE" in
  plan)      phase_plan ;;
  fetch)     phase_fetch ;;
  preflight) phase_preflight ;;
  tokencheck) phase_tokencheck ;;
  run)       phase_run ;;
  release)   phase_release ;;
esac
