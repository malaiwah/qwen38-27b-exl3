#!/bin/bash
# Preserve an expensive artifact tree on the Hub *before* anything deletes it locally.
#
# Why this exists: the v5 KLD ladder captured six models per 512-context shard, verified the
# reports, then deleted that shard's hidden states to make room for the next one - ten times.
# The reports survived; ~190 GB of reusable capture did not.  Re-downloading an artifact is
# minutes of network; recomputing one is minutes of a GPU that is usually busy with something
# else, and sometimes it is not recomputable at all because the upstream input stopped
# existing.  So: upload first, verify the remote copy against the local bytes, and only then
# consider the local copy expendable.
#
# This script NEVER DELETES ANYTHING.  On success it prints the delete command for a human (or
# a calling pipeline) to run.  That is deliberate: a tool that both publishes and deletes will
# eventually delete after a publish that only looked successful.
#
# Three tiers, not one rule.  Which tier an artifact is in decides whether this script is the
# right tool for it at all:
#
#   MIRROR - a third-party artifact that a published number of ours cites.  Unconditional:
#     measured cost is 2.13 GB of wire transfer for 126.8 GB of content (1.7 %), because Xet
#     storage is content-addressed and the bytes already exist on the Hub.  At that price there
#     is no trade-off to weigh.  What a mirror preserves is the CITATION - a resolvable repo
#     id, revision and digest table that survives an upstream squash or delete.  It is NOT a
#     backup and NOT byte-level redundancy: our mirror and upstream's copy plausibly reference
#     the same underlying chunks.  Independent physical redundancy is a separate decision with
#     a real bandwidth cost and needs its own justification.  Use `hf upload` against a
#     dedicated archival repo for this tier, not this script.
#
#   PRESERVE - a locally-produced artifact where (a) recompute costs GPU-hours, or (b) an input
#     has ceased to exist so it cannot be regenerated at all, or (c) it is a shared reference
#     many future runs replay against.  This script is for exactly this tier.
#
#   STATE THE COST - everything else.  Delete it freely, but the receipt that names it must say
#     what recomputing it costs.  The worked example is the ten-shard NVFP4 capture set: 19
#     captures x 10.7 GiB is ~200 GB of upload to avoid ~6 minutes of GPU each, so uploading is
#     the wrong call and `receipts/nvfp4-v5-measurement.json` states the per-artifact cost
#     instead.  `receipts/preserved-artifacts.json` records both sides of that decision.
#
# Verification, and exactly how strong it is:
#   * every file the Hub stores in LFS (>10 MB, or matched by .gitattributes) exposes its
#     sha256 as the LFS oid, which is compared against the local sha256.  That is an
#     end-to-end content check on the full bytes without re-downloading them.
#   * every non-LFS file exposes a git blob id (sha1 over "blob <len>\0" + content), compared
#     against the same value computed locally.
#   * `--deep` additionally re-downloads every non-LFS file and compares real sha256, and
#     re-downloads `--spot N` LFS files to prove what the CDN serves.  Use it for anything you
#     are about to delete.  Without `--deep` the script says so in its own output rather than
#     implying it hashed bytes it never saw.
# Any missing file, any size disagreement, any digest disagreement: non-zero exit, no delete
# command printed.
#
# Usage:
#   tools/preserve_artifacts.sh <local-dir> <repo-path> [options]
#
#   <local-dir>     directory to preserve, e.g. /var/tmp/work/gguf/hidden-bf16
#   <repo-path>     prefix inside the repo, e.g. reference/hidden-bf16, or `.` for the repo
#                   root - which is what publishing a whole checkpoint into a model repo needs
#
#   --repo ID       target repo (default: $PRESERVE_REPO or the v5 fidelity suite)
#   --repo-type T   dataset | model | space (default: dataset)
#   --deep          re-download and sha256 every non-LFS file, plus --spot LFS files
#   --spot N        LFS files to re-download under --deep (default: 3; 0 disables; all = every)
#   --receipt PATH  write the verified inventory as JSON (digests, bytes, per-file check kind)
#   --message MSG   commit message
#   --dry-run       inventory and print what would be uploaded; no network write
#
# Examples:
#   # the shard-0 BF16 reference every future candidate replays against
#   tools/preserve_artifacts.sh /var/tmp/work/gguf/hidden-bf16 reference/hidden-bf16 --deep
#
#   # a candidate capture tree handed over by another agent, before its pipeline cleans up
#   tools/preserve_artifacts.sh /var/tmp/work/kld5/hidden/shard-0000/hidden-eda \
#       captures/shard-0000/hidden-eda --deep --receipt /var/tmp/preserved-eda.json
#
#   # a whole checkpoint published into its own model repo, verified before the local copy goes
#   tools/preserve_artifacts.sh /var/tmp/work/kld6/qwen38-eda . \
#       --repo malaiwah/Qwen3.8-27B-EXL3-EDA-research --repo-type model --deep
#
# Environment:
#   HF_HOME         must point at the authenticated cache (this project: /var/tmp/hf-home-3)
#   PRESERVE_REPO   default target repo id
#   HF              path to the `hf` CLI (default: /home/mbelleau/.local/bin/hf)

set -euo pipefail

die() { echo "FAILED: $*" >&2; exit 1; }
say() { echo "== $*"; }

HF=${HF:-/home/mbelleau/.local/bin/hf}
REPO=${PRESERVE_REPO:-malaiwah/qwen38-27b-fidelity-suite-v5}
REPO_TYPE=dataset
DEEP=0
SPOT=3
RECEIPT=
MESSAGE=
DRY_RUN=0
LOCAL_DIR=
DEST=

while [ $# -gt 0 ]; do
  case "$1" in
    --repo)      REPO=${2:?--repo needs a value}; shift 2 ;;
    --repo-type) REPO_TYPE=${2:?--repo-type needs a value}; shift 2 ;;
    --deep)      DEEP=1; shift ;;
    --spot)      SPOT=${2:?--spot needs a value}; shift 2 ;;
    --receipt)   RECEIPT=${2:?--receipt needs a value}; shift 2 ;;
    --message)   MESSAGE=${2:?--message needs a value}; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)   sed -n '2,/^set -euo pipefail$/{/^set -euo pipefail$/d;p;}' "$0"; exit 0 ;;
    -*)          die "unknown option $1" ;;
    *)
      if [ -z "$LOCAL_DIR" ]; then LOCAL_DIR=$1
      elif [ -z "$DEST" ];    then DEST=$1
      else die "unexpected argument $1"
      fi
      shift ;;
  esac
done

[ -n "$LOCAL_DIR" ] && [ -n "$DEST" ] || die "usage: preserve_artifacts.sh <local-dir> <dataset-path> [options]"
[ -d "$LOCAL_DIR" ] || die "not a directory: $LOCAL_DIR"
[ -x "$HF" ] || die "hf CLI not executable at $HF (set HF=...)"
case "$DEST" in
  /*|*..*) die "repo path must be repo-relative and must not contain '..': $DEST" ;;
esac

# The `hf` CLI ships its own interpreter; that is the one that has huggingface_hub.
HFPY=$(sed -n '1s|^#!||p' "$HF")
[ -x "$HFPY" ] || die "cannot find the interpreter behind $HF"

[ -n "${HF_HOME:-}" ] || die "HF_HOME is unset; point it at the authenticated cache"
# `hf auth whoami` has two output shapes and picks between them on its own: the
# machine-readable `user=NAME`, and a decorated block whose second line is
# `  user: NAME`.  Which one you get depends on whether the CLI thinks it is talking to
# an agent -- setting AGENT or CLAUDECODE flips it -- so a script that parses only
# `user=` works from an agent harness and reports "not logged in" against a perfectly
# authenticated cache when a human runs it. Both shapes are accepted, and the strip of
# any ANSI colouring happens first because the decorated form is colourised.
WHO=$("$HF" auth whoami 2>/dev/null \
  | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/[[:space:]]*$//' \
  | sed -n -e 's/^user=//p' -e 's/^[[:space:]]*user:[[:space:]]*//p' | head -1) \
  || die "hf auth whoami failed; is HF_HOME=$HF_HOME authenticated?"
[ -n "$WHO" ] || die "not logged in under HF_HOME=$HF_HOME"

LOCAL_DIR=$(cd "$LOCAL_DIR" && pwd)
DEST=${DEST%/}
# `.` is the whole-repo destination: `hf upload` already spells the root that way, and a model
# repo published from a checkpoint directory has no prefix.  PREFIX is what the remote paths are
# actually built from, so it is empty at the root and `dir/` below it.
if [ "$DEST" = . ]; then PREFIX=; else PREFIX=$DEST/; fi
FILES=$(find "$LOCAL_DIR" -type f | wc -l)
[ "$FILES" -gt 0 ] || die "$LOCAL_DIR contains no files"
say "preserving $FILES file(s) from $LOCAL_DIR -> $REPO:$DEST ($REPO_TYPE) as $WHO"

if find "$LOCAL_DIR" -type l | grep -q .; then
  say "note: symlinks below are uploaded as their target's bytes, not as links"
  find "$LOCAL_DIR" -type l -printf '   %P -> %l\n'
fi

INVENTORY=$(mktemp)
REMOTE=$(mktemp)
trap 'rm -f "$INVENTORY" "$REMOTE"' EXIT

# ------------------------------------------------------------------ local inventory
say "hashing local bytes (sha256 + git blob id)"
"$HFPY" - "$LOCAL_DIR" "$PREFIX" "$INVENTORY" <<'PY' || die "local inventory failed"
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor

root, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]

def one(abs_path):
    rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
    n = os.path.getsize(abs_path)
    h = hashlib.sha256()
    g = hashlib.sha1(b"blob %d\0" % n)
    with open(abs_path, "rb") as f:
        while (c := f.read(8 << 20)):
            h.update(c); g.update(c)
    return f"{prefix}{rel}", {"local_path": abs_path, "bytes": n,
                             "sha256": h.hexdigest(), "git_blob_sha1": g.hexdigest()}

paths = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
with ThreadPoolExecutor(min(32, (os.cpu_count() or 4) * 2)) as ex:
    inv = dict(ex.map(one, paths))
json.dump(inv, open(out, "w"))
total = sum(v["bytes"] for v in inv.values())
print(f"   {len(inv)} file(s), {total} B ({total / 2**30:.2f} GiB)")
PY

if [ "$DRY_RUN" = 1 ]; then
  say "dry run: nothing uploaded, nothing verified, no delete command printed"
  "$HFPY" - "$INVENTORY" <<'PY'
import json, sys
inv = json.load(open(sys.argv[1]))
for k, v in sorted(inv.items())[:10]:
    print(f"   would upload {k}  {v['bytes']} B  {v['sha256'][:16]}")
if len(inv) > 10:
    print(f"   ... {len(inv)} file(s) total")
PY
  exit 0
fi

# ------------------------------------------------------------------ upload
# One invocation for the whole directory: the Hub batches it into as few commits as it needs,
# and a per-file loop would be both slower and non-atomic.
say "uploading"
UPLOAD_LOG=$(mktemp); trap 'rm -f "$INVENTORY" "$REMOTE" "$UPLOAD_LOG"' EXIT
"$HF" upload "$REPO" "$LOCAL_DIR" "$DEST" \
  --repo-type "$REPO_TYPE" \
  --commit-message "${MESSAGE:-preserve $DEST before local deletion}" 2>&1 | tee "$UPLOAD_LOG"
# Same two output shapes as `auth whoami`: `url=https://...` when the CLI thinks it is
# talking to a machine, and a decorated `  url: https://...` otherwise.  What this guard
# is really asserting is that a commit URL came back at all, so it tests for the URL
# rather than for one of its two spellings -- REVISION below parses either.
grep -qE '^(url=|[[:space:]]*url:[[:space:]]*)https?://.*/commit/' "$UPLOAD_LOG" \
  || die "upload did not report a commit url; nothing was verified"
REVISION=$(sed -n 's|.*/commit/||p' "$UPLOAD_LOG" | tail -1)
[ -n "$REVISION" ] || die "could not read the resulting revision from the upload output"
say "uploaded at revision $REVISION"

# ------------------------------------------------------------------ remote verification
say "re-verifying the remote copy against local bytes"
set +e
"$HFPY" - "$REPO" "$REPO_TYPE" "$REVISION" "$INVENTORY" "$REMOTE" "$DEEP" "$SPOT" <<'PY'
import hashlib, json, os, sys, tempfile

repo, repo_type, revision, inv_path, out_path, deep, spot = sys.argv[1:8]
deep = deep == "1"
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()

inv = json.load(open(inv_path))
info = api.repo_info(repo, repo_type=repo_type, revision=revision, files_metadata=True)
remote = {}
for s in info.siblings:
    lfs = s.lfs
    oid = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    remote[s.rfilename] = {"size": s.size, "blob_id": s.blob_id, "lfs_sha256": oid}

problems, checked = [], {}
for rel, loc in sorted(inv.items()):
    r = remote.get(rel)
    if r is None:
        problems.append(f"{rel}: absent from {repo}@{revision}")
        continue
    if r["size"] != loc["bytes"]:
        problems.append(f"{rel}: remote {r['size']} B != local {loc['bytes']} B")
        continue
    if r["lfs_sha256"]:
        if r["lfs_sha256"] != loc["sha256"]:
            problems.append(f"{rel}: remote LFS sha256 {r['lfs_sha256']} != local {loc['sha256']}")
            continue
        checked[rel] = "remote-sha256-lfs-oid"
    else:
        if r["blob_id"] != loc["git_blob_sha1"]:
            problems.append(f"{rel}: remote git blob {r['blob_id']} != local {loc['git_blob_sha1']}")
            continue
        checked[rel] = "remote-git-blob-id"

# --deep: pull the bytes back and hash them, so the claim is about served content and not
# only about metadata the Hub computed for us.
if deep and not problems:
    def readback(rel):
        with tempfile.TemporaryDirectory() as td:
            p = hf_hub_download(repo, rel, repo_type=repo_type, revision=revision,
                                local_dir=td)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while (c := f.read(8 << 20)):
                    h.update(c)
            return h.hexdigest()

    nonlfs = sorted(r for r in checked if checked[r] == "remote-git-blob-id")
    lfs = sorted(r for r in checked if checked[r] == "remote-sha256-lfs-oid")
    # Spread the LFS sample across the tree rather than taking a prefix: a truncated or
    # mis-ordered upload shows up at the ends and the middle, not in the first three files.
    if spot == "all":
        sample = lfs
    else:
        n = min(max(int(spot), 0), len(lfs))
        sample = [lfs[round(i * (len(lfs) - 1) / (n - 1))] for i in range(n)] if n > 1 else lfs[:n]
    print(f"   deep: re-downloading {len(nonlfs)} non-LFS file(s) "
          f"and {len(sample)} of {len(lfs)} LFS file(s)")
    for rel, kind in [(r, "served-sha256-readback") for r in nonlfs] + \
                     [(r, "served-sha256-readback-lfs-spot") for r in sample]:
        got = readback(rel)
        if got != inv[rel]["sha256"]:
            problems.append(f"{rel}: re-downloaded sha256 {got} != local {inv[rel]['sha256']}")
        else:
            checked[rel] = kind

json.dump({"repo": repo, "repo_type": repo_type, "revision": revision,
           "files": {rel: {**inv[rel], "verified_by": checked.get(rel)} for rel in inv},
           "problems": problems}, open(out_path, "w"))

kinds = {}
for k in checked.values():
    kinds[k] = kinds.get(k, 0) + 1
for k, n in sorted(kinds.items()):
    print(f"   {n} file(s) verified by {k}")
if problems:
    print(f"   {len(problems)} PROBLEM(S):")
    for p in problems[:20]:
        print(f"     {p}")
    sys.exit(1)
total = sum(v["bytes"] for v in inv.values())
print(f"   all {len(inv)} file(s) verified, {total} B")
PY
VERIFY_RC=$?
set -e

if [ "$VERIFY_RC" != 0 ]; then
  echo "FAILED: remote verification did not pass; the local copy is still the only good copy." >&2
  echo "        Nothing was deleted and no delete command is printed.  Re-run this script to" >&2
  echo "        re-upload the mismatched files, or investigate $REPO@$REVISION by hand." >&2
  exit 1
fi

if [ -n "$RECEIPT" ]; then
  # Copy, not move: $REMOTE lives in the trap and the receipt must outlive this process.
  cat "$REMOTE" | "$HFPY" -c 'import json,sys; json.dump(json.load(sys.stdin), open(sys.argv[1],"w"), indent=1, sort_keys=True)' "$RECEIPT"
  say "receipt written: $RECEIPT"
fi

TOTAL=$("$HFPY" -c 'import json,sys; print(sum(v["bytes"] for v in json.load(open(sys.argv[1])).values()))' "$INVENTORY")
# Model repos live at the bare id; every other type is namespaced by its plural.
case "$REPO_TYPE" in
  model) URL=https://huggingface.co/$REPO ;;
  *)     URL=https://huggingface.co/${REPO_TYPE}s/$REPO ;;
esac
cat <<SUMMARY

== PRESERVED
   repo      $URL/tree/$REVISION/$PREFIX
   revision  $REVISION
   files     $FILES
   bytes     $TOTAL
   depth     $([ "$DEEP" = 1 ] && echo "deep (bytes re-downloaded and re-hashed)" || echo "metadata (remote LFS sha256 / git blob ids; pass --deep to re-hash served bytes)")

The remote copy is verified.  This script does not delete anything.  To reclaim the space,
run this yourself once you are satisfied:

   rm -rf -- '$LOCAL_DIR'

SUMMARY
