#!/usr/bin/env python3
"""Publish this project's cards to their Hugging Face repos and prove byte-identity.

Per repo: upload the local card as README.md, refresh that repo's DOCS-SHA256SUMS README
line so the pinned digest is not left stale, then re-fetch both files from the hub with a
throwaway cache and compare bytes. A repo that does not come back byte-identical is reported
as such; it is never reported as published.

Eight repos: the four model cards, the two research checkpoints (error-driven and S16-V), the
K6-parity release candidate, and the v5 fidelity dataset. The last four are here because editing
a card on the hub alone had already let the published dataset card drift ahead of its local
source, and a published repo that is not in this map is one nobody re-checks.

`--only REPO` publishes a subset, for the case where one repo's card is new and the rest are
someone else's in-flight work.

Nothing here computes a card's content. It only moves bytes and checks them.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
HF_HOME = "/var/tmp/hf-home-3"

# repo -> (local card file, repo type)
CARDS = {
    "malaiwah/Qwen3.8-27B-EXL3-K5K6-context": ("MODEL_CARD-K5K6-context.md", "model"),
    "malaiwah/Qwen3.8-27B-EXL3-K5K6": ("MODEL_CARD-K5K6.md", "model"),
    "malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated": ("MODEL_CARD-K5K6-hydrated.md", "model"),
    "malaiwah/Qwen3.8-27B-K4": ("MODEL_CARD-K4.md", "model"),
    "malaiwah/Qwen3.8-27B-EXL3-EDA-research": ("MODEL_CARD-EDA-research.md", "model"),
    "malaiwah/Qwen3.8-27B-EXL3-K6-parity": ("MODEL_CARD-K6-parity.md", "model"),
    "malaiwah/Qwen3.8-27B-EXL3-S16-V-research": ("MODEL_CARD-S16-V-research.md", "model"),
    "malaiwah/qwen38-27b-fidelity-suite-v5": ("DATASET_CARD-v5.md", "dataset"),
}


def hf(*argv, cwd=None):
    env = {"HF_HOME": HF_HOME, "PATH": "/home/mbelleau/.local/bin:/usr/bin:/bin",
           "HOME": "/home/mbelleau"}
    proc = subprocess.run(["hf", *argv], capture_output=True, text=True, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"hf {' '.join(argv)} failed rc={proc.returncode}\n"
                         f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout.strip()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fetch(repo, filename, into, repo_type):
    hf("download", repo, filename, "--repo-type", repo_type, "--local-dir", str(into))
    return (into / filename).read_bytes()


def publish_one(repo, card, repo_type, message, dry_run):
    local = (REPO / card).read_bytes()
    row = {
        "repo": repo, "repo_type": repo_type, "local_card": card,
        "local_bytes": len(local), "local_sha256": sha256_bytes(local),
        "local_lines": local.decode().count("\n"),
    }
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="cards-"))
    try:
        before = fetch(repo, "README.md", scratch / "before", repo_type)
        row["published_before"] = {"bytes": len(before), "sha256": sha256_bytes(before)}
        if before == local:
            row["already_current"] = True

        sums_path = scratch / "before" / "DOCS-SHA256SUMS"
        try:
            fetch(repo, "DOCS-SHA256SUMS", scratch / "before", repo_type)
            sums = sums_path.read_text()
        except SystemExit:
            sums = None
        if sums is not None:
            # `sha256sum -c` needs exactly two characters between digest and name (space plus
            # the ' ' text-mode or '*' binary-mode flag). A single space makes coreutils report
            # "improperly formatted line" and skip the row, which is how every README row this
            # tool had written was unverifiable while every other row in the same file checked.
            lines, seen = [], False
            for line in sums.splitlines():
                if line.endswith(" README.md"):
                    lines.append(f"{row['local_sha256']}  README.md")
                    seen = True
                else:
                    lines.append(line)
            row["docs_sha256sums_had_readme_line"] = seen
            if not seen:
                lines.append(f"{row['local_sha256']}  README.md")
            lines.sort(key=lambda x: x.split(None, 1)[1])  # tolerate a legacy one-space row
            new_sums = "\n".join(lines) + "\n"
        else:
            new_sums = None
            row["docs_sha256sums_had_readme_line"] = None

        if dry_run:
            row["dry_run"] = True
            return row

        staged = scratch / "upload"
        staged.mkdir()
        (staged / "README.md").write_bytes(local)
        if new_sums is not None:
            (staged / "DOCS-SHA256SUMS").write_text(new_sums)
        hf("upload", repo, str(staged), ".", "--repo-type", repo_type,
           "--commit-message", message)

        after = fetch(repo, "README.md", scratch / "after", repo_type)
        row["published_bytes"] = len(after)
        row["published_sha256"] = sha256_bytes(after)
        row["published_lines"] = after.decode().count("\n")
        row["byte_identical"] = after == local
        if new_sums is not None:
            got = fetch(repo, "DOCS-SHA256SUMS", scratch / "after", repo_type).decode()
            row["docs_sha256sums_readme_current"] = \
                f"{row['local_sha256']}  README.md" in got
            row["docs_sha256sums_byte_identical"] = got == new_sums
        return row
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO / "receipts/apc-card-publication.json"))
    ap.add_argument("--only", action="append", metavar="REPO",
                    help="publish only this repo id; repeatable. Default: every repo in CARDS")
    args = ap.parse_args()

    selected = list(CARDS)
    if args.only:
        unknown = [r for r in args.only if r not in CARDS]
        if unknown:
            raise SystemExit(f"--only names repos that are not in CARDS: {', '.join(unknown)}")
        selected = [r for r in CARDS if r in args.only]

    rows = [publish_one(repo, *CARDS[repo], args.message, args.dry_run) for repo in selected]
    payload = {
        "schema": "qwen38-apc-card-publication/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": f"hf upload, HF_HOME={HF_HOME}, user malaiwah",
        "what_changed": args.message,
        "per_repo": rows,
        "all_byte_identical": all(r.get("byte_identical") for r in rows) if not args.dry_run
                              else None,
    }
    if not args.dry_run:
        pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if args.dry_run or payload["all_byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
