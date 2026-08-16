#!/usr/bin/env python3
"""Publish the four model cards to their Hugging Face repos and prove byte-identity.

Per repo: upload the local card as README.md, refresh that repo's DOCS-SHA256SUMS README
line so the pinned digest is not left stale, then re-fetch both files from the hub with a
throwaway cache and compare bytes. A repo that does not come back byte-identical is reported
as such; it is never reported as published.

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

CARDS = {
    "malaiwah/Qwen3.8-27B-EXL3-K5K6-context": "MODEL_CARD-K5K6-context.md",
    "malaiwah/Qwen3.8-27B-EXL3-K5K6": "MODEL_CARD-K5K6.md",
    "malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated": "MODEL_CARD-K5K6-hydrated.md",
    "malaiwah/Qwen3.8-27B-K4": "MODEL_CARD-K4.md",
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


def fetch(repo, filename, into):
    hf("download", repo, filename, "--repo-type", "model", "--local-dir", str(into))
    return (into / filename).read_bytes()


def publish_one(repo, card, message, dry_run):
    local = (REPO / card).read_bytes()
    row = {
        "repo": repo, "local_card": card,
        "local_bytes": len(local), "local_sha256": sha256_bytes(local),
        "local_lines": local.decode().count("\n"),
    }
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="cards-"))
    try:
        before = fetch(repo, "README.md", scratch / "before")
        row["published_before"] = {"bytes": len(before), "sha256": sha256_bytes(before)}
        if before == local:
            row["already_current"] = True

        sums_path = scratch / "before" / "DOCS-SHA256SUMS"
        try:
            fetch(repo, "DOCS-SHA256SUMS", scratch / "before")
            sums = sums_path.read_text()
        except SystemExit:
            sums = None
        if sums is not None:
            lines, seen = [], False
            for line in sums.splitlines():
                if line.endswith(" README.md"):
                    lines.append(f"{row['local_sha256']} README.md")
                    seen = True
                else:
                    lines.append(line)
            row["docs_sha256sums_had_readme_line"] = seen
            if not seen:
                lines.append(f"{row['local_sha256']} README.md")
                lines.sort(key=lambda x: x.split(" ", 1)[1])
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
        hf("upload", repo, str(staged), ".", "--repo-type", "model",
           "--commit-message", message)

        after = fetch(repo, "README.md", scratch / "after")
        row["published_bytes"] = len(after)
        row["published_sha256"] = sha256_bytes(after)
        row["published_lines"] = after.decode().count("\n")
        row["byte_identical"] = after == local
        if new_sums is not None:
            got = fetch(repo, "DOCS-SHA256SUMS", scratch / "after").decode()
            row["docs_sha256sums_readme_current"] = \
                f"{row['local_sha256']} README.md" in got
            row["docs_sha256sums_byte_identical"] = got == new_sums
        return row
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO / "receipts/apc-card-publication.json"))
    args = ap.parse_args()

    rows = [publish_one(repo, card, args.message, args.dry_run)
            for repo, card in CARDS.items()]
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
