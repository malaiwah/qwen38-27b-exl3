#!/usr/bin/env python3
"""Audit and sync the image assets that our published cards reference.

WHY THIS EXISTS. publish_cards.py uploads README.md and the DOCS-SHA256SUMS row and
nothing else, so a card could reference assets/foo.svg, pass its byte-identity check,
and still render a BROKEN IMAGE on the Hub because the asset was never uploaded. That
is exactly what happened: the hydrated card referenced six figures (tb21-topology and
tb21-scaling, light and dark, svg) that had never been pushed to the model repo, so
the multi-GPU serving section showed broken images while every automated check we had
reported success. Byte-identity of the README is not sufficient; asset presence is a
separate invariant and needs its own check.

Default mode is a read-only AUDIT across every repo in publish_cards.CARDS. Pass
--upload to push the missing files. For each referenced .svg we also carry the
matching .png when one exists locally, because every asset already on the Hub ships
both and the cards use <picture> with an svg src.

Usage:
    sync_card_assets.py                 # audit only, non-zero exit if anything missing
    sync_card_assets.py --upload        # audit then upload whatever is missing
    sync_card_assets.py --only REPO     # restrict to one repo id (repeatable)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

ASSET_RE = re.compile(r"assets/[A-Za-z0-9._-]+")


def _publish_cards_module():
    """Import publish_cards.py so CARDS and HF_HOME cannot drift between the two tools."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pc", REPO_ROOT / "tools" / "publish_cards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def referenced(card_path: Path) -> set[str]:
    return set(ASSET_RE.findall(card_path.read_text(encoding="utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="upload missing assets (default: audit only)")
    ap.add_argument("--only", action="append", help="restrict to this repo id; repeatable")
    a = ap.parse_args()

    # publish_cards.py keeps its Hub credentials under a dedicated HF_HOME; without
    # this the default cache is unauthenticated and every upload 401s while the
    # read-only audit still appears to work, which is the worst possible failure mode.
    import os

    CARDS_MOD = _publish_cards_module()
    os.environ["HF_HOME"] = CARDS_MOD.HF_HOME

    from huggingface_hub import HfApi

    api = HfApi()
    CARDS = CARDS_MOD.CARDS
    selected = a.only or list(CARDS)
    unknown = [r for r in selected if r not in CARDS]
    if unknown:
        raise SystemExit(f"--only names repos not in CARDS: {', '.join(unknown)}")

    total_missing = 0
    for repo in selected:
        card_name, repo_type = CARDS[repo]
        card = REPO_ROOT / card_name
        if not card.exists():
            print(f"{repo}: SKIP - local card {card_name} not found")
            continue
        want = referenced(card)
        if not want:
            print(f"{repo}: no asset references")
            continue
        have = set(api.list_repo_files(repo, repo_type=repo_type))
        missing = sorted(w for w in want if w not in have)
        # carry the sibling png for each missing svg when we have one locally
        extra = []
        for m in missing:
            if m.endswith(".svg"):
                png = m[:-4] + ".png"
                if (REPO_ROOT / png).exists() and png not in have:
                    extra.append(png)
        missing = sorted(set(missing) | set(extra))

        if not missing:
            print(f"{repo}: OK - all {len(want)} referenced asset(s) present")
            continue

        total_missing += len(missing)
        print(f"{repo}: MISSING {len(missing)} of {len(want)} referenced asset(s):")
        for m in missing:
            local = REPO_ROOT / m
            state = "local ok" if local.exists() else "NOT PRESENT LOCALLY EITHER"
            print(f"    {m}   [{state}]")

        if a.upload:
            to_push = [m for m in missing if (REPO_ROOT / m).exists()]
            absent = [m for m in missing if not (REPO_ROOT / m).exists()]
            if absent:
                print(f"    refusing to invent {len(absent)} file(s) absent locally: {', '.join(absent)}")
            for m in to_push:
                api.upload_file(path_or_fileobj=str(REPO_ROOT / m), path_in_repo=m,
                                repo_id=repo, repo_type=repo_type,
                                commit_message=f"Add {m} - referenced by the card but missing from the repo")
                print(f"    uploaded {m}")
            after = set(api.list_repo_files(repo, repo_type=repo_type))
            still = sorted(w for w in want if w not in after)
            print(f"    verify: {'OK - every referenced asset now present' if not still else 'STILL MISSING ' + ', '.join(still)}")

    if total_missing and not a.upload:
        print(f"\n{total_missing} missing asset(s). Re-run with --upload to fix.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
