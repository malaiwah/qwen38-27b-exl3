#!/usr/bin/env python3
"""Fetch a held-out evaluation corpus, disjoint from exllamav3's calibration data.

Why this exists: the v2 suite was built from exllamav3's bundled calibration
corpora (`wiki.utf8`, `c4.utf8`, `code.utf8`, ...) — the *same* text our K4
conversion calibrated on. That biases the comparison in our favour, because the
NVFP4 and FP8 candidates were calibrated elsewhere. Any honest number has to come
from text no candidate was tuned on.

Sources, all public and none of them exllamav3 calibration data:
  literary      Project Gutenberg public-domain books
  scientific    arXiv abstracts (API), recent
  encyclopedic  Wikipedia REST extracts, random articles, English
  multilingual  Wikipedia REST extracts in de / fr / es / ja / zh / ru
  code          CPython standard library at a pinned tag
"""
from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("/var/tmp/work/kld3/corpus")
UA = {"User-Agent": "qwen38-exl3-fidelity/1.0 (research; contact malaiwah)"}


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write(stratum: str, name: str, text: str) -> None:
    d = OUT / stratum
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.txt").write_text(text, encoding="utf-8")


def gutenberg() -> None:
    # Varied public-domain prose: fiction, philosophy, history, science, essays.
    ids = [1342, 2701, 84, 98, 1080, 174, 4300, 2814, 1661, 345,
           5200, 1232, 3207, 2600, 244, 16328, 25344, 1400, 768, 219]
    for i in ids:
        for url in (f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt",
                    f"https://www.gutenberg.org/files/{i}/{i}-0.txt"):
            try:
                raw = get(url).decode("utf-8", "ignore")
            except Exception:
                continue
            # strip licence headers/footers
            body = re.split(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", raw)
            body = body[-1] if len(body) > 1 else raw
            body = re.split(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", body)[0]
            body = body.split("\n", 1)[-1]
            if len(body) > 40000:
                write("literary", f"pg{i}", body[:400000])
                print(f"  literary pg{i}: {len(body)//1000} KB", flush=True)
            break


def arxiv() -> None:
    cats = ["cs.LG", "cs.CL", "math.PR", "physics.optics", "q-bio.NC", "econ.EM",
            "stat.ME", "cs.DC", "eess.SP", "astro-ph.GA"]
    for cat in cats:
        url = ("http://export.arxiv.org/api/query?search_query=cat:" + cat +
               "&sortBy=submittedDate&sortOrder=descending&start=0&max_results=120")
        try:
            xml = get(url, timeout=90).decode("utf-8", "ignore")
        except Exception as e:
            print(f"  arxiv {cat} failed: {e}", flush=True)
            continue
        chunks = []
        for m in re.finditer(r"<title>(.*?)</title>\s*<summary>(.*?)</summary>", xml, re.S):
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            summary = re.sub(r"\s+", " ", m.group(2)).strip()
            if len(summary) > 400:
                chunks.append(f"{title}\n\n{summary}")
        if chunks:
            write("scientific", f"arxiv-{cat.replace('.', '-')}", "\n\n".join(chunks))
            print(f"  scientific {cat}: {len(chunks)} abstracts", flush=True)


def wikipedia(lang: str, stratum: str, batches: int, name: str) -> None:
    texts = []
    for b in range(batches):
        url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
               "&generator=random&grnnamespace=0&grnlimit=20"
               "&prop=extracts&explaintext=1&exlimit=20")
        try:
            data = json.loads(get(url, timeout=60))
        except Exception as e:
            print(f"  wiki {lang} batch {b} failed: {e}", flush=True)
            continue
        for page in (data.get("query", {}).get("pages") or {}).values():
            ex = page.get("extract") or ""
            if len(ex) > 1500:
                texts.append(f"{page.get('title','')}\n\n{ex}")
    if texts:
        write(stratum, f"{name}-{lang}", "\n\n".join(texts))
        print(f"  {stratum} {lang}: {len(texts)} articles, "
              f"{sum(map(len, texts))//1000} KB", flush=True)


def cpython() -> None:
    tag = "v3.12.8"
    url = f"https://codeload.github.com/python/cpython/tar.gz/refs/tags/{tag}"
    raw = get(url, timeout=300)
    keep, total = [], 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".py"):
                continue
            if "/Lib/" not in m.name or "/test" in m.name:
                continue
            data = tf.extractfile(m)
            if data is None:
                continue
            text = data.read().decode("utf-8", "ignore")
            if len(text) < 4000:
                continue
            keep.append(f"# {m.name}\n{text}")
            total += len(text)
            if total > 3_000_000:
                break
    for i in range(0, len(keep), 20):
        write("code", f"cpython-{tag}-{i:04d}", "\n\n".join(keep[i:i + 20]))
    print(f"  code: {len(keep)} modules, {total//1000} KB", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("gutenberg", flush=True); gutenberg()
    print("arxiv", flush=True); arxiv()
    print("wikipedia en", flush=True)
    for _ in range(3):
        wikipedia("en", "encyclopedic", 4, "wiki")
    print("wikipedia multilingual", flush=True)
    for lang in ("de", "fr", "es", "ja", "zh", "ru"):
        wikipedia(lang, "multilingual", 3, "wiki")
    print("cpython", flush=True); cpython()
    sizes = {p.name: sum(f.stat().st_size for f in p.glob("*.txt")) // 1000
             for p in sorted(OUT.iterdir()) if p.is_dir()}
    print(json.dumps({"strata_kb": sizes}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
