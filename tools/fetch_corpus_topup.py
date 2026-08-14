#!/usr/bin/env python3
"""Top up the held-out corpus: arXiv abstracts and long Wikipedia articles.

Fixes two bugs in the first pass: the arXiv title/summary regex required
adjacency the Atom feed does not guarantee, and the Wikipedia query needs
`exlimit=max` or only the first page of a multi-title batch carries an extract.
Multilingual articles are resolved through `langlinks` from a fixed English
title list, so every language gets genuinely long articles rather than stubs.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("/var/tmp/work/kld3/corpus")
UA = {"User-Agent": "qwen38-exl3-fidelity/1.0 (research; contact malaiwah)"}

TITLES = [
    "Byzantine Empire", "Photosynthesis", "Quantum computing", "Monetary policy",
    "Plate tectonics", "French Revolution", "Roman Empire", "Immune system",
    "General relativity", "Industrial Revolution", "Ottoman Empire", "Evolution",
    "Climate change", "World War II", "Mathematics", "Music theory",
    "Architecture", "Renaissance", "Buddhism", "Linguistics",
    "Cell (biology)", "Thermodynamics", "Meiji Restoration", "Silk Road",
    "Human brain", "Antarctica", "Cryptography", "Genetics",
    "Napoleon", "Ancient Egypt", "Democracy", "Supply and demand",
    "Neural network (machine learning)", "Volcano", "Ocean current", "Vaccine",
    "Printing press", "Steam engine", "Internet", "Chess",
]
LANGS = ["de", "fr", "es", "ja", "zh", "ru", "it", "pt"]
CATS = ["cs.LG", "cs.CL", "cs.DC", "math.PR", "math.OC", "physics.optics",
        "q-bio.NC", "econ.EM", "stat.ME", "eess.SP", "astro-ph.GA", "cond-mat.soft"]


def get(url: str, timeout: int = 90) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def write(stratum: str, name: str, text: str) -> None:
    (OUT / stratum).mkdir(parents=True, exist_ok=True)
    (OUT / stratum / f"{name}.txt").write_text(text, encoding="utf-8")


def arxiv() -> None:
    for cat in CATS:
        url = ("http://export.arxiv.org/api/query?search_query=cat:" + cat +
               "&sortBy=submittedDate&sortOrder=descending&start=0&max_results=200")
        try:
            xml = get(url).decode("utf-8", "ignore")
        except Exception as e:
            print(f"  arxiv {cat}: {type(e).__name__} {e}", flush=True)
            continue
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
        chunks = []
        for e in entries:
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            s = re.search(r"<summary>(.*?)</summary>", e, re.S)
            if not (t and s):
                continue
            title = re.sub(r"\s+", " ", t.group(1)).strip()
            summary = re.sub(r"\s+", " ", s.group(1)).strip()
            if len(summary) > 500:
                chunks.append(f"{title}\n\n{summary}")
        if chunks:
            write("scientific", f"arxiv-{cat.replace('.', '-')}", "\n\n".join(chunks))
            print(f"  scientific {cat}: {len(chunks)} abstracts, "
                  f"{sum(map(len, chunks))//1000} KB", flush=True)


def extracts(lang: str, titles: list[str]) -> list[tuple[str, str]]:
    got = []
    for i in range(0, len(titles), 10):
        batch = titles[i:i + 10]
        url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
               "&prop=extracts&explaintext=1&exlimit=max&redirects=1&titles="
               + urllib.parse.quote("|".join(batch)))
        try:
            data = json.loads(get(url))
        except Exception as e:
            print(f"  wiki {lang}: {type(e).__name__} {e}", flush=True)
            continue
        for page in (data.get("query", {}).get("pages") or {}).values():
            ex = page.get("extract") or ""
            if len(ex) > 3000:
                got.append((page.get("title", ""), ex))
    return got


def langlinks(titles: list[str], lang: str) -> list[str]:
    out = []
    for i in range(0, len(titles), 10):
        batch = titles[i:i + 10]
        url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
               f"&prop=langlinks&lllang={lang}&lllimit=max&redirects=1&titles="
               + urllib.parse.quote("|".join(batch)))
        try:
            data = json.loads(get(url))
        except Exception:
            continue
        for page in (data.get("query", {}).get("pages") or {}).values():
            for ll in page.get("langlinks", []) or []:
                if ll.get("lang") == lang and ll.get("*"):
                    out.append(ll["*"])
    return out


def main() -> int:
    print("arxiv", flush=True)
    arxiv()

    print("wikipedia en", flush=True)
    en = extracts("en", TITLES)
    for j in range(0, len(en), 5):
        chunk = en[j:j + 5]
        write("encyclopedic", f"wiki-en-{j:03d}",
              "\n\n".join(f"{t}\n\n{x}" for t, x in chunk))
    print(f"  encyclopedic en: {len(en)} articles, "
          f"{sum(len(x) for _, x in en)//1000} KB", flush=True)

    print("wikipedia multilingual", flush=True)
    for lang in LANGS:
        local = langlinks(TITLES[:24], lang)
        arts = extracts(lang, local) if local else []
        for j in range(0, len(arts), 5):
            chunk = arts[j:j + 5]
            write("multilingual", f"wiki-{lang}-{j:03d}",
                  "\n\n".join(f"{t}\n\n{x}" for t, x in chunk))
        print(f"  multilingual {lang}: {len(arts)} articles, "
              f"{sum(len(x) for _, x in arts)//1000} KB", flush=True)

    sizes = {p.name: sum(f.stat().st_size for f in p.glob('*.txt')) // 1000
             for p in sorted(OUT.iterdir()) if p.is_dir()}
    print(json.dumps({"strata_kb": sizes}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
