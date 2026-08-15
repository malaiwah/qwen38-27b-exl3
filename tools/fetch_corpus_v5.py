#!/usr/bin/env python3
"""Fetch the v5 held-out evaluation corpus: a much larger disjoint text pool.

Why this exists: the published KLD claim rests on 136 contexts / 278k scored
positions, because the v3/v4 corpora were only 14 MB + 24 MB of clean text. A
1M-10M position ladder needs *exact-advance, non-overlapping* contexts, so it
needs an order of magnitude more held-out text, and it needs many independent
source documents so the bootstrap has real clusters to resample.

Sources, all public and licence-compatible with the v3/v4 set (none of them is
exllamav3 calibration data):
  literary      Project Gutenberg public-domain books, ids from the Gutendex
                catalogue (popular / English / copyright=false), one file per book
  encyclopedic  English Wikipedia plaintext extracts, one file per article,
                candidate titles from the Wikimedia pageviews REST API
  multilingual  the same pipeline in the v3/v4 language set plus more languages
  scientific    arXiv Atom abstract feeds across many categories, two windows each
  code          CPython and NumPy source files at pinned tags, one file per module

Design rules this tool enforces:
  * one document == one file == one unique stem (a real bootstrap cluster);
  * every written file is content hashed, and the atomic JSON fetch log next to
    the corpus lists every file with its stratum, stem, source URL, bytes, chars
    and sha256;
  * a stem that already exists in a previous corpus (v3/v4) is only accepted when
    the bytes are identical, otherwise the run aborts (fail closed);
  * partial failures are never silent: every fetch error is recorded in the log
    and printed, and unmet byte/document targets make the exit status non-zero.

Reruns are incremental: files already in the output tree are re-hashed, counted
and kept, and their stems are not fetched again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DEFAULT = Path("/var/tmp/work/kld5/corpus")
EXCLUDE_DEFAULT = ("/var/tmp/work/kld3/corpus", "/var/tmp/work/kld4/corpus")
# Wikimedia throttles requests whose User-Agent carries no contact route, so the
# product token matches the v3/v4 fetchers but the contact is spelled out.
UA = {"User-Agent": "qwen38-exl3-fidelity/1.0 (KLD fidelity research; "
                    "contact malaiwah@users.noreply.github.com) python-urllib"}

STRATA = ("code", "encyclopedic", "literary", "multilingual", "scientific")

# Byte targets per stratum; the run stops fetching a stratum once it is met.
TARGETS = {
    "literary": 20_000_000,
    "encyclopedic": 14_000_000,
    "multilingual": 12_000_000,
    "scientific": 12_000_000,
    "code": 12_000_000,
}
TOTAL_TARGET = 64_000_000
MIN_DOCS = 250

MIN_DOC_BYTES = 4_000
CAP_LITERARY = 400_000
CAP_WIKI = 250_000
CAP_CODE = 200_000

# Gutendex catalogue pages to consider (32 books per page).
GUTENDEX_PAGES = 16

# Pageview months used as the Wikipedia candidate-title source. Top-1000 per
# month, deduplicated across months; popular titles are the long articles.
PV_MONTHS_EN = ("2024/07", "2024/10", "2025/01", "2025/03", "2025/04", "2025/05",
                "2025/06", "2024/08", "2024/11", "2025/02")
PV_MONTHS_OTHER = ("2025/01", "2025/03", "2025/05", "2024/10")

# v3/v4 language set first, then the extra languages v5 adds.
LANGS = ("de", "fr", "es", "ja", "zh", "ru", "it", "pt",
         "nl", "pl", "ar", "ko", "tr", "sv", "uk", "id", "vi", "cs", "fa", "hi")

MIN_EXTRACT_EN = 8_000
MIN_EXTRACT_OTHER = 5_000
WIKI_SLEEP = 0.5

# Titles hard-coded by tools/fetch_corpus_topup.py; skipped so the v5
# encyclopedic stratum is not a rerun of the v3/v4 article set.
V4_TITLES = frozenset({
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
})

ARXIV_CATS = (
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CY", "cs.DB", "cs.DM",
    "cs.DS", "cs.ET", "cs.FL", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LO",
    "cs.MA", "cs.MM", "cs.NA", "cs.NE", "cs.NI", "cs.OS", "cs.PF", "cs.PL",
    "cs.SC", "cs.SD", "cs.SI", "cs.SY",
    "math.AC", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", "math.CV",
    "math.FA", "math.GN", "math.GR", "math.GT", "math.HO", "math.KT", "math.LO",
    "math.MG", "math.NT", "math.RA", "math.RT", "math.SG", "math.SP", "math.ST",
    "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus",
    "physics.bio-ph", "physics.chem-ph", "physics.class-ph", "physics.comp-ph",
    "physics.data-an", "physics.ed-ph", "physics.hist-ph", "physics.ins-det",
    "physics.med-ph", "physics.plasm-ph", "physics.pop-ph", "physics.soc-ph",
    "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci",
    "cond-mat.quant-gas", "cond-mat.stat-mech", "cond-mat.str-el",
    "astro-ph.CO", "astro-ph.EP", "astro-ph.HE", "astro-ph.IM",
    "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.QM", "q-bio.SC",
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.RM",
    "q-fin.ST", "q-fin.TR",
    "stat.CO", "stat.ML", "stat.OT", "econ.GN", "econ.TH",
    "eess.AS", "eess.IV", "eess.SY", "nlin.AO", "nlin.PS", "nlin.SI",
    "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nucl-ex",
)
ARXIV_WINDOWS = (0, 200)
ARXIV_PAGE = 200
ARXIV_MIN_ABSTRACTS = 40
ARXIV_SLEEP = 3.1

CODE_PINS = (
    ("cpython5", "python/cpython", "v3.13.1", 5_000_000),
    ("numpy5", "numpy/numpy", "v2.2.1", 4_000_000),
)

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")


class Collision(RuntimeError):
    """A v5 stem reuses a v3/v4 stem with different bytes."""


RETRY_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def get(url: str, timeout: int = 90, retries: int = 4, backoff: float = 2.0) -> bytes:
    """Fetch with backoff. Wikimedia and Gutenberg both throttle bursts with 429/503,
    and a single throttled response must not silently drop a document."""
    last: BaseException = RuntimeError("no attempt made")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES:
                raise
            last = e
            hinted = (e.headers.get("Retry-After") or "").strip()
            wait = float(hinted) if hinted.isdigit() else backoff * (2 ** attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            wait = backoff * (2 ** attempt)
        if attempt < retries:
            time.sleep(min(wait, 30.0))
    raise last


def clean(text: str) -> str:
    """Normalise to clean UTF-8 text: no CR, no control chars, no lone surrogates."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CTRL.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def slug(text: str, fallback_seed: str = "") -> str:
    s = unicodedata.normalize("NFKD", text)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()[:56]
    if len(s) < 3:
        s = "h" + hashlib.sha256((fallback_seed or text).encode("utf-8")).hexdigest()[:12]
    return s


class Ledger:
    """Writes documents, hashes them, and records everything for the fetch log."""

    def __init__(self, out: Path, exclude_roots: list[Path]) -> None:
        self.out = out
        self.exclude_roots = exclude_roots
        self.prior: dict[str, str] = {}      # stem -> sha256 in v3/v4
        self.prior_hashes: set[str] = set()
        for root in exclude_roots:
            for f in sorted(root.rglob("*.txt")):
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                self.prior.setdefault(f.stem, h)
                self.prior_hashes.add(h)
        self.records: dict[str, dict] = {}   # stem -> record
        self.seen_hashes: set[str] = set()
        self.failures: list[dict] = []
        self.skipped: list[dict] = []
        self.bytes_by_stratum: dict[str, int] = {s: 0 for s in STRATA}
        self._adopt_existing()

    def _adopt_existing(self) -> None:
        for stratum in STRATA:
            d = self.out / stratum
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.txt")):
                raw = f.read_bytes()
                h = hashlib.sha256(raw).hexdigest()
                self.records[f.stem] = {
                    "stratum": stratum, "stem": f.stem,
                    "path": str(f.relative_to(self.out)),
                    "url": "preexisting", "bytes": len(raw),
                    "chars": len(raw.decode("utf-8", "replace")), "sha256": h,
                    "fetched_this_run": False,
                }
                self.seen_hashes.add(h)
                self.bytes_by_stratum[stratum] += len(raw)

    # -- accounting -------------------------------------------------------
    def have(self, stem: str) -> bool:
        return stem in self.records

    def stratum_bytes(self, stratum: str) -> int:
        return self.bytes_by_stratum[stratum]

    def total_bytes(self) -> int:
        return sum(self.bytes_by_stratum.values())

    def fail(self, stage: str, url: str, exc: BaseException | str) -> None:
        detail = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        self.failures.append({"stage": stage, "url": url, "error": detail})
        print(f"  ! {stage} failed: {detail}", flush=True)

    def skip(self, reason: str, stratum: str, stem: str, url: str, detail: str = "") -> None:
        self.skipped.append({"reason": reason, "stratum": stratum, "stem": stem,
                             "url": url, "detail": detail})

    # -- writing ----------------------------------------------------------
    def write(self, stratum: str, stem: str, text: str, url: str,
              cap: int | None = None, min_bytes: int = MIN_DOC_BYTES) -> int:
        if stratum not in STRATA:
            raise ValueError(f"unknown stratum {stratum!r}")
        if stem in self.records:
            self.skip("stem-already-present", stratum, stem, url)
            return 0
        text = clean(text)
        if cap is not None and len(text) > cap:
            text = text[:cap].rsplit("\n", 1)[0] + "\n"
        raw = text.encode("utf-8")
        if len(raw) < min_bytes:
            self.skip("too-short", stratum, stem, url, f"{len(raw)} bytes")
            return 0
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as e:            # cannot happen after encode
            self.fail("utf8-roundtrip", url, e)
            return 0
        h = hashlib.sha256(raw).hexdigest()
        prior = self.prior.get(stem)
        if prior is not None and prior != h:
            raise Collision(f"stem {stem!r} exists in v3/v4 with sha256 {prior} "
                            f"but v5 bytes hash {h}")
        if h in self.seen_hashes:
            self.skip("duplicate-bytes", stratum, stem, url, h)
            return 0
        if h in self.prior_hashes:
            self.skip("duplicate-of-prior-corpus", stratum, stem, url, h)
            return 0
        d = self.out / stratum
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{stem}.txt"
        tmp = d / f".{stem}.txt.tmp"
        tmp.write_bytes(raw)
        os.replace(tmp, dst)
        back = dst.read_bytes()
        if hashlib.sha256(back).hexdigest() != h:
            self.fail("readback", url, f"sha mismatch for {dst}")
            return 0
        self.records[stem] = {
            "stratum": stratum, "stem": stem, "path": str(dst.relative_to(self.out)),
            "url": url, "bytes": len(raw), "chars": len(text), "sha256": h,
            "fetched_this_run": True,
        }
        self.seen_hashes.add(h)
        self.bytes_by_stratum[stratum] += len(raw)
        return len(raw)


# ---------------------------------------------------------------- literary


def gutendex_ids(led: Ledger, pages: int) -> list[int]:
    ids: list[int] = []
    url = ("https://gutendex.com/books?languages=en&copyright=false"
           "&sort=popular&page=1")
    for _ in range(pages):
        try:
            data = json.loads(get(url, timeout=60))
        except Exception as e:
            led.fail("gutendex", url, e)
            break
        for book in data.get("results") or []:
            if book.get("copyright") is False and isinstance(book.get("id"), int):
                ids.append(book["id"])
        nxt = data.get("next")
        if not nxt:
            break
        url = nxt
        time.sleep(0.3)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


GUT_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG")
GUT_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG")


def gutenberg_body(raw: str) -> str:
    """Strip Project Gutenberg licence header/footer (same rule as v3)."""
    parts = GUT_START.split(raw)
    body = parts[-1] if len(parts) > 1 else raw
    body = GUT_END.split(body)[0]
    return body.split("\n", 1)[-1]


def literary(led: Ledger, target: int) -> None:
    ids = gutendex_ids(led, GUTENDEX_PAGES)
    print(f"  gutendex: {len(ids)} candidate ids", flush=True)
    books = 0
    for i in ids:
        if led.stratum_bytes("literary") >= target:
            break
        stem = f"pg{i}"
        if led.have(stem) or stem in led.prior:
            continue
        raw = None
        used = ""
        last: BaseException | str = "no url tried"
        for url in (f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt",
                    f"https://www.gutenberg.org/files/{i}/{i}-0.txt"):
            try:
                raw = get(url, timeout=120).decode("utf-8", "ignore")
                used = url
                break
            except Exception as e:
                last = e
        if raw is None:
            led.fail("gutenberg", f"pg{i}", last)
            time.sleep(1.0)
            continue
        body = gutenberg_body(raw)
        if len(body) < 40_000:
            led.skip("too-short", "literary", stem, used, f"{len(body)} chars")
        elif led.write("literary", stem, body, used, cap=CAP_LITERARY):
            books += 1
            if books % 10 == 0:
                print(f"    literary: {books} books, "
                      f"{led.stratum_bytes('literary')//1_000_000} MB", flush=True)
        time.sleep(0.7)
    print(f"  literary: {books} new books, "
          f"{led.stratum_bytes('literary')//1000} KB total", flush=True)


# -------------------------------------------------------------- scientific

ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
TITLE = re.compile(r"<title>(.*?)</title>", re.S)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)


def scientific(led: Ledger, target: int) -> None:
    docs = 0
    for cat in ARXIV_CATS:
        if led.stratum_bytes("scientific") >= target:
            break
        for start in ARXIV_WINDOWS:
            stem = f"arxiv5-{cat.replace('.', '-')}-{start:04d}"
            if led.have(stem):
                continue
            url = ("http://export.arxiv.org/api/query?search_query=cat:"
                   + urllib.parse.quote(cat)
                   + "&sortBy=submittedDate&sortOrder=descending"
                   f"&start={start}&max_results={ARXIV_PAGE}")
            try:
                xml = get(url, timeout=120).decode("utf-8", "ignore")
            except Exception as e:
                led.fail(f"arxiv {cat} start={start}", url, e)
                time.sleep(ARXIV_SLEEP)
                continue
            chunks = []
            for entry in ENTRY.findall(xml):
                t, s = TITLE.search(entry), SUMMARY.search(entry)
                if not (t and s):
                    continue
                title = html.unescape(re.sub(r"\s+", " ", t.group(1)).strip())
                summary = html.unescape(re.sub(r"\s+", " ", s.group(1)).strip())
                if len(summary) > 500:
                    chunks.append(f"{title}\n\n{summary}")
            if len(chunks) < ARXIV_MIN_ABSTRACTS:
                led.skip("thin-feed", "scientific", stem, url, f"{len(chunks)} abstracts")
            elif led.write("scientific", stem, "\n\n".join(chunks), url):
                docs += 1
            time.sleep(ARXIV_SLEEP)
        if docs and docs % 20 == 0:
            print(f"    scientific: {docs} feeds, "
                  f"{led.stratum_bytes('scientific')//1_000_000} MB", flush=True)
    print(f"  scientific: {docs} new feeds, "
          f"{led.stratum_bytes('scientific')//1000} KB total", flush=True)


# ------------------------------------------------- encyclopedic / multilingual

BAD_TITLE = re.compile(r"^(Main[_ ]Page|Special:|Wikipedia:|Portal:|Help:|Category:"
                       r"|File:|Template:|Talk:|User:|Draft:|Module:|MediaWiki:)")


_TITLE_CACHE: dict[str, list[str]] = {}


def pageview_titles(led: Ledger, lang: str, months: tuple[str, ...]) -> list[str]:
    if lang in _TITLE_CACHE:
        return _TITLE_CACHE[lang]
    out: list[str] = []
    seen: set[str] = set()
    for period in months:
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
               f"{lang}.wikipedia/all-access/{period}/all-days")
        try:
            data = json.loads(get(url, timeout=60))
        except Exception as e:
            led.fail(f"pageviews {lang} {period}", url, e)
            continue
        for item in data.get("items") or []:
            for art in item.get("articles") or []:
                name = (art.get("article") or "").replace("_", " ")
                if not name or BAD_TITLE.match(name) or ":" in name:
                    continue
                if name in V4_TITLES or name in seen:
                    continue
                seen.add(name)
                out.append(name)
        time.sleep(0.2)
    _TITLE_CACHE[lang] = out
    return out


def extract(lang: str, title: str) -> tuple[str, str, str]:
    """One title per request: the API only returns full extracts one page at a time."""
    url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
           "&formatversion=2&prop=extracts&explaintext=1&redirects=1&titles="
           + urllib.parse.quote(title))
    data = json.loads(get(url, timeout=60))
    pages = data.get("query", {}).get("pages") or []
    if not pages:
        return "", "", url
    page = pages[0]
    return page.get("title") or title, page.get("extract") or "", url


def wikipedia(led: Ledger, lang: str, stratum: str, target: int,
              months: tuple[str, ...], min_chars: int, prefix: str = "wiki5") -> int:
    titles = pageview_titles(led, lang, months)
    if not titles:
        led.fail(f"pageviews {lang}", "-", "no candidate titles")
        return 0
    docs = 0
    used_stems: set[str] = set()
    for title in titles:
        if led.stratum_bytes(stratum) >= target:
            break
        stem = f"{prefix}-{lang}-{slug(title, f'{lang}:{title}')}"
        if stem in used_stems or led.have(stem):
            continue
        used_stems.add(stem)
        try:
            got_title, text, url = extract(lang, title)
        except Exception as e:
            led.fail(f"wiki {lang} {title}", f"{lang}:{title}", e)
            time.sleep(0.5)
            continue
        if len(text) < min_chars:
            led.skip("too-short", stratum, stem, url, f"{len(text)} chars")
        else:
            docs += bool(led.write(stratum, stem, f"{got_title}\n\n{text}", url,
                                   cap=CAP_WIKI))
        time.sleep(WIKI_SLEEP)
    print(f"  {stratum} {lang}: {docs} new articles, "
          f"{led.stratum_bytes(stratum)//1000} KB stratum total", flush=True)
    return docs


def encyclopedic(led: Ledger, target: int) -> None:
    wikipedia(led, "en", "encyclopedic", target, PV_MONTHS_EN, MIN_EXTRACT_EN)


def multilingual(led: Ledger, target: int) -> None:
    # Round-robin in two passes so an early language cannot eat the whole budget.
    per_lang = max(target // (2 * len(LANGS)), 150_000)
    for _pass in (1, 2):
        for lang in LANGS:
            if led.stratum_bytes("multilingual") >= target:
                return
            budget = min(target, led.stratum_bytes("multilingual") + per_lang)
            wikipedia(led, lang, "multilingual", budget,
                      PV_MONTHS_OTHER, MIN_EXTRACT_OTHER)


# -------------------------------------------------------------------- code


def code_members(name: str) -> bool:
    if "/tests/" in name or "/test/" in name or "/test_" in name or "/Lib/test" in name:
        return False
    # Argument-clinic output and other generated shims are repetitive boilerplate;
    # they would flatter every candidate's agreement without measuring anything.
    if "/clinic/" in name or name.endswith((".c.h", ".h.h")) or "generated" in name:
        return False
    if name.endswith(".py"):
        return "/Lib/" in name or "/numpy/" in name
    if name.endswith((".c", ".h")):
        return ("/Modules/" in name or "/Objects/" in name or "/Python/" in name
                or "/numpy/" in name)
    return False


def code(led: Ledger, target: int) -> None:
    for label, repo, tag, budget in CODE_PINS:
        if led.stratum_bytes("code") >= target:
            break
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}"
        try:
            blob = get(url, timeout=600)
        except Exception as e:
            led.fail(f"code {repo}@{tag}", url, e)
            continue
        cands: list[tuple[str, str]] = []
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for m in tf:
                    if not m.isfile() or not code_members(m.name):
                        continue
                    fh = tf.extractfile(m)
                    if fh is None:
                        continue
                    text = fh.read().decode("utf-8", "ignore")
                    if len(text) >= 6_000:
                        cands.append((m.name, text))
        except Exception as e:
            led.fail(f"code untar {repo}@{tag}", url, e)
            continue
        # Reverse-sorted: v3/v4 took the first alphabetical slice of these trees,
        # so start from the other end to reduce overlap with the earlier corpora.
        cands.sort(key=lambda kv: kv[0], reverse=True)
        start = led.stratum_bytes("code")
        docs = 0
        for name, text in cands:
            if led.stratum_bytes("code") - start >= budget or led.stratum_bytes("code") >= target:
                break
            rel = name.split("/", 1)[-1]
            stem = f"{label}-{tag}-{slug(rel, name)}"
            if led.have(stem):
                continue
            docs += bool(led.write("code", stem, f"# {name}\n{text}", f"{url}#{name}",
                                   cap=CAP_CODE))
        print(f"  code {repo}@{tag}: {docs} files from {len(cands)} candidates, "
              f"{led.stratum_bytes('code')//1000} KB stratum total", flush=True)


# -------------------------------------------------------------------- log


def write_log(path: Path, led: Ledger, args: argparse.Namespace,
              ok: bool, reasons: list[str]) -> dict:
    docs = sorted(led.records.values(), key=lambda r: (r["stratum"], r["stem"]))
    per = {}
    for s in STRATA:
        rows = [d for d in docs if d["stratum"] == s]
        per[s] = {"documents": len(rows), "bytes": sum(d["bytes"] for d in rows)}
    log = {
        "tool": "tools/fetch_corpus_v5.py",
        "schema": "kld5-corpus-fetch-log/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "out": str(led.out),
        "user_agent": UA["User-Agent"],
        "excluded_roots": [str(p) for p in led.exclude_roots],
        "pins": {
            "gutendex_pages": GUTENDEX_PAGES,
            "pageview_months_en": list(PV_MONTHS_EN),
            "pageview_months_other": list(PV_MONTHS_OTHER),
            "languages": list(LANGS),
            "arxiv_categories": list(ARXIV_CATS),
            "arxiv_windows": list(ARXIV_WINDOWS),
            "arxiv_page": ARXIV_PAGE,
            "code": [{"label": l, "repo": r, "tag": t} for l, r, t, _ in CODE_PINS],
            "caps": {"literary": CAP_LITERARY, "wiki": CAP_WIKI, "code": CAP_CODE},
            "min_doc_bytes": MIN_DOC_BYTES,
        },
        "requested": {"strata": list(args.strata), "target_bytes": args.target_bytes,
                      "min_docs": args.min_docs, "stratum_targets": TARGETS},
        "documents": docs,
        "skipped": led.skipped,
        "failures": led.failures,
        "totals": {
            "documents": len(docs),
            "bytes": sum(d["bytes"] for d in docs),
            "chars": sum(d["chars"] for d in docs),
            "per_stratum": per,
            "failures": len(led.failures),
            "skipped": len(led.skipped),
        },
        "ok": ok,
        "unmet": reasons,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1, sort_keys=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return log


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch the v5 held-out corpus (>=60 MB clean UTF-8, "
                    ">=250 distinct source documents) into a strata tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="corpus root")
    ap.add_argument("--log", type=Path, default=None,
                    help="fetch log path (default: <out>/../corpus_fetch_log.json)")
    ap.add_argument("--strata", default=",".join(STRATA),
                    help="comma-separated strata to fetch")
    ap.add_argument("--target-bytes", type=int, default=TOTAL_TARGET,
                    help="minimum total clean bytes required to succeed")
    ap.add_argument("--min-docs", type=int, default=MIN_DOCS,
                    help="minimum distinct documents required to succeed")
    ap.add_argument("--exclude-root", action="append", default=None,
                    help="earlier corpus root to check stems against (repeatable)")
    ap.add_argument("--inventory-only", action="store_true",
                    help="do not fetch; hash what is on disk and rewrite the log")
    args = ap.parse_args()

    args.strata = [s for s in args.strata.split(",") if s]
    bad = [s for s in args.strata if s not in STRATA]
    if bad:
        ap.error(f"unknown strata: {bad}; known: {list(STRATA)}")
    roots = [Path(p) for p in (args.exclude_root or EXCLUDE_DEFAULT)]
    missing = [str(p) for p in roots if not p.is_dir()]
    if missing and args.exclude_root is None:
        print(f"note: exclude roots absent, no stem check against {missing}", flush=True)
    roots = [p for p in roots if p.is_dir()]
    log_path = args.log or (args.out.parent / "corpus_fetch_log.json")

    args.out.mkdir(parents=True, exist_ok=True)
    led = Ledger(args.out, roots)
    print(f"prior stems known: {len(led.prior)}; already on disk: {len(led.records)} "
          f"({led.total_bytes()//1000} KB)", flush=True)

    stages = (("literary", literary), ("scientific", scientific),
              ("encyclopedic", encyclopedic), ("multilingual", multilingual),
              ("code", code))
    try:
        if not args.inventory_only:
            for name, fn in stages:
                if name not in args.strata:
                    continue
                print(f"{name} (target {TARGETS[name]//1_000_000} MB)", flush=True)
                fn(led, TARGETS[name])
    except Collision as e:
        write_log(log_path, led, args, False, [f"stem collision: {e}"])
        print(f"FATAL stem collision: {e}", file=sys.stderr, flush=True)
        return 2

    total = sum(r["bytes"] for r in led.records.values())
    reasons = []
    if total < args.target_bytes:
        reasons.append(f"total bytes {total} < target {args.target_bytes}")
    if len(led.records) < args.min_docs:
        reasons.append(f"documents {len(led.records)} < min {args.min_docs}")
    for s in args.strata:
        if not any(r["stratum"] == s for r in led.records.values()):
            reasons.append(f"stratum {s} is empty")
    log = write_log(log_path, led, args, not reasons, reasons)

    print(json.dumps({"per_stratum": log["totals"]["per_stratum"],
                      "documents": log["totals"]["documents"],
                      "bytes": log["totals"]["bytes"],
                      "failures": len(led.failures),
                      "skipped": len(led.skipped)}), flush=True)
    print(f"log: {log_path}", flush=True)
    if led.failures:
        print(f"PARTIAL: {len(led.failures)} fetch failures recorded in the log:",
              flush=True)
        for f in led.failures[:20]:
            print(f"  - {f['stage']}: {f['error']}", flush=True)
        if len(led.failures) > 20:
            print(f"  ... {len(led.failures) - 20} more", flush=True)
    if reasons:
        for r in reasons:
            print(f"UNMET: {r}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
