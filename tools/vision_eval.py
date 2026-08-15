#!/usr/bin/env python3
"""Paired multimodal check: does quantizing the vision tower to K6 change what the model sees?

Every multimodal claim in this family so far has been "an image answers correctly", which is a
smoke test. This generates deterministic synthetic images with exactly known ground truth,
scores answers by exact match, and is meant to be run against two builds so the comparison is
paired on identical inputs.

Three task families, chosen because they are unambiguous and need no external dataset:

  digits  a 6-digit code drawn with a 5x7 bitmap font - tests fine detail, the first thing a
          coarser vision encoder should lose
  bars    a bar chart with distinct integer heights - tests spatial ordering ("which bar is
          tallest", indexed from the left)
  grid    a k x k grid of primary-coloured cells - tests counting and localisation

No fonts, no PIL: PNGs are written with zlib so the images are byte-identical across runs.

    vision_eval.py --url http://127.0.0.1:8120/v1 --label ctx --out vision-ctx.json
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import struct
import urllib.request
import zlib
from pathlib import Path

# 5x7 bitmap digits, one string per row, '#' = ink
FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
COLOURS = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 70, 220),
           "yellow": (240, 210, 40)}


def png(w: int, h: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in pixels)
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def blank(w: int, h: int, c=(250, 250, 250)) -> list[list[tuple[int, int, int]]]:
    return [[c for _ in range(w)] for _ in range(h)]


def draw_digits(code: str, scale: int = 14, pad: int = 30) -> bytes:
    cw, ch = 5 * scale + scale, 7 * scale
    w, h = pad * 2 + cw * len(code), pad * 2 + ch
    img = blank(w, h)
    for i, d in enumerate(code):
        ox = pad + i * cw
        for r, row in enumerate(FONT[d]):
            for c, bit in enumerate(row):
                if bit == "1":
                    for yy in range(scale):
                        for xx in range(scale):
                            img[pad + r * scale + yy][ox + c * scale + xx] = (20, 20, 20)
    return png(w, h, img)


def draw_bars(heights: list[int], scale: int = 26, pad: int = 24) -> bytes:
    n = len(heights)
    w = pad * 2 + n * scale * 2
    h = pad * 2 + max(heights) * scale
    img = blank(w, h)
    for i, ht in enumerate(heights):
        x0 = pad + i * scale * 2
        for y in range(ht * scale):
            for x in range(scale):
                img[h - pad - 1 - y][x0 + x] = (40, 70, 220)
    return png(w, h, img)


def draw_grid(cells: list[list[str]], scale: int = 46, pad: int = 16) -> bytes:
    k = len(cells)
    w = h = pad * 2 + k * scale
    img = blank(w, h)
    for r in range(k):
        for c in range(k):
            col = COLOURS[cells[r][c]]
            for yy in range(scale - 4):
                for xx in range(scale - 4):
                    img[pad + r * scale + yy][pad + c * scale + xx] = col
    return png(w, h, img)


def ask(url: str, image: bytes, question: str, timeout: int = 600) -> str:
    payload = {"model": "m", "max_tokens": 64, "temperature": 0,
               # thinking mode eats the whole budget before the answer
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{
        "role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image).decode()}},
            {"type": "text", "text": question}]}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cases", type=int, default=10, help="cases per task family")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--dump-dir", default=None, help="write the generated PNGs for inspection")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.cases):
        code = "".join(str(rng.randrange(10)) for _ in range(6))
        rows.append({"task": "digits", "case": i, "truth": code, "image": draw_digits(code),
                     "question": "Read the 6-digit number in this image. Answer with the digits only."})
    for i in range(args.cases):
        n = rng.randrange(4, 7)
        heights = rng.sample(range(2, 12), n)
        tallest = heights.index(max(heights)) + 1
        rows.append({"task": "bars", "case": i, "truth": str(tallest),
                     "image": draw_bars(heights),
                     "question": (f"This bar chart has {n} bars. Counting from the left starting "
                                  "at 1, which bar is the tallest? Answer with the number only.")})
    for i in range(args.cases):
        k = rng.choice((3, 4))
        names = list(COLOURS)
        cells = [[rng.choice(names) for _ in range(k)] for _ in range(k)]
        want = rng.choice(names)
        count = sum(row.count(want) for row in cells)
        rows.append({"task": "grid", "case": i, "truth": str(count),
                     "image": draw_grid(cells),
                     "question": (f"This is a {k}x{k} grid of coloured squares. How many squares "
                                  f"are {want}? Answer with the number only.")})

    if args.dump_dir:
        d = Path(args.dump_dir); d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            (d / f"{r['task']}-{r['case']:02d}.png").write_bytes(r["image"])

    results, score = [], {}
    for r in rows:
        try:
            answer = ask(args.url, r["image"], r["question"])
        except Exception as exc:
            answer = f"ERROR {type(exc).__name__}: {exc}"
        digits = "".join(ch for ch in answer if ch.isdigit())
        ok = digits == r["truth"] if r["task"] == "digits" else digits[:3] == r["truth"]
        results.append({"task": r["task"], "case": r["case"], "truth": r["truth"],
                        "answer": answer.strip()[:60], "extracted": digits[:8], "correct": ok})
        s = score.setdefault(r["task"], {"n": 0, "ok": 0})
        s["n"] += 1
        s["ok"] += int(ok)
        print(f"{r['task']}-{r['case']:02d} truth={r['truth']:>8} got={digits[:8]:>8} "
              f"{'OK' if ok else 'MISS'}", flush=True)

    report = {"schema": "qwen38-vision-eval/1", "label": args.label, "seed": args.seed,
              "per_task": {k: {"n": v["n"], "correct": v["ok"],
                               "accuracy": v["ok"] / v["n"] if v["n"] else None}
                           for k, v in score.items()},
              "overall": {"n": len(results), "correct": sum(r["correct"] for r in results),
                          "accuracy": sum(r["correct"] for r in results) / len(results)},
              "results": results}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["per_task"] | {"overall": report["overall"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
