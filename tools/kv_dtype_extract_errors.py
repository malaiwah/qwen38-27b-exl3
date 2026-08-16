#!/usr/bin/env python3
"""Re-extract verbatim exception lines from the saved server logs.

The driver's inline extractor grabs the first traceback frame; for a refusal
census the line that matters is the exception itself. The full logs are on
disk, so this is a pure re-read, never a re-run.
"""
import json
import re
import sys
from pathlib import Path

W = Path("/home/mbelleau/kvsweep")
EXC = re.compile(r"^(?:\(\w+ pid=\d+\)\s*)?(\w*(?:Error|Exception)): (.*)$")


def digest(logfile: Path) -> dict:
    text = logfile.read_text(errors="ignore")
    lines = [l.rstrip() for l in text.splitlines()]
    exc = []
    for l in lines:
        body = l
        m = re.match(r"^\(\w+ pid=\d+\)\s*(.*)$", l)
        if m:
            body = m.group(1)
        if EXC.match(body) or re.match(r"^usage: vllm", body) or "invalid choice" in body:
            exc.append(l)
    # dedupe preserving order
    seen, out = set(), []
    for l in exc:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return {"exception_lines": out, "log_lines": len(lines)}


def main() -> int:
    target = sys.argv[1]
    path = W / "out" / f"{target}.json"
    data = json.loads(path.read_text())
    for key, rec in data.items():
        tag = rec.get("tag")
        if not tag:
            continue
        lf = W / "logs" / f"server-{tag}.log"
        if lf.exists():
            rec.setdefault("refusal", {})["exception_lines"] = digest(lf)["exception_lines"]
    path.write_text(json.dumps(data, indent=1))
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
