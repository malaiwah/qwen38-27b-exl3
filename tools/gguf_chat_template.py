#!/usr/bin/env python3
"""Extract tokenizer.chat_template from a GGUF file over HTTP range requests.

Reads only the GGUF header + KV metadata section; never downloads tensor data.
Usage: gguf_chat_template.py <url> <out.jinja>
"""
import json
import struct
import sys
import urllib.request

CHUNK = 4 << 20


class RangeReader:
    def __init__(self, url, token=None):
        self.url = url
        self.token = token
        self.buf = b""
        self.pos = 0

    def _fetch(self, start, length):
        req = urllib.request.Request(self.url)
        req.add_header("Range", f"bytes={start}-{start + length - 1}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def need(self, n):
        while len(self.buf) < self.pos + n:
            got = self._fetch(len(self.buf), max(CHUNK, self.pos + n - len(self.buf)))
            if not got:
                raise EOFError("short read")
            self.buf += got

    def read(self, n):
        self.need(n)
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self):
        return struct.unpack("<q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")


# GGUF value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
FIXED = {
    T_U8: ("<B", 1), T_I8: ("<b", 1), T_U16: ("<H", 2), T_I16: ("<h", 2),
    T_U32: ("<I", 4), T_I32: ("<i", 4), T_F32: ("<f", 4), T_BOOL: ("<?", 1),
    T_U64: ("<Q", 8), T_I64: ("<q", 8), T_F64: ("<d", 8),
}


def read_value(r, t, want):
    """Read a GGUF value; when want is False, skip cheaply where possible."""
    if t in FIXED:
        fmt, sz = FIXED[t]
        return struct.unpack(fmt, r.read(sz))[0]
    if t == T_STR:
        n = r.u64()
        raw = r.read(n)
        return raw.decode("utf-8", "replace") if want else None
    if t == T_ARR:
        et = r.u32()
        n = r.u64()
        if et in FIXED:
            sz = FIXED[et][1]
            r.read(n * sz)
            return f"<array {n} x type{et}>"
        if et == T_STR:
            vals = []
            for _ in range(n):
                ln = r.u64()
                raw = r.read(ln)
                if want:
                    vals.append(raw.decode("utf-8", "replace"))
            return vals if want else f"<array {n} strings>"
        raise ValueError(f"nested array type {et}")
    raise ValueError(f"unknown gguf type {t}")


def main():
    url, out = sys.argv[1], sys.argv[2]
    token = None
    try:
        token = open("/var/tmp/hf-home-3/token").read().strip()
    except OSError:
        pass
    r = RangeReader(url, token)
    magic = r.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"not gguf: {magic!r}")
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    print(f"gguf v{ver} tensors={n_tensors} kv={n_kv}", file=sys.stderr)
    found = {}
    keys = []
    for _ in range(n_kv):
        k = r.string()
        t = r.u32()
        want = ("chat_template" in k) or k in (
            "general.architecture", "general.name", "general.basename",
            "general.quantization_version", "tokenizer.ggml.pre",
            "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
        )
        v = read_value(r, t, want)
        keys.append(k)
        if want:
            found[k] = v
    tpl = None
    for k, v in found.items():
        if "chat_template" in k and isinstance(v, str):
            tpl = v
            print(f"KEY {k} len={len(v)}", file=sys.stderr)
    print(json.dumps({k: (v if not isinstance(v, str) or len(v) < 200 else f"<{len(v)} chars>")
                      for k, v in found.items()}, indent=1), file=sys.stderr)
    print(f"kv_bytes_read={r.pos}", file=sys.stderr)
    if tpl is None:
        raise SystemExit("no chat_template KV found; keys: " + ", ".join(keys))
    with open(out, "w") as f:
        f.write(tpl)
    print(f"wrote {out} ({len(tpl)} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
