#!/usr/bin/env python3
"""Pull an OCI/Docker image and flatten its layers into a rootfs directory.

Unprivileged: no docker/podman/skopeo. Device nodes, setuid bits and ownership
are dropped (we run everything as the invoking uid), whiteouts are honoured.
"""
import json, os, shutil, sys, tarfile, urllib.request, hashlib
from concurrent.futures import ThreadPoolExecutor

REPO = "voipmonitor/vllm"
DIGEST = "sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
BLOBS = "/var/tmp/gg-blobs"
ROOT = "/var/tmp/gg-rootfs"
ACCEPT = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def token():
    u = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    return json.load(urllib.request.urlopen(u))["token"]


def get(path, accept, tok, out=None):
    req = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{REPO}/{path}",
        headers={"Authorization": f"Bearer {tok}", "Accept": accept},
    )
    with urllib.request.urlopen(req) as r:
        if out is None:
            return r.read()
        h = hashlib.sha256()
        with open(out, "wb") as f:
            while chunk := r.read(8 << 20):
                f.write(chunk)
                h.update(chunk)
        return h.hexdigest()


def fetch_layer(args):
    tok, i, layer = args
    dst = os.path.join(BLOBS, layer["digest"].split(":")[1])
    if os.path.exists(dst) and os.path.getsize(dst) == layer["size"]:
        return i, dst, "cached"
    got = get(f"blobs/{layer['digest']}", "*/*", tok, out=dst)
    if got != layer["digest"].split(":")[1]:
        raise SystemExit(f"digest mismatch on layer {i}: {got}")
    return i, dst, "fetched"


def extract(path, root):
    """Extract one layer tar over root, honouring .wh. whiteouts."""
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            name = m.name.lstrip("./")
            base = os.path.basename(name)
            target = os.path.join(root, name)
            if base.startswith(".wh."):
                if base == ".wh..wh..opq":
                    d = os.path.dirname(target)
                    if os.path.isdir(d):
                        for e in os.listdir(d):
                            p = os.path.join(d, e)
                            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
                else:
                    victim = os.path.join(os.path.dirname(target), base[4:])
                    if os.path.isdir(victim) and not os.path.islink(victim):
                        shutil.rmtree(victim, ignore_errors=True)
                    elif os.path.lexists(victim):
                        os.remove(victim)
                continue
            if m.ischr() or m.isblk() or m.isfifo():
                continue
            if os.path.lexists(target) and not (m.isdir() and os.path.isdir(target)):
                if os.path.isdir(target) and not os.path.islink(target):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    os.remove(target)
            m.mode = (m.mode | 0o700) & 0o7777 if m.isdir() else (m.mode | 0o600)
            m.uid = os.getuid(); m.gid = os.getgid()
            m.uname = m.gname = ""
            try:
                tf.extract(m, root, set_attrs=True, filter="tar")
            except (PermissionError, OSError) as e:
                print(f"  ! {name}: {e}", flush=True)


def main():
    os.makedirs(BLOBS, exist_ok=True)
    os.makedirs(ROOT, exist_ok=True)
    tok = token()
    man = json.loads(get(f"manifests/{DIGEST}", ACCEPT, tok))
    layers = man["layers"]
    total = sum(l["size"] for l in layers)
    print(f"{len(layers)} layers, {total/1e9:.2f} GB compressed", flush=True)
    with open(f"{BLOBS}/config.json", "wb") as f:
        f.write(get(f"blobs/{man['config']['digest']}", "*/*", tok))
    with ThreadPoolExecutor(6) as ex:
        got = sorted(ex.map(fetch_layer, [(tok, i, l) for i, l in enumerate(layers)]))
    print("all blobs present; extracting in order", flush=True)
    for i, path, how in got:
        sz = os.path.getsize(path)
        print(f"layer {i:2d}/{len(layers)} {sz/1e9:6.3f} GB ({how})", flush=True)
        extract(path, ROOT)
    print("rootfs ready:", ROOT, flush=True)


if __name__ == "__main__":
    main()
