#!/usr/bin/env python3
"""Pull an OCI/Docker image and flatten its layers into a rootfs directory.

Unprivileged: no docker/podman/skopeo. Device nodes, setuid bits and ownership
are dropped (we run everything as the invoking uid), whiteouts are honoured.
"""
import datetime, hashlib, json, os, shutil, stat, sys, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "voipmonitor/vllm"
DIGEST = "sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
BLOBS = "/var/tmp/gg-blobs"
ROOT = "/var/tmp/gg-rootfs"
PROVENANCE = "/var/tmp/work/gg-rootfs-provenance.json"
FILE_MANIFEST = "/var/tmp/work/gg-rootfs-files.jsonl"
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
    expected = layer["digest"].split(":", 1)[1]
    if (os.path.exists(dst) and os.path.getsize(dst) == layer["size"]
            and sha256_file(dst) == expected):
        return i, dst, "cached-authenticated"
    got = get(f"blobs/{layer['digest']}", "*/*", tok, out=dst)
    if got != expected:
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def rootfs_entries(root, relative=""):
    directory = os.path.join(root, relative)
    for entry in sorted(os.scandir(directory), key=lambda item: item.name):
        rel = os.path.join(relative, entry.name)
        info = entry.stat(follow_symlinks=False)
        record = {"path": rel, "mode": stat.S_IMODE(info.st_mode)}
        if stat.S_ISLNK(info.st_mode):
            record.update({"type": "symlink", "target": os.readlink(entry.path)})
        elif stat.S_ISDIR(info.st_mode):
            record["type"] = "directory"
        elif stat.S_ISREG(info.st_mode):
            record.update({
                "type": "file",
                "size": info.st_size,
                "sha256": sha256_file(entry.path),
            })
        else:
            record.update({"type": "special", "rdev": info.st_rdev})
        yield record
        if record["type"] == "directory":
            yield from rootfs_entries(root, rel)


def canonical_line(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


def write_rootfs_provenance(root=ROOT, manifest_path=FILE_MANIFEST,
                            provenance_path=PROVENANCE):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    temporary = manifest_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in rootfs_entries(root):
            handle.write(canonical_line(record))
    os.replace(temporary, manifest_path)
    provenance = {
        "schema": "qwen38-rootfs-provenance/2",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": f"{REPO}@{DIGEST}",
        "rootfs": str(os.path.realpath(root)),
        "coverage": "full-immutable-rootfs",
        "file_manifest": str(os.path.realpath(manifest_path)),
        "file_manifest_sha256": sha256_file(manifest_path),
    }
    temporary = provenance_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, provenance_path)
    return provenance


def verify_rootfs_provenance(provenance_path, image, root):
    if not os.path.isfile(provenance_path):
        raise SystemExit(f"rootfs provenance is missing: {provenance_path}")
    if not os.path.isdir(root):
        raise SystemExit(f"rootfs is missing: {root}")
    provenance = json.load(open(provenance_path, encoding="utf-8"))
    if provenance.get("schema") != "qwen38-rootfs-provenance/2":
        raise SystemExit("unsupported rootfs provenance schema")
    if provenance.get("image") != image:
        raise SystemExit("rootfs provenance image mismatch")
    if provenance.get("rootfs") != str(os.path.realpath(root)):
        raise SystemExit("rootfs provenance path mismatch")
    if provenance.get("coverage") != "full-immutable-rootfs":
        raise SystemExit("rootfs provenance is not a full immutable manifest")
    manifest_path = provenance.get("file_manifest")
    if not isinstance(manifest_path, str) or not os.path.isfile(manifest_path):
        raise SystemExit("rootfs file manifest is missing")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != provenance.get("file_manifest_sha256"):
        raise SystemExit("rootfs file-manifest digest mismatch")
    with open(manifest_path, encoding="utf-8") as expected:
        for record in rootfs_entries(root):
            line = expected.readline()
            if not line:
                raise SystemExit(f"unmanifested rootfs entry: {record['path']}")
            if line != canonical_line(record):
                expected_record = json.loads(line)
                raise SystemExit(
                    "rootfs entry mismatch: "
                    f"expected {expected_record.get('path')}, observed {record['path']}"
                )
        extra = expected.readline()
        if extra:
            raise SystemExit(f"manifested rootfs entry is missing: {json.loads(extra)['path']}")
    print(manifest_path)
    print(manifest_sha256)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        if len(sys.argv) != 5:
            raise SystemExit(
                "usage: pull_rootfs.py verify PROVENANCE IMAGE ROOTFS"
            )
        verify_rootfs_provenance(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if len(sys.argv) != 1:
        raise SystemExit("usage: pull_rootfs.py [verify PROVENANCE IMAGE ROOTFS]")
    os.makedirs(BLOBS, exist_ok=True)
    tok = token()
    raw_manifest = get(f"manifests/{DIGEST}", ACCEPT, tok)
    if hashlib.sha256(raw_manifest).hexdigest() != DIGEST.split(":", 1)[1]:
        raise SystemExit("registry manifest digest mismatch")
    man = json.loads(raw_manifest)
    layers = man["layers"]
    total = sum(layer["size"] for layer in layers)
    print(f"{len(layers)} layers, {total/1e9:.2f} GB compressed", flush=True)
    config_path = f"{BLOBS}/config.json"
    config_sha256 = get(f"blobs/{man['config']['digest']}", "*/*", tok, out=config_path)
    if config_sha256 != man["config"]["digest"].split(":", 1)[1]:
        raise SystemExit("image config digest mismatch")
    with ThreadPoolExecutor(6) as ex:
        got = sorted(ex.map(fetch_layer, [(tok, i, layer) for i, layer in enumerate(layers)]))
    print("all blobs authenticated; extracting in order", flush=True)
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    for i, path, how in got:
        size = os.path.getsize(path)
        print(f"layer {i:2d}/{len(layers)} {size/1e9:6.3f} GB ({how})", flush=True)
        extract(path, ROOT)
    provenance = write_rootfs_provenance()
    print(
        f"rootfs ready: {ROOT}; manifest {provenance['file_manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
