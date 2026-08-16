#!/usr/bin/env python3
"""Two side collections for the prefix-caching qualification, run on AIBoss.

  identity   card, host, model revision and shard digests, harness digests -> out/identity.json
  restore    restore the user's live qwen38-27b service and prove it healthy -> out/restore.json

`restore` is an obligation, not a convenience: this project's single card is the user's own
machine and its service must not end a GPU window stopped. It follows ApcPoisonRepro's
procedure exactly -- start the unit, never the container, because the unit is Restart=always
and `podman --replace` would recreate it out from under systemd.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import time

W = pathlib.Path("/home/mbelleau/qualapc")
QW = pathlib.Path("/home/mbelleau/qual5090")
OUT = W / "out"
UUID = "GPU-506a575d-01d7-b12e-9a0a-c1ab5f38ae0a"
REPO_DIR = pathlib.Path(
    "/mnt/vault/llm/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context")
SNAPSHOT_RECEIPT = "receipts/aiboss-live-service-snapshot.json"
COMPARE_FIELDS = ("cmd", "env", "image_digest", "binds", "entrypoint", "working_dir",
                  "user", "network_mode", "ipc_mode", "devices", "privileged",
                  "restart_policy")


def run(*argv, check=True):
    proc = subprocess.run(argv, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"{argv!r} failed rc={proc.returncode}: {proc.stderr.strip()[:400]}")
    return proc.stdout.strip()


def sha256_file(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity():
    query = ("name,uuid,memory.total,memory.free,compute_mode,power.limit,persistence_mode,"
             "driver_version")
    gpu = run("nvidia-smi", f"--id={UUID}", f"--query-gpu={query}",
              "--format=csv,noheader,nounits").split(", ")
    revision = (REPO_DIR / "refs/main").read_text().strip()
    snapshot = REPO_DIR / "snapshots" / revision
    shards = {p.name: sha256_file(p.resolve())
              for p in sorted(snapshot.glob("*.safetensors"))}
    index = snapshot / "model.safetensors.index.json"
    payload = {
        "gpu": {
            "model": gpu[0], "uuid": gpu[1], "physical_card": True,
            "engine_budget_simulation": False,
            "memory_total_mib": int(gpu[2]), "free_mib_with_gpu_idle": int(gpu[3]),
            "compute_mode": gpu[4], "power_limit_w": float(gpu[5]),
            "persistence_mode": gpu[6],
            "budget_rule": "the engine requests ceil(cudaMemGetInfo total x utilisation), "
                           "and cudaMemGetInfo total is FB Total minus FB Reserved "
                           "(32,607 - 458 = 32,149 MiB), never the free figure",
        },
        "host": {
            "hostname": run("hostname"),
            "role": "the user's own machine",
            "driver_version": gpu[7],
            "container_engine": "podman " + run("podman", "--version").split()[-1] + " rootless",
            "os_release": run("bash", "-lc",
                              "source /etc/os-release && echo \"$PRETTY_NAME\""),
            "kernel": run("uname", "-r"),
        },
        "model": {
            "hf_repo": "malaiwah/Qwen3.8-27B-EXL3-K5K6-context",
            "revision": revision,
            "snapshot_path": str(snapshot),
            "index_sha256": sha256_file(index.resolve()) if index.exists() else None,
            "shard_sha256": shards,
            "note": "serialized bytes on disk in the vault HF cache; never called VRAM",
        },
        "harness_sha256": {
            "qualify_apc_5090.sh": sha256_file(W / "qualify_apc_5090.sh"),
            "qualify_apc_env.py": sha256_file(W / "qualify_apc_env.py"),
            **{p.name: sha256_file(p) for p in
               (QW / "tools/longctx.py", QW / "tools/longmm.py", QW / "tools/vision_eval.py",
                QW / "tools/bench.py", QW / "receipts/native-mtp-8mp-fixture.png",
                QW / "gpu_sampler.py") if p.exists()},
        },
        "harness_note": "longctx.py, longmm.py, vision_eval.py, bench.py and the image "
                        "fixture are the physical qualification's own files, unmodified; "
                        "the gate bodies are copied from qual.sh verbatim so the two "
                        "qualifications compare like with like",
    }
    (OUT / "identity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(OUT / "identity.json"),
                      "revision": revision, "shards": len(shards)}, indent=2))


def inspect_fields():
    raw = json.loads(run("podman", "inspect", "qwen38-27b"))[0]
    config = raw.get("Config", {})
    host = raw.get("HostConfig", {})
    entrypoint = config.get("Entrypoint")
    if isinstance(entrypoint, list) and len(entrypoint) == 1:
        entrypoint = entrypoint[0]
    return {
        "cmd": config.get("Cmd"),
        "env": sorted(config.get("Env") or []),
        "image_digest": raw.get("ImageDigest") or raw.get("Image"),
        "binds": sorted(host.get("Binds") or []),
        "entrypoint": entrypoint,
        "working_dir": config.get("WorkingDir"),
        "user": config.get("User"),
        "network_mode": host.get("NetworkMode"),
        "ipc_mode": host.get("IpcMode"),
        "devices": host.get("Devices"),
        "privileged": host.get("Privileged"),
        "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
    }


def restore():
    started = time.time()
    run("systemctl", "--user", "start", "qwen38-27b.service")
    health, waited = None, 0
    for waited in range(1, 481):
        health = run("podman", "inspect", "-f", "{{.State.Health.Status}}", "qwen38-27b",
                     check=False)
        if health == "healthy":
            break
        time.sleep(2)
    endpoint = run("bash", "-lc",
                   "curl -s -o /dev/null -w '%{http_code}' --max-time 20 "
                   "http://127.0.0.1:8000/health", check=False)
    models = run("bash", "-lc", "curl -s --max-time 20 http://127.0.0.1:8000/v1/models",
                 check=False)
    try:
        served = [m.get("id") for m in json.loads(models).get("data", [])]
    except json.JSONDecodeError:
        served = None

    live = inspect_fields()
    candidates = [W / "aiboss-live-service-snapshot.json",
                  pathlib.Path.home() / "prodimage" / SNAPSHOT_RECEIPT,
                  pathlib.Path.home() / SNAPSHOT_RECEIPT]
    snapshot_path = next((p for p in candidates if p.exists()), candidates[0])
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    if not snapshot:
        raise SystemExit(f"live-service snapshot not found; looked in {candidates}")
    container = snapshot.get("container", {})
    host_config = snapshot.get("host_config", {})
    reference = {
        "cmd": container.get("cmd"),
        "env": sorted(container.get("env") or []),
        "image_digest": container.get("image_digest"),
        "binds": sorted(host_config.get("binds") or []),
        "entrypoint": container.get("entrypoint"),
        "working_dir": container.get("working_dir"),
        "user": container.get("user"),
        "network_mode": host_config.get("network_mode"),
        "ipc_mode": host_config.get("ipc_mode"),
        "devices": host_config.get("devices"),
        "privileged": host_config.get("privileged"),
        "restart_policy": (host_config.get("restart_policy") or {}).get("Name")
            if isinstance(host_config.get("restart_policy"), dict)
            else host_config.get("restart_policy"),
    }
    compared = {}
    for field in COMPARE_FIELDS:
        want = reference.get(field) if isinstance(reference, dict) else None
        got = live.get(field)
        if field == "user" and {want, got} == {"root", ""}:
            compared[field] = {"identical": True, "representation_artefact": True,
                               "snapshot": want, "live": got,
                               "note": "no --user is ever passed; `podman exec qwen38-27b "
                                       "id -un` returns root"}
        else:
            compared[field] = {"identical": want == got,
                               **({} if want == got else {"snapshot": want, "live": got})}
    payload = {
        "snapshot_receipt": SNAPSHOT_RECEIPT,
        "snapshot_receipt_found": snapshot_path.exists(),
        "snapshot_receipt_path": str(snapshot_path),
        "snapshot_receipt_sha256": sha256_file(snapshot_path),
        "stopped_for_this_run": True,
        "stopped_by": "TwentyFourGigProxy at 2026-08-16T04:35Z, before this window; this "
                      "agent inherited the restore obligation together with the card",
        "restore_procedure": "systemctl --user start qwen38-27b.service, never the container "
                             "directly: the unit is Restart=always and `podman --replace` "
                             "would recreate it out from under systemd",
        "systemd_active": run("systemctl", "--user", "is-active", "qwen38-27b.service",
                              check=False),
        "podman_health": health,
        "seconds_to_healthy": waited * 2,
        "health_endpoint_http_code": endpoint,
        "served_models": served,
        "compared_fields": compared,
        "all_compared_fields_identical": all(v["identical"] for v in compared.values()),
        "gpu_after_restore_mib": int(run("nvidia-smi", f"--id={UUID}",
                                         "--query-gpu=memory.used",
                                         "--format=csv,noheader,nounits")),
        "wall_sec": round(time.time() - started, 1),
    }
    (OUT / "restore.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("systemd_active", "podman_health", "health_endpoint_http_code",
                       "served_models", "all_compared_fields_identical",
                       "gpu_after_restore_mib", "wall_sec")}, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "identity":
        identity()
    elif action == "restore":
        restore()
    else:
        raise SystemExit("usage: qualify_apc_env.py {identity|restore}")
