# Exact EXL3 matrix inventory from safetensors headers (read-only, no torch).
# Trellis packed shape is (K/16, N/16, 16*bits) int16 -> recovers K, N, bits.
import json, struct, glob, re, collections, os
SNAP = "/home/mbelleau/.cache/huggingface/hub/models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated/snapshots/ab3a91a13813df8096cb4c1d560ed3669035d0cf"
mats = {}
for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        shp = meta.get("shape", [])
        if name.endswith(".trellis") and len(shp) == 3:
            base = name[: -len(".trellis")]
            K, N, bits = shp[0] * 16, shp[1] * 16, shp[2] // 16
            mats[base] = (K, N, bits)
# collapse layer indices to get multiplicity
groups = collections.Counter()
detail = {}
for base, (K, N, bits) in mats.items():
    key = re.sub(r"\.layers\.\d+\.", ".layers.*.", base)
    groups[(key, K, N, bits)] += 1
    detail.setdefault((key, K, N, bits), base)
print(f"{'module':52} {'K':>6} {'N':>6} {'bits':>4} {'count':>5} {'GiB':>7}")
tot = 0.0
rows = []
for (key, K, N, bits), cnt in sorted(groups.items(), key=lambda kv: -(kv[1] * kv[0][1] * kv[0][2] * kv[0][3])):
    gib = cnt * K * N * bits / 8 / 1024**3
    tot += gib
    rows.append({"module": key, "K": K, "N": N, "bits": bits, "count": cnt, "gib": round(gib, 3)})
    print(f"{key[-52:]:52} {K:6} {N:6} {bits:4} {cnt:5} {gib:7.3f}")
print(f"\ntotal trellis weight: {tot:.2f} GiB across {sum(groups.values())} matrices")
json.dump(rows, open("/tmp/shape_inventory.json", "w"), indent=1)
print("wrote /tmp/shape_inventory.json")
