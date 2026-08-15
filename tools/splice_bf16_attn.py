#!/usr/bin/env python3
"""Splice BF16 attention weights over an EXL3-quantized checkpoint.

exllamav3's converter has no per-module bit override and cannot emit BF16 for a
decoder linear (unquantized linears round-trip as FP16 via
`Linear.load_fp16(float2half=True)`).  So we convert everything at K4, then
replace the attention projections' trellis payloads with the source BF16
weights.  The Gilded Gnosis EXL3 loader claims exactly those BF16 `LinearBase`
modules for its `ONLINE_QUANT=exl3-b6` overlay and encodes them to K6 in VRAM.

Result: MLP + lm_head + MTP stay EXL3 (in `tensor_storage`), attention ships
BF16 (absent from `tensor_storage`, hence online-K6 eligible).

This is a publication build, so every check below is fail-closed: a mismatch
aborts with a non-zero exit instead of writing a checkpoint that only looks
right.  `assert` is deliberately not used -- `python -O` deletes it.

Emits `splice-receipt.json` in the output directory:

    schema                  "qwen38-splice-receipt/1"
    quant_dir               EXL3 conversion the shards were streamed from
    source_checkpoint       BF16 checkpoint the attention weights came from
    source_index_sha256     sha256 of <source_checkpoint>/model.safetensors.index.json
    quant_index_sha256      sha256 of <quant_dir>/model.safetensors.index.json
    include_mtp             whether the MTP draft head was also returned to BF16
    modules_replaced        module counts by role, plus "total"
    modules_expected        the counts the build refused to deviate from
    exl3_tensors_dropped    EXL3 payload tensors removed from the quant shards
    bf16_tensors_spliced    BF16 tensors copied in from the source
    replaced_tensors        sorted names of every tensor copied in from the source
    shards                  per output shard: sha256, bytes, tensors
    total_bytes             sum of the output shard sizes

After this, regenerate metadata:
    python util/add_safetensors_index.py -m OUT
    python util/add_quant_config.py -m OUT
"""
import argparse, hashlib, json, os, re, shutil, struct
from safetensors import safe_open
from safetensors.torch import save_file

# Projections that must ship BF16: the attention stack (claimed by the online-K6
# overlay) and the MTP draft module (kept BF16 like both NVFP4 vendors, so
# --speculative-config '{"method":"mtp"}' keeps an unquantized draft head).
ATTN_ONLY = (
    r"^model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj|in_proj_b|in_proj_a)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))$"
)
WITH_MTP = (
    r"^(model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj|in_proj_b|in_proj_a)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))"
    r"|mtp\.(fc|layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))))$"
)
EXL3_SUFFIX = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1", "scale")

# Qwen3.8-27B topology, from config.json -> layer_types: 48 linear_attention
# layers x {in_proj_qkv, in_proj_z, out_proj} and 16 full_attention layers x
# {q,k,v,o}_proj.  in_proj_a/in_proj_b are rank-projections the converter leaves
# unquantized, so they never appear as EXL3 modules.  The MTP head is
# mtp.fc plus layers.0.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}.
EXPECT_MODULES = {"linear_attention": 144, "full_attention": 64}
EXPECT_MTP_MODULES = 8


def module_of(key: str) -> tuple[str, str]:
    head, _, tail = key.rpartition(".")
    return head, tail


def role_of(mod: str) -> str:
    if mod.startswith("mtp"):
        return "mtp_draft"
    if ".linear_attn." in mod:
        return "linear_attention"
    if ".self_attn." in mod:
        return "full_attention"
    return "other"


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def prepare_out_dir(out_dir: str, force: bool) -> None:
    """Refuse to write into a directory that already holds a build.

    A leftover shard from a previous run with a different shard count keeps its
    filename, survives the rewrite, and gets picked up by add_safetensors_index.py
    -- the checkpoint then indexes tensors from two different builds.
    """
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise SystemExit(f"output path {out_dir} exists and is not a directory")
    existing = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    if existing and not force:
        raise SystemExit(
            f"refusing to build into non-empty directory {out_dir} "
            f"({len(existing)} entries, e.g. {existing[:3]}): stale shards from a "
            f"previous build would survive alongside the new ones. Pass --force to "
            f"clear its shards, index and receipt and build anyway."
        )
    os.makedirs(out_dir, exist_ok=True)
    stale = [fn for fn in existing
             if fn.endswith(".safetensors")
             or fn in ("model.safetensors.index.json", "splice-receipt.json")]
    for fn in stale:
        os.remove(f"{out_dir}/{fn}")
    if stale:
        print(f"--force: removed {len(stale)} stale shard/index/receipt files from {out_dir}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quant_dir", required=True, help="EXL3 K4 conversion output")
    ap.add_argument("-s", "--src_dir", required=True, help="original BF16 checkpoint")
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("-ss", "--shard_size", type=int, default=8192, help="MB")
    ap.add_argument("--include-mtp", action="store_true",
                    help="also return the MTP draft module to BF16 (costs ~0.64 GB)")
    ap.add_argument("--force", action="store_true",
                    help="build into a non-empty output directory, clearing its shards first")
    args = ap.parse_args()
    splice_re = re.compile(WITH_MTP if args.include_mtp else ATTN_ONLY)
    prepare_out_dir(args.out_dir, args.force)

    q_index_path = f"{args.quant_dir}/model.safetensors.index.json"
    s_index_path = f"{args.src_dir}/model.safetensors.index.json"
    q_index = json.load(open(q_index_path))["weight_map"]
    s_index = json.load(open(s_index_path))["weight_map"]

    # 1. which attention modules exist in the quant as EXL3 payloads, and what
    #    logical shape each one dequantizes to.  exl3 stores the input-side
    #    scales in `suh` (length K) and the output-side in `svh` (length N), so
    #    the logical weight is (N, K) -- the same orientation as the source.
    q_headers: dict[str, dict] = {}
    for shard in sorted(set(q_index.values())):
        for key, meta in read_header(f"{args.quant_dir}/{shard}").items():
            if key != "__metadata__":
                q_headers[key] = meta

    wanted: set[str] = set()
    for key in q_index:
        mod, suf = module_of(key)
        if suf in EXL3_SUFFIX and splice_re.match(mod):
            wanted.add(mod)

    expect = dict(EXPECT_MODULES)
    if args.include_mtp:
        expect["mtp_draft"] = EXPECT_MTP_MODULES
    found: dict[str, int] = {role: 0 for role in expect}
    unexpected: list[str] = []
    for mod in sorted(wanted):
        role = role_of(mod)
        if role in found:
            found[role] += 1
        else:
            unexpected.append(mod)
    if unexpected:
        raise SystemExit(
            f"{len(unexpected)} module(s) matched the splice pattern but fall in no "
            f"expected role, e.g. {unexpected[:5]}"
        )
    if found != expect:
        raise SystemExit(
            f"module count mismatch: found {found}, expected {expect} "
            f"(total {sum(found.values())} vs {sum(expect.values())}). The topology "
            f"changed or the conversion is incomplete; refusing to build."
        )
    print(f"{sum(found.values())} attention modules to replace with BF16: "
          f"{', '.join(f'{r}={n}' for r, n in sorted(found.items()))}", flush=True)

    logical_shape: dict[str, list[int]] = {}
    for mod in sorted(wanted):
        suh, svh = q_headers.get(f"{mod}.suh"), q_headers.get(f"{mod}.svh")
        if suh is None or svh is None:
            raise SystemExit(f"{mod}: EXL3 module has no suh/svh, cannot verify its shape")
        logical_shape[mod] = [svh["shape"][0], suh["shape"][0]]

    missing = [f"{m}.weight" for m in sorted(wanted) if f"{m}.weight" not in s_index]
    if missing:
        raise SystemExit(f"source lacks {len(missing)} weights, e.g. {missing[:3]}")

    # 2. stream the quant shards, dropping the EXL3 payloads of those modules
    limit = args.shard_size * 1024 * 1024
    out_shards: list[dict] = []
    cur: dict = {}
    cur_bytes = 0
    dropped = 0

    def flush():
        nonlocal cur, cur_bytes
        if cur:
            out_shards.append(cur)
            cur, cur_bytes = {}, 0

    for shard in sorted(set(q_index.values())):
        with safe_open(f"{args.quant_dir}/{shard}", framework="pt") as f:
            for key in f.keys():
                mod, suf = module_of(key)
                if suf in EXL3_SUFFIX and mod in wanted:
                    dropped += 1
                    continue
                t = f.get_tensor(key)
                nbytes = t.numel() * t.element_size()
                if cur_bytes + nbytes > limit:
                    flush()
                cur[key] = t
                cur_bytes += nbytes
        print(f"  read {shard}", flush=True)
    flush()

    expect_dropped = sum(1 for key in q_headers
                         if module_of(key)[1] in EXL3_SUFFIX and module_of(key)[0] in wanted)
    if dropped != expect_dropped:
        raise SystemExit(f"dropped {dropped} EXL3 tensors, headers list {expect_dropped}")

    # 3. append the BF16 attention weights (and biases) from the source
    by_src: dict[str, list[str]] = {}
    for mod in sorted(wanted):
        for suf in ("weight", "bias"):
            k = f"{mod}.{suf}"
            if k in s_index:
                by_src.setdefault(s_index[k], []).append(k)
    replaced: list[str] = []
    for shard, keys in sorted(by_src.items()):
        with safe_open(f"{args.src_dir}/{shard}", framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                mod, suf = module_of(k)
                if str(t.dtype) != "torch.bfloat16":
                    raise SystemExit(f"{k} is {t.dtype}, expected torch.bfloat16; "
                                     f"the source checkpoint is not the BF16 release")
                want = logical_shape[mod] if suf == "weight" else [logical_shape[mod][0]]
                if list(t.shape) != want:
                    raise SystemExit(f"{k} has shape {list(t.shape)}, but the EXL3 module "
                                     f"it replaces dequantizes to {want}")
                nbytes = t.numel() * t.element_size()
                if cur_bytes + nbytes > limit:
                    flush()
                cur[k] = t
                cur_bytes += nbytes
                replaced.append(k)
        print(f"  spliced from {shard}", flush=True)
    flush()

    expect_replaced = sorted(k for mod in wanted for k in (f"{mod}.weight", f"{mod}.bias")
                             if k in s_index)
    replaced.sort()
    if replaced != expect_replaced:
        raise SystemExit(f"spliced {len(replaced)} tensors, expected {len(expect_replaced)}")
    leftover = sorted(k for shard in out_shards for k in shard
                      if module_of(k)[1] in EXL3_SUFFIX and module_of(k)[0] in wanted)
    if leftover:
        raise SystemExit(f"{len(leftover)} EXL3 payloads of replaced modules survived, "
                         f"e.g. {leftover[:3]}")
    print(f"verified {len(replaced)} BF16 tensors: dtype bfloat16, shape equal to the "
          f"EXL3 module each one replaces", flush=True)

    # 4. write shards
    n = len(out_shards)
    total = 0
    shard_receipt: dict[str, dict] = {}
    for i, tensors in enumerate(out_shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(tensors, f"{args.out_dir}/{name}", metadata={"format": "pt"})
        sz = os.path.getsize(f"{args.out_dir}/{name}")
        total += sz
        shard_receipt[name] = {"sha256": sha256_file(f"{args.out_dir}/{name}"),
                               "bytes": sz, "tensors": len(tensors)}
        print(f"  wrote {name} {sz/1e9:.3f} GB ({len(tensors)} tensors)", flush=True)

    # 5. carry over everything that is not weights
    for fn in os.listdir(args.quant_dir):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = f"{args.quant_dir}/{fn}"
        if os.path.isfile(src):
            shutil.copy2(src, f"{args.out_dir}/{fn}")

    receipt = {
        "schema": "qwen38-splice-receipt/1",
        "quant_dir": os.path.abspath(args.quant_dir),
        "source_checkpoint": os.path.abspath(args.src_dir),
        "source_index_sha256": sha256_file(s_index_path),
        "quant_index_sha256": sha256_file(q_index_path),
        "include_mtp": args.include_mtp,
        "modules_replaced": {**found, "total": sum(found.values())},
        "modules_expected": {**expect, "total": sum(expect.values())},
        "exl3_tensors_dropped": dropped,
        "bf16_tensors_spliced": len(replaced),
        "replaced_tensors": replaced,
        "shards": shard_receipt,
        "total_bytes": total,
    }
    with open(f"{args.out_dir}/splice-receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")

    print(f"dropped {dropped} EXL3 tensors, spliced {len(replaced)} BF16 tensors, "
          f"{n} shards, {total/1e9:.2f} GB", flush=True)
    print("wrote splice-receipt.json", flush=True)
    print("now run: util/add_safetensors_index.py then util/add_quant_config.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
