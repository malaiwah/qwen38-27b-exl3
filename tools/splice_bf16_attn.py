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

After this, regenerate metadata:
    python util/add_safetensors_index.py -m OUT
    python util/add_quant_config.py -m OUT
"""
import argparse, json, os, re, shutil
from safetensors import safe_open
from safetensors.torch import save_file

# Projections that must ship BF16: the attention stack (claimed by the online-K6
# overlay) and the MTP draft module (kept BF16 like both NVFP4 vendors, so
# --speculative-config '{"method":"mtp"}' keeps an unquantized draft head).
SPLICE = re.compile(
    r"^(model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj|in_proj_b|in_proj_a)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))"
    r"|mtp\.(fc|layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))))$"
)
EXL3_SUFFIX = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1", "scale")


def module_of(key: str) -> tuple[str, str]:
    head, _, tail = key.rpartition(".")
    return head, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quant_dir", required=True, help="EXL3 K4 conversion output")
    ap.add_argument("-s", "--src_dir", required=True, help="original BF16 checkpoint")
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("-ss", "--shard_size", type=int, default=8192, help="MB")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    q_index = json.load(open(f"{args.quant_dir}/model.safetensors.index.json"))["weight_map"]
    s_index = json.load(open(f"{args.src_dir}/model.safetensors.index.json"))["weight_map"]

    # 1. which attention modules exist in the quant as EXL3 payloads
    wanted: set[str] = set()
    for key in q_index:
        mod, suf = module_of(key)
        if suf in EXL3_SUFFIX and ATTN.match(mod):
            wanted.add(mod)
    print(f"{len(wanted)} attention modules to replace with BF16", flush=True)

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

    # 3. append the BF16 attention weights (and biases) from the source
    by_src: dict[str, list[str]] = {}
    for mod in sorted(wanted):
        for suf in ("weight", "bias"):
            k = f"{mod}.{suf}"
            if k in s_index:
                by_src.setdefault(s_index[k], []).append(k)
    added = 0
    for shard, keys in by_src.items():
        with safe_open(f"{args.src_dir}/{shard}", framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                assert t.dtype.__str__() == "torch.bfloat16", f"{k} is {t.dtype}"
                nbytes = t.numel() * t.element_size()
                if cur_bytes + nbytes > limit:
                    flush()
                cur[k] = t
                cur_bytes += nbytes
                added += 1
        print(f"  spliced from {shard}", flush=True)
    flush()

    # 4. write shards
    n = len(out_shards)
    total = 0
    for i, tensors in enumerate(out_shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(tensors, f"{args.out_dir}/{name}", metadata={"format": "pt"})
        sz = os.path.getsize(f"{args.out_dir}/{name}")
        total += sz
        print(f"  wrote {name} {sz/1e9:.3f} GB ({len(tensors)} tensors)", flush=True)

    # 5. carry over everything that is not weights
    for fn in os.listdir(args.quant_dir):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = f"{args.quant_dir}/{fn}"
        if os.path.isfile(src):
            shutil.copy2(src, f"{args.out_dir}/{fn}")

    print(f"dropped {dropped} EXL3 tensors, spliced {added} BF16 tensors, "
          f"{n} shards, {total/1e9:.2f} GB", flush=True)
    print("now run: util/add_safetensors_index.py then util/add_quant_config.py", flush=True)


if __name__ == "__main__":
    main()
