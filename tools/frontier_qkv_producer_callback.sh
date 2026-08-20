#!/bin/bash
# Produce and inspect one real gated-Qwen fused-uniform QKV checkpoint fragment.
set -euo pipefail
[[ "${FRONTIER_TRANSACTION_ACTIVE:-}" == "1" ]] || exit 2
SOURCE=/home/mbelleau/final-frontier-g0/converter-source
ENTRY=/home/mbelleau/final-frontier-g0/tools/frontier_converter_deterministic.py
PLAN=/home/mbelleau/final-frontier-g0/qkv-plan.json
BF16_REPO=/home/mbelleau/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B
EXT_CACHE_SOURCE=/home/mbelleau/final-frontier-g0/cache/converter-base-ext
for path in "$SOURCE" "$BF16_REPO" "$EXT_CACHE_SOURCE"; do [[ -d "$path" && ! -L "$path" ]] || exit 2; done
for path in "$ENTRY" "$PLAN"; do [[ -f "$path" && ! -L "$path" && -s "$path" ]] || exit 2; done
cp -a "$EXT_CACHE_SOURCE/." "${FRONTIER_CAMPAIGN_CACHE_DIR}/"
LOG="${FRONTIER_CAMPAIGN_WORK_DIR}/producer.txt"
EVIDENCE="${FRONTIER_CAMPAIGN_WORK_DIR}/producer-evidence.json"
SUMMARY="${FRONTIER_CAMPAIGN_WORK_DIR}/producer-log.json"

podman run --rm --replace --name "${FRONTIER_CAMPAIGN_CONTAINER}" --network none --ipc=host --device nvidia.com/gpu=all \
  --tmpfs /usr/local/cuda-13.2/lib64:rw,size=16m \
  -v "${SOURCE}:/src:ro" -v "${ENTRY}:/opt/frontier/frontier_converter_deterministic.py:ro" -v "${PLAN}:/inputs/qkv-plan.json:ro" \
  -v "${BF16_REPO}:/models/bf16-repo:ro" -v "${FRONTIER_CAMPAIGN_CACHE_DIR}:/cache:rw" -v "${FRONTIER_CAMPAIGN_WORK_DIR}:/work:rw" \
  -e CUDA_HOME=/usr/local/cuda-13.2 -e TORCH_CUDA_ARCH_LIST=12.0a -e TORCH_EXTENSIONS_DIR=/cache -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /bin/bash "${FRONTIER_CAMPAIGN_IMAGE}" \
  -lc "set -euo pipefail; ln -sf /usr/local/cuda-13.2/targets/x86_64-linux/lib/* /usr/local/cuda-13.2/lib64/; export PYTHONPATH=/src; exec /opt/venv/bin/python /opt/frontier/frontier_converter_deterministic.py -i /models/bf16-repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 -w /work/pilot -o /work/pilot-out -b 6 -hb 6 -mb 4 -vb 16 -cr 2 -cc 128 -cpi 0 -cb mcg -d 0 --max_module 4 --qkv_topology_plan /inputs/qkv-plan.json" \
  >"$LOG" 2>&1

python3 - "$FRONTIER_CAMPAIGN_WORK_DIR" "$EVIDENCE" "$SUMMARY" <<'PY'
import hashlib, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); evidence_path=pathlib.Path(sys.argv[2]); summary_path=pathlib.Path(sys.argv[3])

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
args=json.loads((root/'pilot/args.json').read_text(encoding='utf-8'))
topology=args.get('exl3_qkv_topology') or args.get('qkv_topology_plan')
if not isinstance(topology,dict): raise SystemExit('converter emitted no QKV topology')
rows=topology.get('layers')
row=next((item for item in rows or [] if item.get('layer')=='model.language_model.layers.3.self_attn'),None)
if not isinstance(row,dict) or row.get('variant')!='fused_uniform': raise SystemExit('fused QKV metadata mismatch')
from safetensors import safe_open
fragment=root/'pilot/qtensors/model.language_model.layers.3.safetensors'
with safe_open(str(fragment),framework='pt',device='cpu') as handle:
 keys=sorted(handle.keys())
 trellis_key='model.language_model.layers.3.self_attn.qkv_proj.trellis'
 if trellis_key not in keys: raise SystemExit('fused QKV trellis is missing')
 trellis_shape=list(handle.get_slice(trellis_key).get_shape())
fused=[key for key in keys if '.self_attn.qkv_proj.' in key]
split=[key for key in keys if any('.self_attn.'+name+'.' in key for name in ('q_proj','k_proj','v_proj'))]
required={suffix for suffix in ('trellis','suh','svh','mcg') if any(key.endswith('.qkv_proj.'+suffix) for key in fused)}
if required!={'trellis','suh','svh','mcg'} or split: raise SystemExit('fused payload exclusivity failed')
output_splits=[12288,1024,1024]
if len(trellis_shape)!=3 or trellis_shape[1]*16!=sum(output_splits) or trellis_shape[2]//16!=row.get('K'): raise SystemExit('fused QKV physical geometry mismatch')
log=root/'producer.txt'; text=log.read_text(encoding='utf-8',errors='replace')
if any(token in text.lower() for token in ('traceback','cuda out of memory','non-finite')): raise SystemExit('producer log contains fatal marker')
evidence={'schema':'qwen38-frontier-qkv-producer-evidence/1','status':'pass','source_commit':'a71fbd8f841fd8772f4a411e43686f15fb16f166','layer':row['layer'],'variant':row['variant'],'output_splits':output_splits,'K':row['K'],'codebook':row['codebook'],'trellis_shape':trellis_shape,'payload_fragment_sha256':sha(fragment),'payload_key_count':len(keys),'fused_keys':fused,'split_keys':split,'payload_exclusive':True,'converter_log_sha256':sha(log)}
summary={'schema':'qwen38-frontier-qkv-producer-log/1','status':'pass','converter_log':{'path':'producer.txt','sha256':sha(log),'bytes':log.stat().st_size},'evidence_sha256':hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
for path,value in ((evidence_path,evidence),(summary_path,summary)):
 payload=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'; tmp=path.with_name(path.name+'.tmp')
 with tmp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
PY
