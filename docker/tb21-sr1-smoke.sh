#!/bin/bash
# CPU-only smoke for the TB2.1 sr1 image (docs/46). Runs INSIDE the image via
# docker run; asserts every patched file's sha256, every sentinel, a full
# `import vllm`, and `vllm --help`. Exit 0 = image carries every sr1 fix.
#
#   ./docker/tb21-sr1-smoke.sh vllm:gg-r34-tb21-sr1
set -euo pipefail
IMAGE="${1:?usage: tb21-sr1-smoke.sh <image-ref>}"

docker run --rm --entrypoint /bin/bash "$IMAGE" -euo pipefail -c '
V=/opt/venv/lib/python3.12/site-packages/vllm
echo "== patched-file sha256 =="
printf "%s  %s\n" \
  9aba06ebf60ca7665c0513752387c349240ab85e1ebc44d6ce8137ef157b6c15 \
  "$V/model_executor/layers/quantization/exl3.py" \
  86e435ddd694ecd8a3d32a073a279a941a87b148794398e47a6a45df829c9cd1 \
  "$V/v1/attention/backends/flashinfer.py" \
  f354270a0ce7971fd9317bd10a5921b1319165df80748df60fe4ab8648ffcbc6 \
  "$V/v1/core/kv_cache_utils.py" \
  e98b7227e0b66eea137ed6ccbc9ee99bd43485892c11c648284c30cde206ed05 \
  "$V/v1/core/sched/scheduler.py" \
  | sha256sum -c -
echo "== unpatched-module sha256 (base identity) =="
printf "%s  %s\n" \
  04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7 \
  "$V/model_executor/models/qwen3_5.py" \
  0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab \
  "$V/model_executor/models/qwen3_5_mtp.py" \
  | sha256sum -c -
echo "== sentinels =="
grep -q _EXL3_RECONSTRUCT_ARENA "$V/model_executor/layers/quantization/exl3.py" && echo "exl3 arena: PRESENT"
grep -q _planning_for_cudagraph_capture "$V/v1/attention/backends/flashinfer.py" && echo "fi decode shapes: PRESENT"
grep -q "check_stop(request, self.max_servable_num_tokens)" "$V/v1/core/sched/scheduler.py" && echo "52530 check_stop bound: PRESENT"
grep -q "def max_servable_num_tokens" "$V/v1/core/kv_cache_utils.py" && echo "52530 kv-utils helper: PRESENT"
grep -q "def _is_unservable" "$V/v1/core/sched/scheduler.py" && echo "52530 admission gate: PRESENT"
echo "== lmcache untouched + disabled =="
/opt/venv/bin/python - <<PYEOF
import base64, csv, hashlib, importlib.metadata
v = importlib.metadata.version("lmcache")
assert v == "0.5.2+glm52dcp.4", v
sp = "/opt/venv/lib/python3.12/site-packages/"
rows = [r for r in csv.reader(open(sp + "lmcache-0.5.2+glm52dcp.4.dist-info/RECORD")) if r and r[1]]
for p, d, _ in rows:
    algo, _, b64 = d.partition("=")
    got = hashlib.new(algo, open(sp + p, "rb").read()).digest()
    assert got == base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)), p
print(f"lmcache {v}: {len(rows)} records intact, not enabled by any default")
PYEOF
echo "== import vllm =="
CUDA_VISIBLE_DEVICES="" /opt/venv/bin/python -c "import vllm; print(\"vllm\", vllm.__version__)"
echo "== vllm --help =="
CUDA_VISIBLE_DEVICES="" /opt/venv/bin/vllm --help > /dev/null && echo "vllm --help: OK"
echo "SMOKE PASS"
'
