#!/bin/bash
# The context edition: the first build in this family aimed at native 262,144 context on a
# 32 GB card while still measuring below official FP8.
#
# Three changes from the hydrated build, each justified by measurement rather than taste:
#   * attention serialized at K5 instead of K6. The overlay measurement says K5 costs
#     +0.003978 mean KLD and saves 0.50-0.84 GiB; calibrated serialization should cost less
#     than the runtime's calibration-free encoding did (offline K6 beat online K6 by 9.2 %).
#   * vision tower at K6 instead of BF16. It is 0.921 GB of untested BF16; -vb 6 takes it to
#     ~0.345 GB. Nothing in the recipe justified BF16 there beyond nobody having measured it.
#   * MLP unchanged (gate/up K5, down K6), head K6/mcg, MTP quantized - the parts that carry
#     the fidelity result.
#
# Target: ~19.0-19.3 GiB resident, which is what makes 262,144 tokens of fp8 KV fit inside a
# 5090's 30.44 GiB budget at utilisation 0.97.
set -uo pipefail
die() { echo "FAILED: $*" >&2; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_EXTENSIONS_DIR=/work/torch-ext TORCH_CUDA_ARCH_LIST=12.0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export EXL3_BITS_FIXED='{"^.*self_attn\\..*$": 5, "^.*linear_attn\\..*$": 5}'
export EXL3_BITS_OVERRIDE='{"^.*mlp\\.down_proj$": 6, "^.*mlp\\.(gate|up)_proj$": 5}'

cd /work/exllamav3
# tee the per-module proxy errors: the last conversion's stdout was lost to log rotation and
# with it the epsilon ladder that error-driven allocation needs.
python convert.py -i /models/Qwen3.8-27B -o /work/qwen38-ctx -w /work/wd-ctx \
  -b 4 -hb 6 -mb 4 -vb 6 -cb mcg 2>&1 | tee /work/kld3/convert-ctx.log || die convert

echo "== finalize: index, quant config, verified topology"
python util/add_safetensors_index.py -m /work/qwen38-ctx --force || die index
python util/add_quant_config.py -m /work/qwen38-ctx || die quantconfig
python /work/kld3/finalize_checkpoint.py -m /work/qwen38-ctx --upstream /work/upstream-index \
  --recipe "MLP gate/up K5, down K6, attention K5 serialized (calibrated), vision tower K6, lm_head K6/mcg, quantized MTP, BF16 embeddings" \
  --command "convert.py -i Qwen3.8-27B -b 4 -hb 6 -mb 4 -vb 6 -cb mcg with EXL3_BITS_FIXED attention=5 and EXL3_BITS_OVERRIDE mlp gate/up=5 down=6" \
  || die finalize
du -sb /work/qwen38-ctx | awk '{printf "context-edition disk bytes: %.3f GB\n", $1/1e9}'
grep -cE "^ -- Quantized" /work/kld3/convert-ctx.log | awk '{print "per-module proxy errors captured: " $1}'
echo CTXBUILDDONE
