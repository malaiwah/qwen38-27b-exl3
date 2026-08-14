#!/bin/bash
# Run a command inside the extracted Gilded Gnosis image rootfs with no
# container runtime.
#
# This box is itself a container with a seccomp profile that denies
# clone(CLONE_NEWUSER|CLONE_NEWNS) -- `unshare -U`/`-m` return EPERM -- so a
# user-namespace chroot is impossible.  proot emulates chroot + bind mounts via
# ptrace instead, which needs no privileges.  Only syscalls are traced; GPU
# compute is untouched (measured: 20x 4096^2 bf16 matmuls in 0.16 s).
#
#   ggrun.sh python -c 'import torch; print(torch.cuda.get_device_name(0))'
#   GG_ENV="VLLM_EXL3_ONLINE_TRELLIS_BITS=6" ggrun.sh vllm serve /models/...
#
# Guest paths: /models -> $GG_MODELS, /work -> $GG_WORK, /cache -> $GG_CACHE.
set -euo pipefail
R="${GG_ROOTFS:-/var/tmp/gg-rootfs}"
MODELS="${GG_MODELS:-/var/tmp/models}"
CACHE="${GG_CACHE:-/var/tmp/gg-cache}"
WORK="${GG_WORK:-/var/tmp/work}"
PROOT="${PROOT:-/var/tmp/proot}"
mkdir -p "$CACHE" "$WORK"

binds=(-b "$MODELS:/models" -b "$WORK:/work" -b "$CACHE:/cache")
# The image ships the CUDA toolkit; the driver half normally arrives via
# nvidia-container-toolkit.  Bind the host driver libraries onto the guest SONAMEs.
for pair in libcuda.so.1 libnvidia-ml.so.1 libnvidia-nvvm.so.4 libnvidia-ptxjitcompiler.so.1; do
  base="${pair%%.so.*}"
  hostlib=$(ls -1 /usr/lib/x86_64-linux-gnu/${base}.so.[0-9]* 2>/dev/null | grep -E '\.so\.[0-9]+\.[0-9]+' | head -1 || true)
  [ -n "$hostlib" ] && binds+=(-b "$hostlib:/usr/lib/x86_64-linux-gnu/$pair")
done

exec "$PROOT" -R "$R" "${binds[@]}" \
  /usr/bin/env -i \
  PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
  HOME=/root TERM=dumb LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/cache/hf \
  TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-/opt/exllamav3/exllamav3_ext.cpython-312-x86_64-linux-gnu.so}" \
  VLLM_EXL3_ENCODER_SOURCE="${VLLM_EXL3_ENCODER_SOURCE:-/opt/exllamav3-python/exllamav3}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  ${GG_ENV:-} \
  "$@"
