# Running the official GG image with no container runtime

This rental host allows no nested containers. Measured constraints:

- no `docker`, `podman`, `skopeo`, `umoci`, `bwrap`, `crun`, `fakeroot`
- `CapEff: 0000000000000000`
- `unshare -U` and `unshare -m` both return `EPERM`; `Seccomp: 2` with one
  filter, i.e. a seccomp profile denies `clone(CLONE_NEWUSER|CLONE_NEWNS)`
  even though `/proc/sys/user/max_user_namespaces` is 6190298 and
  `unprivileged_userns_clone` is 1

So a user-namespace chroot is impossible. Two pieces solve it:

1. [`tools/pull_rootfs.py`](../tools/pull_rootfs.py) — unprivileged OCI pull.
   Registry v2 token auth, response-manifest/config/layer digest verification (including
   cached layers), then layers flattened in order with `.wh.` whiteout handling, device nodes
   skipped and ownership rewritten to the invoking uid. A canonical full-rootfs manifest
   binds every file, directory and symlink; serving harnesses verify it before installing the
   three separately hashed public patch modules.
2. [`tools/ggrun.sh`](../tools/ggrun.sh) — `proot` (static, ptrace-based, no
   privileges) emulates chroot plus bind mounts. Host NVIDIA driver libraries are
   bound onto the guest SONAMEs, because the image ships the CUDA toolkit but the
   driver half normally arrives via nvidia-container-toolkit.

Image pulled: `voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`
(the r34 registry digest), 36 layers, 12.53 GB compressed, 22 GB flattened.

Only syscalls are traced, so GPU throughput is unaffected — measured inside the
proot rootfs: `torch 2.12.0+cu132`, device `NVIDIA RTX PRO 6000 Blackwell Server
Edition` capability `(12, 0)`, 20 chained 4096x4096 BF16 matmuls in 0.16 s,
100.5/102.0 GB free.

## Baseline 1 — `unsloth/Qwen3.8-27B-NVFP4`: PASS

```bash
GG_MODELS=/var/tmp/models tools/ggrun.sh vllm serve /models/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-nvfp4 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --max-num-seqs 4 --port 8010
```

Engine log evidence:

- version `0.11.2.dev280+gilded.gnosis.v20.vllm4d006a4.b12xcd3ce19.fi1ac6942.cu132.20260810.r34`
- `quantization=compressed-tensors` auto-detected; `Using CutlassNvFp4LinearKernel for NVFP4 GEMM`
- `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` — the
  hybrid linear-attention path is live
- `FLASHINFER` attention backend, `FULL_AND_PIECEWISE` CUDA graphs (no
  `--enforce-eager` needed for this format)
- `GPU KV cache size: 1,071,331 tokens`
- chat completion returned in 1.9 s for 60 tokens

## Baseline 2 — `Qwen/Qwen3.8-27B` BF16: PASS

```bash
GG_MODELS=/var/tmp/models tools/ggrun.sh vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen38-bf16 --max-model-len 8192 \
  --gpu-memory-utilization 0.92 --max-num-seqs 4 --port 8011
```

Serves and answers chat completions (55.56 GB of weights on one 96 GB GPU).
This is the KLD teacher.

Both baselines confirm the image's engine handles this architecture; the open
question was never the model, it is the EXL3 dense path documented in
[03-gg-runtime-contract.md](03-gg-runtime-contract.md).
