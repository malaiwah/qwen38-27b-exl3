# vLLM memory-accounting ledger (running)

Unaccounted / mis-accounted GPU memory in the serving stack, so we can chase
each item down later. Every entry is something that made us OOM (or waste
memory) while the nominal `--gpu-memory-utilization` budget said we were fine.

Reference accounting, RTX 5090 (31.4 GiB usable), K5K6 hydrated, util 0.93:

| component | mnbt 8192 | mnbt 3072 |
|---|---|---|
| model weights | 16.19 GiB | 16.19 GiB |
| **peak activation (profiled)** | **2.84 GiB** | **2.32 GiB** |
| non-torch | 0.30 GiB | 0.30 GiB |
| CUDA graph pool | 0.89 GiB | 0.52 GiB |
| KV cache (available) | 8.89 GiB | **9.82 GiB** |
| context served | 238,400 | **262,144** |

## L1. Profiled "peak activation" under-predicts real GDN-hybrid prefill peak
**Severity: was engine-fatal.** vLLM measures peak activation with a dummy
profiling run and reported 2.84 GiB at mnbt 8192. In practice a 6000-token
prompt OOMed inside the GDN linear-attention prefill
(`vllm/third_party/flash_linear_attention/ops/chunk_o.py:168`,
`o = torch.empty_like(v)`) with only 41-101 MiB free. 4001 tokens was fine.
So the real peak exceeds the profiled peak by more than the ~0.4 GiB of slack
the util setting leaves. Mitigated (not fixed) by lowering the chunk budget.
**To chase:** why the profile run misses the worst case — does the dummy run
exercise the GDN path with a representative chunk/state layout at all?

## L2. Allocator high-water poisoning across requests
**Severity: was engine-fatal.** A large prefill permanently raises the
caching allocator's reserved high-water mark, so a *later* large prefill dies
even though an identical one just succeeded: at mnbt 4096 the ascending
ladder 2k->6k->12k->24k passed, then a repeat 12k killed the engine.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set and does not
prevent it. **To chase:** a periodic/threshold `empty_cache()` hook, or an
allocator policy that returns large transient segments.

## L3. Engine dies instead of degrading on forward-pass OOM
**Severity: availability.** An OOM inside the model forward raises
`EngineDeadError` and the whole engine becomes unusable (subsequent requests
all 500, `/health` 503). There is no request rejection, preemption, or
retry-with-smaller-chunk path. A single oversized prompt from one client takes
the server down. **To chase:** upstream — catch OOM in the execute path and
preempt/reject the offending request instead of killing the core.

## L4. GPU memory retained after engine death
`nvidia-smi` reported 32067 / 32607 MiB still in use with the EngineCore dead.
Restarting the *service* is not enough; the container must be recreated
(`podman rm -f`) to release it. Any restart automation must account for this.

## L5. Hybrid KV padding wastes up to 6.25% of the KV cache
`kv_cache_utils.py:1298`: "Add 3 padding layers, may waste at most 6.25% KV
cache memory" — the 16 full-attention + 48 GDN layer grouping pads to a common
group size. At 9.82 GiB KV that is up to ~0.61 GiB ≈ 16,000 tokens of context
thrown away. **To chase:** whether the padding is avoidable for a 16/48 split.

## L6. CUDA-graph pool size scales with max-num-batched-tokens
0.89 GiB at mnbt 8192 vs 0.52 GiB at mnbt <=4096 — a hidden 0.37 GiB memory
multiplier attached to the chunk budget, on top of activations. Not documented
anywhere we could find; it means mnbt has two independent memory effects.
Also logged: "util 0.93 is equivalent to util 0.8991 without CUDA graph
profiling" — the effective budget is ~1% lower than the flag suggests.

## L7. CORRECTION to our own earlier estimate
`receipts/peer-review-2026-08-18.md` inferred a ~1.29 GiB non-PyTorch gap from
an OOM message. vLLM's own accounting says **non-torch = 0.30 GiB**. The
earlier figure was inflated by comparing `memory_allocated` (not
`memory_reserved`) against the total, i.e. it counted PyTorch's own reserved-
but-unallocated cache as non-torch. The OOMSourceScout ranking should be
re-read with that correction: the CuTe/Triton/exllamav3 cubin text and CUDA
context all live inside 0.30 GiB, so cubin reduction is worth far less than
estimated.

## L8. Measurement hazard (not memory, but it corrupted memory experiments)
`~/.config/systemd/user/qwen38-27b.service` has `Restart=always`,
`RestartSec=10`. When a manual `podman run` instance died, systemd restarted
the service with **default** env, silently replacing the container. Two
"different" configs then measured identically (mnbt 2048 vs 4096) because both
were really the 8192 default. All config experiments must
`systemctl --user stop` first and then **assert** the effective args from the
engine's own `non-default args:` dump (`/tmp/boot_cfg.sh` does this).

## L9. b12x prepared trellis weights cost ~2 GiB (docstring says "zero-copy views")
Measured 2026-08-19 on an all-trellis config routed through b12x
(`VLLM_EXL3_B12X_ANY_BITS=1`): vLLM reports **18.83 GiB for weight** where the
raw trellis payload is 16.82 GiB (`tools/shape-inventory.py`). Peak activation
also rose to 3.25 GiB (vs 2.32 for the balanced FP4/trellis mix), leaving only
**6.28 GiB KV = max ctx 158,304**, and the instance still died on the first
2051-token prefill (same underprofiled-peak pattern as L1). Consequence:
all-trellis cannot reach the 238,400-token context target on this card, so the
KLD-optimal weight format is context-limited for structural reasons, not
tuning. **To chase:** what exactly `prepare_trellis256_dense_weight` retains
beyond views, and whether the b12x dense scratch can be shared across shapes
instead of one buffer per (m, k, n, bits).

## L10. OUR bug: W4A16 scratch sized on a K6-only assumption (fixed)
`_b12x_trellis_c_tmp_elements` returned **1 element** for `rows <= 128` with the
comment "the cooperative K6 small-M kernel does not consume W4A16 scratch".
That is true only for K6. Admitting K5 (this checkpoint's `mlp.gate_proj`/
`up_proj`) made b12x raise *"W4A16 GEMM scratch is not initialized for CUDA
graph capture; provide a preallocated fc*_c_tmp workspace with sufficient
capacity"* during capture. Two-part fix: gate the shortcut on `bits == 6`, and
size the real case from b12x's own formula — noting the dense runner rounds m
UP to a multiple of the routed block size (`route_slots = ceil(m/block)*block`)
before calling `packed_gemm_scratch_elements`, so sizing on raw `rows`
under-allocates by up to 64x (m=1, N=17408 needs 1,114,112 floats, not 17,408).
Cover every entry of `_W4A16_ALLOWED_ROUTED_SIZES = (8,16,32,48,64)`.

## L11. Host b12x checkout != container b12x
`/home/mbelleau/b12x/b12x/moe/_shared/kernels/w4a16/kernel.py` and the
container's copy have different md5sums and different line numbers. Every
b12x claim must be read from the container overlay, not the host checkout.
