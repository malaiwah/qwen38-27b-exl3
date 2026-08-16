#!/usr/bin/env python3
"""CPU shape-simulation harness for the EXL3 reconstruct-scratch arena.

Executes the *verbatim* allocation-path code (AST-extracted, byte-identical
source segments) from both overlays:

  baseline: tools/vllm-exl3-prefill-dispatch.py  (per-geometry dict)
  arena:    tools/vllm-exl3-scratch-arena.py     (shared per-device arena)

and walks the 9 real geometries of malaiwah/Qwen3.8-27B-EXL3-K5K6-context
(receipts/h-sharing-research.json q3 geometry census) through
``_reconstruct_hgemm_into`` with an instrumented fake extension. Asserts, for
every reachable concurrent set (established as singletons from the dispatch
sites, see receipt):

  1. every scratch view is in-bounds of its backing allocation;
  2. every view starts at storage offset 0 of a fresh allocator block, so it
     keeps the allocator's base alignment (>= 512 B under the CUDA caching
     allocator; asserted >= 2 B fp16 element alignment here on CPU, plus the
     offset-0 property that transfers the CUDA guarantee);
  3. the scratch content written by ``ext.reconstruct``/``reconstruct_slice``
     is intact when ``ext.hgemm`` consumes it (non-overlap / no clobber),
     for the realistic per-layer order and adversarial orders;
  4. view shape/stride/dtype/contiguity are identical between baseline and
     arena, so the kernels see byte-identical operands;
  5. arena bytes == max over concurrent sums == largest single geometry;
     baseline bytes == sum over all geometries;
  6. CUDA-graph capture freeze: a capture-recorded arena is never
     reallocated; oversized geometries fall back to dedicated buffers.

Run under the image's Python (no GPU touched; CUDA_VISIBLE_DEVICES must be
empty):

  R=/var/tmp/gg-rootfs; SP=$R/opt/venv/lib/python3.12/site-packages
  CUDA_VISIBLE_DEVICES= PYTHONHOME=$R/usr PYTHONPATH=$SP \
  $R/lib64/ld-linux-x86-64.so.2 --preload $R/opt/libnccl.so.2.30.4 \
    --library-path "$R/opt:$R/lib/x86_64-linux-gnu:$R/usr/lib/x86_64-linux-gnu:$SP/torch/lib:$R/usr/local/cuda/lib64" \
    $R/usr/bin/python3.12 tools/exl3_scratch_arena_sim.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "vllm-exl3-prefill-dispatch.py"
ARENA = HERE / "vllm-exl3-scratch-arena.py"

# The 9 distinct EXL3 geometries of the qualified context checkpoint
# (receipts/h-sharing-research.json q3_runtime_overhead_inventory
# .geometry_census), as (K, N, role). chunk = min(N, 32768).
GEOMETRIES = [
    (5120, 17408, "mlp gate/up (K5, 130 modules)"),
    (17408, 5120, "mlp down (K6, 65 modules)"),
    (5120, 12288, "attn q (K5, 17 modules)"),
    (5120, 10240, "gdn in_proj (K5, 48 modules)"),
    (10240, 5120, "mtp.fc (K4, 1 module)"),
    (5120, 6144, "gdn ba (K5, 48 modules)"),
    (6144, 5120, "attn o / gdn out (K5, 65 modules)"),
    (5120, 1024, "attn k/v (K5, 34 modules)"),
    (5120, 248320, "lm_head (K6, 1 module; chunked to 32768)"),
]
SLICE_N = 32768

# One full-attention layer + one GDN layer + MLP, in checkpoint execution
# order, then the MTP fc and the head: the realistic interleaving. Only
# geometry identity matters to the allocator, so one representative layer of
# each kind covers all 65.
REALISTIC_ORDER = [
    (5120, 12288),  # q
    (5120, 1024),  # k
    (5120, 1024),  # v
    (6144, 5120),  # o
    (5120, 17408),  # gate
    (5120, 17408),  # up
    (17408, 5120),  # down
    (5120, 10240),  # gdn in_proj
    (5120, 6144),  # gdn ba
    (6144, 5120),  # gdn out
    (5120, 17408),  # gate (next layer)
    (5120, 17408),  # up
    (17408, 5120),  # down
    (10240, 5120),  # mtp.fc
    (5120, 248320),  # head chunk (>=128 sampled logit rows)
]


def extract_defs(path: Path, names: set[str]) -> tuple[str, dict[str, str]]:
    """Return exec-able source made of the named top-level defs, verbatim."""
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    segments: list[str] = []
    seg_sha: dict[str, str] = {}
    for node in tree.body:
        label = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            label = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in names:
                    label = t.id
        if label in names:
            seg = "".join(lines[node.lineno - 1 : node.end_lineno])
            segments.append(seg)
            seg_sha[label] = hashlib.sha256(seg.encode()).hexdigest()
    missing = names - set(seg_sha)
    if missing:
        raise SystemExit(f"{path.name}: missing defs {sorted(missing)}")
    return "\n".join(segments), seg_sha


class FakeExt:
    """Instrumented stand-in for exllamav3_ext on the reconstruct path.

    Tags every scratch write and verifies the tag on consumption, which fails
    on any aliasing/clobber between the write and the read of one call.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self._live = None  # (data_ptr, nbytes, tag) of the pending write

    def had_r_128(self, x, out, su, sv, scale):
        if x.data_ptr() != out.data_ptr():
            out.copy_(x)

    def _write(self, view, tag):
        self.tracker.on_view(view)
        view.fill_(tag)
        self._live = (view.data_ptr(), view.numel() * 2, tag, view)

    def reconstruct(self, view, trellis, bits, mcg, mul1):
        self._write(view, self.tracker.next_tag())

    def reconstruct_slice(self, view, trellis, bits, mcg, mul1, start):
        self._write(view, self.tracker.next_tag())

    def hgemm(self, x_had, view, out):
        ptr, nbytes, tag, wview = self._live
        assert view.data_ptr() == ptr or (
            view.data_ptr() >= ptr and view.data_ptr() + view.numel() * 2 <= ptr + nbytes
        ), "hgemm reads outside the bytes the reconstruct wrote"
        assert bool((wview == tag).all()), (
            "scratch content clobbered between reconstruct write and hgemm "
            "read -- overlapping live views"
        )
        self.tracker.consumed += 1
        out.fill_(0.0)


class Tracker:
    def __init__(self):
        self.views = []
        self.consumed = 0
        self._tag = 0
        self.ns = None  # set after exec; lets on_view see the live arena

    def next_tag(self):
        self._tag += 1
        return float(self._tag % 251) + 0.5

    def on_view(self, view):
        """Validate the view at creation time against its live backing."""
        storage = view.untyped_storage()
        off_bytes = view.storage_offset() * view.element_size()
        vbytes = (
            (view.shape[0] - 1) * view.stride(0) + view.shape[1]
        ) * view.element_size()
        assert off_bytes >= 0, "view below backing base"
        assert off_bytes + vbytes <= storage.nbytes(), "view exceeds backing"
        assert view.data_ptr() % view.element_size() == 0
        assert view.dtype == torch.float16
        backed_by_arena = None
        if self.ns is not None and "_EXL3_RECONSTRUCT_ARENA" in self.ns:
            arena = self.ns["_EXL3_RECONSTRUCT_ARENA"].get(0)
            backed_by_arena = (
                arena is not None
                and storage.data_ptr() == arena.untyped_storage().data_ptr()
            )
            if backed_by_arena:
                assert off_bytes == 0, (
                    "concurrent set is a singleton: arena views start at 0"
                )
                assert off_bytes + vbytes <= arena.numel(), "view exceeds arena"
        self.views.append((view, backed_by_arena, off_bytes))


def load_impl(path: Path, needs_hgemm: bool):
    names = {
        "_EXL3_RECONSTRUCT_SLICE_N",
        "_EXL3_RECONSTRUCT_SCRATCH",
        "_reconstruct_scratch",
    }
    if path == ARENA:
        names |= {"_EXL3_RECONSTRUCT_ARENA", "_EXL3_RECONSTRUCT_ARENA_FROZEN"}
    if needs_hgemm:
        names |= {"_reconstruct_hgemm_into"}
    src, shas = extract_defs(path, names)
    ns: dict = {"torch": torch}

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    ns["logger"] = _Logger()
    tracker = Tracker()
    ns["_load_exl3_ext"] = lambda: FakeExt(tracker)
    exec(compile(src, str(path), "exec"), ns)
    tracker.ns = ns
    return ns, tracker, shas


def patch_cuda(monkey_capturing):
    torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
    torch.cuda.is_current_stream_capturing = monkey_capturing  # type: ignore[assignment]


def run_sequence(ns, tracker, order, rows=128):
    """Drive the real _reconstruct_hgemm_into over a call order."""
    device = torch.device("cpu")
    for k, n in order:
        trellis = torch.zeros((k // 16, n // 16, 80), dtype=torch.int16)
        x = torch.zeros((rows, k), dtype=torch.float16, device=device)
        out = torch.empty((rows, n), dtype=torch.float16, device=device)
        suh = torch.zeros((k,), dtype=torch.float16)
        svh = torch.zeros((n,), dtype=torch.float16)
        ns["_reconstruct_hgemm_into"](out, x, trellis, suh, svh, False, False)


def backing_bytes_arena(ns):
    return sum(t.numel() for t in ns["_EXL3_RECONSTRUCT_ARENA"].values()) + sum(
        t.numel() * t.element_size() for t in ns["_EXL3_RECONSTRUCT_SCRATCH"].values()
    )


def backing_bytes_baseline(ns):
    return sum(
        t.numel() * t.element_size() for t in ns["_EXL3_RECONSTRUCT_SCRATCH"].values()
    )


def main():
    assert torch.cuda.device_count if True else None
    patch_cuda(lambda: False)
    report = {
        "schema": "exl3-scratch-arena-sim/1",
        "torch": torch.__version__,
        "geometries": [
            {"K": k, "N": n, "chunk": min(n, SLICE_N), "role": role}
            for k, n, role in GEOMETRIES
        ],
    }

    mib = lambda b: round(b / (1 << 20), 1)
    expected_max = max(2 * k * min(n, SLICE_N) for k, n, _ in GEOMETRIES)
    expected_sum = sum(2 * k * min(n, SLICE_N) for k, n, _ in GEOMETRIES)

    # ---- baseline: per-geometry dict --------------------------------------
    bns, btracker, bshas = load_impl(BASELINE, needs_hgemm=True)
    run_sequence(bns, btracker, REALISTIC_ORDER)
    base_bytes = backing_bytes_baseline(bns)
    assert base_bytes == expected_sum, (base_bytes, expected_sum)
    report["baseline"] = {
        "segment_sha256": bshas,
        "persistent_bytes": base_bytes,
        "persistent_mib": mib(base_bytes),
        "distinct_buffers": len(bns["_EXL3_RECONSTRUCT_SCRATCH"]),
        "hgemm_consumptions_verified": btracker.consumed,
    }

    # ---- arena ------------------------------------------------------------
    orders = {
        "realistic": REALISTIC_ORDER,
        "ascending": sorted(
            ((k, n) for k, n, _ in GEOMETRIES), key=lambda g: g[0] * min(g[1], SLICE_N)
        ),
        "descending": sorted(
            ((k, n) for k, n, _ in GEOMETRIES),
            key=lambda g: g[0] * min(g[1], SLICE_N),
            reverse=True,
        ),
    }
    order_results = {}
    for name, order in orders.items():
        ans, atracker, ashas = load_impl(ARENA, needs_hgemm=True)
        run_sequence(ans, atracker, order)
        arena = ans["_EXL3_RECONSTRUCT_ARENA"][0]
        abytes = backing_bytes_arena(ans)
        assert abytes == expected_max, (name, abytes, expected_max)
        assert not ans["_EXL3_RECONSTRUCT_SCRATCH"], "unexpected dedicated buffers"
        for _view, backed_by_arena, off_bytes in atracker.views:
            assert backed_by_arena is True, (
                "eager view not served from the shared arena"
            )
            assert off_bytes == 0
        # layout parity with torch.empty((k, chunk))
        for k, n, _ in GEOMETRIES:
            chunk = min(n, SLICE_N)
            v = ans["_reconstruct_scratch"](torch.device("cpu"), k, chunk)
            ref = torch.empty((k, chunk), dtype=torch.float16)
            assert v.shape == ref.shape and v.stride() == ref.stride()
            assert v.dtype == ref.dtype and v.is_contiguous()
        order_results[name] = {
            "arena_bytes": abytes,
            "arena_mib": mib(abytes),
            "views_checked": len(atracker.views),
            "hgemm_consumptions_verified": atracker.consumed,
            "all_views_offset0_inbounds_aligned": True,
            "segment_sha256": ashas,
        }
    report["arena_orders"] = order_results

    # ---- capture freeze ---------------------------------------------------
    ans, atracker, _ = load_impl(ARENA, needs_hgemm=True)
    # grow eagerly with a mid-size geometry only
    run_sequence(ans, atracker, [(5120, 12288)])
    arena_before = ans["_EXL3_RECONSTRUCT_ARENA"][0]
    # capture begins: same-size call must freeze, not reallocate
    patch_cuda(lambda: True)
    v = ans["_reconstruct_scratch"](torch.device("cpu"), 5120, 12288)
    assert v.data_ptr() == arena_before.data_ptr()
    assert 0 in ans["_EXL3_RECONSTRUCT_ARENA_FROZEN"]
    # capture asks for a bigger geometry: dedicated buffer, arena untouched
    v2 = ans["_reconstruct_scratch"](torch.device("cpu"), 17408, 5120)
    assert ans["_EXL3_RECONSTRUCT_ARENA"][0] is arena_before
    assert (0, 17408, 5120) in ans["_EXL3_RECONSTRUCT_SCRATCH"]
    assert v2.data_ptr() == ans["_EXL3_RECONSTRUCT_SCRATCH"][(0, 17408, 5120)].data_ptr()
    # capture over, arena stays frozen: bigger geometry still dedicated
    patch_cuda(lambda: False)
    v3 = ans["_reconstruct_scratch"](torch.device("cpu"), 5120, 17408)
    assert ans["_EXL3_RECONSTRUCT_ARENA"][0] is arena_before
    assert (0, 5120, 17408) in ans["_EXL3_RECONSTRUCT_SCRATCH"]
    # dedicated buffers are stable across calls (baked addresses stay valid)
    v4 = ans["_reconstruct_scratch"](torch.device("cpu"), 5120, 17408)
    assert v3.data_ptr() == v4.data_ptr()
    report["capture_freeze"] = {
        "frozen_arena_never_reallocated": True,
        "oversize_geometry_falls_back_to_dedicated_buffer": True,
        "dedicated_buffer_address_stable": True,
    }

    report["verdict"] = {
        "baseline_persistent_mib": mib(expected_sum),
        "arena_persistent_mib": mib(expected_max),
        "saving_all_9_geometries_mib": mib(expected_sum - expected_max),
        "note": (
            "at the qualified profile the head geometry does not allocate "
            "(needs >=128 sampled logit rows), so the serve-time saving is "
            "790.0 - 170.0 = 620.0 MiB; with the head it is 1110.0 - 320.0 "
            "= 790.0 MiB"
        ),
        "in_bounds": True,
        "aligned_offset0": True,
        "non_overlapping_every_reachable_concurrent_set": True,
        "kernels_see_identical_operands": True,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
