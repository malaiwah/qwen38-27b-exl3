# CPU reproduction harness: drive the PINNED WHEEL's (lmcache 0.5.2+glm52dcp.4)
# fs_native L2 adapter + PrefetchController + the MP lookup arithmetic exactly
# as the MP server wires them, with a stub L1 (all misses) — the L3restart
# regime, and (with l1_hits>0) the bounded L1cold/L2warm regime.
#
# Question under test: for a query whose TRUE stored prefix is k chunks,
# what hit count does the server-side pipeline report?
# Honest answer: k.  Measured on GPU (receipts/lmcache-reuse-test.json):
# k+1 for every request with a non-empty store.
import sys
import tempfile
import time

sys.path.append("/var/tmp/gg-rootfs/opt/venv/lib/python3.12/site-packages")

from lmcache.v1.distributed.api import (  # noqa: E402
    DEFAULT_ATTN_WINDOW_DESC,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
    TrimPolicy,
)
from lmcache.v1.distributed.error import L1Error  # noqa: E402
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (  # noqa: E402
    FSNativeL2AdapterConfig,
    _create_fs_native_l2_adapter,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (  # noqa: E402
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (  # noqa: E402
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (  # noqa: E402
    AdapterDescriptor,
)

import torch  # noqa: E402

OBJ_BYTES = 4096


class StubMemoryObj:
    """Minimal MemoryObj lookalike: byte_array + get_size."""

    def __init__(self, fill: int):
        self._buf = bytearray([fill % 256]) * 1
        self._buf = bytearray(OBJ_BYTES)
        for i in range(OBJ_BYTES):
            self._buf[i] = fill % 256
        self.byte_array = memoryview(self._buf)

    def get_size(self) -> int:
        return OBJ_BYTES


class StubL1Manager:
    """L1 with nothing resident; every write reservation succeeds."""

    def __init__(self):
        self.finished_writes = []
        self.deleted = []
        self.read_locked = []

    def reserve_read(self, keys, extra_count=0):
        return {k: (L1Error.KEY_NOT_EXIST, None) for k in keys}

    def finish_read(self, keys, extra_count=0):
        pass

    def reserve_write(self, keys, is_temporary, layout_desc, mode="new"):
        return {k: (L1Error.SUCCESS, StubMemoryObj(0)) for k in keys}

    def finish_write(self, keys):
        self.finished_writes.extend(keys)

    def finish_write_and_reserve_read(self, keys, extra_count=0):
        self.read_locked.extend(keys)

    def delete(self, keys, force=False):
        self.deleted.extend(keys)
        return {k: L1Error.SUCCESS for k in keys}

    def touch_keys(self, keys):
        pass

    def get_memory_usage(self):
        return (0, 1 << 30)


def chunk_key(tag: bytes) -> ObjectKey:
    """32-byte hash from a short tag (stands in for a chained blake3)."""
    return ObjectKey(
        chunk_hash=tag.ljust(32, b"\0"),
        model_name="test/model",
        kv_rank=0,
        object_group_id=0,
        cache_salt="",
    )


def wait_store(adapter, task_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done = adapter.pop_completed_store_tasks()
        if task_id in done:
            return done[task_id]
        time.sleep(0.01)
    raise TimeoutError("store did not complete")


def run_query(controller, keys, layout, label):
    rid = controller.submit_prefetch_request(
        keys, layout, extra_count=0, policy=TrimPolicy.PREFIX,
        mode=PrefetchMode.LOOKUP,
    )
    deadline = time.monotonic() + 10.0
    lookup_hit = None
    result = None
    while time.monotonic() < deadline and result is None:
        if lookup_hit is None:
            lookup_hit = controller.query_lookup_result(rid)
        result = controller.query_prefetch_result(rid)
        time.sleep(0.01)
    if result is None:
        raise TimeoutError(f"{label}: prefetch did not complete")
    # MP server arithmetic (wheel lookup.py):
    #   query_prefetch_lookup_hits: (l1_hits + l2_lookup) // stride, stride=1
    #   query_prefetch_status:      found.count_leading_ones() // stride
    # here l1_hits = 0 (stub L1 empty), so both come straight from the
    # controller's numbers.
    status_hit = result.count_leading_ones()
    print(
        f"{label}: lookup-hits(rpc QUERY_PREFETCH_LOOKUP_HITS)={lookup_hit} "
        f"status-hits(rpc QUERY_PREFETCH_STATUS)={status_hit} "
        f"retained-indices={result.get_indices_list()}"
    )
    return lookup_hit, status_hit


def main():
    tmp = tempfile.mkdtemp(prefix="lmc-l2-")
    cfg = FSNativeL2AdapterConfig(base_path=tmp, num_workers=2)
    adapter = _create_fs_native_l2_adapter(cfg)

    layout = MemoryLayoutDesc(
        shapes=[torch.Size((OBJ_BYTES,))], dtypes=[torch.uint8]
    )

    # ---- store: request A's chunks c0..c3 under distinct chained hashes
    a_keys = [chunk_key(b"reqA-chunk-%d" % i) for i in range(4)]
    objs = [StubMemoryObj(i + 1) for i in range(4)]
    tid = adapter.submit_store_task(a_keys, objs)
    res = wait_store(adapter, tid)
    print("store result:", res)

    stub_l1 = StubL1Manager()
    controller = PrefetchController(
        l1_manager=stub_l1,
        l2_adapters=[adapter],
        adapter_descriptors=[AdapterDescriptor(index=0, config=cfg)],
        policy=DefaultPrefetchPolicy(),
    )
    controller.start()

    # ---- query 1: request B shares a true prefix of k=2 chunks with A,
    # then diverges (2 chunks never stored). Honest hit: 2.
    b_keys = a_keys[:2] + [chunk_key(b"reqB-chunk-%d" % i) for i in (2, 3)]
    run_query(controller, b_keys, layout, "query B (true prefix=2, total=4)")

    # ---- query 2: request C shares NOTHING (true prefix 0). Honest hit: 0.
    c_keys = [chunk_key(b"reqC-chunk-%d" % i) for i in range(2)]
    run_query(controller, c_keys, layout, "query C (true prefix=0, total=2)")

    # ---- query 3: full self-match (all 4 stored). Honest hit: 4.
    run_query(controller, a_keys, layout, "query A (true prefix=4, total=4)")

    controller.stop()
    adapter.close()


if __name__ == "__main__":
    main()
