#!/usr/bin/env python3
"""TCP -> unix-socket forwarder for the Terminal-Bench 2.1 endpoint tunnel.

Why this exists
---------------
The bench runs for hours over a long-haul link, so the reverse tunnel must be
re-establishable at will. A plain `ssh -R <port>:...` makes *sshd* the owner of
the remote TCP listener, and that turned out to be a trap: after an unclean
client death the listener survives in a stale state (it accepts the connection
and immediately resets it, curl rc=56) and is held by sshd's **root** privsep
process, so an unprivileged agent cannot reclaim the port. `fuser -k` and `lsof`
cannot even see the owner. The port stays poisoned until sshd happens to reap
it, and the port cannot simply be changed either: harbor bakes `api_base` into
the job config, and resume requires an identical config, so the port must stay
fixed across all three passes.

The fix is to take port ownership away from sshd:

    ssh -R /run/user/<uid>/tb21-vllm.sock:127.0.0.1:8000 \
        -o StreamLocalBindUnlink=yes  aiboss

`StreamLocalBindUnlink=yes` makes ssh remove a stale socket file before binding,
so the ssh side can never be poisoned. This proxy then owns the TCP port that
harbor actually talks to, and it is an ordinary user process: killable,
restartable, and bound with SO_REUSEADDR.

A connection is refused fast (rather than hanging) whenever the unix socket is
absent or not accepting, which is what makes `wait-ready` and `probe` meaningful
signals instead of timeouts.

Usage:
    tb_tunnel_proxy.py --port 18000 --socket /run/user/1000/tb21-vllm.sock
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pathlib
import signal
import sys
import time

BUF = 1 << 16


class Stats:
    def __init__(self) -> None:
        self.accepted = 0
        self.refused = 0
        self.bytes_up = 0
        self.bytes_down = 0
        self.started = time.time()

    def line(self) -> str:
        return (
            f"[proxy] up={time.time() - self.started:.0f}s "
            f"accepted={self.accepted} refused={self.refused} "
            f"up_bytes={self.bytes_up} down_bytes={self.bytes_down}"
        )


async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
               stats: Stats, up: bool) -> None:
    try:
        while True:
            data = await reader.read(BUF)
            if not data:
                break
            if up:
                stats.bytes_up += len(data)
            else:
                stats.bytes_down += len(data)
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(OSError):
            writer.write_eof()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pidfile", default="")
    ap.add_argument("--stats-every", type=float, default=300.0)
    a = ap.parse_args()

    sock_path = a.socket
    stats = Stats()

    if a.pidfile:
        pathlib.Path(a.pidfile).write_text(f"{os.getpid()}\n")

    async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        try:
            ur, uw = await asyncio.wait_for(
                asyncio.open_unix_connection(sock_path), timeout=10
            )
        except (OSError, asyncio.TimeoutError) as e:
            # Refuse fast so callers see an error instead of hanging: this is
            # the signal `wait-ready` and `probe` rely on.
            stats.refused += 1
            if stats.refused <= 5 or stats.refused % 50 == 0:
                print(f"[proxy] refuse #{stats.refused}: {type(e).__name__} "
                      f"on {sock_path}", flush=True)
            cw.close()
            with contextlib.suppress(Exception):
                await cw.wait_closed()
            return
        stats.accepted += 1
        try:
            await asyncio.gather(
                pump(cr, uw, stats, up=True),
                pump(ur, cw, stats, up=False),
            )
        finally:
            for w in (uw, cw):
                w.close()
                with contextlib.suppress(Exception):
                    await w.wait_closed()

    server = await asyncio.start_server(
        handle, a.host, a.port, reuse_address=True, backlog=256
    )
    print(f"[proxy] listening {a.host}:{a.port} -> unix:{sock_path} "
          f"pid={os.getpid()}", flush=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, stop.set)

    async def ticker() -> None:
        while not stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=a.stats_every)
                return
            print(stats.line(), flush=True)

    async with server:
        await asyncio.gather(stop.wait(), ticker())
    print(f"[proxy] shutting down; {stats.line()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
