#!/usr/bin/env python3
"""Preflight test double for the Terminal-Bench 2.1 endpoint.

WHAT THIS IS FOR
----------------
The oracle-agent smoke test proves the *infrastructure* (podman -> docker CLI ->
compose -> task container -> verifier -> result.json) but it needs no model, so
it leaves the **Terminus-2 <-> endpoint integration** completely unproven: the
LiteLLM `hosted_vllm/` provider naming, the explicit `model_info`, the api_base
travelling through the reverse tunnel, the agent's JSON response parsing, its
multi-turn loop, its token accounting, and its transcript capture.

Discovering a bug in any of those while holding a queued, shared GPU for a
multi-hour bench would be expensive. So this serves a minimal but genuinely
OpenAI-compatible endpoint that returns *valid Terminus-2 responses*, letting the
whole agent path be exercised end to end with no GPU at all.

It is a PREFLIGHT INSTRUMENT, never part of a scored pass. The three protocol
passes talk to the real vLLM server; nothing here is used once the card is in
hand. A pass whose rows were produced against this would be obvious anyway: the
model id it reports is `mock-preflight-endpoint`.

The response schema is taken from harbor's own parser
(`terminus_json_plain_parser.py`): required `analysis`, `plan`, `commands`, with
each command `{"keystrokes": str, "duration": float}`, plus optional
`task_complete`.

Usage:
    tb_mock_endpoint.py --port 8000 --served-name qwen38 [--turns 2]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"calls": 0, "turns": 2, "served": "qwen38"}


def terminus_reply(call_index: int, turns: int) -> str:
    """A valid Terminus-2 JSON response; declares completion on the last turn."""
    last = call_index >= turns
    if last:
        body = {
            "analysis": (
                "Preflight double: this is the final turn, so I declare "
                "completion so the agent loop exits cleanly instead of running "
                "to max_turns."
            ),
            "plan": "Record a marker file and finish.",
            "commands": [
                {
                    "keystrokes": (
                        "echo tb21-preflight-turn-%d > "
                        "/tmp/tb21_preflight_marker\n" % call_index
                    ),
                    "duration": 1.0,
                }
            ],
            "task_complete": True,
        }
    else:
        body = {
            "analysis": (
                "Preflight double, turn %d of %d: issue a harmless command whose "
                "output must come back to me on the next turn, which is what "
                "proves the multi-turn loop and the transcript capture."
                % (call_index, turns)
            ),
            "plan": "Print an identifying marker, then inspect the result.",
            "commands": [
                {
                    "keystrokes": "echo tb21-preflight-turn-%d; uname -m\n"
                    % call_index,
                    "duration": 1.0,
                }
            ],
            "task_complete": False,
        }
    return json.dumps(body, indent=2)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        print("[mock] " + (fmt % args), flush=True)

    def _send(self, obj: dict, code: int = 200) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": STATE["served"],
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "mock-preflight-endpoint",
                        }
                    ],
                }
            )
            return
        self._send({"error": {"message": f"no route {self.path}"}}, 404)

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except ValueError:
            req = {}
        msgs = req.get("messages") or []
        prompt_chars = sum(len(str(m.get("content", ""))) for m in msgs)

        STATE["calls"] += 1
        call_index = STATE["calls"]
        content = terminus_reply(call_index, int(STATE["turns"]))
        print(
            f"[mock] call #{call_index}: {len(msgs)} messages, "
            f"{prompt_chars} prompt chars -> "
            f"{'task_complete' if call_index >= int(STATE['turns']) else 'continue'}",
            flush=True,
        )

        if self.path.rstrip("/").endswith("/v1/completions"):
            # Used by `run_terminal_bench.sh probe`/`headroom`.
            self._send(
                {
                    "id": f"cmpl-{call_index}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": STATE["served"],
                    "choices": [
                        {"index": 0, "text": "ok", "finish_reason": "length"}
                    ],
                    "usage": {
                        "prompt_tokens": max(1, prompt_chars // 4),
                        "completion_tokens": 1,
                        "total_tokens": max(1, prompt_chars // 4) + 1,
                    },
                }
            )
            return

        self._send(
            {
                "id": f"chatcmpl-{call_index}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": STATE["served"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, prompt_chars // 4),
                    "completion_tokens": max(1, len(content) // 4),
                    "total_tokens": max(1, prompt_chars // 4)
                    + max(1, len(content) // 4),
                },
            }
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--served-name", default="qwen38")
    ap.add_argument("--turns", type=int, default=2)
    a = ap.parse_args()
    STATE["served"] = a.served_name
    STATE["turns"] = a.turns
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(
        f"[mock] preflight endpoint on {a.host}:{a.port} as "
        f"'{a.served_name}', {a.turns} turn(s) before task_complete",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
