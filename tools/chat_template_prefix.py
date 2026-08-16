#!/usr/bin/env python3
"""Measure two things no amount of arguing settles:

1. How many prompt tokens each template costs over a growing multi-turn
   conversation, with and without the client returning `reasoning_content`.
2. Where the *re-rendered* history first diverges, in token space, from the
   token sequence that was actually generated on the previous turn.  vLLM's
   prefix cache is keyed on token-block hashes from the start of the prompt,
   so the first divergent token index is exactly the point past which no
   cached block can be reused.

Runs inside the pinned image rootfs. CPU only; no weights, no CUDA.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/work/chat-template-audit")
from chat_template_matrix import (  # noqa: E402
    TEMPLATES, TPL_DIR, compile_template,
)
from tokenizers import Tokenizer  # noqa: E402

TOKENIZER = "/models/Qwen3.8-27B/tokenizer.json"
BLOCK = 16  # vllm default prefix-cache block size in tokens


def build(n_turns, with_reasoning):
    """A conversation of n_turns completed assistant replies, plus a new user turn."""
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"Question number {i}: explain topic {i}."})
        a = {"role": "assistant", "content": f"Answer number {i} for topic {i}."}
        if with_reasoning:
            a["reasoning_content"] = f"Let me think about topic {i} carefully first."
        msgs.append(a)
    msgs.append({"role": "user", "content": "One more: summarise everything."})
    return msgs

def divergence(gen_ids, new_ids):
    """Where the re-rendered prompt stops matching the generated token stream.

    Returns (index, kind).  ``kind == "strict_prefix"`` means the generated
    sequence is a strict prefix of the re-render, i.e. every block the engine
    already holds is reusable and the prefix cache hit is total.
    """
    n = min(len(gen_ids), len(new_ids))
    for i in range(n):
        if gen_ids[i] != new_ids[i]:
            return i, "mismatch"
    if len(new_ids) >= len(gen_ids):
        return len(gen_ids), "strict_prefix"
    return len(new_ids), "truncated"

def main():
    tok = Tokenizer.from_file(TOKENIZER)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    compiled = {}
    for name, fn in TEMPLATES.items():
        src = open(f"{TPL_DIR}/{fn}", encoding="utf-8").read()
        try:
            compiled[name] = compile_template(src)
        except Exception:  # noqa: BLE001
            pass

    out = {"block_size": BLOCK, "growth": {}, "prefix_stability": {}}

    # ---- 1. token growth over turns ------------------------------------
    for name, tpl in compiled.items():
        out["growth"][name] = {}
        for wr in (False, True):
            row = {}
            for n in (1, 2, 5, 10, 20):
                try:
                    s = tpl.render(messages=build(n, wr), tools=None,
                                   documents=None, add_generation_prompt=True)
                    row[n] = len(enc(s))
                except Exception as e:  # noqa: BLE001
                    row[n] = f"ERR {type(e).__name__}"
            out["growth"][name]["with_reasoning_content" if wr else
                                "without_reasoning_content"] = row

    # ---- 2. prefix stability -------------------------------------------
    # Turn N: prompt is rendered for 2 completed turns + a new user question.
    # The model then generates "<reasoning>\n</think>\n\nanswer<|im_end|>\n".
    # Turn N+1 re-renders the whole history plus one more user question.
    # Whatever the client echoes back determines the re-rendered history.
    reasoning = "Let me think about topic {i} carefully first."
    answer = "Answer number {i} for topic {i}."
    for name, tpl in compiled.items():
        rec = {}
        for depth in (2, 10):
            i = depth
            r_txt, a_txt = reasoning.format(i=i), answer.format(i=i)
            sub = {}
            try:
                prompt_n = tpl.render(messages=build(depth, True), tools=None,
                                      documents=None, add_generation_prompt=True)
            except Exception as e:  # noqa: BLE001
                rec[f"depth_{depth}"] = {"error": f"{type(e).__name__}: {e}"}
                continue
            # what the engine's KV cache actually holds once turn N completes
            generated = prompt_n + r_txt + "\n</think>\n\n" + a_txt + "<|im_end|>\n"
            gen_ids = enc(generated)

            base = build(depth, True)
            hist = base + [{"role": "assistant", "content": a_txt,
                            "reasoning_content": r_txt}]
            hist_drop = base + [{"role": "assistant", "content": a_txt}]
            hist_inline = base + [{"role": "assistant",
                                   "content": f"<think>\n{r_txt}\n</think>\n\n{a_txt}"}]

            for label, h in (("client_echoes_reasoning_content", hist),
                             ("client_drops_reasoning_content", hist_drop),
                             ("client_inlines_think_in_content", hist_inline)):
                h2 = list(h) + [{"role": "user", "content": "Next question please."}]
                try:
                    s = tpl.render(messages=h2, tools=None, documents=None,
                                   add_generation_prompt=True)
                except Exception as e:  # noqa: BLE001
                    sub[label] = {"error": f"{type(e).__name__}: {e}"}
                    continue
                ids = enc(s)
                d, kind = divergence(gen_ids, ids)
                sub[label] = {
                    "generated_prefix_tokens": len(gen_ids),
                    "rerendered_tokens": len(ids),
                    "first_divergent_token": d,
                    "kind": kind,
                    "reusable_cache_blocks": d // BLOCK,
                    "total_generated_blocks": len(gen_ids) // BLOCK,
                    "cache_reuse_fraction": round(d / len(gen_ids), 4),
                    "full_prefix_reuse": kind == "strict_prefix",
                }
            sub["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
            rec[f"depth_{depth}"] = sub
        out["prefix_stability"][name] = rec

    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
