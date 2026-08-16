# Card/docs paragraph — scratch-arena lever (handed to Main; I do not touch cards)

Ready to place verbatim (adjust heading level to the target section). Also
for docs/43's owner: §3/§7's "+620 MiB ≈ +18.7k tok MTP-3 \[INFERENCE\]"
rows are now measured — relabel to **measured +17,874 tokens / +0.60 GiB**
(receipts/scratch-arena.json).

---

**Reconstruct-scratch arena (fork patch, opt-in overlay).** The pinned r34
image keeps one persistent fp16 prefill-reconstruct scratch per weight
geometry — 790 MiB across this checkpoint's 9 geometries at serve time. A
2-hunk patch to `exl3.py` (overlay `tools/vllm-exl3-scratch-arena.py`,
sha256 `9aba06eb…`; fork PR
[local-inference-lab/vllm#397](https://github.com/local-inference-lab/vllm/pull/397))
shares one grow-to-max arena per device instead, sized by the largest live
geometry (170 MiB), because each reconstruct is written and consumed inside
one eager call on one stream. The kernels see identical operands — same
shapes, strides, and dtypes. Measured on the physical RTX 5090 A/B at the
qualified 262,144-token profile: engine-reported KV pool **265,122 →
282,996 tokens (+17,874, +6.7 %, ≈ +0.60 GiB)**, reproduced identically
across two server starts per arm; the 30-case deterministic vision suite
returned byte-identical answers on both arms (24/30, equal to the
qualification reference); a full-window needle (258,925 tokens, depth 0.5)
retrieved exactly; decode unchanged (108.5–108.9 vs 109.2–109.7 tok/s over
three warmed C1 runs). The static prediction was +620 MiB / +18.7k tokens;
the measured gain is 95.7 % of it — quote the measured number. On the 24 GB
class the same bytes put the published 24,576-token window at 42,450 raw
headroom (supports 40,960 at the next window step) — arithmetic only until
a 24 GB-class boot confirms it.
