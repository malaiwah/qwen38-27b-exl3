# Publishing iterative variants

The point of iterating is that someone else can re-measure. Each variant is
therefore published as a **named branch of one Hugging Face repo**
(`malaiwah/Qwen3.8-27B-K4`) rather than as a new repo, so `--revision` selects a
variant and `main` always points at the recommended one.

| branch | recipe | VRAM | status |
|---|---|---:|---|
| `main` | A: MLP K4, attention BF16→online K6, `lm_head` K6, MTP/embed/vision BF16 | 19.28 GB | first publish |
| `v-down6` | B: A + all 64 `down_proj` at K6 | 20.71 GB | planned; needs two conversions + `util/optimize.py`, or the 3-function converter patch |
| `v-tail8` | C: B + last-8-layer MLP BF16→K6 (Unsloth's tail-protection pattern) | 21.07 GB | planned |
| `v-serialized-k6` | D: MLP K4 + attention serialized K6, no online overlay | 19.29 GB | planned; halves the download, removes the encode-at-load step |

Measured evidence that motivates B first: in the conversion log, `down_proj`
carries the largest per-tensor proxy error of any projection in every layer —
about 0.0025 versus 0.0011 for `gate_proj` and 0.0010 for `in_proj_qkv`. Bits
spent there should buy the most KLD.

Each branch must carry, in the same commit: its `quantization_config.json`, the
`summary.json` from the KLD run, the exact serve command, and the measured VRAM
figure. A variant without its own KLD receipt is not published to `main`.
