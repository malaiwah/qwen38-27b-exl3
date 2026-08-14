# Recipe: `malaiwah/Qwen3.8-27B-K4`

## Principle

Keep exactly the tensors the NVFP4 recipes protect, but express the protection
in the Gilded Gnosis runtime's own currency: leave them **BF16 on disk** and let
`ONLINE_QUANT=exl3-b6` encode them to **K6** at load time into a
content-addressed cache. Serialize everything else at **EXL3 K4**.

Rationale for BF16-on-disk over serialized K6: the r34 receipt measured merged
runtime K6 against separately serialized K6 on GLM-5.2 and found mean KLD
0.065339 vs 0.064467 (delta 0.00087, overlapping run SD) — no measurable quality
cost — while BF16 on disk keeps the tensors re-encodable at any future bit width.
The cost is download size, quantified below.

## Bit accounting

EXL3 packed size, from the encoder's own shapes
(`trellis` int16 `(in/16, out/16, 256K/16)` + `suh`/`svh` fp16 + one int32
codebook scalar):

```
bytes = in*out*K/8 + 2*(in+out) + 4
bpw   = K + 16*(in+out)/(in*out) + 32/(in*out)
```

For the MLP shapes (5120x17408) at K4 that is **4.00404 bpw** — versus NVFP4's
**4.50 bpw**, because NVFP4 carries an FP8 scale per 16 weights and trellis
carries none. That 0.5 bpw over 17.11 B params is the 1.07 GB that funds the
rest of the recipe.

| role | on disk | in VRAM | vs NVIDIA's choice |
|---|---|---|---|
| MLP `gate/up/down`, 64 L | EXL3 K4, 8.56 GB | 8.56 GB | 9.63 GB NVFP4 → **−1.07 GB** |
| `linear_attn` + `self_attn`, 64 L | BF16, 14.43 GB | K6, 5.41 GB | 7.22 GB FP8 → **−1.80 GB** |
| `lm_head` | EXL3 K6, 0.95 GB | 0.95 GB | 0.72 GB NVFP4 → +0.24 GB |
| `embed_tokens` | BF16, 2.54 GB | 2.54 GB | same |
| vision tower | BF16, 0.92 GB | BF16, 0.92 GB | same |
| MTP | EXL3 K6, 0.32 GB | 0.32 GB | 0.85 GB BF16 → −0.53 GB |
| norms/biases | BF16, 0.05 GB | 0.05 GB | same |

## Candidate footprints

| recipe | disk | VRAM | vs nvidia 21.92 | vs unsloth 23.42 |
|---|---:|---:|---:|---:|
| **A** MLP K4, attn BF16→K6, head K6, MTP K6 | 28.30 | **18.75** | −3.17 | −4.67 |
| **B** = A + all 64 `down_proj` at K6 | 29.73 | **20.18** | −1.74 | −3.24 |
| **C** = B + last-8-layer MLP BF16→K6 | 32.76 | 20.54 | −1.38 | −2.88 |
| **D** all-K4 MLP, attention serialized K6 (no online quant) | **19.29** | 19.29 | −2.63 | −4.13 |

Headroom exchange rates, for spending the gap to NVFP4 parity:

| promotion | VRAM cost |
|---|---:|
| one MLP layer, `gate+up+down`, K4 → K6 | +0.067 GB |
| all 64 `down_proj`, K4 → K6 | +1.426 GB |
| `lm_head`, K6 → BF16 | +1.589 GB |
| MTP, K6 → BF16 | +0.53 GB |

**Ship A first.** It is the literal reading of "everything else in K4", is
3.17 GB under the NVFP4 baseline, and every role is at equal or higher effective
precision than NVIDIA's. B is the follow-up once KLD is measured, because
`down_proj` sums 17408 quantization errors per output element and is the
projection most likely to dominate the residual; reaching B requires either two
conversions plus `util/optimize.py`, or the three-function upstream patch
described in [04-exllamav3-toolchain.md](04-exllamav3-toolchain.md).

## Conversion command

```bash
python convert.py -i /models/Qwen3.8-27B -o /work/qwen38-k4 -w /work/wd-k4 \
  -b 4 -hb 6 -mb 6 -vb 16 -cb mcg -d 0
```

`-vb 16` builds no vision model at all, so the vision tensors pass through the
compile-time extras path with `allow_bf16=True` — genuine BF16 on disk, matching
both vendors' treatment for free. `-cb mcg` matches the codebook the runtime's
native dense path expects. `-hq` is a documented no-op here
(`select_hq_bits = 2 if use_moe else 0`, and this model is dense).

Attention is then spliced back to BF16 with
[`tools/splice_bf16_attn.py`](../tools/splice_bf16_attn.py), because the
converter cannot emit BF16 for a decoder linear.

## Expected quality direction

Every role is at equal or better precision than the NVFP4 baseline except MLP
activations, which stay BF16 here versus Unsloth's W4A4. `[INFERENCE]` The
external EXL3 error ladder measured on MoE experts elsewhere in this project
(per-expert epsilon: K3 0.0231, K4 0.0060, K5 0.0016, about 3.8x per bit)
implies K6 near 4e-4, but that is a different model and a different tensor
population — it is not evidence for this checkpoint. The claim gets settled by
the measurement in [05-kld-protocol.md](05-kld-protocol.md), not by arithmetic.
