# Iteration 1 re-measured under the v2 fidelity protocol

Protocol: [14-fidelity-protocol-v2.md](14-fidelity-protocol-v2.md). **74 contexts x
2047 scored positions = 151,478 positions**, exact full-vocabulary two-pass
`KL(BF16 reference || candidate)` through one shared BF16 LM head, float64
accumulation, source-cluster bootstrap. Suite token SHA-256
`2c943c3972e7d1ca68b8acf5c8ad6f0426c2d5d218c80a9be07166e0646a915a`.

Both operands replay through the *same* BF16 head, so these numbers measure the
**transformer body** with head quantization deliberately factored out.

| candidate | mean KLD | cluster bootstrap 95 % CI | median | p99 | p99.9 | JSD (bits) | top-1 | resident weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **this quant** (MLP K4 + online-K6 attention) | **0.026231** | [0.01259, 0.04276] | 0.002608 | 0.2521 | 4.567 | 0.005826 | 96.03 % | 19.21 GB |
| `unsloth/Qwen3.8-27B-NVFP4` | **0.073006** | [0.03973, 0.11218] | 0.009380 | 1.0395 | 10.291 | 0.016597 | 92.62 % | 23.42 GB |

**Paired over the 74 shared contexts** (negative favours this quant):

| statistic | value |
|---|---:|
| mean difference (ours - NVFP4) | **-0.046775** |
| median context difference | -0.020358 |
| cluster bootstrap 95 % CI of the difference | **[-0.07069, -0.02628]** |
| contexts where ours is closer to BF16 | **74 / 74** |

The CI excludes zero and the win count is unanimous: **2.78x lower mean KLD,
3.60x lower median, 2.85x lower JSD, +3.4 points of top-1 agreement, at 4.2 GB
less resident weight.**

## Per-stratum means

| stratum | contexts | ours | NVFP4 | NVFP4 / ours |
|---|---:|---:|---:|---:|
| code | 16 | 0.010275 | 0.025372 | 2.47x |
| encyclopedic | 16 | 0.006712 | 0.023274 | 3.47x |
| multilingual | 9 | 0.006649 | 0.025559 | 3.84x |
| short_form | 16 | 0.089237 | 0.228902 | 2.57x |
| technical | 16 | 0.010589 | 0.044048 | 4.16x |
| web | 1 | 0.012282 | 0.026900 | 2.19x |

Both degrade most on `short_form` (short, low-context documents), and our margin
is largest exactly there.

## Tails, which the mean hides

Ours: median 0.0026, p99 0.252, p99.9 4.567, max 22.95.
NVFP4: median 0.0094, p99 1.040, p99.9 10.291.

Heavily tail-dominated, which is why top-1 agreement matters: **96.03 %** versus
92.62 % means disagreeing with BF16's greedy choice on 1 token in 25 rather than 1
in 13.5.

## Relationship to the v1 number

v1 (one window, each model's own head): 0.034030 ours vs 0.091457 NVFP4, ratio
2.69x. v2 (74 contexts, shared head): 0.026231 vs 0.073006, ratio 2.78x. Different
corpora and a different head treatment, same ordering and same magnitude of
advantage.

## Official FP8 added as a third data point

`Qwen/Qwen3.8-27B-FP8` (revision `017b9c7a`, 30.9 GB of weights) on the same
suite, same reference captures, same shared BF16 head:

| candidate | mean KLD | bootstrap 95 % CI | median | p99.9 | JSD (bits) | top-1 | weights |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | **0.019309** | [0.00883, 0.03172] | 0.001994 | 3.676 | 0.004087 | 96.68 % | 30.9 GB |
| this quant (K4 + online K6) | 0.026231 | [0.01259, 0.04276] | 0.002608 | 4.567 | 0.005826 | 96.03 % | 19.2 GB |
| `unsloth/Qwen3.8-27B-NVFP4` | 0.073006 | [0.03973, 0.11218] | 0.009380 | 10.291 | 0.016597 | 92.62 % | 23.4 GB |

Paired, ours versus FP8 (positive favours FP8):

| statistic | value |
|---|---:|
| mean difference (ours - FP8) | **+0.006923** |
| median context difference | +0.002568 |
| bootstrap 95 % CI | [0.00314, 0.01167] |
| contexts favouring FP8 | 68 / 74 |

**FP8 is genuinely better, and it should be**: 30.9 GB of weights against our
19.2 GB — 61 % more memory for 26 % less divergence. The CI excludes zero, so the
gap is real. Plainly:

- Against its own size class (NVFP4, 23.4 GB) this quant wins decisively: 2.78x
  lower KLD **and** 4.2 GB smaller.
- Against 8-bit at 1.6x the memory it loses by 0.0069 KLD and 0.65 points of top-1.
- The ordering `FP8 < K4-mixed < NVFP4` is directionally consistent with the
  independently published Qwen3.6 ordering (FP8 0.017, 4-bit GGUF
  0.013-0.035, NVFP4 0.039-0.044) from Quesma. Different models and protocols
  mean this agreement is a cross-check, not validation of either harness.

That **0.0069** is the concrete iteration-2 target, and the budget for it is the
2.7 GB still unspent under the NVFP4-equivalent ceiling.
