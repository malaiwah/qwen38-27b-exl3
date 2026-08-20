# EXL3 graph-decode drift matches the tested BF16 control

A second party qualified [PR #314](https://github.com/local-inference-lab/vllm/pull/314)
on an RTX 5090 and found that CUDA-graph decode is internally deterministic but **not
bit-identical to eager**: 29/32 exact sequences, differences at tied or near-tied top-1
decisions. This is the follow-up that decides what that means.

## What my original parity check got wrong

The receipt I published with #314 reported `KLD 0.000000, top-1 1.000000` and I described
it as graph-vs-eager parity. It was not. `fidelity.py capture` takes **one prefill
forward** and hooks the final norm; `cudagraph_mode=FULL_DECODE_ONLY` captures **no
prefill graph**, so that measurement compared two runs of the same eager prefill code and
could not have detected a decode-path difference. The correct claim from that receipt is
narrow: *enabling graph decode does not change prefill numerics*. It says nothing about
decode. Retracted upstream.

## The right harness

`tools/decode_parity.py`: 32 prompts x 32 tokens, `temperature 0`, fixed seed,
`logprobs 5`, `ignore_eos`, via the OpenAI endpoint, so real decode steps execute. It
records chosen tokens and their logprobs, compares sequences, reports the first
divergence index with both candidates' logprobs, and runs each corpus twice per mode to
prove within-mode determinism. Collection and comparison are separate subcommands because
two servers cannot co-reside on one 96 GB GPU at 0.85 utilisation.

## Results, one RTX PRO 6000 Blackwell (SM120), seed 314159

| pair | exact sequences | matched prefix tokens | mean abs delta logprob | max | self-repeat |
|---|---:|---:|---:|---:|---:|
| EXL3 K5K6: eager vs graph | 24/32 | 853 | 0.01180 | 0.139 | 32/32 both |
| EXL3 K5K6: eager vs graph, **randomised priming** | 25/32 | 856 | 0.01179 | 0.139 | 32/32 both |
| **BF16 Qwen3.8-27B (no EXL3 code at all): eager vs graph** | **24/32** | 885 | **0.01282** | 0.125 | 32/32 both |

Three conclusions, in order of importance:

1. **The measured EXL3 drift is within the BF16 control's aggregate envelope on
   this build.** BF16—which touches no EXL3 kernel, online overlay, or EXL3
   priming—also produced 24/32 exact sequences and a similar mean logprob delta
   (0.0128 versus 0.0118). This rules out a large EXL3-specific excess in this
   sample; identical aggregates do not prove the per-prompt drift mechanism is
   exclusively ambient.
2. **The zero-priming hypothesis is refuted.** I expected autotune to pick a different
   kernel configuration when primed on an all-zero arena than eager picks on real
   activations. Priming on realistic activations (`0.05 * randn`) changed nothing:
   25/32 vs 24/32, identical logprob deltas to five digits. The randomised priming is
   kept anyway - probing a timing-based autotuner with zeros is indefensible on its own
   terms - but it is not a fix, and I am labelling it as such.
3. **Each mode is deterministic; only cross-mode comparison drifts.** 32/32 self-repeat
   in all four configurations. Divergences land on near-ties, e.g. `-2.8823` vs `-2.9020`
   for ` known` vs ` home`.

## What this implies for the production HOLD

The qualification's gate—"keep experimental until maintainers decide whether the
reproducible near-tie drift is within intended numerical tolerance"—should compare
EXL3 graph drift with an unquantized control on the same stack, not demand
bit-exact graph/eager equality that the tested BF16 path also fails. On these 32
prompts the EXL3 result lies within that measured control envelope.

Remaining honest caveat: 32 x 32 tokens with greedy sampling is a numerical probe, not a
task evaluation. Neither this nor the 5090 qualification shows whether the flipped
near-ties matter downstream. That is a graph-path task-retention question; this
sample found no EXL3-specific excess but does not rule one out universally.
