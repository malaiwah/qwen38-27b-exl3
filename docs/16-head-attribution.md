# Is `lm_head` the sensitive tensor? Measured: not at K6

The working assumption entering iteration 2 was that `lm_head` is highly
quantization-sensitive, so promoting it from K6 to BF16 (+1.589 GB) was ranked P1.
The v2 harness can settle that without re-serving anything, because it replays
stored hidden states through an arbitrary head.

## Method

1. Reconstruct the serialized K6 head as a dense BF16 matrix using exllamav3's own
   `reconstruct_had_slice` kernel — original-basis weights with both Hadamards and
   the sign/scale vectors folded in, i.e. exactly the matrix the serving path
   multiplies by, not an independent dequantiser
   ([`tools/dequant_head.py`](../tools/dequant_head.py)). Output verified finite,
   `absmax = 0.34180`, shape `[248320, 5120]`.
2. Add `--candidate-head` to the comparator so the reference operand and the
   candidate operand can use *different* heads. Without it, one head applies to
   both sides and head error cancels by construction (the first attempt did
   exactly that and returned a difference of 4e-06 — a null result for the wrong
   reason).
3. Three replays over the same 74 contexts / 151,478 positions.

## Results

| configuration | reference | candidate | mean KLD | median | p99.9 | top-1 |
|---|---|---|---:|---:|---:|---:|
| **head error alone** | BF16 body + BF16 head | BF16 body + **K6 head** | **0.000367** | 7.12e-05 | 0.0054 | 99.31 % |
| body only | BF16 body + BF16 head | K4 body + BF16 head | 0.026231 | 0.002608 | 4.567 | 96.03 % |
| end to end (what is served) | BF16 body + BF16 head | K4 body + **K6 head** | 0.026299 | 0.002673 | 4.565 | 95.97 % |

Paired, body-only versus end-to-end over the shared contexts:
difference **-6.78e-05**, bootstrap 95 % CI
[-9.01e-05, -4.63e-05],
**64/74 contexts worse with the K6 head** (better in 10).

## Verdict

The K6 head is **detectable but negligible**: it costs
6.78e-05 of KLD on top of the body, which is
**0.26 % of total divergence**, and 0.06 points of top-1
agreement. In isolation it contributes 0.000367 with 99.31 % top-1
agreement against the BF16 head.

So the anecdote does not hold *at 6 bits on this model*. Promoting the head to
BF16 could recover at most 6.78e-05 — about 1 % of the
0.0069 gap to FP8 — in exchange for 1.589 GB, which is 59 % of the entire remaining
memory budget. **P1 is cancelled.**

Two caveats worth keeping:

- This measures K6, not K4. A 4-bit head is a different question and the same
  method answers it in minutes if a K4-head variant is ever built.
- The isolated head error is not additive with body error: 0.000367 alone but only
  0.0000678 incrementally, so the two error sources partially cancel — the same
  cancellation seen when online-K6 attention scored better than BF16 attention.

## Consequence for the budget

The 2.7 GB under the NVFP4-equivalent ceiling should go entirely to the **MLP
stack**, which owns ~99 % of the measured divergence. `down_proj` K4 to K6 costs
1.426 GB and targets the projection with the largest per-tensor proxy error
(2.5e-3 versus 1.1e-3 for `gate_proj`); the remainder buys `gate`/`up` promotions
on the layers an error-driven allocator picks.

## Correction (independent review, 2026-08-14)

The win count in the table above was reported backwards on first publication. In
`paired-head-e2e.json`, candidate A is the body-only configuration and `a_wins = 64`,
so **the K6 head is worse in 64 of 74 contexts and better in 10** — not the reverse.
The aggregate cost and its sign were correct; only the count was misstated.

A second correction from the same review: the claim that "the MLP owns ~99 % of the
divergence" is stronger than this ablation supports. The measured body contains both
the K4 MLP **and** online-K6 attention, and the report itself observes non-additive
error cancellation between them. The defensible statement is that the **head**
contributes ~0.26 % of end-to-end divergence at K6, so the remaining ~99 % lives in
the body, whose split between MLP and attention this experiment does not resolve.

Both corrections are subject to the resolution floor noted in
[18](18-results-fidelity-v3.md): at 6.5e-04 replay error, a 6.8e-05 effect is below
what the current storage format can resolve, so this ablation must be redone with
fp32 captures before it is treated as settled.
