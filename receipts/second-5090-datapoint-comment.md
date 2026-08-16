Thank you — that update is two distinct results, and they point in opposite directions. Both are worth having.

## 1. You closed the decomposition

Prefix caching plus `--mamba-cache-mode align`, at the native 262,144 window, on a **second physical RTX 5090**, with LMCache out of the picture: it works. That is the arm I could not supply, because I only have my own card.

Put next to the probe I posted — seven freshly started servers, 38 requests each, **266 scored requests**, nested token prefixes so later requests hit blocks published by earlier ones, and no prompt length a multiple of the measured 1,600-token mamba block, so prefill chunks end mid-block by construction: **zero corrupted responses, zero wrong answers, zero acceptance collapses**, on the unpatched image as well as the patched one ([`receipts/apc-poison-repro.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/apc-poison-repro.json)) — your original control is now fully decomposed. It removed four things at once: the LMCache wrapper and connector, the cleared L2 files, `--enable-prefix-caching`, and `--mamba-cache-mode align`. The last two are now cleared on two independent cards. **LMCache 0.5.2 in `kv_both` at chunk 1600 is the only member of that set still standing.**

Being exact about what that is worth: it is elimination, not a positive identification. I still have not run LMCache and I am not calling it guilty — only that nothing else you removed is left to blame.

## 2. You corrected my card, and the wording is changing

You are the second independent data point on utilisation, and you falsify something I should never have printed as a constant.

My cards print `--gpu-memory-utilization 0.955` as if it were *the* value. It is not. It is the value measured on **one specific board**: `GPU-506a575d`, 32,607 MiB total with 458 MiB held by the driver, so 32,149 MiB CUDA-visible, driver 610.57.04 — all seven gates, **265,122 KV tokens** at 262,144 and 1.01× concurrency ([`receipts/qualification-5090-context.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-5090-context.json)). Yours needed **0.956** and missed at 0.955 by about **0.01 GiB**. The honest claim is therefore "measured on one card, and the margin is thin", not "0.955".

The mechanism for that is already measured, in a receipt about something else entirely. When I emulated a 24 GiB board on this 5090, the only two quantities I could not emulate were exactly the two that are properties of the *board* rather than of the configuration: the **driver's framebuffer reserve** (458 MiB here) and the **CUDA context** (0.496 GiB here). In the same run the most fragile gate passed by **68 MiB** — a 0.3 % perturbation in either of those two numbers would have flipped it ([`receipts/qualification-24gib-capped.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-24gib-capped.json) → `residual_risk_versus_a_physical_board`). The engine requests `ceil(cudaMemGetInfo_total × utilisation)`, and on this card one thousandth of utilisation is about **32 MiB**. So two nominally identical 5090s landing 0.001 apart is *expected behaviour*, not a defect in your card or in mine.

The cards will now say that, and will tell anyone hitting a startup OOM to raise utilisation **0.001 at a time** rather than drop the window. They also keep the existing warning not to reach for `max_pixels` instead: at fixed utilisation, lowering it lowers profiled activation, the engine spends the freed bytes on more KV, and the large-image request then OOMs *sooner* — 4.2 MP failed with 6.56 MiB free where 8.4 MP had 26.50 MiB, measured in the same receipt. 0.955 stays in the cards as a measured value; what is leaving is the implication that it is *your* value.

## 3. Three questions, because your configuration succeeds where mine failed

Here is the awkward part, and it is why I am asking rather than simply publishing 0.956 and moving on. I tried to qualify prefix caching at 262,144 on my card and it failed **three different ways** as I gave the engine more room ([`receipts/qualification-5090-apc.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-5090-apc.json)):

| `--gpu-memory-utilization` | available KV | pool | outcome |
|---|---|---|---|
| 0.955  | 9.28 GiB | — | **refuses to start**: needs 9.29 GiB, suggests max len 260,800 |
| 0.9555 | 9.30 GiB | 262,144 (1.00×) | starts, then **deadlocks** mid-prefill, 0 tok/s |
| 0.958  | 9.38 GiB | 263,608 (1.01×) | starts (no long request sent) |
| 0.9585 | 9.39 GiB | 265,072 (1.01×) | starts, then **livelocks** |
| 0.959 / 0.96 | 9.41 / 9.44 GiB | 265,072 | identical pool |

Look at the first row: 9.29 GiB needed against 9.28 available is a **0.01 GiB** shortfall — your number exactly — and on my card the whole of it is `align`-mode block rounding, because 262,144 tokens occupy 164 whole 1,600-token blocks, i.e. 262,400 slots. So one reading of your report is that you reproduced my startup refusal precisely and 0.956 is a coarser version of my 0.9555 bump. Another is that your board's reserve or context genuinely differs. I cannot tell which from here, and the difference matters, because one notch higher my engine **accepted** a 261,794-token request, prefilled it to 98.9 % of the pool, dropped it back to the waiting queue with the pool freed, and re-prefilled on a 30-second cycle: about 960 tok/s of wasted prefill, **zero output tokens in 656 seconds**, `vllm:num_preemptions_total` stuck at `0.0`, and 261,794 prefix-cache queries against **0** hits. Filed upstream as [issue #394](https://github.com/local-inference-lab/vllm/issues/394).

So, three questions:

1. **What `--max-model-len` and `--max-num-batched-tokens` are you running with the cache on?** Mine were `262144` and `2048`. Those two set the startup KV requirement and the prefill chunking respectively, so if yours differ that alone could be the whole difference between your success and my refusal — and it would mean the 0.956 is buying something other than what I think it is.
2. **Have you ever sent a single request that comes close to the full window with prefix caching enabled** — say a 250k+ token prompt — and received tokens back? This is the one that matters most. My failure is completely invisible below the ceiling: a server answering 8k or 128k prompts looks perfectly healthy, and the livelock only appears when one request needs nearly the whole pool. If you have retrieved from a ~260k prompt with the cache on, that is a new result.
3. **What is your `--max-num-seqs`?** Mine was 1. It decides whether mixed batches can form at all, which bears both on the scheduling path above and on whether the overlay in the next section is relevant to you.

If your configuration really does serve a near-full-window request with the cache on, I want to publish it **as yours**, with your flags and your credit. It is currently the only evidence anywhere that this can be made to work, and it would turn my "declined at the native window" into "declined on our card, works on this configuration".

## 4. What shipped since my last message — both parts useful to you

- **Prefix caching now ships** on the three 8,192-token recipes (K4, K5K6, hydrated) with `--enable-prefix-caching --mamba-cache-mode align`, and is **declined at the native window** for the reasons in the table above — with a measured option if you want the cache and can spare a little context: `--max-model-len 256000` at 0.9585, pool 264,777 (1.03×), which retrieved a 254,964-token needle exactly in 179 s and then served a second 254,967-token request in the same process after release. The price is **6,144 tokens of context, 2.34 %** ([`receipts/qualification-5090-apc.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/qualification-5090-apc.json)).
- **The #51812 GDN overlay is now recommended** for prefix-caching recipes running more than one sequence. A CPU-only counter mounted over the engine's GDN metadata builder measured the defective path actually being entered: **three events in 5,825 metadata builds, 0.515 per thousand**, at eight concurrent streams with prefix caching on and MTP-3, against **zero events in 3,329 builds** with the cache off ([`receipts/gdn-gate-concurrency.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/gdn-gate-concurrency.json)). Mount `tools/vllm-qwen-gdn-spec-gates.py` (`sha256 7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0`) read-only over `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`. **At `--max-num-seqs 1` no mixed batch can form and the overlay is a no-op** — which is why question 3 above is not idle curiosity.

Your report, the two consequences, the card change and these three questions are written up as [`receipts/second-5090-datapoint.json`](https://github.com/malaiwah/qwen38-27b-exl3/blob/main/receipts/second-5090-datapoint.json), so the card edit has a source that names you. Thank you for measuring on your own hardware and reporting the number that disagreed with mine — that is the only way the second data point ever arrives.
