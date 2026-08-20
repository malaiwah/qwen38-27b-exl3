# The Qwen 3.x chat template under vLLM: what is actually wrong, and whether ours is

Audit date 2026-08-16. Receipt: `receipts/chat-template-audit.json`. Extracted templates and
raw API dumps: `receipts/chat-templates/raw/`. Full rendered strings for every case:
`receipts/chat-templates/renders.json`.

## Bottom line

**Ours is not wrong.** The `chat_template.jinja` in all four published repos is
byte-identical (sha256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`) to
`Qwen/Qwen3.8-27B` at its current HEAD and to `turboderp/Qwen3.8-27B-exl3`. So is the
`tokenizer_config.json` that carries a second copy of it. It is also the most faithful renderer
of the wire format `--tool-call-parser qwen3_coder` expects of the seven templates measured: it
is the only 3.8-generation template that round-trips a whitespace-bearing tool argument
losslessly **and** refuses, rather than silently mangles, a malformed one.

It is not perfect. Three request shapes get an HTTP 400 from the template that a more permissive
template would serve, and one construct fabricates empty `<think>` blocks. All four are
documented below with the exact construct and the exact observation. None is worth forking
upstream for, and the recommendation is to fix them in documentation and upstream, not in our
repos.

**The template is stale by zero revisions.** `Qwen/Qwen3.8-27B` has 16 commits;
`chat_template.jinja` appeared at `72a217afab` (2026-08-13T08:23:30Z) and was revised exactly
once, 1h53m later, at `412f8b6bd7` (2026-08-13T10:16:05Z), to the bytes we ship. Every commit
after that through HEAD `1d4bf0f2` (2026-08-14T15:00:01Z) touches `README.md` or `LICENSE` only.

**The loudest complaint about "official 3.8" is true — of a different model.** The claim that
official 3.8 hard-fails on `enable_thinking=false` is correct for the 3.8 *flagship*,
`Qwen/Qwen3.8-2.4T-A95B`, whose template alone contains
`raise_exception('Disabling thinking is not supported.')`, and it was briefly correct for the
27B's superseded first template revision. It is false for the bytes we ship. Both origins are
dated and measured in [§6](#6-the-two-hour-window-and-the-flagship-that-explain-the-complaint).

**On the reported garbled-output incident: the reporter is right that the template is not the
issue, and nothing in this audit should be cited as a cause of it.** For plain and simple
multi-turn chat the community template he mounts renders a *byte-identical* prompt to ours — 14
of 34 audited cases match sha256 for sha256, including every non-tool single-turn and
system-prompt shape. A template cannot explain a difference between two runs that produce the
same bytes. Separately: because of an artifact-precedence rule in transformers 5.15.0
([§3](#3-which-artifact-the-engine-actually-reads)), if he mounted it the way its README tells
vLLM users to, he has been running *our* template the whole time.

**Fidelity/KLD numbers cannot be affected by any of this, and that is not a hedge.** The KLD
suite replays a pre-tokenised corpus; there is no `apply_chat_template` call and no `messages`
list anywhere in that path. No template change can move a KLD, a top-1 agreement, or a tail
statistic. The 40-task downstream suite and the 70-item MMLU-Pro matrix *do* go through the
template and would need re-running; throughput numbers would too, because prompt token counts
move. Precise costs in [§12](#12-recommendation-and-the-cost-of-changing).

---

## 1. What was captured

Everything below was fetched with its revision and hashed; nothing is paraphrased from a blog
post or a README. **Seven** distinct template digests exist across the whole family.

| repo / source | revision | artifact | sha256 | bytes |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff60` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| `turboderp/Qwen3.8-27B-exl3` (branch `5.00bpw`) | `a35e75a73bae` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| **`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`** | `1b5d076eca7a` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| **`malaiwah/Qwen3.8-27B-EXL3-K5K6`** | `9f02827e28ce` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| **`malaiwah/Qwen3.8-27B-EXL3-K5K6-context`** | `6d554341ea28` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| **`malaiwah/Qwen3.8-27B-K4`** | `25987180ace7` | `chat_template.jinja` | `c3cf9e34abf4f9e36c2d7216…` | 8952 |
| `unsloth/Qwen3.8-27B-NVFP4` | `16b6615af354` | `chat_template.jinja` | `12827f24b742ea4e80cdc12d…` | 9993 |
| `unsloth/Qwen3.8-27B-GGUF` | `f1bfb127c64f` | GGUF KV `tokenizer.chat_template` | `12827f24b742ea4e80cdc12d…` | 9993 |
| `froggeric/Qwen-Fixed-Chat-Templates` | `9f14778c92c3` | `chat_template.jinja` (`qwen3.8-froggeric-v22`) | `398edf5b5bb802fb6b9c9a8d…` | 19262 |
| `froggeric/Qwen-Fixed-Chat-Templates` | `9f14778c92c3` | `archive/v21_chat_template.jinja` | `d203f3342d8a7f8474dd5556…` | 16289 |
| `Qwen/Qwen3.8-2.4T-A95B` (flagship, text-only) | `207bd685a7e3` | `chat_template.jinja` | `40ce34a5bcbc0231462740…` | 7495 |
| `RedHatAI/Qwen3.8-2.4T-A95B` | `0337be2263eb` | `chat_template.jinja` | `40ce34a5bcbc0231462740…` | 7495 |
| `RedHatAI/Qwen3.8-2.4T-A95B-FP8` | `df106d79cb2c` | `chat_template.jinja` | `40ce34a5bcbc0231462740…` | 7495 |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8` | `chat_template.jinja` | `e84f32a23fdda27689f868aa…` | 7764 |
| `nvidia/Qwen3.6-27B-NVFP4` | `0893e1606ff3` | `chat_template.jinja` | `e84f32a23fdda27689f868aa…` | 7764 |
| `RedHatAI/Qwen3.6-27B-FP8` | `57d986c3ab03` | `chat_template.jinja` | `e84f32a23fdda27689f868aa…` | 7764 |
| `Qwen/Qwen3.8-27B` **superseded rev1** | `72a217afab` | `chat_template.jinja` | `49d20d090bd09c0030f3f12d…` | 9421 |
| pinned image `…/rust/src/chat/tests/templates/vllm_examples/tool_chat_template_qwen3coder.jinja` | image digest | Rust **test fixture** | `5a38bfa05833266240066aed…` | 6211 |
| pinned image `…/rust/src/chat/tests/templates/qwen3.jinja` | image digest | Rust **test fixture** | `e132ae041e1217b5e1114eb9…` | 4169 |
| pinned image `…/rust/src/chat/tests/templates/qwen35.jinja` | image digest | Rust **test fixture** | `ad22cb6365e58696ffc10f1f…` | 7756 |

Collection notes that matter:

- **Our four repo revisions move as the model cards are edited; the template digest does not.**
  The revisions above are HEAD as of 2026-08-16T12:05Z. `chat_template.jinja` has carried
  `c3cf9e34…` in all four since publication, and it is pinned as such in
  `receipts/k5k6-build-receipt.json`, `receipts/hydrated-build-receipt.json`,
  `receipts/context-build-receipt.json` and the matching `*-SHA256SUMS`. Verify against those,
  not against a commit id.

- **`turboderp/Qwen3.8-27B-exl3`'s `main` branch carries only `README.md` and
  `.gitattributes`.** The quants and tokenizer live on per-bpw branches
  (`2.00bpw` … `6.00bpw`). The `5.00bpw` branch's template is byte-identical to official.
- **`unsloth/Qwen3.8-27B-GGUF` ships no `chat_template.jinja` and no `tokenizer_config.json`
  at all.** Its template lives in GGUF metadata. `tools/gguf_chat_template.py` extracts it over
  HTTP range requests, reading 10,945,458 bytes of header + KV from
  `Qwen3.8-27B-Q4_K_M.gguf` and no tensor data. It is byte-identical to unsloth's NVFP4
  template.
- **Red Hat ships the 3.8 *flagship*, not the 27B.** `RedHatAI/Qwen3.8-2.4T-A95B` and
  `-FP8` both carry `40ce34a5…`, byte-identical to `Qwen/Qwen3.8-2.4T-A95B`; there is no
  RedHatAI Qwen3.8-27B repo. `RedHatAI/Qwen3.6-27B-FP8` is byte-identical to official 3.6.
  The flagship's template turns out to matter a great deal — see
  [§6](#6-the-two-hour-window-and-the-flagship-that-explain-the-complaint).
- **The community template is `froggeric/Qwen-Fixed-Chat-Templates`**, whose root
  `chat_template.jinja` self-identifies as `qwen3.8-froggeric-v22`. Its README is the best
  available written statement of the complaint and is adjudicated claim-by-claim in
  [§10](#adjudicating-the-froggeric-readme-claim-by-claim).

## 2. Which template the engine actually uses

Answered by reading the pinned image (vLLM `0.11.2.dev280+…20260810`, transformers 5.15.0,
jinja2 3.1.6), not by inference:

- **vLLM carries no built-in Qwen override.**
  `vllm/transformers_utils/chat_templates/registry.py`'s
  `_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK` has no `qwen*` entry, so `resolve_chat_template()`
  falls through to the model repo's own template for `model_type qwen3_5`.
- **The image ships no `examples/` directory.** The only `qwen3coder`/`qwen3`/`qwen35` `.jinja`
  files present anywhere in it are Rust unit-test fixtures under
  `/opt/vllm/rust/src/chat/tests/templates/`, loaded by `#[cfg(test)]` helpers in
  `src/renderer/hf/{mod,format}.rs`. Nothing on a runtime path reads them.
- **None of the image's own `serve-qwen*.sh` scripts pass `--chat-template`.** They rely on the
  checkpoint's template, as we do. (Incidentally `serve-qwen36-27b-nvfp4.sh` uses
  `--tool-call-parser qwen3_xml` while `serve-qwen35-397b-nvfp4.sh` uses `qwen3_coder` — see
  below, they are the same class.)
- **`qwen3_coder` and `qwen3_xml` are aliases in this build.**
  `vllm/tool_parsers/__init__.py:157-164` maps both names to
  `qwen3_engine_tool_parser.Qwen3EngineToolParser`, which subclasses the same
  `Qwen3Parser` with `structural_tag_model = "qwen_3_coder"`. `--reasoning-parser qwen3`
  resolves to `Qwen3ParserReasoningAdapter` over that same `Qwen3Parser`. Switching our flag to
  `qwen3_xml` would change nothing.
- **vLLM normalises two fields before the template runs**, which decides two of the loudest
  complaints:
  - `_postprocess_messages` (`chat_utils.py:1875-1915`) `json.loads()`es every assistant
    `function.arguments` string into a dict, on both the sync and async render paths, and
    turns an absent or empty value into `{}`.
  - an inbound assistant `reasoning_content` is renamed to `reasoning` by
    `ChatCompletionRequest._normalize_messages_before` (`protocol.py:486-509`) and then written
    back onto the conversation message under **both** keys (`chat_utils.py:1832-1836`), so the
    template's `message.reasoning_content` does receive it.

## 3. Which artifact the engine actually reads

This distinction matters, and it is the single most actionable finding for anyone trying to
override our template.

Nine of the ten HF repos audited ship the template **twice** — as `chat_template.jinja` and as
the `chat_template` key of `tokenizer_config.json` — with byte-identical contents in every
single case (`dual_artifact_identical: true` throughout the receipt). Only
`unsloth/Qwen3.8-27B-NVFP4` ships the `.jinja` alone.

In transformers 5.15.0, **`chat_template.jinja` wins.**
`tokenization_utils_base.py:1783` says so in a comment — *"If independent chat template file(s)
exist, they take priority over template entries in the tokenizer config"* — and line 1798
overwrites `init_kwargs["chat_template"]` after `tokenizer_config.json` has been loaded. Proved
empirically rather than taken on faith:

```
$ # tokenizer_config.json chat_template = "SENTINEL_FROM_TOKENIZER_CONFIG"
$ # chat_template.jinja                 = "SENTINEL_FROM_CHAT_TEMPLATE_JINJA"
WINNER: SENTINEL_FROM_CHAT_TEMPLATE_JINJA
AFTER REMOVING .jinja: SENTINEL_FROM_TOKENIZER_CONFIG
```

**Consequence.** froggeric's README instructs vLLM users to *"replace the `chat_template` string
in your `tokenizer_config.json`"*. Against any of our four repos that edit is a **no-op**: the
`.jinja` file still wins. Anyone who mounted the community template that way has been serving
ours. To actually mount it you must replace or delete `chat_template.jinja`, or pass
`--chat-template /path/to/chat_template.jinja`.

## 4. How the measurements were made

CPU only; no GPU was touched, and the rental stayed free.

Templates are rendered inside the pinned image rootfs via `tools/ggrun.sh` (proot) using an
exact reconstruction of the environment transformers builds in
`utils/chat_template_utils.py::_cached_compile_jinja_template`:
`jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
extensions=[AssistantTracker, jinja2.ext.loopcontrols])`, with the transformers `tojson`
override, the `raise_exception` global and `strftime_now`. Tokenisation uses the model's own
`tokenizer.json` with `add_special_tokens=False`, so `<|im_start|>`, `<think>`, `</think>`,
`<tool_call>` and `<|image_pad|>` each count as one token, while `<function=` and `<parameter=`
cost three each (`<`, `function`/`parameter`, `=`).

Round-trip fidelity is tested against the **real** parser code from the image:
`tools/chat_template_roundtrip.py` imports `vllm.parser.qwen3._qwen3_arg_converter` and applies
it to the rendered assistant turns, with the tools system prompt's syntactic example excluded
(every template inlines one, and it is not a call the parser would ever see in model output).

## 5. Named constructs, side by side

Fifteen constructs, each with the behavioural consequence and how to see it. Full text in
`receipts/chat-template-audit.json` → `named_constructs`. Ours is byte-identical to official
3.8-27B, so it is not a separate column.

| # | construct | official 3.8-27B = **ours** | unsloth 3.8 | froggeric v22 | official 3.8 **flagship** | official 3.6 | verdict |
|---|---|---|---|---|---|---|---|
| K1 | `add_generation_prompt` tail: `'<think>\n'`, or `'<think>\n\n</think>\n\n'` when `enable_thinking is false` | opens a think block | same | same | always `'<think>\n'`; no off branch | same | identical; **not a defect** |
| K2 | `reasoning_effort\|default('xhigh')` → 42-token directive in the system block | injected on every thinking request | same | same | same, unconditionally | construct absent | generational change, not a defect |
| K3 | `if resolved_reasoning_effort not in ('xhigh','medium','low') → raise_exception` | **rejects `high`, `minimal`, `max`** | aliases `high`→`xhigh` | aliases `high`, coerces the rest | rejects, and not gated on thinking | ignores the kwarg | **REAL, LIVE** |
| K4 | `if preserve_thinking is undefined or is true or loop.index0 > last_query_index` | preserves history reasoning, **and emits the wrapper even when empty** | same | preserves, gated on `and reasoning_content` | same as ours | **strips** by default | 3.8 default is the better one; empty wrapper is a real defect |
| K5 | in-content `</think>` extractor | **removed** | removed | restored and widened | removed | present | 3.8 regression vs 3.6 |
| K6 | `for … in tool_call.arguments\|items` | `TypeError` on a JSON-string argument | raises a legible message | dumps the raw string, **no `<parameter=>` tags** | same as ours | same as ours | real, **not reachable via vLLM**; froggeric's fix is worse here |
| K7 | `<tool_call>\n<function=N>\n<parameter=K>\nV\n</parameter>…` wire format | canonical | identical | identical, but `\n\n` between parallel calls | identical | identical | **ours is the strictest correct renderer** |
| K8 | tools system block + `<IMPORTANT>` reminders | 4 bullets; *allows* prose before a call | byte-identical to official | +111 tokens, 7 bullets that **reverse** that instruction | byte-identical to ours | as ours minus K2 | froggeric diverges by **opinion** |
| K9 | `<tool_response>` framing | verbatim tool content | identical | injects a fabricated `SYSTEM WARNING` paragraph on heuristic "error" detection | identical | identical | froggeric diverges by **opinion** |
| K10 | `if not loop.first → raise_exception('System message must be at the beginning.')` | **raises on any 2nd system message** | merges a leading *run* | never raises | raises | raises | **REAL, LIVE** |
| K11 | `raise_exception('No user query found in messages.')` | raises | removed | removed | raises | raises | real, immaterial to us |
| K12 | `render_content` item dispatch + `raise_exception('Unexpected item type in content.')` | image/image_url/video/text | same | same | **text only** — no vision at all | same (27B rev1 also had audio) | **REAL, LIVE**; no template fixes it |
| **K15** | `if enable_thinking is false → raise_exception('Disabling thinking is not supported.')` | **ABSENT** | absent | absent | **PRESENT** | absent | a real flagship restriction; irrelevant to our model |
| K13 | MTP / draft tokens | none | none | none | none | none | not applicable |
| K14 | which artifact carries the template | both, identical | `.jinja` only | standalone `.jinja` | both, identical | both, identical | no divergence today; a live footgun |

Two constructs deserve expanding because they are where opinion is easiest to mistake for
correction.

**K1 is not a bug, and this is the claim most often repeated.** The prompt ends with an open
`<think>\n`, so the model emits reasoning text and then `</think>` with no opening tag of its
own. That is exactly what `--reasoning-parser qwen3` expects: `qwen3_config(thinking=True)` sets
`initial_state=ParserState.REASONING`, and `Qwen3Parser.is_reasoning_end_for_prompt` returns
`not self.thinking_enabled` — i.e. `False` — specifically so the prompt's trailing `</think>`
from a prior turn, and the `<tool_call>` example inlined into any tools system prompt, cannot
flip the prompt-end check and make `<think>` markers leak into `content`. The in-image source
carries a comment saying precisely that. With `enable_thinking=false` the template pre-closes
the block and the parser starts in `CONTENT` with `extract_reasoning` short-circuited. Prompt
and parser agree in both directions.

**K7 is where ours is measurably better than the alternative.** The template renders
`<parameter=K>` + `\n` + value + `\n` + `</parameter>`; the pinned image's
`_trim_wrapping_newlines` removes exactly one newline from each side. That is the correct
inverse, so a value whose data legitimately begins with indentation or ends with a newline
survives. vllm#48753 reports a `strip()` regression that destroyed this; the pinned build does
not have it, and `c31` below proves the round trip is lossless in our image.

## 6. The two-hour window and the flagship that explain the complaint

The loudest claim about "official 3.8" is froggeric's headline: *"Official 3.8 throws a fatal
runtime exception if you pass `enable_thinking=false`."* It is **true**, and it is about neither
the template nor the model we publish. There are two separate origins, both dated and both
measured.

### 6a. The flagship really does forbid thinking-off

The 3.8 generation launched with `Qwen/Qwen3.8-2.4T-A95B` on **2026-08-12**, one day before the
27B. Its template (`40ce34a5…`, 7495 bytes, copied byte-for-byte by `RedHatAI/Qwen3.8-2.4T-A95B`
and `-FP8`) contains a construct that exists in **no other template audited** — K15:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- raise_exception('Disabling thinking is not supported.') }}
{%- endif %}
```

Measured: `c07_thinking_off` and `c14_effort_none_via_vllm` both return **400
`Disabling thinking is not supported.`** on the flagship and render at **19 tokens** on ours.
The flagship has no non-reasoning mode at all through an OpenAI-compatible server, and — being
text-only — also 400s on every image case with `Unexpected item type in content.`

froggeric's own v22 changelog is dated 2026-08-13 and says it *"Added full support for the new
Qwen 3.8 model family (`Qwen3.8-2.4T-A95B`)"*. The claim is about the flagship. It does not
transfer to `Qwen3.8-27B`.

### 6b. The 27B's first template revision had its own version of the same trap

Diffing the first published 27B template (`72a217afab`, sha256 `49d20d09…`) against the one we
ship (`412f8b6bd7`, `c3cf9e34…`) — 1h53m apart on 2026-08-13 — shows Qwen changing six things:

1. **audio support removed.** rev1 emitted `<|audio_start|><|audio_pad|><|audio_end|>`. Correct
   removal: `config.json` carries `text_config` and `vision_config` and no `audio_config`,
   though the three audio tokens do exist in the vocab.
2. **`reasoning_effort` default inverted.** rev1 injected nothing unless asked; rev2 defaults to
   `xhigh` and injects 42 tokens on every thinking request. Measured: `c01_plain` is 17 tokens
   under rev1 and 59 under rev2.
3. **the accepted value set flipped on one member.** rev1 accepted `low|high|xhigh` and rejected
   `medium`; rev2 accepts `xhigh|medium|low` and rejects `high`. An OpenAI-canonical
   `reasoning_effort: "high"` worked at 08:23 and became a hard error at 10:16 — and `medium`,
   which our own suites do not use but many clients do, went the other way.
4. **validation moved inside the `enable_thinking` guard.**
5. **`preserve_thinking` default flipped** from strip to preserve (K4).
6. **the in-content `</think>` extractor was deleted** (K5), and the `arguments != ''` guard
   was added (K6).

Item 4 matters because rev1 validated `reasoning_effort` *unconditionally*. vLLM's canonical
thinking-off shape is `reasoning_effort: "none"`, which `protocol.py:549-550` turns into
`enable_thinking=False` while still forwarding `reasoning_effort="none"`. Against rev1 that hit
`raise_exception('Unknown reasoning_effort: none.')`. Measured: `c14_effort_none_via_vllm`
returns **400 under rev1** and renders cleanly at **19 tokens under ours**, byte-identical to
froggeric's own output for the same case.

So for roughly two hours on 2026-08-13, and permanently on the flagship, the complaint was
correct. Against the bytes in our four repos it is not.

## 7. Rendered-output matrix

34 fixed cases × 7 templates. Cell = prompt tokens under the model's own tokenizer;
**400** = the template raised, which vLLM surfaces as an HTTP 400. Full rendered strings and
their sha256 in `receipts/chat-templates/renders.json`.

| case | **ours** = official 3.8-27B | unsloth 3.8 | froggeric v22 | official 3.8 flagship | official 3.8-27B rev1 | official 3.6 | vLLM qwen3coder fixture |
|---|---|---|---|---|---|---|---|
| `c01_plain` | 59 | 59 | 59 | 59 | 17 | 17 | 15 |
| `c02_plain_nogen` | 54 | 54 | 54 | 54 | 12 | 12 | 12 |
| `c03_system` | 64 | 64 | 64 | 64 | 26 | 26 | 24 |
| `c04_multiturn_no_reasoning` | 72 | 72 | 68 | 72 | 26 | 26 | 24 |
| `c05_multiturn_with_reasoning_content` | 78 | 78 | 78 | 78 | 26 | 26 | 24 |
| `c06_multiturn_inline_think_in_content` | 82 | 82 | 78 | 82 | 26 | 26 | 34 |
| `c07_thinking_off` | 19 | 19 | 19 | **400** | 19 | 19 | 15 |
| `c08_thinking_on_explicit` | 59 | 59 | 59 | 59 | 17 | 17 | 15 |
| `c09_effort_low` | 47 | 47 | 47 | 47 | 47 | 17 | 15 |
| `c10_effort_medium` | 17 | 17 | 17 | 17 | **400** | 17 | 15 |
| `c11_effort_xhigh` | 59 | 59 | 59 | 59 | 59 | 17 | 15 |
| `c12_effort_high_OPENAI` | **400** | 59 | 59 | **400** | 17 | 17 | 15 |
| `c13_effort_minimal_OPENAI` | **400** | **400** | 59 | **400** | **400** | 17 | 15 |
| `c14_effort_none_via_vllm` | 19 | 19 | 19 | **400** | **400** | 19 | 15 |
| `c15_tools_defined_only` | 336 | 336 | 447 | 336 | 298 | 298 | 352 |
| `c16_tools_sys_and_call_dictargs` | 404 | 404 | 511 | 404 | 366 | 366 | 397 |
| `c17_tools_call_STRINGargs` | **400** | **400** | 494 | **400** | **400** | **400** | **400** |
| `c18_tools_two_parallel_calls` | 441 | 441 | 548 | 441 | 403 | 403 | 454 |
| `c19_tools_text_then_call` | 406 | 406 | 513 | 406 | 368 | 368 | 419 |
| `c20_tools_call_with_reasoning` | 405 | 405 | 516 | 405 | 367 | 367 | 412 |
| `c21_tool_error_result` | 460 | 460 | 621 | 460 | 422 | 422 | 470 |
| `c31_tool_args_whitespace` | 467 | 467 | 574 | 467 | 429 | 429 | 480 |
| `c32_tool_args_xmlish_value` | 388 | 388 | 495 | 388 | 350 | 350 | 401 |
| `c22_image_block` | 61 | 61 | 61 | **400** | 19 | 19 | **400** |
| `c23_two_images_multiturn` | 81 | 81 | 77 | **400** | 35 | 35 | **400** |
| `c24_image_vision_id` | 66 | 66 | 66 | **400** | 24 | 24 | **400** |
| `c33_tool_result_tool_reference_item` | **400** | **400** | **400** | **400** | **400** | **400** | 424 |
| `c34_unknown_content_item_type` | **400** | **400** | **400** | **400** | **400** | **400** | **400** |
| `c25_two_system_messages` | **400** | 61 | 65 | **400** | **400** | **400** | 25 |
| `c26_mid_conversation_system` | **400** | **400** | 79 | **400** | **400** | **400** | 39 |
| `c27_developer_role` | **400** | 57 | 57 | **400** | **400** | **400** | 17 |
| `c28_no_user_message` | **400** | 51 | 51 | **400** | **400** | **400** | 11 |
| `c29_preserve_thinking_false` | 68 | 68 | 68 | 68 | 26 | 26 | 24 |
| `c30_empty_system_string` | 53 | 53 | 53 | 53 | 16 | 16 | 14 |

Reading it:

- **Ours and froggeric are byte-identical in 14 of the 34 cases**, matching sha256 for sha256:
  `c01`, `c02`, `c03`, `c05`, `c07`, `c08`, `c09`, `c10`, `c11`, `c14`, `c22`, `c24`, `c29`,
  `c30`. They render divergently in 11, and in the remaining 9 at least one of them 400s. Every
  divergent case involves either tools or an assistant history turn carrying no
  `reasoning_content`.
- **`c27_developer_role` is unreachable through vLLM.** The template raises `Unexpected message
  role.`, but `_detect_developer_role_support` is False for it, so `renderers/hf.py:722-726`
  rewrites `developer`→`system` and consolidates *before* rendering (vllm#43590, merged
  2026-06-03). `c25_two_system_messages` is *not* saved by that path, because the consolidation
  only runs when a `developer` message is present.
- **The vLLM Coder fixture is not a substitute template.** It 400s on every image case
  (`TypeError: can only concatenate str (not "list") to str`) — it has no multimodal handling at
  all. Anyone who did point `--chat-template` at it for a vision model would find out
  immediately.
- The 42-token K2 delta is visible as `c01` 59 vs `c10` 17, and the 111-token K8 delta as `c15`
  336 vs 447.
- **The 3.8 flagship never renders *differently* from ours — it only refuses more.** Across all
  34 cases there are **zero** cases where both render and the bytes differ; 20 are
  byte-identical and the remaining 14 are cases where at least one 400s. Qwen kept one
  tool-call and history format across the whole 3.8 generation, which is why K7/K8/K9 match
  exactly.
- **The same is true of unsloth's template: 25 byte-identical, zero divergent renders.** It is
  strictly more permissive than ours (K3 `high`, K10 leading system run, K11), and identical
  everywhere it does not 400. That is what makes it the only opinion-free alternative.

### The empty-`<think>` diff, verbatim

`c06_multiturn_inline_think_in_content`, ours minus froggeric:

```diff
 <|im_start|>assistant
-<think>
-
-</think>
-
 <think>
```

Ours renders two think blocks, the first empty, for an assistant turn whose reasoning is inlined
in `content`. Cost: 4 tokens per affected history turn (`c04` 72 vs 68). This is K4 + K5
together, and both halves were introduced in that same `412f8b6bd7` revision.

## 8. Round-trip through the real `qwen3_coder` argument converter

Render an assistant tool call, then feed the rendered text back through
`vllm.parser.qwen3._qwen3_arg_converter` from the pinned image and compare to what went in.
A template that renders a call the parser cannot read back is broken for multi-turn tool use no
matter how good the prompt looks.

| case | ours / official 3.8 | unsloth 3.8 | froggeric v22 | vLLM fixture |
|---|---|---|---|---|
| `c16` dict args | lossless | lossless | lossless | lossless |
| `c18` two parallel calls | lossless | lossless | lossless | lossless |
| `c19` text then call | lossless | lossless | lossless | lossless |
| `c20` call with reasoning | lossless | lossless | lossless | lossless |
| `c17` **JSON-string args** | render 400 | render 400 (legible) | **`{}` — every argument silently dropped** | render 400 |
| `c31` **whitespace-bearing args** | **lossless** | **lossless** | lossy (`False` for `false`) | lossy (`False` for `false`) |
| `c32` value containing `</parameter>` | lossy | lossy | lossy | lossy |


**Now exercised live, not only through the converter (2026-08-16).** The template's `# Tools` block
round-trips against a real served engine: single, multi-turn, whitespace-bearing and genuinely
parallel calls all come back through the `qwen3_coder` parser without loss, over two passes that agree
byte-for-byte modulo random `tool_call` ids
([`tool-calls-e2e.json`](../receipts/tool-calls-e2e.json), harness
[`tools/tool_calls_e2e.py`](../tools/tool_calls_e2e.py)). Eight cases, **zero structural defects**. One
case does not return 200, and it is a conformance fact rather than a defect: `c16`'s dict-args shape is
legal for the *template* but not on the *wire*, where `function.arguments` must be a string - the
server refuses it with a 400 that names that field, which is a clean refusal rather than a silent
mangle. The suite classifies 5xx, unexplained refusals and acceptance-with-loss as blocking, and that
refusal as conformant.

Three findings here, in descending importance.

**`c17` is the answer to the question Main most wanted measured.** froggeric "fixes" the
JSON-string-arguments crash by emitting the raw JSON string inside `<function=…>` with no
`<parameter=…>` tags at all. `_qwen3_arg_converter` finds no `<parameter=` matches and returns
`{}`. Measured:

```
expected  [["get_weather", {"city": "Paris", "unit": "c"}]]
froggeric [["get_weather", {}]]        lossless=False
ours      render error: TypeError: Can only get item pairs from a mapping.
```

Under `--tool-call-parser qwen3_coder` that is silent data loss on every replayed
string-argument tool call, where ours fails loudly and unsloth fails with a legible message. And
the crash it is fixing **cannot happen on our stack anyway**, because
`_postprocess_messages` has already turned the string into a dict.

**`c31` shows ours is lossless where froggeric is not.** Given
`{"old_string": "    if x:\n        return 1\n", "count": 3, "dry_run": false}`, ours recovers
the indentation and the trailing newline exactly, and renders JSON `false`. froggeric renders
Python `False` (its `args_value | string` path stringifies booleans the Python way where the
official template's `tojson` does not), so an exact-match edit tool sees a different value. This
is precisely the failure class vllm#48753 is about, and ours is on the right side of it.

**`c32` is a format-level limitation shared by all seven templates, ours included.** The Qwen XML
tool format has no escaping, so a parameter value containing a literal `</parameter>` splits
into extra parameters: `{"text": "</parameter>\n<parameter=injected>\nevil\n"}` comes back as
`{"text": "", "injected": "evil\n"}`. This is a property of the wire format Qwen trained on, not
of any template, and no audited template mitigates it. Worth knowing; not a reason to change
ours.

## 9. Prompt growth and prefix-cache stability

`tools/chat_template_prefix.py` measures the first token index at which the *re-rendered* history
stops matching the token stream the engine actually generated. vLLM's prefix cache is keyed on
block hashes from the start of the prompt, so that index is exactly where cached blocks stop
being reusable.

At 10 completed turns (526 generated prefix tokens, block size 16):

| template | client echoes `reasoning_content` | client drops it | client inlines `<think>` in `content` |
|---|---|---|---|
| **ours** / official 3.8 | div@526 — **full prefix reuse** | div@498, reuse 0.9468 | div@498, reuse 0.9468 |
| froggeric v22 | div@526 — full prefix reuse | div@497, reuse 0.9449 | div@526 — **full prefix reuse** |
| official 3.6 | div@305, reuse 0.9132 | div@305, reuse 0.9132 | div@305, reuse 0.9132 |

This settles the "mutated past turns destroy the prefix cache" complaint precisely:

- **It was real for 3.6**, which never reaches full reuse under any client behaviour, because
  its default is to strip prior reasoning — so the last assistant turn's think block is present
  when generated and gone when re-rendered.
- **Qwen already fixed it in 3.8** by flipping `preserve_thinking` to default true. Ours reaches
  **100% prefix reuse** when the client echoes `reasoning_content` back, which vLLM both emits
  and accepts.
- **froggeric's remaining advantage is one narrow case**: a client that inlines `<think>` into
  `content`. In the other two it matches ours or is one token worse.
- No template reaches 100% for a client that discards reasoning entirely; the generated prefix
  contained reasoning that no longer exists. The loss is bounded to the tail — 94.7% of the
  prefix is still reusable at depth 10, and it improves with conversation length.

Prompt growth, same harness, 20 completed turns: ours 759 tokens without `reasoning_content` and
989 with; froggeric 679 / 989; official 3.6 637 either way.

## 10. Upstream issues: numbers, dates, fix status

Ten named claims were searched across `vllm-project/vllm`, `QwenLM/Qwen3`, `QwenLM/Qwen3.6`,
`huggingface/transformers` and `ggml-org/llama.cpp` (note `QwenLM/Qwen3.5` 301-redirects and
`QwenLM/Qwen3.8` 404s, so 3.8 bugs land in `QwenLM/Qwen3`). Every verdict was then checked
against the code installed in the pinned image rather than trusted from issue text. Full survey:
`receipts/chat-templates/raw/issue-hunt.json` and the receipt's `upstream_issue_survey`.

| # | claim | strongest issue | created | state / fix | verdict for us |
|---|---|---|---|---|---|
| 1 | `reasoning_effort: "high"` rejected by the template | [maximhq/bifrost#6193](https://github.com/maximhq/bifrost/issues/6193) | 2026-08-15 | open, no fix anywhere | **real, live** |
| 2 | JSON-string tool `arguments` → `Can only get item pairs from a mapping` | [QwenLM/Qwen3#1894](https://github.com/QwenLM/Qwen3/issues/1894) | 2026-08-14 | open, filed against our exact revision `1d4bf0f2` | real in the template, **not reachable via vLLM** |
| 3 | open `<think>` in the prompt breaks `--reasoning-parser qwen3` | [vllm#51679](https://github.com/vllm-project/vllm/issues/51679) | 2026-08-10 | open, but **retracted in the body** | **folklore** |
| 4 | empty `<think>` in history / cache invalidation | [QwenLM/Qwen3.6#131](https://github.com/QwenLM/Qwen3.6/issues/131) | 2026-04-09 | closed 2026-04-22, proposed guard **never applied** | **real, live** |
| 5 | multiple / mid-conversation system messages → 400 | [vllm#41114](https://github.com/vllm-project/vllm/issues/41114) | 2026-04-28 | open; general fix [#44505](https://github.com/vllm-project/vllm/pull/44505) **closed unmerged** 2026-06-05 | **real, live** |
| 6 | vLLM docs recommend the Coder template for non-Coder Qwen3.x | none exists; closest [vllm#38885](https://github.com/vllm-project/vllm/issues/38885) | 2026-04-03 | closed stale 2026-08-02, no code change | **folklore** |
| 7 | `qwen3_coder` vs `qwen3_xml` rename/deprecation | [vllm#25028](https://github.com/vllm-project/vllm/pull/25028) | 2025-09-23 | merged; both names registered to one class | real, **already neutral** |
| 8 | `chat_template.jinja` vs `tokenizer_config.json` precedence | no issue in any target repo | — | documented intentional behaviour | **folklore as a bug**, but see §3 |
| 9 | `_consolidate_system_messages` / developer→system | [vllm#43590](https://github.com/vllm-project/vllm/pull/43590) | 2026-05-25 | **merged 2026-06-03** (`27f1d34a`), 68 days before our build | real, **fixed** — but gated on a `developer` role, so claim 5 survives |
| 10 | multimodal / unenumerated content item types | [vllm#52489](https://github.com/vllm-project/vllm/issues/52489) | 2026-08-16 | open, filed **6 days after** our image was built | **real, live** |

Supporting issues worth naming:

- [vllm#48753](https://github.com/vllm-project/vllm/issues/48753) (open, 2026-07-15) — the
  `strip()` regression in `_qwen3_arg_converter` that corrupts whitespace-bearing tool
  arguments. **Not present in our build**: the installed `vllm/parser/qwen3.py` uses
  `_trim_wrapping_newlines`, and `c31` proves the round trip is lossless.
- [vllm#47761](https://github.com/vllm-project/vllm/issues/47761) (open, 2026-07-06) and its fix
  PR [vllm#48922](https://github.com/vllm-project/vllm/pull/48922) (**open**) — the `json.loads`
  in `_postprocess_messages` is unguarded, so a malformed `arguments` string in replayed history
  raises instead of 400ing cleanly, and the conversation is unrecoverable because the client
  replays it every turn. **Live in our build**, and it is the residual risk created by the very
  normalisation that makes claim 2 unreachable.
- [vllm#43401](https://github.com/vllm-project/vllm/pull/43401) (merged 2026-05-27) — the PR that
  maps `reasoning_effort` to `enable_thinking` and forwards `reasoning_effort` into
  `chat_template_kwargs`. Present in our build; it is what makes claim 1 reachable at all.
- [vllm#44793](https://github.com/vllm-project/vllm/issues/44793) (closed 2026-06-08) — a
  template `ValueError` used to deadlock the whole `/v1/*` surface. Closed before our build.
  Worth knowing that the three live 400s above are *only* 400s.

**Claim 3 deserves quoting, because it is the loudest one and it does not survive its own
issue.** vllm#51679's body now opens:

> **Edited.** The original body claimed that `--reasoning-parser qwen3` leaves
> `reasoning_content` empty. That was wrong — it works, and it is a usable workaround.

The surviving bug is real but needs the *opposite* of our configuration: it reproduces with
`--tool-call-parser qwen3_xml` and **no reasoning parser**, where the tool parser consumes
`</think>` and the reasoning silently concatenates into `content` with its only boundary marker
removed. We pass `--reasoning-parser qwen3`. The same issue body also confirms the template's
design is understood rather than disputed: *"The Qwen3 chat template opens `<think>\n` in the
generation prompt, so every reply begins inside a think block."*

### Adjudicating the froggeric README, claim by claim

The repo `froggeric/Qwen-Fixed-Chat-Templates` exists solely to fix Qwen chat templates, so its
README is the best available statement of the complaint. Its diff against official is **large**
— 19,262 bytes against 8,952 — but only about a third of it is correction. Full text of each
adjudication in `receipts/chat-template-audit.json` → `froggeric_claim_adjudication`.

| README claim | verdict | evidence |
|---|---|---|
| Official 3.8 throws a fatal exception on `enable_thinking=false` | **real, but about the flagship** (and transiently the 27B's rev1) | K15; `c07`/`c14` 400 on `40ce34a5…`, render at 19 tokens on ours (§6) |
| Official 3.8 injects a duplicate blank `<think></think>` in history | **real, live, measured** | K4+K5; `c06` 82 vs 78 tokens, two think blocks in the diff |
| Official 3.8 crashes on OpenAI JSON-string tool `arguments` | real in the template, **unreachable via vLLM**, and froggeric's fix is *worse* here | K6; `c17` round-trips to `{}` under froggeric — every argument dropped |
| Full `reasoning_effort` support including `high` | **real and live; the one defect worth acting on** | K3; `c12` 400s on ours, renders on both alternatives |
| Mid-conversation system prompts crash the template | **real and live** | K10; `c25`/`c26` 400 on ours |
| Mutated past turns destroy the prefix cache; this template guarantees 100% | real for **3.6**, already fixed by Qwen in 3.8; froggeric wins one narrow case | §9: ours reaches 1.0 reuse when the client echoes `reasoning_content` |
| Deep Jinja nesting costs llama.cpp 80%; Python filters crash C++ engines | **unsubstantiated for us, out of scope** | minijinja/minja properties; we render with CPython jinja2 3.1.6 |
| Use `qwen3_xml` on current vLLM, `qwen3_coder` only on older builds | **moot in our build** | both names → one `Qwen3EngineToolParser` (§2) |
| vLLM setup: replace the `chat_template` string in `tokenizer_config.json` | **wrong instruction for our repos** | K14; `chat_template.jinja` wins, proved with sentinels (§3) |
| Anthropic `message.thinking` payloads are rejected | real in the template, **not reachable through our endpoint** | vLLM writes both `reasoning` and `reasoning_content` onto the message (§2) |
| Two-tier error escalation, payload truncation, `<\|think_off\|>`, `tool_call_format="json"` | **features, not fixes** | K9/K8; they inject text the tool never returned and add a second wire format |

Corrections in froggeric's diff: the `high` alias (K3), the in-content `</think>` extractor (K5),
the empty-think guard (K4), tolerating multiple system messages (K10), dropping the
no-user-query raise (K11). Opinion: the rewritten `<IMPORTANT>` block (K8), the fabricated
tool-error warnings (K9), `<|think_off|>` inline tags, `tool_call_format="json"`, argument
truncation. Regressions under our flags: K6 and K7.

## 11. Claims I could not substantiate

Listed explicitly rather than dropped.

1. **That `--reasoning-parser qwen3` leaves `reasoning_content` empty because the template opens
   `<think>` in the prompt.** Retracted by the reporter in vllm#51679's body; contradicted by
   reading `Qwen3Parser.is_reasoning_end_for_prompt` in the pinned image.
2. **That vLLM docs recommend `--chat-template examples/tool_chat_template_qwen3coder.jinja` for
   non-Coder Qwen3.x models.** No such issue in the five repos searched;
   `docs/features/tool_calling.md` lists only `--tool-call-parser qwen3_xml` with no
   `--chat-template` flag, and the image ships no `examples/` directory to point at.
3. **That deep Jinja nesting costs llama.cpp 80% throughput.** Asserted in froggeric's README
   with no measurement; no upstream issue quantifies it; irrelevant to a CPython jinja2
   renderer. Not reproduced here — we have no llama.cpp serving path in scope.
4. **That Python-only filters crash "the engine".** `|items`, `[::-1]`, `.rstrip('\n')` and
   `loop.previtem` all evaluate fine in the pinned image's jinja2 3.1.6. This is a real
   constraint for minijinja/minja consumers and a non-issue for us.
5. **That the pinned image carries a built-in Qwen template override.** It does not: no fallback
   for `model_type qwen3_5`, no `examples/`, and the only `qwen3*.jinja` files present are Rust
   unit-test fixtures.
6. **That the template is implicated in the reported garbled-output incident.** For the prompt
   shapes involved, the mounted template and ours render byte-identical strings.
7. **A GitHub issue in any of the five target repos reporting the `Unexpected reasoning effort
   high` rejection.** The only reports naming that exact string live in downstream integrator
   repos (maximhq/bifrost#6193, osaurus-ai/osaurus#2395). The defect is real and measured here
   regardless — the absence of an upstream issue is a gap in the trackers, not evidence against
   it.
8. **That "official 3.8" forbids `enable_thinking=false`, as a statement about
   `Qwen3.8-27B`.** The construct is real but belongs to `Qwen/Qwen3.8-2.4T-A95B`, and briefly
   to the 27B's superseded rev1 in a different form. Measured both ways in §6. Cited as a
   property of the 27B it is false.
9. **That vLLM's Rust chat crate renders our template.** The pinned image renders through
   CPython `jinja2` via `tokenizer.apply_chat_template`; the Rust `qwen3.jinja`/`qwen35.jinja`
   files are `include_str!` fixtures inside `#[cfg(test)] mod tests`. We did not attempt to
   determine whether the Rust renderer is reachable by any configuration, only that nothing in
   our serving path uses it.

## 12. Recommendation and the cost of changing

### Do not change `chat_template.jinja` in any of the four repos.

Reasons, in order:

1. **Byte-identity with `Qwen/Qwen3.8-27B` is itself the feature.** One sha256 lets a reader
   verify our repos against upstream, and it is what makes our published numbers comparable to
   anyone else's run of the same checkpoint.
2. **Every live defect is a 400 on a request shape none of our published suites emit**, and each
   has a zero-cost operator-side workaround.
3. **The only opinion-free alternative is unsloth's** (`12827f24`), which fixes K3 and half of
   K10. It does not fix K12, and adopting it costs a full re-measurement while breaking
   byte-identity.
4. **froggeric-v22 is not a candidate.** Measured against our own tool parser it silently drops
   all arguments of a string-argument tool call (K6/`c17`), renders Python `False` where JSON
   `false` is required (K7/`c31`), injects text the tool never returned into `<tool_response>`
   (K9/`c21`), and costs +111 tokens on every tools request (K8/`c15`). Its genuinely good ideas
   — the in-content `</think>` extractor, the `high` alias, tolerating multiple system messages —
   arrive bundled with prompt engineering that would change what our measurements measure.

Its diff against official is **large, and only about a third of it is correction** — 19,262
bytes against 8,952. The claim-by-claim breakdown is in
[§10](#adjudicating-the-froggeric-readme-claim-by-claim).

### Do this instead — documentation and upstream, all zero re-measurement

| action | cost |
|---|---|
| Document the `reasoning_effort` contract on the four model cards: send `xhigh`, `medium`, `low` or `none`; `high`, `minimal` and `max` are rejected by the upstream template with HTTP 400. Point operators whose client hard-codes `high` at `--default-chat-template-kwargs.reasoning_effort=<value>`. | docs only |
| Document that clients must merge multiple `system` messages into one leading message, and that a `developer` role is safe because vLLM folds and consolidates it (vllm#43590). | docs only |
| Document that echoing `reasoning_content` back on assistant history turns is what buys the 100% prefix-cache hit, with the numbers from §9. | docs only |
| Document the artifact-precedence footgun from §3, and that overriding via `tokenizer_config.json` is a no-op — use `--chat-template <file>`. | docs only |
| Validate tool-call arguments before replaying assistant history and reject values containing literal `</parameter>` / `<parameter=` delimiters. The Qwen XML wire format has no escaping, and all audited templates misparse such values (§8 `c32`). Treat tool outputs and replayed calls as untrusted data. | client validation |
| File the K3 rejection upstream with Qwen (alias `high`→`xhigh`, tolerate `minimal`/`max`, exactly as unsloth already does). | none to us |

### If we ever did change it

**Would need re-running:**

- **The 40-task deterministic retention suite.** Its `tool_call` family embeds
  a pseudo-tool spec in user text and checks plain JSON; it does not exercise the
  OpenAI `tools` field.
- **The live tool-call suite.** `tools/tool_calls_e2e.py` now covers eight served
  single, parallel, whitespace-bearing, and multi-turn cases through the real
  `qwen3_coder` parser (`receipts/tool-calls-e2e.json`). This closes the earlier
  `# Tools` coverage gap and must be rerun after any template edit.
- **The 70-item MMLU-Pro capability matrix** (`receipts/public-capability-*.json`). Same
  mechanism: same `chat_template_kwargs`, temperature 0, five-shot prefixes rendered through the
  template.
- **Every throughput and context-capacity number**, because prompt token counts move. Measured
  deltas for a plain single turn: ours 59, unsloth 59, froggeric 59, official 3.6 17. For a
  tools request: 336 / 336 / 447.

**Would not need re-running:**

- **The KLD / fidelity suite.** It replays a pre-tokenised corpus; there is no
  `apply_chat_template` call and no `messages` list in that path. No template change can move a
  KLD, a top-1 agreement, or a tail statistic. Stated explicitly so it cannot be left
  ambiguous.
- **Vision placeholder accounting**, unless `render_content` itself changed: every audited
  3.8-generation template emits exactly `<|vision_start|><|image_pad|><|vision_end|>` per image.

**The qualified serving digest would survive.** `receipts/public-capability-plan.json`'s
`runtime` block pins the image digest, rootfs provenance and file manifest, `ggrun`, the exl3 and
`qwen3_5`/`qwen3_5_mtp` module digests, torch, CUDA, GPU and driver — none of which contain the
model repo's chat template. What *would* need regenerating and republishing is the per-repo
release evidence: `chat_template.jinja`'s sha256 is pinned in
`receipts/k5k6-build-receipt.json`, `receipts/hydrated-build-receipt.json`,
`receipts/context-build-receipt.json` and the matching `*-SHA256SUMS` files.

## 13. Reproducing this

No GPU required; the whole audit runs on CPU in seconds once the templates are fetched.

```bash
cd ~/research
bash tools/fetch_chat_templates.sh                      # HF templates + revisions + digests
python3 tools/gguf_chat_template.py \
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q4_K_M.gguf" \
  receipts/chat-templates/raw/unsloth__Qwen3.8-27B-GGUF__Q4_K_M.chat_template.jinja

# stage the templates and harnesses where ggrun.sh binds them as /work
W=/var/tmp/work/chat-template-audit
mkdir -p $W/raw && cp receipts/chat-templates/raw/*.jinja $W/raw/ && \
  cp tools/chat_template_{matrix,roundtrip,prefix}.py $W/

# note the /work/... paths: inside proot, $W is bound to /work/chat-template-audit
CUDA_VISIBLE_DEVICES="" bash tools/ggrun.sh python /work/chat-template-audit/chat_template_matrix.py \
  > $W/matrix.json
CUDA_VISIBLE_DEVICES="" bash tools/ggrun.sh python /work/chat-template-audit/chat_template_roundtrip.py \
  /work/chat-template-audit/matrix.json > $W/roundtrip.json
CUDA_VISIBLE_DEVICES="" bash tools/ggrun.sh python /work/chat-template-audit/chat_template_prefix.py \
  > $W/prefix.json

python3 tools/chat_template_receipt.py                  # -> receipts/chat-template-audit.json
```

The artifact-precedence result in §3 is reproduced with:

```bash
P=$W/precedence; mkdir -p $P
cp /var/tmp/models/Qwen3.8-27B/tokenizer.json $P/
python3 -c "import json; d=json.load(open('/var/tmp/models/Qwen3.8-27B/tokenizer_config.json')); \
  d['chat_template']='SENTINEL_FROM_TOKENIZER_CONFIG'; json.dump(d, open('$P/tokenizer_config.json','w'))"
printf 'SENTINEL_FROM_CHAT_TEMPLATE_JINJA' > $P/chat_template.jinja
CUDA_VISIBLE_DEVICES="" bash tools/ggrun.sh python -c "
from transformers import AutoTokenizer
print(AutoTokenizer.from_pretrained('/work/chat-template-audit/precedence').chat_template)"
```

`tools/chat_template_findings.py` holds the judgement half (named constructs, claim
adjudication, verdict, recommendation) separately from the mechanical half, so digests and
measurements can be regenerated without touching the analysis.
