#!/usr/bin/env python3
"""The curated half of the chat-template audit: named constructs, verdict,
recommendation and re-measurement cost.  Kept separate from
chat_template_receipt.py so the mechanical digest/measurement half stays
regenerable without touching the judgement half.

Merged into receipts/chat-template-audit.json by chat_template_receipt.py.
"""

# Every entry names the exact Jinja construct, what each template does with it,
# and how to observe the difference.  "ours" is byte-identical to official-3.8
# so it is not listed as a separate column.
CONSTRUCTS = [
    {
        "id": "K1",
        "construct": "add_generation_prompt tail: `'<|im_start|>assistant\\n'` then "
                     "`'<think>\\n'` (thinking) or `'<think>\\n\\n</think>\\n\\n'` "
                     "(enable_thinking is false)",
        "official_3_8": "opens a think block: prompt ends `<think>\\n`",
        "unsloth_3_8": "identical",
        "froggeric_v22": "identical, but keyed on its own `ns_state.thinking` "
                         "(which `<|think_off|>` in any system/user message can flip)",
        "official_3_6": "identical",
        "official_3_8_flagship": "ALWAYS emits `<think>\n`; the "
                                 "`enable_thinking is false` branch does not "
                                 "exist because the flagship raises "
                                 "`Disabling thinking is not supported.` earlier",
        "behavioural_consequence":
            "The model never emits an opening `<think>`; it emits reasoning text "
            "and then `</think>`. This is correct for `--reasoning-parser qwen3`: "
            "`qwen3_config(thinking=True)` sets `initial_state=ParserState.REASONING` "
            "and `Qwen3Parser.is_reasoning_end_for_prompt` returns "
            "`not self.thinking_enabled` (False), so the per-delta state machine "
            "runs from REASONING and `</think>` fires REASONING_END. With "
            "`enable_thinking=false` the prompt pre-closes the block and the "
            "parser starts in CONTENT with `extract_reasoning` short-circuited. "
            "Prompt and parser agree in both directions.",
        "observable": "c01/c07 rendered tails; all four 3.8-generation templates "
                      "byte-identical (59 and 19 tokens).",
        "verdict": "identical across every 3.8 template; not a defect",
    },
    {
        "id": "K2",
        "construct": "unconditional reasoning-effort system message: "
                     "`reasoning_effort|default('xhigh')` -> a 42-token directive "
                     "line prepended to (or synthesised as) the system block",
        "official_3_8": "injects 42 tokens on EVERY thinking request; `medium` is "
                        "the only value that injects nothing",
        "unsloth_3_8": "identical, plus `high` aliased to `xhigh`",
        "froggeric_v22": "identical text; `high` aliased to `xhigh`; anything "
                         "unrecognised silently falls back to `xhigh`",
        "official_3_6": "construct absent (no reasoning-effort block at all)",
        "behavioural_consequence":
            "Every prompt is 42 tokens longer than the 3.6 generation for the "
            "same conversation, and carries an instruction the operator did not "
            "write. This is a deliberate Qwen change introduced in the second "
            "published 3.8 template revision (412f8b6b), not a packaging error.",
        "observable": "c01_plain: official-3.8 59 tokens vs official-3.8-rev1 17 "
                      "and official-3.6 17; c10_effort_medium: 17.",
        "verdict": "identical across every shipped 3.8 template; a generational "
                   "change, not a defect",
    },
    {
        "id": "K3",
        "construct": "reasoning-effort validation: "
                     "`if resolved_reasoning_effort not in ('xhigh','medium','low') "
                     "-> raise_exception`",
        "official_3_8": "accepts xhigh|medium|low. HARD-FAILS on `high`, "
                        "`minimal`, `max`",
        "unsloth_3_8": "maps `high`->`xhigh` first; still hard-fails `minimal`, `max`",
        "froggeric_v22": "maps `high`->`xhigh`, silently coerces every other value "
                         "to `xhigh`; never raises",
        "official_3_6": "construct absent; the kwarg is ignored",
        "behavioural_consequence":
            "vLLM's OpenAI surface declares `reasoning_effort` as "
            "Literal['none','minimal','low','medium','high','xhigh','max'] "
            "(protocol.py:231) and `build_chat_params` forwards it verbatim into "
            "chat_template_kwargs (protocol.py:541). The template declares the "
            "variable, so `resolve_chat_template_kwargs` does not filter it out. "
            "A client sending the OpenAI-canonical `reasoning_effort: \"high\"` "
            "therefore gets HTTP 400 `Unexpected reasoning effort high.` This is "
            "the one genuine, live, reachable defect in the template we ship.",
        "observable": "c12_effort_high_OPENAI: official-3.8 raises TemplateError; "
                      "unsloth-3.8 and froggeric-v22 render 59 tokens.",
        "verdict": "REAL AND LIVE for us; fixed by the unsloth template and by "
                   "froggeric",
    },
    {
        "id": "K4",
        "construct": "history reasoning retention: "
                     "`if preserve_thinking is undefined or preserve_thinking is "
                     "true or loop.index0 > ns.last_query_index`",
        "official_3_8": "PRESERVES prior-turn reasoning by default, and emits the "
                        "`<think>...</think>` wrapper even when reasoning_content "
                        "is empty",
        "unsloth_3_8": "identical",
        "froggeric_v22": "preserves, but gated on `and reasoning_content`, so a "
                         "turn with no reasoning gets no think wrapper",
        "official_3_6": "STRIPS by default "
                        "(`preserve_thinking is defined and ... is true`)",
        "behavioural_consequence":
            "Qwen flipped the default from strip to preserve between 3.6 and 3.8. "
            "Preserving is what makes 100% prefix-cache reuse possible, because "
            "the re-rendered history then matches the tokens the engine "
            "generated. The cost is that a client which does NOT echo "
            "`reasoning_content` gets a fabricated empty `<think>\\n\\n</think>` "
            "block per prior assistant turn: 4 tokens each, and the model reads "
            "its own past turns as having deliberated about nothing.",
        "observable": "c04 official-3.8 72 tokens vs froggeric-v22 68 "
                      "(the 4-token empty block); prefix_stability shows "
                      "official-3.8 at cache_reuse_fraction 1.0 when the client "
                      "echoes reasoning_content and 0.9468 at depth 10 when it "
                      "does not, versus official-3.6 which never reaches 1.0 in "
                      "any case.",
        "verdict": "3.8 behaviour is deliberate and is the better of the two; the "
                   "empty-wrapper emission is a real cosmetic/token defect that "
                   "froggeric fixes",
    },
    {
        "id": "K5",
        "construct": "in-content `</think>` extractor "
                     "(`content.split('</think>')[0].rstrip('\\n')...`)",
        "official_3_8": "REMOVED. Only `message.reasoning_content` is read",
        "unsloth_3_8": "also removed",
        "froggeric_v22": "restored and widened (`</think>`, `</thinking>`, "
                         "`</ think>`, `</think >`, plus `message.thinking`)",
        "official_3_6": "present",
        "behavioural_consequence":
            "A client that stores the assistant turn as one string containing "
            "`<think>...</think>answer` -- which is what any client talking to a "
            "server WITHOUT `--reasoning-parser` receives -- gets two think "
            "blocks from official 3.8: an empty one the template synthesises "
            "followed by the model's real one. froggeric splits it back out. "
            "Under our flags (`--reasoning-parser qwen3` set, and vLLM aliasing "
            "input `reasoning_content` to `reasoning` and back at "
            "chat_utils.py:1832-1836) a well-behaved client round-trips "
            "`reasoning_content` and this path is not taken.",
        "observable": "c06_multiturn_inline_think_in_content: official-3.8 82 "
                      "tokens with `<think>\\n\\n</think>\\n\\n<think>\\n...`, "
                      "froggeric-v22 78 tokens with a single think block; "
                      "prefix_stability `client_inlines_think_in_content` is the "
                      "only row where froggeric reaches full prefix reuse and "
                      "official does not.",
        "verdict": "a real 3.8 regression against 3.6; only bites clients that "
                   "inline reasoning into content",
    },
    {
        "id": "K6",
        "construct": "tool-call argument rendering: "
                     "`for args_name, args_value in tool_call.arguments|items`",
        "official_3_8": "guarded only by `arguments is defined and arguments != ''`; "
                        "a non-empty JSON STRING reaches `|items` and raises "
                        "`TypeError: Can only get item pairs from a mapping.`",
        "unsloth_3_8": "type-tests first and raises a legible TemplateError telling "
                       "the caller to parse the string into an object",
        "froggeric_v22": "dumps the raw JSON string inside `<function=...>` with NO "
                         "`<parameter=...>` wrappers",
        "official_3_6": "same defect as 3.8, without even the `!= ''` guard",
        "behavioural_consequence":
            "NOT reachable through vLLM's OpenAI server in the pinned build: "
            "`_postprocess_messages` (chat_utils.py:1875-1915) json.loads()es "
            "every assistant `function.arguments` string into a dict on both the "
            "sync and async render paths before the template runs. It IS "
            "reachable via llama.cpp/LM Studio/MLX and via raw "
            "`tokenizer.apply_chat_template`. froggeric's 'fix' is worse than the "
            "crash under our flags: the rendered call carries no `<parameter=>` "
            "tags, so `_qwen3_arg_converter` recovers `{}` and the tool is "
            "replayed into history with EVERY ARGUMENT SILENTLY DROPPED.",
        "observable": "c17_tools_call_STRINGargs round-trip: official/unsloth/3.6/"
                      "vllm-fixture raise; froggeric-v22 renders 494 tokens and "
                      "the qwen3_coder arg converter returns "
                      "`[[\"get_weather\", {}]]` against an expected "
                      "`{\"city\": \"Paris\", \"unit\": \"c\"}`.",
        "verdict": "real template defect, NOT live for us; froggeric's fix is a "
                   "silent-data-loss regression under `--tool-call-parser "
                   "qwen3_coder`",
    },
    {
        "id": "K7",
        "construct": "tool-call wire format: `<tool_call>\\n<function=NAME>\\n"
                     "<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>`",
        "official_3_8": "canonical Qwen XML; one leading and one trailing newline "
                        "of markup per value",
        "unsloth_3_8": "identical",
        "froggeric_v22": "identical for mapping arguments, except that a second and "
                         "subsequent parallel call is separated by `\\n\\n` instead "
                         "of `\\n`",
        "official_3_6": "identical",
        "behavioural_consequence":
            "This is exactly the inverse of `_qwen3_arg_converter`'s "
            "`_trim_wrapping_newlines` in the pinned image, so values whose data "
            "legitimately begins or ends with whitespace survive the round trip. "
            "vllm#48753 reports a `strip()` regression that would have corrupted "
            "them; the pinned build does not have it. The format has no escaping, "
            "so a value containing a literal `</parameter>` splits into extra "
            "parameters -- identically in every template audited, ours included.",
        "observable": "c31_tool_args_whitespace round-trips lossless for "
                      "official-3.8/ours/unsloth/3.6 (indentation and trailing "
                      "newlines preserved) and LOSSY for froggeric-v22 and the "
                      "vllm fixture, which render Python `False` instead of JSON "
                      "`false`. c32_tool_args_xmlish_value is lossy for all seven "
                      "templates: `{\"text\": \"</parameter>...\"}` comes back as "
                      "`{\"text\": \"\", \"injected\": \"evil\\n\"}`.",
        "verdict": "ours is the strictest correct renderer measured; froggeric "
                   "introduces a JSON/Python boolean-casing bug",
    },
    {
        "id": "K8",
        "construct": "tools system block: `# Tools` + `<tools>` JSON lines + the "
                     "`If you choose to call a function ONLY reply in the "
                     "following format` example + `<IMPORTANT>` reminder list",
        "official_3_8": "4 reminder bullets; explicitly ALLOWS natural-language "
                        "reasoning before a call ('You may provide optional "
                        "reasoning ... BEFORE the function call, but NOT after')",
        "unsloth_3_8": "byte-identical to official",
        "froggeric_v22": "+111 tokens. Adds a `<think>\\nBrief explanation of tool "
                         "call\\n</think>` step to the example, and 7 reminder "
                         "bullets that REVERSE the official instruction: 'you MUST "
                         "output the <tool_call> block IMMEDIATELY after thinking, "
                         "with NO conversational text before it'",
        "official_3_6": "same as official 3.8 minus the reasoning-effort line",
        "behavioural_consequence":
            "This is opinion, not correction: it is prompt engineering that "
            "changes tool-calling style and costs 111 tokens on every "
            "tools-enabled request. It also contradicts the format Qwen trained "
            "on. Nothing in the wire format or the parser requires it.",
        "observable": "c15_tools_defined_only: official-3.8 336 tokens vs "
                      "froggeric-v22 447.",
        "verdict": "froggeric diverges by opinion; ours matches Qwen",
    },
    {
        "id": "K9",
        "construct": "tool-result rendering: `<|im_start|>user` + "
                     "`\\n<tool_response>\\n` + content + `\\n</tool_response>`, "
                     "with `<|im_end|>` only at the end of a run of tool messages",
        "official_3_8": "verbatim tool content, run-collapsed via "
                        "`loop.previtem`/`loop.nextitem`",
        "unsloth_3_8": "identical",
        "froggeric_v22": "same framing via explicit indexing, PLUS a fabricated "
                         "`WARNING SYSTEM WARNING: ...` paragraph injected into "
                         "the tool_response body when a heuristic decides the "
                         "result looks like an error, escalating on the second "
                         "consecutive hit",
        "official_3_6": "identical to official 3.8",
        "behavioural_consequence":
            "froggeric puts words the tool never returned inside "
            "`<tool_response>`. For a deterministic tool-call benchmark that is "
            "not a template difference, it is a different experiment. It also "
            "means tool output containing the substring `Exception:` mutates the "
            "prompt.",
        "observable": "c21_tool_error_result: official-3.8 460 tokens, "
                      "froggeric-v22 621, with two injected warning paragraphs "
                      "visible in the rendered diff.",
        "verdict": "froggeric diverges by opinion, and in a way that would "
                   "invalidate any tool-call measurement",
    },
    {
        "id": "K10",
        "construct": "system-message position guard: "
                     "`if not loop.first -> raise_exception('System message must "
                     "be at the beginning.')`",
        "official_3_8": "raises on ANY system message after index 0, including two "
                        "leading system messages",
        "unsloth_3_8": "merges a leading RUN of system/developer messages "
                       "(`sysns`), still raises for a genuinely mid-conversation one",
        "froggeric_v22": "never raises; emits every system message inline as its "
                         "own `<|im_start|>system` block",
        "official_3_6": "same as official 3.8",
        "behavioural_consequence":
            "Reachable as HTTP 400 through our endpoint. vLLM's "
            "`_consolidate_system_messages` (renderers/hf.py:465) would merge "
            "them, but hf.py:722-726 only calls it when the conversation also "
            "contains a `developer` role. Two plain `system` messages -- what "
            "LangChain-style stacks and context-compression flows emit -- never "
            "reach it and 400. The `developer` role itself is safe: vLLM rewrites "
            "it to `system` and consolidates, because "
            "`_detect_developer_role_support` is False for our template.",
        "observable": "c25_two_system_messages: official-3.8 raises, unsloth-3.8 "
                      "renders 61 tokens, froggeric-v22 65. "
                      "c27_developer_role raises in the template but is "
                      "unreachable through vLLM.",
        "verdict": "REAL AND LIVE for us, upstream fix (vllm#44505) closed "
                   "unmerged; fixed by unsloth for the leading-run case and by "
                   "froggeric for all cases",
    },
    {
        "id": "K11",
        "construct": "`ns.multi_step_tool` scan + "
                     "`raise_exception('No user query found in messages.')`",
        "official_3_8": "raises when no non-tool-response user message exists",
        "unsloth_3_8": "raise removed",
        "froggeric_v22": "raise removed; falls back to last_query_index 0 or the "
                         "last index past 50 messages",
        "official_3_6": "raises",
        "behavioural_consequence":
            "Blocks assistant-prefill and system-only completions with a 400. "
            "Narrow: our suites always send a user turn.",
        "observable": "c28_no_user_message: official-3.8 raises, the other two "
                      "render 51 tokens.",
        "verdict": "real but immaterial to us",
    },
    {
        "id": "K12",
        "construct": "`render_content` macro item dispatch and its final "
                     "`raise_exception('Unexpected item type in content.')`",
        "official_3_8": "handles image / image_url / video / text only. Emits "
                        "`<|vision_start|><|image_pad|><|vision_end|>` per image "
                        "and optional `Picture N: ` when `add_vision_id`",
        "unsloth_3_8": "identical",
        "froggeric_v22": "identical dispatch, plus a `item | string` fallback for "
                         "non-mapping items -- but the same raise for unknown "
                         "mapping types",
        "official_3_6": "identical; the FIRST published 3.8 revision (49d20d09) "
                        "additionally emitted "
                        "`<|audio_start|><|audio_pad|><|audio_end|>`, which Qwen "
                        "removed 1h53m later -- correctly, since config.json "
                        "carries text_config and vision_config but no audio_config",
        "behavioural_consequence":
            "Any content item type the macro does not enumerate -- Anthropic "
            "`tool_reference` inside a `tool_result`, OpenAI `input_audio` -- "
            "becomes a template exception. vLLM deliberately preserves such items "
            "as structured dicts (chat_utils.py:1846-1860 keeps non-text tool "
            "content 'for the chat template to expand them'), so the endpoint "
            "hands the template something it cannot render. This is vllm#52489, "
            "filed six days after our image was built.",
        "observable": "c33_tool_result_tool_reference_item and "
                      "c34_unknown_content_item_type raise "
                      "`Unexpected item type in content.` in all six Qwen-family "
                      "templates including froggeric.",
        "verdict": "REAL AND LIVE, but not a differentiator: no audited template "
                   "fixes it",
    },
    {
        "id": "K15",
        "construct": "flagship-only guard: `if enable_thinking is defined and "
                     "enable_thinking is false -> raise_exception('Disabling "
                     "thinking is not supported.')`",
        "official_3_8": "ABSENT -- thinking-off is supported",
        "unsloth_3_8": "absent",
        "froggeric_v22": "absent",
        "official_3_6": "absent",
        "official_3_8_flagship": "PRESENT (Qwen/Qwen3.8-2.4T-A95B and both "
                                 "RedHatAI copies of it)",
        "behavioural_consequence":
            "On the flagship, both `enable_thinking: false` and vLLM's canonical "
            "`reasoning_effort: \"none\"` (which protocol.py:549-550 turns into "
            "enable_thinking=False) return HTTP 400. This construct is the origin "
            "of the loudest community claim about 'official 3.8', and it does not "
            "exist in the 27B template we ship. It also means the flagship has no "
            "non-reasoning mode at all through an OpenAI-compatible server.",
        "observable": "c07_thinking_off and c14_effort_none_via_vllm: flagship "
                      "raises `Disabling thinking is not supported.`, ours renders "
                      "19 tokens.",
        "verdict": "a real flagship restriction; irrelevant to our model",
    },
    {
        "id": "K13",
        "construct": "MTP / draft-model tokens",
        "official_3_8": "none",
        "unsloth_3_8": "none",
        "froggeric_v22": "none",
        "official_3_6": "none",
        "behavioural_consequence":
            "No template in this family emits any MTP or draft-specific token. "
            "The MTP module consumes token ids from the scheduler and never sees "
            "a template. A template change cannot affect MTP acceptance except "
            "through the ordinary route of changing the prompt tokens.",
        "observable": "grep: no `mtp`, `draft` or speculative token in any of the "
                      "six templates.",
        "verdict": "not applicable",
    },
    {
        "id": "K14",
        "construct": "which artifact carries the template: `chat_template.jinja` "
                     "at repo root vs the `chat_template` key of "
                     "`tokenizer_config.json` vs GGUF KV `tokenizer.chat_template`",
        "official_3_8": "BOTH files present and byte-identical",
        "unsloth_3_8": "NVFP4 repo ships `chat_template.jinja` only (its "
                       "tokenizer_config.json is 1047 bytes with no "
                       "chat_template key); the GGUF repo ships NEITHER at root "
                       "and carries the same template in GGUF KV "
                       "`tokenizer.chat_template`",
        "froggeric_v22": "distributed as a standalone `chat_template.jinja`",
        "official_3_6": "BOTH files present and byte-identical",
        "behavioural_consequence":
            "transformers 5.15.0 gives `chat_template.jinja` PRIORITY over the "
            "tokenizer_config key (tokenization_utils_base.py:1783-1800, and "
            "proved here with sentinel strings: with both present the .jinja file "
            "wins; delete it and the key takes over). froggeric's own vLLM "
            "install instruction -- 'Replace the \"chat_template\" string in your "
            "tokenizer_config.json' -- is therefore a NO-OP against any of our "
            "four repos. Mounting it requires replacing/removing "
            "chat_template.jinja or passing `--chat-template <file>`.",
        "observable": "tools/chat_template_receipt.py records both digests per "
                      "repo; `dual_artifact_identical` is true for all nine "
                      "HF repos that ship both.",
        "verdict": "no divergence in our repos today; a live footgun for anyone "
                   "following froggeric's instructions",
    },
]

FROGGERIC_CLAIMS = [
    {
        "claim": "Official 3.8 throws a fatal runtime exception if you pass "
                 "enable_thinking=false (README 'Restored Fast Mode')",
        "verdict": "REAL, BUT ABOUT A DIFFERENT MODEL IN THE SAME GENERATION -- "
                   "AND SEPARATELY ABOUT A SUPERSEDED REVISION OF OURS",
        "evidence":
            "TRUE of the 3.8 FLAGSHIP. Qwen/Qwen3.8-2.4T-A95B "
            "(rev 207bd685a7e3696cfaff12ded7c6a7ea0f88c996, 2026-08-12, sha256 "
            "40ce34a5..., copied byte-for-byte by RedHatAI/Qwen3.8-2.4T-A95B and "
            "-FP8) contains a construct that exists in NO other template "
            "audited: `{%- if enable_thinking is defined and enable_thinking is "
            "false %}{{- raise_exception('Disabling thinking is not supported.') "
            "}}`. Measured: cases c07_thinking_off and c14_effort_none_via_vllm "
            "both raise `Disabling thinking is not supported.` on the flagship "
            "and render at 19 tokens on ours. froggeric's own changelog dates "
            "v22 to 2026-08-13 and says it 'Added full support for the new Qwen "
            "3.8 model family (Qwen3.8-2.4T-A95B)', so the claim is about the "
            "flagship, which shipped one day before the 27B. "
            "ALSO TRANSIENTLY TRUE of the 27B's first template revision "
            "(72a217afab, sha256 49d20d09..., 2026-08-13T08:23:30Z): "
            "case c14 (vLLM's canonical thinking-off shape, "
            "reasoning_effort='none' + enable_thinking=False, produced by "
            "protocol.py:549-550) raises "
            "`TemplateError: Unknown reasoning_effort: none.` because rev1 "
            "validated reasoning_effort unconditionally, outside any "
            "enable_thinking guard. Qwen fixed exactly that 1h53m later in rev "
            "412f8b6bd7 by moving the validation inside "
            "`if enable_thinking is undefined or enable_thinking is true`. "
            "Against the revision we actually ship, c07 (enable_thinking=False) "
            "and c14 both render cleanly at 19 tokens, byte-identical to "
            "froggeric's own output. So the claim is real, and it is about "
            "neither the template nor the model we publish.",
    },
    {
        "claim": "Official 3.8 injects a duplicate blank <think></think> in chat "
                 "history ('Empty Think' poisoning)",
        "verdict": "REAL, LIVE, MEASURED",
        "evidence":
            "c06: official-3.8 renders "
            "`<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<think>\\nThe "
            "user greeted me.\\n</think>\\n\\nHello!` -- two think blocks, the "
            "first empty -- 82 tokens against froggeric's 78. Root cause is two "
            "constructs, both changed in rev 412f8b6b: the in-content `</think>` "
            "extractor was deleted (K5) and the think wrapper is emitted "
            "unconditionally (K4). Only reachable when the client inlines "
            "reasoning into `content`; with `reasoning_content` echoed the two "
            "templates are byte-identical (c05).",
    },
    {
        "claim": "Official 3.8 crashes with `TypeError: Can only get item pairs "
                 "from a mapping` on standard OpenAI string arguments",
        "verdict": "REAL IN THE TEMPLATE, NOT REACHABLE THROUGH OUR STACK, AND "
                   "FROGGERIC'S FIX IS WORSE HERE",
        "evidence":
            "c17 reproduces the TypeError exactly. But vLLM's "
            "`_postprocess_messages` json.loads()es the string to a dict before "
            "the template runs, on both render paths, so our endpoint never hits "
            "it (upstream: QwenLM/Qwen3#1894, open, filed against our exact "
            "revision). froggeric renders the raw JSON string with no "
            "`<parameter=>` tags; feeding that back through the real "
            "`_qwen3_arg_converter` from the pinned image recovers "
            "`[[\"get_weather\", {}]]` -- every argument silently dropped.",
    },
    {
        "claim": "Full reasoning_effort support including `high`",
        "verdict": "REAL AND LIVE; the single defect worth acting on",
        "evidence": "c12: official-3.8 (and therefore ours) raises "
                    "`Unexpected reasoning effort high.`; froggeric and unsloth "
                    "render. Reachable because vLLM accepts `high` at the API "
                    "boundary and forwards it. Independent report quoting the "
                    "exact string from a vLLM server running Qwen3.8-27B: "
                    "maximhq/bifrost#6193 (open, 2026-08-15).",
    },
    {
        "claim": "Mid-conversation system prompts crash the template",
        "verdict": "REAL AND LIVE",
        "evidence": "c25 (two leading system messages) and c26 "
                    "(system after an assistant turn) both raise "
                    "`System message must be at the beginning.` for ours; "
                    "froggeric renders both. Upstream vllm#41114 open since "
                    "2026-04-28; the general fix PR#44505 was closed WITHOUT "
                    "merge on 2026-06-05.",
    },
    {
        "claim": "Mutated past turns destroy the prefix cache; this template "
                 "guarantees a 100% KV cache hit rate",
        "verdict": "REAL FOR 3.6, ALREADY FIXED BY QWEN IN 3.8; froggeric's "
                   "remaining advantage is one narrow case",
        "evidence":
            "Measured first-divergent-token between the tokens the engine "
            "generated and the re-rendered history. official-3.6 never reaches "
            "full reuse in any client behaviour (0.9132 at depth 10). "
            "official-3.8 reaches cache_reuse_fraction 1.0 when the client "
            "echoes `reasoning_content`, and 0.9468 at depth 10 when it does "
            "not. froggeric matches official-3.8 in both of those, and beats it "
            "only in `client_inlines_think_in_content` (1.0 vs 0.9468). No "
            "template achieves 100% for a client that discards reasoning "
            "entirely, because the generated prefix contained reasoning that no "
            "longer exists.",
    },
    {
        "claim": "Deep Jinja nesting drops llama.cpp speed by 80%; Python-only "
                 "filters crash C++ engines; AST flattening",
        "verdict": "UNSUBSTANTIATED FOR US, AND OUT OF SCOPE",
        "evidence":
            "These are minijinja/minja (llama.cpp, LM Studio, MLX) properties. "
            "Our stack renders with CPython jinja2 3.1.6 inside the pinned "
            "image; `|items`, slice reversal, `.rstrip('\\n')` and "
            "`loop.previtem` all work. We found no upstream issue quantifying "
            "the 80% figure and did not attempt to reproduce it.",
    },
    {
        "claim": "Use `--tool-call-parser qwen3_xml` on current vLLM; "
                 "`qwen3_coder` only on older builds",
        "verdict": "MOOT IN OUR BUILD",
        "evidence": "`vllm/tool_parsers/__init__.py:157-164` registers both "
                    "`qwen3_coder` and `qwen3_xml` to the same "
                    "`Qwen3EngineToolParser` with the same "
                    "`structural_tag_model = \"qwen_3_coder\"`. Our "
                    "`--tool-call-parser qwen3_coder` is a pure alias; switching "
                    "names would change nothing.",
    },
    {
        "claim": "vLLM setup: replace the `chat_template` string in your "
                 "`tokenizer_config.json`",
        "verdict": "WRONG INSTRUCTION FOR OUR REPOS",
        "evidence": "transformers 5.15.0 prefers `chat_template.jinja`; proved "
                    "with sentinel strings (see K14). Anyone who mounted "
                    "froggeric that way has been serving OUR template.",
    },
    {
        "claim": "Anthropic `message.thinking` payloads are rejected by official "
                 "templates",
        "verdict": "REAL BUT NOT REACHABLE THROUGH OUR ENDPOINT",
        "evidence": "official 3.8 reads only `message.reasoning_content`. "
                    "But vLLM normalises inbound `reasoning_content` to "
                    "`reasoning` and writes BOTH keys back onto the conversation "
                    "message (chat_utils.py:1832-1836), and the Anthropic "
                    "adapter feeds the same conversation shape, so the template "
                    "does receive `reasoning_content`. No measurement in this "
                    "audit exercises `/v1/messages` end to end.",
    },
    {
        "claim": "Two-tier agentic error escalation; dynamic payload truncation; "
                 "`<|think_off|>` inline tags; `tool_call_format=\"json\"`",
        "verdict": "FEATURES, NOT FIXES",
        "evidence": "None of these correspond to a defect in the official "
                    "template. They inject text the tool never returned (K9), "
                    "silently truncate arguments, and add a second tool wire "
                    "format. Adopting them would change what our published "
                    "measurements measured.",
    },
]

VERDICT = {
    "headline": "Ours is not wrong. It is byte-identical to the current official "
                "Qwen template, and it is the most faithful renderer of the wire "
                "format our tool parser expects of everything we measured.",
    "byte_identity": {
        "our_four_repos_sha256":
            "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "identical_to": [
            "Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "Qwen/Qwen3.8-27B-FP8 @ 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "turboderp/Qwen3.8-27B-exl3 @ a35e75a73baee51da709329d19294245cbeeb5d8 (5.00bpw)",
        ],
        "also_identical_in": "tokenizer_config.json (sha256 "
                             "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27) "
                             "across all of the above and all four of our repos",
    },
    "staleness": {
        "question": "Is the template we ship stale relative to current official?",
        "answer": "No. Qwen/Qwen3.8-27B has 16 commits; chat_template.jinja was "
                  "introduced at 72a217afab (2026-08-13T08:23:30Z) and revised "
                  "exactly once, at 412f8b6bd7 (2026-08-13T10:16:05Z), to the "
                  "bytes we ship. Every later commit through HEAD 1d4bf0f2 "
                  "(2026-08-14T15:00:01Z) touches README.md or LICENSE only. "
                  "Qwen/Qwen3.6-27B's template has been byte-stable since "
                  "2026-04-22.",
        "note": "The pinned image itself carries no Qwen template: there is no "
                "chat-template fallback registered for model_type qwen3_5 "
                "(transformers_utils/chat_templates/registry.py), the image ships "
                "no examples/ directory, the only qwen3coder/qwen3/qwen35 .jinja "
                "files in it are Rust unit-test fixtures, and none of the image's "
                "own serve-qwen*.sh scripts pass --chat-template.",
    },
    "generation_has_two_official_templates": {
        "flagship": "Qwen/Qwen3.8-2.4T-A95B @ "
                    "207bd685a7e3696cfaff12ded7c6a7ea0f88c996, sha256 40ce34a5..., "
                    "2026-08-12, text-only, forbids enable_thinking=false (K15), "
                    "copied byte-for-byte by RedHatAI/Qwen3.8-2.4T-A95B and -FP8",
        "27b": "Qwen/Qwen3.8-27B @ 1d4bf0f2..., sha256 c3cf9e34..., 2026-08-13, "
               "vision-capable, supports enable_thinking=false -- what we ship",
        "relationship": "measured over 34 cases there is NOT ONE case where both "
                        "render and the bytes differ: 20 are byte-identical and "
                        "the other 14 are cases where at least one raises. The "
                        "3.8 generation has one tool-call and history format; the "
                        "two templates differ only in which requests they refuse.",
        "why_it_matters": "community claims about 'official 3.8' are frequently "
                          "about the flagship and do not transfer to the 27B.",
    },
    "live_defects_in_what_we_ship": [
        "K3 reasoning_effort='high' -> HTTP 400 (reachable, OpenAI-canonical "
        "value, fixed by both unsloth and froggeric)",
        "K10 two or more system messages -> HTTP 400 (reachable; upstream fix "
        "closed unmerged)",
        "K12 unenumerated content item types -> HTTP 400 (reachable via "
        "/v1/messages; no audited template fixes it)",
        "K4/K5 fabricated empty <think> blocks for history turns whose "
        "reasoning_content the client did not echo (token cost and a bounded "
        "prefix-cache miss at the tail)",
    ],
    "not_defects": [
        "The open `<think>` in the generation prompt. Prompt and "
        "`--reasoning-parser qwen3` agree; vllm#51679's reporter retracted the "
        "reasoning_content half of that claim in the issue body, and the "
        "surviving bug needs NO reasoning parser attached.",
        "`--tool-call-parser qwen3_coder` vs `qwen3_xml`: aliases of one class "
        "in this build.",
        "The 42-token reasoning-effort system message: a deliberate Qwen 3.8 "
        "change, present in every shipped 3.8 template including unsloth's and "
        "froggeric's.",
    ],
    "relation_to_the_reported_garbled_output_incident":
        "None, and the reporter is right that the template is not the issue. "
        "Our template is byte-identical to Qwen's and to turboderp's, and for "
        "plain and simple multi-turn chat froggeric-v22 renders a BYTE-IDENTICAL "
        "prompt to ours in 14 of the 32 audited cases, matching sha256 for "
        "sha256: c01_plain, c02_plain_nogen, c03_system, "
        "c05_multiturn_with_reasoning_content, c07_thinking_off, "
        "c08_thinking_on_explicit, c09_effort_low, c10_effort_medium, "
        "c11_effort_xhigh, c14_effort_none_via_vllm, c22_image_block, "
        "c24_image_vision_id, c29_preserve_thinking_false, "
        "c30_empty_system_string. Every case where they differ involves either "
        "tools or an assistant history turn carrying no reasoning_content. A template cannot explain a difference between two "
        "runs that render the same bytes. This audit is about correctness "
        "against the model's intended format and against what "
        "`--tool-call-parser qwen3_coder` expects, not about that incident, and "
        "no finding here should be cited as a cause of it.",
}

RECOMMENDATION = {
    "decision": "DO NOT change chat_template.jinja in any of the four published "
                "repos.",
    "why": [
        "Byte-identity with Qwen/Qwen3.8-27B is itself the feature. It is what "
        "lets a reader verify our repos against upstream with one sha256, and it "
        "is what makes our published numbers comparable to anyone else's run of "
        "the same checkpoint.",
        "Every live defect (K3, K10, K12) is a 400 on a request shape none of "
        "our published suites emit, and each has a zero-cost operator-side "
        "workaround.",
        "The only alternative that fixes K3 and K10 without adding opinion is "
        "unsloth's (12827f24), but it does not fix K12, and adopting it would "
        "cost a full re-measurement while breaking byte-identity with upstream.",
        "froggeric-v22 is not a candidate: measured against "
        "`--tool-call-parser qwen3_coder` it silently drops all arguments of a "
        "string-argument tool call (K6), renders Python `False` where JSON "
        "`false` is required (K7), injects text the tool never returned into "
        "<tool_response> (K9), and costs +111 tokens per tools request (K8).",
    ],
    "do_instead": [
        {
            "action": "Document the reasoning_effort contract on the four model "
                      "cards: send `xhigh`, `medium`, `low`, or `none`; `high`, "
                      "`minimal` and `max` are rejected by the upstream template "
                      "with HTTP 400. Recommend "
                      "`--default-chat-template-kwargs.reasoning_effort=<value>` "
                      "for operators whose client hard-codes `high`.",
            "cost": "documentation only; no re-measurement",
        },
        {
            "action": "Document that clients must merge multiple system messages "
                      "into one leading message, and that a `developer` role is "
                      "safe because vLLM folds and consolidates it "
                      "(vllm#43590, merged 2026-06-03).",
            "cost": "documentation only",
        },
        {
            "action": "Document that echoing `reasoning_content` back on assistant "
                      "history turns is what buys the 100% prefix-cache hit, with "
                      "the measured numbers from this receipt.",
            "cost": "documentation only",
        },
        {
            "action": "Document the artifact-precedence footgun: our repos ship "
                      "the template twice (chat_template.jinja and the "
                      "tokenizer_config.json key) with identical bytes, and "
                      "transformers 5.15.0 uses the .jinja file. Anyone "
                      "overriding via tokenizer_config.json is silently still on "
                      "ours; use `--chat-template <file>` instead.",
            "cost": "documentation only",
        },
        {
            "action": "File the K3 rejection upstream against Qwen (an alias of "
                      "`high`->`xhigh` plus tolerance for `minimal`/`max`, exactly "
                      "as unsloth already does) rather than forking the template.",
            "cost": "none to us",
        },
    ],
    "if_we_ever_did_change_it": {
        "would_need_rerunning": [
            "The 40-task deterministic downstream retention suite. "
            "tools/task_retention.py sends "
            "`chat_template_kwargs={'enable_thinking': True, "
            "'reasoning_effort': 'low'}` with temperature 0; the template is "
            "applied server-side, so any template edit changes the prompt tokens "
            "and therefore the greedy outputs. Note the suite's `tool_call` "
            "family does NOT use the OpenAI `tools` parameter -- it embeds a "
            "pseudo-tool spec in the user message and checks a JSON object in the "
            "plain-text answer -- so it would have to be re-run, but it is not "
            "evidence about tool-call rendering either way.",
            "The 70-item MMLU-Pro capability matrix "
            "(receipts/public-capability-*.json). Same mechanism: same "
            "chat_template_kwargs, temperature 0, five-shot prefixes rendered "
            "through the template.",
            "Every throughput and context-capacity number, because prompt token "
            "counts move. The measured deltas for a plain single turn are: "
            "official-3.8 59 tokens, unsloth-3.8 59, froggeric-v22 59, "
            "official-3.6 17. For a tools request: 336 / 336 / 447.",
        ],
        "would_NOT_need_rerunning": [
            "The KLD / fidelity suite. It replays a pre-tokenised corpus through "
            "the engine and consults no chat template at all -- there is no "
            "apply_chat_template call and no messages list anywhere in that path "
            "-- so no template change can move a KLD, a top-1 agreement or a "
            "tail statistic. This is stated explicitly so it cannot be left "
            "ambiguous.",
            "Vision placeholder accounting, unless the render_content macro "
            "itself changed: the image path emits exactly "
            "`<|vision_start|><|image_pad|><|vision_end|>` in every audited "
            "3.8-generation template.",
        ],
        "qualified_serving_digest":
            "A template change would NOT invalidate the qualified serving digest "
            "as that digest is constituted: "
            "receipts/public-capability-plan.json's `runtime` block pins the "
            "image digest, rootfs provenance and file manifest, ggrun, the exl3 "
            "and qwen3_5/qwen3_5_mtp module digests, torch, CUDA, GPU and driver "
            "-- none of which contain the model repo's chat template. It WOULD "
            "invalidate the per-repo release evidence: chat_template.jinja's "
            "sha256 is pinned in receipts/k5k6-build-receipt.json, "
            "receipts/hydrated-build-receipt.json, "
            "receipts/context-build-receipt.json and the matching "
            "*-SHA256SUMS files, all of which would need regenerating and "
            "republishing.",
    },
}

UNSUBSTANTIATED = [
    "That `--reasoning-parser qwen3` leaves `reasoning_content` empty because "
    "the template opens `<think>` in the prompt. Retracted by the reporter in "
    "the body of vllm#51679 ('That was wrong -- it works'); contradicted by "
    "reading `Qwen3Parser.is_reasoning_end_for_prompt` in the pinned image.",
    "That vLLM's docs recommend `--chat-template "
    "examples/tool_chat_template_qwen3coder.jinja` for non-Coder Qwen3.x "
    "models. No such issue exists in any of the five repos searched, and "
    "docs/features/tool_calling.md lists only `--tool-call-parser qwen3_xml` "
    "with no --chat-template flag. Closest hit, vllm#38885, was closed stale "
    "2026-08-02 and its specific defect is already absent from the 3.8 template.",
    "That deep Jinja nesting costs llama.cpp 80% throughput. Asserted in "
    "froggeric's README with no measurement; no upstream issue quantifies it; "
    "irrelevant to a CPython jinja2 renderer.",
    "That Python-only filters crash our engine. `|items`, `[::-1]`, "
    "`.rstrip('\\n')` and `loop.previtem` all evaluate in the pinned image's "
    "jinja2 3.1.6.",
    "That the pinned image carries a built-in Qwen template override. It does "
    "not: no fallback registered for model_type qwen3_5, no examples/ directory, "
    "and the only qwen3*.jinja files present are Rust unit-test fixtures.",
    "That the template is implicated in the reporter's garbled-output incident. "
    "For the prompt shapes involved, his mounted template and ours render "
    "byte-identical strings.",
    "A GitHub issue in any of the five target repos reporting the "
    "`Unexpected reasoning effort high` rejection. The only reports naming that "
    "exact string are in downstream integrator repos (maximhq/bifrost#6193, "
    "osaurus-ai/osaurus#2395); the defect is real and measured here regardless.",
    "That \"official 3.8\" forbids enable_thinking=false, as a statement about "
    "Qwen3.8-27B. The construct is real but belongs to Qwen/Qwen3.8-2.4T-A95B "
    "(K15), and briefly to the 27B's superseded rev1 in a different form. "
    "Measured both ways; cited as a property of the 27B it is false.",
    "That vLLM's Rust chat crate renders our template. The pinned image renders "
    "through CPython jinja2 via tokenizer.apply_chat_template; the Rust "
    "qwen3.jinja / qwen35.jinja files are include_str! fixtures inside "
    "#[cfg(test)] mod tests. We did not determine whether the Rust renderer is "
    "reachable by any configuration, only that nothing in our serving path uses "
    "it.",
]
