# Serving Recommendations Across Four Qwen3.x-27B Cards

Comparison of the serving guidance printed on four Hugging Face model cards, cross-checked
against the config files shipped in the same repositories.

## Card identifiers used throughout

| Tag | Repository | Card file read | Card size |
|---|---|---|---|
| **Q3.6** | `Qwen/Qwen3.6-27B` | `/var/tmp/models/Qwen3.6-27B/README.md` | 62,593 B |
| **Q3.8** | `Qwen/Qwen3.8-27B` | `/var/tmp/models/Qwen3.8-27B/README.md` | 65,012 B |
| **NV-3.6** | `nvidia/Qwen3.6-27B-NVFP4` | `/var/tmp/models/Qwen3.6-27B-NVFP4/README.md` | 9,313 B |
| **UN-3.8** | `unsloth/Qwen3.8-27B-NVFP4` | `/var/tmp/models/Qwen3.8-27B-NVFP4/README.md` | 6,586 B |

Two engine documents linked *from* Q3.8 were fetched live and are cited where they resolve a
gap the card itself leaves open:

- **[vLLM-R]** <https://recipes.vllm.ai/Qwen/Qwen3.8-27B> (linked from Q3.8 line 239)
- **[SGL-C]** <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B> (linked from Q3.8 line 238)

Claims that are not traceable to a card or to one of those two linked documents are marked
`[UNVERIFIED]`. Nothing in this file is inferred silently.

---

## 1. Side-by-side serving matrix

### 1.1 Declared engines and version floors

| | Q3.6 | Q3.8 | NV-3.6 | UN-3.8 |
|---|---|---|---|---|
| Compatible runtimes, as worded | "compatible with Hugging Face Transformers, vLLM, SGLang, KTransformers, etc." | "compatible with Hugging Face Transformers, vLLM, SGLang, TokenSpeed, etc." | "**Supported Runtime Engine(s):** vLLM" (only) | none stated |
| SGLang floor | `` `sglang>=0.5.10` is recommended for Qwen3.6 `` | **no pin** — "We recommend using the latest framework versions" | not mentioned | not mentioned |
| vLLM floor | `` `vllm>=0.19.0` is recommended for Qwen3.6 `` | **no pin** | container pin only: "start the docker `vllm/vllm-openai:nightly`" | not mentioned |
| transformers floor | "The latest `transformers` is required for Qwen3.6" + `pip install "transformers[serving]"` | not stated | not mentioned | not mentioned |
| TensorRT-LLM | not mentioned | not mentioned | not mentioned (only `library_name: Model Optimizer` in front matter) | not mentioned |
| llama.cpp / GGUF | not mentioned | not mentioned | not mentioned | not mentioned for this repo; card links "[Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf)" as a *benchmarks* reference only |
| Other | KTransformers section (link only, no command); `transformers serve` command given | SGLang / vLLM / TokenSpeed **links only, no commands** | — | links "[Run Qwen3.8-27B Guide!](https://unsloth.ai/docs/models/qwen3.8)" and Unsloth Desktop |
| Install command printed | `uv pip install sglang[all]` / `uv pip install vllm --torch-backend=auto` | none | none (docker only) | none |

Engine-version floors are mutually inconsistent across sources and I did not reconcile them:
Q3.6 says `vllm>=0.19.0`; [vLLM-R] says "**vLLM 0.27.2**, which is currently nightly … Plain
serving is code-identical on stable 0.27.1" and "**transformers >= 5.8.0**, matching the version
`config.json` was written by". `Qwen3.8-27B/config.json` has `"transformers_version": "5.8.0.dev0"`,
which matches [vLLM-R]. `Qwen3.6-27B/config.json` has `"transformers_version": "4.57.1"`.
`Qwen3.8-27B-NVFP4/config.json` has `"transformers_version": "5.14.1"`.

### 1.2 Launch commands, quoted verbatim

**Q3.6 — SGLang** (four variants; card lines 545, 551, 557):

```shell
python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --port 8000 --tp-size 8 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3
```
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --port 8000 --tp-size 8 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --port 8000 --tp-size 8 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3 --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**Q3.6 — vLLM** (card lines 577, 583, 589, 595):

```shell
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 
```
```shell
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder 
```
```shell
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```
```shell
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
```

**Q3.6 — transformers** (card line 616):

```shell
transformers serve Qwen/Qwen3.6-27B --port 8000 --continuous-batching
```

**Q3.8 —** the card prints **no complete serving command**. Its only `vllm serve` /
`sglang.launch_server` / `tokenspeed serve` lines are the YaRN examples in §1.6 below, all
written with an elided `...` in place of the model and the rest of the arguments.

**NV-3.6 —** exactly one command (card, "Usage"):

```sh
vllm serve nvidia/Qwen3.6-27B-NVFP4 --port 8000 --quantization modelopt --max-model-len 262144 --reasoning-parser qwen3
```

**UN-3.8 —** the card prints **no command of any kind**. No `vllm`, no `sglang`, no
`transformers`, no `llama.cpp`. All deployment guidance is a link to
<https://unsloth.ai/docs/models/qwen3.8>.

### 1.3 Flag-by-flag

| Flag | Q3.6 | Q3.8 | NV-3.6 | UN-3.8 |
|---|---|---|---|---|
| `--quantization` | absent | absent | `--quantization modelopt` | absent |
| `--kv-cache-dtype` | **absent** | **absent** | **absent** (see §3.1 — the checkpoint quantizes KV anyway) | **absent** (same problem) |
| `--max-model-len` | `262144` | only in YaRN example: `1000000` | `262144` | absent |
| `--context-length` (SGLang) | `262144`; YaRN example `1010000` | YaRN example `1000000` | n/a | n/a |
| TP | `--tp-size 8` (SGLang) / `--tensor-parallel-size 8` (vLLM), described as "tensor parallel on 8 GPUs" | absent | **absent** → TP1 implied | absent |
| `--mem-fraction-static` | `0.8` | absent | n/a | n/a |
| Reasoning parser | `--reasoning-parser qwen3` in every command | absent from card | `--reasoning-parser qwen3` | absent |
| Tool-call parser | `--tool-call-parser qwen3_coder`; vLLM additionally needs `--enable-auto-tool-choice` | absent | **absent** | absent |
| Speculative / MTP | vLLM `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`; SGLang `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | absent | **absent** | **absent** |
| Text-only / skip vision | `--language-model-only` ("skips the vision encoder and multimodal profiling to free up memory for additional KV cache") | absent | absent | absent |
| Eager vs graph | never mentioned on any card | " | " | " |

### 1.4 Context-length claims

| | Claim on the card | `max_position_embeddings` in the repo's `config.json` |
|---|---|---|
| Q3.6 | "Context Length: 262,144 natively and extensible up to **1,010,000** tokens." YaRN examples use `1010000`. Card also warns: "we advise maintaining a context length of at least 128K tokens to preserve thinking capabilities." | `text_config.max_position_embeddings: 262144` — consistent with the native figure |
| Q3.8 | "Context Length: 262,144 natively and extensible up to **1,000,000** tokens." YaRN examples use `1000000`. | `262144` — consistent |
| NV-3.6 | "**Other Properties Related to Input:** Context length up to 262K" | `262144` — consistent; NV-3.6 makes **no** 1M extensibility claim and prints no YaRN guidance |
| UN-3.8 | "Context Length: 262,144 natively and extensible up to 1,000,000 tokens." (verbatim copy of Q3.8) | `262144` — consistent |

The 1.01M vs 1.00M discrepancy is a genuine card-to-card difference between the Qwen3.6 and
Qwen3.8 generations, not a config mismatch.

### 1.5 Multimodal (image/video) preprocessing requirements

| | Q3.6 | Q3.8 | NV-3.6 | UN-3.8 |
|---|---|---|---|---|
| Declared modalities | "compatible with … " + `pipeline_tag: image-text-to-text`; image and video API examples | same, `pipeline_tag: image-text-to-text` | "**Input Type(s):** Text, Image, Video"; "**Input Format(s):** String, Red, Green, Blue (RGB), Video (MP4/WebM)" | `pipeline_tag` absent from front matter; prose says "native vision-language model that understands images and videos" |
| Video frame-sampling kwargs | `"mm_processor_kwargs": {"fps": 2, "do_sample_frames": True}` in the live example (line 751); comment: "When vLLM is launched with `--media-io-kwargs '{"video": {"num_frames": -1}}'` … This feature is currently supported only in vLLM"; "By default, `fps=2` and `do_sample_frames=True`" | identical text, but the `mm_processor_kwargs` call is **commented out** (lines 405–417) | not mentioned | not mentioned |
| `video_preprocessor_config.json` retune | Best Practices §4: set `longest_edge` to `469,762,048`; `{"longest_edge": 469762048, "shortest_edge": 4096}`; alternative = "override the default values via engine startup parameters", citing <https://github.com/vllm-project/vllm/pull/34330> and <https://github.com/sgl-project/sglang/pull/18467> | Best Practices §4, same numbers and same two PR links | **omitted entirely** | Best Practices §4 present (copied), **but the PR-link sentence is dropped** |
| Extra prerequisite | "Please also make sure torchvision and pillow are installed." (transformers path) | absent | absent | absent |
| Preprocessor files shipped | `preprocessor_config.json`, `video_preprocessor_config.json` | same | same **plus** `processor_config.json` | same |

All four repos ship byte-identical `preprocessor_config.json`
(`size.longest_edge: 16777216`, `shortest_edge: 65536`, `patch_size: 16`,
`temporal_patch_size: 2`, `merge_size: 2`, `image_mean/std: [0.5,0.5,0.5]`,
`processor_class: "Qwen3VLProcessor"`, `image_processor_type: "Qwen2VLImageProcessorFast"`)
and byte-identical `video_preprocessor_config.json`
(`size.longest_edge: 25165824`, `shortest_edge: 4096`, `video_processor_type: "Qwen3VLVideoProcessor"`).

**Mismatch (NV-3.6, internal):** NV-3.6's extra `processor_config.json` declares
`image_processor.image_processor_type: "Qwen2VLImageProcessor"` (the slow processor) while the
`preprocessor_config.json` in the *same directory* declares
`image_processor_type: "Qwen2VLImageProcessorFast"`. The card never mentions
`processor_config.json`, so an operator cannot tell which of the two the card intends. This
file does not exist in `Qwen/Qwen3.6-27B`.

**Mismatch (both quant cards, Best Practices §4):** UN-3.8 tells the operator to raise
`longest_edge` to `469762048`, and the repo it ships still contains
`video_preprocessor_config.json` → `size.longest_edge: 25165824`. That is exactly the situation
the sentence describes ("the `size` parameter in the released `video_preprocessor_config.json`
is conservatively configured"), so the advice is self-consistent — but UN-3.8 dropped the
"override via engine startup parameters" escape hatch and its two PR links, leaving file
mutation as the only documented route.

### 1.6 `--hf-overrides`, `--json-model-override-args`, and environment variables

**Q3.6 (card lines 957, 962)** — YaRN to 1,010,000:

```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1010000  
```
```shell
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --context-length 1010000
```
Q3.6 states this SGLang form covers "`sglang` and `ktransformers`".

**Q3.8 (card lines 542, 547, 552)** — same three keys, 1,000,000, plus a third engine:

```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
```
```shell
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --context-length 1000000
```
```shell
TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 tokenspeed serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
```

Both cards attach the same caveat: "All the notable open-source frameworks implement static
YaRN … **potentially impacting performance on shorter texts.** We advise modifying the
`rope_parameters` configuration only when processing long contexts is required."

**Config check.** The override payload must *change* `rope_type`. All four repos ship
`text_config.rope_parameters` = `{"mrope_interleaved": true, "mrope_section": [11,11,10],
"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}`. So
`mrope_interleaved`, `mrope_section`, `rope_theta` and `partial_rotary_factor` in the override
are re-statements of the shipped values; the only functional deltas are
`rope_type: "default"` → `"yarn"`, plus the added `factor` and
`original_max_position_embeddings`. Neither card says this, so an operator cannot tell which
keys are load-bearing.

**Divergence from the linked vLLM recipe.** [vLLM-R] reaches 1M by a *different* override —
`--hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'` with
`--max-model-len 1010000`, no `rope_parameters` and no `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, and it
remarks "Note the override is nested under `text_config` here, where the 2.4T takes it flat."
Q3.8 and its own linked recipe therefore prescribe two incompatible 1M procedures.

**NV-3.6 and UN-3.8 print no `--hf-overrides` and no environment variables at all.**

### 1.7 Tool calling and reasoning parsers

| | Q3.6 | Q3.8 | NV-3.6 | UN-3.8 |
|---|---|---|---|---|
| Reasoning parser | `--reasoning-parser qwen3` in all four commands | not on card | `--reasoning-parser qwen3` | not on card |
| Tool parser | `--tool-call-parser qwen3_coder`; vLLM also `--enable-auto-tool-choice` | not on card | not on card | not on card |
| `reasoning_effort` | **not supported**: the string does not appear in Q3.6's card or its `chat_template.jinja` (0 occurrences) | `xhigh` (default) / `medium` / `low`; `reasoning_effort="xhigh"` shown in the SDK example | not mentioned | `reasoning_effort` named in prose ("reasoning depth can be tuned with `reasoning_effort`") but no values listed and no example |
| Thinking switch | `"chat_template_kwargs": {"enable_thinking": False}`; explicit "Qwen3.6 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`." | `"chat_template_kwargs": {"enable_thinking": False}`; note that Qwen Cloud takes bare `"enable_thinking": False` | not mentioned (but `--reasoning-parser qwen3` is in the command) | "Thinking mode is on by default and can be disabled per request" — **no mechanism given** |
| `preserve_thinking` | **off by default**: "By default, only the thinking blocks generated in handling the latest user message is retained … You can enable this behavior by setting the `preserve_thinking` option" | **on by default**: "`preserve_thinking` is enabled by default for all workloads"; disable via `"chat_template_kwargs": {"preserve_thinking": False}` | not mentioned | "reasoning context from historical messages is retained via `preserve_thinking`" — default not stated |
| Agentic harnesses | Qwen-Agent (`'model_type': 'qwenvl_oai'`, `'use_raw_api': True`) and Qwen Code sections with code | none | none | "Developer Role Support so Qwen3.8 can work in agentic tools like Codex and more!" |

NV-3.6 and UN-3.8 both ship a tool-calling chat template and neither prints
`--tool-call-parser qwen3_coder`. [SGL-C] states the consequence explicitly: "without them a
harness receives tool calls as raw text instead of structured `tool_calls`", and warns that
`--tool-call-parser hermes` "reads a *different* payload — bare JSON inside `<tool_call>` — so
pointing a Hermes-format harness at this model without switching the flag yields tool calls
that never parse."

---

## 2. Sampling / generation recommendations vs. shipped `generation_config.json`

### 2.1 What each card recommends

| | Q3.6 | Q3.8 | NV-3.6 | UN-3.8 |
|---|---|---|---|---|
| Thinking, general | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` | identical | — | identical (copied from Q3.8) |
| Thinking, precise coding | **`temperature=0.6`**, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` ("e.g., WebDev") | **no coding-specific set** | — | **no coding-specific set** |
| Instruct / non-thinking | `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` | identical | — | identical |
| `presence_penalty` guidance | "adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance." | same wording ("repetition" singular) | — | same |
| Output length | "32,768 tokens for most queries"; "81,920 tokens" for math/programming competitions. SDK examples pass `max_tokens=81920` (thinking) and `max_tokens=32768` (non-thinking, preserve-thinking) | split budgets: "Reasoning Content: … 262,144 tokens"; "Final Response: … 131,072 tokens", "within the 1M context length". **No `max_tokens` in any SDK example.** | eval-only: "max_num_tokens=81920" | copied Q3.8 split: 262,144 / 131,072 |
| Benchmark prompt scaffolds | Best Practices §3: math → "Please reason step by step, and put your final answer within \boxed{}."; MCQ → "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`." | **omitted** | not applicable | **omitted** |
| System prompt handling | No system message in any example; template puts a leading `system` message first, non-leading raises | same; "System message must be at the beginning." enforced by template | not discussed | "Developer Role Support" claimed; no example |

Q3.6 duplicates the sampling block twice (Tip at lines 634–637 and Best Practices §1 at
974–982) with identical numbers. Q3.8 likewise (Tip at 251–253, Best Practices §1 at 500–503).

### 2.2 Shipped `generation_config.json`

Q3.6, Q3.8 and NV-3.6 ship **byte-identical** files (md5 `efeaa00fbdbaaa33be52e9373c35da5f`):

```json
{"bos_token_id": 248044, "do_sample": true, "eos_token_id": [248046, 248044],
 "pad_token_id": 248044, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
```

UN-3.8's differs only by re-indentation plus one added key (md5 `2824e4ed9e0e55571313f8f013d276fe`):
the same seven keys plus `"transformers_version": "5.14.1"`. No sampling value changed.

### 2.3 Every card-vs-config mismatch

| # | Cards affected | Card prose | Shipped config key/value | Nature of mismatch |
|---|---|---|---|---|
| M1 | Q3.6, Q3.8, UN-3.8 | non-thinking set requires `temperature=0.7`, `top_p=0.80`, `presence_penalty=1.5` | `temperature: 1.0`, `top_p: 0.95`; **`presence_penalty` key absent** | The shipped defaults encode *only* the thinking-mode preset. An engine that seeds sampling from `generation_config.json` and is then asked for `enable_thinking: False` silently serves non-thinking generation at thinking-mode temperature unless the client overrides all three. No card says so. |
| M2 | Q3.6 | thinking-mode coding preset `temperature=0.6` | `temperature: 1.0` | Third documented preset has no representation in the config at all. |
| M3 | all four | all four cards list `min_p=0.0` and `repetition_penalty=1.0` in every preset | **neither `min_p` nor `repetition_penalty` appears in any `generation_config.json`** | Cards specify parameters the checkpoint does not default; values coincide with common engine defaults but the cards assert them as requirements. |
| M4 | all four | every preset specifies `top_k=20` | `top_k: 20` ✓ | Only sampling parameter that matches across the board. |
| M5 | Q3.6 | "Adequate Output Length: … 32,768 tokens for most queries" | no `max_new_tokens` / `max_length` key in `generation_config.json` | The 32,768 / 81,920 figures are unenforced; must come from `--max-model-len` budgeting or per-request `max_tokens`. |
| M6 | Q3.8, UN-3.8 | "Reasoning Content: Set the maximum output length to **262,144** tokens. Final Response: … **131,072** tokens" and "within the 1M context length" | `text_config.max_position_embeddings: 262144` | The recommended split (262,144 + 131,072 = 393,216 output tokens) **exceeds the shipped 262,144-token window** and is only reachable after the YaRN override in §1.6. Q3.8 states the 1M precondition; **UN-3.8 copies the numbers while also copying the YaRN section**, so the precondition survives — but neither card flags that the split is unusable at stock config. |
| M7 | UN-3.8 | "Thinking mode is on by default and can be disabled per request" | `chat_template.jinja` line: `{%- if enable_thinking is undefined or enable_thinking is true %}` | Behaviour is correct, but the card gives no `chat_template_kwargs` example, having deleted Q3.8's "Instruct (or Non-Thinking) Mode" section. The operator has the claim without the switch. |
| M8 | NV-3.6 | eval footnote: "Benchmarked with temperature=1.0, top_p=0.95, max_num_tokens=81920 except SciCode with temperature=0.6, and τ²-Bench Telecom with temperature=0.0 and top_p=1.0" | `temperature: 1.0`, `top_p: 0.95` ✓ | Consistent. NVIDIA's SciCode `temperature=0.6` silently reproduces Q3.6's "precise coding tasks" preset, and the τ²-Bench `temperature=0.0` is greedy — outside every preset any card recommends. NV-3.6 gives **no serving-time sampling recommendation of its own**; the operator must go back to Q3.6. |
| M9 | UN-3.8 | `reasoning_effort` supports, per Q3.8, exactly "`xhigh` (default)", "`medium`", "`low`", and upstream's template raises on anything else | UN-3.8's `chat_template.jinja` adds `{%- if resolved_reasoning_effort == 'high' %}{%- set resolved_reasoning_effort = 'xhigh' %}{%- endif %}` | The quantizer silently **added a fourth accepted value**, `high`, aliased to `xhigh`. A client that sends `reasoning_effort="high"` gets a hard exception on upstream and maximum reasoning depth here. Not documented on the card. |
| M10 | UN-3.8 | card front matter and prose are a verbatim reproduction of Q3.8's model overview, including "Token Embedding: 248,320 (Padded)" and the full architecture block | `config.json` matches (`vocab_size: 248320`) ✓ | Architecture claims check out — see §3.6. |

---

## 3. Differences that matter operationally

### 3.1 KV-cache dtype — the largest undocumented behaviour change

Both quantizers bake KV-cache quantization into the checkpoint and **neither card mentions it**;
neither prints `--kv-cache-dtype`.

| | Config evidence | Card |
|---|---|---|
| Q3.6 / Q3.8 | no KV quantization anywhere in `config.json` | no mention; BF16 KV implied |
| NV-3.6 | `hf_quant_config.json` → `quantization.kv_cache_quant_algo: "FP8"` | **silent** |
| UN-3.8 | `config.json` → `quantization_config.kv_cache_scheme` = `{"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": true, "dynamic": false, "observer": "static_minmax", "group_size": null, "actorder": null}` | **silent** |

Consequences an operator must derive elsewhere: FP8 KV is per-tensor **static** in the unsloth
checkpoint (`dynamic: false`, `observer: "static_minmax"`), i.e. it carries baked calibration
scales that were produced by a calibration pass the card never describes. [SGL-C] confirms the
engine honours it without a flag: "The NVFP4 checkpoint declares `kv_cache_quant_algo: FP8`;
SGLang's default `--kv-cache-dtype auto` honors it, so the KV pool runs in `fp8_e4m3` with the
checkpoint's calibration scales automatically." [vLLM-R] instead passes it explicitly:
`--kv-cache-dtype fp8`. Any accuracy comparison against Q3.6/Q3.8 BF16 that ignores this is
comparing two different cache precisions.

### 3.2 Activation quantization — different kernels, and one card overclaims

NV-3.6's post-training-quantization paragraph reads:

> "This model was obtained by quantizing the weights **and activations** of Qwen3.6-27B to
> NVFP4 data type … **Only** the weights and activations of the linear operators **within
> transformer blocks** are quantized."

Both halves are contradicted by the files in the same repository:

- `hf_quant_config.json` → `quantization.quantized_layers` contains **193 entries with
  `quant_algo: "W4A16_NVFP4"`** (e.g. `model.language_model.layers.0.mlp.gate_proj` →
  `{"quant_algo": "W4A16_NVFP4", "group_size": 16}`) and **208 entries with
  `quant_algo: "FP8"`** (e.g. `model.language_model.layers.3.self_attn.q_proj` →
  `{"quant_algo": "FP8"}`). `W4A16` means **16-bit activations**. Confirmed independently by
  `config.json` → `quantization_config.config_groups.group_1`, whose only two keys are
  `weights` and `targets`: `{"weights": {"dynamic": false, "num_bits": 4, "type": "float",
  "group_size": 16}, "targets": [...]}`. The `input_activations` key is **absent entirely** from
  `group_1`, while `group_0` does carry it (`{"dynamic": false, "num_bits": 8, "type":
  "float"}`) — so the omission is deliberate, not a defaulting artifact, and `group_1` is
  weight-only. **No activation anywhere in this checkpoint is NVFP4.** The only quantized
  activations are FP8.
- `quantization_config.config_groups.group_1.targets[0]` is `"lm_head"`, and
  `quantized_layers` contains `"lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}`.
  `lm_head` is **not** "within transformer blocks", so "only … within transformer blocks" is
  false.

UN-3.8's card makes no quantization claim at all, so it cannot contradict its config — but its
config is a genuinely different kernel target:

| | NV-3.6 | UN-3.8 |
|---|---|---|
| MLP scheme | `W4A16_NVFP4`, `group_size: 16`, all 64 layers | `group_1` = `{"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group", "actorder": "static", "observer": "imatrix_mse", "scale_dtype": "torch.float8_e4m3fn"}` weights **plus** `input_activations` `{"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group", "dynamic": "local", "observer": "static_minmax"}` → **W4A4**; targets `re:.*mlp\.(gate|up|down)_proj$`, but layers 56–63 are pulled out into `group_0` (FP8) |
| Attention / GDN projections | FP8 with `input_activations: {"dynamic": false, ...}` → **static** per-tensor activation scales | `group_0` weights `{"num_bits": 8, "type": "float", "strategy": "channel", "observer": "memoryless_minmax"}`, `input_activations` `{"num_bits": 8, "type": "float", "strategy": "token", "dynamic": true}` → **dynamic per-token**. Targets: `re:.*self_attn\.(q\|k\|v\|o)_proj$`, `re:.*linear_attn\.(in_proj_qkv\|in_proj_z\|out_proj)$`, `re:.*lm_head`, `re:.*layers\.(56\|57\|58\|59\|60\|61\|62\|63)\.mlp\.(gate\|up\|down)_proj$` |
| `lm_head` | NVFP4 `W4A16` | **FP8** (`re:.*lm_head` sits in `group_0`; absent from `ignore`) |
| GDN gating projections | quantized (`linear_attn.in_proj_qkv/in_proj_z/out_proj` → FP8) | `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `linear_attn.norm` and bare `linear_attn` explicitly in `ignore` for all 48 GDN layers |
| `quant_method` string | `"modelopt"`, `quant_algo: "MIXED_PRECISION"`, producer `modelopt 0.45.0` | `"compressed-tensors"`, `format: "mixed-precision"`, `version: "0.17.2.a20260716"`, `quantization_status: "compressed"` |
| Loader flag on card | `--quantization modelopt` | none printed |

Operationally: NV-3.6 is a weight-only 4-bit MLP that runs on any GPU with an INT4/FP4
*dequantizing* GEMM; UN-3.8's MLP requires a genuine **FP4×FP4** GEMM for 56 of 64 layers. The
static-vs-dynamic activation split also matters — NVIDIA's FP8 activation scales are frozen at
calibration time and will clip on distributions unlike its calibration corpus; unsloth's are
recomputed per token.

### 3.3 GPU architecture requirements

NV-3.6 declares:

> "**Supported Hardware Microarchitecture Compatibility:** NVIDIA Hopper, NVIDIA Blackwell"
> and "**Test Hardware:** NVIDIA GB300"

That is coherent for a `W4A16` checkpoint. [SGL-C] describes the Hopper path for a
*W4A4* checkpoint of the same family: "**H200 (SM90)**: BF16 and FP8 only — the card has no
FP4 tensor cores, so the NVFP4 checkpoint's MLP would fall back to the Marlin W4A16
weight-only path and its cell is greyed out."

So the operationally sharp point is the asymmetry: NV-3.6 is *natively* W4A16 and loses nothing
on Hopper beyond FP4 tensor-core throughput it never wanted, while **UN-3.8 is W4A4 and states
no GPU requirement whatsoever** — no Blackwell/SM100/SM120 note, no mention that its
activation quantization has no Hopper kernel. [SGL-C] also pins the backend for consumer
Blackwell: "**SM120/SM121 (RTX PRO 6000 Blackwell, RTX 5090, DGX Spark)**: use
`--attention-backend flashinfer`; `trtllm_mha` is SM100-only." Neither quant card mentions an
attention backend. I did not run either checkpoint, so the practical outcome of loading UN-3.8
on Hopper is `[UNVERIFIED]`.

### 3.4 Vision-tower handling

| | Evidence |
|---|---|
| NV-3.6 | `hf_quant_config.json` → `quantization.exclude_modules: ["mtp*", "mtp.layers.0*"]` — the vision tower is **not named**. Independently, zero targets matching `visual` appear in any `config.json` `config_groups` entry, so the 27 vision blocks are in fact unquantized — but by omission, not by declaration. The card claims Image and Video input support and never states the vision tower's precision. |
| UN-3.8 | `quantization_config.ignore` **explicitly enumerates** all 27 vision blocks (`model.visual.blocks.0.attn.qkv` … `model.visual.blocks.26.mlp.linear_fc2`) plus `model.visual.merger.linear_fc1` and `model.visual.merger.linear_fc2`. Declared in the config, undeclared on the card. |
| Q3.6 | Only card that gives an operator a lever here: `--language-model-only`, "skips the vision encoder and multimodal profiling to free up memory for additional KV cache". Backed by `config.json` → `"language_model_only": false`. Q3.8, NV-3.6 and UN-3.8 all ship the same `language_model_only: false` key and none mentions the flag. |

**Undeclared config edit (UN-3.8):** `vision_config.model_type` was changed from `"qwen3_5"`
(upstream Q3.8) to `"qwen3_5_vision"`, and `dtype: "bfloat16"` was added to `vision_config`.
Top-level keys `head_dim: 256`, `num_attention_heads: 24`, `num_key_value_heads: 4`,
`dtype: "bfloat16"` were hoisted out of `text_config` to the root. `text_config` itself is
byte-identical to upstream after normalisation (verified by `jq -S` diff: empty). None of this
is on the card.

### 3.5 MTP / speculative decoding

All four repos ship MTP weights (`text_config.mtp_num_hidden_layers: 1`,
`mtp_use_dedicated_embeddings: false`) and only **Q3.6** documents how to use them.

- Q3.6 vLLM: `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`
- Q3.6 SGLang: `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
- Q3.8: **nothing**. Its Model Overview says "MTP (Multi-Token Prediction): trained with multiple steps" and stops.
- NV-3.6: nothing on the card, yet the quantizer went out of its way to protect the head —
  `exclude_modules: ["mtp*", "mtp.layers.0*"]` leaves it BF16, which is only useful if someone
  runs speculation.
- UN-3.8: nothing on the card; `ignore` ends with `"re:^mtp.*"`, same intent.

Three mutually incompatible names for one feature are now in circulation:
`qwen3_next_mtp` (Q3.6 card), `mtp` ([vLLM-R]: `--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'`), and `EAGLE` ([SGL-C]:
"`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4` uses the in-checkpoint MTP head. (This recipe was originally
documented with `NEXTN`, an alias of `EAGLE` — same algorithm.)"). Note also that the Q3.6 card
spells the SGLang flag `--speculative-algo` while [SGL-C] spells it
`--speculative-algorithm`.

**Packaging hazard, UN-3.8 only.** The repo ships MTP weights in a *separate* file,
`model_mtp.safetensors` (849,400,392 B), and `model.safetensors.index.json` maps 15 keys
(`mtp.fc.weight`, `mtp.layers.0.input_layernorm.weight`, `mtp.layers.0.mlp.down_proj.weight`, …)
to that filename while every other key maps to `model.safetensors`. So the repo simultaneously
contains a single non-sharded `model.safetensors` (22,568,192,096 B) **and** an index that
references two files, with `metadata.total_size: 23417592488`. Upstream Q3.8 ships 18 numbered
shards with a conventional index. Whether every loader tolerates this layout is
`[UNVERIFIED]` — I did not load the checkpoint — but the card says nothing about it, and an
operator who copies only `model.safetensors` gets a model whose index promises tensors that are
not present.

### 3.6 Calibration-dataset disclosure

| | Disclosure |
|---|---|
| NV-3.6 | Full section. "**Calibration Dataset: Link:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)"; "**Data Collection Method by dataset:** Automated. **Labeling Method by dataset:** Automated." Training dataset declared "Undisclosed" across all modalities. |
| UN-3.8 | **None.** Yet the config proves calibration happened: `group_1.weights.observer: "imatrix_mse"` with `actorder: "static"` is an importance-matrix search, `group_1.input_activations.observer: "static_minmax"`, and `kv_cache_scheme.observer: "static_minmax"` with `dynamic: false` all require a data pass. The corpus, its size, its language mix and its modality coverage are entirely undisclosed. For a checkpoint whose 27-block vision tower is excluded from quantization, whether the calibration set contained any images at all is unknowable from the repo. |
| Q3.6 / Q3.8 | Not applicable (BF16 originals); neither discloses training data either. |

### 3.7 Tensor-parallel constraints

Q3.6 is the only card that names a TP degree, and it names one: `--tp-size 8` /
`--tensor-parallel-size 8`, described as "tensor parallel on 8 GPUs", in **every** command
including the MTP and text-only variants — with no smaller-TP alternative and no statement of
whether 8 is a requirement or a sizing choice for a 55.56 GB BF16 model. NV-3.6 prints no TP
flag at all, implying TP1 on the GB300 it was tested on. UN-3.8 prints nothing. No card states
a divisibility constraint, which is not obviously safe to assume: `num_key_value_heads: 4` and
`linear_num_key_heads: 16` bound the useful TP degree, and 4 KV heads under TP8 requires KV
replication. No card mentions this. `[UNVERIFIED]` — I did not test any TP degree.

### 3.8 Accuracy numbers and their harnesses

**NV-3.6 is the only one of the four quant/base cards that publishes a quantization accuracy
table.** Nine benchmarks, FP8 vs NVFP4:

| | MMLU Pro | GPQA Diamond | HLE | τ²-Bench Telecom | MMMU Pro | SciCode | AIME 2025 | AA-LCR | IFBench |
|---|---|---|---|---|---|---|---|---|---|
| FP8 | 86.1 | 86.0 | 21.7 | 95.2 | 74.6 | 44.8 | 93.1 | 68.8 | 65.1 |
| NVFP4 | 86.3 | 85.5 | 21.8 | 95.4 | 74.3 | 44.5 | 92.7 | 68.3 | 65.5 |

Three caveats the operator has to notice for themselves:

1. **The baseline is not BF16.** "Baseline: [Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)."
   The table therefore measures NVFP4-vs-FP8, and says nothing about the FP8-vs-BF16 step. The
   headline "NVFP4 ≈ baseline" understates cumulative loss from BF16.
2. **No harness is named.** The card lists datasets and sampling parameters
   ("temperature=1.0, top_p=0.95, max_num_tokens=81920 except SciCode with temperature=0.6, and
   τ²-Bench Telecom with temperature=0.0 and top_p=1.0") and "**Acceleration Engine:** vLLM,
   **Test Hardware:** NVIDIA GB300" — but never `lm-evaluation-harness`, `evalchemy`, `simple-evals`
   or any other named runner, no version, no commit, no `n`/`avg@k`, no seed. Contrast the
   upstream cards, which do name harnesses (Q3.6: "Terminal-Bench 2.0: Harbor/Terminus-2
   harness"; "SkillsBench: Evaluated via OpenCode on 78 tasks … avg of 5 runs"; "SWE-Bench
   Series: Internal agent scaffold (bash + file-edit tools)"; Q3.8: "Evaluated with the Claude
   Code harness at temp=1.0, top_p=0.95, and a 256K context window", "HLE: Judged by GPT-4o").
3. **Three of the nine deltas favour NVFP4** (MMLU Pro +0.2, HLE +0.1, τ²-Bench +0.2, IFBench
   +0.4). With no `n` and no error bars, the table cannot distinguish "lossless" from "noise".

**UN-3.8 publishes no accuracy number for this checkpoint.** Its only quantization-quality
pointer is "*See [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf)
for our quantization benchmarks*" — a link about a *different* quantization format (GGUF
dynamic 2.0) for *other* models. No MMLU, no KLD, no perplexity, no NVFP4 vs BF16 comparison
of any kind for `unsloth/Qwen3.8-27B-NVFP4`.

### 3.9 Chat-template divergence — the quantizer that edited the template

`chat_template.jinja` md5:

| Repo | md5 | Bytes |
|---|---|---|
| `Qwen3.6-27B` | `52b6d51ae5b203cb67e64b648494dad2` | 7,764 |
| `Qwen3.6-27B-NVFP4` | `52b6d51ae5b203cb67e64b648494dad2` | 7,764 |
| `Qwen3.8-27B` | `519239a4908bb1f805bbce5fa8c8a242` | 8,952 |
| `Qwen3.8-27B-NVFP4` | `2a79880b328d0e0387c8ecb62c4c0c80` | **9,993** |

NVIDIA passes upstream's template through byte-identically (as it does
`tokenizer_config.json`, md5 `dee7749bf0c52b1185a0b06fa1706724` in both, and
`generation_config.json`). **Unsloth modified the template**, and its card discloses two of the
five behaviour changes. The file's own trailing comment is
`{#- Unsloth fixes - developer role, merged system messages, tool calling #}`.

Verified changes (`diff Qwen3.8-27B/chat_template.jinja Qwen3.8-27B-NVFP4/chat_template.jinja`):

| Change | Card coverage |
|---|---|
| `developer` role accepted wherever `system` was, and **all leading `system`/`developer` messages are merged** with `'\n'` into one block (new `sysns` namespace, `merged_system`, `num_sys`) | Claimed: "Developer Role Support so Qwen3.8 can work in agentic tools like Codex and more!" The *merging* of multiple leading system messages is a separate semantic change and is **not** on the card. |
| `reasoning_effort == 'high'` silently rewritten to `'xhigh'` (upstream raises `'Unexpected reasoning effort …'`) | **Undisclosed.** See M9. |
| The `ns.multi_step_tool` → `raise_exception('No user query found in messages.')` guard **deleted** | **Undisclosed.** Prompts upstream rejects now render. |
| A non-leading `system`/`developer` message now **always** raises `'System message must be at the beginning.'`, where upstream only raised for a `system` message that was not `loop.first` | **Undisclosed** behaviour change at the boundary. |
| Tool-call argument typing tightened: `tool_call.arguments is mapping` required; a JSON **string** now raises `'Tool call arguments for function "…" were passed as a JSON string. Parse them into an object before calling apply_chat_template.'`; a missing name raises `'Tool call is missing a function name.'` | Card says "Tool calling improvements: Makes parsing nested objects to make tool calling succeed more." Partially substantiated: the diff adds **validation and explicit failure**, not nested-object parsing (upstream already did `args_value | tojson` for non-strings). Callers that previously passed stringified arguments and got malformed-but-rendered output now get a hard exception. |

**Additional undocumented packaging change:** upstream `Qwen3.8-27B/tokenizer_config.json`
carries a `chat_template` key (`jq 'has("chat_template")'` → `true`) and an
`added_tokens_decoder` map; UN-3.8's `tokenizer_config.json` is 1,124 B, has
`has("chat_template")` → **`false`**, and drops `added_tokens_decoder`,
`additional_special_tokens` and `add_bos_token`. Any tool that reads the template from
`tokenizer_config.json` rather than `chat_template.jinja` gets **no template at all** from this
repo. The special tokens themselves survive: `tokenizer.json` `added_tokens` length is 33 in
both Q3.8 and UN-3.8 (and 26 in Q3.6), with `model.vocab` length 248,044 in all three.

### 3.10 Checkpoint identity in the linked engine recipes

Neither engine document linked from Q3.8 names `unsloth/Qwen3.8-27B-NVFP4`. [vLLM-R]'s
low-latency command is `vllm serve Inferact/Qwen3.8-27B-NVFP4 --tensor-parallel-size 1
--max-model-len 262144 --kv-cache-dtype fp8 --reasoning-parser qwen3 --enable-auto-tool-choice
--tool-call-parser qwen3_coder`, and lists "NVFP4 build (NVIDIA):
<https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4>". [SGL-C] resolves NVFP4 to
`RadixArk/Qwen3.8-27B-NVFP4` ("NVFP4 W4A4 + FP8 projections") and additionally documents a
separate DSpark draft checkpoint (`--speculative-algorithm DSPARK
--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark`). Whether those repos are the same
artifact as unsloth's is out of scope here and **`[UNVERIFIED]`** — the point is only that a
UN-3.8 operator who follows the upstream-linked recipes will be pointed at a different repo,
with flags (`--kv-cache-dtype fp8`, TP1, both parsers) that UN-3.8's own card never mentions.

### 3.11 Size claims

NV-3.6: "reducing the disk size and GPU memory requirements by approximately 2.5x." Measured
against our established figures: 55.56 GB → 21.92 GB = **2.535×**. Claim checks out.
The equivalent unsloth ratio is 55.56 → 23.42 GB = **2.372×**, and UN-3.8 makes no size claim.
[SGL-C] independently quotes "NVFP4 weights ~16.5GB" and [vLLM-R] "NVFP4 in 24.6 GiB" for
Qwen3.8-27B NVFP4 builds — three different numbers for nominally the same recipe, which is why
a per-repo VRAM table matters (see §4).

---

## 4. What each card omits that an operator needs

### Q3.6 — `Qwen/Qwen3.6-27B`

- **No VRAM/GPU sizing table.** Every command hardcodes 8-way TP with no statement of what 8
  GPUs, how much VRAM each, or what the minimum viable deployment is. The only sizing guidance
  is qualitative: "If you encounter out-of-memory (OOM) errors, consider reducing the context
  window."
- **No eager-vs-graph guidance.** No `--enforce-eager`, no CUDA-graph capture discussion, no
  note that GDN layers interact with graph capture.
- **No attention-backend guidance.** Compare [SGL-C], which makes `--attention-backend
  flashinfer` mandatory on SM120 and calls `trtllm_mha` SM100-only.
- **No KV-cache dtype guidance**, despite `--mem-fraction-static 0.8` implying the operator is
  expected to reason about memory.
- **No quantized variant pointers.** `Qwen/Qwen3.6-27B-FP8` exists (NV-3.6 links it as its own
  baseline) and Q3.6 never mentions it.
- **No throughput or latency numbers** for any configuration, while asserting "Inference
  efficiency and throughput vary significantly across frameworks."
- **No `min_p` / `repetition_penalty` support matrix**, though it prescribes both (M3).

### Q3.8 — `Qwen/Qwen3.8-27B`

Everything above, plus a strictly larger set because Q3.8 **regressed** relative to Q3.6:

- **No launch command at all.** Q3.6 printed seven complete commands; Q3.8 prints zero and
  delegates to three external recipes. An operator reading only the card cannot start a server.
- **No engine version floors.** Q3.6 pinned `sglang>=0.5.10` and `vllm>=0.19.0`; Q3.8 says
  "latest". There is no way to know which release first supported `Qwen3_5ForConditionalGeneration`
  or `reasoning_effort`.
- **No reasoning-parser or tool-parser flag**, despite shipping a tool-calling template and
  documenting `reasoning_content` / `reasoning` stream fields in its own SDK example. Without
  `--reasoning-parser qwen3` the example's `delta.reasoning_content` branch never fires.
- **No `--language-model-only` equivalent**, though `config.json` still carries
  `language_model_only: false`.
- **No MTP guidance**, though the overview advertises the MTP head.
- **Best Practices "Standardize Output Format" dropped.** The section heading occurs once in
  Q3.6 and zero times in Q3.8. The `\boxed{}` math prompt survives only as a MathVision
  *benchmark footnote* ("Qwen3.8-27B is evaluated using the fixed prompt: "Please reason step by
  step, and put your final answer within `\boxed{}`."") rather than as serving guidance, and
  Q3.6's multiple-choice scaffold (the "`answer` field" JSON instruction) is absent entirely, so
  the published numbers are not reproducible from the card's own recommendations.
- **Coding-specific sampling preset dropped** (Q3.6's `temperature=0.6` for WebDev).
- **`reasoning_effort` cost model unquantified**: three levels named, no token-count or latency
  figures, only the qualitative warning at line 267.
- The 262,144 + 131,072 output split is presented without flagging that it needs the YaRN
  override first (M6).

### NV-3.6 — `nvidia/Qwen3.6-27B-NVFP4`

- **No KV-cache disclosure.** `kv_cache_quant_algo: "FP8"` is in the repo and absent from the
  card (§3.1). This is the single most consequential omission: it changes accuracy, memory and
  comparability, and it is enabled by default.
- **No `--kv-cache-dtype`, no `--tensor-parallel-size`, no `--tool-call-parser`, no
  `--enable-auto-tool-choice`, no MTP flags** in its one command — even though it preserved the
  MTP head specifically (`exclude_modules: ["mtp*", "mtp.layers.0*"]`).
- **No VRAM table.** "reducing the disk size and GPU memory requirements by approximately 2.5x"
  is the entire memory story. No KV-cache-at-262K figure, no per-GPU minimum, no statement that
  21.92 GB weights + 262K KV does or does not fit a named card. Tested only on GB300.
- **No mixed-precision map.** The card says "NVFP4" throughout; the checkpoint is 193 NVFP4
  tensors + 208 FP8 tensors + BF16 embeddings/vision/MTP. `MIXED_PRECISION` appears only inside
  `hf_quant_config.json`, never on the card.
- **Overclaims activation quantization** (§3.2) — a reader budgeting for W4A4 kernels will size
  the wrong hardware.
- **No vision-tower precision statement**, while declaring Image and Video input support.
- **No BF16 baseline** in the accuracy table, **no eval harness name**, no `n`, no error bars
  (§3.8).
- **No KLD or perplexity evidence** — nine aggregate task scores only, all against FP8.
- **No eager-vs-graph guidance, no attention-backend guidance.**
- **Version pin is a moving target**: "start the docker `vllm/vllm-openai:nightly`". A nightly
  tag is not a reproducible dependency, and no minimum vLLM release is given.
- **`processor_config.json` is undocumented and contradicts `preprocessor_config.json`** on
  fast-vs-slow image processor (§1.5).

### UN-3.8 — `unsloth/Qwen3.8-27B-NVFP4`

This card omits nearly everything specific to the artifact it describes. It is, in substance,
Q3.8's card text with an Unsloth header prepended and Q3.8's serving sections removed.

- **No mention that the model is quantized.** The word NVFP4 appears in the repo name and
  nowhere in the card body. `base_model: [Qwen/Qwen3.8-27B]` and `tags: [unsloth]` are the only
  front-matter signals. A reader who does not parse the repo name will believe this is BF16.
- **No quantization recipe**: no W4A4, no group size 16, no compressed-tensors, no version, no
  layer map, no note that layers 56–63 are FP8 while 0–55 are NVFP4, no note that `lm_head` is
  FP8 and the vision tower and MTP head are BF16.
- **No KV-cache disclosure**, despite a baked static FP8 `kv_cache_scheme` (§3.1).
- **No calibration corpus**, despite `observer: "imatrix_mse"` and two `static_minmax`
  observers proving one exists (§3.6).
- **No accuracy evidence of any kind** for this checkpoint — no KLD, no perplexity, no MMLU, no
  BF16 comparison. The only benchmark link is to GGUF Dynamic 2.0 results for other models.
- **No GPU requirement**, despite W4A4 needing FP4 tensor cores (§3.3). No Blackwell note, no
  SM100/SM120 note, no statement of what happens on Ampere or Hopper.
- **No launch command, no engine, no version floor, no TP guidance, no `--quantization` value,
  no parser flags, no `--max-model-len`.** The card contains zero shell commands.
- **No VRAM table**, no size figure at all — not even the 23.42 GB on disk.
- **No changelog for the config and template edits it made**: `vision_config.model_type`
  `qwen3_5` → `qwen3_5_vision`, hoisted root-level `head_dim`/`num_attention_heads`/
  `num_key_value_heads`, dropped `chat_template` from `tokenizer_config.json`, and five template
  behaviour changes of which two are described and three are silent (§3.9).
- **No note on the `model.safetensors` + two-file-index layout** or the separate
  `model_mtp.safetensors` (§3.5).
- **No MTP guidance**, despite preserving the head via `"re:^mtp.*"` in `ignore`.
- **No eager-vs-graph, no attention-backend, no KV-cache-dtype, no thinking-mode switch
  example** (it deleted Q3.8's "Instruct (or Non-Thinking) Mode" and "Disable Preserved
  Thinking" sections while keeping the sentences that reference their behaviour).
- Retains Q3.8's Best Practices §4 video advice **minus** the engine-override alternative and
  its two PR links, so the only documented path is editing a file in the repo.

---

## 5. Inherited constraints for a derived quant

Requirements any derivative of `Qwen/Qwen3.8-27B` must carry forward, each with the evidence
that fixes it. NVIDIA's Qwen3.6 checkpoint satisfies all of these by passing artifacts through
unmodified; unsloth's Qwen3.8 checkpoint modifies items 1 and 6 and drops item 1's
`tokenizer_config.json` copy.

1. **Chat template — must be shipped, and shipped in both places.** `chat_template.jinja`
   (8,952 B, md5 `519239a4908bb1f805bbce5fa8c8a242`) is the contract for `enable_thinking`
   (default on: `{%- if enable_thinking is undefined or enable_thinking is true %}`),
   `preserve_thinking` (default on per Q3.8 line 264), `reasoning_effort` ∈
   {`xhigh`, `medium`, `low`} with `xhigh` the default and anything else raising, the
   `<think>\n…</think>\n\n` delimiters, and the `<tool_call><function=…><parameter=…>` payload
   shape that `--tool-call-parser qwen3_coder` decodes. Upstream ships it **twice** — as
   `chat_template.jinja` and as the `chat_template` key inside `tokenizer_config.json`. A
   derivative that keeps only one breaks whichever loader reads the other. If the template is
   edited at all, the edit must be documented flag-by-flag, because `--reasoning-parser qwen3`
   and `--tool-call-parser qwen3_coder` are parsing its literal output.
2. **mrope / vision preprocessing — bit-exact.** `text_config.rope_parameters` must retain
   `mrope_interleaved: true`, `mrope_section: [11, 11, 10]`, `partial_rotary_factor: 0.25`,
   `rope_theta: 10000000`, `rope_type: "default"`; `config.json` must retain
   `image_token_id: 248056`, `video_token_id: 248057`, `vision_start_token_id: 248053`,
   `vision_end_token_id: 248054`. `preprocessor_config.json`
   (`longest_edge: 16777216`, `shortest_edge: 65536`, `patch_size: 16`,
   `temporal_patch_size: 2`, `merge_size: 2`, `image_mean/std: [0.5,0.5,0.5]`,
   `processor_class: "Qwen3VLProcessor"`) and `video_preprocessor_config.json`
   (`longest_edge: 25165824`, `shortest_edge: 4096`,
   `video_processor_type: "Qwen3VLVideoProcessor"`) must be carried over verbatim, and the
   Best Practices §4 `longest_edge: 469762048` retune advice must be carried with them or the
   derived card silently loses hour-scale video capability. The 27-block vision tower
   (`vision_config.depth: 27`, `out_hidden_size: 5120`) should be excluded from quantization —
   and, unlike both existing quant cards, that exclusion should be *stated*.
3. **Generation defaults.** `generation_config.json` must keep `bos_token_id: 248044`,
   `eos_token_id: [248046, 248044]` (**both** — a single-EOS derivative will not stop on
   `<|im_end|>`), `pad_token_id: 248044`, `do_sample: true`, and the thinking-mode triple
   `temperature: 1.0`, `top_k: 20`, `top_p: 0.95`. Because the shipped config encodes only the
   thinking preset, a derived card must reprint the non-thinking preset
   (`temperature=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`) *and* say that the file
   does not contain it (M1) — this is the mismatch every one of the four cards leaves the
   operator to discover.
4. **Tokenizer / vocab.** `text_config.vocab_size: 248320` (padded) with
   `tie_word_embeddings: false` at both root and `text_config` level. `tokenizer.json` must
   retain `model.vocab` length 248,044 and the 33 `added_tokens` of the Qwen3.8 generation —
   Qwen3.6 has only 26, so a Qwen3.6-derived tokenizer is **not** substitutable. Because
   `tie_word_embeddings: false`, `lm_head` is a real 248,320 × 5,120 tensor and a separate
   quantization decision: NVIDIA made it NVFP4 W4A16, unsloth made it FP8, upstream is BF16.
   Whatever a derivative chooses, it must say so, since neither existing quant card does.
5. **MTP presence.** `text_config.mtp_num_hidden_layers: 1` with
   `mtp_use_dedicated_embeddings: false`. The head must either be preserved unquantized — both
   quantizers did, via `exclude_modules: ["mtp*", "mtp.layers.0*"]` (modelopt) and
   `"re:^mtp.*"` in `ignore` (compressed-tensors) — or its removal must be declared, because
   speculative decoding silently stops working otherwise. A derived card should print the
   engine flags, and print all three current spellings, since they are not interchangeable:
   vLLM `--speculative-config '{"method":"qwen3_next_mtp",...}'` (Q3.6 card) or
   `'{"method":"mtp",...}'` ([vLLM-R]), SGLang `--speculative-algorithm EAGLE
   --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
   ([SGL-C], `NEXTN` being an alias). Ship MTP weights in the ordinary shard sequence rather
   than a side file (§3.5).
6. **Hybrid-attention geometry is not negotiable.** 64 layers with `full_attention_interval: 4`
   → 48 `linear_attention` + 16 `full_attention`, `head_dim: 256`, 24 Q / 4 KV heads,
   `linear_num_key_heads: 16` / `linear_num_value_heads: 48` / `linear_key_head_dim: 128` /
   `linear_value_head_dim: 128`, `linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: "float32"`,
   `intermediate_size: 17408`, `attn_output_gate: true`, `output_gate_type: "swish"`. Keep
   `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"` and
   `text_config.model_type: "qwen3_5_text"`; changing `vision_config.model_type` (as unsloth did
   → `"qwen3_5_vision"`) is a loader-visible edit that belongs in the card. The GDN state pool
   is a first-class memory consumer that no card mentions — [SGL-C] sizes it at "48 GDN layers
   x 48 heads x 128 x 128 at `--mamba-ssm-dtype`, plus bf16 conv state: 153.9 MB at fp32,
   78.4 MB at bf16" per slot — so a derived card that publishes a VRAM table must include it.
7. **Context and the 1M path.** `max_position_embeddings: 262144` native. If the derived card
   repeats Q3.8's 262,144 + 131,072 output split, it must also carry the YaRN procedure and say
   that the split requires it (M6). Note the two incompatible 1M recipes in circulation
   (Q3.8's `rope_parameters`-based override vs [vLLM-R]'s `max_position_embeddings` override)
   and pick one explicitly.
8. **Whatever the derived quant does to the KV cache must be on the card.** Both existing
   quantizers changed KV precision and neither said so. This is the clearest inherited lesson
   rather than an inherited constraint: if `kv_cache_quant_algo` / `kv_cache_scheme` is set,
   engines honour it silently by default ([SGL-C]), so accuracy numbers taken against a BF16
   baseline are not comparable unless the cache precision is stated alongside them.

---

## Appendix — evidence index

Local files read (no network): `README.md`, `config.json`, `generation_config.json`,
`preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja`,
`tokenizer_config.json`, `tokenizer.json`, `model.safetensors.index.json` in each of
`/var/tmp/models/{Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-27B-NVFP4, Qwen3.8-27B-NVFP4}`, plus
`Qwen3.6-27B-NVFP4/hf_quant_config.json` and `Qwen3.6-27B-NVFP4/processor_config.json`.

Live pages fetched: [vLLM-R] and [SGL-C] as listed at the top, both reached via links printed
on the Q3.8 card. No GPU work was run, no packages installed, no servers started, and nothing
under `/home/mbelleau/qwen38-27b` was modified.
