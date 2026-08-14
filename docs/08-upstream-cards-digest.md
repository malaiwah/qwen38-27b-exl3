# Upstream Card Digest — Qwen3.6-27B / Qwen3.8-27B and their NVFP4 derivatives

One section per model card. Every bullet is traceable to the card text of the repository named
in that section; quoted strings are verbatim. Claims not present on the card are labelled
"not declared" rather than filled in from elsewhere.

---

## 1. `Qwen/Qwen3.6-27B`

**Source URL:** <https://huggingface.co/Qwen/Qwen3.6-27B>
(card read from `/var/tmp/models/Qwen3.6-27B/README.md`, 62,593 B, 1,010 lines)

### License

- Front matter: `license: apache-2.0`,
  `license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE`.
- `LICENSE` file present in the repo (11,343 B).

### Declared architecture support

- `library_name: transformers`, `pipeline_tag: image-text-to-text`.
- "This repository contains model weights and configuration files for the post-trained model in
  the Hugging Face Transformers format."
- "These artifacts are compatible with Hugging Face Transformers, vLLM, SGLang, KTransformers,
  etc."
- "Type: Causal Language Model with Vision Encoder"; "Training Stage: Pre-training &
  Post-training".
- Language model: "Number of Parameters: 27B"; "Hidden Dimension: 5120"; "Token Embedding:
  248320 (Padded)"; "Number of Layers: 64"; "Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) →
  1 × (Gated Attention → FFN))".
- Gated DeltaNet: "Number of Linear Attention Heads: 48 for V and 16 for QK"; "Head Dimension:
  128".
- Gated Attention: "Number of Attention Heads: 24 for Q and 4 for KV"; "Head Dimension: 256";
  "Rotary Position Embedding Dimension: 64".
- "Feed Forward Network: Intermediate Dimension: 17408"; "LM Output: 248320 (Padded)";
  "MTP: trained with multi-steps".
- "Context Length: 262,144 natively and extensible up to 1,010,000 tokens."
- Engine version floors: "`sglang>=0.5.10` is recommended for Qwen3.6"; "`vllm>=0.19.0` is
  recommended for Qwen3.6"; "The latest `transformers` is required for Qwen3.6".
- Named serving paths with commands: SGLang (`python -m sglang.launch_server`), vLLM
  (`vllm serve`), Hugging Face Transformers (`transformers serve … --continuous-batching`).
  KTransformers named with a link only:
  <https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3.5.md>.
- Agentic integrations declared: Qwen-Agent (<https://github.com/QwenLM/Qwen-Agent>, config
  `'model_type': 'qwenvl_oai'`, `'use_raw_api': True`) and Qwen Code
  (<https://github.com/QwenLM/qwen-code>).

### Declared quantization recipe

- None. This is the BF16 post-trained release; the card contains no quantization section and
  names no quantized variant.

### Declared benchmarks, with harness names as printed

Language / text tables name: MMLU-Pro, MMLU-Redux, C-Eval, SuperGPQA, GPQA Diamond, HLE,
AIME26, HMMT Feb 25, HMMT Nov 25, HMMT Feb 26, IMOAnswerBench, LiveCodeBench v6,
SWE-bench Verified, SWE-bench Pro, SWE-bench Multilingual, NL2Repo, Terminal-Bench 2.0,
SkillsBench, QwenWebBench, QwenClawBench, Claw-Eval. Vision-language tables name: MMMU,
MMMU-Pro, MathVista, MathVision-adjacent entries, CharXiv (RQ), DynaMath, CountBench, V*,
VlmsAreBlind, RealWorldQA, SimpleVQA, MMBench, MMStar, OCRBench, CC-OCR, RefCOCO,
RefSpatialBench, EmbSpatialBench, ERQA, AndroidWorld, MVBench, MLVU, VideoMME, VideoMMMU.
Comparison columns: Qwen3.6-27B, Qwen3.5-27B, Qwen3.6-35B-A3B, Qwen3.5-397B-A17B,
Gemma4-31B, Claude 4.5 Opus.

Harness / protocol footnotes, verbatim:

- "SWE-Bench Series: Internal agent scaffold (bash + file-edit tools); temp=1.0, top_p=0.95,
  200K context window. We correct some problematic tasks in the public set of SWE-bench Pro and
  evaluate all baselines on the refined benchmark."
- "NL2Repo: Others are evaluated via Claude Code (temp=1.0, top_p=0.95, max_turns=900)."
- "Terminal-Bench 2.0: Harbor/Terminus-2 harness; 3h timeout, 32 CPU/48 GB RAM; temp=1.0,
  top_p=0.95, top_k=20, max_tokens=80K, 256K ctx; avg of 5 runs."
- "SkillsBench: Evaluated via OpenCode on 78 tasks (self-contained subset, excluding
  API-dependent tasks); avg of 5 runs."
- "QwenWebBench: An internal front-end code generation benchmark; bilingual (EN/CN), 7
  categories (Web Design, Web Apps, Games, SVG, Data Visualization, Animation, and 3D);
  auto-render + multimodal judge (code/visual correctness); BT/Elo rating system."
- "QwenClawBench: A real-user-distribution Claw agent benchmark; temp=0.6, 256K ctx."
- "AIME 26: We use the full AIME 2026 (I & II), where the scores may differ from Qwen 3.5 notes."
- "Empty cells (--) indicate scores not yet available or not applicable."

### Declared limitations and caveats

- "Qwen3.6 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`."
- "Inference efficiency and throughput vary significantly across frameworks. We recommend using
  the latest framework versions to ensure optimal performance and compatibility. For production
  workloads or high-throughput scenarios, dedicated serving engines such as SGLang,
  KTransformers or vLLM are strongly recommended."
- "The model has a default context length of 262,144 tokens. If you encounter out-of-memory
  (OOM) errors, consider reducing the context window. However, because Qwen3.6 leverages
  extended context for complex tasks, we advise maintaining a context length of at least 128K
  tokens to preserve thinking capabilities."
- "All the notable open-source frameworks implement static YaRN, which means the scaling factor
  remains constant regardless of input length, **potentially impacting performance on shorter
  texts.** We advise modifying the `rope_parameters` configuration only when processing long
  contexts is required. It is also recommended to modify the `factor` as needed. For example, if
  the typical context length for your application is 524,288 tokens, it would be better to set
  `factor` as 2.0."
- On `presence_penalty`: "using a higher value may occasionally result in language mixing and a
  slight decrease in model performance."
- "To optimize inference efficiency for plain text and images, the `size` parameter in the
  released `video_preprocessor_config.json` is conservatively configured."
- Video frame-sampling `extra_body` control: "This feature is currently supported only in vLLM."
- Transformers path: "Hugging Face Transformers contains a _lightweight_ server which can be
  used for quick testing and moderate load deployment." and "Please also make sure torchvision
  and pillow are installed."
- No safety, bias, toxicity or misuse section on the card.
- No hardware requirement, VRAM figure, throughput number or GPU count justification anywhere,
  despite every command using 8-way tensor parallelism.

### Citation as printed

`@misc{qwen3.6-27b, title = {{Qwen3.6-27B}: Flagship-Level Coding in a {27B} Dense Model}, …}`

---

## 2. `Qwen/Qwen3.8-27B`

**Source URL:** <https://huggingface.co/Qwen/Qwen3.8-27B>
(card read from `/var/tmp/models/Qwen3.8-27B/README.md`, 65,012 B, 582 lines)

### License

- Front matter: `license: apache-2.0`. **No `license_link`** (unlike Qwen3.6-27B).
- `LICENSE` file present in the repo (11,544 B).

### Declared architecture support

- `library_name: transformers`, `pipeline_tag: image-text-to-text`.
- "This repository contains model weights and configuration files for the post-trained model in
  the Hugging Face Transformers format."
- "These artifacts are compatible with Hugging Face Transformers, vLLM, SGLang, TokenSpeed, etc."
  — KTransformers dropped, TokenSpeed added, relative to Qwen3.6-27B.
- "Type: Causal Language Model with Vision Encoder"; "Training Stage: Pre-training &
  Post-training".
- "Number of Parameters: 27B"; "Hidden Dimension: 5120"; "Token Embedding: 248,320 (Padded)";
  "Number of Layers: 64"; "Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated
  Attention → FFN))"; Gated DeltaNet "48 for V and 16 for QK", "Head Dimension: 128"; Gated
  Attention "24 for Q and 4 for KV", "Head Dimension: 256", "Rotary Position Embedding
  Dimension: 64"; "Intermediate Dimension: 17,408"; "LM Output: 248,320 (Padded)";
  "MTP (Multi-Token Prediction): trained with multiple steps".
- "Context Length: 262,144 natively and extensible up to 1,000,000 tokens."
- "Built on the architectural foundation of Qwen3.5".
- **No engine version floors.** Deployment is delegated to three linked recipes:
  - "[SGLang](https://www.sglang.io/): [Qwen3.8 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)"
  - "[vLLM](https://vllm.ai/): [Qwen3.8 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)"
  - "[TokenSpeed](https://lightseek.org/tokenspeed/): [Qwen3.8 Recipe](https://lightseek.org/tokenspeed/recipes/models#qwen3-8)"
- Hosted option declared: "the official Qwen API service is provided by
  [Qwen Cloud](https://www.qwencloud.com)"; "**Qwen3.8-27B** will be available as a hosted
  version with more production features, e.g., 1M context length by default, official built-in
  tools. … The service is coming soon."
- Declared control surface: "Thinking mode is on by default and can be disabled per request;
  reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical
  messages is retained via `preserve_thinking`." `reasoning_effort` levels: "`xhigh` (default):
  for complex tasks demanding thorough analysis"; "`medium`: balancing accuracy and speed";
  "`low`: efficient reasoning optimizing for speed and cost". "`preserve_thinking` is enabled by
  default for all workloads".
- Highlights also claim: "**Downstream Compatibility**: Broader support for popular harnesses
  and development tools".

### Declared quantization recipe

- None. BF16 post-trained release; no quantization section and no quantized variant named on the
  card.

### Declared benchmarks, with harness names as printed

Text table names: GPQA Diamond, HLE, Agents' Last Exam, IFBench, LiveCodeBench v6,
SWE-bench Pro, DeepSWE 1.1, QwenSWEBench, NL2Repo-Bench, RecreationBench, CoWorkBench,
JobBench, OSWorld-Verified, AndroidWorld. VL table names: MathVision, BabyVision, CharXiv (RQ),
RealWorldQA, ERQA, OmniDocBench 1.5, SWE-MM, ClawEval-MM. Capability row labels include
"Multidisciplinary reasoning", "Scientific reasoning", "Agentic coding", "Agentic terminal
coding", "Competitive coding", "Repo-level code generation", "Long-horizon office work",
"Professional job tasks", "Computer use", "Mobile use", "Browser use", "Application
recreation", "Document intelligence", "Real-world perception", "General visual reasoning",
"Scientific chart analysis", "Embodied intelligence", "Multimodal software engineering",
"Multimodal tool use". Comparison columns: Qwen3.8-27B, Qwen3.6-27B, Qwen3.7-Plus,
Muse Glimmer-30B, Opus4.6 Max.

Harness / protocol footnotes, verbatim:

- "SWE-bench Pro: Except for Opus4.6 Max, which uses the officially reported score, all models
  are evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window.
  Problematic tasks were corrected, and all baseline models were re-evaluated on the refined
  benchmark."
- "DeepSWE 1.1: Evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K
  context window."
- "QwenSWEBench: In-house coding benchmark for evaluating models' software engineering
  capabilities. Evaluated with the Claude Code harness. Reporting avg@3 with an 8-hour timeout,
  max_tokens=32,768, temperature=1.0, and a 256K context window."
- "NL2Repo-Bench: Evaluated with the Claude Code harness. To prevent reward hacking, we disable
  Bash commands that attempt to access the specific repository, such as pip download, pip
  install, and git clone."
- "SWE-MM: Scores are evaluated on the Claude Code harness using the public dev split of
  SWE-bench Multimodal, with the modifications described in Appendix 8.3 of the Claude Opus 4.7
  system card."
- "HLE: Judged by GPT-4o."
- "ClawEval-MM: Scores are reported as "Pass@3 / average score." Pass@3 is the percentage of
  tasks passed in at least one of three trials; the average score is the mean benchmark score
  across the three trials."
- "RecreationBench: An in-house, long-horizon application-recreation benchmark designed to
  evaluate hybrid-agent capabilities across five platforms: desktop (Ubuntu, macOS, and
  Windows), mobile (Android), and the web."
- "CoWorkBench: In-house cowork benchmark for evaluating long-horizon tasks across computer
  science, finance, law, medical, and other productivity domains."
- "MathVision: Qwen3.8-27B is evaluated using the fixed prompt: "Please reason step by step, and
  put your final answer within …"" ; "MathVision, BabyVision, and CharXiv (RQ): Where both
  settings are available, cells report "Without CI" and "With CI" separately … A small number of
  incorrect ground-truth annotations in MathVision and CharXiv (RQ) were corrected following
  manual verification, and all reported scores on those benchmarks were computed using the
  corrected annotations."
- "Empty cells (--) indicate that results are not yet available or not applicable."

### Declared limitations and caveats

- "Inference efficiency and throughput vary significantly across frameworks. We recommend using
  the latest framework versions to ensure optimal performance and compatibility. For production
  workloads or high-throughput scenarios, dedicated serving engines such as SGLang, vLLM, or
  TokenSpeed are recommended."
- "Please note that the support for sampling parameters varies according to inference
  frameworks."
- "In multi-turn agentic tasks, lower reasoning effort does not always reduce overall task
  completion time. Although it may produce faster per-turn responses, it can also lead to
  insufficient analysis, more failures, and repeated retries, which may increase total latency
  and token consumption."
- Same static-YaRN warning as Qwen3.6-27B, verbatim, including "potentially impacting performance
  on shorter texts."
- Same `presence_penalty` language-mixing warning.
- Same conservative `video_preprocessor_config.json` caveat, with the added escape hatch
  "Alternatively, override the default values via engine startup parameters. For implementation
  details, refer to: [vLLM](https://github.com/vllm-project/vllm/pull/34330) /
  [SGLang](https://github.com/sgl-project/sglang/pull/18467)."
- Qwen Cloud API divergence called out twice: "please use `"enable_thinking": False` instead of
  `"chat_template_kwargs": {"enable_thinking": False}`" and "please use
  `"preserve_thinking": False` directly instead of wrapping it in `chat_template_kwargs`".
- No safety, bias or toxicity section. No hardware requirement, VRAM figure or throughput number.
- Relative to Qwen3.6-27B the card **drops**: all launch commands, all engine version floors,
  the KTransformers path, the `--language-model-only` text-only mode, the MTP serving commands,
  the coding-specific sampling preset, and the benchmark output-format prompts.

### Citation as printed

`@misc{qwen38, title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
url = {https://qwen.ai/blog?id=qwen3.8}, author = {{Qwen Team}}, month = {August}, year = {2026}}`
— note the card is for Qwen3.8-**27B** while its own BibTeX title says Qwen3.8-**Max**.

---

## 3. `nvidia/Qwen3.6-27B-NVFP4`

**Source URL:** <https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4>
(card read from `/var/tmp/models/Qwen3.6-27B-NVFP4/README.md`, 9,313 B)

### License

- Front matter: `license: apache-2.0`.
- "**GOVERNING DOWNLOAD TERMS:** Use of the model is governed by the
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)".
- No `LICENSE` file in the repo.
- Commercial posture declared: "This model is ready for commercial or non-commercial use."
- "**Deployment Geography:** Global"; "**Preferred Operating System(s):** Linux".
- "**Release Date:** Hugging Face 06/26/2026 via <https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4>".
- "**Use Case:** Developers looking to take off-the-shelf, pre-quantized models for deployment
  in AI Agent systems, chatbots, RAG systems, and other AI-powered applications."

### Declared architecture support

- Front matter: `pipeline_tag: text-generation`, `library_name: Model Optimizer`,
  `base_model: [Qwen/Qwen3.6-27B]`, tags `nvidia, ModelOpt, Qwen3.6, quantized, FP4, fp4`.
  Note the `text-generation` pipeline tag despite the declared image and video inputs, and
  despite upstream's `image-text-to-text`.
- "**Architecture Type:** Transformers"; "**Network Architecture:** Hybrid Attention (Gated
  DeltaNet and Gated Attention)"; "**Number of Model Parameters:** 27B".
- "**Input Type(s):** Text, Image, Video"; "**Input Format(s):** String, Red, Green, Blue (RGB),
  Video (MP4/WebM)"; "**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D),
  Three-Dimensional (3D)"; "**Other Properties Related to Input:** Context length up to 262K".
- "**Output Type(s):** Text"; "**Output Format:** String"; "**Output Parameters:** 1D
  (One-Dimensional): Sequences"; "**Other Properties Related to Output:** None".
- "**Supported Runtime Engine(s):** vLLM" — the only engine declared.
- "**Supported Hardware Microarchitecture Compatibility:** NVIDIA Hopper, NVIDIA Blackwell".
- "**Acceleration Engine:** vLLM"; "**Test Hardware:** NVIDIA GB300".
- Deployment instruction: "you can start the docker `vllm/vllm-openai:nightly` and run the
  sample command below", followed by the single command
  `vllm serve nvidia/Qwen3.6-27B-NVFP4 --port 8000 --quantization modelopt --max-model-len 262144 --reasoning-parser qwen3`.

### Declared quantization recipe, as the card states it

- "The model version is NVFP4 1.0 version and is quantized with nvidia-modelopt v0.45.0".
- "The NVIDIA Qwen3.6-27B NVFP4 model is quantized with
  [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)."
- "This model was obtained by quantizing the weights and activations of Qwen3.6-27B to NVFP4
  data type, ready for inference with vLLM."
- "Only the weights and activations of the linear operators within transformer blocks are
  quantized."
- "This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk
  size and GPU memory requirements by approximately 2.5x."
- **Calibration Dataset declared:** "[cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail),
  [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)";
  "**Data Collection Method by dataset:** Automated."; "**Labeling Method by dataset:**
  Automated."; properties described as "just over 300k unique news articles" (cnn_dailymail) and
  "a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse
  topics" (Nemotron-Post-Training-Dataset-v2).
- **Training Dataset:** every field declared "Undisclosed" — Data Modality, Data Collection
  Method, Labeling Method, Properties, Audio/Image/Text/Video Training Data Size.
- Not stated on the card: mixed precision, the FP8 component, group size, KV-cache
  quantization, `lm_head` treatment, vision-tower treatment, MTP exclusion.

### Declared benchmarks with harness names

- "**Datasets:** MMLU Pro, GPQA Diamond, HLE, τ²-Bench Telecom, MMMU Pro, SciCode, AIME 2025,
  AA-LCR, IFBench"; "**Data Collection Method by dataset:** Hybrid: Automated, Human";
  "**Labeling Method by dataset:** Hybrid: Human, Automated".
- Dataset descriptions given for each, e.g. "GPQA Diamond contains 448 graduate-level
  multiple-choice questions written by domain experts in biology, physics, and chemistry";
  "HLE (Humanity's Last Exam) is an expert-level academic benchmark with 2158 text-only
  questions"; "AA-LCR (Artificial Analysis Long Context Recall) evaluates a model's ability to
  accurately retrieve and recall information from long input contexts".
- Results table, FP8 → NVFP4: MMLU Pro 86.1 → 86.3; GPQA Diamond 86.0 → 85.5; HLE 21.7 → 21.8;
  τ²-Bench Telecom 95.2 → 95.4; MMMU Pro 74.6 → 74.3; SciCode 44.8 → 44.5;
  AIME 2025 93.1 → 92.7; AA-LCR 68.8 → 68.3; IFBench 65.1 → 65.5.
- Protocol footnote: "Baseline: [Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8).
  Benchmarked with temperature=1.0, top_p=0.95, max_num_tokens=81920 except SciCode with
  temperature=0.6, and τ²-Bench Telecom with temperature=0.0 and top_p=1.0."
- **No eval harness is named** — no runner, no version, no commit, no `n` or avg@k, no seeds.
  Compare the upstream cards, which do name harnesses (Harbor/Terminus-2, OpenCode, Claude Code,
  GPT-4o judge).
- The baseline is FP8, not BF16, so no BF16-relative degradation figure is declared.

### Declared limitations

- "**Model Limitations:** The base model was trained on data that contains toxic language and
  societal biases originally crawled from the internet. Therefore, the model may amplify those
  biases and return toxic responses especially when prompted with toxic prompts. The model may
  generate answers that may be inaccurate, omit key information, or include irrelevant or
  redundant text producing socially unacceptable or undesirable text, even if the prompt itself
  does not include anything explicitly offensive."
- "The integration of foundation and fine-tuned models into AI systems requires additional
  testing using use-case-specific data to ensure safe and effective deployment. Following the
  V-model methodology, iterative testing and validation at both unit and system levels are
  essential to mitigate risks, meet technical and functional requirements, and ensure compliance
  with safety and ethical standards before deployment."
- "Developers should work with their internal model team to ensure this model meets requirements
  for the relevant industry and use case and addresses unforeseen product misuse."
- "Please make sure you have proper rights and permissions for all input image and video content;
  if image or video includes people, personal health information, or intellectual property, the
  image or video generated will not blur or maintain proportions of image subjects included."
- Vulnerability reporting: <https://app.intigriti.com/programs/nvidia/nvidiavdp/detail>.
- "Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems."
- Not declared: VRAM requirement, minimum GPU, KV-cache dtype, tensor-parallel degree,
  tool-call parser, MTP/speculative decoding, eager-vs-graph, attention backend, minimum vLLM
  release (only the `:nightly` tag), sampling recommendations for serving (only for eval).

### Notes on this card's relationship to upstream

- `chat_template.jinja`, `tokenizer_config.json` and `generation_config.json` in this repo are
  byte-identical to `Qwen/Qwen3.6-27B` (md5 `52b6d51ae5b203cb67e64b648494dad2`,
  `dee7749bf0c52b1185a0b06fa1706724`, `efeaa00fbdbaaa33be52e9373c35da5f`). The card makes no
  claim either way; the identity is a repository fact, not a card claim.
- The card links back to upstream: "For more information, please check
  [here](https://huggingface.co/Qwen/Qwen3.6-27B)."
- Reference: "NVIDIA Model Optimizer: <https://github.com/NVIDIA/Model-Optimizer>".

---

## 4. `unsloth/Qwen3.8-27B-NVFP4`

**Source URL:** <https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>
(card read from `/var/tmp/models/Qwen3.8-27B-NVFP4/README.md`, 6,586 B)

### License

- Front matter: `license: apache-2.0`, `base_model: [Qwen/Qwen3.8-27B]`, `tags: [unsloth]`.
- No `license_link`, no `LICENSE` file in the repo, no license prose in the body.
- No `pipeline_tag` and no `library_name` in front matter.

### Declared architecture support

The card body from "# Qwen3.8-27B" onward is a reproduction of `Qwen/Qwen3.8-27B`'s Model
Overview and Best Practices with the serving and API sections removed. Declared:

- "Type: Causal Language Model with Vision Encoder"; "Training Stage: Pre-training &
  Post-training".
- "Number of Parameters: 27B"; "Hidden Dimension: 5120"; "Token Embedding: 248,320 (Padded)";
  "Number of Layers: 64"; "Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated
  Attention → FFN))"; Gated DeltaNet "48 for V and 16 for QK", "Head Dimension: 128"; Gated
  Attention "24 for Q and 4 for KV", "Head Dimension: 256", "Rotary Position Embedding
  Dimension: 64"; "Intermediate Dimension: 17,408"; "LM Output: 248,320 (Padded)";
  "MTP (Multi-Token Prediction): trained with multiple steps".
- "Context Length: 262,144 natively and extensible up to 1,000,000 tokens."
- "Built on the architectural foundation of Qwen3.5".
- "**Flexible Thinking Control**: Thinking mode is on by default and can be disabled per
  request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from
  historical messages is retained via `preserve_thinking`."
- "**Vision-Language Understanding**: Native support for image and video understanding, from
  STEM diagrams and documents to hour-scale videos."
- **No engine is named.** No vLLM, SGLang, TensorRT-LLM, transformers or llama.cpp mention; no
  version floor; no command of any kind.
- Unsloth-specific runtime claims: "Qwen3.8 can now be run and fine-tuned in
  [Unsloth Desktop](https://unsloth.ai/docs/new/desktop)"; "See below for 1-bit Qwen3.8 run
  inside of Unsloth"; "Developer Role Support so Qwen3.8 can work in agentic tools like Codex
  and more!"; "Tool calling improvements: Makes parsing nested objects to make tool calling
  succeed more."
- Guide links: "# Read our How to [Run Qwen3.8-27B Guide!](https://unsloth.ai/docs/models/qwen3.8)";
  <https://github.com/unslothai/unsloth/>; <https://discord.gg/unsloth>.

### Declared quantization recipe, as the card states it

- **None.** The card body never mentions NVFP4, FP4, 4-bit, W4A4, compressed-tensors, group
  size, calibration, or any precision for this checkpoint. The string "NVFP4" appears only in
  the repository name, which is not part of the card text. The word "quantization" occurs
  exactly once, in the GGUF-benchmarks pointer quoted below; the only bit-width mentioned
  anywhere is in "See below for 1-bit Qwen3.8 run inside of Unsloth", which refers to a
  different artifact.
- The only quantization-adjacent sentence is a pointer to a different format and different
  models: "*See [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf)
  for our quantization benchmarks.*"
- No calibration corpus, no producer/version, no layer map, no excluded-module list, no
  KV-cache statement, no GPU-architecture requirement, no size or VRAM figure.

### Declared benchmarks with harness names

- **None for this checkpoint.** The card contains no benchmark table and no accuracy, KLD or
  perplexity number. The Dynamic 2.0 GGUF link is the only benchmark reference and it does not
  cover this artifact.
- Upstream's entire "Benchmark Results" section (Text Performance and VL Performance tables plus
  all harness footnotes) was removed.

### Declared limitations

- No limitations, safety, bias, toxicity or misuse section of any kind.
- The only caveats present are the two inherited from upstream Best Practices:
  - `presence_penalty`: "using a higher value may occasionally result in language mixing and a
    slight decrease in model performance."
  - "To optimize inference efficiency for plain text and images, the `size` parameter in the
    released `video_preprocessor_config.json` is conservatively configured. It is recommended to
    set the `longest_edge` parameter in the video_preprocessor_config file to 469,762,048
    (corresponding to 224k video tokens)…" — reproduced **without** upstream's "Alternatively,
    override the default values via engine startup parameters" sentence or its two PR links.
- Inherited sampling recommendations are present verbatim: thinking mode `temperature=1.0`,
  `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`;
  instruct mode `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`,
  `presence_penalty=1.5`, `repetition_penalty=1.0`; output split 262,144 reasoning /
  131,072 final; YaRN recommended above 262,144 — but upstream's concrete YaRN command lines and
  static-YaRN warning are **not** reproduced, only the sentence "we recommend using RoPE scaling
  techniques to handle long texts effectively, e.g., YaRN."
- The card retains upstream's `reasoning_effort` and `preserve_thinking` claims while deleting
  the "Instruct (or Non-Thinking) Mode" and "Disable Preserved Thinking" sections that showed how
  to set them, so no mechanism for either is documented.

### Citation as printed

`@misc{qwen38, title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
url = {https://qwen.ai/blog?id=qwen3.8}, author = {{Qwen Team}}, month = {August}, year = {2026}}`
— upstream's BibTeX carried over unchanged, crediting the Qwen Team only; no citation is offered
for the quantization work itself.
