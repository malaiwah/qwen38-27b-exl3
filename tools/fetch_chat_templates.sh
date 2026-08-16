#!/bin/bash
# Fetch chat templates (chat_template.jinja and/or tokenizer_config.json's
# chat_template field) from HF repos, recording revision sha and file sha256.
# Read-only: writes only under receipts/chat-templates/.
set -uo pipefail
OUT="${1:-/home/mbelleau/research/receipts/chat-templates}"
TOK="$(cat /var/tmp/hf-home-3/token 2>/dev/null)"
mkdir -p "$OUT/raw"

auth=()
[ -n "$TOK" ] && auth=(-H "Authorization: Bearer $TOK")

repos=(
  "Qwen/Qwen3.8-27B"
  "Qwen/Qwen3.8-27B-FP8"
  "Qwen/Qwen3.8-2.4T-A95B"
  "Qwen/Qwen3.6-27B"
  "unsloth/Qwen3.8-27B-GGUF"
  "unsloth/Qwen3.8-27B-NVFP4"
  "turboderp/Qwen3.8-27B-exl3"
  "nvidia/Qwen3.6-27B-NVFP4"
  "RedHatAI/Qwen3.8-2.4T-A95B"
  "RedHatAI/Qwen3.8-2.4T-A95B-FP8"
  "RedHatAI/Qwen3.6-27B-FP8"
  "froggeric/Qwen-Fixed-Chat-Templates"
  "malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated"
  "malaiwah/Qwen3.8-27B-EXL3-K5K6"
  "malaiwah/Qwen3.8-27B-EXL3-K5K6-context"
  "malaiwah/Qwen3.8-27B-K4"
)

# repo:branch pairs whose template lives off the default branch, and repo:path
# pairs for files that are not at the repo root.
extra_refs=(
  "turboderp/Qwen3.8-27B-exl3:5.00bpw:chat_template.jinja"
  "turboderp/Qwen3.8-27B-exl3:5.00bpw:tokenizer_config.json"
  "froggeric/Qwen-Fixed-Chat-Templates:main:archive/v21_chat_template.jinja"
  "froggeric/Qwen-Fixed-Chat-Templates:main:README.md"
  "froggeric/Qwen-Fixed-Chat-Templates:main:scripts/test_v22.py"
  # the superseded first revision of the 27B template; see docs/39 section 6b
  "Qwen/Qwen3.8-27B:72a217afab:chat_template.jinja"
)

get() {  # url dest
  local c
  c=$(curl -sSL "${auth[@]}" -o "$2" -w "%{http_code}" "$1" 2>/dev/null)
  if [ "$c" = "200" ]; then
    echo "  GOT $(basename "$2") $(sha256sum "$2" | cut -c1-64) $(wc -c <"$2")"
  else
    rm -f "$2"; echo "  MISS $(basename "$2") ($c)"
  fi
}

for repo in "${repos[@]}"; do
  slug="${repo//\//__}"
  info="$OUT/raw/$slug.info.json"
  code=$(curl -sS "${auth[@]}" -o "$info" -w "%{http_code}" \
         "https://huggingface.co/api/models/$repo" 2>/dev/null)
  echo "REPO $repo api=$code"
  [ "$code" != "200" ] && continue
  rev=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('sha',''))" "$info")
  lm=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('lastModified',''))" "$info")
  echo "  rev=$rev lastModified=$lm"
  for f in chat_template.jinja tokenizer_config.json chat_template.json; do
    get "https://huggingface.co/$repo/resolve/$rev/$f" "$OUT/raw/$slug.$f"
  done
done

for spec in "${extra_refs[@]}"; do
  repo="${spec%%:*}"; rest="${spec#*:}"; ref="${rest%%:*}"; path="${rest#*:}"
  slug="${repo//\//__}"
  # resolve a branch name to its commit so the receipt records an immutable ref
  sha="$ref"
  if curl -sSf "${auth[@]}" "https://huggingface.co/api/models/$repo/refs" \
       -o "$OUT/raw/$slug.refs.json" 2>/dev/null; then
    resolved=$(python3 - "$OUT/raw/$slug.refs.json" "$ref" <<'PY'
import json, sys
refs = json.load(open(sys.argv[1]))
want = sys.argv[2]
for b in refs.get("branches", []):
    if b["name"] == want:
        print(b["targetCommit"]); break
PY
)
    [ -n "$resolved" ] && sha="$resolved"
  fi
  echo "EXTRA $repo@$ref ($sha) $path"
  get "https://huggingface.co/$repo/resolve/$sha/$path" \
      "$OUT/raw/$slug@$ref.$(basename "$path")"
done
