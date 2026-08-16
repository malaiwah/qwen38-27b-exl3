#!/bin/bash
# Progress notifications to ntfy.sh.
#
# The topic name is a shared secret (ntfy has no auth: whoever knows the topic
# can read it), so it is NEVER stored in this repository. Supply it via:
#   NTFY_TOPIC env var, or
#   ~/.config/omp-ntfy-topic  (single line, topic name only)
#
# ntfy limits that shape this script (docs.ntfy.sh/publish/#limitations,
# read 2026-08-16): a message may be at most 4,096 bytes — longer bodies are
# silently converted into an *attachment* rather than a notification, so we
# truncate at 3,900 bytes and mark the cut. Rate limit is a 60-request burst
# refilling ~1 request / 5 s per visitor, so batch progress rather than
# emitting one notification per tool call.
#
# Usage: tools/notify.sh "Title" "body text" [tags] [priority]
set -uo pipefail

topic="${NTFY_TOPIC:-}"
if [[ -z "$topic" && -r "$HOME/.config/omp-ntfy-topic" ]]; then
  topic="$(tr -d '[:space:]' < "$HOME/.config/omp-ntfy-topic")"
fi
if [[ -z "$topic" ]]; then
  echo "notify: no topic (set NTFY_TOPIC or ~/.config/omp-ntfy-topic)" >&2
  exit 2
fi

title="${1:?title required}"
body="${2:?body required}"
tags="${3:-}"
prio="${4:-default}"

max=3900
if (( ${#body} > max )); then
  body="${body:0:$max}"$'\n[truncated at 3900 B — ntfy caps messages at 4096 B]'
fi

args=(-fsS -X POST "https://ntfy.sh/${topic}"
      -H "Title: ${title}"
      -H "Priority: ${prio}"
      -H "Markdown: yes"
      --data-binary "${body}")
[[ -n "$tags" ]] && args+=(-H "Tags: ${tags}")

out="$(curl "${args[@]}" 2>&1)"
rc=$?
if (( rc != 0 )); then
  echo "notify: publish failed rc=$rc: $out" >&2
  exit 1
fi
echo "notify: sent (${#body} B) — $(printf '%s' "$out" | head -c 120)"
