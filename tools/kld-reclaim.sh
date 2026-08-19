#!/usr/bin/env bash
# Archive every KLD report into the repo, then delete captures whose report
# exists. A capture is 10 GB and fully regenerable from its checkpoint; a report
# is ~KB and is the evidence. Reports live in BOTH /tmp/kld-data/reports/ and
# (historically) /tmp/kld-data/ - check both. Never touches reference/hidden-bf16,
# lm-head/ or suite/.
set -uo pipefail
K=/tmp/kld-data
R="$(cd "$(dirname "$0")/.." && pwd)/receipts/kld-reports"
mkdir -p "$R"
for f in "$K"/reports/report-*.json "$K"/report-*.json; do
  [ -s "$f" ] || continue
  b=$(basename "$f"); [ -s "$R/$b" ] || { cp "$f" "$R/$b"; echo "archived $b"; }
done
for d in "$K"/captures/shard-0000/hidden-*; do
  [ -d "$d" ] || continue
  tag=$(basename "$d" | sed 's/^hidden-//')
  if [ -s "$K/reports/report-$tag.json" ] || [ -s "$K/report-$tag.json" ]; then
    sz=$(du -sh "$d" | cut -f1); rm -rf "$d"; echo "reclaimed $tag ($sz)"
  fi
done
df -h / | tail -1 | awk '{print "disk free:", $4}'
