#!/bin/bash
# Terminal-Bench 2.1 speed-run campaign orchestrator -- the DRIVER-LOCAL variant.
#
# This is the clean VPC/docker sibling of tools/run_terminal_bench.sh (the
# AIBoss/tunnel/podman variant, which the in-flight owner-protocol passes still
# use -- do not touch it). Everything tunnel- and proot-shaped is gone: this
# script runs ON the load-driver VM (jl-vm-473296: 32 vCPU / 125 GB / 968 GB
# NVMe, docker 29.x, harbor 0.21.0 via uv, pinned task tree sha-verified at
# ~/tb21/tasks) and reaches vLLM replicas over the plain VPC (<10 ms RTT,
# G-NET passed for phase A at 0.3 ms).
#
# WHAT A TB SCORE IS HERE: every number is a score for an agent+model SYSTEM
# (Terminus-2 2.0.0 driving a vLLM OpenAI endpoint), never a model in
# isolation. Stock --timeout-multiplier 1.0 is the only comparable headline.
#
# THE LOAD-BALANCER IS THE SHARD LIST (docs/46 §5). No router in front of
# vLLM: tb21_shard.py bin-packs the 89 tasks into K shards by declared
# per-task timeout budget (greedy makespan), and pass1 runs K parallel
# `harbor run` jobs, each pinned to its own replica's api_base and subset to
# its shard with repeatable -i/--include-task-name flags (mechanism verified
# against `harbor run --help` on this harbor 0.21.0: -i takes a task name or
# glob and can be used multiple times). Shard affinity is what preserves the
# measured 51.7 % prefix-cache hit rate -- terminus-2 resends the whole
# transcript every turn, so a request must keep landing on the same replica.
#
# RESUME is harbor's, not ours: `harbor job resume` reconciles planned trials
# against completed ones (a trial dir without result.json is deleted and
# re-run; a completed trial is kept byte-identical). Job configs bake in the
# api_base and the -i set, so shard membership and replica URLs MUST NOT
# change across resumes of the same job -- which is why the shard packer is
# deterministic.
#
# Retry classes are TRANSPORT-ONLY, verbatim from
# receipts/terminal-bench-2.1-pins.json (retry_include_transport_only): a VPC
# hiccup or a replica restart must never score as the model failing a task,
# and AgentTimeoutError is deliberately absent -- running out of clock is a
# real TB failure mode and pass 2 retries it anyway.
#
# Usage (on the driver; stage tools/ + the inventory receipt under ~/tb21):
#   tb21_campaign.sh gate   <url> [args...]      # fidelity gate (tb21_gate.py)
#   tb21_campaign.sh ladder <url> [args...]      # synthetic knee (tb21_ladder.py)
#   tb21_campaign.sh shard  <K>                  # (re)pack shards/K<K>/
#   tb21_campaign.sh pass1  <model-tag> <K> <url...>   # K parallel shard jobs
#   tb21_campaign.sh resume <job-name>           # wrap harbor job resume
#   tb21_campaign.sh merge  <out.json> <jobdir...>     # one pass receipt
#   tb21_campaign.sh metrics-snap <url> <tag>    # verbatim /metrics snapshot
#   tb21_campaign.sh disk-guard                  # free-space + orphan report
#
# pass1 details:
#   <model-tag>  hyd -> served name qwen38, bf16 -> qwen38-bf16, anything else
#                is used literally (override with TB21_SERVED).
#   <K>          number of shards == number of replica URLs that follow.
#   <url...>     one base URL per shard, e.g. http://10.0.0.3:8000 (a /v1
#                suffix is appended if absent). Job names are
#                tb21c-<tier>-<model>-p1-s<i> under ~/tb21/jobs; tier from
#                TB21_TIER (default 1x). Re-running pass1 with job dirs
#                present resumes them (completed trials kept).
#
# Env knobs (defaults chosen for the docs/46 §4.1 campaign recipe):
#   TB21_HOME=~/tb21   TB21_TIER=1x   TB21_N=4 (per-shard -n, set from the
#   ladder knee!)   TB21_MAXLEN=262144 (feeds model_info; the pins' 32768 was
#   for the 32k window -- the campaign serves the native window, a named
#   deviation)   TB21_TIMEOUT_MULT=1.0   TB21_MAX_RETRIES=3
#   TB21_SHARD_DIR (default $TB21_HOME/shards/K<K>)   TB21_SERVED
#   TB21_RETRY_INCLUDE (comma list; default = the pins' transport-only set)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

TB21_HOME=${TB21_HOME:-$HOME/tb21}
TASKS=${TB21_TASKS:-$TB21_HOME/tasks}
JOBS=${TB21_JOBS:-$TB21_HOME/jobs}
TIER=${TB21_TIER:-1x}
N=${TB21_N:-4}
MAXLEN=${TB21_MAXLEN:-262144}
TIMEOUT_MULT=${TB21_TIMEOUT_MULT:-1.0}
MAX_RETRIES=${TB21_MAX_RETRIES:-3}
INVENTORY=${TB21_INVENTORY:-$TB21_HOME/receipts/terminal-bench-2.1-task-inventory.json}
HERE=$(cd "$(dirname "$0")" && pwd)

# Verbatim from receipts/terminal-bench-2.1-pins.json .harness_pins.retry_include_transport_only
RETRY_INCLUDE=${TB21_RETRY_INCLUDE:-APIConnectionError,APIError,InternalServerError,ServiceUnavailableError,Timeout,ConnectionError,RemoteProtocolError,EnvironmentStartError,EnvironmentStartTimeoutError,AgentSetupTimeoutError}

# LiteLLM ships no metadata for hosted_vllm/ names; without an explicit
# model_info the agent cannot compute remaining context and its summarisation
# trigger misfires (pins, model_info_required). max_input_tokens tracks the
# served window; cache_* fields only silence LiteLLM's register_model warning.
MODEL_INFO=$(printf '{"max_input_tokens":%s,"max_output_tokens":8192,"input_cost_per_token":0,"output_cost_per_token":0,"cache_creation_input_token_cost":0,"cache_read_input_token_cost":0}' "$MAXLEN")

usage() { sed -n '2,70p' "$0"; exit 1; }
die()   { echo "error: $*" >&2; exit 1; }

served_for() {           # model-tag -> served model name
  case $1 in
    hyd)  echo "${TB21_SERVED:-qwen38}" ;;
    bf16) echo "${TB21_SERVED:-qwen38-bf16}" ;;
    *)    echo "${TB21_SERVED:-$1}" ;;
  esac
}

v1() {                   # append /v1 to a base URL unless already present
  local u=${1%/}
  case $u in */v1) echo "$u" ;; *) echo "$u/v1" ;; esac
}

# ---------------------------------------------------------------- guards ----
# Disk guard, verbatim policy from the pins (disk_hazard): report free space
# before every pass; below 15 GB remove ONLY docker.io/alexgshaw/* task
# images -- never a global prune, never an image with a running container.
# The orphan sweep exists because it was observed (pins,
# orphan_leak_found_and_handled): SIGKILLing harbor mid-pass leaves the task
# container and its compose network behind. Sweeping is only safe when no
# pass is live -- a live pass's containers would match too -- so the sweep is
# skipped (with a warning) whenever a harbor process is running. That is also
# why `resume` does NOT call the guard: a sibling shard job may be mid-pass.
disk_guard() {
  local free_g
  free_g=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  echo "[disk] ${free_g}G free on driver \$HOME"
  if pgrep -f 'harbor (run|job)' >/dev/null 2>&1; then
    echo "[orphans] harbor is live -> sweep skipped (never sweep under a running pass)"
  else
    local orph=0 netn=0 c n
    for c in $(docker ps -a --format '{{.ID}} {{.Image}}' | awk '/alexgshaw/ {print $1}'); do
      docker rm -f "$c" >/dev/null 2>&1 && orph=$((orph+1))
    done
    for n in $(docker network ls --format '{{.Name}}' | grep '__env' || true); do
      docker network rm "$n" >/dev/null 2>&1 && netn=$((netn+1))
    done
    if [ "$orph" -gt 0 ] || [ "$netn" -gt 0 ]; then
      echo "[orphans] reclaimed $orph container(s), $netn compose network(s) left by an interrupted pass"
    else
      echo "[orphans] none"
    fi
  fi
  if [ "$free_g" -lt 15 ]; then
    echo "[disk] below 15G -> removing docker.io/alexgshaw/* task images ONLY (never a global prune); report over hub"
    local live_imgs
    live_imgs=$(docker ps --format '{{.Image}}' | sort -u)
    docker images --format '{{.ID}} {{.Repository}}' | awk '/alexgshaw/ {print $1, $2}' \
      | while read -r id repo; do
          echo "$live_imgs" | grep -q "$repo" && { echo "[disk] keep $repo (live container)"; continue; }
          docker rmi -f "$id" >/dev/null 2>&1 || true
        done
    df -h "$HOME" | tail -1
  fi
}

# ----------------------------------------------------------- gate / ladder ----
# Fidelity gate and synthetic concurrency ladder are owned by tb21_gate.py /
# tb21_ladder.py (same directory, LadderGate's contract: flags only, no
# positionals). The first arg here is the base URL, injected as --base-url;
# everything after passes through verbatim (--out, --api-key, --tp-smoke, ...).
# tb21_gate.py exits nonzero on any FAIL -- let that propagate.
gate()   { local url=${1:?usage: gate <url> [args...]}; shift
           [ -f "$HERE/tb21_gate.py" ]   || die "tb21_gate.py not staged next to this script"
           exec python3 "$HERE/tb21_gate.py" --base-url "$url" "$@"; }
ladder() { local url=${1:?usage: ladder <url> [args...]}; shift
           [ -f "$HERE/tb21_ladder.py" ] || die "tb21_ladder.py not staged next to this script"
           exec python3 "$HERE/tb21_ladder.py" --base-url "$url" "$@"; }

# ----------------------------------------------------------------- shards ----
shard() {
  local k=${1:?usage: shard <K>}
  [ -f "$INVENTORY" ] || die "inventory receipt not found at $INVENTORY"
  python3 "$HERE/tb21_shard.py" --inventory "$INVENTORY" --shards "$k" \
    --outdir "$TB21_HOME/shards/K$k"
}

# ------------------------------------------------------------------ pass 1 ----
# One harbor job per shard, launched in parallel, each in its own session
# (setsid) so a shard can be killed -- or die -- without taking siblings down,
# and so `kill -- -PID` reaches the whole job. PIDs land in $JOBS/<job>.pid,
# logs in $JOBS/<job>.log. A job dir with config.json is resumed instead of
# restarted; harbor enforces that the resumed config is identical.
pass1() {
  local model=${1:?usage: pass1 <model-tag> <K> <url...>} ; shift
  local k=${1:?usage: pass1 <model-tag> <K> <url...>} ; shift
  [ $# -eq "$k" ] || die "pass1: K=$k but $# replica URL(s) given"
  local urls=("$@")
  local served; served=$(served_for "$model")
  local sdir=${TB21_SHARD_DIR:-$TB21_HOME/shards/K$k}

  if [ ! -f "$sdir/shard-0.txt" ]; then
    echo "[pass1] no shards at $sdir -> packing"
    shard "$k"
  fi
  local i
  for i in $(seq 0 $((k-1))); do
    [ -s "$sdir/shard-$i.txt" ] || die "missing or empty $sdir/shard-$i.txt"
  done

  mkdir -p "$JOBS"
  disk_guard

  local pids=() jobs=()
  for i in $(seq 0 $((k-1))); do
    local job="tb21c-$TIER-$model-p1-s$i"
    local url; url=$(v1 "${urls[$i]}")
    local cmd
    if [ -f "$JOBS/$job/config.json" ]; then
      echo "[$job] existing job dir -> harbor job resume (completed trials kept)"
      cmd="harbor job resume -p $(printf %q "$JOBS/$job") \
        $(for e in ${RETRY_INCLUDE//,/ }; do printf -- '-f %s ' "$e"; done) -y"
    else
      local filt="" t
      while IFS= read -r t; do [ -n "$t" ] && filt+=" -i $(printf %q "$t")"; done \
        < "$sdir/shard-$i.txt"
      echo "[$job] fresh: served=$served n=$N api_base=$url shard=$sdir/shard-$i.txt ($(wc -l < "$sdir/shard-$i.txt") tasks)"
      cmd="harbor run -p $(printf %q "$TASKS") -a terminus-2 -m hosted_vllm/$served \
        --ak api_base=$url \
        --ak model_info=$(printf %q "$MODEL_INFO") \
        --ae HOSTED_VLLM_API_KEY=local \
        --timeout-multiplier $TIMEOUT_MULT \
        -r $MAX_RETRIES \
        $(for e in ${RETRY_INCLUDE//,/ }; do printf -- '--retry-include %s ' "$e"; done) \
        -o $(printf %q "$JOBS") --job-name $job -n $N -y $filt"
    fi
    setsid bash -c "$cmd" >"$JOBS/$job.log" 2>&1 &
    echo $! > "$JOBS/$job.pid"
    pids+=($!); jobs+=("$job")
    echo "[$job] pid $! (log: $JOBS/$job.log)"
  done

  local rc=0 j
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then
      echo "[${jobs[$j]}] exited 0"
    else
      local jrc=$?
      echo "[${jobs[$j]}] exited $jrc -- resume with: $0 resume ${jobs[$j]}"
      rc=1
    fi
  done
  return $rc
}

# ------------------------------------------------------------------ resume ----
resume() {
  local job=${1:?usage: resume <job-name>}
  [ -f "$JOBS/$job/config.json" ] || die "no job dir $JOBS/$job (config.json missing)"
  # No disk_guard here on purpose: a sibling shard job may be live, and the
  # orphan sweep must never run under a live pass.
  echo "[$job] harbor job resume (transport-errored trials removed first: $RETRY_INCLUDE)"
  # shellcheck disable=SC2046
  harbor job resume -p "$JOBS/$job" \
    $(for e in ${RETRY_INCLUDE//,/ }; do printf -- '-f %s ' "$e"; done) -y
}

# ------------------------------------------------------------------- merge ----
# Merge K per-shard job dirs into ONE pass receipt: per-task rows via
# tools/tb_rows.py (field names verbatim from harbor result.json), aggregate
# resolved/N + wall clock (min started_at -> max finished_at across shards),
# exception histogram, and a three-way attribution stub -- one entry per
# unresolved task with every field null, for pass 3 to fill
# (quantization-suspect / capability / inconclusive-timeout).
merge() {
  local out=${1:?usage: merge <out.json> <jobdir...>} ; shift
  [ $# -ge 1 ] || die "merge: at least one job dir"
  python3 - "$out" "$@" <<'PYEOF'
import json, sys, pathlib
from datetime import datetime, timezone

import os
# __file__ is "<stdin>" under a heredoc; tb_rows.py lives next to the shell
# script, whose dir the entry point exports as TB21_TOOLS_DIR.
sys.path.insert(0, os.environ["TB21_TOOLS_DIR"])
import tb_rows

out = pathlib.Path(sys.argv[1])
job_dirs = [pathlib.Path(p) for p in sys.argv[2:]]

def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None

all_rows, per_shard, seen = [], [], {}
for jd in job_dirs:
    if not jd.is_dir():
        sys.exit(f"error: {jd} is not a directory")
    rows = tb_rows.load_rows(jd)
    for r in rows:
        r["shard_job"] = jd.name
        prev = seen.setdefault(r["task"], jd.name)
        if prev != jd.name:
            print(f"warning: task {r['task']} appears in {prev} AND {jd.name} "
                  f"(shards should be disjoint)", file=sys.stderr)
    starts = [iso(r.get("started_at")) for r in rows if r.get("started_at")]
    ends   = [iso(r.get("finished_at")) for r in rows if r.get("finished_at")]
    per_shard.append({
        "job": jd.name,
        "job_dir": str(jd),
        "n_trials": len(rows),
        "resolved": sum(1 for r in rows if r.get("resolved")),
        "started_at": min(starts).isoformat() if starts else None,
        "finished_at": max(ends).isoformat() if ends else None,
        "wall_s": round((max(ends) - min(starts)).total_seconds(), 1)
                  if starts and ends else None,
    })
    all_rows.extend(rows)

best = tb_rows.best_by_task(all_rows)
resolved = sorted(t for t, v in best.items() if v == 1.0)
unresolved = sorted(t for t in best if t not in resolved)
exc = {}
for r in all_rows:
    e = r.get("exception_type")
    if e:
        exc[e] = exc.get(e, 0) + 1

starts = [iso(r.get("started_at")) for r in all_rows if r.get("started_at")]
ends   = [iso(r.get("finished_at")) for r in all_rows if r.get("finished_at")]
wall = round((max(ends) - min(starts)).total_seconds(), 1) if starts and ends else None

receipt = {
    "schema": "qwen38-tb21-campaign-pass/1",
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "merge_argv": sys.argv,
    "what_this_is": ("One merged Terminal-Bench 2.1 pass receipt assembled from "
                     "per-shard harbor job dirs (task-sharded DP affinity, docs/46 §5). "
                     "Rows come verbatim from tools/tb_rows.py."),
    "job_dirs": [str(j) for j in job_dirs],
    "per_shard": per_shard,
    "aggregate": {
        "n_tasks": len(best),
        "n_trials": len(all_rows),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "headline": f"{len(resolved)}/{len(best)} resolved",
        "wall_clock_s": wall,
        "wall_clock_note": ("min started_at -> max finished_at across ALL shards; "
                            "shards ran in parallel, so this is the pass wall clock"),
        "exception_histogram": exc,
        "agent_timeout_errors": exc.get("AgentTimeoutError", 0),
        "unresolved_tasks": unresolved,
    },
    "attribution": {
        t: {"classification": None,          # quantization-suspect | capability | inconclusive-timeout
            "bf16_rerun_resolved": None,     # pass-3 measurement
            "bf16_rerun_job": None,
            "notes": None}
        for t in unresolved
    },
    "attribution_note": ("Three-way split left null by design: pass 3 re-runs every "
                         "twice-failed task on BF16 (same tier, agent, harness) and "
                         "fills these fields."),
    "rows": all_rows,
}
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(f"[merge] {len(job_dirs)} shard job(s) -> {out}")
print(f"[merge] {receipt['aggregate']['headline']}, wall {wall}s, "
      f"timeouts {exc.get('AgentTimeoutError', 0)}, exceptions {exc or '{}'}")
PYEOF
}

# ------------------------------------------------------------ metrics-snap ----
# Verbatim /metrics snapshot for the delta discipline: counters are
# contaminated by calibration by design, so every reported number is a DELTA
# between two NAMED snapshots, both quoted in the receipt. Snapshots are
# immutable -- an existing tag is refused, never overwritten.
metrics_snap() {
  local url=${1:?usage: metrics-snap <url> <tag>} tag=${2:?usage: metrics-snap <url> <tag>}
  local base=${url%/}; base=${base%/v1}
  local dir="$TB21_HOME/metrics" f
  mkdir -p "$dir"
  f="$dir/$tag.prom"
  [ -e "$f" ] && die "snapshot $f exists; snapshots are immutable (delta discipline) -- pick a new tag"
  curl -fsS "$base/metrics" > "$f" || { rm -f "$f"; die "GET $base/metrics failed"; }
  echo "[metrics] $f: $(wc -l < "$f") lines, sha256 $(sha256sum "$f" | cut -d' ' -f1)"
  echo "[metrics] report numbers as deltas between two named snapshots, both quoted verbatim"
}

# ------------------------------------------------------------------- entry ----
export TB21_TOOLS_DIR="$HERE"
cmd=${1:-}; shift || true
case $cmd in
  gate)          gate "$@" ;;
  ladder)        ladder "$@" ;;
  shard)         shard "$@" ;;
  pass1)         pass1 "$@" ;;
  resume)        resume "$@" ;;
  merge)         merge "$@" ;;
  metrics-snap)  metrics_snap "$@" ;;
  disk-guard)    disk_guard ;;
  *)             usage ;;
esac
