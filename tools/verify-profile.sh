#!/bin/bash
# verify-profile.sh -- read-only regression gate for the Qwen3.8-27B EXL3
# flagship serving profile.
#
# WHY: the ad-hoc benches (single boot, median-of-5, no CI, no acceptance)
# cannot resolve 3% effects or catch regressions automatically. This gate
# measures a live server against a stored baseline (tools/baseline-flagship.json)
# with explicit per-metric tolerances, so an automated job fails loudly when
# PP / TG / context / MTP-acceptance regress or the sanity / vision / long-context
# invariants break.
#
# Contract -- other automation depends on these exit codes:
#   exit 0  every check passed
#   exit 1  at least one metric regressed past its tolerance
#   exit 2  infrastructure failure (server unreachable on /health, or boot died)
#
# This is a READ-ONLY gate: it never stops or reconfigures the running service.
# The only state change it can make is `systemctl --user start qwen38-27b.service`
# in the --boot path. All measurement primitives (prompts, warmup, PP/TG,
# acceptance sampling, sanity, vision, long-ctx, gpu telemetry) come from
# bench_lib.py, shared with bench-profile.sh, so the two scripts cannot drift on
# prompt or metric definitions.
#
# Health gating is done in bash (curl, fail-fast) so the exit-2 contract holds
# before any python import; the python driver only runs once /health is green.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BASELINE="$SCRIPT_DIR/baseline-flagship.json"
SERVICE="qwen38-27b.service"
CONTAINER="qwen38-27b"
BASE_URL="${BENCH_BASE_URL:-http://localhost:8000}"

usage() {
  cat <<EOF
Usage: verify-profile.sh [--baseline <json>] [--json-out <path>] [--no-boot] [--boot] [--help]

Read-only regression gate against the stored flagship profile baseline.

Options:
  --baseline <json>   Baseline JSON to gate against.
                      Default: $DEFAULT_BASELINE
  --json-out <path>   Write a per-metric JSON result here.
                      Default: /tmp/verify-profile-<UTC-timestamp>.json
  --no-boot           Assume a server is already running and healthy (the
                      default). Accepted as a no-op alias for clarity.
  --boot              Boot the default flagship config first via
                      \`systemctl --user start $SERVICE\`, then wait for /health.
                      Without this flag the script expects an already-running
                      server and fails fast (exit 2) if /health is unreachable.
  --help, -h          Print this help and exit 0.

Exit codes:
  0  all checks passed
  1  at least one metric regressed past tolerance
  2  infrastructure failure (server unreachable / boot died)

Environment:
  BENCH_BASE_URL  Server base URL (default http://localhost:8000).
  BENCH_MODEL     Model id (default Qwen3.8-27B).
EOF
}

baseline="$DEFAULT_BASELINE"
json_out=""
boot=0
no_boot=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) baseline="$2"; shift 2;;
    --json-out) json_out="$2"; shift 2;;
    --no-boot)  no_boot=1; shift;;
    --boot)     boot=1; shift;;
    --help|-h)  usage; exit 0;;
    *) echo "verify-profile.sh: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

# --no-boot is the default behaviour; an explicit --boot alongside it is
# contradictory. Letting either silently win would hide user intent.
if [[ "$boot" -eq 1 && "$no_boot" -eq 1 ]]; then
  echo "verify-profile.sh: --boot and --no-boot are mutually exclusive" >&2
  exit 2
fi

if [[ -z "$json_out" ]]; then
  json_out="/tmp/verify-profile-$(date -u +%Y%m%dT%H%M%SZ).json"
fi

if [[ ! -f "$baseline" ]]; then
  echo "verify-profile.sh: baseline not found: $baseline" >&2
  exit 2
fi

# Ensure the JSON output parent directory exists (default /tmp always does).
mkdir -p "$(dirname "$json_out")" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Health gate (bash, fail-fast). Keeps the exit-2 contract independent of any
# python import and avoids a slow poll when the server is simply absent.
# -----------------------------------------------------------------------------
# Fail-fast probe for the default (already-running) path: a handful of quick
# attempts, then bail with a clear "server unreachable" message and exit 2.
health_probe() {
  local i
  for i in 1 2 3; do
    if curl -sf --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then
      echo "# /health reachable at $BASE_URL" >&2
      return 0
    fi
    sleep 2
  done
  echo "verify-profile.sh: server unreachable at $BASE_URL/health (expected an already-running server; pass --boot to start one)." >&2
  exit 2
}

# Boot wait for the --boot path: poll /health up to a deadline, but also watch
# the container state. If `podman inspect` says anything other than "running"
# the boot died -- surface the last error/OOM/traceback lines and exit 2.
boot_wait() {
  local timeout=$1
  local start=$SECONDS
  while true; do
    if curl -sf --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then
      echo "# /health ready after $((SECONDS - start))s at $BASE_URL" >&2
      return 0
    fi
    local status
    status="$(podman inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"
    if [[ "$status" != "running" ]]; then
      echo "verify-profile.sh: infra failure: boot died -- container $CONTAINER status='${status:-absent}'." >&2
      echo "  last error lines from \`podman logs $CONTAINER\`:" >&2
      podman logs --tail 200 "$CONTAINER" 2>&1 \
        | grep -E 'Error|OutOfMemory|Traceback' | tail -n 5 | sed 's/^/  | /' >&2 || true
      exit 2
    fi
    if (( SECONDS - start >= timeout )); then
      echo "verify-profile.sh: infra failure: timed out after ${timeout}s waiting for $BASE_URL/health (container still running)." >&2
      exit 2
    fi
    sleep 2
  done
}

if [[ "$boot" -eq 1 ]]; then
  echo "# booting $SERVICE ..." >&2
  if ! systemctl --user start "$SERVICE"; then
    echo "verify-profile.sh: infra failure: systemctl --user start $SERVICE failed" >&2
    exit 2
  fi
  boot_wait 600
else
  health_probe
fi

# -----------------------------------------------------------------------------
# Measurement + comparison + output. The python driver imports bench_lib for
# every primitive, runs the ordered sequence, compares against the baseline,
# prints a PASS/FAIL table, writes the JSON result, and exits 0/1/2 itself.
# `exec` replaces the shell so python's exit code is the script's exit code.
# -----------------------------------------------------------------------------
export VERIFY_TOOLS_DIR="$SCRIPT_DIR"
export BENCH_BASE_URL="$BASE_URL"
exec python3 - "$baseline" "$json_out" "$boot" <<'PY'
import sys, os, json, traceback
from datetime import datetime, timezone

# Locate bench_lib next to this script (passed via env by the shell).
TOOLS_DIR = os.environ.get('VERIFY_TOOLS_DIR', '.')
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import bench_lib
import requests

baseline_path = sys.argv[1]
json_out = sys.argv[2]
boot = (sys.argv[3] == '1')

with open(baseline_path) as f:
    baseline = json.load(f)
metrics_cfg = baseline.get('metrics', {})


def now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fmt_val(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if v is None:
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.4g}' if abs(v) < 1000 else f'{v:.1f}'
    return str(v)


# ---- ordered measurement (server already confirmed healthy by the shell) ----
results = {}
gpu_before = {}
gpu_after = {}
try:
    gpu_before = bench_lib.gpu_telemetry()

    # 1. sanity -- must contain 'Paris'
    sanity_text = bench_lib.sanity()
    results['sanity_paris'] = {'measured': ('Paris' in sanity_text), 'raw': sanity_text}

    # 2. max_model_len
    results['max_model_len'] = {'measured': bench_lib.max_model_len()}

    # 3. vision red/blue
    results['vision_red_blue'] = {'measured': bench_lib.vision_check()}

    # 4. long context 200k
    results['long_ctx_200k'] = {'measured': bench_lib.long_ctx_check(target_tokens=200000)}

    # 5. PP (prefill throughput, 2051-tok prompt, max_tokens=1, median of 5)
    pp = bench_lib.measure_pp(reps=5)
    results['pp_tok_s'] = {
        'measured': pp['tok_s_median'],
        'all': pp['tok_s_all'],
        'prompt_tokens': pp['prompt_tokens'],
    }

    # 6. TG-fox (decode throughput, max_tokens=200, median of 3)
    fox = bench_lib.measure_tg(prompt=bench_lib.TG_FOX_PROMPT, max_tokens=200, reps=3)
    results['tg_fox_tok_s'] = {
        'measured': fox['tok_s_median'],
        'all': fox['tok_s_all'],
        'completion_tokens': fox['completion_tokens'],
        'acceptance': fox['acceptance'],
    }
    # Acceptance is strongly prompt-dependent (fox ~0.93-1.00, essay ~0.28-0.30),
    # so it is gated per prompt rather than reported once.  A drop here is an
    # early warning that weight fidelity regressed: FP4 weights draft the fox
    # prompt at 0.930 where trellis weights draft it at 1.000.
    results['acceptance_fox'] = {
        'measured': fox['acceptance'],
        'draft_delta': fox['draft_delta'],
        'accepted_delta': fox['accepted_delta'],
    }

    # 7. TG-essay + acceptance (max_tokens=500, median of 3)
    essay = bench_lib.measure_tg(prompt=bench_lib.TG_ESSAY_PROMPT, max_tokens=500, reps=3)
    results['tg_essay_tok_s'] = {
        'measured': essay['tok_s_median'],
        'all': essay['tok_s_all'],
        'completion_tokens': essay['completion_tokens'],
    }
    results['acceptance_essay'] = {
        'measured': essay['acceptance'],
        'draft_delta': essay['draft_delta'],
        'accepted_delta': essay['accepted_delta'],
    }

    gpu_after = bench_lib.gpu_telemetry()
except requests.exceptions.RequestException as e:
    print(f'verify-profile.sh: server unreachable during measurement at {bench_lib.BASE_URL}: {e}', file=sys.stderr)
    sys.exit(2)
except Exception as e:
    # Unexpected error during measurement -- surface it and treat as infra
    # failure so a broken run is never silently reported as PASS.
    print(f'verify-profile.sh: measurement error: {e}', file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(2)


# ---- compare each measured value against the baseline tolerance rules ----
def eval_metric(cfg, measured):
    rule = cfg.get('rule')
    base = cfg.get('baseline')
    verdict = 'FAIL'
    threshold = ''
    if rule == 'min_ratio':
        ratio = cfg.get('min_ratio')
        thresh = base * ratio
        threshold = f'>= {thresh:.4g} ({base:g} * {ratio})'
        if measured is not None and measured >= thresh:
            verdict = 'PASS'
    elif rule == 'min_absolute':
        mv = cfg.get('min_absolute')
        threshold = f'>= {mv}'
        if measured is not None and measured >= mv:
            verdict = 'PASS'
    elif rule == 'boolean_true':
        threshold = '== true'
        if measured is True:
            verdict = 'PASS'
    else:
        threshold = f'unknown rule: {rule}'
    return verdict, threshold


verdicts = {}
any_fail = False
for name, cfg in metrics_cfg.items():
    measured = results.get(name, {}).get('measured')
    v, thr = eval_metric(cfg, measured)
    if v != 'PASS':
        any_fail = True
    verdicts[name] = {
        'baseline': cfg.get('baseline'),
        'measured': measured,
        'threshold': thr,
        'verdict': v,
        'description': cfg.get('description', ''),
    }

overall_pass = not any_fail

# ---- PASS/FAIL table ----
print()
header = f"{'metric':<20} {'baseline':>12} {'measured':>14} {'threshold':<30} {'verdict':>6}"
print(header)
print('-' * len(header))
for name, cfg in metrics_cfg.items():
    v = verdicts[name]
    print(f"{name:<20} {fmt_val(v['baseline']):>12} {fmt_val(v['measured']):>14} {v['threshold']:<30} {v['verdict']:>6}")
print()
print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")

# ---- JSON result ----
doc = {
    'schema': 'qwen38-profile-verify/1',
    'timestamp': now_iso(),
    'baseline_path': baseline_path,
    'baseline_schema': baseline.get('schema'),
    'base_url': bench_lib.BASE_URL,
    'boot': boot,
    'pass': overall_pass,
    'gpu_before': gpu_before,
    'gpu_after': gpu_after,
    'metrics': verdicts,
    'raw': results,
}
with open(json_out, 'w') as f:
    json.dump(doc, f, indent=2)
print(f'# json result: {json_out}', file=sys.stderr)

sys.exit(0 if overall_pass else 1)
PY
