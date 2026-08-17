#!/bin/bash
# Terminal-Bench 2.1 three-pass protocol for Qwen3.8-27B-EXL3-K5K6-hydrated.
#
# WHAT A TB SCORE IS HERE: every number this script produces is a score for an
# *agent + model system* (Terminus-2 2.0.0 driving a vLLM endpoint), never a
# model-only capability number. Pass 3 exists precisely because a failed task
# does not by itself implicate the quantisation.
#
# Topology (owner-decided route c):
#   - Model serves HERE on the rental RTX PRO 6000 (96 GB) via tools/ggrun.sh
#     (proot into the promoted Gilded Gnosis rootfs; no container runtime on
#     this box: unshare -U/-m are seccomp-EPERM, no docker/podman, no systemd,
#     so task containers cannot run here).
#   - Terminal-Bench task containers run on AIBoss (podman 4.9.3 rootless,
#     static docker CLI 27.5.1 + compose v2.39.1 pointed at the podman socket
#     via DOCKER_HOST). CPU-ONLY there: every TB 2.1 task pins gpus = 0.
#   - The endpoint crosses an ssh -R reverse tunnel (rental:8000 ->
#     aiboss 127.0.0.1:18000) through the jump host, with keepalives and an
#     auto-reconnect loop.
#
# Pins (see receipts/terminal-bench-2.1-*.json for the full recorded set):
#   harness   harbor 0.21.0 (uv tool, PyPI), docker CLI 27.5.1 / compose 2.39.1
#   dataset   terminal-bench/terminal-bench-2-1@6, 89 tasks, pinned local tree
#             ~/tb21/tasks on aiboss, sha256(tree) recorded at pin time
#   agent     terminus-2 (harbor built-in, version 2.0.0) via LiteLLM
#             hosted_vllm/<served-name>, api_base through the tunnel, with an
#             explicit model_info (LiteLLM knows no hosted_vllm/ cost or
#             context window, so it must be supplied or the agent's
#             summarisation trigger misfires)
#   sampling  the agent sends no temperature; vLLM applies each checkpoint's
#             generation_config -- and those two files are byte-identical
#             between the hydrated and BF16 checkpoints, so pass 3 is a
#             like-for-like comparison
#   timeouts  task.toml values x $TB_TIMEOUT_MULT (recorded; raised only if the
#             measured tunnel RTT justifies it)
#
# Resume is the load-bearing design property (the bench is hours long):
#   * Within a pass, `harbor run --max-retries` retries only *transport*
#     exceptions ($RETRY_INCLUDE) so a tunnel drop never scores as a task
#     failure, while a genuine task failure is left alone for pass 2.
#   * Across interruptions, rerunning passN with the job dir present calls
#     `harbor job resume`, which harbor implements by reconciling planned
#     trials against completed ones: a trial dir without result.json is
#     deleted and re-run, a completed trial is preserved and skipped. That is
#     resume-by-task-id, enforced by the harness rather than by this script.
#     `harbor job resume` re-reads the job's own config.json, which is
#     required: harbor refuses to resume a job dir with a different config.
#
# Usage:
#   run_terminal_bench.sh serve hydrated|bf16     # foreground vLLM via ggrun
#   run_terminal_bench.sh tunnel                  # foreground reconnect loop
#   run_terminal_bench.sh wait-ready [secs]       # poll /v1/models via aiboss
#   run_terminal_bench.sh probe [tag]             # 1-token RTT: aiboss vs local
#   run_terminal_bench.sh headroom [tag]          # concurrency sweep -> pick N
#   run_terminal_bench.sh warm hydrated|bf16|both # page-cache weights pre-serve
#   run_terminal_bench.sh warm-images [N]         # bounded task-image warm set
#   run_terminal_bench.sh metrics <tag>           # vLLM /metrics + MTP accept
#   run_terminal_bench.sh gpu-evidence <tag>      # nvidia-smi procs, both hosts
#   run_terminal_bench.sh no-gpu-assert [job]     # hard CPU-only assertions
#   run_terminal_bench.sh dry-run <task>          # one task, full stack
#   run_terminal_bench.sh pass1|pass2|pass3 [--n-concurrent-trials N] [--task T]
#   run_terminal_bench.sh collect                 # rsync job dirs back here
#   run_terminal_bench.sh rows|summary|failures <job-name>
#   run_terminal_bench.sh publish <label> <job-name>   # -> HF dataset
#   run_terminal_bench.sh verify                  # remote check of the dataset
set -euo pipefail

REMOTE=${TB_REMOTE:-aiboss}
RDIR=${TB_RDIR:-tb21}                      # remote workdir (under $HOME)
LPORT=${TB_LOCAL_PORT:-8000}               # vLLM port on the rental
RPORT=${TB_REMOTE_PORT:-18010}               # tunnel listen port on aiboss, owned by OUR proxy
                                           # (18000 was poisoned by a stale root-owned sshd listener
                                           #  during preflight; see tunnel() for why that cannot recur)
MODELS=${GG_MODELS:-/var/tmp/models}
HYD=Qwen3.8-27B-EXL3-K5K6-hydrated
BF16=Qwen3.8-27B
SERVED_HYD=qwen38
SERVED_BF16=qwen38-bf16
MAXLEN=${TB_MAXLEN:-32768}
OUTDIR=${TB_OUTDIR:-$HOME/tb21-results}
DATASET=${TB_DATASET:-malaiwah/qwen38-27b-terminal-bench-2.1}
TIMEOUT_MULT=${TB_TIMEOUT_MULT:-1.0}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_ENV='export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock PATH=$HOME/.local/bin:$PATH'

# Transport-only retries. A tunnel drop, a restarted server or a podman hiccup
# must not be recorded as the model failing a task; a wrong answer must not be
# silently retried. AgentTimeoutError is deliberately NOT here: running out of
# task time is a real TB failure mode, and pass 2 retries it anyway.
#
# NotFoundError was added after pass 2 measured one: the endpoint answered
# `{"message":"The model 'qwen38' does not exist","code":404}` because the pass
# overlapped a model swap, and the trial was scored as a task failure. That is a
# server-side 404 before any prompt reaches a model, so it belongs here.
# Deliberately still absent: bare RuntimeError. Two trials died on "Failed to
# start tmux session", which is equally pre-model, but the *type* is too broad -
# a RuntimeError can also be the model's own doing. Those are classified
# post-hoc as pre_model_void by tb_rows.py and re-run explicitly, rather than
# blanket-retried by type. See receipts/terminal-bench-2.1-pre-model-voids.json.
RETRY_INCLUDE=${TB_RETRY_INCLUDE:-APIConnectionError,APIError,InternalServerError,ServiceUnavailableError,Timeout,ConnectionError,RemoteProtocolError,EnvironmentStartError,EnvironmentStartTimeoutError,AgentSetupTimeoutError,NotFoundError}
MAX_RETRIES=${TB_MAX_RETRIES:-3}

# The dataset publisher runs from this box; keep the shared cache/token store.
export HF_HOME=${HF_HOME:-/var/tmp/hf-home-3}

QCFG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]}'
SPEC='{"method":"mtp","num_speculative_tokens":3}'
# The cache_* costs are present only to silence LiteLLM's register_model warning
# ("cache cost fields will default to 0"), observed in the agent preflight. All
# costs are 0 because this is a local endpoint; the fields that matter are the
# token limits, without which the agent miscomputes remaining context.
MODEL_INFO='{"max_input_tokens":32768,"max_output_tokens":8192,"input_cost_per_token":0,"output_cost_per_token":0,"cache_creation_input_token_cost":0,"cache_read_input_token_cost":0}'

usage() { sed -n '2,80p' "$0"; exit 1; }

rssh() { ssh -o BatchMode=yes "$REMOTE" "$REMOTE_ENV; cd \$HOME/$RDIR; $*"; }

# ---------------------------------------------------------------- serving ----
serve() {
  local which=${1:?serve hydrated|bf16} model quant=() served
  case $which in
    hydrated) model=/models/$HYD; served=$SERVED_HYD
              quant=(--quantization exl3 --quantization-config "$QCFG") ;;
    bf16)     model=/models/$BF16; served=$SERVED_BF16 ;;
    *) usage ;;
  esac
  # Deviations from the published 8,192-token card recipe, recorded in the
  # receipts: window 32,768 + fp8 KV (Terminus-2 summarises when free context
  # drops below its threshold, so an 8k window would thrash on TB's long
  # tool-output transcripts), utilisation 0.92, 16 sequences, MTP-3 per the
  # flagship recipe, prefix caching + mamba align per the shipped 8k profile.
  # --enforce-eager is mandatory for the EXL3 loader (docs/03 runtime
  # contract: Exl3Config._require_enforce_eager) and is kept for BF16 too so
  # pass 3 differs from passes 1-2 in the weights alone.
  # Caveat carried from the promoted image: the un-merged #51812 GDN
  # spec-gate defect (0.515/1000 builds upper bound under adversarial
  # traffic) ships here and is recorded in the receipts.
  exec "$HERE/ggrun.sh" vllm serve "$model" \
    --served-model-name "$served" "${quant[@]}" \
    --mm-processor-kwargs '{"truncation":false}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --max-model-len "$MAXLEN" --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
    --enable-prefix-caching --mamba-cache-mode align \
    --speculative-config "$SPEC" \
    --enforce-eager \
    --host 127.0.0.1 --port "$LPORT"
}

# Two preflight lessons are baked into this function.
#
# (1) The tunnel MUST own a dedicated ssh connection.
#     ~/.ssh/config sets `ControlMaster auto` + `ControlPersist 600` for this
#     host, so by default a forward is registered on a shared, persistent master
#     and OUTLIVES the client process. That silently breaks a reconnect loop in
#     both directions: killing the client does not drop the forward (so drops go
#     undetected and a reconnect test can appear to pass while proving nothing),
#     and the loop's own `ssh -N` can exit immediately having handed the forward
#     to the master. ControlMaster=no + ControlPath=none makes the forward's
#     lifetime equal to this process's lifetime, which is the whole point.
#     Note rssh() deliberately still uses the shared master: for short commands
#     that multiplexing is pure win.
#
# (2) The remote TCP port is owned by OUR OWN proxy, not by sshd.
#     A plain `ssh -R <port>:...` makes sshd own the remote listener, so its
#     state is not ours to manage -- and the port cannot simply be changed when
#     something goes wrong, because harbor bakes api_base into the job config
#     and resume demands an identical config (job.py:246), so it must stay fixed
#     across all three passes. Instead ssh binds a unix SOCKET and
#     tools/tb_tunnel_proxy.py -- an ordinary user process we can kill and
#     restart -- owns 127.0.0.1:$RPORT. It also refuses fast when the socket is
#     absent (FileNotFoundError) or dead (ConnectionRefusedError), which is what
#     makes wait-ready and probe real signals.
#
# (3) The stale remote socket must be unlinked BY US, before every attempt.
#     StreamLocalBindUnlink is a red herring for `-R`: for a REMOTE forward the
#     unlink happens on the server, so it is sshd_config's StreamLocalBindUnlink
#     that governs it, and that defaults to "no". Observed directly -- after an
#     unclean drop every reconnect failed with
#     "Error: remote port forwarding failed for listen path <sock>" in a loop,
#     while the stale socket file sat there and the proxy logged
#     ConnectionRefusedError. We cannot edit sshd_config, but the socket lives
#     under /run/user/1000 and is ours, so the loop deletes it itself.
#     (Deleting it is only safe because of (1): with a persistent master still
#     holding the forward it would have removed a live listener.)
tunnel() {
  local sock=${TB_SOCK:-/run/user/1000/tb21-vllm.sock}
  echo "[tunnel] rental:$LPORT -> unix:$sock -> $REMOTE:127.0.0.1:$RPORT"
  # Deploy the proxy and (re)start exactly our own instance.
  scp -q -o BatchMode=yes "$HERE/tb_tunnel_proxy.py" "$REMOTE:$RDIR/" || {
    echo "[tunnel] FATAL: cannot deploy the proxy" >&2; return 1; }
  # rssh already cd's into $HOME/$RDIR, so these paths are relative to it.
  rssh "if [ -f .tunnel-proxy.pid ]; then kill \$(cat .tunnel-proxy.pid) 2>/dev/null || true; sleep 1; fi;
        nohup python3 tb_tunnel_proxy.py --port $RPORT --socket $sock \
          --pidfile .tunnel-proxy.pid >> tunnel-proxy.log 2>&1 &
        sleep 1; tail -3 tunnel-proxy.log"
  local n=0 rc
  while true; do
    n=$((n+1))
    echo "[tunnel] attempt $n at $(date -Is)"
    # Clear any stale socket left by an unclean drop; sshd will not do it.
    rssh "rm -f $sock" 2>/dev/null || true
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes \
        -o StreamLocalBindUnlink=yes \
        -o ControlMaster=no -o ControlPath=none \
        -R "$sock:127.0.0.1:$LPORT" "$REMOTE" && rc=0 || rc=$?
    echo "[tunnel] dropped rc=$rc at $(date -Is); reconnecting in 5s" >&2
    sleep 5
  done
}

wait_ready() {
  local deadline=$(( $(date +%s) + ${1:-2400} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ssh -o BatchMode=yes "$REMOTE" \
         "curl -sf -m 5 http://127.0.0.1:$RPORT/v1/models" >/dev/null 2>&1; then
      echo "[ready] endpoint live through the tunnel at $(date -Is)"; return 0
    fi
    sleep 10
  done
  echo "[ready] TIMED OUT" >&2; return 1
}

# ------------------------------------------------------------ measurement ----
probe() {
  local tag=${1:-probe} dir=$OUTDIR/probe served
  served=$(served_now); mkdir -p "$dir"
  local body="{\"model\":\"$served\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}"
  local out=$dir/$(date +%Y%m%dT%H%M%S)-$tag.txt
  {
    echo "== served model: $served =="
    echo "== 1-token RTT from aiboss, through the ssh -R tunnel (5 samples) =="
    ssh -o BatchMode=yes "$REMOTE" "for i in 1 2 3 4 5; do curl -s -o /dev/null \
      -w '%{time_total}\n' -m 60 -H 'Content-Type: application/json' \
      -d '$body' http://127.0.0.1:$RPORT/v1/completions; done"
    echo "== 1-token RTT on the rental loopback, same request (5 samples) =="
    for i in 1 2 3 4 5; do curl -s -o /dev/null -w '%{time_total}\n' -m 60 \
      -H 'Content-Type: application/json' -d "$body" \
      "http://127.0.0.1:$LPORT/v1/completions"; done
  } | tee "$out"
  echo "[probe] -> $out"
}

# Concurrency headroom on the 96 GB card: run identical short completions at
# rising concurrency and report aggregate output tok/s, so -n is chosen from a
# measurement rather than a guess.
headroom() {
  local tag=${1:-headroom} dir=$OUTDIR/headroom served
  served=$(served_now); mkdir -p "$dir"
  local out=$dir/$(date +%Y%m%dT%H%M%S)-$tag.json
  python3 - "$served" "$LPORT" "$out" <<'EOF'
import json, sys, time, urllib.request, concurrent.futures as cf
served, port, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPT = ("You are a shell agent. Explain, in detail, how you would find every "
          "file larger than 10 MB under /var and summarise the result. ")
def one(_):
    body = json.dumps({"model": served, "prompt": PROMPT, "max_tokens": 256,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["usage"]["completion_tokens"], time.time() - t0
res = {"served": served, "arms": []}
for n in (1, 2, 4, 8, 16):
    with cf.ThreadPoolExecutor(n) as ex:
        t0 = time.time()
        got = list(ex.map(one, range(n)))
    wall = time.time() - t0
    toks = sum(g[0] for g in got)
    lat = [round(g[1], 3) for g in got]
    arm = {"concurrency": n, "wall_s": round(wall, 3), "output_tokens": toks,
           "aggregate_output_tok_per_s": round(toks / wall, 2),
           "per_request_tok_per_s": round(toks / wall / n, 2),
           "latencies_s": lat, "max_latency_s": max(lat)}
    res["arms"].append(arm)
    print(json.dumps(arm), flush=True)
best = max(res["arms"], key=lambda a: a["aggregate_output_tok_per_s"])
res["peak_aggregate_concurrency"] = best["concurrency"]
res["peak_aggregate_output_tok_per_s"] = best["aggregate_output_tok_per_s"]

# -n SELECTION RULE, fixed in advance so the number is derived and not chosen to
# suit a result. Terminal-Bench charges each task a wall-clock timeout, so raising
# concurrency past the point where per-request speed collapses does not just stop
# helping -- it starts spending the agent's own timeout budget and manufactures
# AgentTimeoutErrors that look like task failures. So:
#   1. Never exceed the server's max_num_seqs (16): beyond it requests queue.
#   2. Take the largest arm whose PER-REQUEST throughput still holds at or above
#      50 % of the single-stream rate. That caps the per-task slowdown at 2x,
#      which the stock 1.0 timeout multiplier absorbs.
#   3. Among arms passing (2), prefer the one with the highest aggregate
#      throughput -- that is the one that finishes 89 tasks soonest.
single = next(a for a in res["arms"] if a["concurrency"] == 1)
floor = 0.5 * single["per_request_tok_per_s"]
eligible = [a for a in res["arms"]
            if a["concurrency"] <= 16 and a["per_request_tok_per_s"] >= floor]
pick = max(eligible, key=lambda a: a["aggregate_output_tok_per_s"]) if eligible else single
res["n_selection_rule"] = (
    "largest arm with per-request tok/s >= 50% of single-stream (cap per-task "
    "slowdown at 2x, absorbed by the stock 1.0 timeout multiplier), capped at "
    "max_num_seqs=16, then highest aggregate throughput among those")
res["single_stream_per_request_tok_per_s"] = single["per_request_tok_per_s"]
res["per_request_floor_tok_per_s"] = round(floor, 2)
res["eligible_concurrencies"] = [a["concurrency"] for a in eligible]
res["recommended_n_concurrent_trials"] = pick["concurrency"]
res["recommended_n_aggregate_tok_per_s"] = pick["aggregate_output_tok_per_s"]
open(out, "w").write(json.dumps(res, indent=2))
print("peak aggregate throughput at concurrency", best["concurrency"])
print("RECOMMENDED -n =", pick["concurrency"],
      f"(per-request {pick['per_request_tok_per_s']} >= floor {floor:.2f} tok/s,"
      f" aggregate {pick['aggregate_output_tok_per_s']} tok/s)")
EOF
  echo "[headroom] -> $out"
}

# Page-cache the checkpoint before serving. This box has ~1.3 TB of free RAM
# against 20.1 GiB of hydrated and 51.7 GiB of BF16 safetensors, so both fit with
# room to spare, and vLLM's weight load then reads from RAM instead of disk.
# Purpose is narrow and worth stating: it shortens the GPU window, which is the
# scarce, queued, shared resource -- warming costs only wall time we are spending
# waiting anyway. Run it again immediately before `serve`: another agent's IO over
# a multi-hour queue can evict the cache.
warm() {
  local which=${1:-hydrated} dir
  case $which in
    hydrated) dir=$MODELS/$HYD ;;
    bf16)     dir=$MODELS/$BF16 ;;
    both)     warm hydrated; warm bf16; return ;;
    *) usage ;;
  esac
  local bytes t0 t1 secs
  # Only the weight shards matter, and du -sb would also count SHA256SUMS etc.
  bytes=$(du -cb "$dir"/*.safetensors 2>/dev/null | tail -1 | cut -f1)
  t0=$(date +%s)
  cat "$dir"/*.safetensors > /dev/null
  t1=$(date +%s)
  secs=$(( t1 - t0 ))
  [ "$secs" -gt 0 ] || secs=1
  # awk, not bc: bc is not installed on this box.
  awk -v b="$bytes" -v s="$secs" -v w="$which" 'BEGIN {
    printf "[warm] %s: %.1f GiB of safetensors read in %ds (%.2f GiB/s effective)\n",
           w, b/1073741824, s, b/1073741824/s }'
  free -g | awk '/^Mem:/ {print "[warm] RAM: " $3 "G used, " $6 "G buff/cache, " $7 "G available"}'
}

# Warm a bounded set of Terminal-Bench task images so pass 1 does not spend GPU
# time on its first pulls. Deliberately NOT all 89: the full set projects to
# ~42 GB against ~68 GB free on a box holding other agents' work, and a mid-pass
# disk-full is the one failure resume cannot make cheap. Stops at a 30 GB floor.
warm_images() {
  local count=${1:-20} floor=${TB_IMAGE_FLOOR_GB:-30}
  echo "[warm-images] pulling up to $count task images, stopping below ${floor}G free"
  rssh "n=0; pulled=0; skipped=0;
        for t in \$(ls -d tasks/*/ | sed 's|tasks/||; s|/||' | sort | head -$count); do
          free_g=\$(df --output=avail -BG \$HOME | tail -1 | tr -dc 0-9);
          if [ \"\$free_g\" -lt $floor ]; then
            echo \"[warm-images] STOP: \${free_g}G free is below the ${floor}G floor\"; break;
          fi;
          img=\$(grep -m1 -oE 'alexgshaw/[a-z0-9._-]+:[0-9]+' tasks/\$t/*.yaml tasks/\$t/task.toml 2>/dev/null | head -1 | cut -d: -f2-);
          [ -n \"\$img\" ] || img=\"alexgshaw/\$t:20251031\";
          if podman image exists \"docker.io/\$img\" 2>/dev/null; then
            skipped=\$((skipped+1));
          else
            podman pull -q \"docker.io/\$img\" >/dev/null 2>&1 && pulled=\$((pulled+1)) || echo \"  pull failed: \$img\";
          fi;
        done;
        echo \"[warm-images] pulled=\$pulled already-warm=\$skipped\";
        df -h \$HOME | tail -1;
        echo \"TB images cached: \$(podman images --format '{{.Repository}}' | grep -c alexgshaw)\""
}

# vLLM counters, including MTP draft/accept so acceptance is per-pass evidence.
metrics() {
  local tag=${1:?metrics <tag>} dir=$OUTDIR/metrics stamp
  stamp=$(date +%Y%m%dT%H%M%S); mkdir -p "$dir"
  local raw=$dir/$stamp-$tag.prom
  curl -s -m 15 "http://127.0.0.1:$LPORT/metrics" > "$raw"
  python3 - "$raw" "$dir/$stamp-$tag.json" <<'EOF'
import json, re, sys
raw, out = sys.argv[1], sys.argv[2]
want = ("spec_decode", "prompt_tokens_total", "generation_tokens_total",
        "request_success", "num_preemptions", "gpu_cache_usage",
        "gpu_prefix_cache_hit_rate", "gpu_prefix_cache_queries",
        "iteration_tokens_total")
vals = {}
for line in open(raw):
    if line.startswith("#") or not line.strip():
        continue
    m = re.match(r"([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+naN]+)$", line.strip())
    if not m:
        continue
    name, val = m.group(1), m.group(3)
    if not any(w in name for w in want):
        continue
    try:
        v = float(val)
    except ValueError:
        continue
    vals[name] = vals.get(name, 0.0) + v
draft = next((v for k, v in vals.items() if "num_draft_tokens" in k), None)
acc = next((v for k, v in vals.items() if "num_accepted_tokens" in k), None)
res = {"counters": vals}
if draft and acc is not None:
    res["mtp_draft_tokens"] = draft
    res["mtp_accepted_tokens"] = acc
    res["mtp_acceptance_rate"] = round(acc / draft, 6)
prm = next((v for k, v in vals.items() if k.endswith("prompt_tokens_total")), None)
gen = next((v for k, v in vals.items() if k.endswith("generation_tokens_total")), None)
res["prompt_tokens_total"] = prm
res["generation_tokens_total"] = gen
open(out, "w").write(json.dumps(res, indent=2, sort_keys=True))
print(json.dumps({k: res[k] for k in res if k != "counters"}, indent=2))
EOF
  echo "[metrics] -> $dir/$stamp-$tag.json"
}

served_now() {
  local s
  s=$(curl -s -m 10 "http://127.0.0.1:$LPORT/v1/models" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' \
      2>/dev/null || true)
  [ -n "$s" ] || { echo "[served] no endpoint on 127.0.0.1:$LPORT" >&2; return 1; }
  echo "$s"
}

gpu_evidence() {
  local tag=${1:?gpu-evidence <tag>} dir=$OUTDIR/gpu-evidence
  mkdir -p "$dir"
  { echo "== rental (serves the model) $(date -Is) =="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "== aiboss (runs the task containers) $(date -Is) =="
    ssh -o BatchMode=yes "$REMOTE" \
      'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
    echo "== aiboss containers, classified TB vs pre-existing $(date -Is) =="
    rssh 'for c in $(docker ps -q 2>/dev/null); do
            img=$(docker inspect -f "{{.Config.Image}}" $c 2>/dev/null);
            case "$img" in *alexgshaw/*) k=TB-TASK;; *) k=OTHER;; esac;
            docker inspect -f "$k {{.Name}} image=$img devices={{.HostConfig.Devices}} devreq={{.HostConfig.DeviceRequests}} runtime={{.HostConfig.Runtime}}" $c;
          done; echo "(no TB-TASK line means no task container was running at this instant)"'
  } | tee "$dir/$(date +%Y%m%dT%H%M%S)-$tag.txt"
}

# Hard CPU-only assertions. Four independent checks, any failure is fatal.
no_gpu_assert() {
  local job=${1:-} rc=0
  echo "== 1/4: every TB 2.1 task pins gpus = 0 in task.toml =="
  rssh 'bad=$(grep -H "^gpus" tasks/*/task.toml | grep -v "gpus = 0" || true);
        n=$(grep -l "^gpus = 0" tasks/*/task.toml | wc -l);
        if [ -n "$bad" ]; then echo "FAIL non-zero gpus:"; echo "$bad"; exit 1; fi;
        echo "OK: $n/89 task.toml files pin gpus = 0, none non-zero"' || rc=1
  echo "== 2/4: harbor's docker environment has no GPU passthrough code path =="
  rssh 'P=$HOME/.local/share/uv/tools/harbor/lib/python3.12/site-packages/harbor/environments/docker;
        hits=$(grep -rn "DeviceRequests\|--gpus\|nvidia\|/dev/nvidia" $P 2>/dev/null || true);
        if [ -n "$hits" ]; then echo "FAIL GPU references:"; echo "$hits"; exit 1; fi;
        echo "OK: no DeviceRequests/--gpus/nvidia reference in $(ls $P/*.py | wc -l) modules";
        echo "    (compose overrides are written by write_resources_compose_file,"
        echo "     which emits cpus/mem_limit/reservations only)"' || rc=1
  echo "== 3/4: every Terminal-Bench container is CPU-only (co-tenant workloads are named, not filtered away) =="
  rssh 'any=0; tb=0; other_gpu="";
        for c in $(docker ps -aq 2>/dev/null); do
          img=$(docker inspect -f "{{.Config.Image}}" $c 2>/dev/null || echo unknown);
          nm=$(docker inspect -f "{{.Name}}" $c 2>/dev/null);
          dev=$(docker inspect -f "{{.HostConfig.Devices}}{{.HostConfig.DeviceRequests}}" $c 2>/dev/null);
          case "$img" in
            *alexgshaw/*) tb=$((tb+1));
              case "$dev" in
                *nvidia*|*Driver:*) echo "  FAIL TB container $nm ($img) holds $dev"; any=1 ;;
                *) echo "  OK   TB container $nm ($img) devices=${dev:-none}" ;;
              esac ;;
            *) case "$dev" in *nvidia*|*Driver:*) other_gpu="$other_gpu $nm";; esac ;;
          esac;
        done;
        echo "  Terminal-Bench containers inspected: $tb (all CPU-only unless a FAIL above)";
        echo "  co-tenant GPU holders, pre-existing and NOT created by this bench:${other_gpu:- none}";
        exit $any' || rc=1
  echo "== 4/4: aiboss GPU is used only by the owner's service, not by TB =="
  ssh -o BatchMode=yes "$REMOTE" \
    'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
  if [ -n "$job" ] && [ -d "$OUTDIR/jobs/$job" ]; then
    echo "== bonus: per-trial recorded override_gpus for job $job =="
    python3 "$HERE/tb_rows.py" gpu-assert "$OUTDIR/jobs/$job" || rc=1
  fi
  [ $rc -eq 0 ] && echo "[no-gpu-assert] ALL CHECKS PASSED" \
                || echo "[no-gpu-assert] FAILED" >&2
  return $rc
}

# ------------------------------------------------------------------ passes ----
# Everything reclaimed here is a Terminal-Bench artifact and nothing else. The
# other images on aiboss are ~192 GB belonging to other agents and services, so a
# global `podman image prune -af` there would be destructive: there is no global
# prune anywhere in this tooling, and if the 15G threshold is ever reached that
# gets reported over hub rather than escalated into a wider cleanup by heuristic.
#
# The orphan sweep exists because it was observed: SIGKILLing harbor mid-pass (the
# resume proof did exactly that) leaves the task container AND its compose network
# behind, because harbor never gets to tear them down. Over a long interrupted
# bench those would accumulate on a shared box.
# Safe because only one pass ever runs at a time -- a live pass's containers would
# also match, so this must not be run concurrently with a pass.
disk_guard() {
  rssh 'free_g=$(df --output=avail -BG $HOME | tail -1 | tr -dc 0-9);
        echo "[disk] ${free_g}G free on aiboss $HOME";
        orph=0;
        for c in $(docker ps -a --format "{{.ID}} {{.Image}}" | awk "/alexgshaw/ {print \$1}"); do
          docker rm -f "$c" >/dev/null 2>&1 && orph=$((orph+1));
        done;
        netn=0;
        for n in $(docker network ls --format "{{.Name}}" | grep "__env" 2>/dev/null); do
          docker network rm "$n" >/dev/null 2>&1 && netn=$((netn+1));
        done;
        [ $orph -gt 0 ] || [ $netn -gt 0 ] \
          && echo "[orphans] reclaimed $orph Terminal-Bench container(s), $netn compose network(s) left by an interrupted pass" \
          || echo "[orphans] none";
        if [ "$free_g" -lt 15 ]; then
          echo "[disk] below 15G -> removing Terminal-Bench task images ONLY (never a global prune); report this over hub";
          podman images --format "{{.ID}} {{.Repository}}" \
            | awk "/alexgshaw/ {print \$1}" | xargs -r podman rmi -f 2>/dev/null || true;
          df -h $HOME | tail -1;
        fi'
}

# run_pass <job-name> <served-model> <include-file|ALL> <n-concurrent> [single-task]
run_pass() {
  local job=$1 served=$2 include=$3 n=$4 single=${5:-}
  local filt=()
  if [ -n "$single" ]; then
    filt=(-i "$single")
  elif [ "$include" != ALL ]; then
    while IFS= read -r t; do [ -n "$t" ] && filt+=(-i "$t"); done < <(rssh "cat $include")
    [ ${#filt[@]} -gt 0 ] && echo "[$job] ${#filt[@]} task(s) from $include" \
      || { echo "[$job] nothing to run (no failures) -- pass is a no-op"; return 0; }
  fi
  disk_guard
  if rssh "test -f jobs/$job/config.json"; then
    echo "[$job] existing job dir -> harbor job resume (completed trials are kept)"
    rssh "harbor job resume -p jobs/$job \
      $(for e in ${RETRY_INCLUDE//,/ }; do printf -- '-f %s ' "$e"; done) 2>&1 | tail -30"
  else
    echo "[$job] fresh run: model=$served n=$n timeout_mult=$TIMEOUT_MULT"
    rssh "harbor run -p tasks -a terminus-2 -m hosted_vllm/$served \
      --ak api_base=http://127.0.0.1:$RPORT/v1 \
      --ak 'model_info=$MODEL_INFO' \
      --ae HOSTED_VLLM_API_KEY=local \
      --timeout-multiplier $TIMEOUT_MULT \
      -r $MAX_RETRIES \
      $(for e in ${RETRY_INCLUDE//,/ }; do printf -- '--retry-include %s ' "$e"; done) \
      -o jobs --job-name $job -n $n -y \
      ${filt[*]+$(printf '%q ' "${filt[@]}")} 2>&1 | tail -40"
  fi
}

# -------------------------------------------------------------- analysis -----
collect() {
  mkdir -p "$OUTDIR/jobs"
  rsync -a --info=stats1 -e 'ssh -o BatchMode=yes' \
    "$REMOTE:$RDIR/jobs/" "$OUTDIR/jobs/"
  echo "[collect] -> $OUTDIR/jobs"
}

analyze() {                       # analyze rows|summary|failures <job>
  local mode=$1 job=${2:?<job-name>}
  [ -d "$OUTDIR/jobs/$job" ] || collect >&2
  python3 "$HERE/tb_rows.py" "$mode" "$OUTDIR/jobs/$job" "$job"
}

# ------------------------------------------------------------- publishing ----
# Publish as passes complete, not only at the end: if the rental is
# decommissioned mid-protocol the evidence already landed.
# <repo-path> is the literal path inside the dataset, e.g.
#   publish passes/pass1-hydrated tb21-pass1-hyd
#   publish harness-validation   oracle-smoke
publish() {
  local label=${1:?publish <repo-path> <job-name>} job=${2:?publish <repo-path> <job-name>}
  collect
  local stage=$OUTDIR/stage/$label
  rm -rf "$stage"; mkdir -p "$stage/trials"
  python3 "$HERE/tb_rows.py" rows    "$OUTDIR/jobs/$job"        > "$stage/results.jsonl"
  python3 "$HERE/tb_rows.py" summary "$OUTDIR/jobs/$job" "$label" > "$stage/summary.json"
  # Transcripts + logs, size-capped so one runaway log cannot bloat the dataset.
  ( cd "$OUTDIR/jobs/$job" && find . -type f \
      \( -name 'result.json' -o -name 'config.json' -o -name 'trial.log' \
         -o -path '*/agent/*' -o -path '*/verifier/*' \) -size -8M -print0 \
    | rsync -a --files-from=- --from0 ./ "$stage/trials/" )
  cp "$OUTDIR/jobs/$job/job.log" "$stage/job.log" 2>/dev/null || true
  echo "[publish] staged $(find "$stage" -type f | wc -l) files for $label"
  hf upload "$DATASET" "$stage" "$label" --repo-type dataset \
     --commit-message "Terminal-Bench 2.1: $label ($job)" 2>&1 | tail -3
  verify "$label"
}

publish_meta() {                  # push the dataset card + the receipt snapshot
  local stage=$OUTDIR/stage/meta
  rm -rf "$stage"; mkdir -p "$stage/receipts"
  cp "$HERE/../receipts/terminal-bench-2.1-"*.json "$stage/receipts/" 2>/dev/null || true
  # The dataset card is versioned in the repo, not in the scratch output dir, so
  # it survives a decommission of this box like every other deliverable.
  cp "$HERE/../receipts/terminal-bench-2.1-dataset-card.md" "$stage/README.md"
  echo "[publish-meta] staging $(find "$stage" -type f | wc -l) files"
  hf upload "$DATASET" "$stage" . --repo-type dataset \
     --commit-message "Terminal-Bench 2.1: dataset card + receipts" 2>&1 | tail -3
  verify
}

# Remote verification: read the dataset back from the Hub API (unauthenticated,
# so it also proves the repo really is public).
verify() {
  local prefix=${1:-}
  python3 - "$DATASET" "$prefix" <<'EOF'
import json, sys, urllib.request
ds, prefix = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
url = f"https://huggingface.co/api/datasets/{ds}"
with urllib.request.urlopen(url + "?full=true", timeout=60) as r:
    d = json.loads(r.read())
files = [s["rfilename"] for s in d.get("siblings", [])]
sel = [f for f in files if not prefix or f.startswith(prefix)]
print(f"dataset   {ds}")
print(f"private   {d.get('private')}   (False == public, verified unauthenticated)")
print(f"sha       {d.get('sha')}")
print(f"modified  {d.get('lastModified')}")
print(f"files     {len(files)} total, {len(sel)} under {prefix or '<root>'}")
for f in sorted(sel)[:15]:
    print("   ", f)
if len(sel) > 15:
    print(f"    ... and {len(sel) - 15} more")
if d.get("private") is not False:
    sys.exit("FAIL: dataset is not public")
if prefix and not sel:
    sys.exit(f"FAIL: nothing published under {prefix}")
EOF
}

# ------------------------------------------------------------------- entry ----
cmd=${1:-}; shift || true
N=${TB_N:-4}; TASK=
args=()
while [ $# -gt 0 ]; do
  case $1 in
    --n-concurrent-trials) N=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done

case $cmd in
  serve)         serve "${args[@]}" ;;
  tunnel)        tunnel ;;
  wait-ready)    wait_ready "${args[@]:-2400}" ;;
  probe)         probe "${args[@]:-probe}" ;;
  headroom)      headroom "${args[@]:-headroom}" ;;
  warm)          warm "${args[@]:-hydrated}" ;;
  warm-images)   warm_images "${args[@]:-20}" ;;
  metrics)       metrics "${args[@]}" ;;
  gpu-evidence)  gpu_evidence "${args[@]}" ;;
  no-gpu-assert) no_gpu_assert "${args[@]:-}" ;;
  # The job name carries the served model, so a dry run against the preflight
  # double (served as 'qwen38' by tb_mock_endpoint.py) can never be mistaken
  # for, or resumed as, a dry run against the real server -- and a stale dry-run
  # dir can never make harbor try to resume a job with a different config.
  dry-run)       s=$(served_now); run_pass "tb21-dryrun-$s-$(date +%Y%m%dT%H%M%S)" "$s" ALL 1 "${args[0]:?dry-run <task>}" ;;
  pass1)         run_pass tb21-pass1-hyd "$SERVED_HYD" ALL "$N" "$TASK" ;;
  pass2)         analyze failures tb21-pass1-hyd | rssh 'cat > pass1-failures.txt'
                 run_pass tb21-pass2-hyd-healing "$SERVED_HYD" pass1-failures.txt "$N" "$TASK" ;;
  pass3)         { analyze failures tb21-pass1-hyd
                   analyze failures tb21-pass2-hyd-healing; } \
                   | sort | uniq -d | rssh 'cat > twice-failures.txt'
                 run_pass tb21-pass3-bf16-attribution "$SERVED_BF16" twice-failures.txt "$N" "$TASK" ;;
  collect)       collect ;;
  rows)          analyze rows "${args[@]}" ;;
  summary)       analyze summary "${args[@]}" ;;
  failures)      analyze failures "${args[@]}" ;;
  publish)       publish "${args[@]}" ;;
  publish-meta)  publish_meta ;;
  verify)        verify "${args[@]:-}" ;;
  *)             usage ;;
esac
