#!/bin/bash
# Drive the eight proxy24 runs: two emulation arms x two published 24 GB-class
# predictions x two engine processes each, because no long-context gate may reuse a
# process that already served a different-length long request (the r34 image lacks
# upstream #51113).
#
# Constants, all measured on this card rather than assumed:
#   nvidia-smi FB total            32,607 MiB
#   nvidia-smi FB reserved            458 MiB   (invisible to CUDA)
#   cudaMemGetInfo total  T =      32,149 MiB = 31.3955 GiB = 33,710,669,824 B
# T is confirmed to the byte by the engine's own log: at util 0.7643 it printed a
# 24.0 GiB budget and advised `--kv-cache-memory-bytes=3385755786` to fit the request
# and `=10799671296` to fill the card, whose difference is free - requested = 6.905 GiB;
# with free = 30.9 GiB that puts requested at 23.995 GiB = ceil(T x 0.7643) exactly.
# The engine's rule (vllm/v1/worker/utils.py request_memory) is
#   requested = ceil(cudaMemGetInfo_total * gpu_memory_utilization)
# i.e. it multiplies TOTAL, never free, and then refuses to start if free < requested.
#
# budget arm: requested must be provably <= 24.0 GiB, so util 0.7635 -> 23.97 GiB.
#   0.7643 was rejected: it prints "24.0 GiB", which is 23.995 rounded and therefore
#   not evidence of anything below 24.0. 0.7635 prints 23.97 and cannot be misread.
# board  arm: requested = what a physical 24 GiB board computes at the qualified 0.955.
#   That board's CUDA total is 24,576 - 458 = 24,118 MiB = 23.5527 GiB, so its budget is
#   0.955 x 23.5527 = 22.4928 GiB, i.e. util = 22.4928 / 31.3955 = 0.71643 -> 0.7164,
#   which prints 22.49. A ballast process holds CUDA free at 23.553 GiB so the engine
#   sees the smaller CARD too: free at its own init lands at 23.553 - 0.496 (its context)
#   = 23.06 GiB, the same figure a physical 24 GiB board would report.
set -uo pipefail
cd /home/mbelleau/proxy24

run() {  # run <tag> <arm> <util> <maxlen> <mtp> <ballast_free_gib> <tokens> <text_tokens> <minprompt> <gates>
  local tag=$1 arm=$2 util=$3 maxlen=$4 mtp=$5 ballast=$6 tokens=$7 text=$8 minp=$9 gates=${10}
  echo "=== $(date -u +%FT%TZ) RUN $tag arm=$arm util=$util len=$maxlen mtp=$mtp ballast=$ballast gates='$gates'"
  TAG=$tag ARM=$arm UTIL=$util MAXLEN=$maxlen MTP=$mtp BALLAST_FREE_GIB=$ballast \
    TOKENS=$tokens TEXT_TOKENS=$text MINPROMPT=$minp GATES="$gates" \
    ./tools/proxy24_capped.sh
  echo "=== $(date -u +%FT%TZ) EXIT $tag rc=$?"
}

ONLY=${ONLY:-all}

if [[ $ONLY == all || $ONLY == budget ]]; then
  run a32e budget 0.7635 32768 3 0 32600 0 0     "g1 g2 g5 g6"
  run a32m budget 0.7635 32768 3 0 0     24000 28000 "g1 g3 g4"
  run a45e budget 0.7635 45056 0 0 44900 0 0     "g1 g2 g5 g6"
  run a45m budget 0.7635 45056 0 0 0     36000 40000 "g1 g3 g4"
fi

if [[ $ONLY == all || $ONLY == board ]]; then
  run b32e board 0.7164 32768 3 23.553 32600 0 0     "g1 g2 g5 g6"
  run b32m board 0.7164 32768 3 23.553 0     24000 28000 "g1 g3 g4"
  run b45e board 0.7164 45056 0 23.553 44900 0 0     "g1 g2 g5 g6"
  run b45m board 0.7164 45056 0 23.553 0     36000 40000 "g1 g3 g4"
fi

# Two startup-only probes at a second window, to pin the KV law rather than bound it.
# SixteenGigVerdict traced the reported token figure to
#   get_kv_cache_capacity() = int(max_concurrency * max_model_len)
#   max_concurrency = num_blocks / cdiv(max_memory_usage_per_request, memory_per_block)
# so pool = a*T + M*(T/L) and one window is one equation in two unknowns. A second
# window closes it. Either outcome is a datum: if 131,072 starts, the (pool, tokens,
# window) triple is the second equation; if it refuses, the engine prints the KV GiB it
# needed for one request, which IS a*L + M directly and needs no regression at all.
if [[ $ONLY == all || $ONLY == probe ]]; then
  run p131k3   budget 0.7635 131072 3 0 0 0 0 "g1"
  run p131koff budget 0.7635 131072 0 0 0 0 0 "g1"
fi

# A second refusal window. The 131,072 refusals printed both the KV needed for one
# request and an "estimated maximum model length" for the pool on hand, which is two
# points on the same line; 262,144 widens the lever and is the native window, so the
# three points together pin a and M without any regression across configurations.
if [[ $ONLY == probe2 ]]; then
  run p262k3   budget 0.7635 262144 3 0 0 0 0 "g1"
  run p262koff budget 0.7635 262144 0 0 0 0 0 "g1"
fi

echo "CHAIN DONE $(date -u +%FT%TZ)"
