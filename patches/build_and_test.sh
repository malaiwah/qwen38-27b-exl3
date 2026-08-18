#!/usr/bin/env bash
# Build and test in a single container run
set -e

export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/exllamav3-python:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

cd /build

echo "=== Building prefill kernel (full ext compilation) ==="
python3 build_full_ext.py 2>&1 | tail -10

echo ""
echo "=== Running ground truth test ==="
timeout 60 python3 test_ground_truth.py 2>&1
