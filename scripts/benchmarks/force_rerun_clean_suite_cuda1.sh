#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-1}"
PYBIN="${PYBIN:-python}"
$PYBIN scripts/benchmarks/preflight_check.py
$PYBIN scripts/benchmarks/generate_clean_benchmark.py
$PYBIN scripts/benchmarks/run_clean_suite.py --python-bin "$PYBIN" --force
$PYBIN scripts/benchmarks/aggregate_clean_results.py
