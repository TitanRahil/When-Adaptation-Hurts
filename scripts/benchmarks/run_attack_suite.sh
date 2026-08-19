#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYBIN="${PYBIN:-python}"
$PYBIN scripts/benchmarks/preflight_check.py
$PYBIN scripts/benchmarks/generate_attack_benchmark.py
$PYBIN scripts/benchmarks/run_attack_suite.py --python-bin "$PYBIN"
$PYBIN scripts/benchmarks/aggregate_attack_results.py
