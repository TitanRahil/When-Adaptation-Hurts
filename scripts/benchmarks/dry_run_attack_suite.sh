#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python scripts/benchmarks/preflight_check.py
python scripts/benchmarks/generate_attack_benchmark.py
python scripts/benchmarks/run_attack_suite.py --dry-run
