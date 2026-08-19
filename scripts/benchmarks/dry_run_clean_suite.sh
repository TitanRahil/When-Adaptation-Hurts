#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python scripts/benchmarks/preflight_check.py
python scripts/benchmarks/generate_clean_benchmark.py
python scripts/benchmarks/run_clean_suite.py --dry-run
