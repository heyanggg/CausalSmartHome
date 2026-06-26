#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="$ROOT" "$PYTHON_BIN" scripts/check_recovery_integrity.py
PYTHONPATH="$ROOT" pytest -q
PYTHONPATH="$ROOT" "$PYTHON_BIN" scripts/summarize_main_experiment.py \
  --runs-root "$ROOT/outputs/main_experiment/downstream_ad" \
  --out-dir "$ROOT/outputs/recovery_summary_spst"
echo "Safe SP-ST summary written to outputs/recovery_summary_spst"
