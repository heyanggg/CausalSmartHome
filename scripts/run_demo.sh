#!/usr/bin/env bash
set -euo pipefail
python -m causal_smart_home.cli demo --out-dir outputs/demo --epochs 5 --lag 3
