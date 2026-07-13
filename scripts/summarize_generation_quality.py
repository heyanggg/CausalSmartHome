#!/usr/bin/env python
"""Collect per-seed generation-quality JSON without averaging seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.json_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def collect(runs_root: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_root.glob("**/generation_quality/*/generation_quality_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(path.resolve())
        rows.append(payload)
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("scenario")), int(row.get("seed") or -1), str(row.get("variant"))))
    return rows


def main() -> None:
    args = parse_args()
    rows = collect(args.runs_root)
    write_json(args.out, {"schema_version": 1, "aggregation": "per_seed_only", "rows": rows})
    print(f"generation quality rows: {len(rows)}")


if __name__ == "__main__":
    main()
