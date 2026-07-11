#!/usr/bin/env python
"""检查代码布局、运行资产和正式主实验结果的一致性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_paths import DEFAULT_INPUT_ROOT, PROPOSED_VARIANT
from causal_smart_home.gen_downstream_ad import DATASETS, ENVIRONMENTS, SCENARIO_BY_ENV
from scripts.check_gen_main_data import build_report as build_gen_data_report

REQUIRED_PROJECT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "causal_smart_home/cli.py",
    "scripts/main_prepare_generation.py",
    "scripts/main_run_downstream_ad.py",
)
REQUIRED_PROPOSED_FIELDS = (
    "status",
    "dataset",
    "scenario",
    "seed",
    "variant",
    "precision",
    "recall",
    "f1",
    "fpr",
    "device",
    "requested_device",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CausalSmartHome project health checks.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON.")
    return parser.parse_args()


def _issue(kind: str, path: Path, detail: str) -> dict[str, str]:
    return {"kind": kind, "path": str(path), "detail": detail}


def _read_json(
    path: Path, issues: list[dict[str, str]], *, require_object: bool = True
) -> dict[str, Any] | list[Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(_issue("invalid_json", path, str(exc)))
        return None
    if require_object and not isinstance(payload, dict):
        issues.append(_issue("invalid_json_shape", path, "top-level value must be an object"))
        return None
    return payload


def build_report(runs_root: Path = DEFAULT_INPUT_ROOT) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    for relative in REQUIRED_PROJECT_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            issues.append(_issue("missing_project_file", path, "required project file is absent"))

    runtime_links = sorted((REPO_ROOT / "causal_smart_home" / "gen_runtime").rglob("*"))
    for path in runtime_links:
        if path.is_symlink() and not path.exists():
            issues.append(_issue("broken_symlink", path, "runtime compatibility link has no target"))

    json_files = sorted((runs_root / "summary").glob("*.json")) if (runs_root / "summary").exists() else []
    metrics_files = sorted(runs_root.glob(f"*_*/seed*/downstream_ad/{PROPOSED_VARIANT}/normalized_metrics.json"))
    json_files.extend(metrics_files)
    for path in json_files:
        _read_json(path, issues, require_object=path in metrics_files)

    cells: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        for env in ENVIRONMENTS:
            key = f"{dataset}_{SCENARIO_BY_ENV[env]}"
            seed_dirs = sorted((runs_root / key).glob("seed*"))
            proposed = []
            for seed_dir in seed_dirs:
                metrics_path = seed_dir / "downstream_ad" / PROPOSED_VARIANT / "normalized_metrics.json"
                if not metrics_path.exists():
                    legacy = seed_dir / "downstream_ad" / "proposed_causal_gss_codex_causal_tof" / "normalized_metrics.json"
                    metrics_path = legacy if legacy.exists() else metrics_path
                if not metrics_path.exists():
                    issues.append(_issue("missing_proposed_metrics", metrics_path, "seed has no normalized proposed result"))
                    continue
                payload = _read_json(metrics_path, issues)
                if not isinstance(payload, dict):
                    continue
                missing_fields = [field for field in REQUIRED_PROPOSED_FIELDS if field not in payload]
                if missing_fields:
                    issues.append(_issue("missing_metric_fields", metrics_path, ", ".join(missing_fields)))
                if payload.get("variant") not in {PROPOSED_VARIANT, "proposed_causal_gss_codex_causal_tof"}:
                    issues.append(_issue("variant_mismatch", metrics_path, str(payload.get("variant"))))
                expected_seed = int(seed_dir.name[4:])
                if payload.get("seed") != expected_seed or payload.get("dataset") != dataset:
                    issues.append(_issue("experiment_coordinate_mismatch", metrics_path, "path and payload disagree"))
                proposed.append({"seed": expected_seed, "status": payload.get("status"), "f1": payload.get("f1")})
            cells[key] = {"seed_dirs": len(seed_dirs), "proposed_results": proposed}

    gen_data = build_gen_data_report()
    for item in gen_data["missing"]:
        issues.append(_issue("missing_gen_asset", Path(item["path"]), f"{item['cell']}: {item['name']}"))
    return {
        "status": "ok" if not issues else "issues",
        "project_root": str(REPO_ROOT),
        "runs_root": str(runs_root.resolve()),
        "checked_proposed_metrics": len(metrics_files),
        "checked_json_files": len(json_files),
        "gen_data_status": gen_data["status"],
        "cells": cells,
        "issues": issues,
    }


def print_text_report(report: dict[str, Any]) -> None:
    print(f"PROJECT_STATUS: {report['status']}")
    print(f"Gen data: {report['gen_data_status']}")
    print(f"Proposed metrics checked: {report['checked_proposed_metrics']}")
    for key, cell in report["cells"].items():
        seeds = ", ".join(str(row["seed"]) for row in cell["proposed_results"]) or "none"
        print(f"- {key}: seed dirs={cell['seed_dirs']}, proposed seeds={seeds}")
    if report["issues"]:
        print("\nIssues:")
        for issue in report["issues"]:
            print(f"- [{issue['kind']}] {issue['path']}: {issue['detail']}")


def main() -> None:
    args = parse_args()
    report = build_report(args.runs_root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report)
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
