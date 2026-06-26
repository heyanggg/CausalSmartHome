#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.gen_downstream_ad import (
    ATTACK_BY_ENV,
    DATASETS,
    DEFAULT_THRESHOLDS,
    DEFAULT_THRESHOLD_PERCENTAGES,
    ENVIRONMENTS,
    SOURCE_ENV_BY_TARGET_ENV,
    default_gen_paths,
)


GEN_ROOT = REPO_ROOT / "causal_smart_home" / "gen_core"
DATA_ROOT = REPO_ROOT / "causal_smart_home" / "resources" / "gen_data"
REFERENCE_ROOT = REPO_ROOT / "outputs" / "reference_gen"
PAPER_PATH = REPO_ROOT / "SmartGen Synthesizing Context-Aware User Behavior Data for Adaptive Smart Home Intelligence  Proce.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the vendored SmartGen data needed by the FR/SP/US main experiments.")
    parser.add_argument("--json", action="store_true", help="Print the full check report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report)
    if report["missing"]:
        raise SystemExit(1)


def build_report() -> dict[str, Any]:
    required_global = {
        "smartgen_paper_pdf": PAPER_PATH,
        "gen_original_tof_security_check": GEN_ROOT / "gen_original_tof" / "security_check.py",
        "gen_original_tof_models": GEN_ROOT / "gen_original_tof" / "models1.py",
        "downstream_ad_models": GEN_ROOT / "anomaly_detection_pipeline" / "models1.py",
        "smartgen_ad_reference_json": REFERENCE_ROOT / "anomaly_detection_pipeline_results" / "SmartGen_results_20250730_221851.json",
        "traditional_ad_baseline_json": REFERENCE_ROOT / "anomaly_detection_baseline_results" / "anomaly_detection_all_20250731_170324.json",
    }

    rows = []
    missing: list[dict[str, str]] = []
    for dataset in DATASETS:
        for env in ENVIRONMENTS:
            row = check_cell(dataset, env)
            rows.append(row)
            missing.extend(row["missing"])

    for name, path in required_global.items():
        if not path.exists():
            missing.append({"cell": "global", "name": name, "path": str(path)})

    return {
        "status": "ok" if not missing else "missing",
        "datasets": list(DATASETS),
        "environments": list(ENVIRONMENTS),
        "num_cells": len(rows),
        "missing": missing,
        "rows": rows,
        "global_paths": {name: str(path) for name, path in required_global.items()},
    }


def check_cell(dataset: str, env: str) -> dict[str, Any]:
    threshold = DEFAULT_THRESHOLDS[(dataset, env)]
    percentage = DEFAULT_THRESHOLD_PERCENTAGES[(dataset, env)]
    source_env = SOURCE_ENV_BY_TARGET_ENV[env]
    defaults = default_gen_paths(GEN_ROOT, dataset, env)

    required_paths = {
        "source_train": DATA_ROOT / dataset / source_env / "trn.pkl",
        "target_test": DATA_ROOT / dataset / env / "test.pkl",
        "target_split_test": DATA_ROOT / dataset / env / "split_test.pkl",
        "downstream_attack": defaults["attack_pkl"],
        "downstream_target_test": defaults["target_test_pkl"],
        "downstream_synthetic_smartgen": GEN_ROOT
        / "anomaly_detection_pipeline"
        / "synthetic_data"
        / f"{dataset}_{env}_generation_SPPC_th={threshold}_gpt-4o_seq_filter_true.pkl",
        "downstream_checkpoint": GEN_ROOT / "anomaly_detection_pipeline" / "check_model" / f"best_{dataset}_gpt-4o_SPPC.pth",
        "tof_checkpoint": GEN_ROOT / "gen_original_tof" / "check_model" / f"best_{dataset}_gpt-4o_SPPC.pth",
    }
    optional_paths = {
        "target_train": DATA_ROOT / dataset / env / "trn.pkl",
        "target_validation": DATA_ROOT / dataset / env / "vld.pkl",
        "target_rs_validation": DATA_ROOT / dataset / env / "rs_vld.pkl",
    }
    paths = {**required_paths, **optional_paths}
    missing = [
        {"cell": f"{dataset}_{env}", "name": name, "path": str(path)}
        for name, path in required_paths.items()
        if not path.exists()
    ]
    return {
        "dataset": dataset,
        "env": env,
        "source_env": source_env,
        "attack": ATTACK_BY_ENV[env],
        "threshold": threshold,
        "threshold_percentage": percentage,
        "status": "ok" if not missing else "missing",
        "missing": missing,
        "counts": {name: pickle_len(path) for name, path in paths.items() if path.suffix == ".pkl" and path.exists()},
        "paths": {name: str(path) for name, path in paths.items()},
        "optional_missing": [
            {"cell": f"{dataset}_{env}", "name": name, "path": str(path)}
            for name, path in optional_paths.items()
            if not path.exists()
        ],
    }


def pickle_len(path: Path) -> int | None:
    try:
        with open(path, "rb") as f:
            return len(pickle.load(f))
    except Exception:
        return None


def print_text_report(report: dict[str, Any]) -> None:
    print(f"GEN_MAIN_DATA_STATUS: {report['status']}")
    print(f"cells: {report['num_cells']} ({', '.join(report['datasets'])} x {', '.join(report['environments'])})")
    for row in report["rows"]:
        label = f"{row['dataset']}_{row['env']}"
        print(
            f"- {label}: {row['status']} "
            f"(source={row['source_env']}, th={row['threshold']}, pct={row['threshold_percentage']})"
        )
    if report["missing"]:
        print("\nMissing files:")
        for item in report["missing"]:
            print(f"- [{item['cell']}] {item['name']}: {item['path']}")


if __name__ == "__main__":
    main()
