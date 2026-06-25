#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_matrix import (
    ABLATION_VARIANT,
    PROPOSED_VARIANT,
    SEEDS,
    experiment_grid,
    existing_stage_dir,
    matrix_stage_dir,
    missing_paths,
    resource_paths,
    scenario_key,
)

RESOURCES_ROOT = REPO_ROOT / "causal_smart_home" / "resources" / "gen_data"
GEN_ROOT = REPO_ROOT / "causal_smart_home" / "gen_core"
OUT_ROOT = REPO_ROOT / "outputs" / "main_experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or inspect the CausalSmartHome main experiment matrix.")
    parser.add_argument("--matrix", default="all", choices=["all"])
    parser.add_argument(
        "--stage",
        default="status",
        choices=["status", "build_generation_package", "validate_generation", "downstream"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_cmd(cmd: list[str], dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "DRY_RUN", "command": cmd}
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "command": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def cell_paths(out_root: Path, dataset: str, scenario: str, seed: int) -> dict[str, Path]:
    key = scenario_key(dataset, scenario)
    return {
        "causal_gss_dir": existing_stage_dir(out_root, "causal_gss", dataset, scenario, seed),
        "generation_package_dir": matrix_stage_dir(out_root, "generation_package", dataset, scenario, seed),
        "generation_dir": existing_stage_dir(out_root, "gpt55_generation", dataset, scenario, seed),
        "generation_pkl": existing_stage_dir(out_root, "gpt55_generation", dataset, scenario, seed) / "generated_gpt55_clean.pkl",
        "generation_jsonl": existing_stage_dir(out_root, "gpt55_generation", dataset, scenario, seed) / "generated_gpt55_clean.jsonl",
        "gen_tof_dir": existing_stage_dir(out_root, "gen_original_tof", dataset, scenario, seed),
        "gen_tof_pkl": existing_stage_dir(out_root, "gen_original_tof", dataset, scenario, seed) / "gen_tof.pkl",
        "causal_tof_dir": existing_stage_dir(out_root, "causal_tof", dataset, scenario, seed),
        "causal_tof_pkl": existing_stage_dir(out_root, "causal_tof", dataset, scenario, seed) / "generated_gen_tof_causal_tof.pkl",
        "downstream_dir": out_root / "downstream_ad" / key / f"seed{seed}",
    }


def status_for_cell(out_root: Path, dataset: str, scenario: str, seed: int) -> dict[str, Any]:
    paths = cell_paths(out_root, dataset, scenario, seed)
    resources = resource_paths(RESOURCES_ROOT, dataset, scenario)
    proposed_metrics = paths["downstream_dir"] / PROPOSED_VARIANT / "normalized_metrics.json"
    ablation_metrics = paths["downstream_dir"] / ABLATION_VARIANT / "normalized_metrics.json"
    required = {
        "source_pkl": resources["source_pkl"],
        "target_split_pkl": resources["target_split_pkl"],
        "target_test_pkl": resources["target_test_pkl"],
        "generated_pkl": paths["generation_pkl"],
        "gen_tof_pkl": paths["gen_tof_pkl"],
        "causal_tof_pkl": paths["causal_tof_pkl"],
        "proposed_metrics": proposed_metrics,
    }
    missing = missing_paths(required)
    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "status": "COMPLETE" if not missing else "MISSING",
        "missing": missing,
        "paths": {key: str(value) for key, value in paths.items()},
        "resources": {key: str(value) for key, value in resources.items()},
        "ablation_metrics": str(ablation_metrics),
        "historical_sp_st_subset": dataset == "sp" and scenario == "st",
    }


def build_generation_package(args: argparse.Namespace, dataset: str, scenario: str, seed: int) -> dict[str, Any]:
    paths = cell_paths(args.out_root, dataset, scenario, seed)
    causal_gss_dir = paths["causal_gss_dir"]
    required = {
        "prompt": causal_gss_dir / "prompt.txt",
        "guard_report": causal_gss_dir / "guard_report.json",
        "guarded_hints": causal_gss_dir / "guarded_reweighted_gss_hints.json",
        "prior": causal_gss_dir / "resolved_causal_relation_prior.json",
    }
    missing = missing_paths(required)
    if missing:
        return {"status": "MISSING", "dataset": dataset, "scenario": scenario, "seed": seed, "missing": missing}
    cmd = [
        args.python_bin,
        "scripts/build_gpt55_generation_package.py",
        "--causal-gss-dir",
        str(causal_gss_dir),
        "--out-dir",
        str(paths["generation_package_dir"]),
        "--dataset",
        dataset,
        "--scenario",
        scenario,
        "--seed",
        str(seed),
    ]
    return run_cmd(cmd, args.dry_run)


def validate_generation(args: argparse.Namespace, dataset: str, scenario: str, seed: int) -> dict[str, Any]:
    paths = cell_paths(args.out_root, dataset, scenario, seed)
    resources = resource_paths(RESOURCES_ROOT, dataset, scenario)
    required = {
        "jsonl": paths["generation_jsonl"],
        "dictionary": RESOURCES_ROOT / "dictionary.py",
    }
    missing = missing_paths(required)
    if missing:
        return {"status": "MISSING", "dataset": dataset, "scenario": scenario, "seed": seed, "missing": missing}
    cmd = [
        args.python_bin,
        "scripts/validate_and_pack_gpt55_generation.py",
        "--input-jsonl",
        str(paths["generation_jsonl"]),
        "--out-pkl",
        str(paths["generation_pkl"]),
        "--out-validation-report",
        str(paths["generation_dir"] / "validation_report.json"),
        "--out-generation-report",
        str(paths["generation_dir"] / "generation_report.json"),
        "--dictionary-py",
        str(RESOURCES_ROOT / "dictionary.py"),
        "--dataset",
        dataset,
        "--scenario",
        scenario,
        "--scenario-key",
        scenario_key(dataset, scenario),
        "--seed",
        str(seed),
        "--source-pkl",
        str(resources["source_pkl"]),
        "--target-pkl",
        str(resources["target_split_pkl"]),
    ]
    return run_cmd(cmd, args.dry_run)


def downstream(args: argparse.Namespace, dataset: str, scenario: str, seed: int) -> dict[str, Any]:
    paths = cell_paths(args.out_root, dataset, scenario, seed)
    resources = resource_paths(RESOURCES_ROOT, dataset, scenario)
    required = {
        "generation_pkl": paths["generation_pkl"],
        "guarded_hints": paths["causal_gss_dir"] / "guarded_reweighted_gss_hints.json",
        "target_split_pkl": resources["target_split_pkl"],
    }
    missing = missing_paths(required)
    if missing:
        return {"status": "MISSING", "dataset": dataset, "scenario": scenario, "seed": seed, "missing": missing}
    commands = [
        [
            args.python_bin,
            "scripts/run_gen_original_tof.py",
            "--generated-pkl",
            str(paths["generation_pkl"]),
            "--dataset",
            dataset,
            "--scenario",
            scenario,
            "--seed",
            str(seed),
            "--gen-root",
            str(GEN_ROOT),
            "--out-dir",
            str(paths["gen_tof_dir"]),
            "--cuda-visible-devices",
            args.cuda_visible_devices,
        ],
        [
            args.python_bin,
            "scripts/run_causal_tof.py",
            "--generated-pkl",
            str(paths["gen_tof_pkl"]),
            "--guarded-hints-json",
            str(paths["causal_gss_dir"] / "guarded_reweighted_gss_hints.json"),
            "--target-pkl",
            str(resources["target_split_pkl"]),
            "--out-scores",
            str(paths["causal_tof_dir"] / "causal_tof_scores.json"),
            "--out-weights",
            str(paths["causal_tof_dir"] / "generated.weights.json"),
            "--out-weighted-resampled-pkl",
            str(paths["causal_tof_pkl"]),
            "--input-stage",
            "gen_original_tof",
            "--mode",
            "weight",
            "--temperature",
            "2.0",
            "--seed",
            str(seed),
        ],
        [
            args.python_bin,
            "scripts/run_gen_downstream_ad.py",
            "--dataset",
            dataset,
            "--scenario",
            scenario,
            "--variant",
            PROPOSED_VARIANT,
            "--generated-pkl",
            str(paths["causal_tof_pkl"]),
            "--pre-tof-pkl",
            str(paths["generation_pkl"]),
            "--gen-tof-pkl",
            str(paths["gen_tof_pkl"]),
            "--seed",
            str(seed),
            "--out-dir",
            str(paths["downstream_dir"] / PROPOSED_VARIANT),
            "--gen-root",
            str(GEN_ROOT),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--cuda-visible-devices",
            args.cuda_visible_devices,
        ],
    ]
    results = []
    for cmd in commands:
        result = run_cmd(cmd, args.dry_run)
        results.append(result)
        if result["status"] == "failed":
            break
    return {"status": "success" if all(item["status"] in {"success", "DRY_RUN"} for item in results) else "failed", "steps": results}


def write_status_report(out_root: Path, rows: list[dict[str, Any]]) -> None:
    summary = out_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "matrix_status_report.json").write_text(json.dumps(jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Matrix Status Report",
        "",
        "Main baseline is SmartGen/Gen Table 3 reference, not ablation_no_causal_tof.",
        "",
        "| dataset | scenario | seed | status | missing |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['scenario']} | {row['seed']} | {row['status']} | "
            f"{'<br>'.join(row.get('missing', []))} |"
        )
    (summary / "matrix_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    stage_results = []
    for item in experiment_grid():
        for seed in SEEDS:
            if args.stage == "build_generation_package":
                stage_results.append(build_generation_package(args, item.dataset, item.scenario, seed))
            elif args.stage == "validate_generation":
                stage_results.append(validate_generation(args, item.dataset, item.scenario, seed))
            elif args.stage == "downstream":
                stage_results.append(downstream(args, item.dataset, item.scenario, seed))
            rows.append(status_for_cell(args.out_root, item.dataset, item.scenario, seed))
    write_status_report(args.out_root, rows)
    if stage_results:
        status_path = args.out_root / "summary" / f"matrix_{args.stage}_results.json"
        status_path.write_text(json.dumps(jsonable(stage_results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"matrix cells: {len(rows)}")
    print(f"complete cells: {sum(1 for row in rows if row['status'] == 'COMPLETE')}")
    print(f"missing cells: {sum(1 for row in rows if row['status'] == 'MISSING')}")
    print(f"status report: {args.out_root / 'summary' / 'matrix_status_report.md'}")


if __name__ == "__main__":
    main()
