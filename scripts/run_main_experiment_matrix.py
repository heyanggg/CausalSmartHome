#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_matrix import (
    ABLATION_VARIANT,
    PROPOSED_VARIANT,
    SEEDS,
    STATUS_COMPLETE,
    STATUS_DATA_READY,
    STATUS_DOWNSTREAM_READY,
    STATUS_GENERATION_MISSING,
    experiment_grid,
    check_matrix_cell_data_ready,
    existing_stage_dir,
    resolve_matrix_cell_paths,
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
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    parser.add_argument(
        "--allow-cpu-smoke-test",
        action="store_true",
        default=os.environ.get("CSH_ALLOW_CPU_SMOKE_TEST") == "1",
        help="Run a tiny CPU smoke test into outputs/smoke_tests only; never marks formal cells COMPLETE.",
    )
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


def cuda_preflight(python_bin: str) -> dict[str, Any]:
    code = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " ok=bool(torch.cuda.is_available())\n"
        " count=int(torch.cuda.device_count())\n"
        " cur=int(torch.cuda.current_device()) if ok and count else None\n"
        " name=torch.cuda.get_device_name(cur) if cur is not None else None\n"
        " payload={'torch_importable': True, 'torch_version': torch.__version__, 'cuda_available': ok, "
        "'cuda_device_count': count, 'device_name': name, 'current_device': cur, "
        "'cudnn_enabled': bool(getattr(torch.backends, 'cudnn', None) and torch.backends.cudnn.enabled)}\n"
        "except Exception as exc:\n"
        " payload={'torch_importable': False, 'torch_version': None, 'cuda_available': False, "
        "'cuda_device_count': 0, 'device_name': None, 'current_device': None, 'cudnn_enabled': None, 'error': str(exc)}\n"
        "print(json.dumps(payload))\n"
    )
    completed = subprocess.run([python_bin, "-c", code], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"torch_importable": False, "cuda_available": False, "error": completed.stderr or completed.stdout}
    payload["returncode"] = completed.returncode
    return payload


def write_failed_env_no_cuda(args: argparse.Namespace, preflight: dict[str, Any]) -> None:
    summary = args.out_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": item.dataset,
            "scenario": item.scenario,
            "seed": seed,
            "status": "FAILED_ENV_NO_CUDA",
            "missing": ["CUDA is required for formal downstream/TOF execution."],
            "cuda_preflight": preflight,
        }
        for item in experiment_grid()
        for seed in SEEDS
    ]
    (summary / "matrix_downstream_results.json").write_text(json.dumps(jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    write_status_report(args.out_root, rows)


def cell_paths(out_root: Path, dataset: str, scenario: str, seed: int) -> dict[str, Path]:
    key = scenario_key(dataset, scenario)
    return {
        "causal_gss_dir": existing_stage_dir(out_root, "causal_gss", dataset, scenario, seed),
        "generation_package_dir": out_root / "gpt55_generation_packages" / dataset / scenario / f"seed{seed}",
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
    data_status = check_matrix_cell_data_ready(dataset, scenario, REPO_ROOT)
    proposed_metrics = paths["downstream_dir"] / PROPOSED_VARIANT / "normalized_metrics.json"
    ablation_metrics = paths["downstream_dir"] / ABLATION_VARIANT / "normalized_metrics.json"
    missing_generation = []
    missing_downstream = []
    if data_status.status == STATUS_DATA_READY and not paths["generation_pkl"].exists():
        missing_generation.append(f"generated_pkl: {paths['generation_pkl']}")
    if paths["generation_pkl"].exists() and (not proposed_metrics.exists() or not ablation_metrics.exists()):
        for name in ("gen_tof_pkl", "causal_tof_pkl"):
            if name == "causal_tof_pkl" and ablation_metrics.exists() and proposed_metrics.exists():
                continue
            if not paths[name].exists():
                missing_downstream.append(f"{name}: {paths[name]}")
        if not ablation_metrics.exists():
            missing_downstream.append(f"ablation_metrics: {ablation_metrics}")
        if not proposed_metrics.exists():
            missing_downstream.append(f"proposed_metrics: {proposed_metrics}")
    if proposed_metrics.exists() and ablation_metrics.exists():
        status = STATUS_COMPLETE
    elif data_status.status != STATUS_DATA_READY:
        status = data_status.status
    elif missing_generation:
        status = STATUS_GENERATION_MISSING
    else:
        status = STATUS_DOWNSTREAM_READY
    missing = list(data_status.missing) + missing_generation + missing_downstream
    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "status": status,
        "missing": missing,
        "missing_data": list(data_status.missing_data),
        "missing_attack": list(data_status.missing_attack),
        "missing_checkpoint": list(data_status.missing_checkpoint),
        "missing_generation": missing_generation,
        "missing_downstream": missing_downstream,
        "paths": {key: str(value) for key, value in paths.items()},
        "resources": data_status.paths.to_dict(),
        "ablation_metrics": str(ablation_metrics),
        "historical_sp_st_subset": dataset == "sp" and scenario == "st",
    }


def build_generation_package(args: argparse.Namespace, dataset: str, scenario: str, seed: int) -> dict[str, Any]:
    paths = cell_paths(args.out_root, dataset, scenario, seed)
    causal_gss_dir = paths["causal_gss_dir"]
    data_status = check_matrix_cell_data_ready(dataset, scenario, REPO_ROOT)
    if data_status.status != STATUS_DATA_READY:
        return {
            "status": data_status.status,
            "dataset": dataset,
            "scenario": scenario,
            "seed": seed,
            "missing": list(data_status.missing),
        }
    build_gss_result = None
    if not (causal_gss_dir / "prompt.txt").exists():
        cell = resolve_matrix_cell_paths(dataset, scenario, REPO_ROOT)
        build_gss_cmd = [
            args.python_bin,
            "scripts/build_causal_gss_prompt.py",
            "--source-pkl",
            str(cell.source_train_pkl),
            "--target-pkl",
            str(cell.target_split_test_pkl),
            "--dataset",
            dataset,
            "--scenario",
            scenario,
            "--seed",
            str(seed),
            "--adapter-mode",
            "compact_fallback",
            "--out-prompt",
            str(causal_gss_dir / "prompt.txt"),
            "--out-prior-json",
            str(causal_gss_dir / "resolved_causal_relation_prior.json"),
            "--out-guard-report",
            str(causal_gss_dir / "guard_report.json"),
            "--out-reweighted-hints",
            str(causal_gss_dir / "guarded_reweighted_gss_hints.json"),
            "--out-config",
            str(causal_gss_dir / "config.json"),
        ]
        build_gss_result = run_cmd(build_gss_cmd, args.dry_run)
        if build_gss_result["status"] == "failed":
            return {"status": "failed", "dataset": dataset, "scenario": scenario, "seed": seed, "steps": [build_gss_result]}
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
    package_result = run_cmd(cmd, args.dry_run)
    out = {
        "status": package_result["status"],
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "generation_package_dir": str(paths["generation_package_dir"]),
        "steps": [step for step in (build_gss_result, package_result) if step is not None],
    }
    if package_result["status"] == "DRY_RUN":
        out["status"] = "DRY_RUN"
    return out


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
    input_root = getattr(args, "input_out_root", args.out_root)
    input_paths = cell_paths(input_root, dataset, scenario, seed)
    paths = cell_paths(args.out_root, dataset, scenario, seed)
    resources = resource_paths(RESOURCES_ROOT, dataset, scenario)
    required = {
        "generation_pkl": input_paths["generation_pkl"],
        "guarded_hints": input_paths["causal_gss_dir"] / "guarded_reweighted_gss_hints.json",
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
            str(input_paths["generation_pkl"]),
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
            "scripts/run_gen_downstream_ad.py",
            "--dataset",
            dataset,
            "--scenario",
            scenario,
            "--variant",
            ABLATION_VARIANT,
            "--generated-pkl",
            str(paths["gen_tof_pkl"]),
            "--pre-tof-pkl",
            str(input_paths["generation_pkl"]),
            "--seed",
            str(seed),
            "--out-dir",
            str(paths["downstream_dir"] / ABLATION_VARIANT),
            "--gen-root",
            str(GEN_ROOT),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--cuda-visible-devices",
            args.cuda_visible_devices,
        ],
        [
            args.python_bin,
            "scripts/run_causal_tof.py",
            "--generated-pkl",
            str(paths["gen_tof_pkl"]),
            "--guarded-hints-json",
            str(input_paths["causal_gss_dir"] / "guarded_reweighted_gss_hints.json"),
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
            str(input_paths["generation_pkl"]),
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
    if getattr(args, "smoke_execution_mode", False):
        commands[0].append("--allow-cpu-smoke-test")
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
        "## Status Counts",
        "",
        f"- COMPLETE: {sum(1 for row in rows if row['status'] == STATUS_COMPLETE)}/{len(rows)}",
        f"- MISSING_DATA: {sum(1 for row in rows if row['status'] == 'MISSING_DATA')}",
        f"- GENERATION_MISSING: {sum(1 for row in rows if row['status'] == STATUS_GENERATION_MISSING)}",
        f"- DOWNSTREAM_MISSING: {sum(1 for row in rows if row['status'] == STATUS_DOWNSTREAM_READY)}",
        f"- FAILED: {sum(1 for row in rows if str(row['status']).startswith('FAILED'))}",
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


def write_generation_package_index(out_root: Path, rows: list[dict[str, Any]]) -> None:
    package_root = out_root / "gpt55_generation_packages"
    package_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for item in experiment_grid():
        for seed in SEEDS:
            package_dir = package_root / item.dataset / item.scenario / f"seed{seed}"
            schema = package_dir / "generation_schema.json"
            instruction = package_dir / "generation_instruction.md"
            status = next((row["status"] for row in rows if row["dataset"] == item.dataset and row["scenario"] == item.scenario and row["seed"] == seed), "UNKNOWN")
            if schema.exists():
                package_status = "PACKAGE_READY"
            else:
                package_status = status
            index_rows.append(
                {
                    "dataset": item.dataset,
                    "scenario": item.scenario,
                    "seed": seed,
                    "source_context": item.source_context,
                    "target_context": item.target_context,
                    "status": package_status,
                    "package_dir": str(package_dir),
                    "schema_json": str(schema),
                    "instruction_md": str(instruction),
                    "expected_jsonl": str(out_root / "gpt55_generation" / f"{item.dataset}_{item.scenario}" / f"seed{seed}" / "generated_gpt55_clean.jsonl"),
                }
            )
    (package_root / "generation_package_index.json").write_text(json.dumps(jsonable(index_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# GPT-5.5 Generation Package Index",
        "",
        "| dataset | scenario | seed | status | package | expected JSONL |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in index_rows:
        lines.append(
            f"| {row['dataset']} | {row['scenario']} | {row['seed']} | {row['status']} | "
            f"{row['package_dir']} | {row['expected_jsonl']} |"
        )
    (package_root / "generation_package_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ready = [row for row in index_rows if row["status"] == "PACKAGE_READY"]
    readme = [
        "# Send To GPT-5.5",
        "",
        "Use one package per dataset-scenario-seed. Do not mix seeds, and do not change the JSON schema.",
        "",
        "GPT-5.5 should return one JSONL file per package using the exact schema in `generation_schema.json`.",
        "",
        "Place returned files under:",
        "",
        "```text",
        "outputs/main_experiment/gpt55_generation/<dataset>_<scenario>/seed<seed>/generated_gpt55_clean.jsonl",
        "```",
        "",
        "Then run:",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/run_main_experiment_matrix.py --stage validate_generation --matrix all",
        "PYTHONPATH=. python scripts/run_main_experiment_matrix.py --stage downstream --matrix all",
        "PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all",
        "```",
        "",
        "## Packages Ready",
        "",
        "| dataset | scenario | seed | package | expected JSONL |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in ready:
        readme.append(f"| {row['dataset']} | {row['scenario']} | {row['seed']} | {row['package_dir']} | {row['expected_jsonl']} |")
    (package_root / "README_SEND_TO_GPT55.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    smoke_execution_mode = False
    if args.stage == "downstream":
        preflight = cuda_preflight(args.python_bin)
        cuda_available = bool(preflight.get("cuda_available"))
        if args.require_cuda and not cuda_available and not args.allow_cpu_smoke_test:
            write_failed_env_no_cuda(args, preflight)
            print("FAILED_ENV_NO_CUDA: CUDA is required for formal downstream/TOF execution.")
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        if not cuda_available and args.allow_cpu_smoke_test:
            smoke_execution_mode = True
            args.input_out_root = args.out_root
            args.out_root = REPO_ROOT / "outputs" / "smoke_tests" / f"cpu_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            args.device = "cpu"
            args.epochs = min(args.epochs, 1)
            print(f"CPU smoke-test mode enabled; outputs will be written to: {args.out_root}")
    args.smoke_execution_mode = smoke_execution_mode
    rows = []
    stage_results = []
    selected_items = experiment_grid()
    selected_seeds = SEEDS
    if smoke_execution_mode:
        selected_items = selected_items[:1]
        selected_seeds = selected_seeds[:1]
    for item in selected_items:
        for seed in selected_seeds:
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
    if args.stage == "build_generation_package":
        write_generation_package_index(args.out_root, rows)
    print(f"matrix cells: {len(rows)}")
    status_counts = {status: sum(1 for row in rows if row["status"] == status) for status in sorted({row["status"] for row in rows})}
    print("status counts:", status_counts)
    print(f"status report: {args.out_root / 'summary' / 'matrix_status_report.md'}")


if __name__ == "__main__":
    main()
