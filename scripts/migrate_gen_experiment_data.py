#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_matrix import (
    DATASETS,
    SCENARIOS,
    SEEDS,
    attack_suffix_for_target_context,
    canonical_context_dir,
    experiment_grid,
    matrix_item,
    resolve_matrix_cell_paths,
)

CONTEXT_SOURCE_ALIASES = {
    "winter": ("winter",),
    "spring": ("spring",),
    "daytime": ("daytime", "day"),
    "nighttime": ("night", "nighttime"),
    "single": ("single",),
    "multiple": ("multiple", "multi"),
}

TARGET_FILES = ("trn.pkl", "vld.pkl", "test.pkl", "split_test.pkl")
SOURCE_FILES = ("trn.pkl",)

FILENAME_ALIASES = {
    "trn.pkl": ("trn.pkl", "train.pkl"),
    "vld.pkl": ("vld.pkl", "val.pkl", "validation.pkl", "rs_vld.pkl"),
    "test.pkl": ("test.pkl",),
    "split_test.pkl": ("split_test.pkl",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate required Gen/SmartGen experiment data into CausalSmartHome.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--matrix", default="all", choices=["all", "single"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--dry-run", action="store_true", help="Plan only. This is the default unless --copy is passed.")
    parser.add_argument("--copy", action="store_true", help="Actually copy or symlink files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def source_roots(source: Path) -> dict[str, Path]:
    return {
        "smartgen": source / "SmartGen",
        "iot_data": source / "SmartGen" / "IoT_data",
        "smartgen_attack": source / "SmartGen" / "attack",
        "smartgen_check_model": source / "SmartGen" / "check_model",
        "smartgen_dictionary": source / "SmartGen" / "dictionary.py",
        "smartgen_models": source / "SmartGen" / "models1.py",
        "smartgen_security_check": source / "SmartGen" / "security_check.py",
        "downstream": source / "anomaly_detection_pipeline",
        "downstream_test": source / "anomaly_detection_pipeline" / "test",
        "downstream_attack": source / "anomaly_detection_pipeline" / "attack",
        "downstream_check_model": source / "anomaly_detection_pipeline" / "check_model",
        "downstream_models": source / "anomaly_detection_pipeline" / "models1.py",
    }


def choose_existing(candidates: list[Path]) -> tuple[Path | None, str, list[str]]:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None, "MISSING", [str(path) for path in candidates]
    # Prefer the first candidate; the lists are ordered by trust.
    status = "FOUND" if len(existing) == 1 else "AMBIGUOUS"
    return existing[0], status, [str(path) for path in existing]


def add_plan(plans: list[dict[str, Any]], source: Path | None, target: Path, role: str, candidates: list[str] | None = None, ambiguous: bool = False) -> None:
    if target.exists():
        status = "EXISTS"
    elif source is None:
        status = "SOURCE_MISSING"
    elif ambiguous:
        status = "AMBIGUOUS"
    else:
        status = "PENDING"
    plans.append(
        {
            "role": role,
            "source_path": str(source) if source is not None else "",
            "target_path": str(target),
            "status": status,
            "candidates": candidates or [],
            "source_size": source.stat().st_size if source is not None and source.exists() else None,
            "source_sha256": sha256(source) if source is not None and source.exists() else None,
            "target_size": target.stat().st_size if target.exists() else None,
            "target_sha256": sha256(target) if target.exists() else None,
            "alias_used": bool(source is not None and source.name != target.name),
            "source_context_resolved": source.parent.name if source is not None else "",
            "target_context": target.parent.name,
        }
    )


def normal_file_plan(source: Path, target: Path, dataset: str, context: str, filename: str) -> tuple[Path | None, str, list[str]]:
    roots = source_roots(source)
    candidates = [
        roots["iot_data"] / dataset / alias / candidate_name
        for alias in CONTEXT_SOURCE_ALIASES[context]
        for candidate_name in FILENAME_ALIASES.get(filename, (filename,))
    ]
    return choose_existing(candidates)


def add_normal_plans(plans: list[dict[str, Any]], source: Path, target: Path, dataset: str, scenario: str) -> None:
    item = matrix_item(dataset, scenario)
    paths = resolve_matrix_cell_paths(dataset, scenario, target)
    source_context_dir = target / "causal_smart_home" / "resources" / "gen_data" / dataset / canonical_context_dir(item.source_context)
    target_context_dir = target / "causal_smart_home" / "resources" / "gen_data" / dataset / canonical_context_dir(item.target_context)
    for filename in SOURCE_FILES:
        src, found_status, candidates = normal_file_plan(source, target, dataset, item.source_context, filename)
        add_plan(plans, src, source_context_dir / filename, f"{dataset}-{scenario}:source_{item.source_context}:{filename}", candidates, ambiguous=found_status == "AMBIGUOUS")
    target_files = ("test.pkl", "split_test.pkl") if item.target_context == "multiple" else TARGET_FILES
    for filename in target_files:
        src, found_status, candidates = normal_file_plan(source, target, dataset, item.target_context, filename)
        add_plan(plans, src, target_context_dir / filename, f"{dataset}-{scenario}:target_{item.target_context}:{filename}", candidates, ambiguous=found_status == "AMBIGUOUS")
    add_plan(plans, source_roots(source)["smartgen_dictionary"], paths.dictionary_py, "dictionary.py")


def add_downstream_plans(plans: list[dict[str, Any]], source: Path, target: Path, dataset: str, scenario: str) -> None:
    roots = source_roots(source)
    item = matrix_item(dataset, scenario)
    paths = resolve_matrix_cell_paths(dataset, scenario, target)
    target_env = canonical_context_dir(item.target_context)
    source_env_candidates = CONTEXT_SOURCE_ALIASES[item.target_context]
    test_candidates = [roots["downstream_test"] / dataset / alias / "test.pkl" for alias in source_env_candidates]
    src, found_status, candidates = choose_existing(test_candidates)
    add_plan(plans, src, paths.downstream_test_pkl, f"{dataset}-{scenario}:downstream_test", candidates, ambiguous=found_status == "AMBIGUOUS")

    attack_suffix = attack_suffix_for_target_context(item.target_context)
    source_attack_suffix = attack_suffix.replace("nighttime_", "night_")
    attack_names = [
        f"labeled_{dataset}_{source_attack_suffix}.pkl",
        f"{dataset}_{source_attack_suffix}.pkl",
    ]
    for name in attack_names:
        src, found_status, candidates = choose_existing([
            roots["downstream_attack"] / dataset / name,
            roots["smartgen_attack"] / dataset / name,
        ])
        target_name = name
        add_plan(plans, src, paths.downstream_attack_dir / target_name, f"{dataset}-{scenario}:attack:{name}", candidates, ambiguous=found_status == "AMBIGUOUS")

    add_plan(plans, roots["downstream_models"], target / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "models1.py", "downstream_models1.py")
    downstream_ckpt = roots["downstream_check_model"] / f"best_{dataset}_gpt-4o_SPPC.pth"
    add_plan(plans, downstream_ckpt, paths.downstream_checkpoint, f"{dataset}:downstream_checkpoint")

    tof_candidates = [
        roots["smartgen_check_model"] / f"best_{dataset}_gpt-4o_SPPC.pth",
        roots["downstream_check_model"] / f"best_{dataset}_gpt-4o_SPPC.pth",
    ]
    src, found_status, candidates = choose_existing(tof_candidates)
    add_plan(plans, src, paths.gen_original_tof_checkpoint, f"{dataset}:gen_original_tof_checkpoint", candidates, ambiguous=found_status == "AMBIGUOUS")
    add_plan(plans, roots["smartgen_security_check"], target / "causal_smart_home" / "gen_core" / "gen_original_tof" / "security_check.py", "gen_original_tof_security_check.py")


def selected_cells(args: argparse.Namespace):
    if args.matrix == "all":
        return experiment_grid()
    if not args.dataset or not args.scenario:
        raise ValueError("--dataset and --scenario are required with --matrix single")
    return [matrix_item(args.dataset, args.scenario)]


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for item in selected_cells(args):
        add_normal_plans(plans, args.source.resolve(), args.target.resolve(), item.dataset, item.scenario)
        add_downstream_plans(plans, args.source.resolve(), args.target.resolve(), item.dataset, item.scenario)
    return plans


def execute_plan(plans: list[dict[str, Any]], copy_enabled: bool, overwrite: bool, link_mode: str) -> None:
    for row in plans:
        if row["status"] == "EXISTS":
            continue
        if row["status"] == "SOURCE_MISSING":
            continue
        if row["status"] == "AMBIGUOUS":
            # Use the first ordered candidate but keep the ambiguity in the manifest.
            pass
        if not copy_enabled:
            row["status"] = "SKIPPED"
            continue
        source = Path(row["source_path"])
        target = Path(row["target_path"])
        if not source.exists():
            row["status"] = "SOURCE_MISSING"
            continue
        if target.exists() and not overwrite:
            row["status"] = "EXISTS"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and overwrite:
            target.unlink()
        if link_mode == "symlink":
            os.symlink(source, target)
            row["status"] = "LINKED"
        else:
            shutil.copy2(source, target)
            row["status"] = "COPIED"
        row["target_size"] = target.stat().st_size if target.exists() else None
        row["target_sha256"] = sha256(target) if target.exists() else None


def summarize(plans: list[dict[str, Any]], target: Path) -> dict[str, Any]:
    from causal_smart_home.experiment_matrix import check_matrix_cell_data_ready

    cell_statuses = [check_matrix_cell_data_ready(item.dataset, item.scenario, target).to_dict() for item in experiment_grid()]
    return {
        "status_counts": {status: sum(1 for row in plans if row["status"] == status) for status in sorted({row["status"] for row in plans})},
        "ready_cells": [f"{row['dataset']}-{row['scenario']}" for row in cell_statuses if row["status"] == "DATA_READY"],
        "missing_cells": [f"{row['dataset']}-{row['scenario']}" for row in cell_statuses if row["status"] != "DATA_READY"],
        "missing_files": [row for row in plans if row["status"] == "SOURCE_MISSING"],
        "ambiguous_files": [row for row in plans if row["status"] == "AMBIGUOUS"],
        "matrix_cell_status": cell_statuses,
    }


def write_manifest(args: argparse.Namespace, plans: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir = args.target.resolve() / "outputs" / "data_migration"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_root": str(args.source.resolve()),
        "target_root": str(args.target.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "copy_enabled": bool(args.copy),
        "overwrite": bool(args.overwrite),
        "link_mode": args.link_mode,
        "matrix": args.matrix,
        "entries": plans,
        "summary": summary,
    }
    (out_dir / "gen_data_migration_manifest.json").write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "gen_data_migration_report.md", manifest)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Gen Data Migration Report",
        "",
        f"Source root: `{manifest['source_root']}`",
        f"Target root: `{manifest['target_root']}`",
        f"Copy enabled: `{manifest['copy_enabled']}`",
        "",
        "## Summary",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(manifest["summary"]["status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Matrix Cell Status", "", "| cell | status | missing |", "| --- | --- | --- |"])
    for row in manifest["summary"]["matrix_cell_status"]:
        lines.append(f"| {row['dataset'].upper()}-{row['scenario'].upper()} | {row['status']} | {'<br>'.join(row['missing'])} |")
    lines.extend(["", "## Files", "", "| status | role | source | target |", "| --- | --- | --- | --- |"])
    for row in manifest["entries"]:
        lines.append(f"| {row['status']} | {row['role']} | {row['source_path']} | {row['target_path']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"--source not found: {args.source}")
    if not args.target.exists():
        raise FileNotFoundError(f"--target not found: {args.target}")
    plans = build_plan(args)
    execute_plan(plans, copy_enabled=bool(args.copy), overwrite=args.overwrite, link_mode=args.link_mode)
    summary = summarize(plans, args.target.resolve())
    if args.write_manifest or args.dry_run or args.copy:
        write_manifest(args, plans, summary)
    print(f"planned files: {len(plans)}")
    print("status counts:", summary["status_counts"])
    print(f"manifest: {args.target.resolve() / 'outputs' / 'data_migration' / 'gen_data_migration_manifest.json'}")


if __name__ == "__main__":
    main()
