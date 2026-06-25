#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    experiment_grid,
    existing_stage_dir,
    load_reference_baseline,
    missing_paths,
    resource_paths,
)

OUT_ROOT = REPO_ROOT / "outputs" / "main_experiment"
RESOURCES_ROOT = REPO_ROOT / "causal_smart_home" / "resources" / "gen_data"
REFERENCE_JSON = REPO_ROOT / "causal_smart_home" / "resources" / "reference" / "smartgen_table3_ad.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the current main experiment matrix status.")
    parser.add_argument("--matrix", default="all", choices=["all"])
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs" / "main_experiment_frozen")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def cell_status(dataset: str, scenario: str, seed: int) -> dict[str, Any]:
    resources = resource_paths(RESOURCES_ROOT, dataset, scenario)
    causal_gss = existing_stage_dir(OUT_ROOT, "causal_gss", dataset, scenario, seed)
    generation = existing_stage_dir(OUT_ROOT, "gpt55_generation", dataset, scenario, seed)
    gen_tof = existing_stage_dir(OUT_ROOT, "gen_original_tof", dataset, scenario, seed)
    causal_tof = existing_stage_dir(OUT_ROOT, "causal_tof", dataset, scenario, seed)
    downstream = OUT_ROOT / "downstream_ad" / f"{dataset}_{scenario}" / f"seed{seed}"
    summary = OUT_ROOT / "summary"
    paths = {
        "generated_data_path": generation / "generated_gpt55_clean.pkl",
        "causal_gss_path": causal_gss / "guarded_reweighted_gss_hints.json",
        "gen_original_tof_path": gen_tof / "gen_tof.pkl",
        "causal_tof_path": causal_tof / "generated_gen_tof_causal_tof.pkl",
        "downstream_ad_path": downstream / PROPOSED_VARIANT / "normalized_metrics.json",
        "ablation_downstream_ad_path": downstream / ABLATION_VARIANT / "normalized_metrics.json",
        "summary_path": summary / "main_comparison_vs_gen.md",
        "source_pkl": resources["source_pkl"],
        "target_split_pkl": resources["target_split_pkl"],
        "original_gen_reference_json": REFERENCE_JSON,
    }
    missing = missing_paths({key: value for key, value in paths.items() if key != "ablation_downstream_ad_path"})
    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "status": "COMPLETE" if not missing else "MISSING",
        "missing": missing,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def copy_existing_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_snapshot(frozen_dir: Path) -> None:
    copy_existing_tree(OUT_ROOT / "summary", frozen_dir / "summary")
    copy_existing_tree(OUT_ROOT / "gpt55_generation", frozen_dir / "generated")
    copy_existing_tree(OUT_ROOT / "causal_gss", frozen_dir / "causal_gss")
    copy_existing_tree(OUT_ROOT / "gen_original_tof", frozen_dir / "gen_original_tof")
    copy_existing_tree(OUT_ROOT / "causal_tof", frozen_dir / "causal_tof")
    copy_existing_tree(OUT_ROOT / "downstream_ad", frozen_dir / "downstream_ad")
    ref_dst = frozen_dir / "reference"
    ref_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFERENCE_JSON, ref_dst / "smartgen_table3_ad.json")


def write_manifest(frozen_dir: Path, statuses: list[dict[str, Any]]) -> None:
    complete = [row for row in statuses if row["status"] == "COMPLETE"]
    missing = [row for row in statuses if row["status"] == "MISSING"]
    manifest = {
        "freeze_name": frozen_dir.name,
        "main_baseline": "original_gen_reference / SmartGen paper Table 3 SmartGen column",
        "proposed_method": PROPOSED_VARIANT,
        "ablation_scope": {
            ABLATION_VARIANT: "Ablation only; never the main baseline.",
        },
        "experiment_matrix": [item.to_dict() for item in experiment_grid()],
        "seeds": SEEDS,
        "status_summary": {
            "complete_cells": len(complete),
            "missing_cells": len(missing),
            "complete_dataset_scenarios": sorted({f"{row['dataset']}-{row['scenario']}" for row in complete}),
            "missing_dataset_scenarios": sorted({f"{row['dataset']}-{row['scenario']}" for row in missing}),
        },
        "per_dataset_scenario_seed_status": statuses,
        "original_gen_reference_json": str(REFERENCE_JSON),
        "original_gen_reference": list(load_reference_baseline(REFERENCE_JSON).values()),
        "historical_note": "Existing SP-ST frozen output is a historical single-scenario subset, not the complete main experiment matrix.",
    }
    (frozen_dir / "MANIFEST.json").write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def write_docs(frozen_dir: Path, statuses: list[dict[str, Any]]) -> None:
    complete_pairs = sorted({f"{row['dataset'].upper()}-{row['scenario'].upper()}" for row in statuses if row["status"] == "COMPLETE"})
    missing_pairs = sorted({f"{row['dataset'].upper()}-{row['scenario'].upper()}" for row in statuses if row["status"] == "MISSING"})
    text = "\n".join(
        [
            "# Frozen Main Matrix Status",
            "",
            "Main baseline is SmartGen/Gen Table 3 reference, not ablation_no_causal_tof.",
            "",
            f"Complete cells: {len([row for row in statuses if row['status'] == 'COMPLETE'])}",
            f"Missing cells: {len([row for row in statuses if row['status'] == 'MISSING'])}",
            "",
            f"Complete dataset-scenarios: {', '.join(complete_pairs) if complete_pairs else 'none'}",
            f"Missing dataset-scenarios: {', '.join(missing_pairs) if missing_pairs else 'none'}",
            "",
            "The legacy `sp_st_gpt55_proposed_3seed_20260623` archive is retained as historical SP-ST-only evidence.",
            "",
        ]
    )
    (frozen_dir / "README_FROZEN.md").write_text(text, encoding="utf-8")


def write_provenance(frozen_dir: Path) -> None:
    prov = frozen_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "git_status.txt").write_text(run_text(["git", "status", "--short"]), encoding="utf-8")
    (prov / "python_env.txt").write_text(
        run_text([sys.executable, "-c", "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"]),
        encoding="utf-8",
    )


def write_checksums(frozen_dir: Path) -> None:
    checksum_path = frozen_dir / "provenance" / "checksums.sha256"
    rows: list[str] = []
    for path in sorted(p for p in frozen_dir.rglob("*") if p.is_file()):
        if path == checksum_path:
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(frozen_dir)}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_text(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    out = completed.stdout
    if completed.stderr:
        out += "\n[stderr]\n" + completed.stderr
    if completed.returncode != 0:
        out += f"\n[returncode] {completed.returncode}\n"
    return out


def main() -> None:
    args = parse_args()
    frozen_dir = (args.out_root / f"main_matrix_fr_sp_us_st_tt_nt_gpt55_3seed_{args.date_tag}").resolve()
    if frozen_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"frozen directory already exists: {frozen_dir}; pass --overwrite to replace it")
        shutil.rmtree(frozen_dir)
    frozen_dir.mkdir(parents=True)
    statuses = [cell_status(item.dataset, item.scenario, seed) for item in experiment_grid() for seed in SEEDS]
    copy_snapshot(frozen_dir)
    write_manifest(frozen_dir, statuses)
    write_docs(frozen_dir, statuses)
    write_provenance(frozen_dir)
    write_checksums(frozen_dir)
    print(f"frozen matrix status saved: {frozen_dir}")
    print(f"complete cells: {sum(1 for row in statuses if row['status'] == 'COMPLETE')}")
    print(f"missing cells: {sum(1 for row in statuses if row['status'] == 'MISSING')}")


if __name__ == "__main__":
    main()
