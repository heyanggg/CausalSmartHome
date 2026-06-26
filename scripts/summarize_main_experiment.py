#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_matrix import (
    ABLATION_VARIANT,
    PROPOSED_VARIANT,
    REFERENCE_VARIANT,
    SEEDS,
    STATUS_GENERATION_MISSING,
    experiment_grid,
    check_matrix_cell_data_ready,
    load_reference_baseline,
)

METRIC_FIELDS = ["precision", "recall", "f1", "accuracy", "fpr", "fnr"]
MAIN_NOTE = "Main baseline is SmartGen/Gen Table 3 reference, not ablation_no_causal_tof."

PER_SEED_FIELDS = [
    "dataset",
    "scenario",
    "seed",
    "variant",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "fpr",
    "fnr",
    "status",
    "run_dir",
    "metrics_path",
]

MAIN_VS_GEN_FIELDS = [
    "dataset",
    "scenario",
    "seed",
    "gen_precision",
    "gen_recall",
    "gen_f1",
    "proposed_precision",
    "proposed_recall",
    "proposed_f1",
    "delta_precision",
    "delta_recall",
    "delta_f1",
    "status",
    "proposed_metrics_path",
]

AGGREGATE_FIELDS = [
    "dataset",
    "scenario",
    "proposed_precision_mean",
    "proposed_precision_std",
    "proposed_recall_mean",
    "proposed_recall_std",
    "proposed_f1_mean",
    "proposed_f1_std",
    "original_gen_reference_precision",
    "original_gen_reference_recall",
    "original_gen_reference_f1",
    "delta_f1_mean_vs_gen",
    "num_proposed_seeds",
    "status",
]

ABLATION_FIELDS = [
    "dataset",
    "scenario",
    "seed",
    "proposed_precision",
    "proposed_recall",
    "proposed_f1",
    "ablation_precision",
    "ablation_recall",
    "ablation_f1",
    "delta_precision",
    "delta_recall",
    "delta_f1",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CausalSmartHome matrix results.")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "outputs" / "main_experiment" / "downstream_ad")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "main_experiment" / "summary")
    parser.add_argument("--reference-json", type=Path, default=REPO_ROOT / "causal_smart_home" / "resources" / "reference" / "smartgen_table3_ad.json")
    parser.add_argument("--metrics-glob", default="**/normalized_metrics.json")
    parser.add_argument("--matrix", default="all", choices=["all"])
    parser.add_argument("--matrix-status-json", type=Path, default=REPO_ROOT / "outputs" / "main_experiment" / "summary" / "matrix_status_report.json")
    parser.add_argument("--ablation", action="store_true", help="Also print ablation summary status; files are always written.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metric_files(runs_root: Path, metrics_glob: str = "**/normalized_metrics.json") -> list[Path]:
    return sorted(path for path in runs_root.glob(metrics_glob) if path.is_file())


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def normalize_metric_row(payload: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    row = dict(payload)
    row["seed"] = _as_int(row.get("seed"))
    row["dataset"] = str(row.get("dataset", "")).lower()
    row["scenario"] = str(row.get("scenario", "")).lower()
    row["variant"] = str(row.get("variant", ""))
    row["metrics_path"] = row.get("metrics_path") or str(metrics_path.resolve())
    row["run_dir"] = row.get("run_dir") or str(metrics_path.parent.resolve())
    for field in METRIC_FIELDS:
        row[field] = _as_float(row.get(field))
    return row


def collect_per_seed_rows(runs_root: Path, metrics_glob: str = "**/normalized_metrics.json") -> list[dict[str, Any]]:
    rows = []
    for path in collect_metric_files(runs_root, metrics_glob):
        row = normalize_metric_row(load_json(path), path)
        if row.get("variant") in {PROPOSED_VARIANT, ABLATION_VARIANT}:
            rows.append(row)
    rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("scenario")), int(r.get("seed") or -1), str(r.get("variant"))))
    return rows


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        seed = row.get("seed")
        if seed is None:
            continue
        indexed[(row["dataset"], row["scenario"], int(seed), row["variant"])] = row
    return indexed


def reference_row(dataset: str, scenario: str, seed: int, reference: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    ref = reference[(dataset, scenario)]
    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "variant": REFERENCE_VARIANT,
        "precision": ref["precision"],
        "recall": ref["recall"],
        "f1": ref["f1"],
        "accuracy": None,
        "fpr": None,
        "fnr": None,
        "status": "reference",
        "run_dir": "",
        "metrics_path": ref.get("source", "SmartGen paper Table 3, SmartGen column"),
    }


def matrix_status_map(status_json: Path) -> dict[tuple[str, str, int], str]:
    if not status_json.exists():
        return {
            (item.dataset, item.scenario, seed): check_matrix_cell_data_ready(item.dataset, item.scenario, REPO_ROOT).status
            for item in experiment_grid()
            for seed in SEEDS
        }
    payload = json.loads(status_json.read_text(encoding="utf-8"))
    return {(row["dataset"], row["scenario"], int(row["seed"])): row["status"] for row in payload}


def missing_row(dataset: str, scenario: str, seed: int, variant: str, status: str = STATUS_GENERATION_MISSING) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "fpr": None,
        "fnr": None,
        "status": status,
        "run_dir": "",
        "metrics_path": "",
    }


def build_main_per_seed_rows(
    metric_rows: list[dict[str, Any]],
    reference: dict[tuple[str, str], dict[str, Any]],
    statuses: dict[tuple[str, str, int], str] | None = None,
) -> list[dict[str, Any]]:
    indexed = index_rows(metric_rows)
    rows = []
    for item in experiment_grid():
        for seed in SEEDS:
            rows.append(reference_row(item.dataset, item.scenario, seed, reference))
            status = (statuses or {}).get((item.dataset, item.scenario, seed), STATUS_GENERATION_MISSING)
            rows.append(indexed.get((item.dataset, item.scenario, seed, PROPOSED_VARIANT), missing_row(item.dataset, item.scenario, seed, PROPOSED_VARIANT, status)))
    return rows


def build_main_vs_gen_rows(
    metric_rows: list[dict[str, Any]],
    reference: dict[tuple[str, str], dict[str, Any]],
    statuses: dict[tuple[str, str, int], str] | None = None,
) -> list[dict[str, Any]]:
    indexed = index_rows(metric_rows)
    rows = []
    for item in experiment_grid():
        gen = reference[(item.dataset, item.scenario)]
        for seed in SEEDS:
            proposed = indexed.get((item.dataset, item.scenario, seed, PROPOSED_VARIANT))
            row = {
                "dataset": item.dataset,
                "scenario": item.scenario,
                "seed": seed,
                "gen_precision": gen["precision"],
                "gen_recall": gen["recall"],
                "gen_f1": gen["f1"],
                "proposed_precision": proposed.get("precision") if proposed else None,
                "proposed_recall": proposed.get("recall") if proposed else None,
                "proposed_f1": proposed.get("f1") if proposed else None,
                "status": "success" if proposed else (statuses or {}).get((item.dataset, item.scenario, seed), STATUS_GENERATION_MISSING),
                "proposed_metrics_path": proposed.get("metrics_path") if proposed else "",
            }
            for metric in ("precision", "recall", "f1"):
                proposed_value = row[f"proposed_{metric}"]
                gen_value = row[f"gen_{metric}"]
                row[f"delta_{metric}"] = proposed_value - gen_value if proposed_value is not None else None
            rows.append(row)
    return rows


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def build_aggregate_rows(
    metric_rows: list[dict[str, Any]],
    reference: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = index_rows(metric_rows)
    rows = []
    for item in experiment_grid():
        proposed_rows = [
            indexed[(item.dataset, item.scenario, seed, PROPOSED_VARIANT)]
            for seed in SEEDS
            if (item.dataset, item.scenario, seed, PROPOSED_VARIANT) in indexed
        ]
        row: dict[str, Any] = {"dataset": item.dataset, "scenario": item.scenario}
        for metric in ("precision", "recall", "f1"):
            values = [float(proposed[metric]) for proposed in proposed_rows if proposed.get(metric) is not None]
            mean, std = mean_std(values)
            row[f"proposed_{metric}_mean"] = mean
            row[f"proposed_{metric}_std"] = std
        ref = reference[(item.dataset, item.scenario)]
        row["original_gen_reference_precision"] = ref["precision"]
        row["original_gen_reference_recall"] = ref["recall"]
        row["original_gen_reference_f1"] = ref["f1"]
        row["delta_f1_mean_vs_gen"] = row["proposed_f1_mean"] - ref["f1"] if row["proposed_f1_mean"] is not None else None
        row["num_proposed_seeds"] = len(proposed_rows)
        row["status"] = "COMPLETE" if len(proposed_rows) == len(SEEDS) else "MISSING"
        rows.append(row)
    return rows


def build_ablation_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = index_rows(metric_rows)
    rows = []
    for item in experiment_grid():
        for seed in SEEDS:
            proposed = indexed.get((item.dataset, item.scenario, seed, PROPOSED_VARIANT))
            ablation = indexed.get((item.dataset, item.scenario, seed, ABLATION_VARIANT))
            row = {
                "dataset": item.dataset,
                "scenario": item.scenario,
                "seed": seed,
                "proposed_precision": proposed.get("precision") if proposed else None,
                "proposed_recall": proposed.get("recall") if proposed else None,
                "proposed_f1": proposed.get("f1") if proposed else None,
                "ablation_precision": ablation.get("precision") if ablation else None,
                "ablation_recall": ablation.get("recall") if ablation else None,
                "ablation_f1": ablation.get("f1") if ablation else None,
                "status": "success" if proposed and ablation else "MISSING_DOWNSTREAM_RESULT",
            }
            for metric in ("precision", "recall", "f1"):
                proposed_value = row[f"proposed_{metric}"]
                ablation_value = row[f"ablation_{metric}"]
                row[f"delta_{metric}"] = proposed_value - ablation_value if proposed_value is not None and ablation_value is not None else None
            rows.append(row)
    return rows


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "MISSING" if value == "MISSING" else ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str], note: str) -> None:
    lines = [f"# {title}", "", MAIN_NOTE, "", note, ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_bundle(out_dir: Path, stem: str, title: str, rows: list[dict[str, Any]], fields: list[str], note: str) -> None:
    write_csv(out_dir / f"{stem}.csv", rows, fields)
    (out_dir / f"{stem}.json").write_text(json.dumps(jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_table(out_dir / f"{stem}.md", title, rows, fields, note)


def write_outputs(out_dir: Path, main_per_seed: list[dict[str, Any]], main_vs_gen: list[dict[str, Any]], aggregate: list[dict[str, Any]], ablation: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table_bundle(
        out_dir,
        "main_comparison_per_seed",
        "Main Comparison Per Seed",
        main_per_seed,
        PER_SEED_FIELDS,
        "The paper-reported SmartGen/Gen reference has no seed; it is repeated per seed for alignment.",
    )
    write_table_bundle(
        out_dir,
        "main_comparison_vs_gen",
        "Main Comparison vs Gen Reference",
        main_vs_gen,
        MAIN_VS_GEN_FIELDS,
        "Deltas are proposed minus SmartGen/Gen Table 3 reference. Missing proposed runs are explicit.",
    )
    write_table_bundle(
        out_dir,
        "main_comparison_aggregate",
        "Main Comparison Aggregate",
        aggregate,
        AGGREGATE_FIELDS,
        "Aggregate rows summarize proposed seeds per dataset-scenario and keep the reference baseline alongside them.",
    )
    write_table_bundle(
        out_dir,
        "ablation_causal_tof",
        "Ablation: effect of Causal-TOF",
        ablation,
        ABLATION_FIELDS,
        "This table compares proposed against ablation_no_causal_tof only; it is not the main baseline.",
    )
    # Compatibility names from the old script now point to the corrected main-vs-reference content.
    write_table_bundle(
        out_dir,
        "main_experiment_aggregate",
        "Main Comparison Aggregate",
        aggregate,
        AGGREGATE_FIELDS,
        "Compatibility copy of main_comparison_aggregate; ablation_no_causal_tof is not a main baseline.",
    )
    write_table_bundle(
        out_dir,
        "main_experiment_per_seed",
        "Main Comparison Per Seed",
        main_per_seed,
        PER_SEED_FIELDS,
        "Compatibility copy of main_comparison_per_seed; ablation rows are excluded from main comparison.",
    )
    write_table_bundle(
        out_dir,
        "main_experiment_seed_deltas",
        "Main Comparison vs Gen Reference",
        main_vs_gen,
        MAIN_VS_GEN_FIELDS,
        "Compatibility copy of main_comparison_vs_gen; deltas are proposed minus SmartGen/Gen Table 3 reference.",
    )


def main() -> None:
    args = parse_args()
    metric_rows = collect_per_seed_rows(args.runs_root.resolve(), args.metrics_glob)
    reference = load_reference_baseline(args.reference_json.resolve())
    statuses = matrix_status_map(args.matrix_status_json.resolve())
    main_per_seed = build_main_per_seed_rows(metric_rows, reference, statuses)
    main_vs_gen = build_main_vs_gen_rows(metric_rows, reference, statuses)
    aggregate = build_aggregate_rows(metric_rows, reference)
    ablation = build_ablation_rows(metric_rows)
    write_outputs(args.out_dir.resolve(), main_per_seed, main_vs_gen, aggregate, ablation)
    print(f"main per-seed rows: {len(main_per_seed)}")
    print(f"main vs Gen rows: {len(main_vs_gen)}")
    print(f"aggregate rows: {len(aggregate)}")
    print(f"ablation rows: {len(ablation)}")
    print(f"saved summaries: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
