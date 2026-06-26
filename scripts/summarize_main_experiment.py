#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PER_SEED_FIELDS = [
    "dataset",
    "scenario",
    "seed",
    "variant",
    "input_pkl",
    "input_stage",
    "used_gen_original_tof",
    "used_causal_tof",
    "downstream_pipeline",
    "generator",
    "generation_model",
    "num_generated_before_tof",
    "num_generated_after_gen_tof",
    "num_generated_after_causal_tof",
    "train_size",
    "validation_size",
    "test_size",
    "threshold",
    "threshold_source",
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

METRIC_FIELDS = ["precision", "recall", "f1", "accuracy", "fpr", "fnr"]

KEPT_VARIANTS = {
    "ablation_no_causal_tof",
    "proposed_causal_gss_codex_causal_tof",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the current main experiment downstream AD results.")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "outputs" / "main_experiment")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "main_experiment" / "summary")
    parser.add_argument("--metrics-glob", default="**/normalized_metrics.json")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metric_files(runs_root: Path, metrics_glob: str = "**/normalized_metrics.json") -> list[Path]:
    """Collect normalized per-run metrics."""
    return sorted(path for path in runs_root.glob(metrics_glob) if path.is_file() and not is_diagnostic_metric_path(path))


def is_diagnostic_metric_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.parent.name.lower()
    return (
        "diagnostic" in parts
        or "diagnostics" in parts
        or "_beta" in name
        or name.endswith("_beta0")
    )


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def normalize_metric_row(payload: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    row = {field: payload.get(field, "") for field in PER_SEED_FIELDS}
    row["seed"] = _as_int(row.get("seed"))
    row["metrics_path"] = row.get("metrics_path") or str(metrics_path.resolve())
    row["run_dir"] = row.get("run_dir") or str(metrics_path.parent.resolve())
    if not row.get("input_pkl"):
        row["input_pkl"] = payload.get("synthetic_pkl", "")
    for key in [
        "used_gen_original_tof",
        "used_causal_tof",
    ]:
        value = row.get(key)
        if isinstance(value, str):
            row[key] = value.lower() == "true" if value.lower() in {"true", "false"} else value
    for key in METRIC_FIELDS + ["threshold"]:
        row[key] = _as_float(row.get(key))
    for key in [
        "num_generated_before_tof",
        "num_generated_after_gen_tof",
        "num_generated_after_causal_tof",
        "train_size",
        "validation_size",
        "test_size",
    ]:
        row[key] = _as_int(row.get(key))
    return row


def collect_per_seed_rows(runs_root: Path, metrics_glob: str = "**/normalized_metrics.json") -> list[dict[str, Any]]:
    rows = []
    for path in collect_metric_files(runs_root, metrics_glob):
        payload = load_json(path)
        row = normalize_metric_row(payload, path)
        if row.get("variant") in KEPT_VARIANTS:
            rows.append(row)
    rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("scenario")), int(r.get("seed") or -1), str(r.get("variant"))))
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
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str], note: str | None = None) -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    if not rows:
        lines.append("No rows found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir: Path, per_seed_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "main_experiment"

    write_csv(out_dir / f"{prefix}_per_seed.csv", per_seed_rows, PER_SEED_FIELDS)
    (out_dir / f"{prefix}_per_seed.json").write_text(json.dumps(jsonable(per_seed_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_table(
        out_dir / f"{prefix}_per_seed.md",
        "Main Experiment Gen Built-in AD Per-Seed Results",
        per_seed_rows,
        PER_SEED_FIELDS,
        note="Each row is one seed. This is the primary results table; do not replace it with mean/std or delta tables.",
    )


def main() -> None:
    args = parse_args()
    per_seed = collect_per_seed_rows(args.runs_root.resolve(), args.metrics_glob)
    write_outputs(args.out_dir.resolve(), per_seed)
    print(f"per-seed rows: {len(per_seed)}")
    print(f"saved summaries: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
