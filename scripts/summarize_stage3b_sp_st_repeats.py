#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "outputs/gcad_gss"
STAGE3B_ROOT = OUT_ROOT / "sp_st_stage3b_ad"

METRICS = [
    "precision",
    "recall",
    "F1",
    "FPR",
    "FNR",
    "accuracy",
    "learned_threshold",
    "synthetic_size",
]


def to_float(value: Any) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def sd(values: list[float]) -> float | None:
    return stdev(values) if len(values) > 1 else 0.0 if len(values) == 1 else None


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.6f}"


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: row for row in payload.get("rows", [])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize SP-ST Stage 3B repeat robustness runs")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=TAG",
        help="Repeat mapping, e.g. seed2024=sp_st_codex_calibrated_seed2024. Can be repeated.",
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_ROOT / "sp_st_codex_calibrated_multiseed_stage3b_summary.csv")
    parser.add_argument("--out-json", type=Path, default=OUT_ROOT / "sp_st_codex_calibrated_multiseed_stage3b_summary.json")
    parser.add_argument("--out-md", type=Path, default=OUT_ROOT / "sp_st_codex_calibrated_multiseed_stage3b_summary.md")
    parser.add_argument("--stage3b-root", type=Path, default=STAGE3B_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.run or [
        "seed2024=sp_st_codex_calibrated_seed2024",
        "seed2025=sp_st_codex_calibrated_seed2025",
        "seed2026=sp_st_codex_calibrated_seed2026",
    ]
    per_repeat: list[dict[str, Any]] = []
    for item in runs:
        name, tag = item.split("=", 1)
        metrics_json = args.stage3b_root / tag / "metrics.json"
        if not metrics_json.exists():
            per_repeat.append({"repeat": name, "tag": tag, "status": "missing_metrics", "metrics_json": str(metrics_json)})
            continue
        groups = read_rows(metrics_json)
        row: dict[str, Any] = {"repeat": name, "tag": tag, "status": "ok", "metrics_json": str(metrics_json)}
        for metric in METRICS:
            original = to_float(groups.get("original_prompt", {}).get(metric))
            enhanced = to_float(groups.get("enhanced_prompt", {}).get(metric))
            row[f"original_{metric}"] = original
            row[f"enhanced_{metric}"] = enhanced
            row[f"delta_{metric}"] = enhanced - original if enhanced is not None and original is not None else None
        per_repeat.append(row)

    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        for prefix in ("original", "enhanced", "delta"):
            values = [row[f"{prefix}_{metric}"] for row in per_repeat if row.get("status") == "ok" and row.get(f"{prefix}_{metric}") is not None]
            aggregate[f"{prefix}_{metric}_mean"] = avg(values)
            aggregate[f"{prefix}_{metric}_std"] = sd(values)

    fieldnames = ["repeat", "tag", "status"]
    for metric in METRICS:
        fieldnames.extend([f"original_{metric}", f"enhanced_{metric}", f"delta_{metric}"])
    fieldnames.append("metrics_json")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_repeat:
            writer.writerow({key: row.get(key) for key in fieldnames})
    args.out_json.write_text(json.dumps({"runs": per_repeat, "aggregate": aggregate}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SP-ST Stage 3B Repeat Robustness",
        "",
        "Codex-calibrated generation setting. SmartGen TOF, target test data, attack data, and Transformer Autoencoder evaluation are unchanged within each run.",
        "",
        "## Delta Mean",
        "",
        "| metric | enhanced-original mean | std |",
        "| --- | ---: | ---: |",
    ]
    for metric in ("precision", "recall", "F1", "FPR", "FNR", "accuracy", "learned_threshold"):
        lines.append(
            f"| {metric} | {fmt(aggregate.get(f'delta_{metric}_mean'))} | "
            f"{fmt(aggregate.get(f'delta_{metric}_std'))} |"
        )
    lines.extend(["", "## Per Seed", ""])
    lines.append("| repeat | F1 Δ | FPR Δ | accuracy Δ | precision Δ | recall Δ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in per_repeat:
        lines.append(
            f"| {row.get('repeat')} | {fmt(row.get('delta_F1'))} | {fmt(row.get('delta_FPR'))} | "
            f"{fmt(row.get('delta_accuracy'))} | {fmt(row.get('delta_precision'))} | {fmt(row.get('delta_recall'))} |"
        )
    lines.extend(["", "## Run Paths", ""])
    for row in per_repeat:
        lines.append(f"- {row.get('repeat')}: `{row.get('metrics_json')}`")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_csv: {args.out_csv}")
    print(f"summary_json: {args.out_json}")
    print(f"summary_md: {args.out_md}")


if __name__ == "__main__":
    main()
