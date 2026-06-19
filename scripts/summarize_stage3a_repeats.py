#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "outputs/gcad_gss"

METRICS = [
    "causal_coverage",
    "violation_rate",
    "low_evidence_rate",
    "action_js_to_target",
    "device_js_to_target",
    "transition_js_to_target",
    "tof_kept_rate",
    "generated_count",
    "raw_generated_count",
    "average_sequence_length",
]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["group"]: row for row in csv.DictReader(f)}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize FR-ST Stage 3A repeat robustness runs")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=TAG",
        help="Repeat mapping, e.g. r1=fr_st_thr_0p001_repeat1. Can be repeated.",
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_ROOT / "fr_st_stage3a_repeat_summary.csv")
    parser.add_argument("--out-md", type=Path, default=OUT_ROOT / "fr_st_stage3a_repeat_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.run or [
        "r1=fr_st_thr_0p001_repeat1",
        "r2=fr_st_thr_0p001_repeat2",
        "r3=fr_st_thr_0p001_repeat3",
    ]
    per_repeat: list[dict[str, Any]] = []
    for item in runs:
        name, tag = item.split("=", 1)
        summary_csv = OUT_ROOT / tag / "fr_st_stage3a_summary.csv"
        if not summary_csv.exists():
            per_repeat.append({"repeat": name, "tag": tag, "status": "missing_summary", "summary_csv": str(summary_csv)})
            continue
        groups = read_rows(summary_csv)
        row: dict[str, Any] = {"repeat": name, "tag": tag, "status": "ok", "summary_csv": str(summary_csv)}
        for metric in METRICS:
            original = to_float(groups.get("original", {}).get(metric))
            enhanced = to_float(groups.get("enhanced", {}).get(metric))
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
    fieldnames.append("summary_csv")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_repeat:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# FR-ST Stage 3A Repeat Robustness",
        "",
        "Fixed sparse-threshold=0.001. Generation quality only; no downstream AD claim is made.",
        "",
        "## Delta Mean",
        "",
        "| metric | enhanced-original mean | std |",
        "| --- | ---: | ---: |",
    ]
    for metric in (
        "causal_coverage",
        "low_evidence_rate",
        "violation_rate",
        "action_js_to_target",
        "device_js_to_target",
        "transition_js_to_target",
        "tof_kept_rate",
        "generated_count",
        "average_sequence_length",
    ):
        lines.append(
            f"| {metric} | {fmt(aggregate.get(f'delta_{metric}_mean'))} | "
            f"{fmt(aggregate.get(f'delta_{metric}_std'))} |"
        )
    lines.extend(["", "## Per Repeat", ""])
    lines.append("| repeat | coverage Δ | low evidence Δ | violation Δ | action JS Δ | device JS Δ | transition JS Δ | TOF kept Δ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_repeat:
        lines.append(
            f"| {row.get('repeat')} | {fmt(row.get('delta_causal_coverage'))} | "
            f"{fmt(row.get('delta_low_evidence_rate'))} | {fmt(row.get('delta_violation_rate'))} | "
            f"{fmt(row.get('delta_action_js_to_target'))} | {fmt(row.get('delta_device_js_to_target'))} | "
            f"{fmt(row.get('delta_transition_js_to_target'))} | {fmt(row.get('delta_tof_kept_rate'))} |"
        )
    lines.extend(["", "## Run Paths", ""])
    for row in per_repeat:
        lines.append(f"- {row.get('repeat')}: `{row.get('summary_csv')}`")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_csv: {args.out_csv}")
    print(f"summary_md: {args.out_md}")


if __name__ == "__main__":
    main()
