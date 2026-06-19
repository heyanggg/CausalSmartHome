#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["group"]: row for row in csv.DictReader(f)}


def as_float(value: Any) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize FR-ST Stage 3A sparse-threshold ablation runs")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="THRESHOLD=TAG",
        help="Ablation run mapping, e.g. 0.001=fr_st_thr_0p001. Can be repeated.",
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_ROOT / "fr_st_stage3a_threshold_ablation_summary.csv")
    parser.add_argument("--out-md", type=Path, default=OUT_ROOT / "fr_st_stage3a_threshold_ablation_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.run or [
        "0.001=fr_st_thr_0p001",
        "0.0005=fr_st_thr_0p0005",
        "0.0002=fr_st_thr_0p0002",
    ]
    rows: list[dict[str, Any]] = []
    for item in runs:
        threshold, tag = item.split("=", 1)
        run_root = OUT_ROOT / tag
        summary_csv = run_root / "fr_st_stage3a_summary.csv"
        prompt_check_path = run_root / "fr_st_prompt_check/prompt_check.json"
        if not summary_csv.exists():
            rows.append({"threshold": threshold, "tag": tag, "status": "missing_summary", "error": str(summary_csv)})
            continue
        groups = read_csv(summary_csv)
        prompt_check = json.loads(prompt_check_path.read_text(encoding="utf-8")) if prompt_check_path.exists() else {}
        row: dict[str, Any] = {
            "threshold": threshold,
            "tag": tag,
            "status": "ok" if {"original", "enhanced"}.issubset(groups) else "missing_group",
            "top20_edges": prompt_check.get("top20_edges"),
            "top10_edges": prompt_check.get("top10_edges"),
            "prompt_est_tokens": prompt_check.get("top20_prompt_est_tokens"),
            "soft_hints": prompt_check.get("causal_hints_are_soft"),
            "gss_retained": prompt_check.get("original_gss_hints_retained"),
            "summary_csv": str(summary_csv),
        }
        original = groups.get("original", {})
        enhanced = groups.get("enhanced", {})
        for metric in METRICS:
            original_value = as_float(original.get(metric))
            enhanced_value = as_float(enhanced.get(metric))
            row[f"original_{metric}"] = original_value
            row[f"enhanced_{metric}"] = enhanced_value
            row[f"delta_{metric}"] = (
                enhanced_value - original_value if original_value is not None and enhanced_value is not None else None
            )
        rows.append(row)

    fieldnames = [
        "threshold",
        "tag",
        "status",
        "top20_edges",
        "top10_edges",
        "prompt_est_tokens",
        "soft_hints",
        "gss_retained",
    ]
    for metric in METRICS:
        fieldnames.extend([f"original_{metric}", f"enhanced_{metric}", f"delta_{metric}"])
    fieldnames.append("summary_csv")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# FR-ST Stage 3A Sparse Threshold Ablation",
        "",
        "This compares generation quality only. No downstream AD claim is made.",
        "",
        "| threshold | status | edges | tokens | coverage delta | low evidence delta | transition JS delta | TOF kept delta | action JS delta | device JS delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('threshold')} | {row.get('status')} | {row.get('top20_edges', 'n/a')} | "
            f"{row.get('prompt_est_tokens', 'n/a')} | {fmt(row.get('delta_causal_coverage'))} | "
            f"{fmt(row.get('delta_low_evidence_rate'))} | {fmt(row.get('delta_transition_js_to_target'))} | "
            f"{fmt(row.get('delta_tof_kept_rate'))} | {fmt(row.get('delta_action_js_to_target'))} | "
            f"{fmt(row.get('delta_device_js_to_target'))} |"
        )
    lines.extend(["", "## Run Paths", ""])
    for row in rows:
        lines.append(f"- {row.get('threshold')}: `{row.get('summary_csv')}`")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_csv: {args.out_csv}")
    print(f"summary_md: {args.out_md}")


if __name__ == "__main__":
    main()
