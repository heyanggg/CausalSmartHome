#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ["precision", "recall", "f1", "fpr", "fnr", "accuracy", "generated_size"]
VARIANT_ORDER = [
    "stage3_prompt_only_baseline",
    "stage4_downweight_multiplicative_raw",
    "stage4_downweight_multiplicative_causal_tof_resampled",
]
COMPARISONS = [
    ("stage4_downweight_multiplicative_raw", "stage3_prompt_only_baseline"),
    ("stage4_downweight_multiplicative_causal_tof_resampled", "stage3_prompt_only_baseline"),
    ("stage4_downweight_multiplicative_causal_tof_resampled", "stage4_downweight_multiplicative_raw"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage4 SmartGen built-in downstream AD runs.")
    parser.add_argument("--root", type=Path, default=Path("outputs/gcad_gss_stage4/gen_builtin_ad"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/gcad_gss_stage4/gen_builtin_ad_summary.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/gcad_gss_stage4/gen_builtin_ad_summary.md"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/gcad_gss_stage4/gen_builtin_ad_summary.json"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = [load_json(path) for path in sorted(root.glob("*_st/*/downstream_ad_metrics.json"))]
    failures = [load_json(path) for path in sorted(root.glob("*_st/*/failure_report.json"))]
    return successes, failures


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return stdev(values)


def summarize(successes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        key = (str(row.get("dataset")), str(row.get("scenario")), str(row.get("variant")))
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, scenario, variant), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], VARIANT_ORDER.index(item[0][2]) if item[0][2] in VARIANT_ORDER else 99)):
        out: dict[str, Any] = {
            "dataset": dataset,
            "scenario": scenario,
            "variant": variant,
            "num_successful_seeds": len(rows),
            "seeds": sorted(int(row["seed"]) for row in rows if row.get("seed") is not None),
            "downstream_pipeline": rows[0].get("downstream_pipeline"),
            "generator": rows[0].get("generator"),
            "api_llm": rows[0].get("api_llm"),
        }
        for metric in METRICS:
            values = [value for value in (numeric(row.get(metric)) for row in rows) if value is not None]
            name = "generated_size" if metric == "generated_size" else metric
            out[f"mean_{name}"] = mean(values) if values else None
            out[f"std_{name}"] = std(values)
        summaries.append(out)
    return summaries


def compare(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["scenario"], row["variant"]): row for row in summaries}
    rows: list[dict[str, Any]] = []
    datasets = sorted({(row["dataset"], row["scenario"]) for row in summaries})
    for dataset, scenario in datasets:
        for lhs, rhs in COMPARISONS:
            left = by_key.get((dataset, scenario, lhs))
            right = by_key.get((dataset, scenario, rhs))
            row: dict[str, Any] = {
                "dataset": dataset,
                "scenario": scenario,
                "comparison": f"{lhs} vs {rhs}",
                "status": "available" if left and right else "missing_baseline_or_variant",
                "left_successful_seeds": left.get("num_successful_seeds") if left else 0,
                "right_successful_seeds": right.get("num_successful_seeds") if right else 0,
            }
            for metric in ["precision", "recall", "f1", "fpr", "fnr", "accuracy"]:
                left_value = numeric(left.get(f"mean_{metric}")) if left else None
                right_value = numeric(right.get(f"mean_{metric}")) if right else None
                row[f"delta_mean_{metric}"] = left_value - right_value if left_value is not None and right_value is not None else None
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "scenario",
        "variant",
        "num_successful_seeds",
        "mean_precision",
        "std_precision",
        "mean_recall",
        "std_recall",
        "mean_f1",
        "std_f1",
        "mean_fpr",
        "std_fpr",
        "mean_fnr",
        "std_fnr",
        "mean_accuracy",
        "std_accuracy",
        "mean_generated_size",
        "std_generated_size",
        "downstream_pipeline",
        "generator",
        "api_llm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_md(path: Path, summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage4 SmartGen Built-in Downstream AD Summary",
        "",
        "SmartGuard AD runs are excluded from this table.",
        "",
        "## Variant Means",
        "",
        "| dataset | scenario | variant | seeds | precision | recall | f1 | fpr | fnr | accuracy | generated |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['scenario']} | {row['variant']} | {row['num_successful_seeds']} | "
            f"{fmt(row.get('mean_precision'))} | {fmt(row.get('mean_recall'))} | {fmt(row.get('mean_f1'))} | "
            f"{fmt(row.get('mean_fpr'))} | {fmt(row.get('mean_fnr'))} | {fmt(row.get('mean_accuracy'))} | "
            f"{fmt(row.get('mean_generated_size'))} |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| dataset | scenario | comparison | status | delta f1 | delta fpr | delta accuracy |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['dataset']} | {row['scenario']} | {row['comparison']} | {row['status']} | "
            f"{fmt(row.get('delta_mean_f1'))} | {fmt(row.get('delta_mean_fpr'))} | {fmt(row.get('delta_mean_accuracy'))} |"
        )
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- None recorded.")
    else:
        for failure in failures:
            lines.append(f"- {failure.get('dataset')}_{failure.get('scenario')} {failure.get('variant')} seed {failure.get('seed')}: {failure.get('reason')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    successes, failures = collect(args.root)
    summaries = summarize(successes)
    comparisons = compare(summaries)
    write_csv(args.out_csv, summaries)
    write_md(args.out_md, summaries, comparisons, failures)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"summaries": summaries, "comparisons": comparisons, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"success metrics: {len(successes)}")
    print(f"failure reports: {len(failures)}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
