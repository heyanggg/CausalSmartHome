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

METRICS = [
    "causal_coverage",
    "violation_rate",
    "low_evidence_rate",
    "action_js_to_target",
    "device_js_to_target",
    "transition_js_to_target",
    "tof_kept_rate",
    "generated_count",
]


def read_group(path: Path, group: str) -> dict[str, Any]:
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] == group:
                return row
    raise KeyError(f"group {group} not found in {path}")


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


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.6f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize SP-ST guarded-edge Stage 3A ablation")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="SEED=BASELINE_TAG=GUARDED_TAG",
        help="Example: seed2024=sp_st_codex_calibrated_seed2024=sp_st_guarded_edge_seed2024",
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_ROOT / "sp_st_guarded_edge_ablation_stage3a_summary.csv")
    parser.add_argument("--out-json", type=Path, default=OUT_ROOT / "sp_st_guarded_edge_ablation_stage3a_summary.json")
    parser.add_argument("--out-md", type=Path, default=OUT_ROOT / "sp_st_guarded_edge_ablation_stage3a_summary.md")
    parser.add_argument("--title", default="SP-ST Guarded-Edge Stage 3A Ablation")
    parser.add_argument(
        "--description",
        default=(
            "Guard mode: `target-overrepresented`; each seed uses its own original prompt TOF output "
            "as reference and SP spring target as the target distribution."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.run or [
        "seed2024=sp_st_codex_calibrated_seed2024=sp_st_guarded_edge_seed2024",
        "seed2025=sp_st_codex_calibrated_seed2025=sp_st_guarded_edge_seed2025",
        "seed2026=sp_st_codex_calibrated_seed2026=sp_st_guarded_edge_seed2026",
    ]
    rows: list[dict[str, Any]] = []
    for item in runs:
        repeat, baseline_tag, guarded_tag = item.split("=", 2)
        baseline_summary = OUT_ROOT / baseline_tag / "sp_st_stage3a_summary.csv"
        guarded_summary = OUT_ROOT / guarded_tag / "sp_st_stage3a_summary.csv"
        original = read_group(baseline_summary, "original")
        unguarded = read_group(baseline_summary, "enhanced")
        guarded = read_group(guarded_summary, "enhanced")
        row: dict[str, Any] = {
            "repeat": repeat,
            "baseline_tag": baseline_tag,
            "guarded_tag": guarded_tag,
            "baseline_summary": str(baseline_summary),
            "guarded_summary": str(guarded_summary),
        }
        for metric in METRICS:
            o = to_float(original.get(metric))
            u = to_float(unguarded.get(metric))
            g = to_float(guarded.get(metric))
            row[f"original_{metric}"] = o
            row[f"unguarded_{metric}"] = u
            row[f"guarded_{metric}"] = g
            row[f"unguarded_delta_{metric}"] = u - o if u is not None and o is not None else None
            row[f"guarded_delta_{metric}"] = g - o if g is not None and o is not None else None
            row[f"guarded_minus_unguarded_{metric}"] = g - u if g is not None and u is not None else None
        rows.append(row)

    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        for prefix in ("unguarded_delta", "guarded_delta", "guarded_minus_unguarded"):
            values = [row[f"{prefix}_{metric}"] for row in rows if row.get(f"{prefix}_{metric}") is not None]
            aggregate[f"{prefix}_{metric}_mean"] = avg(values)
            aggregate[f"{prefix}_{metric}_std"] = sd(values)

    fieldnames = ["repeat", "baseline_tag", "guarded_tag"]
    for metric in METRICS:
        fieldnames.extend(
            [
                f"original_{metric}",
                f"unguarded_{metric}",
                f"guarded_{metric}",
                f"unguarded_delta_{metric}",
                f"guarded_delta_{metric}",
                f"guarded_minus_unguarded_{metric}",
            ]
        )
    fieldnames.extend(["baseline_summary", "guarded_summary"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])
    args.out_json.write_text(json.dumps({"runs": rows, "aggregate": aggregate}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {args.title}",
        "",
        args.description,
        "",
        "## Delta Mean Versus Original",
        "",
        "| metric | unguarded mean | unguarded std | guarded mean | guarded std | guarded - unguarded mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in (
        "low_evidence_rate",
        "action_js_to_target",
        "device_js_to_target",
        "causal_coverage",
        "violation_rate",
        "transition_js_to_target",
        "tof_kept_rate",
    ):
        lines.append(
            f"| {metric} | {fmt(aggregate.get(f'unguarded_delta_{metric}_mean'))} | "
            f"{fmt(aggregate.get(f'unguarded_delta_{metric}_std'))} | "
            f"{fmt(aggregate.get(f'guarded_delta_{metric}_mean'))} | "
            f"{fmt(aggregate.get(f'guarded_delta_{metric}_std'))} | "
            f"{fmt(aggregate.get(f'guarded_minus_unguarded_{metric}_mean'))} |"
        )
    lines.extend(["", "## Per Seed", ""])
    lines.append("| seed | low evidence guarded Δ | action JS guarded Δ | device JS guarded Δ | coverage guarded Δ | TOF kept guarded Δ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['repeat']} | {fmt(row.get('guarded_delta_low_evidence_rate'))} | "
            f"{fmt(row.get('guarded_delta_action_js_to_target'))} | "
            f"{fmt(row.get('guarded_delta_device_js_to_target'))} | "
            f"{fmt(row.get('guarded_delta_causal_coverage'))} | "
            f"{fmt(row.get('guarded_delta_tof_kept_rate'))} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append("- Guarding overrepresented target endpoints mainly tests whether suppressing `* -> Television` edges fixes SP-ST distribution drift.")
    lines.append("- A negative `guarded - unguarded` value for action/device JS means the guard moved enhanced outputs closer to SP spring target distribution than the unguarded enhanced prompt.")
    lines.append("- Compare low evidence and causal coverage deltas to check how much causal/evidence benefit is lost by the guard.")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_csv: {args.out_csv}")
    print(f"summary_json: {args.out_json}")
    print(f"summary_md: {args.out_md}")


if __name__ == "__main__":
    main()
