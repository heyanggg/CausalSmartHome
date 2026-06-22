#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 4 Causal-TOF weighting outputs.")
    parser.add_argument("--stage4-dir", "--root", default="outputs/gcad_gss_stage4")
    parser.add_argument("--out-csv", default="outputs/gcad_gss_stage4/stage4_causal_tof_summary.csv")
    parser.add_argument("--out-md", default="outputs/gcad_gss_stage4/stage4_causal_tof_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for scores_path in sorted(Path(args.stage4_dir).glob("**/causal_tof_scores.json")):
        scores = read_json(scores_path)
        if not isinstance(scores, list):
            continue
        row = {
            "method": "new causal-TOF weighted",
            "status": "ok",
            "source_path": str(scores_path),
            "num_sequences": len(scores),
            "avg_causal_coverage": mean([item.get("causal_coverage") for item in scores]),
            "avg_causal_violation": mean([item.get("causal_violation") for item in scores]),
            "avg_distribution_penalty": mean([item.get("distribution_penalty") for item in scores]),
            "avg_final_score": mean([item.get("final_score") for item in scores]),
            "avg_sample_weight": mean([item.get("sample_weight") for item in scores]),
            "min_sample_weight": min_or_none([item.get("sample_weight") for item in scores]),
            "max_sample_weight": max_or_none([item.get("sample_weight") for item in scores]),
        }
        weights_path = scores_path.parent / "generated.weights.json"
        row["weights_file"] = str(weights_path) if weights_path.exists() else "missing"
        rows.append(row)
    if not rows:
        rows.append({"method": "new causal-TOF weighted", "status": "missing", "note": "no causal_tof_scores.json found"})
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"saved {args.out_csv}")
    print(f"saved {args.out_md}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 4 Causal-TOF Summary", ""]
    lines.append("| status | num_sequences | avg_causal_coverage | avg_causal_violation | avg_sample_weight | weights_file |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| {row.get('status','')} | {row.get('num_sequences','')} | {fmt(row.get('avg_causal_coverage'))} | "
            f"{fmt(row.get('avg_causal_violation'))} | {fmt(row.get('avg_sample_weight'))} | {row.get('weights_file','')} |"
        )
    lines.extend(["", "## 自动结论", ""])
    if any(row.get("status") == "ok" for row in rows):
        lines.append("- 已找到 causal-TOF soft weighting 结果；可与 hard filter 的保留率、FPR/F1 做进一步比较。")
    else:
        lines.append("- 未找到 causal_tof_scores.json，因此不能判断 soft weighting 是否优于 hard deletion。")
    lines.append("- 本 summary 不强行声称所有数据集提升；提升需结合 Stage4B downstream_ad_metrics.json。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def min_or_none(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return min(nums) if nums else None


def max_or_none(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return max(nums) if nums else None


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    main()
