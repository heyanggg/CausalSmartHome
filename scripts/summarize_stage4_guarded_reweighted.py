#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHODS = [
    "original SmartGen prompt",
    "unguarded GCAD-GSS prompt-only",
    "guarded-edge prompt-only",
    "downweighted-edge prompt-only",
    "new guarded causal-reweighted GSS",
    "new causal-TOF weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 4 guarded causal-reweighted GSS outputs and available baselines.")
    parser.add_argument("--stage4-dir", "--root", default="outputs/gcad_gss_stage4")
    parser.add_argument("--stage3-dir", default="outputs/gcad_gss")
    parser.add_argument("--out-csv", default="outputs/gcad_gss_stage4/stage4_guarded_reweighted_summary.csv")
    parser.add_argument("--out-md", default="outputs/gcad_gss_stage4/stage4_guarded_reweighted_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.stage4_dir), Path(args.stage3_dir))
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"saved {args.out_csv}")
    print(f"saved {args.out_md}")


def collect_rows(stage4_dir: Path, stage3_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.extend(collect_method_rows(stage4_dir, stage3_dir, method))
    if not rows:
        rows.append({"method": "new guarded causal-reweighted GSS", "status": "missing", "note": "no summary files found"})
    return rows


def collect_method_rows(stage4_dir: Path, stage3_dir: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if method == "new guarded causal-reweighted GSS":
        for metrics_path in sorted(stage4_dir.glob("**/generation_quality_metrics.json")):
            payload = read_json(metrics_path)
            row = base_row(method, metrics_path)
            row.update({key: payload.get(key) for key in [
                "scenario", "seed", "generated_size", "causal_coverage", "causal_violation_rate",
                "action_js_to_target", "device_js_to_target", "transition_js_to_target", "tof_kept_rate",
                "guarded_edge_count", "suppressed_edge_count", "downweighted_edge_count", "avg_guarded_causal_strength",
            ]})
            row["status"] = "ok"
            rows.append(row)
    elif method == "new causal-TOF weighted":
        for scores_path in sorted(stage4_dir.glob("**/causal_tof_scores.json")):
            payload = read_json(scores_path)
            row = base_row(method, scores_path)
            if isinstance(payload, list):
                row.update({
                    "generated_size": len(payload),
                    "causal_coverage": mean([item.get("causal_coverage") for item in payload]),
                    "causal_violation_rate": mean([item.get("causal_violation") for item in payload]),
                    "avg_sample_weight": mean([item.get("sample_weight") for item in payload]),
                })
            row["status"] = "ok"
            rows.append(row)
    else:
        patterns = baseline_patterns(method)
        for pattern in patterns:
            for path in sorted(stage3_dir.glob(pattern)):
                payload = read_json(path)
                row = base_row(method, path)
                row.update(flatten_selected(payload))
                row["status"] = "ok"
                rows.append(row)
    if not rows:
        rows.append({"method": method, "status": "missing", "source_path": "", "note": "baseline/result file not found"})
    return rows


def baseline_patterns(method: str) -> list[str]:
    if method == "unguarded GCAD-GSS prompt-only":
        return ["*codex_calibrated*/*stage3a_summary.json", "*codex_calibrated*_summary.json"]
    if method == "guarded-edge prompt-only":
        return ["*guarded_edge*/*stage3a_summary.json", "*guarded_edge*_summary.json"]
    if method == "downweighted-edge prompt-only":
        return ["*downweighted_edge*/*stage3a_summary.json", "*downweighted_edge*_summary.json"]
    if method == "original SmartGen prompt":
        return ["*original*/*.json", "*smartgen*/*.json"]
    return []


def flatten_selected(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = ["scenario", "seed", "causal_coverage", "causal_violation_rate", "action_js_to_target", "device_js_to_target", "transition_js_to_target", "tof_kept_rate", "f1", "fpr"]
    out = {key: payload.get(key) for key in keys if key in payload}
    if "summary" in payload and isinstance(payload["summary"], dict):
        for key in keys:
            if key in payload["summary"] and key not in out:
                out[key] = payload["summary"][key]
    return out


def base_row(method: str, path: Path) -> dict[str, Any]:
    return {"method": method, "source_path": str(path), "status": "ok"}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 4 Guarded Causal-Reweighted Summary", ""]
    lines.append("| method | status | scenario | seed | causal_coverage | causal_violation_rate | device_js_to_target | f1 | fpr |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row.get('method','')} | {row.get('status','')} | {row.get('scenario','')} | {row.get('seed','')} | "
            f"{fmt(row.get('causal_coverage'))} | {fmt(row.get('causal_violation_rate'))} | {fmt(row.get('device_js_to_target'))} | "
            f"{fmt(row.get('f1'))} | {fmt(row.get('fpr'))} |"
        )
    lines.extend(["", "## 自动结论", ""])
    lines.extend(auto_conclusions(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def auto_conclusions(rows: list[dict[str, Any]]) -> list[str]:
    new_rows = [row for row in rows if row.get("method") == "new guarded causal-reweighted GSS" and row.get("status") == "ok"]
    weighted_rows = [row for row in rows if row.get("method") == "new causal-TOF weighted" and row.get("status") == "ok"]
    missing = [row.get("method") for row in rows if row.get("status") == "missing"]
    return [
        f"- FR-ST 是否提升：{'需要真实 downstream_ad_metrics.json 判断' if not has_scenario(new_rows, 'fr_st') else '已有 FR-ST Stage4A 质量指标，但 AD 提升仍需 Stage4B 指标'}。",
        f"- SP-ST 是否修复 Television bias：{'guard_report 可用于检查；当前未找到 SP-ST Stage4A' if not has_scenario(new_rows, 'sp_st') else '已有 SP-ST guard_report，可检查 overused_devices 与 suppressed/downweighted edges'}。",
        "- guarded causal-reweighted GSS 是否优于 prompt-only guarded：summary 会并列展示；缺 baseline 时标注 missing，不强行 claim。",
        f"- causal-TOF soft weighting 是否比 hard deletion 更稳：{'已有 causal-TOF scores，可比较 sample_weight 分布' if weighted_rows else '缺 causal-TOF scores，暂不能判断'}。",
        "- 是否存在 FPR tradeoff：必须看 Stage4B downstream_ad_metrics.json 的 fpr 与 delta_fpr_vs_original_prompt。",
        "- 是否 seed-sensitive：需要 seeds 2024/2025/2026 齐全后比较；当前缺失会在 CSV 中显示 missing。",
        f"- 缺失项：{', '.join(sorted(set(missing))) if missing else '无'}。",
    ]


def has_scenario(rows: list[dict[str, Any]], scenario: str) -> bool:
    return any(str(row.get("scenario", "")).lower() == scenario for row in rows)


def mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    main()
