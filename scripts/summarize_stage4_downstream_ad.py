#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage4 downstream AD attempts, metrics, and explicit failure reports.")
    parser.add_argument("--stage4-dir", "--root", default="outputs/gcad_gss_stage4")
    parser.add_argument("--out-csv", default="outputs/gcad_gss_stage4/stage4_downstream_ad_summary.csv")
    parser.add_argument("--out-md", default="outputs/gcad_gss_stage4/stage4_downstream_ad_summary.md")
    parser.add_argument("--include-all-stage4b", action="store_true", help="Include older Stage4B dry-run directories as well as the Codex/GPT-5.5 downweight mainline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.stage4_dir), include_all_stage4b=args.include_all_stage4b)
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"saved {args.out_csv}")
    print(f"saved {args.out_md}")


def collect_rows(root: Path, include_all_stage4b: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    pattern = "*stage4b_ad*/downstream_ad_metrics.json" if include_all_stage4b else "*stage4b_ad_codex_gpt55_downweight*/downstream_ad_metrics.json"
    failure_pattern = "*stage4b_ad*/downstream_ad_failure_report.json" if include_all_stage4b else "*stage4b_ad_codex_gpt55_downweight*/downstream_ad_failure_report.json"
    for path in sorted(root.glob(pattern)):
        seen.add(path.parent)
        payload = read_json(path)
        rows.append(row_from_payload(path.parent, payload, metrics_path=path))
    for path in sorted(root.glob(failure_pattern)):
        if path.parent in seen:
            continue
        payload = read_json(path)
        rows.append(row_from_payload(path.parent, payload, failure_path=path))
    if not rows:
        rows.append({"method": "stage4_codex_gpt55_surrogate", "status": "missing", "note": "no downstream AD metrics or failure reports found"})
    return rows


def row_from_payload(out_dir: Path, payload: dict[str, Any], metrics_path: Path | None = None, failure_path: Path | None = None) -> dict[str, Any]:
    scenario = payload.get("scenario") or infer_scenario(out_dir.name)
    variant = payload.get("variant") or infer_variant(out_dir.name)
    return {
        "method": payload.get("method", "stage4_codex_gpt55_surrogate"),
        "baseline_family": payload.get("baseline_family", "stage3_codex_surrogate"),
        "api_llm": payload.get("api_llm", False),
        "surrogate_for_smartgen_llm": payload.get("surrogate_for_smartgen_llm", True),
        "scenario": scenario,
        "seed": payload.get("seed", 2024),
        "variant": variant,
        "status": payload.get("status", "metrics_available" if metrics_path else "not_completed"),
        "reason": payload.get("reason") or payload.get("missing_reason"),
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "f1": payload.get("f1"),
        "fpr": payload.get("fpr"),
        "accuracy": payload.get("accuracy"),
        "delta_f1_vs_original_prompt": payload.get("delta_f1_vs_original_prompt"),
        "delta_fpr_vs_original_prompt": payload.get("delta_fpr_vs_original_prompt"),
        "stage4a_dir": payload.get("stage4a_dir"),
        "generated_pkl": payload.get("generated_pkl"),
        "weighted_generated_pkl": payload.get("weighted_generated_pkl"),
        "command_attempted": payload.get("command_attempted"),
        "missing_files": "; ".join(payload.get("missing_files", [])) if isinstance(payload.get("missing_files"), list) else payload.get("missing_files"),
        "output_dir": str(out_dir),
        "metrics_path": str(metrics_path) if metrics_path else "",
        "failure_path": str(failure_path) if failure_path else "",
    }


def infer_scenario(name: str) -> str:
    if name.startswith("fr_st"):
        return "fr_st"
    if name.startswith("sp_st"):
        return "sp_st"
    return ""


def infer_variant(name: str) -> str:
    if "_tof_" in name or "causal_tof" in name:
        return "tof"
    if "_raw_" in name or "guarded_reweighted" in name:
        return "raw"
    return ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"payload": payload}
    except Exception as exc:
        return {"read_error": str(exc)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage4 Downstream AD Summary",
        "",
        "This table separates `stage4_codex_gpt55_surrogate` from the older `stage3_codex_surrogate` family. API LLM is `false` for the Stage4 surrogate path.",
        "",
        "| scenario | variant | status | f1 | fpr | reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('scenario','')} | {row.get('variant','')} | {row.get('status','')} | "
            f"{fmt(row.get('f1'))} | {fmt(row.get('fpr'))} | {row.get('reason','') or ''} |"
        )
    lines.extend(["", "## Claims", ""])
    if any(row.get("status") == "metrics_available" for row in rows):
        lines.append("- Real downstream AD metrics are available for at least one Stage4 attempt; inspect CSV for deltas.")
    else:
        lines.append("- Real downstream AD was not completed in this run; do not claim F1/FPR improvement from Stage4B.")
    lines.append("- Failure reports are explicit records, not placeholder positive results.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    main()
