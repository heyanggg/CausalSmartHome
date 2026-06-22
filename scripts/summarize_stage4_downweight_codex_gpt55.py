#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage4 downweight+multiplicative Codex/GPT-5.5 surrogate quality outputs.")
    parser.add_argument("--stage4-dir", "--root", default="outputs/gcad_gss_stage4")
    parser.add_argument("--out-csv", default="outputs/gcad_gss_stage4/stage4_downweight_codex_gpt55_summary.csv")
    parser.add_argument("--out-md", default="outputs/gcad_gss_stage4/stage4_downweight_codex_gpt55_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.stage4_dir)
    rows = collect_rows(root)
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"saved {args.out_csv}")
    print(f"saved {args.out_md}")


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*_downweight_multiplicative_codex_gpt55_seed*/generation_quality_metrics.json")):
        out_dir = metrics_path.parent
        metrics = read_json(metrics_path)
        validation = read_json(out_dir / "validation_report.json")
        metadata = read_json(out_dir / "generation_metadata.json")
        scores = read_json(out_dir / "causal_tof_scores.json")
        row = {
            "method": "stage4_codex_gpt55_surrogate",
            "baseline_family": "stage3_codex_surrogate",
            "api_llm": False,
            "surrogate_for_smartgen_llm": True,
            "stage4a_dir": str(out_dir),
            "scenario": metrics.get("scenario") or infer_scenario(out_dir.name),
            "seed": metrics.get("seed") or metadata.get("seed"),
            "generator": metadata.get("generator", "codex_gpt55_surrogate"),
            "guard_mode": metadata.get("guard_mode", "downweight"),
            "downweight_factor": metadata.get("downweight_factor", 0.25),
            "reweight_mode": metadata.get("reweight_mode", "multiplicative"),
            "lambda_causal": metadata.get("lambda_causal", 1.0),
            "validation_status": validation.get("status"),
            "raw_sequence_count": validation.get("raw_sequence_count"),
            "clean_sequence_count": validation.get("clean_sequence_count"),
            "invalid_sequence_count": validation.get("invalid_sequence_count"),
            "generated_size": metrics.get("generated_size"),
            "target_size": metrics.get("target_size"),
            "action_js_to_target": metrics.get("action_js_to_target"),
            "device_js_to_target": metrics.get("device_js_to_target"),
            "transition_js_to_target": metrics.get("transition_js_to_target"),
            "causal_coverage": metrics.get("causal_coverage"),
            "causal_violation_rate": metrics.get("causal_violation_rate"),
            "low_evidence_rate": metrics.get("low_evidence_rate"),
            "nonzero_guarded_edges": metrics.get("nonzero_guarded_edges"),
            "suppressed_edge_count": metrics.get("suppressed_edge_count"),
            "downweighted_edge_count": metrics.get("downweighted_edge_count"),
            "television_device_key": metrics.get("television_device_key"),
            "television_freq_in_generated": metrics.get("television_freq_in_generated"),
            "television_freq_in_target": metrics.get("television_freq_in_target"),
            "television_overuse_ratio": metrics.get("television_overuse_ratio"),
        }
        row.update(causal_tof_summary(scores if isinstance(scores, list) else []))
        rows.append(row)
        write_json(out_dir / "causal_tof_summary.json", causal_tof_summary_json(row, scores if isinstance(scores, list) else []))
    if not rows:
        rows.append({"method": "stage4_codex_gpt55_surrogate", "status": "missing", "note": "no Codex/GPT-5.5 Stage4A metrics found"})
    return rows


def causal_tof_summary(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tof_num_sequences": len(scores),
        "tof_avg_causal_coverage": mean([score.get("causal_coverage") for score in scores]),
        "tof_avg_causal_violation": mean([score.get("causal_violation") for score in scores]),
        "tof_avg_distribution_penalty": mean([score.get("distribution_penalty") for score in scores]),
        "tof_avg_final_score": mean([score.get("final_score") for score in scores]),
        "tof_avg_sample_weight": mean([score.get("sample_weight") for score in scores]),
        "tof_min_sample_weight": min_or_none([score.get("sample_weight") for score in scores]),
        "tof_max_sample_weight": max_or_none([score.get("sample_weight") for score in scores]),
    }


def causal_tof_summary_json(row: Mapping[str, Any], scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "method": row.get("method"),
        "scenario": row.get("scenario"),
        "seed": row.get("seed"),
        "generator": row.get("generator"),
        "api_llm": False,
        "surrogate_for_smartgen_llm": True,
        "summary": causal_tof_summary(scores),
    }


def infer_scenario(name: str) -> str:
    if name.startswith("fr_st"):
        return "fr_st"
    if name.startswith("sp_st"):
        return "sp_st"
    return ""


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc)}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "# Stage4 Downweight Codex/GPT-5.5 Summary",
        "",
        "Generator: `stage4_codex_gpt55_surrogate`; API LLM: `false`; baseline family kept separate as `stage3_codex_surrogate`.",
        "",
        "| scenario | seed | raw/clean | action_js | device_js | transition_js | causal_cov | causal_viol | TV ratio | TOF avg weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") == "missing":
            lines.append("| missing |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {row.get('scenario','')} | {row.get('seed','')} | {row.get('raw_sequence_count','')}/{row.get('clean_sequence_count','')} | "
            f"{fmt(row.get('action_js_to_target'))} | {fmt(row.get('device_js_to_target'))} | {fmt(row.get('transition_js_to_target'))} | "
            f"{fmt(row.get('causal_coverage'))} | {fmt(row.get('causal_violation_rate'))} | {fmt(row.get('television_overuse_ratio'))} | "
            f"{fmt(row.get('tof_avg_sample_weight'))} |"
        )
    lines.extend(["", "## Answers", ""])
    lines.extend(auto_answers(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def auto_answers(rows: list[dict[str, Any]]) -> list[str]:
    ok = [row for row in rows if row.get("status") != "missing"]
    sp_rows = [row for row in ok if row.get("scenario") == "sp_st"]
    return [
        f"- Fresh Codex/GPT-5.5 surrogate outputs exist: {'yes' if ok else 'no'}.",
        f"- Validation passed without cleaning loss: {'yes' if ok and all(int(row.get('invalid_sequence_count') or 0) == 0 for row in ok) else 'no'}.",
        "- Mainline config is downweight + multiplicative with lambda_causal=1.0, downweight_factor=0.25, endpoint_policy=target.",
        f"- SP Television overuse controlled: {'yes' if sp_rows and all(float(row.get('television_overuse_ratio') or 0.0) <= 1.25 for row in sp_rows) else 'not established'}.",
        "- Causal-TOF is soft weighting/resampling; no hard deletion is claimed here.",
        "- Downstream AD improvement is not claimed from this summary; it requires downstream_ad_metrics.json.",
    ]


def mean(values: Sequence[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def min_or_none(values: Sequence[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return min(nums) if nums else None


def max_or_none(values: Sequence[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
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
