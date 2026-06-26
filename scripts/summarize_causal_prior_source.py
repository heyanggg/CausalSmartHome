#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_matrix import SEEDS, experiment_grid, existing_stage_dir

OUT_ROOT = REPO_ROOT / "outputs" / "main_experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize causal relation prior sources for the full matrix.")
    parser.add_argument("--matrix", default="all", choices=["all"])
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "summary")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(source: str) -> str:
    normalized = source.lower()
    if "gradient" in normalized or "gcad" in normalized or "causalminer" in normalized:
        return "GradientCausalMiner / GCAD-style gradient prior"
    if "compact_fallback" in normalized or "transition_count" in normalized:
        return "compact_fallback_transition_count"
    if "prior_json" in normalized or "existing_prior_json" in normalized:
        return "prior_json"
    return "other"


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for item in experiment_grid():
        for seed in SEEDS:
            causal_gss_dir = existing_stage_dir(OUT_ROOT, "causal_gss", item.dataset, item.scenario, seed)
            prior_path = causal_gss_dir / "resolved_causal_relation_prior.json"
            config_path = causal_gss_dir / "config.json"
            status = "MISSING"
            source = ""
            adapter_mode = ""
            if prior_path.exists():
                payload = load_json(prior_path)
                source = str(payload.get("causal_relation_source") or payload.get("method") or "")
                status = "FOUND"
            if config_path.exists():
                config = load_json(config_path)
                adapter_mode = str(config.get("args", {}).get("adapter_mode") or "")
                source = source or str(config.get("causal_relation_source") or "")
            rows.append(
                {
                    "dataset": item.dataset,
                    "scenario": item.scenario,
                    "seed": seed,
                    "status": status,
                    "prior_source": source or "unknown",
                    "source_class": classify(source or adapter_mode),
                    "adapter_mode": adapter_mode,
                    "prior_json": str(prior_path.resolve()),
                    "config_json": str(config_path.resolve()),
                }
            )
    return rows


def write_reports(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["source_class"] for row in rows)
    payload = {
        "matrix_cells": len(rows),
        "counts": dict(counts),
        "rows": rows,
        "note": (
            "This report records the engineering prior actually used to build generation packages. "
            "When compact_fallback_transition_count dominates, results should not be described as full GCAD gradient-miner outputs."
        ),
    }
    (out_dir / "causal_prior_source_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Causal Prior Source Report",
        "",
        "This report records the prior source used by the generation packages.",
        "",
        "## Counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    if counts.get("compact_fallback_transition_count", 0) > len(rows) / 2:
        lines.extend(
            [
                "",
                "Most cells use compact transition-count fallback prior. Do not describe this run as a full GCAD GradientCausalMiner run.",
            ]
        )
    lines.extend(
        [
            "",
            "| dataset | scenario | seed | source_class | prior_source | adapter_mode |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['scenario']} | {row['seed']} | {row['source_class']} | "
            f"{row['prior_source']} | {row['adapter_mode']} |"
        )
    (out_dir / "causal_prior_source_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows()
    write_reports(args.out_dir.resolve(), rows)
    counts = Counter(row["source_class"] for row in rows)
    print(f"causal prior rows: {len(rows)}")
    print("counts:", dict(counts))
    print(f"saved: {args.out_dir / 'causal_prior_source_report.md'}")


if __name__ == "__main__":
    main()
