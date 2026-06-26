from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MAIN_RESULTS = {
    "ablation_no_causal_tof": {
        "precision_mean": 0.8400260212783585,
        "recall_mean": 0.9922161172161172,
        "f1_mean": 0.8981913177582828,
        "fpr_mean": 0.2564102564102564,
    },
    "proposed_causal_gss_gpt55_causal_tof": {
        "precision_mean": 0.9538611152318317,
        "recall_mean": 0.9977106227106227,
        "f1_mean": 0.9752202755560685,
        "fpr_mean": 0.048534798534798536,
    },
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="csh",
        description="Utilities for the locked CausalSmartHome SP-ST experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-recovery", help="verify locked SP-ST metrics and frozen inputs")
    check.set_defaults(func=_check_recovery_command)

    summarize = subparsers.add_parser("summarize", help="summarize downstream AD metric files")
    summarize.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "outputs" / "main_experiment" / "downstream_ad")
    summarize.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "main_experiment" / "summary")
    summarize.add_argument("--metrics-glob", default="**/normalized_metrics.json")
    summarize.set_defaults(func=_summarize_command)

    check_data = subparsers.add_parser("check-gen-data", help="verify vendored SmartGen data for FR/SP/US main experiments")
    check_data.add_argument("--json", action="store_true")
    check_data.set_defaults(func=_check_gen_data_command)

    args = parser.parse_args(argv)
    args.func(args)


def _check_recovery_command(_args: argparse.Namespace) -> None:
    aggregate_path = PROJECT_ROOT / "outputs" / "main_experiment" / "summary" / "main_experiment_aggregate.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"missing aggregate summary: {aggregate_path}")

    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rows = {row["variant"]: row for row in payload.get("rows", [])}
    missing = sorted(set(EXPECTED_MAIN_RESULTS) - set(rows))
    if missing:
        raise AssertionError(f"missing variants in aggregate summary: {missing}")

    for variant, expected_metrics in EXPECTED_MAIN_RESULTS.items():
        row = rows[variant]
        for metric, expected_value in expected_metrics.items():
            _assert_close(f"{variant}.{metric}", row[metric], expected_value)

    frozen = PROJECT_ROOT / "outputs" / "main_experiment_frozen" / "sp_st_gpt55_proposed_3seed_20260623"
    required = [
        frozen / "generated" / "sp_st" / "seed2024" / "generated_gpt55_clean.pkl",
        frozen / "generated" / "sp_st" / "seed2025" / "generated_gpt55_clean.pkl",
        frozen / "generated" / "sp_st" / "seed2026" / "generated_gpt55_clean.pkl",
        frozen / "run_reproduce_from_frozen.sh",
    ]
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise AssertionError("missing frozen reproducibility files: " + "; ".join(missing_files))

    print("RECOVERY_INTEGRITY_OK: SP-ST good metrics and frozen inputs are present.")


def _summarize_command(args: argparse.Namespace) -> None:
    from scripts.summarize_main_experiment import (
        build_aggregate_rows,
        build_seed_delta_rows,
        collect_per_seed_rows,
        write_outputs,
    )

    per_seed_rows = collect_per_seed_rows(args.runs_root, args.metrics_glob)
    aggregate_rows = build_aggregate_rows(per_seed_rows)
    delta_rows, delta_meta = build_seed_delta_rows(per_seed_rows)
    write_outputs(args.out_dir, per_seed_rows, aggregate_rows, delta_rows, delta_meta)
    print(f"summary written to: {args.out_dir}")


def _check_gen_data_command(args: argparse.Namespace) -> None:
    from scripts.check_gen_main_data import build_report, print_text_report

    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report)
    if report["missing"]:
        raise SystemExit(1)


def _assert_close(name: str, got: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isfinite(float(got)) or abs(float(got) - expected) > tol:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


if __name__ == "__main__":
    main()
