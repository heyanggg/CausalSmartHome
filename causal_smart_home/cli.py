from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="csh",
        description="Utilities for CausalSmartHome main experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="write per-seed downstream AD summary files")
    summarize.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "outputs" / "main_experiment" / "downstream_ad")
    summarize.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "main_experiment" / "summary")
    summarize.add_argument("--metrics-glob", default="**/normalized_metrics.json")
    summarize.set_defaults(func=_summarize_command)

    check_data = subparsers.add_parser("check-gen-data", help="verify vendored SmartGen data for FR/SP/US main experiments")
    check_data.add_argument("--json", action="store_true")
    check_data.set_defaults(func=_check_gen_data_command)

    args = parser.parse_args(argv)
    args.func(args)


def _summarize_command(args: argparse.Namespace) -> None:
    from scripts.summarize_main_experiment import (
        collect_per_seed_rows,
        write_outputs,
    )

    per_seed_rows = collect_per_seed_rows(args.runs_root, args.metrics_glob)
    write_outputs(args.out_dir, per_seed_rows)
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


if __name__ == "__main__":
    main()
