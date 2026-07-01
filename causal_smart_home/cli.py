"""CausalSmartHome 常用维护任务的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """分发 ``csh`` console script 的子命令。"""
    parser = argparse.ArgumentParser(
        prog="csh",
        description="Utilities for CausalSmartHome main experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from .experiment_paths import DEFAULT_INPUT_ROOT

    summarize = subparsers.add_parser("summarize", help="write per-seed downstream AD summary files")
    summarize.add_argument("--runs-root", type=Path, default=DEFAULT_INPUT_ROOT)
    summarize.add_argument("--out-dir", type=Path, default=DEFAULT_INPUT_ROOT / "summary")
    summarize.add_argument("--metrics-glob", default="**/normalized_metrics.json")
    summarize.set_defaults(func=_summarize_command)

    check_data = subparsers.add_parser("check-gen-data", help="verify in-project Gen data for FR/SP/US main experiments")
    check_data.add_argument("--json", action="store_true")
    check_data.set_defaults(func=_check_gen_data_command)

    args = parser.parse_args(argv)
    args.func(args)


def _summarize_command(args: argparse.Namespace) -> None:
    """写出主实验 per-seed 汇总产物。"""
    from scripts.summarize_main_experiment import (
        collect_per_seed_rows,
        write_outputs,
    )

    per_seed_rows = collect_per_seed_rows(args.runs_root, args.metrics_glob)
    write_outputs(args.out_dir, per_seed_rows)
    print(f"summary written to: {args.out_dir}")


def _check_gen_data_command(args: argparse.Namespace) -> None:
    """检查实验所需的项目内 Gen 数据、checkpoint 和参考结果是否齐全。"""
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
