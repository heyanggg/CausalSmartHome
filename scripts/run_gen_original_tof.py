#!/usr/bin/env python
"""仓库内拷贝的 Gen 原始两阶段 TOF 步骤的 CLI 包装器。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.gen_downstream_ad import DATASETS, ENV_BY_SCENARIO
from causal_smart_home.gen_original_tof import GenOriginalTOFConfig, run_gen_original_tof


def default_gen_root() -> Path:
    """返回仓库内拷贝的 Gen 代码根目录。"""
    return REPO_ROOT / "causal_smart_home" / "gen_core"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gen original two-stage TOF before downstream AD.")
    parser.add_argument("--generated-pkl", required=True, type=Path, help="Fresh generated pkl before Gen TOF.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO), help="Gen target context: st/spring, tt/night, or nt/multiple.")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gen-root", type=Path, default=default_gen_root())
    parser.add_argument("--security-check-path", type=Path, help="Optional explicit path to Gen security_check.py.")
    parser.add_argument("--method", default="SPPC")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--threshold", help="Gen threshold string; defaults to the Gen threshold for dataset/env.")
    parser.add_argument("--out-dir", type=Path, help="Defaults to outputs/main_experiment/{dataset}_{scenario}/seed{seed}/gen_original_tof.")
    parser.add_argument("--out-pkl", type=Path, help="Defaults to <out-dir>/gen_tof.pkl.")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Copy input to output and record that original TOF was not executed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "main_experiment" / f"{args.dataset}_{args.scenario}" / f"seed{args.seed}" / "gen_original_tof")
    out_pkl = args.out_pkl or (out_dir / "gen_tof.pkl")
    report = run_gen_original_tof(
        GenOriginalTOFConfig(
            gen_root=args.gen_root,
            dataset=args.dataset,
            scenario=args.scenario,
            generated_pkl=args.generated_pkl,
            out_pkl=out_pkl,
            out_dir=out_dir,
            seed=args.seed,
            method=args.method,
            model=args.model,
            threshold=args.threshold,
            security_check_path=args.security_check_path,
            cuda_visible_devices=args.cuda_visible_devices,
            dry_run=args.dry_run,
        )
    )
    print(f"saved Gen TOF pkl: {report.get('out_pkl')}")
    print(f"saved report: {out_dir / 'gen_original_tof_report.json'}")


if __name__ == "__main__":
    main()
