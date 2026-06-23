#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.smartgen_original_tof import SmartGenOriginalTOFConfig, run_smartgen_original_tof


def default_smartgen_root() -> Path:
    local = REPO_ROOT / "external_sources" / "SmartGen"
    return local if local.exists() else Path("/home/heyang/projects/SmartGen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmartGen original two-stage TOF before Stage4 downstream AD.")
    parser.add_argument("--generated-pkl", required=True, type=Path, help="Fresh Stage4 generated pkl before SmartGen TOF.")
    parser.add_argument("--dataset", required=True, choices=["fr", "sp", "us"])
    parser.add_argument("--scenario", required=True, choices=["st", "tt", "nt"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smartgen-root", type=Path, default=default_smartgen_root())
    parser.add_argument("--security-check-path", type=Path, help="Optional explicit path to SmartGen/security_check.py.")
    parser.add_argument("--method", default="SPPC")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--threshold", help="SmartGen threshold string; defaults to SmartGen paper/script threshold for dataset/env.")
    parser.add_argument("--out-dir", type=Path, help="Defaults to outputs/gcad_gss_stage4/smartgen_original_tof/{dataset}_{scenario}/seed{seed}.")
    parser.add_argument("--out-pkl", type=Path, help="Defaults to <out-dir>/smartgen_tof.pkl.")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Copy input to output and record that original TOF was not executed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "gcad_gss_stage4" / "smartgen_original_tof" / f"{args.dataset}_{args.scenario}" / f"seed{args.seed}")
    out_pkl = args.out_pkl or (out_dir / "smartgen_tof.pkl")
    report = run_smartgen_original_tof(
        SmartGenOriginalTOFConfig(
            smartgen_root=args.smartgen_root,
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
    print(f"saved SmartGen TOF pkl: {report.get('out_pkl')}")
    print(f"saved report: {out_dir / 'smartgen_original_tof_report.json'}")


if __name__ == "__main__":
    main()
