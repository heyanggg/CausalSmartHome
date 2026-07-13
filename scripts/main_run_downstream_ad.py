#!/usr/bin/env python
"""实验级入口：运行单个下游 AD 实验。

该入口面向执行实验的人：只需要指定 dataset/scenario/seed/variant，就会按项目
约定自动定位阶段输入，并调用底层 ``run_gen_downstream_ad.py``。
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_paths import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GEN_ROOT,
    VARIANTS,
    default_downstream_out_dir,
    input_for_variant,
    require_file,
    scenario_key,
    stage_paths,
)
from causal_smart_home.gen_downstream_ad import DATASETS, ENV_BY_SCENARIO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CausalSmartHome downstream AD experiment.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full_causal")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-pkl", type=Path, help="Override the inferred AD input pkl.")
    parser.add_argument("--pre-tof-pkl", type=Path, help="Optional provenance/count pkl before Gen TOF.")
    parser.add_argument("--gen-tof-pkl", type=Path, help="Optional provenance/count pkl after Gen TOF.")
    parser.add_argument("--out-dir", type=Path, help="Defaults to outputs/main_runs/{cell}/seed{seed}/downstream_ad/{variant}.")
    parser.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--threshold-percentage", type=float)
    parser.add_argument("--validation-pkl", type=Path)
    parser.add_argument("--attack-pkl", type=Path)
    parser.add_argument("--target-test-pkl", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Pass dry-run to the downstream AD wrapper.")
    parser.add_argument("--dry-run-command", action="store_true", help="Print the resolved command without executing it.")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    paths = stage_paths(args.input_root, args.dataset, args.scenario, args.seed)
    inferred_generated, inferred_pre_tof, inferred_gen_tof = input_for_variant(paths, args.variant)

    generated_pkl = args.generated_pkl or inferred_generated
    pre_tof_pkl = args.pre_tof_pkl if args.pre_tof_pkl is not None else inferred_pre_tof
    gen_tof_pkl = args.gen_tof_pkl if args.gen_tof_pkl is not None else inferred_gen_tof
    out_dir = args.out_dir or default_downstream_out_dir(args.out_root, args.dataset, args.scenario, args.seed, args.variant)

    require_file(generated_pkl, "--generated-pkl")
    if pre_tof_pkl is not None:
        require_file(pre_tof_pkl, "--pre-tof-pkl")
    if gen_tof_pkl is not None:
        require_file(gen_tof_pkl, "--gen-tof-pkl")

    cmd = [
        sys.executable,
        "scripts/run_gen_downstream_ad.py",
        "--dataset",
        args.dataset,
        "--scenario",
        scenario_key(args.scenario),
        "--variant",
        args.variant,
        "--generated-pkl",
        str(generated_pkl),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(out_dir),
        "--gen-root",
        str(args.gen_root),
        "--epochs",
        str(args.epochs),
        "--split-ratio",
        str(args.split_ratio),
        "--device",
        args.device,
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
    ]
    if pre_tof_pkl is not None:
        cmd.extend(["--pre-tof-pkl", str(pre_tof_pkl)])
    if gen_tof_pkl is not None:
        cmd.extend(["--gen-tof-pkl", str(gen_tof_pkl)])
    if args.threshold_percentage is not None:
        cmd.extend(["--threshold-percentage", str(args.threshold_percentage)])
    if args.validation_pkl is not None:
        cmd.extend(["--validation-pkl", str(args.validation_pkl)])
    if args.attack_pkl is not None:
        cmd.extend(["--attack-pkl", str(args.attack_pkl)])
    if args.target_test_pkl is not None:
        cmd.extend(["--target-test-pkl", str(args.target_test_pkl)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run_command:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
