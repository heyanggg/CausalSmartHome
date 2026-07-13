#!/usr/bin/env python
"""Run one Causal-TOF variant and optionally its unchanged downstream AD."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_paths import (
    CAUSAL_TOF_ONLY_VARIANT,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FULL_CAUSAL_VARIANT,
    GEN_ROOT,
    baseline_gen_pkl_for,
    default_causal_tof_dir,
    default_downstream_out_dir,
    stage_paths,
    target_pkl_for,
)
from causal_smart_home.gen_downstream_ad import DATASETS, ENV_BY_SCENARIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variant", choices=[CAUSAL_TOF_ONLY_VARIANT, FULL_CAUSAL_VARIANT], default=FULL_CAUSAL_VARIANT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-pkl", type=Path)
    parser.add_argument("--guarded-hints-json", type=Path)
    parser.add_argument("--gen-tof-pkl", type=Path)
    parser.add_argument("--mode", choices=["rank", "weight", "filter"], default="weight")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-rec", type=float, default=1.0)
    parser.add_argument("--beta-inconsistency", type=float, default=1.0)
    parser.add_argument("--gamma-dist", type=float, default=1.0)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-copies", type=int, default=3)
    parser.add_argument("--resample-size", type=int)
    parser.add_argument("--skip-ad", action="store_true")
    parser.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-command", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    inputs = stage_paths(args.input_root, args.dataset, args.scenario, args.seed)
    causal_dir = default_causal_tof_dir(args.out_root, args.dataset, args.scenario, args.seed)
    target_pkl = args.target_pkl or target_pkl_for(args.dataset, args.scenario)
    if args.variant == FULL_CAUSAL_VARIANT:
        generated = args.gen_tof_pkl or inputs.gen_tof_pkl
        output = causal_dir / "generated_gen_tof_causal_tof.pkl"
        prefix = "full_causal"
        pre_tof = inputs.pre_tof_pkl if inputs.pre_tof_pkl.exists() else None
        gen_tof = generated
    else:
        generated = baseline_gen_pkl_for(args.dataset, args.scenario)
        output = causal_dir / "baseline_gen_causal_tof.pkl"
        prefix = "causal_tof_only"
        pre_tof = generated
        gen_tof = generated
    hints = args.guarded_hints_json or inputs.guarded_hints_json
    for label, path in (("generated", generated), ("hints", hints), ("target", target_pkl)):
        if not path.exists():
            raise FileNotFoundError(f"{label} input not found: {path}")

    causal = [
        sys.executable,
        "scripts/run_causal_tof.py",
        "--generated-pkl", str(generated),
        "--guarded-hints-json", str(hints),
        "--target-pkl", str(target_pkl),
        "--out-scores", str(causal_dir / f"{prefix}_scores.json"),
        "--out-weights", str(causal_dir / f"{prefix}_weights.json"),
        "--out-weighted-resampled-pkl", str(output),
        "--mode", args.mode,
        "--temperature", str(args.temperature),
        "--alpha-rec", str(args.alpha_rec),
        "--beta-inconsistency", str(args.beta_inconsistency),
        "--gamma-dist", str(args.gamma_dist),
        "--min-weight", str(args.min_weight),
        "--max-copies", str(args.max_copies),
        "--seed", str(args.seed),
    ]
    if args.resample_size is not None:
        causal.extend(["--resample-size", str(args.resample_size)])
    commands = [causal]
    if not args.skip_ad:
        ad = [
            sys.executable,
            "scripts/run_gen_downstream_ad.py",
            "--dataset", args.dataset,
            "--scenario", args.scenario,
            "--variant", args.variant,
            "--generated-pkl", str(output),
            "--gen-tof-pkl", str(gen_tof),
            "--seed", str(args.seed),
            "--out-dir", str(default_downstream_out_dir(args.out_root, args.dataset, args.scenario, args.seed, args.variant)),
            "--gen-root", str(args.gen_root),
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--cuda-visible-devices", args.cuda_visible_devices,
        ]
        if pre_tof is not None:
            ad.extend(["--pre-tof-pkl", str(pre_tof)])
        if args.dry_run:
            ad.append("--dry-run")
        commands.append(ad)
    quality_dir = Path(args.out_root) / inputs.key / f"seed{args.seed}" / "generation_quality" / args.variant
    commands.append(
        [
            sys.executable,
            "scripts/evaluate_generation_quality.py",
            "--target-pkl", str(target_pkl),
            "--synthetic-pkl", str(output),
            "--out-dir", str(quality_dir),
            "--dataset", args.dataset,
            "--scenario", args.scenario,
            "--seed", str(args.seed),
            "--variant", args.variant,
        ]
    )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    print(json.dumps([" ".join(shlex.quote(part) for part in command) for command in commands], ensure_ascii=False, indent=2))
    if args.dry_run_command:
        return
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
