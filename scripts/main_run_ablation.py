#!/usr/bin/env python
"""Run the four-variant ablation for exactly one dataset/scenario/seed cell."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_paths import (
    BASELINE_GEN_VARIANT,
    CAUSAL_GSS_ONLY_VARIANT,
    CAUSAL_TOF_ONLY_VARIANT,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FULL_CAUSAL_VARIANT,
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
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--guarded-hints-json", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-command", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    inputs = stage_paths(args.input_root, args.dataset, args.scenario, args.seed)
    target = target_pkl_for(args.dataset, args.scenario)
    baseline = baseline_gen_pkl_for(args.dataset, args.scenario)
    causal_dir = default_causal_tof_dir(args.out_root, args.dataset, args.scenario, args.seed)
    tof_outputs = {
        CAUSAL_TOF_ONLY_VARIANT: causal_dir / "baseline_gen_causal_tof.pkl",
        FULL_CAUSAL_VARIANT: causal_dir / "generated_gen_tof_causal_tof.pkl",
    }
    commands: list[list[str]] = []
    for variant in (CAUSAL_TOF_ONLY_VARIANT, FULL_CAUSAL_VARIANT):
        causal_command = [
                sys.executable, "scripts/main_run_causal_tof_and_ad.py",
                "--dataset", args.dataset, "--scenario", args.scenario, "--seed", str(args.seed),
                "--variant", variant, "--input-root", str(args.input_root), "--out-root", str(args.out_root),
                "--skip-ad",
            ]
        if args.guarded_hints_json:
            causal_command.extend(["--guarded-hints-json", str(args.guarded_hints_json)])
        commands.append(causal_command)

    generated = {
        BASELINE_GEN_VARIANT: baseline,
        CAUSAL_GSS_ONLY_VARIANT: inputs.gen_tof_pkl,
        **tof_outputs,
    }
    for variant in (BASELINE_GEN_VARIANT, CAUSAL_GSS_ONLY_VARIANT, CAUSAL_TOF_ONLY_VARIANT, FULL_CAUSAL_VARIANT):
        command = [
            sys.executable, "scripts/run_gen_downstream_ad.py",
            "--dataset", args.dataset, "--scenario", args.scenario, "--variant", variant,
            "--generated-pkl", str(generated[variant]), "--seed", str(args.seed),
            "--out-dir", str(default_downstream_out_dir(args.out_root, args.dataset, args.scenario, args.seed, variant)),
            "--epochs", str(args.epochs), "--device", args.device,
            "--cuda-visible-devices", args.cuda_visible_devices,
        ]
        if variant in {CAUSAL_GSS_ONLY_VARIANT, FULL_CAUSAL_VARIANT} and inputs.pre_tof_pkl.exists():
            command.extend(["--pre-tof-pkl", str(inputs.pre_tof_pkl)])
        command.extend(["--gen-tof-pkl", str(baseline if variant in {BASELINE_GEN_VARIANT, CAUSAL_TOF_ONLY_VARIANT} else inputs.gen_tof_pkl)])
        if args.dry_run:
            command.append("--dry-run")
        commands.append(command)

    # Causal variants are evaluated by main_run_causal_tof_and_ad; add the two
    # non-Causal-TOF quality rows here.
    for variant in (BASELINE_GEN_VARIANT, CAUSAL_GSS_ONLY_VARIANT):
        quality_dir = Path(args.out_root) / inputs.key / f"seed{args.seed}" / "generation_quality" / variant
        commands.append(
            [
                sys.executable, "scripts/evaluate_generation_quality.py",
                "--target-pkl", str(target), "--synthetic-pkl", str(generated[variant]),
                "--out-dir", str(quality_dir), "--dataset", args.dataset, "--scenario", args.scenario,
                "--seed", str(args.seed), "--variant", variant,
            ]
        )
    commands.extend(
        [
            [sys.executable, "scripts/summarize_main_experiment.py", "--runs-root", str(args.out_root), "--out-dir", str(Path(args.out_root) / "summary")],
            [sys.executable, "scripts/summarize_generation_quality.py", "--runs-root", str(args.out_root), "--out", str(Path(args.out_root) / "summary" / "generation_quality_summary.json")],
        ]
    )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run_command:
        return
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
