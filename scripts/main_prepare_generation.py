#!/usr/bin/env python
"""实验级入口：准备 causal-GSS prompt 与 generation package。"""

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
    DEFAULT_OUTPUT_ROOT,
    DICTIONARY_PY,
    experiment_key,
    source_pkl_for,
    stage_paths,
    target_pkl_for,
)
from causal_smart_home.gen_downstream_ad import DATASETS, ENV_BY_SCENARIO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare causal-GSS prompt artifacts and generation package.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-pkl", type=Path)
    parser.add_argument("--target-pkl", type=Path)
    parser.add_argument("--method-line", choices=["zero_target", "target_assisted"], default="target_assisted")
    parser.add_argument("--device-dict", type=Path, default=DICTIONARY_PY)
    parser.add_argument("--prior-json")
    parser.add_argument("--prior-matrix-path")
    parser.add_argument("--adapter-mode", default="existing", choices=["existing", "compact_fallback"])
    parser.add_argument("--level", default="device", choices=["device", "action", "device_action"])
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--sparse-threshold", type=float, default=0.001)
    parser.add_argument("--lambda-causal", type=float, default=1.0)
    parser.add_argument("--reweight-mode", choices=["additive", "multiplicative"], default="multiplicative")
    parser.add_argument("--no-add-causal-edges", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--guard-mode", choices=["suppress", "downweight"], default="downweight")
    parser.add_argument("--max-overuse-ratio", type=float, default=1.25)
    parser.add_argument("--min-target-freq", type=float, default=0.001)
    parser.add_argument("--downweight-factor", type=float, default=0.25)
    parser.add_argument("--endpoint-policy", choices=["target", "source_or_target", "both"], default="target")
    parser.add_argument("--dry-run-command", action="store_true", help="Print resolved commands without executing them.")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    paths = stage_paths(args.out_root, args.dataset, args.scenario, args.seed)
    causal_gss_dir = paths.causal_gss_dir
    package_dir = paths.generation_package_dir
    source_pkl = args.source_pkl or source_pkl_for(args.dataset, args.scenario)
    target_pkl = args.target_pkl or (target_pkl_for(args.dataset, args.scenario) if args.method_line == "target_assisted" else None)
    if args.method_line == "zero_target" and args.target_pkl is not None:
        raise ValueError("zero_target generation preparation forbids --target-pkl")

    prompt_cmd = [
        sys.executable,
        "scripts/build_causal_gss_prompt.py",
        "--source-pkl",
        str(source_pkl),
        "--device-dict",
        str(args.device_dict),
        "--out-prompt",
        str(causal_gss_dir / "prompt.txt"),
        "--out-prior-json",
        str(causal_gss_dir / "resolved_causal_relation_prior.json"),
        "--out-reweighted-hints",
        str(causal_gss_dir / "causal_reweighted_gss_hints.json"),
        "--out-config",
        str(causal_gss_dir / "config.json"),
        "--adapter-mode",
        args.adapter_mode,
        "--adaptation-mode",
        "source_only" if args.method_line == "zero_target" else "target_assisted",
        "--level",
        args.level,
        "--lag",
        str(args.lag),
        "--sparse-threshold",
        str(args.sparse_threshold),
        "--seed",
        str(args.seed),
        "--lambda-causal",
        str(args.lambda_causal),
        "--reweight-mode",
        args.reweight_mode,
        "--top-k",
        str(args.top_k),
        "--guard-mode",
        args.guard_mode,
        "--max-overuse-ratio",
        str(args.max_overuse_ratio),
        "--min-target-freq",
        str(args.min_target_freq),
        "--downweight-factor",
        str(args.downweight_factor),
        "--endpoint-policy",
        args.endpoint_policy,
    ]
    if target_pkl is not None:
        prompt_cmd.extend(
            [
                "--target-pkl", str(target_pkl),
                "--out-target-adapted-prior", str(causal_gss_dir / "target_adapted_causal_prior.json"),
                "--out-guard-report", str(causal_gss_dir / "guard_report.json"),
            ]
        )
    if args.prior_json:
        prompt_cmd.extend(["--prior-json", args.prior_json])
    if args.prior_matrix_path:
        prompt_cmd.extend(["--prior-matrix-path", args.prior_matrix_path])
    if args.no_add_causal_edges:
        prompt_cmd.append("--no-add-causal-edges")

    package_cmd = [
        sys.executable,
        "scripts/build_codex_generation_package.py",
        "--causal-gss-dir",
        str(causal_gss_dir),
        "--out-dir",
        str(package_dir),
        "--scenario",
        experiment_key(args.dataset, args.scenario),
        "--seed",
        str(args.seed),
    ]
    return [prompt_cmd, package_cmd]


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for cmd in commands:
        print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run_command:
        return
    for cmd in commands:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
