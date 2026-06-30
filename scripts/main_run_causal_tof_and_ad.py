#!/usr/bin/env python
"""实验级入口：运行 Causal-TOF，并可继续运行 proposed 下游 AD。"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_paths import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GEN_ROOT,
    PROPOSED_VARIANT,
    default_causal_tof_dir,
    default_downstream_out_dir,
    require_file,
    scenario_key,
    stage_paths,
    target_pkl_from_stage_config,
    target_pkl_for,
)
from causal_smart_home.gen_downstream_ad import DATASETS, ENV_BY_SCENARIO  # noqa: E402


CAUSAL_TOF_DEFAULTS = {
    "mode": "weight",
    "temperature": 2.0,
    "alpha_rec": 1.0,
    "beta_violation": 1.0,
    "gamma_dist": 1.0,
    "penalize_downweighted_edges": False,
    "min_weight": 0.05,
    "max_copies": 3,
    "resample_size": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Causal-TOF and optional proposed downstream AD.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gen-tof-pkl", type=Path, help="Defaults to {input-root}/{cell}/seed{seed}/gen_original_tof/gen_tof.pkl.")
    parser.add_argument("--guarded-hints-json", type=Path, help="Defaults to {input-root}/{cell}/seed{seed}/causal_gss/guarded_reweighted_gss_hints.json.")
    parser.add_argument("--target-pkl", type=Path, help="Defaults to the saved causal-GSS target pkl, then the main-experiment target pkl mapping.")
    parser.add_argument("--causal-tof-config", type=Path, help="Defaults to {input-root}/{cell}/seed{seed}/causal_tof/*.config.json when present.")
    parser.add_argument("--ignore-causal-tof-config", action="store_true", help="Use built-in defaults plus explicit CLI options instead of a saved Causal-TOF config.")
    parser.add_argument("--mode", choices=["rank", "weight", "filter"])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--alpha-rec", type=float)
    parser.add_argument("--beta-violation", type=float)
    parser.add_argument("--gamma-dist", type=float)
    parser.set_defaults(penalize_downweighted_edges=None)
    penalize = parser.add_mutually_exclusive_group()
    penalize.add_argument("--penalize-downweighted-edges", dest="penalize_downweighted_edges", action="store_true")
    penalize.add_argument("--no-penalize-downweighted-edges", dest="penalize_downweighted_edges", action="store_false")
    parser.add_argument("--min-weight", type=float)
    parser.add_argument("--max-copies", type=int)
    parser.add_argument("--resample-size", type=int)
    parser.add_argument("--causal-tof-seed", type=int, help="Causal-TOF sampling seed. Defaults to the saved Causal-TOF config seed, then --seed.")
    parser.add_argument("--skip-ad", action="store_true", help="Only run Causal-TOF.")
    parser.add_argument("--gen-root", type=Path, default=GEN_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--threshold-percentage", type=float)
    parser.add_argument("--dry-run-command", action="store_true", help="Print resolved commands without executing them.")
    return parser.parse_args()


def default_causal_tof_config_path(input_paths) -> Path | None:
    """返回该实验单元已有的 Causal-TOF 配置文件；没有则返回 None。"""
    exact = input_paths.causal_tof_pkl.with_suffix(input_paths.causal_tof_pkl.suffix + ".config.json")
    if exact.exists():
        return exact
    candidates = sorted(input_paths.causal_tof_dir.glob("*.config.json"))
    return candidates[0] if candidates else None


def load_causal_tof_config(args: argparse.Namespace, input_paths) -> tuple[dict[str, Any], Path | None]:
    """读取 Causal-TOF 阶段配置，显式命令行参数会在后续解析中覆盖它。"""
    if args.ignore_causal_tof_config:
        return {}, None
    config_path = args.causal_tof_config or default_causal_tof_config_path(input_paths)
    if config_path is None:
        return {}, None
    config_path = require_file(config_path, "--causal-tof-config")
    try:
        return json.loads(config_path.read_text(encoding="utf-8")), config_path
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Causal-TOF config JSON: {config_path}") from exc


def choose_value(args: argparse.Namespace, config: dict[str, Any], attr: str, default: Any) -> Any:
    """命令行显式值优先，其次是配置文件，最后是内置默认值。"""
    value = getattr(args, attr)
    if value is not None:
        return value
    return config.get(attr, default)


def resolve_causal_tof_options(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """合并 CLI、已保存配置和内置默认值，得到本次 Causal-TOF 的参数。"""
    options = {
        "mode": choose_value(args, config, "mode", CAUSAL_TOF_DEFAULTS["mode"]),
        "temperature": float(choose_value(args, config, "temperature", CAUSAL_TOF_DEFAULTS["temperature"])),
        "alpha_rec": float(choose_value(args, config, "alpha_rec", CAUSAL_TOF_DEFAULTS["alpha_rec"])),
        "beta_violation": float(choose_value(args, config, "beta_violation", CAUSAL_TOF_DEFAULTS["beta_violation"])),
        "gamma_dist": float(choose_value(args, config, "gamma_dist", CAUSAL_TOF_DEFAULTS["gamma_dist"])),
        "penalize_downweighted_edges": bool(
            choose_value(args, config, "penalize_downweighted_edges", CAUSAL_TOF_DEFAULTS["penalize_downweighted_edges"])
        ),
        "min_weight": float(choose_value(args, config, "min_weight", CAUSAL_TOF_DEFAULTS["min_weight"])),
        "max_copies": int(choose_value(args, config, "max_copies", CAUSAL_TOF_DEFAULTS["max_copies"])),
        "resample_size": choose_value(args, config, "resample_size", CAUSAL_TOF_DEFAULTS["resample_size"]),
        "seed": args.causal_tof_seed if args.causal_tof_seed is not None else int(config.get("seed", args.seed)),
    }
    if options["resample_size"] is None and options["mode"] in {"rank", "weight"} and "target_size" in config:
        options["resample_size"] = int(config["target_size"])
    if options["resample_size"] is not None:
        options["resample_size"] = int(options["resample_size"])
    return options


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    input_paths = stage_paths(args.input_root, args.dataset, args.scenario, args.seed)
    causal_tof_config, causal_tof_config_path = load_causal_tof_config(args, input_paths)
    causal_tof_options = resolve_causal_tof_options(args, causal_tof_config)
    gen_tof_pkl = require_file(args.gen_tof_pkl or input_paths.gen_tof_pkl, "--gen-tof-pkl")
    guarded_hints_json = require_file(args.guarded_hints_json or input_paths.guarded_hints_json, "--guarded-hints-json")
    inferred_target_pkl = target_pkl_from_stage_config(input_paths) or target_pkl_for(args.dataset, args.scenario)
    target_pkl = require_file(args.target_pkl or inferred_target_pkl, "--target-pkl")

    causal_tof_dir = default_causal_tof_dir(args.out_root, args.dataset, args.scenario, args.seed)
    causal_tof_pkl = causal_tof_dir / "generated_gen_tof_causal_tof.pkl"
    scores_json = causal_tof_dir / "causal_tof_scores.json"
    weights_json = causal_tof_dir / "generated.weights.json"

    causal_cmd = [
        sys.executable,
        "scripts/run_causal_tof.py",
        "--generated-pkl",
        str(gen_tof_pkl),
        "--guarded-hints-json",
        str(guarded_hints_json),
        "--target-pkl",
        str(target_pkl),
        "--out-scores",
        str(scores_json),
        "--out-weights",
        str(weights_json),
        "--out-weighted-resampled-pkl",
        str(causal_tof_pkl),
        "--mode",
        causal_tof_options["mode"],
        "--temperature",
        str(causal_tof_options["temperature"]),
        "--alpha-rec",
        str(causal_tof_options["alpha_rec"]),
        "--beta-violation",
        str(causal_tof_options["beta_violation"]),
        "--gamma-dist",
        str(causal_tof_options["gamma_dist"]),
        "--min-weight",
        str(causal_tof_options["min_weight"]),
        "--max-copies",
        str(causal_tof_options["max_copies"]),
        "--seed",
        str(causal_tof_options["seed"]),
    ]
    if causal_tof_options["penalize_downweighted_edges"]:
        causal_cmd.append("--penalize-downweighted-edges")
    if causal_tof_options["resample_size"] is not None:
        causal_cmd.extend(["--resample-size", str(causal_tof_options["resample_size"])])
    if causal_tof_config_path is not None:
        causal_cmd.extend(["--out-resampling-config", str(causal_tof_pkl.with_suffix(causal_tof_pkl.suffix + ".config.json"))])

    commands = [causal_cmd]
    if not args.skip_ad:
        out_dir = default_downstream_out_dir(args.out_root, args.dataset, args.scenario, args.seed, PROPOSED_VARIANT)
        ad_cmd = [
            sys.executable,
            "scripts/run_gen_downstream_ad.py",
            "--dataset",
            args.dataset,
            "--scenario",
            scenario_key(args.scenario),
            "--variant",
            PROPOSED_VARIANT,
            "--generated-pkl",
            str(causal_tof_pkl),
            "--gen-tof-pkl",
            str(gen_tof_pkl),
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
        if input_paths.pre_tof_pkl.exists():
            ad_cmd.extend(["--pre-tof-pkl", str(input_paths.pre_tof_pkl)])
        if args.threshold_percentage is not None:
            ad_cmd.extend(["--threshold-percentage", str(args.threshold_percentage)])
        commands.append(ad_cmd)
    return commands


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
