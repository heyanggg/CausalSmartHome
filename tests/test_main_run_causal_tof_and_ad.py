from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_entrypoint_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "main_run_causal_tof_and_ad.py"
    spec = importlib.util.spec_from_file_location("main_run_causal_tof_and_ad", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path: Path, **overrides):
    defaults = {
        "dataset": "us",
        "scenario": "st",
        "seed": 2024,
        "input_root": tmp_path / "input",
        "out_root": tmp_path / "out",
        "gen_tof_pkl": None,
        "guarded_hints_json": None,
        "target_pkl": None,
        "causal_tof_config": None,
        "ignore_causal_tof_config": False,
        "mode": None,
        "temperature": None,
        "alpha_rec": None,
        "beta_violation": None,
        "gamma_dist": None,
        "penalize_downweighted_edges": None,
        "min_weight": None,
        "max_copies": None,
        "resample_size": None,
        "causal_tof_seed": None,
        "skip_ad": True,
        "gen_root": tmp_path / "gen_core",
        "epochs": 15,
        "split_ratio": 0.8,
        "device": "cuda",
        "cuda_visible_devices": "0",
        "threshold_percentage": None,
        "dry_run_command": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def prepare_input_cell(tmp_path: Path, config: dict) -> Path:
    seed_dir = tmp_path / "input" / "us_st" / "seed2024"
    (seed_dir / "gen_original_tof").mkdir(parents=True)
    (seed_dir / "causal_gss").mkdir(parents=True)
    (seed_dir / "causal_tof").mkdir(parents=True)
    (seed_dir / "gen_original_tof" / "gen_tof.pkl").write_bytes(b"placeholder")
    (seed_dir / "causal_gss" / "guarded_reweighted_gss_hints.json").write_text("{}", encoding="utf-8")
    config_path = seed_dir / "causal_tof" / "generated_gen_tof_causal_tof.pkl.config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def write_causal_gss_config(tmp_path: Path, target_pkl: Path) -> None:
    config = tmp_path / "input" / "us_st" / "seed2024" / "causal_gss" / "config.json"
    config.write_text(json.dumps({"target_pkl": str(target_pkl)}), encoding="utf-8")


def test_causal_tof_main_reads_saved_config(tmp_path: Path):
    module = load_entrypoint_module()
    prepare_input_cell(
        tmp_path,
        {
            "mode": "filter",
            "min_weight": 0.2,
            "seed": 77,
            "penalize_downweighted_edges": True,
        },
    )

    command = module.build_commands(make_args(tmp_path))[0]

    assert command[command.index("--mode") + 1] == "filter"
    assert command[command.index("--min-weight") + 1] == "0.2"
    assert command[command.index("--seed") + 1] == "77"
    assert "--penalize-downweighted-edges" in command


def test_causal_tof_main_cli_overrides_saved_config(tmp_path: Path):
    module = load_entrypoint_module()
    prepare_input_cell(
        tmp_path,
        {
            "mode": "filter",
            "min_weight": 0.2,
            "seed": 77,
            "penalize_downweighted_edges": True,
        },
    )

    command = module.build_commands(
        make_args(
            tmp_path,
            mode="weight",
            min_weight=0.05,
            causal_tof_seed=2024,
            penalize_downweighted_edges=False,
        )
    )[0]

    assert command[command.index("--mode") + 1] == "weight"
    assert command[command.index("--min-weight") + 1] == "0.05"
    assert command[command.index("--seed") + 1] == "2024"
    assert "--penalize-downweighted-edges" not in command


def test_causal_tof_main_prefers_saved_target_pkl(tmp_path: Path):
    module = load_entrypoint_module()
    prepare_input_cell(tmp_path, {"mode": "weight"})
    target = tmp_path / "target_from_config.pkl"
    target.write_bytes(b"placeholder")
    write_causal_gss_config(tmp_path, target)

    command = module.build_commands(make_args(tmp_path))[0]

    assert command[command.index("--target-pkl") + 1] == str(target)


def test_causal_tof_main_passes_short_scenario_to_downstream_ad(tmp_path: Path):
    module = load_entrypoint_module()
    prepare_input_cell(tmp_path, {"mode": "weight"})
    pre_tof = tmp_path / "input" / "us_st" / "seed2024" / "codex_generation" / "generated_codex.pkl"
    pre_tof.parent.mkdir()
    pre_tof.write_bytes(b"placeholder")

    commands = module.build_commands(make_args(tmp_path, scenario="spring", skip_ad=False))

    ad_command = commands[1]
    assert ad_command[ad_command.index("--scenario") + 1] == "st"
