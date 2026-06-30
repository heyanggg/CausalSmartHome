from __future__ import annotations

from pathlib import Path

from causal_smart_home.experiment_paths import experiment_key, stage_paths, target_pkl_for


def test_experiment_key_normalizes_long_scenario_names(tmp_path: Path):
    paths = stage_paths(tmp_path, "us", "spring", 2024)

    assert experiment_key("us", "spring") == "us_st"
    assert experiment_key("us", "st") == "us_st"
    assert paths.seed_dir == tmp_path / "us_st" / "seed2024"


def test_target_pkl_mapping_matches_main_experiment_policy():
    assert target_pkl_for("fr", "st").as_posix().endswith("fr/spring/split_test.pkl")
    assert target_pkl_for("fr", "tt").as_posix().endswith("fr/night/split_test.pkl")
    assert target_pkl_for("sp", "st").as_posix().endswith("sp/spring/split_test.pkl")
    assert target_pkl_for("us", "st").as_posix().endswith("us/spring/test.pkl")
    assert target_pkl_for("sp", "nt").as_posix().endswith("sp/multiple/test.pkl")
