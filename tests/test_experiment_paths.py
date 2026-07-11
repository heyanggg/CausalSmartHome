from __future__ import annotations

from pathlib import Path

from causal_smart_home.experiment_paths import experiment_key, stage_paths


def test_experiment_key_normalizes_long_scenario_names(tmp_path: Path):
    paths = stage_paths(tmp_path, "us", "spring", 2024)

    assert experiment_key("us", "spring") == "us_st"
    assert experiment_key("us", "st") == "us_st"
    assert paths.seed_dir == tmp_path / "us_st" / "seed2024"
