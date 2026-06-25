import json

from causal_smart_home.experiment_matrix import ABLATION_VARIANT, PROPOSED_VARIANT, REFERENCE_VARIANT
from scripts.summarize_main_experiment import (
    build_ablation_rows,
    build_main_per_seed_rows,
    build_main_vs_gen_rows,
    collect_per_seed_rows,
)


def _write_metrics(root, variant, seed, f1, precision=0.5):
    run = root / f"sp_st_{variant}_seed{seed}"
    run.mkdir(parents=True)
    payload = {
        "dataset": "sp",
        "scenario": "st",
        "seed": seed,
        "variant": variant,
        "precision": precision,
        "recall": 1.0,
        "f1": f1,
        "accuracy": 0.8,
        "fpr": 0.2,
        "fnr": 0.0,
        "status": "success",
        "run_dir": str(run),
        "metrics_path": str(run / "downstream_ad_metrics.json"),
    }
    (run / "normalized_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _reference():
    ref = {
        ("sp", "st"): {
            "dataset": "sp",
            "scenario": "st",
            "precision": 0.8573,
            "recall": 0.9904,
            "f1": 0.9191,
            "source": "SmartGen paper Table 3, SmartGen column",
        }
    }
    for dataset in ("fr", "us"):
        for scenario in ("st", "tt", "nt"):
            ref[(dataset, scenario)] = {
                "dataset": dataset,
                "scenario": scenario,
                "precision": 0.1,
                "recall": 0.1,
                "f1": 0.1,
                "source": "SmartGen paper Table 3, SmartGen column",
            }
    for scenario in ("tt", "nt"):
        ref[("sp", scenario)] = {
            "dataset": "sp",
            "scenario": scenario,
            "precision": 0.1,
            "recall": 0.1,
            "f1": 0.1,
            "source": "SmartGen paper Table 3, SmartGen column",
        }
    return ref


def test_main_summary_uses_reference_not_ablation_as_baseline(tmp_path):
    _write_metrics(tmp_path, ABLATION_VARIANT, 2024, 0.75)
    _write_metrics(tmp_path, PROPOSED_VARIANT, 2024, 0.80)

    rows = collect_per_seed_rows(tmp_path)
    main_rows = build_main_per_seed_rows(rows, _reference())
    main_vs_gen = build_main_vs_gen_rows(rows, _reference())

    sp_st_2024_variants = {
        row["variant"]
        for row in main_rows
        if row["dataset"] == "sp" and row["scenario"] == "st" and row["seed"] == 2024
    }
    assert sp_st_2024_variants == {REFERENCE_VARIANT, PROPOSED_VARIANT}
    assert ABLATION_VARIANT not in sp_st_2024_variants

    sp_st_delta = next(row for row in main_vs_gen if row["dataset"] == "sp" and row["scenario"] == "st" and row["seed"] == 2024)
    assert round(sp_st_delta["delta_f1"], 6) == round(0.80 - 0.9191, 6)


def test_ablation_summary_contains_ablation_only_as_ablation(tmp_path):
    _write_metrics(tmp_path, ABLATION_VARIANT, 2024, 0.75)
    _write_metrics(tmp_path, PROPOSED_VARIANT, 2024, 0.80)

    rows = collect_per_seed_rows(tmp_path)
    ablation_rows = build_ablation_rows(rows)
    sp_st = next(row for row in ablation_rows if row["dataset"] == "sp" and row["scenario"] == "st" and row["seed"] == 2024)

    assert sp_st["ablation_f1"] == 0.75
    assert sp_st["proposed_f1"] == 0.80
    assert round(sp_st["delta_f1"], 6) == 0.05
