from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.run_stage4c_gen_builtin_downstream_ad import normalize_metrics
from scripts.summarize_stage4_gen_builtin_ad import collect, compare, summarize


def test_normalize_metrics_adds_rates_and_stage4_metadata() -> None:
    args = Namespace(
        dataset="fr",
        scenario="st",
        variant="stage4_downweight_multiplicative_raw",
        seed=2024,
        threshold_percentage=None,
        epochs=15,
        split_ratio=0.8,
        device="cpu",
    )
    payload = {
        "precision": 0.8,
        "recall": 0.5,
        "accuracy": 0.75,
        "F1 score": 0.615,
        "learned_threshold": 1.25,
        "train_size": 80,
        "threshold_vld_size": 20,
        "test_size": 100,
        "synthetic_size": 125,
        "TP": 10,
        "TN": 65,
        "FP": 5,
        "FN": 20,
    }

    row = normalize_metrics(payload, args)

    assert row["downstream_pipeline"] == "smartgen_builtin_anomaly_detection_pipeline"
    assert row["generator"] == "codex_gpt55_surrogate"
    assert row["api_llm"] is False
    assert row["surrogate_for_smartgen_llm"] is True
    assert row["f1"] == 0.615
    assert row["fpr"] == 5 / 70
    assert row["fnr"] == 20 / 30
    assert row["validation_size"] == 20
    assert row["threshold_source"] == "validation_percentile_95.5"


def test_summarize_and_compare_multiseed(tmp_path: Path) -> None:
    root = tmp_path / "gen_builtin_ad"
    rows = [
        ("fr_st/stage3_prompt_only_baseline_seed2024", "stage3_prompt_only_baseline", 2024, 0.70, 0.20),
        ("fr_st/stage3_prompt_only_baseline_seed2025", "stage3_prompt_only_baseline", 2025, 0.80, 0.22),
        ("fr_st/stage4_downweight_multiplicative_raw_seed2024", "stage4_downweight_multiplicative_raw", 2024, 0.82, 0.18),
        ("fr_st/stage4_downweight_multiplicative_raw_seed2025", "stage4_downweight_multiplicative_raw", 2025, 0.86, 0.17),
    ]
    for rel, variant, seed, f1, fpr in rows:
        out = root / rel
        out.mkdir(parents=True)
        payload = {
            "dataset": "fr",
            "scenario": "st",
            "variant": variant,
            "seed": seed,
            "precision": f1,
            "recall": f1,
            "f1": f1,
            "fpr": fpr,
            "fnr": 1.0 - f1,
            "accuracy": f1,
            "generated_size": 10,
            "downstream_pipeline": "smartgen_builtin_anomaly_detection_pipeline",
            "generator": "codex_gpt55_surrogate",
            "api_llm": False,
        }
        (out / "downstream_ad_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    successes, failures = collect(root)
    summaries = summarize(successes)
    comparisons = compare(summaries)

    assert failures == []
    baseline = next(row for row in summaries if row["variant"] == "stage3_prompt_only_baseline")
    raw = next(row for row in summaries if row["variant"] == "stage4_downweight_multiplicative_raw")
    assert baseline["num_successful_seeds"] == 2
    assert baseline["mean_f1"] == 0.75
    assert raw["mean_f1"] == 0.84
    raw_vs_baseline = next(
        row
        for row in comparisons
        if row["comparison"] == "stage4_downweight_multiplicative_raw vs stage3_prompt_only_baseline"
    )
    assert raw_vs_baseline["status"] == "available"
    assert raw_vs_baseline["delta_mean_f1"] == 0.08999999999999997
