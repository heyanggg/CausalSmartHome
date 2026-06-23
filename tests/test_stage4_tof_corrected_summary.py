import json
import pickle

from scripts.summarize_stage4_gen_builtin_ad_tof_corrected import (
    build_aggregate_rows,
    build_seed_delta_rows,
    collect_per_seed_rows,
)


def _write_metrics(root, variant, seed, f1, precision=0.5):
    run = root / f"fr_st_{variant}_seed{seed}"
    run.mkdir(parents=True)
    payload = {
        "dataset": "fr",
        "scenario": "st",
        "seed": seed,
        "variant": variant,
        "input_pkl": str(run / "input.pkl"),
        "input_stage": "smartgen_original_tof" if "raw" not in variant else "fresh_generated_no_smartgen_tof",
        "used_smartgen_original_tof": "raw" not in variant,
        "used_causal_tof": "plus_causal" in variant,
        "downstream_pipeline": "smartgen_builtin_anomaly_detection_pipeline",
        "generator": "gpt55_generation",
        "api_llm": False,
        "num_generated_before_tof": 10,
        "num_generated_after_smartgen_tof": 8 if "raw" not in variant else None,
        "num_generated_after_causal_tof": 7 if "plus_causal" in variant else None,
        "train_size": 6,
        "validation_size": 2,
        "test_size": 5,
        "threshold": 0.1,
        "threshold_source": "validation_percentile_95.5",
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


def test_corrected_summary_collects_per_seed_aggregate_and_deltas(tmp_path):
    _write_metrics(tmp_path, "stage3_prompt_only_smartgen_tof", 2024, 0.70)
    _write_metrics(tmp_path, "stage4_raw_no_smartgen_tof", 2024, 0.60)
    _write_metrics(tmp_path, "stage4_smartgen_original_tof", 2024, 0.75)
    _write_metrics(tmp_path, "stage4_smartgen_original_tof_plus_causal_tof", 2024, 0.80)

    rows = collect_per_seed_rows(tmp_path)
    assert len(rows) == 4
    assert {row["variant"] for row in rows} == {
        "stage3_prompt_only_smartgen_tof",
        "stage4_raw_no_smartgen_tof",
        "stage4_smartgen_original_tof",
        "stage4_smartgen_original_tof_plus_causal_tof",
    }
    aggregate = build_aggregate_rows(rows)
    assert all(row["table_type"] == "aggregate_mean_std_not_replacement_for_per_seed" for row in aggregate)

    deltas, meta = build_seed_delta_rows(rows)
    assert meta["stage3_available"] is True
    plus_vs_original = next(row for row in deltas if row["comparison"] == "stage4_smartgen_original_tof_plus_causal_tof vs stage4_smartgen_original_tof")
    assert round(plus_vs_original["f1_delta"], 6) == 0.05
    raw_vs_original = next(row for row in deltas if row["comparison"] == "stage4_raw_no_smartgen_tof vs stage4_smartgen_original_tof")
    assert round(raw_vs_original["f1_delta"], 6) == -0.15


def test_corrected_summary_stage3_absent_does_not_fail(tmp_path):
    _write_metrics(tmp_path, "stage4_raw_no_smartgen_tof", 2024, 0.60)
    _write_metrics(tmp_path, "stage4_smartgen_original_tof", 2024, 0.75)
    rows = collect_per_seed_rows(tmp_path)
    deltas, meta = build_seed_delta_rows(rows)
    assert meta["stage3_available"] is False
    assert len(deltas) == 1
    assert deltas[0]["comparison"] == "stage4_raw_no_smartgen_tof vs stage4_smartgen_original_tof"
