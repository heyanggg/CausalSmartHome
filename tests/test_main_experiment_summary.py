"""测试主实验 per-seed metrics 的保留与汇总逻辑。"""

import json

from scripts.summarize_main_experiment import (
    collect_per_seed_rows,
    write_outputs,
)


def _write_metrics(root, variant, seed, f1, precision=0.5):
    run = root / f"sp_st_{variant}_seed{seed}"
    run.mkdir(parents=True)
    payload = {
        "dataset": "sp",
        "scenario": "st",
        "seed": seed,
        "variant": variant,
        "input_pkl": str(run / "input.pkl"),
        "input_stage": "gen_original_tof",
        "used_gen_original_tof": True,
        "downstream_pipeline": "gen_builtin_downstream_ad",
        "generator": "codex_generation",
        "generation_model": "Codex",
        "num_generated_before_tof": 10,
        "num_generated_after_gen_tof": 8,
        "train_size": 6,
        "validation_size": 2,
        "test_size": 5,
        "threshold": 0.1,
        "threshold_source": "validation_percentile_95.0",
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


def test_main_summary_collects_per_seed_rows_only(tmp_path):
    _write_metrics(tmp_path, "baseline_gen", 2024, 0.75)
    _write_metrics(tmp_path, "full_causal", 2024, 0.80)

    rows = collect_per_seed_rows(tmp_path)
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"baseline_gen", "full_causal"}
    out_dir = tmp_path / "summary"
    write_outputs(out_dir, rows)

    assert (out_dir / "main_experiment_per_seed.json").exists()
    assert not (out_dir / "main_experiment_aggregate.json").exists()
    assert not (out_dir / "main_experiment_seed_deltas.json").exists()


def test_main_summary_ignores_removed_variants(tmp_path):
    _write_metrics(tmp_path, "removed_legacy_variant", 2024, 0.60)
    _write_metrics(tmp_path, "ablation_no_causal_tof", 2024, 0.75)
    rows = collect_per_seed_rows(tmp_path)
    assert rows == []


def test_main_summary_ignores_beta_diagnostics(tmp_path):
    _write_metrics(tmp_path, "full_causal", 2024, 0.80)
    _write_metrics(tmp_path, "full_causal_beta0", 2024, 0.10)
    rows = collect_per_seed_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["variant"] == "full_causal"
