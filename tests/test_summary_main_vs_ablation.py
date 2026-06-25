import json

from causal_smart_home.experiment_matrix import ABLATION_VARIANT, PROPOSED_VARIANT
from scripts.summarize_main_experiment import main as summarize_main


def _write_metrics(root, variant):
    run = root / "sp_st" / "seed2024" / variant
    run.mkdir(parents=True)
    (run / "normalized_metrics.json").write_text(
        json.dumps(
            {
                "dataset": "sp",
                "scenario": "st",
                "seed": 2024,
                "variant": variant,
                "precision": 0.9,
                "recall": 1.0,
                "f1": 0.95,
                "accuracy": 0.9,
                "fpr": 0.1,
                "fnr": 0.0,
            }
        ),
        encoding="utf-8",
    )


def test_written_main_summary_excludes_ablation_baseline(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    out = tmp_path / "summary"
    _write_metrics(runs, PROPOSED_VARIANT)
    _write_metrics(runs, ABLATION_VARIANT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_main_experiment.py",
            "--runs-root",
            str(runs),
            "--out-dir",
            str(out),
            "--matrix",
            "all",
        ],
    )

    summarize_main()

    ablation_text = (out / "ablation_causal_tof.md").read_text(encoding="utf-8")
    main_rows = json.loads((out / "main_comparison_vs_gen.json").read_text(encoding="utf-8"))

    assert "original_gen_reference" in (out / "main_comparison_per_seed.md").read_text(encoding="utf-8")
    assert all("ablation_f1" not in row for row in main_rows)
    assert "ablation_no_causal_tof" in ablation_text
