import json

from causal_smart_home.experiment_matrix import SEEDS
from scripts.run_main_experiment_matrix import write_generation_package_index


def test_generation_package_index_marks_data_ready_packages(tmp_path):
    expected_cells = {
        ("fr", "st"),
        ("fr", "tt"),
        ("fr", "nt"),
        ("sp", "tt"),
    }
    for dataset, scenario in expected_cells:
        for seed in SEEDS:
            package_dir = tmp_path / "gpt55_generation_packages" / dataset / scenario / f"seed{seed}"
            package_dir.mkdir(parents=True)
            (package_dir / "generation_schema.json").write_text("{}", encoding="utf-8")
            (package_dir / "generation_instruction.md").write_text("instructions\n", encoding="utf-8")

    write_generation_package_index(tmp_path, rows=[])

    index = json.loads((tmp_path / "gpt55_generation_packages" / "generation_package_index.json").read_text())
    ready = {
        (row["dataset"], row["scenario"], row["seed"])
        for row in index
        if row["status"] == "PACKAGE_READY"
    }
    expected_ready = {
        (dataset, scenario, seed)
        for dataset, scenario in expected_cells
        for seed in SEEDS
    }

    assert expected_ready <= ready
    assert len(ready) == len(expected_ready)
    readme = (tmp_path / "gpt55_generation_packages" / "README_SEND_TO_GPT55.md").read_text()
    assert "generated_gpt55_clean.jsonl" in readme
