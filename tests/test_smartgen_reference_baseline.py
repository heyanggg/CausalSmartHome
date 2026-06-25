from pathlib import Path

from causal_smart_home.experiment_matrix import load_reference_baseline


REFERENCE_JSON = Path("causal_smart_home/resources/reference/smartgen_table3_ad.json")


def test_smartgen_reference_contains_all_9_cells():
    rows = load_reference_baseline(REFERENCE_JSON)

    assert len(rows) == 9
    assert set(rows) == {
        (dataset, scenario)
        for dataset in ("fr", "sp", "us")
        for scenario in ("st", "tt", "nt")
    }
    assert all(row["source"] == "SmartGen paper Table 3, SmartGen column" for row in rows.values())


def test_smartgen_reference_known_values():
    rows = load_reference_baseline(REFERENCE_JSON)

    assert rows[("sp", "st")]["precision"] == 0.8573
    assert rows[("sp", "st")]["recall"] == 0.9904
    assert rows[("sp", "st")]["f1"] == 0.9191

    assert rows[("fr", "tt")]["precision"] == 0.9416
    assert rows[("fr", "tt")]["recall"] == 1.0
    assert rows[("fr", "tt")]["f1"] == 0.9699
