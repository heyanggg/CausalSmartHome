import csv
import pickle

from causal_smart_home.smartguard_experiment import (
    aggregate_attack_results,
    merge_training_sequences,
    resolve_sweep_rows,
)


def _dump(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_merge_training_sequences_normalizes_added_data(tmp_path):
    base = tmp_path / "base.pkl"
    added = tmp_path / "added.pkl"
    out = tmp_path / "merged.pkl"
    _dump(base, [[1] * 40, [2] * 40])
    _dump(added, [[3] * 8, [4] * 44])

    info = merge_training_sequences(base, added, out, sequence_length=40, pad_value=0)

    with open(out, "rb") as f:
        merged = pickle.load(f)
    assert info["base_size"] == 2
    assert info["added_size"] == 2
    assert info["merged_size"] == 4
    assert {len(seq) for seq in merged} == {40}
    assert merged[2][8:] == [0] * 32
    assert merged[3] == [4] * 40


def test_resolve_sweep_rows_selects_slugs_and_paths(tmp_path):
    kept = tmp_path / "kept.pkl"
    kept.write_bytes(b"placeholder")
    summary = tmp_path / "summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slug", "kept_path"])
        writer.writeheader()
        writer.writerow({"slug": "a", "kept_path": str(kept)})
        writer.writerow({"slug": "b", "kept_path": str(tmp_path / "other.pkl")})

    rows = resolve_sweep_rows(summary, slugs=["a"])

    assert len(rows) == 1
    assert rows[0]["slug"] == "a"
    assert rows[0]["kept_path"] == str(kept)


def test_aggregate_attack_results_from_confusion_counts():
    aggregate = aggregate_attack_results(
        [
            {"TP": 8, "TN": 10, "FP": 2, "FN": 0},
            {"TP": 2, "TN": 8, "FP": 2, "FN": 2},
        ]
    )

    assert aggregate["TP"] == 10
    assert aggregate["TN"] == 18
    assert aggregate["FP"] == 4
    assert aggregate["FN"] == 2
    assert round(aggregate["recall"], 4) == 0.8333
    assert round(aggregate["precision"], 4) == 0.7143
    assert round(aggregate["f1_score"], 4) == 0.7692
