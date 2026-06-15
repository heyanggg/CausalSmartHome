import csv
import pickle

from causal_smart_home.causal_prior import CausalPrior
from causal_smart_home.demo_data import make_toy_generated_candidates
from causal_smart_home.filter_sweep import build_filter_sweep_configs, run_filter_sweep


def _toy_prior() -> CausalPrior:
    keys = ["a:10", "a:11", "a:12", "a:13", "a:14", "a:15", "a:16"]
    matrix = [[0.0 for _ in keys] for _ in keys]
    for src, tgt, weight in [
        ("a:10", "a:11", 1.0),
        ("a:14", "a:15", 1.0),
        ("a:15", "a:16", 1.0),
    ]:
        matrix[keys.index(src)][keys.index(tgt)] = weight
    return CausalPrior(matrix=matrix, channel_to_key=keys, lag=3, sparse_threshold=0.0)


def test_filter_sweep_writes_summary_and_kept_pkls(tmp_path):
    configs = build_filter_sweep_configs(
        top_k_edges=[3],
        min_coverages=[0.8],
        min_checked_edges=[1, 3],
    )
    rows = run_filter_sweep(
        _toy_prior(),
        make_toy_generated_candidates(),
        configs,
        tmp_path,
        tag="toy",
        sequence_length=40,
    )

    assert len(rows) == 2
    assert all(row["raw"] == 5 for row in rows)
    assert all(row["rejected"] >= 1 for row in rows)
    assert (tmp_path / "filter_sweep_summary.csv").exists()
    assert (tmp_path / "filter_sweep_summary.json").exists()

    kept_path = tmp_path / "toy_k3_cov0p8_chk1_kept.pkl"
    assert kept_path.exists()
    with open(kept_path, "rb") as f:
        kept = pickle.load(f)
    assert len(kept) == rows[0]["kept"]
    assert {len(seq) for seq in kept} == {40}

    with open(tmp_path / "filter_sweep_summary.csv", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["slug"] == "k3_cov0p8_chk1"
    assert csv_rows[0]["sequence_length"] == "40"
