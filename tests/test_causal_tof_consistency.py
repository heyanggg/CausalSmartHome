import pytest

from causal_smart_home.causal.refinement.causal_tof import score_sequence_causal_tof


def test_causal_tof_uses_mean_ordered_pair_strength():
    sequence = [0, 0, 1, 1, 0, 1, 2, 1, 0, 2, 3, 1]
    edges = [
        {"source": "d:1", "target": "d:2", "weight": 1.0},
        {"source": "d:2", "target": "d:3", "weight": 0.5},
    ]
    score = score_sequence_causal_tof(sequence, edges, gamma_dist=0.0)
    assert score["causal_consistency_score"] == pytest.approx((1.0 + 0.0 + 0.5) / 3.0)
    assert score["causal_inconsistency"] == pytest.approx(0.5)
    assert score["final_score"] == pytest.approx(-0.5)
    assert score["violated_edges"] == []


def test_causal_tof_final_score_follows_requested_formula():
    sequence = [0, 0, 1, 1, 0, 1, 2, 1]
    score = score_sequence_causal_tof(
        sequence,
        [{"source": "d:1", "target": "d:2", "weight": 1.0}],
        reconstruction_loss=0.25,
        alpha_rec=2.0,
        beta_inconsistency=3.0,
        gamma_dist=0.0,
    )
    assert score["causal_inconsistency"] == 0.0
    assert score["final_score"] == pytest.approx(0.5)
