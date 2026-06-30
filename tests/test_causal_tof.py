"""测试 Causal-TOF 序列评分和加权重采样。"""

import json

from causal_smart_home.causal_tof import score_sequence_causal_tof, weighted_resample_sequences


def test_causal_tof_penalizes_violated_edge_and_lowers_weight():
    edges = [{"source_device": 1, "target_device": 2, "guarded_causal_strength": 1.0, "source_name": "Light", "target_name": "Television"}]
    satisfied = [0, 0, 1, 10, 0, 1, 2, 20]
    violated = [0, 0, 2, 20, 0, 1, 1, 10]
    ok_score = score_sequence_causal_tof(satisfied, edges, temperature=2.0)
    bad_score = score_sequence_causal_tof(violated, edges, temperature=2.0)
    assert ok_score["final_score"] < bad_score["final_score"]
    assert ok_score["sample_weight"] > bad_score["sample_weight"]
    assert bad_score["violated_edges"]
    json.dumps(bad_score)


def test_downweighted_edges_are_diagnostic_by_default():
    edges = [
        {
            "source_device": 1,
            "target_device": 2,
            "guarded_causal_strength": 1.0,
            "guard_action": "downweight",
            "source_name": "Light",
            "target_name": "Television",
        }
    ]
    violated = [0, 0, 2, 20, 0, 1, 1, 10]

    score = score_sequence_causal_tof(violated, edges, temperature=2.0)
    penalized_score = score_sequence_causal_tof(violated, edges, temperature=2.0, penalize_downweighted_edges=True)

    assert score["causal_violation"] == 0.0
    assert score["observed_causal_violation_all_guarded_edges"] == 1.0
    assert score["final_score"] < penalized_score["final_score"]


def test_weighted_resample_limits_copies():
    seqs = [[0, 0, 1, 10], [0, 0, 2, 20]]
    scores = [{"sample_weight": 1.0}, {"sample_weight": 0.01}]
    resampled, config = weighted_resample_sequences(seqs, scores, seed=1, max_copies=2, target_size=4)
    assert len(resampled) <= 4
    assert max(config["copy_counts"]) <= 2
