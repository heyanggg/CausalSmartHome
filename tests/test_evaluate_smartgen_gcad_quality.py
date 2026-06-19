import importlib.util
from pathlib import Path

from causal_smart_home.schema import BehaviorEvent, BehaviorSequence


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_smartgen_gcad_quality.py"
SPEC = importlib.util.spec_from_file_location("evaluate_smartgen_gcad_quality", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_score_sequence_tracks_satisfied_and_violated_edges():
    seq = BehaviorSequence(
        [
            BehaviorEvent(0, 0, 1, 10),
            BehaviorEvent(0, 1, 2, 20),
            BehaviorEvent(0, 2, 3, 30),
        ]
    )
    edges = [
        {"source": "a:10", "target": "a:20", "weight": 2.0, "lag": 4},
        {"source": "a:30", "target": "a:20", "weight": 1.0, "lag": 4},
        {"source": "a:999", "target": "a:20", "weight": 8.0, "lag": 4},
    ]

    score = module.score_sequence(seq, edges, total_edge_weight=11.0, low_evidence_weight=0.1, index=0)

    assert score["checked_edge_weight"] == 3.0
    assert score["satisfied_edge_weight"] == 2.0
    assert score["violated_edge_weight"] == 1.0
    assert round(score["causal_coverage"], 4) == 0.6667
    assert round(score["violation_rate"], 4) == 0.3333
    assert score["low_evidence"] is False


def test_mild_filter_keeps_low_evidence_sequences():
    seqs = [
        BehaviorSequence([BehaviorEvent(0, 0, 1, 10)]),
        BehaviorSequence([BehaviorEvent(0, 0, 2, 20)]),
    ]
    scores = [
        {"causal_score": 0.0, "low_evidence": True},
        {"causal_score": 0.1, "low_evidence": False},
    ]

    kept, summary = module.apply_mode("mild_filter", seqs, scores, min_coverage=0.3)

    assert kept == [seqs[0]]
    assert summary["kept_rate"] == 0.5


def test_exclude_ids_is_level_aware():
    seq = BehaviorSequence(
        [
            BehaviorEvent(0, 0, 0, 10),
            BehaviorEvent(0, 1, 1, 193),
        ],
        sequence_id="s0",
    )

    action_filtered = module.filter_sequences_for_level([seq], "action", {193})
    device_filtered = module.filter_sequences_for_level([seq], "device", {0})
    device_action_filtered = module.filter_sequences_for_level([seq], "device_action", {0, 193})

    assert [ev.action for ev in action_filtered[0]] == [10]
    assert [ev.device for ev in device_filtered[0]] == [1]
    assert len(device_action_filtered[0]) == 0


def test_resample_soft_keeps_low_evidence_once():
    seqs = [
        BehaviorSequence([BehaviorEvent(0, 0, 1, 10)]),
        BehaviorSequence([BehaviorEvent(0, 0, 2, 20)]),
        BehaviorSequence([BehaviorEvent(0, 0, 3, 30)]),
    ]
    scores = [
        {"causal_score": 0.8, "low_evidence": False},
        {"causal_score": 0.8, "low_evidence": True},
        {"causal_score": 0.1, "low_evidence": False},
    ]

    resampled, summary = module.apply_mode("resample_soft", seqs, scores, min_coverage=0.3)

    assert resampled == [seqs[0], seqs[0], seqs[1]]
    assert summary["resampled_size"] == 3
    assert summary["num_duplicated"] == 1
    assert summary["num_dropped"] == 1
