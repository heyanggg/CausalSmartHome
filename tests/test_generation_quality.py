import pytest

from causal_smart_home.causal.evaluation.generation_quality import evaluate_generation_quality
from causal_smart_home.schema import load_numeric_sequences


def test_identical_corpora_have_ideal_generation_quality():
    sequences = load_numeric_sequences([[0, 0, 1, 1, 0, 1, 2, 1], [1, 0, 2, 1, 1, 1, 1, 1]])
    summary, case_study = evaluate_generation_quality(sequences, sequences, variant="full_causal", seed=2024)
    assert summary["device_distribution_kl"] == pytest.approx(0.0)
    assert summary["transition_matrix_similarity"] == pytest.approx(1.0)
    assert summary["causal_graph_similarity"] == pytest.approx(1.0)
    assert case_study["real_target_normal"]["transition_matrix"] == case_study["synthetic"]["transition_matrix"]
