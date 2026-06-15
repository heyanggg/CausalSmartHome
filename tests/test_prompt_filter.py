import importlib.util

import pytest

from causal_smart_home.demo_data import make_toy_normal_sequences, make_toy_generated_candidates
from causal_smart_home.pipeline import CausalSmartHomePipeline
from causal_smart_home.causal_prompt import build_causal_smartgen_prompt
from causal_smart_home.causal_filter import CausalConsistencyFilter


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is required for causal prior training")


def test_prompt_contains_causal_hints_and_filter_scores(tmp_path):
    normal = make_toy_normal_sequences(25)
    pipeline = CausalSmartHomePipeline(tmp_path)
    prior = pipeline.build_prior(normal, lag=3, epochs=2)
    prompt = build_causal_smartgen_prompt(normal[:3], prior, {"devices": []}, "winter", "spring")
    assert "causal_hints" in prompt
    candidates = make_toy_generated_candidates()
    result = CausalConsistencyFilter(prior).filter(candidates, min_coverage=0.0)
    assert len(result.scores) == len(candidates)
