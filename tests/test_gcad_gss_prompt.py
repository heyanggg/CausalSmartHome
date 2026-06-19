from causal_smart_home.causal_gss import (
    build_causal_hints_payload,
    format_causal_hints_for_prompt,
    map_device_edges,
)
from causal_smart_home.causal_prior import CausalPrior
from causal_smart_home.causal_prompt_adapter import enhance_prompt_with_causal_hints, render_prompt_diff


def test_device_gcad_hints_are_soft_and_named():
    prior = CausalPrior(
        matrix=[[0.0, 0.42], [0.0, 0.0]],
        channel_to_key=["d:13", "d:29"],
        lag=4,
        sparse_threshold=0.001,
    )
    edges = map_device_edges(prior, {13: "Light", 29: "Television"}, top_k_edges=20)
    payload = build_causal_hints_payload(edges, prior, source_train_pkl="train.pkl")
    prompt_text = format_causal_hints_for_prompt(payload)

    assert payload["constraint_strength"] == "soft"
    assert "common in the user's historical behavior" in payload["intro"]
    assert edges[0].source_name == "Light"
    assert edges[0].target_name == "Television"
    assert "soft constraints" in prompt_text
    assert "Light usually precedes Television" in prompt_text


def test_prompt_adapter_inserts_after_gss_without_removing_it():
    original = (
        "prefix User's behavior habits: {'Light': ['Television']} "
        "Your task: generate. Requirements: keep format."
    )
    result = enhance_prompt_with_causal_hints(original, "GCAD hints block")
    diff = render_prompt_diff(result.original_prompt, result.enhanced_prompt)

    assert "User's behavior habits" in result.enhanced_prompt
    assert result.enhanced_prompt.index("GCAD hints block") < result.enhanced_prompt.index("Your task:")
    assert "+GCAD hints block" in diff
