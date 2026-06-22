import json

from causal_smart_home.causal_gss_reweight import build_device_transition_graph, reweight_gss_edges


def test_build_device_transition_graph_counts_device_edges():
    graph = build_device_transition_graph([[0, 0, 1, 10, 0, 1, 2, 20, 0, 2, 2, 21]])
    pairs = {(edge["source_device"], edge["target_device"]): edge for edge in graph["edges"]}
    assert pairs[(1, 2)]["count"] == 1
    assert pairs[(2, 2)]["count"] == 1


def test_gcad_edge_changes_final_score_and_modes_differ():
    transitions = [
        {"source_device": 1, "target_device": 2, "transition_score": 0.2, "count": 2},
        {"source_device": 1, "target_device": 3, "transition_score": 0.4, "count": 4},
    ]
    causal = [{"source": "d:1", "target": "d:2", "raw_weight": 0.8, "guarded_weight": 0.8, "guard_action": "keep"}]
    additive = reweight_gss_edges(transitions, causal, lambda_causal=1.0, mode="additive", top_k=10)
    multiplicative = reweight_gss_edges(transitions, causal, lambda_causal=1.0, mode="multiplicative", top_k=10)
    add_12 = next(edge for edge in additive["edges"] if edge["target_device"] == 2)
    add_13 = next(edge for edge in additive["edges"] if edge["target_device"] == 3)
    mult_12 = next(edge for edge in multiplicative["edges"] if edge["target_device"] == 2)
    assert add_12["final_score"] > add_13["final_score"]
    assert add_12["final_score"] != mult_12["final_score"]
    json.dumps(additive)
