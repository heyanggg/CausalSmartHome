from causal_smart_home.causal.adaptation.target_guard import adapt_causal_prior_to_target


def test_target_aware_edge_weighting_and_statistics():
    edges = [{"source": "d:1", "target": "d:2", "weight": 0.8}]
    adapted, report = adapt_causal_prior_to_target(edges, {"d:1": 0.25, "d:2": 0.75})
    assert adapted[0]["target_adapted_weight"] == 0.8 * 0.25 * 0.75
    assert report["formula"] == "C_target(i,j)=C_source(i,j)*P_target(i)*P_target(j)"
    assert report["edge_statistics"]["before"]["mean_strength"] == 0.8
    assert report["edge_statistics"]["after"]["mean_strength"] == adapted[0]["weight"]
