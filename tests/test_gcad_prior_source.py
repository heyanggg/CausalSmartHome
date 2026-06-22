import importlib.util
import json
import pickle

import pytest

from causal_smart_home.causal_prior import CausalPrior
from causal_smart_home.demo_data import make_toy_normal_sequences
from causal_smart_home.gcad_prior_source import resolve_gcad_prior


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch needed for adapter fallback")


def test_resolve_gcad_prior_reads_existing_json(tmp_path):
    prior = CausalPrior(matrix=[[0.0, 0.5], [0.0, 0.0]], channel_to_key=["d:1", "d:2"], lag=4, sparse_threshold=0.001)
    path = tmp_path / "causal_prior.json"
    prior.save(path)

    resolved = resolve_gcad_prior(prior_json=str(path), out_dir=str(tmp_path / "out"), level="device")

    assert resolved.channels == ["d:1", "d:2"]
    assert resolved.top_causal_edges[0]["source"] == "d:1"
    assert resolved.top_causal_edges[0]["target"] == "d:2"
    assert (tmp_path / "out" / "resolved_gcad_prior.json").exists()
    json.dumps(resolved.to_json_dict())


def test_resolve_gcad_prior_without_prior_uses_existing_adapter_fallback(tmp_path, monkeypatch):
    seqs = [seq.to_flat_numeric() for seq in make_toy_normal_sequences(2)]
    source = tmp_path / "source.pkl"
    with open(source, "wb") as f:
        pickle.dump(seqs, f)

    calls = {"count": 0}

    def fake_mine_event_prior(self, tensor, channel_to_key, **kwargs):
        calls["count"] += 1
        return CausalPrior(
            matrix=[[0.0, 1.0], [0.0, 0.0]],
            channel_to_key=list(channel_to_key)[:2],
            lag=kwargs.get("lag", 2),
            sparse_threshold=kwargs.get("sparse_threshold", 0.0),
        )

    monkeypatch.setattr("causal_smart_home.gcad_prior_source.GCADAdapter.mine_event_prior", fake_mine_event_prior)
    resolved = resolve_gcad_prior(source_pkl=str(source), out_dir=str(tmp_path / "out"), level="device", lag=2, sparse_threshold=0.0)

    assert calls["count"] == 1
    assert resolved.gcad_source == "existing_adapter_compact_fallback"
    assert resolved.config["gcad_source"] == "existing_adapter_compact_fallback"
    assert len(resolved.channels) == 2
    assert (tmp_path / "out" / "resolved_gcad_prior.json").exists()
