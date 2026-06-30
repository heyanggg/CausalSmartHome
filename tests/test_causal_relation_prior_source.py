"""测试从 JSON、矩阵和 pkl 解析 causal-relation prior。"""

import importlib.util
import json
import pickle

import pytest

from causal_smart_home.causal_prior import CausalPrior
from causal_smart_home.causal_relation_prior_source import resolve_causal_relation_prior


def make_toy_normal_sequences(num_sequences=20, seed=2024, sequence_length=10):
    """测试用本地 toy 扁平四元组序列生成器。

    返回 Gen 风格扁平序列：
    [day, hour_slot, device_id, action_id, ...]
    """
    import random

    rng = random.Random(seed)
    seqs = []
    for i in range(num_sequences):
        seq = []
        day = i % 7
        for t in range(sequence_length):
            hour_slot = t % 8
            # 使用确定但非恒定的 device/action 模式，方便测试 prior 挖掘和排序。
            device = (t + i) % 4 + 1
            action = device * 10 + (t % 3)
            if rng.random() < 0.15:
                device = (device % 4) + 1
                action = device * 10 + ((t + 1) % 3)
            seq.extend([day, hour_slot, device, action])
        seqs.append(seq)
    return seqs


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch needed for adapter fallback")


def test_resolve_causal_relation_prior_reads_existing_json(tmp_path):
    prior = CausalPrior(matrix=[[0.0, 0.5], [0.0, 0.0]], channel_to_key=["d:1", "d:2"], lag=4, sparse_threshold=0.001)
    path = tmp_path / "causal_prior.json"
    prior.save(path)

    resolved = resolve_causal_relation_prior(prior_json=str(path), out_dir=str(tmp_path / "out"), level="device")

    assert resolved.channels == ["d:1", "d:2"]
    assert resolved.top_causal_edges[0]["source"] == "d:1"
    assert resolved.top_causal_edges[0]["target"] == "d:2"
    assert (tmp_path / "out" / "resolved_causal_relation_prior.json").exists()
    json.dumps(resolved.to_json_dict())


def test_resolve_causal_relation_prior_without_prior_uses_existing_adapter_fallback(tmp_path, monkeypatch):
    seqs = make_toy_normal_sequences(2)
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

    monkeypatch.setattr("causal_smart_home.causal_relation_prior_source.CausalRelationAdapter.mine_event_prior", fake_mine_event_prior)
    resolved = resolve_causal_relation_prior(source_pkl=str(source), out_dir=str(tmp_path / "out"), level="device", lag=2, sparse_threshold=0.0)

    assert calls["count"] == 1
    assert resolved.causal_relation_source == "existing_adapter_compact_fallback"
    assert resolved.config["causal_relation_source"] == "existing_adapter_compact_fallback"
    assert len(resolved.channels) == 2
    assert (tmp_path / "out" / "resolved_causal_relation_prior.json").exists()
