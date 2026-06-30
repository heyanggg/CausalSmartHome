"""测试对源上下文因果边进行 target-distribution guard。"""

import json

from causal_smart_home.target_distribution_guard import TargetDistributionGuardConfig, apply_target_distribution_guard, compute_device_distribution


def test_compute_device_distribution_from_flat_sequences():
    dist = compute_device_distribution([[0, 0, 1, 10, 0, 1, 2, 20], [0, 0, 2, 20, 0, 1, 2, 21]])
    assert round(dist["d:2"], 2) == 0.75


def test_suppress_guard_marks_television_overuse():
    edges = [{"source": "d:1", "target": "d:2", "source_name": "Light", "target_name": "Television", "weight": 0.8}]
    guarded, report = apply_target_distribution_guard(
        edges,
        generated_or_prompt_distribution={"d:2": 0.5, "d:1": 0.5},
        target_distribution={"d:2": 0.1, "d:1": 0.9},
        config=TargetDistributionGuardConfig(mode="suppress", max_overuse_ratio=1.25, endpoint_policy="target"),
    )
    assert guarded[0]["guarded_weight"] == 0.0
    assert guarded[0]["guard_action"] == "suppress"
    assert "Television" in guarded[0]["guard_reason"]
    assert report["num_suppressed_edges"] == 1
    json.dumps(report)


def test_downweight_guard_applies_factor():
    edges = [{"source": "d:1", "target": "d:2", "weight": 0.8}]
    guarded, _ = apply_target_distribution_guard(
        edges,
        generated_or_prompt_distribution={"d:1": 0.5, "d:2": 0.5},
        target_distribution={"d:1": 0.1, "d:2": 0.1},
        config=TargetDistributionGuardConfig(mode="downweight", downweight_factor=0.25, endpoint_policy="source_or_target"),
    )
    assert guarded[0]["guarded_weight"] == 0.2
    assert guarded[0]["guard_action"] == "downweight"
