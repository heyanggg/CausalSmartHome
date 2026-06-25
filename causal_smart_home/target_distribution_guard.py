from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import BehaviorSequence, load_numeric_sequences


@dataclass
class TargetDistributionGuardConfig:
    max_overuse_ratio: float = 1.25
    min_target_freq: float = 0.001
    eps: float = 1e-8
    mode: str = "suppress"  # suppress / downweight
    downweight_factor: float = 0.25
    endpoint_policy: str = "target"  # target / source_or_target / both

    def __post_init__(self) -> None:
        if self.mode not in {"suppress", "downweight"}:
            raise ValueError("mode must be suppress or downweight")
        if self.endpoint_policy not in {"target", "source_or_target", "both"}:
            raise ValueError("endpoint_policy must be target, source_or_target, or both")
        if self.max_overuse_ratio <= 0:
            raise ValueError("max_overuse_ratio must be positive")
        if not (0 <= self.downweight_factor <= 1):
            raise ValueError("downweight_factor must be between 0 and 1")


def compute_device_distribution(sequences) -> dict[str, float]:
    """Compute device-frequency distribution from Gen sequences.

    Input may be BehaviorSequence objects, flattened numeric lists, numpy rows, or
    tuple-wrapped rows from labeled datasets.  Output keys are canonical ``d:<id>``
    strings so they align with device-level causal relation channels.
    """

    behavior_sequences = _coerce_sequences(sequences)
    counts: dict[str, int] = {}
    for seq in behavior_sequences:
        for event in seq:
            key = event.key("device")
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def apply_target_distribution_guard(
    causal_edges: list[dict],
    generated_or_prompt_distribution: dict,
    target_distribution: dict,
    config: TargetDistributionGuardConfig,
) -> tuple[list[dict], dict]:
    """Suppress or downweight causal edges whose endpoints are overused.

    The guard is not a new causal discovery method.  It protects a source-context
    causal relation prior before moving it into target-context Gen guidance.
    """

    observed = _normalize_distribution(generated_or_prompt_distribution)
    target = _normalize_distribution(target_distribution)
    guarded_edges: list[dict[str, Any]] = []
    report_edges: list[dict[str, Any]] = []
    overused_devices: dict[str, dict[str, Any]] = {}

    for edge in causal_edges:
        source_key = _edge_endpoint_key(edge, "source")
        target_key = _edge_endpoint_key(edge, "target")
        raw_weight = float(edge.get("raw_weight", edge.get("weight", edge.get("guarded_weight", 0.0))))
        source_status = _endpoint_status(source_key, observed, target, config)
        target_status = _endpoint_status(target_key, observed, target, config)
        should_guard = _should_guard(source_status["overused"], target_status["overused"], config.endpoint_policy)
        for status in (source_status, target_status):
            if status["overused"]:
                overused_devices[status["device_key"]] = status
        guard_action = "keep"
        guarded_weight = raw_weight
        guard_reason = ""
        if should_guard:
            if config.mode == "suppress":
                guard_action = "suppress"
                guarded_weight = 0.0
            else:
                guard_action = "downweight"
                guarded_weight = raw_weight * config.downweight_factor
            reasons = []
            if config.endpoint_policy in {"source_or_target", "both"} and source_status["overused"]:
                reasons.append(_status_reason("source", edge, source_status))
            if target_status["overused"]:
                reasons.append(_status_reason("target", edge, target_status))
            if not reasons and source_status["overused"]:
                reasons.append(_status_reason("source", edge, source_status))
            guard_reason = "; ".join(reasons)

        guarded = dict(edge)
        guarded.update(
            {
                "raw_weight": raw_weight,
                "guarded_weight": float(guarded_weight),
                "weight": float(guarded_weight),
                "guard_action": guard_action,
                "guard_reason": guard_reason,
                "source_device_key": source_key,
                "target_device_key": target_key,
                "source_overused": bool(source_status["overused"]),
                "target_overused": bool(target_status["overused"]),
                "source_observed_freq": float(source_status["observed_freq"]),
                "source_target_freq": float(source_status["target_freq"]),
                "source_overuse_ratio": float(source_status["ratio"]),
                "target_observed_freq": float(target_status["observed_freq"]),
                "target_target_freq": float(target_status["target_freq"]),
                "target_overuse_ratio": float(target_status["ratio"]),
            }
        )
        guarded_edges.append(guarded)
        report_edges.append(dict(guarded))

    report = {
        "config": asdict(config),
        "overused_devices": sorted(overused_devices.values(), key=lambda item: item["ratio"], reverse=True),
        "num_edges": len(causal_edges),
        "num_suppressed_edges": sum(1 for edge in guarded_edges if edge.get("guard_action") == "suppress"),
        "num_downweighted_edges": sum(1 for edge in guarded_edges if edge.get("guard_action") == "downweight"),
        "edges": report_edges,
    }
    return guarded_edges, report


def _should_guard(source_overused: bool, target_overused: bool, endpoint_policy: str) -> bool:
    if endpoint_policy == "target":
        return target_overused
    if endpoint_policy == "source_or_target":
        return source_overused or target_overused
    if endpoint_policy == "both":
        return source_overused and target_overused
    raise ValueError(f"unknown endpoint policy: {endpoint_policy}")


def _endpoint_status(device_key: str, observed: Mapping[str, float], target: Mapping[str, float], config: TargetDistributionGuardConfig) -> dict[str, Any]:
    observed_freq = float(observed.get(device_key, 0.0))
    raw_target_freq = float(target.get(device_key, 0.0))
    denom = max(raw_target_freq, config.min_target_freq, config.eps)
    ratio = observed_freq / denom
    return {
        "device_key": device_key,
        "device_name": _device_name_from_key(device_key),
        "observed_freq": observed_freq,
        "target_freq": raw_target_freq,
        "ratio": ratio,
        "overused": ratio > config.max_overuse_ratio,
    }


def _status_reason(role: str, edge: Mapping[str, Any], status: Mapping[str, Any]) -> str:
    name_field = f"{role}_name"
    name = str(edge.get(name_field) or status.get("device_name") or status["device_key"])
    return (
        f"{role} endpoint {name} overused: "
        f"observed_freq={float(status['observed_freq']):.6f}, "
        f"target_freq={float(status['target_freq']):.6f}, "
        f"ratio={float(status['ratio']):.3f}"
    )


def _normalize_distribution(distribution: Mapping[Any, Any] | None) -> dict[str, float]:
    if not distribution:
        return {}
    out: dict[str, float] = {}
    for key, value in distribution.items():
        out[_canonical_device_key(key)] = float(value)
    return out


def _edge_endpoint_key(edge: Mapping[str, Any], role: str) -> str:
    # Prefer explicit device fields, then d:<id> channel strings.  If a row only
    # has matrix indices, the caller should provide d:<device> in source/target.
    for field in (f"{role}_device", f"{role}_device_id", f"{role}_device_key"):
        if field in edge and edge[field] is not None:
            return _canonical_device_key(edge[field])
    value = edge.get(role)
    if value is not None:
        return _canonical_device_key(value)
    index_value = edge.get(f"{role}_id", edge.get(f"{role}_index"))
    return _canonical_device_key(index_value if index_value is not None else "unknown")


def _canonical_device_key(value: Any) -> str:
    text = str(value)
    if text.startswith("d:"):
        return text
    if text.startswith("device_"):
        suffix = text[len("device_") :]
        if suffix.isdigit():
            return f"d:{int(suffix)}"
    if text.isdigit():
        return f"d:{int(text)}"
    if text.startswith("a:"):
        # Action-level edges are not valid for a device guard; keep as-is so the
        # report exposes the mismatch rather than silently converting it.
        return text
    return text


def _device_name_from_key(key: str) -> str:
    if key.startswith("d:"):
        return f"device_{key[2:]}"
    return key


def _coerce_sequences(sequences) -> list[BehaviorSequence]:
    if sequences is None:
        return []
    if isinstance(sequences, BehaviorSequence):
        return [sequences]
    if isinstance(sequences, list) and all(isinstance(item, BehaviorSequence) for item in sequences):
        return list(sequences)
    raw_items = list(sequences)
    flats: list[list[int]] = []
    for item in raw_items:
        candidate = _extract_flat(item)
        if candidate is not None:
            flats.append(candidate)
    return load_numeric_sequences(flats)


def _extract_flat(item: Any) -> list[int] | None:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, tuple):
        for part in item:
            candidate = _extract_flat(part)
            if candidate is not None:
                return candidate
        return None
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) > 0 and len(item) % 4 == 0:
        try:
            return [int(value) for value in item]
        except Exception:
            return None
    return None
