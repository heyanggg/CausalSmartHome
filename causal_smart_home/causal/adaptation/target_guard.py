"""Target-aware adaptation of a source-context causal prior.

For every directed edge ``i -> j`` the adapted strength is

``C_target(i, j) = C_source(i, j) * P_target(i) * P_target(j)``.

``P_target`` is the event-level device distribution of target normal behavior.
The earlier over-use guard remains available as a compatibility operation and
can be applied after this deterministic weighting step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from ...schema import BehaviorSequence, load_numeric_sequences
from ..evaluation.causal_metrics import before_after_edge_statistics


@dataclass
class TargetDistributionGuardConfig:
    max_overuse_ratio: float = 1.25
    min_target_freq: float = 0.001
    eps: float = 1e-8
    mode: str = "suppress"
    downweight_factor: float = 0.25
    endpoint_policy: str = "target"

    def __post_init__(self) -> None:
        if self.mode not in {"suppress", "downweight"}:
            raise ValueError("mode must be suppress or downweight")
        if self.endpoint_policy not in {"target", "source_or_target", "both"}:
            raise ValueError("endpoint_policy must be target, source_or_target, or both")
        if self.max_overuse_ratio <= 0:
            raise ValueError("max_overuse_ratio must be positive")
        if not 0 <= self.downweight_factor <= 1:
            raise ValueError("downweight_factor must be between 0 and 1")


def compute_device_distribution(sequences) -> dict[str, float]:
    """Compute event-level ``d:<id>`` probabilities from Gen sequences."""
    counts: dict[str, int] = {}
    for sequence in _coerce_sequences(sequences):
        for event in sequence:
            key = event.key("device")
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def adapt_causal_prior_to_target(
    causal_edges: Sequence[Mapping[str, Any]],
    target_distribution: Mapping[Any, Any],
    *,
    matrix: Sequence[Sequence[float]] | None = None,
    channels: Sequence[str] | None = None,
    threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply target-aware endpoint weighting and return an auditable payload."""
    target = _normalize_distribution(target_distribution)
    adapted: list[dict[str, Any]] = []
    for edge in causal_edges:
        row = dict(edge)
        source_key = _edge_endpoint_key(row, "source")
        target_key = _edge_endpoint_key(row, "target")
        source_probability = float(target.get(source_key, 0.0))
        target_probability = float(target.get(target_key, 0.0))
        raw_weight = _edge_raw_weight(row)
        adapted_weight = raw_weight * source_probability * target_probability
        row.update(
            {
                "source_device_key": source_key,
                "target_device_key": target_key,
                "raw_weight": float(raw_weight),
                "source_target_probability": source_probability,
                "target_target_probability": target_probability,
                "target_endpoint_weight": source_probability * target_probability,
                "target_adapted_weight": float(adapted_weight),
                "weight": float(adapted_weight),
                "adaptation_formula": "C_source(i,j) * P_target(i) * P_target(j)",
            }
        )
        adapted.append(row)

    top_edge_stats = before_after_edge_statistics(causal_edges, adapted, threshold=threshold)
    adapted_matrix = None
    matrix_stats = None
    if matrix is not None:
        if channels is None or len(channels) != len(matrix):
            raise ValueError("channels must match the causal matrix size")
        if any(len(row) != len(channels) for row in matrix):
            raise ValueError("causal matrix must be square")
        adapted_matrix = []
        before_matrix_edges = []
        after_matrix_edges = []
        for source_index, row in enumerate(matrix):
            adapted_row = []
            for target_index, value in enumerate(row):
                source_probability = float(target.get(str(channels[source_index]), 0.0))
                target_probability = float(target.get(str(channels[target_index]), 0.0))
                adapted_value = float(value) * source_probability * target_probability
                adapted_row.append(adapted_value)
                if source_index != target_index:
                    before_matrix_edges.append({"weight": float(value)})
                    after_matrix_edges.append({"weight": adapted_value})
            adapted_matrix.append(adapted_row)
        matrix_stats = before_after_edge_statistics(before_matrix_edges, after_matrix_edges, threshold=threshold)
    report = {
        "schema_version": 1,
        "method": "target_aware_endpoint_probability_weighting",
        "formula": "C_target(i,j)=C_source(i,j)*P_target(i)*P_target(j)",
        "target_distribution": target,
        "channels": list(channels) if channels is not None else None,
        "matrix": adapted_matrix,
        "edge_statistics": matrix_stats or top_edge_stats,
        "top_edge_statistics": top_edge_stats,
        "edges": adapted,
    }
    return adapted, report


def apply_target_distribution_guard(
    causal_edges: list[dict],
    generated_or_prompt_distribution: Mapping[Any, Any],
    target_distribution: Mapping[Any, Any],
    config: TargetDistributionGuardConfig,
) -> tuple[list[dict], dict]:
    """Preserve the historical over-use guard as a post-adaptation option."""
    observed = _normalize_distribution(generated_or_prompt_distribution)
    target = _normalize_distribution(target_distribution)
    guarded_edges: list[dict[str, Any]] = []
    overused: dict[str, dict[str, Any]] = {}

    for edge in causal_edges:
        source_key = _edge_endpoint_key(edge, "source")
        target_key = _edge_endpoint_key(edge, "target")
        raw_weight = _edge_raw_weight(edge)
        source_status = _endpoint_status(source_key, observed, target, config)
        target_status = _endpoint_status(target_key, observed, target, config)
        should_guard = _should_guard(source_status["overused"], target_status["overused"], config.endpoint_policy)
        for status in (source_status, target_status):
            if status["overused"]:
                overused[status["device_key"]] = status

        action = "keep"
        guarded_weight = raw_weight
        if should_guard and config.mode == "suppress":
            action, guarded_weight = "suppress", 0.0
        elif should_guard:
            action, guarded_weight = "downweight", raw_weight * config.downweight_factor

        row = dict(edge)
        row.update(
            {
                "raw_weight": raw_weight,
                "guarded_weight": float(guarded_weight),
                "weight": float(guarded_weight),
                "guard_action": action,
                "guard_reason": _guard_reason(source_status, target_status, config.endpoint_policy) if should_guard else "",
                "source_device_key": source_key,
                "target_device_key": target_key,
                "source_overused": bool(source_status["overused"]),
                "target_overused": bool(target_status["overused"]),
                "source_observed_freq": source_status["observed_freq"],
                "source_target_freq": source_status["target_freq"],
                "source_overuse_ratio": source_status["ratio"],
                "target_observed_freq": target_status["observed_freq"],
                "target_target_freq": target_status["target_freq"],
                "target_overuse_ratio": target_status["ratio"],
            }
        )
        guarded_edges.append(row)

    return guarded_edges, {
        "config": asdict(config),
        "overused_devices": sorted(overused.values(), key=lambda item: item["ratio"], reverse=True),
        "num_edges": len(causal_edges),
        "num_suppressed_edges": sum(edge["guard_action"] == "suppress" for edge in guarded_edges),
        "num_downweighted_edges": sum(edge["guard_action"] == "downweight" for edge in guarded_edges),
        "edges": guarded_edges,
    }


def _edge_raw_weight(edge: Mapping[str, Any]) -> float:
    for key in ("target_adapted_weight", "guarded_weight", "weight", "raw_weight", "causal_strength"):
        if edge.get(key) is not None:
            return float(edge[key])
    return 0.0


def _should_guard(source: bool, target: bool, policy: str) -> bool:
    if policy == "target":
        return target
    if policy == "source_or_target":
        return source or target
    return source and target


def _endpoint_status(
    key: str,
    observed: Mapping[str, float],
    target: Mapping[str, float],
    config: TargetDistributionGuardConfig,
) -> dict[str, Any]:
    observed_frequency = float(observed.get(key, 0.0))
    target_frequency = float(target.get(key, 0.0))
    ratio = observed_frequency / max(target_frequency, config.min_target_freq, config.eps)
    return {
        "device_key": key,
        "device_name": f"device_{key[2:]}" if key.startswith("d:") else key,
        "observed_freq": observed_frequency,
        "target_freq": target_frequency,
        "ratio": ratio,
        "overused": ratio > config.max_overuse_ratio,
    }


def _guard_reason(source: Mapping[str, Any], target: Mapping[str, Any], policy: str) -> str:
    statuses = [target] if policy == "target" else [source, target]
    return "; ".join(
        f"{status['device_key']} overused: observed_freq={status['observed_freq']:.6f}, "
        f"target_freq={status['target_freq']:.6f}, ratio={status['ratio']:.3f}"
        for status in statuses
        if status["overused"]
    )


def _normalize_distribution(distribution: Mapping[Any, Any] | None) -> dict[str, float]:
    if not distribution:
        return {}
    values = {_canonical_device_key(key): max(float(value), 0.0) for key, value in distribution.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total else values


def _edge_endpoint_key(edge: Mapping[str, Any], role: str) -> str:
    for field in (f"{role}_device", f"{role}_device_id", f"{role}_device_key", role, f"{role}_id", f"{role}_index"):
        if edge.get(field) is not None:
            return _canonical_device_key(edge[field])
    return "unknown"


def _canonical_device_key(value: Any) -> str:
    text = str(value)
    if text.startswith("d:"):
        return text
    if text.startswith("device_") and text[7:].isdigit():
        return f"d:{int(text[7:])}"
    if text.isdigit():
        return f"d:{int(text)}"
    return text


def _coerce_sequences(sequences) -> list[BehaviorSequence]:
    if sequences is None:
        return []
    if isinstance(sequences, BehaviorSequence):
        return [sequences]
    items = list(sequences)
    if all(isinstance(item, BehaviorSequence) for item in items):
        return items
    flats = [flat for item in items if (flat := _extract_flat(item)) is not None]
    return load_numeric_sequences(flats)


def _extract_flat(item: Any) -> list[int] | None:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, tuple):
        for part in item:
            if (flat := _extract_flat(part)) is not None:
                return flat
        return None
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item and len(item) % 4 == 0:
        try:
            return [int(value) for value in item]
        except (TypeError, ValueError):
            return None
    return None
