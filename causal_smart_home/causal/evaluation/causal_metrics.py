"""Reusable causal-prior and Causal-TOF metrics."""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Any, Iterable, Mapping


def edge_strength(edge: Mapping[str, Any]) -> float:
    """Read an edge strength across all historical artifact schemas."""
    for key in ("target_adapted_weight", "guarded_weight", "weight", "raw_weight", "causal_strength"):
        value = edge.get(key)
        if value is not None:
            return float(value)
    return 0.0


def causal_edge_statistics(edges: Iterable[Mapping[str, Any]], *, threshold: float = 0.0) -> dict[str, Any]:
    """Return paper-ready edge statistics without aggregating experiment seeds."""
    values = [edge_strength(edge) for edge in edges]
    nonzero = [value for value in values if abs(value) > threshold]
    ordered = sorted(nonzero)
    avg = mean(nonzero) if nonzero else 0.0
    variance = mean([(value - avg) ** 2 for value in nonzero]) if nonzero else 0.0
    return {
        "num_edges": len(values),
        "num_nonzero_edges": len(nonzero),
        "nonzero_ratio": len(nonzero) / len(values) if values else 0.0,
        "sum_strength": float(sum(nonzero)),
        "mean_strength": float(avg),
        "std_strength": float(sqrt(variance)),
        "min_strength": float(ordered[0]) if ordered else 0.0,
        "median_strength": float(_quantile(ordered, 0.5)),
        "p90_strength": float(_quantile(ordered, 0.9)),
        "max_strength": float(ordered[-1]) if ordered else 0.0,
        "threshold": float(threshold),
    }


def before_after_edge_statistics(
    before_edges: Iterable[Mapping[str, Any]],
    after_edges: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    before = causal_edge_statistics(before_edges, threshold=threshold)
    after = causal_edge_statistics(after_edges, threshold=threshold)
    return {
        "before": before,
        "after": after,
        "delta": {
            key: float(after[key]) - float(before[key])
            for key in ("num_nonzero_edges", "sum_strength", "mean_strength", "max_strength")
        },
    }


def summarize_causal_consistency(scores: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    rows = list(scores)
    values = [float(row.get("causal_consistency_score", 0.0)) for row in rows]
    return {
        "num_sequences": len(rows),
        "mean_causal_consistency_score": float(mean(values)) if values else 0.0,
        "min_causal_consistency_score": float(min(values)) if values else 0.0,
        "max_causal_consistency_score": float(max(values)) if values else 0.0,
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
