"""Generation quality metrics and case-study matrix export."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...json_utils import write_json
from ...schema import BehaviorSequence
from ..adaptation.target_guard import compute_device_distribution
from ..refinement.causal_tof import load_pickle_sequences


def device_distribution_kl(
    target_distribution: Mapping[str, float],
    synthetic_distribution: Mapping[str, float],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return KL(target || synthetic) with shared-support smoothing."""
    keys = sorted(set(target_distribution) | set(synthetic_distribution))
    if not keys:
        return 0.0
    target = _smoothed_distribution(target_distribution, keys, epsilon)
    synthetic = _smoothed_distribution(synthetic_distribution, keys, epsilon)
    return float(sum(target[key] * math.log(target[key] / synthetic[key]) for key in keys))


def build_transition_matrix(
    sequences: Sequence[BehaviorSequence], channels: Sequence[str] | None = None
) -> tuple[list[str], list[list[float]]]:
    """Build a row-normalized adjacent device-transition matrix."""
    channels = list(channels or _device_channels(sequences))
    index = {key: position for position, key in enumerate(channels)}
    matrix = [[0.0 for _ in channels] for _ in channels]
    for sequence in sequences:
        devices = [event.key("device") for event in sequence]
        for source, target in zip(devices, devices[1:]):
            if source in index and target in index:
                matrix[index[source]][index[target]] += 1.0
    return channels, _row_normalize(matrix)


def build_empirical_causal_matrix(
    sequences: Sequence[BehaviorSequence], channels: Sequence[str] | None = None
) -> tuple[list[str], list[list[float]]]:
    """Build a directed precedence-strength graph for quality comparison.

    Each entry is the fraction of occurrences of source ``i`` that have target
    ``j`` later in the same sequence.  This deterministic empirical graph is
    used only for generation-quality/case-study evaluation; it does not replace
    gradient-GC discovery in the method pipeline.
    """
    channels = list(channels or _device_channels(sequences))
    index = {key: position for position, key in enumerate(channels)}
    numerator = [[0.0 for _ in channels] for _ in channels]
    denominator = [0.0 for _ in channels]
    for sequence in sequences:
        devices = [event.key("device") for event in sequence]
        for source_position, source in enumerate(devices):
            if source not in index:
                continue
            source_index = index[source]
            denominator[source_index] += 1.0
            later = set(devices[source_position + 1 :])
            for target in later:
                if target in index and target != source:
                    numerator[source_index][index[target]] += 1.0
    matrix = [
        [value / denominator[row] if denominator[row] else 0.0 for value in numerator[row]]
        for row in range(len(channels))
    ]
    return channels, matrix


def matrix_cosine_similarity(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    left_flat = [float(value) for row in left for value in row]
    right_flat = [float(value) for row in right for value in row]
    if len(left_flat) != len(right_flat):
        raise ValueError("matrix shapes differ")
    dot = sum(a * b for a, b in zip(left_flat, right_flat))
    left_norm = math.sqrt(sum(value * value for value in left_flat))
    right_norm = math.sqrt(sum(value * value for value in right_flat))
    if left_norm == 0 and right_norm == 0:
        return 1.0
    return float(dot / (left_norm * right_norm)) if left_norm and right_norm else 0.0


def evaluate_generation_quality(
    target_sequences: Sequence[BehaviorSequence],
    synthetic_sequences: Sequence[BehaviorSequence],
    *,
    variant: str | None = None,
    dataset: str | None = None,
    scenario: str | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    channels = sorted(set(_device_channels(target_sequences)) | set(_device_channels(synthetic_sequences)), key=_channel_sort_key)
    target_distribution = compute_device_distribution(target_sequences)
    synthetic_distribution = compute_device_distribution(synthetic_sequences)
    _, target_transition = build_transition_matrix(target_sequences, channels)
    _, synthetic_transition = build_transition_matrix(synthetic_sequences, channels)
    _, target_causal = build_empirical_causal_matrix(target_sequences, channels)
    _, synthetic_causal = build_empirical_causal_matrix(synthetic_sequences, channels)
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "num_target_sequences": len(target_sequences),
        "num_synthetic_sequences": len(synthetic_sequences),
        "device_distribution_kl": device_distribution_kl(target_distribution, synthetic_distribution),
        "device_distribution_kl_direction": "KL(target_normal || synthetic)",
        "transition_matrix_similarity": matrix_cosine_similarity(target_transition, synthetic_transition),
        "causal_graph_similarity": matrix_cosine_similarity(target_causal, synthetic_causal),
        "similarity_metric": "cosine",
        "causal_graph_estimator": "empirical_directed_precedence_strength",
    }
    case_study = {
        "schema_version": 1,
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "channels": channels,
        "real_target_normal": {
            "device_distribution": target_distribution,
            "transition_matrix": target_transition,
            "causal_matrix": target_causal,
        },
        "synthetic": {
            "device_distribution": synthetic_distribution,
            "transition_matrix": synthetic_transition,
            "causal_matrix": synthetic_causal,
        },
    }
    return summary, case_study


def evaluate_generation_quality_files(
    target_pkl: str | Path,
    synthetic_pkl: str | Path,
    out_dir: str | Path,
    **coordinates: Any,
) -> dict[str, Any]:
    summary, case_study = evaluate_generation_quality(
        load_pickle_sequences(target_pkl),
        load_pickle_sequences(synthetic_pkl),
        **coordinates,
    )
    output = Path(out_dir)
    write_json(output / "generation_quality_summary.json", summary)
    write_json(output / "case_study_matrices.json", case_study)
    return summary


def _device_channels(sequences: Sequence[BehaviorSequence]) -> list[str]:
    return sorted({event.key("device") for sequence in sequences for event in sequence}, key=_channel_sort_key)


def _channel_sort_key(key: str) -> tuple[int, str]:
    suffix = key[2:] if key.startswith("d:") else key
    return (int(suffix), key) if suffix.isdigit() else (10**9, key)


def _row_normalize(matrix: list[list[float]]) -> list[list[float]]:
    return [[value / sum(row) for value in row] if sum(row) else list(row) for row in matrix]


def _smoothed_distribution(distribution: Mapping[str, float], keys: Sequence[str], epsilon: float) -> dict[str, float]:
    values = {key: max(float(distribution.get(key, 0.0)), 0.0) + epsilon for key in keys}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}
