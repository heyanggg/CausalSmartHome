"""Causal consistency refinement after the unchanged Gen original TOF.

Unlike the historical implementation, this module does not count binary
ordering violations.  For a sequence ``S`` it averages the normalized causal
strength of every ordered event pair and exposes that value as
``causal_consistency_score``.  Higher final scores and sample weights are
better.
"""

from __future__ import annotations

import importlib
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...schema import BehaviorSequence, dump_numeric_sequences, load_numeric_sequences
from ..adaptation.target_guard import compute_device_distribution


def score_sequence_causal_tof(
    sequence,
    guarded_edges: list[dict],
    target_distribution: dict | None = None,
    reconstruction_loss: float | None = None,
    alpha_rec: float = 1.0,
    beta_inconsistency: float = 1.0,
    gamma_dist: float = 1.0,
    temperature: float = 2.0,
    beta_violation: float | None = None,
    penalize_downweighted_edges: bool = False,
) -> dict[str, Any]:
    """Score one sequence with continuous causal consistency.

    ``beta_violation`` and ``penalize_downweighted_edges`` are accepted for old
    callers.  The former aliases ``beta_inconsistency``; the latter controls
    whether historically downweighted edges enter the continuous edge table.
    """
    if beta_violation is not None:
        beta_inconsistency = float(beta_violation)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    seq = _coerce_one_sequence(sequence)
    edge_lookup, max_strength = _edge_lookup(guarded_edges, penalize_downweighted_edges)
    pair_rows: list[dict[str, Any]] = []
    strengths: list[float] = []
    events = list(seq)
    for source_index in range(len(events)):
        for target_index in range(source_index + 1, len(events)):
            source_key = events[source_index].key("device")
            target_key = events[target_index].key("device")
            raw_strength = edge_lookup.get((source_key, target_key), 0.0)
            normalized_strength = raw_strength / max_strength if max_strength > 0 else 0.0
            strengths.append(normalized_strength)
            pair_rows.append(
                {
                    "source_index": source_index,
                    "target_index": target_index,
                    "source": source_key,
                    "target": target_key,
                    "causal_strength": float(raw_strength),
                    "normalized_causal_strength": float(normalized_strength),
                }
            )

    consistency = sum(strengths) / len(strengths) if strengths else 1.0
    inconsistency = 1.0 - consistency
    distribution_penalty = _distribution_penalty(seq, target_distribution)
    rec_loss = 0.0 if reconstruction_loss is None else max(float(reconstruction_loss), 0.0)
    # Keep the requested paper formula verbatim. Gen TOF currently does not
    # expose per-row losses to this stage, so the term is normally zero unless
    # an explicit reconstruction-loss vector is supplied.
    reconstruction_term = rec_loss
    final_score = (
        float(alpha_rec) * reconstruction_term
        - float(beta_inconsistency) * inconsistency
        - float(gamma_dist) * distribution_penalty
    )
    sample_weight = math.exp(final_score / float(temperature))

    supported_pairs = [row for row in pair_rows if row["causal_strength"] > 0]
    return {
        "index": None,
        "sequence_id": seq.sequence_id,
        "causal_consistency_score": float(consistency),
        "causal_inconsistency": float(inconsistency),
        "distribution_penalty": float(distribution_penalty),
        "reconstruction_loss": reconstruction_loss,
        "reconstruction_term": float(reconstruction_term),
        "final_score": float(final_score),
        "score_direction": "higher_is_better",
        "score_formula": "alpha*reconstruction_loss-beta*causal_inconsistency-gamma*distribution_penalty",
        "sample_weight": float(sample_weight),
        "num_ordered_event_pairs": len(pair_rows),
        "num_causally_supported_pairs": len(supported_pairs),
        "ordered_event_pair_scores": pair_rows,
        # Compatibility audit fields. They are continuous-score aliases, not
        # the removed binary violation algorithm.
        "causal_coverage": float(consistency),
        "causal_violation": float(inconsistency),
        "satisfied_edges": supported_pairs,
        "violated_edges": [],
        "missing_edges": [],
        "decision": "weight",
    }


def score_sequences_causal_tof(
    sequences: Sequence,
    guarded_edges: list[dict],
    target_distribution: dict | None = None,
    reconstruction_losses: Sequence[float] | None = None,
    mode: str = "weight",
    min_weight: float = 0.05,
    **kwargs,
) -> list[dict[str, Any]]:
    if mode not in {"rank", "weight", "filter"}:
        raise ValueError("mode must be rank, weight, or filter")
    losses = list(reconstruction_losses) if reconstruction_losses is not None else [None] * len(sequences)
    if len(losses) != len(sequences):
        raise ValueError("reconstruction_losses length must match sequences length")
    scores = []
    for index, (sequence, loss) in enumerate(zip(sequences, losses)):
        score = score_sequence_causal_tof(
            sequence,
            guarded_edges,
            target_distribution=target_distribution,
            reconstruction_loss=loss,
            **kwargs,
        )
        score["index"] = index
        score["decision"] = (
            "rank" if mode == "rank" else "keep" if mode == "filter" and score["sample_weight"] >= min_weight
            else "delete" if mode == "filter" else "weight"
        )
        scores.append(score)
    if mode == "rank":
        scores.sort(key=lambda item: float(item["final_score"]), reverse=True)
    return scores


def weighted_resample_sequences(
    sequences: Sequence,
    scores: Sequence[Mapping[str, Any]],
    seed: int = 2024,
    max_copies: int = 3,
    target_size: int | None = None,
) -> tuple[list[BehaviorSequence], dict[str, Any]]:
    seqs = [_coerce_one_sequence(sequence) for sequence in sequences]
    if len(seqs) != len(scores):
        raise ValueError("sequences and scores length mismatch")
    target_size = len(seqs) if target_size is None else target_size
    if target_size <= 0 or not seqs:
        return [], {"seed": seed, "target_size": target_size, "resampled_size": 0, "max_copies": max_copies}
    if target_size > len(seqs) * max_copies:
        raise ValueError("target_size exceeds len(sequences) * max_copies")

    weights = [max(float(score.get("sample_weight", 0.0)), 0.0) for score in scores]
    if not any(weights):
        weights = [1.0] * len(seqs)
    rng = random.Random(seed)
    copies = [0] * len(seqs)
    output: list[BehaviorSequence] = []
    while len(output) < target_size:
        available = [weight if copies[index] < max_copies else 0.0 for index, weight in enumerate(weights)]
        index = _weighted_choice(available, rng)
        output.append(seqs[index])
        copies[index] += 1
    return output, {
        "seed": seed,
        "target_size": target_size,
        "resampled_size": len(output),
        "max_copies": max_copies,
        "copy_counts": copies,
        "num_duplicated_sources": sum(value > 1 for value in copies),
        "num_unused_sources": sum(value == 0 for value in copies),
    }


def load_pickle_sequences(path: str | Path) -> list[BehaviorSequence]:
    _install_numpy_pickle_compat()
    with open(path, "rb") as handle:
        return _coerce_many_sequences(pickle.load(handle))


def save_pickle_sequences(path: str | Path, sequences: Sequence[BehaviorSequence]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(dump_numeric_sequences(sequences), handle)


def extract_guarded_edges(payload: Mapping[str, Any] | list[dict]) -> list[dict]:
    if isinstance(payload, list):
        return list(payload)
    for key in ("edges", "guarded_edges", "top_causal_edges"):
        if isinstance(payload.get(key), list):
            return list(payload[key])
    for container in ("guarded_reweighted_gss_hints", "causal_reweighted_gss_hints"):
        hints = payload.get(container)
        if isinstance(hints, Mapping) and isinstance(hints.get("edges"), list):
            return list(hints["edges"])
    raise ValueError("causal hints JSON must contain an edge list")


def _edge_lookup(edges: Sequence[Mapping[str, Any]], include_downweighted: bool) -> tuple[dict[tuple[str, str], float], float]:
    lookup: dict[tuple[str, str], float] = {}
    for edge in edges:
        if not include_downweighted and edge.get("guard_action") == "downweight":
            continue
        source = _edge_device_key(edge, "source")
        target = _edge_device_key(edge, "target")
        strength = max(_edge_weight(edge), 0.0)
        lookup[(source, target)] = max(lookup.get((source, target), 0.0), strength)
    return lookup, max(lookup.values(), default=0.0)


def _distribution_penalty(seq: BehaviorSequence, target_distribution: Mapping[str, float] | None) -> float:
    if not target_distribution:
        return 0.0
    observed = compute_device_distribution([seq])
    target = {str(key): float(value) for key, value in target_distribution.items()}
    keys = set(observed) | set(target)
    # Total variation distance is bounded and symmetric.
    return 0.5 * sum(abs(observed.get(key, 0.0) - target.get(key, 0.0)) for key in keys)


def _edge_weight(edge: Mapping[str, Any]) -> float:
    for key in ("target_adapted_weight", "guarded_weight", "weight", "causal_strength", "raw_weight", "final_score"):
        if edge.get(key) is not None:
            return float(edge[key])
    return 0.0


def _edge_device_key(edge: Mapping[str, Any], role: str) -> str:
    for field in (f"{role}_device", f"{role}_device_id", f"{role}_device_key", role, f"{role}_id", f"{role}_index"):
        if edge.get(field) is not None:
            return _canonical_device(edge[field])
    return "unknown"


def _canonical_device(value: Any) -> str:
    text = str(value)
    if text.startswith("d:"):
        return text
    if text.startswith("device_") and text[7:].isdigit():
        return f"d:{int(text[7:])}"
    if text.isdigit():
        return f"d:{int(text)}"
    return text


def _install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules:
        try:
            sys.modules["numpy._core"] = importlib.import_module("numpy.core")
        except Exception:
            pass
    for submodule in ("multiarray", "numeric", "umath"):
        try:
            sys.modules.setdefault(f"numpy._core.{submodule}", importlib.import_module(f"numpy.core.{submodule}"))
        except Exception:
            pass


def _coerce_one_sequence(sequence) -> BehaviorSequence:
    if isinstance(sequence, BehaviorSequence):
        return sequence
    flat = _extract_flat(sequence)
    if flat is None:
        raise ValueError(f"cannot convert item to BehaviorSequence: {type(sequence)}")
    return load_numeric_sequences([flat])[0]


def _coerce_many_sequences(raw) -> list[BehaviorSequence]:
    if isinstance(raw, list) and all(isinstance(item, BehaviorSequence) for item in raw):
        return raw
    return [_coerce_one_sequence(item) for item in list(raw) if _extract_flat(item) is not None]


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


def _weighted_choice(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        raise ValueError("no sequence remains below max_copies")
    needle = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= needle:
            return index
    return len(weights) - 1
