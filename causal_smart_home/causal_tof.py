from __future__ import annotations

import math
import pickle
import random
import importlib
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import BehaviorSequence, dump_numeric_sequences, load_numeric_sequences
from .target_distribution_guard import compute_device_distribution


def score_sequence_causal_tof(
    sequence,
    guarded_edges: list[dict],
    target_distribution: dict | None = None,
    reconstruction_loss: float | None = None,
    alpha_rec: float = 1.0,
    beta_violation: float = 1.0,
    gamma_dist: float = 1.0,
    temperature: float = 2.0,
) -> dict:
    """Score one generated sequence for Causal-TOF soft weighting.

    Lower final_score is better.  The default decision is ``weight``; scripts can
    convert it to rank/filter behavior without making hard deletion the default.
    """

    seq = _coerce_one_sequence(sequence)
    positions = _device_positions(seq)
    satisfied: list[dict[str, Any]] = []
    violated: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    satisfied_weight = 0.0
    violated_weight = 0.0
    missing_weight = 0.0

    for edge in guarded_edges:
        weight = _edge_weight(edge)
        if weight <= 0:
            continue
        source_key = _edge_device_key(edge, "source")
        target_key = _edge_device_key(edge, "target")
        src_pos = positions.get(source_key, [])
        tgt_pos = positions.get(target_key, [])
        row = {
            "source": source_key,
            "target": target_key,
            "source_name": edge.get("source_name", source_key),
            "target_name": edge.get("target_name", target_key),
            "weight": weight,
            "final_score": edge.get("final_score"),
            "guard_action": edge.get("guard_action", "keep"),
        }
        if not src_pos or not tgt_pos:
            row["reason"] = "missing_source" if not src_pos else "missing_target"
            missing.append(row)
            missing_weight += weight
            continue
        if any(i < j for i in src_pos for j in tgt_pos):
            satisfied.append(row)
            satisfied_weight += weight
        else:
            row["reason"] = "target_precedes_source_or_no_valid_order"
            violated.append(row)
            violated_weight += weight

    checked_weight = satisfied_weight + violated_weight
    causal_coverage = satisfied_weight / checked_weight if checked_weight > 0 else 1.0
    causal_violation = violated_weight / checked_weight if checked_weight > 0 else 0.0
    dist_penalty = _distribution_penalty(seq, target_distribution)
    rec = 0.0 if reconstruction_loss is None else float(reconstruction_loss)
    final_score = alpha_rec * rec + beta_violation * causal_violation + gamma_dist * dist_penalty
    sample_weight = math.exp(-float(temperature) * final_score)

    return {
        "index": None,
        "sequence_id": seq.sequence_id,
        "causal_coverage": float(causal_coverage),
        "causal_violation": float(causal_violation),
        "distribution_penalty": float(dist_penalty),
        "reconstruction_loss": reconstruction_loss,
        "final_score": float(final_score),
        "sample_weight": float(sample_weight),
        "satisfied_edges": satisfied,
        "violated_edges": violated,
        "missing_edges": missing,
        "checked_edge_weight": float(checked_weight),
        "missing_edge_weight": float(missing_weight),
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
    scores: list[dict[str, Any]] = []
    for index, (sequence, rec_loss) in enumerate(zip(sequences, losses)):
        score = score_sequence_causal_tof(
            sequence,
            guarded_edges,
            target_distribution=target_distribution,
            reconstruction_loss=rec_loss,
            **kwargs,
        )
        score["index"] = index
        if mode == "rank":
            score["decision"] = "rank"
        elif mode == "filter":
            score["decision"] = "keep" if score["sample_weight"] >= min_weight else "delete"
        else:
            score["decision"] = "weight"
        scores.append(score)
    if mode == "rank":
        scores.sort(key=lambda item: float(item["final_score"]))
    return scores


def weighted_resample_sequences(
    sequences: Sequence,
    scores: Sequence[Mapping[str, Any]],
    seed: int = 2024,
    max_copies: int = 3,
    target_size: int | None = None,
) -> tuple[list[BehaviorSequence], dict[str, Any]]:
    """Weighted-resampling fallback for downstream AD code without sample weights."""

    seqs = [_coerce_one_sequence(seq) for seq in sequences]
    if len(seqs) != len(scores):
        raise ValueError("sequences and scores length mismatch")
    target_size = target_size or len(seqs)
    if target_size <= 0:
        return [], {"target_size": target_size, "resampled_size": 0, "max_copies": max_copies}

    weights = [max(float(score.get("sample_weight", 0.0)), 0.0) for score in scores]
    if sum(weights) <= 0:
        weights = [1.0 for _ in seqs]
    rng = random.Random(seed)
    copies = [0 for _ in seqs]
    out: list[BehaviorSequence] = []
    attempts = 0
    max_attempts = max(target_size * 20, 100)
    while len(out) < target_size and attempts < max_attempts:
        attempts += 1
        idx = _weighted_choice(weights, rng)
        if copies[idx] >= max_copies:
            weights[idx] = 0.0
            if sum(weights) <= 0:
                break
            continue
        out.append(seqs[idx])
        copies[idx] += 1
    if len(out) < target_size:
        # Fill deterministically with best still-under-cap samples.
        ordered = sorted(range(len(seqs)), key=lambda i: float(scores[i].get("sample_weight", 0.0)), reverse=True)
        cursor = 0
        while len(out) < target_size and ordered:
            idx = ordered[cursor % len(ordered)]
            if copies[idx] < max_copies:
                out.append(seqs[idx])
                copies[idx] += 1
            cursor += 1
            if cursor > len(ordered) * max_copies * 2:
                break
    config = {
        "seed": seed,
        "target_size": target_size,
        "resampled_size": len(out),
        "max_copies": max_copies,
        "copy_counts": copies,
        "num_duplicated_sources": sum(1 for value in copies if value > 1),
        "num_unused_sources": sum(1 for value in copies if value == 0),
    }
    return out, config


def load_pickle_sequences(path: str | Path) -> list[BehaviorSequence]:
    _install_numpy_pickle_compat()
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return _coerce_many_sequences(raw)


def save_pickle_sequences(path: str | Path, sequences: Sequence[BehaviorSequence]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dump_numeric_sequences(sequences), f)


def extract_guarded_edges(payload: Mapping[str, Any] | list[dict]) -> list[dict]:
    if isinstance(payload, list):
        return list(payload)
    for key in ("edges", "guarded_edges", "top_causal_edges"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    hints = payload.get("guarded_reweighted_gss_hints")
    if isinstance(hints, Mapping) and isinstance(hints.get("edges"), list):
        return list(hints["edges"])
    raise ValueError("guarded hints JSON must contain edges, guarded_edges, or top_causal_edges")


def _distribution_penalty(seq: BehaviorSequence, target_distribution: Mapping[str, float] | None) -> float:
    if not target_distribution:
        return 0.0
    obs = compute_device_distribution([seq])
    target = {str(k): float(v) for k, v in target_distribution.items()}
    penalty = 0.0
    for key, obs_freq in obs.items():
        penalty += max(0.0, float(obs_freq) - float(target.get(key, 0.0)))
    return min(1.0, penalty)


def _device_positions(seq: BehaviorSequence) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for idx, event in enumerate(seq):
        positions.setdefault(event.key("device"), []).append(idx)
    return positions


def _edge_weight(edge: Mapping[str, Any]) -> float:
    for key in ("guarded_weight", "guarded_causal_strength", "weight", "raw_weight", "final_score"):
        if key in edge and edge[key] is not None:
            try:
                return float(edge[key])
            except Exception:
                continue
    return 0.0


def _edge_device_key(edge: Mapping[str, Any], role: str) -> str:
    for field in (f"{role}_device", f"{role}_device_id", f"{role}_device_key"):
        if field in edge and edge[field] is not None:
            return _canonical_device(edge[field])
    if role in edge and edge[role] is not None:
        return _canonical_device(edge[role])
    value = edge.get(f"{role}_id", edge.get(f"{role}_index"))
    return _canonical_device(value if value is not None else "unknown")


def _canonical_device(value: Any) -> str:
    text = str(value)
    if text.startswith("d:"):
        return text
    if text.startswith("device_") and text[len("device_") :].isdigit():
        return f"d:{int(text[len('device_'):])}"
    if text.isdigit():
        return f"d:{int(text)}"
    return text


def _install_numpy_pickle_compat() -> None:
    # Some Gen pickles were produced with newer NumPy module paths
    # (numpy._core.*). Older experiment envs still expose numpy.core.*.
    if "numpy._core" not in sys.modules:
        try:
            sys.modules["numpy._core"] = importlib.import_module("numpy.core")
        except Exception:
            pass
    for submodule in ("multiarray", "numeric", "umath"):
        old_name = f"numpy.core.{submodule}"
        new_name = f"numpy._core.{submodule}"
        if new_name not in sys.modules:
            try:
                sys.modules[new_name] = importlib.import_module(old_name)
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
        return list(raw)
    out: list[BehaviorSequence] = []
    for item in list(raw):
        flat = _extract_flat(item)
        if flat is not None:
            out.append(load_numeric_sequences([flat])[0])
    return out


def _extract_flat(item: Any) -> list[int] | None:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, tuple):
        for part in item:
            flat = _extract_flat(part)
            if flat is not None:
                return flat
        return None
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) > 0 and len(item) % 4 == 0:
        try:
            return [int(value) for value in item]
        except Exception:
            return None
    return None


def _weighted_choice(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    needle = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= needle:
            return index
    return len(weights) - 1
