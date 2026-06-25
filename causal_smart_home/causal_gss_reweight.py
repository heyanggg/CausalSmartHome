from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .schema import BehaviorSequence, load_numeric_sequences


def build_device_transition_graph(
    sequences,
    device_name_map: dict | None = None,
) -> dict:
    """Build SmartGen-style device transition graph from flattened sequences."""

    behavior_sequences = _coerce_sequences(sequences)
    counts: Counter[tuple[str, str]] = Counter()
    outgoing: Counter[str] = Counter()
    for seq in behavior_sequences:
        devices = [_canonical_device(ev.device) for ev in seq]
        for source, target in zip(devices, devices[1:]):
            counts[(source, target)] += 1
            outgoing[source] += 1

    edges: list[dict[str, Any]] = []
    for (source, target), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        source_id = _device_id_value(source)
        target_id = _device_id_value(target)
        transition_score = count / max(outgoing[source], 1)
        edges.append(
            {
                "source_device": source_id,
                "target_device": target_id,
                "source_device_key": source,
                "target_device_key": target,
                "source_name": _device_name(source_id, device_name_map),
                "target_name": _device_name(target_id, device_name_map),
                "count": int(count),
                "transition_score": float(transition_score),
            }
        )
    return {
        "level": "device",
        "num_sequences": len(behavior_sequences),
        "num_edges": len(edges),
        "edges": edges,
    }


def reweight_gss_edges(
    transition_edges: list[dict],
    causal_edges: list[dict],
    lambda_causal: float = 1.0,
    mode: str = "multiplicative",
    add_causal_edges: bool = False,
    top_k: int = 50,
) -> dict:
    """Let guarded causal relation edges participate in SmartGen GSS edge scoring."""

    if mode not in {"multiplicative", "additive"}:
        raise ValueError("mode must be multiplicative or additive")

    causal_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in causal_edges:
        pair = (_edge_device_key(edge, "source"), _edge_device_key(edge, "target"))
        if pair[0] == "unknown" or pair[1] == "unknown":
            continue
        current = causal_by_pair.get(pair)
        strength = _guarded_strength(edge)
        if current is None or strength > _guarded_strength(current):
            causal_by_pair[pair] = dict(edge)

    transition_scores = [float(edge.get("transition_score", 0.0)) for edge in transition_edges]
    max_transition = max(transition_scores) if transition_scores else 0.0
    causal_strengths = [_guarded_strength(edge) for edge in causal_by_pair.values()]
    max_causal = max(causal_strengths) if causal_strengths else 0.0

    out_edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for edge in transition_edges:
        source_key = _canonical_device(edge.get("source_device", edge.get("source_device_key", "unknown")))
        target_key = _canonical_device(edge.get("target_device", edge.get("target_device_key", "unknown")))
        pair = (source_key, target_key)
        seen_pairs.add(pair)
        causal = causal_by_pair.get(pair, {})
        row = _score_edge_row(
            transition_edge=edge,
            causal_edge=causal,
            source_key=source_key,
            target_key=target_key,
            max_transition=max_transition,
            max_causal=max_causal,
            lambda_causal=lambda_causal,
            mode=mode,
            origin="transition_existing",
        )
        out_edges.append(row)

    if add_causal_edges:
        for pair, causal in causal_by_pair.items():
            if pair in seen_pairs:
                continue
            transition_stub = {
                "source_device": _device_id_value(pair[0]),
                "target_device": _device_id_value(pair[1]),
                "source_name": causal.get("source_name") or _device_name(_device_id_value(pair[0]), None),
                "target_name": causal.get("target_name") or _device_name(_device_id_value(pair[1]), None),
                "transition_score": 0.0,
                "count": 0,
            }
            row = _score_edge_row(
                transition_edge=transition_stub,
                causal_edge=causal,
                source_key=pair[0],
                target_key=pair[1],
                max_transition=max_transition,
                max_causal=max_causal,
                lambda_causal=lambda_causal,
                mode="additive" if mode == "additive" else mode,
                origin="causal_relation_augmented",
            )
            if mode == "multiplicative" and row["final_score"] == 0.0:
                # A purely causal augmented edge should not vanish in
                # multiplicative mode just because there was no transition edge.
                row["final_score"] = lambda_causal * row["normalized_guarded_causal_strength"]
            out_edges.append(row)

    out_edges.sort(key=lambda edge: float(edge.get("final_score", 0.0)), reverse=True)
    if top_k and top_k > 0:
        out_edges = out_edges[:top_k]

    return {
        "hint_type": "guarded_causal_reweighted_gss",
        "level": "device",
        "config": {
            "lambda_causal": lambda_causal,
            "mode": mode,
            "add_causal_edges": add_causal_edges,
            "top_k": top_k,
        },
        "summary": {
            "input_transition_edges": len(transition_edges),
            "input_causal_edges": len(causal_edges),
            "output_edges": len(out_edges),
            "num_causal_relation_augmented_edges": sum(1 for edge in out_edges if edge.get("edge_origin") == "causal_relation_augmented"),
            "num_guard_suppressed_edges": sum(1 for edge in out_edges if edge.get("guard_action") == "suppress"),
            "num_guard_downweighted_edges": sum(1 for edge in out_edges if edge.get("guard_action") == "downweight"),
            "avg_guarded_causal_strength": _mean([float(edge.get("guarded_causal_strength", 0.0)) for edge in out_edges]),
        },
        "edges": out_edges,
    }


def _score_edge_row(
    transition_edge: Mapping[str, Any],
    causal_edge: Mapping[str, Any],
    source_key: str,
    target_key: str,
    max_transition: float,
    max_causal: float,
    lambda_causal: float,
    mode: str,
    origin: str,
) -> dict[str, Any]:
    transition_score = float(transition_edge.get("transition_score", 0.0))
    normalized_transition = transition_score / max_transition if max_transition > 0 else 0.0
    raw_causal = _raw_strength(causal_edge)
    guarded_causal = _guarded_strength(causal_edge)
    normalized_causal = guarded_causal / max_causal if max_causal > 0 else 0.0
    if mode == "additive":
        final_score = normalized_transition + lambda_causal * normalized_causal
    else:
        final_score = normalized_transition * (1.0 + lambda_causal * normalized_causal)
    source_id = _device_id_value(source_key)
    target_id = _device_id_value(target_key)
    return {
        "source_device": source_id,
        "target_device": target_id,
        "source_device_key": source_key,
        "target_device_key": target_key,
        "source_name": str(transition_edge.get("source_name") or causal_edge.get("source_name") or _device_name(source_id, None)),
        "target_name": str(transition_edge.get("target_name") or causal_edge.get("target_name") or _device_name(target_id, None)),
        "count": int(transition_edge.get("count", 0)),
        "transition_score": transition_score,
        "normalized_transition_score": normalized_transition,
        "raw_causal_strength": raw_causal,
        "guarded_causal_strength": guarded_causal,
        "normalized_guarded_causal_strength": normalized_causal,
        "final_score": float(final_score),
        "edge_origin": origin,
        "guard_action": str(causal_edge.get("guard_action", "keep")),
        "guard_reason": str(causal_edge.get("guard_reason", "")),
        "lag": causal_edge.get("lag"),
    }


def _raw_strength(edge: Mapping[str, Any]) -> float:
    if not edge:
        return 0.0
    return float(edge.get("raw_weight", edge.get("raw_causal_strength", edge.get("weight", 0.0))))


def _guarded_strength(edge: Mapping[str, Any]) -> float:
    if not edge:
        return 0.0
    return float(edge.get("guarded_weight", edge.get("guarded_causal_strength", edge.get("weight", 0.0))))


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


def _device_id_value(device_key: str) -> int | str:
    text = str(device_key)
    if text.startswith("d:") and text[2:].isdigit():
        return int(text[2:])
    if text.isdigit():
        return int(text)
    return text


def _device_name(device_id: int | str, mapping: Mapping[Any, str] | None) -> str:
    if mapping and device_id in mapping:
        return str(mapping[device_id])
    if mapping and str(device_id) in mapping:
        return str(mapping[str(device_id)])
    return f"device_{device_id}"


def _coerce_sequences(sequences) -> list[BehaviorSequence]:
    if isinstance(sequences, BehaviorSequence):
        return [sequences]
    if isinstance(sequences, list) and all(isinstance(item, BehaviorSequence) for item in sequences):
        return list(sequences)
    flats: list[list[int]] = []
    for item in list(sequences):
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, tuple):
            for part in item:
                flat = _extract_flat(part)
                if flat is not None:
                    flats.append(flat)
                    break
        else:
            flat = _extract_flat(item)
            if flat is not None:
                flats.append(flat)
    return load_numeric_sequences(flats)


def _extract_flat(item: Any) -> list[int] | None:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) > 0 and len(item) % 4 == 0:
        try:
            return [int(value) for value in item]
        except Exception:
            return None
    return None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
