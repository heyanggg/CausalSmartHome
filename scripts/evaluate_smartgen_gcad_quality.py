#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_prior import GradientCausalMiner
from causal_smart_home.event_tensor import EventTensorizer
from causal_smart_home.schema import BehaviorEvent, BehaviorSequence, dump_numeric_sequences


EPS = 1e-12
DEFAULT_SMARTGEN_DICTIONARY = Path("/home/heyang/projects/SmartGen/SmartGen/dictionary.py")


def load_pickle(path: str | Path) -> Any:
    # Some SmartGen pickles were produced with newer NumPy paths
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
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def unwrap_numeric_candidate(item: Any) -> Any:
    """Accept plain SmartGen sequences and a few common tuple wrappers."""
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, tuple):
        for part in item:
            if isinstance(part, np.ndarray):
                part = part.tolist()
            if isinstance(part, list) and all(isinstance(x, (int, np.integer)) for x in part):
                return part
    return item


def load_sequences(path: str | Path, name: str) -> tuple[list[BehaviorSequence], list[list[int]], list[dict[str, Any]]]:
    raw = load_pickle(path)
    warnings: list[dict[str, Any]] = []
    sequences: list[BehaviorSequence] = []
    flat_sequences: list[list[int]] = []
    if not isinstance(raw, Iterable):
        raise ValueError(f"{name}: expected an iterable object in {path}")

    for idx, item in enumerate(raw):
        candidate = unwrap_numeric_candidate(item)
        if isinstance(candidate, np.ndarray):
            candidate = candidate.tolist()
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            warnings.append({"dataset": name, "index": idx, "reason": "not_a_sequence"})
            continue
        if len(candidate) == 0:
            warnings.append({"dataset": name, "index": idx, "reason": "empty_sequence"})
            continue
        if len(candidate) % 4 != 0:
            warnings.append(
                {
                    "dataset": name,
                    "index": idx,
                    "reason": "length_not_divisible_by_4",
                    "length": len(candidate),
                }
            )
            continue
        try:
            flat = [int(x) for x in candidate]
        except Exception:
            warnings.append({"dataset": name, "index": idx, "reason": "non_integer_value"})
            continue
        events = [
            BehaviorEvent(day=flat[i], hour_slot=flat[i + 1], device=flat[i + 2], action=flat[i + 3])
            for i in range(0, len(flat), 4)
        ]
        seq = BehaviorSequence(events=events, sequence_id=str(idx), meta={"source_index": idx})
        sequences.append(seq)
        flat_sequences.append(flat)
    return sequences, flat_sequences, warnings


def parse_exclude_ids(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def event_is_excluded(ev: BehaviorEvent, level: str, exclude_ids: set[int]) -> bool:
    if not exclude_ids:
        return False
    action_excluded = isinstance(ev.action, int) and int(ev.action) in exclude_ids
    device_excluded = isinstance(ev.device, int) and int(ev.device) in exclude_ids
    if level == "action":
        return action_excluded
    if level == "device":
        return device_excluded
    if level == "device_action":
        return action_excluded or device_excluded
    raise ValueError(f"unknown level: {level}")


def filter_sequences_for_level(
    sequences: Sequence[BehaviorSequence],
    level: str,
    exclude_ids: set[int],
) -> list[BehaviorSequence]:
    if not exclude_ids:
        return list(sequences)
    filtered: list[BehaviorSequence] = []
    for seq in sequences:
        events = [ev for ev in seq.events if not event_is_excluded(ev, level, exclude_ids)]
        filtered.append(BehaviorSequence(events=events, sequence_id=seq.sequence_id, meta=seq.meta))
    return filtered


def event_positions(seq: BehaviorSequence) -> dict[str, list[tuple[int, int]]]:
    positions: dict[str, list[tuple[int, int]]] = {}
    for idx, ev in enumerate(seq.events):
        for level in ("action", "device", "device_action"):
            positions.setdefault(ev.key(level), []).append((idx, ev.time_index))
    return positions


def action_id_counter(sequences: Sequence[BehaviorSequence]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for seq in sequences:
        for ev in seq:
            if isinstance(ev.action, int):
                counts[int(ev.action)] += 1
    return counts


def action_id_overlap_report(datasets: dict[str, Sequence[BehaviorSequence]]) -> dict[str, Any]:
    counters = {name: action_id_counter(seqs) for name, seqs in datasets.items() if seqs is not None}
    sets = {name: set(counter) for name, counter in counters.items()}
    report: dict[str, Any] = {
        "counts": {f"{name}_action_id_count": len(ids) for name, ids in sets.items()},
        "total_occurrences": {f"{name}_action_occurrences": int(sum(counter.values())) for name, counter in counters.items()},
        "overlaps": {},
    }
    for left, right in (
        ("source", "target"),
        ("source", "raw"),
        ("source", "tof"),
        ("target", "tof"),
    ):
        if left not in sets or right not in sets:
            continue
        inter = sets[left] & sets[right]
        union = sets[left] | sets[right]
        report["overlaps"][f"{left}_{right}"] = {
            "overlap_count": len(inter),
            "jaccard": float(len(inter) / len(union)) if union else None,
            "only_left_count": len(sets[left] - sets[right]),
            "only_right_count": len(sets[right] - sets[left]),
            "overlap_ids": sorted(inter),
            "only_left_ids": sorted(sets[left] - sets[right]),
            "only_right_ids": sorted(sets[right] - sets[left]),
        }
    return report


def sequences_containing_key(sequences: Sequence[BehaviorSequence], key: str, level: str) -> list[str]:
    ids: list[str] = []
    for i, seq in enumerate(sequences):
        if any(ev.key(level) == key for ev in seq):
            ids.append(seq.sequence_id if seq.sequence_id is not None else str(i))
    return ids


def endpoint_coverage(
    edges: Sequence[dict[str, Any]],
    source_sequences: Sequence[BehaviorSequence],
    generated_sequences: Sequence[BehaviorSequence],
    level: str,
) -> list[dict[str, Any]]:
    source_counts = sequence_counts(source_sequences, level)
    generated_counts = sequence_counts(generated_sequences, level)
    generated_total = len(generated_sequences)
    rows: list[dict[str, Any]] = []
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        contains_source = set(sequences_containing_key(generated_sequences, src, level))
        contains_target = set(sequences_containing_key(generated_sequences, tgt, level))
        contains_both = contains_source & contains_target
        row = dict(edge)
        row.update(
            {
                "source_endpoint_count_in_source": int(source_counts.get(src, 0)),
                "target_endpoint_count_in_source": int(source_counts.get(tgt, 0)),
                "source_endpoint_count_in_generated_tof": int(generated_counts.get(src, 0)),
                "target_endpoint_count_in_generated_tof": int(generated_counts.get(tgt, 0)),
                "generated_sequences_containing_source": len(contains_source),
                "generated_sequences_containing_target": len(contains_target),
                "generated_sequences_containing_both": len(contains_both),
                "both_endpoint_cooccurrence_rate": float(len(contains_both) / generated_total) if generated_total else 0.0,
            }
        )
        rows.append(row)
    return rows


def load_smartgen_fr_mappings(dictionary_path: Path = DEFAULT_SMARTGEN_DICTIONARY) -> dict[str, Any]:
    if not dictionary_path.exists():
        return {"dictionary_path": str(dictionary_path), "found": False}
    spec = importlib.util.spec_from_file_location("smartgen_dictionary", dictionary_path)
    if spec is None or spec.loader is None:
        return {"dictionary_path": str(dictionary_path), "found": False}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action_map = getattr(module, "fr_actions", {})
    device_map = getattr(module, "fr_devices_dict", {})
    return {
        "dictionary_path": str(dictionary_path),
        "found": True,
        "action_id_to_name": {int(v): k for k, v in action_map.items()},
        "device_id_to_name": {int(v): k for k, v in device_map.items()},
        "max_action_id": max(action_map.values()) if action_map else None,
        "max_device_id": max(device_map.values()) if device_map else None,
        "num_actions": len(action_map),
        "num_devices": len(device_map),
        "pad_action_id_candidate": (max(action_map.values()) + 1) if action_map else None,
        "pad_device_id_candidate": (max(device_map.values()) + 1) if device_map else None,
    }


def action_id_diagnosis(
    action_id: int,
    datasets: dict[str, Sequence[BehaviorSequence]],
    mappings: dict[str, Any],
    exclude_ids: set[int],
) -> dict[str, Any]:
    action_map = mappings.get("action_id_to_name", {})
    mapped_name = action_map.get(action_id)
    counts: dict[str, int] = {}
    sequence_ids: dict[str, list[str]] = {}
    for name, seqs in datasets.items():
        ids: list[str] = []
        count = 0
        for i, seq in enumerate(seqs):
            seq_count = sum(1 for ev in seq if ev.action == action_id)
            if seq_count:
                ids.append(seq.sequence_id if seq.sequence_id is not None else str(i))
                count += seq_count
        counts[name] = count
        sequence_ids[name] = ids
    lower_name = (mapped_name or "").lower()
    pad_candidate = mappings.get("pad_action_id_candidate")
    possible_padding = action_id == pad_candidate or any(token in lower_name for token in ("pad", "padding", "unknown", "unk", "end", "eos"))
    reasons = []
    if mapped_name is None:
        reasons.append("action id is absent from SmartGen FR action dictionary")
    else:
        reasons.append(f"mapped to SmartGen FR action: {mapped_name}")
    if action_id == pad_candidate:
        reasons.append("matches max(fr_actions)+1 padding candidate")
    if action_id in exclude_ids:
        reasons.append("explicitly excluded by --exclude-ids")
    if not possible_padding and mapped_name is not None:
        reasons.append("does not look like padding/unknown/end token from dictionary heuristics")
    return {
        "action_id": action_id,
        "key": f"a:{action_id}",
        "mapped_name": mapped_name,
        "counts": counts,
        "sequence_ids": sequence_ids,
        "num_sequences": {name: len(ids) for name, ids in sequence_ids.items()},
        "possible_padding_unknown_end_token": bool(possible_padding or mapped_name is None),
        "reasons": reasons,
        "dictionary": {
            "path": mappings.get("dictionary_path"),
            "found": mappings.get("found", False),
            "num_actions": mappings.get("num_actions"),
            "max_action_id": mappings.get("max_action_id"),
            "pad_action_id_candidate": pad_candidate,
        },
    }


def score_sequence(
    seq: BehaviorSequence,
    edges: Sequence[dict[str, Any]],
    total_edge_weight: float,
    low_evidence_weight: float,
    index: int,
) -> dict[str, Any]:
    positions = event_positions(seq)
    checked_weight = 0.0
    satisfied_weight = 0.0
    violated_weight = 0.0
    checked_edges = 0
    satisfied_edges = 0
    violated_edges = 0

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        weight = float(edge["weight"])
        src_pos = positions.get(src, [])
        tgt_pos = positions.get(tgt, [])
        if not src_pos or not tgt_pos:
            continue
        checked_edges += 1
        checked_weight += weight
        lag = int(edge.get("lag") or 0)
        has_ordered_pair = any(src_idx < tgt_idx for src_idx, _ in src_pos for tgt_idx, _ in tgt_pos)
        has_lag_pair = any(
            0 <= tgt_time - src_time <= lag for _, src_time in src_pos for _, tgt_time in tgt_pos
        )
        if has_ordered_pair or has_lag_pair:
            satisfied_edges += 1
            satisfied_weight += weight
        elif any(tgt_idx < src_idx for src_idx, _ in src_pos for tgt_idx, _ in tgt_pos):
            violated_edges += 1
            violated_weight += weight

    coverage = satisfied_weight / (checked_weight + EPS)
    violation_rate = violated_weight / (checked_weight + EPS)
    low_evidence = checked_weight <= low_evidence_weight
    return {
        "index": index,
        "sequence_id": seq.sequence_id,
        "length": len(seq),
        "causal_coverage": float(coverage),
        "violation_rate": float(violation_rate),
        "causal_score": float(coverage * (1.0 - violation_rate)),
        "checked_edge_weight": float(checked_weight),
        "satisfied_edge_weight": float(satisfied_weight),
        "violated_edge_weight": float(violated_weight),
        "num_checked_edges": checked_edges,
        "num_satisfied_edges": satisfied_edges,
        "num_violated_edges": violated_edges,
        "low_evidence": bool(low_evidence),
        "low_evidence_threshold": float(low_evidence_weight),
        "total_top_edge_weight": float(total_edge_weight),
    }


def score_sequences(
    sequences: Sequence[BehaviorSequence],
    edges: Sequence[dict[str, Any]],
    total_edge_weight: float,
) -> list[dict[str, Any]]:
    low_evidence_weight = max(EPS, total_edge_weight * 0.01)
    return [
        score_sequence(seq, edges, total_edge_weight, low_evidence_weight, index=i)
        for i, seq in enumerate(sequences)
    ]


def sequence_counts(sequences: Sequence[BehaviorSequence], level: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        for ev in seq:
            counts[ev.key(level)] += 1
    return counts


def transition_counts(sequences: Sequence[BehaviorSequence], level: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        keys = [ev.key(level) for ev in seq]
        counts.update(f"{a}->{b}" for a, b in zip(keys, keys[1:]))
    return counts


def js_divergence(p_counts: Counter[str], q_counts: Counter[str]) -> float | None:
    keys = sorted(set(p_counts) | set(q_counts))
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    if not keys or p_total == 0 or q_total == 0:
        return None
    p = np.asarray([p_counts.get(k, 0) / p_total for k in keys], dtype=np.float64)
    q = np.asarray([q_counts.get(k, 0) / q_total for k in keys], dtype=np.float64)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence_array(p, m) + 0.5 * kl_divergence_array(q, m))


def kl_divergence(p_counts: Counter[str], q_counts: Counter[str]) -> float | None:
    keys = sorted(set(p_counts) | set(q_counts))
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    if not keys or p_total == 0 or q_total == 0:
        return None
    alpha = 1e-9
    p = np.asarray([p_counts.get(k, 0) + alpha for k in keys], dtype=np.float64)
    q = np.asarray([q_counts.get(k, 0) + alpha for k in keys], dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    return float(kl_divergence_array(p, q))


def kl_divergence_array(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def metric_summary(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {
            "num_sequences": 0,
            "mean_causal_coverage": None,
            "median_causal_coverage": None,
            "mean_violation_rate": None,
            "median_violation_rate": None,
            "mean_causal_score": None,
            "low_evidence_rate": None,
        }
    coverages = [float(s["causal_coverage"]) for s in scores]
    violations = [float(s["violation_rate"]) for s in scores]
    causal_scores = [float(s["causal_score"]) for s in scores]
    return {
        "num_sequences": len(scores),
        "mean_causal_coverage": float(mean(coverages)),
        "median_causal_coverage": float(median(coverages)),
        "mean_violation_rate": float(mean(violations)),
        "median_violation_rate": float(median(violations)),
        "mean_causal_score": float(mean(causal_scores)),
        "low_evidence_rate": float(sum(1 for s in scores if s["low_evidence"]) / len(scores)),
    }


def distribution_report(
    source: Sequence[BehaviorSequence],
    target: Sequence[BehaviorSequence] | None,
    datasets: dict[str, Sequence[BehaviorSequence]],
    transition_level: str,
) -> dict[str, Any]:
    if not target:
        return {}
    target_action = sequence_counts(target, "action")
    target_device = sequence_counts(target, "device")
    target_transition = transition_counts(target, transition_level)
    result: dict[str, Any] = {}
    all_datasets = {"source": source, **datasets}
    for name, seqs in all_datasets.items():
        action = sequence_counts(seqs, "action")
        device = sequence_counts(seqs, "device")
        trans = transition_counts(seqs, transition_level)
        result[name] = {
            "action_js_to_target": js_divergence(action, target_action),
            "action_kl_to_target": kl_divergence(action, target_action),
            "device_js_to_target": js_divergence(device, target_device),
            "device_kl_to_target": kl_divergence(device, target_device),
            "transition_level": transition_level,
            "transition_js_to_target": js_divergence(trans, target_transition),
            "transition_kl_to_target": kl_divergence(trans, target_transition),
        }
    return result


def add_rank_and_counts(edges: Sequence[dict[str, Any]], source_sequences: Sequence[BehaviorSequence], level: str) -> list[dict[str, Any]]:
    counts = sequence_counts(source_sequences, level)
    enriched = []
    for rank, edge in enumerate(edges, start=1):
        item = dict(edge)
        item["rank"] = rank
        item["source_count"] = int(counts.get(edge["source"], 0))
        item["target_count"] = int(counts.get(edge["target"], 0))
        enriched.append(item)
    return enriched


def apply_mode(
    mode: str,
    sequences: Sequence[BehaviorSequence],
    scores: Sequence[dict[str, Any]],
    min_coverage: float,
) -> tuple[list[BehaviorSequence], dict[str, Any]]:
    if mode == "score_only":
        return list(sequences), {
            "mode": mode,
            "kept_rate": 1.0,
            "num_before": len(sequences),
            "num_after": len(sequences),
        }
    if mode == "mild_filter":
        kept: list[BehaviorSequence] = []
        rejected_scores: list[dict[str, Any]] = []
        for seq, score in zip(sequences, scores):
            reject = (not score["low_evidence"]) and float(score["causal_score"]) < min_coverage
            if reject:
                rejected_scores.append(score)
            else:
                kept.append(seq)
        kept_rate = len(kept) / len(sequences) if sequences else 0.0
        mean_rejected_violation = (
            float(mean(float(score.get("violation_rate", 0.0)) for score in rejected_scores))
            if rejected_scores
            else None
        )
        return kept, {
            "mode": mode,
            "min_coverage": float(min_coverage),
            "num_before": len(sequences),
            "num_after": len(kept),
            "num_deleted": len(rejected_scores),
            "kept_rate": float(kept_rate),
            "deleted_mean_violation": mean_rejected_violation,
            "deleted_low_evidence_count": int(sum(1 for score in rejected_scores if score["low_evidence"])),
        }
    if mode == "resample_soft":
        out: list[BehaviorSequence] = []
        num_duplicated = 0
        num_kept_once = 0
        num_dropped = 0
        for seq, score in zip(sequences, scores):
            causal_score = float(score["causal_score"])
            if score["low_evidence"]:
                out.append(seq)
                num_kept_once += 1
            elif causal_score >= 0.6:
                out.extend([seq, seq])
                num_duplicated += 1
            elif causal_score >= 0.3:
                out.append(seq)
                num_kept_once += 1
            else:
                num_dropped += 1
        return out, {
            "mode": mode,
            "num_before": len(sequences),
            "resampled_size": len(out),
            "expansion_rate": float(len(out) / len(sequences)) if sequences else 0.0,
            "num_duplicated": num_duplicated,
            "num_kept_once": num_kept_once,
            "num_dropped": num_dropped,
        }
    raise ValueError(f"unsupported mode: {mode}")


def build_markdown(report: dict[str, Any], top_edges: Sequence[dict[str, Any]], warnings: Sequence[Any]) -> str:
    lines = [
        "# SmartGen + GCAD Generation Quality Report",
        "",
        "## Basic Counts",
        "",
        "| item | value |",
        "| --- | ---: |",
    ]
    for key, value in report["basic_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Causal Metrics", "", "| dataset | mean coverage | median coverage | mean violation | median violation | low evidence |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, metrics in report["causal_metrics"].items():
        lines.append(
            "| {name} | {mean_cov} | {median_cov} | {mean_vio} | {median_vio} | {low} |".format(
                name=name,
                mean_cov=fmt(metrics.get("mean_causal_coverage")),
                median_cov=fmt(metrics.get("median_causal_coverage")),
                mean_vio=fmt(metrics.get("mean_violation_rate")),
                median_vio=fmt(metrics.get("median_violation_rate")),
                low=fmt(metrics.get("low_evidence_rate")),
            )
        )

    if report.get("distribution_metrics"):
        lines.extend(["", "## Distribution Metrics", "", "| dataset | action JS | device JS | transition JS | transition KL |", "| --- | ---: | ---: | ---: | ---: |"])
        for name, metrics in report["distribution_metrics"].items():
            lines.append(
                f"| {name} | {fmt(metrics.get('action_js_to_target'))} | {fmt(metrics.get('device_js_to_target'))} | "
                f"{fmt(metrics.get('transition_js_to_target'))} | {fmt(metrics.get('transition_kl_to_target'))} |"
            )

    id_space = report.get("id_space_diagnostics", {})
    if id_space:
        lines.extend(["", "## Action ID Space", "", "| item | value |", "| --- | ---: |"])
        for key, value in id_space.get("counts", {}).items():
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "| pair | overlap | jaccard | only left | only right |", "| --- | ---: | ---: | ---: | ---: |"])
        for pair, metrics in id_space.get("overlaps", {}).items():
            lines.append(
                f"| {pair} | {metrics['overlap_count']} | {fmt(metrics['jaccard'])} | "
                f"{metrics['only_left_count']} | {metrics['only_right_count']} |"
            )

    lines.extend(["", "## Top Causal Edges", "", "| rank | source | target | weight | source_count | target_count |", "| ---: | --- | --- | ---: | ---: | ---: |"])
    for edge in top_edges[:30]:
        lines.append(
            f"| {edge['rank']} | {edge['source']} | {edge['target']} | {fmt(edge['weight'])} | "
            f"{edge['source_count']} | {edge['target_count']} |"
        )

    endpoint_rows = report.get("endpoint_coverage_topk", [])
    if endpoint_rows:
        lines.extend(
            [
                "",
                "## Endpoint Coverage",
                "",
                "| rank | edge | src in source | tgt in source | src in tof | tgt in tof | seq src | seq tgt | seq both | both rate |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for edge in endpoint_rows[:30]:
            lines.append(
                f"| {edge['rank']} | {edge['source']}->{edge['target']} | "
                f"{edge['source_endpoint_count_in_source']} | {edge['target_endpoint_count_in_source']} | "
                f"{edge['source_endpoint_count_in_generated_tof']} | {edge['target_endpoint_count_in_generated_tof']} | "
                f"{edge['generated_sequences_containing_source']} | {edge['generated_sequences_containing_target']} | "
                f"{edge['generated_sequences_containing_both']} | {fmt(edge['both_endpoint_cooccurrence_rate'])} |"
            )

    a193 = report.get("action_193_diagnosis")
    if a193:
        lines.extend(["", "## a:193 Diagnosis", ""])
        lines.append(f"- mapped_name: `{a193.get('mapped_name')}`")
        lines.append(f"- possible_padding_unknown_end_token: `{a193.get('possible_padding_unknown_end_token')}`")
        lines.append(f"- counts: `{a193.get('counts')}`")
        lines.append(f"- num_sequences: `{a193.get('num_sequences')}`")
        for reason in a193.get("reasons", []):
            lines.append(f"- {reason}")

    report_warnings = report.get("warnings", [])
    if warnings or report_warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report_warnings:
            lines.append(f"- {warning}")
        for warning in warnings[:50]:
            lines.append(f"- loader: `{warning}`")
        if len(warnings) > 50:
            lines.append(f"- loader: truncated {len(warnings) - 50} additional warnings")

    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.6f}"
    return str(value)


def write_score_csv(path: str | Path, scores: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "length",
        "causal_coverage",
        "violation_rate",
        "causal_score",
        "checked_edge_weight",
        "satisfied_edge_weight",
        "violated_edge_weight",
        "num_checked_edges",
        "low_evidence",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            writer.writerow({key: score.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate full SmartGen outputs with GCAD-style causal scoring")
    parser.add_argument("--source-train-pkl", required=True)
    parser.add_argument("--target-real-pkl")
    parser.add_argument("--smartgen-raw-pkl")
    parser.add_argument("--smartgen-tof-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--level", choices=["action", "device", "device_action"], default="action")
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sparse-threshold", type=float, default=0.001)
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--mode", choices=["score_only", "mild_filter", "resample_soft"], default="score_only")
    parser.add_argument("--top-k-edges", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--exclude-ids", default="", help="comma-separated ids to exclude at the selected level, e.g. 0,193")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    exclude_ids = parse_exclude_ids(args.exclude_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_sequences_original, source_raw, source_warnings = load_sequences(args.source_train_pkl, "source_train")
    target_sequences_original = None
    target_warnings: list[dict[str, Any]] = []
    if args.target_real_pkl:
        target_sequences_original, _, target_warnings = load_sequences(args.target_real_pkl, "target_real")
    raw_sequences_original = None
    raw_warnings: list[dict[str, Any]] = []
    if args.smartgen_raw_pkl:
        raw_sequences_original, _, raw_warnings = load_sequences(args.smartgen_raw_pkl, "smartgen_raw")
    tof_sequences_original, tof_raw, tof_warnings = load_sequences(args.smartgen_tof_pkl, "smartgen_tof")
    loader_warnings = source_warnings + target_warnings + raw_warnings + tof_warnings
    write_json(out_dir / "loader_warnings.json", loader_warnings)

    if not source_sequences_original:
        raise ValueError("No valid source_train sequences after loading.")
    if not tof_sequences_original:
        raise ValueError("No valid SmartGen TOF sequences after loading.")

    original_datasets: dict[str, Sequence[BehaviorSequence]] = {
        "source": source_sequences_original,
        "tof": tof_sequences_original,
    }
    if target_sequences_original is not None:
        original_datasets["target"] = target_sequences_original
    if raw_sequences_original is not None:
        original_datasets["raw"] = raw_sequences_original

    source_sequences = filter_sequences_for_level(source_sequences_original, args.level, exclude_ids)
    target_sequences = (
        filter_sequences_for_level(target_sequences_original, args.level, exclude_ids)
        if target_sequences_original is not None
        else None
    )
    raw_sequences = (
        filter_sequences_for_level(raw_sequences_original, args.level, exclude_ids)
        if raw_sequences_original is not None
        else None
    )
    tof_sequences = filter_sequences_for_level(tof_sequences_original, args.level, exclude_ids)
    effective_datasets: dict[str, Sequence[BehaviorSequence]] = {
        "source": source_sequences,
        "tof": tof_sequences,
    }
    if target_sequences is not None:
        effective_datasets["target"] = target_sequences
    if raw_sequences is not None:
        effective_datasets["raw"] = raw_sequences

    tensorizer = EventTensorizer(level=args.level, count_mode="binary", decay=0.2)
    tensorized = tensorizer.fit_transform(source_sequences)
    miner = GradientCausalMiner(
        lag=args.lag,
        epochs=args.epochs,
        sparse_threshold=args.sparse_threshold,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    prior = miner.fit_prior(tensorized.tensor, tensorized.channel_to_key, sample_limit=args.sample_limit)
    prior.meta = prior.meta or {}
    prior.meta.update(
        {
            "source_train_pkl": str(Path(args.source_train_pkl).resolve()),
            "level": args.level,
            "num_source_train": len(source_sequences),
            "seed": args.seed,
            "exclude_ids": sorted(exclude_ids),
        }
    )
    prior_path = out_dir / "causal_prior_source.json"
    prior.save(prior_path)

    edges = prior.top_edges(k=args.top_k_edges, include_self=False)
    top_edges = add_rank_and_counts(edges, source_sequences, args.level)
    endpoint_rows = endpoint_coverage(top_edges, source_sequences, tof_sequences, args.level)
    write_json(out_dir / "causal_edges_topk.json", top_edges)
    write_json(out_dir / "causal_edges_endpoint_coverage.json", endpoint_rows)
    total_edge_weight = sum(float(edge["weight"]) for edge in edges)

    tof_scores = score_sequences(tof_sequences, edges, total_edge_weight)
    write_json(out_dir / "smartgen_tof_causal_scores.json", tof_scores)
    write_score_csv(out_dir / "smartgen_tof_causal_scores.csv", tof_scores)
    save_pickle(
        out_dir / "smartgen_tof_causal_scored.pkl",
        [{"sequence": flat, "score": score} for flat, score in zip(tof_raw, tof_scores)],
    )

    raw_scores = score_sequences(raw_sequences, edges, total_edge_weight) if raw_sequences is not None else None
    if raw_scores is not None:
        write_json(out_dir / "smartgen_raw_causal_scores.json", raw_scores)

    gcad_sequences, mode_summary = apply_mode(args.mode, tof_sequences, tof_scores, args.min_coverage)
    gcad_scores = score_sequences(gcad_sequences, edges, total_edge_weight)
    if args.mode == "mild_filter":
        save_pickle(out_dir / "smartgen_tof_gcad_mild.pkl", dump_numeric_sequences(gcad_sequences))
    elif args.mode == "resample_soft":
        save_pickle(out_dir / "smartgen_tof_gcad_resampled.pkl", dump_numeric_sequences(gcad_sequences))

    causal_metrics: dict[str, Any] = {
        "smartgen_tof": metric_summary(tof_scores),
        "smartgen_tof_gcad": metric_summary(gcad_scores),
    }
    if raw_scores is not None:
        causal_metrics = {"smartgen_raw": metric_summary(raw_scores), **causal_metrics}

    distribution_datasets: dict[str, Sequence[BehaviorSequence]] = {
        "smartgen_tof": tof_sequences,
        "smartgen_tof_gcad": gcad_sequences,
    }
    if raw_sequences is not None:
        distribution_datasets = {"smartgen_raw": raw_sequences, **distribution_datasets}
    transition_level = "device_action" if args.level == "device_action" else "action"
    distribution_metrics = distribution_report(source_sequences, target_sequences, distribution_datasets, transition_level)
    id_space_original = action_id_overlap_report(original_datasets)
    id_space_effective = action_id_overlap_report(effective_datasets)
    mappings = load_smartgen_fr_mappings()
    action_193 = action_id_diagnosis(193, original_datasets, mappings, exclude_ids)

    warnings: list[str] = []
    kept_rate = mode_summary.get("kept_rate")
    if kept_rate is not None and kept_rate < 0.7:
        warnings.append("GCAD mild filter may be too aggressive.")
    tof_dist = distribution_metrics.get("smartgen_tof", {}) if distribution_metrics else {}
    gcad_dist = distribution_metrics.get("smartgen_tof_gcad", {}) if distribution_metrics else {}
    for key in ("action_js_to_target", "device_js_to_target", "transition_js_to_target"):
        tof_js = tof_dist.get(key)
        gcad_js = gcad_dist.get(key)
        if tof_js is not None and gcad_js is not None and gcad_js > tof_js * 1.1:
            warnings.append("Causal enhancement may hurt target-context distribution.")
            break

    basic_counts = {
        "num_source_train": len(source_sequences),
        "num_target_real": len(target_sequences) if target_sequences is not None else None,
        "num_smartgen_raw": len(raw_sequences) if raw_sequences is not None else None,
        "num_smartgen_tof": len(tof_sequences),
        "num_source_train_original": len(source_sequences_original),
        "num_target_real_original": len(target_sequences_original) if target_sequences_original is not None else None,
        "num_smartgen_raw_original": len(raw_sequences_original) if raw_sequences_original is not None else None,
        "num_smartgen_tof_original": len(tof_sequences_original),
    }
    if args.mode == "mild_filter":
        basic_counts["num_after_gcad_mild"] = len(gcad_sequences)
    elif args.mode == "resample_soft":
        basic_counts["num_after_gcad_resampled"] = len(gcad_sequences)

    report = {
        "inputs": {
            "source_train_pkl": str(Path(args.source_train_pkl).resolve()),
            "target_real_pkl": str(Path(args.target_real_pkl).resolve()) if args.target_real_pkl else None,
            "smartgen_raw_pkl": str(Path(args.smartgen_raw_pkl).resolve()) if args.smartgen_raw_pkl else None,
            "smartgen_tof_pkl": str(Path(args.smartgen_tof_pkl).resolve()),
        },
        "settings": {
            "level": args.level,
            "lag": args.lag,
            "epochs": args.epochs,
            "sparse_threshold": args.sparse_threshold,
            "min_coverage": args.min_coverage,
            "mode": args.mode,
            "top_k_edges": args.top_k_edges,
            "seed": args.seed,
            "sample_limit": args.sample_limit,
            "exclude_ids": sorted(exclude_ids),
        },
        "basic_counts": basic_counts,
        "mode_summary": mode_summary,
        "causal_metrics": causal_metrics,
        "distribution_metrics": distribution_metrics,
        "id_space_diagnostics": id_space_original,
        "id_space_diagnostics_effective_after_exclusions": id_space_effective,
        "endpoint_coverage_topk": endpoint_rows[:50],
        "action_193_diagnosis": action_193,
        "top_edges": top_edges[:30],
        "warnings": warnings,
        "loader_warning_count": len(loader_warnings),
        "artifacts": {
            "causal_prior_source": str(prior_path),
            "causal_edges_topk": str(out_dir / "causal_edges_topk.json"),
            "causal_edges_endpoint_coverage": str(out_dir / "causal_edges_endpoint_coverage.json"),
            "smartgen_tof_causal_scores": str(out_dir / "smartgen_tof_causal_scores.json"),
        },
    }
    write_json(out_dir / "generation_quality_report.json", report)
    (out_dir / "generation_quality_report.md").write_text(
        build_markdown(report, top_edges, loader_warnings),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out_dir / "generation_quality_report.json"), "mode_summary": mode_summary}, indent=2))


if __name__ == "__main__":
    main()
