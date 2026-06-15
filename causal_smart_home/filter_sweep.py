from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import pickle

import numpy as np

from .causal_filter import CausalConsistencyFilter
from .causal_prior import CausalPrior
from .schema import BehaviorSequence, dump_numeric_sequences


@dataclass(frozen=True)
class FilterSweepConfig:
    top_k_edges: int
    min_coverage: float
    min_checked_edges: int = 0
    min_edge_weight: float | None = None

    @property
    def slug(self) -> str:
        parts = [
            f"k{self.top_k_edges}",
            f"cov{_float_slug(self.min_coverage)}",
            f"chk{self.min_checked_edges}",
        ]
        if self.min_edge_weight is not None:
            parts.append(f"w{_float_slug(self.min_edge_weight)}")
        return "_".join(parts)


def _float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _save_pkl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def dump_normalized_numeric_sequences(
    sequences: Sequence[BehaviorSequence],
    sequence_length: int | None = None,
    pad_value: int = 0,
) -> list[list[int]]:
    raw = dump_numeric_sequences(sequences)
    if sequence_length is None:
        return raw
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sequence_length % 4 != 0:
        raise ValueError("sequence_length must be divisible by 4 for [day,hour,device,action] records")
    normalized: list[list[int]] = []
    for seq in raw:
        if len(seq) >= sequence_length:
            normalized.append(seq[:sequence_length])
        else:
            normalized.append(seq + [pad_value] * (sequence_length - len(seq)))
    return normalized


def build_filter_sweep_configs(
    top_k_edges: Sequence[int],
    min_coverages: Sequence[float],
    min_checked_edges: Sequence[int],
    min_edge_weights: Sequence[float | None] | None = None,
) -> list[FilterSweepConfig]:
    weights = list(min_edge_weights) if min_edge_weights is not None else [None]
    configs: list[FilterSweepConfig] = []
    for top_k in top_k_edges:
        for min_coverage in min_coverages:
            for min_checked in min_checked_edges:
                for min_weight in weights:
                    configs.append(
                        FilterSweepConfig(
                            top_k_edges=int(top_k),
                            min_coverage=float(min_coverage),
                            min_checked_edges=int(min_checked),
                            min_edge_weight=None if min_weight is None else float(min_weight),
                        )
                    )
    return configs


def apply_sweep_config(
    prior: CausalPrior,
    sequences: Sequence[BehaviorSequence],
    config: FilterSweepConfig,
) -> tuple[list[BehaviorSequence], list[BehaviorSequence], list[dict[str, Any]]]:
    scorer = CausalConsistencyFilter(
        prior,
        top_k_edges=config.top_k_edges,
        min_edge_weight=config.min_edge_weight,
    )
    kept: list[BehaviorSequence] = []
    rejected: list[BehaviorSequence] = []
    scores: list[dict[str, Any]] = []
    for seq in sequences:
        score = scorer.score_sequence(seq)
        reject = (
            score["num_checked_edges"] >= config.min_checked_edges
            and score["causal_coverage"] < config.min_coverage
        )
        score = dict(score)
        score["decision"] = "rejected" if reject else "kept"
        scores.append(score)
        if reject:
            rejected.append(seq)
        else:
            kept.append(seq)
    return kept, rejected, scores


def summarize_sweep_result(
    config: FilterSweepConfig,
    scores: Sequence[dict[str, Any]],
    kept_path: str | None = None,
    scores_path: str | None = None,
    sequence_length: int | None = None,
    pad_value: int = 0,
) -> dict[str, Any]:
    total = len(scores)
    rejected_ids = [str(s.get("sequence_id")) for s in scores if s.get("decision") == "rejected"]
    kept = total - len(rejected_ids)
    checked_counts = np.asarray([int(s.get("num_checked_edges", 0)) for s in scores], dtype=np.int64)
    coverages = np.asarray([float(s.get("causal_coverage", 1.0)) for s in scores], dtype=np.float32)
    checked_mask = checked_counts > 0
    top_violated_edges = _top_violated_edges(scores)
    row: dict[str, Any] = {
        "top_k_edges": config.top_k_edges,
        "min_coverage": config.min_coverage,
        "min_checked_edges": config.min_checked_edges,
        "min_edge_weight": config.min_edge_weight,
        "raw": total,
        "kept": kept,
        "rejected": len(rejected_ids),
        "reject_ratio": float(len(rejected_ids) / total) if total else 0.0,
        "checked_nonzero": int(np.sum(checked_mask)),
        "checked_nonzero_ratio": float(np.mean(checked_mask)) if total else 0.0,
        "checked_total": int(np.sum(checked_counts)) if total else 0,
        "checked_mean": float(np.mean(checked_counts)) if total else 0.0,
        "checked_max": int(np.max(checked_counts)) if total else 0,
        "coverage_mean": float(np.mean(coverages)) if total else 1.0,
        "coverage_median": float(np.median(coverages)) if total else 1.0,
        "checked_coverage_mean": float(np.mean(coverages[checked_mask])) if np.any(checked_mask) else 1.0,
        "checked_coverage_median": float(np.median(coverages[checked_mask])) if np.any(checked_mask) else 1.0,
        "top_violated_edge": top_violated_edges[0]["edge"] if top_violated_edges else "",
        "top_violated_edge_count": top_violated_edges[0]["count"] if top_violated_edges else 0,
        "sequence_length": sequence_length,
        "pad_value": pad_value if sequence_length is not None else "",
        "kept_path": kept_path or "",
        "scores_path": scores_path or "",
        "slug": config.slug,
        "rejected_ids": rejected_ids,
        "top_violated_edges": top_violated_edges,
    }
    return row


def _top_violated_edges(scores: Sequence[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for score in scores:
        for edge in score.get("violated_edges", []):
            key = f"{edge['source']}->{edge['target']}"
            item = stats.setdefault(
                key,
                {
                    "edge": key,
                    "source": edge["source"],
                    "target": edge["target"],
                    "count": 0,
                    "weight_sum": 0.0,
                },
            )
            item["count"] += 1
            item["weight_sum"] += float(edge["weight"])
    ranked = sorted(stats.values(), key=lambda x: (x["count"], x["weight_sum"]), reverse=True)
    return ranked[:limit]


CSV_FIELDS = [
    "slug",
    "top_k_edges",
    "min_coverage",
    "min_checked_edges",
    "min_edge_weight",
    "raw",
    "kept",
    "rejected",
    "reject_ratio",
    "checked_nonzero",
    "checked_nonzero_ratio",
    "checked_total",
    "checked_mean",
    "checked_max",
    "coverage_mean",
    "coverage_median",
    "checked_coverage_mean",
    "checked_coverage_median",
    "top_violated_edge",
    "top_violated_edge_count",
    "sequence_length",
    "pad_value",
    "kept_path",
    "scores_path",
]


def write_sweep_summary(rows: Sequence[dict[str, Any]], out_dir: str | Path, prefix: str = "filter_sweep") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_summary.csv"
    json_path = out / f"{prefix}_summary.json"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def run_filter_sweep(
    prior: CausalPrior,
    sequences: Sequence[BehaviorSequence],
    configs: Sequence[FilterSweepConfig],
    out_dir: str | Path,
    tag: str = "generated",
    write_kept: bool = True,
    write_scores: bool = False,
    summary_prefix: str = "filter_sweep",
    sequence_length: int | None = None,
    pad_value: int = 0,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config in configs:
        kept, _rejected, scores = apply_sweep_config(prior, sequences, config)
        kept_path: Path | None = None
        scores_path: Path | None = None
        if write_kept:
            kept_path = out / f"{tag}_{config.slug}_kept.pkl"
            _save_pkl(
                kept_path,
                dump_normalized_numeric_sequences(
                    kept,
                    sequence_length=sequence_length,
                    pad_value=pad_value,
                ),
            )
        if write_scores:
            scores_path = out / f"{tag}_{config.slug}_scores.json"
            scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            summarize_sweep_result(
                config,
                scores,
                kept_path=str(kept_path) if kept_path else None,
                scores_path=str(scores_path) if scores_path else None,
                sequence_length=sequence_length,
                pad_value=pad_value,
            )
        )
    write_sweep_summary(rows, out, prefix=summary_prefix)
    return rows
