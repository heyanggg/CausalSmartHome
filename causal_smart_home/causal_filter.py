from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any
import json
from pathlib import Path

import numpy as np

from .schema import BehaviorSequence, load_numeric_sequences, dump_numeric_sequences
from .causal_prior import CausalPrior


@dataclass
class FilterResult:
    kept: list[BehaviorSequence]
    rejected: list[BehaviorSequence]
    scores: list[dict[str, Any]]


class CausalConsistencyFilter:
    """Rank/filter generated sequences by subsequence-level causal consistency."""

    def __init__(self, prior: CausalPrior, top_k_edges: int = 30, min_edge_weight: float | None = None) -> None:
        self.prior = prior
        self.edges = prior.top_edges(k=top_k_edges, min_weight=min_edge_weight, include_self=False)

    @staticmethod
    def _event_positions(seq: BehaviorSequence) -> dict[str, list[int]]:
        positions: dict[str, list[int]] = {}
        for idx, ev in enumerate(seq.events):
            for key in (ev.key("action"), ev.key("device"), ev.key("device_action")):
                positions.setdefault(key, []).append(idx)
        return positions

    def score_sequence(self, seq: BehaviorSequence) -> dict[str, Any]:
        positions = self._event_positions(seq)
        total_weight = 0.0
        satisfied_weight = 0.0
        violated = []
        matched = []
        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
            w = float(edge["weight"])
            src_pos = positions.get(src, [])
            tgt_pos = positions.get(tgt, [])
            if not src_pos or not tgt_pos:
                # missing causal variables are treated as weak evidence, not a hard violation,
                # because context shift may make a behavior disappear.
                continue
            total_weight += w
            ok = any(i < j for i in src_pos for j in tgt_pos)
            if ok:
                satisfied_weight += w
                matched.append({"source": src, "target": tgt, "weight": w})
            else:
                violated.append({"source": src, "target": tgt, "weight": w})
        coverage = satisfied_weight / total_weight if total_weight > 0 else 1.0
        return {
            "sequence_id": seq.sequence_id,
            "length": len(seq),
            "causal_coverage": float(coverage),
            "matched_edges": matched,
            "violated_edges": violated,
            "num_checked_edges": len(matched) + len(violated),
        }

    def filter(self, sequences: Sequence[BehaviorSequence], min_coverage: float = 0.5) -> FilterResult:
        kept: list[BehaviorSequence] = []
        rejected: list[BehaviorSequence] = []
        scores = []
        for seq in sequences:
            score = self.score_sequence(seq)
            scores.append(score)
            if score["causal_coverage"] >= min_coverage:
                kept.append(seq)
            else:
                rejected.append(seq)
        return FilterResult(kept=kept, rejected=rejected, scores=scores)

    def save_scores(self, scores: list[dict[str, Any]], path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
