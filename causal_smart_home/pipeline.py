from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

from .schema import BehaviorSequence
from .event_tensor import EventTensorizer
from .gcad_adapter import GCADAdapter
from .causal_prior import CausalPrior
from .causal_prompt import build_causal_smartgen_prompt
from .causal_filter import CausalConsistencyFilter, FilterResult
from .smartgen_adapter import SmartGenAdapter


@dataclass
class GluePipelineArtifacts:
    prior_path: Path
    prompt_path: Path
    scores_path: Path | None = None
    kept_path: Path | None = None


class CausalSmartHomePipeline:
    """High-level orchestration of the glue layer."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_prior(
        self,
        normal_sequences: Sequence[BehaviorSequence],
        lag: int = 4,
        epochs: int = 80,
        sparse_threshold: float = 0.0,
        level: str = "action",
        sample_limit: int | None = 64,
    ) -> CausalPrior:
        tensorizer = EventTensorizer(level=level, count_mode="binary", decay=0.2)
        tensorized = tensorizer.fit_transform(normal_sequences)
        adapter = GCADAdapter()
        prior = adapter.mine_event_prior(
            tensorized.tensor,
            channel_to_key=tensorized.channel_to_key,
            lag=lag,
            epochs=epochs,
            sparse_threshold=sparse_threshold,
            sample_limit=sample_limit,
        )
        prior.meta = prior.meta or {}
        prior.meta.update({"level": level, "num_sequences": len(normal_sequences)})
        return prior

    def build_prompt(
        self,
        compressed_sequences: Sequence[BehaviorSequence],
        prior: CausalPrior,
        device_information: dict[str, Any] | str,
        original_context: str,
        new_context: str,
    ) -> str:
        transition_hints = SmartGenAdapter.build_transition_hints(compressed_sequences)
        return build_causal_smartgen_prompt(
            original_sequences=compressed_sequences,
            prior=prior,
            device_information=device_information,
            original_context=original_context,
            new_context=new_context,
            transition_hints=transition_hints,
        )

    def filter_generated(
        self,
        generated_sequences: Sequence[BehaviorSequence],
        prior: CausalPrior,
        min_coverage: float = 0.5,
    ) -> FilterResult:
        return CausalConsistencyFilter(prior).filter(generated_sequences, min_coverage=min_coverage)
