from __future__ import annotations

from typing import Sequence
import numpy as np

from .causal_prior import GradientCausalMiner, CausalPrior


class CausalRelationAdapter:
    """Adapter for the compact causal-relation prior miner."""

    def mine_event_prior(
        self,
        tensor: np.ndarray,
        channel_to_key: Sequence[str],
        lag: int = 4,
        epochs: int = 80,
        hidden: int = 16,
        sparse_threshold: float = 0.0,
        batch_size: int = 64,
        sample_limit: int | None = None,
    ) -> CausalPrior:
        miner = GradientCausalMiner(
            lag=lag,
            epochs=epochs,
            hidden=hidden,
            sparse_threshold=sparse_threshold,
            batch_size=batch_size,
        )
        return miner.fit_prior(tensor, channel_to_key=channel_to_key, sample_limit=sample_limit)
