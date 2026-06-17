from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys
import numpy as np

from .causal_prior import GradientCausalMiner, CausalPrior


class GCADAdapter:
    """Adapter for GCAD or a compact GCAD-style fallback.

    The original GCAD code is written as an experiment script around train.csv
    and test.csv. For glue use, the safest non-invasive route is to either:
    1) call the original CLI on prepared CSV folders, or
    2) use the compact GradientCausalMiner implemented in this package for
       event tensors, which keeps the GCAD idea but does not edit GCAD code.
    """

    def __init__(self, gcad_root: str | None = None) -> None:
        self.gcad_root = Path(gcad_root).resolve() if gcad_root else None
        if self.gcad_root:
            sys.path.insert(0, str(self.gcad_root))

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
