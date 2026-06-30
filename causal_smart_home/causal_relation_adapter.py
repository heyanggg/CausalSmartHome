"""causal-relation/GCAD 先验挖掘的适配层。

项目其他模块只依赖这个 adapter，而不直接绑定某个 GCAD 实现。当前版本把调用
转发到本地轻量 ``GradientCausalMiner``，同时保留稳定接口；以后如果要接入
外部 causal-relation 仓库，只需要在这里替换实现，后面的 GSS 融合、TOF 和
下游 AD 代码都不用动。
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from .causal_prior import GradientCausalMiner, CausalPrior


class CausalRelationAdapter:
    """本地轻量 causal-relation 先验挖掘器的统一入口。"""

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
        """从已经张量化的事件序列中挖掘可序列化的因果先验。"""
        miner = GradientCausalMiner(
            lag=lag,
            epochs=epochs,
            hidden=hidden,
            sparse_threshold=sparse_threshold,
            batch_size=batch_size,
        )
        return miner.fit_prior(tensor, channel_to_key=channel_to_key, sample_limit=sample_limit)
