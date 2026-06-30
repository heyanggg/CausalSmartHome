"""CausalSmartHome 主实验包。

这里导出的对象只保留最常用于交互式检查的基础类型：行为事件、行为序列、
事件张量化器以及因果先验挖掘器。完整实验流程的入口分散在各个模块和
``scripts/`` 脚本里，这样每一步输入、输出和实验溯源都更清楚。
"""

from .schema import BehaviorEvent, BehaviorSequence
from .event_tensor import EventTensorizer
from .causal_prior import GradientCausalMiner, CausalPrior

__all__ = [
    "BehaviorEvent",
    "BehaviorSequence",
    "EventTensorizer",
    "GradientCausalMiner",
    "CausalPrior",
]
