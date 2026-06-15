"""Causal Smart Home.

This package is intentionally a glue layer. It wraps or consumes outputs from
SmartGuard, SmartGen and GCAD-style causal mining without editing those projects.
"""

from .schema import BehaviorEvent, BehaviorSequence
from .event_tensor import EventTensorizer
from .causal_prior import GradientCausalMiner, CausalPrior
from .causal_filter import CausalConsistencyFilter

__all__ = [
    "BehaviorEvent",
    "BehaviorSequence",
    "EventTensorizer",
    "GradientCausalMiner",
    "CausalPrior",
    "CausalConsistencyFilter",
]
