"""Gradient-based Granger-causality discovery public API.

The implementation remains shared with the historical ``causal_prior`` module
so old pickle/import paths continue to work.  New code should import from this
module; the compatibility layer can be removed only in a future protocol
version.
"""

from ...causal_prior import CausalPrior, GradientCausalMiner

__all__ = ["CausalPrior", "GradientCausalMiner"]
