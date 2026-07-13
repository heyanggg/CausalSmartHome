"""Target-aware causal-prior adaptation."""

from .target_guard import (
    TargetDistributionGuardConfig,
    adapt_causal_prior_to_target,
    apply_target_distribution_guard,
    compute_device_distribution,
)

__all__ = [
    "TargetDistributionGuardConfig",
    "adapt_causal_prior_to_target",
    "apply_target_distribution_guard",
    "compute_device_distribution",
]
