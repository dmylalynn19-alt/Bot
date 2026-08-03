"""Vol-based allocation strategies.

Maps a detected market regime (and its confidence) to a target portfolio
allocation and leverage level per the strategy configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.hmm_engine import RegimeState


@dataclass
class AllocationTarget:
    """Target allocation and leverage produced by a regime strategy."""

    allocation: float
    leverage: float
    symbol: str


class RegimeStrategy:
    """Translates regime states into target allocations.

    Args:
        low_vol_allocation: Target allocation in a low-volatility regime.
        mid_vol_allocation_trend: Target allocation in mid-vol regime, trend confirmed.
        mid_vol_allocation_no_trend: Target allocation in mid-vol regime, no confirmed trend.
        high_vol_allocation: Target allocation in a high-volatility regime.
        low_vol_leverage: Leverage multiplier applied only in low-volatility regime.
        rebalance_threshold: Fractional drift from target that triggers a rebalance.
        uncertainty_size_mult: Position size multiplier applied when regime confidence is low.
    """

    def __init__(
        self,
        low_vol_allocation: float,
        mid_vol_allocation_trend: float,
        mid_vol_allocation_no_trend: float,
        high_vol_allocation: float,
        low_vol_leverage: float,
        rebalance_threshold: float,
        uncertainty_size_mult: float,
    ) -> None:
        raise NotImplementedError

    def target_allocation(self, symbol: str, state: RegimeState) -> AllocationTarget:
        """Compute the target allocation and leverage for a symbol given its regime state."""
        raise NotImplementedError

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """Check whether drift from target allocation exceeds the rebalance threshold."""
        raise NotImplementedError

    def apply_confidence_discount(self, target: AllocationTarget, confidence: float) -> AllocationTarget:
        """Scale down a target allocation when regime confidence is below min_confidence."""
        raise NotImplementedError
