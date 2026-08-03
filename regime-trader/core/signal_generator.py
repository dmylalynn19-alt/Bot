"""Combines HMM regime detection and allocation strategy into trade signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import RegimeStrategy
from core.risk_manager import RiskManager


class SignalAction(Enum):
    """Directional action a generated signal instructs."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"


@dataclass
class Signal:
    """A single actionable trading signal for a symbol."""

    symbol: str
    action: SignalAction
    target_allocation: float
    confidence: float


class SignalGenerator:
    """Generates trading signals by combining regime detection with strategy and risk rules.

    Args:
        hmm_engine: Fitted HMM regime detection engine.
        strategy: Regime-to-allocation strategy.
        risk_manager: Risk manager enforcing position and portfolio limits.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
    ) -> None:
        raise NotImplementedError

    def generate(self, symbol: str, features: pd.DataFrame, current_allocation: float, equity: float) -> Signal:
        """Generate a single signal for a symbol given its latest features and current position."""
        raise NotImplementedError

    def generate_batch(
        self, features_by_symbol: dict[str, pd.DataFrame], current_allocations: dict[str, float], equity: float
    ) -> list[Signal]:
        """Generate signals for a batch of symbols."""
        raise NotImplementedError
