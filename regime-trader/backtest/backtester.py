"""Walk-forward allocation backtester.

Re-fits the HMM engine on a rolling/expanding window and steps forward in
fixed increments to avoid look-ahead bias, simulating the full signal ->
risk -> execution pipeline over historical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    regime_history: pd.DataFrame


class WalkForwardBacktester:
    """Runs a walk-forward backtest of the regime-based allocation strategy.

    Args:
        hmm_engine: HMM engine used for regime detection during the backtest.
        strategy: Regime-to-allocation strategy under test.
        risk_manager: Risk manager enforcing limits during the backtest.
        step_size: Walk-forward step size in bars.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: StrategyOrchestrator,
        risk_manager: RiskManager,
        step_size: int,
    ) -> None:
        raise NotImplementedError

    def run(self, data: pd.DataFrame, start: str, end: str) -> BacktestResult:
        """Run the walk-forward backtest over the given date range."""
        raise NotImplementedError

    def step(self, window: pd.DataFrame) -> None:
        """Process a single walk-forward step: refit, infer, allocate, simulate."""
        raise NotImplementedError
