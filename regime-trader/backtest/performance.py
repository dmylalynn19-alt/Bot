"""Sharpe, drawdown, regime breakdown, and benchmark comparisons."""

from __future__ import annotations

import pandas as pd


class PerformanceAnalyzer:
    """Computes performance metrics from a backtest equity curve.

    Args:
        risk_free_rate: Annualized risk-free rate used for Sharpe/Sortino calculations.
    """

    def __init__(self, risk_free_rate: float) -> None:
        raise NotImplementedError

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Compute the annualized Sharpe ratio of a returns series."""
        raise NotImplementedError

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Compute the maximum drawdown from an equity curve."""
        raise NotImplementedError

    def regime_breakdown(self, returns: pd.Series, regime_history: pd.DataFrame) -> pd.DataFrame:
        """Break down performance metrics by market regime."""
        raise NotImplementedError

    def compare_to_benchmark(self, equity_curve: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
        """Compare strategy performance to a buy-and-hold benchmark."""
        raise NotImplementedError

    def summary(self, equity_curve: pd.Series) -> dict:
        """Produce a full summary of performance statistics."""
        raise NotImplementedError
