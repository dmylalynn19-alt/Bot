"""Technical indicators and feature computation.

Computes the feature set consumed by the HMM engine (returns, realized
volatility, trend indicators, etc.) from raw OHLCV bars.
"""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Computes technical indicators and model-ready features from OHLCV data."""

    def __init__(self) -> None:
        raise NotImplementedError

    def compute_returns(self, bars: pd.DataFrame) -> pd.Series:
        """Compute log or simple returns from a bar series."""
        raise NotImplementedError

    def compute_volatility(self, bars: pd.DataFrame, window: int) -> pd.Series:
        """Compute rolling realized volatility."""
        raise NotImplementedError

    def compute_trend_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute trend indicators (e.g. moving averages, ADX)."""
        raise NotImplementedError

    def build_feature_matrix(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Assemble the full feature matrix used as HMM input."""
        raise NotImplementedError
