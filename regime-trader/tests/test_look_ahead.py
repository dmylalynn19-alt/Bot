"""Verify no look-ahead bias in feature computation and backtesting."""

from __future__ import annotations


def test_features_use_only_past_data() -> None:
    """FeatureEngineer outputs at time t must not depend on data after t."""
    raise NotImplementedError


def test_backtester_refits_only_on_past_window() -> None:
    """WalkForwardBacktester must only fit the HMM on data available at each step."""
    raise NotImplementedError


def test_signal_generation_does_not_peek_future_bars() -> None:
    """SignalGenerator output at bar t must be reproducible using only bars <= t."""
    raise NotImplementedError
