"""Tests for core.candle_patterns (candle speed/momentum, three-bar entry model)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.candle_patterns import ThreeBarDirection, candle_speed, three_bar_entry


def _calm_baseline(n: int = 27, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15
    return open_, high, low, close


def _bars_with_three_bar_pattern(bullish: bool) -> pd.DataFrame:
    open_, high, low, close = _calm_baseline()
    if bullish:
        open_[-3], close[-3] = 100.0, 103.0
        high[-3], low[-3] = 103.1, 99.9
        open_[-2], close[-2] = 103.0, 102.3
        high[-2], low[-2] = 103.2, 102.2
        open_[-1], close[-1] = 102.3, 103.5
        high[-1], low[-1] = 103.6, 102.2
    else:
        open_[-3], close[-3] = 103.0, 100.0
        high[-3], low[-3] = 103.1, 99.9
        open_[-2], close[-2] = 100.0, 100.7
        high[-2], low[-2] = 100.8, 99.8
        open_[-1], close[-1] = 100.7, 99.5
        high[-1], low[-1] = 100.8, 99.4

    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_candle_speed_flags_a_wide_range_bar() -> None:
    open_, high, low, close = _calm_baseline()
    high[-1] = close[-1] + 3.0  # one abnormally wide bar
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    speed = candle_speed(bars, atr_period=14)
    assert speed.iloc[-1] > 3.0  # far above a normal ~1x-ATR bar
    assert speed.iloc[-2] < 3.0


def test_three_bar_entry_detects_bullish_pattern() -> None:
    bars = _bars_with_three_bar_pattern(bullish=True)
    signal = three_bar_entry(bars, atr_period=14, momentum_multiple=1.3)
    assert signal is not None
    assert signal.direction == ThreeBarDirection.BULLISH
    assert signal.lead_speed >= 1.3


def test_three_bar_entry_detects_bearish_pattern() -> None:
    bars = _bars_with_three_bar_pattern(bullish=False)
    signal = three_bar_entry(bars, atr_period=14, momentum_multiple=1.3)
    assert signal is not None
    assert signal.direction == ThreeBarDirection.BEARISH


def test_three_bar_entry_rejects_a_slow_lead_candle() -> None:
    """A lead candle that isn't actually fast (range not well above ATR)
    must not qualify, even if the reaction/confirmation shape is right."""
    bars = _bars_with_three_bar_pattern(bullish=True)
    assert three_bar_entry(bars, atr_period=14, momentum_multiple=50.0) is None


def test_three_bar_entry_rejects_a_deep_reaction_retrace() -> None:
    """A reaction candle that gives back most of the lead candle's move
    looks like a real reversal attempt, not a shallow pullback - must
    disqualify the pattern."""
    open_, high, low, close = _calm_baseline()
    open_[-3], close[-3] = 100.0, 103.0
    high[-3], low[-3] = 103.1, 99.9
    open_[-2], close[-2] = 103.0, 100.2  # gives back almost the whole lead move
    high[-2], low[-2] = 103.1, 100.1
    open_[-1], close[-1] = 100.2, 103.6
    high[-1], low[-1] = 103.7, 100.0
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    assert three_bar_entry(bars, atr_period=14, momentum_multiple=1.3, max_reaction_retrace_pct=0.75) is None


def test_three_bar_entry_rejects_unconfirmed_reaction() -> None:
    """If the third bar never actually breaks past the reaction candle's
    extreme, there's no confirmation yet - must return None."""
    open_, high, low, close = _calm_baseline()
    open_[-3], close[-3] = 100.0, 103.0
    high[-3], low[-3] = 103.1, 99.9
    open_[-2], close[-2] = 103.0, 102.3
    high[-2], low[-2] = 103.2, 102.2
    open_[-1], close[-1] = 102.3, 102.5  # doesn't close above the reaction high (103.2)
    high[-1], low[-1] = 102.9, 102.1
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    assert three_bar_entry(bars, atr_period=14, momentum_multiple=1.3) is None


def test_three_bar_entry_returns_none_with_insufficient_history() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    bars = pd.DataFrame({"open": [100] * 5, "high": [101] * 5, "low": [99] * 5, "close": [100] * 5}, index=idx)
    assert three_bar_entry(bars) is None
