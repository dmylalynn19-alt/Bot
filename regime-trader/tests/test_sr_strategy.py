"""Tests for core.sr_strategy (support/resistance setups + VWAP/RSI/MACD/volume
confirmation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.sr_strategy import SupportResistanceStrategy, TradeDirection, session_vwap, volume_confirmation


def _bars(open_, high, low, close, volume, freq="B") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_session_vwap_uses_calendar_day_reset_when_intraday() -> None:
    """With multiple bars per calendar day, VWAP must reset each day - a later
    bar's VWAP should equal the cumulative volume-weighted typical price
    since that day's first bar, not since the start of the whole series."""
    idx = pd.date_range("2024-01-01 09:30", periods=8, freq="1h")
    # Two calendar days of 4 bars each (tz-naive, but freq spans a day boundary
    # naturally after 24h - use explicit dates instead to be unambiguous).
    idx = pd.DatetimeIndex(
        [
            "2024-01-01 09:30", "2024-01-01 10:30", "2024-01-01 11:30", "2024-01-01 12:30",
            "2024-01-02 09:30", "2024-01-02 10:30", "2024-01-02 11:30", "2024-01-02 12:30",
        ]
    )
    close = pd.Series([100.0, 101.0, 102.0, 101.5, 200.0, 201.0, 202.0, 201.5], index=idx)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series([1000.0] * 8, index=idx)
    bars = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})

    vwap = session_vwap(bars)
    # Day 2's first-bar VWAP must equal day 2's first-bar typical price alone
    # (~200), not be dragged toward day 1's much lower prices (~100).
    assert vwap.iloc[4] == pytest.approx(200.0, abs=0.01)


def test_session_vwap_falls_back_to_rolling_average_when_daily() -> None:
    """With one bar per calendar day, the calendar-day-reset definition would
    degenerate to each bar's own typical price (no real signal) - VWAP must
    instead be a rolling multi-bar volume-weighted average."""
    close = pd.Series([100.0] * 19 + [110.0])
    close.index = pd.date_range("2024-01-01", periods=20, freq="B")
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series([1000.0] * 20, index=close.index)
    bars = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})

    vwap = session_vwap(bars, rolling_window=5)
    # A single-bar "session" VWAP would just equal the last close (110) -
    # the rolling fallback must instead blend in the trailing bars, landing
    # well below the last bar's own price.
    assert vwap.iloc[-1] < 105.0
    assert vwap.iloc[-1] > 100.0


def test_volume_confirmation_excludes_current_bar_from_its_own_baseline() -> None:
    """A volume spike must not be able to inflate the trailing average it's
    being compared against."""
    volume = pd.Series([1_000_000.0] * 20 + [5_000_000.0])
    confirmed = volume_confirmation(volume, window=20, multiple=1.2)
    assert confirmed.iloc[-1]


def test_generate_returns_none_with_insufficient_history() -> None:
    n = 10
    close = pd.Series(np.linspace(100, 105, n))
    bars = _bars(close, close + 0.2, close - 0.2, close, [1_000_000.0] * n)
    strategy = SupportResistanceStrategy(min_bars=60)
    assert strategy.generate("TEST", bars) is None


def test_generate_fires_confirmed_long_setup_at_support() -> None:
    """A textbook bounce: an established support zone (multiple prior
    touches), a sharp decline into it that leaves RSI oversold, elevated
    volume on the entry bar, MACD momentum turning up, and price back above
    its short-term rolling VWAP - all four confirmations should agree."""
    rng = np.random.default_rng(3)
    t = np.arange(60)
    period = 20
    cycle = 105 + 10 * (1 - 4 * np.abs(((t / period) % 1) - 0.5)) + rng.normal(0, 0.1, 60)
    settle = np.linspace(cycle[-1], 99.5, 8) + rng.normal(0, 0.05, 8)
    plunge = np.array([97.5, 94.9])
    chop = np.array([94.95, 94.80, 94.88, 94.83, 94.90, 95.05])
    close = np.concatenate([cycle, settle, plunge, chop])
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    high = close + 0.15
    low = close - 0.15
    low[len(cycle) + len(settle) + 1] = 94.5  # wick into support on the plunge bar
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 3_000_000.0  # elevated volume on the confirmed entry bar
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

    strategy = SupportResistanceStrategy(
        pivot_window=5,
        cluster_tolerance_pct=0.01,
        min_touches=2,
        level_proximity_pct=0.01,
        volume_window=20,
        volume_multiple=1.2,
        vwap_window=5,
        rsi_period=14,
        rsi_oversold=34.0,
        rsi_overbought=70.0,
        min_required_confirmations=4,
        min_bars=60,
    )
    setup = strategy.generate("TEST", bars)

    assert setup is not None
    assert setup.direction == TradeDirection.LONG
    assert setup.confirmations == {"volume": True, "vwap": True, "rsi": True, "macd": True}
    assert setup.confidence == 1.0
    assert setup.stop_price < setup.entry_price < setup.target_price


def test_should_exit_stop_and_target() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(60)
    period = 20
    cycle = 105 + 10 * (1 - 4 * np.abs(((t / period) % 1) - 0.5)) + rng.normal(0, 0.1, 60)
    settle = np.linspace(cycle[-1], 99.5, 8) + rng.normal(0, 0.05, 8)
    plunge = np.array([97.5, 94.9])
    chop = np.array([94.95, 94.80, 94.88, 94.83, 94.90, 95.05])
    close = np.concatenate([cycle, settle, plunge, chop])
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    high = close + 0.15
    low = close - 0.15
    low[len(cycle) + len(settle) + 1] = 94.5
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 3_000_000.0
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

    strategy = SupportResistanceStrategy(
        pivot_window=5, cluster_tolerance_pct=0.01, min_touches=2, level_proximity_pct=0.01,
        volume_window=20, volume_multiple=1.2, vwap_window=5, rsi_period=14, rsi_oversold=34.0,
        rsi_overbought=70.0, min_required_confirmations=4, min_bars=60,
    )
    setup = strategy.generate("TEST", bars)
    assert setup is not None

    still_open = bars.copy()
    exited, _ = strategy.should_exit(setup, still_open)
    assert not exited

    hit_stop = bars.copy()
    hit_stop.loc[hit_stop.index[-1], "close"] = setup.stop_price - 0.5
    exited, reason = strategy.should_exit(setup, hit_stop)
    assert exited and "stop" in reason

    hit_target = bars.copy()
    hit_target.loc[hit_target.index[-1], "close"] = setup.target_price + 0.5
    exited, reason = strategy.should_exit(setup, hit_target)
    assert exited and "target" in reason
