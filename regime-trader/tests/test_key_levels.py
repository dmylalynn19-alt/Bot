"""Tests for core.key_levels (previous-day levels, pre-market levels, order
blocks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.key_levels import latest_key_levels, order_blocks, premarket_levels, previous_day_levels


def test_previous_day_levels_uses_prior_days_complete_range() -> None:
    idx = pd.DatetimeIndex(
        sum(
            [
                [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h) for h in [9, 10, 11]],
                [pd.Timestamp("2024-01-02") + pd.Timedelta(hours=h) for h in [9, 10, 11]],
                [pd.Timestamp("2024-01-03") + pd.Timedelta(hours=h) for h in [9, 10, 11]],
            ],
            [],
        )
    )
    high = [101, 105, 103, 110, 112, 108, 95, 97, 96]
    low = [99, 100, 101, 106, 107, 105, 90, 92, 93]
    bars = pd.DataFrame({"high": high, "low": low}, index=idx)

    levels = previous_day_levels(bars)

    assert levels.iloc[0:3].isna().all().all()  # first day has no prior day
    assert (levels.iloc[3:6]["prev_day_high"] == 105.0).all()
    assert (levels.iloc[3:6]["prev_day_low"] == 99.0).all()
    assert (levels.iloc[6:9]["prev_day_high"] == 112.0).all()
    assert (levels.iloc[6:9]["prev_day_low"] == 105.0).all()


def test_premarket_levels_freezes_at_open_and_ignores_regular_session() -> None:
    idx = pd.DatetimeIndex(
        [
            "2024-01-01 07:00", "2024-01-01 07:30", "2024-01-01 08:00",
            "2024-01-01 08:30", "2024-01-01 09:00",
            "2024-01-01 09:30", "2024-01-01 10:00", "2024-01-01 10:30",
        ]
    )
    high = [100, 102, 99, 104, 101, 110, 108, 112]
    low = [98, 99, 97, 100, 99, 105, 104, 106]
    bars = pd.DataFrame({"high": high, "low": low}, index=idx)

    pm = premarket_levels(bars)

    # Running cumulative max/min *during* pre-market only grows monotonically.
    assert list(pm["premarket_high"].iloc[:5]) == [100, 102, 102, 104, 104]
    assert list(pm["premarket_low"].iloc[:5]) == [98, 98, 97, 97, 97]
    # Frozen for the whole regular session - never picks up the 112 regular-session high.
    assert (pm["premarket_high"].iloc[5:] == 104).all()
    assert (pm["premarket_low"].iloc[5:] == 97).all()


def test_premarket_levels_all_nan_without_premarket_bars() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")  # daily bars, no time-of-day info
    bars = pd.DataFrame({"high": [101, 102, 103, 104, 105], "low": [99, 100, 101, 102, 103]}, index=idx)
    pm = premarket_levels(bars)
    assert pm.isna().all().all()


def test_order_blocks_identifies_last_opposite_candle_before_displacement() -> None:
    rng = np.random.default_rng(1)
    n = 40
    close = 100 + np.cumsum(rng.normal(0, 0.15, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1

    # A red candle at 20, then a strong green displacement bar at 21.
    open_[20], close[20] = 105.0, 104.3
    high[20], low[20] = 105.2, 104.1
    open_[21], close[21] = 104.4, 110.0
    high[21], low[21] = 110.2, 104.3

    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    levels = order_blocks(bars, lookback=3, displacement_atr_mult=1.5, atr_period=14)

    assert len(levels) >= 1
    matches = [lvl for lvl in levels if lvl.first_touch == idx[20]]
    assert len(matches) == 1
    assert matches[0].kind == "support"
    assert matches[0].source == "order_block"
    assert low[20] <= matches[0].price <= high[20]


def test_order_blocks_requires_a_real_displacement() -> None:
    """A calm, low-volatility series with no big range-expansion bar should
    produce no order blocks at all."""
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    close = pd.Series(100.0, index=idx) + np.linspace(0, 0.5, 40)
    bars = pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05, "close": close}, index=idx)
    assert order_blocks(bars) == []


def test_latest_key_levels_skips_unavailable_sources_without_error() -> None:
    """On plain daily bars (no pre-market data, single day of history),
    latest_key_levels should just quietly return whatever is available -
    order blocks only, most likely - not raise or fabricate NaN levels."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 0.2, 30))
    bars = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=idx)

    levels = latest_key_levels(bars)
    for level in levels:
        assert level.source in {"prev_day", "premarket", "order_block"}
        assert not pd.isna(level.price)
