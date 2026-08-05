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
        # This scenario is purpose-built for the volume/VWAP/RSI/MACD
        # confluence - it doesn't happen to contain a market structure
        # break at the entry bar. That gate has its own dedicated tests
        # below (test_market_structure_gate_*), so it's disabled here to
        # isolate what this test is actually checking.
        require_market_structure=False,
        use_key_levels=False,
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
        require_market_structure=False, use_key_levels=False,
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


def _zigzag(anchors: list[float], seg_len: int = 6) -> np.ndarray:
    """A continuous piecewise-linear price path through `anchors` - see
    tests/test_market_structure.py for the same helper."""
    segs = []
    for i in range(len(anchors) - 1):
        seg = np.linspace(anchors[i], anchors[i + 1], seg_len + 1)
        if i > 0:
            seg = seg[1:]
        segs.append(seg)
    return np.concatenate(segs)


def _downtrend_then_mss_bullish_near_support_bars() -> pd.DataFrame:
    """A downtrend (lower highs, lower lows) that then reverses hard enough
    to break above its most recent swing high - a market-structure-shift
    bullish event - while still close enough to the support cluster
    (formed by the downtrend's own lows) to register as "near a level"."""
    close = _zigzag([103, 100, 102, 99.5, 101, 98, 103], seg_len=6)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    volume = np.full(len(close), 1_000_000.0)
    return pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05, "close": close, "volume": volume}, index=idx)


def test_market_structure_gate_blocks_setup_without_a_favorable_event() -> None:
    """With require_market_structure on and no confirmed BOS/MSS in the
    trade's favor, generate() must reject the setup even if every other
    gate (level proximity, indicator confirmations) would pass."""
    bars = _downtrend_then_mss_bullish_near_support_bars()
    # Truncate before the reversal actually breaks structure, so there's a
    # nearby level but no favorable event yet.
    truncated = bars.iloc[:25]
    strategy = SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=True,
        structure_pivot_window=3, min_bars=20,
    )
    assert strategy.generate("TEST", truncated) is None


def test_market_structure_gate_allows_setup_with_a_favorable_event() -> None:
    bars = _downtrend_then_mss_bullish_near_support_bars()
    strategy = SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=True,
        structure_pivot_window=3, min_bars=20,
    )
    setup = strategy.generate("TEST", bars)
    assert setup is not None
    assert setup.direction == TradeDirection.LONG
    assert setup.metadata["market_structure_event"] == "mss_bullish"
    assert setup.metadata["market_structure_trend"] == "downtrend"


def test_market_structure_gate_disabled_ignores_structure() -> None:
    bars = _downtrend_then_mss_bullish_near_support_bars().iloc[:25]  # no favorable event yet
    strategy = SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=False,
        structure_pivot_window=3, min_bars=20,
    )
    assert strategy.generate("TEST", bars) is not None


def test_htf_bias_vetoes_long_when_higher_timeframe_is_in_a_downtrend() -> None:
    bars = _downtrend_then_mss_bullish_near_support_bars()
    htf_close = _zigzag([120, 100, 110, 90, 105, 80, 95], seg_len=6)
    htf_bars = pd.DataFrame(
        {"open": htf_close, "high": htf_close + 0.5, "low": htf_close - 0.5, "close": htf_close},
        index=pd.date_range("2024-01-01", periods=len(htf_close), freq="D"),
    )
    strategy = SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=True,
        structure_pivot_window=3, min_bars=20, require_htf_bias=True, htf_pivot_window=3,
    )
    assert strategy.generate("TEST", bars, htf_bars=htf_bars) is None


def test_htf_bias_allows_long_when_higher_timeframe_is_in_an_uptrend() -> None:
    bars = _downtrend_then_mss_bullish_near_support_bars()
    htf_close = _zigzag([80, 90, 85, 100, 95, 110, 108], seg_len=6)
    htf_bars = pd.DataFrame(
        {"open": htf_close, "high": htf_close + 0.5, "low": htf_close - 0.5, "close": htf_close},
        index=pd.date_range("2024-01-01", periods=len(htf_close), freq="D"),
    )
    strategy = SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=True,
        structure_pivot_window=3, min_bars=20, require_htf_bias=True, htf_pivot_window=3,
    )
    setup = strategy.generate("TEST", bars, htf_bars=htf_bars)
    assert setup is not None
    assert setup.metadata["htf_bias"] == "uptrend"


def test_htf_bias_ignored_when_htf_bars_not_provided() -> None:
    bars = _downtrend_then_mss_bullish_near_support_bars()
    strategy = SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=True,
        structure_pivot_window=3, min_bars=20, require_htf_bias=True, htf_pivot_window=3,
    )
    setup = strategy.generate("TEST", bars)  # no htf_bars at all
    assert setup is not None
    assert setup.metadata["htf_bias"] is None


def test_key_levels_are_merged_into_candidate_pool() -> None:
    """A previous-day low, with nothing from pivot clustering nearby,
    should still be found and traded when use_key_levels is on."""
    idx = pd.DatetimeIndex(
        sum(
            [[pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h) for h in range(9, 16)],
             [pd.Timestamp("2024-01-02") + pd.Timedelta(hours=h) for h in range(9, 16)]],
            [],
        )
    )
    day1_close = np.array([100.0, 101, 99, 102, 98, 103, 97])  # day 1 low = 97
    day2_close = np.array([97.2, 97.5, 97.1, 97.6, 97.05, 97.3, 97.15])  # day 2 hovers just above day-1 low
    close = np.concatenate([day1_close, day2_close])
    volume = np.full(len(close), 1_000_000.0)
    bars = pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": volume}, index=idx
    )

    strategy = SupportResistanceStrategy(
        pivot_window=2, min_touches=5,  # min_touches high enough that pivot clustering finds nothing usable
        level_proximity_pct=0.01, min_required_confirmations=0, use_key_levels=True,
        require_market_structure=False, min_bars=10,
    )
    setup = strategy.generate("TEST", bars)
    assert setup is not None
    # Either a previous-day or pre-market key level is acceptable here (both
    # are anchored near day 1's 97 low) - the point is it's NOT a pivot
    # level, since min_touches=5 makes pivot clustering find nothing.
    assert setup.metadata["level_source"] in {"prev_day", "premarket"}
