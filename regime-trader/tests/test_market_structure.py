"""Tests for core.market_structure (swing detection, trend classification,
BOS/MSS event detection)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.market_structure import (
    MarketStructureAnalyzer,
    StructureEvent,
    Trend,
    current_leg_acceleration,
    detect_swings,
    identify_legs,
)


def _zigzag(anchors: list[float], seg_len: int = 8) -> np.ndarray:
    """A continuous piecewise-linear price path through `anchors`, `seg_len`
    bars per leg (endpoints shared between consecutive legs, not doubled)."""
    segs = []
    for i in range(len(anchors) - 1):
        seg = np.linspace(anchors[i], anchors[i + 1], seg_len + 1)
        if i > 0:
            seg = seg[1:]
        segs.append(seg)
    return np.concatenate(segs)


def _bars(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05, "close": close}, index=idx)


def test_uptrend_with_breakout_is_bos_bullish() -> None:
    """Higher-high + higher-low structure, then price breaks above the most
    recent swing high => continuation (BOS bullish)."""
    close = _zigzag([100, 108, 103, 112, 108, 118])
    state = MarketStructureAnalyzer(pivot_window=3).analyze(_bars(close))
    assert state.trend == Trend.UPTREND
    assert state.last_event == StructureEvent.BOS_BULLISH


def test_uptrend_broken_down_is_mss_bearish() -> None:
    """An established uptrend (HH+HL), then price breaks below the most
    recent swing low => reversal (MSS bearish), even though the trend
    label itself (computed from confirmed swings only) hasn't flipped yet."""
    close = _zigzag([100, 108, 103, 112, 107, 116, 95])
    state = MarketStructureAnalyzer(pivot_window=3).analyze(_bars(close))
    assert state.trend == Trend.UPTREND
    assert state.last_event == StructureEvent.MSS_BEARISH
    assert state.last_event_price == state.last_swing_low.price


def test_downtrend_with_breakdown_is_bos_bearish() -> None:
    close = _zigzag([110, 102, 107, 98, 103, 90])
    state = MarketStructureAnalyzer(pivot_window=3).analyze(_bars(close))
    assert state.trend == Trend.DOWNTREND
    assert state.last_event == StructureEvent.BOS_BEARISH


def test_downtrend_broken_up_is_mss_bullish() -> None:
    close = _zigzag([100, 92, 97, 88, 93, 84, 105])
    state = MarketStructureAnalyzer(pivot_window=3).analyze(_bars(close))
    assert state.trend == Trend.DOWNTREND
    assert state.last_event == StructureEvent.MSS_BULLISH
    assert state.last_event_price == state.last_swing_high.price


def test_insufficient_swings_is_ranging_with_no_event() -> None:
    close = _zigzag([100, 108, 103])  # only one high, one low
    state = MarketStructureAnalyzer(pivot_window=3).analyze(_bars(close))
    assert state.trend == Trend.RANGING
    assert state.last_event == StructureEvent.NONE
    assert state.last_event_price is None


def test_detect_swings_no_look_ahead() -> None:
    """A swing's price/kind, as of any cutoff, must not depend on whether
    bars after that cutoff exist - identical guarantee to, and reusing the
    same pivot functions as, tests/test_look_ahead.py's S/R coverage."""
    rng = np.random.default_rng(4)
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    base = 100 + 8 * np.sin(np.linspace(0, 12, 200)) + rng.normal(0, 0.2, 200)
    close = pd.Series(base, index=idx)
    bars = pd.DataFrame({"open": close, "high": close + rng.uniform(0.1, 0.3, 200), "low": close - rng.uniform(0.1, 0.3, 200), "close": close}, index=idx)

    window = 4
    short_swings = detect_swings(bars.iloc[:100], window)
    long_swings = detect_swings(bars.iloc[:200], window)
    cutoff = bars.index[100 - window]

    short_confirmed = [(s.kind, round(s.price, 6)) for s in short_swings if s.timestamp <= cutoff]
    long_confirmed_up_to_cutoff = [(s.kind, round(s.price, 6)) for s in long_swings if s.timestamp <= cutoff]
    assert sorted(short_confirmed) == sorted(long_confirmed_up_to_cutoff)


def test_identify_legs_computes_speed_between_consecutive_swings() -> None:
    close = _zigzag([100, 108, 103, 112, 107, 116], seg_len=10)
    legs = identify_legs(_bars(close), pivot_window=3)
    assert len(legs) == 3
    for leg in legs:
        assert leg.bar_count == 10
        assert leg.speed == pytest.approx(abs(leg.price_change) / leg.bar_count)


def test_current_leg_acceleration_flags_a_fast_final_push() -> None:
    """A steady uptrend of similar-speed legs, then a much faster, shorter
    final leg - acceleration ratio should be well above 1."""
    close = _zigzag([100, 108, 103, 112, 107, 116], seg_len=10)
    final = np.linspace(116, 130, 4)[1:]  # big move, few bars -> fast
    close = np.concatenate([close, final])
    ratio = current_leg_acceleration(_bars(close), pivot_window=3, lookback_legs=3)
    assert ratio is not None
    assert ratio > 1.5


def test_current_leg_acceleration_none_without_enough_leg_history() -> None:
    close = _zigzag([100, 108, 103], seg_len=10)  # only one leg
    ratio = current_leg_acceleration(_bars(close), pivot_window=3, lookback_legs=3)
    assert ratio is None


def test_identify_legs_no_look_ahead() -> None:
    """A confirmed leg's speed, as of any cutoff, must not depend on bars
    after that cutoff - same guarantee as the underlying swings."""
    rng = np.random.default_rng(6)
    idx = pd.date_range("2024-01-01", periods=250, freq="B")
    base = 100 + 10 * np.sin(np.linspace(0, 14, 250)) + rng.normal(0, 0.2, 250)
    close = pd.Series(base, index=idx)
    bars = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close}, index=idx)

    window = 4
    short_legs = identify_legs(bars.iloc[:120], window)
    long_legs = identify_legs(bars.iloc[:250], window)
    cutoff = bars.index[120 - window]

    short_confirmed = [(round(l.start.price, 6), round(l.end.price, 6)) for l in short_legs if l.end.timestamp <= cutoff]
    long_confirmed = [(round(l.start.price, 6), round(l.end.price, 6)) for l in long_legs if l.end.timestamp <= cutoff]
    assert sorted(short_confirmed) == sorted(long_confirmed)
