"""Tests for core.consolidation (range detection, true/false breakout
classification)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.consolidation import BreakoutStatus, classify_breakout, detect_consolidation, find_last_consolidation


def _consolidating_bars(n: int = 60, tight_n: int = 20, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    volatile = 100 + np.cumsum(rng.normal(0, 0.5, n - tight_n))
    tight = 100 + rng.normal(0, 0.15, tight_n) + (volatile[-1] - 100)
    close = np.concatenate([volatile, tight])
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.3, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.3, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _extend(bars: pd.DataFrame, closes: list[float]) -> pd.DataFrame:
    ext_idx = pd.bdate_range(bars.index[-1] + pd.tseries.offsets.BDay(1), periods=len(closes))
    prev_close = bars["close"].iloc[-1]
    opens = [prev_close] + closes[:-1]
    ext = pd.DataFrame(
        {
            "open": opens,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
        },
        index=ext_idx,
    )
    return pd.concat([bars, ext])


def test_detect_consolidation_finds_a_tight_range() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    assert cons.bar_count == 20
    assert cons.low < cons.high


def test_detect_consolidation_returns_none_for_a_wide_recent_range() -> None:
    bars = _consolidating_bars(tight_n=0)  # no tight tail at all
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=1.2)
    assert cons is None


def test_detect_consolidation_returns_none_with_insufficient_history() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    bars = pd.DataFrame({"open": [100] * 5, "high": [101] * 5, "low": [99] * 5, "close": [100] * 5}, index=idx)
    assert detect_consolidation(bars, lookback=20) is None


def test_classify_breakout_true_breakout_after_retest_holds() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    extended = _extend(bars, [cons.high + 1.5, cons.high + 0.05, cons.high + 2.0, cons.high + 3.0])
    state = classify_breakout(extended, cons, retest_tolerance_pct=0.01)
    assert state.status == BreakoutStatus.TRUE_BREAKOUT
    assert state.direction == "up"


def test_classify_breakout_false_breakout_reverses_back_through() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    extended = _extend(bars, [cons.high + 1.0, cons.high + 0.3, cons.low - 0.5])
    state = classify_breakout(extended, cons, retest_tolerance_pct=0.01)
    assert state.status == BreakoutStatus.FALSE_BREAKOUT


def test_classify_breakout_pending_retest_before_any_retest() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    extended = _extend(bars, [cons.high + 1.0, cons.high + 1.2])  # broke out, no retest yet
    state = classify_breakout(extended, cons, retest_tolerance_pct=0.005)
    assert state.status == BreakoutStatus.PENDING_RETEST


def test_classify_breakout_none_while_still_inside_range() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    extended = _extend(bars, [cons.mid, cons.mid + 0.05])
    state = classify_breakout(extended, cons)
    assert state.status == BreakoutStatus.NONE


def test_find_last_consolidation_locates_a_range_that_has_since_broken_out() -> None:
    """detect_consolidation only checks the very last `lookback` bars, so it
    goes blind the moment a breakout has moved price away from the range -
    find_last_consolidation must still find the range a few bars later."""
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None

    extended = _extend(bars, [cons.high + 1.5, cons.high + 2.0, cons.high + 2.5])
    assert detect_consolidation(extended, lookback=20, max_range_atr_mult=2.5) is None  # scrolled out of view

    found = find_last_consolidation(extended, lookback=20, max_range_atr_mult=2.5, search_bars=60)
    assert found is not None
    assert found.high == pytest.approx(cons.high)
    assert found.low == pytest.approx(cons.low)


def test_find_last_consolidation_none_outside_search_window() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    extended = _extend(bars, [cons.high + 1.5] * 10)
    # The consolidation is now more than `search_bars` back - shouldn't be found.
    assert find_last_consolidation(extended, lookback=20, max_range_atr_mult=2.5, search_bars=5) is None


def test_classify_breakout_downside_true_breakout() -> None:
    bars = _consolidating_bars()
    cons = detect_consolidation(bars, lookback=20, max_range_atr_mult=2.5)
    assert cons is not None
    extended = _extend(bars, [cons.low - 1.5, cons.low - 0.05, cons.low - 2.0, cons.low - 3.0])
    state = classify_breakout(extended, cons, retest_tolerance_pct=0.01)
    assert state.status == BreakoutStatus.TRUE_BREAKOUT
    assert state.direction == "down"
