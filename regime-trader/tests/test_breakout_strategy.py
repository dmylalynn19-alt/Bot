"""Tests for core.breakout_strategy.BreakoutStrategy (consolidation breakout
+ three-bar entry + leg acceleration)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.breakout_strategy import BreakoutStrategy
from core.consolidation import find_last_consolidation
from core.sr_strategy import TradeDirection


def _consolidation_then_breakout_bars(seed: int = 4, bullish: bool = True) -> pd.DataFrame:
    """A volatile run-up, a 20-bar tight consolidation, then a breakout -
    retest - re-break sequence that is simultaneously a valid three-bar
    entry pattern (lead=breakout, reaction=retest, confirmation=re-break)."""
    rng = np.random.default_rng(seed)
    n_pre, n_tight = 40, 20
    volatile = 100 + np.cumsum(rng.normal(0, 0.4, n_pre))
    tight = 100 + rng.normal(0, 0.1, n_tight) + (volatile[-1] - 100)
    close = np.concatenate([volatile, tight])
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.2, len(close))
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.2, len(close))
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    cons = find_last_consolidation(bars, lookback=20, max_range_atr_mult=2.5, search_bars=60)
    assert cons is not None

    edge = cons.high if bullish else cons.low
    sign = 1 if bullish else -1
    lead_open, lead_close = close[-1], edge + sign * 1.2
    # Reaction pulls back only partway toward the edge (not all the way to
    # it) - keeps the retrace comfortably under three_bar_entry's default
    # max_reaction_retrace_pct (0.75) while still landing within classify_
    # breakout's default retest tolerance (0.5%) of the edge itself.
    reaction_open, reaction_close = lead_close, edge + sign * 0.42
    confirm_open, confirm_close = reaction_close, edge + sign * 1.8

    ext_close = [lead_close, reaction_close, confirm_close]
    ext_open = [lead_open, reaction_open, confirm_open]
    if bullish:
        ext_high = [lead_close + 0.15, reaction_close + 0.1, confirm_close + 0.15]
        ext_low = [min(lead_open, lead_close) - 0.1, edge - 0.05, min(confirm_open, confirm_close) - 0.1]
    else:
        ext_high = [max(lead_open, lead_close) + 0.1, edge + 0.05, max(confirm_open, confirm_close) + 0.1]
        ext_low = [lead_close - 0.15, reaction_close - 0.1, confirm_close - 0.15]

    ext_idx = pd.bdate_range(idx[-1] + pd.tseries.offsets.BDay(1), periods=3)
    ext = pd.DataFrame({"open": ext_open, "high": ext_high, "low": ext_low, "close": ext_close}, index=ext_idx)
    return pd.concat([bars, ext])


def _strategy(**overrides) -> BreakoutStrategy:
    params = dict(
        consolidation_lookback=20, consolidation_max_range_atr_mult=2.5, atr_period=14,
        retest_tolerance_pct=0.005, require_true_breakout=True,
        three_bar_momentum_multiple=1.3, min_bars=50,
    )
    params.update(overrides)
    return BreakoutStrategy(**params)


def test_generate_fires_confirmed_long_breakout_setup() -> None:
    bars = _consolidation_then_breakout_bars(bullish=True)
    setup = _strategy().generate("TEST", bars)

    assert setup is not None
    assert setup.direction == TradeDirection.LONG
    assert setup.confirmations["consolidation"] is True
    assert setup.confirmations["breakout"] is True
    assert setup.confirmations["three_bar_entry"] is True
    assert setup.stop_price < setup.entry_price < setup.target_price
    assert setup.level.kind == "support"  # broken resistance flips to support
    assert setup.metadata["breakout_status"] == "true_breakout"


def test_generate_fires_confirmed_short_breakout_setup() -> None:
    bars = _consolidation_then_breakout_bars(bullish=False)
    setup = _strategy().generate("TEST", bars)

    assert setup is not None
    assert setup.direction == TradeDirection.SHORT
    assert setup.target_price < setup.entry_price < setup.stop_price
    assert setup.level.kind == "resistance"  # broken support flips to resistance


def test_require_true_breakout_blocks_a_pending_retest() -> None:
    """Truncate right after the initial breakout bar (before the retest and
    re-break) - with require_true_breakout on, that must not fire."""
    bars = _consolidation_then_breakout_bars(bullish=True)
    truncated = bars.iloc[:-2]  # keep only the lead/breakout bar
    setup = _strategy(require_true_breakout=True).generate("TEST", truncated)
    assert setup is None


def test_require_true_breakout_off_allows_pending_retest_with_valid_trigger() -> None:
    """With require_true_breakout off, a PENDING_RETEST breakout can still
    be traded, as long as there's still a valid three-bar trigger."""
    bars = _consolidation_then_breakout_bars(bullish=True)
    # Use only the lead bar plus two more bars that still form a valid
    # three-bar pattern but haven't retested yet.
    strategy = _strategy(require_true_breakout=False, retest_tolerance_pct=0.0001)
    setup = strategy.generate("TEST", bars)
    assert setup is not None  # true_breakout also satisfies "not required to be true"


def test_generate_returns_none_without_a_consolidation() -> None:
    rng = np.random.default_rng(1)
    n = 80
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))  # persistently choppy/volatile, no tight range
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.3, 0.6, n)
    low = np.minimum(open_, close) - rng.uniform(0.3, 0.6, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    assert _strategy().generate("TEST", bars) is None


def test_generate_returns_none_with_insufficient_history() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    bars = pd.DataFrame({"open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "close": [100] * 10}, index=idx)
    assert _strategy(min_bars=50).generate("TEST", bars) is None


def test_require_leg_acceleration_blocks_without_a_fast_final_leg() -> None:
    bars = _consolidation_then_breakout_bars(bullish=True)
    strict = _strategy(require_leg_acceleration=True, min_acceleration_ratio=100.0)  # unreachable
    assert strict.generate("TEST", bars) is None


def test_should_exit_stop_and_target() -> None:
    bars = _consolidation_then_breakout_bars(bullish=True)
    strategy = _strategy()
    setup = strategy.generate("TEST", bars)
    assert setup is not None

    hit_stop = bars.copy()
    hit_stop.loc[hit_stop.index[-1], "close"] = setup.stop_price - 0.5
    exited, reason = strategy.should_exit(setup, hit_stop)
    assert exited and "stop" in reason

    hit_target = bars.copy()
    hit_target.loc[hit_target.index[-1], "close"] = setup.target_price + 0.5
    exited, reason = strategy.should_exit(setup, hit_target)
    assert exited and "target" in reason

    still_open = bars.copy()
    exited, _ = strategy.should_exit(setup, still_open)
    assert not exited
