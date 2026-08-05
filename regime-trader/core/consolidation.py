"""Consolidation ranges and breakout classification.

The idea: a market that has been trading in a tight range (low volatility
relative to its own recent history) is "coiled" - the strategy wants to
catch the trend that emerges once it finally breaks out, not trade inside
the chop. And not every break of the range is real: price can push through
the edge and immediately fail back in (a false breakout / stop run) just as
often as it can actually run (a true breakout). The standard way to tell
them apart without guessing is to wait for the retest: does price come back
to the broken edge and hold (true) or does it push back through into the
range (false)?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import ta


class BreakoutStatus(str, Enum):
    NONE = "none"  # still inside the range, or no breakout yet
    PENDING_RETEST = "pending_retest"  # broke out, hasn't come back to retest the edge yet
    TRUE_BREAKOUT = "true_breakout"  # retested the broken edge and held
    FALSE_BREAKOUT = "false_breakout"  # broke out then pushed back through the edge


@dataclass
class ConsolidationRange:
    high: float
    low: float
    start: pd.Timestamp
    end: pd.Timestamp
    bar_count: int

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def width_pct(self) -> float:
        return (self.high - self.low) / self.mid if self.mid else 0.0


@dataclass
class BreakoutState:
    status: BreakoutStatus
    direction: str | None  # "up" or "down", None if status is NONE
    breakout_price: float | None
    breakout_time: pd.Timestamp | None
    range: ConsolidationRange


def detect_consolidation(
    bars: pd.DataFrame,
    lookback: int = 20,
    max_range_atr_mult: float = 2.5,
    atr_period: int = 14,
) -> ConsolidationRange | None:
    """Check whether the trailing `lookback` bars of `bars` form a
    consolidation: their total high-low range is tight relative to the
    market's own recent typical bar range (ATR) - i.e. volatility has
    contracted, not just "price didn't move much by coincidence over 2
    bars". Returns None if there isn't a tight range.

    `max_range_atr_mult` bounds the *whole `lookback`-bar range* (not a
    single bar) as a multiple of ATR - e.g. 2.5 means the entire window's
    high-low spread is no wider than 2.5 average bars, a real squeeze for
    anything more than a handful of bars.
    """
    if len(bars) < max(lookback, atr_period) + 1:
        return None

    window = bars.iloc[-lookback:]
    atr = ta.volatility.AverageTrueRange(bars["high"], bars["low"], bars["close"], window=atr_period).average_true_range()
    current_atr = float(atr.iloc[-1])
    if pd.isna(current_atr) or current_atr <= 0:
        return None

    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    if (range_high - range_low) > max_range_atr_mult * current_atr:
        return None

    return ConsolidationRange(high=range_high, low=range_low, start=window.index[0], end=window.index[-1], bar_count=len(window))


def find_last_consolidation(
    bars: pd.DataFrame,
    lookback: int = 20,
    max_range_atr_mult: float = 2.5,
    atr_period: int = 14,
    search_bars: int = 60,
) -> ConsolidationRange | None:
    """Scan backward from the end of `bars` for the most recent `lookback`-
    bar window that qualifies as a consolidation (see `detect_consolidation`),
    searching within the trailing `search_bars` bars.

    `detect_consolidation` only ever checks the *very last* `lookback` bars -
    fine for "is the market consolidating right now", but not for a breakout
    strategy: by the time a breakout has happened and (a few bars later)
    been confirmed, the tight range itself has already scrolled out of the
    immediate trailing window. This looks further back to find it, so a
    breakout can still be traced to the range it broke from.
    """
    if len(bars) < lookback + atr_period:
        return None

    atr = ta.volatility.AverageTrueRange(bars["high"], bars["low"], bars["close"], window=atr_period).average_true_range()
    earliest_end = max(lookback, len(bars) - search_bars)

    for end in range(len(bars), earliest_end - 1, -1):
        window = bars.iloc[end - lookback : end]
        if len(window) < lookback:
            break
        current_atr = float(atr.iloc[end - 1])
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        range_high = float(window["high"].max())
        range_low = float(window["low"].min())
        if (range_high - range_low) <= max_range_atr_mult * current_atr:
            return ConsolidationRange(high=range_high, low=range_low, start=window.index[0], end=window.index[-1], bar_count=len(window))
    return None


def classify_breakout(
    bars: pd.DataFrame,
    consolidation: ConsolidationRange,
    retest_tolerance_pct: float = 0.003,
    max_bars_to_classify: int = 20,
) -> BreakoutState:
    """Walk `bars` (everything from the end of `consolidation` onward, most
    recent bar last) forward and classify the breakout, if any, from
    `consolidation`.

    - NONE: price hasn't closed outside the range yet.
    - PENDING_RETEST: price closed outside the range but hasn't come back
      within `retest_tolerance_pct` of the broken edge yet.
    - TRUE_BREAKOUT: after breaking out, price retested the broken edge
      (traded back within `retest_tolerance_pct` of it) and then closed
      back beyond it again in the breakout direction - the edge held as
      support (if broken up) or resistance (if broken down).
    - FALSE_BREAKOUT: after breaking out, price closed back inside the
      original range - the breakout failed.

    Only looks at bars strictly after `consolidation.end` - a breakout is a
    property of what happens *after* the range was established, and only
    within `max_bars_to_classify` of it (an old, long-since-resolved
    breakout shouldn't still gate new decisions indefinitely).
    """
    after = bars.loc[bars.index > consolidation.end]
    if after.empty:
        return BreakoutState(BreakoutStatus.NONE, None, None, None, consolidation)
    after = after.iloc[:max_bars_to_classify]

    direction: str | None = None
    breakout_price: float | None = None
    breakout_time: pd.Timestamp | None = None
    retested = False

    for ts, bar in after.iterrows():
        close = float(bar["close"])
        if direction is None:
            if close > consolidation.high:
                direction, breakout_price, breakout_time = "up", close, ts
            elif close < consolidation.low:
                direction, breakout_price, breakout_time = "down", close, ts
            continue

        edge = consolidation.high if direction == "up" else consolidation.low
        distance_pct = abs(close - edge) / edge if edge else 0.0

        if direction == "up":
            if close < consolidation.high:
                return BreakoutState(BreakoutStatus.FALSE_BREAKOUT, direction, breakout_price, breakout_time, consolidation)
            if not retested and distance_pct <= retest_tolerance_pct:
                retested = True
            elif retested and close > edge:
                return BreakoutState(BreakoutStatus.TRUE_BREAKOUT, direction, breakout_price, breakout_time, consolidation)
        else:
            if close > consolidation.low:
                return BreakoutState(BreakoutStatus.FALSE_BREAKOUT, direction, breakout_price, breakout_time, consolidation)
            if not retested and distance_pct <= retest_tolerance_pct:
                retested = True
            elif retested and close < edge:
                return BreakoutState(BreakoutStatus.TRUE_BREAKOUT, direction, breakout_price, breakout_time, consolidation)

    if direction is None:
        return BreakoutState(BreakoutStatus.NONE, None, None, None, consolidation)
    return BreakoutState(BreakoutStatus.PENDING_RETEST, direction, breakout_price, breakout_time, consolidation)
