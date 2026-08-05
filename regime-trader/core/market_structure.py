"""Market structure: swing sequence, trend classification, and break-of-
structure (BOS) vs. market-structure-shift (MSS) detection.

Two concepts, both computed only from confirmed swing pivots plus the
current bar's own close (never a future bar - see the look-ahead tests):

- Trend: classified from the last two swing highs and last two swing lows.
  Higher-high + higher-low => uptrend. Lower-high + lower-low => downtrend.
  Anything else (not enough swings yet, or a mixed pattern) => ranging.
- BOS vs. MSS, evaluated against the *current* close relative to the most
  recent opposing-side swing:
    - In an uptrend, the last swing low is the structure "floor" and the
      last swing high is the "ceiling". Close breaking above the ceiling
      is a BOS (bullish continuation - the uptrend is extending). Close
      breaking below the floor is an MSS (bearish reversal - the higher-low
      pattern that defined the uptrend has just failed).
    - In a downtrend it's the mirror image: breaking below the floor (last
      swing low) is a BOS (bearish continuation); breaking above the
      ceiling (last swing high) is an MSS (bullish reversal).
    - In a ranging market there's no established floor/ceiling to break,
      so no event is classified.

Swing pivots reuse core.support_resistance.find_pivot_highs/find_pivot_lows,
so they inherit the same look-ahead discipline: a pivot is only usable
`pivot_window` bars after it forms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from core.support_resistance import find_pivot_highs, find_pivot_lows


class Trend(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"


class StructureEvent(str, Enum):
    BOS_BULLISH = "bos_bullish"  # uptrend continuation
    BOS_BEARISH = "bos_bearish"  # downtrend continuation
    MSS_BULLISH = "mss_bullish"  # reversal up (out of a downtrend, or out of ranging)
    MSS_BEARISH = "mss_bearish"  # reversal down (out of an uptrend, or out of ranging)
    NONE = "none"


@dataclass
class SwingPoint:
    price: float
    kind: str  # "high" or "low"
    timestamp: pd.Timestamp


@dataclass
class MarketStructureState:
    trend: Trend
    last_event: StructureEvent
    last_event_price: float | None
    last_swing_high: SwingPoint | None
    last_swing_low: SwingPoint | None


def detect_swings(bars: pd.DataFrame, pivot_window: int = 5) -> list[SwingPoint]:
    """Confirmed swing highs/lows in `bars`, in time order.

    Only pivots confirmable within the given history are returned - never
    one within the trailing `pivot_window` bars, since those can't be
    confirmed yet (identical rule to SupportResistanceDetector.detect).
    """
    if len(bars) < 2 * pivot_window + 1:
        return []

    pivot_high_mask = find_pivot_highs(bars["high"], pivot_window)
    pivot_low_mask = find_pivot_lows(bars["low"], pivot_window)
    confirmable = bars.index[: len(bars) - pivot_window]

    highs = bars.loc[confirmable, "high"][pivot_high_mask.loc[confirmable]]
    lows = bars.loc[confirmable, "low"][pivot_low_mask.loc[confirmable]]

    swings = [SwingPoint(price=float(price), kind="high", timestamp=ts) for ts, price in highs.items()]
    swings += [SwingPoint(price=float(price), kind="low", timestamp=ts) for ts, price in lows.items()]
    swings.sort(key=lambda s: s.timestamp)
    return swings


class MarketStructureAnalyzer:
    """Classifies trend and the most recent BOS/MSS event from `bars`.

    Args:
        pivot_window: Passed to detect_swings - bars on each side a swing
            high/low must beat to count as a pivot.
    """

    def __init__(self, pivot_window: int = 5) -> None:
        self.pivot_window = pivot_window

    def analyze(self, bars: pd.DataFrame) -> MarketStructureState:
        """Analyze `bars` (OHLC, most recent bar last) as of its last bar.

        Re-derives trend + event from scratch each call (like every other
        stateless computation in this project) rather than maintaining
        incremental state across calls - the caller is expected to pass
        `bars.loc[:t]` at each decision point, exactly as the rest of the
        strategy does.
        """
        swings = detect_swings(bars, self.pivot_window)
        highs = [s for s in swings if s.kind == "high"]
        lows = [s for s in swings if s.kind == "low"]

        trend = self._classify_trend(highs, lows)
        last_high = highs[-1] if highs else None
        last_low = lows[-1] if lows else None
        event, event_price = self._classify_event(bars, trend, last_high, last_low)

        return MarketStructureState(
            trend=trend,
            last_event=event,
            last_event_price=event_price,
            last_swing_high=last_high,
            last_swing_low=last_low,
        )

    @staticmethod
    def _classify_trend(highs: list[SwingPoint], lows: list[SwingPoint]) -> Trend:
        if len(highs) < 2 or len(lows) < 2:
            return Trend.RANGING
        high_rising = highs[-1].price > highs[-2].price
        low_rising = lows[-1].price > lows[-2].price
        if high_rising and low_rising:
            return Trend.UPTREND
        if not high_rising and not low_rising:
            return Trend.DOWNTREND
        return Trend.RANGING

    @staticmethod
    def _classify_event(
        bars: pd.DataFrame, trend: Trend, last_high: SwingPoint | None, last_low: SwingPoint | None
    ) -> tuple[StructureEvent, float | None]:
        if trend == Trend.RANGING or last_high is None or last_low is None:
            return StructureEvent.NONE, None

        close = float(bars["close"].iloc[-1])
        if trend == Trend.UPTREND:
            if close > last_high.price:
                return StructureEvent.BOS_BULLISH, last_high.price
            if close < last_low.price:
                return StructureEvent.MSS_BEARISH, last_low.price
        else:  # DOWNTREND
            if close < last_low.price:
                return StructureEvent.BOS_BEARISH, last_low.price
            if close > last_high.price:
                return StructureEvent.MSS_BULLISH, last_high.price
        return StructureEvent.NONE, None
