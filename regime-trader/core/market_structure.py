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

A third concept lives here too: trend "legs" (the price move between one
confirmed swing and the next) and the *current*, still-forming leg's speed
relative to recent confirmed legs. A trend's final push toward its target is
typically its fastest - `current_leg_acceleration` gives a ratio (current
leg speed / recent average leg speed) so callers can flag that condition and
actually trade it, rather than avoid it.
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


@dataclass
class TrendLeg:
    """The confirmed price move from one swing point to the next."""

    start: SwingPoint
    end: SwingPoint
    bar_count: int
    price_change: float
    speed: float  # abs(price_change) / bar_count


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


def identify_legs(bars: pd.DataFrame, pivot_window: int = 5) -> list[TrendLeg]:
    """Confirmed trend legs: the move from each swing point to the next,
    in time order (regardless of whether both ends are the same kind of
    swing - consecutive pivots normally alternate high/low, but this
    doesn't assume it).
    """
    swings = detect_swings(bars, pivot_window)
    if len(swings) < 2:
        return []

    legs = []
    for start, end in zip(swings[:-1], swings[1:]):
        bar_count = bars.index.get_loc(end.timestamp) - bars.index.get_loc(start.timestamp)
        if bar_count <= 0:
            continue
        price_change = end.price - start.price
        legs.append(TrendLeg(start=start, end=end, bar_count=bar_count, price_change=price_change, speed=abs(price_change) / bar_count))
    return legs


def current_leg_acceleration(bars: pd.DataFrame, pivot_window: int = 5, lookback_legs: int = 3) -> float | None:
    """Ratio of the CURRENT, still-forming leg's speed (from the last
    confirmed swing to `bars`' last close) to the average speed of the last
    `lookback_legs` confirmed legs.

    A ratio well above 1 means the live move is outrunning the trend's
    recent typical leg - the classic signature of a trend's final,
    accelerating push toward its target (the "last leg" - the video notes
    call this the fastest, and best, move to be trading, not one to avoid).

    Returns None when there isn't enough leg history to compare against
    (fewer than `lookback_legs` confirmed legs), or the reference average
    speed is zero.
    """
    swings = detect_swings(bars, pivot_window)
    legs = identify_legs(bars, pivot_window)
    if not swings or len(legs) < lookback_legs:
        return None

    recent_legs = legs[-lookback_legs:]
    avg_speed = sum(leg.speed for leg in recent_legs) / len(recent_legs)
    if avg_speed <= 0:
        return None

    last_swing = swings[-1]
    bar_count = (len(bars) - 1) - bars.index.get_loc(last_swing.timestamp)
    if bar_count <= 0:
        return None

    current_speed = abs(float(bars["close"].iloc[-1]) - last_swing.price) / bar_count
    return current_speed / avg_speed


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
