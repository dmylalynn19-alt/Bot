"""Single-candle and three-candle price-action primitives: range/speed
(momentum) and the three-bar entry model (lead / reaction / confirmation).

Candle "speed": how large a bar's range is relative to its recent typical
range (ATR). A fast/high-momentum bar is one that moved further in a single
bar than usual - the video notes call this out directly as something worth
trading for, options especially, since a fast move reaches a profitable
strike distance before theta decay eats the premium.

Three-bar entry model: a lead candle (a fast, strongly directional bar),
followed by a reaction candle (a smaller pullback against the lead - the
market "catching its breath", not reversing), followed by a confirmation
candle that breaks back past the reaction candle's extreme in the lead's
direction (proof the pullback is over and the original move is resuming).
This is a purely mechanical, three-bar pattern - it says nothing about
*where* price is (that's what core.support_resistance / core.consolidation
are for); core.breakout_strategy combines both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import ta


class ThreeBarDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class ThreeBarSignal:
    direction: ThreeBarDirection
    lead_index: pd.Timestamp
    reaction_index: pd.Timestamp
    confirmation_index: pd.Timestamp
    lead_speed: float  # lead candle's range / ATR at the time


def candle_range(bars: pd.DataFrame) -> pd.Series:
    """Each bar's high-low range."""
    return bars["high"] - bars["low"]


def candle_body(bars: pd.DataFrame) -> pd.Series:
    """Each bar's absolute open-close body size."""
    return (bars["close"] - bars["open"]).abs()


def is_bullish_candle(bars: pd.DataFrame) -> pd.Series:
    """True where a bar closed above where it opened."""
    return bars["close"] > bars["open"]


def candle_speed(bars: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """Each bar's range as a multiple of the trailing `atr_period`-bar ATR -
    a momentum measure. >1 means the bar moved further than its recent
    typical range; well above 1 (see `momentum_multiple` in
    `three_bar_entry`) is a genuinely fast, high-momentum bar.

    The ATR itself is trailing/causal (ta.volatility.AverageTrueRange over
    bars up to and including bar i), so a bar's speed only ever depends on
    bars at or before it - look-ahead safe.
    """
    atr = ta.volatility.AverageTrueRange(bars["high"], bars["low"], bars["close"], window=atr_period).average_true_range()
    return candle_range(bars) / atr


def three_bar_entry(
    bars: pd.DataFrame,
    atr_period: int = 14,
    momentum_multiple: float = 1.3,
    max_reaction_retrace_pct: float = 0.75,
) -> ThreeBarSignal | None:
    """Check whether the LAST THREE bars of `bars` form a valid three-bar
    entry: lead candle (fast, directional) -> reaction candle (a smaller
    pullback that doesn't erase most of the lead's move) -> confirmation
    candle (breaks past the reaction candle's extreme, continuing the
    lead's direction).

    Args:
        atr_period: ATR lookback for `candle_speed`.
        momentum_multiple: The lead candle's speed (range / ATR) must be at
            least this - it has to actually be a fast bar, not just any
            directional one.
        max_reaction_retrace_pct: The reaction candle's body can retrace at
            most this fraction of the lead candle's body - too deep a
            pullback isn't "catching its breath", it's a real reversal
            attempt, and disqualifies the pattern.

    Returns None if there isn't enough history, or the last three bars
    don't form a valid pattern in either direction. Only ever looks at
    `bars` itself (the last three rows) plus the ATR computed from bars up
    to the lead candle - no look-ahead.
    """
    if len(bars) < atr_period + 3:
        return None

    lead = bars.iloc[-3]
    reaction = bars.iloc[-2]
    confirmation = bars.iloc[-1]

    speed = candle_speed(bars.iloc[:-2], atr_period)  # ATR as of the lead candle, not later
    lead_speed = float(speed.iloc[-1])
    if pd.isna(lead_speed) or lead_speed < momentum_multiple:
        return None

    lead_bullish = lead["close"] > lead["open"]
    lead_body = abs(lead["close"] - lead["open"])
    if lead_body <= 0:
        return None

    reaction_body = abs(reaction["close"] - reaction["open"])
    reaction_against_lead = (reaction["close"] < reaction["open"]) if lead_bullish else (reaction["close"] > reaction["open"])
    if not reaction_against_lead:
        return None
    if reaction_body > lead_body * max_reaction_retrace_pct:
        return None

    if lead_bullish:
        confirmed = confirmation["close"] > reaction["high"]
        direction = ThreeBarDirection.BULLISH
    else:
        confirmed = confirmation["close"] < reaction["low"]
        direction = ThreeBarDirection.BEARISH
    if not confirmed:
        return None

    return ThreeBarSignal(
        direction=direction,
        lead_index=bars.index[-3],
        reaction_index=bars.index[-2],
        confirmation_index=bars.index[-1],
        lead_speed=lead_speed,
    )
