"""Key levels beyond pivot-clustered support/resistance: previous day's
high/low, the pre-market session's high/low, and a simplified order-block
heuristic.

All three are look-ahead safe by construction:

- previous_day_levels: a day's value is always yesterday's completed
  high/low, computed via a groupby + shift, so it's fixed before that day's
  first bar even happens.
- premarket_levels: a running cumulative max/min *within* the pre-market
  window (never the pre-market session's eventual final high/low handed
  out early), frozen for the rest of that same day once the regular
  session starts.
- order_blocks: a candle is only ever labeled an order block once the
  displacement move that identifies it has actually happened.

premarket_levels only produces real values when `bars` actually contains
extended-hours bars (e.g. Alpaca's intraday data with extended_hours=True);
on daily bars, or intraday data without a pre-market feed, it's all NaN -
callers should treat that as "not available", not "premarket = 0".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import ta

from core.support_resistance import Level


def previous_day_levels(bars: pd.DataFrame) -> pd.DataFrame:
    """Previous calendar day's high/low, aligned to every bar of `bars`.

    Returns a DataFrame (index matches `bars.index`) with columns
    `prev_day_high` / `prev_day_low`. NaN for the first calendar day in
    `bars` (no prior day available).
    """
    day = bars.index.normalize()
    daily = bars.groupby(day).agg(day_high=("high", "max"), day_low=("low", "min"))
    prev = daily.shift(1)
    aligned = prev.reindex(day)
    aligned.index = bars.index
    return aligned.rename(columns={"day_high": "prev_day_high", "day_low": "prev_day_low"})


def premarket_levels(bars: pd.DataFrame, market_open_time: str = "09:30") -> pd.DataFrame:
    """Running pre-market high/low for each calendar day, aligned to every
    bar of `bars`.

    Returns a DataFrame with columns `premarket_high` / `premarket_low`.
    During the pre-market window itself, the value is the cumulative
    max/min *so far that morning* (not the eventual pre-market close) -
    after `market_open_time`, it's frozen at the pre-market session's final
    value for the rest of that day.

    On daily-or-slower bars, "pre-market" is meaningless - and one bar per
    calendar day is conventionally timestamped at midnight, which would
    otherwise look like "before market open" for every single bar. Detected
    the same way core.sr_strategy.session_vwap detects daily bars (no two
    bars sharing a calendar date), and returns all-NaN in that case.
    """
    day = bars.index.normalize()
    is_intraday = day.nunique() < len(bars.index)
    if not is_intraday:
        return pd.DataFrame({"premarket_high": float("nan"), "premarket_low": float("nan")}, index=bars.index)

    open_time = pd.Timestamp(market_open_time).time()
    is_premarket = bars.index.time < open_time

    pm_high = bars["high"].where(is_premarket)
    pm_low = bars["low"].where(is_premarket)

    running_max = pm_high.groupby(day).cummax().groupby(day).ffill()
    running_min = pm_low.groupby(day).cummin().groupby(day).ffill()

    return pd.DataFrame({"premarket_high": running_max, "premarket_low": running_min}, index=bars.index)


def order_blocks(
    bars: pd.DataFrame,
    lookback: int = 3,
    displacement_atr_mult: float = 1.5,
    atr_period: int = 14,
) -> list[Level]:
    """Simplified order-block heuristic (not the full ICT toolkit - no fair
    value gap / imbalance / mitigation logic).

    A "displacement" bar is one whose range is at least
    `displacement_atr_mult` times the trailing `atr_period`-bar ATR. The
    order block is the last opposite-colored candle in the `lookback` bars
    immediately before it:

    - A strong up-displacement bar's last preceding down-close (red) candle
      is a bullish order block (a support zone - the last real selling
      before the move that broke it).
    - A strong down-displacement bar's last preceding up-close (green)
      candle is a bearish order block (a resistance zone).

    Each is returned once, priced at the order-block candle's own
    high/low midpoint, touches=1 (order blocks aren't validated by repeat
    touches the way pivot-clustered levels are - they're validated by the
    displacement that follows them).
    """
    if len(bars) <= atr_period:
        return []

    atr = ta.volatility.AverageTrueRange(bars["high"], bars["low"], bars["close"], window=atr_period).average_true_range()
    bar_range = bars["high"] - bars["low"]
    is_up_candle = bars["close"] > bars["open"]

    levels: list[Level] = []
    n = len(bars)
    for i in range(atr_period, n):
        atr_i = atr.iloc[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue
        if bar_range.iloc[i] < displacement_atr_mult * atr_i:
            continue

        is_up_move = bool(is_up_candle.iloc[i])
        for j in range(i - 1, max(i - 1 - lookback, -1), -1):
            if is_up_move and not is_up_candle.iloc[j]:
                price = float((bars["high"].iloc[j] + bars["low"].iloc[j]) / 2)
                ts = bars.index[j]
                levels.append(Level(price=price, kind="support", touches=1, first_touch=ts, last_touch=ts, source="order_block"))
                break
            if not is_up_move and is_up_candle.iloc[j]:
                price = float((bars["high"].iloc[j] + bars["low"].iloc[j]) / 2)
                ts = bars.index[j]
                levels.append(
                    Level(price=price, kind="resistance", touches=1, first_touch=ts, last_touch=ts, source="order_block")
                )
                break
    return levels


def latest_key_levels(
    bars: pd.DataFrame,
    market_open_time: str = "09:30",
    order_block_lookback: int = 3,
    order_block_displacement_atr_mult: float = 1.5,
    order_block_atr_period: int = 14,
) -> list[Level]:
    """Convenience: previous-day + pre-market levels as of `bars`' last bar,
    plus every order block found in `bars`, as a single `Level` list ready
    to merge with pivot-clustered support/resistance zones.

    Previous-day / pre-market values that are NaN (not enough history, or
    no pre-market data in `bars`) are silently skipped rather than turned
    into a bogus zero-price level.
    """
    levels: list[Level] = []
    ts = bars.index[-1]

    prev_day = previous_day_levels(bars).iloc[-1]
    if pd.notna(prev_day["prev_day_high"]):
        levels.append(
            Level(price=float(prev_day["prev_day_high"]), kind="resistance", touches=1, first_touch=ts, last_touch=ts, source="prev_day")
        )
    if pd.notna(prev_day["prev_day_low"]):
        levels.append(
            Level(price=float(prev_day["prev_day_low"]), kind="support", touches=1, first_touch=ts, last_touch=ts, source="prev_day")
        )

    premarket = premarket_levels(bars, market_open_time).iloc[-1]
    if pd.notna(premarket["premarket_high"]):
        levels.append(
            Level(price=float(premarket["premarket_high"]), kind="resistance", touches=1, first_touch=ts, last_touch=ts, source="premarket")
        )
    if pd.notna(premarket["premarket_low"]):
        levels.append(
            Level(price=float(premarket["premarket_low"]), kind="support", touches=1, first_touch=ts, last_touch=ts, source="premarket")
        )

    levels.extend(
        order_blocks(bars, order_block_lookback, order_block_displacement_atr_mult, order_block_atr_period)
    )
    return levels
