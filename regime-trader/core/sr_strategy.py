"""Support/resistance trade setups, confirmed by volume, VWAP, RSI, and MACD.

The core idea: a bounce off support (or rejection at resistance) is only
tradeable when it's confirmed, not just present. All four confirmations are
required by default (configurable via `min_required_confirmations`) since
volume and VWAP in particular were called out as must-haves, not just
votes:

- Volume: the entry bar's volume must be elevated vs. its trailing average -
  a bounce/rejection on thin volume is not trustworthy.
- VWAP: price must be on the correct side of VWAP for the trade direction -
  reclaiming/holding it for longs, rejected below it for shorts.
- RSI: interpreted relative to *which* level is in play - oversold at
  support confirms a long, overbought at resistance confirms a short (not
  the other way around - RSI's meaning here depends on the level, exactly
  as it should for a support/resistance strategy).
- MACD: histogram must be turning in the trade's direction (today's value
  vs. yesterday's) - not necessarily net positive/negative yet, since MACD
  lags and rarely has fully crossed by the time RSI is at an extreme right
  at the level. Momentum shifting your way is the confirmation.

On top of those four, two more gates (both on by default):

- Market structure: the entry timeframe's own market structure (see
  core/market_structure.py) must show a BOS or MSS event in the trade's
  favor right at the level - a bullish event (continuation of an uptrend,
  or a reversal shift up) for a long, bearish for a short. This is what
  turns "price happens to be near a level and the indicators agree" into
  "the market has actually just confirmed a trend/reversal there."
- Higher-timeframe bias: when `generate()` is given `htf_bars` (a higher
  timeframe's history of the same symbol), a long is only taken if the
  higher timeframe isn't in a downtrend, and a short only if it isn't in
  an uptrend (a ranging higher timeframe doesn't veto either direction -
  there's no bias to contradict). This is the "higher timeframe for bias,
  lower timeframe for the entry" pairing - e.g. 1H bias / 5-minute entries
  for day trading, daily bias / 1H entries for swing trading.

Levels themselves come from two sources, merged into one candidate pool:
pivot-clustered support/resistance (core/support_resistance.py) and, when
enabled, "key levels" - previous day's high/low, the pre-market session's
high/low, and a simplified order-block heuristic (core/key_levels.py).

Works on whatever timeframe `bars` is in - a 5-minute chart for day trading,
a daily chart for swing trading - the setup is only valid if it's confirmed
within that same timeframe's own history (see backtest/sr_backtester.py for
walking this forward without look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import ta

from core.key_levels import latest_key_levels
from core.market_structure import MarketStructureAnalyzer, StructureEvent, Trend
from core.support_resistance import Level, SupportResistanceDetector

_BULLISH_STRUCTURE_EVENTS = {StructureEvent.BOS_BULLISH, StructureEvent.MSS_BULLISH}
_BEARISH_STRUCTURE_EVENTS = {StructureEvent.BOS_BEARISH, StructureEvent.MSS_BEARISH}


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeSetup:
    """A confirmed, tradeable support/resistance setup."""

    symbol: str
    direction: TradeDirection
    level: Level
    entry_price: float
    stop_price: float
    target_price: float
    confirmations: dict[str, bool]
    confidence: float
    timestamp: pd.Timestamp
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


def session_vwap(bars: pd.DataFrame, rolling_window: int = 20) -> pd.Series:
    """Volume-weighted average price.

    On intraday bars (more than one bar sharing a calendar day), this is
    the standard session VWAP, reset every calendar day. On daily-or-slower
    bars a calendar-day reset is meaningless - each bar *is* its own
    "session", so the reset version degenerates to just that bar's own
    typical price, which carries no real multi-bar signal. In that case
    this falls back to a rolling `rolling_window`-bar volume-weighted
    average instead - the standard swing-trading adaptation of VWAP - so
    "price vs. VWAP" reflects an actual trend/level rather than noise.

    Which case applies is auto-detected from `bars.index` itself (more than
    one bar per calendar date => intraday), so callers don't need to know
    or declare their own timeframe.
    """
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = typical_price * bars["volume"]

    is_intraday = bars.index.normalize().nunique() < len(bars.index)
    if is_intraday:
        day = bars.index.normalize()
        cum_pv = pv.groupby(day).cumsum()
        cum_vol = bars["volume"].groupby(day).cumsum()
        return cum_pv / cum_vol

    rolling_pv = pv.rolling(rolling_window).sum()
    rolling_vol = bars["volume"].rolling(rolling_window).sum()
    return rolling_pv / rolling_vol


def volume_confirmation(volume: pd.Series, window: int, multiple: float) -> pd.Series:
    """True where a bar's volume exceeds `multiple`x its trailing `window`-bar
    average (the average excludes the bar itself, so a spike can't inflate
    its own baseline)."""
    avg_volume = volume.shift(1).rolling(window).mean()
    return volume > avg_volume * multiple


class SupportResistanceStrategy:
    """Detects and confirms support/resistance bounce/rejection setups.

    Args:
        pivot_window / cluster_tolerance_pct / min_touches: passed to
            SupportResistanceDetector.
        level_proximity_pct: How close price must be to a level to consider
            a setup at all.
        volume_window / volume_multiple: Entry bar's volume must be at least
            `volume_multiple`x the trailing `volume_window`-bar average.
        vwap_window: Only used on daily-or-slower bars (intraday bars use
            the standard calendar-day session VWAP instead) - the rolling
            window, in bars, for the volume-weighted average price
            fallback. Kept short (default 10, ~2 trading weeks) rather than
            matching the other lookbacks: a fresh bounce off support is, by
            definition, still below a slow-moving average, so a long
            rolling window would make "price reclaimed VWAP" nearly
            impossible to satisfy on the very bar RSI is oversold.
        rsi_period / rsi_oversold / rsi_overbought: RSI thresholds - near
            support, RSI at/below `rsi_oversold` confirms a long; near
            resistance, RSI at/above `rsi_overbought` confirms a short.
        macd_fast / macd_slow / macd_signal: MACD periods; histogram must be
            turning in the trade's direction vs. the prior bar (see
            module docstring for why this isn't a strict sign check).
        min_required_confirmations: How many of {volume, vwap, rsi, macd}
            must pass (out of 4) - defaults to requiring all four, since
            volume and VWAP were both called out as must-confirm, not
            optional votes.
        risk_reward_ratio: Target distance from entry, as a multiple of the
            stop distance, used when there's no opposing level to target.
        min_bars: Minimum bars of history required before evaluating a setup.
        use_key_levels: Merge previous-day high/low, pre-market high/low,
            and order-block zones (core/key_levels.py) into the candidate
            level pool alongside pivot-clustered support/resistance.
        premarket_open_time: Market open time (local to `bars`' timestamps,
            "HH:MM") used to separate the pre-market session for key levels.
        order_block_lookback / order_block_displacement_atr_mult /
            order_block_atr_period: Passed to core.key_levels.order_blocks.
        require_market_structure: Require the entry timeframe's own market
            structure (core/market_structure.py) to show a BOS or MSS event
            in the trade's favor at the level - not just the indicator
            confluence, but confirmation the market has actually broken
            structure that way.
        structure_pivot_window: Pivot window used for the entry-timeframe
            market structure analyzer. Defaults to `pivot_window`.
        require_htf_bias: When `generate()` is given `htf_bars`, veto a long
            if the higher timeframe is in a downtrend (and a short if it's
            in an uptrend). Has no effect if `htf_bars` isn't provided.
        htf_pivot_window: Pivot window used for the higher-timeframe bias
            analyzer (independent of the entry timeframe's own pivot_window,
            since higher-timeframe swings are usually meant to be read on a
            slower cadence).
    """

    def __init__(
        self,
        pivot_window: int = 5,
        cluster_tolerance_pct: float = 0.005,
        min_touches: int = 2,
        level_proximity_pct: float = 0.005,
        volume_window: int = 20,
        volume_multiple: float = 1.2,
        vwap_window: int = 10,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        min_required_confirmations: int = 4,
        risk_reward_ratio: float = 2.0,
        min_bars: int = 60,
        use_key_levels: bool = True,
        premarket_open_time: str = "09:30",
        order_block_lookback: int = 3,
        order_block_displacement_atr_mult: float = 1.5,
        order_block_atr_period: int = 14,
        require_market_structure: bool = True,
        structure_pivot_window: int | None = None,
        require_htf_bias: bool = True,
        htf_pivot_window: int = 5,
    ) -> None:
        self.detector = SupportResistanceDetector(pivot_window, cluster_tolerance_pct, min_touches)
        self.level_proximity_pct = level_proximity_pct
        self.volume_window = volume_window
        self.volume_multiple = volume_multiple
        self.vwap_window = vwap_window
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.min_required_confirmations = min_required_confirmations
        self.risk_reward_ratio = risk_reward_ratio
        self.min_bars = min_bars
        self.use_key_levels = use_key_levels
        self.premarket_open_time = premarket_open_time
        self.order_block_lookback = order_block_lookback
        self.order_block_displacement_atr_mult = order_block_displacement_atr_mult
        self.order_block_atr_period = order_block_atr_period
        self.require_market_structure = require_market_structure
        self.structure_analyzer = MarketStructureAnalyzer(structure_pivot_window or pivot_window)
        self.require_htf_bias = require_htf_bias
        self.htf_structure_analyzer = MarketStructureAnalyzer(htf_pivot_window)

    def generate(self, symbol: str, bars: pd.DataFrame, htf_bars: pd.DataFrame | None = None) -> TradeSetup | None:
        """Evaluate `bars` (OHLCV, most recent bar last, this timeframe's own
        history) for a confirmed support/resistance setup.

        Args:
            symbol: Instrument label.
            bars: The entry timeframe's own OHLCV history.
            htf_bars: Optional higher-timeframe OHLCV history of the same
                symbol, used for the `require_htf_bias` gate. Omit to skip
                the higher-timeframe check entirely (single-timeframe mode).

        Returns None if there's no history, no nearby level, higher-
        timeframe bias vetoes both directions, or too few confirmations
        pass. If both a long-at-support and a short-at-resistance setup
        pass at once (price sitting between two close levels), the higher-
        confidence one wins.
        """
        if len(bars) < self.min_bars:
            return None

        levels = list(self.detector.detect(bars))
        if self.use_key_levels:
            levels += latest_key_levels(
                bars,
                self.premarket_open_time,
                self.order_block_lookback,
                self.order_block_displacement_atr_mult,
                self.order_block_atr_period,
            )
        if not levels:
            return None

        htf_trend: Trend | None = None
        if htf_bars is not None and self.require_htf_bias:
            htf_trend = self.htf_structure_analyzer.analyze(htf_bars).trend

        price = float(bars["close"].iloc[-1])
        support = self.detector.nearest_support(levels, price)
        resistance = self.detector.nearest_resistance(levels, price)

        candidates = []
        if support is not None and htf_trend != Trend.DOWNTREND:
            long_setup = self._evaluate(symbol, bars, price, support, TradeDirection.LONG, htf_trend)
            if long_setup is not None:
                candidates.append(long_setup)
        if resistance is not None and htf_trend != Trend.UPTREND:
            short_setup = self._evaluate(symbol, bars, price, resistance, TradeDirection.SHORT, htf_trend)
            if short_setup is not None:
                candidates.append(short_setup)

        return max(candidates, key=lambda setup: setup.confidence) if candidates else None

    def _evaluate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        price: float,
        level: Level,
        direction: TradeDirection,
        htf_trend: Trend | None,
    ) -> TradeSetup | None:
        distance_pct = abs(price - level.price) / level.price
        if distance_pct > self.level_proximity_pct:
            return None

        structure_state = self.structure_analyzer.analyze(bars)
        if self.require_market_structure:
            favorable_events = _BULLISH_STRUCTURE_EVENTS if direction == TradeDirection.LONG else _BEARISH_STRUCTURE_EVENTS
            if structure_state.last_event not in favorable_events:
                return None

        close = bars["close"]
        rsi = float(ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi().iloc[-1])
        macd_hist_series = ta.trend.MACD(
            close, window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal
        ).macd_diff()
        macd_hist = float(macd_hist_series.iloc[-1])
        macd_hist_prev = float(macd_hist_series.iloc[-2])
        vwap = float(session_vwap(bars, rolling_window=self.vwap_window).iloc[-1])
        vol_confirmed = bool(volume_confirmation(bars["volume"], self.volume_window, self.volume_multiple).iloc[-1])

        # MACD is a lagging indicator - right at a fresh bounce/rejection its
        # histogram is usually still on the "wrong" side of zero (a decline
        # sharp enough to reach oversold RSI hasn't given MACD time to cross
        # yet). Requiring the histogram to already be net positive/negative
        # would make this confirmation almost never coincide with an RSI
        # extreme at the level. Instead this requires the histogram to be
        # *turning* the trade's way (today vs. yesterday) - genuine momentum
        # confirmation, still look-ahead-safe (both bars already closed).
        if direction == TradeDirection.LONG:
            confirmations = {
                "volume": vol_confirmed,
                "vwap": price >= vwap,
                "rsi": rsi <= self.rsi_oversold,
                "macd": macd_hist > macd_hist_prev,
            }
            stop_price = level.price * (1 - self.level_proximity_pct * 2)
        else:
            confirmations = {
                "volume": vol_confirmed,
                "vwap": price <= vwap,
                "rsi": rsi >= self.rsi_overbought,
                "macd": macd_hist < macd_hist_prev,
            }
            stop_price = level.price * (1 + self.level_proximity_pct * 2)

        passed = sum(confirmations.values())
        if passed < self.min_required_confirmations:
            return None

        risk = abs(price - stop_price)
        target_price = (
            price + risk * self.risk_reward_ratio if direction == TradeDirection.LONG else price - risk * self.risk_reward_ratio
        )

        return TradeSetup(
            symbol=symbol,
            direction=direction,
            level=level,
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            confirmations=confirmations,
            confidence=passed / len(confirmations),
            timestamp=bars.index[-1],
            reasoning=(
                f"{direction.value} at {level.kind} {level.price:.2f} ({level.touches} touches, "
                f"source={level.source}): {passed}/{len(confirmations)} confirmations {confirmations}, "
                f"structure={structure_state.trend.value}/{structure_state.last_event.value}"
                + (f", htf_bias={htf_trend.value}" if htf_trend is not None else "")
            ),
            metadata={
                "rsi": rsi,
                "macd_histogram": macd_hist,
                "vwap": vwap,
                "level_source": level.source,
                "market_structure_trend": structure_state.trend.value,
                "market_structure_event": structure_state.last_event.value,
                "htf_bias": htf_trend.value if htf_trend is not None else None,
            },
        )

    def should_exit(self, setup: TradeSetup, bars: pd.DataFrame) -> tuple[bool, str]:
        """Check whether an open position from `setup` should be closed:
        stop hit, target hit, or neither yet."""
        price = float(bars["close"].iloc[-1])
        if setup.direction == TradeDirection.LONG:
            if price <= setup.stop_price:
                return True, f"stop hit ({price:.2f} <= {setup.stop_price:.2f})"
            if price >= setup.target_price:
                return True, f"target hit ({price:.2f} >= {setup.target_price:.2f})"
        else:
            if price >= setup.stop_price:
                return True, f"stop hit ({price:.2f} >= {setup.stop_price:.2f})"
            if price <= setup.target_price:
                return True, f"target hit ({price:.2f} <= {setup.target_price:.2f})"
        return False, ""
