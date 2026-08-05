"""Classic technical-indicator entry/exit signals.

Computes RSI, MACD, a fast/slow SMA crossover, and Bollinger Bands directly
from OHLCV price data (the same indicators most charting platforms,
including TradingView, are built around - no external platform needed,
everything here runs on the bars this bot already pulls from Alpaca) and
combines them by confluence voting: a BULLISH or BEARISH call only fires
when enough of the four independently agree, which avoids over-trusting any
single indicator.

This produces a *directional* read only (bullish/bearish/neutral, with a
confidence score) - core.options_strategy turns that into an actual sized
options trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import ta


class Direction(str, Enum):
    """Directional read from the indicator confluence."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class IndicatorSignal:
    """Result of one confluence read: direction, confidence, and the
    individual indicator votes that produced it (for logging/debugging)."""

    symbol: str
    direction: Direction
    confidence: float
    votes: dict[str, Direction]
    timestamp: pd.Timestamp
    reasoning: str
    metadata: dict = field(default_factory=dict)


def rsi_vote(close: pd.Series, period: int, oversold: float, overbought: float) -> tuple[Direction, float]:
    """RSI oversold/overbought: below `oversold` is a bullish (bounce) vote,
    above `overbought` is bearish. Returns (vote, latest RSI value)."""
    rsi = ta.momentum.RSIIndicator(close, window=period).rsi()
    value = float(rsi.iloc[-1])
    if value < oversold:
        return Direction.BULLISH, value
    if value > overbought:
        return Direction.BEARISH, value
    return Direction.NEUTRAL, value


def macd_vote(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[Direction, float]:
    """MACD histogram sign: positive (MACD above its signal line) is
    bullish, negative is bearish. Returns (vote, latest histogram value)."""
    macd = ta.trend.MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
    histogram = float(macd.macd_diff().iloc[-1])
    return (Direction.BULLISH if histogram > 0 else Direction.BEARISH), histogram


def sma_crossover_vote(close: pd.Series, fast: int, slow: int) -> tuple[Direction, float]:
    """Fast SMA above slow SMA is a bullish (uptrend) vote, below is bearish.
    Returns (vote, fast - slow spread as a fraction of price)."""
    fast_sma = close.rolling(fast).mean().iloc[-1]
    slow_sma = close.rolling(slow).mean().iloc[-1]
    spread_pct = (fast_sma - slow_sma) / slow_sma
    return (Direction.BULLISH if fast_sma > slow_sma else Direction.BEARISH), float(spread_pct)


def bollinger_vote(close: pd.Series, period: int, num_std: float) -> tuple[Direction, float]:
    """Price near/below the lower Bollinger Band is a bullish (oversold
    bounce) vote, near/above the upper band is bearish. Middle third of the
    band is neutral. Returns (vote, %B - 0.0 at lower band, 1.0 at upper)."""
    bb = ta.volatility.BollingerBands(close, window=period, window_dev=num_std)
    upper, lower = bb.bollinger_hband().iloc[-1], bb.bollinger_lband().iloc[-1]
    price = float(close.iloc[-1])
    percent_b = (price - lower) / (upper - lower) if upper != lower else 0.5
    if percent_b <= 0.2:
        return Direction.BULLISH, percent_b
    if percent_b >= 0.8:
        return Direction.BEARISH, percent_b
    return Direction.NEUTRAL, percent_b


class IndicatorSignalGenerator:
    """Combines RSI/MACD/SMA-crossover/Bollinger votes into one directional
    confluence signal.

    Args:
        rsi_period: RSI lookback.
        rsi_oversold / rsi_overbought: RSI thresholds for a bullish/bearish vote.
        macd_fast / macd_slow / macd_signal: MACD EMA periods.
        sma_fast / sma_slow: Fast/slow SMA periods for the crossover vote.
        bb_period / bb_std: Bollinger Band period and width (in std devs).
        min_confluence: Minimum number of the 4 indicators that must agree
            for a non-neutral signal to fire (out of 4).
        min_bars: Minimum bars of history required to compute all indicators.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_fast: int = 20,
        sma_slow: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        min_confluence: int = 3,
        min_bars: int = 60,
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.min_confluence = min_confluence
        self.min_bars = min_bars

    def generate(self, symbol: str, bars: pd.DataFrame) -> IndicatorSignal:
        """Compute the confluence signal for the latest bar of `bars`.

        Args:
            symbol: Ticker the bars belong to.
            bars: OHLCV history, most recent bar last. Needs at least
                `min_bars` rows (dominated by sma_slow's warm-up).
        """
        if len(bars) < self.min_bars:
            return IndicatorSignal(
                symbol=symbol,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                votes={},
                timestamp=bars.index[-1] if len(bars) else pd.Timestamp.now(tz="UTC"),
                reasoning=f"not enough history ({len(bars)} bars, need {self.min_bars})",
            )

        close = bars["close"]
        rsi_dir, rsi_val = rsi_vote(close, self.rsi_period, self.rsi_oversold, self.rsi_overbought)
        macd_dir, macd_val = macd_vote(close, self.macd_fast, self.macd_slow, self.macd_signal)
        sma_dir, sma_val = sma_crossover_vote(close, self.sma_fast, self.sma_slow)
        bb_dir, bb_val = bollinger_vote(close, self.bb_period, self.bb_std)

        votes = {"rsi": rsi_dir, "macd": macd_dir, "sma_crossover": sma_dir, "bollinger": bb_dir}
        bullish_count = sum(1 for v in votes.values() if v == Direction.BULLISH)
        bearish_count = sum(1 for v in votes.values() if v == Direction.BEARISH)

        if bullish_count >= self.min_confluence:
            direction, confidence = Direction.BULLISH, bullish_count / len(votes)
        elif bearish_count >= self.min_confluence:
            direction, confidence = Direction.BEARISH, bearish_count / len(votes)
        else:
            direction, confidence = Direction.NEUTRAL, max(bullish_count, bearish_count) / len(votes)

        return IndicatorSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            votes=votes,
            timestamp=bars.index[-1],
            reasoning=f"{bullish_count} bullish / {bearish_count} bearish of {len(votes)} indicators (need {self.min_confluence})",
            metadata={"rsi": rsi_val, "macd_histogram": macd_val, "sma_spread_pct": sma_val, "bollinger_percent_b": bb_val},
        )
