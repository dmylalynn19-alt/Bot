"""Vol-based allocation strategies.

DESIGN INSIGHT
--------------
The HMM (core.hmm_engine) is a volatility classifier, not a direction
predictor. Empirically, stocks trend upward most of the time during calm,
low-volatility periods, while the worst drawdowns cluster in high-volatility
spikes. The allocation policy follows directly from that:

- Low vol  -> be fully invested, even leveraged a little (calm markets trend up)
- Mid vol  -> stay invested if the trend is intact, reduce if it's broken
- High vol -> reduce exposure but stay partially invested, long only, no
              leverage - enough to catch a V-shaped rebound without betting
              short into a regime the HMM only knows is "turbulent", not
              "still falling"

Which of the three postures applies to a given regime is decided purely by
that regime's *volatility rank* among all fitted regimes - never by its
return-based label. A HMM regime labeled "BULL" is not assumed to be
low-volatility; see StrategyOrchestrator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import pandas as pd
import ta

from core.hmm_engine import RegimeInfo, RegimeLabel, RegimeState

logger = logging.getLogger(__name__)


def _ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def _atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    return ta.volatility.AverageTrueRange(
        bars["high"], bars["low"], bars["close"], window=window, fillna=False
    ).average_true_range()


class Direction(str, Enum):
    """Position direction. All three strategies below are long-only; FLAT
    is available for callers that want to represent "no position"."""

    LONG = "LONG"
    FLAT = "FLAT"


@dataclass
class Signal:
    """A single sized, regime-conditioned trading signal for one symbol."""

    symbol: str
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    position_size_pct: float
    leverage: float
    regime_id: int
    regime_name: RegimeLabel
    regime_probability: float
    timestamp: pd.Timestamp
    reasoning: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Common plumbing for regime-conditioned position-sizing strategies.

    Concrete strategies only compute a stop-loss, allocation, and leverage
    from price/regime context; `_make_signal` fills in the fields that come
    straight from `regime_state` so subclasses don't repeat them.
    """

    MIN_BARS = 50

    @abstractmethod
    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Signal | None:
        """Produce a sized position signal for `symbol`, or None if `bars` is too short."""
        raise NotImplementedError

    def _has_enough_bars(self, bars: pd.DataFrame) -> bool:
        return len(bars) >= self.MIN_BARS

    def _current_price(self, bars: pd.DataFrame) -> float:
        return float(bars["close"].iloc[-1])

    def _ema(self, bars: pd.DataFrame, window: int) -> float:
        return float(_ema(bars["close"], window).iloc[-1])

    def _atr(self, bars: pd.DataFrame, window: int) -> float:
        return float(_atr(bars, window).iloc[-1])

    def _make_signal(
        self,
        symbol: str,
        regime_state: RegimeState,
        *,
        entry_price: float,
        stop_loss: float,
        position_size_pct: float,
        leverage: float,
        reasoning: str,
        take_profit: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            strategy_name=type(self).__name__,
            metadata=metadata or {},
        )


class LowVolBullStrategy(BaseStrategy):
    """Lowest-volatility-third regimes: calm markets trend up most of the time.

    Full allocation with modest leverage - this is where most of the
    strategy's returns are generated (calm markets + leverage = compounding).
    """

    def __init__(
        self,
        allocation: float = 0.95,
        leverage: float = 1.25,
        atr_stop_mult: float = 3.0,
        ema_stop_mult: float = 0.5,
        ema_window: int = 50,
        atr_window: int = 14,
    ) -> None:
        self.allocation = allocation
        self.leverage = leverage
        self.atr_stop_mult = atr_stop_mult
        self.ema_stop_mult = ema_stop_mult
        self.ema_window = ema_window
        self.atr_window = atr_window

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Signal | None:
        if not self._has_enough_bars(bars):
            return None
        price = self._current_price(bars)
        ema = self._ema(bars, self.ema_window)
        atr = self._atr(bars, self.atr_window)
        stop_loss = max(price - self.atr_stop_mult * atr, ema - self.ema_stop_mult * atr)
        return self._make_signal(
            symbol,
            regime_state,
            entry_price=price,
            stop_loss=stop_loss,
            position_size_pct=self.allocation,
            leverage=self.leverage,
            reasoning="Low-volatility regime: calm markets trend up - full allocation with modest leverage.",
            metadata={f"ema_{self.ema_window}": ema, f"atr_{self.atr_window}": atr},
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Middle-volatility-third regimes: stay invested while the trend holds, reduce if not."""

    def __init__(
        self,
        allocation_trend: float = 0.95,
        allocation_no_trend: float = 0.60,
        leverage: float = 1.0,
        ema_stop_mult: float = 0.5,
        ema_window: int = 50,
        atr_window: int = 14,
    ) -> None:
        self.allocation_trend = allocation_trend
        self.allocation_no_trend = allocation_no_trend
        self.leverage = leverage
        self.ema_stop_mult = ema_stop_mult
        self.ema_window = ema_window
        self.atr_window = atr_window

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Signal | None:
        if not self._has_enough_bars(bars):
            return None
        price = self._current_price(bars)
        ema = self._ema(bars, self.ema_window)
        atr = self._atr(bars, self.atr_window)
        trend_intact = price > ema
        allocation = self.allocation_trend if trend_intact else self.allocation_no_trend
        stop_loss = ema - self.ema_stop_mult * atr
        reasoning = (
            f"Mid-volatility regime, price above {self.ema_window} EMA: trend intact - stay invested."
            if trend_intact
            else f"Mid-volatility regime, price below {self.ema_window} EMA: trend broken - reduce exposure."
        )
        return self._make_signal(
            symbol,
            regime_state,
            entry_price=price,
            stop_loss=stop_loss,
            position_size_pct=allocation,
            leverage=self.leverage,
            reasoning=reasoning,
            metadata={f"ema_{self.ema_window}": ema, f"atr_{self.atr_window}": atr, "trend_intact": trend_intact},
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Highest-volatility-third regimes: reduce but stay partially invested.

    Long only, no leverage - the point is to still catch a V-shaped rebound
    instead of being flat or short into a regime the HMM only knows is
    turbulent, not necessarily still falling.
    """

    def __init__(
        self,
        allocation: float = 0.60,
        leverage: float = 1.0,
        atr_stop_mult: float = 2.0,
        atr_window: int = 14,
    ) -> None:
        self.allocation = allocation
        self.leverage = leverage
        # Tighter than LowVolBullStrategy's 3x: higher realized volatility warrants a closer stop.
        self.atr_stop_mult = atr_stop_mult
        self.atr_window = atr_window

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Signal | None:
        if not self._has_enough_bars(bars):
            return None
        price = self._current_price(bars)
        atr = self._atr(bars, self.atr_window)
        stop_loss = price - self.atr_stop_mult * atr
        return self._make_signal(
            symbol,
            regime_state,
            entry_price=price,
            stop_loss=stop_loss,
            position_size_pct=self.allocation,
            leverage=self.leverage,
            reasoning="High-volatility regime: reduced long exposure to catch a V-shaped rebound, no leverage.",
            metadata={f"atr_{self.atr_window}": atr},
        )


# Backward-compatible aliases: earlier, label-driven naming for the same
# three strategy classes above.
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy


# Naive label -> strategy fallback, provided for convenience/back-compat only.
# StrategyOrchestrator does NOT use this: it ranks regimes by
# expected_volatility, ignoring their labels entirely, since a "BULL"-labeled
# regime is not assumed to be low-volatility.
LABEL_TO_STRATEGY: dict[RegimeLabel, type[BaseStrategy]] = {
    RegimeLabel.CRASH: CrashDefensiveStrategy,
    RegimeLabel.STRONG_BEAR: HighVolDefensiveStrategy,
    RegimeLabel.BEAR: BearTrendStrategy,
    RegimeLabel.WEAK_BEAR: MeanReversionStrategy,
    RegimeLabel.NEUTRAL: MeanReversionStrategy,
    RegimeLabel.WEAK_BULL: MeanReversionStrategy,
    RegimeLabel.BULL: BullTrendStrategy,
    RegimeLabel.STRONG_BULL: LowVolBullStrategy,
    RegimeLabel.EUPHORIA: EuphoriaCautiousStrategy,
}


def _strategy_class_for_vol_position(position: float) -> type[BaseStrategy]:
    """position: 0.0 = lowest-volatility regime, 1.0 = highest, per StrategyOrchestrator."""
    if position <= 0.33:
        return LowVolBullStrategy
    if position >= 0.67:
        return HighVolDefensiveStrategy
    return MidVolCautiousStrategy


class StrategyOrchestrator:
    """Maps each of the HMM's fitted regimes to a strategy, by volatility rank.

    Args:
        config: The `strategy:` section of settings.yaml (low_vol_allocation,
            mid_vol_allocation_trend, mid_vol_allocation_no_trend,
            high_vol_allocation, low_vol_leverage, rebalance_threshold,
            uncertainty_size_mult).
        regime_infos: RegimeInfo per label, as produced by
            HMMEngine.regime_info_ after fit().
        min_confidence: Minimum posterior probability required to act on a
            regime call without triggering uncertainty mode.
    """

    def __init__(
        self,
        config: dict[str, float],
        regime_infos: dict[RegimeLabel, RegimeInfo],
        min_confidence: float = 0.55,
    ) -> None:
        self.config = config
        self.min_confidence = min_confidence
        self.regime_infos: dict[RegimeLabel, RegimeInfo] = {}
        self._strategy_by_state_id: dict[int, BaseStrategy] = {}
        self._vol_rank_by_state_id: dict[int, float] = {}
        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: dict[RegimeLabel, RegimeInfo]) -> None:
        """Rebuild the regime_id -> strategy mapping from fresh HMM regime metadata.

        Ranks regimes by expected_volatility (ascending) to compute each
        regime's volatility rank - a sort that is completely independent of
        the HMM's own label sort (which is by expected_return). Call this
        after every HMM retrain.
        """
        self.regime_infos = dict(regime_infos)
        infos_by_vol = sorted(self.regime_infos.values(), key=lambda info: info.expected_volatility)
        n_regimes = len(infos_by_vol)

        strategy_by_state_id: dict[int, BaseStrategy] = {}
        vol_rank_by_state_id: dict[int, float] = {}
        for rank, info in enumerate(infos_by_vol):
            position = rank / (n_regimes - 1) if n_regimes > 1 else 0.0
            strategy_cls = _strategy_class_for_vol_position(position)
            strategy_by_state_id[info.regime_id] = self._build_strategy(strategy_cls)
            vol_rank_by_state_id[info.regime_id] = position

        self._strategy_by_state_id = strategy_by_state_id
        self._vol_rank_by_state_id = vol_rank_by_state_id

        logger.info(
            "Strategy orchestrator rebuilt: %s",
            {
                info.regime_id: (info.regime_name.value, type(strategy_by_state_id[info.regime_id]).__name__)
                for info in infos_by_vol
            },
        )

    def _build_strategy(self, strategy_cls: type[BaseStrategy]) -> BaseStrategy:
        if strategy_cls is LowVolBullStrategy:
            return LowVolBullStrategy(
                allocation=self.config.get("low_vol_allocation", 0.95),
                leverage=self.config.get("low_vol_leverage", 1.25),
            )
        if strategy_cls is MidVolCautiousStrategy:
            return MidVolCautiousStrategy(
                allocation_trend=self.config.get("mid_vol_allocation_trend", 0.95),
                allocation_no_trend=self.config.get("mid_vol_allocation_no_trend", 0.60),
            )
        if strategy_cls is HighVolDefensiveStrategy:
            return HighVolDefensiveStrategy(
                allocation=self.config.get("high_vol_allocation", 0.60),
            )
        raise ValueError(f"Unknown strategy class: {strategy_cls}")

    def generate_signals(
        self,
        symbols: list[str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool,
    ) -> list[Signal]:
        """Generate one signal per symbol under the current regime.

        `regime_state` reflects the (single, confirmed) regime call driving
        this round of allocation; `is_flickering` is HMMEngine.is_flickering()
        for the same call. Symbols with missing/empty bars, or too little
        history for the mapped strategy, are silently skipped.
        """
        strategy = self._strategy_by_state_id.get(regime_state.state_id)
        if strategy is None:
            logger.warning(
                "No strategy mapped for regime_id=%d (%s) - call update_regime_infos() after training; skipping",
                regime_state.state_id,
                regime_state.label.value,
            )
            return []

        uncertain = is_flickering or regime_state.probability < self.min_confidence

        signals: list[Signal] = []
        for symbol in symbols:
            symbol_bars = bars.get(symbol)
            if symbol_bars is None or symbol_bars.empty:
                continue
            signal = strategy.generate_signal(symbol, symbol_bars, regime_state)
            if signal is None:
                continue
            if uncertain:
                signal = self._apply_uncertainty(signal)
            signals.append(signal)
        return signals

    def _apply_uncertainty(self, signal: Signal) -> Signal:
        """Halve position size and force leverage to 1.0x (min_confidence breach or flicker)."""
        size_mult = self.config.get("uncertainty_size_mult", 0.50)
        return replace(
            signal,
            position_size_pct=signal.position_size_pct * size_mult,
            leverage=1.0,
            reasoning=f"{signal.reasoning} [UNCERTAINTY - size halved]",
        )

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """True if target allocation differs from current by more than rebalance_threshold
        (default 10%) - prevents churn from minor probability fluctuations."""
        threshold = self.config.get("rebalance_threshold", 0.10)
        return abs(target_allocation - current_allocation) > threshold
