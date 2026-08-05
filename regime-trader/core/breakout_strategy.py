"""Consolidation-breakout / momentum continuation strategy.

A different playbook from core.sr_strategy's bounce/rejection setups: this
one trades the break OUT of a level (continuation), not a reversal at one.
The four ideas from the video notes it implements:

1. Price control (uptrend / downtrend / consolidation): core.consolidation.
   detect_consolidation flags when the market has coiled into a tight
   range - that's the setup this strategy waits for, since the goal is to
   catch the trend that emerges once it finally breaks.
2. True vs. false breakouts: core.consolidation.classify_breakout requires
   the broken edge to be retested and hold before calling a breakout real
   (by default - `require_true_breakout=False` trades the initial break
   itself, faster but with more false-breakout risk).
3. The three-bar entry model (lead / reaction / confirmation candles,
   core.candle_patterns.three_bar_entry) is the actual entry trigger, used
   once a (true) breakout is in hand - not a standalone signal on its own.
4. Candle speed and "last leg" acceleration: the lead candle in the
   three-bar pattern already has to be fast (see three_bar_entry's
   momentum_multiple); `require_leg_acceleration` additionally requires the
   live move to be outrunning the market's recent typical leg speed
   (core.market_structure.current_leg_acceleration) - the classic signature
   of a trend's final, fastest push toward its target, which the video
   notes call out as a prime opportunity, not something to avoid.

Produces the same TradeSetup/TradeDirection as core.sr_strategy, so it's
interchangeable with it wherever a strategy is called (risk manager, order
executor, backtester) - only the *setup detection* differs.
"""

from __future__ import annotations

import pandas as pd

from core.candle_patterns import ThreeBarDirection, three_bar_entry
from core.consolidation import BreakoutStatus, classify_breakout, find_last_consolidation
from core.market_structure import current_leg_acceleration
from core.sr_strategy import TradeDirection, TradeSetup
from core.support_resistance import Level


class BreakoutStrategy:
    """Detects and confirms consolidation-breakout continuation setups.

    Args:
        consolidation_lookback / consolidation_max_range_atr_mult /
            atr_period: passed to core.consolidation.find_last_consolidation -
            how many bars define "the range", and how tight (relative to
            ATR) that range must be to count as consolidation.
        retest_tolerance_pct / max_bars_to_classify: passed to
            core.consolidation.classify_breakout.
        consolidation_search_bars: how far back (from the current bar) to
            search for the most recent qualifying consolidation - needs to
            be wider than consolidation_lookback + max_bars_to_classify,
            since by the time a breakout is confirmed a few bars later, the
            range itself has already scrolled out of the immediate trailing
            window (see find_last_consolidation).
        require_true_breakout: only trade breakouts classified TRUE (edge
            retested and held) - the true-vs-false-breakout gate. Off trades
            the initial break itself (PENDING_RETEST allowed too), faster
            but exposed to more false-breakout risk. A breakout already
            classified FALSE is never tradeable in the breakout direction
            either way.
        three_bar_momentum_multiple / three_bar_max_reaction_retrace_pct:
            passed to core.candle_patterns.three_bar_entry - the entry
            trigger, evaluated only once a qualifying breakout is in hand.
        require_leg_acceleration / acceleration_lookback_legs /
            min_acceleration_ratio: if require_leg_acceleration, only enter
            when the live move's speed is at least min_acceleration_ratio
            times the trend's recent average leg speed (a "last leg" /
            climax-move filter, off by default since it's a strong
            additional condition on top of everything else).
        structure_pivot_window: pivot window for the market-structure leg
            analysis used by the acceleration check.
        risk_reward_ratio: target distance from entry, as a multiple of the
            stop distance (stop placed at the far side of the consolidation
            range that was broken).
        min_bars: minimum bars of history required before evaluating.
    """

    def __init__(
        self,
        consolidation_lookback: int = 20,
        consolidation_max_range_atr_mult: float = 2.5,
        atr_period: int = 14,
        retest_tolerance_pct: float = 0.003,
        max_bars_to_classify: int = 20,
        consolidation_search_bars: int = 60,
        require_true_breakout: bool = True,
        three_bar_momentum_multiple: float = 1.3,
        three_bar_max_reaction_retrace_pct: float = 0.75,
        require_leg_acceleration: bool = False,
        acceleration_lookback_legs: int = 3,
        min_acceleration_ratio: float = 1.3,
        structure_pivot_window: int = 5,
        risk_reward_ratio: float = 2.0,
        min_bars: int = 60,
    ) -> None:
        self.consolidation_lookback = consolidation_lookback
        self.consolidation_max_range_atr_mult = consolidation_max_range_atr_mult
        self.atr_period = atr_period
        self.retest_tolerance_pct = retest_tolerance_pct
        self.max_bars_to_classify = max_bars_to_classify
        self.consolidation_search_bars = consolidation_search_bars
        self.require_true_breakout = require_true_breakout
        self.three_bar_momentum_multiple = three_bar_momentum_multiple
        self.three_bar_max_reaction_retrace_pct = three_bar_max_reaction_retrace_pct
        self.require_leg_acceleration = require_leg_acceleration
        self.acceleration_lookback_legs = acceleration_lookback_legs
        self.min_acceleration_ratio = min_acceleration_ratio
        self.structure_pivot_window = structure_pivot_window
        self.risk_reward_ratio = risk_reward_ratio
        self.min_bars = min_bars

    def generate(self, symbol: str, bars: pd.DataFrame) -> TradeSetup | None:
        """Evaluate `bars` (OHLCV, most recent bar last) for a confirmed
        consolidation-breakout setup. Returns None at any gate that isn't
        satisfied - no consolidation found, no (qualifying) breakout from
        it, no three-bar entry trigger, entry direction disagreeing with
        the breakout direction, or (if required) insufficient leg
        acceleration.
        """
        if len(bars) < self.min_bars:
            return None

        consolidation = find_last_consolidation(
            bars, self.consolidation_lookback, self.consolidation_max_range_atr_mult, self.atr_period, self.consolidation_search_bars
        )
        if consolidation is None:
            return None

        breakout = classify_breakout(bars, consolidation, self.retest_tolerance_pct, self.max_bars_to_classify)
        if breakout.status == BreakoutStatus.NONE or breakout.status == BreakoutStatus.FALSE_BREAKOUT:
            return None
        if self.require_true_breakout and breakout.status != BreakoutStatus.TRUE_BREAKOUT:
            return None

        signal = three_bar_entry(bars, self.atr_period, self.three_bar_momentum_multiple, self.three_bar_max_reaction_retrace_pct)
        if signal is None:
            return None

        direction = TradeDirection.LONG if signal.direction == ThreeBarDirection.BULLISH else TradeDirection.SHORT
        if (direction == TradeDirection.LONG) != (breakout.direction == "up"):
            return None  # three-bar trigger fired against the breakout - not this setup

        acceleration_ratio = current_leg_acceleration(bars, self.structure_pivot_window, self.acceleration_lookback_legs)
        if self.require_leg_acceleration and (acceleration_ratio is None or acceleration_ratio < self.min_acceleration_ratio):
            return None

        price = float(bars["close"].iloc[-1])
        # The broken edge flips polarity - resistance that gave way becomes
        # support (and vice versa) - so the stop sits on the *other* side
        # of the range, not the broken edge itself (that's the retest zone).
        if direction == TradeDirection.LONG:
            stop_price = consolidation.low
            level = Level(
                price=consolidation.high, kind="support", touches=1,
                first_touch=consolidation.start, last_touch=consolidation.end, source="consolidation_breakout",
            )
        else:
            stop_price = consolidation.high
            level = Level(
                price=consolidation.low, kind="resistance", touches=1,
                first_touch=consolidation.start, last_touch=consolidation.end, source="consolidation_breakout",
            )

        risk = abs(price - stop_price)
        if risk <= 0:
            return None
        target_price = price + risk * self.risk_reward_ratio if direction == TradeDirection.LONG else price - risk * self.risk_reward_ratio

        confidence = 0.60
        if breakout.status == BreakoutStatus.TRUE_BREAKOUT:
            confidence += 0.20
        if acceleration_ratio is not None and acceleration_ratio >= self.min_acceleration_ratio:
            confidence += 0.20
        confidence = min(confidence, 1.0)

        return TradeSetup(
            symbol=symbol,
            direction=direction,
            level=level,
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            confirmations={
                "consolidation": True,
                "breakout": breakout.status == BreakoutStatus.TRUE_BREAKOUT,
                "three_bar_entry": True,
                "leg_acceleration": acceleration_ratio is not None and acceleration_ratio >= self.min_acceleration_ratio,
            },
            confidence=confidence,
            timestamp=bars.index[-1],
            reasoning=(
                f"{direction.value} breakout of {consolidation.width_pct:.2%}-wide consolidation "
                f"({consolidation.bar_count} bars, {breakout.status.value}), three-bar entry confirmed"
                + (f", leg acceleration={acceleration_ratio:.2f}x" if acceleration_ratio is not None else "")
            ),
            metadata={
                "consolidation_high": consolidation.high,
                "consolidation_low": consolidation.low,
                "breakout_status": breakout.status.value,
                "lead_speed": signal.lead_speed,
                "leg_acceleration_ratio": acceleration_ratio,
            },
        )

    def should_exit(self, setup: TradeSetup, bars: pd.DataFrame) -> tuple[bool, str]:
        """Check whether an open position from `setup` should be closed:
        stop hit, target hit, or neither yet. Identical stop/target logic
        to core.sr_strategy.SupportResistanceStrategy.should_exit."""
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
