"""Position sizing, leverage, and drawdown limits - the system's last line of defense.

DESIGN PHILOSOPHY
------------------
The risk manager operates INDEPENDENTLY of the HMM: its circuit breakers fire
on actual portfolio P&L (PortfolioState.daily_pnl_pct / weekly_pnl_pct /
drawdown_from_peak), never on a regime label or confidence score. Even if the
HMM/strategy layer is completely broken, wrong, or absent, the circuit
breakers still catch real drawdowns - defense in depth. RiskManager has
absolute veto power over every signal: `validate_signal` is the single gate
every signal must pass before it reaches order execution, and it can reject,
shrink, or de-lever a signal but never make it bigger or riskier than what
the strategy asked for.

(RiskManager does import core.regime_strategies.Signal/Direction for typing -
that is a structural dependency on the *shape* of a signal, not a behavioral
one on the HMM being correct; nothing here reads a regime label to decide
whether a drawdown happened.)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import pandas as pd

from broker.position_tracker import Position
from core.options_strategy import OptionSignal
from core.regime_strategies import Direction, Signal
from core.sr_strategy import TradeSetup

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """A snapshot of live account/portfolio state, as seen by the risk manager.

    Everything here is a plain, already-computed number - RiskManager never
    reaches into the broker or the HMM itself, it only reads what it's handed.
    """

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    position_sectors: dict[str, str] = field(default_factory=dict)
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0
    flicker_rate: float = 0.0
    # For circuit-breaker audit logging only ("track if HMM was wrong") - never
    # read by any risk *decision*, only recorded alongside breaker events.
    current_regime_name: str = "UNKNOWN"
    current_regime_confidence: float = float("nan")

    def __post_init__(self) -> None:
        if self.peak_equity <= 0:
            self.peak_equity = self.equity

    @property
    def drawdown_from_peak(self) -> float:
        """Fractional drawdown from the equity peak (<= 0)."""
        if self.peak_equity <= 0:
            return 0.0
        return self.equity / self.peak_equity - 1.0

    @property
    def total_exposure(self) -> float:
        """Total notional across all positions, as a fraction of equity."""
        if self.equity <= 0:
            return 0.0
        notional = sum(abs(pos.quantity) * pos.current_price for pos in self.positions.values())
        return notional / self.equity


@dataclass
class RiskDecision:
    """The outcome of validate_signal: either an approved (possibly resized)
    signal, or a rejection with a reason."""

    approved: bool
    modified_signal: Signal | None
    rejection_reason: str | None
    modifications: list[str] = field(default_factory=list)


@dataclass
class OptionRiskDecision:
    """The outcome of validate_option_signal - the options counterpart to
    RiskDecision, carrying an OptionSignal instead of a stock Signal."""

    approved: bool
    modified_signal: OptionSignal | None
    rejection_reason: str | None
    modifications: list[str] = field(default_factory=list)


@dataclass
class TradeSetupRiskDecision:
    """The outcome of validate_trade_setup - the core.sr_strategy.TradeSetup
    (support/resistance and breakout strategies both produce this type)
    counterpart to RiskDecision/OptionRiskDecision.

    Unlike Signal (position_size_pct already on the object) or OptionSignal
    (contracts already on the object), a TradeSetup carries no size of its
    own - it's a *signal*, not a sized order - so `shares` is computed here
    and carried alongside the (unmodified) setup rather than folded into it.
    """

    approved: bool
    modified_setup: TradeSetup | None
    shares: int
    rejection_reason: str | None
    modifications: list[str] = field(default_factory=list)


class CircuitBreakerStatus(str, Enum):
    """Active drawdown-triggered state. REDUCE halves new position sizes;
    HALT blocks all new trading (and, for daily/weekly halts, closes every
    open position)."""

    NORMAL = "normal"
    DAILY_REDUCE = "daily_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


_SEVERITY = {
    CircuitBreakerStatus.NORMAL: 0,
    CircuitBreakerStatus.DAILY_REDUCE: 1,
    CircuitBreakerStatus.WEEKLY_REDUCE: 1,
    CircuitBreakerStatus.DAILY_HALT: 2,
    CircuitBreakerStatus.WEEKLY_HALT: 2,
    CircuitBreakerStatus.PEAK_HALT: 3,
}
_HALT_STATUSES = {CircuitBreakerStatus.DAILY_HALT, CircuitBreakerStatus.WEEKLY_HALT, CircuitBreakerStatus.PEAK_HALT}


@dataclass
class CircuitBreakerEvent:
    """One circuit-breaker trigger, for the audit log."""

    timestamp: pd.Timestamp
    breaker: CircuitBreakerStatus
    drawdown: float
    equity: float
    positions_closed: list[str]
    regime_at_trigger: str
    regime_confidence_at_trigger: float


class CircuitBreaker:
    """P&L-based circuit breakers, independent of regime/strategy state.

    Args:
        daily_dd_reduce / daily_dd_halt: Daily drawdown levels (fractions,
            e.g. 0.02 = -2%) that trigger a 50%-size reduction / full halt
            for the rest of the trading day.
        weekly_dd_reduce / weekly_dd_halt: Same, for the trading week.
        max_dd_from_peak: All-time peak-to-trough drawdown that halts all
            trading and writes `lock_file_path`; only deleting that file
            (manually) resumes trading.
        lock_file_path: Path of the halt lock file.
    """

    def __init__(
        self,
        daily_dd_reduce: float,
        daily_dd_halt: float,
        weekly_dd_reduce: float,
        weekly_dd_halt: float,
        max_dd_from_peak: float,
        lock_file_path: str = "trading_halted.lock",
    ) -> None:
        self.daily_dd_reduce = daily_dd_reduce
        self.daily_dd_halt = daily_dd_halt
        self.weekly_dd_reduce = weekly_dd_reduce
        self.weekly_dd_halt = weekly_dd_halt
        self.max_dd_from_peak = max_dd_from_peak
        self.lock_file_path = Path(lock_file_path)

        self.status: CircuitBreakerStatus = CircuitBreakerStatus.NORMAL
        self.active_breakers: set[CircuitBreakerStatus] = set()
        self._history: list[CircuitBreakerEvent] = []

        if self.lock_file_path.exists():
            self.active_breakers.add(CircuitBreakerStatus.PEAK_HALT)
            self.status = CircuitBreakerStatus.PEAK_HALT
            logger.warning("Halt lock file %s present on startup - starting in PEAK_HALT", self.lock_file_path)

    def check(self, portfolio: PortfolioState) -> set[CircuitBreakerStatus]:
        """Pure check: which breakers WOULD be active given `portfolio`'s current
        P&L, without mutating any state."""
        breakers: set[CircuitBreakerStatus] = set()

        if portfolio.daily_pnl_pct <= -self.daily_dd_halt:
            breakers.add(CircuitBreakerStatus.DAILY_HALT)
        elif portfolio.daily_pnl_pct <= -self.daily_dd_reduce:
            breakers.add(CircuitBreakerStatus.DAILY_REDUCE)

        if portfolio.weekly_pnl_pct <= -self.weekly_dd_halt:
            breakers.add(CircuitBreakerStatus.WEEKLY_HALT)
        elif portfolio.weekly_pnl_pct <= -self.weekly_dd_reduce:
            breakers.add(CircuitBreakerStatus.WEEKLY_REDUCE)

        if portfolio.drawdown_from_peak <= -self.max_dd_from_peak:
            breakers.add(CircuitBreakerStatus.PEAK_HALT)

        return breakers

    def update(self, portfolio: PortfolioState) -> CircuitBreakerStatus:
        """Re-evaluate breaker thresholds against `portfolio`'s actual P&L,
        update self.status, and log + record a CircuitBreakerEvent for every
        newly tripped breaker (writing the halt lock file for PEAK_HALT).
        Call this once per portfolio update, before validating any signal.
        """
        if CircuitBreakerStatus.PEAK_HALT in self.active_breakers:
            if self.lock_file_path.exists():
                return self.status
            logger.warning("Halt lock file %s removed - resuming from PEAK_HALT", self.lock_file_path)
            self.active_breakers.discard(CircuitBreakerStatus.PEAK_HALT)

        newly_active = self.check(portfolio)
        for breaker in newly_active - self.active_breakers:
            self._record_trigger(breaker, portfolio)
        self.active_breakers = newly_active
        self.status = self._worst(newly_active)
        return self.status

    def _record_trigger(self, breaker: CircuitBreakerStatus, portfolio: PortfolioState) -> None:
        positions_closed = list(portfolio.positions.keys()) if breaker in _HALT_STATUSES else []
        if breaker == CircuitBreakerStatus.PEAK_HALT:
            drawdown = portfolio.drawdown_from_peak
        elif breaker in (CircuitBreakerStatus.WEEKLY_REDUCE, CircuitBreakerStatus.WEEKLY_HALT):
            drawdown = portfolio.weekly_pnl_pct
        else:
            drawdown = portfolio.daily_pnl_pct

        event = CircuitBreakerEvent(
            timestamp=pd.Timestamp.now(tz="UTC"),
            breaker=breaker,
            drawdown=drawdown,
            equity=portfolio.equity,
            positions_closed=positions_closed,
            regime_at_trigger=portfolio.current_regime_name,
            regime_confidence_at_trigger=portfolio.current_regime_confidence,
        )
        self._history.append(event)
        logger.warning(
            "CIRCUIT BREAKER %s: drawdown=%.2f%%, equity=%.2f, positions_closed=%s, regime=%s (confidence=%.2f)",
            breaker.value,
            event.drawdown * 100,
            event.equity,
            positions_closed,
            event.regime_at_trigger,
            event.regime_confidence_at_trigger,
        )
        if breaker == CircuitBreakerStatus.PEAK_HALT:
            self._write_lock_file(event)

    def _write_lock_file(self, event: CircuitBreakerEvent) -> None:
        self.lock_file_path.write_text(
            f"Trading halted at {event.timestamp} - peak drawdown {event.drawdown:.2%} "
            f"(equity {event.equity:.2f}). Delete this file to resume trading.\n"
        )
        logger.warning("Wrote halt lock file %s - manual deletion required to resume trading", self.lock_file_path)

    def reset_daily(self) -> None:
        """Clear daily-scoped breakers. Call once at the start of a new trading day."""
        self.active_breakers -= {CircuitBreakerStatus.DAILY_REDUCE, CircuitBreakerStatus.DAILY_HALT}
        self.status = self._worst(self.active_breakers)

    def reset_weekly(self) -> None:
        """Clear weekly-scoped breakers. Call once at the start of a new trading week."""
        self.active_breakers -= {CircuitBreakerStatus.WEEKLY_REDUCE, CircuitBreakerStatus.WEEKLY_HALT}
        self.status = self._worst(self.active_breakers)

    def get_history(self) -> list[CircuitBreakerEvent]:
        """Full history of every breaker trigger, in order."""
        return list(self._history)

    def is_halted(self) -> bool:
        return self.status in _HALT_STATUSES

    def size_multiplier(self) -> float:
        """0.0 if halted, 0.5 if a REDUCE breaker is active (and nothing halted), else 1.0."""
        if self.is_halted():
            return 0.0
        return 0.5 if self.active_breakers else 1.0

    def _worst(self, breakers: set[CircuitBreakerStatus]) -> CircuitBreakerStatus:
        if not breakers:
            return CircuitBreakerStatus.NORMAL
        return max(breakers, key=lambda b: _SEVERITY[b])


class RiskManager:
    """Portfolio-level limits, position sizing, leverage control, and order
    validation. `validate_signal` is the single entry point; nothing should
    reach the broker without going through it.

    Args:
        max_risk_per_trade: Max fraction of equity risked on a single trade
            (used for stop-distance position sizing).
        max_exposure: Max total portfolio notional exposure, as a fraction of equity.
        max_leverage: Hard cap on portfolio leverage.
        max_single_position: Max fraction of equity in any single position.
        max_concurrent: Max number of concurrently open positions.
        max_daily_trades: Max number of trades allowed per day.
        daily_dd_reduce / daily_dd_halt: Daily circuit breaker levels.
        weekly_dd_reduce / weekly_dd_halt: Weekly circuit breaker levels.
        max_dd_from_peak: Peak-to-trough circuit breaker level.
        max_correlated_exposure: Max fraction of equity concentrated in one sector.
        min_confidence: Regime confidence below this forces leverage to 1.0x.
        min_position_dollars: Reject a sized position smaller than this.
        max_spread_pct: Reject an order when the bid-ask spread is this wide or more.
        duplicate_window_seconds: Block a duplicate same-symbol/same-direction
            order within this many seconds.
        correlation_reduce_threshold: 60-day correlation with an existing
            position above this halves size.
        correlation_reject_threshold: ...above this rejects the trade outright.
        overnight_gap_multiple: Overnight sizing assumes a gap-through this
            many multiples of the per-share stop distance.
        overnight_max_loss_pct: ...sized so a gap-through loses at most this
            fraction of equity.
        leverage_force_position_count: This many or more open positions
            forces leverage back to 1.0x.
        trading_halt_lock_file: Path of the peak-drawdown halt lock file.
    """

    def __init__(
        self,
        max_risk_per_trade: float,
        max_exposure: float,
        max_leverage: float,
        max_single_position: float,
        max_concurrent: int,
        max_daily_trades: int,
        daily_dd_reduce: float,
        daily_dd_halt: float,
        weekly_dd_reduce: float,
        weekly_dd_halt: float,
        max_dd_from_peak: float,
        max_correlated_exposure: float = 0.30,
        min_confidence: float = 0.55,
        min_position_dollars: float = 100.0,
        max_spread_pct: float = 0.005,
        duplicate_window_seconds: float = 60.0,
        correlation_reduce_threshold: float = 0.70,
        correlation_reject_threshold: float = 0.85,
        overnight_gap_multiple: float = 3.0,
        overnight_max_loss_pct: float = 0.02,
        leverage_force_position_count: int = 3,
        trading_halt_lock_file: str = "trading_halted.lock",
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_exposure = max_exposure
        self.max_leverage = max_leverage
        self.max_single_position = max_single_position
        self.max_concurrent = max_concurrent
        self.max_daily_trades = max_daily_trades
        self.max_correlated_exposure = max_correlated_exposure
        self.min_confidence = min_confidence
        self.min_position_dollars = min_position_dollars
        self.max_spread_pct = max_spread_pct
        self.duplicate_window_seconds = duplicate_window_seconds
        self.correlation_reduce_threshold = correlation_reduce_threshold
        self.correlation_reject_threshold = correlation_reject_threshold
        self.overnight_gap_multiple = overnight_gap_multiple
        self.overnight_max_loss_pct = overnight_max_loss_pct
        self.leverage_force_position_count = leverage_force_position_count

        self.circuit_breaker = CircuitBreaker(
            daily_dd_reduce=daily_dd_reduce,
            daily_dd_halt=daily_dd_halt,
            weekly_dd_reduce=weekly_dd_reduce,
            weekly_dd_halt=weekly_dd_halt,
            max_dd_from_peak=max_dd_from_peak,
            lock_file_path=trading_halt_lock_file,
        )
        self._recent_orders: list[tuple[str, str, pd.Timestamp]] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        *,
        tradeable: bool = True,
        spread_pct: float | None = None,
        correlations: dict[str, float] | None = None,
        sector: str | None = None,
        is_overnight: bool = False,
        now: pd.Timestamp | None = None,
    ) -> RiskDecision:
        """Absolute veto over `signal`. Runs, in order:

        1. Circuit breakers (portfolio.daily_pnl_pct/weekly_pnl_pct/
           drawdown_from_peak only - never the regime) - reject outright if halted.
        2. Hard order-validation rejects: missing/invalid stop loss, not
           tradeable, spread too wide, duplicate within the last minute,
           correlation > reject threshold, daily trade cap, concurrent
           position cap.
        3. Leverage resolution (1.0x unless the strategy asked for more and
           nothing forces it back down).
        4. Risk-based position sizing (1% risk-per-trade, capped by the
           strategy's own requested allocation, then the single-position
           ceiling, then overnight gap sizing).
        5. Correlation size reduction (0.70-0.85 band), sector concentration
           cap, portfolio exposure/leverage caps, and circuit-breaker REDUCE.
        6. Minimum position size / buying power, on the final numbers.

        Returns a RiskDecision with a (possibly resized/de-levered)
        modified_signal, or approved=False with rejection_reason set.
        """
        now = now if now is not None else pd.Timestamp.now(tz="UTC")
        modifications: list[str] = []

        breaker_status = self.circuit_breaker.update(portfolio)
        if self.circuit_breaker.is_halted():
            return self._reject(f"circuit breaker active: {breaker_status.value} - all trading halted")

        reject_reason = self._check_hard_rejects(signal, portfolio, tradeable, spread_pct, correlations, now)
        if reject_reason is not None:
            return self._reject(reject_reason)

        leverage, leverage_mods = self._resolve_leverage(signal, portfolio)
        modifications += leverage_mods

        target_allocation, sizing_mods = self._size_position(signal, portfolio, leverage, is_overnight)
        modifications += sizing_mods

        target_allocation, corr_mods = self._apply_correlation_reduction(target_allocation, correlations)
        modifications += corr_mods

        target_allocation, sector_mods = self._apply_sector_cap(target_allocation, portfolio, signal.symbol, sector)
        modifications += sector_mods

        target_allocation, exposure_mods = self._apply_portfolio_caps(target_allocation, portfolio)
        modifications += exposure_mods

        size_mult = self.circuit_breaker.size_multiplier()
        if size_mult < 1.0:
            target_allocation *= size_mult
            leverage = min(leverage, 1.0)
            modifications.append(f"circuit breaker {breaker_status.value} active: size x{size_mult}")

        notional = target_allocation * portfolio.equity
        if notional < self.min_position_dollars:
            return self._reject(f"sized position (${notional:,.2f}) below minimum (${self.min_position_dollars:,.2f})")
        if notional > portfolio.buying_power:
            return self._reject(f"required notional (${notional:,.2f}) exceeds buying power (${portfolio.buying_power:,.2f})")

        self._record_order(signal.symbol, signal.direction.value, now)

        modified = replace(signal, position_size_pct=target_allocation / leverage, leverage=leverage)
        return RiskDecision(approved=True, modified_signal=modified, rejection_reason=None, modifications=modifications)

    # ------------------------------------------------------------------
    # Hard rejects
    # ------------------------------------------------------------------

    def _check_hard_rejects(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        tradeable: bool,
        spread_pct: float | None,
        correlations: dict[str, float] | None,
        now: pd.Timestamp,
    ) -> str | None:
        if signal.stop_loss is None or not math.isfinite(signal.stop_loss):
            return "no stop loss set - every position must have one"
        if signal.direction == Direction.LONG and signal.stop_loss >= signal.entry_price:
            return f"invalid stop loss {signal.stop_loss} is not below entry price {signal.entry_price} for a LONG"

        if not tradeable:
            return f"{signal.symbol} is not currently tradeable"

        if spread_pct is not None and spread_pct >= self.max_spread_pct:
            return f"bid-ask spread {spread_pct:.2%} >= max {self.max_spread_pct:.2%}"

        if self._is_duplicate(signal.symbol, signal.direction.value, now):
            return f"duplicate order for {signal.symbol} {signal.direction.value} within {self.duplicate_window_seconds:.0f}s"

        if portfolio.trades_today >= self.max_daily_trades:
            return f"max daily trades ({self.max_daily_trades}) reached"

        if signal.symbol not in portfolio.positions and len(portfolio.positions) >= self.max_concurrent:
            return f"max concurrent positions ({self.max_concurrent}) reached"

        if correlations:
            worst_symbol, worst_corr = max(correlations.items(), key=lambda kv: kv[1])
            if worst_corr > self.correlation_reject_threshold:
                return (
                    f"60-day correlation with {worst_symbol} ({worst_corr:.2f}) "
                    f"exceeds reject threshold {self.correlation_reject_threshold:.2f}"
                )

        return None

    def _is_duplicate(self, symbol: str, direction: str, now: pd.Timestamp) -> bool:
        cutoff = now - pd.Timedelta(seconds=self.duplicate_window_seconds)
        return any(sym == symbol and dirn == direction and ts >= cutoff for sym, dirn, ts in self._recent_orders)

    def _record_order(self, symbol: str, direction: str, now: pd.Timestamp) -> None:
        cutoff = now - pd.Timedelta(seconds=self.duplicate_window_seconds)
        self._recent_orders = [order for order in self._recent_orders if order[2] >= cutoff]
        self._recent_orders.append((symbol, direction, now))

    def _reject(self, reason: str) -> RiskDecision:
        logger.warning("Signal rejected: %s", reason)
        return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason, modifications=[])

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def _resolve_leverage(self, signal: Signal, portfolio: PortfolioState) -> tuple[float, list[str]]:
        mods: list[str] = []
        requested = min(signal.leverage, self.max_leverage)
        if requested < signal.leverage:
            mods.append(f"leverage capped at portfolio max {self.max_leverage:.2f}x (strategy requested {signal.leverage:.2f}x)")

        if requested <= 1.0:
            return requested, mods

        force_reasons = []
        if signal.confidence < self.min_confidence:
            force_reasons.append(f"regime confidence {signal.confidence:.2f} < {self.min_confidence:.2f}")
        if self.circuit_breaker.status != CircuitBreakerStatus.NORMAL:
            force_reasons.append(f"circuit breaker {self.circuit_breaker.status.value} active")
        if len(portfolio.positions) >= self.leverage_force_position_count:
            force_reasons.append(f"{len(portfolio.positions)} positions already open")
        if portfolio.flicker_rate > 0:
            force_reasons.append("high regime flicker rate")

        if force_reasons:
            mods.append(f"leverage forced to 1.0x ({'; '.join(force_reasons)})")
            return 1.0, mods
        return requested, mods

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _size_position(
        self, signal: Signal, portfolio: PortfolioState, leverage: float, is_overnight: bool
    ) -> tuple[float, list[str]]:
        mods: list[str] = []
        risk_per_share = abs(signal.entry_price - signal.stop_loss)

        # Position size = (equity * max_risk_per_trade) / risk_per_share, expressed
        # as a fraction of equity: multiply by entry_price and divide by equity.
        risk_based_pct = self.max_risk_per_trade * signal.entry_price / risk_per_share
        regime_max_pct = signal.position_size_pct * leverage
        portfolio_max_pct = self.max_single_position

        capped_by = {
            "1% risk-per-trade rule": risk_based_pct,
            "regime-requested allocation": regime_max_pct,
            f"{self.max_single_position:.0%} single-position cap": portfolio_max_pct,
        }
        final_pct = min(capped_by.values())
        if final_pct < regime_max_pct:
            binding_name = min(capped_by, key=capped_by.get)
            mods.append(f"position sized to {final_pct:.1%} of equity (bound by {binding_name})")

        if is_overnight:
            overnight_max_pct = self.overnight_max_loss_pct * signal.entry_price / (self.overnight_gap_multiple * risk_per_share)
            if overnight_max_pct < final_pct:
                mods.append(
                    f"overnight gap risk (assuming {self.overnight_gap_multiple:.0f}x stop gap-through) "
                    f"caps size to {overnight_max_pct:.1%} (was {final_pct:.1%})"
                )
                final_pct = overnight_max_pct

        return final_pct, mods

    def _apply_correlation_reduction(
        self, target_allocation: float, correlations: dict[str, float] | None
    ) -> tuple[float, list[str]]:
        if not correlations:
            return target_allocation, []
        worst_symbol, worst_corr = max(correlations.items(), key=lambda kv: kv[1])
        if worst_corr > self.correlation_reduce_threshold:
            return target_allocation * 0.5, [
                f"60-day correlation with {worst_symbol} ({worst_corr:.2f}) > "
                f"{self.correlation_reduce_threshold:.2f}: size halved"
            ]
        return target_allocation, []

    def _apply_sector_cap(
        self, target_allocation: float, portfolio: PortfolioState, symbol: str, sector: str | None
    ) -> tuple[float, list[str]]:
        if sector is None or portfolio.equity <= 0:
            return target_allocation, []

        existing_sector_notional = sum(
            abs(pos.quantity) * pos.current_price
            for sym, pos in portfolio.positions.items()
            if sym != symbol and portfolio.position_sectors.get(sym) == sector
        )
        existing_sector_pct = existing_sector_notional / portfolio.equity
        headroom = self.max_correlated_exposure - existing_sector_pct

        if headroom <= 0:
            return 0.0, [f"sector '{sector}' already at or above {self.max_correlated_exposure:.0%} exposure cap"]
        if target_allocation > headroom:
            return headroom, [f"position reduced to fit {self.max_correlated_exposure:.0%} sector '{sector}' exposure cap"]
        return target_allocation, []

    def _apply_portfolio_caps(self, target_allocation: float, portfolio: PortfolioState) -> tuple[float, list[str]]:
        mods: list[str] = []
        existing_exposure = portfolio.total_exposure

        exposure_headroom = self.max_exposure - existing_exposure
        if exposure_headroom <= 0:
            return 0.0, [f"portfolio already at or above {self.max_exposure:.0%} max exposure"]
        if target_allocation > exposure_headroom:
            mods.append(f"position reduced to fit {self.max_exposure:.0%} max total exposure cap")
            target_allocation = exposure_headroom

        leverage_headroom = max(self.max_leverage - existing_exposure, 0.0)
        if target_allocation > leverage_headroom:
            mods.append(f"position reduced to fit {self.max_leverage:.2f}x max portfolio leverage")
            target_allocation = leverage_headroom

        return target_allocation, mods

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def validate_option_signal(
        self,
        signal: OptionSignal,
        portfolio: PortfolioState,
        now: pd.Timestamp | None = None,
    ) -> OptionRiskDecision:
        """Absolute veto over an options trade - the options counterpart to
        validate_signal. An option's max loss is simply the premium paid (no
        stop-distance-based sizing like stocks), so this caps *premium at
        risk* against portfolio limits instead of share notional.

        Runs: circuit breakers (same P&L-based checks as validate_signal,
        independent of any options-specific state), duplicate-order block,
        max daily trades, max concurrent option positions, then caps total
        premium at risk (this trade + existing option positions) at
        max_exposure * equity, and halves contracts if the circuit breaker
        is in REDUCE state.
        """
        now = now if now is not None else pd.Timestamp.now(tz="UTC")

        breaker_status = self.circuit_breaker.update(portfolio)
        if self.circuit_breaker.is_halted():
            return self._reject_option(f"circuit breaker active: {breaker_status.value} - all trading halted")

        if self._is_duplicate(signal.symbol, signal.right.value, now):
            return self._reject_option(
                f"duplicate order for {signal.symbol} {signal.right.value} within {self.duplicate_window_seconds:.0f}s"
            )

        if portfolio.trades_today >= self.max_daily_trades:
            return self._reject_option(f"max daily trades ({self.max_daily_trades}) reached")

        option_positions = {
            sym: pos for sym, pos in portfolio.positions.items() if sym != signal.occ_symbol and pos.asset_class == "us_option"
        }
        if len(option_positions) >= self.max_concurrent:
            return self._reject_option(f"max concurrent option positions ({self.max_concurrent}) reached")

        contracts = signal.contracts
        modifications: list[str] = []

        existing_premium_at_risk = sum(abs(pos.quantity) * pos.avg_entry_price * 100 for pos in option_positions.values())
        premium_cap = self.max_exposure * portfolio.equity
        headroom = premium_cap - existing_premium_at_risk
        if headroom <= 0:
            return self._reject_option(f"portfolio already at or above {self.max_exposure:.0%} max premium-at-risk cap")

        new_premium = contracts * signal.limit_price * 100
        if new_premium > headroom:
            contracts = int(headroom / (signal.limit_price * 100))
            if contracts < 1:
                return self._reject_option(f"sized position exceeds {self.max_exposure:.0%} max premium-at-risk cap")
            modifications.append(f"contracts reduced to {contracts} to fit {self.max_exposure:.0%} max premium-at-risk cap")

        size_mult = self.circuit_breaker.size_multiplier()
        if size_mult < 1.0:
            contracts = int(contracts * size_mult)
            if contracts < 1:
                return self._reject_option(f"circuit breaker {breaker_status.value} active - size reduced to zero contracts")
            modifications.append(f"circuit breaker {breaker_status.value} active: contracts x{size_mult}")

        notional = contracts * signal.limit_price * 100
        if notional < self.min_position_dollars:
            return self._reject_option(f"sized position (${notional:,.2f}) below minimum (${self.min_position_dollars:,.2f})")
        if notional > portfolio.buying_power:
            return self._reject_option(f"required premium (${notional:,.2f}) exceeds buying power (${portfolio.buying_power:,.2f})")

        self._record_order(signal.symbol, signal.right.value, now)

        modified = replace(signal, contracts=contracts)
        return OptionRiskDecision(approved=True, modified_signal=modified, rejection_reason=None, modifications=modifications)

    def _reject_option(self, reason: str) -> OptionRiskDecision:
        logger.warning("Option signal rejected: %s", reason)
        return OptionRiskDecision(approved=False, modified_signal=None, rejection_reason=reason, modifications=[])

    # ------------------------------------------------------------------
    # core.sr_strategy.TradeSetup / core.breakout_strategy validation
    # ------------------------------------------------------------------

    def validate_trade_setup(
        self,
        setup: TradeSetup,
        portfolio: PortfolioState,
        now: pd.Timestamp | None = None,
    ) -> TradeSetupRiskDecision:
        """Absolute veto over a core.sr_strategy.TradeSetup - the shared
        counterpart to validate_signal/validate_option_signal for the
        support/resistance and breakout strategies (both produce this same
        type). Since a TradeSetup carries no size of its own, this also
        computes it: risk-based (`max_risk_per_trade` of equity / the
        setup's own stop distance, capped by `max_single_position` of
        notional) - the same sizing convention backtest.sr_backtester.
        SRBacktester uses, so live and backtested position sizes are
        computed identically for the same setup.

        Runs: circuit breakers, a sane stop check, duplicate-order block,
        max daily trades, max concurrent position cap, then risk-based
        share sizing (capped by max_single_position notional and circuit-
        breaker REDUCE), then minimum position size / buying power on the
        final number.
        """
        now = now if now is not None else pd.Timestamp.now(tz="UTC")

        breaker_status = self.circuit_breaker.update(portfolio)
        if self.circuit_breaker.is_halted():
            return self._reject_trade_setup(f"circuit breaker active: {breaker_status.value} - all trading halted")

        per_share_risk = abs(setup.entry_price - setup.stop_price)
        if per_share_risk <= 0 or not math.isfinite(per_share_risk):
            return self._reject_trade_setup("invalid stop price - cannot size position")

        if self._is_duplicate(setup.symbol, setup.direction.value, now):
            return self._reject_trade_setup(
                f"duplicate order for {setup.symbol} {setup.direction.value} within {self.duplicate_window_seconds:.0f}s"
            )

        if portfolio.trades_today >= self.max_daily_trades:
            return self._reject_trade_setup(f"max daily trades ({self.max_daily_trades}) reached")

        if setup.symbol not in portfolio.positions and len(portfolio.positions) >= self.max_concurrent:
            return self._reject_trade_setup(f"max concurrent positions ({self.max_concurrent}) reached")

        modifications: list[str] = []
        risk_dollars = portfolio.equity * self.max_risk_per_trade
        shares = int(risk_dollars / per_share_risk)

        max_notional_shares = int(portfolio.equity * self.max_single_position / setup.entry_price)
        if max_notional_shares < shares:
            shares = max_notional_shares
            modifications.append(f"shares capped at {self.max_single_position:.0%} of equity notional")

        size_mult = self.circuit_breaker.size_multiplier()
        if size_mult < 1.0:
            shares = int(shares * size_mult)
            modifications.append(f"circuit breaker {breaker_status.value} active: shares x{size_mult}")

        if shares < 1:
            return self._reject_trade_setup("sized position rounds to zero shares")

        notional = shares * setup.entry_price
        if notional < self.min_position_dollars:
            return self._reject_trade_setup(f"sized position (${notional:,.2f}) below minimum (${self.min_position_dollars:,.2f})")
        if notional > portfolio.buying_power:
            return self._reject_trade_setup(f"required notional (${notional:,.2f}) exceeds buying power (${portfolio.buying_power:,.2f})")

        self._record_order(setup.symbol, setup.direction.value, now)
        return TradeSetupRiskDecision(approved=True, modified_setup=setup, shares=shares, rejection_reason=None, modifications=modifications)

    def _reject_trade_setup(self, reason: str) -> TradeSetupRiskDecision:
        logger.warning("Trade setup rejected: %s", reason)
        return TradeSetupRiskDecision(approved=False, modified_setup=None, shares=0, rejection_reason=reason, modifications=[])
