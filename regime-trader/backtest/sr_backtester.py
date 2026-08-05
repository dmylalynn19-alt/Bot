"""Trade-by-trade backtester for core.sr_strategy.SupportResistanceStrategy.

This is the "back tested confirmation of price levels shown on that
timeframe" the strategy requires before it's eligible to enter a trade -
run this against a symbol's history, on the exact timeframe you intend to
trade it on, before trusting the strategy's live signals for that symbol.

Unlike backtest/backtester.py (an allocation-based backtester with no
individual entries/exits, built around the HMM regime strategy), this is a
genuine trade-by-trade simulation: at most one open position at a time,
sized by risk to the setup's own stop distance, held until
SupportResistanceStrategy.should_exit says to close it.

REALISTIC SIMULATION
---------------------
- No look-ahead: every decision at bar N only ever sees `bars.loc[:N]] -
  SupportResistanceStrategy.generate/should_exit are themselves look-ahead
  free (see core/support_resistance.py + tests/test_look_ahead.py), and this
  backtester never hands them a future bar.
- Fill delay: a setup found (or exit condition hit) at bar N's close is
  entered/exited at bar N+1's open, never at bar N's own close - matching
  backtest/backtester.py's convention, since you cannot actually trade at a
  price you only just saw print.
- Slippage: the fill price is nudged against the trade (configurable,
  default 0.05%) - buys (entries long / exits short) fill worse (higher),
  sells (entries short / exits long) fill worse (lower).
- Position sizing: risk-based, not allocation-based - shares are sized so
  that a stop-out loses `risk_per_trade` of equity, capped at
  `max_position_pct` of equity in notional so a single trade's position
  size stays bounded even when the stop is unusually tight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.support_resistance import Level
from core.sr_strategy import SupportResistanceStrategy, TradeDirection, TradeSetup

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One completed (or still-open) round-trip trade."""

    symbol: str
    direction: TradeDirection
    entry_time: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    level: Level
    confirmations: dict[str, bool]
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float | None = None
    pnl_pct: float | None = None


@dataclass
class SRBacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)

    def trades_frame(self) -> pd.DataFrame:
        """`trades` as a flat DataFrame, one row per completed round trip
        (open trades at the end of the run are excluded - they have no
        exit/pnl yet)."""
        rows = [
            {
                "symbol": t.symbol,
                "direction": t.direction.value,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "shares": t.shares,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "level_price": t.level.price,
                "level_kind": t.level.kind,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
            }
            for t in self.trades
            if t.exit_time is not None
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """Headline round-trip trade stats (win rate, profit factor, ...)."""
        closed = [t for t in self.trades if t.pnl is not None]
        if not closed:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
            }
        pnls = [t.pnl for t in closed if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        starting_equity = float(self.equity_curve.iloc[0]) if not self.equity_curve.empty else 0.0
        return {
            "num_trades": len(closed),
            "win_rate": len(wins) / len(closed),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
            "avg_win": (gross_profit / len(wins)) if wins else 0.0,
            "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
            "total_pnl": sum(pnls),
            "total_return_pct": (sum(pnls) / starting_equity) if starting_equity else 0.0,
        }


class SRBacktester:
    """Walks `bars` chronologically, trading SupportResistanceStrategy's
    confirmed setups one at a time.

    Args:
        strategy: A configured SupportResistanceStrategy. Its own thresholds
            (pivot window, required confirmations, timeframe-derived VWAP
            behavior, ...) fully determine what counts as a tradeable setup -
            this backtester only handles sizing, fills, and bookkeeping.
        risk_per_trade: Fraction of current equity risked (at the stop) on
            each trade.
        max_position_pct: Position notional is capped at this fraction of
            equity, regardless of how tight the stop is.
        initial_cash: Starting account equity.
        slippage: Fractional price impact applied against every fill.
        min_position_shares: A sized trade below this many shares is skipped
            (avoids noise trades on tiny accounts/tight stops).
    """

    def __init__(
        self,
        strategy: SupportResistanceStrategy,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.20,
        initial_cash: float = 100_000.0,
        slippage: float = 0.0005,
        min_position_shares: int = 1,
    ) -> None:
        self.strategy = strategy
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.initial_cash = initial_cash
        self.slippage = slippage
        self.min_position_shares = min_position_shares

    def run(
        self,
        symbol: str,
        bars: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        htf_bars: pd.DataFrame | None = None,
    ) -> SRBacktestResult:
        """Run the backtest over `bars` (single-symbol OHLCV, sorted by time).

        Args:
            symbol: Label for the instrument.
            bars: OHLCV history. Pass everything you have - history before
                `start` is used as look-ahead-free warm-up context (the
                strategy needs `min_bars` of it before it can evaluate a
                setup at all), never itself traded.
            start: First bar eligible to be evaluated for entry. Defaults to
                the first bar with enough warm-up history.
            end: Last bar to evaluate. Defaults to the end of `bars`.
            htf_bars: Optional higher-timeframe OHLCV history of the same
                symbol, for the strategy's `require_htf_bias` gate. At each
                entry-timeframe bar `ts`, the strategy only ever sees
                `htf_bars.loc[:ts]` - the higher-timeframe bar that was
                actually the most recent *closed* one as of `ts`, never a
                still-forming or future one. This assumes `htf_bars`' index
                is timestamped the same way `bars`' is (bar close/start
                time, consistently) - the same assumption the rest of this
                backtester already makes about `bars` itself.
        """
        bars = bars.sort_index()
        if bars.empty:
            return SRBacktestResult(equity_curve=pd.Series(dtype=float))

        start_ts = pd.Timestamp(start) if start is not None else bars.index[min(self.strategy.min_bars, len(bars) - 1)]
        end_ts = pd.Timestamp(end) if end is not None else bars.index[-1]
        eval_index = bars.index[(bars.index >= start_ts) & (bars.index <= end_ts)]

        cash = self.initial_cash
        shares = 0  # positive = long, negative = short
        open_trade: TradeRecord | None = None
        pending_action: dict[str, Any] | None = None  # {"kind": "entry"/"exit", ...}
        equity_records: list[dict] = []
        trades: list[TradeRecord] = []

        for ts in eval_index:
            bar_open = float(bars.loc[ts, "open"])
            bar_close = float(bars.loc[ts, "close"])

            # Execute any fill decided on the previous bar, at THIS bar's open.
            if pending_action is not None:
                if pending_action["kind"] == "entry":
                    setup: TradeSetup = pending_action["setup"]
                    target_shares = self._size_position(cash + shares * bar_open, setup, bar_open)
                    if target_shares >= self.min_position_shares:
                        signed_shares = target_shares if setup.direction == TradeDirection.LONG else -target_shares
                        fill_price = self._fill_price(bar_open, buying=signed_shares > 0)
                        cash -= signed_shares * fill_price
                        shares = signed_shares
                        open_trade = TradeRecord(
                            symbol=symbol,
                            direction=setup.direction,
                            entry_time=ts,
                            entry_price=fill_price,
                            shares=abs(signed_shares),
                            stop_price=setup.stop_price,
                            target_price=setup.target_price,
                            level=setup.level,
                            confirmations=dict(setup.confirmations),
                        )
                        logger.info(
                            "%s ENTRY %s %d shares @ %.4f (level=%.4f, stop=%.4f, target=%.4f)",
                            ts, setup.direction.value, abs(signed_shares), fill_price,
                            setup.level.price, setup.stop_price, setup.target_price,
                        )
                elif pending_action["kind"] == "exit" and open_trade is not None:
                    fill_price = self._fill_price(bar_open, buying=shares < 0)
                    cash -= (-shares) * fill_price  # closing: delta = 0 - shares = -shares
                    direction_sign = 1 if open_trade.direction == TradeDirection.LONG else -1
                    pnl = direction_sign * open_trade.shares * (fill_price - open_trade.entry_price)
                    open_trade.exit_time = ts
                    open_trade.exit_price = fill_price
                    open_trade.exit_reason = pending_action["reason"]
                    open_trade.pnl = pnl
                    open_trade.pnl_pct = pnl / (open_trade.entry_price * open_trade.shares) if open_trade.shares else 0.0
                    trades.append(open_trade)
                    logger.info("%s EXIT %s @ %.4f (%s) pnl=%.2f", ts, open_trade.direction.value, fill_price, open_trade.exit_reason, pnl)
                    shares = 0
                    open_trade = None
                pending_action = None

            bars_so_far = bars.loc[:ts]

            if open_trade is None:
                htf_bars_so_far = htf_bars.loc[:ts] if htf_bars is not None else None
                setup = self.strategy.generate(symbol, bars_so_far, htf_bars=htf_bars_so_far)
                if setup is not None:
                    pending_action = {"kind": "entry", "setup": setup}
            else:
                should_exit, reason = self.strategy.should_exit(
                    TradeSetup(
                        symbol=symbol,
                        direction=open_trade.direction,
                        level=open_trade.level,
                        entry_price=open_trade.entry_price,
                        stop_price=open_trade.stop_price,
                        target_price=open_trade.target_price,
                        confirmations=open_trade.confirmations,
                        confidence=1.0,
                        timestamp=open_trade.entry_time,
                        reasoning="",
                    ),
                    bars_so_far,
                )
                if should_exit:
                    pending_action = {"kind": "exit", "reason": reason}

            equity = cash + shares * bar_close
            equity_records.append({"timestamp": ts, "equity": equity})

        equity_curve = pd.Series(
            [r["equity"] for r in equity_records],
            index=pd.Index([r["timestamp"] for r in equity_records], name="timestamp"),
            name="equity",
        )
        return SRBacktestResult(equity_curve=equity_curve, trades=trades)

    def _size_position(self, equity: float, setup: TradeSetup, price: float) -> int:
        per_share_risk = abs(setup.entry_price - setup.stop_price)
        if per_share_risk <= 0 or price <= 0:
            return 0
        risk_dollars = equity * self.risk_per_trade
        risk_sized_shares = int(risk_dollars / per_share_risk)
        max_notional_shares = int(equity * self.max_position_pct / price)
        return max(0, min(risk_sized_shares, max_notional_shares))

    def _fill_price(self, price: float, buying: bool) -> float:
        return price * (1 + self.slippage) if buying else price * (1 - self.slippage)
