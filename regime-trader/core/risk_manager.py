"""Position sizing, leverage, and drawdown limits.

Enforces per-trade, per-position, and portfolio-level risk limits, and halts
or reduces trading in response to daily/weekly drawdown breaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskAction(Enum):
    """Action the risk manager instructs the system to take."""

    NORMAL = "normal"
    REDUCE = "reduce"
    HALT = "halt"


@dataclass
class PositionSizeResult:
    """Result of a position sizing calculation."""

    symbol: str
    quantity: float
    notional: float
    capped: bool


class RiskManager:
    """Applies portfolio and per-trade risk limits.

    Args:
        max_risk_per_trade: Max fraction of equity risked on a single trade.
        max_exposure: Max total portfolio exposure as a fraction of equity.
        max_leverage: Hard cap on account leverage.
        max_single_position: Max fraction of equity in any single position.
        max_concurrent: Max number of concurrently open positions.
        max_daily_trades: Max number of trades allowed per day.
        daily_dd_reduce: Daily drawdown level that triggers exposure reduction.
        daily_dd_halt: Daily drawdown level that triggers a full trading halt.
        weekly_dd_reduce: Weekly drawdown level that triggers exposure reduction.
        weekly_dd_halt: Weekly drawdown level that triggers a full trading halt.
        max_dd_from_peak: Max drawdown from equity peak before halting trading.
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
    ) -> None:
        raise NotImplementedError

    def size_position(self, symbol: str, target_allocation: float, equity: float, price: float) -> PositionSizeResult:
        """Compute a risk-capped position size for a symbol."""
        raise NotImplementedError

    def check_drawdown(self, daily_pnl_pct: float, weekly_pnl_pct: float, drawdown_from_peak: float) -> RiskAction:
        """Determine the risk action given current drawdown metrics."""
        raise NotImplementedError

    def can_open_position(self, open_position_count: int, trades_today: int) -> bool:
        """Check whether a new position can be opened under concurrency/rate limits."""
        raise NotImplementedError

    def cap_exposure(self, requested_exposure: float, current_exposure: float) -> float:
        """Cap requested total exposure to the configured maximum."""
        raise NotImplementedError
