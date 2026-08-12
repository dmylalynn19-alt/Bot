"""Order placement, modification, and cancellation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.options_strategy import OptionSignal
from core.regime_strategies import Signal

if TYPE_CHECKING:
    from broker.base import BrokerClient

logger = logging.getLogger(__name__)

# Alpaca's order lifecycle has more states than we act on distinctly; anything
# not in this map (e.g. "accepted", "pending_new") is treated as PENDING.
_STATUS_MAP = {
    "new": "pending",
    "accepted": "pending",
    "pending_new": "pending",
    "filled": "filled",
    "partially_filled": "partially_filled",
    "canceled": "canceled",
    "expired": "canceled",
    "rejected": "rejected",
}


class OrderStatus(Enum):
    """Lifecycle status of a submitted order."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a submitted or tracked order."""

    order_id: str
    symbol: str
    quantity: float
    side: str
    status: OrderStatus


def _to_order(raw: dict) -> Order:
    return Order(
        order_id=raw["id"],
        symbol=raw["symbol"],
        quantity=float(raw["qty"]),
        side=raw["side"],
        status=OrderStatus(_STATUS_MAP.get(raw["status"], "pending")),
    )


class OrderExecutor:
    """Places, modifies, and cancels orders against the broker.

    Args:
        client: Any broker satisfying broker.base.BrokerClient (currently
            AlpacaClient or SchwabAdapter-wrapped SchwabClient) - this class
            never branches on which one it's holding.
    """

    def __init__(self, client: "BrokerClient") -> None:
        self.client = client

    def execute_signal(self, signal: Signal, current_quantity: float, equity: float) -> Order | None:
        """Translate a risk-approved signal into a submitted market order.

        Computes the target share count from `signal.position_size_pct *
        signal.leverage` (the same allocation math as
        backtest.backtester.WalkForwardBacktester), diffs it against
        `current_quantity` already held, and submits the delta as a single
        market order. Returns None (submits nothing) if the delta rounds to
        zero shares - i.e. no rebalance is actually needed.
        """
        target_allocation = signal.position_size_pct * signal.leverage
        target_shares = int(equity * target_allocation / signal.entry_price)
        delta = target_shares - current_quantity
        if delta == 0:
            logger.info("%s: target allocation already matches current position, no order needed", signal.symbol)
            return None

        side = "buy" if delta > 0 else "sell"
        logger.info("%s: submitting %s %d shares (target=%d, current=%d)", signal.symbol, side, abs(delta), target_shares, current_quantity)
        raw_order = self.client.submit_order(
            symbol=signal.symbol, qty=abs(delta), side=side, asset_class="equity", order_type="market", time_in_force="day"
        )
        return _to_order(raw_order)

    def execute_option_signal(self, signal: OptionSignal) -> Order | None:
        """Submit a limit order to open a directional options position.

        Always a limit order at `signal.limit_price` (the mid-price used for
        sizing), never a market order - option bid/ask spreads can be wide,
        and a market order risks a much worse fill than what was sized for.
        Returns None if `signal.contracts` is less than 1.
        """
        if signal.contracts < 1:
            return None
        logger.info(
            "%s: submitting buy-to-open %d contracts of %s @ $%.2f limit",
            signal.symbol, signal.contracts, signal.occ_symbol, signal.limit_price,
        )
        raw_order = self.client.submit_order(
            symbol=signal.occ_symbol, qty=signal.contracts, side="buy_to_open", asset_class="option",
            order_type="limit", time_in_force="day", limit_price=signal.limit_price,
        )
        return _to_order(raw_order)

    def close_option_position(self, occ_symbol: str, contracts: float, reason: str = "") -> Order | None:
        """Submit a market order to close (all or part of) an open option position."""
        if contracts <= 0:
            return None
        logger.info("%s: closing %d contracts (%s)", occ_symbol, contracts, reason or "no reason given")
        raw_order = self.client.submit_order(
            symbol=occ_symbol, qty=contracts, side="sell_to_close", asset_class="option", order_type="market", time_in_force="day"
        )
        return _to_order(raw_order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID. Returns True if the cancel request succeeded."""
        try:
            self.client.cancel_order(order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def modify_order(self, order_id: str, **kwargs) -> Order:
        """Modify an open order's parameters (qty, limit_price, stop_price, time_in_force)."""
        raw_order = self.client.replace_order(order_id, **kwargs)
        return _to_order(raw_order)

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Fetch the current status of an order."""
        raw_order = self.client.get_order(order_id)
        return _to_order(raw_order).status
