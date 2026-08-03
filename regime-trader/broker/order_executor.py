"""Order placement, modification, and cancellation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from broker.webull_client import WebullClient
from core.signal_generator import Signal


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


class OrderExecutor:
    """Places, modifies, and cancels orders against the broker.

    Args:
        client: Configured Webull client used to submit orders. NOTE: Webull's
            OpenAPI order-placement endpoints are documented as not yet
            available for US brokerage accounts (see broker.webull_client's
            module docstring) - this class's methods will need to target
            whichever of OrderOperation/OrderOperationV2's place_order,
            replace_order, cancel_order(_v2) Webull enables for your account,
            using client._api.order / client._api.order_v2 and the
            OrderSide/OrderType/OrderTIF enums from webullsdktrade.common.
    """

    def __init__(self, client: WebullClient) -> None:
        raise NotImplementedError

    def execute_signal(self, signal: Signal) -> Order:
        """Translate a trading signal into a submitted order."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        raise NotImplementedError

    def modify_order(self, order_id: str, **kwargs) -> Order:
        """Modify an open order's parameters (e.g. quantity, limit price)."""
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Fetch the current status of an order."""
        raise NotImplementedError
