"""Track open positions and P&L."""

from __future__ import annotations

from dataclasses import dataclass

from broker.webull_client import WebullClient


@dataclass
class Position:
    """A single open position and its P&L state."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float


class PositionTracker:
    """Tracks open positions and portfolio-level P&L.

    Args:
        client: Configured Webull client used to fetch position/account
            state. NOTE: Webull's OpenAPI account-balance/position endpoints
            (Account.get_account_position(_details), AccountV2.get_account_*)
            are documented as not yet available for US brokerage accounts
            (see broker.webull_client's module docstring) - verify against
            your own approved App Key before relying on this.
    """

    def __init__(self, client: WebullClient) -> None:
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        """Fetch all currently open positions."""
        raise NotImplementedError

    def get_position(self, symbol: str) -> Position | None:
        """Fetch the open position for a single symbol, if any."""
        raise NotImplementedError

    def get_total_exposure(self) -> float:
        """Compute total portfolio exposure as a fraction of equity."""
        raise NotImplementedError

    def get_daily_pnl(self) -> float:
        """Compute realized + unrealized P&L for the current trading day."""
        raise NotImplementedError

    def get_drawdown_from_peak(self) -> float:
        """Compute the current drawdown from the equity peak."""
        raise NotImplementedError
