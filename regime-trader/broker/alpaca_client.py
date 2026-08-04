"""Alpaca API wrapper.

Thin client around Alpaca's REST API (alpaca-trade-api), handling
authentication and connection setup for both paper and live trading.
"""

from __future__ import annotations

import pandas as pd
from alpaca_trade_api import REST, TimeFrame

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_LIVE_BASE_URL = "https://api.alpaca.markets"

_TIMEFRAME_MAP = {
    "1Min": TimeFrame.Minute,
    "1Hour": TimeFrame.Hour,
    "1Day": TimeFrame.Day,
    "1Week": TimeFrame.Week,
    "1Month": TimeFrame.Month,
}


class AlpacaClient:
    """Wraps the Alpaca trade and data APIs.

    Args:
        api_key: Alpaca API key.
        secret_key: Alpaca API secret key.
        paper: Whether to connect to the paper trading endpoint.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool) -> None:
        self.paper = paper
        base_url = _PAPER_BASE_URL if paper else _LIVE_BASE_URL
        self._api = REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version="v2")

    def get_account(self) -> dict:
        """Fetch account details (equity, buying power, status)."""
        account = self._api.get_account()
        return account._raw

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical bars for a symbol."""
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'; supported: {sorted(_TIMEFRAME_MAP)}")
        bars = self._api.get_bars(symbol, _TIMEFRAME_MAP[timeframe], start=start, end=end)
        return bars.df

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest quote for a symbol."""
        quote = self._api.get_latest_quote(symbol)
        return quote._raw

    def stream_trades(self, symbols: list[str]) -> None:
        """Subscribe to a real-time trade/quote stream for the given symbols.

        Not yet implemented: Alpaca's real-time data is a separate websocket
        connection (alpaca_trade_api.stream.Stream), needing its own
        connection-management/reconnect loop - out of scope for this pass.
        """
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        return bool(self._api.get_clock().is_open)

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Submit an order. `side` is "buy" or "sell"; `qty` must be positive."""
        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        return order._raw

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by ID."""
        self._api.cancel_order(order_id)

    def replace_order(
        self,
        order_id: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str | None = None,
    ) -> dict:
        """Modify an open order's quantity/price/time-in-force."""
        order = self._api.replace_order(
            order_id, qty=qty, limit_price=limit_price, stop_price=stop_price, time_in_force=time_in_force
        )
        return order._raw

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order's current state by ID."""
        return self._api.get_order(order_id)._raw

    def list_orders(self, status: str = "open", after: str | None = None) -> list[dict]:
        """List orders. `status` is "open", "closed", or "all". `after` (ISO
        8601) restricts to orders submitted after that time."""
        return [order._raw for order in self._api.list_orders(status=status, after=after)]

    def list_positions(self) -> list[dict]:
        """List all currently open positions."""
        return [position._raw for position in self._api.list_positions()]

    def get_position(self, symbol: str) -> dict | None:
        """Fetch the open position for a single symbol, or None if there isn't one."""
        try:
            return self._api.get_position(symbol)._raw
        except Exception as exc:
            if "position does not exist" in str(exc).lower() or getattr(exc, "status_code", None) == 404:
                return None
            raise
