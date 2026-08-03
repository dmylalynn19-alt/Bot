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
