"""Alpaca API wrapper.

Thin client around the Alpaca trading and market data APIs, handling
authentication and connection setup for both paper and live trading.
"""

from __future__ import annotations

import pandas as pd


class AlpacaClient:
    """Wraps the Alpaca trade and data APIs.

    Args:
        api_key: Alpaca API key.
        secret_key: Alpaca API secret key.
        paper: Whether to connect to the paper trading endpoint.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool) -> None:
        raise NotImplementedError

    def get_account(self) -> dict:
        """Fetch account details (equity, buying power, status)."""
        raise NotImplementedError

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical bars for a symbol."""
        raise NotImplementedError

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest quote for a symbol."""
        raise NotImplementedError

    def stream_trades(self, symbols: list[str]) -> None:
        """Subscribe to a real-time trade/quote stream for the given symbols."""
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        raise NotImplementedError
