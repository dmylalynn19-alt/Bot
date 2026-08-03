"""Real-time and historical data fetching."""

from __future__ import annotations

import pandas as pd

from broker.webull_client import WebullClient


class MarketDataFeed:
    """Fetches and caches historical and real-time market data.

    Args:
        client: Configured Webull client used as the data source. Market
            data (get_bars/get_latest_quote) is confirmed to work for US
            symbols - see broker.webull_client's module docstring.
        symbols: Universe of symbols to fetch data for.
        timeframe: Bar timeframe (e.g. "1Day", "1Hour").
    """

    def __init__(self, client: WebullClient, symbols: list[str], timeframe: str) -> None:
        raise NotImplementedError

    def get_historical_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a symbol over a date range."""
        raise NotImplementedError

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Fetch the most recent completed bar for a symbol."""
        raise NotImplementedError

    def update(self) -> None:
        """Refresh cached data for all symbols in the universe."""
        raise NotImplementedError

    def subscribe_realtime(self) -> None:
        """Subscribe to a real-time data stream for the symbol universe."""
        raise NotImplementedError
