"""Real-time and historical data fetching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# Enough calendar days to comfortably cover core.hmm_engine's minimum training
# history (2+ years of trading days - see HMMEngine/backtest.backtester docs)
# even after weekends/holidays thin it out.
_DEFAULT_LOOKBACK_DAYS = 1000


class MarketDataFeed:
    """Fetches and caches historical and real-time market data.

    Args:
        client: Configured Alpaca client used as the data source.
        symbols: Universe of symbols to fetch data for.
        timeframe: Bar timeframe (e.g. "1Day", "1Hour").
        lookback_days: Calendar days of history `update()` fetches per symbol.
    """

    def __init__(
        self,
        client: AlpacaClient,
        symbols: list[str],
        timeframe: str,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self.client = client
        self.symbols = list(symbols)
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self._bars: dict[str, pd.DataFrame] = {}

    def get_historical_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a symbol over a date range (not cached)."""
        return self.client.get_bars(symbol, self.timeframe, start=start, end=end)

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Fetch the most recent completed bar for a symbol.

        Uses the cache populated by `update()` if available; otherwise fetches
        a short recent window directly.
        """
        if symbol in self._bars and not self._bars[symbol].empty:
            return self._bars[symbol].iloc[-1]
        end = datetime.now(timezone.utc).date().isoformat()
        start = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
        bars = self.get_historical_bars(symbol, start, end)
        if bars.empty:
            raise ValueError(f"No bars available for {symbol}")
        return bars.iloc[-1]

    def update(self) -> None:
        """Refresh cached data for every symbol in the universe.

        Fetches `lookback_days` of history per symbol - enough for the HMM's
        feature warm-up plus training window (see data.feature_engineering,
        core.hmm_engine). Logs (rather than raises) on a per-symbol fetch
        failure so one bad symbol doesn't block the rest of the universe.
        """
        end = datetime.now(timezone.utc).date().isoformat()
        start = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).date().isoformat()
        for symbol in self.symbols:
            try:
                self._bars[symbol] = self.get_historical_bars(symbol, start, end)
                logger.info("Updated %s: %d bars (%s to %s)", symbol, len(self._bars[symbol]), start, end)
            except Exception:
                logger.exception("Failed to update bars for %s", symbol)

    def get_cached_bars(self, symbol: str) -> pd.DataFrame:
        """Return the cached bar history for `symbol` (populated by `update()`)."""
        if symbol not in self._bars:
            raise KeyError(f"No cached bars for {symbol} - call update() first")
        return self._bars[symbol]

    def subscribe_realtime(self) -> None:
        """Subscribe to a real-time data stream for the symbol universe.

        Not yet implemented: real-time streaming needs its own
        connection-management/reconnect loop - see
        AlpacaClient.stream_trades - out of scope for this pass. This bot
        operates on daily bars (see config/settings.yaml broker.timeframe),
        so it doesn't need real-time data to function.
        """
        raise NotImplementedError
