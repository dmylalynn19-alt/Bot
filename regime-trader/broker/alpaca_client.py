"""Alpaca API wrapper.

Thin client around Alpaca's official `alpaca-py` SDK, covering both equities
and options (the legacy `alpaca-trade-api` package has no options support at
all - verified by inspecting its methods directly - so this wraps the
actively maintained SDK instead). Handles authentication and connection
setup for both paper and live trading.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, MarketOrderRequest

_TIMEFRAME_MAP = {
    "1Min": TimeFrame.Minute,
    "1Hour": TimeFrame.Hour,
    "1Day": TimeFrame.Day,
    "1Week": TimeFrame(1, TimeFrameUnit.Week),
    "1Month": TimeFrame(1, TimeFrameUnit.Month),
}

_TIME_IN_FORCE_MAP = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK}
_CONTRACT_TYPE_MAP = {"call": ContractType.CALL, "put": ContractType.PUT}


class AlpacaClient:
    """Wraps Alpaca's trading and market-data APIs (equities + options).

    Args:
        api_key: Alpaca API key.
        secret_key: Alpaca API secret key.
        paper: Whether to connect to the paper trading endpoint.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool) -> None:
        self.paper = paper
        self._trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self._stock_data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self._option_data = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    # ------------------------------------------------------------------
    # Account / clock
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Fetch account details (equity, buying power, status, options approval level)."""
        return self._trading.get_account().model_dump()

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        return bool(self._trading.get_clock().is_open)

    # ------------------------------------------------------------------
    # Equity market data
    # ------------------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical equity bars for a symbol."""
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'; supported: {sorted(_TIMEFRAME_MAP)}")
        bars = self._stock_data.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbol, timeframe=_TIMEFRAME_MAP[timeframe], start=start, end=end)
        )
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")
        return df

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest equity quote for a symbol."""
        from alpaca.data.requests import StockLatestQuoteRequest

        quotes = self._stock_data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        return quotes[symbol].model_dump()

    # ------------------------------------------------------------------
    # Options market data
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        underlying_symbol: str,
        contract_type: str,
        expiration_gte: str | date,
        expiration_lte: str | date,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict]:
        """Fetch the live option chain for `underlying_symbol`, filtered to one
        side (calls or puts) and an expiration/strike window.

        Returns a flat list of dicts, one per contract:
        {occ_symbol, strike, expiration, bid, ask, mid, delta, gamma, theta,
        vega, implied_volatility} - bid/ask/greeks are None if Alpaca doesn't
        have a current quote for that contract (e.g. illiquid strikes).
        """
        if contract_type not in _CONTRACT_TYPE_MAP:
            raise ValueError(f"contract_type must be 'call' or 'put', got '{contract_type}'")

        snapshots = self._option_data.get_option_chain(
            OptionChainRequest(
                underlying_symbol=underlying_symbol,
                type=_CONTRACT_TYPE_MAP[contract_type],
                expiration_date_gte=expiration_gte,
                expiration_date_lte=expiration_lte,
                strike_price_gte=strike_gte,
                strike_price_lte=strike_lte,
            )
        )

        contracts_by_symbol = {
            c.symbol: c
            for c in self._trading.get_option_contracts(
                GetOptionContractsRequest(
                    underlying_symbols=[underlying_symbol],
                    type=_CONTRACT_TYPE_MAP[contract_type],
                    expiration_date_gte=expiration_gte,
                    expiration_date_lte=expiration_lte,
                    strike_price_gte=str(strike_gte) if strike_gte is not None else None,
                    strike_price_lte=str(strike_lte) if strike_lte is not None else None,
                )
            ).option_contracts
        }

        rows = []
        for occ_symbol, snapshot in snapshots.items():
            contract = contracts_by_symbol.get(occ_symbol)
            quote = snapshot.latest_quote
            greeks = snapshot.greeks
            bid = float(quote.bid_price) if quote and quote.bid_price else None
            ask = float(quote.ask_price) if quote and quote.ask_price else None
            rows.append(
                {
                    "occ_symbol": occ_symbol,
                    "strike": float(contract.strike_price) if contract else None,
                    "expiration": contract.expiration_date.isoformat() if contract else None,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2 if bid is not None and ask is not None else None,
                    "delta": greeks.delta if greeks else None,
                    "gamma": greeks.gamma if greeks else None,
                    "theta": greeks.theta if greeks else None,
                    "vega": greeks.vega if greeks else None,
                    "implied_volatility": snapshot.implied_volatility,
                }
            )
        return rows

    def get_option_contract_details(self, occ_symbol: str) -> dict:
        """Look up a single option contract's metadata (underlying_symbol,
        strike_price, expiration_date, type, ...) by its OCC symbol."""
        return self._trading.get_option_contract(occ_symbol).model_dump()

    # ------------------------------------------------------------------
    # Orders (works for both equity symbols and OCC option symbols - Alpaca
    # routes on the symbol format)
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        asset_class: str = "equity",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Submit an order. `symbol` may be an equity ticker or an OCC
        option symbol (from get_option_chain) - Alpaca routes on the symbol
        format itself, so `asset_class` is accepted only for interface
        parity with broker.base.BrokerClient (e.g. SchwabAdapter, which
        does need it) and otherwise ignored here.

        `side` accepts both Alpaca's plain "buy"/"sell" and the open/close
        vocabulary broker.order_executor.OrderExecutor uses for options
        ("buy_to_open", "sell_to_close", "sell_to_open", "buy_to_close") -
        Alpaca itself doesn't distinguish open/close for a simple long-only
        single-leg order, so these all collapse to plain buy/sell here.
        """
        tif = _TIME_IN_FORCE_MAP.get(time_in_force, TimeInForce.DAY)
        order_side = OrderSide.BUY if side in ("buy", "buy_to_open", "buy_to_close") else OrderSide.SELL
        if order_type == "limit":
            request = LimitOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=tif, limit_price=limit_price, client_order_id=client_order_id
            )
        else:
            request = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=tif, client_order_id=client_order_id)
        return self._trading.submit_order(request).model_dump()

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by ID."""
        self._trading.cancel_order_by_id(order_id)

    def replace_order(
        self,
        order_id: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str | None = None,
    ) -> dict:
        """Modify an open order's quantity/price/time-in-force."""
        from alpaca.trading.requests import ReplaceOrderRequest

        tif = _TIME_IN_FORCE_MAP.get(time_in_force) if time_in_force else None
        request = ReplaceOrderRequest(qty=qty, limit_price=limit_price, time_in_force=tif)
        return self._trading.replace_order_by_id(order_id, request).model_dump()

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order's current state by ID."""
        return self._trading.get_order_by_id(order_id).model_dump()

    def list_orders(self, status: str = "open", after: str | datetime | None = None) -> list[dict]:
        """List orders. `status` is "open", "closed", or "all"."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(status=QueryOrderStatus(status), after=after)
        return [order.model_dump() for order in self._trading.get_orders(request)]

    # ------------------------------------------------------------------
    # Positions (equity and option positions both come back from
    # get_all_positions - check the "asset_class" field to tell them apart)
    # ------------------------------------------------------------------

    def list_positions(self) -> list[dict]:
        """List all currently open positions (equity and option)."""
        return [position.model_dump() for position in self._trading.get_all_positions()]

    def get_position(self, symbol: str) -> dict | None:
        """Fetch the open position for a single symbol, or None if there isn't one."""
        try:
            return self._trading.get_open_position(symbol).model_dump()
        except Exception as exc:
            if "position does not exist" in str(exc).lower() or "404" in str(exc):
                return None
            raise

    def stream_trades(self, symbols: list[str]) -> None:
        """Subscribe to a real-time trade/quote stream for the given symbols.

        Not yet implemented: real-time streaming needs its own
        connection-management/reconnect loop (alpaca.data.live) - out of
        scope for this pass.
        """
        raise NotImplementedError
