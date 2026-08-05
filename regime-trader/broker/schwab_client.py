"""Charles Schwab (Trader API) broker wrapper, via the `schwab-py` SDK.

Auth: Schwab uses OAuth2, not a static API key/secret pair like Alpaca. The
first time this runs (no token file yet), `client_from_manual_flow` walks
you through a copy-paste login in the terminal (works over SSH/Codespaces -
no local browser/webserver needed, unlike `client_from_login_flow`): it
prints a URL, you open it, log in, and paste the URL you land on back into
the terminal. After that, the resulting token is cached at `token_path` and
silently refreshed on every subsequent run via `client_from_token_file` -
no more interaction needed unless the refresh token itself expires (Schwab
currently expires it after 7 days of the app going *completely* unused;
running this daily keeps it alive).

Schwab responses are NOT normalized to match AlpacaClient's shape here -
Schwab's JSON schema (`securitiesAccount.currentBalances.*`,
`securitiesAccount.positions[]`, etc.) is quite different from Alpaca's,
and a broker-agnostic normalization/routing layer is still a separate,
not-yet-built piece of this project (see README). Every method below
returns Schwab's own response JSON as a dict - check the Trader API docs
(https://developer.schwab.com) or the docstrings here for the exact paths.

Options: `symbol` for options is Schwab's own OSI-style symbol format
(`[underlying padded to 6 chars][YYMMDD][C/P][8-digit strike*1000]`) -
identical in structure to the standard OCC format Alpaca uses. Symbols
returned by `get_option_chain` are already in this format and can be
passed straight to `submit_order`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from schwab.auth import client_from_manual_flow, client_from_token_file
from schwab.orders import equities as equity_orders
from schwab.orders import options as option_orders
from schwab.orders.common import Duration
from schwab.orders.options import OptionSymbol
from schwab.utils import Utils

# (period_type, period, frequency_type, frequency) - schwab-py's own
# get_price_history_every_* helpers use these same combinations; built out
# explicitly here since we need start/end-datetime-driven fetches, which
# those convenience wrappers also support via passthrough kwargs.
_INTRADAY_MINUTES = {"1Min": 1, "5Min": 5, "10Min": 10, "15Min": 15, "30Min": 30}

_DURATION_MAP = {
    "day": Duration.DAY,
    "gtc": Duration.GOOD_TILL_CANCEL,
    "ioc": Duration.IMMEDIATE_OR_CANCEL,
    "fok": Duration.FILL_OR_KILL,
}

_OPEN_ORDER_STATUSES = [
    "AWAITING_PARENT_ORDER", "AWAITING_CONDITION", "AWAITING_STOP_CONDITION", "AWAITING_MANUAL_REVIEW",
    "ACCEPTED", "AWAITING_UR_OUT", "PENDING_ACTIVATION", "QUEUED", "WORKING", "PENDING_CANCEL",
    "PENDING_REPLACE", "NEW", "AWAITING_RELEASE_TIME", "PENDING_ACKNOWLEDGEMENT", "PENDING_RECALL",
]
_CLOSED_ORDER_STATUSES = ["FILLED", "CANCELED", "REJECTED", "REPLACED", "EXPIRED"]

_EQUITY_ORDER_BUILDERS = {
    ("buy", "market"): lambda symbol, qty, price: equity_orders.equity_buy_market(symbol, qty),
    ("buy", "limit"): lambda symbol, qty, price: equity_orders.equity_buy_limit(symbol, qty, price),
    ("sell", "market"): lambda symbol, qty, price: equity_orders.equity_sell_market(symbol, qty),
    ("sell", "limit"): lambda symbol, qty, price: equity_orders.equity_sell_limit(symbol, qty, price),
}

_OPTION_ORDER_BUILDERS = {
    ("buy_to_open", "market"): lambda symbol, qty, price: option_orders.option_buy_to_open_market(symbol, qty),
    ("buy_to_open", "limit"): lambda symbol, qty, price: option_orders.option_buy_to_open_limit(symbol, qty, price),
    ("sell_to_close", "market"): lambda symbol, qty, price: option_orders.option_sell_to_close_market(symbol, qty),
    ("sell_to_close", "limit"): lambda symbol, qty, price: option_orders.option_sell_to_close_limit(symbol, qty, price),
    ("sell_to_open", "market"): lambda symbol, qty, price: option_orders.option_sell_to_open_market(symbol, qty),
    ("sell_to_open", "limit"): lambda symbol, qty, price: option_orders.option_sell_to_open_limit(symbol, qty, price),
    ("buy_to_close", "market"): lambda symbol, qty, price: option_orders.option_buy_to_close_market(symbol, qty),
    ("buy_to_close", "limit"): lambda symbol, qty, price: option_orders.option_buy_to_close_limit(symbol, qty, price),
}


class SchwabClient:
    """Wraps Schwab's Trader API (equities + options) via schwab-py.

    Args:
        api_key: Your Schwab app's "App Key" (from the app's detail page on
            developer.schwab.com).
        app_secret: Your Schwab app's "Secret".
        callback_url: Your Schwab app's registered callback URL (must match
            exactly, including trailing slash).
        token_path: Where the OAuth token is cached/refreshed. Created on
            first run via the manual login flow; reused (and silently
            refreshed) on every run after that.
        account_index: Which of your linked Schwab accounts to use, if more
            than one is linked to this app (0 = first one returned by
            Schwab - order is not guaranteed to be meaningful; check
            `list_account_numbers` if you have more than one account and
            need a specific one).
    """

    def __init__(self, api_key: str, app_secret: str, callback_url: str, token_path: str, account_index: int = 0) -> None:
        self.token_path = token_path
        if Path(token_path).exists():
            self._client = client_from_token_file(token_path, api_key, app_secret)
        else:
            self._client = client_from_manual_flow(api_key, app_secret, callback_url, token_path)

        account_numbers = self._json(self._client.get_account_numbers())
        if not account_numbers:
            raise RuntimeError("Schwab returned no linked accounts for this token - check app/account linkage")
        self.account_hash = account_numbers[account_index]["hashValue"]
        self._utils = Utils(self._client, self.account_hash)

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        response.raise_for_status()
        return response.json()

    def list_account_numbers(self) -> list[dict]:
        """Raw accountNumber -> hashValue mappings for every account linked
        to this app (use to pick `account_index` if you have more than one)."""
        return self._json(self._client.get_account_numbers())

    # ------------------------------------------------------------------
    # Account / clock
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Fetch account details, including positions
        (`securitiesAccount.positions`) and balances
        (`securitiesAccount.currentBalances`)."""
        return self._json(self._client.get_account(self.account_hash, fields=self._client.Account.Fields.POSITIONS))

    def is_market_open(self) -> bool:
        """Check whether the equity market is open right now."""
        hours = self._json(self._client.get_market_hours([self._client.MarketHours.Market.EQUITY]))
        equity_hours = hours.get("equity", {})
        # Response is keyed by product (e.g. "EQ") under "equity" - check
        # every product Schwab returns for this market; any open counts.
        return any(product.get("isOpen", False) for product in equity_hours.values())

    # ------------------------------------------------------------------
    # Equity market data
    # ------------------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical equity bars for a symbol.

        `timeframe` is one of "1Min"/"5Min"/"10Min"/"15Min"/"30Min" (native
        Schwab intraday frequencies), "1Hour" (fetched as 30Min bars and
        resampled - Schwab has no native hourly frequency), "1Day", or
        "1Week". Returns a DataFrame indexed by UTC timestamp with
        open/high/low/close/volume columns, most recent bar last.
        """
        start_dt, end_dt = pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()
        PH = self._client.PriceHistory

        if timeframe == "1Day":
            response = self._client.get_price_history(
                symbol, period_type=PH.PeriodType.YEAR, frequency_type=PH.FrequencyType.DAILY,
                frequency=PH.Frequency.DAILY, start_datetime=start_dt, end_datetime=end_dt,
            )
        elif timeframe == "1Week":
            response = self._client.get_price_history(
                symbol, period_type=PH.PeriodType.YEAR, frequency_type=PH.FrequencyType.WEEKLY,
                frequency=PH.Frequency.WEEKLY, start_datetime=start_dt, end_datetime=end_dt,
            )
        elif timeframe in _INTRADAY_MINUTES or timeframe == "1Hour":
            minutes = 30 if timeframe == "1Hour" else _INTRADAY_MINUTES[timeframe]
            frequency = {1: PH.Frequency.EVERY_MINUTE, 5: PH.Frequency.EVERY_FIVE_MINUTES, 10: PH.Frequency.EVERY_TEN_MINUTES,
                         15: PH.Frequency.EVERY_FIFTEEN_MINUTES, 30: PH.Frequency.EVERY_THIRTY_MINUTES}[minutes]
            response = self._client.get_price_history(
                symbol, period_type=PH.PeriodType.DAY, frequency_type=PH.FrequencyType.MINUTE,
                frequency=frequency, start_datetime=start_dt, end_datetime=end_dt, need_extended_hours_data=True,
            )
        else:
            raise ValueError(f"Unsupported timeframe '{timeframe}'")

        payload = self._json(response)
        candles = payload.get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].sort_index()

        if timeframe == "1Hour":
            df = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        return df

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest equity quote for a symbol (raw Schwab quote JSON)."""
        return self._json(self._client.get_quote(symbol))

    # ------------------------------------------------------------------
    # Options market data
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        underlying_symbol: str,
        contract_type: str,
        expiration_gte: str | dt.date,
        expiration_lte: str | dt.date,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict]:
        """Fetch the live option chain for `underlying_symbol`, filtered to
        one side (calls or puts) and an expiration window.

        Returns a flat list of dicts, one per contract: {occ_symbol,
        strike, expiration, bid, ask, mid, delta, gamma, theta, vega,
        implied_volatility} - same shape as AlpacaClient.get_option_chain,
        for interchangeability once a broker-routing layer exists.

        `strike_gte`/`strike_lte` aren't native Schwab chain filters (Schwab
        filters by `strike_count` around the money, not a strike range) -
        applied client-side after fetching instead.
        """
        contract_type_map = {"call": self._client.Options.ContractType.CALL, "put": self._client.Options.ContractType.PUT}
        if contract_type not in contract_type_map:
            raise ValueError(f"contract_type must be 'call' or 'put', got '{contract_type}'")

        payload = self._json(
            self._client.get_option_chain(
                underlying_symbol,
                contract_type=contract_type_map[contract_type],
                from_date=expiration_gte,
                to_date=expiration_lte,
            )
        )

        map_key = "callExpDateMap" if contract_type == "call" else "putExpDateMap"
        rows = []
        for _expiration_key, strikes in payload.get(map_key, {}).items():
            for strike_str, contracts in strikes.items():
                strike = float(strike_str)
                if strike_gte is not None and strike < strike_gte:
                    continue
                if strike_lte is not None and strike > strike_lte:
                    continue
                for contract in contracts:  # normally a single-element list per strike
                    bid = contract.get("bid")
                    ask = contract.get("ask")
                    rows.append(
                        {
                            "occ_symbol": contract.get("symbol"),
                            "strike": strike,
                            "expiration": contract.get("expirationDate"),
                            "bid": bid,
                            "ask": ask,
                            "mid": (bid + ask) / 2 if bid is not None and ask is not None else None,
                            "delta": contract.get("delta"),
                            "gamma": contract.get("gamma"),
                            "theta": contract.get("theta"),
                            "vega": contract.get("vega"),
                            "implied_volatility": contract.get("volatility"),
                        }
                    )
        return rows

    def get_option_contract_details(self, occ_symbol: str) -> dict:
        """Look up a single option contract by requesting a 1-strike chain
        for it (Schwab has no direct "look up by symbol" endpoint - this
        parses the symbol to derive the underlying/expiration/strike/type
        needed to query for it)."""
        parsed = OptionSymbol.parse_symbol(occ_symbol)
        contract_type = "call" if parsed.contract_type == "C" else "put"
        strike = float(parsed.strike_price)
        matches = self.get_option_chain(
            parsed.underlying_symbol, contract_type, parsed.expiration_date, parsed.expiration_date, strike, strike
        )
        for match in matches:
            if match["occ_symbol"] == occ_symbol:
                return match
        raise ValueError(f"No contract found for symbol '{occ_symbol}'")

    # ------------------------------------------------------------------
    # Orders
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
    ) -> dict:
        """Submit an order.

        `asset_class` is "equity" or "option" - Schwab uses entirely
        different order-builder semantics for each, so unlike Alpaca this
        isn't auto-detected from the symbol format. For equities, `side` is
        "buy" or "sell". For options, `side` is one of "buy_to_open",
        "sell_to_close", "sell_to_open", "buy_to_close".

        Returns {"order_id": int | None, "status_code": int}. `order_id` is
        parsed from the response's Location header - see schwab.utils.Utils.
        """
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders")

        builders = _EQUITY_ORDER_BUILDERS if asset_class == "equity" else _OPTION_ORDER_BUILDERS if asset_class == "option" else None
        if builders is None:
            raise ValueError(f"asset_class must be 'equity' or 'option', got '{asset_class}'")

        key = (side, order_type)
        if key not in builders:
            raise ValueError(f"Unsupported (side={side!r}, order_type={order_type!r}) for asset_class={asset_class!r}")

        # schwab-py deprecated passing prices as float (silent truncation
        # footguns) in favor of an explicit string - format to cents here
        # rather than let the SDK truncate it implicitly.
        price_str = f"{limit_price:.2f}" if limit_price is not None else None
        order = builders[key](symbol, int(qty), price_str)
        order.set_duration(_DURATION_MAP.get(time_in_force, Duration.DAY))

        response = self._client.place_order(self.account_hash, order)
        order_id = self._utils.extract_order_id(response)
        return {"order_id": order_id, "status_code": response.status_code}

    def cancel_order(self, order_id: str | int) -> None:
        """Cancel an open order by ID."""
        self._json_or_none(self._client.cancel_order(order_id, self.account_hash))

    def get_order(self, order_id: str | int) -> dict:
        """Fetch a single order's current state by ID."""
        return self._json(self._client.get_order(order_id, self.account_hash))

    def list_orders(self, status: str = "open", after: str | dt.datetime | None = None) -> list[dict]:
        """List orders for this account. `status` is "open", "closed", or
        "all". Schwab requires an explicit date window - defaults to the
        last 60 days (the widest Schwab supports) through now if `after`
        isn't given."""
        from_dt = pd.Timestamp(after).to_pydatetime() if after is not None else dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=60)
        to_dt = dt.datetime.now(dt.timezone.utc)

        statuses = {"open": _OPEN_ORDER_STATUSES, "closed": _CLOSED_ORDER_STATUSES, "all": None}.get(status)
        if status not in {"open", "closed", "all"}:
            raise ValueError(f"status must be 'open', 'closed', or 'all', got '{status}'")

        response = self._client.get_orders_for_account(
            self.account_hash, from_entered_datetime=from_dt, to_entered_datetime=to_dt,
            statuses=statuses,
        )
        return self._json(response)

    @staticmethod
    def _json_or_none(response: httpx.Response) -> None:
        response.raise_for_status()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def list_positions(self) -> list[dict]:
        """List all currently open positions (equity and option) - each
        dict's shape follows Schwab's `securitiesAccount.positions[]`
        entries (instrument.symbol, longQuantity/shortQuantity,
        averagePrice, marketValue, ...)."""
        return self.get_account().get("securitiesAccount", {}).get("positions", [])

    def get_position(self, symbol: str) -> dict | None:
        """Fetch the open position for a single symbol, or None if there isn't one."""
        for position in self.list_positions():
            if position.get("instrument", {}).get("symbol") == symbol:
                return position
        return None

    def stream_trades(self, symbols: list[str]) -> None:
        """Subscribe to a real-time trade/quote stream for the given symbols.

        Not yet implemented: schwab-py's streaming client (schwab.streaming)
        needs its own connection-management/reconnect loop, same scope note
        as AlpacaClient.stream_trades.
        """
        raise NotImplementedError
