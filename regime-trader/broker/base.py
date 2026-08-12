"""Broker-agnostic client interface.

Both broker/alpaca_client.py's AlpacaClient and broker/schwab_adapter.py's
SchwabAdapter satisfy this shape, so broker/order_executor.py,
broker/position_tracker.py, and main.py can hold either one without knowing
or caring which broker is actually behind it. AlpacaClient was written
first and this Protocol is deliberately shaped to match it exactly (zero
changes needed there beyond the two documented in AlpacaClient itself) -
SchwabAdapter is what does the work of making Schwab's very differently-
shaped API look the same.

This is intentionally a runtime-unenforced Protocol (structural typing) -
nothing here actually checks an object against it at import time. It exists
for documentation and static type-checking (mypy/pyright), and as the single
place that spells out exactly what "a broker" means in this codebase.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class BrokerClient(Protocol):
    """Everything core trading code needs from a broker, independent of
    which one is actually behind it.

    Dict shapes (deliberately Alpaca's own field names, since that client
    came first and needed zero changes to conform to this Protocol):

    - get_account(): {"equity", "cash", "buying_power", "last_equity", ...}
    - list_positions() / get_position(): [{"symbol", "qty",
      "avg_entry_price", "current_price", "unrealized_pl", "market_value",
      "asset_class"}, ...] - asset_class is "us_equity" or "us_option".
    - submit_order() / get_order() / list_orders(): {"id", "symbol", "qty",
      "side", "status", ...} - status is one of Alpaca's own vocabulary
      (broker/order_executor.py's _STATUS_MAP normalizes it further).
    - get_option_contract_details(): {"underlying_symbol", "type"
      ("call"/"put"), "expiration_date", "strike_price", ...}.
    """

    def get_account(self) -> dict: ...
    def is_market_open(self) -> bool: ...
    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame: ...
    def get_latest_quote(self, symbol: str) -> dict: ...

    def get_option_chain(
        self,
        underlying_symbol: str,
        contract_type: str,
        expiration_gte,
        expiration_lte,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict]: ...

    def get_option_contract_details(self, occ_symbol: str) -> dict: ...

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
    ) -> dict: ...

    def cancel_order(self, order_id: str) -> None: ...

    def replace_order(
        self,
        order_id: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str | None = None,
    ) -> dict: ...

    def get_order(self, order_id: str) -> dict: ...
    def list_orders(self, status: str = "open", after: str | datetime | None = None) -> list[dict]: ...
    def list_positions(self) -> list[dict]: ...
    def get_position(self, symbol: str) -> dict | None: ...
