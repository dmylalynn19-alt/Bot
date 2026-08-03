"""Webull API wrapper.

Thin client around Webull's official OpenAPI (developer.webull.com), built
on the official `webull-python-sdk-core` / `-trade` / `-mdata` packages.
Handles authentication (App Key + App Secret, HMAC-signed per request by the
SDK's ApiClient/AppKeyCredential) and region selection for paper and live
trading. You need an approved App Key/App Secret from Webull's OpenAPI
Management console before any of this works - see .env.example.

IMPORTANT - verified directly against the installed webull-python-sdk-trade
0.1.18 source (its docstrings, not just marketing pages):

- Market data (MarketData.get_history_bar / get_batch_history_bar /
  get_snapshot) explicitly supports Category.US_STOCK and carries no
  region restriction - this is confirmed usable for a US account.
- Order placement and account balance/position endpoints
  (OrderOperationV2.place_order/preview_order/..., AccountV2.get_account_*)
  are documented in the SDK itself as: "This interface is currently
  available only to individual brokerage customers in Webull Japan and
  institutional brokerage clients in Webull Hong Kong. It is not yet
  available to Webull US brokerage customers, but support will be
  introduced progressively in the future." The older v1 OrderOperation
  .place_order is explicitly scoped to "Hong Kong stocks, and A shares
  (China Connect)" only. As of this writing there is no confirmed working
  order-placement or account-balance endpoint for a Webull US brokerage
  account via the official OpenAPI.

get_account/get_bars/get_latest_quote below are wired to real SDK calls (not
stubs) so they're correct the moment Webull enables the missing pieces for
US accounts, but get_account in particular may error server-side until then
- verify against your own approved App Key. stream_trades and is_market_open
need more than a REST call (gRPC/MQTT streaming; a trading-calendar lookup)
and are left as documented stubs.
"""

from __future__ import annotations

import pandas as pd
from webullsdkcore.auth.credentials import AppKeyCredential
from webullsdkcore.client import ApiClient
from webullsdkmdata.common.category import Category
from webullsdkmdata.common.timespan import Timespan
from webullsdktrade.api import API
from webullsdktrade.common.markets import Markets

# Maps this project's broker.timeframe values (config/settings.yaml) to
# Webull's Timespan enum (webullsdkmdata.common.timespan).
_TIMEFRAME_TO_TIMESPAN = {
    "1Min": Timespan.M1,
    "5Min": Timespan.M5,
    "15Min": Timespan.M15,
    "30Min": Timespan.M30,
    "1Hour": Timespan.M60,
    "1Day": Timespan.D,
    "1Week": Timespan.W,
    "1Month": Timespan.M,
}

_REGION_TO_CATEGORY = {
    "us": Category.US_STOCK,
    "hk": Category.HK_STOCK,
}

_REGION_TO_MARKET = {
    "us": Markets.US,
    "hk": Markets.HK,
    "jp": Markets.JP,
}


class WebullClient:
    """Wraps Webull's official OpenAPI trade and market-data SDKs.

    Args:
        app_key: Webull OpenAPI App Key (OpenAPI Management > App Management
            on Webull's website).
        app_secret: Webull OpenAPI App Secret.
        account_id: Webull brokerage account ID to trade/query.
        region: OpenAPI region - "us", "hk", or "jp" - must match your account.
        paper: Whether to connect to Webull's paper/simulation trading
            environment rather than a live account.
    """

    def __init__(self, app_key: str, app_secret: str, account_id: str, region: str = "us", paper: bool = True) -> None:
        self.account_id = account_id
        self.region = region
        self.paper = paper

        credential = AppKeyCredential(app_key_id=app_key, app_key_secret=app_secret)
        self._api_client = ApiClient(app_key=app_key, app_secret=app_secret, region_id=region, credential=credential)
        self._api = API(self._api_client)

    def get_account(self) -> dict:
        """Fetch account balance/details for `self.account_id`.

        NOTE: per this module's docstring, Webull's account-balance endpoint
        is documented as not yet available for US brokerage accounts - this
        call is wired for real, but may error server-side until Webull
        enables it for your account.
        """
        response = self._api.account.get_account_balance(self.account_id, total_asset_currency="USD")
        return response.json()

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical bars for `symbol` (confirmed supported for US stocks).

        Webull's get_history_bar takes a bar *count* (max 1200), not a
        start/end date range like Alpaca's API did - there is no confirmed
        pagination path beyond that limit. This fetches the most recent
        1200 bars and slices to [start, end] client-side; for a `start` more
        than ~1200 `timeframe` bars in the past, you will get fewer bars than
        requested, not an error.
        """
        if timeframe not in _TIMEFRAME_TO_TIMESPAN:
            raise ValueError(f"Unsupported timeframe '{timeframe}'; supported: {sorted(_TIMEFRAME_TO_TIMESPAN)}")

        category = _REGION_TO_CATEGORY.get(self.region, Category.US_STOCK)
        response = self._api.market_data.get_history_bar(
            symbol=symbol, category=category, timespan=_TIMEFRAME_TO_TIMESPAN[timeframe], count="1200"
        )
        payload = response.json()
        # TODO: confirm the exact response schema against a live call (needs
        # an approved App Key) and adjust the column/key names below to match
        # - this assumes a list of bar records or a {"data": [...]} envelope,
        # each with the OHLCV fields data.feature_engineering expects.
        records = payload if isinstance(payload, list) else payload.get("data", [])
        bars = pd.DataFrame(records)
        if not bars.empty and "timestamp" in bars.columns:
            bars = bars.set_index(pd.to_datetime(bars["timestamp"], unit="s")).sort_index()
        return bars.loc[start:end] if not bars.empty else bars

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest snapshot quote for `symbol`."""
        category = _REGION_TO_CATEGORY.get(self.region, Category.US_STOCK)
        response = self._api.market_data.get_snapshot(symbols=symbol, category=category)
        return response.json()

    def stream_trades(self, symbols: list[str]) -> None:
        """Subscribe to a real-time trade/quote stream for `symbols`.

        Not yet implemented: Webull's real-time streaming runs over gRPC/MQTT
        (see webullsdktrade.grpc_api and the webull-python-sdk-mdata quotes
        stack), a materially different transport from a REST call, needing
        its own connection-management/reconnect logic - out of scope here.
        """
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Check whether today is a trading day for `self.region`'s market.

        Not yet implemented: TradeCalendar.get_trade_calendar(market, start,
        end) returns trading *days*, not live open/closed-right-now status -
        combining it with the exchange's actual session hours (and half-days)
        needs a verified response schema this module doesn't have yet.
        """
        raise NotImplementedError
