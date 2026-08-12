"""Targeted tests for broker.alpaca_client.AlpacaClient.

Only covers the one behavior actually changed in this session (submit_order
accepting broker.base.BrokerClient's broader side vocabulary / an
asset_class param for interface parity with SchwabAdapter) - AlpacaClient's
broader request-building logic was already verified structurally against
real request construction in an earlier session (see project history), not
re-covered here from scratch.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from alpaca.trading.enums import OrderSide

from broker.alpaca_client import AlpacaClient


def _client() -> AlpacaClient:
    # TradingClient/StockHistoricalDataClient/OptionHistoricalDataClient
    # don't make network calls on construction - just set up a session -
    # so dummy credentials are safe here.
    client = AlpacaClient(api_key="fake", secret_key="fake", paper=True)
    client._trading = MagicMock()
    return client


def test_submit_order_plain_buy_sell_still_works() -> None:
    client = _client()
    client.submit_order("AAPL", 10, "buy", order_type="market")
    request = client._trading.submit_order.call_args[0][0]
    assert request.side == OrderSide.BUY

    client.submit_order("AAPL", 10, "sell", order_type="market")
    request = client._trading.submit_order.call_args[0][0]
    assert request.side == OrderSide.SELL


def test_submit_order_accepts_open_close_vocabulary_for_options() -> None:
    client = _client()

    client.submit_order("AAPL  240621C00160000", 1, "buy_to_open", asset_class="option", order_type="limit", limit_price=2.5)
    assert client._trading.submit_order.call_args[0][0].side == OrderSide.BUY

    client.submit_order("AAPL  240621C00160000", 1, "sell_to_close", asset_class="option", order_type="market")
    assert client._trading.submit_order.call_args[0][0].side == OrderSide.SELL

    client.submit_order("AAPL  240621C00160000", 1, "buy_to_close", asset_class="option", order_type="market")
    assert client._trading.submit_order.call_args[0][0].side == OrderSide.BUY

    client.submit_order("AAPL  240621C00160000", 1, "sell_to_open", asset_class="option", order_type="market")
    assert client._trading.submit_order.call_args[0][0].side == OrderSide.SELL


def test_submit_order_asset_class_param_is_accepted_and_ignored() -> None:
    """asset_class exists only for interface parity with SchwabAdapter -
    Alpaca infers equity vs. option from the symbol format itself."""
    client = _client()
    client.submit_order("AAPL", 10, "buy", asset_class="equity", order_type="market")
    client.submit_order("AAPL  240621C00160000", 1, "buy_to_open", asset_class="option", order_type="market")
    assert client._trading.submit_order.call_count == 2
