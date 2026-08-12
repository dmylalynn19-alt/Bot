"""Tests for broker.schwab_adapter.SchwabAdapter.

Mocks broker.schwab_client.SchwabClient itself (already tested against the
real schwab-py SDK in tests/test_schwab_client.py) - these verify only the
adapter's normalization logic: Schwab's nested/differently-named JSON
mapped onto broker.base.BrokerClient's Alpaca-shaped interface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from broker.schwab_adapter import SchwabAdapter


@pytest.fixture
def fake_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(fake_client) -> SchwabAdapter:
    return SchwabAdapter(fake_client)


def test_get_account_normalizes_balances(adapter, fake_client) -> None:
    fake_client.get_account.return_value = {
        "securitiesAccount": {
            "currentBalances": {"liquidationValue": 105_000.0, "cashBalance": 20_000.0, "buyingPower": 40_000.0},
            "initialBalances": {"liquidationValue": 100_000.0},
        }
    }
    account = adapter.get_account()
    assert account == {"equity": 105_000.0, "cash": 20_000.0, "buying_power": 40_000.0, "last_equity": 100_000.0, "status": "ACTIVE"}


def test_get_account_handles_missing_fields_gracefully(adapter, fake_client) -> None:
    fake_client.get_account.return_value = {"securitiesAccount": {}}
    account = adapter.get_account()
    assert account["equity"] == 0.0
    assert account["last_equity"] == 0.0


def test_normalize_position_long_equity(adapter, fake_client) -> None:
    fake_client.list_positions.return_value = [
        {
            "instrument": {"symbol": "AAPL", "assetType": "EQUITY"},
            "longQuantity": 10.0,
            "shortQuantity": 0.0,
            "averageLongPrice": 150.0,
            "marketValue": 1600.0,
        }
    ]
    positions = adapter.list_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos["symbol"] == "AAPL"
    assert pos["qty"] == 10.0
    assert pos["avg_entry_price"] == 150.0
    assert pos["current_price"] == pytest.approx(160.0)
    assert pos["unrealized_pl"] == pytest.approx(100.0)
    assert pos["asset_class"] == "us_equity"


def test_normalize_position_short_position_has_negative_qty(adapter, fake_client) -> None:
    fake_client.list_positions.return_value = [
        {
            "instrument": {"symbol": "TSLA", "assetType": "EQUITY"},
            "longQuantity": 0.0,
            "shortQuantity": 5.0,
            "averageShortPrice": 200.0,
            "marketValue": -900.0,
        }
    ]
    pos = adapter.list_positions()[0]
    assert pos["qty"] == -5.0


def test_normalize_position_option_asset_class(adapter, fake_client) -> None:
    fake_client.list_positions.return_value = [
        {
            "instrument": {"symbol": "AAPL  240621C00160000", "assetType": "OPTION"},
            "longQuantity": 2.0,
            "shortQuantity": 0.0,
            "averageLongPrice": 2.10,
            "marketValue": 500.0,
        }
    ]
    pos = adapter.list_positions()[0]
    assert pos["asset_class"] == "us_option"


def test_get_position_returns_none_when_absent(adapter, fake_client) -> None:
    fake_client.get_position.return_value = None
    assert adapter.get_position("MSFT") is None


def test_get_option_contract_details_parses_symbol_without_extra_call(adapter, fake_client) -> None:
    details = adapter.get_option_contract_details("AAPL  240621C00160000")
    assert details["underlying_symbol"] == "AAPL"
    assert details["type"] == "call"
    assert details["strike_price"] == 160.0
    assert details["expiration_date"] == "2024-06-21"
    fake_client.get_option_contract_details.assert_not_called()


def test_submit_order_fetches_full_state_after_placing(adapter, fake_client) -> None:
    fake_client.submit_order.return_value = {"order_id": 12345, "status_code": 201}
    fake_client.get_order.return_value = {
        "orderId": 12345,
        "status": "WORKING",
        "quantity": 10,
        "filledQuantity": 0,
        "orderLegCollection": [{"instruction": "BUY", "instrument": {"symbol": "AAPL"}}],
    }
    order = adapter.submit_order("AAPL", 10, "buy", asset_class="equity", order_type="market")
    assert order == {"id": "12345", "symbol": "AAPL", "qty": 10.0, "side": "buy", "status": "pending"}
    fake_client.get_order.assert_called_once_with("12345")


def test_submit_order_handles_missing_order_id(adapter, fake_client) -> None:
    fake_client.submit_order.return_value = {"order_id": None, "status_code": 500}
    order = adapter.submit_order("AAPL", 10, "buy")
    assert order["status"] == "pending"
    assert order["id"] == ""


def test_submit_order_survives_a_failed_followup_lookup(adapter, fake_client) -> None:
    fake_client.submit_order.return_value = {"order_id": 999, "status_code": 201}
    fake_client.get_order.side_effect = RuntimeError("network blip")
    order = adapter.submit_order("AAPL", 10, "buy")
    assert order["id"] == "999"
    assert order["status"] == "pending"


@pytest.mark.parametrize(
    "schwab_status,filled_qty,requested_qty,expected",
    [
        ("FILLED", 10, 10, "filled"),
        ("CANCELED", 0, 10, "canceled"),
        ("EXPIRED", 0, 10, "canceled"),
        ("REPLACED", 0, 10, "canceled"),
        ("REJECTED", 0, 10, "rejected"),
        ("WORKING", 4, 10, "partially_filled"),
        ("WORKING", 0, 10, "pending"),
        ("ACCEPTED", 0, 10, "pending"),
    ],
)
def test_normalize_order_status_mapping(adapter, fake_client, schwab_status, filled_qty, requested_qty, expected) -> None:
    fake_client.get_order.return_value = {
        "orderId": 1,
        "status": schwab_status,
        "quantity": requested_qty,
        "filledQuantity": filled_qty,
        "orderLegCollection": [{"instruction": "SELL_TO_CLOSE", "instrument": {"symbol": "AAPL"}}],
    }
    order = adapter.get_order("1")
    assert order["status"] == expected
    assert order["side"] == "sell"


def test_replace_order_raises_not_implemented(adapter) -> None:
    with pytest.raises(NotImplementedError):
        adapter.replace_order("1", qty=5)


def test_passthrough_methods_delegate_to_client(adapter, fake_client) -> None:
    adapter.is_market_open()
    fake_client.is_market_open.assert_called_once()

    adapter.get_bars("AAPL", "1Day", "2024-01-01", "2024-02-01")
    fake_client.get_bars.assert_called_once_with("AAPL", "1Day", "2024-01-01", "2024-02-01")

    adapter.get_latest_quote("AAPL")
    fake_client.get_latest_quote.assert_called_once_with("AAPL")

    adapter.get_option_chain("AAPL", "call", "2024-06-01", "2024-07-01", strike_gte=100, strike_lte=200)
    fake_client.get_option_chain.assert_called_once_with("AAPL", "call", "2024-06-01", "2024-07-01", 100, 200)

    adapter.cancel_order("42")
    fake_client.cancel_order.assert_called_once_with("42")
