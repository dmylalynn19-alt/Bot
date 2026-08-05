"""Tests for broker.schwab_client.SchwabClient.

Mocks the underlying schwab-py `Client` entirely (no live network access to
Schwab's servers, and none should ever be needed for these tests) - these
verify SchwabClient's own logic (bar resampling, chain filtering/mapping,
order-id extraction, order-builder dispatch), not schwab-py itself or a
live account.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

import broker.schwab_client as sc


def _fake_response(payload, status_code: int = 200, headers: dict | None = None) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    response.is_error = status_code >= 400
    return response


@pytest.fixture
def fake_client() -> MagicMock:
    client = MagicMock()
    client.get_account_numbers.return_value = _fake_response([{"accountNumber": "123", "hashValue": "HASH1"}])
    return client


@pytest.fixture
def schwab_client(fake_client) -> sc.SchwabClient:
    with patch.object(sc, "client_from_token_file", return_value=fake_client), patch("pathlib.Path.exists", return_value=True):
        return sc.SchwabClient("key", "secret", "https://callback", "/tmp/fake_token.json")


def test_init_resolves_account_hash_from_token_file(schwab_client, fake_client) -> None:
    assert schwab_client.account_hash == "HASH1"
    fake_client.get_account_numbers.assert_called_once()


def test_init_raises_with_no_linked_accounts(fake_client) -> None:
    fake_client.get_account_numbers.return_value = _fake_response([])
    with patch.object(sc, "client_from_token_file", return_value=fake_client), patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(RuntimeError):
            sc.SchwabClient("key", "secret", "https://callback", "/tmp/fake_token.json")


def test_get_account_returns_raw_json(schwab_client, fake_client) -> None:
    payload = {"securitiesAccount": {"currentBalances": {"liquidationValue": 100_000}, "positions": []}}
    fake_client.get_account.return_value = _fake_response(payload)
    assert schwab_client.get_account() == payload


def test_is_market_open_true_when_any_equity_product_open(schwab_client, fake_client) -> None:
    fake_client.get_market_hours.return_value = _fake_response({"equity": {"EQ": {"isOpen": True}}})
    assert schwab_client.is_market_open() is True


def test_is_market_open_false_when_closed(schwab_client, fake_client) -> None:
    fake_client.get_market_hours.return_value = _fake_response({"equity": {"EQ": {"isOpen": False}}})
    assert schwab_client.is_market_open() is False


def test_get_bars_daily_parses_candles_into_a_dataframe(schwab_client, fake_client) -> None:
    candles = [
        {"datetime": 1704067200000, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000},
        {"datetime": 1704153600000, "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1200},
    ]
    fake_client.get_price_history.return_value = _fake_response({"candles": candles})
    df = schwab_client.get_bars("AAPL", "1Day", "2024-01-01", "2024-01-05")

    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 101.5


def test_get_bars_empty_candles_returns_empty_dataframe_with_expected_columns(schwab_client, fake_client) -> None:
    fake_client.get_price_history.return_value = _fake_response({"candles": []})
    df = schwab_client.get_bars("AAPL", "1Day", "2024-01-01", "2024-01-05")
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_get_bars_1hour_resamples_from_thirty_minute_candles(schwab_client, fake_client) -> None:
    base_ts = 1704110400000
    candles = [
        {"datetime": base_ts, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 500},
        {"datetime": base_ts + 30 * 60 * 1000, "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 500},
    ]
    fake_client.get_price_history.return_value = _fake_response({"candles": candles})
    df = schwab_client.get_bars("AAPL", "1Hour", "2024-01-01", "2024-01-02")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["open"] == 100  # first
    assert row["high"] == 102  # max
    assert row["low"] == 99  # min
    assert row["close"] == 101.5  # last
    assert row["volume"] == 1000  # sum


def test_get_bars_rejects_unsupported_timeframe(schwab_client) -> None:
    with pytest.raises(ValueError):
        schwab_client.get_bars("AAPL", "3Min", "2024-01-01", "2024-01-05")


def test_get_option_chain_filters_by_strike_and_maps_fields(schwab_client, fake_client) -> None:
    payload = {
        "callExpDateMap": {
            "2024-06-21:30": {
                "150.0": [
                    {
                        "symbol": "AAPL  240621C00150000", "bid": 5.0, "ask": 5.2, "delta": 0.5,
                        "gamma": 0.02, "theta": -0.05, "vega": 0.1, "volatility": 25.0, "expirationDate": "2024-06-21",
                    }
                ],
                "160.0": [
                    {
                        "symbol": "AAPL  240621C00160000", "bid": 2.0, "ask": 2.2, "delta": 0.3,
                        "gamma": 0.01, "theta": -0.03, "vega": 0.08, "volatility": 24.0, "expirationDate": "2024-06-21",
                    }
                ],
            }
        }
    }
    fake_client.get_option_chain.return_value = _fake_response(payload)
    chain = schwab_client.get_option_chain("AAPL", "call", "2024-06-01", "2024-07-01", strike_gte=155)

    assert len(chain) == 1
    contract = chain[0]
    assert contract["occ_symbol"] == "AAPL  240621C00160000"
    assert contract["strike"] == 160.0
    assert contract["mid"] == pytest.approx(2.1)
    assert contract["delta"] == 0.3


def test_get_option_chain_rejects_bad_contract_type(schwab_client) -> None:
    with pytest.raises(ValueError):
        schwab_client.get_option_chain("AAPL", "straddle", "2024-06-01", "2024-07-01")


def test_submit_order_equity_market_extracts_order_id_from_location_header(schwab_client, fake_client) -> None:
    fake_client.place_order.return_value = _fake_response(
        {}, status_code=201, headers={"Location": "https://api.schwabapi.com/trader/v1/accounts/HASH1/orders/998877"}
    )
    result = schwab_client.submit_order("AAPL", 10, "buy", asset_class="equity", order_type="market")
    assert result == {"order_id": 998877, "status_code": 201}
    fake_client.place_order.assert_called_once()
    assert fake_client.place_order.call_args[0][0] == "HASH1"


def test_submit_order_option_buy_to_open_limit_requires_price(schwab_client) -> None:
    with pytest.raises(ValueError):
        schwab_client.submit_order("AAPL  240621C00160000", 1, "buy_to_open", asset_class="option", order_type="limit")


def test_submit_order_option_buy_to_open_limit_dispatches_correctly(schwab_client, fake_client) -> None:
    fake_client.place_order.return_value = _fake_response(
        {}, status_code=201, headers={"Location": "https://api.schwabapi.com/trader/v1/accounts/HASH1/orders/555"}
    )
    result = schwab_client.submit_order(
        "AAPL  240621C00160000", 2, "buy_to_open", asset_class="option", order_type="limit", limit_price=2.10
    )
    assert result["order_id"] == 555


def test_submit_order_rejects_bad_asset_class(schwab_client) -> None:
    with pytest.raises(ValueError):
        schwab_client.submit_order("AAPL", 1, "buy", asset_class="futures")


def test_submit_order_rejects_unsupported_side_order_type_combo(schwab_client) -> None:
    with pytest.raises(ValueError):
        schwab_client.submit_order("AAPL", 1, "short_sell_special", asset_class="equity")


def test_list_orders_rejects_bad_status(schwab_client) -> None:
    with pytest.raises(ValueError):
        schwab_client.list_orders(status="pending")


def test_get_position_finds_matching_symbol(schwab_client, fake_client) -> None:
    payload = {
        "securitiesAccount": {
            "positions": [
                {"instrument": {"symbol": "AAPL"}, "longQuantity": 10},
                {"instrument": {"symbol": "MSFT"}, "longQuantity": 5},
            ]
        }
    }
    fake_client.get_account.return_value = _fake_response(payload)
    position = schwab_client.get_position("MSFT")
    assert position is not None
    assert position["longQuantity"] == 5
    assert schwab_client.get_position("TSLA") is None
