"""Tests for main.py's `close` command (main.run_close) - the emergency
manual flatten-a-position-now path."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import main


class FakeCloseClient:
    def __init__(self, positions: dict[str, dict]) -> None:
        self._positions = positions
        self.submitted_orders: list[dict] = []

    def get_position(self, symbol: str) -> dict | None:
        return self._positions.get(symbol)

    def list_positions(self) -> list[dict]:
        return list(self._positions.values())

    def submit_order(self, **kwargs) -> dict:
        self.submitted_orders.append(kwargs)
        return {"id": "1", "symbol": kwargs["symbol"], "qty": kwargs["qty"], "side": kwargs["side"], "status": "accepted"}


def _config(tmp_path) -> dict:
    return {
        "sr_state_file": str(tmp_path / "sr.json"),
        "breakout_state_file": str(tmp_path / "breakout.json"),
    }


def test_run_close_by_symbol_submits_a_closing_order(tmp_path, capsys) -> None:
    client = FakeCloseClient({"AAPL": {"symbol": "AAPL", "qty": 10, "asset_class": "us_equity"}})
    with patch("main.build_broker_client", return_value=client):
        main.run_close(main.parse_args(["close", "--symbol", "AAPL"]), _config(tmp_path))

    assert len(client.submitted_orders) == 1
    assert client.submitted_orders[0]["side"] == "sell"
    assert "submitted sell order" in capsys.readouterr().out


def test_run_close_by_symbol_with_no_position_reports_nothing_to_close(tmp_path, capsys) -> None:
    client = FakeCloseClient({})
    with patch("main.build_broker_client", return_value=client):
        main.run_close(main.parse_args(["close", "--symbol", "AAPL"]), _config(tmp_path))

    assert client.submitted_orders == []
    assert "nothing to close" in capsys.readouterr().out


def test_run_close_all_closes_every_open_position(tmp_path) -> None:
    client = FakeCloseClient(
        {
            "AAPL": {"symbol": "AAPL", "qty": 10, "asset_class": "us_equity"},
            "MSFT": {"symbol": "MSFT", "qty": -5, "asset_class": "us_equity"},
        }
    )
    with patch("main.build_broker_client", return_value=client):
        main.run_close(main.parse_args(["close", "--all"]), _config(tmp_path))

    assert len(client.submitted_orders) == 2
    symbols_closed = {o["symbol"] for o in client.submitted_orders}
    assert symbols_closed == {"AAPL", "MSFT"}


def test_run_close_all_with_no_positions_reports_none(tmp_path, capsys) -> None:
    client = FakeCloseClient({})
    with patch("main.build_broker_client", return_value=client):
        main.run_close(main.parse_args(["close", "--all"]), _config(tmp_path))

    assert client.submitted_orders == []
    assert "No open positions" in capsys.readouterr().out


def test_run_close_clears_saved_trade_setup_state(tmp_path) -> None:
    from broker.setup_state import TradeSetupStore
    from core.sr_strategy import TradeDirection, TradeSetup
    from core.support_resistance import Level
    import pandas as pd

    config = _config(tmp_path)
    sr_store = TradeSetupStore(config["sr_state_file"])
    ts = pd.Timestamp("2024-01-01")
    level = Level(price=95.0, kind="support", touches=2, first_touch=ts, last_touch=ts)
    sr_store.save(
        TradeSetup(
            symbol="AAPL", direction=TradeDirection.LONG, level=level, entry_price=100.0, stop_price=95.0,
            target_price=110.0, confirmations={}, confidence=1.0, timestamp=ts, reasoning="test",
        )
    )
    assert sr_store.load("AAPL") is not None

    client = FakeCloseClient({"AAPL": {"symbol": "AAPL", "qty": 10, "asset_class": "us_equity"}})
    with patch("main.build_broker_client", return_value=client):
        main.run_close(main.parse_args(["close", "--symbol", "AAPL"]), config)

    assert TradeSetupStore(config["sr_state_file"]).load("AAPL") is None
