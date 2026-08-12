"""Tests for broker.order_executor.OrderExecutor.

Uses a fake broker client (satisfying broker.base.BrokerClient's shape)
rather than a real AlpacaClient/SchwabAdapter - OrderExecutor's job is
translating signals into the right submit_order call and normalizing the
response, not talking to any particular broker.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from broker.order_executor import OrderExecutor, OrderStatus
from core.options_strategy import OptionRight, OptionSignal
from core.regime_strategies import Direction, RegimeLabel, Signal
from core.sr_strategy import TradeDirection, TradeSetup
from core.support_resistance import Level


class FakeBrokerClient:
    """Records every submit_order call and returns a canned response."""

    def __init__(self) -> None:
        self.submitted_orders: list[dict] = []
        self.canceled_order_ids: list[str] = []
        self.next_order_id = 1
        self.raise_on_cancel = False

    def submit_order(self, **kwargs) -> dict:
        self.submitted_orders.append(kwargs)
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        return {"id": order_id, "symbol": kwargs["symbol"], "qty": kwargs["qty"], "side": kwargs["side"], "status": "accepted"}

    def cancel_order(self, order_id: str) -> None:
        if self.raise_on_cancel:
            raise RuntimeError("cancel failed")
        self.canceled_order_ids.append(order_id)

    def get_order(self, order_id: str) -> dict:
        return {"id": order_id, "symbol": "AAPL", "qty": 10, "side": "buy", "status": "filled"}

    def replace_order(self, order_id: str, **kwargs) -> dict:
        return {"id": order_id, "symbol": "AAPL", "qty": kwargs.get("qty", 10), "side": "buy", "status": "accepted"}


def _stock_signal(symbol: str = "AAPL", entry_price: float = 100.0, position_size_pct: float = 0.5, leverage: float = 1.0) -> Signal:
    return Signal(
        symbol=symbol, direction=Direction.LONG, confidence=0.8, entry_price=entry_price, stop_loss=90.0, take_profit=120.0,
        position_size_pct=position_size_pct, leverage=leverage, regime_id=0, regime_name=RegimeLabel.NEUTRAL,
        regime_probability=0.8, timestamp=pd.Timestamp("2024-01-01", tz=timezone.utc), reasoning="test",
        strategy_name="test_strategy",
    )


def _option_signal(contracts: int = 2, limit_price: float = 3.5) -> OptionSignal:
    return OptionSignal(
        symbol="AAPL", occ_symbol="AAPL  240621C00160000", right=OptionRight.CALL, strike=160.0, expiration="2024-06-21",
        contracts=contracts, limit_price=limit_price, stop_loss_pct=0.5, take_profit_pct=1.0, confidence=0.8,
        timestamp=pd.Timestamp("2024-01-01", tz=timezone.utc), reasoning="test",
    )


def _trade_setup(symbol: str = "AAPL", direction: TradeDirection = TradeDirection.LONG) -> TradeSetup:
    ts = pd.Timestamp("2024-01-15", tz=timezone.utc)
    level = Level(price=95.0, kind="support" if direction == TradeDirection.LONG else "resistance", touches=2, first_touch=ts, last_touch=ts)
    return TradeSetup(
        symbol=symbol, direction=direction, level=level, entry_price=100.0, stop_price=95.0, target_price=110.0,
        confirmations={}, confidence=1.0, timestamp=ts, reasoning="test",
    )


def test_execute_trade_setup_submits_a_buy_market_order() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    order = executor.execute_trade_setup(_trade_setup(), shares=50)

    assert order is not None
    call = client.submitted_orders[0]
    assert call["symbol"] == "AAPL"
    assert call["side"] == "buy"
    assert call["asset_class"] == "equity"
    assert call["order_type"] == "market"
    assert call["qty"] == 50


def test_execute_trade_setup_returns_none_for_zero_shares() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    assert executor.execute_trade_setup(_trade_setup(), shares=0) is None
    assert client.submitted_orders == []


def test_execute_trade_setup_refuses_short_setups() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    order = executor.execute_trade_setup(_trade_setup(direction=TradeDirection.SHORT), shares=50)
    assert order is None
    assert client.submitted_orders == []


def test_close_trade_setup_position_submits_a_sell_market_order() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    order = executor.close_trade_setup_position(_trade_setup(), shares=50, reason="target hit")

    assert order is not None
    call = client.submitted_orders[0]
    assert call["side"] == "sell"
    assert call["asset_class"] == "equity"
    assert call["order_type"] == "market"


def test_close_trade_setup_position_returns_none_for_short_setups() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    order = executor.close_trade_setup_position(_trade_setup(direction=TradeDirection.SHORT), shares=50)
    assert order is None


def test_close_trade_setup_position_returns_none_for_zero_shares() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    assert executor.close_trade_setup_position(_trade_setup(), shares=0) is None


def test_execute_signal_submits_a_market_order_for_the_delta() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    order = executor.execute_signal(_stock_signal(entry_price=100.0, position_size_pct=0.5, leverage=1.0), current_quantity=0, equity=100_000)

    assert order is not None
    assert len(client.submitted_orders) == 1
    call = client.submitted_orders[0]
    assert call["symbol"] == "AAPL"
    assert call["side"] == "buy"
    assert call["asset_class"] == "equity"
    assert call["order_type"] == "market"
    assert call["qty"] == 500  # 100_000 * 0.5 / 100.0
    assert order.status == OrderStatus.PENDING  # "accepted" maps to pending


def test_execute_signal_sells_when_target_is_below_current() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    executor.execute_signal(_stock_signal(entry_price=100.0, position_size_pct=0.1, leverage=1.0), current_quantity=500, equity=100_000)

    assert client.submitted_orders[0]["side"] == "sell"


def test_execute_signal_returns_none_when_no_rebalance_needed() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    order = executor.execute_signal(_stock_signal(entry_price=100.0, position_size_pct=0.5, leverage=1.0), current_quantity=500, equity=100_000)

    assert order is None
    assert client.submitted_orders == []


def test_execute_option_signal_submits_buy_to_open_limit_order() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    order = executor.execute_option_signal(_option_signal(contracts=2, limit_price=3.5))

    assert order is not None
    call = client.submitted_orders[0]
    assert call["symbol"] == "AAPL  240621C00160000"
    assert call["side"] == "buy_to_open"
    assert call["asset_class"] == "option"
    assert call["order_type"] == "limit"
    assert call["limit_price"] == 3.5
    assert call["qty"] == 2


def test_execute_option_signal_returns_none_for_zero_contracts() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    assert executor.execute_option_signal(_option_signal(contracts=0)) is None
    assert client.submitted_orders == []


def test_close_option_position_submits_sell_to_close_market_order() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)

    order = executor.close_option_position("AAPL  240621C00160000", 2, reason="stop loss hit")

    assert order is not None
    call = client.submitted_orders[0]
    assert call["side"] == "sell_to_close"
    assert call["asset_class"] == "option"
    assert call["order_type"] == "market"


def test_close_option_position_returns_none_for_zero_contracts() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    assert executor.close_option_position("AAPL  240621C00160000", 0) is None


def test_cancel_order_returns_true_on_success() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    assert executor.cancel_order("42") is True
    assert client.canceled_order_ids == ["42"]


def test_cancel_order_returns_false_on_failure() -> None:
    client = FakeBrokerClient()
    client.raise_on_cancel = True
    executor = OrderExecutor(client)
    assert executor.cancel_order("42") is False


def test_get_order_status_reflects_broker_state() -> None:
    client = FakeBrokerClient()
    executor = OrderExecutor(client)
    assert executor.get_order_status("1") == OrderStatus.FILLED
