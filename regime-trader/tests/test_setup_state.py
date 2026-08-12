"""Tests for broker.setup_state.TradeSetupStore."""

from __future__ import annotations

import pandas as pd
import pytest

from broker.setup_state import TradeSetupStore
from core.sr_strategy import TradeDirection, TradeSetup
from core.support_resistance import Level


def _setup(symbol: str = "AAPL") -> TradeSetup:
    ts = pd.Timestamp("2024-01-15")
    level = Level(price=94.81, kind="support", touches=3, first_touch=ts - pd.Timedelta(days=30), last_touch=ts, source="pivot")
    return TradeSetup(
        symbol=symbol, direction=TradeDirection.LONG, level=level, entry_price=95.05, stop_price=92.85, target_price=99.45,
        confirmations={"volume": True, "vwap": True, "rsi": True, "macd": True}, confidence=1.0, timestamp=ts,
        reasoning="long at support 94.81 (3 touches): 4/4 confirmations",
    )


def test_load_returns_none_when_nothing_saved(tmp_path) -> None:
    store = TradeSetupStore(str(tmp_path / "setups.json"))
    assert store.load("AAPL") is None


def test_save_then_load_round_trips_the_setup(tmp_path) -> None:
    store = TradeSetupStore(str(tmp_path / "setups.json"))
    original = _setup()
    store.save(original)

    loaded = store.load("AAPL")
    assert loaded is not None
    assert loaded.symbol == "AAPL"
    assert loaded.direction == TradeDirection.LONG
    assert loaded.entry_price == pytest.approx(95.05)
    assert loaded.stop_price == pytest.approx(92.85)
    assert loaded.target_price == pytest.approx(99.45)
    assert loaded.level.price == pytest.approx(94.81)
    assert loaded.level.kind == "support"
    assert loaded.level.source == "pivot"


def test_save_persists_across_new_store_instances(tmp_path) -> None:
    path = str(tmp_path / "setups.json")
    TradeSetupStore(path).save(_setup("MSFT"))

    reopened = TradeSetupStore(path)
    loaded = reopened.load("MSFT")
    assert loaded is not None
    assert loaded.symbol == "MSFT"


def test_multiple_symbols_coexist(tmp_path) -> None:
    store = TradeSetupStore(str(tmp_path / "setups.json"))
    store.save(_setup("AAPL"))
    store.save(_setup("MSFT"))

    assert store.load("AAPL") is not None
    assert store.load("MSFT") is not None
    assert store.load("TSLA") is None


def test_save_overwrites_prior_entry_for_same_symbol(tmp_path) -> None:
    store = TradeSetupStore(str(tmp_path / "setups.json"))
    store.save(_setup("AAPL"))

    updated = _setup("AAPL")
    updated.stop_price = 88.0
    store.save(updated)

    loaded = store.load("AAPL")
    assert loaded.stop_price == pytest.approx(88.0)


def test_clear_removes_the_entry(tmp_path) -> None:
    store = TradeSetupStore(str(tmp_path / "setups.json"))
    store.save(_setup("AAPL"))
    store.clear("AAPL")
    assert store.load("AAPL") is None


def test_clear_on_unknown_symbol_is_a_noop(tmp_path) -> None:
    store = TradeSetupStore(str(tmp_path / "setups.json"))
    store.clear("AAPL")  # no file exists yet at all
    assert store.load("AAPL") is None
