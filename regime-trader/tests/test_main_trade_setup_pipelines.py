"""End-to-end tests for main.py's run_once_sr/run_once_breakout wiring
(build_sr_pipeline/build_breakout_pipeline, _run_trade_setup_strategy_pass,
RiskManager.validate_trade_setup, OrderExecutor.execute_trade_setup/
close_trade_setup_position, TradeSetupStore) - using a fake broker client,
never a real Alpaca/Schwab connection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import main
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from broker.setup_state import TradeSetupStore
from core.risk_manager import RiskManager
from core.sr_strategy import SupportResistanceStrategy


def _bounce_bars() -> pd.DataFrame:
    """Same confirmed-LONG-at-support scenario as
    tests/test_sr_strategy.py's _downtrend_then_mss_bullish_near_support_bars."""

    def zigzag(anchors, seg_len=6):
        segs = []
        for i in range(len(anchors) - 1):
            seg = np.linspace(anchors[i], anchors[i + 1], seg_len + 1)
            if i > 0:
                seg = seg[1:]
            segs.append(seg)
        return np.concatenate(segs)

    close = zigzag([103, 100, 102, 99.5, 101, 98, 103], seg_len=6)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    volume = np.full(len(close), 1_000_000.0)
    return pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05, "close": close, "volume": volume}, index=idx)


def _sr_strategy() -> SupportResistanceStrategy:
    return SupportResistanceStrategy(
        pivot_window=3, cluster_tolerance_pct=0.02, min_touches=1, level_proximity_pct=0.05,
        min_required_confirmations=0, use_key_levels=False, require_market_structure=True,
        structure_pivot_window=3, min_bars=20,
    )


class FakeBrokerClient:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars
        self.equity = 100_000.0
        self.positions: dict[str, dict] = {}
        self.submitted_orders: list[dict] = []
        self._next_order_id = 1

    def is_market_open(self) -> bool:
        return True

    def get_bars(self, symbol, timeframe, start, end) -> pd.DataFrame:
        return self._bars

    def get_account(self) -> dict:
        return {"equity": self.equity, "cash": self.equity, "buying_power": self.equity, "last_equity": self.equity}

    def list_orders(self, status="open", after=None) -> list[dict]:
        return []

    def list_positions(self) -> list[dict]:
        return [
            {"symbol": sym, "qty": p["qty"], "avg_entry_price": p["price"], "current_price": p["price"], "unrealized_pl": 0.0, "asset_class": "us_equity"}
            for sym, p in self.positions.items()
        ]

    def get_position(self, symbol: str) -> dict | None:
        p = self.positions.get(symbol)
        if p is None:
            return None
        return {"symbol": symbol, "qty": p["qty"], "avg_entry_price": p["price"], "current_price": p["price"], "unrealized_pl": 0.0, "asset_class": "us_equity"}

    def submit_order(self, **kwargs) -> dict:
        self.submitted_orders.append(kwargs)
        order_id = str(self._next_order_id)
        self._next_order_id += 1
        symbol, qty, side = kwargs["symbol"], kwargs["qty"], kwargs["side"]
        if side == "buy":
            self.positions[symbol] = {"qty": qty, "price": 100.0}
        elif side == "sell":
            self.positions.pop(symbol, None)
        return {"id": order_id, "symbol": symbol, "qty": qty, "side": side, "status": "filled"}


def _config(symbols: list[str]) -> dict:
    return {
        "broker": {"symbols": symbols, "timeframe": "1Day"},
        "risk": dict(
            max_risk_per_trade=0.01, max_exposure=0.80, max_leverage=1.25, max_single_position=0.20, max_concurrent=5,
            max_daily_trades=20, daily_dd_reduce=0.02, daily_dd_halt=0.03, weekly_dd_reduce=0.05, weekly_dd_halt=0.07,
            max_dd_from_peak=0.10, min_position_dollars=100.0,
        ),
    }


def _pipeline(client: FakeBrokerClient, state_file: str) -> dict:
    from data.market_data import MarketDataFeed

    return {
        "client": client,
        "feed": MarketDataFeed(client=client, symbols=["TEST"], timeframe="1Day"),
        "risk_manager": RiskManager(**_config(["TEST"])["risk"]),
        "order_executor": OrderExecutor(client),
        "position_tracker": PositionTracker(client),
        "strategy": _sr_strategy(),
        "setup_store": TradeSetupStore(state_file),
    }


def test_run_once_sr_opens_a_position_and_saves_the_setup(tmp_path) -> None:
    client = FakeBrokerClient(_bounce_bars())
    pipeline = _pipeline(client, str(tmp_path / "setups.json"))
    config = _config(["TEST"])

    results = main.run_once_sr(config, pipeline=pipeline)

    assert client.submitted_orders, "expected an order to be submitted"
    assert client.submitted_orders[0]["side"] == "buy"
    assert "TEST" in client.positions
    saved = pipeline["setup_store"].load("TEST")
    assert saved is not None
    assert saved.direction.value == "long"
    assert results["TEST"].approved


def test_run_once_sr_skips_market_closed() -> None:
    client = FakeBrokerClient(_bounce_bars())
    client.is_market_open = lambda: False
    pipeline = _pipeline(client, "unused.json")
    results = main.run_once_sr(_config(["TEST"]), pipeline=pipeline)
    assert results == {}
    assert client.submitted_orders == []


def test_run_once_sr_manual_mode_declines_without_confirmation(tmp_path, monkeypatch) -> None:
    client = FakeBrokerClient(_bounce_bars())
    pipeline = _pipeline(client, str(tmp_path / "setups.json"))
    monkeypatch.setattr("builtins.input", lambda _: "n")

    results = main.run_once_sr(_config(["TEST"]), pipeline=pipeline, mode="manual")

    assert client.submitted_orders == []
    assert results["TEST"] == "entry skipped (manual)"
    assert pipeline["setup_store"].load("TEST") is None


def test_run_once_sr_manual_mode_executes_on_confirmation(tmp_path, monkeypatch) -> None:
    client = FakeBrokerClient(_bounce_bars())
    pipeline = _pipeline(client, str(tmp_path / "setups.json"))
    monkeypatch.setattr("builtins.input", lambda _: "y")

    main.run_once_sr(_config(["TEST"]), pipeline=pipeline, mode="manual")

    assert len(client.submitted_orders) == 1


def test_run_once_sr_closes_position_on_stop_hit(tmp_path) -> None:
    client = FakeBrokerClient(_bounce_bars())
    state_file = str(tmp_path / "setups.json")
    pipeline = _pipeline(client, state_file)
    config = _config(["TEST"])

    main.run_once_sr(config, pipeline=pipeline)  # opens the position
    setup = pipeline["setup_store"].load("TEST")
    assert setup is not None

    # Extend bars so the last close is well below the stop.
    bars = client._bars.copy()
    below_stop = pd.DataFrame(
        {"open": [setup.stop_price - 1], "high": [setup.stop_price - 0.5], "low": [setup.stop_price - 2], "close": [setup.stop_price - 1], "volume": [1_000_000.0]},
        index=[bars.index[-1] + pd.tseries.offsets.BDay(1)],
    )
    client._bars = pd.concat([bars, below_stop])

    results = main.run_once_sr(config, pipeline=pipeline)

    assert "TEST" not in client.positions
    assert pipeline["setup_store"].load("TEST") is None
    assert "closed" in results["TEST"]
    assert any(o["side"] == "sell" for o in client.submitted_orders)


def test_run_once_sr_leaves_position_alone_without_a_saved_setup(tmp_path) -> None:
    """A position with no TradeSetupStore entry (e.g. state file lost, or a
    pre-existing position) shouldn't be touched."""
    client = FakeBrokerClient(_bounce_bars())
    client.positions["TEST"] = {"qty": 10, "price": 100.0}
    pipeline = _pipeline(client, str(tmp_path / "setups.json"))

    results = main.run_once_sr(_config(["TEST"]), pipeline=pipeline)

    assert client.submitted_orders == []
    assert "TEST" not in results


def test_run_once_breakout_uses_the_same_shared_pass(tmp_path) -> None:
    """Just confirms build_breakout_pipeline/run_once_breakout are wired
    through _run_trade_setup_strategy_pass without error - the strategy's
    own setup-detection logic is covered in tests/test_breakout_strategy.py."""
    from broker.setup_state import TradeSetupStore
    from core.breakout_strategy import BreakoutStrategy

    client = FakeBrokerClient(pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []}))
    pipeline = {
        "client": client,
        "feed": __import__("data.market_data", fromlist=["MarketDataFeed"]).MarketDataFeed(client=client, symbols=["TEST"], timeframe="1Day"),
        "risk_manager": RiskManager(**_config(["TEST"])["risk"]),
        "order_executor": OrderExecutor(client),
        "position_tracker": PositionTracker(client),
        "strategy": BreakoutStrategy(min_bars=60),
        "setup_store": TradeSetupStore(str(tmp_path / "setups.json")),
    }

    results = main.run_once_breakout(_config(["TEST"]), pipeline=pipeline)
    assert results == {}  # empty bars -> no setup, no crash
    assert client.submitted_orders == []
