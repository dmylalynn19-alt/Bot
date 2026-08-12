"""Tests for core.risk_manager.RiskManager.validate_trade_setup."""

from __future__ import annotations

import pandas as pd
import pytest

from core.risk_manager import PortfolioState, RiskManager
from core.sr_strategy import TradeDirection, TradeSetup
from core.support_resistance import Level


def _setup(symbol: str = "AAPL", entry: float = 100.0, stop: float = 95.0, target: float = 110.0) -> TradeSetup:
    ts = pd.Timestamp("2024-01-15", tz="UTC")
    level = Level(price=95.0, kind="support", touches=2, first_touch=ts, last_touch=ts)
    return TradeSetup(
        symbol=symbol, direction=TradeDirection.LONG, level=level, entry_price=entry, stop_price=stop, target_price=target,
        confirmations={}, confidence=1.0, timestamp=ts, reasoning="test",
    )


def _portfolio(equity: float = 100_000.0, buying_power: float = 100_000.0, **kwargs) -> PortfolioState:
    return PortfolioState(equity=equity, cash=equity, buying_power=buying_power, **kwargs)


def _risk_manager(**overrides) -> RiskManager:
    params = dict(
        max_risk_per_trade=0.01, max_exposure=0.80, max_leverage=1.25, max_single_position=0.15, max_concurrent=5,
        max_daily_trades=20, daily_dd_reduce=0.02, daily_dd_halt=0.03, weekly_dd_reduce=0.05, weekly_dd_halt=0.07,
        max_dd_from_peak=0.10, min_position_dollars=100.0,
    )
    params.update(overrides)
    return RiskManager(**params)


def test_approves_and_sizes_by_risk_per_trade() -> None:
    # max_single_position raised well above what risk-sizing alone would
    # want here, to isolate the risk-based sizing math from the notional cap
    # (covered separately by test_shares_capped_by_max_single_position).
    rm = _risk_manager(max_risk_per_trade=0.01, max_single_position=0.90)
    setup = _setup(entry=100.0, stop=95.0)  # $5/share risk
    decision = rm.validate_trade_setup(setup, _portfolio(equity=100_000.0))

    assert decision.approved
    # risk_dollars = 100_000 * 0.01 = 1000; shares = 1000 / 5 = 200
    assert decision.shares == 200
    assert decision.modified_setup is setup


def test_shares_capped_by_max_single_position() -> None:
    rm = _risk_manager(max_risk_per_trade=0.05, max_single_position=0.10)
    # risk sizing alone would want: 100_000*0.05/5 = 1000 shares = $100k notional,
    # but max_single_position caps notional at 10% of equity = $10k -> 100 shares.
    setup = _setup(entry=100.0, stop=95.0)
    decision = rm.validate_trade_setup(setup, _portfolio(equity=100_000.0))

    assert decision.approved
    assert decision.shares == 100
    assert any("capped" in m for m in decision.modifications)


def test_rejects_invalid_stop_price() -> None:
    rm = _risk_manager()
    setup = _setup(entry=100.0, stop=100.0)  # zero distance
    decision = rm.validate_trade_setup(setup, _portfolio())
    assert not decision.approved
    assert "stop" in decision.rejection_reason.lower()


def test_rejects_duplicate_within_window() -> None:
    rm = _risk_manager(duplicate_window_seconds=60.0)
    now = pd.Timestamp("2024-01-15T10:00:00", tz="UTC")
    setup = _setup()
    first = rm.validate_trade_setup(setup, _portfolio(), now=now)
    assert first.approved

    second = rm.validate_trade_setup(setup, _portfolio(), now=now + pd.Timedelta(seconds=10))
    assert not second.approved
    assert "duplicate" in second.rejection_reason.lower()


def test_rejects_at_max_daily_trades() -> None:
    rm = _risk_manager(max_daily_trades=1)
    decision = rm.validate_trade_setup(_setup(), _portfolio(trades_today=1))
    assert not decision.approved
    assert "daily trades" in decision.rejection_reason.lower()


def test_rejects_at_max_concurrent_positions() -> None:
    rm = _risk_manager(max_concurrent=1)
    from broker.position_tracker import Position

    existing = {"MSFT": Position(symbol="MSFT", quantity=10, avg_entry_price=300, current_price=310, unrealized_pnl=100)}
    decision = rm.validate_trade_setup(_setup(symbol="AAPL"), _portfolio(positions=existing))
    assert not decision.approved
    assert "concurrent" in decision.rejection_reason.lower()


def test_allows_adding_to_an_existing_position_at_the_concurrent_cap() -> None:
    """max_concurrent shouldn't block a symbol that's already an existing position."""
    rm = _risk_manager(max_concurrent=1)
    from broker.position_tracker import Position

    existing = {"AAPL": Position(symbol="AAPL", quantity=10, avg_entry_price=100, current_price=101, unrealized_pnl=10)}
    decision = rm.validate_trade_setup(_setup(symbol="AAPL"), _portfolio(positions=existing))
    assert decision.approved


def test_rejects_below_minimum_position_dollars() -> None:
    rm = _risk_manager(max_risk_per_trade=0.0001, min_position_dollars=500.0)
    setup = _setup(entry=100.0, stop=95.0)
    decision = rm.validate_trade_setup(setup, _portfolio(equity=100_000.0))
    assert not decision.approved
    assert "minimum" in decision.rejection_reason.lower()


def test_rejects_when_notional_exceeds_buying_power() -> None:
    rm = _risk_manager(max_risk_per_trade=0.5, max_single_position=0.9)
    setup = _setup(entry=100.0, stop=99.0)  # tiny risk/share -> huge share count
    decision = rm.validate_trade_setup(setup, _portfolio(equity=1_000_000.0, buying_power=1000.0))
    assert not decision.approved
    assert "buying power" in decision.rejection_reason.lower()


def test_circuit_breaker_halt_rejects_outright(tmp_path) -> None:
    rm = _risk_manager(
        daily_dd_reduce=0.02, daily_dd_halt=0.03, trading_halt_lock_file=str(tmp_path / "halt.lock")
    )
    portfolio = _portfolio(equity=96_500.0, peak_equity=100_000.0, daily_pnl_pct=-0.04)  # breaches daily halt
    decision = rm.validate_trade_setup(_setup(), portfolio)
    assert not decision.approved
    assert "circuit breaker" in decision.rejection_reason.lower()
