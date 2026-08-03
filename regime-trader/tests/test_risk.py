"""Tests for core.risk_manager.RiskManager."""

from __future__ import annotations


def test_size_position_respects_max_risk_per_trade() -> None:
    """size_position must not exceed max_risk_per_trade of equity."""
    raise NotImplementedError


def test_check_drawdown_halts_at_thresholds() -> None:
    """check_drawdown must return HALT when daily/weekly drawdown limits are breached."""
    raise NotImplementedError


def test_can_open_position_respects_limits() -> None:
    """can_open_position must respect max_concurrent and max_daily_trades."""
    raise NotImplementedError
