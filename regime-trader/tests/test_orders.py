"""Tests for broker.order_executor.OrderExecutor."""

from __future__ import annotations


def test_execute_signal_submits_order() -> None:
    """execute_signal should submit an order matching the signal's direction and size."""
    raise NotImplementedError


def test_cancel_order() -> None:
    """cancel_order should cancel a pending order and report success."""
    raise NotImplementedError


def test_get_order_status() -> None:
    """get_order_status should reflect the broker's reported order state."""
    raise NotImplementedError
