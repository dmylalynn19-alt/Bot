"""Tests for core.regime_strategies.RegimeStrategy."""

from __future__ import annotations


def test_target_allocation_by_regime() -> None:
    """target_allocation should return the configured allocation for each regime."""
    raise NotImplementedError


def test_needs_rebalance_threshold() -> None:
    """needs_rebalance should trigger only when drift exceeds rebalance_threshold."""
    raise NotImplementedError


def test_apply_confidence_discount() -> None:
    """apply_confidence_discount should scale allocation down below min_confidence."""
    raise NotImplementedError
