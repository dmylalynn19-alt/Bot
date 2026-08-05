"""Tests for core.support_resistance (pivot detection, clustering, zone lookup)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.support_resistance import Level, SupportResistanceDetector


def _oscillating_bars(n: int = 90, low: float = 95.0, high: float = 115.0, seed: int = 1) -> pd.DataFrame:
    """A triangle wave between `low` and `high`, touching each extreme
    several times - enough for the detector to build multi-touch zones."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    period = 20
    mid = (low + high) / 2
    amplitude = (high - low) / 2
    close = mid + amplitude * (1 - 4 * np.abs(((t / period) % 1) - 0.5)) + rng.normal(0, 0.1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1_000_000.0}, index=idx
    )


def test_detect_finds_support_and_resistance_zones() -> None:
    bars = _oscillating_bars()
    detector = SupportResistanceDetector(pivot_window=5, cluster_tolerance_pct=0.01, min_touches=2)
    levels = detector.detect(bars)

    assert any(level.kind == "support" for level in levels)
    assert any(level.kind == "resistance" for level in levels)
    for level in levels:
        assert level.touches >= 2


def test_detect_drops_zones_below_min_touches() -> None:
    bars = _oscillating_bars()
    lenient = SupportResistanceDetector(pivot_window=5, cluster_tolerance_pct=0.01, min_touches=2).detect(bars)
    strict = SupportResistanceDetector(pivot_window=5, cluster_tolerance_pct=0.01, min_touches=50).detect(bars)
    assert len(strict) <= len(lenient)
    assert strict == []


def test_detect_requires_minimum_bar_count() -> None:
    bars = _oscillating_bars(n=5)
    detector = SupportResistanceDetector(pivot_window=5)
    assert detector.detect(bars) == []


def test_nearest_support_and_resistance_pick_closest_bracketing_levels() -> None:
    ts = pd.Timestamp("2024-01-01")
    levels = [
        Level(price=90.0, kind="support", touches=3, first_touch=ts, last_touch=ts),
        Level(price=95.0, kind="support", touches=2, first_touch=ts, last_touch=ts),
        Level(price=110.0, kind="resistance", touches=2, first_touch=ts, last_touch=ts),
        Level(price=120.0, kind="resistance", touches=4, first_touch=ts, last_touch=ts),
    ]
    detector = SupportResistanceDetector()

    # At price 100: nearest support below is 95 (not 90), nearest resistance
    # above is 110 (not 120).
    assert detector.nearest_support(levels, 100.0).price == 95.0
    assert detector.nearest_resistance(levels, 100.0).price == 110.0

    # At price exactly on a level, that level counts as both its own nearest
    # (support if kind==support, resistance lookup requires price<=level).
    assert detector.nearest_support(levels, 95.0).price == 95.0

    # Below every support / above every resistance -> None.
    assert detector.nearest_support(levels, 80.0) is None
    assert detector.nearest_resistance(levels, 130.0) is None
