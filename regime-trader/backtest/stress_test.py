"""Crash injection and gap simulation for stress testing the strategy."""

from __future__ import annotations

import pandas as pd


class StressTester:
    """Injects synthetic market shocks into historical data for stress testing."""

    def __init__(self) -> None:
        raise NotImplementedError

    def inject_crash(self, data: pd.DataFrame, magnitude: float, duration_bars: int) -> pd.DataFrame:
        """Inject a synthetic crash of given magnitude and duration into the data."""
        raise NotImplementedError

    def inject_gap(self, data: pd.DataFrame, gap_pct: float) -> pd.DataFrame:
        """Inject a synthetic overnight/weekend gap into the data."""
        raise NotImplementedError

    def run_scenarios(self, data: pd.DataFrame, scenarios: list[dict]) -> dict:
        """Run a set of predefined stress scenarios and collect results."""
        raise NotImplementedError
