"""Persists core.sr_strategy.TradeSetup state across process runs, keyed by
symbol.

Needed because should_exit() needs the ORIGINAL setup a position was opened
from (its level, stop, target) - unlike the regime strategy (which only
ever needs the latest target allocation) or the options strategy (whose
stop/target are simple percentages of premium, recomputable from the
position alone), the support/resistance and breakout strategies' stop/
target are anchored to a specific price level computed at entry time.
Neither Alpaca nor Schwab's API remembers *why* a position was opened, only
that it exists - so that has to be tracked locally, the same reason
broker/position_tracker.py already persists the equity peak locally.

core.breakout_strategy.BreakoutStrategy produces the same TradeSetup type
(see its module docstring), so this store works for either strategy - keep
each on its own state_file if running both against overlapping symbols.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.sr_strategy import TradeDirection, TradeSetup
from core.support_resistance import Level


class TradeSetupStore:
    """JSON-file-backed store of open TradeSetups, one per symbol.

    Args:
        state_file: Path to the JSON file. Created on first save; missing
            entirely is treated as "no open setups yet".
    """

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file)

    def save(self, setup: TradeSetup) -> None:
        """Record `setup` as the open position's origin for its symbol,
        overwriting any prior entry for that symbol."""
        data = self._read_all()
        data[setup.symbol] = {
            "direction": setup.direction.value,
            "level_price": setup.level.price,
            "level_kind": setup.level.kind,
            "level_touches": setup.level.touches,
            "level_source": getattr(setup.level, "source", "pivot"),
            "entry_price": setup.entry_price,
            "stop_price": setup.stop_price,
            "target_price": setup.target_price,
            "timestamp": str(setup.timestamp),
        }
        self._write_all(data)

    def load(self, symbol: str) -> TradeSetup | None:
        """Reconstruct the TradeSetup last saved for `symbol`, or None if
        there isn't one. The reconstructed setup has empty `confirmations`
        and confidence=1.0 (that context isn't needed for should_exit,
        which only reads direction/stop_price/target_price) - it's meant
        for should_exit(), not for re-evaluating entry confirmations."""
        raw = self._read_all().get(symbol)
        if raw is None:
            return None
        ts = pd.Timestamp(raw["timestamp"])
        level = Level(
            price=raw["level_price"], kind=raw["level_kind"], touches=raw["level_touches"],
            first_touch=ts, last_touch=ts, source=raw.get("level_source", "pivot"),
        )
        return TradeSetup(
            symbol=symbol, direction=TradeDirection(raw["direction"]), level=level,
            entry_price=raw["entry_price"], stop_price=raw["stop_price"], target_price=raw["target_price"],
            confirmations={}, confidence=1.0, timestamp=ts, reasoning="restored from state file",
        )

    def clear(self, symbol: str) -> None:
        """Remove `symbol`'s saved setup (call once its position is fully closed)."""
        data = self._read_all()
        if data.pop(symbol, None) is not None:
            self._write_all(data)

    def _read_all(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write_all(self, data: dict) -> None:
        self._path.write_text(json.dumps(data))
