"""Track open positions and P&L."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from broker.alpaca_client import AlpacaClient

_DEFAULT_STATE_FILE = "peak_equity.json"


@dataclass
class Position:
    """A single open position and its P&L state."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float


def _to_position(raw: dict) -> Position:
    return Position(
        symbol=raw["symbol"],
        quantity=float(raw["qty"]),
        avg_entry_price=float(raw["avg_entry_price"]),
        current_price=float(raw["current_price"]),
        unrealized_pnl=float(raw["unrealized_pl"]),
    )


class PositionTracker:
    """Tracks open positions and portfolio-level P&L.

    Args:
        client: Configured Alpaca client used to fetch position/account state.
        state_file: Where the observed all-time equity peak is persisted
            across runs (Alpaca's API doesn't track this for you) - used by
            get_drawdown_from_peak. Feed the same value into
            core.risk_manager.PortfolioState.peak_equity for the circuit
            breaker to see the same number.
    """

    def __init__(self, client: AlpacaClient, state_file: str = _DEFAULT_STATE_FILE) -> None:
        self.client = client
        self._state_path = Path(state_file)

    def get_positions(self) -> list[Position]:
        """Fetch all currently open positions."""
        return [_to_position(raw) for raw in self.client.list_positions()]

    def get_position(self, symbol: str) -> Position | None:
        """Fetch the open position for a single symbol, if any."""
        raw = self.client.get_position(symbol)
        return _to_position(raw) if raw is not None else None

    def get_total_exposure(self) -> float:
        """Compute total portfolio exposure as a fraction of equity."""
        account = self.client.get_account()
        equity = float(account["equity"])
        if equity <= 0:
            return 0.0
        notional = sum(abs(float(raw["market_value"])) for raw in self.client.list_positions())
        return notional / equity

    def get_daily_pnl(self) -> float:
        """Compute today's P&L as a fraction of yesterday's closing equity.

        Uses Alpaca's account.last_equity (equity as of the last market
        close), which already includes realized + unrealized P&L.
        """
        account = self.client.get_account()
        equity = float(account["equity"])
        last_equity = float(account["last_equity"])
        if last_equity <= 0:
            return 0.0
        return equity / last_equity - 1.0

    def get_peak_equity(self) -> float:
        """Return the observed all-time equity peak, updating/persisting it
        to `state_file` if current equity is a new high.

        Alpaca's API only reports current and yesterday's equity, not a
        running peak, so this is tracked locally across runs.
        """
        equity = float(self.client.get_account()["equity"])
        stored_peak = json.loads(self._state_path.read_text())["peak_equity"] if self._state_path.exists() else 0.0
        peak = max(equity, stored_peak)
        if peak > stored_peak:
            self._state_path.write_text(json.dumps({"peak_equity": peak}))
        return peak

    def get_drawdown_from_peak(self) -> float:
        """Compute the current drawdown from the observed all-time equity peak."""
        peak = self.get_peak_equity()
        if peak <= 0:
            return 0.0
        equity = float(self.client.get_account()["equity"])
        return equity / peak - 1.0
