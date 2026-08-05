"""Turns a directional indicator signal into a specific, sized options trade.

Directional-only (buying calls or puts, never selling/writing) - the
simplest and most common way to trade a directional view with defined risk:
the most you can lose is the premium paid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any

import pandas as pd

from core.indicator_signals import Direction, IndicatorSignal


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionSignal:
    """A specific, sized options trade to place."""

    symbol: str  # underlying ticker
    occ_symbol: str  # specific contract, e.g. AAPL260117C00150000
    right: OptionRight
    strike: float
    expiration: str
    contracts: int
    limit_price: float  # per-contract premium used for sizing/the limit order
    stop_loss_pct: float  # exit if premium drops this fraction from entry
    take_profit_pct: float  # exit if premium rises this fraction from entry
    confidence: float
    timestamp: pd.Timestamp
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OptionsStrategy:
    """Selects a specific option contract and position size from a
    directional indicator signal.

    Args:
        target_delta: Preferred absolute delta for the selected contract
            (e.g. 0.35 - a common balance of leverage vs win-rate for
            directional plays). The contract closest to this delta, among
            those with a live quote in the expiration window, is chosen.
        min_days_to_expiration / max_days_to_expiration: Expiration window to
            search (avoids very-near-term contracts, where theta decay
            dominates, and very-far-term ones, where leverage is diluted).
        max_risk_per_trade: Max fraction of equity risked on a single trade -
            the entire premium paid, since buying options has defined risk
            (you can't lose more than you paid).
        stop_loss_pct / take_profit_pct: Exit thresholds as a fraction of
            premium paid (0.50 = exit at -50%, 1.00 = exit at +100%).
        min_confidence: Minimum IndicatorSignal.confidence required to act.
    """

    def __init__(
        self,
        target_delta: float = 0.35,
        min_days_to_expiration: int = 21,
        max_days_to_expiration: int = 45,
        max_risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.50,
        take_profit_pct: float = 1.00,
        min_confidence: float = 0.75,
    ) -> None:
        self.target_delta = target_delta
        self.min_days_to_expiration = min_days_to_expiration
        self.max_days_to_expiration = max_days_to_expiration
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence

    def expiration_window(self, today: date | None = None) -> tuple[date, date]:
        """Return (gte, lte) expiration dates to search, from `today`."""
        today = today or date.today()
        return today + timedelta(days=self.min_days_to_expiration), today + timedelta(days=self.max_days_to_expiration)

    def select_contract(self, chain: list[dict], right: OptionRight) -> dict | None:
        """Pick the contract from `chain` (see AlpacaClient.get_option_chain)
        whose delta is closest to `target_delta` (sign-aware: puts have
        negative delta), among contracts with a live bid/ask quote."""
        target = self.target_delta if right == OptionRight.CALL else -self.target_delta
        candidates = [c for c in chain if c.get("delta") is not None and c.get("bid") and c.get("ask")]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c["delta"] - target))

    def generate(self, indicator_signal: IndicatorSignal, chain: list[dict], equity: float) -> OptionSignal | None:
        """Turn a directional signal + live option chain into a sized trade.

        Returns None if the signal is neutral, below `min_confidence`, no
        suitable contract is found in the chain, or the sized position would
        round to zero contracts.
        """
        if indicator_signal.direction == Direction.NEUTRAL:
            return None
        if indicator_signal.confidence < self.min_confidence:
            return None

        right = OptionRight.CALL if indicator_signal.direction == Direction.BULLISH else OptionRight.PUT
        contract = self.select_contract(chain, right)
        if contract is None:
            return None

        premium = contract["mid"] if contract.get("mid") else (contract["bid"] + contract["ask"]) / 2
        if not premium or premium <= 0:
            return None

        max_premium_budget = equity * self.max_risk_per_trade
        contracts = int(max_premium_budget / (premium * 100))
        if contracts < 1:
            return None

        return OptionSignal(
            symbol=indicator_signal.symbol,
            occ_symbol=contract["occ_symbol"],
            right=right,
            strike=contract["strike"],
            expiration=contract["expiration"],
            contracts=contracts,
            limit_price=premium,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            confidence=indicator_signal.confidence,
            timestamp=indicator_signal.timestamp,
            reasoning=(
                f"{indicator_signal.reasoning}; selected {right.value} @ {contract['strike']} "
                f"exp {contract['expiration']} (delta={contract['delta']:.2f})"
            ),
            metadata={"delta": contract["delta"], "premium_budget": max_premium_budget},
        )
