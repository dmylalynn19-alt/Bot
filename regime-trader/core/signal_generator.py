"""Combines HMM regime detection, allocation strategy, and risk management
into final, risk-approved trading decisions.

Wires together, in order: HMMEngine (regime detection, forward-filtered only
- no look-ahead), StrategyOrchestrator (regime -> target allocation), and
RiskManager (absolute veto/resize over every signal - see
core.risk_manager's module docstring). The caller is responsible for
retraining `hmm_engine` and calling `strategy.update_regime_infos(...)`
periodically (see main.py) - this class only runs inference with whatever
model each is currently holding.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import PortfolioState, RiskDecision, RiskManager
from data.feature_engineering import build_feature_matrix

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates a risk-approved trading decision for one symbol.

    Args:
        hmm_engine: Fitted HMM regime detection engine for this symbol.
        strategy: Regime-to-allocation strategy orchestrator for this symbol.
        risk_manager: Risk manager enforcing position and portfolio limits
            (shared across symbols - it's portfolio-, not symbol-, scoped).
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: StrategyOrchestrator,
        risk_manager: RiskManager,
    ) -> None:
        self.hmm_engine = hmm_engine
        self.strategy = strategy
        self.risk_manager = risk_manager

    def generate(self, symbol: str, bars: pd.DataFrame, portfolio: PortfolioState, **risk_kwargs) -> RiskDecision:
        """Generate a single risk-approved decision for `symbol`.

        Args:
            symbol: Ticker to evaluate.
            bars: OHLCV history for `symbol`, most recent bar last - must
                include enough history for feature warm-up (see
                data.feature_engineering.build_feature_matrix) plus whatever
                `hmm_engine` was trained on.
            portfolio: Current portfolio/account state, for the risk checks.
            **risk_kwargs: Forwarded to RiskManager.validate_signal
                (tradeable, spread_pct, correlations, sector, is_overnight).
        """
        features = build_feature_matrix(bars).dropna()
        if features.empty:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=f"not enough history for {symbol} to compute features",
                modifications=[],
            )

        regime_state = self.hmm_engine.predict_regime_filtered(features)
        is_flickering = self.hmm_engine.is_flickering()

        # `portfolio` is shared across every symbol's call in a single run
        # (see main.py) - stamp it with *this* symbol's regime call so the
        # circuit breaker logs/audits against the right context, and so
        # RiskManager's flicker-based leverage check sees this symbol's
        # actual flicker state rather than a stale/previous symbol's.
        portfolio.current_regime_name = regime_state.label.value
        portfolio.current_regime_confidence = regime_state.probability
        portfolio.flicker_rate = float(is_flickering)

        signals = self.strategy.generate_signals([symbol], {symbol: bars}, regime_state, is_flickering)
        if not signals:
            logger.info(
                "No signal produced for %s (regime=%s, confidence=%.2f)", symbol, regime_state.label.value, regime_state.probability
            )
            return RiskDecision(approved=False, modified_signal=None, rejection_reason="strategy produced no signal", modifications=[])

        return self.risk_manager.validate_signal(signals[0], portfolio, **risk_kwargs)
