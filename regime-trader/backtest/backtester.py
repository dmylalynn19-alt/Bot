"""Walk-forward allocation backtester.

This is an ALLOCATION-BASED backtester, not a trade-by-trade one: it does
not track individual entries/exits. Each in-sample window trains a fresh
HMM (with BIC model selection); each out-of-sample bar gets a look-ahead-free
regime call (forward algorithm only, via HMMEngine.predict_regime_filtered)
which the strategy orchestrator turns into a target portfolio allocation.
The position is rebalanced toward that target only when it has drifted more
than the configured threshold (default 10%) from the current allocation -
this is how real systematic strategies operate, and it avoids churning on
every minor probability wobble.

Walk-forward windows:
    in-sample (IS):      252 bars - HMM training + BIC model selection
    out-of-sample (OOS):  126 bars - evaluation, one regime call per bar
    step:                 126 bars - how far the window advances each round

All three are measured in *feature* rows (i.e. bars with a complete,
non-NaN feature vector - see data.feature_engineering.build_feature_matrix),
not raw price bars: the feature warm-up itself (252-bar rolling z-score,
200-bar SMA, ...) consumes about a year of history before the first row is
usable, so `data` passed to `run()` needs roughly 2 years of lookback before
`start` for the very first window to have enough in-sample rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager
from data.feature_engineering import build_feature_matrix

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    regime_history: pd.DataFrame


class WalkForwardBacktester:
    """Runs a walk-forward backtest of the regime-based allocation strategy.

    Args:
        hmm_engine: HMM engine retrained on each window's in-sample data.
        strategy: Strategy orchestrator; its regime->strategy mapping is
            rebuilt from scratch after every retrain via `update_regime_infos`.
        risk_manager: Risk manager to apply position/exposure limits.
            Not yet wired into position sizing - core/risk_manager.py is
            still a stub - so the strategy's raw target allocation is used
            as-is for now.
        step_size: How many out-of-sample bars the window advances each round.
        in_sample_bars: In-sample window length, in clean feature rows.
        out_of_sample_bars: Out-of-sample window length, in clean feature rows.
        symbol: Label for the single instrument this backtest trades.
        initial_cash: Starting cash balance.
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: StrategyOrchestrator,
        risk_manager: RiskManager,
        step_size: int = 126,
        in_sample_bars: int = 252,
        out_of_sample_bars: int = 126,
        symbol: str = "SYMBOL",
        initial_cash: float = 100_000.0,
    ) -> None:
        self.hmm_engine = hmm_engine
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.step_size = step_size
        self.in_sample_bars = in_sample_bars
        self.out_of_sample_bars = out_of_sample_bars
        self.symbol = symbol
        self.initial_cash = initial_cash

        self._cash = initial_cash
        self._shares = 0
        self._current_allocation = 0.0
        self._equity_records: list[dict] = []
        self._trade_records: list[dict] = []
        self._regime_records: list[dict] = []

    def run(self, data: pd.DataFrame, start: str, end: str) -> BacktestResult:
        """Run the walk-forward backtest over `data` from `start` to `end`.

        Args:
            data: Single-symbol OHLCV bars, sorted or not, covering at least
                ~2 years before `start` (in-sample window + feature warm-up)
                through `end`.
            start: First bar the out-of-sample evaluation may cover.
            end: Last bar to evaluate.
        """
        data = data.sort_index()
        full_features = build_feature_matrix(data)
        clean_index = full_features.dropna().index

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        first_oos_pos = clean_index.searchsorted(start_ts)
        window_start_pos = max(0, first_oos_pos - self.in_sample_bars)

        if window_start_pos + self.in_sample_bars >= len(clean_index):
            raise ValueError(
                f"Not enough history before '{start}' for a {self.in_sample_bars}-bar in-sample "
                f"window (only {len(clean_index) - window_start_pos} clean feature rows available)"
            )

        self._reset_portfolio_state()

        n = len(clean_index)
        while window_start_pos + self.in_sample_bars < n:
            is_end_pos = window_start_pos + self.in_sample_bars
            oos_end_pos = min(is_end_pos + self.out_of_sample_bars, n)

            is_index = clean_index[window_start_pos:is_end_pos]
            oos_index = clean_index[is_end_pos:oos_end_pos]
            oos_index = oos_index[oos_index <= end_ts]
            if oos_index.empty:
                break

            self.step(data, full_features, is_index, oos_index)

            if oos_index[-1] >= end_ts:
                break
            window_start_pos += self.step_size

        equity_curve = pd.Series(
            [r["equity"] for r in self._equity_records],
            index=pd.Index([r["timestamp"] for r in self._equity_records], name="timestamp"),
            name="equity",
        )
        trades = pd.DataFrame(self._trade_records)
        regime_history = pd.DataFrame(self._regime_records)
        return BacktestResult(equity_curve=equity_curve, trades=trades, regime_history=regime_history)

    def step(
        self,
        data: pd.DataFrame,
        full_features: pd.DataFrame,
        is_index: pd.Index,
        oos_index: pd.Index,
    ) -> None:
        """Process a single walk-forward window.

        a. Train the HMM on this window's in-sample features (BIC model selection).
        b. Rebuild the strategy's vol-rank -> strategy mapping from the freshly
           trained model's regime metadata.
        c. Walk out-of-sample bar by bar: a look-ahead-free filtered regime call
           (forward algorithm only) feeds the strategy, which returns a target
           allocation; rebalance only when it has drifted >10% from current.
        d. Record the regime call and mark-to-market equity at every OOS bar.
        e. Record a "trade" whenever a rebalance actually happens.
        """
        is_features = full_features.loc[is_index]
        self.hmm_engine.fit(is_features)
        self.strategy.update_regime_infos(self.hmm_engine.regime_info_)

        logger.info(
            "Walk-forward window: IS %s -> %s (%d bars, n_components=%d), OOS %s -> %s (%d bars)",
            is_index[0],
            is_index[-1],
            len(is_index),
            self.hmm_engine.n_components,
            oos_index[0],
            oos_index[-1],
            len(oos_index),
        )

        # Prime the forward filter over the in-sample window itself so the
        # first out-of-sample call starts from a real filtered belief and a
        # warmed-up stability tracker, rather than a flat prior.
        for ts in is_index:
            self.hmm_engine.predict_regime_filtered(full_features.loc[is_index[0] : ts].dropna())

        for ts in oos_index:
            features_up_to_now = full_features.loc[is_index[0] : ts].dropna()
            regime_state = self.hmm_engine.predict_regime_filtered(features_up_to_now)
            is_flickering = self.hmm_engine.is_flickering()

            price = float(data.loc[ts, "close"])
            bars_so_far = data.loc[:ts]

            signals = self.strategy.generate_signals(
                [self.symbol], {self.symbol: bars_so_far}, regime_state, is_flickering
            )
            signal = signals[0] if signals else None
            # NOTE: leverage above 1.0x is deployed notional exposure funded on
            # margin, not a separate scaling of shares - so the allocation the
            # strategy actually wants is position_size_pct * leverage.
            target_allocation = signal.position_size_pct * signal.leverage if signal is not None else 0.0

            if self.strategy.needs_rebalance(self._current_allocation, target_allocation):
                self._rebalance(target_allocation, price, ts)

            equity = self._cash + self._shares * price  # mark to market
            self._equity_records.append({"timestamp": ts, "equity": equity})
            self._regime_records.append(
                {
                    "timestamp": ts,
                    "regime_id": regime_state.state_id,
                    "regime_name": regime_state.label.value,
                    "probability": regime_state.probability,
                    "is_confirmed": regime_state.is_confirmed,
                    "consecutive_bars": regime_state.consecutive_bars,
                    "is_flickering": is_flickering,
                    "target_allocation": target_allocation,
                    "current_allocation": self._current_allocation,
                }
            )

    def _rebalance(self, target_allocation: float, price: float, timestamp: pd.Timestamp) -> None:
        # ALLOCATION MATH - must be exactly this:
        equity = self._cash + self._shares * price
        target_shares = int(equity * target_allocation / price)
        delta = target_shares - self._shares
        self._cash -= delta * price

        allocation_before = self._current_allocation
        self._shares = target_shares
        self._current_allocation = target_allocation

        self._trade_records.append(
            {
                "timestamp": timestamp,
                "price": price,
                "shares_delta": delta,
                "shares_after": self._shares,
                "allocation_before": allocation_before,
                "allocation_after": target_allocation,
                "cash_after": self._cash,
            }
        )

    def _reset_portfolio_state(self) -> None:
        self._cash = self.initial_cash
        self._shares = 0
        self._current_allocation = 0.0
        self._equity_records = []
        self._trade_records = []
        self._regime_records = []
