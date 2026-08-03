"""Crash injection, gap simulation, and regime-misclassification stress testing.

Three scenarios, each answering a different "what if" about the strategy in
backtest.backtester.WalkForwardBacktester:

a. Crash injection - what if the market gaps down hard at random points?
   100 Monte Carlo runs, each with 10 random single-day -5%/-15% gaps
   inserted into the price series, then a full backtest re-run. Reports the
   distribution of max drawdown and how often a hypothetical circuit breaker
   at `circuit_breaker_drawdown` would have fired.
b. Gap risk - do overnight gaps (2-5x ATR) cost the strategy about what its
   currently-held allocation would predict, or more?
c. Regime misclassification - if the HMM's regime CALL were wrong (labels
   shuffled to a random alternative every bar), does the strategy's built-in
   allocation/leverage bounds contain the damage, or does it blow up? Bounded
   damage means risk management isn't relying on the regime call being
   correct; a blowup means it is.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.backtester import WalkForwardBacktester
from backtest.performance import PerformanceAnalyzer
from core.hmm_engine import HMMEngine, RegimeState

logger = logging.getLogger(__name__)

_OHLC_COLUMNS = ["open", "high", "low", "close"]


class _RegimeShufflingEngine:
    """Wraps an HMMEngine; every filtered regime call gets its label/state_id
    replaced with a uniformly random alternative from the fitted model's own
    regimes. Everything else (fit, is_flickering, regime_info_, ...) is
    forwarded unchanged to the real engine. Used by run_regime_shuffle_test.
    """

    def __init__(self, engine: HMMEngine, rng: np.random.Generator) -> None:
        self._engine = engine
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def predict_regime_filtered(self, features_up_to_now: pd.DataFrame) -> RegimeState:
        state = self._engine.predict_regime_filtered(features_up_to_now)
        candidates = list(self._engine.state_labels_.items())
        shuffled_state_id, shuffled_label = candidates[self._rng.integers(len(candidates))]
        return replace(state, label=shuffled_label, state_id=shuffled_state_id)


class StressTester:
    """Injects synthetic market shocks and re-runs `backtester` end to end.

    Args:
        backtester: A configured WalkForwardBacktester. Its `hmm_engine` is
            retrained from scratch on each simulation (or temporarily wrapped,
            for the regime-shuffle test), so a single instance can be reused
            across many Monte Carlo runs.
        circuit_breaker_drawdown: Drawdown level (fraction, e.g. 0.10) treated
            as "the circuit breaker would have fired" when reporting the
            crash-test results - matches config/settings.yaml's
            risk.max_dd_from_peak. core/risk_manager.py doesn't exist yet, so
            this is a standalone threshold check, not a call into it.
        shuffle_damage_tolerance: In the regime-shuffle test, the shuffled run
            is considered "contained" if its max drawdown is no worse than
            this multiple of the baseline (unshuffled) run's max drawdown.
    """

    def __init__(
        self,
        backtester: WalkForwardBacktester,
        circuit_breaker_drawdown: float = 0.10,
        shuffle_damage_tolerance: float = 3.0,
    ) -> None:
        self.backtester = backtester
        self.circuit_breaker_drawdown = circuit_breaker_drawdown
        self.shuffle_damage_tolerance = shuffle_damage_tolerance
        self._analyzer = PerformanceAnalyzer(risk_free_rate=0.0)

    # ------------------------------------------------------------------
    # Shock injection (pure data transforms)
    # ------------------------------------------------------------------

    def inject_crash(self, data: pd.DataFrame, magnitude: float, at: pd.Timestamp) -> pd.DataFrame:
        """Permanent single-day gap of `magnitude` (e.g. -0.10 = -10%) at bar `at`:
        every OHLC value from `at` onward is scaled by (1 + magnitude), as if the
        market gapped down that much at the open and never gapped back."""
        shocked = data.copy()
        pos = shocked.index.get_loc(at)
        cols = shocked.columns.get_indexer(_OHLC_COLUMNS)
        shocked.iloc[pos:, cols] = shocked.iloc[pos:, cols] * (1.0 + magnitude)
        return shocked

    def inject_gap(self, data: pd.DataFrame, gap_multiple: float, at: pd.Timestamp, atr_window: int = 14) -> pd.DataFrame:
        """Permanent level shift like `inject_crash`, but sized as `gap_multiple`
        times the ATR just before `at` rather than a fixed percentage. Positive
        `gap_multiple` gaps up, negative gaps down."""
        import ta

        shocked = data.copy()
        pos = shocked.index.get_loc(at)
        ref_pos = max(pos - 1, 0)
        atr = ta.volatility.AverageTrueRange(
            shocked["high"], shocked["low"], shocked["close"], window=atr_window, fillna=False
        ).average_true_range()
        atr_at = float(atr.iloc[ref_pos])
        price_at = float(shocked["close"].iloc[ref_pos])
        if not np.isfinite(atr_at) or price_at == 0:
            return shocked
        factor = 1.0 + (gap_multiple * atr_at) / price_at
        cols = shocked.columns.get_indexer(_OHLC_COLUMNS)
        shocked.iloc[pos:, cols] = shocked.iloc[pos:, cols] * factor
        return shocked

    # ------------------------------------------------------------------
    # a. Crash injection - Monte Carlo
    # ------------------------------------------------------------------

    def run_crash_test(
        self,
        data: pd.DataFrame,
        start: str,
        end: str,
        n_simulations: int = 100,
        n_shocks: int = 10,
        magnitude_range: tuple[float, float] = (-0.15, -0.05),
        seed: int = 0,
    ) -> dict:
        """Monte Carlo crash injection: `n_simulations` runs, each with `n_shocks`
        random single-day gaps (magnitude ~ Uniform(*magnitude_range)) inserted
        into `data` before a full backtest re-run.

        Warning: this reruns the full walk-forward backtest (including HMM
        retraining) `n_simulations` times - expensive for large HMMEngine
        configs (many n_candidates/n_init) or long date ranges.
        """
        rng = np.random.default_rng(seed)
        eligible_dates = data.loc[start:end].index
        max_losses = []
        circuit_breaker_fired = 0

        for sim in range(n_simulations):
            shock_dates = rng.choice(eligible_dates, size=min(n_shocks, len(eligible_dates)), replace=False)
            shocked = data
            for shock_date in shock_dates:
                magnitude = float(rng.uniform(*magnitude_range))
                shocked = self.inject_crash(shocked, magnitude, pd.Timestamp(shock_date))

            result = self.backtester.run(shocked, start=start, end=end)
            max_dd = self._analyzer.max_drawdown(result.equity_curve)
            max_losses.append(max_dd)
            if abs(max_dd) >= self.circuit_breaker_drawdown:
                circuit_breaker_fired += 1

        max_losses_arr = np.array(max_losses)
        logger.info(
            "Crash test: %d sims, mean max loss=%.2f%%, worst=%.2f%%, circuit breaker fired %d/%d",
            n_simulations,
            float(max_losses_arr.mean()) * 100,
            float(max_losses_arr.min()) * 100,
            circuit_breaker_fired,
            n_simulations,
        )
        return {
            "n_simulations": n_simulations,
            "mean_max_loss": float(max_losses_arr.mean()),
            "worst_case_loss": float(max_losses_arr.min()),
            "circuit_breaker_fired_pct": circuit_breaker_fired / n_simulations,
        }

    # ------------------------------------------------------------------
    # b. Gap risk - expected vs. actual
    # ------------------------------------------------------------------

    def run_gap_test(
        self,
        data: pd.DataFrame,
        start: str,
        end: str,
        n_simulations: int = 100,
        n_gaps: int = 10,
        gap_multiple_range: tuple[float, float] = (2.0, 5.0),
        seed: int = 0,
    ) -> dict:
        """Insert `n_gaps` random overnight gaps (2-5x ATR, sign random) per
        simulation and compare the allocation-implied "expected" loss on each
        gap bar (current_allocation * raw gap %) against the strategy's
        actually realized equity change on that bar. A large, consistent gap
        between expected and actual reveals that fills/leverage/rebalance
        timing are amplifying gap risk beyond plain allocation exposure.
        """
        rng = np.random.default_rng(seed)
        eligible_dates = data.loc[start:end].index
        expected_losses: list[float] = []
        actual_losses: list[float] = []

        for sim in range(n_simulations):
            shocked = data
            gap_dates = rng.choice(eligible_dates, size=min(n_gaps, len(eligible_dates)), replace=False)
            gap_dates = [pd.Timestamp(d) for d in gap_dates]
            for gap_date in gap_dates:
                gap_multiple = float(rng.uniform(*gap_multiple_range)) * float(rng.choice([-1.0, 1.0]))
                shocked = self.inject_gap(shocked, gap_multiple, gap_date)

            result = self.backtester.run(shocked, start=start, end=end)
            equity = result.equity_curve
            alloc_by_ts = (
                result.regime_history.set_index("timestamp")["current_allocation"]
                if not result.regime_history.empty
                else pd.Series(dtype=float)
            )

            for gap_date in gap_dates:
                if gap_date not in equity.index or gap_date not in shocked.index:
                    continue
                eq_pos = equity.index.get_loc(gap_date)
                price_pos = shocked.index.get_loc(gap_date)
                if eq_pos == 0 or price_pos == 0:
                    continue

                price_change_pct = shocked["open"].iloc[price_pos] / data["close"].iloc[price_pos - 1] - 1.0
                allocation = float(alloc_by_ts.get(gap_date, 0.0))
                expected_losses.append(allocation * price_change_pct)
                actual_losses.append(float(equity.iloc[eq_pos] / equity.iloc[eq_pos - 1] - 1.0))

        return {
            "n_gap_events": len(actual_losses),
            "mean_expected_loss": float(np.mean(expected_losses)) if expected_losses else float("nan"),
            "mean_actual_loss": float(np.mean(actual_losses)) if actual_losses else float("nan"),
        }

    # ------------------------------------------------------------------
    # c. Regime misclassification
    # ------------------------------------------------------------------

    def run_regime_shuffle_test(self, data: pd.DataFrame, start: str, end: str, seed: int = 0) -> dict:
        """Run the backtest once normally, then again with every filtered
        regime call's label/state_id replaced by a uniformly random
        alternative (see _RegimeShufflingEngine) - i.e. the strategy always
        acts on the WRONG regime. Compares max drawdown between the two runs:
        `risk_contained` is True if the shuffled run's drawdown is no worse
        than `shuffle_damage_tolerance` times the baseline's. If it blows up
        far beyond that, the strategy's risk management isn't independent of
        the regime call being right.
        """
        baseline_result = self.backtester.run(data, start=start, end=end)
        baseline_dd = self._analyzer.max_drawdown(baseline_result.equity_curve)

        rng = np.random.default_rng(seed)
        original_engine = self.backtester.hmm_engine
        self.backtester.hmm_engine = _RegimeShufflingEngine(original_engine, rng)
        try:
            shuffled_result = self.backtester.run(data, start=start, end=end)
        finally:
            self.backtester.hmm_engine = original_engine

        shuffled_dd = self._analyzer.max_drawdown(shuffled_result.equity_curve)
        risk_contained = abs(shuffled_dd) <= abs(baseline_dd) * self.shuffle_damage_tolerance

        logger.info(
            "Regime shuffle test: baseline max_dd=%.2f%%, shuffled max_dd=%.2f%%, risk_contained=%s",
            baseline_dd * 100,
            shuffled_dd * 100,
            risk_contained,
        )
        return {
            "baseline_max_drawdown": baseline_dd,
            "shuffled_max_drawdown": shuffled_dd,
            "risk_contained": risk_contained,
        }

    # ------------------------------------------------------------------
    # Convenience: run everything
    # ------------------------------------------------------------------

    def run_scenarios(self, data: pd.DataFrame, start: str, end: str, **kwargs) -> dict:
        """Run all three stress scenarios (crash, gap, regime-shuffle) and
        return a combined report keyed by scenario name."""
        return {
            "crash": self.run_crash_test(data, start, end, **kwargs.get("crash", {})),
            "gap": self.run_gap_test(data, start, end, **kwargs.get("gap", {})),
            "regime_shuffle": self.run_regime_shuffle_test(data, start, end, **kwargs.get("regime_shuffle", {})),
        }
