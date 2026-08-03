"""Sharpe, drawdown, regime breakdown, and benchmark comparisons.

Consumes the outputs of backtest.backtester.WalkForwardBacktester
(BacktestResult: equity_curve, trades, regime_history) plus the raw OHLCV
bars used for the backtest, and produces every metric/table/CSV described in
the backtester spec: headline risk/return stats, a regime-by-regime
breakdown, a confidence-bucketed breakdown, three benchmark comparisons
(buy-and-hold, 200-SMA trend, random entry with the same risk management),
worst-case statistics, a rich terminal report, and CSV exports.

A "trade" throughout is one rebalance: the P&L between one rebalance and the
next (or the end of the equity curve), since this is an allocation-based
backtester with no individual entries/exits - see backtest/backtester.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CONFIDENCE_BINS = [0.0, 0.50, 0.60, 0.70, 1.0001]
_CONFIDENCE_LABELS = ["<50%", "50-60%", "60-70%", "70%+"]


def _max_consecutive_true(mask: pd.Series) -> int:
    """Longest run of consecutive True values in a boolean Series."""
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _fmt(value: object, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(value)
    if pct:
        return f"{value:.1%}"
    return f"{value:.4f}" if abs(value) < 10 else f"{value:,.2f}"


class PerformanceAnalyzer:
    """Computes performance metrics from a backtest equity curve.

    Args:
        risk_free_rate: Annualized risk-free rate used for Sharpe/Sortino calculations.
        trading_days_per_year: Annualization factor for Sharpe/Sortino/CAGR.
    """

    def __init__(self, risk_free_rate: float, trading_days_per_year: int = 252) -> None:
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

    # ------------------------------------------------------------------
    # Core return/risk metrics
    # ------------------------------------------------------------------

    def returns(self, equity_curve: pd.Series) -> pd.Series:
        """Bar-over-bar percentage returns of the equity curve."""
        return equity_curve.pct_change().dropna()

    def total_return(self, equity_curve: pd.Series) -> float:
        """Total percentage return over the whole equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    def cagr(self, equity_curve: pd.Series) -> float:
        """Compound annual growth rate."""
        if len(equity_curve) < 2:
            return 0.0
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
        if growth <= 0:
            return -1.0
        return float(growth ** (1 / years) - 1.0)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio of a bar-level returns series."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess = returns - daily_rf
        return float(excess.mean() / returns.std() * np.sqrt(self.trading_days_per_year))

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation only) of a returns series."""
        if len(returns) < 2:
            return 0.0
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess = returns - daily_rf
        downside = excess[excess < 0]
        downside_std = downside.std()
        if not downside_std or np.isnan(downside_std):
            return 0.0
        return float(excess.mean() / downside_std * np.sqrt(self.trading_days_per_year))

    def drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Percentage drawdown from the running peak, at every bar."""
        running_max = equity_curve.cummax()
        return equity_curve / running_max - 1.0

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Maximum (most negative) drawdown from an equity curve."""
        if equity_curve.empty:
            return 0.0
        return float(self.drawdown_series(equity_curve).min())

    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Longest streak of consecutive bars spent underwater (in bars/trading days)."""
        if equity_curve.empty:
            return 0
        underwater = self.drawdown_series(equity_curve) < 0
        return _max_consecutive_true(underwater)

    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """CAGR / |max drawdown|."""
        mdd = abs(self.max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        return self.cagr(equity_curve) / mdd

    # ------------------------------------------------------------------
    # Trade stats (a "trade" = the P&L held between two consecutive rebalances)
    # ------------------------------------------------------------------

    def trade_pnls(self, equity_curve: pd.Series, trades: pd.DataFrame) -> pd.Series:
        """Equity change from each rebalance to the next (or to the end of the curve)."""
        if trades.empty or equity_curve.empty:
            return pd.Series(dtype=float)
        boundaries = list(trades["timestamp"]) + [equity_curve.index[-1]]
        pnls = [
            float(equity_curve.loc[boundaries[i + 1]] - equity_curve.loc[boundaries[i]])
            for i in range(len(trades))
        ]
        return pd.Series(pnls, index=pd.Index(trades["timestamp"], name="timestamp"))

    def win_rate(self, trade_pnls: pd.Series) -> float:
        """Fraction of trades with positive P&L."""
        if trade_pnls.empty:
            return 0.0
        return float((trade_pnls > 0).mean())

    def avg_win_loss(self, trade_pnls: pd.Series) -> tuple[float, float]:
        """(average winning trade P&L, average losing trade P&L)."""
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        return (
            float(wins.mean()) if not wins.empty else 0.0,
            float(losses.mean()) if not losses.empty else 0.0,
        )

    def profit_factor(self, trade_pnls: pd.Series) -> float:
        """Gross profit / |gross loss|. Infinite if there are no losing trades."""
        wins = trade_pnls[trade_pnls > 0].sum()
        losses = trade_pnls[trade_pnls < 0].sum()
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return float(wins / abs(losses))

    def avg_holding_period(self, trades: pd.DataFrame, equity_curve: pd.Series) -> float:
        """Average bars held between consecutive rebalances."""
        if trades.empty or equity_curve.empty:
            return 0.0
        boundaries = list(trades["timestamp"]) + [equity_curve.index[-1]]
        positions = equity_curve.index.get_indexer(pd.Index(boundaries))
        holding_periods = np.diff(positions)
        return float(holding_periods.mean()) if len(holding_periods) else 0.0

    # ------------------------------------------------------------------
    # Regime / confidence breakdowns
    # ------------------------------------------------------------------

    def regime_breakdown(self, equity_curve: pd.Series, regime_history: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Per-regime table: % time in, return contribution, avg trade P&L, win rate, Sharpe."""
        if regime_history.empty or equity_curve.empty:
            return pd.DataFrame(columns=["regime", "pct_time_in", "return_contribution", "avg_trade_pnl", "win_rate", "sharpe"])

        returns = self.returns(equity_curve)
        regime_by_ts = regime_history.set_index("timestamp")["regime_name"]
        bar_regime = regime_by_ts.reindex(returns.index).ffill()

        pnls = self.trade_pnls(equity_curve, trades)
        trade_regime = regime_by_ts.reindex(trades["timestamp"]) if not trades.empty else pd.Series(dtype=object)

        rows = []
        for regime in sorted(bar_regime.dropna().unique()):
            bar_mask = bar_regime == regime
            regime_returns = returns[bar_mask]
            trade_mask = trade_regime.to_numpy() == regime if len(trade_regime) else np.array([], dtype=bool)
            regime_pnls = pnls[trade_mask] if trade_mask.any() else pd.Series(dtype=float)
            rows.append(
                {
                    "regime": regime,
                    "pct_time_in": float(bar_mask.mean()),
                    "return_contribution": float(regime_returns.sum()),
                    "avg_trade_pnl": float(regime_pnls.mean()) if not regime_pnls.empty else float("nan"),
                    "win_rate": self.win_rate(regime_pnls) if not regime_pnls.empty else float("nan"),
                    "sharpe": self.sharpe_ratio(regime_returns) if len(regime_returns) > 1 else float("nan"),
                }
            )
        return pd.DataFrame(rows).sort_values("pct_time_in", ascending=False).reset_index(drop=True)

    def confidence_breakdown(self, equity_curve: pd.Series, regime_history: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Bucketed by regime-call confidence (<50%, 50-60%, 60-70%, 70%+): trades, Sharpe, win rate, avg P&L.

        Shows whether high-confidence regime calls actually outperform
        low-confidence ones - if they do, the HMM is adding value.
        """
        columns = ["confidence_bucket", "trades", "sharpe", "win_rate", "avg_pnl"]
        if regime_history.empty:
            return pd.DataFrame(columns=columns)

        returns = self.returns(equity_curve)
        prob_by_ts = regime_history.set_index("timestamp")["probability"]
        bar_bucket = pd.cut(prob_by_ts.reindex(returns.index).ffill(), bins=_CONFIDENCE_BINS, labels=_CONFIDENCE_LABELS, right=False)

        pnls = self.trade_pnls(equity_curve, trades)
        if not trades.empty:
            trade_confidence = trades["timestamp"].map(prob_by_ts)
            trade_bucket = pd.cut(trade_confidence, bins=_CONFIDENCE_BINS, labels=_CONFIDENCE_LABELS, right=False)
        else:
            trade_bucket = pd.Series(dtype=object)

        rows = []
        for label in _CONFIDENCE_LABELS:
            bucket_returns = returns[bar_bucket == label]
            bucket_pnls = pnls[trade_bucket.to_numpy() == label] if len(trade_bucket) else pd.Series(dtype=float)
            rows.append(
                {
                    "confidence_bucket": label,
                    "trades": int(len(bucket_pnls)),
                    "sharpe": self.sharpe_ratio(bucket_returns) if len(bucket_returns) > 1 else float("nan"),
                    "win_rate": self.win_rate(bucket_pnls) if not bucket_pnls.empty else float("nan"),
                    "avg_pnl": float(bucket_pnls.mean()) if not bucket_pnls.empty else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Worst-case statistics
    # ------------------------------------------------------------------

    def worst_case_stats(self, equity_curve: pd.Series) -> dict:
        """Worst day/week/month, max consecutive losing days, longest time underwater."""
        if equity_curve.empty:
            return {
                "worst_day": float("nan"),
                "worst_week": float("nan"),
                "worst_month": float("nan"),
                "max_consecutive_losing_days": 0,
                "longest_underwater_days": 0,
            }
        daily = self.returns(equity_curve)
        weekly = equity_curve.resample("W").last().pct_change().dropna()
        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        return {
            "worst_day": float(daily.min()) if not daily.empty else float("nan"),
            "worst_week": float(weekly.min()) if not weekly.empty else float("nan"),
            "worst_month": float(monthly.min()) if not monthly.empty else float("nan"),
            "max_consecutive_losing_days": _max_consecutive_true(daily < 0),
            "longest_underwater_days": self.max_drawdown_duration(equity_curve),
        }

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def buy_and_hold_curve(self, bars: pd.DataFrame, initial_cash: float) -> pd.Series:
        """Equity curve of buying the asset once at the first bar and holding."""
        shares = initial_cash / float(bars["close"].iloc[0])
        return (bars["close"] * shares).rename("buy_and_hold")

    def sma_trend_curve(self, bars: pd.DataFrame, initial_cash: float, window: int = 200) -> pd.Series:
        """Equity curve of a simple trend benchmark: long (100%) above the SMA, cash below."""
        sma = bars["close"].rolling(window).mean()
        cash = initial_cash
        shares = 0.0
        values = []
        for ts, price in bars["close"].items():
            want_in = bool(price > sma.loc[ts]) if not pd.isna(sma.loc[ts]) else False
            equity = cash + shares * price
            target_shares = (equity / price) if want_in else 0.0
            cash -= (target_shares - shares) * price
            shares = target_shares
            values.append(cash + shares * price)
        return pd.Series(values, index=bars.index, name=f"sma_{window}_trend")

    def random_entry_curves(
        self,
        bars: pd.DataFrame,
        initial_cash: float,
        allocation_choices: list[float],
        rebalance_dates: pd.DatetimeIndex,
        n_seeds: int = 100,
    ) -> pd.DataFrame:
        """`n_seeds` equity curves that rebalance to a uniformly random allocation
        (drawn from `allocation_choices`, the same choices the real strategy can
        produce) at exactly `rebalance_dates` - i.e. same frequency, same position
        sizing rules as the real strategy, but blind to the actual regime.
        """
        rebalance_set = set(rebalance_dates)
        curves: dict[str, pd.Series] = {}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            cash = initial_cash
            shares = 0.0
            values = []
            for ts, price in bars["close"].items():
                if ts in rebalance_set:
                    target_allocation = float(rng.choice(allocation_choices))
                    equity = cash + shares * price
                    target_shares = int(equity * target_allocation / price)
                    cash -= (target_shares - shares) * price
                    shares = target_shares
                values.append(cash + shares * price)
            curves[f"seed_{seed}"] = pd.Series(values, index=bars.index)
        return pd.DataFrame(curves)

    def _benchmark_row(self, name: str, curve: pd.Series) -> dict:
        returns = self.returns(curve)
        return {
            "strategy": name,
            "total_return": self.total_return(curve),
            "cagr": self.cagr(curve),
            "sharpe": self.sharpe_ratio(returns),
            "max_drawdown": self.max_drawdown(curve),
        }

    def compare_to_benchmark(self, equity_curve: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
        """Compare the strategy's equity curve to a single benchmark curve."""
        return pd.DataFrame([self._benchmark_row("Strategy", equity_curve), self._benchmark_row("Benchmark", benchmark)])

    def compare_to_benchmarks(
        self,
        equity_curve: pd.Series,
        bars: pd.DataFrame,
        initial_cash: float,
        trades: pd.DataFrame | None = None,
        n_random_seeds: int = 100,
    ) -> pd.DataFrame:
        """Full --compare table: strategy vs. buy-and-hold, 200-SMA trend, and
        random entry (same risk management, 100 seeds - mean and std reported)."""
        if equity_curve.empty:
            return pd.DataFrame()
        window_bars = bars.loc[equity_curve.index[0] : equity_curve.index[-1]]

        rows = [self._benchmark_row("Strategy", equity_curve)]
        rows.append(self._benchmark_row("Buy & Hold", self.buy_and_hold_curve(window_bars, initial_cash)))

        sma_curve = self.sma_trend_curve(bars, initial_cash).loc[equity_curve.index[0] : equity_curve.index[-1]]
        rows.append(self._benchmark_row("200 SMA Trend", sma_curve))

        trades = trades if trades is not None else pd.DataFrame()
        allocation_choices = sorted(trades["allocation_after"].unique().tolist()) if not trades.empty else [0.0, 0.60, 0.95]
        rebalance_dates = pd.DatetimeIndex(trades["timestamp"]) if not trades.empty else pd.DatetimeIndex([])
        random_curves = self.random_entry_curves(window_bars, initial_cash, allocation_choices, rebalance_dates, n_random_seeds)
        random_metrics = pd.DataFrame(
            [self._benchmark_row(col, random_curves[col]) for col in random_curves.columns]
        ).drop(columns="strategy")
        rows.append({"strategy": "Random Entry (mean)", **random_metrics.mean().to_dict()})
        rows.append({"strategy": "Random Entry (std)", **random_metrics.std().to_dict()})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary / reporting / export
    # ------------------------------------------------------------------

    def summary(self, equity_curve: pd.Series, trades: pd.DataFrame | None = None) -> dict:
        """Full headline performance summary."""
        trades = trades if trades is not None else pd.DataFrame()
        returns = self.returns(equity_curve)
        pnls = self.trade_pnls(equity_curve, trades)
        avg_win, avg_loss = self.avg_win_loss(pnls)
        return {
            "total_return": self.total_return(equity_curve),
            "cagr": self.cagr(equity_curve),
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "calmar_ratio": self.calmar_ratio(equity_curve),
            "max_drawdown": self.max_drawdown(equity_curve),
            "max_drawdown_duration_days": self.max_drawdown_duration(equity_curve),
            "win_rate": self.win_rate(pnls),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": self.profit_factor(pnls),
            "total_trades": int(len(trades)),
            "avg_holding_period_days": self.avg_holding_period(trades, equity_curve),
            **self.worst_case_stats(equity_curve),
        }

    def print_report(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        regime_history: pd.DataFrame,
        benchmark_comparison: pd.DataFrame | None = None,
    ) -> None:
        """Print the summary, regime breakdown, confidence breakdown, and (if
        given) the benchmark comparison as rich tables to the terminal."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        overview = Table(title="Performance Summary")
        overview.add_column("Metric")
        overview.add_column("Value", justify="right")
        for key, value in self.summary(equity_curve, trades).items():
            overview.add_row(key.replace("_", " ").title(), _fmt(value))
        console.print(overview)

        if not regime_history.empty:
            regime_table = Table(title="Regime Breakdown")
            for col in ["Regime", "% Time In", "Return Contribution", "Avg Trade P&L", "Win Rate", "Sharpe"]:
                regime_table.add_column(col)
            for _, row in self.regime_breakdown(equity_curve, regime_history, trades).iterrows():
                regime_table.add_row(
                    str(row["regime"]),
                    _fmt(row["pct_time_in"], pct=True),
                    _fmt(row["return_contribution"], pct=True),
                    _fmt(row["avg_trade_pnl"]),
                    _fmt(row["win_rate"], pct=True),
                    _fmt(row["sharpe"]),
                )
            console.print(regime_table)

            confidence_table = Table(title="Confidence-Bucketed Performance")
            for col in ["Confidence", "Trades", "Sharpe", "Win Rate", "Avg P&L"]:
                confidence_table.add_column(col)
            for _, row in self.confidence_breakdown(equity_curve, regime_history, trades).iterrows():
                confidence_table.add_row(
                    str(row["confidence_bucket"]),
                    str(row["trades"]),
                    _fmt(row["sharpe"]),
                    _fmt(row["win_rate"], pct=True),
                    _fmt(row["avg_pnl"]),
                )
            console.print(confidence_table)

        if benchmark_comparison is not None and not benchmark_comparison.empty:
            bench_table = Table(title="Benchmark Comparison")
            for col in benchmark_comparison.columns:
                bench_table.add_column(str(col).replace("_", " ").title())
            for _, row in benchmark_comparison.iterrows():
                bench_table.add_row(*[_fmt(v) if isinstance(v, float) else str(v) for v in row])
            console.print(bench_table)

    def export_csv(
        self,
        output_dir: str,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        regime_history: pd.DataFrame,
        benchmark_comparison: pd.DataFrame | None = None,
    ) -> None:
        """Write equity_curve.csv, trade_log.csv, regime_history.csv, and
        (if given) benchmark_comparison.csv to `output_dir`."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        equity_curve.rename("equity").to_csv(out / "equity_curve.csv", header=True)
        trades.to_csv(out / "trade_log.csv", index=False)
        regime_history.to_csv(out / "regime_history.csv", index=False)
        if benchmark_comparison is not None and not benchmark_comparison.empty:
            benchmark_comparison.to_csv(out / "benchmark_comparison.csv", index=False)
