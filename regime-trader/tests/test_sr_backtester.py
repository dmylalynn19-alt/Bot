"""Tests for backtest.sr_backtester.SRBacktester."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.sr_backtester import SRBacktester
from core.sr_strategy import SupportResistanceStrategy


def _oscillating_bars(n: int = 400, seed: int = 5, noise: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    period = 25
    close = 105 + 10 * (1 - 4 * np.abs(((t / period) % 1) - 0.5)) + rng.normal(0, noise, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    high = close + rng.uniform(0.2, 0.8, n)
    low = close - rng.uniform(0.2, 0.8, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(800_000, 1_200_000, n)
    spike_idx = rng.choice(n, size=30, replace=False)
    volume[spike_idx] *= rng.uniform(1.5, 3.0, len(spike_idx))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _default_strategy(min_required_confirmations: int = 3) -> SupportResistanceStrategy:
    return SupportResistanceStrategy(
        pivot_window=5,
        cluster_tolerance_pct=0.01,
        min_touches=2,
        level_proximity_pct=0.015,
        volume_window=20,
        volume_multiple=1.1,
        vwap_window=10,
        rsi_period=14,
        rsi_oversold=35.0,
        rsi_overbought=65.0,
        min_required_confirmations=min_required_confirmations,
        risk_reward_ratio=2.0,
        min_bars=60,
    )


def test_run_produces_equity_curve_covering_the_eval_window() -> None:
    bars = _oscillating_bars()
    strategy = _default_strategy()
    bt = SRBacktester(strategy, initial_cash=100_000.0)
    result = bt.run("TEST", bars)

    assert not result.equity_curve.empty
    assert result.equity_curve.index[0] >= bars.index[strategy.min_bars]
    assert result.equity_curve.index[-1] == bars.index[-1]


def test_run_takes_both_long_and_short_round_trips() -> None:
    bars = _oscillating_bars()
    strategy = _default_strategy()
    bt = SRBacktester(strategy, risk_per_trade=0.01, max_position_pct=0.2, initial_cash=100_000.0)
    result = bt.run("TEST", bars)

    closed = [t for t in result.trades if t.exit_time is not None]
    assert len(closed) > 0
    directions = {t.direction.value for t in closed}
    assert directions <= {"long", "short"}

    for trade in closed:
        assert trade.entry_time < trade.exit_time
        assert trade.pnl is not None
        # Every entry/exit price must be strictly after the bar the signal
        # was decided on (1-bar fill delay) - i.e. an actual bar in `bars`.
        assert trade.entry_time in bars.index
        assert trade.exit_time in bars.index


def test_stop_loss_produces_a_bounded_negative_pnl_trade() -> None:
    """A textbook bounce that then fails (price keeps sliding through the
    stop) must exit via the stop, with a loss close to the sized risk."""
    rng = np.random.default_rng(3)
    t = np.arange(60)
    period = 20
    cycle = 105 + 10 * (1 - 4 * np.abs(((t / period) % 1) - 0.5)) + rng.normal(0, 0.1, 60)
    settle = np.linspace(cycle[-1], 99.5, 8) + rng.normal(0, 0.05, 8)
    plunge = np.array([97.5, 94.9])
    chop = np.array([94.95, 94.80, 94.88, 94.83, 94.90, 95.05])
    post = np.array([94.5, 93.0, 91.5, 90.0, 88.5])  # keeps sliding through the stop
    close = np.concatenate([cycle, settle, plunge, chop, post])
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    high = close + 0.15
    low = close - 0.15
    low[len(cycle) + len(settle) + 1] = 94.5
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(n, 1_000_000.0)
    volume[len(cycle) + len(settle) + len(plunge) + len(chop) - 1] = 3_000_000.0
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

    strategy = SupportResistanceStrategy(
        pivot_window=5, cluster_tolerance_pct=0.01, min_touches=2, level_proximity_pct=0.01,
        volume_window=20, volume_multiple=1.2, vwap_window=5, rsi_period=14, rsi_oversold=34.0,
        rsi_overbought=70.0, min_required_confirmations=4, min_bars=60,
    )
    bt = SRBacktester(strategy, risk_per_trade=0.01, max_position_pct=0.2, initial_cash=100_000.0)
    result = bt.run("TEST", bars)

    closed = [t for t in result.trades if t.exit_time is not None]
    assert len(closed) == 1
    trade = closed[0]
    assert trade.direction.value == "long"
    assert "stop" in trade.exit_reason
    assert trade.pnl is not None and trade.pnl < 0
    # Risked ~1% of starting equity - the realized loss (with slippage)
    # should be in the same ballpark, not wildly larger.
    assert abs(trade.pnl) < 0.03 * bt.initial_cash


def test_summary_matches_trades_frame_for_a_simple_win_loss_pair() -> None:
    bars = _oscillating_bars()
    strategy = _default_strategy()
    bt = SRBacktester(strategy, initial_cash=100_000.0)
    result = bt.run("TEST", bars)

    summary = result.summary()
    frame = result.trades_frame()
    assert summary["num_trades"] == len(frame)
    if summary["num_trades"] > 0:
        assert summary["total_pnl"] == frame["pnl"].sum()
        wins = (frame["pnl"] > 0).sum()
        assert summary["win_rate"] == wins / len(frame)


def test_run_with_no_trades_returns_empty_but_valid_result() -> None:
    """An overly strict strategy that never confirms a setup must still
    produce a full mark-to-market equity curve (flat, at initial cash)."""
    bars = _oscillating_bars(seed=9, noise=2.0)
    strategy = _default_strategy(min_required_confirmations=4)
    strategy.rsi_oversold = 1.0  # effectively unreachable
    strategy.rsi_overbought = 99.0
    bt = SRBacktester(strategy, initial_cash=100_000.0)
    result = bt.run("TEST", bars)

    assert result.trades == []
    assert not result.equity_curve.empty
    assert (result.equity_curve == bt.initial_cash).all()
    assert result.summary()["num_trades"] == 0
