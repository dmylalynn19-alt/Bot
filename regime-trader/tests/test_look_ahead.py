"""Verify no look-ahead bias in feature computation and HMM regime inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine
from core.support_resistance import _cluster_mean, cluster_levels, find_pivot_highs, find_pivot_lows
from data.feature_engineering import build_feature_matrix


def _make_synthetic_bars(n_bars: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic OHLCV series, long enough for every indicator's warm-up."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_bars, freq="B")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars))), index=idx)
    high = close * (1 + rng.uniform(0, 0.01, n_bars))
    low = close * (1 - rng.uniform(0, 0.01, n_bars))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def _make_synthetic_features(n_bars: int, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    """Features sampled from a known 3-state GaussianHMM, so a fit is fast and well-separated."""
    from hmmlearn import hmm as hmmlearn_hmm

    rng = np.random.default_rng(seed)
    true_model = hmmlearn_hmm.GaussianHMM(n_components=3, covariance_type="diag", random_state=seed)
    true_model.startprob_ = np.array([1 / 3, 1 / 3, 1 / 3])
    true_model.transmat_ = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.03, 0.94, 0.03],
            [0.01, 0.04, 0.95],
        ]
    )
    true_model.means_ = np.array(
        [
            [-2.0] * n_features,
            [0.0] * n_features,
            [2.0] * n_features,
        ]
    )
    true_model.covars_ = np.tile(np.ones(n_features), (3, 1))

    X, _ = true_model.sample(n_bars, random_state=rng.integers(0, 2**31 - 1))
    idx = pd.date_range("2018-01-01", periods=n_bars, freq="B")
    columns = ["ret_1"] + [f"feat_{i}" for i in range(1, n_features)]
    return pd.DataFrame(X, index=idx, columns=columns)


def test_features_use_only_past_data() -> None:
    """FeatureEngineer outputs at time t must not depend on data after t."""
    bars = _make_synthetic_bars(300)

    features_short = build_feature_matrix(bars.iloc[:200])
    features_long = build_feature_matrix(bars.iloc[:300]).iloc[:200]

    pd.testing.assert_frame_equal(features_short, features_long)


def test_no_look_ahead_bias() -> None:
    """Filtered regime probabilities at bar T must be identical whether computed from
    data[0:T] or as an intermediate step of data[0:T+100] - the defining property of
    forward-only (filtered) inference, and the reason predict_regime_filtered/
    filtered_probabilities must never be implemented with Viterbi (model.predict) or
    forward-backward (model.predict_proba), both of which revise past states using
    future observations.
    """
    features = _make_synthetic_features(n_bars=600)

    engine = HMMEngine(
        n_candidates=[3],
        n_init=2,
        covariance_type="diag",
        min_train_bars=100,
        stability_bars=3,
        flicker_window=20,
        flicker_threshold=4,
        min_confidence=0.55,
        return_feature="ret_1",
    )
    engine.fit(features)

    filtered_short = engine.filtered_probabilities(features.iloc[:400])
    filtered_long = engine.filtered_probabilities(features.iloc[:500])

    # The filtered distribution for bar 399 must be identical in both runs: it must not
    # be affected by bars 400-499, which did not exist yet in the first run.
    np.testing.assert_allclose(filtered_short[-1], filtered_long[399], atol=1e-9)

    regime_short = engine.predict_regime_proba(features.iloc[:400])
    regime_long_at_400 = {
        engine.state_labels_[i]: float(p) for i, p in enumerate(filtered_long[399])
    }
    assert regime_short == pytest.approx(regime_long_at_400, abs=1e-9)


def test_support_resistance_no_look_ahead() -> None:
    """A support/resistance zone's price/touch-count, as of any given cutoff
    time, must not depend on whether pivots after that cutoff exist.

    find_pivot_highs/find_pivot_lows are checked directly (a pivot at bar i
    only ever looks at bars within pivot_window of i, so truncating the
    series can only turn a *confirmed* pivot into an *unconfirmable* one
    near the new edge - it can never change an already-confirmable one).

    cluster_levels is checked against the "obviously correct" reference
    implementation: process every pivot in time order, and the clusters
    that exist at the instant a given pivot is processed must be identical
    whether or not later pivots are ever appended - i.e. clustering must
    never retroactively reassign an earlier pivot once later ones arrive.
    Note: a real cluster's *final* touch count legitimately grows as more
    time passes (that's new data arriving, not look-ahead bias) - what must
    never happen is an *earlier* touch's assignment changing after the fact,
    which is what this test isolates by snapshotting mid-loop.
    """
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    base = 105 + 5 * np.sin(np.linspace(0, 20, 300)) + rng.normal(0, 0.3, 300)
    close = pd.Series(base, index=idx)
    high = close + rng.uniform(0.1, 0.5, 300)
    low = close - rng.uniform(0.1, 0.5, 300)

    window = 5
    pivot_highs_short = find_pivot_highs(high.iloc[:150], window)
    pivot_highs_long = find_pivot_highs(high.iloc[:300], window)
    confirmable = high.index[: 150 - window]
    pd.testing.assert_series_equal(pivot_highs_short.loc[confirmable], pivot_highs_long.loc[confirmable])

    pivot_lows_short = find_pivot_lows(low.iloc[:150], window)
    pivot_lows_long = find_pivot_lows(low.iloc[:300], window)
    pd.testing.assert_series_equal(pivot_lows_short.loc[confirmable], pivot_lows_long.loc[confirmable])

    # cluster_levels: snapshot cluster state at a cutoff, mid-loop, within a
    # single pass over the full series, and compare against running the
    # same clustering on only the data up to that cutoff.
    prices = close
    cutoff = prices.index[39]
    early_only = cluster_levels(prices.iloc[:40], "support", tolerance_pct=0.02, min_touches=1)

    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    snapshot = None
    for ts, price in prices.sort_index().items():
        best_cluster, best_distance = None, None
        for cluster in clusters:
            distance = abs(price - _cluster_mean(cluster)) / _cluster_mean(cluster)
            if distance <= 0.02 and (best_distance is None or distance < best_distance):
                best_cluster, best_distance = cluster, distance
        if best_cluster is not None:
            best_cluster.append((ts, price))
        else:
            clusters.append([(ts, price)])
        if ts == cutoff:
            snapshot = [list(c) for c in clusters]

    snapshot_levels = sorted((round(_cluster_mean(c), 6), len(c)) for c in snapshot)
    early_levels = sorted((round(level.price, 6), level.touches) for level in early_only)
    assert snapshot_levels == early_levels


def test_backtester_refits_only_on_past_window() -> None:
    """WalkForwardBacktester must only fit the HMM on data available at each step."""
    raise NotImplementedError


def test_signal_generation_does_not_peek_future_bars() -> None:
    """SignalGenerator output at bar t must be reproducible using only bars <= t."""
    raise NotImplementedError
