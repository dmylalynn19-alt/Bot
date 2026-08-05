"""Support/resistance level detection from swing pivots.

A pivot high at bar i is a bar whose high is the highest within a window of
`pivot_window` bars on both sides of it - which means a pivot is only
*confirmed* `pivot_window` bars after it actually happened (you can't know a
swing high was the swing high until price has moved away from it). This is
deliberate and important: `SupportResistanceDetector.detect` only ever
returns levels built from pivots confirmable using bars already seen, so a
backtest walking bar by bar never gets a level earlier than it could have in
real time - see tests/test_look_ahead.py for the equivalent guarantee on the
HMM/feature side of this project.

Nearby pivots are clustered into zones (exact repeated price levels are rare;
price reacts to a *zone*), each with a strength (touch count) - more touches
is a more significant level.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Level:
    """A clustered support or resistance zone."""

    price: float
    kind: str  # "support" or "resistance"
    touches: int
    first_touch: pd.Timestamp
    last_touch: pd.Timestamp


def find_pivot_highs(high: pd.Series, window: int) -> pd.Series:
    """Boolean mask: True at bar i if high[i] is the max within [i-window, i+window].

    Confirmable only once `window` bars past i are available - the caller
    must not treat a pivot as known before then (see module docstring).
    """
    rolling_max = high.rolling(2 * window + 1, center=True).max()
    return high == rolling_max


def find_pivot_lows(low: pd.Series, window: int) -> pd.Series:
    """Boolean mask: True at bar i if low[i] is the min within [i-window, i+window]."""
    rolling_min = low.rolling(2 * window + 1, center=True).min()
    return low == rolling_min


def cluster_levels(prices: pd.Series, kind: str, tolerance_pct: float, min_touches: int) -> list[Level]:
    """Incrementally cluster `prices` (a Series of pivot prices indexed by
    timestamp) into zones, processing pivots in TIME order: each new pivot
    joins whichever existing cluster's running mean it's closest to (if
    within `tolerance_pct`), or starts a new cluster.

    Time order is deliberate, not price order: clustering in price order
    would let a *later* pivot change which cluster an *earlier* one lands in
    (inserting a point between two others can split or merge clusters
    retroactively), which is look-ahead bias - a level computed from
    data[0:T] must not change once T grows. Processing in time order means a
    pivot's cluster assignment only ever depends on pivots at or before it.
    Clusters with fewer than `min_touches` are dropped (not significant
    enough to trade against).
    """
    if prices.empty:
        return []

    ordered = prices.sort_index()
    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    for timestamp, price in ordered.items():
        best_cluster, best_distance = None, None
        for cluster in clusters:
            distance = abs(price - _cluster_mean(cluster)) / _cluster_mean(cluster)
            if distance <= tolerance_pct and (best_distance is None or distance < best_distance):
                best_cluster, best_distance = cluster, distance
        if best_cluster is not None:
            best_cluster.append((timestamp, price))
        else:
            clusters.append([(timestamp, price)])

    levels = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        timestamps = [t for t, _ in cluster]
        levels.append(
            Level(
                price=_cluster_mean(cluster),
                kind=kind,
                touches=len(cluster),
                first_touch=min(timestamps),
                last_touch=max(timestamps),
            )
        )
    return levels


def _cluster_mean(cluster: list[tuple[pd.Timestamp, float]]) -> float:
    return sum(price for _, price in cluster) / len(cluster)


class SupportResistanceDetector:
    """Detects support/resistance zones from swing pivots in OHLC history.

    Args:
        pivot_window: Bars on each side a swing high/low must beat to count
            as a pivot (also how many bars "stale" the most recent usable
            pivot is - a genuine no-look-ahead cost, not a bug).
        cluster_tolerance_pct: Pivots within this fractional distance of each
            other are merged into one zone.
        min_touches: Minimum pivots required to keep a zone (filters out
            one-off levels with no real significance).
    """

    def __init__(self, pivot_window: int = 5, cluster_tolerance_pct: float = 0.005, min_touches: int = 2) -> None:
        self.pivot_window = pivot_window
        self.cluster_tolerance_pct = cluster_tolerance_pct
        self.min_touches = min_touches

    def detect(self, bars: pd.DataFrame) -> list[Level]:
        """Detect support and resistance zones from `bars` (OHLC, most recent
        bar last). Only uses pivots confirmable within the given history -
        i.e. never a pivot within the last `pivot_window` bars, since those
        can't be confirmed yet.
        """
        if len(bars) < 2 * self.pivot_window + 1:
            return []

        pivot_high_mask = find_pivot_highs(bars["high"], self.pivot_window)
        pivot_low_mask = find_pivot_lows(bars["low"], self.pivot_window)

        # Pivots need pivot_window bars *after* them to be confirmed - drop
        # any within the trailing window of the given history.
        confirmable = bars.index[: len(bars) - self.pivot_window]
        pivot_highs = bars.loc[confirmable, "high"][pivot_high_mask.loc[confirmable]]
        pivot_lows = bars.loc[confirmable, "low"][pivot_low_mask.loc[confirmable]]

        resistance = cluster_levels(pivot_highs, "resistance", self.cluster_tolerance_pct, self.min_touches)
        support = cluster_levels(pivot_lows, "support", self.cluster_tolerance_pct, self.min_touches)
        return sorted(support + resistance, key=lambda level: level.price)

    def nearest_support(self, levels: list[Level], price: float) -> Level | None:
        """Nearest support level at or below `price`, if any."""
        candidates = [level for level in levels if level.kind == "support" and level.price <= price]
        return max(candidates, key=lambda level: level.price) if candidates else None

    def nearest_resistance(self, levels: list[Level], price: float) -> Level | None:
        """Nearest resistance level at or above `price`, if any."""
        candidates = [level for level in levels if level.kind == "resistance" and level.price >= price]
        return min(candidates, key=lambda level: level.price) if candidates else None
