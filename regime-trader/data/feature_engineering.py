"""Technical indicators and feature computation.

Computes the observable feature set fed into the HMM volatility classifier
(core.hmm_engine.HMMEngine) from raw OHLCV bars: returns, realized
volatility, trend, mean-reversion, momentum, and range indicators. Every
indicator is implemented as a pure function of its inputs (same inputs
always produce the same outputs, no hidden state) and every rolling window
used only ever looks backward, so none of them can introduce look-ahead
bias - see tests/test_look_ahead.py::test_features_use_only_past_data.

All features are standardized with a rolling 252-bar z-score before being
handed to the HMM, since GaussianHMM emissions assume comparable feature
scales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

# Column names of the raw (pre-standardization) feature matrix produced by
# build_raw_features(). core.hmm_engine.HMMEngine.return_feature /
# vol_feature default to entries in this list.
RAW_FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_5",
    "ret_20",
    "realized_vol_20",
    "vol_ratio_5_20",
    "adx_14",
    "sma_50_slope",
    "rsi_14",
    "dist_from_sma_200_pct",
    "roc_10",
    "roc_20",
    "atr_norm_14",
]


def log_return(close: pd.Series, period: int) -> pd.Series:
    """Log return of `close` over `period` bars: ln(close_t / close_{t-period})."""
    return np.log(close / close.shift(period))


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized volatility: standard deviation of returns over `window` bars."""
    return returns.rolling(window).std()


def volatility_ratio(returns: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """Ratio of short-window to long-window realized volatility (vol-of-vol proxy)."""
    short_vol = realized_volatility(returns, short_window)
    long_vol = realized_volatility(returns, long_window)
    return short_vol / long_vol


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index: trend strength, independent of direction."""
    return ta.trend.ADXIndicator(high, low, close, window=window, fillna=False).adx()


def sma_slope(close: pd.Series, window: int = 50) -> pd.Series:
    """Bar-over-bar percentage slope of the `window`-period simple moving average."""
    sma = close.rolling(window).mean()
    return sma.pct_change()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    return ta.momentum.RSIIndicator(close, window=window, fillna=False).rsi()


def distance_from_sma_pct(close: pd.Series, window: int = 200) -> pd.Series:
    """Distance of `close` from its `window`-period SMA, as a fraction of price."""
    sma = close.rolling(window).mean()
    return (close - sma) / sma


def roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of change of `close` over `period` bars, in percent."""
    return ta.momentum.ROCIndicator(close, window=period, fillna=False).roc()


def normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range normalized by price, so it is comparable across symbols."""
    atr = ta.volatility.AverageTrueRange(high, low, close, window=window, fillna=False).average_true_range()
    return atr / close


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score: (x_t - rolling_mean_t) / rolling_std_t, using only past+current bars."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def build_raw_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute the raw (unstandardized) feature matrix from OHLCV bars.

    Args:
        bars: DataFrame indexed by time with columns "open", "high", "low", "close".

    Returns:
        DataFrame with columns RAW_FEATURE_COLUMNS, same index as `bars`. Rows
        within each indicator's warm-up window are NaN.
    """
    close, high, low = bars["close"], bars["high"], bars["low"]
    returns_1 = log_return(close, 1)

    features = pd.DataFrame(index=bars.index)
    features["ret_1"] = returns_1
    features["ret_5"] = log_return(close, 5)
    features["ret_20"] = log_return(close, 20)
    features["realized_vol_20"] = realized_volatility(returns_1, 20)
    features["vol_ratio_5_20"] = volatility_ratio(returns_1, 5, 20)
    features["adx_14"] = adx(high, low, close, 14)
    features["sma_50_slope"] = sma_slope(close, 50)
    features["rsi_14"] = rsi(close, 14)
    features["dist_from_sma_200_pct"] = distance_from_sma_pct(close, 200)
    features["roc_10"] = roc(close, 10)
    features["roc_20"] = roc(close, 20)
    features["atr_norm_14"] = normalized_atr(high, low, close, 14)
    return features[RAW_FEATURE_COLUMNS]


def build_feature_matrix(bars: pd.DataFrame, zscore_window: int = 252) -> pd.DataFrame:
    """Compute the full, standardized feature matrix consumed by the HMM.

    Every raw feature (see `build_raw_features`) is passed through a rolling
    `zscore_window`-bar z-score so all features share a comparable scale.
    Rows before the longest warm-up window (indicator lookback + zscore_window)
    are NaN and should be dropped by the caller before fitting a model.

    Args:
        bars: DataFrame indexed by time with columns "open", "high", "low", "close".
        zscore_window: Rolling lookback (bars) used to standardize every feature.

    Returns:
        DataFrame with columns RAW_FEATURE_COLUMNS, same index as `bars`.
    """
    raw = build_raw_features(bars)
    return raw.apply(lambda column: rolling_zscore(column, zscore_window))


class FeatureEngineer:
    """Thin object wrapper around the pure feature functions above.

    Args:
        zscore_window: Rolling lookback (bars) used to standardize every feature.
    """

    def __init__(self, zscore_window: int = 252) -> None:
        self.zscore_window = zscore_window

    def compute_returns(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute 1/5/20-period log returns."""
        close = bars["close"]
        return pd.DataFrame(
            {
                "ret_1": log_return(close, 1),
                "ret_5": log_return(close, 5),
                "ret_20": log_return(close, 20),
            },
            index=bars.index,
        )

    def compute_volatility(self, bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Compute realized volatility and the short/long volatility ratio."""
        returns_1 = log_return(bars["close"], 1)
        return pd.DataFrame(
            {
                f"realized_vol_{window}": realized_volatility(returns_1, window),
                "vol_ratio_5_20": volatility_ratio(returns_1, 5, window),
            },
            index=bars.index,
        )

    def compute_trend_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute ADX(14) and the slope of the 50-period SMA."""
        return pd.DataFrame(
            {
                "adx_14": adx(bars["high"], bars["low"], bars["close"], 14),
                "sma_50_slope": sma_slope(bars["close"], 50),
            },
            index=bars.index,
        )

    def build_feature_matrix(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Build the full standardized feature matrix (see module-level build_feature_matrix)."""
        return build_feature_matrix(bars, self.zscore_window)
