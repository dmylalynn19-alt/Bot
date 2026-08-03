"""HMM regime detection engine.

Fits candidate Gaussian Hidden Markov Models to return/feature series, selects
the best model, and infers the current market regime with a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Regime(Enum):
    """Labeled volatility/trend regime for a symbol."""

    LOW_VOL = "low_vol"
    MID_VOL_TREND = "mid_vol_trend"
    MID_VOL_NO_TREND = "mid_vol_no_trend"
    HIGH_VOL = "high_vol"


@dataclass
class RegimeState:
    """Result of a regime inference for a single symbol at a point in time."""

    regime: Regime
    confidence: float
    state_id: int
    is_stable: bool


class HMMEngine:
    """Fits and applies Gaussian HMMs for regime detection.

    Args:
        n_candidates: Candidate hidden-state counts to evaluate for model selection.
        n_init: Number of random initializations per candidate model.
        covariance_type: HMM emission covariance type.
        min_train_bars: Minimum bars of history required before fitting.
        stability_bars: Consecutive bars a regime must persist before being confirmed.
        flicker_window: Rolling window (bars) used to detect regime flicker.
        flicker_threshold: Max allowed regime changes within flicker_window.
        min_confidence: Minimum posterior probability required to act on a regime call.
    """

    def __init__(
        self,
        n_candidates: list[int],
        n_init: int,
        covariance_type: str,
        min_train_bars: int,
        stability_bars: int,
        flicker_window: int,
        flicker_threshold: int,
        min_confidence: float,
    ) -> None:
        raise NotImplementedError

    def fit(self, features: pd.DataFrame) -> None:
        """Fit candidate HMMs on the given feature matrix and select the best one."""
        raise NotImplementedError

    def select_best_model(self, features: pd.DataFrame) -> int:
        """Select the best candidate hidden-state count via information criteria."""
        raise NotImplementedError

    def predict_state(self, features: pd.DataFrame) -> RegimeState:
        """Infer the current regime and confidence for the latest observation."""
        raise NotImplementedError

    def is_flickering(self, state_history: list[int]) -> bool:
        """Check whether recent state transitions exceed the flicker threshold."""
        raise NotImplementedError

    def label_states(self, features: pd.DataFrame) -> dict[int, Regime]:
        """Map raw HMM hidden state indices to semantic Regime labels."""
        raise NotImplementedError
