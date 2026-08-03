"""HMM regime detection engine.

DESIGN PHILOSOPHY
-----------------
This engine is a **volatility classifier**, not a price predictor. It fits a
Gaussian Hidden Markov Model to a standardized feature vector (returns,
realized volatility, trend, mean-reversion, momentum, range - see
data.feature_engineering) and groups history into a small number of
recurring "regimes". Regimes are labeled BEAR/BULL/CRASH/EUPHORIA etc. only
because they are ordered by mean return for a human-readable name; the
actual downstream use (core.regime_strategies) is to size exposure by how
calm or turbulent the *current* regime is - fully invested when calm,
de-risked when turbulent - not to bet on direction.

REGIME INFERENCE HAS NO LOOK-AHEAD BIAS
----------------------------------------
`model.predict()` / `model.decode()` (Viterbi) and `model.predict_proba()`
(forward-backward) both run over the *entire* sequence and revise past
states using future observations. Neither is used here. Regime inference
is done with the forward algorithm only (`filtered_probabilities`,
`predict_regime_filtered`): P(state_t | obs_1..t), which by construction
cannot change when bars after t are appended. See
tests/test_look_ahead.py::test_no_look_ahead_bias.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class RegimeLabel(str, Enum):
    """Human-readable regime name, ordered ascending by mean return."""

    CRASH = "CRASH"
    STRONG_BEAR = "STRONG_BEAR"
    BEAR = "BEAR"
    WEAK_BEAR = "WEAK_BEAR"
    NEUTRAL = "NEUTRAL"
    WEAK_BULL = "WEAK_BULL"
    BULL = "BULL"
    STRONG_BULL = "STRONG_BULL"
    EUPHORIA = "EUPHORIA"


# Label scheme per selected hidden-state count, in ascending-mean-return order.
_LABEL_SCHEMES: dict[int, list[RegimeLabel]] = {
    3: [RegimeLabel.BEAR, RegimeLabel.NEUTRAL, RegimeLabel.BULL],
    4: [RegimeLabel.CRASH, RegimeLabel.BEAR, RegimeLabel.BULL, RegimeLabel.EUPHORIA],
    5: [
        RegimeLabel.CRASH,
        RegimeLabel.BEAR,
        RegimeLabel.NEUTRAL,
        RegimeLabel.BULL,
        RegimeLabel.EUPHORIA,
    ],
    6: [
        RegimeLabel.CRASH,
        RegimeLabel.STRONG_BEAR,
        RegimeLabel.WEAK_BEAR,
        RegimeLabel.WEAK_BULL,
        RegimeLabel.STRONG_BULL,
        RegimeLabel.EUPHORIA,
    ],
    7: [
        RegimeLabel.CRASH,
        RegimeLabel.STRONG_BEAR,
        RegimeLabel.WEAK_BEAR,
        RegimeLabel.NEUTRAL,
        RegimeLabel.WEAK_BULL,
        RegimeLabel.STRONG_BULL,
        RegimeLabel.EUPHORIA,
    ],
}

# Static per-regime archetype defaults. These are independent of
# config/settings.yaml's vol-tier strategy/risk parameters (low/mid/high vol
# allocation, leverage, etc.) - the strategy layer combines both: it reads
# the RegimeInfo for the currently confirmed label to decide how the position
# should generally be postured, then applies the configured allocation
# tables on top.
_REGIME_DEFAULTS: dict[RegimeLabel, dict[str, float | str]] = {
    RegimeLabel.CRASH: dict(
        recommended_strategy_type="defensive", max_leverage_allowed=0.0, max_position_size_pct=0.05, min_confidence_to_act=0.70
    ),
    RegimeLabel.STRONG_BEAR: dict(
        recommended_strategy_type="defensive", max_leverage_allowed=0.0, max_position_size_pct=0.10, min_confidence_to_act=0.65
    ),
    RegimeLabel.BEAR: dict(
        recommended_strategy_type="reduced", max_leverage_allowed=0.0, max_position_size_pct=0.10, min_confidence_to_act=0.60
    ),
    RegimeLabel.WEAK_BEAR: dict(
        recommended_strategy_type="reduced", max_leverage_allowed=1.0, max_position_size_pct=0.12, min_confidence_to_act=0.58
    ),
    RegimeLabel.NEUTRAL: dict(
        recommended_strategy_type="neutral", max_leverage_allowed=1.0, max_position_size_pct=0.15, min_confidence_to_act=0.55
    ),
    RegimeLabel.WEAK_BULL: dict(
        recommended_strategy_type="growth", max_leverage_allowed=1.0, max_position_size_pct=0.15, min_confidence_to_act=0.55
    ),
    RegimeLabel.BULL: dict(
        recommended_strategy_type="growth", max_leverage_allowed=1.25, max_position_size_pct=0.15, min_confidence_to_act=0.55
    ),
    RegimeLabel.STRONG_BULL: dict(
        recommended_strategy_type="growth", max_leverage_allowed=1.25, max_position_size_pct=0.15, min_confidence_to_act=0.55
    ),
    RegimeLabel.EUPHORIA: dict(
        recommended_strategy_type="cautious_growth",
        max_leverage_allowed=1.0,
        max_position_size_pct=0.12,
        min_confidence_to_act=0.60,
    ),
}


@dataclass
class RegimeInfo:
    """Static metadata describing one regime of a fitted model."""

    regime_id: int
    regime_name: RegimeLabel
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Result of a single filtered regime call.

    `label`/`state_id` reflect the *confirmed* regime (see
    HMMEngine's stability filter), not necessarily the raw argmax of this
    bar's filtered distribution - a single contradictory bar does not flip
    the regime until it has persisted for `stability_bars` bars.
    """

    label: RegimeLabel
    state_id: int
    probability: float
    state_probabilities: dict[RegimeLabel, float]
    timestamp: pd.Timestamp
    is_confirmed: bool
    consecutive_bars: int


def _n_free_params(n_components: int, n_features: int, covariance_type: str) -> int:
    """Number of free parameters of a GaussianHMM, for BIC."""
    n_startprob = n_components - 1
    n_transmat = n_components * (n_components - 1)
    n_means = n_components * n_features
    if covariance_type == "full":
        n_covars = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        n_covars = n_components * n_features
    elif covariance_type == "tied":
        n_covars = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        n_covars = n_components
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    return n_startprob + n_transmat + n_means + n_covars


def _bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """BIC = -2 * log_likelihood + n_params * log(n_samples). Lower is better."""
    return -2.0 * log_likelihood + n_params * np.log(n_samples)


class HMMEngine:
    """Fits Gaussian HMMs for regime detection and performs look-ahead-free inference.

    Args:
        n_candidates: Candidate hidden-state counts to evaluate for model selection.
        n_init: Number of random initializations per candidate model.
        covariance_type: HMM emission covariance type (full, diag, tied, spherical).
        min_train_bars: Minimum bars of history required before fitting.
        stability_bars: Consecutive bars a regime must persist before being confirmed.
        flicker_window: Rolling window (bars) used to detect regime flicker.
        flicker_threshold: Max allowed confirmed-regime changes within flicker_window.
        min_confidence: Minimum posterior probability required to act on a regime call.
        return_feature: Feature column used as "mean return" for regime labeling.
        vol_feature: Feature column used as "mean volatility" for RegimeInfo metadata.
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
        return_feature: str = "ret_1",
        vol_feature: str = "realized_vol_20",
    ) -> None:
        self.n_candidates = list(n_candidates)
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_train_bars = min_train_bars
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence = min_confidence
        self.return_feature = return_feature
        self.vol_feature = vol_feature

        self.model: hmm.GaussianHMM | None = None
        self.n_components: int | None = None
        self.bic_scores: dict[int, float] = {}
        self.state_labels_: dict[int, RegimeLabel] = {}
        self.regime_info_: dict[RegimeLabel, RegimeInfo] = {}
        self.feature_columns_: list[str] = []
        self.training_date_: pd.Timestamp | None = None

        self._reset_inference_state()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> None:
        """Fit candidate HMMs on `features` and select the best one by BIC.

        Args:
            features: Standardized feature matrix (see
                data.feature_engineering.build_feature_matrix), indexed by time.
                Must contain `self.return_feature`.
        """
        if self.return_feature not in features.columns:
            raise ValueError(f"features must include the '{self.return_feature}' column for regime labeling")

        clean = features.dropna()
        if len(clean) < self.min_train_bars:
            raise ValueError(
                f"Need at least {self.min_train_bars} non-NaN bars to train, got {len(clean)} "
                f"(of {len(features)} total rows - check indicator/z-score warm-up)"
            )

        self.feature_columns_ = list(features.columns)
        self.select_best_model(clean)
        self.state_labels_ = self.label_states()
        self.regime_info_ = self._build_regime_info()
        self.training_date_ = pd.Timestamp.now(tz="UTC")
        self._reset_inference_state()

        logger.info(
            "HMM training complete: n_components=%d, labels=%s",
            self.n_components,
            {state_id: label.value for state_id, label in self.state_labels_.items()},
        )

    def select_best_model(self, features: pd.DataFrame) -> int:
        """Train every candidate n_components (with n_init restarts each), select lowest BIC.

        Stores the winning model on `self.model` / `self.n_components` and all
        candidates' BIC scores on `self.bic_scores`. Returns the selected
        n_components.
        """
        X = features[self.feature_columns_].to_numpy(dtype=float) if self.feature_columns_ else features.to_numpy(dtype=float)
        n_samples, n_features = X.shape

        candidate_bics: dict[int, float] = {}
        best_k: int | None = None
        best_bic = np.inf
        best_model: hmm.GaussianHMM | None = None

        for k in self.n_candidates:
            best_ll_for_k = -np.inf
            best_model_for_k: hmm.GaussianHMM | None = None

            for init in range(self.n_init):
                candidate = hmm.GaussianHMM(
                    n_components=k,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    random_state=init,
                )
                try:
                    candidate.fit(X)
                    log_likelihood = candidate.score(X)
                except Exception as exc:  # hmmlearn can raise on degenerate inits
                    logger.warning("HMM fit failed for n_components=%d init=%d: %s", k, init, exc)
                    continue

                if log_likelihood > best_ll_for_k:
                    best_ll_for_k = log_likelihood
                    best_model_for_k = candidate

            if best_model_for_k is None:
                logger.warning("All %d initializations failed for n_components=%d - skipping candidate", self.n_init, k)
                continue

            n_params = _n_free_params(k, n_features, self.covariance_type)
            bic = _bic(best_ll_for_k, n_params, n_samples)
            candidate_bics[k] = bic

            logger.info(
                "HMM candidate n_components=%d: log_likelihood=%.2f, n_params=%d, bic=%.2f, converged=%s, iterations=%d",
                k,
                best_ll_for_k,
                n_params,
                bic,
                best_model_for_k.monitor_.converged,
                best_model_for_k.monitor_.iter,
            )

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = best_model_for_k

        if best_model is None or best_k is None:
            raise RuntimeError("HMM training failed for every candidate n_components")

        logger.info("Selected n_components=%d with BIC=%.2f (all candidates: %s)", best_k, best_bic, candidate_bics)

        self.bic_scores = candidate_bics
        self.model = best_model
        self.n_components = best_k
        return best_k

    def label_states(self) -> dict[int, RegimeLabel]:
        """Map raw HMM hidden state indices to RegimeLabel, sorted by mean return ascending."""
        self._check_fitted()
        if self.n_components not in _LABEL_SCHEMES:
            raise ValueError(f"No label scheme defined for n_components={self.n_components}")

        return_idx = self.feature_columns_.index(self.return_feature)
        mean_returns = self.model.means_[:, return_idx]
        order = np.argsort(mean_returns)  # ascending: lowest return first
        scheme = _LABEL_SCHEMES[self.n_components]
        return {int(state_id): label for state_id, label in zip(order, scheme)}

    def _build_regime_info(self) -> dict[RegimeLabel, RegimeInfo]:
        return_idx = self.feature_columns_.index(self.return_feature)
        vol_idx = self.feature_columns_.index(self.vol_feature) if self.vol_feature in self.feature_columns_ else None

        info: dict[RegimeLabel, RegimeInfo] = {}
        for state_id, label in self.state_labels_.items():
            defaults = _REGIME_DEFAULTS[label]
            info[label] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                expected_return=float(self.model.means_[state_id, return_idx]),
                expected_volatility=float(self.model.means_[state_id, vol_idx]) if vol_idx is not None else float("nan"),
                recommended_strategy_type=str(defaults["recommended_strategy_type"]),
                max_leverage_allowed=float(defaults["max_leverage_allowed"]),
                max_position_size_pct=float(defaults["max_position_size_pct"]),
                min_confidence_to_act=float(defaults["min_confidence_to_act"]),
            )
        return info

    def get_regime_info(self, label: RegimeLabel) -> RegimeInfo:
        """Look up static metadata for a regime label of the fitted model."""
        self._check_fitted()
        return self.regime_info_[label]

    # ------------------------------------------------------------------
    # Look-ahead-free inference (forward algorithm only)
    # ------------------------------------------------------------------

    def filtered_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """P(state_t | obs_1..t) for every row t in `features`, via the forward algorithm.

        Pure/stateless: depends only on the rows actually passed in, never on
        rows that come after them and never on any previous call. This is the
        core no-look-ahead-bias primitive - `filtered_probabilities(x)[-1]`
        for `x = features.iloc[:T]` is guaranteed identical regardless of
        what (if anything) is appended to `features` after row T-1.

        Args:
            features: Feature matrix with the columns this model was fit on. No NaNs.

        Returns:
            Array of shape (len(features), n_components).
        """
        self._check_fitted()
        X = self._to_matrix(features)
        log_alpha = self._forward_log_alpha(X)
        return np.exp(log_alpha)

    def predict_regime_proba(self, features_up_to_now: pd.DataFrame) -> dict[RegimeLabel, float]:
        """Filtered probability distribution over regime labels for the latest bar."""
        probs = self.filtered_probabilities(features_up_to_now)[-1]
        return {self.state_labels_[state_id]: float(p) for state_id, p in enumerate(probs)}

    def predict_regime_filtered(self, features_up_to_now: pd.DataFrame) -> RegimeState:
        """Stateful, cached wrapper for the live/backtest loop.

        Intended to be called once per new bar, in increasing time order
        (`features_up_to_now` growing by one row each call). Internally caches
        the previous log-alpha vector so each call does O(1) work instead of
        replaying the whole history, then feeds the result through the
        regime-stability filter (see class docstring / config: stability_bars,
        flicker_window, flicker_threshold, min_confidence).

        Returns the *confirmed* regime (label/state_id), which only changes
        once a new raw regime call has persisted for `stability_bars` bars.
        """
        self._check_fitted()
        index = features_up_to_now.index

        if self._cache_index is not None and len(index) == len(self._cache_index) + 1 and index[:-1].equals(self._cache_index):
            new_row = self._to_matrix(features_up_to_now.iloc[[-1]])
            log_alpha_t = self._forward_log_alpha(new_row, prior_log_alpha=self._cache_log_alpha)[-1]
        elif self._cache_index is not None and index.equals(self._cache_index):
            log_alpha_t = self._cache_log_alpha
        else:
            log_alpha_t = self._forward_log_alpha(self._to_matrix(features_up_to_now))[-1]

        self._cache_index = index
        self._cache_log_alpha = log_alpha_t

        probs = np.exp(log_alpha_t)
        return self._update_stability(probs, timestamp=index[-1])

    def _to_matrix(self, features: pd.DataFrame) -> np.ndarray:
        return features[self.feature_columns_].to_numpy(dtype=float)

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        n_components = self.model.n_components
        log_probs = np.empty((X.shape[0], n_components))
        for state in range(n_components):
            log_probs[:, state] = multivariate_normal.logpdf(X, mean=self.model.means_[state], cov=self._covariance_for_state(state))
        return log_probs

    def _covariance_for_state(self, state: int) -> np.ndarray:
        covariance_type = self.model.covariance_type
        if covariance_type == "full":
            return self.model.covars_[state]
        if covariance_type == "diag":
            return np.diag(self.model.covars_[state])
        if covariance_type == "tied":
            return self.model.covars_
        if covariance_type == "spherical":
            return np.eye(self.model.n_features) * self.model.covars_[state]
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    def _forward_log_alpha(self, X: np.ndarray, prior_log_alpha: np.ndarray | None = None) -> np.ndarray:
        """Scaled forward recursion in log space (filtering only, no backward pass).

        alpha_t(i) = logsumexp_j(alpha_{t-1}(j) + log_transmat[j, i]) + log_emission_i(x_t)
        normalized to sum to 1 (in probability space) at every step.
        """
        log_emission = self._log_emission(X)
        log_startprob = np.log(np.clip(self.model.startprob_, 1e-300, None))
        log_transmat = np.log(np.clip(self.model.transmat_, 1e-300, None))
        n_samples, n_components = log_emission.shape

        log_alpha = np.empty((n_samples, n_components))
        if prior_log_alpha is None:
            log_alpha[0] = log_startprob + log_emission[0]
        else:
            log_alpha[0] = logsumexp(prior_log_alpha[:, None] + log_transmat, axis=0) + log_emission[0]
        log_alpha[0] -= logsumexp(log_alpha[0])

        for t in range(1, n_samples):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + log_emission[t]
            log_alpha[t] -= logsumexp(log_alpha[t])

        return log_alpha

    # ------------------------------------------------------------------
    # Regime stability filter
    # ------------------------------------------------------------------

    def _update_stability(self, probs: np.ndarray, timestamp: pd.Timestamp) -> RegimeState:
        raw_state_id = int(np.argmax(probs))
        raw_label = self.state_labels_[raw_state_id]
        confidence = float(probs[raw_state_id])

        if confidence < self.min_confidence and self._confirmed_label is not None:
            logger.info(
                "Low-confidence regime read (%.2f < %.2f) at %s - not counted as evidence of change",
                confidence,
                self.min_confidence,
                timestamp,
            )
            raw_label = self._confirmed_label

        if self._confirmed_label is None:
            self._confirmed_label = raw_label
            self._consecutive_bars = 1
            self._candidate_label = None
            self._candidate_count = 0
        elif raw_label == self._confirmed_label:
            self._consecutive_bars += 1
            self._candidate_label = None
            self._candidate_count = 0
        else:
            if raw_label == self._candidate_label:
                self._candidate_count += 1
            else:
                self._candidate_label = raw_label
                self._candidate_count = 1

            if self._candidate_count >= self.stability_bars:
                logger.warning(
                    "Regime change confirmed: %s -> %s at %s (persisted %d bars)",
                    self._confirmed_label.value,
                    raw_label.value,
                    timestamp,
                    self._candidate_count,
                )
                self._confirmed_label = raw_label
                self._consecutive_bars = self._candidate_count
                self._candidate_label = None
                self._candidate_count = 0
            else:
                logger.info(
                    "Regime candidate %s pending confirmation (%d/%d bars) at %s - holding %s",
                    raw_label.value,
                    self._candidate_count,
                    self.stability_bars,
                    timestamp,
                    self._confirmed_label.value,
                )
                self._consecutive_bars += 1

        is_confirmed = self._candidate_count == 0

        self._confirmed_history.append(self._confirmed_label)
        flicker_rate = sum(
            1 for prev, curr in zip(self._confirmed_history, list(self._confirmed_history)[1:]) if prev != curr
        )
        was_uncertain = self._uncertainty_mode
        self._uncertainty_mode = flicker_rate > self.flicker_threshold
        if self._uncertainty_mode and not was_uncertain:
            logger.warning(
                "Flicker rate %d exceeds threshold %d over last %d bars at %s - entering uncertainty mode",
                flicker_rate,
                self.flicker_threshold,
                self.flicker_window,
                timestamp,
            )
        elif was_uncertain and not self._uncertainty_mode:
            logger.info("Flicker rate back within threshold at %s - leaving uncertainty mode", timestamp)

        return RegimeState(
            label=self._confirmed_label,
            state_id=self._label_to_state()[self._confirmed_label],
            probability=confidence,
            state_probabilities={self.state_labels_[i]: float(p) for i, p in enumerate(probs)},
            timestamp=timestamp,
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    def _label_to_state(self) -> dict[RegimeLabel, int]:
        return {label: state_id for state_id, label in self.state_labels_.items()}

    def get_regime_stability(self) -> int:
        """Consecutive bars the currently confirmed regime has held."""
        return self._consecutive_bars

    def is_flickering(self) -> bool:
        """Whether the confirmed regime has changed more than flicker_threshold times
        within the last flicker_window bars (uncertainty mode)."""
        return self._uncertainty_mode

    def get_transition_matrix(self) -> np.ndarray:
        """Learned hidden-state transition matrix, shape (n_components, n_components)."""
        self._check_fitted()
        return self.model.transmat_.copy()

    def _reset_inference_state(self) -> None:
        self._cache_index: pd.Index | None = None
        self._cache_log_alpha: np.ndarray | None = None
        self._confirmed_label: RegimeLabel | None = None
        self._candidate_label: RegimeLabel | None = None
        self._candidate_count: int = 0
        self._consecutive_bars: int = 0
        self._confirmed_history: deque[RegimeLabel | None] = deque(maxlen=self.flicker_window)
        self._uncertainty_mode: bool = False

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("HMMEngine must be fit() before this call")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the fitted model plus training metadata (n_components, BIC, labels, date)."""
        self._check_fitted()
        payload = {
            "model": self.model,
            "n_components": self.n_components,
            "bic_scores": self.bic_scores,
            "state_labels": {state_id: label.value for state_id, label in self.state_labels_.items()},
            "feature_columns": self.feature_columns_,
            "training_date": self.training_date_,
            "config": {
                "n_candidates": self.n_candidates,
                "n_init": self.n_init,
                "covariance_type": self.covariance_type,
                "min_train_bars": self.min_train_bars,
                "stability_bars": self.stability_bars,
                "flicker_window": self.flicker_window,
                "flicker_threshold": self.flicker_threshold,
                "min_confidence": self.min_confidence,
                "return_feature": self.return_feature,
                "vol_feature": self.vol_feature,
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Saved HMM model to %s (n_components=%d, trained %s)", path, self.n_components, self.training_date_)

    @classmethod
    def load(cls, path: str) -> "HMMEngine":
        """Restore a model previously written by `save`."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        engine = cls(**payload["config"])
        engine.model = payload["model"]
        engine.n_components = payload["n_components"]
        engine.bic_scores = payload["bic_scores"]
        engine.state_labels_ = {state_id: RegimeLabel(value) for state_id, value in payload["state_labels"].items()}
        engine.feature_columns_ = payload["feature_columns"]
        engine.training_date_ = payload["training_date"]
        engine.regime_info_ = engine._build_regime_info()
        engine._reset_inference_state()

        logger.info("Loaded HMM model from %s (n_components=%d, trained %s)", path, engine.n_components, engine.training_date_)
        return engine
