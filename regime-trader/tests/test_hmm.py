"""Tests for core.hmm_engine.HMMEngine."""

from __future__ import annotations


def test_fit_selects_best_model() -> None:
    """HMMEngine.fit should select a hidden-state count from n_candidates."""
    raise NotImplementedError


def test_predict_state_returns_confidence() -> None:
    """predict_state should return a RegimeState with a confidence score."""
    raise NotImplementedError


def test_flicker_detection() -> None:
    """is_flickering should flag regimes that change too often within the flicker window."""
    raise NotImplementedError
