"""Structured logging."""

from __future__ import annotations

import logging


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create or retrieve a configured structured logger."""
    raise NotImplementedError


class TradeLogger:
    """Structured logger for trades, signals, and regime changes.

    Args:
        log_dir: Directory where structured log files are written.
    """

    def __init__(self, log_dir: str) -> None:
        raise NotImplementedError

    def log_signal(self, symbol: str, action: str, confidence: float) -> None:
        """Log a generated trading signal."""
        raise NotImplementedError

    def log_order(self, order_id: str, symbol: str, status: str) -> None:
        """Log an order lifecycle event."""
        raise NotImplementedError

    def log_regime_change(self, symbol: str, old_regime: str, new_regime: str) -> None:
        """Log a detected regime transition."""
        raise NotImplementedError
