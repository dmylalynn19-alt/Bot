"""Alerting on risk events, regime changes, and system errors."""

from __future__ import annotations

from enum import Enum


class AlertSeverity(Enum):
    """Severity level of an alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """Sends rate-limited alerts for significant events.

    Args:
        rate_limit_minutes: Minimum minutes between repeat alerts of the same type.
    """

    def __init__(self, rate_limit_minutes: int) -> None:
        raise NotImplementedError

    def send_alert(self, message: str, severity: AlertSeverity) -> None:
        """Send an alert, subject to rate limiting."""
        raise NotImplementedError

    def should_rate_limit(self, alert_key: str) -> bool:
        """Check whether an alert of this type was sent too recently."""
        raise NotImplementedError
