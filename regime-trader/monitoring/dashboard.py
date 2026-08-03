"""Live monitoring dashboard."""

from __future__ import annotations


class Dashboard:
    """Displays live account, position, and regime state.

    Args:
        refresh_seconds: How often the dashboard refreshes.
    """

    def __init__(self, refresh_seconds: int) -> None:
        raise NotImplementedError

    def render(self) -> None:
        """Render the current dashboard view."""
        raise NotImplementedError

    def run(self) -> None:
        """Start the dashboard refresh loop."""
        raise NotImplementedError
