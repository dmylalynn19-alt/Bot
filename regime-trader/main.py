"""Entry point for the regime-trader system."""

from __future__ import annotations

import argparse


def load_config(path: str) -> dict:
    """Load and merge settings.yaml and environment/credentials into a config dict."""
    raise NotImplementedError


def build_pipeline(config: dict):
    """Wire up the data feed, HMM engine, strategy, risk manager, and executor."""
    raise NotImplementedError


def run(config: dict) -> None:
    """Run the live trading loop."""
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    raise NotImplementedError


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
