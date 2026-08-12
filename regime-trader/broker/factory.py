"""Builds whichever broker client is configured (config `broker.provider`,
default "alpaca"), returning something satisfying broker.base.BrokerClient
either way - callers (main.py, and everything downstream: OrderExecutor,
PositionTracker) never need to know or care which one they got.

Only one broker executes trades at a time (the configured `provider`) -
this project doesn't (yet) split execution across both simultaneously. See
core/key_levels.py/broker/schwab_client.py's docstrings for why Webull
isn't an execution option here at all (data-only, if wired in at all).
"""

from __future__ import annotations

import os


def build_broker_client(config: dict):
    """Construct the configured broker client from `config` + environment
    variables (.env). Returns an AlpacaClient or a SchwabAdapter-wrapped
    SchwabClient - both satisfy broker.base.BrokerClient.

    `config["broker"]["provider"]` selects which - "alpaca" (default) or
    "schwab". Credentials always come from the environment (never
    settings.yaml, which is safe to commit) - see .env.example.
    """
    provider = config.get("broker", {}).get("provider", "alpaca").lower()

    if provider == "alpaca":
        return _build_alpaca_client()
    if provider == "schwab":
        return _build_schwab_client()
    raise ValueError(f"Unknown broker.provider '{provider}'; expected 'alpaca' or 'schwab'")


def _build_alpaca_client():
    from broker.alpaca_client import AlpacaClient

    return AlpacaClient(
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
    )


def _build_schwab_client():
    from broker.schwab_adapter import SchwabAdapter
    from broker.schwab_client import SchwabClient

    client = SchwabClient(
        api_key=os.environ.get("SCHWAB_API_KEY", ""),
        app_secret=os.environ.get("SCHWAB_APP_SECRET", ""),
        callback_url=os.environ.get("SCHWAB_CALLBACK_URL", ""),
        token_path=os.environ.get("SCHWAB_TOKEN_PATH", "schwab_token.json"),
    )
    return SchwabAdapter(client)
