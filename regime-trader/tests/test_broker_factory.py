"""Tests for broker.factory.build_broker_client's provider dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from broker.factory import build_broker_client


def test_defaults_to_alpaca_when_provider_unset() -> None:
    with patch("broker.alpaca_client.AlpacaClient") as mock_alpaca:
        build_broker_client({"broker": {}})
        mock_alpaca.assert_called_once()


def test_builds_alpaca_client_explicitly() -> None:
    with patch("broker.alpaca_client.AlpacaClient") as mock_alpaca:
        build_broker_client({"broker": {"provider": "alpaca"}})
        mock_alpaca.assert_called_once()


def test_builds_schwab_client_wrapped_in_adapter() -> None:
    with patch("broker.schwab_client.SchwabClient") as mock_schwab, patch("broker.schwab_adapter.SchwabAdapter") as mock_adapter:
        build_broker_client({"broker": {"provider": "schwab"}})
        mock_schwab.assert_called_once()
        mock_adapter.assert_called_once_with(mock_schwab.return_value)


def test_provider_is_case_insensitive() -> None:
    with patch("broker.alpaca_client.AlpacaClient") as mock_alpaca:
        build_broker_client({"broker": {"provider": "ALPACA"}})
        mock_alpaca.assert_called_once()


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        build_broker_client({"broker": {"provider": "robinhood"}})
