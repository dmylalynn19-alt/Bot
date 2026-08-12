"""Tests for main.py's run() scheduling dispatch (daily vs. interval-based) -
not the pipelines themselves (see tests/test_main_trade_setup_pipelines.py),
just that `interval_minutes` picks the right `schedule` call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import main


def _stub_strategy(monkeypatch) -> tuple[MagicMock, MagicMock]:
    """Swap "options" in main._STRATEGIES for fakes, so run() never touches
    a real broker/pipeline - isolates the scheduling logic being tested."""
    fake_build = MagicMock(return_value=object())
    fake_run = MagicMock()
    monkeypatch.setitem(main._STRATEGIES, "options", (fake_build, fake_run))
    return fake_build, fake_run


def test_run_once_ignores_interval_minutes(monkeypatch) -> None:
    fake_build, fake_run = _stub_strategy(monkeypatch)
    main.run({}, once=True, strategy="options", mode="automatic", interval_minutes=15)
    fake_build.assert_called_once_with({})
    fake_run.assert_called_once_with({}, pipeline=fake_build.return_value, mode="automatic")


def test_run_schedules_daily_when_no_interval_given(monkeypatch) -> None:
    _stub_strategy(monkeypatch)
    fake_job = MagicMock()

    with patch("schedule.every", return_value=fake_job) as mock_every, patch(
        "schedule.run_pending", side_effect=KeyboardInterrupt
    ), patch("time.sleep"):
        with pytest.raises(KeyboardInterrupt):
            main.run({}, once=False, strategy="options", mode="automatic", interval_minutes=None)

    mock_every.assert_called_once_with()
    fake_job.day.at.assert_called_once_with("09:35")


def test_run_schedules_by_interval_when_given(monkeypatch) -> None:
    _stub_strategy(monkeypatch)
    fake_job = MagicMock()

    with patch("schedule.every", return_value=fake_job) as mock_every, patch(
        "schedule.run_pending", side_effect=KeyboardInterrupt
    ), patch("time.sleep"):
        with pytest.raises(KeyboardInterrupt):
            main.run({}, once=False, strategy="options", mode="automatic", interval_minutes=15)

    mock_every.assert_called_once_with(15)
    fake_job.minutes.do.assert_called_once()


def test_cli_parses_interval_minutes() -> None:
    args = main.parse_args(["run", "--strategy", "options", "--interval-minutes", "15"])
    assert args.interval_minutes == 15


def test_cli_interval_minutes_defaults_to_none() -> None:
    args = main.parse_args(["run", "--once"])
    assert args.interval_minutes is None
