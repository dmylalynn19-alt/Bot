"""Entry point for the regime-trader system.

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --stress-test

`backtest` is fully wired to backtest.backtester.WalkForwardBacktester,
backtest.performance.PerformanceAnalyzer, and backtest.stress_test.StressTester.
It still depends on broker.alpaca_client.AlpacaClient and
data.market_data.MarketDataFeed for historical bars, both of which remain
unimplemented stubs - so `backtest` will raise NotImplementedError at the
data-fetch step until those are built. Live trading (`run`/`build_pipeline`)
is out of scope here and remains a stub too.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import yaml


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load settings.yaml and merge in Alpaca credentials from the environment (.env)."""
    from dotenv import load_dotenv

    load_dotenv()
    with open(path) as fh:
        config = yaml.safe_load(fh)
    return config


def build_pipeline(config: dict):
    """Wire up the data feed, HMM engine, strategy, risk manager, and executor."""
    raise NotImplementedError


def run(config: dict) -> None:
    """Run the live trading loop."""
    raise NotImplementedError


def run_backtest(args: argparse.Namespace, config: dict) -> None:
    """Fetch historical bars and run the walk-forward backtest for each symbol,
    printing a rich report and writing the standard CSV outputs. Also runs the
    benchmark comparison (--compare) and/or stress-test suite (--stress-test)
    if requested.
    """
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataFeed

    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer
    from backtest.stress_test import StressTester
    from core.hmm_engine import HMMEngine, RegimeInfo, RegimeLabel
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import RiskManager

    broker_cfg = config["broker"]
    client = AlpacaClient(
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
    )
    feed = MarketDataFeed(client=client, symbols=args.symbols, timeframe=broker_cfg["timeframe"])

    hmm_config = config["hmm"]
    strategy_config = config["strategy"]
    backtest_config = config["backtest"]

    # A single placeholder regime so StrategyOrchestrator can construct before
    # the first in-sample fit; WalkForwardBacktester.step() replaces this with
    # the real, freshly trained regime_info_ on every window via
    # strategy.update_regime_infos().
    placeholder_regime_infos = {
        RegimeLabel.NEUTRAL: RegimeInfo(0, RegimeLabel.NEUTRAL, 0.0, 0.0, "neutral", 1.0, 0.15, hmm_config["min_confidence"])
    }

    for symbol in args.symbols:
        bars = feed.get_historical_bars(symbol, start=args.start, end=args.end)

        engine = HMMEngine(**hmm_config)
        strategy = StrategyOrchestrator(strategy_config, placeholder_regime_infos, min_confidence=hmm_config["min_confidence"])
        # RiskManager.validate_signal is built for live, multi-symbol,
        # per-trade order validation (stop-distance sizing, correlation/
        # sector caps, buying power, ...); this single-symbol allocation-based
        # backtester doesn't call into it yet - see backtest/backtester.py.
        # Constructed here anyway so its circuit breakers are exercised
        # end to end once that wiring is added.
        risk_manager = RiskManager(**config["risk"])
        backtester = WalkForwardBacktester(
            hmm_engine=engine,
            strategy=strategy,
            risk_manager=risk_manager,
            step_size=backtest_config["step_size"],
            symbol=symbol,
        )
        result = backtester.run(bars, start=args.start, end=args.end)

        analyzer = PerformanceAnalyzer(risk_free_rate=backtest_config["risk_free_rate"])
        benchmark_comparison = None
        if args.compare:
            benchmark_comparison = analyzer.compare_to_benchmarks(
                result.equity_curve, bars, backtester.initial_cash, result.trades
            )

        analyzer.print_report(result.equity_curve, result.trades, result.regime_history, benchmark_comparison)
        analyzer.export_csv(
            os.path.join("backtest_output", symbol),
            result.equity_curve,
            result.trades,
            result.regime_history,
            benchmark_comparison,
        )

        if args.stress_test:
            tester = StressTester(backtester)
            report = tester.run_scenarios(bars, args.start, args.end)
            print(f"\nStress test results for {symbol}:")
            for scenario, metrics in report.items():
                print(f"  {scenario}: {metrics}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="regime-trader")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings.yaml")
    subparsers = parser.add_subparsers(dest="command")

    backtest_parser = subparsers.add_parser("backtest", help="Run the walk-forward allocation backtest")
    backtest_parser.add_argument(
        "--symbols", nargs="+", default=None, help="Symbols to backtest (default: config broker.symbols)"
    )
    backtest_parser.add_argument("--start", default="2019-01-01", help="Backtest start date (YYYY-MM-DD)")
    backtest_parser.add_argument(
        "--end", default=datetime.now(timezone.utc).date().isoformat(), help="Backtest end date (YYYY-MM-DD)"
    )
    backtest_parser.add_argument("--compare", action="store_true", help="Also run the benchmark comparison suite")
    backtest_parser.add_argument(
        "--stress-test", dest="stress_test", action="store_true", help="Also run the stress-test suite (crash/gap/regime-shuffle)"
    )

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_config(args.config)

    if args.command == "backtest":
        if args.symbols is None:
            args.symbols = config["broker"]["symbols"]
        run_backtest(args, config)
    else:
        run(config)


if __name__ == "__main__":
    main()
