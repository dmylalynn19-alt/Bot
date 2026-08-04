"""Entry point for the regime-trader system.

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --stress-test
    python main.py run --once      # execute a single live/paper pass now and exit
    python main.py run             # run forever, one pass per day at 09:35

`backtest` is fully wired to backtest.backtester.WalkForwardBacktester,
backtest.performance.PerformanceAnalyzer, and backtest.stress_test.StressTester.

`run`/`run_once`/`build_pipeline` wire together broker.alpaca_client.AlpacaClient,
data.market_data.MarketDataFeed, one core.hmm_engine.HMMEngine +
core.regime_strategies.StrategyOrchestrator per symbol, a shared
core.risk_manager.RiskManager, and broker.order_executor.OrderExecutor /
broker.position_tracker.PositionTracker for a real (paper-by-default) daily
trading loop. This strategy operates on DAILY bars (config broker.timeframe) -
it checks the regime and rebalances at most once per day, it does not trade
intraday.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load settings.yaml and merge in Alpaca credentials from the environment (.env)."""
    from dotenv import load_dotenv

    load_dotenv()
    with open(path) as fh:
        config = yaml.safe_load(fh)
    return config


def _build_alpaca_client():
    from broker.alpaca_client import AlpacaClient

    return AlpacaClient(
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
    )


def build_pipeline(config: dict) -> dict:
    """Wire up the client, data feed, one HMMEngine/StrategyOrchestrator pair
    per symbol, a shared RiskManager, and the OrderExecutor/PositionTracker.

    Returns a dict consumed by run_once()/run(): client, feed, risk_manager,
    order_executor, position_tracker, signal_generators (dict[symbol,
    SignalGenerator]).
    """
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.hmm_engine import HMMEngine, RegimeInfo, RegimeLabel
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import RiskManager
    from core.signal_generator import SignalGenerator
    from data.market_data import MarketDataFeed

    broker_cfg = config["broker"]
    hmm_config = config["hmm"]
    strategy_config = config["strategy"]

    client = _build_alpaca_client()
    feed = MarketDataFeed(client=client, symbols=broker_cfg["symbols"], timeframe=broker_cfg["timeframe"])
    risk_manager = RiskManager(**config["risk"])
    order_executor = OrderExecutor(client)
    position_tracker = PositionTracker(client)

    # A single placeholder regime so each symbol's StrategyOrchestrator can
    # construct before its first fit; run_once() replaces this with the real,
    # freshly trained regime_info_ every pass via strategy.update_regime_infos().
    placeholder_regime_infos = {
        RegimeLabel.NEUTRAL: RegimeInfo(0, RegimeLabel.NEUTRAL, 0.0, 0.0, "neutral", 1.0, 0.15, hmm_config["min_confidence"])
    }

    signal_generators = {}
    for symbol in broker_cfg["symbols"]:
        engine = HMMEngine(**hmm_config)
        strategy = StrategyOrchestrator(strategy_config, placeholder_regime_infos, min_confidence=hmm_config["min_confidence"])
        signal_generators[symbol] = SignalGenerator(engine, strategy, risk_manager)

    return {
        "client": client,
        "feed": feed,
        "risk_manager": risk_manager,
        "order_executor": order_executor,
        "position_tracker": position_tracker,
        "signal_generators": signal_generators,
    }


def run_once(config: dict, pipeline: dict | None = None) -> dict:
    """Execute a single trading pass across every configured symbol.

    For each symbol: refits its HMM on the trailing `hmm.min_train_bars`
    clean feature rows, rebuilds its strategy's vol-rank mapping from that
    fit, generates a risk-approved signal, and submits any resulting
    rebalance order. Skips entirely (returns {}) if the market is closed.
    Returns {symbol: RiskDecision} so the caller/CLI can log what happened.

    Safe to call directly (e.g. for a dry run outside the scheduler, or from
    tests with a pre-built `pipeline`).
    """
    from core.risk_manager import PortfolioState
    from data.feature_engineering import build_feature_matrix

    pipeline = pipeline if pipeline is not None else build_pipeline(config)
    client = pipeline["client"]
    feed = pipeline["feed"]
    order_executor = pipeline["order_executor"]
    position_tracker = pipeline["position_tracker"]
    signal_generators = pipeline["signal_generators"]
    min_train_bars = config["hmm"]["min_train_bars"]

    if not client.is_market_open():
        logger.info("Market is closed - skipping this run")
        return {}

    logger.info("Refreshing market data for %d symbols", len(signal_generators))
    feed.update()

    account = client.get_account()
    equity = float(account["equity"])
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    portfolio = PortfolioState(
        equity=equity,
        cash=float(account["cash"]),
        buying_power=float(account["buying_power"]),
        positions={p.symbol: p for p in position_tracker.get_positions()},
        daily_pnl_pct=position_tracker.get_daily_pnl(),
        peak_equity=position_tracker.get_peak_equity(),
        trades_today=len(client.list_orders(status="closed", after=today_start)),
    )

    decisions = {}
    for symbol, signal_generator in signal_generators.items():
        try:
            bars = feed.get_cached_bars(symbol)
        except KeyError:
            logger.warning("No data for %s (update() likely failed) - skipping", symbol)
            continue

        features = build_feature_matrix(bars).dropna()
        if len(features) < min_train_bars:
            logger.warning(
                "Not enough history for %s (%d clean feature rows, need %d) - skipping", symbol, len(features), min_train_bars
            )
            continue

        engine = signal_generator.hmm_engine
        engine.fit(features.iloc[-min_train_bars:])
        signal_generator.strategy.update_regime_infos(engine.regime_info_)

        decision = signal_generator.generate(symbol, bars, portfolio)
        decisions[symbol] = decision

        if not decision.approved:
            logger.info("%s: not approved (%s)", symbol, decision.rejection_reason)
            continue
        if decision.modifications:
            logger.info("%s: approved with modifications: %s", symbol, decision.modifications)

        current_position = position_tracker.get_position(symbol)
        current_quantity = current_position.quantity if current_position else 0.0
        order = order_executor.execute_signal(decision.modified_signal, current_quantity, equity)
        if order is not None:
            logger.info("%s: submitted order %s", symbol, order)

    return decisions


def run(config: dict, once: bool = False) -> None:
    """Run the live/paper trading loop.

    once=True: build the pipeline, execute a single run_once() pass, and
    return - useful for testing or a manual/cron-triggered run.
    once=False (default): run forever, executing one pass per day at 09:35
    system-local time via the `schedule` library. Ctrl+C to stop.
    """
    pipeline = build_pipeline(config)

    if once:
        run_once(config, pipeline=pipeline)
        return

    import schedule

    schedule.every().day.at("09:35").do(run_once, config=config, pipeline=pipeline)
    logger.info("Scheduled to run daily at 09:35 (system-local time) - press Ctrl+C to stop")
    while True:
        schedule.run_pending()
        time.sleep(30)


def run_backtest(args: argparse.Namespace, config: dict) -> None:
    """Fetch historical bars and run the walk-forward backtest for each symbol,
    printing a rich report and writing the standard CSV outputs. Also runs the
    benchmark comparison (--compare) and/or stress-test suite (--stress-test)
    if requested.
    """
    from data.market_data import MarketDataFeed

    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer
    from backtest.stress_test import StressTester
    from core.hmm_engine import HMMEngine, RegimeInfo, RegimeLabel
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import RiskManager

    broker_cfg = config["broker"]
    client = _build_alpaca_client()
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

    run_parser = subparsers.add_parser("run", help="Run the live/paper trading loop")
    run_parser.add_argument(
        "--once", action="store_true", help="Execute a single trading pass immediately and exit, instead of scheduling daily"
    )

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args = parse_args()
    config = load_config(args.config)

    if args.command == "backtest":
        if args.symbols is None:
            args.symbols = config["broker"]["symbols"]
        run_backtest(args, config)
    elif args.command == "run":
        run(config, once=args.once)
    else:
        run(config)


if __name__ == "__main__":
    main()
