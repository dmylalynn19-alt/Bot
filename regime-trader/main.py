"""Entry point for the regime-trader system.

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --stress-test
    python main.py run --once                    # options strategy, single pass now
    python main.py run                            # options strategy, daily at 09:35
    python main.py run --once --strategy regime   # HMM/regime strategy instead

Two independent live/paper strategies share one broker connection (Alpaca
or Schwab - see config `broker.provider` / broker/factory.py) and one
RiskManager:

- "options" (default): core.indicator_signals.IndicatorSignalGenerator reads
  RSI/MACD/SMA-crossover/Bollinger confluence on each symbol, and
  core.options_strategy.OptionsStrategy turns a strong-enough directional
  read into a sized, single-leg call or put (buying only, never
  writing/selling) via the live option chain. Built via
  build_options_pipeline()/run_once_options().
- "regime": the original HMM volatility-regime stock allocator (one
  HMMEngine + StrategyOrchestrator per symbol). Built via
  build_regime_pipeline()/run_once_regime() (build_pipeline()/run_once() are
  kept as aliases for these).

`backtest` is fully wired to backtest.backtester.WalkForwardBacktester,
backtest.performance.PerformanceAnalyzer, and backtest.stress_test.StressTester
(the "regime" strategy only - the options strategy doesn't have a backtester
yet).

Both live strategies operate on DAILY bars (config broker.timeframe) - they
check for a signal and act at most once per day, not intraday.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import yaml

from broker.factory import build_broker_client

logger = logging.getLogger(__name__)


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load settings.yaml and merge in broker credentials from the environment (.env)."""
    from dotenv import load_dotenv

    load_dotenv()
    with open(path) as fh:
        config = yaml.safe_load(fh)
    return config


def build_pipeline(config: dict) -> dict:
    """Alias for build_regime_pipeline (kept for backward compatibility)."""
    return build_regime_pipeline(config)


def build_regime_pipeline(config: dict) -> dict:
    """Wire up the client, data feed, one HMMEngine/StrategyOrchestrator pair
    per symbol, a shared RiskManager, and the OrderExecutor/PositionTracker.

    Returns a dict consumed by run_once_regime()/run(): client, feed,
    risk_manager, order_executor, position_tracker, signal_generators
    (dict[symbol, SignalGenerator]).
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

    client = build_broker_client(config)
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
    """Alias for run_once_regime (kept for backward compatibility)."""
    return run_once_regime(config, pipeline=pipeline)


def run_once_regime(config: dict, pipeline: dict | None = None) -> dict:
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

    pipeline = pipeline if pipeline is not None else build_regime_pipeline(config)
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


def build_options_pipeline(config: dict) -> dict:
    """Wire up the client, data feed, indicator/options strategy engines, a
    shared RiskManager, and the OrderExecutor/PositionTracker.

    Returns a dict consumed by run_once_options()/run(): client, feed,
    risk_manager, order_executor, position_tracker, indicator_generator,
    options_strategy.
    """
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.indicator_signals import IndicatorSignalGenerator
    from core.options_strategy import OptionsStrategy
    from core.risk_manager import RiskManager
    from data.market_data import MarketDataFeed

    broker_cfg = config["broker"]
    client = build_broker_client(config)

    return {
        "client": client,
        "feed": MarketDataFeed(client=client, symbols=broker_cfg["symbols"], timeframe=broker_cfg["timeframe"]),
        "risk_manager": RiskManager(**config["risk"]),
        "order_executor": OrderExecutor(client),
        "position_tracker": PositionTracker(client),
        "indicator_generator": IndicatorSignalGenerator(**config["indicators"]),
        "options_strategy": OptionsStrategy(**config["options_strategy"]),
    }


def run_once_options(config: dict, pipeline: dict | None = None) -> dict:
    """Execute a single options-trading pass.

    For every open option position: checks stop-loss/take-profit thresholds
    on premium P&L and closes it if hit, or if the underlying's indicator
    confluence has reversed against the position's direction. For every
    configured symbol WITHOUT an open option position: computes the
    indicator confluence, and if it's a strong enough directional signal,
    fetches the live option chain, selects a contract, sizes it,
    risk-checks it, and submits it.

    Skips entirely (returns {}) if the market is closed. Returns
    {symbol_or_occ_symbol: OptionRiskDecision | "closed: <reason>"} so the
    caller/CLI can log what happened.
    """
    from core.indicator_signals import Direction
    from core.risk_manager import PortfolioState

    pipeline = pipeline if pipeline is not None else build_options_pipeline(config)
    client = pipeline["client"]
    feed = pipeline["feed"]
    order_executor = pipeline["order_executor"]
    position_tracker = pipeline["position_tracker"]
    risk_manager = pipeline["risk_manager"]
    indicator_generator = pipeline["indicator_generator"]
    options_strategy = pipeline["options_strategy"]
    symbols = config["broker"]["symbols"]

    if not client.is_market_open():
        logger.info("Market is closed - skipping this run")
        return {}

    logger.info("Refreshing market data for %d symbols", len(symbols))
    feed.update()

    account = client.get_account()
    equity = float(account["equity"])
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    all_positions = position_tracker.get_positions()
    option_positions = [p for p in all_positions if p.asset_class == "us_option"]

    portfolio = PortfolioState(
        equity=equity,
        cash=float(account["cash"]),
        buying_power=float(account["buying_power"]),
        positions={p.symbol: p for p in all_positions},
        daily_pnl_pct=position_tracker.get_daily_pnl(),
        peak_equity=position_tracker.get_peak_equity(),
        trades_today=len(client.list_orders(status="closed", after=today_start)),
    )

    results: dict = {}
    open_underlyings: set[str] = set()

    # 1. Manage existing option positions: exit on stop/target/reversal.
    for pos in option_positions:
        try:
            details = client.get_option_contract_details(pos.symbol)
        except Exception:
            logger.exception("Could not look up contract details for %s - skipping exit check", pos.symbol)
            continue
        underlying = details["underlying_symbol"]
        open_underlyings.add(underlying)

        cost_basis = abs(pos.quantity) * pos.avg_entry_price * 100
        pnl_pct = pos.unrealized_pnl / cost_basis if cost_basis else 0.0

        exit_reason = None
        if pnl_pct <= -options_strategy.stop_loss_pct:
            exit_reason = f"stop loss hit ({pnl_pct:.1%})"
        elif pnl_pct >= options_strategy.take_profit_pct:
            exit_reason = f"take profit hit ({pnl_pct:.1%})"
        else:
            try:
                bars = feed.get_cached_bars(underlying)
                indicator_signal = indicator_generator.generate(underlying, bars)
                position_direction = Direction.BULLISH if str(details["type"]).lower() == "call" else Direction.BEARISH
                opposite = Direction.BEARISH if position_direction == Direction.BULLISH else Direction.BULLISH
                if indicator_signal.direction == opposite and indicator_signal.confidence >= options_strategy.min_confidence:
                    exit_reason = f"indicator reversal ({indicator_signal.reasoning})"
            except KeyError:
                pass

        if exit_reason:
            order = order_executor.close_option_position(pos.symbol, abs(pos.quantity), reason=exit_reason)
            results[pos.symbol] = f"closed: {exit_reason}"
            if order is not None:
                logger.info("%s: closed with order %s", pos.symbol, order)

    # 2. Look for new entries on symbols without an open position.
    for symbol in symbols:
        if symbol in open_underlyings:
            continue
        try:
            bars = feed.get_cached_bars(symbol)
        except KeyError:
            logger.warning("No data for %s - skipping", symbol)
            continue

        indicator_signal = indicator_generator.generate(symbol, bars)
        if indicator_signal.direction == Direction.NEUTRAL:
            logger.info("%s: no signal (%s)", symbol, indicator_signal.reasoning)
            continue

        right = "call" if indicator_signal.direction == Direction.BULLISH else "put"
        expiration_gte, expiration_lte = options_strategy.expiration_window()
        chain = client.get_option_chain(symbol, right, expiration_gte.isoformat(), expiration_lte.isoformat())

        option_signal = options_strategy.generate(indicator_signal, chain, equity)
        if option_signal is None:
            logger.info("%s: %s signal did not produce a tradeable contract", symbol, indicator_signal.direction.value)
            continue

        decision = risk_manager.validate_option_signal(option_signal, portfolio)
        results[symbol] = decision
        if not decision.approved:
            logger.info("%s: not approved (%s)", symbol, decision.rejection_reason)
            continue
        if decision.modifications:
            logger.info("%s: approved with modifications: %s", symbol, decision.modifications)

        order = order_executor.execute_option_signal(decision.modified_signal)
        if order is not None:
            logger.info("%s: submitted order %s", symbol, order)

    return results


def run(config: dict, once: bool = False, strategy: str = "options") -> None:
    """Run the live/paper trading loop.

    strategy: "options" (default) runs the indicator-driven directional
    calls/puts strategy; "regime" runs the HMM/volatility-regime stock
    allocator instead.
    once=True: build the pipeline, execute a single pass, and return -
    useful for testing or a manual/cron-triggered run.
    once=False (default): run forever, executing one pass per day at 09:35
    system-local time via the `schedule` library. Ctrl+C to stop.
    """
    if strategy == "regime":
        build_fn, run_fn = build_regime_pipeline, run_once_regime
    elif strategy == "options":
        build_fn, run_fn = build_options_pipeline, run_once_options
    else:
        raise ValueError(f"Unknown strategy '{strategy}'; expected 'options' or 'regime'")

    pipeline = build_fn(config)

    if once:
        run_fn(config, pipeline=pipeline)
        return

    import schedule

    schedule.every().day.at("09:35").do(run_fn, config=config, pipeline=pipeline)
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
    client = build_broker_client(config)
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
    run_parser.add_argument(
        "--strategy",
        choices=["options", "regime"],
        default="options",
        help="'options' (default): indicator-driven directional calls/puts. 'regime': HMM/volatility-regime stock allocator.",
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
        run(config, once=args.once, strategy=args.strategy)
    else:
        run(config)


if __name__ == "__main__":
    main()
