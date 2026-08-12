"""Entry point for the regime-trader system.

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --stress-test
    python main.py run --once                     # options strategy, single pass now
    python main.py run                             # options strategy, daily at 09:35
    python main.py run --once --strategy sr        # support/resistance strategy instead
    python main.py run --once --strategy breakout  # consolidation-breakout strategy instead
    python main.py run --once --strategy regime    # HMM/regime strategy instead
    python main.py run --once --mode manual        # ask y/N before every trade, any strategy

Four independent live/paper strategies share one broker connection (Alpaca
or Schwab - see config `broker.provider` / broker/factory.py) and one
RiskManager:

- "options" (default): core.indicator_signals.IndicatorSignalGenerator reads
  RSI/MACD/SMA-crossover/Bollinger confluence on each symbol, and
  core.options_strategy.OptionsStrategy turns a strong-enough directional
  read into a sized, single-leg call or put (buying only, never
  writing/selling) via the live option chain. Built via
  build_options_pipeline()/run_once_options().
- "sr": core.sr_strategy.SupportResistanceStrategy - bounce/rejection
  setups at pivot/key-price levels, confirmed by volume/VWAP/RSI/MACD plus
  market structure. Built via build_sr_pipeline()/run_once_sr().
- "breakout": core.breakout_strategy.BreakoutStrategy - consolidation-
  breakout continuation setups, confirmed by a true (retested) breakout and
  the three-bar entry model. Built via
  build_breakout_pipeline()/run_once_breakout().
- "regime": the original HMM volatility-regime stock allocator (one
  HMMEngine + StrategyOrchestrator per symbol). Built via
  build_regime_pipeline()/run_once_regime() (build_pipeline()/run_once() are
  kept as aliases for these).

"sr" and "breakout" both produce core.sr_strategy.TradeSetup objects (LONG
only for live execution right now - see broker/order_executor.py) and share
the same execution loop (_run_trade_setup_strategy_pass) and risk gate
(RiskManager.validate_trade_setup) - only the strategy object differs.
Since should_exit() needs the ORIGINAL setup a position was opened from
(its level/stop/target), each persists open setups locally via
broker/setup_state.py's TradeSetupStore (config `sr_state_file` /
`breakout_state_file`) - neither broker's API remembers why a position was
opened.

Every strategy accepts `mode="manual"` (CLI `--mode manual`, default
"automatic"): each risk-approved trade is printed (symbol, direction, size,
stop/target, reasoning) and asks for y/N confirmation before submitting,
instead of executing immediately.

`backtest` is fully wired to backtest.backtester.WalkForwardBacktester,
backtest.performance.PerformanceAnalyzer, and backtest.stress_test.StressTester
(the "regime" strategy only - see backtest/sr_backtester.py for a dedicated
trade-by-trade backtester covering "sr" and "breakout" instead; "options"
doesn't have one yet).

All four live strategies operate on whatever bar timeframe you configure
(config broker.timeframe) - they check for a signal and act at most once
per pass, not tick-by-tick.
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


def _confirm_trade(description: str, mode: str) -> bool:
    """Manual/automated trading toggle. In "automatic" mode (default),
    approves every trade without asking. In "manual" mode, prints
    `description` and blocks on a y/N prompt - the trade is only submitted
    if the answer is yes. Applies uniformly across all four strategies.
    """
    if mode != "manual":
        return True
    response = input(f"\n{description}\nExecute this trade? [y/N]: ").strip().lower()
    return response in ("y", "yes")


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


def run_once_regime(config: dict, pipeline: dict | None = None, mode: str = "automatic") -> dict:
    """Execute a single trading pass across every configured symbol.

    For each symbol: refits its HMM on the trailing `hmm.min_train_bars`
    clean feature rows, rebuilds its strategy's vol-rank mapping from that
    fit, generates a risk-approved signal, and submits any resulting
    rebalance order. Skips entirely (returns {}) if the market is closed.
    Returns {symbol: RiskDecision | "skipped (manual)"} so the caller/CLI
    can log what happened.

    `mode`: "automatic" (default) submits every approved rebalance
    immediately; "manual" asks for y/N confirmation first (see
    _confirm_trade).

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
        signal = decision.modified_signal
        target_allocation = signal.position_size_pct * signal.leverage
        if not _confirm_trade(
            f"{symbol}: {signal.direction.value} target allocation {target_allocation:.1%} @ ${signal.entry_price:.2f}\n{signal.reasoning}",
            mode,
        ):
            decisions[symbol] = "skipped (manual)"
            continue

        order = order_executor.execute_signal(signal, current_quantity, equity)
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


def run_once_options(config: dict, pipeline: dict | None = None, mode: str = "automatic") -> dict:
    """Execute a single options-trading pass.

    For every open option position: checks stop-loss/take-profit thresholds
    on premium P&L and closes it if hit, or if the underlying's indicator
    confluence has reversed against the position's direction. For every
    configured symbol WITHOUT an open option position: computes the
    indicator confluence, and if it's a strong enough directional signal,
    fetches the live option chain, selects a contract, sizes it,
    risk-checks it, and submits it.

    `mode`: "automatic" (default) submits every approved trade immediately;
    "manual" asks for y/N confirmation first (see _confirm_trade).

    Skips entirely (returns {}) if the market is closed. Returns
    {symbol_or_occ_symbol: OptionRiskDecision | "closed: <reason>" |
    "entry/exit skipped (manual)"} so the caller/CLI can log what happened.
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
            if not _confirm_trade(f"{pos.symbol}: EXIT ({exit_reason})", mode):
                results[pos.symbol] = f"exit skipped (manual): {exit_reason}"
                continue
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

        approved_signal = decision.modified_signal
        if not _confirm_trade(
            f"{symbol}: BUY {approved_signal.contracts} {approved_signal.right.value.upper()} contracts of "
            f"{approved_signal.occ_symbol} @ ${approved_signal.limit_price:.2f} limit\n{approved_signal.reasoning}",
            mode,
        ):
            results[symbol] = "entry skipped (manual)"
            continue

        order = order_executor.execute_option_signal(approved_signal)
        if order is not None:
            logger.info("%s: submitted order %s", symbol, order)

    return results


def _run_trade_setup_strategy_pass(config: dict, pipeline: dict, mode: str = "automatic") -> dict:
    """Shared execution loop for any strategy producing core.sr_strategy.
    TradeSetup objects - both core.sr_strategy.SupportResistanceStrategy and
    core.breakout_strategy.BreakoutStrategy do, so run_once_sr/
    run_once_breakout both delegate here; only `pipeline["strategy"]`
    differs between them.

    For every symbol WITH an open position (tracked via
    pipeline["setup_store"], since should_exit() needs the ORIGINAL setup a
    position was opened from): checks should_exit() and closes it if
    triggered. A position with no saved setup (state file lost, or a
    pre-existing/manually-opened position) is left alone - there's nothing
    to check should_exit against.

    For every symbol WITHOUT one: generates a setup, risk-checks it
    (RiskManager.validate_trade_setup), and opens it if approved. SHORT
    setups are risk-checked the same as LONG but never actually submitted -
    see broker/order_executor.py's execute_trade_setup.

    `mode`: "automatic" (default) submits every approved trade immediately;
    "manual" asks for y/N confirmation first (see _confirm_trade).

    Skips entirely (returns {}) if the market is closed. Returns
    {symbol: TradeSetupRiskDecision | "closed: <reason>" |
    "entry/exit skipped (manual)"} so the caller/CLI can log what happened.
    """
    from core.risk_manager import PortfolioState

    client = pipeline["client"]
    feed = pipeline["feed"]
    order_executor = pipeline["order_executor"]
    position_tracker = pipeline["position_tracker"]
    risk_manager = pipeline["risk_manager"]
    strategy = pipeline["strategy"]
    setup_store = pipeline["setup_store"]
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
    positions_by_symbol = {p.symbol: p for p in all_positions}

    portfolio = PortfolioState(
        equity=equity,
        cash=float(account["cash"]),
        buying_power=float(account["buying_power"]),
        positions=positions_by_symbol,
        daily_pnl_pct=position_tracker.get_daily_pnl(),
        peak_equity=position_tracker.get_peak_equity(),
        trades_today=len(client.list_orders(status="closed", after=today_start)),
    )

    results: dict = {}

    for symbol in symbols:
        position = positions_by_symbol.get(symbol)

        if position is not None:
            saved_setup = setup_store.load(symbol)
            if saved_setup is None:
                continue
            try:
                bars = feed.get_cached_bars(symbol)
            except KeyError:
                continue

            should_exit, reason = strategy.should_exit(saved_setup, bars)
            if not should_exit:
                continue
            if not _confirm_trade(f"{symbol}: EXIT {saved_setup.direction.value} position ({reason})", mode):
                results[symbol] = f"exit skipped (manual): {reason}"
                continue

            order = order_executor.close_trade_setup_position(saved_setup, abs(position.quantity), reason=reason)
            setup_store.clear(symbol)
            results[symbol] = f"closed: {reason}"
            if order is not None:
                logger.info("%s: closed with order %s", symbol, order)
            continue

        # No open position - look for a new entry.
        try:
            bars = feed.get_cached_bars(symbol)
        except KeyError:
            logger.warning("No data for %s - skipping", symbol)
            continue

        setup = strategy.generate(symbol, bars)
        if setup is None:
            continue

        decision = risk_manager.validate_trade_setup(setup, portfolio)
        results[symbol] = decision
        if not decision.approved:
            logger.info("%s: not approved (%s)", symbol, decision.rejection_reason)
            continue
        if decision.modifications:
            logger.info("%s: approved with modifications: %s", symbol, decision.modifications)

        if not _confirm_trade(
            f"{symbol}: ENTER {setup.direction.value} {decision.shares} shares @ ${setup.entry_price:.2f} "
            f"(stop=${setup.stop_price:.2f}, target=${setup.target_price:.2f})\n{setup.reasoning}",
            mode,
        ):
            results[symbol] = "entry skipped (manual)"
            continue

        order = order_executor.execute_trade_setup(setup, decision.shares)
        if order is not None:
            setup_store.save(setup)
            logger.info("%s: submitted order %s", symbol, order)

    return results


def build_sr_pipeline(config: dict) -> dict:
    """Wire up the client, data feed, the support/resistance strategy, a
    shared RiskManager, the OrderExecutor/PositionTracker, and the
    TradeSetupStore that remembers each open position's originating setup.

    Returns a dict consumed by run_once_sr()/run(): client, feed,
    risk_manager, order_executor, position_tracker, strategy, setup_store.
    """
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from broker.setup_state import TradeSetupStore
    from core.risk_manager import RiskManager
    from core.sr_strategy import SupportResistanceStrategy
    from data.market_data import MarketDataFeed

    broker_cfg = config["broker"]
    client = build_broker_client(config)

    return {
        "client": client,
        "feed": MarketDataFeed(client=client, symbols=broker_cfg["symbols"], timeframe=broker_cfg["timeframe"]),
        "risk_manager": RiskManager(**config["risk"]),
        "order_executor": OrderExecutor(client),
        "position_tracker": PositionTracker(client),
        "strategy": SupportResistanceStrategy(**config["sr_strategy"]),
        "setup_store": TradeSetupStore(config.get("sr_state_file", "sr_open_setups.json")),
    }


def run_once_sr(config: dict, pipeline: dict | None = None, mode: str = "automatic") -> dict:
    """Execute a single support/resistance-strategy pass - see
    _run_trade_setup_strategy_pass for the shared entry/exit logic."""
    pipeline = pipeline if pipeline is not None else build_sr_pipeline(config)
    return _run_trade_setup_strategy_pass(config, pipeline, mode)


def build_breakout_pipeline(config: dict) -> dict:
    """Wire up the client, data feed, the consolidation-breakout strategy, a
    shared RiskManager, the OrderExecutor/PositionTracker, and the
    TradeSetupStore that remembers each open position's originating setup.

    Returns a dict consumed by run_once_breakout()/run(): client, feed,
    risk_manager, order_executor, position_tracker, strategy, setup_store.
    """
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from broker.setup_state import TradeSetupStore
    from core.breakout_strategy import BreakoutStrategy
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
        "strategy": BreakoutStrategy(**config["breakout_strategy"]),
        "setup_store": TradeSetupStore(config.get("breakout_state_file", "breakout_open_setups.json")),
    }


def run_once_breakout(config: dict, pipeline: dict | None = None, mode: str = "automatic") -> dict:
    """Execute a single breakout-strategy pass - see
    _run_trade_setup_strategy_pass for the shared entry/exit logic."""
    pipeline = pipeline if pipeline is not None else build_breakout_pipeline(config)
    return _run_trade_setup_strategy_pass(config, pipeline, mode)


_STRATEGIES = {
    "regime": (build_regime_pipeline, run_once_regime),
    "options": (build_options_pipeline, run_once_options),
    "sr": (build_sr_pipeline, run_once_sr),
    "breakout": (build_breakout_pipeline, run_once_breakout),
}


def run(config: dict, once: bool = False, strategy: str = "options", mode: str = "automatic") -> None:
    """Run the live/paper trading loop.

    strategy: "options" (default) indicator-driven directional calls/puts;
    "sr" support/resistance bounce/rejection setups; "breakout"
    consolidation-breakout continuation setups; "regime" HMM/volatility-
    regime stock allocator. See the module docstring for details on each.
    mode: "automatic" (default) executes every risk-approved trade
    immediately; "manual" asks for y/N confirmation before each one (see
    _confirm_trade) - applies to all four strategies.
    once=True: build the pipeline, execute a single pass, and return -
    useful for testing or a manual/cron-triggered run.
    once=False (default): run forever, executing one pass per day at 09:35
    system-local time via the `schedule` library. Ctrl+C to stop.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'; expected one of {sorted(_STRATEGIES)}")
    build_fn, run_fn = _STRATEGIES[strategy]

    pipeline = build_fn(config)

    if once:
        run_fn(config, pipeline=pipeline, mode=mode)
        return

    import schedule

    schedule.every().day.at("09:35").do(run_fn, config=config, pipeline=pipeline, mode=mode)
    logger.info("Scheduled to run daily at 09:35 (system-local time, mode=%s) - press Ctrl+C to stop", mode)
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
        choices=["options", "sr", "breakout", "regime"],
        default="options",
        help=(
            "'options' (default): indicator-driven directional calls/puts. 'sr': support/resistance "
            "bounce/rejection setups. 'breakout': consolidation-breakout continuation setups. "
            "'regime': HMM/volatility-regime stock allocator."
        ),
    )
    run_parser.add_argument(
        "--mode",
        choices=["automatic", "manual"],
        default="automatic",
        help="'automatic' (default): execute every risk-approved trade immediately. 'manual': ask y/N before each one.",
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
        run(config, once=args.once, strategy=args.strategy, mode=args.mode)
    else:
        run(config)


if __name__ == "__main__":
    main()
