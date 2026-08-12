# regime-trader

A systematic trading system for Alpaca and Schwab (equities + options) with
four independent live/paper strategies sharing one risk manager and one
broker connection:

- **sr** (`python main.py run --strategy sr`): support/resistance
  bounce/rejection setups. Detects zones from swing pivots plus previous-
  day/pre-market highs-lows and order blocks, and only trades a
  bounce/rejection off one when volume, VWAP, RSI, MACD, and the entry
  timeframe's own market structure (a break-of-structure or
  market-structure-shift event in the trade's favor) all confirm it.
  Optionally vetoed by a higher timeframe's trend if you pass one in.
- **breakout** (`python main.py run --strategy breakout`): consolidation-
  breakout continuation setups - the complementary setup type to `sr`
  (trading the break *through* a level, not a reversal at one). Waits for
  a tight consolidation range, requires a *true* breakout (the broken edge
  retested and held, not just poked through), and triggers entry via the
  three-bar model (a fast lead candle, a shallow pullback, a confirmation
  break), with an optional "last leg" momentum-acceleration filter.
- **options** (default, `python main.py run`): reads RSI/MACD/SMA-crossover/
  Bollinger Band confluence on each symbol and, when enough of the four
  agree, buys a single-leg call or put (never sells/writes) sized by defined
  premium risk, with a stop-loss/take-profit/indicator-reversal exit.
  Defaults to 0-2 day expirations (`options_strategy.max_days_to_expiration`)
  - same-day/next-day contracts move (and decay) fast, so running this
  strategy **requires** `--interval-minutes` (see below), not the once-daily
  default - a same-day option can lose most of its value in hours, well
  before a once-a-day check would catch it.
- **regime** (`python main.py run --strategy regime`): a Hidden Markov Model
  classifies each symbol into a volatility/trend regime and an allocation
  strategy maps that to a target stock exposure and leverage.

All four go through the same independent, P&L-based risk manager (circuit
breakers, position/exposure caps, correlation checks, and - for `sr`/
`breakout` - risk-based position sizing off the setup's own stop distance)
before anything is ever submitted to the broker.

**Manual or automated:** every strategy accepts `--mode manual` (default is
`automatic`) - in manual mode, each risk-approved trade is printed (symbol,
direction, size, stop/target, reasoning) and asks for a y/N confirmation
before submitting, instead of executing immediately.

**Emergency exit:** `python main.py close --symbol AAPL` (or `--all`)
flattens a position with a single market order right now - no waiting for
the next scheduled check, no strategy exit logic, no confirmation prompt.
Works for stock or option positions, long or short, regardless of which
strategy (or none) opened it.

**Broker:** set `broker.provider` in `config/settings.yaml` to `alpaca`
(default) or `schwab` - see broker/factory.py. Webull is data-only, not a
trading option here: its official API does not support order placement for
US brokerage accounts (verified by inspecting the SDK directly) - Webull
*is* usable as a broker through TradingView's own trading panel or a
third-party bridge like TradersPost, but neither exposes an API this bot's
own code can call into.

**Status:** the full pipeline is implemented and wired end-to-end for all
four strategies - HMM engine, feature engineering, regime strategies,
technical indicator confluence, support/resistance + market structure +
key-level detection, consolidation/breakout classification, options
contract selection/sizing, risk manager (stock, options, and trade-setup
variants), a walk-forward backtester (regime strategy) plus a dedicated
trade-by-trade backtester (`sr`/`breakout`), performance analytics, stress
testing, both broker clients (Alpaca via `alpaca-py`, Schwab via
`schwab-py`, covering equities and options), market data fetching, order
execution, position tracking, and the live/paper daily trading loop.
`monitoring/*` (structured logging, dashboard, alerts) is still an
interface-only stub - everything currently logs via the standard `logging`
module instead. SHORT setups (`sr`/`breakout`) are detected and risk-
checked but not yet auto-executed live - see `broker/order_executor.py`.

This system checks for a signal and acts at most once per configured bar
(see `config/settings.yaml`'s `broker.timeframe`) - not an intraday/
tick-by-tick trader, though `sr`/`breakout` work on any timeframe you feed
them (5-minute bars for day trading, daily for swing trading). Paper
trading is the default and strongly recommended until you've watched it run
correctly for a while (`ALPACA_PAPER=true` in `.env`; Schwab has no
separate paper endpoint - use a Schwab paperMoney account instead). Options
trading also requires options approval on your account.

## Project structure

```
regime-trader/
├── config/
│   ├── settings.yaml              # All configurable parameters
│   └── credentials.yaml.example
├── core/
│   ├── hmm_engine.py               # HMM regime detection engine
│   ├── regime_strategies.py        # Vol-based allocation strategies
│   ├── indicator_signals.py        # RSI/MACD/SMA/Bollinger confluence signals
│   ├── options_strategy.py         # Directional call/put contract selection + sizing
│   ├── support_resistance.py       # Pivot-based support/resistance zone detection
│   ├── market_structure.py         # Trend/BOS/MSS classification, trend-leg speed
│   ├── key_levels.py                # Previous-day/pre-market highs-lows, order blocks
│   ├── sr_strategy.py               # Support/resistance bounce/rejection setups
│   ├── consolidation.py             # Consolidation range detection, true/false breakout
│   ├── candle_patterns.py           # Candle speed/momentum, three-bar entry model
│   ├── breakout_strategy.py         # Consolidation-breakout continuation setups
│   ├── risk_manager.py             # Position sizing, leverage, drawdown limits (all strategies)
│   └── signal_generator.py         # Combines HMM + strategy into signals (regime strategy)
├── broker/
│   ├── base.py                     # BrokerClient interface every broker satisfies
│   ├── alpaca_client.py            # Alpaca API wrapper (equities + options, via alpaca-py)
│   ├── schwab_client.py            # Schwab Trader API wrapper (via schwab-py)
│   ├── schwab_adapter.py           # Normalizes Schwab's shapes onto BrokerClient
│   ├── factory.py                  # Builds whichever broker is configured
│   ├── order_executor.py           # Order placement, modification, cancellation
│   ├── position_tracker.py         # Track open positions, P&L
│   └── setup_state.py              # Persists sr/breakout TradeSetups across runs
├── data/
│   ├── market_data.py              # Real-time and historical data fetching
│   └── feature_engineering.py      # Technical indicators, feature computation
├── monitoring/
│   ├── logger.py                   # Structured logging
│   ├── dashboard.py
│   └── alerts.py
├── backtest/
│   ├── backtester.py               # Walk-forward allocation backtester (regime strategy)
│   ├── sr_backtester.py            # Trade-by-trade backtester (sr/breakout strategies)
│   ├── performance.py              # Sharpe, drawdown, regime breakdown, benchmarks
│   └── stress_test.py              # Crash injection, gap simulation
├── tests/                          # One file per module above, plus test_look_ahead.py
├── main.py                         # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your Alpaca and/or Schwab API credentials

cp config/credentials.yaml.example config/credentials.yaml
# edit config/credentials.yaml if you prefer YAML-based credentials
```

Set `broker.provider` in `config/settings.yaml` to pick Alpaca or Schwab.
Schwab's first run needs an interactive one-time login (copy a URL into
your browser, paste back the URL you land on) - after that, its token is
cached and refreshes itself silently.

Review and adjust `config/settings.yaml` before running - in particular
`broker.symbols`/`broker.timeframe`, and the `sr_strategy`/
`breakout_strategy` sections if you're using those.

## Running

```python
# Backtest sr/breakout on one symbol's history (backtest/sr_backtester.py) -
# no CLI subcommand yet, run it as a script:
from broker.factory import build_broker_client
from backtest.sr_backtester import SRBacktester
from core.sr_strategy import SupportResistanceStrategy
from data.market_data import MarketDataFeed
import main

config = main.load_config()
client = build_broker_client(config)
bars = MarketDataFeed(client, ["AAPL"], "1Day").get_historical_bars("AAPL", "2022-01-01", "2024-12-31")
result = SRBacktester(SupportResistanceStrategy(**config["sr_strategy"])).run("AAPL", bars)
print(result.summary())
```

```bash
# Walk-forward backtest (regime strategy)
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare

# Support/resistance strategy - single pass now, then exit
python main.py run --once --strategy sr

# Breakout strategy - run forever, one pass per bar-close
python main.py run --strategy breakout

# Options strategy (default) - single pass now
python main.py run --once

# Options strategy, running for real: checks every 15 minutes during market
# hours (REQUIRED for the default 0-2 DTE window - see above)
python main.py run --strategy options --interval-minutes 15

# Regime/stock strategy instead
python main.py run --once --strategy regime

# Any strategy, asking for confirmation before every trade instead of executing automatically
python main.py run --once --strategy sr --mode manual

# Emergency exit - flatten one position, or everything, right now
python main.py close --symbol AAPL
python main.py close --all
```

## Testing

```bash
pytest tests/
```
