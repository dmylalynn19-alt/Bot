# regime-trader

A systematic trading system for Alpaca (equities + options) with two
independent live/paper strategies sharing one risk manager:

- **options** (default, `python main.py run`): reads RSI/MACD/SMA-crossover/
  Bollinger Band confluence on each symbol and, when enough of the four
  agree, buys a single-leg call or put (never sells/writes) sized by defined
  premium risk, with a stop-loss/take-profit/indicator-reversal exit.
- **regime** (`python main.py run --strategy regime`): a Hidden Markov Model
  classifies each symbol into a volatility/trend regime and an allocation
  strategy maps that to a target stock exposure and leverage.

Both go through the same independent, P&L-based risk manager (circuit
breakers, position/exposure caps, correlation checks) before anything is
ever submitted to Alpaca.

**Status:** the full pipeline is implemented and wired end-to-end for both
strategies - HMM engine, feature engineering, regime strategies, technical
indicator confluence, options contract selection/sizing, risk manager
(stock and options), walk-forward backtester (regime strategy only),
performance analytics, stress testing, the Alpaca broker client (via
`alpaca-py`, covering both equities and options), market data fetching,
order execution, position tracking, and the live/paper daily trading loop.
`monitoring/*` (structured logging, dashboard, alerts) is still an
interface-only stub - everything currently logs via the standard `logging`
module instead.

This is a **daily-bar** system: each strategy checks for a signal and acts
at most once per day (see `config/settings.yaml`'s `broker.timeframe`), not
an intraday/tick-by-tick trader. `ALPACA_PAPER=true` in `.env` is the
default and strongly recommended until you've watched it run correctly for
a while. Options trading also requires options approval on your Alpaca
account (check `options_approved_level`/`options_trading_level` on
`AlpacaClient.get_account()`).

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
│   ├── risk_manager.py             # Position sizing, leverage, drawdown limits (stock + options)
│   └── signal_generator.py         # Combines HMM + strategy into signals (regime strategy)
├── broker/
│   ├── alpaca_client.py            # Alpaca API wrapper (equities + options, via alpaca-py)
│   ├── order_executor.py           # Order placement, modification, cancellation
│   └── position_tracker.py         # Track open positions, P&L
├── data/
│   ├── market_data.py              # Real-time and historical data fetching
│   └── feature_engineering.py      # Technical indicators, feature computation
├── monitoring/
│   ├── logger.py                   # Structured logging
│   ├── dashboard.py
│   └── alerts.py
├── backtest/
│   ├── backtester.py               # Walk-forward allocation backtester
│   ├── performance.py              # Sharpe, drawdown, regime breakdown, benchmarks
│   └── stress_test.py              # Crash injection, gap simulation
├── tests/
│   ├── test_hmm.py
│   ├── test_look_ahead.py          # Verify no look-ahead bias
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_orders.py
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
# edit .env with your Alpaca API credentials

cp config/credentials.yaml.example config/credentials.yaml
# edit config/credentials.yaml if you prefer YAML-based credentials
```

Review and adjust `config/settings.yaml` before running.

## Running

```bash
# Run the walk-forward backtest (regime strategy only)
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare

# Options strategy (default) - single pass right now, then exit
python main.py run --once

# Options strategy - run forever, one pass per day at 09:35 system-local time
python main.py run

# Regime/stock strategy instead
python main.py run --once --strategy regime
```

## Testing

```bash
pytest tests/
```
