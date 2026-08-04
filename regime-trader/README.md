# regime-trader

A regime-detection based systematic trading system for Alpaca. A Hidden
Markov Model classifies the market into volatility/trend regimes; an
allocation strategy maps each regime to a target exposure and leverage; a
risk manager enforces position, exposure, and drawdown limits; and a signal
generator drives order execution through Alpaca.

**Status:** the full pipeline is implemented and wired end-to-end - HMM
engine, feature engineering, regime strategies, risk manager, walk-forward
backtester, performance analytics, stress testing, the Alpaca broker client,
market data fetching, order execution, position tracking, and the live/paper
daily trading loop (`python main.py run`). `monitoring/*` (structured
logging, dashboard, alerts) is still an interface-only stub - everything
currently logs via the standard `logging` module instead.

This is a **daily-bar** strategy: it checks each symbol's regime and
rebalances at most once per day (see `config/settings.yaml`'s
`broker.timeframe`), not an intraday/tick-by-tick trader. `ALPACA_PAPER=true`
in `.env` is the default and strongly recommended until you've watched it
run correctly for a while.

## Project structure

```
regime-trader/
├── config/
│   ├── settings.yaml              # All configurable parameters
│   └── credentials.yaml.example
├── core/
│   ├── hmm_engine.py               # HMM regime detection engine
│   ├── regime_strategies.py        # Vol-based allocation strategies
│   ├── risk_manager.py             # Position sizing, leverage, drawdown limits
│   └── signal_generator.py         # Combines HMM + strategy into signals
├── broker/
│   ├── alpaca_client.py            # Alpaca API wrapper
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
# Run the walk-forward backtest
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare

# Execute a single live/paper trading pass right now, then exit
python main.py run --once

# Run forever, one trading pass per day at 09:35 system-local time
python main.py run
```

## Testing

```bash
pytest tests/
```
