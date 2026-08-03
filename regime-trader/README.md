# regime-trader

A regime-detection based systematic trading system for Alpaca. A Hidden
Markov Model classifies the market into volatility/trend regimes; an
allocation strategy maps each regime to a target exposure and leverage; a
risk manager enforces position, exposure, and drawdown limits; and a signal
generator drives order execution through Alpaca.

**Status:** skeleton only. Module interfaces (classes, type hints,
docstrings) are in place; no trading logic is implemented yet.

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
python main.py
```

## Testing

```bash
pytest tests/
```
