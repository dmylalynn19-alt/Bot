# regime-trader

A regime-detection based systematic trading system for Webull. A Hidden
Markov Model classifies the market into volatility/trend regimes; an
allocation strategy maps each regime to a target exposure and leverage; a
risk manager enforces position, exposure, and drawdown limits; and a signal
generator drives order execution through Webull's official OpenAPI.

**Status:** the HMM engine, feature engineering, regime strategies, risk
manager, walk-forward backtester, performance analytics, and stress testing
are implemented and tested against synthetic data. broker/webull_client.py
wraps Webull's real SDK (market data is confirmed working for US symbols;
order placement/account-balance endpoints are documented by Webull itself
as not yet available for US brokerage accounts - see that file's
docstring). data/market_data.py, broker/order_executor.py,
broker/position_tracker.py, core/signal_generator.py, and monitoring/* are
still interface-only stubs.

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
│   ├── webull_client.py            # Webull OpenAPI wrapper
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
# edit .env with your Webull OpenAPI App Key/App Secret (apply for access
# via Webull's OpenAPI Management console) and account ID

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
