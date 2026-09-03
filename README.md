# Market Intelligence Pro V7 — Production Candidate

This version is intended as the most complete research version of the project.

## Core engines
- Separate Long and Short ranking models
- Technical / Momentum / Volatility / Liquidity
- Fundamentals / Growth / Quality / Balance Sheet / FCF
- Absolute and sector-relative valuation
- Earnings surprise / revisions proxy / earnings blackout
- News sentiment
- Insider and institutional proxies
- Options IV and Put/Call OI
- SPY/QQQ/VIX market regime
- Market regime drift penalty
- Confidence score and data-health circuit breaker
- Sector diversification guard

## Risk & portfolio analytics
- ATR stop/targets
- Position sizing and per-position cap
- Correlation matrix
- Beta vs SPY
- 95% historical VaR
- 95% historical CVaR
- Monte Carlo scenario simulation
- Portfolio annualized volatility
- Portfolio Lab weighting

## Validation
- Backtest with trading cost
- CAGR / Max DD / Sharpe / Sortino / Win Rate / Profit Factor
- Walk-forward out-of-sample validation
- Benchmark alpha tracking at 1D / 5D / 21D / 63D
- Daily historical picks database

## Reliability
- Retry logic
- Logging
- SQLite WAL mode
- Schema migrations
- Data health score
- Circuit breaker
- Health check
- Automated unit tests
- Dockerfile
- GitHub Actions
- Daily runner

## Run
```bash
pip install -r requirements.txt
python tests.py
python healthcheck.py
streamlit run app.py
```

## Daily scan
```bash
python daily_runner.py
```

The daily runner refuses to save picks when the data-health circuit breaker fails.

## AI
Works without external AI.
Optional OpenAI analysis uses `OPENAI_API_KEY` and `OPENAI_MODEL` from environment variables.

## Important
No financial system can be guaranteed 100% error-free or profitable.
The largest remaining production dependency is the market-data provider. yfinance is suitable for research/prototyping but not an institutional SLA-backed feed.

Before using real money, paper-trade and collect enough historical live signals to evaluate alpha, drawdown, stability, and data quality.
