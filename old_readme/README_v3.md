# Alpha Research Platform - Complete System Documentation

## Overview

A comprehensive stock research and trading signal platform that combines:
- **News Sentiment Analysis** (dual LLM: Qwen-80B + GPT-OSS-20B)
- **Fundamental Analysis** (via yfinance)
- **Technical Analysis**
- **AI Committee Decision System** (multi-agent voting)
- **Historical Learning** (learns from past signal accuracy)
- **Streamlit Dashboard** (6 tabs for full control)

**Owner:** Hasan
**Stack:** Python, PostgreSQL/TimescaleDB, Kafka, Docker, WSL2
**Hardware:** AMD Ryzen 9 9950X, 128GB RAM, RTX 5090

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                        │
│  (Universe | Signals | Deep Dive | Portfolio | Backtest | System)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SCREENER WORKER                            │
│  - News Collection (6 sources)                                  │
│  - LLM Sentiment (Qwen-80B @ 8090, GPT-OSS-20B @ 8091)         │
│  - Fundamental Scoring                                          │
│  - Technical Scoring                                            │
│  - Signal Generation                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TIMESCALEDB (Docker)                         │
│  Tables: screener_scores, trading_signals, fundamentals,        │
│          analyst_ratings, price_targets, prices, news_articles, │
│          committee_decisions, agent_votes, historical_scores,   │
│          insider_transactions, sec_filings, sec_chunks          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema (Key Tables)

### screener_scores
Latest scores per ticker with all metrics:
- sentiment_score, sentiment_weighted
- fundamental_score, growth_score, dividend_score
- technical_score, gap_score, gap_type
- likelihood_score, composite_score, total_score
- insider_signal, institutional_signal
- article_count

### fundamentals
- ticker, date, sector, market_cap, pe_ratio, forward_pe
- pb_ratio, roe, profit_margin, revenue_growth
- dividend_yield, earnings_date, ex_dividend_date

### trading_signals
- ticker, signal_type (STRONG BUY/BUY/WEAK BUY/NEUTRAL/WEAK SELL/SELL/STRONG SELL)
- signal_strength, signal_color, signal_reason
- sentiment_score, fundamental_score, gap_score, likelihood_score

### historical_scores
- Imported from CSV files for backtesting
- Contains return_1d, return_5d, return_10d, return_20d
- signal_correct boolean for accuracy tracking
- 2,238 records currently

### news_articles
- ticker, headline, source, published_at
- credibility_score (1-10 based on source)
- 5,933 articles imported

### committee_decisions
- ticker, verdict, conviction, expected_alpha_bps
- horizon_days, risks

### agent_votes
- ticker, agent_role, buy_prob, confidence, rationale

---

## Project Structure

```
C:\Develop\Latest_2025\HH_research_platform_v1\
│
├── config/
│   └── universe.csv              # 100 stock tickers with sector/industry
│
├── dashboard/
│   └── app.py                    # Main Streamlit dashboard (6 tabs)
│
├── scripts/
│   ├── run_full_screener.py      # Run sentiment analysis on all/specific tickers
│   ├── run_committee.py          # Run AI committee analysis for a ticker
│   ├── manage_universe.py        # Add/remove stocks from universe
│   ├── update_stocks.py          # Quick stock manager (edit lists, run from PyCharm)
│   ├── populate_sectors.py       # Fetch sectors from yfinance
│   ├── populate_dates.py         # Fetch earnings/ex-dividend dates
│   ├── import_historical_scores.py   # Import historical CSV scores
│   ├── import_historical_news.py     # Import historical news CSV files
│   └── ingest_all.py             # Full data ingestion
│
├── src/
│   ├── ai/
│   │   ├── __init__.py
│   │   └── learning.py           # AI learning from historical performance
│   │
│   ├── screener/
│   │   ├── worker.py             # Main screener worker
│   │   └── sentiment.py          # LLM sentiment analysis
│   │
│   ├── db/
│   │   └── connection.py         # Database connection pool
│   │
│   └── utils/
│       └── logging.py            # Logging utilities
│
├── historical_csv/
│   ├── *.csv                     # Historical score CSVs
│   └── news/                     # Historical news CSVs
│       └── (2025, M, D)_TICKER_news_data.csv
│
└── .env                          # Environment variables
```

---

## Key Scripts & Usage

### Daily Operations

```powershell
# Run full screener (all 100 stocks) - click button in dashboard or:
python scripts/run_full_screener.py

# Run screener for specific ticker(s)
python scripts/run_full_screener.py --ticker AAPL
python scripts/run_full_screener.py --ticker AAPL MSFT GOOGL

# Update earnings dates
python scripts/populate_dates.py

# Start dashboard
streamlit run dashboard/app.py
```

### Stock Management

```powershell
# Add stock(s)
python scripts/manage_universe.py --add INTC
python scripts/manage_universe.py --add INTC IBM QCOM

# Remove stock(s)
python scripts/manage_universe.py --remove COIN

# List universe
python scripts/manage_universe.py --list

# Or use update_stocks.py - edit the lists at top and run from PyCharm
```

### Historical Data Import

```powershell
# Import historical scores
python scripts/import_historical_scores.py --dir .\historical_csv\
python scripts/import_historical_scores.py --update-returns

# Import historical news
python scripts/import_historical_news.py --dir .\historical_csv\news\
python scripts/import_historical_news.py --stats
python scripts/import_historical_news.py --analyze

# Populate/sync sectors
python scripts/populate_sectors.py
python scripts/populate_sectors.py --sync-historical
```

---

## Dashboard (app.py) - 6 Tabs

### Tab 1: 📊 Universe
- All 100 stocks with scores and signals
- Columns: Ticker, Sector, Earnings, Ex-Div, Signal, Total, Sentiment, Fundamental, etc.
- Earnings dates highlighted (amber) if within 7 days
- Signal column color-coded (green=buy, red=sell)
- Filters: by signal type, min scores
- Sort by any column
- Download CSV

### Tab 2: 📈 Signals
- Trading signals summary
- Buy/Sell signal lists
- Recent insider activity

### Tab 3: 🔍 Deep Dive
- Select ticker from dropdown
- Run Committee Analysis button (runs AI agents)
- Committee verdict display (BUY/SELL/HOLD with conviction)
- Agent votes with reasoning
- Ticker details (all scores)
- Recent news for ticker
- **AI Historical Learning Insights:**
  - Sector performance (bullish/bearish bias)
  - Signal accuracy by type
  - Similar historical cases
  - Success rate

### Tab 4: 💼 Portfolio
- Placeholder (future: connect to IBKR)

### Tab 5: 📉 Backtest
- Placeholder (future: show historical performance)

### Tab 6: ⚙️ System
- Database table row counts
- Quick commands reference
- LLM status

### Sidebar Control Panel
- 📊 Run Full Screener button
- 📅 Update Dates button
- 🔄 Refresh Dashboard button
- Energy Mode selector
- Last run timestamp

---

## AI Learning Module (src/ai/learning.py)

The `SignalLearner` class provides:

1. **get_sector_insights(sector)** - Returns sector bias, best/worst signals, accuracy
2. **get_similar_historical_signals(ticker, sector, sentiment, fundamental)** - RAG-style retrieval of similar past cases
3. **adjust_signal_confidence(signal_type, sector)** - Calculates confidence multiplier based on history
4. **Performance cache** with 1-hour TTL

### Key Findings from Historical Analysis:
- WEAK_SELL most accurate (60%)
- HIGH sentiment (70+) → +1.13% avg 10D return, 53% accuracy
- LOW sentiment (30-49) → -0.18% avg return, worst performer
- Technology sector: BULLISH bias, +1.31% avg

---

## LLM Configuration

Two local models via llama.cpp:
- **Qwen-Next-80B** @ http://172.23.193.91:8090/v1 (primary sentiment)
- **GPT-OSS-20B** @ http://172.23.193.91:8091/v1 (secondary/fast)

---

## Database Connection

Docker container: `alpha-timescaledb`
```powershell
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform
```

---

## Current Stats

| Metric | Value |
|--------|-------|
| Stocks in Universe | 100 |
| Historical Scores | 2,238 |
| News Articles | 5,933 |
| Date Range (news) | 2011-06-15 to 2025-12-12 |
| Sectors Covered | 11 |

---

## Pending/Future Features

1. **Backtest Tab** - Visualize historical signal performance, equity curves
2. **Portfolio Tab** - Connect to IBKR (Alpha has separate rebalancing project)
3. **Alerts** - Email/Telegram notifications on STRONG BUY/SELL
4. **Scheduled Runs** - Auto-run screener daily at 6 AM
5. **Finviz Institutional Data** - Add institutional ownership %

---

## Session Notes

- User prefers minimal code changes (show exact find/replace)
- User runs from PyCharm primarily
- User has dark mode dashboard
- All buttons should work from dashboard (no terminal commands needed)
- When adding stocks, only process the new stock (not all 100)

---

## Quick Reference

```powershell
# Start everything
streamlit run dashboard/app.py

# Morning routine: Click "Run Full Screener" button in dashboard

# Add a stock (edit update_stocks.py, add to ADD_STOCKS list, run)

# Check database
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT * FROM screener_scores LIMIT 5"
```