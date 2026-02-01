# Alpha Research Platform

## Complete Hedge Fund-Grade Trading System

A comprehensive quantitative trading platform that combines AI-powered sentiment analysis, fundamental analysis, technical indicators, options flow, macro regime detection, and bond trading into a unified signal generation system.

**Author:** Hasan  
**Version:** 2.0  
**Last Updated:** December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Signal Generation Pipeline](#signal-generation-pipeline)
5. [Database Schema](#database-schema)
6. [Module Reference](#module-reference)
7. [Configuration](#configuration)
8. [API Integrations](#api-integrations)
9. [Running the Platform](#running-the-platform)
10. [File Structure](#file-structure)

---

## Overview

### What This Platform Does

1. **Collects Data** from multiple sources (news, fundamentals, prices, options, SEC filings)
2. **Analyzes** using AI models (local Qwen LLM) and quantitative methods
3. **Generates Signals** (BUY/SELL/HOLD) with confidence scores
4. **Detects Market Regime** (Risk-On vs Risk-Off)
5. **Provides Trade Ideas** ranked by AI
6. **Sends Alerts** via Telegram
7. **Displays Everything** in a Streamlit dashboard

### Key Features

| Feature | Description |
|---------|-------------|
| Multi-Factor Scoring | Sentiment + Fundamental + Technical |
| AI Committee | Multiple AI agents vote on each stock |
| Options Flow Analysis | Detect institutional positioning |
| Short Squeeze Detection | Find squeeze candidates |
| Earnings Analysis | Auto-fetch and analyze earnings |
| Macro Regime Detection | Risk-On/Risk-Off with signal adjustments |
| Bond Trading | Yield-based signals for TLT, ZROZ, etc. |
| Trade Ideas Generator | AI-ranked recommendations |
| Telegram Alerts | Real-time notifications |
| IBKR Integration | Live portfolio data |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT DASHBOARD                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Universe │ │ Signals │ │Portfolio│ │Analytics│ │ AI Chat │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Trade Ideas  │  │   AI Chat    │  │   Alerts     │              │
│  │  Generator   │  │   (Qwen)     │  │  (Telegram)  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ANALYTICS LAYER                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │  Options   │ │   Short    │ │  Earnings  │ │   Macro    │       │
│  │   Flow     │ │  Squeeze   │ │  Analyzer  │ │   Regime   │       │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘       │
│        │              │              │              │               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │   Bond     │ │ Technical  │ │   Risk     │ │  Signal    │       │
│  │  Signals   │ │  Analysis  │ │ Dashboard  │ │Performance │       │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘       │
└────────┼──────────────┼──────────────┼──────────────┼───────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     PostgreSQL Database                       │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │  │
│  │  │screener_    │ │ trading_    │ │fundamentals │             │  │
│  │  │  scores     │ │  signals    │ │             │             │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘             │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │  │
│  │  │news_        │ │ committee_  │ │ earnings_   │             │  │
│  │  │ articles    │ │ decisions   │ │  analysis   │             │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Yahoo   │ │ NewsAPI │ │  IBKR   │ │  SEC    │ │  Qwen   │       │
│  │ Finance │ │ Finnhub │ │   TWS   │ │ EDGAR   │ │  LLM    │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Signal Generation System

The platform generates trading signals using a multi-factor approach:

```
Total Score = Sentiment Score (40%) + Fundamental Score (30%) + Technical Score (30%)

Signal:
- STRONG_BUY: Score >= 80
- BUY: Score >= 65
- HOLD: Score 35-65
- SELL: Score <= 35
- STRONG_SELL: Score <= 20
```

#### Sentiment Analysis
- **Sources**: NewsAPI, Finnhub, Google News, Reddit
- **Method**: AI-powered (Qwen LLM) analysis of news articles
- **Output**: Sentiment score 0-100

#### Fundamental Analysis
- **Data**: P/E, PEG, Revenue Growth, Profit Margin, Debt/Equity
- **Method**: Scoring based on value and growth metrics
- **Output**: Fundamental score 0-100

#### Technical Analysis
- **Indicators**: RSI, MACD, Moving Averages, Volume
- **Method**: Rule-based scoring
- **Output**: Technical score 0-100

### 2. AI Committee System

Multiple AI "agents" vote on each stock:

| Agent | Focus | Weight |
|-------|-------|--------|
| Fundamental Agent | Value, growth metrics | 25% |
| Sentiment Agent | News sentiment | 25% |
| Technical Agent | Price patterns | 25% |
| Valuation Agent | Fair value vs price | 25% |

Each agent provides:
- Vote: BUY / HOLD / SELL
- Confidence: 0-100
- Reasoning: Text explanation

Final verdict is weighted consensus.

### 3. Macro Regime Detection

Detects market environment using multiple indicators:

| Indicator | Risk-On Signal | Risk-Off Signal |
|-----------|----------------|-----------------|
| VIX | < 20 | > 25 |
| Yield Curve (10Y-2Y) | Positive | Inverted |
| SPY vs TLT | SPY winning | TLT winning |
| Dollar Index | Falling | Rising |
| Sector Leadership | Growth leading | Defensive leading |
| Market Breadth | RSP > SPY | SPY > RSP |

**Regime Score**: 0-100 (higher = more risk-on)

**Signal Adjustments**:
- Risk-On: Growth stocks +15, Defensive -5
- Risk-Off: Defensive +15, Growth -15

### 4. Options Flow Analysis

Detects institutional positioning via options activity:

```python
Options Score = Call_Volume_Weighted - Put_Volume_Weighted + Unusual_Activity_Bonus

Sentiment:
- VERY_BULLISH: Score >= 70
- BULLISH: Score >= 55
- NEUTRAL: Score 45-55
- BEARISH: Score <= 45
- VERY_BEARISH: Score <= 30
```

### 5. Short Squeeze Detection

Identifies squeeze candidates:

| Factor | High Squeeze Risk |
|--------|-------------------|
| Short % of Float | > 20% |
| Days to Cover | > 5 |
| Borrow Fee | > 50% |
| Recent Price Action | Up on high volume |

### 6. Earnings Analysis

Auto-fetches and analyzes earnings:

| EPS Surprise | Sentiment | Score Adjustment |
|--------------|-----------|------------------|
| ≥ 10% beat | VERY_BULLISH | +15 to +20 |
| 5-10% beat | BULLISH | +10 |
| 0-5% beat | NEUTRAL | +5 |
| 0-5% miss | BEARISH | -5 |
| > 5% miss | VERY_BEARISH | -15 to -20 |

### 7. Bond Trading Module

Generates signals for bond ETFs based on:

| Factor | Weight | Bullish When |
|--------|--------|--------------|
| Yield Trend | 25% | Yields falling |
| Curve Shape | 15% | Inverted (rate cuts expected) |
| Fed Policy | 20% | Dovish |
| Inflation | 15% | Low/falling |
| Macro Regime | 15% | Risk-Off |
| Technicals | 10% | Oversold |

**Supported Instruments**: TLT, ZROZ, EDV, TMF, SHY, IEF

### 8. Trade Ideas Generator

Combines ALL data to rank trade opportunities:

```python
Idea Score = (
    Base_Signal_Score +
    Options_Flow_Bonus (0-15) +
    Squeeze_Bonus (0-12) +
    Earnings_Adjustment (-20 to +20) +
    Macro_Regime_Adjustment (-15 to +15) +
    Technical_Setup_Bonus (0-8) +
    Analyst_Upside_Bonus (0-5) -
    Risk_Penalties
)
```

---

## Database Schema

### Main Tables

```sql
-- Core scoring table
screener_scores (
    id, ticker, date,
    sentiment_score, fundamental_score, technical_score, total_score,
    article_count, options_flow_score, short_squeeze_score,
    options_sentiment, squeeze_risk,
    created_at
)

-- Trading signals
trading_signals (
    id, ticker, date,
    signal_type,      -- BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    signal_strength,  -- 0-100
    created_at
)

-- Fundamental data
fundamentals (
    id, ticker, date,
    sector, pe_ratio, forward_pe, peg_ratio,
    revenue_growth, earnings_growth, profit_margin,
    debt_to_equity,
    created_at
)

-- News articles
news_articles (
    id, ticker, title, content, source, url,
    published_at, sentiment_score, sentiment_label,
    created_at
)

-- AI Committee decisions
committee_decisions (
    id, ticker, date,
    verdict,          -- BUY, SELL, HOLD
    conviction,       -- 0-100
    bull_case, bear_case, recommendation,
    created_at
)

-- Agent votes
agent_votes (
    id, decision_id, agent_name,
    vote, confidence, reasoning,
    created_at
)

-- Earnings analysis
earnings_analysis (
    id, ticker, filing_date,
    eps_actual, eps_estimate, eps_surprise_pct,
    guidance_direction, overall_sentiment,
    sentiment_score, score_adjustment,
    key_highlights, concerns,
    created_at
)

-- Bond signals
bond_signals (
    id, ticker, date,
    signal, score, confidence,
    yield_trend, curve_shape,
    yield_30y, yield_10y, spread_10y_2y,
    current_price, target_price, stop_loss,
    recommendation,
    created_at
)
```

---

## Module Reference

### Analytics Modules (`src/analytics/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `trade_ideas.py` | AI trade recommendations | `generate_trade_ideas()` |
| `macro_regime.py` | Risk-On/Off detection | `get_current_regime()` |
| `bond_signals.py` | Bond ETF signals | `get_bond_signal(ticker)` |
| `earnings_analyzer.py` | Earnings analysis | `analyze_ticker_earnings(ticker)` |
| `options_flow.py` | Options analysis | `analyze_options_flow(ticker)` |
| `short_squeeze.py` | Squeeze detection | `detect_squeeze(ticker)` |
| `signal_performance.py` | Track signal accuracy | `get_performance_summary()` |
| `risk_dashboard.py` | Portfolio risk metrics | `calculate_var()` |
| `portfolio_optimizer.py` | Optimization | `optimize_portfolio()` |
| `technical_analysis.py` | TA indicators | `analyze_technicals(ticker)` |
| `market_context.py` | Market overview | `get_market_context()` |
| `economic_calendar.py` | Economic events | `get_upcoming_events()` |

### Alert Modules (`src/alerts/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `telegram_alerts.py` | Send Telegram alerts | `send_alert()`, `send_daily_summary()` |
| `scheduler.py` | Schedule automated alerts | `start_scheduler()` |

### AI Module (`src/ai/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `chat.py` | AI Chat assistant | `chat(message)` |
| `llm_client.py` | LLM interface | `generate()` |

### Dashboard (`dashboard/`)

| Module | Purpose |
|--------|---------|
| `app.py` | Main Streamlit app |
| `analytics_tab.py` | Analytics sub-tabs |
| `trade_ideas_tab.py` | Trade ideas UI |
| `bond_trading_tab.py` | Bond trading UI |
| `earnings_tab.py` | Earnings UI |
| `macro_regime_tab.py` | Regime UI |

---

## Configuration

### Environment Variables (`.env`)

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/alpha_research

# LLM (Local Qwen)
LLM_ENDPOINT=http://localhost:8090/v1
LLM_MODEL=qwen

# News APIs
NEWS_API_KEY=your_key
FINNHUB_API_KEY=your_key

# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# OpenAI (optional backup)
OPENAI_API_KEY=your_key
```

### Local LLM Setup

The platform uses a local Qwen model via llama.cpp:

```bash
# Running on WSL2 with RTX 5090
# Model: Qwen3-32B-Q6_K
# Endpoint: http://localhost:8090/v1

# Start command (example):
./llama-server -m qwen3-32b-q6_k.gguf -c 8192 --port 8090 -ngl 99
```

---

## API Integrations

| Service | Purpose | Rate Limits |
|---------|---------|-------------|
| Yahoo Finance | Prices, fundamentals, options | Unofficial, be gentle |
| NewsAPI | News articles | 100/day (free) |
| Finnhub | News, fundamentals | 60/min (free) |
| SEC EDGAR | Filings, CIK mapping | Public, no key needed |
| IBKR TWS | Live portfolio | Requires TWS running |
| Telegram | Alerts | 30 msg/sec |

---

## Running the Platform

### 1. Start Database
```bash
# PostgreSQL should be running
pg_isready
```

### 2. Start Local LLM
```bash
# In WSL2
./llama-server -m model.gguf -c 8192 --port 8090 -ngl 99
```

### 3. Start IBKR TWS (optional)
```
Open TWS, enable API on port 7497
```

### 4. Run Dashboard
```bash
cd C:\Develop\Latest_2025\HH_research_platform_v1
streamlit run dashboard/app.py
```

### 5. Start Alert Scheduler (optional)
```bash
python -m src.alerts.scheduler run
```

### Common Commands

```bash
# Run full screener
python scripts/run_full_screener.py

# Run committee analysis
python scripts/run_committee.py --ticker AAPL

# Test Telegram
python -c "from src.alerts.telegram_alerts import send_alert; send_alert('Test!')"

# Check bond signals
python -c "from src.analytics.bond_signals import get_bond_signal; print(get_bond_signal('TLT'))"

# Check regime
python -c "from src.analytics.macro_regime import get_current_regime; print(get_current_regime())"
```

---

## File Structure

```
HH_research_platform_v1/
├── dashboard/
│   ├── app.py                    # Main Streamlit application
│   ├── analytics_tab.py          # Analytics sub-tabs
│   ├── trade_ideas_tab.py        # Trade ideas UI
│   ├── bond_trading_tab.py       # Bond trading UI
│   ├── earnings_tab.py           # Earnings analysis UI
│   ├── macro_regime_tab.py       # Macro regime UI
│   └── economic_calendar_widget.py
│
├── src/
│   ├── ai/
│   │   ├── chat.py               # AI Chat with context
│   │   ├── llm_client.py         # LLM interface
│   │   └── committee.py          # AI Committee system
│   │
│   ├── analytics/
│   │   ├── trade_ideas.py        # Trade ideas generator
│   │   ├── macro_regime.py       # Regime detection
│   │   ├── bond_signals.py       # Bond trading signals
│   │   ├── earnings_analyzer.py  # Earnings analysis
│   │   ├── options_flow.py       # Options flow analysis
│   │   ├── short_squeeze.py      # Squeeze detection
│   │   ├── signal_performance.py # Signal tracking
│   │   ├── risk_dashboard.py     # Risk metrics
│   │   ├── portfolio_optimizer.py# Portfolio optimization
│   │   ├── technical_analysis.py # TA indicators
│   │   ├── market_context.py     # Market overview
│   │   └── economic_calendar.py  # Economic events
│   │
│   ├── alerts/
│   │   ├── telegram_alerts.py    # Telegram notifications
│   │   └── scheduler.py          # Alert scheduler
│   │
│   ├── data/
│   │   ├── news.py               # News fetching
│   │   ├── fundamentals.py       # Fundamental data
│   │   └── prices.py             # Price data
│   │
│   ├── db/
│   │   └── connection.py         # Database connection
│   │
│   └── utils/
│       └── logging.py            # Logging setup
│
├── scripts/
│   ├── run_full_screener.py      # Full analysis script
│   ├── run_committee.py          # Committee analysis
│   └── ingest_all.py             # Data ingestion
│
├── migrations/
│   ├── create_earnings_analysis.sql
│   ├── create_options_flow_tables.sql
│   └── add_flow_squeeze_scores.sql
│
├── data/
│   ├── earnings_calendar_cache.json
│   └── macro_regime_cache.json
│
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Signal Flow Example

Here's how a complete signal is generated for a stock like NVDA:

```
1. DATA COLLECTION
   ├── Fetch 50 news articles (NewsAPI, Finnhub)
   ├── Fetch fundamentals (Yahoo Finance)
   ├── Fetch options chain (Yahoo Finance)
   ├── Fetch price history (Yahoo Finance)
   └── Fetch short interest (various sources)

2. ANALYSIS
   ├── Sentiment Analysis (Qwen LLM)
   │   └── Score: 75/100 (Bullish)
   ├── Fundamental Analysis
   │   └── Score: 60/100 (Good growth, high P/E)
   ├── Technical Analysis
   │   └── Score: 70/100 (Above MAs, RSI 55)
   ├── Options Flow
   │   └── Score: 72/100 (Bullish flow, unusual calls)
   ├── Short Squeeze
   │   └── Score: 25/100 (Low squeeze potential)
   └── Earnings
       └── Last: +15% surprise, VERY_BULLISH

3. SCORING
   Base Score = (75*0.4) + (60*0.3) + (70*0.3) = 69
   + Options bonus: +10
   + Earnings adjustment: +15
   + Macro regime (Risk-On): +10
   - No significant risks
   = Final Score: 94

4. SIGNAL
   Signal: STRONG_BUY
   Strength: 94/100
   Confidence: High

5. TRADE IDEA
   Action: BUY NVDA
   Entry: $140.00
   Target: $160.00 (+14%)
   Stop: $125.00 (-11%)
   Risk/Reward: 1.3x
```

---

## Extending the Platform

### Adding a New Analytics Module

```python
# src/analytics/new_module.py

from src.utils.logging import get_logger
from src.db.connection import get_engine

logger = get_logger(__name__)

class NewAnalyzer:
    def __init__(self):
        self.engine = get_engine()
    
    def analyze(self, ticker: str) -> dict:
        # Your analysis logic
        return {"score": 50, "signal": "HOLD"}

# Convenience function
def analyze_new(ticker: str) -> dict:
    analyzer = NewAnalyzer()
    return analyzer.analyze(ticker)
```

### Adding to Trade Ideas

In `trade_ideas.py`, add to the scoring section:

```python
# In _score_candidates method
new_score = analyze_new(c.ticker)
if new_score['score'] > 70:
    score += 10
    reasons.append("New factor bullish")
```

### Adding New Alert Type

In `telegram_alerts.py`:

```python
def alert_new_event(self, data: dict) -> bool:
    message = f"🆕 <b>New Event</b>\n\n{data['details']}"
    alert = Alert(
        alert_type=AlertType.CUSTOM,
        title="New Event Alert",
        message=message,
        priority="MEDIUM"
    )
    return self.send_alert(alert)
```

---

## Troubleshooting

### Common Issues

**LLM not responding:**
```bash
# Check if running
curl http://localhost:8090/v1/models

# Restart
./llama-server -m model.gguf -c 8192 --port 8090 -ngl 99
```

**Database connection error:**
```bash
# Check PostgreSQL
pg_isready -h localhost -p 5432

# Check connection string in .env
echo $DATABASE_URL
```

**IBKR not connecting:**
- Ensure TWS is running
- Enable API in TWS settings (port 7497)
- Check `IBKR_HOST` and `IBKR_PORT` in .env

**Telegram not sending:**
```python
# Test directly
from src.alerts.telegram_alerts import TelegramAlerter
alerter = TelegramAlerter()
print(f"Enabled: {alerter.enabled}")
alerter.send_message("Test")
```

---

## Performance Notes

- **LLM**: Using Qwen3-32B on RTX 5090 (32GB VRAM)
- **Database**: PostgreSQL with indexes on ticker, date
- **Caching**: File-based cache for earnings, regime (24h)
- **Rate Limits**: Respect API limits, use caching

---

## License

Private - Internal Use Only

---

## Contact

For questions about this platform, refer to this README or the code comments.