# Alpha Research Platform v7.0

## 🎯 Project Overview

The Alpha Research Platform is a comprehensive **hedge fund-style** stock analysis, trading signal generation, and portfolio management system. It combines AI-powered sentiment analysis, fundamental/technical analysis, backtesting, risk management, trade journaling, performance tracking, options flow analysis, and an AI chat assistant with unified portfolio awareness.

**Owner:** Hasan  
**Database:** TimescaleDB (Docker container: `alpha-timescaledb`)  
**AI Models:** Qwen3-32B (local via llama.cpp on WSL2)  
**Broker Integration:** Interactive Brokers (IBKR) TWS/Gateway

---

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 9950X (16 cores) |
| **RAM** | 128GB |
| **GPU** | NVIDIA RTX 5090 (32GB VRAM) |
| **OS** | Windows 11 with WSL2 (Ubuntu) |

---

## ✨ Features

### Core Platform
| Feature | Description |
|---------|-------------|
| **Stock Screener** | Multi-factor scoring (sentiment, fundamental, technical, growth, dividend) |
| **Trading Signals** | AI-generated BUY/SELL/HOLD signals with strength indicators |
| **Committee Analysis** | Multi-agent AI consensus for trading decisions |
| **News Sentiment** | Real-time news collection with LLM-powered sentiment scoring |
| **Price Targets** | Analyst ratings and price target aggregation |

### Portfolio Management
| Feature | Description |
|---------|-------------|
| **IBKR Integration** | Live positions, P&L, orders from Interactive Brokers |
| **ETF Creator** | AI-powered stock selection with target weights |
| **Rebalancer** | Automated trade execution with drift analysis |
| **Unified Context** | Combined view of IBKR + ETF Creator + Rebalancer |

### Analytics & Optimization (NEW in v7.0)
| Feature | Description |
|---------|-------------|
| **Signal Performance Tracker** | Track win rates by signal type (5d, 10d, 30d returns) |
| **Risk Dashboard** | VaR, Beta, Correlation matrix, Sector concentration |
| **Portfolio Optimizer** | Max Sharpe, Min Volatility, Risk Parity optimization |
| **Options Flow Analyzer** | Unusual activity detection, Put/Call ratios, Max Pain |

### AI Assistant
| Feature | Description |
|---------|-------------|
| **AI Chat** | Natural language queries about portfolio and market |
| **Web Search** | Real-time market data via DuckDuckGo |
| **File Upload** | Analyze uploaded documents |
| **Portfolio Sync** | AI sees live IBKR positions + options flow |
| **Streaming Responses** | Real-time token streaming |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT DASHBOARD                                  │
│  ┌──────────┬─────────┬───────────┬───────────┬───────────┬────────┬────────────┐│
│  │ Universe │ Signals │ Deep Dive │ Portfolio │ Analytics │ System │  AI Chat   ││
│  └──────────┴─────────┴───────────┴───────────┴───────────┴────────┴────────────┘│
│                                                                                   │
│  Analytics Tab Features (NEW v7.0):                                              │
│  ├── 📈 Signal Performance (Win rates, returns tracking)                         │
│  ├── ⚠️ Risk Dashboard (VaR, Beta, Correlation, Concentration)                   │
│  ├── 🎯 Portfolio Optimizer (Max Sharpe, Min Vol, Risk Parity)                   │
│  └── 🔮 Options Flow (Unusual activity, P/C ratios, Max Pain)                    │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────────┐          ┌───────────────────┐          ┌───────────────────┐
│  ETF CREATOR      │          │   REBALANCER      │          │   IBKR LIVE       │
│  (External)       │          │   (External)      │          │   (Portfolio Tab) │
│                   │          │                   │          │                   │
│  • Target weights │          │  • Trade execution│          │  • Real positions │
│  • Selection      │          │  • Drift analysis │          │  • Real values    │
│    reasoning      │          │  • Failed trades  │          │  • Open orders    │
│  • Scores         │          │  • Accuracy       │          │  • P&L            │
└─────────┬─────────┘          └─────────┬─────────┘          └─────────┬─────────┘
          │                              │                              │
          └──────────────────────────────┼──────────────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │   UNIFIED PORTFOLIO CONTEXT   │
                         │   + OPTIONS FLOW ANALYSIS     │
                         │                               │
                         │   Per-symbol merge:           │
                         │   • ACTUAL (IBKR live)        │
                         │   • TARGET (ETF Creator)      │
                         │   • LAST REBALANCE (status)   │
                         │   • OPTIONS FLOW (sentiment)  │
                         │   • DRIFT (calculated)        │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │          AI CHAT              │
                         │   (chat.py - AlphaChat)       │
                         │                               │
                         │   Single source of truth      │
                         │   Sees all context            │
                         └───────────────────────────────┘
```

---

## 📁 Project Structure

```
HH_research_platform_v1/
├── config/
│   ├── universe.csv              # Stock universe (181 tickers)
│   └── settings.yaml             # Platform settings
│
├── dashboard/
│   ├── app.py                    # Main Streamlit application
│   ├── portfolio_tab.py          # Portfolio management UI
│   └── analytics_tab.py          # Analytics & Optimization UI (NEW)
│
├── src/
│   ├── ai/
│   │   ├── chat.py               # AI Chat assistant
│   │   ├── unified_portfolio_context.py  # Portfolio + Options context
│   │   └── sentiment.py          # LLM sentiment analysis
│   │
│   ├── analytics/                # NEW in v7.0
│   │   ├── __init__.py
│   │   ├── signal_performance.py # Signal win rate tracking
│   │   ├── risk_dashboard.py     # Risk metrics (VaR, Beta, etc.)
│   │   ├── portfolio_optimizer.py # Portfolio optimization
│   │   └── options_flow.py       # Options flow analyzer
│   │
│   ├── backtest/
│   │   ├── engine.py             # Backtesting engine
│   │   └── strategies.py         # Trading strategies
│   │
│   ├── broker/
│   │   ├── ibkr_client.py        # IBKR TWS connection
│   │   └── ibkr_utils.py         # IBKR helper functions
│   │
│   ├── data/
│   │   ├── news.py               # News collection (multiple sources)
│   │   ├── fundamentals.py       # Fundamental data fetching
│   │   └── prices.py             # Price data management
│   │
│   ├── db/
│   │   ├── connection.py         # Database connection pool
│   │   └── repository.py         # Database CRUD operations
│   │
│   ├── screener/
│   │   ├── scorer.py             # Multi-factor scoring
│   │   ├── signals.py            # Signal generation
│   │   └── unified_portfolio_context.py
│   │
│   └── utils/
│       └── logging.py            # Logging configuration
│
├── scripts/
│   ├── run_full_screener.py      # Run complete screening
│   ├── run_committee.py          # Run AI committee analysis
│   └── ingest_all.py             # Ingest all data
│
├── data/
│   ├── portfolio_*.json          # ETF Creator outputs
│   └── rebalance_*.json          # Rebalancer outputs
│
├── migrations/
│   └── create_options_flow_tables.sql  # Options flow DB schema
│
├── .env                          # Environment configuration
├── docker-compose.yml            # Docker services
└── requirements.txt              # Python dependencies
```

---

## 🗄️ Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `screener_scores` | Daily multi-factor scores per ticker |
| `trading_signals` | Generated trading signals |
| `fundamentals` | Company fundamentals (P/E, margins, etc.) |
| `analyst_ratings` | Analyst buy/sell/hold ratings |
| `price_targets` | Analyst price targets |
| `prices` | Historical price data |
| `news_articles` | Collected news articles |
| `committee_decisions` | AI committee verdicts |
| `agent_votes` | Individual AI agent votes |

### Portfolio Tables

| Table | Description |
|-------|-------------|
| `portfolio_snapshots` | Daily portfolio snapshots |
| `trade_journal` | Trade journal entries |
| `portfolio_history` | Historical portfolio values |

### Options Flow Tables (NEW)

| Table | Description |
|-------|-------------|
| `options_flow_alerts` | Unusual options activity alerts |
| `options_flow_daily` | Daily options flow summaries |

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=alpha_platform
DB_USER=alpha
DB_PASSWORD=your_password

# LLM Configuration
LLM_QWEN_BASE_URL=http://172.23.193.91:8090/v1
LLM_QWEN_MODEL=Qwen3-32B-Q6_K.gguf

# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# News API Keys (optional)
NEWSAPI_KEY=your_key
FINNHUB_API_KEY=your_key
```

### Docker Services

```yaml
# docker-compose.yml services
services:
  alpha-timescaledb:    # TimescaleDB on port 5432
  alpha-redpanda:       # Kafka alternative on port 19092
  alpha-redpanda-console: # Redpanda UI on port 8080
```

---

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd HH_research_platform_v1
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Docker Services
```bash
docker-compose up -d
```

### 5. Run Database Migrations
```powershell
# PowerShell (Windows)
docker exec -i alpha-timescaledb psql -U alpha -d alpha_platform -c "
CREATE TABLE IF NOT EXISTS options_flow_alerts (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    alert_date DATE NOT NULL DEFAULT CURRENT_DATE,
    alert_time TIMESTAMP NOT NULL DEFAULT NOW(),
    alert_type VARCHAR(50) NOT NULL,
    direction VARCHAR(20) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    option_type VARCHAR(10) NOT NULL,
    strike DECIMAL(10, 2) NOT NULL,
    expiry DATE NOT NULL,
    volume INTEGER,
    open_interest INTEGER,
    volume_oi_ratio DECIMAL(10, 2),
    implied_volatility DECIMAL(10, 4),
    stock_price DECIMAL(10, 2),
    distance_from_strike_pct DECIMAL(10, 2),
    days_to_expiry INTEGER,
    description TEXT,
    put_call_volume_ratio DECIMAL(10, 2),
    put_call_oi_ratio DECIMAL(10, 2),
    overall_sentiment VARCHAR(20),
    sentiment_score DECIMAL(10, 2),
    max_pain_price DECIMAL(10, 2),
    notified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_alert UNIQUE (ticker, alert_date, option_type, strike, expiry, alert_type)
);

CREATE TABLE IF NOT EXISTS options_flow_daily (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    scan_date DATE NOT NULL DEFAULT CURRENT_DATE,
    stock_price DECIMAL(10, 2),
    total_call_volume INTEGER,
    total_put_volume INTEGER,
    total_call_oi INTEGER,
    total_put_oi INTEGER,
    put_call_volume_ratio DECIMAL(10, 2),
    put_call_oi_ratio DECIMAL(10, 2),
    avg_call_iv DECIMAL(10, 4),
    avg_put_iv DECIMAL(10, 4),
    iv_skew DECIMAL(10, 4),
    overall_sentiment VARCHAR(20),
    sentiment_score DECIMAL(10, 2),
    max_pain_price DECIMAL(10, 2),
    high_alerts INTEGER DEFAULT 0,
    medium_alerts INTEGER DEFAULT 0,
    low_alerts INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_daily UNIQUE (ticker, scan_date)
);

CREATE INDEX IF NOT EXISTS idx_options_flow_ticker ON options_flow_alerts(ticker);
CREATE INDEX IF NOT EXISTS idx_options_flow_date ON options_flow_alerts(alert_date);
CREATE INDEX IF NOT EXISTS idx_options_daily_ticker ON options_flow_daily(ticker);
CREATE INDEX IF NOT EXISTS idx_options_daily_date ON options_flow_daily(scan_date);
"
```

### 6. Configure .env
```bash
cp .env.example .env
# Edit .env with your settings
```

### 7. Start Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Usage Guide

### Dashboard Tabs

#### 📊 Universe
- View all 181 tickers with multi-factor scores
- Sort/filter by any metric
- Color-coded signals (green=BUY, red=SELL)

#### 📈 Signals
- Top BUY and SELL signals
- Signal strength and reasons
- Quick access to analysis

#### 🔍 Deep Dive
- Select any ticker for detailed analysis
- News sentiment breakdown
- Committee analysis results
- **🔄 Refresh News** button for single-ticker update

#### 💼 Portfolio
- Live IBKR positions
- P&L tracking
- Open orders
- Account summary
- **Syncs to AI Chat** for context-aware responses

#### 📊 Analytics (NEW)

**📈 Signal Performance**
- Track win rates by signal type
- 5-day, 10-day, 30-day returns
- Best/worst performers

**⚠️ Risk Dashboard**
- Portfolio Beta vs SPY
- VaR 95% and 99%
- Correlation matrix heatmap
- Sector concentration
- Diversification score

**🎯 Portfolio Optimizer**
- Max Sharpe Ratio optimization
- Minimum Volatility
- Risk Parity
- Shows optimal weights and suggested trades

**🔮 Options Flow**
- Unusual activity detection
- Put/Call volume ratios
- Max Pain calculation
- **💾 Save to DB** for historical tracking
- **📋 Recent Alerts** from database
- **📈 History** charts over time

#### ⚙️ System
- Database table row counts
- Quick commands reference
- LLM status

#### 🤖 AI Chat
- Natural language portfolio queries
- Web search integration
- File upload analysis
- **Sees options flow sentiment** automatically

---

## 🔮 Options Flow Analysis

### Detection Rules

| Rule | Threshold | Description |
|------|-----------|-------------|
| Volume > OI | 1.0x | New positions being opened |
| High IV | > 50% | Volatility event expected |
| IV Cap | < 200% | Filter distorted data |
| Near-the-money | ±20% ITM, ±30% OTM | Focus on relevant strikes |

### Sentiment Scoring

| Put/Call Ratio | Sentiment |
|----------------|-----------|
| < 0.5 | BULLISH |
| 0.5 - 1.5 | NEUTRAL |
| > 1.5 | BEARISH |

### AI Integration
When you ask the AI about a stock, it automatically includes:
- Put/Call ratios
- Overall sentiment (BULLISH/BEARISH/NEUTRAL)
- Max Pain price
- Unusual activity alerts

---

## 📈 Portfolio Optimization

### Available Strategies

| Strategy | Objective |
|----------|-----------|
| **Max Sharpe** | Maximize risk-adjusted returns |
| **Min Volatility** | Minimize portfolio variance |
| **Risk Parity** | Equal risk contribution from each asset |
| **Max Diversification** | Maximize diversification ratio |

### Constraints
- Min weight: 0%
- Max weight: 5-30% (configurable)
- Min positions: 10 (configurable)

---

## 🛡️ Risk Dashboard Metrics

| Metric | Description |
|--------|-------------|
| **VaR 95%** | Maximum daily loss at 95% confidence |
| **VaR 99%** | Maximum daily loss at 99% confidence |
| **Portfolio Beta** | Sensitivity to market (SPY) |
| **Volatility** | Annualized standard deviation |
| **Diversification Score** | 0-100, higher is better |
| **Effective N** | How many equal-weight stocks your portfolio behaves like |
| **HHI Index** | Concentration measure (lower is better) |
| **Max Drawdown** | Largest peak-to-trough decline |

---

## 🤖 AI Chat Commands

### Portfolio Queries
```
"What's my portfolio performance today?"
"How much AAPL do I own?"
"What's my sector allocation?"
"Show my biggest winners"
```

### Stock Analysis
```
"Analyze NVDA"
"What's the options flow for AMD?"
"Should I buy TSLA?"
"Compare AAPL vs MSFT"
```

### Market Research
```
"What's the latest news on Tesla?"
"Search for Fed rate decision"
"What's moving the market today?"
```

---

## 🔧 Scripts

### Run Full Screener
```bash
python scripts/run_full_screener.py
```
Runs complete multi-factor scoring on all universe tickers.

### Run Committee Analysis
```bash
python scripts/run_committee.py --ticker AAPL
```
Runs AI committee analysis for a specific ticker.

### Ingest All Data
```bash
python scripts/ingest_all.py
```
Fetches fundamentals, prices, news, and analyst data.

---

## 📝 API Reference

### OptionsFlowAnalyzer

```python
from src.analytics.options_flow import OptionsFlowAnalyzer

analyzer = OptionsFlowAnalyzer()

# Analyze single ticker
summary = analyzer.analyze_ticker("AAPL")

# Analyze and save to database
summary = analyzer.analyze_and_save("AAPL")

# Scan multiple tickers
results = analyzer.scan_universe(["AAPL", "MSFT", "NVDA"])

# Scan and save all
results = analyzer.scan_and_save(["AAPL", "MSFT", "NVDA"])

# Get recent alerts from DB
alerts_df = OptionsFlowAnalyzer.get_recent_alerts(days=7, severity="HIGH")

# Get sentiment history
history_df = OptionsFlowAnalyzer.get_sentiment_history("AAPL", days=30)
```

### RiskDashboard

```python
from src.analytics.risk_dashboard import RiskDashboard

dashboard = RiskDashboard(positions, total_value)

# Get all risk metrics
metrics = dashboard.get_full_risk_metrics()

# Get correlation matrix
corr_matrix, corr_metrics = dashboard.calculate_correlation_matrix()

# Get VaR
var_metrics = dashboard.calculate_var([0.95, 0.99])

# Get beta
beta = dashboard.calculate_beta()
```

### PortfolioOptimizer

```python
from src.analytics.portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints

optimizer = PortfolioOptimizer(symbols, current_weights)

# Run optimization
constraints = OptimizationConstraints(max_weight=0.15, min_positions=10)
result = optimizer.optimize("max_sharpe", constraints)

# Compare all strategies
comparisons = optimizer.compare_strategies()

# Get efficient frontier
frontier = optimizer.efficient_frontier(n_points=20)
```

---

## 🐛 Troubleshooting

### Database Connection
```bash
# Test database connection
docker exec -i alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT 1"
```

### IBKR Connection
- Ensure TWS/Gateway is running
- Check API settings: Edit > Global Configuration > API > Settings
- Enable "Enable ActiveX and Socket Clients"
- Port: 7497 (paper) or 7496 (live)

### LLM Connection
- Ensure llama.cpp server is running on WSL2
- Check port 8090 is accessible
- Test: `curl http://172.23.193.91:8090/v1/models`

### Options Flow Errors
- Run database migration first
- Check yfinance is updated: `pip install -U yfinance`

---

## 📜 Version History

| Version | Date | Changes |
|---------|------|---------|
| v7.0 | Dec 2024 | Analytics tab (Signal Performance, Risk Dashboard, Portfolio Optimizer, Options Flow) |
| v6.0 | Dec 2024 | Unified Portfolio Context, AI Chat improvements |
| v5.0 | Nov 2024 | ETF Creator + Rebalancer integration |
| v4.0 | Oct 2024 | IBKR integration, Trade Journal |
| v3.0 | Sep 2024 | Committee Analysis, Multi-agent AI |
| v2.0 | Aug 2024 | News Sentiment, LLM integration |
| v1.0 | Jul 2024 | Initial screener with basic signals |

---

## 📄 License

Proprietary - Personal Use Only

---

## 🙏 Acknowledgments

- **yfinance** - Free market data
- **Streamlit** - Dashboard framework
- **TimescaleDB** - Time-series database
- **Interactive Brokers** - Brokerage integration
- **Qwen** - Local LLM for sentiment analysis