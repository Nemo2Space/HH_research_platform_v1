# Alpha Research Platform - Complete Documentation

## Project Overview

The Alpha Research Platform is a comprehensive **hedge fund-style** stock analysis and trading signal generation system built for personal portfolio management. It combines AI-powered sentiment analysis, fundamental/technical analysis, backtesting, risk management, trade journaling, performance tracking, and an AI chat assistant.

**Owner:** Alpha
**Location:** `C:\Develop\Latest_2025\HH_research_platform_v1`
**Database:** TimescaleDB in Docker (`alpha-timescaledb`)  
**AI Models:** Qwen3-32B (local via llama.cpp on WSL2)  
**Broker Integration:** Interactive Brokers (IBKR) TWS/Gateway

---

## Hardware Setup

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 9950X (16 cores) |
| **RAM** | 128GB |
| **GPU** | NVIDIA RTX 5090 (32GB VRAM) |
| **OS** | Windows 11 with WSL2 (Ubuntu) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT DASHBOARD                                  │
│  ┌──────────┬─────────┬───────────┬───────────┬──────────┬────────┬────────┐│
│  │ Universe │ Signals │ Deep Dive │ Portfolio │ Backtest │ System │AI Chat ││
│  └──────────┴─────────┴───────────┴───────────┴──────────┴────────┴────────┘│
│                                                                              │
│  AI Chat Tab Features:                                                       │
│  ├── 📊 Daily Briefing (Risk alerts, portfolio health, daily questions)     │
│  ├── 📓 Trade Journal (Thesis tracking, targets, stop-loss)                 │
│  ├── 📈 Performance Tracking (P&L charts, vs SPY comparison)                │
│  ├── 🔍 Stock Analysis (Dropdown + AI report)                               │
│  └── 💬 Interactive Chat (Web search, streaming responses)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   SCREENER    │          │ RISK ANALYZER │          │   BACKTEST    │
│ • Sentiment   │          │ • Concentration│          │ • Engine      │
│ • Technicals  │          │ • Sectors      │          │ • Strategies  │
│ • Gap Analysis│          │ • Beta         │          │ • Metrics     │
│ • Signals     │          │ • Correlations │          └───────────────┘
└───────────────┘          │ • Alerts       │                 │
        │                  └───────────────┘                  │
        └───────────────────────────┼─────────────────────────┘
                                    ▼
                         ┌───────────────────┐
                         │   AI LEARNING     │
                         │ • SignalLearner   │
                         │ • AlphaChat       │
                         │ • Pattern Match   │
                         └───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIMESCALEDB                                          │
│  Tables: screener_scores, trading_signals, fundamentals, prices,            │
│          historical_scores, news_articles, committee_decisions,              │
│          trade_journal, portfolio_snapshots, backtest_results               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  QWEN3-32B    │          │  TOOL SERVER  │          │     IBKR      │
│  (WSL:8090)   │          │  (WSL:7001)   │          │  (TWS:7496)   │
│  AI Model     │          │  Web Search   │          │  Broker API   │
└───────────────┘          └───────────────┘          └───────────────┘
```

---

## Project Structure

```
C:\Develop\Latest_2025\HH_research_platform_v1\
│
├── .env                          # Environment variables (DB, API keys)
├── requirements.txt              # Python dependencies
├── portfolio_tab.py              # Portfolio tab component
│
├── src/
│   ├── __init__.py
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── learning.py           # SignalLearner - historical pattern matching
│   │   ├── chat.py               # AlphaChat - AI assistant with web search
│   │   ├── risk_analyzer.py      # Portfolio risk analysis (concentration, beta, etc.)
│   │   └── performance_tracker.py # Daily P&L tracking, benchmark comparison
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py             # BacktestEngine - runs strategies
│   │   ├── strategies.py         # 9 predefined strategies, OPTIMIZATION_GRIDS
│   │   └── metrics.py            # Sharpe, Sortino, drawdown, etc.
│   │
│   ├── broker/
│   │   ├── __init__.py
│   │   └── ibkr_utils.py         # IBKR connection, positions, orders
│   │
│   ├── committee/
│   │   ├── __init__.py
│   │   ├── agents.py             # FundamentalAgent, SentimentAgent, etc.
│   │   └── coordinator.py        # CommitteeCoordinator - orchestrates voting
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── news.py               # NewsCollector - 6 sources
│   │   ├── insider.py            # InsiderDataFetcher
│   │   └── finviz.py             # FinvizDataFetcher
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py         # get_connection(), get_engine()
│   │   └── repository.py         # Repository class - all DB operations
│   │
│   ├── screener/
│   │   ├── __init__.py
│   │   ├── sentiment.py          # SentimentAnalyzer - dual LLM pipeline
│   │   ├── signals.py            # SignalGenerator, composite scores
│   │   ├── technicals.py         # TechnicalAnalyzer (RSI, MACD, Bollinger)
│   │   ├── gap_analysis.py       # GapAnalyzer
│   │   └── worker.py             # ScreenerWorker - orchestrates all analysis
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py            # get_logger()
│
├── scripts/
│   ├── run_full_screener.py      # CLI: Process all tickers
│   ├── run_committee.py          # CLI: Run committee decisions
│   ├── run_backtest.py           # CLI: Run backtests
│   ├── test_risk_analyzer.py     # Test risk analyzer module
│   ├── test_ai_chat.py           # Test AI chat
│   └── fetch_historical_prices.py # Fetch 3yr prices from Yahoo Finance
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (7 tabs)
│
├── config/
│   └── universe.csv              # List of tickers to track
│
└── migrations/
    └── *.sql                     # Database migration scripts
```

---

## Database Schema

### Core Tables

```sql
-- Stock scores from screener
CREATE TABLE screener_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    sentiment_score DECIMAL,
    sentiment_weighted DECIMAL,
    fundamental_score DECIMAL,
    growth_score DECIMAL,
    dividend_score DECIMAL,
    technical_score DECIMAL,
    gap_score DECIMAL,
    gap_type VARCHAR(50),
    likelihood_score DECIMAL,
    analyst_positivity DECIMAL,
    target_upside_pct DECIMAL,
    insider_signal DECIMAL,
    institutional_signal DECIMAL,
    composite_score DECIMAL,
    total_score DECIMAL,
    article_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Trading signals
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    signal_type VARCHAR(20),      -- STRONG_BUY, BUY, WEAK_BUY, NEUTRAL, WEAK_SELL, SELL, STRONG_SELL
    signal_strength INTEGER,
    signal_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Price data (3 years history)
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    adj_close DECIMAL,
    volume BIGINT,
    UNIQUE(ticker, date)
);

-- Trade Journal (NEW - for thesis tracking)
CREATE TABLE trade_journal (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,           -- BUY, SELL, ADD, TRIM
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4),
    quantity DECIMAL(12,4),
    thesis TEXT,                            -- Why you bought/sold
    target_price DECIMAL(12,4),             -- Your price target
    stop_loss DECIMAL(12,4),                -- Your stop-loss level
    time_horizon VARCHAR(20),               -- short-term, medium-term, long-term
    conviction INTEGER CHECK (conviction BETWEEN 1 AND 10),
    tags VARCHAR(255),
    status VARCHAR(20) DEFAULT 'open',      -- open, closed, stopped_out
    exit_date DATE,
    exit_price DECIMAL(12,4),
    exit_reason TEXT,
    pnl_amount DECIMAL(12,4),
    pnl_percent DECIMAL(8,4),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio Snapshots (NEW - for performance tracking)
CREATE TABLE portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    account_id VARCHAR(50),
    net_liquidation DECIMAL(14,2),
    total_cash DECIMAL(14,2),
    gross_position_value DECIMAL(14,2),
    realized_pnl DECIMAL(14,2),
    unrealized_pnl DECIMAL(14,2),
    daily_pnl DECIMAL(14,2),
    daily_return_pct DECIMAL(8,4),
    cumulative_return_pct DECIMAL(8,4),
    benchmark_value DECIMAL(14,2),
    benchmark_return_pct DECIMAL(8,4),
    alpha_vs_benchmark DECIMAL(8,4),
    position_count INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(snapshot_date, account_id)
);

-- Fundamentals
CREATE TABLE fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    sector VARCHAR(100),
    market_cap DECIMAL,
    pe_ratio DECIMAL,
    forward_pe DECIMAL,
    roe DECIMAL,
    profit_margin DECIMAL,
    revenue_growth DECIMAL,
    dividend_yield DECIMAL,
    beta DECIMAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Backtest results
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100),
    run_date DATE,
    start_date DATE,
    end_date DATE,
    holding_period INTEGER,
    benchmark VARCHAR(10),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5,4),
    total_return DECIMAL(10,4),
    avg_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    benchmark_return DECIMAL(10,4),
    alpha DECIMAL(10,4),
    parameters TEXT,
    returns_by_signal TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Key Components

### 1. Risk Analyzer (`src/ai/risk_analyzer.py`)

Provides hedge fund-style risk analysis:

| Feature | Description |
|---------|-------------|
| **Concentration Risk** | Top 5/10 position weights, flags positions >10% |
| **Sector Exposure** | Gets sector from DB or Yahoo Finance, flags sectors >30% |
| **Portfolio Beta** | Weighted average beta, interprets defensive/aggressive |
| **Correlation Analysis** | Finds correlated pairs (>70%) using 6-month price history |
| **Signal Conflicts** | Finds positions you hold that have SELL signals |
| **P&L Alerts** | Profit taking (>40%), Stop loss review (<-20%) |
| **Risk Score** | A-F grade based on all metrics |
| **Daily Questions** | Generates challenging questions for the user |

**Usage:**
```python
from src.ai.risk_analyzer import RiskAnalyzer

analyzer = RiskAnalyzer()
metrics = analyzer.analyze_portfolio(positions, account_summary)
print(metrics.risk_score)  # e.g., "C"
print(metrics.risk_level)  # e.g., "Medium-High"
```

### 2. Performance Tracker (`src/ai/performance_tracker.py`)

Tracks daily portfolio performance:

| Feature | Description |
|---------|-------------|
| **Daily Snapshots** | Captures portfolio value, P&L, position count |
| **Benchmark Comparison** | Compares to SPY (configurable) |
| **Alpha Calculation** | Your return minus benchmark return |
| **Chart Data** | Normalized returns for plotting |

**Usage:**
```python
from src.ai.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
snapshot_id = tracker.capture_snapshot(positions, account_summary)
summary = tracker.get_summary()
```

### 3. Trade Journal (via Repository)

Tracks your trading thesis:

| Field | Purpose |
|-------|---------|
| **thesis** | Why you bought (AI references this!) |
| **target_price** | Your profit target |
| **stop_loss** | Your exit point if wrong |
| **conviction** | 1-10 confidence scale |
| **time_horizon** | short/medium/long term |

**AI Integration:** When you ask the AI about a stock you own, it says:
> "You bought ORCL at $188 because 'Growth in AI, oversold'. Your target is $205, stop-loss $175."

### 4. AI Chat (`src/ai/chat.py`)

**Features:**
- Stock analysis using database data
- References your trade journal thesis
- Web search for general questions (via Tavily)
- Streaming responses
- Auto-clears context when full
- Portfolio briefing generation

**Configuration:**
```python
@dataclass
class ChatConfig:
    base_url: str = "http://172.23.193.91:8090/v1"
    model: str = "Qwen3-32B-Q6_K.gguf"
    temperature: float = 0.3
    max_tokens: int = 2000
```

### 5. IBKR Integration (`src/broker/ibkr_utils.py`)

**Features:**
- Connect to TWS/Gateway
- Get account summary
- Get positions with live prices
- Get open orders
- Cancel orders
- Close positions

**Usage:**
```python
from src.broker.ibkr_utils import load_ibkr_data_cached

data = load_ibkr_data_cached(account_id="", host="127.0.0.1", port=7496, fetch_live_prices=True)
positions = data['positions']
summary = data['summary']
```

---

## WSL Services

### Terminal 1: Qwen3-32B AI Model (Port 8090)

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/Qwen3-32B-Q6_K.gguf \
  -c 16384 \
  -b 512 \
  -ub 256 \
  -ngl 99 \
  -t 8 \
  --flash-attn on \
  --no-mmap \
  --host 0.0.0.0 \
  --port 8090
```

**Parameters:**
| Param | Value | Description |
|-------|-------|-------------|
| `-c` | 16384 | Context window (tokens) |
| `-b` | 512 | Batch size |
| `-ngl` | 99 | GPU layers (all on GPU) |
| `-t` | 8 | CPU threads |
| `--flash-attn` | on | Flash attention for speed |

**Performance:** ~45-47 tokens/second, 2100+ tokens/second prompt processing

### Terminal 2: Tool Server - Web Search (Port 7001)

```bash
cd ~/ai_tools
source venv/bin/activate
export TAVILY_API_KEY="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"
export BRAVE_API_KEY="BSA8i6DP1kdtTndCDBgBMMKW3LJ48QH"
python tools_api/tool_server.py
```

**Endpoints:**
- `POST /search` - Web search via Tavily (with Brave fallback)

---

## Running the Platform

### 1. Start Database (Docker)

```powershell
docker start alpha-timescaledb
```

### 2. Start AI Model (WSL Terminal 1)

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/Qwen3-32B-Q6_K.gguf \
  -c 16384 \
  -b 512 \
  -ub 256 \
  -ngl 99 \
  -t 8 \
  --flash-attn on \
  --no-mmap \
  --host 0.0.0.0 \
  --port 8090
```

### 3. Start Tool Server (WSL Terminal 2)

```bash
cd ~/ai_tools
source venv/bin/activate
export TAVILY_API_KEY="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"
export BRAVE_API_KEY="BSA8i6DP1kdtTndCDBgBMMKW3LJ48QH"
python tools_api/tool_server.py
```

### 4. Start IBKR TWS/Gateway

- Open TWS or IB Gateway
- Ensure API is enabled on port 7496

### 5. Start Dashboard (Windows PowerShell)

```powershell
cd C:\Develop\Latest_2025\HH_research_platform_v1
.\.venv\Scripts\Activate
streamlit run dashboard/app.py
```

**URL:** http://localhost:8501

---

## Dashboard Tabs

### Tab 1: 📊 Universe
All stocks with scores, signals, fundamentals in a sortable table.

### Tab 2: 📈 Signals
Current trading signals filtered by type.

### Tab 3: 🔍 Deep Dive
Single stock detailed analysis with charts.

### Tab 4: 💼 Portfolio
IBKR portfolio positions with P&L.

### Tab 5: 📉 Backtest
Strategy backtesting with visualization.

### Tab 6: ⚙️ System
System status and controls.

### Tab 7: 🤖 AI Chat
**Main command center with:**

1. **Daily Briefing** (click "Load Daily Briefing")
   - Risk Score (A-F)
   - Portfolio value
   - Alerts (signal conflicts, stop-loss, profit-taking, concentration)
   - Daily Questions (AI challenges your thinking)

2. **Trade Journal** (expandable)
   - Open Positions with thesis
   - Add new entries
   - History with stats

3. **Performance Tracking** (expandable)
   - Capture daily snapshots
   - P&L chart vs SPY
   - Historical returns table

4. **Stock Analysis**
   - Enter ticker → Get full AI report
   - Includes your journal thesis if exists

5. **Chat**
   - Ask anything
   - Web search for current events
   - Streaming responses

---

## CLI Commands

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate

# Run full screener for all tickers
python scripts/run_full_screener.py

# Run committee decisions
python scripts/run_committee.py

# Run backtest
python scripts/run_backtest.py --strategy buy_signals --holding-period 5

# Test risk analyzer
python scripts/test_risk_analyzer.py

# Fetch historical prices
python scripts/fetch_historical_prices.py

# Run dashboard
streamlit run dashboard/app.py
```

---

## Database Access

```powershell
# Access database CLI
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform

# Example queries
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT COUNT(*) FROM prices"
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT * FROM trade_journal"
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT * FROM portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 5"
```

---

## Environment Variables (.env)

```ini
# Database (TimescaleDB)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=alpha_platform
POSTGRES_USER=alpha
POSTGRES_PASSWORD=alpha_secure_2024

# LLM Configuration
LLM_QWEN_BASE_URL=http://172.23.193.91:8090/v1
LLM_QWEN_MODEL=Qwen3-32B-Q6_K.gguf
TOOL_SERVER_URL=http://172.23.193.91:7001

# API Keys
TAVILY_API_KEY=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
BRAVE_API_KEY=BSA8i6DP1kdtTndCDBgBMMKW3LJ48QH
FINNHUB_API_KEY=cm5ea91r01qjc6l4ao60cm5ea91r01qjc6l4ao6g
NEWSAPI_API_KEY=1680c33795ce4486a4b12475ecb2df9d

# Application Settings
LOG_LEVEL=INFO
UNIVERSE_SIZE=100
```

---

## Signal Types

| Signal | Strength | Description |
|--------|----------|-------------|
| STRONG BUY | +5 | Multiple indicators strongly positive |
| BUY | +4 | Positive outlook with technical support |
| WEAK BUY | +3 | Modestly positive indicators |
| INCOME BUY | +3 | High dividend with solid fundamentals |
| GROWTH BUY | +3 | Strong growth with positive sentiment |
| NEUTRAL+ | +1 | Slightly positive |
| NEUTRAL | 0 | Balanced |
| NEUTRAL- | -1 | Slightly negative |
| WEAK SELL | -3 | Modestly negative |
| SELL | -4 | Negative outlook with weakness |
| STRONG SELL | -5 | Multiple indicators strongly negative |

---

## Risk Thresholds (Configurable)

| Metric | Default | Description |
|--------|---------|-------------|
| MAX_POSITION_WEIGHT | 10% | Alert if single position exceeds |
| MAX_SECTOR_WEIGHT | 30% | Alert if sector exceeds |
| PROFIT_TAKING_THRESHOLD | 40% | Suggest profit taking above |
| STOP_LOSS_THRESHOLD | -20% | Suggest review below |
| CORRELATION_WARNING | 0.70 | Warn if pair correlation exceeds |

---

## Best Performing Strategy (from Backtest)

| Metric | Value |
|--------|-------|
| **Strategy** | signal_based (BUY + STRONG_BUY) |
| **Holding Period** | 5 days |
| **Sharpe Ratio** | 1.26 |
| **Win Rate** | 54.2% |
| **Alpha vs SPY** | +40.65% |

---

## Recent Session Summary (December 2024)

### Features Built:
1. ✅ Risk Analyzer (concentration, sector, beta, correlation, alerts)
2. ✅ AI Chat Integration (daily briefing, risk metrics, daily questions)
3. ✅ Trade Journal (thesis tracking, AI references it)
4. ✅ Performance Tracking (daily snapshots, P&L charts, vs SPY)
5. ✅ Navigation fix (sidebar radio buttons for tab persistence)
6. ✅ Context overflow handling (auto-clear old messages)

### Key Files Modified:
- `src/ai/risk_analyzer.py` (NEW)
- `src/ai/performance_tracker.py` (NEW)
- `src/ai/chat.py` (added briefing, journal integration, context handling)
- `src/db/repository.py` (added trade_journal and portfolio_snapshots methods)
- `dashboard/app.py` (added journal UI, performance UI, navigation fix)

---

## Troubleshooting

### AI Model Context Full
Error: `exceed_context_size_error`
- Auto-handled: Old messages cleared automatically
- Manual: Click "Clear" button in chat
- Permanent: Increase `-c` parameter in llama-server (e.g., `-c 32768`)

### IBKR Connection Failed
- Ensure TWS/Gateway is running
- Check API settings: Edit → Global Configuration → API → Settings
- Port should be 7496 (TWS) or 4001 (Gateway)
- Enable "Allow connections from localhost"

### Database Connection Failed
- Ensure Docker is running: `docker ps`
- Start container: `docker start alpha-timescaledb`

### Web Search Not Working
- Ensure tool server is running on WSL port 7001
- Check API keys are exported

---

## Future Enhancements (TODO)

1. **Automated Daily Snapshots** - Scheduled task to capture at market close
2. **Close Journal Entry UI** - Mark trades as closed with exit price
3. **Correlation Matrix Heatmap** - Visual display of position correlations
4. **Scenario Analysis** - "What if market drops 10%?"
5. **Rebalancing Rules** - Automatic trim suggestions
6. **Mobile Access** - Tailscale setup (IP: 100.93.153.6)

---

## Contact & Notes

This platform is designed for personal use by Alpha for managing a real portfolio with IBKR integration. The AI assistant acts as a risk manager, research analyst, and accountability partner - similar to how hedge funds operate but consolidated into a single AI interface.

**Key Philosophy:**
- One AI, one tab, all answers
- Document your thesis (Trade Journal)
- Track your performance (Daily Snapshots)
- Challenge your thinking (Daily Questions)
- Data-driven decisions (Risk Metrics)