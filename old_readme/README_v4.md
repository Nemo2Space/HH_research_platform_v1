# Alpha Research Platform - Complete Documentation

## Project Overview

The Alpha Research Platform is a comprehensive stock analysis and trading signal generation system. It combines AI-powered sentiment analysis, fundamental/technical analysis, backtesting, and an AI chat assistant to help make informed trading decisions.

**Owner:** Hasan
**Location:** `C:\Develop\Latest_2025\HH_research_platform_v1`
**Database:** TimescaleDB in Docker (`alpha-timescaledb`)
**AI Models:** Qwen3-32B (local via llama.cpp)

---

Run model on wsl
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/Qwen3-32B-Q6_K.gguf \
  -c 4096 \
  -b 512 \
  -ub 256 \
  -ngl 99 \
  -t 8 \
  --flash-attn on \
  --no-mmap \
  --host 0.0.0.0 \
  --port 8090

Run sarching tool on wsl
cd ~/ai_tools
source venv/bin/activate
export TAVILY_API_KEY="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"
export BRAVE_API_KEY="BSA8i6DP1kdtTndCDBgBMMKW3LJ48QH"
python tools_api/tool_server.py

## Hardware Setup

- **CPU:** AMD Ryzen 9 9950X (16 cores)
- **RAM:** 128GB
- **GPU:** NVIDIA RTX 5090 (32GB VRAM)
- **OS:** Windows 11 with WSL2 (Ubuntu)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT DASHBOARD                              │
│  ┌─────────┬─────────┬──────────┬───────────┬──────────┬────────┬──────┐│
│  │Universe │ Signals │Deep Dive │ Portfolio │ Backtest │ System │AI Chat││
│  └─────────┴─────────┴──────────┴───────────┴──────────┴────────┴──────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   SCREENER    │          │   COMMITTEE   │          │   BACKTEST    │
│ • Sentiment   │          │ • Agents      │          │ • Engine      │
│ • Technicals  │          │ • Coordinator │          │ • Strategies  │
│ • Gap Analysis│          │ • Voting      │          │ • Metrics     │
│ • Signals     │          └───────────────┘          └───────────────┘
└───────────────┘                   │                        │
        │                           │                        │
        └───────────────────────────┼────────────────────────┘
                                    ▼
                         ┌───────────────────┐
                         │   AI LEARNING     │
                         │ • SignalLearner   │
                         │ • AlphaChat       │
                         │ • Pattern Match   │
                         └───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TIMESCALEDB                                      │
│  Tables: screener_scores, trading_signals, fundamentals, prices,        │
│          historical_scores, news_articles, committee_decisions,          │
│          agent_votes, backtest_results, insider_transactions            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES (WSL)                               │
│  • Qwen3-32B (port 8090) - Main AI model                                │
│  • Tool Server (port 7001) - Web search via Tavily                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
C:\Develop\Latest_2025\HH_research_platform_v1\
│
├── .env                          # Environment variables (DB, API keys)
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── __init__.py
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── learning.py           # SignalLearner - historical pattern matching
│   │   └── chat.py               # AlphaChat - AI assistant with web search
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py             # BacktestEngine - runs strategies
│   │   ├── strategies.py         # 9 predefined strategies, OPTIMIZATION_GRIDS
│   │   └── metrics.py            # Sharpe, Sortino, drawdown, etc.
│   │
│   ├── committee/
│   │   ├── __init__.py
│   │   ├── agents.py             # FundamentalAgent, SentimentAgent, TechnicalAgent, ValuationAgent
│   │   └── coordinator.py        # CommitteeCoordinator - orchestrates voting
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── news.py               # NewsCollector - 6 sources
│   │   └── insider.py            # InsiderDataFetcher
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
│   ├── test_backtest.py          # Test backtest module
│   ├── test_ai_chat.py           # Test AI chat
│   ├── test_learning_backtest.py # Test AI learning integration
│   ├── save_all_backtests.py     # Save all strategies to DB
│   └── fetch_historical_prices.py # Fetch 3yr prices from Yahoo Finance
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (7 tabs)
│
└── migrations/
    └── 001_create_backtest_results.sql
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
    fundamental_score DECIMAL,
    growth_score DECIMAL,
    technical_score DECIMAL,
    gap_score DECIMAL,
    total_score DECIMAL,
    article_count INTEGER,
    insider_signal VARCHAR(50),
    institutional_signal VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
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

-- Historical scores with returns
CREATE TABLE historical_scores (
    id SERIAL PRIMARY KEY,
    score_date DATE,
    ticker VARCHAR(10),
    sector VARCHAR(50),
    sentiment DECIMAL,
    fundamental_score DECIMAL,
    growth_score DECIMAL,
    dividend_score DECIMAL,
    total_score DECIMAL,
    gap_score DECIMAL,
    mkt_score DECIMAL,
    signal_type VARCHAR(20),
    signal_correct BOOLEAN,
    op_price DECIMAL,             -- Entry price
    return_1d DECIMAL,
    return_5d DECIMAL,
    return_10d DECIMAL,
    return_20d DECIMAL,
    price_1d DECIMAL,
    price_5d DECIMAL,
    price_10d DECIMAL,
    price_20d DECIMAL
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

-- Committee decisions
CREATE TABLE committee_decisions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    verdict VARCHAR(20),
    conviction INTEGER,
    expected_alpha_bps INTEGER,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent votes
CREATE TABLE agent_votes (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER REFERENCES committee_decisions(id),
    agent_name VARCHAR(50),
    vote VARCHAR(20),
    confidence INTEGER,
    reasoning TEXT
);

-- News articles
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP,
    sentiment_score DECIMAL,
    created_at TIMESTAMP DEFAULT NOW()
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
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Key Components

### 1. Sentiment Analysis (`src/screener/sentiment.py`)

**Dual-LLM Pipeline:**
- **Qwen3-32B** (port 8090): Main sentiment analysis
- Previously used GPT-OSS for filtering, now using Qwen for all

**Configuration:**
```python
@dataclass
class LLMConfig:
    qwen_base_url: str = "http://172.23.193.91:8090/v1"
    qwen_model: str = "Qwen3-32B-Q6_K.gguf"
    gpt_oss_base_url: str = "http://172.23.193.91:8090/v1"  # Now same as Qwen
    gpt_oss_model: str = "Qwen3-32B-Q6_K.gguf"
```

### 2. Backtest Engine (`src/backtest/engine.py`)

**Strategies Available:**
| Strategy | Type | Description |
|----------|------|-------------|
| aggressive_buy | signal_based | Buy on STRONG_BUY, BUY, WEAK_BUY |
| conservative_buy | signal_based | Only STRONG_BUY |
| buy_signals | signal_based | BUY + STRONG_BUY |
| long_short | signal_based | Long on buys, short on sells |
| high_sentiment | sentiment_only | Buy when sentiment >= 65 |
| high_fundamental | fundamental_only | Buy when fundamental >= 65 |
| high_total_score | score_threshold | Buy when total >= 55 |
| quality_momentum | composite | 2+ conditions met |
| triple_screen | composite | All 3 conditions met |

**Best Performing (from backtest):**
- **Strategy:** signal_based (BUY + STRONG_BUY)
- **Holding Period:** 5 days
- **Sharpe Ratio:** 1.26
- **Win Rate:** 54.2%
- **Alpha vs SPY:** +40.65%

### 3. AI Chat (`src/ai/chat.py`)

**Features:**
- Stock analysis using database data
- Web search for general questions (via Tavily)
- Streaming responses
- Conversation history
- No hallucination (strict data rules)

**Configuration:**
```python
@dataclass
class ChatConfig:
    base_url: str = "http://172.23.193.91:8090/v1"
    model: str = "Qwen3-32B-Q6_K.gguf"
    temperature: float = 0.7
    max_tokens: int = 2000
```

### 4. AI Learning (`src/ai/learning.py`)

**Features:**
- Historical pattern matching
- Sector insights
- Signal confidence adjustment based on backtest results
- Methods: `get_backtest_insights()`, `adjust_signal_confidence()`, `get_pattern()`

---

## WSL Services

### Terminal 1: Qwen3-32B (Port 8090)
```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/Qwen3-32B-Q6_K.gguf \
  -c 4096 \
  -b 512 \
  -ub 256 \
  -ngl 99 \
  -t 8 \
  --flash-attn on \
  --no-mmap \
  --host 0.0.0.0 \
  --port 8090
```

**Performance:** ~45-47 tokens/second, 2100+ tokens/second prompt processing

### Terminal 2: Tool Server (Port 7001)
```bash
cd ~/ai_tools
source venv/bin/activate
export TAVILY_API_KEY="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"
export BRAVE_API_KEY="BSA8i6DP1kdtTndCDBgBMMKW3LJ48QH"
python tools_api/tool_server.py
```

**Endpoints:**
- `POST /search` - Web search via Tavily

---

## Dashboard (Streamlit)

**URL:** http://localhost:8501
**Remote Access:** http://100.93.153.6:8501 (via Tailscale)

### Tabs:
1. **📊 Universe** - All stocks with scores
2. **📈 Signals** - Current trading signals
3. **🔍 Deep Dive** - Single stock analysis
4. **💼 Portfolio** - Portfolio management
5. **📉 Backtest** - Strategy backtesting with visualization
6. **⚙️ System** - System status and controls
7. **🤖 AI Chat** - Chat with AI about stocks or anything

### Tab 5 Features (Backtest):
- Strategy selector dropdown
- Holding period selector (1/5/10/20 days)
- Benchmark selector (SPY/VOO/QQQ/IWM/DIA)
- Run Backtest button
- Compare All Strategies button
- Results display with metrics cards
- Equity curve chart
- Returns by signal table
- Save for AI Learning button

### Tab 7 Features (AI Chat):
- Ticker context input
- Streaming responses
- Web search for general questions
- Clear chat button
- Quick action buttons

---

## CLI Commands

```powershell
# Run full screener for all tickers
python scripts/run_full_screener.py

# Run committee decisions
python scripts/run_committee.py

# Run backtest
python scripts/run_backtest.py --strategy buy_signals --holding-period 5
python scripts/run_backtest.py --compare-all
python scripts/run_backtest.py --optimize --strategy sentiment_only

# Fetch historical prices
python scripts/fetch_historical_prices.py
python scripts/fetch_historical_prices.py --benchmarks-only

# Save all backtests for AI learning
python scripts/save_all_backtests.py

# Test scripts
python scripts/test_backtest.py
python scripts/test_ai_chat.py
python scripts/test_learning_backtest.py

# Run dashboard
streamlit run dashboard/app.py
```

---

## Database Connection

**Docker Container:** `alpha-timescaledb`
**Connection String:** In `.env` file

```powershell
# Access database
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform

# Example queries
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT COUNT(*) FROM prices"
docker exec -it alpha-timescaledb psql -U alpha -d alpha_platform -c "SELECT * FROM backtest_results ORDER BY sharpe_ratio DESC LIMIT 5"
```

---

## Data Statistics

| Table | Records | Date Range |
|-------|---------|------------|
| prices | 79,065 | 2022-12-13 to 2025-12-12 |
| historical_scores | 2,253 | 2025-08-30 to 2025-12-10 |
| news_articles | 5,933 | Recent |
| backtest_results | 27+ | 2025-12-12 |
| Tickers | 105 | Universe + benchmarks |

---

## API Keys (In .env or exported)

```
TAVILY_API_KEY=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
BRAVE_API_KEY=BSA8i6DP1kdtTndCDBgBMMKW3LJ48QH
```

---

## Models Available

| Model | Size | Location | Purpose |
|-------|------|----------|---------|
| Qwen3-32B-Q6_K | 26GB | ~/models/ | Main AI (chat, sentiment) |
| Qwen3-Next-80B | 45GB | ~/models/ | Heavy analysis (slower) |
| Qwen3-VL-30B | ~20GB | ~/models/ | Vision-language |

---

## Known Issues / TODO

1. **Tab switching bug** - Dashboard resets to Universe tab on some interactions (ticker input)
2. **Add chat to Deep Dive tab** - Discuss selected ticker inline
3. **Phone access** - Tailscale setup complete (IP: 100.93.153.6)

---

## Session Summary (2025-12-12)

### Accomplished:
1. ✅ Built complete backtest engine with 9 strategies
2. ✅ Created backtest dashboard tab with visualization
3. ✅ Integrated backtest results into AI learning
4. ✅ Migrated from 80B to 32B Qwen model (6x faster)
5. ✅ Built AI Chat tab with streaming responses
6. ✅ Connected web search (Tavily) for general questions
7. ✅ Fetched 3 years of price data (79K records)
8. ✅ Saved 27 backtest results for AI learning
9. ✅ Setup Tailscale for remote access

### Key Findings:
- Best strategy: BUY + STRONG_BUY signals with 5-day hold
- Sharpe ratio: 1.26, Win rate: 54.2%, Alpha: +40.65%
- Qwen3-32B runs at 45+ tok/s on RTX 5090
- Web search working via Tavily API