# Alpha Research Platform - Complete Documentation v6.0

## Project Overview

The Alpha Research Platform is a comprehensive **hedge fund-style** stock analysis and trading signal generation system built for personal portfolio management. It combines AI-powered sentiment analysis, fundamental/technical analysis, backtesting, risk management, trade journaling, performance tracking, and an AI chat assistant with **unified portfolio awareness**.

**Owner:** Alpha  
**Database:** TimescaleDB in Docker (`alpha-timescaledb`)  
**AI Models:** Qwen3-32B (local via llama.cpp on WSL2)  
**Broker Integration:** Interactive Brokers (IBKR) TWS/Gateway

---

## What's New in v6.0

| Feature | Description |
|---------|-------------|
| **Unified Portfolio Context** | AI Chat now sees combined view from IBKR Live + ETF Creator + Rebalancer |
| **ETF Creator JSON Integration** | Enhanced output with AI reasoning for stock selection |
| **Rebalancer JSON Integration** | Enhanced output with trade execution details and drift analysis |
| **Yahoo Finance Batch Pricing** | 10x faster portfolio loading (batch vs one-by-one) |
| **Smart Context Detection** | AI automatically loads portfolio context for relevant questions |

---

## Hardware Setup

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 9950X (16 cores) |
| **RAM** | 128GB |
| **GPU** | NVIDIA RTX 5090 (32GB VRAM) |
| **OS** | Windows 11 with WSL2 (Ubuntu) |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT DASHBOARD                                  │
│  ┌──────────┬─────────┬───────────┬───────────┬──────────┬────────┬────────────┐│
│  │ Universe │ Signals │ Deep Dive │ Portfolio │ Backtest │ System │  AI Chat   ││
│  └──────────┴─────────┴───────────┴───────────┴──────────┴────────┴────────────┘│
│                                                                                   │
│  AI Chat Features (ENHANCED in v6.0):                                            │
│  ├── 📊 Daily Briefing (Risk alerts, portfolio health, daily questions)          │
│  ├── 📓 Trade Journal (Thesis tracking, targets, stop-loss)                      │
│  ├── 📈 Performance Tracking (P&L charts, vs SPY comparison)                     │
│  ├── 🔍 Stock Analysis (Dropdown + AI report)                                    │
│  ├── 💬 Interactive Chat (Web search, streaming responses)                       │
│  └── 🎯 UNIFIED PORTFOLIO CONTEXT (NEW - IBKR + ETF Creator + Rebalancer)       │
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
│                   │          │                   │          │                   │
│  JSON Output:     │          │  JSON Output:     │          │  Data Source:     │
│  portfolio_*.json │          │  rebalance_*.json │          │  TWS API          │
└─────────┬─────────┘          └─────────┬─────────┘          └─────────┬─────────┘
          │                              │                              │
          └──────────────────────────────┼──────────────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │   UNIFIED PORTFOLIO CONTEXT   │
                         │   (unified_portfolio_context) │
                         │                               │
                         │   Per-symbol merge:           │
                         │   • ACTUAL (IBKR live)        │
                         │   • TARGET (ETF Creator)      │
                         │   • LAST REBALANCE (status)   │
                         │   • DRIFT (calculated)        │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │          AI CHAT              │
                         │   (chat.py - AlphaChat)       │
                         │                               │
                         │   Single source of truth      │
                         │   No confusion between        │
                         │   actual vs target            │
                         └───────────────────────────────┘
```

---

## Project Locations

| Project | Location | Purpose |
|---------|----------|---------|
| **Alpha Platform** | `C:\Develop\Latest_2025\HH_research_platform_v1` | Main dashboard & AI |
| **ETF Creator** | `C:\Develop\Latest_2025\create_etf_v5` | Build target portfolios |
| **Rebalancer** | `C:\Develop\Latest_2025\rebalancing_portfolio_website_v45_allWorks` | Execute rebalances |

---

## Project Structure (Alpha Platform)

```
C:\Develop\Latest_2025\HH_research_platform_v1\
│
├── .env                              # Environment variables (DB, API keys, paths)
├── requirements.txt                  # Python dependencies
├── portfolio_tab.py                  # Portfolio tab component
│
├── src/
│   ├── __init__.py
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── learning.py               # SignalLearner - historical pattern matching
│   │   ├── chat.py                   # AlphaChat - AI assistant (ENHANCED v6.0)
│   │   ├── risk_analyzer.py          # Portfolio risk analysis
│   │   ├── performance_tracker.py    # Daily P&L tracking
│   │   └── unified_portfolio_context.py  # NEW: Unified portfolio loader
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                 # BacktestEngine - runs strategies
│   │   ├── strategies.py             # 9 predefined strategies
│   │   └── metrics.py                # Sharpe, Sortino, drawdown
│   │
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── ibkr_utils.py             # IBKR connection, positions, orders
│   │   └── yahoo_prices.py           # NEW: Batch price fetching
│   │
│   ├── committee/
│   │   ├── __init__.py
│   │   ├── agents.py                 # FundamentalAgent, SentimentAgent, etc.
│   │   └── coordinator.py            # CommitteeCoordinator
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── news.py                   # NewsCollector - 6 sources
│   │   ├── insider.py                # InsiderDataFetcher
│   │   └── finviz.py                 # FinvizDataFetcher
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py             # get_connection(), get_engine()
│   │   └── repository.py             # Repository class - all DB operations
│   │
│   ├── screener/
│   │   ├── __init__.py
│   │   ├── sentiment.py              # SentimentAnalyzer - dual LLM pipeline
│   │   ├── signals.py                # SignalGenerator
│   │   ├── technicals.py             # TechnicalAnalyzer
│   │   ├── gap_analysis.py           # GapAnalyzer
│   │   └── worker.py                 # ScreenerWorker
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py                # get_logger()
│
├── scripts/
│   ├── run_full_screener.py          # CLI: Process all tickers
│   ├── run_committee.py              # CLI: Run committee decisions
│   ├── run_backtest.py               # CLI: Run backtests
│   └── fetch_historical_prices.py    # Fetch 3yr prices
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard (7 tabs)
│
├── config/
│   └── universe.csv                  # List of tickers to track
│
├── json/                             # NEW: Debug outputs
│   └── debug/
│       ├── debug_ibkr_positions.json
│       ├── debug_ibkr_summary.json
│       └── debug_unified_holdings.json
│
└── migrations/
    └── *.sql                         # Database migration scripts
```

---

## Unified Portfolio Context (NEW in v6.0)

The AI Chat now has access to a **unified view** of your portfolio combining three data sources:

### Data Sources

| Source | Contains | Answers Questions Like |
|--------|----------|------------------------|
| **IBKR Live** | Real positions, shares, values, P&L | "Do I own NVDA?", "What's my portfolio value?" |
| **ETF Creator** | Target weights, selection reasoning, scores | "Why was NVDA selected?", "What's the target weight?" |
| **Rebalancer** | Trade execution, drift, failed trades | "Did my trades execute?", "What failed?" |

### Unified Holding Structure

For each symbol, the AI sees:

```
NVDA:
├── ACTUAL (from IBKR - Live)
│   ├── Shares: 50
│   ├── Value: $22,500
│   ├── Weight: 6.2%
│   ├── Avg Cost: $400.00
│   └── Unrealized P&L: +$2,500
│
├── TARGET (from ETF Creator)
│   ├── Target Weight: 6.5%
│   ├── Selection Reason: "Strong AI growth, dominant GPU market..."
│   ├── Scores: Sentiment=75, Fundamental=82, Growth=90
│   └── Risks: ["Valuation risk", "Competition from AMD"]
│
├── LAST REBALANCE (from Rebalancer)
│   ├── Action: BUY 5 shares
│   ├── Status: FILLED
│   ├── Execution Accuracy: 100%
│   └── Date: 2024-12-14
│
└── CALCULATED
    ├── Drift: -0.3% (underweight)
    └── Status: ✓ Held, slightly below target
```

### Holding Status Icons

| Status | Meaning |
|--------|---------|
| ✓ On target | Actual weight within 0.5% of target |
| ⚠️ Underweight | You own less than target |
| ⚠️ Overweight | You own more than target |
| ❌ Not held | In target but you don't own it |
| ⚠️ Not in target | You own it but it's not in target |

### Example AI Questions

| Question | AI Answer Uses |
|----------|----------------|
| "Do I own NVDA?" | ACTUAL (IBKR) - checks real shares |
| "Why do I hold NVDA?" | TARGET (ETF Creator) - explains reasoning |
| "Should I rebalance?" | DRIFT analysis - compares actual vs target |
| "What trades failed?" | REBALANCER - checks execution status |
| "What's my portfolio worth?" | ACTUAL (IBKR) - sums market values |

---

## AI Model Setup (Qwen3-32B via llama.cpp)

### Hardware Requirements
- **GPU:** NVIDIA RTX 5090 (32GB VRAM) or similar
- **RAM:** 64GB+ recommended
- **OS:** WSL2 Ubuntu on Windows 11

### Installation (WSL2)

```bash
# 1. Clone llama.cpp
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2. Build with CUDA support
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

# 3. Download Qwen3-32B model
cd ~/llama.cpp/models
wget https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q6_K.gguf
```

### Running the Model

```bash
# Start llama-server (WSL2)
cd ~/llama.cpp/build/bin

./llama-server \
    -m ~/llama.cpp/models/Qwen3-32B-Q6_K.gguf \
    --host 0.0.0.0 \
    --port 8090 \
    -c 32768 \
    -ngl 99 \
    --chat-template chatml
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-m` | model path | Path to GGUF model file |
| `--host` | 0.0.0.0 | Listen on all interfaces |
| `--port` | 8090 | API port (OpenAI compatible) |
| `-c` | 32768 | Context size (tokens) |
| `-ngl` | 99 | GPU layers (99 = all on GPU) |
| `--chat-template` | chatml | Chat format template |

### Verify Model Running

```bash
curl http://localhost:8090/v1/models
```

### API Endpoint (from Windows)

The WSL2 IP can be found with:
```bash
ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
```

Configure in `.env`:
```ini
LLM_QWEN_BASE_URL=http://172.23.193.91:8090/v1
LLM_QWEN_MODEL=Qwen3-32B-Q6_K.gguf
```

---

## ETF Creator Integration

### Enhanced JSON Output

The ETF Creator now outputs detailed JSON files with AI reasoning:

**Location:** `C:\Develop\Latest_2025\create_etf_v5\json\`

**Files:**
- `portfolio_SC_20241214_143052.json` (timestamped)
- `portfolio_SC_latest.json` (symlink to latest)

### JSON Structure

```json
{
  "metadata": {
    "generated_at": "2024-12-14T14:30:52",
    "etf_sources": ["KAIF"],
    "total_tickers_analyzed": 150,
    "tickers_selected": 98,
    "initial_investment": 1330000
  },
  "summary": {
    "total_holdings": 98,
    "total_weight": 100.0,
    "top_10_concentration": 42.5,
    "sector_distribution": [
      {"name": "Technology", "value": 35.2},
      {"name": "Healthcare", "value": 15.8}
    ]
  },
  "holdings": [
    {
      "symbol": "NVDA",
      "name": "NVIDIA Corporation",
      "weight": 6.05,
      "target_shares": 100,
      "sector": "Technology",
      "scores": {
        "sentiment": 75,
        "fundamental": 82,
        "growth": 90,
        "technical": 68
      },
      "weight_breakdown": {
        "market_cap_weight": 3.2,
        "sentiment_weight": 1.5,
        "fundamental_weight": 1.35
      },
      "reasoning": "Selected for strong AI/GPU growth trajectory. Dominant market position in data center GPUs. High sentiment score reflects positive analyst outlook.",
      "risks": ["High valuation multiple", "Competition from AMD/Intel"]
    }
  ]
}
```

### Key Files (ETF Creator)

| File | Purpose |
|------|---------|
| `create_etf_weights.py` | Main weight calculation (ENHANCED) |
| `enhanced_output.py` | JSON output manager with reasoning |
| `reasoning_generator.py` | AI reasoning for selections |

---

## Rebalancer Integration

### Enhanced JSON Output

The Rebalancer now outputs detailed JSON with trade execution details:

**Location:** `C:\Develop\Latest_2025\rebalancing_portfolio_website_v45_allWorks\json\`

**Files:**
- `rebalance_KAIF_20241214_214052.json` (timestamped)
- `rebalance_KAIF_latest.json` (symlink to latest)

### JSON Structure

```json
{
  "metadata": {
    "generated_at": "2024-12-14T21:40:52",
    "etf_sources": ["KAIF"],
    "account_id": "U1234567",
    "initial_investment": 1330000,
    "execution_time_seconds": 45.2
  },
  "summary": {
    "total_holdings": 98,
    "holdings_rebalanced": 45,
    "turnover_ratio": 18.3,
    "average_accuracy": 96.5,
    "total_buy_value": 25000.00,
    "total_sell_value": 18000.00,
    "net_cash_flow": -7000.00,
    "estimated_commission": 45.00
  },
  "holdings": [
    {
      "symbol": "NVDA",
      "drift": {
        "target_weight": 6.05,
        "actual_weight_before": 5.50,
        "actual_weight_after": 6.02,
        "drift_before": -0.55,
        "drift_after": -0.03
      },
      "trade": {
        "action": "BUY",
        "target_quantity": 5,
        "executed_quantity": 5,
        "price": 450.00,
        "total_value": 2250.00,
        "execution_accuracy": 100.0,
        "status": "FILLED"
      },
      "reasoning": "Underweight by 0.55% vs target 6.05%. Buying 5 shares at $450.00 to align with target allocation."
    }
  ],
  "failed_holdings": [
    {
      "symbol": "XYZ",
      "error": "Insufficient liquidity",
      "attempted_action": "BUY",
      "attempted_quantity": 100
    }
  ]
}
```

### Key Files (Rebalancer)

| File | Purpose |
|------|---------|
| `rebalance_etf_v4.py` | Main rebalancer class (ENHANCED) |
| `rebalance_output_enhanced.py` | JSON output with reasoning |
| `app.py` | Flask web interface |

---

## Yahoo Finance Batch Pricing

### Why?

| Method | Time for 100 stocks |
|--------|---------------------|
| IBKR (one-by-one) | 30-60 seconds |
| Yahoo Finance (batch) | 2-5 seconds |

### Usage

```python
from src.broker.yahoo_prices import get_batch_prices, update_positions_with_yahoo_prices

# Get prices for multiple symbols at once
prices = get_batch_prices(["NVDA", "AAPL", "MSFT", "GOOGL"])
# Returns: {"NVDA": 450.00, "AAPL": 195.50, ...}

# Update positions list with Yahoo prices
positions = update_positions_with_yahoo_prices(positions)
```

### File Location

`src/broker/yahoo_prices.py`

---

## Environment Variables (.env)

```ini
# Database (TimescaleDB)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=alpha_platform
POSTGRES_USER=alpha
POSTGRES_PASSWORD=alpha_secure_2024

# LLM Configuration (Qwen3-32B)
LLM_QWEN_BASE_URL=http://172.23.193.91:8090/v1
LLM_QWEN_MODEL=Qwen3-32B-Q6_K.gguf

# Tool Server (Web Search)
TOOL_SERVER_URL=http://172.23.193.91:7001

# Portfolio JSON Paths (NEW in v6.0)
ETF_CREATOR_JSON_DIR=C:\Develop\Latest_2025\create_etf_v5\json
REBALANCER_JSON_DIR=C:\Develop\Latest_2025\rebalancing_portfolio_website_v45_allWorks\json

# API Keys
TAVILY_API_KEY=your_key_here
BRAVE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
NEWSAPI_API_KEY=your_key_here

# Application Settings
LOG_LEVEL=INFO
UNIVERSE_SIZE=100
```

---

## Startup Sequence

### 1. Start Database (Docker)

```powershell
docker start alpha-timescaledb
```

### 2. Start AI Model (WSL2)

```bash
cd ~/llama.cpp/build/bin
./llama-server -m ~/llama.cpp/models/Qwen3-32B-Q6_K.gguf --host 0.0.0.0 --port 8090 -c 32768 -ngl 99 --chat-template chatml
```

### 3. Start Tool Server (WSL2) - Optional for web search

```bash
cd ~/tool-server
python server.py
```

### 4. Start IBKR TWS/Gateway

- Launch TWS or IB Gateway
- Configure API: Edit → Global Configuration → API → Settings
- Enable: "Enable ActiveX and Socket Clients"
- Port: 7496 (TWS) or 4001 (Gateway)
- Allow connections from localhost

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
IBKR portfolio positions with P&L, sector allocation.

### Tab 5: 📉 Backtest
Strategy backtesting with visualization.

### Tab 6: ⚙️ System
System status and controls.

### Tab 7: 🤖 AI Chat (ENHANCED in v6.0)

**Features:**

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

5. **Interactive Chat** (ENHANCED)
   - **Unified Portfolio Context** - AI knows your actual positions, targets, and drift
   - Ask: "Do I own NVDA?" → Checks actual IBKR positions
   - Ask: "Why do I hold NVDA?" → Explains ETF Creator reasoning
   - Ask: "Should I rebalance?" → Analyzes drift and recommends
   - Web search for current events
   - Streaming responses

---

## CLI Commands

```powershell
# Activate virtual environment
cd C:\Develop\Latest_2025\HH_research_platform_v1
.\.venv\Scripts\Activate

# Run full screener for all tickers
python scripts/run_full_screener.py

# Run committee decisions
python scripts/run_committee.py

# Run backtest
python scripts/run_backtest.py --strategy buy_signals --holding-period 5

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
    technical_score DECIMAL,
    total_score DECIMAL,
    article_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trading signals
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    signal_type VARCHAR(20),  -- STRONG_BUY, BUY, NEUTRAL, SELL, etc.
    signal_strength INTEGER,
    signal_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trade Journal
CREATE TABLE trade_journal (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- BUY, SELL, ADD, TRIM
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4),
    thesis TEXT,
    target_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    conviction INTEGER,
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio Snapshots
CREATE TABLE portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    net_liquidation DECIMAL(15,2),
    daily_pnl DECIMAL(12,2),
    daily_return_pct DECIMAL(8,4),
    cumulative_return_pct DECIMAL(8,4),
    alpha_vs_benchmark DECIMAL(8,4),
    position_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
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

## Risk Thresholds

| Metric | Default | Description |
|--------|---------|-------------|
| MAX_POSITION_WEIGHT | 10% | Alert if single position exceeds |
| MAX_SECTOR_WEIGHT | 30% | Alert if sector exceeds |
| PROFIT_TAKING_THRESHOLD | 40% | Suggest profit taking above |
| STOP_LOSS_THRESHOLD | -20% | Suggest review below |
| CORRELATION_WARNING | 0.70 | Warn if pair correlation exceeds |

---

## Troubleshooting

### AI Model Context Full
**Error:** `exceed_context_size_error`
- Auto-handled: Old messages cleared automatically
- Manual: Click "Clear" button in chat
- Permanent: Increase `-c` parameter in llama-server (e.g., `-c 65536`)

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
- Check API keys in `.env`

### Portfolio Context Not Loading
- Check JSON files exist in configured directories
- Verify `.env` paths: `ETF_CREATOR_JSON_DIR`, `REBALANCER_JSON_DIR`
- Check debug files in `json/debug/` folder

### AI Says "I don't own X" but I do
- IBKR field mapping issue - check `debug_ibkr_positions.json`
- Ensure `unified_portfolio_context.py` has correct field mappings:
  - `quantity` (not `position` or `shares`)
  - `market_value` (not `marketValue`)
  - `avg_cost` (not `avgCost`)

---

## Files Created in v6.0

| File | Location | Purpose |
|------|----------|---------|
| `unified_portfolio_context.py` | `src/ai/` | Merges IBKR + ETF Creator + Rebalancer |
| `chat.py` (v6) | `src/ai/` | Enhanced AI chat with unified context |
| `yahoo_prices.py` | `src/broker/` | Batch price fetching |
| `enhanced_output.py` | ETF Creator | JSON output with reasoning |
| `rebalance_output_enhanced.py` | Rebalancer | JSON output with execution details |

---

## Architecture Flow

```
User asks: "Why do I hold NVDA?"
                │
                ▼
        ┌───────────────┐
        │   AI Chat     │
        │   (chat.py)   │
        └───────┬───────┘
                │
                │ Detects "hold" keyword
                │ → needs portfolio context
                ▼
        ┌───────────────────────┐
        │ UnifiedPortfolioLoader│
        └───────┬───────────────┘
                │
        ┌───────┴───────┬───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │  IBKR   │    │   ETF   │    │ Rebal   │
   │  Live   │    │ Creator │    │  ancer  │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        ▼              ▼              ▼
   actual_shares  target_weight  last_action
   actual_value   reasoning      status
   actual_weight  scores         accuracy
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ UnifiedHolding  │
              │                 │
              │ NVDA:           │
              │ • Actual: 6.2%  │
              │ • Target: 6.5%  │
              │ • Drift: -0.3%  │
              │ • Reason: "..." │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   AI Response   │
              │                 │
              │ "You hold NVDA  │
              │  at 6.2% (50    │
              │  shares worth   │
              │  $22,500). It   │
              │  was selected   │
              │  because..."    │
              └─────────────────┘
```

---

## Key Philosophy

- **One AI, one tab, all answers** - The AI Chat is your command center
- **Unified truth** - No confusion between actual vs target positions
- **Document your thesis** - Trade Journal tracks why you bought
- **Track your performance** - Daily Snapshots measure results
- **Challenge your thinking** - Daily Questions from AI
- **Data-driven decisions** - Risk Metrics & Scores guide action

---

## Contact & Notes

This platform is designed for personal use by Alpha for managing a real portfolio with IBKR integration. The AI assistant acts as a risk manager, research analyst, and accountability partner - similar to how hedge funds operate but consolidated into a single AI interface.

**Version:** 6.0  
**Last Updated:** December 2024  
**Author:** Hasan
