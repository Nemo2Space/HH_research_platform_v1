# HH Research Platform

## Complete Technical Documentation for AI Model Handoff

**Version:** 2026-01-08  
**Author:** Hasan (Senior Portfolio Manager at BlackRock)  
**Portfolio:** $1.3M with 144 positions (mega/large-cap stocks only)

---

## 1. Platform Overview

The **HH Research Platform** is an institutional-grade quantitative trading system built in Python with a Streamlit dashboard. It combines multiple data sources, AI-powered analysis, and systematic signal generation to support investment decision-making.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Signal Generation** | Multi-factor scoring (sentiment, technical, fundamental, options) |
| **AI Committee** | Multi-agent LLM voting system for trade decisions |
| **RAG System** | SEC filing retrieval with pgvector embeddings |
| **Dual Analyst** | SQL (quantitative) + RAG (qualitative) parallel analysis |
| **Options Flow** | Real-time IBKR integration with max pain, P/C ratios |
| **Earnings Intelligence** | Pre/post earnings analysis with reaction tracking |
| **Bond Prediction** | Economic data integration for fixed income |
| **Portfolio Tracking** | IBKR live positions with drift monitoring |

### Technology Stack

```
Backend:        Python 3.11+
Database:       PostgreSQL 15+ with pgvector extension
Vector Store:   pgvector (1536-dim OpenAI embeddings)
LLM:            Qwen3-32B (local via llama.cpp) + OpenAI fallback
Broker:         Interactive Brokers (TWS/Gateway)
Dashboard:      Streamlit 1.40+
Data Sources:   Yahoo Finance, NewsAPI, Finnhub, SEC EDGAR, Reddit
```

---

## 2. Directory Structure

```
HH_research_platform_v1/
├── dashboard/
│   ├── app.py                    # Main Streamlit application (1300+ lines)
│   ├── portfolio_tab.py          # Portfolio management UI
│   ├── analytics_tab.py          # Analytics & optimization
│   ├── trade_ideas_tab.py        # AI trade recommendations
│   └── bond_signals_dashboard.py # Bond trading interface
│
├── src/
│   ├── core/
│   │   ├── __init__.py           # Exports UnifiedSignal, generate_signal
│   │   ├── unified_signal.py     # Central data model (40+ fields)
│   │   ├── unified_scorer.py     # Single source of truth for scoring
│   │   └── signal_engine.py      # Orchestrates all signal generation
│   │
│   ├── ai/
│   │   ├── chat.py               # AlphaChat - main conversational AI
│   │   ├── dual_analyst.py       # SQL + RAG parallel analysis
│   │   ├── ai_research_agent_v4.py # Structured research responses
│   │   ├── committee/
│   │   │   ├── coordinator.py    # Multi-agent committee orchestration
│   │   │   └── agents.py         # Individual agent implementations
│   │   └── provider.py           # Multi-LLM provider abstraction
│   │
│   ├── signals/
│   │   ├── filing_signal.py      # SEC filing-based signals
│   │   ├── sentiment.py          # News sentiment analysis
│   │   ├── technical.py          # Technical indicators
│   │   └── fundamental.py        # Fundamental scoring
│   │
│   ├── rag/
│   │   ├── schema.sql            # pgvector schema (rag.chunks, rag.filing_facts)
│   │   ├── sec_ingestion.py      # SEC EDGAR 10-K/10-Q fetching
│   │   ├── transcript_ingestion.py # Earnings call transcript upload
│   │   ├── chunking.py           # Document chunking with overlap
│   │   ├── retriever.py          # RAGRetriever with hybrid search
│   │   └── fact_extractor.py     # LLM-based structured extraction
│   │
│   ├── analytics/
│   │   ├── trade_ideas.py        # TradeIdeasGenerator
│   │   ├── options_flow.py       # OptionsFlowAnalyzer
│   │   ├── short_squeeze.py      # ShortSqueezeDetector
│   │   ├── earnings_intelligence.py # IES/ECS calculations
│   │   └── market_context.py     # MarketContextAnalyzer
│   │
│   ├── ml/
│   │   ├── multi_factor_alpha.py # ML-based return predictions
│   │   └── alpha_enhancements.py # Reliability metrics, shrinkage
│   │
│   ├── broker/
│   │   ├── ibkr.py               # Interactive Brokers integration
│   │   └── ibkr_options.py       # Options chain fetching
│   │
│   ├── tabs/
│   │   ├── signals_tab.py        # Main signals hub (7500+ lines)
│   │   ├── deep_dive_tab.py      # Single ticker analysis
│   │   └── ai_assistant_tab_v4.py # AI research interface
│   │
│   ├── db/
│   │   └── connection.py         # PostgreSQL connection pool
│   │
│   └── utils/
│       ├── logging.py            # Centralized logging
│       └── config.py             # Configuration management
│
├── scripts/
│   ├── run_full_screener.py      # Daily screening pipeline
│   ├── run_committee.py          # Committee analysis CLI
│   └── alpha_model_cli.py        # ML model training/prediction
│
└── config/
    └── .env                      # Environment variables
```

---

## 3. Database Schema

### Core Tables (public schema)

```sql
-- Main scoring table (updated daily)
CREATE TABLE screener_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sentiment_score INTEGER,           -- 0-100
    sentiment_weighted FLOAT,
    fundamental_score INTEGER,         -- 0-100
    technical_score INTEGER,           -- 0-100
    gap_score INTEGER,
    options_flow_score INTEGER,        -- 0-100
    short_squeeze_score INTEGER,
    total_score INTEGER,               -- Composite
    article_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Trading signals
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    signal_type VARCHAR(20),           -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    signal_strength INTEGER,
    signal_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Committee decisions
CREATE TABLE committee_decisions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    verdict VARCHAR(20),               -- STRONG_BUY, BUY, HOLD, SELL
    conviction FLOAT,                  -- 0.0-1.0
    expected_alpha_bps INTEGER,
    horizon_days INTEGER,
    rationale TEXT,
    UNIQUE(ticker, date)
);

-- Agent votes (committee members)
CREATE TABLE agent_votes (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    agent_role VARCHAR(50),            -- technical, fundamental, sentiment, risk, macro
    buy_prob FLOAT,
    confidence FLOAT,
    rationale TEXT,
    UNIQUE(ticker, date, agent_role)
);

-- News articles
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP,
    ai_sentiment_fast INTEGER,         -- Quick LLM sentiment 0-100
    ai_sentiment_deep INTEGER,         -- Deep analysis sentiment
    created_at TIMESTAMP DEFAULT NOW()
);

-- Fundamentals
CREATE TABLE fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    sector VARCHAR(100),
    market_cap BIGINT,
    pe_ratio FLOAT,
    forward_pe FLOAT,
    pb_ratio FLOAT,
    roe FLOAT,
    profit_margin FLOAT,
    revenue_growth FLOAT,
    dividend_yield FLOAT,
    earnings_date DATE,
    UNIQUE(ticker, date)
);

-- Options flow
CREATE TABLE options_flow_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    analysis_date DATE,
    call_volume BIGINT,
    put_volume BIGINT,
    put_call_ratio FLOAT,
    max_pain FLOAT,
    max_pain_expiry DATE,
    options_sentiment VARCHAR(20),
    short_squeeze_score INTEGER,
    UNIQUE(ticker, analysis_date)
);
```

### RAG Tables (rag schema)

```sql
-- Vector storage for document chunks
CREATE TABLE rag.chunks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    doc_type VARCHAR(50) NOT NULL,     -- '10-K', '10-Q', 'transcript'
    doc_date DATE,
    section VARCHAR(100),              -- 'Risk Factors', 'MD&A', 'Q&A'
    content TEXT NOT NULL,
    embedding vector(1536),            -- OpenAI ada-002 embeddings
    token_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Extracted facts from filings
CREATE TABLE rag.filing_facts (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    doc_type VARCHAR(50),
    doc_date DATE,
    key_risks TEXT[],                  -- Array of risk themes
    risk_severity JSONB,               -- {theme: severity_score}
    china_exposure TEXT,
    material_litigation BOOLEAN,
    extracted_at_utc TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, doc_type, doc_date)
);

-- Extracted facts from transcripts
CREATE TABLE rag.transcript_facts (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    call_date DATE,
    guidance_direction VARCHAR(20),    -- raised, maintained, lowered, withdrawn
    demand_tone VARCHAR(20),           -- bullish, neutral, cautious, bearish
    ai_mentions_count INTEGER,
    ai_sentiment VARCHAR(20),
    extracted_at_utc TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, call_date)
);

-- Indexes for vector search
CREATE INDEX idx_chunks_embedding ON rag.chunks 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_ticker ON rag.chunks(ticker);
CREATE INDEX idx_chunks_doc_type ON rag.chunks(doc_type);
```

---

## 4. Core Data Models

### UnifiedSignal (src/core/unified_signal.py)

The central data structure representing a complete stock analysis:

```python
@dataclass
class UnifiedSignal:
    # Identity
    ticker: str
    company_name: str
    sector: str
    asset_type: AssetType  # EQUITY, BOND_ETF
    
    # Scores (0-100)
    today_score: int
    longterm_score: int
    sentiment_score: Optional[int]
    fundamental_score: Optional[int]
    technical_score: Optional[int]
    options_score: Optional[int]
    
    # Signals
    today_signal: SignalStrength  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    sentiment_signal: str
    fundamental_signal: str
    technical_signal: str
    options_signal: str
    
    # Risk
    risk_level: RiskLevel  # LOW, MEDIUM, HIGH, EXTREME
    risk_score: int
    risk_factors: List[str]
    
    # Committee
    committee_verdict: Optional[str]
    committee_confidence: Optional[float]
    expected_alpha_bps: Optional[int]
    
    # Price data
    current_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    pct_from_high: float
    pct_from_low: float
    
    # Earnings
    earnings_date: Optional[date]
    days_to_earnings: Optional[int]
    
    # Portfolio
    in_portfolio: bool
    portfolio_weight: Optional[float]
    portfolio_pnl_pct: Optional[float]
    
    # Metadata
    flags: List[str]  # Special indicators like "📊 EARNINGS_SOON"
    signal_reason: str
    generated_at: datetime
```

### FilingSignal (src/signals/filing_signal.py)

SEC filing-based signal with factor breakdown:

```python
@dataclass
class FilingSignal:
    ticker: str
    score: float  # 0-100 composite
    
    factors: FilingSignalFactors
    # - guidance_score: float
    # - risk_score: float
    # - litigation_score: float
    # - china_score: float
    # - ai_demand_score: float
    
    bullish_signals: List[str]
    bearish_signals: List[str]
    
    has_filing_data: bool
    has_transcript_data: bool
    data_freshness_days: int
```

### DualAnalysisResult (src/ai/dual_analyst.py)

Result from parallel SQL + RAG analysis:

```python
@dataclass
class DualAnalysisResult:
    ticker: str
    question: str
    timestamp: datetime
    
    sql_opinion: Opinion      # Quantitative analyst
    rag_opinion: Opinion      # Qualitative analyst
    
    agreement_score: float    # 0.0-1.0
    sentiment_match: bool
    conflicts: List[str]
    
    synthesis: str            # Combined analysis
    final_sentiment: Sentiment
    final_confidence: float
    
    combined_bullish: List[str]
    combined_bearish: List[str]
    combined_risks: List[str]
```

---

## 5. Key Modules

### 5.1 Signal Engine (src/core/signal_engine.py)

Orchestrates all signal generation:

```python
class SignalEngine:
    def generate_signal(self, ticker: str) -> UnifiedSignal:
        """
        1. Fetch latest scores from DB
        2. Get fundamentals, technicals, sentiment
        3. Fetch options flow data
        4. Get committee decision if available
        5. Calculate composite score
        6. Determine signal strength
        7. Return UnifiedSignal
        """
```

### 5.2 AlphaChat (src/ai/chat.py)

Main conversational AI with portfolio context:

```python
class AlphaChat:
    def chat_stream(self, message: str, ticker: str = None) -> Generator[str, None, None]:
        """
        1. Build context from portfolio, signals, news
        2. Inject macro events if relevant
        3. Stream response from LLM
        4. Track in conversation history
        """
    
    def set_ibkr_data(self, positions: list, summary: dict):
        """Sync live portfolio data for context injection."""
```

### 5.3 RAG Retriever (src/rag/retriever.py)

Hybrid search over SEC filings:

```python
class RAGRetriever:
    def retrieve(
        self,
        query: str,
        ticker: str,
        doc_types: List[str] = None,
        min_date: date = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        1. Embed query using OpenAI ada-002
        2. Vector search in pgvector
        3. Filter by ticker, doc_type, date
        4. Return ranked chunks with scores
        """
```

### 5.4 Dual Analyst (src/ai/dual_analyst.py)

Parallel quantitative + qualitative analysis:

```python
class DualAnalystService:
    def analyze(self, ticker: str, question: str) -> DualAnalysisResult:
        """
        1. Run SQL analyst (uses DB scores, signals)
        2. Run RAG analyst (uses SEC filings, transcripts)
        3. Evaluate agreement/conflicts
        4. Synthesize final verdict
        """
```

### 5.5 Committee Coordinator (src/ai/committee/coordinator.py)

Multi-agent voting system:

```python
class CommitteeCoordinator:
    agents = [
        TechnicalAgent(),
        FundamentalAgent(), 
        SentimentAgent(),
        RiskAgent(),
        MacroAgent(),
    ]
    
    def run_committee(self, ticker: str) -> CommitteeDecision:
        """
        1. Gather data for ticker
        2. Each agent analyzes independently
        3. Collect votes (buy_prob, confidence)
        4. Aggregate to final verdict
        5. Save to database
        """
```

---

## 6. Dashboard Structure

### Main Application (dashboard/app.py)

8 main pages accessible via sidebar:

| Page | Description |
|------|-------------|
| 🔬 Research | Signals Hub, Trade Ideas, Deep Dive, AI Chat (sub-tabs) |
| 💼 Portfolio | Live IBKR positions, P&L, rebalancing |
| 📊 Analytics | Risk dashboard, optimizer, factor analysis |
| 🏦 Bonds | Bond ETF signals with economic data |
| 📈 Performance | Backtest results, signal tracking |
| 🧠 Alpha Model | ML predictions, feature importance |
| 🤖 AI Assistant | General AI research interface |
| ⚙️ System | Database status, quick commands |

### Signals Tab (src/tabs/signals_tab.py)

The most complex tab (7500+ lines) with:

- **Signal table** with sorting/filtering
- **Deep dive panel** for selected ticker
- **SEC Filing Insights** expander
- **Dual Analyst** expander
- **Options flow** analysis
- **AI Chat** integration

---

## 7. Data Flow

### Daily Screening Pipeline

```
1. run_full_screener.py
   ├── Fetch news (NewsAPI, Finnhub, Reddit)
   ├── Analyze sentiment (LLM)
   ├── Fetch fundamentals (Yahoo Finance)
   ├── Calculate technical indicators
   ├── Fetch options flow (IBKR/Yahoo)
   ├── Compute composite scores
   └── Save to screener_scores table

2. For high-scoring tickers:
   ├── Run committee analysis
   └── Save decisions to committee_decisions
```

### Signal Generation Flow

```
User selects ticker → SignalEngine.generate_signal()
                        ├── Load from screener_scores
                        ├── Enrich with fundamentals
                        ├── Add options data
                        ├── Get committee verdict
                        └── Return UnifiedSignal
```

### RAG Query Flow

```
User asks question → Detect ticker → RAGRetriever.retrieve()
                                       ├── Embed query
                                       ├── Vector search
                                       └── Return chunks

                  → DualAnalystService.analyze()
                      ├── SQL Analyst (DB data)
                      ├── RAG Analyst (filing chunks)
                      └── Synthesize verdict
```

---

## 8. Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/HH

# AI Providers
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=qwen3-32b
OPENAI_API_KEY=sk-...

# Broker
IBKR_HOST=127.0.0.1
IBKR_PORT=7496
IBKR_CLIENT_ID=1

# Data Sources
NEWSAPI_KEY=...
FINNHUB_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...

# RAG
EMBEDDING_MODEL=text-embedding-ada-002
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=200
```

---

## 9. Key Algorithms

### Composite Score Calculation

```python
def calculate_total_score(scores: dict) -> int:
    weights = {
        'sentiment': 0.25,
        'fundamental': 0.25,
        'technical': 0.20,
        'options': 0.15,
        'earnings': 0.15,
    }
    
    total = 0
    weight_sum = 0
    
    for key, weight in weights.items():
        score = scores.get(key)
        if score is not None:
            total += score * weight
            weight_sum += weight
    
    return int(total / weight_sum) if weight_sum > 0 else 50
```

### Signal Strength Determination

```python
def determine_signal(score: int) -> SignalStrength:
    if score >= 80:
        return SignalStrength.STRONG_BUY
    elif score >= 65:
        return SignalStrength.BUY
    elif score >= 45:
        return SignalStrength.HOLD
    elif score >= 30:
        return SignalStrength.SELL
    else:
        return SignalStrength.STRONG_SELL
```

### Filing Signal Scoring

```python
DEFAULT_WEIGHTS = {
    'guidance': 0.30,    # Management guidance direction
    'risk': 0.25,        # Risk factor severity
    'litigation': 0.15,  # Material litigation
    'china': 0.15,       # China exposure
    'ai_demand': 0.15,   # AI demand mentions
}

def calculate_filing_score(factors: FilingSignalFactors) -> float:
    return (
        factors.guidance_score * 0.30 +
        factors.risk_score * 0.25 +
        factors.litigation_score * 0.15 +
        factors.china_score * 0.15 +
        factors.ai_demand_score * 0.15
    )
```

---

## 10. Recent Enhancements (2026-01)

### Data Quality Audit
- Eliminated 20+ instances of hardcoded defaults (50, 0, "NEUTRAL")
- Implemented proper `Optional[float]` types throughout
- Added explicit status tracking for missing data

### RAG System (3 Batches)
- **Batch A**: pgvector setup, SEC ingestion, chunking pipeline
- **Batch B**: Retrieval pipeline with hybrid search
- **Batch C**: Fact extraction with LLM-based parsing

### Dual Analyst Integration
- SQL analyst uses database scores/signals
- RAG analyst uses SEC filing embeddings
- Agreement scoring identifies conflicts
- Integrated into Deep Dive and AI Chat

### Bond Prediction Engine
- Economic data integration (FRED)
- Calendar effects analysis
- Seasonality patterns
- Alpha model integration

### Macro Event Engine
- Political/geopolitical news tracking
- Commodity shock detection
- Policy change monitoring
- Exposure mapping per ticker

---

## 11. Running the Platform

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up database
psql -U postgres -c "CREATE DATABASE HH"
psql -U postgres -d HH -f src/rag/schema.sql

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run daily screener
python scripts/run_full_screener.py

# 5. Start dashboard
streamlit run dashboard/app.py
```

### Key Commands

```bash
# Run committee analysis for a ticker
python scripts/run_committee.py --ticker NVDA

# Train alpha model
python scripts/alpha_model_cli.py train --min-date 2025-01-01

# Ingest SEC filings
python -m src.rag.sec_ingestion --ticker MU --years 2

# Test RAG retrieval
python -m src.rag.retriever --ticker MU --query "AI demand"
```

---

## 12. Important Notes for AI Models

### Data Quality Rules

1. **Never use hardcoded defaults** - Always return `None` for missing data
2. **Never use mocked/fake data** - Fetch real data or show "Data not available"
3. **Always check for None before calculations** - Use `Optional[]` types

### Code Style Preferences

1. **No line numbers in edits** - Provide actual code snippets to search/replace
2. **Complete methods when editing** - Don't reference line numbers
3. **PyCharm IDE** - User works in PyCharm for Python projects

### Platform Constraints

1. **Mega/large-cap only** - No small-cap or penny stocks
2. **$1.3M portfolio** - 144 positions, institutional constraints
3. **Local LLM preferred** - Qwen3-32B via llama.cpp on RTX 5090
4. **No cryptocurrency** - Equity and bond ETF focus only

---

## 13. File Locations Quick Reference

| Component | File |
|-----------|------|
| Main app | `dashboard/app.py` |
| Signal generation | `src/core/signal_engine.py` |
| Data model | `src/core/unified_signal.py` |
| AI Chat | `src/ai/chat.py` |
| Dual Analyst | `src/ai/dual_analyst.py` |
| RAG Retriever | `src/rag/retriever.py` |
| SEC Ingestion | `src/rag/sec_ingestion.py` |
| Filing Signal | `src/signals/filing_signal.py` |
| Options Flow | `src/analytics/options_flow.py` |
| Trade Ideas | `src/analytics/trade_ideas.py` |
| Committee | `src/ai/committee/coordinator.py` |
| Signals Tab | `src/tabs/signals_tab.py` |
| Deep Dive Tab | `src/tabs/deep_dive_tab.py` |
| Config | `src/utils/config.py` |
| DB Connection | `src/db/connection.py` |

---

## 14. Contact & Support

**Developer:** Hasan  
**Hardware:** AMD Ryzen 9 9950X, 128GB RAM, RTX 5090  
**LLM:** Qwen3-32B via llama.cpp on WSL2  
**Database:** PostgreSQL with TimescaleDB

---

*This documentation was generated to enable seamless handoff to another AI model for continued development and support of the HH Research Platform.*