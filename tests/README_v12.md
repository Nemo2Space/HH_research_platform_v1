# HH Research Platform v1

## Executive Summary

The HH Research Platform is a **hedge fund-grade equity research and portfolio management system** built for a senior portfolio manager at BlackRock managing a $1.3M personal portfolio with 144 positions. The platform provides institutional-quality signal generation, risk analytics, and AI-powered research capabilities.

**Key Differentiators:**
- Real-time IBKR integration for live portfolio data and execution
- 12+ institutional signal modules (GEX, Dark Pool, Insider, 13F, etc.)
- AI Chat with full portfolio context and web search
- Unified scoring pipeline with transaction cost modeling
- Performance tracking for all signals (including institutional)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Directory Structure](#directory-structure)
4. [Core Modules](#core-modules)
5. [Signal System](#signal-system)
6. [Institutional Signal Modules](#institutional-signal-modules)
7. [Risk Analytics](#risk-analytics)
8. [Portfolio Management](#portfolio-management)
9. [AI Integration](#ai-integration)
10. [Database Schema](#database-schema)
11. [Dashboard UI](#dashboard-ui)
12. [Data Sources](#data-sources)
13. [Configuration](#configuration)
14. [Setup & Installation](#setup--installation)
15. [API Reference](#api-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HH RESEARCH PLATFORM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Dashboard  │  │   AI Chat    │  │  Trade Ideas │  │  Backtester  │ │
│  │  (Streamlit) │  │  (Claude)    │  │  Generator   │  │   Engine     │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │          │
│  ┌──────▼─────────────────▼─────────────────▼─────────────────▼───────┐ │
│  │                      UNIFIED SIGNAL ENGINE                         │ │
│  │  ┌─────────────────────────────────────────────────────────────┐  │ │
│  │  │                    UnifiedSignal Model                       │  │ │
│  │  │  - Technical (RSI, MACD, BB, ADX)                           │  │ │
│  │  │  - Fundamental (PE, PB, ROE, Debt)                          │  │ │
│  │  │  - Sentiment (News, Social)                                  │  │ │
│  │  │  - Options (IV, Put/Call, Flow)                             │  │ │
│  │  │  - Institutional (GEX, Dark Pool, Insider, 13F, etc.)       │  │ │
│  │  └─────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │                         ANALYTICS LAYER                            │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │   GEX    │ │  Dark    │ │ Cross-   │ │ Sentiment│ │ Earnings │ │  │
│  │  │ Analysis │ │  Pool    │ │  Asset   │ │   NLP    │ │ Whisper  │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │ Insider  │ │   13F    │ │ Factor   │ │ Crowding │ │  Regime  │ │  │
│  │  │ Tracker  │ │ Tracker  │ │ Decomp   │ │ Detector │ │ Analysis │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │                         DATA LAYER                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │  │
│  │  │  PostgreSQL  │  │    IBKR      │  │    External APIs         │ │  │
│  │  │  Database    │  │  TWS/Gateway │  │  (Yahoo, SEC, News)      │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Ingestion**: IBKR provides real-time portfolio data, Yahoo Finance provides market data, SEC EDGAR provides filings
2. **Signal Generation**: SignalEngine orchestrates all analytics modules to produce UnifiedSignal
3. **Storage**: Signals stored in PostgreSQL with full JSON for historical analysis
4. **Presentation**: Dashboard displays signals, AI Chat provides natural language interface
5. **Execution**: Trade ideas generated, validated, and can be executed via IBKR

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.11+ | Core application logic |
| **Database** | PostgreSQL 15+ | Signal storage, portfolio history |
| **UI Framework** | Streamlit | Web dashboard |
| **AI/LLM** | Claude API + Local Qwen3-32B | Chat, analysis, sentiment |
| **Broker** | Interactive Brokers API | Live data, execution |
| **ML/Analytics** | pandas, numpy, scipy, scikit-learn | Data processing |
| **Visualization** | Plotly, Matplotlib | Charts and graphs |

### Key Python Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
ibapi>=10.19.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
anthropic>=0.18.0
plotly>=5.18.0
scipy>=1.11.0
scikit-learn>=1.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-docx>=0.8.11
openpyxl>=3.1.0
```

---

## Directory Structure

```
HH_research_platform_v1/
├── dashboard/                    # Streamlit UI
│   ├── app.py                   # Main dashboard entry point
│   ├── portfolio_tab.py         # Portfolio management UI
│   ├── signals_tab.py           # Signal analysis UI (7 sub-tabs)
│   ├── chat_tab.py              # AI Chat interface
│   ├── trade_ideas_tab.py       # Trade recommendations
│   ├── backtest_tab.py          # Backtesting interface
│   └── components/              # Reusable UI components
│
├── src/
│   ├── core/                    # Core business logic
│   │   ├── unified_signal.py    # UnifiedSignal dataclass (central model)
│   │   ├── signal_engine.py     # Signal generation orchestrator
│   │   ├── trade_ideas.py       # Trade idea generation
│   │   ├── unified_scorer.py    # Scoring pipeline
│   │   └── repository.py        # Data access layer
│   │
│   ├── analytics/               # Analysis modules
│   │   ├── gex_analysis.py      # GEX/Gamma exposure analysis
│   │   ├── dark_pool.py         # Dark pool flow proxy
│   │   ├── cross_asset.py       # Cross-asset signals
│   │   ├── sentiment_nlp.py     # Sentiment analysis (Qwen3)
│   │   ├── earnings_whisper.py  # Earnings expectations
│   │   ├── insider_tracker.py   # SEC Form 4 tracking
│   │   ├── institutional_13f_tracker.py  # 13F holdings (Buffett, etc.)
│   │   ├── factor_decomposition.py       # Factor analysis
│   │   ├── crowding_detector.py          # Crowded trade detection
│   │   ├── regime_analysis.py            # Market regime detection
│   │   ├── options_flow.py               # Options flow analysis
│   │   ├── pairs_correlation.py          # Correlation/pairs finder
│   │   ├── signal_performance.py         # Signal accuracy tracking
│   │   └── economic_calendar.py          # Economic events
│   │
│   ├── broker/                  # Broker integrations
│   │   ├── ibkr_client.py       # IBKR connection manager
│   │   ├── ibkr_options.py      # Real-time options from IBKR
│   │   └── execution.py         # Order execution
│   │
│   ├── db/                      # Database layer
│   │   ├── connection.py        # PostgreSQL connection
│   │   ├── models.py            # SQLAlchemy models
│   │   └── migrations/          # Schema migrations
│   │
│   ├── ai/                      # AI/LLM integration
│   │   ├── chat.py              # Claude chat handler
│   │   ├── context_builder.py   # Portfolio context for AI
│   │   └── tools.py             # AI tool definitions
│   │
│   └── utils/                   # Utilities
│       ├── logging.py           # Logging configuration
│       ├── config.py            # App configuration
│       └── helpers.py           # Common helpers
│
├── config/
│   ├── settings.yaml            # Main configuration
│   └── .env                     # Environment variables (secrets)
│
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
└── requirements.txt             # Python dependencies
```

---

## Core Modules

### 1. UnifiedSignal (`src/core/unified_signal.py`)

The **central data model** for all signal information. Every analysis flows into this structure.

```python
@dataclass
class UnifiedSignal:
    # Identification
    ticker: str
    generated_at: datetime
    
    # Price Data
    current_price: float
    price_change_1d: float
    price_change_5d: float
    
    # === CORE SCORES (0-100) ===
    technical_score: int
    fundamental_score: int
    sentiment_score: int
    options_score: int
    
    # === TECHNICAL INDICATORS ===
    rsi: float
    macd_signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    bb_position: str  # 'UPPER', 'MIDDLE', 'LOWER'
    adx: float
    trend: str
    
    # === FUNDAMENTAL DATA ===
    pe_ratio: float
    pb_ratio: float
    roe: float
    debt_to_equity: float
    revenue_growth: float
    
    # === OPTIONS DATA ===
    iv_rank: float
    put_call_ratio: float
    max_pain: float
    options_sentiment: str
    
    # === EARNINGS (IES/ECS/EQS System) ===
    days_to_earnings: int
    earnings_surprise_history: List[float]
    ies_score: float  # Implied Expectations Score
    ecs_score: float  # Earnings Catalyst Score
    eqs_score: float  # Earnings Quality Score
    
    # === INSTITUTIONAL SIGNALS (Phase 2-4) ===
    
    # GEX/Gamma Exposure
    gex_score: int = 50
    gex_signal: str = 'NEUTRAL'  # 'BULLISH', 'BEARISH', 'PINNED', 'NEUTRAL'
    gex_regime: str = 'NEUTRAL'  # 'POSITIVE_GEX', 'NEGATIVE_GEX', 'NEUTRAL'
    gex_reason: str = ''
    
    # Dark Pool Activity
    dark_pool_score: int = 50
    dark_pool_signal: str = 'NEUTRAL'  # 'ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL'
    institutional_bias: str = 'NEUTRAL'  # 'BUYING', 'SELLING', 'NEUTRAL'
    dark_pool_reason: str = ''
    
    # Cross-Asset Signals
    cross_asset_score: int = 50
    cross_asset_signal: str = 'NEUTRAL'  # 'RISK_ON', 'RISK_OFF', 'NEUTRAL'
    cycle_phase: str = ''  # Economic cycle phase
    cross_asset_reason: str = ''
    
    # Sentiment NLP (Qwen3)
    sentiment_nlp_score: int = 50
    sentiment_nlp_signal: str = 'NEUTRAL'
    sentiment_nlp_reason: str = ''
    
    # Earnings Whisper
    whisper_score: int = 50
    whisper_signal: str = 'NEUTRAL'  # 'BEAT_EXPECTED', 'MISS_EXPECTED', 'NEUTRAL'
    whisper_reason: str = ''
    
    # Insider Trading (Form 4)
    insider_score: int = 50
    insider_signal: str = 'NEUTRAL'  # 'STRONG_BUY' to 'STRONG_SELL'
    insider_ceo_bought: bool = False
    insider_cfo_bought: bool = False
    insider_cluster_buying: bool = False
    insider_cluster_selling: bool = False
    insider_net_value: float = 0.0
    insider_reason: str = ''
    
    # 13F Institutional Holdings
    inst_13f_score: int = 50
    inst_13f_signal: str = 'NEUTRAL'
    inst_buffett_owns: bool = False
    inst_buffett_added: bool = False
    inst_activist_involved: bool = False
    inst_notable_holders: List[str] = field(default_factory=list)
    inst_13f_reason: str = ''
    
    # === FINAL OUTPUT ===
    composite_score: int  # Weighted combination of all scores
    signal_type: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    confidence: float  # 0.0 to 1.0
    components_available: List[str]  # Which modules contributed
```

### 2. SignalEngine (`src/core/signal_engine.py`)

**Orchestrates all signal generation**. Calls each analytics module and assembles the UnifiedSignal.

```python
class SignalEngine:
    """
    Central signal generation engine.
    
    Workflow:
    1. Load base scores from database (technical, fundamental, sentiment, options)
    2. Call each institutional module
    3. Apply unified scoring weights
    4. Generate final signal type and confidence
    5. Save snapshot to database
    """
    
    def generate_signal(self, ticker: str) -> UnifiedSignal:
        """Generate complete signal for a ticker."""
        
        # 1. Technical Analysis
        technical = self._analyze_technical(ticker)
        
        # 2. Fundamental Analysis
        fundamental = self._analyze_fundamental(ticker)
        
        # 3. Sentiment Analysis
        sentiment = self._analyze_sentiment(ticker)
        
        # 4. Options Analysis
        options = self._analyze_options(ticker)
        
        # 5. Earnings Intelligence
        earnings = self._analyze_earnings(ticker)
        
        # 5.5 INSTITUTIONAL SIGNALS (added Phase 2-4)
        gex = analyze_gex(ticker)
        dark_pool = analyze_dark_pool(ticker)
        cross_asset = get_cross_asset_signals(ticker)
        sentiment_nlp = analyze_news_sentiment(headlines)
        whisper = get_earnings_whisper(ticker)
        insider = get_insider_signal(ticker)
        inst_13f = get_institutional_ownership(ticker)
        
        # 6. Calculate composite score
        composite = self._calculate_composite(...)
        
        # 7. Determine signal type
        signal_type = self._get_signal_type(composite)
        
        return UnifiedSignal(...)
    
    def save_signal_snapshot(self, signal: UnifiedSignal):
        """Save signal to database with full JSON for historical analysis."""
        # Saved to signal_snapshots table
        # full_signal_json contains ALL fields for performance tracking
```

### 3. TradeIdeasGenerator (`src/core/trade_ideas.py`)

**Generates actionable trade recommendations** with position sizing and risk controls.

```python
class TradeIdeasGenerator:
    """
    Generate trade ideas with:
    - Signal-based filtering
    - Position sizing (Kelly criterion adjusted)
    - Transaction cost modeling
    - Exposure controls (sector limits, position limits)
    - Portfolio correlation check
    """
    
    def generate_ideas(self, signals: List[UnifiedSignal], 
                       portfolio: Portfolio) -> List[TradeIdea]:
        
        ideas = []
        for signal in signals:
            # 1. Check if signal meets threshold
            if signal.composite_score < self.min_score:
                continue
            
            # 2. Check exposure limits
            if self._would_exceed_sector_limit(signal.ticker, portfolio):
                continue
            
            # 3. Calculate position size
            size = self._calculate_position_size(signal, portfolio)
            
            # 4. Estimate transaction costs
            costs = self._estimate_costs(signal.ticker, size)
            
            # 5. Check correlation with existing positions
            correlation_risk = self._check_correlation(signal.ticker, portfolio)
            
            ideas.append(TradeIdea(
                ticker=signal.ticker,
                action='BUY' if signal.signal_type in ['STRONG_BUY', 'BUY'] else 'SELL',
                size=size,
                estimated_cost=costs,
                signal=signal,
                correlation_risk=correlation_risk
            ))
        
        return ideas
```

---

## Signal System

### Signal Generation Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                       SIGNAL GENERATION PIPELINE                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐                                                     │
│  │   Ticker    │                                                     │
│  │   Input     │                                                     │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    PARALLEL ANALYSIS                             ││
│  │                                                                  ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │Technical │ │Fundamental│ │Sentiment │ │ Options  │           ││
│  │  │  Score   │ │  Score   │ │  Score   │ │  Score   │           ││
│  │  │  0-100   │ │  0-100   │ │  0-100   │ │  0-100   │           ││
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           ││
│  │       │            │            │            │                   ││
│  │  ┌────▼────────────▼────────────▼────────────▼────┐             ││
│  │  │           INSTITUTIONAL SIGNALS                 │             ││
│  │  │                                                 │             ││
│  │  │  GEX │ Dark Pool │ Cross-Asset │ Insider │ 13F │             ││
│  │  │                                                 │             ││
│  │  └────────────────────┬────────────────────────────┘             ││
│  │                       │                                          ││
│  └───────────────────────┼──────────────────────────────────────────┘│
│                          │                                           │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    UNIFIED SCORER                                ││
│  │                                                                  ││
│  │   Weights:                                                       ││
│  │   - Technical: 25%                                              ││
│  │   - Fundamental: 25%                                            ││
│  │   - Sentiment: 15%                                              ││
│  │   - Options: 15%                                                ││
│  │   - Institutional: 20%                                          ││
│  │                                                                  ││
│  │   Adjustments:                                                   ││
│  │   - Regime-based weight shifts                                  ││
│  │   - Earnings proximity boost/penalty                            ││
│  │   - Correlation penalty for concentrated bets                   ││
│  │                                                                  ││
│  └────────────────────────┬────────────────────────────────────────┘│
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    SIGNAL OUTPUT                                 ││
│  │                                                                  ││
│  │   composite_score: 0-100                                        ││
│  │   signal_type: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL    ││
│  │   confidence: 0.0-1.0                                           ││
│  │                                                                  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Signal Type Thresholds

| Score Range | Signal Type | Action |
|-------------|-------------|--------|
| 80-100 | STRONG_BUY | High conviction long |
| 65-79 | BUY | Standard long |
| 45-64 | HOLD | No action |
| 30-44 | SELL | Reduce/exit |
| 0-29 | STRONG_SELL | High conviction short/avoid |

### Signal Storage

Signals are stored in two tables:

1. **`trading_signals`**: Daily signal records with returns tracking
2. **`signal_snapshots`**: Full JSON snapshots for detailed analysis

```sql
-- signal_snapshots table
CREATE TABLE signal_snapshots (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    snapshot_date DATE,
    price_at_snapshot DECIMAL(10,2),
    today_signal VARCHAR(20),
    today_score INTEGER,
    full_signal_json JSONB,  -- Contains ALL UnifiedSignal fields
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Institutional Signal Modules

### Phase 2: Market Microstructure

#### 1. GEX Analysis (`src/analytics/gex_analysis.py`)

**Gamma Exposure Analysis** - Measures dealer hedging flows that affect price.

```python
def analyze_gex(ticker: str) -> Dict:
    """
    Calculate net gamma exposure across the options chain.
    
    Outputs:
    - gex_score: 0-100 (higher = more bullish positioning)
    - gex_signal: 'BULLISH', 'BEARISH', 'PINNED', 'NEUTRAL'
    - gex_regime: 'POSITIVE_GEX' (dealers buy dips) or 'NEGATIVE_GEX' (dealers sell into strength)
    - key_levels: Gamma flip points, major strike walls
    
    Interpretation:
    - Positive GEX: Market makers are net short gamma, will buy on dips
    - Negative GEX: Market makers are net long gamma, will sell into rallies
    - High gamma at strike: Price tends to get "pinned" there
    """
```

#### 2. Dark Pool Analysis (`src/analytics/dark_pool.py`)

**Institutional Order Flow Proxy** - Detects accumulation/distribution patterns.

```python
def analyze_dark_pool(ticker: str) -> Dict:
    """
    Proxy for dark pool activity using volume analysis.
    
    Signals:
    - ACCUMULATION: Large blocks being bought (bullish)
    - DISTRIBUTION: Large blocks being sold (bearish)
    - NEUTRAL: No clear pattern
    
    Methodology:
    - Block trade detection (volume spikes)
    - Price-volume divergence
    - Relative volume vs historical
    """
```

#### 3. Cross-Asset Signals (`src/analytics/cross_asset.py`)

**Inter-market analysis** for macro context.

```python
def get_cross_asset_signals(ticker: str) -> Dict:
    """
    Analyze cross-asset relationships.
    
    Inputs:
    - VIX level and trend
    - Treasury yields (2Y, 10Y, yield curve)
    - USD index
    - Credit spreads
    - Sector rotation patterns
    
    Output:
    - cycle_phase: 'EARLY_CYCLE', 'MID_CYCLE', 'LATE_CYCLE', 'RECESSION'
    - risk_regime: 'RISK_ON', 'RISK_OFF'
    - sector_favor: Which sectors are favored in current regime
    """
```

### Phase 3: Alternative Data

#### 4. Sentiment NLP (`src/analytics/sentiment_nlp.py`)

**News Sentiment Analysis** using local Qwen3-32B model.

```python
def analyze_news_sentiment(headlines: List[str]) -> Dict:
    """
    Analyze news headlines using Qwen3-32B via llama.cpp.
    
    Process:
    1. Fetch recent headlines from news_articles table
    2. Send to local LLM for sentiment analysis
    3. Aggregate sentiment scores
    
    Output:
    - sentiment_score: -100 to +100
    - sentiment_signal: 'VERY_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'VERY_BEARISH'
    - key_themes: Extracted topics from news
    """
```

#### 5. Earnings Whisper (`src/analytics/earnings_whisper.py`)

**Earnings Expectations Intelligence** - Beyond consensus estimates.

```python
def get_earnings_whisper(ticker: str) -> Dict:
    """
    The "whisper number" - what the market actually expects vs published consensus.
    
    Components:
    - IES (Implied Expectations Score): Options-implied move vs historical
    - ECS (Earnings Catalyst Score): How much earnings matters for this stock
    - EQS (Earnings Quality Score): Reliability of earnings
    
    Solves the "great report, bad trade" problem:
    - Stock can beat estimates but miss whisper → stock falls
    - Stock can miss estimates but beat whisper → stock rises
    """
```

### Phase 4: Smart Money Tracking

#### 6. Insider Tracker (`src/analytics/insider_tracker.py`)

**SEC Form 4 Analysis** - Track insider buying/selling.

```python
def get_insider_signal(ticker: str, days_back: int = 90) -> Dict:
    """
    Fetch and analyze SEC Form 4 filings.
    
    High-value signals:
    - CEO/CFO open market purchases (very bullish)
    - Cluster buying (multiple insiders buying)
    - Cluster selling (multiple insiders selling)
    - Large transactions relative to holdings
    
    Output:
    - insider_score: 0-100
    - insider_signal: 'STRONG_BUY' to 'STRONG_SELL'
    - insider_ceo_bought: bool
    - insider_cfo_bought: bool
    - insider_cluster_buying: bool
    - insider_cluster_selling: bool
    - insider_net_value: Net $ bought/sold
    """
```

#### 7. 13F Institutional Tracker (`src/analytics/institutional_13f_tracker.py`)

**Hedge Fund Holdings Analysis** - Track what the best investors own.

```python
def get_institutional_ownership(ticker: str) -> Dict:
    """
    Analyze 13F filings from top investors.
    
    Tracked funds (16 total):
    - Berkshire Hathaway (Buffett)
    - Bridgewater Associates
    - Renaissance Technologies
    - Citadel
    - Two Sigma
    - DE Shaw
    - Millennium Management
    - Point72
    - Appaloosa Management
    - Pershing Square (Ackman)
    - Third Point
    - ValueAct Capital
    - Elliott Management
    - Icahn Enterprises
    - Baupost Group
    - Tiger Global
    
    Output:
    - inst_13f_score: 0-100
    - inst_buffett_owns: bool
    - inst_buffett_added: bool (position increased)
    - inst_activist_involved: bool
    - inst_notable_holders: List of fund names
    """
```

### Phase 1: Risk Infrastructure

#### 8. Factor Decomposition (`src/analytics/factor_decomposition.py`)

**Attribution analysis** - Understand return drivers.

```python
def decompose_returns(ticker: str) -> Dict:
    """
    Decompose returns into factor exposures.
    
    Factors:
    - Market (SPY beta)
    - Size (SMB)
    - Value (HML)
    - Momentum
    - Quality
    - Volatility
    
    Output:
    - factor_exposures: Dict[factor_name, beta]
    - residual_return: Alpha after factor adjustment
    - r_squared: How much explained by factors
    """
```

#### 9. Crowding Detector (`src/analytics/crowding_detector.py`)

**Detect crowded trades** that may reverse.

```python
def detect_crowding(ticker: str) -> Dict:
    """
    Identify crowded positions that may snap back.
    
    Indicators:
    - Short interest vs float
    - Days to cover
    - Institutional ownership concentration
    - ETF ownership (passive flows)
    - Recent 13F filing changes
    
    Output:
    - crowding_score: 0-100 (higher = more crowded)
    - crowding_type: 'LONG_CROWDED', 'SHORT_CROWDED', 'NORMAL'
    - reversal_risk: float (probability of mean reversion)
    """
```

#### 10. Regime Analysis (`src/analytics/regime_analysis.py`)

**Market regime detection** for adaptive strategies.

```python
def analyze_regime() -> Dict:
    """
    Detect current market regime.
    
    Regimes:
    - BULL_LOW_VOL: Trending up, low volatility
    - BULL_HIGH_VOL: Trending up, high volatility
    - BEAR_LOW_VOL: Trending down, low volatility
    - BEAR_HIGH_VOL: Trending down, high volatility (crisis)
    - RANGE_BOUND: No clear trend
    
    Uses:
    - VIX level and term structure
    - Moving average relationships
    - Breadth indicators
    - Credit spreads
    """
```

### Phase 5: Portfolio Tools

#### 11. Pairs/Correlation Finder (`src/analytics/pairs_correlation.py`)

**Find correlated pairs** for hedging and pairs trading.

```python
def find_correlated_pairs(tickers: List[str]) -> Dict:
    """
    Calculate correlation matrix and find trading pairs.
    
    Output:
    - correlation_matrix: Full NxN correlation matrix
    - highly_correlated_pairs: Pairs with corr > 0.7
    - negative_correlations: Pairs with corr < -0.3 (hedge candidates)
    - diversification_ratio: Portfolio diversification metric
    - clusters: Groups of correlated stocks
    """

def suggest_hedges(ticker: str, portfolio: List[str]) -> List[Dict]:
    """
    Suggest hedge instruments for a position.
    
    Candidates:
    - Inverse ETFs (SH, SDS, SQQQ)
    - Sector ETFs (short the sector)
    - Individual stocks with negative correlation
    - Options strategies
    """
```

#### 12. Portfolio Optimizer (`src/analytics/portfolio_optimizer.py`)

**Modern Portfolio Theory** optimization.

```python
def optimize_portfolio(tickers: List[str], 
                       constraints: Dict) -> Dict:
    """
    Optimize portfolio weights using various strategies.
    
    Strategies:
    - MAX_SHARPE: Maximize risk-adjusted return
    - MIN_VARIANCE: Minimize portfolio volatility
    - RISK_PARITY: Equal risk contribution
    - MAX_DIVERSIFICATION: Maximize diversification ratio
    - MEAN_VARIANCE: Classic Markowitz
    
    Constraints:
    - min_weight, max_weight per position
    - sector limits
    - total exposure limits
    """
```

---

## Risk Analytics

### VaR Calculation (`src/analytics/risk_dashboard.py`)

```python
def calculate_var(portfolio: Portfolio, 
                  confidence: float = 0.95,
                  horizon_days: int = 1) -> Dict:
    """
    Calculate Value at Risk.
    
    Methods:
    - Historical VaR: Based on actual return distribution
    - Parametric VaR: Assumes normal distribution
    - Monte Carlo VaR: Simulated scenarios
    
    Output:
    - var_95: 95% VaR in dollars
    - var_99: 99% VaR in dollars
    - cvar_95: Conditional VaR (expected shortfall)
    - worst_case_loss: Maximum historical drawdown
    """
```

### Position Concentration

```python
def analyze_concentration(portfolio: Portfolio) -> Dict:
    """
    Analyze portfolio concentration risk.
    
    Metrics:
    - top_5_weight: Weight in top 5 positions
    - herfindahl_index: Concentration measure
    - sector_weights: Exposure by sector
    - largest_position: Single stock concentration
    """
```

---

## Portfolio Management

### IBKR Integration (`src/broker/ibkr_client.py`)

```python
class IBKRClient:
    """
    Interactive Brokers connection manager.
    
    Features:
    - Real-time portfolio positions
    - Account balances and P&L
    - Order execution
    - Real-time market data
    - Options chain (real-time via ibkr_options.py)
    
    Connection:
    - Default port: 7496 (TWS) or 4001 (Gateway)
    - Client ID: Configurable to avoid conflicts
    """
    
    def get_positions(self) -> List[Position]:
        """Get current portfolio positions."""
    
    def get_account_summary(self) -> Dict:
        """Get account balances, buying power, P&L."""
    
    def place_order(self, order: Order) -> str:
        """Submit order to IBKR."""
```

### Real-Time Options (`src/broker/ibkr_options.py`)

```python
class OptionsDataFetcher:
    """
    Fetch options data with IBKR first, Yahoo Finance fallback.
    
    Features:
    - Real-time Greeks (Delta, Gamma, Vega, Theta)
    - Live bid/ask spreads
    - Smart filtering (near-the-money only for speed)
    - Data source tracking (shows "IBKR" or "YAHOO")
    
    Optimization:
    - Only fetches strikes within ±15% of stock price
    - Limits to 2 nearest expiries
    - ~200 contracts instead of 5000+ for AAPL
    """
    
    def get_options_chain(self, ticker: str) -> OptionsChainResult:
        """
        Returns:
        - calls: DataFrame of call options
        - puts: DataFrame of put options
        - stock_price: Current price
        - data_source: DataSource.IBKR or DataSource.YAHOO
        """
```

---

## AI Integration

### Claude Chat (`src/ai/chat.py`)

```python
class ChatHandler:
    """
    AI chat interface powered by Claude API.
    
    Features:
    - Full portfolio context injection
    - Real-time signal data
    - Web search capability
    - File upload support (PDF, images, documents)
    - Conversation memory
    
    Context includes:
    - Current positions with P&L
    - Today's signals for portfolio
    - Recent trade ideas
    - Market conditions
    - Institutional signals (GEX, Insider, 13F, etc.)
    """
    
    def chat(self, message: str, 
             include_portfolio: bool = True,
             include_signals: bool = True) -> str:
        """Send message to Claude with context."""
```

### Context Builder (`src/ai/context_builder.py`)

```python
def build_portfolio_context(portfolio: Portfolio) -> str:
    """
    Build rich context for AI.
    
    Includes:
    - Position list with weights and P&L
    - Sector exposures
    - Risk metrics (VaR, concentration)
    - Today's signals with institutional data
    - Recent insider activity
    - 13F holdings overlap
    - Correlation clusters
    """
```

### Local LLM (Qwen3-32B)

```python
# Sentiment NLP uses local Qwen3-32B via llama.cpp
# Hardware: AMD Ryzen 9 9950X, 128GB RAM, RTX 5090

def call_local_llm(prompt: str) -> str:
    """
    Call local Qwen3-32B for sentiment analysis.
    
    Endpoint: http://localhost:8080/completion
    Model: Qwen3-32B Q4_K_M quantization
    """
```

---

## Database Schema

### Core Tables

```sql
-- Stock universe
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT true
);

-- Historical scores (daily)
CREATE TABLE historical_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    technical_score INTEGER,
    fundamental_score INTEGER,
    sentiment_score INTEGER,
    options_score INTEGER,
    composite_score INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Trading signals with performance tracking
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    signal_type VARCHAR(20),  -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    signal_strength INTEGER,
    entry_price DECIMAL(10,2),
    return_2d DECIMAL(8,4),
    return_5d DECIMAL(8,4),
    return_10d DECIMAL(8,4),
    return_30d DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Full signal snapshots (JSON)
CREATE TABLE signal_snapshots (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    snapshot_date DATE,
    price_at_snapshot DECIMAL(10,2),
    today_signal VARCHAR(20),
    today_score INTEGER,
    full_signal_json JSONB,  -- Complete UnifiedSignal as JSON
    created_at TIMESTAMP DEFAULT NOW()
);

-- News articles for sentiment
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP,
    sentiment_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Options flow alerts
CREATE TABLE options_flow (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    alert_date DATE,
    alert_type VARCHAR(50),
    direction VARCHAR(20),
    severity VARCHAR(20),
    option_type VARCHAR(10),
    strike DECIMAL(10,2),
    expiry DATE,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DECIMAL(6,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Backtest results
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100),
    run_date TIMESTAMP,
    start_date DATE,
    end_date DATE,
    initial_capital DECIMAL(15,2),
    final_value DECIMAL(15,2),
    total_return DECIMAL(8,4),
    sharpe_ratio DECIMAL(6,3),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(6,4),
    total_trades INTEGER,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Dashboard UI

### Main Dashboard (`dashboard/app.py`)

Streamlit-based UI with multiple tabs:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HH RESEARCH PLATFORM                             │
├─────────────────────────────────────────────────────────────────────────┤
│  [Portfolio] [Signals] [Trade Ideas] [Backtest] [AI Chat] [Settings]    │
├─────────────────────────────────────────────────────────────────────────┤
```

### Tab 1: Portfolio (`dashboard/portfolio_tab.py`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO OVERVIEW                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Account Summary          │  Positions Table                            │
│  ─────────────────────   │  ─────────────────────────────────────────  │
│  Net Liquidation: $1.3M   │  Ticker │ Shares │ Value │ P&L │ Weight    │
│  Day P&L: +$5,234        │  AAPL   │ 100    │ $27K  │ +5% │ 2.1%      │
│  Total P&L: +$150K       │  MSFT   │ 50     │ $21K  │ +3% │ 1.6%      │
│  Buying Power: $200K     │  ...                                         │
│                          │                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  🔗 Portfolio Correlation Analysis [Expandable]                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  [Risk Summary] [Heatmap] [Hedging]                               │  │
│  │                                                                    │  │
│  │  Risk Summary:                                                    │  │
│  │  - Avg Pairwise Correlation: 0.45                                │  │
│  │  - Highly Correlated Pairs: 12                                   │  │
│  │  - Diversification Ratio: 1.8                                    │  │
│  │  - Most Correlated: AAPL-MSFT (0.82)                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 2: Signals (`dashboard/signals_tab.py`)

**7 Sub-Tabs:**

1. **Overview**: Composite scores, signal distribution
2. **Technical**: RSI, MACD, Bollinger Bands, ADX charts
3. **Fundamental**: PE, PB, ROE, Debt metrics
4. **Sentiment**: News sentiment, social mentions
5. **Options**: IV rank, put/call ratios, unusual activity
6. **Institutional**: GEX, Dark Pool, Insider, 13F data
7. **Earnings**: IES/ECS/EQS scores, earnings calendar

### Tab 3: Trade Ideas

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TODAY'S TRADE IDEAS                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Strong Buys (3)          │  Trade Details                              │
│  ─────────────────────    │  ─────────────────────────────────────────  │
│  ⭐ NVDA - Score: 85      │  NVDA Analysis:                             │
│  ⭐ META - Score: 82      │                                              │
│  ⭐ GOOGL - Score: 80     │  📊 Technical: 78/100 (BULLISH)            │
│                           │  📈 Fundamental: 75/100 (STRONG)           │
│  Buys (5)                 │  💬 Sentiment: 82/100 (POSITIVE)           │
│  ─────────────────────    │  📉 Options: 90/100 (CALL FLOW)            │
│  AAPL - Score: 72         │                                              │
│  MSFT - Score: 70         │  🏛️ Institutional:                          │
│  ...                      │  - GEX: BULLISH (dealer buying dips)        │
│                           │  - Insider: CEO bought $2M last week        │
│                           │  - 13F: Buffett added position              │
│                           │                                              │
│                           │  💰 Suggested: Buy 50 shares                │
│                           │  📍 Entry: $875.00                          │
│                           │  🎯 Target: $950.00 (+8.6%)                 │
│                           │  🛑 Stop: $825.00 (-5.7%)                   │
│                           │                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 4: Backtest

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKTESTING ENGINE                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Strategy: [Signal-Based ▼]    Period: [2024-01-01] to [2024-12-27]    │
│                                                                          │
│  Parameters:                                                             │
│  - Min Signal Score: [70]                                               │
│  - Position Size: [Equal Weight ▼]                                      │
│  - Max Positions: [20]                                                  │
│  - Rebalance: [Weekly ▼]                                                │
│                                                                          │
│  [▶ Run Backtest]                                                        │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════════│
│                                                                          │
│  Results:                                                                │
│  ─────────────────────────────────────────────────────────────────────  │
│  Total Return: +45.2%     │  Sharpe Ratio: 1.42                        │
│  Max Drawdown: -12.3%     │  Win Rate: 62%                             │
│  Alpha vs SPY: +18.5%     │  Total Trades: 156                         │
│                                                                          │
│  [Equity Curve Chart]                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 5: AI Chat

```
┌─────────────────────────────────────────────────────────────────────────┐
│  AI RESEARCH ASSISTANT (Claude + Portfolio Context)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ You: What's your view on my tech exposure?                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Claude: Looking at your portfolio, tech represents 35% of your    │  │
│  │ holdings, concentrated in AAPL (8%), MSFT (7%), NVDA (6%).       │  │
│  │                                                                    │  │
│  │ Key observations:                                                 │  │
│  │ • High correlation cluster (AAPL-MSFT: 0.82)                     │  │
│  │ • GEX is positive for all three - dealers will buy dips          │  │
│  │ • Insider activity: NVDA CFO sold $5M (routine)                  │  │
│  │ • 13F: Buffett trimmed AAPL but still largest holding            │  │
│  │                                                                    │  │
│  │ Recommendation: Consider adding hedges given concentration.       │  │
│  │ QQQ puts or SQQQ could reduce your tech beta.                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  [Type your question...]                                      [Send]    │
│                                                                          │
│  ☑ Include portfolio context  ☑ Include signals  ☑ Web search          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Data Type | Source | Update Frequency | Cost |
|-----------|--------|------------------|------|
| Stock Prices | Yahoo Finance / IBKR | Real-time (IBKR) or 15min delay | Free |
| Options Chains | IBKR (primary) / Yahoo (fallback) | Real-time / 15min | Free |
| Fundamentals | Yahoo Finance | Daily | Free |
| Insider Trades | SEC EDGAR Form 4 | Daily | Free |
| 13F Holdings | SEC EDGAR 13F | Quarterly | Free |
| News | News APIs / Web scraping | Continuous | Free |
| Economic Calendar | investpy | Daily | Free |

---

## Configuration

### Environment Variables (`.env`)

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/HH

# IBKR
IBKR_HOST=127.0.0.1
IBKR_PORT=7496
IBKR_CLIENT_ID=1

# AI
ANTHROPIC_API_KEY=sk-ant-xxxxx
LOCAL_LLM_URL=http://localhost:8080

# Optional
NEWS_API_KEY=xxxxx
```

### Settings (`config/settings.yaml`)

```yaml
signal_weights:
  technical: 0.25
  fundamental: 0.25
  sentiment: 0.15
  options: 0.15
  institutional: 0.20

thresholds:
  strong_buy: 80
  buy: 65
  hold_min: 45
  sell: 30

position_limits:
  max_position_pct: 5.0
  max_sector_pct: 25.0
  min_position_size: 1000

transaction_costs:
  commission_per_share: 0.005
  spread_estimate_bps: 5
  market_impact_bps: 2
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Interactive Brokers TWS or IB Gateway
- NVIDIA GPU (optional, for local LLM)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/xxx/HH-research-platform.git
cd HH-research-platform

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install IBKR API
pip install ibapi

# 5. Setup database
createdb HH

# 6. Configure environment
cp config/.env.example config/.env
# Edit .env with your credentials

# 7. Run dashboard
streamlit run dashboard/app.py
```

### IBKR Setup

1. Download TWS or IB Gateway from Interactive Brokers
2. Enable API connections: Configure → API → Settings
3. Check "Enable ActiveX and Socket Clients"
4. Set port: 7496 (TWS) or 4001 (Gateway)
5. Add 127.0.0.1 to trusted IPs

---

## API Reference

### Signal Generation

```python
from src.core.signal_engine import SignalEngine

engine = SignalEngine()
signal = engine.generate_signal('AAPL')

print(f"Score: {signal.composite_score}")
print(f"Signal: {signal.signal_type}")
print(f"GEX: {signal.gex_signal}")
print(f"Insider: {signal.insider_signal}")
```

### Options Flow

```python
from src.analytics.options_flow import OptionsFlowAnalyzer

analyzer = OptionsFlowAnalyzer()
summary = analyzer.analyze_ticker('AAPL')

print(f"Data Source: {summary.data_source}")  # 'IBKR' or 'YAHOO'
print(f"Sentiment: {summary.overall_sentiment}")
print(f"Put/Call Ratio: {summary.put_call_volume_ratio}")
```

### Institutional Signals

```python
from src.analytics.insider_tracker import get_insider_signal
from src.analytics.institutional_13f_tracker import get_institutional_ownership

insider = get_insider_signal('AAPL')
print(f"CEO Bought: {insider['ceo_bought']}")

holdings = get_institutional_ownership('AAPL')
print(f"Buffett Owns: {holdings['buffett_owns']}")
```

### Signal Performance Tracking

```python
from src.analytics.signal_performance import (
    SignalPerformanceTracker,
    InstitutionalSignalTracker
)

# Regular signals
tracker = SignalPerformanceTracker()
summary = tracker.get_performance_summary(days_back=90)

# Institutional signals
inst_tracker = InstitutionalSignalTracker()
inst_tracker.print_leaderboard(days_back=90)
```

---

## Performance Tracking

### Signal Accuracy Measurement

The platform tracks win rates and returns for all signals:

```
🏆 INSTITUTIONAL SIGNAL LEADERBOARD
======================================================================

Top signals by 10-day win rate (last 90 days):

Signal              Value      Count   Win Rate 10d   Avg Return 10d
─────────────────────────────────────────────────────────────────────
Insider CEO Bought  YES        12      75.0%          +3.42%
Buffett Added       YES        8       71.4%          +2.89%
GEX Signal          BULLISH    45      64.2%          +1.56%
Dark Pool           ACCUMUL    34      61.8%          +1.23%
...
```

---

## Roadmap

### Completed ✅

- [x] Core signal engine
- [x] Technical/Fundamental/Sentiment/Options analysis
- [x] IBKR real-time integration
- [x] AI Chat with portfolio context
- [x] GEX/Gamma analysis
- [x] Dark pool proxy
- [x] Cross-asset signals
- [x] Sentiment NLP (local LLM)
- [x] Earnings whisper
- [x] Insider tracker (Form 4)
- [x] 13F institutional tracker
- [x] Factor decomposition
- [x] Crowding detection
- [x] Regime analysis
- [x] Pairs/Correlation finder
- [x] Portfolio correlation heatmap
- [x] IBKR real-time options (with Yahoo fallback)
- [x] Institutional signal performance tracking
- [x] Transaction cost modeling
- [x] Exposure controls

### Future Enhancements

- [ ] Level 2 order book data
- [ ] Real FINRA dark pool data
- [ ] Earnings transcript NLP
- [ ] Alert system (email/Slack/Discord)
- [ ] Mobile app
- [ ] Multi-strategy backtester
- [ ] Paper trading mode

---

## License

Proprietary - Personal Use Only

---

## Contact

For questions about this platform, contact the maintainer.

---

*Last Updated: December 27, 2025*