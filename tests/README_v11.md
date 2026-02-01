# HH Research Platform

## Complete Hedge Fund-Grade Trading Intelligence System

A comprehensive quantitative trading platform that combines AI-powered sentiment analysis, fundamental analysis, technical indicators, options flow, macro regime detection, earnings intelligence, bond trading, and performance tracking into a unified signal generation system with full backtesting capabilities.

**Author:** Hasan  
**Version:** 3.1  
**Last Updated:** December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Dashboard Tabs](#dashboard-tabs)
4. [Signal Generation Pipeline](#signal-generation-pipeline)
5. [Core Analytics Modules](#core-analytics-modules)
6. [Earnings Intelligence System](#earnings-intelligence-system)
7. [Bond Trading System](#bond-trading-system)
8. [AI Integration](#ai-integration)
9. [Caching Architecture](#caching-architecture)
10. [Database Schema](#database-schema)
11. [Configuration](#configuration)
12. [File Structure](#file-structure)
13. [API Reference](#api-reference)
14. [Running the Platform](#running-the-platform)
15. [Recent Updates](#recent-updates)
16. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Platform Does

The HH Research Platform is a professional-grade trading intelligence system that:

1. **Collects Data** from multiple sources (news, fundamentals, prices, options, SEC filings)
2. **Analyzes** using AI models (local Qwen LLM) and quantitative methods
3. **Generates Signals** (BUY/SELL/HOLD) with confidence scores
4. **Detects Market Regime** (Risk-On vs Risk-Off)
5. **Analyzes Earnings** with reaction prediction and beat/miss detection
6. **Tracks Performance** of all signals with win rates and returns
7. **Backtests Strategies** with 9+ predefined trading strategies
8. **Provides Trade Ideas** ranked by AI
9. **Trades Bonds** with yield-curve and Fed policy analysis
10. **Sends Alerts** via Telegram
11. **Integrates with IBKR** for live portfolio data
12. **Displays Everything** in a Streamlit dashboard

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Factor Scoring** | Sentiment + Fundamental + Technical + Options + Macro |
| **AI Committee** | Multiple AI agents vote on each stock |
| **Unified Signal Engine** | Combines 6 analyzers into single recommendation |
| **Options Flow Analysis** | Detect institutional positioning |
| **Short Squeeze Detection** | Find squeeze candidates |
| **Earnings Intelligence** | IES/EQS/ECS scores, reaction analysis, beat/miss detection |
| **Macro Regime Detection** | Risk-On/Risk-Off with signal adjustments |
| **Bond Trading** | Yield-based signals with institutional intelligence |
| **Trade Ideas Generator** | AI-ranked recommendations with entry/stop/target |
| **Performance Tracking** | Win rates, returns by signal type, attribution |
| **Strategy Backtesting** | 9 strategies with Sharpe/Sortino/Alpha metrics |
| **Signal Combinations** | Find which signal combos work best |
| **Trade Journal** | Log trades with signal attribution |
| **Telegram Alerts** | Real-time notifications |
| **IBKR Integration** | Live portfolio data and execution |

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.11+ |
| **Database** | PostgreSQL + TimescaleDB |
| **AI/LLM** | Qwen3-32B via llama.cpp (local) |
| **Data Sources** | Yahoo Finance, NewsAPI, Finnhub, Google News, Reddit |
| **Broker** | Interactive Brokers TWS API |
| **Alerts** | Telegram Bot API |
| **Hardware** | AMD Ryzen 9 9950X, 128GB RAM, NVIDIA RTX 5090 32GB |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT DASHBOARD                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │Universe │ │ Signals │ │DeepDive │ │Portfolio│ │Analytics│ │Performance│  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └─────┬─────┘  │
│       │          │          │          │          │            │           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│  │ Bonds   │ │Trade    │ │ System  │ │ AI Chat │                           │
│  │         │ │ Ideas   │ │ Status  │ │ (Qwen)  │                           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                           │
└───────┼──────────┼──────────┼──────────┼────────────────────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Trade Ideas  │  │ Bond Signal  │  │   AI Chat    │  │   Alerts     │    │
│  │  Generator   │  │  Generator   │  │   (Qwen)     │  │  (Telegram)  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYTICS LAYER                                    │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │  Options   │ │   Short    │ │  Earnings  │ │   Macro    │               │
│  │   Flow     │ │  Squeeze   │ │  Analyzer  │ │   Regime   │               │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘               │
│        │              │              │              │                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ Technical  │ │ Sentiment  │ │   Risk     │ │  Signal    │               │
│  │  Analysis  │ │  Analysis  │ │ Dashboard  │ │Performance │               │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘               │
└────────┼──────────────┼──────────────┼──────────────┼───────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     PostgreSQL + TimescaleDB                          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │  │screener_    │ │ trading_    │ │fundamentals │ │ earnings_   │     │  │
│  │  │  scores     │ │  signals    │ │             │ │  analysis   │     │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │  │news_        │ │ committee_  │ │ bond_       │ │ price_      │     │  │
│  │  │ articles    │ │ decisions   │ │ signals     │ │ history     │     │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     File-Based Cache (JSON)                           │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                     │  │
│  │  │ bond_       │ │ earnings_   │ │ macro_      │                     │  │
│  │  │ analysis    │ │ calendar    │ │ regime      │                     │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Yahoo   │ │ NewsAPI │ │  IBKR   │ │  SEC    │ │  Qwen   │ │Telegram │  │
│  │ Finance │ │ Finnhub │ │   TWS   │ │ EDGAR   │ │  LLM    │ │   Bot   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dashboard Tabs

### Tab 1: 📊 Universe
**Purpose:** Stock universe management and screening

**Features:**
- View all tracked tickers (144+ positions)
- Add/remove tickers
- Sector breakdown
- Quick filters by market cap, sector
- Bulk import from watchlists

**Data Displayed:**
- Ticker, Name, Sector
- Current Price, Change %
- Market Cap
- Last Analysis Date
- Signal Status

---

### Tab 2: 📈 Signals
**Purpose:** Main trading signals dashboard

**Features:**
- Full analysis pipeline (6 steps)
- 28-column data table
- Dropdown sorting (more reliable than headers)
- Stock selector for quick navigation
- Individual ticker refresh
- Smart caching with skip option

**Analysis Steps:**
1. Fetch news articles
2. Analyze sentiment (AI)
3. Get fundamentals
4. Run technical analysis
5. Check options flow
6. Detect short squeeze

**Key Columns:**
| Column | Description |
|--------|-------------|
| Signal | BUY/SELL/HOLD recommendation |
| Score | Composite score 0-100 |
| Confidence | High/Medium/Low |
| Sentiment | AI-analyzed news sentiment |
| Options | Institutional flow signal |
| Squeeze | Short squeeze risk |
| Earnings | Date + Beat/Miss indicator |
| Entry/Stop/Target | Trade setup prices |

---

### Tab 3: 🔍 Deep Dive
**Purpose:** Single-stock detailed analysis

**Sections:**
- Price chart with technicals
- News feed with sentiment
- Fundamental metrics
- Options chain analysis
- AI Committee verdict
- Trade recommendation

---

### Tab 4: 💼 Portfolio
**Purpose:** IBKR portfolio integration

**Features:**
- Live positions from TWS
- P&L tracking
- Position sizing
- Risk metrics
- Correlation matrix

---

### Tab 5: 📊 Analytics
**Purpose:** Market-wide analytics

**Sub-tabs:**
- Market Overview
- Sector Rotation
- Economic Calendar
- Macro Regime

---

### Tab 6: 🏦 Bonds
**Purpose:** Bond ETF trading signals with institutional intelligence

**Supported Instruments:**
| Ticker | Name | Duration |
|--------|------|----------|
| TLT | iShares 20+ Year Treasury | ~17 years |
| ZROZ | PIMCO 25+ Year Zero Coupon | ~27 years |
| EDV | Vanguard Extended Duration | ~24 years |
| TMF | Direxion 3x Long Treasury | Leveraged |
| SHY | iShares 1-3 Year Treasury | ~2 years |
| IEF | iShares 7-10 Year Treasury | ~8 years |

**Signal Factors:**
| Factor | Weight | Bullish When |
|--------|--------|--------------|
| Technical | 30% | RSI oversold, MACD bullish |
| Fundamental | 25% | Yields falling, curve inverted |
| Flow | 20% | Auction demand strong |
| Macro | 15% | Fed dovish, risk-off |
| Sentiment | 10% | News bullish on bonds |

**Features:**
- Treasury yield tracking (10Y, 30Y)
- Fed policy analysis
- Auction results
- MOVE Index (bond volatility)
- Curve trade analysis (steepener/flattener)
- AI Chat with full context
- Persistent cache (survives browser refresh)

**Data Flow:**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│   Generators     │────▶│   Dashboard     │
│                 │     │                  │     │                 │
│ • Yahoo Finance │     │ • BondSignalGen  │     │ • Unified View  │
│ • News APIs     │     │ • YieldGenerator │     │ • Signal Table  │
│ • Treasury Data │     │ • NewsAnalyzer   │     │ • AI Chat       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   JSON Cache     │
                        │                  │
                        │ • Signals        │
                        │ • Yields         │
                        │ • News + Articles│
                        │ • Intelligence   │
                        └──────────────────┘
```

---

### Tab 7: 💡 Trade Ideas
**Purpose:** AI-ranked trading opportunities

**Scoring Formula:**
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

**Output:**
- Ranked list of opportunities
- Entry/Stop/Target prices
- Risk/Reward ratio
- Catalyst explanation
- Timeframe (swing/position)

---

### Tab 8: 📊 Performance
**Purpose:** Unified performance tracking and backtesting

**Sub-tabs:**

#### 📈 Signal Performance
- Total signals analyzed
- Win rate by holding period (5d/10d/30d)
- Average returns by signal type
- Best/worst performers

#### 🔬 Strategy Backtester
9 predefined strategies:
| Strategy | Description |
|----------|-------------|
| Signal-Based | Trade on BUY/SELL signals |
| Momentum | RSI + MACD confirmation |
| Mean Reversion | Oversold bounces |
| Trend Following | MA crossovers |
| Earnings Drift | Post-earnings momentum |
| Options Flow | Follow smart money |
| Squeeze Play | Short squeeze candidates |
| Macro Aligned | Regime-adjusted |
| Combined Alpha | Best of all factors |

**Metrics:**
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Alpha vs SPY
- Win Rate

#### 🎯 Signal Combinations
- Find which signal combos work best
- Correlation analysis
- Multi-factor backtests

#### 📋 Trade Journal
- Log trades with signal attribution
- Performance by signal type
- Learning insights

---

### Tab 9: ⚙️ System Status
**Purpose:** System health monitoring

**Displays:**
- LLM status
- Database connection
- API rate limits
- Cache status
- Last update times

---

### Tab 10: 🤖 AI Chat
**Purpose:** Interactive AI assistant

**Context Includes:**
- Current portfolio
- Recent signals
- Market regime
- Open positions
- Earnings calendar

---

## Signal Generation Pipeline

### Unified Signal Engine

```python
class UnifiedSignalEngine:
    """Combines all analyzers into single recommendation"""
    
    def generate_signal(self, ticker: str) -> UnifiedSignal:
        # 1. Gather all scores
        sentiment = self.sentiment_analyzer.analyze(ticker)
        fundamental = self.fundamental_analyzer.analyze(ticker)
        technical = self.technical_analyzer.analyze(ticker)
        options = self.options_analyzer.analyze(ticker)
        squeeze = self.squeeze_analyzer.analyze(ticker)
        macro = self.macro_detector.get_regime()
        
        # 2. Apply weights and adjustments
        base_score = weighted_average(scores, weights)
        adjusted_score = apply_regime_adjustment(base_score, macro)
        
        # 3. Generate signal
        return UnifiedSignal(
            signal=score_to_signal(adjusted_score),
            confidence=calculate_confidence(scores),
            scores=all_component_scores
        )
```

### Scoring Weights

```python
SIGNAL_WEIGHTS = {
    'sentiment': 0.25,      # News sentiment
    'fundamental': 0.20,    # Value/growth metrics
    'technical': 0.20,      # Price patterns
    'options_flow': 0.15,   # Institutional flow
    'squeeze': 0.10,        # Short squeeze potential
    'macro': 0.10,          # Regime alignment
}

REGIME_ADJUSTMENTS = {
    'RISK_ON': {
        'growth': +10,
        'momentum': +10,
        'defensive': -5,
    },
    'RISK_OFF': {
        'growth': -15,
        'momentum': -10,
        'defensive': +15,
        'bonds': +10,
    }
}
```

### Signal Thresholds

| Signal | Score Range | Description |
|--------|-------------|-------------|
| STRONG_BUY | ≥ 80 | High conviction long |
| BUY | 65-79 | Moderate conviction long |
| HOLD | 35-64 | No clear edge |
| SELL | 20-34 | Moderate conviction short |
| STRONG_SELL | < 20 | High conviction short |

### Signal Confidence

```python
def calculate_confidence(scores: dict) -> str:
    # Based on score agreement and strength
    agreement = std_dev(scores.values())
    strength = abs(mean(scores.values()) - 50)
    
    if agreement < 10 and strength > 25:
        return "HIGH"
    elif agreement < 20 and strength > 15:
        return "MEDIUM"
    else:
        return "LOW"
```

---

## Core Analytics Modules

### 1. Sentiment Analysis (`sentiment.py`)

**Sources:**
- NewsAPI (100/day free tier)
- Finnhub (60/min)
- Google News (unofficial)
- Reddit (PRAW)

**Process:**
1. Fetch articles (last 7 days)
2. Filter by relevance
3. AI-analyze each article (Qwen)
4. Aggregate scores

**Output:**
```python
SentimentResult(
    score=75,           # 0-100
    label="BULLISH",    # VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH
    article_count=15,
    bullish_count=10,
    bearish_count=2,
    neutral_count=3,
    key_themes=["earnings beat", "guidance raised"],
    articles=[...]
)
```

---

### 2. Fundamental Analysis (`fundamentals.py`)

**Metrics Analyzed:**
| Metric | Bullish | Bearish |
|--------|---------|---------|
| P/E Ratio | < 15 | > 30 |
| PEG Ratio | < 1 | > 2 |
| Revenue Growth | > 20% | < 0% |
| Profit Margin | > 15% | < 5% |
| Debt/Equity | < 0.5 | > 2 |
| ROE | > 15% | < 5% |

**Scoring:**
```python
def score_fundamentals(data: dict) -> int:
    score = 50  # Base
    
    # Value score
    if data['pe_ratio'] < 15:
        score += 10
    elif data['pe_ratio'] > 30:
        score -= 10
    
    # Growth score
    if data['revenue_growth'] > 0.20:
        score += 15
    
    # Quality score
    if data['profit_margin'] > 0.15:
        score += 10
    
    return clamp(score, 0, 100)
```

---

### 3. Technical Analysis (`technical_analysis.py`)

**Indicators:**
| Indicator | Bullish | Bearish |
|-----------|---------|---------|
| RSI (14) | < 30 | > 70 |
| MACD | Bullish crossover | Bearish crossover |
| Price vs SMA50 | Above | Below |
| Price vs SMA200 | Above | Below |
| Volume | Above average | Below average |
| Bollinger Bands | Near lower | Near upper |

**Output:**
```python
TechnicalResult(
    score=65,
    trend="UPTREND",
    rsi=45,
    macd_signal="BULLISH",
    support=140.00,
    resistance=155.00,
    volume_signal="ABOVE_AVERAGE"
)
```

---

### 4. Options Flow Analysis (`options_flow.py`)

**Signals Detected:**
- Unusual volume (> 2x average)
- Put/Call ratio extremes
- Large block trades
- Sweep orders

**Scoring:**
```python
Options Score = (
    Call_Volume_Weight * call_score +
    Put_Volume_Weight * put_score +
    Unusual_Activity_Bonus +
    Block_Trade_Signal
)

Sentiment:
- VERY_BULLISH: Score >= 70
- BULLISH: Score >= 55
- NEUTRAL: Score 45-55
- BEARISH: Score <= 45
- VERY_BEARISH: Score <= 30
```

---

### 5. Short Squeeze Analysis (`short_squeeze.py`)

**Factors:**
| Factor | High Risk | Low Risk |
|--------|-----------|----------|
| Short % of Float | > 20% | < 5% |
| Days to Cover | > 5 | < 2 |
| Borrow Fee | > 50% | < 5% |
| Recent Volume | Spike | Normal |

**Squeeze Score:**
```python
squeeze_score = (
    short_percent_score * 0.35 +
    days_to_cover_score * 0.25 +
    borrow_fee_score * 0.20 +
    volume_score * 0.20
)
```

---

### 6. Macro Regime Detection (`macro_regime.py`)

**Indicators:**
| Indicator | Risk-On | Risk-Off |
|-----------|---------|----------|
| VIX | < 20 | > 25 |
| Yield Curve (10Y-2Y) | Positive | Inverted |
| SPY vs TLT (20d) | SPY winning | TLT winning |
| Dollar Index | Falling | Rising |
| Sector Leadership | Growth/Cyclical | Defensive |
| Market Breadth (RSP/SPY) | RSP leading | SPY leading |

**Regime Score:** 0-100 (higher = more risk-on)

---

## Earnings Intelligence System

### Overview

The earnings system provides comprehensive pre and post-earnings analysis with three key scores:

### IES (Implied Expectations Score)

**Purpose:** Measures how much is "priced in" before earnings

**Components:**
- Options IV percentile
- Price move into earnings
- Analyst estimate revisions
- Whisper numbers vs consensus

**Interpretation:**
| IES | Meaning | Risk |
|-----|---------|------|
| > 80 | Very high expectations | Priced for perfection |
| 60-80 | Elevated expectations | Beat required |
| 40-60 | Neutral expectations | Could go either way |
| 20-40 | Low expectations | Beat could surprise |
| < 20 | Very low expectations | Low bar to clear |

---

### EQS (Earnings Quality Score)

**Purpose:** Assesses quality of reported earnings

**Factors:**
- Revenue growth (not just EPS)
- Guidance direction
- Margins improvement
- Cash flow vs earnings
- Beat/miss consistency

---

### ECS (Expectation Clearance Score)

**Purpose:** Did results exceed what was priced in?

```python
ECS = Actual_Result - Implied_Expectations

# If IES was 80 (high expectations) and they just met:
ECS = 50 - 80 = -30  # Negative, stock may drop

# If IES was 30 (low expectations) and they beat:
ECS = 70 - 30 = +40  # Positive, stock may pop
```

---

### Reaction Analysis

Predicts post-earnings price action:

```python
Reaction Prediction = (
    EPS_Surprise_Weight * eps_surprise +
    Revenue_Surprise_Weight * revenue_surprise +
    Guidance_Weight * guidance_signal +
    IES_Adjustment * ies_factor +
    Historical_Pattern * pattern_weight
)
```

**Caching:** Results cached for 1 hour (with refresh button)

---

### Beat/Miss Detection

**Priority Order:**
1. `eps_beat` flag (if available)
2. EPS surprise percentage
3. Actual vs Estimate comparison
4. ECS fallback

```python
def detect_beat_miss(data):
    # Priority 1: Explicit flag
    if data.get('eps_beat') is not None:
        return "BEAT" if data['eps_beat'] else "MISS"
    
    # Priority 2: Surprise percentage
    if data.get('eps_surprise_pct'):
        return "BEAT" if data['eps_surprise_pct'] > 0 else "MISS"
    
    # Priority 3: Actual vs Estimate
    if data.get('eps_actual') and data.get('eps_estimate'):
        return "BEAT" if data['eps_actual'] > data['eps_estimate'] else "MISS"
    
    # Priority 4: ECS
    if data.get('ecs'):
        return "BEAT" if data['ecs'] > 50 else "MISS"
    
    return "UNKNOWN"
```

---

## Bond Trading System

### Architecture

```
bond_dashboard_complete.py (1,823 lines)
├── Cache System (lines 28-660)
│   ├── _serialize_signal()      - Convert BondSignal to dict
│   ├── _deserialize_signal()    - Reconstruct BondSignal from dict
│   ├── save_bond_cache()        - Save analysis to JSON
│   ├── load_bond_cache()        - Load analysis from JSON
│   └── get_cache_info()         - Get cache age/timestamp
│
├── Main Dashboard (lines 736-1077)
│   └── render_bond_trading_tab() - Main entry point
│
├── Display Components (lines 1078-1557)
│   ├── _render_quick_summary()      - Compact summary view
│   ├── _render_unified_summary()    - Detailed summary panel
│   ├── _render_news_expander()      - Expandable news section
│   ├── _render_news_section()       - Full news display
│   ├── _render_yields_section()     - Yield charts and data
│   ├── _render_signals_section()    - Signal table
│   └── _render_detailed_analysis()  - Deep dive analysis
│
├── AI Integration (lines 1558-1789)
│   ├── _render_ai_chat()            - Chat interface
│   └── _build_complete_ai_context() - Context builder
│
└── Education (lines 1790-1823)
    └── _render_education()          - Learning resources
```

### Cache Serialization

| Component | Saved Fields |
|-----------|--------------|
| **Signals** | ticker, price, composite_score, technical metrics, bull/bear cases |
| **News** | overall_score, overall_sentiment, article counts, themes, articles[] |
| **Articles** | title, source, sentiment, score, url, category, description |
| **Intelligence** | fed_policy, rate_probabilities, futures, auctions |

### Signal Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Technical | 30% | RSI, MACD, Trend, Support/Resistance |
| Fundamental | 25% | Yield analysis, Duration, Target prices |
| Flow | 20% | Institutional positioning, Auction demand |
| Macro | 15% | Fed policy, Economic outlook |
| Sentiment | 10% | News sentiment, Market mood |

### Signal Interpretation

| Signal | Composite Score | Action |
|--------|----------------|--------|
| 🟢 STRONG BUY | > 75 | High conviction long |
| 🟢 BUY | 60-75 | Moderate long |
| 🟡 HOLD | 40-60 | No action |
| 🔴 SELL | 25-40 | Moderate short/reduce |
| 🔴 STRONG SELL | < 25 | High conviction short |

---

## AI Integration

### Local LLM Setup

```bash
# Running on WSL2 with RTX 5090
# Model: Qwen3-32B-Q6_K
# Endpoint: http://localhost:8090/v1

# Start command:
./llama-server -m qwen3-32b-q6_k.gguf -c 32768 --port 8090 -ngl 99
```

### AI Chat System (`chat.py`)

**Context Injection:**
```python
def build_context(ticker: str) -> str:
    context = f"""
    Current Market Regime: {regime}
    
    Portfolio Context:
    {portfolio_summary}
    
    Signal for {ticker}:
    - Score: {signal.score}
    - Sentiment: {sentiment_summary}
    - Technical: {technical_summary}
    - Options Flow: {options_summary}
    
    Recent News:
    {news_headlines}
    
    Earnings:
    {earnings_summary}
    """
    return context
```

### AI Committee System (`committee.py`)

| Agent | Focus | Weight |
|-------|-------|--------|
| Fundamental Agent | Value, growth metrics | 25% |
| Sentiment Agent | News sentiment | 25% |
| Technical Agent | Price patterns | 25% |
| Valuation Agent | Fair value vs price | 25% |

---

## Caching Architecture

### Multi-Level Cache

```
┌─────────────────────────────────────────┐
│           Session State Cache            │
│  (Streamlit - per browser session)       │
│  • signals_data                          │
│  • portfolio_data                        │
│  • ui_state                              │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Engine Internal Cache          │
│  (SignalEngine._cache - 15 min TTL)      │
│  • Per-ticker signal results             │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           File-Based Cache (JSON)        │
│  (data/cache/ - persistent)              │
│  • bond_analysis_cache.json              │
│  • earnings_calendar_cache.json          │
│  • macro_regime_cache.json               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│              PostgreSQL                  │
│  (Persistent storage)                    │
│  • All analysis results                  │
│  • Historical data                       │
└─────────────────────────────────────────┘
```

### Cache Invalidation

**Signals Tab Refresh:**
```python
def _run_single_analysis(ticker):
    # 1. Run fresh analysis
    run_analysis_pipeline(ticker)
    
    # 2. Clear session state
    st.session_state.pop('signals_data', None)
    
    # 3. Clear engine cache (CRITICAL!)
    from src.core import get_signal_engine
    engine = get_signal_engine()
    if hasattr(engine, '_cache') and ticker in engine._cache:
        del engine._cache[ticker]
    
    # 4. Rerun
    st.rerun()
```

**Bond Cache Refresh:**
```powershell
# Delete cache to force fresh analysis
Remove-Item "data\cache\bond_analysis_cache.json" -Force

# Clear Python cache
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## Database Schema

### Core Tables

```sql
-- Screener scores
CREATE TABLE screener_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sentiment_score FLOAT,
    fundamental_score FLOAT,
    technical_score FLOAT,
    total_score FLOAT,
    article_count INT,
    options_flow_score FLOAT,
    short_squeeze_score FLOAT,
    options_sentiment VARCHAR(20),
    squeeze_risk VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Trading signals
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    signal_type VARCHAR(20),
    signal_strength FLOAT,
    entry_price FLOAT,
    target_price FLOAT,
    stop_loss FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- News articles
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    title TEXT,
    content TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP,
    sentiment_score FLOAT,
    sentiment_label VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Earnings analysis
CREATE TABLE earnings_analysis (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    filing_date DATE,
    eps_actual FLOAT,
    eps_estimate FLOAT,
    eps_surprise_pct FLOAT,
    eps_beat BOOLEAN,
    revenue_actual FLOAT,
    revenue_estimate FLOAT,
    guidance_direction VARCHAR(20),
    ies_score FLOAT,
    eqs_score FLOAT,
    ecs_score FLOAT,
    overall_sentiment VARCHAR(20),
    reaction_prediction VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Bond signals
CREATE TABLE bond_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    signal VARCHAR(20),
    score FLOAT,
    confidence FLOAT,
    yield_10y FLOAT,
    yield_30y FLOAT,
    spread_10y_2y FLOAT,
    current_price FLOAT,
    target_price FLOAT,
    stop_loss FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### TimescaleDB Hypertables

```sql
-- Price history as hypertable
CREATE TABLE price_history (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT
);

SELECT create_hypertable('price_history', 'time');
```

---

## Configuration

### Environment Variables (.env)

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

### Stock Universe

```python
# config/universe.py
UNIVERSE = {
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    'growth': ['CRM', 'SNOW', 'PLTR', 'DDOG', 'NET', 'CRWD'],
    'value': ['BRK.B', 'JPM', 'V', 'UNH', 'JNJ', 'PG'],
    'small_cap': ['SMCI', 'UPST', 'SOFI', 'AFRM'],
    'bonds': ['TLT', 'ZROZ', 'EDV', 'TMF', 'SHY', 'IEF'],
}
```

---

## File Structure

```
HH_research_platform_v1/
├── dashboard/
│   ├── app.py                      # Main Streamlit application
│   ├── signals_tab.py              # Signals tab with full analysis
│   ├── bond_signals_dashboard.py   # Bond trading dashboard
│   ├── analytics_tab.py            # Analytics sub-tabs
│   ├── trade_ideas_tab.py          # Trade ideas UI
│   ├── earnings_tab.py             # Earnings analysis UI
│   ├── macro_regime_tab.py         # Macro regime UI
│   ├── performance_tab.py          # Performance tracking
│   └── economic_calendar_widget.py
│
├── src/
│   ├── ai/
│   │   ├── chat.py                 # AI Chat with context
│   │   ├── llm_client.py           # LLM interface
│   │   └── committee.py            # AI Committee system
│   │
│   ├── analytics/
│   │   ├── trade_ideas.py          # Trade ideas generator
│   │   ├── macro_regime.py         # Regime detection
│   │   ├── earnings_analyzer.py    # Earnings analysis
│   │   ├── options_flow.py         # Options flow analysis
│   │   ├── short_squeeze.py        # Squeeze detection
│   │   ├── signal_performance.py   # Signal tracking
│   │   ├── risk_dashboard.py       # Risk metrics
│   │   ├── portfolio_optimizer.py  # Portfolio optimization
│   │   ├── technical_analysis.py   # TA indicators
│   │   ├── market_context.py       # Market overview
│   │   └── economic_calendar.py    # Economic events
│   │
│   ├── bond_signals/
│   │   ├── generator.py            # Bond signal generator
│   │   ├── yields.py               # Yield data fetching
│   │   └── intelligence.py         # Institutional intelligence
│   │
│   ├── core/
│   │   ├── signal_engine.py        # Unified signal engine
│   │   └── backtest.py             # Backtesting engine
│   │
│   ├── alerts/
│   │   ├── telegram_alerts.py      # Telegram notifications
│   │   └── scheduler.py            # Alert scheduler
│   │
│   ├── data/
│   │   ├── news.py                 # News fetching
│   │   ├── fundamentals.py         # Fundamental data
│   │   └── prices.py               # Price data
│   │
│   ├── db/
│   │   └── connection.py           # Database connection
│   │
│   └── utils/
│       └── logging.py              # Logging setup
│
├── scripts/
│   ├── run_full_screener.py        # Full analysis script
│   ├── run_committee.py            # Committee analysis
│   └── ingest_all.py               # Data ingestion
│
├── migrations/
│   ├── create_earnings_analysis.sql
│   ├── create_options_flow_tables.sql
│   └── add_flow_squeeze_scores.sql
│
├── data/
│   └── cache/
│       ├── bond_analysis_cache.json
│       ├── earnings_calendar_cache.json
│       └── macro_regime_cache.json
│
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## API Reference

### Signal Engine

```python
from src.core import get_signal_engine

engine = get_signal_engine()

# Generate signal for single ticker
signal = engine.generate_signal("AAPL")
print(f"Signal: {signal.signal}, Score: {signal.score}")

# Batch generate
signals = engine.generate_signals(["AAPL", "MSFT", "GOOGL"])
```

### Backtest Engine

```python
from src.core.backtest import BacktestEngine

engine = BacktestEngine()

# Run strategy backtest
result = engine.run_backtest(
    strategy="signal_based",
    start_date="2024-01-01",
    end_date="2024-12-01",
    initial_capital=100000
)

print(f"Return: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max DD: {result.max_drawdown:.2%}")
```

### Earnings Analyzer

```python
from src.analytics.earnings_analyzer import EarningsAnalyzer

analyzer = EarningsAnalyzer()

# Get earnings analysis
result = analyzer.analyze(ticker="AAPL")
print(f"IES: {result.ies_score}")
print(f"Beat/Miss: {result.beat_miss}")
print(f"Reaction: {result.reaction_prediction}")
```

### AI Chat

```python
from src.ai.chat import AIChat

chat = AIChat()

# Ask about a stock
response = chat.ask(
    message="What's the outlook for NVDA?",
    context={"ticker": "NVDA", "include_signals": True}
)
print(response)
```

---

## Running the Platform

### Prerequisites

1. PostgreSQL with TimescaleDB
2. Python 3.11+
3. Local LLM (Qwen via llama.cpp)
4. IBKR TWS (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/your/repo.git
cd HH_research_platform_v1

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup database
psql -U postgres -c "CREATE DATABASE alpha_research"
psql -U postgres -d alpha_research -f migrations/create_tables.sql

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### Starting Services

```bash
# 1. Start PostgreSQL
pg_ctl start

# 2. Start LLM (in WSL2)
./llama-server -m qwen3-32b-q6_k.gguf -c 32768 --port 8090 -ngl 99

# 3. Start IBKR TWS (optional)

# 4. Run Dashboard
cd HH_research_platform_v1
streamlit run dashboard/app.py
```

### Common Commands

```bash
# Run full screener
python scripts/run_full_screener.py

# Run committee analysis
python scripts/run_committee.py --ticker AAPL

# Test Telegram
python -c "from src.alerts.telegram_alerts import send_alert; send_alert('Test!')"

# Check macro regime
python -c "from src.analytics.macro_regime import get_current_regime; print(get_current_regime())"

# Clear all caches
Remove-Item "data\cache\*.json" -Force
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## Recent Updates (December 2025)

### v3.1 - Bond Dashboard Enhancements
- ✅ Converted bond cache from Pickle to JSON for better compatibility
- ✅ Fixed news sentiment save/load (stores `overall_sentiment` explicitly)
- ✅ Added all article attributes to cache (url, category, description)
- ✅ Fixed `published` → `published_at` attribute naming consistency
- ✅ Added debug logging for cache save/load operations
- ✅ Fixed "BEARISH (0%)" display bug when articles were actually bullish

### v3.0 - Signals Tab Enhancements
- ✅ News dates fixed to use `published_at` (not `fetched_at`)
- ✅ Source mapping for 30+ publishers
- ✅ News sorted by date descending
- ✅ "Run Analysis" upgraded to 6-step full analysis
- ✅ 100 ticker limit removed
- ✅ Smart caching with skip checkbox
- ✅ Universe-style table with 28 columns
- ✅ Dropdown sorting (more reliable than column headers)
- ✅ Stock selector dropdown for quick navigation
- ✅ Trade recommendations with entry/stop/target
- ✅ AI summary explanations for component scores
- ✅ Fixed SignalEngine cache not clearing on refresh

### v2.9 - Earnings Intelligence
- ✅ Fixed BEAT/MISS detection (priority order implemented)
- ✅ Earnings column shows past (✓) and future dates
- ✅ Reaction analysis caching (1 hour, with refresh button)
- ✅ Earnings calendar auto-population during analysis

### v2.8 - Performance & Backtesting
- ✅ Unified Performance & Backtesting tab
- ✅ Signal Performance tracking with win rates
- ✅ 9 predefined backtesting strategies
- ✅ Signal Combination analysis
- ✅ Trade Journal with signal attribution
- ✅ Sharpe/Sortino/Alpha calculations
- ✅ Equity curve visualization

---

## Troubleshooting

### LLM Not Responding

```bash
# Check if running
curl http://localhost:8090/v1/models

# Restart
./llama-server -m model.gguf -c 32768 --port 8090 -ngl 99
```

### Database Connection Error

```bash
# Check PostgreSQL
pg_isready -h localhost -p 5432

# Check connection string
echo $DATABASE_URL
```

### Import Errors

```bash
# Ensure you're in the project root
cd HH_research_platform_v1

# Run from correct location
streamlit run dashboard/app.py
```

### IBKR Not Connecting

- Ensure TWS is running
- Enable API in TWS settings (port 7497)
- Check `IBKR_HOST` and `IBKR_PORT` in .env

### Telegram Not Sending

```python
from src.alerts.telegram_alerts import TelegramAlerter
alerter = TelegramAlerter()
print(f"Enabled: {alerter.enabled}")
alerter.send_message("Test")
```

### Bond Cache Issues

```powershell
# Clear cache to force fresh analysis
Remove-Item "data\cache\bond_analysis_cache.json" -Force

# Clear Python cache
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

### Signals Not Updating After Refresh

```python
# The fix clears both session state AND engine cache
# If still having issues, restart Streamlit:
# Ctrl+C, then: streamlit run dashboard/app.py
```

### News Dates Not Showing

- Ensure `python-dateutil` is installed: `pip install python-dateutil`
- Check that news articles have `published_at` field populated
- Run migration script if upgrading: `python fix_news_dates.py`

---

## Performance Notes

- **LLM**: Qwen3-32B on RTX 5090 (32GB VRAM) - ~50 tokens/sec
- **Database**: PostgreSQL with TimescaleDB, indexed on ticker/date
- **Caching**: News (6h), earnings (24h), regime (1h), bond (persistent)
- **Analysis Time**: ~15-30 seconds per ticker (full analysis)
- **Batch Processing**: 20 tickers parallel with rate limiting

---

## License

Private - Internal Use Only

---

## Contact

For questions about this platform, refer to this README or the code comments.