# HH Research Platform

## Complete Hedge Fund-Grade Trading Intelligence System

A comprehensive quantitative trading platform that combines AI-powered sentiment analysis, fundamental analysis, technical indicators, options flow, macro regime detection, earnings intelligence, and bond trading into a unified signal generation system with full backtesting and performance tracking.

**Author:** Hasan  
**Version:** 3.0  
**Last Updated:** December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Dashboard Tabs](#dashboard-tabs)
4. [Signal Generation Pipeline](#signal-generation-pipeline)
5. [Core Analytics Modules](#core-analytics-modules)
6. [Earnings Intelligence System](#earnings-intelligence-system)
7. [AI Integration](#ai-integration)
8. [Database Schema](#database-schema)
9. [Configuration](#configuration)
10. [File Structure](#file-structure)
11. [API Reference](#api-reference)
12. [Running the Platform](#running-the-platform)
13. [Recent Updates](#recent-updates)

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
9. **Sends Alerts** via Telegram
10. **Integrates with IBKR** for live portfolio data
11. **Displays Everything** in a Streamlit dashboard

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
| **Bond Trading** | Yield-based signals for TLT, ZROZ, etc. |
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
│                          SIGNAL ENGINE                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      UnifiedSignal                                    │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │   │
│  │  │ Sentiment │ │Fundamental│ │ Technical │ │ Options   │            │   │
│  │  │ Analyzer  │ │ Analyzer  │ │ Analyzer  │ │   Flow    │            │   │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘            │   │
│  │        │             │             │             │                   │   │
│  │  ┌───────────┐ ┌───────────┐                                        │   │
│  │  │  Short    │ │  Macro    │                                        │   │
│  │  │ Squeeze   │ │  Regime   │                                        │   │
│  │  └─────┬─────┘ └─────┬─────┘                                        │   │
│  │        │             │                                               │   │
│  │        └─────────────┴──────────────────────────────────────────────│   │
│  │                              ▼                                       │   │
│  │                      Total Score (0-100)                            │   │
│  │                      Signal: BUY/HOLD/SELL                          │   │
│  │                      Confidence: High/Medium/Low                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYTICS LAYER                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │  Earnings  │ │   Risk     │ │  Portfolio │ │   Signal   │               │
│  │Intelligence│ │ Dashboard  │ │ Optimizer  │ │Performance │               │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘               │
│        │              │              │              │                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │  Backtest  │ │   Trade    │ │   Bond     │ │  Economic  │               │
│  │  Engine    │ │   Ideas    │ │  Signals   │ │  Calendar  │               │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘               │
└────────┼──────────────┼──────────────┼──────────────┼───────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   PostgreSQL + TimescaleDB                            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │screener_    │ │ trading_    │ │ news_       │ │fundamentals │    │   │
│  │  │  scores     │ │  signals    │ │  articles   │ │             │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │ committee_  │ │ earnings_   │ │ backtest_   │ │ trade_      │    │   │
│  │  │  decisions  │ │  calendar   │ │  results    │ │  journal    │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │ portfolio_  │ │ historical_ │ │ earnings_   │ │ reaction_   │    │   │
│  │  │  snapshots  │ │   scores    │ │  analysis   │ │  analysis   │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Yahoo   │ │ NewsAPI │ │ Finnhub │ │  IBKR   │ │ Qwen    │ │Telegram │  │
│  │ Finance │ │         │ │         │ │  TWS    │ │  LLM    │ │  Bot    │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dashboard Tabs

### Tab 1: 📊 Universe
**Purpose:** Overview of all stocks with scoring data

| Column | Description |
|--------|-------------|
| Ticker | Stock symbol |
| Sector | GICS sector |
| Signal | STRONG_BUY to STRONG_SELL |
| Total | Combined score 0-100 |
| Sentiment | AI sentiment score |
| OptFlow | Options flow score (50=neutral) |
| Squeeze | Short squeeze potential |
| Fundamental | Value/growth score |
| Technical | TA-based score |
| Earnings | Next/recent earnings date (✓=reported) |
| Upside% | Analyst target upside |

**Features:**
- Sector/signal filtering
- Column sorting
- Export to CSV

---

### Tab 2: 📈 Signals
**Purpose:** Main analysis interface with full deep dive capability

**Features:**
- **Universe Table:** 28 columns with all scores
- **Dropdown Sorting:** Reliable sort by any column
- **Stock Selector:** Quick jump to any stock
- **Skip Cache:** Force fresh analysis
- **Run Analysis:** 6-step full analysis pipeline

**Deep Dive Sections:**
1. **Signal Summary** - Total score, signal type, AI explanation
2. **Component Scores** - Radar chart + score bars
3. **Trade Recommendation** - Entry, stop loss, target, risk/reward
4. **News Analysis** - Recent articles with sentiment
5. **Options Flow** - Call/put ratio, unusual activity
6. **Short Squeeze** - Short interest, days to cover
7. **Earnings Intelligence** - Beat/miss, reaction analysis
8. **Committee Decision** - AI agent votes

**Analysis Pipeline (6 Steps):**
```
Step 1: Sentiment Analysis (NewsAPI, Finnhub, Google)
Step 2: Fundamental Analysis (Yahoo Finance)
Step 3: Technical Analysis (RSI, MACD, MAs)
Step 4: Options Flow (Call/Put volumes, unusual activity)
Step 5: Short Squeeze (Short %, borrow fee)
Step 6: Save to Database + Update Earnings Calendar
```

---

### Tab 3: 🔍 Deep Dive
**Purpose:** Standalone deep dive for any ticker

Same features as Signals tab deep dive but accessible independently.

---

### Tab 4: 💼 Portfolio
**Purpose:** IBKR portfolio integration

**Features:**
- Live positions from IBKR
- P&L tracking (daily, total)
- Sector allocation chart
- Position analysis
- AI insights on portfolio

**Requires:** IBKR TWS running with API enabled

---

### Tab 5: 📊 Analytics
**Purpose:** Advanced analytics sub-tabs

**Sub-tabs:**
1. **Signal Performance** - Win rates by signal type
2. **Risk Dashboard** - VaR, correlation, drawdown
3. **Portfolio Optimizer** - Mean-variance optimization
4. **Options Flow** - Market-wide options analysis
5. **Short Squeeze** - Squeeze candidates screener
6. **Earnings Analysis** - Upcoming/recent earnings
7. **Macro Regime** - Risk-On/Risk-Off indicators
8. **Economic Calendar** - Fed, CPI, jobs data

---

### Tab 6: 🏦 Bonds
**Purpose:** Bond ETF trading signals

**Supported Instruments:**
- TLT (20+ Year Treasury)
- ZROZ (25+ Year Zero Coupon)
- EDV (Extended Duration)
- TMF (3x Long Treasury)
- SHY (1-3 Year Treasury)
- IEF (7-10 Year Treasury)

**Signal Factors:**
| Factor | Weight | Bullish When |
|--------|--------|--------------|
| Yield Trend | 25% | Yields falling |
| Curve Shape | 15% | Inverted |
| Fed Policy | 20% | Dovish |
| Inflation | 15% | Low/falling |
| Macro Regime | 15% | Risk-Off |
| Technicals | 10% | Oversold |

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

### Tab 8: 📊 Performance (NEW)
**Purpose:** Unified performance tracking and backtesting

**Sub-tabs:**

#### 📈 Signal Performance
- Total signals analyzed
- Win rate by holding period (5d/10d/30d)
- Average returns by signal type
- Best/worst performers
- Recent signals with outcomes

#### 🔬 Strategy Backtester
**9 Predefined Strategies:**
| Strategy | Description |
|----------|-------------|
| Aggressive Buy | Buy on any buy signal |
| Conservative Buy | Only STRONG_BUY signals |
| Buy Signals | BUY + STRONG_BUY |
| Long/Short | Long buys, short sells |
| High Sentiment | Sentiment >= 65 |
| High Fundamental | Fundamental >= 65 |
| High Total Score | Total >= 55 |
| Quality + Momentum | 2+ conditions met |
| Triple Screen | ALL conditions met |

**Metrics Calculated:**
- Total trades, win rate
- Average/median return
- Sharpe ratio, Sortino ratio
- Max drawdown
- Alpha vs benchmark (SPY/QQQ)
- Equity curve

#### 🎯 Signal Combinations
Analyze which signal combinations work best:
- Single signals alone
- Double combinations (Sentiment + Fundamental, etc.)
- Triple screen (all 3 conditions)
- High score rates by combination

#### 📋 Trade Journal
- Log actual trades with ticker, entry, exit
- Link to entry signals (why you bought)
- Track P&L and win rate
- Attribution analysis

---

### Tab 9: ⚙️ System Status
**Purpose:** System health monitoring

**Features:**
- Database table counts
- Last run timestamps
- Cache status
- API health checks

---

### Tab 10: 🤖 AI Chat
**Purpose:** Conversational AI assistant with full context

**Features:**
- Full access to all platform data
- Portfolio-aware responses
- Signal explanations
- Trade idea generation
- File upload support (CSV, PDF, Excel)

**Context Includes:**
- Current positions
- Recent signals
- Earnings calendar
- Market regime
- News summaries

---

## Signal Generation Pipeline

### Unified Signal Engine

The `UnifiedSignal` class combines 6 analyzers:

```python
class UnifiedSignal:
    """
    Combines all analyzers into single recommendation.
    
    Components:
    1. SentimentAnalyzer - News-based sentiment (0-100)
    2. FundamentalAnalyzer - Value/growth metrics (0-100)
    3. TechnicalAnalyzer - Price/volume indicators (0-100)
    4. OptionsFlowAnalyzer - Institutional positioning (0-100)
    5. ShortSqueezeAnalyzer - Squeeze potential (0-100)
    6. MacroRegimeAnalyzer - Market environment (Risk-On/Off)
    """
```

### Scoring Weights

```python
# Default weights (configurable)
WEIGHTS = {
    'sentiment': 0.25,      # 25%
    'fundamental': 0.20,    # 20%
    'technical': 0.20,      # 20%
    'options_flow': 0.15,   # 15%
    'short_squeeze': 0.10,  # 10%
    'macro_regime': 0.10,   # 10%
}

# Score calculation
total_score = sum(score * weight for score, weight in components)

# Adjustments
if earnings_beat: total_score += 5-15
if earnings_miss: total_score -= 5-15
if risk_on and growth_stock: total_score += 5-10
if risk_off and defensive_stock: total_score += 5-10
```

### Signal Thresholds

```python
def get_signal(total_score: float) -> str:
    if total_score >= 80: return "STRONG_BUY"
    if total_score >= 65: return "BUY"
    if total_score >= 55: return "WEAK_BUY"
    if total_score >= 45: return "HOLD"
    if total_score >= 35: return "WEAK_SELL"
    if total_score >= 20: return "SELL"
    return "STRONG_SELL"
```

### Signal Confidence

```python
def get_confidence(scores: dict) -> str:
    """
    Confidence based on agreement between components.
    """
    # All components agree (within 15 points)
    if std_deviation(scores) < 10: return "HIGH"
    
    # Most components agree
    if std_deviation(scores) < 20: return "MEDIUM"
    
    # Components disagree
    return "LOW"
```

---

## Core Analytics Modules

### 1. Sentiment Analysis (`sentiment.py`)

**Sources:**
- NewsAPI (general news)
- Finnhub (financial news)
- Google News (broad coverage)
- Reddit (social sentiment)

**Process:**
```python
1. Fetch articles for ticker (last 7 days)
2. Filter duplicates and irrelevant
3. Send to Qwen LLM for analysis
4. Extract sentiment score (0-100)
5. Weight by source quality and recency
6. Cache results (6 hours)
```

**LLM Prompt:**
```
Analyze the following news articles about {ticker}.
Rate the overall sentiment from 0-100 where:
- 0-20: Very Bearish
- 20-40: Bearish  
- 40-60: Neutral
- 60-80: Bullish
- 80-100: Very Bullish

Provide a brief explanation of key factors.
```

---

### 2. Fundamental Analysis (`fundamentals.py`)

**Metrics:**

| Metric | Bullish Range | Weight |
|--------|---------------|--------|
| P/E Ratio | 10-25 | 15% |
| PEG Ratio | 0-1.5 | 15% |
| Revenue Growth | > 10% | 20% |
| Profit Margin | > 15% | 15% |
| ROE | > 15% | 15% |
| Debt/Equity | < 0.5 | 10% |
| Current Ratio | > 1.5 | 10% |

**Scoring:**
```python
fundamental_score = (
    value_score * 0.4 +    # P/E, P/B, EV/EBITDA
    growth_score * 0.3 +   # Revenue, earnings growth
    quality_score * 0.3    # Margins, ROE, debt
)
```

---

### 3. Technical Analysis (`technical_analysis.py`)

**Indicators:**

| Indicator | Bullish Signal | Weight |
|-----------|----------------|--------|
| RSI | 30-70 (not extreme) | 15% |
| MACD | Above signal line | 20% |
| SMA 20 | Price above | 15% |
| SMA 50 | Price above | 15% |
| SMA 200 | Price above | 15% |
| Volume | Above average | 10% |
| Bollinger | Near lower band | 10% |

**Scoring:**
```python
technical_score = (
    trend_score * 0.4 +      # MA alignment
    momentum_score * 0.3 +   # RSI, MACD
    volume_score * 0.3       # Volume confirmation
)
```

---

### 4. Options Flow Analysis (`options_flow.py`)

**Metrics:**

| Metric | Description |
|--------|-------------|
| Put/Call Ratio | < 0.7 = bullish, > 1.0 = bearish |
| Call Volume | Unusual = 2x+ average |
| Put Volume | Unusual = 2x+ average |
| IV Percentile | Options pricing |
| OI Change | Open interest trend |

**Scoring:**
```python
options_score = (
    50 +  # Neutral baseline
    call_volume_bonus (0-20) +
    put_volume_penalty (0-20) +
    unusual_activity_bonus (0-15) +
    iv_adjustment (-10 to +10)
)

# Sentiment mapping
if options_score >= 70: sentiment = "VERY_BULLISH"
elif options_score >= 55: sentiment = "BULLISH"
elif options_score >= 45: sentiment = "NEUTRAL"
elif options_score >= 30: sentiment = "BEARISH"
else: sentiment = "VERY_BEARISH"
```

---

### 5. Short Squeeze Analysis (`short_squeeze.py`)

**Metrics:**

| Metric | High Squeeze Risk |
|--------|-------------------|
| Short % of Float | > 20% |
| Days to Cover | > 5 days |
| Borrow Fee | > 50% annual |
| Short % Change | Increasing |
| Price Momentum | Up on high volume |

**Scoring:**
```python
squeeze_score = (
    short_interest_score * 0.30 +   # % of float
    days_to_cover_score * 0.25 +    # Days to cover
    borrow_fee_score * 0.20 +       # Cost to short
    momentum_score * 0.15 +         # Recent price action
    volume_score * 0.10             # Volume surge
)

# Risk levels
if squeeze_score >= 75: risk = "EXTREME"
elif squeeze_score >= 60: risk = "HIGH"
elif squeeze_score >= 40: risk = "MODERATE"
else: risk = "LOW"
```

---

### 6. Macro Regime Detection (`macro_regime.py`)

**Indicators:**

| Indicator | Risk-On Signal | Risk-Off Signal |
|-----------|----------------|-----------------|
| VIX | < 18 | > 25 |
| 10Y-2Y Spread | > 0 (positive) | < 0 (inverted) |
| SPY vs TLT | SPY outperforming | TLT outperforming |
| Dollar Index | Falling | Rising |
| Sector Leadership | Growth leading | Defensive leading |
| Market Breadth | RSP > SPY | SPY > RSP |
| Credit Spreads | Tightening | Widening |

**Regime Score:**
```python
regime_score = sum(indicator_scores) / num_indicators  # 0-100

if regime_score >= 65: regime = "STRONG_RISK_ON"
elif regime_score >= 55: regime = "RISK_ON"
elif regime_score >= 45: regime = "NEUTRAL"
elif regime_score >= 35: regime = "RISK_OFF"
else: regime = "STRONG_RISK_OFF"
```

**Signal Adjustments:**
```python
if regime == "RISK_ON":
    growth_stocks += 10
    tech_stocks += 10
    defensive_stocks -= 5

if regime == "RISK_OFF":
    defensive_stocks += 10
    utility_stocks += 10
    growth_stocks -= 15
    tech_stocks -= 10
```

---

## Earnings Intelligence System

### Overview

The Earnings Intelligence System predicts stock reactions to earnings announcements using three calculated scores:

### IES (Implied Expectations Score)

**Purpose:** Measure how much optimism is priced into the stock

```python
IES = (
    pre_earnings_drift * 0.25 +      # Stock move before earnings
    options_iv_percentile * 0.25 +   # Options pricing
    analyst_revision_trend * 0.20 +  # Recent estimate changes
    short_interest_change * 0.15 +   # Short covering/building
    sector_momentum * 0.15           # Sector performance
)

# Range: 0-100
# High IES = High expectations priced in = Need big beat
# Low IES = Low expectations = Easier to surprise
```

### EQS (Earnings Quality Score)

**Purpose:** Assess the quality/sustainability of earnings

```python
EQS = (
    revenue_beat * 0.30 +           # Revenue vs estimates
    margin_expansion * 0.20 +       # Profit margin trend
    guidance_quality * 0.25 +       # Forward guidance
    cash_flow_quality * 0.15 +      # FCF vs earnings
    accounting_quality * 0.10       # Accruals, one-time items
)

# High EQS = Quality beat, likely sustained
# Low EQS = Low quality, may fade
```

### ECS (Expectation Clearance Score)

**Purpose:** Did the stock "clear" priced-in expectations?

```python
# Calculate required Z-score based on IES
required_z = calculate_required_z(ies_score)

# Calculate actual Z-score from surprise
event_z = calculate_event_z(eps_surprise_pct, rev_surprise_pct)

# Did it clear?
if event_z >= required_z:
    ecs_category = "BEAT"  # Cleared expectations
else:
    ecs_category = "MISS"  # Failed to clear

# Note: ECS != actual beat/miss
# A stock can BEAT estimates but MISS ECS if expectations were too high
```

### Reaction Analysis

The `ReactionAnalyzer` tracks post-earnings price action:

```python
@dataclass
class EarningsReaction:
    ticker: str
    earnings_date: date
    earnings_result: str         # "BEAT" or "MISS" (actual)
    eps_surprise_pct: float      # +20.71% for MU
    
    # Price reactions
    gap_pct: float              # Overnight gap
    day1_return: float          # Close vs pre-earnings close
    day2_return: float          # 2-day cumulative
    day5_return: float          # 5-day cumulative
    
    # ECS data (priced-in expectations)
    ecs_category: str           # "BEAT" or "MISS" (expectations)
    ecs_required_z: float       # What was needed
    ecs_event_z: float          # What was delivered
```

### Beat/Miss Detection

**Priority Order (most reliable first):**
```python
@property
def earnings_result(self) -> str:
    # Method 1: Use explicit eps_beat flag (most reliable)
    if self.eps_beat is not None:
        return "BEAT" if self.eps_beat else "MISS"
    
    # Method 2: Calculate from surprise percentage
    if self.eps_surprise_pct is not None:
        if self.eps_surprise_pct > 1: return "BEAT"
        elif self.eps_surprise_pct < -1: return "MISS"
        else: return "INLINE"
    
    # Method 3: Calculate from actual vs estimate
    if self.eps_actual and self.eps_estimate:
        if self.eps_actual > self.eps_estimate * 1.01: return "BEAT"
        elif self.eps_actual < self.eps_estimate * 0.99: return "MISS"
        else: return "INLINE"
    
    # Method 4: Use ECS only for strong signals
    if self.ecs_category and 'STRONG' in self.ecs_category:
        return "BEAT" if 'BEAT' in self.ecs_category else "MISS"
    
    return "N/A"
```

---

## AI Integration

### Local LLM Setup

The platform uses **Qwen3-32B** running locally via **llama.cpp**:

```bash
# Start llama.cpp server
./llama-server \
    -m qwen3-32b-q4_k_m.gguf \
    -c 32768 \
    --port 8090 \
    -ngl 99 \
    --flash-attn
```

### AI Chat System (`chat.py`)

**Features:**
- Full context awareness (positions, signals, news)
- Streaming responses
- File upload support
- Unified portfolio context

**Context Building:**
```python
def build_context(self, ticker: str = None) -> str:
    context = []
    
    # Add market overview
    context.append(self._get_market_context())
    
    # Add macro regime
    context.append(self._get_regime_context())
    
    # Add portfolio data (if available)
    if self.positions:
        context.append(self._get_portfolio_context())
    
    # Add ticker-specific data
    if ticker:
        context.append(self._get_ticker_signals(ticker))
        context.append(self._get_ticker_news(ticker))
        context.append(self._get_ticker_earnings(ticker))
    
    return "\n\n".join(context)
```

### AI Committee System (`committee.py`)

**Agents:**

| Agent | Focus | Weight |
|-------|-------|--------|
| Fundamental Agent | Value, growth, quality | 25% |
| Sentiment Agent | News, social sentiment | 25% |
| Technical Agent | Price patterns, indicators | 25% |
| Valuation Agent | Fair value vs price | 25% |

**Process:**
```python
1. Each agent analyzes the stock independently
2. Each provides: Vote (BUY/HOLD/SELL), Confidence (0-100), Reasoning
3. Votes are weighted by confidence
4. Final verdict is consensus with explanation
```

---

## Database Schema

### Core Tables

```sql
-- Main scoring table (updated by screener)
CREATE TABLE screener_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- Component scores
    sentiment_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    technical_score DECIMAL(5,2),
    options_flow_score DECIMAL(5,2),
    short_squeeze_score DECIMAL(5,2),
    total_score DECIMAL(5,2),
    
    -- Metadata
    article_count INTEGER,
    options_sentiment VARCHAR(20),
    squeeze_risk VARCHAR(20),
    
    -- Signal
    signal_type VARCHAR(20),
    signal_strength DECIMAL(5,2),
    
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Trading signals history
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    signal_type VARCHAR(20),
    signal_strength DECIMAL(5,2),
    sentiment_score DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    
    -- Returns (filled later)
    return_5d DECIMAL(8,4),
    return_10d DECIMAL(8,4),
    return_30d DECIMAL(8,4),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- News articles
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    title TEXT,
    summary TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP,
    sentiment_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Earnings calendar
CREATE TABLE earnings_calendar (
    ticker VARCHAR(10) PRIMARY KEY,
    earnings_date DATE,
    eps_estimate DECIMAL(10,4),
    eps_actual DECIMAL(10,4),
    surprise_pct DECIMAL(8,4),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Committee decisions
CREATE TABLE committee_decisions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    decision_date DATE NOT NULL,
    final_verdict VARCHAR(20),
    confidence DECIMAL(5,2),
    agent_votes JSONB,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Historical scores for backtesting
CREATE TABLE historical_scores (
    id SERIAL PRIMARY KEY,
    score_date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    sector VARCHAR(50),
    sentiment DECIMAL(5,2),
    fundamental_score DECIMAL(5,2),
    technical_score DECIMAL(5,2),
    total_score DECIMAL(5,2),
    signal_type VARCHAR(20),
    op_price DECIMAL(12,4),
    return_1d DECIMAL(8,4),
    return_5d DECIMAL(8,4),
    return_10d DECIMAL(8,4),
    return_20d DECIMAL(8,4),
    UNIQUE(score_date, ticker)
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
    win_rate DECIMAL(5,4),
    avg_return DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    alpha DECIMAL(8,4),
    returns_by_signal TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trade journal
CREATE TABLE trade_journal (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    direction VARCHAR(10) DEFAULT 'LONG',
    shares INTEGER DEFAULT 1,
    exit_date DATE,
    exit_price DECIMAL(12,4),
    entry_signals TEXT[],
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    account_id VARCHAR(50),
    net_liquidation DECIMAL(14,2),
    total_cash DECIMAL(14,2),
    gross_position_value DECIMAL(14,2),
    unrealized_pnl DECIMAL(12,2),
    daily_pnl DECIMAL(12,2),
    daily_return_pct DECIMAL(8,4),
    cumulative_return_pct DECIMAL(8,4),
    benchmark_value DECIMAL(12,4),
    benchmark_return_pct DECIMAL(8,4),
    alpha_vs_benchmark DECIMAL(8,4),
    position_count INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Fundamentals cache
CREATE TABLE fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    pe_ratio DECIMAL(10,4),
    peg_ratio DECIMAL(10,4),
    revenue_growth DECIMAL(8,4),
    profit_margin DECIMAL(8,4),
    roe DECIMAL(8,4),
    debt_to_equity DECIMAL(10,4),
    current_ratio DECIMAL(10,4),
    price DECIMAL(12,4),
    target_mean DECIMAL(12,4),
    target_upside_pct DECIMAL(8,4),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### TimescaleDB Hypertables

```sql
-- Convert to hypertable for time-series queries
SELECT create_hypertable('screener_scores', 'date', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

SELECT create_hypertable('historical_scores', 'score_date',
    chunk_time_interval => INTERVAL '1 month', 
    if_not_exists => TRUE
);
```

### Indexes

```sql
-- Performance indexes
CREATE INDEX idx_scores_ticker_date ON screener_scores(ticker, date DESC);
CREATE INDEX idx_signals_ticker_date ON trading_signals(ticker, date DESC);
CREATE INDEX idx_news_ticker_published ON news_articles(ticker, published_at DESC);
CREATE INDEX idx_historical_date_ticker ON historical_scores(score_date, ticker);
CREATE INDEX idx_earnings_date ON earnings_calendar(earnings_date);
```

---

## Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/alpha_platform

# LLM
LLM_HOST=http://localhost:8090
LLM_MODEL=qwen3-32b

# News APIs
NEWSAPI_KEY=your_newsapi_key
FINNHUB_API_KEY=your_finnhub_key

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# IBKR
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# OpenAI (optional, for backup)
OPENAI_API_KEY=your_openai_key

# Platform Settings
CACHE_TTL_HOURS=6
MAX_ARTICLES_PER_TICKER=50
ANALYSIS_BATCH_SIZE=20
```

### Stock Universe

The platform monitors ~180 stocks defined in `src/data/tickers.py`:

```python
UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', ...],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', ...],
    'Financials': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA', ...],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', ...],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', ...],
    'Industrials': ['CAT', 'BA', 'UNP', 'HON', ...],
    'Materials': ['LIN', 'APD', 'FCX', 'NEM', ...],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', ...],
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', ...],
    'Communication': ['META', 'NFLX', 'DIS', 'CMCSA', ...],
}
```

---

## File Structure

```
HH_research_platform_v1/
├── dashboard/
│   ├── app.py                          # Main Streamlit application
│   ├── portfolio_tab.py                # IBKR portfolio integration
│   └── components/
│       └── economic_calendar_widget.py
│
├── src/
│   ├── tabs/
│   │   ├── signals_tab.py              # Main signals/analysis tab
│   │   ├── deep_dive_tab.py            # Deep dive analysis
│   │   └── performance_backtest_tab.py # Performance & backtesting (NEW)
│   │
│   ├── ai/
│   │   ├── chat.py                     # AI chat with context
│   │   ├── chat_integration.py         # Chat UI integration
│   │   ├── llm_client.py               # LLM interface
│   │   ├── committee.py                # AI committee system
│   │   ├── agents.py                   # Individual AI agents
│   │   ├── learning.py                 # Signal learning
│   │   └── signal_performance.py       # Signal tracking
│   │
│   ├── analytics/
│   │   ├── signal_engine.py            # UnifiedSignal engine
│   │   ├── unified_signal.py           # Signal combination logic
│   │   ├── trade_ideas.py              # Trade ideas generator
│   │   ├── macro_regime.py             # Regime detection
│   │   ├── bond_signals.py             # Bond trading signals
│   │   ├── options_flow.py             # Options flow analysis
│   │   ├── short_squeeze.py            # Squeeze detection
│   │   ├── technical_analysis.py       # TA indicators
│   │   ├── risk_dashboard.py           # Risk metrics
│   │   ├── portfolio_optimizer.py      # Mean-variance optimization
│   │   ├── market_context.py           # Market overview
│   │   └── economic_calendar.py        # Economic events
│   │   │
│   │   ├── earnings_intelligence/
│   │   │   ├── ies_calculator.py       # Implied Expectations Score
│   │   │   ├── eqs_calculator.py       # Earnings Quality Score
│   │   │   ├── ecs_calculator.py       # Expectation Clearance Score
│   │   │   ├── reaction_analyzer.py    # Post-earnings reaction
│   │   │   ├── earnings_analyzer.py    # Main earnings analysis
│   │   │   ├── models.py               # Data models
│   │   │   └── backtesting.py          # Earnings backtesting
│   │   │
│   │   └── backtesting/
│   │       ├── engine.py               # Backtest engine
│   │       └── strategies.py           # Strategy definitions
│   │
│   ├── data/
│   │   ├── news.py                     # News fetching
│   │   ├── sentiment.py                # Sentiment analysis
│   │   ├── fundamentals.py             # Fundamental data
│   │   ├── technicals.py               # Technical indicators
│   │   ├── analyst_inputs.py           # Analyst data
│   │   ├── options_inputs.py           # Options data
│   │   └── yahoo_prices.py             # Price data
│   │
│   ├── db/
│   │   ├── connection.py               # Database connection
│   │   └── repository.py               # Data access layer
│   │
│   ├── alerts/
│   │   ├── telegram_alerts.py          # Telegram notifications
│   │   └── scheduler.py                # Alert scheduler
│   │
│   ├── ibkr/
│   │   └── ibkr_utils.py               # IBKR TWS integration
│   │
│   └── utils/
│       └── logging.py                  # Logging setup
│
├── scripts/
│   ├── run_full_screener.py            # Full analysis script
│   ├── run_committee.py                # Committee analysis
│   ├── populate_earnings_calendar.py   # Update earnings dates
│   ├── populate_dates.py               # Update all dates
│   └── ingest_all.py                   # Data ingestion
│
├── migrations/
│   ├── 001_schema.sql                  # Initial schema
│   ├── create_earnings_analysis.sql
│   ├── create_earnings_calendar.sql
│   ├── create_earnings_intelligence.sql
│   ├── create_signal_hub.sql
│   └── add_flow_squeeze_scores.sql
│
├── data/
│   ├── earnings_calendar_cache.json
│   └── macro_regime_cache.json
│
├── .env                                # Environment variables
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## API Reference

### Signal Engine

```python
from src.analytics.signal_engine import SignalEngine

engine = SignalEngine()

# Analyze single ticker
result = engine.analyze(ticker="NVDA")
print(f"Signal: {result.signal_type}, Score: {result.total_score}")

# Analyze multiple tickers
results = engine.analyze_batch(tickers=["AAPL", "MSFT", "NVDA"])
```

### Backtest Engine

```python
from src.analytics.backtesting.engine import BacktestEngine

engine = BacktestEngine()

# Run backtest
result = engine.run_backtest(
    strategy='high_sentiment',
    start_date='2024-01-01',
    end_date='2024-12-01',
    holding_period=10,
    benchmark='SPY',
    params={'buy_threshold': 65}
)

print(f"Win Rate: {result.win_rate:.1%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Alpha: {result.alpha:.2f}%")
```

### Signal Performance Tracker

```python
from src.ai.signal_performance import SignalPerformanceTracker

tracker = SignalPerformanceTracker()

# Get performance summary
summary = tracker.get_performance_summary(days_back=90)
print(f"Win Rate: {summary['overall_win_rate']:.1f}%")

# Get performance by signal type
by_signal = tracker.get_performance_by_signal_type(days_back=90)
for signal_type, perf in by_signal.items():
    print(f"{signal_type}: {perf.win_rate_10d:.1f}% win rate")
```

### Earnings Analyzer

```python
from src.analytics.earnings_intelligence.reaction_analyzer import analyze_post_earnings

# Analyze recent earnings
reaction = analyze_post_earnings("MU")
print(f"Result: {reaction.earnings_result}")
print(f"Surprise: {reaction.eps_surprise_pct:+.1f}%")
print(f"5-Day Return: {reaction.day5_return:+.1f}%")
```

### AI Chat

```python
from src.ai.chat import AIChat

chat = AIChat()

# Simple query
response = chat.query("Analyze NVDA and give me a trade recommendation")

# With ticker context
response = chat.query("What's the sentiment?", ticker="AAPL")

# Streaming
for chunk in chat.chat_stream("Explain the macro regime"):
    print(chunk, end="")
```

---

## Running the Platform

### Prerequisites

1. **PostgreSQL** with TimescaleDB extension
2. **Python 3.11+** with virtualenv
3. **llama.cpp** server with Qwen model
4. **IBKR TWS** (optional, for portfolio)

### Installation

```bash
# Clone repository
git clone <repository_url>
cd HH_research_platform_v1

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Run migrations
psql -d alpha_platform -f migrations/001_schema.sql
```

### Starting Services

```bash
# 1. Start LLM server
./llama-server -m qwen3-32b-q4_k_m.gguf -c 32768 --port 8090 -ngl 99

# 2. Start IBKR TWS (optional)
# Open TWS, enable API on port 7497

# 3. Run dashboard
cd HH_research_platform_v1
streamlit run dashboard/app.py
```

### Common Commands

```bash
# Run full screener
python scripts/run_full_screener.py

# Run with fresh news (bypass cache)
python scripts/run_full_screener.py --fresh-news

# Run committee analysis for ticker
python scripts/run_committee.py --ticker AAPL

# Update earnings calendar
python scripts/populate_earnings_calendar.py

# Test Telegram alerts
python -c "from src.alerts.telegram_alerts import send_alert; send_alert('Test!')"

# Check current regime
python -c "from src.analytics.macro_regime import get_current_regime; print(get_current_regime())"

# Check bond signal
python -c "from src.analytics.bond_signals import get_bond_signal; print(get_bond_signal('TLT'))"
```

---

## Recent Updates (December 2025)

### Signals Tab Enhancements
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

### Earnings Intelligence
- ✅ Fixed BEAT/MISS detection (MU showing MISS when it beat by 20.71%)
- ✅ Priority order: eps_beat flag → surprise% → actual vs estimate → ECS
- ✅ Earnings column shows past (✓) and future dates
- ✅ Reaction analysis caching (1 hour, with refresh button)
- ✅ Earnings calendar auto-population during analysis

### Performance & Backtesting (NEW)
- ✅ Unified Performance & Backtesting tab
- ✅ Signal Performance tracking with win rates
- ✅ 9 predefined backtesting strategies
- ✅ Signal Combination analysis
- ✅ Trade Journal with signal attribution
- ✅ Sharpe/Sortino/Alpha calculations
- ✅ Equity curve visualization

### Bug Fixes
- ✅ Import path issues resolved (`sys.path` before imports)
- ✅ CRLF line ending issues
- ✅ Sorting reliability improvements
- ✅ ECS vs actual beat/miss confusion resolved

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

---

## Performance Notes

- **LLM**: Qwen3-32B on RTX 5090 (32GB VRAM) - ~50 tokens/sec
- **Database**: PostgreSQL with TimescaleDB, indexed on ticker/date
- **Caching**: News (6h), earnings (24h), regime (1h)
- **Analysis Time**: ~15-30 seconds per ticker (full analysis)
- **Batch Processing**: 20 tickers parallel with rate limiting

---

## License

Private - Internal Use Only

---

## Contact

For questions about this platform, refer to this README, code comments, or the conversation history with the AI assistant.