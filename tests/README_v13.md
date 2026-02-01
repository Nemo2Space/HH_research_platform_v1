# HH Research Platform v2.0

## Institutional-Grade Alpha Research & Portfolio Management System

A comprehensive quantitative research platform designed for systematic equity analysis, featuring a 23-factor machine learning alpha model, real-time signal generation, earnings intelligence, and AI-powered research capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Alpha Model Enhancements](#alpha-model-enhancements)
5. [Signal Generation Pipeline](#signal-generation-pipeline)
6. [Earnings Intelligence System](#earnings-intelligence-system)
7. [AI Chat Integration](#ai-chat-integration)
8. [Dashboard Components](#dashboard-components)
9. [Database Schema](#database-schema)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Usage](#usage)
13. [API Reference](#api-reference)

---

## Overview

The HH Research Platform is a hedge fund-grade research infrastructure built for systematic equity analysis. It combines:

- **Multi-Factor Alpha Model**: 23-factor ML model with IC 0.1191, ICIR 2.04
- **Signal Quality Enhancements**: Forecast shrinkage, reliability gating, probability calibration
- **Earnings Intelligence**: IES/ECS/EQS pipeline to distinguish consensus from market expectations
- **Real-Time Portfolio Management**: IBKR integration, position tracking, drift analysis
- **AI Research Assistant**: LLM-powered analysis with strict decision policy binding

### Performance Metrics

| Metric | Value |
|--------|-------|
| Information Coefficient (IC) | 0.1191 |
| IC Information Ratio (ICIR) | 2.04 |
| Backtest Sharpe (Signal Strategy) | 1.26 |
| Alpha vs SPY | +40.65% |
| Portfolio Positions | 144 |
| AUM | $1.3M |

---

## Key Features

### 1. Multi-Factor Alpha Model
- 23 orthogonalized factors across momentum, value, quality, sentiment
- Sector-aware predictions with regime detection
- 5/10/20-day return forecasts with confidence intervals

### 2. Signal Quality Framework (NEW)
- **Forecast Shrinkage**: Context-aware k-factor (0.0-0.80)
- **ML Reliability Gate**: Vol-scaled bias, EWMA accuracy, soft N gate
- **Probability Calibration**: Empirical calibration with smoothing
- **Decision Policy Layer**: Strict binding rules for LLM

### 3. Sector Neutralization (NEW)
- Identifies sector leaders vs laggards
- Prevents sector momentum masquerading as alpha
- Z-score relative strength within sector

### 4. Sentiment Velocity (NEW)
- Tracks rate of change of sentiment (not just level)
- Detects "ignition patterns" (accelerating sentiment + flat price)
- Identifies exhaustion patterns

### 5. Earnings Intelligence
- **IES (Implied Expectations Score)**: Market-priced expectations
- **ECS (Earnings Catalyst Score)**: Event risk assessment
- **EQS (Earnings Quality Score)**: Quality of reported earnings

### 6. Portfolio Management
- Real-time IBKR position sync
- Target vs actual weight tracking
- Automated drift detection
- Transaction cost modeling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HH RESEARCH PLATFORM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Data       │  │   Signal     │  │   Alpha Model        │   │
│  │   Ingestion  │──│   Engine     │──│   (23 Factors)       │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│         │                │                      │                │
│         ▼                ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ALPHA ENHANCEMENTS LAYER (NEW)               │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │  │ Shrinkage  │ │ Reliability│ │ Calibration│            │   │
│  │  │ k=0.0-0.80 │ │ Gate       │ │ P(Win)     │            │   │
│  │  └────────────┘ └────────────┘ └────────────┘            │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │  │ Decision   │ │ Sector     │ │ Sentiment  │            │   │
│  │  │ Policy     │ │ Neutral    │ │ Velocity   │            │   │
│  │  └────────────┘ └────────────┘ └────────────┘            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                │                      │                │
│         ▼                ▼                      ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   AI Chat    │  │   Dashboard  │  │   Portfolio          │   │
│  │   (LLM)      │──│   (Streamlit)│──│   Management         │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Alpha Model Enhancements

### 1. Forecast Shrinkage

Extreme predictions are shrunk toward zero using a context-aware k-factor:

```
k = k_base + k_catalyst - k_regime - k_uncertainty
```

| Component | Default | Description |
|-----------|---------|-------------|
| k_base | 0.35 | Base shrinkage factor |
| k_catalyst | 0.0-0.35 | Bonus for confirmed catalysts |
| k_regime | 0.10 | Penalty for high VIX (>25) |
| k_uncertainty | 0.0-0.15 | Penalty for ML uncertainty |

**Example**: Raw +10% prediction with no catalyst, high VIX:
```
k = 0.35 + 0.00 - 0.10 - 0.05 = 0.20
Shrunk prediction = +10% × 0.20 = +2.0%
```

### 2. ML Reliability Gate

ML predictions are gated based on historical performance:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Sample count | < 40 | BLOCKED - ML informational only |
| Sample count | 40-200 | Soft weight degradation |
| Accuracy (EWMA) | < 50% | BLOCKED |
| Accuracy (EWMA) | 50-55% | DEGRADED |
| Accuracy (EWMA) | ≥ 58% | STRONG_ELIGIBLE |
| Vol-scaled bias | > 0.25 | BLOCKED |

### 3. Probability Calibration

Raw probabilities are calibrated using empirical win rates:

```python
# Binned calibration
bins = [(-100,-5), (-5,-3), (-3,-1), (-1,0), (0,1), (1,3), (3,5), (5,100)]
p_calibrated = actual_win_rate_in_bin

# Smoothing toward base rate
p_smoothed = w × p_calibrated + (1-w) × base_rate
```

### 4. Decision Policy Layer

The LLM is bound by deterministic policy rules:

```
===================================================================
🎯 DECISION POLICY (BINDING)
===================================================================
Trade Allowed: ✅ YES / ❌ NO
Action: BUY / SELL / HOLD / WAIT_FOR_TRIGGER / NO_TRADE
Size Cap: 0.50x MAXIMUM
Confidence: LOW / MEDIUM / HIGH

LLM RULES:
  ✅ MAY: Explain, suggest timing, add risks, downgrade, reduce size
  ❌ MAY NOT: Upgrade action, exceed size cap, ignore blockers
===================================================================
```

### 5. Catalyst Detection

Catalysts are scored 0-3 based on:

| Score | Level | Criteria |
|-------|-------|----------|
| 0 | NONE | No catalysts detected |
| 1 | WEAK | News keywords only |
| 2 | MEDIUM | Earnings in 3-10 days OR IV Rank 60-75 |
| 3 | STRONG | Earnings in ≤3 days AND IV Rank >75 |

### 6. Sector Neutralization

Identifies if a stock is outperforming its sector:

```
📊 SECTOR CONTEXT (Technology):
   Rank: #3 of 28 in sector
   Percentile: 89th
   Score: 72.5 vs Sector Avg: 58.2
   ⭐ SECTOR LEADER - Outperforming peers
```

### 7. Sentiment Velocity

Tracks rate of change of sentiment:

```
📈 SENTIMENT VELOCITY:
   3-Day Change: +8.2 points
   7-Day Change: +12.5 points
   Signal: 🚀 ACCELERATING_BULLISH
   💡 IGNITION PATTERN: Rising sentiment - watch for price breakout
```

---

## Signal Generation Pipeline

### Unified Signal Flow

```
Market Data → Technical Analysis ──┐
News Feed → Sentiment Analysis ────┼─→ Unified Signal → Decision Policy
Options Flow → Flow Analysis ──────┤
Fundamentals → Factor Model ───────┘
```

### Signal Components

| Component | Weight | Data Source |
|-----------|--------|-------------|
| Technical Score | 25% | Price/Volume |
| Sentiment Score | 25% | News/Social |
| Options Flow | 25% | Options data |
| Fundamental Score | 25% | Financials |

### Conflict Resolution

When platform signal conflicts with Alpha Model:

| Conflict Level | Platform | Alpha | Action |
|----------------|----------|-------|--------|
| 0 (Aligned) | BUY | BUY | Full size allowed |
| 1 (Mild) | HOLD | BUY | WAIT_FOR_TRIGGER |
| 2 (Moderate) | SELL | BUY | 0.25x size, requires triggers |
| 3+ (Hard) | SELL | STRONG_BUY | NO_TRADE |

---

## Earnings Intelligence System

### IES (Implied Expectations Score)

Measures market-priced expectations vs consensus:

| IES Score | Interpretation |
|-----------|----------------|
| > 75 | HYPED - Very high expectations, risky |
| 50-75 | MODERATE expectations |
| < 35 | FEARED - Low bar, opportunity |

### ECS (Earnings Catalyst Score)

Assesses event risk around earnings:

- Days to earnings
- Historical earnings volatility
- Options implied move
- Analyst revision momentum

### EQS (Earnings Quality Score)

Evaluates quality of reported earnings:

- Accruals ratio
- Cash conversion
- Revenue quality
- Guidance credibility

---

## AI Chat Integration

### System Prompt Features

The AI Chat acts as a "Senior Quantitative Analyst" with:

1. **Evidence Rules**: Must cite sources, no hallucinated numbers
2. **Decision Policy Compliance**: Bound by deterministic policy
3. **Regime Awareness**: Adjusts for market conditions
4. **Skepticism Rules**: Flags extreme predictions

### Context Data Flow

```python
# Data provided to LLM
context = {
    "decision_policy": "BINDING rules",
    "catalyst_info": "Earnings, IV, VIX",
    "reliability_metrics": "Accuracy, bias, samples",
    "shrinkage_result": "Raw → Shrunk prediction",
    "calibration": "Raw → Calibrated probability",
    "sector_context": "Rank within sector",
    "sentiment_velocity": "Rate of change"
}
```

### Temperature Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| Temperature | 0.15 | Consistent, factual outputs |
| Top-p | 0.85 | Tighter nucleus sampling |
| Repeat penalty | 1.1 | Reduce repetition |

---

## Dashboard Components

### Available Tabs

| Tab | Description |
|-----|-------------|
| Research | AI Chat with full context |
| Signals | Real-time signal generation |
| Alpha Model | ML predictions and factors |
| Portfolio | IBKR positions and drift |
| Backtesting | Strategy backtesting |
| Risk | VaR, exposure analysis |
| Options | Options flow analysis |
| Earnings | Earnings calendar and IES |

---

## Database Schema

### Core Tables

```sql
-- Alpha predictions with outcome tracking
CREATE TABLE alpha_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    expected_return_5d DECIMAL(8, 4),
    expected_return_10d DECIMAL(8, 4),
    expected_return_20d DECIMAL(8, 4),
    prob_positive_5d DECIMAL(5, 4),
    signal VARCHAR(20),
    conviction DECIMAL(4, 3),
    actual_return_5d DECIMAL(8, 4),  -- Outcome tracking
    actual_return_10d DECIMAL(8, 4),
    prediction_error_5d DECIMAL(8, 4),
    UNIQUE(ticker, prediction_date)
);

-- Earnings calendar
CREATE TABLE earnings_calendar (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(10),
    UNIQUE(ticker, earnings_date)
);

-- Options summary
CREATE TABLE options_summary (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    iv_rank DECIMAL(5, 2),
    implied_volatility DECIMAL(8, 4),
    UNIQUE(ticker, date)
);
```

---

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (TimescaleDB recommended)
- Node.js 18+ (for docx generation)

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/HH-research-platform.git
cd HH-research-platform

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup database
psql -U alpha -d alpha_platform -f setup_alpha_tables.sql

# Copy environment file
cp .env.example .env
# Edit .env with your credentials
```

---

## Configuration

### Environment Variables (.env)

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=alpha_platform
POSTGRES_USER=alpha
POSTGRES_PASSWORD=your_password

# LLM
LLM_QWEN_BASE_URL=http://localhost:8090/v1
LLM_QWEN_MODEL=Qwen3-32B-Q6_K.gguf

# API Keys
FINNHUB_API_KEY=your_key
NEWSAPI_API_KEY=your_key
```

### Alpha Enhancement Config

```python
# src/ml/alpha_enhancements.py
CONFIG = EnhancementConfig(
    # Shrinkage
    shrinkage_k_base=0.35,
    shrinkage_k_catalyst_max=0.35,
    shrinkage_k_regime_penalty=0.10,
    shrinkage_k_uncertainty_max=0.15,
    
    # Reliability Gate
    min_samples_hard_block=40,
    min_samples_full_weight=200,
    accuracy_hard_block=0.50,
    accuracy_degraded=0.55,
    accuracy_strong_signal=0.58,
    
    # Catalyst
    earnings_days_strong=3,
    vix_high_threshold=25,
)
```

---

## Usage

### Start Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Run Alpha Model

```bash
# Generate predictions
python -m src.ml.multi_factor_alpha --predict

# Update outcomes
python -m src.ml.prediction_tracker --update
```

### Test Enhancements

```bash
python -m src.ml.alpha_enhancements
```

---

## API Reference

### Alpha Enhancements Module

```python
from src.ml.alpha_enhancements import (
    build_enhanced_alpha_context,
    get_reliability_metrics,
    apply_forecast_shrinkage,
    get_calibrator,
    compute_decision_policy,
    get_sector_context,
    get_sentiment_velocity
)

# Get full enhanced context for a ticker
context = build_enhanced_alpha_context(
    ticker="AAPL",
    alpha_prediction=prediction_dict,
    platform_score=65.0,
    platform_signal="BUY",
    technical_score=55.0
)

# Get reliability metrics
reliability = get_reliability_metrics(days_back=90)
print(f"Status: {reliability.get_status(5)}")
print(f"Weight: {reliability.get_reliability_weight(5):.0%}")

# Get sector context
sector = get_sector_context("AAPL")
print(f"Rank: #{sector.sector_rank} of {sector.sector_count}")
```

### Chat Module

```python
from src.ai.chat import AlphaChat

chat = AlphaChat()
response = chat.chat("Analyze NVDA", ticker="NVDA")
print(response)
```

---

## Version History

### v2.0.0 (December 2025)
- Added Alpha Model Enhancements
  - Forecast Shrinkage
  - ML Reliability Gate
  - Probability Calibration
  - Decision Policy Layer
  - Catalyst Detection
- Added Sector Neutralization
- Added Sentiment Velocity
- Added Evidence Rules to AI prompt
- Fixed P0 bugs (SQL, escalation, size cap)
- Unit consistency fixes

### v1.0.0 (November 2025)
- Initial release
- 23-factor Alpha Model
- Signal generation pipeline
- IBKR integration
- Streamlit dashboard

---

## License

Proprietary - HH Research Platform

---

## Support

For issues or questions, contact the development team.