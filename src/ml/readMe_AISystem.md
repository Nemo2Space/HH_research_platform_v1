# AI Trading System

## Intelligent Signal Enhancement for HH Research Platform

A professional-grade AI system that learns from trading history to improve signal quality.

**Version:** 1.0  
**Last Updated:** December 2025

---

## Overview

This AI system implements the architecture recommended by quantitative trading experts:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI TRADING SYSTEM                                    │
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Signal    │────▶│  ML Model   │────▶│   RAG       │                  │
│   │   Scores    │     │  (XGBoost)  │     │  Memory     │                  │
│   │             │     │             │     │             │                  │
│   │ sentiment   │     │ Probability │     │ Similar     │                  │
│   │ fundamental │     │ + EV        │     │ Setups      │                  │
│   │ technical   │     │             │     │             │                  │
│   └─────────────┘     └─────────────┘     └─────────────┘                  │
│          │                  │                   │                           │
│          ▼                  ▼                   ▼                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      META-LABELER                                    │  │
│   │              "Should we take this trade?"                           │  │
│   │         Filters false positives from primary model                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     DECISION LAYER                                   │  │
│   │           EV Calculation + Position Sizing + Risk Limits            │  │
│   │                                                                      │  │
│   │  EV = p × avg_win - (1-p) × avg_loss - costs                       │  │
│   │  Only trade when EV > threshold AND risk limits pass               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      LLM INTEGRATION                                 │  │
│   │         Qwen explains decisions (no invented statistics)            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    MONITORING + FEEDBACK                             │  │
│   │         Log recommendations → Track outcomes → Detect drift         │  │
│   │                  Retrain only when metrics degrade                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Principles

### 1. ML for Patterns, LLM for Explanation

| Component | Good At | Bad At |
|-----------|---------|--------|
| **XGBoost** | Finding statistical patterns, probabilities | Explaining why, edge cases |
| **Qwen LLM** | Explaining reasoning, synthesizing info | Computing statistics reliably |

**We use each for what it's good at.**

### 2. Expected Value, Not Just Probability

```python
# Don't trade on probability alone!
EV = probability × avg_win - (1 - probability) × avg_loss - costs

# Only trade when:
# 1. EV > 0.1% (minimum edge)
# 2. Probability > 55% (confidence)
# 3. All risk limits pass
```

### 3. Leakage-Safe Validation

```
❌ WRONG: Random train/test split (leaks future data)
✅ RIGHT: Walk-forward validation (train on past, test on future)

[====TRAIN====][PURGE][==TEST==]
                    [====TRAIN====][PURGE][==TEST==]
```

### 4. Triggered Retraining, Not Scheduled

```
❌ WRONG: Retrain every week (overfits to noise)
✅ RIGHT: Retrain when drift detected
   - Calibration error > 10%
   - Win rate dropped > 10%
   - Regime changed significantly
```

---

## Components

### 1. ML Signal Predictor (`signal_predictor.py`)

**Purpose:** Predict win probability with calibrated outputs

**Features:**
- Multi-horizon predictions (1d, 2d, 5d, 10d)
- Walk-forward validation (no data leakage)
- Probability calibration (Isotonic scaling)
- Must beat logistic regression baseline
- Cost-aware evaluation

**Usage:**
```python
from signal_predictor import MLSignalPredictor

predictor = MLSignalPredictor()
predictor.train()  # Walk-forward validation

result = predictor.predict({
    'sentiment_score': 72,
    'fundamental_score': 65,
    'technical_score': 68,
    'options_flow_score': 75,
    'total_score': 70
})

print(f"5-Day Win Prob: {result.prob_win_5d:.1%}")
print(f"Expected Value: {result.ev_5d:.3%}")
print(f"Should Trade: {result.should_trade}")
```

---

### 2. Decision Layer (`decision_layer.py`)

**Purpose:** Convert probabilities into actionable trades

**Features:**
- EV calculation with transaction costs
- Position sizing (Kelly, Volatility Targeting)
- Risk limits (max position, sector, drawdown)
- VIX-based adjustments
- Earnings blackout periods

**Risk Limits:**
| Limit | Default | Description |
|-------|---------|-------------|
| Max Position | 5% | Max single position size |
| Max Sector | 25% | Max sector exposure |
| Max Drawdown | 10% | Halt trading threshold |
| Max VIX | 35 | Don't open new positions |
| Earnings Blackout | 2 days | No trades near earnings |

**Usage:**
```python
from decision_layer import DecisionLayer, PortfolioState

dl = DecisionLayer(min_ev=0.001, min_probability=0.55)

recommendation = dl.evaluate(
    ticker="NVDA",
    ml_prediction={'prob_win_5d': 0.72, 'ev_5d': 0.008},
    portfolio=portfolio_state,
    stock_data={'price': 140, 'sector': 'Technology'}
)

if recommendation.approved:
    print(f"BUY {recommendation.position_size.shares} shares")
    print(f"Entry: ${recommendation.entry_price:.2f}")
    print(f"Stop: ${recommendation.stop_loss:.2f}")
else:
    print(f"Skip: {recommendation.rejection_reasons}")
```

---

### 3. RAG Memory (`rag_memory.py`)

**Purpose:** Case-based reasoning from similar historical setups

**Features:**
- Stores "setup cards" (snapshots at decision time)
- Similarity search on features (NOT outcomes)
- Shows empirical win rates from similar trades
- Leakage-safe: only uses T0 data

**Usage:**
```python
from rag_memory import RAGMemoryStore, SetupCard

store = RAGMemoryStore()

# Find similar setups
result = store.find_similar(current_setup_card, top_k=20)

print(f"Found {result.closed_count} similar trades")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Avg Return: {result.avg_return:+.2f}%")
print(result.get_insight_summary())
```

---

### 4. Meta-Labeler (`meta_labeler.py`)

**Purpose:** Second model that filters false positives

**Insight:** Not all BUY signals should be traded. This model learns which setups actually profit.

**Features:**
- Combines primary probability with meta confidence
- Identifies positive/negative factors
- Adjusts position size based on confidence

**Usage:**
```python
from meta_labeler import MetaLabeler

labeler = MetaLabeler()

result = labeler.predict(
    primary_prediction={'prob_win_5d': 0.68},
    scores={'sentiment_score': 72, 'options_flow_score': 65},
    market_context={'vix': 18},
    similar_setups={'win_rate': 0.65}
)

print(f"Should Trade: {result.should_trade}")
print(f"Combined Prob: {result.combined_prob:.1%}")
print(f"Positive: {result.positive_factors}")
print(f"Negative: {result.negative_factors}")
```

---

### 5. LLM Integration (`llm_integration.py`)

**Purpose:** Generate explanations without invented statistics

**CRITICAL RULE:** LLM never invents numbers. All statistics come from tools.

**Features:**
- Tool-enforced computation
- Explains ML model outputs
- Identifies "what would change my mind"
- Highlights risks and thesis breakers

**Usage:**
```python
from llm_integration import LLMIntegration

llm = LLMIntegration()

analysis = llm.analyze_trade(
    ticker="NVDA",
    ml_prediction=ml_output,
    decision_result=decision_output,
    similar_setups=rag_output,
    scores=signal_scores
)

print(analysis.summary)
print(f"Recommendation: {analysis.recommendation}")
print(f"Risks: {analysis.risks}")
```

---

### 6. Monitoring (`monitoring.py`)

**Purpose:** Feedback loop and drift detection

**Features:**
- Logs all recommendations + outcomes
- Detects calibration drift
- Detects performance degradation
- Triggers retraining when needed (not on schedule)

**Drift Types:**
| Type | Detection | Threshold |
|------|-----------|-----------|
| Calibration | Predicted vs Actual win rate | > 10% difference |
| Performance | Win rate drop from baseline | > 10% drop |
| Regime | VIX or regime score | Extreme values |

**Usage:**
```python
from monitoring import RecommendationLogger, DriftDetector

logger = RecommendationLogger()
detector = DriftDetector()

# Log recommendation
logger.log_recommendation(ticker, ml_pred, meta_result, decision)

# Later: update outcome
logger.update_outcome(ticker, date, "WIN", 3.5, 5, "TARGET")

# Check for drift
trigger = detector.should_retrain(logger.get_recent_recommendations(30))
if trigger.triggered:
    print(f"Retrain needed: {trigger.trigger_reason}")
```

---

### 7. Main Orchestrator (`ai_trading_system.py`)

**Purpose:** Ties all components together

**Usage:**
```python
from ai_trading_system import AITradingSystem

# Initialize
system = AITradingSystem()
system.initialize()

# Analyze stock
result = system.analyze("NVDA", scores, stock_data)

print(result.get_summary())
# Shows: ML prob, EV, similar setups, decision, position size, etc.

# Check health
health = system.check_health()
print(f"Win Rate: {health['performance']['win_rate']}")
print(f"Needs Retrain: {health['drift']['needs_retrain']}")

# Update outcome (feedback loop)
system.update_outcome("NVDA", date(2025, 1, 15), "WIN", 4.2, 5, "TARGET")
```

---

## Installation

### 1. Install Dependencies

```bash
pip install xgboost scikit-learn pandas numpy scipy
```

### 2. Run Database Migrations

```bash
psql -d alpha_platform -f migrations/ai_system_tables.sql
```

### 3. Copy Files to Project

```
src/ml/
├── signal_predictor.py
├── decision_layer.py
├── rag_memory.py
├── meta_labeler.py
├── llm_integration.py
├── monitoring.py
├── ai_trading_system.py
└── integration.py
```

### 4. Initialize System

```python
from ai_trading_system import AITradingSystem

system = AITradingSystem()
system.initialize(train_if_needed=True)  # Will train if no model exists
```

---

## Integration with Existing Platform

### Add to Signals Tab

```python
# In signals_tab.py, after traditional analysis:
from integration import render_ai_analysis_section

render_ai_analysis_section(ticker, scores)
```

### Add to AI Chat Context

```python
# In chat.py, when building context:
from integration import get_ai_context_for_chat

ai_context = get_ai_context_for_chat(ticker)
context += ai_context
```

### Add as New Tab

```python
# In app.py:
from integration import render_ai_system_tab

if selected_tab == "AI System":
    render_ai_system_tab()
```

---

## File Structure

```
src/ml/
├── signal_predictor.py     # ML model with walk-forward validation
├── decision_layer.py       # Position sizing + risk limits
├── rag_memory.py           # Similar historical setups (RAG)
├── meta_labeler.py         # False positive filter
├── llm_integration.py      # LLM with tool enforcement
├── monitoring.py           # Drift detection + feedback loop
├── ai_trading_system.py    # Main orchestrator
└── integration.py          # Platform integration helpers

migrations/
└── ai_system_tables.sql    # Database tables

models/
├── signal_predictor.pkl    # Trained ML model
└── meta_labeler.pkl        # Trained meta-labeler
```

---

## Best Practices

### DO ✅

- **Validate with walk-forward splits** (not random)
- **Calibrate probabilities** (so 70% means 70%)
- **Include transaction costs** in training and evaluation
- **Track outcomes** for continuous improvement
- **Retrain when metrics degrade** (not on schedule)
- **Use meta-labeling** to filter false positives

### DON'T ❌

- **Random train/test splits** (causes leakage)
- **Trust raw ML probabilities** (often miscalibrated)
- **Ignore transaction costs** (makes backtests unrealistic)
- **Weekly retraining** (overfits to noise)
- **Let LLM invent statistics** (hallucination risk)

---

## Performance Expectations

Based on walk-forward validation:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| AUC | 0.55 - 0.65 | Must beat 0.52 baseline |
| Win Rate | 55% - 65% | After costs |
| Calibration | < 10% error | Predicted vs actual |
| Avg Return | 2% - 4% | Per trade, 5-day horizon |

**Important:** If the model doesn't beat the logistic regression baseline, the system will warn you. This usually means the signal data needs improvement.

---

## Troubleshooting

### Model Won't Train

```
Check: Do you have enough data in historical_scores?
Need: At least 500 rows with return_5d not null
```

### Low AUC / Doesn't Beat Baseline

```
Check: Are your signal scores actually predictive?
Try: Test individual signals separately
Fix: Improve signal quality before AI enhancement
```

### Calibration Drift Detected

```
Cause: Market regime changed, model not adapted
Fix: Review recent performance by regime
Action: Retrain if drift persists > 2 weeks
```

### Too Many Rejections

```
Check: Are risk limits too strict?
Try: Lower min_probability to 0.52
Try: Increase max_position_pct
```

---

## Credits

This system implements recommendations from quantitative trading experts:

- Walk-forward validation (standard in quant finance)
- Meta-labeling (López de Prado)
- Probability calibration (Platt, Isotonic)
- Expected value gating (Kelly Criterion)
- Drift detection (statistical process control)

---

## License

Private - Internal Use Only