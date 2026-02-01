# 🔬 AI Research Agent - HH Research Platform
## Version 3.0 - Institutional Grade

## Overview

The AI Research Agent is a sophisticated, **institutional-grade** stock research assistant designed with the rigor expected at firms like BlackRock. It combines multiple data sources to provide **evidence-based investment recommendations** while enforcing strict guardrails against hallucination and errors.

### Key Differentiators from Basic AI Chatbots:

| Feature | Basic Chatbot | This Agent |
|---------|---------------|------------|
| Math/Calculations | LLM estimates (unreliable) | **Deterministic Python calculations** |
| Data Freshness | Assumes current | **Staleness circuit breaker** |
| Source Quality | All equal | **Tiered credibility (Tier 1-5)** |
| Edge Cases | LLM judgment | **Hard-coded guardrails** |
| Null Values | May hallucinate | **Explicit N/A handling** |
| Recommendations | "BUY/SELL" | **Outlook language + disclaimers** |

---

## Institutional Features Audit

Based on a comprehensive audit against institutional standards, here's what's implemented:

### ✅ 1. Deterministic Math (No LLM Calculations)

**Problem:** LLMs are stochastic - if Qwen says "20% upside" but math is actually 18.5%, you have liability.

**Solution Implemented:**
```python
def compute_upside_pct(current_price: float, target_price: float) -> Optional[float]:
    """DETERMINISTIC calculation - computed in Python, NOT by LLM."""
    if current_price <= 0 or target_price <= 0:
        return None
    upside = ((target_price - current_price) / current_price) * 100
    return round(upside, 2)

def compute_price_change(current: float, previous: float) -> Dict:
    """DETERMINISTIC price change calculation."""
    change = current - previous
    change_pct = round((change / previous) * 100, 2)
    return {"change": round(change, 2), "change_pct": change_pct, ...}
```

The LLM only **reads** pre-computed values, never calculates them.

---

### ✅ 2. Data Staleness Circuit Breaker

**Problem:** If `MAX(date)` is 5 days old due to pipeline failure, agent presents stale data as "current."

**Solution Implemented:**
```python
STALENESS_THRESHOLDS = {
    "intraday": 1,      # 1 hour max for intraday questions
    "short": 24,        # 24 hours for short-term
    "medium": 72,       # 3 days for medium-term
    "long": 168,        # 1 week for long-term
    "very_long": 336,   # 2 weeks for very long-term
}

def check_data_staleness(data_timestamp, horizon) -> Dict:
    """Returns staleness warning if data is too old for the question type."""
    age_hours = (now - data_timestamp).total_seconds() / 3600
    threshold = STALENESS_THRESHOLDS.get(horizon, 24)
    
    if age_hours > threshold:
        return {
            "is_stale": True,
            "warning": f"⚠️ DELAYED DATA: {age_hours:.1f} hours old..."
        }
```

Intraday questions with >1 hour old data get explicit warnings.

---

### ✅ 3. Null/NaN Handling

**Problem:** A valid row with `pe_ratio = NULL` could cause LLM to hallucinate a number.

**Solution Implemented:**
```python
def sanitize_value(value: Any, default: str = "N/A") -> Any:
    """Convert NaN/None to explicit 'N/A' string."""
    if value is None:
        return default
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return default
    if pd.isna(value):
        return default
    return value

def sanitize_dict(d: Dict) -> Dict:
    """Sanitize all values in a dictionary before prompt injection."""
    return {k: sanitize_value(v) for k, v in d.items()}
```

All data is sanitized before reaching the prompt.

---

### ✅ 4. Source Credibility Tiering

**Problem:** `fool.com` and `sec.gov` shouldn't be treated as equal sources.

**Solution Implemented:**
```python
SOURCE_TIERS = {
    # Tier 1: Regulatory/Official (100% weight) - FACT
    "sec.gov": {"tier": 1, "weight": 100, "label": "REGULATORY"},
    "edgar": {"tier": 1, "weight": 100, "label": "REGULATORY"},
    "federalreserve.gov": {"tier": 1, "weight": 100, "label": "REGULATORY"},
    
    # Tier 2: Premium Financial News (80% weight) - VERIFIED
    "bloomberg": {"tier": 2, "weight": 80, "label": "VERIFIED"},
    "reuters": {"tier": 2, "weight": 80, "label": "VERIFIED"},
    "wsj": {"tier": 2, "weight": 80, "label": "VERIFIED"},
    
    # Tier 3: Financial Media (60% weight) - CREDIBLE
    "marketwatch": {"tier": 3, "weight": 60, "label": "CREDIBLE"},
    "yahoo": {"tier": 3, "weight": 60, "label": "CREDIBLE"},
    
    # Tier 4: Opinion/Analysis (40% weight) - OPINION
    "seekingalpha": {"tier": 4, "weight": 40, "label": "OPINION"},
    "fool.com": {"tier": 4, "weight": 40, "label": "OPINION"},
    
    # Tier 5: Social/Unverified (20% weight) - UNVERIFIED
    "reddit": {"tier": 5, "weight": 20, "label": "UNVERIFIED"},
}
```

News is labeled in the prompt: `[TIER 1 - REGULATORY]` vs `[TIER 4 - OPINION]`

---

### ✅ 5. Headline Deduplication

**Problem:** 5 outlets reporting the same rumor shouldn't count as "strong consensus."

**Solution Implemented:**
```python
def deduplicate_headlines(articles: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
    """
    Deduplication using word overlap as similarity proxy.
    5 articles about the same event count as 1 event, not 5 evidence points.
    """
    seen_hashes = set()
    unique_articles = []
    
    for article in articles:
        # Create content hash from key words
        words = set(headline.lower().split())
        
        # Check similarity with existing
        is_duplicate = False
        for seen_words in seen_hashes:
            overlap = len(words & seen_words) / max(len(words | seen_words), 1)
            if overlap > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_articles.append(article)
            seen_hashes.add(frozenset(words))
    
    return unique_articles
```

---

### ✅ 6. Hard-Coded Guardrails

**Problem:** If sentiment is 90/100 but P/E is 500x, LLM might say "BUY." A professional analyst would say "CAUTION."

**Solution Implemented:**
```python
GUARDRAIL_RULES = {
    "overbought_euphoria": {
        "conditions": {"sentiment_score": (">", 80), "technical_score": (">", 75)},
        "override": "CAUTION - OVERBOUGHT",
        "message": "⚠️ GUARDRAIL: High sentiment + technicals suggest overbought conditions."
    },
    "extreme_valuation": {
        "conditions": {"pe_ratio": (">", 100)},
        "override": "CAUTION - EXTREME VALUATION",
        "message": "⚠️ GUARDRAIL: P/E >100 indicates extreme valuation risk."
    },
    "panic_selling": {
        "conditions": {"sentiment_score": ("<", 20), "technical_score": ("<", 25)},
        "override": "CAUTION - OVERSOLD",
        "message": "⚠️ GUARDRAIL: Extreme negative sentiment may indicate panic."
    },
    "low_liquidity": {
        "conditions": {"options_flow_score": ("<", 15)},
        "override": "CAUTION - LOW LIQUIDITY",
        "message": "⚠️ GUARDRAIL: Low options activity suggests limited institutional interest."
    }
}

def check_guardrails(scores: Dict, fundamentals: Dict) -> List[Dict]:
    """Check if guardrails are triggered - OVERRIDES LLM output."""
    triggered = []
    for rule_name, rule in GUARDRAIL_RULES.items():
        # Check all conditions...
        if all_conditions_met:
            triggered.append(rule)
    return triggered
```

These guardrails are evaluated in Python and **override** whatever the LLM says.

---

### ✅ 7. Compliance Disclaimers

**Problem:** Agent giving "BUY/SELL" recommendations without disclaimers.

**Solution Implemented:**
```python
DISCLAIMERS = {
    "standard": "⚖️ DISCLAIMER: This analysis is for informational purposes only and does not constitute financial advice. Past performance is not indicative of future results. Always consult a licensed financial advisor.",
    "high_risk": "🚨 HIGH RISK WARNING: This security exhibits elevated risk characteristics.",
    "speculative": "⚠️ SPECULATIVE: Limited institutional coverage or high volatility."
}
```

Every response includes appropriate disclaimers. Language uses "Outlook: CONSTRUCTIVE" instead of "BUY."

---

## Features NOT Yet Implemented (Future Roadmap)

| Feature | Status | Priority |
|---------|--------|----------|
| Async/Parallel Queries | ❌ Not implemented | Medium |
| JSON Structured Logging | ❌ Not implemented | Low |
| Click-to-Verify Citations (UI) | ❌ Not implemented | Medium |
| Thread-Safe Agent Instances | ⚠️ Relies on Streamlit session_state | Low |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER QUESTION                                        │
│            "Why is Nike falling? Should I buy the dip?"                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      1. CLAIM DETECTION & VERIFICATION                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Detected: User claims NKE is "falling"                              │    │
│  │ Verification: Query get_price_change('NKE')                         │    │
│  │ Result: NKE is DOWN -2.3% ✓ (Claim verified)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      2. MULTI-SOURCE DATA GATHERING                          │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  WEB SEARCH  │  │ YOUR NEWS DB │  │PLATFORM SCORES│  │ FUNDAMENTALS │    │
│  │  (Tavily/    │  │ news_articles│  │screener_scores│  │fundamentals  │    │
│  │   Brave)     │  │              │  │               │  │price_targets │    │
│  │              │  │ • Headlines  │  │ • Total Score │  │              │    │
│  │ • Current    │  │ • Sentiment  │  │ • Sentiment   │  │ • P/E Ratio  │    │
│  │   articles   │  │ • Source     │  │ • Technical   │  │ • Growth     │    │
│  │ • Analyst    │  │ • Snippet    │  │ • Options     │  │ • Targets    │    │
│  │   opinions   │  │              │  │ • Squeeze     │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐                                         │
│  │ PRICE DATA   │  │ SCORE TREND  │                                         │
│  │   prices     │  │  (5-day)     │                                         │
│  │              │  │              │                                         │
│  │ • Latest     │  │ • Direction  │                                         │
│  │ • Change %   │  │ • Momentum   │                                         │
│  │ • Volume     │  │              │                                         │
│  └──────────────┘  └──────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      3. COMPREHENSIVE PROMPT BUILDING                        │
│                                                                              │
│  The agent builds a detailed prompt with ALL data, including:               │
│  • Fact-check results (corrections if user was wrong)                       │
│  • Full stock analysis with all scores                                      │
│  • Actual news headlines with sources                                       │
│  • Fundamentals and price targets                                           │
│  • Price history and trends                                                 │
│  • Strict instructions to cite sources and not hallucinate                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      4. AI ANALYSIS & RESPONSE                               │
│                                                                              │
│  The AI (Qwen) generates a structured response:                             │
│  1. Summary - Direct answer to the question                                 │
│  2. News Analysis - Citing specific headlines                               │
│  3. Data Analysis - Referencing actual scores/fundamentals                  │
│  4. Bull vs Bear Case - With data citations                                 │
│  5. Recommendation - Clear BUY/HOLD/SELL with reasoning                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🔍 Fact-Checking & Claim Verification

The agent automatically detects claims in your questions and verifies them:

| User Says | Agent Does |
|-----------|------------|
| "AAPL dropped today" | Checks actual price → Corrects if wrong |
| "Nike beat earnings" | Searches news to verify |
| "I heard MSFT is buying..." | Searches web to confirm/deny |

**Example Correction:**
```
User: "Why did AAPL crash today?"

Agent: "I need to correct something first - you mentioned AAPL crashed, 
but according to my price data, AAPL is actually UP +1.2% to $243.50 today.

Let me analyze what's actually happening with Apple..."
```

### 📰 News Integration

Uses your **already-collected news** from the `news_articles` table:

- Full headlines (not truncated)
- Article snippets for context
- Sentiment scores (positive/negative/neutral)
- Source attribution
- Publication dates

**News Display in Prompt:**
```
### NKE - Recent News Headlines:

1. [🔴 NEGATIVE] **Nike Reports Q4 Earnings: Revenue Down 17% Year-Over-Year**
   Source: Bloomberg | Date: 2025-01-02
   Summary: Nike Inc. reported disappointing Q4 results with revenue declining...

2. [🟢 POSITIVE] **Nike CEO Acquires 50,000 Shares in Open Market Purchase**
   Source: SEC Filings | Date: 2025-01-01
   Summary: Nike's CEO purchased 50,000 shares at $75.20, signaling confidence...
```

### 📊 Comprehensive Stock Analysis

For any stock mentioned, the agent retrieves:

| Data Category | Fields |
|---------------|--------|
| **Platform Scores** | Total, Sentiment, Fundamental, Technical, Options Flow, Short Squeeze |
| **Fundamentals** | P/E Ratio, Revenue Growth, Profit Margin, Dividend Yield, Sector |
| **Price Targets** | Low, Mean, High, Recommendation, # Analysts |
| **Price History** | Latest price, Daily range, Volume, Multi-day change |
| **Score Trend** | 5-day trend direction (improving/declining/stable) |

### 🌐 Web Search Integration

Searches the web via **Tavily** and **Brave** APIs for:

- Current news not yet in your database
- Analyst opinions and reports
- Market-moving events
- Ticker-specific developments

### ⏰ Time Horizon Awareness

Adjusts analysis based on investment timeframe:

| Time Horizon | Focus Areas |
|--------------|-------------|
| **Intraday** | Technical momentum, news catalysts |
| **Short-term** (days/weeks) | Technicals, sentiment, options flow |
| **Medium-term** (months) | Fundamentals + technicals balanced |
| **Long-term** (1-5 years) | Fundamentals, valuation, growth |
| **Very Long** (10+ years) | Moat, sustainability, industry trends |

---

## Installation

### 1. Copy Files

```powershell
# Copy the main agent file
Copy-Item ai_research_agent.py src/ai/ai_research_agent.py -Force

# Copy the Streamlit tab
Copy-Item ai_assistant_tab.py src/tabs/ai_assistant_tab.py -Force
```

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# LLM Configuration (Qwen)
LLM_QWEN_BASE_URL=http://172.23.193.91:8090/v1
LLM_QWEN_MODEL=Qwen3-32B-Q6_K.gguf

# Web Search APIs (at least one required for web search)
TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxxx
BRAVE_API_KEY=BSAxxxxxxxxxxxxx

# Optional: Finnhub for additional news
FINNHUB_API_KEY=your_key_here
```

### 3. Start Tool Server (for web search)

```bash
cd ~/ai_tools
source venv/bin/activate
export TAVILY_API_KEY="your_key"
export BRAVE_API_KEY="your_key"
python tools_api/tool_server.py
```

### 4. Verify in Streamlit

The AI Research Agent should appear as a navigation option: **🤖 AI Assistant**

---

## Usage

### Quick Research Buttons

| Button | What It Researches |
|--------|-------------------|
| 🚀 Best for Next Week | Short-term momentum plays |
| 📈 Best for 2025 | Medium-term growth opportunities |
| 💎 Best for 10 Years | Long-term compounders with moat |
| ⚠️ Stocks to Avoid | Low-score stocks with risks |
| 🏭 Sector Analysis | Sector strength comparison |
| 📰 Market News | Current events impact |
| 💰 Value Opportunities | Undervalued fundamentals |
| ⚡ Momentum Plays | Technical breakouts |

### Custom Questions

**Good questions to ask:**

```
• "Why is NKE falling today?"
• "Should I buy NVDA at current prices?"
• "Compare AAPL vs MSFT for long-term investment"
• "What are the best AI stocks for the next 5 years?"
• "Which stocks have insider buying recently?"
• "What does the news say about Tesla?"
• "Is now a good time to buy tech stocks?"
```

### Expected Response Format

The AI follows a structured format:

```markdown
**1. SUMMARY**
[Direct answer to your question]

**2. NEWS ANALYSIS**
According to [Source], "[Headline]" - [Explanation of impact]
...

**3. DATA ANALYSIS**
- Platform Score: X/100 (interpretation)
- Fundamental Score: Y/100
- Technical Score: Z/100
- Analyst Target: $XX vs Current $YY = Z% upside
...

**4. BULL VS BEAR CASE**
🟢 BULLISH:
- [Factor 1] (Source: [data source])
- [Factor 2] (Source: [data source])

🔴 BEARISH:
- [Factor 1] (Source: [data source])
- [Factor 2] (Source: [data source])

**5. CONCLUSION & RECOMMENDATION**
[BUY/HOLD/SELL] - [Reasoning backed by data]
```

---

## Database Tables Used

| Table | Purpose |
|-------|---------|
| `screener_scores` | Platform scoring (total, sentiment, technical, etc.) |
| `fundamentals` | Company fundamentals (P/E, growth, margins) |
| `price_targets` | Analyst targets and recommendations |
| `prices` | Historical price data |
| `news_articles` | Collected news with sentiment |
| `universe` | Stock universe for validation |

---

## Anti-Hallucination Rules

The agent is programmed with strict rules to prevent fabrication:

### Rule 1: Cite Sources
```
❌ "Nike has strong fundamentals"
✅ "According to the platform data, Nike has a fundamental score of 52/100"
```

### Rule 2: Use Actual Data
```
❌ "The stock is down about 5%"
✅ "The price data shows NKE is down 2.3% from $77.00 to $75.23"
```

### Rule 3: Acknowledge Missing Data
```
❌ "Revenue growth is 15%" (when not in data)
✅ "Revenue growth data is not available in the current dataset"
```

### Rule 4: Verify User Claims
```
User: "Why did AAPL crash?"
✅ Agent: "First, I should note that AAPL didn't crash - it's actually up 1.2% today..."
```

### Rule 5: Reference Specific Headlines
```
❌ "There's been negative news about Nike"
✅ "According to Bloomberg, 'Nike Reports Q4 Earnings: Revenue Down 17%'..."
```

---

## File Structure

```
src/
├── ai/
│   ├── ai_research_agent.py    # Main research agent
│   └── ai_research_assistant.py # (Legacy - can be removed)
├── tabs/
│   └── ai_assistant_tab.py     # Streamlit UI
└── db/
    ├── connection.py           # Database connection
    └── repository.py           # Data access methods
```

---

## Classes & Methods

### AIResearchAgent

Main agent class that orchestrates research.

```python

# Get singleton instance
agent = get_research_agent()

# Ask a question
response = agent.research("What are the best stocks to buy?")

# Quick market scan (no AI, just data)
scan = agent.quick_scan()

# Clear conversation history
agent.clear_history()
```

### DatabaseQuerier

Handles all database queries.

```python
# Key methods:
db.get_full_stock_analysis(ticker)    # Comprehensive analysis
db.get_stock_details(ticker)          # Basic scores
db.get_price_change(ticker, days)     # Price movement
db.verify_price_claim(ticker, direction)  # Fact-check
db.get_recent_news(ticker, days, limit)   # News articles
db.get_latest_headlines(limit)        # Recent headlines
db.get_sentiment_movers(days)         # Extreme sentiment stocks
db.get_top_stocks_by_score(limit)     # Top scored stocks
db.get_sector_summary()               # Sector analysis
db.get_market_overview()              # Market stats
```

### WebSearcher

Handles web searches via Tavily/Brave.

```python
# Search methods:
searcher.search(query, max_results)      # Best available API
searcher.search_tavily(query, max_results)  # Tavily specifically
searcher.search_brave(query, max_results)   # Brave specifically
```

---

## Configuration

### ResearchAgentConfig

```python
@dataclass
class ResearchAgentConfig:
    # LLM Settings
    llm_base_url: str = "http://172.23.193.91:8090/v1"
    llm_model: str = "Qwen3-32B-Q6_K.gguf"
    
    # Search APIs
    tavily_api_key: str = ""
    brave_api_key: str = ""
    finnhub_api_key: str = ""
    
    # Generation settings
    temperature: float = 0.3  # Lower = more consistent
    max_tokens: int = 4000
```

---

## Troubleshooting

### "AI not connected"
- Check if Qwen server is running at the configured URL
- Verify `LLM_QWEN_BASE_URL` in `.env`

### "Web Search: OFF"
- Set `TAVILY_API_KEY` or `BRAVE_API_KEY` in environment
- Restart the tool server

### "No news found for ticker"
- Run news collection for that ticker
- Check `news_articles` table has recent data
- Increase `days` parameter in news query

### Generic/Hallucinated Responses
- Ensure news data exists in database
- Check if queries are returning data (check logs)
- Verify database connection

---

## Example Interaction

**User:** "Why is Nike falling today?"

**Agent Response:**

> **1. SUMMARY**
> Nike is falling primarily due to disappointing Q4 earnings with a 17% revenue decline, 
> compounded by continued weakness in China sales.
> 
> **2. NEWS ANALYSIS**
> According to Bloomberg, "Nike Reports Q4 Earnings: Revenue Down 17% Year-Over-Year" - 
> This significant miss on revenue expectations is the primary driver of today's decline.
> 
> Reuters reports "Nike China Sales Slump Continues Amid Consumer Weakness" - 
> The Greater China segment, once a growth engine, continues to struggle.
> 
> However, SEC Filings show "Nike CEO Acquires 50,000 Shares at $75.20" - 
> This insider buying suggests management believes the stock is undervalued.
> 
> **3. DATA ANALYSIS**
> - Platform Score: 50/100 (NEUTRAL)
> - Fundamental Score: 52/100 - Below average
> - Technical Score: 48/100 - Weak momentum
> - Sentiment Score: 45/100 - Negative news sentiment
> - Score Trend: 📉 DECLINING (55 → 50 over 5 days)
> - Analyst Target: $95 vs Current $75.20 = 26% upside
> 
> **4. BULL VS BEAR CASE**
> 🟢 BULLISH:
> - CEO insider buying at $75.20 (Source: SEC Filings)
> - 26% upside to analyst target (Source: Price Targets)
> - Consumer Defensive sector has 4 bullish signals
> 
> 🔴 BEARISH:
> - Revenue down 17% YoY (Source: Bloomberg)
> - China sales declining (Source: Reuters)
> - Platform score declining 55→50 (Source: Score Trend)
> - Below-average fundamental score 52/100
> 
> **5. CONCLUSION & RECOMMENDATION**
> **HOLD** - The 17% revenue decline is concerning and justifies the selloff. 
> However, CEO insider buying and 26% upside to analyst targets suggest the 
> worst may be priced in. Wait for platform score to stabilize above 55 before 
> adding. Current score of 50 doesn't signal strong conviction either way.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-01-02 | Complete rewrite with fact-checking, comprehensive analysis |
| 1.5 | 2025-01-02 | Added news integration, claim verification |
| 1.0 | 2025-01-01 | Initial release with basic database queries |

---

## Author

HH Research Platform

---

## License

Proprietary - For personal use only