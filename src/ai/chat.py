import json
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from openai import OpenAI

from src.db.connection import get_connection, get_engine
from src.db.repository import Repository
from src.ai.learning import SignalLearner
from src.utils.logging import get_logger

# NEW: Import unified portfolio context
from src.ai.unified_portfolio_context import (
    get_unified_portfolio_loader,
    get_unified_context,
    get_options_flow,
    UnifiedPortfolioLoader
)

# NEW: Import market context with economic news analysis
try:
    from src.analytics.market_context import (
        MarketContextAnalyzer,
        get_market_context,
        get_market_context_for_ai
    )

    MARKET_CONTEXT_AVAILABLE = True
except ImportError:
    MARKET_CONTEXT_AVAILABLE = False

# NEW: Import macro event engine for geopolitical context
try:
    from src.analytics.macro_event_engine import (
        get_macro_engine,
        get_macro_factors,
        get_macro_context,
        MacroFactorScores,
    )

    MACRO_ENGINE_AVAILABLE = True
except ImportError:
    MACRO_ENGINE_AVAILABLE = False

logger = get_logger(__name__)

# NEW: Import alpha enhancements for signal quality improvements
try:
    from src.ml.alpha_enhancements import (
        get_enhanced_alpha_context,
        get_reliability_metrics,
        apply_forecast_shrinkage,
        get_calibrator,
        compute_decision_policy,
        ReliabilityMetrics
    )

    ALPHA_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ALPHA_ENHANCEMENTS_AVAILABLE = False

import os
from dotenv import load_dotenv

load_dotenv()



@dataclass
class ChatConfig:
    """Chat configuration - loads from .env"""
    base_url: str = os.getenv("LLM_QWEN_BASE_URL", "http://172.23.193.91:8090/v1")
    model: str = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")
    temperature: float = 0.15  # Lower for consistent, factual analysis
    max_tokens: int = 3000  # Room for detailed analysis
    top_p: float = 0.85  # Tighter nucleus sampling
    repeat_penalty: float = 1.1  # Reduce repetition


# Alternative fast config (for filtering, less accurate)
FAST_CONFIG = ChatConfig(
    base_url=os.getenv("LLM_GPT_OSS_BASE_URL", "http://172.23.193.91:8091/v1"),
    model=os.getenv("LLM_GPT_OSS_MODEL", "Qwen3-32B-Q6_K.gguf"),
    temperature=0.15,  # Lower for consistent outputs
    max_tokens=2000,
    top_p=0.85,  # Tighter nucleus sampling
    repeat_penalty=1.1
)

# Portfolio-related keywords for context detection
PORTFOLIO_KEYWORDS = [
    'portfolio', 'holdings', 'positions', 'rebalance', 'rebalancing',
    'drift', 'allocation', 'weight', 'sector', 'concentration',
    'why do i hold', 'why am i holding', 'should i sell',
    'should i buy more', 'target weight', 'actual weight',
    'failed trades', 'execution', 'trade history',
    'etf creator', 'my stocks', 'my position', 'my holding',
    'do i own', 'do i have', 'how many shares', 'what stocks',
    'underweight', 'overweight', 'not held', 'not in target'
]

# Economic news and market context keywords
ECONOMIC_KEYWORDS = [
    'news', 'economic', 'fed', 'fomc', 'rate', 'rates', 'interest rate',
    'cpi', 'inflation', 'ppi', 'gdp', 'jobs', 'employment', 'unemployment',
    'nonfarm', 'payroll', 'retail sales', 'pmi', 'manufacturing',
    'calendar', 'today', 'market', 'macro', 'economy',
    'bonds', 'treasury', 'yield', 'yields', 'bond',
    'bullish', 'bearish', 'sentiment', 'outlook',
    'what to trade', 'trade idea', 'trade ideas', 'opportunities',
    'affected', 'impact', 'effect', 'how does', 'how will',
    'sector rotation', 'risk on', 'risk off', 'vix',
    'hawkish', 'dovish', 'tightening', 'easing', 'pivot'
]

# =============================================================================
# BOND AI OUTPUT SANITIZER - Fixes corrupted number patterns
# =============================================================================
import re as _re


def _sanitize_bond_output(text: str) -> str:
    """
    Fix corrupted number patterns in bond AI output.

    Patterns fixed:
    - 89.97(+2.389.97(+2.387.93) -> $89.97 (+2.3% from $87.93)
    - 85.29–85.29–87.05 -> 85.29–87.05
    - 87.93vsentry87.93vsentry85.29 -> $87.93 vs entry $85.29
    - Put/Call ratio mentions (hallucinated)
    """
    if not text:
        return text

    out = text

    # Remove hallucinated options data
    out = _re.sub(r'(?i)put[/-]?call\s*ratio\s*[=:]\s*[\d.]+[^.\n]*[.\n]?', '', out)
    out = _re.sub(r'(?i)neutral\s*options\s*flow[^.\n]*[.\n]?', '', out)

    # Double merged target: 89.97(+2.389.97(+2.387.93) -> $89.97 (+2.3% from $87.93)
    out = _re.sub(
        r'\$?(\d+\.\d+)\s*\(\+(\d+\.\d)\s*\1\s*\(\+\2\s*(\d+\.\d+)\)',
        r'$\1 (+\2% from $\3)',
        out
    )

    # Simpler: 89.97(+2.387.93) -> $89.97 (+2.3% from $87.93)
    out = _re.sub(
        r'\$?(\d+\.\d+)\s*\(\+(\d+\.\d)(\d+\.\d+)\)',
        r'$\1 (+\2% from $\3)',
        out
    )

    # "and" pattern: (87.93)andZROZ(87.93)andZROZ(65.02)
    out = _re.sub(
        r'\(\$?(\d+\.\d+)\)\s*and\s*ZROZ\s*\(\$?\1\)\s*and\s*ZROZ\s*\(\$?(\d+\.\d+)\)',
        r'$\1 (TLT) and $\2 (ZROZ)',
        out
    )

    # vsentry pattern: 87.93vsentry87.93vsentry85.29
    out = _re.sub(
        r'\$?(\d+\.\d+)\s*vs\s*entry\s*\$?\1\s*vs\s*entry\s*\$?(\d+\.\d+)',
        r'$\1 vs entry $\2',
        out
    )

    # Entry zone duplicates (all dash types): 85.29–85.29–87.05 -> 85.29–87.05
    out = _re.sub(r'\$(\d+\.?\d*)\s*[-–—]\s*\$?\1\s*[-–—]\s*\$?(\d+\.?\d*)', r'$\1–$\2', out)
    out = _re.sub(r'(\d+\.\d+)\s*[-–—]\s*\1\s*[-–—]\s*(\d+\.\d+)', r'\1–\2', out)
    out = _re.sub(r'(\d+)\s*[-–—]\s*\1\s*[-–—]\s*(\d+)', r'\1–\2', out)

    # vs duplications
    out = _re.sub(r'(\d+\.\d+)\s*vs\s*\1\s*vs\s*(\d+\.\d+)', r'\1 vs \2', out)

    # Remove asterisks around prices
    out = _re.sub(r'\*+(\$\d+\.?\d*)\*+', r'\1', out)
    out = _re.sub(r'(\$\d+\.?\d*)\*+', r'\1', out)

    return out


class AlphaChat:
    """
    AI Chat interface for discussing stocks and trading signals.

    Provides Qwen with UNIFIED portfolio context combining:
    - IBKR Live data (actual positions, values)
    - ETF Creator (target weights, selection reasoning)
    - Rebalancer (trade execution status)
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.repo = Repository()
        self.learner = SignalLearner()

        # Portfolio loader
        self.portfolio_loader = UnifiedPortfolioLoader()

        # Cache for IBKR data (updated when portfolio questions asked)
        self._ibkr_positions = None
        self._ibkr_summary = None

        # Initialize OpenAI client for Qwen
        try:
            self.client = OpenAI(
                base_url=self.config.base_url,
                api_key="not-needed",
                timeout=300
            )
            self.available = True
            logger.info(f"AlphaChat connected to {self.config.base_url}")
        except Exception as e:
            logger.error(f"AlphaChat initialization failed: {e}")
            self.available = False
            self.client = None

        # Conversation history
        self.history: List[Dict[str, str]] = []

        # chat.py (inside class AlphaChat) — ADD this helper method anywhere above `chat()`.

    def _should_web_search(
            self,
            message: str,
            *,
            needs_search: bool,
            is_general: bool,
            is_stock_question: bool,
            needs_portfolio: bool,
            needs_economic: bool,
            ticker: Optional[str],
    ) -> bool:
        """
        Decide when to call _search_web().

        Goal: allow portfolio/stock/economic questions to use web search when it helps,
        while avoiding pointless searches for purely internal/gating questions.
        """
        m = (message or "").strip()
        ml = m.lower()

        # User-forced web search: "/web ..." or "web: ..."
        if ml.startswith("/web") or ml.startswith("web:"):
            return True

        # If explicitly asking to browse/search the web
        if any(x in ml for x in ["search the web", "browse the web", "look it up online", "internet"]):
            return True

        # General "lookup" or "latest" intent
        if needs_search or is_general:
            return True

        # Avoid web search for purely internal gates unless user forced it
        internal_only = [
            "hard gate", "hard gates", "soft gate", "gates", "blocked", "turnover cap",
            "cash_min", "planner warnings", "missing price", "missing prices",
            "order sizing", "trade count capped",
        ]
        if any(x in ml for x in internal_only):
            return False

        # Finance/portfolio questions: allow search when it is likely useful
        finance_intent = [
            "why", "should i", "is this good", "good stock", "risk", "catalyst", "news",
            "earnings", "guidance", "downgrade", "upgrade", "valuation", "outlook",
            "macro", "rates", "fed", "inflation", "cpi", "ppi", "gdp",
            "dividend", "buy", "sell", "hold",
        ]

        if (is_stock_question or needs_portfolio or needs_economic) and (
                ticker or any(x in ml for x in finance_intent)):
            return True

        return False

    def set_ibkr_data(self, positions: List[Dict], account_summary: Dict = None):
        """
        Set IBKR data for portfolio context.
        Call this when portfolio data is available (e.g., from Portfolio Tab).

        Args:
            positions: List of position dicts from IBKR
            account_summary: Account summary dict from IBKR
        """
        self._ibkr_positions = positions
        self._ibkr_summary = account_summary

        # Reload unified portfolio with new data
        self.portfolio_loader.load_unified_portfolio(positions, account_summary)
        logger.info(f"IBKR data set: {len(positions)} positions")

    def _load_ibkr_data(self) -> bool:
        """
        Load IBKR data if not already loaded.
        Returns True if data is available.
        """
        if self._ibkr_positions is not None:
            return True

        # Try to load from IBKR
        try:
            from src.broker.ibkr_utils import load_ibkr_data_cached

            ibkr_data = load_ibkr_data_cached(
                account_id="",
                host="127.0.0.1",
                port=7496,
                fetch_live_prices=False  # Use cached if market closed
            )

            if ibkr_data and not ibkr_data.get('error'):
                self._ibkr_positions = ibkr_data.get('positions', [])
                self._ibkr_summary = ibkr_data.get('summary', {})

                # Load unified portfolio
                self.portfolio_loader.load_unified_portfolio(
                    self._ibkr_positions,
                    self._ibkr_summary
                )
                logger.info(f"Loaded IBKR data: {len(self._ibkr_positions)} positions")
                return True
            else:
                logger.warning(f"IBKR data not available: {ibkr_data.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Failed to load IBKR data: {e}")
            return False

    def generate_daily_briefing(self, positions: List[Dict], account_summary: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate daily briefing with risk analysis for AI Chat tab.
        Also updates IBKR data cache for portfolio context.
        """
        from src.ai.risk_analyzer import RiskAnalyzer

        # Update IBKR cache
        self.set_ibkr_data(positions, account_summary)

        analyzer = RiskAnalyzer()
        metrics = analyzer.analyze_portfolio(positions, account_summary)
        alerts = analyzer.get_all_alerts(metrics)
        questions = analyzer.generate_daily_questions(metrics, positions)
        summary_text = analyzer.generate_risk_summary(metrics)

        return {
            'metrics': metrics,
            'alerts': alerts,
            'questions': questions,
            'summary_text': summary_text,
            'portfolio_value': account_summary.get('net_liquidation', 0) if account_summary else 0,
            'position_count': len(positions)
        }

    def _search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web using the tools API."""
        import requests

        logger.info(f"Searching web for: {query}")

        try:
            response = requests.post(
                f"{os.getenv('TOOL_SERVER_URL', 'http://172.23.193.91:7001')}/search",
                json={"query": query, "max_results": max_results},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                if results:
                    search_text = "WEB SEARCH RESULTS:\n"
                    for i, r in enumerate(results[:5], 1):
                        title = r.get('title', 'No title')
                        snippet = r.get('snippet', r.get('description', r.get('url', '')))[:200]
                        search_text += f"{i}. {title}: {snippet}\n"
                    return search_text
            return ""
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ""

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def _prioritize_context(self, context: str, question: str) -> str:
        """
        Prioritize most relevant context sections based on the question.
        Reduces noise and focuses AI on relevant data.
        """
        if not context or len(context) < 500:
            return context  # Too short to prioritize

        question_lower = question.lower()
        sections = context.split('\n\n')

        if len(sections) <= 3:
            return context  # Too few sections

        # Score each section by relevance to question
        scored = []
        for section in sections:
            if not section.strip():
                continue

            score = 5  # Base score
            section_lower = section.lower()

            # High priority - Alpha Model
            if 'alpha model' in section_lower or 'ml prediction' in section_lower:
                score += 15 if any(kw in question_lower for kw in ['predict', 'ml', 'model', 'alpha', 'expect']) else 8

            # High priority - Signals
            if 'signal' in section_lower and 'component' not in section_lower:
                score += 12 if any(kw in question_lower for kw in ['buy', 'sell', 'signal', 'recommend']) else 6

            # Options flow
            if 'option' in section_lower or 'put/call' in section_lower:
                score += 10 if any(kw in question_lower for kw in ['option', 'flow', 'call', 'put']) else 5

            # Short squeeze
            if 'squeeze' in section_lower or 'short interest' in section_lower:
                score += 12 if any(kw in question_lower for kw in ['squeeze', 'short']) else 4

            # Earnings
            if 'earning' in section_lower or 'ies' in section_lower or 'ecs' in section_lower:
                score += 12 if any(kw in question_lower for kw in ['earning', 'eps', 'quarter', 'report']) else 5

            # Fundamentals
            if 'fundamental' in section_lower or 'p/e' in section_lower or 'market cap' in section_lower:
                score += 10 if any(
                    kw in question_lower for kw in ['value', 'pe', 'fundamental', 'cheap', 'expensive']) else 5

            # News
            if 'news' in section_lower or 'headline' in section_lower:
                score += 10 if any(kw in question_lower for kw in ['news', 'catalyst', 'headline', 'recent']) else 4

            # MACRO/GEOPOLITICAL - Always important for stock analysis
            if 'macro' in section_lower or 'geopolitical' in section_lower or 'opec' in section_lower:
                score += 15 if any(kw in question_lower for kw in
                                   ['macro', 'geopolitical', 'oil', 'war', 'conflict', 'political']) else 10

            # Energy/commodity factors - boost for energy stocks
            if 'oil supply' in section_lower or 'energy disruption' in section_lower or 'tailwind' in section_lower:
                score += 12

            # Portfolio
            if 'portfolio' in section_lower or 'actual' in section_lower or 'ibkr' in section_lower:
                score += 15 if any(
                    kw in question_lower for kw in ['own', 'hold', 'portfolio', 'position', 'weight']) else 3

            # Always keep basic info sections
            if section_lower.startswith('===') or 'current price' in section_lower:
                score += 20  # Always keep header and price

            scored.append((score, section))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top sections (up to 80% or until we hit a low-score section)
        keep_count = max(3, int(len(scored) * 0.8))
        top_sections = [s[1] for s in scored[:keep_count]]

        return '\n\n'.join(top_sections)

    def _get_system_prompt(self) -> str:
        """Build comprehensive system prompt for trading analysis."""
        return """You are a Senior Quantitative Analyst at a hedge fund with 15+ years of experience in equity research, portfolio management, and algorithmic trading. You have access to the Alpha Research Platform which provides real-time data, signals, and ML predictions.

## YOUR EXPERTISE
- Deep knowledge of technical analysis, fundamentals, and market microstructure
- Expert in options flow analysis and institutional positioning
- Skilled at interpreting conflicting signals and quantifying uncertainty
- Experience with factor investing, momentum strategies, and mean reversion

## SIGNAL INTERPRETATION RULES

### Platform Scores (0-100)
- 0-30: BEARISH (Strong sell pressure, avoid or short)
- 31-45: CAUTIOUS (Lean negative, reduce exposure)
- 46-55: NEUTRAL (No clear direction, hold current position)
- 56-70: BULLISH (Lean positive, consider adding)
- 71-100: VERY BULLISH (Strong buy pressure, high conviction long)

### Alpha Model (ML Predictions)
- STRONG_BUY: Expected return > +3%, high conviction
- BUY: Expected return +1% to +3%
- HOLD: Expected return -1% to +1%
- SELL: Expected return -1% to -3%
- STRONG_SELL: Expected return < -3%, high conviction

### SKEPTICISM FOR EXTREME PREDICTIONS (Critical)
When Alpha Model shows extreme values, be skeptical:
- Expected return > +5% in 5 days: VERY RARE, likely model overconfidence
- Sharpe > 5: Unusually high, reduce trust in prediction
- Conviction > 85%: Model may be overfit, verify with other signals
- P(Positive) > 90%: Almost nothing is 90% certain in markets

When you see extreme predictions:
1. MENTION the prediction seems unusually optimistic/pessimistic
2. WEIGHT other signals more heavily (platform signals, fundamentals)
3. REDUCE position size recommendation
4. SET tighter stop losses

### When Signals Conflict (CRITICAL)
When Platform signals and Alpha Model disagree:
1. State the conflict explicitly to the user
2. Explain WHY each signal might be correct
3. Consider time horizons (Platform = immediate sentiment, ML = 5-20 day prediction)
4. Weigh momentum vs mean reversion
5. Default to LOWER conviction and SMALLER position size
6. Suggest waiting for confirmation if conflict is severe

### Options Flow Interpretation
- Put/Call Ratio < 0.5: Very bullish (unusual call buying, possible smart money)
- Put/Call Ratio 0.5-0.8: Moderately bullish
- Put/Call Ratio 0.8-1.2: Neutral
- Put/Call Ratio > 1.2: Bearish (hedging activity or bearish bets)
- Unusual Activity Score > 70: Significant institutional activity detected

### Short Squeeze Signals
- Score > 70: HIGH squeeze potential - momentum trade opportunity
- Score 50-70: ELEVATED - watch for catalysts
- Score < 50: LOW probability - don't trade for squeeze

### Earnings Context (IES Score)
- IES > 75: HYPED - very high expectations, needs blowout to move up, RISKY
- IES 50-75: MODERATE expectations
- IES < 35: FEARED - low bar to clear, potential upside surprise, OPPORTUNITY
- Within 7 days of earnings: Reduce position size 50% or hedge with options

## POSITION SIZING RULES
- High conviction (signals aligned, >80%): Full position (1x)
- Medium conviction (60-80% or minor conflict): Half position (0.5x)
- Low conviction (<60% or major conflict): Quarter position (0.25x) or WAIT
- NEVER recommend >5% of portfolio in any single stock
- NEVER recommend all-in on any position

## RESPONSE FORMAT

For stock analysis, ALWAYS structure your response as:

📊 **SIGNAL SUMMARY**
- Platform Signal: [BUY/HOLD/SELL] ([score]%)
- Alpha Model: [signal] ([expected return], [conviction]%)
- Consensus: [ALIGNED ✅ / CONFLICTING ⚠️]

📈 **KEY METRICS**
- [List 3-5 most relevant data points from context]

⚖️ **BULL vs BEAR CASE**
- Bulls: [2-3 key bullish factors]
- Bears: [2-3 key bearish factors]

🎯 **RECOMMENDATION**
- Action: [BUY / HOLD / SELL / WAIT FOR PULLBACK / AVOID]
- Position Size: [0.25x / 0.5x / 1x] based on conviction
- Entry Zone: $[price range]
- Stop Loss: $[price] ([X]% below entry)
- Target: $[price] ([X]% upside)
- Time Horizon: [days/weeks/months]
- Confidence: [LOW / MEDIUM / HIGH]

⚠️ **KEY RISKS**
- [List 2-3 specific risks to monitor]

## CHAIN OF THOUGHT

Before answering complex questions, think through:
1. What are the KEY data points from the context?
2. What do BULLISH factors suggest?
3. What do BEARISH factors suggest?
4. Where do signals CONFLICT and why?
5. What's the RISK-ADJUSTED recommendation?

## EVIDENCE RULES (Critical - Prevents Hallucination)

You MUST follow these rules strictly:

1. **NUMBERS USED**: Every specific number in your response (prices, returns, percentages) 
   MUST come from the context provided. Do NOT invent statistics.

2. **MISSING DATA**: If data you need is not in context, explicitly state:
   "Data not available: [what's missing]"
   Do NOT guess or use generic values.

3. **SOURCES**: When citing a metric, be clear about the source:
   - "Platform Score shows..." (from unified signal)
   - "Alpha Model predicts..." (from ML)
   - "Options flow indicates..." (from options data)

4. **UNCERTAINTY**: Always acknowledge uncertainty:
   - Use "suggests" not "proves"
   - Use "approximately" not exact numbers you're unsure of
   - Use "based on available data" to qualify predictions

5. **AUDIT CHECK**: Before finalizing, verify:
   - Every % return mentioned appears in context
   - Every price target is derived from context data
   - Every conviction level matches what Alpha Model provided
   - No invented "typical", "usually", or "historically" claims without data

## DECISION POLICY COMPLIANCE (When Present)

If context includes a "🎯 DECISION POLICY (BINDING)" section:
- You MUST follow the action and size cap specified
- You MAY NOT upgrade the action (e.g., HOLD→BUY)
- You MAY NOT exceed the size cap
- You MAY downgrade (BUY→HOLD) or reduce size if you see additional risks
- You MAY suggest entry timing within allowed entry styles
- You MUST mention any blocking factors listed

## PORTFOLIO CONTEXT
- "ACTUAL" = Current IBKR holdings (what you REALLY own)
- "TARGET" = ETF Creator recommended weights
- "DRIFT" = Deviation from target (triggers rebalancing)
- If ACTUAL shows 0 shares, you DON'T own it

## MARKET CONTEXT
- VIX < 15: Low fear, risk-on environment
- VIX 15-20: Normal volatility
- VIX 20-30: Elevated fear, caution warranted
- VIX > 30: High fear, potential capitulation/opportunity

## REGIME ADJUSTMENTS (Critical for Position Sizing)
When market conditions are stressed, REDUCE risk:
- BEAR regime (SPY below 200 DMA): Cut position sizes by 50%
- HIGH VOLATILITY (VIX > 25): Cut position sizes by 50%, use wider stops
- BOTH conditions: Cut position sizes by 75%, only highest conviction trades
- BULL regime (SPY > 200 DMA, VIX < 20): Normal position sizing

**IMPORTANT**: If Alpha Model shows "Market Regime: BEAR", you MUST:
1. Mention this in your analysis
2. Reduce recommended position size
3. Suggest tighter stops or waiting for confirmation

## PORTFOLIO CORRELATION AWARENESS
When recommending a stock, consider existing portfolio exposure:
- If user already owns 3+ stocks in same sector → WARN about concentration
- If adding would make sector weight > 25% → RECOMMEND smaller position
- If stock correlates highly with existing holdings → MENTION diversification concern

## STOP LOSS GUIDELINES
Don't use arbitrary percentages. Consider:
- Volatile stocks (beta > 1.5): Wider stops (8-10%)
- Stable stocks (beta < 1): Tighter stops (4-6%)
- Near support level: Stop just below support
- After big run-up: Tighter trailing stop to protect gains

## FORBIDDEN ACTIONS
- NEVER invent data not provided in the context
- NEVER give recommendations without mentioning risks
- NEVER ignore conflicting signals - always address them
- NEVER recommend position sizes > 5% of portfolio
- NEVER be overconfident - markets are inherently uncertain
- NEVER say "you should definitely" - say "the data suggests"

## EXAMPLE GOOD RESPONSES

Example 1 - Conflicting Signals:
"MRK shows CONFLICTING signals. The Platform is bullish (71%) driven by strong options flow (P/C 0.15) and sentiment (78), but the Alpha Model predicts -2.51% over 5 days with 68% conviction.

This divergence suggests short-term overbought conditions (RSI 73) despite positive momentum. The ML model may be detecting mean-reversion risk after the recent +5.6% run.

📊 **SIGNAL SUMMARY**
- Platform Signal: BUY (71%)
- Alpha Model: STRONG_SELL (-2.51%, 68%)
- Consensus: CONFLICTING ⚠️

🎯 **RECOMMENDATION**
- Action: WAIT FOR PULLBACK
- Position Size: 0.5x when entry reached
- Entry Zone: $103-104 (support zone)
- Stop Loss: $100 (5% below entry)
- Target: $114 (10% upside)
- Time Horizon: 2-4 weeks
- Confidence: MEDIUM

⚠️ **KEY RISKS**
- RSI overbought at 73, near-term pullback likely
- If wrong, could miss continued momentum"

Example 2 - Aligned Signals:
"NVDA shows ALIGNED bullish signals across all indicators.

📊 **SIGNAL SUMMARY**
- Platform Signal: BUY (82%)
- Alpha Model: STRONG_BUY (+5.2%, 85%)
- Consensus: ALIGNED ✅

🎯 **RECOMMENDATION**
- Action: BUY
- Position Size: 1x (full position, signals aligned)
- Entry: Current price or limit 2% below
- Stop Loss: 8% below entry
- Target: +15% (3-month horizon)
- Confidence: HIGH"

## DECISION POLICY COMPLIANCE (CRITICAL)

If a "DECISION POLICY" block appears in the context, you MUST follow it exactly:
1. The "Trade Allowed" field is BINDING - do NOT recommend trades if it says NO
2. The "Size Cap" is the MAXIMUM - never recommend larger
3. The "Action" is the allowed action - do NOT upgrade (e.g., if it says WAIT, don't say BUY)
4. The "Required Triggers" must be mentioned if present

You may EXPLAIN the decision and add nuance, but you cannot OVERRIDE it.

## EVIDENCE RULES (Prevents Hallucination)

1. Every numeric claim MUST appear in the provided context
2. If a metric is not provided, say "Not available" - do NOT invent it
3. Include an "Evidence Used" section listing key data points from context
4. If context shows "SHRINKAGE APPLIED", use the SHRUNK prediction, not raw
5. If "ML RELIABILITY" shows UNRELIABLE, explicitly state ML predictions are low-confidence

## DATA FRESHNESS

Pay attention to timestamps in the data. If data seems stale:
- Mention it: "Note: This analysis uses data from [date]"
- Be more cautious with recommendations

You are analytical, precise, data-driven, and always honest about uncertainty. You prioritize risk management over return chasing."""

    def _get_unified_portfolio_context(self, ticker: str = None) -> str:
        """
        Get unified portfolio context combining IBKR, ETF Creator, and Rebalancer.

        Args:
            ticker: Optional specific ticker

        Returns:
            Formatted unified context string
        """
        try:
            # Load IBKR data if needed
            self._load_ibkr_data()

            # Get unified context
            if ticker:
                # Specific holding
                full_context = self.portfolio_loader.get_unified_context(
                    self._ibkr_positions,
                    self._ibkr_summary,
                    max_holdings=10
                )
                holding_context = self.portfolio_loader.get_holding_context(ticker.upper())
                return f"{holding_context}\n\n{full_context}"
            else:
                # Full portfolio
                return self.portfolio_loader.get_unified_context(
                    self._ibkr_positions,
                    self._ibkr_summary,
                    max_holdings=20
                )

        except Exception as e:
            logger.error(f"Error getting unified portfolio context: {e}")
            return ""

    def _get_ticker_context(self, ticker: str) -> str:
        """Get context for a specific ticker (scores, signals, news)."""
        context_parts = []
        import pandas as pd

        # 1. Current scores
        try:
            query = """
                    SELECT ticker, date, sentiment_score, fundamental_score, total_score, article_count
                    FROM screener_scores
                    WHERE ticker = %(ticker)s
                    ORDER BY date DESC LIMIT 1
                    """
            df = pd.read_sql(query, get_engine(), params={"ticker": ticker})

            if not df.empty:
                row = df.iloc[0]
                context_parts.append(
                    f"{ticker} SCORES ({row['date']}): "
                    f"Sentiment={row['sentiment_score']}, "
                    f"Fundamental={row['fundamental_score']}, "
                    f"Total={row['total_score']}, "
                    f"Articles={row['article_count']}"
                )
        except Exception as e:
            logger.debug(f"Error getting scores: {e}")

        # 2. Current signal
        try:
            query = """
                    SELECT signal_type, signal_strength
                    FROM trading_signals
                    WHERE ticker = %(ticker)s
                    ORDER BY date DESC LIMIT 1
                    """
            df = pd.read_sql(query, get_engine(), params={"ticker": ticker})

            if not df.empty:
                row = df.iloc[0]
                context_parts.append(
                    f"{ticker} SIGNAL: {row['signal_type']} (strength={row['signal_strength']})"
                )
        except Exception as e:
            logger.debug(f"Error getting signal: {e}")

        # 3. Fundamentals
        try:
            query = """
                    SELECT sector, pe_ratio, revenue_growth, profit_margin
                    FROM fundamentals
                    WHERE ticker = %(ticker)s
                    ORDER BY date DESC LIMIT 1
                    """
            df = pd.read_sql(query, get_engine(), params={"ticker": ticker})

            if not df.empty:
                row = df.iloc[0]
                context_parts.append(
                    f"{ticker} FUNDAMENTALS: "
                    f"Sector={row['sector']}, "
                    f"P/E={row['pe_ratio']}, "
                    f"RevGrowth={row['revenue_growth']}, "
                    f"Margin={row['profit_margin']}"
                )
        except Exception as e:
            logger.debug(f"Error getting fundamentals: {e}")

        # 4. Committee decision
        try:
            query = """
                    SELECT verdict, conviction
                    FROM committee_decisions
                    WHERE ticker = %(ticker)s
                    ORDER BY date DESC LIMIT 1
                    """
            df = pd.read_sql(query, get_engine(), params={"ticker": ticker})

            if not df.empty:
                row = df.iloc[0]
                context_parts.append(f"{ticker} COMMITTEE: {row['verdict']} ({row['conviction']}% conviction)")
        except Exception as e:
            logger.debug(f"Error getting committee: {e}")

        # 5. Trade Journal
        try:
            journal_entry = self.repo.get_journal_entry_for_ticker(ticker)
            if journal_entry:
                journal_context = f"\n{ticker} TRADE JOURNAL:\n"
                journal_context += f"  Action: {journal_entry.get('action')} on {journal_entry.get('entry_date')}\n"
                if journal_entry.get('thesis'):
                    journal_context += f"  Thesis: {journal_entry.get('thesis')}\n"
                context_parts.append(journal_context)
        except Exception as e:
            logger.debug(f"Error getting journal entry: {e}")

        # 6. Options Flow (for market sentiment)
        try:
            options_context = get_options_flow(ticker)
            if options_context and "not available" not in options_context.lower():
                context_parts.append(options_context)
        except Exception as e:
            logger.debug(f"Error getting options flow: {e}")

        # 7. Short Squeeze Analysis
        try:
            from src.analytics.short_squeeze import get_squeeze_report
            squeeze_context = get_squeeze_report(ticker)
            if squeeze_context:
                context_parts.append(squeeze_context)
        except Exception as e:
            logger.debug(f"Error getting short squeeze data: {e}")

        # 8. Earnings Analysis
        try:
            from src.analytics.earnings_intelligence import get_earnings_for_ai
            earnings_context = get_earnings_for_ai(ticker)
            if earnings_context:
                context_parts.append(earnings_context)
        except Exception as e:
            logger.debug(f"Error getting earnings data: {e}")

        # 9. Alpha Model Predictions (ML-based forward returns)
        try:
            alpha_context = self._get_alpha_model_context(ticker)
            if alpha_context:
                context_parts.append(alpha_context)
        except Exception as e:
            logger.debug(f"Error getting alpha model predictions: {e}")

        # 10. UnifiedSignal (comprehensive signal from signal engine)
        try:
            signal_context = self._get_unified_signal_context(ticker)
            if signal_context:
                context_parts.append(signal_context)
        except Exception as e:
            logger.debug(f"Error getting unified signal: {e}")

        # 11. Portfolio Sector Exposure (correlation warnings)
        try:
            exposure_context = self._get_portfolio_sector_exposure(ticker)
            if exposure_context:
                context_parts.append(exposure_context)
        except Exception as e:
            logger.debug(f"Error getting portfolio exposure: {e}")

        # 12. Alpha Model Accuracy (calibration context)
        try:
            accuracy_context = self._get_prediction_accuracy_context(ticker)
            if accuracy_context:
                context_parts.append(accuracy_context)
        except Exception as e:
            logger.debug(f"Error getting prediction accuracy: {e}")

        # 13. Macro/Geopolitical Impact (NEW)
        try:
            macro_impact = self._get_macro_impact_for_ticker(ticker)
            if macro_impact:
                context_parts.append(macro_impact)
        except Exception as e:
            logger.debug(f"Error getting macro impact: {e}")

        return "\n".join(context_parts) if context_parts else ""

    def _get_platform_context(self) -> str:
        """Get platform context (backtest insights, signals)."""
        context_parts = []
        import pandas as pd

        # Best backtest strategy
        try:
            query = """
                    SELECT strategy_name, holding_period, sharpe_ratio, win_rate
                    FROM backtest_results
                    WHERE total_trades >= 50
                    ORDER BY sharpe_ratio DESC LIMIT 1
                    """
            df = pd.read_sql(query, get_engine())

            if not df.empty:
                row = df.iloc[0]
                context_parts.append(
                    f"BEST STRATEGY: {row['strategy_name']} "
                    f"({row['holding_period']}d hold, Sharpe={row['sharpe_ratio']:.2f}, "
                    f"WinRate={row['win_rate']:.1%})"
                )
        except Exception as e:
            logger.debug(f"Error getting backtest: {e}")

        # Top signals
        try:
            query = """
                    SELECT signal_type,
                           ROUND(AVG(return_10d)::numeric, 2) as avg_return,
                           COUNT(*) as count
                    FROM historical_scores
                    WHERE return_10d IS NOT NULL
                    GROUP BY signal_type
                    ORDER BY avg_return DESC LIMIT 3
                    """
            df = pd.read_sql(query, get_engine())

            if not df.empty:
                signals = [f"{row['signal_type']}={row['avg_return']}%" for _, row in df.iterrows()]
                context_parts.append("TOP SIGNALS: " + ", ".join(signals))
        except Exception as e:
            logger.debug(f"Error getting signals: {e}")

        return "\n".join(context_parts)

    def _get_portfolio_sector_exposure(self, ticker: str = None) -> str:
        """
        Get portfolio sector exposure for correlation warnings.
        Returns context about existing sector concentration.
        """
        try:
            if not self._ibkr_positions:
                self._load_ibkr_data()

            if not self._ibkr_positions:
                return ""

            # Count positions by sector
            from collections import defaultdict
            sector_counts = defaultdict(list)
            total_value = 0

            for pos in self._ibkr_positions:
                sector = pos.get('sector', 'Unknown')
                symbol = pos.get('symbol', '')
                value = pos.get('market_value', 0) or 0
                sector_counts[sector].append(symbol)
                total_value += value

            # Get the sector of the ticker being analyzed
            ticker_sector = None
            if ticker:
                try:
                    import yfinance as yf
                    info = yf.Ticker(ticker).info
                    ticker_sector = info.get('sector', 'Unknown')
                except:
                    pass

            # Build context
            context_parts = []

            if ticker_sector and ticker_sector in sector_counts:
                same_sector_stocks = sector_counts[ticker_sector]
                if len(same_sector_stocks) >= 3:
                    context_parts.append(
                        f"⚠️ CONCENTRATION WARNING: You already own {len(same_sector_stocks)} "
                        f"{ticker_sector} stocks: {', '.join(same_sector_stocks[:5])}"
                    )

            # Calculate sector weights
            if total_value > 0:
                sector_weights = {}
                for pos in self._ibkr_positions:
                    sector = pos.get('sector', 'Unknown')
                    value = pos.get('market_value', 0) or 0
                    sector_weights[sector] = sector_weights.get(sector, 0) + value

                # Find overweight sectors (>25%)
                overweight = [(s, v / total_value) for s, v in sector_weights.items() if v / total_value > 0.25]
                if overweight:
                    for sector, weight in overweight:
                        context_parts.append(f"⚠️ OVERWEIGHT: {sector} at {weight:.1%} of portfolio")

            if context_parts:
                return "\nPORTFOLIO EXPOSURE:\n" + "\n".join(context_parts)
            return ""

        except Exception as e:
            logger.debug(f"Error getting sector exposure: {e}")
            return ""

    def _needs_portfolio_context(self, message: str) -> bool:
        """Check if message needs portfolio context."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in PORTFOLIO_KEYWORDS)

    def _get_prediction_accuracy_context(self, ticker: str = None) -> str:
        """
        Get Alpha Model prediction accuracy context for calibration.
        Helps AI understand how reliable its predictions have been.
        """
        try:
            from src.ml.prediction_tracker import get_accuracy_for_ai
            return get_accuracy_for_ai(ticker)
        except ImportError:
            logger.debug("Prediction tracker not available")
            return ""
        except Exception as e:
            logger.debug(f"Error getting prediction accuracy: {e}")
            return ""

    def _needs_economic_context(self, message: str) -> bool:
        """Check if message needs economic/market context."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in ECONOMIC_KEYWORDS)

    def _get_economic_context(self, run_ai_analysis: bool = True) -> str:
        """
        Get market context with economic news analysis.

        Args:
            run_ai_analysis: If True, runs AI analysis (slower but more insightful)

        Returns:
            Formatted market context string
        """
        if not MARKET_CONTEXT_AVAILABLE:
            return ""

        try:
            # Get full market context with AI analysis
            context = get_market_context_for_ai(run_ai_analysis=run_ai_analysis)
            return context
        except Exception as e:
            logger.error(f"Error getting economic context: {e}")
            return ""

    def _get_macro_context(self, ticker: str = None) -> str:
        """Get macro/geopolitical context for AI analysis."""
        if not MACRO_ENGINE_AVAILABLE:
            return ""

        try:
            # Refresh the engine to get latest events
            engine = get_macro_engine()
            engine.refresh()

            # Get portfolio tickers if we have them
            portfolio_tickers = []
            if hasattr(self, 'portfolio_loader') and self.portfolio_loader:
                try:
                    positions = self.portfolio_loader.get_positions()
                    if positions:
                        portfolio_tickers = [p.get('symbol', p.get('ticker', '')) for p in positions if p]
                except:
                    pass

            # Add the ticker being analyzed
            if ticker and ticker not in portfolio_tickers:
                portfolio_tickers.append(ticker)

            # Get macro context
            if portfolio_tickers:
                context = get_macro_context(portfolio_tickers)
            else:
                context = get_macro_context()

            return context

        except Exception as e:
            logger.debug(f"Error getting macro context: {e}")
            return ""

    def _get_macro_impact_for_ticker(self, ticker: str) -> str:
        """Get macro headwind/tailwind analysis for a specific ticker."""
        if not MACRO_ENGINE_AVAILABLE:
            return ""

        try:
            engine = get_macro_engine()
            exposure = engine.compute_portfolio_exposure([ticker])

            impact = exposure.ticker_impacts.get(ticker, 0)
            factors = engine.factors
            elevated = factors.get_elevated_factors(threshold=60)

            if abs(impact) < 1:
                return f"MACRO IMPACT for {ticker}: Neutral (score: {impact:+.1f})"
            elif impact > 0:
                factor_str = ", ".join(
                    [f.replace('_', ' ') for f, v in elevated[:2]]) if elevated else "current environment"
                return f"MACRO TAILWIND for {ticker}: +{impact:.1f} - Benefits from {factor_str}"
            else:
                factor_str = ", ".join(
                    [f.replace('_', ' ') for f, v in elevated[:2]]) if elevated else "current environment"
                return f"MACRO HEADWIND for {ticker}: {impact:.1f} - Pressured by {factor_str}"

        except Exception as e:
            logger.debug(f"Error getting macro impact: {e}")
            return ""

    def _needs_macro_context(self, message: str) -> bool:
        """Check if message needs macro/geopolitical context."""
        macro_keywords = [
            'macro', 'geopolitical', 'political', 'war', 'conflict', 'sanction',
            'tariff', 'trade war', 'oil', 'inflation', 'recession', 'fed',
            'interest rate', 'election', 'crisis', 'opec', 'commodity',
            'global', 'economy', 'economic'
        ]
        message_lower = message.lower()
        return any(kw in message_lower for kw in macro_keywords)

    def _get_alpha_model_context(self, ticker: str) -> str:
        """
        Get Alpha Model ML predictions with all enhancements applied:
        - Forecast shrinkage (context-aware)
        - ML reliability gate (vol-scaled bias, EWMA accuracy)
        - Calibrated probabilities (smoothed by sample size)
        - Decision policy (binding rules for LLM)

        Also saves predictions to database for reliability tracking.
        """
        try:
            from src.ml.multi_factor_alpha import MultiFactorAlphaModel
            import os

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models", "multi_factor_alpha.pkl"
            )

            if not os.path.exists(model_path):
                return ""

            model = MultiFactorAlphaModel()
            model.load_model(model_path)

            # Generate prediction
            prediction = model.predict_single(ticker)

            if not prediction:
                return ""

            # Save prediction to database for reliability tracking
            self._save_alpha_prediction(ticker, prediction)

            # Get platform signals for decision policy
            # CRITICAL: Don't silently replace missing data with neutral defaults
            platform_score = None
            platform_signal = None
            technical_score = None
            platform_data_available = False

            try:
                query = """
                        SELECT total_score, signal_type, technical_score
                        FROM latest_scores \
                        WHERE ticker = %s \
                        """
                df = pd.read_sql(query, get_engine(), params=(ticker,))
                if not df.empty:
                    row = df.iloc[0]
                    # Check for actual values, not just non-empty
                    if pd.notna(row['total_score']) and row['total_score'] != 0:
                        platform_score = float(row['total_score'])
                    if pd.notna(row['signal_type']) and row['signal_type']:
                        platform_signal = str(row['signal_type'])
                    if pd.notna(row['technical_score']) and row['technical_score'] != 0:
                        technical_score = float(row['technical_score'])

                    # Only mark as available if we have the critical fields
                    platform_data_available = platform_score is not None and platform_signal is not None
            except Exception as e:
                logger.debug(f"Could not get platform scores: {e}")

            # If platform data missing, use conservative defaults but flag it
            if platform_score is None:
                platform_score = 50
                logger.warning(f"Platform score missing for {ticker}, using neutral default")
            if platform_signal is None:
                platform_signal = "HOLD"
                logger.warning(f"Platform signal missing for {ticker}, using HOLD default")
            if technical_score is None:
                technical_score = 50

            # Use new enhancement module if available
            if ALPHA_ENHANCEMENTS_AVAILABLE:
                try:
                    from src.ml.alpha_enhancements import build_enhanced_alpha_context
                    context = build_enhanced_alpha_context(
                        ticker=ticker,
                        alpha_prediction=prediction,
                        platform_score=platform_score,
                        platform_signal=platform_signal,
                        technical_score=technical_score
                    )
                    # Add warning if platform data was missing
                    if not platform_data_available:
                        context = f"⚠️ WARNING: Platform scores unavailable for {ticker} - using conservative defaults\n\n" + context
                    return context
                except Exception as e:
                    logger.debug(f"Enhancement module error, using fallback: {e}")

            # Fallback: basic format without enhancements
            context = f"""
🧠 ALPHA MODEL PREDICTION: {ticker}
{'=' * 50}
ML Signal: {prediction.get('signal', 'N/A')} (Strength: {prediction.get('conviction', 0):.0%})

Expected Returns:
  5-day:  {prediction.get('expected_return_5d', 0):+.2%}
  10-day: {prediction.get('expected_return_10d', 0):+.2%}

Probabilities:
  P(Positive 5d): {prediction.get('prob_positive_5d', 0):.1%}

Market Regime: {prediction.get('regime', 'UNKNOWN')}

Top Bullish Factors:
"""
            for factor, contrib in prediction.get('top_bullish_factors', [])[:3]:
                context += f"  ✅ {factor}: +{contrib:.3f}\n"

            context += "\nTop Bearish Factors:\n"
            for factor, contrib in prediction.get('top_bearish_factors', [])[:3]:
                context += f"  ❌ {factor}: {contrib:.3f}\n"

            return context.strip()

        except ImportError:
            logger.debug("Alpha model not available")
            return ""
        except Exception as e:
            logger.debug(f"Error getting alpha model prediction: {e}")
            return ""

    def _save_alpha_prediction(self, ticker: str, prediction: dict):
        """
        Save alpha prediction to database for reliability tracking.

        Predictions are saved once per day per ticker. Outcomes are updated
        later when actual returns become available.
        """
        try:
            from datetime import date
            import json
            import psycopg2
            from dotenv import load_dotenv
            import os

            load_dotenv()

            # Use psycopg2 directly for reliable inserts
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'alpha_platform'),
                user=os.getenv('POSTGRES_USER', 'alpha'),
                password=os.getenv('POSTGRES_PASSWORD', '')
            )

            with conn.cursor() as cur:
                # Check if prediction already exists for today
                cur.execute(
                    "SELECT id FROM alpha_predictions WHERE ticker = %s AND prediction_date = %s",
                    (ticker, date.today())
                )
                if cur.fetchone():
                    conn.close()
                    # Still update outcomes for old predictions
                    self._update_prediction_outcomes()
                    return  # Already exists

                # Insert new prediction
                # UNIT CONSISTENCY: Store as percentage (multiply by 100) to match multi_factor_alpha.py
                expected_5d = prediction.get('expected_return_5d')
                expected_10d = prediction.get('expected_return_10d')
                expected_20d = prediction.get('expected_return_20d')

                # Only multiply if values are decimals (< 1 means decimal format)
                if expected_5d is not None and abs(expected_5d) < 1:
                    expected_5d = expected_5d * 100
                if expected_10d is not None and abs(expected_10d) < 1:
                    expected_10d = expected_10d * 100
                if expected_20d is not None and abs(expected_20d) < 1:
                    expected_20d = expected_20d * 100

                cur.execute("""
                            INSERT INTO alpha_predictions (ticker, prediction_date, expected_return_5d,
                                                           expected_return_10d,
                                                           expected_return_20d, prob_positive_5d, signal, conviction,
                                                           regime, factor_contributions, top_bullish_factors,
                                                           top_bearish_factors,
                                                           created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                    NOW()) ON CONFLICT (ticker, prediction_date) DO NOTHING
                            """, (
                    ticker,
                    date.today(),
                    expected_5d,
                    expected_10d,
                    expected_20d,
                    prediction.get('prob_positive_5d'),
                    prediction.get('signal'),
                    prediction.get('conviction'),
                    prediction.get('regime'),
                    json.dumps(prediction.get('factor_contributions', {})),
                    json.dumps(prediction.get('top_bullish_factors', [])),
                    json.dumps(prediction.get('top_bearish_factors', [])),
                ))
                conn.commit()

            conn.close()
            logger.debug(f"Saved alpha prediction for {ticker}")

            # Update outcomes for old predictions
            self._update_prediction_outcomes()

        except Exception as e:
            logger.debug(f"Could not save alpha prediction: {e}")

    def _update_prediction_outcomes(self):
        """
        Update actual outcomes for predictions that are old enough.

        This runs automatically when predictions are generated to keep
        the reliability metrics up to date.
        """
        try:
            import psycopg2
            from dotenv import load_dotenv
            import os

            load_dotenv()

            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'alpha_platform'),
                user=os.getenv('POSTGRES_USER', 'alpha'),
                password=os.getenv('POSTGRES_PASSWORD', '')
            )

            with conn.cursor() as cur:
                # Update 5-day outcomes for predictions 5+ days old
                cur.execute("""
                            UPDATE alpha_predictions ap
                            SET actual_return_5d    = sub.actual_return,
                                prediction_error_5d = sub.actual_return - ap.expected_return_5d,
                                updated_at          = NOW() FROM (
                        SELECT 
                            ap2.id,
                            (m2.close - m1.close) / m1.close as actual_return
                        FROM alpha_predictions ap2
                        JOIN market_data m1 ON m1.ticker = ap2.ticker AND m1.date = ap2.prediction_date
                        JOIN market_data m2 ON m2.ticker = ap2.ticker AND m2.date = ap2.prediction_date + INTERVAL '5 days'
                        WHERE ap2.actual_return_5d IS NULL
                        AND ap2.prediction_date <= CURRENT_DATE - INTERVAL '5 days'
                    ) sub
                            WHERE ap.id = sub.id
                            """)

                # Update 10-day outcomes for predictions 10+ days old
                cur.execute("""
                            UPDATE alpha_predictions ap
                            SET actual_return_10d = sub.actual_return,
                                updated_at        = NOW() FROM (
                        SELECT 
                            ap2.id,
                            (m2.close - m1.close) / m1.close as actual_return
                        FROM alpha_predictions ap2
                        JOIN market_data m1 ON m1.ticker = ap2.ticker AND m1.date = ap2.prediction_date
                        JOIN market_data m2 ON m2.ticker = ap2.ticker AND m2.date = ap2.prediction_date + INTERVAL '10 days'
                        WHERE ap2.actual_return_10d IS NULL
                        AND ap2.prediction_date <= CURRENT_DATE - INTERVAL '10 days'
                    ) sub
                            WHERE ap.id = sub.id
                            """)

                conn.commit()

            conn.close()
            logger.debug("Updated prediction outcomes")

        except Exception as e:
            logger.debug(f"Could not update prediction outcomes: {e}")

    def _get_unified_signal_context(self, ticker: str) -> str:
        """
        Get UnifiedSignal context from the signal engine.

        Returns comprehensive signal analysis including all components.
        """
        try:
            from src.core import generate_signal, UnifiedSignal

            signal = generate_signal(ticker)

            if not signal:
                return ""

            # Format comprehensive signal context
            context = f"""
📊 UNIFIED SIGNAL ANALYSIS: {ticker} - {signal.company_name}
{'=' * 60}
Sector: {signal.sector}
Current Price: ${signal.current_price:.2f}

SIGNALS:
  Today Signal: {signal.today_signal.value} ({signal.today_score}%)
  Long-term: {signal.longterm_score}/100
  Risk Level: {signal.risk_level.value} (Score: {signal.risk_score})
  Reason: {signal.signal_reason}

COMPONENT SCORES:
  Technical:   {signal.technical_score} ({signal.technical_signal}) - {signal.technical_reason}
  Fundamental: {signal.fundamental_score} ({signal.fundamental_signal}) - {signal.fundamental_reason}
  Sentiment:   {signal.sentiment_score} ({signal.sentiment_signal}) - {signal.sentiment_reason}
  Options:     {signal.options_score} ({signal.options_signal}) - {signal.options_reason}

PRICE TARGETS:
  Target Price: ${signal.target_price:.2f} ({signal.upside_pct:+.1f}% upside)
  Stop Loss: ${signal.stop_loss:.2f} ({signal.downside_pct:.1f}% downside)
  Risk/Reward: {signal.risk_reward:.2f}
  52W Range: ${signal.week_52_low:.2f} - ${signal.week_52_high:.2f}
  From High: {signal.pct_from_high:.1f}% | From Low: +{signal.pct_from_low:.1f}%

COMMITTEE:
  Verdict: {signal.committee_verdict} (Confidence: {signal.committee_confidence:.0%})
  Agreement: {signal.committee_agreement:.0%}
"""
            # Add earnings if available
            if signal.earnings_date:
                days = signal.days_to_earnings or 0
                if days > 0:
                    context += f"\nEARNINGS: In {days} days ({signal.earnings_date})"
                elif days == 0:
                    context += f"\n⚠️ EARNINGS TODAY"
                else:
                    context += f"\nLast Earnings: {signal.earnings_date} - {signal.last_earnings_result}"

                if signal.ies_score:
                    context += f"\n  IES Score: {signal.ies_score}/100"
                if signal.ecs_category:
                    context += f"\n  ECS Category: {signal.ecs_category}"

            # Add catalyst
            if signal.next_catalyst:
                context += f"\nNEXT CATALYST: {signal.next_catalyst}"
                if signal.days_to_catalyst:
                    context += f" ({signal.days_to_catalyst} days)"

            # Add flags
            if signal.flags:
                context += f"\nFLAGS: {' '.join(signal.flags)}"

            # Add portfolio context if held
            if signal.in_portfolio:
                context += f"""
PORTFOLIO POSITION:
  Weight: {signal.portfolio_weight:.1%}
  Target: {signal.target_weight:.1%}
  P&L: {signal.portfolio_pnl_pct:+.1f}%
  Days Held: {signal.days_held}
"""

            # Add trade idea if available
            if signal.has_trade_idea:
                context += f"""
TRADE IDEA:
  Action: {signal.trade_action}
  Conviction: {signal.trade_conviction}
  Timeframe: {signal.trade_timeframe}
  Thesis: {signal.trade_thesis}
"""

            return context.strip()

        except ImportError:
            logger.debug("Signal engine not available")
            return ""
        except Exception as e:
            logger.debug(f"Error getting unified signal: {e}")
            return ""

    def _get_live_earnings_context(self, ticker: str) -> str:
        """
        Fetch and analyze live earnings data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Formatted earnings context string
        """
        try:
            from src.analytics.earnings_analyzer import (
                EarningsTranscriptAnalyzer,
                analyze_ticker_earnings
            )

            logger.info(f"Analyzing earnings for {ticker}...")

            # Get live earnings analysis
            result = analyze_ticker_earnings(ticker)

            if not result.eps_actual and not result.eps_surprise_pct:
                return ""

            # Format for AI context
            context = f"""
📊 LIVE EARNINGS ANALYSIS: {ticker}
{'=' * 50}

💰 LATEST EARNINGS RESULTS:
   EPS Actual: ${result.eps_actual:.2f}
   EPS Estimate: ${result.eps_estimate:.2f if result.eps_estimate else 'N/A'}
   EPS Surprise: {result.eps_surprise_pct:.2f}% {'✅ BEAT' if result.eps_surprise_pct and result.eps_surprise_pct > 0 else '❌ MISS' if result.eps_surprise_pct and result.eps_surprise_pct < 0 else ''}

📈 AI ASSESSMENT:
   Overall Sentiment: {result.overall_sentiment}
   Sentiment Score: {result.sentiment_score}/100
   Guidance: {result.guidance_direction}

📊 SIGNAL IMPACT:
   Score Adjustment: {result.score_adjustment:+d} points
   Reason: {result.adjustment_reason}
"""

            # Add guidance summary if available
            if result.guidance_summary:
                context += f"\n📝 GUIDANCE SUMMARY:\n   {result.guidance_summary}\n"

            # Add key highlights if available
            if result.key_highlights:
                context += "\n✅ KEY HIGHLIGHTS:\n"
                for h in result.key_highlights[:3]:
                    context += f"   • {h}\n"

            # Add concerns if available
            if result.concerns:
                context += "\n⚠️ CONCERNS:\n"
                for c in result.concerns[:3]:
                    context += f"   • {c}\n"

            return context

        except Exception as e:
            logger.error(f"Error getting live earnings for {ticker}: {e}")
            return ""

    def _get_macro_regime_context(self) -> str:
        """
        Get current macro regime analysis for AI context.

        Returns:
            Formatted macro regime context string
        """
        try:
            from src.analytics.macro_regime import get_regime_for_ai

            context = get_regime_for_ai()
            return context

        except ImportError:
            logger.debug("Macro regime module not available")
            return ""
        except Exception as e:
            logger.error(f"Error getting macro regime context: {e}")
            return ""

    def _get_bond_context(self, ticker: str = None) -> str:
        """
        Get bond market analysis for AI context.

        Args:
            ticker: Optional specific bond ticker (TLT, ZROZ, etc.)

        Returns:
            Formatted bond context string
        """
        try:
            from src.analytics.bond_signals_analytics import get_bond_context_for_ai

            # Check if ticker is a bond instrument
            bond_tickers = ['TLT', 'ZROZ', 'EDV', 'TMF', 'TBT', 'SHY', 'IEF']
            if ticker and ticker.upper() in bond_tickers:
                context = get_bond_context_for_ai(ticker.upper())
            else:
                context = get_bond_context_for_ai()

            return context

        except ImportError:
            logger.debug("Bond signals module not available")
            return ""
        except Exception as e:
            logger.error(f"Error getting bond context: {e}")
            return ""

    def _extract_ticker(self, message: str) -> Optional[str]:
        """Extract ticker from message if mentioned."""
        import re

        # Common patterns
        patterns = [
            r'\b([A-Z]{1,5})\b',  # Uppercase 1-5 letters
            r'about\s+([A-Z]{1,5})',
            r'hold\s+([A-Z]{1,5})',
            r'own\s+([A-Z]{1,5})',
        ]

        # Common words to exclude (expanded list)
        exclude = {
            # Articles, pronouns, prepositions
            'I', 'A', 'AN', 'THE', 'TO', 'IN', 'ON', 'AT', 'FOR', 'AND', 'OR', 'BUT',
            'IS', 'IT', 'MY', 'ME', 'DO', 'IF', 'SO', 'NO', 'YES', 'NOT', 'CAN',
            'WHY', 'HOW', 'WHAT', 'WHEN', 'BUY', 'SELL', 'HOLD', 'GET',
            # Added common words
            'YOU', 'YOUR', 'WE', 'US', 'HE', 'SHE', 'THEY', 'THEM', 'THIS', 'THAT',
            'OF', 'BE', 'ARE', 'WAS', 'WERE', 'BEEN', 'HAS', 'HAVE', 'HAD',
            'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST',
            'KNOW', 'THINK', 'ABOUT', 'WITH', 'FROM', 'INTO', 'OVER',
            # Finance terms that aren't tickers
            'OPTIONS', 'OPTION', 'FLOW', 'STOCK', 'STOCKS', 'SHARE', 'SHARES',
            'PRICE', 'CALL', 'PUT', 'TRADE', 'MARKET', 'DATA', 'NEWS',
            'TELL', 'SHOW', 'GIVE', 'HELP', 'PLEASE', 'THANKS', 'NEED',
            'ANY', 'ALL', 'SOME', 'MORE', 'LESS', 'MUCH', 'MANY',
            # Short squeeze related
            'SHORT', 'SQUEEZE', 'COVER', 'FLOAT', 'DAYS', 'HIGH', 'LOW',
            'POTENTIAL', 'RISK', 'CHECK', 'ANALYSIS', 'ANALYZE', 'LOOK',
        }

        for pattern in patterns:
            matches = re.findall(pattern, message.upper())
            for match in matches:
                if match not in exclude and len(match) >= 2:
                    return match

        return None

        # chat.py — REPLACE the entire `chat()` method with this version.

    def chat(self, message: str, ticker: Optional[str] = None) -> str:
        """
        Send a message and get AI response with unified portfolio context.
        """
        if not self.available:
            return "AI Chat is not available. Check Qwen server connection."

        context_parts = []

        # Detect what type of question this is
        message_lower = (message or "").lower()

        search_keywords = ['weather', 'price now', 'current price', 'latest news',
                           'what is happening', 'breaking', 'live', 'right now', 'currently']
        needs_search = any(kw in message_lower for kw in search_keywords)

        general_keywords = ['who is', 'what is', 'how to', 'explain', 'tell me about',
                            'search for', 'find', 'look up']
        is_general = any(kw in message_lower for kw in general_keywords)

        stock_keywords = ['signal', 'score', 'backtest', 'strategy', 'buy or sell',
                          'recommendation', 'analysis', 'should i buy', 'committee']
        is_stock_question = any(kw in message_lower for kw in stock_keywords)

        # Detect earnings-related questions
        earnings_keywords = ['earnings', 'earning', 'eps', 'revenue', 'quarterly', 'quarter',
                             'q1', 'q2', 'q3', 'q4', 'beat', 'miss', 'guidance', 'outlook',
                             'results', 'report', 'fiscal', 'profit', 'income', 'surprise']
        needs_earnings = any(kw in message_lower for kw in earnings_keywords)

        # Detect macro/regime questions
        macro_keywords = ['macro', 'regime', 'risk on', 'risk off', 'risk-on', 'risk-off',
                          'market environment', 'market condition', 'vix', 'fear', 'sentiment',
                          'growth vs defensive', 'sector rotation', 'market mood']
        needs_macro = any(kw in message_lower for kw in macro_keywords)

        # Detect bond/rates questions
        bond_keywords = ['bond', 'bonds', 'treasury', 'treasuries', 'yield', 'yields',
                         'tlt', 'zroz', 'edv', 'tmf', 'rates', 'duration', 'fed funds',
                         'interest rate', '10 year', '30 year', '10y', '30y', 'curve']
        needs_bonds = any(kw in message_lower for kw in bond_keywords)

        needs_portfolio = self._needs_portfolio_context(message)
        needs_economic = self._needs_economic_context(message)

        # Extract ticker from message if not provided
        if not ticker:
            ticker = self._extract_ticker(message)

        logger.info(
            f"Query analysis: portfolio={needs_portfolio}, economic={needs_economic}, stock={is_stock_question}, earnings={needs_earnings}, ticker={ticker}"
        )

        # Add ECONOMIC/MARKET context if needed (includes AI news analysis)
        if needs_economic:
            logger.info("Loading economic/market context with AI analysis...")
            economic_context = self._get_economic_context(run_ai_analysis=True)
            if economic_context:
                context_parts.append(economic_context)

        # Web search (NOW allowed for stock/portfolio/economic when useful)
        if self._should_web_search(
                message,
                needs_search=needs_search,
                is_general=is_general,
                is_stock_question=is_stock_question,
                needs_portfolio=needs_portfolio,
                needs_economic=needs_economic,
                ticker=ticker,
        ):
            q = (message or "").strip()
            ql = q.lower()
            if ql.startswith("/web"):
                q = q[4:].strip()
            if ql.startswith("web:"):
                q = q[4:].strip()

            # If ticker exists, prepend it (unless already present)
            if ticker and ticker.upper() not in q.upper():
                q = f"{ticker} {q}"

            search_results = self._search_web(q)
            if search_results:
                context_parts.append(search_results)

        # Add platform context
        platform_context = self._get_platform_context()
        if platform_context:
            context_parts.append(platform_context)

        # Add ticker-specific context
        if ticker:
            ticker_context = self._get_ticker_context(ticker.upper())
            if ticker_context:
                context_parts.append(ticker_context)

        # Add LIVE EARNINGS analysis if earnings question detected
        if needs_earnings and ticker:
            logger.info(f"Fetching live earnings data for {ticker}...")
            earnings_context = self._get_live_earnings_context(ticker.upper())
            if earnings_context:
                context_parts.append(earnings_context)

        # Add MACRO REGIME context if macro question detected
        if needs_macro or is_stock_question:
            logger.info("Loading macro regime context...")
            macro_context = self._get_macro_regime_context()
            if macro_context:
                context_parts.append(macro_context)

        # Add BOND context if bond/rates question detected
        if needs_bonds:
            logger.info("Loading bond market context...")
            bond_context = self._get_bond_context(ticker)
            if bond_context:
                context_parts.append(bond_context)

        # Add MACRO/GEOPOLITICAL context if needed
        needs_macro_geo = self._needs_macro_context(message)
        if needs_macro_geo or is_stock_question or needs_portfolio:
            logger.info("Loading macro/geopolitical context...")
            macro_geo_context = self._get_macro_context(ticker)
            if macro_geo_context:
                context_parts.append(macro_geo_context)

        # Add UNIFIED portfolio context
        if needs_portfolio or ticker:
            logger.info("Loading unified portfolio context...")
            portfolio_context = self._get_unified_portfolio_context(ticker)
            if portfolio_context:
                context_parts.append(portfolio_context)

        # Prioritize context based on question relevance
        full_context = "\n\n".join(context_parts)
        prioritized_context = self._prioritize_context(full_context, message)

        # Build messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]

        if prioritized_context:
            context_msg = """CURRENT DATA (USE ONLY THIS DATA - DO NOT INVENT ANY VALUES):

""" + prioritized_context + """

═══════════════════════════════════════════════════════════════════
ANALYSIS INSTRUCTIONS:
═══════════════════════════════════════════════════════════════════

Before answering, think through these steps:

1. **IDENTIFY KEY DATA**: What are the most important metrics for this question?
2. **BULLISH FACTORS**: What supports a positive outlook?
3. **BEARISH FACTORS**: What supports a negative outlook?
4. **MACRO FACTORS**: Is the macro environment a tailwind or headwind for this stock?
5. **SIGNAL CONFLICTS**: Do Platform and Alpha Model agree? If not, why might each be right?
6. **RISK ASSESSMENT**: What could go wrong? What's the downside?
7. **RECOMMENDATION**: Synthesize into actionable advice with position sizing.

WEB SEARCH POLICY:
- If web results are present in the context, use them for CURRENT EVENTS / NEWS / EARNINGS.
- If web results are absent, do not pretend you checked the internet.

REMINDERS:
- For portfolio questions → Use UNIFIED PORTFOLIO data ("ACTUAL" = real holdings)
- For economic questions → Use MARKET CONTEXT and AI ANALYSIS
- For macro questions → Use MACRO/GEOPOLITICAL CONTEXT (oil, conflict, trade war factors)
- When signals conflict → State it explicitly and explain both sides
- Always include position sizing based on conviction
- Reference specific numbers from the data
- Be honest about uncertainty - don't oversimplify"""
            messages.append({"role": "system", "content": context_msg})

        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=getattr(self.config, 'top_p', 0.9),
                frequency_penalty=getattr(self.config, 'repeat_penalty', 1.1) - 1.0
            )

            assistant_message = response.choices[0].message.content

            if '</think>' in assistant_message:
                assistant_message = assistant_message.split('</think>')[-1].strip()

            assistant_message = _sanitize_bond_output(assistant_message)

            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": assistant_message})

            if len(self.history) > 20:
                self.history = self.history[-20:]

            return assistant_message

        except Exception as e:
            error_str = str(e).lower()
            if 'context' in error_str or 'exceed' in error_str or 'n_ctx' in error_str:
                logger.warning(f"Context full - clearing history. Error: {e}")
                self.history = self.history[-4:]
                return "⚠️ **Context was full.** I've cleared older messages. Please ask again."
            else:
                logger.error(f"Chat error: {e}")
                return f"Error communicating with AI: {str(e)}"

        # chat.py — REPLACE the entire `chat_stream()` method with this version.

    def chat_stream(self, message: str, ticker: Optional[str] = None):
        """
        Send a message and stream the AI response with unified portfolio context.
        """
        if not self.available:
            yield "AI Chat is not available. Check Qwen server connection."
            return

        context_parts = []
        message_lower = (message or "").lower()

        search_keywords = ['weather', 'price now', 'current price', 'latest news',
                           'what is happening', 'breaking', 'live', 'right now', 'currently']
        needs_search = any(kw in message_lower for kw in search_keywords)

        general_keywords = ['who is', 'what is', 'how to', 'explain', 'tell me about',
                            'search for', 'find', 'look up']
        is_general = any(kw in message_lower for kw in general_keywords)

        stock_keywords = ['signal', 'score', 'backtest', 'strategy', 'buy or sell',
                          'recommendation', 'analysis', 'should i buy', 'committee']
        is_stock_question = any(kw in message_lower for kw in stock_keywords)

        needs_portfolio = self._needs_portfolio_context(message)
        needs_economic = self._needs_economic_context(message)

        if not ticker:
            ticker = self._extract_ticker(message)

        logger.info(
            f"Stream query: portfolio={needs_portfolio}, economic={needs_economic}, stock={is_stock_question}, ticker={ticker}"
        )

        # ECONOMIC/MARKET context (includes AI news analysis)
        if needs_economic:
            logger.info("Loading economic/market context for stream...")
            economic_context = self._get_economic_context(run_ai_analysis=True)
            if economic_context:
                context_parts.append(economic_context)

        # Web search (NOW allowed for stock/portfolio/economic when useful)
        if self._should_web_search(
                message,
                needs_search=needs_search,
                is_general=is_general,
                is_stock_question=is_stock_question,
                needs_portfolio=needs_portfolio,
                needs_economic=needs_economic,
                ticker=ticker,
        ):
            q = (message or "").strip()
            ql = q.lower()
            if ql.startswith("/web"):
                q = q[4:].strip()
            if ql.startswith("web:"):
                q = q[4:].strip()

            if ticker and ticker.upper() not in q.upper():
                q = f"{ticker} {q}"

            search_results = self._search_web(q)
            if search_results:
                context_parts.append(search_results)

        # Platform context
        platform_context = self._get_platform_context()
        if platform_context:
            context_parts.append(platform_context)

        # Ticker context
        if ticker:
            ticker_context = self._get_ticker_context(ticker.upper())
            if ticker_context:
                context_parts.append(ticker_context)

        # MACRO/GEOPOLITICAL context
        needs_macro_geo = self._needs_macro_context(message)
        if needs_macro_geo or is_stock_question or ticker:
            logger.info("Loading macro/geopolitical context for stream...")
            macro_geo_context = self._get_macro_context(ticker)
            if macro_geo_context:
                context_parts.append(macro_geo_context)

        # UNIFIED portfolio context
        if needs_portfolio or ticker:
            portfolio_context = self._get_unified_portfolio_context(ticker)
            if portfolio_context:
                context_parts.append(portfolio_context)

        # Prioritize context based on question relevance
        full_context = "\n\n".join(context_parts)
        prioritized_context = self._prioritize_context(full_context, message)

        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]

        if prioritized_context:
            context_msg = """CURRENT DATA (USE ONLY THIS DATA - DO NOT INVENT ANY VALUES):

""" + prioritized_context + """

═══════════════════════════════════════════════════════════════════
ANALYSIS INSTRUCTIONS:
═══════════════════════════════════════════════════════════════════

Think through step-by-step:
1. KEY DATA: What metrics matter most?
2. BULLISH vs BEARISH: What factors support each side?
3. CONFLICTS: Do signals agree? Address disagreements explicitly.
4. RISK: What could go wrong?
5. RECOMMENDATION: Action + position size + entry/exit levels.

WEB SEARCH POLICY:
- If web results are present in the context, use them for CURRENT EVENTS / NEWS / EARNINGS.
- If web results are absent, do not pretend you checked the internet.

Use the structured format (📊 Signal Summary → 📈 Key Metrics → ⚖️ Bull/Bear → 🎯 Recommendation → ⚠️ Risks)"""
            messages.append({"role": "system", "content": context_msg})

        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=getattr(self.config, 'top_p', 0.9),
                frequency_penalty=getattr(self.config, 'repeat_penalty', 1.1) - 1.0,
                stream=True
            )

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            clean_response = full_response
            if '</think>' in clean_response:
                clean_response = clean_response.split('</think>')[-1].strip()

            sanitized_response = _sanitize_bond_output(clean_response)

            if sanitized_response != clean_response:
                yield "\n\n---\n*[Numbers auto-corrected for display accuracy]*"

            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": sanitized_response})

            if len(self.history) > 12:
                self.history = self.history[-12:]

        except Exception as e:
            error_str = str(e).lower()
            if 'context' in error_str or 'exceed' in error_str or 'n_ctx' in error_str:
                logger.warning("Context full - clearing history")
                self.history = self.history[-4:]
                yield "⚠️ **Context was full.** I've cleared older messages. Please ask again."
            else:
                logger.error(f"Chat stream error: {e}")
                yield f"Error: {str(e)}"

    # Convenience methods
    def ask_about_ticker(self, ticker: str, question: str = None) -> str:
        """Ask about a specific ticker."""
        if not question:
            question = f"Give me your analysis of {ticker}. Include actual holdings, target weight, and recommendation."
        return self.chat(question, ticker=ticker)

    def get_recommendation(self, ticker: str) -> str:
        """Get recommendation for a ticker."""
        question = f"""Based on all available data for {ticker}:
1. Do I currently own it? (actual shares and value)
2. What's the target weight?
3. Your recommendation (BUY, HOLD, or SELL)
4. Key reasons (3 points)
5. Main risks (2 points)"""
        return self.chat(question, ticker=ticker)

    def get_portfolio_summary(self) -> str:
        """Get portfolio summary."""
        return self.chat("Give me a summary of my portfolio. Include total value, holdings count, and any issues.")

    def should_rebalance(self) -> str:
        """Check if rebalancing is needed."""
        return self.chat("Should I rebalance my portfolio? Check the drift data and give a clear recommendation.")

    def explain_holding(self, ticker: str) -> str:
        """Explain why a stock is held."""
        return self.chat(f"Why do I hold {ticker}? Show actual position, target weight, and selection reasoning.",
                         ticker=ticker)

    def compare_tickers(self, tickers: List[str]) -> str:
        """Compare multiple tickers."""
        question = f"Compare these stocks: {', '.join(tickers)}. Show actual holdings, target weights, and which is the best opportunity."
        return self.chat(question)